"""System and inference profiler with flame-graph support.

Tracks CPU, memory, disk, and network metrics in real time, provides
per-layer latency and memory breakdowns for inference workloads, and builds
flame-graph data structures for call-stack analysis.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ======================================================================
# Statistical helpers
# ======================================================================

def _percentile(sorted_data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (0-100) from already-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@dataclass
class LatencyStats:
    """Computed latency statistics over a window of samples."""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    throughput: float = 0.0  # requests / second

    @classmethod
    def from_samples(cls, samples: list[float], window_secs: float = 1.0) -> "LatencyStats":
        if not samples:
            return cls()
        s = sorted(samples)
        return cls(
            count=len(s),
            mean=statistics.mean(s),
            std=statistics.pstdev(s),
            p50=_percentile(s, 50),
            p95=_percentile(s, 95),
            p99=_percentile(s, 99),
            min_val=s[0],
            max_val=s[-1],
            throughput=len(s) / max(window_secs, 1e-9),
        )


# ======================================================================
# Metric collector
# ======================================================================

class _RingBuffer:
    """Fixed-capacity ring buffer of (timestamp, value) pairs."""

    def __init__(self, capacity: int = 4096) -> None:
        self._buf: deque[tuple[float, float]] = deque(maxlen=capacity)

    def push(self, value: float, ts: Optional[float] = None) -> None:
        self._buf.append((ts or time.monotonic(), value))

    def values_since(self, since: float) -> list[float]:
        return [v for t, v in self._buf if t >= since]

    def all_values(self) -> list[float]:
        return [v for _, v in self._buf]

    def __len__(self) -> int:
        return len(self._buf)


# ======================================================================
# System-level profiler
# ======================================================================

class SystemProfiler:
    """Collects system-level metrics from ``/proc`` (Linux) or via stubs.

    Metrics are stored in ring buffers and can be queried for a time window.
    A background thread samples at a configurable interval.
    """

    def __init__(self, interval_ms: float = 1000.0, history: int = 4096) -> None:
        self._interval = interval_ms / 1000.0
        self._metrics: dict[str, _RingBuffer] = defaultdict(lambda: _RingBuffer(history))
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[dict[str, float]], None]] = []

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._interval * 3)

    def on_sample(self, cb: Callable[[dict[str, float]], None]) -> None:
        """Register a callback invoked on every sample."""
        self._callbacks.append(cb)

    def snapshot(self) -> dict[str, float]:
        """Return the latest value for every metric."""
        with self._lock:
            out: dict[str, float] = {}
            for name, ring in self._metrics.items():
                vals = ring.all_values()
                if vals:
                    out[name] = vals[-1]
            return out

    def query(self, metric: str, window_secs: float = 60.0) -> LatencyStats:
        since = time.monotonic() - window_secs
        with self._lock:
            vals = self._metrics[metric].values_since(since)
        return LatencyStats.from_samples(vals, window_secs)

    # -- sampling -------------------------------------------------------

    def _sample_loop(self) -> None:
        while self._running:
            sample = self._collect()
            with self._lock:
                for k, v in sample.items():
                    self._metrics[k].push(v)
            for cb in self._callbacks:
                try:
                    cb(sample)
                except Exception:
                    pass
            time.sleep(self._interval)

    def _collect(self) -> dict[str, float]:
        data: dict[str, float] = {}
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
                data["cpu_load_1m"] = float(parts[0])
                data["cpu_load_5m"] = float(parts[1])
        except OSError:
            data["cpu_load_1m"] = 0.0

        try:
            with open("/proc/meminfo") as f:
                info: dict[str, int] = {}
                for line in f:
                    tokens = line.split()
                    if len(tokens) >= 2:
                        info[tokens[0].rstrip(":")] = int(tokens[1])
                total = info.get("MemTotal", 1)
                avail = info.get("MemAvailable", total)
                data["mem_total_kb"] = float(total)
                data["mem_used_pct"] = 100.0 * (1 - avail / max(total, 1))
        except OSError:
            data["mem_used_pct"] = 0.0

        try:
            st = os.statvfs("/")
            data["disk_used_pct"] = 100.0 * (1 - st.f_bavail / max(st.f_blocks, 1))
        except OSError:
            data["disk_used_pct"] = 0.0

        return data


# ======================================================================
# Inference profiler
# ======================================================================

class InferenceProfiler:
    """Per-model, per-layer latency and memory profiling.

    Use as a context manager around each layer::

        with profiler.layer("attention"):
            ...
    """

    def __init__(self, model_name: str = "", history: int = 2048) -> None:
        self.model_name = model_name
        self._layer_latencies: dict[str, _RingBuffer] = defaultdict(lambda: _RingBuffer(history))
        self._layer_memory: dict[str, _RingBuffer] = defaultdict(lambda: _RingBuffer(history))
        self._request_latencies = _RingBuffer(history)
        self._lock = threading.Lock()

    class _LayerCtx:
        def __init__(self, profiler: "InferenceProfiler", name: str) -> None:
            self._profiler = profiler
            self._name = name
            self._start = 0.0

        def __enter__(self) -> "InferenceProfiler._LayerCtx":
            self._start = time.perf_counter()
            return self

        def __exit__(self, *exc: Any) -> None:
            elapsed = (time.perf_counter() - self._start) * 1000  # ms
            with self._profiler._lock:
                self._profiler._layer_latencies[self._name].push(elapsed)

    def layer(self, name: str) -> _LayerCtx:
        """Return a context manager that records latency for *name*."""
        return self._LayerCtx(self, name)

    def record_memory(self, layer_name: str, bytes_used: int) -> None:
        with self._lock:
            self._layer_memory[layer_name].push(float(bytes_used))

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self._request_latencies.push(latency_ms)

    def layer_stats(self, window_secs: float = 60.0) -> dict[str, LatencyStats]:
        since = time.monotonic() - window_secs
        out: dict[str, LatencyStats] = {}
        with self._lock:
            for name, ring in self._layer_latencies.items():
                vals = ring.values_since(since)
                out[name] = LatencyStats.from_samples(vals, window_secs)
        return out

    def request_stats(self, window_secs: float = 60.0) -> LatencyStats:
        since = time.monotonic() - window_secs
        with self._lock:
            vals = self._request_latencies.values_since(since)
        return LatencyStats.from_samples(vals, window_secs)


# ======================================================================
# FlameGraph data structure
# ======================================================================

@dataclass
class FlameNode:
    """A single node in a flame graph (a stack frame)."""
    name: str
    self_time: float = 0.0
    total_time: float = 0.0
    call_count: int = 0
    children: dict[str, "FlameNode"] = field(default_factory=dict)


class FlameGraph:
    """Accumulates call-stack samples into a tree suitable for flame-graph
    visualisation.

    Usage::

        fg = FlameGraph()
        fg.add_stack(["main", "forward", "attention", "softmax"], 2.5)
        fg.add_stack(["main", "forward", "attention", "matmul"], 1.3)
        print(fg.to_folded())
    """

    def __init__(self) -> None:
        self.root = FlameNode(name="root")
        self._lock = threading.Lock()

    def add_stack(self, frames: list[str], duration: float) -> None:
        """Record a sample with the given call stack and duration."""
        with self._lock:
            node = self.root
            node.total_time += duration
            node.call_count += 1
            for frame in frames:
                if frame not in node.children:
                    node.children[frame] = FlameNode(name=frame)
                child = node.children[frame]
                child.total_time += duration
                child.call_count += 1
                node = child
            # Leaf gets the self-time.
            node.self_time += duration

    def to_folded(self) -> str:
        """Return the Brendan-Gregg folded-stack format."""
        lines: list[str] = []
        self._fold(self.root, [], lines)
        return "\n".join(lines)

    def hotspots(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the *top_n* frames by total time."""
        flat: dict[str, float] = {}
        self._flatten(self.root, flat)
        ranked = sorted(flat.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    def _fold(self, node: FlameNode, stack: list[str], out: list[str]) -> None:
        if node.self_time > 0 and stack:
            out.append(f"{';'.join(stack)} {node.self_time:.3f}")
        for name, child in sorted(node.children.items()):
            self._fold(child, stack + [name], out)

    def _flatten(self, node: FlameNode, acc: dict[str, float]) -> None:
        if node.name != "root":
            acc[node.name] = acc.get(node.name, 0.0) + node.total_time
        for child in node.children.values():
            self._flatten(child, acc)
