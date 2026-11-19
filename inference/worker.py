"""Inference workers that run on individual nodes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WorkerStats:
    requests_processed: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_request_time: float = 0.0


class InferenceWorker:
    """Runs model inference on a single node.

    Handles model loading/caching, GPU memory tracking, request processing
    with timeouts, warm-up, and graceful shutdown with draining.
    """

    def __init__(
        self,
        worker_id: str,
        gpu_memory_mb: float = 8192.0,
        max_batch: int = 32,
        timeout: float = 30.0,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_memory_mb = gpu_memory_mb
        self.max_batch = max_batch
        self.timeout = timeout

        self._models: dict[str, Any] = {}
        self._model_sizes: dict[str, float] = {}  # MB
        self._used_memory_mb: float = 0.0
        self._stats = WorkerStats()
        self._active_requests = 0
        self._running = False
        self._draining = False
        self._lock = asyncio.Lock()

    @property
    def available_memory_mb(self) -> float:
        return self.gpu_memory_mb - self._used_memory_mb

    @property
    def is_busy(self) -> bool:
        return self._active_requests >= self.max_batch

    async def start(self) -> None:
        self._running = True
        self._draining = False

    async def stop(self, drain: bool = True) -> None:
        """Stop the worker, optionally waiting for in-flight requests."""
        if drain:
            self._draining = True
            while self._active_requests > 0:
                await asyncio.sleep(0.05)
        self._running = False

    def load_model(
        self, name: str, model: Any, size_mb: float = 500.0
    ) -> bool:
        """Load a model into GPU memory. Returns False if OOM."""
        if size_mb > self.available_memory_mb:
            return False
        self._models[name] = model
        self._model_sizes[name] = size_mb
        self._used_memory_mb += size_mb
        return True

    def unload_model(self, name: str) -> bool:
        if name not in self._models:
            return False
        self._used_memory_mb -= self._model_sizes.pop(name, 0)
        del self._models[name]
        return True

    def has_model(self, name: str) -> bool:
        return name in self._models

    async def warmup(self, model_name: str, n: int = 2) -> list[float]:
        """Run warm-up inferences to mitigate cold-start latency."""
        latencies: list[float] = []
        dummy = np.zeros((1, 8), dtype=np.float32)
        for _ in range(n):
            start = time.time()
            await self.process(model_name, dummy, {})
            latencies.append((time.time() - start) * 1000)
        return latencies

    async def process(
        self,
        model_name: str,
        data: Any,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Execute inference for a single request."""
        if not self._running:
            raise RuntimeError("worker is not running")
        if self._draining:
            raise RuntimeError("worker is draining")

        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"model {model_name!r} not loaded on {self.worker_id}")

        async with self._lock:
            self._active_requests += 1

        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._invoke(model, data, params), timeout=self.timeout
            )
            latency = (time.time() - start) * 1000
            self._stats.requests_processed += 1
            self._stats.total_latency_ms += latency
            self._stats.last_request_time = time.time()
            return result
        except Exception:
            self._stats.errors += 1
            raise
        finally:
            async with self._lock:
                self._active_requests -= 1

    async def _invoke(
        self, model: Any, data: Any, params: dict[str, Any]
    ) -> np.ndarray:
        if callable(model):
            result = model(data)
            if asyncio.iscoroutine(result):
                return await result
            return result
        # Treat model as a simple weight matrix: matmul
        if isinstance(model, np.ndarray) and isinstance(data, np.ndarray):
            return data @ model.T if model.ndim == 2 else data * model
        raise TypeError("unsupported model type")

    def stats(self) -> dict[str, Any]:
        s = self._stats
        avg = s.total_latency_ms / s.requests_processed if s.requests_processed else 0
        return {
            "worker_id": self.worker_id,
            "requests_processed": s.requests_processed,
            "errors": s.errors,
            "avg_latency_ms": round(avg, 2),
            "active_requests": self._active_requests,
            "used_memory_mb": self._used_memory_mb,
            "models_loaded": list(self._models.keys()),
        }


class WorkerPool:
    """Manages a pool of InferenceWorkers."""

    def __init__(self) -> None:
        self._workers: dict[str, InferenceWorker] = {}

    def add(self, worker: InferenceWorker) -> None:
        self._workers[worker.worker_id] = worker

    def remove(self, worker_id: str) -> InferenceWorker | None:
        return self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> InferenceWorker | None:
        return self._workers.get(worker_id)

    @property
    def size(self) -> int:
        return len(self._workers)

    def available_workers(self, model_name: str | None = None) -> list[InferenceWorker]:
        """Return workers that are running and not fully busy."""
        result = [
            w for w in self._workers.values()
            if w._running and not w._draining and not w.is_busy
        ]
        if model_name:
            result = [w for w in result if w.has_model(model_name)]
        return result

    def least_loaded(self, model_name: str | None = None) -> InferenceWorker | None:
        candidates = self.available_workers(model_name)
        if not candidates:
            return None
        return min(candidates, key=lambda w: w._active_requests)

    async def start_all(self) -> None:
        for w in self._workers.values():
            await w.start()

    async def stop_all(self, drain: bool = True) -> None:
        await asyncio.gather(*(w.stop(drain=drain) for w in self._workers.values()))

    def all_stats(self) -> list[dict[str, Any]]:
        return [w.stats() for w in self._workers.values()]
