"""InferBench dimension measurement modules.

Each function measures one of the 8 benchmark dimensions and returns
a structured result dict. All measurements are self-contained and
reproducible.
"""

import gc
import os
import platform
import statistics
import subprocess
import sys
import time
import resource


def measure_throughput(predict_fn, payloads, batch_sizes=(1, 8, 32, 128),
                       duration_seconds=10):
    """Sequential call throughput, grouped by loop size; not true batching."""
    results = {}
    for batch_size in batch_sizes:
        count = 0
        start = time.perf_counter()
        deadline = start + duration_seconds
        while time.perf_counter() < deadline:
            for i in range(batch_size):
                predict_fn(payloads[i % len(payloads)])
                count += 1
        elapsed = time.perf_counter() - start
        results[f"batch_{batch_size}"] = {
            "requests_per_second": round(count / elapsed, 2),
            "total_requests": count,
            "duration_seconds": round(elapsed, 3),
        }
    return {"dimension": "throughput", "method": "sequential_grouped_calls", "results": results}


def measure_latency(predict_fn, payloads, concurrency_levels=(1,),
                    iterations=200):
    """Dimension 2: Latency — p50, p95, p99 under varying concurrency.

    Note: True concurrency requires threads/processes. This measures
    sequential latency at different queue depths as a baseline.
    For real concurrency use a separate HTTP benchmark.
    """
    results = {}
    for concurrency in concurrency_levels:
        if concurrency != 1:
            raise ValueError("This sequential measurement supports concurrency=1 only")
        latencies = []
        for i in range(iterations):
            payload = payloads[i % len(payloads)]
            start = time.perf_counter()
            predict_fn(payload)
            latencies.append((time.perf_counter() - start) * 1000)

        ordered = sorted(latencies)

        def percentile(p):
            idx = (len(ordered) - 1) * p / 100
            lo = int(idx)
            hi = min(lo + 1, len(ordered) - 1)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

        results[f"concurrency_{concurrency}"] = {
            "p50_ms": round(percentile(50), 4),
            "p95_ms": round(percentile(95), 4),
            "p99_ms": round(percentile(99), 4),
            "mean_ms": round(statistics.mean(latencies), 4),
            "min_ms": round(min(latencies), 4),
            "max_ms": round(max(latencies), 4),
            "stdev_ms": round(statistics.stdev(latencies), 4)
            if len(latencies) > 1 else 0,
            "iterations": iterations,
        }
    return {"dimension": "latency", "results": results}


def measure_cold_start(engine_init_fn, predict_fn_from_engine, payload,
                       trials=5):
    """Dimension 3: Model initialization to first prediction in a warm process.

    engine_init_fn: callable that returns an initialized engine/model
    predict_fn_from_engine: callable(engine) -> predict_fn
    """
    times = []
    for _ in range(trials):
        gc.collect()
        start = time.perf_counter()
        engine = engine_init_fn()
        predict_fn = predict_fn_from_engine(engine)
        predict_fn(payload)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        del engine
        gc.collect()

    return {
        "dimension": "cold_start",
        "method": "in_process_model_initialization",
        "results": {
            "mean_ms": round(statistics.mean(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "stdev_ms": round(statistics.stdev(times), 2) if len(times) > 1
            else 0,
            "trials": trials,
        },
    }


def measure_memory(predict_fn, payloads, num_requests=100):
    """Dimension 4: Memory footprint — RSS at idle, under load."""
    gc.collect()

    def get_rss_mb():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024), 2)

    rss_before = get_rss_mb()

    for i in range(num_requests):
        predict_fn(payloads[i % len(payloads)])

    gc.collect()
    rss_after = get_rss_mb()

    return {
        "dimension": "memory_footprint",
        "method": "process_peak_rss_not_current_rss",
        "results": {
            "rss_before_mb": rss_before,
            "rss_after_load_mb": rss_after,
            "rss_delta_mb": round(rss_after - rss_before, 2),
            "requests_processed": num_requests,
        },
    }


def measure_dependencies(package_name=None, requirements_file=None):
    """Dimension 5: Dependency weight — install size, dep count."""
    results = {}

    if requirements_file and os.path.exists(requirements_file):
        with open(requirements_file) as f:
            lines = [
                ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
        results["direct_dependencies"] = len(lines)
        results["requirements"] = lines

    # Count installed packages in current environment
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            stderr=subprocess.DEVNULL, text=True
        )
        import json
        packages = json.loads(output)
        results["total_installed_packages"] = len(packages)
    except Exception:
        results["total_installed_packages"] = "unknown"

    # Measure import time for the engine
    if package_name:
        try:
            start = time.perf_counter()
            __import__(package_name)
            results["import_time_ms"] = round(
                (time.perf_counter() - start) * 1000, 2
            )
        except ImportError:
            results["import_time_ms"] = "not_importable"

    return {"dimension": "dependency_weight", "results": results}


def measure_portability():
    """Dimension 6: Portability — hardware/OS information."""
    return {
        "dimension": "portability",
        "results": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "cpu_count": os.cpu_count(),
        },
    }


def measure_observability(engine_name, has_request_tracing=False,
                          has_layer_metrics=False,
                          has_memory_profiling=False,
                          has_scheduling_visibility=False,
                          has_consensus_visibility=False,
                          has_model_internals=False,
                          has_error_traces=False,
                          has_latency_breakdown=False):
    """Dimension 7: Observability depth — can you trace end to end?"""
    features = {
        "request_tracing": has_request_tracing,
        "layer_metrics": has_layer_metrics,
        "memory_profiling": has_memory_profiling,
        "scheduling_visibility": has_scheduling_visibility,
        "consensus_visibility": has_consensus_visibility,
        "model_internals": has_model_internals,
        "error_traces": has_error_traces,
        "latency_breakdown": has_latency_breakdown,
    }
    score = sum(1 for v in features.values() if v)
    return {
        "dimension": "observability",
        "method": "declared_adapter_capabilities_not_measured",
        "results": {
            "engine": engine_name,
            "features": features,
            "score": f"{score}/{len(features)}",
            "depth": (
                "full" if score >= 7
                else "high" if score >= 5
                else "medium" if score >= 3
                else "low"
            ),
        },
    }


def measure_agent_readiness(engine_name, has_native_agents=False,
                            has_multi_step=False, has_tool_use=False,
                            has_planning=False, has_multi_agent=False,
                            has_memory=False):
    """Dimension 8: Agent readiness — native multi-step reasoning."""
    features = {
        "native_agents": has_native_agents,
        "multi_step_reasoning": has_multi_step,
        "tool_use": has_tool_use,
        "planning": has_planning,
        "multi_agent_orchestration": has_multi_agent,
        "agent_memory": has_memory,
    }
    score = sum(1 for v in features.values() if v)
    return {
        "dimension": "agent_readiness",
        "method": "declared_adapter_capabilities_not_measured",
        "results": {
            "engine": engine_name,
            "features": features,
            "score": f"{score}/{len(features)}",
            "readiness": (
                "native" if score >= 5
                else "partial" if score >= 3
                else "external_only" if score >= 1
                else "none"
            ),
        },
    }
