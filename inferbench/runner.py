"""InferBench Runner — orchestrates all 8 dimension measurements.

Usage:
    python -m inferbench.runner                      # All engines
    python -m inferbench.runner --engine apicalai    # Single engine
    python -m inferbench.runner --output results.json # Save results
"""

import argparse
import json
from datetime import datetime, timezone

from inferbench.dimensions import (
    measure_throughput,
    measure_latency,
    measure_cold_start,
    measure_memory,
    measure_dependencies,
    measure_portability,
    measure_observability,
    measure_agent_readiness,
)
from inferbench.engines import get_engine, list_engines


# Standard test payloads for sentiment analysis
PAYLOADS = [
    "excellent wonderful great product",
    "awful broken disappointing",
    "good reliable helpful experience",
    "horrible bad useless",
    "love this amazing product",
    "poor quality slow",
    "happy with the service",
    "terrible unhappy experience",
]


def run_benchmark(engine_name, duration=5, iterations=200,
                  cold_start_trials=5, verbose=True):
    """Run all 8 InferBench dimensions on a single engine."""
    if duration <= 0 or iterations < 1 or cold_start_trials < 1:
        raise ValueError("duration, iterations and cold_start_trials must be positive")
    adapter = get_engine(engine_name)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  InferBench v0.2.0 — {adapter['name']}")
        print(f"{'='*60}\n")

    results = {
        "engine": adapter["name"],
        "engine_version": adapter["version"],
        "scope": "toy_sentiment_in_process_cpu_microbenchmark",
        "limitations": "No HTTP, true batching, GPU, distributed execution, or held-out quality evaluation.",
        "inferbench_version": "0.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dimensions": {},
    }

    # Dimension 1: Throughput
    if verbose:
        print("[1/8] Measuring throughput...")
    engine = adapter["init_fn"]()
    predict_fn = adapter["predict_fn"](engine)
    results["dimensions"]["throughput"] = measure_throughput(
        predict_fn, PAYLOADS, batch_sizes=(1, 8, 32, 128),
        duration_seconds=duration
    )

    # Dimension 2: Latency
    if verbose:
        print("[2/8] Measuring latency...")
    results["dimensions"]["latency"] = measure_latency(
        predict_fn, PAYLOADS, concurrency_levels=(1,),
        iterations=iterations
    )

    # Dimension 3: Cold Start
    if verbose:
        print("[3/8] Measuring cold start...")
    results["dimensions"]["cold_start"] = measure_cold_start(
        adapter["init_fn"],
        adapter["predict_fn"],
        PAYLOADS[0],
        trials=cold_start_trials,
    )

    # Dimension 4: Memory Footprint
    if verbose:
        print("[4/8] Measuring memory footprint...")
    results["dimensions"]["memory"] = measure_memory(
        predict_fn, PAYLOADS, num_requests=100
    )

    # Dimension 5: Dependency Weight
    if verbose:
        print("[5/8] Measuring dependency weight...")
    results["dimensions"]["dependencies"] = measure_dependencies(
        package_name=adapter.get("import_name"),
        requirements_file=adapter.get("requirements_file"),
    )

    # Dimension 6: Portability
    if verbose:
        print("[6/8] Measuring portability...")
    results["dimensions"]["portability"] = measure_portability()

    # Dimension 7: Observability Depth
    if verbose:
        print("[7/8] Assessing observability...")
    results["dimensions"]["observability"] = measure_observability(
        adapter["name"], **adapter["observability"]
    )

    # Dimension 8: Agent Readiness
    if verbose:
        print("[8/8] Assessing agent readiness...")
    results["dimensions"]["agent_readiness"] = measure_agent_readiness(
        adapter["name"], **adapter["agent_readiness"]
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Benchmark complete for {adapter['name']}")
        print(f"{'='*60}\n")

    return results


def run_all(duration=5, iterations=200, cold_start_trials=5, verbose=True):
    """Run InferBench on all registered engines."""
    all_results = {
        "inferbench_version": "0.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engines": {},
    }
    for name in list_engines():
        try:
            result = run_benchmark(
                name, duration=duration, iterations=iterations,
                cold_start_trials=cold_start_trials, verbose=verbose
            )
            all_results["engines"][name] = result
        except Exception as e:
            if verbose:
                print(f"  ERROR benchmarking {name}: {e}")
            all_results["engines"][name] = {"error": str(e)}
    return all_results


def print_summary(results):
    """Print a human-readable comparison table."""
    engines = results.get("engines", {})
    if not engines:
        print("No results to display.")
        return

    print(f"\n{'='*70}")
    print("  InferBench Summary — Multi-Dimensional Comparison")
    print(f"{'='*70}\n")

    # Header
    names = []
    for key, data in engines.items():
        if "error" not in data:
            names.append((key, data.get("engine", key)))

    if not names:
        print("All engines errored.")
        return

    header = f"{'Dimension':<25}"
    for _, display_name in names:
        header += f" | {display_name:>20}"
    print(header)
    print("-" * len(header))

    # Throughput (batch_1)
    row = f"{'Throughput (req/s)':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            rps = data["dimensions"]["throughput"]["results"]["batch_1"][
                "requests_per_second"]
            row += f" | {rps:>20.1f}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Latency p50
    row = f"{'Latency p50 (ms)':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            p50 = data["dimensions"]["latency"]["results"][
                "concurrency_1"]["p50_ms"]
            row += f" | {p50:>20.4f}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Latency p99
    row = f"{'Latency p99 (ms)':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            p99 = data["dimensions"]["latency"]["results"][
                "concurrency_1"]["p99_ms"]
            row += f" | {p99:>20.4f}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Cold start
    row = f"{'Cold start (ms)':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            cs = data["dimensions"]["cold_start"]["results"]["mean_ms"]
            row += f" | {cs:>20.1f}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Memory
    row = f"{'Memory delta (MB)':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            mem = data["dimensions"]["memory"]["results"]["rss_delta_mb"]
            row += f" | {mem:>20.1f}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Dependencies
    row = f"{'Direct dependencies':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            deps = data["dimensions"]["dependencies"]["results"].get(
                "direct_dependencies", "N/A")
            row += f" | {str(deps):>20}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Observability
    row = f"{'Observability':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            obs = data["dimensions"]["observability"]["results"]["score"]
            row += f" | {obs:>20}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    # Agent readiness
    row = f"{'Agent readiness':<25}"
    for key, _ in names:
        data = engines[key]
        try:
            ar = data["dimensions"]["agent_readiness"]["results"]["score"]
            row += f" | {ar:>20}"
        except (KeyError, TypeError):
            row += f" | {'N/A':>20}"
    print(row)

    print("-" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="InferBench — Multi-Dimensional Inference Engine Benchmark"
    )
    parser.add_argument(
        "--engine", type=str, default=None,
        help=f"Engine to benchmark. Available: {', '.join(list_engines())}. "
             f"Default: all engines."
    )
    parser.add_argument(
        "--duration", type=int, default=5,
        help="Throughput measurement duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="Latency measurement iterations (default: 200)"
    )
    parser.add_argument(
        "--cold-start-trials", type=int, default=5,
        help="Cold start measurement trials (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    if args.engine:
        results = {
            "inferbench_version": "0.2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engines": {
                args.engine: run_benchmark(
                    args.engine,
                    duration=args.duration,
                    iterations=args.iterations,
                    cold_start_trials=args.cold_start_trials,
                    verbose=not args.quiet,
                )
            },
        }
    else:
        results = run_all(
            duration=args.duration,
            iterations=args.iterations,
            cold_start_trials=args.cold_start_trials,
            verbose=not args.quiet,
        )

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
