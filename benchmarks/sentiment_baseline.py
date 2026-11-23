"""Reproducible local benchmark for ApicalAI's first real inference model.

Run: python -m benchmarks.sentiment_baseline --iterations 200
All results are measured on the machine executing this script.
"""
import argparse
import json
import platform
import statistics
import time

from inference.demo_sentiment import DemoSentimentModel

PROMPTS = [
    "excellent wonderful great",
    "awful broken disappointing",
    "good reliable helpful",
    "horrible bad useless",
]


def benchmark(iterations=200):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    train_start = time.perf_counter()
    model = DemoSentimentModel()
    train_ms = (time.perf_counter() - train_start) * 1000
    latencies = []
    labels = {"positive": 0, "negative": 0}
    for i in range(iterations):
        prompt = PROMPTS[i % len(PROMPTS)]
        start = time.perf_counter()
        output = model.predict(prompt)
        latencies.append((time.perf_counter() - start) * 1000)
        labels[output["label"]] += 1
    ordered = sorted(latencies)

    def percentile(p):
        index = (len(ordered) - 1) * p / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "benchmark": "numpy_sentiment_cpu_baseline",
        "iterations": iterations,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "training_ms": round(train_ms, 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 5),
            "p50": round(percentile(50), 5),
            "p95": round(percentile(95), 5),
            "p99": round(percentile(99), 5),
        },
        "prediction_counts": labels,
        "note": "Microbenchmark only; not a language-model or distributed inference benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.iterations), indent=2))
