"""Paired, measured CPU batch study on identical model weights.

Exploratory microbenchmark, not GPU batching or a publication result.
"""
import argparse
import json
import platform
import statistics
import time

import numpy as np

from inference.demo_sentiment import DemoSentimentModel


PAYLOADS = ("wonderful great", "awful terrible", "good reliable", "bad useless")


def run_study(sizes=(1, 8, 32, 128), repeats=20):
    if repeats < 2 or not sizes or any(size < 1 for size in sizes):
        raise ValueError("Positive sizes and at least two repeats are required")
    model = DemoSentimentModel()
    report = {
        "scope": "in_process_cpu_same_weights",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeats": repeats,
        "results": [],
    }
    for size in sizes:
        texts = [PAYLOADS[i % len(PAYLOADS)] for i in range(size)]
        methods = {"sequential": lambda: [model.predict(text) for text in texts],
                   "vectorized": lambda: model.predict_batch(texts)}
        if methods["sequential"]() != methods["vectorized"]():
            raise RuntimeError("Prediction mismatch")
        times = {name: [] for name in methods}
        for repeat in range(repeats):
            order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
            for name in order:
                start = time.perf_counter_ns()
                methods[name]()
                times[name].append((time.perf_counter_ns() - start) / 1e6)
        record = {"batch_size": size, "equal_predictions": True, "measurements_ms": times}
        record["median_ms"] = {name: statistics.median(values) for name, values in times.items()}
        record["median_speedup"] = record["median_ms"]["sequential"] / record["median_ms"]["vectorized"]
        report["results"].append(record)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(run_study(repeats=args.repeats), indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
