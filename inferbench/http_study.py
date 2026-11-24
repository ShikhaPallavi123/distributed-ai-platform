"""Real concurrent HTTP clients against an already-running inference API.

The workload is a tiny CPU sentiment model; measurements include HTTP overhead.
"""
import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import time
from urllib.request import Request, urlopen

from inference.demo_sentiment import DemoSentimentModel


CASES = ("wonderful great", "awful terrible", "good reliable", "bad useless")


def percentile(values, percent):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def run_study(url, concurrency=(1, 4, 8), requests=60, key=None):
    if not concurrency or any(not 1 <= count <= 64 for count in concurrency):
        raise ValueError("concurrency must be 1–64")
    if not 1 <= requests <= 100000:
        raise ValueError("requests must be 1–100000")
    model = DemoSentimentModel()
    expected = {text: model.predict(text)["label"] for text in CASES}
    endpoint = url.rstrip("/") + "/api/inference"

    def execute(index):
        text = CASES[index % len(CASES)]
        payload = json.dumps({"model": "demo-sentiment-v1", "input": {"text": text}}).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        start = time.perf_counter()
        try:
            with urlopen(Request(endpoint, data=payload, headers=headers), timeout=30) as response:
                body = json.load(response)
            valid = body["output"]["label"] == expected[text]
            return {"latency_ms": (time.perf_counter() - start) * 1000, "ok": valid}
        except (OSError, ValueError, KeyError, TypeError):
            return {"latency_ms": (time.perf_counter() - start) * 1000, "ok": False}

    if not all(execute(i)["ok"] for i in range(len(CASES))):
        raise RuntimeError("Warmup failed; check server and authentication")

    report = {
        "scope": "toy_model_closed_loop_http_clients",
        "client_platform": platform.platform(),
        "python": platform.python_version(),
        "server_url": url,
        "note": "Output consistency against local model, not independent test-set accuracy.",
        "runs": [],
    }
    for workers in concurrency:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(execute, range(requests)))
        elapsed = time.perf_counter() - start
        latencies = [item["latency_ms"] for item in results if item["ok"]]
        run = {
            "concurrent_clients": workers,
            "attempted": requests,
            "successful": len(latencies),
            "failed": requests - len(latencies),
            "successful_requests_per_second": len(latencies) / elapsed,
            "elapsed_seconds": elapsed,
            "successful_latency_ms": latencies,
        }
        if latencies:
            run.update({"median_ms": statistics.median(latencies),
                        "p95_ms": percentile(latencies, 95),
                        "p99_ms": percentile(latencies, 99)})
        report["runs"].append(run)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_study(args.url, args.concurrency, args.requests, os.getenv("APICALAI_API_KEY"))
    content = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)
    return int(any(run["failed"] for run in result["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
