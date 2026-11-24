"""Run a bounded CPU/GPU server benchmark against an authorized loopback provider.

No packages required. Does not start a server or change configuration.
Compare concurrency levels using the same prompt corpus and output cap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import statistics
import time

from inference.openai_compatible import CompatibleAdapter

PROMPTS = [
    "Explain what a computer is in two short sentences.",
    "Describe the purpose of a database in two short sentences.",
    "Explain why testing software is useful in two short sentences.",
    "Explain the difference between RAM and storage in two short sentences.",
    "Explain what a network is in two short sentences.",
    "Explain what an algorithm is in two short sentences.",
    "Explain what a GPU does in two short sentences.",
    "Explain why backups are useful in two short sentences.",
]


def benchmark(adapter, concurrency, repeats=3):
    records = []
    for _ in range(repeats):
        def invoke(prompt):
            start = time.perf_counter()
            result = adapter.generate(prompt)
            tokens = result.get("generated_tokens")
            if type(tokens) is not int or tokens < 1 or not result["text"].strip():
                raise RuntimeError("Provider must return actual generated token counts and text")
            return {"seconds": time.perf_counter() - start, "tokens": tokens}

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            responses = list(executor.map(invoke, PROMPTS))
        elapsed = time.perf_counter() - start
        tokens = sum(r["tokens"] for r in responses)
        records.append({"seconds": elapsed, "generated_tokens": tokens,
                        "aggregate_tokens_per_second": tokens / elapsed,
                        "mean_request_seconds": statistics.mean(r["seconds"] for r in responses),
                        "requests": responses})
    return {"concurrency": concurrency, "repeats": records,
            "mean_aggregate_tokens_per_second": statistics.mean(
                r["aggregate_tokens_per_second"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="qwen")
    args = parser.parse_args()
    adapter = CompatibleAdapter(args.model, args.origin, os.getenv("APICALAI_PROVIDER_KEY"), max_tokens=64)
    adapter.generate("Say hello briefly.")  # warmup excluded
    print(json.dumps({"model": args.model, "requests_per_repeat": len(PROMPTS),
                      "max_output_tokens": 64,
                      "results": [benchmark(adapter, c) for c in (1, 2, 4)],
                      "scope": "HTTP client concurrency; provider configuration must be recorded separately"},
                     indent=2))


if __name__ == "__main__":
    main()
