"""Measure a REAL configured Ollama language model, never synthetic latency.

Example:
  python -m benchmarks.ollama_baseline --model gemma3:1b --repeats 3

Requires Ollama and the model to be installed separately. Results measure
single-request end-to-end CPU/GPU + network latency on the executing machine.
"""
import argparse
import json
import platform
import statistics
import time

from inference.ollama_adapter import OllamaAdapter

DEFAULT_PROMPTS = [
    "Explain the difference between a process and a thread in two sentences.",
    "What are two advantages of batching inference requests?",
    "Summarize why testing an AI model matters in three short points.",
]


def _percentile(values, percentage):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measure(adapter, prompts, repeats=3, warmup=1):
    """Call the real adapter and calculate observed summary statistics."""
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("prompts must contain non-empty strings")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >=1 and warmup must be >=0")

    for _ in range(warmup):
        adapter.generate(prompts[0])

    latencies = []
    generated_tokens = []
    for _ in range(repeats):
        for prompt in prompts:
            start = time.perf_counter()
            result = adapter.generate(prompt)
            latencies.append((time.perf_counter() - start) * 1000)
            if isinstance(result.get("generated_tokens"), int):
                generated_tokens.append(result["generated_tokens"])

    return {
        "benchmark": "sequential_real_ollama_inference",
        "model": adapter.model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sample_count": len(latencies),
        "warmup_requests": warmup,
        "repeats": repeats,
        "prompt_count": len(prompts),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
        },
        "generated_tokens_total": sum(generated_tokens) if generated_tokens else None,
        "note": "Measured sequential requests only; answer quality is NOT evaluated.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="An Ollama model already pulled locally")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompts-file", help="Optional UTF-8 file with one prompt per line")
    args = parser.parse_args()
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    report = measure(
        OllamaAdapter(args.model, base_url=args.url),
        prompts, repeats=args.repeats, warmup=args.warmup,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
