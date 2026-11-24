"""Evaluate labeled JSONL datasets; errors count as incorrect.

Exact match is task-specific correctness, not a general hallucination metric.
"""
import argparse
import hashlib
import json
import statistics
import time

from inference.demo_sentiment import DemoSentimentModel, EXAMPLES
from inference.ollama_adapter import OllamaAdapter, OllamaBackendError


def read_dataset(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("dataset is empty")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not isinstance(row.get("expected"), str):
            raise ValueError("Each JSONL row needs text and expected strings")
        text = row["text"].strip()
        if not text or not row["expected"].strip() or text.casefold() in seen:
            raise ValueError("Empty or duplicate examples")
        seen.add(text.casefold())
    return rows, hashlib.sha256(raw).hexdigest()


def evaluate(predict, rows, output_key="label", recoverable_errors=(ValueError, OllamaBackendError)):
    if not rows:
        raise ValueError("At least one labeled example required")
    correct = 0
    errors = 0
    timings = []
    outcomes = []
    for row in rows:
        start = time.perf_counter()
        try:
            output = predict(row["text"])[output_key]
            matched = str(output).strip().casefold() == row["expected"].strip().casefold()
        except recoverable_errors:
            matched = False
            errors += 1
        correct += int(matched)
        timings.append((time.perf_counter() - start) * 1000)
        outcomes.append(matched)
    return {
        "sample_count": len(rows), "correct": correct, "errors": errors,
        "accuracy": correct / len(rows),
        "median_latency_ms": statistics.median(timings),
        "outcomes": outcomes, "metric": "casefolded_exact_match",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="demo-sentiment-v1")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows, digest = read_dataset(args.dataset)
    if args.model == "demo-sentiment-v1":
        training = {text.casefold() for text, _ in EXAMPLES}
        if any(row["text"].strip().casefold() in training for row in rows):
            raise ValueError("Dataset contains exact training examples")
        result = evaluate(DemoSentimentModel().predict, rows)
    else:
        result = evaluate(OllamaAdapter(args.model, args.ollama_url).generate, rows, "text")
    result.update({"model": args.model, "dataset_sha256": digest,
                   "note": "Exact match on supplied examples, not general factuality."})
    content = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
