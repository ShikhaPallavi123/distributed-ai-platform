"""Controlled routing experiment on distinct calibration and test examples.

Uses two real CPU sentiment classifiers. This is a toy-methodology study,
not an LLM, distributed-GPU, or publication-quality experiment.
"""
import argparse
import json
import platform
import statistics
import time

from inference.adaptive_router import Candidate, NoEligibleModel, route
from inference.demo_sentiment import EXAMPLES, DemoSentimentModel
from inferbench.engines import get_engine
from inferbench.quality import evaluate, read_dataset


def validate_split(calibration, test):
    training = {text.strip().casefold() for text, _ in EXAMPLES}
    cal = {row["text"].strip().casefold() for row in calibration}
    held = {row["text"].strip().casefold() for row in test}
    if training & (cal | held) or cal & held:
        raise ValueError("Exact overlap between training, calibration or test examples")


def experiment(calibration, test, predictors, thresholds=(0.3, 0.5, 0.7)):
    validate_split(calibration, test)
    if not predictors or not calibration or not test:
        raise ValueError("Nonempty models and both dataset splits required")
    calibration_reports = {
        name: evaluate(predict, calibration) for name, predict in predictors.items()
    }
    candidates = [
        Candidate(name, report["accuracy"], report["sample_count"],
                  report["median_latency_ms"], predictors[name])
        for name, report in calibration_reports.items()
    ]
    fastest = min(candidates, key=lambda item: (item.latency_ms, item.name))
    highest = max(candidates, key=lambda item: (item.quality, -item.latency_ms))
    policies = {
        "always_fastest": lambda text: (fastest.name, fastest.predict(text)),
        "always_highest_calibration_accuracy": lambda text: (highest.name, highest.predict(text)),
    }

    def constrained(text, threshold):
        result = route(candidates, text, threshold, recoverable_errors=(ValueError,))
        return result["selected_model"], result["output"]

    for threshold in thresholds:
        policies[f"quality_lower_bound_{threshold}"] = (
            lambda text, t=threshold: constrained(text, t)
        )

    reports = {}
    for policy, predict in policies.items():
        correct = errors = 0
        times = []
        selections = {}
        for row in test:
            start = time.perf_counter()
            try:
                name, output = predict(row["text"])
                correct += int(output["label"].casefold() == row["expected"].casefold())
                selections[name] = selections.get(name, 0) + 1
            except (ValueError, NoEligibleModel):
                errors += 1
            times.append((time.perf_counter() - start) * 1000)
        reports[policy] = {
            "accuracy_all_examples": correct / len(test),
            "coverage": (len(test) - errors) / len(test),
            "correct": correct, "errors_or_abstentions": errors,
            "sample_count": len(test), "median_elapsed_ms": statistics.median(times),
            "selected_models": selections, "latencies_ms": times,
        }

    return {"scope": "exploratory_cpu_sentiment_routing", "platform": platform.platform(),
            "calibration": calibration_reports, "test_policies": reports,
            "note": "Calibration and test are disjoint; tiny hand-labeled data are not population evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cal, cal_sha = read_dataset(args.calibration)
    held, held_sha = read_dataset(args.test)
    baseline = get_engine("sklearn")
    result = experiment(cal, held, {
        "apicalai": DemoSentimentModel().predict,
        "sklearn": baseline["predict_fn"](baseline["init_fn"]()),
    })
    result["calibration_sha256"] = cal_sha
    result["test_sha256"] = held_sha
    content = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
