"""Cross-domain sentiment cascade study using UCI data, not an LLM benchmark."""
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from urllib.request import urlopen

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import __version__ as sklearn_version

URL = "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
FILES = {"amazon": "amazon_cells_labelled.txt", "imdb": "imdb_labelled.txt", "yelp": "yelp_labelled.txt"}


def parse_labeled_text(raw):
    """Treat a terminal tab and binary label as a record delimiter."""
    import re
    rows = []
    pending = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        match = re.fullmatch(r"(.*)\t([01])\s*", line.rstrip("\r"))
        if match:
            text = " ".join(pending + [match.group(1)]).strip()
            if not text:
                raise ValueError("Empty labeled row")
            rows.append((text, int(match.group(2))))
            pending.clear()
        else:
            pending.append(line.strip())
    if pending:
        raise ValueError("Unterminated record in dataset")
    return rows


def load_archive(raw):
    domains = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for domain, filename in FILES.items():
            matches = [n for n in archive.namelist() if n.endswith("/" + filename) or n == filename]
            if len(matches) != 1 or archive.getinfo(matches[0]).file_size > 1000000:
                raise ValueError("Unexpected dataset archive structure")
            rows = parse_labeled_text(archive.read(matches[0]).decode("utf-8", errors="strict"))
            if len(rows) != 1000:
                raise ValueError("Unexpected source example count")
            domains[domain] = rows
    return domains


def unique_rows(rows, forbidden=None):
    seen = set(forbidden or ())
    clean = []
    for text, label in rows:
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            clean.append((text, label))
    return clean


def measure(model, texts):
    probabilities = []
    times = []
    for text in texts:
        start = time.perf_counter_ns()
        probabilities.append(model.predict_proba([text])[0])
        times.append((time.perf_counter_ns() - start) / 1e6)
    return np.array(probabilities), np.array(times)


def cascade(cheap, strong, cheap_ms, strong_ms, threshold):
    escalated = cheap.max(axis=1) < threshold
    predicted = np.where(escalated, strong.argmax(axis=1), cheap.argmax(axis=1))
    return predicted, cheap_ms + escalated * strong_ms, escalated


def choose_threshold(cheap, strong, labels, tolerance=.02):
    """Select on calibration only, with no test-set information."""
    target = max(0, float(np.mean(strong.argmax(axis=1) == labels)) - tolerance)
    viable = []
    for threshold in np.linspace(.5, 1, 26):
        pred, _, escalated = cascade(cheap, strong, np.zeros(len(labels)), np.zeros(len(labels)), threshold)
        accuracy = float(np.mean(pred == labels))
        if accuracy >= target:
            viable.append((float(escalated.mean()), float(threshold), accuracy))
    if not viable:
        raise RuntimeError("No calibration policy satisfies target")
    return min(viable), target


def summarize(pred, labels, times, escalated, seed):
    correct = pred == labels
    rng = np.random.default_rng(seed)
    bootstrap = [float(correct[rng.integers(0, len(correct), len(correct))].mean()) for _ in range(1000)]
    return {"accuracy": float(correct.mean()),
            "accuracy_bootstrap_95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "mean_component_ms": float(times.mean()), "p95_component_ms": float(np.percentile(times, 95)),
            "escalation_fraction": float(escalated.mean()), "sample_count": len(labels)}


def study(domains, seed=42):
    results = {}
    for held in sorted(domains):
        test = unique_rows(domains[held])
        train_all = unique_rows(
            [row for domain, rows in domains.items() if domain != held for row in rows],
            forbidden={text.casefold() for text, _ in test},
        )
        train, cal = train_test_split(train_all, test_size=.25, random_state=seed,
                                      stratify=[label for _, label in train_all])
        xtrain, ytrain = zip(*train)
        xcal, ycal = zip(*cal)
        xtest, ytest = zip(*test)
        ycal, ytest = np.array(ycal), np.array(ytest)
        cheap = make_pipeline(TfidfVectorizer(max_features=300, ngram_range=(1, 1)),
                              LogisticRegression(random_state=seed, max_iter=500))
        strong = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=12000),
                               LogisticRegression(random_state=seed, max_iter=500))
        cheap.fit(xtrain, ytrain)
        strong.fit(xtrain, ytrain)
        cp, _ = measure(cheap, xcal)
        sp, _ = measure(strong, xcal)
        (cal_escalation, threshold, cal_accuracy), target = choose_threshold(cp, sp, ycal)
        # Warm-up before measuring per-example component times. These times are
        # offline estimates of cascade compute, not observed end-to-end service.
        cheap.predict_proba([xtest[0]])
        strong.predict_proba([xtest[0]])
        c, c_ms = measure(cheap, xtest)
        s, s_ms = measure(strong, xtest)
        p, latency, escalation = cascade(c, s, c_ms, s_ms, threshold)
        rng = np.random.default_rng(seed)
        random_mask = rng.random(len(ytest)) < cal_escalation
        random_pred = np.where(random_mask, s.argmax(axis=1), c.argmax(axis=1))
        results[held] = {
            "train_count": len(train), "calibration_count": len(cal), "test_count": len(test),
            "threshold": threshold, "calibration_target_accuracy": target, "calibration_accuracy": cal_accuracy,
            "calibration_escalation": cal_escalation,
            "policies": {
                "cheap_only": summarize(c.argmax(axis=1), ytest, c_ms, np.zeros(len(ytest)), seed),
                "strong_only": summarize(s.argmax(axis=1), ytest, s_ms, np.ones(len(ytest)), seed),
                "random_escalation": summarize(random_pred, ytest, c_ms + random_mask * s_ms, random_mask, seed),
                "confidence_cascade": summarize(p, ytest, latency, escalation, seed),
            },
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Optional previously downloaded official ZIP")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.archive:
        with open(args.archive, "rb") as handle:
            raw = handle.read(2000000)
    else:
        with urlopen(URL, timeout=60) as response:
            raw = response.read(2000000)
    result = {"dataset": "Kotzias 2015, UCI Sentiment Labelled Sentences, DOI 10.24432/C57604, CC BY 4.0",
              "archive_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed,
              "platform": platform.platform(), "numpy": np.__version__, "sklearn": sklearn_version,
              "domains": study(load_archive(raw), args.seed),
              "limitations": "Offline component timings; no LLM/GPU results. Bootstrap assumes independent examples. Models differ in features; the strong model is not guaranteed stronger."}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({domain: data["policies"] for domain, data in result["domains"].items()}, indent=2))


if __name__ == "__main__":
    main()
