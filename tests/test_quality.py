import json
import os
import tempfile
import unittest

from inferbench.quality import evaluate, read_dataset


class TestQuality(unittest.TestCase):
    def test_errors_count_as_wrong(self):
        def predict(text):
            if text == "unknown":
                raise ValueError("unsupported")
            return {"label": "positive"}

        report = evaluate(predict, [
            {"text": "good", "expected": "positive"},
            {"text": "bad", "expected": "negative"},
            {"text": "unknown", "expected": "negative"},
        ])
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["errors"], 1)
        self.assertAlmostEqual(report["accuracy"], 1 / 3)

    def test_dataset_identity_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test.jsonl")
            row = json.dumps({"text": "example", "expected": "positive"})
            with open(path, "w") as handle:
                handle.write(row + "\n")
            rows, digest = read_dataset(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(digest), 64)
            with open(path, "a") as handle:
                handle.write(row + "\n")
            with self.assertRaises(ValueError):
                read_dataset(path)
