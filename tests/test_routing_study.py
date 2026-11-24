import unittest

from inferbench.routing_study import experiment, validate_split
from inference.demo_sentiment import DemoSentimentModel


class TestRoutingStudy(unittest.TestCase):
    def test_overlap_rejected(self):
        rows = [{"text": "wonderful great", "expected": "positive"}]
        with self.assertRaises(ValueError):
            validate_split(rows, rows)

    def test_real_classifier_and_policy_denominators(self):
        calibration = [{"text": "wonderful great", "expected": "positive"},
                       {"text": "awful terrible", "expected": "negative"}]
        test = [{"text": "excellent amazing", "expected": "positive"},
                {"text": "horrible broken", "expected": "negative"}]
        model = DemoSentimentModel()
        report = experiment(calibration, test, {"numpy": model.predict}, thresholds=(0, 1))
        policies = report["test_policies"]
        self.assertEqual(policies["always_fastest"]["correct"], 2)
        self.assertEqual(policies["quality_lower_bound_0"]["coverage"], 1)
        self.assertEqual(policies["quality_lower_bound_1"]["coverage"], 0)
        self.assertEqual(policies["quality_lower_bound_1"]["accuracy_all_examples"], 0)
