import unittest

from inferbench.batch_study import run_study
from inference.demo_sentiment import DemoSentimentModel


class TestBatchStudy(unittest.TestCase):
    def test_batch_equivalence_and_validation(self):
        model = DemoSentimentModel()
        texts = ["wonderful great", "awful terrible", "good reliable"]
        self.assertEqual(model.predict_batch(texts), [model.predict(text) for text in texts])
        for invalid in ([], ["zzzz"], [""]):
            with self.assertRaises(ValueError):
                model.predict_batch(invalid)

    def test_repeated_measurements(self):
        results = run_study(sizes=(1, 8), repeats=3)["results"]
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result["equal_predictions"])
            self.assertEqual(len(result["measurements_ms"]["vectorized"]), 3)
            self.assertGreater(result["median_speedup"], 0)
