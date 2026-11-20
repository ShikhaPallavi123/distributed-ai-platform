"""Checks for the reproducible baseline benchmark."""
import unittest

from benchmarks.sentiment_baseline import benchmark


class TestSentimentBenchmark(unittest.TestCase):
    def test_benchmark_reports_measured_metrics(self):
        result = benchmark(iterations=8)
        self.assertEqual(result["iterations"], 8)
        self.assertEqual(sum(result["prediction_counts"].values()), 8)
        self.assertGreaterEqual(result["latency_ms"]["p99"], result["latency_ms"]["p50"])
        self.assertGreaterEqual(result["training_ms"], 0)

    def test_invalid_iterations(self):
        with self.assertRaises(ValueError):
            benchmark(iterations=0)


if __name__ == "__main__":
    unittest.main()
