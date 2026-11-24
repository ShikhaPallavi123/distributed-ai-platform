"""Tests for benchmark methodology and the actual optional baseline."""
import importlib.util
import unittest

from inferbench.dimensions import measure_latency, measure_throughput
from inferbench.engines import get_engine
from inferbench.report import generate_report


class TestInferBench(unittest.TestCase):
    def test_sequential_measurement_rejects_concurrency_claim(self):
        with self.assertRaises(ValueError):
            measure_latency(lambda x: x, ["x"], (2,), 2)

    def test_throughput_scope(self):
        result = measure_throughput(lambda x: x, ["x"], (1,), 0.001)
        self.assertEqual(result["method"], "sequential_grouped_calls")
        self.assertGreater(result["results"]["batch_1"]["total_requests"], 0)

    def test_old_results_rejected(self):
        with self.assertRaises(ValueError):
            generate_report({"inferbench_version": "0.1.0"})

    def test_adapter_scope(self):
        adapter = get_engine("apicalai")
        self.assertFalse(adapter["agent_readiness"])
        with self.assertRaises(ValueError):
            get_engine("flask_sklearn")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional research dependency")
    def test_real_sklearn_predictions(self):
        adapter = get_engine("sklearn")
        model = adapter["init_fn"]()
        self.assertEqual(model.classifier.__class__.__module__.split(".")[0], "sklearn")
        self.assertEqual(model.predict("wonderful excellent")["label"], "positive")
        self.assertEqual(model.predict("awful terrible")["label"], "negative")
        with self.assertRaises(ValueError):
            model.predict("zzzz")
