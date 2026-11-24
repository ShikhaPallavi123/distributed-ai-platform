import unittest

from inference.adaptive_router import Candidate, NoEligibleModel, route
from inference.model_pool import load_pool


class TestAdaptiveRouter(unittest.TestCase):
    def test_quality_constraint_and_latency(self):
        candidates = [
            Candidate("fast", .70, 1000, 1, lambda x: x),
            Candidate("accurate", .95, 1000, 10, lambda x: x),
        ]
        self.assertEqual(route(candidates, "query", .6)["selected_model"], "fast")
        self.assertEqual(route(candidates, "query", .8)["selected_model"], "accurate")
        with self.assertRaises(NoEligibleModel):
            route(candidates, "query", .8, 2)

    def test_uncertainty_and_fallback(self):
        def unavailable(text):
            raise RuntimeError("offline")

        candidates = [
            Candidate("insufficient", 1, 1, 0, lambda x: x),
            Candidate("offline", .99, 1000, 1, unavailable),
            Candidate("working", .99, 1000, 2, lambda x: x),
        ]
        result = route(candidates, "test", .8, recoverable_errors=(RuntimeError,))
        self.assertEqual(result["selected_model"], "working")
        self.assertEqual(result["attempts"][0]["status"], "unavailable")
        self.assertEqual(result["output"], "test")

    def test_empty_pool_and_invalid_constraint(self):
        self.assertEqual(load_pool(""), ({}, None))
        with self.assertRaises(NoEligibleModel):
            route([], "text")
        with self.assertRaises(ValueError):
            route([], "text", 2)
