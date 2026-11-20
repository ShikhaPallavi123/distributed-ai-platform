"""Test benchmark arithmetic without claiming an LLM ran in CI."""
import unittest

from benchmarks.ollama_baseline import measure


class FakeAdapter:
    model = "fake-for-tests-only"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return {"text": "test", "generated_tokens": 12}


class TestOllamaBenchmark(unittest.TestCase):
    def test_repeats_and_warmups(self):
        adapter = FakeAdapter()
        result = measure(adapter, ["Hello", "Explain inference"], repeats=2, warmup=1)
        self.assertEqual(adapter.calls, 5)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["generated_tokens_total"], 48)
        self.assertGreaterEqual(result["latency_ms"]["p99"], result["latency_ms"]["p50"])

    def test_invalid_prompt_set(self):
        with self.assertRaises(ValueError):
            measure(FakeAdapter(), [])

    def test_invalid_repeats(self):
        with self.assertRaises(ValueError):
            measure(FakeAdapter(), ["Hi"], repeats=0)


if __name__ == "__main__":
    unittest.main()
