import unittest
from unittest.mock import Mock

from inference.grounded_agent import run_agent
from inference.semantic_retrieval import semantic_retrieve, verify_citations
from inference.endpoint_recovery import RecoveringEndpoint
from inference.ollama_adapter import OllamaBackendError


class TestGroundedAgent(unittest.TestCase):
    def test_model_selects_calculator(self):
        result = run_agent(lambda _: {"text": '{"tool":"calculator","expression":"3*7"}'}, "Multiply 3 by 7")
        self.assertEqual(result["result"]["value"], 21)

    def test_untrusted_actions_fail_closed(self):
        model = Mock(return_value={"text": '{"tool":"shell","command":"whoami"}'})
        result = run_agent(model, "Use a tool", max_steps=2)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(model.call_count, 2)

    def test_quote_index_cannot_be_forged(self):
        docs = [{"id": "d", "text": "The apple is red."}]
        result = run_agent(lambda _: {"text": '{"tool":"evidence","indices":[0]}'}, "apple", docs)
        self.assertTrue(verify_citations(docs, result["result"]["citations"])["all_spans_verified"])
        result = run_agent(lambda _: {"text": '{"tool":"evidence","indices":[99]}'}, "apple", docs)
        self.assertEqual(result["status"], "abstained")

    def test_semantic_index_and_verification(self):
        docs = [{"id": "a", "text": "Apples are red."}, {"id": "b", "text": "Grass is green."}]
        result = semantic_retrieve(lambda x: [[1, 0], [1, 0], [0, 1]], "fruit", docs)
        self.assertEqual(result["citations"][0]["document_id"], "a")
        self.assertEqual(len(result["citations"]), 1)
        self.assertTrue(verify_citations(docs, result["citations"])["all_spans_verified"])
        result["citations"][0]["quote"] = "forged"
        self.assertFalse(verify_citations(docs, result["citations"])["all_spans_verified"])
        with self.assertRaises(OllamaBackendError):
            semantic_retrieve(lambda x: [[0, 0]] * len(x), "fruit", docs)

    def test_circuit_breaker_recovers(self):
        now = [0]
        model = Mock(side_effect=[OllamaBackendError("down"), OllamaBackendError("down"), {"text": "ok"}])
        endpoint = RecoveringEndpoint(model, failures=2, cooldown=10, clock=lambda: now[0])
        for _ in range(3):
            with self.assertRaises(OllamaBackendError):
                endpoint("query")
        self.assertEqual(model.call_count, 2)
        now[0] = 11
        self.assertEqual(endpoint("query"), {"text": "ok"})
        self.assertEqual(endpoint.failures, 0)
