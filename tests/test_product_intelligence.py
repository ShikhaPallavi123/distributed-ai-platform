import io
import json
import unittest
from unittest.mock import patch

from inference.grounded import retrieve
from inference.safe_tools import calculate, run_tool_workflow
from inference.ollama_adapter import OllamaAdapter, OllamaBackendError


class TestIntelligence(unittest.TestCase):
    def test_citations_and_abstention(self):
        text = "  Apples are red. Bananas are yellow."
        result = retrieve("What color are bananas?", [{"id": "d", "text": text}])
        citation = result["citations"][0]
        self.assertEqual(text[citation["start"]:citation["end"]], "Bananas are yellow.")
        self.assertEqual(retrieve("quantum", [{"id": "d", "text": text}])["status"], "abstained")

    def test_bounded_tools(self):
        self.assertEqual(calculate("(2 + 3) * 4"), 20)
        for expression in ("__import__('os')", "2**100", "1/0"):
            with self.assertRaises(ValueError):
                calculate(expression)
        self.assertEqual(run_tool_workflow([
            {"tool": "calculator", "arguments": {"expression": "1+2"}},
        ])["result"], 3)

    def test_chat_transport(self):
        model = OllamaAdapter("test")
        messages = [{"role": "user", "content": "hello"}]
        with patch("inference.ollama_adapter.urlopen", return_value=io.BytesIO(
            json.dumps({"message": {"content": "hi"}}).encode()
        )):
            self.assertEqual(model.chat(messages)["text"], "hi")
        payload = b'{"message":{"content":"hi"},"done":false}\n{"done":true}\n'
        with patch("inference.ollama_adapter.urlopen", return_value=io.BytesIO(payload)):
            self.assertEqual(list(model.chat(messages, stream=True)), [
                {"text": "hi", "done": False}, {"done": True, "generated_tokens": None},
            ])
        with self.assertRaises(ValueError):
            model.chat([{"role": "system", "content": "bad"}])
        with patch("inference.ollama_adapter.urlopen", return_value=io.BytesIO(b"")):
            with self.assertRaises(OllamaBackendError):
                list(model.chat(messages, stream=True))
