"""Unit tests for the optional Ollama integration (no external model required)."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from inference.ollama_adapter import OllamaAdapter, OllamaBackendError


class TestOllamaAdapter(unittest.TestCase):
    def test_request_and_response(self):
        payload = {"response": "Hello from an LLM", "eval_count": 7, "prompt_eval_count": 5}
        with patch("inference.ollama_adapter.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            result = OllamaAdapter("example-model").generate("Hello")
        self.assertEqual(result["text"], "Hello from an LLM")
        self.assertEqual(result["generated_tokens"], 7)
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "example-model")
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["prompt"], "Hello")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            OllamaAdapter("example-model").generate(" ")

    def test_very_long_prompt_rejected(self):
        with self.assertRaises(ValueError):
            OllamaAdapter("example-model").generate("x" * 8001)

    def test_unreachable_server_reports_error(self):
        with patch("inference.ollama_adapter.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(OllamaBackendError):
                OllamaAdapter("example-model").generate("Hi")

    def test_invalid_json_is_an_error(self):
        with patch("inference.ollama_adapter.urlopen", return_value=io.BytesIO(b"invalid")):
            with self.assertRaises(OllamaBackendError):
                OllamaAdapter("example-model").generate("Hi")


if __name__ == "__main__":
    unittest.main()
