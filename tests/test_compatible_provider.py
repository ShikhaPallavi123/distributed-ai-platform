import io
import json
import unittest
from unittest.mock import patch
from inference.openai_compatible import CompatibleAdapter
from inference.model_pool import load_pool
from inference.ollama_adapter import OllamaBackendError


class TestCompatibleProvider(unittest.TestCase):
    def test_response_and_cap(self):
        adapter = CompatibleAdapter("qwen", "http://127.0.0.1:18080", "test-key", 64)
        data = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        with patch("inference.openai_compatible.build_opener") as opener:
            call = opener.return_value.open
            call.return_value = io.BytesIO(json.dumps(data).encode())
            self.assertEqual(adapter.generate("hello")["generated_tokens"], 2)
            req = call.call_args.args[0]
            self.assertEqual(json.loads(req.data)["max_tokens"], 64)
            self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
        with self.assertRaises(ValueError):
            CompatibleAdapter("qwen", "http://example.com")

    def test_invalid_response(self):
        adapter = CompatibleAdapter("qwen", "http://127.0.0.1:18080")
        with patch("inference.openai_compatible.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(b"{}")
            with self.assertRaises(OllamaBackendError):
                adapter.generate("hello")

    def test_config(self):
        row = {"name": "llm-qwen", "model": "qwen", "url": "http://127.0.0.1:18080",
               "provider": "llama.cpp", "calibration_id": "example",
               "accuracy": .8, "sample_count": 100, "median_latency_ms": 50}
        pool, protocol = load_pool(json.dumps([row]))
        self.assertEqual(protocol, "example")
        self.assertIn("llm-qwen", pool)
