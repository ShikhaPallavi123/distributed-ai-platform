import io
import json
import unittest
from unittest.mock import patch

from apicalai_sdk import Client
from api.app import allow_auth_attempt, _auth_attempts


class TestSDKSecurity(unittest.TestCase):
    def test_sdk_origin_and_request(self):
        for origin in ("http://example.com", "https://user:pw@example.com", "https://example.com/path"):
            with self.assertRaises(ValueError):
                Client(origin)
        client = Client("https://example.com", "test-key")
        with patch.object(client._opener, "open", return_value=io.BytesIO(b'{"output":1}')) as mocked:
            self.assertEqual(client.infer("model", "hello"), {"output": 1})
            req = mocked.call_args.args[0]
            self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(json.loads(req.data)["input"]["text"], "hello")

    def test_auth_rate_limit(self):
        _auth_attempts.clear()
        try:
            self.assertTrue(all(allow_auth_attempt("peer", now=i) for i in range(5)))
            self.assertFalse(allow_auth_attempt("peer", now=5))
            self.assertTrue(allow_auth_attempt("other", now=5))
            self.assertTrue(allow_auth_attempt("peer", now=65))
        finally:
            _auth_attempts.clear()
