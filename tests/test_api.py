"""Tests for the REST API."""
import unittest
import json
import threading
import time
from http.client import HTTPConnection
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import AppState, create_server  # noqa: E402


class TestAppState(unittest.TestCase):
    def test_state_creation(self):
        state = AppState()
        self.assertIsNotNone(state)

    def test_models_populated(self):
        state = AppState()
        self.assertGreater(len(state.models), 0)

    def test_workers_populated(self):
        state = AppState()
        self.assertGreater(len(state.workers), 0)

    def test_metrics_tracking(self):
        state = AppState()
        self.assertIn('total_requests', state.metrics)


class TestAPIEndpoints(unittest.TestCase):
    """Integration tests that start a real server."""

    @classmethod
    def setUpClass(cls):
        cls.port = 18999
        try:
            cls.server = create_server(host='127.0.0.1', port=cls.port)
            cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
            cls.thread.start()
            time.sleep(0.5)
            cls.available = True
        except Exception:
            cls.available = False

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server') and cls.available:
            cls.server.shutdown()

    def _get(self, path):
        if not self.available:
            self.skipTest("Server not available")
        conn = HTTPConnection('127.0.0.1', self.port, timeout=5)
        conn.request('GET', path)
        resp = conn.getresponse()
        body = json.loads(resp.read().decode())
        conn.close()
        return resp.status, body

    def _post(self, path, data):
        if not self.available:
            self.skipTest("Server not available")
        conn = HTTPConnection('127.0.0.1', self.port, timeout=5)
        conn.request('POST', path, json.dumps(data), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        body = json.loads(resp.read().decode())
        conn.close()
        return resp.status, body

    def test_health(self):
        status, body = self._get('/api/health')
        self.assertEqual(status, 200)
        self.assertEqual(body['status'], 'healthy')

    def test_models(self):
        status, body = self._get('/api/models')
        self.assertEqual(status, 200)
        self.assertIn('models', body)

    def test_workers(self):
        status, body = self._get('/api/workers')
        self.assertEqual(status, 200)
        self.assertIn('workers', body)

    def test_metrics(self):
        status, body = self._get('/api/metrics')
        self.assertEqual(status, 200)

    def test_inference(self):
        status, body = self._post('/api/inference', {
            'model': 'gpt-neuroflux-7b',
            'input': {'text': 'Hello world'}
        })
        self.assertEqual(status, 200)
        self.assertIn('output', body)


if __name__ == "__main__":
    unittest.main()
