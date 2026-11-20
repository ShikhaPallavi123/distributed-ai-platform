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
from unittest import mock  # noqa: E402
import api.app as api_module  # noqa: E402


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
        self.assertEqual(status, 404)
        self.assertIn('not found', body['error'])

    def test_real_sentiment_inference(self):
        status, body = self._post('/api/inference', {
            'model': 'demo-sentiment-v1',
            'input': {'text': 'excellent wonderful great'}
        })
        self.assertEqual(status, 200)
        self.assertEqual(body['output']['label'], 'positive')
        self.assertEqual(body['output']['model_type'], 'trained_numpy_logistic_regression')

    def test_sentiment_invalid_input(self):
        status, body = self._post('/api/inference', {
            'model': 'demo-sentiment-v1', 'input': {'text': ''}
        })
        self.assertEqual(status, 400)

    def test_only_real_model_registered(self):
        status, body = self._get('/api/models')
        self.assertEqual(status, 200)
        self.assertTrue(any(m['id'] == 'demo-sentiment-v1' for m in body['models']))
        self.assertFalse(any(m['name'] == 'gpt-neuroflux-7b' for m in body['models']))

    def test_ollama_model_api_with_mock_backend(self):
        class FakeOllama:
            def generate(self, prompt):
                return {'text': 'A model answer to ' + prompt, 'model_type': 'ollama_pretrained_llm'}

        api_module.state.models['ollama-local'] = api_module.ModelInfo(
            'ollama-local', '1.0', 'Ollama', [1], [1], 0,
        )
        api_module.state.workers['ollama-backend'] = api_module.WorkerInfo(
            'ollama-backend', 'localhost', 11434,
        )
        try:
            with mock.patch.object(api_module, 'ollama_client', FakeOllama()):
                status, body = self._post('/api/inference', {
                    'model': 'ollama-local',
                    'input': {'text': 'What is inference?'},
                })
            self.assertEqual(status, 200)
            self.assertEqual(body['output']['text'], 'A model answer to What is inference?')
        finally:
            del api_module.state.models['ollama-local']
            del api_module.state.workers['ollama-backend']

    def test_batch_executes_real_sentiment_predictions(self):
        start = api_module.state.metrics['total_requests']
        status, body = self._post('/api/batch', {
            'requests': [
                {'model': 'demo-sentiment-v1', 'input': {'text': 'wonderful great'}},
                {'model': 'demo-sentiment-v1', 'input': {'text': 'awful terrible'}},
            ],
        })
        self.assertEqual(status, 200)
        self.assertEqual(body['batch_size'], 2)
        self.assertEqual(body['execution_mode'], 'sequential_cpu')
        self.assertEqual(body['results'][0]['output']['label'], 'positive')
        self.assertEqual(body['results'][1]['output']['label'], 'negative')
        self.assertEqual(api_module.state.metrics['total_requests'], start + 2)

    def test_batch_rejects_invalid_inputs(self):
        status, body = self._post('/api/batch', {'requests': []})
        self.assertEqual(status, 400)
        self.assertIn('1 to 32', body['error'])
        status, body = self._post('/api/batch', {'requests': [
            {'model': 'demo-sentiment-v1', 'input': {'text': ''}},
        ]})
        self.assertEqual(status, 400)

    def test_batch_rejects_missing_model(self):
        status, body = self._post('/api/batch', {'requests': [
            {'model': 'not-a-model', 'input': {'text': 'hi'}},
        ]})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
