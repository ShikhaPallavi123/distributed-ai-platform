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

    def _post(self, path, data, headers=None):
        if not self.available:
            self.skipTest("Server not available")
        conn = HTTPConnection('127.0.0.1', self.port, timeout=5)
        request_headers = {'Content-Type': 'application/json'}
        request_headers.update(headers or {})
        conn.request('POST', path, json.dumps(data), request_headers)
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
            'model': 'gpt-apicalai-7b',
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
        self.assertFalse(any(m['name'] == 'gpt-apicalai-7b' for m in body['models']))

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

    def test_admin_registration_denied_without_token(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('APICALAI_ADMIN_TOKEN', None)
            status, body = self._post('/api/models', {'name': 'unauthorized'})
        self.assertEqual(status, 403)
        self.assertIn('APICALAI_ADMIN_TOKEN', body['error'])
        self.assertNotIn('unauthorized', api_module.state.models)

    def test_admin_registration_metadata_only(self):
        with mock.patch.dict(os.environ, {'APICALAI_ADMIN_TOKEN': 'temporary-test-secret'}):
            status, body = self._post(
                '/api/models',
                {'name': 'registered-only-test'},
                headers={'Authorization': 'Bearer temporary-test-secret'},
            )
        try:
            self.assertEqual(status, 201)
            self.assertEqual(body['model']['status'], 'registered')
        finally:
            api_module.state.models.pop('registered-only-test', None)

    def test_no_synthetic_agent_success(self):
        status, body = self._post('/api/agents/run', {'goal': 'Test a tool'})
        self.assertEqual(status, 501)
        self.assertEqual(body['code'], 'AGENT_NOT_DEPLOYED')

    def test_api_key_protects_inference(self):
        request = {"model": "demo-sentiment-v1", "input": {"text": "wonderful great"}}
        with mock.patch.dict(os.environ, {"APICALAI_API_KEY": "test-client-key"}):
            status, _ = self._post("/api/inference", request)
            self.assertEqual(status, 401)
            status, body = self._post(
                "/api/inference", request,
                headers={"Authorization": "Bearer test-client-key"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["output"]["label"], "positive")
            self.assertEqual(self._get("/api/health")[0], 200)
            self.assertEqual(self._get("/api/models")[0], 401)

    def test_admin_token_distinct_from_client_key(self):
        with mock.patch.dict(os.environ, {
            "APICALAI_API_KEY": "client-only",
            "APICALAI_ADMIN_TOKEN": "admin-only",
        }):
            status, _ = self._post(
                "/api/models", {"name": "denied-test"},
                headers={"Authorization": "Bearer client-only"},
            )
            self.assertEqual(status, 403)
            status, _ = self._post(
                "/api/models", {"name": "demo-sentiment-v1"},
                headers={"Authorization": "Bearer admin-only"},
            )
            self.assertEqual(status, 409)

    def test_invalid_json_shape_and_unknown_route(self):
        self.assertEqual(self._post("/api/inference", [1, 2])[0], 400)
        self.assertEqual(self._post("/api/inference", {"model": []})[0], 400)
        self.assertEqual(self._get("/api/unknown")[0], 404)

    def test_vectorized_batch_matches_sequential(self):
        items = [
            {"model": "demo-sentiment-v1", "input": {"text": "wonderful great"}},
            {"model": "demo-sentiment-v1", "input": {"text": "awful terrible"}},
        ]
        status, baseline = self._post("/api/batch", {"requests": items})
        self.assertEqual(status, 200)
        status, batch = self._post("/api/batch", {"requests": items, "mode": "vectorized_cpu"})
        self.assertEqual(status, 200)
        self.assertEqual(batch["execution_mode"], "vectorized_cpu")
        self.assertEqual(batch["latency_scope"], "shared_batch_compute")
        self.assertEqual([r["output"] for r in baseline["results"]],
                         [r["output"] for r in batch["results"]])

    def test_routed_api_with_mocked_provider_transport(self):
        from inference.adaptive_router import Candidate
        candidate = Candidate("llm-test", .99, 1000, 1, lambda text: {"text": text})
        worker = api_module.WorkerInfo("llm-test", "test", 11434, gpu_memory_total=0)
        with mock.patch.object(api_module, "llm_pool", {"llm-test": candidate}), \
                mock.patch.object(api_module, "calibration_id", "test-split"), \
                mock.patch.dict(api_module.state.workers, {"llm-test": worker}):
            status, body = self._post("/api/route", {"text": "test", "min_quality": .8})
            self.assertEqual(status, 200)
            self.assertEqual(body["selected_model"], "llm-test")
            self.assertEqual(body["calibration_id"], "test-split")
            self.assertEqual(body["output"], {"text": "test"})
            self.assertEqual(worker.tasks_completed, 1)
            status, body = self._post("/api/route", {"text": "test", "min_quality": 1})
            self.assertEqual(status, 503)
            self.assertEqual(body["code"], "NO_ELIGIBLE_MODEL")


if __name__ == "__main__":
    unittest.main()
