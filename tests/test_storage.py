import os
import tempfile
import unittest

from api.app import AppState, ModelInfo, InferenceResponse
from api.storage import Storage
from inference.worker import InferenceWorker


class TestStorage(unittest.TestCase):
    def test_persistence_reopens_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test.sqlite3")
            state = AppState(path)
            model = ModelInfo("registered-test", "1", "NumPy", [], [], 0, status="registered")
            state.persist_model(model)
            state.persist_usage(InferenceResponse("id", model.name, {"text": "private"}, 1.5))
            reopened = AppState(path)
            self.assertIn(model.name, reopened.models)
            self.assertEqual(reopened.storage.usage_summary()["recorded_requests"], 1)
            self.assertEqual(reopened.storage.usage_summary()["avg_latency_ms"], 1.5)
            with open(path, "rb") as handle:
                self.assertNotIn(b"private", handle.read())
            Storage(path).delete_model(model.name)
            self.assertNotIn(model.name, AppState(path).models)


class TestWorkerStatus(unittest.IsolatedAsyncioTestCase):
    async def test_worker_health_and_memory(self):
        worker = InferenceWorker("test", gpu_memory_mb=10)
        self.assertEqual(worker.status, "offline")
        await worker.start()
        self.assertEqual(worker.status, "healthy")
        self.assertIsNone(worker.gpu_utilization)
        self.assertTrue(worker.load_model("x", lambda x: x, 5))
        self.assertFalse(worker.load_model("x", lambda x: x, 5))
        self.assertEqual(worker.available_memory_mb, 5)
        await worker.stop(drain=False)
        self.assertEqual(worker.status, "offline")
