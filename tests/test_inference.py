"""Tests for the Inference Engine module."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestBatching(unittest.TestCase):
    def test_dynamic_batcher(self):
        from inference.batching import DynamicBatcher
        batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)
        self.assertIsNotNone(batcher)

    def test_batcher_has_submit(self):
        from inference.batching import DynamicBatcher
        batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)
        self.assertTrue(hasattr(batcher, 'submit'))


class TestLoadBalancer(unittest.TestCase):
    def test_round_robin(self):
        from inference.load_balancer import LoadBalancer, Endpoint, RoundRobin
        lb = LoadBalancer(strategy=RoundRobin())
        lb.add_endpoint(Endpoint(endpoint_id="w1"))
        lb.add_endpoint(Endpoint(endpoint_id="w2"))
        ep = lb.select()
        self.assertIsNotNone(ep)
        self.assertIn(ep.endpoint_id, ["w1", "w2"])

    def test_default_strategy(self):
        from inference.load_balancer import LoadBalancer, Endpoint
        lb = LoadBalancer()
        lb.add_endpoint(Endpoint(endpoint_id="w1"))
        ep = lb.select()
        self.assertIsNotNone(ep)


class TestModelRegistry(unittest.TestCase):
    def test_register_model(self):
        from inference.model_registry import ModelRegistry, ModelArtifact
        registry = ModelRegistry()
        artifact = ModelArtifact(model_name="test-model", version="1.0")
        registry.register(artifact)
        info = registry.get("test-model")
        self.assertIsNotNone(info)

    def test_list_models(self):
        from inference.model_registry import ModelRegistry, ModelArtifact
        registry = ModelRegistry()
        registry.register(ModelArtifact(model_name="model-a", version="1.0"))
        registry.register(ModelArtifact(model_name="model-b", version="2.0"))
        models = registry.list_models()
        self.assertEqual(len(models), 2)


class TestTaskQueue(unittest.TestCase):
    def test_priority_queue(self):
        import asyncio
        from inference.task_queue import PriorityTaskQueue, TaskItem

        async def _test():
            q = PriorityTaskQueue()
            await q.enqueue(TaskItem(task_id="t1", payload={"task": "low"}, priority=1))
            await q.enqueue(TaskItem(task_id="t2", payload={"task": "high"}, priority=10))
            task = await q.dequeue()
            self.assertEqual(task.payload["task"], "high")
        asyncio.run(_test())


class TestPipeline(unittest.TestCase):
    def test_inference_pipeline(self):
        from inference.pipeline import InferencePipeline
        pipeline = InferencePipeline()
        self.assertIsNotNone(pipeline)


class TestEngine(unittest.TestCase):
    def test_engine_creation(self):
        from inference.engine import InferenceEngine
        engine = InferenceEngine()
        self.assertIsNotNone(engine)


if __name__ == "__main__":
    unittest.main()
