"""Tests for OS-level primitives: memory, scheduler, IPC, profiler."""
import unittest
import time
from core import (
    MemoryPool, AllocationStrategy, TensorMemoryManager,
    TaskScheduler, Task, TaskState, SchedulingAlgorithm,
    MessageBus, Message, MessageType,
    SystemProfiler, InferenceProfiler, FlameGraph,
)


class TestMemoryPool(unittest.TestCase):
    def test_allocate_and_free(self):
        pool = MemoryPool(4096, AllocationStrategy.FIRST_FIT)
        blk = pool.allocate(256)
        self.assertIsNotNone(blk)
        self.assertGreaterEqual(blk.size, 256)
        pool.free(blk.address)
        self.assertEqual(pool.used, 0)

    def test_allocate_too_large(self):
        pool = MemoryPool(1024)
        blk = pool.allocate(2048)
        self.assertIsNone(blk)

    def test_multiple_allocations(self):
        pool = MemoryPool(4096)
        blocks = [pool.allocate(64) for _ in range(8)]
        self.assertTrue(all(b is not None for b in blocks))
        self.assertGreater(pool.used, 0)

    def test_defragment(self):
        pool = MemoryPool(4096)
        b1 = pool.allocate(256)
        b2 = pool.allocate(256)
        pool.free(b1.address)
        pool.defragment()
        self.assertGreater(pool.free_space, 0)

    def test_stats(self):
        pool = MemoryPool(4096)
        pool.allocate(512)
        stats = pool.stats()
        self.assertIsInstance(stats, dict)

    def test_tensor_memory_manager(self):
        tmm = TensorMemoryManager()
        handle = tmm.alloc_tensor((32, 64))
        self.assertIsNotNone(handle)


class TestTaskScheduler(unittest.TestCase):
    def test_submit_and_complete(self):
        sched = TaskScheduler(SchedulingAlgorithm.FIFO, num_workers=2)
        results = []
        task = Task(task_id=1, fn=lambda: results.append(42) or 42, priority=5)
        sched.submit_task(task)
        time.sleep(0.3)
        sched.shutdown()
        self.assertIn(42, results)

    def test_priority_scheduling(self):
        sched = TaskScheduler(SchedulingAlgorithm.PRIORITY, num_workers=1)
        order = []
        for i in range(3):
            t = Task(task_id=i, fn=lambda x=i: order.append(x) or x, priority=10 - i)
            sched.submit_task(t)
        time.sleep(0.5)
        sched.shutdown()
        self.assertEqual(len(order), 3)

    def test_scheduler_creation(self):
        sched = TaskScheduler(num_workers=2)
        self.assertIsNotNone(sched)
        sched.shutdown()


class TestMessageBus(unittest.TestCase):
    def test_pubsub(self):
        bus = MessageBus()
        received = []
        bus.subscribe("test.topic", lambda msg: received.append(msg))
        msg = Message(topic="test.topic", payload={"data": "hello"})
        bus.publish(msg)
        time.sleep(0.1)
        self.assertEqual(len(received), 1)

    def test_unsubscribe(self):
        bus = MessageBus()
        received = []
        handler = lambda msg: received.append(msg)
        bus.subscribe("topic.a", handler)
        bus.unsubscribe("topic.a", handler)
        msg = Message(topic="topic.a", payload={"data": "nope"})
        bus.publish(msg)
        time.sleep(0.1)
        self.assertEqual(len(received), 0)

    def test_stats(self):
        bus = MessageBus()
        msg = Message(topic="x", payload={"v": 1})
        bus.publish(msg)
        stats = bus.stats()
        self.assertIsInstance(stats, dict)


class TestSystemProfiler(unittest.TestCase):
    def test_profiler_creation(self):
        p = SystemProfiler(interval_ms=5000)
        self.assertIsNotNone(p)

    def test_inference_profiler(self):
        ip = InferenceProfiler()
        with ip.layer("linear"):
            time.sleep(0.01)
        self.assertIsNotNone(ip)

    def test_flame_graph(self):
        fg = FlameGraph()
        fg.add_stack(["main", "forward", "linear"], 1.5)
        fg.add_stack(["main", "forward", "relu"], 0.5)
        hotspots = fg.hotspots(top_n=2)
        self.assertGreater(len(hotspots), 0)


if __name__ == "__main__":
    unittest.main()
