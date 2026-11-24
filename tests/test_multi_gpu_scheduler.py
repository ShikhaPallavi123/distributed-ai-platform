import unittest
from scripts.multi_gpu_benchmark import Scheduler


class FakeEndpoint:
    def __init__(self, label, failed=False):
        self.label = label
        self.failed = failed

    def generate(self, prompt):
        if self.failed:
            raise OSError("offline")
        return 10


class TestScheduler(unittest.TestCase):
    def test_round_robin(self):
        s = Scheduler([FakeEndpoint("a"), FakeEndpoint("b")], "round_robin")
        self.assertEqual([s.generate("test")["worker"] for _ in range(4)], ["a", "b", "a", "b"])

    def test_failover_and_recovery(self):
        now = [100]
        a, b = FakeEndpoint("a", True), FakeEndpoint("b")
        s = Scheduler([a, b], "adaptive", clock=lambda: now[0])
        first = s.generate("test")
        self.assertEqual(first["worker"], "b")
        self.assertEqual(first["attempts"], 2)
        self.assertEqual(s.stats[0]["failures"], 1)
        self.assertEqual(s.generate("test")["worker"], "b")
        a.failed = False
        now[0] += 6
        self.assertEqual(s.generate("test")["worker"], "a")

    def test_all_unavailable(self):
        s = Scheduler([FakeEndpoint("a", True), FakeEndpoint("b", True)], "round_robin")
        self.assertEqual(s.generate("test")["status"], "failed")
