"""Tests for the Distributed Systems module."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestNode(unittest.TestCase):
    def test_node_creation(self):
        from distributed.node import Node
        node = Node(node_id="test-1", host="localhost", port=8000)
        self.assertIsNotNone(node)

    def test_node_info(self):
        from distributed.node import Node
        node = Node(node_id="test-2", host="localhost", port=8001)
        info = node.info
        self.assertIsNotNone(info)
        self.assertEqual(info.node_id, "test-2")


class TestCluster(unittest.TestCase):
    def test_cluster_creation(self):
        from distributed.cluster import Cluster
        cluster = Cluster()
        self.assertIsNotNone(cluster)

    def test_register_node(self):
        import asyncio
        from distributed.cluster import Cluster
        from distributed.node import Node
        async def _test():
            cluster = Cluster()
            node = Node(node_id="n1", host="localhost", port=8001)
            await cluster.register(node)
            self.assertEqual(cluster.size, 1)
        asyncio.run(_test())


class TestConsensus(unittest.TestCase):
    def test_raft_node_creation(self):
        from distributed.consensus import RaftNode, RaftState
        node = RaftNode(node_id="raft-1", peers=["raft-2", "raft-3"])
        self.assertEqual(node.node_id, "raft-1")
        self.assertEqual(node.state, RaftState.FOLLOWER)

    def test_raft_term(self):
        from distributed.consensus import RaftNode
        node = RaftNode(node_id="raft-1", peers=[])
        self.assertEqual(node.current_term, 0)


class TestDiscovery(unittest.TestCase):
    def test_service_registry(self):
        from distributed.discovery import ServiceRegistry
        registry = ServiceRegistry()
        registry.register("api-service", "inst-1", "localhost", 8080)
        services = registry.discover("api-service")
        self.assertGreater(len(services), 0)


class TestFaultTolerance(unittest.TestCase):
    def test_circuit_breaker(self):
        from distributed.fault_tolerance import CircuitBreaker, CBState
        cb = CircuitBreaker(failure_threshold=3)
        self.assertEqual(cb.state, CBState.CLOSED)

    def test_circuit_breaker_state_enum(self):
        from distributed.fault_tolerance import CircuitBreaker, CBState
        cb = CircuitBreaker(failure_threshold=2)
        self.assertIsNotNone(cb.state)
        self.assertIn(cb.state, list(CBState))

    def test_retry_policy(self):
        from distributed.fault_tolerance import RetryPolicy
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        self.assertIsNotNone(policy)


class TestReplication(unittest.TestCase):
    def test_model_replicator(self):
        from distributed.replication import ModelReplicator
        replicator = ModelReplicator()
        self.assertIsNotNone(replicator)


if __name__ == "__main__":
    unittest.main()
