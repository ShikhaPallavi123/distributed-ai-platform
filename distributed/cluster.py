"""Cluster management: topology, membership, and state propagation."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .node import Node, NodeInfo, NodeState


class Topology(Enum):
    """Supported cluster topologies."""
    RING = auto()
    MESH = auto()


@dataclass
class ClusterState:
    """Consistent snapshot of the cluster at a point in time."""
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    version: int = 0
    timestamp: float = field(default_factory=time.time)

    def active_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes.values() if n.state == NodeState.ACTIVE]

    def node_ids(self) -> list[str]:
        return sorted(self.nodes.keys())


class Cluster:
    """Manages a fleet of inference nodes.

    Supports ring and mesh topologies, gossip-based state propagation,
    and automatic rebalancing on membership changes.
    """

    def __init__(self, topology: Topology = Topology.MESH) -> None:
        self.topology = topology
        self._state = ClusterState()
        self._nodes: dict[str, Node] = {}
        self._gossip_task: asyncio.Task[None] | None = None
        self._gossip_interval: float = 1.0
        self._rebalance_callbacks: list[Any] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ClusterState:
        return self._state

    @property
    def size(self) -> int:
        return len(self._state.nodes)

    async def register(self, node: Node) -> None:
        """Add a node to the cluster and trigger rebalancing."""
        async with self._lock:
            self._nodes[node.node_id] = node
            self._state.nodes[node.node_id] = node.info
            self._state.version += 1
            self._state.timestamp = time.time()
        await self._rebalance("join", node.node_id)

    async def deregister(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        async with self._lock:
            self._nodes.pop(node_id, None)
            self._state.nodes.pop(node_id, None)
            self._state.version += 1
            self._state.timestamp = time.time()
        await self._rebalance("leave", node_id)

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def get_node_info(self, node_id: str) -> NodeInfo | None:
        return self._state.nodes.get(node_id)

    def neighbours(self, node_id: str) -> list[str]:
        """Return the neighbours of *node_id* according to the topology."""
        ids = self._state.node_ids()
        if node_id not in ids:
            return []
        if self.topology == Topology.MESH:
            return [nid for nid in ids if nid != node_id]
        # Ring: immediate predecessor and successor
        idx = ids.index(node_id)
        result: list[str] = []
        if len(ids) > 1:
            result.append(ids[(idx - 1) % len(ids)])
            result.append(ids[(idx + 1) % len(ids)])
        return list(dict.fromkeys(result))  # deduplicate for 2-node ring

    # -- gossip protocol -------------------------------------------------

    async def start_gossip(self) -> None:
        """Begin background gossip protocol for state propagation."""
        self._gossip_task = asyncio.create_task(self._gossip_loop())

    async def stop_gossip(self) -> None:
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass

    async def _gossip_loop(self) -> None:
        while True:
            await asyncio.sleep(self._gossip_interval)
            await self._gossip_round()

    async def _gossip_round(self) -> None:
        """One round: pick a random peer and exchange state."""
        ids = list(self._nodes.keys())
        if len(ids) < 2:
            return
        src_id = random.choice(ids)
        peers = self.neighbours(src_id)
        if not peers:
            return
        target_id = random.choice(peers)
        # Merge: propagate heartbeat freshness
        src_info = self._state.nodes.get(src_id)
        tgt_info = self._state.nodes.get(target_id)
        if src_info and tgt_info:
            # In a real system we'd merge vector clocks; here we propagate
            # the most recent heartbeat knowledge.
            if src_info.last_heartbeat > tgt_info.last_heartbeat:
                tgt_info.last_heartbeat = max(
                    tgt_info.last_heartbeat, time.time()
                )

    # -- rebalancing ------------------------------------------------------

    def on_rebalance(self, callback: Any) -> None:
        self._rebalance_callbacks.append(callback)

    async def _rebalance(self, event: str, node_id: str) -> None:
        """Notify listeners that the cluster membership changed."""
        for cb in self._rebalance_callbacks:
            try:
                result = cb(event, node_id, self._state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # -- consistent hashing helper ----------------------------------------

    def hash_key(self, key: str) -> str:
        """Return the node id responsible for *key* via consistent hashing."""
        ids = self._state.node_ids()
        if not ids:
            raise RuntimeError("cluster is empty")
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return ids[h % len(ids)]
