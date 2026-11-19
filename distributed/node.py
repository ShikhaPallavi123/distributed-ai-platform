"""Node abstraction for distributed inference cluster."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class NodeState(Enum):
    """Lifecycle states for a cluster node."""
    JOINING = auto()
    ACTIVE = auto()
    DRAINING = auto()
    DEAD = auto()


@dataclass
class NodeCapabilities:
    """Hardware and software capabilities of a node."""
    gpu_count: int = 0
    memory_gb: float = 0.0
    models_loaded: list[str] = field(default_factory=list)
    max_batch_size: int = 32
    supported_dtypes: list[str] = field(default_factory=lambda: ["float32"])


@dataclass
class NodeInfo:
    """Metadata describing a node in the cluster."""
    node_id: str
    host: str
    port: int
    capabilities: NodeCapabilities
    state: NodeState = NodeState.JOINING
    last_heartbeat: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    zone: str = "default"


class Node:
    """A single node in the distributed inference cluster.

    Handles its own lifecycle, heartbeat broadcasting, and health reporting.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        capabilities: NodeCapabilities | None = None,
        node_id: str | None = None,
    ) -> None:
        self.info = NodeInfo(
            node_id=node_id or uuid.uuid4().hex[:12],
            host=host,
            port=port,
            capabilities=capabilities or NodeCapabilities(),
        )
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_interval: float = 2.0
        self._running = False
        self._on_heartbeat: list[Any] = []

    @property
    def node_id(self) -> str:
        return self.info.node_id

    @property
    def state(self) -> NodeState:
        return self.info.state

    async def start(self) -> None:
        """Transition to ACTIVE and begin heartbeat loop."""
        self.info.state = NodeState.ACTIVE
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self, drain: bool = True) -> None:
        """Gracefully stop the node, optionally draining first."""
        if drain:
            self.info.state = NodeState.DRAINING
            await asyncio.sleep(0)  # yield so watchers see DRAINING
        self._running = False
        self.info.state = NodeState.DEAD
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    def heartbeat(self) -> float:
        """Record a heartbeat and return the timestamp."""
        self.info.last_heartbeat = time.time()
        return self.info.last_heartbeat

    def health_check(self, timeout: float = 5.0) -> dict[str, Any]:
        """Return current health status of this node."""
        now = time.time()
        age = now - self.info.last_heartbeat
        healthy = self.info.state == NodeState.ACTIVE and age < timeout
        return {
            "node_id": self.node_id,
            "state": self.info.state.name,
            "healthy": healthy,
            "heartbeat_age_s": round(age, 3),
            "capabilities": {
                "gpu_count": self.info.capabilities.gpu_count,
                "memory_gb": self.info.capabilities.memory_gb,
                "models_loaded": list(self.info.capabilities.models_loaded),
            },
        }

    def metadata(self) -> dict[str, Any]:
        """Serialisable snapshot of the node."""
        return {
            "node_id": self.node_id,
            "host": self.info.host,
            "port": self.info.port,
            "state": self.info.state.name,
            "zone": self.info.zone,
            "tags": dict(self.info.tags),
            "last_heartbeat": self.info.last_heartbeat,
        }

    async def _heartbeat_loop(self) -> None:
        """Periodically emit heartbeats while running."""
        while self._running:
            self.heartbeat()
            for cb in self._on_heartbeat:
                try:
                    cb(self.info)
                except Exception:
                    pass
            await asyncio.sleep(self._heartbeat_interval)
