"""Load balancing strategies for inference request routing."""

from __future__ import annotations

import hashlib
import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Endpoint:
    """A backend endpoint that can receive requests."""
    endpoint_id: str
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    gpu_free_mb: float = 0.0
    memory_free_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_latency(self, ms: float, alpha: float = 0.2) -> None:
        self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * ms


class LoadBalancerStrategy(ABC):
    """Base class for load-balancing strategies."""

    @abstractmethod
    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        ...


class RoundRobin(LoadBalancerStrategy):
    def __init__(self) -> None:
        self._counter = itertools.cycle(range(10**9))

    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        idx = next(self._counter) % len(alive)
        return alive[idx]


class WeightedRoundRobin(LoadBalancerStrategy):
    def __init__(self) -> None:
        self._idx = 0

    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        expanded = [e for e in alive for _ in range(e.weight)]
        if not expanded:
            return None
        self._idx = (self._idx + 1) % len(expanded)
        return expanded[self._idx]


class LeastConnections(LoadBalancerStrategy):
    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        return min(alive, key=lambda e: e.active_connections)


class LatencyBased(LoadBalancerStrategy):
    """Route to the endpoint with the lowest observed latency."""
    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        return min(alive, key=lambda e: e.avg_latency_ms)


class ResourceAware(LoadBalancerStrategy):
    """Route based on available GPU/memory resources."""
    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        return max(alive, key=lambda e: e.gpu_free_mb + e.memory_free_mb)


class ConsistentHashing(LoadBalancerStrategy):
    """Route requests for the same key to the same endpoint (model affinity)."""

    def __init__(self, vnodes: int = 150) -> None:
        self._vnodes = vnodes

    def select(self, endpoints: list[Endpoint], key: str = "") -> Endpoint | None:
        alive = [e for e in endpoints if e.healthy]
        if not alive:
            return None
        ring: list[tuple[int, Endpoint]] = []
        for ep in alive:
            for i in range(self._vnodes):
                h = self._hash(f"{ep.endpoint_id}:{i}")
                ring.append((h, ep))
        ring.sort(key=lambda x: x[0])

        kh = self._hash(key)
        for pos, ep in ring:
            if pos >= kh:
                return ep
        return ring[0][1]  # wrap around

    @staticmethod
    def _hash(value: str) -> int:
        return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (2**32)


class LoadBalancer:
    """Unified load balancer with pluggable strategies and health filtering."""

    def __init__(self, strategy: LoadBalancerStrategy | None = None) -> None:
        self.strategy = strategy or RoundRobin()
        self._endpoints: dict[str, Endpoint] = {}

    def add_endpoint(self, ep: Endpoint) -> None:
        self._endpoints[ep.endpoint_id] = ep

    def remove_endpoint(self, endpoint_id: str) -> None:
        self._endpoints.pop(endpoint_id, None)

    def set_strategy(self, strategy: LoadBalancerStrategy) -> None:
        self.strategy = strategy

    def select(self, key: str = "") -> Endpoint | None:
        return self.strategy.select(list(self._endpoints.values()), key)

    def mark_healthy(self, endpoint_id: str) -> None:
        ep = self._endpoints.get(endpoint_id)
        if ep:
            ep.healthy = True

    def mark_unhealthy(self, endpoint_id: str) -> None:
        ep = self._endpoints.get(endpoint_id)
        if ep:
            ep.healthy = False

    def healthy_endpoints(self) -> list[Endpoint]:
        return [e for e in self._endpoints.values() if e.healthy]
