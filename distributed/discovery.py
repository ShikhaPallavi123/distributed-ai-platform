"""Service registration and discovery."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceInstance:
    """A single instance of a registered service."""
    service_name: str
    instance_id: str
    host: str
    port: int
    metadata: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    healthy: bool = True
    last_check: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)


class ServiceRegistry:
    """Central registry for service instances.

    Supports registration, deregistration, health-aware lookup,
    and tag-based filtering.
    """

    def __init__(self) -> None:
        self._services: dict[str, dict[str, ServiceInstance]] = {}
        self._health_ttl: float = 30.0  # seconds before unhealthy

    def register(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> ServiceInstance:
        """Register a service instance."""
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            metadata=metadata or {},
            tags=tags or [],
        )
        self._services.setdefault(service_name, {})[instance_id] = instance
        return instance

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Remove a service instance. Returns True if it existed."""
        bucket = self._services.get(service_name, {})
        return bucket.pop(instance_id, None) is not None

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Record a health heartbeat for an instance."""
        inst = self._get_instance(service_name, instance_id)
        if inst is None:
            return False
        inst.last_check = time.time()
        inst.healthy = True
        return True

    def mark_unhealthy(self, service_name: str, instance_id: str) -> None:
        inst = self._get_instance(service_name, instance_id)
        if inst:
            inst.healthy = False

    # -- discovery --------------------------------------------------------

    def discover(
        self,
        service_name: str,
        *,
        healthy_only: bool = True,
        tags: list[str] | None = None,
    ) -> list[ServiceInstance]:
        """Find instances of a service, optionally filtered by health/tags."""
        bucket = self._services.get(service_name, {})
        instances = list(bucket.values())
        if healthy_only:
            now = time.time()
            instances = [
                i
                for i in instances
                if i.healthy and (now - i.last_check) < self._health_ttl
            ]
        if tags:
            tag_set = set(tags)
            instances = [i for i in instances if tag_set.issubset(i.tags)]
        return instances

    def discover_one(
        self, service_name: str, *, healthy_only: bool = True
    ) -> ServiceInstance | None:
        """Return a random healthy instance (simple load-spread)."""
        candidates = self.discover(service_name, healthy_only=healthy_only)
        return random.choice(candidates) if candidates else None

    def list_services(self) -> list[str]:
        """Return names of all registered services."""
        return sorted(self._services.keys())

    def all_instances(self, service_name: str) -> list[ServiceInstance]:
        return list(self._services.get(service_name, {}).values())

    # -- DNS-style lookup -------------------------------------------------

    def dns_lookup(self, service_name: str) -> list[tuple[str, int]]:
        """Return (host, port) pairs, mimicking DNS SRV records."""
        return [
            (i.host, i.port)
            for i in self.discover(service_name, healthy_only=True)
        ]

    # -- gossip-style fanout ----------------------------------------------

    def gossip_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Serialise registry state for gossip exchange."""
        result: dict[str, list[dict[str, Any]]] = {}
        for svc, bucket in self._services.items():
            result[svc] = [
                {
                    "instance_id": i.instance_id,
                    "host": i.host,
                    "port": i.port,
                    "healthy": i.healthy,
                    "last_check": i.last_check,
                    "metadata": i.metadata,
                    "tags": i.tags,
                }
                for i in bucket.values()
            ]
        return result

    def merge_gossip(self, snapshot: dict[str, list[dict[str, Any]]]) -> None:
        """Merge a gossip snapshot into local state (crdt-ish LWW)."""
        for svc, instances in snapshot.items():
            bucket = self._services.setdefault(svc, {})
            for data in instances:
                iid = data["instance_id"]
                existing = bucket.get(iid)
                if existing is None or data["last_check"] > existing.last_check:
                    bucket[iid] = ServiceInstance(
                        service_name=svc,
                        instance_id=iid,
                        host=data["host"],
                        port=data["port"],
                        metadata=data.get("metadata", {}),
                        tags=data.get("tags", []),
                        healthy=data.get("healthy", True),
                        last_check=data["last_check"],
                    )

    # -- helpers ----------------------------------------------------------

    def _get_instance(
        self, service_name: str, instance_id: str
    ) -> ServiceInstance | None:
        return self._services.get(service_name, {}).get(instance_id)
