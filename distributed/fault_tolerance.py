"""Fault tolerance patterns: circuit breaker, retry, health checks, bulkhead."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


# ── Circuit Breaker ─────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Prevents cascading failures by tracking error rates.

    CLOSED  -> normal operation, errors counted
    OPEN    -> calls rejected immediately
    HALF_OPEN -> one probe call allowed to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._state = CBState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CBState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any) -> T:
        """Execute *fn* through the circuit breaker."""
        st = self.state
        if st == CBState.OPEN:
            raise RuntimeError("circuit breaker is OPEN")
        if st == CBState.HALF_OPEN and self._half_open_calls >= self.half_open_max:
            raise RuntimeError("circuit breaker HALF_OPEN limit reached")

        try:
            if st == CBState.HALF_OPEN:
                self._half_open_calls += 1
            result = await fn(*args)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = CBState.CLOSED

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CBState.OPEN


# ── Retry Policy ────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """Exponential back-off with jitter."""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.25

    async def execute(self, fn: Callable[..., Awaitable[T]], *args: Any) -> T:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(*args)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                delay += random.uniform(0, self.jitter * delay)
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ── Health Checker ──────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    name: str
    healthy: bool
    latency_ms: float = 0.0
    detail: str = ""


class HealthChecker:
    """Runs liveness and readiness probes against a target."""

    def __init__(self) -> None:
        self._probes: dict[str, Callable[[], Awaitable[bool]]] = {}

    def register_probe(
        self, name: str, fn: Callable[[], Awaitable[bool]]
    ) -> None:
        self._probes[name] = fn

    async def check_all(self) -> list[ProbeResult]:
        results: list[ProbeResult] = []
        for name, fn in self._probes.items():
            start = time.time()
            try:
                ok = await fn()
                latency = (time.time() - start) * 1000
                results.append(ProbeResult(name=name, healthy=ok, latency_ms=latency))
            except Exception as exc:
                latency = (time.time() - start) * 1000
                results.append(
                    ProbeResult(name=name, healthy=False, latency_ms=latency, detail=str(exc))
                )
        return results

    async def is_healthy(self) -> bool:
        results = await self.check_all()
        return all(r.healthy for r in results)


# ── Failover Manager ───────────────────────────────────────────────────

class Failover:
    """Automatic failover to replica nodes when the primary fails."""

    def __init__(self, replicas: list[str] | None = None) -> None:
        self._primary: str | None = None
        self._replicas: list[str] = list(replicas or [])
        self._failed: set[str] = set()

    def set_primary(self, node_id: str) -> None:
        self._primary = node_id

    def add_replica(self, node_id: str) -> None:
        if node_id not in self._replicas:
            self._replicas.append(node_id)

    def report_failure(self, node_id: str) -> str | None:
        """Mark *node_id* as failed and return the new primary (if any)."""
        self._failed.add(node_id)
        if node_id == self._primary:
            return self._promote_next()
        return self._primary

    def current_primary(self) -> str | None:
        return self._primary

    def _promote_next(self) -> str | None:
        for r in self._replicas:
            if r not in self._failed:
                self._primary = r
                self._replicas.remove(r)
                return r
        self._primary = None
        return None


# ── Bulkhead ────────────────────────────────────────────────────────────

class BulkheadPattern:
    """Resource isolation using bounded concurrency per partition."""

    def __init__(self, max_concurrent: int = 10) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._max = max_concurrent

    def _sem(self, partition: str) -> asyncio.Semaphore:
        if partition not in self._semaphores:
            self._semaphores[partition] = asyncio.Semaphore(self._max)
        return self._semaphores[partition]

    async def execute(
        self, partition: str, fn: Callable[..., Awaitable[T]], *args: Any
    ) -> T:
        """Run *fn* inside the bulkhead for *partition*."""
        sem = self._sem(partition)
        async with sem:
            return await fn(*args)

    def available(self, partition: str) -> int:
        """How many slots remain in *partition*."""
        sem = self._sem(partition)
        return sem._value  # type: ignore[attr-defined]
