"""Coordinator: central brain for worker management and request dispatching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .worker import InferenceWorker, WorkerPool


@dataclass
class WorkerHealth:
    worker_id: str
    healthy: bool = True
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class ScalingDecision:
    action: str  # "scale_up", "scale_down", "none"
    reason: str = ""
    target_count: int = 0


class Coordinator:
    """Central brain that manages workers, dispatches requests, and
    makes auto-scaling decisions.
    """

    def __init__(
        self,
        health_interval: float = 5.0,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
    ) -> None:
        self.pool = WorkerPool()
        self._health: dict[str, WorkerHealth] = {}
        self._health_interval = health_interval
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._health_task: asyncio.Task[None] | None = None
        self._running = False
        self._metrics = _CoordinatorMetrics()
        self._deployed_models: dict[str, list[str]] = {}  # model -> worker_ids

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        await self.pool.start_all()
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        await self.pool.stop_all()

    # -- worker registration ----------------------------------------------

    def register_worker(self, worker: InferenceWorker) -> None:
        self.pool.add(worker)
        self._health[worker.worker_id] = WorkerHealth(worker_id=worker.worker_id)

    def deregister_worker(self, worker_id: str) -> None:
        self.pool.remove(worker_id)
        self._health.pop(worker_id, None)

    # -- request dispatch -------------------------------------------------

    async def dispatch(
        self,
        model_name: str,
        data: Any,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Route a request to the best available worker."""
        worker = self._select_worker(model_name)
        if worker is None:
            raise RuntimeError(f"no healthy worker available for {model_name!r}")

        self._metrics.dispatched += 1
        start = time.time()
        try:
            result = await worker.process(model_name, data, params or {})
            latency = (time.time() - start) * 1000
            health = self._health.get(worker.worker_id)
            if health:
                health.avg_latency_ms = (
                    0.9 * health.avg_latency_ms + 0.1 * latency
                )
                health.consecutive_failures = 0
            return result
        except Exception:
            health = self._health.get(worker.worker_id)
            if health:
                health.consecutive_failures += 1
                if health.consecutive_failures >= 3:
                    health.healthy = False
            raise

    # -- model deployment -------------------------------------------------

    def deploy_model(
        self,
        model_name: str,
        model: Any,
        size_mb: float = 500.0,
        worker_ids: list[str] | None = None,
    ) -> list[str]:
        """Load a model onto workers. Returns the ids that succeeded."""
        targets = worker_ids or [w.worker_id for w in self.pool._workers.values()]
        loaded: list[str] = []
        for wid in targets:
            w = self.pool.get(wid)
            if w and w.load_model(model_name, model, size_mb):
                loaded.append(wid)
        self._deployed_models[model_name] = loaded
        return loaded

    def undeploy_model(self, model_name: str) -> None:
        for wid in self._deployed_models.pop(model_name, []):
            w = self.pool.get(wid)
            if w:
                w.unload_model(model_name)

    # -- auto-scaling -----------------------------------------------------

    def evaluate_scaling(self) -> ScalingDecision:
        """Decide whether to scale up, down, or hold."""
        workers = list(self.pool._workers.values())
        if not workers:
            return ScalingDecision("scale_up", "no workers", 1)

        utilizations = [
            w._active_requests / max(w.max_batch, 1) for w in workers
        ]
        avg_util = sum(utilizations) / len(utilizations)

        if avg_util > self._scale_up_threshold:
            return ScalingDecision(
                "scale_up",
                f"avg utilization {avg_util:.0%} > {self._scale_up_threshold:.0%}",
                len(workers) + 1,
            )
        if avg_util < self._scale_down_threshold and len(workers) > 1:
            return ScalingDecision(
                "scale_down",
                f"avg utilization {avg_util:.0%} < {self._scale_down_threshold:.0%}",
                len(workers) - 1,
            )
        return ScalingDecision("none", f"utilization {avg_util:.0%} within bounds")

    # -- health -----------------------------------------------------------

    async def _health_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._health_interval)
            await self._check_health()

    async def _check_health(self) -> None:
        for wid, health in self._health.items():
            w = self.pool.get(wid)
            if w is None:
                health.healthy = False
                continue
            health.healthy = w.status in ("healthy", "busy")
            health.last_check = time.time()

    def _select_worker(self, model_name: str) -> InferenceWorker | None:
        """Pick the best healthy worker that has the model loaded."""
        candidates = [
            w
            for w in self.pool.available_workers(model_name)
            if self._health.get(w.worker_id, WorkerHealth(w.worker_id)).healthy
        ]
        if not candidates:
            return None
        # Prefer lowest latency among healthy, non-busy workers

        def score(w: InferenceWorker) -> float:
            h = self._health.get(w.worker_id)
            return (h.avg_latency_ms if h else 0) + w._active_requests * 10
        return min(candidates, key=score)

    def collect_metrics(self) -> dict[str, Any]:
        return {
            "dispatched": self._metrics.dispatched,
            "workers": self.pool.size,
            "healthy_workers": sum(1 for h in self._health.values() if h.healthy),
            "deployed_models": {k: len(v) for k, v in self._deployed_models.items()},
            "worker_stats": self.pool.all_stats(),
        }


@dataclass
class _CoordinatorMetrics:
    dispatched: int = 0
