"""Main inference engine orchestrator."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class RequestStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()


@dataclass
class InferenceConfig:
    """Tuning knobs for the engine."""
    batch_size: int = 32
    timeout: float = 30.0
    max_retries: int = 3
    max_concurrent: int = 64
    warmup_requests: int = 2


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_name: str = ""
    inputs: np.ndarray | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING


@dataclass
class InferenceResponse:
    """Result of an inference request."""
    request_id: str
    outputs: np.ndarray | None = None
    model_name: str = ""
    latency_ms: float = 0.0
    status: RequestStatus = RequestStatus.COMPLETED
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """Central orchestrator that routes requests through the inference pipeline.

    Integrates with the model registry, dispatches to workers, aggregates
    results, and runs the preprocessing-inference-postprocessing pipeline.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._model_registry: dict[str, Any] = {}  # model_name -> model object
        self._workers: list[Any] = []
        self._pipeline: Any | None = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._metrics = _EngineMetrics()
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def register_model(self, name: str, model: Any) -> None:
        """Add a model to the local registry."""
        self._model_registry[name] = model

    def set_pipeline(self, pipeline: Any) -> None:
        """Attach an InferencePipeline for pre/post processing."""
        self._pipeline = pipeline

    def add_worker(self, worker: Any) -> None:
        self._workers.append(worker)

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request end-to-end."""
        async with self._semaphore:
            return await self._process(request)

    async def infer_batch(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResponse]:
        """Process multiple requests concurrently."""
        tasks = [self.infer(r) for r in requests]
        return await asyncio.gather(*tasks)

    async def _process(self, request: InferenceRequest) -> InferenceResponse:
        start = time.time()
        request.status = RequestStatus.RUNNING
        self._metrics.requests_started += 1

        try:
            data = request.inputs

            # Preprocessing
            if self._pipeline and hasattr(self._pipeline, "preprocess"):
                data = await self._pipeline.preprocess(data)

            # Model inference (with timeout)
            result = await asyncio.wait_for(
                self._run_model(request.model_name, data, request.parameters),
                timeout=self.config.timeout,
            )

            # Postprocessing
            if self._pipeline and hasattr(self._pipeline, "postprocess"):
                result = await self._pipeline.postprocess(result)

            latency = (time.time() - start) * 1000
            self._metrics.requests_completed += 1
            self._metrics.total_latency_ms += latency
            request.status = RequestStatus.COMPLETED

            return InferenceResponse(
                request_id=request.request_id,
                outputs=result,
                model_name=request.model_name,
                latency_ms=latency,
                status=RequestStatus.COMPLETED,
            )

        except asyncio.TimeoutError:
            request.status = RequestStatus.TIMEOUT
            self._metrics.requests_failed += 1
            return InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                latency_ms=(time.time() - start) * 1000,
                status=RequestStatus.TIMEOUT,
                error="request timed out",
            )
        except Exception as exc:
            request.status = RequestStatus.FAILED
            self._metrics.requests_failed += 1
            return InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                latency_ms=(time.time() - start) * 1000,
                status=RequestStatus.FAILED,
                error=str(exc),
            )

    async def _run_model(
        self, model_name: str, data: Any, params: dict[str, Any]
    ) -> np.ndarray:
        """Execute model inference, dispatching to a worker if available."""
        # Dispatch to first available worker
        for worker in self._workers:
            if hasattr(worker, "process"):
                return await worker.process(model_name, data, params)

        # Fallback: use local model registry
        model = self._model_registry.get(model_name)
        if model is None:
            raise ValueError(f"model {model_name!r} not found")

        if callable(model):
            result = model(data)
            if asyncio.iscoroutine(result):
                return await result
            return result

        raise TypeError(f"model {model_name!r} is not callable")

    def metrics(self) -> dict[str, Any]:
        m = self._metrics
        avg = (
            m.total_latency_ms / m.requests_completed
            if m.requests_completed
            else 0
        )
        return {
            "started": m.requests_started,
            "completed": m.requests_completed,
            "failed": m.requests_failed,
            "avg_latency_ms": round(avg, 2),
        }


@dataclass
class _EngineMetrics:
    requests_started: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    total_latency_ms: float = 0.0
