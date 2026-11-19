"""Inference pipeline: chain of preprocessing, inference, and postprocessing stages."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StageMetrics:
    calls: int = 0
    total_ms: float = 0.0
    errors: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class PipelineStage(ABC):
    """Base class for a stage in the inference pipeline."""

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self.metrics = StageMetrics()

    @abstractmethod
    async def execute(self, data: Any) -> Any:
        """Process data and return the result."""
        ...

    async def run(self, data: Any) -> Any:
        """Execute with metrics collection and error tracking."""
        start = time.time()
        try:
            result = await self.execute(data)
            self.metrics.calls += 1
            self.metrics.total_ms += (time.time() - start) * 1000
            return result
        except Exception:
            self.metrics.errors += 1
            raise


class PreprocessStage(PipelineStage):
    """Normalize, tokenize, or resize input data."""

    def __init__(self, name: str = "preprocess", fn: Any = None) -> None:
        super().__init__(name)
        self._fn = fn

    async def execute(self, data: Any) -> Any:
        if self._fn:
            result = self._fn(data)
            if asyncio.iscoroutine(result):
                return await result
            return result
        # Default: cast to float32 and normalize to [0, 1]
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
            mx = arr.max()
            if mx > 0:
                arr = arr / mx
            return arr
        return data


class InferenceStage(PipelineStage):
    """Run the model forward pass."""

    def __init__(self, model: Any = None, name: str = "inference") -> None:
        super().__init__(name)
        self._model = model

    def set_model(self, model: Any) -> None:
        self._model = model

    async def execute(self, data: Any) -> Any:
        if self._model is None:
            raise RuntimeError("no model set on InferenceStage")
        result = self._model(data)
        if asyncio.iscoroutine(result):
            return await result
        return result


class PostprocessStage(PipelineStage):
    """Decode, threshold, or format model outputs."""

    def __init__(self, name: str = "postprocess", fn: Any = None) -> None:
        super().__init__(name)
        self._fn = fn

    async def execute(self, data: Any) -> Any:
        if self._fn:
            result = self._fn(data)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return data


class InferencePipeline:
    """Chain stages sequentially: preprocess -> inference -> postprocess."""

    def __init__(self, stages: list[PipelineStage] | None = None) -> None:
        self.stages: list[PipelineStage] = list(stages or [])

    def add_stage(self, stage: PipelineStage) -> None:
        self.stages.append(stage)

    async def run(self, data: Any) -> Any:
        """Execute all stages in order."""
        result = data
        for stage in self.stages:
            result = await stage.run(result)
        return result

    async def preprocess(self, data: Any) -> Any:
        """Run only preprocessing stages."""
        result = data
        for s in self.stages:
            if isinstance(s, PreprocessStage):
                result = await s.run(result)
        return result

    async def postprocess(self, data: Any) -> Any:
        """Run only postprocessing stages."""
        result = data
        for s in self.stages:
            if isinstance(s, PostprocessStage):
                result = await s.run(result)
        return result

    def stage_metrics(self) -> dict[str, dict[str, float]]:
        return {
            s.name: {
                "calls": s.metrics.calls,
                "avg_ms": round(s.metrics.avg_ms, 3),
                "errors": s.metrics.errors,
            }
            for s in self.stages
        }


class ParallelPipeline:
    """Run multiple stages concurrently and aggregate results."""

    def __init__(self, stages: list[PipelineStage] | None = None) -> None:
        self.stages: list[PipelineStage] = list(stages or [])

    def add_stage(self, stage: PipelineStage) -> None:
        self.stages.append(stage)

    async def run(self, data: Any) -> list[Any]:
        """Execute all stages in parallel, returning list of results."""
        tasks = [stage.run(data) for stage in self.stages]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def stage_metrics(self) -> dict[str, dict[str, float]]:
        return {
            s.name: {
                "calls": s.metrics.calls,
                "avg_ms": round(s.metrics.avg_ms, 3),
                "errors": s.metrics.errors,
            }
            for s in self.stages
        }
