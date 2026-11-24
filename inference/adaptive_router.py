"""Quality-constrained routing using explicitly supplied calibration evidence.

This is a baseline policy, not a guarantee of per-request quality or novelty.
Only candidates for the same task and evaluation protocol may be compared.
"""
from dataclasses import dataclass
import math
import time


class NoEligibleModel(RuntimeError):
    """No model satisfies the specified constraints or is available."""


@dataclass(frozen=True)
class Candidate:
    name: str
    quality: float
    sample_count: int
    latency_ms: float
    predict: object

    def __post_init__(self):
        if not self.name or not callable(self.predict):
            raise ValueError("Candidate requires a name and callable")
        if not math.isfinite(self.quality) or not 0 <= self.quality <= 1:
            raise ValueError("quality must be between zero and one")
        if not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency must be finite and nonnegative")

    @property
    def quality_lower_bound(self):
        """Wilson 95% lower bound for binomial calibration accuracy."""
        z = 1.96
        n = self.sample_count
        p = self.quality
        denominator = 1 + z * z / n
        center = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (center - margin) / denominator)


def route(candidates, text, min_quality=0.5, max_latency_ms=None,
          recoverable_errors=()):
    """Select lowest predicted latency among quality-eligible candidates.

    Fallback attempts share the observed time budget. Calibration quality and
    latency estimates describe past samples, never individual guarantees.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("text must contain 1–8000 characters")
    if not isinstance(min_quality, (float, int)) or not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between zero and one")
    if max_latency_ms is not None and (
        not isinstance(max_latency_ms, (int, float)) or not math.isfinite(max_latency_ms)
        or max_latency_ms <= 0
    ):
        raise ValueError("max_latency_ms must be positive and finite")
    eligible = sorted(
        (candidate for candidate in candidates if candidate.quality_lower_bound >= min_quality),
        key=lambda candidate: (candidate.latency_ms, -candidate.quality_lower_bound, candidate.name),
    )
    started = time.perf_counter()
    attempts = []
    for candidate in eligible:
        elapsed = (time.perf_counter() - started) * 1000
        if max_latency_ms is not None and elapsed + candidate.latency_ms > max_latency_ms:
            continue
        try:
            result = candidate.predict(text)
        except recoverable_errors:
            attempts.append({"model": candidate.name, "status": "unavailable"})
            continue
        actual = (time.perf_counter() - started) * 1000
        attempts.append({"model": candidate.name, "status": "success"})
        return {
            "selected_model": candidate.name, "output": result,
            "calibration_accuracy": candidate.quality,
            "calibration_sample_count": candidate.sample_count,
            "calibration_wilson_lower_95": candidate.quality_lower_bound,
            "estimated_latency_ms": candidate.latency_ms,
            "actual_latency_ms": actual,
            "latency_budget_exceeded": max_latency_ms is not None and actual > max_latency_ms,
            "attempts": attempts,
            "policy": "quality_lower_bound_then_latency",
        }
    raise NoEligibleModel("No eligible model completed within the routing policy")
