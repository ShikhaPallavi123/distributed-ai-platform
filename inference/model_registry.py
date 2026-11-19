"""Model versioning, metadata, and A/B testing support."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ModelStatus(Enum):
    REGISTERED = auto()
    ACTIVE = auto()
    DEPRECATED = auto()
    ARCHIVED = auto()


@dataclass
class ModelArtifact:
    """Metadata for a single model version."""
    model_name: str
    version: str
    framework: str = "pytorch"
    input_shape: list[int] = field(default_factory=lambda: [1, 3, 224, 224])
    output_shape: list[int] = field(default_factory=lambda: [1, 1000])
    accuracy: float = 0.0
    size_mb: float = 0.0
    status: ModelStatus = ModelStatus.REGISTERED
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class TrafficSplit:
    """Defines traffic splitting for A/B testing between model versions."""
    splits: dict[str, float] = field(default_factory=dict)  # version -> weight

    def select(self) -> str:
        """Randomly select a version according to the weights."""
        if not self.splits:
            raise ValueError("no versions in traffic split")
        versions = list(self.splits.keys())
        weights = [self.splits[v] for v in versions]
        total = sum(weights)
        if total <= 0:
            return versions[0]
        normalised = [w / total for w in weights]
        return random.choices(versions, weights=normalised, k=1)[0]


class ModelRegistry:
    """Registry for model versions with promotion, rollback, and A/B testing."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, ModelArtifact]] = {}  # name -> {ver -> art}
        self._active: dict[str, str] = {}  # name -> active version
        self._traffic: dict[str, TrafficSplit] = {}

    # -- CRUD -------------------------------------------------------------

    def register(self, artifact: ModelArtifact) -> ModelArtifact:
        """Register a new model version."""
        bucket = self._models.setdefault(artifact.model_name, {})
        if artifact.version in bucket:
            raise ValueError(
                f"{artifact.model_name} v{artifact.version} already registered"
            )
        bucket[artifact.version] = artifact
        # First version becomes active by default
        if artifact.model_name not in self._active:
            self.promote(artifact.model_name, artifact.version)
        return artifact

    def get(self, name: str, version: str | None = None) -> ModelArtifact | None:
        bucket = self._models.get(name, {})
        if version:
            return bucket.get(version)
        active_ver = self._active.get(name)
        return bucket.get(active_ver) if active_ver else None

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def list_versions(self, name: str) -> list[str]:
        return sorted(self._models.get(name, {}).keys())

    def delete(self, name: str, version: str) -> bool:
        bucket = self._models.get(name, {})
        if version not in bucket:
            return False
        del bucket[version]
        if self._active.get(name) == version:
            remaining = sorted(bucket.keys())
            if remaining:
                self._active[name] = remaining[-1]
                bucket[remaining[-1]].status = ModelStatus.ACTIVE
            else:
                del self._active[name]
                del self._models[name]
        return True

    # -- version management -----------------------------------------------

    def promote(self, name: str, version: str) -> None:
        """Make *version* the active serving version."""
        bucket = self._models.get(name, {})
        if version not in bucket:
            raise KeyError(f"{name} v{version} not found")
        # Demote current active
        cur = self._active.get(name)
        if cur and cur in bucket:
            bucket[cur].status = ModelStatus.REGISTERED
        bucket[version].status = ModelStatus.ACTIVE
        self._active[name] = version

    def rollback(self, name: str) -> str | None:
        """Rollback to the previous version (by registration time)."""
        bucket = self._models.get(name, {})
        cur = self._active.get(name)
        candidates = sorted(
            bucket.values(), key=lambda a: a.created_at, reverse=True
        )
        for art in candidates:
            if art.version != cur:
                self.promote(name, art.version)
                return art.version
        return None

    # -- A/B testing ------------------------------------------------------

    def set_traffic_split(self, name: str, splits: dict[str, float]) -> None:
        """Configure traffic splitting between versions of *name*."""
        bucket = self._models.get(name, {})
        for v in splits:
            if v not in bucket:
                raise KeyError(f"{name} v{v} not found")
        self._traffic[name] = TrafficSplit(splits=splits)

    def resolve_version(self, name: str) -> str:
        """Return the version to serve, respecting any traffic split."""
        split = self._traffic.get(name)
        if split and split.splits:
            return split.select()
        ver = self._active.get(name)
        if ver is None:
            raise KeyError(f"no active version for {name!r}")
        return ver

    def clear_traffic_split(self, name: str) -> None:
        self._traffic.pop(name, None)

    # -- query ------------------------------------------------------------

    def find_by_tag(self, tag: str) -> list[ModelArtifact]:
        results: list[ModelArtifact] = []
        for bucket in self._models.values():
            for art in bucket.values():
                if tag in art.tags:
                    results.append(art)
        return results

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "active": self._active.get(name),
                "versions": list(bucket.keys()),
                "traffic_split": (
                    self._traffic[name].splits if name in self._traffic else None
                ),
            }
            for name, bucket in self._models.items()
        }
