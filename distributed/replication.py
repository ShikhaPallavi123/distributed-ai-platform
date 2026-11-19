"""Model replication and sharding across cluster nodes."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class ShardingStrategy(Enum):
    """How model weights are distributed across nodes."""
    DATA_PARALLEL = auto()      # full copy on each node
    MODEL_PARALLEL = auto()     # layers split across nodes
    PIPELINE_PARALLEL = auto()  # stages split across nodes


@dataclass
class ShardInfo:
    """Metadata about one shard of a model."""
    shard_id: str
    model_name: str
    node_id: str
    layer_range: tuple[int, int]  # (start, end) layer indices
    size_bytes: int = 0
    checksum: str = ""
    loaded: bool = False


@dataclass
class Checkpoint:
    """A snapshot of replicated model state."""
    checkpoint_id: str
    model_name: str
    timestamp: float = field(default_factory=time.time)
    shard_checksums: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelReplicator:
    """Distributes model weights across cluster nodes.

    Supports data-parallel, model-parallel, and pipeline-parallel sharding,
    plus checkpointing and partial model loading.
    """

    def __init__(self) -> None:
        self._shards: dict[str, ShardInfo] = {}
        self._checkpoints: dict[str, Checkpoint] = {}
        self._node_shards: dict[str, list[str]] = {}  # node_id -> shard_ids

    def shard_model(
        self,
        model_name: str,
        weights: np.ndarray,
        node_ids: list[str],
        strategy: ShardingStrategy = ShardingStrategy.DATA_PARALLEL,
    ) -> list[ShardInfo]:
        """Split model weights into shards assigned to nodes."""
        if not node_ids:
            raise ValueError("need at least one node")

        shards: list[ShardInfo] = []
        n_nodes = len(node_ids)
        total_layers = weights.shape[0] if weights.ndim > 0 else 1

        if strategy == ShardingStrategy.DATA_PARALLEL:
            # Full copy on each node
            for nid in node_ids:
                shard = self._make_shard(
                    model_name, nid, (0, total_layers), weights
                )
                shards.append(shard)

        elif strategy == ShardingStrategy.MODEL_PARALLEL:
            # Split layers evenly
            chunk = max(1, total_layers // n_nodes)
            for i, nid in enumerate(node_ids):
                start = i * chunk
                end = total_layers if i == n_nodes - 1 else (i + 1) * chunk
                shard_weights = weights[start:end] if weights.ndim > 0 else weights
                shard = self._make_shard(
                    model_name, nid, (start, end), shard_weights
                )
                shards.append(shard)

        elif strategy == ShardingStrategy.PIPELINE_PARALLEL:
            # Same split as model-parallel but tagged for pipeline use
            chunk = max(1, total_layers // n_nodes)
            for i, nid in enumerate(node_ids):
                start = i * chunk
                end = total_layers if i == n_nodes - 1 else (i + 1) * chunk
                shard_weights = weights[start:end] if weights.ndim > 0 else weights
                shard = self._make_shard(
                    model_name, nid, (start, end), shard_weights
                )
                shards.append(shard)

        for s in shards:
            self._shards[s.shard_id] = s
            self._node_shards.setdefault(s.node_id, []).append(s.shard_id)

        return shards

    def get_shards_for_node(self, node_id: str) -> list[ShardInfo]:
        ids = self._node_shards.get(node_id, [])
        return [self._shards[sid] for sid in ids if sid in self._shards]

    def get_shards_for_model(self, model_name: str) -> list[ShardInfo]:
        return [s for s in self._shards.values() if s.model_name == model_name]

    # -- checkpointing ----------------------------------------------------

    def create_checkpoint(
        self, model_name: str, metadata: dict[str, Any] | None = None
    ) -> Checkpoint:
        """Create a checkpoint of all shards for a model."""
        shards = self.get_shards_for_model(model_name)
        cp = Checkpoint(
            checkpoint_id=f"cp-{model_name}-{int(time.time())}",
            model_name=model_name,
            shard_checksums={s.shard_id: s.checksum for s in shards},
            metadata=metadata or {},
        )
        self._checkpoints[cp.checkpoint_id] = cp
        return cp

    def list_checkpoints(self, model_name: str) -> list[Checkpoint]:
        return [
            cp for cp in self._checkpoints.values()
            if cp.model_name == model_name
        ]

    # -- partial loading --------------------------------------------------

    def load_partial(
        self,
        model_name: str,
        layer_range: tuple[int, int],
        weights: np.ndarray,
        node_id: str,
    ) -> ShardInfo:
        """Load a subset of layers onto a specific node."""
        shard = self._make_shard(model_name, node_id, layer_range, weights)
        shard.loaded = True
        self._shards[shard.shard_id] = shard
        self._node_shards.setdefault(node_id, []).append(shard.shard_id)
        return shard

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_shard(
        model_name: str,
        node_id: str,
        layer_range: tuple[int, int],
        weights: np.ndarray,
    ) -> ShardInfo:
        raw = weights.tobytes()
        checksum = hashlib.md5(raw).hexdigest()
        shard_id = f"{model_name}-{node_id}-{layer_range[0]}-{layer_range[1]}"
        return ShardInfo(
            shard_id=shard_id,
            model_name=model_name,
            node_id=node_id,
            layer_range=layer_range,
            size_bytes=len(raw),
            checksum=checksum,
        )
