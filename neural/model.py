"""Model base class with save/load, parameter counting, and train/eval modes.

Provides a convenient superclass for building neural-network models on top of
the custom tensor / autograd framework.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neural.tensor import Tensor
from neural.layers import Layer


class Model(Layer):
    """Base class for all models.

    Subclass this and define ``forward()`` to build your network.
    """

    def save(self, path: str) -> None:
        """Save all parameters to a ``.npz`` file."""
        state: Dict[str, np.ndarray] = {}
        for i, p in enumerate(self.parameters()):
            state[f"param_{i}"] = p.data
        np.savez(path, **state)

    def load(self, path: str) -> None:
        """Load parameters from a ``.npz`` file previously created by :meth:`save`."""
        data = np.load(path)
        params = self.parameters()
        for i, p in enumerate(params):
            key = f"param_{i}"
            if key in data:
                loaded = data[key]
                if loaded.shape != p.data.shape:
                    raise ValueError(
                        f"Shape mismatch for param {i}: "
                        f"expected {p.data.shape}, got {loaded.shape}"
                    )
                p.data = loaded.astype(p.data.dtype)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of (trainable) parameters."""
        total = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.data.size
        return total

    def summary(self) -> str:
        """Return a human-readable model summary (layer shapes and param counts)."""
        return model_summary(self)


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------
def model_summary(model: Layer) -> str:
    """Generate a text summary of a :class:`Layer` / :class:`Model`.

    Prints every sub-layer, parameter shapes, and total parameter count.
    """
    lines: List[str] = []
    sep = "-" * 60
    lines.append(sep)
    lines.append(f"{'Layer':<30} {'Shape':<20} {'Params':>8}")
    lines.append(sep)

    total_params = 0
    trainable_params = 0

    def _walk(prefix: str, layer: Layer) -> None:
        nonlocal total_params, trainable_params
        name = prefix or type(layer).__name__
        own_params = 0
        for attr_name, attr_val in layer.__dict__.items():
            if isinstance(attr_val, Tensor) and attr_val.requires_grad:
                shape_str = str(attr_val.shape)
                n = attr_val.data.size
                lines.append(f"  {name + '.' + attr_name:<28} {shape_str:<20} {n:>8,}")
                total_params += n
                trainable_params += n
                own_params += n
            elif isinstance(attr_val, Layer):
                _walk(f"{name}.{attr_name}", attr_val)
            elif isinstance(attr_val, (list, tuple)):
                for idx, item in enumerate(attr_val):
                    if isinstance(item, Layer):
                        _walk(f"{name}[{idx}]", item)

    _walk("", model)

    lines.append(sep)
    lines.append(f"Total params:      {total_params:>12,}")
    lines.append(f"Trainable params:  {trainable_params:>12,}")
    lines.append(sep)
    return "\n".join(lines)
