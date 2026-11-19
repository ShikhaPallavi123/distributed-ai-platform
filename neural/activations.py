"""Activation functions for the neural network framework.

Each activation is a thin wrapper that operates on ``Tensor`` objects and
supports autograd through the ops defined in ``tensor.py`` and ``autograd.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from neural.tensor import Tensor
from neural.autograd import Function


# ---------------------------------------------------------------------------
# Autograd helpers for activations without a dedicated backward in autograd.py
# ---------------------------------------------------------------------------
class _LeakyReLUBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: Tensor, negative_slope: float = 0.01) -> np.ndarray:
        ctx.save_for_backward(a.data, negative_slope)
        return np.where(a.data > 0, a.data, a.data * negative_slope)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray):
        a_data, ns = ctx.saved
        return (grad * np.where(a_data > 0, 1.0, ns),)


class _GELUBackward(Function):
    """Gaussian Error Linear Unit (approximation)."""
    _SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi).astype(np.float32)
    _K = 0.044715

    @staticmethod
    def _forward(ctx: Function, a: Tensor) -> np.ndarray:
        x = a.data
        inner = _GELUBackward._SQRT_2_OVER_PI * (x + _GELUBackward._K * x ** 3)
        tanh_val = np.tanh(inner)
        ctx.save_for_backward(x, tanh_val)
        return 0.5 * x * (1.0 + tanh_val)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray):
        x, tanh_val = ctx.saved
        k = _GELUBackward._K
        s2p = _GELUBackward._SQRT_2_OVER_PI
        sech2 = 1.0 - tanh_val ** 2
        inner_deriv = s2p * (1.0 + 3.0 * k * x ** 2)
        dx = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv
        return (grad * dx,)


class _SiLUBackward(Function):
    """SiLU / Swish: x * sigmoid(x)."""
    @staticmethod
    def _forward(ctx: Function, a: Tensor) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-np.clip(a.data, -500, 500)))
        ctx.save_for_backward(a.data, sig)
        return a.data * sig

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray):
        x, sig = ctx.saved
        dx = sig * (1.0 + x * (1.0 - sig))
        return (grad * dx,)


# ---------------------------------------------------------------------------
# Public activation classes (callable as layers or functions)
# ---------------------------------------------------------------------------

class ReLU:
    """Rectified Linear Unit: max(0, x)."""
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class LeakyReLU:
    """Leaky ReLU with configurable negative slope."""
    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def __call__(self, x: Tensor) -> Tensor:
        return Function.forward_call(_LeakyReLUBackward, x, negative_slope=self.negative_slope)


class GELU:
    """Gaussian Error Linear Unit."""
    def __call__(self, x: Tensor) -> Tensor:
        return Function.forward_call(_GELUBackward, x)


class Sigmoid:
    """Logistic sigmoid: 1 / (1 + exp(-x))."""
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh:
    """Hyperbolic tangent."""
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax:
    """Softmax along a given axis (default -1)."""
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class SiLU:
    """SiLU / Swish activation: x * sigmoid(x)."""
    def __call__(self, x: Tensor) -> Tensor:
        return Function.forward_call(_SiLUBackward, x)


# Alias
Swish = SiLU
