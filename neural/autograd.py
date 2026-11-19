"""Automatic differentiation engine for the neural network framework.

Implements reverse-mode autodiff with a computation graph, topological sorting,
and gradient accumulation. All gradient functions follow PyTorch-style conventions.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural.tensor import Tensor


# ---------------------------------------------------------------------------
# Global flag: when True, no graph is recorded
# ---------------------------------------------------------------------------
_grad_enabled = threading.local()


def is_grad_enabled() -> bool:
    return getattr(_grad_enabled, "value", True)


class no_grad:
    """Context manager that disables gradient tracking."""

    def __enter__(self) -> "no_grad":
        self._prev = is_grad_enabled()
        _grad_enabled.value = False
        return self

    def __exit__(self, *args: Any) -> None:
        _grad_enabled.value = self._prev


# ---------------------------------------------------------------------------
# Base Function class
# ---------------------------------------------------------------------------
class Function:
    """Base class for all differentiable operations.

    Every subclass must implement ``forward`` and ``backward``.
    ``forward`` is a classmethod-style helper that also wires up the graph.
    """

    def __init__(self) -> None:
        self.saved: Tuple[Any, ...] = ()
        self.parents: List[Tuple[Tensor, int]] = []

    def save_for_backward(self, *args: Any) -> None:
        self.saved = args

    @staticmethod
    def forward_call(cls: type, *inputs: "Tensor", **kwargs: Any) -> "Tensor":
        from neural.tensor import Tensor

        ctx = cls()
        raw = cls._forward(ctx, *inputs, **kwargs)
        out = Tensor(raw, requires_grad=any(t.requires_grad for t in inputs if isinstance(t, Tensor)))
        if out.requires_grad and is_grad_enabled():
            out._grad_fn = ctx
            for i, inp in enumerate(inputs):
                if isinstance(inp, Tensor) and inp.requires_grad:
                    ctx.parents.append((inp, i))
        return out

    # Subclasses override these two ---
    @staticmethod
    def _forward(ctx: "Function", *args: Any, **kw: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _backward(ctx: "Function", grad: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Topological sort for backward pass
# ---------------------------------------------------------------------------
def _toposort(root: "Tensor") -> List["Tensor"]:
    order: List["Tensor"] = []
    visited: set = set()

    def dfs(t: "Tensor") -> None:
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        if t._grad_fn is not None:
            for parent, _ in t._grad_fn.parents:
                dfs(parent)
        order.append(t)

    dfs(root)
    return order


def backward(root: "Tensor", grad: Optional[np.ndarray] = None) -> None:
    """Run backward pass from *root* through the computation graph."""
    if grad is None:
        if root.data.size == 1:
            grad = np.ones_like(root.data)
        else:
            raise RuntimeError("grad must be specified for non-scalar outputs")

    root.grad = grad
    order = list(reversed(_toposort(root)))

    for t in order:
        if t._grad_fn is None:
            continue
        grads = t._grad_fn._backward(t._grad_fn, t.grad)
        for (parent, idx), g in zip(t._grad_fn.parents, _match_grads(t._grad_fn, grads)):
            if g is None:
                continue
            # Reduce broadcasted dims
            g = _unbroadcast(g, parent.data.shape)
            if parent.grad is None:
                parent.grad = g
            else:
                parent.grad = parent.grad + g


def _match_grads(ctx: Function, grads: Tuple[Optional[np.ndarray], ...]) -> List[Optional[np.ndarray]]:
    """Return one gradient per parent, handling mismatches in tuple length."""
    result: list = []
    parent_indices = [idx for _, idx in ctx.parents]
    for _, idx in ctx.parents:
        if idx < len(grads):
            result.append(grads[idx])
        else:
            result.append(None)
    return result


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out dimensions that were broadcast during the forward pass."""
    if g.shape == shape:
        return g
    # Pad shape on the left with 1s so ndims match
    ndims_added = g.ndim - len(shape)
    for i in range(ndims_added):
        g = g.sum(axis=0)
    for i, (gs, ss) in enumerate(zip(g.shape, shape)):
        if ss == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# ===================================================================
# Concrete gradient functions
# ===================================================================

class AddBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", b: "Tensor") -> np.ndarray:
        return a.data + b.data

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (grad, grad)


class MulBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", b: "Tensor") -> np.ndarray:
        ctx.save_for_backward(a.data, b.data)
        return a.data * b.data

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_data, b_data = ctx.saved
        return (grad * b_data, grad * a_data)


class MatMulBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", b: "Tensor") -> np.ndarray:
        ctx.save_for_backward(a.data, b.data)
        return a.data @ b.data

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a_data, b_data = ctx.saved
        if a_data.ndim == 1 and b_data.ndim == 1:
            return (grad * b_data, grad * a_data)
        ga = grad @ np.swapaxes(b_data, -2, -1)
        gb = np.swapaxes(a_data, -2, -1) @ grad
        return (ga, gb)


class ReLUBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        ctx.save_for_backward(a.data)
        return np.maximum(a.data, 0)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (a_data,) = ctx.saved
        return (grad * (a_data > 0).astype(grad.dtype),)


class SigmoidBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -500, 500)))
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (s,) = ctx.saved
        return (grad * s * (1 - s),)


class SoftmaxBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", axis: int = -1) -> np.ndarray:
        shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        s = exp / np.sum(exp, axis=axis, keepdims=True)
        ctx.save_for_backward(s, axis)
        return s

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        s, axis = ctx.saved
        dot = np.sum(grad * s, axis=axis, keepdims=True)
        return (s * (grad - dot),)


class SumBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", axis: Optional[int] = None,
                 keepdims: bool = False) -> np.ndarray:
        ctx.save_for_backward(a.data.shape, axis, keepdims)
        return np.sum(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        shape, axis, keepdims = ctx.saved
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis)
        if not keepdims and axis is None:
            grad = grad.reshape((1,) * len(shape))
        return (np.broadcast_to(grad, shape).copy(),)


class MeanBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", axis: Optional[int] = None,
                 keepdims: bool = False) -> np.ndarray:
        ctx.save_for_backward(a.data.shape, axis, keepdims)
        return np.mean(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        shape, axis, keepdims = ctx.saved
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis)
        if not keepdims and axis is None:
            grad = grad.reshape((1,) * len(shape))
        n = shape[axis] if axis is not None else np.prod(shape)
        return (np.broadcast_to(grad / n, shape).copy(),)


class ReshapeBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", shape: Tuple[int, ...]) -> np.ndarray:
        ctx.save_for_backward(a.data.shape)
        return a.data.reshape(shape)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (orig_shape,) = ctx.saved
        return (grad.reshape(orig_shape),)


class TransposeBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        ctx.save_for_backward(axes)
        return np.transpose(a.data, axes)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (axes,) = ctx.saved
        if axes is None:
            return (np.transpose(grad),)
        inv = [0] * len(axes)
        for i, a in enumerate(axes):
            inv[a] = i
        return (np.transpose(grad, inv),)


class NegBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        return -a.data

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (-grad,)


class ExpBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        out = np.exp(np.clip(a.data, -500, 500))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (out,) = ctx.saved
        return (grad * out,)


class LogBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        ctx.save_for_backward(a.data)
        return np.log(np.clip(a.data, 1e-12, None))

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (a_data,) = ctx.saved
        return (grad / np.clip(a_data, 1e-12, None),)


class TanhBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor") -> np.ndarray:
        out = np.tanh(a.data)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        (out,) = ctx.saved
        return (grad * (1 - out ** 2),)


class PowBackward(Function):
    @staticmethod
    def _forward(ctx: Function, a: "Tensor", exponent: float) -> np.ndarray:
        ctx.save_for_backward(a.data, exponent)
        return np.power(a.data, exponent)

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray) -> Tuple[np.ndarray]:
        a_data, exponent = ctx.saved
        return (grad * exponent * np.power(a_data, exponent - 1),)
