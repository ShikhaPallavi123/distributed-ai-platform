"""Custom Tensor class with autograd support.

Wraps numpy arrays and records a computation graph so that calling
``.backward()`` performs reverse-mode automatic differentiation,
similar to ``torch.Tensor``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from neural.autograd import (
    Function, AddBackward, MulBackward, MatMulBackward, NegBackward,
    SumBackward, MeanBackward, ReshapeBackward, TransposeBackward,
    ReLUBackward, SigmoidBackward, SoftmaxBackward, TanhBackward,
    ExpBackward, LogBackward, PowBackward,
    backward as _run_backward,
)

# Type alias for things convertible to a Tensor
_ArrayLike = Union["Tensor", np.ndarray, float, int, list]


class Tensor:
    """A multi-dimensional array with automatic differentiation.

    Parameters
    ----------
    data : array-like
        The raw numeric data (converted to ``np.ndarray`` internally).
    requires_grad : bool
        Whether to track operations for gradient computation.
    dtype : numpy dtype, optional
        Force a specific dtype; defaults to float32.
    """

    __slots__ = ("data", "grad", "requires_grad", "_grad_fn")

    def __init__(
        self,
        data: _ArrayLike,
        requires_grad: bool = False,
        dtype: Any = None,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype or np.float32)
        elif dtype is not None:
            data = data.astype(dtype)
        elif data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float32)

        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad
        self._grad_fn: Optional[Function] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array (detached)."""
        return self.data.copy()

    def detach(self) -> "Tensor":
        """Return a new Tensor sharing the same data but detached from the graph."""
        t = Tensor(self.data, requires_grad=False)
        return t

    def zero_grad(self) -> None:
        """Reset the gradient to ``None``."""
        self.grad = None

    def item(self) -> float:
        """Return the scalar value."""
        return float(self.data.ravel()[0])

    def clone(self) -> "Tensor":
        """Return a copy of this tensor (still in the graph if grad is on)."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Compute gradients via reverse-mode autodiff."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        _run_backward(self, grad)

    # ------------------------------------------------------------------
    # Arithmetic helpers (dispatch to autograd Functions)
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_tensor(x: _ArrayLike) -> "Tensor":
        if isinstance(x, Tensor):
            return x
        return Tensor(x)

    def __add__(self, other: _ArrayLike) -> "Tensor":
        return Function.forward_call(AddBackward, self, Tensor._ensure_tensor(other))

    def __radd__(self, other: _ArrayLike) -> "Tensor":
        return Function.forward_call(AddBackward, Tensor._ensure_tensor(other), self)

    def __neg__(self) -> "Tensor":
        return Function.forward_call(NegBackward, self)

    def __sub__(self, other: _ArrayLike) -> "Tensor":
        return self + (-Tensor._ensure_tensor(other))

    def __rsub__(self, other: _ArrayLike) -> "Tensor":
        return Tensor._ensure_tensor(other) + (-self)

    def __mul__(self, other: _ArrayLike) -> "Tensor":
        return Function.forward_call(MulBackward, self, Tensor._ensure_tensor(other))

    def __rmul__(self, other: _ArrayLike) -> "Tensor":
        return Function.forward_call(MulBackward, Tensor._ensure_tensor(other), self)

    def __truediv__(self, other: _ArrayLike) -> "Tensor":
        other_t = Tensor._ensure_tensor(other)
        return self * Function.forward_call(PowBackward, other_t, -1.0)

    def __matmul__(self, other: _ArrayLike) -> "Tensor":
        return Function.forward_call(MatMulBackward, self, Tensor._ensure_tensor(other))

    def __pow__(self, exponent: float) -> "Tensor":
        return Function.forward_call(PowBackward, self, exponent)

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """Sum elements along *axis*."""
        return Function.forward_call(SumBackward, self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """Mean of elements along *axis*."""
        return Function.forward_call(MeanBackward, self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int) -> "Tensor":
        """Return a tensor with a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Function.forward_call(ReshapeBackward, self, shape=shape)

    def transpose(self, *axes: int) -> "Tensor":
        """Transpose dimensions."""
        ax: Optional[Tuple[int, ...]] = axes if axes else None
        return Function.forward_call(TransposeBackward, self, axes=ax)

    def relu(self) -> "Tensor":
        return Function.forward_call(ReLUBackward, self)

    def sigmoid(self) -> "Tensor":
        return Function.forward_call(SigmoidBackward, self)

    def tanh(self) -> "Tensor":
        return Function.forward_call(TanhBackward, self)

    def softmax(self, axis: int = -1) -> "Tensor":
        return Function.forward_call(SoftmaxBackward, self, axis=axis)

    def exp(self) -> "Tensor":
        return Function.forward_call(ExpBackward, self)

    def log(self) -> "Tensor":
        return Function.forward_call(LogBackward, self)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_info})"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, key: Any) -> "Tensor":
        """Basic indexing (no gradient support — treat as detach)."""
        return Tensor(self.data[key], requires_grad=False)


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------
def tensor(data: _ArrayLike, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Create a Tensor (convenience wrapper)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def rand(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    return Tensor(arr, requires_grad=requires_grad)


# ===================================================================
# Self-test / demo
# ===================================================================
if __name__ == "__main__":
    print("=== Tensor autograd self-test ===\n")

    # --- Test 1: simple scalar backward ---
    a = Tensor(np.array([2.0]), requires_grad=True)
    b = Tensor(np.array([3.0]), requires_grad=True)
    c = a * b + a  # c = a*b + a = 2*3 + 2 = 8
    c.backward()
    print(f"c = a*b + a = {c.item():.1f}  (expected 8.0)")
    print(f"dc/da = {a.grad.item():.1f}  (expected 4.0: b+1)")
    print(f"dc/db = {b.grad.item():.1f}  (expected 2.0: a)")
    assert np.isclose(c.item(), 8.0)
    assert np.isclose(a.grad.item(), 4.0)
    assert np.isclose(b.grad.item(), 2.0)
    print("  PASSED\n")

    # --- Test 2: matmul backward ---
    x = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    w = Tensor(np.random.randn(3, 2).astype(np.float32), requires_grad=True)
    y = x @ w                     # (4, 2)
    loss = y.sum()
    loss.backward()
    print(f"matmul output shape: {y.shape}  grads: x.grad={x.grad.shape}, w.grad={w.grad.shape}")
    assert x.grad.shape == (4, 3)
    assert w.grad.shape == (3, 2)
    print("  PASSED\n")

    # --- Test 3: small network forward + backward ---
    np.random.seed(42)
    X = Tensor(np.random.randn(8, 4).astype(np.float32), requires_grad=False)
    W1 = Tensor(np.random.randn(4, 16).astype(np.float32) * 0.1, requires_grad=True)
    b1 = Tensor(np.zeros(16, dtype=np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(16, 1).astype(np.float32) * 0.1, requires_grad=True)
    b2 = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)

    # Forward pass
    h = (X @ W1 + b1).relu()      # hidden layer
    out = (h @ W2 + b2).sigmoid()  # output
    loss = ((out - Tensor(np.ones((8, 1)))) ** 2).mean()

    loss.backward()
    print(f"Network loss: {loss.item():.4f}")
    print(f"W1 grad norm: {np.linalg.norm(W1.grad):.4f}")
    print(f"W2 grad norm: {np.linalg.norm(W2.grad):.4f}")
    assert W1.grad is not None and W2.grad is not None
    print("  PASSED\n")

    # --- Test 4: no_grad context ---
    from neural.autograd import no_grad as _no_grad
    with _no_grad():
        z = a * b
        assert z._grad_fn is None, "Graph should not be recorded inside no_grad"
    print("no_grad context: PASSED\n")

    print("All self-tests passed.")
