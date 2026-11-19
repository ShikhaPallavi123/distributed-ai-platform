"""Neural network layers built on the custom Tensor / autograd engine.

Includes fully-connected, convolutional, normalisation, dropout, embedding,
multi-head attention, and a Sequential container.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from neural.tensor import Tensor
from neural.autograd import Function


# ===================================================================
# Base class
# ===================================================================
class Layer:
    """Abstract base for all layers."""

    _training: bool = True

    def parameters(self) -> List[Tensor]:
        """Return all trainable :class:`Tensor` objects in this layer."""
        params: List[Tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Layer):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Layer):
                        params.extend(item.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True) -> "Layer":
        self._training = mode
        for v in self.__dict__.values():
            if isinstance(v, Layer):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Layer):
                        item.train(mode)
        return self

    def eval(self) -> "Layer":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


# ===================================================================
# Linear (fully connected)
# ===================================================================
class Linear(Layer):
    """Fully-connected layer with optional bias and Xavier initialisation.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Xavier / Glorot uniform
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True,
        )
        self.bias_tensor: Optional[Tensor] = None
        if bias:
            self.bias_tensor = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias_tensor is not None:
            out = out + self.bias_tensor
        return out

    def parameters(self) -> List[Tensor]:
        ps = [self.weight]
        if self.bias_tensor is not None:
            ps.append(self.bias_tensor)
        return ps


# ===================================================================
# Conv2D with im2col
# ===================================================================
def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> Tuple[np.ndarray, Tuple]:
    """Extract image patches as columns for efficient convolution."""
    n, c, h, w = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    _, _, hp, wp = x.shape
    out_h = (hp - kh) // stride + 1
    out_w = (wp - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.reshape(n, c * kh * kw, out_h * out_w)
    return cols, (n, c, h, w, out_h, out_w)


class _Conv2DBackward(Function):
    @staticmethod
    def _forward(ctx: Function, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                 stride: int = 1, padding: int = 0) -> np.ndarray:
        kh, kw = weight.data.shape[2], weight.data.shape[3]
        cols, info = _im2col(x.data, kh, kw, stride, padding)
        n, c_in, h, w, out_h, out_w = info
        c_out = weight.data.shape[0]
        w_col = weight.data.reshape(c_out, -1)  # (c_out, c_in*kh*kw)
        out = w_col @ cols  # (n, c_out, out_h*out_w)
        out = out.reshape(n, c_out, out_h, out_w)
        if bias is not None:
            out = out + bias.data.reshape(1, -1, 1, 1)
        ctx.save_for_backward(x.data, weight.data, cols, info, stride, padding)
        return out

    @staticmethod
    def _backward(ctx: Function, grad: np.ndarray):
        x_data, w_data, cols, info, stride, padding = ctx.saved
        n, c_in, h, w, out_h, out_w = info
        c_out = w_data.shape[0]
        # grad: (n, c_out, out_h, out_w)
        grad_flat = grad.reshape(n, c_out, -1)  # (n, c_out, out_h*out_w)
        w_col = w_data.reshape(c_out, -1)

        # weight grad
        dw_col = np.zeros_like(w_col)
        for i in range(n):
            dw_col += grad_flat[i] @ cols[i].T
        dw = dw_col.reshape(w_data.shape)

        # bias grad
        db = grad.sum(axis=(0, 2, 3))

        # input grad (simplified – accumulate via col2im)
        dx_cols = np.zeros_like(cols)
        for i in range(n):
            dx_cols[i] = w_col.T @ grad_flat[i]
        # col2im
        kh, kw = w_data.shape[2], w_data.shape[3]
        hp = h + 2 * padding
        wp = w + 2 * padding
        dx_padded = np.zeros((n, c_in, hp, wp), dtype=grad.dtype)
        col_reshaped = dx_cols.reshape(n, c_in, kh, kw, out_h, out_w)
        for i in range(kh):
            i_max = i + stride * out_h
            for j in range(kw):
                j_max = j + stride * out_w
                dx_padded[:, :, i:i_max:stride, j:j_max:stride] += col_reshaped[:, :, i, j, :, :]
        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded
        return (dx, dw, db)


class Conv2D(Layer):
    """2-D convolution using im2col.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int
    stride : int
    padding : int
    bias : bool
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        k = kernel_size
        limit = math.sqrt(6.0 / (in_channels * k * k + out_channels * k * k))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_channels, in_channels, k, k)).astype(np.float32),
            requires_grad=True,
        )
        self.bias_tensor: Optional[Tensor] = None
        if bias:
            self.bias_tensor = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return Function.forward_call(
            _Conv2DBackward, x, self.weight, bias=self.bias_tensor,
            stride=self.stride, padding=self.padding,
        )

    def parameters(self) -> List[Tensor]:
        ps = [self.weight]
        if self.bias_tensor is not None:
            ps.append(self.bias_tensor)
        return ps


# ===================================================================
# BatchNorm
# ===================================================================
class BatchNorm(Layer):
    """Batch normalisation for 2-D inputs ``(N, C)`` or 4-D ``(N, C, H, W)``.

    Tracks running mean/var for inference.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self._training:
            if x.data.ndim == 4:
                axis = (0, 2, 3)
            else:
                axis = (0,)
            mean = np.mean(x.data, axis=axis)
            var = np.var(x.data, axis=axis)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        if x.data.ndim == 4:
            shape = (1, -1, 1, 1)
        else:
            shape = (1, -1)
        mean_t = Tensor(mean.reshape(shape))
        Tensor(var.reshape(shape))
        gamma_r = self.gamma.reshape(shape)
        beta_r = self.beta.reshape(shape)

        x_norm = (x - mean_t) * Tensor(1.0 / np.sqrt(var.reshape(shape) + self.eps))
        return x_norm * gamma_r + beta_r

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


# ===================================================================
# LayerNorm
# ===================================================================
class LayerNorm(Layer):
    """Layer normalisation over the last dimension(s)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        diff = x - mean
        var = (diff * diff).mean(axis=-1, keepdims=True)
        x_norm = diff * Tensor(1.0 / np.sqrt(var.data + self.eps))
        return x_norm * self.gamma + self.beta

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


# ===================================================================
# Dropout
# ===================================================================
class Dropout(Layer):
    """Inverted dropout (scales at train time)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * Tensor(mask)

    def parameters(self) -> List[Tensor]:
        return []


# ===================================================================
# Embedding
# ===================================================================
class Embedding(Layer):
    """Learnable lookup table of fixed-size vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )

    def forward(self, indices: Tensor) -> Tensor:
        """``indices`` should contain integer data."""
        idx = indices.data.astype(int)
        return Tensor(self.weight.data[idx], requires_grad=self.weight.requires_grad)

    def parameters(self) -> List[Tensor]:
        return [self.weight]


# ===================================================================
# Multi-Head Attention
# ===================================================================
class MultiHeadAttention(Layer):
    """Scaled dot-product multi-head attention (transformer style).

    Parameters
    ----------
    d_model : int   – input / output dimensionality
    num_heads : int – number of attention heads
    dropout : float – attention dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, query: Tensor, key: Optional[Tensor] = None,
                value: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        batch, seq_len, _ = query.shape

        # Project and reshape to (batch, heads, seq, d_k)
        q = self.q_proj(query).reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k_len = key.shape[1]
        k = self.k_proj(key).reshape(batch, k_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.v_proj(value).reshape(batch, k_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = q @ k.transpose(0, 1, 3, 2) * Tensor(np.float32(self.scale))
        if mask is not None:
            scores = scores + mask  # mask should contain -inf for masked positions

        attn = scores.softmax(axis=-1)
        attn = self.dropout(attn)
        context = attn @ v  # (batch, heads, seq, d_k)

        # Concat heads
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() + self.k_proj.parameters()
                + self.v_proj.parameters() + self.out_proj.parameters())


# ===================================================================
# Sequential container
# ===================================================================
class Sequential(Layer):
    """Container that chains layers in order."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        self.layers: List[Layer] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)
