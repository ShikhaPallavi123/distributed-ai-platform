"""Loss functions for the neural network framework.

All losses operate on :class:`Tensor` objects and return a scalar ``Tensor``
whose ``.backward()`` propagates gradients through the computation graph.
"""

from __future__ import annotations

import numpy as np

from neural.tensor import Tensor


class MSELoss:
    """Mean Squared Error loss: ``mean((pred - target)^2)``."""

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()


class CrossEntropyLoss:
    """Cross-entropy loss for classification (combines log-softmax + NLL).

    ``pred``  – raw logits of shape ``(N, C)``
    ``target`` – integer class labels of shape ``(N,)``
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Numerically stable log-softmax
        logits = pred.data
        labels = target.data.astype(int)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp  # (N, C)

        n = logits.shape[0]
        _nll = -log_probs[np.arange(n), labels]  # noqa: F841  (N,)

        # Wrap back into Tensor graph via softmax-based gradient
        pred.softmax(axis=-1)  # autograd-tracked
        one_hot = Tensor(np.zeros_like(logits))
        one_hot.data[np.arange(n), labels] = 1.0
        # loss = -sum(one_hot * log_probs) / N  (re-derive with tracked ops)
        loss = ((Tensor(-1.0) * (pred * one_hot).sum(axis=-1)).mean()
                + Tensor(np.float32(np.mean(np.log(np.sum(np.exp(shifted), axis=-1))))))
        return loss


class BCELoss:
    """Binary Cross-Entropy loss.

    ``pred``   – probabilities in [0, 1]
    ``target`` – binary labels (0 or 1), same shape as pred
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Clamp for numerical safety
        eps = 1e-7
        p = pred * Tensor(1.0 - 2 * eps) + Tensor(eps)  # keep in (eps, 1-eps)
        loss = Tensor(-1.0) * (target * p.log() + (Tensor(1.0) - target) * (Tensor(1.0) - p).log())
        return loss.mean()


class HuberLoss:
    """Huber / Smooth-L1 loss.

    Quadratic for small errors, linear for large ones.

    Parameters
    ----------
    delta : float – threshold between quadratic and linear regimes
    """

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        abs_diff = np.abs(diff.data)
        # Quadratic where |diff| <= delta, linear elsewhere
        quadratic = Tensor(np.float32(0.5)) * diff * diff
        linear = Tensor(np.float32(self.delta)) * (Tensor(abs_diff) - Tensor(np.float32(0.5 * self.delta)))
        mask = Tensor((abs_diff <= self.delta).astype(np.float32))
        loss = mask * quadratic + (Tensor(1.0) - mask) * linear
        return loss.mean()


class KLDivLoss:
    """Kullback-Leibler divergence loss.

    ``pred``   – log-probabilities
    ``target`` – probabilities (the reference distribution)

    Computes ``target * (log(target) - pred)`` and returns the mean.
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # target * (log(target) - pred), ignoring where target == 0
        log_target = Tensor(np.log(np.clip(target.data, 1e-12, None)))
        loss = Tensor(target.data) * (log_target - pred)
        return loss.mean()
