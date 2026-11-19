"""Optimisers and learning-rate schedulers for the neural network framework.

Implements SGD (with momentum and weight decay), Adam, AdamW, and two
common LR schedulers (StepLR, CosineAnnealingLR).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from neural.tensor import Tensor


# ===================================================================
# Base optimiser
# ===================================================================
class Optimizer:
    """Base class for all optimisers."""

    def __init__(self, params: List[Tensor], lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        """Perform a single optimisation step."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset all parameter gradients to ``None``."""
        for p in self.params:
            p.zero_grad()


# ===================================================================
# SGD
# ===================================================================
class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and weight decay.

    Parameters
    ----------
    params : list of Tensor
    lr : float
    momentum : float
    weight_decay : float – L2 regularisation coefficient
    """

    def __init__(self, params: List[Tensor], lr: float = 1e-2,
                 momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.copy()
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0:
                pid = id(p)
                if pid not in self._velocity:
                    self._velocity[pid] = np.zeros_like(g)
                v = self._velocity[pid]
                v[:] = self.momentum * v + g
                p.data -= self.lr * v
            else:
                p.data -= self.lr * g


# ===================================================================
# Adam
# ===================================================================
class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    params : list of Tensor
    lr : float
    beta1 : float – exponential decay rate for the first moment
    beta2 : float – exponential decay rate for the second moment
    eps : float – term added for numerical stability
    weight_decay : float – L2 regularisation (added to grad)
    """

    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._state: Dict[int, dict] = {}
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                self._state[pid] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }
            st = self._state[pid]
            g = p.grad.copy()
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data

            st["m"] = self.beta1 * st["m"] + (1 - self.beta1) * g
            st["v"] = self.beta2 * st["v"] + (1 - self.beta2) * g ** 2

            m_hat = st["m"] / (1 - self.beta1 ** self._t)
            v_hat = st["v"] / (1 - self.beta2 ** self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===================================================================
# AdamW (decoupled weight decay)
# ===================================================================
class AdamW(Optimizer):
    """AdamW – Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Weight decay is applied directly to the parameters, not to the gradient.
    """

    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 1e-2) -> None:
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._state: Dict[int, dict] = {}
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                self._state[pid] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }
            st = self._state[pid]
            g = p.grad

            st["m"] = self.beta1 * st["m"] + (1 - self.beta1) * g
            st["v"] = self.beta2 * st["v"] + (1 - self.beta2) * g ** 2

            m_hat = st["m"] / (1 - self.beta1 ** self._t)
            v_hat = st["v"] / (1 - self.beta2 ** self._t)

            # Decoupled weight decay
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)


# ===================================================================
# Learning-rate schedulers
# ===================================================================
class _LRScheduler:
    """Base learning-rate scheduler."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepLR(_LRScheduler):
    """Decay the LR by *gamma* every *step_size* epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int = 30,
                 gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self._step_count // self.step_size))


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing schedule (Loshchilov & Hutter, 2017).

    Parameters
    ----------
    optimizer : Optimizer
    T_max : int – maximum number of steps (half-period)
    eta_min : float – minimum learning rate
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        t = self._step_count % self.T_max
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
