"""neural – A from-scratch neural network framework built on numpy.

Provides tensors with autograd, layers, activations, optimisers, losses,
and a model base class, comparable to a minimal PyTorch.
"""

# --- Tensor and autograd primitives ---
from neural.tensor import Tensor, tensor, zeros, ones, randn, rand, from_numpy
from neural.autograd import no_grad

# --- Layers ---
from neural.layers import (
    Layer,
    Linear,
    Conv2D,
    BatchNorm,
    LayerNorm,
    Dropout,
    Embedding,
    MultiHeadAttention,
    Sequential,
)

# --- Activations ---
from neural.activations import (
    ReLU,
    LeakyReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    SiLU,
    Swish,
)

# --- Optimisers ---
from neural.optimizers import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    StepLR,
    CosineAnnealingLR,
)

# --- Losses ---
from neural.losses import (
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    HuberLoss,
    KLDivLoss,
)

# --- Model ---
from neural.model import Model, model_summary

__all__ = [
    # tensor
    "Tensor", "tensor", "zeros", "ones", "randn", "rand", "from_numpy",
    # autograd
    "no_grad",
    # layers
    "Layer", "Linear", "Conv2D", "BatchNorm", "LayerNorm", "Dropout",
    "Embedding", "MultiHeadAttention", "Sequential",
    # activations
    "ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "SiLU", "Swish",
    # optimizers
    "Optimizer", "SGD", "Adam", "AdamW", "StepLR", "CosineAnnealingLR",
    # losses
    "MSELoss", "CrossEntropyLoss", "BCELoss", "HuberLoss", "KLDivLoss",
    # model
    "Model", "model_summary",
]
