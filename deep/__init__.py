"""Deep learning architectures built on numpy."""

from .cnn import Conv2D, MaxPool2D, Dense, Flatten, ReLU, CNN, ResidualBlock, ResNet
from .rnn import RNNCell, LSTMCell, GRUCell, SequenceModel
from .transformer import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderBlock,
    Transformer,
    GPTModel,
)
from .vae import VAE
from .gan import Generator, Discriminator, GAN, WGAN

__all__ = [
    # CNN
    "Conv2D", "MaxPool2D", "Dense", "Flatten", "ReLU",
    "CNN", "ResidualBlock", "ResNet",
    # RNN
    "RNNCell", "LSTMCell", "GRUCell", "SequenceModel",
    # Transformer
    "PositionalEncoding", "MultiHeadAttention",
    "TransformerBlock", "TransformerEncoder",
    "TransformerDecoder", "TransformerDecoderBlock",
    "Transformer", "GPTModel",
    # VAE
    "VAE",
    # GAN
    "Generator", "Discriminator", "GAN", "WGAN",
]
