"""Convolutional Neural Network components built on numpy."""

import numpy as np
from typing import Optional


class Conv2D:
    """2D convolution layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output filters.
        kernel_size: Size of the convolution kernel (square).
        stride: Convolution stride.
        padding: Zero-padding added to input borders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, channels, height, width)."""
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        batch, _, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        output = np.zeros((batch, self.out_channels, out_h, out_w))

        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        receptive = x[b, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                        output[b, oc, i, j] = np.sum(receptive * self.weights[oc]) + self.bias[oc]
        return output


class MaxPool2D:
    """2D max pooling layer."""

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None) -> None:
        self.pool_size = pool_size
        self.stride = stride or pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, channels, h, w = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((batch, channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * self.stride, j * self.stride
                output[:, :, i, j] = np.max(
                    x[:, :, h_s:h_s + self.pool_size, w_s:w_s + self.pool_size],
                    axis=(2, 3),
                )
        return output


class ReLU:
    """ReLU activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Flatten:
    """Flatten spatial dimensions."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)


class Dense:
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias


class CNN:
    """Convolutional Neural Network: Conv2D -> ReLU -> Pool -> Conv2D -> ReLU -> Pool -> Flatten -> Dense.

    Args:
        in_channels: Input image channels (e.g. 1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        input_size: Spatial dimension of input (assumes square images).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, input_size: int = 28) -> None:
        self.layers = [
            Conv2D(in_channels, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(2),
            Conv2D(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
        ]
        flat_size = 32 * (input_size // 4) ** 2
        self.fc = Dense(flat_size, num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network. x shape: (batch, channels, H, W)."""
        for layer in self.layers:
            x = layer.forward(x)
        return self.fc.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)


class ResidualBlock:
    """Residual block with skip connection.

    Two conv layers with a shortcut: output = ReLU(conv2(ReLU(conv1(x))) + x).
    """

    def __init__(self, channels: int) -> None:
        self.conv1 = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.relu = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x
        out = self.relu.forward(self.conv1.forward(x))
        out = self.conv2.forward(out)
        return self.relu.forward(out + identity)


class ResNet:
    """Simplified ResNet model using residual blocks.

    Args:
        in_channels: Input channels.
        num_classes: Number of output classes.
        num_blocks: Number of residual blocks.
        base_channels: Channels after the initial convolution.
        input_size: Spatial dimension of input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        num_blocks: int = 3,
        base_channels: int = 16,
        input_size: int = 28,
    ) -> None:
        self.stem = Conv2D(in_channels, base_channels, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.blocks = [ResidualBlock(base_channels) for _ in range(num_blocks)]
        self.pool = MaxPool2D(pool_size=input_size)  # global average-ish pool
        self.flatten = Flatten()
        self.fc = Dense(base_channels, num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.relu.forward(self.stem.forward(x))
        for block in self.blocks:
            x = block.forward(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        return self.fc.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=1)
