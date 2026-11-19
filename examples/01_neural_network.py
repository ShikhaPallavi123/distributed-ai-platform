"""
Example 1: Train a neural network from scratch using NeuroFlux.

This demonstrates:
- Creating tensors with autograd
- Building a simple neural network
- Training with backpropagation
- Using the Adam optimizer
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from neural.tensor import Tensor
from neural.layers import Linear
from neural.activations import ReLU, Sigmoid
from neural.losses import MSELoss
from neural.optimizers import Adam


def main():
    print("=" * 50)
    print("NeuroFlux — Neural Network Training Example")
    print("=" * 50)

    # Generate synthetic data: learn XOR
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    x_tensor = Tensor(X)
    y_tensor = Tensor(y)

    # Build network
    layer1 = Linear(2, 8)
    relu = ReLU()
    layer2 = Linear(8, 1)
    sigmoid = Sigmoid()

    # Optimizer
    params = layer1.parameters() + layer2.parameters()
    optimizer = Adam(params, lr=0.05)
    loss_fn = MSELoss()

    print("\nTraining on XOR problem...")
    print("-" * 40)

    # Training loop
    for epoch in range(500):
        # Forward pass
        h = relu(layer1(x_tensor))
        pred = sigmoid(layer2(h))

        # Loss
        loss = loss_fn(pred, y_tensor)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:4d} | Loss: {loss.data.item():.6f}")

    # Test
    print("\nResults:")
    print("-" * 40)
    h = relu(layer1(x_tensor))
    pred = sigmoid(layer2(h))
    for i in range(4):
        actual = y[i, 0]
        predicted = pred.data[i, 0]
        print(f"  Input: {X[i]} → Predicted: {predicted:.4f} (actual: {actual})")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
