"""Tests for the custom neural network framework."""
import unittest
import numpy as np
from neural import (
    Tensor, Linear, Sequential, Conv2D, BatchNorm, LayerNorm,
    Dropout, Embedding, MultiHeadAttention,
    SGD, Adam, AdamW, StepLR,
    MSELoss, CrossEntropyLoss, BCELoss,
    ReLU, Sigmoid, Softmax, GELU,
    Model, no_grad, randn, zeros, ones,
)


class TestTensor(unittest.TestCase):
    def test_creation(self):
        t = Tensor([1, 2, 3])
        self.assertEqual(t.shape, (3,))

    def test_addition(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        self.assertTrue(np.allclose(c.data, [4.0, 6.0]))

    def test_multiplication(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        self.assertTrue(np.allclose(c.data, [8.0, 15.0]))

    def test_matmul(self):
        a = Tensor(np.random.randn(2, 3), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        c = a @ b
        self.assertEqual(c.shape, (2, 4))

    def test_backward(self):
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2, dy/dx = 2x = 4
        y.backward()
        self.assertTrue(np.allclose(x.grad.data, [4.0]))

    def test_no_grad(self):
        with no_grad():
            a = Tensor([1.0], requires_grad=True)
            b = a + a
            # In no_grad context, graph shouldn't be tracked
            self.assertIsNotNone(b)

    def test_constructors(self):
        z = zeros(2, 3)
        self.assertEqual(z.shape, (2, 3))
        o = ones(2, 3)
        self.assertTrue(np.allclose(o.data, 1.0))
        r = randn(2, 3)
        self.assertEqual(r.shape, (2, 3))


class TestLayers(unittest.TestCase):
    def test_linear(self):
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(4, 10))
        y = layer(x)
        self.assertEqual(y.shape, (4, 5))

    def test_sequential(self):
        net = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        x = Tensor(np.random.randn(4, 10))
        y = net(x)
        self.assertEqual(y.shape, (4, 5))

    def test_batch_norm(self):
        bn = BatchNorm(10)
        x = Tensor(np.random.randn(8, 10))
        y = bn(x)
        self.assertEqual(y.shape, (8, 10))

    def test_layer_norm(self):
        ln = LayerNorm(10)
        x = Tensor(np.random.randn(4, 10))
        y = ln(x)
        self.assertEqual(y.shape, (4, 10))

    def test_embedding(self):
        emb = Embedding(100, 32)
        # Provide integer indices
        idx = Tensor(np.array([1, 5, 10]))
        y = emb(idx)
        self.assertEqual(y.shape, (3, 32))

    def test_dropout(self):
        dp = Dropout(0.5)
        x = Tensor(np.ones((10, 10)))
        dp.training = True
        y = dp(x)
        self.assertEqual(y.shape, (10, 10))

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        x = Tensor(np.random.randn(2, 5, 16))
        y = mha(x)
        self.assertEqual(y.shape, (2, 5, 16))


class TestOptimizers(unittest.TestCase):
    def test_sgd_step(self):
        layer = Linear(4, 1)
        opt = SGD(layer.parameters(), lr=0.01)
        x = Tensor(np.random.randn(8, 4))
        y = layer(x)
        loss = (y * y).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_adam_training(self):
        layer = Linear(4, 1)
        opt = Adam(layer.parameters(), lr=0.01)
        loss_fn = MSELoss()
        x = Tensor(np.random.randn(16, 4))
        target = Tensor(np.random.randn(16, 1))
        losses = []
        for _ in range(20):
            pred = layer(x)
            loss = loss_fn(pred, target)
            losses.append(loss.data.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        # Loss should decrease
        self.assertLess(losses[-1], losses[0])


class TestLosses(unittest.TestCase):
    def test_mse_loss(self):
        loss_fn = MSELoss()
        pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        target = Tensor(np.array([1.5, 2.5, 3.5]))
        loss = loss_fn(pred, target)
        self.assertGreater(loss.data.item(), 0)
        loss.backward()

    def test_cross_entropy(self):
        loss_fn = CrossEntropyLoss()
        logits = Tensor(np.random.randn(4, 3), requires_grad=True)
        targets = Tensor(np.array([0, 1, 2, 1]))
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.data.item(), 0)

    def test_bce_loss(self):
        loss_fn = BCELoss()
        pred = Tensor(np.array([0.7, 0.3, 0.9]), requires_grad=True)
        target = Tensor(np.array([1.0, 0.0, 1.0]))
        loss = loss_fn(pred, target)
        self.assertGreater(loss.data.item(), 0)


class TestActivations(unittest.TestCase):
    def test_relu(self):
        r = ReLU()
        x = Tensor(np.array([-1, 0, 1, 2.0]))
        y = r(x)
        self.assertTrue(np.allclose(y.data, [0, 0, 1, 2]))

    def test_sigmoid(self):
        s = Sigmoid()
        x = Tensor(np.array([0.0]))
        y = s(x)
        self.assertAlmostEqual(y.data.item(), 0.5, places=5)

    def test_softmax(self):
        sm = Softmax()
        x = Tensor(np.array([[1.0, 2.0, 3.0]]))
        y = sm(x)
        self.assertAlmostEqual(y.data.sum(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
