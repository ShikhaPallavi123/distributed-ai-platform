"""Tests for deep learning architectures."""
import unittest
import numpy as np
from deep import (
    CNN, ResNet, ResidualBlock,
    RNNCell, LSTMCell, GRUCell, SequenceModel,
    TransformerEncoder, TransformerBlock, GPTModel,
    VAE, GAN, WGAN,
)


class TestCNN(unittest.TestCase):
    def test_cnn_forward(self):
        cnn = CNN(in_channels=1, num_classes=10, input_size=28)
        x = np.random.randn(2, 1, 28, 28)
        out = cnn.forward(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 10)

    def test_residual_block(self):
        rb = ResidualBlock(channels=8)
        x = np.random.randn(1, 8, 14, 14)
        out = rb.forward(x)
        self.assertEqual(out.shape, x.shape)


class TestRNN(unittest.TestCase):
    def test_rnn_cell(self):
        cell = RNNCell(input_size=10, hidden_size=20)
        x = np.random.randn(4, 10)
        h = np.zeros((4, 20))
        h_new = cell.forward(x, h)
        self.assertEqual(h_new.shape, (4, 20))

    def test_lstm_cell(self):
        cell = LSTMCell(input_size=10, hidden_size=20)
        x = np.random.randn(4, 10)
        h = np.zeros((4, 20))
        c = np.zeros((4, 20))
        h_new, c_new = cell.forward(x, h, c)
        self.assertEqual(h_new.shape, (4, 20))
        self.assertEqual(c_new.shape, (4, 20))

    def test_gru_cell(self):
        cell = GRUCell(input_size=10, hidden_size=20)
        x = np.random.randn(4, 10)
        h = np.zeros((4, 20))
        h_new = cell.forward(x, h)
        self.assertEqual(h_new.shape, (4, 20))

    def test_sequence_model(self):
        cell = LSTMCell(input_size=8, hidden_size=16)
        model = SequenceModel(cell, output_size=4, return_sequences=False)
        x = np.random.randn(2, 5, 8)
        out = model.forward(x)
        self.assertEqual(out.shape, (2, 4))


class TestTransformer(unittest.TestCase):
    def test_transformer_encoder(self):
        enc = TransformerEncoder(n_layers=2, d_model=16, n_heads=4, d_ff=32)
        x = np.random.randn(2, 5, 16)
        out = enc.forward(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_gpt_model(self):
        gpt = GPTModel(vocab_size=100, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_len=20)
        tokens = np.random.randint(0, 100, (2, 10))
        logits = gpt.forward(tokens)
        self.assertEqual(logits.shape, (2, 10, 100))


class TestVAE(unittest.TestCase):
    def test_vae_forward(self):
        vae = VAE(input_dim=784, hidden_dim=128, latent_dim=16)
        x = np.random.randn(4, 784)
        recon, mu, logvar = vae.forward(x)
        self.assertEqual(recon.shape, (4, 784))
        self.assertEqual(mu.shape, (4, 16))

    def test_vae_sample(self):
        vae = VAE(input_dim=784, hidden_dim=128, latent_dim=16)
        samples = vae.sample(3)
        self.assertEqual(samples.shape, (3, 784))

    def test_vae_loss(self):
        vae = VAE(input_dim=784, hidden_dim=128, latent_dim=16)
        x = np.random.rand(4, 784)
        recon, mu, logvar = vae.forward(x)
        recon_loss = vae.reconstruction_loss(x, recon)
        kl_loss = vae.kl_divergence(mu, logvar)
        self.assertGreater(float(recon_loss), 0)
        self.assertGreaterEqual(float(kl_loss), 0)


class TestGAN(unittest.TestCase):
    def test_gan_generate(self):
        gan = GAN(latent_dim=64, data_dim=100, hidden_dim=128)
        samples = gan.generate(4)
        self.assertEqual(samples.shape[0], 4)

    def test_gan_train_step(self):
        gan = GAN(latent_dim=64, data_dim=100, hidden_dim=128)
        real = np.random.randn(4, 100)
        d_loss, g_loss = gan.train_step(real)
        self.assertIsInstance(d_loss, float)
        self.assertIsInstance(g_loss, float)

    def test_wgan_creation(self):
        wgan = WGAN(latent_dim=64, data_dim=100, hidden_dim=128)
        samples = wgan.generate(2)
        self.assertEqual(samples.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
