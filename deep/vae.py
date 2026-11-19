"""Variational Autoencoder with ELBO loss."""

import numpy as np


class _Linear:
    """Simple linear layer for VAE internals."""

    def __init__(self, in_features: int, out_features: int) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class VAE:
    """Variational Autoencoder.

    Architecture: input -> encoder -> (mu, logvar) -> reparameterize -> decoder -> output
    Loss: ELBO = reconstruction_loss + KL_divergence

    Args:
        input_dim: Dimensionality of input data.
        hidden_dim: Hidden layer size for encoder/decoder.
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 32) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.enc_fc1 = _Linear(input_dim, hidden_dim)
        self.enc_mu = _Linear(hidden_dim, latent_dim)
        self.enc_logvar = _Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.dec_fc1 = _Linear(latent_dim, hidden_dim)
        self.dec_out = _Linear(hidden_dim, input_dim)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input of shape (batch, input_dim).

        Returns:
            Tuple of (mu, logvar), each of shape (batch, latent_dim).
        """
        h = self._relu(self.enc_fc1.forward(x))
        mu = self.enc_mu.forward(h)
        logvar = self.enc_logvar.forward(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * epsilon.

        Allows gradients to flow through the sampling step.
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector of shape (batch, latent_dim).

        Returns:
            Reconstruction of shape (batch, input_dim), values in [0, 1].
        """
        h = self._relu(self.dec_fc1.forward(z))
        return self._sigmoid(self.dec_out.forward(h))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass.

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def reconstruction_loss(x: np.ndarray, x_recon: np.ndarray) -> float:
        """Binary cross-entropy reconstruction loss."""
        x_recon = np.clip(x_recon, 1e-8, 1 - 1e-8)
        bce = -np.mean(x * np.log(x_recon) + (1 - x) * np.log(1 - x_recon))
        return float(bce)

    @staticmethod
    def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> float:
        """KL divergence between N(mu, sigma^2) and N(0, 1)."""
        return float(-0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar)))

    def elbo_loss(self, x: np.ndarray) -> tuple[float, float, float]:
        """Compute ELBO loss components.

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_divergence).
        """
        x_recon, mu, logvar = self.forward(x)
        recon = self.reconstruction_loss(x, x_recon)
        kl = self.kl_divergence(mu, logvar)
        return recon + kl, recon, kl

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate new samples by decoding random latent vectors."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)
