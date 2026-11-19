"""Image generation module with diffusion models and variational autoencoders.

Implements noise scheduling, forward/reverse diffusion processes,
and a simplified VAE for latent-space generation -- all in pure numpy.
"""

from __future__ import annotations

import math

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ScheduleType(Enum):
    """Beta schedule types for the diffusion process."""
    LINEAR = "linear"
    COSINE = "cosine"


class NoiseScheduler:
    """Manages the noise schedule for diffusion models.

    Controls how much noise is added at each timestep during forward diffusion,
    and provides the corresponding coefficients for reverse diffusion.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: ScheduleType = ScheduleType.LINEAR,
    ) -> None:
        self.num_timesteps = num_timesteps
        if schedule == ScheduleType.LINEAR:
            self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        else:
            # Cosine schedule (Nichol & Dhariwal 2021)
            steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 1e-6, 0.999)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

        # Coefficients for reverse (posterior) step
        alpha_cumprod_prev = np.append(1.0, self.alpha_cumprod[:-1])
        self.posterior_variance = self.betas * (1.0 - alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_mean_coef1 = self.betas * np.sqrt(alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_mean_coef2 = (1.0 - alpha_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alpha_cumprod)

    def add_noise(self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: add noise to data at timestep t.

        Returns:
            Tuple of (noisy_sample, noise_added).
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)
        noisy = self.sqrt_alpha_cumprod[t] * x0 + self.sqrt_one_minus_alpha_cumprod[t] * noise
        return noisy.astype(np.float32), noise.astype(np.float32)

    def step(self, model_output: np.ndarray, x_t: np.ndarray, t: int) -> np.ndarray:
        """Single reverse diffusion step: denoise from t to t-1.

        Args:
            model_output: Predicted noise from the denoising network.
            x_t: Current noisy sample.
            t: Current timestep.

        Returns:
            Denoised sample at t-1.
        """
        # Predict x0
        x0_pred = (x_t - self.sqrt_one_minus_alpha_cumprod[t] * model_output) / self.sqrt_alpha_cumprod[t]
        x0_pred = np.clip(x0_pred, -1.0, 1.0)

        mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x_t
        if t > 0:
            noise = np.random.randn(*x_t.shape).astype(np.float32)
            return (mean + np.sqrt(self.posterior_variance[t]) * noise).astype(np.float32)
        return mean.astype(np.float32)


def _dense(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ W + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class SimpleDiffusion:
    """Simplified diffusion model using a dense denoising network.

    Mimics UNet behavior with skip connections using dense layers,
    suitable for small image generation without convolution dependencies.
    """

    def __init__(
        self,
        data_dim: int = 784,
        hidden_dim: int = 256,
        time_emb_dim: int = 32,
        num_timesteps: int = 200,
        schedule: ScheduleType = ScheduleType.LINEAR,
        seed: int = 42,
    ) -> None:
        self.data_dim = data_dim
        self.scheduler = NoiseScheduler(num_timesteps, schedule=schedule)
        rng = np.random.RandomState(seed)
        s = 0.02

        self.time_emb_dim = time_emb_dim
        input_dim = data_dim + time_emb_dim

        # Encoder path
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * s
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * s
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Decoder path with skip connection
        self.W3 = rng.randn(hidden_dim * 2, hidden_dim).astype(np.float32) * s
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)
        self.W4 = rng.randn(hidden_dim, data_dim).astype(np.float32) * s
        self.b4 = np.zeros(data_dim, dtype=np.float32)

    def _time_embedding(self, t: int) -> np.ndarray:
        """Sinusoidal timestep embedding."""
        half = self.time_emb_dim // 2
        freqs = np.exp(-math.log(10000.0) * np.arange(half) / half)
        emb = np.zeros(self.time_emb_dim, dtype=np.float32)
        emb[:half] = np.sin(t * freqs)
        emb[half:] = np.cos(t * freqs)
        return emb

    def _predict_noise(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """Predict the noise component using the denoising network."""
        t_emb = self._time_embedding(t)
        inp = np.concatenate([x_t.flatten(), t_emb])

        h1 = _relu(_dense(inp, self.W1, self.b1))
        h2 = _relu(_dense(h1, self.W2, self.b2))
        # Skip connection
        h_cat = np.concatenate([h2, h1])
        h3 = _relu(_dense(h_cat, self.W3, self.b3))
        return _dense(h3, self.W4, self.b4).reshape(x_t.shape)

    def forward_diffusion(self, x0: np.ndarray, num_steps: Optional[int] = None) -> List[np.ndarray]:
        """Run the full forward diffusion process, returning noisy images at each step."""
        T = num_steps or self.scheduler.num_timesteps
        trajectory = [x0.copy()]
        step_indices = np.linspace(0, self.scheduler.num_timesteps - 1, T, dtype=int)
        for t in step_indices:
            noisy, _ = self.scheduler.add_noise(x0, t)
            trajectory.append(noisy)
        return trajectory

    def generate(
        self,
        shape: Tuple[int, ...] = (28, 28),
        num_steps: int = 50,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a sample via reverse diffusion.

        Args:
            shape: Output image shape (will be flattened internally).
            num_steps: Number of denoising steps.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated sample array of the requested shape.
        """
        if seed is not None:
            np.random.seed(seed)

        x = np.random.randn(int(np.prod(shape))).astype(np.float32)
        step_indices = np.linspace(self.scheduler.num_timesteps - 1, 0, num_steps, dtype=int)

        for t in step_indices:
            noise_pred = self._predict_noise(x, int(t))
            x = self.scheduler.step(noise_pred, x, int(t))

        return x.reshape(shape)


class ImageVAE:
    """Variational Autoencoder for latent-space image generation.

    Encodes images into a low-dimensional latent space and decodes
    back, enabling generation by sampling from the latent prior.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        rng = np.random.RandomState(seed)
        s = 0.02

        # Encoder
        self.enc_W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * s
        self.enc_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.mu_W = rng.randn(hidden_dim, latent_dim).astype(np.float32) * s
        self.mu_b = np.zeros(latent_dim, dtype=np.float32)
        self.logvar_W = rng.randn(hidden_dim, latent_dim).astype(np.float32) * s
        self.logvar_b = np.zeros(latent_dim, dtype=np.float32)

        # Decoder
        self.dec_W1 = rng.randn(latent_dim, hidden_dim).astype(np.float32) * s
        self.dec_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.dec_W2 = rng.randn(hidden_dim, input_dim).astype(np.float32) * s
        self.dec_b2 = np.zeros(input_dim, dtype=np.float32)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution parameters (mu, log_var)."""
        h = _relu(_dense(x.flatten(), self.enc_W1, self.enc_b1))
        mu = _dense(h, self.mu_W, self.mu_b)
        log_var = _dense(h, self.logvar_W, self.logvar_b)
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape).astype(np.float32)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstructed output."""
        h = _relu(_dense(z, self.dec_W1, self.dec_b1))
        return _sigmoid(_dense(h, self.dec_W2, self.dec_b2))

    def reconstruct(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full encode-decode pass. Returns (reconstruction, mu, log_var)."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def generate(self, num_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Generate new samples by sampling from the latent prior N(0, I).

        Args:
            num_samples: Number of samples to generate.
            seed: Optional random seed.

        Returns:
            Array of shape (num_samples, input_dim) with generated data.
        """
        if seed is not None:
            np.random.seed(seed)
        samples = []
        for _ in range(num_samples):
            z = np.random.randn(self.latent_dim).astype(np.float32)
            samples.append(self.decode(z))
        return np.array(samples)

    @staticmethod
    def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
        """KL divergence between latent distribution and standard normal."""
        return float(-0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var)))
