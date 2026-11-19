"""Generative Adversarial Networks: GAN and WGAN."""

import numpy as np
from typing import Optional


class _Linear:
    """Linear layer with weight initialization."""

    def __init__(self, in_features: int, out_features: int) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class Generator:
    """Generator network that maps latent vectors to data space.

    Args:
        latent_dim: Dimensionality of the noise input.
        hidden_dim: Hidden layer size.
        output_dim: Dimensionality of generated output.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, output_dim: int = 784) -> None:
        self.latent_dim = latent_dim
        self.fc1 = _Linear(latent_dim, hidden_dim)
        self.fc2 = _Linear(hidden_dim, hidden_dim)
        self.fc3 = _Linear(hidden_dim, output_dim)

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generate data from noise. z: (batch, latent_dim)."""
        h = self._leaky_relu(self.fc1.forward(z))
        h = self._leaky_relu(self.fc2.forward(h))
        return self._tanh(self.fc3.forward(h))

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from random noise."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.forward(z)

    def parameters(self) -> list[np.ndarray]:
        """Return all trainable parameters."""
        return [self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]


class Discriminator:
    """Discriminator network that classifies real vs. fake.

    Args:
        input_dim: Dimensionality of input data.
        hidden_dim: Hidden layer size.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256) -> None:
        self.fc1 = _Linear(input_dim, hidden_dim)
        self.fc2 = _Linear(hidden_dim, hidden_dim)
        self.fc3 = _Linear(hidden_dim, 1)

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Classify input as real or fake. Returns probability in [0, 1]."""
        h = self._leaky_relu(self.fc1.forward(x))
        h = self._leaky_relu(self.fc2.forward(h))
        return self._sigmoid(self.fc3.forward(h))

    def forward_raw(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits without sigmoid (for WGAN)."""
        h = self._leaky_relu(self.fc1.forward(x))
        h = self._leaky_relu(self.fc2.forward(h))
        return self.fc3.forward(h)

    def parameters(self) -> list[np.ndarray]:
        return [self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]


class GAN:
    """Standard GAN with alternating generator/discriminator training.

    Args:
        latent_dim: Noise dimension for the generator.
        data_dim: Dimensionality of real data.
        hidden_dim: Hidden size for both networks.
        learning_rate: SGD learning rate.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        data_dim: int = 784,
        hidden_dim: int = 256,
        learning_rate: float = 0.0002,
    ) -> None:
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.generator = Generator(latent_dim, hidden_dim, data_dim)
        self.discriminator = Discriminator(data_dim, hidden_dim)

    @staticmethod
    def _bce_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        p = np.clip(predictions, 1e-8, 1 - 1e-8)
        return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))

    def discriminator_loss(self, real_data: np.ndarray, fake_data: np.ndarray) -> float:
        """Compute discriminator loss on real and fake data."""
        real_preds = self.discriminator.forward(real_data)
        fake_preds = self.discriminator.forward(fake_data)
        real_loss = self._bce_loss(real_preds, np.ones_like(real_preds))
        fake_loss = self._bce_loss(fake_preds, np.zeros_like(fake_preds))
        return (real_loss + fake_loss) / 2

    def generator_loss(self, fake_data: np.ndarray) -> float:
        """Compute generator loss (fool the discriminator)."""
        fake_preds = self.discriminator.forward(fake_data)
        return self._bce_loss(fake_preds, np.ones_like(fake_preds))

    def train_step(
        self, real_data: np.ndarray, lr: Optional[float] = None
    ) -> tuple[float, float]:
        """One training step with numerical gradient estimation.

        Args:
            real_data: Batch of real data, shape (batch, data_dim).
            lr: Override learning rate.

        Returns:
            Tuple of (discriminator_loss, generator_loss).
        """
        lr = lr or self.learning_rate
        batch_size = real_data.shape[0]
        eps = 1e-5

        # Generate fake data
        z = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(z)

        # Update discriminator via numerical gradients
        d_loss = self.discriminator_loss(real_data, fake_data)
        for param in self.discriminator.parameters():
            grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            # Sample a subset for efficiency
            indices = np.random.choice(param.size, min(param.size, 50), replace=False)
            flat = param.ravel()
            for idx in indices:
                orig = flat[idx]
                flat[idx] = orig + eps
                loss_plus = self.discriminator_loss(real_data, fake_data)
                flat[idx] = orig - eps
                loss_minus = self.discriminator_loss(real_data, fake_data)
                flat[idx] = orig
                multi_idx = np.unravel_index(idx, param.shape)
                grad[multi_idx] = (loss_plus - loss_minus) / (2 * eps)
            param -= lr * grad

        # Update generator
        z = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(z)
        g_loss = self.generator_loss(fake_data)
        for param in self.generator.parameters():
            grad = np.zeros_like(param)
            indices = np.random.choice(param.size, min(param.size, 50), replace=False)
            flat = param.ravel()
            for idx in indices:
                orig = flat[idx]
                flat[idx] = orig + eps
                z_tmp = np.random.randn(batch_size, self.latent_dim)
                fake_tmp = self.generator.forward(z_tmp)
                loss_plus = self.generator_loss(fake_tmp)
                flat[idx] = orig - eps
                fake_tmp = self.generator.forward(z_tmp)
                loss_minus = self.generator_loss(fake_tmp)
                flat[idx] = orig
                multi_idx = np.unravel_index(idx, param.shape)
                grad[multi_idx] = (loss_plus - loss_minus) / (2 * eps)
            param -= lr * grad

        return d_loss, g_loss

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake samples."""
        return self.generator.sample(n_samples)


class WGAN:
    """Wasserstein GAN with weight clipping.

    Uses Wasserstein distance (Earth Mover's distance) instead of JS divergence
    for more stable training.

    Args:
        latent_dim: Noise dimension.
        data_dim: Data dimensionality.
        hidden_dim: Hidden layer size.
        learning_rate: Learning rate for updates.
        clip_value: Weight clipping range for the critic.
        n_critic: Number of critic updates per generator update.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        data_dim: int = 784,
        hidden_dim: int = 256,
        learning_rate: float = 0.00005,
        clip_value: float = 0.01,
        n_critic: int = 5,
    ) -> None:
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.generator = Generator(latent_dim, hidden_dim, data_dim)
        self.critic = Discriminator(data_dim, hidden_dim)  # used without sigmoid

    def wasserstein_distance(self, real_data: np.ndarray, fake_data: np.ndarray) -> float:
        """Compute estimated Wasserstein distance."""
        real_scores = self.critic.forward_raw(real_data)
        fake_scores = self.critic.forward_raw(fake_data)
        return float(np.mean(real_scores) - np.mean(fake_scores))

    def critic_loss(self, real_data: np.ndarray, fake_data: np.ndarray) -> float:
        """Critic loss (negative Wasserstein distance, to minimize)."""
        return -self.wasserstein_distance(real_data, fake_data)

    def generator_loss(self, fake_data: np.ndarray) -> float:
        """Generator loss (negative critic score on fakes)."""
        return float(-np.mean(self.critic.forward_raw(fake_data)))

    def clip_critic_weights(self) -> None:
        """Clip critic weights to enforce Lipschitz constraint."""
        for param in self.critic.parameters():
            np.clip(param, -self.clip_value, self.clip_value, out=param)

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake samples."""
        return self.generator.sample(n_samples)
