"""Support Vector Machine with simplified SMO algorithm."""

import numpy as np
from typing import Optional


class SVM:
    """Linear SVM classifier using a simplified SMO algorithm.

    Args:
        C: Regularization parameter (penalty for misclassification).
        tol: Tolerance for KKT condition violation.
        max_passes: Maximum passes over the dataset without alpha changes.
        learning_rate: Step size for subgradient fallback.
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-3,
        max_passes: int = 20,
        learning_rate: float = 0.001,
    ) -> None:
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.learning_rate = learning_rate
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.alphas: Optional[np.ndarray] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_vector_indices_: Optional[np.ndarray] = None

    def _smo_simplified(self, X: np.ndarray, y: np.ndarray) -> None:
        """Simplified SMO algorithm (Platt, 1998 simplified version)."""
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.bias = 0.0
        K = X @ X.T  # linear kernel

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n_samples):
                E_i = float(np.sum(self.alphas * y * K[i]) + self.bias) - y[i]
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (
                    y[i] * E_i > self.tol and self.alphas[i] > 0
                ):
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    E_j = float(np.sum(self.alphas * y * K[j]) + self.bias) - y[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    if y[i] != y[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    b1 = self.bias - E_i - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.bias - E_j - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.bias = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1 + b2) / 2
                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """Fit the SVM to training data. Labels should be +1/-1."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # Convert 0/1 labels to -1/+1 if needed
        unique = np.unique(y)
        if set(unique) == {0, 1} or set(unique) == {0.0, 1.0}:
            y = 2 * y - 1

        self._smo_simplified(X, y)

        # Extract weights from support vectors
        self.weights = np.sum((self.alphas * y)[:, np.newaxis] * X, axis=0)
        sv_mask = self.alphas > 1e-5
        self.support_vector_indices_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance to the decision boundary."""
        X = np.asarray(X, dtype=np.float64)
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (+1 or -1)."""
        return np.sign(self.decision_function(X))

    def hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean hinge loss."""
        y = np.asarray(y, dtype=np.float64).ravel()
        margins = y * self.decision_function(X)
        return float(np.mean(np.maximum(0, 1 - margins)))
