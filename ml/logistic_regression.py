"""Logistic Regression with binary and multiclass (one-vs-rest) support."""

import numpy as np
from typing import Optional, Literal


class LogisticRegression:
    """Logistic regression classifier with gradient descent optimization.

    Args:
        learning_rate: Step size for gradient descent.
        n_iterations: Number of training iterations.
        regularization: L2 regularization strength.
        multi_class: Strategy for multiclass - 'ovr' (one-vs-rest).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.0,
        multi_class: Literal["ovr"] = "ovr",
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.multi_class = multi_class
        self.classes_: Optional[np.ndarray] = None
        self._models: list[dict[str, np.ndarray]] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = np.float64(0.0)
        for _ in range(self.n_iterations):
            z = X @ weights + bias
            y_pred = self._sigmoid(z)
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error + self.regularization * weights)
            db = (1 / n_samples) * np.sum(error)
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        return {"weights": weights, "bias": np.array([bias])}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self._models = []

        if len(self.classes_) == 2:
            binary_y = (y == self.classes_[1]).astype(np.float64)
            self._models.append(self._fit_binary(X, binary_y))
        else:
            for cls in self.classes_:
                binary_y = (y == cls).astype(np.float64)
                self._models.append(self._fit_binary(X, binary_y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates of shape (n_samples, n_classes)."""
        X = np.asarray(X, dtype=np.float64)
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if len(self.classes_) == 2:
            m = self._models[0]
            p1 = self._sigmoid(X @ m["weights"] + m["bias"][0])
            return np.column_stack([1 - p1, p1])
        else:
            scores = np.column_stack([
                self._sigmoid(X @ m["weights"] + m["bias"][0]) for m in self._models
            ])
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]  # type: ignore[index]
