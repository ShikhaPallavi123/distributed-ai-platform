"""Linear Regression with closed-form and gradient descent solvers."""

import numpy as np
from typing import Optional, Literal


class LinearRegression:
    """Linear regression model supporting OLS closed-form and gradient descent.

    Args:
        method: Solver method - 'closed_form' (normal equation) or 'gradient_descent'.
        learning_rate: Step size for gradient descent.
        n_iterations: Number of gradient descent iterations.
        regularization: L2 regularization strength (ridge penalty). 0 = no regularization.
    """

    def __init__(
        self,
        method: Literal["closed_form", "gradient_descent"] = "closed_form",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.0,
    ) -> None:
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if self.method == "closed_form":
            X_b = np.column_stack([np.ones(n_samples), X])
            reg_matrix = self.regularization * np.eye(X_b.shape[1])
            reg_matrix[0, 0] = 0  # don't regularize bias
            theta = np.linalg.solve(X_b.T @ X_b + reg_matrix, X_b.T @ y)
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            self.loss_history = []
            for _ in range(self.n_iterations):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y
                self.loss_history.append(float(np.mean(error ** 2)))
                dw = (1 / n_samples) * (X.T @ error + self.regularization * self.weights)
                db = (1 / n_samples) * np.sum(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for input features."""
        X = np.asarray(X, dtype=np.float64)
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return X @ self.weights + self.bias

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error on the given data."""
        y = np.asarray(y, dtype=np.float64).ravel()
        return float(np.mean((self.predict(X) - y) ** 2))

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute coefficient of determination (R^2)."""
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - ss_res / ss_tot)
