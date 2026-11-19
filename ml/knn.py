"""K-Nearest Neighbors for classification and regression."""

import numpy as np
from typing import Optional, Literal


class KNearestNeighbors:
    """K-Nearest Neighbors classifier/regressor.

    Args:
        k: Number of neighbors to consider.
        metric: Distance metric - 'euclidean' or 'manhattan'.
        mode: 'classification' for majority vote, 'regression' for mean.
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        mode: Literal["classification", "regression"] = "classification",
    ) -> None:
        self.k = k
        self.metric = metric
        self.mode = mode
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        """Store the training data."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y).ravel()
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between X and training data.

        Returns array of shape (n_test, n_train).
        """
        if self.metric == "euclidean":
            # Efficient vectorized computation
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            TT = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # type: ignore[union-attr]
            dists = np.sqrt(np.maximum(XX + TT - 2.0 * X @ self.X_train.T, 0))  # type: ignore[union-attr]
        else:  # manhattan
            dists = np.sum(
                np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]),  # type: ignore[index]
                axis=2,
            )
        return dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for input samples."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        dists = self._compute_distances(X)
        k_indices = np.argpartition(dists, self.k, axis=1)[:, : self.k]

        predictions = []
        for i in range(len(X)):
            neighbor_labels = self.y_train[k_indices[i]]
            if self.mode == "classification":
                values, counts = np.unique(neighbor_labels, return_counts=True)
                predictions.append(values[np.argmax(counts)])
            else:
                predictions.append(np.mean(neighbor_labels))
        return np.array(predictions)
