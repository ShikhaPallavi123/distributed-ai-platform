"""Ensemble methods: Random Forest, Gradient Boosting, Bagging."""

import numpy as np
from typing import Optional, Literal
from .decision_tree import DecisionTree


class RandomForest:
    """Random Forest classifier using bagged decision trees.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        max_features: Features considered per split. 'sqrt' or int.
        criterion: Split criterion for each tree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: Literal["sqrt"] | int | None = "sqrt",
        criterion: Literal["gini", "entropy"] = "gini",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.trees: list[DecisionTree] = []
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Fit the random forest on bootstrap samples."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))

        if self.max_features == "sqrt":
            mf = int(np.sqrt(n_features))
        elif self.max_features is None:
            mf = n_features
        else:
            mf = self.max_features

        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=mf,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by majority vote across all trees."""
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote per sample
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            result[i] = values[np.argmax(counts)]
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class probabilities across all trees."""
        probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(probas, axis=0)


class GradientBoosting:
    """Gradient Boosting regressor using decision tree stumps.

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage rate applied to each tree's prediction.
        max_depth: Maximum depth of each tree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: list[DecisionTree] = []
        self.initial_prediction: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        """Fit by sequentially adding trees to minimize residuals."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.initial_prediction = float(np.mean(y))
        residuals = y - self.initial_prediction
        self.trees = []

        for _ in range(self.n_estimators):
            # Bin residuals into classes for the tree
            median = np.median(residuals)
            pseudo_labels = (residuals > median).astype(int)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, pseudo_labels)
            # Use leaf mean residual as prediction
            predictions = self._tree_predict_values(tree, X, residuals)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)
        return self

    def _tree_predict_values(
        self, tree: DecisionTree, X: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """Get leaf-mean residual predictions from a fitted tree."""
        # Get leaf assignments
        from .decision_tree import _Node

        def _get_leaf_id(x: np.ndarray, node: _Node, path: int = 0) -> int:
            if node.is_leaf:
                return path
            if x[node.feature] <= node.threshold:
                return _get_leaf_id(x, node.left, 2 * path + 1)  # type: ignore[arg-type]
            return _get_leaf_id(x, node.right, 2 * path + 2)  # type: ignore[arg-type]

        leaf_ids = np.array([_get_leaf_id(x, tree.root) for x in X])  # type: ignore[arg-type]
        predictions = np.zeros(len(X))
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            predictions[mask] = np.mean(residuals[mask])
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        X = np.asarray(X, dtype=np.float64)
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            # Approximate: use tree class prediction scaled
            proba = tree.predict_proba(X)
            y_pred += self.learning_rate * (proba[:, -1] - 0.5) * 2
        return y_pred


class BaggingClassifier:
    """Bootstrap aggregating (bagging) meta-classifier.

    Args:
        base_estimator_params: Keyword args passed to DecisionTree.
        n_estimators: Number of base estimators.
        max_samples: Fraction of samples to draw for each base estimator.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        **base_estimator_params: int | str | None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.base_params = base_estimator_params
        self.estimators: list[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """Fit each base estimator on a bootstrap sample."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = X.shape[0]
        sample_size = max(1, int(n_samples * self.max_samples))

        self.estimators = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, sample_size, replace=True)
            est = DecisionTree(**self.base_params)  # type: ignore[arg-type]
            est.fit(X[indices], y[indices])
            self.estimators.append(est)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by majority vote."""
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.array([est.predict(X) for est in self.estimators])
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            result[i] = values[np.argmax(counts)]
        return result
