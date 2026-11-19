"""Decision Tree classifier with Gini impurity and entropy criteria."""

import numpy as np
from typing import Optional, Literal


class _Node:
    """Internal tree node."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(
        self,
        feature: int = -1,
        threshold: float = 0.0,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        value: Optional[np.ndarray] = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class distribution at leaf

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """CART decision tree classifier.

    Args:
        criterion: Split quality measure - 'gini' or 'entropy'.
        max_depth: Maximum depth of the tree. None for unlimited.
        min_samples_split: Minimum samples required to split a node.
        max_features: Number of features to consider per split. None = all.
    """

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root: Optional[_Node] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _impurity(self, y: np.ndarray) -> float:
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / len(y)
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:  # entropy
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feature, best_threshold = 0, 0.0
        parent_impurity = self._impurity(y)

        features = np.arange(n_features)
        if self.max_features and self.max_features < n_features:
            features = np.random.choice(n_features, self.max_features, replace=False)

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                gain = parent_impurity - (
                    n_l / n_samples * self._impurity(y[left_mask])
                    + n_r / n_samples * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feat)
                    best_threshold = float(thresh)
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)
        distribution = np.bincount(y, minlength=self.n_classes_).astype(np.float64)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return _Node(value=distribution)

        feat, thresh, gain = self._best_split(X, y)
        if gain <= 0:
            return _Node(value=distribution)

        left_mask = X[:, feat] <= thresh
        self.feature_importances_[feat] += gain * n_samples  # type: ignore[index]
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Build the decision tree from training data."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_sample(self, x: np.ndarray, node: _Node) -> np.ndarray:
        if node.is_leaf:
            return node.value  # type: ignore[return-value]
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)  # type: ignore[arg-type]
        return self._predict_sample(x, node.right)  # type: ignore[arg-type]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        X = np.asarray(X, dtype=np.float64)
        if self.root is None:
            raise RuntimeError("Model has not been fitted yet.")
        distributions = np.array([self._predict_sample(x, self.root) for x in X])
        return np.argmax(distributions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        X = np.asarray(X, dtype=np.float64)
        if self.root is None:
            raise RuntimeError("Model has not been fitted yet.")
        distributions = np.array([self._predict_sample(x, self.root) for x in X])
        row_sums = distributions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return distributions / row_sums
