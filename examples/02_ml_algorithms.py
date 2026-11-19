"""
Example 2: Classical ML algorithms from scratch.

Demonstrates:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from ml.linear_regression import LinearRegression
from ml.logistic_regression import LogisticRegression
from ml.decision_tree import DecisionTree
from ml.knn import KNearestNeighbors
from ml.ensemble import RandomForest


def main():
    print("=" * 50)
    print("NeuroFlux — Classical ML Algorithms Example")
    print("=" * 50)
    np.random.seed(42)

    # --- Linear Regression ---
    print("\n📈 Linear Regression")
    print("-" * 40)
    X = np.random.randn(100, 1) * 2
    y = 3 * X.squeeze() + 7 + np.random.randn(100) * 0.5

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    print(f"  R² Score: {r2:.4f}")
    print(f"  Coefficients: {model.weights}")

    # --- Logistic Regression ---
    print("\n🎯 Logistic Regression")
    print("-" * 40)
    X_cls = np.random.randn(200, 2)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

    lr = LogisticRegression(lr=0.1, max_iter=200)
    lr.fit(X_cls, y_cls)
    acc = np.mean(lr.predict(X_cls) == y_cls)
    print(f"  Accuracy: {acc:.4f}")

    # --- Decision Tree ---
    print("\n🌳 Decision Tree")
    print("-" * 40)
    dt = DecisionTree(max_depth=5)
    dt.fit(X_cls, y_cls)
    acc = np.mean(dt.predict(X_cls) == y_cls)
    print(f"  Accuracy: {acc:.4f}")

    # --- KNN ---
    print("\n📍 K-Nearest Neighbors")
    print("-" * 40)
    knn = KNearestNeighbors(k=5)
    knn.fit(X_cls, y_cls)
    acc = np.mean(knn.predict(X_cls) == y_cls)
    print(f"  Accuracy: {acc:.4f}")

    # --- Random Forest ---
    print("\n🌲 Random Forest")
    print("-" * 40)
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X_cls, y_cls)
    acc = np.mean(rf.predict(X_cls) == y_cls)
    print(f"  Accuracy: {acc:.4f}")

    print("\n✓ All ML algorithms working!")


if __name__ == "__main__":
    main()
