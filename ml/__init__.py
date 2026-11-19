"""Classical machine learning algorithms."""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .knn import KNearestNeighbors
from .svm import SVM
from .ensemble import RandomForest, GradientBoosting, BaggingClassifier

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "DecisionTree",
    "KNearestNeighbors",
    "SVM",
    "RandomForest",
    "GradientBoosting",
    "BaggingClassifier",
]
