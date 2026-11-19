"""Tests for classical ML algorithms."""
import unittest
import numpy as np
from ml import (
    LinearRegression, LogisticRegression, DecisionTree,
    KNearestNeighbors, SVM, RandomForest, GradientBoosting,
)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = self.X @ np.array([2.0, -1.0, 0.5]) + 0.1 * np.random.randn(100)

    def test_closed_form_fit(self):
        model = LinearRegression(method="closed_form")
        model.fit(self.X, self.y)
        pred = model.predict(self.X)
        self.assertEqual(pred.shape, (100,))

    def test_gradient_descent_fit(self):
        model = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iterations=500)
        model.fit(self.X, self.y)
        pred = model.predict(self.X)
        mse = np.mean((pred - self.y) ** 2)
        self.assertLess(mse, 1.0)

    def test_r2_score(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        r2 = model.r2_score(self.X, self.y)
        self.assertGreater(r2, 0.9)


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

    def test_binary_classification(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(self.X, self.y)
        pred = model.predict(self.X)
        accuracy = np.mean(pred == self.y)
        self.assertGreater(accuracy, 0.75)

    def test_predict_proba(self):
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 4)
        self.y = (self.X[:, 0] > 0).astype(int)

    def test_fit_predict(self):
        dt = DecisionTree(max_depth=5)
        dt.fit(self.X, self.y)
        pred = dt.predict(self.X)
        accuracy = np.mean(pred == self.y)
        self.assertGreater(accuracy, 0.85)

    def test_gini_vs_entropy(self):
        dt_gini = DecisionTree(criterion="gini", max_depth=3)
        dt_entropy = DecisionTree(criterion="entropy", max_depth=3)
        dt_gini.fit(self.X, self.y)
        dt_entropy.fit(self.X, self.y)
        self.assertEqual(dt_gini.predict(self.X).shape, (100,))
        self.assertEqual(dt_entropy.predict(self.X).shape, (100,))


class TestKNN(unittest.TestCase):
    def test_classification(self):
        np.random.seed(42)
        X = np.random.randn(60, 2)
        y = (X[:, 0] > 0).astype(int)
        knn = KNearestNeighbors(k=5)
        knn.fit(X, y)
        pred = knn.predict(X[:10])
        self.assertEqual(pred.shape, (10,))

    def test_different_k(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        for k in [1, 3, 7]:
            knn = KNearestNeighbors(k=k)
            knn.fit(X, y)
            pred = knn.predict(X[:5])
            self.assertEqual(len(pred), 5)


class TestSVM(unittest.TestCase):
    def test_linear_svm(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(30, 2) + 2, np.random.randn(30, 2) - 2])
        y = np.array([1] * 30 + [-1] * 30)
        svm = SVM(C=1.0, max_passes=20)
        svm.fit(X, y)
        pred = svm.predict(X)
        accuracy = np.mean(pred == y)
        self.assertGreater(accuracy, 0.7)


class TestEnsemble(unittest.TestCase):
    def test_random_forest(self):
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        rf = RandomForest(n_estimators=10, max_depth=5)
        rf.fit(X, y)
        pred = rf.predict(X)
        accuracy = np.mean(pred == y)
        self.assertGreater(accuracy, 0.75)

    def test_gradient_boosting(self):
        np.random.seed(42)
        X = np.random.randn(80, 3)
        y = (X[:, 0] > 0).astype(int)
        gb = GradientBoosting(n_estimators=10, max_depth=3)
        gb.fit(X, y)
        pred = gb.predict(X)
        self.assertEqual(pred.shape, (80,))


if __name__ == "__main__":
    unittest.main()
