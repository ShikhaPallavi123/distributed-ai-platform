"""Small, genuinely trained NumPy text classifier for the ApicalAI MVP.

This is a demonstration classifier, NOT a general-purpose language model.
It learns weights from labeled examples at startup and predicts sentiment.
"""
import re
import numpy as np

EXAMPLES = [
    ("i love this excellent product", 1),
    ("wonderful helpful and amazing", 1),
    ("this is great and useful", 1),
    ("very happy with the service", 1),
    ("fantastic quality and fast", 1),
    ("good reliable experience", 1),
    ("i hate this terrible product", 0),
    ("awful broken and disappointing", 0),
    ("this is bad and useless", 0),
    ("very unhappy with the service", 0),
    ("horrible quality and slow", 0),
    ("poor unreliable experience", 0),
]


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


class DemoSentimentModel:
    """Binary bag-of-words logistic regression trained using gradient descent."""

    def __init__(self):
        self.vocabulary = sorted({word for text, _ in EXAMPLES for word in tokenize(text)})
        self.index = {word: i for i, word in enumerate(self.vocabulary)}
        x = np.stack([self._features(text) for text, _ in EXAMPLES])
        y = np.array([label for _, label in EXAMPLES], dtype=float)
        self.weights = np.zeros(len(self.vocabulary), dtype=float)
        self.bias = 0.0
        for _ in range(600):
            logits = np.clip(x @ self.weights + self.bias, -30, 30)
            probabilities = 1 / (1 + np.exp(-logits))
            error = probabilities - y
            self.weights -= 0.3 * (x.T @ error / len(y))
            self.bias -= 0.3 * float(error.mean())

    def _features(self, text):
        features = np.zeros(len(self.vocabulary), dtype=float)
        for word in tokenize(text):
            if word in self.index:
                features[self.index[word]] += 1.0
        return features

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input.text must be a non-empty string")
        features = self._features(text)
        if not np.any(features):
            raise ValueError("No recognized vocabulary; try a short sentiment sentence")
        logit = float(np.clip(features @ self.weights + self.bias, -30, 30))
        positive = float(1 / (1 + np.exp(-logit)))
        label = "positive" if positive >= 0.5 else "negative"
        return {"label": label, "confidence": round(max(positive, 1 - positive), 4),
                "positive_probability": round(positive, 4),
                "model_type": "trained_numpy_logistic_regression",
                "note": "Small demonstration model; not validated for production accuracy"}

    def predict_batch(self, texts):
        """Vectorized CPU inference using the same weights as predict()."""
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("texts must be a non-empty list or tuple")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("input.text must be a non-empty string")
        x = np.stack([self._features(text) for text in texts])
        if not np.all(np.any(x, axis=1)):
            raise ValueError("No recognized vocabulary; try a short sentiment sentence")
        logits = np.clip(x @ self.weights + self.bias, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        return [
            {"label": "positive" if p >= 0.5 else "negative",
             "confidence": round(max(float(p), 1 - float(p)), 4),
             "positive_probability": round(float(p), 4),
             "model_type": "trained_numpy_logistic_regression",
             "note": "Small demonstration model; not validated for production accuracy"}
            for p in probabilities
        ]
