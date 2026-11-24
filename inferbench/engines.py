"""InferBench engine adapters.

Each adapter wraps an inference engine with a standard interface:
- init_fn() -> engine instance
- predict_fn(engine) -> callable(payload) -> result
- metadata dict with observability/agent capabilities

This allows InferBench to benchmark any engine uniformly.
"""


# --- ApicalAI Adapter ---

def apicalai_init():
    """Initialize ApicalAI's DemoSentimentModel."""
    from inference.demo_sentiment import DemoSentimentModel
    return DemoSentimentModel()


def apicalai_predict(engine):
    """Return a predict function for ApicalAI."""
    def predict(payload):
        return engine.predict(payload)
    return predict


APICALAI_ADAPTER = {
    "name": "ApicalAI NumPy classifier (in-process)",
    "version": "0.1.0",
    "init_fn": apicalai_init,
    "predict_fn": apicalai_predict,
    "requirements_file": "requirements.txt",
    "import_name": "inference.demo_sentiment",
    "observability": {"has_model_internals": True},
    "agent_readiness": {},
}


# --- scikit-learn Baseline Adapter ---

class SklearnSentiment:
    """Actual scikit-learn in-process classifier, not Flask or a simulation."""

    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from inference.demo_sentiment import EXAMPLES, tokenize

        self.vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
        x = self.vectorizer.fit_transform([text for text, _ in EXAMPLES])
        self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
        self.classifier.fit(x, [label for _, label in EXAMPLES])

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input must be a non-empty string")
        x = self.vectorizer.transform([text])
        if x.nnz == 0:
            raise ValueError("No recognized vocabulary")
        probability = float(self.classifier.predict_proba(x)[0, 1])
        return {
            "label": "positive" if probability >= 0.5 else "negative",
            "positive_probability": probability,
            "confidence": max(probability, 1 - probability),
        }


def sklearn_init():
    return SklearnSentiment()


SKLEARN_ADAPTER = {
    "name": "scikit-learn logistic regression (in-process)",
    "version": "0.2.0",
    "init_fn": sklearn_init,
    "predict_fn": apicalai_predict,
    "requirements_file": "requirements-research.txt",
    "import_name": "sklearn",
    "observability": {"has_model_internals": True},
    "agent_readiness": {},
}


# --- Engine Registry ---

ENGINES = {
    "apicalai": APICALAI_ADAPTER,
    "sklearn": SKLEARN_ADAPTER,
}


def get_engine(name):
    """Get an engine adapter by name."""
    if name not in ENGINES:
        available = ", ".join(ENGINES.keys())
        raise ValueError(f"Unknown engine '{name}'. Available: {available}")
    return ENGINES[name]


def list_engines():
    """List all registered engine names."""
    return list(ENGINES.keys())
