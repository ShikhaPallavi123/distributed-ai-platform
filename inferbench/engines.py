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
    "name": "ApicalAI",
    "version": "0.1.0",
    "init_fn": apicalai_init,
    "predict_fn": apicalai_predict,
    "requirements_file": "requirements.txt",
    "import_name": "inference.demo_sentiment",
    "observability": {
        "has_request_tracing": True,
        "has_layer_metrics": True,
        "has_memory_profiling": True,
        "has_scheduling_visibility": True,
        "has_consensus_visibility": True,
        "has_model_internals": True,
        "has_error_traces": True,
        "has_latency_breakdown": True,
    },
    "agent_readiness": {
        "has_native_agents": True,
        "has_multi_step": True,
        "has_tool_use": True,
        "has_planning": True,
        "has_multi_agent": True,
        "has_memory": True,
    },
}


# --- Flask + scikit-learn Baseline Adapter ---

def flask_sklearn_init():
    """Initialize a simple sklearn sentiment classifier as a baseline."""
    # We simulate a sklearn-like classifier using the same approach
    # but wrapped as if it were a Flask endpoint's model
    import numpy as np

    class SklearnBaseline:
        """Mimics sklearn LogisticRegression for sentiment analysis."""

        def __init__(self):
            import re
            examples = [
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
            self.vocabulary = sorted(
                {w for t, _ in examples
                 for w in re.findall(r"[a-z]+", t.lower())}
            )
            self.index = {w: i for i, w in enumerate(self.vocabulary)}

            x = np.stack([self._features(t) for t, _ in examples])
            y = np.array([l for _, l in examples], dtype=float)

            # Train with sklearn-style: normal equation approximation
            # (simulates the overhead of sklearn's fit)
            self.weights = np.linalg.lstsq(x, y, rcond=None)[0]
            self.tokenize = lambda text: re.findall(r"[a-z]+", text.lower())

        def _features(self, text):
            import re
            features = np.zeros(len(self.vocabulary), dtype=float)
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in self.index:
                    features[self.index[word]] += 1.0
            return features

        def predict(self, text):
            features = self._features(text)
            score = float(features @ self.weights)
            label = "positive" if score >= 0.5 else "negative"
            return {"label": label, "confidence": round(abs(score - 0.5) + 0.5, 4)}

    return SklearnBaseline()


def flask_sklearn_predict(engine):
    """Return a predict function for the Flask+sklearn baseline."""
    def predict(payload):
        return engine.predict(payload)
    return predict


FLASK_SKLEARN_ADAPTER = {
    "name": "Flask+sklearn Baseline",
    "version": "baseline",
    "init_fn": flask_sklearn_init,
    "predict_fn": flask_sklearn_predict,
    "requirements_file": None,
    "import_name": None,
    "observability": {
        "has_request_tracing": False,
        "has_layer_metrics": False,
        "has_memory_profiling": False,
        "has_scheduling_visibility": False,
        "has_consensus_visibility": False,
        "has_model_internals": False,
        "has_error_traces": True,
        "has_latency_breakdown": False,
    },
    "agent_readiness": {
        "has_native_agents": False,
        "has_multi_step": False,
        "has_tool_use": False,
        "has_planning": False,
        "has_multi_agent": False,
        "has_memory": False,
    },
}


# --- Engine Registry ---

ENGINES = {
    "apicalai": APICALAI_ADAPTER,
    "flask_sklearn": FLASK_SKLEARN_ADAPTER,
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
