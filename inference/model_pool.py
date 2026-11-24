"""Operator-configured language model endpoints with explicit calibration metadata."""
import json
from inference.adaptive_router import Candidate
from inference.ollama_adapter import OllamaAdapter


def load_pool(config):
    """Load a JSON array from trusted server configuration, never client URLs.

    Quality data must describe a shared task and calibration dataset. The
    server cannot independently verify operator-supplied evidence.
    """
    if not config:
        return {}, None
    rows = json.loads(config)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("Pool configuration must be a list of 1–16 models")
    result = {}
    protocol = None
    for row in rows:
        name = row["name"]
        evidence = row["calibration_id"]
        if not isinstance(name, str) or not name.startswith("llm-") or len(name) > 128 or name in result:
            raise ValueError("Unique llm- model aliases required")
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("calibration_id is required")
        if protocol is not None and protocol != evidence:
            raise ValueError("All models must share one calibration_id")
        protocol = evidence
        adapter = OllamaAdapter(row["model"], row.get("url", "http://127.0.0.1:11434"))
        result[name] = Candidate(name, row["accuracy"], row["sample_count"], row["median_latency_ms"], adapter.generate)
    return result, protocol
