"""Exercise a real, already downloaded Ollama model through ApicalAI HTTP.

This is an execution smoke test with a tiny exact-match calibration set,
not an accuracy study or a benchmark of production LLM serving.
"""
import argparse
import json
import os
import threading
from urllib.request import urlopen

from apicalai_cli import request
from inference.ollama_adapter import OllamaAdapter
from inferbench.quality import evaluate


CALIBRATION = [
    {"text": "Answer with only the number: what is 2 plus 3?", "expected": "5"},
    {"text": "Answer with only the number: what is 8 minus 2?", "expected": "6"},
    {"text": "Answer with only one word: what is the opposite of hot?", "expected": "cold"},
]


def run(model, url):
    adapter = OllamaAdapter(model, url)
    calibration = evaluate(adapter.generate, CALIBRATION, "text")
    if calibration["errors"]:
        raise RuntimeError("Real model calibration had provider failures")
    config = [{
        "name": "llm-smoke", "model": model, "url": url,
        "calibration_id": "three_prompt_smoke_v1",
        "accuracy": calibration["accuracy"],
        "sample_count": calibration["sample_count"],
        "median_latency_ms": calibration["median_latency_ms"],
    }]
    os.environ["APICALAI_MODEL_POOL"] = json.dumps(config)
    os.environ["APICALAI_OLLAMA_MODEL"] = model
    os.environ["APICALAI_OLLAMA_URL"] = url
    from api.app import create_server

    server = create_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    key = os.getenv("APICALAI_API_KEY")
    try:
        direct = request(base, "/api/inference", {
            "model": "ollama-local", "input": {"text": "Say hello in one word."},
        }, key)
        routed = request(base, "/api/route", {
            "text": "Answer with a single word: name a primary color.",
            "min_quality": 0,
        }, key)
        for output in (direct["output"], routed["output"]):
            if output.get("provider_model") != model or not output.get("text", "").strip():
                raise RuntimeError("No actual model text returned")
        if routed["selected_model"] != "llm-smoke":
            raise RuntimeError("Unexpected route")
        with urlopen(url.rstrip("/") + "/api/tags", timeout=10) as response:
            tags = json.load(response)
        return {"scope": "live_pretrained_model_smoke_test", "model": model,
                "calibration": calibration, "direct": direct, "routed": routed,
                "installed_model_metadata": tags,
                "note": "A single real CPU model; no quality or multi-GPU performance claim."}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.model, args.url)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Live pretrained inference and routed inference returned actual model text.")


if __name__ == "__main__":
    main()
