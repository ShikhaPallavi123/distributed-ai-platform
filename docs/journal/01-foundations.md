# 01 — Foundations

## What this phase covers

Getting comfortable with the basic building blocks before touching
distribution or local LLMs: running a real pretrained model, and wrapping
it behind a simple web API.

## Steps

### step2_basic_inference.py
Loads a small real pretrained model (DistilBERT fine-tuned for sentiment
analysis) directly in Python and runs inference on a few example
sentences. No server, no API yet — just proving a real model can be
loaded and used.

### step3_flask_server.py
Wraps the same model in a Flask server. Exposes `POST /predict`: send
JSON like `{"text": "..."}`, get back `{"sentiment": ..., "confidence": ...}`.
The model is loaded once when the server starts (not per-request) to
avoid reloading it on every call.

## Concepts learned

- Python basics translated from prior language experience (indentation
  instead of braces, f-strings, `if __name__ == "__main__":` as an entry
  point).
- What an API is, and the client/server relationship (browser vs. server,
  `GET` vs `POST`).
- What `localhost` / `127.0.0.1` and ports mean.
- Why a model should be loaded once at server startup, not inside the
  request handler.

## Status

Code written; not yet run locally (sandbox environment can't reach
huggingface.co to download model weights — needs to be run on a machine
with internet access, e.g. personal laptop).

## Next

Move to Phase 2: get a local LLM (Llama, via Ollama) running fully
offline on the M4 MacBook Air.
