# NeuroFlux — real inference MVP

This page separates **running features** from planned features. The public
GitHub Pages dashboard is a **static preview**; it does not host a Python
server or execute GPU inference. The backend must be run separately.

## Working backend in the current development branch

- `demo-sentiment-v1`: a **small trained NumPy bag-of-words logistic regression**
  classifier. It is not a language model, and it is not production-validated.
- `ollama-local` (optional): sends requests to an Ollama server running a real
  pretrained language model. It appears only if configured.
- `/api/inference`: returns the actual executed model output, or a clear error.
- `/api/metrics/latency` and `/api/metrics/throughput`: use real successful
  request observations. Empty history means no recent measurements.
- `/api/batch`: **not implemented** (HTTP 501), not synthetic batch results.

**Not yet production ready:** accounts, authentication, secure deployment,
multiple remote GPU workers, model-upload safety, billing, quality evaluation,
multi-model research routing, and production monitoring. Existing agent screens
may contain demonstration behavior.

## Run the NumPy CPU model

From the repository root, using Python 3.10+:

```bash
python3 -m pip install -r requirements.txt
python3 -m api.app
```

In another terminal:

```bash
curl -s http://localhost:8080/api/inference \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-sentiment-v1","input":{"text":"excellent great useful"}}'
```

The response includes a real classifier prediction and measured latency.

## Run a pretrained language model through Ollama (optional)

1. Install Ollama from https://ollama.com/ and start the Ollama server.
2. Download an Ollama model compatible with your hardware, for example:

```bash
ollama pull gemma3:1b
```

3. Start NeuroFlux with the model name set:

```bash
NEUROFLUX_OLLAMA_MODEL=gemma3:1b python3 -m api.app
```

4. In another terminal, send an inference request:

```bash
curl -s http://localhost:8080/api/inference \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama-local","input":{"text":"Explain distributed inference simply."}}'
```

A text response is returned from **the configured Ollama model**, not the
built-in NumPy classifier. If Ollama or the model is unavailable, the API returns
HTTP 503. NeuroFlux does not install or download a model automatically.

To use a custom Ollama server URL, set `NEUROFLUX_OLLAMA_URL` before starting
NeuroFlux. Do not expose an unauthenticated NeuroFlux API to the public Internet.

## Validate

```bash
python3 -m unittest discover tests/ -v
python3 -m benchmarks.sentiment_baseline --iterations 200
```

The Ollama adapter is tested in CI **using mock network responses**. These
tests validate protocol handling, but do not certify the quality or speed of an
actual language model. Real-model integration tests and comparative GPU
benchmarks are future work.

## Research direction

Investigate quality-constrained adaptive model and worker routing.
Before publication, implement and benchmark strong baselines, validate model
quality against held-out labeled datasets, conduct multiple repetitions,
report confidence intervals, and investigate relevant existing literature.
The current sentiment microbenchmark is **not** evidence of a distributed
inference research contribution.
