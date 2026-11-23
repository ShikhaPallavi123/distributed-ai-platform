# ApicalAI — real inference MVP

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
- `/api/batch`: genuinely executes 1–32 sentiment-classifier requests, sequentially on CPU, with measured outputs and latency. Other models return HTTP 501 until batch support is implemented for them.

**Not yet production ready:** accounts, authentication, secure deployment,
multiple remote GPU workers, secure model uploads, billing, quality evaluation,
multi-model research routing, and production monitoring. The agent execution
endpoint currently returns HTTP 501 instead of fabricated success.

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

## Real batch inference (sentiment classifier)

```bash
curl -s http://localhost:8080/api/batch \\
  -H "Content-Type: application/json" \\
  -d '{"requests":[{"model":"demo-sentiment-v1","input":{"text":"wonderful great"}},{"model":"demo-sentiment-v1","input":{"text":"awful terrible"}}]}'
```

This runs two individual NumPy predictions. It does **not** claim parallelism,
continuous GPU batching, or multi-worker scheduling.

## Run a pretrained language model through Ollama (optional)

1. Install Ollama from https://ollama.com/ and start the Ollama server.
2. Download an Ollama model compatible with your hardware, for example:

```bash
ollama pull gemma3:1b
```

3. Start ApicalAI with the model name set:

```bash
APICALAI_OLLAMA_MODEL=gemma3:1b python3 -m api.app
```

4. In another terminal, send an inference request:

```bash
curl -s http://localhost:8080/api/inference \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama-local","input":{"text":"Explain distributed inference simply."}}'
```

A text response is returned from **the configured Ollama model**, not the
built-in NumPy classifier. If Ollama or the model is unavailable, the API returns
HTTP 503. ApicalAI does not install or download a model automatically.

To use a custom Ollama server URL, set `APICALAI_OLLAMA_URL` before starting
ApicalAI. Do not expose an unauthenticated ApicalAI API to the public Internet.

## Benchmark a real pretrained language model

With Ollama running and the named model already pulled:

```bash
python3 -m benchmarks.ollama_baseline --model gemma3:1b --repeats 3
```

This performs real model calls and reports observed mean, p50, p95 and p99
end-to-end latency. Warmup requests are excluded from the measurement.
You can also supply `--prompts-file prompts.txt` (one prompt per line).

The benchmark does not evaluate answer accuracy, hallucinations, routing
strategies, or GPU energy consumption. CI uses a fake adapter to test the
benchmark calculations; no measured model performance is claimed from CI.

## Administrative API access

Model and worker metadata registration, plus model deletion, are blocked by
default. These operations require the server-side `APICALAI_ADMIN_TOKEN` and
an `Authorization: Bearer <token>` header. For example:

```bash
APICALAI_ADMIN_TOKEN=replace-with-long-random-secret python3 -m api.app
```

Registering metadata does **not** load a model or verify a worker. Such entries
are labeled `registered` or `unverified`. This is a temporary safeguard,
not a substitute for authentication, authorization, quotas and rate limiting
in a publicly hosted product.

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
