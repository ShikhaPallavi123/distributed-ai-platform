# Live browser inference demo

The GitHub Pages website is a static application. Static hosting **cannot**
execute the Python backend or provision GPUs.

The **Inference** tab therefore runs a real, tiny logistic-regression classifier
directly in the visitor's browser using JavaScript and the device CPU. A trained
model is constructed from the same 12 labeled examples and the same full-batch
gradient-descent rule as `inference/demo_sentiment.py`.

## What is real?

- Visitors enter text and receive a prediction calculated by a trained model.
- The single and batch inference interfaces execute on the visitor's CPU.
- Completed requests, latency measurements and recent throughput reflect only
  actual predictions made during the **current browser session**.
- Input stays in the browser; the page does not send it to the project server.
- The dashboard lists a local CPU worker rather than pretending that a remote
  GPU cluster exists.

## What is not demonstrated?

- **No LLM or chatbot:** this classifier predicts positive/negative sentiment.
- **No accuracy claim:** training on 12 examples does not establish benchmark
  accuracy, generalization, robustness or fairness.
- **No distributed execution:** only one browser CPU worker; the site's
  architecture view describes the codebase, not deployed compute nodes.
- **No hosted model server:** GitHub Pages cannot execute Ollama or Python.
- **No production telemetry:** browser timing excludes network and server time,
  and session statistics reset when the page is reloaded.

The Python API and the Ollama adapter are separate runtime integrations; to
allow visitors to use hosted language models in the future, a properly
authenticated and maintained backend must be deployed on a server.

## Try it

1. Open the GitHub Pages project site.
2. Select **Inference** or **Try a real prediction**.
3. Enter `wonderful great excellent` or `awful broken disappointing`.
4. Select **Run Inference** to see the prediction and local latency.
5. In batch mode, enter one sentence per line (1–32 lines).

Unknown vocabulary returns a clear error instead of fabricated confidence.

## Tests

The test suite uses only Node.js (no npm packages needed):

```bash
node tests/browser_model.test.cjs
node tests/browser_dashboard.test.cjs
```

GitHub Actions validates both copies of the model and dashboard, runs these
tests, and verifies files match across `docs/` and `ui/`.
All backend Python tests and the Docker build continue to run independently.

## Research use

This is a **functional smoke demonstration**, not a publishable result.
The accompanying `inferbench/` package can form a basis for research
measurements, but any paper needs controlled workloads, comparison systems,
repeated measurements, uncertainty estimates, hardware specifications and
publicly documented protocols. Never compare browser predictions against
server-side GPU results as if the hardware and model were equivalent.
