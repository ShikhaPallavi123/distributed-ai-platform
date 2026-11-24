# InferBench research protocol (development)

The historical 0.1.0 results and HTML report are exploratory legacy artifacts:
the former Flask/scikit-learn adapter was a NumPy simulation, loop grouping
was not batching, and capability scores were declarations. Do not cite those
results as a real engine comparison.

Version 0.2.0 compares the ApicalAI NumPy classifier against actual scikit-learn
logistic regression using the same 12 training examples and tokenizer. This
is an in-process CPU microbenchmark, not a comparison of inference servers.
The algorithms and regularization differ; quality equivalence is not established.

Run with optional research dependencies:
```bash
pip install -r requirements-research.txt
python -m inferbench.runner --duration 1 --iterations 100 --cold-start-trials 3 --output results.json
python -m inferbench.report results.json --output report.html
```

CI publishes exploratory results as artifacts. Repeated independent runs,
held-out test data, hardware details, dependency isolation, equal model weights,
HTTP concurrency, genuine vectorized batches and confidence intervals are
required for publication. Peak RSS is not current RAM; warm-process model
initialization is not process cold start. No GPU or agent capability is measured.

Future paper: characterize resource tradeoffs for comparable models across
constrained systems. Literature review and external baselines must establish
the research gap before novelty is claimed.
