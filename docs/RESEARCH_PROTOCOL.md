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

## Paired vectorized CPU experiment

Run `python -m inferbench.batch_study --repeats 20 --output batch-study.json`.
Both variants use exactly the same trained weights, validate equal outputs,
execute warm-up predictions and alternate order. Raw timings are retained.
This is a CPU microbenchmark of vectorized operations, not GPU dynamic batching.
The API also accepts `mode: vectorized_cpu` on `/api/batch` (up to 32 items).

## HTTP serving experiment

Start a real API server, then run `python -m inferbench.http_study --requests 60 --output http-study.json`.
It uses concurrent clients and includes HTTP overhead, reports failures, and
checks output consistency. It does not measure independently labeled accuracy.
See [preliminary related work](RELATED_WORK.md) before making novelty claims.
