# ApicalAI: Evaluating confidence-constrained inference cascades under domain shift

**Working manuscript — not submission-ready.** No original-contribution or
performance-superiority claim has been established. Do not submit until the
authors review the results, methods, authorship and venue requirements.

## Abstract (provisional)

Resource-limited inference deployments face tradeoffs between model quality,
latency and availability. ApicalAI is a research prototype for comparing
model-selection policies under explicit calibration constraints. We propose
to evaluate confidence-based cascades across review domains using held-out
data, fixed-model baselines and randomized escalation. The aim is to determine
when calibration-time tradeoffs transfer to an unseen domain and when they
fail. Results and conclusions will be added only after verified experiments.

## Introduction and research questions

RQ1: Does a cascade reduce stronger-model invocations while preserving
held-out accuracy relative to fixed-model policies?
RQ2: How does domain shift affect the calibration target and coverage?
RQ3: How do measured component timings differ from actual end-to-end serving
latency under concurrent load?

## Related work

RouteLLM already learns routing from preference data (Ong et al., ICLR 2025,
https://arxiv.org/abs/2406.18665). vLLM addresses KV-cache memory and serving
throughput (Kwon et al., SOSP 2023, https://doi.org/10.1145/3600006.3613165).
MLPerf provides standardized inference benchmarks. The proposed contribution
must be distinguished from these and prior confidence-based classification
cascades, selective prediction and distribution-shift calibration. A broader
systematic literature review remains necessary.

## Methods

Use UCI Sentiment Labelled Sentences (Kotzias 2015, DOI 10.24432/C57604,
CC BY 4.0) and cite Kotzias et al., From Group to Individual Labels using Deep
Features, KDD 2015. Hold out each review domain, remove exact cross-split
duplicates, train two TF-IDF logistic regression variants on remaining domains,
and select an escalation threshold on a separate calibration split only.
Compare cheap-only, stronger-feature-model-only, randomly escalated and
confidence-based cascades. Report accuracy, bootstrap intervals, escalation
rates, component timing and calibration target violations. The term
stronger-feature-model describes capacity, not assured accuracy.

## Results

Pending verified public dataset artifacts. No numerical results are asserted.
The existing tiny classifier and live Ollama smoke tests demonstrate software
execution, not scientific superiority or real-world LLM accuracy.

## Limitations and future experiments

Offline timing sums omit routing, networking, queueing and batching overhead.
A single CPU environment and sentiment task cannot establish general LLM
routing performance. Repeat across seeds and hardware, evaluate independent
datasets and LLM routing baselines, measure actual serving latency, and perform
paired statistical tests. Test actual GPU workers and document cost and memory
measurements. Check for domain shift and calibration failures.

## Reproducibility

See inferbench/public_sentiment.py and the public-research CI workflow. Record
archive hash, package versions, hardware, seeds and raw result artifacts.
