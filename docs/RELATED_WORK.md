# Preliminary related work and research boundaries

This is a starting-point literature review, not proof of novelty.

- Reddi et al., *MLPerf Inference Benchmark* (2019 preprint, 2020 ISCA):
  https://arxiv.org/abs/1911.02549. Standardized scenarios and quality targets
  are already established in inference benchmarking.
- MLCommons, MLPerf Inference Edge:
  https://mlcommons.org/benchmarks/inference-edge/. Edge inference benchmarking,
  standardized load generation and accuracy constraints already exist.
- MLCommons, MLPerf Inference Tiny:
  https://mlcommons.org/benchmarks/inference-tiny/. Very small devices are
  already evaluated using task quality, latency and optional energy metrics.
- Ong et al., *RouteLLM: Learning to Route LLMs from Preference Data*, ICLR
  2025: https://arxiv.org/abs/2406.18665. Learned strong/weak model selection
  already addresses the cost–quality tradeoff.
- NVIDIA, GenAI-Perf documentation:
  https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html.
  Serving benchmarks already measure token throughput, latency and request load;
  the documentation recommends AIPerf for new experiments.

## Candidate contribution, not established novelty

Investigate the reproducible tradeoff among resource footprint, service
latency, and selective prediction quality for lightweight heterogeneous
inference systems. Evaluate quality constraints with held-out datasets, not
confidence scores alone. Compare to published baselines and test across
actual machines, recording hardware, model version, seeds and uncertainty.

Existing toy-model experiments demonstrate methodology and software
correctness only. No claim of superiority, novel routing, clinical utility,
patentability, or publication readiness is supported by them.

## Next controlled experiments

1. Equal model weights and workload across runtime implementations.
2. Independent labeled evaluation of model correctness.
3. Repeated, randomized CPU and edge runs, plus true process cold start.
4. Closed-loop HTTP concurrency and open-loop arrival-rate tests separately.
5. Quality-constrained routing with fixed-model and published routing baselines.
6. Report failure rates, confidence intervals and limitations.
