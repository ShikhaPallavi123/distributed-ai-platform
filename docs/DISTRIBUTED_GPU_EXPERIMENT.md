# Distributed replica experiment

The standalone multi_gpu_benchmark.py accepts distinct local forwarded
ports, a local key file and a model name. Keep servers on loopback and
use authorized SSH tunnels only. Do not expose GPU services publicly.
Do not commit institutional hostnames, keys, or private allocation details.

Each GPU runs a separate replica of the same model. Compare 1 versus N
replicas at 4 concurrent clients, three repeats, 16 requests per repeat,
with fixed prompts and a 64-token output cap. Round robin is the baseline.
The experimental scheduler chooses using current in-flight count and
observed latency (EWMA). On failure, it retries another replica and
temporarily skips the failed replica. Unit tests use simulated providers;
actual hardware scaling and recovery remain to be measured.

This is request-level distributed inference, not tensor parallelism or
distributed KV-cache optimization. Record model digest, runtime version,
server flags, node hardware, throughput, individual latency, and failures
before interpreting any performance difference.
