# Quality-constrained model routing (experimental)

The backend now supports multiple operator-configured Ollama endpoints.
Models may reside on different machines. This is request routing across
endpoints, not distributed execution of one model across GPUs.

Each pool entry requires a unique alias starting with llm-, a real model name,
an optional trusted Ollama URL, measured calibration accuracy, sample count,
median latency in milliseconds, and a shared calibration_id. Set the JSON
array in APICALAI_MODEL_POOL on the server. Do not accept model endpoint URLs
from arbitrary users. No model is automatically downloaded.

Generate measurements using:
```bash
python -m inferbench.quality --dataset calibration.jsonl --model YOUR_MODEL --output calibration.json
```
Each JSONL row contains text and expected strings. Use a task where exact
match is meaningful. Evaluate all models with the same calibration dataset
and protocol; use a different held-out test set for final evaluation.
A toy classifier and a general chat model are not comparable without a common task.

POST /api/route with {"text":"...", "min_quality":0.5, "max_latency_ms":5000}.
The policy uses a Wilson lower confidence bound on calibration accuracy and
selects the eligible model with lowest measured median latency. It tries the
next candidate only on provider errors, and reports attempts and observed time.
Budget checks use historical estimates; they cannot guarantee a hard deadline.
If no eligible model exists, the API returns 503, not a made-up answer.

The operator supplies the calibration evidence; no guarantee of correctness
on new inputs or general hallucination detection is implied. The policy is
a simple research baseline, not an established novel contribution.

## Exploratory real-model study

CI evaluates two actual local classifiers on separate hand-written calibration
and test sets of 16 examples each, including negation and unknown vocabulary.
The sets were authored as examples for this project; they are not independently
annotated or representative. Exact overlap with model training is rejected.
The study compares fastest-only, highest-calibration-accuracy-only, and
quality-constrained policies. Failed or abstained predictions count against
accuracy; coverage and raw latency measurements are reported separately.
Use a substantial public dataset with an untouched test split for a paper.
