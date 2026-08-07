<<<<<<< HEAD
# Distributed AI Platform (local, multi-modal, offline)

A step-by-step learning project building toward an internal, fully offline
multi-modal AI platform: local LLM, speech, image, audio, and video
inference, distributed across worker processes, with GPU optimization and
an agent layer on top.

This repo doubles as a build log. Every phase has its own folder for code,
and a matching entry in `docs/journal/` explaining what was built and what
was learned along the way.

## Phases

| Phase | Folder | Status |
|---|---|---|
| 1. Foundations (Python, APIs, Flask, first real model) | `phase1-foundations/` | In progress |
| 2. Local LLM (offline, text-only) | `phase2-local-llm/` | Not started |
| 3. Distribution (multiple workers in parallel) | `phase3-distribution/` | Not started |
| 4. Fine-tuning | `phase4-finetuning/` | Not started |
| 5. Modalities (speech, image, audio, video) | `phase5-modalities/` | Not started |
| 6. GPU optimization | `phase6-gpu-optimization/` | Not started |
| 7. Agents | `phase7-agents/` | Not started |

See `docs/journal/00-architecture.md` for the full system design.

## Hardware

Developing primarily on a MacBook Air (Apple M4, 16GB RAM). GPU-specific
work (Phase 6) requires NVIDIA hardware (CUDA) and will be done on
cloud/lab GPU access separately.
=======
# distributed-ai-platform
Local, offline, multi-modal AI platform: distributed LLM inference, speech/image recognition, GPU optimization, and agents. Built step-by-step as a learning project.
>>>>>>> 512760107bf1a417535078e6b5148bc41d459296
