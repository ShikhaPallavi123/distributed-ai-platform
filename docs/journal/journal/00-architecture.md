# 00 — High-level architecture

## Goal

Build an internal, offline, multi-modal AI platform: a user sends a
request (text, speech, image, audio, or video), a coordinator routes it
to the right worker, the worker runs a real model, and results come back.
No internet required once models are downloaded. Later, an agent layer
sits on top and can chain workers together (e.g. transcribe audio, then
summarize the text).

## Layers

1. **User interface** — how a request enters the system. Starts as a
   command-line test (`curl`), later becomes a real frontend.
2. **Coordinator (Flask API)** — receives requests, decides which worker
   handles them. This is the "front door" of the system.
3. **Worker pool** — one worker per model/modality. Each worker loads its
   model once and stays ready. This is the "distributed" part: multiple
   workers can run in parallel instead of one process doing everything
   serially.
   - Text worker: local LLM (Llama, via Ollama)
   - Speech worker: Whisper-style speech-to-text (later)
   - Image worker: vision model (later)
4. **Hardware** — where the actual computation happens. On the Mac this is
   Apple's Metal / Neural Engine. On NVIDIA hardware (cloud/lab GPU) this
   becomes CUDA. Same architecture, different engine underneath — this is
   where Phase 6 (GPU optimization) plugs in.
5. **Agent layer (future)** — plans multi-step tasks and calls workers as
   tools, e.g. "listen to this audio, then answer questions about it."

## Key concepts learned

- **Client vs server**: a browser (client) asks; a server waits and
  responds. Flask lets us build the server side.
- **API**: a defined way for programs to talk to each other. Our
  coordinator exposes `POST /predict`-style endpoints.
- **localhost / 127.0.0.1**: "this same machine" — a server running here
  is only reachable by the machine it runs on, until deployed elsewhere
  (e.g. AWS EC2).
- **Data parallelism vs model parallelism**: data parallelism = same
  model copied across workers, each handling different requests (our
  approach for now). Model parallelism = splitting one huge model across
  workers because it doesn't fit on one machine (relevant later for very
  large models).

## Hardware notes

- Development machine: MacBook Air, Apple M4, 16GB RAM. Good for small
  LLMs (3B-8B params), Whisper, small vision models, and learning the
  full pipeline.
- GPU optimization work (CUDA, Nsight profiling) requires actual NVIDIA
  hardware — not available on Apple Silicon (Metal is a different stack).
  Plan: pursue lab/cloud GPU access in parallel with Mac-based development.
