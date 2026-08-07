# 02 — Local LLM (offline, text-only)

## Goal

Get a real Llama model running fully offline on the M4 MacBook Air, with
no internet required after the initial download.

## Tool used: Ollama

Ollama is a tool that packages everything needed to run a local LLM:
- Downloads an optimized version of the model
- Runs a local server in the background (`localhost:11434` by default)
- Handles hardware acceleration automatically (uses Apple Metal on this
  Mac; would use CUDA on NVIDIA hardware)
- Provides both an interactive chat (in Terminal) and an API other
  programs can call

Conceptually, Ollama sits inside the "text worker" box from the
architecture diagram — it's the engine that actually runs the model. Our
Flask coordinator (built in Phase 1) will later send requests to Ollama's
API instead of loading a model directly with `transformers`.

## What we did

1. Installed Ollama (Mac app, from ollama.com/download). Had to open the
   app once before the `ollama` command was available in Terminal — the
   app registers the CLI tool on first launch.
2. Pulled a model:
   ```bash
   ollama pull llama3.2:3b
   ```
   Llama 3.2, 3-billion-parameter version — chosen as a balance between
   capability and running smoothly on 16GB RAM. ~2GB download, one-time.
3. Ran it interactively:
   ```bash
   ollama run llama3.2:3b
   ```
   Confirmed a real response to "hi" — fully offline, running locally.

## Concepts learned

- A model "size" like 3b (3 billion parameters) is a tradeoff: bigger
  models are generally more capable but need more RAM/compute and run
  slower. 3b is a reasonable starting point for a 16GB laptop.
- Ollama's local server (port 11434) follows the same client/server
  pattern learned in Phase 1 — it's just a server we didn't have to write
  ourselves.
- The `ollama` command wasn't available until the app was opened once —
  installing an app and registering its CLI tool aren't always the same
  step on Mac.

## Status

Model downloaded and confirmed working via interactive chat.

## Next

Connect Ollama's API to our Flask coordinator, so `/predict`-style
requests go to the real local LLM instead of the tiny sentiment model
from Phase 1.
