# 01 — Foundations

## What this phase covers

Getting comfortable with the basic building blocks before touching
distribution or local LLMs: what an API/server actually is, running a
real pretrained model, wrapping it behind a simple web API, setting up
this Git repo, and learning core terminal commands needed for everything
else in the project.

## Background concepts (learned before writing any code)

### Client vs server
A **browser** (Chrome, Safari) is a *client* — it asks for things and
displays what comes back. A **server** is a different program, usually
running elsewhere, that *waits* for requests and *responds* to them. The
browser's address bar is part of the client; it has nothing to do with
the server itself. Flask is a tool for writing servers — not an app like
Chrome, and not something you click around in. It has no window; it just
runs in the background and listens.

### What an API is
An API is a contract for how programs talk to each other. Analogy: a
restaurant menu. The client sends a request ("give me the pasta"), the
server does the work, and sends back a response. Concretely, our API
will accept `POST /predict` with a JSON body like
`{"text": "this movie was amazing"}` and respond with
`{"sentiment": "positive", "confidence": 0.98}`.

- **JSON**: a universal text format for structured data (key-value
  pairs), the standard way APIs exchange data between languages.
- **GET vs POST**: `GET` = "give me data" (loading a page). `POST` =
  "here's data, do something with it" (submitting a form, or sending
  text to be classified).

### localhost / 127.0.0.1 / ports
When a server runs on your own machine, it's reachable only by that
machine, at the address `127.0.0.1` (a.k.a. `localhost`) — a special
address that always means "this same computer," not the internet. A
**port** is like a numbered door on the computer; a server "listens" on
one port (e.g. `5000`), and requests to that port get handed to that
server. Turning off the computer running the server ends the server,
since it only ever existed in that machine's memory. Later (Phase 3+),
the same code can run on a rented cloud machine (e.g. AWS EC2) with a
real public address, so anyone on the internet can reach it — same
concept, different location.

### Distributed inference (the "why" for later phases)
One server handling requests one at a time is like a single cashier with
a long line — everyone waits. **Distribution** means adding more workers
(cashiers) and splitting requests across them so work happens in
parallel.
- **Coordinator**: decides who does the work (doesn't do the work itself).
- **Worker**: has its own copy of the model, does the actual inference.
- **Data parallelism**: same model copied across workers, each handling
  different requests in parallel — our approach.
- **Model parallelism**: the model itself is split across workers
  because it's too big for one machine — relevant later for very large
  models.

## Code written

### step2_basic_inference.py
Loads a small real pretrained model (DistilBERT fine-tuned for sentiment
analysis) directly in Python and runs inference on a few example
sentences. No server, no API yet — just proving a real model can be
loaded and used.

Python notes for someone coming from another language:
- Indentation (whitespace) defines code blocks instead of `{ }`.
- `def function_name():` defines a function.
- `if __name__ == "__main__":` is the Python idiom for "only run this if
  this file is executed directly" — similar to a `main()` entry point.
- f-strings (`f"Text: {sentence}"`) embed variables into strings, similar
  to template literals in JS.

### step3_flask_server.py
Wraps the same model in a Flask server. Exposes `POST /predict`: send
JSON like `{"text": "..."}`, get back `{"sentiment": ..., "confidence": ...}`.

Key pieces:
- `@app.route("/predict", methods=["POST"])` — a decorator connecting a
  URL path + HTTP verb to the function below it.
- `request.json` — reads the JSON body sent by the client.
- `jsonify({...})` — converts a Python dict into a JSON HTTP response.
- The model is loaded once, at the top level of the file (server
  startup), not inside the `predict()` function — otherwise every single
  request would reload the whole model, which is slow.

## Terminal / Unix command-line basics learned

Commands here aren't "Mac-specific" — they're Unix commands, shared by
macOS, Linux, and (via WSL) Windows. Learning them now transfers directly
to cloud/Linux servers used later.

| Command | What it does |
|---|---|
| `pwd` | Print working directory — shows current folder location |
| `ls` | List files/folders in the current directory |
| `ls <folder>` | List contents of a specific folder without `cd`-ing into it |
| `cd <folder>` | Change directory — move into a folder |
| `cd ~` | Go to home directory (`~` is shorthand for it) |
| `cd ..` | Go up one directory level |

Mistakes made and what they taught us:
- `cd~` (no space) failed with "command not found" — most commands need
  a space between the command and its argument; without it, the shell
  reads it as one unknown word.
- `cd distributed-ai-platform.tar.gz` failed — `cd` only works on
  folders, never on files.
- `LS` / `CD` (capital letters) failed — Unix commands are
  case-sensitive; only lowercase works.
- `ls Downloads` run *while already inside* `Downloads` looked for a
  nonexistent `Downloads/Downloads` folder — a reminder to check `pwd`
  first when a path-based command doesn't behave as expected.

## Git / GitHub setup

- Repo structure: one repo for the whole project
  (`distributed-ai-platform`), with `docs/journal/` holding a written log
  per phase, and one folder per phase for code.
- Added `.gitignore` to keep model weights, virtual environments, and OS
  junk files out of version control.
- Added `requirements.txt` for Python dependencies.
- Initialized the repo, made the first commit, and pushed it to
  `https://github.com/ShikhaPallavi123/distributed-ai-platform`.
- Learned that a downloaded `.tar` archive (used to transfer the project
  from the build environment to the local Mac) may auto-extract in
  Safari — both the `.tar` file and the extracted folder can end up
  sitting side by side in `Downloads`.
- GitHub repo metadata: added a short description and topics
  (`machine-learning`, `llm`, `flask`, `distributed-systems`,
  `gpu-computing`, `cuda`, `ollama`, `python`, `offline-ai`,
  `agentic-ai`) so the repo's purpose and scope are clear at a glance.
  Topics/description are edited via the gear icon next to "About" on the
  repo's main page, and confirmed by refreshing the page and looking for
  the blue topic tags under the repo name.

## Running it (M4 MacBook Air)

### Virtual environment setup

macOS's Homebrew-managed Python blocks installing packages directly into
the system Python (PEP 668) to avoid breaking the OS's own Python setup.
The fix is a **virtual environment** -- an isolated, self-contained
Python install just for this project, so dependencies never conflict
with the system or other projects.

    python3 -m venv venv          # create the environment (folder: venv/)
    source venv/bin/activate      # activate it -- prompt shows (venv) prefix
    pip install transformers torch flask

The environment must be re-activated (`source venv/bin/activate`) every
time a new terminal tab/window is opened for this project -- it does not
stay active permanently.

### Running step2_basic_inference.py

    python3 step2_basic_inference.py

First run downloads the model (~250MB) from HuggingFace. Correctly
classified all 3 test sentences, including the subtle neutral-leaning
one ("The weather today is fine, nothing special" -> NEGATIVE, 0.98 --
a reasonable call, since "nothing special" reads as mildly negative).

### Running step3_flask_server.py

Run in one terminal tab (it blocks and waits for requests -- that's
expected, a server doesn't "finish"):

    python3 step3_flask_server.py

Tested from a second terminal tab with curl. Hit a zsh quoting issue
first: using double quotes with escaped inner quotes
(`-d "{\"text\": \"I love learning to code!\"}"`) failed with
`zsh: event not found: \` -- the `!` character has special meaning in
zsh (history expansion) and broke the escaped string. Fixed by using
single quotes around the whole JSON body instead, which avoids the
escaping entirely:

    curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d '{"text": "I love learning to code"}'

Response:

    {"confidence": 0.9998, "sentiment": "POSITIVE", "text": "I love learning to code"}

## Status

Both scripts have been run successfully on the M4 MacBook Air, inside a
Python virtual environment (venv). `step2_basic_inference.py` correctly
classified all 3 test sentences. `step3_flask_server.py` was run
alongside a `curl` request in a second terminal tab, returning a real
JSON response as shown above.

Phase 1 is complete: concepts understood, code written, and both scripts
verified working end to end.

## Next

Move to Phase 2: get a local LLM (Llama, via Ollama) running fully
offline on the M4 MacBook Air.
