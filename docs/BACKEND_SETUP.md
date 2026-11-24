# Local backend MVP

Run `docker compose up --build` to start a real CPU inference API and
dashboard on localhost:8080. SQLite model metadata and usage measurements
persist in the app-data Docker volume; prompt and output content is not stored.
The former Compose worker containers did not run an actual worker service and
were removed. This is a single-host service, not distributed GPU deployment.

Optional environment settings:
- APICALAI_API_KEY: shared bearer key for all API routes except health.
- APICALAI_ADMIN_TOKEN: distinct bearer key for administrative writes. If the
  general API key is enabled, a valid admin token also authorizes API access.
  Multi-user RBAC is not implemented.
- APICALAI_DB_PATH: persistent SQLite database path (otherwise in-memory).
- APICALAI_ALLOWED_ORIGIN: explicit trusted cross-origin frontend origin.
- APICALAI_OLLAMA_MODEL and APICALAI_OLLAMA_URL: optional configured LLM.

Docker binds only to localhost. A public service requires HTTPS, suitable
authentication, rate limits, isolated tenants, load testing and hosting.
The browser demo is independent of this backend.

## CLI

`python -m apicalai_cli serve` starts the local server. In another shell use
`python -m apicalai_cli status`, `python -m apicalai_cli models`, or
`python -m apicalai_cli infer 'wonderful great'`. Configure a shared API key
via the APICALAI_API_KEY environment variable, never in a public URL.

## Public website remote console

The Remote Inference page connects to a user-selected trusted API origin.
Configure HTTPS through a reverse proxy and set APICALAI_ALLOWED_ORIGIN to
`https://shikhapallavi123.github.io` for the project's Pages website.
Use a non-admin client API key, never a model-provider credential. The
console retains the connection and key only in memory, uses explicit
requests without redirects, and displays all responses as text. No hosted
backend or cloud account is provisioned automatically. Docker on Linux may
require explicit networking configuration to reach an Ollama server.
