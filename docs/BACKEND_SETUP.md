# Local backend MVP

Run `docker compose up --build` to start a real CPU inference API and
dashboard on localhost:8080. SQLite model metadata and usage measurements
persist in the app-data Docker volume; prompt and output content is not stored.
The former Compose worker containers did not run an actual worker service and
were removed. This is a single-host service, not distributed GPU deployment.

Optional environment settings:
- APICALAI_API_KEY: shared bearer key for all API routes except health.
- APICALAI_ADMIN_TOKEN: distinct bearer key for administrative writes. If the
  general API key is enabled, administrative operations require the same value
  for both keys in this MVP; multi-user RBAC is not implemented.
- APICALAI_DB_PATH: persistent SQLite database path (otherwise in-memory).
- APICALAI_ALLOWED_ORIGIN: explicit trusted cross-origin frontend origin.
- APICALAI_OLLAMA_MODEL and APICALAI_OLLAMA_URL: optional configured LLM.

Docker binds only to localhost. A public service requires HTTPS, suitable
authentication, rate limits, isolated tenants, load testing and hosting.
The browser demo is independent of this backend.
