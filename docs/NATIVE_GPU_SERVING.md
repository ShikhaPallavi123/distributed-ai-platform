# Native GPU serving (experimental)

The configured model pool accepts provider llama.cpp or vllm through
the OpenAI-compatible chat endpoint. No client request can specify a URL.
Use loopback or HTTPS; for private remote nodes use an authorized SSH tunnel.
Set the operator's provider key using api_key_env, never commit credentials.

The benchmark is run from a repository checkout using python3 -m
scripts.llm_concurrency. It measures completed requests and actual provider
token counts for a fixed corpus at 1, 2 and 4 concurrent clients. It does not
establish an improvement over llama.cpp or independently prove GPU batching.
Record server flags, hardware, model digest, commit and raw results. Use
the same workload when comparing server slot counts. No permanent public
service is needed. The adapter and tests are not yet hardware-validated.
