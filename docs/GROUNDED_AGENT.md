# Grounded agent and semantic retrieval (experimental)

The model selects a bounded calculator, evidence index, or abstention action.
No shell, external URL, filesystem operation, or arbitrary code is exposed.
Only deterministic arithmetic output or exact document passages are returned.
Model selection may misunderstand intent; citations do not establish truth.
This is a restricted single-agent policy, not general autonomous execution.

Configure APICALAI_EMBED_MODEL with an installed Ollama embedding model.
POST /api/semantic-retrieve takes question and documents, uses the actual
Ollama embedding endpoint and cosine similarity over an ephemeral NumPy
vector index. It is not a persistent vector database. Documents are sent
to the trusted embedding server. POST /api/verify-citations independently
checks exact offsets and quotation contents; it does not test entailment.

POST /api/agents/run takes goal, optional documents and max_steps (1–5).
Configured model pool endpoints open a circuit after two provider failures,
then attempt recovery after 30 seconds. This is endpoint failover, not
physical GPU process recovery. Tests use mocked provider responses.
