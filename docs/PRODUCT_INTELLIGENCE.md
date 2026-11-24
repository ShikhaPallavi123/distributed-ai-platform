# Product intelligence features (development)

- POST /api/chat accepts bounded user/assistant conversation messages and optionally streams NDJSON from a configured real Ollama model. Conversation history is client-managed and not stored by this API.
- POST /api/retrieve returns exact matching quotations with verified source offsets, or abstains when no keyword match exists. This is extractive retrieval, not semantic RAG or factual verification.
- POST /api/tools/run executes an explicit list of bounded calculator/retrieval operations. No arbitrary execution, shell, external network tools, or autonomous agent planning.
- The Remote Inference console exposes all three features. The operator must provide a trusted backend; GitHub Pages does not host an LLM.

The live Ollama CI workflow verifies direct generation, chat and streaming on temporary CPU infrastructure. Tests do not establish answer correctness, multi-tenant isolation, or production scalability.
