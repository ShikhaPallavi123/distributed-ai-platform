"""Bounded embedding-based retrieval with an exact in-memory vector index.

No document content is persisted. Source spans are returned verbatim; similarity
does not establish entailment, truth, or an absence of hallucinations.
"""
import math
import numpy as np
from inference.ollama_adapter import OllamaBackendError


def build_passages(documents):
    if not isinstance(documents, list) or not 1 <= len(documents) <= 20:
        raise ValueError("provide 1–20 documents")
    passages = []
    seen = set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("document must be an object")
        name, text = doc.get("id"), doc.get("text")
        if not isinstance(name, str) or not name or len(name) > 100 or name in seen:
            raise ValueError("unique document IDs required")
        seen.add(name)
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError("document must contain 1–20000 characters")
        # Character windows bound embedding cost and preserve source offsets.
        for start in range(0, len(text), 600):
            end = min(start + 800, len(text))
            quote = text[start:end]
            if quote.strip():
                passages.append({"document_id": name, "start": start, "end": end, "quote": quote})
            if end == len(text):
                break
    if not 1 <= len(passages) <= 128:
        raise ValueError("embedding index supports at most 128 passages per request")
    return passages


def semantic_retrieve(embed, question, documents, limit=3, min_similarity=.35):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("question must contain 1–2000 characters")
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("limit must be 1–5")
    if not isinstance(min_similarity, (float, int)) or not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("invalid similarity threshold")
    passages = build_passages(documents)
    inputs = [question] + [p["quote"] for p in passages]
    vectors = []
    for i in range(0, len(inputs), 16):
        batch = embed(inputs[i:i + 16])
        if not isinstance(batch, list) or len(batch) != len(inputs[i:i + 16]):
            raise OllamaBackendError("Embedding count mismatch")
        vectors.extend(batch)
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(inputs) or not 1 <= matrix.shape[1] <= 4096:
            raise ValueError("invalid embedding dimensions")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError("invalid embedding values")
        matrix /= norms[:, None]
    except (ValueError, TypeError, OverflowError) as exc:
        raise OllamaBackendError("Invalid embedding response") from exc
    scores = matrix[1:] @ matrix[0]
    ranked = sorted(range(len(passages)), key=lambda i: (-float(scores[i]), i))
    citations = [{**passages[i], "cosine_similarity": float(scores[i])}
                 for i in ranked[:limit] if scores[i] >= min_similarity]
    return {"status": "retrieved" if citations else "abstained",
            "citations": citations, "answer": "\n".join(p["quote"] for p in citations)
            if citations else "No passage meets the similarity threshold.",
            "verification": "exact_source_spans_only",
            "index": "exact_cosine_in_memory", "note": "Similarity is not factual verification."}


def verify_citations(documents, citations):
    """Independent exact-span verifier; returns no semantic truth claim."""
    build_passages(documents)
    if not isinstance(citations, list) or not 1 <= len(citations) <= 5:
        raise ValueError("provide 1–5 citations")
    source = {d["id"]: d["text"] for d in documents}
    checks = []
    for item in citations:
        if not isinstance(item, dict):
            raise ValueError("citation must be an object")
        name, start, end, quote = (item.get(k) for k in ("document_id", "start", "end", "quote"))
        text = source.get(name) if isinstance(name, str) else None
        valid = (isinstance(text, str) and type(start) is int and type(end) is int
                 and 0 <= start < end <= len(text) and isinstance(quote, str)
                 and text[start:end] == quote)
        checks.append(bool(valid))
    return {"all_spans_verified": all(checks), "checks": checks,
            "scope": "source_provenance_not_entailment_or_truth"}
