"""Conservative extractive retrieval. Quotes are verified against supplied text."""
import re


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "a", "an", "the", "is", "are", "of", "to", "in", "and", "what", "how", "does", "which",
    }


def retrieve(question, documents, limit=3):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("question must contain 1–2000 characters")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 20:
        raise ValueError("provide 1–20 documents")
    if not isinstance(limit, int) or not 1 <= limit <= 5:
        raise ValueError("limit must be 1–5")
    query = words(question)
    ranked = []
    seen = set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("documents must be objects")
        source, text = doc.get("id"), doc.get("text")
        if not isinstance(source, str) or not source.strip() or len(source) > 100 or source in seen:
            raise ValueError("unique document ids required")
        seen.add(source)
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError("document text must contain 1–20000 characters")
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
            quote = match.group().strip()
            if not quote:
                continue
            overlap = len(query & words(quote))
            if overlap:
                start = match.start() + len(match.group()) - len(match.group().lstrip())
                ranked.append((overlap, source, start, quote, text))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    citations = []
    for score, source, start, quote, original in ranked[:limit]:
        # Exact provenance; never claim semantic entailment or factual truth.
        if original[start:start + len(quote)] != quote:
            raise RuntimeError("Citation verification failed")
        citations.append({"document_id": source, "start": start, "end": start + len(quote),
                          "quote": quote, "matched_terms": score})
    return {"status": "retrieved" if citations else "abstained", "citations": citations,
            "answer": "\n".join(c["quote"] for c in citations) if citations else "No matching passage found.",
            "verification": "exact_source_spans_only",
            "note": "Keyword matches may be irrelevant or incorrect. No factuality or entailment guarantee."}
