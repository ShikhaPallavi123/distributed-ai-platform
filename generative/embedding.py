"""Embedding module with Word2Vec, dense vector representations, and semantic search.

Provides skip-gram and CBOW training (simplified), cosine similarity operations,
nearest-neighbor lookups, and an in-memory vector index for semantic search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Word2VecMode(Enum):
    """Training mode for Word2Vec."""
    SKIPGRAM = "skipgram"
    CBOW = "cbow"


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class Word2Vec:
    """Simplified Word2Vec implementation supporting Skip-gram and CBOW.

    Trains word embeddings by predicting context from target (skip-gram)
    or target from context (CBOW) using a shallow two-layer network.
    """

    def __init__(
        self,
        embedding_dim: int = 50,
        window_size: int = 2,
        mode: Word2VecMode = Word2VecMode.SKIPGRAM,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.mode = mode
        self.lr = learning_rate
        self.rng = np.random.RandomState(seed)

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.W_in: Optional[np.ndarray] = None   # input embeddings
        self.W_out: Optional[np.ndarray] = None   # output embeddings

    def _build_vocab(self, tokens: List[str]) -> None:
        unique = sorted(set(tokens))
        self.word_to_id = {w: i for i, w in enumerate(unique)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        V = len(unique)
        self.W_in = self.rng.randn(V, self.embedding_dim).astype(np.float32) * 0.1
        self.W_out = self.rng.randn(self.embedding_dim, V).astype(np.float32) * 0.1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _generate_pairs(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """Generate (target, context) pairs for training."""
        ids = [self.word_to_id[t] for t in tokens if t in self.word_to_id]
        pairs = []
        for i, target in enumerate(ids):
            start = max(0, i - self.window_size)
            end = min(len(ids), i + self.window_size + 1)
            for j in range(start, end):
                if j != i:
                    if self.mode == Word2VecMode.SKIPGRAM:
                        pairs.append((target, ids[j]))
                    else:
                        pairs.append((ids[j], target))
        return pairs

    def train(self, corpus: str, epochs: int = 50) -> List[float]:
        """Train embeddings on a text corpus.

        Args:
            corpus: Raw text to train on.
            epochs: Number of training epochs.

        Returns:
            List of average loss per epoch.
        """
        tokens = _tokenize_simple(corpus)
        if not self.word_to_id:
            self._build_vocab(tokens)

        pairs = self._generate_pairs(tokens)
        if not pairs:
            return []

        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.rng.shuffle(pairs)  # type: ignore[arg-type]
            for target_id, context_id in pairs:
                # Forward: compute softmax over vocabulary
                h = self.W_in[target_id]  # (embedding_dim,)
                scores = h @ self.W_out   # (V,)
                probs = self._softmax(scores)

                # Cross-entropy loss
                epoch_loss -= np.log(probs[context_id] + 1e-10)

                # Backward pass (SGD)
                grad_scores = probs.copy()
                grad_scores[context_id] -= 1.0  # (V,)

                # Update W_out
                grad_W_out = np.outer(h, grad_scores)   # (d, V)
                self.W_out -= self.lr * grad_W_out

                # Update W_in
                grad_h = self.W_out @ grad_scores  # (d,)
                self.W_in[target_id] -= self.lr * grad_h

            losses.append(epoch_loss / len(pairs))
        return losses

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a word."""
        idx = self.word_to_id.get(word.lower())
        if idx is None or self.W_in is None:
            return None
        return self.W_in[idx].copy()

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar words by cosine similarity."""
        vec = self.get_embedding(word)
        if vec is None or self.W_in is None:
            return []
        sims = cosine_similarity_matrix(vec.reshape(1, -1), self.W_in)[0]
        idx = np.argsort(sims)[::-1]
        results = []
        for i in idx:
            w = self.id_to_word[int(i)]
            if w != word.lower():
                results.append((w, float(sims[i])))
            if len(results) >= top_k:
                break
        return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between query vectors and a matrix of vectors."""
    norms_q = np.linalg.norm(query, axis=1, keepdims=True)
    norms_v = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms_q = np.maximum(norms_q, 1e-10)
    norms_v = np.maximum(norms_v, 1e-10)
    return (query / norms_q) @ (vectors / norms_v).T


class EmbeddingModel:
    """General-purpose embedding model producing dense vector representations.

    Uses a randomly initialized projection (simulating a pre-trained encoder)
    to convert tokenized text into fixed-dimensional dense vectors.
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, seed: int = 42) -> None:
        self.embedding_dim = embedding_dim
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1
        self._vocab_size = vocab_size

    def embed_tokens(self, token_ids: List[int]) -> np.ndarray:
        """Embed a sequence of token IDs, returning mean-pooled representation."""
        valid = [t for t in token_ids if 0 <= t < self._vocab_size]
        if not valid:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        vecs = self.embeddings[valid]
        return vecs.mean(axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed raw text using simple hashing to simulate tokenization."""
        tokens = _tokenize_simple(text)
        ids = [hash(t) % self._vocab_size for t in tokens]
        return self.embed_tokens(ids)


@dataclass
class SearchResult:
    """Result from semantic search."""
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)


class SemanticSearch:
    """In-memory vector index for semantic similarity search.

    Stores documents as dense embeddings and retrieves the most
    similar ones to a query using cosine similarity.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None) -> None:
        self.model = embedding_model or EmbeddingModel()
        self._texts: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict] = []

    @property
    def size(self) -> int:
        return len(self._texts)

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add a document to the search index."""
        vec = self.model.embed_text(text)
        self._texts.append(text)
        self._vectors.append(vec)
        self._metadata.append(metadata or {})

    def add_batch(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> None:
        """Add multiple documents at once."""
        metas = metadata_list or [{} for _ in texts]
        for text, meta in zip(texts, metas):
            self.add(text, meta)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for the most similar documents to the query.

        Args:
            query: The search query text.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if not self._vectors:
            return []

        query_vec = self.model.embed_text(query)
        matrix = np.array(self._vectors)
        sims = cosine_similarity_matrix(query_vec.reshape(1, -1), matrix)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            SearchResult(
                text=self._texts[i],
                score=float(sims[i]),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]

    def remove(self, index: int) -> None:
        """Remove a document from the index by position."""
        if 0 <= index < len(self._texts):
            self._texts.pop(index)
            self._vectors.pop(index)
            self._metadata.pop(index)
