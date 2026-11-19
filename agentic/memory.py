"""Memory systems for agentic AI: short-term, long-term, episodic, and working memory.

Provides sliding-window context, embedding-based retrieval, episodic experience
storage, and a combined working memory that merges short-term with retrieved long-term.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


def _simple_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic pseudo-embedding for text (for use without a trained model)."""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-10)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryEntry:
    """A single memory item."""
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    importance: float = 1.0

    def __post_init__(self) -> None:
        if self.embedding is None:
            self.embedding = _simple_embed(self.content)


class ShortTermMemory:
    """Fixed-size sliding window of recent observations.

    Operates like a conversation buffer, keeping only the last ``capacity`` items.
    """

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._buffer: Deque[MemoryEntry] = deque(maxlen=capacity)

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._buffer.append(MemoryEntry(content=content, metadata=metadata or {}))

    def get_recent(self, n: Optional[int] = None) -> List[MemoryEntry]:
        """Return the most recent n entries (all if n is None)."""
        items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return items

    def get_context_string(self, n: Optional[int] = None) -> str:
        """Concatenate recent entries into a context string."""
        return "\n".join(e.content for e in self.get_recent(n))

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class LongTermMemory:
    """Vector-store based long-term memory with embedding retrieval.

    Stores memories with their embeddings and retrieves the most relevant
    ones via cosine similarity search.
    """

    def __init__(self, embed_dim: int = 64) -> None:
        self.embed_dim = embed_dim
        self._entries: List[MemoryEntry] = []

    def store(self, content: str, importance: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = MemoryEntry(content=content, importance=importance, metadata=metadata or {})
        self._entries.append(entry)

    def retrieve(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve memories most similar to the query.

        Args:
            query: Text query to match against stored memories.
            top_k: Maximum number of results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (MemoryEntry, similarity_score) tuples, descending.
        """
        if not self._entries:
            return []
        q_emb = _simple_embed(query, self.embed_dim)
        scored = []
        for entry in self._entries:
            sim = _cosine_sim(q_emb, entry.embedding)  # type: ignore[arg-type]
            # Weight by importance
            weighted = sim * entry.importance
            if weighted >= min_similarity:
                scored.append((entry, weighted))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def forget(self, older_than: float) -> int:
        """Remove memories older than a timestamp. Returns count removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= older_than]
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class Episode:
    """A recorded episode of agent experience."""
    goal: str
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    outcome: str = ""
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        parts = [f"Goal: {self.goal}"]
        for obs, act in zip(self.observations, self.actions):
            parts.append(f"  Obs: {obs} -> Act: {act}")
        parts.append(f"Outcome: {self.outcome} (reward={self.reward:.2f})")
        return "\n".join(parts)


class EpisodicMemory:
    """Storage for past agent episodes, enabling experience-based learning."""

    def __init__(self, max_episodes: int = 100) -> None:
        self.max_episodes = max_episodes
        self._episodes: Deque[Episode] = deque(maxlen=max_episodes)

    def record(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def find_similar(self, goal: str, top_k: int = 3) -> List[Episode]:
        """Find episodes with the most similar goals."""
        if not self._episodes:
            return []
        q_emb = _simple_embed(goal)
        scored = []
        for ep in self._episodes:
            sim = _cosine_sim(q_emb, _simple_embed(ep.goal))
            scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_k]]

    def get_successful(self, min_reward: float = 0.5) -> List[Episode]:
        """Return episodes that achieved at least min_reward."""
        return [ep for ep in self._episodes if ep.reward >= min_reward]

    def __len__(self) -> int:
        return len(self._episodes)


class WorkingMemory:
    """Combines short-term context with retrieved long-term memories.

    Acts as the agent's active working set -- recent observations
    plus relevant retrieved knowledge.
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        retrieval_top_k: int = 3,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
        self.episodic = episodic or EpisodicMemory()
        self.retrieval_top_k = retrieval_top_k
        self._scratch: Dict[str, Any] = {}

    def add_observation(self, content: str, persist: bool = False, importance: float = 1.0) -> None:
        """Add an observation to short-term memory, optionally persisting to long-term."""
        self.short_term.add(content)
        if persist:
            self.long_term.store(content, importance=importance)

    def set_scratch(self, key: str, value: Any) -> None:
        """Store a temporary value in the scratchpad."""
        self._scratch[key] = value

    def get_scratch(self, key: str, default: Any = None) -> Any:
        return self._scratch.get(key, default)

    def build_context(self, query: Optional[str] = None) -> str:
        """Build a combined context string for the agent.

        Merges recent short-term memories with relevant long-term retrievals
        and similar past episodes.
        """
        sections: List[str] = []

        # Short-term
        recent = self.short_term.get_context_string()
        if recent:
            sections.append(f"[Recent context]\n{recent}")

        # Long-term retrieval
        if query:
            retrieved = self.long_term.retrieve(query, top_k=self.retrieval_top_k)
            if retrieved:
                ltm_lines = [f"- {entry.content} (relevance={score:.2f})" for entry, score in retrieved]
                sections.append("[Retrieved knowledge]\n" + "\n".join(ltm_lines))

            # Episodic
            episodes = self.episodic.find_similar(query, top_k=2)
            if episodes:
                ep_lines = [ep.summary for ep in episodes]
                sections.append("[Past experiences]\n" + "\n---\n".join(ep_lines))

        return "\n\n".join(sections)

    def consolidate(self, summarizer=None) -> None:
        """Consolidate short-term memories into long-term.

        Args:
            summarizer: Optional callable(str) -> str that summarizes text.
                        If None, stores raw concatenation.
        """
        recent = self.short_term.get_recent()
        if not recent:
            return
        combined = " ".join(e.content for e in recent)
        if summarizer:
            combined = summarizer(combined)
        self.long_term.store(combined, importance=0.8, metadata={"consolidated": True})
        self.short_term.clear()
