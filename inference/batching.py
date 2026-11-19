"""Dynamic and continuous batching for inference requests."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BatchItem:
    """One item waiting to be batched."""
    item_id: str
    data: np.ndarray
    future: asyncio.Future[np.ndarray] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )
    arrived_at: float = field(default_factory=time.time)


class DynamicBatcher:
    """Groups incoming requests into batches by time or size.

    Flush triggers:
      - *max_batch_size* items accumulated
      - *max_wait_ms* elapsed since the first item arrived
      - Adaptive: when load exceeds *adaptive_threshold*, batch size doubles
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        adaptive_threshold: int = 64,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.adaptive_threshold = adaptive_threshold
        self._buffer: list[BatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._pending_count = 0
        self._process_fn: Any = None
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def set_processor(self, fn: Any) -> None:
        """Set the callable that processes a batch (np.ndarray -> np.ndarray)."""
        self._process_fn = fn

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._running = False
        self._flush_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Flush remaining
        await self._flush()

    async def submit(self, item_id: str, data: np.ndarray) -> np.ndarray:
        """Submit data and await its result after batching."""
        bi = BatchItem(item_id=item_id, data=data)
        async with self._lock:
            self._buffer.append(bi)
            self._pending_count += 1
            effective_size = self._effective_batch_size()
            if len(self._buffer) >= effective_size:
                self._flush_event.set()
        return await bi.future

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(),
                    timeout=self.max_wait_ms / 1000,
                )
            except asyncio.TimeoutError:
                pass
            self._flush_event.clear()
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        if not batch:
            return

        try:
            stacked = np.stack([b.data for b in batch])
            if self._process_fn:
                result = self._process_fn(stacked)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                result = stacked  # passthrough

            for i, bi in enumerate(batch):
                if not bi.future.done():
                    bi.future.set_result(result[i] if i < len(result) else result[-1])
        except Exception as exc:
            for bi in batch:
                if not bi.future.done():
                    bi.future.set_exception(exc)
        finally:
            self._pending_count = max(0, self._pending_count - len(batch))

    def _effective_batch_size(self) -> int:
        if self._pending_count > self.adaptive_threshold:
            return min(self.max_batch_size * 2, 256)
        return self.max_batch_size


class ContinuousBatcher:
    """LLM-style continuous batching: new requests join a running batch.

    Maintains a set of active sequences. Each iteration processes all
    active sequences together, and completed sequences are removed.
    """

    def __init__(self, max_active: int = 64) -> None:
        self.max_active = max_active
        self._active: dict[str, _Sequence] = {}
        self._waiting: list[_Sequence] = []
        self._lock = asyncio.Lock()

    async def add_sequence(
        self, seq_id: str, input_tokens: np.ndarray, max_new_tokens: int = 128
    ) -> asyncio.Future[np.ndarray]:
        seq = _Sequence(seq_id=seq_id, tokens=input_tokens, max_new=max_new_tokens)
        async with self._lock:
            if len(self._active) < self.max_active:
                self._active[seq_id] = seq
            else:
                self._waiting.append(seq)
        return seq.future

    async def step(self, generate_fn: Any) -> int:
        """Run one decode step for all active sequences.

        *generate_fn(batch_tokens: np.ndarray) -> np.ndarray* returns
        next-token logits / ids for the batch.  Returns count of completed.
        """
        async with self._lock:
            if not self._active:
                return 0
            seqs = list(self._active.values())

        # Pad to equal length for batching
        max_len = max(len(s.tokens) for s in seqs)
        padded = np.zeros((len(seqs), max_len), dtype=seqs[0].tokens.dtype)
        for i, s in enumerate(seqs):
            padded[i, : len(s.tokens)] = s.tokens

        new_tokens = generate_fn(padded)
        if asyncio.iscoroutine(new_tokens):
            new_tokens = await new_tokens

        completed = 0
        async with self._lock:
            for i, s in enumerate(seqs):
                tok = int(new_tokens[i]) if new_tokens.ndim == 1 else int(new_tokens[i, -1])
                s.tokens = np.append(s.tokens, tok)
                s.generated += 1
                if s.generated >= s.max_new:
                    if not s.future.done():
                        s.future.set_result(s.tokens)
                    self._active.pop(s.seq_id, None)
                    completed += 1

            # Admit waiting sequences
            while self._waiting and len(self._active) < self.max_active:
                s = self._waiting.pop(0)
                self._active[s.seq_id] = s

        return completed

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)


@dataclass
class _Sequence:
    seq_id: str
    tokens: np.ndarray
    max_new: int = 128
    generated: int = 0
    future: asyncio.Future[np.ndarray] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )
