"""Priority task queues with rate limiting and dead-letter support."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskItem:
    """A unit of work in the queue."""
    task_id: str
    payload: Any
    priority: int = 0  # higher = more urgent
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    deadline: float | None = None


@dataclass
class QueueMetrics:
    enqueued: int = 0
    dequeued: int = 0
    dead_lettered: int = 0
    total_wait_ms: float = 0.0

    @property
    def avg_wait_ms(self) -> float:
        return self.total_wait_ms / self.dequeued if self.dequeued else 0


class PriorityTaskQueue:
    """In-process priority queue with dead-letter support and rate limiting."""

    def __init__(self, max_size: int = 10_000, rate_limit: float = 0.0) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, float, TaskItem]] = (
            asyncio.PriorityQueue(maxsize=max_size)
        )
        self._dlq: list[TaskItem] = []
        self._rate_limit = rate_limit  # max dequeues per second, 0 = unlimited
        self._last_dequeue = 0.0
        self._metrics = QueueMetrics()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def dead_letter_count(self) -> int:
        return len(self._dlq)

    async def enqueue(self, item: TaskItem) -> None:
        """Add a task. Lower priority number = lower urgency."""
        # negate priority so higher priority dequeues first
        await self._queue.put((-item.priority, item.created_at, item))
        self._metrics.enqueued += 1

    async def dequeue(self, timeout: float | None = None) -> TaskItem:
        """Get the highest-priority task, respecting rate limits."""
        if self._rate_limit > 0:
            interval = 1.0 / self._rate_limit
            elapsed = time.time() - self._last_dequeue
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

        try:
            if timeout is not None:
                _, _, item = await asyncio.wait_for(
                    self._queue.get(), timeout=timeout
                )
            else:
                _, _, item = await self._queue.get()
        except asyncio.TimeoutError:
            raise

        self._last_dequeue = time.time()
        wait = (self._last_dequeue - item.created_at) * 1000
        self._metrics.dequeued += 1
        self._metrics.total_wait_ms += wait
        item.attempts += 1
        return item

    def send_to_dlq(self, item: TaskItem) -> None:
        """Move a failed task to the dead-letter queue."""
        self._dlq.append(item)
        self._metrics.dead_lettered += 1

    async def nack(self, item: TaskItem) -> None:
        """Return a failed task: re-enqueue or dead-letter."""
        if item.attempts >= item.max_attempts:
            self.send_to_dlq(item)
        else:
            await self.enqueue(item)

    def drain_dlq(self) -> list[TaskItem]:
        """Return and clear the dead-letter queue."""
        items = list(self._dlq)
        self._dlq.clear()
        return items

    def metrics(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "enqueued": self._metrics.enqueued,
            "dequeued": self._metrics.dequeued,
            "dead_lettered": self._metrics.dead_lettered,
            "avg_wait_ms": round(self._metrics.avg_wait_ms, 2),
        }


class DistributedQueue:
    """Partitions tasks across multiple PriorityTaskQueues by key hash."""

    def __init__(self, partitions: int = 4, max_size: int = 10_000) -> None:
        self._partitions = [
            PriorityTaskQueue(max_size=max_size) for _ in range(partitions)
        ]

    @property
    def num_partitions(self) -> int:
        return len(self._partitions)

    def _partition_for(self, key: str) -> int:
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return h % len(self._partitions)

    async def enqueue(self, item: TaskItem, partition_key: str = "") -> int:
        """Enqueue to the partition for *partition_key*. Returns partition index."""
        idx = self._partition_for(partition_key or item.task_id)
        await self._partitions[idx].enqueue(item)
        return idx

    async def dequeue(self, partition: int, timeout: float | None = None) -> TaskItem:
        return await self._partitions[partition].dequeue(timeout=timeout)

    def depth(self, partition: int | None = None) -> int:
        if partition is not None:
            return self._partitions[partition].depth
        return sum(p.depth for p in self._partitions)

    def all_metrics(self) -> list[dict[str, Any]]:
        return [p.metrics() for p in self._partitions]
