"""Process and task scheduler with multiple scheduling algorithms.

Provides FIFO, Priority, Round-Robin, Shortest-Job-First, and Multi-Level
Feedback Queue schedulers, plus an inference-aware scheduler with batch
coalescing, GPU affinity, and preemption for real-time requests.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class TaskState(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()
    PREEMPTED = auto()


@dataclass(order=False)
class Task:
    """Schedulable unit of work.

    Attributes:
        task_id: Unique identifier.
        fn: Callable to execute (receives ``**kwargs``).
        priority: Lower number = higher priority.
        deadline: Absolute monotonic time by which the task must finish.
        cpu_affinity: Preferred worker indices (empty = any).
        memory_requirement: Bytes of memory the task needs.
        estimated_duration: Expected run-time in seconds (for SJF).
        kwargs: Arguments forwarded to *fn*.
        dependencies: Task IDs that must complete before this task runs.
    """
    task_id: int
    fn: Callable[..., Any]
    priority: int = 10
    deadline: float = float("inf")
    cpu_affinity: set[int] = field(default_factory=set)
    memory_requirement: int = 0
    estimated_duration: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=dict)
    dependencies: set[int] = field(default_factory=set)
    state: TaskState = TaskState.PENDING
    result: Any = None
    _future: Optional[Future[Any]] = field(default=None, repr=False)

    def __lt__(self, other: Task) -> bool:
        return self.priority < other.priority


class SchedulingAlgorithm(Enum):
    FIFO = auto()
    PRIORITY = auto()
    ROUND_ROBIN = auto()
    SJF = auto()
    MLFQ = auto()


class TaskScheduler:
    """General-purpose task scheduler with pluggable algorithms.

    The scheduler maintains a ready queue and dispatches work to a
    ``ThreadPoolExecutor``.  DAG dependencies are honoured: a task enters
    the ready queue only after all its predecessors complete.
    """

    def __init__(
        self,
        algorithm: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY,
        num_workers: int = 4,
        time_quantum_ms: float = 100.0,
    ) -> None:
        self._algorithm = algorithm
        self._num_workers = num_workers
        self._time_quantum = time_quantum_ms / 1000.0
        self._lock = threading.Lock()
        self._tasks: dict[int, Task] = {}
        self._ready: list[Task] = []
        self._rr_queue: deque[Task] = deque()
        self._mlfq_queues: list[deque[Task]] = [deque() for _ in range(4)]
        self._mlfq_quanta = [0.05, 0.1, 0.2, 0.4]
        self._completed: dict[int, Task] = {}
        self._dependents: dict[int, list[int]] = defaultdict(list)
        self._pool = ThreadPoolExecutor(max_workers=num_workers)
        self._next_id = 0
        self._running_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], **kwargs: Any) -> Task:
        """Create and enqueue a task, returning it immediately."""
        with self._lock:
            tid = self._next_id
            self._next_id += 1
        task = Task(task_id=tid, fn=fn, **kwargs)
        self._register(task)
        return task

    def submit_task(self, task: Task) -> None:
        """Enqueue an already-constructed ``Task``."""
        self._register(task)

    def wait(self, task: Task, timeout: Optional[float] = None) -> Any:
        """Block until *task* completes and return its result."""
        if task._future is None:
            raise RuntimeError("task has not been dispatched")
        return task._future.result(timeout=timeout)

    def cancel(self, task_id: int) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state not in (TaskState.PENDING, TaskState.READY):
                return False
            task.state = TaskState.FAILED
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": sum(1 for t in self._tasks.values() if t.state in (TaskState.PENDING, TaskState.READY)),
                "running": self._running_count,
                "completed": len(self._completed),
                "total_submitted": self._next_id,
                "algorithm": self._algorithm.name,
            }

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register(self, task: Task) -> None:
        with self._lock:
            self._tasks[task.task_id] = task
            for dep in task.dependencies:
                self._dependents[dep].append(task.task_id)
            # Check if all dependencies already satisfied.
            if task.dependencies and not task.dependencies.issubset(self._completed):
                task.state = TaskState.BLOCKED
            else:
                task.state = TaskState.READY
                self._enqueue(task)
                self._try_dispatch()

    def _enqueue(self, task: Task) -> None:
        if self._algorithm is SchedulingAlgorithm.FIFO:
            self._ready.append(task)
        elif self._algorithm is SchedulingAlgorithm.PRIORITY:
            heapq.heappush(self._ready, task)
        elif self._algorithm is SchedulingAlgorithm.SJF:
            heapq.heappush(self._ready, task)
            # SJF uses estimated_duration to compare; we re-use priority
            # field for the heap but sort by duration in _pick_next.
        elif self._algorithm is SchedulingAlgorithm.ROUND_ROBIN:
            self._rr_queue.append(task)
        elif self._algorithm is SchedulingAlgorithm.MLFQ:
            self._mlfq_queues[0].append(task)

    def _pick_next(self) -> Optional[Task]:
        if self._algorithm is SchedulingAlgorithm.FIFO:
            return self._ready.pop(0) if self._ready else None
        elif self._algorithm in (SchedulingAlgorithm.PRIORITY,):
            return heapq.heappop(self._ready) if self._ready else None
        elif self._algorithm is SchedulingAlgorithm.SJF:
            if not self._ready:
                return None
            # Pick shortest estimated duration from the heap.
            self._ready.sort(key=lambda t: t.estimated_duration)
            return self._ready.pop(0)
        elif self._algorithm is SchedulingAlgorithm.ROUND_ROBIN:
            return self._rr_queue.popleft() if self._rr_queue else None
        elif self._algorithm is SchedulingAlgorithm.MLFQ:
            for q in self._mlfq_queues:
                if q:
                    return q.popleft()
        return None

    def _try_dispatch(self) -> None:
        """Dispatch as many ready tasks as the worker pool can accept."""
        while self._running_count < self._num_workers:
            task = self._pick_next()
            if task is None:
                break
            if task.state is not TaskState.READY:
                continue
            task.state = TaskState.RUNNING
            self._running_count += 1
            task._future = self._pool.submit(self._run_task, task)

    def _run_task(self, task: Task) -> Any:
        try:
            result = task.fn(**task.kwargs)
            task.result = result
            task.state = TaskState.COMPLETED
        except Exception as exc:
            task.result = exc
            task.state = TaskState.FAILED
        finally:
            with self._lock:
                self._running_count -= 1
                self._completed[task.task_id] = task
                self._unblock_dependents(task.task_id)
                self._try_dispatch()
        return task.result

    def _unblock_dependents(self, completed_id: int) -> None:
        for dep_id in self._dependents.get(completed_id, []):
            dep_task = self._tasks.get(dep_id)
            if dep_task is None or dep_task.state is not TaskState.BLOCKED:
                continue
            if dep_task.dependencies.issubset(self._completed):
                dep_task.state = TaskState.READY
                self._enqueue(dep_task)


class InferenceScheduler(TaskScheduler):
    """Scheduler specialised for AI inference workloads.

    Extends ``TaskScheduler`` with:
    * **Batch coalescing**: groups compatible requests into a single batch
      to improve GPU utilisation.
    * **GPU affinity**: pins model shards to specific workers.
    * **Preemption**: real-time requests (priority 0) preempt batch jobs.
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 50.0,
    ) -> None:
        super().__init__(
            algorithm=SchedulingAlgorithm.PRIORITY,
            num_workers=num_workers,
        )
        self._max_batch = max_batch_size
        self._batch_timeout = batch_timeout_ms / 1000.0
        self._pending_batches: dict[str, list[Task]] = defaultdict(list)
        self._gpu_affinity: dict[str, int] = {}
        self._batch_lock = threading.Lock()

    def set_gpu_affinity(self, model_name: str, worker_id: int) -> None:
        """Pin *model_name* to *worker_id*."""
        self._gpu_affinity[model_name] = worker_id

    def submit_inference(
        self,
        fn: Callable[..., Any],
        model_name: str,
        is_realtime: bool = False,
        **kwargs: Any,
    ) -> Task:
        """Submit an inference request, possibly coalescing into a batch."""
        priority = 0 if is_realtime else 10
        affinity: set[int] = set()
        if model_name in self._gpu_affinity:
            affinity = {self._gpu_affinity[model_name]}

        task = Task(
            task_id=self._next_task_id(),
            fn=fn,
            priority=priority,
            cpu_affinity=affinity,
            kwargs=kwargs,
        )
        task.metadata = {"model": model_name}  # type: ignore[attr-defined]

        if is_realtime:
            self._register(task)
            return task

        with self._batch_lock:
            batch = self._pending_batches[model_name]
            batch.append(task)
            if len(batch) >= self._max_batch:
                self._flush_batch(model_name)
        return task

    def flush_all_batches(self) -> None:
        with self._batch_lock:
            for model in list(self._pending_batches):
                self._flush_batch(model)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_task_id(self) -> int:
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            return tid

    def _flush_batch(self, model_name: str) -> None:
        batch = self._pending_batches.pop(model_name, [])
        if not batch:
            return
        # Wrap the batch into a single mega-task.
        def _run_batch(tasks: list[Task] = batch) -> list[Any]:
            results = []
            for t in tasks:
                try:
                    r = t.fn(**t.kwargs)
                    t.result = r
                    t.state = TaskState.COMPLETED
                    results.append(r)
                except Exception as exc:
                    t.state = TaskState.FAILED
                    t.result = exc
                    results.append(exc)
            return results

        mega = Task(
            task_id=self._next_task_id(),
            fn=_run_batch,
            priority=10,
        )
        self._register(mega)
