"""Core OS-level primitives for the Distributed AI Inference Engine.

Submodules
----------
memory.pool_allocator
    Custom memory pool allocator with first-fit / best-fit strategies,
    tensor-aware memory management, and defragmentation.

scheduler.task_scheduler
    Multi-algorithm task scheduler (FIFO, Priority, Round-Robin, SJF, MLFQ)
    with DAG dependency resolution and an inference-aware extension.

ipc.message_bus
    Inter-process communication: pub/sub message bus, shared-memory
    zero-copy channels, pipeline channels with back-pressure, and
    compact binary serialisation.

profiler.system_profiler
    System and inference profiler with real-time metric streaming,
    per-layer latency breakdown, and flame-graph construction.
"""

from core.memory.pool_allocator import (
    AllocationStrategy,
    MemoryBlock,
    MemoryPool,
    MMapRegion,
    TensorMemoryManager,
)
from core.scheduler.task_scheduler import (
    InferenceScheduler,
    SchedulingAlgorithm,
    Task,
    TaskScheduler,
    TaskState,
)
from core.ipc.message_bus import (
    Message,
    MessageBus,
    MessageType,
    PipelineChannel,
    SharedMemoryChannel,
)
from core.profiler.system_profiler import (
    FlameGraph,
    FlameNode,
    InferenceProfiler,
    LatencyStats,
    SystemProfiler,
)

__all__ = [
    # memory
    "AllocationStrategy",
    "MemoryBlock",
    "MemoryPool",
    "MMapRegion",
    "TensorMemoryManager",
    # scheduler
    "InferenceScheduler",
    "SchedulingAlgorithm",
    "Task",
    "TaskScheduler",
    "TaskState",
    # ipc
    "Message",
    "MessageBus",
    "MessageType",
    "PipelineChannel",
    "SharedMemoryChannel",
    # profiler
    "FlameGraph",
    "FlameNode",
    "InferenceProfiler",
    "LatencyStats",
    "SystemProfiler",
]
