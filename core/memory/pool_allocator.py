"""Custom memory pool allocator for tensor operations.

Provides pre-allocated memory pools with first-fit and best-fit allocation
strategies, GPU-like memory management for tensor workloads, defragmentation,
and thread-safe operations via locking.
"""

from __future__ import annotations

import mmap
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class AllocationStrategy(Enum):
    """Memory allocation strategy."""
    FIRST_FIT = auto()
    BEST_FIT = auto()


@dataclass
class MemoryBlock:
    """Represents a contiguous block within a memory pool.

    Attributes:
        address: Starting offset within the pool (bytes).
        size: Size of the block (bytes).
        is_free: Whether the block is available for allocation.
        metadata: Arbitrary metadata attached by the allocator or caller.
        alloc_time: Monotonic timestamp when the block was allocated.
        tag: Optional human-readable label for debugging.
    """
    address: int
    size: int
    is_free: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    alloc_time: float = 0.0
    tag: str = ""


class MemoryPool:
    """Fixed-size memory pool with block-level allocation and coalescing.

    The pool manages a contiguous region of *logical* memory, subdivided into
    variable-sized blocks.  Freed blocks are eagerly merged with adjacent free
    neighbours to combat external fragmentation.

    Thread-safety: every public method acquires ``_lock`` before mutating state.
    """

    def __init__(
        self,
        total_size: int,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
        alignment: int = 64,
    ) -> None:
        if total_size <= 0:
            raise ValueError("total_size must be positive")
        self._total_size = total_size
        self._strategy = strategy
        self._alignment = alignment
        self._lock = threading.Lock()
        # Start with a single free block spanning the entire pool.
        self._blocks: list[MemoryBlock] = [MemoryBlock(address=0, size=total_size)]
        self._alloc_count = 0
        self._free_count = 0
        self._peak_usage = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, size: int, tag: str = "") -> Optional[MemoryBlock]:
        """Allocate *size* bytes, returning the block or ``None``."""
        aligned = self._align_up(size)
        with self._lock:
            idx = self._find_block(aligned)
            if idx is None:
                return None
            block = self._blocks[idx]
            remainder = block.size - aligned
            # Replace the free block with an allocated one.
            allocated = MemoryBlock(
                address=block.address,
                size=aligned,
                is_free=False,
                alloc_time=time.monotonic(),
                tag=tag,
            )
            if remainder > 0:
                leftover = MemoryBlock(address=block.address + aligned, size=remainder)
                self._blocks[idx: idx + 1] = [allocated, leftover]
            else:
                self._blocks[idx] = allocated
            self._alloc_count += 1
            self._peak_usage = max(self._peak_usage, self.used)
            return allocated

    def free(self, address: int) -> bool:
        """Free the block at *address*.  Returns ``True`` on success."""
        with self._lock:
            for i, blk in enumerate(self._blocks):
                if blk.address == address and not blk.is_free:
                    blk.is_free = True
                    blk.metadata.clear()
                    blk.tag = ""
                    self._free_count += 1
                    self._coalesce(i)
                    return True
            return False

    def defragment(self) -> int:
        """Compact free space to the end.  Returns bytes moved."""
        with self._lock:
            moved = 0
            offset = 0
            new_blocks: list[MemoryBlock] = []
            total_free = 0
            for blk in self._blocks:
                if blk.is_free:
                    total_free += blk.size
                else:
                    if blk.address != offset:
                        moved += blk.size
                    blk.address = offset
                    new_blocks.append(blk)
                    offset += blk.size
            if total_free > 0:
                new_blocks.append(MemoryBlock(address=offset, size=total_free))
            self._blocks = new_blocks
            return moved

    @property
    def used(self) -> int:
        return sum(b.size for b in self._blocks if not b.is_free)

    @property
    def free_space(self) -> int:
        return self._total_size - self.used

    def stats(self) -> dict[str, Any]:
        with self._lock:
            free_blocks = [b for b in self._blocks if b.is_free]
            return {
                "total": self._total_size,
                "used": self.used,
                "free": self.free_space,
                "num_blocks": len(self._blocks),
                "num_free_blocks": len(free_blocks),
                "largest_free": max((b.size for b in free_blocks), default=0),
                "fragmentation": 1.0 - (max((b.size for b in free_blocks), default=0) / max(self.free_space, 1)),
                "alloc_count": self._alloc_count,
                "free_count": self._free_count,
                "peak_usage": self._peak_usage,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _align_up(self, size: int) -> int:
        return (size + self._alignment - 1) & ~(self._alignment - 1)

    def _find_block(self, size: int) -> Optional[int]:
        if self._strategy is AllocationStrategy.FIRST_FIT:
            for i, blk in enumerate(self._blocks):
                if blk.is_free and blk.size >= size:
                    return i
            return None
        # BEST_FIT
        best_idx: Optional[int] = None
        best_waste = float("inf")
        for i, blk in enumerate(self._blocks):
            if blk.is_free and blk.size >= size:
                waste = blk.size - size
                if waste < best_waste:
                    best_waste = waste
                    best_idx = i
        return best_idx

    def _coalesce(self, idx: int) -> None:
        """Merge block at *idx* with adjacent free neighbours."""
        # Merge with successor first so indices stay valid.
        while idx + 1 < len(self._blocks) and self._blocks[idx + 1].is_free:
            self._blocks[idx].size += self._blocks[idx + 1].size
            del self._blocks[idx + 1]
        while idx > 0 and self._blocks[idx - 1].is_free:
            self._blocks[idx - 1].size += self._blocks[idx].size
            del self._blocks[idx]
            idx -= 1


class MMapRegion:
    """Thin wrapper around ``mmap`` for memory-mapped I/O."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._mm = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)

    def write(self, offset: int, data: bytes) -> None:
        self._mm.seek(offset)
        self._mm.write(data)

    def read(self, offset: int, length: int) -> bytes:
        self._mm.seek(offset)
        return self._mm.read(length)

    def close(self) -> None:
        self._mm.close()


class TensorMemoryManager:
    """Manages multiple memory pools for tensor workloads.

    Maintains separate pools for different tensor-size classes (small, medium,
    large) so that small activations do not fragment the large-weight pool.
    Tracks tensor lifecycle and provides caching for frequently reused shapes.
    """

    _SIZE_CLASSES = [
        ("small", 16 * 1024 * 1024),        # 16 MiB
        ("medium", 256 * 1024 * 1024),       # 256 MiB
        ("large", 1024 * 1024 * 1024),       # 1 GiB
    ]

    def __init__(self, pool_sizes: Optional[dict[str, int]] = None) -> None:
        sizes = pool_sizes or {name: sz for name, sz in self._SIZE_CLASSES}
        self._pools: dict[str, MemoryPool] = {
            name: MemoryPool(sz) for name, sz in sizes.items()
        }
        self._tensor_map: dict[int, tuple[str, MemoryBlock]] = {}
        self._cache: dict[tuple[int, ...], list[int]] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def alloc_tensor(self, shape: tuple[int, ...], dtype_size: int = 4, tag: str = "") -> Optional[int]:
        """Allocate memory for a tensor.  Returns a tensor handle or ``None``."""
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= dtype_size
        pool_name = self._pick_pool(nbytes)
        blk = self._pools[pool_name].allocate(nbytes, tag=tag)
        if blk is None:
            return None
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            self._tensor_map[tid] = (pool_name, blk)
            blk.metadata["shape"] = shape
            blk.metadata["dtype_size"] = dtype_size
        return tid

    def free_tensor(self, tensor_id: int) -> bool:
        with self._lock:
            entry = self._tensor_map.pop(tensor_id, None)
        if entry is None:
            return False
        pool_name, blk = entry
        return self._pools[pool_name].free(blk.address)

    def pool_stats(self) -> dict[str, dict[str, Any]]:
        return {name: pool.stats() for name, pool in self._pools.items()}

    def _pick_pool(self, nbytes: int) -> str:
        names = list(self._pools.keys())
        for name in names[:-1]:
            if nbytes <= self._pools[name]._total_size // 4:
                return name
        return names[-1]
