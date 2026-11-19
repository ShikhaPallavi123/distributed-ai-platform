"""Inter-process communication primitives.

Provides a pub/sub message bus, shared-memory zero-copy channels, streaming
pipeline channels with back-pressure, and a compact binary serialization
format inspired by msgpack.
"""

from __future__ import annotations

import mmap
import struct
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# ======================================================================
# Compact binary serialization (msgpack-like, stdlib only)
# ======================================================================

def _encode(obj: Any) -> bytes:
    """Encode a Python object into a compact binary format."""
    if obj is None:
        return b"\xc0"
    if isinstance(obj, bool):
        return b"\xc3" if obj else b"\xc2"
    if isinstance(obj, int):
        return b"\xd3" + struct.pack(">q", obj)
    if isinstance(obj, float):
        return b"\xcb" + struct.pack(">d", obj)
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        return b"\xdb" + struct.pack(">I", len(raw)) + raw
    if isinstance(obj, bytes):
        return b"\xc6" + struct.pack(">I", len(obj)) + obj
    if isinstance(obj, (list, tuple)):
        parts = b"".join(_encode(item) for item in obj)
        return b"\xdd" + struct.pack(">I", len(obj)) + parts
    if isinstance(obj, dict):
        parts = b"".join(_encode(k) + _encode(v) for k, v in obj.items())
        return b"\xdf" + struct.pack(">I", len(obj)) + parts
    raise TypeError(f"unsupported type: {type(obj)}")


def _decode(data: bytes, offset: int = 0) -> tuple[Any, int]:
    """Decode one value from *data* at *offset*.  Returns (value, new_offset)."""
    tag = data[offset]
    offset += 1
    if tag == 0xC0:
        return None, offset
    if tag == 0xC2:
        return False, offset
    if tag == 0xC3:
        return True, offset
    if tag == 0xD3:
        return struct.unpack_from(">q", data, offset)[0], offset + 8
    if tag == 0xCB:
        return struct.unpack_from(">d", data, offset)[0], offset + 8
    if tag == 0xDB:
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        return data[offset: offset + length].decode("utf-8"), offset + length
    if tag == 0xC6:
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        return data[offset: offset + length], offset + length
    if tag == 0xDD:
        count = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        items: list[Any] = []
        for _ in range(count):
            val, offset = _decode(data, offset)
            items.append(val)
        return items, offset
    if tag == 0xDF:
        count = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        d: dict[Any, Any] = {}
        for _ in range(count):
            k, offset = _decode(data, offset)
            v, offset = _decode(data, offset)
            d[k] = v
        return d, offset
    raise ValueError(f"unknown tag 0x{tag:02x} at offset {offset - 1}")


# ======================================================================
# Message types
# ======================================================================

class MessageType(Enum):
    DATA = auto()
    REQUEST = auto()
    REPLY = auto()
    BROADCAST = auto()
    CONTROL = auto()


@dataclass
class Message:
    """Wire-level message with headers, payload, and routing information."""
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    msg_type: MessageType = MessageType.DATA
    topic: str = ""
    sender: str = ""
    reply_to: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    headers: dict[str, str] = field(default_factory=dict)
    payload: Any = None

    def encode(self) -> bytes:
        return _encode({
            "id": self.msg_id,
            "type": self.msg_type.value,
            "topic": self.topic,
            "sender": self.sender,
            "reply_to": self.reply_to,
            "ts": self.timestamp,
            "headers": self.headers,
            "payload": self.payload,
        })

    @classmethod
    def decode(cls, data: bytes) -> "Message":
        obj, _ = _decode(data)
        return cls(
            msg_id=obj["id"],
            msg_type=MessageType(obj["type"]),
            topic=obj["topic"],
            sender=obj["sender"],
            reply_to=obj["reply_to"],
            timestamp=obj["ts"],
            headers=obj["headers"],
            payload=obj["payload"],
        )


# ======================================================================
# MessageBus - pub/sub + request/reply
# ======================================================================

Subscriber = Callable[[Message], None]


class MessageBus:
    """Thread-safe in-process message bus with pub/sub, request/reply,
    and broadcast patterns.
    """

    def __init__(self, max_pending: int = 1024) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._reply_waiters: dict[str, threading.Event] = {}
        self._replies: dict[str, Message] = {}
        self._pending_counts: dict[str, int] = defaultdict(int)
        self._max_pending = max_pending
        self._total_published = 0
        self._total_dropped = 0

    def subscribe(self, topic: str, handler: Subscriber) -> None:
        with self._lock:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Subscriber) -> bool:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if handler in subs:
                subs.remove(handler)
                return True
            return False

    def publish(self, msg: Message) -> bool:
        """Publish a message.  Returns ``False`` if back-pressure drops it."""
        with self._lock:
            if self._pending_counts[msg.topic] >= self._max_pending:
                self._total_dropped += 1
                return False
            self._pending_counts[msg.topic] += 1
            self._total_published += 1
            handlers = list(self._subscribers.get(msg.topic, []))
        for h in handlers:
            try:
                h(msg)
            except Exception:
                pass
        with self._lock:
            self._pending_counts[msg.topic] -= 1
        return True

    def request(self, msg: Message, timeout: float = 5.0) -> Optional[Message]:
        """Send a request and block until a reply arrives."""
        msg.msg_type = MessageType.REQUEST
        event = threading.Event()
        with self._lock:
            self._reply_waiters[msg.msg_id] = event
        self.publish(msg)
        event.wait(timeout)
        with self._lock:
            self._reply_waiters.pop(msg.msg_id, None)
            return self._replies.pop(msg.msg_id, None)

    def reply(self, original: Message, payload: Any, sender: str = "") -> None:
        """Send a reply to a previous request."""
        rep = Message(
            msg_type=MessageType.REPLY,
            topic=original.topic,
            sender=sender,
            reply_to=original.msg_id,
            payload=payload,
        )
        with self._lock:
            evt = self._reply_waiters.get(original.msg_id)
            if evt is not None:
                self._replies[original.msg_id] = rep
                evt.set()

    def broadcast(self, payload: Any, sender: str = "") -> int:
        """Broadcast to all topics.  Returns the number of topics reached."""
        with self._lock:
            topics = list(self._subscribers.keys())
        count = 0
        for topic in topics:
            msg = Message(msg_type=MessageType.BROADCAST, topic=topic,
                          sender=sender, payload=payload)
            if self.publish(msg):
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "topics": len(self._subscribers),
                "total_published": self._total_published,
                "total_dropped": self._total_dropped,
                "pending_replies": len(self._reply_waiters),
            }


# ======================================================================
# SharedMemoryChannel - zero-copy via mmap
# ======================================================================

class SharedMemoryChannel:
    """Ring-buffer channel backed by ``mmap`` for zero-copy transfer.

    Layout (all big-endian):
        [0..4)   write_offset  (uint32)
        [4..8)   read_offset   (uint32)
        [8..N)   data ring
    """

    _HEADER = 8

    def __init__(self, capacity: int = 1024 * 1024) -> None:
        self._cap = capacity
        self._buf = mmap.mmap(-1, self._HEADER + capacity, access=mmap.ACCESS_WRITE)
        struct.pack_into(">II", self._buf, 0, 0, 0)
        self._wlock = threading.Lock()
        self._rlock = threading.Lock()

    def write(self, data: bytes) -> bool:
        framed = struct.pack(">I", len(data)) + data
        with self._wlock:
            w = struct.unpack_from(">I", self._buf, 0)[0]
            r = struct.unpack_from(">I", self._buf, 4)[0]
            used = (w - r) % self._cap
            if used + len(framed) >= self._cap:
                return False  # back-pressure
            for b in framed:
                self._buf[self._HEADER + (w % self._cap)] = b
                w += 1
            struct.pack_into(">I", self._buf, 0, w)
        return True

    def read(self) -> Optional[bytes]:
        with self._rlock:
            w = struct.unpack_from(">I", self._buf, 0)[0]
            r = struct.unpack_from(">I", self._buf, 4)[0]
            if w == r:
                return None
            length_bytes = bytes(
                self._buf[self._HEADER + ((r + i) % self._cap)] for i in range(4)
            )
            length = struct.unpack(">I", length_bytes)[0]
            r += 4
            payload = bytes(
                self._buf[self._HEADER + ((r + i) % self._cap)] for i in range(length)
            )
            r += length
            struct.pack_into(">I", self._buf, 4, r)
        return payload

    def close(self) -> None:
        self._buf.close()


# ======================================================================
# PipelineChannel - stage-to-stage streaming with back-pressure
# ======================================================================

class PipelineChannel:
    """Bounded channel connecting two pipeline stages.

    The producer blocks when the buffer is full (back-pressure).
    The consumer blocks when the buffer is empty.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._buf: deque[Any] = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self._total_sent = 0
        self._total_recv = 0

    def send(self, item: Any, timeout: Optional[float] = None) -> bool:
        with self._not_full:
            if self._closed:
                return False
            if not self._not_full.wait_for(
                lambda: len(self._buf) < self._maxsize or self._closed,
                timeout=timeout,
            ):
                return False
            if self._closed:
                return False
            self._buf.append(item)
            self._total_sent += 1
            self._not_empty.notify()
        return True

    def recv(self, timeout: Optional[float] = None) -> Optional[Any]:
        with self._not_empty:
            if not self._not_empty.wait_for(
                lambda: len(self._buf) > 0 or self._closed,
                timeout=timeout,
            ):
                return None
            if not self._buf:
                return None
            item = self._buf.popleft()
            self._total_recv += 1
            self._not_full.notify()
        return item

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_full.notify_all()
            self._not_empty.notify_all()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "buffered": len(self._buf),
                "capacity": self._maxsize,
                "total_sent": self._total_sent,
                "total_recv": self._total_recv,
                "closed": self._closed,
            }
