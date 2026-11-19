"""Simplified Raft consensus protocol implementation."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RaftState(Enum):
    """Role a node plays in the Raft cluster."""
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


@dataclass
class LogEntry:
    """A single entry in the replicated log."""
    term: int
    index: int
    command: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class VoteResponse:
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    match_index: int


class RaftNode:
    """A single participant in the Raft consensus protocol.

    Implements leader election, log replication, and commit advancement.
    """

    def __init__(self, node_id: str, peers: list[str] | None = None) -> None:
        self.node_id = node_id
        self.peers: list[str] = list(peers or [])

        # Persistent state
        self.current_term: int = 0
        self.voted_for: str | None = None
        self.log: list[LogEntry] = []

        # Volatile state
        self.state = RaftState.FOLLOWER
        self.commit_index: int = -1
        self.last_applied: int = -1
        self.leader_id: str | None = None

        # Leader-only volatile state
        self.next_index: dict[str, int] = {}
        self.match_index: dict[str, int] = {}

        # Timing
        self._election_timeout: float = random.uniform(1.5, 3.0)
        self._heartbeat_interval: float = 0.5
        self._last_heartbeat: float = time.time()
        self._running = False
        self._task: asyncio.Task[None] | None = None

        # Cluster reference (set by RaftCluster)
        self._cluster: RaftCluster | None = None

    # -- public api -------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._last_heartbeat = time.time()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def propose(self, command: Any) -> LogEntry | None:
        """Propose a new command.  Only the leader can append."""
        if self.state != RaftState.LEADER:
            return None
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command,
        )
        self.log.append(entry)
        # Immediately try replicating
        await self._send_heartbeats()
        return entry

    # -- RPC handlers -----------------------------------------------------

    def handle_vote_request(self, req: VoteRequest) -> VoteResponse:
        if req.term > self.current_term:
            self._become_follower(req.term)

        if req.term < self.current_term:
            return VoteResponse(term=self.current_term, vote_granted=False)

        log_ok = self._candidate_log_up_to_date(
            req.last_log_index, req.last_log_term
        )
        if (self.voted_for is None or self.voted_for == req.candidate_id) and log_ok:
            self.voted_for = req.candidate_id
            self._last_heartbeat = time.time()
            return VoteResponse(term=self.current_term, vote_granted=True)

        return VoteResponse(term=self.current_term, vote_granted=False)

    def handle_append_entries(
        self, req: AppendEntriesRequest
    ) -> AppendEntriesResponse:
        if req.term > self.current_term:
            self._become_follower(req.term)

        if req.term < self.current_term:
            return AppendEntriesResponse(
                term=self.current_term, success=False, match_index=-1
            )

        self._last_heartbeat = time.time()
        self.leader_id = req.leader_id
        if self.state == RaftState.CANDIDATE:
            self._become_follower(req.term)

        # Log consistency check
        if req.prev_log_index >= 0:
            if req.prev_log_index >= len(self.log):
                return AppendEntriesResponse(
                    term=self.current_term, success=False, match_index=len(self.log) - 1
                )
            if self.log[req.prev_log_index].term != req.prev_log_term:
                self.log = self.log[: req.prev_log_index]
                return AppendEntriesResponse(
                    term=self.current_term, success=False, match_index=len(self.log) - 1
                )

        # Append new entries
        for entry in req.entries:
            if entry.index < len(self.log):
                if self.log[entry.index].term != entry.term:
                    self.log = self.log[: entry.index]
                    self.log.append(entry)
            else:
                self.log.append(entry)

        if req.leader_commit > self.commit_index:
            self.commit_index = min(req.leader_commit, len(self.log) - 1)

        return AppendEntriesResponse(
            term=self.current_term, success=True, match_index=len(self.log) - 1
        )

    # -- internal ---------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            if self.state == RaftState.LEADER:
                await self._send_heartbeats()
                await asyncio.sleep(self._heartbeat_interval)
            else:
                await asyncio.sleep(0.1)
                if time.time() - self._last_heartbeat > self._election_timeout:
                    await self._start_election()

    async def _start_election(self) -> None:
        self.state = RaftState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self._last_heartbeat = time.time()
        self._election_timeout = random.uniform(1.5, 3.0)

        votes = 1  # vote for self
        last_idx = len(self.log) - 1
        last_term = self.log[last_idx].term if self.log else 0
        req = VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_idx,
            last_log_term=last_term,
        )

        if self._cluster:
            for peer_id in self.peers:
                peer = self._cluster.get_node(peer_id)
                if peer:
                    resp = peer.handle_vote_request(req)
                    if resp.vote_granted:
                        votes += 1

        majority = (len(self.peers) + 1) // 2 + 1
        if votes >= majority and self.state == RaftState.CANDIDATE:
            self._become_leader()

    def _become_leader(self) -> None:
        self.state = RaftState.LEADER
        self.leader_id = self.node_id
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = -1

    def _become_follower(self, term: int) -> None:
        self.state = RaftState.FOLLOWER
        self.current_term = term
        self.voted_for = None

    async def _send_heartbeats(self) -> None:
        if not self._cluster:
            return
        for peer_id in self.peers:
            peer = self._cluster.get_node(peer_id)
            if not peer:
                continue
            ni = self.next_index.get(peer_id, len(self.log))
            prev_idx = ni - 1
            prev_term = self.log[prev_idx].term if 0 <= prev_idx < len(self.log) else 0
            entries = self.log[ni:]
            req = AppendEntriesRequest(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_idx,
                prev_log_term=prev_term,
                entries=entries,
                leader_commit=self.commit_index,
            )
            resp = peer.handle_append_entries(req)
            if resp.success:
                self.next_index[peer_id] = resp.match_index + 1
                self.match_index[peer_id] = resp.match_index
            elif resp.term > self.current_term:
                self._become_follower(resp.term)
                return
            else:
                self.next_index[peer_id] = max(0, self.next_index.get(peer_id, 1) - 1)

        self._advance_commit_index()

    def _advance_commit_index(self) -> None:
        """Advance commit_index if a majority has replicated."""
        for idx in range(self.commit_index + 1, len(self.log)):
            if self.log[idx].term != self.current_term:
                continue
            replicas = 1  # self
            for peer_id in self.peers:
                if self.match_index.get(peer_id, -1) >= idx:
                    replicas += 1
            majority = (len(self.peers) + 1) // 2 + 1
            if replicas >= majority:
                self.commit_index = idx

    def _candidate_log_up_to_date(self, last_idx: int, last_term: int) -> bool:
        my_last_term = self.log[-1].term if self.log else 0
        my_last_idx = len(self.log) - 1
        if last_term != my_last_term:
            return last_term >= my_last_term
        return last_idx >= my_last_idx


class RaftCluster:
    """Coordinate multiple RaftNodes for testing and in-process consensus."""

    def __init__(self) -> None:
        self._nodes: dict[str, RaftNode] = {}

    def add_node(self, node: RaftNode) -> None:
        node._cluster = self
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> RaftNode | None:
        return self._nodes.get(node_id)

    def leader(self) -> RaftNode | None:
        for n in self._nodes.values():
            if n.state == RaftState.LEADER:
                return n
        return None

    async def start_all(self) -> None:
        for node in self._nodes.values():
            await node.start()

    async def stop_all(self) -> None:
        for node in self._nodes.values():
            await node.stop()
