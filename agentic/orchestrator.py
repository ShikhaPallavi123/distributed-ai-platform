"""Multi-agent orchestration with supervisor delegation, debate protocol, and consensus.

Manages multiple specialized agents, routes tasks to the right specialist,
supports multi-agent deliberation via structured debate, and resolves
conflicting outputs through voting or weighted consensus.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from agentic.agent import Agent, AgentConfig, AgentState, ReactAgent, StepRecord
from agentic.memory import WorkingMemory
from agentic.tools import ToolRegistry, create_default_registry


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DEBATE_ARGUMENT = "debate_argument"
    VOTE = "vote"


@dataclass
class AgentMessage:
    """A message passed between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    recipient: str = ""  # empty = broadcast
    msg_type: MessageType = MessageType.TASK
    content: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None


@dataclass
class AgentHandle:
    """Handle to a managed agent with metadata."""
    agent: Agent
    specialization: str = ""
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True
    tasks_completed: int = 0
    total_steps: int = 0


class MultiAgentOrchestrator:
    """Manages a pool of specialized agents, routing tasks and collecting results.

    Agents register with their specializations and capabilities. The orchestrator
    matches incoming tasks to the best-suited agent and handles lifecycle management.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentHandle] = {}
        self._message_log: List[AgentMessage] = []
        self._results: Dict[str, Any] = {}

    def register_agent(self, agent: Agent, specialization: str = "",
                       capabilities: Optional[List[str]] = None) -> str:
        """Register an agent with the orchestrator.

        Returns:
            The agent's name (used as its ID).
        """
        handle = AgentHandle(
            agent=agent,
            specialization=specialization,
            capabilities=capabilities or [],
        )
        self._agents[agent.name] = handle
        return agent.name

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the pool."""
        handle = self._agents.pop(name, None)
        if handle:
            handle.is_active = False
            return True
        return False

    def list_agents(self) -> List[Dict[str, Any]]:
        """List registered agents with their status."""
        return [
            {
                "name": h.agent.name,
                "specialization": h.specialization,
                "capabilities": h.capabilities,
                "active": h.is_active,
                "tasks_completed": h.tasks_completed,
            }
            for h in self._agents.values()
        ]

    def send_message(self, msg: AgentMessage) -> None:
        """Send a message between agents (logged for audit)."""
        self._message_log.append(msg)

    def route_task(self, task: str, required_capability: Optional[str] = None) -> Optional[str]:
        """Find the best agent for a task.

        Args:
            task: Task description.
            required_capability: If specified, only agents with this capability.

        Returns:
            Name of the selected agent, or None.
        """
        candidates = []
        task_lower = task.lower()
        for name, handle in self._agents.items():
            if not handle.is_active:
                continue
            if required_capability and required_capability not in handle.capabilities:
                continue
            # Score by keyword overlap between task and specialization/capabilities
            score = 0
            if handle.specialization.lower() in task_lower:
                score += 5
            score += sum(1 for cap in handle.capabilities if cap.lower() in task_lower)
            candidates.append((name, score))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def execute_task(self, task: str, agent_name: Optional[str] = None,
                     max_steps: int = 5) -> Dict[str, Any]:
        """Execute a task on a specific or auto-selected agent.

        Returns:
            Dict with agent name, steps taken, and final result.
        """
        name = agent_name or self.route_task(task)
        if name is None or name not in self._agents:
            return {"error": "No suitable agent found", "task": task}

        handle = self._agents[name]
        history = handle.agent.run(task, max_steps=max_steps)
        handle.tasks_completed += 1
        handle.total_steps += len(history)

        result = {
            "agent": name,
            "task": task,
            "steps": len(history),
            "final_state": handle.agent.state.value,
            "history": history,
        }
        self._results[task[:50]] = result
        return result

    def broadcast(self, content: str, sender: str = "orchestrator") -> List[AgentMessage]:
        """Send a message to all active agents."""
        messages = []
        for name, handle in self._agents.items():
            if handle.is_active:
                msg = AgentMessage(sender=sender, recipient=name,
                                   msg_type=MessageType.BROADCAST, content=content)
                self.send_message(msg)
                messages.append(msg)
        return messages

    @property
    def message_history(self) -> List[AgentMessage]:
        return list(self._message_log)

    @property
    def active_agent_count(self) -> int:
        return sum(1 for h in self._agents.values() if h.is_active)


class Supervisor:
    """A supervisor agent that delegates tasks to specialist agents.

    Analyzes incoming goals, decomposes them if needed, and routes
    sub-tasks to the most appropriate specialist via the orchestrator.
    """

    def __init__(self, orchestrator: MultiAgentOrchestrator, name: str = "Supervisor") -> None:
        self.orchestrator = orchestrator
        self.name = name
        self._delegation_log: List[Dict[str, Any]] = []

    def delegate(self, goal: str, max_steps: int = 5) -> Dict[str, Any]:
        """Delegate a goal to the best specialist agent.

        Args:
            goal: The task to delegate.
            max_steps: Maximum steps for the agent.

        Returns:
            Execution result from the agent.
        """
        agent_name = self.orchestrator.route_task(goal)
        result = self.orchestrator.execute_task(goal, agent_name=agent_name, max_steps=max_steps)
        self._delegation_log.append({"goal": goal, "agent": agent_name, "result_summary": result.get("final_state")})
        return result

    def delegate_parallel(self, goals: List[str], max_steps: int = 5) -> List[Dict[str, Any]]:
        """Delegate multiple goals (sequentially in this implementation)."""
        return [self.delegate(goal, max_steps) for goal in goals]

    @property
    def delegation_history(self) -> List[Dict[str, Any]]:
        return list(self._delegation_log)


@dataclass
class DebateArgument:
    """An argument in a multi-agent debate."""
    agent_name: str
    position: str
    reasoning: str
    confidence: float = 0.5
    round_num: int = 0
    rebuts: Optional[str] = None  # position it's responding to


class DebateProtocol:
    """Structured multi-agent debate for deliberation.

    Agents take positions on a question, present arguments, and
    can rebut each other's positions across multiple rounds.
    """

    def __init__(self, question: str, max_rounds: int = 3) -> None:
        self.question = question
        self.max_rounds = max_rounds
        self.arguments: List[DebateArgument] = []
        self._round = 0

    def add_argument(self, agent_name: str, position: str, reasoning: str,
                     confidence: float = 0.5, rebuts: Optional[str] = None) -> DebateArgument:
        """Record an agent's argument."""
        arg = DebateArgument(
            agent_name=agent_name,
            position=position,
            reasoning=reasoning,
            confidence=confidence,
            round_num=self._round,
            rebuts=rebuts,
        )
        self.arguments.append(arg)
        return arg

    def next_round(self) -> int:
        """Advance to the next debate round."""
        self._round += 1
        return self._round

    def get_positions(self) -> Dict[str, List[DebateArgument]]:
        """Group arguments by position."""
        positions: Dict[str, List[DebateArgument]] = {}
        for arg in self.arguments:
            positions.setdefault(arg.position, []).append(arg)
        return positions

    @property
    def current_round(self) -> int:
        return self._round

    def summary(self) -> str:
        positions = self.get_positions()
        lines = [f"Debate: {self.question}", f"Rounds: {self._round}, Arguments: {len(self.arguments)}"]
        for pos, args in positions.items():
            avg_conf = sum(a.confidence for a in args) / len(args)
            lines.append(f"  Position '{pos}': {len(args)} arguments, avg confidence={avg_conf:.2f}")
        return "\n".join(lines)


def majority_vote(arguments: List[DebateArgument]) -> str:
    """Resolve a debate by simple majority vote on positions."""
    if not arguments:
        return ""
    counter = Counter(a.position for a in arguments)
    return counter.most_common(1)[0][0]


def weighted_consensus(arguments: List[DebateArgument]) -> str:
    """Resolve a debate by confidence-weighted voting."""
    if not arguments:
        return ""
    scores: Dict[str, float] = {}
    for arg in arguments:
        scores[arg.position] = scores.get(arg.position, 0.0) + arg.confidence
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def spawn_agent(
    name: str,
    specialization: str,
    tools: Optional[ToolRegistry] = None,
    system_prompt: str = "",
) -> Agent:
    """Factory function to create and configure a new agent.

    Args:
        name: Agent name / identifier.
        specialization: Domain specialization description.
        tools: Optional tool registry (default tools if None).
        system_prompt: Custom system prompt.

    Returns:
        Configured ReactAgent instance.
    """
    config = AgentConfig(
        name=name,
        system_prompt=system_prompt or f"You are a specialist in {specialization}.",
    )
    return ReactAgent(config=config, tools=tools or create_default_registry())
