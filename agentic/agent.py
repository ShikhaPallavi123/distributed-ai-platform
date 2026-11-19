"""Core agent framework with perception-reasoning-action loop and ReAct pattern.

Defines the Agent base class with a structured cognitive loop, state management,
and the ReactAgent implementing the Reasoning + Acting (ReAct) paradigm used
in modern LLM agent systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agentic.memory import WorkingMemory
from agentic.reasoning import ChainOfThought
from agentic.tools import Tool, ToolCall, ToolRegistry


class AgentState(Enum):
    """Lifecycle states of an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    name: str = "Agent"
    system_prompt: str = "You are a helpful AI agent."
    max_steps: int = 10
    verbose: bool = False


@dataclass
class StepRecord:
    """Record of a single agent step."""
    step: int
    state: AgentState
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


class Agent:
    """Base agent with a perception-reasoning-action cognitive loop.

    Implements the core agent cycle:
      1. Perceive -- observe the environment / tool results
      2. Think -- reason about observations using the reasoning engine
      3. Act -- select and invoke a tool
      4. Observe -- process the tool's output

    Subclasses override think/act/observe for custom behavior.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[WorkingMemory] = None,
        reasoning: Optional[ChainOfThought] = None,
    ) -> None:
        self.config = config
        self.tools = tools or ToolRegistry()
        self.memory = memory or WorkingMemory()
        self.reasoning = reasoning or ChainOfThought()
        self.state = AgentState.IDLE
        self.history: List[StepRecord] = []
        self._goal: str = ""

    @property
    def name(self) -> str:
        return self.config.name

    def think(self, observation: str) -> str:
        """Produce reasoning from an observation.

        Args:
            observation: Latest observation or input text.

        Returns:
            Reasoning string describing what to do next.
        """
        self.state = AgentState.THINKING
        context = self.memory.build_context(query=observation)
        trace = self.reasoning.reason(
            question=f"Given the goal '{self._goal}', what should I do next? Observation: {observation}",
            context=context,
        )
        return trace.conclusion

    def act(self, reasoning: str) -> Optional[ToolCall]:
        """Select and execute a tool based on reasoning.

        Args:
            reasoning: The agent's reasoning about what tool to use.

        Returns:
            ToolCall result, or None if no suitable tool is found.
        """
        self.state = AgentState.ACTING
        # Simple keyword-based tool selection
        tool = self._select_tool(reasoning)
        if tool is None:
            return None

        args = self._extract_args(reasoning, tool)
        return tool.safe_execute(**args)

    def observe(self, result: Optional[ToolCall]) -> str:
        """Process tool output into an observation string.

        Args:
            result: The ToolCall from the act step.

        Returns:
            Observation string to feed back into thinking.
        """
        self.state = AgentState.OBSERVING
        if result is None:
            obs = "No tool was executed."
        elif result.success:
            obs = f"Tool '{result.tool_name}' returned: {result.result}"
        else:
            obs = f"Tool '{result.tool_name}' failed: {result.error}"
        self.memory.add_observation(obs)
        return obs

    def run(self, goal: str, max_steps: Optional[int] = None) -> List[StepRecord]:
        """Execute the full agent loop for a given goal.

        Args:
            goal: The agent's objective.
            max_steps: Override for maximum steps.

        Returns:
            List of StepRecord documenting each cycle.
        """
        self._goal = goal
        self.state = AgentState.IDLE
        self.history.clear()
        steps = max_steps or self.config.max_steps

        observation = f"Starting goal: {goal}"
        self.memory.add_observation(observation, persist=True)

        for step_num in range(1, steps + 1):
            try:
                thought = self.think(observation)
                action = self.act(thought)
                observation = self.observe(action)

                record = StepRecord(
                    step=step_num,
                    state=self.state,
                    thought=thought,
                    action=action,
                    observation=observation,
                )
                self.history.append(record)

                if self.config.verbose:
                    print(f"[{self.name}] Step {step_num}: {thought[:80]}...")

                if self._is_done(thought, observation):
                    self.state = AgentState.DONE
                    break

            except Exception as exc:
                self.state = AgentState.ERROR
                self.history.append(StepRecord(
                    step=step_num, state=AgentState.ERROR,
                    thought=f"Error: {exc}", observation=str(exc),
                ))
                break

        if self.state != AgentState.DONE:
            self.state = AgentState.DONE
        return self.history

    def _select_tool(self, reasoning: str) -> Optional[Tool]:
        """Select a tool based on reasoning text (keyword matching)."""
        reasoning_lower = reasoning.lower()
        for name in self.tools.tool_names:
            if name.replace("_", " ") in reasoning_lower or name in reasoning_lower:
                return self.tools.get(name)
        # Fallback: return first tool if any exist
        if self.tools.tool_names:
            return self.tools.get(self.tools.tool_names[0])
        return None

    def _extract_args(self, reasoning: str, tool: Tool) -> Dict[str, Any]:
        """Extract tool arguments from reasoning text (simple heuristic)."""
        params = tool.parameters()
        args: Dict[str, Any] = {}
        for param in params:
            if param.type == "string" and param.required:
                # Use the goal or reasoning as the argument
                args[param.name] = self._goal
                break
        return args

    def _is_done(self, thought: str, observation: str) -> bool:
        """Check if the agent should stop."""
        done_signals = ["done", "complete", "finished", "answer found", "goal achieved"]
        combined = (thought + " " + observation).lower()
        return any(signal in combined for signal in done_signals)


class ReactAgent(Agent):
    """Agent implementing the ReAct (Reasoning + Acting) pattern.

    At each step, the agent explicitly generates:
      - Thought: reasoning about the current state
      - Action: which tool to call and with what arguments
      - Observation: the result of the action

    This produces an interpretable trace of the agent's decision-making.
    """

    def __init__(self, config: AgentConfig, tools: Optional[ToolRegistry] = None,
                 memory: Optional[WorkingMemory] = None) -> None:
        super().__init__(config, tools, memory)
        self._react_traces: List[Dict[str, str]] = []

    def think(self, observation: str) -> str:
        """ReAct-style thinking: produce explicit thought with action intent."""
        self.state = AgentState.THINKING
        context = self.memory.build_context(query=observation)
        available_tools = ", ".join(self.tools.tool_names) if self.tools.tool_names else "none"

        thought = (
            f"Goal: {self._goal}\n"
            f"Observation: {observation}\n"
            f"Available tools: {available_tools}\n"
            f"Thought: I need to analyze this observation and decide the next action.\n"
            f"Context summary length: {len(context)} chars"
        )

        self._react_traces.append({"thought": thought, "observation": observation})
        return thought

    def act(self, reasoning: str) -> Optional[ToolCall]:
        """ReAct-style action: explicitly select tool from reasoning."""
        self.state = AgentState.ACTING
        # Parse tool name from reasoning
        tool = self._select_tool(reasoning)
        if tool is None:
            return ToolCall(tool_name="none", arguments={}, result="No applicable tool found", success=True)

        args = self._extract_args(reasoning, tool)
        call = tool.safe_execute(**args)

        if self._react_traces:
            self._react_traces[-1]["action"] = f"{call.tool_name}({call.arguments})"
            self._react_traces[-1]["result"] = str(call.result) if call.success else str(call.error)

        return call

    @property
    def react_trace(self) -> List[Dict[str, str]]:
        """Return the full ReAct trace for inspection."""
        return list(self._react_traces)
