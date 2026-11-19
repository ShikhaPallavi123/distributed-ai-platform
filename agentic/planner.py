"""Task planning and decomposition for agentic AI.

Provides plan datastructures, a task planner that decomposes goals into
sub-tasks, a hierarchical planner, DAG-based execution ordering,
and replanning on failure.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a planned task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single task within a plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite tasks
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    estimated_cost: float = 1.0
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    subtasks: List[Task] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class Plan:
    """An execution plan consisting of ordered tasks."""
    goal: str
    tasks: List[Task] = field(default_factory=list)
    estimated_total_cost: float = 0.0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for t in self.tasks)

    @property
    def failed_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def summary(self) -> str:
        lines = [f"Plan: {self.goal}", f"Tasks: {len(self.tasks)}, Cost: {self.estimated_total_cost:.1f}"]
        for t in self.tasks:
            deps = f" (after: {', '.join(t.dependencies)})" if t.dependencies else ""
            lines.append(f"  [{t.status.value}] {t.name}{deps}")
        return "\n".join(lines)


class TaskPlanner:
    """Decomposes a complex goal into executable sub-tasks.

    Uses heuristics and keyword analysis to break goals into
    ordered steps that an agent can execute sequentially.
    """

    # Mapping of keywords to common task templates
    TASK_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
        "research": [
            {"name": "search_information", "description": "Search for relevant information", "tool": "web_search"},
            {"name": "analyze_findings", "description": "Analyze search results", "tool": "data_analyzer"},
            {"name": "summarize", "description": "Summarize findings", "tool": None},
        ],
        "calculate": [
            {"name": "parse_expression", "description": "Parse the mathematical expression", "tool": None},
            {"name": "compute", "description": "Perform the calculation", "tool": "calculator"},
        ],
        "analyze": [
            {"name": "gather_data", "description": "Collect relevant data", "tool": "file_reader"},
            {"name": "process_data", "description": "Process and analyze the data", "tool": "data_analyzer"},
            {"name": "interpret_results", "description": "Interpret the analysis", "tool": None},
        ],
        "code": [
            {"name": "understand_requirements", "description": "Parse requirements", "tool": None},
            {"name": "write_code", "description": "Write the code solution", "tool": "code_executor"},
            {"name": "verify", "description": "Verify the solution", "tool": "code_executor"},
        ],
    }

    def __init__(self, available_tools: Optional[List[str]] = None) -> None:
        self.available_tools = set(available_tools or [])

    def plan(self, goal: str) -> Plan:
        """Create a plan to achieve the given goal.

        Args:
            goal: Natural language description of the goal.

        Returns:
            A Plan with ordered tasks.
        """
        tasks = self._decompose(goal)
        plan = Plan(goal=goal, tasks=tasks)
        plan.estimated_total_cost = sum(t.estimated_cost for t in tasks)
        self._validate_plan(plan)
        return plan

    def _decompose(self, goal: str) -> List[Task]:
        """Decompose a goal into tasks using template matching."""
        goal_lower = goal.lower()

        # Find matching template
        for keyword, templates in self.TASK_TEMPLATES.items():
            if keyword in goal_lower:
                return self._tasks_from_template(templates, goal)

        # Default decomposition
        return [
            Task(name="analyze_goal", description=f"Understand: {goal}", estimated_cost=0.5),
            Task(name="execute", description=f"Execute: {goal}", estimated_cost=1.0),
            Task(name="verify", description=f"Verify result for: {goal}", estimated_cost=0.5),
        ]

    def _tasks_from_template(self, templates: List[Dict[str, str]], goal: str) -> List[Task]:
        tasks: List[Task] = []
        prev_id: Optional[str] = None
        for tmpl in templates:
            task = Task(
                name=tmpl["name"],
                description=tmpl["description"],
                tool_name=tmpl.get("tool"),
                estimated_cost=1.0,
                dependencies=[prev_id] if prev_id else [],
            )
            tasks.append(task)
            prev_id = task.id
        return tasks

    def _validate_plan(self, plan: Plan) -> None:
        """Validate plan structure and dependencies."""
        task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    plan.validation_errors.append(f"Task '{task.name}' depends on unknown task '{dep}'")
                    plan.is_valid = False
            if task.tool_name and self.available_tools and task.tool_name not in self.available_tools:
                plan.validation_errors.append(f"Task '{task.name}' requires unavailable tool '{task.tool_name}'")


class HierarchicalPlanner:
    """Hierarchical planner with plan -> sub-plan -> action decomposition.

    Recursively breaks high-level plans into sub-plans until reaching
    directly executable actions.
    """

    def __init__(self, base_planner: Optional[TaskPlanner] = None, max_depth: int = 3) -> None:
        self.base_planner = base_planner or TaskPlanner()
        self.max_depth = max_depth

    def plan(self, goal: str, depth: int = 0) -> Plan:
        """Create a hierarchical plan.

        Args:
            goal: The high-level goal.
            depth: Current recursion depth.

        Returns:
            Plan with tasks that may contain subtasks.
        """
        top_plan = self.base_planner.plan(goal)

        if depth < self.max_depth:
            for task in top_plan.tasks:
                if not task.tool_name and task.description:
                    sub_plan = self.plan(task.description, depth + 1)
                    task.subtasks = sub_plan.tasks
                    task.estimated_cost = sub_plan.estimated_total_cost

        top_plan.estimated_total_cost = sum(t.estimated_cost for t in top_plan.tasks)
        return top_plan

    def flatten(self, plan: Plan) -> List[Task]:
        """Flatten a hierarchical plan into a linear task list (leaf tasks only)."""
        result: List[Task] = []
        for task in plan.tasks:
            if task.subtasks:
                result.extend(self._flatten_task(task))
            else:
                result.append(task)
        return result

    def _flatten_task(self, task: Task) -> List[Task]:
        leaves: List[Task] = []
        for sub in task.subtasks:
            if sub.subtasks:
                leaves.extend(self._flatten_task(sub))
            else:
                leaves.append(sub)
        return leaves


def topological_sort(tasks: List[Task]) -> List[Task]:
    """Sort tasks in dependency order using Kahn's algorithm.

    Args:
        tasks: List of tasks with dependency relationships.

    Returns:
        Topologically sorted list of tasks.

    Raises:
        ValueError: If the dependency graph has cycles.
    """
    id_to_task = {t.id: t for t in tasks}
    in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
    dependents: Dict[str, List[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep in t.dependencies:
            if dep in id_to_task:
                in_degree[t.id] += 1
                dependents[dep].append(t.id)

    queue: deque = deque(tid for tid, deg in in_degree.items() if deg == 0)
    result: List[Task] = []

    while queue:
        tid = queue.popleft()
        result.append(id_to_task[tid])
        for child_id in dependents[tid]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    if len(result) != len(tasks):
        raise ValueError("Cyclic dependencies detected in task graph")
    return result


def replan_on_failure(planner: TaskPlanner, plan: Plan) -> Plan:
    """Create a revised plan when tasks have failed.

    Keeps completed tasks, removes failed ones, and replans
    the remaining goal based on what has been accomplished.

    Args:
        planner: The planner to use for replanning.
        plan: The current (partially failed) plan.

    Returns:
        A new plan incorporating completed work.
    """
    completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
    failed = plan.failed_tasks

    if not failed:
        return plan

    # Build a modified goal that accounts for completed work
    completed_desc = "; ".join(t.name for t in completed) if completed else "nothing"
    failed_desc = "; ".join(t.name for t in failed)
    new_goal = f"{plan.goal} (completed: {completed_desc}; failed: {failed_desc}; replanning)"

    new_plan = planner.plan(new_goal)

    # Preserve completed tasks at the front
    new_plan.tasks = completed + [
        t for t in new_plan.tasks
        if t.name not in {c.name for c in completed}
    ]
    new_plan.estimated_total_cost = sum(t.estimated_cost for t in new_plan.tasks)
    return new_plan
