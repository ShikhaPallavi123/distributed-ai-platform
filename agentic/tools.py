"""Tool system for agentic AI with registry, validation, and built-in tools.

Provides a base Tool class, a ToolRegistry for discovery, and several
built-in tools (Calculator, WebSearch, CodeExecutor, FileReader, DataAnalyzer).
"""

from __future__ import annotations

import ast
import math
import operator
import os
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParameterSchema:
    """Schema for a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "list"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    success: bool = True


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__.lower()

    @abstractmethod
    def parameters(self) -> List[ParameterSchema]:
        """Return the parameter schema for this tool."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        ...

    def validate(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against the parameter schema. Returns error message or None."""
        schema = {p.name: p for p in self.parameters()}
        for pname, pschema in schema.items():
            if pschema.required and pname not in kwargs:
                return f"Missing required parameter: {pname}"
        for key in kwargs:
            if key not in schema:
                return f"Unknown parameter: {key}"
        return None

    def safe_execute(self, **kwargs: Any) -> ToolCall:
        """Execute with validation and error handling, returning a ToolCall record."""
        call = ToolCall(tool_name=self.name, arguments=kwargs)
        error = self.validate(kwargs)
        if error:
            call.error = error
            call.success = False
            return call
        try:
            call.result = self.execute(**kwargs)
        except Exception as exc:
            call.error = str(exc)
            call.success = False
        return call


class ToolRegistry:
    """Registry for discovering and managing available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their schemas."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in t.parameters()
                ],
            }
            for t in self._tools.values()
        ]

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.AST) -> float:
    """Safely evaluate a numeric AST expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](float(_safe_eval_expr(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return float(_SAFE_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        allowed = {"sqrt": math.sqrt, "abs": abs, "log": math.log, "sin": math.sin, "cos": math.cos}
        fn = allowed.get(node.func.id)
        if fn and len(node.args) == 1:
            return float(fn(_safe_eval_expr(node.args[0])))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class Calculator(Tool):
    """Safe arithmetic expression evaluator."""
    name = "calculator"
    description = "Evaluate a mathematical expression safely (supports +, -, *, /, **, %, sqrt, abs, log, sin, cos)."

    def parameters(self) -> List[ParameterSchema]:
        return [ParameterSchema("expression", "string", "Math expression to evaluate")]

    def execute(self, expression: str = "") -> float:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval_expr(tree.body)


class WebSearch(Tool):
    """Simulated web search tool for testing agent workflows."""
    name = "web_search"
    description = "Search the web for information (simulated for testing)."

    _MOCK_DATA: Dict[str, str] = {
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "machine learning": "Machine learning is a subset of AI that learns from data.",
        "weather": "Current weather: 72F, partly cloudy.",
    }

    def parameters(self) -> List[ParameterSchema]:
        return [ParameterSchema("query", "string", "Search query")]

    def execute(self, query: str = "") -> str:
        query_lower = query.lower()
        for key, result in self._MOCK_DATA.items():
            if key in query_lower:
                return result
        return f"Search results for '{query}': No specific results found. Try a more specific query."


class CodeExecutor(Tool):
    """Execute simple Python expressions in a sandboxed AST evaluator."""
    name = "code_executor"
    description = "Execute a simple Python expression and return the result."

    def parameters(self) -> List[ParameterSchema]:
        return [ParameterSchema("code", "string", "Python expression to evaluate")]

    def execute(self, code: str = "") -> Any:
        # Only allow simple expressions, not statements
        tree = ast.parse(code, mode="eval")
        # Walk the AST and reject anything dangerous
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call,)):
                if isinstance(node.func, ast.Name) and node.func.id not in (
                    "len", "range", "list", "dict", "set", "tuple", "sorted",
                    "min", "max", "sum", "abs", "round", "int", "float", "str",
                ):
                    raise ValueError(f"Function not allowed: {node.func.id}")
        return eval(compile(tree, "<agent>", "eval"), {"__builtins__": {}})


class FileReader(Tool):
    """Read file contents (with size limit for safety)."""
    name = "file_reader"
    description = "Read the contents of a text file."

    MAX_SIZE = 100_000  # bytes

    def parameters(self) -> List[ParameterSchema]:
        return [
            ParameterSchema("path", "string", "File path to read"),
            ParameterSchema("max_lines", "number", "Maximum lines to return", required=False, default=100),
        ]

    def execute(self, path: str = "", max_lines: int = 100) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        size = os.path.getsize(path)
        if size > self.MAX_SIZE:
            raise ValueError(f"File too large ({size} bytes, limit {self.MAX_SIZE})")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"... (truncated at {max_lines} lines)")
                    break
                lines.append(line.rstrip())
        return "\n".join(lines)


class DataAnalyzer(Tool):
    """Compute basic statistics on a list of numbers."""
    name = "data_analyzer"
    description = "Compute descriptive statistics (mean, median, stdev, min, max) on numeric data."

    def parameters(self) -> List[ParameterSchema]:
        return [ParameterSchema("data", "list", "List of numbers to analyze")]

    def execute(self, data: list = ()) -> Dict[str, float]:  # type: ignore[assignment]
        nums = [float(x) for x in data]
        if not nums:
            raise ValueError("Data list is empty")
        return {
            "count": len(nums),
            "mean": statistics.mean(nums),
            "median": statistics.median(nums),
            "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0,
            "min": min(nums),
            "max": max(nums),
            "sum": sum(nums),
        }


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    for tool_cls in (Calculator, WebSearch, CodeExecutor, FileReader, DataAnalyzer):
        registry.register(tool_cls())
    return registry
