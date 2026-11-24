"""Explicit, bounded tool workflows; no shell, arbitrary code or network access."""
import ast
import math
import operator
from inference.grounded import retrieve

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("expression must contain 1–200 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("expression too complex")

    def solve(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            try:
                result = OPS[type(node.op)](solve(node.left), solve(node.right))
            except ArithmeticError as exc:
                raise ValueError("arithmetic error") from exc
        elif isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            result = OPS[type(node.op)](solve(node.operand))
        else:
            raise ValueError("only +, -, *, / and numeric literals are supported")
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("result outside allowed range")
        return result

    return solve(tree.body)


def run_tool_workflow(steps):
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("provide 1–8 explicit tool steps")
    results = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("invalid step")
        name = step.get("tool")
        args = step.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        if name == "calculator":
            output = calculate(args.get("expression"))
        elif name == "retrieve":
            output = retrieve(args.get("question"), args.get("documents"), args.get("limit", 3))
        else:
            raise ValueError("tool not allowed")
        results.append({"tool": name, "status": "executed", "output": output})
    return {"status": "success", "steps": results, "result": results[-1]["output"],
            "mode": "explicit_tool_workflow_not_autonomous_agent"}
