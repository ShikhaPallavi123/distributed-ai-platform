"""
Example 3: Agentic AI — ReAct agent with tool use.

Demonstrates:
- Creating a ReAct agent
- Registering tools
- Running multi-step reasoning
- Chain-of-thought reasoning
"""

import sys
sys.path.insert(0, '..')

from agentic.tools import create_default_registry  # noqa: E402
from agentic.reasoning import ChainOfThought, TreeOfThought  # noqa: E402


def main():
    print("=" * 50)
    print("NeuroFlux — Agentic AI Example")
    print("=" * 50)

    # --- Chain of Thought ---
    print("\n🧠 Chain-of-Thought Reasoning")
    print("-" * 40)
    cot = ChainOfThought()
    result = cot.reason("What is the area of a circle with radius 5?")
    print("  Question: What is the area of a circle with radius 5?")
    for i, step in enumerate(result.get('steps', [])):
        print(f"  Step {i + 1}: {step}")
    print(f"  Conclusion: {result.get('conclusion', 'N/A')}")

    # --- Tool Registry ---
    print("\n🛠️ Tool Registry")
    print("-" * 40)
    registry = create_default_registry()
    for name in registry.list_tools():
        print(f"  📌 {name}")

    # --- Calculator Tool ---
    print("\n🔢 Calculator Tool")
    print("-" * 40)
    calc = registry.get_tool("calculator")
    if calc:
        result = calc.execute(expression="25 * 4 + 10")
        print(f"  25 * 4 + 10 = {result}")

    # --- Tree of Thought ---
    print("\n🌳 Tree-of-Thought Reasoning")
    print("-" * 40)
    tot = TreeOfThought(branching_factor=3)
    result = tot.reason("How should we optimize inference latency?")
    print(f"  Branches explored: {result.get('branches_explored', 'N/A')}")
    print(f"  Best path: {result.get('best_path', 'N/A')}")

    print("\n✓ Agentic AI demo complete!")


if __name__ == "__main__":
    main()
