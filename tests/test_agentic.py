"""Tests for the Agentic AI module."""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestTools(unittest.TestCase):
    def test_tool_registry(self):
        from agentic.tools import ToolRegistry, Calculator
        registry = ToolRegistry()
        calc = Calculator()
        registry.register(calc)
        self.assertIn("calculator", registry.tool_names)
        tool = registry.get("calculator")
        self.assertIsNotNone(tool)

    def test_calculator_tool(self):
        from agentic.tools import Calculator
        calc = Calculator()
        result = calc.execute(expression="2 + 3")
        self.assertIsNotNone(result)

    def test_default_registry(self):
        from agentic.tools import create_default_registry
        registry = create_default_registry()
        tools = registry.list_tools()
        self.assertGreater(len(tools), 0)
        self.assertIsInstance(tools[0], dict)

    def test_list_tools_returns_dicts(self):
        from agentic.tools import ToolRegistry, Calculator
        registry = ToolRegistry()
        registry.register(Calculator())
        tools = registry.list_tools()
        self.assertIsInstance(tools[0], dict)
        self.assertIn("name", tools[0])


class TestAgent(unittest.TestCase):
    def test_agent_creation(self):
        from agentic.agent import Agent, AgentConfig
        config = AgentConfig(name="test-agent")
        agent = Agent(config)
        self.assertIsNotNone(agent)

    def test_react_agent(self):
        from agentic.agent import ReactAgent, AgentConfig
        from agentic.tools import create_default_registry
        config = AgentConfig(name="react-test", max_steps=3)
        agent = ReactAgent(config, tools=create_default_registry())
        result = agent.run("What is 2 + 2?")
        self.assertIsNotNone(result)


class TestPlanner(unittest.TestCase):
    def test_task_planner(self):
        from agentic.planner import TaskPlanner
        planner = TaskPlanner()
        plan = planner.plan("Build a web application")
        self.assertIsNotNone(plan)

    def test_hierarchical_planner(self):
        from agentic.planner import HierarchicalPlanner
        planner = HierarchicalPlanner()
        plan = planner.plan("Design and deploy a system")
        self.assertIsNotNone(plan)


class TestMemory(unittest.TestCase):
    def test_short_term_memory(self):
        from agentic.memory import ShortTermMemory
        mem = ShortTermMemory(capacity=5)
        mem.add("item 1")
        mem.add("item 2")
        self.assertEqual(len(mem), 2)

    def test_working_memory(self):
        from agentic.memory import WorkingMemory
        wm = WorkingMemory()
        wm.set_scratch("key1", "value1")
        self.assertEqual(wm.get_scratch("key1"), "value1")

    def test_episodic_memory(self):
        from agentic.memory import EpisodicMemory
        em = EpisodicMemory(max_episodes=10)
        self.assertIsNotNone(em)


class TestReasoning(unittest.TestCase):
    def test_chain_of_thought(self):
        from agentic.reasoning import ChainOfThought
        cot = ChainOfThought()
        result = cot.reason("What is 2+2?")
        self.assertIsNotNone(result)

    def test_tree_of_thought(self):
        from agentic.reasoning import TreeOfThought
        tot = TreeOfThought(max_depth=2, branching_factor=2)
        result = tot.explore("How to optimize?")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
