"""Reasoning engines for agentic AI: chain-of-thought, tree-of-thought,
reflection, analogical reasoning, and logical inference.

These provide structured reasoning patterns that agents use to process
observations, plan actions, and evaluate outcomes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ThoughtStep:
    """A single step in a reasoning chain."""
    step_number: int
    thought: str
    reasoning_type: str = "deduction"
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    question: str
    steps: List[ThoughtStep] = field(default_factory=list)
    conclusion: str = ""
    overall_confidence: float = 0.0

    def add_step(self, thought: str, reasoning_type: str = "deduction",
                 confidence: float = 1.0, evidence: Optional[List[str]] = None) -> ThoughtStep:
        step = ThoughtStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            reasoning_type=reasoning_type,
            confidence=confidence,
            evidence=evidence or [],
        )
        self.steps.append(step)
        return step

    def finalize(self, conclusion: str) -> None:
        self.conclusion = conclusion
        if self.steps:
            self.overall_confidence = sum(s.confidence for s in self.steps) / len(self.steps)

    def to_string(self) -> str:
        lines = [f"Question: {self.question}"]
        for s in self.steps:
            lines.append(f"Step {s.step_number} [{s.reasoning_type}] (conf={s.confidence:.2f}): {s.thought}")
        lines.append(f"Conclusion (conf={self.overall_confidence:.2f}): {self.conclusion}")
        return "\n".join(lines)


class ChainOfThought:
    """Linear chain-of-thought reasoning engine.

    Decomposes a question into sequential reasoning steps,
    building each step on the conclusions of the previous ones.
    """

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps

    def reason(self, question: str, context: str = "",
               step_generator: Optional[Callable[[str, List[ThoughtStep]], Optional[ThoughtStep]]] = None,
               ) -> ReasoningTrace:
        """Execute chain-of-thought reasoning.

        Args:
            question: The question to reason about.
            context: Background context for reasoning.
            step_generator: Optional function(accumulated_reasoning, steps) -> next ThoughtStep.
                           If None, uses a simple decomposition heuristic.

        Returns:
            Complete ReasoningTrace.
        """
        trace = ReasoningTrace(question=question)

        if step_generator:
            accumulated = f"Context: {context}\nQuestion: {question}" if context else question
            for _ in range(self.max_steps):
                step = step_generator(accumulated, trace.steps)
                if step is None:
                    break
                trace.steps.append(step)
                accumulated += f"\nStep {step.step_number}: {step.thought}"
        else:
            # Default heuristic: break question into sub-parts
            trace.add_step(f"Analyzing the question: {question}", "analysis", 0.9)
            if context:
                trace.add_step(f"Considering context: {context[:200]}", "context", 0.85)
            trace.add_step("Synthesizing available information to form a conclusion.", "synthesis", 0.8)

        conclusion = f"Based on {len(trace.steps)} reasoning steps about: {question}"
        trace.finalize(conclusion)
        return trace


@dataclass
class ThoughtNode:
    """A node in a tree-of-thought search."""
    id: str
    thought: str
    score: float = 0.0
    children: List[ThoughtNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class TreeOfThought:
    """Tree-of-thought reasoning with branching and evaluation.

    Explores multiple reasoning paths in parallel, evaluating
    each branch to find the most promising line of thought.
    """

    def __init__(self, max_depth: int = 3, branching_factor: int = 3) -> None:
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self._node_counter = 0

    def _new_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def explore(
        self,
        question: str,
        branch_fn: Optional[Callable[[ThoughtNode], List[str]]] = None,
        evaluate_fn: Optional[Callable[[ThoughtNode], float]] = None,
    ) -> ThoughtNode:
        """Explore reasoning tree and return the root with the full tree.

        Args:
            question: Starting question.
            branch_fn: Given a node, return a list of child thoughts.
            evaluate_fn: Score a leaf node (higher is better).

        Returns:
            Root ThoughtNode of the explored tree.
        """
        root = ThoughtNode(id=self._new_id(), thought=question, depth=0)
        self._expand(root, branch_fn, evaluate_fn)
        return root

    def _expand(
        self,
        node: ThoughtNode,
        branch_fn: Optional[Callable[[ThoughtNode], List[str]]],
        evaluate_fn: Optional[Callable[[ThoughtNode], float]]
    ) -> None:
        if node.depth >= self.max_depth:
            node.score = evaluate_fn(node) if evaluate_fn else self._default_evaluate(node)
            return

        if branch_fn:
            branches = branch_fn(node)
        else:
            branches = self._default_branch(node)

        for thought in branches[:self.branching_factor]:
            child = ThoughtNode(
                id=self._new_id(),
                thought=thought,
                parent_id=node.id,
                depth=node.depth + 1,
            )
            self._expand(child, branch_fn, evaluate_fn)
            node.children.append(child)

        # Propagate best child score upward
        if node.children:
            node.score = max(c.score for c in node.children)

    def _default_branch(self, node: ThoughtNode) -> List[str]:
        base = node.thought
        return [
            f"Approach A for '{base[:50]}': break into sub-problems",
            f"Approach B for '{base[:50]}': reason by analogy",
            f"Approach C for '{base[:50]}': consider edge cases",
        ]

    @staticmethod
    def _default_evaluate(node: ThoughtNode) -> float:
        # Simple heuristic: longer, more detailed thoughts score higher
        return min(len(node.thought) / 100.0, 1.0)

    def best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Extract the best path from root to a leaf."""
        path = [root]
        current = root
        while current.children:
            best_child = max(current.children, key=lambda c: c.score)
            path.append(best_child)
            current = best_child
        return path


@dataclass
class Critique:
    """A self-critique of reasoning or output."""
    aspect: str
    assessment: str
    score: float  # 0.0 to 1.0
    suggestion: str = ""


class ReflectionEngine:
    """Self-critique and iterative improvement engine.

    Evaluates reasoning traces or outputs against quality criteria
    and suggests improvements.
    """

    DEFAULT_CRITERIA = ["completeness", "logical_consistency", "evidence_quality", "clarity"]

    def __init__(self, criteria: Optional[List[str]] = None) -> None:
        self.criteria = criteria or self.DEFAULT_CRITERIA

    def reflect(self, reasoning: ReasoningTrace) -> List[Critique]:
        """Generate critiques of a reasoning trace."""
        critiques: List[Critique] = []
        for criterion in self.criteria:
            critique = self._evaluate_criterion(criterion, reasoning)
            critiques.append(critique)
        return critiques

    def _evaluate_criterion(self, criterion: str, reasoning: ReasoningTrace) -> Critique:
        steps = reasoning.steps
        if criterion == "completeness":
            score = min(len(steps) / 3.0, 1.0)
            suggestion = "Add more reasoning steps" if score < 0.7 else ""
            return Critique(criterion, f"{len(steps)} steps in reasoning chain", score, suggestion)
        elif criterion == "logical_consistency":
            avg_conf = sum(s.confidence for s in steps) / max(len(steps), 1)
            return Critique(criterion, f"Average step confidence: {avg_conf:.2f}", avg_conf,
                            "Re-examine low-confidence steps" if avg_conf < 0.7 else "")
        elif criterion == "evidence_quality":
            with_evidence = sum(1 for s in steps if s.evidence)
            ratio = with_evidence / max(len(steps), 1)
            return Critique(criterion, f"{with_evidence}/{len(steps)} steps have evidence", ratio,
                            "Provide evidence for unsupported claims" if ratio < 0.5 else "")
        else:
            has_conclusion = bool(reasoning.conclusion)
            score = 0.9 if has_conclusion else 0.3
            return Critique(criterion, f"Conclusion present: {has_conclusion}", score,
                            "Add a clear conclusion" if not has_conclusion else "")

    def should_revise(self, critiques: List[Critique], threshold: float = 0.6) -> bool:
        """Determine if the reasoning should be revised based on critiques."""
        if not critiques:
            return False
        avg = sum(c.score for c in critiques) / len(critiques)
        return avg < threshold

    def improvement_plan(self, critiques: List[Critique]) -> List[str]:
        """Generate a list of improvements based on critiques."""
        return [c.suggestion for c in critiques if c.suggestion and c.score < 0.7]


class AnalogicalReasoner:
    """Finds similar past problems and transfers solutions.

    Stores problem-solution pairs and retrieves analogous ones
    to inform reasoning about new problems.
    """

    def __init__(self) -> None:
        self._cases: List[Dict[str, Any]] = []

    def add_case(self, problem: str, solution: str, domain: str = "", metadata: Optional[Dict] = None) -> None:
        self._cases.append({
            "problem": problem,
            "solution": solution,
            "domain": domain,
            "metadata": metadata or {},
            "embedding": self._embed(problem),
        })

    @staticmethod
    def _embed(text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = __import__("numpy").random.RandomState(seed)
        vec = rng.randn(32).tolist()
        norm = sum(v * v for v in vec) ** 0.5
        return [v / max(norm, 1e-10) for v in vec]

    def find_analogies(self, problem: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most analogous past cases."""
        q_emb = self._embed(problem)
        scored = []
        for case in self._cases:
            sim = sum(a * b for a, b in zip(q_emb, case["embedding"]))
            scored.append((case, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"problem": c["problem"], "solution": c["solution"], "domain": c["domain"], "similarity": s}
            for c, s in scored[:top_k]
        ]

    def reason_by_analogy(self, problem: str) -> str:
        """Generate reasoning by finding and applying analogies."""
        analogies = self.find_analogies(problem, top_k=2)
        if not analogies:
            return f"No analogous cases found for: {problem}"
        lines = [f"Problem: {problem}", "Analogous cases found:"]
        for a in analogies:
            lines.append(f"  - Similar problem: {a['problem']}")
            lines.append(f"    Solution used: {a['solution']}")
            lines.append(f"    Similarity: {a['similarity']:.2f}")
        lines.append(f"Suggested approach: adapt the solution from the most similar case.")
        return "\n".join(lines)


@dataclass
class Proposition:
    """A logical proposition."""
    name: str
    value: Optional[bool] = None


@dataclass
class Rule:
    """An inference rule: if all premises are true, then conclusion is true."""
    premises: List[str]
    conclusion: str
    confidence: float = 1.0


class LogicalReasoner:
    """Simple first-order logic inference with forward chaining.

    Maintains a knowledge base of facts and rules, then derives
    new facts via forward chaining until a query is resolved or
    no more inferences can be made.
    """

    def __init__(self) -> None:
        self.facts: Dict[str, bool] = {}
        self.rules: List[Rule] = []

    def assert_fact(self, name: str, value: bool = True) -> None:
        """Add a fact to the knowledge base."""
        self.facts[name] = value

    def add_rule(self, premises: List[str], conclusion: str, confidence: float = 1.0) -> None:
        """Add an inference rule."""
        self.rules.append(Rule(premises=premises, conclusion=conclusion, confidence=confidence))

    def forward_chain(self, max_iterations: int = 100) -> List[str]:
        """Apply forward chaining to derive all possible new facts.

        Returns:
            List of newly derived fact names.
        """
        derived: List[str] = []
        for _ in range(max_iterations):
            new_facts_found = False
            for rule in self.rules:
                if rule.conclusion in self.facts:
                    continue
                if all(self.facts.get(p, False) for p in rule.premises):
                    self.facts[rule.conclusion] = True
                    derived.append(rule.conclusion)
                    new_facts_found = True
            if not new_facts_found:
                break
        return derived

    def query(self, proposition: str) -> Optional[bool]:
        """Query whether a proposition is true, applying forward chaining first."""
        self.forward_chain()
        return self.facts.get(proposition)

    def explain(self, proposition: str) -> str:
        """Explain how a proposition was derived (or why it cannot be)."""
        value = self.query(proposition)
        if value is None:
            return f"'{proposition}' cannot be determined from current facts and rules."
        if not value:
            return f"'{proposition}' is explicitly false."

        # Find which rule derived it
        for rule in self.rules:
            if rule.conclusion == proposition:
                if all(self.facts.get(p, False) for p in rule.premises):
                    premises_str = " AND ".join(rule.premises)
                    return (f"'{proposition}' is true because: {premises_str} => {proposition} "
                            f"(confidence={rule.confidence:.2f})")

        return f"'{proposition}' is an asserted fact."
