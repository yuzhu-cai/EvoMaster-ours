"""DAG state objects for the Flash-Searcher style WebMaster playground."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchNode:
    """One DAG subtask in the Flash-Searcher execution graph."""

    node_id: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    node_type: str = "search"
    priority: int = 0
    status: str = "pending"
    speculative: bool = False
    error: str = ""


@dataclass
class SearchNodeResult:
    """Execution result for one DAG node."""

    node_id: str
    status: str
    answer: str = ""
    confidence: str = "unknown"
    evidence: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    dependency_node_ids: list[str] = field(default_factory=list)
    missing_dependency_node_ids: list[str] = field(default_factory=list)
    speculative: bool = False
    raw_output: str = ""
    steps: int = 0
    error: str = ""


@dataclass
class SchedulerRound:
    """A single adaptive scheduling round."""

    round_index: int
    ready_node_ids: list[str] = field(default_factory=list)
    speculative_node_ids: list[str] = field(default_factory=list)
    completed_node_ids: list[str] = field(default_factory=list)
    optimizer_notes: str = ""


@dataclass
class SearchDAG:
    """Mutable execution graph used by the DAG scheduler."""

    nodes: dict[str, SearchNode]
    hard_constraints: list[str] = field(default_factory=list)
    final_goal: str = ""

    def ready_nodes(self, completed: set[str]) -> list[SearchNode]:
        ready = [
            node
            for node in self.nodes.values()
            if node.status == "pending"
            and all(dep in completed for dep in node.depends_on)
        ]
        return sorted(ready, key=lambda item: (-item.priority, item.node_id))

    def partially_ready_nodes(
        self,
        completed: set[str],
        exclude: set[str] | None = None,
    ) -> list[SearchNode]:
        exclude = exclude or set()
        partial = []
        for node in self.nodes.values():
            if node.status != "pending" or node.node_id in exclude:
                continue
            if not node.depends_on:
                continue
            if any(dep in completed for dep in node.depends_on):
                partial.append(node)
        return sorted(partial, key=lambda item: (-item.priority, item.node_id))

    def unfinished_nodes(self) -> list[SearchNode]:
        return [node for node in self.nodes.values() if node.status not in {"completed", "failed", "skipped"}]

    def add_or_update_node(self, node: SearchNode) -> None:
        node.depends_on = [dep for dep in node.depends_on if dep != node.node_id]
        if node.node_id in self.nodes:
            existing = self.nodes[node.node_id]
            existing.goal = node.goal or existing.goal
            existing.depends_on = [dep for dep in node.depends_on if dep in self.nodes]
            existing.node_type = node.node_type or existing.node_type
            existing.priority = node.priority
            return
        node.depends_on = [dep for dep in node.depends_on if dep in self.nodes]
        self.nodes[node.node_id] = node

    def critical_path_length(self) -> int:
        memo: dict[str, int] = {}

        def visit(node_id: str, stack: set[str]) -> int:
            if node_id in memo:
                return memo[node_id]
            if node_id in stack or node_id not in self.nodes:
                return 1
            node = self.nodes[node_id]
            if not node.depends_on:
                memo[node_id] = 1
                return 1
            stack.add(node_id)
            depth = 1 + max((visit(dep, stack) for dep in node.depends_on), default=0)
            stack.remove(node_id)
            memo[node_id] = depth
            return depth

        if not self.nodes:
            return 0
        return max(visit(node_id, set()) for node_id in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_constraints": self.hard_constraints,
            "final_goal": self.final_goal,
            "nodes": {
                node_id: {
                    "goal": node.goal,
                    "depends_on": node.depends_on,
                    "node_type": node.node_type,
                    "priority": node.priority,
                    "status": node.status,
                    "speculative": node.speculative,
                    "error": node.error,
                }
                for node_id, node in self.nodes.items()
            },
        }
