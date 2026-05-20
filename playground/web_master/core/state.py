"""Lightweight state objects for the Flash-Searcher style playground."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchNode:
    """A coarse-grained searchable intent in the DAG."""

    node_id: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    error: str = ""


@dataclass
class SearchNodeResult:
    """Result produced by one search node execution."""

    node_id: str
    status: str
    answer: str = ""
    evidence: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    raw_output: str = ""
    steps: int = 0
    error: str = ""


@dataclass
class SearchDAG:
    """In-memory DAG used by FlashSearchExp."""

    nodes: dict[str, SearchNode]

    def ready_nodes(self, completed: set[str]) -> list[SearchNode]:
        return [
            node
            for node in self.nodes.values()
            if node.status == "pending"
            and all(dep in completed for dep in node.depends_on)
        ]

    def unfinished_nodes(self) -> list[SearchNode]:
        return [node for node in self.nodes.values() if node.status != "completed"]

    def critical_path_length(self) -> int:
        memo: dict[str, int] = {}

        def visit(node_id: str, stack: set[str]) -> int:
            if node_id in memo:
                return memo[node_id]
            if node_id in stack:
                return 1
            node = self.nodes[node_id]
            if not node.depends_on:
                memo[node_id] = 1
                return 1
            stack.add(node_id)
            depth = 1 + max(
                visit(dep, stack)
                for dep in node.depends_on
                if dep in self.nodes
            )
            stack.remove(node_id)
            memo[node_id] = depth
            return depth

        if not self.nodes:
            return 0
        return max(visit(node_id, set()) for node_id in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            node_id: {
                "goal": node.goal,
                "depends_on": node.depends_on,
                "status": node.status,
                "error": node.error,
            }
            for node_id, node in self.nodes.items()
        }
