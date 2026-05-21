"""State objects for the Flash-Searcher style WebMaster playground."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlashMemoryStep:
    """One high-level memory item in a Flash-Searcher run."""

    step_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlashSearcherRunState:
    """Compact run state persisted outside the raw EvoMaster trajectory."""

    task_id: str
    question: str
    planning: str = ""
    final_reasoning: str = ""
    final_answer: str = ""
    search_status: str = ""
    search_steps: int = 0
    plan_attempts: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    memory: list[FlashMemoryStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "planning": self.planning,
            "final_reasoning": self.final_reasoning,
            "final_answer": self.final_answer,
            "search_status": self.search_status,
            "search_steps": self.search_steps,
            "plan_attempts": self.plan_attempts,
            "tool_call_counts": self.tool_call_counts,
            "queries": self.queries,
            "urls": self.urls,
            "memory": [
                {
                    "step_type": step.step_type,
                    "content": step.content,
                    "metadata": step.metadata,
                }
                for step in self.memory
            ],
        }
