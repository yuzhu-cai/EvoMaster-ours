"""Experiment modules for FrontierScience."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance


class NodeExp(BaseExp):
    """Thin exp wrapper around one agent run plus metric evaluation."""

    def __init__(self, agent, metric_evaluator=None) -> None:
        super().__init__(agent=agent, config=None)
        self.metric_evaluator = metric_evaluator

    @staticmethod
    def _load_metric_answer(solution_path: str, fallback_answer: str) -> str:
        path = Path(solution_path)
        if path.exists():
            saved_answer = path.read_text(encoding="utf-8", errors="ignore").strip()
            if saved_answer:
                return saved_answer
        return fallback_answer

    def _run_agent(
        self,
        *,
        prompt_kwargs: dict[str, Any],
        problem: str,
        task_id: str,
        task_type: str,
        task_input_data: dict[str, Any],
        previous_answer: str,
        reference_rubric: str,
        solution_path: str,
        images: list[str] | None,
    ) -> dict[str, Any]:
        original_kwargs = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(prompt_kwargs)
        try:
            task = TaskInstance(
                task_id=task_id,
                task_type=task_type,
                description=problem,
                input_data=task_input_data,
                images=images or [],
            )
            trajectory = self.agent.run(task)
            current_answer = self._extract_agent_response(trajectory)
        finally:
            self.agent._prompt_format_kwargs = original_kwargs

        metric_answer = self._load_metric_answer(solution_path, current_answer)
        metric_result = None
        if self.metric_evaluator is not None:
            metric_result = self.metric_evaluator.evaluate(
                problem=problem,
                final_answer=metric_answer,
                trajectory=trajectory,
                previous_answer=previous_answer,
                reference_rubric=reference_rubric,
                action_type=task_type,
                is_final=False,
            )

        return {
            "trajectory": trajectory,
            "current_answer": current_answer,
            "answer_path": solution_path,
            "metric_result": metric_result,
            "state_summary": (
                f"{task_type}; status={getattr(trajectory, 'status', 'unknown')}; "
                f"answer_chars={len(current_answer)}; "
                f"metric_overall={None if metric_result is None else metric_result.get('overall_score')}"
            ),
        }


BaseNodeExp = NodeExp

from .draft_exp import DraftExp
from .improve_exp import ImproveExp

__all__ = ["DraftExp", "ImproveExp"]
