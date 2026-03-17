"""Reflect experiment for FrontierScience."""

from __future__ import annotations

from typing import Any

from evomaster.core.exp import BaseExp
from evomaster import TaskInstance


class ReflectExp(BaseExp):
    """ReflectExp - 对初始答案进行反思和改进"""

    @property
    def exp_name(self) -> str:
        return "Reflect"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        initial_answer: str = "",
    ) -> dict[str, Any]:
        self.logger.info("ReflectExp started (task_id=%s)", task_id)

        task = TaskInstance(
            task_id=f"{task_id}_reflect",
            task_type="reflect",
            description=task_description,
            input_data={"initial_answer": initial_answer},
        )

        trajectory = self.agent.run(task)
        status = str(getattr(trajectory, "status", "unknown"))
        steps = len(getattr(trajectory, "steps", []) or [])
        refined_answer = self._extract_agent_response(trajectory)

        self.logger.info("ReflectExp completed (task_id=%s, status=%s)", task_id, status)

        result = {
            "task_id": task_id,
            "status": status,
            "steps": steps,
            "trajectory": trajectory,
            "initial_answer": initial_answer,
            "refined_answer": refined_answer,
        }
        self.results.append(result)
        return result
