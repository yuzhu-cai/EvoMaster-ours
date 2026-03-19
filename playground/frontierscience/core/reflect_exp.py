"""Reflection stage for FrontierScience."""

from __future__ import annotations

from evomaster import TaskInstance
from evomaster.core.exp import BaseExp, extract_agent_response


class ReflectExp(BaseExp):
    """Run the reflection agent to improve an initial answer."""

    @property
    def exp_name(self) -> str:
        return "Reflect"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        initial_answer: str = "",
        input_data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.logger.info("ReflectExp started (task_id=%s)", task_id)

        try:
            task = TaskInstance(
                task_id=f"{task_id}_reflect",
                task_type="reflect",
                description=task_description,
                input_data={
                    "initial_answer": initial_answer,
                    **(input_data or {}),
                },
            )

            trajectory = self.agent.run(task)
            status = str(getattr(trajectory, "status", "unknown"))
            steps = len(getattr(trajectory, "steps", []) or [])
            refined_answer = (extract_agent_response(trajectory) or "").strip()

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
        except Exception as exc:
            self.logger.error("ReflectExp failed (task_id=%s): %s", task_id, exc, exc_info=True)
            result = {
                "task_id": task_id,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
                "initial_answer": initial_answer,
                "refined_answer": "",
            }
            self.results.append(result)
            return result
