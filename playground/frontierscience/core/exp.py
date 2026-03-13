"""Experiment implementation for FrontierScience."""

from __future__ import annotations

from typing import Any

from evomaster.core.exp import BaseExp, extract_agent_response
from evomaster.utils.types import TaskInstance


class FrontierScienceExp(BaseExp):
    """Single-agent experiment that returns a task-level result."""

    @property
    def exp_name(self) -> str:
        return "FrontierScience"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        image_list = images or []
        self.logger.info(
            "FrontierScienceExp started (task_id=%s, text_len=%d, image_count=%d)",
            task_id,
            len(task_description),
            len(image_list),
        )

        try:
            task = TaskInstance(
                task_id=task_id,
                task_type="frontier_science",
                description=task_description,
                images=image_list,
            )

            trajectory = self.agent.run(task)
            status = str(getattr(trajectory, "status", "unknown"))
            steps = len(getattr(trajectory, "steps", []) or [])
            final_answer = (extract_agent_response(trajectory) or "").strip()

            self.logger.info(
                "FrontierScienceExp completed (task_id=%s, status=%s, steps=%d)",
                task_id,
                status,
                steps,
            )
            if final_answer:
                preview = final_answer if len(final_answer) <= 300 else f"{final_answer[:300]}..."
                self.logger.info("Final answer preview: %s", preview)
            else:
                self.logger.warning("No final answer extracted (task_id=%s)", task_id)

            result = {
                "task_id": task_id,
                "status": status,
                "steps": steps,
                "trajectory": trajectory,
                "final_answer": final_answer,
            }
            self.results.append(result)
            return result
        except Exception as exc:
            self.logger.error(
                "FrontierScienceExp failed (task_id=%s): %s",
                task_id,
                exc,
                exc_info=True,
            )
            result = {
                "task_id": task_id,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
                "final_answer": "",
            }
            self.results.append(result)
            return result

