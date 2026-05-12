"""BrowseMaster Experiment Implementation

Single agent searches the web to answer a question.
"""

import json
import logging
from pathlib import Path

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance


class BrowseMasterExp(BaseExp):
    """BrowseMaster Experiment: Single agent searches web to answer a question."""

    def __init__(self, agent, config):
        super().__init__(agent, config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "BrowseMaster"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self.logger.info(f"Starting search task: {task_id}")
        self.logger.info(f"Question: {task_description}")

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)

        task = TaskInstance(
            task_id=task_id,
            task_type="search",
            description=task_description,
            input_data={},
        )

        trajectory = self.agent.run(task, on_step=on_step)
        agent_answer = self._extract_agent_response(trajectory)

        self.logger.info(f"Agent final answer: {agent_answer}")

        result = {
            "task_id": task_id,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
            "trajectory": trajectory,
            "agent_answer": agent_answer,
            "ground_truth": self.ground_truth,
        }
        self.results.append(result)
        return result

    def save_results(self, output_file: str):
        """Save experiment results including agent_answer and ground_truth."""
        output_data = []
        for result in self.results:
            output_data.append({
                "task_id": result["task_id"],
                "status": result["status"],
                "steps": result["steps"],
                "agent_answer": result.get("agent_answer", ""),
                "ground_truth": result.get("ground_truth"),
                "trajectory": result["trajectory"].model_dump(),
            })

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")
