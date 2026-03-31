"""FrontierScience playground implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from ..tools import build_frontier_tools
from .utils import (
    FrontierScienceSearchRunner,
    build_task_runtime,
    enable_frontier_tools_for_agents,
    resolve_mcts_limits,
)


@register_playground("frontierscience")
class FrontierSciencePlayground(BasePlayground):
    """FrontierScience draft/improve evolving playground."""

    @staticmethod
    def _resolve_mcts_limits(search_cfg: dict[str, Any]) -> tuple[int, int]:
        return resolve_mcts_limits(search_cfg)

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).resolve().parents[3] / "configs" / "frontierscience"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("draft_agent", "improve_agent", "metric_agent")
        self._frontier_tool_names: list[str] = []
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {}) or {}
        self.max_workers = int(parallel_config.get("max_parallel", 1)) if parallel_config.get("enabled", False) else 1
        self._parallel_config = parallel_config

    def setup(self) -> None:
        self.logger.info("Setting up FrontierScience playground")
        self._setup_session()
        self._setup_agents()
        self.logger.info("FrontierScience playground setup complete")

    def _setup_tools(
        self,
        skill_config: dict | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        super()._setup_tools(skill_config=skill_config, tool_config=tool_config)
        custom_tools = build_frontier_tools()
        for tool in custom_tools:
            self.tools.register(tool)
        self._frontier_tool_names = [tool.name for tool in custom_tools]

    def _setup_agents(self) -> None:
        super()._setup_agents()
        enable_frontier_tools_for_agents(self.agents, self._frontier_tool_names)

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        default_task_id = getattr(self, "task_id", "exp_001")
        runtime_task = build_task_runtime(task_description, default_task_id)

        self.task_id = runtime_task.task_id
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            self.logger.info(
                "Running FrontierScience draft/improve search (task_id=%s, task_type=%s, image_count=%d, workers=%d)",
                runtime_task.task_id,
                runtime_task.task_type,
                len(images or []),
                self.max_workers,
            )
            return FrontierScienceSearchRunner(
                playground=self,
                problem=runtime_task.problem,
                task_id=runtime_task.task_id,
                task_type=runtime_task.task_type,
                task_input_data=runtime_task.task_input_data,
                images=images,
            ).run()
        except Exception as exc:
            self.logger.error("FrontierScience playground failed (task_id=%s): %s", runtime_task.task_id, exc, exc_info=True)
            return {
                "task_id": runtime_task.task_id,
                "task_type": runtime_task.task_type,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
            }
        finally:
            self.cleanup()
