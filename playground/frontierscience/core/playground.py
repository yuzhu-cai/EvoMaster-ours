"""FrontierScience playground implementation."""
# 经过solveexp之后的答案保存在solution.md，经过reflectexp之后的答案保存在solution_refined.md
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from .solve_exp import SolveExp
from .reflect_exp import ReflectExp
from ..tools import build_frontier_tools


@register_playground("frontierscience")
class FrontierSciencePlayground(BasePlayground):
    """A minimal playground specialized for science QA with web/scholar tools."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).resolve().parents[3] / "configs" / "frontierscience"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("solve_agent", "reflection_agent")
        self._frontier_tool_names: list[str] = []

    def setup(self) -> None:
        self.logger.info("Setting up minimal FrontierScience playground...")
        self._setup_session()
        self._setup_agents()
        self.logger.info("Minimal FrontierScience playground setup complete")

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
        self.logger.info(
            "Registered FrontierScience tools (%d): %s",
            len(self._frontier_tool_names),
            self._frontier_tool_names,
        )

    def _setup_agents(self) -> None:
        super()._setup_agents()
        if not self._frontier_tool_names:
            return

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            if agent.enabled_tool_names is None:
                agent.enabled_tool_names = []
            added_names: list[str] = []
            for tool_name in self._frontier_tool_names:
                if tool_name not in agent.enabled_tool_names:
                    agent.enabled_tool_names.append(tool_name)
                    added_names.append(tool_name)
            self.logger.info(
                "Agent '%s' tool setup complete, newly added=%s, total=%d",
                slot_name,
                added_names if added_names else "[]",
                len(agent.enabled_tool_names),
            )
            self.logger.debug(
                "Agent '%s' final enabled tools: %s",
                slot_name,
                agent.enabled_tool_names,
            )

    def _create_exp(self, name:str):
        if name=="solve_exp":
            agent = self.agents.get("solve_agent") 
            if agent is None:
                raise RuntimeError("No available agent for SolveExp.")
            exp = SolveExp(agent, self.config)
            if self.run_dir:
                exp.set_run_dir(self.run_dir)
        elif name == "reflect_exp":
            agent = self.agents.get("reflect_agent")
            if agent is None:
                raise RuntimeError("No available agent for ReflectExp.")
            exp = ReflectExp(agent, self.config)
            if self.run_dir:
                exp.set_run_dir(self.run_dir)
        
        return exp

    def _extract_task_meta(self, task_description: str) -> tuple[str, dict[str, str]]:
        pattern = re.compile(
            r"^\s*\[frontierscience_task_meta\]\s*\n(?P<body>.*?)\n\s*\[/frontierscience_task_meta\]\s*\n?",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.match(task_description or "")
        if match is None:
            return task_description, {}

        body = match.group("body")
        meta: dict[str, str] = {}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key:
                meta[key] = val

        cleaned_description = (task_description[match.end() :]).lstrip()
        return cleaned_description, meta

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        default_task_id = getattr(self, "task_id", "exp_001")
        cleaned_description, task_meta = self._extract_task_meta(task_description)
        runtime_task_id = task_meta.get("task_id", default_task_id) or default_task_id
        runtime_task_type = task_meta.get("task_type", "frontier_science") or "frontier_science"
        task_input_data = {k: v for k, v in task_meta.items() if k not in {"task_id", "task_type"}}
        self.task_id = runtime_task_id
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            # Stage 1: FrontierScienceExp
            image_count = len(images or [])
            self.logger.info(
                "Stage 1: Running FrontierScienceExp (task_id=%s, task_type=%s, image_count=%d)",
                runtime_task_id,
                runtime_task_type,
                image_count,
            )
            solve_exp = self._create_exp("solve_exp")
            solve_result = solve_exp.run(
                task_description=cleaned_description,
                task_id=runtime_task_id,
                task_type=runtime_task_type,
                input_data=task_input_data,
                images=images,
            )
            initial_answer = solve_result.get("final_answer", "")
            self.logger.info(
                "Stage 1: SolveExp finished (task_id=%s, status=%s, steps=%s)",
                runtime_task_id,
                solve_result.get("status"),
                solve_result.get("steps"),
            )

            # Stage 2: ReflectExp
            self.logger.info(
                "Stage 2: Running ReflectExp (task_id=%s)",
                runtime_task_id,
            )
            reflect_exp = self._create_exp("reflect_exp")
            reflect_result = reflect_exp.run(
                task_description=cleaned_description,
                task_id=runtime_task_id,
                initial_answer=initial_answer,
            )
            self.logger.info(
                "Stage 2: ReflectExp finished (task_id=%s, status=%s, steps=%s)",
                runtime_task_id,
                reflect_result.get("status"),
                reflect_result.get("steps"),
            )

            # Merge results
            return {
                "task_id": runtime_task_id,
                "task_type": runtime_task_type,
                "status": reflect_result.get("status"),
                "steps": reflect_result.get("steps"),
                "initial_answer": initial_answer,
                "refined_answer": reflect_result.get("refined_answer"),
                "solve_trajectory": solve_result.get("trajectory"),
                "reflect_trajectory": reflect_result.get("trajectory"),
            }
        except Exception as exc:
            self.logger.error(
                "Minimal FrontierScience playground failed (task_id=%s): %s",
                runtime_task_id,
                exc,
                exc_info=True,
            )
            return {
                "task_id": runtime_task_id,
                "task_type": runtime_task_type,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
            }
        finally:
            self.cleanup()
