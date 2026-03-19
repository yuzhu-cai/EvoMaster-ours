"""FrontierScience playground implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from ..tools import build_frontier_tools
from .reflect_exp import ReflectExp
from .solve_exp import SolveExp

TASK_META_START = "[frontierscience_task_meta]"
TASK_META_END = "[/frontierscience_task_meta]"
EXTERNAL_TOOL_NAMES = {"search_web", "google_scholar", "visit_web"}


@register_playground("frontierscience")
class FrontierSciencePlayground(BasePlayground):
    """Two-stage science QA playground with solve and reflect agents."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).resolve().parents[3] / "configs" / "frontierscience"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("solve_agent", "reflect_agent")
        self._frontier_tool_names: list[str] = []
        self._base_enabled_tool_names: dict[str, list[str] | None] = {}

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
        self.logger.info("Registered FrontierScience tools: %s", self._frontier_tool_names)

    def _setup_agents(self) -> None:
        super()._setup_agents()
        if not self._frontier_tool_names:
            return

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            enabled_tool_names = list(agent.enabled_tool_names or [])
            for tool_name in self._frontier_tool_names:
                if tool_name not in enabled_tool_names:
                    enabled_tool_names.append(tool_name)
            agent.enabled_tool_names = enabled_tool_names
            self._base_enabled_tool_names[slot_name] = list(enabled_tool_names)
            self.logger.debug("Agent %s enabled tools: %s", slot_name, enabled_tool_names)

    def _configure_retrieval_tools(self, allow_external_retrieval: bool) -> None:
        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            base_enabled = self._base_enabled_tool_names.get(slot_name)
            if base_enabled is None:
                continue
            if allow_external_retrieval:
                agent.enabled_tool_names = list(base_enabled)
            else:
                agent.enabled_tool_names = [
                    tool_name for tool_name in base_enabled if tool_name not in EXTERNAL_TOOL_NAMES
                ]

    def _create_exp(self, name: str):
        if name == "solve_exp":
            agent = self.agents["solve_agent"]
            exp = SolveExp(agent, self.config)
        elif name == "reflect_exp":
            agent = self.agents["reflect_agent"]
            exp = ReflectExp(agent, self.config)
        else:
            raise ValueError(f"Unknown experiment: {name}")

        exp.set_run_dir(self.run_dir)
        return exp

    def _extract_task_meta(self, task_description: str) -> tuple[str, dict[str, str]]:
        text = task_description or ""
        if not text.lstrip().startswith(TASK_META_START):
            return text, {}

        lines = text.splitlines()
        if not lines:
            return text, {}

        meta: dict[str, str] = {}
        inside_meta = False
        body_start = 0

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not inside_meta:
                if line == TASK_META_START:
                    inside_meta = True
                elif line:
                    return text, {}
                continue

            if line == TASK_META_END:
                body_start = index + 1
                break

            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                meta[key] = value
        else:
            return text, {}

        cleaned_description = "\n".join(lines[body_start:]).lstrip()
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
        allow_external_retrieval = task_meta.get("allow_external_retrieval", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        task_input_data = {
            key: value
            for key, value in task_meta.items()
            if key not in {"task_id", "task_type", "allow_external_retrieval"}
        }
        task_input_data["allow_external_retrieval"] = allow_external_retrieval

        self.task_id = runtime_task_id
        try:
            self.setup()
            self._configure_retrieval_tools(allow_external_retrieval)
            self._setup_trajectory_file(output_file)

            solve_workspace = None
            if self.run_dir:
                solve_workspace = self.run_dir / "workspaces" / runtime_task_id

            self.logger.info(
                "Running SolveExp (task_id=%s, task_type=%s, image_count=%d)",
                runtime_task_id,
                runtime_task_type,
                len(images or []),
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

            self.logger.info("Running ReflectExp (task_id=%s)", runtime_task_id)
            if solve_workspace is not None:
                self.config.session["local"]["workspace_path"] = str(solve_workspace.absolute())
                self.config.session["local"]["working_dir"] = str(solve_workspace.absolute())
                reflect_trajectory_dir = self.run_dir / "trajectories" / f"{runtime_task_id}_reflect"
                reflect_trajectory_dir.mkdir(parents=True, exist_ok=True)
                reflect_trajectory_file = reflect_trajectory_dir / "trajectory.json"
                reflect_agent = self.agents.get("reflect_agent")
                if reflect_agent is not None:
                    reflect_agent.set_trajectory_file_path(reflect_trajectory_file)

            reflect_exp = self._create_exp("reflect_exp")
            reflect_result = reflect_exp.run(
                task_description=cleaned_description,
                task_id=runtime_task_id,
                initial_answer=initial_answer,
                input_data=task_input_data,
            )

            return {
                "task_id": runtime_task_id,
                "task_type": runtime_task_type,
                "status": reflect_result.get("status"),
                "steps": (solve_result.get("steps") or 0) + (reflect_result.get("steps") or 0),
                "initial_answer": initial_answer,
                "refined_answer": reflect_result.get("refined_answer"),
                "final_answer": reflect_result.get("refined_answer") or initial_answer,
                "solve_trajectory": solve_result.get("trajectory"),
                "reflect_trajectory": reflect_result.get("trajectory"),
            }
        except Exception as exc:
            self.logger.error("FrontierScience playground failed (task_id=%s): %s", runtime_task_id, exc, exc_info=True)
            return {
                "task_id": runtime_task_id,
                "task_type": runtime_task_type,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
            }
        finally:
            self.cleanup()
