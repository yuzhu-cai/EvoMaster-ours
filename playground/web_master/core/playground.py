"""WebMaster playground implementation.

Version A implements a Flash-Searcher style DAG-parallel BrowseComp agent.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground

from ..tools import build_browse_tools
from .exp import FlashSearchExp


WEB_MASTER_DATASET_PATH = Path(__file__).parent.parent / "test" / "browsecomp_decrypted.json"
BROWSE_TOOL_NAMES = {"think", "finish", "google_search", "web_fetch"}
ROLE_TOOL_NAMES = {
    "planner_agent": [],
    "search_agent": ["think", "finish", "google_search", "web_fetch"],
    "finalizer_agent": [],
}


@register_playground("web_master")
class WebMasterPlayground(BasePlayground):
    """Flash-Searcher style BrowseComp playground."""

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = project_root / "configs" / "web_master"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("planner_agent", "search_agent", "finalizer_agent")
        self._browse_tool_names: list[str] = []
        self._base_enabled_tool_names: dict[str, list[str] | None] = {}
        self.dataset = self._load_dataset()

    @property
    def planner(self):
        return self.agents.planner_agent

    @property
    def searcher(self):
        return self.agents.search_agent

    @property
    def finalizer(self):
        return self.agents.finalizer_agent

    def _load_dataset(self) -> dict[int, dict]:
        """Load BrowseComp data and index it by id."""
        if not WEB_MASTER_DATASET_PATH.exists():
            self.logger.warning("Dataset not found: %s", WEB_MASTER_DATASET_PATH)
            return {}

        with open(WEB_MASTER_DATASET_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {int(item["id"]): item for item in data if "id" in item}

    def _parse_task(self, task_description: str) -> tuple[str, str | None]:
        """Support `dataset:<id>` / `id:<id>` shortcuts in addition to raw text."""
        match = re.match(
            r"^(?:dataset|id):\s*(\d+)$",
            task_description.strip(),
            re.IGNORECASE,
        )
        if match:
            qid = int(match.group(1))
            item = self.dataset.get(qid)
            if item:
                self.logger.info("Loaded dataset entry id=%s", qid)
                return item.get("question", ""), item.get("answer")
            self.logger.warning("Dataset id %s not found, using raw text", qid)
        return task_description, None

    def _setup_tools(
        self,
        skill_config: dict | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        super()._setup_tools(skill_config=skill_config, tool_config=tool_config)

        custom_tools = build_browse_tools()
        for tool in custom_tools:
            if self.tools.get_tool(tool.name) is None:
                self.tools.register(tool)

        self._browse_tool_names = [tool.name for tool in custom_tools]
        self.logger.info("Registered WebMaster browse tools: %s", self._browse_tool_names)

    def _setup_agents(self) -> None:
        super()._setup_agents()
        if not self._browse_tool_names:
            return

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            role_enabled = ROLE_TOOL_NAMES.get(slot_name)
            if role_enabled is None:
                enabled_tool_names = self._filter_solver_tools(
                    list(agent.enabled_tool_names or [])
                )
            else:
                enabled_tool_names = [
                    tool_name
                    for tool_name in role_enabled
                    if agent.tools.get_tool(tool_name) is not None
                ]
            agent.enabled_tool_names = enabled_tool_names
            self._base_enabled_tool_names[slot_name] = list(enabled_tool_names)
            self.logger.debug("Agent %s enabled tools: %s", slot_name, enabled_tool_names)

            # web_fetch_tool = agent.tools.get_tool("web_fetch")
            # if web_fetch_tool is not None and hasattr(web_fetch_tool, "set_llm"):
            #     web_fetch_tool.set_llm(agent.llm)

    @staticmethod
    def _filter_solver_tools(tool_names: list[str]) -> list[str]:
        filtered = [name for name in tool_names if name in BROWSE_TOOL_NAMES]
        deduped: list[str] = []
        seen: set[str] = set()
        for name in filtered:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def setup(self) -> None:
        self.logger.info("Setting up WebMaster playground...")
        self._setup_session()
        self._setup_agents()
        self.logger.info("WebMaster setup complete")

    def _create_exp(self):
        experiment_config = getattr(self.config, "experiment", {}) or {}
        exp = FlashSearchExp(
            planner=self.planner,
            searcher=self.searcher,
            finalizer=self.finalizer,
            config=self.config,
            agent_copier=self.copy_agent,
            max_workers=int(experiment_config.get("max_workers", 3)),
            max_rounds=int(experiment_config.get("max_rounds", 4)),
        )
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
        on_step=None,
    ) -> dict:
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            question, gt_from_task = self._parse_task(task_description)
            ground_truth = getattr(self, "_ground_truth", None) or gt_from_task

            exp = self._create_exp()
            exp.ground_truth = ground_truth

            task_id = getattr(self, "task_id", None) or "exp_001"
            result = exp.run(
                question,
                task_id=task_id,
                images=images,
                on_step=on_step,
            )

            if ground_truth:
                self.logger.info("Ground truth: %s", ground_truth)
                self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))

            return result
        finally:
            self.cleanup()
