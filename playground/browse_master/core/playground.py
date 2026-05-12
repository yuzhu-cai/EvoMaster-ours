"""BrowseMaster Playground Implementation

Agent searches the web to answer questions from the BrowseComp dataset.
"""

import json
import logging
import re
import sys
from typing import Any
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground

from .exp import BrowseMasterExp
from ..tools import build_browse_tools

DATASET_PATH = Path(__file__).parent.parent / "test" / "browsecomp_decrypted.json"
EXTERNAL_TOOL_NAMES = {"google_search","web_fetch"}


@register_playground("browse_master")
class BrowseMasterPlayground(BasePlayground):
    """BrowseMaster Playground: Agent searches the web to answer questions."""

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = project_root / "configs" / "browse_master"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("search_agent")
        self.mcp_manager = None
        self._browse_tool_names: list[str] = []
        self._base_enabled_tool_names: dict[str, list[str] | None] = {}
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> dict:
        """Load dataset and index by id."""
        if not DATASET_PATH.exists():
            self.logger.warning(f"Dataset not found: {DATASET_PATH}")
            return {}
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item["id"]: item for item in data}

    def _parse_task(self, task_description: str) -> tuple[str, str | None]:
        """Parse task description.

        Supports:
          - "dataset:0" or "id:42" -> load from dataset
          - raw question text -> use directly

        Returns:
            (question, ground_truth_answer)
        """
        match = re.match(
            r"^(?:dataset|id):\s*(\d+)$", task_description.strip(), re.IGNORECASE
        )
        if match:
            qid = int(match.group(1))
            item = self.dataset.get(qid)
            if item:
                self.logger.info(f"Loaded dataset entry id={qid}")
                return item["question"], item.get("answer")
            self.logger.warning(f"Dataset id {qid} not found, using as raw text")
        return task_description, None
    
    def _setup_tools(
        self,
        skill_config: dict | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        super()._setup_tools(skill_config=skill_config, tool_config=tool_config)

        custom_tools = build_browse_tools()
        for tool in custom_tools:
            self.tools.register(tool)

        self._browse_tool_names = [tool.name for tool in custom_tools]
        self.logger.info("Registered BrowseScience tools: %s", self._browse_tool_names)

    def _setup_agents(self) -> None:
        super()._setup_agents()
        if not self._browse_tool_names:
            return

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            enabled_tool_names = list(agent.enabled_tool_names or [])
            for tool_name in self._browse_tool_names:
                if tool_name not in enabled_tool_names:
                    enabled_tool_names.append(tool_name)
            agent.enabled_tool_names = enabled_tool_names
            self._base_enabled_tool_names[slot_name] = list(enabled_tool_names)
            self.logger.debug("Agent %s enabled tools: %s", slot_name, enabled_tool_names)

    def setup(self) -> None:
        self._setup_session()
        self._setup_agents()

    def _create_exp(self):
        exp = BrowseMasterExp(
            agent=self.agents.search_agent,
            config=self.config,
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
            # Prefer ground_truth passed from run.py via --json/--id
            ground_truth = getattr(self, '_ground_truth', None) or gt_from_task
            exp = self._create_exp()
            exp.ground_truth = ground_truth

            task_id = getattr(self, "task_id", None) or "exp_001"
            result = exp.run(
                question, task_id=task_id, images=images, on_step=on_step
            )

            if ground_truth:
                agent_answer = result.get("agent_answer", "")
                self.logger.info(f"Ground truth: {ground_truth}")
                self.logger.info(f"Agent answer: {agent_answer}")

            return result
        finally:
            self.cleanup()
