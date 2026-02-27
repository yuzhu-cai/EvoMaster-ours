"""PhysMaster Playground implementation.

Workflow:
1. Clarifier analyzes and splits the original problem into subtasks.
2. Supervisor schedules next subtasks based on critic feedback.
3. Theoretician solves subtasks with reasoning/calculation.
4. Critic evaluates each attempt and guides further exploration.
5. Summarizer agent reports the best MCTS path.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.agent.tools import ToolRegistry

from .exp import PhyMasterExp

if TYPE_CHECKING:
    from evomaster.agent import Agent


@register_playground("phy_master")
class PhyMasterPlayground(BasePlayground):
    """PHY Master Playground.

    A multi-agent physics workflow with an MCTS-style exploration loop.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "phy_master"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.clarifier_agent = None
        self.supervisor_agent = None
        self.theoretician_agent = None
        self.critic_agent = None
        self.summarizer_agent = None
        self.supervisor_agent_pool = []
        self.theoretician_agent_pool = []
        self.critic_agent_pool = []
        self.parallel_nodes = 1

        self.mcp_manager = None

    def setup(self) -> None:
        """Initialize all components and create five agents."""
        self.logger.info("Setting up PhysMaster playground...")

        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict

        self._setup_session()

        # Optional skills support.
        skill_registry = None
        config_dict = self.config.model_dump()
        skills_config = config_dict.get("skills", {})
        if skills_config.get("enabled", False):
            from pathlib import Path
            from evomaster.skills import SkillRegistry

            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
            skill_registry = SkillRegistry(skills_root)
            self.logger.info("Loaded %s skills", len(skill_registry.get_all_skills()))

        self._setup_tools(skill_registry)

        phy_cfg = self.config.model_dump().get("phy_mcts", {})
        parallel_nodes = phy_cfg.get("parallel_nodes", phy_cfg.get("parallel_processes", 1))
        self.parallel_nodes = max(1, int(parallel_nodes))

        agents_config = getattr(self.config, "agents", {})
        required = ["clarifier", "supervisor", "theoretician", "critic", "summarizer"]
        original_tools = self.tools
        for name in required:
            if name not in agents_config:
                raise ValueError(f"Missing required agent config: {name}")

            cfg = agents_config[name]
            # Clarifier must only decompose contract/subtasks and use workflow retrieval.
            self.tools = self._tool_registry_for_agent(name, original_tools)
            agent = self._create_agent(
                name=name,
                agent_config=cfg,
                enable_tools=cfg.get("enable_tools", False),
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,
            )
            self.tools = original_tools
            setattr(self, f"{name}_agent", agent)
            self.logger.info("Agent created: %s", name)

        # Build dedicated agent pools for parallel node evaluation.
        self.supervisor_agent_pool = [self.supervisor_agent]
        self.theoretician_agent_pool = [self.theoretician_agent]
        self.critic_agent_pool = [self.critic_agent]
        if self.parallel_nodes > 1:
            for idx in range(1, self.parallel_nodes):
                supervisor_clone = self._create_agent(
                    name=f"supervisor_{idx}",
                    agent_config=agents_config["supervisor"],
                    enable_tools=agents_config["supervisor"].get("enable_tools", False),
                    llm_config_dict=llm_config_dict,
                    skill_registry=skill_registry,
                )
                theoretician_clone = self._create_agent(
                    name=f"theoretician_{idx}",
                    agent_config=agents_config["theoretician"],
                    enable_tools=agents_config["theoretician"].get("enable_tools", False),
                    llm_config_dict=llm_config_dict,
                    skill_registry=skill_registry,
                )
                critic_clone = self._create_agent(
                    name=f"critic_{idx}",
                    agent_config=agents_config["critic"],
                    enable_tools=agents_config["critic"].get("enable_tools", False),
                    llm_config_dict=llm_config_dict,
                    skill_registry=skill_registry,
                )
                self.supervisor_agent_pool.append(supervisor_clone)
                self.theoretician_agent_pool.append(theoretician_clone)
                self.critic_agent_pool.append(critic_clone)

        self.logger.info(
            "PHY parallel settings: parallel_nodes=%s (supervisor/theoretician/critic pools ready)",
            self.parallel_nodes,
        )
        self.logger.info("PhysMaster playground setup complete")

    def _tool_registry_for_agent(self, agent_name: str, full_registry: ToolRegistry) -> ToolRegistry:
        """Return per-agent tool registry to enforce side-effect boundaries."""
        if agent_name != "clarifier":
            return full_registry

        allowlist = ("use_skill", "finish")
        registry = ToolRegistry()
        for tool_name in allowlist:
            tool = full_registry.get_tool(tool_name)
            if tool is not None:
                registry.register(tool)
        return registry

    def _create_exp(self):
        exp = PhyMasterExp(
            clarifier_agent=self.clarifier_agent,
            supervisor_agent=self.supervisor_agent,
            theoretician_agent=self.theoretician_agent,
            critic_agent=self.critic_agent,
            summarizer_agent=self.summarizer_agent,
            supervisor_agent_pool=self.supervisor_agent_pool,
            theoretician_agent_pool=self.theoretician_agent_pool,
            critic_agent_pool=self.critic_agent_pool,
            parallel_nodes=self.parallel_nodes,
            config=self.config,
        )
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Run PHY Master workflow."""
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            exp = self._create_exp()

            self.logger.info("Running PHY Master experiment...")
            task_id = getattr(self, "task_id", None)
            if task_id:
                return exp.run(task_description, task_id=task_id)
            return exp.run(task_description)

        finally:
            self.cleanup()
