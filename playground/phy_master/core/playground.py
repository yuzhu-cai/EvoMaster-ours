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

        self.agents.declare(
            "clarifier_agent",
            "supervisor_agent",
            "theoretician_agent",
            "critic_agent",
            "summarizer_agent",
        )
        self.supervisor_agent_pool = []
        self.theoretician_agent_pool = []
        self.critic_agent_pool = []
        self.parallel_nodes = 1

        self.mcp_manager = None

    def setup(self) -> None:
        """Initialize all components and create five agents."""
        self.logger.info("Setting up PhysMaster playground...")

        self._setup_session()
        self._setup_agents()

        phy_cfg = self.config.model_dump().get("phy_mcts", {})
        parallel_nodes = phy_cfg.get("parallel_nodes", phy_cfg.get("parallel_processes", 1))
        self.parallel_nodes = max(1, int(parallel_nodes))

        agents_config = getattr(self.config, "agents", {})
        required = ["clarifier", "supervisor", "theoretician", "critic", "summarizer"]
        for name in required:
            if name not in agents_config:
                raise ValueError(f"Missing required agent config: {name}")

        # Build dedicated agent pools for parallel node evaluation.
        self.supervisor_agent_pool = [self.agents.supervisor_agent]
        self.theoretician_agent_pool = [self.agents.theoretician_agent]
        self.critic_agent_pool = [self.agents.critic_agent]
        if self.parallel_nodes > 1:
            for idx in range(1, self.parallel_nodes):
                supervisor_clone = self.copy_agent(
                    self.agents.supervisor_agent,
                    new_agent_name=f"supervisor_{idx}",
                )
                theoretician_clone = self.copy_agent(
                    self.agents.theoretician_agent,
                    new_agent_name=f"theoretician_{idx}",
                )
                critic_clone = self.copy_agent(
                    self.agents.critic_agent,
                    new_agent_name=f"critic_{idx}",
                )
                self.supervisor_agent_pool.append(supervisor_clone)
                self.theoretician_agent_pool.append(theoretician_clone)
                self.critic_agent_pool.append(critic_clone)

        self.logger.info(
            "PHY parallel settings: parallel_nodes=%s (supervisor/theoretician/critic pools ready)",
            self.parallel_nodes,
        )
        self.logger.info("PhysMaster playground setup complete")

    def _create_exp(self):
        exp = PhyMasterExp(
            clarifier_agent=self.agents.clarifier_agent,
            supervisor_agent=self.agents.supervisor_agent,
            theoretician_agent=self.agents.theoretician_agent,
            critic_agent=self.agents.critic_agent,
            summarizer_agent=self.agents.summarizer_agent,
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
