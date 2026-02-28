"""Agent Builder Playground

元级 Agent 系统，包含两个 agent：
- planner: 深度研究用户需求，设计 Agent 方案，写入飞书文档
- builder: 根据方案生成 Agent 文件（config, prompts）
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("agent_builder")
class AgentBuilderPlayground(BasePlayground):
    """Agent Builder Playground

    双 agent 系统：
    1. planner_agent: 研究框架 → 分析需求 → 架构决策 → 设计 prompt → 写飞书文档
    2. builder_agent: 读飞书文档 → 生成目录/文件 → 验证
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent_builder"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        # 声明两个 agent slot（与 config.yaml 中 agents.planner / agents.builder 对应）
        self.agents.declare("planner_agent", "builder_agent")

    def setup(self) -> None:
        self.logger.info("Setting up Agent Builder playground...")
        self._setup_session()
        self._setup_agents()
        # planner 作为主 agent：dispatcher 首次 run() 时使用 planner
        # builder 由 dispatcher 在确认后单独调用
        self.agent = self.agents.planner_agent
        self.logger.info("Agent Builder playground setup complete (planner as primary agent)")
