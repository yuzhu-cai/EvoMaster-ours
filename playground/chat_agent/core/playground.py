"""Chat Agent Playground

面向对话式交互的 Agent，适用于飞书等即时通讯场景。
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("chat_agent")
class ChatAgentPlayground(BasePlayground):
    """Chat Agent Playground

    面向对话式 Q&A 的 playground，默认用于飞书 Bot 交互。
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "chat_agent"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup(self):
        super().setup()
        self._register_web_search_tool()
        self._register_delegate_tool()

    def _register_web_search_tool(self):
        """从配置中读取 web_search 段，注册到所有 agent 的 tool registry。"""
        ws_config = getattr(self.config, "web_search", None)
        if ws_config is None:
            return

        if isinstance(ws_config, dict):
            cfg = ws_config
        else:
            cfg = ws_config.__dict__ if hasattr(ws_config, "__dict__") else {}

        api_key = cfg.get("api_key")
        base_url = cfg.get("base_url")
        model = cfg.get("model")

        if not all([api_key, base_url, model]):
            self.logger.warning("web_search config incomplete, skipping")
            return

        from playground.chat_agent.tools.web_search import WebSearchTool

        tool = WebSearchTool(api_key=api_key, base_url=base_url, model=model)
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered web_search tool (model: %s)", model)

    def _register_delegate_tool(self):
        """注册委派工具，允许 chat_agent 将任务转交给专业 Agent。"""
        from playground.chat_agent.tools.delegate import DelegateToAgentTool

        tool = DelegateToAgentTool()
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered delegate_to_agent tool")
