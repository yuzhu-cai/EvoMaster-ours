"""Chat Agent Playground

面向对话式交互的 Agent，适用于飞书等即时通讯场景。
"""

import logging
import os
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
        self._register_search_tools()
        self._register_delegate_tool()

    def _register_search_tools(self):
        """根据 tools.search 配置决定注册哪套搜索工具。"""
        tool_config = self._setup_agent_tools("general")
        provider = tool_config.get("search", "google")
        self.logger.info("Search provider: %s", provider)

        if provider == "ai_search":
            self._register_ai_search_tool()
        else:
            self._register_google_search_tool()
            self._register_web_fetch_tool()

    def _register_ai_search_tool(self):
        """从配置中读取 ai_search 段，注册 AI 综合搜索工具。"""
        ws_config = getattr(self.config, "ai_search", None)
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
            self.logger.warning("ai_search config incomplete, skipping")
            return

        from playground.chat_agent.tools.ai_search import AISearchTool

        tool = AISearchTool(api_key=api_key, base_url=base_url, model=model)
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered ai_search tool (model: %s)", model)

    def _register_google_search_tool(self):
        """从配置或环境变量读取 Serper API key，注册 Google 搜索工具。"""
        gs_config = getattr(self.config, "google_search", None)
        cfg = {}
        if gs_config is not None:
            cfg = gs_config if isinstance(gs_config, dict) else (
                gs_config.__dict__ if hasattr(gs_config, "__dict__") else {}
            )

        api_key = os.environ.get("SERPER_KEY_ID") or cfg.get("api_key")
        if not api_key:
            self.logger.warning("google_search: no SERPER_KEY_ID env var or config api_key, skipping")
            return

        from playground.chat_agent.tools.google_search import GoogleSearchTool

        tool = GoogleSearchTool(api_key=api_key)
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered google_search tool")

    def _register_web_fetch_tool(self):
        """注册网页抓取工具，LLM 摘要复用 chat_agent 默认 LLM 配置。"""
        wf_config = getattr(self.config, "web_fetch", None)
        cfg = {}
        if wf_config is not None:
            cfg = wf_config if isinstance(wf_config, dict) else (
                wf_config.__dict__ if hasattr(wf_config, "__dict__") else {}
            )

        jina_api_key = os.environ.get("JINA_API_KEY") or cfg.get("api_key")

        # 复用 chat_agent 的默认 LLM 创建 BaseLLM 实例做摘要提取
        from evomaster.utils.llm import LLMConfig, create_llm

        llm_cfg = self.config_manager.get_llm_config()
        llm = create_llm(LLMConfig(**llm_cfg))

        from playground.chat_agent.tools.web_fetch import WebFetchTool

        tool = WebFetchTool(jina_api_key=jina_api_key, llm=llm)
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered web_fetch tool (llm: %s)", llm_cfg.get("model"))

    def _register_delegate_tool(self):
        """注册委派工具，允许 chat_agent 将任务转交给专业 Agent。"""
        from playground.chat_agent.tools.delegate import DelegateToAgentTool

        tool = DelegateToAgentTool()
        for agent in self.agents.values():
            agent.tools.register(tool)

        self.logger.info("Registered delegate_to_agent tool")
