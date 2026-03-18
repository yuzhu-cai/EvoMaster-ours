"""Web Search Agent Playground"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("web_search_agent")
class WebSearchAgentPlayground(BasePlayground):
    """Web Search Agent Playground

    深度网络搜索、多源信息综合、事实核查
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs" / "web_search_agent"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_custom_tool_instance(self, tool_class, tool_name, tool_key):
        """为 web_fetch 注入 LLM 实例"""
        instance = super()._create_custom_tool_instance(tool_class, tool_name, tool_key)
        if instance is not None and tool_name == "web_fetch":
            from evomaster.utils.llm import LLMConfig, create_llm
            llm_cfg = self.config_manager.get_llm_config()
            instance._llm = create_llm(LLMConfig(**llm_cfg))
        return instance
