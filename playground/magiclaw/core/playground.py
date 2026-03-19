"""MagiClaw Playground

A conversation-oriented Agent for instant messaging scenarios such as Feishu.
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("magiclaw")
class MagiClawPlayground(BasePlayground):
    """MagiClaw Playground

    A conversation-oriented Q&A playground, used by default for Feishu Bot interactions.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize ChatAgentPlayground.

        Args:
            config_dir: Configuration directory path, defaults to configs/chat_agent/
            config_path: Full path to config file (overrides config_dir if provided)
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "magiclaw"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup(self):
        """Set up the playground and initialize memory."""
        super().setup()
        self._setup_memory()

    def _create_custom_tool_instance(self, tool_class, tool_name, tool_key):
        """为需要依赖注入的自定义工具提供初始化逻辑"""
        instance = super()._create_custom_tool_instance(tool_class, tool_name, tool_key)
        if instance is not None and tool_name == "web_fetch":
            from evomaster.utils.llm import LLMConfig, create_llm
            llm_cfg = self.config_manager.get_llm_config()
            instance._llm = create_llm(LLMConfig(**llm_cfg))
        return instance

    def _setup_memory(self):
        """Initialize the memory system (if enabled in config)."""
        memory_cfg = self.config_manager.get("memory") or {}
        if not memory_cfg.get("enabled", False):
            self._memory_manager = None
            self._memory_config = {}
            return

        from evomaster.memory.store import MemoryStore
        from evomaster.memory.manager import MemoryManager

        db_path = memory_cfg.get("db_path", "./data/memory/memories.db")
        store = MemoryStore(db_path)

        # Optional: use LLM for memory extraction during compaction
        llm = None
        if memory_cfg.get("capture_with_llm", False):
            from evomaster.utils.llm import LLMConfig, create_llm

            llm_cfg = self.config_manager.get_llm_config()
            llm = create_llm(LLMConfig(**llm_cfg))

        self._memory_manager = MemoryManager(store, llm=llm, config=memory_cfg)
        self._memory_config = memory_cfg
        self.logger.info("Memory system initialized (db: %s)", db_path)
