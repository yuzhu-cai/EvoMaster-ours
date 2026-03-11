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
        self._setup_memory()

    def _setup_memory(self):
        """初始化记忆系统（如果在 config 中启用）。"""
        memory_cfg = self.config_manager.get("memory") or {}
        if not memory_cfg.get("enabled", False):
            self._memory_manager = None
            self._memory_config = {}
            return

        from evomaster.memory.store import MemoryStore
        from evomaster.memory.manager import MemoryManager

        db_path = memory_cfg.get("db_path", "./data/memory/memories.db")
        store = MemoryStore(db_path)

        # 可选：用 LLM 做 compaction 时的记忆提取
        llm = None
        if memory_cfg.get("capture_with_llm", False):
            from evomaster.utils.llm import LLMConfig, create_llm

            llm_cfg = self.config_manager.get_llm_config()
            llm = create_llm(LLMConfig(**llm_cfg))

        self._memory_manager = MemoryManager(store, llm=llm, config=memory_cfg)
        self._memory_config = memory_cfg
        self.logger.info("Memory system initialized (db: %s)", db_path)
