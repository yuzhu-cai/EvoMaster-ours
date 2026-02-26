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
