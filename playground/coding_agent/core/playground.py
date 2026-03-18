"""Coding Agent Playground"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("coding_agent")
class CodingAgentPlayground(BasePlayground):
    """Coding Agent Playground

    代码编写、调试、执行、文件操作
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs" / "coding_agent"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
