"""Report Writer Agent Playground"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("report_writer_agent")
class ReportWriterAgentPlayground(BasePlayground):
    """Report Writer Agent Playground

    接收调研资料，撰写专业报告（工作报告、分析报告、研究报告）
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs" / "report_writer_agent"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
