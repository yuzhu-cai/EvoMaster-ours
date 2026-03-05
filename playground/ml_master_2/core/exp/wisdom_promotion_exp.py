import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code, save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent
import json
import re

def _parse_json_from_response(text: str) -> dict:
    """解析模型返回的 JSON，兼容纯 JSON 和 ```json ... ``` 代码块格式"""
    text = text.strip()
    # 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    return json.loads(text)

class WisdomPromotionExp(BaseExp):
    def __init__(self, wisdom_promotion_agent, config, exp_name):
        super().__init__(wisdom_promotion_agent, config)
        self.wisdom_promotion_agent = wisdom_promotion_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.wisdom_promotion_agent.session.config.workspace_path
        self._exp_name = exp_name

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return self._exp_name

    def run(
        self,
        task_description: str,
        best_solution: str,
        task_id: str = "exp_001",
    ) -> dict:
        self.logger.info("Starting wisdom promotion task execution")

        wisdom_promotion_original_format_kwargs = self.wisdom_promotion_agent._prompt_format_kwargs.copy()
        self.wisdom_promotion_agent._prompt_format_kwargs.update({
            'task_description': task_description,
            'best_solution': best_solution,
        })
        wisdom_promotion_task = TaskInstance(
            task_id=f"{task_id}_wisdom_promotion",
            task_type="wisdom_promotion",
            description=task_description,
            input_data={},
        )

        wisdom_promotion_trajectory = self.wisdom_promotion_agent.run(wisdom_promotion_task)
        wisdom_promotion_result = self._extract_agent_response(wisdom_promotion_trajectory)
        wisdom_promotion_result = _parse_json_from_response(wisdom_promotion_result)
        self.logger.info(f"Wisdom promotion result: {wisdom_promotion_result}")
        self.wisdom_promotion_agent._prompt_format_kwargs = wisdom_promotion_original_format_kwargs

        return wisdom_promotion_result