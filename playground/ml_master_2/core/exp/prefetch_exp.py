import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code,save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent
import json

class PrefetchExp(BaseExp):
    def __init__(self, prefetch_agent, config,exp_name):
        super().__init__(prefetch_agent, config)
        self.prefetch_agent = prefetch_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.prefetch_agent.session.config.workspace_path
        self._exp_name = exp_name
        self.wisdom_file_path = os.path.join(os.getcwd(), self.config.agents.get("prefetch", {}).get("wisdom_file"))

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return self._exp_name

    def run(self,task_description: str, task_id: str = "aerial-cactus-identification") -> dict:
        self.logger.info("Starting prefetch task execution")
        self.logger.info(f"Task: {task_description}")

        with open(self.wisdom_file_path, "r") as f:
            wisdom = json.load(f)
        # data_knowledge = wisdom[task_id].get("data_knowledge", "NO DATA KNOWLEDGE this time")
        # model_knowledge = wisdom[task_id].get("model_knowledge", "NO MODEL KNOWLEDGE this time")

        data_knowledge = "NO DATA KNOWLEDGE this time"
        model_knowledge = "NO MODEL KNOWLEDGE this time"

        return data_knowledge, model_knowledge