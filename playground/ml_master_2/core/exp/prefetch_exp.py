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

        if self.prefetch_agent:
            self.logger.info("=" * 60)
            self.logger.info("Step 1: Prefetch Agent analyzing task...")
            self.logger.info("=" * 60)
            BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
            
            prefetch_original_format_kwargs = self.prefetch_agent._prompt_format_kwargs.copy()
            self.prefetch_agent._prompt_format_kwargs.update({
                'task_description': task_description,
            })
            prefetch_task = TaskInstance(
                task_id=f"{task_id}_prefetch",
                task_type="prefetch",
                task_description=task_description,
                input_data={},
            )
            prefetch_trajectory = self.prefetch_agent.run(prefetch_task)
            prefetch_result = self._extract_agent_response(prefetch_trajectory)
            self.prefetch_agent._prompt_format_kwargs = prefetch_original_format_kwargs
            self.logger.info(f"Prefetch result: {prefetch_result}")
            # 执行 RAG 检索：use_skill run_script 调用 search.py
            vec_dir = "evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft"
            nodes_data = "evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json"
            model = "evomaster/skills/rag/local_models/all-mpnet-base-v2"
            query_escaped = json.dumps(prefetch_result)
            script_args = (
                f"--vec_dir {vec_dir} --query {query_escaped} --nodes_data {nodes_data} "
                f"--top_k 1 --threshold 0.6 --output json --model {model}"
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(
                    name="use_skill",
                    arguments=json.dumps({
                        "skill_name": "rag",
                        "action": "run_script",
                        "script_name": "search.py",
                        "script_args": script_args,
                    }),
                )
            )
            observation, info =self.prefetch_agent._execute_tool(tool_call_obj) 
            exit()
        ## 测试代码
        with open(self.wisdom_file_path, "r") as f:
            wisdom = json.load(f)
        # data_knowledge = wisdom[task_id].get("data_knowledge", "NO DATA KNOWLEDGE this time")
        # model_knowledge = wisdom[task_id].get("model_knowledge", "NO MODEL KNOWLEDGE this time")

        data_knowledge = "NO DATA KNOWLEDGE this time"
        model_knowledge = "NO MODEL KNOWLEDGE this time"

        return data_knowledge, model_knowledge