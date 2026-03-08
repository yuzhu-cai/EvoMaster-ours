import logging
import re
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
import uuid
import os
import json
from evomaster.agent import BaseAgent


def _parse_json_from_response(text: str) -> dict:
    """解析模型返回的 JSON，兼容纯 JSON 和 ```json ... ``` 代码块格式"""
    text = text.strip()
    # 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    return json.loads(text)

class ResearchExp(BaseExp):
    def __init__(self, research_agent, config, initial_code, exp_name):
        super().__init__(research_agent, config)
        self.research_agent = research_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.research_agent.session.config.workspace_path
        self.initial_code = initial_code
        self._exp_name = exp_name
    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return self._exp_name


    def run(self, task_description: str, data_preview: str, best_solution: str, research_plan_and_result: list, task_id: str = "exp_001") -> dict:
        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        # 将 research_plan_and_result 拼接为文本：list 中奇数位为 plan，偶数位为对应结果
        if not research_plan_and_result:
            research_plan_and_result_text = "You have not made any improvement attempts and results yet."
        else:
            research_plan_and_result_text = ""
            for i in range(0, len(research_plan_and_result), 2):
                plan = research_plan_and_result[i] if i < len(research_plan_and_result) else ""
                result = research_plan_and_result[i + 1] if i + 1 < len(research_plan_and_result) else ""
                block = (
                    "Based on the above code, you tried the following research plan:\n"
                    f"{plan}\n"
                    "Conclusion:\n"
                    f"{result}"
                )
                research_plan_and_result_text += block
                if i + 2 < len(research_plan_and_result):
                    research_plan_and_result_text += "\n\n"

        try:
            if self.research_agent:
                self.logger.info("=" * 60)
                self.logger.info("Step 1: Research Agent analyzing task...")
                self.logger.info("=" * 60)
                BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
                research_original_format_kwargs = self.research_agent._prompt_format_kwargs.copy()

                self.research_agent._prompt_format_kwargs.update({
                    'task_description': task_description,
                    'data_preview': data_preview,
                    "initial_code": self.initial_code,
                    'best_code': best_solution,
                    'research_plan_and_result_text': research_plan_and_result_text,
                })

                research_task = TaskInstance(
                    task_id=f"{task_id}_research",
                    task_type="research",
                    description=task_description,
                    input_data={},
                )

                research_trajectory = self.research_agent.run(research_task)
                research_result = self._extract_agent_response(research_trajectory)
                # for debugging
#                 research_plan = {"major area 1": {
#         "1": "Replace the TinyCNN with a deeper convolutional network: use four convolutional blocks each consisting of a Conv2d layer with 3x3 kernel and padding 1, BatchNorm2d, ReLU activation, and MaxPool2d(2). Set output channels to 32, 64, 128, and 256 respectively. After the last block, apply AdaptiveAvgPool2d(1) to obtain a 256-dimensional feature vector, then pass through a Linear layer with 256 inputs and 1 output followed by sigmoid.",
#         "2": "Add regularization by inserting a Dropout layer with probability 0.5 after the final linear layer (before sigmoid). Also, apply L2 regularization by setting weight_decay=1e-4 in the optimizer."
#     },
#     "major area 2": {
#         "1": "Normalize the image data by computing the mean and standard deviation of the training set across all channels and applying a Normalize transform with these statistics to the train, validation, and test datasets.",
#         "2": "Apply data augmentation to the training set using transforms: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(degrees=15), and ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1). The validation and test sets should only use ToTensor and Normalize without augmentation."
#     },
#     "major area 3": {
#         "1": "Train for up to 50 epochs with early stopping: monitor validation loss after each epoch and stop training if the loss does not decrease for 5 consecutive epochs. Save the model checkpoint with the lowest validation loss.",
#         "2": "Use Adam optimizer with initial learning rate 0.001. Replace BCELoss with BCEWithLogitsLoss and compute pos_weight as the ratio of negative to positive samples in the training set to handle class imbalance. Additionally, compute ROC AUC on the validation set after each epoch to directly track competition metric performance."
#     }
# }
                research_plan = _parse_json_from_response(research_result)
                
                self.logger.info("Research completed")
                self.logger.info(f"Research result: {research_result[:2000]}...")
                self.logger.info(f"Research plan: {research_plan}")
                self.research_agent._prompt_format_kwargs = research_original_format_kwargs

            return research_plan

        except Exception as e:
            self.logger.error(f"Research task execution failed: {e}", exc_info=True)
            raise ValueError(f"Research task execution failed: {e}")




