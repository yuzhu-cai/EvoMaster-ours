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


def _format_score(score: float | None) -> str:
    """将分数格式化为可读字符串"""
    if score is None:
        return "运行失败 / N/A"
    return f"{score:.6f}"


def generate_improvement_summary_text(
    base_solution: str,
    best_solution: str,
    research_plan: dict,
    research_round_idea_results: dict,
) -> str:
    """
    根据 research_plan 和 research_round_idea_results 自动生成改进过程的可读摘要文本。
    让读者一目了然地知道：在什么代码基础上、依次尝试了哪些改进、各自得分与是否带来提升、最终采纳了哪些修改。
    """
    lines = []
    lines.append("## 改进过程摘要")
    lines.append("")
    lines.append("**基线代码**：以上述最佳代码为起点，依次按 research_plan 中的方向进行改进实验。")
    lines.append("")

    current_base_note = "在上述最佳代码的基础上"
    for direction_idx, direction in enumerate(research_plan, start=1):
        direction_results = research_round_idea_results.get(direction, {})
        if not direction_results:
            continue

        # 方向标题
        ordinal = "首先" if direction_idx == 1 else f"随后"
        lines.append(f"### 方向 {direction_idx}：{direction}")
        lines.append("")
        lines.append(f"{current_base_note}，{ordinal}尝试了该方向下的所有 idea，结果如下：")
        lines.append("")

        # 各 idea 的结果
        best_idea_in_direction = None
        for idea_idx, (idea_key, idea_desc) in enumerate(research_plan[direction].items(), start=1):
            idea_tuple = (idea_key, idea_desc)
            result = direction_results.get(idea_tuple, {})
            score = result.get("score")
            improved = result.get("improved", False)
            is_best = result.get("is_best_in_direction", False)

            score_str = _format_score(score)
            improved_str = "✓ 带来提升" if improved else "✗ 未带来提升"
            if is_best:
                best_idea_in_direction = (idea_key, idea_desc)
                improved_str += " 【本方向最佳，已采纳】"

            lines.append(f"- **Idea {idea_idx}**（{idea_key}）：{idea_desc}")
            lines.append(f"  - 得分：{score_str} | {improved_str}")
            lines.append("")

        # 本方向最终选择
        if best_idea_in_direction:
            idea_key, idea_desc = best_idea_in_direction
            lines.append(f"**本方向最终采纳**：Idea {idea_key} 的修改（{idea_desc}）")
        else:
            lines.append("**本方向最终采纳**：无（所有 idea 均未带来提升，保持原代码）")
        lines.append("")

        # 下一方向的基线说明
        current_base_note = "在采纳上述修改后的代码基础上"

    lines.append("---")
    lines.append("")
    lines.append("**最终最佳代码**：经过上述各方向的依次改进后，得到当前最佳代码如下：")
    lines.append("")
    lines.append("```python")
    lines.append(best_solution)
    lines.append("```")

    return "\n".join(lines)


class KnowledgePromotionExp(BaseExp):
    def __init__(self, knowledge_promotion_agent, config, exp_index):
        super().__init__(knowledge_promotion_agent, config)
        self.knowledge_promotion_agent = knowledge_promotion_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.knowledge_promotion_agent.session.config.workspace_path
        self.exp_index = exp_index

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return f"KnowledgePromotion_{self.exp_index}"

    def run(
        self,
        task_description: str,
        data_preview: str,
        base_solution: str,
        best_solution: str,
        research_plan: dict,
        research_round_idea_results: dict,
        task_id: str = "exp_001",
    ) -> dict:
        self.logger.info("Starting knowledge promotion task execution")

        results_text = generate_improvement_summary_text(
            base_solution=base_solution,
            best_solution=best_solution,
            research_plan=research_plan,
            research_round_idea_results=research_round_idea_results,
        )
        self.logger.info("Generated improvement summary:\n%s", results_text)
        knowledge_promotion_original_format_kwargs = self.knowledge_promotion_agent._prompt_format_kwargs.copy()
        self.knowledge_promotion_agent._prompt_format_kwargs.update({
            'task_description': task_description,
            'current_base_code': base_solution,
            'research_plan': research_plan,
            'results': results_text,
        })
        knowledge_promotion_task = TaskInstance(
            task_id=f"{task_id}_knowledge_promotion",
            task_type="knowledge_promotion",
            description=task_description,
            input_data={},
        )

        knowledge_promotion_trajectory = self.knowledge_promotion_agent.run(knowledge_promotion_task)
        knowledge_promotion_result = self._extract_agent_response(knowledge_promotion_trajectory)
        self.logger.info(f"Knowledge promotion result: {knowledge_promotion_result}")
        self.knowledge_promotion_agent._prompt_format_kwargs = knowledge_promotion_original_format_kwargs

        return knowledge_promotion_result