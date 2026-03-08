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
        return "Run failed / N/A"
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
    "带来提升"指相对本方向基线（上一方向采纳后的代码）的改进；同一方向内多个 idea 可能都优于基线，
    仅采纳得分最高者，次优者标记为"优于基线，未采纳"以体现其价值。
    """
    lines = []
    lines.append("## Improvement Process Summary")
    lines.append("")
    lines.append("**Baseline Code**: Starting from the best code above, we conducted improvement experiments in order according to the directions in research_plan.")
    lines.append("")

    current_base_note = "Based on the above best code"
    for direction_idx, direction in enumerate(research_plan, start=1):
        direction_results = research_round_idea_results.get(direction, {})
        if not direction_results:
            continue

        # 方向标题
        ordinal = "First" if direction_idx == 1 else "Subsequently"
        lines.append(f"### Direction {direction_idx}: {direction}")
        lines.append("")
        lines.append(f"{current_base_note}, {ordinal} tried all ideas under this direction, results as follows:")
        lines.append("")

        # 各 idea 的结果
        # improved: 相对本方向基线是否带来提升；is_best: 本方向中得分最高且已采纳
        # 多个 idea 可能都优于基线，仅采纳最高分者，次优者仍标记为"带来提升"以体现其价值
        best_idea_in_direction = None
        for idea_idx, (idea_key, idea_desc) in enumerate(research_plan[direction].items(), start=1):
            idea_tuple = (idea_key, idea_desc)
            result = direction_results.get(idea_tuple, {})
            score = result.get("score")
            improved = result.get("improved", False)
            is_best = result.get("is_best_in_direction", False)

            score_str = _format_score(score)
            if improved:
                if is_best:
                    improved_str = "✓ Improved [Best in this direction, adopted]"
                    best_idea_in_direction = (idea_key, idea_desc)
                else:
                    improved_str = "✓ Improved [Better than baseline, not adopted]"
            else:
                improved_str = "✗ No improvement"

            lines.append(f"- **Idea {idea_idx}** ({idea_key}): {idea_desc}")
            lines.append(f"  - Score: {score_str} | {improved_str}")
            lines.append("")

        # 本方向最终选择
        num_improved = sum(1 for r in direction_results.values() if r.get("improved", False))
        if best_idea_in_direction:
            idea_key, idea_desc = best_idea_in_direction
            if num_improved > 1:
                lines.append(f"**Final adoption in this direction**: Modifications from Idea {idea_key} ({idea_desc}), {num_improved - 1} other ideas also better than baseline")
            else:
                lines.append(f"**Final adoption in this direction**: Modifications from Idea {idea_key} ({idea_desc})")
        else:
            lines.append("**Final adoption in this direction**: None (all ideas failed to improve, keeping original code)")
        lines.append("")

        # 下一方向的基线说明
        current_base_note = "Based on the code after adopting the above modifications"

    lines.append("---")
    lines.append("")
    lines.append("**Final Best Code**: After the sequential improvements in each direction above, the current best code is as follows:")
    lines.append("")
    lines.append("```python")
    lines.append(best_solution)
    lines.append("```")

    return "\n".join(lines)


class KnowledgePromotionExp(BaseExp):
    def __init__(self, knowledge_promotion_agent, config, exp_name):
        super().__init__(knowledge_promotion_agent, config)
        self.knowledge_promotion_agent = knowledge_promotion_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.knowledge_promotion_agent.session.config.workspace_path
        self._exp_name = exp_name

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return self._exp_name

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