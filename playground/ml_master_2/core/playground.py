import os
import logging
import sys
import json
from pathlib import Path
import shutil
import copy
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent


from evomaster.utils.types import TaskInstance
from .exp.draft_exp import DraftExp
from .exp.research_exp import ResearchExp
from .exp.improve_exp import ImproveExp
from .exp.prefetch_exp import PrefetchExp
from .exp.knowledge_promotion_exp import KnowledgePromotionExp
from .utils.data_preview import generate
from .utils.code import save_code_to_file

@register_playground("ml_master_2")
class MLMaster2Playground(BasePlayground):
    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "ml_master_2"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("draft_agent", "debug_agent", "improve_agent", "reseach_agent", "knowledge_promotion_agent", "metric_agent", "prefetch_agent")

        self.initial_code = None
        self.best_score = None
        self.best_solution = None
        self.research_plan_and_result = []
        
        self.is_lower_better = False
        self.mcp_manager = None

        self.exp_index = 0 # for trajectory visualizing

    def setup(self) -> None:
        self.logger.info("Setting up ml master 2 playground...")

        self._setup_session()
        self._setup_agents()
        self._setup_workspace()

        self.logger.info("ML Master 2 playground setup complete")

    def _setup_workspace(self):
        os.makedirs(os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_submission"), exist_ok=True)
        os.makedirs(os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_solution"), exist_ok=True)
        os.makedirs(os.path.join(self.agents.draft_agent.session.config.workspace_path, "submission"), exist_ok=True)
        os.makedirs(os.path.join(self.agents.draft_agent.session.config.workspace_path, "working"), exist_ok=True)
        self.logger.info(f"working_dir: {self.agents.draft_agent.session.config.workspace_path}")

    def compare_score(self, old_score, new_score):
        if old_score is None or new_score is None:
            return True if new_score is not None else False
        if old_score < new_score and self.is_lower_better == False:
            return True
        elif old_score > new_score and self.is_lower_better == True:
            return True
        else:
            return False

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()

            self._setup_trajectory_file(output_file)
            

            prefetch_exp = PrefetchExp(self.agents.prefetch_agent, self.config,self.exp_index)
            data_knowledge, model_knowledge = prefetch_exp.run(task_description)

            data_preview = generate(self.agents.draft_agent.session.config.workspace_path)
            self.logger.info(f"Data preview: {data_preview}")
            self.logger.info("Running experiment...")
            draft_exp = DraftExp(self.agents.draft_agent, self.agents.debug_agent, self.agents.metric_agent, self.config,self.exp_index)
            self.exp_index += 1
            is_sucess, validation_score, uid,self.best_solution = draft_exp.run(task_description, data_preview, data_knowledge, model_knowledge)
            self.initial_code = self.best_solution
            if is_sucess:
                self.best_score = validation_score
                shutil.copy(os.path.join(self.agents.draft_agent.session.config.workspace_path, "submission", f"submission_{uid}.csv"), os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_submission", f"submission.csv"))
                save_code_to_file(os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)

            for reseach_round in range(10):
                # 记录当前 research_round 中每个 direction 的每个 idea 的结果
                # 结构: {direction: {idea: {"improved": bool, "is_best_in_direction": bool, "score": float|None}}}
                research_round_idea_results: dict[str, dict[tuple, dict]] = {}
                base_solution = self.best_solution  # 本轮开始时的最佳代码

                research_exp = ResearchExp(self.agents.reseach_agent, self.config, self.initial_code, self.exp_index)
                self.exp_index += 1
                research_plan = research_exp.run(task_description, data_preview, self.best_solution, self.research_plan_and_result)
                for direction in research_plan:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_best_idea = None  # 记录该 direction 中带来最佳分数的 idea
                    research_round_idea_results[direction] = {}

                    ideas = list(research_plan[direction].items())
                    for idea in ideas:
                        improve_exp = ImproveExp(self.agents.improve_agent, self.agents.debug_agent, self.agents.metric_agent, self.config, self.exp_index)
                        self.exp_index += 1
                        is_sucess, validation_score, uid, self.best_solution = improve_exp.run(task_description, data_preview, direction_best_solution, idea)
                        improved = self.compare_score(direction_best_score, validation_score)
                        research_round_idea_results[direction][idea] = {"improved": improved, "is_best_in_direction": False, "score": validation_score}
                        if improved:
                            direction_best_score = validation_score
                            direction_best_solution = self.best_solution
                            direction_best_idea = idea
                            shutil.copy(os.path.join(self.agents.improve_agent.session.config.workspace_path, "submission", f"submission_{uid}.csv"), os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_submission", f"submission.csv"))
                            save_code_to_file(os.path.join(self.agents.improve_agent.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)

                    # 标记该 direction 中最佳的 idea
                    if direction_best_idea is not None:
                        research_round_idea_results[direction][direction_best_idea]["is_best_in_direction"] = True

                    self.best_solution = direction_best_solution
                    self.best_score = direction_best_score

                # research_round 结束时，research_round_idea_results 已完整记录
                # 将 research_plan 和结果以文本形式存入 research_plan_and_result
                plan_text = json.dumps(research_plan, ensure_ascii=False, indent=2)
                self.research_plan_and_result.extend([plan_text])

                self.logger.info(f"Round {reseach_round} results: {research_round_idea_results}")
                knowledge_promotion_exp = KnowledgePromotionExp(self.agents.knowledge_promotion_agent, self.config, self.exp_index)
                self.exp_index += 1
                knowledge_promotion_result = knowledge_promotion_exp.run(task_description, data_preview, base_solution, self.best_solution, research_plan, research_round_idea_results)
                self.research_plan_and_result.extend([knowledge_promotion_result])
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result
        except Exception as e:
            self.logger.error(f"ML Master 2 task execution failed: {e}", exc_info=True)
            result = {
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
            return result

        finally:
            self.cleanup()

