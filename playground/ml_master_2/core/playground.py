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
from evomaster.agent.session import (
    LocalSessionConfig,
    DockerSession,
    DockerSessionConfig,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from ..agent.session.local import MLMaster2LocalSession
from evomaster.utils.types import TaskInstance
from .exp.draft_exp import DraftExp
from .exp.research_exp import ResearchExp
from .exp.improve_exp import ImproveExp
from .exp.prefetch_exp import PrefetchExp
from .exp.knowledge_promotion_exp import KnowledgePromotionExp
from .utils.data_preview import generate
from .utils.code import save_code_to_file
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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

    def _setup_session(self) -> None:
        """创建并打开 Session，使用 MLMaster2LocalSession 替代默认 LocalSession"""
        if self.session is None:
            session_type = self.config.session.get("type", "local")
            if session_type == "docker":
                session_config_dict = self.config.session.get("docker", {}).copy()
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                elif "workspace_path" not in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["workspace_path"] = "/workspace"
                    session_config_dict["working_dir"] = "/workspace"
                session_config = DockerSessionConfig(**session_config_dict)
                self.session = DockerSession(session_config)
                self.logger.info(f"Using Docker session with image: {session_config.image}")
            else:
                session_config_dict = self.config.session.get("local", {}).copy()
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                if "config_dir" not in session_config_dict:
                    session_config_dict["config_dir"] = str(self.config_dir)
                session_config = LocalSessionConfig(**session_config_dict)
                self.session = MLMaster2LocalSession(session_config)
                self.logger.info("Using ML Master 2 Local session")

        if not self.session.is_open:
            self.session.open()
        else:
            self.logger.debug("Session already open, reusing existing session")

    def _setup_workspace(self):
        os.makedirs(os.path.join(self.session.config.workspace_path, "best_submission"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "best_solution"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "submission"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "working"), exist_ok=True)
        self.logger.info(f"working_dir: {self.session.config.workspace_path}")

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
            

            prefetch_exp = PrefetchExp(self.agents.prefetch_agent, self.config,f"exp_{self.exp_index}_prefetch")
            self.exp_index += 1
            data_knowledge, model_knowledge = prefetch_exp.run(task_description)

            data_preview = generate(self.session.config.workspace_path)
            self.logger.info(f"Data preview: {data_preview}")
            self.logger.info("Running experiment...")
            draft_exp = DraftExp(self.agents.draft_agent, self.agents.debug_agent, self.agents.metric_agent, self.config,f"exp_{self.exp_index}_draft")
            draft_workspace_name = f"exp_{self.exp_index}_draft"
            self.exp_index += 1
            draft_result = self.execute_parallel_tasks([partial(draft_exp.run, task_description=task_description, data_preview=data_preview, data_knowledge=data_knowledge, model_knowledge=model_knowledge)], max_workers=1, workspace_names=[draft_workspace_name])
            is_sucess, validation_score, uid, self.best_solution = draft_result[0]
            self.initial_code = self.best_solution
            if is_sucess:
                self.best_score = validation_score
                shutil.copy(os.path.join(draft_exp.workspace_path, "submission", f"submission_{uid}.csv"), os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_submission", f"submission.csv"))
                save_code_to_file(os.path.join(self.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)

            for reseach_round in range(10):
                # 记录当前 research_round 中每个 direction 的每个 idea 的结果
                # 结构: {direction: {idea: {"improved": bool, "is_best_in_direction": bool, "score": float|None}}}
                research_round_idea_results: dict[str, dict[tuple, dict]] = {}
                base_solution = self.best_solution  # 本轮开始时的最佳代码

                research_exp = ResearchExp(self.agents.reseach_agent, self.config, self.initial_code, f"exp_{self.exp_index}_research")
                self.exp_index += 1
                research_plan = research_exp.run(task_description, data_preview, self.best_solution, self.research_plan_and_result)
                for direction in research_plan:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_best_idea = None  # 记录该 direction 中带来最佳分数的 idea
                    research_round_idea_results[direction] = {}

                    ideas = list(research_plan[direction].items())
                    for idea in ideas:
                        improve_exp = ImproveExp(self.agents.improve_agent, self.agents.debug_agent, self.agents.metric_agent, self.config, f"exp_{self.exp_index}_improve")
                        improve_workspace_name = f"exp_{self.exp_index}_improve"
                        self.exp_index += 1
                        improve_result = self.execute_parallel_tasks([partial(improve_exp.run, task_description=task_description, data_preview=data_preview, best_solution=direction_best_solution, idea=idea)], max_workers=1, workspace_names=[improve_workspace_name])
                        is_sucess, validation_score, uid, self.best_solution = improve_result[0]
                        improved = self.compare_score(direction_best_score, validation_score)
                        research_round_idea_results[direction][idea] = {"improved": improved, "is_best_in_direction": False, "score": validation_score}
                        if improved:
                            direction_best_score = validation_score
                            direction_best_solution = self.best_solution
                            direction_best_idea = idea
                            shutil.copy(os.path.join(improve_exp.workspace_path, "submission", f"submission_{uid}.csv"), os.path.join(self.agents.improve_agent.session.config.workspace_path, "best_submission", f"submission.csv"))
                            save_code_to_file(os.path.join(self.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)

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
                knowledge_promotion_exp = KnowledgePromotionExp(self.agents.knowledge_promotion_agent, self.config, f"exp_{self.exp_index}_knowledge_promotion")
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


    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 3, workspace_names: List[str] | None = None) -> List[Any]:
            """通用并行任务执行器

            Args:
                tasks: 这里的每个元素应该是一个可调用的对象。
                    如果是带参数的函数，请使用 functools.partial 封装。
                    例如: [partial(exp1.run, task="A"), partial(exp2.run, task="B")]
                max_workers: 最大并行线程数
                workspace_names: 可选。为每个任务指定独立工作空间的子目录名。
                    若提供且 split_workspace_for_exp 启用，则使用此列表中的名称替代默认的 exp_{parallel_index}。
                    长度应与 tasks 一致。

            Returns:
                List[Any]: 按照输入 tasks 的顺序返回结果列表。
                        如果任务抛出异常，结果列表中对应位置将是该 Exception 对象。
            """
            self.logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers.")
            
            results = [None] * len(tasks)
            
            # 检查是否启用了并行资源分配
            session_config = self.config.session.get("local", {})
            parallel_config = session_config.get("parallel", {})
            parallel_enabled = parallel_config.get("enabled", False)
            
            # 检查是否启用了 split_workspace_for_exp
            split_workspace = parallel_config.get("split_workspace_for_exp", False)
            
            # 包装任务函数，设置并行索引和独立工作空间
            def wrap_task(task_func, parallel_index):
                def wrapped():
                    try:
                        # 如果启用了并行资源分配，设置 session 的并行索引
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(parallel_index)
                                self.logger.debug(f"设置并行索引: {parallel_index}")
                                
                                # 如果启用了 split_workspace_for_exp，为当前 exp 创建独立工作空间
                                if split_workspace:
                                    import os
                                    main_workspace = self.session.config.workspace_path
                                    exp_name = workspace_names[parallel_index] if workspace_names and parallel_index < len(workspace_names) else f"exp_{parallel_index}"
                                    exp_workspace = os.path.join(main_workspace, exp_name)
                                    # 通过 env 创建 exp 工作空间（含软链接）
                                    self.session._env.setup_exp_workspace(exp_workspace)
                                    os.makedirs(os.path.join(exp_workspace, "submission"), exist_ok=True)
                                    os.makedirs(os.path.join(exp_workspace, "working"), exist_ok=True)
                                    # 设置线程本地的工作空间路径
                                    self.session.set_workspace_path(exp_workspace)
                                    self.logger.info(
                                        f"Exp {parallel_index} 使用独立工作空间: {exp_workspace}"
                                    )
                        return task_func()
                    finally:
                        # 清理线程本地状态
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(None)
                                if split_workspace:
                                    self.session.set_workspace_path(None)
                return wrapped
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，建立 future 到 index 的映射，以保证返回顺序
                wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
                future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

                # 处理完成的任务
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        # 获取返回值
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        # 将异常对象作为结果返回，避免打断其他任务
                        results[index] = exc

            self.logger.info("Parallel execution completed.")
            return results
