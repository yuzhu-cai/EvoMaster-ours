import math
import os
import logging
import sys
import json
from pathlib import Path
import shutil
import copy
import threading
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
from .exp.wisdom_promotion_exp import WisdomPromotionExp
from .utils.data_preview import generate
from .utils.code import save_code_to_file
from .utils.watch_dog import (
    TimeoutWatchdog,
    GlobalTimeoutInterrupt,
    RUN_TIMEOUT_SECONDS,
    _async_raise,
)
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
        self.agents.declare("draft_agent", "debug_agent", "improve_agent", "reseach_agent", "knowledge_promotion_agent", "metric_agent", "prefetch_agent","wisdom_promotion_agent")

        self.initial_code = None
        self.best_score = None
        self.best_solution = None
        self.real_time_best_solution = None
        self.research_plan_and_result = []
        self.prefetch_descriptor = None
        self.is_lower_better = self.config_manager.get("is_lower_better", False)
        self.competition_id = self.config_manager.get("competition_id", "detecting-insults-in-social-commentary")
        # for grading server
        self.ground_truth_dir = os.path.join(os.getcwd(), self.config_manager.get("data_root", "playground/ml_master_2/data"))
        self.mcp_manager = None

        self.exp_index = 0

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
                raise ValueError("Docker session is not supported for ML Master 2")
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

    def _is_valid_score(self, score) -> bool:
        """NaN 视为无效分数，优先级低于任何有效分数。"""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def compare_score(self, old_score, new_score):
        # new_score 无效（None/NaN）→ 永远不算提升
        if not self._is_valid_score(new_score):
            return False
        # old_score 无效（None/NaN）→ 任何有效 new_score 都算提升
        if not self._is_valid_score(old_score):
            return True
        # 两者都有效，按原逻辑比较
        if old_score < new_score and self.is_lower_better == False:
            return True
        elif old_score > new_score and self.is_lower_better == True:
            return True
        else:
            return False

    def _create_improve_exp(self, exp_index: int) -> ImproveExp:
        """为并行任务创建独立的 ImproveExp 实例，使用 copy_agent 避免上下文冲突。

        参考 minimal_multi_agent_parallel 的并行设计，每个并行任务拥有独立的 Agent 副本，
        确保 LLM 调用和上下文不冲突。

        Args:
            exp_index: 实验索引，用于生成唯一的 exp_name 和 agent 名称

        Returns:
            ImproveExp 实例
        """
        improve_agent_copy = self.copy_agent(
            self.agents.improve_agent, new_agent_name=f"improve_exp_{exp_index}"
        ) if self.agents.improve_agent else None
        debug_agent_copy = self.copy_agent(
            self.agents.debug_agent, new_agent_name=f"debug_exp_{exp_index}"
        ) if self.agents.debug_agent else None
        metric_agent_copy = self.copy_agent(
            self.agents.metric_agent, new_agent_name=f"metric_exp_{exp_index}"
        ) if self.agents.metric_agent else None
        exp_name = f"exp_{exp_index}_improve"
        return ImproveExp(
            improve_agent_copy, debug_agent_copy, metric_agent_copy,
            self.config, exp_name
        )

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        # 启动看门狗守护线程
        watchdog = TimeoutWatchdog(RUN_TIMEOUT_SECONDS)
        watchdog.start()
        self.logger.info(f"已启动看门狗线程（{RUN_TIMEOUT_SECONDS} 秒）")
        try:
            self.setup()

            self._setup_trajectory_file(output_file)
            
            prefetch_exp = PrefetchExp(self.agents.prefetch_agent, self.config,f"exp_{self.exp_index}_prefetch")
            self.exp_index += 1
            embedding_config = getattr(self.config, "embedding", {})
            embedding_model = embedding_config.get("openai", {}).get("model", "text-embedding-3-large")
            data_knowledge, model_knowledge, self.prefetch_descriptor = prefetch_exp.run(task_description,vec_dir=os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom"),nodes_data=os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom/db.json"),model=embedding_model)
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
                self.real_time_best_solution = self.best_solution
            for reseach_round in range(20):
                # 记录当前 research_round 中每个 direction 的每个 idea 的结果
                # 结构: {direction: {idea: {"improved": bool, "is_best_in_direction": bool, "score": float|None}}}
                research_round_idea_results: dict[str, dict[tuple, dict]] = {}
                base_solution = self.best_solution  # 本轮开始时的最佳代码

                research_exp = ResearchExp(self.agents.reseach_agent, self.config, self.initial_code, f"exp_{self.exp_index}_research")
                research_workspace_name = f"exp_{self.exp_index}_research"
                self.exp_index += 1
                research_results = self.execute_parallel_tasks(
                    [partial(research_exp.run, task_description=task_description, data_preview=data_preview, best_solution=self.best_solution, research_plan_and_result=self.research_plan_and_result)],
                    max_workers=1,
                    workspace_names=[research_workspace_name]
                )
                research_result = research_results[0]
                if isinstance(research_result, Exception):
                    self.logger.error(f"Research failed: {research_result}")
                    raise research_result
                research_plan = research_result
                # 从配置读取 idea 并行数，默认最多 2 个
                session_config = self.config.session.get("local", {})
                parallel_config = session_config.get("parallel", {})
                idea_max_workers = parallel_config.get("max_parallel", 2)

                for direction in research_plan:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_baseline_score = self.best_score  # 本方向基线分数，用于判断各 idea 是否带来提升（不随迭代更新）
                    direction_best_idea = None  # 记录该 direction 中带来最佳分数的 idea
                    research_round_idea_results[direction] = {}

                    ideas = list(research_plan[direction].items())
                    if not ideas:
                        continue

                    # 构建并行任务：每个 idea 一个任务，均以 direction_best_solution 为 base
                    tasks = []
                    workspace_names = []
                    improve_exp_list = []
                    for i, idea in enumerate(ideas):
                        exp_index = self.exp_index + i
                        improve_exp = self._create_improve_exp(exp_index)
                        improve_exp_list.append(improve_exp)
                        task = partial(
                            improve_exp.run,
                            task_description=task_description,
                            data_preview=data_preview,
                            best_solution=direction_best_solution,
                            idea=idea,
                        )
                        tasks.append(task)
                        workspace_names.append(improve_exp.exp_name)
                    self.exp_index += len(ideas)

                    # 并行执行该 direction 下所有 idea，最多 idea_max_workers 个同时运行
                    improve_results = self.execute_parallel_tasks(
                        tasks, max_workers=idea_max_workers, workspace_names=workspace_names
                    )

                    # 处理结果：按顺序收集，找出最佳
                    # improved 与 direction_baseline_score 比较，表示相对本方向开始时的基线是否带来提升
                    # 这样同一 direction 内多个优于基线的 idea 都会被正确标记为"带来提升"
                    for i, (idea, result) in enumerate(zip(ideas, improve_results)):
                        improve_exp = improve_exp_list[i]
                        if isinstance(result, Exception):
                            self.logger.error(f"Idea {idea} failed: {result}")
                            validation_score = None
                            is_sucess = False
                            uid = None
                            solution = None
                        else:
                            is_sucess, validation_score, uid, solution = result

                        improved = self.compare_score(direction_baseline_score, validation_score)
                        research_round_idea_results[direction][idea] = {
                            "improved": improved,
                            "is_best_in_direction": False,
                            "score": validation_score,
                        }
                        if improved and is_sucess and solution is not None:
                            direction_best_score = validation_score
                            direction_best_solution = solution
                            direction_best_idea = idea
                            shutil.copy(
                                os.path.join(improve_exp.workspace_path, "submission", f"submission_{uid}.csv"),
                                os.path.join(self.agents.improve_agent.session.config.workspace_path, "best_submission", f"submission.csv"),
                            )
                            save_code_to_file(
                                os.path.join(self.session.config.workspace_path, "best_solution"),
                                "best_solution.py",
                                direction_best_solution,
                            )
                            self.real_time_best_solution = direction_best_solution
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
                knowledge_promotion_workspace_name = f"exp_{self.exp_index}_knowledge_promotion"
                self.exp_index += 1
                knowledge_promotion_results = self.execute_parallel_tasks(
                    [partial(knowledge_promotion_exp.run, task_description=task_description, data_preview=data_preview, base_solution=base_solution, best_solution=self.best_solution, research_plan=research_plan, research_round_idea_results=research_round_idea_results)],
                    max_workers=1,
                    workspace_names=[knowledge_promotion_workspace_name]
                )
                knowledge_promotion_result = knowledge_promotion_results[0]
                if isinstance(knowledge_promotion_result, Exception):
                    self.logger.error(f"Knowledge promotion failed: {knowledge_promotion_result}")
                    raise knowledge_promotion_result
                self.research_plan_and_result.extend([knowledge_promotion_result])
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result
        except GlobalTimeoutInterrupt:
            # 精准捕获看门狗抛出的中断异常
            self.logger.warning(f"看门狗触发：实验已运行满 {RUN_TIMEOUT_SECONDS} 秒，强制打断，开始进行wisdom promotion")
            wisdom_promotion_exp = WisdomPromotionExp(self.agents.wisdom_promotion_agent, self.config, f"exp_{self.exp_index}_wisdom_promotion")
            wisdom_promotion_workspace_name = f"exp_{self.exp_index}_wisdom_promotion"
            self.exp_index += 1
            wisdom_promotion_results = self.execute_parallel_tasks(
                [partial(wisdom_promotion_exp.run, task_description=task_description, best_solution=self.real_time_best_solution)],
                max_workers=1,
                workspace_names=[wisdom_promotion_workspace_name]
            )
            self.logger.info(f"Wisdom promotion finished")
            self.logger.info(f"Task descriptor: {self.prefetch_descriptor}")
            self.logger.info(f"Wisdom promotion result: {wisdom_promotion_results}")
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
            if 'watchdog' in locals():
                watchdog.stop()
            self.cleanup()

    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 3, workspace_names: List[str] | None = None) -> List[Any]:
        """通用并行任务执行器"""
        self.logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers.")
        
        results = [None] * len(tasks)
        
        # 检查是否启用了并行资源分配
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        parallel_enabled = parallel_config.get("enabled", False)
        split_workspace = parallel_config.get("split_workspace_for_exp", False)
        
        # 【新增】用于记录当前正在运行的子线程 ID，以便在发生全局中断时一并强杀
        active_worker_tids = set()
        tids_lock = threading.Lock()

        # 包装任务函数，设置并行索引和独立工作空间
        def wrap_task(task_func, parallel_index):
            def wrapped():
                current_tid = threading.get_ident()
                with tids_lock:
                    active_worker_tids.add(current_tid)
                    
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
                                self.logger.info(f"Exp {parallel_index} 使用独立工作空间: {exp_workspace}")
                                
                    return task_func()
                    
                except GlobalTimeoutInterrupt:
                    self.logger.warning(f"并行任务 {parallel_index} (TID: {current_tid}) 收到中断信号，正在退出并释放资源...")
                    raise  # 继续抛出以便 Executor 捕获并标记 Future 为失败
                finally:
                    # 清理线程本地状态
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession
                        if isinstance(self.session, LocalSession):
                            self.session.set_parallel_index(None)
                            if split_workspace:
                                self.session.set_workspace_path(None)
                                
                    # 【新增】任务结束，移除线程 ID 记录
                    with tids_lock:
                        active_worker_tids.discard(current_tid)
            return wrapped
        
        executor = ThreadPoolExecutor(max_workers=max_workers)
        wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
        future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

        try:
            from concurrent.futures import wait, FIRST_COMPLETED
            not_done = set(future_to_index.keys())
            
            while not_done:
                done, not_done = wait(
                    not_done, 
                    timeout=0.5, 
                    return_when=FIRST_COMPLETED
                )
                
                for future in done:
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        results[index] = exc

            self.logger.info("Parallel execution completed.")
            return results

        finally:
            # 1. 取消所有还在排队、未开始的 Future
            for future in future_to_index:
                future.cancel()
            
            # 【新增】2. 向所有仍在运行的子线程主动注入全局超时异常
            # 这会强制正在执行 task_func 的子线程跳入 wrapped() 的 finally 块
            with tids_lock:
                for tid in active_worker_tids:
                    try:
                        _async_raise(tid, GlobalTimeoutInterrupt)
                    except Exception as e:
                        self.logger.error(f"无法向子线程 {tid} 发送中断信号: {e}")
            
            # 3. 强行关闭线程池
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
