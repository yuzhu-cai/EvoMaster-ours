import os
import logging
import sys
import json
from pathlib import Path
import shutil
import copy
import ctypes
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
from .utils.data_preview import generate
from .utils.code import save_code_to_file
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

RUN_TIMEOUT_SECONDS = 5*60
# 必须继承自 BaseException，防止被底层的 except Exception: 吞噬
class GlobalTimeoutInterrupt(BaseException):
    """用于看门狗强制打断的全局超时异常"""
    pass

def _async_raise(target_tid, exception_type):
    """通过 C-API 向指定线程强制抛出异常"""
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(target_tid), 
        ctypes.py_object(exception_type)
    )
    if ret == 0:
        raise ValueError("无效的线程 ID")
    elif ret > 1:
        # 如果返回值大于 1，说明状态异常，需要撤销操作
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), None)
        raise SystemError("PyThreadState_SetAsyncExc 调用失败")

class TimeoutWatchdog:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.main_thread_id = threading.get_ident() # 记录启动看门狗的主线程 ID
        self._thread = None

    def start(self):
        """启动看门狗"""
        self._thread = threading.Thread(target=self._watch, daemon=True, name="TimeoutWatchdog")
        self._thread.start()

    def _watch(self):
        # 等待指定的超时时间，或者直到 stop() 被调用触发 event
        is_cancelled = self.cancel_event.wait(self.timeout_seconds)
        if not is_cancelled:
            # 时间到了，且没有被正常取消 -> 触发主线程中断！
            _async_raise(self.main_thread_id, GlobalTimeoutInterrupt)

    def stop(self):
        """主逻辑正常结束时，调用此方法取消看门狗"""
        self.cancel_event.set()

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
            self.logger.warning(f"看门狗触发：Run 已运行满 {RUN_TIMEOUT_SECONDS} 秒，强制打断当前迭代")
            print("测试代码执行完毕（因超时）")
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
            
            # with ThreadPoolExecutor(max_workers=max_workers) as executor:
            #     # 提交所有任务，建立 future 到 index 的映射，以保证返回顺序
            #     wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
            #     future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

            #     # 处理完成的任务
            #     for future in as_completed(future_to_index):
            #         index = future_to_index[future]
            #         try:
            #             # 获取返回值
            #             result = future.result()
            #             results[index] = result
            #         except Exception as exc:
            #             self.logger.error(f"Task {index} generated an exception: {exc}")
            #             # 将异常对象作为结果返回，避免打断其他任务
            #             results[index] = exc

            # self.logger.info("Parallel execution completed.")
            # return results

            # 【关键修改】：不再使用 with ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=max_workers)
            wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
            future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

            try:
                # 【修改这里】：使用带有短超时的 wait，让主线程定期执行字节码   
                from concurrent.futures import wait, FIRST_COMPLETED
                
                not_done = set(future_to_index.keys())
                
                while not_done:
                    # 每次最多阻塞 0.5 秒。如果没完成，会返回继续 while 循环
                    # 这个瞬间主线程会执行 Python 字节码，从而立刻响应看门狗抛出的 GlobalTimeoutInterrupt
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
                # # 处理完成的任务
                # for future in as_completed(future_to_index):
                #     index = future_to_index[future]
                #     try:
                #         result = future.result()
                #         results[index] = result
                #     except Exception as exc:
                #         self.logger.error(f"Task {index} generated an exception: {exc}")
                #         results[index] = exc

                # self.logger.info("Parallel execution completed.")
                # return results

            finally:
                # 当 TimeoutError 打断 as_completed 时，会直接进入这里
                # 1. 取消所有还在排队、未开始的 Future
                for future in future_to_index:
                    future.cancel()
                
                # 2. 强行关闭线程池，wait=False 表示不等待正在运行的子线程结束
                # cancel_futures=True 是 Python 3.9+ 的特性，能更干净地清理排队任务
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)
