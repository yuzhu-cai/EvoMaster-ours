"""任务调度器

将飞书消息分发到线程池，通过 ChatSessionManager 实现多轮对话上下文延续。
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_playgrounds_imported = False


def _ensure_playgrounds_imported(project_root: Path) -> None:
    """确保所有 playground 模块已导入（触发 @register_playground 装饰器）

    复用 run.py:auto_import_playgrounds() 的逻辑。
    """
    global _playgrounds_imported
    if _playgrounds_imported:
        return

    # 确保 project_root 在 sys.path 中
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    playground_dir = project_root / "playground"
    if not playground_dir.exists():
        logger.warning("Playground directory not found: %s", playground_dir)
        _playgrounds_imported = True
        return

    imported_count = 0
    for agent_dir in playground_dir.iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue

        module_name = f"playground.{agent_dir.name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Imported playground: %s", module_name)
            imported_count += 1
        except ImportError as e:
            logger.warning("Failed to import %s: %s", module_name, e)
        except Exception as e:
            logger.warning("Error importing %s: %s", module_name, e)

    logger.info("Auto-imported %d playground modules", imported_count)
    _playgrounds_imported = True


def _extract_final_answer(result: dict[str, Any]) -> str:
    """从执行结果中提取最终回答"""
    from evomaster.core import extract_agent_response

    trajectory = result.get("trajectory")
    if not trajectory:
        error = result.get("error", "")
        if error:
            return f"任务执行失败: {error}"
        return f"任务完成，状态: {result.get('status', 'unknown')}"

    answer = extract_agent_response(trajectory)
    if answer:
        return answer

    status = result.get("status", "unknown")
    steps = result.get("steps", 0)
    return f"任务完成（状态: {status}，步骤: {steps}），但未提取到文本回答。"


class TaskDispatcher:
    """任务调度器：通过会话管理实现多轮对话上下文延续"""

    def __init__(
        self,
        project_root: Path,
        default_agent: str = "chat_agent",
        default_config_path: Optional[str] = None,
        max_workers: int = 4,
        task_timeout: int = 600,
        max_sessions: int = 100,
        on_result: Optional[Callable[[str, str, str], None]] = None,
        step_reporter_factory: Optional[Callable[[str, str | None], Any]] = None,
    ):
        """
        Args:
            project_root: 项目根目录
            default_agent: 默认 agent 名称
            default_config_path: 默认配置文件路径（相对于 project_root）
            max_workers: 最大并发线程数
            task_timeout: 单轮任务超时（秒）
            max_sessions: 最大并发会话数
            on_result: 结果回调 (chat_id, message_id, result_text) -> None
            step_reporter_factory: 创建 FeishuStepReporter 的工厂函数
        """
        from .session_manager import ChatSessionManager

        self._project_root = project_root
        self._default_agent = default_agent
        self._default_config_path = default_config_path
        self._task_timeout = task_timeout
        self._on_result = on_result
        self._step_reporter_factory = step_reporter_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feishu-task",
        )
        self._active_tasks: dict[str, Any] = {}
        self._session_manager = ChatSessionManager(max_sessions=max_sessions)

        # 预加载 playgrounds
        _ensure_playgrounds_imported(project_root)

    def dispatch(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: Optional[str] = None,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """提交任务到线程池

        特殊命令：
        - /new: 清除当前会话上下文
        - /shutdown: 关闭 bot 进程
        """
        stripped = task_text.strip()

        # /new 命令：清除会话
        if stripped == "/new":
            self._session_manager.remove(chat_id)
            if self._on_result:
                self._on_result(chat_id, message_id, "新会话已开始，上下文已清除。")
            return

        # /shutdown 命令：关闭 bot
        if stripped == "/shutdown":
            self._handle_shutdown(chat_id, message_id)
            return

        agent = agent_name or self._default_agent
        future = self._executor.submit(
            self._run_task_with_session,
            chat_id,
            message_id,
            task_text,
            agent,
            sender_open_id,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(lambda f: self._on_task_done(f, chat_id, message_id))

        # 超时守护线程
        def _timeout_guard():
            try:
                future.result(timeout=self._task_timeout)
            except TimeoutError:
                logger.warning(
                    "Task timed out: message_id=%s, timeout=%ds",
                    message_id,
                    self._task_timeout,
                )
                future.cancel()
            except Exception:
                pass

        threading.Thread(
            target=_timeout_guard,
            daemon=True,
            name=f"timeout-{message_id[:8]}",
        ).start()

    def _create_playground(self, agent_name: str):
        """创建 playground 实例（不调用 setup）。"""
        from evomaster.core import get_playground_class

        if self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        playground = get_playground_class(agent_name, config_path=config_path)

        # 创建 run 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._project_root / "runs" / f"feishu_{agent_name}_{timestamp}"
        task_id = f"feishu_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        return playground

    def _run_task_with_session(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: str,
        sender_open_id: Optional[str] = None,
    ) -> str:
        """在后台线程中执行任务，复用会话上下文。

        如果 agent_name 与默认 agent 不同，采用子任务模式：
        独立运行指定 agent，结果注入 chat_agent 上下文。
        """
        from evomaster.utils.types import TaskInstance

        # 始终用默认 agent 创建/获取 session
        session = self._session_manager.get_or_create(
            chat_id,
            playground_factory=lambda: self._create_playground(self._default_agent),
        )

        # 同一 chat 串行处理
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            # 创建实时进度报告器
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    reporter.send_initial_card(task_text)
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter")

            try:
                # 子任务模式：/agent 指定了非默认 agent
                if agent_name != self._default_agent:
                    answer = self._run_subtask(agent_name, task_text, on_step)

                    # 将结果注入 chat_agent 的 dialog 作为上下文
                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"用户请求: {task_text}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)

                    if reporter:
                        try:
                            reporter.finalize("completed", answer)
                        except Exception:
                            logger.exception("Failed to finalize step reporter")

                    return answer

                # 正常 chat_agent 流程
                if not session.initialized:
                    # 首次消息：完整 setup + agent.run()
                    logger.info(
                        "First message in session chat_id=%s, running setup",
                        chat_id,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    task = TaskInstance(
                        task_id=f"feishu_{message_id}",
                        task_type="chat",
                        description=task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    # 后续消息：continue_run()
                    logger.info(
                        "Continuing session chat_id=%s (message #%d)",
                        chat_id,
                        session.message_count,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                logger.info(
                    "Task completed in session chat_id=%s, status=%s",
                    chat_id,
                    trajectory.status,
                )

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                        return None  # 卡片已包含回答，不需要额外消息
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                return answer

            except Exception as e:
                logger.exception("Task failed in session chat_id=%s", chat_id)
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception(
                            "Failed to finalize step reporter on error"
                        )
                return f"任务执行出错: {e}"

    def _run_subtask(
        self, agent_name: str, task_text: str, on_step: Optional[Callable] = None
    ) -> str:
        """独立运行指定 agent 的子任务，不复用会话上下文。"""
        from evomaster.utils.types import TaskInstance

        logger.info("Running subtask with agent=%s", agent_name)
        playground = self._create_playground(agent_name)
        try:
            playground.setup()
            playground._setup_trajectory_file()
            agent = playground.agent
            task = TaskInstance(
                task_id=f"subtask_{agent_name}",
                task_type="subtask",
                description=task_text,
            )
            trajectory = agent.run(task, on_step=on_step)
            return _extract_final_answer(
                {"trajectory": trajectory, "status": trajectory.status}
            )
        except Exception as e:
            logger.exception("Subtask failed: agent=%s", agent_name)
            return f"子任务执行出错: {e}"
        finally:
            try:
                playground.cleanup()
            except Exception:
                logger.exception("Subtask cleanup failed")

    def _on_task_done(self, future, chat_id: str, message_id: str) -> None:
        """任务完成回调"""
        self._active_tasks.pop(message_id, None)

        try:
            result_text = future.result(timeout=0)
        except TimeoutError:
            result_text = f"任务超时（超过 {self._task_timeout} 秒）"
        except Exception as e:
            result_text = f"任务执行异常: {e}"

        # None 表示 reporter 卡片已包含回答，无需再发消息
        if result_text is None:
            return

        if self._on_result:
            try:
                self._on_result(chat_id, message_id, result_text)
            except Exception:
                logger.exception("Error in on_result callback")

    def _handle_shutdown(self, chat_id: str, message_id: str) -> None:
        """处理 /shutdown 命令：关闭 bot 进程。"""
        logger.info("Shutdown requested from chat_id=%s", chat_id)
        if self._on_result:
            try:
                self._on_result(chat_id, message_id, "Bot 正在关闭...")
            except Exception:
                logger.exception("Error sending shutdown message")

        # 在新线程中执行关闭，避免阻塞当前回调
        def _do_shutdown():
            time.sleep(1)  # 等待回复消息发送完成
            self.shutdown()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_do_shutdown, daemon=True).start()

    def shutdown(self, wait: bool = False) -> None:
        """关闭调度器和所有会话"""
        logger.info("Shutting down task dispatcher...")
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shut down")
