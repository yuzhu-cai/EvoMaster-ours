"""任务调度器

将飞书消息分发到线程池，通过 ChatSessionManager 实现多轮对话上下文延续。
"""

from __future__ import annotations

import importlib
import logging
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

    # 收集需要扫描的 agent 目录：顶层 + _generated/ 子目录
    agent_dirs: list[tuple[Path, str]] = []  # (dir_path, module_prefix)
    for child in playground_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "_generated":
            for gen_dir in child.iterdir():
                if gen_dir.is_dir() and not gen_dir.name.startswith("_"):
                    agent_dirs.append((gen_dir, f"playground._generated.{gen_dir.name}"))
        elif not child.name.startswith("_"):
            agent_dirs.append((child, f"playground.{child.name}"))

    for agent_dir, module_prefix in agent_dirs:
        module_name = f"{module_prefix}.core.playground"
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

    # 检查是否达到步数上限
    traj_result = getattr(trajectory, "result", None)
    if isinstance(traj_result, dict) and traj_result.get("reason") == "max_turns_exceeded":
        return "超过步数限制"

    answer = extract_agent_response(trajectory)
    if answer:
        return answer

    status = result.get("status", "unknown")
    steps = result.get("steps", 0)
    return f"任务完成（状态: {status}，步骤: {steps}），但未提取到文本回答。"


# 需要多轮会话的子任务 agent（使用独立会话 key）
_SESSION_SUBTASK_AGENTS = {"agent_builder"}

# 完成后需要确认按钮的子任务 agent（Phase 1 结束后显示「确认生成」按钮）
_CONFIRM_SUBTASK_AGENTS = {"agent_builder"}


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
        feishu_app_id: Optional[str] = None,
        feishu_app_secret: Optional[str] = None,
        feishu_domain: str = "https://open.feishu.cn",
        feishu_doc_folder_token: Optional[str] = None,
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
            feishu_app_id: 飞书 App ID（用于注入飞书特有工具）
            feishu_app_secret: 飞书 App Secret
            feishu_domain: 飞书 API 域名
            feishu_doc_folder_token: 飞书文件夹 token（用于文档写入工具）
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

        # 存储飞书凭证（用于动态创建工具）
        self._feishu_app_id = feishu_app_id
        self._feishu_app_secret = feishu_app_secret
        self._feishu_domain = feishu_domain
        self._feishu_doc_folder_token = feishu_doc_folder_token

        # 飞书 Client（用于 patch 卡片等操作）
        self._feishu_client = None
        if feishu_app_id and feishu_app_secret:
            from .messaging.client import create_feishu_client
            self._feishu_client = create_feishu_client(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
                domain=feishu_domain,
            )

        # 飞书特有工具（所有 agent 共用）
        self._feishu_tools: list = []
        if feishu_app_id and feishu_app_secret:
            from .tools.doc_reader import FeishuDocReadTool

            self._feishu_tools.append(
                FeishuDocReadTool(
                    app_id=feishu_app_id,
                    app_secret=feishu_app_secret,
                    domain=feishu_domain,
                )
            )

        # 确保 _generated 目录存在（agent_builder 生成的 agent 放在这里）
        (project_root / "configs" / "_generated").mkdir(parents=True, exist_ok=True)
        (project_root / "playground" / "_generated").mkdir(parents=True, exist_ok=True)

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
        - /help: 显示使用帮助
        """
        stripped = task_text.strip()

        # /new 命令：清除会话
        if stripped == "/new":
            self._session_manager.remove(chat_id)
            # 同时清除该 chat 的所有会话级子任务会话
            for agent_name in _SESSION_SUBTASK_AGENTS:
                self._session_manager.remove(f"{chat_id}:{agent_name}")
            self._send_welcome_card(chat_id, message_id)
            return

        # /help 命令：显示使用帮助
        if stripped == "/help":
            self._send_help_card(chat_id, message_id)
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

        if agent_name == self._default_agent and self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"
            # Fallback: 检查 _generated 目录（agent_builder 生成的 agent 放在这里）
            if not config_path.exists():
                config_path = self._project_root / "configs" / "_generated" / agent_name / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 动态导入 _generated 下的 playground（可能在 bot 启动后才生成）
        self._try_import_generated_playground(agent_name)

        playground = get_playground_class(agent_name, config_path=config_path)

        # 创建 run 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._project_root / "runs" / f"feishu_{agent_name}_{timestamp}"
        task_id = f"feishu_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        return playground

    def _try_import_generated_playground(self, agent_name: str) -> None:
        """尝试动态导入 _generated 下的 playground 模块。

        agent_builder 生成的 agent 可能在 bot 启动后才创建，
        启动时的 _ensure_playgrounds_imported 不会扫描到它们。
        """
        from evomaster.core.registry import _PLAYGROUND_REGISTRY

        if agent_name in _PLAYGROUND_REGISTRY:
            return  # 已注册，无需再导入

        module_name = f"playground._generated.{agent_name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Dynamically imported generated playground: %s", module_name)
        except ImportError:
            pass  # 没有自定义 playground，将 fallback 到 BasePlayground
        except Exception:
            logger.warning("Error importing generated playground: %s", module_name, exc_info=True)

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
                    # 会话级子任务：支持多轮对话（如 agent_builder）
                    if agent_name in _SESSION_SUBTASK_AGENTS:
                        answer, sub_trajectory = self._run_session_subtask(
                            chat_id, agent_name, task_text, on_step, sender_open_id
                        )
                    else:
                        answer = self._run_subtask(agent_name, task_text, on_step)
                        sub_trajectory = None

                    # 检查 waiting_for_input（agent 在向用户提问）
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                sub_session_key = f"{chat_id}:{agent_name}"
                                sub_session = self._session_manager.get(sub_session_key)
                                self._finalize_subtask_with_question(
                                    reporter, sub_trajectory, sub_session_key,
                                    agent_name, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return answer

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
                            # 确认类 agent：finalize 时添加确认/取消按钮
                            if agent_name in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{agent_name}"
                                # 截断 answer 嵌入按钮 value，回调时用于保留原始卡片内容
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                            else:
                                reporter.finalize("completed", answer)
                            return None  # 卡片已包含回答
                        except Exception:
                            logger.exception("Failed to finalize step reporter")

                    return answer

                # === 活跃子任务路由 ===
                # 如果有活跃的子任务会话（如 agent_builder planner），
                # 后续消息直接路由过去，支持多轮修改 plan
                active_subtask = self._find_active_subtask(chat_id)
                if active_subtask:
                    # 多轮修改：patch 旧卡片移除按钮
                    sub_session_key = f"{chat_id}:{active_subtask}"
                    sub_session = self._session_manager.get(sub_session_key)
                    if sub_session and sub_session.last_card_message_id and self._feishu_client:
                        self._patch_phase1_card(
                            sub_session.last_card_message_id,
                            "📝 方案修改中",
                            "用户正在修改方案，请以最新卡片为准。",
                            "grey",
                        )
                        sub_session.last_card_message_id = None

                    answer, sub_trajectory = self._run_session_subtask(
                        chat_id, active_subtask, task_text, on_step, sender_open_id
                    )

                    # 检查 waiting_for_input（agent 在向用户提问）
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                self._finalize_subtask_with_question(
                                    reporter, sub_trajectory, sub_session_key,
                                    active_subtask, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return answer

                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {active_subtask}]\n"
                            f"用户请求: {task_text}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)
                    if reporter:
                        try:
                            if active_subtask in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{active_subtask}"
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": active_subtask,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": active_subtask,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                                # 存储当前卡片 ID，下次多轮时可 patch 移除按钮
                                if sub_session:
                                    sub_session.last_card_message_id = reporter.card_message_id
                            else:
                                reporter.finalize("completed", answer)
                            return None
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

                    # 注入飞书特有工具
                    self._inject_feishu_tools(session.playground)
                    self._inject_ask_user_tool(session.agent)

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

                # === ask_user 检测 ===
                # chat_agent 调用了 ask_user，显示问题卡片
                if trajectory and trajectory.status == "waiting_for_input":
                    if reporter:
                        try:
                            questions = (trajectory.result or {}).get("questions", [])
                            question_text = self._format_questions_for_card(questions)
                            # chat_agent 的 ask_user 按钮也走 answer_question，
                            # 但 session_key 用 chat_id 本身（后续消息自然路由回 chat_agent）
                            option_actions = self._build_question_actions(
                                questions, chat_id, "chat_agent"
                            )
                            reporter.finalize_as_question(question_text, actions=option_actions)
                            return None
                        except Exception:
                            logger.exception("Failed to finalize chat_agent question card")
                    return _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )

                # === 委派检测 ===
                # chat_agent 可能通过 delegate_to_agent 工具触发了委派
                delegation = self._check_delegation(session)
                if delegation:
                    delegated_agent = delegation["agent_name"]
                    delegated_task = delegation["task"]
                    logger.info(
                        "Delegation detected: agent=%s, task=%s",
                        delegated_agent, delegated_task[:100],
                    )

                    # 先 finalize chat_agent 的卡片（显示委派消息）
                    chat_answer = _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )
                    if reporter:
                        try:
                            reporter.finalize("completed", chat_answer)
                        except Exception:
                            logger.exception("Failed to finalize chat reporter")

                    # 创建子任务的 reporter
                    subtask_reporter = None
                    subtask_on_step = None
                    if self._step_reporter_factory:
                        try:
                            subtask_reporter = self._step_reporter_factory(
                                chat_id, message_id, sender_open_id
                            )
                            subtask_reporter.send_initial_card(
                                f"[{delegated_agent}] {delegated_task[:200]}"
                            )
                            subtask_on_step = subtask_reporter.on_step
                        except Exception:
                            logger.exception("Failed to create subtask reporter")

                    answer, sub_trajectory = self._run_session_subtask(
                        chat_id, delegated_agent, delegated_task,
                        subtask_on_step, sender_open_id,
                    )

                    # 检查 waiting_for_input（agent 在向用户提问）
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if subtask_reporter:
                            try:
                                sub_session_key = f"{chat_id}:{delegated_agent}"
                                sub_session = self._session_manager.get(sub_session_key)
                                self._finalize_subtask_with_question(
                                    subtask_reporter, sub_trajectory, sub_session_key,
                                    delegated_agent, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return None

                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {delegated_agent}]\n"
                            f"用户请求: {delegated_task}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)
                    if subtask_reporter:
                        try:
                            if delegated_agent in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{delegated_agent}"
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": delegated_agent,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": delegated_agent,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                subtask_reporter.finalize(
                                    "completed", answer, actions=actions
                                )
                                # 存储卡片 ID，下次多轮时可 patch 移除旧按钮
                                sub_session = self._session_manager.get(session_key)
                                if sub_session:
                                    sub_session.last_card_message_id = subtask_reporter.card_message_id
                            else:
                                subtask_reporter.finalize("completed", answer)
                            return None
                        except Exception:
                            logger.exception("Failed to finalize subtask reporter")
                    return None

                # 无委派：正常返回
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
            self._inject_feishu_tools(playground)
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

    def _inject_feishu_tools(self, playground) -> None:
        """将飞书特有工具注入 playground 的所有 agent。"""
        if not self._feishu_tools:
            return
        for agent in playground.agents.values():
            for tool in self._feishu_tools:
                agent.tools.register(tool)

    def _run_session_subtask(
        self,
        chat_id: str,
        agent_name: str,
        task_text: str,
        on_step: Optional[Callable] = None,
        sender_open_id: Optional[str] = None,
    ) -> tuple[str, Any]:
        """运行会话级子任务：支持多轮对话的独立 agent 会话。

        使用 {chat_id}:{agent_name} 作为会话 key，支持 continue_run()。

        Returns:
            (answer_text, trajectory) 元组。trajectory 可能为 None（异常时）。
        """
        from evomaster.utils.types import TaskInstance

        session_key = f"{chat_id}:{agent_name}"
        session = self._session_manager.get_or_create(
            session_key,
            playground_factory=lambda: self._create_playground(agent_name),
        )

        # 会话级子任务也串行处理
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            try:
                if not session.initialized:
                    logger.info(
                        "First message in session subtask key=%s, agent=%s",
                        session_key, agent_name,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    self._inject_feishu_tools(session.playground)
                    self._inject_doc_write_tool(session.playground, sender_open_id)
                    self._inject_ask_user_tool(session.agent)

                    task = TaskInstance(
                        task_id=f"session_subtask_{agent_name}",
                        task_type="session_subtask",
                        description=task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    logger.info(
                        "Continuing session subtask key=%s (message #%d)",
                        session_key, session.message_count,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                return answer, trajectory

            except Exception as e:
                logger.exception(
                    "Session subtask failed: key=%s, agent=%s", session_key, agent_name
                )
                return f"会话子任务执行出错: {e}", None

    def _inject_doc_write_tool(self, playground, sender_open_id: str | None) -> None:
        """将飞书文档写入工具注入 playground 的所有 agent。"""
        if not self._feishu_app_id or not self._feishu_app_secret:
            return

        from .messaging.client import create_feishu_client
        from .messaging.document import FeishuDocumentWriter
        from playground.agent_builder.tools.feishu_doc_write import FeishuDocWriteTool

        client = create_feishu_client(
            app_id=self._feishu_app_id,
            app_secret=self._feishu_app_secret,
            domain=self._feishu_domain,
        )
        writer = FeishuDocumentWriter(
            client,
            folder_token=self._feishu_doc_folder_token,
            domain=self._feishu_domain,
        )
        tool = FeishuDocWriteTool(
            document_writer=writer,
            sender_open_id=sender_open_id,
        )

        for agent in playground.agents.values():
            agent.tools.register(tool)

    @staticmethod
    def _inject_ask_user_tool(agent) -> None:
        """注入 ask_user 工具（仅在交互式上下文中使用）。"""
        from evomaster.interface.tools.ask_user import AskUserTool
        agent.tools.register(AskUserTool())

    @staticmethod
    def _format_questions_for_card(questions: list[dict]) -> str:
        """格式化问题为卡片 markdown。"""
        parts = []
        for q in questions:
            parts.append(f"**{q.get('question', '')}**")
            for opt in q.get("options", []):
                desc = f" — {opt['description']}" if opt.get("description") else ""
                parts.append(f"  - {opt['label']}{desc}")
        parts.append("\n> 也可以直接回复文字补充更多细节")
        return "\n".join(parts)

    @staticmethod
    def _build_question_actions(
        questions: list[dict], session_key: str, agent_name: str
    ) -> list[dict]:
        """为第一个问题的选项构建按钮。"""
        if not questions or not questions[0].get("options"):
            return []
        actions = []
        for opt in questions[0]["options"][:4]:
            actions.append({
                "text": opt.get("label", ""),
                "type": "default",
                "value": {
                    "action": "answer_question",
                    "session_key": session_key,
                    "agent_name": agent_name,
                    "answer_text": opt.get("label", ""),
                },
            })
        return actions

    def _finalize_subtask_with_question(
        self, reporter, trajectory, sub_session_key: str, agent_name: str, sub_session
    ) -> None:
        """当子任务返回 waiting_for_input 时，显示问题卡片。"""
        questions = (getattr(trajectory, "result", None) or {}).get("questions", [])
        question_text = self._format_questions_for_card(questions)
        option_actions = self._build_question_actions(
            questions, sub_session_key, agent_name
        )
        reporter.finalize_as_question(question_text, actions=option_actions)
        if sub_session:
            sub_session.last_card_message_id = reporter.card_message_id

    def dispatch_card_action(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
        action_type: str = "confirm",
    ) -> None:
        """处理卡片按钮回调，触发会话级子任务的 continue_run。

        Args:
            chat_id: 聊天 ID（用于发送结果）
            session_key: 会话 key（格式 {chat_id}:{agent_name}）
            agent_name: agent 名称
            task_text: 发送给 agent 的文本（如 "确认"）
            sender_open_id: 操作者 open_id
            card_message_id: 触发按钮的卡片消息 ID
            original_answer: Phase 1 的原始回答内容（用于更新卡片时保留）
            action_type: 按钮类型 ("confirm" = Phase 2 生成, "answer_question" = 回答提问继续 Phase 1)
        """
        message_id = card_message_id or f"card_action_{session_key}"
        future = self._executor.submit(
            self._continue_session_subtask,
            chat_id,
            session_key,
            agent_name,
            task_text,
            sender_open_id,
            card_message_id,
            original_answer,
            action_type,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(
            lambda f: self._on_task_done(f, chat_id, message_id)
        )

    def _continue_session_subtask(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
        action_type: str = "confirm",
    ) -> str | None:
        """继续已有的会话级子任务（由卡片按钮触发）。

        Args:
            action_type: "confirm" = Phase 2 builder run, "answer_question" = continue planner
        """
        session = self._session_manager.get(session_key)
        if session is None or not session.initialized:
            logger.warning(
                "No active session for card action: key=%s", session_key
            )
            return f"会话已过期或不存在，请重新发起 /agent {agent_name} 命令。"

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            # 创建进度报告器
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, card_message_id, sender_open_id
                    )
                    # agent_builder: 延迟发送卡片，等 TODO 解析后一次性发送
                    if agent_name not in _CONFIRM_SUBTASK_AGENTS:
                        reporter.send_initial_card(f"[{agent_name}] {task_text}")
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter for card action")

            try:
                logger.info(
                    "Continuing session subtask via card action: key=%s (message #%d)",
                    session_key, session.message_count,
                )

                # agent_builder 双 agent 模式：Phase 2 使用 builder_agent（全新 run）
                # 仅在 confirm 时触发，answer_question 走 planner continue_run
                if (
                    action_type == "confirm"
                    and agent_name == "agent_builder"
                    and hasattr(session.playground, "agents")
                    and hasattr(session.playground.agents, "builder_agent")
                ):
                    from evomaster.utils.types import TaskInstance

                    # 解析 planner 输出中的 TODO 清单，设置到 reporter
                    todo_items = self._parse_plan_todos(original_answer)
                    if reporter:
                        if todo_items:
                            reporter.set_todo_items(todo_items)
                        reporter.send_initial_card(
                            f"[{agent_name}] 正在生成 Agent 文件..."
                        )
                        on_step = reporter.on_step

                    builder_agent = session.playground.agents.builder_agent
                    # 注入飞书工具到 builder agent（setup 时已注入，但确保可用）
                    if self._feishu_tools:
                        for tool in self._feishu_tools:
                            builder_agent.tools.register(tool)
                    # 构造 handoff 任务：将 planner 的方案摘要传递给 builder
                    plan_task = TaskInstance(
                        task_id=f"builder_{agent_name}",
                        task_type="builder",
                        description=(
                            "请根据以下设计方案生成 Agent 文件。\n\n"
                            f"## 方案摘要\n{original_answer}\n\n"
                            "请使用 feishu_doc_read 工具读取飞书文档获取完整方案，然后生成所有文件。"
                        ),
                    )
                    trajectory = builder_agent.run(plan_task, on_step=on_step)
                else:
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )
                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

                # === answer_question 路径：planner continue_run 后的处理 ===
                if action_type == "answer_question":
                    # 检查 planner 是否又在提问
                    if trajectory and trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                self._finalize_subtask_with_question(
                                    reporter, trajectory, session_key,
                                    agent_name, session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card (answer_question)")
                        return None

                    # planner 完成了：显示 confirm/cancel 按钮
                    if reporter:
                        try:
                            if agent_name in _CONFIRM_SUBTASK_AGENTS:
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                                session.last_card_message_id = reporter.card_message_id
                            else:
                                reporter.finalize("completed", answer)
                        except Exception:
                            logger.exception("Failed to finalize step reporter (answer_question)")

                    # 将结果注入 chat_agent 上下文
                    chat_session = self._session_manager.get(chat_id)
                    if chat_session and chat_session.initialized and chat_session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"结果: {answer}"
                        )
                        chat_session.agent.add_user_message(summary)

                    return None

                # === confirm 路径：Phase 2 builder 完成后的处理 ===
                # 将结果注入 chat_agent 上下文
                chat_session = self._session_manager.get(chat_id)
                if chat_session and chat_session.initialized and chat_session.agent:
                    summary = (
                        f"[子任务结果 - {agent_name} Phase 2]\n"
                        f"结果: {answer}"
                    )
                    chat_session.agent.add_user_message(summary)

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                # 更新 Phase 1 卡片：从 "生成中" 变为 "完成"，保留原始计划内容
                phase1_content = original_answer + "\n\n---\n> ✅ Agent 已成功创建。详情请查看下方回复。" if original_answer else "Agent 已成功创建。\n\n详情请查看下方回复。"
                self._patch_phase1_card(
                    card_message_id, "✅ Agent 创建完成",
                    phase1_content, "green",
                )

                # Phase 2 完成，清理子任务会话，后续消息重新走 chat_agent
                self._session_manager.remove(session_key)

                return None

            except Exception as e:
                logger.exception(
                    "Card action subtask failed: key=%s", session_key
                )
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception("Failed to finalize reporter on error")

                if action_type == "confirm":
                    # 更新 Phase 1 卡片：显示失败状态，保留原始计划内容
                    phase1_content = original_answer + f"\n\n---\n> ❌ Agent 创建过程中出错：{str(e)[:500]}" if original_answer else f"Agent 创建过程中出错。\n\n{str(e)[:500]}"
                    self._patch_phase1_card(
                        card_message_id, "❌ Agent 创建失败",
                        phase1_content, "red",
                    )

                return f"会话子任务执行出错: {e}"

    def _patch_phase1_card(
        self,
        card_message_id: str | None,
        title: str,
        content: str,
        header_template: str,
    ) -> None:
        """更新 Phase 1 卡片状态（Phase 2 完成/失败后调用）。"""
        if not card_message_id or not self._feishu_client:
            return
        try:
            from .messaging.sender import patch_card_message
            patch_card_message(
                self._feishu_client,
                card_message_id,
                title=title,
                content=content,
                header_template=header_template,
            )
        except Exception:
            logger.exception("Failed to update Phase 1 card: %s", card_message_id)

    @staticmethod
    def _check_delegation(session) -> dict[str, str] | None:
        """检查 chat_agent 是否通过 delegate_to_agent 触发了委派。

        扫描 trajectory 最近几步的 ToolMessage，查找 delegated=True 标记。
        """
        if not session.initialized or not session.agent:
            return None
        traj = session.agent.trajectory
        if not traj or not traj.steps:
            return None
        for step in reversed(traj.steps[-3:]):
            for resp in step.tool_responses:
                if getattr(resp, "name", "") == "delegate_to_agent":
                    info = (getattr(resp, "meta", None) or {}).get("info", {})
                    if info.get("delegated"):
                        return {
                            "agent_name": info["agent_name"],
                            "task": info["task"],
                        }
        return None

    def _find_active_subtask(self, chat_id: str) -> str | None:
        """查找该 chat 下是否有活跃的子任务会话。

        如果存在，后续消息直接路由到子任务会话（支持多轮修改 plan 等）。
        """
        for agent_name in _SESSION_SUBTASK_AGENTS:
            session_key = f"{chat_id}:{agent_name}"
            sub = self._session_manager.get(session_key)
            if sub and sub.initialized:
                logger.info(
                    "Active subtask session found: key=%s, routing there",
                    session_key,
                )
                return agent_name
        return None

    @staticmethod
    def _parse_plan_todos(plan_text: str) -> list[str]:
        """从 planner 输出中解析 TODO 列表。

        格式::

            ---PLAN_TODO---
            - [ ] 创建目录结构
            - [ ] 创建 system_prompt.txt
            ---END_TODO---
        """
        todos: list[str] = []
        in_todo = False
        for line in plan_text.split("\n"):
            stripped = line.strip()
            if "---PLAN_TODO---" in stripped:
                in_todo = True
                continue
            if "---END_TODO---" in stripped:
                break
            if in_todo and stripped.startswith("- [ ]"):
                label = stripped[5:].strip()
                if label:
                    todos.append(label)
        return todos

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

    def _send_welcome_card(self, chat_id: str, message_id: str) -> None:
        """发送欢迎卡片，介绍 bot 功能和使用方法。"""
        if not self._feishu_client:
            # fallback: 纯文本
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "新会话已开始。直接发送消息即可对话，或使用 /agent <名称> <任务> 调用专属智能体。",
                )
            return

        from .messaging.sender import send_card_message

        content = (
            "**直接对话**\n"
            "发送任何消息即可开始对话，我会记住上下文进行多轮交流。\n\n"
            "**创建智能体**\n"
            "直接告诉我你想创建什么智能体，例如：「帮我创建一个能总结文档的 agent」，"
            "我会自动委派给 Agent Builder 完成设计与构建。\n\n"
            "**指定智能体执行任务**\n"
            "`/agent <名称> <任务描述>`\n"
            "例如：`/agent doc_summarizer 总结这个文件 README.md`\n\n"
            "---\n"
            "**常用命令**\n"
            "`/help` — 显示本帮助信息\n"
            "`/new` — 清除上下文，开始新会话"
        )

        send_card_message(
            self._feishu_client,
            chat_id,
            title="👋 新会话已开始",
            content=content,
            reply_to_message_id=message_id,
            header_template="green",
        )

    def _send_help_card(self, chat_id: str, message_id: str) -> None:
        """发送使用帮助卡片。"""
        if not self._feishu_client:
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "使用帮助：直接发消息对话；/agent <名称> <任务> 调用智能体；/new 新会话。",
                )
            return

        from .messaging.sender import send_card_message

        content = (
            "**直接对话**\n"
            "发送任何消息即可开始多轮对话，我会记住上下文。\n\n"
            "**创建智能体**\n"
            "直接描述你的需求，例如：「帮我创建一个能总结文档的 agent」，"
            "我会自动委派给 Agent Builder 完成设计与构建。\n\n"
            "**指定智能体执行任务**\n"
            "`/agent <名称> <任务描述>`\n"
            "例如：`/agent doc_summarizer 总结这个文件 README.md`\n\n"
            "---\n"
            "**命令列表**\n"
            "`/help` — 显示本帮助信息\n"
            "`/new` — 清除上下文，开始新会话"
        )

        send_card_message(
            self._feishu_client,
            chat_id,
            title="📖 使用帮助",
            content=content,
            reply_to_message_id=message_id,
            header_template="blue",
        )

    def shutdown(self, wait: bool = False) -> None:
        """关闭调度器和所有会话"""
        logger.info("Shutting down task dispatcher...")
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shut down")
