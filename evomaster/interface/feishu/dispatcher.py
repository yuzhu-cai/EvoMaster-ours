"""Task dispatcher

Dispatch Feishu messages to a thread pool, using ChatSessionManager for multi-turn conversation context persistence.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_playgrounds_imported = False


def _ensure_playgrounds_imported(project_root: Path) -> None:
    """Ensure all playground modules are imported (triggering @register_playground decorators).

    Reuses the logic from run.py:auto_import_playgrounds().
    """
    global _playgrounds_imported
    if _playgrounds_imported:
        return

    # Ensure project_root is in sys.path
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    playground_dir = project_root / "playground"
    if not playground_dir.exists():
        logger.warning("Playground directory not found: %s", playground_dir)
        _playgrounds_imported = True
        return

    imported_count = 0

    # Collect agent directories to scan: top-level + _generated/ subdirectories
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
    """Extract the final answer from an execution result."""
    from evomaster.core import extract_agent_response

    trajectory = result.get("trajectory")
    if not trajectory:
        error = result.get("error", "")
        if error:
            return f"任务执行失败: {error}"
        return f"任务完成，状态: {result.get('status', 'unknown')}"

    # Check if step limit was reached
    traj_result = getattr(trajectory, "result", None)
    if isinstance(traj_result, dict) and traj_result.get("reason") == "max_turns_exceeded":
        return "超过步数限制"

    answer = extract_agent_response(trajectory)
    if answer:
        return answer

    status = result.get("status", "unknown")
    steps = result.get("steps", 0)
    return f"任务完成（状态: {status}，步骤: {steps}），但未提取到文本回答。"


# Subtask agents requiring multi-turn sessions (use independent session keys)
# agent_builder 改为每次使用唯一 session key，不再隐式路由后续消息
_SESSION_SUBTASK_AGENTS = set()

# Subtask agents requiring confirmation buttons after completion (show "Confirm Generation" button after Phase 1)
_CONFIRM_SUBTASK_AGENTS = {"agent_builder"}

# 保留同步委派流程的 agent（agent_builder 的 planner→确认→builder 流程）
_SYNCHRONOUS_DELEGATION_AGENTS = {"agent_builder"}

# 始终使用 local 模式的 agent（不进入容器池）
# agent_builder 需要读写宿主机上的 playground/ 和 configs/ 目录
_LOCAL_ONLY_AGENTS = {"agent_builder"}


class TaskDispatcher:
    """Task dispatcher: implements multi-turn conversation context persistence via session management."""

    def __init__(
        self,
        project_root: Path,
        default_agent: str = "magiclaw",
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
        available_agents: dict[str, str] | None = None,
        container_pool: Any = None,
        scheduler_config: Any = None,
    ):
        """
        Args:
            project_root: Project root directory.
            default_agent: Default agent name.
            default_config_path: Default config file path (relative to project_root).
            max_workers: Maximum number of concurrent threads.
            task_timeout: Single-turn task timeout in seconds.
            max_sessions: Maximum number of concurrent sessions.
            on_result: Result callback (chat_id, message_id, result_text) -> None.
            step_reporter_factory: Factory function for creating FeishuStepReporter instances.
            feishu_app_id: Feishu App ID (used for injecting Feishu-specific tools).
            feishu_app_secret: Feishu App Secret.
            feishu_domain: Feishu API domain.
            feishu_doc_folder_token: Feishu folder token (used by document writing tools).
            available_agents: 可用子智能体白名单 {name: description}
            container_pool: ContainerPool 实例，None 表示不使用容器池
        """
        from .session_manager import ChatSessionManager

        self._project_root = project_root
        self._default_agent = default_agent
        self._default_config_path = default_config_path
        self._task_timeout = task_timeout
        self._on_result = on_result
        self._step_reporter_factory = step_reporter_factory
        self._server_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feishu-task",
        )
        self._active_tasks: dict[str, Any] = {}
        self._container_pool = container_pool

        # 会话清理回调：不释放容器 — 用户容器在 bot 生命周期内持久存在
        on_session_cleanup = None
        if self._container_pool is not None:
            def _on_session_cleanup(session):
                # 不释放容器，容器释放只在 ContainerPool.shutdown() 时发生
                pass
            on_session_cleanup = _on_session_cleanup

        self._session_manager = ChatSessionManager(
            max_sessions=max_sessions,
            on_session_cleanup=on_session_cleanup,
        )

        # Store Feishu credentials (used for dynamically creating tools)
        self._feishu_app_id = feishu_app_id
        self._feishu_app_secret = feishu_app_secret
        self._feishu_domain = feishu_domain
        self._feishu_doc_folder_token = feishu_doc_folder_token
        self._available_agents = available_agents or {}

        # 后台任务注册表
        from .background_task import BackgroundTaskRegistry
        self._bg_task_registry = BackgroundTaskRegistry()

        # Feishu Client (used for card patching and other operations)
        self._feishu_client = None
        if feishu_app_id and feishu_app_secret:
            from .messaging.client import create_feishu_client
            self._feishu_client = create_feishu_client(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
                domain=feishu_domain,
            )

        # Feishu-specific tools (shared across all agents)
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

        # Ensure _generated directories exist (agents generated by agent_builder are placed here)
        (project_root / "configs" / "_generated").mkdir(parents=True, exist_ok=True)
        (project_root / "playground" / "_generated").mkdir(parents=True, exist_ok=True)

        # Preload playgrounds
        _ensure_playgrounds_imported(project_root)

        # 定时任务调度器（可选）
        self._scheduler_service = None
        self._scheduler_store = None
        self._scheduler_config = scheduler_config
        if scheduler_config and getattr(scheduler_config, "enabled", False):
            try:
                from evomaster.scheduler import ScheduledJobStore, SchedulerService
                from evomaster.scheduler.service import SchedulerConfig as SchedSvcConfig
                from .scheduler_adapter import FeishuTaskExecutor, FeishuResultNotifier

                db_path = project_root / scheduler_config.db_path
                self._scheduler_store = ScheduledJobStore(db_path)
                executor = FeishuTaskExecutor(self._run_subtask)
                notifier = FeishuResultNotifier(self._feishu_client)
                svc_config = SchedSvcConfig(
                    max_concurrent_runs=scheduler_config.max_concurrent_runs,
                    max_poll_interval=scheduler_config.max_poll_interval,
                )
                self._scheduler_service = SchedulerService(
                    self._scheduler_store, executor, notifier, svc_config,
                )
                self._scheduler_service.start()
                logger.info("Scheduler service started (db=%s)", db_path)
            except Exception:
                logger.exception("Failed to initialize scheduler service")
                self._scheduler_service = None
                self._scheduler_store = None

    def dispatch(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: Optional[str] = None,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """Submit a task to the thread pool.

        Special commands:
        - /new: Clear the current session context.
        - /help: Show usage help.
        """
        stripped = task_text.strip()

        # /new command: clear session
        if stripped == "/new":
            self._session_manager.remove(chat_id)
            # Also clear all session-level subtask sessions for this chat
            for agent_name in _SESSION_SUBTASK_AGENTS:
                self._session_manager.remove(f"{chat_id}:{agent_name}")
            # 清理旧的后台任务记录（仅已完成且已审阅的）
            self._bg_task_registry.cleanup_old(chat_id, max_age_seconds=0)
            self._send_welcome_card(chat_id, message_id)
            return

        # /help command: show usage help
        if stripped == "/help":
            self._send_help_card(chat_id, message_id)
            return

        # /list 命令：显示可用子智能体
        if stripped == "/list":
            self._send_list_card(chat_id, message_id)
            return

        # /stop 命令：停止当前 chat 的所有运行中任务
        if stripped == "/stop":
            self._handle_stop_command(chat_id, message_id)
            return

        # /schedule 命令：管理定时任务
        if stripped.startswith("/schedule"):
            self._handle_schedule_command(chat_id, message_id, stripped, sender_open_id)
            return

        # 白名单校验：/agent 指定的 agent 必须在允许列表中
        if agent_name and agent_name != self._default_agent:
            allowed = self._get_allowed_agent_names()
            if agent_name not in allowed:
                if self._on_result:
                    self._on_result(
                        chat_id, message_id,
                        f"智能体 `{agent_name}` 不在可用列表中。\n发送 /list 查看可用的智能体。",
                    )
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

        # Timeout watchdog thread
        def _timeout_guard():
            """Wait for the task future and handle timeout or errors."""
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

    def _create_playground(self, agent_name: str, sender_open_id: str | None = None, chat_id: str | None = None):
        """Create a playground instance (without calling setup)."""
        from evomaster.core import get_playground_class

        if agent_name == self._default_agent and self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"
            # Fallback: check _generated directory (agents generated by agent_builder are placed here)
            if not config_path.exists():
                config_path = self._project_root / "configs" / "_generated" / agent_name / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # Dynamically import _generated playgrounds (may be generated after bot startup)
        self._try_import_generated_playground(agent_name)

        playground = get_playground_class(agent_name, config_path=config_path)

        # Create hierarchical run directory: runs/feishu_{server_start}/{user_id}/{agent}_{timestamp}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # run_dir: 始终在 feishu_{ts}/{user_id}/ 下（容器池模式时不在挂载目录内）
        if self._container_pool is not None:
            feishu_run_base = Path(self._container_pool.shared_mount_host).parent
        else:
            feishu_run_base = self._project_root / "runs" / f"feishu_{self._server_start_time}"
        user_dir = sender_open_id or "unknown"
        run_dir = feishu_run_base / user_dir / f"{agent_name}_{timestamp}"
        task_id = f"feishu_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        # 容器池模式：注入 use_existing_container
        if self._container_pool is not None and agent_name not in _LOCAL_ONLY_AGENTS:
            # workspace = shared_mount_host（现在是 .../workspaces/）下的用户目录
            user_dir = sender_open_id or "unknown"
            feishu_ws_base = Path(self._container_pool.shared_mount_host)
            user_workspace_host = str((feishu_ws_base / user_dir).absolute())
            container_info = self._container_pool.acquire(
                sender_open_id or "unknown", user_workspace_host
            )
            container_workspace = f"/workspaces/{user_dir}"

            # 覆盖 session 配置
            session_cfg = playground.config.session
            session_cfg["type"] = "docker"
            docker_cfg = session_cfg.setdefault("docker", {})
            docker_cfg["use_existing_container"] = container_info.container_name
            docker_cfg["auto_remove"] = False  # 池化容器不自动删除
            docker_cfg["working_dir"] = container_workspace
            docker_cfg["workspace_path"] = container_workspace
            # 设置 volumes 以便 DockerEnv.is_mounted_path() 优化 host-side 文件 I/O
            docker_cfg["volumes"] = {self._container_pool.shared_mount_host: "/workspaces"}

            # 存储 container_info 到 playground 以便后续释放
            playground._pool_container_info = container_info

        return playground

    def _try_import_generated_playground(self, agent_name: str) -> None:
        """Try to dynamically import a _generated playground module.

        Agents generated by agent_builder may be created after the bot starts,
        so _ensure_playgrounds_imported at startup will not scan for them.
        """
        from evomaster.core.registry import _PLAYGROUND_REGISTRY

        if agent_name in _PLAYGROUND_REGISTRY:
            return  # Already registered, no need to import again

        module_name = f"playground._generated.{agent_name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Dynamically imported generated playground: %s", module_name)
        except ImportError:
            pass  # No custom playground, will fallback to BasePlayground
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
        """Execute a task in a background thread, reusing session context.

        If agent_name differs from the default agent, use subtask mode:
        run the specified agent independently and inject results into the magiclaw context.
        """
        from evomaster.utils.types import TaskInstance

        # Always use the default agent to create/get the session
        session = self._session_manager.get_or_create(
            chat_id,
            playground_factory=lambda: self._create_playground(self._default_agent, sender_open_id, chat_id=chat_id),
        )

        # Serialize processing within the same chat
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            session.playground.register_thread()

            # Create real-time progress reporter
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    # 添加停止按钮
                    reporter.set_stop_actions(
                        self._build_stop_actions(chat_id, target="orchestrator")
                    )
                    reporter.send_initial_card(task_text)
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter")

            # 跟踪当前 reporter 和 running agent（供停止按钮使用）
            session.current_reporter = reporter

            try:
                # Subtask mode: /agent specified a non-default agent
                if agent_name != self._default_agent:
                    # 会话级子任务（如 agent_builder）：后台线程执行，避免阻塞 magiclaw lock
                    if agent_name in _CONFIRM_SUBTASK_AGENTS:
                        # Finalize 当前 reporter（显示"正在启动..."）
                        if reporter:
                            try:
                                reporter.finalize("completed", f"正在启动 {agent_name}...")
                            except Exception:
                                logger.exception("Failed to finalize reporter")
                        # 后台线程执行
                        self._dispatch_sync_delegation(
                            chat_id, agent_name, task_text,
                            message_id, sender_open_id,
                        )
                        return None
                    else:
                        answer = self._run_subtask(agent_name, task_text, on_step, chat_id=chat_id, sender_open_id=sender_open_id)

                    # Inject result into magiclaw's dialog as context
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
                            return None  # 卡片已包含回答
                        except Exception:
                            logger.exception("Failed to finalize step reporter")

                    return answer

                # === Active subtask routing ===
                # If there is an active subtask session (e.g. agent_builder planner),
                # route subsequent messages directly to it for multi-turn plan modification
                active_subtask = self._find_active_subtask(chat_id)
                if active_subtask:
                    # Multi-turn modification: patch old card to remove buttons
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

                    # Check waiting_for_input (agent is asking the user a question)
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
                                # Store current card ID, can be patched to remove buttons on next multi-turn
                                if sub_session:
                                    sub_session.last_card_message_id = reporter.card_message_id
                            else:
                                reporter.finalize("completed", answer)
                            return None
                        except Exception:
                            logger.exception("Failed to finalize step reporter")
                    return answer

                # 正常 magiclaw 流程
                # 包装 on_step：注入委派拦截逻辑（即时 dispatch）
                on_step = self._make_on_step_with_delegation(
                    on_step, session, chat_id, message_id, sender_open_id
                )
                # 获取记忆系统（如果 playground 已初始化）
                memory_manager = getattr(session.playground, "_memory_manager", None)
                memory_config = getattr(session.playground, "_memory_config", {})
                user_id = sender_open_id or "unknown"

                if not session.initialized:
                    # First message: full setup + agent.run()
                    logger.info(
                        "First message in session chat_id=%s, running setup",
                        chat_id,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    # Re-fetch (only available after setup for _memory_manager)
                    memory_manager = getattr(session.playground, "_memory_manager", None)
                    memory_config = getattr(session.playground, "_memory_config", {})

                    # Inject Feishu-specific tools
                    self._inject_feishu_tools(session.playground)
                    self._inject_send_file_tool(session.playground, chat_id)
                    self._inject_ask_user_tool(session.agent)
                    self._inject_memory_tools(session.agent, memory_manager, user_id)
                    self._inject_background_tools(session.agent, chat_id)
                    self._inject_schedule_tools(session.agent, chat_id, sender_open_id or "")

                    # Set up memory extraction hook before compaction
                    if memory_manager and memory_config.get("auto_capture", True):
                        _mm = memory_manager
                        _uid = user_id
                        def _on_compaction(old_messages, mm=_mm, uid=_uid):
                            """Callback invoked on context compaction to capture memories from discarded messages."""
                            from evomaster.utils.types import UserMessage
                            for msg in old_messages:
                                if isinstance(msg, UserMessage):
                                    text = msg.content if isinstance(msg.content, str) else ""
                                    if text:
                                        mm.extract_from_message(uid, text)
                        session.agent.context_manager.on_before_compaction = _on_compaction

                    task = TaskInstance(
                        task_id=f"feishu_{message_id}",
                        task_type="chat",
                        description=task_text,
                    )

                    # Automatically recall related memories (inject into system prompt)
                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config, user_id, task_text,
                    )

                    session.current_running_agent = session.agent
                    session.agent._cancel_event.clear()
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                    self._memory_auto_capture(memory_manager, memory_config, user_id, task_text)
                else:
                    # Subsequent messages: continue_run()
                    logger.info(
                        "Continuing session chat_id=%s (message #%d)",
                        chat_id,
                        session.message_count,
                    )

                    # Automatically recall related memories (inject into system prompt)
                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config, user_id, task_text,
                    )

                    session.current_running_agent = session.agent
                    session.agent._cancel_event.clear()
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                    # Automatically extract memories from user messages
                    self._memory_auto_capture(memory_manager, memory_config, user_id, task_text)

                # === 取消检测 ===
                if trajectory and trajectory.status == "cancelled":
                    logger.info("Task cancelled by user in chat_id=%s", chat_id)
                    if reporter:
                        try:
                            reporter.finalize("cancelled", "任务已被用户停止。")
                        except Exception:
                            logger.exception("Failed to finalize cancelled reporter")
                    session.current_reporter = None
                    session.current_running_agent = None
                    session.dispatched_delegation_keys.clear()
                    return None

                # === ask_user detection ===
                # magiclaw called ask_user, present questions one by one via cards
                if trajectory and trajectory.status == "waiting_for_input":
                    if reporter:
                        try:
                            questions = (trajectory.result or {}).get("questions", [])
                            if questions:
                                # Only present the first question
                                first = [questions[0]]
                                question_text = self._format_questions_for_card(first)
                                option_actions = self._build_question_actions(
                                    first, chat_id, "magiclaw",
                                    question_text=question_text,
                                )
                                reporter.finalize_as_question(question_text, actions=option_actions)
                                # Store remaining questions to session
                                session.pending_questions = questions[1:]
                                session.collected_answers = []
                            return None
                        except Exception:
                            logger.exception("Failed to finalize magiclaw question card")
                    return _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )

                # === 委派检测 ===
                # Safety net: 捕获 on_step 拦截器可能遗漏的委派（如 on_step 异常时）
                safety_delegations = self._check_all_delegations(session)
                for delegation in safety_delegations:
                    delegated_agent = delegation["agent_name"]
                    delegated_task = delegation["task"]
                    key = (delegated_agent, delegated_task)
                    session.dispatched_delegation_keys.add(key)
                    logger.info(
                        "Safety-net delegation detected: agent=%s, task=%s",
                        delegated_agent, delegated_task[:100],
                    )
                    if delegated_agent in _SYNCHRONOUS_DELEGATION_AGENTS:
                        self._dispatch_sync_delegation(
                            chat_id, delegated_agent, delegated_task,
                            message_id, sender_open_id,
                        )
                    else:
                        bg_task = self._bg_task_registry.create(
                            chat_id, delegated_agent, delegated_task
                        )
                        self._dispatch_background_subtask(
                            chat_id, bg_task, message_id, sender_open_id
                        )

                # 判断本轮是否有委派（包括 on_step 已 dispatch 的 + safety net）
                had_delegations = bool(session.dispatched_delegation_keys)

                if had_delegations:
                    # finalize magiclaw 卡片（显示委派消息）
                    chat_answer = _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )
                    if reporter:
                        try:
                            reporter.finalize("completed", chat_answer)
                        except Exception:
                            logger.exception("Failed to finalize chat reporter")

                    # 处理待审阅的后台任务（在仍持有 lock 时）
                    self._process_pending_reviews(session, chat_id, message_id, sender_open_id)
                    # 重置追踪集合（为下一轮用户消息做准备）
                    session.dispatched_delegation_keys.clear()
                    session.current_reporter = None
                    session.current_running_agent = None
                    return None

                # No delegation: return normally
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
                        # 处理待审阅的后台任务（在仍持有 lock 时）
                        self._process_pending_reviews(session, chat_id, message_id, sender_open_id)
                        session.current_reporter = None
                        session.current_running_agent = None
                        return None  # Card already contains the answer, no extra message needed
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                # 处理待审阅的后台任务（在仍持有 lock 时）
                self._process_pending_reviews(session, chat_id, message_id, sender_open_id)
                session.current_reporter = None
                session.current_running_agent = None
                return answer

            except Exception as e:
                logger.exception("Task failed in session chat_id=%s", chat_id)
                session.current_reporter = None
                session.current_running_agent = None
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception(
                            "Failed to finalize step reporter on error"
                        )
                return f"任务执行出错: {e}"

    def _run_subtask(
        self, agent_name: str, task_text: str, on_step: Optional[Callable] = None,
        chat_id: Optional[str] = None, sender_open_id: Optional[str] = None,
        on_agent_ready: Optional[Callable] = None,
    ) -> str:
        """Run a subtask with the specified agent independently, without reusing session context.

        Args:
            on_agent_ready: 可选回调，在 agent 创建后、run() 前调用，签名 (agent) -> None。
                用于外部保存 agent 引用（如停止按钮需要）。
        """
        from evomaster.utils.types import TaskInstance

        logger.info("Running subtask with agent=%s", agent_name)
        playground = self._create_playground(agent_name, sender_open_id, chat_id=chat_id)
        # Register the current thread with the playground (for log filtering)
        playground.register_thread()
        try:
            playground.setup()
            playground._setup_trajectory_file()
            self._inject_feishu_tools(playground)
            if chat_id:
                self._inject_send_file_tool(playground, chat_id)
            agent = playground.agent
            if on_agent_ready:
                on_agent_ready(agent)
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
            # 注意：不释放容器回池。
            # 容器按用户分配，在 bot 生命周期内持久存在，
            # 由 ContainerPool.shutdown() 统一释放。
            # 此处释放会导致: (1) rmtree 删除整个用户目录，
            # (2) pkill 杀死同容器内其他仍在运行的 agent 进程。

    def _inject_feishu_tools(self, playground) -> None:
        """Inject Feishu-specific tools into all agents in the playground."""
        if not self._feishu_tools:
            return
        for agent in playground.agents.values():
            for tool in self._feishu_tools:
                agent.tools.register(tool)

    def _inject_send_file_tool(self, playground, chat_id: str) -> None:
        """Inject the file/image sending tool into all agents in the playground."""
        if not self._feishu_client:
            return
        from .tools.send_file import SendFileTool

        tool = SendFileTool(client=self._feishu_client, chat_id=chat_id)
        for agent in playground.agents.values():
            agent.tools.register(tool)

    def _run_session_subtask(
        self,
        chat_id: str,
        agent_name: str,
        task_text: str,
        on_step: Optional[Callable] = None,
        sender_open_id: Optional[str] = None,
        session_key: str | None = None,
        reporter=None,
    ) -> tuple[str, Any]:
        """Run a session-level subtask: an independent agent session supporting multi-turn conversation.

        Args:
            session_key: 自定义会话 key。不传时 fallback 到 {chat_id}:{agent_name}。
            reporter: 可选 FeishuStepReporter，存储到 session 供停止按钮使用。

        Returns:
            (answer_text, trajectory) tuple. trajectory may be None on exception.
        """
        from evomaster.utils.types import TaskInstance

        session_key = session_key or f"{chat_id}:{agent_name}"
        session = self._session_manager.get_or_create(
            session_key,
            playground_factory=lambda: self._create_playground(agent_name, sender_open_id, chat_id=chat_id),
        )

        # Session-level subtask also processes serially
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            
            session.playground.register_thread()

            # 跟踪 reporter 和 running agent（供停止按钮使用）
            if reporter:
                session.current_reporter = reporter

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
                    self._inject_send_file_tool(session.playground, chat_id)
                    self._inject_doc_write_tool(session.playground, sender_open_id)
                    self._inject_ask_user_tool(session.agent)

                    task = TaskInstance(
                        task_id=f"session_subtask_{agent_name}",
                        task_type="session_subtask",
                        description=task_text,
                    )
                    session.current_running_agent = session.agent
                    session.agent._cancel_event.clear()
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    logger.info(
                        "Continuing session subtask key=%s (message #%d)",
                        session_key, session.message_count,
                    )
                    session.current_running_agent = session.agent
                    session.agent._cancel_event.clear()
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
            finally:
                session.current_reporter = None
                session.current_running_agent = None

    def _inject_doc_write_tool(self, playground, sender_open_id: str | None) -> None:
        """Inject the Feishu document writing tool into all agents in the playground."""
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
        """Inject the ask_user tool (only used in interactive contexts)."""
        from evomaster.interface.tools.ask_user import AskUserTool
        agent.tools.register(AskUserTool())

    @staticmethod
    def _inject_memory_tools(agent, memory_manager, user_id: str) -> None:
        """Inject memory tools into the agent (memory_search / memory_save / memory_forget)."""
        if memory_manager is None:
            return
        from playground.magiclaw.tools.memory_tools import (
            MemorySearchTool, MemorySaveTool, MemoryForgetTool,
        )
        for tool_cls in (MemorySearchTool, MemorySaveTool, MemoryForgetTool):
            tool = tool_cls(memory_manager=memory_manager, user_id=user_id)
            agent.tools.register(tool)

    def _inject_background_tools(self, agent, chat_id: str) -> None:
        """注入后台任务相关工具：动态 agent 列表 + check_background_tasks。"""
        # 1. 更新 delegate_to_agent 的可用 agent 列表
        delegate_tool = agent.tools.get_tool("delegate_to_agent")
        if delegate_tool and hasattr(delegate_tool, "set_available_agents"):
            # 合并内置 + 生成的 agent
            all_agents = dict(self._available_agents)
            gen_dir = self._project_root / "configs" / "_generated"
            if gen_dir.exists():
                for child in sorted(gen_dir.iterdir()):
                    if child.is_dir() and (child / "config.yaml").exists():
                        if child.name not in all_agents:
                            desc = self._extract_config_description(child / "config.yaml")
                            all_agents[child.name] = desc or "自定义智能体"
            delegate_tool.set_available_agents(all_agents)

        # 2. 注入/更新 check_background_tasks 工具
        check_tool = agent.tools.get_tool("check_background_tasks")
        if check_tool and hasattr(check_tool, "set_context"):
            check_tool.set_context(self._bg_task_registry, chat_id)
        else:
            from playground.magiclaw.tools.check_background_tasks import CheckBackgroundTasksTool
            check_tool = CheckBackgroundTasksTool(
                task_registry=self._bg_task_registry, chat_id=chat_id
            )
            agent.tools.register(check_tool)

    def _inject_schedule_tools(self, agent, chat_id: str, sender_open_id: str) -> None:
        """注入定时任务工具：schedule_task + manage_schedules。"""
        if not self._scheduler_store or not self._scheduler_service:
            return

        sched_cfg = self._scheduler_config

        # 1. schedule_task
        sched_tool = agent.tools.get_tool("schedule_task")
        if sched_tool and hasattr(sched_tool, "set_context"):
            sched_tool.set_context(
                self._scheduler_store, self._scheduler_service,
                chat_id, sender_open_id,
                max_jobs_per_chat=getattr(sched_cfg, "max_jobs_per_chat", 20),
                max_jobs_total=getattr(sched_cfg, "max_jobs_total", 200),
                default_timezone=getattr(sched_cfg, "default_timezone", "Asia/Shanghai"),
            )
        else:
            from playground.magiclaw.tools.schedule_task import ScheduleTaskTool
            sched_tool = ScheduleTaskTool()
            sched_tool.set_context(
                self._scheduler_store, self._scheduler_service,
                chat_id, sender_open_id,
                max_jobs_per_chat=getattr(sched_cfg, "max_jobs_per_chat", 20),
                max_jobs_total=getattr(sched_cfg, "max_jobs_total", 200),
                default_timezone=getattr(sched_cfg, "default_timezone", "Asia/Shanghai"),
            )
            agent.tools.register(sched_tool)

        # 2. manage_schedules
        manage_tool = agent.tools.get_tool("manage_schedules")
        if manage_tool and hasattr(manage_tool, "set_context"):
            manage_tool.set_context(self._scheduler_store, chat_id)
        else:
            from playground.magiclaw.tools.schedule_task import ManageSchedulesTool
            manage_tool = ManageSchedulesTool()
            manage_tool.set_context(self._scheduler_store, chat_id)
            agent.tools.register(manage_tool)

        # 3. 确保 schedule 工具加入 enabled_tool_names（使 LLM 能看到函数 schema）
        if hasattr(agent, 'enabled_tool_names') and agent.enabled_tool_names is not None:
            for name in ('schedule_task', 'manage_schedules'):
                if name not in agent.enabled_tool_names:
                    agent.enabled_tool_names.append(name)

    def _dispatch_delegation_from_step(
        self, step_record, session, chat_id, message_id, sender_open_id
    ) -> int:
        """扫描单个 step 的 tool_responses，立即 dispatch 发现的委派。

        在 on_step 回调中调用——agent.run() 每一步之后、下一步之前触发。
        Returns: 本次 dispatch 的委派数量。
        """
        count = 0
        for resp in step_record.tool_responses:
            if getattr(resp, "name", "") != "delegate_to_agent":
                continue
            info = (getattr(resp, "meta", None) or {}).get("info", {})
            if not info.get("delegated"):
                continue

            agent_name = info["agent_name"]
            task = info["task"]
            key = (agent_name, task)

            # 去重：已 dispatch 过就跳过
            if key in session.dispatched_delegation_keys:
                continue
            session.dispatched_delegation_keys.add(key)

            logger.info(
                "Immediate delegation dispatch: agent=%s, task=%s",
                agent_name, task[:100],
            )

            if agent_name in _SYNCHRONOUS_DELEGATION_AGENTS:
                self._dispatch_sync_delegation(
                    chat_id, agent_name, task, message_id, sender_open_id
                )
            else:
                bg_task = self._bg_task_registry.create(chat_id, agent_name, task)
                self._dispatch_background_subtask(
                    chat_id, bg_task, message_id, sender_open_id
                )
            count += 1
        return count

    def _make_on_step_with_delegation(
        self, base_on_step, session, chat_id, message_id, sender_open_id
    ):
        """包装 on_step 回调，注入委派拦截逻辑。

        返回的 wrapped_on_step 会先执行原始 on_step（如 reporter.on_step），
        然后扫描当前 step 的 tool_responses 发现委派标记就立即 dispatch。
        """
        def wrapped_on_step(step_record, step_number, max_steps):
            # 1. 先执行原始 on_step（如 reporter.on_step，更新飞书卡片）
            if base_on_step:
                try:
                    base_on_step(step_record, step_number, max_steps)
                except Exception:
                    pass
            # 2. 拦截委派
            try:
                self._dispatch_delegation_from_step(
                    step_record, session, chat_id, message_id, sender_open_id
                )
            except Exception:
                logger.exception("Delegation interceptor failed in on_step")
        return wrapped_on_step

    def _dispatch_sync_delegation(
        self,
        chat_id: str,
        delegated_agent: str,
        delegated_task: str,
        message_id: str,
        sender_open_id: str | None = None,
    ) -> None:
        """后台线程执行同步委派（如 agent_builder planner），避免阻塞 magiclaw lock。"""
        task_id = uuid.uuid4().hex[:8]
        unique_session_key = f"{chat_id}:{delegated_agent}:{task_id}"

        def _run():
            subtask_reporter = None
            subtask_on_step = None
            if self._step_reporter_factory:
                try:
                    subtask_reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    # 添加停止按钮
                    subtask_reporter.set_stop_actions(
                        self._build_stop_actions(
                            chat_id, target="session_subtask",
                            session_key=unique_session_key,
                        )
                    )
                    subtask_reporter.send_initial_card(
                        f"[{delegated_agent}] {delegated_task[:200]}"
                    )
                    subtask_on_step = subtask_reporter.on_step
                except Exception:
                    logger.exception("Failed to create subtask reporter")

            try:
                answer, sub_trajectory = self._run_session_subtask(
                    chat_id, delegated_agent, delegated_task,
                    subtask_on_step, sender_open_id,
                    session_key=unique_session_key,
                    reporter=subtask_reporter,
                )

                # 检查是否被取消
                if sub_trajectory and sub_trajectory.status == "cancelled":
                    if subtask_reporter:
                        try:
                            subtask_reporter.finalize("cancelled", "任务已被用户停止。")
                        except Exception:
                            pass
                    return

                # waiting_for_input: 显示提问卡片
                if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                    if subtask_reporter:
                        try:
                            sub_session = self._session_manager.get(unique_session_key)
                            self._finalize_subtask_with_question(
                                subtask_reporter, sub_trajectory, unique_session_key,
                                delegated_agent, sub_session,
                            )
                        except Exception:
                            logger.exception("Failed to finalize question card")
                    return

                # 注入结果到 magiclaw（短暂持锁）
                magiclaw_session = self._session_manager.get(chat_id)
                if magiclaw_session and magiclaw_session.initialized and magiclaw_session.agent:
                    acquired = magiclaw_session.lock.acquire(timeout=30)
                    if acquired:
                        try:
                            magiclaw_session.agent.add_user_message(
                                f"[子任务结果 - {delegated_agent}]\n"
                                f"用户请求: {delegated_task}\n"
                                f"结果: {answer}"
                            )
                        finally:
                            magiclaw_session.lock.release()

                # Finalize reporter：确认/取消按钮
                if subtask_reporter:
                    try:
                        if delegated_agent in _CONFIRM_SUBTASK_AGENTS:
                            _answer_for_button = answer[:2000] if answer else ""
                            actions = [
                                {
                                    "text": "✅ 确认生成",
                                    "type": "primary",
                                    "value": {
                                        "action": "confirm_agent_build",
                                        "session_key": unique_session_key,
                                        "agent_name": delegated_agent,
                                        "original_answer": _answer_for_button,
                                    },
                                },
                                {
                                    "text": "❌ 取消",
                                    "type": "danger",
                                    "value": {
                                        "action": "cancel_agent_build",
                                        "session_key": unique_session_key,
                                        "agent_name": delegated_agent,
                                        "original_answer": _answer_for_button,
                                    },
                                },
                            ]
                            subtask_reporter.finalize(
                                "completed", answer, actions=actions
                            )
                            sub_session = self._session_manager.get(unique_session_key)
                            if sub_session:
                                sub_session.last_card_message_id = subtask_reporter.card_message_id
                        else:
                            subtask_reporter.finalize("completed", answer)
                    except Exception:
                        logger.exception("Failed to finalize subtask reporter")

            except Exception:
                logger.exception(
                    "Sync delegation failed: key=%s", unique_session_key
                )
                if subtask_reporter:
                    try:
                        subtask_reporter.finalize("failed")
                    except Exception:
                        pass

        thread = threading.Thread(
            target=_run,
            name=f"sync-deleg-{unique_session_key}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Sync delegation dispatched: key=%s, thread=%s",
            unique_session_key, thread.name,
        )

    def _dispatch_background_subtask(
        self,
        chat_id: str,
        bg_task,
        message_id: str,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """启动后台 daemon 线程执行子智能体任务。"""
        from .background_task import BackgroundTaskStatus

        def _run():
            logger.info(
                "Background subtask started: task_id=%s, agent=%s",
                bg_task.task_id, bg_task.agent_name,
            )
            # 创建独立的 step reporter（用户看到独立进度卡片）
            bg_reporter = None
            bg_on_step = None
            if self._step_reporter_factory:
                try:
                    bg_reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    # 添加停止按钮
                    bg_reporter.set_stop_actions(
                        self._build_stop_actions(
                            chat_id, target="subtask", task_id=bg_task.task_id,
                        )
                    )
                    bg_reporter.send_initial_card(
                        f"🔄 [{bg_task.agent_name}] {bg_task.task_description[:200]}"
                    )
                    bg_reporter.start_heartbeat(interval=15)
                    bg_task.reporter = bg_reporter
                except Exception:
                    logger.exception("Failed to create background reporter")

            # 包装 on_step 回调，同时更新 registry 进度
            # agent.run() 调用签名: on_step(step_record, step_number, max_steps)
            def _on_step(step_record, step_number, max_steps):
                # 提取工具名
                tool_name = None
                if hasattr(step_record, "tool_calls") and step_record.tool_calls:
                    tool_name = step_record.tool_calls[0].name if hasattr(step_record.tool_calls[0], "name") else None
                elif hasattr(step_record, "tool_responses") and step_record.tool_responses:
                    tool_name = getattr(step_record.tool_responses[0], "name", None)
                self._bg_task_registry.update_step(bg_task, step_number, tool_name)
                # 转发给 reporter（同样需要 3 个参数）
                if bg_reporter:
                    try:
                        bg_reporter.on_step(step_record, step_number, max_steps)
                    except Exception:
                        logger.warning(
                            "bg_reporter.on_step failed for task_id=%s",
                            bg_task.task_id, exc_info=True,
                        )

            def _store_agent(agent):
                bg_task.agent = agent

            try:
                answer = self._run_subtask(
                    bg_task.agent_name, bg_task.task_description,
                    _on_step, chat_id=chat_id, sender_open_id=sender_open_id,
                    on_agent_ready=_store_agent,
                )

                # 检查是否被用户取消
                if bg_task.agent and bg_task.agent.is_cancelled:
                    self._bg_task_registry.mark_cancelled(bg_task)
                    if bg_reporter:
                        try:
                            bg_reporter.stop_heartbeat()
                            bg_reporter.finalize("cancelled", "任务已被用户停止。")
                        except Exception:
                            logger.exception("Failed to finalize cancelled background reporter")
                    return  # 取消的任务跳过自动审阅

                self._bg_task_registry.mark_completed(bg_task, answer)

                if bg_reporter:
                    try:
                        bg_reporter.stop_heartbeat()
                        bg_reporter.finalize("completed", answer)
                    except Exception:
                        logger.exception("Failed to finalize background reporter")

            except Exception as e:
                error_msg = str(e)
                self._bg_task_registry.mark_failed(bg_task, error_msg)
                logger.exception(
                    "Background subtask failed: task_id=%s", bg_task.task_id
                )
                if bg_reporter:
                    try:
                        bg_reporter.stop_heartbeat()
                        bg_reporter.finalize("failed")
                    except Exception:
                        pass

            # 任务完成后触发审阅（取消的任务已 return，不会到达此处）
            self._on_background_task_completed(bg_task, chat_id, message_id, sender_open_id)

        thread = threading.Thread(
            target=_run,
            name=f"bg-subtask-{bg_task.task_id}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Background subtask dispatched: task_id=%s, thread=%s",
            bg_task.task_id, thread.name,
        )

    def _on_background_task_completed(
        self,
        bg_task,
        chat_id: str,
        message_id: str,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """后台任务完成后的回调：尝试立即审阅或排队。"""
        from .background_task import BackgroundTaskStatus

        session = self._session_manager.get(chat_id)
        if not session:
            logger.warning(
                "Session not found for review: chat_id=%s, task_id=%s",
                chat_id, bg_task.task_id,
            )
            return

        review_info = {
            "task_id": bg_task.task_id,
            "agent_name": bg_task.agent_name,
            "task_description": bg_task.task_description,
            "result": bg_task.result or bg_task.error or "",
            "status": bg_task.status.value,
        }

        # 尝试非阻塞获取锁
        acquired = session.lock.acquire(blocking=False)
        if acquired:
            try:
                self._run_review(session, review_info, chat_id, message_id, sender_open_id)
                self._bg_task_registry.mark_reviewed(bg_task)
            except Exception:
                logger.exception("Failed to run review for task_id=%s", bg_task.task_id)
            finally:
                session.lock.release()
        else:
            # magiclaw 正在处理用户消息，排队等待
            logger.info(
                "Session busy, queueing review: task_id=%s", bg_task.task_id
            )
            session.pending_reviews.append(review_info)

    def _run_review(
        self,
        session,
        review_info: dict,
        chat_id: str,
        message_id: str,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """持有 session lock 时调用：magiclaw 审阅后台任务结果并汇报。"""
        if not session.initialized or not session.agent:
            logger.warning("Cannot review: session not initialized for chat_id=%s", chat_id)
            return

        agent_name = review_info["agent_name"]
        task_desc = review_info["task_description"]
        result = review_info["result"]
        status = review_info["status"]

        status_label = "成功" if status == "completed" else "失败"
        review_prompt = (
            f"[后台任务审阅]\n"
            f"Agent: {agent_name}\n"
            f"任务: {task_desc}\n"
            f"状态: {status_label}\n"
            f"结果:\n{result}\n\n"
            f"请审阅上述后台任务的结果，并向用户汇报关键信息。如果任务失败，解释原因并建议下一步。"
        )

        # 创建审阅卡片的 reporter
        review_reporter = None
        review_on_step = None
        if self._step_reporter_factory:
            try:
                review_reporter = self._step_reporter_factory(
                    chat_id, message_id, sender_open_id
                )
                review_reporter.send_initial_card(
                    f"📋 审阅 [{agent_name}] 任务结果"
                )
                review_on_step = review_reporter.on_step
            except Exception:
                logger.exception("Failed to create review reporter")

        # 包装 review_on_step：注入委派拦截（pipeline 模式：search → review → report_writer）
        review_on_step = self._make_on_step_with_delegation(
            review_on_step, session, chat_id, message_id, sender_open_id
        )

        try:
            # 注册当前线程到 playground（可能是后台线程）
            session.playground.register_thread()

            trajectory = session.agent.continue_run(
                review_prompt, on_step=review_on_step
            )
            answer = _extract_final_answer(
                {"trajectory": trajectory, "status": trajectory.status}
            )

            if review_reporter:
                try:
                    review_reporter.finalize("completed", answer)
                except Exception:
                    logger.exception("Failed to finalize review reporter")

            logger.info("Review completed for task by %s", agent_name)

            # Pipeline 模式：review 中可能产生新委派（如 search → report_writer）
            # on_step 拦截器已 dispatch 大部分，此处为 safety net
            review_safety = self._check_all_delegations(session)
            for delegation in review_safety:
                d_agent = delegation["agent_name"]
                d_task = delegation["task"]
                key = (d_agent, d_task)
                session.dispatched_delegation_keys.add(key)
                logger.info("Safety-net delegation in review: agent=%s", d_agent)
                if d_agent in _SYNCHRONOUS_DELEGATION_AGENTS:
                    self._dispatch_sync_delegation(
                        chat_id, d_agent, d_task, message_id, sender_open_id
                    )
                else:
                    bg_task = self._bg_task_registry.create(chat_id, d_agent, d_task)
                    self._dispatch_background_subtask(
                        chat_id, bg_task, message_id, sender_open_id
                    )

        except Exception:
            logger.exception("Review failed for task by %s", agent_name)
            if review_reporter:
                try:
                    review_reporter.finalize("failed")
                except Exception:
                    pass

    def _process_pending_reviews(
        self,
        session,
        chat_id: str,
        message_id: str,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """处理排队的后台任务审阅（在持有 session lock 时调用）。"""
        if not session.pending_reviews:
            return

        reviews = list(session.pending_reviews)
        session.pending_reviews.clear()

        for review_info in reviews:
            try:
                self._run_review(session, review_info, chat_id, message_id, sender_open_id)
                # 标记为已审阅
                task_id = review_info["task_id"]
                tasks = self._bg_task_registry.get_tasks_for_chat(chat_id)
                for t in tasks:
                    if t.task_id == task_id:
                        self._bg_task_registry.mark_reviewed(t)
                        break
            except Exception:
                logger.exception(
                    "Failed to process pending review: task_id=%s",
                    review_info.get("task_id"),
                )

    @staticmethod
    def _memory_auto_recall(agent, memory_manager, memory_config, user_id: str, query: str) -> None:
        """Automatically recall related memories and inject them at the end of the system prompt."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_recall", True):
            return
        limit = memory_config.get("recall_limit", 5)
        memory_context = memory_manager.recall_for_context(
            user_id=user_id, query=query, limit=limit,
        )
        if not memory_context:
            return
        # Append memories to the end of the system prompt
        dialog = agent.current_dialog
        if dialog and dialog.messages and dialog.messages[0].role.value == "system":
            dialog.messages[0].content = (
                dialog.messages[0].content + "\n\n" + memory_context
            )

    @staticmethod
    def _memory_auto_capture(memory_manager, memory_config, user_id: str, message: str) -> None:
        """Automatically extract memories from user messages."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_capture", True):
            return
        try:
            memory_manager.extract_from_message(user_id, message)
        except Exception:
            logger.debug("Memory auto-capture failed", exc_info=True)

    @staticmethod
    def _format_questions_for_card(questions: list[dict]) -> str:
        """Format questions as card markdown (with header grouping support)."""
        parts = []
        for q in questions:
            header = q.get("header", "")
            title = f"**{header}: {q.get('question', '')}**" if header else f"**{q.get('question', '')}**"
            parts.append(title)
            for opt in q.get("options", []):
                desc = f" — {opt['description']}" if opt.get("description") else ""
                parts.append(f"  - {opt['label']}{desc}")
            parts.append("")  # Blank line to separate questions
        parts.append("> You can also reply with text to provide more details")
        return "\n".join(parts)

    @staticmethod
    def _build_question_actions(
        questions: list[dict], session_key: str, agent_name: str,
        question_text: str = "",
    ) -> list[dict]:
        """Build buttons for the first question's options only (maximum 4)."""
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
                    "original_question": question_text[:1500],
                },
            })
        return actions

    def _finalize_subtask_with_question(
        self, reporter, trajectory, sub_session_key: str, agent_name: str, sub_session
    ) -> None:
        """When a subtask returns waiting_for_input, present questions one by one via cards.

        Only render the first question (with full buttons); remaining questions are stored
        in session.pending_questions and presented one by one in _continue_session_subtask
        after the user answers.
        """
        questions = (getattr(trajectory, "result", None) or {}).get("questions", [])
        if not questions:
            return

        # Only present the first question
        first = [questions[0]]
        question_text = self._format_questions_for_card(first)
        option_actions = self._build_question_actions(
            first, sub_session_key, agent_name, question_text=question_text
        )
        reporter.finalize_as_question(question_text, actions=option_actions)
        if sub_session:
            sub_session.last_card_message_id = reporter.card_message_id
            # Store remaining questions, clear collected answers
            sub_session.pending_questions = questions[1:]
            sub_session.collected_answers = []

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
        """Handle card button callbacks, triggering continue_run for session-level subtasks.

        Args:
            chat_id: Chat ID (used for sending results).
            session_key: Session key (format: {chat_id}:{agent_name}).
            agent_name: Agent name.
            task_text: Text to send to the agent (e.g. "Confirm").
            sender_open_id: Operator's open_id.
            card_message_id: Card message ID that triggered the button.
            original_answer: Phase 1's original answer content (preserved when updating card).
            action_type: Button type ("confirm" = Phase 2 generation, "answer_question" = answer question to continue Phase 1).
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
        """Continue an existing session-level subtask (triggered by card button).

        Args:
            action_type: "confirm" = Phase 2 builder run, "answer_question" = continue planner.
        """
        # 立即移除 Phase 1 卡片上的按钮（此处在线程池中运行，callback 已完成，
        # 避免在 WebSocket callback 中 REST patch 与 ACK 竞态）
        if card_message_id and action_type in ("confirm", "cancel"):
            from .messaging.sender import patch_card_message as _patch_card
            _status_line = {
                "confirm": "> ⏳ 方案已确认，正在生成 Agent 文件...",
                "cancel": "> ❌ Agent 生成已取消。",
            }
            _titles = {
                "confirm": "⏳ Agent 生成中...",
                "cancel": "❌ 已取消",
            }
            _templates = {"confirm": "wathet", "cancel": "red"}
            parts = []
            if original_answer:
                parts.append(original_answer)
            parts.append("---")
            parts.append(_status_line[action_type])
            _patch_card(
                self._feishu_client, card_message_id,
                title=_titles[action_type],
                content="\n\n".join(parts),
                header_template=_templates[action_type],
            )

        session = self._session_manager.get(session_key)
        if session is None or not session.initialized:
            logger.warning(
                "No active session for card action: key=%s", session_key
            )
            return f"会话已过期或不存在，请重新发起 /agent {agent_name} 命令。"

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            session.playground.register_thread()

            # Create progress reporter
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, card_message_id, sender_open_id
                    )
                    # 添加停止按钮
                    reporter.set_stop_actions(
                        self._build_stop_actions(
                            chat_id, target="session_subtask",
                            session_key=session_key,
                        )
                    )
                    # agent_builder confirm: delay sending card, wait for TODO parsing then send at once
                    # answer_question: send immediately, so on_step can update in real-time
                    if action_type == "answer_question" or agent_name not in _CONFIRM_SUBTASK_AGENTS:
                        reporter.send_initial_card(f"[{agent_name}] {task_text}")
                    on_step = self._make_on_step_with_delegation(
                        reporter.on_step, session, chat_id,
                        card_message_id or f"card_{session_key}", sender_open_id,
                    )
                except Exception:
                    logger.exception("Failed to create step reporter for card action")

            # 跟踪 reporter（供停止按钮使用）
            session.current_reporter = reporter

            try:
                # === Sequential questioning: check if there are follow-up questions to present ===
                if action_type == "answer_question" and session.pending_questions:
                    session.collected_answers.append(task_text)
                    next_q = session.pending_questions.pop(0)
                    first = [next_q]
                    question_text = self._format_questions_for_card(first)
                    option_actions = self._build_question_actions(
                        first, session_key, agent_name,
                        question_text=question_text,
                    )
                    if reporter:
                        reporter.finalize_as_question(
                            question_text, actions=option_actions
                        )
                        session.last_card_message_id = reporter.card_message_id
                    return None  # Don't resume agent, wait for the next answer

                # === Sequential questioning: all questions answered, merge answers ===
                if action_type == "answer_question" and session.collected_answers:
                    session.collected_answers.append(task_text)
                    task_text = "\n".join(session.collected_answers)
                    session.collected_answers = []

                logger.info(
                    "Continuing session subtask via card action: key=%s (message #%d)",
                    session_key, session.message_count,
                )

                # agent_builder dual-agent mode: Phase 2 uses builder_agent (fresh run)
                # Only triggered on confirm, answer_question goes through planner continue_run
                if (
                    action_type == "confirm"
                    and agent_name == "agent_builder"
                    and hasattr(session.playground, "agents")
                    and hasattr(session.playground.agents, "builder_agent")
                ):
                    from evomaster.utils.types import TaskInstance

                    # Parse TODO checklist from planner output and set it on the reporter
                    todo_items = self._parse_plan_todos(original_answer)
                    if reporter:
                        if todo_items:
                            reporter.set_todo_items(todo_items)
                        reporter.send_initial_card(
                            f"[{agent_name}] 正在生成 Agent 文件..."
                        )
                        on_step = reporter.on_step

                    builder_agent = session.playground.agents.builder_agent
                    # Inject Feishu tools into builder agent (already injected during setup, but ensure availability)
                    if self._feishu_tools:
                        for tool in self._feishu_tools:
                            builder_agent.tools.register(tool)
                    # Build handoff task: pass the planner's plan summary to the builder
                    plan_task = TaskInstance(
                        task_id=f"builder_{agent_name}",
                        task_type="builder",
                        description=(
                            "请根据以下设计方案生成 Agent 文件。\n\n"
                            f"## 方案摘要\n{original_answer}\n\n"
                            "请使用 feishu_doc_read 工具读取飞书文档获取完整方案，然后生成所有文件。"
                        ),
                    )
                    session.current_running_agent = builder_agent
                    builder_agent._cancel_event.clear()
                    trajectory = builder_agent.run(plan_task, on_step=on_step)

                    # After builder completes, check if all TODOs are done; if not, trigger another round
                    if reporter and reporter.has_incomplete_todos() and not builder_agent.is_cancelled:
                        incomplete = reporter.get_incomplete_todo_labels()
                        reminder = (
                            "你还有以下 TODO 项未完成，请逐一完成并上报 PROGRESS 后再调用 finish：\n"
                            + "\n".join(f"- [ ] {label}" for label in incomplete)
                        )
                        logger.info(
                            "Builder has %d incomplete TODOs, triggering continue_run",
                            len(incomplete),
                        )
                        trajectory = builder_agent.continue_run(reminder, on_step=on_step)
                else:
                    session.current_running_agent = session.agent
                    session.agent._cancel_event.clear()
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                # === 取消检测 ===
                if trajectory and trajectory.status == "cancelled":
                    if reporter:
                        try:
                            reporter.finalize("cancelled", "任务已被用户停止。")
                        except Exception:
                            pass
                    return None

                # === 委派检测 (safety net) ===
                safety_delegations = self._check_all_delegations(session)
                for delegation in safety_delegations:
                    delegated_agent = delegation["agent_name"]
                    delegated_task = delegation["task"]
                    key = (delegated_agent, delegated_task)
                    session.dispatched_delegation_keys.add(key)
                    logger.info(
                        "Safety-net delegation in card action: agent=%s, task=%s",
                        delegated_agent, delegated_task[:100],
                    )
                    if delegated_agent in _SYNCHRONOUS_DELEGATION_AGENTS:
                        self._dispatch_sync_delegation(
                            chat_id, delegated_agent, delegated_task,
                            card_message_id or f"card_{session_key}", sender_open_id,
                        )
                    else:
                        bg_task = self._bg_task_registry.create(
                            chat_id, delegated_agent, delegated_task
                        )
                        self._dispatch_background_subtask(
                            chat_id, bg_task,
                            card_message_id or f"card_{session_key}", sender_open_id,
                        )

                had_delegations = bool(session.dispatched_delegation_keys)
                if had_delegations:
                    # finalize card and return early — delegations are dispatched
                    chat_answer = _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )
                    if reporter:
                        try:
                            reporter.finalize("completed", chat_answer)
                        except Exception:
                            logger.exception("Failed to finalize reporter after delegation")
                    session.dispatched_delegation_keys.clear()
                    return None

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

                # === answer_question path: processing after planner continue_run ===
                if action_type == "answer_question":
                    # Check if the planner is asking again
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

                    # planner finished: show confirm/cancel buttons
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

                    # Inject result into magiclaw context
                    chat_session = self._session_manager.get(chat_id)
                    if chat_session and chat_session.initialized and chat_session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"结果: {answer}"
                        )
                        chat_session.agent.add_user_message(summary)

                    return None

                # === confirm path: processing after Phase 2 builder completion ===
                # Inject result into magiclaw context
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

                # Update Phase 1 card: change from "generating" to "completed", keeping original plan content
                phase1_content = original_answer + "\n\n---\n> ✅ Agent 已成功创建。详情请查看下方回复。" if original_answer else "Agent 已成功创建。\n\n详情请查看下方回复。"
                self._patch_phase1_card(
                    card_message_id, "✅ Agent 创建完成",
                    phase1_content, "green",
                )

                # Phase 2 complete, clean up subtask session, subsequent messages go back to magiclaw
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
                    # Update Phase 1 card: show failure status, keeping original plan content
                    phase1_content = original_answer + f"\n\n---\n> ❌ Agent 创建过程中出错：{str(e)[:500]}" if original_answer else f"Agent 创建过程中出错。\n\n{str(e)[:500]}"
                    self._patch_phase1_card(
                        card_message_id, "❌ Agent 创建失败",
                        phase1_content, "red",
                    )

                return f"会话子任务执行出错: {e}"
            finally:
                session.current_reporter = None
                session.current_running_agent = None

    def _patch_phase1_card(
        self,
        card_message_id: str | None,
        title: str,
        content: str,
        header_template: str,
    ) -> None:
        """Update the Phase 1 card status (called after Phase 2 completion/failure)."""
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
        """Check if magiclaw triggered delegation via delegate_to_agent.

        Scan the last few steps of the trajectory's ToolMessages for a delegated=True marker.
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

    @staticmethod
    def _check_all_delegations(session) -> list[dict[str, str]]:
        """检查 trajectory 中所有未 dispatch 的 delegate_to_agent 调用。

        作为 safety net：on_step 拦截器应已 dispatch 大部分委派，
        此方法捕获任何遗漏（如 on_step 异常）。
        全量扫描所有 steps，跳过已通过 on_step 即时 dispatch 的委派。
        """
        if not session.initialized or not session.agent:
            return []
        traj = session.agent.trajectory
        if not traj or not traj.steps:
            return []

        dispatched = session.dispatched_delegation_keys
        delegations = []
        seen = set()

        for step in reversed(traj.steps):
            for resp in step.tool_responses:
                if getattr(resp, "name", "") == "delegate_to_agent":
                    info = (getattr(resp, "meta", None) or {}).get("info", {})
                    if info.get("delegated"):
                        key = (info["agent_name"], info["task"])
                        if key not in seen and key not in dispatched:
                            seen.add(key)
                            delegations.append({
                                "agent_name": info["agent_name"],
                                "task": info["task"],
                            })
        return delegations

    def _find_active_subtask(self, chat_id: str) -> str | None:
        """Find if there is an active subtask session for this chat.

        If one exists, subsequent messages are routed directly to the subtask session
        (supporting multi-turn plan modification, etc.).
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
        """Parse the TODO list from planner output.

        Expected format::

            ---PLAN_TODO---
            - [ ] Create directory structure
            - [ ] Create system_prompt.txt
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
        """Task completion callback."""
        self._active_tasks.pop(message_id, None)

        try:
            result_text = future.result(timeout=0)
        except TimeoutError:
            result_text = f"任务超时（超过 {self._task_timeout} 秒）"
        except Exception as e:
            result_text = f"任务执行异常: {e}"

        # None means the reporter card already contains the answer, no extra message needed
        if result_text is None:
            return

        if self._on_result:
            try:
                self._on_result(chat_id, message_id, result_text)
            except Exception:
                logger.exception("Error in on_result callback")

    def _get_allowed_agent_names(self) -> set[str]:
        """返回所有允许调用的 agent 名称集合"""
        allowed = set(self._available_agents.keys())
        # _generated/ 下的自定义 agent 始终允许
        gen_dir = self._project_root / "configs" / "_generated"
        if gen_dir.exists():
            for child in gen_dir.iterdir():
                if child.is_dir() and (child / "config.yaml").exists():
                    allowed.add(child.name)
        return allowed

    def _collect_available_agents(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """收集可用子智能体列表。

        Returns:
            (builtin_agents, generated_agents): 每项为 (name, description)
        """
        # 1. 内置智能体：从配置白名单获取
        builtin = [(name, desc) for name, desc in self._available_agents.items()]

        # 2. 自定义智能体：扫描 configs/_generated/ 目录
        generated = []
        gen_dir = self._project_root / "configs" / "_generated"
        if gen_dir.exists():
            for child in sorted(gen_dir.iterdir()):
                if child.is_dir() and (child / "config.yaml").exists():
                    desc = self._extract_config_description(child / "config.yaml")
                    generated.append((child.name, desc))

        return builtin, generated

    @staticmethod
    def _extract_config_description(config_path: Path) -> str:
        """从 config.yaml 注释头提取描述"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("#"):
                        break
                    text = line.lstrip("#").strip()
                    if not text or "配置文件" in text:
                        continue
                    return text
        except Exception:
            pass
        return ""

    def _send_list_card(self, chat_id: str, message_id: str) -> None:
        """发送可用子智能体列表卡片"""
        builtin, generated = self._collect_available_agents()

        parts = []

        if builtin:
            parts.append("**内置智能体**")
            for name, desc in builtin:
                line = f"• `{name}`"
                if desc:
                    line += f" — {desc}"
                parts.append(line)

        if generated:
            if parts:
                parts.append("")
            parts.append("**自定义智能体**（通过 Agent Builder 创建）")
            for name, desc in generated:
                line = f"• `{name}`"
                if desc:
                    line += f" — {desc}"
                parts.append(line)

        if not builtin and not generated:
            parts.append("暂无可用的子智能体。\n\n你可以直接告诉我想创建什么智能体，我会帮你设计并构建。")

        parts.append("\n---")
        parts.append("使用方式: `/agent <名称> <任务描述>`")

        content = "\n".join(parts)

        if self._feishu_client:
            from .messaging.sender import send_card_message
            send_card_message(
                self._feishu_client, chat_id,
                title="📋 可用智能体",
                content=content,
                reply_to_message_id=message_id,
                header_template="indigo",
            )
        elif self._on_result:
            self._on_result(chat_id, message_id, content)

    def _send_welcome_card(self, chat_id: str, message_id: str) -> None:
        """Send a welcome card introducing bot features and usage."""
        if not self._feishu_client:
            # fallback: plain text
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "新会话已开始。直接发送消息即可对话，或使用 /agent <名称> <任务> 调用专属智能体。\n命令: /help /list /new",
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
            "`/list` — 查看可用的子智能体列表\n"
            "`/new` — 清除上下文，开始新会话\n"
            "`/stop` — 停止当前正在执行的任务\n"
            "`/schedule` — 查看/管理定时任务"
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
        """Send a usage help card."""
        if not self._feishu_client:
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "使用帮助：直接发消息对话；/agent <名称> <任务> 调用智能体；/list 查看可用智能体；/new 新会话。",
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
            "`/list` — 查看可用的子智能体列表\n"
            "`/new` — 清除上下文，开始新会话\n"
            "`/stop` — 停止当前正在执行的任务\n"
            "`/schedule` — 查看/管理定时任务"
        )

        send_card_message(
            self._feishu_client,
            chat_id,
            title="📖 使用帮助",
            content=content,
            reply_to_message_id=message_id,
            header_template="blue",
        )

    # ------------------------------------------------------------------
    # Stop / Cancel
    # ------------------------------------------------------------------

    def _build_stop_actions(
        self, chat_id: str, target: str = "orchestrator",
        task_id: str = "", session_key: str = "",
    ) -> list[dict]:
        """构建停止按钮配置。"""
        return [{
            "text": "🛑 停止",
            "type": "danger",
            "value": {
                "action": "stop_agent",
                "chat_id": chat_id,
                "target": target,
                "task_id": task_id,
                "session_key": session_key,
            },
        }]

    def _stop_all_for_chat(self, chat_id: str) -> tuple[bool, list[str]]:
        """停止指定 chat 的 orchestrator + 所有后台子任务 + 子会话。

        线程安全，不需要 session.lock（仅设置 threading.Event 和 bool）。

        Returns:
            (orchestrator_stopped, cancelled_task_ids)
        """
        orchestrator_stopped = False
        cancelled_task_ids: list[str] = []

        # 1. 停止 orchestrator
        session = self._session_manager.get(chat_id)
        if session and session.current_running_agent:
            session.current_running_agent.cancel()
            if session.current_reporter:
                session.current_reporter.mark_stopping()
                session.current_reporter.finalize("cancelled", "任务已被用户停止。")
            orchestrator_stopped = True

        # 2. 停止后台子任务
        for task in self._bg_task_registry.get_active_tasks(chat_id):
            if task.agent:
                task.agent.cancel()
            if task.reporter:
                task.reporter.mark_stopping()
                task.reporter.finalize("cancelled", "任务已被用户停止。")
            self._bg_task_registry.mark_cancelled(task)
            cancelled_task_ids.append(task.task_id)

        # 3. 停止同步委派子会话（key pattern: {chat_id}:*）
        for sub in self._session_manager.get_sessions_by_prefix(chat_id):
            if sub.current_running_agent:
                sub.current_running_agent.cancel()
            if sub.current_reporter:
                sub.current_reporter.mark_stopping()
                sub.current_reporter.finalize("cancelled", "任务已被用户停止。")

        return orchestrator_stopped, cancelled_task_ids

    def _handle_stop_command(self, chat_id: str, message_id: str) -> None:
        """处理 /stop 命令：停止当前 chat 的所有运行中任务。"""
        orchestrator_stopped, cancelled_ids = self._stop_all_for_chat(chat_id)

        if orchestrator_stopped or cancelled_ids:
            parts = ["已发送停止信号。"]
            if orchestrator_stopped:
                parts.append("- 主智能体将在当前步骤完成后停止")
            for tid in cancelled_ids:
                parts.append(f"- 后台任务 {tid} 已取消")
            msg = "\n".join(parts)
        else:
            msg = "当前没有正在运行的任务。"

        if self._feishu_client:
            from .messaging.sender import send_text_message
            send_text_message(self._feishu_client, chat_id, msg, reply_to_message_id=message_id)
        elif self._on_result:
            self._on_result(chat_id, message_id, msg)

    def _handle_schedule_command(
        self, chat_id: str, message_id: str, text: str,
        sender_open_id: str | None = None,
    ) -> None:
        """处理 /schedule 命令：list / cancel <id>"""
        if not self._scheduler_store:
            msg = "定时任务功能未启用。"
            if self._feishu_client:
                from .messaging.sender import send_text_message
                send_text_message(self._feishu_client, chat_id, msg, reply_to_message_id=message_id)
            elif self._on_result:
                self._on_result(chat_id, message_id, msg)
            return

        parts = text.strip().split(maxsplit=2)
        sub = parts[1] if len(parts) > 1 else "list"

        if sub == "list":
            jobs = self._scheduler_store.get_active_for_chat(chat_id)
            if not jobs:
                msg = "当前没有活跃的定时任务。"
            else:
                from datetime import datetime
                lines = [f"共 {len(jobs)} 个活跃定时任务:\n"]
                for j in jobs:
                    try:
                        from zoneinfo import ZoneInfo
                        tzinfo = ZoneInfo(j.timezone)
                    except Exception:
                        tzinfo = None
                    next_str = datetime.fromtimestamp(j.next_run_at, tz=tzinfo).strftime("%m-%d %H:%M")
                    lines.append(
                        f"• `{j.job_id}` {j.schedule_type.value}({j.schedule_expr}) "
                        f"→ {j.task_description[:60]}\n"
                        f"  下次: {next_str} | 执行: {j.run_count}次"
                    )
                msg = "\n".join(lines)

        elif sub == "cancel":
            job_id = parts[2].strip() if len(parts) > 2 else ""
            if not job_id:
                msg = "用法: /schedule cancel <任务ID>"
            else:
                from evomaster.scheduler.models import JobStatus
                job = self._scheduler_store.get(job_id)
                if not job:
                    msg = f"未找到任务: {job_id}"
                elif job.chat_id != chat_id:
                    msg = "无权取消其他会话的任务。"
                elif job.status != JobStatus.ACTIVE:
                    msg = f"任务 {job_id} 状态为 {job.status.value}，无需取消。"
                else:
                    self._scheduler_store.mark_cancelled(job_id)
                    msg = f"已取消定时任务 `{job_id}`: {job.task_description[:80]}"
        else:
            msg = (
                "定时任务命令:\n"
                "`/schedule` 或 `/schedule list` — 查看活跃任务\n"
                "`/schedule cancel <ID>` — 取消指定任务\n\n"
                "创建定时任务请直接对话，例如：「每小时检查一下xxx」"
            )

        if self._feishu_client:
            from .messaging.sender import send_text_message
            send_text_message(self._feishu_client, chat_id, msg, reply_to_message_id=message_id)
        elif self._on_result:
            self._on_result(chat_id, message_id, msg)

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the dispatcher and all sessions."""
        logger.info("Shutting down task dispatcher...")
        if self._scheduler_service is not None:
            self._scheduler_service.stop()
        if self._scheduler_store is not None:
            self._scheduler_store.close()
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        if self._container_pool is not None:
            self._container_pool.shutdown()
        logger.info("Task dispatcher shut down")
