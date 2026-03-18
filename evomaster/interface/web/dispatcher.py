"""Web chat task dispatcher

Adapts the Feishu dispatcher for WebSocket-based communication via Flask-SocketIO.
Reuses shared session management, background task infrastructure, and playground
orchestration from the feishu module.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from evomaster.interface.feishu.dispatcher import (
    _ensure_playgrounds_imported,
    _extract_final_answer,
    _SESSION_SUBTASK_AGENTS,
    _CONFIRM_SUBTASK_AGENTS,
    _SYNCHRONOUS_DELEGATION_AGENTS,
)
from evomaster.interface.feishu.session_manager import (
    ChatSessionManager,
    PlaygroundSession,
)
from evomaster.interface.feishu.background_task import (
    BackgroundTaskRegistry,
    BackgroundTask,
    BackgroundTaskStatus,
)

logger = logging.getLogger(__name__)

# Agents that must run locally (no container pool)
_LOCAL_ONLY_AGENTS: set[str] = {"agent_builder"}


class WebTaskDispatcher:
    """Task dispatcher for the web chat interface.

    Routes WebSocket messages through a thread pool, manages multi-turn
    sessions via ``ChatSessionManager``, and reports progress through
    Flask-SocketIO events instead of Feishu cards.
    """

    def __init__(
        self,
        project_root: Path,
        socketio,
        default_agent: str = "magiclaw",
        default_config_path: str | None = None,
        max_workers: int = 4,
        task_timeout: int = 600,
        max_sessions: int = 100,
        available_agents: dict[str, str] | None = None,
        container_pool=None,
        server_start_time: str | None = None,
    ):
        self._project_root = project_root
        self._socketio = socketio
        self._default_agent = default_agent
        self._default_config_path = default_config_path
        self._task_timeout = task_timeout
        self._available_agents = available_agents or {}
        self._container_pool = container_pool
        self._server_start_time = server_start_time or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="web-task",
        )
        self._active_tasks: dict[str, Any] = {}
        self._session_manager = ChatSessionManager(max_sessions=max_sessions)
        self._bg_task_registry = BackgroundTaskRegistry()

        # Ensure _generated directories exist for agent_builder output
        (project_root / "configs" / "_generated").mkdir(parents=True, exist_ok=True)
        (project_root / "playground" / "_generated").mkdir(parents=True, exist_ok=True)

        _ensure_playgrounds_imported(project_root)

    # ------------------------------------------------------------------
    # Public dispatch API
    # ------------------------------------------------------------------

    def dispatch(
        self,
        session_id: str,
        message_id: str,
        task_text: str,
        agent_name: str | None = None,
        room: str | None = None,
    ) -> None:
        """Submit a task to the thread pool.

        Handles special commands (``/new``, ``/help``, ``/list``) synchronously
        and delegates normal messages to ``_run_task_with_session``.
        """
        stripped = task_text.strip()
        target_room = room or session_id

        # /new — reset session
        if stripped == "/new":
            self._session_manager.remove(session_id)
            for sub_agent in _SESSION_SUBTASK_AGENTS:
                self._session_manager.remove(f"{session_id}:{sub_agent}")
            self._bg_task_registry.cleanup_old(session_id, max_age_seconds=0)
            self._send_welcome(target_room)
            return

        # /help
        if stripped == "/help":
            self._send_help(target_room)
            return

        # /list
        if stripped == "/list":
            self._send_list(target_room)
            return

        # Whitelist check for /agent specified agents
        if agent_name and agent_name != self._default_agent:
            allowed = self._get_allowed_agent_names()
            if agent_name not in allowed:
                self._emit_error(
                    target_room,
                    f"Agent `{agent_name}` is not available. "
                    "Send /list to see available agents.",
                    message_id=message_id,
                )
                return

        agent = agent_name or self._default_agent
        future = self._executor.submit(
            self._run_task_with_session,
            session_id,
            message_id,
            task_text,
            agent,
            target_room,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(
            lambda f: self._on_task_done(f, session_id, message_id, target_room)
        )

        # Timeout guard
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
                self._emit_error(
                    target_room,
                    f"Task timed out after {self._task_timeout}s.",
                    message_id=message_id,
                )
            except Exception:
                pass

        threading.Thread(
            target=_timeout_guard,
            daemon=True,
            name=f"timeout-{message_id[:8]}",
        ).start()

    def dispatch_card_action(
        self,
        session_id: str,
        action: str,
        session_key: str,
        agent_name: str,
        task_text: str | None = None,
        room: str | None = None,
        **kwargs,
    ) -> None:
        """Route an interactive action (confirm, cancel, answer) from the web UI.

        ``action`` values:
        - ``confirm_agent_build`` — Phase 2 builder run
        - ``cancel_agent_build`` — discard the plan and remove sub-session
        - ``answer_question`` — user answered an ``ask_user`` question
        """
        target_room = room or session_id
        message_id = kwargs.get("message_id", f"action_{session_key}")

        if action == "cancel_agent_build":
            self._session_manager.remove(session_key)
            self._socketio.emit(
                "agent_response",
                {
                    "message_id": message_id,
                    "status": "cancelled",
                    "final_answer": "Agent build cancelled.",
                },
                room=target_room,
            )
            return

        if action in ("confirm_agent_build", "answer_question"):
            action_type = "confirm" if action == "confirm_agent_build" else "answer_question"
            original_answer = kwargs.get("original_answer", "")
            future = self._executor.submit(
                self._continue_session_subtask,
                session_id,
                session_key,
                agent_name,
                task_text or "",
                target_room,
                message_id,
                original_answer,
                action_type,
            )
            self._active_tasks[message_id] = future
            future.add_done_callback(
                lambda f: self._on_task_done(
                    f, session_id, message_id, target_room
                )
            )
            return

        logger.warning("Unknown card action: %s", action)

    # ------------------------------------------------------------------
    # Playground creation
    # ------------------------------------------------------------------

    def _create_playground(self, agent_name: str, session_id: str | None = None):
        """Create a playground instance without calling ``setup()``."""
        from evomaster.core import get_playground_class

        if agent_name == self._default_agent and self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"
            if not config_path.exists():
                config_path = (
                    self._project_root
                    / "configs"
                    / "_generated"
                    / agent_name
                    / "config.yaml"
                )

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        self._try_import_generated_playground(agent_name)

        playground = get_playground_class(agent_name, config_path=config_path)

        # Inject container pool for agents that need it
        if (
            self._container_pool
            and agent_name not in _LOCAL_ONLY_AGENTS
            and hasattr(playground, "set_container_pool")
        ):
            playground.set_container_pool(self._container_pool)

        # Build a hierarchical run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        web_base = self._project_root / "runs" / f"web_{self._server_start_time}"
        user_dir = session_id or "unknown"
        run_dir = web_base / user_dir / f"{agent_name}_{timestamp}"
        task_id = f"web_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        return playground

    def _try_import_generated_playground(self, agent_name: str) -> None:
        """Dynamically import a ``_generated`` playground if not yet registered."""
        from evomaster.core.registry import _PLAYGROUND_REGISTRY

        if agent_name in _PLAYGROUND_REGISTRY:
            return

        module_name = f"playground._generated.{agent_name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Dynamically imported generated playground: %s", module_name)
        except ImportError:
            pass
        except Exception:
            logger.warning(
                "Error importing generated playground: %s",
                module_name,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Core task execution
    # ------------------------------------------------------------------

    def _run_task_with_session(
        self,
        session_id: str,
        message_id: str,
        task_text: str,
        agent_name: str,
        room: str,
    ) -> str | None:
        """Execute a task inside a managed session, reusing context for multi-turn."""
        from evomaster.utils.types import TaskInstance
        from evomaster.interface.web.step_reporter import WebStepReporter

        session = self._session_manager.get_or_create(
            session_id,
            playground_factory=lambda: self._create_playground(
                self._default_agent, session_id
            ),
        )

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            session.playground.register_thread()

            # Create step reporter for real-time progress
            reporter = WebStepReporter(self._socketio, room, message_id)
            reporter.send_initial_card(task_text)
            on_step = reporter.on_step

            try:
                # -- Sub-agent mode: /agent specified a non-default agent --
                if agent_name != self._default_agent:
                    if agent_name in _SESSION_SUBTASK_AGENTS:
                        answer, sub_trajectory = self._run_session_subtask(
                            session_id, agent_name, task_text, on_step, room
                        )
                    else:
                        answer = self._run_subtask(
                            agent_name, task_text, on_step, session_id=session_id
                        )
                        sub_trajectory = None

                    # ask_user waiting
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        sub_session_key = f"{session_id}:{agent_name}"
                        sub_session = self._session_manager.get(sub_session_key)
                        self._finalize_subtask_with_question(
                            reporter, sub_trajectory, sub_session_key,
                            agent_name, sub_session,
                        )
                        return None

                    # Inject result into default agent context
                    if session.initialized and session.agent:
                        summary = (
                            f"[Sub-task result — {agent_name}]\n"
                            f"User request: {task_text}\n"
                            f"Result: {answer}"
                        )
                        session.agent.add_user_message(summary)

                    # Confirm subtask agents get confirm/cancel buttons
                    if agent_name in _CONFIRM_SUBTASK_AGENTS:
                        sk = f"{session_id}:{agent_name}"
                        answer_preview = (answer or "")[:2000]
                        actions = self._build_confirm_cancel_actions(
                            sk, agent_name, answer_preview
                        )
                        reporter.finalize("completed", answer, actions=actions)
                    else:
                        reporter.finalize("completed", answer)
                    return None

                # -- Active sub-task routing --
                active_subtask = self._find_active_subtask(session_id)
                if active_subtask:
                    sub_session_key = f"{session_id}:{active_subtask}"
                    answer, sub_trajectory = self._run_session_subtask(
                        session_id, active_subtask, task_text, on_step, room
                    )

                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        sub_session = self._session_manager.get(sub_session_key)
                        self._finalize_subtask_with_question(
                            reporter, sub_trajectory, sub_session_key,
                            active_subtask, sub_session,
                        )
                        return None

                    if session.initialized and session.agent:
                        summary = (
                            f"[Sub-task result — {active_subtask}]\n"
                            f"User request: {task_text}\n"
                            f"Result: {answer}"
                        )
                        session.agent.add_user_message(summary)

                    if active_subtask in _CONFIRM_SUBTASK_AGENTS:
                        sk = f"{session_id}:{active_subtask}"
                        answer_preview = (answer or "")[:2000]
                        actions = self._build_confirm_cancel_actions(
                            sk, active_subtask, answer_preview
                        )
                        reporter.finalize("completed", answer, actions=actions)
                        sub_session = self._session_manager.get(sk)
                        if sub_session:
                            sub_session.last_card_message_id = reporter.card_message_id
                    else:
                        reporter.finalize("completed", answer)
                    return None

                # -- Normal default-agent flow --
                # Wrap on_step with delegation interception (immediate dispatch)
                on_step = self._make_on_step_with_delegation(
                    on_step, session, session_id, message_id, room
                )
                memory_manager = getattr(session.playground, "_memory_manager", None)
                memory_config = getattr(session.playground, "_memory_config", {})
                user_id = session_id or "unknown"

                if not session.initialized:
                    logger.info(
                        "First message in session session_id=%s, running setup",
                        session_id,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    memory_manager = getattr(
                        session.playground, "_memory_manager", None
                    )
                    memory_config = getattr(
                        session.playground, "_memory_config", {}
                    )

                    self._inject_ask_user_tool(session.agent)
                    self._inject_memory_tools(session.agent, memory_manager, user_id)
                    self._inject_background_tools(session.agent, session_id)

                    # Memory extraction hook on context compaction
                    if memory_manager and memory_config.get("auto_capture", True):
                        _mm = memory_manager
                        _uid = user_id

                        def _on_compaction(old_messages, mm=_mm, uid=_uid):
                            from evomaster.utils.types import UserMessage as UM

                            for msg in old_messages:
                                if isinstance(msg, UM):
                                    text = (
                                        msg.content
                                        if isinstance(msg.content, str)
                                        else ""
                                    )
                                    if text:
                                        mm.extract_from_message(uid, text)

                        session.agent.context_manager.on_before_compaction = (
                            _on_compaction
                        )

                    task = TaskInstance(
                        task_id=f"web_{message_id}",
                        task_type="chat",
                        description=task_text,
                    )

                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config,
                        user_id, task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True

                    self._memory_auto_capture(
                        memory_manager, memory_config, user_id, task_text
                    )
                else:
                    logger.info(
                        "Continuing session session_id=%s (message #%d)",
                        session_id,
                        session.message_count,
                    )

                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config,
                        user_id, task_text,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )
                    self._memory_auto_capture(
                        memory_manager, memory_config, user_id, task_text
                    )

                # -- ask_user detection --
                if trajectory and trajectory.status == "waiting_for_input":
                    questions = (trajectory.result or {}).get("questions", [])
                    if questions:
                        first = [questions[0]]
                        question_text = self._format_questions(first)
                        option_actions = self._build_question_actions(
                            first, session_id, "magiclaw",
                            question_text=question_text,
                        )
                        options = self._build_question_options(first)
                        reporter.finalize_as_question(
                            question_text,
                            actions=option_actions,
                            options=options,
                            session_key=session_id,
                            agent_name="magiclaw",
                        )
                        session.pending_questions = questions[1:]
                        session.collected_answers = []
                        return None

                # -- Delegation detection --
                # Safety net: catch delegations the on_step interceptor may have missed
                safety_delegations = self._check_all_delegations(session)
                for delegation in safety_delegations:
                    delegated_agent = delegation["agent_name"]
                    delegated_task = delegation["task"]
                    key = (delegated_agent, delegated_task)
                    session.dispatched_delegation_keys.add(key)
                    logger.info(
                        "Safety-net delegation detected: agent=%s, task=%s",
                        delegated_agent,
                        delegated_task[:100],
                    )
                    if delegated_agent in _SYNCHRONOUS_DELEGATION_AGENTS:
                        self._dispatch_sync_delegation(
                            session, session_id, message_id,
                            delegated_agent, delegated_task, room,
                        )
                    else:
                        bg_task = self._bg_task_registry.create(
                            session_id, delegated_agent, delegated_task
                        )
                        self._dispatch_background_subtask(
                            session_id, bg_task, message_id, room
                        )

                # Check if any delegations happened this turn (on_step + safety net)
                had_delegations = bool(session.dispatched_delegation_keys)

                if had_delegations:
                    chat_answer = _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )
                    reporter.finalize("completed", chat_answer)

                    self._process_pending_reviews(
                        session, session_id, message_id, room
                    )
                    # Reset tracking set for next user message
                    session.dispatched_delegation_keys.clear()
                    return None

                # Normal completion
                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                logger.info(
                    "Task completed in session session_id=%s, status=%s",
                    session_id,
                    trajectory.status,
                )
                reporter.finalize("completed", answer)

                self._process_pending_reviews(
                    session, session_id, message_id, room
                )
                return None

            except Exception as e:
                logger.exception(
                    "Task failed in session session_id=%s", session_id
                )
                reporter.finalize("failed")
                return f"Task execution error: {e}"

    # ------------------------------------------------------------------
    # Sub-task helpers
    # ------------------------------------------------------------------

    def _run_subtask(
        self,
        agent_name: str,
        task_text: str,
        on_step: Callable | None = None,
        session_id: str | None = None,
    ) -> str:
        """Run a one-shot sub-task with its own playground (no session reuse)."""
        from evomaster.utils.types import TaskInstance

        logger.info("Running subtask with agent=%s", agent_name)
        playground = self._create_playground(agent_name, session_id)
        playground.register_thread()
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
            return f"Sub-task execution error: {e}"
        finally:
            try:
                playground.cleanup()
            except Exception:
                logger.exception("Subtask cleanup failed")

    def _run_session_subtask(
        self,
        session_id: str,
        agent_name: str,
        task_text: str,
        on_step: Callable | None = None,
        room: str | None = None,
    ) -> tuple[str, Any]:
        """Run a session-level sub-task that supports multi-turn dialogue.

        Uses ``{session_id}:{agent_name}`` as the session key.

        Returns:
            ``(answer_text, trajectory)`` tuple.
        """
        from evomaster.utils.types import TaskInstance

        session_key = f"{session_id}:{agent_name}"
        session = self._session_manager.get_or_create(
            session_key,
            playground_factory=lambda: self._create_playground(
                agent_name, session_id
            ),
        )

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            session.playground.register_thread()

            try:
                if not session.initialized:
                    logger.info(
                        "First message in session subtask key=%s, agent=%s",
                        session_key,
                        agent_name,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

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
                        session_key,
                        session.message_count,
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
                    "Session subtask failed: key=%s, agent=%s",
                    session_key,
                    agent_name,
                )
                return f"Session sub-task error: {e}", None

    def _continue_session_subtask(
        self,
        session_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        room: str,
        message_id: str,
        original_answer: str = "",
        action_type: str = "confirm",
    ) -> str | None:
        """Continue an existing session sub-task triggered by a UI action."""
        from evomaster.utils.types import TaskInstance
        from evomaster.interface.web.step_reporter import WebStepReporter

        session = self._session_manager.get(session_key)
        if session is None or not session.initialized:
            logger.warning(
                "No active session for card action: key=%s", session_key
            )
            return (
                f"Session expired or not found. Please re-invoke "
                f"/agent {agent_name}."
            )

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            session.playground.register_thread()

            reporter = WebStepReporter(self._socketio, room, message_id)
            if (
                action_type == "answer_question"
                or agent_name not in _CONFIRM_SUBTASK_AGENTS
            ):
                reporter.send_initial_card(f"[{agent_name}] {task_text}")
            on_step = reporter.on_step

            try:
                # Sequential question flow: more pending questions
                if (
                    action_type == "answer_question"
                    and session.pending_questions
                ):
                    session.collected_answers.append(task_text)
                    next_q = session.pending_questions.pop(0)
                    first = [next_q]
                    question_text = self._format_questions(first)
                    option_actions = self._build_question_actions(
                        first, session_key, agent_name,
                        question_text=question_text,
                    )
                    options = self._build_question_options(first)
                    reporter.finalize_as_question(
                        question_text,
                        actions=option_actions,
                        options=options,
                        session_key=session_key,
                        agent_name=agent_name,
                    )
                    session.last_card_message_id = reporter.card_message_id
                    return None

                # All questions answered — merge answers
                if (
                    action_type == "answer_question"
                    and session.collected_answers
                ):
                    session.collected_answers.append(task_text)
                    task_text = "\n".join(session.collected_answers)
                    session.collected_answers = []

                logger.info(
                    "Continuing session subtask via action: key=%s (message #%d)",
                    session_key,
                    session.message_count,
                )

                # agent_builder Phase 2: builder agent run
                if (
                    action_type == "confirm"
                    and agent_name == "agent_builder"
                    and hasattr(session.playground, "agents")
                    and hasattr(session.playground.agents, "builder_agent")
                ):
                    # Use a new message_id so Phase 2 gets its own card
                    import uuid as _uuid
                    phase2_message_id = str(_uuid.uuid4())
                    reporter = WebStepReporter(
                        self._socketio, room, phase2_message_id
                    )
                    todo_items = self._parse_plan_todos(original_answer)
                    if todo_items:
                        reporter.set_todo_items(todo_items)
                    reporter.send_initial_card(
                        f"[{agent_name}] Generating agent files..."
                    )
                    on_step = reporter.on_step

                    builder_agent = session.playground.agents.builder_agent
                    plan_task = TaskInstance(
                        task_id=f"builder_{agent_name}",
                        task_type="builder",
                        description=(
                            "Please generate the Agent files according to "
                            "the following design plan.\n\n"
                            f"## Plan Summary\n{original_answer}\n\n"
                            "Generate all required files now."
                        ),
                    )
                    trajectory = builder_agent.run(plan_task, on_step=on_step)

                    # Check for incomplete TODOs
                    if reporter.has_incomplete_todos():
                        incomplete = reporter.get_incomplete_todo_labels()
                        reminder = (
                            "The following TODO items are still incomplete. "
                            "Please finish each one:\n"
                            + "\n".join(
                                f"- [ ] {label}" for label in incomplete
                            )
                        )
                        logger.info(
                            "Builder has %d incomplete TODOs, triggering continue_run",
                            len(incomplete),
                        )
                        trajectory = builder_agent.continue_run(
                            reminder, on_step=on_step
                        )
                else:
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

                # answer_question path
                if action_type == "answer_question":
                    if trajectory and trajectory.status == "waiting_for_input":
                        self._finalize_subtask_with_question(
                            reporter, trajectory, session_key,
                            agent_name, session,
                        )
                        return None

                    if agent_name in _CONFIRM_SUBTASK_AGENTS:
                        answer_preview = (answer or "")[:2000]
                        actions = self._build_confirm_cancel_actions(
                            session_key, agent_name, answer_preview
                        )
                        reporter.finalize("completed", answer, actions=actions)
                        session.last_card_message_id = reporter.card_message_id
                    else:
                        reporter.finalize("completed", answer)

                    # Inject result into default agent context
                    chat_session = self._session_manager.get(session_id)
                    if (
                        chat_session
                        and chat_session.initialized
                        and chat_session.agent
                    ):
                        summary = (
                            f"[Sub-task result — {agent_name}]\n"
                            f"Result: {answer}"
                        )
                        chat_session.agent.add_user_message(summary)
                    return None

                # confirm path (Phase 2 complete)
                chat_session = self._session_manager.get(session_id)
                if (
                    chat_session
                    and chat_session.initialized
                    and chat_session.agent
                ):
                    summary = (
                        f"[Sub-task result — {agent_name} Phase 2]\n"
                        f"Result: {answer}"
                    )
                    chat_session.agent.add_user_message(summary)

                reporter.finalize("completed", answer)

                # Phase 2 done — remove sub-session
                self._session_manager.remove(session_key)
                return None

            except Exception as e:
                logger.exception(
                    "Card action subtask failed: key=%s", session_key
                )
                reporter.finalize("failed")
                return f"Session sub-task error: {e}"

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def _dispatch_sync_delegation(
        self,
        session: PlaygroundSession,
        session_id: str,
        message_id: str,
        delegated_agent: str,
        delegated_task: str,
        room: str,
    ) -> None:
        """Handle a synchronous delegation (e.g. agent_builder planner)."""
        from evomaster.interface.web.step_reporter import WebStepReporter

        # Use a new message_id so the delegation gets its own card
        import uuid as _uuid
        sub_message_id = str(_uuid.uuid4())

        sub_reporter = WebStepReporter(self._socketio, room, sub_message_id)
        sub_reporter.send_initial_card(
            f"[{delegated_agent}] {delegated_task[:200]}"
        )
        sub_on_step = sub_reporter.on_step

        self._socketio.emit(
            "delegation_status",
            {
                "message_id": message_id,
                "agent_name": delegated_agent,
                "task_preview": delegated_task[:200],
                "type": "sync",
            },
            room=room,
        )

        answer, sub_trajectory = self._run_session_subtask(
            session_id, delegated_agent, delegated_task, sub_on_step, room
        )

        # ask_user waiting
        if sub_trajectory and sub_trajectory.status == "waiting_for_input":
            sub_session_key = f"{session_id}:{delegated_agent}"
            sub_session = self._session_manager.get(sub_session_key)
            self._finalize_subtask_with_question(
                sub_reporter, sub_trajectory, sub_session_key,
                delegated_agent, sub_session,
            )
            return

        if session.initialized and session.agent:
            summary = (
                f"[Sub-task result — {delegated_agent}]\n"
                f"User request: {delegated_task}\n"
                f"Result: {answer}"
            )
            session.agent.add_user_message(summary)

        if delegated_agent in _CONFIRM_SUBTASK_AGENTS:
            sk = f"{session_id}:{delegated_agent}"
            answer_preview = (answer or "")[:2000]
            actions = self._build_confirm_cancel_actions(
                sk, delegated_agent, answer_preview
            )
            sub_reporter.finalize("completed", answer, actions=actions)
            sub_session = self._session_manager.get(sk)
            if sub_session:
                sub_session.last_card_message_id = sub_reporter.card_message_id
        else:
            sub_reporter.finalize("completed", answer)

    def _dispatch_delegation_from_step(
        self, step_record, session, session_id, message_id, room,
    ) -> int:
        """Scan a single step's tool_responses and immediately dispatch delegations.

        Called from the on_step callback — fires after each agent step, before
        the next step begins.  Returns the number of delegations dispatched.
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

            if key in session.dispatched_delegation_keys:
                continue
            session.dispatched_delegation_keys.add(key)

            logger.info(
                "Immediate delegation dispatch: agent=%s, task=%s",
                agent_name, task[:100],
            )

            if agent_name in _SYNCHRONOUS_DELEGATION_AGENTS:
                self._dispatch_sync_delegation(
                    session, session_id, message_id,
                    agent_name, task, room,
                )
            else:
                bg_task = self._bg_task_registry.create(
                    session_id, agent_name, task
                )
                self._dispatch_background_subtask(
                    session_id, bg_task, message_id, room
                )
            count += 1
        return count

    def _make_on_step_with_delegation(
        self, base_on_step, session, session_id, message_id, room,
    ):
        """Wrap an on_step callback to inject delegation interception logic.

        The returned wrapper first invokes the original on_step (e.g. the
        WebStepReporter callback), then scans for delegation markers and
        dispatches them immediately.
        """
        def wrapped_on_step(step_record, step_number, max_steps):
            if base_on_step:
                try:
                    base_on_step(step_record, step_number, max_steps)
                except Exception:
                    pass
            try:
                self._dispatch_delegation_from_step(
                    step_record, session, session_id, message_id, room
                )
            except Exception:
                logger.exception("Delegation interceptor failed in on_step")
        return wrapped_on_step

    def _dispatch_background_subtask(
        self,
        session_id: str,
        bg_task: BackgroundTask,
        message_id: str,
        room: str,
    ) -> None:
        """Launch a background daemon thread for a non-synchronous sub-agent."""
        from evomaster.interface.web.step_reporter import WebStepReporter

        self._socketio.emit(
            "delegation_status",
            {
                "message_id": message_id,
                "agent_name": bg_task.agent_name,
                "task_preview": bg_task.task_description[:200],
                "type": "background",
            },
            room=room,
        )

        def _run():
            logger.info(
                "Background subtask started: task_id=%s, agent=%s",
                bg_task.task_id,
                bg_task.agent_name,
            )
            bg_reporter = WebStepReporter(
                self._socketio, room,
                f"bg_{bg_task.task_id}",
            )
            bg_reporter.send_initial_card(
                f"[background] [{bg_task.agent_name}] "
                f"{bg_task.task_description[:200]}"
            )

            def _on_step(step_record, step_number, max_steps):
                tool_name = None
                if (
                    hasattr(step_record, "tool_calls")
                    and step_record.tool_calls
                ):
                    tool_name = getattr(
                        step_record.tool_calls[0], "name", None
                    )
                elif (
                    hasattr(step_record, "tool_responses")
                    and step_record.tool_responses
                ):
                    tool_name = getattr(
                        step_record.tool_responses[0], "name", None
                    )
                self._bg_task_registry.update_step(
                    bg_task, step_number, tool_name
                )
                bg_reporter.on_step(step_record, step_number, max_steps)
                self._socketio.emit(
                    "background_task_update",
                    {
                        "task_id": bg_task.task_id,
                        "agent_name": bg_task.agent_name,
                        "status": "running",
                        "step_count": step_number,
                    },
                    room=room,
                )

            try:
                answer = self._run_subtask(
                    bg_task.agent_name,
                    bg_task.task_description,
                    _on_step,
                    session_id=session_id,
                )
                self._bg_task_registry.mark_completed(bg_task, answer)
                bg_reporter.finalize("completed", answer)
                self._socketio.emit(
                    "background_task_update",
                    {
                        "task_id": bg_task.task_id,
                        "agent_name": bg_task.agent_name,
                        "status": "completed",
                        "step_count": bg_task.step_count,
                        "result": answer[:500] if answer else "",
                    },
                    room=room,
                )
            except Exception as e:
                error_msg = str(e)
                self._bg_task_registry.mark_failed(bg_task, error_msg)
                logger.exception(
                    "Background subtask failed: task_id=%s", bg_task.task_id
                )
                bg_reporter.finalize("failed")
                self._socketio.emit(
                    "background_task_update",
                    {
                        "task_id": bg_task.task_id,
                        "agent_name": bg_task.agent_name,
                        "status": "failed",
                        "step_count": bg_task.step_count,
                    },
                    room=room,
                )

            self._on_background_task_completed(
                bg_task, session_id, message_id, room
            )

        thread = threading.Thread(
            target=_run,
            name=f"bg-subtask-{bg_task.task_id}",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Background subtask dispatched: task_id=%s, thread=%s",
            bg_task.task_id,
            thread.name,
        )

    def _on_background_task_completed(
        self,
        bg_task: BackgroundTask,
        session_id: str,
        message_id: str,
        room: str,
    ) -> None:
        """Callback after a background task finishes: immediate review or queue."""
        session = self._session_manager.get(session_id)
        if not session:
            logger.warning(
                "Session not found for review: session_id=%s, task_id=%s",
                session_id,
                bg_task.task_id,
            )
            return

        review_info = {
            "task_id": bg_task.task_id,
            "agent_name": bg_task.agent_name,
            "task_description": bg_task.task_description,
            "result": bg_task.result or bg_task.error or "",
            "status": bg_task.status.value,
        }

        acquired = session.lock.acquire(blocking=False)
        if acquired:
            try:
                self._run_review(
                    session, review_info, session_id, message_id, room
                )
                self._bg_task_registry.mark_reviewed(bg_task)
            except Exception:
                logger.exception(
                    "Failed to run review for task_id=%s", bg_task.task_id
                )
            finally:
                session.lock.release()
        else:
            logger.info(
                "Session busy, queueing review: task_id=%s", bg_task.task_id
            )
            session.pending_reviews.append(review_info)

    def _run_review(
        self,
        session: PlaygroundSession,
        review_info: dict,
        session_id: str,
        message_id: str,
        room: str,
    ) -> None:
        """Run magiclaw review of a completed background task (holds session lock)."""
        from evomaster.interface.web.step_reporter import WebStepReporter

        if not session.initialized or not session.agent:
            logger.warning(
                "Cannot review: session not initialized for session_id=%s",
                session_id,
            )
            return

        agent_name = review_info["agent_name"]
        task_desc = review_info["task_description"]
        result = review_info["result"]
        status = review_info["status"]

        status_label = "completed" if status == "completed" else "failed"
        review_prompt = (
            f"[Background task review]\n"
            f"Agent: {agent_name}\n"
            f"Task: {task_desc}\n"
            f"Status: {status_label}\n"
            f"Result:\n{result}\n\n"
            f"Please review the above background task result and report "
            f"the key findings to the user."
        )

        review_reporter = WebStepReporter(
            self._socketio, room, f"review_{review_info['task_id']}"
        )
        review_reporter.send_initial_card(
            f"[review] [{agent_name}] task result"
        )
        review_on_step = review_reporter.on_step

        # Wrap review_on_step with delegation interception (pipeline mode)
        review_on_step = self._make_on_step_with_delegation(
            review_on_step, session, session_id, message_id, room
        )

        try:
            session.playground.register_thread()
            trajectory = session.agent.continue_run(
                review_prompt, on_step=review_on_step
            )
            answer = _extract_final_answer(
                {"trajectory": trajectory, "status": trajectory.status}
            )
            review_reporter.finalize("completed", answer)
            logger.info("Review completed for task by %s", agent_name)

            # Pipeline mode: review may trigger new delegations (e.g. report_writer)
            review_safety = self._check_all_delegations(session)
            for delegation in review_safety:
                d_agent = delegation["agent_name"]
                d_task = delegation["task"]
                key = (d_agent, d_task)
                session.dispatched_delegation_keys.add(key)
                logger.info("Safety-net delegation in review: agent=%s", d_agent)
                if d_agent in _SYNCHRONOUS_DELEGATION_AGENTS:
                    self._dispatch_sync_delegation(
                        session, session_id, message_id,
                        d_agent, d_task, room,
                    )
                else:
                    bg_task = self._bg_task_registry.create(
                        session_id, d_agent, d_task
                    )
                    self._dispatch_background_subtask(
                        session_id, bg_task, message_id, room
                    )

        except Exception:
            logger.exception("Review failed for task by %s", agent_name)
            review_reporter.finalize("failed")

    def _process_pending_reviews(
        self,
        session: PlaygroundSession,
        session_id: str,
        message_id: str,
        room: str,
    ) -> None:
        """Process queued background task reviews (called while holding lock)."""
        if not session.pending_reviews:
            return

        reviews = list(session.pending_reviews)
        session.pending_reviews.clear()

        for review_info in reviews:
            try:
                self._run_review(
                    session, review_info, session_id, message_id, room
                )
                task_id = review_info["task_id"]
                tasks = self._bg_task_registry.get_tasks_for_chat(session_id)
                for t in tasks:
                    if t.task_id == task_id:
                        self._bg_task_registry.mark_reviewed(t)
                        break
            except Exception:
                logger.exception(
                    "Failed to process pending review: task_id=%s",
                    review_info.get("task_id"),
                )

    # ------------------------------------------------------------------
    # Tool injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_ask_user_tool(agent) -> None:
        """Inject the ask_user tool for interactive contexts."""
        from evomaster.interface.tools.ask_user import AskUserTool

        agent.tools.register(AskUserTool())

    @staticmethod
    def _inject_memory_tools(agent, memory_manager, user_id: str) -> None:
        """Inject memory tools (search/save/forget)."""
        if memory_manager is None:
            return
        from playground.magiclaw.tools.memory_tools import (
            MemorySearchTool,
            MemorySaveTool,
            MemoryForgetTool,
        )

        for tool_cls in (MemorySearchTool, MemorySaveTool, MemoryForgetTool):
            tool = tool_cls(memory_manager=memory_manager, user_id=user_id)
            agent.tools.register(tool)

    def _inject_background_tools(self, agent, session_id: str) -> None:
        """Inject delegate_to_agent list and check_background_tasks tool."""
        # Update delegate_to_agent available agent list
        delegate_tool = agent.tools.get_tool("delegate_to_agent")
        if delegate_tool and hasattr(delegate_tool, "set_available_agents"):
            all_agents = dict(self._available_agents)
            gen_dir = self._project_root / "configs" / "_generated"
            if gen_dir.exists():
                for child in sorted(gen_dir.iterdir()):
                    if child.is_dir() and (child / "config.yaml").exists():
                        if child.name not in all_agents:
                            desc = self._extract_config_description(
                                child / "config.yaml"
                            )
                            all_agents[child.name] = desc or "Custom agent"
            delegate_tool.set_available_agents(all_agents)

        # Inject/update check_background_tasks tool
        check_tool = agent.tools.get_tool("check_background_tasks")
        if check_tool and hasattr(check_tool, "set_context"):
            check_tool.set_context(self._bg_task_registry, session_id)
        else:
            from playground.magiclaw.tools.check_background_tasks import (
                CheckBackgroundTasksTool,
            )

            check_tool = CheckBackgroundTasksTool(
                task_registry=self._bg_task_registry, chat_id=session_id
            )
            agent.tools.register(check_tool)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_auto_recall(
        agent, memory_manager, memory_config, user_id: str, query: str
    ) -> None:
        """Recall relevant memories and inject into the system prompt."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_recall", True):
            return
        limit = memory_config.get("recall_limit", 5)
        memory_context = memory_manager.recall_for_context(
            user_id=user_id, query=query, limit=limit
        )
        if not memory_context:
            return
        dialog = agent.current_dialog
        if (
            dialog
            and dialog.messages
            and dialog.messages[0].role.value == "system"
        ):
            dialog.messages[0].content = (
                dialog.messages[0].content + "\n\n" + memory_context
            )

    @staticmethod
    def _memory_auto_capture(
        memory_manager, memory_config, user_id: str, message: str
    ) -> None:
        """Extract memories from the user message."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_capture", True):
            return
        try:
            memory_manager.extract_from_message(user_id, message)
        except Exception:
            logger.debug("Memory auto-capture failed", exc_info=True)

    # ------------------------------------------------------------------
    # Question / action helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_questions(questions: list[dict]) -> str:
        """Format questions as markdown for the web UI."""
        parts: list[str] = []
        for q in questions:
            header = q.get("header", "")
            title = (
                f"**{header}: {q.get('question', '')}**"
                if header
                else f"**{q.get('question', '')}**"
            )
            parts.append(title)
            for opt in q.get("options", []):
                desc = f" — {opt['description']}" if opt.get("description") else ""
                parts.append(f"  - {opt['label']}{desc}")
            parts.append("")
        parts.append("> You can also reply with free text to provide more detail")
        return "\n".join(parts)

    @staticmethod
    def _build_question_actions(
        questions: list[dict],
        session_key: str,
        agent_name: str,
        question_text: str = "",
    ) -> list[dict]:
        """Build option button actions for the first question (max 4)."""
        if not questions or not questions[0].get("options"):
            return []
        actions: list[dict] = []
        for opt in questions[0]["options"][:4]:
            actions.append(
                {
                    "text": opt.get("label", ""),
                    "type": "default",
                    "value": {
                        "action": "answer_question",
                        "session_key": session_key,
                        "agent_name": agent_name,
                        "answer_text": opt.get("label", ""),
                        "original_question": question_text[:1500],
                    },
                }
            )
        return actions

    @staticmethod
    def _build_question_options(questions: list[dict]) -> list[dict[str, str]]:
        """Build simple option list for the frontend option buttons."""
        if not questions or not questions[0].get("options"):
            return []
        return [
            {
                "label": opt.get("label", ""),
                "description": opt.get("description", ""),
            }
            for opt in questions[0]["options"][:4]
        ]

    @staticmethod
    def _build_confirm_cancel_actions(
        session_key: str, agent_name: str, original_answer: str
    ) -> list[dict]:
        """Build confirm/cancel button actions for a CONFIRM_SUBTASK agent."""
        return [
            {
                "label": "Confirm",
                "action": "confirm_agent_build",
                "type": "confirm",
                "session_key": session_key,
                "agent_name": agent_name,
                "original_answer": original_answer,
            },
            {
                "label": "Cancel",
                "action": "cancel_agent_build",
                "type": "cancel",
                "session_key": session_key,
                "agent_name": agent_name,
                "original_answer": original_answer,
            },
        ]

    def _finalize_subtask_with_question(
        self,
        reporter,
        trajectory,
        sub_session_key: str,
        agent_name: str,
        sub_session: PlaygroundSession | None,
    ) -> None:
        """Present the first ask_user question; queue remaining questions."""
        questions = (getattr(trajectory, "result", None) or {}).get(
            "questions", []
        )
        if not questions:
            return

        first = [questions[0]]
        question_text = self._format_questions(first)
        option_actions = self._build_question_actions(
            first, sub_session_key, agent_name, question_text=question_text
        )
        options = self._build_question_options(first)
        reporter.finalize_as_question(
            question_text,
            actions=option_actions,
            options=options,
            session_key=sub_session_key,
            agent_name=agent_name,
        )
        if sub_session:
            sub_session.last_card_message_id = reporter.card_message_id
            sub_session.pending_questions = questions[1:]
            sub_session.collected_answers = []

    # ------------------------------------------------------------------
    # Delegation detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_all_delegations(
        session: PlaygroundSession,
    ) -> list[dict[str, str]]:
        """Scan all trajectory steps for undispatched delegate_to_agent calls.

        Acts as a safety net: the on_step interceptor should have dispatched
        most delegations already.  This method catches any that were missed
        (e.g. due to on_step exceptions).  Scans all steps and skips those
        already present in ``session.dispatched_delegation_keys``.
        """
        if not session.initialized or not session.agent:
            return []
        traj = session.agent.trajectory
        if not traj or not traj.steps:
            return []

        dispatched = session.dispatched_delegation_keys
        delegations: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for step in reversed(traj.steps):
            for resp in step.tool_responses:
                if getattr(resp, "name", "") == "delegate_to_agent":
                    info = (getattr(resp, "meta", None) or {}).get("info", {})
                    if info.get("delegated"):
                        key = (info["agent_name"], info["task"])
                        if key not in seen and key not in dispatched:
                            seen.add(key)
                            delegations.append(
                                {
                                    "agent_name": info["agent_name"],
                                    "task": info["task"],
                                }
                            )
        return delegations

    def _find_active_subtask(self, session_id: str) -> str | None:
        """Check if any session-level sub-task is active for this session."""
        for agent_name in _SESSION_SUBTASK_AGENTS:
            session_key = f"{session_id}:{agent_name}"
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
        """Parse TODO items from planner output.

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

    # ------------------------------------------------------------------
    # Agent listing / config helpers
    # ------------------------------------------------------------------

    def _get_allowed_agent_names(self) -> set[str]:
        """Return all agent names that are allowed to be invoked."""
        allowed = set(self._available_agents.keys())
        gen_dir = self._project_root / "configs" / "_generated"
        if gen_dir.exists():
            for child in gen_dir.iterdir():
                if child.is_dir() and (child / "config.yaml").exists():
                    allowed.add(child.name)
        return allowed

    def _collect_available_agents(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Collect built-in and generated agent lists.

        Returns:
            ``(builtin_agents, generated_agents)`` each as ``(name, description)``.
        """
        builtin = [
            (name, desc) for name, desc in self._available_agents.items()
        ]
        generated: list[tuple[str, str]] = []
        gen_dir = self._project_root / "configs" / "_generated"
        if gen_dir.exists():
            for child in sorted(gen_dir.iterdir()):
                if child.is_dir() and (child / "config.yaml").exists():
                    desc = self._extract_config_description(
                        child / "config.yaml"
                    )
                    generated.append((child.name, desc))
        return builtin, generated

    @staticmethod
    def _extract_config_description(config_path: Path) -> str:
        """Extract a description from the YAML comment header."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("#"):
                        break
                    text = line.lstrip("#").strip()
                    if not text:
                        continue
                    return text
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # SocketIO emit helpers
    # ------------------------------------------------------------------

    def _send_welcome(self, room: str) -> None:
        """Emit a welcome message when a session is reset."""
        self._socketio.emit(
            "welcome",
            {
                "message": (
                    "New session started. Send any message to begin a "
                    "conversation, or use /agent <name> <task> to invoke "
                    "a specific agent.\n\n"
                    "Commands: /help  /list  /new"
                ),
            },
            room=room,
        )

    def _send_help(self, room: str) -> None:
        """Emit usage help."""
        self._socketio.emit(
            "welcome",
            {
                "message": (
                    "**Usage Help**\n\n"
                    "**Direct conversation**\n"
                    "Send any message to start a multi-turn conversation.\n\n"
                    "**Create an agent**\n"
                    "Describe what you need, e.g. 'Create an agent that "
                    "summarises documents', and the Agent Builder will "
                    "handle design and construction.\n\n"
                    "**Invoke a specific agent**\n"
                    "`/agent <name> <task description>`\n\n"
                    "**Commands**\n"
                    "`/help` — Show this help\n"
                    "`/list` — List available agents\n"
                    "`/new`  — Reset session context"
                ),
            },
            room=room,
        )

    def _send_list(self, room: str) -> None:
        """Emit the list of available agents."""
        builtin, generated = self._collect_available_agents()
        self._socketio.emit(
            "agent_list",
            {
                "builtin": [
                    {"name": n, "description": d} for n, d in builtin
                ],
                "generated": [
                    {"name": n, "description": d} for n, d in generated
                ],
            },
            room=room,
        )

    def _emit_error(
        self, room: str, error: str, message_id: str | None = None
    ) -> None:
        """Emit an error event."""
        payload: dict[str, Any] = {"error": error}
        if message_id:
            payload["message_id"] = message_id
        self._socketio.emit("error", payload, room=room)

    # ------------------------------------------------------------------
    # Task completion callback
    # ------------------------------------------------------------------

    def _on_task_done(
        self,
        future,
        session_id: str,
        message_id: str,
        room: str,
    ) -> None:
        """Handle future completion — emit result or error via SocketIO."""
        self._active_tasks.pop(message_id, None)

        try:
            result_text = future.result(timeout=0)
        except TimeoutError:
            result_text = f"Task timed out (>{self._task_timeout}s)."
        except Exception as e:
            result_text = f"Task execution error: {e}"

        # None means the reporter already sent the response
        if result_text is None:
            return

        # Fallback: emit a plain agent_response for results not covered by a reporter
        self._socketio.emit(
            "agent_response",
            {
                "message_id": message_id,
                "status": "completed",
                "final_answer": result_text,
            },
            room=room,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the thread pool and all sessions."""
        logger.info("Shutting down WebTaskDispatcher...")
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        logger.info("WebTaskDispatcher shut down")
