"""Feishu Bot session manager

Manages PlaygroundSession instances keyed by chat_id, enabling multi-turn conversation context persistence.
Sessions never expire automatically; they are only cleaned up via the /new command or when the bot shuts down.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlaygroundSession:
    """An active playground session corresponding to a single chat_id."""

    chat_id: str
    playground: Any  # BasePlayground instance
    agent: Any = None  # BaseAgent instance, set after setup()
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    message_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    initialized: bool = False
    last_card_message_id: str | None = None  # ID of the last card with buttons (used to remove old buttons in multi-turn)
    pending_questions: list[dict] = field(default_factory=list)   # Sequential questioning: queue of follow-up questions to present
    collected_answers: list[str] = field(default_factory=list)    # Sequential questioning: collected answers so far
    pending_reviews: list[dict] = field(default_factory=list)     # 后台任务完成后待审阅队列
    # 每项: {"task_id", "agent_name", "task_description", "result", "status"}
    dispatched_delegation_keys: set = field(default_factory=set)  # (agent_name, task) 已即时 dispatch 的委派，防止重复
    current_reporter: Any = None          # FeishuStepReporter, set during task execution
    current_running_agent: Any = None     # agent actively in run() loop (may differ from self.agent for builder phase2)


class ChatSessionManager:
    """Manage the chat_id -> PlaygroundSession mapping.

    - One chat_id corresponds to one PlaygroundSession.
    - Playgrounds are reused across messages within a session (no setup/cleanup per message).
    - Sessions never time out; they are only cleaned up via remove() or shutdown().
    - Thread-safe: a global lock protects the sessions dict; a per-session lock serializes messages within the same chat.
    """

    def __init__(self, max_sessions: int = 100, on_session_cleanup: Optional[Callable[[PlaygroundSession], None]] = None):
        self._sessions: dict[str, PlaygroundSession] = {}
        self._global_lock = threading.Lock()
        self._max_sessions = max_sessions
        self._on_session_cleanup = on_session_cleanup

    def get_or_create(
        self,
        chat_id: str,
        playground_factory: Callable[[], Any],
    ) -> PlaygroundSession:
        """Get an existing session or create a new one.

        Args:
            chat_id: Feishu chat ID.
            playground_factory: Factory function to create a new playground instance (does not call setup).

        Returns:
            A PlaygroundSession instance.
        """
        with self._global_lock:
            session = self._sessions.get(chat_id)
            if session is not None:
                return session

            # Check session count limit
            if len(self._sessions) >= self._max_sessions:
                # Evict the least recently active session
                oldest_key = min(
                    self._sessions,
                    key=lambda k: self._sessions[k].last_activity,
                )
                logger.warning(
                    "Max sessions (%d) reached, evicting oldest: %s",
                    self._max_sessions,
                    oldest_key,
                )
                self._cleanup_session(self._sessions.pop(oldest_key))

            # Create new session
            playground = playground_factory()
            session = PlaygroundSession(
                chat_id=chat_id,
                playground=playground,
            )
            self._sessions[chat_id] = session
            logger.info("Created new session for chat_id=%s", chat_id)
            return session

    def get(self, chat_id: str) -> Optional[PlaygroundSession]:
        """Get an existing session without creating one.

        Args:
            chat_id: Chat ID or session key.

        Returns:
            A PlaygroundSession, or None if not found.
        """
        with self._global_lock:
            return self._sessions.get(chat_id)

    def remove(self, chat_id: str) -> None:
        """Remove and clean up a session (called on /new command)."""
        with self._global_lock:
            session = self._sessions.pop(chat_id, None)

        if session is not None:
            self._cleanup_session(session)
            logger.info("Removed session for chat_id=%s", chat_id)

    def get_session_count(self) -> int:
        """Return the number of currently active sessions."""
        with self._global_lock:
            return len(self._sessions)

    def get_sessions_by_prefix(self, prefix: str) -> list[PlaygroundSession]:
        """查找 key 以 prefix: 开头的所有会话（用于 /stop 查找子会话）。"""
        with self._global_lock:
            return [s for k, s in self._sessions.items() if k.startswith(prefix + ":")]

    def shutdown(self) -> None:
        """Shut down all sessions and release resources."""
        with self._global_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            self._cleanup_session(session)

        logger.info("ChatSessionManager shut down, cleaned up %d sessions", len(sessions))

    def _cleanup_session(self, session: PlaygroundSession) -> None:
        """Clean up resources for a single session."""
        try:
            if session.initialized and session.playground is not None:
                session.playground.cleanup()
        except Exception:
            logger.exception(
                "Error cleaning up session for chat_id=%s", session.chat_id
            )
        # 释放容器回池（通过回调）
        if self._on_session_cleanup:
            try:
                self._on_session_cleanup(session)
            except Exception:
                logger.exception(
                    "Error in on_session_cleanup callback for chat_id=%s",
                    session.chat_id,
                )
