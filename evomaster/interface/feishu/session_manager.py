"""飞书 Bot 会话管理器

以 chat_id 为 key 管理 PlaygroundSession，实现多轮对话上下文延续。
会话永不自动过期，只通过 \\new 命令或 bot 关闭时清理。
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
    """一个活跃的 playground 会话，对应一个 chat_id。"""

    chat_id: str
    playground: Any  # BasePlayground instance
    agent: Any = None  # BaseAgent instance, set after setup()
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    message_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    initialized: bool = False
    last_card_message_id: str | None = None  # 最后一张带按钮的卡片 ID（多轮时移除旧按钮）


class ChatSessionManager:
    """管理 chat_id -> PlaygroundSession 的映射。

    - 一个 chat_id 对应一个 PlaygroundSession
    - Playground 在会话间复用（不会每条消息都 setup/cleanup）
    - 会话永不超时，只通过 remove() 或 shutdown() 清理
    - 线程安全：全局锁保护 sessions 字典，per-session 锁串行处理同一 chat 的消息
    """

    def __init__(self, max_sessions: int = 100):
        self._sessions: dict[str, PlaygroundSession] = {}
        self._global_lock = threading.Lock()
        self._max_sessions = max_sessions

    def get_or_create(
        self,
        chat_id: str,
        playground_factory: Callable[[], Any],
    ) -> PlaygroundSession:
        """获取已有会话或创建新会话。

        Args:
            chat_id: 飞书聊天 ID
            playground_factory: 创建新 playground 实例的工厂函数（不调用 setup）

        Returns:
            PlaygroundSession 实例
        """
        with self._global_lock:
            session = self._sessions.get(chat_id)
            if session is not None:
                return session

            # 检查会话数量限制
            if len(self._sessions) >= self._max_sessions:
                # 移除最久未活跃的会话
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

            # 创建新会话
            playground = playground_factory()
            session = PlaygroundSession(
                chat_id=chat_id,
                playground=playground,
            )
            self._sessions[chat_id] = session
            logger.info("Created new session for chat_id=%s", chat_id)
            return session

    def get(self, chat_id: str) -> Optional[PlaygroundSession]:
        """获取已有会话（不创建）。

        Args:
            chat_id: 聊天 ID 或会话 key

        Returns:
            PlaygroundSession 或 None
        """
        with self._global_lock:
            return self._sessions.get(chat_id)

    def remove(self, chat_id: str) -> None:
        """移除并清理一个会话（\\new 命令时调用）。"""
        with self._global_lock:
            session = self._sessions.pop(chat_id, None)

        if session is not None:
            self._cleanup_session(session)
            logger.info("Removed session for chat_id=%s", chat_id)

    def get_session_count(self) -> int:
        """返回当前活跃会话数。"""
        with self._global_lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        """关闭所有会话，释放资源。"""
        with self._global_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            self._cleanup_session(session)

        logger.info("ChatSessionManager shut down, cleaned up %d sessions", len(sessions))

    def _cleanup_session(self, session: PlaygroundSession) -> None:
        """清理单个会话的资源。"""
        try:
            if session.initialized and session.playground is not None:
                session.playground.cleanup()
        except Exception:
            logger.exception(
                "Error cleaning up session for chat_id=%s", session.chat_id
            )
