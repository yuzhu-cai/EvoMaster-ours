"""后台子智能体任务管理

提供 BackgroundTask 数据类和线程安全的 BackgroundTaskRegistry，
用于跟踪 magiclaw 委派给子智能体的后台任务。
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """一个后台子智能体任务。"""

    task_id: str
    chat_id: str
    agent_name: str
    task_description: str
    status: BackgroundTaskStatus = BackgroundTaskStatus.PENDING
    created_at: float = field(default_factory=time.monotonic)
    step_count: int = 0
    latest_tool_calls: list[str] = field(default_factory=list)  # 最近 3 个工具调用名
    result: str | None = None
    error: str | None = None
    reviewed: bool = False
    agent: Any = None       # BaseAgent reference for cancellation
    reporter: Any = None    # FeishuStepReporter reference

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def is_active(self) -> bool:
        return self.status in (BackgroundTaskStatus.PENDING, BackgroundTaskStatus.RUNNING)


class BackgroundTaskRegistry:
    """线程安全的后台任务注册表，按 chat_id 分组管理。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # chat_id → {task_id → BackgroundTask}
        self._tasks: dict[str, dict[str, BackgroundTask]] = {}

    def create(
        self, chat_id: str, agent_name: str, task_description: str
    ) -> BackgroundTask:
        """创建并注册一个新的后台任务。"""
        task_id = uuid.uuid4().hex[:8]
        task = BackgroundTask(
            task_id=task_id,
            chat_id=chat_id,
            agent_name=agent_name,
            task_description=task_description,
        )
        with self._lock:
            self._tasks.setdefault(chat_id, {})[task_id] = task
        logger.info(
            "Background task created: id=%s, chat=%s, agent=%s",
            task_id, chat_id, agent_name,
        )
        return task

    def get_tasks_for_chat(self, chat_id: str) -> list[BackgroundTask]:
        """获取某个 chat 的所有后台任务。"""
        with self._lock:
            tasks = self._tasks.get(chat_id, {})
            return list(tasks.values())

    def get_active_tasks(self, chat_id: str) -> list[BackgroundTask]:
        """获取某个 chat 的所有活跃（PENDING/RUNNING）任务。"""
        with self._lock:
            tasks = self._tasks.get(chat_id, {})
            return [t for t in tasks.values() if t.is_active]

    def get_task(self, chat_id: str, task_id: str) -> BackgroundTask | None:
        """获取指定的后台任务。"""
        with self._lock:
            return self._tasks.get(chat_id, {}).get(task_id)

    def get_unreviewed_completed(self, chat_id: str) -> list[BackgroundTask]:
        """获取某个 chat 已完成但未审阅的任务。"""
        with self._lock:
            tasks = self._tasks.get(chat_id, {})
            return [
                t for t in tasks.values()
                if t.status in (BackgroundTaskStatus.COMPLETED, BackgroundTaskStatus.FAILED)
                and not t.reviewed
            ]

    def update_step(self, task: BackgroundTask, step_count: int, tool_name: str | None = None) -> None:
        """更新任务进度（线程安全）。"""
        with self._lock:
            task.status = BackgroundTaskStatus.RUNNING
            task.step_count = step_count
            if tool_name:
                task.latest_tool_calls.append(tool_name)
                # 只保留最近 3 个
                if len(task.latest_tool_calls) > 3:
                    task.latest_tool_calls = task.latest_tool_calls[-3:]

    def mark_completed(self, task: BackgroundTask, result: str) -> None:
        """标记任务完成。"""
        with self._lock:
            task.status = BackgroundTaskStatus.COMPLETED
            task.result = result
        logger.info("Background task completed: id=%s, agent=%s", task.task_id, task.agent_name)

    def mark_failed(self, task: BackgroundTask, error: str) -> None:
        """标记任务失败。"""
        with self._lock:
            task.status = BackgroundTaskStatus.FAILED
            task.error = error
        logger.warning("Background task failed: id=%s, agent=%s, error=%s", task.task_id, task.agent_name, error)

    def mark_cancelled(self, task: BackgroundTask) -> None:
        """标记任务被用户取消。仅在任务仍活跃时生效。"""
        with self._lock:
            if task.is_active:
                task.status = BackgroundTaskStatus.CANCELLED
                task.reviewed = True  # 取消的任务跳过自动审阅
                logger.info("Background task cancelled: id=%s, agent=%s", task.task_id, task.agent_name)
            else:
                logger.info("Background task %s already %s, skip cancel", task.task_id, task.status.value)

    def mark_reviewed(self, task: BackgroundTask) -> None:
        """标记任务已审阅。"""
        with self._lock:
            task.reviewed = True

    def cleanup_old(self, chat_id: str, max_age_seconds: float = 3600) -> int:
        """清理指定 chat 过老的已完成任务，返回清理数量。"""
        now = time.monotonic()
        removed = 0
        with self._lock:
            tasks = self._tasks.get(chat_id, {})
            to_remove = [
                tid for tid, t in tasks.items()
                if not t.is_active and t.reviewed and (now - t.created_at) > max_age_seconds
            ]
            for tid in to_remove:
                del tasks[tid]
                removed += 1
        return removed
