"""Feishu adapter for scheduler Protocols

Implements TaskExecutor and ResultNotifier — the only file
in the scheduler system that knows about Feishu.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from evomaster.scheduler.models import JobStatus, ScheduledJob

if TYPE_CHECKING:
    import lark_oapi as lark

logger = logging.getLogger(__name__)


def _format_time(epoch: float, tz_str: str = "Asia/Shanghai") -> str:
    """Format epoch → human-readable local time string."""
    try:
        from zoneinfo import ZoneInfo
        tzinfo = ZoneInfo(tz_str)
    except Exception:
        tzinfo = None
    dt = datetime.fromtimestamp(epoch, tz=tzinfo)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


_REMINDER_KEYWORDS = ("提醒", "remind", "通知", "notify", "告诉", "tell")


def _is_simple_reminder(desc: str) -> bool:
    """Short task containing a reminder keyword → send message directly."""
    return len(desc) < 100 and any(kw in desc.lower() for kw in _REMINDER_KEYWORDS)


class FeishuTaskExecutor:
    """Implements TaskExecutor Protocol — runs tasks via dispatcher._run_subtask."""

    def __init__(self, run_subtask_fn: Callable[..., str]):
        self._run_subtask = run_subtask_fn

    def execute(self, job: ScheduledJob) -> str:
        # 简单提醒：跳过 agent，直接返回描述（由 notifier 发消息）
        if _is_simple_reminder(job.task_description):
            return job.task_description

        # 复杂任务：走 agent 但加上下文前缀防止递归调度
        wrapped_task = (
            "[定时任务] 这是由定时调度器自动触发的任务。"
            "请直接执行以下任务并汇报结果，不要创建新的定时任务。\n\n"
            f"任务: {job.task_description}"
        )
        return self._run_subtask(
            job.agent_name,
            wrapped_task,
            on_step=None,
            chat_id=job.chat_id,
            sender_open_id=job.creator_id,
        )


class FeishuResultNotifier:
    """Implements ResultNotifier Protocol — sends results via Feishu messages."""

    def __init__(self, feishu_client: lark.Client, max_retries: int = 2):
        self._client = feishu_client
        self._max_retries = max_retries

    def notify_success(self, job: ScheduledJob, result: str) -> None:
        if _is_simple_reminder(job.task_description):
            text = f"⏰ 提醒: {job.task_description}"
        else:
            text = (
                f"⏰ 定时任务完成\n"
                f"任务: {job.task_description[:200]}\n"
                f"结果:\n{result[:2000]}"
            )
        if job.status == JobStatus.ACTIVE:
            text += f"\n\n下次执行: {_format_time(job.next_run_at, job.timezone)}"
        self._send_with_retry(job.chat_id, text)

    def notify_failure(self, job: ScheduledJob, error: str) -> None:
        text = (
            f"⏰ 定时任务失败\n"
            f"任务: {job.task_description[:200]}\n"
            f"错误: {error[:500]}"
        )
        if job.consecutive_failures > 0 and job.consecutive_failures < job.max_retries:
            text += f"\n\n将自动重试 ({job.consecutive_failures}/{job.max_retries})"
        self._send_with_retry(job.chat_id, text)

    def _send_with_retry(self, chat_id: str, text: str) -> None:
        from .messaging.sender import send_text_message

        for attempt in range(self._max_retries + 1):
            try:
                send_text_message(self._client, chat_id, text)
                return
            except Exception:
                if attempt == self._max_retries:
                    logger.error(
                        "Failed to send scheduler notification after %d retries",
                        self._max_retries,
                    )
                else:
                    time.sleep(1)
