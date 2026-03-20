"""Scheduler data models

Pure dataclasses — no external dependencies.
chat_id / creator_id are opaque strings (not feishu-specific).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ScheduleType(enum.Enum):
    AT = "at"        # One-shot: ISO 8601 or relative ("30m", "2h")
    EVERY = "every"  # Fixed interval ("30m", "1h")
    CRON = "cron"    # 5-field cron ("0 9 * * 1-5")


class JobStatus(enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"      # One-shot finished successfully
    FAILED = "failed"            # Exceeded retry limit
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    job_id: str                  # uuid hex[:12]
    chat_id: str                 # Opaque target conversation key
    creator_id: str              # Opaque creator key
    schedule_type: ScheduleType
    schedule_expr: str
    task_description: str
    agent_name: str
    status: JobStatus
    created_at: float            # epoch seconds
    next_run_at: float           # pre-computed next fire time
    timezone: str = "Asia/Shanghai"
    last_run_at: float | None = None
    last_error: str | None = None
    run_count: int = 0
    max_runs: int | None = None  # 1 = one-shot, None = unlimited
    consecutive_failures: int = 0
    max_retries: int = 3


@dataclass
class RunRecord:
    run_id: str
    job_id: str
    started_at: float
    finished_at: float | None = None
    success: bool = False
    result: str | None = None    # truncated to 2000 chars
    error: str | None = None
    error_kind: str | None = None  # "transient" / "permanent"
