"""Protocol-based scheduler module

Zero external dependencies on feishu/dispatcher/agent.
Adapter layer implements TaskExecutor/ResultNotifier protocols.
"""

from .errors import ErrorKind, classify_error
from .models import JobStatus, RunRecord, ScheduleType, ScheduledJob
from .parser import compute_next_run, parse_at_expr, parse_every_expr, validate_cron_expr
from .service import ResultNotifier, SchedulerService, TaskExecutor
from .store import ScheduledJobStore

__all__ = [
    "ErrorKind",
    "JobStatus",
    "ResultNotifier",
    "RunRecord",
    "ScheduleType",
    "ScheduledJob",
    "ScheduledJobStore",
    "SchedulerService",
    "TaskExecutor",
    "classify_error",
    "compute_next_run",
    "parse_at_expr",
    "parse_every_expr",
    "validate_cron_expr",
]
