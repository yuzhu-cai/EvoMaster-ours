"""Schedule expression parser

Supports three schedule types:
- AT:    one-shot — ISO 8601 ("2026-03-21T15:00:00") or relative ("30m", "2h", "1d")
- EVERY: fixed interval — "30m", "1h", "6h"
- CRON:  5-field cron — "0 9 * * 1-5"
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

from .models import ScheduleType, ScheduledJob

# Relative time pattern: 30m, 2h, 1d, 30s
_RELATIVE_RE = re.compile(r"^(\d+)\s*([smhd])$", re.IGNORECASE)

# Interval units → seconds
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _get_tz(timezone_str: str):
    """Get a timezone object from an IANA timezone string."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(timezone_str)
    except ImportError:
        # Fallback for Python < 3.9 (shouldn't happen with 3.12)
        import dateutil.tz
        return dateutil.tz.gettz(timezone_str)


def parse_relative_duration(expr: str) -> float:
    """Parse a relative duration like '30m', '2h', '1d' → seconds.

    Raises ValueError on invalid format.
    """
    m = _RELATIVE_RE.match(expr.strip())
    if not m:
        raise ValueError(f"Invalid relative duration: {expr!r}")
    amount = int(m.group(1))
    unit = m.group(2).lower()
    return amount * _UNIT_SECONDS[unit]


def parse_at_expr(expr: str, tz: str = "Asia/Shanghai") -> float:
    """Parse an AT expression → epoch timestamp.

    Accepts:
    - ISO 8601: "2026-03-21T15:00:00" (interpreted in given timezone)
    - Relative: "30m", "2h", "1d" (from now)

    Raises ValueError on invalid format or past timestamp.
    """
    expr = expr.strip()

    # Try relative first
    m = _RELATIVE_RE.match(expr)
    if m:
        seconds = parse_relative_duration(expr)
        return time.time() + seconds

    # Try ISO 8601
    try:
        tzinfo = _get_tz(tz)
        dt = datetime.fromisoformat(expr)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        ts = dt.timestamp()
        if ts < time.time():
            raise ValueError(f"Scheduled time is in the past: {expr}")
        return ts
    except (ValueError, TypeError) as e:
        if "in the past" in str(e):
            raise
        raise ValueError(
            f"Invalid AT expression: {expr!r}. "
            "Expected ISO 8601 (e.g. '2026-03-21T15:00:00') "
            "or relative duration (e.g. '30m', '2h')."
        ) from e


def parse_every_expr(expr: str) -> float:
    """Parse an EVERY expression → interval in seconds.

    Accepts: "30m", "1h", "6h", "30s", "1d"
    Minimum interval: 60 seconds (1 minute).

    Raises ValueError on invalid format or too-short interval.
    """
    seconds = parse_relative_duration(expr)
    if seconds < 60:
        raise ValueError(f"Minimum interval is 60 seconds (1 minute), got {seconds}s")
    return seconds


def validate_cron_expr(expr: str) -> bool:
    """Validate a 5-field cron expression.

    Returns True if valid, raises ValueError if not.
    """
    try:
        from croniter import croniter
        croniter(expr)
        return True
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid cron expression: {expr!r}. {e}") from e


def compute_next_run(job: ScheduledJob, after: float | None = None) -> float:
    """Compute the next fire time for a job.

    Args:
        job: The scheduled job
        after: Compute next run after this timestamp (default: now)

    Returns:
        Epoch timestamp for next run
    """
    if after is None:
        after = time.time()

    if job.schedule_type == ScheduleType.AT:
        # One-shot: next_run_at is the target time, no recurrence
        return parse_at_expr(job.schedule_expr, job.timezone)

    elif job.schedule_type == ScheduleType.EVERY:
        interval = parse_every_expr(job.schedule_expr)
        return after + interval

    elif job.schedule_type == ScheduleType.CRON:
        from croniter import croniter
        tzinfo = _get_tz(job.timezone)
        base_dt = datetime.fromtimestamp(after, tz=tzinfo)
        cron = croniter(job.schedule_expr, base_dt)
        next_dt = cron.get_next(datetime)
        return next_dt.timestamp()

    raise ValueError(f"Unknown schedule type: {job.schedule_type}")
