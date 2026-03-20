"""Error classification for scheduler retry logic"""

from __future__ import annotations

import enum


class ErrorKind(enum.Enum):
    TRANSIENT = "transient"    # Rate limit, timeout, network — worth retrying
    PERMANENT = "permanent"    # Task logic error — don't retry


_TRANSIENT_KEYWORDS = (
    "rate", "limit", "429", "timeout", "overloaded",
    "connection", "频率", "throttl", "503", "502",
    "temporary", "unavailable",
)


def classify_error(error: Exception) -> ErrorKind:
    """Classify an error as transient or permanent based on message keywords."""
    msg = str(error).lower()
    if any(kw in msg for kw in _TRANSIENT_KEYWORDS):
        return ErrorKind.TRANSIENT
    return ErrorKind.PERMANENT
