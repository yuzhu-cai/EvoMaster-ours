"""SQLite persistence for scheduled jobs and run history

Thread-safe via threading.Lock. Follows evomaster/memory/store.py pattern.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .models import JobStatus, RunRecord, ScheduleType, ScheduledJob

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id              TEXT PRIMARY KEY,
    chat_id             TEXT NOT NULL,
    creator_id          TEXT NOT NULL,
    schedule_type       TEXT NOT NULL,
    schedule_expr       TEXT NOT NULL,
    timezone            TEXT NOT NULL DEFAULT 'Asia/Shanghai',
    task_description    TEXT NOT NULL,
    agent_name          TEXT NOT NULL DEFAULT 'magiclaw',
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          REAL NOT NULL,
    next_run_at         REAL NOT NULL,
    last_run_at         REAL,
    last_error          TEXT,
    run_count           INTEGER NOT NULL DEFAULT 0,
    max_runs            INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    max_retries         INTEGER NOT NULL DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_jobs_next ON scheduled_jobs(status, next_run_at);
CREATE INDEX IF NOT EXISTS idx_jobs_chat ON scheduled_jobs(chat_id, status);

CREATE TABLE IF NOT EXISTS scheduled_runs (
    run_id      TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    success     INTEGER,
    result      TEXT,
    error       TEXT,
    error_kind  TEXT,
    FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_job ON scheduled_runs(job_id, started_at DESC);
"""


class ScheduledJobStore:
    """SQLite store for scheduled jobs and run records.

    Thread-safe: all DB operations protected by a threading.Lock.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def add(self, job: ScheduledJob) -> None:
        """Insert a new job."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO scheduled_jobs "
                "(job_id, chat_id, creator_id, schedule_type, schedule_expr, timezone, "
                " task_description, agent_name, status, created_at, next_run_at, "
                " last_run_at, last_error, run_count, max_runs, consecutive_failures, max_retries)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.job_id, job.chat_id, job.creator_id,
                    job.schedule_type.value, job.schedule_expr, job.timezone,
                    job.task_description, job.agent_name, job.status.value,
                    job.created_at, job.next_run_at,
                    job.last_run_at, job.last_error, job.run_count,
                    job.max_runs, job.consecutive_failures, job.max_retries,
                ),
            )
            self._conn.commit()

    def get(self, job_id: str) -> ScheduledJob | None:
        """Get a job by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scheduled_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def get_active_for_chat(self, chat_id: str) -> list[ScheduledJob]:
        """Get all active jobs for a chat."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_jobs WHERE chat_id = ? AND status = 'active' "
                "ORDER BY created_at DESC",
                (chat_id,),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def count_active_for_chat(self, chat_id: str) -> int:
        """Count active jobs for a chat."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM scheduled_jobs "
                "WHERE chat_id = ? AND status = 'active'",
                (chat_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def count_active_total(self) -> int:
        """Count all active jobs globally."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM scheduled_jobs WHERE status = 'active'"
            ).fetchone()
        return row["cnt"] if row else 0

    def get_due_jobs(self, now: float) -> list[ScheduledJob]:
        """Get all active jobs whose next_run_at <= now."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_jobs "
                "WHERE status = 'active' AND next_run_at <= ? "
                "ORDER BY next_run_at ASC",
                (now,),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def get_next_due_time(self) -> float | None:
        """Get the earliest next_run_at among active jobs, or None if no active jobs."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(next_run_at) AS nxt FROM scheduled_jobs "
                "WHERE status = 'active'"
            ).fetchone()
        return row["nxt"] if row and row["nxt"] is not None else None

    # ------------------------------------------------------------------
    # Job updates
    # ------------------------------------------------------------------

    def update_next_run(self, job_id: str, next_run_at: float) -> None:
        """Update only the next_run_at field."""
        with self._lock:
            self._conn.execute(
                "UPDATE scheduled_jobs SET next_run_at = ? WHERE job_id = ?",
                (next_run_at, job_id),
            )
            self._conn.commit()

    def update_after_run(
        self, job_id: str, success: bool, error: str | None, error_kind: str | None,
    ) -> None:
        """Update job state after a run attempt."""
        now = time.time()
        with self._lock:
            if success:
                self._conn.execute(
                    "UPDATE scheduled_jobs SET "
                    "  last_run_at = ?, run_count = run_count + 1, "
                    "  consecutive_failures = 0, last_error = NULL "
                    "WHERE job_id = ?",
                    (now, job_id),
                )
            else:
                self._conn.execute(
                    "UPDATE scheduled_jobs SET "
                    "  last_run_at = ?, run_count = run_count + 1, "
                    "  consecutive_failures = consecutive_failures + 1, "
                    "  last_error = ? "
                    "WHERE job_id = ?",
                    (now, error, job_id),
                )
            self._conn.commit()

    def mark_completed(self, job_id: str) -> None:
        """Mark a job as completed (one-shot finished)."""
        with self._lock:
            self._conn.execute(
                "UPDATE scheduled_jobs SET status = 'completed' WHERE job_id = ?",
                (job_id,),
            )
            self._conn.commit()

    def mark_cancelled(self, job_id: str) -> None:
        """Mark a job as cancelled."""
        with self._lock:
            self._conn.execute(
                "UPDATE scheduled_jobs SET status = 'cancelled' WHERE job_id = ?",
                (job_id,),
            )
            self._conn.commit()

    def mark_failed(self, job_id: str) -> None:
        """Mark a job as failed (exceeded retry limit)."""
        with self._lock:
            self._conn.execute(
                "UPDATE scheduled_jobs SET status = 'failed' WHERE job_id = ?",
                (job_id,),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def add_run(self, record: RunRecord) -> None:
        """Insert a run record."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO scheduled_runs "
                "(run_id, job_id, started_at, finished_at, success, result, error, error_kind)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    record.run_id, record.job_id, record.started_at,
                    record.finished_at, int(record.success) if record.success is not None else None,
                    record.result, record.error, record.error_kind,
                ),
            )
            self._conn.commit()

    def update_run(self, run_id: str, finished_at: float, success: bool,
                   result: str | None, error: str | None, error_kind: str | None) -> None:
        """Update a run record after execution."""
        with self._lock:
            self._conn.execute(
                "UPDATE scheduled_runs SET "
                "  finished_at = ?, success = ?, result = ?, error = ?, error_kind = ? "
                "WHERE run_id = ?",
                (finished_at, int(success), result, error, error_kind, run_id),
            )
            self._conn.commit()

    def get_runs(self, job_id: str, limit: int = 20) -> list[RunRecord]:
        """Get recent run records for a job."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scheduled_runs WHERE job_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def prune_old_runs(self, max_per_job: int = 50) -> int:
        """Delete old run records, keeping at most max_per_job per job."""
        pruned = 0
        with self._lock:
            job_ids = self._conn.execute(
                "SELECT DISTINCT job_id FROM scheduled_runs"
            ).fetchall()
            for row in job_ids:
                jid = row["job_id"]
                count = self._conn.execute(
                    "SELECT COUNT(*) AS cnt FROM scheduled_runs WHERE job_id = ?",
                    (jid,),
                ).fetchone()["cnt"]
                if count > max_per_job:
                    excess = count - max_per_job
                    self._conn.execute(
                        "DELETE FROM scheduled_runs WHERE run_id IN ("
                        "  SELECT run_id FROM scheduled_runs WHERE job_id = ? "
                        "  ORDER BY started_at ASC LIMIT ?"
                        ")",
                        (jid, excess),
                    )
                    pruned += excess
            if pruned:
                self._conn.commit()
        return pruned

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ScheduledJob:
        return ScheduledJob(
            job_id=row["job_id"],
            chat_id=row["chat_id"],
            creator_id=row["creator_id"],
            schedule_type=ScheduleType(row["schedule_type"]),
            schedule_expr=row["schedule_expr"],
            timezone=row["timezone"],
            task_description=row["task_description"],
            agent_name=row["agent_name"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            next_run_at=row["next_run_at"],
            last_run_at=row["last_run_at"],
            last_error=row["last_error"],
            run_count=row["run_count"],
            max_runs=row["max_runs"],
            consecutive_failures=row["consecutive_failures"],
            max_retries=row["max_retries"],
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            job_id=row["job_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            success=bool(row["success"]) if row["success"] is not None else False,
            result=row["result"],
            error=row["error"],
            error_kind=row["error_kind"],
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a 12-char hex ID for jobs/runs."""
        return uuid.uuid4().hex[:12]
