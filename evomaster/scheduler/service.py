"""Scheduler service — timer loop with Protocol-based execution

Decoupled from feishu/dispatcher/agent via two Protocols:
- TaskExecutor: run a job's task, return result text
- ResultNotifier: notify success/failure to the user

Reliability features (inspired by OpenClaw):
- Startup catch-up for missed one-shots
- Concurrent execution limit (Semaphore)
- Error classification + exponential backoff
- Run history recording + pruning
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from .errors import ErrorKind, classify_error
from .models import JobStatus, RunRecord, ScheduledJob
from .parser import compute_next_run
from .store import ScheduledJobStore

logger = logging.getLogger(__name__)

# Backoff schedule (seconds): 30s, 1m, 5m, 15m, 60m — same as OpenClaw
BACKOFF_SCHEDULE = [30, 60, 300, 900, 3600]


class TaskExecutor(Protocol):
    """Execute a scheduled job's task. Return result text. Raise on failure."""
    def execute(self, job: ScheduledJob) -> str: ...


class ResultNotifier(Protocol):
    """Notify the user about job execution results."""
    def notify_success(self, job: ScheduledJob, result: str) -> None: ...
    def notify_failure(self, job: ScheduledJob, error: str) -> None: ...


@dataclass
class SchedulerConfig:
    """Runtime config for the scheduler service."""
    max_concurrent_runs: int = 2
    max_poll_interval: float = 60.0
    max_catch_up: int = 5         # Max one-shot jobs to catch up on startup
    catch_up_stagger: float = 5.0  # Seconds between catch-up jobs
    prune_interval: int = 100      # Prune run history every N ticks


class SchedulerService:
    """Timer-loop scheduler that delegates execution via Protocols.

    Usage:
        service = SchedulerService(store, executor, notifier, config)
        service.start()   # spawns daemon thread
        ...
        service.stop()    # graceful shutdown
    """

    def __init__(
        self,
        store: ScheduledJobStore,
        executor: TaskExecutor,
        notifier: ResultNotifier,
        config: SchedulerConfig | None = None,
    ):
        self._store = store
        self._executor = executor
        self._notifier = notifier
        self._config = config or SchedulerConfig()

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._exec_semaphore = threading.Semaphore(self._config.max_concurrent_runs)
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    def start(self) -> None:
        """Start the scheduler daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="scheduler-loop",
        )
        self._thread.start()
        logger.info("Scheduler service started")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._stop_event.set()
        self._wake_event.set()  # Unblock the wait
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("Scheduler service stopped")

    def wake(self) -> None:
        """Wake the scheduler loop immediately (e.g. after adding a new job)."""
        self._wake_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main timer loop — runs in daemon thread."""
        try:
            self._catch_up_missed()
        except Exception:
            logger.exception("Error during catch-up")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Error in scheduler tick")

            # Sleep until next due time or max interval
            next_t = self._store.get_next_due_time()
            if next_t is not None:
                delay = next_t - time.time()
                delay = min(delay, self._config.max_poll_interval)
            else:
                delay = self._config.max_poll_interval

            self._wake_event.wait(timeout=max(0.5, delay))
            self._wake_event.clear()

    def _tick(self) -> None:
        """Process all due jobs in this tick."""
        now = time.time()
        due_jobs = self._store.get_due_jobs(now)

        for job in due_jobs:
            self._submit_job(job)

        # Periodic run history pruning
        self._tick_count += 1
        if self._tick_count % self._config.prune_interval == 0:
            try:
                pruned = self._store.prune_old_runs(max_per_job=50)
                if pruned:
                    logger.info("Pruned %d old run records", pruned)
            except Exception:
                logger.exception("Error pruning run history")

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _submit_job(self, job: ScheduledJob) -> None:
        """Submit a job for execution in a separate thread."""
        # Advance next_run_at immediately to prevent double-fire
        if job.max_runs == 1:
            # One-shot: set far future to avoid re-pick
            self._store.update_next_run(job.job_id, time.time() + 86400 * 365)
        else:
            try:
                next_run = compute_next_run(job)
                self._store.update_next_run(job.job_id, next_run)
            except Exception:
                logger.exception("Failed to compute next run for job %s", job.job_id)
                self._store.update_next_run(job.job_id, time.time() + 3600)

        thread = threading.Thread(
            target=self._execute_and_finalize,
            args=(job,),
            daemon=True,
            name=f"sched-exec-{job.job_id}",
        )
        thread.start()

    def _execute_and_finalize(self, job: ScheduledJob) -> None:
        """Execute a job and handle results (runs in worker thread)."""
        acquired = self._exec_semaphore.acquire(timeout=300)
        if not acquired:
            logger.warning("Semaphore timeout for job %s, skipping", job.job_id)
            return

        run_id = uuid.uuid4().hex[:12]
        started_at = time.time()

        # Record run start
        run = RunRecord(
            run_id=run_id, job_id=job.job_id, started_at=started_at,
        )
        try:
            self._store.add_run(run)
        except Exception:
            logger.exception("Failed to record run start for job %s", job.job_id)

        success = False
        result: str | None = None
        error_str: str | None = None
        error_kind: ErrorKind | None = None

        try:
            result = self._executor.execute(job)
            success = True
            # Truncate result
            if result and len(result) > 2000:
                result = result[:2000]
        except Exception as e:
            error_str = str(e)
            error_kind = classify_error(e)
            logger.warning(
                "Job %s execution failed (%s): %s",
                job.job_id, error_kind.value if error_kind else "unknown", error_str[:200],
            )
        finally:
            self._exec_semaphore.release()

        finished_at = time.time()

        # Update run record
        try:
            self._store.update_run(
                run_id, finished_at, success,
                result, error_str,
                error_kind.value if error_kind else None,
            )
        except Exception:
            logger.exception("Failed to update run record %s", run_id)

        # Update job state
        try:
            self._finalize_job(job, success, error_str, error_kind)
        except Exception:
            logger.exception("Failed to finalize job %s", job.job_id)

        # Notify user
        try:
            if success:
                # Re-fetch job for updated next_run_at
                updated_job = self._store.get(job.job_id) or job
                self._notifier.notify_success(updated_job, result or "")
            else:
                self._notifier.notify_failure(job, error_str or "Unknown error")
        except Exception:
            logger.exception("Failed to notify for job %s", job.job_id)

    def _finalize_job(
        self, job: ScheduledJob, success: bool,
        error_str: str | None, error_kind: ErrorKind | None,
    ) -> None:
        """Update job status after execution."""
        if success:
            self._store.update_after_run(job.job_id, True, None, None)
            # One-shot: mark completed
            if job.max_runs == 1:
                self._store.mark_completed(job.job_id)
        else:
            kind_val = error_kind.value if error_kind else "permanent"
            self._store.update_after_run(job.job_id, False, error_str, kind_val)

            # Re-fetch to get updated consecutive_failures
            updated = self._store.get(job.job_id)
            if not updated:
                return

            if error_kind == ErrorKind.TRANSIENT and updated.consecutive_failures < updated.max_retries:
                # Backoff retry
                idx = min(updated.consecutive_failures - 1, len(BACKOFF_SCHEDULE) - 1)
                backoff = BACKOFF_SCHEDULE[max(0, idx)]
                self._store.update_next_run(job.job_id, time.time() + backoff)
                logger.info(
                    "Job %s: transient failure %d/%d, retry in %ds",
                    job.job_id, updated.consecutive_failures, updated.max_retries, backoff,
                )
            elif error_kind == ErrorKind.PERMANENT or updated.consecutive_failures >= updated.max_retries:
                self._store.mark_failed(job.job_id)
                logger.info(
                    "Job %s: marked failed (%s, failures=%d)",
                    job.job_id,
                    error_kind.value if error_kind else "permanent",
                    updated.consecutive_failures,
                )
            # For recurring jobs with transient errors within retry limit,
            # next_run_at was already advanced above

    # ------------------------------------------------------------------
    # Startup catch-up
    # ------------------------------------------------------------------

    def _catch_up_missed(self) -> None:
        """On startup, handle jobs that were due while we were down.

        - One-shot (AT) jobs: execute up to max_catch_up, staggered.
        - Recurring (EVERY/CRON) jobs: just advance next_run_at.
        """
        now = time.time()
        missed = self._store.get_due_jobs(now)
        if not missed:
            return

        logger.info("Catch-up: %d missed jobs found", len(missed))

        one_shots = [j for j in missed if j.max_runs == 1]
        recurring = [j for j in missed if j.max_runs != 1]

        # Execute missed one-shots (limited + staggered)
        for i, job in enumerate(one_shots[: self._config.max_catch_up]):
            if self._stop_event.is_set():
                break
            if i > 0:
                time.sleep(self._config.catch_up_stagger)
            logger.info("Catch-up: executing one-shot job %s", job.job_id)
            self._submit_job(job)

        # Skip remaining one-shots
        for job in one_shots[self._config.max_catch_up :]:
            logger.info("Catch-up: skipping old one-shot job %s", job.job_id)
            self._store.mark_failed(job.job_id)

        # Recurring: just advance next_run_at
        for job in recurring:
            try:
                next_run = compute_next_run(job)
                self._store.update_next_run(job.job_id, next_run)
                logger.info(
                    "Catch-up: advanced recurring job %s to next run",
                    job.job_id,
                )
            except Exception:
                logger.exception(
                    "Catch-up: failed to advance job %s", job.job_id,
                )
