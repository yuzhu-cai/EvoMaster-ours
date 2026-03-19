#!/usr/bin/env python3
"""Run FrontierScience tasks from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

TASK_JSONL = Path("playground/frontierscience/test/test.jsonl")


@dataclass
class TaskJob:
    line_no: int
    task_dir: Path
    command: list[str]


@dataclass
class TaskResult:
    line_no: int
    task_dir: Path
    exit_code: int
    skipped: bool = False
    reason: str = ""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run FrontierScience tasks from JSONL.")
    parser.add_argument("--jsonl", type=Path, default=TASK_JSONL, help="Input dataset JSONL path.")
    parser.add_argument("--agent", default="frontierscience", help="Agent name passed to run.py.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file.")
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=project_root / "runs" / "frontierscience_run",
        help="Base run directory.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent worker count.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N tasks.")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def build_task_markdown(problem: str, task_id: str, task_type: str) -> str:
    header = [
        "[frontierscience_task_meta]",
        f"task_id={task_id}",
        f"task_type={task_type}",
        "allow_external_retrieval=true",
        "[/frontierscience_task_meta]",
        "",
    ]
    return "\n".join(header) + problem


def load_rows(path: Path, limit: int | None) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append((line_no, row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def prepare_job(
    line_no: int,
    row: dict,
    args: argparse.Namespace,
    project_root: Path,
) -> TaskJob | TaskResult:
    problem = str(row.get("problem", "")).strip()
    subject = sanitize_name(str(row.get("subject", "frontier_science")))
    task_group_id = sanitize_name(str(row.get("task_group_id", f"line{line_no:04d}")))
    task_dir = args.base_run_dir / f"{line_no:04d}_{subject}{task_group_id}"

    if not problem:
        return TaskResult(line_no=line_no, task_dir=task_dir, exit_code=1, skipped=True, reason="missing problem")

    task_dir.mkdir(parents=True, exist_ok=True)
    task_md = task_dir / "task.md"
    task_json = task_dir / "task.json"
    task_md.write_text(build_task_markdown(problem, task_group_id, subject), encoding="utf-8")
    task_json.write_text(
        json.dumps(
            [
                {
                    "id": task_group_id,
                    "description": str(task_md),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    command = [args.python, str(project_root / "run.py"), "--agent", args.agent, "--task-file", str(task_json), "--run-dir", str(task_dir)]
    if args.config is not None:
        command.extend(["--config", str(args.config)])

    return TaskJob(line_no=line_no, task_dir=task_dir, command=command)


def run_job(job: TaskJob) -> TaskResult:
    completed = subprocess.run(job.command, capture_output=True, text=True)
    (job.task_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (job.task_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    reason = ""
    if completed.returncode != 0:
        stderr_text = (completed.stderr or "").strip()
        stdout_text = (completed.stdout or "").strip()
        reason = (stderr_text or stdout_text).splitlines()[-1][:300] if (stderr_text or stdout_text) else "subprocess failed"
    return TaskResult(line_no=job.line_no, task_dir=job.task_dir, exit_code=completed.returncode, reason=reason)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    if not args.jsonl.exists():
        print(f"[ERROR] dataset not found: {args.jsonl}")
        return 1
    if args.workers < 1:
        print("[ERROR] --workers must be >= 1")
        return 1

    rows = load_rows(args.jsonl, args.limit)
    args.base_run_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[TaskJob] = []
    results: list[TaskResult] = []
    for line_no, row in rows:
        prepared = prepare_job(line_no, row, args, project_root)
        if isinstance(prepared, TaskResult):
            results.append(prepared)
        else:
            jobs.append(prepared)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_map):
            results.append(future.result())
            done = len(results)
            total = len(jobs) + len([r for r in results if r.skipped])
            if done % 10 == 0 or done == len(jobs) + len([r for r in results if r.skipped]):
                print(f"[INFO] progress: {done}/{len(jobs) + len([r for r in results if r.skipped])}")

    failed = [result for result in results if not result.skipped and result.exit_code != 0]
    skipped = [result for result in results if result.skipped]

    for result in sorted(results, key=lambda item: item.line_no):
        status = "SKIP" if result.skipped else ("OK" if result.exit_code == 0 else "FAIL")
        suffix = f" ({result.reason})" if result.reason else ""
        print(f"[{status}] line={result.line_no} dir={result.task_dir}{suffix}")

    print(f"[DONE] total={len(results)} failed={len(failed)} skipped={len(skipped)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
