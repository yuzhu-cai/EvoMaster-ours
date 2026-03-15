#!/usr/bin/env python3
"""Run FrontierScience tasks line-by-line from a JSONL file.

For each JSON object:
1. Read `problem` as task content.
2. Write task metadata header (`task_id`, `task_type`) into `task.md`.
3. Build a single-row `task.json` and run `python run.py --agent ... --task-file <task.json> --run-dir <task_run_dir>`.
4. Rename `<task_run_dir>/logs` to `<subject><task_group_id>`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TaskResult:
    line_no: int
    run_dir: Path
    log_dir_name: str
    exit_code: int
    skipped: bool = False
    reason: str = ""


@dataclass
class TaskJob:
    line_no: int
    run_dir: Path
    log_dir_name: str
    cmd: list[str]


def build_task_markdown(problem: str, task_id: str, task_type: str, extra_meta: dict[str, str] | None = None) -> str:
    lines = [
        "[frontierscience_task_meta]",
        f"task_id={task_id}",
        f"task_type={task_type}",
    ]
    for k, v in (extra_meta or {}).items():
        if not k or v is None:
            continue
        lines.append(f"{k}={v}")
    lines.append("[/frontierscience_task_meta]")
    lines.append("")
    lines.append(problem)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    default_jsonl = project_root / "test.jsonl"
    default_base_run_dir = project_root / "runs" / f"frontierscience_jsonl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    parser = argparse.ArgumentParser(
        description=(
            "Run each row in test.jsonl and rename the logs folder "
            "to subject+task_group_id."
        )
    )
    parser.add_argument("--jsonl", type=Path, default=default_jsonl, help="Input JSONL file path.")
    parser.add_argument("--agent", default="frontierscience", help="Agent name passed to run.py.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path passed to run.py.")
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=default_base_run_dir,
        help="Base output directory. One sub-directory per row will be created.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to call run.py.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker count. Use >1 for parallel run.")
    parser.add_argument(
        "--lines",
        default=None,
        help="Only run specific JSONL lines, e.g. '2', '2,5,9', '2,5,8-10'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N valid rows.")
    parser.add_argument("--start-line", type=int, default=1, help="Start from this JSONL line number (1-based).")
    parser.add_argument(
        "--task-id-field",
        default="task_group_id",
        help="JSONL field used as runtime task_id. Default: task_group_id",
    )
    parser.add_argument(
        "--task-type-field",
        default="subject",
        help="JSONL field used as runtime task_type. Default: subject",
    )
    parser.add_argument(
        "--default-task-type",
        default="frontier_science",
        help="Fallback task_type when --task-type-field is missing/empty.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a row fails.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "unknown"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def parse_line_selector(lines_text: str | None) -> set[int] | None:
    if lines_text is None:
        return None

    selected: set[int] = set()
    for token in lines_text.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            seg = part.split("-", 1)
            if len(seg) != 2 or not seg[0].strip().isdigit() or not seg[1].strip().isdigit():
                raise ValueError(f"Invalid range token: '{part}'")
            start = int(seg[0].strip())
            end = int(seg[1].strip())
            if start <= 0 or end <= 0:
                raise ValueError(f"Line number must be >= 1: '{part}'")
            if start > end:
                raise ValueError(f"Range start must be <= end: '{part}'")
            for line_no in range(start, end + 1):
                selected.add(line_no)
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid line token: '{part}'")
            line_no = int(part)
            if line_no <= 0:
                raise ValueError(f"Line number must be >= 1: '{part}'")
            selected.add(line_no)

    if not selected:
        raise ValueError("No valid line number selected.")
    return selected


def load_jsonl(path: Path, start_line: int, selected_lines: set[int] | None):
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if line_no < start_line:
                continue
            if selected_lines is not None and line_no not in selected_lines:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            yield line_no, obj


def build_jobs(args: argparse.Namespace, run_py: Path, selected_lines: set[int] | None) -> tuple[list[TaskJob], list[TaskResult]]:
    jobs: list[TaskJob] = []
    pre_results: list[TaskResult] = []
    executed_count = 0

    for line_no, row in load_jsonl(args.jsonl, args.start_line, selected_lines):
        if args.limit is not None and executed_count >= args.limit:
            break

        problem = str(row.get("problem", "")).strip()
        subject = str(row.get("subject", "unknown")).strip() or "unknown"
        task_group_id = str(row.get("task_group_id", f"line{line_no}")).strip() or f"line{line_no}"
        raw_task_id = str(row.get(args.task_id_field, task_group_id)).strip() or task_group_id
        task_id = sanitize_name(raw_task_id)
        raw_task_type = str(row.get(args.task_type_field, args.default_task_type)).strip()
        task_type = raw_task_type or args.default_task_type

        if not problem:
            pre_results.append(
                TaskResult(
                    line_no=line_no,
                    run_dir=args.base_run_dir,
                    log_dir_name="",
                    exit_code=0,
                    skipped=True,
                    reason="empty problem",
                )
            )
            print(f"[SKIP] line={line_no}: empty `problem`", flush=True)
            continue

        raw_log_dir_name = f"{subject}{task_group_id}"
        safe_log_dir_name = sanitize_name(raw_log_dir_name)
        row_run_dir = args.base_run_dir / f"{line_no:04d}_{safe_log_dir_name}"
        row_run_dir = unique_path(row_run_dir)
        row_run_dir.mkdir(parents=True, exist_ok=True)

        task_file = row_run_dir / "task.md"
        task_markdown = build_task_markdown(
            problem=problem,
            task_id=task_id,
            task_type=task_type,
            extra_meta={
                "line_no": str(line_no),
                "subject": subject,
                "task_group_id": task_group_id,
            },
        )
        task_file.write_text(task_markdown, encoding="utf-8")
        task_json_file = row_run_dir / "task.json"
        task_json_file.write_text(
            json.dumps([{"id": task_id, "description": task_markdown}], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = [
            args.python,
            str(run_py),
            "--agent",
            args.agent,
            "--task-file",
            str(task_json_file),
            "--run-dir",
            str(row_run_dir),
        ]
        if args.config is not None:
            cmd.extend(["--config", str(args.config)])

        jobs.append(
            TaskJob(
                line_no=line_no,
                run_dir=row_run_dir,
                log_dir_name=safe_log_dir_name,
                cmd=cmd,
            )
        )
        executed_count += 1

    return jobs, pre_results


def run_single_job(job: TaskJob, project_root: Path) -> TaskResult:
    print(f"[RUN ] line={job.line_no} run_dir={job.run_dir}", flush=True)
    exit_code = subprocess.run(job.cmd, cwd=project_root).returncode

    logs_dir = job.run_dir / "logs"
    renamed_log_dir = job.run_dir / job.log_dir_name
    if logs_dir.exists():
        renamed_log_dir = unique_path(renamed_log_dir)
        logs_dir.rename(renamed_log_dir)
        print(f"[LOG ] line={job.line_no} renamed logs -> {renamed_log_dir.name}", flush=True)
    else:
        print(f"[WARN] line={job.line_no} logs directory not found, skip renaming", flush=True)

    return TaskResult(
        line_no=job.line_no,
        run_dir=job.run_dir,
        log_dir_name=renamed_log_dir.name if renamed_log_dir.exists() else "",
        exit_code=exit_code,
    )


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    run_py = project_root / "run.py"

    if not args.jsonl.exists():
        print(f"[ERROR] JSONL file not found: {args.jsonl}")
        return 1
    if not run_py.exists():
        print(f"[ERROR] run.py not found: {run_py}")
        return 1

    if args.workers < 1:
        print("[ERROR] --workers must be >= 1")
        return 1

    try:
        selected_lines = parse_line_selector(args.lines)
    except ValueError as exc:
        print(f"[ERROR] invalid --lines value: {exc}")
        return 1

    if selected_lines is not None:
        print(f"[INFO] selected lines: {sorted(selected_lines)}", flush=True)

    args.base_run_dir.mkdir(parents=True, exist_ok=True)
    jobs, pre_results = build_jobs(args, run_py, selected_lines)
    results: list[TaskResult] = list(pre_results)

    if not jobs:
        print("[DONE] no runnable rows found.", flush=True)
        return 0

    if args.workers == 1:
        for job in jobs:
            result = run_single_job(job, project_root)
            results.append(result)
            if result.exit_code != 0 and args.stop_on_error:
                print("[STOP] stop on first error.", flush=True)
                break
    else:
        print(f"[INFO] parallel mode enabled, workers={args.workers}", flush=True)
        future_to_job = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for job in jobs:
                future = executor.submit(run_single_job, job, project_root)
                future_to_job[future] = job

            stop_triggered = False
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                if args.stop_on_error and result.exit_code != 0 and not stop_triggered:
                    stop_triggered = True
                    print("[STOP] stop-on-error triggered, cancelling pending tasks.", flush=True)
                    for pending_future in future_to_job:
                        pending_future.cancel()

    results.sort(key=lambda r: r.line_no)
    success_count = sum(1 for r in results if not r.skipped and r.exit_code == 0)
    fail_count = sum(1 for r in results if not r.skipped and r.exit_code != 0)
    skip_count = sum(1 for r in results if r.skipped)

    print("=" * 60)
    print(f"[DONE] base_run_dir: {args.base_run_dir}")
    print(f"[DONE] success={success_count}, fail={fail_count}, skipped={skip_count}")
    if fail_count > 0:
        failed_lines = [str(r.line_no) for r in results if not r.skipped and r.exit_code != 0]
        print(f"[DONE] failed lines: {', '.join(failed_lines)}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
