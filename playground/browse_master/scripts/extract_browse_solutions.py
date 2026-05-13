#!/usr/bin/env python3
"""Extract final answers from browse run logs into solution.txt files."""

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path


FINAL_ANSWER_RE = re.compile(r"Agent final answer:\s*(.*?)\s*$")


def read_last_lines(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return list(deque(fh, maxlen=limit))


def extract_solution(path: Path, tail_lines: int) -> str | None:
    lines = read_last_lines(path, tail_lines)
    for line in reversed(lines):
        match = FINAL_ANSWER_RE.search(line)
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 'Agent final answer' from each task log under a browse run "
            "directory and write it to task_xxxx/solution.txt."
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/browse_dsv4pro",
        help="Browse run directory that contains task_xxxx subdirectories.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=20,
        help="Only inspect the last N lines of each task_0.log (default: 20).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    log_paths = sorted(run_dir.glob("task_*/logs/task_0.log"))
    if not log_paths:
        raise SystemExit(f"No task logs found under: {run_dir}")

    written = 0
    missing: list[Path] = []

    for log_path in log_paths:
        solution = extract_solution(log_path, args.tail_lines)
        if solution is None:
            missing.append(log_path)
            continue

        task_dir = log_path.parent.parent
        output_path = task_dir / "solution.txt"
        output_path.write_text(f"{solution}\n", encoding="utf-8")
        written += 1

    print(f"Wrote {written} solution file(s) under {run_dir}.")
    if missing:
        print("No final answer found in:")
        for path in missing:
            print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
