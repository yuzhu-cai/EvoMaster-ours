#!/usr/bin/env python3
"""Merge solution files back into the FrontierScience dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_JSONL = Path("playground/frontierscience/test/test.jsonl")
TASK_DIR_PATTERN = re.compile(r"^(\d{4})_(physics|chemistry|biology)([0-9a-f\-]+)$")
SOLUTION_FILES = {
    "solution": "solution.md",
    "solution_refined": "solution_refined.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated solutions into JSONL rows.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Source dataset JSONL.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Run directory produced by run.py.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <runs-dir>/eval")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append(row)
    return rows


def build_task_map(runs_dir: Path) -> dict[str, Path]:
    task_map: dict[str, Path] = {}
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or child.name == "eval":
            continue
        match = TASK_DIR_PATTERN.match(child.name)
        if not match:
            continue
        task_map[match.group(3)] = child
    return task_map


def read_solution(task_dir: Path, task_id: str, field: str) -> str:
    filename = SOLUTION_FILES[field]
    candidates = [
        task_dir / filename,
        task_dir / "workspaces" / task_id / filename,
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.jsonl)
    task_map = build_task_map(args.runs_dir)
    output_dir = args.output_dir or (args.runs_dir / "eval")

    merged_solution: list[dict] = []
    merged_refined: list[dict] = []

    for row in rows:
        task_id = str(row.get("task_group_id", "")).strip()
        task_dir = task_map.get(task_id)
        base = dict(row)
        refined = dict(row)
        base["solution"] = read_solution(task_dir, task_id, "solution") if task_dir else ""
        refined["solution_refined"] = read_solution(task_dir, task_id, "solution_refined") if task_dir else ""
        merged_solution.append(base)
        merged_refined.append(refined)

    solution_path = output_dir / "solution.jsonl"
    refined_path = output_dir / "solution_refined.jsonl"
    write_jsonl(solution_path, merged_solution)
    write_jsonl(refined_path, merged_refined)

    print(f"[DONE] wrote {solution_path}")
    print(f"[DONE] wrote {refined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
