#!/usr/bin/env python3
"""Summarize FrontierScience scores and trajectory steps."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

TASK_DIR_PATTERN = re.compile(r"^(\d{4})_(physics|chemistry|biology)([0-9a-f\-]+)$")
SUBJECTS = ("physics", "chemistry", "biology")
MODES = (
    ("without_reflect", "solution_scored", False),
    ("with_reflect", "solution_refined_scored", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FrontierScience eval outputs.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Run directory to summarize.")
    parser.add_argument("--eval-repeat-dir", type=Path, default=None, help="Directory containing repeated eval outputs.")
    return parser.parse_args()


def iter_task_dirs(runs_dir: Path):
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or child.name == "eval":
            continue
        match = TASK_DIR_PATTERN.match(child.name)
        if match:
            yield child, match.group(2), match.group(3)


def extract_steps_from_trajectory(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, list):
        if not data:
            return 0
        last_item = data[-1]
        if isinstance(last_item, dict) and isinstance(last_item.get("steps"), int):
            return last_item["steps"]
        return len(data)

    if isinstance(data, dict):
        if isinstance(data.get("steps"), int):
            return data["steps"]
        for key in ("trajectory", "trajectories", "dialogs"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)

    return None


def safe_mean(values: list[float]) -> float:
    return math.nan if not values else sum(values) / len(values)


def format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def load_scored_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_steps(runs_dir: Path) -> dict[str, dict[str, list[int]]]:
    overall = {mode: [] for mode, _, _ in MODES}
    by_subject = {mode: defaultdict(list) for mode, _, _ in MODES}

    for task_dir, subject, task_id in iter_task_dirs(runs_dir):
        for mode, _, is_reflect in MODES:
            name = f"{task_id}_reflect" if is_reflect else task_id
            path = task_dir / "trajectories" / name / "trajectory.json"
            if not path.exists():
                continue
            steps = extract_steps_from_trajectory(path)
            if steps is None:
                continue
            overall[mode].append(steps)
            by_subject[mode][subject].append(steps)

    return {"overall": overall, "by_subject": by_subject}


def summarize_eval(eval_repeat_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode, prefix, _ in MODES:
        paths = sorted(eval_repeat_dir.glob(f"{prefix}*.jsonl"))
        overall_scores: list[float] = []
        per_subject_scores: dict[str, list[float]] = defaultdict(list)
        for path in paths:
            rows = load_scored_rows(path)
            if not rows:
                continue
            overall_scores.append(sum(int(row.get("score", 0)) for row in rows) / len(rows))
            for subject in SUBJECTS:
                subject_rows = [row for row in rows if row.get("subject") == subject]
                if subject_rows:
                    per_subject_scores[subject].append(
                        sum(int(row.get("score", 0)) for row in subject_rows) / len(subject_rows)
                    )
        results[mode] = {
            "overall_acc": safe_mean(overall_scores),
            **{f"{subject}_acc": safe_mean(per_subject_scores[subject]) for subject in SUBJECTS},
        }
    return results


def main() -> int:
    args = parse_args()
    eval_repeat_dir = args.eval_repeat_dir or (args.runs_dir / "eval1")

    step_stats = summarize_steps(args.runs_dir)
    eval_stats = summarize_eval(eval_repeat_dir) if eval_repeat_dir.exists() else {}

    print(f"runs_dir={args.runs_dir}")
    print(f"eval_repeat_dir={eval_repeat_dir}")
    for mode, _, _ in MODES:
        print(
            f"[{mode}] avg_steps={format_float(safe_mean(step_stats['overall'][mode]))} "
            f"count={len(step_stats['overall'][mode])}"
        )
        for subject in SUBJECTS:
            print(
                f"  {subject}: avg_steps={format_float(safe_mean(step_stats['by_subject'][mode][subject]))} "
                f"count={len(step_stats['by_subject'][mode][subject])}"
            )
        if mode in eval_stats:
            print(
                f"  acc_overall={format_float(eval_stats[mode]['overall_acc'])} "
                f"physics={format_float(eval_stats[mode]['physics_acc'])} "
                f"chemistry={format_float(eval_stats[mode]['chemistry_acc'])} "
                f"biology={format_float(eval_stats[mode]['biology_acc'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
