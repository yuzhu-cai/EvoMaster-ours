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
    ("draft", "solution_scored"),
    ("final", "solution_refined_scored"),
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


def load_trajectory_steps(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def find_reflect_step(steps: list[dict]) -> int | None:
    for item in steps:
        if not isinstance(item, dict):
            continue
        trajectory = item.get("trajectory") or {}
        dialogs = trajectory.get("dialogs") or []
        for dialog in dialogs:
            messages = dialog.get("messages") or []
            for message in messages:
                tool_calls = message.get("tool_calls") or []
                for call in tool_calls:
                    function = call.get("function") or {}
                    if function.get("name") == "reflect_answer":
                        step_no = item.get("steps")
                        if isinstance(step_no, int):
                            return step_no
    return None


def extract_step_stats(path: Path) -> dict[str, int] | None:
    steps = load_trajectory_steps(path)
    if not steps:
        return None
    total_steps = 0
    last = steps[-1]
    if isinstance(last, dict) and isinstance(last.get("steps"), int):
        total_steps = last["steps"]
    else:
        total_steps = len(steps)
    reflect_step = find_reflect_step(steps)
    return {
        "total_steps": total_steps,
        "pre_reflect_steps": reflect_step - 1 if reflect_step else total_steps,
        "reflect_extra_steps": total_steps - reflect_step + 1 if reflect_step else 0,
        "reflect_called": 1 if reflect_step else 0,
    }


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
    overall = {
        "draft": [],
        "final": [],
        "reflect_extra": [],
        "reflect_called": [],
    }
    by_subject = {
        key: defaultdict(list)
        for key in overall
    }

    for task_dir, subject, task_id in iter_task_dirs(runs_dir):
        path = task_dir / "trajectories" / task_id / "trajectory.json"
        if not path.exists():
            continue
        stats = extract_step_stats(path)
        if not stats:
            continue
        overall["draft"].append(stats["pre_reflect_steps"])
        overall["final"].append(stats["total_steps"])
        overall["reflect_extra"].append(stats["reflect_extra_steps"])
        overall["reflect_called"].append(stats["reflect_called"])
        by_subject["draft"][subject].append(stats["pre_reflect_steps"])
        by_subject["final"][subject].append(stats["total_steps"])
        by_subject["reflect_extra"][subject].append(stats["reflect_extra_steps"])
        by_subject["reflect_called"][subject].append(stats["reflect_called"])

    return {"overall": overall, "by_subject": by_subject}


def summarize_eval(eval_repeat_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode, prefix in MODES:
        paths = sorted(eval_repeat_dir.glob(f"{prefix}*.jsonl"))
        overall_scores: list[float] = []
        per_subject_scores: dict[str, list[float]] = defaultdict(list)
        for path in paths:
            rows = load_scored_rows(path)
            if not rows:
                continue
            score_key = "solution_refined" if mode == "final" else "solution"
            del score_key
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
    print(
        f"[draft] avg_steps={format_float(safe_mean(step_stats['overall']['draft']))} "
        f"count={len(step_stats['overall']['draft'])}"
    )
    print(
        f"[final] avg_steps={format_float(safe_mean(step_stats['overall']['final']))} "
        f"count={len(step_stats['overall']['final'])}"
    )
    print(
        f"[reflect_extra] avg_steps={format_float(safe_mean(step_stats['overall']['reflect_extra']))} "
        f"reflect_call_rate={format_float(safe_mean(step_stats['overall']['reflect_called']))}"
    )
    for subject in SUBJECTS:
        print(
            f"  {subject}: draft={format_float(safe_mean(step_stats['by_subject']['draft'][subject]))} "
            f"final={format_float(safe_mean(step_stats['by_subject']['final'][subject]))} "
            f"reflect_extra={format_float(safe_mean(step_stats['by_subject']['reflect_extra'][subject]))}"
        )
    for mode, _ in MODES:
        if mode in eval_stats:
            print(
                f"  eval[{mode}] overall={format_float(eval_stats[mode]['overall_acc'])} "
                f"physics={format_float(eval_stats[mode]['physics_acc'])} "
                f"chemistry={format_float(eval_stats[mode]['chemistry_acc'])} "
                f"biology={format_float(eval_stats[mode]['biology_acc'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
