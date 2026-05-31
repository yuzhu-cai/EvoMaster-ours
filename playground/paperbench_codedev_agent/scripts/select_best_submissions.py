#!/usr/bin/env python3
"""Select the best scored submission per paper across multiple grade runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade-run", type=Path, action="append", required=True, help="A graded run with manifest.json and */grader_output.json.")
    parser.add_argument("--out-grade-run", type=Path, required=True, help="Output grade-run directory containing selected workspaces/manifest.")
    parser.add_argument("--expected-n", type=int, default=0, help="Fail if fewer than this many papers are selected.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads((path / "manifest.json").read_text())
    return {row["paper_id"]: row for row in rows}


def score_for(grade_run: Path, paper_id: str) -> float | None:
    output = grade_run / paper_id / "grader_output.json"
    if not output.exists():
        return None
    data = json.loads(output.read_text())
    judge_output = data.get("judge_output") or {}
    score = data.get("score", judge_output.get("score"))
    return float(score) if isinstance(score, (int, float)) else None


def copy_submission(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc", "*.pyo"),
    )


def main() -> int:
    args = parse_args()
    candidates: dict[str, list[dict[str, Any]]] = {}
    for grade_run in args.grade_run:
        if not (grade_run / "manifest.json").exists():
            raise SystemExit(f"missing manifest.json in grade run: {grade_run}")
        manifest = load_manifest(grade_run)
        for paper_id, row in manifest.items():
            score = score_for(grade_run, paper_id)
            if score is None:
                continue
            candidates.setdefault(paper_id, []).append(
                {
                    "paper_id": paper_id,
                    "score": score,
                    "grade_run": str(grade_run),
                    "submission": row["submission"],
                    "manifest_row": row,
                }
            )

    selected = []
    for paper_id, rows in sorted(candidates.items()):
        best = max(rows, key=lambda row: row["score"])
        dst = args.out_grade_run / "workspaces" / paper_id / "submission"
        copy_submission(Path(best["submission"]), dst)
        selected.append(
            {
                "paper_id": paper_id,
                "submission": str(dst),
                "source_grade_run": best["grade_run"],
                "source_submission": best["submission"],
                "selection_score": best["score"],
            }
        )

    if args.expected_n and len(selected) < args.expected_n:
        raise SystemExit(f"selected {len(selected)} papers, expected at least {args.expected_n}")

    values = [row["selection_score"] for row in selected]
    source_counts: dict[str, int] = {}
    for row in selected:
        source = row["source_grade_run"]
        source_counts[source] = source_counts.get(source, 0) + 1
    args.out_grade_run.mkdir(parents=True, exist_ok=True)
    (args.out_grade_run / "manifest.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "out_grade_run": str(args.out_grade_run),
        "n": len(selected),
        "mean_selection_score": sum(values) / len(values) if values else None,
        "source_counts": source_counts,
        "papers": selected,
    }
    (args.out_grade_run / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out_grade_run": str(args.out_grade_run), "n": len(selected), "mean_selection_score": summary["mean_selection_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
