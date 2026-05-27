#!/usr/bin/env python3
"""Summarize PaperBench Code-Dev grader outputs in a run directory."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def score_from_file(path: Path) -> tuple[str, float] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    score = data.get("score")
    if score is None:
        score = data.get("judge_output", {}).get("score")
    if score is None:
        score = data.get("paperbench_result", {}).get("judge_output", {}).get("score")
    if score is None:
        return None
    paper_id = str(data.get("paper_id") or path.parent.parent.name)
    if paper_id == "grade" and path.parent.parent.name:
        paper_id = path.parent.parent.name
    return paper_id, float(score)


def main() -> int:
    args = parse_args()
    rows: list[tuple[str, float, Path]] = []
    for path in sorted(args.run_dir.rglob("grader_output.json")) + sorted(args.run_dir.rglob("grade.json")):
        item = score_from_file(path)
        if item is None:
            continue
        paper_id, score = item
        rows.append((paper_id, score, path))

    if not rows:
        print(f"No grade outputs found under {args.run_dir}")
        return 1

    for paper_id, score, path in rows:
        print(f"{paper_id}\t{score:.6f}\t{path}")
    print(f"count\t{len(rows)}")
    print(f"average\t{statistics.mean(score for _, score, _ in rows):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
