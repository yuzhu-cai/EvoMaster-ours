#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("grade_run", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    for output in sorted(args.grade_run.glob("*/grader_output.json")):
        obj = json.loads(output.read_text())
        judge = obj.get("judge_output") or {}
        rows.append(
            {
                "paper_id": obj.get("paper_id") or output.parent.name,
                "score": float(obj.get("score", judge.get("score", 0.0))),
                "num_leaf_nodes": judge.get("num_leaf_nodes"),
                "num_invalid_leaf_nodes": judge.get("num_invalid_leaf_nodes"),
                "path": output.as_posix(),
            }
        )

    rows.sort(key=lambda row: row["paper_id"])
    mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    summary = {
        "grade_run": args.grade_run.as_posix(),
        "n": len(rows),
        "mean_score": mean,
        "papers": rows,
    }
    path = args.grade_run / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
