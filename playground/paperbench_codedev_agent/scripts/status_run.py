#!/usr/bin/env python3
"""Summarize an EvoMaster PaperBench Code-Dev generation or grade run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CODE_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".md", ".cfg", ".ini"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Generation run directory with workspaces/<paper_id>.")
    parser.add_argument("--grade-run", type=Path, help="Grade run directory with summary.json or grader outputs.")
    return parser.parse_args()


def load_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last if isinstance(last, dict) else None


def live_counts(submission: Path) -> dict[str, int]:
    """Best-effort file counts even before the harness writes round audits."""
    if not submission.exists() or not submission.is_dir():
        return {}
    files = [p for p in submission.rglob("*") if p.is_file() and ".git" not in p.parts]
    loc = 0
    for path in files:
        if path.suffix not in CODE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        loc += sum(1 for line in text.splitlines() if line.strip())
    return {
        "all_files": len(files),
        "python_files": sum(1 for p in files if p.suffix == ".py"),
        "script_files": sum(1 for p in files if "scripts" in p.parts or p.suffix == ".sh"),
        "test_files": sum(1 for p in files if "tests" in p.parts or p.name.startswith("test_")),
        "config_files": sum(1 for p in files if p.suffix in {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"}),
        "nonblank_lines": loc,
    }


def generation_status(run_dir: Path) -> dict[str, Any]:
    workspaces = run_dir / "workspaces"
    papers = []
    if workspaces.exists():
        for paper_dir in sorted(p for p in workspaces.iterdir() if p.is_dir()):
            rounds = paper_dir / "logs" / "rounds.jsonl"
            last = load_last_jsonl(rounds)
            audit = (last or {}).get("audit") or {}
            quality = (last or {}).get("quality_gate") or {}
            submission = paper_dir / "submission"
            papers.append(
                {
                    "paper_id": paper_dir.name,
                    "complete": (paper_dir / "artifacts" / "EVOMASTER_COMPLETE.json").exists(),
                    "tar": (paper_dir / "artifacts" / "submission.tar.gz").exists(),
                    "round_records": sum(1 for _ in rounds.open(encoding="utf-8")) if rounds.exists() else 0,
                    "last_round": (last or {}).get("round"),
                    "trajectory_status": (last or {}).get("trajectory_status"),
                    "gate_passed": quality.get("passed"),
                    "missing_gate": quality.get("missing", [])[:5] if isinstance(quality.get("missing"), list) else quality.get("missing"),
                    "counts": audit.get("counts") or {},
                    "live_counts": live_counts(submission) if not audit.get("counts") else {},
                }
            )
    marker = run_dir / "EVOMASTER_RUN_FINISHED.json"
    return {
        "run_dir": str(run_dir),
        "run_finished": marker.exists(),
        "run_marker": json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None,
        "num_workspaces": len(papers),
        "num_complete": sum(1 for row in papers if row["complete"]),
        "num_tars": sum(1 for row in papers if row["tar"]),
        "papers": papers,
    }


def grade_status(grade_run: Path) -> dict[str, Any]:
    summary = grade_run / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        return {
            "grade_run": str(grade_run),
            "summary_exists": True,
            "n": data.get("n"),
            "mean_score": data.get("mean_score"),
            "papers": data.get("papers", []),
        }

    outputs = sorted(grade_run.glob("*/grader_output.json"))
    rows = []
    for path in outputs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        judge_output = data.get("judge_output") or {}
        rows.append({"paper_id": path.parent.name, "score": data.get("score", judge_output.get("score"))})
    values = [row["score"] for row in rows if isinstance(row.get("score"), (int, float))]
    return {
        "grade_run": str(grade_run),
        "summary_exists": False,
        "n": len(values),
        "mean_score": sum(values) / len(values) if values else None,
        "papers": rows,
    }


def main() -> int:
    args = parse_args()
    if not args.run_dir and not args.grade_run:
        raise SystemExit("provide --run-dir and/or --grade-run")
    payload: dict[str, Any] = {}
    if args.run_dir:
        payload["generation"] = generation_status(args.run_dir)
    if args.grade_run:
        payload["grade"] = grade_status(args.grade_run)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
