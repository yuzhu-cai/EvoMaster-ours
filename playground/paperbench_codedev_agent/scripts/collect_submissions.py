#!/usr/bin/env python3
"""Collect latest live EvoMaster PaperBench submissions for grading.

This intentionally copies from each run's live workspaces/<paper>/submission
directory instead of trusting an older submission.tar.gz. A completion marker is
required by default so a grading manifest cannot silently use mid-run artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="EvoMaster generation run directory.")
    parser.add_argument("--grade-run", type=Path, required=True, help="Output directory for copied workspaces and manifest.json.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow papers without EVOMASTER_COMPLETE.json.")
    parser.add_argument("--paper-id", action="append", default=[], help="Optional paper id filter; can be repeated.")
    return parser.parse_args()


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
    wanted = set(args.paper_id or [])
    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for run_dir in args.run_dir:
        workspace_root = run_dir / "workspaces"
        if not workspace_root.exists():
            skipped.append({"run_dir": str(run_dir), "reason": "missing_workspaces"})
            continue
        for paper_dir in sorted(p for p in workspace_root.iterdir() if p.is_dir()):
            paper_id = paper_dir.name
            if wanted and paper_id not in wanted:
                continue
            submission = paper_dir / "submission"
            marker = paper_dir / "artifacts" / "EVOMASTER_COMPLETE.json"
            if not submission.exists():
                skipped.append({"paper_id": paper_id, "run_dir": str(run_dir), "reason": "missing_submission"})
                continue
            if not args.allow_incomplete and not marker.exists():
                skipped.append({"paper_id": paper_id, "run_dir": str(run_dir), "reason": "missing_completion_marker"})
                continue
            dst = args.grade_run / "workspaces" / paper_id / "submission"
            copy_submission(submission, dst)
            rows.append(
                {
                    "paper_id": paper_id,
                    "submission": str(dst),
                    "source_run": str(run_dir),
                    "source_submission": str(submission),
                    "completion_marker": str(marker) if marker.exists() else "",
                }
            )

    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        deduped[row["paper_id"]] = row
    manifest = [deduped[key] for key in sorted(deduped)]
    args.grade_run.mkdir(parents=True, exist_ok=True)
    (args.grade_run / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (args.grade_run / "collect_status.json").write_text(
        json.dumps({"n": len(manifest), "skipped": skipped}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"grade_run": str(args.grade_run), "n": len(manifest), "skipped": len(skipped)}, indent=2))
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
