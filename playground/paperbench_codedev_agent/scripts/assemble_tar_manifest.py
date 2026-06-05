#!/usr/bin/env python3
"""Assemble a grade manifest from completed base papers and repair tarballs."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--repair-run", type=Path, action="append", default=[])
    parser.add_argument("--grade-run", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=20)
    return parser.parse_args()


def marker_complete(paper_dir: Path) -> tuple[bool, str]:
    marker = paper_dir / "artifacts" / "EVOMASTER_COMPLETE.json"
    tar_path = paper_dir / "artifacts" / "submission.tar.gz"
    if not marker.exists():
        return False, "missing_completion_marker"
    if not tar_path.exists():
        return False, "missing_submission_tar"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"bad_completion_marker:{exc}"
    if payload.get("status") != "completed":
        return False, f"marker_status_{payload.get('status') or 'missing'}"
    artifact_status = payload.get("artifact_status") or {}
    if artifact_status and not artifact_status.get("ok", False):
        return False, "artifact_status_not_ok"
    return True, "ok"


def extract_submission_tar(tar_path: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp)
    src = tmp / "submission"
    if not src.exists():
        entries = [p for p in tmp.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            src = entries[0]
        else:
            src = tmp
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    args = parse_args()
    base_workspaces = args.base_run / "workspaces"
    if not base_workspaces.exists():
        raise SystemExit(f"missing base workspaces: {base_workspaces}")

    sources: dict[str, dict[str, str]] = {}
    missing: dict[str, list[str]] = {}
    base_failed: list[str] = []

    for paper_dir in sorted(p for p in base_workspaces.iterdir() if p.is_dir()):
        paper = paper_dir.name
        ok, reason = marker_complete(paper_dir)
        if ok:
            sources[paper] = {
                "paper_id": paper,
                "source_run": str(args.base_run),
                "source_tar": str(paper_dir / "artifacts" / "submission.tar.gz"),
                "source_kind": "base_completed",
            }
        else:
            base_failed.append(paper)
            missing.setdefault(paper, []).append(f"base:{reason}")

    for run_dir in args.repair_run:
        workspace_root = run_dir / "workspaces"
        if not workspace_root.exists():
            continue
        for paper_dir in sorted(p for p in workspace_root.iterdir() if p.is_dir()):
            paper = paper_dir.name
            if paper not in base_failed:
                continue
            tar_path = paper_dir / "artifacts" / "submission.tar.gz"
            if not tar_path.exists():
                missing.setdefault(paper, []).append(f"{run_dir}:missing_repair_tar")
                continue
            sources[paper] = {
                "paper_id": paper,
                "source_run": str(run_dir),
                "source_tar": str(tar_path),
                "source_kind": "repair_tar",
            }

    args.grade_run.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for paper in sorted(sources):
        src = sources[paper]
        dst = args.grade_run / "workspaces" / paper / "submission"
        extract_submission_tar(Path(src["source_tar"]), dst)
        row = {
            "paper_id": paper,
            "submission": str(dst),
            "source_run": src["source_run"],
            "source_tar": src["source_tar"],
            "source_kind": src["source_kind"],
        }
        rows.append(row)

    unresolved = [paper for paper in sorted(base_failed) if paper not in sources]
    (args.grade_run / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.grade_run / "collect_status.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "expected": args.expected,
                "base_failed": sorted(base_failed),
                "unresolved": unresolved,
                "missing": missing,
                "repair_runs": [str(p) for p in args.repair_run],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"grade_run": str(args.grade_run), "n": len(rows), "unresolved": unresolved}, indent=2))
    return 0 if len(rows) == args.expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
