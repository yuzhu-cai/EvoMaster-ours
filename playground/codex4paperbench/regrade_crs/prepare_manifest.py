#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def _paper_id(run_dir: Path) -> str:
    grade_path = run_dir / "grade.json"
    if grade_path.exists():
        try:
            grade = json.loads(grade_path.read_text())
            paper_id = (grade.get("paperbench_result") or {}).get("paper_id")
            if paper_id:
                return str(paper_id)
        except Exception:
            pass

    suffix = run_dir.name.rsplit("_", 1)[-1]
    if len(suffix) == 36 and suffix.count("-") == 4:
        return run_dir.name[: -(len(suffix) + 1)]
    return run_dir.name


def _latest_submission(run_dir: Path) -> Path | None:
    submissions = sorted(run_dir.glob("submissions/*/submission.tar.gz"))
    return submissions[-1] if submissions else None


def _run_dirs_by_paper(run_group: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for run_dir in sorted(run_group.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "grade.json").exists():
            continue
        submission = _latest_submission(run_dir)
        if submission is None:
            continue
        paper_id = _paper_id(run_dir)
        result[paper_id] = run_dir
    return result


def _extract_tar(tar_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-group", type=Path, required=True)
    parser.add_argument("--grade-run", type=Path, required=True)
    parser.add_argument(
        "--replace-run-group",
        action="append",
        default=[],
        type=Path,
        help="Run group whose papers override the source run group, e.g. an lbcs rerun.",
    )
    parser.add_argument("--include", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grade_run: Path = args.grade_run
    workspaces = grade_run / "workspaces"
    workspaces.mkdir(parents=True, exist_ok=True)

    by_paper = _run_dirs_by_paper(args.source_run_group)
    for replacement in args.replace_run_group:
        by_paper.update(_run_dirs_by_paper(replacement))

    paper_ids = sorted(args.include or by_paper)
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for paper_id in paper_ids:
        run_dir = by_paper.get(paper_id)
        if run_dir is None:
            missing.append(paper_id)
            continue
        source_tar = _latest_submission(run_dir)
        if source_tar is None:
            missing.append(paper_id)
            continue

        submission_dir = workspaces / paper_id / "submission"
        _extract_tar(source_tar, submission_dir)
        rows.append(
            {
                "paper_id": paper_id,
                "submission": submission_dir.as_posix(),
                "source_tar": source_tar.as_posix(),
                "source_run_dir": run_dir.as_posix(),
            }
        )

    if missing:
        raise RuntimeError(f"Missing submissions for: {', '.join(sorted(missing))}")

    manifest = grade_run / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"grade_run": grade_run.as_posix(), "n": len(rows), "manifest": manifest.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
