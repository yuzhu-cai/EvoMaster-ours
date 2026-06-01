#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OUT_DIR = Path("runs/evomaster4paperbench")
ORGANIZED_CATEGORIES = (
    "generation/full",
    "generation/targeted",
    "grades/final",
    "grades/targeted",
    "grades/bestof",
    "grades/live",
    "grades/live-delta",
    "plans",
    "misc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tidy symlink index for EvoMaster PaperBench Code-Dev runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--prefix",
        default="paperbench_codedev",
        help="Only organize run directories with this prefix.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def classify(path: Path) -> str:
    name = path.name
    if name == "paperbench_codedev_launch_logs":
        return "launch-logs"
    if name == "paperbench_codedev_model_results":
        return "summaries"
    if "gap_plan" in name:
        return "plans"

    has_summary = (path / "summary.json").exists()
    has_manifest = (path / "manifest.json").exists()
    has_generation_marker = (path / "EVOMASTER_RUN_FINISHED.json").exists()
    has_workspaces = (path / "workspaces").is_dir()
    has_trajectories = (path / "trajectories").is_dir()

    if has_summary or has_manifest or "grade" in name or "crs_gpt55" in name:
        if "live20_delta" in name:
            return "grades/live-delta"
        if "live20" in name:
            return "grades/live"
        if "bestof" in name or "portfolio" in name or "best" in name:
            return "grades/bestof"
        if "targeted" in name:
            return "grades/targeted"
        return "grades/final"

    if has_generation_marker or has_workspaces or has_trajectories:
        if "targeted" in name or "rice_rerun" in name:
            return "generation/targeted"
        return "generation/full"

    return "misc"


def summarize(path: Path, category: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": path.name,
        "category": category,
        "source": str(path),
        "link": "",
        "n": "",
        "mean_score": "",
        "status": "",
        "finished_at": "",
        "submission_tars": "",
        "grader_outputs": "",
    }

    finished = read_json(path / "EVOMASTER_RUN_FINISHED.json")
    if finished:
        row["status"] = finished.get("status", "")
        row["finished_at"] = finished.get("finished_at", "")

    summary = read_json(path / "summary.json")
    if summary:
        row["n"] = summary.get("n", "")
        mean_score = summary.get("mean_score", "")
        row["mean_score"] = mean_score if mean_score is not None else ""

    if (path / "workspaces").is_dir():
        row["submission_tars"] = len(list(path.glob("workspaces/*/artifacts/submission.tar.gz")))
    row["grader_outputs"] = len(list(path.glob("*/grader_output.json")))
    return row


def write_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
    if dst.is_symlink():
        if os.readlink(dst) == rel_src:
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {dst}")
    dst.symlink_to(rel_src)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "name",
        "category",
        "source",
        "link",
        "n",
        "mean_score",
        "status",
        "finished_at",
        "submission_tars",
        "grader_outputs",
    ]
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1,
        "description": "Organized symlink view of EvoMaster PaperBench Code-Dev runs.",
        "runs": rows,
    }
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    lines = [
        "# EvoMaster PaperBench Runs",
        "",
        "This directory stores new EvoMaster PaperBench Code-Dev outputs and also exposes an",
        "organized symlink view over the legacy `runs/paperbench_codedev*` directories.",
        "Legacy source directories are not moved, so older scripts and configs keep working.",
        "",
        "## Layout",
        "",
        "| category | count | purpose |",
        "|---|---:|---|",
    ]
    purposes = {
        "generation/full": "full split generation runs",
        "generation/targeted": "targeted or single-paper generation runs",
        "grades/final": "official or final grade runs",
        "grades/targeted": "targeted grade runs",
        "grades/bestof": "selected or best-of grade runs",
        "grades/live": "live grade snapshots",
        "grades/live-delta": "incremental live grade snapshots",
        "launch-logs": "launcher logs and generated command files",
        "summaries": "aggregate model result tables",
        "plans": "gap-selection plans",
        "misc": "uncategorized PaperBench Code-Dev artifacts",
    }
    for category in sorted(counts):
        lines.append(f"| `{category}` | {counts[category]} | {purposes.get(category, '')} |")
    lines.extend(
        [
            "",
            "## Index Files",
            "",
            "- `index.csv`: one row per organized run directory.",
            "- `index.json`: structured copy of the same index.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python playground/paperbench_codedev_agent/scripts/organize_runs.py",
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir
    out_dir = args.out_dir
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or path.name == out_dir.name:
            continue
        if not path.name.startswith(args.prefix):
            continue
        category = classify(path)
        row = summarize(path, category)
        link = out_dir / category / path.name
        write_link(path, link)
        row["link"] = str(link)
        rows.append(row)
        seen.add(path.resolve())

    for category in ORGANIZED_CATEGORIES:
        category_dir = out_dir / category
        if not category_dir.is_dir():
            continue
        for path in sorted(category_dir.iterdir()):
            if path.is_symlink() or not path.is_dir() or not path.name.startswith(args.prefix):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            row = summarize(path, category)
            row["source"] = str(path)
            row["link"] = str(path)
            rows.append(row)
            seen.add(resolved)

    write_outputs(out_dir, rows)
    print(f"organized {len(rows)} runs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
