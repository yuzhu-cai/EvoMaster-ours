#!/usr/bin/env python3
"""Merge dataset answers with agent solutions into a single jsonl file."""

import argparse
import json
import re
import sys
from pathlib import Path


TASK_DIR_RE = re.compile(r"task_(\d+)$")


def get_question_text(item: dict) -> str:
    """Return the normalized question text from a dataset item."""
    return (item.get("question") or item.get("prompt") or "").strip()


def get_answer_text(item: dict) -> str:
    """Return the normalized answer text from a dataset item."""
    return (item.get("answer") or item.get("Answer") or "").strip()


def get_run_ids(run_dir: Path) -> list[int]:
    """Collect task ids that were actually run under the run directory."""
    ids = []
    for task_dir in sorted(run_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        match = TASK_DIR_RE.fullmatch(task_dir.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def main():
    parser = argparse.ArgumentParser(description="Merge dataset and solutions")
    parser.add_argument("--json", required=True, help="Path to dataset JSON file")
    parser.add_argument("--run-dir", required=True, help="Root run directory containing task_xxxx folders")
    parser.add_argument("--output-dir", required=True, help="Output directory for merge.jsonl")
    args = parser.parse_args()

    json_path = Path(args.json)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = {int(item["id"]): item for item in data if "id" in item}
    run_ids = get_run_ids(run_dir)

    output_file = output_dir / "merge.jsonl"
    merged_count = 0
    missing_count = 0
    skipped_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry_id in run_ids:
            item = dataset.get(entry_id)
            if item is None:
                skipped_count += 1
                continue
            task_name = f"task_{entry_id:04d}"
            solution_file = run_dir / task_name / "solution.txt"

            solution = ""
            if solution_file.exists():
                with open(solution_file, "r", encoding="utf-8") as f:
                    solution = f.read().strip()
                merged_count += 1
            else:
                missing_count += 1

            record = {
                "id": entry_id,
                "question": get_question_text(item),
                "answer": get_answer_text(item),
                "solution": solution,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Merge complete: {merged_count} merged, {missing_count} missing, "
        f"{skipped_count} skipped"
    )
    print(f"Output: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
