#!/usr/bin/env python3
"""Merge dataset answers with agent solutions into a single jsonl file."""

import argparse
import json
import sys
from pathlib import Path


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

    output_file = output_dir / "merge.jsonl"
    merged_count = 0
    missing_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry_id in sorted(dataset.keys()):
            item = dataset[entry_id]
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
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "solution": solution,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Merge complete: {merged_count} merged, {missing_count} missing")
    print(f"Output: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
