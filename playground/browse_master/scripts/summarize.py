#!/usr/bin/env python3
"""Summarize evaluation results."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation scores")
    parser.add_argument("--jsonl", required=True, help="Input eval jsonl file")
    parser.add_argument("--result", required=True, help="Output result JSON file")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    correct = sum(1 for r in records if r.get("score") == 1)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": f"{accuracy * 100:.2f}%",
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary: {correct}/{total} correct ({accuracy * 100:.2f}%)")
    print(f"Result saved to: {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
