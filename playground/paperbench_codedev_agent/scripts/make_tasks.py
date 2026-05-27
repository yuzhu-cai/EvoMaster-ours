#!/usr/bin/env python3
"""Create EvoMaster task JSON for PaperBench Code-Dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("/data/yuzhu/Devs/third_party/frontier-evals/project/paperbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paperbench-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="debug", help="PaperBench split name, e.g. debug/dev/all.")
    parser.add_argument("--output", type=Path, required=True, help="Output task JSON path.")
    parser.add_argument("--resource-cycle", type=int, default=0, help="If >0, assign resource_index modulo this value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_path = args.paperbench_root / "experiments" / "splits" / f"{args.split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    paper_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    tasks = []
    for idx, paper_id in enumerate(paper_ids):
        paper_dir = args.paperbench_root / "data" / "papers" / paper_id
        task = {
            "id": paper_id,
            "description": json.dumps(
                {
                    "paper_id": paper_id,
                    "paper_dir": str(paper_dir),
                    "paperbench_root": str(args.paperbench_root),
                    "description": f"Reproduce PaperBench Code-Dev paper {paper_id}.",
                },
                ensure_ascii=False,
            ),
        }
        if args.resource_cycle > 0:
            task["resource_index"] = idx % args.resource_cycle
        tasks.append(task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

