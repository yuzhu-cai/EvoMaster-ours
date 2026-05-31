#!/usr/bin/env python3
"""Compare an EvoMaster CRS regrade summary against the Codex GPT-5.4 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CODEX = Path(
    "runs/codex4paperbench/codex_gpt54_regen_regrade_crs_gpt55_responses_medium_c4x40_20260527T071410Z/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evomaster-summary", type=Path, required=True)
    parser.add_argument("--codex-summary", type=Path, default=DEFAULT_CODEX)
    parser.add_argument("--expected-n", type=int, default=20, help="Require this many common papers before declaring a win; 0 disables.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def paper_scores(summary: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in summary.get("papers", []):
        score = row.get("score")
        if isinstance(score, (int, float)):
            scores[str(row.get("paper_id"))] = float(score)
    return scores


def main() -> int:
    args = parse_args()
    evomaster = json.loads(args.evomaster_summary.read_text(encoding="utf-8"))
    codex = json.loads(args.codex_summary.read_text(encoding="utf-8"))
    evo_scores = paper_scores(evomaster)
    codex_scores = paper_scores(codex)
    common = sorted(set(evo_scores) & set(codex_scores))
    rows = [
        {
            "paper_id": paper,
            "evomaster": evo_scores[paper],
            "codex": codex_scores[paper],
            "delta": evo_scores[paper] - codex_scores[paper],
        }
        for paper in common
    ]
    evo_mean = sum(evo_scores[p] for p in common) / len(common) if common else None
    codex_mean = sum(codex_scores[p] for p in common) / len(common) if common else None
    enough_papers = args.expected_n <= 0 or len(common) >= args.expected_n
    result = {
        "evomaster_summary": str(args.evomaster_summary),
        "codex_summary": str(args.codex_summary),
        "n_common": len(common),
        "expected_n": args.expected_n,
        "evomaster_mean": evo_mean,
        "codex_mean": codex_mean,
        "delta_mean": None if evo_mean is None or codex_mean is None else evo_mean - codex_mean,
        "evomaster_beats_codex": bool(enough_papers and evo_mean is not None and codex_mean is not None and evo_mean > codex_mean),
        "papers": sorted(rows, key=lambda row: row["delta"]),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["evomaster_beats_codex"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
