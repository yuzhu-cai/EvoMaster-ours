#!/usr/bin/env python3
"""Select papers where EvoMaster still trails the Codex GPT-5.4 baseline."""

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
    parser.add_argument("--max-papers", type=int, default=8)
    parser.add_argument("--min-gap", type=float, default=0.01, help="Require codex_score - evomaster_score >= this value.")
    parser.add_argument("--out", type=Path, help="Optional one-paper-id-per-line output file.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON detail output file.")
    return parser.parse_args()


def scores(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in summary.get("papers", []):
        score = row.get("score")
        if isinstance(score, (int, float)):
            out[str(row.get("paper_id"))] = float(score)
    return out


def main() -> int:
    args = parse_args()
    evomaster = scores(json.loads(args.evomaster_summary.read_text(encoding="utf-8")))
    codex = scores(json.loads(args.codex_summary.read_text(encoding="utf-8")))
    rows = []
    for paper in sorted(set(codex) & set(evomaster)):
        gap = codex[paper] - evomaster[paper]
        if gap >= args.min_gap:
            rows.append({"paper_id": paper, "gap": gap, "evomaster": evomaster[paper], "codex": codex[paper]})
    rows.sort(key=lambda row: row["gap"], reverse=True)
    if args.max_papers > 0:
        rows = rows[: args.max_papers]
    paper_ids = [row["paper_id"] for row in rows]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(paper_ids) + ("\n" if paper_ids else ""), encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"n": len(rows), "papers": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(",".join(paper_ids))
    return 0 if paper_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
