#!/usr/bin/env python3
"""Search LANDAU technique skill YAML files by lightweight keyword matching."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def tokenize(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
        if len(t) >= 2
    }


def extract_line_value(text: str, key: str) -> str:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def score_doc(query: str, path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    q = tokenize(query)
    t = tokenize(raw)
    overlap = sorted(q & t)
    score = len(overlap) / max(1, len(q))

    skill_id = extract_line_value(raw, "skill_id") or path.parent.name
    category = extract_line_value(raw, "category")
    goal = extract_line_value(raw, "goal")
    preview = " ".join(raw.splitlines()[:20])

    return {
        "skill_id": skill_id,
        "category": category,
        "goal": goal,
        "path": str(path),
        "score": round(score, 4),
        "overlap_tokens": overlap,
        "preview": preview[:800],
    }


def default_technique_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "playground" / "phy_master" / "LANDAU" / "technique"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LANDAU technique files")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top_k", type=int, default=5, help="number of results")
    parser.add_argument("--technique_dir", default=str(default_technique_dir()), help="technique directory")
    args = parser.parse_args()

    base = Path(args.technique_dir)
    files = sorted(base.glob("**/skill.yaml"))

    ranked = sorted(
        [score_doc(args.query, p) for p in files],
        key=lambda x: x["score"],
        reverse=True,
    )

    top_k = max(1, args.top_k)
    results = ranked[:top_k]

    print(
        json.dumps(
            {
                "query": args.query,
                "technique_dir": str(base),
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
