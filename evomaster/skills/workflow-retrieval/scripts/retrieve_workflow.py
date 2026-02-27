#!/usr/bin/env python3
"""Retrieve workflow templates by lightweight keyword matching.

When no workflow is sufficiently relevant, return an explicit empty result:
- best_match: null
- ranked_candidates: []
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "without",
    "via",
    "use",
    "using",
    "based",
    "problem",
    "task",
    "workflow",
    "methodology",
    "method",
}


def tokenize(text: str) -> set[str]:
    # Keep both latin tokens and Chinese word blocks to support bilingual queries.
    tokens = re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+", (text or "").lower())
    cleaned: set[str] = set()
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        if re.fullmatch(r"[A-Za-z0-9_\-]+", t):
            if len(t) < 2:
                continue
            if t in STOPWORDS:
                continue
        cleaned.add(t)
    return cleaned


def extract_workflow_payload(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    wf = data.get("Workflow", {}) if isinstance(data, dict) else {}
    goal = str(wf.get("Goal", "")).strip()
    stages = wf.get("Stages", []) if isinstance(wf.get("Stages", []), list) else []

    stage_texts = []
    stage_objs = []
    def normalize_method_items(items) -> list[str]:
        normalized: list[str] = []
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    normalized.append(s)
            elif isinstance(item, dict):
                for k, v in item.items():
                    head = str(k).strip()
                    if isinstance(v, list):
                        for vv in v:
                            tail = str(vv).strip()
                            if head and tail:
                                normalized.append(f"{head}: {tail}")
                            elif head:
                                normalized.append(head)
                            elif tail:
                                normalized.append(tail)
                    else:
                        tail = str(v).strip()
                        if head and tail:
                            normalized.append(f"{head}: {tail}")
                        elif head:
                            normalized.append(head)
                        elif tail:
                            normalized.append(tail)
            else:
                s = str(item).strip()
                if s:
                    normalized.append(s)
        return normalized

    for stage in stages:
        if not isinstance(stage, dict):
            continue
        sid = str(stage.get("id", "")).strip()
        task = str(stage.get("task", "")).strip()
        methods = normalize_method_items(stage.get("method", []))
        output = stage.get("output", [])
        if not isinstance(output, list):
            output = [str(output)]
        output = [str(o).strip() for o in output if str(o).strip()]

        stage_texts.append(" ".join([sid, task, " ".join(methods), " ".join(output)]))
        stage_objs.append(
            {
                "id": sid,
                "task": task,
                "method": methods,
                "output": output,
            }
        )

    text_for_match = " ".join([path.stem, goal, " ".join(stage_texts)])
    goal_tokens = tokenize(goal)
    return {
        "name": path.stem,
        "path": str(path),
        "goal": goal,
        "goal_tokens": sorted(goal_tokens),
        "stages": stage_objs,
        "text_for_match": text_for_match,
    }


def score_workflow(
    query: str,
    payload: dict,
    *,
    min_query_overlap: float = 0.12,
    min_goal_overlap: float = 0.0,
) -> dict:
    q_tokens = tokenize(query)
    w_tokens = tokenize(payload.get("text_for_match", ""))
    goal_tokens = set(payload.get("goal_tokens", []))

    overlap = sorted(q_tokens & w_tokens)
    overlap_count = len(overlap)

    query_overlap_ratio = overlap_count / max(1, len(q_tokens))
    workflow_overlap_ratio = overlap_count / max(1, len(w_tokens))
    goal_overlap_ratio = (
        len(q_tokens & goal_tokens) / max(1, len(goal_tokens)) if goal_tokens else 0.0
    )

    # Slightly bias exact workflow-name hit.
    name = payload.get("name", "")
    name_bonus = 0.25 if name and name.lower() in query.lower() else 0.0
    base_score = 0.55 * query_overlap_ratio + 0.45 * workflow_overlap_ratio
    final = min(1.0, base_score + name_bonus)

    is_relevant = (
        overlap_count >= 1
        and query_overlap_ratio >= min_query_overlap
        and goal_overlap_ratio >= min_goal_overlap
    )

    return {
        "workflow": payload["name"],
        "score": round(final, 4),
        "overlap_count": overlap_count,
        "query_overlap_ratio": round(query_overlap_ratio, 4),
        "goal_overlap_ratio": round(goal_overlap_ratio, 4),
        "is_relevant": is_relevant,
        "overlap_tokens": overlap,
        "goal": payload.get("goal", ""),
        "stages": payload.get("stages", []),
        "path": payload.get("path", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve workflow templates")
    parser.add_argument("--query", required=True, help="Task query text")
    parser.add_argument("--top_k", type=int, default=3, help="How many ranked results to return")
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.2,
        help="Minimum relevance score to keep as candidate",
    )
    parser.add_argument(
        "--min_query_overlap",
        type=float,
        default=0.12,
        help="Minimum overlap ratio wrt query tokens",
    )
    parser.add_argument(
        "--min_goal_overlap",
        type=float,
        default=0.0,
        help="Minimum overlap ratio wrt workflow goal tokens",
    )
    parser.add_argument(
        "--workflow_dir",
        default=None,
        help="Optional override for workflow directory",
    )
    args = parser.parse_args()

    if args.workflow_dir:
        workflow_dir = Path(args.workflow_dir)
    else:
        workflow_dir = Path(__file__).resolve().parent.parent / "references" / "workflow"

    workflow_files = sorted(workflow_dir.glob("*.yaml"))
    payloads = [extract_workflow_payload(p) for p in workflow_files]

    scored = sorted(
        [
            score_workflow(
                args.query,
                p,
                min_query_overlap=args.min_query_overlap,
                min_goal_overlap=args.min_goal_overlap,
            )
            for p in payloads
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    relevant = [
        item
        for item in scored
        if item.get("is_relevant", False) and item.get("score", 0.0) >= args.min_score
    ]
    top_k = max(1, args.top_k)
    ranked = relevant[:top_k]
    best = ranked[0] if ranked else None

    if best is None:
        note = (
            "No sufficiently relevant workflow matched the query. "
            "Fallback to task-driven decomposition without forcing workflow template alignment."
        )
    else:
        note = (
            "Use best_match stages as decomposition scaffold and adapt to task-specific constraints."
        )
    result = {
        "query": args.query,
        "best_match": best,
        "ranked_candidates": ranked,
        "note": note,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
