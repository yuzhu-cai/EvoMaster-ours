#!/usr/bin/env python3
"""Retrieve workflow templates by lightweight keyword matching."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 2}


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
    return {
        "name": path.stem,
        "path": str(path),
        "goal": goal,
        "stages": stage_objs,
        "text_for_match": text_for_match,
    }


def score_workflow(query: str, payload: dict) -> dict:
    q_tokens = tokenize(query)
    w_tokens = tokenize(payload.get("text_for_match", ""))

    overlap = sorted(q_tokens & w_tokens)
    denom = max(1, len(q_tokens))
    overlap_score = len(overlap) / denom

    # Slightly bias exact workflow-name hit.
    name = payload.get("name", "")
    name_bonus = 0.25 if name and name.lower() in query.lower() else 0.0
    final = min(1.0, overlap_score + name_bonus)

    return {
        "workflow": payload["name"],
        "score": round(final, 4),
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

    ranked = sorted(
        [score_workflow(args.query, p) for p in payloads],
        key=lambda x: x["score"],
        reverse=True,
    )

    top_k = max(1, args.top_k)
    ranked = ranked[:top_k]
    best = ranked[0] if ranked else None

    result = {
        "query": args.query,
        "best_match": best,
        "ranked_candidates": ranked,
        "note": "Use best_match stages as decomposition scaffold and adapt to task-specific constraints.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
