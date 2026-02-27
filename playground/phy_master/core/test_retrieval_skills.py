#!/usr/bin/env python3
"""Minimal smoke test for prior/technique/workflow retrieval skills."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    name: str
    script: Path
    args: list[str]
    required_keys: list[str]


def run_case(case: TestCase, cwd: Path) -> tuple[bool, str]:
    if not case.script.exists():
        return False, f"script not found: {case.script}"

    cmd = [sys.executable, str(case.script), *case.args]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        return False, f"exit_code={proc.returncode}, error={err[:300]}"

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON output: {exc}"

    missing = [k for k in case.required_keys if k not in payload]
    if missing:
        return False, f"missing keys: {missing}"

    return True, "ok"


def main() -> int:
    root = Path(__file__).resolve().parents[3]

    prior_candidates = [
        root / "prior",
        root / "playground" / "phy_master" / "landau" / "prior",
        root / "playground" / "phy_master" / "LANDAU" / "prior",
    ]
    prior_base = next((p for p in prior_candidates if p.exists()), prior_candidates[0])

    cases = [
        TestCase(
            name="prior retrieval",
            script=root / "evomaster" / "skills" / "prior-retrieval" / "scripts" / "prior_search.py",
            args=[
                "--query",
                "critical level views violate which condition",
                "--top_k",
                "3",
                "--base_dir",
                str(prior_base),
            ],
            required_keys=["query", "results"],
        ),
        TestCase(
            name="technique retrieval",
            script=root / "evomaster" / "skills" / "technique-retrieval" / "scripts" / "technique_search.py",
            args=["--query", "LaMET expansion for large Pz", "--top_k", "3"],
            required_keys=["query", "results"],
        ),
        TestCase(
            name="workflow retrieval",
            script=root / "evomaster" / "skills" / "workflow-retrieval" / "scripts" / "retrieve_workflow.py",
            args=["--query", "build a cs kernel workflow", "--top_k", "2"],
            required_keys=["query", "best_match", "ranked_candidates"],
        ),
    ]

    all_ok = True
    print("Retrieval skill smoke test\n")
    for case in cases:
        ok, detail = run_case(case, cwd=root)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.name}: {detail}")
        all_ok = all_ok and ok

    print("\nSummary:", "ALL PASS" if all_ok else "HAS FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
