#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from paperbench.grade import JudgeOutput
from paperbench.judge.graded_task_node import GradedTaskNode, update_all_grades
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from paperbench.utils import get_timestamp

SKIP_PREFIXES = (
    "./logs/",
    "logs/",
    "./submission/.git/",
    "submission/.git/",
    "./submission/.venv/",
    "submission/.venv/",
)
SKIP_EXACT = {
    ".",
    "./logs",
    "logs",
    "./submission",
    "submission",
    "./submission/.git",
    "submission/.git",
    "./submission/.venv",
    "submission/.venv",
}


def real_submission_files(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path) as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
    return [
        name
        for name in names
        if name not in SKIP_EXACT and not any(name.startswith(p) for p in SKIP_PREFIXES)
    ]


def latest_submission(run_dir: Path) -> Path | None:
    submissions = sorted((run_dir / "submissions").glob("*/submission.tar.gz"))
    return submissions[-1] if submissions else None


def zero_tree_for_paper(paper_id: str, code_only: bool = True, resources_provided: bool = False) -> GradedTaskNode:
    paper = paper_registry.get_paper(paper_id)
    task_tree = TaskNode.from_dict(json.loads(paper.rubric.read_text()))
    if code_only:
        task_tree = task_tree.code_only() or task_tree.set_task_category("Code Development").set_sub_tasks([])
    if resources_provided:
        task_tree = task_tree.resources_provided()

    explanation = (
        "The submitted archive contains no source files outside logs, git metadata, "
        "or virtual-environment metadata, so there is no evidence that this Code "
        "Development criterion was implemented. Auto-zeroed by the local PaperBench "
        "wrapper to avoid spending judge-model calls on an empty submission."
    )
    metadata: dict[str, Any] = {
        "autograded_empty_submission": True,
        "full_judge_response": explanation,
    }
    graded = GradedTaskNode.from_task(
        task_tree,
        score=0.0,
        valid_score=True,
        explanation=explanation,
        judge_metadata=metadata,
    )
    return update_all_grades(graded)


def update_grade_json(run_dir: Path, judge_output: dict[str, Any]) -> None:
    grade_path = run_dir / "grade.json"
    grade = json.loads(grade_path.read_text()) if grade_path.exists() else {}
    pb_result = grade.setdefault("paperbench_result", {})
    pb_result["judge_output"] = judge_output
    grade["score"] = 0.0
    grade["grader_log"] = "Grading completed successfully (empty submission auto-zeroed)"
    grade_path.write_text(json.dumps(grade, indent=2) + "\n")


def paper_id_from_run_dir(run_dir: Path) -> str:
    return run_dir.name.rsplit("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_group", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = []
    skipped_nonempty = []
    skipped_existing = []
    for run_dir in sorted(p for p in args.run_group.iterdir() if p.is_dir()):
        grade_path = run_dir / "grade.json"
        if grade_path.exists():
            grade = json.loads(grade_path.read_text())
            if grade.get("paperbench_result", {}).get("judge_output") is not None:
                skipped_existing.append(run_dir.name)
                continue
        submission = latest_submission(run_dir)
        if submission is None:
            continue
        real_files = real_submission_files(submission)
        if real_files:
            skipped_nonempty.append((run_dir.name, len(real_files)))
            continue

        paper_id = paper_id_from_run_dir(run_dir)
        graded_tree = zero_tree_for_paper(paper_id=paper_id)
        judge = JudgeOutput(
            judge_type="simple",
            score=0.0,
            num_leaf_nodes=len(graded_tree.get_leaf_nodes()),
            num_invalid_leaf_nodes=0,
            graded_at=get_timestamp(),
            graded_task_tree=graded_tree,
            completer_config=None,
            token_usage=None,
        )
        judge_dict = judge.to_dict()
        grader_output = Path(str(submission).replace(".tar.gz", "_grader_output_0.json"))
        changed.append((run_dir.name, grader_output, judge.num_leaf_nodes))
        if not args.dry_run:
            grader_output.write_text(json.dumps(judge_dict, indent=4) + "\n")
            update_grade_json(run_dir, judge_dict)

    print("auto_zeroed", len(changed))
    for name, grader_output, leaves in changed:
        print(f"zeroed\t{name}\tleaves={leaves}\t{grader_output}")
    print("skipped_nonempty", len(skipped_nonempty))
    for name, count in skipped_nonempty:
        print(f"nonempty\t{name}\treal_files={count}")
    print("skipped_existing", len(skipped_existing))


if __name__ == "__main__":
    main()
