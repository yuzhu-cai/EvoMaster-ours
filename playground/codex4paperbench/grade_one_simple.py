from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from paperbench.grade import grade_submission
from paperbench.monitor.monitor import BasicMonitor
from paperbench.paper_registry import paper_registry
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.utils import RetryConfig


def _latest_submission(run_dir: Path) -> Path | None:
    submissions = sorted(run_dir.glob("submissions/*/submission.tar.gz"))
    return submissions[-1] if submissions else None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_grade(
    run_dir: Path,
    *,
    paper_id: str,
    run_id: str,
    judge_output: dict | None,
    monitor_result: dict | None,
    grader_log: str,
) -> None:
    old_grade_path = run_dir / "grade.json"
    old_agent_output = None
    if old_grade_path.exists():
        old = _load_json(old_grade_path)
        old_agent_output = (old.get("paperbench_result") or {}).get("agent_output")

    score = float(judge_output["score"]) if judge_output else 0.0
    grade = {
        "paperbench_result": {
            "paper_id": paper_id,
            "run_id": run_id,
            "submission_exists": judge_output is not None,
            "skipped_reproduction": True,
            "code_only": True,
            "resources_provided": False,
            "agent_output": old_agent_output,
            "judge_output": judge_output,
            "reproduction_metadata": None,
            "monitor_result": monitor_result,
            "monitor_ran": monitor_result is not None,
        },
        "score": score,
        "grader_log": grader_log,
    }
    old_grade_path.write_text(json.dumps(grade, indent=2))


async def _grade(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    grade_path = run_dir / "grade.json"
    if not grade_path.exists():
        raise FileNotFoundError(f"Missing grade.json in {run_dir}")

    old_grade = _load_json(grade_path)
    pb = old_grade.get("paperbench_result") or {}
    paper_id = args.paper_id or pb.get("paper_id")
    run_id = pb.get("run_id") or run_dir.name
    if not paper_id:
        raise ValueError(f"Could not determine paper_id for {run_dir}")

    submission = _latest_submission(run_dir)
    if not submission:
        _write_grade(
            run_dir,
            paper_id=paper_id,
            run_id=run_id,
            judge_output=None,
            monitor_result=None,
            grader_log="No checkpoint exists, skipping grading!",
        )
        return 2

    grader_output = Path(str(submission).replace(".tar.gz", f"_grader_output_{args.grade_id}.json"))

    monitor_result = None
    agent_log = run_dir / "agent.log"
    if agent_log.exists():
        monitor = BasicMonitor(paper=paper_registry.get_paper(paper_id))
        monitor_result_obj = monitor.check_log(agent_log.as_posix())
        monitor_result = monitor_result_obj.to_dict()
        if monitor_result_obj.violations:
            _write_grade(
                run_dir,
                paper_id=paper_id,
                run_id=run_id,
                judge_output=None,
                monitor_result=monitor_result,
                grader_log="Submission flagged by monitor",
            )
            return 3

    if grader_output.exists() and not args.force:
        judge_output_dict = _load_json(grader_output)
        _write_grade(
            run_dir,
            paper_id=paper_id,
            run_id=run_id,
            judge_output=judge_output_dict,
            monitor_result=monitor_result,
            grader_log="Grading completed successfully",
        )
        return 0

    model = os.environ.get("GPT_CHAT_MODEL") or os.environ.get(
        "CODEX4PAPERBENCH_JUDGE_STRUCTURED_MODEL"
    )
    if not model:
        raise ValueError("Set GPT_CHAT_MODEL or CODEX4PAPERBENCH_JUDGE_STRUCTURED_MODEL")

    completer_config = OpenAICompletionsTurnCompleter.Config(
        model=model,
        retry_config=RetryConfig(
            wait_min=args.retry_wait_min,
            wait_max=args.retry_wait_max,
            stop_after=args.retry_stop_after,
        ),
    )

    judge_output = await grade_submission(
        submission_path=submission.as_posix(),
        grader_upload_path=grader_output.as_posix(),
        paper_id=paper_id,
        judge_type="simple",
        completer_config=completer_config,
        run_group_id=run_dir.parent.name,
        runs_dir=run_dir.parent.parent.as_posix(),
        run_id=run_id,
        code_only=True,
        resources_provided=False,
        computer=None,
    )
    if judge_output is None:
        _write_grade(
            run_dir,
            paper_id=paper_id,
            run_id=run_id,
            judge_output=None,
            monitor_result=monitor_result,
            grader_log="Grading failed",
        )
        return 4

    _write_grade(
        run_dir,
        paper_id=paper_id,
        run_id=run_id,
        judge_output=judge_output.to_dict(),
        monitor_result=monitor_result,
        grader_log="Grading completed successfully",
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--paper-id")
    parser.add_argument("--grade-id", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-wait-min", type=float, default=1.0)
    parser.add_argument("--retry-wait-max", type=float, default=30.0)
    parser.add_argument("--retry-stop-after", type=float, default=900.0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_grade(args)))


if __name__ == "__main__":
    main()
