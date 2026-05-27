#!/usr/bin/env python3
"""Grade one PaperBench Code-Dev submission directory.

Run this inside the `paperbench` conda environment. It loads the EvoMaster
`.env`, maps GPT_BASE_URL to OPENAI_BASE_URL for the OpenAI SDK used by
PaperBench's judge, and runs the simple code-only judge.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import types
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer import utils as turn_completer_utils
from preparedness_turn_completer.utils import RetryConfig

from paperbench.grade import JudgeOutput, run_judge
from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.simple import ParsedJudgeResponseFloat, ParsedJudgeResponseInt
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from paperbench.utils import get_timestamp


DEFAULT_ENV = Path("/data/yuzhu/Devs/EvoMaster-ours/.env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="Path to /home/submission mirror on host.")
    parser.add_argument("--paper-id", required=True, help="PaperBench paper id, e.g. rice.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for grader_output.json and logs.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="Env file containing OPENAI_API_KEY/GPT_* vars.")
    parser.add_argument("--model", default="", help="Judge model override. Defaults to GPT_CHAT_MODEL from env.")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-depth", type=int, default=999)
    parser.add_argument("--leaf-concurrency", type=int, default=3, help="Concurrent leaf grading calls.")
    parser.add_argument("--leaf-timeout", type=float, default=2400, help="Wall-clock timeout per leaf in seconds.")
    parser.add_argument("--retry-stop-after", type=float, default=900, help="Per-call retry budget in seconds.")
    parser.add_argument(
        "--remote-parse",
        action="store_true",
        help="Use PaperBench's extra LLM parser calls instead of the local score parser.",
    )
    parser.add_argument(
        "--print-failures",
        type=int,
        default=0,
        help="Print the top N low-scoring Code-Dev leaves for feedback-augmented runs.",
    )
    return parser.parse_args()


async def grade(args: argparse.Namespace) -> None:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)

    if os.getenv("GPT_BASE_URL") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_BASE_URL"]

    model = args.model or os.getenv("GPT_CHAT_MODEL")
    if not model:
        raise RuntimeError("No judge model configured: set GPT_CHAT_MODEL or pass --model.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    turn_completer_utils.CONTEXT_WINDOW_LENGTHS.setdefault(model, 400_000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completer_config = OpenAICompletionsTurnCompleter.Config(
        model=model,
        reasoning_effort=args.reasoning_effort,
        retry_config=RetryConfig(wait_min=2, wait_max=60, stop_after=args.retry_stop_after),
    )
    graded_task_tree = await _run_judge_controlled(
        submission_path=args.submission,
        paper_id=args.paper_id,
        completer_config=completer_config,
        out_dir=args.out_dir,
        max_depth=args.max_depth,
        leaf_concurrency=args.leaf_concurrency,
        leaf_timeout=args.leaf_timeout,
        local_parse=not args.remote_parse,
    )
    token_usage = get_total_token_usage(graded_task_tree)
    judge_output = JudgeOutput(
        judge_type="simple",
        completer_config=completer_config,
        score=graded_task_tree.score,
        num_leaf_nodes=len(graded_task_tree.get_leaf_nodes()),
        num_invalid_leaf_nodes=len(
            [node for node in graded_task_tree.get_leaf_nodes() if not node.valid_score]
        ),
        graded_at=get_timestamp(),
        graded_task_tree=graded_task_tree,
        token_usage=token_usage,
    )
    output = {
        "paper_id": args.paper_id,
        "submission": str(args.submission),
        "judge_output": judge_output.to_dict(),
        "score": judge_output.score,
    }
    path = args.out_dir / "grader_output.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"paper_id": args.paper_id, "score": judge_output.score, "out": str(path)}, indent=2))
    if args.print_failures > 0:
        print(_format_failure_feedback(judge_output.graded_task_tree.to_dict(), args.print_failures))


async def _run_judge_controlled(
    *,
    submission_path: Path,
    paper_id: str,
    completer_config: OpenAICompletionsTurnCompleter.Config,
    out_dir: Path,
    max_depth: int,
    leaf_concurrency: int,
    leaf_timeout: float,
    local_parse: bool,
):
    paper = paper_registry.get_paper(paper_id)
    with open(paper.rubric, "r", encoding="utf-8") as f:
        task_tree = TaskNode.from_dict(json.load(f)).code_only()
    if task_tree is None:
        task_tree = TaskNode.from_dict(json.load(open(paper.rubric))).set_task_category("Code Development").set_sub_tasks([])

    judge_kwargs = handle_judge_kwargs("simple", True, paper, completer_config)
    if not local_parse:
        judge_kwargs["int_completer_config"] = OpenAICompletionsTurnCompleter.Config(
            model=completer_config.model,
            reasoning_effort=completer_config.reasoning_effort,
            response_format=ParsedJudgeResponseInt,
            retry_config=completer_config.retry_config,
        )
        judge_kwargs["float_completer_config"] = OpenAICompletionsTurnCompleter.Config(
            model=completer_config.model,
            reasoning_effort=completer_config.reasoning_effort,
            response_format=ParsedJudgeResponseFloat,
            retry_config=completer_config.retry_config,
        )

    judge = create_judge(
        judge_type="simple",
        judge_kwargs=judge_kwargs,
        paper_path=paper.paper_pdf,
        rubric=task_tree,
        addendum=paper.addendum.read_text() if paper.addendum else None,
        judge_addendum=paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None,
        submission_dir=submission_path,
        paper_md=paper.paper_md,
        log_path=out_dir,
        max_depth=max_depth,
        computer=None,
    )
    outer_leaf_semaphore = asyncio.Semaphore(max(1, leaf_concurrency))
    # PaperBench launches all leaves with asyncio.gather; keep the timeout
    # inside our own semaphore so queued leaves are not marked as timeouts.
    judge.leaf_semaphore = asyncio.Semaphore(1_000_000)
    if local_parse:
        judge._parse_model_response = types.MethodType(_local_parse_model_response, judge)

    async def _timed_grade_leaf(task):
        try:
            async with outer_leaf_semaphore:
                return await asyncio.wait_for(judge.grade_leaf(task), timeout=leaf_timeout)
        except TimeoutError:
            from paperbench.judge.graded_task_node import GradedTaskNode

            return GradedTaskNode.from_task(
                task,
                score=0,
                valid_score=False,
                explanation=f"Leaf grading timed out after {leaf_timeout} seconds.",
                judge_metadata=None,
            )
        except Exception as exc:
            from paperbench.judge.graded_task_node import GradedTaskNode

            return GradedTaskNode.from_task(
                task,
                score=0,
                valid_score=False,
                explanation=(
                    f"Leaf grading failed: {type(exc).__name__}: "
                    f"{str(exc)[:1000]}"
                ),
                judge_metadata=None,
            )

    return await judge.judge(grade_leaf_fn=_timed_grade_leaf)


async def _local_parse_model_response(self, response: str | None, continuous: bool = False):
    text = response or ""
    score = _extract_score(text, continuous=continuous)
    parsed_cls = ParsedJudgeResponseFloat if continuous else ParsedJudgeResponseInt
    return parsed_cls(valid_score=True, score=score, explanation=_extract_explanation(text)), None


def _extract_score(text: str, *, continuous: bool) -> float | int:
    patterns = [
        r"(?:^|\n)\s*#+\s*Score\s*\n\s*\*{0,2}([01](?:\.\d+)?)\*{0,2}",
        r"(?:^|\n)\s*Score\s*[:=]\s*\*{0,2}([01](?:\.\d+)?)\*{0,2}",
        r"(?:score|Score)[^\n]{0,40}?\b([01](?:\.\d+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            value = max(0.0, min(1.0, value))
            return value if continuous else int(round(value))
    return 0.0 if continuous else 0


def _extract_explanation(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:1200] if compact else "Local parser found no explanation."


def _format_failure_feedback(tree: dict, limit: int) -> str:
    rows: list[dict[str, object]] = []

    def visit(node: dict, path: list[str], weight_product: float) -> None:
        req = str(node.get("requirements") or "").strip()
        score = float(node.get("score") or 0)
        weight = float(node.get("weight") or 1)
        children = node.get("sub_tasks") or []
        next_path = [*path, req[:140]] if req else path
        if children:
            for child in children:
                visit(child, next_path, weight_product * weight)
            return
        if score >= 0.999:
            return
        rows.append(
            {
                "score": score,
                "deficit": weight_product * weight * (1.0 - score),
                "requirements": req,
                "path": " > ".join(p for p in next_path[-4:] if p),
                "explanation": str(node.get("explanation") or "").strip(),
            }
        )

    visit(tree, [], 1.0)
    rows.sort(key=lambda item: float(item["deficit"]), reverse=True)
    if not rows:
        return "\nFailure feedback: all graded leaves scored 1.0.\n"

    lines = [
        "\nFailure feedback for feedback-augmented development only:",
        "These are the highest weighted low-scoring Code-Dev leaves. Use them to add concrete source files, not README-only claims.",
    ]
    for idx, row in enumerate(rows[:limit], 1):
        explanation = str(row["explanation"]).replace("\n", " ")
        if len(explanation) > 700:
            explanation = explanation[:700] + "..."
        lines.extend(
            [
                f"\n{idx}. score={row['score']:.3f}, weighted_deficit={row['deficit']:.4f}",
                f"   requirement: {row['requirements']}",
                f"   path: {row['path']}",
                f"   judge_note: {explanation}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    asyncio.run(grade(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
