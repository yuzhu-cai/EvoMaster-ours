#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict


JUDGE_PROMPT_TEMPLATE = """You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.

Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric). ***

The problem: {problem}

***

The rubric: {rubric}

***

The attempted answer: {answer}

***

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: {{total_points}} in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone FrontierScience JSONL evaluator (rubric-based)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input jsonl path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Default: <input_stem>.scored.jsonl",
    )

    parser.add_argument(
        "--model",
        default="Vendor2/GPT-5",
        help="Judge model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("JUDGE_MODEL_URL", "https://api.gpugeek.com/v1"),
        help="Judge model base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="009njbjuxu4q9001000degiwa1xfovvi008wbdqe",
        help="Judge API key.",
    )
    parser.add_argument("--workers", type=int, default=20, help="Parallel worker count.")
    parser.add_argument("--max-tokens", type=int, default=128000, help="Max tokens for judge response.")
    parser.add_argument("--timeout", type=int, default=600, help="Judge request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count per sample on API errors.")
    parser.add_argument("--pass-threshold", type=float, default=7.0, help="Binary pass threshold on rubric score.")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort passed to judge model (none disables this parameter).",
    )
    parser.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="Keep full judge reasoning text in output field `reasoning`.",
    )
    parser.add_argument(
        "--problem-keys",
        default="problem,query,question",
        help="Comma-separated candidate keys for problem text.",
    )
    parser.add_argument(
        "--rubric-keys",
        default="answer",
        help="Comma-separated candidate keys for rubric text.",
    )
    parser.add_argument(
        "--answer-keys",
        default="solution,solution_refined",
        help="Comma-separated candidate keys for attempted answer text.",
    )
    return parser.parse_args()


def parse_key_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def pick_first(item: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def parse_verdict(text: str) -> float:
    matches = re.findall(r"VERDICT:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        return -1.0
    score = float(matches[-1])
    if score < 0 or score > 10:
        return -1.0
    return score


def call_judge(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        # Some providers may not support reasoning_effort.
        kwargs.pop("reasoning_effort", None)
        response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def judge_one(
    index: int,
    item: Dict[str, Any],
    args: argparse.Namespace,
    problem_keys: list[str],
    rubric_keys: list[str],
    answer_keys: list[str],
) -> Dict[str, Any]:
    output = dict(item)
    output["score"] = 0
    output["rubric_score"] = -1.0

    problem = pick_first(item, problem_keys)
    rubric = pick_first(item, rubric_keys)
    attempted_answer = pick_first(item, answer_keys)

    if not problem or not rubric or not attempted_answer:
        output["judge_error"] = (
            "missing required fields: "
            f"problem={'ok' if problem else 'missing'}, "
            f"rubric={'ok' if rubric else 'missing'}, "
            f"attempted_answer={'ok' if attempted_answer else 'missing'}"
        )
        if args.keep_reasoning:
            output["reasoning"] = ""
        return output

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        rubric=rubric,
        answer=attempted_answer,
    )

    try:
        from openai import OpenAI
    except Exception as exc:
        output["judge_error"] = f"openai package import failed: {exc}"
        if args.keep_reasoning:
            output["reasoning"] = ""
        return output

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    last_error = ""
    reasoning_text = ""
    for attempt in range(1, args.retries + 2):
        try:
            reasoning_text = call_judge(
                client=client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            break
        except Exception as exc:
            last_error = f"attempt {attempt} failed: {exc}"
            if attempt < args.retries + 1:
                time.sleep(min(2 * attempt, 5))

    if not reasoning_text:
        output["judge_error"] = last_error or "empty judge response"
        if args.keep_reasoning:
            output["reasoning"] = ""
        return output

    rubric_score = parse_verdict(reasoning_text)
    output["rubric_score"] = rubric_score
    output["score"] = int(rubric_score >= args.pass_threshold) if rubric_score >= 0 else 0
    if args.keep_reasoning:
        output["reasoning"] = reasoning_text
    if rubric_score < 0:
        output["judge_error"] = "failed to parse VERDICT in [0,10]"

    return output


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            items.append(obj)
    return items


def write_jsonl(path: Path, items: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"[ERROR] input not found: {args.input}")
        return 1
    if args.workers < 1:
        print("[ERROR] --workers must be >= 1")
        return 1
    if not args.api_key:
        print("[ERROR] missing API key. Set --api-key or JUDGE_MODEL_APIKEY / OPENAI_API_KEY.")
        return 1

    output_path = args.output or args.input.with_name(f"{args.input.stem}.scored.jsonl")
    problem_keys = parse_key_list(args.problem_keys)
    rubric_keys = parse_key_list(args.rubric_keys)
    answer_keys = parse_key_list(args.answer_keys)

    try:
        items = load_jsonl(args.input)
    except Exception as exc:
        print(f"[ERROR] failed to load input: {exc}")
        return 1

    total = len(items)
    if total == 0:
        print("[ERROR] empty input jsonl.")
        return 1

    print(f"[INFO] loaded {total} rows from {args.input}")
    print(f"[INFO] judging with model={args.model}, workers={args.workers}")

    results: list[Dict[str, Any]] = [None] * total  # type: ignore[assignment]
    completed = 0

    if args.workers == 1:
        for i, item in enumerate(items):
            results[i] = judge_one(i, item, args, problem_keys, rubric_keys, answer_keys)
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[INFO] progress: {completed}/{total}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(judge_one, i, item, args, problem_keys, rubric_keys, answer_keys): i
                for i, item in enumerate(items)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    fallback = dict(items[idx])
                    fallback["score"] = 0
                    fallback["rubric_score"] = -1.0
                    fallback["judge_error"] = f"worker exception: {exc}"
                    if args.keep_reasoning:
                        fallback["reasoning"] = ""
                    results[idx] = fallback
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"[INFO] progress: {completed}/{total}")

    write_jsonl(output_path, results)

    parsed_count = sum(1 for x in results if isinstance(x.get("rubric_score"), (int, float)) and x["rubric_score"] >= 0)
    pass_count = sum(1 for x in results if x.get("score") == 1)
    error_count = sum(1 for x in results if x.get("judge_error"))
    avg_score = (
        sum(float(x["rubric_score"]) for x in results if isinstance(x.get("rubric_score"), (int, float)) and x["rubric_score"] >= 0)
        / parsed_count
        if parsed_count > 0
        else 0.0
    )

    print("=" * 60)
    print(f"[DONE] output: {output_path}")
    print(f"[DONE] total={total}, parsed={parsed_count}, pass={pass_count}, errors={error_count}, avg_rubric_score={avg_score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
