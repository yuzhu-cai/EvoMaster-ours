#!/usr/bin/env python3
"""Evaluate agent solutions against ground truth using LLM."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env from project root
project_root = Path(__file__).parent.parent.parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass


JUDGE_PROMPT = """You are an exact evaluator. Determine if the following two answers are semantically equivalent.

Ground Truth: {answer}
Agent Output: {solution}

Rules:
- If they mean the same thing (same name, date, number, etc.), output exactly: 1
- If they differ in any meaningful way, output exactly: 0
- Output ONLY the digit 1 or 0, nothing else."""


def judge_single(client, answer: str, solution: str, model: str = "gpt-4o-mini") -> int:
    """Ask LLM to judge semantic equivalence. Returns 0 or 1."""
    try:
        prompt = JUDGE_PROMPT.format(answer=answer, solution=solution)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        if "1" in text:
            return 1
        return 0
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate solutions with LLM")
    parser.add_argument("--input", required=True, help="Input jsonl file (merge.jsonl)")
    parser.add_argument("--output", required=True, help="Output jsonl file (eval.jsonl)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel evaluation workers")
    parser.add_argument("--model", default="Vendor2/GPT-5", help="Judge model name")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client from .env
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("GPT_BASE_URL")
    model = args.model

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set (check .env file)")
        sys.exit(1)

    try:
        from openai import OpenAI
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
    except ImportError:
        print("❌ Error: openai package not installed. Run: pip install openai")
        sys.exit(1)

    # Load records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records for evaluation")

    # Evaluate in parallel
    def eval_record(record):
        score = judge_single(client, record.get("answer", ""), record.get("solution", ""), model=model)
        record["score"] = score
        return record

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_record, r): r for r in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Evaluated {completed}/{len(records)}...")

    # Sort by id and write output
    results.sort(key=lambda x: x.get("id", 0))
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r.get("score") == 1)
    print(f"Evaluation complete: {correct}/{len(results)} correct")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
