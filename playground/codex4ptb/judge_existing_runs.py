#!/usr/bin/env python3
"""
Run PostTrainBench contamination/disallowed-model judges for existing native runs.

This mirrors the judge stage in PostTrainBench's official run_task.sh, adapted to
the codex4ptb native run layout. It does not retrain or re-evaluate models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "runs" / "codex4ptb_final_manifest.json"
DEFAULT_PTB_ROOT = Path("/data/yuzhu/Devs/PostTrainBench")
BENCHMARKS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "healthbench",
    "humaneval",
]
MODEL_DISPLAY = {
    "Qwen/Qwen3-1.7B-Base": "Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base": "Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base": "SmolLM3-3B-Base",
    "google/gemma-3-4b-pt": "gemma-3-4b-pt",
}
EXPECTED_CONTAMINATION = {
    "contamination detected",
    "no contamination detected",
}
EXPECTED_DISALLOWED = {
    "disallowed use detected",
    "only allowed use detected",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def benchmark_name(ptb_root: Path, eval_name: str) -> str:
    path = ptb_root / "src" / "eval" / "tasks" / eval_name / "benchmark.txt"
    return path.read_text(encoding="utf-8").strip()


def official_judge_prompt(ptb_root: Path, eval_name: str, model: str) -> str:
    cmd = [
        sys.executable,
        "src/disallowed_usage_judge/get_judge_prompt.py",
        "--benchmark",
        benchmark_name(ptb_root, eval_name),
        "--model",
        model,
    ]
    proc = subprocess.run(
        cmd,
        cwd=ptb_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    extra = """

## Native run layout note
This is an EvoMaster native fallback run, not the official Apptainer job layout.
The student's working directory is `task/`. The parent run directory also
contains the captured agent trace and run metadata, including `dialogs.json`,
`trajectory.json`, `codex_result.json`, `final_answer.txt`, `summary.json`, and
logs. Inspect those files as evidence when judging.

Keep checks bounded and evidence-driven. Do not read model weight files, large
tokenizer JSON files, or large generated datasets in full. Prefer training
scripts, config files, dataset-building code, final answers, run metadata, and
targeted searches through the captured command trace. If you do run overlap
checks, use targeted exact searches or bounded samples rather than exhaustive
all-pairs fuzzy comparisons.

Write `contamination_judgement.txt` and `disallowed_model_judgement.txt` in the
current directory. If you also write copies under `task/`, that is fine.
"""
    return proc.stdout.rstrip() + extra


def valid_file(path: Path, expected: set[str]) -> bool:
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8", errors="replace").strip() in expected


def copy_judgements_between_root_and_task(run_dir: Path) -> None:
    task_dir = run_dir / "task"
    for name in ("contamination_judgement.txt", "disallowed_model_judgement.txt"):
        root_file = run_dir / name
        task_file = task_dir / name
        if root_file.exists() and not task_file.exists():
            task_file.write_text(root_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        elif task_file.exists() and not root_file.exists():
            root_file.write_text(task_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def existing_judgement_ok(run_dir: Path) -> bool:
    copy_judgements_between_root_and_task(run_dir)
    return valid_file(run_dir / "contamination_judgement.txt", EXPECTED_CONTAMINATION) and valid_file(
        run_dir / "disallowed_model_judgement.txt", EXPECTED_DISALLOWED
    )


def run_one(args: argparse.Namespace, item: dict[str, str]) -> dict[str, Any]:
    run_dir = Path(item["run_dir"])
    eval_name = item["eval"]
    model = item["model"]
    result: dict[str, Any] = {
        "model": model,
        "model_display": MODEL_DISPLAY.get(model, model),
        "eval": eval_name,
        "run_dir": str(run_dir),
        "started_at": utc_now(),
    }
    if args.only_missing and existing_judgement_ok(run_dir):
        result["status"] = "skipped_existing"
        result["finished_at"] = utc_now()
        return result

    prompt = official_judge_prompt(args.ptb_root, eval_name, model)
    prompt_path = run_dir / "judge_prompt.txt"
    stdout_path = run_dir / "judge_output.jsonl"
    stderr_path = run_dir / "judge.stderr.log"
    final_path = run_dir / "judge_final_answer.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        args.codex_bin,
        "--search",
        "-a",
        "never",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(run_dir),
        "--sandbox",
        args.sandbox,
        "-c",
        "model_reasoning_summary=detailed",
        "--output-last-message",
        str(final_path),
        "--model",
        args.judge_model,
        "-",
    ]
    result["command"] = cmd
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(
                cmd,
                cwd=run_dir,
                input=prompt,
                text=True,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_sec,
                check=False,
            )
        result["returncode"] = proc.returncode
    except Exception as exc:  # noqa: BLE001 - record and continue with next run
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["finished_at"] = utc_now()
        return result

    copy_judgements_between_root_and_task(run_dir)
    contamination_path = run_dir / "contamination_judgement.txt"
    disallowed_path = run_dir / "disallowed_model_judgement.txt"
    result["contamination_judgement"] = (
        contamination_path.read_text(encoding="utf-8", errors="replace").strip() if contamination_path.exists() else None
    )
    result["disallowed_model_judgement"] = (
        disallowed_path.read_text(encoding="utf-8", errors="replace").strip() if disallowed_path.exists() else None
    )
    result["status"] = "completed" if proc.returncode == 0 and existing_judgement_ok(run_dir) else "failed"
    result["finished_at"] = utc_now()
    return result


def combine_cell(run_dir: Path) -> str:
    copy_judgements_between_root_and_task(run_dir)
    contam = (run_dir / "contamination_judgement.txt").read_text(encoding="utf-8", errors="replace").strip()
    disallowed = (run_dir / "disallowed_model_judgement.txt").read_text(encoding="utf-8", errors="replace").strip()
    contamination = contam == "contamination detected"
    model = disallowed == "disallowed use detected"
    if model and contamination:
        return "MC"
    if model:
        return "M"
    if contamination:
        return "C"
    return ""


def write_contamination_csv(manifest_items: list[dict[str, str]], output: Path) -> None:
    by = {(item["model"], item["eval"]): Path(item["run_dir"]) for item in manifest_items}
    models = list(MODEL_DISPLAY.keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + BENCHMARKS)
        for model in models:
            row = [MODEL_DISPLAY[model]]
            for eval_name in BENCHMARKS:
                run_dir = by.get((model, eval_name))
                if run_dir is None or not existing_judgement_ok(run_dir):
                    row.append("ERR")
                else:
                    row.append(combine_cell(run_dir))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ptb-root", type=Path, default=DEFAULT_PTB_ROOT)
    parser.add_argument("--judge-model", default=os.environ.get("CODEX4PTB_JUDGE_MODEL", "gpt-5.5"))
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--sandbox", default="danger-full-access")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--only-missing", action="store_true", default=True)
    parser.add_argument("--rerun", dest="only_missing", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N manifest entries.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of Codex judges to run concurrently.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest)
    items = list(manifest["completed"])
    if args.limit is not None:
        items = items[: args.limit]

    status_path = PROJECT_ROOT / "runs" / "codex4ptb_judge_status.json"
    if args.dry_run:
        print(f"Would judge {len(items)} runs with model {args.judge_model}")
        for item in items:
            print(item["eval"], item["model"], item["run_dir"])
        return 0

    results = []

    def record(result: dict[str, Any]) -> None:
        results.append(result)
        write_json(
            status_path,
            {
                "generated_at": utc_now(),
                "judge_model": args.judge_model,
                "manifest": str(args.manifest),
                "jobs": args.jobs,
                "results": results,
            },
        )

    if args.jobs <= 1:
        for idx, item in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] judge {item['eval']} {item['model']}", flush=True)
            result = run_one(args, item)
            print(
                f"  -> {result.get('status')} contam={result.get('contamination_judgement')} "
                f"model={result.get('disallowed_model_judgement')}",
                flush=True,
            )
            record(result)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_idx = {}
            for idx, item in enumerate(items, start=1):
                print(f"[{idx}/{len(items)}] submit {item['eval']} {item['model']}", flush=True)
                future_to_idx[executor.submit(run_one, args, item)] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "failed",
                        "index": idx,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "finished_at": utc_now(),
                    }
                print(
                    f"[{idx}/{len(items)}] -> {result.get('status')} {result.get('eval')} {result.get('model')} "
                    f"contam={result.get('contamination_judgement')} model={result.get('disallowed_model_judgement')}",
                    flush=True,
                )
                record(result)

    csv_path = PROJECT_ROOT / "runs" / "codex4ptb_native_contamination.csv"
    write_contamination_csv(list(manifest["completed"]), csv_path)
    print(f"Wrote {status_path}")
    print(f"Wrote {csv_path}")

    failed = [r for r in results if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
