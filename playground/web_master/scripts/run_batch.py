#!/usr/bin/env python3
"""Batch runner for WebMaster evaluation.

Runs the web_master playground on selected BrowseComp dataset entries.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

FINAL_ANSWER_RE = re.compile(r"Agent final answer:\s*(.*?)\s*$")
DEFAULT_TAIL_LINES = 40


def parse_id_ranges(id_str: str) -> list[int]:
    """Parse id range string into sorted unique integers."""
    ids = set()
    for part in id_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start.strip()), int(end.strip())
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def read_last_lines(path: Path, limit: int = DEFAULT_TAIL_LINES) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return list(deque(handle, maxlen=limit))


def extract_solution_from_log(log_path: Path, tail_lines: int = DEFAULT_TAIL_LINES) -> str | None:
    if not log_path.exists():
        return None

    for line in reversed(read_last_lines(log_path, tail_lines)):
        match = FINAL_ANSWER_RE.search(line)
        if match:
            return match.group(1).strip()
    return None


def get_question_text(item: dict) -> str:
    return (item.get("question") or item.get("prompt") or "").strip()


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run_single_entry(entry_id: int, question: str, run_dir: Path, logger: logging.Logger) -> dict:
    """Run one dataset entry through the web_master playground."""
    task_name = f"task_{entry_id:04d}"
    task_path = run_dir / task_name
    task_path.mkdir(parents=True, exist_ok=True)
    log_path = task_path / "logs" / "task_0.log"
    solution_file = task_path / "solution.txt"

    if not question.strip():
        logger.error("[%s] Empty question, skipping", task_name)
        solution_file.write_text("", encoding="utf-8")
        return {
            "id": entry_id,
            "status": "failed",
            "task_path": str(task_path),
            "error": "empty_question",
        }

    logger.info("[%s] Starting: %s...", task_name, question[:80])

    cmd = [
        sys.executable,
        str(project_root / "run.py"),
        "--agent",
        "web_master",
        "--config",
        str(project_root / "configs" / "web_master" / "config.yaml"),
        "--task",
        question,
        "--run-dir",
        str(task_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10800,
        )

        if result.returncode != 0:
            logger.error("[%s] run.py exited with code %s", task_name, result.returncode)
            if result.stderr:
                logger.debug("[%s] stderr: %s", task_name, result.stderr[:500])

        solution = extract_solution_from_log(log_path) or ""
        solution_file.write_text(solution, encoding="utf-8")

        if result.returncode != 0:
            return {
                "id": entry_id,
                "status": "failed",
                "task_path": str(task_path),
                "error": f"run.py exit code {result.returncode}",
            }

        if not solution:
            logger.error("[%s] No final answer found in log: %s", task_name, log_path)
            return {
                "id": entry_id,
                "status": "failed",
                "task_path": str(task_path),
                "error": "solution_not_found",
            }

        logger.info("[%s] Completed. Solution: %s", task_name, solution[:100])
        return {"id": entry_id, "status": "completed", "task_path": str(task_path)}

    except subprocess.TimeoutExpired:
        logger.error("[%s] Timeout after 180 minutes", task_name)
        return {"id": entry_id, "status": "timeout", "task_path": str(task_path)}
    except Exception as exc:
        logger.error("[%s] Failed: %s", task_name, exc)
        return {
            "id": entry_id,
            "status": "failed",
            "task_path": str(task_path),
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch run web_master on dataset entries")
    parser.add_argument("--json", required=True, help="Path to dataset JSON file")
    parser.add_argument("--lines", required=True, help="ID ranges, e.g., 0, 0-9, 0,5,10")
    parser.add_argument("--run-dir", required=True, help="Root run directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    json_path = Path(args.json)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "workflow.log"
    logger = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("run_batch.py started")
    logger.info("Dataset: %s", json_path)
    logger.info("IDs: %s", args.lines)
    logger.info("Run dir: %s", run_dir)
    logger.info("Workers: %s", args.workers)
    logger.info("=" * 60)

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    dataset = {int(item["id"]): item for item in data if "id" in item}

    ids = parse_id_ranges(args.lines)
    logger.info("Total tasks to run: %s", len(ids))

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {}
        for entry_id in ids:
            item = dataset.get(entry_id)
            if item is None:
                logger.warning("ID %s not found in dataset, skipping", entry_id)
                continue
            future = executor.submit(
                run_single_entry,
                entry_id,
                get_question_text(item),
                run_dir,
                logger,
            )
            future_to_id[future] = entry_id

        for future in as_completed(future_to_id):
            entry_id = future_to_id[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1
                logger.info(
                    "Progress: %s/%s (completed=%s, failed=%s)",
                    completed + failed,
                    len(future_to_id),
                    completed,
                    failed,
                )
            except Exception as exc:
                failed += 1
                logger.error("ID %s raised exception: %s", entry_id, exc)

    logger.info("=" * 60)
    logger.info("run_batch.py finished: completed=%s, failed=%s", completed, failed)
    logger.info("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
