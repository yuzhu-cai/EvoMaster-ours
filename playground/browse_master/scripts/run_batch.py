#!/usr/bin/env python3
"""Batch runner for BrowseMaster evaluation.

Runs browse_master agent on selected dataset entries in parallel.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core.exp import extract_agent_response


def parse_id_ranges(id_str: str) -> list[int]:
    """Parse id range string into sorted unique integers.

    Supports: '0', '0-9', '0,5,10', '0-2,5,8-10'
    """
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


def setup_logging(log_path: Path):
    """Setup logging to file and console."""
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


def run_single_entry(
    entry_id: int,
    question: str,
    run_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Run one dataset entry through browse_master agent.

    Returns dict with id, status, and task_path.
    """
    task_name = f"task_{entry_id:04d}"
    task_path = run_dir / task_name
    task_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{task_name}] Starting: {question[:80]}...")

    cmd = [
        sys.executable,
        str(project_root / "run.py"),
        "--agent", "browse_master",
        "--config", str(project_root / "configs" / "browse_master" / "config.yaml"),
        "--task", question,
        "--run-dir", str(task_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per task
        )

        if result.returncode != 0:
            logger.warning(f"[{task_name}] run.py exited with code {result.returncode}")
            logger.debug(f"[{task_name}] stderr: {result.stderr[:500]}")

        # Extract answer from trajectory file
        trajectory_file = task_path / "trajectories" / "trajectory.json"
        solution = ""
        if trajectory_file.exists():
            try:
                with open(trajectory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # data is a list of trajectory entries
                if data:
                    # Try to extract from the last entry's trajectory
                    last_entry = data[-1]
                    traj = last_entry.get("trajectory", {})
                    # Build a simple dict that extract_agent_response can handle
                    solution = extract_agent_response({"dialogs": [traj]})
            except Exception as e:
                logger.warning(f"[{task_name}] Failed to extract answer: {e}")

        # Write solution.txt
        solution_file = task_path / "solution.txt"
        with open(solution_file, "w", encoding="utf-8") as f:
            f.write(solution)

        logger.info(f"[{task_name}] Completed. Solution: {solution[:100]}")
        return {"id": entry_id, "status": "completed", "task_path": str(task_path)}

    except subprocess.TimeoutExpired:
        logger.error(f"[{task_name}] Timeout after 30 minutes")
        return {"id": entry_id, "status": "timeout", "task_path": str(task_path)}
    except Exception as e:
        logger.error(f"[{task_name}] Failed: {e}")
        return {"id": entry_id, "status": "failed", "task_path": str(task_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch run browse_master on dataset entries")
    parser.add_argument("--json", required=True, help="Path to dataset JSON file")
    parser.add_argument("--lines", required=True, help="ID ranges, e.g., 0, 0-9, 0,5,10")
    parser.add_argument("--run-dir", required=True, help="Root run directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    json_path = Path(args.json)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = run_dir / "workflow.log"
    logger = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("run_batch.py started")
    logger.info(f"Dataset: {json_path}")
    logger.info(f"IDs: {args.lines}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info("=" * 60)

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = {int(item["id"]): item for item in data if "id" in item}

    # Parse IDs
    ids = parse_id_ranges(args.lines)
    logger.info(f"Total tasks to run: {len(ids)}")

    # Run tasks in parallel
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {}
        for entry_id in ids:
            item = dataset.get(entry_id)
            if item is None:
                logger.warning(f"ID {entry_id} not found in dataset, skipping")
                continue
            question = item.get("question", "")
            future = executor.submit(run_single_entry, entry_id, question, run_dir, logger)
            future_to_id[future] = entry_id

        for future in as_completed(future_to_id):
            entry_id = future_to_id[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1
                logger.info(f"Progress: {completed + failed}/{len(future_to_id)} (completed={completed}, failed={failed})")
            except Exception as e:
                failed += 1
                logger.error(f"ID {entry_id} raised exception: {e}")

    logger.info("=" * 60)
    logger.info(f"run_batch.py finished: completed={completed}, failed={failed}")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
