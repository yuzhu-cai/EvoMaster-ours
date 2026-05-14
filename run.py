#!/usr/bin/env python3
"""EvoMaster unified entry point

Usage:
  python run.py --agent minimal --task "your task description"
  python run.py --agent minimal_multi_agent --config configs/minimal_multi_agent/config.yaml
  python run.py --agent minimal_skill_task --config configs/minimal_skill/config.yaml

Arguments:
  --agent: Specify the playground name (required)
  --config: Specify the config file path (optional, defaults to configs/{agent}/config.yaml)
  --task: Task description (optional, enters interactive input if not provided)
  --interactive: Interactive mode (optional)
  --run-dir: Specify the run directory (optional, auto-creates runs/{agent}_{timestamp}/ by default)
"""

import argparse
import logging
import os
import sys
import importlib
import time
from pathlib import Path
from datetime import datetime

# Add project root directory to sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import get_playground_class, list_registered_playgrounds


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="EvoMaster unified entry point - run the specified playground agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the minimal agent with default config
  python run.py --agent minimal --task "analyze data"

  # Use a custom config
  python run.py --agent minimal --config my_config.yaml --task "analyze data"

  # Interactive mode
  python run.py --agent agent-builder --interactive

  # Specify a run directory
  python run.py --agent minimal --task "analyze data" --run-dir runs/my_experiment

  # Batch tasks (sequential)
  python run.py --agent minimal --task-file tasks.json

  # Batch tasks (parallel)
  python run.py --agent minimal --task-file tasks.json --parallel
        """
    )

    parser.add_argument(
        "--agent",
        required=True,
        help="Playground agent name (e.g., minimal, agent-builder, mcp-example)"
    )

    parser.add_argument(
        "--config",
        help="Config file path (default: configs/{agent}/config.yaml)"
    )

    # Task input (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        help="Single task description, or path to a task file (.txt or .md)"
    )
    task_group.add_argument(
        "--task-file",
        help="Path to a JSON file containing multiple tasks"
    )
    task_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (manual task input)"
    )

    parser.add_argument(
        "--run-dir",
        help="Specify run directory (default: auto-creates runs/{agent}_{timestamp}/)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute multiple tasks in parallel (only effective with --task-file)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers for --parallel (default: number of tasks)"
    )

    parser.add_argument(
        "--images",
        nargs="+",
        help="List of image file paths (supports PNG/JPG), for multimodal task input"
    )

    return parser.parse_args()


def setup_logging():
    """Configure basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress httpx INFO-level logs (keep WARNING and above only)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_task_description(args):
    """Get task description

    If args.task is a file path (.txt or .md), read the file content;
    otherwise return args.task directly as the task description.
    """
    if args.task:
        task_path = Path(args.task)
        # Check if it is a file path (.txt or .md)
        if task_path.suffix.lower() in ['.txt', '.md'] and task_path.exists() and task_path.is_file():
            try:
                with open(task_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if not content:
                    print(f"❌ Error: file {task_path} is empty")
                    sys.exit(1)
                return content
            except Exception as e:
                print(f"❌ Error: failed to read file {task_path}: {e}")
                sys.exit(1)
        # Not a file path or file does not exist, return directly as task description
        return args.task

    if args.interactive:
        print("\n" + "=" * 60)
        print("📝 Please enter task description (empty line to finish):")
        print("=" * 60)
        lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except EOFError:
                break

        if not lines:
            print("❌ Error: no task description provided")
            sys.exit(1)

        return '\n'.join(lines)

    # Neither --task nor --interactive provided
    print("❌ Error: please use --task to provide task description or --interactive for interactive mode")
    sys.exit(1)


def _sanitize_task_id_component(value: str) -> str:
    """Convert benchmark names into safe path components."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("._-")
    return cleaned or "task"


def _load_task_spec_from_description(description: str, task_file_path: Path) -> dict | None:
    """Best-effort load of a benchmark spec referenced by a task description."""
    import json
    import yaml

    raw = str(description or "").strip()
    if not raw:
        return None

    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    if len(raw) >= 4096:
        return None

    candidates: list[Path] = []
    try:
        direct = Path(raw).expanduser()
        candidates.append(direct)
        if not direct.is_absolute():
            candidates.append(task_file_path.parent / direct)
    except (OSError, ValueError):
        return None

    for candidate in candidates:
        try:
            if (
                candidate.exists()
                and candidate.is_file()
                and candidate.suffix.lower() in {".yaml", ".yml", ".json"}
            ):
                with open(candidate, "r", encoding="utf-8") as f:
                    if candidate.suffix.lower() == ".json":
                        parsed = json.load(f) or {}
                    else:
                        parsed = yaml.safe_load(f) or {}
                return parsed if isinstance(parsed, dict) else None
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue

    return None


def _infer_task_id(task_obj: dict, idx: int, task_file_path: Path) -> str:
    """Use benchmark identity for readable workspace names when possible."""
    spec = _load_task_spec_from_description(str(task_obj.get("description", "")), task_file_path)
    semantic_name = None
    if spec:
        for key in ("competition_id", "task_id", "name"):
            if spec.get(key):
                semantic_name = str(spec[key])
                break

    if semantic_name:
        return _sanitize_task_id_component(semantic_name)
    return f"task_{idx}"


def _explicit_resource_index(task_obj: dict) -> int | None:
    """Return an explicitly configured resource/GPU slot index, if present."""
    for key in ("resource_index", "parallel_index", "gpu_index", "task_index"):
        if key not in task_obj:
            continue
        raw = task_obj.get(key)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {key}: expected integer, got {raw!r}")
    return None


def parse_task_file(task_file_path: Path):
    """Parse task JSON file

    Args:
        task_file_path: Path to the JSON file

    Returns:
        List of tasks, each containing {id, description} fields
    """
    import json

    with open(task_file_path, 'r', encoding='utf-8') as f:
        tasks_raw = json.load(f)

    if not isinstance(tasks_raw, list):
        raise ValueError(f"Invalid task file format: expected list, got {type(tasks_raw).__name__}")

    tasks = []
    used_ids: set[str] = set()
    for idx, task in enumerate(tasks_raw):
        if isinstance(task, str):
            # Compatible with simple list format: ["task1", "task2"]
            task_obj = {"description": task}
        elif isinstance(task, dict):
            task_obj = task.copy()
        else:
            raise ValueError(f"Task {idx} has invalid format: expected string or dict, got {type(task).__name__}")

        # Validate required fields
        if "description" not in task_obj:
            raise ValueError(f"Task {idx} is missing the required field 'description'")

        # Keep the parallel resource slot separate from the user-facing task
        # id. The run loop supplies the default GPU index; an explicit value
        # in the task file can override it.
        explicit_resource_index = _explicit_resource_index(task_obj)
        if explicit_resource_index is not None:
            task_obj["resource_index"] = explicit_resource_index

        # Auto-generate readable benchmark IDs when possible. Treat task_N as the
        # generic default too, so MLE specs become e.g. spaceship-titanic.
        if "id" not in task_obj or task_obj.get("id") == f"task_{idx}":
            task_obj["id"] = _infer_task_id(task_obj, idx, task_file_path)

        # Avoid workspace collisions while preserving the clean benchmark name
        # for the common one-task-per-competition case.
        base_id = str(task_obj["id"])
        if base_id in used_ids:
            task_obj["id"] = f"{base_id}_{idx}"
        used_ids.add(str(task_obj["id"]))

        tasks.append(task_obj)

    return tasks


def run_single_task(agent_name: str, config_path: Path, run_dir: Path,
                    task_id: str, task_description: str, images: list[str] | None = None,
                    resource_index: int | None = None):
    """Run a single task (in the main process)

    Note: This function runs in the main process, not in a separate process.
    Each task has its own workspace, distinguished by task_id.

    Args:
        agent_name: Agent name
        config_path: Config file path
        run_dir: Run directory
        task_id: Task ID
        task_description: Task description
        images: List of image file paths (optional)
        resource_index: Stable parallel resource/GPU slot index for the task

    Returns:
        Task result dictionary
    """
    logger = logging.getLogger(__name__)
    previous_task_index = os.environ.get("EVOMASTER_TASK_INDEX")
    previous_parallel_index = os.environ.get("EVOMASTER_PARALLEL_INDEX")

    try:
        if resource_index is not None:
            os.environ["EVOMASTER_TASK_INDEX"] = str(resource_index)
            os.environ["EVOMASTER_PARALLEL_INDEX"] = str(resource_index)
            stagger_seconds = float(os.environ.get("EVOMASTER_TASK_STAGGER_SECONDS", "0") or 0)
            if resource_index > 0 and stagger_seconds > 0:
                delay = resource_index * stagger_seconds
                logger.info(f"Staggering task {task_id} startup by {delay:.1f}s")
                time.sleep(delay)
            logger.info(f"Task {task_id} resource/GPU index: {resource_index}")

        # Load Playground
        playground = get_playground_class(agent_name, config_path=config_path)

        # Set run_dir and task_id (creates an independent workspace)
        playground.set_run_dir(run_dir, task_id=task_id)

        # Run task
        if images:
            result = playground.run(task_description=task_description, images=images)
        else:
            result = playground.run(task_description=task_description)
        result["task_id"] = task_id
        if resource_index is not None:
            result["resource_index"] = resource_index

        logger.info(f"✅ Task {task_id} completed: {result['status']}")
        return result

    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}", exc_info=True)
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "steps": 0,
            "resource_index": resource_index,
        }
    finally:
        if resource_index is not None:
            if previous_task_index is None:
                os.environ.pop("EVOMASTER_TASK_INDEX", None)
            else:
                os.environ["EVOMASTER_TASK_INDEX"] = previous_task_index
            if previous_parallel_index is None:
                os.environ.pop("EVOMASTER_PARALLEL_INDEX", None)
            else:
                os.environ["EVOMASTER_PARALLEL_INDEX"] = previous_parallel_index


def _task_resource_index(task: dict, default_index: int) -> int:
    """Return the GPU/resource slot for a task."""
    explicit = _explicit_resource_index(task)
    return default_index if explicit is None else explicit


def run_tasks_sequential(agent_name: str, config_path: Path, run_dir: Path,
                         tasks: list, images: list[str] | None = None):
    """Run multiple tasks sequentially

    Args:
        agent_name: Agent name
        config_path: Config file path
        run_dir: Run directory
        tasks: List of tasks
        images: List of image file paths (optional, shared across all tasks)

    Returns:
        List of results for all tasks
    """
    results = []
    for idx, task in enumerate(tasks):
        task_images = task.get("images", images)
        resource_index = _task_resource_index(task, idx)
        result = run_single_task(
            agent_name,
            config_path,
            run_dir,
            task["id"],
            task["description"],
            images=task_images,
            resource_index=resource_index,
        )
        results.append(result)
    return results


def run_tasks_parallel(agent_name: str, config_path: Path, run_dir: Path,
                       tasks: list, max_workers: int | None = None, images: list[str] | None = None):
    """Run multiple tasks in parallel

    Uses ProcessPoolExecutor for parallel task execution.

    Args:
        agent_name: Agent name
        config_path: Config file path
        run_dir: Run directory
        tasks: List of tasks
        max_workers: Maximum number of parallel worker processes
        images: List of image file paths (optional, shared across all tasks)

    Returns:
        List of results for all tasks
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger = logging.getLogger(__name__)
    results = [None] * len(tasks)
    worker_count = max_workers if max_workers is not None else len(tasks)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    worker_count = min(worker_count, len(tasks))

    logger.info(f"Parallel workers: {worker_count}")

    def submit_task(executor, task_pos: int, slot_index: int):
        task = tasks[task_pos]
        task_images = task.get("images", images)
        resource_index = _task_resource_index(task, slot_index)
        logger.info(
            f"Submitting task {task['id']} with resource/GPU index {resource_index}"
        )
        future = executor.submit(
            run_single_task,
            agent_name,
            config_path,
            run_dir,
            task["id"],
            task["description"],
            task_images,
            resource_index,
        )
        return future

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_task: dict = {}
        next_task_pos = 0

        # Submit one task per parallel slot. When a slot finishes, reuse the
        # same slot index for the next task so active tasks never collide on
        # automatically assigned GPUs.
        for slot_index in range(worker_count):
            future = submit_task(executor, next_task_pos, slot_index)
            future_to_task[future] = (tasks[next_task_pos], next_task_pos, slot_index)
            next_task_pos += 1

        while future_to_task:
            for future in as_completed(future_to_task):
                task, task_pos, slot_index = future_to_task.pop(future)
                break
            try:
                result = future.result()
                results[task_pos] = result
            except Exception as e:
                logger.error(f"❌ Task {task['id']} failed: {e}")
                results[task_pos] = {
                    "task_id": task["id"],
                    "status": "failed",
                    "error": str(e),
                    "steps": 0,
                    "resource_index": _task_resource_index(task, slot_index),
                }

            if next_task_pos < len(tasks):
                future = submit_task(executor, next_task_pos, slot_index)
                future_to_task[future] = (
                    tasks[next_task_pos],
                    next_task_pos,
                    slot_index,
                )
                next_task_pos += 1

    return results



def auto_import_playgrounds():
    """Auto-import all playground modules to trigger decorator registration

    Iterates over all agent subdirectories under the playground directory
    and attempts to import their core.playground module.
    This ensures all classes using the @register_playground decorator are registered.
    """
    logger = logging.getLogger(__name__)
    playground_dir = project_root / "playground"

    if not playground_dir.exists():
        logger.warning(f"Playground directory does not exist: {playground_dir}")
        return

    imported_count = 0

    # Collect agent directories to scan: top-level + _generated/ subdirectories
    agent_dirs: list[tuple[Path, str]] = []  # (dir_path, module_prefix)
    for child in playground_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "_generated":
            # Scan each subdirectory under _generated/
            for gen_dir in child.iterdir():
                if gen_dir.is_dir() and not gen_dir.name.startswith("_"):
                    agent_dirs.append((gen_dir, f"playground._generated.{gen_dir.name}"))
        elif not child.name.startswith("_"):
            agent_dirs.append((child, f"playground.{child.name}"))

    for agent_dir, module_prefix in agent_dirs:
        module_name = f"{module_prefix}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ Successfully imported {module_name}")
            imported_count += 1
        except ImportError as e:
            # If no core/playground.py exists, skip (agent may use the default BasePlayground)
            # But if it's another import error (e.g., missing dependency), we should warn
            error_msg = str(e)
            if "No module named" in error_msg or "cannot import name" in error_msg or "core.playground" not in error_msg:
                logger.warning(f"❌ Failed to import {module_name}: {e}", exc_info=True)
            else:
                logger.debug(f"No custom playground for '{agent_dir.name}': {e}")
        except Exception as e:
            # Other errors (syntax errors, etc.) should trigger a warning
            logger.warning(f"❌ Failed to import {module_name}: {e}", exc_info=True)

    logger.info(f"Auto-imported {imported_count} playground modules")


def main():
    """Main entry function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Auto-import all playground modules (trigger decorator registration)
    auto_import_playgrounds()

    # Debug: display registered playgrounds
    registered = list_registered_playgrounds()
    if registered:
        logger.debug(f"Registered playgrounds: {registered}")

    args = parse_args()

    # 1. Determine config file path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_root / "configs" / args.agent / "config.yaml"

    if not config_path.exists():
        logger.error(f"Config file does not exist: {config_path}")
        sys.exit(1)

    # 2. Determine run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_root / "runs" / f"{args.agent}_{timestamp}"

    # 3. Validate image files (if provided)
    images = None
    if args.images:
        images = []
        for img_path_str in args.images:
            img_path = Path(img_path_str)
            if not img_path.exists():
                logger.error(f"Image file does not exist: {img_path}")
                sys.exit(1)
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                logger.error(f"Unsupported image format: {img_path.suffix} (only PNG/JPG supported)")
                sys.exit(1)
            images.append(str(img_path.absolute()))
        logger.info(f"Loaded {len(images)} images")

    # 4. Parse tasks
    if args.task_file:
        # Batch task mode
        task_file = Path(args.task_file)
        if not task_file.exists():
            logger.error(f"Task file does not exist: {task_file}")
            sys.exit(1)

        try:
            tasks = parse_task_file(task_file)
            logger.info(f"📋 Loaded {len(tasks)} tasks")
        except Exception as e:
            logger.error(f"Failed to parse task file: {e}")
            sys.exit(1)
    else:
        # Single task mode
        task_description = get_task_description(args)
        tasks = [{
            "id": "task_0",
            "description": task_description
        }]

    # 5. Print run information
    logger.info("=" * 60)
    logger.info("🚀 EvoMaster starting")
    logger.info("=" * 60)
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Run Directory: {run_dir}")
    logger.info(f"Tasks: {len(tasks)}")
    if images:
        logger.info(f"Images: {len(images)} files")
    if len(tasks) > 1:
        mode = "parallel" if args.parallel else "sequential"
        logger.info(f"Execution mode: {mode}")
    logger.info("=" * 60)

    # 6. Run tasks
    try:
        if len(tasks) > 1 and args.parallel:
            # Parallel mode
            logger.info("🔄 Executing tasks in parallel...")
            results = run_tasks_parallel(
                args.agent,
                config_path,
                run_dir,
                tasks,
                max_workers=args.max_workers,
                images=images,
            )
        else:
            # Sequential mode (including single task)
            if len(tasks) > 1:
                logger.info("🔄 Executing tasks sequentially...")
            results = run_tasks_sequential(args.agent, config_path, run_dir, tasks, images=images)

        # 7. Output results
        logger.info("=" * 60)
        logger.info("✅ All tasks completed")
        logger.info("=" * 60)

        # Tally results (note: trajectory.status values are "completed"/"failed"/"cancelled")
        success_count = sum(1 for r in results if r.get('status') == 'completed')
        failed_count = len(results) - success_count

        if len(tasks) == 1:
            # Single task mode: display detailed results
            result = results[0]
            logger.info(f"Status: {result['status']}")
            logger.info(f"Steps: {result.get('steps', 0)}")
        else:
            # Batch task mode: display summary and each task status
            logger.info(f"Succeeded: {success_count}/{len(results)}")
            logger.info(f"Failed: {failed_count}/{len(results)}")
            logger.info("")
            logger.info("Task status:")
            for result in results:
                status_icon = "✅" if result.get('status') == 'completed' else "❌"
                resource_index = result.get("resource_index")
                resource_info = (
                    f", resource/GPU index {resource_index}"
                    if resource_index is not None
                    else ""
                )
                logger.info(
                    f"  {status_icon} {result['task_id']}: {result['status']} "
                    f"({result.get('steps', 0)} steps{resource_info})"
                )

        logger.info("")
        logger.info(f"Results directory: {run_dir}")
        logger.info(f"  - Config: {run_dir}/config.yaml")
        logger.info(f"  - Logs: {run_dir}/logs/")
        logger.info(f"  - Trajectories: {run_dir}/trajectories/")
        if len(tasks) > 1:
            logger.info(f"  - Workspaces: {run_dir}/workspaces/")
        else:
            logger.info(f"  - Workspace: {run_dir}/workspace/")
        logger.info("=" * 60)

        return 0 if failed_count == 0 else 1

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
