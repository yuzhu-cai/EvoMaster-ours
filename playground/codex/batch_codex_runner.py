#!/usr/bin/env python3
"""Batch submit tasks to local Codex CLI and capture trajectory/final answer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evomaster.utils.types import Trajectory


DEFAULT_TIMEOUT_SEC = 1800
SANDBOX_CHOICES = ("read-only", "workspace-write", "danger-full-access")
APPROVAL_CHOICES = ("untrusted", "on-failure", "on-request", "never")


def _json_default(value: Any):
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _safe_task_id(value: Any, index: int) -> str:
    raw = str(value or f"task_{index:04d}").strip()
    raw = re.sub(r"[^\w.-]+", "_", raw, flags=re.ASCII).strip("._-")
    return raw or f"task_{index:04d}"


def _normalize_task(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"prompt": item}
    if not isinstance(item, dict):
        raise ValueError(f"Task {index} must be a string or object")

    task = dict(item)
    task_id = _safe_task_id(task.get("id") or task.get("instance_id"), index)
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = task.get("description")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Task `{task_id}` is missing `prompt` or `description`")

    task["id"] = task_id
    task["instance_id"] = str(task.get("instance_id") or task_id)
    task["prompt"] = prompt.strip()
    return task


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Task file must be a JSON array")
    return [_normalize_task(item, index) for index, item in enumerate(payload)]


def _load_single_task(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_file:
        prompt = Path(args.task_file).resolve().read_text(encoding="utf-8").strip()
    else:
        prompt = str(args.task or "").strip()
    if not prompt:
        raise ValueError("Single task prompt is empty")
    return [
        _normalize_task(
            {
                "id": args.task_id or "task_0000",
                "prompt": prompt,
            },
            0,
        )
    ]


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / f"codex_batch_{timestamp}"


def _find_cmd(name: str) -> str:
    cmd = shutil.which(name)
    if not cmd:
        raise RuntimeError(f"Missing executable command: {name}")
    return cmd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_content_to_text(v) for v in value) if part)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "aggregated_output", "message"):
            if key in value:
                text = _content_to_text(value.get(key))
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _format_command_tool_result(item: dict[str, Any]) -> str:
    command = str(item.get("command") or "").strip()
    status = str(item.get("status") or "").strip()
    output = _content_to_text(item.get("aggregated_output")).rstrip()
    exit_code = item.get("exit_code")

    parts = []
    if command:
        parts.append(f"$ {command}")
    if status or exit_code is not None:
        details = []
        if status:
            details.append(f"status={status}")
        if exit_code is not None:
            details.append(f"exit_code={exit_code}")
        parts.append(f"[{', '.join(details)}]")
    if output:
        parts.append(output)
    return "\n".join(parts).strip()


def _command_tool_call_message(item: dict[str, Any], index: int) -> dict[str, Any]:
    item_id = str(item.get("id") or f"command_{index}")
    command = str(item.get("command") or "").strip()
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": item_id,
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": command}, ensure_ascii=False),
                },
            }
        ],
        "meta": {
            "source": "codex_json",
            "event_index": index,
            "event_type": "item.started",
            "codex_item_type": "command_execution",
            "codex_item_id": item_id,
        },
    }


def _command_tool_result_message(item: dict[str, Any], index: int) -> dict[str, Any]:
    item_id = str(item.get("id") or f"command_{index}")
    return {
        "role": "tool",
        "content": _format_command_tool_result(item),
        "tool_call_id": item_id,
        "name": "exec_command",
        "meta": {
            "source": "codex_json",
            "event_index": index,
            "event_type": "item.completed",
            "codex_item_type": "command_execution",
            "codex_item_id": item_id,
            "status": item.get("status"),
            "exit_code": item.get("exit_code"),
        },
    }


def _message_from_item(event: dict[str, Any], index: int) -> dict[str, Any] | None:
    event_type = str(event.get("type") or "")
    item = event.get("item")
    if not isinstance(item, dict):
        return None

    item_type = str(item.get("type") or "unknown")
    item_id = item.get("id")

    if item_type == "command_execution":
        if event_type == "item.started":
            return _command_tool_call_message(item, index)
        if event_type == "item.completed":
            return _command_tool_result_message(item, index)
        return None

    if item_type == "agent_message":
        content = _content_to_text(item.get("text")).strip()
    elif item_type == "reasoning":
        content = _content_to_text(item.get("text")).strip()
    else:
        content = _content_to_text(item).strip()

    if not content:
        return None

    return {
        "role": "assistant",
        "content": content,
        "meta": {
            "source": "codex_json",
            "event_index": index,
            "event_type": event_type,
            "codex_item_type": item_type,
            "codex_item_id": item_id,
        },
    }


def _extract_dialogs_from_events(
    events: list[dict[str, Any]],
    prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "meta": {"source": "codex_prompt"},
        }
    ]
    meta: dict[str, Any] = {"event_count": len(events)}
    saw_command_tool = False

    for index, event in enumerate(events):
        event_type = str(event.get("type") or "")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if thread_id:
                meta["thread_id"] = thread_id
        elif event_type == "turn.completed":
            if isinstance(event.get("usage"), dict):
                meta["usage"] = event["usage"]
        elif event_type in {"item.started", "item.completed"}:
            message = _message_from_item(event, index)
            if message is not None:
                if (message.get("meta") or {}).get("codex_item_type") == "command_execution":
                    saw_command_tool = True
                messages.append(message)
        elif event_type:
            meta.setdefault("event_types", {})
            meta["event_types"][event_type] = meta["event_types"].get(event_type, 0) + 1

    tools = []
    if saw_command_tool:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Execute a shell command in the Codex task workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                        "required": ["cmd"],
                    },
                },
            }
        )

    return [{"messages": messages, "tools": tools, "meta": meta}], meta


def _extract_final_answer(dialogs: list[dict[str, Any]]) -> str:
    for dialog in reversed(dialogs):
        messages = dialog.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            meta = message.get("meta") or {}
            if meta.get("codex_item_type") != "agent_message":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _build_trajectory(task_id: str, dialogs: list[dict[str, Any]], result: dict[str, Any]) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        dialogs=dialogs,
        status=result["status"],
        result={
            "final_answer": result.get("final_answer", ""),
            "stdout_path": result.get("stdout_path", ""),
            "stderr_path": result.get("stderr_path", ""),
        },
        meta=result.get("meta", {}),
    )


def _resolve_images(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("images must be a string or list of strings")
    return [str(Path(str(value)).expanduser().resolve()) for value in values if str(value).strip()]


def _build_codex_cmd(
    *,
    codex_bin: str,
    workspace_root: Path,
    final_answer_path: Path,
    model: str | None,
    profile: str | None,
    sandbox: str,
    approval_policy: str,
    images: list[str],
    config_overrides: list[str],
    ignore_user_config: bool,
    ignore_rules: bool,
) -> list[str]:
    cmd = [
        codex_bin,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--cd",
        str(workspace_root),
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(final_answer_path),
        "-c",
        f'approval_policy="{approval_policy}"',
    ]
    if ignore_user_config:
        cmd.append("--ignore-user-config")
    if ignore_rules:
        cmd.append("--ignore-rules")
    if model:
        cmd.extend(["--model", model])
    if profile:
        cmd.extend(["--profile", profile])
    for image in images:
        cmd.extend(["--image", image])
    for config in config_overrides:
        cmd.extend(["-c", config])
    cmd.append("-")
    return cmd


def _run_one(task: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    task_id = task["id"]
    task_dir = Path(args.output_dir).resolve() / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "workspace").mkdir(parents=True, exist_ok=True)

    codex_bin = str(args.codex_bin or "codex")
    if os.sep not in codex_bin:
        codex_bin = _find_cmd(codex_bin)

    workspace_root = Path(str(task.get("workspace_root") or (task_dir / "workspace"))).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    timeout_sec = int(task.get("codex_timeout_sec") or task.get("timeout_sec") or args.timeout_sec)
    system_prompt = str(task.get("codex_system_prompt") or task.get("system_prompt") or args.system_prompt or "").strip()
    prompt = str(task["prompt"]).strip()
    full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
    model = str(task.get("codex_model") or task.get("model") or args.model or "").strip() or None
    profile = str(task.get("codex_profile") or task.get("profile") or args.profile or "").strip() or None
    sandbox = str(task.get("codex_sandbox") or args.sandbox).strip()
    approval_policy = str(task.get("codex_approval_policy") or args.approval_policy).strip()
    images = [*_resolve_images(args.image), *_resolve_images(task.get("images"))]

    stdout_path = task_dir / "codex.stdout.jsonl"
    stderr_path = task_dir / "codex.stderr.log"
    final_answer_path = task_dir / "final_answer.txt"
    started_at = datetime.now().isoformat()

    cmd = _build_codex_cmd(
        codex_bin=codex_bin,
        workspace_root=workspace_root,
        final_answer_path=final_answer_path,
        model=model,
        profile=profile,
        sandbox=sandbox,
        approval_policy=approval_policy,
        images=images,
        config_overrides=args.config or [],
        ignore_user_config=bool(args.ignore_user_config),
        ignore_rules=bool(args.ignore_rules),
    )
    meta = {
        "model": model,
        "profile": profile,
        "sandbox": sandbox,
        "approval_policy": approval_policy,
        "workspace_root": str(workspace_root),
        "task_dir": str(task_dir),
        "command": cmd,
        "prompt_source": "stdin",
    }

    returncode: int | None = None
    error: str | None = None
    tb: str | None = None

    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                env=os.environ.copy(),
                input=full_prompt,
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout_sec + 30,
            )
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        error = f"Codex timed out: {exc}"
        tb = traceback.format_exc()
    except Exception as exc:
        error = str(exc)
        tb = traceback.format_exc()

    events = _read_jsonl(stdout_path)
    dialogs, event_meta = _extract_dialogs_from_events(events, full_prompt)
    meta.update(event_meta)

    final_answer = ""
    if final_answer_path.exists():
        final_answer = final_answer_path.read_text(encoding="utf-8", errors="replace").strip()
    if not final_answer:
        final_answer = _extract_final_answer(dialogs)
        final_answer_path.write_text(final_answer, encoding="utf-8")

    status = "completed" if returncode == 0 and error is None else "failed"
    result = {
        "task_id": task_id,
        "instance_id": task["instance_id"],
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "returncode": returncode,
        "final_answer": final_answer,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "meta": meta,
    }
    if error:
        result["error"] = error
    if tb:
        result["traceback"] = tb

    trajectory = _build_trajectory(task_id, dialogs, result)
    _write_json(task_dir / "dialogs.json", dialogs)
    _write_json(task_dir / "trajectory.json", trajectory)
    _write_json(task_dir / "meta.json", result["meta"])
    _write_json(task_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Codex CLI tasks and capture trajectory/final answer")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Single task prompt")
    task_group.add_argument("--task-file", help="Text/Markdown file containing a single task prompt")
    task_group.add_argument("--tasks", help="Task JSON array path")
    parser.add_argument("--task-id", help="Task id for --task/--task-file (default: task_0000)")
    parser.add_argument("--output-dir", help="Output run directory (default: runs/codex_batch_<timestamp>)")
    parser.add_argument("--codex-bin", default="codex", help="Codex executable path/name")
    parser.add_argument("--model", help="Default Codex model")
    parser.add_argument("--profile", help="Default Codex config profile")
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Per-task timeout in seconds")
    parser.add_argument("--system-prompt", help="Default system prompt prepended to each task prompt")
    parser.add_argument("--sandbox", choices=SANDBOX_CHOICES, default="workspace-write", help="Codex sandbox mode")
    parser.add_argument("--approval-policy", choices=APPROVAL_CHOICES, default="never", help="Codex approval policy passed via -c")
    parser.add_argument("--image", action="append", default=[], help="Image path to attach to each task (repeatable)")
    parser.add_argument("--config", action="append", default=[], help="Extra Codex -c key=value override (repeatable)")
    parser.add_argument("--ignore-user-config", action="store_true", help="Pass --ignore-user-config to codex exec")
    parser.add_argument("--ignore-rules", action="store_true", help="Pass --ignore-rules to codex exec")
    parser.add_argument("--parallel", type=int, default=1, help="Number of tasks to run concurrently")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tasks:
        tasks = _load_tasks(Path(args.tasks).resolve())
    else:
        tasks = _load_single_task(args)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    results: list[dict[str, Any]] = []
    if args.parallel <= 1:
        for index, task in enumerate(tasks):
            results.append(_run_one(task, args, index))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_map = {
                executor.submit(_run_one, task, args, index): task
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_map):
                results.append(future.result())

    results.sort(key=lambda item: item.get("task_id", ""))
    summary = {
        "total": len(results),
        "completed": sum(1 for item in results if item.get("status") == "completed"),
        "failed": sum(1 for item in results if item.get("status") != "completed"),
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
    }
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
