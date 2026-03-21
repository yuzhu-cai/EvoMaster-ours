#!/usr/bin/env python3
"""Batch submit tasks to local OpenClaw and capture trajectory/final answer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from contextlib import suppress
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evomaster.utils.types import Trajectory
from playground.openclaw.openclaw_log_extract import (
    _build_dialog_snapshots,
    _has_any_session_headers,
    _iter_log_entries,
    _merge_dialog_snapshots,
)


FASTAPI_SERVER = PROJECT_ROOT / "playground" / "openclaw" / "env_tools" / "openclaw" / "fastapi_server.py"
MODELS_PROXY = PROJECT_ROOT / "playground" / "openclaw" / "env_tools" / "openclaw" / "models_proxy.py"


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
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("任务文件必须是 JSON 数组")
    tasks = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个任务不是合法对象")
        task = dict(item)
        task_id = str(task.get("id") or task.get("instance_id") or f"task_{index:04d}")
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = task.get("description")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"任务 `{task_id}` 缺少 `prompt` 或 `description`")
        task["id"] = task_id
        task["instance_id"] = str(task.get("instance_id") or task_id)
        task["prompt"] = prompt
        tasks.append(task)
    return tasks


def _find_cmd(name: str) -> str:
    cmd = shutil.which(name)
    if not cmd:
        raise RuntimeError(f"缺少可执行命令: {name}")
    return cmd


def _read_ini_port(path: Path, timeout_sec: float = 30.0) -> int:
    start = time.time()
    while time.time() - start < timeout_sec:
        if path.exists():
            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")
            with suppress(Exception):
                return int(parser["Server"]["port"])
        time.sleep(0.1)
    raise TimeoutError(f"等待 FastAPI 端口超时: {path}")


def _wait_port(host: str, port: int, timeout_sec: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"端口未就绪: {host}:{port}")


def _extract_dialogs_from_logs(log_dir: Path, session_id: str | None) -> list[dict[str, Any]]:
    entries = list(_iter_log_entries(log_dir))
    snapshots = _build_dialog_snapshots(entries, session_id)
    if not snapshots and session_id and (not _has_any_session_headers(entries)):
        snapshots = _build_dialog_snapshots(entries, None)
    return _merge_dialog_snapshots(snapshots)


def _extract_final_answer(dialogs: list[dict[str, Any]]) -> str:
    for dialog in reversed(dialogs):
        messages = dialog.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "assistant":
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


def _terminate_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    with suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    with suppress(Exception):
        proc.kill()
    with suppress(Exception):
        proc.wait(timeout=5)


def _start_fastapi(task_dir: Path, model_endpoint: str, env: dict[str, str]) -> tuple[subprocess.Popen[Any], int]:
    fastapi_ini = task_dir / "fastapi.ini"
    env = dict(env)
    env["FASTAPI_LOG_DIR"] = str(task_dir / "fastapi_logs")

    stdout_file = (task_dir / "fastapi_server.stdout.log").open("w", encoding="utf-8")
    stderr_file = (task_dir / "fastapi_server.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(FASTAPI_SERVER),
            "--model-endpoint",
            model_endpoint,
        ],
        cwd=str(task_dir),
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
    )
    port = _read_ini_port(fastapi_ini)
    _wait_port("127.0.0.1", port)
    return proc, port


def _start_models_proxy(task_dir: Path, upstream_base: str, model_id: str, env: dict[str, str]) -> tuple[subprocess.Popen[Any], int]:
    port_file = task_dir / "openclaw_proxy_port.txt"
    pid_file = task_dir / "openclaw_proxy_pid.txt"
    log_file = (task_dir / "openclaw_models_proxy.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(MODELS_PROXY),
            "--upstream",
            upstream_base,
            "--model-id",
            model_id,
            "--port-file",
            str(port_file),
            "--pid-file",
            str(pid_file),
            "--state-file",
            str(task_dir / "openclaw_chat_state.json"),
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            "0",
        ],
        cwd=str(task_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    start = time.time()
    while time.time() - start < 30:
        if port_file.exists():
            raw = port_file.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                port = int(raw)
                _wait_port("127.0.0.1", port)
                return proc, port
        if proc.poll() is not None:
            raise RuntimeError("models proxy 提前退出")
        time.sleep(0.1)
    raise TimeoutError("等待 models proxy 启动超时")


def _build_openclaw_config(
    task_dir: Path,
    provider: str,
    model_api: str,
    base_url: str,
    model_id: str,
    api_key: str,
    max_tokens: int | None,
) -> Path:
    provider_id = "anthropic" if model_api == "anthropic-messages" else "openai"
    config = {
        "$schema": "https://openclaw.ai/config.json",
        "providers": {
            provider_id: {
                "type": provider_id,
                "name": provider_id,
                "apiKey": api_key,
                "baseURL": base_url,
                "models": {
                    model_id: {
                        "name": model_id,
                    }
                },
            }
        },
        "agents": {
            "defaults": {
                "model": f"{provider_id}/{model_id}",
            }
        },
    }
    if max_tokens is not None:
        config["providers"][provider_id]["models"][model_id]["options"] = {"maxTokens": max_tokens}
    config_path = task_dir / "openclaw.json"
    _write_json(config_path, config)
    return config_path


def _run_one(task: dict[str, Any], args: argparse.Namespace, index: int) -> dict[str, Any]:
    task_id = task["id"]
    task_dir = Path(args.output_dir).resolve() / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (task_dir / "fastapi_logs").mkdir(parents=True, exist_ok=True)

    openclaw_bin = _find_cmd("openclaw")
    api_key = str(task.get("api_key") or args.api_key or os.getenv("MODEL_PROXY_API_KEY") or os.getenv("MODELPROXY_APIKEY") or "")
    if not api_key:
        raise RuntimeError(f"任务 `{task_id}` 缺少 api key")

    model_id = str(task.get("openclaw_model") or args.model or "")
    if not model_id:
        raise RuntimeError(f"任务 `{task_id}` 缺少模型名，请传 `--model` 或任务内 `openclaw_model`")

    model_endpoint = str(task.get("model_endpoint") or args.model_endpoint or args.endpoint)
    model_api = str(task.get("model_api") or args.model_api).strip().lower()
    if model_api not in {"openai-responses", "openai-completions", "anthropic-messages"}:
        raise RuntimeError(f"任务 `{task_id}` 的 model_api 非法: {model_api}")
    provider = "anthropic" if model_api == "anthropic-messages" else "openai"

    workspace_root = Path(str(task.get("workspace_root") or (task_dir / "workspace"))).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    timeout_sec = int(task.get("openclaw_timeout_sec") or args.timeout_sec)
    thinking = str(task.get("openclaw_thinking") or args.thinking).strip().lower()
    system_prompt = str(task.get("openclaw_system_prompt") or args.system_prompt or "").strip()
    prompt = str(task["prompt"]).strip()
    full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt

    fastapi_env = os.environ.copy()
    fastapi_proc = None
    proxy_proc = None
    stdout_path = task_dir / "openclaw.stdout.log"
    stderr_path = task_dir / "openclaw.stderr.log"
    started_at = datetime.now().isoformat()
    session_id = str(task.get("session_id") or f"{task_id}-{index}")

    try:
        fastapi_proc, port = _start_fastapi(task_dir, model_endpoint, fastapi_env)
        if model_api == "anthropic-messages":
            base_url = f"http://127.0.0.1:{port}"
        else:
            proxy_proc, proxy_port = _start_models_proxy(
                task_dir,
                upstream_base=f"http://127.0.0.1:{port}",
                model_id=model_id,
                env=fastapi_env,
            )
            base_url = f"http://127.0.0.1:{proxy_port}/v1"

        config_path = _build_openclaw_config(
            task_dir=task_dir,
            provider=provider,
            model_api=model_api,
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            max_tokens=task.get("openclaw_max_tokens") or args.max_tokens,
        )

        run_env = os.environ.copy()
        run_env.update(
            {
                "OPENCLAW_CONFIG_PATH": str(config_path),
                "OPENCLAW_API_KEY": api_key,
                "OPENAI_API_KEY": api_key,
                "ANTHROPIC_API_KEY": api_key,
                "OPENCLAW_SKIP_CHANNELS": "1",
                "CLAWDBOT_SKIP_CHANNELS": "1",
                "OPENCLAW_SESSION_ID": session_id,
                "SWE_OPENCLAW_MODEL_API": model_api,
                "SWE_OPENCLAW_PROVIDER": provider,
            }
        )

        cmd = [
            openclaw_bin,
            "agent",
            "--local",
            "--agent",
            "main",
            "--session-id",
            session_id,
            "--message",
            full_prompt,
            "--thinking",
            thinking,
            "--json",
            "--timeout",
            str(timeout_sec),
        ]

        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_root),
                env=run_env,
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout_sec + 30,
            )

        dialogs = _extract_dialogs_from_logs(task_dir / "fastapi_logs", session_id)
        final_answer = _extract_final_answer(dialogs)
        status = "completed" if proc.returncode == 0 else "failed"
        result = {
            "task_id": task_id,
            "instance_id": task["instance_id"],
            "status": status,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "returncode": proc.returncode,
            "session_id": session_id,
            "final_answer": final_answer,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "meta": {
                "model_api": model_api,
                "provider": provider,
                "model": model_id,
                "base_url": base_url,
                "workspace_root": str(workspace_root),
                "task_dir": str(task_dir),
            },
        }
        trajectory = _build_trajectory(task_id, dialogs, result)
        _write_json(task_dir / "dialogs.json", dialogs)
        _write_json(task_dir / "trajectory.json", trajectory)
        _write_json(task_dir / "meta.json", result["meta"])
        _write_json(task_dir / "result.json", result)
        (task_dir / "final_answer.txt").write_text(final_answer, encoding="utf-8")
        return result
    except subprocess.TimeoutExpired as exc:
        result = {
            "task_id": task_id,
            "instance_id": task["instance_id"],
            "status": "failed",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "error": f"OpenClaw 超时: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(task_dir / "result.json", result)
        return result
    except Exception as exc:
        result = {
            "task_id": task_id,
            "instance_id": task["instance_id"],
            "status": "failed",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(task_dir / "result.json", result)
        return result
    finally:
        _terminate_process(proxy_proc)
        _terminate_process(fastapi_proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量向本机 OpenClaw 提交任务并抓取轨迹/最终回答")
    parser.add_argument("--tasks", required=True, help="任务 JSON 文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--endpoint", required=True, help="上游模型 endpoint")
    parser.add_argument("--model-endpoint", help="可选模型 endpoint，默认同 `--endpoint`")
    parser.add_argument("--model", help="默认模型名")
    parser.add_argument("--api-key", help="默认 API key；不传则读环境变量")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="单任务默认超时")
    parser.add_argument("--max-tokens", type=int, help="默认最大输出 token")
    parser.add_argument("--system-prompt", help="默认 system prompt")
    parser.add_argument("--thinking", default="high", help="OpenClaw thinking 级别")
    parser.add_argument("--model-api", default="openai-responses", help="openai-responses/openai-completions/anthropic-messages")
    parser.add_argument("--parallel", type=int, default=1, help="并发数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _load_tasks(Path(args.tasks).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
