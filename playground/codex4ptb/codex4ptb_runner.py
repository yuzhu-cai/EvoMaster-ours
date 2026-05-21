#!/usr/bin/env python3
"""Native Codex runner for PostTrainBench tasks.

This is a fallback for environments where Apptainer/FUSE/HTCondor cannot run.
It keeps per-run HOME/TMP/work directories isolated, while using a local venv
and HuggingFace cache under playground/codex4ptb/local_state by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evomaster.utils.types import Trajectory
except Exception:  # pragma: no cover - keep the runner usable outside EvoMaster.
    Trajectory = None  # type: ignore[assignment]

CODEX4PTB_ROOT = Path(__file__).resolve().parent
DEFAULT_PTB_ROOT = Path(os.environ.get("POSTTRAINBENCH_ROOT", "/data/yuzhu/Devs/PostTrainBench"))
DEFAULT_STATE_DIR = CODEX4PTB_ROOT / "local_state"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_PROXY = "http://127.0.0.1:7890"
DEFAULT_BASE_VENV_NAME = "base_venv"

BASE_MODELS = [
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
    # Gemma is gated on Hugging Face; keep it last so public models download first.
    "google/gemma-3-4b-pt",
]

DEFAULT_DOWNLOAD_IGNORE_PATTERNS = [
    "onnx/*",
    "*.onnx",
    "*.onnx_data",
]

BENCHMARKS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "healthbench",
]


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "value"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _venv_bin(venv_dir: Path) -> Path:
    return venv_dir / "bin"


def _state_paths(args: argparse.Namespace) -> dict[str, Path]:
    state_dir = Path(args.state_dir).expanduser().resolve()
    return {
        "state_dir": state_dir,
        "venv_dir": Path(args.venv).expanduser().resolve() if args.venv else state_dir / DEFAULT_BASE_VENV_NAME,
        "hf_home": Path(args.hf_home).expanduser().resolve() if args.hf_home else state_dir / "hf_cache",
        "pip_cache": state_dir / "pip_cache",
        "uv_cache": state_dir / "uv_cache",
        "npm_cache": state_dir / "npm_cache",
        "xdg_cache": state_dir / "xdg_cache",
        "torchinductor_cache": state_dir / "torchinductor_cache",
        "triton_cache": state_dir / "triton_cache",
        "torch_extensions": state_dir / "torch_extensions",
    }


def _env_with_proxy(env: dict[str, str], proxy: str | None) -> dict[str, str]:
    env = dict(env)
    if proxy and proxy.lower() not in {"none", "off", "false", "0"}:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["ALL_PROXY"] = proxy
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
        env["all_proxy"] = proxy
        no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
        entries = [item.strip() for item in no_proxy.split(",") if item.strip()]
        for item in ("localhost", "127.0.0.1", "::1"):
            if item not in entries:
                entries.append(item)
        joined = ",".join(entries)
        env["NO_PROXY"] = joined
        env["no_proxy"] = joined
    return env


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout: Path | None = None,
    stderr: Path | None = None,
    input_text: str | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
    if stderr:
        stderr.parent.mkdir(parents=True, exist_ok=True)
    out_handle = stdout.open("w", encoding="utf-8") if stdout else None
    err_handle = stderr.open("w", encoding="utf-8") if stderr else None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=input_text,
            stdout=out_handle if out_handle else subprocess.PIPE,
            stderr=err_handle if err_handle else subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        if out_handle:
            out_handle.close()
        if err_handle:
            err_handle.close()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


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


def _build_trajectory(task_id: str, dialogs: list[dict[str, Any]], result: dict[str, Any]) -> Any:
    payload = {
        "task_id": task_id,
        "dialogs": dialogs,
        "status": result["status"],
        "result": {
            "final_answer": result.get("final_answer", ""),
            "stdout_path": result.get("stdout_path", ""),
            "stderr_path": result.get("stderr_path", ""),
            "metrics_path": result.get("metrics_path", ""),
        },
        "meta": result.get("meta", {}),
    }
    if Trajectory is None:
        return payload
    return Trajectory(**payload)


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    paths = _state_paths(args)
    env = _env_with_proxy(os.environ.copy(), getattr(args, "proxy", DEFAULT_PROXY))
    env["HF_HOME"] = str(paths["hf_home"])
    env["HF_DATASETS_CACHE"] = str(paths["hf_home"] / "datasets")
    env["PIP_CACHE_DIR"] = str(paths["pip_cache"])
    env["UV_CACHE_DIR"] = str(paths["uv_cache"])
    env["NPM_CONFIG_CACHE"] = str(paths["npm_cache"])
    env["XDG_CACHE_HOME"] = str(paths["xdg_cache"])
    env["TORCHINDUCTOR_CACHE_DIR"] = str(paths["torchinductor_cache"])
    env["TRITON_CACHE_DIR"] = str(paths["triton_cache"])
    env["TORCH_EXTENSIONS_DIR"] = str(paths["torch_extensions"])
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("VLLM_API_KEY", "inspectai")
    if paths["venv_dir"].exists():
        env["VIRTUAL_ENV"] = str(paths["venv_dir"])
        env["PATH"] = f"{_venv_bin(paths['venv_dir'])}:{env.get('PATH', '')}"
    return env


def _ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def _check_call_display(cmd: list[str], **kwargs: Any) -> None:
    print("+", " ".join(cmd), flush=True)
    _run(cmd, **kwargs)


def _site_packages(py: Path, env: dict[str, str]) -> Path:
    code = "import site; paths=site.getsitepackages(); print(paths[0])"
    proc = _run([str(py), "-c", code], env=env, check=True)
    return Path((proc.stdout or "").strip())


def _create_agent_venv(args: argparse.Namespace, run_dir: Path, base_env: dict[str, str]) -> tuple[Path, dict[str, str]]:
    """Create a per-run venv that can read base_venv packages via a .pth file."""
    paths = _state_paths(args)
    base_venv = paths["venv_dir"]
    base_py = _venv_python(base_venv)
    if not base_py.exists():
        raise RuntimeError(f"base venv missing: {base_venv}. Run setup-env first.")

    if not args.fresh_agent_venv:
        env = dict(base_env)
        env["CODEX4PTB_AGENT_VENV"] = str(base_venv)
        return base_venv, env

    agent_venv = run_dir / "venv_agent"
    agent_py = _venv_python(agent_venv)
    if not agent_py.exists():
        _check_call_display([args.python, "-m", "venv", str(agent_venv)], env=base_env)
        _check_call_display([str(agent_py), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools", "uv"], env=base_env)

    base_site = _site_packages(base_py, base_env)
    agent_site = _site_packages(agent_py, base_env)
    agent_site.mkdir(parents=True, exist_ok=True)
    (agent_site / "codex4ptb_base_venv.pth").write_text(str(base_site) + "\n", encoding="utf-8")

    env = dict(base_env)
    env["VIRTUAL_ENV"] = str(agent_venv)
    env["CODEX4PTB_BASE_VENV"] = str(base_venv)
    env["CODEX4PTB_AGENT_VENV"] = str(agent_venv)
    env["PATH"] = f"{_venv_bin(agent_venv)}:{_venv_bin(base_venv)}:{env.get('PATH', '')}"
    return agent_venv, env


def cmd_doctor(args: argparse.Namespace) -> int:
    paths = _state_paths(args)
    env = _base_env(args)
    print("EvoMaster root:", PROJECT_ROOT)
    print("codex4ptb root:", CODEX4PTB_ROOT)
    print("PostTrainBench root:", Path(args.ptb_root).expanduser().resolve())
    print("state_dir:", paths["state_dir"])
    print("base_venv:", paths["venv_dir"])
    print("hf_home:", paths["hf_home"])
    print("runs_dir:", Path(args.runs_dir).expanduser().resolve())
    print("proxy:", args.proxy)
    print()

    for name in ("codex", "python3", "nvidia-smi", "git", "curl"):
        found = shutil.which(name)
        print(f"{name:12s}", found or "MISSING")
    print()

    for cmd in (["codex", "--version"], ["python3", "--version"], ["nvidia-smi", "-L"]):
        try:
            proc = _run(cmd, env=env, check=False)
            text = ((proc.stdout or "") + (proc.stderr or "")).strip()
            print("$", " ".join(cmd))
            print(text[:2000] if text else f"(exit {proc.returncode}, no output)")
        except Exception as exc:
            print("$", " ".join(cmd), "FAILED:", exc)
        print()

    py = _venv_python(paths["venv_dir"])
    if py.exists():
        probe = "import sys; print(sys.executable); import transformers, datasets, inspect_ai; print('core imports ok')"
        proc = _run([str(py), "-c", probe], env=env, check=False)
        print("base venv probe:")
        print(((proc.stdout or "") + (proc.stderr or "")).strip())
    else:
        print("base venv probe: missing, run `setup-env` first")
    return 0


def cmd_setup_env(args: argparse.Namespace) -> int:
    paths = _state_paths(args)
    _ensure_dirs(paths)
    env = _base_env(args)
    venv_dir = paths["venv_dir"]
    py = _venv_python(venv_dir)

    if not py.exists():
        python_bin = args.python
        _check_call_display([python_bin, "-m", "venv", str(venv_dir)], env=env)
    else:
        print(f"Using existing venv: {venv_dir}")

    py = _venv_python(venv_dir)
    _check_call_display([str(py), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools", "uv"], env=env)

    uv_bin = str(_venv_bin(venv_dir) / "uv")
    requirements = Path(args.ptb_root).expanduser().resolve() / "containers" / "requirements-direct.txt"
    _check_call_display([uv_bin, "pip", "install", "--python", str(py), "vllm==0.11.0", "--torch-backend=auto"], env=env)
    if requirements.exists():
        _check_call_display([uv_bin, "pip", "install", "--python", str(py), "-r", str(requirements)], env=env)
    else:
        raise FileNotFoundError(f"Missing requirements file: {requirements}")

    inspect_evals_spec = "git+https://github.com/UKGovernmentBEIS/inspect_evals.git@06001a83e6d7c709c2ede0570dce7f1031a0bad8"
    _check_call_display([uv_bin, "pip", "install", "--python", str(py), inspect_evals_spec], env=env)

    extra_packages = ["python-dotenv", "requests", "tqdm", "jinja2"]
    _check_call_display([uv_bin, "pip", "install", "--python", str(py), *extra_packages], env=env)

    if args.flash_attn:
        _check_call_display([uv_bin, "pip", "install", "--python", str(py), "flash-attn==2.8.3", "--no-build-isolation"], env=env)

    print(f"Base environment ready: {venv_dir}")
    return 0


def cmd_download_cache(args: argparse.Namespace) -> int:
    paths = _state_paths(args)
    _ensure_dirs(paths)
    env = _base_env(args)
    py = _venv_python(paths["venv_dir"])
    if not py.exists():
        raise RuntimeError(f"venv missing: {py}. Run setup-env first.")

    models = args.model or BASE_MODELS
    script = paths["state_dir"] / "download_models.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from huggingface_hub import snapshot_download

            ignore_patterns = json.loads(os.environ.get("CODEX4PTB_DOWNLOAD_IGNORE_PATTERNS", "[]"))
            for model in sys.argv[1:]:
                print(f"Downloading {model} into HF_HOME={os.environ.get('HF_HOME')}", flush=True)
                snapshot_download(
                    repo_id=model,
                    repo_type="model",
                    resume_download=True,
                    ignore_patterns=ignore_patterns,
                )
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    env["CODEX4PTB_DOWNLOAD_IGNORE_PATTERNS"] = json.dumps(args.ignore_pattern, ensure_ascii=False)
    _check_call_display([str(py), str(script), *models], env=env)
    print(f"HF cache ready: {paths['hf_home']}")
    return 0


def _copy_optional_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _prepare_task_dir(ptb_root: Path, eval_name: str, task_dir: Path) -> None:
    src_task = ptb_root / "src" / "eval" / "tasks" / eval_name
    if not src_task.is_dir():
        raise FileNotFoundError(f"Unknown PTB eval `{eval_name}`: {src_task}")
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_task / "evaluate.py", task_dir / "evaluate.py")
    _copy_optional_tree(src_task / "evaluation_code", task_dir / "evaluation_code")
    _copy_optional_tree(src_task / "task_context", task_dir)
    shutil.copytree(ptb_root / "src" / "eval" / "templates", task_dir / "templates", dirs_exist_ok=True)


def _write_timer(task_dir: Path, hours: float) -> None:
    seconds = int(hours * 3600)
    end_ts = int(datetime.now().timestamp()) + seconds
    timer = task_dir / "timer.sh"
    timer.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            END_TS={end_ts}
            NOW=$(date +%s)
            REM=$((END_TS - NOW))
            if [ "$REM" -le 0 ]; then
              echo "expired"
              exit 0
            fi
            printf '%02d:%02d:%02d\\n' $((REM / 3600)) $(((REM % 3600) / 60)) $((REM % 60))
            """
        ),
        encoding="utf-8",
    )
    timer.chmod(0o755)


def _generate_prompt(args: argparse.Namespace, ptb_root: Path) -> str:
    cmd = [
        sys.executable,
        str(ptb_root / "src" / "eval" / "general" / "get_prompt.py"),
        "--model-to-train",
        args.model_to_train,
        "--benchmark-id",
        args.eval,
        "--num-hours",
        str(args.hours),
        "--num-gpus",
        str(args.num_gpus),
        "--agent",
        "codex",
    ]
    env = os.environ.copy()
    if args.ptb_prompt:
        env["POST_TRAIN_BENCH_PROMPT"] = args.ptb_prompt
    proc = _run(cmd, cwd=ptb_root, env=env, check=True)
    prompt = proc.stdout or ""
    if args.native_note:
        prompt += textwrap.dedent(
            f"""

            ## Native runner note
            This run is executed by EvoMaster's codex4ptb native runner, not by the
            official Apptainer wrapper. The Python virtual environment is already
            first in PATH. Use `python`, `pip`, and `uv` normally. HF_HOME points to
            the runner cache. The visible GPU hardware may differ from the official
            H100 setup; use `nvidia-smi` if you need exact details. Keep your work in
            the current directory and store the best trained model in `final_model`.

            GPU isolation is strict: this run has already been pinned to exactly
            {args.num_gpus} GPU(s) by the runner through CUDA_VISIBLE_DEVICES.
            Do not override CUDA_VISIBLE_DEVICES to `0` or to a comma-separated
            list, and do not launch multi-GPU torchrun/accelerate jobs unless the
            prompt explicitly gives you more than one GPU. Prefer omitting
            CUDA_VISIBLE_DEVICES in commands so the inherited runner value is used.

            TMPDIR/VLLM_RPC_BASE_PATH point to a short per-run directory under
            /tmp. Keep that setting for vLLM commands; Unix IPC socket paths fail
            when TMPDIR is inside the long run directory.

            Do not stop early because of a self-imposed token, context, or planning
            budget. The real budget is the wall-clock time shown by `./timer.sh`.
            If an ambitious plan is too large, immediately switch to a minimal
            viable training or packaging path. You must leave a loadable full
            Hugging Face model directory at `final_model/`; if no improved model
            can be trained in time, package the original base model or the best
            partial candidate so the benchmark evaluation can still run.
            """
        )
    return prompt.strip() + "\n"


def _copy_codex_home(source: Path, run_home: Path) -> None:
    dst = run_home / ".codex"
    if dst.exists() or not source.exists():
        return
    shutil.copytree(source, dst)


def _build_codex_cmd(args: argparse.Namespace, task_dir: Path, final_answer_path: Path) -> list[str]:
    cmd = [args.codex_bin]
    if args.search:
        cmd.append("--search")
    cmd.extend(
        [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--cd",
            str(task_dir),
            "--sandbox",
            args.sandbox,
            "--output-last-message",
            str(final_answer_path),
            "-c",
            f'approval_policy="{args.approval_policy}"',
        ]
    )
    if args.codex_model:
        cmd.extend(["--model", args.codex_model])
    if args.codex_profile:
        cmd.extend(["--profile", args.codex_profile])
    for item in args.codex_config or []:
        cmd.extend(["-c", item])
    if args.ignore_user_config:
        cmd.append("--ignore-user-config")
    if args.ignore_rules:
        cmd.append("--ignore-rules")
    cmd.append("-")
    return cmd


def _run_codex(args: argparse.Namespace, run_dir: Path, task_dir: Path, prompt: str, env: dict[str, str]) -> dict[str, Any]:
    stdout_path = run_dir / "codex.stdout.jsonl"
    stderr_path = run_dir / "codex.stderr.log"
    final_answer_path = run_dir / "final_answer.txt"
    cmd = _build_codex_cmd(args, task_dir, final_answer_path)
    timeout_sec = int(args.hours * 3600 + args.codex_extra_timeout_sec)
    started = datetime.now()
    error = None
    tb = None
    returncode = None
    try:
        proc = _run(
            cmd,
            cwd=task_dir,
            env=env,
            input_text=prompt,
            stdout=stdout_path,
            stderr=stderr_path,
            timeout=timeout_sec,
            check=False,
        )
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        error = f"Codex timed out after {timeout_sec}s: {exc}"
        tb = traceback.format_exc()
    except Exception as exc:
        error = str(exc)
        tb = traceback.format_exc()

    final_answer = ""
    if final_answer_path.exists():
        final_answer = final_answer_path.read_text(encoding="utf-8", errors="replace").strip()
    events = _read_jsonl(stdout_path)
    dialogs, event_meta = _extract_dialogs_from_events(events, prompt)
    if not final_answer:
        final_answer = _extract_final_answer(dialogs)
        final_answer_path.write_text(final_answer, encoding="utf-8")

    result = {
        "task_id": f"{args.eval}_{_safe_name(args.model_to_train)}",
        "status": "completed" if returncode == 0 and error is None else "failed",
        "returncode": returncode,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "final_answer_path": str(final_answer_path),
        "final_answer": final_answer,
        "meta": {
            "command": cmd,
            "event_meta": event_meta,
            "task_dir": str(task_dir),
            "run_dir": str(run_dir),
            "model": args.codex_model,
            "profile": args.codex_profile,
            "sandbox": args.sandbox,
            "approval_policy": args.approval_policy,
            "search": args.search,
        },
    }
    if error:
        result["error"] = error
    if tb:
        result["traceback"] = tb
    trajectory = _build_trajectory(str(result["task_id"]), dialogs, result)
    _write_json(run_dir / "dialogs.json", dialogs)
    _write_json(run_dir / "trajectory.json", trajectory)
    _write_json(run_dir / "codex_meta.json", result["meta"])
    _write_json(run_dir / "codex_result.json", result)
    return result


def _eval_cmd(args: argparse.Namespace, model_path: str, json_output: Path) -> list[str]:
    cmd = [
        str(_venv_python(_state_paths(args)["venv_dir"])),
        "evaluate.py",
        "--model-path",
        model_path,
        "--templates-dir",
        str(Path(args.ptb_root).expanduser().resolve() / "src" / "eval" / "templates"),
        "--limit",
        str(args.eval_limit),
        "--json-output-file",
        str(json_output),
    ]
    if args.max_tokens is not None:
        if args.eval in {"arenahardwriting", "healthbench"}:
            cmd.extend(["--max-new-tokens", str(args.max_tokens)])
        else:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
    return cmd


def _run_eval(args: argparse.Namespace, run_dir: Path, model_path: str, output_name: str, env: dict[str, str]) -> dict[str, Any]:
    ptb_root = Path(args.ptb_root).expanduser().resolve()
    eval_dir = ptb_root / "src" / "eval" / "tasks" / args.eval
    json_output = run_dir / output_name
    stdout = run_dir / f"{Path(output_name).stem}.stdout.log"
    stderr = run_dir / f"{Path(output_name).stem}.stderr.log"
    cmd = _eval_cmd(args, model_path, json_output)
    result = {
        "model_path": model_path,
        "metrics_path": str(json_output),
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "command": cmd,
    }
    try:
        proc = _run(cmd, cwd=eval_dir, env=env, stdout=stdout, stderr=stderr, timeout=args.eval_timeout_sec, check=False)
        result["returncode"] = proc.returncode
        result["status"] = "completed" if proc.returncode == 0 and json_output.exists() else "failed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    _write_json(run_dir / f"{Path(output_name).stem}_result.json", result)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    ptb_root = Path(args.ptb_root).expanduser().resolve()
    if args.eval not in BENCHMARKS:
        raise ValueError(f"Unknown eval `{args.eval}`. Known: {', '.join(BENCHMARKS)}")
    paths = _state_paths(args)
    _ensure_dirs(paths)
    if not args.dry_run and not _venv_python(paths["venv_dir"]).exists():
        raise RuntimeError(f"venv missing: {paths['venv_dir']}. Run setup-env first.")

    run_name = args.run_name or f"codex4ptb_{_timestamp()}_{args.eval}_{_safe_name(args.model_to_train)}"
    run_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.runs_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    task_dir = run_dir / "task"
    run_home = run_dir / "home"
    run_tmp = run_dir / "tmp"
    short_tmp = Path("/tmp") / f"c4ptb_{os.getpid()}_{hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]}"
    run_home.mkdir(parents=True, exist_ok=True)
    run_tmp.mkdir(parents=True, exist_ok=True)
    short_tmp.mkdir(parents=True, exist_ok=True)

    _prepare_task_dir(ptb_root, args.eval, task_dir)
    _write_timer(task_dir, args.hours)
    _copy_codex_home(Path(args.codex_home_source).expanduser(), run_home)

    prompt = _generate_prompt(args, ptb_root)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    base_env = _base_env(args)
    if args.dry_run:
        agent_venv = run_dir / "venv_agent"
        agent_env = dict(base_env)
    else:
        agent_venv, agent_env = _create_agent_venv(args, run_dir, base_env)

    for env in (base_env, agent_env):
        env["HOME"] = str(run_home)
        env["CODEX4PTB_RUN_TMP"] = str(run_tmp)
        env["TMPDIR"] = str(short_tmp)
        env["TMP"] = str(short_tmp)
        env["TEMP"] = str(short_tmp)
        env["VLLM_RPC_BASE_PATH"] = str(short_tmp)
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_config = {
        "ptb_root": str(ptb_root),
        "eval": args.eval,
        "model_to_train": args.model_to_train,
        "codex_model": args.codex_model,
        "hours": args.hours,
        "eval_limit": args.eval_limit,
        "run_dir": str(run_dir),
        "task_dir": str(task_dir),
        "run_tmp": str(run_tmp),
        "short_tmp": str(short_tmp),
        "base_venv_dir": str(paths["venv_dir"]),
        "agent_venv_dir": str(agent_venv),
        "hf_home": str(paths["hf_home"]),
        "proxy": args.proxy,
        "cuda_visible_devices": args.cuda_visible_devices,
        "native_note": args.native_note,
        "search": args.search,
    }
    _write_json(run_dir / "run_config.json", run_config)

    print(f"Run dir: {run_dir}", flush=True)
    if args.dry_run:
        summary = {
            "status": "dry_run",
            "run_dir": str(run_dir),
            "prompt_path": str(run_dir / "prompt.txt"),
            "task_dir": str(task_dir),
            "generated_at": datetime.now().isoformat(),
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return 0

    codex_result = _run_codex(args, run_dir, task_dir, prompt, agent_env)

    final_model = task_dir / "final_model"
    eval_result = {"status": "skipped", "reason": "final_model missing", "model_path": str(final_model)}
    if not args.skip_eval and final_model.is_dir():
        eval_result = _run_eval(args, run_dir, str(final_model), "metrics.json", base_env)

    baseline_result = {"status": "skipped"}
    if args.eval_baseline:
        baseline_result = _run_eval(args, run_dir, args.model_to_train, "baseline_metrics.json", base_env)

    summary = {
        "status": "completed" if codex_result.get("status") == "completed" else "failed",
        "run_dir": str(run_dir),
        "codex": codex_result,
        "eval": eval_result,
        "baseline_eval": baseline_result,
        "final_model_exists": final_model.is_dir(),
        "generated_at": datetime.now().isoformat(),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if summary["status"] == "completed" else 1


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ptb-root", default=str(DEFAULT_PTB_ROOT), help="PostTrainBench checkout path")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), help="Directory for venv/cache state")
    parser.add_argument("--venv", default=None, help="Override venv directory")
    parser.add_argument("--hf-home", default=None, help="Override HuggingFace cache directory")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Default root for run outputs")
    parser.add_argument("--proxy", default=DEFAULT_PROXY, help="HTTP/HTTPS/ALL proxy, or 'none'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Codex runner for PostTrainBench")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local prerequisites")
    add_common_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    setup = sub.add_parser("setup-env", help="Create venv and install PostTrainBench runtime deps")
    add_common_args(setup)
    setup.add_argument("--python", default="python3.10", help="Python executable used to create the venv")
    setup.add_argument("--flash-attn", action="store_true", help="Also install flash-attn; may compile for a long time")
    setup.set_defaults(func=cmd_setup_env)

    download = sub.add_parser("download-cache", help="Download base model snapshots into HF cache")
    add_common_args(download)
    download.add_argument("--model", action="append", help="Model repo to download; repeatable. Defaults to 4 PTB base models")
    download.add_argument(
        "--ignore-pattern",
        action="append",
        default=list(DEFAULT_DOWNLOAD_IGNORE_PATTERNS),
        help="Hugging Face snapshot ignore pattern; repeatable. Defaults skip ONNX artifacts.",
    )
    download.set_defaults(func=cmd_download_cache)

    run = sub.add_parser("run", help="Run one Codex PostTrainBench task")
    add_common_args(run)
    run.add_argument("--eval", required=True, choices=BENCHMARKS, help="PTB evaluation id")
    run.add_argument("--model-to-train", required=True, help="Base model repo/path")
    run.add_argument("--codex-model", default=None, help="Codex model override")
    run.add_argument("--codex-profile", default=None, help="Codex profile override")
    run.add_argument("--codex-bin", default="codex", help="Codex executable")
    run.add_argument("--no-search", dest="search", action="store_false", help="Do not pass global `codex --search`")
    run.set_defaults(search=True)
    run.add_argument("--codex-home-source", default=str(Path.home() / ".codex"), help="Codex home config copied into each run")
    run.add_argument("--codex-config", action="append", default=["model_reasoning_summary=detailed"], help="Extra codex -c override")
    run.add_argument("--codex-extra-timeout-sec", type=int, default=300, help="Extra seconds beyond --hours for codex process")
    run.add_argument("--python", default="python3.10", help="Python executable used to create per-run agent venv")
    run.add_argument("--no-fresh-agent-venv", dest="fresh_agent_venv", action="store_false", help="Use base venv directly for the agent")
    run.set_defaults(fresh_agent_venv=True)
    run.add_argument("--hours", type=float, default=1.0, help="Task time budget in hours")
    run.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs exposed in prompt")
    run.add_argument("--cuda-visible-devices", default="0", help="CUDA_VISIBLE_DEVICES for agent/eval; empty to inherit")
    run.add_argument("--sandbox", default="danger-full-access", choices=("read-only", "workspace-write", "danger-full-access"))
    run.add_argument("--approval-policy", default="never", choices=("untrusted", "on-failure", "on-request", "never"))
    run.add_argument("--ignore-user-config", action="store_true")
    run.add_argument("--ignore-rules", action="store_true")
    run.add_argument("--ptb-prompt", default=None, help="POST_TRAIN_BENCH_PROMPT template base name")
    run.add_argument("--no-native-note", dest="native_note", action="store_false", help="Do not append native runner note to PTB prompt")
    run.set_defaults(native_note=True)
    run.add_argument("--output-dir", default=None, help="Exact output run directory")
    run.add_argument("--run-name", default=None, help="Run directory name under --runs-dir")
    run.add_argument("--skip-eval", action="store_true", help="Do not evaluate final_model after Codex")
    run.add_argument("--dry-run", action="store_true", help="Prepare run directory/prompt only; do not invoke Codex or eval")
    run.add_argument("--eval-baseline", action="store_true", help="Also evaluate the base model for comparison")
    run.add_argument("--eval-limit", default="20", help="Evaluation limit; use -1 for full eval")
    run.add_argument("--eval-timeout-sec", type=int, default=28800, help="Final evaluation timeout")
    run.add_argument("--max-tokens", type=int, default=None, help="Optional max tokens/new tokens for evaluate.py")
    run.set_defaults(func=cmd_run)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
