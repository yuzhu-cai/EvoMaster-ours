from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import parse_qs, urlparse

try:
    from flask import Flask, jsonify, request, send_from_directory
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean local envs.
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    send_from_directory = None  # type: ignore[assignment]

logger = logging.getLogger("trajectory_viewer")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = ROOT / "runs"
MAX_INLINE_TEXT = 250_000


def _iso_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _safe_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _read_text(path: Path, limit: int = MAX_INLINE_TEXT) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<unable to read {path.name}: {exc}>"
    if len(data) > limit:
        return data[:limit] + f"\n\n... truncated at {limit:,} characters ..."
    return data


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _flatten_messages(trajectory: Dict[str, Any]) -> list[Dict[str, Any]]:
    dialogs = _as_list(trajectory.get("dialogs"))
    if not dialogs:
        return []
    # Current trajectory files store the cumulative conversation in the last dialog.
    last_dialog = _as_dict(dialogs[-1])
    return [m for m in _as_list(last_dialog.get("messages")) if isinstance(m, dict)]


def _dialog_meta(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    dialogs = _as_list(trajectory.get("dialogs"))
    if not dialogs:
        return {}
    return _as_dict(_as_dict(dialogs[-1]).get("meta"))


def _message_role(msg: Dict[str, Any]) -> str:
    role = msg.get("role") or "unknown"
    if hasattr(role, "value"):
        return str(role.value)
    return str(role)


def _message_content(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(content)


def _message_signature(msg: Dict[str, Any]) -> tuple[Any, ...]:
    """Stable enough identity for matching saved prompt tails to step records."""
    tool_calls = []
    for call in _as_list(msg.get("tool_calls")):
        fn = _as_dict(_as_dict(call).get("function"))
        tool_calls.append((fn.get("name"), fn.get("arguments")))
    return (
        _message_role(msg),
        msg.get("name"),
        msg.get("tool_call_id"),
        _message_content(msg),
        tuple(tool_calls),
    )


def _same_message(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _message_signature(a) == _message_signature(b)


def _is_summary_message(msg: Dict[str, Any]) -> bool:
    return bool(_as_dict(msg.get("meta")).get("summary"))


def _is_compaction_request(msg: Dict[str, Any]) -> bool:
    return _message_role(msg) == "user" and _message_content(msg).strip() == "What did we do so far?"


def _is_pruned_tool_message(msg: Dict[str, Any]) -> bool:
    return _message_role(msg) == "tool" and _message_content(msg).strip() == "[Old tool output cleared]"


def _message_token_estimate(msg: Dict[str, Any]) -> int:
    total_chars = len(_message_content(msg))
    for call in _as_list(msg.get("tool_calls")):
        fn = _as_dict(_as_dict(call).get("function"))
        total_chars += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
    return max(1, total_chars // 4)


def _messages_stats(messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    role_counts = Counter(_message_role(msg) for msg in messages)
    tools = _tool_names(messages)
    return {
        "message_count": len(messages),
        "role_counts": dict(role_counts),
        "assistant_count": role_counts.get("assistant", 0),
        "tool_count": role_counts.get("tool", 0),
        "user_count": role_counts.get("user", 0),
        "system_count": role_counts.get("system", 0),
        "summary_count": sum(1 for msg in messages if _is_summary_message(msg)),
        "pruned_tool_count": sum(1 for msg in messages if _is_pruned_tool_message(msg)),
        "estimated_tokens": sum(_message_token_estimate(msg) for msg in messages),
        "top_tools": tools.most_common(8),
    }


def _entry_step_messages(entry: Dict[str, Any]) -> list[Dict[str, Any]]:
    trajectory = _as_dict(entry.get("trajectory"))
    result: list[Dict[str, Any]] = []
    for step in _as_list(trajectory.get("steps")):
        step_obj = _as_dict(step)
        assistant = _as_dict(step_obj.get("assistant_message"))
        if assistant:
            result.append(dict(assistant))
        for tool_response in _as_list(step_obj.get("tool_responses")):
            tool_obj = _as_dict(tool_response)
            if tool_obj:
                result.append(dict(tool_obj))
    return result


def _leading_context_messages(data: list[Any]) -> list[Dict[str, Any]]:
    """Return the stable task prefix before the first assistant/tool turn."""
    for entry in data:
        messages = _flatten_messages(_as_dict(_as_dict(entry).get("trajectory")))
        if not messages:
            continue
        prefix: list[Dict[str, Any]] = []
        for msg in messages:
            if _message_role(msg) in {"assistant", "tool"}:
                break
            if not _is_summary_message(msg) and not _is_compaction_request(msg):
                prefix.append(dict(msg))
        if prefix:
            return prefix
    return []


def _strip_current_step_from_tail(
    messages: list[Dict[str, Any]],
    step_messages: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Best-effort recovery of the prompt before the selected step ran.

    The writer receives a dialog object by reference, so many saved prompt
    snapshots already include the assistant/tool messages of the current step.
    Strip that tail when it exactly matches the separately stored step record.
    """
    if not messages or not step_messages or len(step_messages) > len(messages):
        return list(messages)
    tail = messages[-len(step_messages):]
    if all(_same_message(a, b) for a, b in zip(tail, step_messages)):
        return list(messages[:-len(step_messages)])
    return list(messages)


def _append_new_prompt_messages(
    reconstructed: list[Dict[str, Any]],
    prompt_before_step: list[Dict[str, Any]],
) -> None:
    """Append non-step messages added between rounds, usually iteration prompts.

    Reconstructed full history is based on per-step response records. Those
    records do not include user continuation prompts, so we align the saved
    compressed prompt against the reconstructed suffix and append only the new
    prompt tail after that suffix. Summary compaction artifacts are skipped so
    this view remains the best available "before compression" conversation.
    """
    if not reconstructed or not prompt_before_step:
        return

    max_k = min(len(reconstructed), len(prompt_before_step), 24)
    boundary: int | None = None
    for k in range(max_k, 0, -1):
        suffix = [_message_signature(msg) for msg in reconstructed[-k:]]
        for end in range(len(prompt_before_step), k - 1, -1):
            window = [_message_signature(msg) for msg in prompt_before_step[end - k:end]]
            if window == suffix:
                boundary = end
                break
        if boundary is not None:
            break

    if boundary is None:
        return

    for msg in prompt_before_step[boundary:]:
        # These two messages are synthetic compaction artifacts, not the
        # original conversation before compression.
        if _is_summary_message(msg) or _is_compaction_request(msg):
            continue
        if reconstructed and _same_message(reconstructed[-1], msg):
            continue
        reconstructed.append(dict(msg))


def _reconstruct_full_messages(data: list[Any], index: int) -> list[Dict[str, Any]]:
    """Reconstruct a best-effort uncompressed conversation through an entry.

    The saved dialog in a trajectory entry can be summarized/truncated. However,
    each entry also stores the assistant response and tool responses for that
    step. Appending those step records gives a much clearer pre-compaction view.
    """
    reconstructed = _leading_context_messages(data[: index + 1])
    for raw_entry in data[: index + 1]:
        entry = _as_dict(raw_entry)
        compressed = _flatten_messages(_as_dict(entry.get("trajectory")))
        step_messages = _entry_step_messages(entry)
        prompt_before_step = _strip_current_step_from_tail(compressed, step_messages)
        _append_new_prompt_messages(reconstructed, prompt_before_step)
        for msg in step_messages:
            if reconstructed and _same_message(reconstructed[-1], msg):
                continue
            reconstructed.append(dict(msg))
    return reconstructed


def _entry_views(data: list[Any], index: int, include_full: bool = False) -> Dict[str, list[Dict[str, Any]]]:
    entry = _as_dict(data[index])
    compressed = _flatten_messages(_as_dict(entry.get("trajectory")))
    step_messages = _entry_step_messages(entry)
    prompt = _strip_current_step_from_tail(compressed, step_messages)
    full = _reconstruct_full_messages(data, index) if include_full else []
    if include_full and not full:
        full = compressed
    return {
        "compressed": compressed,
        "prompt": prompt,
        "step": step_messages,
        "full": full,
    }


def _latest_assistant_usage(messages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    for msg in reversed(list(messages)):
        if msg.get("role") != "assistant":
            continue
        meta = _as_dict(msg.get("meta"))
        usage = _as_dict(meta.get("usage"))
        if usage:
            return usage
    return {}


def _cumulative_tokens(messages: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        usage = _as_dict(_as_dict(msg.get("meta")).get("usage"))
        try:
            total += int(usage.get("total_tokens") or 0)
        except Exception:
            pass
    return total


def _tool_names(messages: Iterable[Dict[str, Any]]) -> Counter[str]:
    names: Counter[str] = Counter()
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name"):
            names[str(msg.get("name"))] += 1
        for call in _as_list(msg.get("tool_calls")):
            fn = _as_dict(call).get("function")
            name = _as_dict(fn).get("name")
            if name:
                names[str(name)] += 1
    return names


def _message_excerpt(messages: list[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        content = msg.get("content")
        if content is None and msg.get("tool_calls"):
            calls = []
            for call in _as_list(msg.get("tool_calls"))[:3]:
                fn = _as_dict(_as_dict(call).get("function"))
                if fn.get("name"):
                    calls.append(str(fn.get("name")))
            if calls:
                return "Tool calls: " + ", ".join(calls)
        if isinstance(content, str) and content.strip():
            compact = " ".join(content.split())
            return compact[:220]
    return "No textual message content"


def _best_meta_for(run_dir: Path, trajectory_id: str) -> Dict[str, Any]:
    safe_id = Path(trajectory_id).name
    candidates = [
        run_dir / "workspaces" / trajectory_id / "artifacts" / "best_meta.json",
        run_dir / "workspaces" / safe_id / "artifacts" / "best_meta.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _safe_json(candidate, {}) or {}
    return {}


def _resolve_under(base: Path, child: str) -> Path:
    if not child or Path(child).is_absolute():
        raise ValueError("invalid path")
    path = (base / child).resolve()
    base_resolved = base.resolve()
    if path != base_resolved and base_resolved not in path.parents:
        raise ValueError("path escapes base")
    return path


def _experiment_dir(runs_root: Path, experiment: str) -> Path:
    if "/" in experiment or "\\" in experiment or experiment in {"", ".", ".."}:
        raise ValueError("invalid experiment")
    run_dir = _resolve_under(runs_root, experiment)
    if not run_dir.is_dir():
        raise FileNotFoundError(experiment)
    return run_dir


def _trajectory_file(run_dir: Path, trajectory_id: str) -> Path:
    traj_root = run_dir / "trajectories"
    if trajectory_id in {"trajectory", "trajectory.json"}:
        path = traj_root / "trajectory.json"
    elif trajectory_id.endswith(".json"):
        path = _resolve_under(traj_root, trajectory_id)
    else:
        path = _resolve_under(traj_root, trajectory_id) / "trajectory.json"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(trajectory_id)
    return path


@lru_cache(maxsize=32)
def _load_trajectory_cached(path_str: str, mtime_ns: int, size: int) -> Any:
    del mtime_ns, size
    with Path(path_str).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_trajectory(path: Path) -> list[Any]:
    last_error: Exception | None = None
    for attempt in range(4):
        stat = path.stat()
        try:
            data = _load_trajectory_cached(str(path), stat.st_mtime_ns, stat.st_size)
            break
        except json.JSONDecodeError as exc:
            # Live runs rewrite trajectory.json in place. A viewer request can
            # catch the file between truncate/write, so retry briefly.
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(0.15)
    else:  # pragma: no cover - loop always breaks or raises.
        raise last_error or RuntimeError("failed to load trajectory")
    return data if isinstance(data, list) else [data]


def _summarize_entry(entry: Any, index: int) -> Dict[str, Any]:
    obj = _as_dict(entry)
    trajectory = _as_dict(obj.get("trajectory"))
    messages = _flatten_messages(trajectory)
    step_messages = _entry_step_messages(obj)
    dialog_meta = _dialog_meta(trajectory)
    role_counts = Counter(str(m.get("role") or "unknown") for m in messages)
    usage = _latest_assistant_usage(messages)
    tools = _tool_names(messages)
    step_tools = _tool_names(step_messages)
    return {
        "index": index,
        "task_id": obj.get("task_id") or trajectory.get("task_id") or f"entry-{index}",
        "status": obj.get("status") or trajectory.get("status") or "unknown",
        "steps": obj.get("steps") or trajectory.get("step") or len(_as_list(trajectory.get("steps"))),
        "agent_name": trajectory.get("agent_name"),
        "message_count": len(messages),
        "step_message_count": len(step_messages),
        "assistant_count": role_counts.get("assistant", 0),
        "tool_count": role_counts.get("tool", 0),
        "user_count": role_counts.get("user", 0),
        "latest_tokens": usage.get("total_tokens"),
        "cumulative_tokens": _cumulative_tokens(messages),
        "top_tools": tools.most_common(5),
        "step_tools": step_tools.most_common(5),
        "compacted": bool(dialog_meta.get("truncated") or dialog_meta.get("pruned")),
        "compaction_strategy": dialog_meta.get("strategy") or ("prune" if dialog_meta.get("pruned") else ""),
        "summary_count": sum(1 for msg in messages if _is_summary_message(msg)),
        "pruned_tool_count": sum(1 for msg in messages if _is_pruned_tool_message(msg)),
        "excerpt": _message_excerpt(messages),
    }


def _summarize_trajectory(run_dir: Path, trajectory_id: str, path: Path, load: bool = True) -> Dict[str, Any]:
    stat = path.stat()
    summary: Dict[str, Any] = {
        "id": trajectory_id,
        "name": Path(trajectory_id).name,
        "relative_path": str(path.relative_to(run_dir)),
        "size": stat.st_size,
        "size_human": _human_bytes(stat.st_size),
        "modified_at": _iso_from_mtime(path),
        "best_meta": _best_meta_for(run_dir, trajectory_id),
    }
    if not load:
        return summary
    try:
        data = _load_trajectory(path)
    except Exception as exc:
        summary.update({"error": str(exc), "entry_count": 0})
        return summary
    entries = [_summarize_entry(entry, index) for index, entry in enumerate(data)]
    last = entries[-1] if entries else {}
    summary.update(
        {
            "entry_count": len(data),
            "last_status": last.get("status"),
            "last_steps": last.get("steps"),
            "last_message_count": last.get("message_count"),
            "last_cumulative_tokens": last.get("cumulative_tokens"),
            "entries": entries,
        }
    )
    return summary


def _list_trajectory_files(run_dir: Path) -> list[tuple[str, Path]]:
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return []
    found: list[tuple[str, Path]] = []
    direct = traj_root / "trajectory.json"
    if direct.is_file():
        found.append(("trajectory", direct))
    for path in sorted(traj_root.rglob("trajectory.json")):
        if path == direct:
            continue
        rel_parent = path.parent.relative_to(traj_root)
        found.append((rel_parent.as_posix(), path))
    return found


def _experiment_summary(run_dir: Path, runs_root: Path) -> Dict[str, Any]:
    traj_files = _list_trajectory_files(run_dir)
    best_scores = []
    for trajectory_id, _ in traj_files:
        meta = _best_meta_for(run_dir, trajectory_id)
        if meta:
            best_scores.append(
                {
                    "trajectory": trajectory_id,
                    "metric": meta.get("metric"),
                    "score": meta.get("score"),
                    "direction": meta.get("direction"),
                    "experiment_name": meta.get("experiment_name"),
                }
            )
    log_dir = run_dir / "logs"
    logs = sorted([p.name for p in log_dir.glob("*.log")]) if log_dir.is_dir() else []
    config = run_dir / "config.yaml"
    return {
        "id": run_dir.name,
        "path": str(run_dir),
        "relative_path": str(run_dir.relative_to(runs_root)),
        "modified_at": _iso_from_mtime(run_dir),
        "trajectory_count": len(traj_files),
        "has_config": config.is_file(),
        "has_runner_log": (run_dir / "runner.log").is_file(),
        "logs": logs,
        "best_scores": best_scores,
    }


def _experiments_payload(runs_root: Path) -> Dict[str, Any]:
    if not runs_root.exists():
        return {"runs_root": str(runs_root), "experiments": []}
    experiments = []
    for child in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if child.is_dir():
            experiments.append(_experiment_summary(child, runs_root))
    return {"runs_root": str(runs_root), "experiments": experiments}


def _experiment_payload(runs_root: Path, experiment: str) -> Dict[str, Any]:
    run_dir = _experiment_dir(runs_root, experiment)
    payload = _experiment_summary(run_dir, runs_root)
    config_path = run_dir / "config.yaml"
    runner_log = run_dir / "runner.log"
    payload["config_text"] = _read_text(config_path, 120_000) if config_path.is_file() else ""
    payload["runner_log_tail"] = _read_text(runner_log, 120_000)[-120_000:] if runner_log.is_file() else ""
    return payload


def _trajectories_payload(runs_root: Path, experiment: str) -> Dict[str, Any]:
    run_dir = _experiment_dir(runs_root, experiment)
    trajectories = [
        _summarize_trajectory(run_dir, trajectory_id, path, load=True)
        for trajectory_id, path in _list_trajectory_files(run_dir)
    ]
    return {"experiment": experiment, "trajectories": trajectories}


def _trajectory_payload(runs_root: Path, experiment: str, trajectory_id: str) -> Dict[str, Any]:
    run_dir = _experiment_dir(runs_root, experiment)
    path = _trajectory_file(run_dir, trajectory_id)
    return _summarize_trajectory(run_dir, trajectory_id, path, load=True)


def _view_descriptor(name: str, messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {
        "step": "当前步输出",
        "prompt": "请求前上下文",
        "compressed": "压缩后轨迹",
        "full": "压缩前重建",
    }
    descriptions = {
        "step": "本轮 assistant 响应和 tool 返回，最适合快速看 agent 做了什么。",
        "prompt": "尽力从保存快照中剥离当前步输出，接近本轮 LLM 调用前的上下文。",
        "compressed": "trajectory.json 中保存的上下文快照；如果发生 summary/sliding-window，这就是压缩后的轨迹。",
        "full": "从每一步 assistant/tool 记录重新拼出的完整历史，用来对照压缩前信息。",
    }
    return {
        "key": name,
        "label": labels.get(name, name),
        "description": descriptions.get(name, ""),
        "available": True,
        **_messages_stats(messages),
    }


def _unavailable_view_descriptor(name: str) -> Dict[str, Any]:
    labels = {
        "full": "压缩前重建",
    }
    return {
        "key": name,
        "label": labels.get(name, name),
        "description": "切到“压缩前重建”或“压缩对比”时再加载完整历史，避免默认打开每一步都很慢。",
        "available": False,
        "message_count": None,
        "role_counts": {},
        "assistant_count": 0,
        "tool_count": 0,
        "user_count": 0,
        "system_count": 0,
        "summary_count": 0,
        "pruned_tool_count": 0,
        "estimated_tokens": None,
        "top_tools": [],
    }


def _normalize_messages(messages: list[Dict[str, Any]], source: str) -> list[Dict[str, Any]]:
    return [_normalize_message(msg, i, source=source) for i, msg in enumerate(messages)]


def _entry_payload(
    runs_root: Path,
    experiment: str,
    trajectory_id: str,
    index: int,
    view: str = "step",
) -> Dict[str, Any]:
    run_dir = _experiment_dir(runs_root, experiment)
    path = _trajectory_file(run_dir, trajectory_id)
    data = _load_trajectory(path)
    if not data:
        raise FileNotFoundError("empty trajectory")
    if index < 0:
        index = len(data) + index
    if index < 0 or index >= len(data):
        raise IndexError("index out of range")
    entry = _as_dict(data[index])
    trajectory = _as_dict(entry.get("trajectory"))
    view = view if view in {"step", "prompt", "compressed", "full", "compare"} else "step"
    include_full = view in {"full", "compare"}
    views = _entry_views(data, index, include_full=include_full)
    selected_view = "compressed" if view == "compare" else view
    selected_messages = _normalize_messages(views[selected_view], selected_view)
    role_counts = Counter(m.get("role") for m in selected_messages)
    dialog_meta = _dialog_meta(trajectory)
    view_stats = {
        name: (
            _view_descriptor(name, messages)
            if name != "full" or include_full
            else _unavailable_view_descriptor(name)
        )
        for name, messages in views.items()
    }
    message_delta = None
    token_delta = None
    if include_full:
        message_delta = view_stats["full"]["message_count"] - view_stats["compressed"]["message_count"]
        token_delta = view_stats["full"]["estimated_tokens"] - view_stats["compressed"]["estimated_tokens"]
    compression = {
        "truncated": bool(dialog_meta.get("truncated")),
        "pruned": bool(dialog_meta.get("pruned")),
        "strategy": dialog_meta.get("strategy"),
        "message_delta": message_delta,
        "token_delta_estimate": token_delta,
        "summary_count": view_stats["compressed"]["summary_count"],
        "pruned_tool_count": view_stats["compressed"]["pruned_tool_count"],
    }
    compare = None
    if view == "compare":
        compare = {
            "before": {
                "key": "full",
                "label": view_stats["full"]["label"],
                "description": view_stats["full"]["description"],
                "stats": view_stats["full"],
                "messages": _normalize_messages(views["full"], "full"),
            },
            "after": {
                "key": "compressed",
                "label": view_stats["compressed"]["label"],
                "description": view_stats["compressed"]["description"],
                "stats": view_stats["compressed"],
                "messages": _normalize_messages(views["compressed"], "compressed"),
            },
        }
    return {
        "experiment": experiment,
        "trajectory": trajectory_id,
        "index": index,
        "entry_count": len(data),
        "view": view,
        "entry": {k: v for k, v in entry.items() if k != "trajectory"},
        "trajectory_meta": {
            k: v
            for k, v in trajectory.items()
            if k not in {"dialogs", "steps"}
        },
        "summary": _summarize_entry(entry, index),
        "role_counts": dict(role_counts),
        "views": view_stats,
        "compression": compression,
        "messages": selected_messages,
        "compare": compare,
        "steps": _as_list(trajectory.get("steps")),
    }


def create_app(runs_root: Path = DEFAULT_RUNS_ROOT) -> Flask:
    if Flask is None:
        raise RuntimeError("Flask is not installed. Use main() for the built-in stdlib server fallback.")
    runs_root = runs_root.resolve()
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "runs_root": str(runs_root)})

    @app.get("/api/experiments")
    def api_experiments():
        return jsonify(_experiments_payload(runs_root))

    @app.get("/api/experiment")
    def api_experiment():
        experiment = request.args.get("experiment", "")
        try:
            return jsonify(_experiment_payload(runs_root, experiment))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/trajectories")
    def api_trajectories():
        experiment = request.args.get("experiment", "")
        try:
            return jsonify(_trajectories_payload(runs_root, experiment))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/trajectory")
    def api_trajectory():
        experiment = request.args.get("experiment", "")
        trajectory_id = request.args.get("trajectory", "")
        try:
            return jsonify(_trajectory_payload(runs_root, experiment, trajectory_id))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            logger.exception("failed to load trajectory")
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/entry")
    def api_entry():
        experiment = request.args.get("experiment", "")
        trajectory_id = request.args.get("trajectory", "")
        view = request.args.get("view", "step")
        try:
            index = int(request.args.get("index", "-1"))
        except ValueError:
            return jsonify({"error": "index must be an integer"}), 400
        try:
            return jsonify(_entry_payload(runs_root, experiment, trajectory_id, index, view=view))
        except (ValueError, FileNotFoundError, IndexError) as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            logger.exception("failed to load entry")
            return jsonify({"error": str(exc)}), 500

    return app


def _parse_tool_args(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _normalize_message(msg: Dict[str, Any], index: int, source: str = "compressed") -> Dict[str, Any]:
    meta = _as_dict(msg.get("meta"))
    tool_calls = []
    for call in _as_list(msg.get("tool_calls")):
        call_obj = _as_dict(call)
        fn = _as_dict(call_obj.get("function"))
        tool_calls.append(
            {
                "id": call_obj.get("id"),
                "type": call_obj.get("type"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
                "parsed_arguments": _parse_tool_args(fn.get("arguments")),
            }
        )
    return {
        "index": index,
        "role": _message_role(msg),
        "source": source,
        "name": msg.get("name"),
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content"),
        "tool_call_id": msg.get("tool_call_id"),
        "tool_calls": tool_calls,
        "meta": meta,
        "is_summary": _is_summary_message(msg),
        "is_pruned_tool": _is_pruned_tool_message(msg),
        "token_estimate": _message_token_estimate(msg),
    }


def _write_json(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _write_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        _write_json(handler, {"error": "file not found"}, HTTPStatus.NOT_FOUND)
        return
    body = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _run_stdlib_server(runs_root: Path, host: str, port: int) -> None:
    static_dir = Path(__file__).resolve().parent / "static"

    class TrajectoryHandler(BaseHTTPRequestHandler):
        server_version = "TrajectoryViewer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name.
            parsed = urlparse(self.path)
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/":
                    _write_file(self, static_dir / "index.html")
                    return
                if parsed.path.startswith("/static/"):
                    rel_path = parsed.path.removeprefix("/static/")
                    _write_file(self, _resolve_under(static_dir, rel_path))
                    return
                if parsed.path == "/health":
                    _write_json(self, {"ok": True, "runs_root": str(runs_root)})
                    return
                if parsed.path == "/api/experiments":
                    _write_json(self, _experiments_payload(runs_root))
                    return
                if parsed.path == "/api/experiment":
                    _write_json(self, _experiment_payload(runs_root, query.get("experiment", "")))
                    return
                if parsed.path == "/api/trajectories":
                    _write_json(self, _trajectories_payload(runs_root, query.get("experiment", "")))
                    return
                if parsed.path == "/api/trajectory":
                    _write_json(
                        self,
                        _trajectory_payload(
                            runs_root,
                            query.get("experiment", ""),
                            query.get("trajectory", ""),
                        ),
                    )
                    return
                if parsed.path == "/api/entry":
                    try:
                        index = int(query.get("index", "-1"))
                    except ValueError:
                        _write_json(self, {"error": "index must be an integer"}, HTTPStatus.BAD_REQUEST)
                        return
                    _write_json(
                        self,
                        _entry_payload(
                            runs_root,
                            query.get("experiment", ""),
                            query.get("trajectory", ""),
                            index,
                            view=query.get("view", "step"),
                        ),
                    )
                    return
                _write_json(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, FileNotFoundError, IndexError) as exc:
                _write_json(self, {"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                logger.exception("request failed")
                _write_json(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    server = ThreadingHTTPServer((host, port), TrajectoryHandler)
    logger.info("trajectory viewer serving runs_root=%s", runs_root)
    logger.info("open: http://%s:%s/", host, port)
    server.serve_forever()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("trajectory_viewer")
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("EVOMASTER_RUNS_ROOT", str(DEFAULT_RUNS_ROOT)),
        help="Directory that contains EvoMaster run directories.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8777, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        raise SystemExit(f"runs root not found: {runs_root}")
    if Flask is None:
        logger.info("Flask is not installed; using built-in stdlib HTTP server.")
        _run_stdlib_server(runs_root, args.host, args.port)
    else:
        app = create_app(runs_root)
        logger.info("trajectory viewer serving runs_root=%s", runs_root)
        logger.info("open: http://%s:%s/", args.host, args.port)
        app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
