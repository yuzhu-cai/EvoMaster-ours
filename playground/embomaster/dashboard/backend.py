"""Backend indexers for the EmboMaster dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROUND_CONTEXT_PATTERNS = {
    "task_id": re.compile(r"Task ID:\s*([^\n]+)"),
    "task_type": re.compile(r"Task Type:\s*([^\n]+)"),
    "workspace_id": re.compile(r"Workspace ID:\s*([^\n]+)"),
    "parent_workspace_id": re.compile(r"Parent Workspace ID:\s*([^\n]+)"),
    "workspace_codebase_path": re.compile(r"Workspace Codebase Path:\s*([^\n]+)"),
    "workspace_source_type": re.compile(r"Workspace Source Type:\s*([^\n]+)"),
    "workspace_large_dirs_count": re.compile(r"Workspace Large Dir Mount Count:\s*(\d+)"),
}

OUTPUT_DIR_NAMES = {
    "eval_result",
    "submission",
    "run_results",
    "output",
    "outputs",
    "checkpoints",
    "act_ckpt",
}

TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
    ".csv",
    ".tsv",
    ".py",
    ".sh",
    ".html",
    ".js",
    ".css",
    ".xml",
}


def _iso_time_from_ns(mtime_ns: int) -> str:
    if mtime_ns <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime_ns / 1_000_000_000))


def _iso_time_from_ts(ts: float) -> str:
    if ts <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated] ..."


def _as_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _safe_messages(dialog: dict[str, Any]) -> list[dict[str, Any]]:
    messages = dialog.get("messages", [])
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    return []


def _safe_steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    steps = traj.get("steps", [])
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]
    return []


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _extract_finish_message(tool_calls: list[dict[str, Any]]) -> str:
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if not isinstance(func, dict):
            continue
        if str(func.get("name", "")) != "finish":
            continue
        args_raw = func.get("arguments", "")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                return args_raw
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return ""
        return str(args.get("message", "") or "")
    return ""


def _parse_debug_status(content: str) -> tuple[str, int | None]:
    text = str(content or "")
    match = re.search(
        r"\[debug_test\]\s+(\w+).*?exit_code=(-?\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).lower(), _as_int(match.group(2), None)
    lowered = text.lower()
    if "success" in lowered:
        return "success", None
    if "failed" in lowered or "error" in lowered:
        return "failed", None
    return "unknown", None


def _split_debug_streams(content: str) -> tuple[str, str]:
    text = str(content or "").replace("\r\n", "\n")
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("[debug_test]"):
        body = "\n".join(lines[1:]).strip()
    else:
        body = text.strip()
    if not body:
        return "", ""
    marker_patterns = [
        r"(?is)^\s*\[?stdout\]?\s*:?\s*\n(.*?)\n\s*\[?stderr\]?\s*:?\s*\n(.*)$",
        r"(?is)^\s*stdout\s*=\s*\n(.*?)\n\s*stderr\s*=\s*\n(.*)$",
        r"(?is)^\s*stdout\s*:\s*(.*?)\n\s*stderr\s*:\s*(.*)$",
    ]
    for pattern in marker_patterns:
        match = re.match(pattern, body)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return body, ""


def _parse_feedback_block(text: str) -> dict[str, Any] | None:
    content = str(text or "")
    if "K8S Status" not in content or "Round:" not in content:
        return None
    round_match = re.search(r"Round:\s*(\d+)", content, flags=re.IGNORECASE)
    if not round_match:
        return None
    round_index = _as_int(round_match.group(1), -1)
    if round_index is None or round_index <= 0:
        return None
    status_match = re.search(r"K8S Status:\s*\n?\s*([^\n]+)", content, flags=re.IGNORECASE)
    metric_match = re.search(r"Metric:\s*\n?\s*([^\n]+)", content, flags=re.IGNORECASE)
    tail_match = re.search(
        r"K8S Log Tail:\s*\n?(.*?)(?:\n\s*Please provide improvement suggestions|\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "round_index": round_index,
        "k8s_status": status_match.group(1).strip() if status_match else "unknown",
        "metric": metric_match.group(1).strip() if metric_match else "",
        "metric_value": _as_float(metric_match.group(1), None) if metric_match else None,
        "k8s_log_tail": _clip_text(tail_match.group(1).strip(), 5000) if tail_match else "",
    }


def _parse_round_context(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() != "user":
            continue
        text = _message_text(msg)
        if "Workspace ID:" not in text:
            continue
        payload: dict[str, Any] = {}
        for key, pattern in ROUND_CONTEXT_PATTERNS.items():
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1).strip()
            if key == "workspace_large_dirs_count":
                payload[key] = _as_int(value, 0)
            else:
                payload[key] = value
        round_match = re.search(r"Round:\s*(\d+)(?:/\d+)?", text)
        if round_match:
            payload["round_index"] = _as_int(round_match.group(1), 0)
        return payload
    return {}


def _tool_call_names(assistant_message: dict[str, Any]) -> list[str]:
    tool_calls = assistant_message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if not isinstance(func, dict):
            continue
        name = str(func.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _summarize_container_state(state: Any) -> str:
    if not isinstance(state, dict):
        return "unknown"
    for key in ("running", "waiting", "terminated"):
        value = state.get(key)
        if isinstance(value, dict):
            reason = str(value.get("reason", "") or "").strip()
            return f"{key}:{reason}" if reason else key
    return "unknown"


def _build_entry_detail(raw: dict[str, Any], index: int, preview_chars: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    traj = raw.get("trajectory")
    if not isinstance(traj, dict):
        traj = {}
    dialogs = traj.get("dialogs", [])
    if not isinstance(dialogs, list):
        dialogs = []
    last_dialog = dialogs[-1] if dialogs and isinstance(dialogs[-1], dict) else {}
    messages = _safe_messages(last_dialog)
    round_context = _parse_round_context(messages)
    step_list = _safe_steps(traj)
    step_obj = step_list[-1] if step_list and isinstance(step_list[-1], dict) else {}
    assistant_message = step_obj.get("assistant_message", {})
    if not isinstance(assistant_message, dict):
        assistant_message = {}

    tool_names = _tool_call_names(assistant_message)
    finish_message = _extract_finish_message(assistant_message.get("tool_calls", []))
    assistant_text = finish_message or _message_text(assistant_message)
    assistant_preview = _clip_text(assistant_text, preview_chars)

    prompt_user = ""
    recent_prompt_messages: list[dict[str, Any]] = []
    feedback_hint: dict[str, Any] | None = None
    for msg in messages[-6:]:
        role = str(msg.get("role", "unknown")).lower()
        content = _message_text(msg)
        recent_prompt_messages.append({"role": role, "content": _clip_text(content, preview_chars)})
        if role == "user":
            prompt_user = content
            if feedback_hint is None:
                feedback_hint = _parse_feedback_block(content)

    tool_responses = step_obj.get("tool_responses", [])
    if not isinstance(tool_responses, list):
        tool_responses = []

    debug_events: list[dict[str, Any]] = []
    tool_response_items: list[dict[str, Any]] = []
    for tr in tool_responses:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name", "") or "")
        content = str(tr.get("content", "") or "")
        meta = tr.get("meta", {})
        info = {}
        if isinstance(meta, dict):
            raw_info = meta.get("info")
            if isinstance(raw_info, dict):
                info = raw_info
        item = {
            "name": name,
            "content": _clip_text(content, 14000),
            "info": info,
        }
        tool_response_items.append(item)
        if name != "debug_test":
            continue
        debug_status, exit_code_guess = _parse_debug_status(content)
        stdout_text, stderr_text = _split_debug_streams(content)
        debug_events.append(
            {
                "entry_index": index,
                "exp_index": _as_int(raw.get("exp_index"), -1),
                "step": _as_int(raw.get("steps", traj.get("step", 0)), 0),
                "agent_name": str(traj.get("agent_name", "unknown")),
                "status": debug_status,
                "exit_code": info.get("exit_code", exit_code_guess),
                "mode": str(info.get("mode", "") or ""),
                "command": str(info.get("command", "") or ""),
                "full_command": str(info.get("full_command", "") or ""),
                "pod_name": str(info.get("pod_name", "") or ""),
                "namespace": str(info.get("namespace", "default") or "default"),
                "working_dir": str(info.get("working_dir", "") or ""),
                "stdout": _clip_text(stdout_text, 16000),
                "stderr": _clip_text(stderr_text, 10000),
                "output": _clip_text(content, 16000),
            }
        )

    detail = {
        "index": index,
        "task_id": raw.get("task_id"),
        "exp_name": raw.get("exp_name"),
        "exp_index": _as_int(raw.get("exp_index"), -1),
        "status": str(raw.get("status", "running")),
        "step": _as_int(raw.get("steps", traj.get("step", 0)), 0),
        "trajectory_step": _as_int(traj.get("step"), 0),
        "agent_name": str(traj.get("agent_name", "unknown")),
        "task_type": str(round_context.get("task_type", "") or ""),
        "workspace_id": str(round_context.get("workspace_id", "") or ""),
        "parent_workspace_id": str(round_context.get("parent_workspace_id", "") or ""),
        "workspace_codebase_path": str(round_context.get("workspace_codebase_path", "") or ""),
        "workspace_source_type": str(round_context.get("workspace_source_type", "") or ""),
        "workspace_large_dirs_count": _as_int(round_context.get("workspace_large_dirs_count"), 0),
        "assistant_preview": assistant_preview,
        "assistant_text": _clip_text(assistant_text, 12000),
        "prompt_user_preview": _clip_text(prompt_user, 2200),
        "tool_names": tool_names,
        "tool_response_count": len(tool_response_items),
        "debug_test_count": len(debug_events),
        "recent_prompt_messages": recent_prompt_messages,
        "tool_responses": tool_response_items,
    }
    return detail, debug_events, feedback_hint, round_context


class TaskRunStore:
    def __init__(
        self,
        trajectory_path: Path,
        run_label: str,
        run_dir: Path | None = None,
        task_id: str | None = None,
        preview_chars: int = 220,
    ) -> None:
        self.path = trajectory_path
        self.run_label = run_label
        self.run_dir = run_dir
        self.task_id = task_id or "task_0"
        self.preview_chars = preview_chars
        self._format = "jsonl" if trajectory_path.suffix.lower() == ".jsonl" else "json"

        self._lock = threading.Lock()

        self._entries: list[dict[str, Any]] = []
        self._exp_latest: dict[int, dict[str, Any]] = {}
        self._entries_by_round: dict[int, list[int]] = {}
        self._round_context: dict[int, dict[str, Any]] = {}
        self._debug_events: list[dict[str, Any]] = []
        self._feedback_by_round: dict[int, dict[str, Any]] = {}
        self._parse_errors = 0
        self._warning = ""
        self._last_mtime_ns = 0
        self._last_size = 0
        self._jsonl_offset = 0
        self._jsonl_remainder = b""

        self.monitor_root = (
            self.run_dir / "monitor" / "tasks" / self.task_id
            if self.run_dir is not None
            else None
        )
        self.latest_path = self.monitor_root / "latest.json" if self.monitor_root else None
        self.rounds_dir = self.monitor_root / "rounds" if self.monitor_root else None
        self.final_path = self.monitor_root / "final.json" if self.monitor_root else None
        self.log_path = self._guess_log_path()
        self.manifests_dir = self._guess_manifests_dir()

        self._monitor_cache_mtime_ns = 0
        self._monitor_latest: dict[str, Any] = {}
        self._monitor_rounds: dict[int, dict[str, Any]] = {}
        self._monitor_final: dict[str, Any] = {}

        self._log_cache_mtime_ns = 0
        self._log_final_status = "unknown"
        self._log_final_rounds: int | None = None
        self._log_round_results: dict[int, dict[str, Any]] = {}

        self._manifest_cache_mtime_ns = 0
        self._manifest_items: list[dict[str, Any]] = []

        self._cluster_cache_lock = threading.Lock()
        self._cluster_poll_interval_sec = 8.0
        self._cluster_log_tail_default = 300
        self._cluster_last_refresh_ts = 0.0
        self._cluster_refresh_inflight = False
        self._cluster_cache_error = ""
        self._cluster_wake_event = threading.Event()
        self._cluster_poller_started = False
        self._cluster_pod_cache: dict[str, dict[str, Any]] = {}
        self._cluster_log_cache: dict[str, dict[str, Any]] = {}

    def _guess_log_path(self) -> Path | None:
        if not self.run_dir:
            return None
        path = self.run_dir / "logs" / f"{self.task_id}.log"
        return path if path.exists() else None

    def _guess_manifests_dir(self) -> Path | None:
        if not self.run_dir:
            return None
        path = self.run_dir / "workspaces" / self.task_id / ".embomaster" / "k8s_manifests"
        return path if path.exists() else None

    def refresh(self) -> None:
        with self._lock:
            self._refresh_trajectory_locked()
            self._refresh_monitor_locked()
            self._refresh_log_cache_locked()
            self._refresh_manifest_cache_locked()

    def _reset_runtime_locked(self) -> None:
        self._entries = []
        self._exp_latest = {}
        self._entries_by_round = {}
        self._round_context = {}
        self._debug_events = []
        self._feedback_by_round = {}

    def _append_entry_locked(self, raw: dict[str, Any]) -> None:
        index = len(self._entries)
        detail, debug_events, feedback_hint, round_context = _build_entry_detail(
            raw=raw,
            index=index,
            preview_chars=self.preview_chars,
        )
        self._entries.append(detail)
        round_index = _as_int(detail.get("exp_index"), -1)
        if round_index is not None and round_index > 0:
            self._entries_by_round.setdefault(round_index, []).append(index)
            previous = self._exp_latest.get(round_index)
            if previous is None or int(detail.get("step", 0) or 0) >= int(previous.get("step", 0) or 0):
                self._exp_latest[round_index] = detail
            if round_context:
                current_context = dict(self._round_context.get(round_index, {}))
                current_context.update(round_context)
                self._round_context[round_index] = current_context
            if feedback_hint:
                self._feedback_by_round[round_index] = feedback_hint
        self._debug_events.extend(debug_events)

    def _refresh_trajectory_locked(self) -> None:
        if not self.path.exists():
            self._warning = f"trajectory not found: {self.path}"
            return
        if self._format == "jsonl":
            self._refresh_jsonl_locked()
        else:
            self._refresh_json_locked()

    def _refresh_jsonl_locked(self) -> None:
        stat = self.path.stat()
        size = stat.st_size
        mtime_ns = int(stat.st_mtime_ns)
        if size < self._jsonl_offset:
            self._reset_runtime_locked()
            self._jsonl_offset = 0
            self._jsonl_remainder = b""
        if size == self._jsonl_offset and mtime_ns == self._last_mtime_ns:
            return

        with self.path.open("rb") as fh:
            fh.seek(self._jsonl_offset)
            chunk = fh.read()
        self._jsonl_offset += len(chunk)

        data = self._jsonl_remainder + chunk
        parts = data.split(b"\n")
        remainder = b""
        if data.endswith(b"\n"):
            lines = parts[:-1] if parts and parts[-1] == b"" else parts
        else:
            lines = parts[:-1]
            remainder = parts[-1] if parts else b""

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                self._parse_errors += 1
                continue
            if isinstance(item, dict):
                self._append_entry_locked(item)
            else:
                self._parse_errors += 1

        self._jsonl_remainder = remainder
        self._last_mtime_ns = mtime_ns
        self._last_size = size
        self._warning = (
            "detected an incomplete jsonl line at file tail (likely still writing)"
            if remainder
            else ""
        )

    def _refresh_json_locked(self) -> None:
        stat = self.path.stat()
        size = stat.st_size
        mtime_ns = int(stat.st_mtime_ns)
        if size == self._last_size and mtime_ns == self._last_mtime_ns:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._warning = "legacy json parse failed; keeping last good snapshot"
            self._parse_errors += 1
            return
        if not isinstance(payload, list):
            self._warning = "legacy json must be a list; keeping last good snapshot"
            self._parse_errors += 1
            return

        self._reset_runtime_locked()
        for item in payload:
            if isinstance(item, dict):
                self._append_entry_locked(item)
            else:
                self._parse_errors += 1
        self._last_mtime_ns = mtime_ns
        self._last_size = size
        self._warning = ""

    def _refresh_monitor_locked(self) -> None:
        if not self.monitor_root or not self.monitor_root.exists():
            self._monitor_latest = {}
            self._monitor_rounds = {}
            self._monitor_final = {}
            self._monitor_cache_mtime_ns = 0
            return
        mtimes: list[int] = []
        if self.latest_path and self.latest_path.exists():
            mtimes.append(int(self.latest_path.stat().st_mtime_ns))
        if self.final_path and self.final_path.exists():
            mtimes.append(int(self.final_path.stat().st_mtime_ns))
        if self.rounds_dir and self.rounds_dir.exists():
            mtimes.extend(int(path.stat().st_mtime_ns) for path in self.rounds_dir.glob("round_*.json"))
        latest_mtime = max(mtimes) if mtimes else 0
        if latest_mtime == self._monitor_cache_mtime_ns:
            return

        latest_payload: dict[str, Any] = {}
        if self.latest_path and self.latest_path.exists():
            try:
                loaded = json.loads(self.latest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    latest_payload = loaded
            except json.JSONDecodeError:
                latest_payload = {}
        final_payload: dict[str, Any] = {}
        if self.final_path and self.final_path.exists():
            try:
                loaded = json.loads(self.final_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    final_payload = loaded
            except json.JSONDecodeError:
                final_payload = {}

        rounds_payload: dict[int, dict[str, Any]] = {}
        if self.rounds_dir and self.rounds_dir.exists():
            for path in sorted(self.rounds_dir.glob("round_*.json")):
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(loaded, dict):
                    continue
                round_index = _as_int(loaded.get("round_index"), -1)
                if round_index is None or round_index <= 0:
                    continue
                rounds_payload[round_index] = loaded

        self._monitor_latest = latest_payload
        self._monitor_final = final_payload
        self._monitor_rounds = rounds_payload
        self._monitor_cache_mtime_ns = latest_mtime

    def _refresh_log_cache_locked(self) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        mtime_ns = int(self.log_path.stat().st_mtime_ns)
        if mtime_ns == self._log_cache_mtime_ns:
            return
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        statuses = re.findall(r"状态:\s*([^\n]+)", text)
        self._log_final_status = statuses[-1].strip() if statuses else "unknown"
        rounds = re.findall(r"Round:\s*(\d+)", text)
        self._log_final_rounds = _as_int(rounds[-1], 0) if rounds else None

        block_results: dict[int, dict[str, Any]] = {}
        pattern = re.compile(
            r"Round:\s*(?P<round>\d+).*?K8S Status:\s*(?P<status>[^\n]+).*?Metric:\s*(?P<metric>[^\n]+).*?K8S Log Tail:\s*(?P<tail>.*?)(?:\n\s*Please provide improvement suggestions|\Z)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            round_index = _as_int(match.group("round"), -1)
            if round_index is None or round_index <= 0:
                continue
            block_results[round_index] = {
                "round_index": round_index,
                "k8s_status": match.group("status").strip(),
                "metric": match.group("metric").strip(),
                "metric_value": _as_float(match.group("metric"), None),
                "k8s_log_tail": _clip_text(match.group("tail").strip(), 6000),
            }
        self._log_round_results = block_results
        self._log_cache_mtime_ns = mtime_ns

    def _refresh_manifest_cache_locked(self) -> None:
        if not self.manifests_dir or not self.manifests_dir.exists():
            self._manifest_items = []
            self._manifest_cache_mtime_ns = 0
            return
        files = sorted(self.manifests_dir.glob("*.yaml"))
        if not files:
            self._manifest_items = []
            self._manifest_cache_mtime_ns = 0
            return
        latest_mtime = max(int(path.stat().st_mtime_ns) for path in files)
        if latest_mtime == self._manifest_cache_mtime_ns:
            return
        items: list[dict[str, Any]] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            meta_match = re.search(r"(?ms)^metadata:\s*\n((?:[ \t]+.*\n)+)", text)
            namespace = "default"
            job_name = path.stem
            if meta_match:
                block = meta_match.group(1)
                name_match = re.search(r"(?m)^[ \t]+name:\s*([^\s#]+)", block)
                namespace_match = re.search(r"(?m)^[ \t]+namespace:\s*([^\s#]+)", block)
                if name_match:
                    job_name = name_match.group(1).strip()
                if namespace_match:
                    namespace = namespace_match.group(1).strip()
            round_match = re.search(r"-r(\d+)-", path.stem)
            round_index = _as_int(round_match.group(1), -1) if round_match else -1
            items.append(
                {
                    "round_index": round_index if round_index and round_index > 0 else None,
                    "job_name": job_name,
                    "namespace": namespace,
                    "manifest_path": str(path),
                }
            )
        self._manifest_items = items
        self._manifest_cache_mtime_ns = latest_mtime

    def _manifest_by_round_locked(self) -> dict[int, dict[str, Any]]:
        mapping: dict[int, dict[str, Any]] = {}
        for item in self._manifest_items:
            round_index = _as_int(item.get("round_index"), -1)
            if round_index is None or round_index <= 0:
                continue
            previous = mapping.get(round_index)
            if previous is None or str(item.get("manifest_path", "")) > str(previous.get("manifest_path", "")):
                mapping[round_index] = item
        return mapping

    def _build_legacy_rounds_locked(self) -> dict[int, dict[str, Any]]:
        rounds: dict[int, dict[str, Any]] = {}
        candidate_rounds = set(self._exp_latest.keys())
        candidate_rounds.update(self._round_context.keys())
        candidate_rounds.update(self._feedback_by_round.keys())
        candidate_rounds.update(self._log_round_results.keys())
        candidate_rounds.update(
            round_index
            for round_index in (
                _as_int(item.get("round_index"), -1) for item in self._manifest_items
            )
            if round_index is not None and round_index > 0
        )

        for round_index in sorted(candidate_rounds):
            latest_entry = self._exp_latest.get(round_index, {})
            context = self._round_context.get(round_index, {})
            feedback = self._feedback_by_round.get(round_index, {})
            log_item = self._log_round_results.get(round_index, {})
            rounds[round_index] = {
                "round_index": round_index,
                "status": str(latest_entry.get("status", "unknown")),
                "steps": int(latest_entry.get("step", 0) or 0),
                "coding_result": str(latest_entry.get("assistant_text", "") or ""),
                "feedback": "",
                "k8s_status": str(
                    log_item.get("k8s_status")
                    or feedback.get("k8s_status")
                    or "unknown"
                ),
                "metric_value": (
                    _as_float(log_item.get("metric_value"), None)
                    if log_item
                    else _as_float(feedback.get("metric_value"), None)
                ),
                "metric_source": "legacy",
                "metric_valid": _as_float(log_item.get("metric_value"), None) is not None,
                "result_valid": True,
                "validation_errors": [],
                "workspace_id": str(context.get("workspace_id", "") or ""),
                "parent_workspace_id": str(context.get("parent_workspace_id", "") or ""),
                "parent_choice_used": "",
                "parent_choice_reason": "",
                "workspace_codebase_path": str(context.get("workspace_codebase_path", "") or ""),
                "workspace_source_type": str(context.get("workspace_source_type", "") or ""),
                "workspace_large_dirs_count": _as_int(context.get("workspace_large_dirs_count"), 0) or 0,
                "workspace_large_dirs": [],
                "submission_dir": "",
                "session_dir": "",
                "artifacts_summary": {},
                "parent_recommendation": None,
                "k8s": None,
                "updated_at": "",
            }
        return rounds

    def _combined_rounds_locked(self) -> list[dict[str, Any]]:
        manifest_by_round = self._manifest_by_round_locked()
        rounds = self._build_legacy_rounds_locked()
        for round_index, monitor_item in self._monitor_rounds.items():
            base = dict(rounds.get(round_index, {}))
            base.update(monitor_item)
            rounds[round_index] = base

        best_metric: float | None = None
        best_round_index: int | None = None
        for round_index, item in rounds.items():
            metric_value = _as_float(item.get("metric_value"), None)
            if metric_value is not None and (best_metric is None or metric_value > best_metric):
                best_metric = metric_value
                best_round_index = round_index

        output: list[dict[str, Any]] = []
        for round_index in sorted(rounds.keys()):
            item = dict(rounds[round_index])
            entry_indexes = self._entries_by_round.get(round_index, [])
            debug_count = sum(1 for event in self._debug_events if _as_int(event.get("exp_index"), -1) == round_index)
            item["entry_count"] = len(entry_indexes)
            item["debug_test_count"] = debug_count
            item["latest_entry_index"] = entry_indexes[-1] if entry_indexes else None
            latest_entry = self._exp_latest.get(round_index)
            if latest_entry:
                item.setdefault("coding_result", latest_entry.get("assistant_text", ""))
                item["latest_assistant"] = latest_entry.get("assistant_preview", "")
                item["agent_name"] = latest_entry.get("agent_name", "")
            else:
                item["latest_assistant"] = ""
                item["agent_name"] = ""
            item["is_best_round"] = round_index == best_round_index

            k8s_obj = item.get("k8s")
            if not isinstance(k8s_obj, dict):
                k8s_obj = {}
                item["k8s"] = k8s_obj
            manifest_item = manifest_by_round.get(round_index)
            if manifest_item:
                if not k8s_obj.get("manifest_path"):
                    k8s_obj["manifest_path"] = manifest_item.get("manifest_path", "")
                if not k8s_obj.get("job_name"):
                    k8s_obj["job_name"] = manifest_item.get("job_name", "")
                if not k8s_obj.get("namespace"):
                    k8s_obj["namespace"] = manifest_item.get("namespace", "default")
            output.append(item)
        return output

    def get_overview(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            rounds = self._combined_rounds_locked()
            structured = bool(self._monitor_rounds)
            best_round = next((item for item in rounds if item.get("is_best_round")), None)
            pod_items = self._pod_items_locked(rounds)
            final_payload = self._monitor_final if self._monitor_final else {}
            return {
                "task_id": self.task_id,
                "label": self.run_label,
                "format": self._format,
                "path": str(self.path),
                "run_dir": str(self.run_dir) if self.run_dir else "",
                "monitor_mode": "structured" if structured else "legacy",
                "total_entries": len(self._entries),
                "total_rounds": len(rounds),
                "total_debug_tests": len(self._debug_events),
                "pod_count": len([item for item in pod_items if item.get("pod_name")]),
                "job_count": len({str(item.get("job_name", "")) for item in pod_items if item.get("job_name")}),
                "best_metric": (
                    final_payload.get("best_metric")
                    if final_payload.get("best_metric") is not None
                    else (best_round.get("metric_value") if isinstance(best_round, dict) else None)
                ),
                "best_round_index": (
                    final_payload.get("best_round_index")
                    if final_payload.get("best_round_index") is not None
                    else (best_round.get("round_index") if isinstance(best_round, dict) else None)
                ),
                "invalid_round_count": (
                    final_payload.get("invalid_round_count")
                    if final_payload.get("invalid_round_count") is not None
                    else sum(1 for item in rounds if not bool(item.get("result_valid", True)))
                ),
                "status": (
                    final_payload.get("status")
                    or self._monitor_latest.get("overview", {}).get("status")
                    or self._log_final_status
                    or (rounds[-1].get("status") if rounds else "unknown")
                ),
                "stopped_reason": (
                    final_payload.get("stopped_reason")
                    or self._monitor_latest.get("overview", {}).get("stopped_reason", "")
                ),
                "updated_at": self._monitor_latest.get("updated_at", _iso_time_from_ns(self._last_mtime_ns)),
                "parse_errors": self._parse_errors,
                "warning": self._warning,
            }

    def get_rounds(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            return self._combined_rounds_locked()

    def get_route(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            rounds = self._combined_rounds_locked()
            workspace_to_round = {
                str(item.get("workspace_id", "")).strip(): int(item.get("round_index", 0) or 0)
                for item in rounds
                if str(item.get("workspace_id", "")).strip()
            }
            nodes = []
            edges = []
            for item in rounds:
                round_index = int(item.get("round_index", 0) or 0)
                parent_workspace_id = str(item.get("parent_workspace_id", "") or "").strip()
                parent_round_index = workspace_to_round.get(parent_workspace_id)
                nodes.append(
                    {
                        "round_index": round_index,
                        "workspace_id": str(item.get("workspace_id", "") or ""),
                        "parent_workspace_id": parent_workspace_id,
                        "parent_round_index": parent_round_index,
                        "parent_choice_used": str(item.get("parent_choice_used", "") or ""),
                        "metric_value": _as_float(item.get("metric_value"), None),
                        "status": str(item.get("status", "unknown")),
                        "k8s_status": str(item.get("k8s_status", "unknown")),
                        "result_valid": bool(item.get("result_valid", True)),
                        "debug_test_count": int(item.get("debug_test_count", 0) or 0),
                        "entry_count": int(item.get("entry_count", 0) or 0),
                        "is_best_round": bool(item.get("is_best_round", False)),
                    }
                )
                if parent_round_index is not None and parent_round_index > 0:
                    edges.append(
                        {
                            "source": parent_round_index,
                            "target": round_index,
                            "type": str(item.get("parent_choice_used", "") or "parent"),
                        }
                    )
            return {"nodes": nodes, "edges": edges}

    def get_stream(self, round_index: int | None, cursor: int, limit: int) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            limit = max(1, min(int(limit), 500))
            if round_index is None:
                entries = self._entries
            else:
                indexes = self._entries_by_round.get(int(round_index), [])
                entries = [self._entries[i] for i in indexes]
            cursor = max(0, int(cursor))
            if cursor > len(entries):
                cursor = 0
            chunk = entries[cursor: cursor + limit]
            return {
                "cursor_next": cursor + len(chunk),
                "total": len(entries),
                "entries": chunk,
            }

    def get_entry(self, index: int) -> dict[str, Any] | None:
        self.refresh()
        with self._lock:
            if index < 0 or index >= len(self._entries):
                return None
            return self._entries[index]

    def get_round_detail(self, round_index: int) -> dict[str, Any] | None:
        self.refresh()
        with self._lock:
            rounds = {int(item.get("round_index", 0) or 0): item for item in self._combined_rounds_locked()}
            item = rounds.get(round_index)
            if item is None:
                return None
            detail = dict(item)
            k8s_obj = detail.get("k8s")
            if isinstance(k8s_obj, dict):
                manifest_path = str(k8s_obj.get("manifest_path", "") or "").strip()
                detail["manifest_text"] = self._read_text_preview(manifest_path, 32000)
                logs_path = str(k8s_obj.get("logs_path", "") or "").strip()
                if logs_path and self.monitor_root is not None:
                    detail["k8s_logs_preview"] = self._read_text_preview(self.monitor_root / logs_path, 32000)
                else:
                    detail["k8s_logs_preview"] = str(k8s_obj.get("logs_preview", "") or "")
            else:
                detail["manifest_text"] = ""
                detail["k8s_logs_preview"] = ""
            detail["debug_events"] = [
                event for event in self._debug_events if _as_int(event.get("exp_index"), -1) == round_index
            ]
            return detail

    def get_pod_items(self) -> list[dict[str, Any]]:
        return self.get_pod_payload()["items"]

    def get_pod_payload(
        self,
        refresh: bool = False,
        tail: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_cluster_poller()
        if refresh or self._cluster_last_refresh_ts <= 0:
            self._refresh_cluster_cache_sync(force=True, tail=tail)
        else:
            self._maybe_trigger_cluster_refresh(tail=tail)
        items = self._build_base_pod_items()
        return self._merge_pod_payload(items)

    def _build_base_pod_items(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            return [dict(item) for item in self._pod_items_locked(self._combined_rounds_locked())]

    def _pod_cache_key(self, item: dict[str, Any]) -> str:
        return "|".join(
            [
                str(item.get("round_index", "") or ""),
                str(item.get("namespace", "default") or "default"),
                str(item.get("pod_name", "") or ""),
                str(item.get("job_name", "") or ""),
            ]
        )

    def _ensure_cluster_poller(self) -> None:
        with self._cluster_cache_lock:
            if self._cluster_poller_started:
                return
            self._cluster_poller_started = True
        worker = threading.Thread(
            target=self._cluster_poller_loop,
            name=f"embomaster-cluster-cache-{self.task_id}",
            daemon=True,
        )
        worker.start()

    def _cluster_poller_loop(self) -> None:
        while True:
            try:
                self._refresh_cluster_cache_sync(force=True, tail=self._cluster_log_tail_default)
            except Exception as exc:
                with self._cluster_cache_lock:
                    self._cluster_cache_error = str(exc)
            if self._cluster_wake_event.wait(self._cluster_poll_interval_sec):
                self._cluster_wake_event.clear()

    def _maybe_trigger_cluster_refresh(self, tail: int | None = None) -> None:
        desired_tail = max(
            20,
            min(
                int(tail or self._cluster_log_tail_default or 300),
                5000,
            ),
        )
        with self._cluster_cache_lock:
            if desired_tail > self._cluster_log_tail_default:
                self._cluster_log_tail_default = desired_tail
            age = time.monotonic() - self._cluster_last_refresh_ts if self._cluster_last_refresh_ts else float("inf")
            should_wake = age >= self._cluster_poll_interval_sec or not self._cluster_pod_cache
        if should_wake:
            self._cluster_wake_event.set()

    def _refresh_cluster_cache_sync(self, force: bool = False, tail: int | None = None) -> None:
        desired_tail = max(
            20,
            min(
                int(tail or self._cluster_log_tail_default or 300),
                5000,
            ),
        )
        with self._cluster_cache_lock:
            if desired_tail > self._cluster_log_tail_default:
                self._cluster_log_tail_default = desired_tail
            if self._cluster_refresh_inflight:
                return
            if (
                not force
                and self._cluster_last_refresh_ts > 0
                and (time.monotonic() - self._cluster_last_refresh_ts) < self._cluster_poll_interval_sec
            ):
                return
            self._cluster_refresh_inflight = True

        try:
            items = self._build_base_pod_items()
            pod_cache: dict[str, dict[str, Any]] = {}
            log_cache: dict[str, dict[str, Any]] = {}
            errors: list[str] = []
            for item in items:
                if not item.get("pod_name") and not item.get("job_name"):
                    continue
                cache_key = self._pod_cache_key(item)
                runtime_item, log_item = self._refresh_one_cluster_item(item, tail=desired_tail)
                pod_cache[cache_key] = runtime_item
                log_cache[cache_key] = log_item
                for error_text in (runtime_item.get("error"), log_item.get("error")):
                    error_text = str(error_text or "").strip()
                    if error_text:
                        errors.append(error_text)
            with self._cluster_cache_lock:
                self._cluster_pod_cache = pod_cache
                self._cluster_log_cache = log_cache
                self._cluster_last_refresh_ts = time.monotonic()
                self._cluster_cache_error = "; ".join(errors[:4])
        finally:
            with self._cluster_cache_lock:
                self._cluster_refresh_inflight = False

    def _refresh_single_pod_cache_sync(self, item: dict[str, Any], tail: int) -> None:
        desired_tail = max(20, min(int(tail), 5000))
        with self._cluster_cache_lock:
            if desired_tail > self._cluster_log_tail_default:
                self._cluster_log_tail_default = desired_tail
        runtime_item, log_item = self._refresh_one_cluster_item(item, tail=desired_tail)
        cache_key = self._pod_cache_key(item)
        with self._cluster_cache_lock:
            self._cluster_pod_cache[cache_key] = runtime_item
            self._cluster_log_cache[cache_key] = log_item
            self._cluster_last_refresh_ts = time.monotonic()
            self._cluster_cache_error = str(runtime_item.get("error") or log_item.get("error") or "")

    def _refresh_one_cluster_item(
        self,
        item: dict[str, Any],
        tail: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        namespace = str(item.get("namespace", "default") or "default").strip() or "default"
        requested_pod_name = str(item.get("pod_name", "") or "").strip()
        job_name = str(item.get("job_name", "") or "").strip()
        resolved_pod_name = requested_pod_name
        resolve_error = ""
        if not resolved_pod_name and job_name:
            resolved_pod_name, resolve_error = self.resolve_job_to_pod(job_name=job_name, namespace=namespace)
            resolved_pod_name = str(resolved_pod_name or "").strip()

        snapshot: dict[str, Any] = {
            "ok": False,
            "status": str(item.get("status", "") or "unknown"),
            "node_name": str(item.get("node_name", "") or ""),
            "start_time": str(item.get("start_time", "") or ""),
            "host_ip": "",
            "pod_ip": "",
            "container_statuses": [],
            "error": resolve_error,
        }
        logs_payload: dict[str, Any] = {
            "ok": False,
            "source": "live_kubectl",
            "pod_name": resolved_pod_name or requested_pod_name,
            "namespace": namespace,
            "logs": "",
            "error": resolve_error,
        }
        if resolved_pod_name:
            snapshot = self._fetch_live_pod_snapshot(namespace=namespace, pod_name=resolved_pod_name)
            logs_payload = self._fetch_live_pod_logs(
                namespace=namespace,
                pod_name=resolved_pod_name,
                tail=tail,
            )
        updated_at_ts = time.time()
        container_statuses = list(snapshot.get("container_statuses", []) or [])
        ready_containers = sum(1 for cs in container_statuses if bool(cs.get("ready", False)))
        container_count = len(container_statuses)
        runtime_item = {
            "resolved_pod_name": resolved_pod_name or requested_pod_name,
            "status": str(snapshot.get("status", "") or item.get("status", "") or "unknown"),
            "node_name": str(snapshot.get("node_name", "") or item.get("node_name", "") or ""),
            "start_time": str(snapshot.get("start_time", "") or item.get("start_time", "") or ""),
            "host_ip": str(snapshot.get("host_ip", "") or ""),
            "pod_ip": str(snapshot.get("pod_ip", "") or ""),
            "container_statuses": container_statuses,
            "ready_containers": ready_containers,
            "container_count": container_count,
            "ready_summary": (
                f"{ready_containers}/{container_count}" if container_count > 0 else "-"
            ),
            "ok": bool(snapshot.get("ok", False)),
            "error": str(snapshot.get("error", "") or resolve_error or ""),
            "updated_at": _iso_time_from_ts(updated_at_ts),
            "updated_at_ts": updated_at_ts,
        }
        logs_payload["tail"] = tail
        logs_payload["resolved_pod_name"] = resolved_pod_name or requested_pod_name
        logs_payload["updated_at"] = _iso_time_from_ts(updated_at_ts)
        logs_payload["updated_at_ts"] = updated_at_ts
        return runtime_item, logs_payload

    def _fetch_live_pod_snapshot(self, namespace: str, pod_name: str) -> dict[str, Any]:
        cmd = ["kubectl", "get", "pod", "-n", namespace, pod_name, "-o", "json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception as exc:
            return {
                "ok": False,
                "status": "unknown",
                "container_statuses": [],
                "error": f"failed to run kubectl get pod: {exc}",
            }
        if proc.returncode != 0:
            return {
                "ok": False,
                "status": "unknown",
                "container_statuses": [],
                "error": proc.stderr.strip() or proc.stdout.strip() or "kubectl get pod failed",
            }
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "status": "unknown",
                "container_statuses": [],
                "error": "invalid kubectl get pod json output",
            }
        status_obj = payload.get("status", {})
        spec = payload.get("spec", {})
        container_statuses_raw = status_obj.get("containerStatuses", [])
        container_statuses: list[dict[str, Any]] = []
        if isinstance(container_statuses_raw, list):
            for cs in container_statuses_raw:
                if not isinstance(cs, dict):
                    continue
                container_statuses.append(
                    {
                        "name": str(cs.get("name", "") or ""),
                        "ready": bool(cs.get("ready", False)),
                        "restart_count": int(cs.get("restartCount", 0) or 0),
                        "state": _summarize_container_state(cs.get("state")),
                        "image": str(cs.get("image", "") or ""),
                    }
                )
        return {
            "ok": True,
            "status": str(status_obj.get("phase", "") or "unknown").lower(),
            "node_name": str(spec.get("nodeName", "") or ""),
            "start_time": str(status_obj.get("startTime", "") or ""),
            "host_ip": str(status_obj.get("hostIP", "") or ""),
            "pod_ip": str(status_obj.get("podIP", "") or ""),
            "container_statuses": container_statuses,
            "error": "",
        }

    def _fetch_live_pod_logs(self, namespace: str, pod_name: str, tail: int) -> dict[str, Any]:
        cmd = ["kubectl", "logs", "-n", namespace, pod_name, f"--tail={int(tail)}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        except Exception as exc:
            return {
                "ok": False,
                "source": "live_kubectl",
                "pod_name": pod_name,
                "namespace": namespace,
                "logs": "",
                "error": f"failed to run kubectl logs: {exc}",
            }
        if proc.returncode != 0:
            return {
                "ok": False,
                "source": "live_kubectl",
                "pod_name": pod_name,
                "namespace": namespace,
                "logs": "",
                "error": proc.stderr.strip() or proc.stdout.strip() or "kubectl logs failed",
            }
        return {
            "ok": True,
            "source": "live_kubectl",
            "pod_name": pod_name,
            "namespace": namespace,
            "logs": proc.stdout,
            "error": "",
        }

    def _merge_pod_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        with self._cluster_cache_lock:
            pod_cache = dict(self._cluster_pod_cache)
            log_cache = dict(self._cluster_log_cache)
            last_refresh_ts = self._cluster_last_refresh_ts
            inflight = self._cluster_refresh_inflight
            cache_error = self._cluster_cache_error
            log_tail = self._cluster_log_tail_default

        merged_items: list[dict[str, Any]] = []
        for item in items:
            merged_item = dict(item)
            cache_key = self._pod_cache_key(item)
            runtime_item = pod_cache.get(cache_key, {})
            log_item = log_cache.get(cache_key, {})
            if runtime_item:
                merged_item["resolved_pod_name"] = str(
                    runtime_item.get("resolved_pod_name", "") or item.get("pod_name", "") or ""
                )
                merged_item["status"] = str(runtime_item.get("status", "") or item.get("status", "") or "")
                merged_item["node_name"] = str(
                    runtime_item.get("node_name", "") or item.get("node_name", "") or ""
                )
                merged_item["start_time"] = str(
                    runtime_item.get("start_time", "") or item.get("start_time", "") or ""
                )
                merged_item["container_statuses"] = list(runtime_item.get("container_statuses", []) or [])
                merged_item["ready_summary"] = str(runtime_item.get("ready_summary", "") or "-")
                merged_item["cache_updated_at"] = str(runtime_item.get("updated_at", "") or "")
            else:
                merged_item["resolved_pod_name"] = str(item.get("pod_name", "") or "")
                merged_item["container_statuses"] = []
                merged_item["ready_summary"] = "-"
                merged_item["cache_updated_at"] = ""
            merged_item["has_live_logs"] = bool(log_item.get("ok", False))
            merged_item["log_updated_at"] = str(log_item.get("updated_at", "") or "")
            merged_item["cache_error"] = str(
                runtime_item.get("error", "") or log_item.get("error", "") or ""
            )
            merged_items.append(merged_item)

        age_sec: float | None = None
        if last_refresh_ts > 0:
            age_sec = round(max(0.0, time.monotonic() - last_refresh_ts), 1)
        return {
            "items": merged_items,
            "cache": {
                "enabled": True,
                "poll_interval_sec": self._cluster_poll_interval_sec,
                "last_refresh": _iso_time_from_ts(last_refresh_ts),
                "age_sec": age_sec,
                "inflight": inflight,
                "error": cache_error,
                "log_tail": log_tail,
            },
        }

    def _pod_items_locked(self, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, int]] = set()
        for round_item in rounds:
            round_index = int(round_item.get("round_index", 0) or 0)
            for key in (
                "debug_pod_prepare",
                "debug_pod_cleanup_before_submit",
                "debug_pod_cleanup_final",
            ):
                payload = round_item.get(key)
                if not isinstance(payload, dict):
                    continue
                pod_name = str(payload.get("pod_name", "") or "").strip()
                namespace = str(payload.get("namespace", "default") or "default").strip()
                if not pod_name:
                    continue
                item_key = (pod_name, namespace, key, round_index)
                if item_key in seen:
                    continue
                seen.add(item_key)
                items.append(
                    {
                        "source": key,
                        "round_index": round_index,
                        "pod_name": pod_name,
                        "job_name": "",
                        "namespace": namespace,
                        "status": str(payload.get("status", "") or ""),
                        "node_name": "",
                    }
                )
            k8s_obj = round_item.get("k8s")
            if not isinstance(k8s_obj, dict):
                continue
            job_name = str(k8s_obj.get("job_name", "") or "").strip()
            namespace = str(k8s_obj.get("namespace", "default") or "default").strip()
            pods = k8s_obj.get("pods", [])
            if isinstance(pods, list) and pods:
                for pod in pods:
                    if not isinstance(pod, dict):
                        continue
                    pod_name = str(pod.get("pod_name", "") or "").strip()
                    item_key = (pod_name or job_name, namespace, "job_pod", round_index)
                    if item_key in seen:
                        continue
                    seen.add(item_key)
                    items.append(
                        {
                            "source": "job_pod",
                            "round_index": round_index,
                            "pod_name": pod_name,
                            "job_name": job_name,
                            "namespace": namespace,
                            "status": str(pod.get("status", "") or ""),
                            "node_name": str(pod.get("node_name", "") or ""),
                            "start_time": str(pod.get("start_time", "") or ""),
                        }
                    )
            elif job_name:
                item_key = (job_name, namespace, "job", round_index)
                if item_key not in seen:
                    seen.add(item_key)
                    items.append(
                        {
                            "source": "job",
                            "round_index": round_index,
                            "pod_name": "",
                            "job_name": job_name,
                            "namespace": namespace,
                            "status": str(round_item.get("k8s_status", "") or ""),
                            "node_name": "",
                        }
                    )
        for event in reversed(self._debug_events):
            pod_name = str(event.get("pod_name", "")).strip()
            namespace = str(event.get("namespace", "default")).strip() or "default"
            round_index = _as_int(event.get("exp_index"), -1) or -1
            if not pod_name:
                continue
            item_key = (pod_name, namespace, "debug_event", round_index)
            if item_key in seen:
                continue
            seen.add(item_key)
            items.append(
                {
                    "source": "debug_test",
                    "round_index": round_index if round_index > 0 else None,
                    "pod_name": pod_name,
                    "job_name": "",
                    "namespace": namespace,
                    "status": str(event.get("status", "") or ""),
                    "node_name": "",
                }
            )
        return items

    def resolve_job_to_pod(self, job_name: str, namespace: str) -> tuple[str | None, str]:
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception as exc:
            return None, f"failed to run kubectl get pods: {exc}"
        if proc.returncode != 0:
            return None, proc.stderr.strip() or proc.stdout.strip() or "kubectl get pods failed"
        names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not names:
            return None, "no pod found for this job"
        return names[-1], "ok"

    def get_pod_logs(
        self,
        pod_name: str,
        namespace: str,
        tail: int = 200,
        job_name: str = "",
        round_index: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        namespace = str(namespace).strip() or "default"
        tail = max(20, min(int(tail), 5000))
        pod_name = str(pod_name).strip()
        job_name = str(job_name).strip()
        self._ensure_cluster_poller()
        item = {
            "source": "query",
            "round_index": round_index,
            "pod_name": pod_name,
            "job_name": job_name,
            "namespace": namespace,
            "status": "",
            "node_name": "",
        }
        cache_key = self._pod_cache_key(item)
        with self._cluster_cache_lock:
            cached_log = dict(self._cluster_log_cache.get(cache_key, {}))
            cached_runtime = dict(self._cluster_pod_cache.get(cache_key, {}))
        cached_tail = int(cached_log.get("tail", 0) or 0)
        if refresh or not cached_log or cached_tail < tail:
            self._refresh_single_pod_cache_sync(item, tail=tail)
            with self._cluster_cache_lock:
                cached_log = dict(self._cluster_log_cache.get(cache_key, {}))
                cached_runtime = dict(self._cluster_pod_cache.get(cache_key, {}))
        else:
            self._maybe_trigger_cluster_refresh(tail=tail)

        if cached_log.get("ok"):
            updated_at_ts = float(cached_log.get("updated_at_ts", 0) or 0)
            age_seconds = round(max(0.0, time.time() - updated_at_ts), 1) if updated_at_ts > 0 else None
            return {
                "ok": True,
                "source": "live_cache",
                "cached": True,
                "pod_name": str(
                    cached_runtime.get("resolved_pod_name", "") or cached_log.get("resolved_pod_name", "") or pod_name
                ),
                "namespace": namespace,
                "logs": str(cached_log.get("logs", "") or ""),
                "error": "",
                "updated_at": str(cached_log.get("updated_at", "") or ""),
                "age_seconds": age_seconds,
            }

        if round_index is not None:
            detail = self.get_round_detail(round_index)
            if isinstance(detail, dict):
                fallback_logs = str(detail.get("k8s_logs_preview", "") or "")
                if fallback_logs:
                    return {
                        "ok": True,
                        "source": "stored_round_logs",
                        "cached": False,
                        "pod_name": pod_name or "",
                        "namespace": namespace,
                        "logs": fallback_logs,
                        "error": "",
                        "updated_at": "",
                        "age_seconds": None,
                    }

        return {
            "ok": False,
            "source": "live_cache",
            "cached": True,
            "pod_name": str(cached_runtime.get("resolved_pod_name", "") or pod_name or ""),
            "namespace": namespace,
            "logs": "",
            "error": str(
                cached_log.get("error", "")
                or cached_runtime.get("error", "")
                or "no logs available"
            ),
            "updated_at": str(cached_log.get("updated_at", "") or ""),
            "age_seconds": None,
        }

    def get_artifacts(self, round_index: int) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            rounds = {int(item.get("round_index", 0) or 0): item for item in self._combined_rounds_locked()}
            round_item = rounds.get(round_index)
            if round_item is None:
                return {"round_index": round_index, "root": "", "categories": []}
            root_path_raw = str(round_item.get("workspace_codebase_path", "") or "").strip()
            root_path = Path(root_path_raw).expanduser().resolve() if root_path_raw else None
            if not root_path or not root_path.exists():
                return {"round_index": round_index, "root": root_path_raw, "categories": []}

            categories = []
            for label, category_path in self._discover_output_dirs(root_path):
                files = self._collect_files(category_path, limit=220)
                categories.append(
                    {
                        "label": label,
                        "path": str(category_path),
                        "relative_root": str(category_path.relative_to(root_path)),
                        "file_count": len(files),
                        "files": files,
                    }
                )
            return {
                "round_index": round_index,
                "root": str(root_path),
                "categories": categories,
            }

    def preview_artifact(self, round_index: int, absolute_path: str, max_bytes: int = 32768) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            rounds = {int(item.get("round_index", 0) or 0): item for item in self._combined_rounds_locked()}
            round_item = rounds.get(round_index)
            if round_item is None:
                return {"ok": False, "error": "round not found", "content": ""}
            root_path_raw = str(round_item.get("workspace_codebase_path", "") or "").strip()
            if not root_path_raw:
                return {"ok": False, "error": "workspace path missing", "content": ""}
            root_path = Path(root_path_raw).expanduser().resolve()
            path = Path(absolute_path).expanduser().resolve()
            try:
                path.relative_to(root_path)
            except ValueError:
                return {"ok": False, "error": "path outside workspace", "content": ""}
            if not path.exists() or not path.is_file():
                return {"ok": False, "error": "file not found", "content": ""}
            if not self._is_text_file(path):
                return {
                    "ok": False,
                    "error": "binary file preview is not supported",
                    "content": "",
                    "path": str(path),
                }
            data = path.read_text(encoding="utf-8", errors="replace")
            return {
                "ok": True,
                "error": "",
                "path": str(path),
                "content": _clip_text(data, max_bytes),
            }

    def _discover_output_dirs(self, root_path: Path) -> list[tuple[str, Path]]:
        seen: set[Path] = set()
        items: list[tuple[str, Path]] = []
        for name in ("eval_result", "submission"):
            candidate = root_path / name
            if candidate.exists() and candidate.is_dir():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    items.append((name, resolved))
        policy_root = root_path / "policy"
        if policy_root.exists():
            for path in policy_root.rglob("*"):
                if not path.is_dir():
                    continue
                if path.name not in OUTPUT_DIR_NAMES:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                items.append((str(resolved.relative_to(root_path)), resolved))
                if len(items) >= 10:
                    break
        return items

    def _collect_files(self, root_path: Path, limit: int) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        count = 0
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "relative_path": str(path.relative_to(root_path)),
                    "absolute_path": str(path),
                    "size_bytes": int(stat.st_size),
                    "mtime": _iso_time_from_ns(int(stat.st_mtime_ns)),
                    "is_text": self._is_text_file(path),
                }
            )
            count += 1
            if count >= limit:
                break
        return files

    def _is_text_file(self, path: Path) -> bool:
        if path.suffix.lower() in TEXT_SUFFIXES:
            return True
        try:
            sample = path.read_bytes()[:2048]
        except OSError:
            return False
        return b"\x00" not in sample

    def _read_text_preview(self, value: str | Path, max_chars: int) -> str:
        path = Path(value).expanduser().resolve() if not isinstance(value, Path) else value.expanduser().resolve()
        if not path.exists() or not path.is_file():
            return ""
        try:
            return _clip_text(path.read_text(encoding="utf-8", errors="replace"), max_chars)
        except OSError:
            return ""


class DashboardManager:
    def __init__(self, source_path: str | Path, preview_chars: int = 220) -> None:
        self.source_path = Path(source_path)
        self.preview_chars = preview_chars
        self._lock = threading.Lock()
        self._runs: list[dict[str, Any]] = []
        self._stores: dict[str, TaskRunStore] = {}
        self._last_scan_ts = 0.0
        self._scan_interval_sec = 5.0
        self.refresh(force=True)

    def _discover_run_candidates(self) -> list[tuple[Path, Path | None, str | None]]:
        src = self.source_path.expanduser().resolve()
        candidates: list[tuple[Path, Path | None, str | None]] = []
        seen: set[Path] = set()

        def add_candidate(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            task_id: str | None = None
            run_dir: Path | None = None
            if len(resolved.parents) >= 3 and resolved.parents[1].name == "trajectories":
                task_id = resolved.parents[0].name
                run_dir = resolved.parents[2]
            candidates.append((resolved, run_dir, task_id))

        def collect_from_run_dir(run_dir: Path) -> None:
            trajectories_dir = run_dir / "trajectories"
            if not trajectories_dir.exists() or not trajectories_dir.is_dir():
                return
            for task_dir in sorted(trajectories_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                jsonl_file = task_dir / "trajectory.jsonl"
                json_file = task_dir / "trajectory.json"
                if jsonl_file.exists():
                    add_candidate(jsonl_file)
                elif json_file.exists():
                    add_candidate(json_file)

        if src.is_file():
            add_candidate(src)
            return candidates

        if src.exists() and src.is_dir():
            stack: list[Path] = [src]
            visited_dirs: set[Path] = set()
            while stack:
                current = stack.pop()
                try:
                    resolved_current = current.resolve()
                except OSError:
                    continue
                if resolved_current in visited_dirs:
                    continue
                visited_dirs.add(resolved_current)

                trajectories_dir = resolved_current / "trajectories"
                if trajectories_dir.exists() and trajectories_dir.is_dir():
                    collect_from_run_dir(resolved_current)
                    continue

                child_dirs: list[Path] = []
                try:
                    for entry in os.scandir(resolved_current):
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        child_dirs.append(Path(entry.path))
                except OSError:
                    continue

                for child in sorted(child_dirs, reverse=True):
                    stack.append(child)
        return candidates

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_scan_ts) < self._scan_interval_sec:
            return

        with self._lock:
            now = time.monotonic()
            if not force and (now - self._last_scan_ts) < self._scan_interval_sec:
                return
            candidates = self._discover_run_candidates()
            new_runs: list[dict[str, Any]] = []
            new_stores: dict[str, TaskRunStore] = {}
            seen_run_ids: set[str] = set()
            for path, run_dir, task_id in candidates:
                stat = path.stat()
                fmt = "jsonl" if path.suffix.lower() == ".jsonl" else "json"
                if run_dir and task_id:
                    run_key = run_dir.name
                    try:
                        rel_run_dir = run_dir.resolve().relative_to(self.source_path.expanduser().resolve())
                        rel_text = str(rel_run_dir).strip()
                        if rel_text and rel_text != ".":
                            run_key = rel_text
                    except Exception:
                        pass
                    run_id_base = f"{run_key}:{task_id}"
                    label = f"{run_key} / {task_id}"
                else:
                    run_id_base = path.stem
                    label = path.name
                run_id = run_id_base
                suffix = 2
                while run_id in seen_run_ids:
                    run_id = f"{run_id_base}#{suffix}"
                    suffix += 1
                seen_run_ids.add(run_id)

                store = self._stores.get(run_id)
                if store is None or store.path != path:
                    store = TaskRunStore(
                        trajectory_path=path,
                        run_label=label,
                        run_dir=run_dir,
                        task_id=task_id,
                        preview_chars=self.preview_chars,
                    )
                new_stores[run_id] = store
                new_runs.append(
                    {
                        "run_id": run_id,
                        "label": label,
                        "format": fmt,
                        "path": str(path),
                        "run_dir": str(run_dir) if run_dir else "",
                        "task_id": task_id or "",
                        "mtime_ns": int(stat.st_mtime_ns),
                        "size_bytes": int(stat.st_size),
                    }
                )

            new_runs.sort(key=lambda item: item.get("mtime_ns", 0), reverse=True)
            self._runs = new_runs
            self._stores = new_stores
            self._last_scan_ts = time.monotonic()

    def list_runs(self, force: bool = False) -> dict[str, Any]:
        self.refresh(force=force)
        with self._lock:
            runs = []
            for item in self._runs:
                runs.append(
                    {
                        "run_id": item["run_id"],
                        "label": item["label"],
                        "format": item["format"],
                        "path": item["path"],
                        "run_dir": item["run_dir"],
                        "task_id": item["task_id"],
                        "mtime": _iso_time_from_ns(item["mtime_ns"]),
                        "size_bytes": item["size_bytes"],
                    }
                )
            return {
                "runs": runs,
                "default_run_id": runs[0]["run_id"] if runs else "",
            }

    def get_store(self, run_id: str | None) -> tuple[str | None, TaskRunStore | None]:
        self.refresh(force=False)
        with self._lock:
            if not self._runs:
                return None, None
            selected = run_id or self._runs[0]["run_id"]
            if selected not in self._stores:
                selected = self._runs[0]["run_id"]
            return selected, self._stores.get(selected)

    def get_run_meta(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            for item in self._runs:
                if item["run_id"] == run_id:
                    return {
                        "run_id": item["run_id"],
                        "label": item["label"],
                        "format": item["format"],
                        "path": item["path"],
                        "run_dir": item["run_dir"],
                        "task_id": item["task_id"],
                        "mtime": _iso_time_from_ns(item["mtime_ns"]),
                        "size_bytes": item["size_bytes"],
                    }
            return None
