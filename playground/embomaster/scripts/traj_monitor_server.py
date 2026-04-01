#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


APP_JS_PATH = Path(__file__).with_name("traj_monitor_app.js")


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EmboMaster Traj Multi Monitor</title>
  <style>
    :root {
      --bg-a: #eef7f3;
      --bg-b: #dfeee8;
      --ink: #13261f;
      --muted: #496358;
      --card: rgba(255, 255, 255, 0.86);
      --line: rgba(19, 38, 31, 0.14);
      --accent: #00795f;
      --warn: #a35a06;
      --danger: #a12424;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(900px 420px at 8% -10%, #d0ebdf 0%, transparent 70%),
        radial-gradient(1000px 520px at 100% -4%, #ffe9d2 0%, transparent 70%),
        linear-gradient(165deg, var(--bg-a), var(--bg-b));
      min-height: 100vh;
    }
    .wrap { max-width: 1500px; margin: 0 auto; padding: 16px; }
    .panel {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      backdrop-filter: blur(4px);
      padding: 12px;
      animation: rise .35s ease-out;
    }
    .top {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }
    .title { margin: 0; font-size: 24px; font-weight: 700; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 4px; word-break: break-all; }
    .warn { color: var(--warn); font-size: 12px; margin-top: 4px; white-space: pre-wrap; }
    .ctrls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select, input {
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 10px;
      font-weight: 600;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    .label { font-size: 11px; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }
    .cards { display: grid; grid-template-columns: repeat(5, minmax(120px,1fr)); gap: 8px; margin-top: 10px; }
    .card { border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: rgba(255,255,255,.72); }
    .val { font-size: 22px; font-weight: 800; margin-top: 2px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    .section { margin-top: 12px; }
    .list { display: grid; gap: 8px; max-height: 260px; overflow: auto; }
    .exp-item { border: 1px solid var(--line); border-radius: 10px; padding: 8px; background: rgba(255,255,255,.72); cursor: pointer; }
    .exp-item.pick { outline: 2px solid rgba(0, 121, 95, .35); }
    .row { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }
    .bar { margin-top: 6px; height: 7px; border-radius: 99px; overflow: hidden; background: rgba(19,38,31,.08); }
    .bar > span { display: block; height: 100%; background: linear-gradient(90deg, #00a481, #00c19a); }
    .table-wrap { max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid rgba(19,38,31,.08); padding: 7px 6px; text-align: left; font-size: 12px; vertical-align: top; }
    th { position: sticky; top: 0; background: rgba(255,255,255,.9); color: var(--muted); }
    tr.pick { background: rgba(0,121,95,.08); }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .ok { color: #0d6a56; background: #dcf8ef; border-color: #8fdcc3; }
    .run { color: #8a4a00; background: #fff1df; border-color: #f3c58f; }
    .bad { color: #8a1d1d; background: #ffe4e4; border-color: #f2b2b2; }
    .mono { font-family: "IBM Plex Mono", "Menlo", "Consolas", monospace; }
    .box { border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.72); padding: 8px; }
    .scroll { max-height: 320px; overflow: auto; }
    .log-view { white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.35; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .round-item { border-top: 1px dashed rgba(19,38,31,.14); padding-top: 6px; margin-top: 6px; }
    @media (max-width: 1080px) {
      .grid, .split { grid-template-columns: 1fr; }
      .cards { grid-template-columns: repeat(2, minmax(120px,1fr)); }
      .top { grid-template-columns: 1fr; }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(5px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel top">
      <div>
        <h1 class="title">EmboMaster Traj Multi Monitor</h1>
        <div id="meta" class="meta">loading...</div>
        <div id="warn" class="warn"></div>
      </div>
      <div class="ctrls">
        <label class="label" for="runSel">Run</label>
        <select id="runSel"></select>
        <button id="runRefresh">Reload Runs</button>
        <button id="pullBtn">Refresh</button>
        <button id="autoBtn" class="primary">Auto ON</button>
      </div>
    </section>

    <section class="panel section">
      <div class="label">Summary</div>
      <div class="cards">
        <div class="card"><div class="label">Entries</div><div id="totalEntries" class="val">0</div></div>
        <div class="card"><div class="label">Exps</div><div id="totalExps" class="val">0</div></div>
        <div class="card"><div class="label">Max Step</div><div id="maxStep" class="val">0</div></div>
        <div class="card"><div class="label">Debug Tests</div><div id="debugCount" class="val">0</div></div>
        <div class="card"><div class="label">Parse Errors</div><div id="parseErrors" class="val">0</div></div>
      </div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="label">Experiments (click to expand)</div>
        <div id="expList" class="list"></div>
        <div class="label section">Expanded Experiment Content</div>
        <div id="expContent" class="box scroll mono">Select an exp card above.</div>
      </section>

      <section class="panel">
        <div class="label">Incremental Stream</div>
        <div class="table-wrap" style="margin-top:8px;">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>exp</th>
                <th>step</th>
                <th>status</th>
                <th>messages</th>
                <th>debug</th>
                <th>assistant preview</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
        <div class="label section">Selected Entry</div>
        <div id="entryDetail" class="box scroll mono">Select a stream row.</div>
      </section>
    </div>

    <section class="panel section">
      <div class="split">
        <div>
          <div class="label">Debug Test Logs</div>
          <div class="ctrls" style="margin-top:6px;">
            <label class="label" for="debugExpSel">Exp</label>
            <select id="debugExpSel"><option value="all">all</option></select>
            <button id="debugRefresh">Load Debug Logs</button>
          </div>
          <div id="debugBox" class="box scroll mono" style="margin-top:8px;"></div>
        </div>
        <div>
          <div class="label">Pod Logs</div>
          <div class="ctrls" style="margin-top:6px;">
            <label class="label" for="podSel">Pod/Job</label>
            <select id="podSel"></select>
            <label class="label" for="podTail">Tail</label>
            <input id="podTail" type="number" value="200" min="20" max="5000" style="width:92px;" />
            <button id="podFetch">Fetch Logs</button>
          </div>
          <div id="podBox" class="box scroll log-view mono" style="margin-top:8px;">Select pod/job then fetch logs.</div>
        </div>
      </div>
    </section>

    <section class="panel section">
      <div class="label">Run Results</div>
      <div id="runResult" class="box scroll mono" style="margin-top:8px;"></div>
    </section>
  </div>

  <script src="/app.js"></script>
</body>
</html>
"""


def _iso_time_from_ns(mtime_ns: int) -> str:
    if mtime_ns <= 0:
        return "-"
    ts = mtime_ns / 1_000_000_000
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def _as_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
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


def _find_latest_message(messages: list[dict[str, Any]], role: str) -> str:
    role_l = role.lower()
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() != role_l:
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)
    return ""


def _parse_debug_status(content: str) -> tuple[str, int | None]:
    text = str(content)
    m = re.search(r"\[debug_test\]\s+(\w+).*?exit_code=(-?\d+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).lower(), _as_int(m.group(2), None)  # type: ignore[arg-type]
    if "success" in text.lower():
        return "success", None
    if "failed" in text.lower() or "error" in text.lower():
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
        m = re.match(pattern, body)
        if m:
            return m.group(1).strip(), m.group(2).strip()

    # Fallback: no explicit stderr marker, treat all as stdout-equivalent output.
    return body, ""


def _parse_feedback_block(text: str) -> dict[str, Any] | None:
    content = str(text)
    if "K8S Status" not in content or "Round:" not in content:
        return None

    m_round = re.search(r"Round:\s*(\d+)", content, flags=re.IGNORECASE)
    if not m_round:
        return None
    round_index = _as_int(m_round.group(1), -1)

    m_status = re.search(r"K8S Status:\s*\n?\s*([^\n]+)", content, flags=re.IGNORECASE)
    m_metric = re.search(r"Metric:\s*\n?\s*([^\n]+)", content, flags=re.IGNORECASE)
    m_tail = re.search(
        r"K8S Log Tail:\s*\n?(.*?)(?:\n\s*Please provide improvement suggestions|\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return {
        "round_index": round_index,
        "k8s_status": (m_status.group(1).strip() if m_status else "unknown"),
        "metric": (m_metric.group(1).strip() if m_metric else "None"),
        "k8s_log_tail": _clip_text(m_tail.group(1).strip(), 4000) if m_tail else "",
    }


def _build_entry_summary(raw: dict[str, Any], index: int, preview_chars: int = 220) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    traj = raw.get("trajectory")
    if not isinstance(traj, dict):
        traj = {}

    dialogs = traj.get("dialogs", [])
    if not isinstance(dialogs, list):
        dialogs = []
    last_dialog = dialogs[-1] if dialogs and isinstance(dialogs[-1], dict) else {}
    messages = _safe_messages(last_dialog)

    role_counts = Counter(str(m.get("role", "unknown")).lower() for m in messages)
    assistant_preview = _clip_text(_find_latest_message(messages, "assistant"), preview_chars)

    step_list = _safe_steps(traj)
    tool_response_count = 0
    debug_events: list[dict[str, Any]] = []

    exp_index = _as_int(raw.get("exp_index"), -1)
    step_value = _as_int(raw.get("steps", traj.get("step", 0)), 0)

    for step_obj in step_list:
        tool_responses = step_obj.get("tool_responses", [])
        if not isinstance(tool_responses, list):
            continue
        tool_response_count += len(tool_responses)

        for tr in tool_responses:
            if not isinstance(tr, dict):
                continue
            if str(tr.get("name", "")) != "debug_test":
                continue
            meta = tr.get("meta")
            info = {}
            if isinstance(meta, dict):
                raw_info = meta.get("info")
                if isinstance(raw_info, dict):
                    info = raw_info
            content = str(tr.get("content", ""))
            status, exit_code_guess = _parse_debug_status(content)
            stdout_text, stderr_text = _split_debug_streams(content)
            event = {
                "entry_index": index,
                "exp_index": exp_index,
                "step": step_value,
                "agent_name": str(traj.get("agent_name", "unknown")),
                "status": status,
                "exit_code": info.get("exit_code", exit_code_guess),
                "mode": info.get("mode", ""),
                "command": str(info.get("command", "")),
                "full_command": str(info.get("full_command", "")),
                "pod_name": str(info.get("pod_name", "")),
                "namespace": str(info.get("namespace", "default")),
                "working_dir": str(info.get("working_dir", "")),
                "stdout": _clip_text(stdout_text, 12000),
                "stderr": _clip_text(stderr_text, 6000),
                "output": _clip_text(content, 12000),
            }
            debug_events.append(event)

    recent_messages = []
    feedback_hint: dict[str, Any] | None = None
    for msg in messages[-8:]:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        role = str(msg.get("role", "unknown")).lower()
        recent_messages.append({"role": role, "content": _clip_text(content, preview_chars)})

        if role == "user" and feedback_hint is None:
            parsed = _parse_feedback_block(content)
            if parsed:
                feedback_hint = parsed

    summary = {
        "index": index,
        "task_id": raw.get("task_id"),
        "exp_name": raw.get("exp_name"),
        "exp_index": exp_index,
        "status": str(raw.get("status", "running")),
        "step": step_value,
        "trajectory_step": _as_int(traj.get("step"), 0),
        "agent_name": str(traj.get("agent_name", "unknown")),
        "message_count": len(messages),
        "tool_response_count": tool_response_count,
        "debug_test_count": len(debug_events),
        "role_counts": dict(role_counts),
        "last_assistant": assistant_preview,
        "recent_messages": recent_messages,
    }
    return summary, debug_events, feedback_hint


class TrajectoryStore:
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
        self.task_id = task_id
        self.preview_chars = preview_chars

        self._lock = threading.Lock()

        self._entries: list[dict[str, Any]] = []
        self._exp_latest: dict[int, dict[str, Any]] = {}
        self._debug_events: list[dict[str, Any]] = []
        self._round_results: dict[int, dict[str, Any]] = {}
        self._pod_from_debug: set[tuple[str, str]] = set()

        self._format = "jsonl" if trajectory_path.suffix.lower() == ".jsonl" else "json"
        self._warning = ""
        self._parse_errors = 0

        self._last_mtime_ns = 0
        self._last_size = 0

        self._jsonl_offset = 0
        self._jsonl_remainder = b""

        self.log_path = self._guess_log_path()
        self.manifests_dir = self._guess_manifests_dir()

        self._log_cache_mtime_ns = 0
        self._log_final_status = "unknown"
        self._log_final_rounds: int | None = None
        self._log_round_results: dict[int, dict[str, Any]] = {}

        self._manifest_cache_mtime_ns = 0
        self._manifest_items: list[dict[str, Any]] = []

    def _guess_log_path(self) -> Path | None:
        if not self.run_dir or not self.task_id:
            return None
        p = self.run_dir / "logs" / f"{self.task_id}.log"
        return p if p.exists() else None

    def _guess_manifests_dir(self) -> Path | None:
        if not self.run_dir or not self.task_id:
            return None
        p = self.run_dir / "workspaces" / self.task_id / ".embomaster" / "k8s_manifests"
        return p if p.exists() else None

    def _reset_runtime_locked(self) -> None:
        self._entries = []
        self._exp_latest = {}
        self._debug_events = []
        self._round_results = {}
        self._pod_from_debug = set()

    def _append_entry_locked(self, raw: dict[str, Any]) -> None:
        index = len(self._entries)
        summary, debug_events, feedback_hint = _build_entry_summary(raw, index=index, preview_chars=self.preview_chars)
        self._entries.append(summary)

        exp_index = summary.get("exp_index", -1)
        if isinstance(exp_index, int) and exp_index >= 0:
            prev = self._exp_latest.get(exp_index)
            if prev is None or summary.get("step", 0) >= prev.get("step", 0):
                self._exp_latest[exp_index] = summary

        for event in debug_events:
            self._debug_events.append(event)
            pod_name = str(event.get("pod_name", "")).strip()
            namespace = str(event.get("namespace", "default")).strip() or "default"
            if pod_name:
                self._pod_from_debug.add((pod_name, namespace))

        if feedback_hint and isinstance(feedback_hint.get("round_index"), int):
            round_index = int(feedback_hint["round_index"])
            if round_index > 0:
                merged = dict(feedback_hint)
                merged["entry_index"] = index
                merged["exp_index"] = summary.get("exp_index")
                self._round_results[round_index] = merged

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

        with open(self.path, "rb") as f:
            f.seek(self._jsonl_offset)
            chunk = f.read()
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
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            self._warning = "legacy json parse failed; keeping last good snapshot"
            self._parse_errors += 1
            return

        if not isinstance(payload, list):
            self._warning = "legacy json must be a list; keeping last good snapshot"
            self._parse_errors += 1
            return

        self._reset_runtime_locked()
        for raw in payload:
            if isinstance(raw, dict):
                self._append_entry_locked(raw)
            else:
                self._parse_errors += 1

        self._last_mtime_ns = mtime_ns
        self._last_size = size
        self._warning = ""

    def _refresh_log_cache_locked(self) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        stat = self.log_path.stat()
        mtime_ns = int(stat.st_mtime_ns)
        if mtime_ns == self._log_cache_mtime_ns:
            return

        text = self.log_path.read_text(encoding="utf-8", errors="replace")

        statuses = re.findall(r"状态:\s*([^\n]+)", text)
        self._log_final_status = statuses[-1].strip() if statuses else "unknown"

        rounds = re.findall(r"Round:\s*(\d+)", text)
        self._log_final_rounds = _as_int(rounds[-1], 0) if rounds else None

        block_results: dict[int, dict[str, Any]] = {}
        block_pattern = re.compile(
            r"Round:\s*(?P<round>\d+).*?K8S Status:\s*(?P<status>[^\n]+).*?Metric:\s*(?P<metric>[^\n]+).*?K8S Log Tail:\s*(?P<tail>.*?)(?:\n\s*Please provide improvement suggestions|\Z)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for m in block_pattern.finditer(text):
            round_index = _as_int(m.group("round"), -1)
            if round_index <= 0:
                continue
            block_results[round_index] = {
                "round_index": round_index,
                "k8s_status": m.group("status").strip(),
                "metric": m.group("metric").strip(),
                "k8s_log_tail": _clip_text(m.group("tail").strip(), 4000),
                "entry_index": None,
                "exp_index": round_index,
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

        latest_mtime = max(int(f.stat().st_mtime_ns) for f in files)
        if latest_mtime == self._manifest_cache_mtime_ns:
            return

        items: list[dict[str, Any]] = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            meta_match = re.search(r"(?ms)^metadata:\s*\n((?:[ \t]+.*\n)+)", text)
            namespace = "default"
            job_name = f.stem
            if meta_match:
                block = meta_match.group(1)
                name_match = re.search(r"(?m)^[ \t]+name:\s*([^\s#]+)", block)
                ns_match = re.search(r"(?m)^[ \t]+namespace:\s*([^\s#]+)", block)
                if name_match:
                    job_name = name_match.group(1).strip()
                if ns_match:
                    namespace = ns_match.group(1).strip()

            round_match = re.search(r"-r(\d+)-", f.stem)
            exp_index = _as_int(round_match.group(1), -1) if round_match else -1
            items.append(
                {
                    "source": "manifest",
                    "exp_index": exp_index if exp_index > 0 else None,
                    "job_name": job_name,
                    "pod_name": "",
                    "namespace": namespace,
                    "manifest_path": str(f),
                }
            )

        self._manifest_items = items
        self._manifest_cache_mtime_ns = latest_mtime

    def refresh(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._warning = f"file not found: {self.path}"
                return

            if self._format == "jsonl":
                self._refresh_jsonl_locked()
            else:
                self._refresh_json_locked()

            self._refresh_log_cache_locked()
            self._refresh_manifest_cache_locked()

    def _summary_locked(self) -> dict[str, Any]:
        max_step = 0
        for entry in self._entries:
            max_step = max(max_step, _as_int(entry.get("step"), 0))
        latest_values = list(self._exp_latest.values())
        status_counter = Counter(str(x.get("status", "unknown")) for x in latest_values)

        return {
            "total_entries": len(self._entries),
            "total_exps": len(self._exp_latest),
            "max_step": max_step,
            "status_counts": dict(status_counter),
            "parse_errors": self._parse_errors,
            "debug_test_count": len(self._debug_events),
        }

    def get_updates(self, cursor: int, limit: int) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            cursor = max(0, int(cursor))
            limit = max(1, min(int(limit), 2000))
            cursor_reset = False
            if cursor > len(self._entries):
                cursor = 0
                cursor_reset = True

            new_entries = self._entries[cursor: cursor + limit]
            cursor_next = cursor + len(new_entries)
            exp_latest = [
                {
                    "exp_index": exp_index,
                    "step": item.get("step", 0),
                    "status": item.get("status", "running"),
                    "index": item.get("index", 0),
                }
                for exp_index, item in sorted(self._exp_latest.items(), key=lambda x: x[0])
            ]

            return {
                "format": self._format,
                "mode": "incremental_jsonl" if self._format == "jsonl" else "compat_json_snapshot",
                "warning": self._warning,
                "cursor_reset": cursor_reset,
                "cursor_next": cursor_next,
                "new_entries": new_entries,
                "summary": self._summary_locked(),
                "exp_latest": exp_latest,
                "file": {
                    "path": str(self.path),
                    "size_bytes": self._last_size,
                    "mtime": _iso_time_from_ns(self._last_mtime_ns),
                },
            }

    def get_entry(self, index: int) -> dict[str, Any] | None:
        self.refresh()
        with self._lock:
            idx = int(index)
            if idx < 0 or idx >= len(self._entries):
                return None
            return self._entries[idx]

    def get_exp_entries(self, exp_index: int, limit: int = 300) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            arr = [e for e in self._entries if _as_int(e.get("exp_index"), -1) == exp_index]
            arr = arr[-max(1, min(limit, 2000)):]
            return arr

    def get_debug_tests(self, exp_index: int | None, limit: int = 260) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            events = self._debug_events
            if exp_index is not None:
                events = [e for e in events if _as_int(e.get("exp_index"), -1) == exp_index]
            events = events[-max(1, min(limit, 2000)):]
            return list(events)

    def get_debug_test_cards(
        self,
        exp_index: int | None = None,
        limit: int = 260,
    ) -> list[dict[str, Any]]:
        events = self.get_debug_tests(exp_index=exp_index, limit=limit)
        grouped: dict[int, list[dict[str, Any]]] = {}
        for e in events:
            exp = _as_int(e.get("exp_index"), -1)
            if exp is None or exp <= 0:
                continue
            grouped.setdefault(exp, []).append(e)

        cards: list[dict[str, Any]] = []
        for exp in sorted(grouped.keys()):
            calls = grouped[exp]
            success_count = sum(1 for c in calls if str(c.get("status", "")).lower() == "success")
            failed_count = sum(1 for c in calls if str(c.get("status", "")).lower() == "failed")
            cards.append(
                {
                    "exp_index": exp,
                    "total_calls": len(calls),
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "calls": calls,
                }
            )
        return cards

    def get_pod_items(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            items: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()

            for e in reversed(self._debug_events):
                pod_name = str(e.get("pod_name", "")).strip()
                if not pod_name:
                    continue
                namespace = str(e.get("namespace", "default")).strip() or "default"
                exp_index = _as_int(e.get("exp_index"), -1)
                key = (pod_name, namespace, "debug")
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "source": "debug_test",
                        "exp_index": exp_index if exp_index > 0 else None,
                        "pod_name": pod_name,
                        "job_name": "",
                        "namespace": namespace,
                    }
                )

            for m in self._manifest_items:
                job_name = str(m.get("job_name", "")).strip()
                namespace = str(m.get("namespace", "default")).strip() or "default"
                key = (job_name, namespace, "manifest")
                if key in seen:
                    continue
                seen.add(key)
                items.append(dict(m))

            return items

    def _resolve_pod_from_job(self, job_name: str, namespace: str) -> tuple[str | None, str]:
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
        except Exception as e:
            return None, f"failed to run kubectl get pods: {e}"
        if proc.returncode != 0:
            return None, proc.stderr.strip() or proc.stdout.strip() or "kubectl get pods failed"

        names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not names:
            return None, "no pod found for this job"
        return names[-1], "ok"

    def get_pod_logs(self, pod_name: str, namespace: str, tail: int = 200) -> dict[str, Any]:
        pod_name = str(pod_name).strip()
        namespace = str(namespace).strip() or "default"
        tail = max(20, min(5000, int(tail)))

        if not pod_name:
            return {
                "ok": False,
                "source": "live_kubectl",
                "pod_name": "",
                "namespace": namespace,
                "logs": "",
                "error": "pod_name is required",
            }

        cmd = ["kubectl", "logs", "-n", namespace, pod_name, f"--tail={tail}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        except Exception as e:
            return {
                "ok": False,
                "source": "live_kubectl",
                "pod_name": pod_name,
                "namespace": namespace,
                "logs": "",
                "error": f"failed to run kubectl logs: {e}",
            }

        if proc.returncode != 0:
            return {
                "ok": False,
                "source": "live_kubectl",
                "pod_name": pod_name,
                "namespace": namespace,
                "logs": proc.stdout,
                "error": proc.stderr.strip() or "kubectl logs failed",
            }

        return {
            "ok": True,
            "source": "live_kubectl",
            "pod_name": pod_name,
            "namespace": namespace,
            "logs": proc.stdout,
            "error": "",
        }

    def get_run_results(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            rounds: dict[int, dict[str, Any]] = {}
            rounds.update(self._log_round_results)
            rounds.update(self._round_results)

            manifest_by_round: dict[int, dict[str, Any]] = {}
            for item in self._manifest_items:
                exp = _as_int(item.get("exp_index"), -1)
                if exp is None or exp <= 0:
                    continue
                # same round keep the latest file path lexicographically
                prev = manifest_by_round.get(exp)
                if prev is None:
                    manifest_by_round[exp] = item
                    continue
                prev_path = str(prev.get("manifest_path", ""))
                cur_path = str(item.get("manifest_path", ""))
                if cur_path > prev_path:
                    manifest_by_round[exp] = item

            ordered: list[dict[str, Any]] = []
            for round_index in sorted(rounds.keys()):
                item = dict(rounds[round_index])
                manifest = manifest_by_round.get(round_index)
                if manifest:
                    manifest_path = str(manifest.get("manifest_path", "")).strip()
                    item["manifest_path"] = manifest_path
                    item["job_name"] = str(manifest.get("job_name", ""))
                    item["namespace"] = str(manifest.get("namespace", "default"))
                    if manifest_path and Path(manifest_path).exists():
                        text = Path(manifest_path).read_text(encoding="utf-8", errors="replace")
                        item["submit_script"] = _clip_text(text, 32000)
                        item["submit_script_truncated"] = len(text) > 32000
                    else:
                        item["submit_script"] = ""
                        item["submit_script_truncated"] = False
                else:
                    item["manifest_path"] = ""
                    item["job_name"] = ""
                    item["namespace"] = "default"
                    item["submit_script"] = ""
                    item["submit_script_truncated"] = False
                ordered.append(item)

            final_rounds = self._log_final_rounds if self._log_final_rounds is not None else (ordered[-1]["round_index"] if ordered else None)
            return {
                "final": {
                    "status": self._log_final_status,
                    "rounds": final_rounds,
                    "source": "log+trajectory",
                },
                "rounds": ordered,
            }


class RunManager:
    def __init__(self, source_path: Path, preview_chars: int = 220) -> None:
        self.source_path = source_path
        self.preview_chars = preview_chars

        self._lock = threading.Lock()
        self._stores: dict[str, TrajectoryStore] = {}
        self._runs: list[dict[str, Any]] = []

        self._last_scan_ts = 0.0
        self._scan_interval_sec = 5.0

        self.refresh(force=True)

    def _discover_run_candidates(self) -> list[tuple[Path, Path | None, str | None]]:
        src = self.source_path.expanduser().resolve()
        candidates: list[tuple[Path, Path | None, str | None]] = []
        seen_paths: set[Path] = set()

        def add_candidate(path: Path) -> None:
            path = path.resolve()
            if path in seen_paths:
                return
            seen_paths.add(path)
            task_id: str | None = None
            run_dir: Path | None = None
            if len(path.parents) >= 3 and path.parents[1].name == "trajectories":
                task_id = path.parents[0].name
                run_dir = path.parents[2]
            candidates.append((path, run_dir, task_id))

        def collect_from_run_dir(run_dir: Path) -> None:
            tr_dir = run_dir / "trajectories"
            if not tr_dir.exists() or not tr_dir.is_dir():
                return
            for task_dir in sorted(tr_dir.iterdir()):
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

        # 1) src is a run directory
        collect_from_run_dir(src)
        if candidates:
            return candidates

        # 2) src is workspace root containing many run directories (flat legacy layout)
        if src.exists() and src.is_dir():
            for run_dir in sorted(src.iterdir()):
                if not run_dir.is_dir():
                    continue
                collect_from_run_dir(run_dir)

            if candidates:
                return candidates

            # 3) recursive scan for grouped layout:
            # workspace_root/simulator/task/model/date/run_id/trajectories/task_x/trajectory.jsonl
            for tr_dir in sorted(src.rglob("trajectories")):
                if not tr_dir.is_dir():
                    continue
                collect_from_run_dir(tr_dir.parent)

        return candidates

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_scan_ts) < self._scan_interval_sec:
            return
        self._last_scan_ts = now

        with self._lock:
            candidates = self._discover_run_candidates()

            new_runs: list[dict[str, Any]] = []
            seen_run_ids: set[str] = set()
            new_stores: dict[str, TrajectoryStore] = {}

            for path, run_dir, task_id in candidates:
                if not path.exists():
                    continue
                stat = path.stat()
                fmt = "jsonl" if path.suffix.lower() == ".jsonl" else "json"

                if run_dir and task_id:
                    run_key = run_dir.name
                    try:
                        source_root = self.source_path.expanduser().resolve()
                        rel_run_dir = run_dir.resolve().relative_to(source_root)
                        rel_text = str(rel_run_dir).strip()
                        if rel_text and rel_text != ".":
                            run_key = rel_text
                    except Exception:
                        pass
                    run_id_base = f"{run_key}:{task_id}"
                    label = f"{run_key} / {task_id} ({fmt})"
                else:
                    run_id_base = path.stem
                    label = f"{path.name} ({fmt})"

                run_id = run_id_base
                suffix = 2
                while run_id in seen_run_ids:
                    run_id = f"{run_id_base}#{suffix}"
                    suffix += 1
                seen_run_ids.add(run_id)

                store = self._stores.get(run_id)
                if store is None or store.path != path:
                    store = TrajectoryStore(
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
                        "mtime_ns": int(stat.st_mtime_ns),
                        "size_bytes": int(stat.st_size),
                    }
                )

            new_runs.sort(key=lambda x: x.get("mtime_ns", 0), reverse=True)
            self._runs = new_runs
            self._stores = new_stores

    def list_runs(self, force: bool = False) -> dict[str, Any]:
        self.refresh(force=force)
        with self._lock:
            runs = []
            for run in self._runs:
                runs.append(
                    {
                        "run_id": run["run_id"],
                        "label": run["label"],
                        "format": run["format"],
                        "path": run["path"],
                        "size_bytes": run["size_bytes"],
                        "mtime": _iso_time_from_ns(run["mtime_ns"]),
                    }
                )
            default_run_id = runs[0]["run_id"] if runs else ""
            return {"runs": runs, "default_run_id": default_run_id}

    def get_store(self, run_id: str | None) -> tuple[str | None, TrajectoryStore | None]:
        self.refresh(force=False)
        with self._lock:
            if not self._runs:
                return None, None

            pick = run_id or self._runs[0]["run_id"]
            if pick not in self._stores:
                pick = self._runs[0]["run_id"]
            return pick, self._stores.get(pick)

    def get_run_meta(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            for run in self._runs:
                if run["run_id"] == run_id:
                    return {
                        "run_id": run["run_id"],
                        "label": run["label"],
                        "format": run["format"],
                        "path": run["path"],
                        "size_bytes": run["size_bytes"],
                        "mtime": _iso_time_from_ns(run["mtime_ns"]),
                    }
            return None


class TrajectoryHandler(BaseHTTPRequestHandler):
    manager: RunManager

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_js_file(self, path: Path) -> None:
        if not path.exists():
            self._send_json({"error": "js file not found", "path": str(path)}, code=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return

        if parsed.path == "/app.js":
            self._send_js_file(APP_JS_PATH)
            return

        if parsed.path == "/api/runs":
            refresh = str(query.get("refresh", ["0"])[0]).lower() in {"1", "true", "yes"}
            self._send_json(self.manager.list_runs(force=refresh))
            return

        run_id = str(query.get("run_id", [""])[0]).strip() or None
        actual_run_id, store = self.manager.get_store(run_id)
        if store is None or actual_run_id is None:
            self._send_json({"error": "no run available"}, code=404)
            return

        run_meta = self.manager.get_run_meta(actual_run_id) or {"run_id": actual_run_id, "label": actual_run_id}

        if parsed.path == "/api/updates":
            cursor = _as_int(query.get("cursor", ["0"])[0], 0)
            limit = _as_int(query.get("limit", ["500"])[0], 500)
            payload = store.get_updates(cursor=cursor, limit=limit)
            payload["run"] = run_meta
            payload["run_id"] = actual_run_id
            self._send_json(payload)
            return

        if parsed.path == "/api/entry":
            index = _as_int(query.get("index", ["-1"])[0], -1)
            item = store.get_entry(index)
            if item is None:
                self._send_json({"error": "entry not found", "index": index}, code=404)
            else:
                self._send_json({"entry": item, "run": run_meta, "run_id": actual_run_id})
            return

        if parsed.path == "/api/exp_entries":
            exp_index = _as_int(query.get("exp_index", ["-1"])[0], -1)
            if exp_index <= 0:
                self._send_json({"error": "exp_index is required and must be > 0"}, code=400)
                return
            limit = _as_int(query.get("limit", ["300"])[0], 300)
            entries = store.get_exp_entries(exp_index=exp_index, limit=limit)
            self._send_json(
                {
                    "run": run_meta,
                    "run_id": actual_run_id,
                    "exp_index": exp_index,
                    "entries": entries,
                }
            )
            return

        if parsed.path == "/api/debug_tests":
            exp_index_raw = str(query.get("exp_index", [""])[0]).strip()
            exp_index = _as_int(exp_index_raw, -1) if exp_index_raw else None
            if exp_index is not None and exp_index <= 0:
                exp_index = None
            limit = _as_int(query.get("limit", ["260"])[0], 260)
            events = store.get_debug_tests(exp_index=exp_index, limit=limit)
            cards = store.get_debug_test_cards(exp_index=exp_index, limit=limit)
            self._send_json(
                {
                    "run": run_meta,
                    "run_id": actual_run_id,
                    "exp_index": exp_index,
                    "events": events,
                    "cards": cards,
                }
            )
            return

        if parsed.path == "/api/pods":
            items = store.get_pod_items()
            self._send_json({"run": run_meta, "run_id": actual_run_id, "items": items})
            return

        if parsed.path == "/api/pod_logs":
            pod_name = str(query.get("pod_name", [""])[0]).strip()
            job_name = str(query.get("job_name", [""])[0]).strip()
            namespace = str(query.get("namespace", ["default"])[0]).strip() or "default"
            tail = _as_int(query.get("tail", ["200"])[0], 200)

            if not pod_name and job_name:
                resolved, reason = store._resolve_pod_from_job(job_name=job_name, namespace=namespace)
                if not resolved:
                    self._send_json(
                        {
                            "ok": False,
                            "source": "resolve_job",
                            "job_name": job_name,
                            "pod_name": "",
                            "namespace": namespace,
                            "logs": "",
                            "error": reason,
                        }
                    )
                    return
                pod_name = resolved

            payload = store.get_pod_logs(pod_name=pod_name, namespace=namespace, tail=tail)
            payload["job_name"] = job_name
            payload["run"] = run_meta
            payload["run_id"] = actual_run_id
            self._send_json(payload)
            return

        if parsed.path == "/api/run_results":
            result = store.get_run_results()
            result["run"] = run_meta
            result["run_id"] = actual_run_id
            self._send_json(result)
            return

        self._send_json({"error": "not found", "path": parsed.path}, code=404)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web monitor for EmboMaster trajectory files")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspaces",
        help="Trajectory source path: workspace root, one run dir, or one trajectory file",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Deprecated alias of --source; kept for compatibility",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", default=8765, type=int, help="Bind port")
    parser.add_argument("--preview-chars", default=220, type=int, help="Message preview max chars")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = args.trajectory if args.trajectory is not None else args.source
    manager = RunManager(source_path=source, preview_chars=max(60, args.preview_chars))
    runs = manager.list_runs(force=True)

    handler_cls = TrajectoryHandler
    handler_cls.manager = manager

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"[traj-monitor] serving on http://{args.host}:{args.port}")
    print(f"[traj-monitor] source: {source}")
    print(f"[traj-monitor] discovered runs: {len(runs.get('runs', []))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
