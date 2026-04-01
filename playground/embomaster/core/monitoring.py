"""Structured monitoring snapshots for EmboMaster runs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated] ..."


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


class EmboMasterMonitorWriter:
    """Persist round and run summaries for dashboard consumption."""

    VERSION = 1

    def __init__(self, run_dir: str | Path, task_id: str) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.task_id = str(task_id).strip() or "task_0"
        self.root = self.run_dir / "monitor" / "tasks" / self.task_id
        self.rounds_dir = self.root / "rounds"
        self.logs_dir = self.root / "logs"
        self.events_path = self.root / "events.jsonl"
        self.latest_path = self.root / "latest.json"
        self.final_path = self.root / "final.json"
        self._lock = threading.Lock()

        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def record_round(
        self,
        round_result: dict[str, Any],
        best_metric: float | None,
        best_round_index: int | None,
        total_rounds_so_far: int,
    ) -> None:
        round_index = int(round_result.get("round_index", 0) or 0)
        if round_index <= 0:
            return

        with self._lock:
            snapshot = self._build_round_snapshot(
                round_result=round_result,
                best_metric=best_metric,
                best_round_index=best_round_index,
                total_rounds_so_far=total_rounds_so_far,
            )
            round_path = self.rounds_dir / f"round_{round_index:03d}.json"
            round_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._append_event(
                {
                    "event": "round_recorded",
                    "round_index": round_index,
                    "status": snapshot.get("status"),
                    "k8s_status": snapshot.get("k8s_status"),
                    "metric_value": snapshot.get("metric_value"),
                    "updated_at": snapshot.get("updated_at"),
                }
            )
            self._rebuild_latest_locked()

    def record_final_result(
        self,
        result: dict[str, Any],
        task_description: str,
    ) -> None:
        with self._lock:
            final_payload = self._build_final_payload(
                result=result,
                task_description=task_description,
            )
            self.final_path.write_text(
                json.dumps(final_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._append_event(
                {
                    "event": "run_completed",
                    "status": final_payload.get("status"),
                    "total_rounds": final_payload.get("total_rounds"),
                    "best_metric": final_payload.get("best_metric"),
                    "updated_at": final_payload.get("updated_at"),
                }
            )
            self._rebuild_latest_locked(final_payload=final_payload)

    def _build_round_snapshot(
        self,
        round_result: dict[str, Any],
        best_metric: float | None,
        best_round_index: int | None,
        total_rounds_so_far: int,
    ) -> dict[str, Any]:
        round_index = int(round_result.get("round_index", 0) or 0)
        k8s_payload = self._build_k8s_payload(round_result, round_index)
        return {
            "monitor_version": self.VERSION,
            "task_id": self.task_id,
            "round_index": round_index,
            "status": str(round_result.get("status", "unknown")),
            "steps": int(round_result.get("steps", 0) or 0),
            "coding_result": str(round_result.get("coding_result", "") or ""),
            "feedback": str(round_result.get("feedback", "") or ""),
            "k8s_status": str(round_result.get("k8s_status", "unknown")),
            "metric_value": _to_float(round_result.get("metric_value")),
            "metric_source": str(round_result.get("metric_source", "") or ""),
            "metric_valid": bool(round_result.get("metric_valid", False)),
            "result_valid": bool(round_result.get("result_valid", True)),
            "validation_errors": list(round_result.get("validation_errors", []) or []),
            "workspace_id": str(round_result.get("workspace_id", "") or ""),
            "parent_workspace_id": str(round_result.get("parent_workspace_id", "") or ""),
            "parent_choice_used": str(round_result.get("parent_choice_used", "") or ""),
            "parent_choice_reason": str(round_result.get("parent_choice_reason", "") or ""),
            "parent_recommendation": round_result.get("parent_recommendation"),
            "workspace_codebase_path": str(round_result.get("workspace_codebase_path", "") or ""),
            "workspace_source_type": str(round_result.get("workspace_source_type", "") or ""),
            "workspace_large_dirs_count": int(round_result.get("workspace_large_dirs_count", 0) or 0),
            "workspace_large_dirs": list(round_result.get("workspace_large_dirs", []) or []),
            "submission_dir": str(round_result.get("submission_dir", "") or ""),
            "session_dir": str(round_result.get("session_dir", "") or ""),
            "artifacts_summary": dict(round_result.get("artifacts_summary", {}) or {}),
            "debug_pod_prepare": self._sanitize_command_payload(round_result.get("debug_pod_prepare")),
            "debug_pod_cleanup_before_submit": self._sanitize_command_payload(
                round_result.get("debug_pod_cleanup_before_submit")
            ),
            "debug_pod_cleanup_final": self._sanitize_command_payload(
                round_result.get("debug_pod_cleanup_final")
            ),
            "k8s": k8s_payload,
            "best_metric_so_far": _to_float(best_metric),
            "best_round_index_so_far": best_round_index,
            "total_rounds_so_far": int(total_rounds_so_far),
            "updated_at": _utc_now_iso(),
        }

    def _build_k8s_payload(
        self,
        round_result: dict[str, Any],
        round_index: int,
    ) -> dict[str, Any] | None:
        raw = round_result.get("k8s_result")
        if not isinstance(raw, dict):
            return None

        logs_obj = raw.get("logs")
        logs_text = ""
        if isinstance(logs_obj, dict):
            logs_text = str(logs_obj.get("output", "") or logs_obj.get("stdout", ""))
        logs_rel_path = ""
        if logs_text:
            log_path = self.logs_dir / f"round_{round_index:03d}.k8s.log"
            log_path.write_text(logs_text, encoding="utf-8")
            logs_rel_path = str(log_path.relative_to(self.root))

        pods_raw = raw.get("pods")
        pods: list[dict[str, Any]] = []
        if isinstance(pods_raw, list):
            for item in pods_raw:
                if not isinstance(item, dict):
                    continue
                pods.append(
                    {
                        "pod_name": str(item.get("pod_name", "") or ""),
                        "namespace": str(item.get("namespace", "") or ""),
                        "status": str(item.get("status", "") or ""),
                        "node_name": str(item.get("node_name", "") or ""),
                        "start_time": str(item.get("start_time", "") or ""),
                        "host_ip": str(item.get("host_ip", "") or ""),
                        "pod_ip": str(item.get("pod_ip", "") or ""),
                        "container_statuses": list(item.get("container_statuses", []) or []),
                    }
                )

        return {
            "manifest_path": str(raw.get("manifest_path", "") or ""),
            "prepared_manifest_path": str(raw.get("prepared_manifest_path", "") or ""),
            "job_name": str(raw.get("job_name", "") or ""),
            "namespace": str(raw.get("namespace", "") or ""),
            "submit": self._sanitize_command_payload(raw.get("submit")),
            "wait": self._sanitize_command_payload(raw.get("wait")),
            "cleanup": self._sanitize_command_payload(raw.get("cleanup")),
            "logs_preview": _clip_text(logs_text, 16000),
            "logs_path": logs_rel_path,
            "pods": pods,
        }

    def _sanitize_command_payload(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        payload: dict[str, Any] = {}
        for key in (
            "status",
            "pod_name",
            "namespace",
            "job_name",
            "previous_status",
            "source",
            "manifest_path",
            "command",
            "exit_code",
            "error",
            "reason",
        ):
            if key in value:
                payload[key] = value.get(key)
        for key in ("stdout", "stderr", "output", "wait_stdout", "wait_stderr", "describe"):
            if key in value and value.get(key):
                payload[key] = _clip_text(value.get(key), 8000)
        if "details" in value and isinstance(value.get("details"), dict):
            payload["details"] = value.get("details")
        return payload or None

    def _build_final_payload(
        self,
        result: dict[str, Any],
        task_description: str,
    ) -> dict[str, Any]:
        return {
            "monitor_version": self.VERSION,
            "task_id": self.task_id,
            "status": str(result.get("status", "unknown")),
            "total_rounds": int(result.get("total_rounds", 0) or 0),
            "stopped_reason": str(result.get("stopped_reason", "") or ""),
            "best_metric": _to_float(result.get("best_metric")),
            "best_round_index": result.get("best_round_index"),
            "invalid_round_count": int(result.get("invalid_round_count", 0) or 0),
            "task_description_excerpt": _clip_text(task_description, 4000),
            "updated_at": _utc_now_iso(),
        }

    def _rebuild_latest_locked(self, final_payload: dict[str, Any] | None = None) -> None:
        rounds: list[dict[str, Any]] = []
        for path in sorted(self.rounds_dir.glob("round_*.json")):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rounds.append(item)

        best_metric: float | None = None
        best_round_index: int | None = None
        invalid_round_count = 0
        for item in rounds:
            round_index = int(item.get("round_index", 0) or 0)
            metric_value = _to_float(item.get("metric_value"))
            if metric_value is not None and (best_metric is None or metric_value > best_metric):
                best_metric = metric_value
                best_round_index = round_index
            if not bool(item.get("result_valid", True)):
                invalid_round_count += 1

        if final_payload is None and self.final_path.exists():
            try:
                loaded_final = json.loads(self.final_path.read_text(encoding="utf-8"))
                if isinstance(loaded_final, dict):
                    final_payload = loaded_final
            except json.JSONDecodeError:
                final_payload = None

        round_overview = []
        for item in rounds:
            round_overview.append(
                {
                    "round_index": int(item.get("round_index", 0) or 0),
                    "status": str(item.get("status", "unknown")),
                    "k8s_status": str(item.get("k8s_status", "unknown")),
                    "metric_value": _to_float(item.get("metric_value")),
                    "result_valid": bool(item.get("result_valid", True)),
                    "workspace_id": str(item.get("workspace_id", "") or ""),
                    "parent_workspace_id": str(item.get("parent_workspace_id", "") or ""),
                    "parent_choice_used": str(item.get("parent_choice_used", "") or ""),
                    "steps": int(item.get("steps", 0) or 0),
                }
            )

        latest_payload = {
            "monitor_version": self.VERSION,
            "task_id": self.task_id,
            "updated_at": _utc_now_iso(),
            "overview": {
                "status": (
                    str(final_payload.get("status", "running"))
                    if isinstance(final_payload, dict)
                    else (str(rounds[-1].get("status", "running")) if rounds else "unknown")
                ),
                "total_rounds": len(rounds),
                "best_metric": (
                    _to_float(final_payload.get("best_metric"))
                    if isinstance(final_payload, dict) and final_payload.get("best_metric") is not None
                    else best_metric
                ),
                "best_round_index": (
                    final_payload.get("best_round_index")
                    if isinstance(final_payload, dict) and final_payload.get("best_round_index") is not None
                    else best_round_index
                ),
                "invalid_round_count": (
                    int(final_payload.get("invalid_round_count", invalid_round_count) or 0)
                    if isinstance(final_payload, dict)
                    else invalid_round_count
                ),
                "stopped_reason": (
                    str(final_payload.get("stopped_reason", "") or "")
                    if isinstance(final_payload, dict)
                    else ""
                ),
            },
            "rounds": round_overview,
            "final": final_payload,
        }
        self.latest_path.write_text(
            json.dumps(latest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_event(self, payload: dict[str, Any]) -> None:
        item = {"timestamp": _utc_now_iso(), "task_id": self.task_id, **payload}
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False))
            fh.write("\n")
