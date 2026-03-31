"""Task-level shared tool trace tree for FrontierScience search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACEABLE_TOOL_NAMES = {"search_web", "google_scholar", "visit_web", "read_paper_pdf"}


def _safe_text(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


@dataclass
class FrontierScienceSharedContext:
    """Maintain a tree-structured directory of tool calls across one task."""

    root_dir: Path
    trajectory_dir: Path | None = None
    max_records: int = 32
    records: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: Path = field(init=False)
    nodes_dir: Path = field(init=False)
    trajectory_jsonl_path: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir = self.root_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root_dir / "manifest.json"
        if self.trajectory_dir is not None:
            self.trajectory_dir = Path(self.trajectory_dir)
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            self.trajectory_jsonl_path = self.trajectory_dir / "trajectory.jsonl"
        self._flush_manifest()

    def add_trajectory(
        self,
        *,
        node_id: str,
        parent_id: str | None,
        action_type: str,
        trajectory: Any,
        metric_result: dict[str, Any] | None = None,
        search_node: Any | None = None,
        exploration_weight: float | None = None,
    ) -> None:
        traj_dict = trajectory.model_dump() if hasattr(trajectory, "model_dump") else trajectory if isinstance(trajectory, dict) else {}
        dialogs = traj_dict.get("dialogs") if isinstance(traj_dict, dict) else getattr(trajectory, "dialogs", []) or []
        node_records: list[dict[str, Any]] = []
        for dialog in dialogs or []:
            messages = dialog.get("messages", []) if isinstance(dialog, dict) else getattr(dialog, "messages", []) or []
            pending: dict[str, list[dict[str, Any]]] = {}
            for message in messages:
                role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
                if hasattr(role, "value"):
                    role = role.value
                if role == "assistant":
                    tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
                    tool_calls = tool_calls or []
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                        name = function.get("name", "") if isinstance(function, dict) else getattr(function, "name", "") or ""
                        if name not in TRACEABLE_TOOL_NAMES:
                            continue
                        arguments = function.get("arguments", "") if isinstance(function, dict) else getattr(function, "arguments", "") or ""
                        record = {
                            "node_id": node_id,
                            "parent_id": parent_id,
                            "action_type": action_type,
                            "tool": name,
                            "arguments": _safe_text(arguments),
                            "results": [],
                        }
                        pending.setdefault(name, []).append(record)
                        node_records.append(record)
                        self.records.append(record)
                elif role == "tool":
                    name = message.get("name") if isinstance(message, dict) else getattr(message, "name", "") or ""
                    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "") or ""
                    queue = pending.get(name) or []
                    if queue:
                        queue.pop(0)["results"].append(_safe_text(content))

        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

        metric_summary = None
        if metric_result:
            metric_summary = {
                "overall_score": metric_result.get("overall_score"),
                "final_answer_score": metric_result.get("final_answer_score"),
                "trajectory_score": metric_result.get("trajectory_score"),
                "is_valid": metric_result.get("is_valid"),
                "reason": metric_result.get("reason"),
                "strengths": list(metric_result.get("strengths") or [])[:3],
                "weaknesses": list(metric_result.get("weaknesses") or [])[:3],
                "improvement_suggestions": list(metric_result.get("improvement_suggestions") or [])[:3],
            }
        search_stats = None
        if search_node is not None and hasattr(search_node, "search_stats"):
            search_stats = search_node.search_stats(exploration_weight)
        node_payload = {
            "node_id": node_id,
            "parent_id": parent_id,
            "action_type": action_type,
            "search_stats": search_stats,
            "metric": metric_summary,
            "tool_records": node_records,
        }
        (self.nodes_dir / f"{node_id}.json").write_text(
            json.dumps(node_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._append_trajectory_jsonl(
            node_id=node_id,
            parent_id=parent_id,
            action_type=action_type,
            trajectory=traj_dict,
            metric_summary=metric_summary,
            search_stats=search_stats,
            node_records=node_records,
            search_node=search_node,
        )
        self._flush_manifest()

    def _append_trajectory_jsonl(
        self,
        *,
        node_id: str,
        parent_id: str | None,
        action_type: str,
        trajectory: dict[str, Any],
        metric_summary: dict[str, Any] | None,
        search_stats: dict[str, Any] | None,
        node_records: list[dict[str, Any]],
        search_node: Any | None,
    ) -> None:
        if self.trajectory_jsonl_path is None:
            return

        payload = {
            "node_id": node_id,
            "parent_id": parent_id,
            "action_type": action_type,
            "depth": getattr(search_node, "depth", None),
            "current_answer": getattr(search_node, "current_answer", None),
            "state_summary": getattr(search_node, "state_summary", None),
            "metric": metric_summary,
            "search_stats": search_stats,
            "tool_records": node_records,
            "trajectory": trajectory,
        }
        with self.trajectory_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _flush_manifest(self) -> None:
        payload = {
            "root_dir": str(self.root_dir),
            "records": self.records[-self.max_records :],
            "node_files": sorted(path.name for path in self.nodes_dir.glob("*.json")),
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def render_for_prompt(self, limit: int = 10) -> str:
        recent = self.records[-limit:]
        lines = [
            f"Tool trace root: {self.root_dir}",
            f"Manifest: {self.manifest_path}",
            "Recent tool records:",
        ]
        if not recent:
            lines.append("No tool history yet.")
        else:
            for item in recent:
                lines.append(
                    json.dumps(
                        {
                            "node_id": item["node_id"],
                            "action_type": item["action_type"],
                            "tool": item["tool"],
                            "arguments": item["arguments"],
                            "results": item["results"][:2],
                        },
                        ensure_ascii=False,
                    )
                )
        lines.append("Use the manifest or node JSON files directly if you need exact previous tool operations/results.")
        return "\n".join(lines)
