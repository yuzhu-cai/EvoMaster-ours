"""MCTS visualization utilities for PHY Master."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "template" / "visualization_template.html"
INJECT_MARKER = "<!-- __DATA_INJECT__ -->"


def _compute_tree_layout(nodes: list[dict[str, Any]], root_id: str = "root") -> dict[str, tuple[float, float]]:
    by_id = {n["node_id"]: n for n in nodes}
    children_map: dict[str, list[str]] = {n["node_id"]: list(n.get("children", [])) for n in nodes}

    levels: dict[int, list[str]] = {}
    queue = [(root_id, 0)]
    visited = set()
    while queue:
        node_id, depth = queue.pop(0)
        if node_id in visited or node_id not in by_id:
            continue
        visited.add(node_id)
        levels.setdefault(depth, []).append(node_id)
        for child_id in children_map.get(node_id, []):
            queue.append((child_id, depth + 1))

    if not levels:
        return {}

    max_depth = max(levels.keys())
    coords: dict[str, tuple[float, float]] = {}
    for depth, ids in levels.items():
        count = max(1, len(ids))
        y = 0.08 + (0.84 * (depth / max(1, max_depth)))
        for i, node_id in enumerate(ids):
            x = 0.1 + (0.8 * ((i + 1) / (count + 1)))
            coords[node_id] = (x, y)
    return coords


def build_payload(
    *,
    nodes: list[dict[str, Any]],
    root_id: str,
    best_node_id: str | None,
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> dict[str, Any]:
    coords = _compute_tree_layout(nodes, root_id=root_id)

    edges: list[list[str]] = []
    for node in nodes:
        for child_id in node.get("children", []):
            if child_id in coords and node["node_id"] in coords:
                edges.append([node["node_id"], child_id])

    payload_nodes = []
    for node in nodes:
        x, y = coords.get(node["node_id"], (0.5, 0.5))
        copied = dict(node)
        copied["viz_x"] = x
        copied["viz_y"] = y
        payload_nodes.append(copied)

    return {
        "task_description": task_description,
        "subtasks": subtasks if subtasks is not None else [],
        "summary": summary or "",
        "root_id": root_id,
        "best_node_id": best_node_id,
        "nodes": payload_nodes,
        "edges": edges,
    }


def build_mcts_html(
    *,
    nodes: list[dict[str, Any]],
    root_id: str,
    best_node_id: str | None,
    task_description: str,
    subtasks: Any = None,
    summary: str = "",
) -> str:
    payload = build_payload(
        nodes=nodes,
        root_id=root_id,
        best_node_id=best_node_id,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    payload_json = json.dumps(payload, ensure_ascii=False)

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    inject = f"<script>window.__PHY_MCTS_DATA__ = {payload_json};</script>"
    return template.replace(INJECT_MARKER, inject)


def write_mcts_html(
    output_path: str | Path,
    *,
    nodes: list[dict[str, Any]],
    root_id: str = "root",
    best_node_id: str | None = None,
    task_description: str = "",
    subtasks: Any = None,
    summary: str = "",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_mcts_html(
        nodes=nodes,
        root_id=root_id,
        best_node_id=best_node_id,
        task_description=task_description,
        subtasks=subtasks,
        summary=summary,
    )
    out.write_text(html, encoding="utf-8")
    return out
