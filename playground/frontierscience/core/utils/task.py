"""Task parsing helpers for FrontierScience playground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK_META_START = "[frontierscience_task_meta]"
TASK_META_END = "[/frontierscience_task_meta]"
TASK_META_RESERVED_KEYS = {"task_id", "task_type"}


@dataclass(frozen=True)
class FrontierScienceTaskRuntime:
    """Normalized runtime task payload derived from task description metadata."""

    task_id: str
    task_type: str
    problem: str
    task_input_data: dict[str, Any]


def extract_task_meta(task_description: str) -> tuple[str, dict[str, str]]:
    """Split optional task metadata block from the user-visible problem text."""
    text = task_description or ""
    if not text.lstrip().startswith(TASK_META_START):
        return text, {}

    lines = text.splitlines()
    meta: dict[str, str] = {}
    inside_meta = False
    body_start = 0
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not inside_meta:
            if line == TASK_META_START:
                inside_meta = True
            elif line:
                return text, {}
            continue
        if line == TASK_META_END:
            body_start = index + 1
            break
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            meta[key] = value
    else:
        return text, {}

    return "\n".join(lines[body_start:]).lstrip(), meta


def build_task_runtime(task_description: str, default_task_id: str) -> FrontierScienceTaskRuntime:
    """Normalize task metadata into a compact runtime struct."""
    cleaned_description, task_meta = extract_task_meta(task_description)
    runtime_task_id = task_meta.get("task_id", default_task_id) or default_task_id
    runtime_task_type = task_meta.get("task_type", "frontier_science") or "frontier_science"
    task_input_data = {
        key: value
        for key, value in task_meta.items()
        if key not in TASK_META_RESERVED_KEYS
    }
    return FrontierScienceTaskRuntime(
        task_id=runtime_task_id,
        task_type=runtime_task_type,
        problem=cleaned_description,
        task_input_data=task_input_data,
    )
