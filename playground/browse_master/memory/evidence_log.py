"""Evidence log kept outside the MDP workspace."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_EVIDENCE_CONTENT = 12000


@dataclass
class EvidenceEntry:
    """One tool observation stored for debugging and source tracking."""

    evidence_id: str
    tool_name: str
    arguments: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EvidenceLog:
    """Append-only evidence store that stays out of the prompt by default."""

    def __init__(self) -> None:
        self.entries: list[EvidenceEntry] = []

    def add(
        self,
        *,
        tool_name: str,
        arguments: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> EvidenceEntry:
        evidence_id = f"E{len(self.entries) + 1}"
        entry = EvidenceEntry(
            evidence_id=evidence_id,
            tool_name=tool_name,
            arguments=arguments,
            content=self._trim_content(content),
            meta=meta or {},
        )
        self.entries.append(entry)
        return entry

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self.entries]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dicts(), handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _trim_content(content: str) -> str:
        if len(content) <= MAX_EVIDENCE_CONTENT:
            return content
        half = MAX_EVIDENCE_CONTENT // 2
        return (
            content[:half]
            + "\n\n...[evidence truncated]...\n\n"
            + content[-half:]
        )
