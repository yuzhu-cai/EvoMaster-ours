"""Bounded PaperBench paper reader.

The agent can always use bash to read files, but whole-paper dumps quickly
consume the model context. This tool returns concise, bounded slices of
paper.md/addendum.md while never reading blacklist.txt.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class PaperBenchPaperToolParams(BaseToolParams):
    """Read bounded PaperBench paper context without touching blacklist.txt."""

    name: ClassVar[str] = "paperbench_paper"

    action: Literal["summary", "outline", "grep", "section"] = Field(
        default="summary",
        description="summary, outline, grep snippets, or a section matching query.",
    )
    paper_dir: str = Field(default="/home/paper", description="Read-only PaperBench paper directory.")
    query: str = Field(default="", description="Plain-text query for grep/section.")
    include_addendum: bool = Field(default=True, description="Include addendum.md when useful.")
    max_chars: int = Field(default=6000, ge=1000, le=30000, description="Maximum returned characters.")
    max_matches: int = Field(default=12, ge=1, le=50, description="Maximum grep snippets.")


class PaperBenchPaperTool(BaseTool):
    """Safe, bounded helper for reading PaperBench paper text."""

    name: ClassVar[str] = "paperbench_paper"
    params_class: ClassVar[type[BaseToolParams]] = PaperBenchPaperToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as exc:
            return f"Parameter validation error: {exc}", {"error": str(exc)}

        assert isinstance(params, PaperBenchPaperToolParams)
        paper_md = _safe_join(params.paper_dir, "paper.md")
        addendum_md = _safe_join(params.paper_dir, "addendum.md")
        paper = _read_text(session, paper_md)
        addendum = _read_text(session, addendum_md) if params.include_addendum else ""

        max_chars = min(params.max_chars, _action_cap(params.action))

        if params.action == "outline":
            text = _outline(paper, addendum, max_chars)
        elif params.action == "grep":
            text = _grep(paper, addendum, params.query, params.max_matches, max_chars)
        elif params.action == "section":
            text = _section(paper, addendum, params.query, max_chars)
        else:
            text = _summary(paper, addendum, max_chars)

        return text, {
            "action": params.action,
            "paper_chars": len(paper),
            "addendum_chars": len(addendum),
            "truncated_to": max_chars,
        }


def _safe_join(base: str, name: str) -> str:
    base_path = PurePosixPath(base)
    if name == "blacklist.txt":
        raise ValueError("blacklist.txt is intentionally inaccessible")
    return str(base_path / name)


def _read_text(session, path: str) -> str:
    try:
        if not session.is_file(path):
            return ""
        return session.read_file(path, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"[failed to read {path}: {exc}]"


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + "\n\n... [paperbench_paper output clipped] ...\n\n" + text[-tail:].lstrip()


def _action_cap(action: str) -> int:
    """Hard caps keep repeated paper reads from dominating the agent context."""
    return {
        "summary": 3500,
        "outline": 2500,
        "grep": 3500,
        "section": 4500,
    }.get(action, 3500)


def _headings(text: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    patterns = [
        re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE),
        re.compile(r"^\\(?:section|subsection|subsubsection)\*?\{(.+?)\}\s*$", re.MULTILINE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            title = match.group(2) if match.lastindex and match.lastindex >= 2 else match.group(1)
            rows.append((re.sub(r"\s+", " ", title).strip(), match.start()))
    rows.sort(key=lambda item: item[1])
    return rows


def _outline(paper: str, addendum: str, max_chars: int) -> str:
    lines = ["# Paper Outline", ""]
    heads = _headings(paper)
    if not heads:
        lines.append("(No Markdown/LaTeX headings detected in paper.md.)")
    else:
        for title, offset in heads[:220]:
            lines.append(f"- {title}  [char {offset}]")

    if addendum:
        lines.extend(["", "# Addendum Outline"])
        add_heads = _headings(addendum)
        if add_heads:
            for title, offset in add_heads[:120]:
                lines.append(f"- {title}  [addendum char {offset}]")
        else:
            lines.extend(_first_nonempty_lines(addendum, 20))
    return _clip("\n".join(lines), max_chars)


def _summary(paper: str, addendum: str, max_chars: int) -> str:
    chunks: list[str] = ["# PaperBench Paper Summary Extract", ""]
    chunks.append("## Title")
    chunks.append(_clip("\n".join(_first_nonempty_lines(paper, 6)), 900))

    abstract = _named_section(paper, "abstract", max_chars=5000)
    if abstract:
        chunks.extend(["", "## Abstract", _clip(abstract, 1800)])

    chunks.extend(["", "## High-Signal Snippets"])
    for keyword in (
        "contribution",
        "Algorithm 1",
        "Algorithm 2",
        "algorithm",
        "Experiment I",
        "Experiment II",
        "Experiment V",
        "baseline",
        "metric",
    ):
        snippet = _first_snippet(paper, keyword, context=420)
        if snippet:
            chunks.extend([f"\n### {keyword}", _clip(snippet, 950)])

    chunks.extend(["", "## Outline"])
    chunks.append(_outline(paper, "", 3500))

    if addendum:
        chunks.extend(["", "## Addendum Highlights"])
        chunks.append(_clip("\n".join(_first_nonempty_lines(addendum, 32)), 2500))

    return _clip("\n".join(chunks), max_chars)


def _grep(paper: str, addendum: str, query: str, max_matches: int, max_chars: int) -> str:
    query = query.strip()
    if not query:
        return "Provide a non-empty query for action='grep'."
    haystacks = [("paper.md", paper)]
    if addendum:
        haystacks.append(("addendum.md", addendum))

    lines = [f"# grep snippets for: {query!r}", ""]
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    count = 0
    for name, text in haystacks:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 650)
            end = min(len(text), match.end() + 650)
            snippet = text[start:end].strip()
            snippet = re.sub(r"\n{3,}", "\n\n", snippet)
            lines.extend([f"## {name} char {match.start()}", snippet, ""])
            count += 1
            if count >= max_matches:
                return _clip("\n".join(lines), max_chars)
    if count == 0:
        lines.append("No matches.")
    return _clip("\n".join(lines), max_chars)


def _section(paper: str, addendum: str, query: str, max_chars: int) -> str:
    query_low = query.strip().lower()
    if not query_low:
        return "Provide a non-empty query for action='section'."
    for name, text in (("paper.md", paper), ("addendum.md", addendum)):
        if not text:
            continue
        heads = _headings(text)
        for idx, (title, start) in enumerate(heads):
            if query_low not in title.lower():
                continue
            end = heads[idx + 1][1] if idx + 1 < len(heads) else len(text)
            return _clip(f"# {name}: {title}\n\n{text[start:end]}", max_chars)
    return _grep(paper, addendum, query, max_matches=4, max_chars=max_chars)


def _named_section(text: str, name: str, max_chars: int) -> str:
    if name.lower() == "abstract":
        latex = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.IGNORECASE | re.DOTALL)
        if latex:
            return _clip(latex.group(1).strip(), max_chars)
    pattern = re.compile(rf"(^|\n)(#+\s*)?{re.escape(name)}\b.*?\n", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    next_heading = re.search(r"\n(?:#{1,6}\s+|\\(?:section|subsection|subsubsection)\*?\{)", text[match.end():])
    end = match.end() + next_heading.start() if next_heading else min(len(text), match.end() + max_chars)
    return _clip(text[start:end], max_chars)


def _first_snippet(text: str, keyword: str, context: int) -> str:
    match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(0, match.start() - context)
    end = min(len(text), match.end() + context)
    return text[start:end].strip()


def _first_nonempty_lines(text: str, limit: int) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[:limit]
