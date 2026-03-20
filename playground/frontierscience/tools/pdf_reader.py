"""FrontierScience PDF reading tool."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

logger = logging.getLogger(__name__)


class PdfReadParams(BaseToolParams):
    """Read a scientific PDF with a fixed, efficient workflow."""

    name: ClassVar[str] = "read_paper_pdf"
    pdf_path: str = Field(description="Path to a local PDF file, usually `paper.pdf`.")
    goal: str = Field(description="The information goal to guide focused reading.")
    mode: str = Field(default="auto", description="One of `auto`, `overview`, `focus`, `pages`, `find`.")
    page_start: int | None = Field(default=None, description="1-based start page for `pages` mode.")
    page_end: int | None = Field(default=None, description="1-based end page for `pages` mode.")
    query: str | None = Field(default=None, description="Keyword for `find` mode.")


class PdfReaderTool(BaseTool):
    """Structured paper reading tool for local PDFs."""

    name: ClassVar[str] = "read_paper_pdf"
    params_class: ClassVar[type[BaseToolParams]] = PdfReadParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, PdfReadParams)
            text, meta = local_read_paper_pdf(
                pdf_path=params.pdf_path,
                goal=params.goal,
                mode=params.mode,
                page_start=params.page_start,
                page_end=params.page_end,
                query=params.query,
            )
            return text, {"tool": self.name, **meta}
        except Exception as exc:
            logger.error("read_paper_pdf failed: %s", exc, exc_info=True)
            return f"[read_paper_pdf] Error: {exc}", {"tool": self.name, "error": str(exc)}


def _resolve_pdf_path(pdf_path: str) -> Path:
    path = Path(pdf_path).expanduser()
    if path.exists():
        return path.resolve()
    cwd_path = Path.cwd() / pdf_path
    if cwd_path.exists():
        return cwd_path.resolve()
    raise FileNotFoundError(f"PDF not found: {pdf_path}")


def _run_pdftotext(pdf_path: Path, txt_path: Path) -> tuple[bool, str]:
    binary = shutil.which("pdftotext")
    if not binary:
        return False, "pdftotext not available"
    completed = subprocess.run([binary, "-layout", str(pdf_path), str(txt_path)], capture_output=True, text=True)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "pdftotext failed").strip()
    return True, "ok"


def _extract_text_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Neither pdftotext nor pypdf is available for PDF reading") from exc

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        parts.append(f"\n\n=== Page {index} ===\n{content}")
    return "".join(parts).strip()


def _load_or_build_text(pdf_path: Path) -> tuple[str, Path | None, str]:
    txt_path = pdf_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore"), txt_path, "cached_txt"

    success, status = _run_pdftotext(pdf_path, txt_path)
    if success and txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore"), txt_path, "pdftotext"

    text = _extract_text_with_pypdf(pdf_path)
    txt_path.write_text(text, encoding="utf-8")
    return text, txt_path, f"pypdf_fallback({status})"


def _split_pages(text: str) -> list[str]:
    if "=== Page " not in text:
        return [text]
    parts = text.split("=== Page ")
    pages: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        marker_end = part.find("===")
        if marker_end != -1:
            part = part[marker_end + 3 :].strip()
        pages.append(part)
    return pages or [text]


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]..."


def _pick_focus_pages(pages: list[str], goal: str) -> list[tuple[int, str]]:
    terms = [token.lower() for token in goal.replace(",", " ").split() if len(token) >= 4]
    ranked: list[tuple[int, int]] = []
    for idx, page in enumerate(pages, start=1):
        score = sum(page.lower().count(term) for term in terms)
        if idx <= 2:
            score += 2
        ranked.append((score, idx))
    chosen = [idx for score, idx in sorted(ranked, reverse=True) if score > 0][:3]
    if not chosen:
        chosen = list(range(1, min(len(pages), 3) + 1))
    return [(idx, pages[idx - 1]) for idx in chosen]


def _overview_response(pdf_path: Path, pages: list[str], goal: str) -> str:
    selected = _pick_focus_pages(pages, goal)
    snippets = [f"## Focus page {idx}\n{_truncate(content, 1800)}" for idx, content in selected]
    return "\n".join(
        [
            f"[read_paper_pdf] File: {pdf_path}",
            "Route: extract full text -> inspect first page -> rank focus pages -> return targeted snippets",
            f"Goal: {goal}",
            f"PageCount: {len(pages)}",
            "## First page preview",
            _truncate(pages[0] if pages else "", 2200),
            *snippets,
            "## Suggested next steps",
            "Use mode='find' to locate equations/keywords, or mode='pages' to inspect a page range.",
        ]
    )


def _pages_response(pdf_path: Path, pages: list[str], goal: str, start: int | None, end: int | None) -> str:
    start = max(1, start or 1)
    end = min(len(pages), end or start)
    if start > end:
        raise ValueError("page_start must be <= page_end")
    return "\n".join(
        [
            f"[read_paper_pdf] File: {pdf_path}",
            "Route: targeted page range reading",
            f"Goal: {goal}",
            f"Pages: {start}-{end}",
            *[f"## Page {idx}\n{_truncate(pages[idx - 1], 2200)}" for idx in range(start, end + 1)],
        ]
    )


def _find_response(pdf_path: Path, pages: list[str], goal: str, query: str | None) -> str:
    if not query:
        raise ValueError("query is required when mode='find'")
    matches: list[str] = []
    needle = query.lower()
    for idx, page in enumerate(pages, start=1):
        pos = page.lower().find(needle)
        if pos == -1:
            continue
        start = max(0, pos - 400)
        end = min(len(page), pos + len(query) + 900)
        matches.append(f"## Match page {idx}\n{_truncate(page[start:end], 1500)}")
        if len(matches) >= 5:
            break
    return "\n".join(
        [
            f"[read_paper_pdf] File: {pdf_path}",
            "Route: keyword search over extracted PDF text",
            f"Goal: {goal}",
            f"Query: {query}",
            *(matches or ["No matches found."]),
        ]
    )


def local_read_paper_pdf(
    pdf_path: str,
    goal: str,
    mode: str = "auto",
    page_start: int | None = None,
    page_end: int | None = None,
    query: str | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_pdf = _resolve_pdf_path(pdf_path)
    text, txt_path, extractor = _load_or_build_text(resolved_pdf)
    pages = _split_pages(text)
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode in {"auto", "overview", "focus"}:
        result = _overview_response(resolved_pdf, pages, goal)
        normalized_mode = "overview"
    elif normalized_mode == "pages":
        result = _pages_response(resolved_pdf, pages, goal, page_start, page_end)
    elif normalized_mode == "find":
        result = _find_response(resolved_pdf, pages, goal, query)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    meta = {
        "pdf_path": str(resolved_pdf),
        "text_cache_path": str(txt_path) if txt_path else None,
        "extractor": extractor,
        "mode": normalized_mode,
        "page_count": len(pages),
    }
    logger.info("read_paper_pdf completed: %s", json.dumps(meta, ensure_ascii=False))
    return result, meta
