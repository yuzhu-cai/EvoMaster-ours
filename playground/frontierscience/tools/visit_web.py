"""FrontierScience Web Visit Tool.

Visits webpages and extracts relevant content.
Supports Jina Reader API and direct HTTP fallback.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote, urlparse

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

from .common import (
    call_external_function,
    ensure_text,
    extract_text_from_html,
    get_external_script_path,
)

logger = logging.getLogger(__name__)
TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_ALLOWED_DOMAINS = ("arxiv.org",)


class VisitWebParams(BaseToolParams):
    """Visit webpages and extract relevant content.

    Fetches webpage content and extracts text, supporting both single URLs and
    batch processing.
    """

    name: ClassVar[str] = "visit_web"
    url: str | list[str] = Field(description="URL string or list of URLs.")
    goal: str = Field(description="What to extract from the webpage(s).")


class VisitWebTool(BaseTool):
    """Web visit tool with allowlist and paper PDF hints."""

    name: ClassVar[str] = "visit_web"
    params_class: ClassVar[type[BaseToolParams]] = VisitWebParams

    def __init__(self, script_path: Path | None = None):
        super().__init__()
        self.external_script_path = get_external_script_path("FRONTIER_VISIT_WEB_SCRIPT", script_path)

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, VisitWebParams)
            url_count = len(params.url) if isinstance(params.url, list) else 1
            self.logger.info("visit_web called (url_count=%d)", url_count)

            enforce, allowed_domains = _get_allowlist_policy()
            workspace_path = session.get_workspace_path() if hasattr(session, "get_workspace_path") else None
            if not workspace_path:
                workspace_path = getattr(getattr(session, "config", None), "workspace_path", None)
            result = local_visit_web(
                params.url,
                params.goal,
                external_script_path=self.external_script_path,
                workspace_path=workspace_path,
            )
            mode = "external_script" if self.external_script_path is not None else "builtin"
            self.logger.info("visit_web completed via %s", mode)
            return ensure_text(result), {
                "tool": self.name,
                "mode": mode,
                "allowlist_enforced": enforce,
                "allowed_domains": allowed_domains,
                "script_path": str(self.external_script_path) if self.external_script_path is not None else "",
            }
        except Exception as exc:
            self.logger.error("visit_web failed: %s", exc, exc_info=True)
            return f"[{self.name}] Error: {exc}", {"tool": self.name, "error": str(exc)}


def _get_allowlist_policy() -> tuple[bool, list[str]]:
    enforce = os.getenv("FRONTIER_VISIT_ENFORCE_ALLOWLIST", "1").strip().lower() in TRUE_VALUES
    raw_domains = os.getenv("FRONTIER_VISIT_ALLOWED_DOMAINS", ",".join(DEFAULT_ALLOWED_DOMAINS))
    domains = [one.strip().lower() for one in raw_domains.split(",") if one.strip()]
    if not domains:
        domains = list(DEFAULT_ALLOWED_DOMAINS)
    return enforce, domains


def _normalize_url(url: str) -> str:
    normalized = (url or "").strip()
    if normalized and "://" not in normalized:
        normalized = "https://" + normalized
    return normalized


def _is_allowed_url(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _extract_arxiv_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host.endswith("arxiv.org"):
        return None

    path = parsed.path or ""
    for prefix in ("/abs/", "/pdf/", "/html/"):
        if path.startswith(prefix):
            candidate = path[len(prefix) :].strip("/")
            if candidate.endswith(".pdf"):
                candidate = candidate[:-4]
            return candidate or None
    return None


def _build_paper_pdf_url(url: str) -> str | None:
    arxiv_id = _extract_arxiv_id(url)
    if not arxiv_id:
        return None
    encoded_id = quote(arxiv_id, safe="/._-")
    return f"https://arxiv.org/pdf/{encoded_id}.pdf"


def _build_download_hint(pdf_url: str | None) -> str:
    if not pdf_url:
        return "N/A"
    return "Handled automatically by visit_web for arXiv papers."


def _slugify_title(title: str | None, fallback: str) -> str:
    raw = (title or "").strip().lower()
    raw = re.sub(r"^\[[^\]]+\]\s*", "", raw)
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    raw = raw[:120].rstrip("_")
    return raw or fallback


def _trim_title_candidate(candidate: str) -> str:
    text = " ".join((candidate or "").split()).strip(" -:\t")
    if not text:
        return ""
    separators = [
        " Authors:",
        " Author:",
        " Abstract",
        " Comments:",
        " Subjects:",
        " Journal",
        " DOI",
        " MSC",
        " ACM",
        " Report number",
        " View PDF",
        " View a PDF",
        " Full-text links",
        " Submission history",
        " Cite as",
        " arXiv:",
    ]
    for sep in separators:
        if sep in text:
            text = text.split(sep, 1)[0].strip(" -:\t")
    return text[:200].strip(" -:\t")


def _extract_title_from_content(content: str, url: str) -> str | None:
    arxiv_id = _extract_arxiv_id(url)
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if arxiv_id and line.startswith(f"[{arxiv_id}]"):
            candidate = _trim_title_candidate(line.split("]", 1)[-1].strip())
            if candidate:
                return candidate
        if line.lower().startswith("title:"):
            candidate = _trim_title_candidate(line.split(":", 1)[-1].strip())
            if candidate:
                return candidate
    return None


def _download_pdf_bytes(pdf_url: str) -> bytes:
    import requests

    timeout = int(os.getenv("FRONTIER_PDF_DOWNLOAD_TIMEOUT", "180"))
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    response = requests.get(pdf_url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.content


def _validate_pdf_file(pdf_path: Path) -> tuple[bool, str]:
    if not pdf_path.exists():
        return False, "file_missing"
    data = pdf_path.read_bytes()
    if not data.startswith(b"%PDF-"):
        return False, "missing_pdf_header"
    if b"%%EOF" not in data[-4096:]:
        return False, "missing_eof_marker"
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        if len(reader.pages) <= 0:
            return False, "zero_pages"
    except Exception as exc:
        return False, f"pypdf_validation_failed: {exc}"
    return True, "ok"


def _auto_download_pdf(
    pdf_url: str | None,
    *,
    content: str,
    workspace_path: str | None,
    source_url: str,
) -> tuple[str | None, str]:
    if not pdf_url:
        return None, "not_applicable"
    if not workspace_path:
        return None, "workspace_unavailable"

    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    fallback_name = _extract_arxiv_id(source_url) or "paper"
    title = _extract_title_from_content(content, source_url)
    filename = _slugify_title(title, fallback=fallback_name) + ".pdf"
    target_path = workspace / filename

    if target_path.exists():
        is_valid, status = _validate_pdf_file(target_path)
        if is_valid:
            return str(target_path.resolve()), "cached_valid_pdf"
        target_path.unlink(missing_ok=True)

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    try:
        tmp_path.write_bytes(_download_pdf_bytes(pdf_url))
        is_valid, status = _validate_pdf_file(tmp_path)
        if not is_valid:
            tmp_path.unlink(missing_ok=True)
            return None, f"downloaded_file_invalid: {status}"
        tmp_path.replace(target_path)
        return str(target_path.resolve()), "downloaded_valid_pdf"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return None, f"download_failed: {exc}"


def _format_policy(enforce: bool, allowed_domains: list[str], blocked: bool) -> str:
    status = "BLOCKED" if blocked else "ALLOW"
    return f"{status} (allowed={','.join(allowed_domains)}, enforce={1 if enforce else 0})"


def _format_visit_result(
    source: str,
    url: str,
    goal: str,
    enforce: bool,
    allowed_domains: list[str],
    paper_pdf_url: str | None,
    local_pdf_path: str | None,
    pdf_status: str,
    paper_title: str | None,
    content: str,
    blocked: bool = False,
) -> str:
    header = "[visit_web] BLOCKED by domain allowlist." if blocked else f"[visit_web:{source}] URL: {url}"
    return "\n".join(
        [
            header,
            f"URL: {url}",
            f"DomainPolicy: {_format_policy(enforce, allowed_domains, blocked=blocked)}",
            f"Paper Title: {paper_title or 'N/A'}",
            f"Paper PDF: {paper_pdf_url or 'N/A'}",
            f"Downloaded PDF Path: {local_pdf_path or 'N/A'}",
            f"PDF Download Status: {pdf_status}",
            f"Download Hint: {_build_download_hint(paper_pdf_url)}",
            f"Goal: {goal}",
            "",
            content.strip(),
        ]
    ).strip()


def _fetch_via_jina(url: str) -> str:
    import requests

    key = os.getenv("JINA_API_KEY", "").strip()
    if not key:
        return ""
    headers = {"Authorization": f"Bearer {key}"}
    resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=50)
    resp.raise_for_status()
    return resp.text


def _fetch_direct(url: str) -> str:
    import requests

    timeout = int(os.getenv("VISIT_SERVER_TIMEOUT", "60"))
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def _fetch_builtin_content(url: str) -> tuple[str, str]:
    max_chars = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
    try:
        text = _fetch_via_jina(url)
        if text:
            return "jina", text.strip()[:max_chars]
    except Exception:
        pass

    raw = _fetch_direct(url)
    text = extract_text_from_html(raw)
    return "direct", text.strip()[:max_chars]


def _visit_one(url: str, goal: str, external_script_path: Path | None = None, workspace_path: str | None = None) -> str:
    normalized_url = _normalize_url(url)
    enforce, allowed_domains = _get_allowlist_policy()
    paper_pdf_url = _build_paper_pdf_url(normalized_url)

    if enforce and not _is_allowed_url(normalized_url, allowed_domains):
        return _format_visit_result(
            source="blocked",
            url=normalized_url,
            goal=goal,
            enforce=enforce,
            allowed_domains=allowed_domains,
            paper_pdf_url=paper_pdf_url,
            local_pdf_path=None,
            pdf_status="blocked",
            paper_title=None,
            content="[visit_web] Request blocked before network call.",
            blocked=True,
        )

    content = ""
    source = "error"
    if external_script_path is not None:
        try:
            content = str(call_external_function(external_script_path, "visit_web", normalized_url, goal))
            source = "external_script"
        except Exception as ext_exc:
            logger.warning("visit_web external script failed, fallback to builtin: %s", ext_exc)
            try:
                source, text = _fetch_builtin_content(normalized_url)
                content = f"{text[:6000]}\n\n[external visit_web error] {ext_exc}"
            except Exception as fetch_exc:
                content = f"[visit_web] Failed to fetch '{normalized_url}': {fetch_exc}"
                source = "error"
    else:
        try:
            source, text = _fetch_builtin_content(normalized_url)
            content = text[:6000]
        except Exception as exc:
            content = f"[visit_web] Failed to fetch '{normalized_url}': {exc}"
            source = "error"

    paper_title = _extract_title_from_content(content, normalized_url)
    local_pdf_path, pdf_status = _auto_download_pdf(
        paper_pdf_url,
        content=content,
        workspace_path=workspace_path,
        source_url=normalized_url,
    )
    return _format_visit_result(
        source=source,
        url=normalized_url,
        goal=goal,
        enforce=enforce,
        allowed_domains=allowed_domains,
        paper_pdf_url=paper_pdf_url,
        local_pdf_path=local_pdf_path,
        pdf_status=pdf_status,
        paper_title=paper_title,
        content=content,
        blocked=False,
    )


def local_visit_web(
    url: str | list[str],
    goal: str,
    external_script_path: Path | None = None,
    workspace_path: str | None = None,
) -> str:
    if url is None or goal is None:
        return "[visit_web] Invalid request: missing url or goal."
    if isinstance(url, str):
        return _visit_one(url, goal, external_script_path=external_script_path, workspace_path=workspace_path)
    if isinstance(url, list) and all(isinstance(u, str) for u in url):
        parts = [
            _visit_one(one, goal, external_script_path=external_script_path, workspace_path=workspace_path)
            for one in url
        ]
        return "\n\n---\n\n".join(parts)
    return "[visit_web] Invalid url: expected string or list of strings."
