"""FrontierScience Web Visit Tool.

Visits webpages and extracts relevant content.
Supports Jina Reader API and direct HTTP fallback.
"""

from __future__ import annotations

import logging
import os
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
            result = local_visit_web(params.url, params.goal, external_script_path=self.external_script_path)
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
    return f'wget -O paper.pdf "{pdf_url}"'


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
    content: str,
    blocked: bool = False,
) -> str:
    header = "[visit_web] BLOCKED by domain allowlist." if blocked else f"[visit_web:{source}] URL: {url}"
    return "\n".join(
        [
            header,
            f"URL: {url}",
            f"DomainPolicy: {_format_policy(enforce, allowed_domains, blocked=blocked)}",
            f"Paper PDF: {paper_pdf_url or 'N/A'}",
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


def _visit_one(url: str, goal: str, external_script_path: Path | None = None) -> str:
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
            content="[visit_web] Request blocked before network call.",
            blocked=True,
        )

    if external_script_path is not None:
        try:
            content = str(call_external_function(external_script_path, "visit_web", normalized_url, goal))
            return _format_visit_result(
                source="external_script",
                url=normalized_url,
                goal=goal,
                enforce=enforce,
                allowed_domains=allowed_domains,
                paper_pdf_url=paper_pdf_url,
                content=content,
                blocked=False,
            )
        except Exception as ext_exc:
            logger.warning("visit_web external script failed, fallback to builtin: %s", ext_exc)
            try:
                source, text = _fetch_builtin_content(normalized_url)
                merged = f"{text[:6000]}\n\n[external visit_web error] {ext_exc}"
                return _format_visit_result(
                    source=source,
                    url=normalized_url,
                    goal=goal,
                    enforce=enforce,
                    allowed_domains=allowed_domains,
                    paper_pdf_url=paper_pdf_url,
                    content=merged,
                    blocked=False,
                )
            except Exception as fetch_exc:
                return _format_visit_result(
                    source="error",
                    url=normalized_url,
                    goal=goal,
                    enforce=enforce,
                    allowed_domains=allowed_domains,
                    paper_pdf_url=paper_pdf_url,
                    content=f"[visit_web] Failed to fetch '{normalized_url}': {fetch_exc}",
                    blocked=False,
                )

    try:
        source, text = _fetch_builtin_content(normalized_url)
        return _format_visit_result(
            source=source,
            url=normalized_url,
            goal=goal,
            enforce=enforce,
            allowed_domains=allowed_domains,
            paper_pdf_url=paper_pdf_url,
            content=text[:6000],
            blocked=False,
        )
    except Exception as exc:
        return _format_visit_result(
            source="error",
            url=normalized_url,
            goal=goal,
            enforce=enforce,
            allowed_domains=allowed_domains,
            paper_pdf_url=paper_pdf_url,
            content=f"[visit_web] Failed to fetch '{normalized_url}': {exc}",
            blocked=False,
        )


def local_visit_web(url: str | list[str], goal: str, external_script_path: Path | None = None) -> str:
    if url is None or goal is None:
        return "[visit_web] Invalid request: missing url or goal."
    if isinstance(url, str):
        return _visit_one(url, goal, external_script_path=external_script_path)
    if isinstance(url, list) and all(isinstance(u, str) for u in url):
        parts = [_visit_one(one, goal, external_script_path=external_script_path) for one in url]
        return "\n\n---\n\n".join(parts)
    return "[visit_web] Invalid url: expected string or list of strings."
