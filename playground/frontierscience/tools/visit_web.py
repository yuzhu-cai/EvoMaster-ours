"""FrontierScience Web Visit Tool

访问网页并提取相关内容。支持通过 Jina API 或直接访问获取网页内容，
自动提取 HTML 中的文本内容。支持单个 URL 或批量 URL 访问。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

from .common import (
    call_external_function,
    ensure_text,
    extract_text_from_html,
    get_external_script_path,
)

logger = logging.getLogger(__name__)


class VisitWebParams(BaseToolParams):
    """Visit webpages and extract relevant content.

    Fetches webpage content and extracts text, supporting both single URLs and batch processing.
    Automatically tries Jina API first (if JINA_API_KEY is set) for better content extraction,
    then falls back to direct HTTP requests with HTML parsing.

    Use this when you need to read the actual content of specific web pages.
    For finding pages to visit, use search_web or google_scholar first.
    """

    name: ClassVar[str] = "visit_web"
    url: str | list[str] = Field(description="URL string or list of URLs.")
    goal: str = Field(description="What to extract from the webpage(s).")


class VisitWebTool(BaseTool):
    """网页访问工具（支持 Jina API 和直接访问）"""

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
            if self.external_script_path is not None:
                try:
                    result = call_external_function(self.external_script_path, "visit_web", params.url, params.goal)
                    self.logger.info("visit_web completed via external script: %s", self.external_script_path)
                    return ensure_text(result), {
                        "tool": self.name,
                        "mode": "external_script",
                        "script_path": str(self.external_script_path),
                    }
                except Exception as ext_exc:
                    self.logger.warning("visit_web external script failed, fallback to builtin: %s", ext_exc)
                    local = local_visit_web(params.url, params.goal)
                    merged = f"{local}\n\n[external visit_web error] {ext_exc}"
                    return ensure_text(merged), {"tool": self.name, "mode": "builtin_fallback"}
            result = local_visit_web(params.url, params.goal)
            self.logger.info("visit_web completed via builtin")
            return ensure_text(result), {"tool": self.name, "mode": "builtin"}
        except Exception as exc:
            self.logger.error("visit_web failed: %s", exc, exc_info=True)
            return f"[{self.name}] Error: {exc}", {"tool": self.name, "error": str(exc)}


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


def _visit_one(url: str, goal: str) -> str:
    max_chars = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
    text = ""
    source = "direct"
    try:
        text = _fetch_via_jina(url)
        if text:
            source = "jina"
    except Exception:
        text = ""

    if not text:
        raw = _fetch_direct(url)
        text = extract_text_from_html(raw)

    text = text.strip()[:max_chars]
    return f"[visit_web:{source}] URL: {url}\nGoal: {goal}\n\n{text[:6000]}"


def local_visit_web(url: str | list[str], goal: str) -> str:
    if url is None or goal is None:
        return "[visit_web] Invalid request: missing url or goal."
    if isinstance(url, str):
        try:
            return _visit_one(url, goal)
        except Exception as exc:
            return f"[visit_web] Failed to fetch '{url}': {exc}"
    if isinstance(url, list) and all(isinstance(u, str) for u in url):
        parts = []
        for one in url:
            try:
                parts.append(_visit_one(one, goal))
            except Exception as exc:
                parts.append(f"[visit_web] Failed to fetch '{one}': {exc}")
        return "\n\n---\n\n".join(parts)
    return "[visit_web] Invalid url: expected string or list of strings."
