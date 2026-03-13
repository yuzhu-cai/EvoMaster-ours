"""FrontierScience Google Scholar Tool

通过 Serper API 进行 Google Scholar 学术搜索，返回学术论文结果（标题、链接、摘要、日期、来源）。
支持单个查询或批量查询，自动检测中文并调整搜索区域。
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

from .common import call_external_function, ensure_text, get_external_script_path

logger = logging.getLogger(__name__)


class GoogleScholarParams(BaseToolParams):
    """Search academic papers using Google Scholar.

    Returns scholarly articles with titles, URLs, snippets, publication dates and sources.
    Supports both single query strings and batch queries (list of strings).
    Automatically detects Chinese queries and adjusts search region accordingly.

    Use this when you need to find academic papers, research articles, or scholarly publications.
    This is specialized for academic content and differs from general web search.
    """

    name: ClassVar[str] = "google_scholar"
    query: str | list[str] = Field(description="Single query string or list of queries.")


class GoogleScholarTool(BaseTool):
    """学术搜索工具（Google Scholar via Serper API）"""

    name: ClassVar[str] = "google_scholar"
    params_class: ClassVar[type[BaseToolParams]] = GoogleScholarParams

    def __init__(self, script_path: Path | None = None):
        super().__init__()
        self.external_script_path = get_external_script_path("FRONTIER_GOOGLE_SCHOLAR_SCRIPT", script_path)

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, GoogleScholarParams)
            query_type = "list" if isinstance(params.query, list) else "string"
            self.logger.info("google_scholar called (query_type=%s)", query_type)
            if self.external_script_path is not None:
                result = call_external_function(self.external_script_path, "google_scholar", params.query)
                self.logger.info("google_scholar completed via external script: %s", self.external_script_path)
                return ensure_text(result), {
                    "tool": self.name,
                    "mode": "external_script",
                    "script_path": str(self.external_script_path),
                }
            result = local_google_scholar(params.query)
            self.logger.info("google_scholar completed via builtin")
            return ensure_text(result), {"tool": self.name, "mode": "builtin"}
        except Exception as exc:
            self.logger.error("google_scholar failed: %s", exc, exc_info=True)
            return f"[{self.name}] Error: {exc}", {"tool": self.name, "error": str(exc)}


def _contains_chinese(text: str) -> bool:
    return any("\u4E00" <= ch <= "\u9FFF" for ch in text)


def _scholar_one(query: str) -> str:
    key = os.getenv("SERPER_KEY_ID", "").strip()
    if not key:
        return "[google_scholar] SERPER_KEY_ID is not set."

    payload = (
        {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        if _contains_chinese(query)
        else {"q": query, "location": "United States", "gl": "us", "hl": "en"}
    )
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}

    conn = http.client.HTTPSConnection("google.serper.dev")
    for i in range(5):
        try:
            conn.request("POST", "/scholar", json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read()
            if res.status != 200:
                return f"[google_scholar] API error {res.status} for query '{query}'."
            results = json.loads(data.decode("utf-8"))
            organic = results.get("organic", [])
            if not organic:
                return f"[google_scholar] No results for '{query}'."
            snippets: list[str] = []
            for idx, page in enumerate(organic, start=1):
                date = f"\nDate: {page['date']}" if page.get("date") else ""
                source = f"\nSource: {page['source']}" if page.get("source") else ""
                snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
                line = f"{idx}. [{page.get('title','')}]({page.get('link','')}){date}{source}\n{snippet}"
                snippets.append(line.replace("Your browser can't play this video.", ""))
            return f"### Google Scholar results for '{query}' ({len(snippets)} found):\n\n" + "\n\n".join(snippets)
        except Exception as exc:
            if i == 4:
                return f"[google_scholar] Error for '{query}': {exc}"
    return f"[google_scholar] Failed for '{query}'."


def local_google_scholar(query: str | list[str]) -> str:
    if query is None:
        return "[google_scholar] Invalid request: missing query."
    if isinstance(query, str):
        return _scholar_one(query)
    if isinstance(query, list) and all(isinstance(q, str) for q in query):
        return "\n\n---\n\n".join(_scholar_one(q) for q in query)
    return "[google_scholar] Invalid query: expected string or list of strings."
