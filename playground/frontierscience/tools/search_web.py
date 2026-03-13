"""FrontierScience Web Search Tool

通过 Serper API 进行 Google 搜索，返回原始搜索结果列表（标题、链接、摘要）。
支持单个查询或批量查询，自动检测中文并调整搜索区域。
"""

from __future__ import annotations

import http.client
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

from .common import call_external_function, ensure_text, get_external_script_path, get_serper_api_key

logger = logging.getLogger(__name__)


class SearchWebParams(BaseToolParams):
    """Search the web for real-time results.

    Returns top organic search results with titles, URLs, snippets and dates.
    Supports both single query strings and batch queries (list of strings).
    Automatically detects Chinese queries and adjusts search region accordingly.

    Use this when you need raw search result links to visit specific pages,
    or when you want to see multiple sources before diving deeper.
    """

    name: ClassVar[str] = "search_web"
    query: str | list[str] = Field(description="Single query string or list of queries.")


class SearchWebTool(BaseTool):
    """网页搜索工具（Serper API）"""

    name: ClassVar[str] = "search_web"
    params_class: ClassVar[type[BaseToolParams]] = SearchWebParams

    def __init__(self, script_path: Path | None = None):
        super().__init__()
        self.external_script_path = get_external_script_path("FRONTIER_SEARCH_WEB_SCRIPT", script_path)

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, SearchWebParams)
            query_type = "list" if isinstance(params.query, list) else "string"
            self.logger.info("search_web called (query_type=%s)", query_type)
            if self.external_script_path is not None:
                result = call_external_function(self.external_script_path, "search_web", params.query)
                self.logger.info("search_web completed via external script: %s", self.external_script_path)
                return ensure_text(result), {
                    "tool": self.name,
                    "mode": "external_script",
                    "script_path": str(self.external_script_path),
                }
            result = local_search_web(params.query)
            self.logger.info("search_web completed via builtin")
            return ensure_text(result), {"tool": self.name, "mode": "builtin"}
        except Exception as exc:
            self.logger.error("search_web failed: %s", exc, exc_info=True)
            return f"[{self.name}] Error: {exc}", {"tool": self.name, "error": str(exc)}


def _contains_chinese(text: str) -> bool:
    return any("\u4E00" <= ch <= "\u9FFF" for ch in text)


def _search_one(query: str) -> str:
    key = get_serper_api_key()
    if not key:
        return "[search_web] Serper API key is not set. Use SERPER_API_KEY."

    payload = (
        {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        if _contains_chinese(query)
        else {"q": query, "location": "United States", "gl": "us", "hl": "en"}
    )
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}

    conn = http.client.HTTPSConnection("google.serper.dev")
    for i in range(5):
        try:
            conn.request("POST", "/search", json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read()
            if res.status != 200:
                return f"[search_web] API error {res.status} for query '{query}'."
            results = json.loads(data.decode("utf-8"))
            organic = results.get("organic", [])
            if not organic:
                return f"[search_web] No results for '{query}'."
            snippets: list[str] = []
            for idx, page in enumerate(organic, start=1):
                date = f"\nDate: {page['date']}" if page.get("date") else ""
                source = f"\nSource: {page['source']}" if page.get("source") else ""
                snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
                line = f"{idx}. [{page.get('title','')}]({page.get('link','')}){date}{source}\n{snippet}"
                snippets.append(line.replace("Your browser can't play this video.", ""))
            return f"### Search results for '{query}' ({len(snippets)} found):\n\n" + "\n\n".join(snippets)
        except Exception as exc:
            if i == 4:
                return f"[search_web] Error for '{query}': {exc}"
    return f"[search_web] Failed for '{query}'."


def local_search_web(query: str | list[str]) -> str:
    if query is None:
        return "[search_web] Invalid request: missing query."
    if isinstance(query, str):
        return _search_one(query)
    if isinstance(query, list) and all(isinstance(q, str) for q in query):
        return "\n\n---\n\n".join(_search_one(q) for q in query)
    return "[search_web] Invalid query: expected string or list of strings."
