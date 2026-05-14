"""Chat Agent Google Search Tool

Performs Google searches via the Serper API, returning raw search result lists (title, link, snippet).
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)


class GoogleSearchToolParams(BaseToolParams):
    """Search Google for real-time web results.

    Returns top 10 organic search results with titles, URLs, snippets and dates.
    Use this when you need raw search result links to visit specific pages,
    or when you want to see multiple sources before diving deeper.

    For a synthesized AI-generated answer, use ai_search instead.
    """

    name: ClassVar[str] = "google_search"

    query: list[str] = Field(
        description="Array of search queries. Include multiple complementary queries for broader coverage."
    )


class GoogleSearchTool(BaseTool):
    """Google Search Tool (Serper API)."""

    name: ClassVar[str] = "google_search"
    params_class: ClassVar[type[BaseToolParams]] = GoogleSearchToolParams

    def __init__(self):
        super().__init__()
        self._seen_queries: set[str] = set()
        self._seen_result_urls: set[str] = set()
        self._stagnation_streak = 0

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute Google search."""
        api_key = os.environ.get("SERPER_KEY_ID")
        if not api_key:
            return (
                "google_search: SERPER_KEY_ID environment variable is not set. "
                "Please set it to use Google search.",
                {"error": "SERPER_KEY_ID not configured"},
            )

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, GoogleSearchToolParams)
        queries = params.query

        self.logger.info("Google search queries: %s", queries)

        results = []
        executed_queries = []
        skipped_duplicate_queries = []
        zero_result_queries = []
        new_result_urls = 0
        batch_seen: set[str] = set()

        for query in queries:
            normalized_query = self._normalize_query(query)
            if not normalized_query:
                skipped_duplicate_queries.append(query)
                continue

            if normalized_query in batch_seen or normalized_query in self._seen_queries:
                skipped_duplicate_queries.append(query)
                continue

            batch_seen.add(normalized_query)
            self._seen_queries.add(normalized_query)

            search_text, meta = self._search_single(query, api_key)
            executed_queries.append(query)

            result_urls = meta.get("result_urls", [])
            unseen_urls = [
                url for url in result_urls
                if url not in self._seen_result_urls
            ]
            if meta.get("result_count", 0) == 0 or unseen_urls:
                results.append(search_text)
            else:
                results.append(
                    f"[google_search] Query '{query}' returned only previously seen result URLs."
                )
            new_result_urls += len(unseen_urls)
            self._seen_result_urls.update(result_urls)

            if meta.get("result_count", 0) == 0:
                zero_result_queries.append(query)

        productive = new_result_urls > 0
        stagnation = bool(queries) and (not executed_queries or not productive)
        if stagnation:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0

        notes = []
        if skipped_duplicate_queries:
            notes.append(
                "[google_search] Skipped duplicate queries already tried earlier in this task: "
                + "; ".join(skipped_duplicate_queries)
            )
        if zero_result_queries:
            notes.append(
                "[google_search] These queries returned 0 results: "
                + "; ".join(zero_result_queries)
            )
        if stagnation:
            notes.append(
                "[google_search] This search call produced no unseen result URLs. "
                "Do not retry the same angle again; change strategy or finish with the best-supported answer."
            )

        response_parts = [*notes, *results]
        if not response_parts:
            response_parts.append(
                "[google_search] No new query was executed. Change strategy or finish with the best-supported answer."
            )

        response = "\n---\n".join(response_parts)
        info = {
            "queries": queries,
            "executed_queries": executed_queries,
            "skipped_duplicate_queries": skipped_duplicate_queries,
            "zero_result_queries": zero_result_queries,
            "new_result_urls": new_result_urls,
            "guard": {
                "productive": productive,
                "stagnation": stagnation,
                "stagnation_streak": self._stagnation_streak,
                "message": notes[-1] if stagnation and notes else "",
            },
        }
        return response, info

    def _search_single(self, query: str, api_key: str) -> tuple[str, dict[str, Any]]:
        """Execute a single-query Google search."""
        conn = http.client.HTTPSConnection("google.serper.dev")

        if self._contains_chinese(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
            })

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        results: dict[str, Any] = {}
        try:
            for attempt in range(3):
                try:
                    conn.request("POST", "/search", payload, headers)
                    res = conn.getresponse()
                    data = res.read()
                    results = json.loads(data.decode("utf-8"))
                    break
                except Exception as e:
                    self.logger.warning("Google search attempt %d failed: %s", attempt + 1, e)
                    if attempt == 2:
                        return (
                            f"Google search failed for '{query}'. Please try again later.",
                            {"result_count": 0, "result_urls": []},
                        )
                    continue
        finally:
            conn.close()

        if "organic" not in results:
            return (
                f"No results found for '{query}'. Try with a more general query.",
                {"result_count": 0, "result_urls": []},
            )

        web_snippets = []
        result_urls = []
        for idx, page in enumerate(results["organic"], 1):
            normalized_url = self._normalize_url(page.get("link", ""))
            if normalized_url:
                result_urls.append(normalized_url)

            date_published = ""
            if "date" in page:
                date_published = f"\nDate published: {page['date']}"

            source = ""
            if "source" in page:
                source = f"\nSource: {page['source']}"

            snippet = ""
            if "snippet" in page:
                snippet = f"\n{page['snippet']}"

            entry = (
                f"{idx}. [{page.get('title', '')}]({page.get('link', '')})"
                f"{date_published}{source}{snippet}"
            )
            web_snippets.append(entry)

        return (
            f"### A Google search for '{query}' found {len(web_snippets)} results:\n\n"
            + "\n\n".join(web_snippets),
            {"result_count": len(web_snippets), "result_urls": result_urls},
        )

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Detect whether the text contains Chinese characters."""
        return any("\u4E00" <= char <= "\u9FFF" for char in text)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not url:
            return ""

        parts = urlsplit(url.strip())
        path = parts.path.rstrip("/") or "/"
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                path,
                parts.query,
                "",
            )
        )
