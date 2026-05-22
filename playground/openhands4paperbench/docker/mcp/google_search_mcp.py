#!/usr/bin/env python3
"""MCP tools for Google search via Serper and webpage fetch via Jina Reader."""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("google_search_mcp")

SERPER_URL = "https://google.serper.dev/search"
JINA_READER_PREFIX = "https://r.jina.ai/"
DEFAULT_TIMEOUT = 45.0


def _contains_chinese(text: str) -> bool:
    return any("\u4E00" <= char <= "\u9FFF" for char in text)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _format_error(prefix: str, exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return f"{prefix}: authentication failed. Check the API key configured for this tool."
        if status == 429:
            return f"{prefix}: rate limit exceeded. Retry later or reduce request volume."
        body = exc.response.text[:500].strip()
        suffix = f": {body}" if body else ""
        return f"{prefix}: HTTP {status}{suffix}"
    if isinstance(exc, httpx.TimeoutException):
        return f"{prefix}: request timed out."
    return f"{prefix}: {type(exc).__name__}: {exc}"


async def _serper_search_one(client: httpx.AsyncClient, query: str, num_results: int) -> str:
    api_key = os.environ.get("SERPER_KEY_ID")
    if not api_key:
        return "google_search: SERPER_KEY_ID environment variable is not set."

    payload: dict[str, Any] = {
        "q": query,
        "num": num_results,
        "location": "China" if _contains_chinese(query) else "United States",
        "gl": "cn" if _contains_chinese(query) else "us",
        "hl": "zh-cn" if _contains_chinese(query) else "en",
    }
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = await client.post(SERPER_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return _format_error(f"Google search failed for {query!r}", exc)

    organic = data.get("organic") or []
    if not organic:
        return f"No Google results found for {query!r}."

    lines = [f"### Google search results for {query!r}", ""]
    for idx, page in enumerate(organic[:num_results], 1):
        title = page.get("title", "").strip()
        link = page.get("link", "").strip()
        snippet = page.get("snippet", "").strip()
        date = page.get("date", "").strip()
        source = page.get("source", "").strip()

        if link:
            lines.append(f"{idx}. [{title or link}]({link})")
        else:
            lines.append(f"{idx}. {title or '(untitled result)'}")
        if source:
            lines.append(f"   Source: {source}")
        if date:
            lines.append(f"   Date published: {date}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip()


async def _google_search_impl(query: str, num_results: int = 10) -> str:
    clean_query = query.strip()
    if not clean_query:
        return "google_search: query must be a non-empty search string."

    limit = _clamp(int(num_results), 1, 10)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
        return await _serper_search_one(client, clean_query, limit)


@mcp.tool(
    name="google_search",
    structured_output=False,
    annotations={
        "title": "Google Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def google_search(query: str, num_results: int = 10) -> str:
    """Search Google through the Serper API and return titles, URLs, snippets, sources, and dates.

    Args:
        query: Search query. Prefer official docs, package APIs, datasets, or model pages.
        num_results: Number of organic results per query, clamped to 1-10.

    Returns:
        Markdown search results. Use web_fetch on promising URLs when page content is needed.
    """

    return await _google_search_impl(query, num_results)


@mcp.tool(
    name="web_search",
    structured_output=False,
    annotations={
        "title": "Web Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def web_search(query: str, num_results: int = 10) -> str:
    """Alias for google_search, exposed for agents that look for a web_search tool."""

    return await _google_search_impl(query, num_results)


def _html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?is)<br\\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\\s*>", "\n\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\n\\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def _direct_fetch(client: httpx.AsyncClient, url: str) -> str:
    headers = {
        "Accept": "text/plain, text/markdown, text/html;q=0.9, */*;q=0.8",
        "User-Agent": "openhands4paperbench-mcp/1.0",
    }
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    text = response.text
    if "html" in content_type.lower():
        return _html_to_text(text)
    return text


@mcp.tool(
    name="web_fetch",
    structured_output=False,
    annotations={
        "title": "Fetch Webpage",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def web_fetch(url: str, max_chars: int = 12000) -> str:
    """Fetch a public webpage as LLM-friendly Markdown using Jina Reader.

    Args:
        url: Absolute http(s) URL to fetch.
        max_chars: Maximum characters to return, clamped to 1,000-60,000.

    Returns:
        Markdown/text content from the requested page, truncated when needed.
    """

    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        return "web_fetch: url must start with http:// or https://."

    limit = _clamp(int(max_chars), 1000, 60000)
    headers = {
        "Accept": "text/plain",
        "X-Return-Format": "markdown",
    }
    jina_api_key = os.environ.get("JINA_API_KEY")
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"

    reader_url = f"{JINA_READER_PREFIX}{clean_url}"
    errors: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
            text = response.text
    except Exception as exc:
        errors.append(_format_error("Jina Reader failed", exc))
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                text = await _direct_fetch(client, clean_url)
        except Exception as direct_exc:
            errors.append(_format_error("direct fetch failed", direct_exc))
            return f"web_fetch failed for {clean_url!r}: " + " | ".join(errors)

    if len(text) > limit:
        text = text[:limit].rstrip() + "\n\n[Truncated by web_fetch]"
    if errors:
        text = f"[Jina Reader fallback: {errors[0]}]\n\n{text}"
    return text


@mcp.tool(
    name="web_search_health",
    structured_output=False,
    annotations={
        "title": "Web Search Health",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def web_search_health() -> str:
    """Report whether required web-search credentials are visible to this MCP process."""

    status = {
        "serper_key_configured": bool(os.environ.get("SERPER_KEY_ID")),
        "jina_key_configured": bool(os.environ.get("JINA_API_KEY")),
        "tools": ["google_search", "web_search", "web_fetch"],
    }
    return json.dumps(status, indent=2)


if __name__ == "__main__":
    mcp.run()
