"""WebMaster web-fetch tool with goal-focused extraction."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import requests
from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.utils.llm import BaseLLM

logger = logging.getLogger(__name__)

# Maximum fetched content length (characters); truncated before sending to LLM
MAX_CONTENT_LENGTH = 80000

EXTRACTOR_PROMPT = """You are extracting benchmark evidence from a fetched webpage.

Webpage content:
{webpage_content}

User goal:
{goal}

Instructions:
1. Ignore navigation chrome, cookie banners, login walls, robot checks, repeated menus, and unrelated sidebar content.
2. Extract only evidence that directly helps solve the goal.
3. Prefer exact names, dates, numbers, places, role relationships, episode/season identifiers, and explicit yes/no contradiction signals.
4. If the page is mostly a challenge screen, navigation shell, or obviously unrelated, say so explicitly.
5. Keep the output compact and decision-oriented.

Return valid JSON with this schema:
{{
  "page_quality": "high|medium|low",
  "page_type": "primary|entity_adjacent|aggregator|challenge|unknown",
  "evidence": [
    "Short bullet with the exact fact and enough context to identify it",
    "Another short bullet"
  ],
  "summary": "2-4 sentences explaining what this page does or does not verify for the goal."
}}
"""


class WebFetchToolParams(BaseToolParams):
    """Fetch and read webpage content from a URL, then extract key information based on your goal.

    Returns extracted evidence and summary from the page content.
    Use this to read articles, documentation, or any web page content.
    Pair with google_search to first find relevant URLs, then fetch their content.
    """

    name: ClassVar[str] = "web_fetch"

    url: list[str] = Field(
        description="URL(s) of the webpage(s) to fetch. Can be one or multiple URLs."
    )
    goal: str = Field(
        description="The goal or purpose of fetching the page(s). Used to extract the most relevant information."
    )


class WebFetchTool(BaseTool):
    """Web content fetch + LLM summary extraction tool (Jina Reader API + LLM)."""

    name: ClassVar[str] = "web_fetch"
    params_class: ClassVar[type[BaseToolParams]] = WebFetchToolParams

    def __init__(self):
        super().__init__()
        self._llm = None
        self._content_cache: dict[str, str] = {}
        self._extract_cache: dict[tuple[str, str], str] = {}
        self._failed_urls: dict[str, int] = {}
        self._stagnation_streak = 0

    def set_llm(self, llm: "BaseLLM") -> None:
        """Inject the agent LLM for extraction."""
        self._llm = llm

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Fetch web page content and extract key information using LLM."""
        jina_api_key = os.environ.get("JINA_API_KEY")

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, WebFetchToolParams)
        urls = params.url
        goal = params.goal

        self.logger.info("Web fetch URLs: %s, goal: %s", urls, goal)

        goal_key = self._normalize_goal(goal)
        results = []
        duplicate_urls = []
        reused_cached_urls = []
        skipped_failed_urls = []
        fresh_extractions = 0
        executed_urls = []
        batch_seen: set[str] = set()

        for url in urls:
            normalized_url = self._normalize_url(url)
            cache_key = (normalized_url, goal_key)

            if not normalized_url:
                duplicate_urls.append(url)
                continue
            if normalized_url in batch_seen:
                duplicate_urls.append(url)
                continue
            batch_seen.add(normalized_url)

            if cache_key in self._extract_cache:
                reused_cached_urls.append(url)
                results.append(
                    f"[web_fetch] Skipped duplicate URL+goal pair already fetched earlier in this task: {url}"
                )
                continue

            if normalized_url in self._failed_urls and normalized_url not in self._content_cache:
                skipped_failed_urls.append(url)
                results.append(f"[web_fetch] Skipped previously failed URL: {url}")
                continue

            executed_urls.append(url)

            if normalized_url in self._content_cache:
                content = self._content_cache[normalized_url]
            else:
                content = self._fetch_single(url, jina_api_key)
                if content.startswith("[web_fetch]"):
                    self._failed_urls[normalized_url] = self._failed_urls.get(normalized_url, 0) + 1
                    results.append(content)
                    continue
                self._content_cache[normalized_url] = content

            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH]

            extracted = self._extract_with_llm(content, url, goal)
            self._extract_cache[cache_key] = extracted
            fresh_extractions += 1
            results.append(extracted)

        productive = fresh_extractions > 0
        stagnation = bool(urls) and not productive
        if stagnation:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0

        notes = []
        if duplicate_urls:
            notes.append(
                "[web_fetch] Skipped duplicate URLs already requested in this task: "
                + "; ".join(duplicate_urls)
            )
        if reused_cached_urls:
            notes.append(
                "[web_fetch] Reused cached extraction for URLs already fetched with the same goal: "
                + "; ".join(reused_cached_urls)
            )
        if skipped_failed_urls:
            notes.append(
                "[web_fetch] Skipped URLs that already failed earlier in this task: "
                + "; ".join(skipped_failed_urls)
            )
        if stagnation:
            notes.append(
                "[web_fetch] This fetch call produced no new evidence. Do not retry the same URL/goal pair again; change strategy or finish with the best-supported answer."
            )

        response_parts = [*notes, *results]
        if not response_parts:
            response_parts.append(
                "[web_fetch] No new fetch was executed. Change strategy or finish with the best-supported answer."
            )

        response = "\n---\n".join(response_parts)
        info = {
            "urls": urls,
            "goal": goal,
            "executed_urls": executed_urls,
            "duplicate_urls": duplicate_urls,
            "reused_cached_urls": reused_cached_urls,
            "skipped_failed_urls": skipped_failed_urls,
            "fresh_extractions": fresh_extractions,
            "guard": {
                "productive": productive,
                "stagnation": stagnation,
                "stagnation_streak": self._stagnation_streak,
                "message": notes[-1] if stagnation and notes else "",
            },
        }
        return response, info

    def _fetch_single(self, url: str, jina_api_key: str | None) -> str:
        """Fetch a single page using Jina Reader API."""
        headers = {}
        if jina_api_key:
            headers["Authorization"] = f"Bearer {jina_api_key}"

        proxies = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }
        for attempt in range(3):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    proxies=proxies,
                    timeout=50,
                )
                if response.status_code == 200:
                    content = response.text
                    if not content or not content.strip():
                        return f"[web_fetch] Empty content from {url}"
                    return content
                else:
                    self.logger.warning(
                        "Jina API returned %d for %s", response.status_code, url
                    )
                    if attempt == 2:
                        return f"[web_fetch] Failed to fetch {url}: HTTP {response.status_code}"
            except Exception as e:
                self.logger.warning("Web fetch attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    return f"[web_fetch] Failed to fetch {url}: {e}"

        return f"[web_fetch] Failed to fetch {url}"

    def _extract_with_llm(self, content: str, url: str, goal: str) -> str:
        """Use LLM to extract goal-relevant information from web page content."""
        if self._llm is None:
            self.logger.warning("No LLM set for web_fetch extraction, returning raw content")
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[web_fetch] Content truncated."
            return content

        from evomaster.utils.types import Dialog, UserMessage

        try:
            # Retry summary up to 3 times, progressively truncating content
            raw = ""
            summary_content = content
            for retry in range(3):
                prompt = EXTRACTOR_PROMPT.format(webpage_content=summary_content, goal=goal)
                dialog = Dialog(
                    messages=[UserMessage(content=prompt)],
                    tools=[],
                )
                try:
                    response = self._llm.query(dialog)
                    raw = response.content or ""
                except Exception as e:
                    self.logger.warning("LLM extraction attempt %d failed: %s", retry + 1, e)
                    raw = ""

                if len(raw) >= 10:
                    break

                # Truncate content and retry
                truncate_length = int(0.7 * len(summary_content)) if retry < 2 else 25000
                self.logger.info(
                    "[web_fetch] Summary for %s attempt %d/3, truncating to %d chars",
                    url, retry + 1, truncate_length,
                )
                summary_content = summary_content[:truncate_length]

            # Parse JSON
            parsed = self._parse_json(raw)

            if parsed:
                useful = f"The useful information in {url} for user goal \"{goal}\" as follows:\n\n"
                useful += f"Page quality: {parsed.get('page_quality', 'unknown')}\n"
                useful += f"Page type: {parsed.get('page_type', 'unknown')}\n\n"
                evidence = parsed.get("evidence", [])
                if isinstance(evidence, list) and evidence:
                    useful += "Evidence in page:\n"
                    for item in evidence[:8]:
                        useful += f"- {item}\n"
                    useful += "\n"
                else:
                    useful += f"Evidence in page:\n{evidence or 'N/A'}\n\n"
                useful += f"Summary:\n{parsed.get('summary', 'N/A')}\n\n"
                return useful
            else:
                # JSON parse failed, return raw LLM output
                return f"The useful information in {url} for user goal \"{goal}\" as follows:\n\n{raw}"

        except Exception as e:
            self.logger.error("LLM extraction failed for %s: %s", url, e)
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[web_fetch] Content truncated."
            return content

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Attempt to parse JSON from LLM output."""
        if not text:
            return None

        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON wrapped in {}
            left = text.find("{")
            right = text.rfind("}")
            if left != -1 and right != -1 and left < right:
                try:
                    return json.loads(text[left:right + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _normalize_goal(goal: str) -> str:
        return re.sub(r"\s+", " ", goal.strip().lower())

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
