"""Chat Agent AI Search 工具

通过 Perplexity (或其他 OpenAI 兼容搜索 API) 进行 AI 综合搜索。
返回基于多个网页内容合成的答案，而非原始搜索结果列表。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)


class AISearchToolParams(BaseToolParams):
    """Search the web using AI-powered search engine (Perplexity-style).

    Returns a synthesized answer based on multiple web sources, with citations.
    Use this when you need a comprehensive, AI-generated answer combining information
    from multiple web pages, rather than a list of raw search results.

    For raw search result links, use google_search instead.
    """

    name: ClassVar[str] = "ai_search"

    query: str = Field(description="The search query to look up on the web.")


class AISearchTool(BaseTool):
    """AI 综合搜索工具（Perplexity 风格）"""

    name: ClassVar[str] = "ai_search"
    params_class: ClassVar[type[BaseToolParams]] = AISearchToolParams

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行 AI 综合搜索"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, AISearchToolParams)
        query = params.query

        self.logger.info("AI search query: %s", query)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
            )
            result = response.choices[0].message.content or ""

            # Perplexity 通过 OpenRouter 返回的 citations 在 response 顶层字段
            citations = getattr(response, "citations", None)
            if citations and isinstance(citations, list):
                refs = "\n".join(
                    f"[{i + 1}] {url}" for i, url in enumerate(citations)
                )
                result = f"{result}\n\nSources:\n{refs}"

            self.logger.info("AI search completed, result length: %d", len(result))
            return result, {"query": query, "model": self.model, "citations": citations}

        except Exception as e:
            self.logger.error("AI search failed: %s", e)
            return f"AI search failed: {e}", {"error": str(e), "query": query}
