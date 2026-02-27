"""Chat Agent Web Search 工具

通过 Perplexity (或其他 OpenAI 兼容搜索 API) 进行联网搜索。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)


class WebSearchToolParams(BaseToolParams):
    """Search the web for real-time information using an online search engine.

    Use this tool when you need up-to-date information that may not be in your training data, such as:
    - Current events and news
    - Latest documentation or API references
    - Real-time data (weather, stock prices, etc.)
    - Verifying facts or claims

    The tool returns a synthesized answer based on web search results.
    """

    name: ClassVar[str] = "web_search"

    query: str = Field(description="The search query to look up on the web.")


class WebSearchTool(BaseTool):
    """联网搜索工具"""

    name: ClassVar[str] = "web_search"
    params_class: ClassVar[type[BaseToolParams]] = WebSearchToolParams

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行联网搜索"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, WebSearchToolParams)
        query = params.query

        self.logger.info("Web search query: %s", query)

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

            self.logger.info("Web search completed, result length: %d", len(result))
            return result, {"query": query, "model": self.model, "citations": citations}

        except Exception as e:
            self.logger.error("Web search failed: %s", e)
            return f"Web search failed: {e}", {"error": str(e), "query": query}
