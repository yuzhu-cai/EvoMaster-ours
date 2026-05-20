"""Tool entrypoints for WebMaster."""

from evomaster.agent.tools.base import BaseTool

from .google_search import GoogleSearchTool, GoogleSearchToolParams
from .web_fetch import WebFetchTool, WebFetchToolParams


def build_browse_tools() -> list[BaseTool]:
    return [
        GoogleSearchTool(),
        WebFetchTool(),
    ]


__all__ = [
    "GoogleSearchToolParams",
    "WebFetchToolParams",
    "GoogleSearchTool",
    "WebFetchTool",
]
