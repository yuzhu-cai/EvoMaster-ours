"""Tool entrypoints for FrontierScience."""

from evomaster.agent.tools.base import BaseTool

from .google_scholar_tool import GoogleScholarParams, GoogleScholarTool
from .search_web_tool import SearchWebParams, SearchWebTool
from .visit_web_tool import VisitWebParams, VisitWebTool


def build_frontier_tools() -> list[BaseTool]:
    return [
        SearchWebTool(),
        GoogleScholarTool(),
        VisitWebTool(),
    ]


__all__ = [
    "SearchWebParams",
    "GoogleScholarParams",
    "VisitWebParams",
    "SearchWebTool",
    "GoogleScholarTool",
    "VisitWebTool",
    "build_frontier_tools",
]
