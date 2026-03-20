"""Tool entrypoints for FrontierScience."""

from evomaster.agent.tools.base import BaseTool

from .google_scholar import GoogleScholarParams, GoogleScholarTool
from .pdf_reader import PdfReadParams, PdfReaderTool
from .reflect_answer import ReflectAnswerParams, ReflectAnswerTool
from .search_web import SearchWebParams, SearchWebTool
from .visit_web import VisitWebParams, VisitWebTool


def build_frontier_tools() -> list[BaseTool]:
    return [
        SearchWebTool(),
        GoogleScholarTool(),
        VisitWebTool(),
        PdfReaderTool(),
        ReflectAnswerTool(),
    ]


__all__ = [
    "SearchWebParams",
    "GoogleScholarParams",
    "VisitWebParams",
    "PdfReadParams",
    "ReflectAnswerParams",
    "SearchWebTool",
    "GoogleScholarTool",
    "VisitWebTool",
    "PdfReaderTool",
    "ReflectAnswerTool",
    "build_frontier_tools",
]
