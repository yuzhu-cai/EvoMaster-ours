"""Compatibility shim for older imports.

Tool implementation has moved to:
`playground/minimal_FrontierScience/tools/frontier_local_tools.py`.
"""

from ..tools.frontier_local_tools import (
    GoogleScholarParams,
    GoogleScholarTool,
    SearchWebParams,
    SearchWebTool,
    VisitWebParams,
    VisitWebTool,
    build_frontier_tools,
)

__all__ = [
    "SearchWebParams",
    "GoogleScholarParams",
    "VisitWebParams",
    "SearchWebTool",
    "GoogleScholarTool",
    "VisitWebTool",
    "build_frontier_tools",
]
