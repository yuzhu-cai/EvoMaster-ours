#!/usr/bin/env python3
"""Smoke test FrontierScience web tools.

Tools under test:
- search_web
- google_scholar
- visit_web

Default mode (`--mode auto`) is CI-friendly:
- If required external conditions are missing (e.g., SERPER_API_KEY, network),
  the case is marked as SKIP instead of FAIL.

Strict mode (`--mode strict`) is production-like:
- Any unsuccessful case is treated as FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import types


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _install_mcp_stubs_if_missing() -> None:
    """Install lightweight MCP stubs so imports work in envs without `mcp` package."""
    if "mcp" in sys.modules:
        return
    mcp_mod = types.ModuleType("mcp")

    class _Dummy:  # pragma: no cover - import compatibility shim
        def __init__(self, *args, **kwargs):
            pass

    mcp_mod.ClientSession = _Dummy
    mcp_mod.StdioServerParameters = _Dummy
    sys.modules["mcp"] = mcp_mod

    client_mod = types.ModuleType("mcp.client")
    sse_mod = types.ModuleType("mcp.client.sse")
    stdio_mod = types.ModuleType("mcp.client.stdio")
    http_mod = types.ModuleType("mcp.client.streamable_http")

    async def _dummy_client(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("MCP stub client should not be used in this script.")

    sse_mod.sse_client = _dummy_client
    stdio_mod.stdio_client = _dummy_client
    http_mod.streamablehttp_client = _dummy_client

    sys.modules["mcp.client"] = client_mod
    sys.modules["mcp.client.sse"] = sse_mod
    sys.modules["mcp.client.stdio"] = stdio_mod
    sys.modules["mcp.client.streamable_http"] = http_mod


_install_mcp_stubs_if_missing()

from playground.frontierscience.tools.google_scholar import GoogleScholarTool
from playground.frontierscience.tools.search_web import SearchWebTool
from playground.frontierscience.tools.visit_web import VisitWebTool


NETWORK_ERROR_HINTS = [
    "name resolution",
    "temporary failure",
    "failed to establish a new connection",
    "connectionerror",
    "max retries exceeded",
    "httpsconnectionpool",
    "timed out",
    "timeout",
    "connection reset",
]


@dataclass
class CaseResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    elapsed_sec: float
    detail: str
    output_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FrontierScience web tools.")
    parser.add_argument(
        "--mode",
        choices=["auto", "strict"],
        default="auto",
        help="auto: skip environment-related failures; strict: any issue is fail.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "playground" / "frontierscience" / "workspace" / "tool_test_report.json",
        help="Path to save JSON report.",
    )
    return parser.parse_args()


def _preview(text: str, max_len: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _is_network_error(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in NETWORK_ERROR_HINTS)


def _run_case(
    name: str,
    fn: Callable[[], tuple[str, str, str]],
    strict: bool,
) -> CaseResult:
    started = time.time()
    try:
        status, detail, output = fn()
    except Exception as exc:  # pragma: no cover - defensive
        status = "FAIL"
        detail = f"Unhandled exception: {exc}"
        output = ""

    if strict and status == "SKIP":
        status = "FAIL"
        detail = f"(strict mode) {detail}"

    return CaseResult(
        name=name,
        status=status,
        elapsed_sec=round(time.time() - started, 3),
        detail=detail,
        output_preview=_preview(output),
    )


def _call_tool(tool, session: object, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    args_json = json.dumps(payload, ensure_ascii=False)
    output, info = tool.execute(session, args_json)
    return str(output), info if isinstance(info, dict) else {}


def _judge_serper_output(tool_name: str, output: str, has_api_key: bool) -> tuple[str, str]:
    # Expected successful prefix from builtin implementations.
    success_prefix = "### Search results for" if tool_name == "search_web" else "### Google Scholar results for"
    if output.startswith(success_prefix):
        return "PASS", "Received structured search results."

    if f"[{tool_name}] Serper API key is not set" in output:
        return "SKIP", "SERPER_API_KEY is missing."

    if output.startswith(f"[{tool_name}]"):
        if (not has_api_key) and ("API error" in output or "Error for" in output or "Failed for" in output):
            return "SKIP", "No API key or unavailable upstream in current environment."
        return "FAIL", f"{tool_name} returned error text."

    # External-script mode may use a different textual format. Treat non-empty as pass.
    if output.strip():
        return "PASS", "Received non-empty output."
    return "FAIL", "Empty output."


def _judge_visit_output(output: str) -> tuple[str, str]:
    if "[visit_web:" in output:
        return "PASS", "Fetched and extracted webpage content."

    if output.startswith("[visit_web]"):
        if _is_network_error(output):
            return "SKIP", "Network appears unavailable in current environment."
        return "FAIL", "visit_web returned tool-level error."

    if output.strip():
        return "PASS", "Received non-empty output."
    return "FAIL", "Empty output."


def main() -> int:
    args = parse_args()
    strict = args.mode == "strict"

    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    # These three tools do not use session methods, so a dummy object is enough.
    session = object()

    has_serper_key = bool(os.getenv("SERPER_API_KEY", "").strip())
    search_tool = SearchWebTool()
    scholar_tool = GoogleScholarTool()
    visit_tool = VisitWebTool()

    def case_search_single() -> tuple[str, str, str]:
        output, _ = _call_tool(
            search_tool,
            session,
            {"query": "weak value amplification quantum metrology"},
        )
        status, detail = _judge_serper_output("search_web", output, has_serper_key)
        return status, detail, output

    def case_search_batch() -> tuple[str, str, str]:
        output, _ = _call_tool(
            search_tool,
            session,
            {"query": ["quantum fisher information", "postselection weak measurement"]},
        )
        status, detail = _judge_serper_output("search_web", output, has_serper_key)
        return status, detail, output

    def case_scholar_single() -> tuple[str, str, str]:
        output, _ = _call_tool(
            scholar_tool,
            session,
            {"query": "weak value amplification entanglement postselection"},
        )
        status, detail = _judge_serper_output("google_scholar", output, has_serper_key)
        return status, detail, output

    def case_scholar_batch() -> tuple[str, str, str]:
        output, _ = _call_tool(
            scholar_tool,
            session,
            {"query": ["quantum metrology Heisenberg limit", "Aharonov Albert Vaidman 1988 weak value"]},
        )
        status, detail = _judge_serper_output("google_scholar", output, has_serper_key)
        return status, detail, output

    def case_visit_single() -> tuple[str, str, str]:
        output, _ = _call_tool(
            visit_tool,
            session,
            {
                "url": "https://example.com",
                "goal": "extract page topic and key facts",
            },
        )
        status, detail = _judge_visit_output(output)
        return status, detail, output

    def case_visit_batch() -> tuple[str, str, str]:
        output, _ = _call_tool(
            visit_tool,
            session,
            {
                "url": ["https://example.com", "https://www.iana.org/domains/reserved"],
                "goal": "extract main topics and compare overlap",
            },
        )
        status, detail = _judge_visit_output(output)
        return status, detail, output

    cases: list[tuple[str, Callable[[], tuple[str, str, str]]]] = [
        ("search_web_single_query", case_search_single),
        ("search_web_batch_query", case_search_batch),
        ("google_scholar_single_query", case_scholar_single),
        ("google_scholar_batch_query", case_scholar_batch),
        ("visit_web_single_url", case_visit_single),
        ("visit_web_batch_url", case_visit_batch),
    ]

    results: list[CaseResult] = []
    for name, fn in cases:
        results.append(_run_case(name, fn, strict))

    summary = {
        "mode": args.mode,
        "has_serper_api_key": has_serper_key,
        "pass": sum(1 for r in results if r.status == "PASS"),
        "fail": sum(1 for r in results if r.status == "FAIL"),
        "skip": sum(1 for r in results if r.status == "SKIP"),
        "results": [asdict(r) for r in results],
    }

    args.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Report: {args.report_json}")
    print("-" * 72)
    for r in results:
        print(f"[{r.status:<4}] {r.name:<30} {r.elapsed_sec:>6.3f}s  {r.detail}")
    print("-" * 72)
    print(f"PASS={summary['pass']}  FAIL={summary['fail']}  SKIP={summary['skip']}")
    print("=" * 72)

    return 1 if summary["fail"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
