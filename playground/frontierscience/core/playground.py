"""FrontierScience playground implementation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from .exp import FrontierScienceExp
from ..tools import build_frontier_tools


@register_playground("frontierscience")
class FrontierSciencePlayground(BasePlayground):
    """A minimal playground specialized for science QA with web/scholar tools."""

    _TASK_MODE_OPEN = "open_book"
    _TASK_MODE_CLOSED = "closed_book"
    _TASK_MODE_CLOSED_STRICT = "closed_book_strict"

    _EXTERNAL_RETRIEVAL_TOOLS = {"search_web", "google_scholar", "visit_web"}
    _SKILL_ENTRY_TOOL = "use_skill"

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).resolve().parents[3] / "configs" / "frontierscience"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("general_agent")
        self._frontier_tool_names: list[str] = []
        self._base_enabled_tool_names: dict[str, list[str] | None] = {}

    def setup(self) -> None:
        self.logger.info("Setting up minimal FrontierScience playground...")
        self._setup_session()
        self._setup_agents()
        self.logger.info("Minimal FrontierScience playground setup complete")

    def _setup_tools(
        self,
        skill_config: dict | None = None,
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        super()._setup_tools(skill_config=skill_config, tool_config=tool_config)

        custom_tools = build_frontier_tools()
        for tool in custom_tools:
            self.tools.register(tool)

        self._frontier_tool_names = [tool.name for tool in custom_tools]
        self.logger.info(
            "Registered FrontierScience tools (%d): %s",
            len(self._frontier_tool_names),
            self._frontier_tool_names,
        )

    def _setup_agents(self) -> None:
        super()._setup_agents()
        if not self._frontier_tool_names:
            return

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            if agent.enabled_tool_names is None:
                agent.enabled_tool_names = []
            added_names: list[str] = []
            for tool_name in self._frontier_tool_names:
                if tool_name not in agent.enabled_tool_names:
                    agent.enabled_tool_names.append(tool_name)
                    added_names.append(tool_name)
            self.logger.info(
                "Agent '%s' tool setup complete, newly added=%s, total=%d",
                slot_name,
                added_names if added_names else "[]",
                len(agent.enabled_tool_names),
            )
            self.logger.debug(
                "Agent '%s' final enabled tools: %s",
                slot_name,
                agent.enabled_tool_names,
            )
            self._base_enabled_tool_names[slot_name] = (
                list(agent.enabled_tool_names) if agent.enabled_tool_names is not None else None
            )

    def _route_task_mode(self, task_description: str, task_meta: dict[str, str]) -> tuple[str, str]:
        raw_mode = str(task_meta.get("task_mode", "")).strip().lower()
        mode_alias_map = {
            self._TASK_MODE_OPEN: self._TASK_MODE_OPEN,
            "open": self._TASK_MODE_OPEN,
            "retrieval": self._TASK_MODE_OPEN,
            "research": self._TASK_MODE_OPEN,
            "open_book": self._TASK_MODE_OPEN,
            self._TASK_MODE_CLOSED: self._TASK_MODE_CLOSED,
            "closed": self._TASK_MODE_CLOSED,
            "closed_book": self._TASK_MODE_CLOSED,
            self._TASK_MODE_CLOSED_STRICT: self._TASK_MODE_CLOSED_STRICT,
            "strict_closed": self._TASK_MODE_CLOSED_STRICT,
            "no_retrieval": self._TASK_MODE_CLOSED_STRICT,
            "no_research": self._TASK_MODE_CLOSED_STRICT,
        }
        mapped_mode = mode_alias_map.get(raw_mode)
        if mapped_mode is not None:
            return mapped_mode, f"task_meta_override:{raw_mode}"

        text = (task_description or "").lower()

        # Only use closed-book mode when the task explicitly forbids external sources.
        strict_closed_markers = [
            "without external",
            "no external sources",
            "do not use external",
            "do not search the web",
            "no web search",
            "using only the provided context",
            "from the provided context only",
            "self-contained question",
            "closed-book",
        ]
        for marker in strict_closed_markers:
            if marker in text:
                return self._TASK_MODE_CLOSED_STRICT, f"strict_closed_marker:{marker}"

        # Open-book signals: freshness, citations, paper-level evidence, or source hunting.
        open_keywords = [
            "latest",
            "recent",
            "state-of-the-art",
            "literature",
            "review",
            "survey",
            "citation",
            "cite",
            "arxiv",
            "pubmed",
            "google scholar",
            "web search",
            "find papers",
            "latest work",
            "references",
            "bibliography",
            "citation needed",
            "find source",
            "look up",
            "clinical trial",
            "trial phase",
            "gaia dr3",
        ]
        for kw in open_keywords:
            if kw in text:
                return self._TASK_MODE_OPEN, f"open_keyword:{kw}"

        if "doi:" in text or "arxiv:" in text:
            return self._TASK_MODE_OPEN, "open_reference_marker:doi_or_arxiv"
        if re.search(r"\[[0-9]{1,3}\]", text):
            return self._TASK_MODE_OPEN, "open_reference_marker:bracket_citation"

        # Default to open-book with conditional retrieval to avoid under-retrieval.
        return self._TASK_MODE_OPEN, "default_open_book"

    def _apply_task_routing(self, task_mode: str) -> None:
        if task_mode not in {self._TASK_MODE_OPEN, self._TASK_MODE_CLOSED, self._TASK_MODE_CLOSED_STRICT}:
            task_mode = self._TASK_MODE_OPEN

        for slot_name, agent in self.agents.items():
            if agent is None:
                continue
            base_names = self._base_enabled_tool_names.get(slot_name)
            if base_names is None:
                continue

            routed_names = list(base_names)
            if task_mode == self._TASK_MODE_CLOSED_STRICT:
                routed_names = [
                    name
                    for name in routed_names
                    if name not in self._EXTERNAL_RETRIEVAL_TOOLS and name != self._SKILL_ENTRY_TOOL
                ]

            agent.enabled_tool_names = routed_names
            self.logger.info(
                "Task routing applied to '%s': mode=%s, enabled_tools=%s",
                slot_name,
                task_mode,
                agent.enabled_tool_names,
            )

    def _create_exp(self):
        agent = self.agents.get("general_agent") or self.agent
        if agent is None:
            raise RuntimeError("No available agent for FrontierScienceExp.")
        exp = FrontierScienceExp(agent, self.config)
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def _extract_task_meta(self, task_description: str) -> tuple[str, dict[str, str]]:
        pattern = re.compile(
            r"^\s*\[frontierscience_task_meta\]\s*\n(?P<body>.*?)\n\s*\[/frontierscience_task_meta\]\s*\n?",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.match(task_description or "")
        if match is None:
            return task_description, {}

        body = match.group("body")
        meta: dict[str, str] = {}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key:
                meta[key] = val

        cleaned_description = (task_description[match.end() :]).lstrip()
        return cleaned_description, meta

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        default_task_id = getattr(self, "task_id", "exp_001")
        cleaned_description, task_meta = self._extract_task_meta(task_description)
        runtime_task_id = task_meta.get("task_id", default_task_id) or default_task_id
        runtime_task_type = task_meta.get("task_type", "frontier_science") or "frontier_science"
        task_input_data = {k: v for k, v in task_meta.items() if k not in {"task_id", "task_type"}}
        task_mode, routing_reason = self._route_task_mode(cleaned_description, task_meta)
        task_input_data["task_mode"] = task_mode
        task_input_data["routing_reason"] = routing_reason

        if task_mode == self._TASK_MODE_OPEN:
            retrieval_policy = "conditional_external_research"
        elif task_mode == self._TASK_MODE_CLOSED:
            retrieval_policy = "prefer_local_reasoning_allow_external_if_required"
        else:
            retrieval_policy = "no_external_retrieval"
        task_input_data["retrieval_policy"] = retrieval_policy

        self.task_id = runtime_task_id
        try:
            self.setup()
            self._apply_task_routing(task_mode)
            self._setup_trajectory_file(output_file)

            exp = self._create_exp()
            image_count = len(images or [])
            self.logger.info(
                "Running FrontierScienceExp (task_id=%s, task_type=%s, mode=%s, reason=%s, image_count=%d)",
                runtime_task_id,
                runtime_task_type,
                task_mode,
                routing_reason,
                image_count,
            )
            result = exp.run(
                task_description=cleaned_description,
                task_id=runtime_task_id,
                task_type=runtime_task_type,
                input_data=task_input_data,
                images=images,
            )
            self.logger.info(
                "FrontierScienceExp finished (task_id=%s, task_type=%s, status=%s, steps=%s)",
                runtime_task_id,
                runtime_task_type,
                result.get("status"),
                result.get("steps"),
            )
            return result
        except Exception as exc:
            self.logger.error(
                "Minimal FrontierScience playground failed (task_id=%s): %s",
                runtime_task_id,
                exc,
                exc_info=True,
            )
            return {
                "task_id": runtime_task_id,
                "task_type": runtime_task_type,
                "status": "failed",
                "steps": 0,
                "error": str(exc),
            }
        finally:
            self.cleanup()
