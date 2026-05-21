"""Flash-Searcher style BrowseComp experiment workflow.

This module intentionally keeps the agent topology close to
OPPO-PersonalAI/Flash-Searcher: a planning step, one tool-calling search agent
that keeps the task memory in its dialog, and a no-tool final-answer fallback
used when the search agent does not finish cleanly.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from .state import FlashMemoryStep, FlashSearcherRunState


class FlashSearchExp(BaseExp):
    """Single-agent Flash-Searcher workflow adapted to EvoMaster."""

    def __init__(
        self,
        planner,
        searcher,
        finalizer,
        config,
        max_steps: int = 20,
        max_plans: int = 3,
        planning_interval: int = 1,
        summary_interval: int = 5,
    ):
        super().__init__(agent=searcher, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.searcher = searcher
        self.finalizer = finalizer or planner
        self.max_steps = max(1, int(max_steps))
        self.max_plans = max(1, int(max_plans))
        self.planning_interval = max(1, int(planning_interval))
        self.summary_interval = max(1, int(summary_interval))
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "WebMasterFlashSearcher"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self.logger.info("=" * 80)
        self.logger.info("WebMaster Flash-Searcher task start: %s", task_id)
        self.logger.info("Question: %s", task_description)
        self.logger.info("=" * 80)

        state = FlashSearcherRunState(task_id=task_id, question=task_description)
        result: dict[str, Any] = {
            "task_id": task_id,
            "status": "running",
            "steps": 0,
            "agent_answer": "",
            "ground_truth": self.ground_truth,
        }

        try:
            planning_trajectory = None
            search_trajectory = None
            planning = ""
            search_output = ""
            search_answer = ""
            final_answer = ""
            final_reasoning = ""
            attempt_summaries: list[dict[str, Any]] = []

            for attempt_index in range(1, self.max_plans + 1):
                state.plan_attempts = attempt_index
                planning_trajectory, planning = self._run_text_agent(
                    self.planner,
                    exp_index=(attempt_index - 1) * 2,
                    task_id=f"{task_id}_planning_{attempt_index}",
                    task_type="flash_planning",
                    description=self._planning_input(task_description, attempt_summaries),
                    on_step=on_step,
                )
                state.planning = planning
                state.memory.append(
                    FlashMemoryStep(
                        step_type="planning",
                        content=planning,
                        metadata={
                            "attempt": attempt_index,
                            "max_plans": self.max_plans,
                            "planning_interval": self.planning_interval,
                        },
                    )
                )

                search_trajectory, search_output = self._run_text_agent(
                    self.searcher,
                    exp_index=(attempt_index - 1) * 2 + 1,
                    task_id=f"{task_id}_search_{attempt_index}",
                    task_type="flash_search",
                    description=self._search_input(task_description, planning, attempt_index),
                    images=images,
                    on_step=on_step,
                )
                search_answer = self._sanitize_answer(search_output)
                state.search_status = str(getattr(search_trajectory, "status", "") or "")
                state.search_steps += len(getattr(search_trajectory, "steps", []) or [])
                usage = self._extract_tool_usage(search_trajectory)
                self._merge_tool_usage(state, usage)
                trace_digest = self._trajectory_digest(search_trajectory)
                state.memory.append(
                    FlashMemoryStep(
                        step_type="action_trace_digest",
                        content=trace_digest,
                        metadata={
                            "attempt": attempt_index,
                            "raw_final_output": search_output[:2000],
                            "search_status": state.search_status,
                        },
                    )
                )

                attempt_summary = {
                    "attempt": attempt_index,
                    "planning": planning,
                    "search_status": state.search_status,
                    "search_answer": search_answer,
                    "search_output": search_output[:2000],
                    "trace_digest": trace_digest,
                    "tool_call_counts": usage["tool_call_counts"],
                    "queries": usage["queries"],
                    "urls": usage["urls"],
                }
                attempt_summaries.append(attempt_summary)

                if not self._needs_final_answer_fallback(search_answer, search_trajectory):
                    final_answer = search_answer
                    break

                state.memory.append(
                    FlashMemoryStep(
                        step_type="replan_trigger",
                        content="Search did not produce a clean final answer; returning to planner."
                        if attempt_index < self.max_plans
                        else "Search did not produce a clean final answer after the final plan attempt.",
                        metadata={"attempt": attempt_index, "max_plans": self.max_plans},
                    )
                )

            if not final_answer:
                final_trajectory, final_reasoning = self._run_text_agent(
                    self.finalizer,
                    exp_index=self.max_plans * 2,
                    task_id=f"{task_id}_final_answer",
                    task_type="flash_final_answer",
                    description=self._final_answer_input(
                        question=task_description,
                        attempt_summaries=attempt_summaries,
                    ),
                    on_step=on_step,
                )
                final_answer = self._sanitize_answer(final_reasoning)
                state.memory.append(
                    FlashMemoryStep(
                        step_type="final_answer_fallback",
                        content=final_reasoning,
                        metadata={
                            "fallback_status": str(getattr(final_trajectory, "status", "") or ""),
                        },
                    )
                )

            if not final_answer or final_answer.upper() == "UNKNOWN":
                final_answer = self._best_answer_from_text(search_output, final_reasoning)

            state.final_reasoning = final_reasoning
            state.final_answer = final_answer

            result.update(
                {
                    "status": "completed" if final_answer else "no_answer",
                    "steps": state.search_steps,
                    "agent_answer": final_answer,
                    "flash_searcher_state": state.to_dict(),
                    "planning_trajectory": planning_trajectory,
                    "search_trajectory": search_trajectory,
                }
            )
            self.results.append(result)
            self._save_run_result(result)
        except Exception as exc:
            self.logger.error("WebMaster Flash-Searcher task crashed: %s", exc, exc_info=True)
            result.update({"status": "failed", "error": str(exc)})
            self.results.append(result)
            self._save_run_result(result)

        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info(
            "WebMaster Flash-Searcher task end: status=%s steps=%s",
            result.get("status"),
            result.get("steps"),
        )
        return result

    def save_results(self, output_file: str):
        output_data = [self._serialize_result(result) for result in self.results]
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False, default=str)
        self.logger.info("Results saved to %s", output_file)

    def _run_text_agent(
        self,
        agent,
        exp_index: int,
        task_id: str,
        task_type: str,
        description: str,
        images=None,
        on_step=None,
    ) -> tuple[object, str]:
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=exp_index)
        task = TaskInstance(
            task_id=task_id,
            task_type=task_type,
            description=description,
            images=images or [],
            input_data={
                "benchmark": "browsecomp",
                "architecture": "flash_searcher",
                "max_steps": self.max_steps,
                "max_plans": self.max_plans,
                "planning_interval": self.planning_interval,
                "summary_interval": self.summary_interval,
            },
        )
        trajectory = agent.run(task, on_step=on_step)
        response = self._extract_agent_response(trajectory).strip()
        return trajectory, response

    def _planning_input(
        self,
        question: str,
        attempt_summaries: list[dict[str, Any]] | None = None,
    ) -> str:
        attempt_summaries = attempt_summaries or []
        prompt_parts = [
            "Run the Flash-Searcher planning step for this BrowseComp task.",
            "Decompose the question into concrete search directions, hard constraints, likely pivots, and verification checks.",
            "Do not answer yet unless the answer is already certain. This is a plan/memory step, not a final response.",
            "Return concise Markdown with sections: Facts, Hard constraints, Search plan, Verification plan.",
            f"Question:\n{question}",
        ]
        if attempt_summaries:
            prompt_parts.extend(
                [
                    "Previous plan/search attempts failed to produce a clean final answer.",
                    "Use the summaries below to re-plan: avoid repeated dead ends, keep useful evidence, and propose new search angles.",
                    json.dumps(
                        [
                            {
                                "attempt": item.get("attempt"),
                                "search_status": item.get("search_status"),
                                "search_answer": item.get("search_answer"),
                                "search_output": item.get("search_output"),
                                "tool_call_counts": item.get("tool_call_counts"),
                                "queries": item.get("queries", [])[:20],
                                "urls": item.get("urls", [])[:20],
                                "trace_digest": self._coerce_str(item.get("trace_digest"))[:6000],
                            }
                            for item in attempt_summaries
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
        return "\n\n".join(prompt_parts)

    def _search_input(self, question: str, planning: str, attempt_index: int) -> str:
        return "\n\n".join(
            [
                "You are now the Flash-Searcher tool-calling agent.",
                f"This is plan/search attempt {attempt_index} of {self.max_plans}.",
                "Use the initial plan as memory, then iteratively search and fetch webpages until the answer is verified.",
                "Use google_search for broad discovery and web_fetch for evidence extraction from promising pages.",
                "Before each tool call, reason briefly about what uncertainty the call resolves.",
                "Do not stop at a plausible candidate: verify every hard constraint from the original question.",
                "Ignore benchmark mirrors, answer dumps, and pages that only restate this exact question.",
                "When the answer is ready, call finish with only the exact short answer in the message field.",
                "If evidence remains incomplete near the step limit, still call finish with the best-supported answer, not UNKNOWN.",
                f"Original question:\n{question}",
                "Initial Flash-Searcher plan:",
                planning or "(planner returned no plan)",
            ]
        )

    def _final_answer_input(
        self,
        question: str,
        attempt_summaries: list[dict[str, Any]],
    ) -> str:
        return "\n\n".join(
            [
                f"The Flash-Searcher tool-calling agent did not provide a clean final answer after {len(attempt_summaries)} plan/search attempts.",
                "Use all plans and action traces to provide the final answer now.",
                "Rules:",
                "- Return only the exact short answer; no explanation or citation.",
                "- Do not return UNKNOWN; choose the best-supported answer from the trace.",
                "- Reject candidates that violate hard constraints in the original question.",
                "- Preserve exact names/titles, including subtitles after a colon.",
                f"Original question:\n{question}",
                "Plan/search attempt summaries:",
                json.dumps(
                    [
                        {
                            "attempt": item.get("attempt"),
                            "planning": self._coerce_str(item.get("planning"))[:3000],
                            "search_status": item.get("search_status"),
                            "search_answer": item.get("search_answer"),
                            "search_output": item.get("search_output"),
                            "queries": item.get("queries", [])[:30],
                            "urls": item.get("urls", [])[:30],
                            "trace_digest": self._coerce_str(item.get("trace_digest"))[:8000],
                        }
                        for item in attempt_summaries
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    def _needs_final_answer_fallback(self, answer: str, trajectory) -> bool:
        if not answer or answer.strip().upper() == "UNKNOWN":
            return True
        status = str(getattr(trajectory, "status", "") or "").lower()
        if status and status not in {"completed", "success"}:
            return True
        result = getattr(trajectory, "result", {}) or {}
        return isinstance(result, dict) and result.get("reason") == "max_turns_exceeded"

    def _extract_tool_usage(self, trajectory) -> dict[str, Any]:
        tool_call_counts: Counter[str] = Counter()
        queries: list[str] = []
        urls: list[str] = []

        for step in getattr(trajectory, "steps", []) or []:
            assistant_message = getattr(step, "assistant_message", None)
            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                tool_name = getattr(function, "name", "")
                args_raw = getattr(function, "arguments", "") or "{}"
                tool_call_counts[tool_name] += 1
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                if tool_name == "google_search":
                    queries.extend(self._coerce_str_list(args.get("query")))
                elif tool_name == "web_fetch":
                    urls.extend(self._coerce_str_list(args.get("url")))

        return {
            "tool_call_counts": dict(tool_call_counts),
            "queries": queries,
            "urls": urls,
        }

    def _merge_tool_usage(
        self,
        state: FlashSearcherRunState,
        usage: dict[str, Any],
    ) -> None:
        counts = Counter(state.tool_call_counts)
        counts.update(usage.get("tool_call_counts", {}))
        state.tool_call_counts = dict(counts)
        state.queries.extend(usage.get("queries", []))
        state.urls.extend(usage.get("urls", []))

    def _trajectory_digest(self, trajectory, max_chars: int = 8000) -> str:
        chunks: list[str] = []
        for step in getattr(trajectory, "steps", []) or []:
            step_id = getattr(step, "step_id", "?")
            assistant_message = getattr(step, "assistant_message", None)
            content = self._coerce_str(getattr(assistant_message, "content", ""))
            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            if content:
                chunks.append(f"Step {step_id} assistant: {content[:1200]}")
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                chunks.append(
                    f"Step {step_id} tool_call {getattr(function, 'name', '')}: "
                    f"{self._coerce_str(getattr(function, 'arguments', ''))[:1000]}"
                )
            for tool_response in getattr(step, "tool_responses", []) or []:
                name = getattr(tool_response, "name", "")
                content = self._coerce_str(getattr(tool_response, "content", ""))
                chunks.append(f"Step {step_id} tool_response {name}: {content[:1600]}")
        digest = "\n\n".join(chunks)
        if len(digest) > max_chars:
            return digest[: max_chars // 2] + "\n\n...[trace truncated]...\n\n" + digest[-max_chars // 2 :]
        return digest

    def _sanitize_answer(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        payload = self._extract_json_object(text)
        if isinstance(payload, dict):
            for key in ("answer", "final_answer", "message"):
                value = self._coerce_str(payload.get(key))
                if value:
                    text = value
                    break

        tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
        if tag_match:
            text = tag_match.group(1).strip()
        text = re.sub(r"^Final answer\s*:\s*", "", text, flags=re.I).strip()
        text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.I).strip()
        text = text.strip().strip('"').strip("'").strip()
        for line in text.splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if line:
                return line[:300].strip()
        return ""

    def _best_answer_from_text(self, *texts: str) -> str:
        for text in reversed(texts):
            answer = self._sanitize_answer(text)
            if answer and answer.upper() != "UNKNOWN":
                return answer
        return "best guess unavailable"

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
        candidates = [fenced.group(1)] if fenced else []
        candidates.append(text)
        for candidate in candidates:
            candidate = candidate.strip()
            try:
                payload = json.loads(candidate)
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(candidate[start : end + 1])
                    return payload if isinstance(payload, dict) else {}
                except json.JSONDecodeError:
                    continue
        return {}

    def _coerce_str(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _coerce_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [self._coerce_str(item) for item in value if self._coerce_str(item)]
        return [self._coerce_str(value)] if self._coerce_str(value) else []

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "steps": result.get("steps", 0),
            "agent_answer": result.get("agent_answer", ""),
            "ground_truth": result.get("ground_truth"),
            "flash_searcher_state": result.get("flash_searcher_state", {}),
            "error": result.get("error"),
        }

    def _save_run_result(self, result: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        output_path = Path(self.run_dir) / "result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._serialize_result(result), handle, indent=2, ensure_ascii=False, default=str)
