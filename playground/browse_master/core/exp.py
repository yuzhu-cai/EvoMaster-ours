"""BrowseMaster experiment workflow.

Organizes the search run into explicit stages:
1. Build the benchmark task payload
2. Execute the browsing agent
3. Extract the final answer
4. Summarize trajectory diagnostics for later analysis
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance


BROWSE_TOOL_NAMES = {"google_search", "web_fetch", "think", "finish"}


class BrowseMasterExp(BaseExp):
    """Single-agent benchmark web search experiment."""

    def __init__(self, agent, config):
        super().__init__(agent, config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "BrowseMaster"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self._log_task_start(task_id, task_description)
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)

        task = self._build_task(
            task_id=task_id,
            task_description=task_description,
            images=images,
        )

        try:
            trajectory = self.agent.run(task, on_step=on_step)
        except Exception as exc:
            self.logger.error("Browse task crashed: %s", exc, exc_info=True)
            result = self._build_exception_result(task_id, exc)
            self.results.append(result)
            return result

        result = self._build_result(task_id, trajectory)
        self.results.append(result)
        self._log_task_end(result)
        return result

    def save_results(self, output_file: str):
        """Save experiment results including diagnostics."""
        output_data = [self._serialize_result(result) for result in self.results]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info("Results saved to %s", output_file)

    def _build_task(
        self,
        task_id: str,
        task_description: str,
        images: list[str] | None = None,
    ) -> TaskInstance:
        return TaskInstance(
            task_id=task_id,
            task_type="search",
            description=task_description,
            input_data={"benchmark": "browsecomp"},
            images=images or [],
        )

    def _build_result(self, task_id: str, trajectory) -> dict[str, Any]:
        agent_answer = self._normalize_answer(self._extract_answer(trajectory))
        analysis = self._analyze_trajectory(trajectory, agent_answer)

        return {
            "task_id": task_id,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
            "trajectory": trajectory,
            "agent_answer": agent_answer,
            "ground_truth": self.ground_truth,
            "analysis": analysis,
        }

    def _build_exception_result(self, task_id: str, exc: Exception) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "status": "failed",
            "steps": 0,
            "trajectory": None,
            "agent_answer": "",
            "ground_truth": self.ground_truth,
            "analysis": {
                "answer_found": False,
                "answer_source": "none",
                "finish_called": False,
                "failure_reason": str(exc),
                "tool_call_counts": {},
                "browse_tool_counts": {},
                "non_browse_tool_counts": {},
                "stagnation_steps": 0,
                "last_tool_names": [],
            },
            "error": str(exc),
        }

    def _analyze_trajectory(self, trajectory, agent_answer: str) -> dict[str, Any]:
        tool_call_counts: Counter[str] = Counter()
        browse_tool_counts: Counter[str] = Counter()
        non_browse_tool_counts: Counter[str] = Counter()
        finish_called = False
        stagnation_steps = 0
        last_tool_names: list[str] = []

        for step in trajectory.steps:
            tool_names: list[str] = []
            assistant_message = step.assistant_message
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_names.append(tool_name)
                tool_call_counts[tool_name] += 1
                if tool_name in BROWSE_TOOL_NAMES:
                    browse_tool_counts[tool_name] += 1
                else:
                    non_browse_tool_counts[tool_name] += 1
                if tool_name == "finish":
                    finish_called = True

            if tool_names:
                last_tool_names = tool_names

            if self._step_has_stagnation_signal(step):
                stagnation_steps += 1

        answer_source = "none"
        if finish_called and agent_answer:
            answer_source = "finish"
        elif agent_answer:
            answer_source = "assistant_text"

        failure_reason = self._extract_failure_reason(trajectory)

        return {
            "answer_found": bool(agent_answer),
            "answer_source": answer_source,
            "finish_called": finish_called,
            "failure_reason": failure_reason,
            "tool_call_counts": dict(tool_call_counts),
            "browse_tool_counts": dict(browse_tool_counts),
            "non_browse_tool_counts": dict(non_browse_tool_counts),
            "stagnation_steps": stagnation_steps,
            "last_tool_names": last_tool_names,
        }

    def _extract_failure_reason(self, trajectory) -> str:
        result = getattr(trajectory, "result", {}) or {}
        if isinstance(result, dict):
            reason = result.get("reason")
            if isinstance(reason, str):
                return reason
        return ""

    def _step_has_stagnation_signal(self, step) -> bool:
        for tool_message in step.tool_responses:
            info = getattr(tool_message, "meta", {}).get("info", {})
            if not isinstance(info, dict):
                continue
            guard = info.get("guard", {})
            if isinstance(guard, dict) and guard.get("stagnation"):
                return True
        return False

    def _extract_answer(self, trajectory) -> str:
        answer = self._extract_agent_response(trajectory)
        if answer:
            return answer

        for step in reversed(getattr(trajectory, "steps", [])):
            assistant_message = step.assistant_message
            if assistant_message is None:
                continue

            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if tool_call.function.name != "finish":
                    continue
                try:
                    payload = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    payload = {}
                finish_message = payload.get("message", "")
                if isinstance(finish_message, str) and finish_message.strip():
                    return finish_message

            if not tool_calls:
                content = getattr(assistant_message, "content", "")
                if isinstance(content, str) and content.strip():
                    return content

        return ""

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        trajectory = result.get("trajectory")
        trajectory_dump = trajectory.model_dump() if trajectory is not None else None

        payload = {
            "task_id": result["task_id"],
            "status": result["status"],
            "steps": result["steps"],
            "agent_answer": result.get("agent_answer", ""),
            "ground_truth": result.get("ground_truth"),
            "analysis": result.get("analysis", {}),
            "trajectory": trajectory_dump,
        }
        if "error" in result:
            payload["error"] = result["error"]
        return payload

    def _log_task_start(self, task_id: str, task_description: str) -> None:
        self.logger.info("=" * 80)
        self.logger.info("Browse task start: %s", task_id)
        self.logger.info("Question: %s", task_description)
        self.logger.info("=" * 80)

    def _log_task_end(self, result: dict[str, Any]) -> None:
        analysis = result.get("analysis", {})
        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info(
            "Browse task end: status=%s steps=%s finish_called=%s answer_source=%s stagnation_steps=%s",
            result.get("status"),
            result.get("steps"),
            analysis.get("finish_called"),
            analysis.get("answer_source"),
            analysis.get("stagnation_steps"),
        )
        if analysis.get("failure_reason"):
            self.logger.info("Failure reason: %s", analysis["failure_reason"])
        if analysis.get("non_browse_tool_counts"):
            self.logger.warning(
                "Non-browse tools were used during benchmark solving: %s",
                analysis["non_browse_tool_counts"],
            )
        self.logger.info("Tool call counts: %s", analysis.get("tool_call_counts", {}))
        self.logger.info("=" * 80)

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return answer.strip()
