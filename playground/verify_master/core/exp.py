"""VerifyMaster experiment workflow.

This workflow uses a minimal BrowseComp architecture:
- Planner proposes the next subtask or emits a final answer.
- Executor privately browses and returns only a short answer to that subtask.
- A separate finalizer agent produces a forced final answer at max steps.

The planner only sees the original task plus the history of `(subtask, answer)` pairs.
It does not inspect evidence reports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance


UNKNOWN_ANSWER = "UNKNOWN"


@dataclass
class ExecutionStep:
    step_number: int
    subtask: str
    answer: str = ""
    note: str = ""
    retry_count: int = 0


@dataclass
class ResearchTrajectory:
    steps: list[ExecutionStep] = field(default_factory=list)

    def add_step(self, step: ExecutionStep) -> None:
        self.steps.append(step)

    def to_context(self, max_recent: int = 8) -> str:
        if not self.steps:
            return "(no prior subtask-answer pairs)"

        parts: list[str] = []
        if len(self.steps) > max_recent:
            earlier_steps = self.steps[:-max_recent]
            known = sum(1 for step in earlier_steps if step.answer and step.answer != UNKNOWN_ANSWER)
            unknown = len(earlier_steps) - known
            parts.append(
                "[Earlier steps summarized: "
                f"{len(earlier_steps)} steps, {known} answered, {unknown} unknown]"
            )

        for step in self.steps[-max_recent:]:
            parts.append(
                f"S{step.step_number}\n"
                f"Subtask: {step.subtask}\n"
                f"Answer: {step.answer or UNKNOWN_ANSWER}"
            )
        return "\n\n".join(parts)


class VerifyMasterExp(BaseExp):
    """Planner-executor BrowseComp experiment with a forced finalizer."""

    def __init__(
        self,
        planner,
        executor,
        verifier,
        config,
        finalizer=None,
        max_steps: int = 15,
        local_threshold: float = 0.55,
        global_threshold: float = 0.65,
        max_retries: int = 0,
    ):
        super().__init__(agent=None, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.executor = executor
        self.verifier = verifier  # unused; kept only for compatibility / easy rollback
        self.finalizer = finalizer
        self.max_steps = max_steps
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.max_retries = max_retries
        self.trajectory = ResearchTrajectory()
        self.final_found = False
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "VerifyMaster"

    def _run_text_agent(
        self,
        agent,
        task_id: str,
        task_type: str,
        description: str,
        on_step=None,
    ) -> tuple[object, str]:
        task = TaskInstance(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_data={"benchmark": "browsecomp"},
        )
        trajectory = agent.run(task, on_step=on_step)
        response = self._extract_agent_response(trajectory).strip()
        return trajectory, response

    def _extract_failure_reason(self, trajectory) -> str:
        result = getattr(trajectory, "result", {}) or {}
        if isinstance(result, dict):
            reason = result.get("reason")
            if isinstance(reason, str):
                return reason
        return ""

    def _truncate(self, text: str, limit: int = 220) -> str:
        compact = re.sub(r"\s+", " ", (text or "")).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.I | re.S)
        if match:
            return match.group(1).strip()
        return None

    def _sanitize_executor_answer(self, text: str) -> str:
        answer = self._extract_tag(text, "answer") or text.strip()
        answer = re.sub(r"^Answer\s*:\s*", "", answer, flags=re.I).strip()
        answer = re.sub(r"^Final answer\s*:\s*", "", answer, flags=re.I).strip()
        if not answer:
            return UNKNOWN_ANSWER

        for line in answer.splitlines():
            line = line.strip()
            if line:
                answer = line
                break

        answer = self._truncate(answer, 120)
        if not answer or answer.lower().startswith("<task>"):
            return UNKNOWN_ANSWER
        return answer

    def _sanitize_final_answer(self, text: str) -> str:
        answer = self._extract_tag(text, "answer") or text.strip()
        answer = re.sub(r"^Answer\s*:\s*", "", answer, flags=re.I).strip()
        if not answer:
            return ""
        for line in answer.splitlines():
            line = line.strip()
            if line:
                answer = line
                break
        if answer.lower().startswith("<task>"):
            return ""
        return self._truncate(answer, 160)

    def _best_known_answer(self) -> str:
        for step in reversed(self.trajectory.steps):
            if step.answer and step.answer != UNKNOWN_ANSWER:
                return step.answer
        return "Unknown"

    def _planner_input(self, task: str, step: int) -> str:
        return "\n\n".join(
            [
                f"Original task:\n{task}",
                f"Current step: {step}/{self.max_steps}",
                f"Subtask-answer history:\n{self.trajectory.to_context()}",
                "Return exactly one block: either <task>...</task> for the next focused lookup, "
                "or <answer>...</answer> if the history already lets you answer the original task.",
            ]
        )

    def _executor_input(self, task: str, step: int, retry_feedback: str = "") -> str:
        parts = [
            f"Assigned subtask (step {step}):\n{task}",
            f"Recent subtask-answer history:\n{self.trajectory.to_context(4)}",
        ]
        if retry_feedback:
            parts.append(f"Retry note:\n{retry_feedback}")
        parts.append(
            "Use the browsing tools privately to solve the subtask, then return only the subtask answer."
        )
        return "\n\n".join(parts)

    def _finalizer_input(self, task: str) -> str:
        return "\n\n".join(
            [
                f"Original task:\n{task}",
                f"Subtask-answer history:\n{self.trajectory.to_context(20)}",
                "Return <answer>...</answer> with the best final answer you can infer from the history.",
            ]
        )

    def _run_executor_step(
        self,
        subtask: str,
        step_num: int,
        on_step=None,
        retry_feedback: str = "",
        retry_index: int = 0,
    ) -> tuple[str, str]:
        task_id = f"e{step_num}" if retry_index == 0 else f"e{step_num}r{retry_index}"
        trajectory, raw_output = self._run_text_agent(
            self.executor,
            task_id=task_id,
            task_type="executor",
            description=self._executor_input(
                subtask,
                step_num,
                retry_feedback=retry_feedback,
            ),
            on_step=on_step,
        )
        answer = self._sanitize_executor_answer(raw_output)
        failure_reason = self._extract_failure_reason(trajectory)

        if answer != UNKNOWN_ANSWER:
            return answer, ""

        if failure_reason == "max_turns_exceeded":
            return UNKNOWN_ANSWER, "Executor ran out of turns on this subtask."
        if failure_reason:
            return UNKNOWN_ANSWER, f"Executor failed: {failure_reason}."
        return UNKNOWN_ANSWER, "Executor did not return a usable short answer."

    def _run_final_fallback(self, task_description: str) -> str:
        finalizer = self.finalizer or self.planner
        _, output = self._run_text_agent(
            finalizer,
            task_id="finalizer",
            task_type="finalizer",
            description=self._finalizer_input(task_description),
        )
        self.logger.info("Finalizer output: %s", output[:300])

        final_answer = self._sanitize_final_answer(output)
        if final_answer:
            return final_answer
        return self._best_known_answer()

    def run(self, task_description: str, task_id: str = "exp_001", images=None, on_step=None) -> dict:
        self.logger.info("=" * 80)
        self.logger.info("Verify task start: %s", task_id)
        self.logger.info("Question: %s", task_description)
        self.logger.info("=" * 80)

        self.trajectory = ResearchTrajectory()
        self.final_found = False

        result = {
            "task_id": task_id,
            "status": "running",
            "steps": 0,
            "final_answer": "",
            "agent_answer": "",
            "ground_truth": self.ground_truth,
        }

        try:
            for step_num in range(1, self.max_steps + 1):
                _, planner_output = self._run_text_agent(
                    self.planner,
                    task_id=f"p{step_num}",
                    task_type="planner",
                    description=self._planner_input(task_description, step_num),
                )
                self.logger.info("Planner output: %s", planner_output[:300])

                answer_candidate = self._extract_tag(planner_output, "answer")
                if answer_candidate:
                    answer_candidate = self._sanitize_final_answer(answer_candidate)
                    if answer_candidate:
                        self.final_found = True
                        result["status"] = "completed"
                        result["final_answer"] = answer_candidate
                        result["agent_answer"] = answer_candidate
                        break

                subtask = self._extract_tag(planner_output, "task") or planner_output.strip()
                if not subtask:
                    subtask = task_description
                self.logger.info("Executor subtask: %s", subtask[:200])

                answer, note = self._run_executor_step(
                    subtask,
                    step_num,
                    on_step=on_step,
                    retry_index=0,
                )

                retry_count = 0
                while answer == UNKNOWN_ANSWER and retry_count < self.max_retries:
                    retry_count += 1
                    self.logger.warning("Executor answer unknown for step %s: %s", step_num, note)
                    answer, note = self._run_executor_step(
                        subtask,
                        step_num,
                        on_step=on_step,
                        retry_feedback=note,
                        retry_index=retry_count,
                    )

                self.trajectory.add_step(
                    ExecutionStep(
                        step_number=step_num,
                        subtask=subtask,
                        answer=answer,
                        note=note,
                        retry_count=retry_count,
                    )
                )
                self.logger.info("Executor answer: %s", answer)

            if not self.final_found:
                final_answer = self._run_final_fallback(task_description)
                result["status"] = "max_steps"
                result["final_answer"] = final_answer
                result["agent_answer"] = final_answer

            result["steps"] = len(self.trajectory.steps)
            if result["status"] == "running":
                result["status"] = "completed" if self.final_found else "max_steps"
            self.results.append(result)

        except Exception as exc:
            self.logger.error("Verify task crashed: %s", exc, exc_info=True)
            result.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "steps": len(self.trajectory.steps),
                }
            )
            self.results.append(result)

        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info(
            "Verify task end: status=%s steps=%s",
            result.get("status"),
            result.get("steps"),
        )
        return result
