"""VerifyMaster experiment workflow.

This workflow uses a simplified BrowseComp architecture:
- Planner proposes the next subtask or emits a final answer.
- Executor uses web tools to gather evidence for one focused subtask.
- A separate finalizer agent produces a forced final answer at max steps.

The goal of this file is to keep the loop simple while preserving convergence
signals and a reliable fallback answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance


@dataclass
class VerificationResult:
    passed: bool
    confidence: float
    feedback: str = ""


@dataclass
class ExecutionStep:
    step_number: int
    subtask: str
    evidence: str = ""
    verification: Optional[VerificationResult] = None
    retry_count: int = 0


@dataclass
class ResearchTrajectory:
    steps: list[ExecutionStep] = field(default_factory=list)

    def add_step(self, step: ExecutionStep) -> None:
        self.steps.append(step)

    def to_context(self, max_recent: int = 5) -> str:
        if not self.steps:
            return "(no research steps yet)"

        parts: list[str] = []
        if len(self.steps) > max_recent:
            earlier_steps = self.steps[:-max_recent]
            earlier_pass = sum(
                1 for step in earlier_steps if step.verification and step.verification.passed
            )
            earlier_fail = sum(
                1 for step in earlier_steps if step.verification and not step.verification.passed
            )
            parts.append(
                "[Earlier steps summarized: "
                f"{len(earlier_steps)} steps, {earlier_pass} useful, {earlier_fail} blocked]"
            )

        for step in self.steps[-max_recent:]:
            if step.verification is None:
                verdict = "unrated"
            elif step.verification.passed:
                verdict = f"useful/{step.verification.confidence:.2f}"
            else:
                verdict = f"blocked/{step.verification.confidence:.2f}"
            parts.append(
                f"S{step.step_number} {verdict}\n"
                f"Subtask: {step.subtask}\n"
                f"Evidence: {step.evidence[:800]}"
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
        max_retries: int = 1,
    ):
        super().__init__(agent=None, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.executor = executor
        self.verifier = verifier  # kept only for compatibility / easy rollback
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

    def _normalize_subtask_signature(self, text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r"https?://\S+", " ", normalized)
        normalized = re.sub(r"\b\d{4}\b", " <year> ", normalized)
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", " <num> ", normalized)
        normalized = re.sub(r"[^a-z0-9<>\s]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _subtask_complexity(self, text: str) -> int:
        lowered = text.lower()
        score = 0
        score += len(re.findall(r"\b(and|or|with|whose|which|that|between|from|while)\b", lowered))
        score += len(re.findall(r"\bverify\b|\bcheck\b|\bidentify\b|\bfind\b|\breturn\b", lowered))
        score += len(re.findall(r"\b\d{4}\b", lowered))
        score += text.count(";")
        if len(text) > 180:
            score += 3
        elif len(text) > 120:
            score += 2
        elif len(text) > 80:
            score += 1
        return score

    def _evidence_gist(self, evidence: str) -> str:
        cleaned = evidence.replace("Evidence report:", "").strip()
        for line in cleaned.splitlines():
            stripped = line.strip(" -*")
            if stripped:
                return self._truncate(stripped, 180)
        return self._truncate(cleaned, 180)

    def _is_timeout_step(self, step: ExecutionStep) -> bool:
        evidence = (step.evidence or "").lower()
        feedback = (step.verification.feedback if step.verification else "").lower()
        return "max turns" in evidence or "max turns" in feedback

    def _repeat_count_for_subtask(self, subtask: str) -> int:
        signature = self._normalize_subtask_signature(subtask)
        if not signature:
            return 0
        return sum(
            1
            for step in self.trajectory.steps
            if not step.subtask.startswith("[")
            and self._normalize_subtask_signature(step.subtask) == signature
        )

    def _recent_pass_steps(self, limit: int = 4) -> list[ExecutionStep]:
        steps = [
            step
            for step in self.trajectory.steps
            if step.verification and step.verification.passed and step.evidence
        ]
        return steps[-limit:]

    def _recent_fail_steps(self, limit: int = 4) -> list[ExecutionStep]:
        steps = [
            step for step in self.trajectory.steps if step.verification and not step.verification.passed
        ]
        return steps[-limit:]

    def _repeated_angle_summaries(self, limit: int = 3) -> list[str]:
        groups: dict[str, list[ExecutionStep]] = {}
        for step in self.trajectory.steps:
            if step.subtask.startswith("["):
                continue
            signature = self._normalize_subtask_signature(step.subtask)
            if not signature:
                continue
            groups.setdefault(signature, []).append(step)

        repeated = [group for group in groups.values() if len(group) >= 2]
        repeated.sort(key=lambda group: (len(group), group[-1].step_number), reverse=True)

        summaries: list[str] = []
        for group in repeated[:limit]:
            representative = self._truncate(group[0].subtask, 120)
            summaries.append(f"{representative} (attempted {len(group)}x)")
        return summaries

    def _infer_phase(self) -> str:
        passed_steps = [
            step for step in self.trajectory.steps if step.verification and step.verification.passed
        ]
        if not passed_steps:
            return "candidate_discovery"

        last_pass_text = passed_steps[-1].subtask.lower()
        attribute_keywords = (
            "birth year",
            "birthplace",
            "runtime",
            "distance",
            "founder",
            "date",
            "year",
            "full name",
            "surname",
            "month",
            "title",
        )
        if any(keyword in last_pass_text for keyword in attribute_keywords):
            return "attribute_extraction"
        if len(passed_steps) <= 2:
            return "candidate_narrowing"
        return "candidate_verification"

    def _convergence_advice(self) -> str:
        recent_fails = self._recent_fail_steps(limit=3)
        timeout_fails = [step for step in recent_fails if self._is_timeout_step(step)]
        if len(timeout_fails) >= 2:
            if all(self._subtask_complexity(step.subtask) < 8 for step in timeout_fails):
                return (
                    "Repeated narrow steps are stalling. Switch source type, query wording, or return to a "
                    "small candidate-generation step instead of shrinking forever."
                )
            return "The recent steps were still too bundled. Split the next lookup so it checks one clue on one target."

        repeated_angles = self._repeated_angle_summaries(limit=1)
        if repeated_angles:
            return "You already revisited the same angle. Compare, eliminate, or move to the next decisive clue."

        if len(self._recent_pass_steps(limit=6)) >= 3:
            return "Prefer extracting the final missing attribute from the best-supported candidate."

        return "Keep one leading hypothesis or one tiny candidate set, then resolve one decisive missing clue."

    def _build_state_snapshot(self) -> str:
        lines = [f"Phase: {self._infer_phase()}"]

        pass_steps = self._recent_pass_steps(limit=4)
        if pass_steps:
            lines.append("Confirmed progress:")
            for step in pass_steps:
                lines.append(
                    f"- S{step.step_number}: {self._truncate(step.subtask, 120)} -> {self._evidence_gist(step.evidence)}"
                )
        else:
            lines.append("Confirmed progress: none yet")

        repeated_angles = self._repeated_angle_summaries(limit=2)
        if repeated_angles:
            lines.append("Repeated angles to avoid:")
            for summary in repeated_angles:
                lines.append(f"- {summary}")

        fail_steps = self._recent_fail_steps(limit=3)
        if fail_steps:
            lines.append("Recent blockers:")
            for step in fail_steps:
                label = "timeout" if self._is_timeout_step(step) else "weak evidence"
                lines.append(
                    f"- S{step.step_number} {label}: {self._truncate(step.subtask, 100)} -> "
                    f"{self._truncate(step.verification.feedback if step.verification else step.evidence, 180)}"
                )

        lines.append(f"Convergence rule: {self._convergence_advice()}")
        return "\n".join(lines)

    def _build_executor_state(self) -> str:
        lines = [f"Phase: {self._infer_phase()}"]
        repeated_angles = self._repeated_angle_summaries(limit=1)
        if repeated_angles:
            lines.append(f"Repeated angle to avoid: {repeated_angles[0]}")
        lines.append(f"Current guidance: {self._convergence_advice()}")
        return "\n".join(lines)

    def _planner_input(self, task: str, step: int) -> str:
        return "\n\n".join(
            [
                f"Original task:\n{task}",
                f"Current step: {step}/{self.max_steps}",
                f"Structured state:\n{self._build_state_snapshot()}",
                f"Research progress:\n{self.trajectory.to_context()}",
                "Return exactly one block: either <task>...</task> for the next focused lookup, "
                "or <answer>...</answer> if you are ready to answer.",
            ]
        )

    def _executor_input(self, task: str, step: int, retry_feedback: str = "") -> str:
        parts = [
            f"Assigned subtask (step {step}):\n{task}",
            f"Investigation state:\n{self._build_executor_state()}",
            f"Research context:\n{self.trajectory.to_context(3)}",
        ]
        if retry_feedback:
            parts.append(
                "Feedback from the failed attempt:\n"
                f"{retry_feedback}\n\n"
                "Do not repeat the same search path. Change query wording, target source, or verification angle."
            )
        parts.append(
            "Use the browsing tools to collect concrete evidence for this subtask, then write a concise evidence report with source URLs."
        )
        return "\n\n".join(parts)

    def _finalizer_input(self, task: str) -> str:
        return "\n\n".join(
            [
                f"Original task:\n{task}",
                f"Structured state:\n{self._build_state_snapshot()}",
                f"Research progress:\n{self.trajectory.to_context(20)}",
                "Return <answer>...</answer> with the single best final answer from the current evidence.",
            ]
        )

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.I | re.S)
        if match:
            return match.group(1).strip()
        return None

    def _extract_candidate_from_evidence(self, evidence: str) -> str:
        patterns = [
            r"Best-supported extraction:\s*\**(.+?)\**(?:\.|\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Therefore,\s*\**(.+?)\**(?:\.|\n|$)",
            r"The person is\s+\**(.+?)\**(?:\.|\n|$)",
            r"Conclusion:\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, evidence, re.I | re.S)
            if match:
                return self._truncate(match.group(1).strip(" *"), 120)
        return ""

    def _sanitize_final_answer(self, text: str) -> str:
        answer = text.strip()
        answer = re.sub(r"^<answer>|</answer>$", "", answer, flags=re.I).strip()
        answer = re.sub(r"^Answer\s*:\s*", "", answer, flags=re.I).strip()
        if answer.lower().startswith("<task>"):
            return ""
        if "\n" in answer:
            answer = answer.splitlines()[0].strip()
        return answer

    def _best_known_answer(self) -> str:
        for step in reversed(self.trajectory.steps):
            candidate = self._extract_candidate_from_evidence(step.evidence)
            if candidate:
                return candidate

        for step in reversed(self.trajectory.steps):
            if step.verification and step.verification.passed and step.evidence:
                gist = self._evidence_gist(step.evidence)
                if gist:
                    return gist

        return "Unknown"

    def _run_final_fallback(self, task_description: str) -> str:
        finalizer = self.finalizer or self.planner
        _, output = self._run_text_agent(
            finalizer,
            task_id="finalizer",
            task_type="finalizer",
            description=self._finalizer_input(task_description),
        )
        self.logger.info("Finalizer output: %s", output[:300])

        final_answer = self._extract_tag(output, "answer") or output.strip()
        final_answer = self._sanitize_final_answer(final_answer)
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

                evidence, local_verdict = self._run_executor_step(
                    subtask,
                    step_num,
                    on_step=on_step,
                    retry_index=0,
                )

                retry_count = 0
                while (
                    (not local_verdict.passed or local_verdict.confidence < self.local_threshold)
                    and retry_count < self.max_retries
                ):
                    retry_count += 1
                    self.logger.warning(
                        "Executor step failed for step %s: %s",
                        step_num,
                        local_verdict.feedback,
                    )
                    evidence, local_verdict = self._run_executor_step(
                        subtask,
                        step_num,
                        on_step=on_step,
                        retry_feedback=local_verdict.feedback,
                        retry_index=retry_count,
                    )

                self.trajectory.add_step(
                    ExecutionStep(
                        step_number=step_num,
                        subtask=subtask,
                        evidence=evidence,
                        verification=local_verdict,
                        retry_count=retry_count,
                    )
                )

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

    def _run_executor_step(
        self,
        subtask: str,
        step_num: int,
        on_step=None,
        retry_feedback: str = "",
        retry_index: int = 0,
    ) -> tuple[str, VerificationResult]:
        task_id = f"e{step_num}" if retry_index == 0 else f"e{step_num}r{retry_index}"
        trajectory, evidence = self._run_text_agent(
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
        failure_reason = self._extract_failure_reason(trajectory)

        if evidence.strip():
            return evidence, VerificationResult(
                passed=True,
                confidence=1.0,
                feedback="Executor returned usable evidence.",
            )

        if failure_reason == "max_turns_exceeded":
            repeat_count = self._repeat_count_for_subtask(subtask)
            complexity = self._subtask_complexity(subtask)

            if complexity >= 8:
                diagnostic = (
                    "Executor exhausted max turns before producing an evidence report. "
                    "The subtask still bundled multiple filters or verification demands."
                )
                verdict = VerificationResult(
                    passed=False,
                    confidence=0.15,
                    feedback=(
                        "Executor hit max turns because the subtask was still too bundled. "
                        "Split the next step so it checks one clue on one target or returns a tiny candidate set."
                    ),
                )
                return diagnostic, verdict

            if repeat_count >= 2:
                diagnostic = (
                    "Executor exhausted max turns on a repeated narrow search angle before producing an evidence report. "
                    "The issue is likely retrieval sparsity or source mismatch, not just task size."
                )
                verdict = VerificationResult(
                    passed=False,
                    confidence=0.15,
                    feedback=(
                        "Executor hit max turns on a repeated narrow angle. Do not keep shrinking forever; "
                        "switch source type, query wording, or return to a different candidate-generation step."
                    ),
                )
                return diagnostic, verdict

            diagnostic = (
                "Executor exhausted max turns before producing an evidence report. "
                "The clue may be retrieval-sparse or source-mismatched."
            )
            verdict = VerificationResult(
                passed=False,
                confidence=0.15,
                feedback=(
                    "Executor hit max turns. Keep the next step small, but change source type or search angle "
                    "instead of repeating the same pattern."
                ),
            )
            return diagnostic, verdict

        if failure_reason:
            diagnostic = f"Executor failed before producing an evidence report: {failure_reason}."
            verdict = VerificationResult(
                passed=False,
                confidence=0.2,
                feedback=(
                    f"Executor failed before returning evidence ({failure_reason}). "
                    "Use a narrower subtask with fewer simultaneous constraints."
                ),
            )
            return diagnostic, verdict

        return evidence, VerificationResult(
            passed=False,
            confidence=0.2,
            feedback="Executor returned no evidence.",
        )
