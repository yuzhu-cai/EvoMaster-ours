"""VerifyMaster experiment workflow.

This workflow keeps the original verification-oriented architecture:
- Planner proposes the next subtask or emits a final answer candidate.
- Executor uses web tools to gather evidence for one focused subtask.
- Verifier scores each evidence block and audits the final answer.

The goal of this file is functional wiring, not benchmark optimization.
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
            return "(no verified research steps yet)"

        parts: list[str] = []
        if len(self.steps) > max_recent:
            earlier = ", ".join(f"S{s.step_number}" for s in self.steps[:-max_recent])
            parts.append(f"[Earlier steps summarized: {earlier}]")

        for step in self.steps[-max_recent:]:
            if step.verification is None:
                verdict = "unverified"
            elif step.verification.passed:
                verdict = f"PASS/{step.verification.confidence:.2f}"
            else:
                verdict = f"FAIL/{step.verification.confidence:.2f}"
            parts.append(
                f"S{step.step_number} {verdict}\n"
                f"Subtask: {step.subtask}\n"
                f"Evidence: {step.evidence[:800]}"
            )
        return "\n\n".join(parts)


def parse_verdict(text: str) -> VerificationResult:
    verdict_match = re.search(r"<verdict>\s*(PASS|FAIL)\s*</verdict>", text, re.I)
    confidence_match = re.search(r"<confidence>\s*(\d+(?:\.\d+)?)\s*</confidence>", text)
    feedback_match = re.search(r"<feedback>\s*(.*?)\s*</feedback>", text, re.I | re.S)

    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return VerificationResult(
        passed=bool(verdict_match and verdict_match.group(1).upper() == "PASS"),
        confidence=confidence,
        feedback=feedback_match.group(1).strip() if feedback_match else text.strip(),
    )


class VerifyMasterExp(BaseExp):
    """Verification-first multi-agent BrowseComp experiment."""

    def __init__(
        self,
        planner,
        executor,
        verifier,
        config,
        max_steps: int = 20,
        local_threshold: float = 0.55,
        global_threshold: float = 0.65,
        max_retries: int = 1,
    ):
        super().__init__(agent=None, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
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

    def _local_verify(self, subtask: str, evidence: str) -> VerificationResult:
        if not self.verifier:
            return VerificationResult(True, 1.0, "Verifier disabled")

        description = (
            "Local evidence verification for a BrowseComp research step.\n\n"
            f"Subtask:\n{subtask}\n\n"
            f"Evidence:\n{evidence[:4000]}\n\n"
            "Judge whether the evidence is directly relevant, concrete, and useful for the subtask. "
            "PASS if it contains specific facts or strong directional evidence, even if it is not exhaustive. "
            "FAIL only if it is mostly vague, off-target, contradictory, or guess-based."
        )
        _, response = self._run_text_agent(
            self.verifier,
            task_id=f"lv_{len(self.trajectory.steps) + 1}",
            task_type="verifier",
            description=description,
        )
        return parse_verdict(response)

    def _global_verify(self, task: str, answer: str) -> VerificationResult:
        if not self.verifier:
            return VerificationResult(True, 1.0, "Verifier disabled")

        chain = "\n\n".join(
            f"S{s.step_number}\nSubtask: {s.subtask}\nEvidence: {s.evidence[:1200]}"
            for s in self.trajectory.steps
            if s.evidence
        ) or "(no evidence collected)"

        description = (
            "Global answer verification for a BrowseComp question.\n\n"
            f"Original question:\n{task}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            f"Evidence chain:\n{chain}\n\n"
            "Primary question: is the candidate answer directly supported by the collected evidence?\n"
            "PASS when the answer itself is strongly supported and the identified entity is reasonably established.\n"
            "Do not require every background clue to be independently proven if the remaining uncertainty is minor "
            "and would not realistically change the final answer.\n"
            "FAIL only if the answer is weakly supported, the entity is genuinely ambiguous, major competing "
            "candidates remain plausible, or the answer still depends on guesswork."
        )
        _, response = self._run_text_agent(
            self.verifier,
            task_id="gv",
            task_type="verifier",
            description=description,
        )
        return parse_verdict(response)

    def _planner_input(self, task: str, step: int) -> str:
        return "\n\n".join(
            [
                f"Original task:\n{task}",
                f"Current step: {step}/{self.max_steps}",
                f"Research progress:\n{self.trajectory.to_context()}",
                "Return exactly one block: either <task>...</task> for the next focused lookup, "
                "or <answer>...</answer> if the evidence is already sufficient.",
            ]
        )

    def _executor_input(self, task: str, step: int, retry_feedback: str = "") -> str:
        parts = [
            f"Assigned subtask (step {step}):\n{task}",
            f"Research context:\n{self.trajectory.to_context(3)}",
        ]
        if retry_feedback:
            parts.append(
                "Verifier feedback from the failed attempt:\n"
                f"{retry_feedback}\n\n"
                "Do not repeat the same search path. Change query wording, target source, or verification angle."
            )
        parts.append(
            "Use the browsing tools to collect concrete evidence for this subtask, then write a concise evidence report with source URLs."
        )
        return "\n\n".join(parts)

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.I | re.S)
        if match:
            return match.group(1).strip()
        return None

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
                    global_verdict = self._global_verify(task_description, answer_candidate)
                    self.logger.info(
                        "Global verification: %s (%.2f)",
                        "PASS" if global_verdict.passed else "FAIL",
                        global_verdict.confidence,
                    )
                    if global_verdict.passed and global_verdict.confidence >= self.global_threshold:
                        self.final_found = True
                        result["status"] = "completed"
                        result["final_answer"] = answer_candidate
                        result["agent_answer"] = answer_candidate
                        break

                    self.trajectory.add_step(
                        ExecutionStep(
                            step_number=step_num,
                            subtask=f"[Global verification failed] {global_verdict.feedback}",
                            verification=global_verdict,
                        )
                    )
                    continue

                subtask = self._extract_tag(planner_output, "task") or planner_output.strip()
                if not subtask:
                    subtask = task_description
                self.logger.info("Executor subtask: %s", subtask[:200])

                _, evidence = self._run_text_agent(
                    self.executor,
                    task_id=f"e{step_num}",
                    task_type="executor",
                    description=self._executor_input(subtask, step_num),
                    on_step=on_step,
                )
                local_verdict = self._local_verify(subtask, evidence)

                retry_count = 0
                while (
                    (not local_verdict.passed or local_verdict.confidence < self.local_threshold)
                    and retry_count < self.max_retries
                ):
                    retry_count += 1
                    self.logger.warning(
                        "Local verification failed for step %s: %s",
                        step_num,
                        local_verdict.feedback,
                    )
                    _, evidence = self._run_text_agent(
                        self.executor,
                        task_id=f"e{step_num}r{retry_count}",
                        task_type="executor",
                        description=self._executor_input(
                            subtask,
                            step_num,
                            retry_feedback=local_verdict.feedback,
                        ),
                        on_step=on_step,
                    )
                    local_verdict = self._local_verify(subtask, evidence)

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
                _, planner_output = self._run_text_agent(
                    self.planner,
                    task_id="final",
                    task_type="planner",
                    description=(
                        f"Original task:\n{task_description}\n\n"
                        f"Research progress:\n{self.trajectory.to_context(20)}\n\n"
                        "Return <answer>...</answer> with the best-supported answer from the current evidence."
                    ),
                )
                final_answer = self._extract_tag(planner_output, "answer") or planner_output.strip()
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
