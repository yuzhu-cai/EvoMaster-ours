"""FrontierScience iterative reflection tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

logger = logging.getLogger(__name__)


class ReflectAnswerParams(BaseToolParams):
    """Audit the current draft answer, create a concrete repair plan, and save a reflection checkpoint for another solve iteration."""

    name: ClassVar[str] = "reflect_answer"
    question: str = Field(description="Original task question.")
    draft_answer: str = Field(description="Current draft answer to inspect before another iteration.")
    output_file: str = Field(default="solution_refined.md", description="Checkpoint file for the next refined answer.")


class ReflectAnswerTool(BaseTool):
    """Structured reflection scaffold that triggers another solve iteration."""

    name: ClassVar[str] = "reflect_answer"
    params_class: ClassVar[type[BaseToolParams]] = ReflectAnswerParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
            assert isinstance(params, ReflectAnswerParams)
            text, meta = local_reflect_answer(
                params.question,
                params.draft_answer,
                params.output_file,
                base_dir=_resolve_session_workspace(session),
            )
            return text, {"tool": self.name, **meta}
        except Exception as exc:
            logger.error("reflect_answer failed: %s", exc, exc_info=True)
            return f"[reflect_answer] Error: {exc}", {"tool": self.name, "error": str(exc)}


def _resolve_session_workspace(session: Any) -> Path | None:
    workspace_path = getattr(session, "get_workspace_path", lambda: None)()
    if workspace_path:
        return Path(workspace_path).expanduser().resolve()

    session_config = getattr(session, "config", None)
    config_workspace = getattr(session_config, "workspace_path", None)
    if config_workspace:
        return Path(config_workspace).expanduser().resolve()

    return None


def _build_reflection_payload(question: str, draft_answer: str) -> dict[str, Any]:
    stripped = draft_answer.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    answer_length = len(stripped)

    likely_gaps: list[str] = []
    if answer_length < 600:
        likely_gaps.append("The draft may be too short for a complete scientific answer; re-check whether all requested parts are covered.")
    if "uncertain" in stripped.lower() or "not sure" in stripped.lower():
        likely_gaps.append("The draft signals uncertainty; verify the uncertain claims with targeted evidence if retrieval is allowed.")
    if not any(token in stripped.lower() for token in ["because", "therefore", "thus", "hence"]):
        likely_gaps.append("The reasoning chain may be under-explained; ensure the final answer justifies its key conclusion.")
    if not any(ch.isdigit() for ch in stripped):
        likely_gaps.append("No numeric detail is present; confirm whether the task requires values, formulas, units, or quantitative comparison.")
    if not likely_gaps:
        likely_gaps.append("Re-check for missing task parts, unsupported claims, vague wording, and opportunities for stronger evidence.")

    repair_plan = [
        "Re-read the task and make a private checklist of every requested deliverable, comparison axis, and constraint.",
        "Compare the current draft against that checklist and identify what is still missing, weakly supported, or vague.",
        "If any missing point depends on external evidence and retrieval is allowed, do another targeted tool pass before revising.",
        "Revise the answer completely rather than making only cosmetic edits; strengthen coverage, evidence, and precision.",
        "Keep solution.md as the original pre-reflection draft. Apply post-reflection changes only to solution_refined.md so the refinement is easy to inspect before finish.",
    ]

    return {
        "question": question,
        "draft_length": answer_length,
        "draft_paragraphs": len(lines),
        "likely_gaps": likely_gaps,
        "repair_plan": repair_plan,
        "iteration_required": True,
    }


def local_reflect_answer(
    question: str,
    draft_answer: str,
    output_file: str = "solution_refined.md",
    base_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    refined = draft_answer.strip()
    if not refined:
        raise ValueError("draft_answer is empty")

    payload = _build_reflection_payload(question, refined)

    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        root_dir = (base_dir or Path.cwd()).expanduser().resolve()
        output_path = root_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(refined + "\n", encoding="utf-8")

    reflection_path = output_path.with_name("reflection_plan.json")
    reflection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    text = "\n".join(
        [
            "[reflect_answer] Reflection audit completed. Another solve iteration is now required.",
            "You must not finish yet.",
            "Next actions:",
            "1. Re-check the task against the current draft and identify missing deliverables.",
            "2. If needed, use tools again to fill evidence gaps or inspect the paper more precisely.",
            "3. Keep solution.md unchanged, and write all post-reflection improvements only into solution_refined.md.",
            "4. Only call finish after the revised answer is complete.",
            f"Reflection plan file: {reflection_path}",
            f"Checkpoint file: {output_path}",
            "Likely gaps:",
            *[f"- {gap}" for gap in payload["likely_gaps"]],
        ]
    )
    return text, {
        "output_file": str(output_path),
        "reflection_plan": str(reflection_path),
        "draft_length": payload["draft_length"],
        "iteration_required": True,
    }
