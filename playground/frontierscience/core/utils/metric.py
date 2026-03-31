"""Rubric-based metric for FrontierScience evolving search."""

from __future__ import annotations

import json
import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from evomaster.core.exp import extract_agent_response
from evomaster.utils.types import TaskInstance

logger = logging.getLogger(__name__)

FINAL_SUBSCORE_KEYS = (
    "task_completion",
    "factual_correctness",
    "evidence_grounding",
    "completeness",
    "scientific_reasonableness",
    "clarity",
)
TRAJECTORY_SUBSCORE_KEYS = (
    "retrieval_quality",
    "evidence_usage",
    "reasoning_consistency",
    "revision_effectiveness",
    "exploration_efficiency",
)
ALL_SUBSCORE_KEYS = FINAL_SUBSCORE_KEYS + TRAJECTORY_SUBSCORE_KEYS
RETRIEVAL_TOOL_NAMES = {"search_web", "google_scholar", "visit_web", "read_paper_pdf"}


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def clamp_score(value: Any, lower: float = 0.0, upper: float = 5.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return lower
    if math.isnan(numeric) or math.isinf(numeric):
        return lower
    return max(lower, min(upper, numeric))


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _listify_texts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _weighted_average(score_map: dict[str, float], weights: dict[str, float], keys: tuple[str, ...]) -> float:
    total_weight = sum(max(weights.get(key, 0.0), 0.0) for key in keys)
    if total_weight <= 0:
        return 0.0
    score = sum(clamp_score(score_map.get(key, 0.0)) * max(weights.get(key, 0.0), 0.0) for key in keys)
    return score / (5.0 * total_weight)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _count_markers(text: str, patterns: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(lowered.count(pattern) for pattern in patterns)


def summarize_trajectory(trajectory: Any, final_answer: str = "", previous_answer: str = "") -> dict[str, Any]:
    """Extract stable, judge-friendly trajectory signals from runtime trajectory data."""
    traj_dict = _safe_dict(trajectory)
    dialogs = traj_dict.get("dialogs") or getattr(trajectory, "dialogs", []) or []

    step_count = 0
    assistant_messages = 0
    tool_sequence: list[str] = []
    tool_arguments: list[str] = []
    assistant_texts: list[str] = []

    for dialog in dialogs:
        messages = dialog.get("messages", []) if isinstance(dialog, dict) else getattr(dialog, "messages", []) or []
        for message in messages:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if hasattr(role, "value"):
                role = role.value
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "") or ""
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
            tool_calls = tool_calls or []
            if role == "assistant":
                assistant_messages += 1
                if content and not tool_calls:
                    assistant_texts.append(str(content))
            for tool_call in tool_calls:
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                name = function.get("name", "") if isinstance(function, dict) else getattr(function, "name", "") or ""
                arguments = function.get("arguments", "") if isinstance(function, dict) else getattr(function, "arguments", "") or ""
                if name:
                    tool_sequence.append(name)
                    tool_arguments.append(str(arguments))
        steps = dialog.get("steps") if isinstance(dialog, dict) else getattr(dialog, "steps", None)
        if isinstance(steps, int):
            step_count = max(step_count, steps)

    if step_count <= 0:
        step_count = len(dialogs) or assistant_messages

    combined_text = "\n".join(assistant_texts + [final_answer])
    retrieval_count = sum(1 for tool in tool_sequence if tool in RETRIEVAL_TOOL_NAMES)
    pdf_reads = sum(1 for tool in tool_sequence if tool == "read_paper_pdf")
    repeated_calls = sum(1 for idx in range(1, len(tool_sequence)) if tool_sequence[idx] == tool_sequence[idx - 1])
    evidence_markers = _count_markers(
        combined_text,
        ["according to", "et al.", "doi", "arxiv", "figure", "table", "page ", "section ", "pmid", "http"],
    )
    formula_markers = len(re.findall(r"\\\(|\\\[|=|∝|≤|≥", combined_text))
    answer_words = _word_count(final_answer)
    answer_similarity = SequenceMatcher(None, previous_answer or "", final_answer or "").ratio() if previous_answer else 0.0

    summary_lines = [
        f"steps={step_count}",
        f"assistant_messages={assistant_messages}",
        f"tools={tool_sequence or ['none']}",
        f"retrieval_calls={retrieval_count}",
        f"pdf_reads={pdf_reads}",
        f"repeated_tool_calls={repeated_calls}",
        f"evidence_markers={evidence_markers}",
        f"formula_markers={formula_markers}",
        f"answer_words={answer_words}",
    ]
    if previous_answer:
        summary_lines.append(f"answer_delta_similarity={answer_similarity:.3f}")

    return {
        "step_count": step_count,
        "assistant_messages": assistant_messages,
        "tool_sequence": tool_sequence,
        "tool_arguments_preview": tool_arguments[:8],
        "retrieval_count": retrieval_count,
        "pdf_reads": pdf_reads,
        "repeated_tool_calls": repeated_calls,
        "evidence_markers": evidence_markers,
        "formula_markers": formula_markers,
        "answer_words": answer_words,
        "answer_delta_similarity": answer_similarity,
        "trajectory_summary": "; ".join(summary_lines),
    }


def default_metric_weights() -> dict[str, Any]:
    return {
        "overall": {"final_answer": 0.7, "trajectory": 0.3},
        "final_answer": {
            "task_completion": 1.2,
            "factual_correctness": 1.3,
            "evidence_grounding": 1.2,
            "completeness": 1.0,
            "scientific_reasonableness": 1.0,
            "clarity": 0.8,
        },
        "trajectory": {
            "retrieval_quality": 1.0,
            "evidence_usage": 1.2,
            "reasoning_consistency": 1.0,
            "revision_effectiveness": 0.9,
            "exploration_efficiency": 0.9,
        },
        "penalties": {
            "missing_evidence": 0.08,
            "repeated_loop": 0.06,
            "no_improvement": 0.05,
            "empty_answer": 0.4,
        },
    }


def heuristic_metric_result(
    *,
    problem: str,
    final_answer: str,
    trajectory_features: dict[str, Any],
    previous_answer: str = "",
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fallback metric when judge output is unavailable."""
    del problem
    cfg = deepcopy(weights or default_metric_weights())
    answer = final_answer.strip()
    answer_words = trajectory_features.get("answer_words", _word_count(answer))
    evidence_markers = int(trajectory_features.get("evidence_markers", 0))
    retrieval_count = int(trajectory_features.get("retrieval_count", 0))
    pdf_reads = int(trajectory_features.get("pdf_reads", 0))
    repeated_calls = int(trajectory_features.get("repeated_tool_calls", 0))
    answer_delta_similarity = float(trajectory_features.get("answer_delta_similarity", 0.0))

    final_scores = {
        "task_completion": 4.0 if answer_words >= 220 else 2.5 if answer_words >= 120 else 1.0 if answer_words else 0.0,
        "factual_correctness": 3.0 if answer else 0.0,
        "evidence_grounding": min(5.0, 1.0 + evidence_markers * 0.7 + pdf_reads * 0.5),
        "completeness": 4.0 if answer_words >= 260 else 2.5 if answer_words >= 140 else 1.0 if answer_words else 0.0,
        "scientific_reasonableness": 3.5 if evidence_markers or formula_markers_in_text(answer) else 2.5 if answer else 0.0,
        "clarity": 4.0 if "\n" in answer or len(re.findall(r"\d+\.", answer)) >= 2 else 3.0 if answer else 0.0,
    }
    trajectory_scores = {
        "retrieval_quality": min(5.0, 1.0 + retrieval_count * 1.1),
        "evidence_usage": min(5.0, 1.0 + evidence_markers * 0.8 + pdf_reads * 0.8),
        "reasoning_consistency": 3.5 if answer else 0.0,
        "revision_effectiveness": 4.0 if previous_answer and answer_delta_similarity < 0.95 else 2.0 if previous_answer else 2.5,
        "exploration_efficiency": 4.0 if repeated_calls == 0 else 2.5 if repeated_calls <= 1 else 1.0,
    }
    subscores = {**final_scores, **trajectory_scores}
    final_answer_score = _weighted_average(subscores, cfg["final_answer"], FINAL_SUBSCORE_KEYS)
    trajectory_score = _weighted_average(subscores, cfg["trajectory"], TRAJECTORY_SUBSCORE_KEYS)
    penalty = 0.0
    if not answer:
        penalty += cfg["penalties"]["empty_answer"]
    if evidence_markers == 0:
        penalty += cfg["penalties"]["missing_evidence"]
    if repeated_calls > 1:
        penalty += cfg["penalties"]["repeated_loop"]
    if previous_answer and answer_delta_similarity > 0.97:
        penalty += cfg["penalties"]["no_improvement"]
    overall = max(
        0.0,
        min(
            1.0,
            cfg["overall"]["final_answer"] * final_answer_score + cfg["overall"]["trajectory"] * trajectory_score - penalty,
        ),
    )
    strengths: list[str] = []
    weaknesses: list[str] = []
    if retrieval_count > 0:
        strengths.append("Trajectory includes explicit retrieval rather than answering without evidence.")
    if evidence_markers > 0:
        strengths.append("Answer contains citation-like or evidence-grounded phrasing.")
    if not strengths:
        strengths.append("Answer is non-empty and can serve as a search candidate.")
    if evidence_markers == 0:
        weaknesses.append("Evidence grounding is weak because the answer does not visibly cite or anchor claims.")
    if repeated_calls > 1:
        weaknesses.append("Trajectory repeats similar tool usage, which suggests inefficient exploration.")
    if previous_answer and answer_delta_similarity > 0.97:
        weaknesses.append("Improve action changed the answer very little, so revision effectiveness is low.")
    if answer_words < 140:
        weaknesses.append("Answer is short relative to the benchmark's rubric style, which favors complete multi-step explanations.")
    suggestions = [
        "Tie major claims to retrieved evidence or paper content instead of unsupported summary.",
        "Preserve rubric-style completeness by covering each requested sub-question in order.",
        "Make improve steps substantive: fix missing derivations, quantities, or comparisons rather than paraphrasing.",
    ]
    return {
        "overall_score": overall,
        "final_answer_score": final_answer_score,
        "trajectory_score": trajectory_score,
        "subscores": subscores,
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:4],
        "improvement_suggestions": suggestions[:4],
        "is_valid": bool(answer),
        "reason": "heuristic_fallback",
        "trajectory_summary": trajectory_features.get("trajectory_summary", ""),
        "evaluation_mode": "heuristic",
    }


def formula_markers_in_text(text: str) -> int:
    return len(re.findall(r"\\\(|\\\[|=|∝|≤|≥", text or ""))


@dataclass
class FrontierScienceMetricEvaluator:
    """Structured metric evaluator for MCTS reward and final scoring."""

    metric_agent: Any | None = None
    weights: dict[str, Any] = field(default_factory=default_metric_weights)
    judge_fn: Callable[..., dict[str, Any] | str] | None = None

    def evaluate(
        self,
        *,
        problem: str,
        final_answer: str,
        trajectory: Any,
        previous_answer: str = "",
        reference_rubric: str = "",
        action_type: str = "draft",
        is_final: bool = False,
    ) -> dict[str, Any]:
        trajectory_features = summarize_trajectory(trajectory, final_answer=final_answer, previous_answer=previous_answer)
        heuristic = heuristic_metric_result(
            problem=problem,
            final_answer=final_answer,
            trajectory_features=trajectory_features,
            previous_answer=previous_answer,
            weights=self.weights,
        )

        raw_result: dict[str, Any] | None = None
        if self.judge_fn is not None:
            try:
                response = self.judge_fn(
                    problem=problem,
                    final_answer=final_answer,
                    trajectory_features=trajectory_features,
                    previous_answer=previous_answer,
                    reference_rubric=reference_rubric,
                    action_type=action_type,
                    is_final=is_final,
                )
                raw_result = response if isinstance(response, dict) else _extract_json(str(response))
            except Exception as exc:
                logger.warning("Judge callback failed, fallback to heuristic metric: %s", exc)
        elif self.metric_agent is not None:
            raw_result = self._evaluate_with_agent(
                problem=problem,
                final_answer=final_answer,
                trajectory_features=trajectory_features,
                previous_answer=previous_answer,
                reference_rubric=reference_rubric,
                action_type=action_type,
                is_final=is_final,
            )

        return self._normalize_result(
            raw_result=raw_result,
            heuristic=heuristic,
            trajectory_features=trajectory_features,
            final_answer=final_answer,
            previous_answer=previous_answer,
        )

    def _evaluate_with_agent(
        self,
        *,
        problem: str,
        final_answer: str,
        trajectory_features: dict[str, Any],
        previous_answer: str,
        reference_rubric: str,
        action_type: str,
        is_final: bool,
    ) -> dict[str, Any] | None:
        prompt_kwargs = {
            "problem": problem,
            "final_answer": final_answer or "[empty]",
            "previous_answer": previous_answer or "[none]",
            "reference_rubric": reference_rubric or "Use the FrontierScience benchmark style: multi-dimensional, partial-credit, evidence-grounded rubric.",
            "trajectory_summary": trajectory_features.get("trajectory_summary", ""),
            "trajectory_features_json": json.dumps(trajectory_features, ensure_ascii=False, indent=2),
            "evaluation_mode": "final" if is_final else "intermediate",
            "action_type": action_type,
            "weights_json": json.dumps(self.weights, ensure_ascii=False, indent=2),
        }
        original_kwargs = self.metric_agent._prompt_format_kwargs.copy()
        self.metric_agent._prompt_format_kwargs.update(prompt_kwargs)
        try:
            task = TaskInstance(
                task_id=f"metric_{action_type}_{'final' if is_final else 'intermediate'}",
                task_type="metric",
                description="Evaluate the scientific answer and trajectory with the configured rubric metric.",
                input_data={},
            )
            trajectory = self.metric_agent.run(task)
            response = extract_agent_response(trajectory)
            return _extract_json(response or "")
        except Exception as exc:
            logger.warning("Metric agent failed, fallback to heuristic metric: %s", exc)
            return None
        finally:
            self.metric_agent._prompt_format_kwargs = original_kwargs

    def _normalize_result(
        self,
        *,
        raw_result: dict[str, Any] | None,
        heuristic: dict[str, Any],
        trajectory_features: dict[str, Any],
        final_answer: str,
        previous_answer: str,
    ) -> dict[str, Any]:
        subscores = {key: 0.0 for key in ALL_SUBSCORE_KEYS}
        if raw_result:
            for key in ALL_SUBSCORE_KEYS:
                subscores[key] = clamp_score(_safe_dict(raw_result.get("subscores")).get(key, heuristic["subscores"].get(key, 0.0)))
        else:
            subscores.update({key: clamp_score(value) for key, value in heuristic["subscores"].items()})

        final_answer_score = _weighted_average(subscores, self.weights["final_answer"], FINAL_SUBSCORE_KEYS)
        trajectory_score = _weighted_average(subscores, self.weights["trajectory"], TRAJECTORY_SUBSCORE_KEYS)
        overall = (
            self.weights["overall"]["final_answer"] * final_answer_score
            + self.weights["overall"]["trajectory"] * trajectory_score
        )
        penalties = 0.0
        if trajectory_features.get("evidence_markers", 0) == 0:
            penalties += self.weights["penalties"]["missing_evidence"]
        if trajectory_features.get("repeated_tool_calls", 0) > 1:
            penalties += self.weights["penalties"]["repeated_loop"]
        if previous_answer and trajectory_features.get("answer_delta_similarity", 0.0) > 0.97:
            penalties += self.weights["penalties"]["no_improvement"]
        if not final_answer.strip():
            penalties += self.weights["penalties"]["empty_answer"]

        normalized = {
            "overall_score": max(0.0, min(1.0, float(raw_result.get("overall_score", overall)) if raw_result else overall)),
            "final_answer_score": max(0.0, min(1.0, float(raw_result.get("final_answer_score", final_answer_score)) if raw_result else final_answer_score)),
            "trajectory_score": max(0.0, min(1.0, float(raw_result.get("trajectory_score", trajectory_score)) if raw_result else trajectory_score)),
            "subscores": subscores,
            "strengths": _listify_texts(raw_result.get("strengths") if raw_result else heuristic["strengths"])[:4],
            "weaknesses": _listify_texts(raw_result.get("weaknesses") if raw_result else heuristic["weaknesses"])[:5],
            "improvement_suggestions": _listify_texts(
                raw_result.get("improvement_suggestions") if raw_result else heuristic["improvement_suggestions"]
            )[:5],
            "is_valid": bool(raw_result.get("is_valid")) if raw_result is not None else bool(heuristic["is_valid"]),
            "reason": _safe_text(raw_result.get("reason") if raw_result else heuristic["reason"]) or "metric_normalized",
            "trajectory_summary": trajectory_features.get("trajectory_summary", ""),
        }
        normalized["overall_score"] = max(0.0, min(1.0, normalized["overall_score"] - penalties))
        if raw_result is None:
            normalized["final_answer_score"] = max(0.0, min(1.0, final_answer_score))
            normalized["trajectory_score"] = max(0.0, min(1.0, trajectory_score))
        if not normalized["strengths"]:
            normalized["strengths"] = heuristic["strengths"][:3]
        if not normalized["weaknesses"]:
            normalized["weaknesses"] = heuristic["weaknesses"][:4]
        if not normalized["improvement_suggestions"]:
            normalized["improvement_suggestions"] = heuristic["improvement_suggestions"][:4]
        return normalized
