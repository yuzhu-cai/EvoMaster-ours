"""Experiment tracking and best-artifact promotion tool.

The tool intentionally stays small and ReAct-friendly: it records plans and
experiments, preserves the best artifact, and summarizes simple stagnation
signals. It does not schedule experiments or replace the agent's judgment.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ExperimentTrackerToolParams(BaseToolParams):
    """Record ML experiments and maintain the best known benchmark artifact."""

    name: ClassVar[str] = "experiment_tracker"

    action: Literal["record", "best", "plan"] = Field(
        default="record",
        description="Operation to run: record an experiment, inspect best state, or log a branch plan.",
    )
    workspace: str = Field(default="/workspace", description="Writable workspace path.")

    # record fields
    experiment_name: str = Field(default="", description="Short unique name for this experiment.")
    metric: str = Field(default="", description="Validation metric name, e.g. cv_accuracy or cv_log_loss.")
    score: float | None = Field(default=None, description="Validation score for this experiment.")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize",
        description="Whether larger or smaller scores are better for the metric.",
    )
    submission_path: str = Field(default="", description="Optional path to the submission/artifact.")
    code_path: str = Field(default="", description="Optional path to the reproducible solution script.")
    command: str = Field(default="", description="Command used to reproduce the experiment.")
    notes: str = Field(default="", description="Hypothesis, result, failure, or next idea.")
    round_index: int | None = Field(default=None, description="Optional iteration round number.")
    force_best: bool = Field(default=False, description="Promote even when not numerically better.")
    branch_name: str = Field(default="", description="Stable branch/candidate name for this experiment.")
    model_family: str = Field(default="", description="Broad model family, e.g. transformer, cnn, lightgbm.")
    validation_reliability: str = Field(
        default="",
        description="Validation strength, e.g. single_split, kfold, repeated_cv, seed_ensemble.",
    )

    # plan fields. Scores/costs/risks are optional hints, not a controller.
    candidates: list[str] = Field(default_factory=list, description="Candidate branches to consider.")
    candidate_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Optional expected score/gain estimate for each candidate.",
    )
    candidate_costs: dict[str, float] = Field(
        default_factory=dict,
        description="Optional rough runtime/complexity cost for each candidate.",
    )
    candidate_risks: dict[str, float] = Field(
        default_factory=dict,
        description="Optional overfit/failure risk estimate for each candidate.",
    )
    candidate_families: dict[str, str] = Field(
        default_factory=dict,
        description="Optional broad model family for each candidate.",
    )
    selected_candidate: str = Field(default="", description="Branch chosen by the agent, if already decided.")
    rationale: str = Field(default="", description="Why this branch or plan makes sense.")


class ExperimentTrackerTool(BaseTool):
    """Persistent experiment memory for a single ReAct agent."""

    name: ClassVar[str] = "experiment_tracker"
    params_class: ClassVar[type[BaseToolParams]] = ExperimentTrackerToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:  # noqa: BLE001
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, ExperimentTrackerToolParams)
        if params.action == "plan":
            return self._plan(session, params)
        if params.action == "best":
            return self._best(session, params)
        return self._record(session, params)

    def _plan(self, session, params: ExperimentTrackerToolParams) -> tuple[str, dict[str, Any]]:
        if not params.candidates and not params.notes:
            return "experiment_tracker(plan) requires candidates or notes.", {"ok": False, "error": "missing_plan"}
        if params.selected_candidate and params.candidates and params.selected_candidate not in params.candidates:
            return (
                "experiment_tracker(plan) selected_candidate must be one of candidates.",
                {
                    "ok": False,
                    "error": "selected_candidate_not_in_candidates",
                    "selected_candidate": params.selected_candidate,
                    "candidates": params.candidates,
                },
            )

        paths = self._paths(params.workspace)
        self._ensure_dirs(session, params.workspace)
        now = datetime.now(timezone.utc).isoformat()
        search_state = self._read_json(session, paths["search_state"])
        guidance = self._exploration_guidance(search_state)
        plan = {
            "timestamp": now,
            "round_index": params.round_index,
            "candidates": params.candidates,
            "candidate_scores": params.candidate_scores,
            "candidate_costs": params.candidate_costs,
            "candidate_risks": params.candidate_risks,
            "candidate_families": params.candidate_families,
            "candidate_summaries": self._candidate_summaries(params, search_state),
            "selected_candidate": params.selected_candidate,
            "rationale": params.rationale,
            "notes": params.notes,
            "guidance": guidance,
            "estimate_warnings": self._candidate_estimate_warnings(params),
        }
        self._append_jsonl(session, paths["backlog_jsonl"], plan)
        self._append_backlog(session, paths["backlog_md"], plan)

        selected = params.selected_candidate or "(agent has not selected yet)"
        observation = f"Recorded branch plan with {len(params.candidates)} candidates. selected={selected}"
        if guidance.get("consider_pivot"):
            observation += " Guidance: " + "; ".join(guidance.get("reasons", [])[:2])
        if plan["estimate_warnings"]:
            observation += " Estimate notes: " + "; ".join(plan["estimate_warnings"][:2])
        return observation, {"ok": True, **plan}

    def _record(self, session, params: ExperimentTrackerToolParams) -> tuple[str, dict[str, Any]]:
        if params.score is None:
            return "experiment_tracker(record) requires a numeric score.", {"ok": False, "error": "missing_score"}
        if not params.metric.strip():
            return "experiment_tracker(record) requires a metric name.", {"ok": False, "error": "missing_metric"}

        paths = self._paths(params.workspace)
        self._ensure_dirs(session, params.workspace)
        now = datetime.now(timezone.utc).isoformat()

        current_best = self._read_json(session, paths["best_meta"])
        branch_info = self._record_branch_info(params, paths, session)
        comparison = self._compare(params, current_best)
        score_would_improve_best = bool(comparison["is_best"])
        promote = score_would_improve_best

        record: dict[str, Any] = {
            "timestamp": now,
            "experiment_name": params.experiment_name or f"experiment_{now}",
            "metric": params.metric,
            "score": params.score,
            "direction": params.direction,
            "round_index": params.round_index,
            "branch_name": branch_info["branch_name"],
            "model_family": branch_info["model_family"],
            "validation_reliability": params.validation_reliability,
            "submission_path": params.submission_path,
            "code_path": params.code_path,
            "command": params.command,
            "notes": params.notes,
            "score_would_improve_best": score_would_improve_best,
            "promoted_to_best": promote,
            "comparison": comparison,
        }

        if score_would_improve_best:
            copy_info = self._promote_artifacts(session, params, paths)
            record["copy_info"] = copy_info
            promotion_blocked = self._promotion_block_reason(params, copy_info)
            if promotion_blocked:
                promote = False
                record["promoted_to_best"] = False
                record["promotion_blocked"] = promotion_blocked

        if promote:
            best_meta = {
                **record,
                "best_submission_path": paths["best_submission"],
                "best_solution_path": paths["best_solution"],
                "previous_best": current_best or None,
            }
            session.write_file(paths["best_meta"], json.dumps(best_meta, indent=2, ensure_ascii=False) + "\n")

        self._append_jsonl(session, paths["experiments_jsonl"], record)
        self._append_ledger(session, paths["iteration_ledger"], record)
        self._update_search_state(session, params, paths, record)

        if promote:
            status = "promoted new best"
        elif score_would_improve_best and record.get("promotion_blocked"):
            status = f"recorded score improvement but did not promote best artifact ({record['promotion_blocked']})"
        else:
            status = "recorded, best unchanged"
        observation = (
            f"Experiment '{record['experiment_name']}' {status}. "
            f"score={params.score} metric={params.metric} direction={params.direction}. "
            f"best={self._best_summary(session, paths)}"
        )
        guidance = self._exploration_guidance(self._read_json(session, paths["search_state"]))
        if guidance.get("consider_pivot"):
            observation += " Guidance: " + "; ".join(guidance.get("reasons", [])[:2])
        return observation, {"ok": True, **record}

    def _best(self, session, params: ExperimentTrackerToolParams) -> tuple[str, dict[str, Any]]:
        paths = self._paths(params.workspace)
        best_meta = self._read_json(session, paths["best_meta"])
        history_preview = self._tail_lines(session, paths["experiments_jsonl"], limit=5)
        backlog_preview = self._tail_lines(session, paths["backlog_jsonl"], limit=3)
        search_state = self._read_json(session, paths["search_state"])
        search_preview = self._search_state_preview(search_state)

        best_text = "No best experiment recorded yet. Call experiment_tracker(action='record') after validation."
        if best_meta:
            best_text = "Current best experiment:\n" + json.dumps(best_meta, indent=2, ensure_ascii=False)[-8000:]
        return (
            best_text
            + "\n\nRecent experiments:\n"
            + "\n".join(history_preview)
            + "\n\nRecent branch plans:\n"
            + "\n".join(backlog_preview)
            + "\n\nBranch memory:\n"
            + search_preview,
            {
                "ok": True,
                "best": best_meta or None,
                "recent": history_preview,
                "backlog": backlog_preview,
                "search_state": search_state,
                "guidance": self._exploration_guidance(search_state),
            },
        )

    def _paths(self, workspace: str) -> dict[str, str]:
        root = workspace.rstrip("/") or "/workspace"
        return {
            "logs_dir": f"{root}/logs",
            "artifacts_dir": f"{root}/artifacts",
            "experiments_jsonl": f"{root}/logs/experiments.jsonl",
            "backlog_jsonl": f"{root}/logs/experiment_backlog.jsonl",
            "backlog_md": f"{root}/logs/experiment_backlog.md",
            "search_state": f"{root}/logs/branch_search_state.json",
            "iteration_ledger": f"{root}/logs/iteration_ledger.md",
            "best_meta": f"{root}/artifacts/best_meta.json",
            "best_submission": f"{root}/artifacts/best_submission.csv",
            "best_solution": f"{root}/artifacts/best_solution.py",
        }

    def _ensure_dirs(self, session, workspace: str) -> None:
        root = workspace.rstrip("/") or "/workspace"
        session.exec_bash(
            f"mkdir -p {shlex.quote(root + '/logs')} {shlex.quote(root + '/artifacts')}",
            timeout=30,
        )

    def _read_json(self, session, path: str) -> dict[str, Any]:
        if not session.is_file(path):
            return {}
        try:
            payload = json.loads(session.read_file(path))
        except Exception:  # noqa: BLE001
            return {}
        return payload if isinstance(payload, dict) else {}

    def _append_jsonl(self, session, path: str, record: dict[str, Any]) -> None:
        old = session.read_file(path) if session.is_file(path) else ""
        session.write_file(path, old + json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _append_ledger(self, session, path: str, record: dict[str, Any]) -> None:
        old = session.read_file(path) if session.is_file(path) else "# Iteration ledger\n\n"
        comparison = record.get("comparison", {}) if isinstance(record.get("comparison"), dict) else {}
        block = [
            f"## {record['timestamp']} - {record['experiment_name']}",
            "",
            f"- round: {record.get('round_index')}",
            f"- branch: {record.get('branch_name') or '(none)'}",
            f"- model_family: {record.get('model_family') or '(unknown)'}",
            f"- validation_reliability: {record.get('validation_reliability') or '(not recorded)'}",
            f"- metric: {record['metric']} ({record['direction']})",
            f"- score: {record['score']}",
            f"- improved_best: {'YES' if record.get('promoted_to_best') else 'NO'}",
            f"- score_would_improve_best: {record.get('score_would_improve_best')}",
            f"- promotion_blocked: {record.get('promotion_blocked') or '(none)'}",
            f"- delta_vs_previous_best: {comparison.get('delta')}",
            f"- submission_path: {record.get('submission_path') or '(none)'}",
            f"- code_path: {record.get('code_path') or '(none)'}",
            f"- command: {record.get('command') or '(not recorded)'}",
            f"- notes: {record.get('notes') or '(none)'}",
            "",
        ]
        session.write_file(path, old.rstrip() + "\n\n" + "\n".join(block))

    def _append_backlog(self, session, path: str, plan: dict[str, Any]) -> None:
        old = session.read_file(path) if session.is_file(path) else "# Experiment branch backlog\n\n"
        lines = [f"## {plan['timestamp']} - round {plan.get('round_index')}", ""]
        candidates = plan.get("candidates") or []
        if candidates:
            lines.append("Candidates:")
            summaries = {item["candidate"]: item for item in plan.get("candidate_summaries", [])}
            for idx, candidate in enumerate(candidates, start=1):
                summary = summaries.get(candidate, {})
                lines.append(
                    f"{idx}. {candidate} "
                    f"[family={summary.get('model_family') or 'unknown'}, "
                    f"visits={summary.get('visits', 0)}, no_improve={summary.get('no_improve_visits', 0)}]"
                )
            lines.append("")
        guidance = plan.get("guidance") or {}
        if guidance:
            lines.append(f"Guidance: consider_pivot={guidance.get('consider_pivot')}")
            for reason in guidance.get("reasons", [])[:4]:
                lines.append(f"- {reason}")
            lines.append("")
        warnings = plan.get("estimate_warnings") or []
        if warnings:
            lines.append("Estimate notes:")
            lines.extend(f"- {warning}" for warning in warnings)
            lines.append("")
        lines.extend(
            [
                f"Selected: {plan.get('selected_candidate') or '(not selected)'}",
                f"Rationale: {plan.get('rationale') or '(none)'}",
                f"Notes: {plan.get('notes') or '(none)'}",
                "",
            ]
        )
        session.write_file(path, old.rstrip() + "\n\n" + "\n".join(lines))

    def _candidate_summaries(
        self,
        params: ExperimentTrackerToolParams,
        search_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        branches = search_state.get("branches", {}) if isinstance(search_state.get("branches"), dict) else {}
        families = search_state.get("families", {}) if isinstance(search_state.get("families"), dict) else {}
        rows = []
        for candidate in params.candidates:
            family_name = str(params.candidate_families.get(candidate) or self._infer_family(candidate)).strip()
            branch = branches.get(candidate, {}) if isinstance(branches.get(candidate), dict) else {}
            family = families.get(family_name, {}) if isinstance(families.get(family_name), dict) else {}
            rows.append(
                {
                    "candidate": candidate,
                    "model_family": family_name,
                    "score_hint": params.candidate_scores.get(candidate),
                    "cost_hint": params.candidate_costs.get(candidate),
                    "risk_hint": params.candidate_risks.get(candidate),
                    "visits": int(branch.get("visits", 0) or 0),
                    "no_improve_visits": int(branch.get("no_improve_visits", 0) or 0),
                    "improved_count": int(branch.get("improved_count", 0) or 0),
                    "family_visits": int(family.get("visits", 0) or 0),
                }
            )
        return rows

    def _candidate_estimate_warnings(self, params: ExperimentTrackerToolParams) -> list[str]:
        warnings = []
        for candidate in params.candidates:
            missing = []
            if candidate not in params.candidate_scores:
                missing.append("score")
            if candidate not in params.candidate_costs:
                missing.append("cost")
            if candidate not in params.candidate_risks:
                missing.append("risk")
            if missing:
                warnings.append(f"{candidate}: missing optional {', '.join(missing)} estimate(s)")
        return warnings

    def _record_branch_info(self, params: ExperimentTrackerToolParams, paths: dict[str, str], session) -> dict[str, str]:
        last_plan = self._read_last_jsonl(session, paths["backlog_jsonl"])
        branch_name = params.branch_name.strip()
        if not branch_name and last_plan:
            branch_name = str(last_plan.get("selected_candidate") or "").strip()
        if not branch_name:
            branch_name = "unplanned"

        family = params.model_family.strip()
        if not family and last_plan:
            candidate_families = last_plan.get("candidate_families") or {}
            if isinstance(candidate_families, dict):
                family = str(candidate_families.get(branch_name) or "").strip()
        if not family:
            family = self._infer_family(branch_name)
        return {"branch_name": branch_name, "model_family": family}

    def _update_search_state(self, session, params: ExperimentTrackerToolParams, paths: dict[str, str], record: dict[str, Any]) -> None:
        branch_name = str(record.get("branch_name") or params.branch_name or "unplanned").strip() or "unplanned"
        model_family = str(record.get("model_family") or params.model_family or self._infer_family(branch_name)).strip()
        model_family = model_family or "unknown"
        improved = bool(record.get("promoted_to_best"))
        reward = self._record_reward(params, record)

        state = self._read_json(session, paths["search_state"])
        if not state:
            state = {"total_visits": 0, "branches": {}, "families": {}}
        branches = state.setdefault("branches", {})
        if not isinstance(branches, dict):
            branches = {}
            state["branches"] = branches
        families = state.setdefault("families", {})
        if not isinstance(families, dict):
            families = {}
            state["families"] = families

        branch = branches.setdefault(
            branch_name,
            {"visits": 0, "total_reward": 0.0, "best_reward": None, "improved_count": 0, "no_improve_visits": 0, "experiments": []},
        )
        branch["visits"] = int(branch.get("visits", 0) or 0) + 1
        branch["total_reward"] = float(branch.get("total_reward", 0.0) or 0.0) + reward
        branch["mean_reward"] = branch["total_reward"] / max(branch["visits"], 1)
        branch["last_reward"] = reward
        branch["last_score"] = params.score
        branch["last_metric"] = params.metric
        branch["last_experiment_name"] = record.get("experiment_name")
        branch["last_round_index"] = params.round_index
        branch["model_family"] = model_family
        branch["no_improve_visits"] = 0 if improved else int(branch.get("no_improve_visits", 0) or 0) + 1
        if improved:
            branch["improved_count"] = int(branch.get("improved_count", 0) or 0) + 1
            branch["last_improved_experiment"] = record.get("experiment_name")
        best_reward = branch.get("best_reward")
        if best_reward is None or reward > float(best_reward):
            branch["best_reward"] = reward
        experiments = branch.setdefault("experiments", [])
        if isinstance(experiments, list):
            experiments.append(
                {
                    "experiment_name": record.get("experiment_name"),
                    "score": params.score,
                    "metric": params.metric,
                    "reward": reward,
                    "promoted_to_best": improved,
                    "model_family": model_family,
                    "validation_reliability": params.validation_reliability,
                    "round_index": params.round_index,
                }
            )
            branch["experiments"] = experiments[-20:]

        family = families.setdefault(model_family, {"visits": 0, "improved_count": 0, "no_improve_visits": 0, "branches": []})
        family["visits"] = int(family.get("visits", 0) or 0) + 1
        family["no_improve_visits"] = 0 if improved else int(family.get("no_improve_visits", 0) or 0) + 1
        if improved:
            family["improved_count"] = int(family.get("improved_count", 0) or 0) + 1
        family_branches = family.setdefault("branches", [])
        if isinstance(family_branches, list) and branch_name not in family_branches:
            family_branches.append(branch_name)
            family["branches"] = family_branches[-20:]

        state["total_visits"] = int(state.get("total_visits", 0) or 0) + 1
        state["consecutive_non_improving_records"] = 0 if improved else int(state.get("consecutive_non_improving_records", 0) or 0) + 1
        state["last_selected_candidate"] = branch_name
        state["last_selected_family"] = model_family
        state["last_experiment_name"] = record.get("experiment_name")
        if improved:
            state["last_improved_candidate"] = branch_name
            state["last_improved_family"] = model_family
            state["last_improved_experiment"] = record.get("experiment_name")
        recent = state.setdefault("recent_branches", [])
        if isinstance(recent, list):
            recent.append(branch_name)
            state["recent_branches"] = recent[-10:]
        state["updated_at"] = record.get("timestamp")
        session.write_file(paths["search_state"], json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    def _record_reward(self, params: ExperimentTrackerToolParams, record: dict[str, Any]) -> float:
        comparison = record.get("comparison", {}) if isinstance(record.get("comparison"), dict) else {}
        delta = comparison.get("delta")
        if delta is None:
            return 0.0
        try:
            value = float(delta)
        except (TypeError, ValueError):
            return 0.0
        return value if params.direction == "maximize" else -value

    def _exploration_guidance(self, state: dict[str, Any]) -> dict[str, Any]:
        branches = state.get("branches", {}) if isinstance(state.get("branches"), dict) else {}
        consecutive = int(state.get("consecutive_non_improving_records", 0) or 0)
        stale_branches = []
        for name, stats in branches.items():
            if not isinstance(stats, dict):
                continue
            visits = int(stats.get("visits", 0) or 0)
            no_improve = int(stats.get("no_improve_visits", 0) or 0)
            if visits >= 2 and no_improve >= 2:
                stale_branches.append(name)
        reasons = []
        if consecutive >= 3:
            reasons.append(f"recent {consecutive} records did not improve; consider a different model family or validation redesign")
        elif consecutive >= 2:
            reasons.append(f"recent {consecutive} records did not improve; avoid another small local tweak unless justified")
        if stale_branches:
            reasons.append("stale branch(es): " + ", ".join(stale_branches[:5]))
        if not reasons:
            reasons.append("no strong stagnation signal; continue with the most justified experiment")
        return {
            "consider_pivot": consecutive >= 2 or bool(stale_branches),
            "consecutive_non_improving_records": consecutive,
            "stale_branches": stale_branches,
            "last_selected_candidate": state.get("last_selected_candidate"),
            "last_selected_family": state.get("last_selected_family"),
            "reasons": reasons,
        }

    def _search_state_preview(self, state: dict[str, Any]) -> str:
        if not state:
            return "(No branch memory recorded yet.)"
        guidance = self._exploration_guidance(state)
        lines = [
            f"total_visits={state.get('total_visits', 0)}",
            f"consecutive_non_improving_records={guidance.get('consecutive_non_improving_records', 0)}",
            f"consider_pivot={guidance.get('consider_pivot', False)}",
        ]
        for reason in guidance.get("reasons", [])[:4]:
            lines.append(f"guidance: {reason}")

        branches = state.get("branches", {}) if isinstance(state.get("branches"), dict) else {}
        rows = []
        for name, stats in branches.items():
            if isinstance(stats, dict):
                rows.append(
                    (
                        name,
                        int(stats.get("visits", 0) or 0),
                        int(stats.get("improved_count", 0) or 0),
                        int(stats.get("no_improve_visits", 0) or 0),
                        stats.get("model_family") or "unknown",
                        stats.get("last_experiment_name"),
                    )
                )
        rows.sort(key=lambda item: (item[2], -item[3], item[1]), reverse=True)
        for name, visits, improved_count, no_improve, family, last_exp in rows[:10]:
            lines.append(
                f"- {name}: family={family}, visits={visits}, improved={improved_count}, "
                f"no_improve={no_improve}, last_experiment={last_exp}"
            )
        return "\n".join(lines)

    def _compare(self, params: ExperimentTrackerToolParams, current_best: dict[str, Any]) -> dict[str, Any]:
        if params.force_best:
            return {"is_best": True, "reason": "force_best", "previous_score": current_best.get("score")}
        if not current_best or current_best.get("score") is None:
            return {"is_best": True, "reason": "first_recorded_score", "previous_score": None}

        previous_score = float(current_best["score"])
        current_score = float(params.score)
        same_metric = str(current_best.get("metric", "")).strip().lower() == params.metric.strip().lower()
        same_direction = str(current_best.get("direction", params.direction)) == params.direction
        if not same_metric or not same_direction:
            return {
                "is_best": False,
                "reason": "metric_or_direction_mismatch",
                "previous_score": previous_score,
                "previous_metric": current_best.get("metric"),
                "previous_direction": current_best.get("direction"),
            }

        delta = current_score - previous_score
        is_best = delta > 1e-12 if params.direction == "maximize" else delta < -1e-12
        return {"is_best": is_best, "reason": "numeric_comparison", "previous_score": previous_score, "delta": delta}

    def _promote_artifacts(self, session, params: ExperimentTrackerToolParams, paths: dict[str, str]) -> dict[str, Any]:
        info: dict[str, Any] = {"submission_copied": False, "code_copied": False}
        if params.submission_path:
            info["submission_copied"] = self._copy_file(session, params.submission_path, paths["best_submission"])
        if params.code_path:
            info["code_copied"] = self._copy_file(session, params.code_path, paths["best_solution"])
        return info

    def _promotion_block_reason(self, params: ExperimentTrackerToolParams, copy_info: dict[str, Any]) -> str:
        if not params.submission_path.strip():
            return "missing_submission_path"
        if not copy_info.get("submission_copied"):
            return "submission_copy_failed"
        return ""

    def _copy_file(self, session, src: str, dst: str) -> bool:
        if not session.is_file(src):
            return False
        result = session.exec_bash(f"cp {shlex.quote(src)} {shlex.quote(dst)}", timeout=60)
        return int(result.get("exit_code", 1) or 0) == 0 and session.is_file(dst)

    def _read_last_jsonl(self, session, path: str) -> dict[str, Any]:
        if not session.is_file(path):
            return {}
        for line in reversed(session.read_file(path).splitlines()):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            return payload if isinstance(payload, dict) else {}
        return {}

    def _tail_lines(self, session, path: str, *, limit: int) -> list[str]:
        if not session.is_file(path):
            return []
        text = session.read_file(path)
        return [line for line in text.splitlines() if line.strip()][-limit:]

    def _best_summary(self, session, paths: dict[str, str]) -> str:
        best_meta = self._read_json(session, paths["best_meta"])
        if not best_meta:
            return "(none)"
        return (
            f"{best_meta.get('experiment_name')} "
            f"{best_meta.get('metric')}={best_meta.get('score')} "
            f"({best_meta.get('direction')})"
        )

    def _infer_family(self, name: str) -> str:
        text = name.lower()
        if any(token in text for token in ("roberta", "deberta", "bert", "transformer", "huggingface", "hf_")):
            return "transformer"
        if any(token in text for token in ("resnet", "efficientnet", "cnn", "timm", "vit")):
            return "cnn"
        if "catboost" in text:
            return "catboost"
        if "lightgbm" in text or "lgbm" in text:
            return "lightgbm"
        if "xgboost" in text or "xgb" in text:
            return "xgboost"
        if any(token in text for token in ("tfidf", "logreg", "linear", "svm", "nb", "sparse")):
            return "linear_sparse"
        if any(token in text for token in ("ensemble", "blend", "stack")):
            return "ensemble"
        if any(token in text for token in ("feature", "clean", "preprocess")):
            return "feature_engineering"
        return "unknown"
