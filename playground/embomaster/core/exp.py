"""EmboMaster experiment workflow."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from .services import K8SExperimentRunner
from .monitoring import EmboMasterMonitorWriter
from .utils import WorkspaceCodebaseInfo, cleanup_eval_result, prepare_workspace_codebase


class EmboMasterExp(BaseExp):
    """Iterative coding + optional K8S experiment workflow."""

    def __init__(
        self,
        coding_agent,
        feedback_agent,
        config,
        k8s_runner: K8SExperimentRunner | None = None,
    ):
        super().__init__(coding_agent, config)
        self.coding_agent = coding_agent
        self.feedback_agent = feedback_agent
        self.k8s_runner = k8s_runner
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return "EmboMaster"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images: list[str] | None = None,
    ) -> dict:
        self.logger.info("Starting EmboMaster experiment")
        settings = self._get_experiment_settings()
        k8s_cfg = self._get_k8s_config()
        workspace_cfg = self._get_workspace_isolation_config()
        parent_selection_cfg = self._get_parent_selection_config()
        result_validation_cfg = self._get_result_validation_config()
        max_rounds = int(settings.get("steps", 1))
        time_limit_seconds = int(settings.get("time_limit_seconds", 0))

        round_results: list[dict[str, Any]] = []
        best_round: dict[str, Any] | None = None
        best_metric: float | None = None
        previous_workspace_id: str | None = None
        previous_feedback = "N/A"
        start_time = time.monotonic()
        stopped_reason = "completed_all_rounds"
        monitor_writer = self._create_monitor_writer(task_id)

        for round_index in range(1, max_rounds + 1):
            if time_limit_seconds > 0 and (time.monotonic() - start_time) >= time_limit_seconds:
                stopped_reason = "time_limit_reached"
                break

            self.logger.info("=== Round %s/%s ===", round_index, max_rounds)
            BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=round_index)

            parent_decision = self._choose_parent_workspace(
                workspace_cfg=workspace_cfg,
                parent_selection_cfg=parent_selection_cfg,
                previous_workspace_id=previous_workspace_id,
                best_round=best_round,
                last_round=(round_results[-1] if round_results else None),
            )
            round_result = self._run_one_round(
                task_description=task_description,
                task_id=task_id,
                round_index=round_index,
                max_rounds=max_rounds,
                previous_feedback=previous_feedback,
                best_metric=best_metric,
                best_round=best_round,
                k8s_cfg=k8s_cfg,
                metric_cfg=settings,
                workspace_cfg=workspace_cfg,
                parent_workspace_id=parent_decision.get("workspace_id"),
                parent_decision=parent_decision,
                round_history=round_results,
                parent_selection_cfg=parent_selection_cfg,
                result_validation_cfg=result_validation_cfg,
            )
            round_results.append(round_result)
            previous_workspace_id = round_result.get("workspace_id")

            metric_value = round_result.get("metric_value")
            if self._is_better_metric(best_metric, metric_value, settings):
                best_metric = metric_value
                best_round = round_result

            feedback_text = round_result.get("feedback", "")
            if feedback_text:
                previous_feedback = feedback_text

            if monitor_writer is not None:
                try:
                    monitor_writer.record_round(
                        round_result=round_result,
                        best_metric=best_metric,
                        best_round_index=(
                            best_round.get("round_index") if isinstance(best_round, dict) else None
                        ),
                        total_rounds_so_far=len(round_results),
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to persist monitor round snapshot for round %s: %s",
                        round_index,
                        exc,
                    )

            if settings.get("stop_on_job_failed", False):
                if round_result.get("k8s_status") in {"failed", "timeout"}:
                    stopped_reason = f"stop_on_{round_result.get('k8s_status')}"
                    break

        valid_rounds = [r for r in round_results if bool(r.get("result_valid", True))]
        invalid_rounds = [r for r in round_results if not bool(r.get("result_valid", True))]

        status = "completed"
        if not round_results:
            status = "failed"
        elif invalid_rounds and not valid_rounds:
            status = "failed"
        elif invalid_rounds:
            status = "completed_with_invalid_rounds"

        result: dict[str, Any] = {
            "status": status,
            "task_id": task_id,
            "total_rounds": len(round_results),
            "stopped_reason": stopped_reason,
            "rounds": round_results,
            "best_metric": best_metric,
            "best_round_index": best_round.get("round_index") if best_round else None,
            "best_round": best_round,
            "invalid_round_count": len(invalid_rounds),
        }

        total_steps = sum(int(r.get("steps", 0)) for r in round_results)
        trajectories = [r.get("trajectory") for r in round_results if r.get("trajectory") is not None]
        self.results.append(
            {
                "task_id": task_id,
                "status": result["status"],
                "steps": total_steps,
                "trajectory": trajectories,
            }
        )
        if monitor_writer is not None:
            try:
                monitor_writer.record_final_result(
                    result=result,
                    task_description=task_description,
                )
            except Exception as exc:
                self.logger.warning("Failed to persist final monitor snapshot: %s", exc)
        return result

    def _run_one_round(
        self,
        task_description: str,
        task_id: str,
        round_index: int,
        max_rounds: int,
        previous_feedback: str,
        best_metric: float | None,
        best_round: dict[str, Any] | None,
        k8s_cfg: dict[str, Any],
        metric_cfg: dict[str, Any],
        workspace_cfg: dict[str, Any],
        parent_workspace_id: str | None,
        parent_decision: dict[str, Any],
        round_history: list[dict[str, Any]],
        parent_selection_cfg: dict[str, Any],
        result_validation_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        workspace_context = self._build_round_workspace_context(
            task_id=task_id,
            round_index=round_index,
            workspace_cfg=workspace_cfg,
            parent_workspace_id=parent_workspace_id,
        )

        coding_task = TaskInstance(
            task_id=f"{task_id}_coding_r{round_index}",
            task_type="coding",
            description=task_description,
            input_data={},
        )

        original_kwargs = self.coding_agent._prompt_format_kwargs.copy()
        debug_container_codebase_path = "N/A"
        if isinstance(k8s_cfg, dict):
            debug_container_codebase_path = str(
                k8s_cfg.get("codebase_mount_path")
                or k8s_cfg.get("container_workspace")
                or "N/A"
            )
        self.coding_agent._prompt_format_kwargs.update(
            {
                "round_index": round_index,
                "max_rounds": max_rounds,
                "feedback_for_next_round": previous_feedback,
                "best_metric": "None" if best_metric is None else str(best_metric),
                "workspace_id": workspace_context.get("workspace_id", "N/A"),
                "parent_workspace_id": workspace_context.get("parent_workspace_id", "N/A"),
                "workspace_codebase_path": workspace_context.get("workspace_codebase_path", "N/A"),
                "debug_container_codebase_path": debug_container_codebase_path,
                "workspace_source_type": workspace_context.get("source_type", "N/A"),
                "workspace_large_dirs_count": workspace_context.get("large_dirs_count", 0),
            }
        )
        session = getattr(self.coding_agent, "session", None)
        original_workspace_path = None
        workspace_codebase_path = workspace_context.get("workspace_codebase_path")
        debug_pod_prepare_result: dict[str, Any] | None = None
        debug_pod_cleanup_before_submit: dict[str, Any] | None = None
        debug_pod_cleanup_final: dict[str, Any] | None = None
        trajectory = None
        if session and workspace_codebase_path:
            original_workspace_path = getattr(session.config, "workspace_path", None)
            session.config.workspace_path = str(workspace_codebase_path)
        try:
            if self.k8s_runner and workspace_codebase_path:
                try:
                    debug_pod_prepare_result = self.k8s_runner.prepare_round_debug_pod(
                        workspace_codebase_path
                    )
                except Exception as exc:
                    debug_pod_prepare_result = {
                        "status": "prepare_failed",
                        "error": str(exc),
                    }
                    self.logger.warning(
                        "Failed to prepare debug pod for round %s: %s",
                        round_index,
                        exc,
                    )
            trajectory = self.coding_agent.run(coding_task)
        finally:
            if trajectory is None and self.k8s_runner and workspace_codebase_path:
                try:
                    self.k8s_runner.cleanup_debug_pod_for_workspace(
                        workspace_codebase_path,
                        wait=False,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed debug pod cleanup after coding exception in round %s: %s",
                        round_index,
                        exc,
                    )
            if session and original_workspace_path is not None:
                session.config.workspace_path = str(original_workspace_path)
            self.coding_agent._prompt_format_kwargs = original_kwargs

        coding_result = self._extract_agent_response(trajectory)
        round_result: dict[str, Any] = {
            "round_index": round_index,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
            "coding_result": coding_result,
            "trajectory": trajectory,
            "k8s_status": "skipped",
            "metric_value": None,
            "feedback": "",
            "workspace_id": workspace_context.get("workspace_id"),
            "parent_workspace_id": workspace_context.get("parent_workspace_id"),
            "parent_choice_used": parent_decision.get("choice", "previous"),
            "parent_choice_reason": parent_decision.get("reason", ""),
            "workspace_codebase_path": workspace_context.get("workspace_codebase_path"),
            "workspace_source_type": workspace_context.get("source_type"),
            "workspace_large_dirs_count": workspace_context.get("large_dirs_count", 0),
            "workspace_large_dirs": workspace_context.get("workspace_large_dirs", []),
            "submission_dir": workspace_context.get("submission_dir"),
            "session_dir": workspace_context.get("session_dir"),
            "metric_valid": False,
            "artifacts_summary": {},
            "parent_recommendation": None,
            "result_valid": True,
            "validation_errors": [],
        }
        if debug_pod_prepare_result is not None:
            round_result["debug_pod_prepare"] = debug_pod_prepare_result

        try:
            if self.k8s_runner and k8s_cfg.get("enabled", False):
                manifest_path = str(k8s_cfg.get("manifest_path", "")).strip()
                job_prefix = str(k8s_cfg.get("job_name_prefix", "embomaster-job")).strip()
                if manifest_path:
                    suffix = datetime.utcnow().strftime("%H%M%S%f")
                    workspace_tag = (workspace_context.get("workspace_id", "") or "")[:8]
                    if workspace_tag:
                        job_name = f"{job_prefix}-r{round_index}-{workspace_tag}-{suffix}".lower()
                    else:
                        job_name = f"{job_prefix}-r{round_index}-{suffix}".lower()
                    job_name = re.sub(r"[^a-z0-9-]+", "-", job_name).strip("-")[:63]
                    env_map = k8s_cfg.get("manifest_env", {})
                    if not isinstance(env_map, dict):
                        env_map = {}
                    if workspace_codebase_path:
                        debug_pod_cleanup_before_submit = self.k8s_runner.cleanup_debug_pod_for_workspace(
                            workspace_codebase_path,
                            wait=True,
                        )
                        round_result["debug_pod_cleanup_before_submit"] = (
                            debug_pod_cleanup_before_submit
                        )
                        cleanup_exit_code = int(
                            debug_pod_cleanup_before_submit.get("exit_code", 0)
                        )
                        cleanup_status = str(
                            debug_pod_cleanup_before_submit.get("status", "")
                        ).lower()
                        if cleanup_exit_code != 0 and cleanup_status not in {
                            "not_found",
                            "disabled",
                        }:
                            raise RuntimeError(
                                "failed to cleanup debug pod before job submission: "
                                + (
                                    debug_pod_cleanup_before_submit.get("stderr", "")
                                    or debug_pod_cleanup_before_submit.get("output", "")
                                    or str(debug_pod_cleanup_before_submit)
                                )
                            )
                    self.logger.info("Submitting K8S job for round %s: %s", round_index, job_name)
                    k8s_result = self.k8s_runner.run(
                        manifest_path=manifest_path,
                        job_name=job_name,
                        manifest_env={str(k): str(v) for k, v in env_map.items()},
                        workspace_context=workspace_context,
                    )
                    round_result["k8s_result"] = k8s_result
                    round_result["k8s_status"] = k8s_result.get("wait", {}).get("status", "unknown")
                    metric_value, metric_source = self._extract_metric_from_k8s(k8s_result, metric_cfg)
                    round_result["metric_value"] = metric_value
                    round_result["metric_source"] = metric_source
                else:
                    self.logger.warning("k8s_runner.enabled=true but manifest_path is empty")
                    round_result["k8s_status"] = "skipped_missing_manifest"
        finally:
            if self.k8s_runner and workspace_codebase_path:
                try:
                    debug_pod_cleanup_final = self.k8s_runner.cleanup_debug_pod_for_workspace(
                        workspace_codebase_path,
                        wait=False,
                    )
                except Exception as exc:
                    debug_pod_cleanup_final = {
                        "status": "cleanup_failed",
                        "error": str(exc),
                    }
                    self.logger.warning(
                        "Failed final debug pod cleanup for round %s: %s",
                        round_index,
                        exc,
                    )
                round_result["debug_pod_cleanup_final"] = debug_pod_cleanup_final

        round_result["metric_valid"] = round_result.get("metric_value") is not None
        round_result["artifacts_summary"] = self._collect_artifacts_summary(
            round_result=round_result,
            k8s_cfg=k8s_cfg,
        )
        self._apply_result_validation(
            round_result=round_result,
            result_validation_cfg=result_validation_cfg,
        )

        if self.feedback_agent:
            feedback_result = self._run_feedback_round(
                task_description=task_description,
                task_id=task_id,
                round_index=round_index,
                round_result=round_result,
                best_round=best_round,
                metric_cfg=metric_cfg,
                round_history=round_history,
                parent_selection_cfg=parent_selection_cfg,
            )
            round_result["feedback"] = feedback_result.get("text", "")
            round_result["parent_recommendation"] = feedback_result.get("parent_recommendation")

        return round_result

    def _run_feedback_round(
        self,
        task_description: str,
        task_id: str,
        round_index: int,
        round_result: dict[str, Any],
        best_round: dict[str, Any] | None,
        metric_cfg: dict[str, Any],
        round_history: list[dict[str, Any]],
        parent_selection_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        logs = ""
        if isinstance(round_result.get("k8s_result"), dict):
            logs = str(round_result["k8s_result"].get("logs", {}).get("output", ""))
        logs_tail = logs[-3000:] if logs else "N/A"
        best_candidate = self._candidate_best_round(best_round, round_result, metric_cfg)
        recent_rounds = list(round_history[-int(parent_selection_cfg.get("recent_rounds", 3)) :])

        task = TaskInstance(
            task_id=f"{task_id}_feedback_r{round_index}",
            task_type="feedback",
            description=task_description,
            input_data={},
        )
        original_kwargs = self.feedback_agent._prompt_format_kwargs.copy()
        self.feedback_agent._prompt_format_kwargs.update(
            {
                "round_index": round_index,
                "k8s_status": round_result.get("k8s_status", "unknown"),
                "metric_value": round_result.get("metric_value"),
                "k8s_log_tail": logs_tail,
                "candidate_last_summary": self._format_round_summary(round_result, label="last"),
                "candidate_best_summary": self._format_round_summary(best_candidate, label="best"),
                "recent_round_summaries": self._format_recent_round_summaries(recent_rounds),
                "parent_decision_constraints": self._format_parent_constraints(
                    parent_selection_cfg
                ),
            }
        )
        try:
            trajectory = self.feedback_agent.run(task)
            feedback_text = self._extract_agent_response(trajectory)
            return {
                "text": feedback_text,
                "parent_recommendation": self._parse_parent_recommendation(feedback_text),
            }
        finally:
            self.feedback_agent._prompt_format_kwargs = original_kwargs

    def _extract_metric_from_k8s(
        self, k8s_result: dict[str, Any], metric_cfg: dict[str, Any]
    ) -> tuple[float | None, str]:
        logs_obj = k8s_result.get("logs", {})
        logs = ""
        if isinstance(logs_obj, dict):
            logs = str(logs_obj.get("output", "") or logs_obj.get("stdout", ""))
        if not logs:
            return None, "no_logs"

        metric_pattern = str(metric_cfg.get("metric_pattern", "")).strip()
        if metric_pattern:
            try:
                matches = re.findall(metric_pattern, logs, flags=re.IGNORECASE | re.MULTILINE)
                if matches:
                    value = self._to_float(matches[-1])
                    if value is not None:
                        return value, f"pattern:{metric_pattern}"
            except re.error:
                self.logger.warning("Invalid metric_pattern regex: %s", metric_pattern)

        fallback_patterns = [
            r"success\s*rate\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"validation\s*score\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"metric\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"\\boxed\{([0-9]*\.?[0-9]+)\}",
        ]
        for pattern in fallback_patterns:
            matches = re.findall(pattern, logs, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                value = self._to_float(matches[-1])
                if value is not None:
                    return value, f"pattern:{pattern}"
        return None, "not_found"

    def _is_better_metric(
        self,
        current_best: float | None,
        candidate: float | None,
        metric_cfg: dict[str, Any],
    ) -> bool:
        if candidate is None:
            return False
        if current_best is None:
            return True
        mode = str(metric_cfg.get("metric_mode", "maximize")).strip().lower()
        if mode == "minimize":
            return candidate < current_best
        return candidate > current_best

    def _to_float(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, tuple):
            for item in value:
                num = self._to_float(item)
                if num is not None:
                    return num
            return None
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def _get_k8s_config(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)
        return cfg_dict.get("k8s_runner", {})

    def _get_experiment_settings(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)
        settings = cfg_dict.get("experiment", {})
        if not isinstance(settings, dict):
            settings = {}
        settings.setdefault("steps", 1)
        settings.setdefault("time_limit_seconds", 0)
        settings.setdefault("metric_mode", "maximize")
        settings.setdefault("metric_pattern", "")
        settings.setdefault("stop_on_job_failed", False)
        return settings

    def _get_workspace_isolation_config(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)

        workspace_cfg = cfg_dict.get("workspace_isolation", {})
        if not isinstance(workspace_cfg, dict):
            workspace_cfg = {}

        workspace_cfg.setdefault("enabled", True)
        workspace_cfg.setdefault("size_threshold_mb", 30)
        workspace_cfg.setdefault("parent_strategy", "previous")
        workspace_cfg.setdefault("submission_subdir", "submission")
        workspace_cfg.setdefault("source_codebase_dir", "")
        workspace_cfg.setdefault("bootstrap_codebase_dir", "")
        workspace_cfg.setdefault("session_dir", "")
        workspace_cfg.setdefault("copy_plan_cache_enabled", True)
        workspace_cfg.setdefault("copy_plan_cache_file", ".embomaster_copy_plan.json")
        workspace_cfg.setdefault("copy_plan_rebuild", False)
        return workspace_cfg

    def _get_parent_selection_config(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)

        selection_cfg = cfg_dict.get("parent_selection", {})
        if not isinstance(selection_cfg, dict):
            selection_cfg = {}

        selection_cfg.setdefault("enabled", True)
        selection_cfg.setdefault("allow_none", True)
        selection_cfg.setdefault("fallback_strategy", "best")
        selection_cfg.setdefault("require_valid_eval_for_last", True)
        selection_cfg.setdefault("require_metric_for_last", True)
        selection_cfg.setdefault("reject_last_on_k8s_status", ["failed", "timeout"])
        selection_cfg.setdefault("max_feedback_chars", 1200)
        selection_cfg.setdefault("recent_rounds", 3)
        return selection_cfg

    def _get_result_validation_config(self) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)

        validation_cfg = cfg_dict.get("result_validation", {})
        if not isinstance(validation_cfg, dict):
            validation_cfg = {}

        validation_cfg.setdefault("enabled", False)
        validation_cfg.setdefault("require_eval_dir", True)
        validation_cfg.setdefault("require_success_rate_txt", True)
        validation_cfg.setdefault("require_checkpoint", True)
        validation_cfg.setdefault("require_dataset_stats", True)
        validation_cfg.setdefault("invalidate_metric_on_failure", True)
        validation_cfg.setdefault("invalid_k8s_status", "invalid_artifacts")
        return validation_cfg

    def _choose_parent_workspace(
        self,
        workspace_cfg: dict[str, Any],
        parent_selection_cfg: dict[str, Any],
        previous_workspace_id: str | None,
        best_round: dict[str, Any] | None,
        last_round: dict[str, Any] | None,
    ) -> dict[str, Any]:
        strategy = str(workspace_cfg.get("parent_strategy", "previous")).strip().lower()
        if strategy == "none":
            return {"choice": "none", "workspace_id": None, "reason": "configured parent_strategy=none"}
        if strategy == "best":
            best_workspace_id = None
            if isinstance(best_round, dict):
                best_workspace_id = best_round.get("workspace_id")
            return {
                "choice": "best",
                "workspace_id": best_workspace_id or previous_workspace_id,
                "reason": "configured parent_strategy=best",
            }
        if strategy == "advisor":
            if not bool(parent_selection_cfg.get("enabled", True)):
                return {
                    "choice": "last",
                    "workspace_id": previous_workspace_id,
                    "reason": "advisor disabled, falling back to previous",
                }
            return self._choose_parent_workspace_with_advisor(
                parent_selection_cfg=parent_selection_cfg,
                previous_workspace_id=previous_workspace_id,
                best_round=best_round,
                last_round=last_round,
            )
        return {
            "choice": "last",
            "workspace_id": previous_workspace_id,
            "reason": "configured parent_strategy=previous",
        }

    def _choose_parent_workspace_with_advisor(
        self,
        parent_selection_cfg: dict[str, Any],
        previous_workspace_id: str | None,
        best_round: dict[str, Any] | None,
        last_round: dict[str, Any] | None,
    ) -> dict[str, Any]:
        recommendation = None
        if isinstance(last_round, dict):
            recommendation = last_round.get("parent_recommendation")
        recommended_choice = ""
        if isinstance(recommendation, dict):
            recommended_choice = str(recommendation.get("choice", "")).strip().lower()

        fallback_choice = str(parent_selection_cfg.get("fallback_strategy", "best")).strip().lower()
        if fallback_choice not in {"best", "last", "none"}:
            fallback_choice = "best"
        if recommended_choice not in {"best", "last", "none"}:
            recommended_choice = fallback_choice

        resolved = self._resolve_parent_choice(
            choice=recommended_choice,
            parent_selection_cfg=parent_selection_cfg,
            previous_workspace_id=previous_workspace_id,
            best_round=best_round,
            last_round=last_round,
        )
        if resolved is not None:
            return resolved

        fallback_resolved = self._resolve_parent_choice(
            choice=fallback_choice,
            parent_selection_cfg=parent_selection_cfg,
            previous_workspace_id=previous_workspace_id,
            best_round=best_round,
            last_round=last_round,
        )
        if fallback_resolved is not None:
            fallback_resolved["reason"] = (
                f"advisor fallback to {fallback_choice}: invalid recommendation {recommended_choice}"
            )
            return fallback_resolved

        return {
            "choice": "none",
            "workspace_id": None,
            "reason": "advisor fallback exhausted, using none",
        }

    def _resolve_parent_choice(
        self,
        choice: str,
        parent_selection_cfg: dict[str, Any],
        previous_workspace_id: str | None,
        best_round: dict[str, Any] | None,
        last_round: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if choice == "none":
            if not bool(parent_selection_cfg.get("allow_none", True)):
                return None
            return {"choice": "none", "workspace_id": None, "reason": "advisor selected none"}

        if choice == "best":
            best_workspace_id = None
            if isinstance(best_round, dict):
                best_workspace_id = best_round.get("workspace_id")
            if not best_workspace_id:
                return None
            return {
                "choice": "best",
                "workspace_id": best_workspace_id,
                "reason": "advisor selected best",
            }

        if choice == "last":
            if not previous_workspace_id or not self._is_round_valid_for_last(
                last_round, parent_selection_cfg
            ):
                return None
            return {
                "choice": "last",
                "workspace_id": previous_workspace_id,
                "reason": "advisor selected last",
            }
        return None

    def _is_round_valid_for_last(
        self, round_result: dict[str, Any] | None, parent_selection_cfg: dict[str, Any]
    ) -> bool:
        if not isinstance(round_result, dict):
            return False
        rejected_statuses = {
            str(s).strip().lower()
            for s in parent_selection_cfg.get("reject_last_on_k8s_status", [])
            if str(s).strip()
        }
        k8s_status = str(round_result.get("k8s_status", "")).strip().lower()
        if k8s_status in rejected_statuses:
            return False

        if bool(parent_selection_cfg.get("require_metric_for_last", True)):
            if round_result.get("metric_value") is None:
                return False

        artifacts = round_result.get("artifacts_summary", {})
        if bool(parent_selection_cfg.get("require_valid_eval_for_last", True)):
            if not isinstance(artifacts, dict) or not artifacts.get("has_eval_artifacts", False):
                return False
        return True

    def _candidate_best_round(
        self,
        best_round: dict[str, Any] | None,
        current_round: dict[str, Any],
        metric_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        if best_round is None:
            return current_round
        if self._is_better_metric(
            best_round.get("metric_value"), current_round.get("metric_value"), metric_cfg
        ):
            return current_round
        return best_round

    def _collect_artifacts_summary(
        self,
        round_result: dict[str, Any],
        k8s_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        codebase_path_raw = str(round_result.get("workspace_codebase_path", "")).strip()
        codebase_path = Path(codebase_path_raw) if codebase_path_raw else None
        if not codebase_path or not codebase_path.exists():
            return {
                "has_metric": round_result.get("metric_value") is not None,
                "has_eval_artifacts": False,
                "has_task_eval_dir": False,
                "has_result_txt": False,
                "has_submission_csv": False,
                "has_policy_best_ckpt": False,
                "has_policy_last_ckpt": False,
                "has_any_epoch_ckpt": False,
                "has_dataset_stats": False,
            }

        manifest_env = k8s_cfg.get("manifest_env", {})
        task_name = ""
        if isinstance(manifest_env, dict):
            task_name = str(manifest_env.get("TASK_NAME", "")).strip()

        eval_root = codebase_path / "eval_result"
        task_eval_dir = eval_root / task_name if task_name else eval_root
        ckpt_root = codebase_path / "policy" / "ACT" / "act_ckpt"
        submission_dir = codebase_path / "submission"

        return {
            "has_metric": round_result.get("metric_value") is not None,
            "has_eval_artifacts": self._path_has_files(task_eval_dir),
            "has_task_eval_dir": task_eval_dir.exists(),
            "has_result_txt": self._tree_has_pattern(task_eval_dir, "_result.txt")
            or self._tree_has_pattern(task_eval_dir, "result_*.txt"),
            "has_submission_csv": self._tree_has_pattern(submission_dir, "submission_*.csv"),
            "has_policy_best_ckpt": self._tree_has_named_file(ckpt_root, "policy_best.ckpt"),
            "has_policy_last_ckpt": self._tree_has_named_file(ckpt_root, "policy_last.ckpt"),
            "has_any_epoch_ckpt": self._tree_has_pattern(ckpt_root, "policy_epoch_*.ckpt"),
            "has_dataset_stats": self._tree_has_named_file(ckpt_root, "dataset_stats.pkl"),
        }

    def _apply_result_validation(
        self,
        round_result: dict[str, Any],
        result_validation_cfg: dict[str, Any],
    ) -> None:
        if not bool(result_validation_cfg.get("enabled", True)):
            return

        artifacts = round_result.get("artifacts_summary", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        errors: list[str] = []
        if bool(result_validation_cfg.get("require_eval_dir", True)):
            if not artifacts.get("has_task_eval_dir", False):
                errors.append("missing_task_eval_dir")
        if bool(result_validation_cfg.get("require_success_rate_txt", True)):
            if not artifacts.get("has_result_txt", False):
                errors.append("missing_success_rate_txt")
        if bool(result_validation_cfg.get("require_checkpoint", True)):
            has_checkpoint = any(
                bool(artifacts.get(key, False))
                for key in ("has_policy_best_ckpt", "has_policy_last_ckpt", "has_any_epoch_ckpt")
            )
            if not has_checkpoint:
                errors.append("missing_checkpoint")
        if bool(result_validation_cfg.get("require_dataset_stats", True)):
            if not artifacts.get("has_dataset_stats", False):
                errors.append("missing_dataset_stats")

        round_result["validation_errors"] = errors
        round_result["result_valid"] = not errors

        if not errors:
            return

        invalid_status = str(result_validation_cfg.get("invalid_k8s_status", "invalid_artifacts")).strip()
        if invalid_status:
            round_result["k8s_status"] = invalid_status
        round_result["status"] = "invalid_artifacts"
        round_result["metric_valid"] = False
        if bool(result_validation_cfg.get("invalidate_metric_on_failure", True)):
            round_result["metric_value"] = None
            round_result["metric_source"] = "invalidated:artifact_validation"

    def _path_has_files(self, path: Path) -> bool:
        if not path.exists():
            return False
        for child in path.rglob("*"):
            if child.is_file():
                return True
        return False

    def _tree_has_named_file(self, root: Path, filename: str) -> bool:
        if not root.exists():
            return False
        return any(root.rglob(filename))

    def _tree_has_pattern(self, root: Path, pattern: str) -> bool:
        if not root.exists():
            return False
        return any(root.rglob(pattern))

    def _format_round_summary(
        self, round_result: dict[str, Any] | None, label: str
    ) -> str:
        if not isinstance(round_result, dict):
            return f"Candidate: {label}\n- unavailable"
        artifacts = round_result.get("artifacts_summary", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        coding_result = str(round_result.get("coding_result", "") or "").strip()
        coding_result = re.sub(r"\s+", " ", coding_result)
        coding_result = coding_result[:280] if coding_result else "N/A"
        return (
            f"Candidate: {label}\n"
            f"- round_index: {round_result.get('round_index', 'N/A')}\n"
            f"- workspace_id: {round_result.get('workspace_id', 'N/A')}\n"
            f"- k8s_status: {round_result.get('k8s_status', 'unknown')}\n"
            f"- metric: {round_result.get('metric_value', 'None')}\n"
            f"- result_valid: {round_result.get('result_valid', True)}\n"
            f"- validation_errors: {','.join(round_result.get('validation_errors', [])) or 'none'}\n"
            f"- has_eval_artifacts: {artifacts.get('has_eval_artifacts', False)}\n"
            f"- has_result_txt: {artifacts.get('has_result_txt', False)}\n"
            f"- has_policy_best_ckpt: {artifacts.get('has_policy_best_ckpt', False)}\n"
            f"- has_policy_last_ckpt: {artifacts.get('has_policy_last_ckpt', False)}\n"
            f"- has_any_epoch_ckpt: {artifacts.get('has_any_epoch_ckpt', False)}\n"
            f"- has_dataset_stats: {artifacts.get('has_dataset_stats', False)}\n"
            f"- coding_summary: {coding_result}"
        )

    def _format_recent_round_summaries(self, rounds: list[dict[str, Any]]) -> str:
        if not rounds:
            return "No recent rounds."
        lines: list[str] = []
        for round_result in rounds:
            artifacts = round_result.get("artifacts_summary", {})
            if not isinstance(artifacts, dict):
                artifacts = {}
            lines.append(
                (
                    f"Round {round_result.get('round_index', 'N/A')}: "
                    f"status={round_result.get('k8s_status', 'unknown')}, "
                    f"metric={round_result.get('metric_value', 'None')}, "
                    f"valid={round_result.get('result_valid', True)}, "
                    f"eval={artifacts.get('has_eval_artifacts', False)}, "
                    f"parent_used={round_result.get('parent_choice_used', 'N/A')}"
                )
            )
        return "\n".join(lines)

    def _format_parent_constraints(self, parent_selection_cfg: dict[str, Any]) -> str:
        fallback_strategy = str(parent_selection_cfg.get("fallback_strategy", "best")).strip()
        allow_none = bool(parent_selection_cfg.get("allow_none", True))
        require_eval = bool(parent_selection_cfg.get("require_valid_eval_for_last", True))
        require_metric = bool(parent_selection_cfg.get("require_metric_for_last", True))
        reject_statuses = ", ".join(
            str(s).strip()
            for s in parent_selection_cfg.get("reject_last_on_k8s_status", [])
            if str(s).strip()
        ) or "none"
        return (
            f"- Allowed choices: best, last, none\n"
            f"- Fallback strategy: {fallback_strategy}\n"
            f"- Allow none: {allow_none}\n"
            f"- last requires metric: {require_metric}\n"
            f"- last requires eval artifacts: {require_eval}\n"
            f"- last is rejected on k8s_status in: {reject_statuses}"
        )

    def _parse_parent_recommendation(self, feedback_text: str) -> dict[str, Any] | None:
        content = str(feedback_text or "")
        match = re.search(
            r"##\s*Parent Recommendation\s*"
            r"Choice:\s*(best|last|none)\s*"
            r"Confidence:\s*([0-9]*\.?[0-9]+)\s*"
            r"Reason:\s*(.+?)(?:\n\s*\n|\Z)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        confidence = self._to_float(match.group(2))
        if confidence is None:
            confidence = 0.0
        return {
            "choice": match.group(1).strip().lower(),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": match.group(3).strip(),
        }

    def _build_round_workspace_context(
        self,
        task_id: str,
        round_index: int,
        workspace_cfg: dict[str, Any],
        parent_workspace_id: str | None,
    ) -> dict[str, Any]:
        workspace_id = self._build_workspace_id(task_id, round_index)
        workspace_path = self._get_workspace_path()
        session_dir = self._resolve_session_dir(workspace_path, workspace_cfg)
        source_codebase_dir = self._resolve_source_codebase_dir(workspace_path, workspace_cfg)
        bootstrap_codebase_dir = self._resolve_bootstrap_codebase_dir(workspace_path, workspace_cfg)
        submission_subdir = (
            str(workspace_cfg.get("submission_subdir", "submission")).strip() or "submission"
        )
        if not bool(workspace_cfg.get("enabled", True)):
            disabled_path = source_codebase_dir or workspace_path
            submission_dir = disabled_path / submission_subdir
            submission_dir.mkdir(parents=True, exist_ok=True)
            return {
                "workspace_id": workspace_id,
                "parent_workspace_id": parent_workspace_id,
                "workspace_codebase_path": str(disabled_path),
                "workspace_large_dirs": [],
                "large_dirs_count": 0,
                "source_type": "disabled",
                "submission_dir": str(submission_dir),
                "session_dir": str(session_dir),
            }

        size_threshold_mb = int(workspace_cfg.get("size_threshold_mb", 30))
        size_threshold_bytes = size_threshold_mb * 1024 * 1024
        use_copy_plan_cache = bool(workspace_cfg.get("copy_plan_cache_enabled", True))
        force_rebuild_copy_plan = bool(workspace_cfg.get("copy_plan_rebuild", False))
        copy_plan_cache_file: Path | None = None
        cache_file_raw = str(workspace_cfg.get("copy_plan_cache_file", "")).strip()
        if use_copy_plan_cache:
            if not cache_file_raw:
                cache_file_raw = ".embomaster_copy_plan.json"
            cache_file_path = Path(cache_file_raw).expanduser()
            if not cache_file_path.is_absolute():
                base = source_codebase_dir if source_codebase_dir else workspace_path
                cache_file_path = (base / cache_file_path).resolve()
            copy_plan_cache_file = cache_file_path

        try:
            codebase_info = prepare_workspace_codebase(
                session_dir=session_dir,
                workspace_id=workspace_id,
                source_codebase_dir=source_codebase_dir,
                bootstrap_codebase_dir=bootstrap_codebase_dir,
                parent_workspace_id=parent_workspace_id,
                size_threshold=size_threshold_bytes,
                copy_plan_cache_file=copy_plan_cache_file,
                use_copy_plan_cache=use_copy_plan_cache,
                force_rebuild_copy_plan=force_rebuild_copy_plan,
            )
            cleanup_status = cleanup_eval_result(codebase_info.path)
            if any(value == "failed" for value in cleanup_status.values()):
                self.logger.warning(
                    "Workspace cleanup incomplete for %s: %s",
                    codebase_info.path,
                    cleanup_status,
                )
        except Exception as exc:
            self.logger.warning(
                "Failed to prepare isolated workspace (fallback to shared codebase): %r", exc
            )
            fallback_path = workspace_path / "codebase"
            codebase_info = WorkspaceCodebaseInfo(
                path=fallback_path if fallback_path.exists() else workspace_path,
                large_dirs=[],
                source_type="fallback",
                parent_workspace_id=parent_workspace_id,
            )

        submission_dir = Path(codebase_info.path) / submission_subdir
        submission_dir.mkdir(parents=True, exist_ok=True)
        large_dirs = list(codebase_info.large_dirs or [])

        return {
            "workspace_id": workspace_id,
            "parent_workspace_id": parent_workspace_id,
            "workspace_codebase_path": str(codebase_info.path),
            "workspace_large_dirs": large_dirs,
            "large_dirs_count": len(large_dirs),
            "source_type": str(codebase_info.source_type),
            "submission_dir": str(submission_dir),
            "session_dir": str(session_dir),
        }

    def _build_workspace_id(self, task_id: str, round_index: int) -> str:
        task_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", task_id).strip("-") or "task"
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{task_tag}-r{round_index}-{ts}"

    def _get_workspace_path(self) -> Path:
        session = getattr(self.coding_agent, "session", None)
        workspace = "."
        if session and hasattr(session, "config"):
            workspace = getattr(session.config, "workspace_path", ".")
        return Path(str(workspace)).expanduser().resolve()

    def _resolve_session_dir(self, workspace_path: Path, workspace_cfg: dict[str, Any]) -> Path:
        session_dir_raw = str(workspace_cfg.get("session_dir", "")).strip()
        if not session_dir_raw:
            workspace_path.mkdir(parents=True, exist_ok=True)
            return workspace_path
        session_dir = Path(session_dir_raw).expanduser()
        if not session_dir.is_absolute():
            session_dir = (workspace_path / session_dir).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _resolve_source_codebase_dir(
        self, workspace_path: Path, workspace_cfg: dict[str, Any]
    ) -> Path | None:
        src_raw = str(workspace_cfg.get("source_codebase_dir", "")).strip()
        if src_raw:
            src_path = Path(src_raw).expanduser()
            if not src_path.is_absolute():
                src_path = (workspace_path / src_path).resolve()
            if src_path.exists():
                return src_path
            self.logger.warning("Configured source_codebase_dir not found: %s", src_path)
            return None

        workspace_codebase = workspace_path / "codebase"
        if workspace_codebase.exists():
            return workspace_codebase.resolve()
        return None

    def _resolve_bootstrap_codebase_dir(
        self, workspace_path: Path, workspace_cfg: dict[str, Any]
    ) -> Path | None:
        bootstrap_raw = str(workspace_cfg.get("bootstrap_codebase_dir", "")).strip()
        if not bootstrap_raw:
            return None
        bootstrap_path = Path(bootstrap_raw).expanduser()
        if not bootstrap_path.is_absolute():
            bootstrap_path = (workspace_path / bootstrap_path).resolve()
        if bootstrap_path.exists():
            return bootstrap_path
        self.logger.warning("Configured bootstrap_codebase_dir not found: %s", bootstrap_path)
        return None

    def _create_monitor_writer(self, task_id: str) -> EmboMasterMonitorWriter | None:
        if not self.run_dir:
            return None
        try:
            return EmboMasterMonitorWriter(run_dir=self.run_dir, task_id=task_id)
        except Exception as exc:
            self.logger.warning("Failed to initialize monitor writer: %s", exc)
            return None
