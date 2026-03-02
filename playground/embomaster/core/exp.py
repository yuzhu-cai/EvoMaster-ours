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

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        self.logger.info("Starting EmboMaster experiment")
        settings = self._get_experiment_settings()
        k8s_cfg = self._get_k8s_config()
        workspace_cfg = self._get_workspace_isolation_config()
        max_rounds = int(settings.get("steps", 1))
        time_limit_seconds = int(settings.get("time_limit_seconds", 0))

        round_results: list[dict[str, Any]] = []
        best_round: dict[str, Any] | None = None
        best_metric: float | None = None
        previous_workspace_id: str | None = None
        previous_feedback = "N/A"
        start_time = time.monotonic()
        stopped_reason = "completed_all_rounds"

        for round_index in range(1, max_rounds + 1):
            if time_limit_seconds > 0 and (time.monotonic() - start_time) >= time_limit_seconds:
                stopped_reason = "time_limit_reached"
                break

            self.logger.info("=== Round %s/%s ===", round_index, max_rounds)
            BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=round_index)

            parent_workspace_id = self._choose_parent_workspace_id(
                workspace_cfg, previous_workspace_id, best_round
            )
            round_result = self._run_one_round(
                task_description=task_description,
                task_id=task_id,
                round_index=round_index,
                max_rounds=max_rounds,
                previous_feedback=previous_feedback,
                best_metric=best_metric,
                k8s_cfg=k8s_cfg,
                metric_cfg=settings,
                workspace_cfg=workspace_cfg,
                parent_workspace_id=parent_workspace_id,
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

            if settings.get("stop_on_job_failed", False):
                if round_result.get("k8s_status") in {"failed", "timeout"}:
                    stopped_reason = f"stop_on_{round_result.get('k8s_status')}"
                    break

        status = "completed"
        if not round_results:
            status = "failed"

        result: dict[str, Any] = {
            "status": status,
            "task_id": task_id,
            "total_rounds": len(round_results),
            "stopped_reason": stopped_reason,
            "rounds": round_results,
            "best_metric": best_metric,
            "best_round_index": best_round.get("round_index") if best_round else None,
            "best_round": best_round,
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
        return result

    def _run_one_round(
        self,
        task_description: str,
        task_id: str,
        round_index: int,
        max_rounds: int,
        previous_feedback: str,
        best_metric: float | None,
        k8s_cfg: dict[str, Any],
        metric_cfg: dict[str, Any],
        workspace_cfg: dict[str, Any],
        parent_workspace_id: str | None,
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
        self.coding_agent._prompt_format_kwargs.update(
            {
                "round_index": round_index,
                "max_rounds": max_rounds,
                "feedback_for_next_round": previous_feedback,
                "best_metric": "None" if best_metric is None else str(best_metric),
                "workspace_id": workspace_context.get("workspace_id", "N/A"),
                "parent_workspace_id": workspace_context.get("parent_workspace_id", "N/A"),
                "workspace_codebase_path": workspace_context.get("workspace_codebase_path", "N/A"),
                "workspace_source_type": workspace_context.get("source_type", "N/A"),
                "workspace_large_dirs_count": workspace_context.get("large_dirs_count", 0),
            }
        )
        session = getattr(self.coding_agent, "session", None)
        original_workspace_path = None
        workspace_codebase_path = workspace_context.get("workspace_codebase_path")
        if session and workspace_codebase_path:
            original_workspace_path = getattr(session.config, "workspace_path", None)
            session.config.workspace_path = str(workspace_codebase_path)
        try:
            trajectory = self.coding_agent.run(coding_task)
        finally:
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
            "workspace_codebase_path": workspace_context.get("workspace_codebase_path"),
            "workspace_source_type": workspace_context.get("source_type"),
            "workspace_large_dirs_count": workspace_context.get("large_dirs_count", 0),
        }

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

        if self.feedback_agent:
            feedback = self._run_feedback_round(
                task_description=task_description,
                task_id=task_id,
                round_index=round_index,
                round_result=round_result,
            )
            round_result["feedback"] = feedback

        return round_result

    def _run_feedback_round(
        self,
        task_description: str,
        task_id: str,
        round_index: int,
        round_result: dict[str, Any],
    ) -> str:
        logs = ""
        if isinstance(round_result.get("k8s_result"), dict):
            logs = str(round_result["k8s_result"].get("logs", {}).get("output", ""))
        logs_tail = logs[-3000:] if logs else "N/A"

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
            }
        )
        try:
            trajectory = self.feedback_agent.run(task)
            return self._extract_agent_response(trajectory)
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
        workspace_cfg.setdefault("session_dir", "")
        workspace_cfg.setdefault("copy_plan_cache_enabled", True)
        workspace_cfg.setdefault("copy_plan_cache_file", ".embomaster_copy_plan.json")
        workspace_cfg.setdefault("copy_plan_rebuild", False)
        return workspace_cfg

    def _choose_parent_workspace_id(
        self,
        workspace_cfg: dict[str, Any],
        previous_workspace_id: str | None,
        best_round: dict[str, Any] | None,
    ) -> str | None:
        strategy = str(workspace_cfg.get("parent_strategy", "previous")).strip().lower()
        if strategy == "none":
            return None
        if strategy == "best":
            best_workspace_id = None
            if isinstance(best_round, dict):
                best_workspace_id = best_round.get("workspace_id")
            return best_workspace_id or previous_workspace_id
        return previous_workspace_id

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
