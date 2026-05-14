"""Single-agent playground for AI R&D benchmarks.

The playground keeps benchmark policy inside one tool-using agent and keeps the
outer Python harness thin: mount inputs, write task instructions, run the agent,
and report whether the required artifact exists.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import signal
import shutil
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from evomaster.agent import BaseAgent
from evomaster.core import BaseExp, BasePlayground, register_playground


_ALLOWED_BENCHMARKS = {"mlebench", "paperbench", "posttrainbench"}
_DEFAULT_INPUT_MOUNTS = {
    "mlebench": "/mlebench/input",
    "paperbench": "/paperbench/input",
    "posttrainbench": "/posttrainbench/task",
}
_DEFAULT_ARTIFACTS = {
    "mlebench": "/workspace/submission.csv",
    "paperbench": "/workspace/reproduce.sh",
    "posttrainbench": "/workspace/final_model",
}


class _RuntimeDeadlineExceeded(BaseException):
    """Raised by the per-task hard wall-clock timer."""


@register_playground("ai_rd_benchmark_agent")
class AIRDBenchmarkAgentPlayground(BasePlayground):
    """A thin, Docker-first benchmark harness for one coding/research agent."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "configs"
                / "ai_rd_benchmark_agent"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("general_agent")
        self._benchmark_spec: dict[str, Any] = {}

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
        on_step=None,
    ) -> dict:
        """Run one benchmark task with a single persistent-workspace agent."""
        self._benchmark_spec = self._load_task_spec(task_description)
        self._apply_dynamic_mounts(self._benchmark_spec)

        try:
            self.register_thread()
            self.setup()
            self._setup_trajectory_file(output_file)
            self._prepare_workspace(self._benchmark_spec)

            agent = self.agents.general_agent
            original_kwargs = getattr(agent, "_prompt_format_kwargs", {}).copy()
            agent._prompt_format_kwargs.update(self._prompt_kwargs(self._benchmark_spec))

            BaseAgent.set_exp_info(exp_name="AIRDBenchmark", exp_index=0)
            exp = BaseExp(agent, self.config)
            if self.run_dir:
                exp.set_run_dir(self.run_dir)

            task_id = getattr(self, "task_id", None) or "task_0"
            iteration_results: list[dict[str, Any]] = []
            result: dict[str, Any] | None = None
            deadline_reached = False
            start_time = time.monotonic()
            deadline = self._runtime_deadline(start_time)
            if deadline is not None and hasattr(self.session, "set_deadline"):
                self.session.set_deadline(deadline)

            try:
                try:
                    with self._hard_runtime_deadline(deadline):
                        result = exp.run(
                            self._agent_description(self._benchmark_spec),
                            task_id=task_id,
                            images=images,
                            on_step=on_step,
                        )
                    iteration_results.append(
                        self._finalize_iteration_round(
                            round_index=0,
                            result=result,
                            elapsed_seconds=time.monotonic() - start_time,
                        )
                    )
                    self._write_strategy_memory(iteration_results[-1])

                    if self._iteration_enabled(self._benchmark_spec):
                        iteration_results.extend(
                            self._run_iteration_rounds(
                                agent=agent,
                                task_id=task_id,
                                on_step=on_step,
                                start_time=start_time,
                                deadline=deadline,
                            )
                        )
                except _RuntimeDeadlineExceeded:
                    deadline_reached = True
                    self.logger.info("Runtime deadline reached; aborting active benchmark work.")
                    self._kill_active_session_processes()
                    self._finish_agent_trajectory_due_to_deadline(agent)
                    result = result or self._deadline_result(task_id, agent)
                    timeout_record = self._finalize_iteration_round(
                        round_index=len(iteration_results),
                        result=result,
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                    timeout_record["deadline_reached"] = True
                    iteration_results.append(timeout_record)
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_reached = True
            finally:
                agent._prompt_format_kwargs = original_kwargs
                if hasattr(self.session, "clear_deadline"):
                    self.session.clear_deadline()

            self._promote_best_artifact(self._benchmark_spec)
            artifact_status = self._collect_artifact_status(self._benchmark_spec)
            grade_status = self._run_optional_grade(
                self._benchmark_spec,
                deadline=deadline,
                skip_if_deadline_reached=deadline_reached,
            )
            if result is None:
                result = self._deadline_result(task_id, agent)
            if deadline_reached:
                result["deadline_reached"] = True
                if artifact_status.get("exists"):
                    result["status"] = "completed"
                else:
                    result["status"] = "failed"
                    result["error"] = "runtime_deadline_reached_without_required_artifact"
            result.update(
                {
                    "benchmark": self._benchmark_spec["benchmark"],
                    "artifact_status": artifact_status,
                    "grade_status": grade_status,
                    "iterations": iteration_results,
                }
            )
            return result
        finally:
            self.cleanup()

    def _runtime_deadline(self, start_time: float) -> float | None:
        """Return the absolute monotonic deadline for this benchmark run."""
        cfg = self._iteration_config()
        max_runtime_seconds = float(cfg.get("max_runtime_seconds", 0) or 0)
        if max_runtime_seconds <= 0:
            return None
        return start_time + max_runtime_seconds

    @contextlib.contextmanager
    def _hard_runtime_deadline(self, deadline: float | None):
        """Raise from in-flight agent work when the task wall clock expires."""
        if deadline is None:
            yield
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RuntimeDeadlineExceeded("runtime deadline reached")

        # Python can only deliver SIGALRM to the main thread. If a caller runs
        # this playground in a helper thread, fall back to the cooperative
        # session deadline checks.
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.getitimer(signal.ITIMER_REAL)

        def _raise_deadline(_signum, _frame):
            raise _RuntimeDeadlineExceeded("runtime deadline reached")

        signal.signal(signal.SIGALRM, _raise_deadline)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

    def _kill_active_session_processes(self) -> None:
        """Best-effort stop commands still active when the hard timer fires."""
        killer = getattr(self.session, "kill_active_processes", None)
        if not callable(killer):
            return
        try:
            killer()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to kill active session processes: %s", exc)

    @staticmethod
    def _finish_agent_trajectory_due_to_deadline(agent: BaseAgent) -> None:
        trajectory = getattr(agent, "trajectory", None)
        if trajectory is not None and getattr(trajectory, "status", None) == "running":
            trajectory.finish("completed", {"reason": "runtime_deadline_reached"})

    @staticmethod
    def _deadline_result(task_id: str, agent: BaseAgent) -> dict[str, Any]:
        trajectory = getattr(agent, "trajectory", None)
        steps = len(getattr(trajectory, "steps", []) or []) if trajectory else 0
        return {
            "status": "completed",
            "steps": steps,
            "task_id": task_id,
            "deadline_reached": True,
        }

    def _load_task_spec(self, task_description: str) -> dict[str, Any]:
        """Load a benchmark spec from YAML/JSON or derive one from plain text."""
        raw = (task_description or "").strip()
        spec: dict[str, Any]

        maybe_path: Path | None = None
        if raw and len(raw) < 4096:
            try:
                maybe_path = Path(raw)
            except (OSError, ValueError):
                maybe_path = None

        path_is_spec = False
        if maybe_path is not None:
            try:
                path_is_spec = (
                    maybe_path.exists()
                    and maybe_path.is_file()
                    and maybe_path.suffix.lower() in {".yaml", ".yml", ".json"}
                )
            except OSError:
                path_is_spec = False

        if path_is_spec and maybe_path is not None:
            with open(maybe_path, "r", encoding="utf-8") as f:
                if maybe_path.suffix.lower() == ".json":
                    spec = json.load(f) or {}
                else:
                    spec = yaml.safe_load(f) or {}
            spec.setdefault("spec_path", str(maybe_path.resolve()))
        else:
            try:
                parsed = json.loads(raw) if raw.startswith("{") else None
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                spec = parsed
            else:
                cfg = self._benchmark_config()
                spec = {
                    "benchmark": cfg.get("default_benchmark", "mlebench"),
                    "description": raw,
                }

        cfg = self._benchmark_config()
        benchmark = str(spec.get("benchmark") or cfg.get("default_benchmark") or "mlebench").lower()
        benchmark = benchmark.replace("-", "").replace("_", "")
        alias = {
            "mle": "mlebench",
            "mlebench": "mlebench",
            "paper": "paperbench",
            "paperbench": "paperbench",
            "posttrain": "posttrainbench",
            "posttraining": "posttrainbench",
            "posttrainbench": "posttrainbench",
        }.get(benchmark, benchmark)
        if alias not in _ALLOWED_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {spec.get('benchmark')!r}")
        spec["benchmark"] = alias
        spec.setdefault("workspace", cfg.get("workspace", "/workspace"))
        spec.setdefault("input_mount", cfg.get(f"{alias}_input_mount", _DEFAULT_INPUT_MOUNTS[alias]))
        spec.setdefault("artifact_path", cfg.get(f"{alias}_artifact", _DEFAULT_ARTIFACTS[alias]))
        spec.setdefault("description", raw)
        return spec

    def _benchmark_config(self) -> dict[str, Any]:
        cfg = getattr(self.config, "benchmark_agent", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        return dict(cfg)

    def _apply_dynamic_mounts(self, spec: dict[str, Any]) -> None:
        """Add benchmark input mounts from the task spec before Docker starts."""
        if self.config.session.get("type", "local") != "docker":
            return

        docker_cfg = self.config.session.setdefault("docker", {})
        volumes = dict(docker_cfg.get("volumes") or {})
        benchmark = spec["benchmark"]

        host_input = self._host_input_path(spec)
        if host_input:
            volumes[str(host_input)] = {
                "target": spec.get("input_mount") or _DEFAULT_INPUT_MOUNTS[benchmark],
                "read_only": True,
            }

        # Optional extra read-only mounts for large caches / model stores.
        for item in spec.get("extra_mounts", []) or []:
            if not isinstance(item, dict) or not item.get("host") or not item.get("target"):
                continue
            volumes[str(item["host"])] = {
                "target": str(item["target"]),
                "read_only": bool(item.get("read_only", True)),
            }

        docker_cfg["volumes"] = volumes

    def _host_input_path(self, spec: dict[str, Any]) -> str | None:
        benchmark = spec["benchmark"]
        keys = {
            "mlebench": ("input_dir", "public_dir", "competition_dir"),
            "paperbench": ("paper_dir", "input_dir", "dataset_dir"),
            "posttrainbench": ("task_dir", "input_dir", "benchmark_dir"),
        }[benchmark]
        for key in keys:
            value = spec.get(key)
            if value:
                return str(value)
        return None

    def _prepare_workspace(self, spec: dict[str, Any]) -> None:
        workspace = spec.get("workspace", "/workspace")
        workspace_q = shlex.quote(workspace.rstrip("/") or "/")
        self.session.exec_bash(
            f"mkdir -p {workspace_q}/work {workspace_q}/artifacts {workspace_q}/logs && "
            f"chmod -R a+rwX {workspace_q}/work {workspace_q}/artifacts {workspace_q}/logs "
            "2>/dev/null || true",
            timeout=60,
        )

    def _iteration_config(self) -> dict[str, Any]:
        cfg = self._benchmark_config().get("iteration", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        return dict(cfg)

    def _iteration_enabled(self, spec: dict[str, Any]) -> bool:
        cfg = self._iteration_config()
        if spec.get("iteration_enabled") is not None:
            return bool(spec.get("iteration_enabled"))
        return bool(cfg.get("enabled", False))

    def _iteration_prompt_template(self) -> str:
        cfg = self._iteration_config()
        prompt_file = cfg.get("prompt_file", "prompts/iteration_prompt.txt")
        path = Path(str(prompt_file))
        if not path.is_absolute():
            playground_base = Path(str(self.config_dir).replace("configs", "playground"))
            path = playground_base / path
        if path.exists():
            return path.read_text(encoding="utf-8")
        return (
            "Continue improving this ML solution in the same workspace. "
            "Do not repeat completed experiments; run at least one new atomic "
            "improvement, update logs, keep the best artifact, validate, then finish."
        )

    def _iteration_prompt(self, *, round_index: int, max_rounds: int, elapsed_seconds: float) -> str:
        cfg = self._iteration_config()
        return self._iteration_prompt_template().format(
            round_index=round_index,
            max_rounds=max_rounds,
            elapsed_seconds=int(elapsed_seconds),
            elapsed_minutes=round(elapsed_seconds / 60.0, 2),
            elapsed_hours=round(elapsed_seconds / 3600.0, 3),
            max_runtime_seconds=int(cfg.get("max_runtime_seconds", 0) or 0),
            min_rounds=int(cfg.get("min_rounds", 1) or 1),
            benchmark=self._benchmark_spec.get("benchmark", ""),
            competition_id=self._benchmark_spec.get("competition_id", ""),
            workspace=self._benchmark_spec.get("workspace", "/workspace"),
            input_mount=self._benchmark_spec.get("input_mount", ""),
            artifact_path=self._benchmark_spec.get("artifact_path", ""),
            benchmark_targets=self._benchmark_target_summary(self._benchmark_spec),
            current_best_state=self._current_best_state_summary(),
            experiment_memory=self._experiment_memory_summary(),
            strategy_memory=self._strategy_memory_summary(),
        )

    def _run_iteration_rounds(
        self,
        *,
        agent: BaseAgent,
        task_id: str,
        on_step,
        start_time: float,
        deadline: float | None,
    ) -> list[dict[str, Any]]:
        """Continue the same dialog/workspace for long-horizon improvement rounds."""
        cfg = self._iteration_config()
        max_rounds = int(cfg.get("max_rounds", 1) or 1)
        min_rounds = int(cfg.get("min_rounds", 1) or 1)
        max_runtime_seconds = float(cfg.get("max_runtime_seconds", 0) or 0)
        stop_file = str(cfg.get("stop_file", "/workspace/artifacts/STOP_ITERATION"))
        records: list[dict[str, Any]] = []
        max_stagnant_rounds = int(cfg.get("max_stagnant_rounds", 0) or 0)
        stagnant_rounds = 0

        for round_index in range(1, max_rounds):
            elapsed = time.monotonic() - start_time
            if max_runtime_seconds > 0 and elapsed >= max_runtime_seconds:
                self.logger.info("Stopping iterative mode: runtime budget reached.")
                break

            if round_index >= min_rounds and stop_file and self.session.is_file(stop_file):
                self.logger.info("Stopping iterative mode: agent created stop file %s", stop_file)
                break

            before_state = self._experiment_history_state()
            prompt = self._iteration_prompt(
                round_index=round_index,
                max_rounds=max_rounds,
                elapsed_seconds=elapsed,
            )
            BaseAgent.set_exp_info(exp_name=f"AIRDBenchmarkIter{round_index}", exp_index=round_index)
            self.logger.info("Starting iterative improvement round %s/%s", round_index + 1, max_rounds)
            with self._hard_runtime_deadline(deadline):
                trajectory = agent.continue_run(prompt, on_step=on_step)
            result = {
                "trajectory": trajectory,
                "status": trajectory.status,
                "steps": len(trajectory.steps),
                "task_id": f"{task_id}_iter_{round_index}",
            }
            records.append(
                round_record := self._finalize_iteration_round(
                    round_index=round_index,
                    result=result,
                    elapsed_seconds=time.monotonic() - start_time,
                )
            )
            after_state = round_record.get("experiment_state", {})
            progress = self._round_progress(before_state, after_state)
            round_record["progress"] = progress
            if progress["improved_best"]:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            round_record["stagnant_rounds"] = stagnant_rounds
            self._snapshot_iteration(round_index, round_record)
            self._write_strategy_memory(round_record)

            if not progress["recorded_new_experiment"]:
                self.logger.warning(
                    "Iteration round %s did not add an experiment_tracker record; treating as stagnant.",
                    round_index,
                )

            if round_index >= min_rounds and max_stagnant_rounds > 0 and stagnant_rounds >= max_stagnant_rounds:
                self.logger.info(
                    "Stopping iterative mode: %s stagnant rounds without best-score improvement.",
                    stagnant_rounds,
                )
                break

            if trajectory.status not in {"completed", "waiting_for_input"}:
                self.logger.warning("Stopping iterative mode: round %s ended with %s", round_index, trajectory.status)
                break

        return records

    def _finalize_iteration_round(
        self,
        *,
        round_index: int,
        result: dict[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        self._promote_best_artifact(self._benchmark_spec)
        artifact_status = self._collect_artifact_status(self._benchmark_spec)
        experiment_state = self._experiment_history_state()
        record = {
            "round": round_index,
            "status": result.get("status"),
            "steps": result.get("steps"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "artifact_status": artifact_status,
            "experiment_state": experiment_state,
        }
        self._snapshot_iteration(round_index, record)
        return record

    def _experiment_history_state(self) -> dict[str, Any]:
        """Summarize structured experiment memory from the host workspace."""
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return {"experiment_count": 0, "best": None}

        root = Path(host_workspace)
        experiments_path = root / "logs" / "experiments.jsonl"
        best_meta_path = root / "artifacts" / "best_meta.json"
        experiment_count = 0
        last_experiment: dict[str, Any] | None = None
        if experiments_path.exists():
            try:
                for line in experiments_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    experiment_count += 1
                    try:
                        last_experiment = json.loads(line)
                    except json.JSONDecodeError:
                        last_experiment = {"raw": line[-500:]}
            except Exception as exc:  # noqa: BLE001
                return {"experiment_count": experiment_count, "best": None, "error": str(exc)}

        best: dict[str, Any] | None = None
        if best_meta_path.exists():
            try:
                payload = json.loads(best_meta_path.read_text(encoding="utf-8", errors="replace"))
                best = {
                    "experiment_name": payload.get("experiment_name"),
                    "metric": payload.get("metric"),
                    "score": payload.get("score"),
                    "direction": payload.get("direction"),
                    "round_index": payload.get("round_index"),
                }
            except Exception as exc:  # noqa: BLE001
                best = {"error": str(exc)}

        return {
            "experiment_count": experiment_count,
            "best": best,
            "last_experiment": last_experiment,
        }

    def _experiment_memory_summary(self, *, experiment_limit: int = 12, plan_limit: int = 6) -> str:
        """Compile recent experiment/search memory for long-horizon prompts."""
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return "(No host workspace available yet.)"

        root = Path(host_workspace)
        experiments = self._read_jsonl_file(root / "logs" / "experiments.jsonl")
        plans = self._read_jsonl_file(root / "logs" / "experiment_backlog.jsonl")
        best_meta = self._read_json_file(root / "artifacts" / "best_meta.json")
        search_state = self._read_json_file(root / "logs" / "branch_search_state.json")
        if not experiments and not plans and not best_meta and not search_state:
            return "(No structured experiment memory recorded yet.)"

        lines: list[str] = []
        if best_meta:
            lines.extend(
                [
                    "Best recorded experiment:",
                    (
                        "- "
                        f"{best_meta.get('experiment_name')}: "
                        f"{best_meta.get('metric')}={best_meta.get('score')} "
                        f"({best_meta.get('direction')}, round={best_meta.get('round_index')})"
                    ),
                ]
            )
            notes = str(best_meta.get("notes") or "").strip()
            if notes:
                lines.append(f"  Notes: {notes[:900]}")

        if experiments:
            lines.append("")
            lines.append(f"Recent experiments (last {min(experiment_limit, len(experiments))}):")
            for rec in experiments[-experiment_limit:]:
                comparison = rec.get("comparison") if isinstance(rec.get("comparison"), dict) else {}
                delta = comparison.get("delta")
                improved = "YES" if rec.get("promoted_to_best") else "NO"
                line = (
                    "- "
                    f"{rec.get('experiment_name')}: {rec.get('metric')}={rec.get('score')} "
                    f"({rec.get('direction')}, round={rec.get('round_index')}, "
                    f"improved_best={improved}"
                )
                if delta is not None:
                    line += f", delta={delta}"
                line += ")"
                lines.append(line)
                notes = str(rec.get("notes") or "").strip()
                if notes:
                    lines.append(f"  Evidence: {notes[:700]}")

            failed = [
                rec.get("experiment_name")
                for rec in experiments
                if rec.get("promoted_to_best") is False and rec.get("experiment_name")
            ][-8:]
            if failed:
                lines.append("")
                lines.append("Recently non-improving branches to avoid repeating exactly:")
                for name in failed:
                    lines.append(f"- {name}")

        if plans:
            lines.append("")
            lines.append(f"Recent branch plans (last {min(plan_limit, len(plans))}):")
            for plan in plans[-plan_limit:]:
                selected = plan.get("selected_candidate") or "(not selected)"
                lines.append(f"- round={plan.get('round_index')}: selected={selected}")
                ranked = plan.get("ranked_candidates") or []
                if ranked:
                    top = ranked[0]
                    lines.append(
                        "  Top priority: "
                        f"{top.get('candidate')} "
                        f"(priority={top.get('priority')}, "
                        f"gain={top.get('expected_gain')}, "
                        f"cost={top.get('cost')}, risk={top.get('risk')})"
                    )
                rationale = str(plan.get("rationale") or "").strip()
                if rationale:
                    lines.append(f"  Rationale: {rationale[:500]}")

        branches = search_state.get("branches", {}) if isinstance(search_state.get("branches"), dict) else {}
        if branches:
            lines.append("")
            lines.append("Branch search state:")
            rows = []
            for name, stats in branches.items():
                if not isinstance(stats, dict):
                    continue
                rows.append(
                    (
                        name,
                        int(stats.get("visits", 0) or 0),
                        float(stats.get("mean_reward", 0.0) or 0.0),
                        int(stats.get("improved_count", 0) or 0),
                    )
                )
            for name, visits, mean_reward, improved_count in sorted(rows, key=lambda item: item[2], reverse=True)[:8]:
                lines.append(
                    f"- {name}: visits={visits}, mean_reward={mean_reward:.6g}, improved_count={improved_count}"
                )

        return "\n".join(lines)[-12000:]

    def _strategy_memory_summary(self) -> str:
        """Read deterministic long-horizon strategy memory from the host workspace."""
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return "(No host workspace available yet.)"
        path = Path(host_workspace) / "logs" / "strategy_memory.md"
        if not path.exists():
            return "(No strategy memory recorded yet.)"
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-12000:]
        except Exception as exc:  # noqa: BLE001
            return f"(Failed to read strategy memory: {exc})"

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        except Exception:  # noqa: BLE001
            return records
        return records

    def _write_strategy_memory(self, round_record: dict[str, Any]) -> None:
        """Persist a concise, harness-side improvement brief for future rounds."""
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return

        root = Path(host_workspace)
        logs_dir = root / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to prepare strategy memory directory: %s", exc)
            return

        experiments = self._read_jsonl_file(logs_dir / "experiments.jsonl")
        best = self._read_json_file(root / "artifacts" / "best_meta.json")
        search_state = self._read_json_file(logs_dir / "branch_search_state.json")
        progress = round_record.get("progress") or {}
        lines = [
            "# Long-Horizon Strategy Memory",
            "",
            "This file is written by the harness after each round so the next continuation prompt has compact memory, similar to ml_master's child/parent memory and ml_master_2's knowledge-promotion summaries.",
            "",
            "## Current State",
            f"- last_round: {round_record.get('round')}",
            f"- last_round_status: {round_record.get('status')}",
            f"- elapsed_seconds: {round_record.get('elapsed_seconds')}",
            f"- experiment_count: {len(experiments)}",
            f"- stagnant_rounds: {round_record.get('stagnant_rounds', 0)}",
            f"- recorded_new_experiment: {progress.get('recorded_new_experiment')}",
            f"- improved_best_this_round: {progress.get('improved_best')}",
            "",
            "## Benchmark Targets",
            self._benchmark_target_summary(self._benchmark_spec),
            "",
            "## Best",
        ]
        if best:
            lines.extend(
                [
                    f"- experiment: {best.get('experiment_name')}",
                    f"- metric: {best.get('metric')}",
                    f"- score: {best.get('score')}",
                    f"- direction: {best.get('direction')}",
                    f"- round_index: {best.get('round_index')}",
                    f"- notes: {str(best.get('notes') or '')[:1200]}",
                ]
            )
        else:
            lines.append("- no best experiment recorded yet")

        if experiments:
            lines.extend(["", "## Score Trajectory"])
            for rec in experiments[-20:]:
                marker = "BEST" if rec.get("promoted_to_best") else "tried"
                lines.append(
                    "- "
                    f"{marker}: {rec.get('experiment_name')} "
                    f"{rec.get('metric')}={rec.get('score')} "
                    f"round={rec.get('round_index')}"
                )

            lines.extend(["", "## Negative Evidence"])
            negative = [rec for rec in experiments if rec.get("promoted_to_best") is False][-10:]
            if negative:
                for rec in negative:
                    notes = str(rec.get("notes") or "").strip()
                    lines.append(f"- {rec.get('experiment_name')}: {notes[:500]}")
            else:
                lines.append("- no non-improving experiments recorded yet")

        branches = search_state.get("branches", {}) if isinstance(search_state.get("branches"), dict) else {}
        if branches:
            lines.extend(["", "## Branch Search State"])
            rows = []
            for name, stats in branches.items():
                if not isinstance(stats, dict):
                    continue
                rows.append(
                    (
                        name,
                        int(stats.get("visits", 0) or 0),
                        float(stats.get("mean_reward", 0.0) or 0.0),
                        int(stats.get("improved_count", 0) or 0),
                        stats.get("last_experiment_name"),
                    )
                )
            rows.sort(key=lambda item: (item[2], item[3]), reverse=True)
            for name, visits, mean_reward, improved_count, last_exp in rows[:10]:
                lines.append(
                    f"- {name}: visits={visits}, mean_reward={mean_reward:.6g}, "
                    f"improved_count={improved_count}, last_experiment={last_exp}"
                )

        lines.extend(
            [
                "",
                "## Next-Round Rules",
                "- Do not restart from scratch; edit or branch from the current best solution.",
                "- Do not repeat an exact non-improving branch listed above.",
                "- Before running the next experiment, record a plan with numeric candidate_scores, candidate_costs, and candidate_risks for every candidate; the tool rejects unscored plans.",
                "- Prefer one atomic experiment per continuation round unless a cheap sweep is directly tied to the same hypothesis.",
                "- If no experiment improves after the configured stagnant-round budget, create artifacts/STOP_ITERATION with the exhaustion rationale.",
                "",
            ]
        )

        try:
            (logs_dir / "strategy_memory.md").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to write strategy memory: %s", exc)

    def _round_progress(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """Detect whether a continuation round created evidence and improved best."""
        before_count = int(before.get("experiment_count") or 0)
        after_count = int(after.get("experiment_count") or 0)
        before_best = before.get("best") or {}
        after_best = after.get("best") or {}
        before_sig = (
            before_best.get("experiment_name"),
            before_best.get("metric"),
            before_best.get("score"),
            before_best.get("direction"),
        )
        after_sig = (
            after_best.get("experiment_name"),
            after_best.get("metric"),
            after_best.get("score"),
            after_best.get("direction"),
        )
        return {
            "recorded_new_experiment": after_count > before_count,
            "experiment_count_delta": after_count - before_count,
            "improved_best": after_sig != before_sig and after_best.get("score") is not None,
            "previous_best": before_sig,
            "current_best": after_sig,
        }

    def _snapshot_iteration(self, round_index: int, record: dict[str, Any]) -> None:
        cfg = self._iteration_config()
        if not bool(cfg.get("snapshot_each_round", True)):
            return
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return

        src_root = Path(host_workspace)
        dst = src_root / "iterations" / f"round_{round_index:02d}"
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for rel in ("submission.csv", "work", "logs", "artifacts"):
                src = src_root / rel
                target = dst / rel
                if not src.exists():
                    continue
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if src.is_dir():
                    shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    shutil.copy2(src, target)
            (dst / "summary.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to snapshot iteration %s: %s", round_index, exc)

    def _promote_best_artifact(self, spec: dict[str, Any]) -> None:
        """Copy agent-maintained best artifact back to the required artifact path."""
        cfg = self._iteration_config()
        if not bool(cfg.get("promote_best_artifact", True)):
            return
        workspace = str(spec.get("workspace", "/workspace")).rstrip("/") or "/workspace"
        artifact = str(spec.get("artifact_path") or _DEFAULT_ARTIFACTS[spec["benchmark"]])
        best_artifact = str(cfg.get("best_artifact_path", f"{workspace}/artifacts/best_submission.csv"))
        if spec["benchmark"] != "mlebench":
            return
        if not best_artifact or not self.session.is_file(best_artifact):
            return
        cmd = f"cp {shlex.quote(best_artifact)} {shlex.quote(artifact)}"
        self.logger.info("Promoting best iterative artifact: %s -> %s", best_artifact, artifact)
        self.session.exec_bash(cmd, timeout=60)

    def _prompt_kwargs(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "benchmark": spec["benchmark"],
            "competition_id": spec.get("competition_id", spec.get("name", "unknown")),
            "workspace": spec.get("workspace", "/workspace"),
            "input_mount": spec.get("input_mount"),
            "artifact_path": spec.get("artifact_path"),
            "task_spec_json": json.dumps(spec, indent=2, ensure_ascii=False),
            "competition_description": self._competition_description_excerpt(spec),
            "benchmark_targets": self._benchmark_target_summary(spec),
            "current_best_state": self._current_best_state_summary(),
            "experiment_memory": self._experiment_memory_summary(),
            "strategy_memory": self._strategy_memory_summary(),
        }

    def _agent_description(self, spec: dict[str, Any]) -> str:
        return str(spec.get("description") or f"Solve the {spec['benchmark']} task.")

    def _competition_description_excerpt(self, spec: dict[str, Any]) -> str:
        host_input = self._host_input_path(spec)
        if not host_input:
            return "(No host input directory was provided.)"
        desc_path = Path(host_input) / "description.md"
        if not desc_path.exists():
            return f"(No description.md found under {host_input}.)"
        try:
            text = desc_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            return f"(Failed to read {desc_path}: {e})"
        max_chars = int(self._benchmark_config().get("description_max_chars", 24000))
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[description.md truncated]"
        return text

    def _benchmark_target_summary(self, spec: dict[str, Any]) -> str:
        """Summarize public benchmark target metadata when available."""
        prepared_dir = self._prepared_dir(spec)
        if not prepared_dir:
            return "(No benchmark target metadata available.)"
        baseline_path = Path(prepared_dir) / "baseline.json"
        if not baseline_path.exists():
            return f"(No baseline.json found under {prepared_dir}.)"
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return f"(Failed to read benchmark target metadata from {baseline_path}: {exc})"

        baseline = payload.get("baseline", {}) if isinstance(payload, dict) else {}
        optim = payload.get("optim", "") if isinstance(payload, dict) else ""
        metric = payload.get("metric", "") if isinstance(payload, dict) else ""
        total = payload.get("total_teams", "") if isinstance(payload, dict) else ""
        ordered_keys = ["random", "median", "bronze", "silver", "gold", "best"]
        lines = [
            "Benchmark target metadata from baseline.json.",
            f"- metric: {metric or '(unknown)'}",
            f"- optimization: {'higher is better' if optim == 'high' else 'lower is better' if optim == 'low' else optim or '(unknown)'}",
            f"- total_teams: {total or '(unknown)'}",
        ]
        for key in ordered_keys:
            if key in baseline:
                lines.append(f"- {key}: {baseline[key]}")
        lines.append(
            "Use these only as target context; choose models by local validation on public training data, not by private grader feedback."
        )
        return "\n".join(lines)

    def _current_best_state_summary(self) -> str:
        """Read the tool-maintained best state from the mounted host workspace."""
        host_workspace = self._host_workspace_path()
        if not host_workspace:
            return "(No host workspace available yet.)"
        best_meta_path = Path(host_workspace) / "artifacts" / "best_meta.json"
        if not best_meta_path.exists():
            return "(No experiment_tracker best state recorded yet.)"
        try:
            payload = json.loads(best_meta_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return f"(Failed to read {best_meta_path}: {exc})"

        fields = {
            "experiment_name": payload.get("experiment_name"),
            "metric": payload.get("metric"),
            "score": payload.get("score"),
            "direction": payload.get("direction"),
            "round_index": payload.get("round_index"),
            "notes": payload.get("notes"),
            "best_submission_path": payload.get("best_submission_path"),
            "best_solution_path": payload.get("best_solution_path"),
        }
        return json.dumps(fields, indent=2, ensure_ascii=False, default=str)

    def _collect_artifact_status(self, spec: dict[str, Any]) -> dict[str, Any]:
        artifact = spec.get("artifact_path") or _DEFAULT_ARTIFACTS[spec["benchmark"]]
        if spec["benchmark"] == "posttrainbench":
            exists = self.session.is_directory(artifact)
        else:
            exists = self.session.is_file(artifact)
        return {"artifact": artifact, "exists": bool(exists)}

    @staticmethod
    def _remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _run_optional_grade(
        self,
        spec: dict[str, Any],
        *,
        deadline: float | None = None,
        skip_if_deadline_reached: bool = False,
    ) -> dict[str, Any] | None:
        cfg = self._benchmark_config()
        command = spec.get("grade_command") or cfg.get("grade_command")
        if not command:
            return None

        remaining = self._remaining_seconds(deadline)
        if skip_if_deadline_reached or (remaining is not None and remaining <= 0):
            return {
                "status": "skipped",
                "reason": "runtime_deadline_reached",
            }

        values = {
            "benchmark": spec["benchmark"],
            "competition_id": spec.get("competition_id", ""),
            "workspace": spec.get("workspace", "/workspace"),
            "artifact_path": spec.get("artifact_path"),
            "host_artifact_path": self._host_artifact_path(spec),
            "host_workspace": self._host_workspace_path(),
            "host_input_path": self._host_input_path(spec) or "",
            "prepared_dir": self._prepared_dir(spec),
            "run_dir": str(self.run_dir) if self.run_dir else "",
        }
        values.update({k: str(v) for k, v in spec.items() if isinstance(v, (str, int, float, bool))})
        try:
            rendered = str(command).format(**values)
        except KeyError as e:
            return {"status": "failed", "error": f"Unknown grade command placeholder: {e}"}

        grade_timeout = int(cfg.get("grade_timeout", 3600))
        if remaining is not None:
            grade_timeout = max(1, min(grade_timeout, int(remaining)))

        self.logger.info("Running optional grade command on host: %s", rendered)
        try:
            proc = subprocess.run(
                rendered,
                shell=True,
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                timeout=grade_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            return {
                "status": "failed",
                "exit_code": -1,
                "error": f"Grade command timed out after {grade_timeout}s",
                "output": out[-20000:],
            }
        out = (proc.stdout or "") + (proc.stderr or "")
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": out[-20000:],
        }

    def _host_workspace_path(self) -> str:
        if not self.run_dir:
            return ""
        task_id = getattr(self, "task_id", None)
        if task_id:
            return str((Path(self.run_dir) / "workspaces" / str(task_id)).resolve())
        return str((Path(self.run_dir) / "workspace").resolve())

    def _host_artifact_path(self, spec: dict[str, Any]) -> str:
        artifact = str(spec.get("artifact_path") or "")
        workspace = str(spec.get("workspace") or "/workspace").rstrip("/")
        host_workspace = self._host_workspace_path()
        if artifact and host_workspace and artifact == workspace:
            return host_workspace
        if artifact and host_workspace and artifact.startswith(workspace + "/"):
            rel = artifact[len(workspace) + 1 :]
            return str((Path(host_workspace) / rel).resolve())
        return artifact

    def _prepared_dir(self, spec: dict[str, Any]) -> str:
        if spec.get("prepared_dir"):
            return str(spec["prepared_dir"])
        host_input = self._host_input_path(spec)
        if not host_input:
            return ""
        path = Path(host_input).resolve()
        if path.name == "public" and path.parent.name == "prepared":
            return str(path.parent)
        if (path.parent / "grade.py").exists():
            return str(path.parent)
        if (path / "grade.py").exists():
            return str(path)
        return ""

    @staticmethod
    def task_index_from_id(task_id: str | None) -> int:
        if not task_id:
            return 0
        match = re.search(r"(\d+)$", str(task_id))
        return int(match.group(1)) if match else 0
