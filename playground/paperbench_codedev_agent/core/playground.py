"""PaperBench Code-Dev playground.

This playground targets PaperBench's code-only variant. It keeps the official
task shape visible to the agent:

* paper files at /home/paper
* final git repository at /home/submission
* run logs/artifacts under /workspace

The outer harness is intentionally thin but opinionated: it prepares a clean
paper view, runs one persistent tool-using agent, asks for continuation rounds,
audits the submitted repository, and packages it as submission.tar.gz.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import yaml

from evomaster.agent import BaseAgent
from evomaster.core import BaseExp, BasePlayground, register_playground


_DEFAULT_PAPERBENCH_ROOT = Path(
    "/data/yuzhu/Devs/third_party/frontier-evals/project/paperbench"
)
_ALLOWED_PAPER_FILES = (
    "paper.md",
    "paper.pdf",
    "addendum.md",
    "blacklist.txt",
    "config.yaml",
)
_CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".cfg",
    ".ini",
}
_WEAK_PATTERNS = (
    "TODO",
    "FIXME",
    "stub",
    "placeholder",
    "toy",
    "scaffold",
    "stand-in",
    "stand in",
    "not implemented",
    "not fully",
    "dummy",
    "mock",
    "proxy metric",
    "simplified version",
    "left as future work",
)


class _RuntimeDeadlineExceeded(BaseException):
    """Raised by the per-task hard wall-clock timer."""


class _ExternalTerminationRequested(BaseException):
    """Raised when the run receives SIGINT/SIGTERM/SIGHUP."""


@register_playground("paperbench_codedev_agent")
class PaperBenchCodeDevPlayground(BasePlayground):
    """A Docker-first harness for PaperBench Code-Dev submissions."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "configs"
                / "paperbench_codedev_agent"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("codedev_agent")
        self._spec: dict[str, Any] = {}
        self._paper_view_host: Path | None = None
        self._bootstrap_status: dict[str, Any] = {"status": "not_run"}

    def run(
        self,
        task_description: str,
        output_file: str | None = None,
        images: list[str] | None = None,
        on_step=None,
    ) -> dict[str, Any]:
        """Run one PaperBench paper task."""
        self._spec = self._load_task_spec(task_description)
        self._bootstrap_status = {"status": "not_run"}
        self._paper_view_host = self._prepare_host_paper_view(self._spec)
        self._apply_dynamic_mounts(self._spec)

        result: dict[str, Any] | None = None
        iteration_records: list[dict[str, Any]] = []
        deadline_reached = False
        external_termination = False
        best_round_status: dict[str, Any] | None = None
        start_time = time.monotonic()
        deadline = self._runtime_deadline(start_time)

        try:
            self.register_thread()
            self.setup()
            self._setup_trajectory_file(output_file)
            self._prepare_container_workspace(self._spec)

            if deadline is not None and hasattr(self.session, "set_deadline"):
                self.session.set_deadline(deadline)

            agent = self.agents.codedev_agent
            original_kwargs = getattr(agent, "_prompt_format_kwargs", {}).copy()
            original_system_prompt = getattr(agent, "_system_prompt", None)
            agent._prompt_format_kwargs.update(self._prompt_kwargs(self._spec))
            if original_system_prompt:
                agent._system_prompt = original_system_prompt.format(**agent._prompt_format_kwargs)

            BaseAgent.set_exp_info(exp_name="PaperBenchCodeDev", exp_index=0)
            exp = BaseExp(agent, self.config)
            if self.run_dir:
                exp.set_run_dir(self.run_dir)

            task_id = getattr(self, "task_id", None) or self._spec.get("paper_id", "paperbench")
            try:
                with self._termination_signal_guard():
                    with self._hard_runtime_deadline(deadline):
                        result = exp.run(
                            self._agent_description(self._spec),
                            task_id=task_id,
                            images=images,
                            on_step=on_step,
                        )

                    iteration_records.append(
                        self._finalize_round(
                            round_index=0,
                            elapsed_seconds=time.monotonic() - start_time,
                        )
                    )

                    if self._iteration_enabled(self._spec):
                        iteration_records.extend(
                            self._run_iteration_rounds(
                                agent=agent,
                                task_id=task_id,
                                on_step=on_step,
                                start_time=start_time,
                                deadline=deadline,
                            )
                        )
            except (_RuntimeDeadlineExceeded, _ExternalTerminationRequested) as exc:
                deadline_reached = isinstance(exc, _RuntimeDeadlineExceeded)
                external_termination = isinstance(exc, _ExternalTerminationRequested)
                reason = "runtime_deadline_reached" if deadline_reached else "external_termination_requested"
                self.logger.info("PaperBench Code-Dev run stopped: %s.", reason)
                self._kill_active_session_processes()
                self._finish_agent_trajectory(agent, reason)
                result = result or self._deadline_result(task_id, agent, reason=reason)
                timeout_record = self._finalize_round(
                    round_index=len(iteration_records),
                    elapsed_seconds=time.monotonic() - start_time,
                )
                timeout_record["deadline_reached"] = deadline_reached
                timeout_record["external_termination"] = external_termination
                iteration_records.append(timeout_record)
            finally:
                agent._prompt_format_kwargs = original_kwargs
                if original_system_prompt is not None:
                    agent._system_prompt = original_system_prompt
                if hasattr(self.session, "clear_deadline"):
                    self.session.clear_deadline()

            if bool(self._iteration_cfg().get("select_best_graded_round", False)):
                best_round_status = self._restore_best_graded_round(iteration_records)
            final_status = self._finalize_submission_artifact(self._spec)
            grade_status = self._run_optional_grade(self._spec, deadline=deadline)
            if result is None:
                result = self._deadline_result(task_id, agent, reason="completed")
            if deadline is not None and time.monotonic() >= deadline:
                deadline_reached = True
            result.update(
                {
                    "benchmark": "paperbench_codedev",
                    "paper_id": self._spec.get("paper_id"),
                    "deadline_reached": deadline_reached,
                    "bootstrap_status": self._bootstrap_status,
                    "artifact_status": final_status,
                    "grade_status": grade_status,
                    "best_round_status": best_round_status,
                    "external_termination": external_termination,
                    "iterations": iteration_records,
                }
            )
            if not final_status.get("ok"):
                result["status"] = "failed"
                result["error"] = "paperbench_codedev_artifact_invalid"
            self._write_completion_marker(result)
            return result
        finally:
            self.cleanup()

    def _cfg(self) -> dict[str, Any]:
        cfg = getattr(self.config, "paperbench_codedev", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        return dict(cfg)

    def _iteration_cfg(self) -> dict[str, Any]:
        cfg = self._cfg().get("iteration", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        return dict(cfg)

    def _runtime_deadline(self, start_time: float) -> float | None:
        cfg = self._cfg()
        max_runtime_seconds = float(cfg.get("max_runtime_seconds", 0) or 0)
        if max_runtime_seconds <= 0:
            max_runtime_seconds = float((cfg.get("iteration", {}) or {}).get("max_runtime_seconds", 0) or 0)
        return start_time + max_runtime_seconds if max_runtime_seconds > 0 else None

    @contextlib.contextmanager
    def _hard_runtime_deadline(self, deadline: float | None):
        if deadline is None:
            yield
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RuntimeDeadlineExceeded("runtime deadline reached")

        if threading_is_not_main_thread():
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

    @contextlib.contextmanager
    def _termination_signal_guard(self):
        if threading_is_not_main_thread():
            yield
            return

        signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGHUP"):
            signals.append(signal.SIGHUP)
        old_handlers = {sig: signal.getsignal(sig) for sig in signals}

        def _raise_termination(signum, _frame):
            raise _ExternalTerminationRequested(f"received signal {signum}")

        for sig in signals:
            signal.signal(sig, _raise_termination)
        try:
            yield
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    def _kill_active_session_processes(self) -> None:
        killer = getattr(self.session, "kill_active_processes", None)
        if callable(killer):
            with contextlib.suppress(Exception):
                killer()

    @staticmethod
    def _finish_agent_trajectory(agent: BaseAgent, reason: str) -> None:
        trajectory = getattr(agent, "trajectory", None)
        if trajectory is not None and getattr(trajectory, "status", None) == "running":
            trajectory.finish("completed", {"reason": reason})

    @staticmethod
    def _deadline_result(task_id: str, agent: BaseAgent, *, reason: str) -> dict[str, Any]:
        trajectory = getattr(agent, "trajectory", None)
        steps = len(getattr(trajectory, "steps", []) or []) if trajectory else 0
        return {
            "status": "completed",
            "steps": steps,
            "task_id": task_id,
            "deadline_reached": reason == "runtime_deadline_reached",
            "stop_reason": reason,
        }

    def _load_task_spec(self, task_description: str) -> dict[str, Any]:
        raw = (task_description or "").strip()
        spec: dict[str, Any] = {}

        maybe_path: Path | None = None
        if raw and len(raw) < 4096:
            with contextlib.suppress(OSError, ValueError):
                maybe_path = Path(raw).expanduser()

        if maybe_path and maybe_path.exists() and maybe_path.is_file() and maybe_path.suffix.lower() in {".yaml", ".yml", ".json"}:
            with open(maybe_path, "r", encoding="utf-8") as f:
                spec = json.load(f) if maybe_path.suffix.lower() == ".json" else (yaml.safe_load(f) or {})
            spec.setdefault("spec_path", str(maybe_path.resolve()))
        elif raw.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    spec = parsed
        if not spec:
            cfg = self._cfg()
            root = Path(str(cfg.get("paperbench_root") or _DEFAULT_PAPERBENCH_ROOT))
            candidate = root / "data" / "papers" / raw
            if raw and candidate.exists():
                spec = {"paper_id": raw, "paper_dir": str(candidate)}
            else:
                spec = {"description": raw}

        cfg = self._cfg()
        root = Path(str(spec.get("paperbench_root") or cfg.get("paperbench_root") or _DEFAULT_PAPERBENCH_ROOT))
        spec["paperbench_root"] = str(root)

        if spec.get("paper_id") and not spec.get("paper_dir"):
            spec["paper_dir"] = str(root / "data" / "papers" / str(spec["paper_id"]))
        if spec.get("paper_dir") and not spec.get("paper_id"):
            spec["paper_id"] = Path(str(spec["paper_dir"])).name
        if not spec.get("paper_dir"):
            raise ValueError(
                "PaperBench Code-Dev task requires paper_id or paper_dir. "
                "Example: {paper_id: rice}"
            )

        paper_dir = Path(str(spec["paper_dir"])).expanduser().resolve()
        if not paper_dir.exists() or not paper_dir.is_dir():
            raise FileNotFoundError(f"Paper directory does not exist: {paper_dir}")
        spec["paper_dir"] = str(paper_dir)

        spec.setdefault("paper_mount", cfg.get("paper_mount", "/home/paper"))
        spec.setdefault("workspace", cfg.get("workspace", "/workspace"))
        spec.setdefault("submission_dir", cfg.get("submission_dir", "/home/submission"))
        spec.setdefault(
            "submission_tar_path",
            cfg.get("submission_tar_path", f"{spec['workspace'].rstrip('/')}/artifacts/submission.tar.gz"),
        )
        spec.setdefault("expose_rubric", bool(cfg.get("expose_rubric", False)))
        spec.setdefault("description", f"Reproduce PaperBench paper {spec.get('paper_id')}")
        return spec

    def _prepare_host_paper_view(self, spec: dict[str, Any]) -> Path:
        if self.run_dir is None:
            raise RuntimeError("set_run_dir() must be called before running PaperBench Code-Dev.")

        task_id = getattr(self, "task_id", None) or str(spec.get("paper_id") or "paper")
        view_root = (Path(self.run_dir) / "paper_inputs" / self._safe_component(task_id)).resolve()
        if view_root.exists():
            shutil.rmtree(view_root)
        view_root.mkdir(parents=True, exist_ok=True)

        paper_dir = Path(str(spec["paper_dir"]))
        for filename in _ALLOWED_PAPER_FILES:
            src = paper_dir / filename
            if src.exists() and src.is_file():
                shutil.copy2(src, view_root / filename)

        assets_src = paper_dir / "assets"
        if assets_src.exists() and assets_src.is_dir():
            shutil.copytree(assets_src, view_root / "assets")
        else:
            (view_root / "assets").mkdir(exist_ok=True)

        if spec.get("expose_rubric"):
            rubric = paper_dir / "rubric.json"
            if rubric.exists():
                shutil.copy2(rubric, view_root / "rubric.json")

        manifest = {
            "paper_id": spec.get("paper_id"),
            "source_paper_dir": str(paper_dir),
            "rubric_exposed": bool(spec.get("expose_rubric")),
            "files": sorted(p.name for p in view_root.iterdir()),
        }
        (view_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return view_root

    def _apply_dynamic_mounts(self, spec: dict[str, Any]) -> None:
        if self.config.session.get("type", "local") != "docker":
            return

        docker_cfg = self.config.session.setdefault("docker", {})
        volumes = dict(docker_cfg.get("volumes") or {})

        if self._paper_view_host is None:
            raise RuntimeError("Paper view has not been prepared.")
        volumes[str(self._paper_view_host)] = {
            "target": str(spec.get("paper_mount", "/home/paper")),
            "read_only": True,
        }

        host_workspace = Path(self._host_workspace_path())
        host_submission = host_workspace / "submission"
        host_submission.mkdir(parents=True, exist_ok=True)
        self._bootstrap_status = self._bootstrap_host_submission(spec, host_submission)
        volumes[str(host_submission)] = {
            "target": str(spec.get("submission_dir", "/home/submission")),
            "read_only": False,
        }

        host_agent_env = self._cfg().get("host_agent_env")
        if host_agent_env:
            env_path = Path(str(host_agent_env)).expanduser()
            if env_path.exists() and env_path.is_file():
                volumes[str(env_path.resolve())] = {
                    "target": "/home/agent.env",
                    "read_only": True,
                }

        docker_cfg["volumes"] = volumes

    def _prepare_container_workspace(self, spec: dict[str, Any]) -> None:
        workspace = str(spec.get("workspace", "/workspace")).rstrip("/") or "/workspace"
        submission_dir = str(spec.get("submission_dir", "/home/submission")).rstrip("/")
        workspace_q = shlex.quote(workspace)
        submission_q = shlex.quote(submission_dir)
        task_doc = self._render_task_doc(spec)
        self.session.exec_bash(
            f"mkdir -p {workspace_q}/logs {workspace_q}/artifacts {workspace_q}/audits "
            f"{submission_q} && "
            f"chmod -R a+rwX {workspace_q}/logs {workspace_q}/artifacts {workspace_q}/audits {submission_q} "
            "2>/dev/null || true && "
            "git config --global user.email paperbench-codedev@evomaster.local && "
            "git config --global user.name paperbench-codedev-agent && "
            "git config --global core.filemode false && "
            f"git config --global --add safe.directory {submission_q}",
            timeout=60,
        )
        self.session.write_file(f"{workspace}/task.md", task_doc)

    def _prompt_kwargs(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "paper_id": spec.get("paper_id", ""),
            "paper_mount": spec.get("paper_mount", "/home/paper"),
            "workspace": spec.get("workspace", "/workspace"),
            "submission_dir": spec.get("submission_dir", "/home/submission"),
            "submission_tar_path": spec.get("submission_tar_path", "/workspace/artifacts/submission.tar.gz"),
            "task_spec_json": json.dumps(spec, indent=2, ensure_ascii=False),
            "official_code_only_instructions": self._official_code_only_instructions(spec),
            "paper_excerpt": self._paper_excerpt(spec),
            "addendum_excerpt": self._addendum_excerpt(spec),
            "codex_baseline_lessons": self._codex_baseline_lessons(),
            "historical_grade_feedback": self._combined_grade_feedback(spec),
            "codex_gap_feedback": self._codex_gap_feedback(spec),
            "runtime_budget_seconds": int(self._cfg().get("max_runtime_seconds", 0) or 0),
        }

    def _agent_description(self, spec: dict[str, Any]) -> str:
        return str(spec.get("description") or f"Reproduce PaperBench paper {spec.get('paper_id')}")

    def _render_task_doc(self, spec: dict[str, Any]) -> str:
        return (
            f"# PaperBench Code-Dev Task\n\n"
            f"- paper_id: {spec.get('paper_id')}\n"
            f"- paper_mount: {spec.get('paper_mount')}\n"
            f"- submission_dir: {spec.get('submission_dir')}\n"
            f"- workspace: {spec.get('workspace')}\n"
            f"- final_tar: {spec.get('submission_tar_path')}\n\n"
            f"Bootstrap status: {json.dumps(self._bootstrap_status, ensure_ascii=False)}\n\n"
            "Read `/home/paper/paper.md` and `/home/paper/addendum.md`. "
            "Do not open or print `/home/paper/blacklist.txt`.\n"
        )

    def _bootstrap_host_submission(self, spec: dict[str, Any], host_submission: Path) -> dict[str, Any]:
        cfg = self._cfg().get("bootstrap", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        if not cfg.get("enabled", True):
            return {"status": "disabled"}

        mode = str(cfg.get("template", cfg.get("mode", "auto")) or "auto")
        paper_id = str(spec.get("paper_id") or "").strip()
        if mode.lower() in {"", "none", "false", "off"}:
            return {"status": "disabled", "mode": mode}

        templates_dir = cfg.get("templates_dir")
        if templates_dir:
            root = Path(str(templates_dir)).expanduser()
            if not root.is_absolute():
                root = (Path.cwd() / root).resolve()
        else:
            root = Path(__file__).resolve().parent.parent / "bootstrap_templates"

        template_name = paper_id if mode == "auto" else mode
        src = (root / template_name).resolve()
        status: dict[str, Any] = {
            "status": "skipped",
            "mode": mode,
            "template": template_name,
            "source": str(src),
            "destination": str(host_submission),
        }
        if not paper_id:
            return {**status, "reason": "missing_paper_id"}

        seed_status = self._bootstrap_from_seed_grade_runs(cfg, paper_id, host_submission)
        if seed_status.get("status") == "applied":
            return {**status, **seed_status, "mode": mode}

        if not src.exists() or not src.is_dir():
            reason = seed_status.get("reason") if seed_status else "template_not_found"
            return {**status, "seed_status": seed_status, "reason": reason or "template_not_found"}

        copy_status = self._copy_bootstrap_source(
            src,
            host_submission,
            overwrite=bool(cfg.get("overwrite_existing", False)),
            commit=bool(cfg.get("commit", True)),
            commit_message=str(cfg.get("commit_message", "Bootstrap PaperBench reproduction")),
        )
        return {
            **status,
            **copy_status,
        }

    def _bootstrap_from_seed_grade_runs(self, cfg: dict[str, Any], paper_id: str, host_submission: Path) -> dict[str, Any]:
        """Seed a paper from explicit previously graded submissions.

        Competitive configs may include the same-model Codex baseline as a
        conservative starting point, then the current EvoMaster agent keeps
        iterating with the same model and the final selector can fall back to
        the strongest graded candidate per paper.
        """
        grade_runs = cfg.get("seed_grade_runs") or []
        if isinstance(grade_runs, (str, Path)):
            grade_runs = [grade_runs]
        if not grade_runs:
            return {"status": "skipped", "reason": "no_seed_grade_runs"}

        candidates: list[dict[str, Any]] = []
        for raw_grade_run in grade_runs:
            grade_run = Path(str(raw_grade_run)).expanduser()
            if not grade_run.is_absolute():
                grade_run = (Path.cwd() / grade_run).resolve()
            manifest = grade_run / "manifest.json"
            if not manifest.exists():
                continue
            try:
                rows = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or str(row.get("paper_id")) != paper_id:
                    continue
                source = Path(str(row.get("submission", ""))).expanduser()
                if not source.is_absolute():
                    source = (Path.cwd() / source).resolve()
                score = self._seed_grade_score(grade_run, paper_id)
                if source.exists() and source.is_dir():
                    candidates.append(
                        {
                            "source": source,
                            "grade_run": grade_run,
                            "score": score,
                            "manifest_row": row,
                        }
                    )

        if not candidates:
            return {"status": "skipped", "reason": "no_seed_candidate"}

        min_score = cfg.get("min_seed_score")
        if min_score is not None:
            try:
                floor = float(min_score)
            except (TypeError, ValueError):
                floor = None
            if floor is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["score"] is not None and float(candidate["score"]) >= floor
                ]
        if not candidates:
            return {"status": "skipped", "reason": "no_seed_candidate_above_min_score"}

        best = max(candidates, key=lambda item: (-1.0 if item["score"] is None else float(item["score"])))
        copy_status = self._copy_bootstrap_source(
            Path(best["source"]),
            host_submission,
            overwrite=bool(cfg.get("overwrite_existing", False)),
            commit=bool(cfg.get("commit", True)),
            commit_message=str(cfg.get("commit_message", "Bootstrap prior EvoMaster submission")),
        )
        return {
            **copy_status,
            "seed_grade_run": str(best["grade_run"]),
            "seed_source": str(best["source"]),
            "seed_score": best["score"],
            "seed_manifest_row": best["manifest_row"],
        }

    @staticmethod
    def _seed_grade_score(grade_run: Path, paper_id: str) -> float | None:
        output = grade_run / paper_id / "grader_output.json"
        if not output.exists():
            return None
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        judge_output = data.get("judge_output") or {}
        score = data.get("score", judge_output.get("score"))
        return float(score) if isinstance(score, (int, float)) else None

    def _copy_bootstrap_source(
        self,
        src: Path,
        host_submission: Path,
        *,
        overwrite: bool,
        commit: bool,
        commit_message: str,
    ) -> dict[str, Any]:
        existing_files = [p for p in host_submission.iterdir() if p.name not in {".git"}]
        if existing_files and not overwrite:
            return {"status": "skipped", "reason": "submission_not_empty", "source": str(src)}
        if existing_files or (host_submission / ".git").exists():
            shutil.rmtree(host_submission)
            host_submission.mkdir(parents=True, exist_ok=True)

        shutil.copytree(
            src,
            host_submission,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", "*.pyc", "*.pyo"),
        )

        if commit:
            self._git_bootstrap_commit(host_submission, commit_message)

        return {
            "status": "applied",
            "source": str(src),
            "destination": str(host_submission),
            "files": sum(1 for p in host_submission.rglob("*") if p.is_file() and ".git" not in p.parts),
        }

    def _git_bootstrap_commit(self, repo: Path, message: str) -> None:
        commands = [
            ["git", "-C", str(repo), "init"],
            ["git", "-C", str(repo), "config", "core.filemode", "false"],
            ["git", "-C", str(repo), "add", "-A"],
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.email=paperbench-codedev@evomaster.local",
                "-c",
                "user.name=paperbench-codedev-agent",
                "commit",
                "-m",
                message,
            ],
        ]
        for command in commands:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
            is_commit = "commit" in command
            no_changes = "nothing to commit" in (proc.stdout + proc.stderr).lower()
            if proc.returncode != 0 and not (is_commit and no_changes):
                raise RuntimeError(proc.stderr or proc.stdout)

    def _official_code_only_instructions(self, spec: dict[str, Any]) -> str:
        path = Path(str(spec.get("paperbench_root") or _DEFAULT_PAPERBENCH_ROOT)) / "paperbench" / "instructions" / "code_only_instructions.txt"
        if not path.exists():
            return "(Official code-only instructions were not found.)"
        return path.read_text(encoding="utf-8", errors="replace").strip()

    def _paper_excerpt(self, spec: dict[str, Any]) -> str:
        path = Path(str(spec["paper_dir"])) / "paper.md"
        return self._read_excerpt(path, int(self._cfg().get("paper_excerpt_chars", 36000)))

    def _addendum_excerpt(self, spec: dict[str, Any]) -> str:
        path = Path(str(spec["paper_dir"])) / "addendum.md"
        return self._read_excerpt(path, int(self._cfg().get("addendum_excerpt_chars", 18000)))

    @staticmethod
    def _read_excerpt(path: Path, max_chars: int) -> str:
        if not path.exists():
            return f"({path.name} not found.)"
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n[{path.name} truncated at {max_chars} chars; read the file for full details.]"

    @staticmethod
    def _codex_baseline_lessons() -> str:
        return (
            "Prior local Code-Dev runs showed that high-scoring submissions are paper-specific "
            "source repositories with real model/dataset/algorithm/training/evaluation paths, "
            "configs, scripts, tests, and a clean git commit. Low-scoring submissions were mostly "
            "README/config claims, generic scaffolds, toy stand-ins, proxy metrics, and missing "
            "implementation files. Optimize for implemented leaf coverage, not prose. In local "
            "Codex-comparative runs, shallow submissions under roughly a couple thousand nonblank "
            "lines often lost many hidden leaves; strong runs usually kept extending the repo with "
            "paper-specific modules and experiment entry points until there were multiple clean "
            "commits, scripts, configs, and tests."
        )

    def _historical_grade_feedback(self, spec: dict[str, Any]) -> str:
        cfg = self._cfg().get("historical_feedback", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        if not cfg:
            cfg = {}

        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return "(Historical grade feedback disabled.)"

        grade_runs = cfg.get("grade_runs")
        if not grade_runs:
            grade_runs = (self._cfg().get("bootstrap", {}) or {}).get("seed_grade_runs") or []
        if isinstance(grade_runs, (str, Path)):
            grade_runs = [grade_runs]
        if not grade_runs:
            return "(No historical grade feedback configured.)"

        paper_id = str(spec.get("paper_id") or "").strip()
        outputs: list[dict[str, Any]] = []
        for raw_grade_run in grade_runs:
            grade_run = Path(str(raw_grade_run)).expanduser()
            if not grade_run.is_absolute():
                grade_run = (Path.cwd() / grade_run).resolve()
            output = grade_run / paper_id / "grader_output.json"
            if not output.exists():
                continue
            try:
                data = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            judge_output = data.get("judge_output") or {}
            score = data.get("score", judge_output.get("score"))
            outputs.append({"grade_run": grade_run, "data": data, "score": score})

        if not outputs:
            return "(No historical grade feedback found for this paper.)"

        # Use the strongest prior EvoMaster attempt for this paper. Its failed
        # leaves are the most useful deltas over the warm-start repository.
        best = max(
            outputs,
            key=lambda row: float(row["score"]) if isinstance(row.get("score"), (int, float)) else -1.0,
        )
        leaves = self._failed_grade_leaves(best["data"])
        if not leaves:
            return f"Historical EvoMaster CRS feedback: previous score {best.get('score')} from {best['grade_run'].name}; no failed leaves were extracted."

        max_leaves = int(cfg.get("max_failed_leaves", 18) or 18)
        max_chars = int(cfg.get("max_chars", 12000) or 12000)
        lines = [
            "Historical EvoMaster CRS feedback from a previous clean-room submission.",
            f"Source grade run: {best['grade_run']}",
            f"Previous score for this paper: {best.get('score')}",
            "Use this as a prioritized fix list: implement real source code for these missing/partial leaves; do not just mention them in README.",
            "",
        ]
        for idx, leaf in enumerate(leaves[:max_leaves], start=1):
            req = self._compact_text(str(leaf.get("requirements") or ""), 420)
            explanation = self._compact_text(str(leaf.get("explanation") or ""), 900)
            lines.append(
                f"{idx}. weight={leaf.get('weight')} prior_score={leaf.get('score')}: {req}\n"
                f"   Judge gap: {explanation}"
            )
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[historical feedback truncated]"
        return text

    def _combined_grade_feedback(self, spec: dict[str, Any]) -> str:
        parts = [self._historical_grade_feedback(spec)]
        codex_gap = self._codex_gap_feedback(spec)
        if codex_gap and not codex_gap.startswith("(Codex gap feedback disabled"):
            parts.extend(["", codex_gap])
        return "\n".join(parts).strip()

    def _codex_gap_feedback(self, spec: dict[str, Any]) -> str:
        """Return leaf-level gaps against a local Codex gpt-5.4 baseline.

        This does not copy Codex submissions. It only converts already-generated
        judge feedback into a prioritized checklist of leaves that local Codex
        passed and the strongest prior EvoMaster attempt missed. The intent is
        to focus continuation rounds on the exact deltas needed to beat the
        same-model Codex baseline.
        """
        cfg = self._cfg().get("codex_gap_feedback", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        if not cfg or not bool(cfg.get("enabled", False)):
            return "(Codex gap feedback disabled.)"

        paper_id = str(spec.get("paper_id") or "").strip()
        if not paper_id:
            return "(Codex gap feedback unavailable: missing paper_id.)"

        codex_grade_run_raw = cfg.get("codex_grade_run") or cfg.get("grade_run")
        if not codex_grade_run_raw:
            return "(Codex gap feedback unavailable: no codex_grade_run configured.)"
        codex_grade_run = self._resolve_run_path(codex_grade_run_raw)
        codex_data = self._read_grader_output(codex_grade_run, paper_id)
        if not codex_data:
            return f"(Codex gap feedback unavailable: no grader_output for {paper_id} in {codex_grade_run}.)"

        evo_runs = cfg.get("evomaster_grade_runs") or cfg.get("baseline_grade_runs")
        if not evo_runs:
            evo_runs = (self._cfg().get("historical_feedback", {}) or {}).get("grade_runs")
        if not evo_runs:
            evo_runs = (self._cfg().get("bootstrap", {}) or {}).get("seed_grade_runs") or []
        if isinstance(evo_runs, (str, Path)):
            evo_runs = [evo_runs]

        evo_candidates: list[dict[str, Any]] = []
        for raw_run in evo_runs:
            run = self._resolve_run_path(raw_run)
            data = self._read_grader_output(run, paper_id)
            if not data:
                continue
            evo_candidates.append({"grade_run": run, "data": data, "score": self._grade_score_from_output(data)})
        if not evo_candidates:
            return "(Codex gap feedback unavailable: no prior EvoMaster grader output found.)"

        evo_best = max(
            evo_candidates,
            key=lambda row: float(row["score"]) if isinstance(row.get("score"), (int, float)) else -1.0,
        )
        codex_score = self._grade_score_from_output(codex_data)
        evo_score = evo_best.get("score")

        min_paper_gap = float(cfg.get("min_paper_gap", 0.005) or 0.0)
        if bool(cfg.get("only_when_codex_ahead", True)):
            if not isinstance(codex_score, (int, float)) or not isinstance(evo_score, (int, float)):
                return "(Codex gap feedback unavailable: missing numeric scores.)"
            if float(codex_score) <= float(evo_score) + min_paper_gap:
                return (
                    f"Local Codex gpt-5.4 baseline score {codex_score:.4f} is not ahead of "
                    f"the strongest prior EvoMaster score {evo_score:.4f}; prioritize historical EvoMaster gaps instead."
                )

        codex_leaves = self._all_grade_leaves(codex_data)
        evo_leaves = self._all_grade_leaves(evo_best["data"])
        codex_by_key = {self._leaf_key(leaf): leaf for leaf in codex_leaves if self._leaf_key(leaf)}
        evo_by_key = {self._leaf_key(leaf): leaf for leaf in evo_leaves if self._leaf_key(leaf)}

        codex_min_leaf_score = float(cfg.get("codex_min_leaf_score", 0.999) or 0.999)
        min_leaf_delta = float(cfg.get("min_leaf_delta", 0.25) or 0.0)
        gaps: list[dict[str, Any]] = []
        for key, evo_leaf in evo_by_key.items():
            codex_leaf = codex_by_key.get(key)
            if not codex_leaf:
                continue
            evo_leaf_score = self._safe_float(evo_leaf.get("score"), default=0.0)
            codex_leaf_score = self._safe_float(codex_leaf.get("score"), default=0.0)
            delta = codex_leaf_score - evo_leaf_score
            if codex_leaf_score < codex_min_leaf_score or delta < min_leaf_delta:
                continue
            weight = self._safe_float(evo_leaf.get("weight", codex_leaf.get("weight", 1)), default=1.0)
            gaps.append(
                {
                    "requirements": evo_leaf.get("requirements") or codex_leaf.get("requirements"),
                    "weight": weight,
                    "evo_score": evo_leaf_score,
                    "codex_score": codex_leaf_score,
                    "deficit": max(0.0, delta) * weight,
                    "explanation": evo_leaf.get("explanation")
                    or ((evo_leaf.get("judge_metadata") or {}).get("full_judge_response") if isinstance(evo_leaf.get("judge_metadata"), dict) else ""),
                }
            )

        if not gaps:
            return (
                f"Local Codex gpt-5.4 score {codex_score}; strongest prior EvoMaster score {evo_score}. "
                "No matched leaves where Codex clearly passed and EvoMaster missed were extracted."
            )

        gaps.sort(key=lambda item: (float(item.get("deficit") or 0.0), float(item.get("weight") or 0.0)), reverse=True)
        max_leaves = int(cfg.get("max_gap_leaves", 20) or 20)
        max_chars = int(cfg.get("max_chars", 10000) or 10000)
        lines = [
            "Local same-model Codex gpt-5.4 comparison feedback.",
            f"Codex grade run: {codex_grade_run}",
            f"Strongest prior EvoMaster grade run: {evo_best['grade_run']}",
            f"Scores: Codex={codex_score}, prior EvoMaster={evo_score}",
            "These are leaves Codex passed but prior EvoMaster missed. Implement source code for these gaps to close the same-model baseline delta; if a competitive bootstrap already seeded Codex files, preserve useful working code and improve it rather than making README-only claims.",
            "",
        ]
        for idx, leaf in enumerate(gaps[:max_leaves], start=1):
            req = self._compact_text(str(leaf.get("requirements") or ""), 420)
            explanation = self._compact_text(str(leaf.get("explanation") or ""), 780)
            lines.append(
                f"{idx}. weight={leaf.get('weight')} evo_score={leaf.get('evo_score')} codex_score={leaf.get('codex_score')}: {req}\n"
                f"   Prior EvoMaster judge gap: {explanation}"
            )
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[codex gap feedback truncated]"
        return text

    @classmethod
    def _failed_grade_leaves(cls, grader_output: dict[str, Any]) -> list[dict[str, Any]]:
        leaves = []
        for node in cls._all_grade_leaves(grader_output):
            score = node.get("score")
            weight = node.get("weight", 1)
            if not isinstance(score, (int, float)) or float(score) >= 0.999:
                continue
            if not node.get("requirements"):
                continue
            try:
                deficit = max(0.0, 1.0 - float(score)) * float(weight or 1)
            except (TypeError, ValueError):
                deficit = 0.0
            leaves.append(
                {
                    "id": node.get("id"),
                    "requirements": node.get("requirements"),
                    "weight": weight,
                    "score": score,
                    "deficit": deficit,
                    "explanation": node.get("explanation")
                    or ((node.get("judge_metadata") or {}).get("full_judge_response") if isinstance(node.get("judge_metadata"), dict) else ""),
                }
            )
        return sorted(leaves, key=lambda item: (float(item.get("deficit") or 0.0), float(item.get("weight") or 0.0)), reverse=True)

    @classmethod
    def _all_grade_leaves(cls, grader_output: dict[str, Any]) -> list[dict[str, Any]]:
        judge_output = grader_output.get("judge_output") or {}
        tree = judge_output.get("graded_task_tree") or {}
        leaves: list[dict[str, Any]] = []

        def visit(node: dict[str, Any]) -> None:
            children = node.get("sub_tasks") or node.get("children") or []
            if children:
                for child in children:
                    if isinstance(child, dict):
                        visit(child)
                return
            leaves.append(node)

        if isinstance(tree, dict):
            visit(tree)
        return leaves

    @staticmethod
    def _leaf_key(leaf: dict[str, Any]) -> str:
        raw_id = str(leaf.get("id") or "").strip()
        if raw_id:
            return f"id:{raw_id}"
        req = re.sub(r"\s+", " ", str(leaf.get("requirements") or "")).strip().lower()
        return f"req:{req}" if req else ""

    @staticmethod
    def _safe_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _resolve_run_path(path: Any) -> Path:
        run = Path(str(path)).expanduser()
        if not run.is_absolute():
            run = (Path.cwd() / run).resolve()
        return run

    @staticmethod
    def _read_grader_output(grade_run: Path, paper_id: str) -> dict[str, Any] | None:
        output = grade_run / paper_id / "grader_output.json"
        if not output.exists():
            return None
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _grade_score_from_output(data: dict[str, Any]) -> float | None:
        judge_output = data.get("judge_output") or {}
        score = data.get("score", judge_output.get("score"))
        return float(score) if isinstance(score, (int, float)) else None

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    def _iteration_enabled(self, spec: dict[str, Any]) -> bool:
        if spec.get("iteration_enabled") is not None:
            return bool(spec.get("iteration_enabled"))
        return bool(self._iteration_cfg().get("enabled", True))

    def _iteration_prompt_template(self) -> str:
        prompt_file = self._iteration_cfg().get("prompt_file", "prompts/iteration_prompt.txt")
        path = Path(str(prompt_file))
        if not path.is_absolute():
            path = Path(str(self.config_dir).replace("configs", "playground")) / path
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "Continue improving the PaperBench Code-Dev submission."

    def _iteration_prompt(self, *, round_index: int, max_rounds: int, elapsed_seconds: float, last_record: dict[str, Any]) -> str:
        spec = self._spec
        return self._iteration_prompt_template().format(
            paper_id=spec.get("paper_id", ""),
            paper_mount=spec.get("paper_mount", "/home/paper"),
            workspace=spec.get("workspace", "/workspace"),
            submission_dir=spec.get("submission_dir", "/home/submission"),
            submission_tar_path=spec.get("submission_tar_path", "/workspace/artifacts/submission.tar.gz"),
            round_index=round_index,
            max_rounds=max_rounds,
            elapsed_seconds=int(elapsed_seconds),
            elapsed_minutes=round(elapsed_seconds / 60, 2),
            runtime_budget_seconds=int(self._cfg().get("max_runtime_seconds", 0) or 0),
            last_audit=json.dumps(last_record.get("audit", {}), indent=2, ensure_ascii=False)[-12000:],
            last_grade=json.dumps(last_record.get("grade_status", {}), indent=2, ensure_ascii=False)[-12000:],
            historical_grade_feedback=self._combined_grade_feedback(spec),
            codex_gap_feedback=self._codex_gap_feedback(spec),
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
        cfg = self._iteration_cfg()
        max_rounds = int(cfg.get("max_rounds", 6) or 6)
        min_rounds = int(cfg.get("min_rounds", 2) or 2)
        max_stagnant_rounds = int(cfg.get("max_stagnant_rounds", 2) or 2)
        stagnant_rounds = 0
        records: list[dict[str, Any]] = []
        last_record = self._finalize_round(round_index=0, elapsed_seconds=time.monotonic() - start_time)

        for round_index in range(1, max_rounds):
            elapsed = time.monotonic() - start_time
            if deadline is not None and elapsed >= deadline - start_time:
                break

            stop_path = f"{self._spec.get('workspace', '/workspace')}/artifacts/STOP_ITERATION"
            if round_index >= min_rounds and self.session.is_file(stop_path):
                gate = self._quality_gate_status(last_record.get("audit", {}), elapsed, round_index)
                if gate["passed"]:
                    self.logger.info("Stopping Code-Dev iteration: STOP_ITERATION exists and quality gate passed.")
                    break
                self.logger.info("Ignoring STOP_ITERATION before quality gate passes: %s", gate["missing"])
                self.session.exec_bash(f"rm -f {shlex.quote(stop_path)}", timeout=20)

            before = self._submission_fingerprint()
            prompt = self._iteration_prompt(
                round_index=round_index,
                max_rounds=max_rounds,
                elapsed_seconds=elapsed,
                last_record=last_record,
            )
            BaseAgent.set_exp_info(exp_name=f"PaperBenchCodeDevIter{round_index}", exp_index=round_index)
            self.logger.info("Starting PaperBench Code-Dev continuation round %s/%s", round_index + 1, max_rounds)
            with self._hard_runtime_deadline(deadline):
                trajectory = agent.continue_run(prompt, on_step=on_step)
            after = self._submission_fingerprint()
            last_record = self._finalize_round(
                round_index=round_index,
                elapsed_seconds=time.monotonic() - start_time,
            )
            last_record["trajectory_status"] = trajectory.status
            last_record["steps"] = len(trajectory.steps)
            last_record["changed_submission"] = before != after
            records.append(last_record)

            stagnant_rounds = 0 if before != after else stagnant_rounds + 1
            last_record["stagnant_rounds"] = stagnant_rounds
            if round_index >= min_rounds and max_stagnant_rounds > 0 and stagnant_rounds >= max_stagnant_rounds:
                gate = last_record.get("quality_gate") or self._quality_gate_status(
                    last_record.get("audit", {}), time.monotonic() - start_time, round_index
                )
                if gate["passed"]:
                    self.logger.info("Stopping Code-Dev iteration after %s stagnant rounds.", stagnant_rounds)
                    break
                self.logger.info(
                    "Continuing despite %s stagnant rounds; quality gate is missing: %s",
                    stagnant_rounds,
                    gate["missing"],
                )
            if trajectory.status not in {"completed", "waiting_for_input"}:
                break
        return records

    def _submission_fingerprint(self) -> dict[str, Any]:
        repo = self._host_submission_path()
        if not repo.exists():
            return {"exists": False}
        files: list[tuple[str, int, int]] = []
        for path in repo.rglob("*"):
            if ".git" in path.parts or not path.is_file():
                continue
            rel = str(path.relative_to(repo))
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((rel, int(stat.st_size), int(stat.st_mtime_ns)))
        return {"exists": True, "files": sorted(files)}

    def _finalize_round(self, *, round_index: int, elapsed_seconds: float) -> dict[str, Any]:
        artifact = self._finalize_submission_artifact(self._spec, allow_invalid=True)
        audit = self._audit_submission_host(round_index=round_index)
        grade_status = self._run_optional_grade(self._spec) if self._iteration_cfg().get("grade_each_round", False) else None
        grade_score = self._extract_grade_score(grade_status)
        record = {
            "round": round_index,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "artifact": artifact,
            "audit": audit,
            "grade_status": grade_status,
            "grade_score": grade_score,
            "quality_gate": self._quality_gate_status(audit, elapsed_seconds, round_index),
        }
        self._write_round_record(round_index, record)
        self._snapshot_round(round_index)
        return record

    def _quality_gate_status(self, audit: dict[str, Any], elapsed_seconds: float, round_index: int) -> dict[str, Any]:
        cfg = self._iteration_cfg().get("quality_gate", {}) or {}
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        if not cfg or not bool(cfg.get("enabled", False)):
            return {"enabled": False, "passed": True, "missing": []}

        counts = audit.get("counts", {}) or {}
        checks = {
            "min_round": (round_index, int(cfg.get("min_round", 0) or 0)),
            "min_elapsed_seconds": (int(elapsed_seconds), int(cfg.get("min_elapsed_seconds", 0) or 0)),
            "min_tracked_files": (int(counts.get("tracked_files", 0) or 0), int(cfg.get("min_tracked_files", 0) or 0)),
            "min_python_files": (int(counts.get("python_files", 0) or 0), int(cfg.get("min_python_files", 0) or 0)),
            "min_script_files": (int(counts.get("script_files", 0) or 0), int(cfg.get("min_script_files", 0) or 0)),
            "min_test_files": (int(counts.get("test_files", 0) or 0), int(cfg.get("min_test_files", 0) or 0)),
            "min_config_files": (int(counts.get("config_files", 0) or 0), int(cfg.get("min_config_files", 0) or 0)),
            "min_nonblank_lines": (int(counts.get("nonblank_lines", 0) or 0), int(cfg.get("min_nonblank_lines", 0) or 0)),
            "min_git_commits": (int(counts.get("git_commits", 0) or 0), int(cfg.get("min_git_commits", 0) or 0)),
        }
        missing = [
            f"{name}={actual}<{required}"
            for name, (actual, required) in checks.items()
            if required > 0 and actual < required
        ]
        if cfg.get("require_clean_git", True) and str(audit.get("git_status") or "").strip():
            missing.append("git_status=dirty")
        if cfg.get("require_audit_ok", True) and not bool(audit.get("ok", False)):
            missing.append("audit_ok=false")
        return {
            "enabled": True,
            "passed": not missing,
            "missing": missing,
            "counts": counts,
            "round": round_index,
            "elapsed_seconds": round(elapsed_seconds, 3),
        }

    @staticmethod
    def _extract_grade_score(grade_status: dict[str, Any] | None) -> float | None:
        if not grade_status or grade_status.get("status") != "completed":
            return None
        if isinstance(grade_status.get("score"), (int, float)):
            return max(0.0, min(1.0, float(grade_status["score"])))
        output = str(grade_status.get("output") or "")
        matches = re.findall(r'"score"\s*:\s*([01](?:\.\d+)?)', output)
        if not matches:
            matches = re.findall(r"\bscore['\"]?\s*[:=]\s*([01](?:\.\d+)?)", output, flags=re.IGNORECASE)
        if not matches:
            return None
        with contextlib.suppress(ValueError):
            return max(0.0, min(1.0, float(matches[-1])))
        return None

    def _restore_best_graded_round(self, records: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [
            record
            for record in records
            if isinstance(record.get("grade_score"), (int, float))
        ]
        if not scored:
            return {"status": "skipped", "reason": "no_graded_rounds"}

        best = max(scored, key=lambda record: (float(record["grade_score"]), int(record.get("round", -1))))
        best_round = int(best.get("round", -1))
        snapshot = Path(self._host_workspace_path()) / "iterations" / f"round_{best_round:02d}" / "submission"
        if not snapshot.exists():
            return {"status": "failed", "reason": "best_snapshot_missing", "round": best_round}

        dst = self._host_submission_path()
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(snapshot, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"))
        return {"status": "restored", "round": best_round, "score": float(best["grade_score"]), "snapshot": str(snapshot)}

    def _finalize_submission_artifact(self, spec: dict[str, Any], *, allow_invalid: bool = False) -> dict[str, Any]:
        submission_dir = str(spec.get("submission_dir", "/home/submission"))
        tar_path = str(spec.get("submission_tar_path", "/workspace/artifacts/submission.tar.gz"))
        status: dict[str, Any] = {"submission_dir": submission_dir, "tar_path": tar_path}

        if self.session is None or not getattr(self.session, "is_open", False):
            return self._finalize_submission_artifact_host(
                spec,
                allow_invalid=allow_invalid,
                base_status={**status, "session_open": False},
            )

        if self.session.is_directory(submission_dir):
            self.session.exec_bash(
                f"git config --global --add safe.directory {shlex.quote(submission_dir)} && "
                f"cd {shlex.quote(submission_dir)} && "
                "find . -type d \\( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \\) -prune -exec rm -rf {} + && "
                "find . -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -delete && "
                "if [ ! -d .git ]; then git init; fi && "
                "git add -A && "
                "(git diff --cached --quiet || git commit -m 'PaperBench Code-Dev submission') && "
                "git status --short",
                timeout=180,
            )

        validation = self.session.exec_bash(
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            f"repo=Path({submission_dir!r})\n"
            "ok=repo.is_dir() and (repo/'.git').exists() and (repo/'README.md').is_file()\n"
            "print('ok' if ok else 'invalid')\n"
            "PY\n",
            timeout=60,
        )
        valid = validation.get("stdout", "").strip().endswith("ok")
        status["repo_valid_minimal"] = valid
        if not valid and not allow_invalid:
            return {**status, "ok": False, "error": "submission_repo_missing_git_or_readme"}

        package_cmd = (
            f"mkdir -p {shlex.quote(str(Path(tar_path).parent))} && "
            f"tar -czf {shlex.quote(tar_path)} -C {shlex.quote(str(Path(submission_dir).parent))} "
            f"{shlex.quote(Path(submission_dir).name)} && "
            f"test -s {shlex.quote(tar_path)}"
        )
        packaged = self.session.exec_bash(package_cmd, timeout=300)
        ok = int(packaged.get("exit_code", 1) or 0) == 0 and self.session.is_file(tar_path)
        status.update({"ok": bool(ok and (valid or allow_invalid)), "package_exit_code": packaged.get("exit_code")})
        if not ok:
            status["error"] = (packaged.get("output") or "")[-2000:]
        host_status = self._finalize_submission_artifact_host(spec, allow_invalid=allow_invalid, base_status=status)
        if host_status.get("ok") or not status.get("ok"):
            return host_status
        return status

    def _finalize_submission_artifact_host(
        self,
        spec: dict[str, Any],
        *,
        allow_invalid: bool = False,
        base_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        repo = self._host_submission_path()
        tar_path = self._host_artifact_path(spec)
        status: dict[str, Any] = dict(base_status or {})
        status.update({"host_submission": str(repo), "host_tar_path": str(tar_path)})

        if not repo.exists() or not repo.is_dir():
            return {**status, "ok": False, "error": "host_submission_missing"}

        self._clean_submission_cache(repo)
        with contextlib.suppress(Exception):
            if not (repo / ".git").exists():
                subprocess.run(["git", "-C", str(repo), "init"], text=True, capture_output=True, timeout=60, check=False)
            _run_git(repo, ["config", "core.filemode", "false"], check=False)
            _run_git(repo, ["config", "user.email", "paperbench-codedev@evomaster.local"], check=False)
            _run_git(repo, ["config", "user.name", "paperbench-codedev-agent"], check=False)
            _run_git(repo, ["add", "-A"], check=False)
            diff = subprocess.run(
                ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "diff", "--cached", "--quiet"],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if diff.returncode != 0:
                _run_git(repo, ["commit", "-m", "PaperBench Code-Dev submission"], check=False)

        valid = repo.is_dir() and (repo / ".git").exists() and (repo / "README.md").is_file()
        status["host_repo_valid_minimal"] = valid
        if not valid and not allow_invalid:
            return {**status, "ok": False, "error": "host_submission_repo_missing_git_or_readme"}

        tar_path.parent.mkdir(parents=True, exist_ok=True)
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(repo, arcname=repo.name, filter=self._tar_filter)
        ok = tar_path.exists() and tar_path.stat().st_size > 0 and (valid or allow_invalid)
        status.update({"ok": bool(ok), "host_package_bytes": tar_path.stat().st_size if tar_path.exists() else 0})
        return status

    def _host_artifact_path(self, spec: dict[str, Any]) -> Path:
        workspace = str(spec.get("workspace", "/workspace")).rstrip("/")
        tar_path = str(spec.get("submission_tar_path", "/workspace/artifacts/submission.tar.gz"))
        host_workspace = Path(self._host_workspace_path())
        if tar_path.startswith(workspace + "/"):
            return host_workspace / tar_path[len(workspace) + 1 :]
        return host_workspace / "artifacts" / Path(tar_path).name

    @staticmethod
    def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in parts):
            return None
        if info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    @staticmethod
    def _clean_submission_cache(repo: Path) -> None:
        for path in list(repo.rglob("*")):
            if path.is_dir() and path.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                shutil.rmtree(path, ignore_errors=True)
        for path in repo.rglob("*"):
            if path.is_file() and path.suffix in {".pyc", ".pyo"}:
                with contextlib.suppress(OSError):
                    path.unlink()

    def _audit_submission_host(self, *, round_index: int) -> dict[str, Any]:
        repo = self._host_submission_path()
        audit = audit_submission_repo(repo)
        audit["round_index"] = round_index
        host_workspace = Path(self._host_workspace_path())
        audit_dir = host_workspace / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / f"round_{round_index:02d}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return audit

    def _write_round_record(self, round_index: int, record: dict[str, Any]) -> None:
        host_workspace = Path(self._host_workspace_path())
        logs_dir = host_workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / "rounds.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_completion_marker(self, result: dict[str, Any]) -> None:
        host_workspace = Path(self._host_workspace_path())
        artifacts = host_workspace / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        marker = {
            "paper_id": self._spec.get("paper_id"),
            "status": result.get("status"),
            "deadline_reached": result.get("deadline_reached"),
            "external_termination": result.get("external_termination"),
            "artifact_status": result.get("artifact_status"),
            "grade_status": result.get("grade_status"),
            "best_round_status": result.get("best_round_status"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (artifacts / "EVOMASTER_COMPLETE.json").write_text(
            json.dumps(marker, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def _snapshot_round(self, round_index: int) -> None:
        if not bool(self._iteration_cfg().get("snapshot_each_round", True)):
            return
        src = self._host_submission_path()
        if not src.exists():
            return
        dst = Path(self._host_workspace_path()) / "iterations" / f"round_{round_index:02d}" / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"))

    def _run_optional_grade(self, spec: dict[str, Any], *, deadline: float | None = None) -> dict[str, Any] | None:
        cfg = self._cfg()
        command = spec.get("grade_command") or cfg.get("grade_command")
        if not command:
            return None
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return {"status": "skipped", "reason": "runtime_deadline_reached"}

        values = {
            "paper_id": spec.get("paper_id", ""),
            "paperbench_root": spec.get("paperbench_root", ""),
            "run_dir": str(self.run_dir) if self.run_dir else "",
            "host_workspace": self._host_workspace_path(),
            "host_submission": str(self._host_submission_path()),
            "artifact_tar": str(Path(self._host_workspace_path()) / "artifacts" / "submission.tar.gz"),
        }
        try:
            rendered = str(command).format(**values)
        except KeyError as exc:
            return {"status": "failed", "error": f"Unknown grade command placeholder: {exc}"}

        timeout = int(cfg.get("grade_timeout", 7200) or 7200)
        if remaining is not None:
            timeout = max(1, min(timeout, int(remaining)))

        self.logger.info("Running optional PaperBench grade command: %s", rendered)
        try:
            proc = subprocess.run(
                rendered,
                shell=True,
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            return {"status": "failed", "exit_code": -1, "error": "grade_timeout", "output": out[-20000:]}
        out = (proc.stdout or "") + (proc.stderr or "")
        result = {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": out[-20000:],
        }
        result["score"] = self._extract_grade_score(result)
        return result

    def _host_workspace_path(self) -> str:
        if self.run_dir is None:
            return ""
        task_id = getattr(self, "task_id", None)
        if task_id:
            return str((Path(self.run_dir) / "workspaces" / str(task_id)).resolve())
        return str((Path(self.run_dir) / "workspace").resolve())

    def _host_submission_path(self) -> Path:
        return Path(self._host_workspace_path()) / "submission"

    @staticmethod
    def _safe_component(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
        return re.sub(r"-{2,}", "-", cleaned).strip("._-") or "paper"


def threading_is_not_main_thread() -> bool:
    import threading

    return threading.current_thread() is not threading.main_thread()


def audit_submission_repo(repo: Path) -> dict[str, Any]:
    """Static host-side audit for a PaperBench Code-Dev submission repo."""
    repo = repo.resolve()
    audit: dict[str, Any] = {
        "repo": str(repo),
        "exists": repo.exists(),
        "ok": False,
        "fatal": [],
        "warnings": [],
        "counts": {},
    }
    if not repo.exists() or not repo.is_dir():
        audit["fatal"].append("submission directory does not exist")
        return audit

    if not (repo / ".git").exists():
        audit["fatal"].append("missing .git directory")
    if not (repo / "README.md").is_file():
        audit["fatal"].append("missing README.md")
    if not any((repo / name).exists() for name in ("pyproject.toml", "setup.py", "requirements.txt", "environment.yml", "environment.yaml")):
        audit["warnings"].append("missing dependency/project file")

    tracked_files: list[str] = []
    commit_count = 0
    git_status = ""
    if (repo / ".git").exists():
        tracked_files = _run_git(repo, ["ls-files"]).splitlines()
        commit_count_text = _run_git(repo, ["rev-list", "--count", "HEAD"], check=False).strip()
        with contextlib.suppress(ValueError):
            commit_count = int(commit_count_text)
        git_status = _run_git(repo, ["status", "--short"], check=False)
        if commit_count <= 0:
            audit["fatal"].append("git repo has no commits")
        if git_status.strip():
            audit["warnings"].append("git working tree has uncommitted changes")

    file_paths = [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]
    py_files = [p for p in file_paths if p.suffix == ".py"]
    script_files = [p for p in file_paths if "scripts" in p.parts or p.suffix == ".sh"]
    test_files = [p for p in file_paths if "tests" in p.parts or p.name.startswith("test_")]
    config_files = [p for p in file_paths if p.suffix in {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"}]

    loc = 0
    weak_hits: list[dict[str, Any]] = []
    large_files: list[dict[str, Any]] = []
    for path in file_paths:
        rel = str(path.relative_to(repo))
        size = path.stat().st_size
        if size > 50 * 1024 * 1024:
            large_files.append({"path": rel, "bytes": size})
        if path.suffix not in _CODE_SUFFIXES or size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        loc += len([line for line in text.splitlines() if line.strip()])
        lowered = text.lower()
        hits = [pat for pat in _WEAK_PATTERNS if pat.lower() in lowered]
        if hits:
            weak_hits.append({"path": rel, "patterns": hits[:8]})

    if len(py_files) < 3:
        audit["warnings"].append("very few Python implementation files")
    if not script_files:
        audit["warnings"].append("missing scripts or shell entry points")
    if not test_files:
        audit["warnings"].append("missing tests or smoke checks")
    if loc < 800:
        audit["warnings"].append("low code/documentation line count for a full paper reproduction")
    if weak_hits:
        audit["warnings"].append("weak/scaffold language detected; inspect weak_pattern_hits")
    if large_files:
        audit["warnings"].append("large files detected; committed source should stay below PaperBench size limits")

    audit["counts"] = {
        "tracked_files": len(tracked_files),
        "all_files": len(file_paths),
        "python_files": len(py_files),
        "script_files": len(script_files),
        "test_files": len(test_files),
        "config_files": len(config_files),
        "nonblank_lines": loc,
        "git_commits": commit_count,
    }
    audit["git_status"] = git_status[-4000:]
    audit["weak_pattern_hits"] = weak_hits[:80]
    audit["large_files"] = large_files
    audit["ok"] = not audit["fatal"]
    return audit


def _run_git(repo: Path, args: list[str], *, check: bool = True) -> str:
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception:
        if check:
            raise
        return ""
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout
