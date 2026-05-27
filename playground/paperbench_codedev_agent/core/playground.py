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
            except _RuntimeDeadlineExceeded:
                deadline_reached = True
                self.logger.info("PaperBench Code-Dev runtime deadline reached.")
                self._kill_active_session_processes()
                self._finish_agent_trajectory_due_to_deadline(agent)
                result = result or self._deadline_result(task_id, agent)
                timeout_record = self._finalize_round(
                    round_index=len(iteration_records),
                    elapsed_seconds=time.monotonic() - start_time,
                )
                timeout_record["deadline_reached"] = True
                iteration_records.append(timeout_record)
            finally:
                agent._prompt_format_kwargs = original_kwargs
                if original_system_prompt is not None:
                    agent._system_prompt = original_system_prompt
                if hasattr(self.session, "clear_deadline"):
                    self.session.clear_deadline()

            final_status = self._finalize_submission_artifact(self._spec)
            grade_status = self._run_optional_grade(self._spec, deadline=deadline)
            if result is None:
                result = self._deadline_result(task_id, agent)
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
                    "iterations": iteration_records,
                }
            )
            if not final_status.get("ok"):
                result["status"] = "failed"
                result["error"] = "paperbench_codedev_artifact_invalid"
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

    def _kill_active_session_processes(self) -> None:
        killer = getattr(self.session, "kill_active_processes", None)
        if callable(killer):
            with contextlib.suppress(Exception):
                killer()

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
        if not src.exists() or not src.is_dir():
            return {**status, "reason": "template_not_found"}

        existing_files = [p for p in host_submission.iterdir() if p.name not in {".git"}]
        overwrite = bool(cfg.get("overwrite_existing", False))
        if existing_files and not overwrite:
            return {**status, "reason": "submission_not_empty"}
        if existing_files or (host_submission / ".git").exists():
            shutil.rmtree(host_submission)
            host_submission.mkdir(parents=True, exist_ok=True)

        shutil.copytree(
            src,
            host_submission,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".git"),
        )

        if bool(cfg.get("commit", True)):
            self._git_bootstrap_commit(host_submission, str(cfg.get("commit_message", "Bootstrap PaperBench reproduction")))

        return {
            **status,
            "status": "applied",
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
            "implementation files. Optimize for implemented leaf coverage, not prose."
        )

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

            if round_index >= min_rounds and self.session.is_file(f"{self._spec.get('workspace', '/workspace')}/artifacts/STOP_ITERATION"):
                self.logger.info("Stopping Code-Dev iteration: STOP_ITERATION exists.")
                break

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
                self.logger.info("Stopping Code-Dev iteration after %s stagnant rounds.", stagnant_rounds)
                break
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
        record = {
            "round": round_index,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "artifact": artifact,
            "audit": audit,
            "grade_status": grade_status,
        }
        self._write_round_record(round_index, record)
        self._snapshot_round(round_index)
        return record

    def _finalize_submission_artifact(self, spec: dict[str, Any], *, allow_invalid: bool = False) -> dict[str, Any]:
        submission_dir = str(spec.get("submission_dir", "/home/submission"))
        tar_path = str(spec.get("submission_tar_path", "/workspace/artifacts/submission.tar.gz"))
        status: dict[str, Any] = {"submission_dir": submission_dir, "tar_path": tar_path}

        if self.session is None or not getattr(self.session, "is_open", False):
            return {**status, "ok": False, "error": "session_not_open"}

        if self.session.is_directory(submission_dir):
            self.session.exec_bash(
                f"git config --global --add safe.directory {shlex.quote(submission_dir)} && "
                f"cd {shlex.quote(submission_dir)} && "
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
        return status

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
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))

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
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": out[-20000:],
        }

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
