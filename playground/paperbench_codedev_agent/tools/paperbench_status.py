"""PaperBench Code-Dev submission inspection tool."""

from __future__ import annotations

import json
import shlex
from typing import Any, ClassVar, Literal

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class PaperBenchStatusToolParams(BaseToolParams):
    """Inspect, audit, validate, or package a PaperBench Code-Dev submission.

    Use this before finish. The final repository must be under submission_dir,
    must be a git repository, must contain README.md, and should contain real
    code paths, scripts, configs, and tests.
    """

    name: ClassVar[str] = "paperbench_status"

    action: Literal["inspect", "audit", "validate", "package"] = Field(
        default="audit",
        description="Operation to run.",
    )
    paper_dir: str = Field(default="/home/paper", description="Read-only PaperBench paper directory.")
    submission_dir: str = Field(default="/home/submission", description="Submission git repository path.")
    workspace: str = Field(default="/workspace", description="Writable workspace path.")
    artifact_path: str = Field(default="/workspace/artifacts/submission.tar.gz", description="Output tar.gz path.")


class PaperBenchStatusTool(BaseTool):
    """Status tool for PaperBench Code-Dev repositories."""

    name: ClassVar[str] = "paperbench_status"
    params_class: ClassVar[type[BaseToolParams]] = PaperBenchStatusToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as exc:
            return f"Parameter validation error: {exc}", {"error": str(exc)}

        assert isinstance(params, PaperBenchStatusToolParams)
        if params.action == "inspect":
            return self._inspect(session, params)
        if params.action == "package":
            return self._package(session, params)
        if params.action == "validate":
            observation, info = self._audit(session, params)
            if info.get("ok"):
                return "Validation passed.\n\n" + observation, info
            return "Validation failed.\n\n" + observation, info
        return self._audit(session, params)

    def _inspect(self, session, params: PaperBenchStatusToolParams) -> tuple[str, dict[str, Any]]:
        cmd = (
            f"echo 'Paper files (blacklist content intentionally not printed):' && "
            f"find {shlex.quote(params.paper_dir)} -maxdepth 2 -type f "
            f"! -name blacklist.txt | sort | sed -n '1,200p' && "
            f"echo '\nSubmission files:' && "
            f"find {shlex.quote(params.submission_dir)} -maxdepth 3 -type f "
            f"! -path '*/.git/*' | sort | sed -n '1,240p'"
        )
        result = session.exec_bash(cmd, timeout=60)
        return result.get("output", ""), {"exit_code": result.get("exit_code")}

    def _audit(self, session, params: PaperBenchStatusToolParams) -> tuple[str, dict[str, Any]]:
        script = f"""python - <<'PY'
import json, os, re, subprocess
from pathlib import Path

repo = Path({params.submission_dir!r})
weak = {[
    "TODO", "FIXME", "stub", "placeholder", "toy", "scaffold", "stand-in",
    "stand in", "not implemented", "not fully", "dummy", "mock",
    "proxy metric", "simplified version", "left as future work",
]!r}
code_suffixes = {{'.py','.sh','.toml','.yaml','.yml','.json','.md','.cfg','.ini'}}
audit = {{"repo": str(repo), "exists": repo.exists(), "ok": False, "fatal": [], "warnings": [], "counts": {{}}}}
if not repo.exists() or not repo.is_dir():
    audit["fatal"].append("submission directory does not exist")
else:
    if not (repo / ".git").exists():
        audit["fatal"].append("missing .git directory")
    if not (repo / "README.md").is_file():
        audit["fatal"].append("missing README.md")
    if not any((repo / n).exists() for n in ("pyproject.toml","setup.py","requirements.txt","environment.yml","environment.yaml")):
        audit["warnings"].append("missing dependency/project file")
    tracked = []
    commits = 0
    git_status = ""
    if (repo / ".git").exists():
        def git(*args):
            p = subprocess.run(["git","-c",f"safe.directory={{repo}}","-C",str(repo),*args], text=True, capture_output=True, timeout=30)
            return p.stdout if p.returncode == 0 else ""
        tracked = [x for x in git("ls-files").splitlines() if x.strip()]
        try:
            commits = int((git("rev-list","--count","HEAD") or "0").strip())
        except Exception:
            commits = 0
        git_status = git("status","--short")
        if commits <= 0:
            audit["fatal"].append("git repo has no commits")
        if git_status.strip():
            audit["warnings"].append("git working tree has uncommitted changes")
    files = [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]
    py_files = [p for p in files if p.suffix == ".py"]
    script_files = [p for p in files if "scripts" in p.parts or p.suffix == ".sh"]
    test_files = [p for p in files if "tests" in p.parts or p.name.startswith("test_")]
    config_files = [p for p in files if p.suffix in {{".yaml",".yml",".toml",".json",".ini",".cfg"}}]
    loc = 0
    weak_hits = []
    large_files = []
    for path in files:
        rel = str(path.relative_to(repo))
        size = path.stat().st_size
        if size > 50 * 1024 * 1024:
            large_files.append({{"path": rel, "bytes": size}})
        if path.suffix not in code_suffixes or size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        loc += len([line for line in text.splitlines() if line.strip()])
        low = text.lower()
        hits = [pat for pat in weak if pat.lower() in low]
        if hits:
            weak_hits.append({{"path": rel, "patterns": hits[:8]}})
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
    audit["counts"] = {{
        "tracked_files": len(tracked),
        "all_files": len(files),
        "python_files": len(py_files),
        "script_files": len(script_files),
        "test_files": len(test_files),
        "config_files": len(config_files),
        "nonblank_lines": loc,
        "git_commits": commits,
    }}
    audit["git_status"] = git_status[-4000:]
    audit["weak_pattern_hits"] = weak_hits[:80]
    audit["large_files"] = large_files
audit["ok"] = not audit["fatal"]
print(json.dumps(audit, indent=2))
PY"""
        result = session.exec_bash(script, timeout=120)
        raw = result.get("stdout") or result.get("output") or ""
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = {"ok": False, "fatal": ["audit_json_parse_failed"], "raw": raw[-4000:]}
        status = "OK" if info.get("ok") else "NOT OK"
        observation = f"PaperBench Code-Dev audit: {status}\n" + json.dumps(info, indent=2, ensure_ascii=False)
        return observation, info

    def _package(self, session, params: PaperBenchStatusToolParams) -> tuple[str, dict[str, Any]]:
        cmd = (
            f"mkdir -p {shlex.quote(params.workspace.rstrip('/') + '/artifacts')} && "
            f"git config --global --add safe.directory {shlex.quote(params.submission_dir)} && "
            f"cd {shlex.quote(params.submission_dir)} && "
            "if [ ! -d .git ]; then git init; fi && "
            "git add -A && "
            "(git diff --cached --quiet || git commit -m 'PaperBench Code-Dev submission') && "
            f"tar -czf {shlex.quote(params.artifact_path)} -C "
            f"{shlex.quote(str(PathLikeParent(params.submission_dir)))} "
            f"{shlex.quote(str(PathLikeName(params.submission_dir)))} && "
            f"test -s {shlex.quote(params.artifact_path)} && "
            f"ls -lh {shlex.quote(params.artifact_path)}"
        )
        result = session.exec_bash(cmd, timeout=300)
        ok = int(result.get("exit_code", 1) or 0) == 0 and session.is_file(params.artifact_path)
        return result.get("output", ""), {"ok": ok, "artifact_path": params.artifact_path, "exit_code": result.get("exit_code")}


def PathLikeParent(path: str) -> str:
    parts = path.rstrip("/").rsplit("/", 1)
    return parts[0] if len(parts) == 2 and parts[0] else "/"


def PathLikeName(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1] or "submission"
