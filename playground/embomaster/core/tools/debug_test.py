"""Debug test tool for EmboMaster playground.

Runs quick validation commands in a controlled working directory.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from ..services import K8SExperimentRunner


class DebugTestToolParams(BaseToolParams):
    """Execute a debug validation command in the workspace.

    This tool is intended for short validation runs after code edits.
    """

    name: ClassVar[str] = "debug_test"

    command: str = Field(description="Shell command to execute.")
    timeout: int = Field(
        default=120,
        ge=5,
        le=3600,
        description="Hard timeout in seconds.",
    )
    working_dir: str = Field(
        default="",
        description="Optional working directory. Relative paths are resolved from workspace root.",
    )
    env_init: str = Field(
        default="",
        description="Optional shell init command, e.g. conda activation.",
    )
    fresh_pod: bool = Field(
        default=False,
        description=(
            "Only for k8s debug pod mode. If true, recreate a clean debug pod before "
            "running the command and delete it after execution. Use for stateful or "
            "complex validations such as training, evaluation, package installation, "
            "or commands that may leave residual processes or files."
        ),
    )


class DebugTestTool(BaseTool):
    """Tool wrapper for quick debug test commands."""

    name: ClassVar[str] = "debug_test"
    params_class: ClassVar[type[BaseToolParams]] = DebugTestToolParams

    def __init__(
        self,
        default_timeout: int = 120,
        default_env_init: str = "",
        k8s_runner: "K8SExperimentRunner | None" = None,
        use_k8s_debug_pod: bool = False,
        k8s_fallback_to_local: bool = True,
    ) -> None:
        super().__init__()
        self.default_timeout = default_timeout
        self.default_env_init = default_env_init
        self.k8s_runner = k8s_runner
        self.use_k8s_debug_pod = use_k8s_debug_pod
        self.k8s_fallback_to_local = k8s_fallback_to_local

    def configure_k8s_debug(
        self,
        k8s_runner: "K8SExperimentRunner | None",
        use_k8s_debug_pod: bool,
        k8s_fallback_to_local: bool,
    ) -> None:
        self.k8s_runner = k8s_runner
        self.use_k8s_debug_pod = use_k8s_debug_pod
        self.k8s_fallback_to_local = k8s_fallback_to_local

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}

        assert isinstance(params, DebugTestToolParams)

        workspace = Path(getattr(session.config, "workspace_path", ".")).resolve()
        if params.working_dir:
            working_path = Path(params.working_dir)
            if working_path.is_absolute():
                run_dir = working_path
            else:
                run_dir = workspace / working_path
        else:
            run_dir = workspace

        timeout = params.timeout or self.default_timeout
        env_init = (params.env_init or self.default_env_init).strip()

        if self._can_use_k8s_debug_pod():
            try:
                assert self.k8s_runner is not None
                k8s_result = self.k8s_runner.exec_debug_command(
                    command=params.command,
                    timeout=timeout,
                    working_dir=str(run_dir),
                    env_init=env_init,
                    workspace_path=str(workspace),
                    fresh_pod=bool(params.fresh_pod),
                )
                return self._build_observation(
                    result=k8s_result,
                    run_dir=run_dir,
                    command=params.command,
                    timeout=timeout,
                    mode="k8s_debug_pod",
                )
            except Exception as e:
                if not self.k8s_fallback_to_local:
                    return (
                        f"[debug_test] k8s_debug_pod error: {str(e)}",
                        {"tool": self.name, "mode": "k8s_debug_pod", "error": str(e)},
                    )
                observation, info = self._execute_local(
                    session=session,
                    run_dir=run_dir,
                    command=params.command,
                    timeout=timeout,
                    env_init=env_init,
                )
                info["mode"] = "local_fallback"
                info["k8s_error"] = str(e)
                observation = (
                    f"[debug_test] k8s_debug_pod unavailable, fallback to local: {str(e)}\n"
                    f"{observation}"
                )
                return observation, info

        return self._execute_local(
            session=session,
            run_dir=run_dir,
            command=params.command,
            timeout=timeout,
            env_init=env_init,
        )

    def _can_use_k8s_debug_pod(self) -> bool:
        if not self.use_k8s_debug_pod or self.k8s_runner is None:
            return False
        if not hasattr(self.k8s_runner, "is_debug_pod_enabled"):
            return False
        try:
            return bool(self.k8s_runner.is_debug_pod_enabled())
        except Exception:
            return False

    def _execute_local(
        self,
        session: BaseSession,
        run_dir: Path,
        command: str,
        timeout: int,
        env_init: str,
    ) -> tuple[str, dict[str, Any]]:
        cmd_parts: list[str] = []
        if env_init:
            cmd_parts.append(env_init)
        cmd_parts.append(f"cd {shlex.quote(str(run_dir))}")
        cmd_parts.append(command)
        full_command = " && ".join(cmd_parts)

        result = session.exec_bash(full_command, timeout=timeout)
        result["full_command"] = full_command
        return self._build_observation(
            result=result,
            run_dir=run_dir,
            command=command,
            timeout=timeout,
            mode="local",
        )

    def _build_observation(
        self,
        result: dict[str, Any],
        run_dir: Path,
        command: str,
        timeout: int,
        mode: str,
    ) -> tuple[str, dict[str, Any]]:
        output = result.get("output", "") or result.get("stdout", "")
        exit_code = int(result.get("exit_code", -1))
        status_line = (
            f"[debug_test] success (exit_code=0) mode={mode} working_dir={run_dir}"
            if exit_code == 0
            else f"[debug_test] failed (exit_code={exit_code}) mode={mode} working_dir={run_dir}"
        )
        observation = f"{status_line}\n{output}".strip()

        info = {
            "tool": self.name,
            "mode": mode,
            "exit_code": exit_code,
            "working_dir": str(run_dir),
            "command": command,
            "full_command": result.get("full_command", result.get("command", "")),
            "timeout": timeout,
        }
        if "pod_name" in result:
            info["pod_name"] = result.get("pod_name")
        if "namespace" in result:
            info["namespace"] = result.get("namespace")
        return observation, info
