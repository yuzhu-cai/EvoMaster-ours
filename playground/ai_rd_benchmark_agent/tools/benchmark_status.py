"""Benchmark artifact inspection and validation tool."""

from __future__ import annotations

import csv
import io
import shlex
from typing import Any, ClassVar, Literal

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class BenchmarkStatusToolParams(BaseToolParams):
    """Inspect or validate benchmark artifacts in the active workspace.

    Supported benchmarks:
    - mlebench: validates /workspace/submission.csv, optionally against sample_submission.csv.
    - paperbench: validates that /workspace/reproduce.sh exists and is executable.
    - posttrainbench: validates that /workspace/final_model exists and is non-empty.
    """

    name: ClassVar[str] = "benchmark_status"

    action: Literal["inspect", "validate"] = Field(
        default="validate",
        description="Operation to run: inspect workspace inputs or validate final artifact.",
    )
    benchmark: Literal["mlebench", "paperbench", "posttrainbench"] = Field(
        description="Benchmark family for the current task.",
    )
    workspace: str = Field(default="/workspace", description="Writable workspace path.")
    input_dir: str = Field(
        default="",
        description="Read-only input directory inside the container, if applicable.",
    )
    artifact_path: str = Field(
        default="",
        description="Optional explicit artifact path to validate.",
    )


class BenchmarkStatusTool(BaseTool):
    """Lightweight benchmark artifact status tool."""

    name: ClassVar[str] = "benchmark_status"
    params_class: ClassVar[type[BaseToolParams]] = BenchmarkStatusToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, BenchmarkStatusToolParams)
        if params.action == "inspect":
            return self._inspect(session, params)
        return self._validate(session, params)

    def _inspect(self, session, params: BenchmarkStatusToolParams) -> tuple[str, dict[str, Any]]:
        workspace_q = shlex.quote(params.workspace)
        commands = [f"echo 'Workspace:' && pwd && find {workspace_q} -maxdepth 2 -type f | sort | head -200"]
        if params.input_dir:
            input_q = shlex.quote(params.input_dir)
            commands.append(f"echo 'Input:' && find {input_q} -maxdepth 2 -type f | sort | head -200")
        result = session.exec_bash(" && ".join(commands), timeout=60)
        output = result.get("output", "")
        return output or "No files found.", {"action": "inspect", "exit_code": result.get("exit_code")}

    def _validate(self, session, params: BenchmarkStatusToolParams) -> tuple[str, dict[str, Any]]:
        if params.benchmark == "mlebench":
            return self._validate_mlebench(session, params)
        if params.benchmark == "paperbench":
            return self._validate_paperbench(session, params)
        return self._validate_posttrainbench(session, params)

    def _validate_mlebench(self, session, params: BenchmarkStatusToolParams) -> tuple[str, dict[str, Any]]:
        submission_path = params.artifact_path or f"{params.workspace.rstrip('/')}/submission.csv"
        if not session.is_file(submission_path):
            return (
                f"Missing MLE-Bench artifact: {submission_path}. Create this CSV before finishing.",
                {"ok": False, "artifact": submission_path},
            )

        info: dict[str, Any] = {"ok": True, "artifact": submission_path}
        try:
            submission_text = session.read_file(submission_path)
            sub_rows = list(csv.reader(io.StringIO(submission_text)))
            if not sub_rows:
                return "submission.csv is empty.", {"ok": False, "artifact": submission_path}
            info["submission_columns"] = sub_rows[0]
            info["submission_rows"] = max(0, len(sub_rows) - 1)
        except Exception as e:
            return f"Failed to read submission.csv: {e}", {"ok": False, "artifact": submission_path, "error": str(e)}

        sample_path = self._find_mlebench_sample_submission(session, params.input_dir)
        if sample_path and session.is_file(sample_path):
            try:
                sample_text = session.read_file(sample_path)
                sample_rows = list(csv.reader(io.StringIO(sample_text)))
                if sample_rows:
                    info["sample_columns"] = sample_rows[0]
                    info["sample_rows"] = max(0, len(sample_rows) - 1)
                    if sub_rows[0] != sample_rows[0]:
                        return (
                            f"submission.csv columns do not match {sample_path}: "
                            f"got {sub_rows[0]}, expected {sample_rows[0]}",
                            {**info, "ok": False},
                        )
                    if len(sub_rows) != len(sample_rows):
                        return (
                            f"submission.csv row count does not match {sample_path}: "
                            f"got {len(sub_rows) - 1}, expected {len(sample_rows) - 1}",
                            {**info, "ok": False},
                        )
            except Exception as e:
                info["sample_check_error"] = str(e)

        return f"MLE-Bench submission looks valid: {submission_path}\n{info}", info

    def _find_mlebench_sample_submission(self, session, input_dir: str) -> str:
        if not input_dir:
            return ""
        base = input_dir.rstrip("/")
        for filename in (
            "sample_submission.csv",
            "sampleSubmission.csv",
            "sample-submission.csv",
            "sample_submission.CSV",
            "SampleSubmission.csv",
        ):
            sample_path = f"{base}/{filename}"
            if session.is_file(sample_path):
                return sample_path
        return ""

    def _validate_paperbench(self, session, params: BenchmarkStatusToolParams) -> tuple[str, dict[str, Any]]:
        script_path = params.artifact_path or f"{params.workspace.rstrip('/')}/reproduce.sh"
        if not session.is_file(script_path):
            return (
                f"Missing PaperBench artifact: {script_path}. Create a runnable reproduce.sh before finishing.",
                {"ok": False, "artifact": script_path},
            )
        check = session.exec_bash(f"test -x {shlex.quote(script_path)} && echo executable || echo not_executable", timeout=30)
        executable = "executable" in (check.get("stdout", "")) and "not_executable" not in check.get("stdout", "")
        if not executable:
            return (
                f"PaperBench reproduce script exists but is not executable: {script_path}. Run chmod +x reproduce.sh.",
                {"ok": False, "artifact": script_path, "executable": False},
            )
        return f"PaperBench reproduce script is present and executable: {script_path}", {
            "ok": True,
            "artifact": script_path,
            "executable": True,
        }

    def _validate_posttrainbench(self, session, params: BenchmarkStatusToolParams) -> tuple[str, dict[str, Any]]:
        model_dir = params.artifact_path or f"{params.workspace.rstrip('/')}/final_model"
        if not session.is_directory(model_dir):
            return (
                f"Missing PostTrainBench artifact directory: {model_dir}. Save the final model there before finishing.",
                {"ok": False, "artifact": model_dir},
            )
        listing = session.exec_bash(f"find {shlex.quote(model_dir)} -maxdepth 2 -type f | head -50", timeout=30)
        files = [line for line in listing.get("stdout", "").splitlines() if line.strip()]
        if not files:
            return f"PostTrainBench final_model exists but has no files: {model_dir}", {
                "ok": False,
                "artifact": model_dir,
            }
        return f"PostTrainBench final_model exists and is non-empty: {model_dir}\n" + "\n".join(files[:20]), {
            "ok": True,
            "artifact": model_dir,
            "files_preview": files[:20],
        }
