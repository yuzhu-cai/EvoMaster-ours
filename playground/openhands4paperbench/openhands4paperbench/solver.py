from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

import blobfile as bf
import chz
import structlog.stdlib
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from typing_extensions import override

from paperbench.constants import AGENT_DIR_CONFIG, LOGS_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.upload import upload_heavy_logs, upload_status
from paperbench.solvers.utils import check_for_existing_run

logger = structlog.stdlib.get_logger(component=__name__)


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


@chz.chz
class OpenHandsCliSolver(BasePBSolver):
    """PaperBench solver that runs the OpenHands CLI inside the agent container."""

    openhands_bin: str = chz.field(default="openhands")
    model: str | None = chz.field(default=None)
    base_url: str | None = chz.field(default=None)
    reasoning_effort: str | None = chz.field(default="medium")
    native_tool_calling: bool = chz.field(default=True)
    force_chat_completion: bool = chz.field(default=True)
    use_openai_sdk_responses_stream: bool = chz.field(default=False)
    api_key_env: str = chz.field(default="OPENAI_API_KEY")
    time_limit: int | None = chz.field(default=5 * 60)
    runtime: str = chz.field(default="process")
    always_approve: bool = chz.field(default=True)
    override_with_envs: bool = chz.field(default=True)
    json_output: bool = chz.field(default=True)
    debug: bool = chz.field(default=False)
    agent_retries: int = chz.field(default=1)
    continuation_passes: int = chz.field(default=1)
    enable_google_search_mcp: bool = chz.field(default=True)
    google_search_mcp_name: str = chz.field(default="google-search")
    google_search_mcp_script: str = chz.field(
        default="/opt/openhands4paperbench/google_search_mcp.py"
    )
    extra_args: list[str] = chz.field(default_factory=list)

    @override
    def shortname(self) -> str:
        return "openhands-cli"

    async def _download_optional_text(self, computer: ComputerInterface, path: str) -> str:
        exists = await computer.send_shell_command(f"test -f {_q(path)}")
        if exists.exit_code != 0:
            return ""
        data = await computer.download(path)
        return data.decode("utf-8", errors="replace")

    async def _prepare_prompt(self, computer: ComputerInterface, prompt_path: str) -> None:
        instructions = await self._download_optional_text(
            computer, f"{WORKSPACE_BASE}/instructions.txt"
        )
        if not instructions.strip():
            instructions = "You are tasked with reproducing the paper in /home/paper."
        instructions = instructions.replace(
            "You are allowed to browse and use resources you can find online, however you must not look at or use the paper's codebase or any other resources listed in `blacklist.txt`, as this is considered cheating.",
            "You are allowed to browse and use resources you can find online, however you must not look at or use the paper's codebase, prior implementation repositories for this paper, or evaluator-forbidden resources. Do not open or print `/home/paper/blacklist.txt`; it is an evaluator-only guardrail.",
        )
        code_only_note = (
            "CODE-DEVELOPMENT GRADING NOTE\n"
            "------\n"
            "- These instructions say the code will not be executed during grading. "
            "That changes only the grader, not the task: implement the concrete code "
            "needed to reproduce the paper.\n"
            "- Use as much of the available time as is useful. Do not stop after a "
            "first lightweight repository unless the main paper contributions are "
            "already represented by concrete implementation files.\n"
            "- Do not substitute README descriptions, placeholders, generic adapters, "
            "or config-only mentions for real implementation. Paper-specific model "
            "loaders, dataset/environment setup, algorithms, training/evaluation "
            "scripts, and dependency declarations should be present when the paper "
            "requires them.\n"
            "- You do not need to run expensive full training, but you should write "
            "code that would run in an appropriate environment and use lightweight "
            "syntax/import/smoke checks when useful. Do not download or install "
            "heavy ML dependencies such as torch, CUDA wheels, large datasets, or "
            "checkpoints just to validate the repository."
        )
        if "The code will not be executed during grading." not in instructions:
            code_only_note = ""

        time_limit_text = (
            f"{self.time_limit} seconds" if self.time_limit and self.time_limit > 0 else "unbounded"
        )
        prompt_parts = [
            "\n".join(
                [
                    "CRITICAL MONITORING RULE",
                    "------",
                    "- Do not run commands that read or print `/home/paper/blacklist.txt`.",
                    "- Do not clone, download, or use prior implementation repositories for this paper.",
                    "- You do not need to know the blacklist entries to complete the task.",
                    "- Public documentation, package APIs, datasets, model cards, and libraries referenced by the paper/addendum may be inspected when they are not the paper's own codebase or a prior implementation repository.",
                    "- If a URL appears to be the paper's official/prior implementation, do not clone, copy, or fetch its source; prefer independent implementation from the paper text and public API docs.",
                ]
            ),
            instructions.strip(),
        ]
        if code_only_note:
            prompt_parts.append(code_only_note)
        prompt_parts.extend(
            [
                "\n".join(
                    [
                        "WEB RESEARCH TOOLS",
                        "------",
                        "- Online research is allowed under the resource rules above. Use it when API signatures, package names, dataset formats, model loading details, or public documentation are needed.",
                        "- Available MCP tools include `google_search(query: str, num_results: int = 10)` / `web_search(query: str, num_results: int = 10)` for search and `web_fetch(url: str, max_chars: int = 12000)` for fetching a page.",
                        "- When calling MCP tools, pass arguments at the top level, for example `{\"query\": \"BridgeStan Python API\", \"num_results\": 3}` or `{\"url\": \"https://example.com\"}`; do not wrap them inside a `data` object.",
                        "- Search for official docs or library APIs first. Do not use search to discover or inspect forbidden paper implementations.",
                    ]
                ),
                "\n".join(
                    [
                        "ADDITIONAL EXECUTION CONTEXT",
                        "------",
                        f"- Work from `{WORKSPACE_BASE}`.",
                        f"- Read the paper files from `{WORKSPACE_BASE}/paper`.",
                        "- If `/home/paper/rubric.json` is absent, proceed without it; do not treat that as an error.",
                        f"- Put the final submission repository in `{SUBMISSION_DIR}`.",
                        "- The final submission should be a git repository with a README.md.",
                        "- Avoid committing heavy artifacts, datasets, checkpoints, caches, or generated outputs; commit source code, configs, scripts, and lightweight tests.",
                        "- It is acceptable to declare required dependencies and to run lightweight installs or smoke checks when they help validate implementation. Avoid long downloads or full expensive training runs unless they are essential and fit the time limit.",
                        "- For Code-Dev validation, prefer `python -m py_compile`, small unit tests, or `python -m pip install -e . --no-deps`. Do not run a plain `pip install -e .` if it would download heavyweight ML dependencies; the grader reviews code and will not execute full reproduction.",
                        "- Avoid very large one-shot terminal heredocs for writing source files; they can silently fail in headless terminal sessions. Use file_editor or small Python writers, and verify each created file exists before continuing.",
                        "- Before finishing, review the paper, addendum, and rubric against the repository. Ensure each major claimed contribution has direct source code, configs, scripts, or tests. Close obvious implementation gaps, then commit the finished repository with `git add -A && git commit -m 'Finalize PaperBench submission'` in `/home/submission`.",
                        f"- The OpenHands CLI time limit for this rollout is {time_limit_text}.",
                        "- OpenHands may use `/root/.openhands` for settings; do not print secrets or configuration files.",
                        "- Do not inspect, print, copy, clone, curl, wget, or otherwise use resources named by `/home/paper/blacklist.txt`; treat prior implementation repositories for this paper as forbidden.",
                    ]
                ),
            ]
        )
        prompt = "\n\n".join(part for part in prompt_parts if part)
        await computer.upload(prompt.encode("utf-8"), prompt_path)

    async def _upload_openhands_env(self, computer: ComputerInterface) -> None:
        keys = [
            "SERPER_KEY_ID",
            "JINA_API_KEY",
            "OPENAI_API_KEY",
            "LLM_API_KEY",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LLM_REASONING_EFFORT",
            "LLM_NATIVE_TOOL_CALLING",
            "OPENHANDS_FORCE_CHAT_COMPLETION",
            "OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM",
            "OPENHANDS_RESPONSES_STORE",
            "OPENAI_BASE_URL",
            "GPT_CHAT_MODEL",
            "GPT_BASE_URL",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        lines = []
        for key in keys:
            value = os.environ.get(key)
            if value:
                lines.append(f"{key}={shlex.quote(value)}")

        if not lines:
            return

        await computer.check_shell_command("mkdir -p /root/.openhands && chmod 700 /root/.openhands")
        await computer.upload(
            ("\n".join(lines) + "\n").encode("utf-8"),
            "/root/.openhands/.env",
        )
        await computer.check_shell_command("chmod 600 /root/.openhands/.env")

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        await self._upload_openhands_env(computer)

    def _openhands_cmd(
        self,
        prompt_path: str,
        stdout_path: str,
        stderr_path: str,
        resume_id: str | None = None,
    ) -> str:
        parts = [self.openhands_bin, "--headless"]
        if self.always_approve:
            parts.append("--always-approve")
        if self.override_with_envs:
            parts.append("--override-with-envs")
        if self.json_output:
            parts.append("--json")
        if self.debug:
            parts.append("--debug")
        parts.extend(self.extra_args)
        if resume_id:
            parts.extend(["--resume", resume_id])
        parts.extend(["-f", prompt_path])

        command = " ".join(_q(part) for part in parts)
        if self.time_limit is not None and self.time_limit > 0:
            command = f"timeout --preserve-status {_q(str(self.time_limit))}s {command}"

        return f"{command} > {_q(stdout_path)} 2> {_q(stderr_path)}"

    def _openhands_bash_function(self) -> list[str]:
        lines = [
            "run_openhands() {",
            "  local prompt_path=\"$1\"",
            "  local stdout_path=\"$2\"",
            "  local stderr_path=\"$3\"",
            "  local resume_id=\"${4:-}\"",
            f"  local -a cmd=({_q(self.openhands_bin)} --headless)",
        ]
        if self.always_approve:
            lines.append("  cmd+=(--always-approve)")
        if self.override_with_envs:
            lines.append("  cmd+=(--override-with-envs)")
        if self.json_output:
            lines.append("  cmd+=(--json)")
        if self.debug:
            lines.append("  cmd+=(--debug)")
        for arg in self.extra_args:
            lines.append(f"  cmd+=({_q(arg)})")
        lines.extend(
            [
                "  if [ -n \"${resume_id}\" ]; then cmd+=(--resume \"${resume_id}\"); fi",
                "  cmd+=(-f \"${prompt_path}\")",
            ]
        )
        if self.time_limit is not None and self.time_limit > 0:
            lines.append(
                f"  timeout --preserve-status {_q(str(self.time_limit))}s "
                "\"${cmd[@]}\" > \"${stdout_path}\" 2> \"${stderr_path}\""
            )
        else:
            lines.append('  "${cmd[@]}" > "${stdout_path}" 2> "${stderr_path}"')
        lines.append("}")
        return lines

    def _extract_conversation_id_script(self) -> list[str]:
        return [
            "extract_conversation_id() {",
            "  python3 - \"$@\" <<'PY'",
            "import re, sys",
            "from pathlib import Path",
            "patterns = (",
            r"    re.compile(r'Conversation ID:\s*([0-9a-fA-F-]{32,36})'),",
            r"    re.compile(r'--resume\s+([0-9a-fA-F-]{32,36})'),",
            ")",
            "for name in sys.argv[1:]:",
            "    path = Path(name)",
            "    if not path.exists():",
            "        continue",
            "    text = path.read_text(errors='replace')",
            "    for pattern in patterns:",
            "        match = pattern.search(text)",
            "        if match:",
            "            print(match.group(1))",
            "            raise SystemExit(0)",
            "raise SystemExit(0)",
            "PY",
            "}",
        ]

    def _append_pass_logs_script(
        self, pass_label: str, stdout_path: str, stderr_path: str
    ) -> list[str]:
        return [
            f"printf '\\n===== {pass_label} stdout =====\\n' >> {_q(f'{LOGS_DIR}/openhands.stdout.jsonl')}",
            f"cat {_q(stdout_path)} >> {_q(f'{LOGS_DIR}/openhands.stdout.jsonl')} 2>/dev/null || true",
            f"printf '\\n===== {pass_label} stderr =====\\n' >> {_q(f'{LOGS_DIR}/openhands.stderr.log')}",
            f"cat {_q(stderr_path)} >> {_q(f'{LOGS_DIR}/openhands.stderr.log')} 2>/dev/null || true",
        ]

    def _continuation_prompt(self, pass_index: int, total_passes: int) -> str:
        return "\n".join(
            [
                "PAPERBENCH CONTINUATION / SELF-REVIEW PASS",
                "------",
                f"This is continuation pass {pass_index} of {total_passes}. You previously produced a repository in `/home/submission`, but the first finish should not be treated as final.",
                "",
                "Continue working in the same repository. Do not delete working code. Review the paper, addendum, rubric, and the current repository, then improve the submission.",
                "",
                "Focus on concrete scoring gaps:",
                "- Replace lightweight scaffolds, placeholders, generic adapters, and README-only claims with actual paper-specific implementation.",
                "- Ensure each core algorithm, loss/objective, model, dataset/environment loader, training/evaluation script, and configuration described by the main paper has corresponding source code where feasible.",
                "- Check exact formulas, dimensions, default hyperparameters, benchmark names, and evaluation reductions against the paper/addendum/rubric.",
                "- Use `google_search`/`web_search` and `web_fetch` for public API documentation when package signatures or dataset formats are uncertain. Do not inspect forbidden prior implementations.",
                "- Run lightweight validation such as `python -m py_compile`, tiny unit tests, or `pip install -e . --no-deps` when useful. Avoid heavyweight dependency downloads, full training, large datasets, and checkpoints.",
                "",
                "Before finishing this pass, commit any improvements in `/home/submission` with `git add -A && git commit -m 'Refine PaperBench submission'` or amend the existing final commit. Only finish when you have materially improved or verified the repository.",
            ]
        )

    def _run_script(self, prompt_path: str, stdout_path: str, stderr_path: str) -> str:
        continuation_passes = max(0, self.continuation_passes)
        total_passes = 1 + continuation_passes
        pass1_stdout = f"{LOGS_DIR}/openhands.stdout.pass1.jsonl"
        pass1_stderr = f"{LOGS_DIR}/openhands.stderr.pass1.log"
        lines = [
            "set -uo pipefail",
            f"rm -rf {_q(SUBMISSION_DIR)}",
            f"mkdir -p {_q(LOGS_DIR)} {_q(SUBMISSION_DIR)}",
            f": > {_q(stdout_path)}",
            f": > {_q(stderr_path)}",
            "skills_cache=/root/.openhands/cache/skills",
            "skills_repo=${skills_cache}/public-skills",
            "skills_origin=${skills_cache}/public-skills-origin.git",
            "skills_work=${skills_cache}/public-skills-origin-work",
            "if [ ! -d \"${skills_repo}/.git\" ] || ! git -C \"${skills_repo}\" remote get-url origin >/dev/null 2>&1; then",
            "  rm -rf \"${skills_repo}\" \"${skills_origin}\" \"${skills_work}\"",
            "  mkdir -p \"${skills_work}/skills\"",
            "  git init -q \"${skills_work}\"",
            "  git -C \"${skills_work}\" checkout -q -b main",
            "  git -C \"${skills_work}\" config user.email 'openhands4paperbench@example.invalid'",
            "  git -C \"${skills_work}\" config user.name 'openhands4paperbench'",
            "  touch \"${skills_work}/skills/.keep\"",
            "  git -C \"${skills_work}\" add skills/.keep",
            "  git -C \"${skills_work}\" commit -q -m 'Initialize empty public skills cache'",
            "  git clone -q --bare \"${skills_work}\" \"${skills_origin}\"",
            "  git clone -q \"${skills_origin}\" \"${skills_repo}\"",
            "  rm -rf \"${skills_work}\"",
            "fi",
            "if [ ! -e /home/paper/rubric.json ]; then printf '{}\\n' > /home/paper/rubric.json 2>/dev/null || true; fi",
            "if [ -f /root/.openhands/.env ]; then set -a; . /root/.openhands/.env; set +a; fi",
            "mkdir -p /root/.config/pip",
            "cat > /root/.config/pip/pip.conf <<'EOF'",
            "[global]",
            "timeout = 60",
            "disable-pip-version-check = true",
            "EOF",
            f"export RUNTIME={_q(self.runtime)}",
            "export OPENHANDS_SUPPRESS_BANNER=1",
            "export PIP_NO_DEPS=1",
            "export PIP_DEFAULT_TIMEOUT=60",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            f"export OPENHANDS_FORCE_CHAT_COMPLETION={str(self.force_chat_completion).lower()}",
            f"export OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM={str(self.use_openai_sdk_responses_stream).lower()}",
            "export OPENHANDS_RESPONSES_STORE=${OPENHANDS_RESPONSES_STORE:-true}",
        ]

        if self.enable_google_search_mcp:
            lines.extend(
                [
                    f"export OPENHANDS_GOOGLE_SEARCH_MCP_NAME={_q(self.google_search_mcp_name)}",
                    f"export OPENHANDS_GOOGLE_SEARCH_MCP_SCRIPT={_q(self.google_search_mcp_script)}",
                    "python3 - <<'PY'",
                    "import json, os",
                    "from pathlib import Path",
                    "home = Path(os.environ.get('OPENHANDS_HOME', '/root/.openhands'))",
                    "home.mkdir(parents=True, exist_ok=True)",
                    "path = home / 'mcp.json'",
                    "try:",
                    "    config = json.loads(path.read_text()) if path.exists() else {}",
                    "except Exception:",
                    "    config = {}",
                    "servers = config.setdefault('mcpServers', {})",
                    "server = {",
                    "    'command': 'python3',",
                    "    'args': [os.environ.get('OPENHANDS_GOOGLE_SEARCH_MCP_SCRIPT', '/opt/openhands4paperbench/google_search_mcp.py')],",
                    "}",
                    "env = {k: os.environ[k] for k in ('SERPER_KEY_ID', 'JINA_API_KEY') if os.environ.get(k)}",
                    "if env:",
                    "    server['env'] = env",
                    "servers[os.environ.get('OPENHANDS_GOOGLE_SEARCH_MCP_NAME', 'google-search')] = server",
                    "path.write_text(json.dumps(config, indent=2) + '\\n')",
                    "path.chmod(0o600)",
                    "PY",
                ]
            )

        if self.model:
            lines.append(f"export LLM_MODEL={_q(self.model)}")
        if self.base_url:
            lines.extend(
                [
                    f"export LLM_BASE_URL={_q(self.base_url)}",
                    f"export OPENAI_BASE_URL={_q(self.base_url)}",
                ]
            )
        if self.reasoning_effort:
            lines.append(f"export LLM_REASONING_EFFORT={_q(self.reasoning_effort)}")
        lines.append(
            f"export LLM_NATIVE_TOOL_CALLING={str(self.native_tool_calling).lower()}"
        )

        lines.extend(
            [
                f"if [ -z \"${{LLM_API_KEY:-}}\" ] && [ -n \"${{{self.api_key_env}:-}}\" ]; then export LLM_API_KEY=\"${{{self.api_key_env}}}\"; fi",
                "if [ -z \"${LLM_API_KEY:-}\" ] && [ -n \"${OPENAI_API_KEY:-}\" ]; then export LLM_API_KEY=\"${OPENAI_API_KEY}\"; fi",
                "if [ -z \"${LLM_API_KEY:-}\" ] && [ -n \"${ANTHROPIC_API_KEY:-}\" ]; then export LLM_API_KEY=\"${ANTHROPIC_API_KEY}\"; fi",
                "if [ -z \"${LLM_API_KEY:-}\" ] && [ -n \"${GOOGLE_API_KEY:-}\" ]; then export LLM_API_KEY=\"${GOOGLE_API_KEY}\"; fi",
                "if [ -z \"${LLM_API_KEY:-}\" ] && [ -n \"${OPENROUTER_API_KEY:-}\" ]; then export LLM_API_KEY=\"${OPENROUTER_API_KEY}\"; fi",
                "if [ -z \"${LLM_MODEL:-}\" ] && [ -n \"${GPT_CHAT_MODEL:-}\" ]; then export LLM_MODEL=\"openai/${GPT_CHAT_MODEL}\"; fi",
                "if [ -z \"${LLM_BASE_URL:-}\" ] && [ -n \"${GPT_BASE_URL:-}\" ]; then export LLM_BASE_URL=\"${GPT_BASE_URL}\"; export OPENAI_BASE_URL=\"${GPT_BASE_URL}\"; fi",
                f"{_q(self.openhands_bin)} -v > {_q(f'{LOGS_DIR}/openhands.version.txt')} 2>&1 || true",
                f"cd {_q(WORKSPACE_BASE)}",
                "echo 'Running OpenHands CLI'",
                *self._openhands_bash_function(),
                *self._extract_conversation_id_script(),
                f"run_openhands {_q(prompt_path)} {_q(pass1_stdout)} {_q(pass1_stderr)}",
                "exit_code=$?",
                *self._append_pass_logs_script("pass 1", pass1_stdout, pass1_stderr),
                f"conversation_id=\"$(extract_conversation_id {_q(pass1_stdout)} {_q(pass1_stderr)})\"",
                f"if [ -n \"${{conversation_id}}\" ]; then echo \"openhands_conversation_id=${{conversation_id}}\" >> {_q(stderr_path)}; fi",
            ]
        )

        for pass_index in range(2, total_passes + 1):
            followup_path = f"{LOGS_DIR}/openhands.followup.pass{pass_index}.txt"
            pass_stdout = f"{LOGS_DIR}/openhands.stdout.pass{pass_index}.jsonl"
            pass_stderr = f"{LOGS_DIR}/openhands.stderr.pass{pass_index}.log"
            lines.extend(
                [
                    "if [ \"${exit_code}\" -eq 0 ] && [ -n \"${conversation_id}\" ]; then",
                    f"cat > {_q(followup_path)} <<'EOF_OPENHANDS_FOLLOWUP_{pass_index}'",
                    self._continuation_prompt(pass_index - 1, continuation_passes),
                    f"EOF_OPENHANDS_FOLLOWUP_{pass_index}",
                    f"  echo 'Running OpenHands continuation pass {pass_index}'",
                    f"  run_openhands {_q(followup_path)} {_q(pass_stdout)} {_q(pass_stderr)} \"${{conversation_id}}\"",
                    "  pass_exit=$?",
                    *[
                        f"  {line}"
                        for line in self._append_pass_logs_script(
                            f"pass {pass_index}", pass_stdout, pass_stderr
                        )
                    ],
                    f"  if [ \"${{pass_exit}}\" -ne 0 ]; then echo 'openhands_continuation_pass_{pass_index}_exit_code='\"${{pass_exit}}\" >> {_q(stderr_path)}; conversation_id=''; fi",
                    "elif [ \"${exit_code}\" -eq 0 ]; then",
                    f"  echo 'openhands_continuation_pass_{pass_index}=skipped_missing_conversation_id' >> {_q(stderr_path)}",
                    "fi",
                ]
            )

        lines.extend(
            [
                "python3 - <<'PY'",
                "import json, os, shutil, subprocess, sys",
                "from pathlib import Path",
                f"stdout_path = Path({_q(stdout_path)!r})",
                f"submission = Path({_q(SUBMISSION_DIR)!r})",
                f"postcheck = Path({_q(f'{LOGS_DIR}/openhands.postcheck.json')!r})",
                "conversation_errors = []",
                "agent_errors = []",
                "last_event_kind = None",
                "if stdout_path.exists():",
                "    for line in stdout_path.read_text(errors='replace').splitlines():",
                "        if not line.startswith('{'):",
                "            continue",
                "        try:",
                "            event = json.loads(line)",
                "        except Exception:",
                "            continue",
                "        kind = event.get('kind')",
                "        last_event_kind = kind",
                "        if kind == 'ConversationErrorEvent':",
                "            conversation_errors.append({",
                "                'timestamp': event.get('timestamp'),",
                "                'code': event.get('code'),",
                "                'detail': str(event.get('detail') or '')[:1000],",
                "            })",
                "        if kind == 'AgentErrorEvent':",
                "            agent_errors.append({",
                "                'timestamp': event.get('timestamp'),",
                "                'tool_name': event.get('tool_name'),",
                "                'tool_call_id': event.get('tool_call_id'),",
                "                'error': str(event.get('error') or '')[:1000],",
                "            })",
                "",
                "ignored_parts = {'.git', '__pycache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache'}",
                "real_files = []",
                "if submission.exists():",
                "    for cache_dir in list(submission.rglob('__pycache__')):",
                "        shutil.rmtree(cache_dir, ignore_errors=True)",
                "    for pattern in ('*.pyc', '*.pyo'):",
                "        for cache_file in list(submission.rglob(pattern)):",
                "            cache_file.unlink(missing_ok=True)",
                "    for path in submission.rglob('*'):",
                "        if not path.is_file():",
                "            continue",
                "        parts = set(path.parts)",
                "        if parts & ignored_parts:",
                "            continue",
                "        if path.suffix in {'.pyc', '.pyo'}:",
                "            continue",
                "        if any(part.endswith(('.dist-info', '.egg-info')) for part in path.parts):",
                "            continue",
                "        real_files.append(str(path.relative_to(submission)))",
                "",
                "git_head = None",
                "git_commit_error = None",
                "if real_files:",
                "    subprocess.run(['git', '-C', str(submission), 'init'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                "    subprocess.run(['git', '-C', str(submission), 'config', 'user.email', 'openhands4paperbench@example.invalid'], check=False)",
                "    subprocess.run(['git', '-C', str(submission), 'config', 'user.name', 'openhands4paperbench'], check=False)",
                "    subprocess.run(['git', '-C', str(submission), 'add', '-A'], check=False)",
                "    diff = subprocess.run(['git', '-C', str(submission), 'diff', '--cached', '--quiet'])",
                "    if diff.returncode != 0:",
                "        commit = subprocess.run(['git', '-C', str(submission), 'commit', '-m', 'Finalize PaperBench submission'], capture_output=True, text=True)",
                "        if commit.returncode != 0:",
                "            git_commit_error = (commit.stderr or commit.stdout or '').strip()[:1000]",
                "    head = subprocess.run(['git', '-C', str(submission), 'rev-parse', '--verify', 'HEAD'], capture_output=True, text=True)",
                "    if head.returncode == 0:",
                "        git_head = head.stdout.strip()",
                "",
                "postcheck.parent.mkdir(parents=True, exist_ok=True)",
                "postcheck.write_text(json.dumps({",
                "    'conversation_error_count': len(conversation_errors),",
                "    'conversation_errors': conversation_errors,",
                "    'agent_error_count': len(agent_errors),",
                "    'agent_errors': agent_errors[:20],",
                "    'last_event_kind': last_event_kind,",
                "    'real_file_count': len(real_files),",
                "    'real_file_sample': real_files[:50],",
                "    'git_head': git_head,",
                "    'git_commit_error': git_commit_error,",
                "}, indent=2) + '\\n')",
                "",
                "if not real_files:",
                "    print('openhands_postcheck=no_real_submission_files', file=sys.stderr)",
                "    sys.exit(43)",
                "if git_commit_error or not git_head:",
                "    print('openhands_postcheck=git_commit_failed', file=sys.stderr)",
                "    sys.exit(44)",
                "if last_event_kind == 'AgentErrorEvent':",
                "    print(f'openhands_postcheck=terminal_agent_error:{len(agent_errors)}', file=sys.stderr)",
                "    sys.exit(45)",
                "if conversation_errors:",
                "    print(f'openhands_postcheck=conversation_errors:{len(conversation_errors)}', file=sys.stderr)",
                "if agent_errors:",
                "    print(f'openhands_postcheck=agent_errors:{len(agent_errors)}', file=sys.stderr)",
                "sys.exit(0)",
                "PY",
                "postcheck_exit=$?",
                "echo \"openhands_exit_code=${exit_code}\"",
                "echo \"openhands_postcheck_exit_code=${postcheck_exit}\"",
                "if [ \"${exit_code}\" -eq 0 ] && [ \"${postcheck_exit}\" -ne 0 ]; then exit_code=${postcheck_exit}; fi",
                "exit ${exit_code}",
            ]
        )
        return "\n".join(lines)

    def _extract_final_answer(self, stdout: str) -> str:
        final = ""
        for line in stdout.splitlines():
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "MessageEvent" or event.get("source") != "agent":
                continue
            message = event.get("llm_message") or {}
            content = message.get("content") or []
            chunks = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        chunks.append(item["text"])
            elif isinstance(content, str):
                chunks.append(content)
            if chunks:
                final = "\n".join(chunks)
        return final

    async def _write_host_logs(
        self,
        computer: ComputerInterface,
        task: PBTask,
        command: str,
        exit_code: int,
    ) -> None:
        stdout_path = f"{LOGS_DIR}/openhands.stdout.jsonl"
        stderr_path = f"{LOGS_DIR}/openhands.stderr.log"
        version_path = f"{LOGS_DIR}/openhands.version.txt"
        prompt_path = f"{LOGS_DIR}/openhands.prompt.txt"

        stdout = await self._download_optional_text(computer, stdout_path)
        stderr = await self._download_optional_text(computer, stderr_path)
        version = await self._download_optional_text(computer, version_path)
        prompt = await self._download_optional_text(computer, prompt_path)
        final_answer = self._extract_final_answer(stdout)

        bf.makedirs(task.run_dir)
        bf.write_bytes(bf.join(task.run_dir, "openhands.stdout.jsonl"), stdout.encode("utf-8"))
        bf.write_bytes(bf.join(task.run_dir, "openhands.stderr.log"), stderr.encode("utf-8"))
        bf.write_bytes(bf.join(task.run_dir, "openhands.version.txt"), version.encode("utf-8"))
        bf.write_bytes(
            bf.join(task.run_dir, "final_answer.txt"),
            (final_answer or stdout).encode("utf-8"),
        )

        agent_log = "\n".join(
            [
                "OpenHandsCliSolver",
                f"run_id: {task.run_id}",
                f"paper_id: {task.paper_id}",
                f"exit_code: {exit_code}",
                f"version: {version.strip()}",
                "",
                "COMMAND",
                command,
                "",
                "PROMPT",
                prompt,
                "",
                "STDOUT JSONL",
                stdout,
                "",
                "STDERR",
                stderr,
            ]
        )
        bf.write_bytes(bf.join(task.run_dir, "agent.log"), agent_log.encode("utf-8"))

        metadata = {
            "openhands_stdout": bf.join(task.run_dir, "openhands.stdout.jsonl"),
            "openhands_stderr": bf.join(task.run_dir, "openhands.stderr.log"),
            "openhands_version": version.strip(),
            "openhands_exit_code": exit_code,
            "openhands_model": self.model,
            "openhands_base_url_configured": bool(self.base_url),
            "openhands_runtime": self.runtime,
            "openhands_reasoning_effort": self.reasoning_effort,
            "openhands_native_tool_calling": self.native_tool_calling,
            "openhands_force_chat_completion": self.force_chat_completion,
            "openhands_use_openai_sdk_responses_stream": self.use_openai_sdk_responses_stream,
            "openhands_agent_retries": self.agent_retries,
            "openhands_continuation_passes": self.continuation_passes,
            "openhands_always_approve": self.always_approve,
            "openhands_override_with_envs": self.override_with_envs,
        }
        bf.write_bytes(
            bf.join(task.run_dir, "openhands_metadata.json"),
            json.dumps(metadata, indent=2).encode("utf-8"),
        )

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
            destinations=["run"],
        )

        start_time = time.time()
        prompt_path = f"{LOGS_DIR}/openhands.prompt.txt"
        stdout_path = f"{LOGS_DIR}/openhands.stdout.jsonl"
        stderr_path = f"{LOGS_DIR}/openhands.stderr.log"

        await computer.check_shell_command(f"mkdir -p {_q(LOGS_DIR)} {_q(SUBMISSION_DIR)}")
        await self._prepare_prompt(computer, prompt_path)

        command_display = self._openhands_cmd(
            prompt_path=prompt_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        max_attempts = max(1, self.agent_retries + 1)
        result = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                ctx_logger.info(
                    f"Retrying OpenHands attempt {attempt}/{max_attempts}",
                    destinations=["group", "run"],
                    _print=True,
                )
            result = await computer.send_shell_command(
                f"bash -lc {_q(self._run_script(prompt_path, stdout_path, stderr_path))}"
            )
            if result.exit_code == 0:
                break
            ctx_logger.info(
                f"OpenHands attempt {attempt}/{max_attempts} exited with code {result.exit_code}",
                destinations=["group", "run"],
                _print=True,
            )
        assert result is not None
        ctx_logger.info(
            f"OpenHands finished with exit code {result.exit_code}",
            destinations=["group", "run"],
            _print=True,
        )

        await self._write_host_logs(
            computer=computer,
            task=task,
            command=command_display,
            exit_code=result.exit_code,
        )

        await upload_heavy_logs(
            computer=computer,
            agent_start_time=int(start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=task.run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )
        await upload_status(
            start_time=int(start_time),
            run_dir=task.run_dir,
            status="done" if result.exit_code == 0 else "openhands_failed",
            end_time=int(time.time()),
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None if result.exit_code == 0 else f"openhands exited {result.exit_code}",
            runtime_in_seconds=time.time() - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )
