"""OpenClaw rollout pipeline for swe-agents StepTron adaptor.

This pipeline mirrors the special CLI routing (Codex / Claude Code / Kilo Code):
- Allocate (or resume) a SessionRouter container for the task Docker image
- Start the local FastAPI passthrough proxy (OpenAI-compatible base URL only)
- Install and run OpenClaw CLI inside the container (embedded agent mode, no Gateway)
- Extract a compact `meta.sft_data` transcript from FastAPI logs

OpenClaw is a Node.js CLI. This integration uses a helper init script to ensure
Node >=22 and `openclaw` are available inside the container.
"""

from __future__ import annotations

from contextlib import nullcontext, suppress
import ipaddress
import json
import os
from pathlib import Path
import shlex
from typing import Any, Callable
from urllib.parse import urlparse

from loguru import logger as default_logger

from swe_agents.core.types import Dialog, TaskTrajectory
from swe_agents.impl.openclaw.openclaw_support import (
    build_fastapi_proxy_env,
    extract_patch_auto,
    maybe_attach_fastapi_logs_tar_base64,
    pick_existing_file,
    prepare_patch_workspace_auto,
    resolve_fastapi_bootstrap_assets,
    step2_package_candidates,
)
from swe_agents.utils.metrics import record_session_create_metrics
from swe_agents.utils.session_router_client import ExecResult, SessionRouterClient


DEFAULT_REQUEST_CPU = os.getenv("SWE_OPENCLAW_REQUEST_CPU", "1000m")
DEFAULT_REQUEST_MEMORY = os.getenv("SWE_OPENCLAW_REQUEST_MEMORY", "4Gi")
DEFAULT_LIMIT_CPU = os.getenv("SWE_OPENCLAW_LIMIT_CPU", "1000m")
DEFAULT_LIMIT_MEMORY = os.getenv("SWE_OPENCLAW_LIMIT_MEMORY", "4Gi")
DEFAULT_PULL_POLICY = "IfNotPresent"
_OPENCLAW_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high"}
_OPENCLAW_MODEL_APIS = {"openai-responses", "openai-completions", "anthropic-messages"}

def _resolve_openclaw_runtime_bundle(*, openclaw_tools: Path) -> Path:
    candidate_names = (
        "openclaw-runtime-bundle.tar.gz",
        "openclaw-runtime-2026.2.6-3-node-v22.13.1-linux-x64.tar.gz",
    )
    candidates = [
        *(openclaw_tools / "bin" / name for name in candidate_names),
        *step2_package_candidates("openclaw", candidate_names),
    ]

    existing = pick_existing_file(candidates)
    if existing is not None:
        return existing
    raise RuntimeError(
        "missing openclaw runtime bundle under env_tools/openclaw/bin or mounted /mnt packages"
    )


def _require_str(instance: dict[str, Any], name: str) -> str:
    if name not in instance:
        raise RuntimeError(f"Missing required field: {name}")
    value = instance[name]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid {name} for OpenClaw pipeline")
    return value.strip()


def _require_optional_str(instance: dict[str, Any], name: str) -> str | None:
    if name not in instance:
        return None
    value = instance[name]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid {name} for OpenClaw pipeline")
    return value.strip()


def _require_timeout(instance: dict[str, Any], name: str) -> float:
    if name not in instance:
        raise RuntimeError(f"Missing required field: {name}")
    value = instance[name]
    try:
        timeout = float(value)
    except Exception as exc:
        raise RuntimeError(f"Invalid {name} for OpenClaw pipeline") from exc
    if timeout <= 0:
        raise RuntimeError(f"Invalid {name} for OpenClaw pipeline")
    return timeout


def _get_timeout(instance: dict[str, Any]) -> float:
    for key in ("openclaw_timeout_sec", "cc_timeout_sec", "codex_timeout_sec"):
        if key in instance:
            return _require_timeout(instance, key)
    raise RuntimeError("Missing timeout for OpenClaw pipeline")


def _build_session_env_vars(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env_vars: dict[str, str] = {
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R -F -X",
    }
    # SessionRouter pods may have very different DNS/proxy topology than the
    # workspace. In particular, some clusters cannot resolve the Shaipower
    # proxy hostname, which would break model_proxy calls if we pass it through.
    #
    # Default: do NOT propagate http_proxy/https_proxy into the SessionRouter
    # container. Operators can opt-in via SWE_OPENCLAW_PASS_PROXY=1.
    if _env_truthy("SWE_OPENCLAW_PASS_PROXY"):
        for lo, up in (
            ("http_proxy", "HTTP_PROXY"),
            ("https_proxy", "HTTPS_PROXY"),
            ("all_proxy", "ALL_PROXY"),
        ):
            value = os.getenv(lo) or os.getenv(up)
            if value:
                env_vars[lo] = value
                env_vars[up] = value
    value = os.getenv("no_proxy") or os.getenv("NO_PROXY")
    if value:
        env_vars["no_proxy"] = value
        env_vars["NO_PROXY"] = value
    if extra_env:
        env_vars.update({str(k): str(v) for k, v in extra_env.items()})
    return env_vars


def _format_env_prefix(env_vars: dict[str, str | None]) -> str:
    items = []
    for key, value in env_vars.items():
        if value is None:
            continue
        items.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(items)


def _resolve_model_endpoint(endpoint: str, model_endpoint: str | None) -> str:
    if isinstance(model_endpoint, str) and model_endpoint.strip():
        return model_endpoint.strip()
    lowered = endpoint.strip().lower()
    if lowered in {"model_proxy", "stepcast", "agent_router"}:
        return lowered
    if lowered.startswith("http"):
        if "models-proxy" in lowered or "model-proxy" in lowered or "model_proxy" in lowered:
            return "model_proxy"
        if "agentrouter" in lowered or "agent-router" in lowered or "agent_router" in lowered:
            return "agent_router"
        if "stepcast" in lowered or ":9200" in lowered:
            return "stepcast"
    return "stepcast"


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_openclaw_thinking(instance: dict[str, Any]) -> str:
    for raw in (
        instance.get("openclaw_thinking"),
        os.getenv("SWE_OPENCLAW_THINKING"),
        os.getenv("OPENCLAW_THINKING"),
    ):
        if not isinstance(raw, str):
            continue
        level = raw.strip().lower()
        if level in _OPENCLAW_THINKING_LEVELS:
            return level
    # Default to explicit high reasoning effort for trace-rich OpenClaw runs.
    return "high"


def _normalize_openclaw_model_api(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _OPENCLAW_MODEL_APIS:
        return normalized
    return None


def _resolve_openclaw_model_api(
    instance: dict[str, Any],
    init_kwargs: Any,
    model_endpoint: str,
) -> str:
    for raw in (
        instance.get("openclaw_model_api"),
        getattr(init_kwargs, "openclaw_model_api", None),
        os.getenv("SWE_OPENCLAW_MODEL_API"),
        os.getenv("OPENCLAW_MODEL_API"),
    ):
        normalized = _normalize_openclaw_model_api(raw)
        if normalized:
            return normalized
    if model_endpoint == "stepcast":
        # Stepcast passthrough is validated on the OpenAI Responses path.
        return "openai-responses"
    return "openai-responses"


def _normalize_openclaw_max_tokens(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _resolve_openclaw_max_tokens(
    instance: dict[str, Any],
    init_kwargs: Any,
) -> int | None:
    for raw in (
        instance.get("openclaw_max_tokens"),
        getattr(init_kwargs, "openclaw_max_tokens", None),
        os.getenv("SWE_OPENCLAW_MAX_TOKENS"),
        os.getenv("OPENCLAW_MAX_TOKENS"),
    ):
        normalized = _normalize_openclaw_max_tokens(raw)
        if normalized is not None:
            return normalized
    return None


def _is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _extract_host_from_endpoint(endpoint: str) -> str | None:
    if not isinstance(endpoint, str) or not endpoint.strip():
        return None
    endpoint = endpoint.strip()
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    try:
        parsed = urlparse(endpoint)
    except Exception:
        return None
    return parsed.hostname


def _exec_bash_checked(
    client: SessionRouterClient,
    command: str,
    *,
    timeout: float | None = None,
    display_command: str | None = None,
) -> ExecResult:
    ret = client.exec_bash(command, timeout=timeout)
    if "error" in ret:
        ret_any = ret  # TypedDict
        stdout = ret_any.get("stdout")
        stderr = ret_any.get("stderr")
        error = ret_any.get("error")
        exit_status = ret_any.get("exit_status")
        raise RuntimeError(
            f"Command failed: {display_command or command}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}\n"
            f"error: {error}\n"
            f"exit_status: {exit_status}"
        )
    return ret


def _maybe_patch_stepcast_hosts(
    session_client: SessionRouterClient,
    *,
    endpoint: str,
    model_endpoint: str,
    host_override: str | None = None,
    aliases_override: list[str] | None = None,
    logger=default_logger,
) -> None:
    if model_endpoint != "stepcast":
        return
    has_override = bool(
        host_override
        or aliases_override
        or _env_truthy("SWE_OPENCLAW_TRAIN_MODE")
        or _env_truthy("SWE_OPENCLAW_STEPCAST_HOST_OVERRIDE")
        or _env_truthy("SWE_CC_TRAIN_MODE")
        or _env_truthy("SWE_CC_STEPCAST_HOST_OVERRIDE")
    )
    host = (
        host_override
        or os.getenv("SWE_OPENCLAW_STEPCAST_HOST")
        or os.getenv("SWE_CC_STEPCAST_HOST")
        or ""
    ).strip()
    if not host:
        host = _extract_host_from_endpoint(endpoint) or ""
    if not host:
        return
    if not has_override and not _is_ip_address(host):
        return
    if not _is_ip_address(host):
        logger.warning(f"Skip patch /etc/hosts for stepcast aliases: non-IP host={host}")
        return

    aliases: list[str] = []
    if aliases_override:
        aliases = [str(a).strip() for a in aliases_override if str(a).strip()]
    if not aliases:
        aliases_raw = os.getenv(
            "SWE_OPENCLAW_STEPCAST_HOST_ALIASES",
            os.getenv(
                "SWE_CC_STEPCAST_HOST_ALIASES",
                "stepcast-router.basemind-core,stepcast-router",
            ),
        )
        aliases = [a for a in (aliases_raw or "").replace(",", " ").split() if a]
    if not aliases:
        return

    alias_list = " ".join(shlex.quote(a) for a in aliases)
    script = (
        "set -e\n"
        f"HOST={shlex.quote(host)}\n"
        f"for name in {alias_list}; do\n"
        "  if [ \"$name\" = \"$HOST\" ]; then\n"
        "    continue\n"
        "  fi\n"
        "  if grep -qE \"[[:space:]]${name}([[:space:]]|$)\" /etc/hosts; then\n"
        "    sed -i.bak \"/[[:space:]]${name}([[:space:]]|$)/d\" /etc/hosts\n"
        "  fi\n"
        "  echo \"$HOST $name\" >> /etc/hosts\n"
        "done\n"
    )
    try:
        _exec_bash_checked(session_client, script)
        logger.info(f"Patched /etc/hosts for stepcast aliases {aliases} -> {host}")
    except Exception as exc:
        logger.warning(f"Failed to patch /etc/hosts for stepcast: {exc}")


def _diagnose_stepcast_hosts(
    session_client: SessionRouterClient,
    *,
    endpoint: str,
    host_override: str | None = None,
    aliases_override: list[str] | None = None,
    logger=default_logger,
) -> None:
    host = (host_override or _extract_host_from_endpoint(endpoint) or "").strip()
    aliases: list[str] = []
    if aliases_override:
        aliases = [str(a).strip() for a in aliases_override if str(a).strip()]
    if not aliases:
        aliases_raw = os.getenv(
            "SWE_OPENCLAW_STEPCAST_HOST_ALIASES",
            os.getenv(
                "SWE_CC_STEPCAST_HOST_ALIASES",
                "stepcast-router.basemind-core,stepcast-router",
            ),
        )
        aliases = [a for a in (aliases_raw or "").replace(",", " ").split() if a]
    targets = [t for t in [host, *aliases] if t]
    if not targets:
        return
    target_list = " ".join(shlex.quote(t) for t in targets)
    cmd = (
        "set +e\n"
        "echo '--- /etc/hosts (tail) ---'\n"
        "tail -n 200 /etc/hosts || true\n"
        "echo '--- getent hosts ---'\n"
        f"for name in {target_list}; do echo \"## $name\"; getent hosts \"$name\" || true; done\n"
        "echo '--- env proxies ---'\n"
        "env | grep -E '^(http_proxy|https_proxy|no_proxy|HTTP_PROXY|HTTPS_PROXY|NO_PROXY)=' || true\n"
    )
    ret = session_client.exec_bash(cmd)
    if isinstance(ret, dict):
        stdout = ret.get("stdout") or ""
        stderr = ret.get("stderr") or ""
        if stdout or stderr:
            logger.error(f"openclaw_run diagnostics\nstdout:\n{stdout}\nstderr:\n{stderr}")


def _extract_request_id_from_logs(
    session_client: SessionRouterClient,
    session_id: str,
    *,
    logger=default_logger,
) -> str | None:
    safe_sid = shlex.quote(session_id)
    cmd = (
        "set +e\n"
        "for f in $(ls -1t /tmp/fastapi_logs/*.jsonl 2>/dev/null); do\n"
        f"  rid=$(grep -a {safe_sid} \"$f\" | grep -Eao '(chatcmpl|resp)-[A-Za-z0-9_-]*' | tail -n 1)\n"
        "  if [ -n \"$rid\" ]; then echo \"$rid\"; exit 0; fi\n"
        "done\n"
        "exit 0\n"
    )
    try:
        ret = session_client.exec_bash(cmd)
    except Exception as exc:
        logger.warning(f"openclaw_log request_id extract failed: {exc}")
        return None
    if not isinstance(ret, dict):
        return None
    stdout = (ret.get("stdout") or "").strip()
    if not stdout:
        return None
    request_id = stdout.splitlines()[-1].strip()
    return request_id or None


def _collect_openclaw_sft_data(
    session_client: SessionRouterClient,
    session_id: str,
    *,
    logger=default_logger,
) -> str | None:
    out_path = "/tmp/openclaw_log_sft.json"
    try:
        _exec_bash_checked(
            session_client,
            "python3 /tmp/openclaw_log_extract.py "
            f"--session-id {shlex.quote(session_id)} "
            f"--output {shlex.quote(out_path)}",
            timeout=1800,
        )
    except Exception as exc:
        logger.warning(f"openclaw_log_extract failed: {exc}")
        return None
    try:
        raw = session_client.download(out_path)
    except Exception as exc:
        logger.warning(f"openclaw_log_extract download failed: {exc}")
        return None
    if not raw:
        return None
    try:
        return raw.decode("utf-8")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _is_empty_sft_data(raw: str | None) -> bool:
    if not raw or not raw.strip():
        return True
    try:
        payload = json.loads(raw)
    except Exception:
        return True
    for messages, _dialog in _iter_sft_conversations(payload):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role:
                return False
    return True


def _collect_fastapi_debug_snapshot(
    session_client: SessionRouterClient,
    *,
    clip: int = 4000,
) -> str:
    cmd = (
        "set +e\n"
        "echo '--- /tmp/fastapi.ini ---'\n"
        "cat /tmp/fastapi.ini 2>/dev/null || true\n"
        "echo\n"
        "echo '--- /tmp/fastapi_logs (ls) ---'\n"
        "ls -la /tmp/fastapi_logs 2>/dev/null || true\n"
        "echo\n"
        "echo '--- /tmp/fastapi_log.txt (tail) ---'\n"
        "tail -n 80 /tmp/fastapi_log.txt 2>/dev/null || true\n"
        "echo\n"
        "latest=$(ls -1t /tmp/fastapi_logs/*.jsonl 2>/dev/null | head -n 1)\n"
        "echo \"--- latest jsonl: ${latest} ---\"\n"
        "if [ -n \"$latest\" ]; then\n"
        "  echo '--- latest jsonl (tail 2) ---'\n"
        "  tail -n 2 \"$latest\" | head -c 20000\n"
        "  echo\n"
        "fi\n"
        "exit 0\n"
    )
    ret = session_client.exec_bash(cmd)
    if not isinstance(ret, dict):
        return ""
    stdout = ret.get("stdout") or ""
    stderr = ret.get("stderr") or ""
    text = "\n".join([s for s in [stdout, stderr] if isinstance(s, str) and s.strip()])
    if clip > 0 and len(text) > clip:
        return text[:clip] + "...(truncated)"
    return text.strip()


def _download_text_if_exists(
    session_client: SessionRouterClient,
    path: str,
    *,
    clip: int = 8000,
) -> str:
    ret = session_client.exec_bash(
        f"if [ -f {shlex.quote(path)} ]; then echo 'Exists'; else echo 'Not Exists'; fi"
    )
    if not isinstance(ret, dict) or (ret.get("stdout") or "").strip() != "Exists":
        return ""
    raw = session_client.download(path)
    text = raw.decode("utf-8", errors="replace") if raw else ""
    if clip > 0 and len(text) > clip:
        return text[-clip:]
    return text


_DIALOG_REASONING_KEYS = ("reasoning_content", "thinking", "analysis", "reasoning")
_DIALOG_REASONING_TYPES = {"thinking", "analysis", "reasoning"}


def _content_to_text(value: Any, *, include_reasoning: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            _content_to_text(item, include_reasoning=include_reasoning)
            for item in value
        ]
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        item_type = str(value.get("type") or "").strip().lower()
        if not include_reasoning and item_type in _DIALOG_REASONING_TYPES:
            return ""
        keys = ("text", "content", "output_text", "input_text") + (
            _DIALOG_REASONING_KEYS if include_reasoning else ()
        )
        for key in keys:
            v = value.get(key)
            if v is not None:
                text = _content_to_text(v, include_reasoning=include_reasoning)
                if text:
                    return text
        return ""
    return str(value)


def _compact_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _iter_sft_conversations(payload: Any) -> list[tuple[list[Any], dict[str, Any] | None]]:
    conversations: list[tuple[list[Any], dict[str, Any] | None]] = []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "role" in payload[0]:
            conversations.append((payload, None))
            return conversations
        for candidate in payload:
            if isinstance(candidate, list):
                conversations.append((candidate, None))
            elif isinstance(candidate, dict) and isinstance(candidate.get("messages"), list):
                conversations.append((candidate["messages"], candidate))
        return conversations
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        conversations.append((payload["messages"], payload))
    return conversations


def _normalize_tool_call(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    call_id = raw.get("id") or raw.get("tool_call_id") or raw.get("call_id")
    if call_id is not None:
        call_id = str(call_id).strip()
    if not call_id:
        return None

    function = raw.get("function")
    fn_name: str | None = None
    fn_args: Any = None
    if isinstance(function, dict):
        fn_name = function.get("name")
        fn_args = function.get("arguments")
    if not fn_name:
        fn_name = raw.get("name")
    if fn_name is not None:
        fn_name = str(fn_name).strip()
    if not fn_name:
        return None
    if fn_args is None:
        fn_args = raw.get("arguments")
    if fn_args is None:
        fn_args = "{}"

    return {
        "type": "function",
        "id": call_id,
        "function": {
            "name": fn_name,
            "arguments": _compact_json(fn_args),
        },
    }


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        norm = _normalize_tool_call(item)
        if norm is not None:
            out.append(norm)
    return out


def _extract_msg_meta(msg: dict[str, Any]) -> dict[str, Any]:
    for key in ("meta", "metadata"):
        raw = msg.get(key)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _extract_dialog_meta(dialog: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(dialog, dict):
        return {}
    for key in ("meta", "metadata"):
        raw = dialog.get(key)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _normalize_dialog_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function")
        if not isinstance(fn, dict):
            name = raw.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            fn = {
                "name": name.strip(),
                "description": str(raw.get("description") or ""),
                "parameters": raw.get("parameters"),
            }
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = fn.get("description")
        if not isinstance(description, str):
            description = ""
        parameters = fn.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        out_fn: dict[str, Any] = {
            "name": name.strip(),
            "description": description,
            "parameters": parameters,
        }
        strict = fn.get("strict")
        if isinstance(strict, bool):
            out_fn["strict"] = strict
        normalized.append(
            {
                "type": "function",
                "function": out_fn,
            }
        )
    return normalized


def _extract_tool_call_id(msg: dict[str, Any]) -> str:
    for key in ("tool_call_id", "toolUseId", "tool_use_id", "call_id", "id"):
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _content_to_text_with_json_fallback(value: Any, *, include_reasoning: bool = True) -> str:
    text = _content_to_text(value, include_reasoning=include_reasoning).strip()
    if text:
        return text
    if isinstance(value, (dict, list)):
        return _compact_json(value)
    return text


_DIALOG_ALLOWED_ROLES = {"user", "assistant", "system", "developer", "tool"}


def _collect_reasoning_parts_from_content(content: Any, out: list[str]) -> None:
    if content is None:
        return
    if isinstance(content, list):
        for item in content:
            _collect_reasoning_parts_from_content(item, out)
        return
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").strip().lower()
        if item_type in _DIALOG_REASONING_TYPES:
            raw = content.get(item_type)
            if raw is None:
                raw = content.get("content")
            if raw is None:
                raw = content.get("text")
            text = _content_to_text(raw, include_reasoning=True).strip()
            if text:
                out.append(text)
            return

        for key in _DIALOG_REASONING_KEYS:
            raw = content.get(key)
            if raw is None:
                continue
            text = _content_to_text(raw, include_reasoning=True).strip()
            if text:
                out.append(text)

        nested = content.get("content")
        if isinstance(nested, (list, dict)):
            _collect_reasoning_parts_from_content(nested, out)


def _extract_message_reasoning_content(msg: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _DIALOG_REASONING_KEYS:
        raw = msg.get(key)
        if raw is None:
            continue
        text = _content_to_text(raw, include_reasoning=True).strip()
        if text:
            parts.append(text)

    _collect_reasoning_parts_from_content(msg.get("content"), parts)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return "\n\n".join(deduped).strip()


def _dialogs_from_sft_data(raw_sft_data: str | None) -> list[Dialog]:
    if not isinstance(raw_sft_data, str) or not raw_sft_data.strip():
        return []
    try:
        payload = json.loads(raw_sft_data)
    except Exception:
        return []

    dialogs: list[Dialog] = []
    for raw_messages, raw_dialog in _iter_sft_conversations(payload):
        messages: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in _DIALOG_ALLOWED_ROLES:
                continue
            raw_content = msg.get("content")
            if role == "assistant":
                content = _content_to_text(
                    raw_content,
                    include_reasoning=False,
                ).strip()
            else:
                content = _content_to_text_with_json_fallback(
                    raw_content,
                    include_reasoning=True,
                ).strip()
            item: dict[str, Any] = {
                "role": role,
                "content": content,
            }
            if role == "assistant":
                reasoning_content = _extract_message_reasoning_content(msg)
                if reasoning_content:
                    item["reasoning_content"] = reasoning_content
                tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
                if tool_calls:
                    item["tool_calls"] = tool_calls
                reasoning_signature = msg.get("reasoning_signature")
                if isinstance(reasoning_signature, str) and reasoning_signature.strip():
                    item["reasoning_signature"] = reasoning_signature.strip()
            if role == "tool":
                tool_call_id = _extract_tool_call_id(msg)
                if tool_call_id:
                    item["tool_call_id"] = tool_call_id
                if not content and raw_content is None:
                    item["content"] = ""
            name = msg.get("name")
            if isinstance(name, str) and name.strip():
                item["name"] = name.strip()
            msg_meta = _extract_msg_meta(msg)
            if msg_meta:
                item["meta"] = msg_meta

            has_payload = bool(content)
            if role == "assistant":
                has_payload = bool(
                    content
                    or item.get("reasoning_content")
                    or item.get("tool_calls")
                    or item.get("reasoning_signature")
                )
            elif role == "tool":
                has_payload = bool(content or item.get("tool_call_id"))
            if not has_payload:
                continue
            messages.append(item)
        if not messages:
            continue
        dialog_data: dict[str, Any] = {"messages": messages}
        dialog_tools = _normalize_dialog_tools(
            raw_dialog.get("tools") if isinstance(raw_dialog, dict) else None
        )
        if dialog_tools:
            dialog_data["tools"] = dialog_tools
        dialog_meta = _extract_dialog_meta(raw_dialog)
        if dialog_meta:
            dialog_data["meta"] = dialog_meta
        try:
            dialogs.append(Dialog(**dialog_data))
        except Exception:
            # Keep the transcript even if optional dialog-level fields are malformed.
            dialogs.append(Dialog(messages=messages))
    return dialogs


def _build_task_trajectory(
    meta: dict[str, Any],
) -> TaskTrajectory:
    raw_sft_data = meta.get("sft_data") or meta.get("cc_sft_data")
    dialogs = _dialogs_from_sft_data(raw_sft_data if isinstance(raw_sft_data, str) else None)
    return TaskTrajectory(dialogs=dialogs, meta=meta)


def run_openclaw_pipeline(
    init_kwargs: Any,
    task: Any,
    *,
    check_cancel: Callable[[], None],
    register_cleanup: Callable[[Callable[[], None]], None],
    stage_context: Callable[[str], Any] | None = None,
    logger=default_logger,
    keep_session_open: bool = False,
) -> TaskTrajectory:
    instance: dict[str, Any] = getattr(task, "swe_instance", None) or {}
    if not isinstance(instance, dict):
        raise RuntimeError("Missing swe_instance for OpenClaw pipeline")

    stage_ctx = stage_context or (lambda _: nullcontext())

    instance_id = _require_str(instance, "instance_id")
    docker_image = _require_str(instance, "docker_image")
    workspace_root = _require_str(instance, "workspace_root")
    prompt_text = _require_str(instance, "prompt")
    system_prompt = _require_optional_str(instance, "openclaw_system_prompt")
    openclaw_timeout = _get_timeout(instance)
    openclaw_thinking = _resolve_openclaw_thinking(instance)

    api_key = getattr(init_kwargs, "api_key", None)
    if api_key is not None and isinstance(api_key, str) and not api_key.strip():
        api_key = None
    if api_key is None:
        api_key = os.getenv("MODEL_PROXY_API_KEY") or os.getenv("MODELPROXY_APIKEY")
    if api_key is None:
        raise RuntimeError("Missing api_key for OpenClaw pipeline")

    model_names = getattr(init_kwargs, "model_names", None) or []
    if not model_names:
        raise RuntimeError("Missing model_names for OpenClaw pipeline")
    model_name = instance.get("openclaw_model") or model_names[0]
    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError("Missing model name for OpenClaw pipeline")
    model_name = model_name.strip()

    endpoint = getattr(init_kwargs, "endpoint", None)
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RuntimeError("Missing init_kwargs.endpoint for OpenClaw pipeline")
    model_endpoint = _resolve_model_endpoint(
        endpoint,
        getattr(init_kwargs, "model_endpoint", None),
    )
    openclaw_model_api = _resolve_openclaw_model_api(
        instance,
        init_kwargs,
        model_endpoint,
    )
    openclaw_max_tokens = _resolve_openclaw_max_tokens(instance, init_kwargs)
    openclaw_provider = "anthropic" if openclaw_model_api == "anthropic-messages" else "openai"
    stepcast_host_override = getattr(init_kwargs, "stepcast_host_override", None)
    stepcast_host_aliases = getattr(init_kwargs, "stepcast_host_aliases", None)

    sr_user_token = os.getenv("SESSION_ROUTER_USER_TOKEN") or os.getenv("SR_USER_TOKEN")
    if not sr_user_token:
        raise RuntimeError("SESSION_ROUTER_USER_TOKEN is not set")
    sr_endpoint = os.getenv(
        "SESSION_ROUTER_ENDPOINT",
        "https://session-router-basemind.stepfun-inc.com",
    )
    resume_session_id = getattr(init_kwargs, "resume_session_id", None)
    if isinstance(resume_session_id, str):
        resume_session_id = resume_session_id.strip() or None
    else:
        resume_session_id = None

    env_vars = _build_session_env_vars()
    session_client: SessionRouterClient | None = None
    try:
        proxy_snapshot = {
            k: v
            for k, v in env_vars.items()
            if k in {"http_proxy", "https_proxy", "no_proxy"}
        }
        logger.info(
            "openclaw init "
            f"instance_id={instance_id} endpoint={endpoint} model_endpoint={model_endpoint} "
            f"model_name={model_name} stepcast_host_override={stepcast_host_override} "
            f"stepcast_host_aliases={stepcast_host_aliases} openclaw_timeout={openclaw_timeout} "
            f"openclaw_thinking={openclaw_thinking} openclaw_model_api={openclaw_model_api} "
            f"openclaw_provider={openclaw_provider} openclaw_max_tokens={openclaw_max_tokens} "
            f"workspace_root={workspace_root} docker_image={docker_image} "
            f"sr_endpoint={sr_endpoint} resume_session_id={resume_session_id or ''} "
            f"proxies={proxy_snapshot}"
        )

        with stage_ctx("env_init"):
            if resume_session_id:
                session_client = SessionRouterClient(
                    resume_session_id,
                    user_token=sr_user_token,
                    sr_endpoint=sr_endpoint,
                    logger=logger,
                )
                if not session_client.wait_ready(timeout=300, interval=3):
                    raise RuntimeError(
                        f"resume session is not ready: session_id={resume_session_id}"
                    )
                # Keep the session alive across turns; cleanup callbacks will
                # still close it when the StepTron token is deleted/cancelled.
                session_client._start_heartbeat()
            else:
                with record_session_create_metrics():
                    session_client = SessionRouterClient.create_and_wait_ready(
                        docker_image=docker_image,
                        user_token=sr_user_token,
                        endpoint=sr_endpoint,
                        command=["tail", "-f"],
                        request_cpu=DEFAULT_REQUEST_CPU,
                        request_memory=DEFAULT_REQUEST_MEMORY,
                        limit_cpu=DEFAULT_LIMIT_CPU,
                        limit_memory=DEFAULT_LIMIT_MEMORY,
                        env_vars=env_vars,
                        pull_policy=DEFAULT_PULL_POLICY,
                        logger=logger,
                        # Keep session alive even when keep_session_open=True,
                        # otherwise SessionRouter may GC it before a resumed turn.
                        start_heartbeat=True,
                    )

        register_cleanup(session_client.close)
        check_cancel()

        _maybe_patch_stepcast_hosts(
            session_client,
            endpoint=endpoint,
            model_endpoint=model_endpoint,
            host_override=stepcast_host_override,
            aliases_override=stepcast_host_aliases,
            logger=logger,
        )

        impl_root = Path(__file__).resolve().parents[1]
        openclaw_tools = impl_root / "openclaw" / "env_tools" / "openclaw"
        fastapi_assets = resolve_fastapi_bootstrap_assets(consumer_name="OpenClaw")
        fastapi_py_src = fastapi_assets.server_src
        fastapi_init_src = fastapi_assets.init_src

        openclaw_init_src = openclaw_tools / "openclaw_init.sh"
        openclaw_run_src = openclaw_tools / "openclaw_run.sh"
        openclaw_log_extract_src = impl_root / "openclaw" / "openclaw_log_extract.py"
        openclaw_models_proxy_src = fastapi_assets.utils_dir / "models_proxy.py"

        for required in (
            fastapi_py_src,
            openclaw_init_src,
            openclaw_run_src,
            openclaw_log_extract_src,
            openclaw_models_proxy_src,
        ):
            if not required.exists():
                raise RuntimeError(f"missing OpenClaw tooling: {required}")
        runtime_bundle_local_src = _resolve_openclaw_runtime_bundle(openclaw_tools=openclaw_tools)
        openclaw_runtime_bundle_dst = "/tmp/openclaw-runtime-bundle.tar.gz"
        logger.info(
            "openclaw runtime bundle upload "
            f"path={openclaw_runtime_bundle_dst} "
            f"local_src={runtime_bundle_local_src}"
        )

        upload_files: list[tuple[Path, str, bool]] = [
            (runtime_bundle_local_src, "/tmp/openclaw-runtime-bundle.tar.gz", False),
            *fastapi_assets.bundle_upload_files(),
            (fastapi_py_src, "/tmp/fastapi_server.py", False),
            (fastapi_init_src, "/tmp/fastapi_init.sh", True),
            (openclaw_log_extract_src, "/tmp/openclaw_log_extract.py", False),
            (openclaw_init_src, "/tmp/openclaw_init.sh", True),
            (openclaw_run_src, "/tmp/openclaw_run.sh", True),
            (openclaw_models_proxy_src, "/tmp/openclaw_models_proxy.py", False),
        ]

        with stage_ctx("tool_upload"):
            resume_skip_uploads = (
                bool(resume_session_id)
                and (os.getenv("SWE_OPENCLAW_RESUME_SKIP_EXISTING_UPLOADS", "1").strip().lower() in {"1", "true", "yes", "y", "on"})
            )
            for src, dst, make_exec in upload_files:
                if resume_skip_uploads and dst:
                    lowered = dst.lower()
                    if lowered.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".xz")):
                        ret = session_client.exec_bash(
                            f"test -s {shlex.quote(dst)} && echo OK || echo MISSING"
                        )
                        if isinstance(ret, dict) and (ret.get("stdout") or "").strip() == "OK":
                            logger.debug(f"tool_upload skip existing artifact: {dst}")
                            continue
                session_client.upload(str(src), dst)
                if make_exec:
                    _exec_bash_checked(session_client, f"chmod +x {dst}")

            # Prompt file consumed by openclaw_run.sh.
            if system_prompt:
                combined = f"{system_prompt.strip()}\n\n{prompt_text}"
            else:
                combined = prompt_text
            session_client.upload_content(combined, "/tmp/OPENCLAW.md")
            _exec_bash_checked(session_client, f"mkdir -p {workspace_root}")
            _exec_bash_checked(
                session_client,
                "set -e; test -s /tmp/openclaw-runtime-bundle.tar.gz",
                timeout=60,
            )

        check_cancel()
        with stage_ctx("workspace_patch_prepare"):
            prepare_patch_workspace_auto(
                session_client,
                instance,
                workspace_root=workspace_root,
            )

        check_cancel()
        session_id = session_client.session_id

        fastapi_env = build_fastapi_proxy_env(
            model_endpoint=model_endpoint,
            endpoint=endpoint,
            include_stepcast_endpoint=True,
        )
        fastapi_prefix = _format_env_prefix(fastapi_env)
        with stage_ctx("fastapi_init"):
            _exec_bash_checked(
                session_client,
                f"{fastapi_prefix} /tmp/fastapi_init.sh {shlex.quote(model_endpoint)}",
                timeout=600,
                display_command=f"/tmp/fastapi_init.sh {shlex.quote(model_endpoint)}",
            )

        check_cancel()
        with stage_ctx("openclaw_init"):
            _exec_bash_checked(
                session_client,
                "/tmp/openclaw_init.sh",
                timeout=3600,
                display_command="/tmp/openclaw_init.sh",
            )

        timeout_arg = str(int(openclaw_timeout))
        openclaw_env = {
            "OPENCLAW_SESSION_ID": session_id,
            "OPENCLAW_THINKING": openclaw_thinking,
            "SWE_OPENCLAW_MODEL_API": openclaw_model_api,
            "SWE_OPENCLAW_PROVIDER": openclaw_provider,
        }
        if openclaw_max_tokens is not None:
            openclaw_env["OPENCLAW_MAX_TOKENS"] = str(openclaw_max_tokens)
        openclaw_prefix = _format_env_prefix(openclaw_env)
        openclaw_cmd = (
            f"{openclaw_prefix} /tmp/openclaw_run.sh {shlex.quote(str(api_key))} "
            f"{shlex.quote(model_name)} {shlex.quote(workspace_root)} "
            f"{shlex.quote(timeout_arg)} "
            f"> /tmp/fastapi_logs/openclaw_logs_latest.txt 2>&1"
        )
        check_cancel()
        with stage_ctx("openclaw_run"):
            try:
                _exec_bash_checked(
                    session_client,
                    openclaw_cmd,
                    timeout=openclaw_timeout,
                    display_command=(
                        "OPENCLAW_SESSION_ID=<redacted> /tmp/openclaw_run.sh <redacted> "
                        f"{shlex.quote(model_name)} {shlex.quote(workspace_root)} "
                        f"{shlex.quote(timeout_arg)}"
                    ),
                )
            except Exception as exc:
                _diagnose_stepcast_hosts(
                    session_client,
                    endpoint=endpoint,
                    host_override=stepcast_host_override,
                    aliases_override=stepcast_host_aliases,
                    logger=logger,
                )
                log_paths = [
                    "/tmp/fastapi_logs/openclaw_logs_latest.txt",
                    "/tmp/fastapi_logs/openclaw_models_proxy.log",
                    "/tmp/fastapi_log.txt",
                ]
                log_dump: list[str] = []
                for path in log_paths:
                    ret = session_client.exec_bash(f"tail -n 200 {path} || true")
                    if isinstance(ret, dict):
                        stdout = ret.get("stdout") or ""
                        stderr = ret.get("stderr") or ""
                        if stdout or stderr:
                            logger.error(
                                f"openclaw_run tail {path}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                            )
                            if stdout:
                                log_dump.append(f"--- {path} stdout ---\n{stdout}")
                            if stderr:
                                log_dump.append(f"--- {path} stderr ---\n{stderr}")
                if log_dump:
                    raise RuntimeError(
                        f"{exc}\n\nopenclaw_run_logs:\n" + "\n".join(log_dump)
                    ) from exc
                raise exc

        meta: dict[str, Any] = {
            "session_id": session_id,
            "chat_id": session_id,
        }
        request_id = _extract_request_id_from_logs(session_client, session_id, logger=logger)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
            elif request_id.startswith("resp-"):
                request_id = request_id[len("resp-") :]
            meta["request_id"] = request_id

        sft_data = _collect_openclaw_sft_data(session_client, session_id, logger=logger)
        if sft_data:
            meta["sft_data"] = sft_data
        if _is_empty_sft_data(sft_data):
            meta["sft_debug"] = _collect_fastapi_debug_snapshot(session_client)

        openclaw_stdout = _download_text_if_exists(
            session_client, "/tmp/fastapi_logs/openclaw_logs_latest.txt", clip=8000
        )
        if openclaw_stdout:
            meta["openclaw_stdout_tail"] = openclaw_stdout
        maybe_attach_fastapi_logs_tar_base64(
            meta=meta,
            session_client=session_client,
            instance=instance,
            logger=logger,
        )
        with stage_ctx("patch_extract"):
            meta["patch"] = extract_patch_auto(
                session_client,
                instance,
                workspace_root=workspace_root,
            )

        task_trajectory = _build_task_trajectory(meta)
        logger.success(f"openclaw instance finished: {instance_id} session_id={session_id}")
        return task_trajectory
    finally:
        if session_client is not None:
            if not keep_session_open:
                with suppress(Exception):
                    session_client.close()
            # keep_session_open=True: leave client+heartbeat alive.
            # The StepTron job's cleanup callbacks will close the session when
            # the token is deleted/cancelled.
