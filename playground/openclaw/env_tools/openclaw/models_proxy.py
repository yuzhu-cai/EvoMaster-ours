#!/usr/bin/env python3
"""Shared reverse proxy that adds /v1/models support for non-Codex CLIs.

Why: some fastapi_server.bin builds used in StepTron only proxy POST endpoints
(/v1/chat/completions, /v1/responses) but do not implement GET /v1/models.
Several OpenAI-compatible CLIs call /v1/models during startup.

This proxy:
- Serves GET /v1/models and /v1/models/{id} locally (static payload).
- Forwards all other requests to an upstream OpenAI-compatible endpoint
  (typically the existing fastapi_server.bin running on localhost).

Stdlib-only: avoids pip installing FastAPI inside SessionRouter containers.
"""

import argparse
import gzip
import json
import os
import signal
import sys
import threading
import time
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import zlib

try:
    from http.server import ThreadingHTTPServer as _StdlibThreadingHTTPServer
except ImportError:
    _StdlibThreadingHTTPServer = None


_HOP_BY_HOP = {
    "connection",
    "proxy-connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

_STRIP_AUTH_HEADERS = {"authorization", "x-api-key"}

_DEFAULT_STATE_FILE = "/tmp/models_proxy_chat_state.json"
_DEFAULT_DEBUG_LOG_FILE = "/tmp/fastapi_logs/models_proxy_debug.jsonl"

# Kilo Code uses OpenRouter's OpenAI-compatible endpoints, but may include
# OpenRouter-only parameters (e.g. provider/transforms) that strict upstream
# proxies reject with 400. We keep a conservative allowlist for chat payloads.
_CHAT_COMPLETIONS_ALLOWED_KEYS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "max_tokens",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "tools",
    "tool_choice",
    "function_call",
    "functions",
    "seed",
    "response_format",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    # Preserve reasoning/thinking fields from clients that already emit them.
    "thinking",
    "reasoning",
    "reasoning_effort",
}


if _StdlibThreadingHTTPServer is not None:

    class ThreadingHTTPServerCompat(_StdlibThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

else:

    class ThreadingHTTPServerCompat(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True


def _responses_input_to_messages(input_value: object) -> List[dict]:
    """Normalize Responses API `input` to Chat Completions `messages`."""
    if isinstance(input_value, str):
        text = input_value.strip()
        return [{"role": "user", "content": text}] if text else []

    out: List[dict] = []
    if not isinstance(input_value, list):
        return out

    for item in input_value:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "").strip().lower()

        # Tool result (tool -> model).
        if item_type == "function_call_output":
            call_id = str(item.get("call_id") or "").strip()
            if call_id:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _coerce_message_content_to_text(item.get("output")),
                    }
                )
            continue

        # Prior tool call (assistant -> tool).
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            arguments = item.get("arguments")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if call_id and name:
                out.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": str(arguments or "{}"),
                                },
                            }
                        ],
                    }
                )
            continue

        role = str(item.get("role") or "").strip().lower()
        if role:
            if role == "developer":
                role = "system"
            content = _coerce_message_content_to_text(item.get("content"))
            out.append({"role": role, "content": content})
            continue

        if item_type in {"message", "input_text", "text"}:
            role = str(item.get("role") or "user").strip().lower()
            if role == "developer":
                role = "system"
            content = _coerce_message_content_to_text(item.get("content") or item.get("text"))
            out.append({"role": role or "user", "content": content})
            continue
    return out

def _safe_write_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        return


def _normalize_path(raw_path: str) -> str:
    """Canonicalize OpenAI-compatible request paths (preserve query string).

    This proxy is used by multiple CLIs that may format base URLs differently:
    - no version prefix:           /responses
    - standard version prefix:     /v1/responses
    - accidental double prefix:    /v1/v1/responses
    """
    if "?" in raw_path:
        path_part, query = raw_path.split("?", 1)
    else:
        path_part, query = raw_path, ""
    if not path_part:
        path_part = "/"
    if not path_part.startswith("/"):
        path_part = "/" + path_part
    while "//" in path_part:
        path_part = path_part.replace("//", "/")

    # Normalize accidental repeated API version prefixes.
    while path_part.startswith("/v1/v1/"):
        path_part = "/v1/" + path_part[len("/v1/v1/") :]
    if path_part == "/v1/v1":
        path_part = "/v1"

    # Accept both versioned and unversioned OpenAI-compatible routes.
    if path_part == "/responses":
        path_part = "/v1/responses"
    elif path_part == "/chat/completions":
        path_part = "/v1/chat/completions"
    elif path_part == "/models" or path_part.startswith("/models/"):
        path_part = "/v1" + path_part

    return path_part + (("?" + query) if query else "")


def _coerce_message_content_to_text(content: object) -> str:
    """Best-effort convert OpenAI "content parts" into a plain string.

    Some providers (e.g. models-proxy) only accept `content: str`, while clients
    like Kilo Code may send `content: [{"type":"text","text":...}, ...]`.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if part is None:
                continue
            if isinstance(part, str):
                if part:
                    parts.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
                alt = part.get("content")
                if isinstance(alt, str) and alt:
                    parts.append(alt)
                    continue
                parts.append(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
                continue
            parts.append(str(part))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        alt = content.get("content")
        if isinstance(alt, str):
            return alt
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return str(content)


def _normalize_chat_messages(messages: object) -> object:
    if not isinstance(messages, list):
        return messages
    out: List[object] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        msg2 = dict(msg)
        role = str(msg2.get("role") or "").strip().lower()
        if role == "developer":
            role = "system"
        if role:
            msg2["role"] = role
        if "content" in msg2:
            msg2["content"] = _coerce_message_content_to_text(msg2.get("content"))
        out.append(msg2)
    return out


def _normalize_tools_for_chat(raw_tools: object) -> List[dict]:
    """Best-effort normalize heterogeneous tool schemas to Chat Completions tools."""
    if not isinstance(raw_tools, list):
        return []

    out: List[dict] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        t = str(tool.get("type") or "").strip().lower()
        if t != "function":
            continue

        # Already in chat-completions format: {"type":"function","function":{...}}
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            fn2: Dict[str, object] = {"name": name}
            desc = fn.get("description")
            if isinstance(desc, str):
                fn2["description"] = desc
            params = fn.get("parameters")
            if isinstance(params, dict):
                fn2["parameters"] = params
            out.append({"type": "function", "function": fn2})
            continue

        # Responses-like tools: {"type":"function","name":...,"parameters":{...}}
        name = str(tool.get("name") or "").strip()
        params = tool.get("parameters")
        if not name or not isinstance(params, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": params,
                },
            }
        )
    return out


def _normalize_tool_choice(tool_choice: object, *, has_tools: bool) -> object:
    if tool_choice is None:
        return "auto" if has_tools else "none"
    if isinstance(tool_choice, str):
        lowered = tool_choice.strip().lower()
        if lowered in {"auto", "none", "required"}:
            return lowered
        return "auto" if has_tools else "none"
    if isinstance(tool_choice, dict):
        t = str(tool_choice.get("type") or "").strip().lower()
        if t in {"auto", "none", "required"}:
            return t
        if t == "function":
            fn = tool_choice.get("function")
            name = ""
            if isinstance(fn, dict):
                name = str(fn.get("name") or "").strip()
            if not name:
                name = str(tool_choice.get("name") or "").strip()
            if name:
                return {"type": "function", "function": {"name": name}}
        return "auto" if has_tools else "none"
    return "auto" if has_tools else "none"


def _sanitize_chat_payload(payload: dict) -> Tuple[dict, List[str]]:
    """Drop unknown keys and normalize common incompatibilities for strict upstreams."""
    dropped: List[str] = []
    out: Dict[str, object] = {}

    # Map max_completion_tokens -> max_tokens (some OpenAI clients send this).
    if "max_completion_tokens" in payload and "max_tokens" not in payload:
        out["max_tokens"] = payload.get("max_completion_tokens")
        dropped.append("max_completion_tokens")

    for k, v in payload.items():
        if k in _CHAT_COMPLETIONS_ALLOWED_KEYS:
            out[k] = v
        elif k != "max_completion_tokens":
            dropped.append(k)

    # Normalize messages/tool schemas.
    out["messages"] = _normalize_chat_messages(out.get("messages"))

    tools = _normalize_tools_for_chat(out.get("tools"))
    if tools:
        out["tools"] = tools
        out["tool_choice"] = _normalize_tool_choice(out.get("tool_choice"), has_tools=True)
    else:
        out.pop("tools", None)
        if "tool_choice" in out:
            out["tool_choice"] = _normalize_tool_choice(out.get("tool_choice"), has_tools=False)

    # Keep only simple response_format variants; degrade json_schema to json_object.
    if "response_format" in out:
        rf = out.get("response_format")
        if isinstance(rf, dict):
            t = str(rf.get("type") or "").strip().lower()
            if t == "json_schema":
                out["response_format"] = {"type": "json_object"}
            elif t in {"json_object", "text"}:
                out["response_format"] = {"type": t}
            else:
                out.pop("response_format", None)
        else:
            out.pop("response_format", None)

    # Optional guardrail: keep one shared cap knob with FastAPI server.
    raw_cap = (os.getenv("SWE_FASTAPI_MAX_TOKENS") or "").strip()
    if raw_cap:
        try:
            cap = int(raw_cap)
        except Exception:
            cap = 0
        if cap > 0:
            raw_mt = out.get("max_tokens")
            try:
                mt = int(raw_mt) if raw_mt is not None else None
            except Exception:
                mt = None
            if isinstance(mt, int) and mt > cap:
                out["max_tokens"] = cap

    return out, dropped


def _summarize_chat_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, object] = {"keys": sorted([str(k) for k in payload.keys()])}
    model = payload.get("model")
    if isinstance(model, str):
        summary["model"] = model
    msgs = payload.get("messages")
    msg_summ: List[Dict[str, str]] = []
    if isinstance(msgs, list):
        for m in msgs[:12]:
            if not isinstance(m, dict):
                msg_summ.append({"role": "?", "content_type": type(m).__name__})
                continue
            role = str(m.get("role") or "?")
            content_type = type(m.get("content")).__name__
            msg_summ.append({"role": role, "content_type": content_type})
    if msg_summ:
        summary["messages"] = msg_summ
    rf = payload.get("response_format")
    if isinstance(rf, dict):
        summary["response_format_type"] = str(rf.get("type") or "")
    return summary


def _summarize_responses_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, object] = {"keys": sorted([str(k) for k in payload.keys()])}
    model = payload.get("model")
    if isinstance(model, str):
        summary["model"] = model
    summary["stream"] = bool(payload.get("stream", False))
    tools = payload.get("tools")
    if isinstance(tools, list):
        summary["tools_count"] = len(tools)
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, str):
        summary["tool_choice"] = tool_choice
    elif isinstance(tool_choice, dict):
        summary["tool_choice"] = str(tool_choice.get("type") or "dict")

    input_value = payload.get("input")
    if isinstance(input_value, str):
        summary["input_type"] = "str"
        summary["input_chars"] = len(input_value)
    elif isinstance(input_value, list):
        summary["input_type"] = "list"
        item_summaries: List[Dict[str, str]] = []
        for item in input_value[:16]:
            if not isinstance(item, dict):
                item_summaries.append({"type": type(item).__name__})
                continue
            item_summaries.append(
                {
                    "type": str(item.get("type") or ""),
                    "role": str(item.get("role") or ""),
                }
            )
        summary["input_items"] = item_summaries
    return summary


def _summarize_upstream_chat_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, object] = {}
    rid = payload.get("id")
    if isinstance(rid, str):
        summary["id"] = rid
    model = payload.get("model")
    if isinstance(model, str):
        summary["model"] = model
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        summary["choices"] = 0
        return summary

    summary["choices"] = len(choices)
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    if isinstance(choice0, dict):
        finish_reason = choice0.get("finish_reason")
        if isinstance(finish_reason, str):
            summary["finish_reason"] = finish_reason
        msg = choice0.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            summary["content_type"] = type(content).__name__
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                summary["tool_call_count"] = len(tool_calls)
                names: List[str] = []
                for call in tool_calls[:8]:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if isinstance(fn, dict):
                        name = str(fn.get("name") or "").strip()
                        if name:
                            names.append(name)
                if names:
                    summary["tool_call_names"] = names
            reasoning_fields = [
                key
                for key in (
                    "reasoning_content",
                    "reasoning",
                    "reasoningContent",
                    "thinking",
                    "analysis",
                )
                if msg.get(key) is not None
            ]
            if reasoning_fields:
                summary["reasoning_fields"] = reasoning_fields
    usage = payload.get("usage")
    if isinstance(usage, dict):
        usage_summary: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                usage_summary[key] = value
        if usage_summary:
            summary["usage"] = usage_summary
    return summary


def _summarize_responses_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, object] = {}
    status = payload.get("status")
    if isinstance(status, str):
        summary["status"] = status
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        summary["output_text_chars"] = len(output_text)
    output = payload.get("output")
    if isinstance(output, list):
        types: List[str] = []
        function_call_count = 0
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type:
                types.append(item_type)
            if item_type == "function_call":
                function_call_count += 1
        if types:
            summary["output_types"] = types
        if function_call_count:
            summary["function_call_count"] = function_call_count
    return summary


def _append_debug_jsonl(record: object, *, path: str = _DEFAULT_DEBUG_LOG_FILE) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        return


def _should_use_session_fallback(messages: object) -> bool:
    """Heuristic: if caller only sends user/system, we can prepend stored history."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role in {"assistant", "tool"}:
            return False
    return True


def _load_state_messages(path: str) -> List[dict]:
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except Exception:
        return []
    try:
        payload = json.loads(data)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out: List[dict] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def _decode_upstream_body(
    response: http.client.HTTPResponse,
    body: bytes,
) -> bytes:
    if not body:
        return body

    content_encoding = ""
    for key, value in response.getheaders():
        if str(key).lower() == "content-encoding":
            content_encoding = str(value or "").strip().lower()
            break
    if not content_encoding:
        return body

    try:
        if "gzip" in content_encoding or "x-gzip" in content_encoding:
            return gzip.decompress(body)
        if "deflate" in content_encoding:
            return zlib.decompress(body)
    except Exception:
        return body
    return body


def _atomic_write_json(path: str, payload: object) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except Exception:
        # Best-effort; failures should not break the proxy.
        return


def _extract_assistant_text_content(content: object) -> str:
    """Extract assistant visible text while skipping reasoning blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = str(part.get("type") or "").strip().lower()
                if ptype in {"reasoning", "thinking", "analysis"}:
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
                alt = part.get("content")
                if isinstance(alt, str) and alt:
                    parts.append(alt)
                    continue
                if isinstance(alt, (list, dict)):
                    parts.append(_coerce_message_content_to_text(alt))
                    continue
            else:
                text = _coerce_message_content_to_text(part)
                if text:
                    parts.append(text)
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(content, dict):
        ptype = str(content.get("type") or "").strip().lower()
        if ptype in {"reasoning", "thinking", "analysis"}:
            return ""
        text = content.get("text")
        if isinstance(text, str):
            return text
        alt = content.get("content")
        return _coerce_message_content_to_text(alt)
    return _coerce_message_content_to_text(content)


def _extract_assistant_reasoning(message: dict) -> str:
    """Best-effort collect reasoning from multiple provider field variants."""
    parts: List[str] = []

    for key in (
        "reasoning_content",
        "reasoning",
        "reasoningContent",
        "thinking",
        "analysis",
    ):
        raw = message.get(key)
        if raw is None:
            continue
        text = _coerce_message_content_to_text(raw).strip()
        if text:
            parts.append(text)

    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").strip().lower()
            if ptype not in {"reasoning", "thinking", "analysis"}:
                continue
            reason_raw = (
                part.get(ptype)
                if part.get(ptype) is not None
                else part.get("text")
                if part.get("text") is not None
                else part.get("content")
            )
            text = _coerce_message_content_to_text(reason_raw).strip()
            if text:
                parts.append(text)

    dedup: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        dedup.append(part)
    return "\n\n".join(dedup).strip()


def _extract_assistant_message(upstream_json: object) -> Optional[dict]:
    if not isinstance(upstream_json, dict):
        return None
    choices = upstream_json.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    msg = choice0.get("message") or {}
    if not isinstance(msg, dict):
        return None
    role = str(msg.get("role") or "assistant")
    content = _extract_assistant_text_content(msg.get("content"))
    reasoning = _extract_assistant_reasoning(msg)
    if not content and reasoning:
        content = reasoning
    out: Dict[str, object] = {"role": role, "content": content}
    if reasoning:
        out["reasoning_content"] = reasoning
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _extract_assistant_message_from_responses(upstream_json: object) -> Optional[dict]:
    """Best-effort extract assistant turn from a Responses API payload."""
    if not isinstance(upstream_json, dict):
        return None

    output_raw = upstream_json.get("output")
    output = output_raw if isinstance(output_raw, list) else []

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, object]] = []

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()

        if item_type == "function_call":
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            if not call_id:
                call_id = f"call_{int(time.time())}_{len(tool_calls)}"
            arguments = item.get("arguments")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": str(arguments or "{}"),
                    },
                }
            )
            continue

        if item_type == "reasoning":
            text = _coerce_message_content_to_text(
                item.get("text") if item.get("text") is not None else item.get("content")
            ).strip()
            if text:
                reasoning_parts.append(text)
            continue

        if item_type != "message":
            continue

        role = str(item.get("role") or "").strip().lower()
        if role and role != "assistant":
            continue

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "").strip().lower()
                text = _coerce_message_content_to_text(
                    block.get("text") if block.get("text") is not None else block.get("content")
                ).strip()
                if not text:
                    continue
                if block_type in {"reasoning", "thinking", "analysis"}:
                    reasoning_parts.append(text)
                else:
                    text_parts.append(text)
            continue

        text = _extract_assistant_text_content(content).strip()
        if text:
            text_parts.append(text)

    if not text_parts:
        output_text = upstream_json.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            text_parts.append(output_text.strip())

    content_out = "\n".join([p for p in text_parts if p]).strip()
    reasoning_out = "\n\n".join([p for p in reasoning_parts if p]).strip()
    if not content_out and reasoning_out:
        content_out = reasoning_out

    if not content_out and not tool_calls:
        return None

    out: Dict[str, object] = {"role": "assistant", "content": content_out}
    if reasoning_out:
        out["reasoning_content"] = reasoning_out
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _normalize_tool_call_finish_reason(upstream_json: object) -> object:
    """Normalize tool-call turns to finish_reason=tool_calls.

    Some upstream deployments return:
      message.tool_calls != [] and finish_reason == "stop"
    which breaks clients that key the tool loop off finish_reason.
    """
    if not isinstance(upstream_json, dict):
        return upstream_json
    raw_choices = upstream_json.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        return upstream_json

    changed = False
    choices_norm: List[object] = []
    for choice in raw_choices:
        if not isinstance(choice, dict):
            choices_norm.append(choice)
            continue
        msg = choice.get("message")
        has_tool_calls = False
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
            has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if not has_tool_calls:
            choices_norm.append(choice)
            continue

        finish_reason = choice.get("finish_reason")
        finish_reason_lc = (
            finish_reason.strip().lower() if isinstance(finish_reason, str) else ""
        )
        if finish_reason is None or finish_reason_lc in {"", "stop"}:
            choice2 = dict(choice)
            choice2["finish_reason"] = "tool_calls"
            choices_norm.append(choice2)
            changed = True
            continue
        choices_norm.append(choice)

    if not changed:
        return upstream_json
    out = dict(upstream_json)
    out["choices"] = choices_norm
    return out


def _normalize_assistant_message(upstream_json: object) -> object:
    """Fill missing assistant fields from reasoning/tool-call variants when possible."""
    if not isinstance(upstream_json, dict):
        return upstream_json
    raw_choices = upstream_json.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        return upstream_json

    assistant = _extract_assistant_message(upstream_json)
    if not isinstance(assistant, dict):
        return upstream_json

    changed = False
    choices_norm: List[object] = []
    for idx, choice in enumerate(raw_choices):
        if idx != 0 or not isinstance(choice, dict):
            choices_norm.append(choice)
            continue

        message = choice.get("message")
        if not isinstance(message, dict):
            choices_norm.append(choice)
            continue

        choice2 = dict(choice)
        message2 = dict(message)

        role = str(assistant.get("role") or "assistant").strip() or "assistant"
        if str(message2.get("role") or "").strip() != role:
            message2["role"] = role
            changed = True

        normalized_content = _coerce_message_content_to_text(assistant.get("content")).strip()
        if normalized_content and not _extract_assistant_text_content(message2.get("content")).strip():
            message2["content"] = normalized_content
            changed = True

        normalized_tool_calls = assistant.get("tool_calls")
        if (
            isinstance(normalized_tool_calls, list)
            and normalized_tool_calls
            and not (isinstance(message2.get("tool_calls"), list) and message2.get("tool_calls"))
        ):
            message2["tool_calls"] = normalized_tool_calls
            changed = True

        normalized_reasoning = _coerce_message_content_to_text(
            assistant.get("reasoning_content")
        ).strip()
        if normalized_reasoning:
            has_reasoning = any(
                message2.get(key) is not None
                for key in (
                    "reasoning_content",
                    "reasoning",
                    "reasoningContent",
                    "thinking",
                    "analysis",
                )
            )
            if not has_reasoning:
                message2["reasoning_content"] = normalized_reasoning
                changed = True

        choice2["message"] = message2
        choices_norm.append(choice2)

    if not changed:
        return upstream_json

    out = dict(upstream_json)
    out["choices"] = choices_norm
    return out


def _responses_payload_to_chat_payload(payload: dict) -> dict:
    """Best-effort map Responses API request shape to chat.completions."""
    out: Dict[str, object] = {}

    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        out["model"] = model.strip()

    messages = _responses_input_to_messages(payload.get("input"))
    if not messages:
        messages = _normalize_chat_messages(payload.get("messages"))  # type: ignore[assignment]
        if not isinstance(messages, list):
            messages = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages = [{"role": "system", "content": instructions.strip()}] + messages
    out["messages"] = messages

    # Map common generation knobs.
    for key in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
        "user",
        "response_format",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "n",
    ):
        if key in payload:
            out[key] = payload.get(key)

    if "max_output_tokens" in payload and "max_tokens" not in payload:
        out["max_tokens"] = payload.get("max_output_tokens")
    elif "max_tokens" in payload:
        out["max_tokens"] = payload.get("max_tokens")

    tools = _normalize_tools_for_chat(payload.get("tools"))
    if tools:
        out["tools"] = tools
        out["tool_choice"] = _normalize_tool_choice(payload.get("tool_choice"), has_tools=True)
    elif "tool_choice" in payload:
        out["tool_choice"] = _normalize_tool_choice(payload.get("tool_choice"), has_tools=False)

    # Let the existing sanitizer normalize/degrade any strict fields.
    sanitized, _dropped = _sanitize_chat_payload(out)
    return sanitized


def _chat_completion_to_responses_response(
    upstream_json: dict,
    *,
    requested_model: str,
) -> dict:
    """Convert a chat.completion response into a minimal Responses payload."""
    now = int(time.time())
    upstream_id = str(upstream_json.get("id") or f"chatcmpl-proxy-{now}")
    resp_id = upstream_id.replace("chatcmpl", "resp", 1) if upstream_id.startswith("chatcmpl") else upstream_id
    model = str(upstream_json.get("model") or requested_model or "")

    assistant = _extract_assistant_message(upstream_json) or {
        "role": "assistant",
        "content": "",
    }
    text = _coerce_message_content_to_text(assistant.get("content")).strip()
    reasoning = _coerce_message_content_to_text(
        assistant.get("reasoning_content")
    ).strip()
    tool_calls = assistant.get("tool_calls")
    function_call_items: List[Dict[str, object]] = []
    if isinstance(tool_calls, list):
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            arguments = fn.get("arguments")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)

            raw_id = str(tool_call.get("id") or "").strip()
            call_id = raw_id
            item_id = ""
            if raw_id and "|" in raw_id:
                left, right = raw_id.split("|", 1)
                if left:
                    call_id = left
                if right:
                    item_id = right
            if not call_id:
                call_id = f"call_{resp_id}_{idx}"
            if not item_id:
                item_id = f"fc_{call_id}"
            function_call_items.append(
                {
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": str(arguments or "{}"),
                }
            )
    has_function_calls = bool(function_call_items)
    if not has_function_calls and not text and reasoning:
        # Keep compatibility with clients that read `output_text` only.
        text = reasoning

    usage_raw = upstream_json.get("usage")
    usage: Dict[str, object] = {}
    if isinstance(usage_raw, dict):
        prompt_tokens = usage_raw.get("prompt_tokens")
        completion_tokens = usage_raw.get("completion_tokens")
        total_tokens = usage_raw.get("total_tokens")
        if isinstance(prompt_tokens, int):
            usage["input_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["output_tokens"] = completion_tokens
        if isinstance(total_tokens, int):
            usage["total_tokens"] = total_tokens

    content_blocks: List[Dict[str, object]] = []
    if reasoning:
        content_blocks.append(
            {
                "type": "reasoning",
                "text": reasoning,
            }
        )
    if text:
        content_blocks.append(
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        )

    output: List[Dict[str, object]] = []
    if has_function_calls:
        # OpenResponses tool-call turns must stay "incomplete" and should not
        # include assistant text blocks, otherwise some clients stop the loop.
        output.extend(function_call_items)
    else:
        if reasoning:
            output.append(
                {
                    "id": f"rs_{resp_id}",
                    "type": "reasoning",
                    "text": reasoning,
                }
            )
        if content_blocks:
            output.append(
                {
                    "id": f"msg_{resp_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": content_blocks,
                }
            )

    return {
        "id": resp_id,
        "object": "response",
        "created_at": now,
        "status": "incomplete" if has_function_calls else "completed",
        "error": None,
        "model": model,
        "output": output,
        "output_text": "" if has_function_calls else text,
        "usage": usage,
    }


def _maybe_normalize_chat_completion_body(path: str, body: bytes) -> bytes:
    raw_path = _normalize_path(path)
    path_only = raw_path.split("?", 1)[0]
    if path_only != "/v1/chat/completions":
        return body
    if not body:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    messages = payload.get("messages")
    normalized = _normalize_chat_messages(messages)
    if normalized is messages:
        return body
    payload2 = dict(payload)
    payload2["messages"] = normalized
    return json.dumps(payload2, ensure_ascii=False).encode("utf-8")


def _chat_completion_to_chunks(upstream_json: dict) -> List[dict]:
    """Convert a non-stream chat completion response into streaming chunks.

    Many OpenAI clients (incl. Kilo Code) request `stream=true`. Stepcast may not
    support streaming for all deployments; we emulate a minimal SSE stream.
    """
    now = int(time.time())
    resp_id = str(upstream_json.get("id") or f"chatcmpl-proxy-{now}")
    created = upstream_json.get("created")
    if not isinstance(created, int):
        created = now
    model = str(upstream_json.get("model") or "")

    raw_choices = upstream_json.get("choices") or []
    if not isinstance(raw_choices, list):
        raw_choices = []

    out: List[dict] = []
    if not raw_choices:
        out.append(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        return out

    for choice in raw_choices:
        if not isinstance(choice, dict):
            continue
        idx = choice.get("index")
        if not isinstance(idx, int):
            idx = 0
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            msg = {}
        role = str(msg.get("role") or "assistant")
        content = _extract_assistant_text_content(msg.get("content")).strip()
        reasoning = _extract_assistant_reasoning(msg)
        tool_calls = msg.get("tool_calls")
        if not content and isinstance(reasoning, str) and reasoning:
            # Some Stepcast deployments return text in reasoning_content only.
            content = reasoning

        delta: Dict[str, object] = {"role": role}
        if isinstance(content, str) and content:
            delta["content"] = content
        if isinstance(reasoning, str) and reasoning and reasoning != content:
            delta["reasoning_content"] = reasoning
        if isinstance(tool_calls, list) and tool_calls:
            # Minimal tool_calls streaming emulation: emit the full tool_calls list
            # in one chunk. Include `index` fields expected by some OpenAI clients.
            normalized_calls: List[Dict[str, object]] = []
            for i, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                call2: Dict[str, object] = dict(call)
                if "index" not in call2:
                    call2["index"] = i
                normalized_calls.append(call2)
            if normalized_calls:
                delta["tool_calls"] = normalized_calls

        finish_reason = choice.get("finish_reason")
        finish_reason_lc = (
            finish_reason.strip().lower() if isinstance(finish_reason, str) else ""
        )
        if isinstance(tool_calls, list) and tool_calls:
            if finish_reason is None or finish_reason_lc in {"", "stop"}:
                finish_reason = "tool_calls"

        out.append(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": idx, "delta": delta, "finish_reason": None}],
            }
        )
        out.append(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": idx,
                        "delta": {},
                        "finish_reason": str(finish_reason or "stop"),
                    }
                ],
            }
        )
    return out


def _normalize_responses_usage(usage_raw: object) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    if not isinstance(usage_raw, dict):
        usage_raw = {}

    def _pick_int(src: dict, key: str) -> int:
        value = src.get(key)
        return value if isinstance(value, int) and value >= 0 else 0

    input_tokens = _pick_int(usage_raw, "input_tokens")
    output_tokens = _pick_int(usage_raw, "output_tokens")
    total_tokens = _pick_int(usage_raw, "total_tokens")
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    usage["input_tokens"] = input_tokens
    usage["output_tokens"] = output_tokens
    usage["total_tokens"] = total_tokens
    return usage


def _responses_to_stream_events(response_json: dict) -> List[dict]:
    """Convert a final Responses payload into OpenResponses-style SSE events."""
    rid = str(response_json.get("id") or f"resp-proxy-{int(time.time())}")
    model = str(response_json.get("model") or "")
    status = str(response_json.get("status") or "completed")
    created_at = response_json.get("created_at")
    if not isinstance(created_at, int):
        created_at = int(time.time())

    output_raw = response_json.get("output")
    output = output_raw if isinstance(output_raw, list) else []
    usage = _normalize_responses_usage(response_json.get("usage"))

    def _base_response(with_output: List[dict], *, st: str) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": rid,
            "object": "response",
            "created_at": created_at,
            "status": st,
            "model": model,
            "output": with_output,
            "usage": usage,
        }
        err = response_json.get("error")
        if err is not None:
            payload["error"] = err
        output_text = response_json.get("output_text")
        if isinstance(output_text, str):
            payload["output_text"] = output_text
        return payload

    events: List[dict] = []
    events.append({"type": "response.created", "response": _base_response([], st="in_progress")})
    events.append(
        {"type": "response.in_progress", "response": _base_response([], st="in_progress")}
    )

    completed_items: List[dict] = []
    for idx, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        item_id = str(item.get("id") or f"item_{rid}_{idx}").strip() or f"item_{rid}_{idx}"

        if item_type == "function_call":
            full_args = str(item.get("arguments") or "")
            added_item = dict(item)
            added_item["id"] = item_id
            added_item["arguments"] = ""
            added_item["status"] = "in_progress"
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": added_item,
                }
            )
            if full_args:
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": idx,
                        "item_id": item_id,
                        "delta": full_args,
                    }
                )
                events.append(
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": idx,
                        "item_id": item_id,
                        "arguments": full_args,
                    }
                )
            done_item = dict(item)
            done_item["id"] = item_id
            done_item["status"] = "completed"
            events.append(
                {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": done_item,
                }
            )
            completed_items.append(done_item)
            continue

        if item_type == "message":
            added_item = dict(item)
            added_item["id"] = item_id
            added_item["status"] = "in_progress"
            added_item["content"] = []
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": added_item,
                }
            )

            content = item.get("content")
            text_parts: List[str] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type") or "").strip() != "output_text":
                        continue
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            full_text = "".join(text_parts)
            if full_text:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "output_index": idx,
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": full_text,
                    }
                )
                events.append(
                    {
                        "type": "response.output_text.done",
                        "output_index": idx,
                        "item_id": item_id,
                        "content_index": 0,
                        "text": full_text,
                    }
                )

            done_item = dict(item)
            done_item["id"] = item_id
            done_item["status"] = "completed"
            events.append(
                {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": done_item,
                }
            )
            completed_items.append(done_item)
            continue

        generic_item = dict(item)
        generic_item["id"] = item_id
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": generic_item,
            }
        )
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": generic_item,
            }
        )
        completed_items.append(generic_item)

    events.append(
        {
            "type": "response.completed",
            "response": _base_response(completed_items, st=status),
        }
    )
    return events


class _ProxyHandler(BaseHTTPRequestHandler):
    server: "ModelsProxyServer"  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        # Quiet by default; SessionRouter containers already capture logs.
        return

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()
        self.close_connection = True

    def _send_raw_response(
        self,
        *,
        status: int,
        body: bytes,
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()
        self.close_connection = True

    def _build_upstream_headers(
        self,
        *,
        upstream_host: str,
        upstream_port: int,
        body: bytes,
        force_content_length: bool,
        ensure_json_content_type: bool = False,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for k, v in self.headers.items():
            lk = str(k).lower()
            if lk in _HOP_BY_HOP:
                continue
            if bool(getattr(self.server, "strip_auth_headers", False)) and lk in _STRIP_AUTH_HEADERS:
                continue
            headers[str(k)] = str(v)

        inject = str(getattr(self.server, "inject_api_key", "") or "").strip()
        if inject:
            for hk in list(headers.keys()):
                if str(hk).lower() in _STRIP_AUTH_HEADERS:
                    headers.pop(hk, None)
            headers["Authorization"] = f"Bearer {inject}"

        headers["Host"] = f"{upstream_host}:{upstream_port}"
        if force_content_length or body:
            headers["Content-Length"] = str(len(body))
        if ensure_json_content_type and not any(
            str(k).lower() == "content-type" for k in headers.keys()
        ):
            headers["Content-Type"] = "application/json"
        return headers

    def _response_content_type(
        self,
        response: http.client.HTTPResponse,
        *,
        default: str = "application/json; charset=utf-8",
    ) -> str:
        for key, value in response.getheaders():
            if str(key).lower() == "content-type":
                return str(value)
        return default

    def _append_state_from_turn(
        self,
        *,
        new_messages: object,
        upstream_json: object,
    ) -> None:
        if not isinstance(new_messages, list):
            return

        to_append: List[dict] = []
        for msg in new_messages:
            if not isinstance(msg, dict):
                continue
            msg2 = dict(msg)
            if "content" in msg2:
                msg2["content"] = _coerce_message_content_to_text(msg2.get("content"))
            to_append.append(msg2)

        assistant = _extract_assistant_message(upstream_json)
        if assistant:
            to_append.append(assistant)
        if to_append:
            self._append_state_messages(to_append)

    def _append_state_from_responses_turn(
        self,
        *,
        new_messages: object,
        upstream_json: object,
    ) -> None:
        if not isinstance(new_messages, list):
            return

        to_append: List[dict] = []
        for msg in new_messages:
            if not isinstance(msg, dict):
                continue
            msg2 = dict(msg)
            if "content" in msg2:
                msg2["content"] = _coerce_message_content_to_text(msg2.get("content"))
            to_append.append(msg2)

        assistant = _extract_assistant_message_from_responses(upstream_json)
        if assistant:
            to_append.append(assistant)
        if to_append:
            self._append_state_messages(to_append)

    def _handle_models(self) -> bool:
        # Strip query for routing.
        path = _normalize_path(self.path).split("?", 1)[0]
        if not path.startswith("/v1/models"):
            return False

        # /v1/models or /v1/models/<id>
        parts = [p for p in path.split("/") if p]
        model_id = ""
        if len(parts) >= 3:
            # parts: ["v1","models","<id...>"] where <id> may contain slashes (e.g. ccr/foo).
            model_id = "/".join(parts[2:])

        now = int(time.time())
        ctx_env = str(os.getenv("MODELS_PROXY_CONTEXT_WINDOW") or "").strip()
        if not ctx_env:
            ctx_env = str(os.getenv("MODELS_PROXY_CONTEXT_LENGTH") or "").strip()
        try:
            context_window = int(ctx_env) if ctx_env else 131072
        except Exception:
            context_window = 131072
        max_tokens_env = str(os.getenv("SWE_FASTAPI_MAX_TOKENS") or "").strip()
        try:
            configured_max_tokens = int(max_tokens_env) if max_tokens_env else 0
        except Exception:
            configured_max_tokens = 0
        max_tokens = configured_max_tokens if configured_max_tokens > 0 else context_window

        # Roo Code "provider=roo" expects additional metadata on /v1/models entries
        # (name/description/context_window/max_tokens/type/pricing). Extra fields
        # are safe for OpenAI clients, and fix Roo's strict validation.
        def _model_obj(mid: str) -> Dict[str, object]:
            return {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": "",
                "name": mid,
                "description": "",
                "context_window": context_window,
                "context_length": context_window,
                "max_tokens": max_tokens,
                # Roo requires literal("language") here.
                "type": "language",
                # Roo requires {input, output, input_cache_read?, input_cache_write?} as strings.
                "pricing": {
                    "input": "0",
                    "output": "0",
                    "input_cache_read": "0",
                    "input_cache_write": "0",
                },
            }
        if model_id:
            self._send_json(_model_obj(model_id))
            return True

        configured = (getattr(self.server, "model_id", "") or "").strip()
        data = []
        if configured:
            data.append(_model_obj(configured))
        self._send_json({"object": "list", "data": data})
        return True

    def _get_state_messages(self) -> List[dict]:
        lock = getattr(self.server, "state_lock", None)
        if lock is None:
            return []
        with lock:
            msgs = getattr(self.server, "state_messages", None)
            if not isinstance(msgs, list):
                return []
            # Defensive copy.
            out: List[dict] = []
            for m in msgs:
                if isinstance(m, dict):
                    out.append(dict(m))
            return out

    def _append_state_messages(self, new_messages: List[dict]) -> None:
        lock = getattr(self.server, "state_lock", None)
        if lock is None:
            return
        state_file = getattr(self.server, "state_file", _DEFAULT_STATE_FILE)
        max_messages = int(getattr(self.server, "max_state_messages", 200) or 200)
        with lock:
            msgs = getattr(self.server, "state_messages", None)
            if not isinstance(msgs, list):
                msgs = []
            msgs.extend([dict(m) for m in new_messages if isinstance(m, dict)])
            if max_messages > 0 and len(msgs) > max_messages:
                msgs = msgs[-max_messages:]
            self.server.state_messages = msgs
            _atomic_write_json(state_file, msgs)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            length = 0
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _handle_responses(self, body: bytes) -> bool:
        raw_path = _normalize_path(self.path)
        path_only = raw_path.split("?", 1)[0]
        if path_only != "/v1/responses":
            return False
        if not body:
            self._proxy(body=body)
            return True

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._proxy(body=body)
            return True
        if not isinstance(payload, dict):
            self._proxy(body=body)
            return True

        wants_stream = bool(payload.get("stream", False))
        requested_model = str(payload.get("model") or "")
        request_summary = _summarize_responses_payload(payload)

        chat_payload = _responses_payload_to_chat_payload(payload)

        # Preserve best-effort session continuity across turns.
        new_messages = _normalize_chat_messages(chat_payload.get("messages"))
        if isinstance(new_messages, list):
            chat_payload["messages"] = new_messages
            if _should_use_session_fallback(new_messages):
                history = self._get_state_messages()
                if history:
                    chat_payload["messages"] = history + new_messages
        chat_summary = _summarize_chat_payload(chat_payload)

        payload_up = dict(chat_payload)
        payload_up["stream"] = False
        payload_up.pop("stream_options", None)
        upstream_body = json.dumps(payload_up, ensure_ascii=False).encode("utf-8")

        upstream_host = self.server.upstream_host
        upstream_port = self.server.upstream_port
        timeout = getattr(self.server, "upstream_timeout_sec", 3600.0)
        headers = self._build_upstream_headers(
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            body=upstream_body,
            force_content_length=True,
            ensure_json_content_type=True,
        )

        conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
        try:
            conn.request(self.command, "/v1/chat/completions", body=upstream_body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_body_decoded = _decode_upstream_body(resp, resp_body)
            resp_content_type = self._response_content_type(resp)

            if resp.status == 404:
                # Some Stepcast deployments only expose /v1/responses. Fallback to direct pass-through.
                _append_debug_jsonl(
                    {
                        "ts": time.time(),
                        "path": raw_path,
                        "proxy_mode": "responses_to_chat",
                        "upstream": f"http://{upstream_host}:{upstream_port}",
                        "status": 404,
                        "request_summary": request_summary,
                        "chat_payload_summary": chat_summary,
                        "resp_body_trunc": resp_body_decoded[:800].decode(
                            "utf-8", errors="replace"
                        )
                        if resp_body_decoded
                        else "",
                        "fallback": "responses_passthrough",
                    }
                )

                passthrough_body = body
                passthrough_headers = self._build_upstream_headers(
                    upstream_host=upstream_host,
                    upstream_port=upstream_port,
                    body=passthrough_body,
                    force_content_length=True,
                    ensure_json_content_type=True,
                )
                conn2 = http.client.HTTPConnection(
                    upstream_host, upstream_port, timeout=timeout
                )
                try:
                    conn2.request(
                        self.command,
                        raw_path,
                        body=passthrough_body,
                        headers=passthrough_headers,
                    )
                    resp2 = conn2.getresponse()
                    resp2_body = resp2.read()
                    resp2_body_decoded = _decode_upstream_body(resp2, resp2_body)
                    resp2_content_type = self._response_content_type(resp2)

                    if resp2.status != 200:
                        _append_debug_jsonl(
                            {
                                "ts": time.time(),
                                "path": raw_path,
                                "proxy_mode": "responses_passthrough",
                                "upstream": f"http://{upstream_host}:{upstream_port}",
                                "status": int(resp2.status),
                                "request_summary": request_summary,
                                "resp_body_trunc": resp2_body_decoded[:800].decode(
                                    "utf-8", errors="replace"
                                )
                                if resp2_body_decoded
                                else "",
                            }
                        )
                        self._send_raw_response(
                            status=resp2.status,
                            body=resp2_body_decoded,
                            content_type=resp2_content_type,
                        )
                        return True

                    try:
                        upstream_resp_json = json.loads(
                            resp2_body_decoded.decode("utf-8", errors="replace")
                        )
                    except Exception:
                        _append_debug_jsonl(
                            {
                                "ts": time.time(),
                                "path": raw_path,
                                "proxy_mode": "responses_passthrough",
                                "upstream": f"http://{upstream_host}:{upstream_port}",
                                "status": 200,
                                "request_summary": request_summary,
                                "response_summary": {"parse_error": True},
                            }
                        )
                        self._send_raw_response(
                            status=200,
                            body=resp2_body_decoded,
                            content_type=resp2_content_type,
                        )
                        return True

                    self._append_state_from_responses_turn(
                        new_messages=new_messages,
                        upstream_json=upstream_resp_json,
                    )
                    _append_debug_jsonl(
                        {
                            "ts": time.time(),
                            "path": raw_path,
                            "proxy_mode": "responses_passthrough",
                            "upstream": f"http://{upstream_host}:{upstream_port}",
                            "status": 200,
                            "request_summary": request_summary,
                            "response_summary": _summarize_responses_response(
                                upstream_resp_json
                            ),
                        }
                    )

                    if wants_stream and isinstance(upstream_resp_json, dict):
                        events = _responses_to_stream_events(upstream_resp_json)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for event in events:
                            data = json.dumps(
                                event, ensure_ascii=False, separators=(",", ":")
                            ).encode("utf-8")
                            self.wfile.write(b"data: " + data + b"\n\n")
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        self.close_connection = True
                        return True

                    if isinstance(upstream_resp_json, dict):
                        self._send_json(upstream_resp_json, status=200)
                    else:
                        self._send_raw_response(
                            status=200,
                            body=resp2_body_decoded,
                            content_type=resp2_content_type,
                        )
                    return True
                finally:
                    try:
                        conn2.close()
                    except Exception:
                        pass

            if resp.status != 200:
                _append_debug_jsonl(
                    {
                        "ts": time.time(),
                        "path": raw_path,
                        "proxy_mode": "responses_to_chat",
                        "upstream": f"http://{upstream_host}:{upstream_port}",
                        "status": int(resp.status),
                        "request_summary": request_summary,
                        "chat_payload_summary": chat_summary,
                        "resp_body_trunc": resp_body_decoded[:800].decode(
                            "utf-8", errors="replace"
                        )
                        if resp_body_decoded
                        else "",
                    }
                )
                self._send_raw_response(
                    status=resp.status,
                    body=resp_body_decoded,
                    content_type=resp_content_type,
                )
                return True

            try:
                upstream_json = json.loads(resp_body_decoded.decode("utf-8", errors="replace"))
            except Exception:
                _append_debug_jsonl(
                    {
                        "ts": time.time(),
                        "path": raw_path,
                        "proxy_mode": "responses_to_chat",
                        "upstream": f"http://{upstream_host}:{upstream_port}",
                        "status": 200,
                        "request_summary": request_summary,
                        "chat_payload_summary": chat_summary,
                        "upstream_summary": {"parse_error": True},
                    }
                )
                self._send_raw_response(status=200, body=resp_body_decoded)
                return True
            upstream_json = _normalize_tool_call_finish_reason(upstream_json)
            upstream_json = _normalize_assistant_message(upstream_json)

            self._append_state_from_turn(
                new_messages=new_messages,
                upstream_json=upstream_json,
            )

            response_json = _chat_completion_to_responses_response(
                upstream_json if isinstance(upstream_json, dict) else {},
                requested_model=requested_model,
            )
            _append_debug_jsonl(
                {
                    "ts": time.time(),
                    "path": raw_path,
                    "proxy_mode": "responses_to_chat",
                    "upstream": f"http://{upstream_host}:{upstream_port}",
                    "status": 200,
                    "request_summary": request_summary,
                    "chat_payload_summary": chat_summary,
                    "upstream_summary": _summarize_upstream_chat_response(upstream_json),
                    "response_summary": _summarize_responses_response(response_json),
                }
            )

            if wants_stream:
                events = _responses_to_stream_events(response_json)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for event in events:
                    data = json.dumps(
                        event, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return True

            self._send_json(response_json, status=200)
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_chat_completions(self, body: bytes) -> bool:
        raw_path = _normalize_path(self.path)
        path_only = raw_path.split("?", 1)[0]
        if path_only != "/v1/chat/completions":
            return False
        if not body:
            self._proxy(body=body)
            return True
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._proxy(body=body)
            return True
        if not isinstance(payload, dict):
            self._proxy(body=body)
            return True
        wants_stream = bool(payload.get("stream", False))

        # Normalize messages and apply session fallback if client didn't send history.
        payload_norm = dict(payload)
        new_messages = _normalize_chat_messages(payload_norm.get("messages"))
        payload_norm["messages"] = new_messages
        if _should_use_session_fallback(new_messages):
            history = self._get_state_messages()
            if history:
                payload_norm["messages"] = history + (
                    new_messages if isinstance(new_messages, list) else []
                )

        payload_sanitized, dropped_keys = _sanitize_chat_payload(payload_norm)

        # Force non-stream upstream (some upstreams do not support streaming) and emulate SSE.
        payload_up = dict(payload_sanitized)
        payload_up["stream"] = False if wants_stream else bool(payload_up.get("stream", False))
        # Stepcast validates this strictly: stream_options is only allowed when stream=True.
        payload_up.pop("stream_options", None)
        upstream_body = json.dumps(payload_up, ensure_ascii=False).encode("utf-8")

        upstream_host = self.server.upstream_host
        upstream_port = self.server.upstream_port
        timeout = getattr(self.server, "upstream_timeout_sec", 3600.0)
        headers = self._build_upstream_headers(
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            body=upstream_body,
            force_content_length=True,
            ensure_json_content_type=True,
        )

        conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
        try:
            conn.request(self.command, raw_path, body=upstream_body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            resp_body_decoded = _decode_upstream_body(resp, resp_body)

            if resp.status != 200:
                _append_debug_jsonl(
                    {
                        "ts": time.time(),
                        "path": raw_path,
                        "upstream": f"http://{upstream_host}:{upstream_port}",
                        "status": int(resp.status),
                        "dropped_keys": sorted(list(set(dropped_keys))),
                        "payload_summary": _summarize_chat_payload(payload_sanitized),
                        "resp_body_trunc": resp_body_decoded[:800].decode(
                            "utf-8", errors="replace"
                        )
                        if resp_body_decoded
                        else "",
                    }
                )
                self._send_raw_response(
                    status=resp.status,
                    body=resp_body_decoded,
                    content_type=self._response_content_type(resp),
                )
                return True

            try:
                upstream_json = json.loads(resp_body_decoded.decode("utf-8", errors="replace"))
            except Exception:
                # Fallback: return raw upstream body.
                self._send_raw_response(status=200, body=resp_body_decoded)
                return True
            upstream_json = _normalize_tool_call_finish_reason(upstream_json)
            upstream_json = _normalize_assistant_message(upstream_json)

            self._append_state_from_turn(
                new_messages=new_messages,
                upstream_json=upstream_json,
            )

            if not wants_stream:
                if isinstance(upstream_json, dict):
                    self._send_json(upstream_json, status=200)
                else:
                    self._send_raw_response(status=200, body=resp_body_decoded)
                return True

            chunks = _chat_completion_to_chunks(
                upstream_json if isinstance(upstream_json, dict) else {}
            )
            sse_parts: List[bytes] = []
            for chunk in chunks:
                data = json.dumps(chunk, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                sse_parts.append(b"data: " + data + b"\n\n")
            sse_parts.append(b"data: [DONE]\n\n")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            # Avoid chunked encoding; just close the connection at the end.
            self.end_headers()
            for part in sse_parts:
                self.wfile.write(part)
                self.wfile.flush()
            self.close_connection = True
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _proxy(self, *, body: Optional[bytes] = None) -> None:
        upstream_host = self.server.upstream_host
        upstream_port = self.server.upstream_port
        timeout = getattr(self.server, "upstream_timeout_sec", 3600.0)
        if body is None:
            body = self._read_body()
        body = _maybe_normalize_chat_completion_body(self.path, body)
        headers = self._build_upstream_headers(
            upstream_host=upstream_host,
            upstream_port=upstream_port,
            body=body,
            force_content_length=False,
            ensure_json_content_type=False,
        )

        # Normalize request path to avoid forwarding /v1//... to upstream FastAPI.
        norm_path = _normalize_path(self.path)

        conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
        try:
            conn.request(self.command, norm_path, body=body, headers=headers)
            resp = conn.getresponse()

            # Decide framing. If upstream provides Content-Length, forward it.
            upstream_headers = {k.lower(): v for k, v in resp.getheaders()}
            content_length = upstream_headers.get("content-length")

            self.send_response(resp.status)
            for k, v in resp.getheaders():
                lk = str(k).lower()
                if lk in _HOP_BY_HOP:
                    continue
                # We'll control framing below.
                if lk in {"content-length", "connection"}:
                    continue
                self.send_header(k, v)

            if content_length:
                self.send_header("Content-Length", content_length)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(resp.read())
                self.wfile.flush()
                self.close_connection = True
                return

            # Streaming/unknown length: end by closing the connection.
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def do_GET(self) -> None:  # noqa: N802
        if self._handle_models():
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        if self._handle_responses(body):
            return
        if self._handle_chat_completions(body):
            return
        self._proxy(body=body)

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()


class ModelsProxyServer(ThreadingHTTPServerCompat):
    # Avoids mypy confusion for dynamically attached attrs.
    upstream_host: str
    upstream_port: int
    model_id: str
    upstream_timeout_sec: float
    state_file: str
    max_state_messages: int
    state_messages: List[dict]
    state_lock: threading.Lock
    strip_auth_headers: bool
    inject_api_key: str


def main(argv: Optional[List[str]] = None) -> int:
    default_model_id = (
        os.getenv("MODELS_PROXY_MODEL_ID")
        or os.getenv("KILOCODE_MODEL_ID")
        or ""
    ).strip()
    default_state_file = (
        os.getenv("MODELS_PROXY_STATE_FILE")
        or os.getenv("KILOCODE_STATE_FILE")
        or _DEFAULT_STATE_FILE
    )
    default_max_state_messages = (
        os.getenv("MODELS_PROXY_STATE_MAX_MESSAGES")
        or os.getenv("KILOCODE_STATE_MAX_MESSAGES")
        or "200"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="Upstream base URL, e.g. http://localhost:23333")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--port-file", default="/tmp/models_proxy_port.txt")
    parser.add_argument("--pid-file", default="/tmp/models_proxy_pid.txt")
    parser.add_argument("--model-id", default=default_model_id)
    parser.add_argument("--upstream-timeout-sec", type=float, default=3600.0)
    parser.add_argument(
        "--state-file",
        default=default_state_file,
    )
    parser.add_argument(
        "--max-state-messages",
        type=int,
        default=int(default_max_state_messages),
    )
    parser.add_argument(
        "--strip-auth-headers",
        action="store_true",
        help="Drop Authorization/x-api-key when forwarding to upstream (let upstream proxy inject).",
    )
    parser.add_argument(
        "--inject-api-key",
        action="store_true",
        help="Inject Authorization: Bearer <MODEL_PROXY_API_KEY> when forwarding to upstream.",
    )
    args = parser.parse_args(argv)

    parsed = urlparse(args.upstream)
    upstream_host = (parsed.hostname or "").strip() or "127.0.0.1"
    upstream_port = parsed.port
    if upstream_port is None:
        upstream_port = 443 if parsed.scheme == "https" else 80

    httpd = ModelsProxyServer((args.listen_host, args.listen_port), _ProxyHandler)
    httpd.upstream_host = upstream_host
    httpd.upstream_port = int(upstream_port)
    httpd.model_id = str(args.model_id or "").strip()
    httpd.upstream_timeout_sec = float(args.upstream_timeout_sec)
    httpd.state_file = str(args.state_file or _DEFAULT_STATE_FILE)
    httpd.max_state_messages = int(args.max_state_messages or 200)
    httpd.state_lock = threading.Lock()
    httpd.state_messages = _load_state_messages(httpd.state_file)
    httpd.strip_auth_headers = bool(args.strip_auth_headers)
    inject = ""
    if bool(args.inject_api_key):
        inject = (
            os.getenv("MODEL_PROXY_API_KEY")
            or os.getenv("MODELPROXY_APIKEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        if inject.lower() in {"not-required", "not_required", "none"}:
            inject = ""
    httpd.inject_api_key = inject

    actual_port = httpd.server_address[1]
    _safe_write_text(args.port_file, str(actual_port))
    _safe_write_text(args.pid_file, str(os.getpid()))

    stop_event = threading.Event()

    def _handle_stop(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
        stop_event.set()
        try:
            httpd.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
