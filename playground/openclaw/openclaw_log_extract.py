"""Extract ChatML dialogs from FastAPI proxy logs for the OpenClaw pipeline.

This extractor is intentionally "raw-first":
- Keep model outputs as they appear in proxy responses.
- Do NOT merge reasoning into content.
- Do NOT backfill/repair content across turns.
- Preserve tool-call-only assistant turns (`content=""` + `tool_calls`).
"""
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

def _load_json_maybe(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return value

def _header_value(headers, key):
    if not isinstance(headers, dict):
        return ''
    target = key.lower()
    for (k, v) in headers.items():
        if str(k).lower() == target:
            return '' if v is None else str(v)
    return ''

def _iter_log_entries(log_dir):
    if not log_dir.exists():
        return
    files = sorted([p for p in log_dir.glob('*.jsonl') if p.is_file() and (p.stem.isdigit() or p.stem.startswith('fastapi_'))], key=lambda p: p.stat().st_mtime)
    for path in files:
        try:
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    yield item
        except Exception:
            continue

def _extract_chat_messages(request_obj):
    messages = request_obj.get('messages')
    if isinstance(messages, list) and messages:
        out = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role')
            if not isinstance(role, str) or not role:
                continue
            out.append(dict(msg))
        return out or None
    return None

def _extract_responses_messages(request_obj):
    value = request_obj.get('input')
    if isinstance(value, str) and value.strip():
        return [{'role': 'user', 'content': value}]
    if isinstance(value, list) and value:
        if isinstance(value[0], dict) and 'role' in value[0]:
            out = []
            for msg in value:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role')
                if isinstance(role, str) and role.strip().lower() == 'developer':
                    role = 'system'
                if not isinstance(role, str) or not role:
                    continue
                if 'content' not in msg:
                    continue
                out.append(dict(msg))
            return out or None
    return None

def _extract_completions_prompt_messages(request_obj):
    prompt = request_obj.get('prompt')
    if isinstance(prompt, str) and prompt.strip():
        return [{'role': 'user', 'content': prompt}]
    for key in ('prompt_text', 'input_text'):
        value = request_obj.get(key)
        if isinstance(value, str) and value.strip():
            return [{'role': 'user', 'content': value}]
    return None

def _extract_system_messages(request_obj):
    """Best-effort extract system/developer instructions from request payload."""
    items = []

    def _append(raw):
        text = _content_to_text(raw, include_reasoning=False).strip()
        if not text:
            return
        items.append({'role': 'system', 'content': text})
    if 'system' in request_obj:
        system_raw = request_obj.get('system')
        if isinstance(system_raw, list):
            for part in system_raw:
                if isinstance(part, dict):
                    _append(part.get('text') or part.get('content') or part)
                else:
                    _append(part)
        elif isinstance(system_raw, dict):
            _append(system_raw.get('text') or system_raw.get('content') or system_raw)
        else:
            _append(system_raw)
    if 'instructions' in request_obj:
        _append(request_obj.get('instructions'))
    return items
_REASONING_KEYS = ('reasoning_content', 'reasoning', 'reasoningContent', 'thinking', 'analysis')
_REASONING_TYPES = {'reasoning', 'analysis', 'thinking'}

def _content_to_text(value, *, include_reasoning=True):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_to_text(item, include_reasoning=include_reasoning) for item in value]
        return '\n'.join([p for p in parts if p])
    if isinstance(value, dict):
        item_type = str(value.get('type') or '').strip().lower()
        if not include_reasoning and item_type in _REASONING_TYPES:
            return ''
        keys = ('text', 'content', 'output_text', 'input_text') + (_REASONING_KEYS if include_reasoning else ())
        for key in keys:
            if key not in value:
                continue
            text = _content_to_text(value.get(key), include_reasoning=include_reasoning)
            if text:
                return text
        return ''
    return str(value)

def _compact_json(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return str(value)

def _extract_assistant_reasoning(message):
    for key in _REASONING_KEYS:
        if key not in message:
            continue
        text = _content_to_text(message.get(key), include_reasoning=True).strip()
        if text:
            return text
    content = message.get('content')
    parts = []
    if isinstance(content, list):
        for block in content:
            (_text, reasoning) = _extract_text_reasoning_from_anthropic_block(block)
            reasoning = reasoning.strip()
            if reasoning:
                parts.append(reasoning)
    elif isinstance(content, dict):
        (_text, reasoning) = _extract_text_reasoning_from_anthropic_block(content)
        reasoning = reasoning.strip()
        if reasoning:
            parts.append(reasoning)
    if parts:
        deduped = []
        seen = set()
        for item in parts:
            key = ' '.join(item.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        if deduped:
            return '\n\n'.join(deduped).strip()
    return ''

def _normalize_tool_call(raw, *, index):
    if not isinstance(raw, dict):
        return None
    call_id = raw.get('id') or raw.get('tool_call_id') or raw.get('call_id')
    if isinstance(call_id, str):
        call_id = call_id.strip()
    elif call_id is not None:
        call_id = str(call_id).strip()
    if not call_id:
        call_id = ''.join(['tool_call_', str(index)])
    function = raw.get('function')
    fn_name = None
    fn_args = None
    if isinstance(function, dict):
        fn_name = function.get('name')
        fn_args = function.get('arguments')
    if not fn_name:
        fn_name = raw.get('name')
    if isinstance(fn_name, str):
        fn_name = fn_name.strip()
    elif fn_name is not None:
        fn_name = str(fn_name).strip()
    if not fn_name:
        return None
    if fn_args is None:
        fn_args = raw.get('arguments')
    if fn_args is None:
        fn_args = '{}'
    fn_args_str = fn_args if isinstance(fn_args, str) else _compact_json(fn_args)
    return {'type': 'function', 'id': call_id, 'function': {'name': fn_name, 'arguments': fn_args_str}}

def _normalize_tool_calls(raw):
    if not isinstance(raw, list):
        return []
    out = []
    for (index, item) in enumerate(raw):
        norm = _normalize_tool_call(item, index=index)
        if norm is not None:
            out.append(norm)
    return out

def _extract_tool_calls_from_anthropic_content(content):
    blocks = []
    if isinstance(content, list):
        blocks = content
    elif isinstance(content, dict):
        blocks = [content]
    if not blocks:
        return []
    tool_calls = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        if block_type != 'tool_use':
            continue
        norm = _normalize_tool_call({'id': block.get('id'), 'name': block.get('name'), 'arguments': block.get('input')}, index=len(tool_calls))
        if norm is not None:
            tool_calls.append(norm)
    return tool_calls

def _sanitize_assistant_message(message):
    out = dict(message)
    out['role'] = 'assistant'
    raw_content = out.get('content')
    if raw_content is None:
        out['content'] = ''
    elif isinstance(raw_content, (str, list, dict)):
        out['content'] = raw_content
    else:
        out['content'] = _content_to_text(raw_content, include_reasoning=False)
    tool_calls = _normalize_tool_calls(out.get('tool_calls'))
    if tool_calls:
        out['tool_calls'] = tool_calls
    elif 'tool_calls' in out:
        out['tool_calls'] = []
    if 'reasoning_content' not in out:
        reasoning = _extract_assistant_reasoning(out)
        if reasoning:
            out['reasoning_content'] = reasoning
    return out

def _is_probe_request(request_obj, messages):
    if len(messages) != 1:
        return False
    msg = messages[0]
    role = str(msg.get('role') or '').strip().lower()
    if role != 'user':
        return False
    text = _content_to_text(msg.get('content'), include_reasoning=False).strip().lower()
    if text != 'ping':
        return False
    max_output_tokens = request_obj.get('max_output_tokens')
    max_tokens = request_obj.get('max_tokens')
    if str(max_output_tokens).strip() == '1' or str(max_tokens).strip() == '1':
        return True
    tools = request_obj.get('tools')
    if (not isinstance(tools, list) or not tools) and isinstance(request_obj.get('functions'), list):
        tools = request_obj.get('functions')
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get('name') or '').strip().lower()
            if not name:
                fn = tool.get('function')
                if isinstance(fn, dict):
                    name = str(fn.get('name') or '').strip().lower()
            if name == 'noop':
                return True
    return True

def _extract_request_tools(request_obj):
    tools = request_obj.get('tools')
    if isinstance(tools, list) and tools:
        return [dict(item) for item in tools if isinstance(item, dict)]
    functions = request_obj.get('functions')
    if isinstance(functions, list) and functions:
        converted = []
        for item in functions:
            if not isinstance(item, dict):
                continue
            fn_name = item.get('name')
            if not isinstance(fn_name, str) or not fn_name.strip():
                continue
            params = item.get('parameters')
            if not isinstance(params, dict):
                params = {'type': 'object', 'properties': {}, 'additionalProperties': True}
            converted.append({'type': 'function', 'function': {'name': fn_name.strip(), 'description': str(item.get('description') or ''), 'parameters': params}})
        return converted
    return []

def _extract_assistant_from_chat_response(obj):
    choices = obj.get('choices') or []
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if not isinstance(choice, dict):
        return None
    message = choice.get('message')
    if isinstance(message, dict):
        return _sanitize_assistant_message(message)
    text = choice.get('text')
    if isinstance(text, str):
        return {'role': 'assistant', 'content': text}
    return None

def _extract_assistant_from_responses_response(obj):
    assistant = {'role': 'assistant'}
    for key in ('output_text', 'text'):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            assistant['content'] = value
            return _sanitize_assistant_message(assistant)
    output = obj.get('output')
    if not isinstance(output, list):
        return None
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get('type') or '').strip().lower()
        if item_type in {'function_call', 'tool_call', 'response.function_call'}:
            norm = _normalize_tool_call(item, index=len(tool_calls))
            if norm is not None:
                tool_calls.append(norm)
            continue
        if item_type == 'reasoning':
            reason = item.get('content')
            if reason is None:
                reason = item.get('text')
            reason_text = _content_to_text(reason, include_reasoning=True).strip()
            if reason_text:
                reasoning_parts.append(reason_text)
            continue
        if item_type == 'message':
            content = item.get('content')
            if isinstance(content, list):
                for chunk in content:
                    if not isinstance(chunk, dict):
                        continue
                    chunk_type = str(chunk.get('type') or '').strip().lower()
                    if chunk_type in {'reasoning', 'analysis', 'thinking'}:
                        reason_text = _content_to_text(chunk, include_reasoning=True).strip()
                        if reason_text:
                            reasoning_parts.append(reason_text)
                        continue
                    if chunk_type in {'function_call', 'tool_call', 'response.function_call'}:
                        norm = _normalize_tool_call(chunk, index=len(tool_calls))
                        if norm is not None:
                            tool_calls.append(norm)
                        continue
                    if isinstance(chunk.get('text'), str):
                        text_parts.append(chunk['text'])
                    else:
                        text = _content_to_text(chunk, include_reasoning=False).strip()
                        if text:
                            text_parts.append(text)
            else:
                text = _content_to_text(content, include_reasoning=False).strip()
                if text:
                    text_parts.append(text)
            continue
        if isinstance(item.get('text'), str):
            text_parts.append(item['text'])
    text = '\n'.join([p.strip() for p in text_parts if p and p.strip()]).strip()
    reasoning = '\n\n'.join([p.strip() for p in reasoning_parts if p and p.strip()]).strip()
    if text:
        assistant['content'] = text
    if reasoning:
        assistant['reasoning_content'] = reasoning
    if tool_calls:
        assistant['tool_calls'] = tool_calls
    if not any((key in assistant for key in ('content', 'reasoning_content', 'tool_calls'))):
        return None
    return _sanitize_assistant_message(assistant)

def _extract_text_reasoning_from_anthropic_block(block):
    if not isinstance(block, dict):
        text = _content_to_text(block, include_reasoning=False).strip()
        return (text, '')
    block_type = str(block.get('type') or '').strip().lower()
    text = ''
    reasoning = ''
    if block_type in {'thinking', 'analysis', 'reasoning', 'redacted_thinking'}:
        reasoning = _content_to_text(block.get('thinking') or block.get('analysis') or block.get('reasoning') or block.get('redacted_thinking') or block.get('text') or block.get('content'), include_reasoning=True).strip()
        return ('', reasoning)
    if block_type in {'text', 'output_text', 'input_text', 'text_delta'}:
        text = _content_to_text(block.get('text') or block.get('output_text') or block.get('input_text') or block.get('content'), include_reasoning=False).strip()
        return (text, '')
    text = _content_to_text(block.get('text') or block.get('content'), include_reasoning=False).strip()
    reasoning = _content_to_text(block.get('thinking') or block.get('analysis') or block.get('reasoning'), include_reasoning=True).strip()
    return (text, reasoning)

def _extract_assistant_from_messages_response(obj):
    role = str(obj.get('role') or '').strip().lower()
    if role and role != 'assistant':
        return None
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    content = obj.get('content')
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and str(block.get('type') or '').strip().lower() == 'tool_use':
                norm = _normalize_tool_call({'id': block.get('id'), 'name': block.get('name'), 'arguments': block.get('input')}, index=len(tool_calls))
                if norm is not None:
                    tool_calls.append(norm)
                continue
            (text, reasoning) = _extract_text_reasoning_from_anthropic_block(block)
            if text:
                text_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
    else:
        text = _content_to_text(content, include_reasoning=False).strip()
        if text:
            text_parts.append(text)
    top_reasoning = _content_to_text(obj.get('thinking') or obj.get('analysis') or obj.get('reasoning'), include_reasoning=True).strip()
    if top_reasoning:
        reasoning_parts.append(top_reasoning)
    text = '\n'.join([p for p in text_parts if p]).strip()
    reasoning = '\n\n'.join([p for p in reasoning_parts if p]).strip()
    assistant = {'role': 'assistant'}
    if text:
        assistant['content'] = text
    if reasoning:
        assistant['reasoning_content'] = reasoning
    if tool_calls:
        assistant['tool_calls'] = _normalize_tool_calls(tool_calls)
    if not any((key in assistant for key in ('content', 'reasoning_content', 'tool_calls'))):
        return None
    return _sanitize_assistant_message(assistant)

def _parse_openai_chat_sse(sse):
    if not isinstance(sse, str):
        return None
    text_parts = []
    reasoning_parts = []
    streamed_tool_calls = {}
    for line in sse.splitlines():
        line = line.strip()
        if not line.startswith('data:'):
            continue
        payload = line[len('data:'):].strip()
        if not payload or payload == '[DONE]':
            continue
        try:
            event = json.loads(payload)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        choices = event.get('choices') or []
        if not isinstance(choices, list) or not choices:
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get('content')
            if isinstance(content, str):
                text_parts.append(content)
            reasoning = delta.get('reasoning_content')
            if reasoning is None:
                reasoning = delta.get('reasoning')
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
            raw_tool_calls = delta.get('tool_calls')
            if isinstance(raw_tool_calls, list):
                for raw_call in raw_tool_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    try:
                        idx = int(raw_call.get('index', len(streamed_tool_calls)))
                    except Exception:
                        idx = len(streamed_tool_calls)
                    slot = streamed_tool_calls.setdefault(idx, {})
                    if 'id' in raw_call:
                        slot['id'] = raw_call.get('id')
                    if 'tool_call_id' in raw_call and 'id' not in slot:
                        slot['id'] = raw_call.get('tool_call_id')
                    if 'call_id' in raw_call and 'id' not in slot:
                        slot['id'] = raw_call.get('call_id')
                    fn = raw_call.get('function')
                    if isinstance(fn, dict):
                        slot_fn = slot.setdefault('function', {})
                        if isinstance(fn.get('name'), str):
                            slot_fn['name'] = ''.join([str(slot_fn.get('name', '')), str(fn['name'])])
                        if isinstance(fn.get('arguments'), str):
                            slot_fn['arguments'] = ''.join([str(slot_fn.get('arguments', '')), str(fn['arguments'])])
            if isinstance(choice.get('text'), str):
                text_parts.append(choice['text'])
    joined_text = ''.join(text_parts)
    joined_reasoning = ''.join(reasoning_parts).strip()
    tool_calls = [norm for (_, norm) in sorted(((idx, _normalize_tool_call(raw, index=idx)) for (idx, raw) in streamed_tool_calls.items()), key=lambda pair: pair[0]) if norm is not None]
    if not joined_text and (not joined_reasoning) and (not tool_calls):
        return None
    assistant = {'role': 'assistant', 'content': joined_text}
    if joined_reasoning:
        assistant['reasoning_content'] = joined_reasoning
    if tool_calls:
        assistant['tool_calls'] = tool_calls
    return _sanitize_assistant_message(assistant)

def _parse_openai_responses_sse(sse):
    if not isinstance(sse, str):
        return None
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    for line in sse.splitlines():
        line = line.strip()
        if not line.startswith('data:'):
            continue
        payload = line[len('data:'):].strip()
        if not payload or payload == '[DONE]':
            continue
        try:
            event = json.loads(payload)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        for key in ('item', 'response'):
            candidate = event.get(key)
            if key == 'item' and isinstance(candidate, dict):
                candidate = {'output': [candidate]}
            if not isinstance(candidate, dict):
                continue
            assistant = _extract_assistant_from_responses_response(candidate)
            if not isinstance(assistant, dict):
                continue
            text = _content_to_text(assistant.get('content'), include_reasoning=False).strip()
            if text:
                text_parts.append(text)
            reasoning = _extract_assistant_reasoning(assistant)
            if reasoning:
                reasoning_parts.append(reasoning)
            tool_calls.extend(_normalize_tool_calls(assistant.get('tool_calls')))
    if not text_parts and (not reasoning_parts) and (not tool_calls):
        return None
    dedup_reason = []
    seen_reason = set()
    for part in reasoning_parts:
        if part and part not in seen_reason:
            seen_reason.add(part)
            dedup_reason.append(part)
    dedup_text = []
    seen_text = set()
    for part in text_parts:
        if part and part not in seen_text:
            seen_text.add(part)
            dedup_text.append(part)
    assistant = {'role': 'assistant', 'content': '\n'.join(dedup_text)}
    if dedup_reason:
        assistant['reasoning_content'] = '\n\n'.join(dedup_reason)
    if tool_calls:
        assistant['tool_calls'] = _normalize_tool_calls(tool_calls)
    return _sanitize_assistant_message(assistant)

def _parse_anthropic_messages_sse(sse):
    if not isinstance(sse, str):
        return None
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    tool_call_index_by_block_index = {}
    tool_arg_chunks_by_call_index = {}

    def _parse_block_index(value):
        try:
            idx = int(value)
        except Exception:
            return None
        return idx if idx >= 0 else None
    for line in sse.splitlines():
        line = line.strip()
        if not line.startswith('data:'):
            continue
        payload = line[len('data:'):].strip()
        if not payload or payload == '[DONE]':
            continue
        try:
            event = json.loads(payload)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get('type') or '').strip().lower()
        if event_type == 'content_block_start':
            block = event.get('content_block')
            if isinstance(block, dict) and str(block.get('type') or '').strip().lower() == 'tool_use':
                block_index = _parse_block_index(event.get('index'))
                call_index = len(tool_calls)
                norm = _normalize_tool_call({'id': block.get('id'), 'name': block.get('name'), 'arguments': block.get('input')}, index=call_index)
                if norm is not None:
                    tool_calls.append(norm)
                    if block_index is not None:
                        tool_call_index_by_block_index[block_index] = call_index
                continue
            (text, reasoning) = _extract_text_reasoning_from_anthropic_block(block)
            if text:
                text_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
            continue
        if event_type == 'content_block_delta':
            delta = event.get('delta')
            if isinstance(delta, dict):
                delta_type = str(delta.get('type') or '').strip().lower()
                if delta_type == 'input_json_delta':
                    partial_json = delta.get('partial_json')
                    if partial_json is None:
                        continue
                    partial_chunk = partial_json if isinstance(partial_json, str) else _compact_json(partial_json)
                    call_index = None
                    block_index = _parse_block_index(event.get('index'))
                    if block_index is not None:
                        call_index = tool_call_index_by_block_index.get(block_index)
                    if call_index is None and tool_calls:
                        call_index = len(tool_calls) - 1
                    if call_index is not None:
                        tool_arg_chunks_by_call_index.setdefault(call_index, []).append(partial_chunk)
                    continue
                if delta_type in {'thinking_delta', 'reasoning_delta'}:
                    reasoning = _content_to_text(delta.get('thinking') or delta.get('reasoning') or delta.get('analysis') or delta.get('text'), include_reasoning=True)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    continue
                if delta_type in {'text_delta', 'output_text_delta'}:
                    raw_text = delta.get('text')
                    if isinstance(raw_text, str):
                        text_parts.append(raw_text)
                    else:
                        text = _content_to_text(raw_text, include_reasoning=False).strip()
                        if text:
                            text_parts.append(text)
                    continue
            (text, reasoning) = _extract_text_reasoning_from_anthropic_block(delta)
            if text:
                text_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
            continue
        if event_type in {'message', 'message_start'}:
            message_obj = event.get('message')
            if isinstance(message_obj, dict):
                assistant = _extract_assistant_from_messages_response(message_obj)
                if isinstance(assistant, dict):
                    text = _content_to_text(assistant.get('content'), include_reasoning=False).strip()
                    if text:
                        text_parts.append(text)
                    reasoning = _extract_assistant_reasoning(assistant)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    tool_calls.extend(_normalize_tool_calls(assistant.get('tool_calls')))
    if tool_arg_chunks_by_call_index:
        for (call_index, chunks) in tool_arg_chunks_by_call_index.items():
            if call_index < 0 or call_index >= len(tool_calls):
                continue
            merged_args = ''.join(chunks).strip()
            if not merged_args:
                continue
            parsed_args = _load_json_maybe(merged_args)
            if isinstance(parsed_args, (dict, list)):
                arguments_str = _compact_json(parsed_args)
            else:
                arguments_str = merged_args
            function = tool_calls[call_index].get('function')
            if isinstance(function, dict):
                function['arguments'] = arguments_str
    if not text_parts and (not reasoning_parts) and (not tool_calls):
        return None
    assistant = {'role': 'assistant', 'content': ''.join(text_parts)}
    if reasoning_parts:
        assistant['reasoning_content'] = ''.join(reasoning_parts).strip()
    if tool_calls:
        assistant['tool_calls'] = _normalize_tool_calls(tool_calls)
    return _sanitize_assistant_message(assistant)

def _extract_assistant_content(resp_raw):
    obj = _load_json_maybe(resp_raw)
    if isinstance(obj, dict):
        if 'choices' in obj:
            return _extract_assistant_from_chat_response(obj)
        if 'output' in obj or 'output_text' in obj or 'text' in obj:
            return _extract_assistant_from_responses_response(obj)
        if obj.get('type') == 'message' or 'stop_reason' in obj:
            return _extract_assistant_from_messages_response(obj)
        return None
    if isinstance(obj, str):
        assistant = _parse_openai_chat_sse(obj)
        if assistant is not None:
            return assistant
        assistant = _parse_openai_responses_sse(obj)
        if assistant is not None:
            return assistant
        return _parse_anthropic_messages_sse(obj)
    return None

def _message_has_trace_payload(msg):
    role = str(msg.get('role') or '').strip().lower()
    if role == 'assistant':
        content = msg.get('content')
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and content:
            if any((_content_to_text(item, include_reasoning=False).strip() for item in content)):
                return True
        if isinstance(content, dict) and _content_to_text(content, include_reasoning=False).strip():
            return True
        if _extract_assistant_reasoning(msg):
            return True
        if _normalize_tool_calls(msg.get('tool_calls')):
            return True
        return msg.get('function_call') is not None
    if role in {'tool', 'function'}:
        if msg.get('content') is not None:
            return True
        for key in ('tool_call_id', 'toolUseId', 'tool_use_id', 'call_id', 'id', 'name'):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False
    return msg.get('content') is not None

def _is_tool_result_user_message(msg):
    role = str(msg.get('role') or '').strip().lower()
    if role != 'user':
        return False
    content = msg.get('content')
    if not isinstance(content, list) or not content:
        return False
    has_tool_result = False
    for item in content:
        if not isinstance(item, dict):
            return False
        item_type = str(item.get('type') or '').strip().lower()
        if item_type != 'tool_result':
            return False
        has_tool_result = True
    return has_tool_result

def _normalize_tool_result_user_message(msg):
    out = dict(msg)
    out.setdefault('origin_role', str(msg.get('role') or 'user'))
    out['role'] = 'tool'
    content = out.get('content')
    if isinstance(content, list):
        tool_call_id = ''
        for item in content:
            if not isinstance(item, dict):
                continue
            raw_id = item.get('tool_use_id') or item.get('tool_call_id') or item.get('call_id') or item.get('id')
            if isinstance(raw_id, str) and raw_id.strip():
                tool_call_id = raw_id.strip()
                break
        if tool_call_id:
            out['tool_call_id'] = tool_call_id
    return out

def _clean_messages(messages):
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        if not isinstance(role, str) or not role:
            continue
        copied = dict(msg)
        role_l = role.strip().lower()
        if role_l == 'developer':
            copied.setdefault('origin_role', str(role))
            copied['role'] = 'system'
            role_l = 'system'
        if _is_tool_result_user_message(copied):
            copied = _normalize_tool_result_user_message(copied)
            role_l = 'tool'
        if role_l == 'assistant':
            copied = _sanitize_assistant_message(copied)
        if not _message_has_trace_payload(copied):
            continue
        cleaned.append(copied)
    return cleaned

def _is_success_status(entry):
    raw_status = entry.get('status_code')
    if raw_status is None:
        raw_status = entry.get('code')
    if raw_status is None:
        return True
    try:
        status = int(raw_status)
    except Exception:
        return True
    return 200 <= status < 300

def _collect_turn_entries(entries, session_id):
    matched = []
    for entry in entries:
        if not _is_success_status(entry):
            continue
        headers = entry.get('header') or {}
        if session_id:
            sid = _header_value(headers, 'x-session-id') or _header_value(headers, 'x-chat-id')
            if sid != session_id:
                continue
        req = _load_json_maybe(entry.get('request'))
        if not isinstance(req, dict):
            continue
        messages = _extract_chat_messages(req) or _extract_responses_messages(req) or _extract_completions_prompt_messages(req)
        if not messages:
            continue
        system_messages = _extract_system_messages(req)
        if system_messages and (not any((str(msg.get('role') or '').strip().lower() in {'system', 'developer'} for msg in messages if isinstance(msg, dict)))):
            messages = system_messages + messages
        is_probe = _is_probe_request(req, messages)
        assistant_content = _extract_assistant_content(entry.get('response'))
        cleaned_messages = _clean_messages(messages)
        if not cleaned_messages:
            continue
        req_tools = _extract_request_tools(req)
        turn_meta = {}
        for key in ('model', 'tool_choice', 'parallel_tool_calls', 'response_format', 'temperature', 'top_p', 'max_tokens', 'max_output_tokens', 'stream'):
            if key in req:
                turn_meta[key] = req.get(key)
        matched.append((cleaned_messages, assistant_content, is_probe, req_tools, turn_meta))
    return matched

def _append_assistant_message(messages, content):
    if content is None:
        return
    if isinstance(content, dict):
        msg = _sanitize_assistant_message(content)
        if not _message_has_trace_payload(msg):
            return
        messages.append(msg)
        return
    if isinstance(content, str):
        messages.append({'role': 'assistant', 'content': content})
        return
    messages.append({'role': 'assistant', 'content': _content_to_text(content, include_reasoning=False)})

def _message_fingerprint(msg):

    def _strip_volatile(value):
        if isinstance(value, list):
            return [_strip_volatile(item) for item in value]
        if isinstance(value, dict):
            volatile_keys = {'cache_control', 'cacheControl', 'timestamp', 'meta', 'name', 'origin_role', 'reasoning_signature', 'reasoningSignature', 'id', 'tool_call_id', 'toolUseId', 'tool_use_id', 'call_id'}
            return {str(k): _strip_volatile(v) for (k, v) in value.items() if str(k) not in volatile_keys}
        return value

    def _normalize_text(value, *, include_reasoning=False):
        text = _content_to_text(value, include_reasoning=include_reasoning)
        return ' '.join(text.split())
    role = str(msg.get('role') or '').strip().lower()
    if role == 'developer':
        role = 'system'
    content_value = _strip_volatile(msg.get('content'))
    if role in {'system', 'user', 'tool', 'function'}:
        content_value = _normalize_text(content_value, include_reasoning=False)
    canonical = {'role': role, 'content': content_value}
    if role == 'assistant':
        tool_calls = _normalize_tool_calls(msg.get('tool_calls'))
        if not tool_calls:
            tool_calls = _extract_tool_calls_from_anthropic_content(msg.get('content'))
        if tool_calls:
            canonical['tool_calls'] = tool_calls
        reasoning = _extract_assistant_reasoning(msg).strip()
        if reasoning:
            canonical['reasoning_content'] = reasoning
        if msg.get('function_call') is not None:
            canonical['function_call'] = _strip_volatile(msg.get('function_call'))
    elif role in {'tool', 'function'}:
        for key in ('tool_call_id', 'toolUseId', 'tool_use_id', 'call_id', 'id', 'name'):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                canonical[key] = value.strip()
    try:
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        return _compact_json(canonical)

def _merge_turn_messages(existing, incoming):
    if not incoming:
        return existing
    if not existing:
        return [dict(msg) for msg in incoming]
    cur = [dict(msg) for msg in existing]
    inc = [dict(msg) for msg in incoming]
    cur_keys = [_message_fingerprint(msg) for msg in cur]
    inc_keys = [_message_fingerprint(msg) for msg in inc]
    if len(cur_keys) <= len(inc_keys) and cur_keys == inc_keys[:len(cur_keys)]:
        cur.extend(inc[len(cur_keys):])
        return cur
    if len(inc_keys) <= len(cur_keys) and inc_keys == cur_keys[:len(inc_keys)]:
        return cur
    lcp = _longest_common_prefix_len(cur, inc)
    if lcp >= 2:
        cur.extend(inc[lcp:])
        return _dedup_adjacent_messages(cur)
    if len(inc_keys) <= len(cur_keys) and cur_keys[-len(inc_keys):] == inc_keys:
        return cur
    max_overlap = min(len(cur_keys), len(inc_keys))
    for overlap in range(max_overlap, 0, -1):
        if cur_keys[-overlap:] == inc_keys[:overlap]:
            cur.extend(inc[overlap:])
            return _dedup_adjacent_messages(cur)
    cur.extend(inc)
    return _dedup_adjacent_messages(cur)

def _append_assistant_if_new(messages, content):
    before = len(messages)
    _append_assistant_message(messages, content)
    if len(messages) <= before:
        return
    if len(messages) < 2:
        return
    if _message_fingerprint(messages[-1]) == _message_fingerprint(messages[-2]):
        messages.pop()

def _build_dialog_from_all_turns(entries, session_id):
    turns = _collect_turn_entries(entries, session_id)
    if not turns:
        return {}
    non_probe_turns = [turn for turn in turns if not turn[2]]
    chosen_turns = non_probe_turns or turns
    dialog_messages = []
    tools = []
    turn_meta = {}
    for (messages, assistant_content, _is_probe, req_tools, req_meta) in chosen_turns:
        dialog_messages = _merge_turn_messages(dialog_messages, messages)
        _append_assistant_if_new(dialog_messages, assistant_content)
        if req_tools:
            tools = [dict(tool) for tool in req_tools]
        if req_meta:
            turn_meta = dict(req_meta)
    if not dialog_messages:
        return {}
    dialog = {'messages': dialog_messages}
    if tools:
        dialog['tools'] = [dict(tool) for tool in tools]
    if turn_meta:
        dialog['meta'] = dict(turn_meta)
    return dialog

def _build_dialog_snapshots(entries, session_id):
    turns = _collect_turn_entries(entries, session_id)
    if not turns:
        return []
    non_probe_turns = [turn for turn in turns if not turn[2]]
    chosen_turns = non_probe_turns or turns
    snapshots = []
    for (messages, assistant_content, _is_probe, req_tools, req_meta) in chosen_turns:
        snapshot_messages = [dict(msg) for msg in messages if isinstance(msg, dict)]
        _append_assistant_if_new(snapshot_messages, assistant_content)
        if not snapshot_messages:
            continue
        snapshot = {'messages': snapshot_messages}
        if req_tools:
            snapshot['tools'] = [dict(tool) for tool in req_tools if isinstance(tool, dict)]
        if req_meta:
            snapshot['meta'] = dict(req_meta)
        snapshots.append(snapshot)
    return snapshots

def _is_messages_prefix(shorter, longer):
    if len(shorter) > len(longer):
        return False
    for (idx, msg) in enumerate(shorter):
        if _message_fingerprint(msg) != _message_fingerprint(longer[idx]):
            return False
    return True

def _longest_common_prefix_len(left, right):
    max_len = min(len(left), len(right))
    for idx in range(max_len):
        if _message_fingerprint(left[idx]) != _message_fingerprint(right[idx]):
            return idx
    return max_len

def _assistant_content_signal(msg):
    content = msg.get('content')
    has_structured = int(isinstance(content, list))
    has_thinking_block = 0
    has_tool_block = 0
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get('type') or '').strip().lower()
            if item_type in {'thinking', 'reasoning', 'analysis'}:
                has_thinking_block = 1
            if item_type in {'tool_use', 'tool_call', 'function_call'}:
                has_tool_block = 1
    text_len = len(_content_to_text(content, include_reasoning=False).strip())
    return (has_tool_block, has_thinking_block, has_structured, text_len)

def _assistant_payload_score(msg):
    tool_calls = _normalize_tool_calls(msg.get('tool_calls'))
    tool_count = len(tool_calls)
    if msg.get('function_call') is not None:
        tool_count = max(tool_count, 1)
    (has_reasoning, reasoning_len) = _assistant_reasoning_score_from_message(msg)
    content_signal = _assistant_content_signal(msg)
    return (1 if tool_count > 0 else 0, tool_count, has_reasoning, reasoning_len) + content_signal

def _merge_adjacent_assistant_messages(previous, current):
    prev = _sanitize_assistant_message(previous)
    cur = _sanitize_assistant_message(current)
    merged = dict(cur) if _assistant_payload_score(cur) > _assistant_payload_score(prev) else dict(prev)
    prev_content = prev.get('content')
    cur_content = cur.get('content')
    merged_content = merged.get('content')
    merged_content_signal = _assistant_content_signal(merged)
    prev_content_signal = _assistant_content_signal(prev)
    cur_content_signal = _assistant_content_signal(cur)
    if prev_content_signal > merged_content_signal:
        merged_content = prev_content
        merged_content_signal = prev_content_signal
    if cur_content_signal > merged_content_signal:
        merged_content = cur_content
    prev_text_key = _assistant_text_key_from_message(prev)
    cur_text_key = _assistant_text_key_from_message(cur)
    prev_reason = _assistant_reasoning_text_from_message(prev)
    cur_reason = _assistant_reasoning_text_from_message(cur)
    richer_reason = prev_reason if len(prev_reason) >= len(cur_reason) else cur_reason
    if prev_text_key and prev_text_key == cur_text_key:
        merged_content = _merge_assistant_content_same_text(prev_content, cur_content, richer_reason)
    merged['content'] = merged_content
    merged_tool_calls = _normalize_tool_calls(merged.get('tool_calls'))
    if not merged_tool_calls:
        merged_tool_calls = _normalize_tool_calls(cur.get('tool_calls'))
    if not merged_tool_calls:
        merged_tool_calls = _normalize_tool_calls(prev.get('tool_calls'))
    if merged_tool_calls:
        merged['tool_calls'] = merged_tool_calls
    elif 'tool_calls' in merged:
        merged['tool_calls'] = []
    if merged.get('function_call') is None:
        if cur.get('function_call') is not None:
            merged['function_call'] = cur.get('function_call')
        elif prev.get('function_call') is not None:
            merged['function_call'] = prev.get('function_call')
    merged_reason = _assistant_reasoning_text_from_message(merged)
    if richer_reason and len(richer_reason) > len(merged_reason):
        merged['reasoning_content'] = richer_reason
    for key in ('reasoning_content', 'reasoning', 'reasoningContent', 'thinking', 'analysis'):
        if key in merged:
            continue
        if key in cur and cur.get(key) not in (None, ''):
            merged[key] = cur.get(key)
            continue
        if key in prev and prev.get(key) not in (None, ''):
            merged[key] = prev.get(key)
    return _sanitize_assistant_message(merged)

def _dedup_adjacent_messages(messages):
    deduped = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if deduped:
            prev = deduped[-1]
            prev_role = str(prev.get('role') or '').strip().lower()
            cur_role = str(msg.get('role') or '').strip().lower()
            if prev_role == 'assistant' and cur_role == 'assistant':
                if _assistant_messages_should_merge(prev, msg):
                    deduped[-1] = _merge_adjacent_assistant_messages(prev, msg)
                    continue
        if deduped and _message_fingerprint(deduped[-1]) == _message_fingerprint(msg):
            continue
        deduped.append(dict(msg))
    return deduped

def _normalize_text_key(text):
    return ' '.join((text or '').strip().lower().split())

def _normalize_tool_arguments_signature(arguments):
    if arguments is None:
        return ''
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return ''
        parsed = _load_json_maybe(raw)
        if isinstance(parsed, (dict, list)):
            return _compact_json(parsed)
        return _normalize_text_key(raw)
    return _compact_json(arguments)

def _assistant_tool_signature_from_message(msg):
    tool_calls = _normalize_tool_calls(msg.get('tool_calls'))
    if not tool_calls:
        tool_calls = _extract_tool_calls_from_anthropic_content(msg.get('content'))
    if not tool_calls:
        return ''
    parts = []
    for call in tool_calls:
        function = call.get('function')
        if not isinstance(function, dict):
            continue
        name = str(function.get('name') or '').strip().lower()
        if not name:
            continue
        arguments_sig = _normalize_tool_arguments_signature(function.get('arguments'))
        parts.append(''.join([str(name), '(', str(arguments_sig), ')']))
    return '|'.join(parts)

def _assistant_function_call_signature_from_message(msg):
    raw = msg.get('function_call')
    if not isinstance(raw, dict):
        return ''
    name = str(raw.get('name') or '').strip().lower()
    if not name:
        return ''
    arguments_sig = _normalize_tool_arguments_signature(raw.get('arguments'))
    return ''.join([str(name), '(', str(arguments_sig), ')'])

def _assistant_messages_should_merge(previous, current):
    if _message_fingerprint(previous) == _message_fingerprint(current):
        return True
    prev_tool_sig = _assistant_tool_signature_from_message(previous)
    cur_tool_sig = _assistant_tool_signature_from_message(current)
    if prev_tool_sig and cur_tool_sig and (prev_tool_sig == cur_tool_sig):
        return True
    prev_text_key = _assistant_text_key_from_message(previous)
    cur_text_key = _assistant_text_key_from_message(current)
    if prev_text_key and cur_text_key and (prev_text_key == cur_text_key):
        if prev_tool_sig and cur_tool_sig and (prev_tool_sig != cur_tool_sig):
            return False
        return True
    prev_fn_sig = _assistant_function_call_signature_from_message(previous)
    cur_fn_sig = _assistant_function_call_signature_from_message(current)
    if prev_fn_sig and cur_fn_sig and (prev_fn_sig == cur_fn_sig):
        return True
    return False

def _assistant_text_key_from_message(msg):
    text = _content_to_text(msg.get('content'), include_reasoning=False).strip()
    if not text:
        return ''
    return _normalize_text_key(text)

def _assistant_backfill_key_from_message(msg):
    tool_sig = _assistant_tool_signature_from_message(msg)
    if tool_sig:
        return ''.join(['tool:', str(tool_sig)])
    text_key = _assistant_text_key_from_message(msg)
    if text_key:
        return ''.join(['text:', str(text_key)])
    fn_sig = _assistant_function_call_signature_from_message(msg)
    if fn_sig:
        return ''.join(['function:', str(fn_sig)])
    return ''.join(['fp:', str(_message_fingerprint(msg))])

def _assistant_reasoning_text_from_message(msg):
    parts = []
    top_reasoning = _extract_assistant_reasoning(msg).strip()
    if top_reasoning:
        parts.append(top_reasoning)
    content = msg.get('content')
    if isinstance(content, list):
        for block in content:
            (_text, reasoning) = _extract_text_reasoning_from_anthropic_block(block)
            reasoning = reasoning.strip()
            if reasoning:
                parts.append(reasoning)
    elif isinstance(content, dict):
        (_text, reasoning) = _extract_text_reasoning_from_anthropic_block(content)
        reasoning = reasoning.strip()
        if reasoning:
            parts.append(reasoning)
    if not parts:
        return ''
    deduped = []
    seen = set()
    for item in parts:
        key = _normalize_text_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    return '\n\n'.join([p for p in deduped if p]).strip()

def _assistant_tool_block_count(content):
    if not isinstance(content, list):
        return 0
    count = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get('type') or '').strip().lower()
        if item_type in {'tool_use', 'tool_call', 'function_call'}:
            count += 1
    return count

def _assistant_upsert_thinking_block(content, reasoning_text):
    reasoning_text = (reasoning_text or '').strip()
    if not reasoning_text:
        return content
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return [{'type': 'thinking', 'thinking': reasoning_text}]
        return [{'type': 'thinking', 'thinking': reasoning_text}, {'type': 'text', 'text': text}]
    if not isinstance(content, list):
        text = _content_to_text(content, include_reasoning=False).strip()
        if not text:
            return [{'type': 'thinking', 'thinking': reasoning_text}]
        return [{'type': 'thinking', 'thinking': reasoning_text}, {'type': 'text', 'text': text}]
    merged = [dict(item) if isinstance(item, dict) else item for item in content]
    for (idx, item) in enumerate(merged):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get('type') or '').strip().lower()
        if item_type not in {'thinking', 'reasoning', 'analysis', 'redacted_thinking'}:
            continue
        existing = _content_to_text(item, include_reasoning=True).strip()
        if len(existing) >= len(reasoning_text):
            return merged
        updated = dict(item)
        if item_type == 'analysis':
            updated['analysis'] = reasoning_text
        elif item_type == 'reasoning':
            updated['reasoning'] = reasoning_text
        elif item_type == 'redacted_thinking':
            updated['redacted_thinking'] = reasoning_text
        else:
            updated['type'] = 'thinking'
            updated['thinking'] = reasoning_text
        merged[idx] = updated
        return merged
    return [{'type': 'thinking', 'thinking': reasoning_text}] + merged

def _merge_assistant_content_same_text(previous, current, reasoning_text):
    prev_tools = _assistant_tool_block_count(previous)
    cur_tools = _assistant_tool_block_count(current)
    if prev_tools > cur_tools:
        base = previous
    elif cur_tools > prev_tools:
        base = current
    else:
        prev_signal = _assistant_content_signal({'content': previous})
        cur_signal = _assistant_content_signal({'content': current})
        base = previous if prev_signal >= cur_signal else current
    return _assistant_upsert_thinking_block(base, reasoning_text)

def _assistant_reasoning_score_from_message(msg):
    reasoning = _assistant_reasoning_text_from_message(msg).strip()
    return (1 if reasoning else 0, len(reasoning))

def _merge_assistant_message_with_reasoning_backfill(previous, current):
    prev_key = _assistant_backfill_key_from_message(previous)
    cur_key = _assistant_backfill_key_from_message(current)
    if not prev_key or prev_key != cur_key:
        return dict(current)
    return _merge_adjacent_assistant_messages(previous, current)

def _overlay_dialog_snapshot_keep_richer_assistant_reasoning(previous, current):
    prev_messages = previous.get('messages')
    cur_messages = current.get('messages')
    if not isinstance(prev_messages, list) or not isinstance(cur_messages, list):
        return dict(current)
    merged = dict(current)
    merged_messages = [dict(msg) for msg in cur_messages if isinstance(msg, dict)]
    prev_norm = [dict(msg) for msg in prev_messages if isinstance(msg, dict)]
    overlap = min(len(prev_norm), len(merged_messages))
    for idx in range(overlap):
        prev_msg = prev_norm[idx]
        cur_msg = merged_messages[idx]
        prev_role = str(prev_msg.get('role') or '').strip().lower()
        cur_role = str(cur_msg.get('role') or '').strip().lower()
        if prev_role != 'assistant' or cur_role != 'assistant':
            continue
        merged_messages[idx] = _merge_assistant_message_with_reasoning_backfill(prev_msg, cur_msg)
    merged['messages'] = merged_messages
    return merged

def _is_dialog_prefix_related(left, right):
    left_messages = left.get('messages')
    right_messages = right.get('messages')
    if not isinstance(left_messages, list) or not isinstance(right_messages, list):
        return False
    left_norm = [dict(msg) for msg in left_messages if isinstance(msg, dict)]
    right_norm = [dict(msg) for msg in right_messages if isinstance(msg, dict)]
    return _is_messages_prefix(left_norm, right_norm) or _is_messages_prefix(right_norm, left_norm)

def _dialog_starts_with_system(snapshot):
    messages = snapshot.get('messages')
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict):
        return False
    role = str(first.get('role') or '').strip().lower()
    return role in {'system', 'developer'}

def _merge_dialog_snapshots(snapshots):
    if not snapshots:
        return []
    normalized = []
    for snapshot in snapshots:
        messages = snapshot.get('messages')
        if not isinstance(messages, list):
            continue
        normalized_messages = [dict(msg) for msg in messages if isinstance(msg, dict)]
        if not normalized_messages:
            continue
        normalized_snapshot = dict(snapshot)
        normalized_snapshot['messages'] = normalized_messages
        normalized.append(normalized_snapshot)
    if not normalized:
        return []
    merged = []
    best = dict(normalized[0])
    for snapshot in normalized[1:]:
        cur = dict(snapshot)
        if _is_dialog_prefix_related(best, cur):
            best_len = len(best.get('messages') or [])
            cur_len = len(cur.get('messages') or [])
            if cur_len >= best_len:
                best = _overlay_dialog_snapshot_keep_richer_assistant_reasoning(best, cur)
            continue
        best_messages = best.get('messages')
        cur_messages = cur.get('messages')
        if not isinstance(best_messages, list) or not isinstance(cur_messages, list):
            if _dialog_starts_with_system(cur):
                merged.append(best)
                best = cur
            continue
        lcp = _longest_common_prefix_len([dict(msg) for msg in best_messages if isinstance(msg, dict)], [dict(msg) for msg in cur_messages if isinstance(msg, dict)])
        if lcp >= 2:
            best = dict(best)
            best['messages'] = _dedup_adjacent_messages([dict(msg) for msg in best_messages if isinstance(msg, dict)] + [dict(msg) for msg in cur_messages[lcp:] if isinstance(msg, dict)])
            if cur.get('tools'):
                best['tools'] = [dict(tool) for tool in cur.get('tools') if isinstance(tool, dict)]
            if cur.get('meta'):
                best['meta'] = dict(cur['meta'])
            continue
        if _dialog_starts_with_system(cur):
            merged.append(best)
            best = cur
            continue
        best = dict(best)
        best['messages'] = _merge_turn_messages([dict(msg) for msg in best_messages if isinstance(msg, dict)], [dict(msg) for msg in cur_messages if isinstance(msg, dict)])
        if cur.get('tools'):
            best['tools'] = [dict(tool) for tool in cur.get('tools') if isinstance(tool, dict)]
        if cur.get('meta'):
            best['meta'] = dict(cur['meta'])
    merged.append(best)
    return merged

def _has_any_session_headers(entries):
    for entry in entries:
        headers = entry.get('header') or {}
        if _header_value(headers, 'x-session-id').strip():
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--session-id', default='', help='Optional session_id (filters X-Session-ID in fastapi logs)')
    parser.add_argument('--log-dir', default='/tmp/fastapi_logs', help='FastAPI log dir (default: /tmp/fastapi_logs)')
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    entries = list(_iter_log_entries(log_dir))
    session_id = args.session_id.strip() or None
    snapshots = _build_dialog_snapshots(entries, session_id)
    if not snapshots and session_id is not None and (not _has_any_session_headers(entries)):
        snapshots = _build_dialog_snapshots(entries, None)
    output_payload = _merge_dialog_snapshots(snapshots)
    Path(args.output).write_text(json.dumps(output_payload, ensure_ascii=False), encoding='utf-8')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
