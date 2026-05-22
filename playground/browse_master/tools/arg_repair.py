"""Utilities to recover malformed tool-call arguments from model output.

The browse benchmark often asks the model to put quoted search phrases inside a
JSON string. Some models emit tool arguments like:

    {"query": ["\"phrase\"" other words]}

which is not valid JSON, even though the intended search string is recoverable.
These helpers salvage the common failure modes for browse tools so a task does
not waste turns on a parameter-format error.
"""

from __future__ import annotations

import re


def recover_list_of_strings(
    args_json: str,
    field_name: str,
    *,
    strip_wrapping_quotes: bool,
) -> list[str] | None:
    """Recover a list[str] field from malformed JSON-like text."""
    field_pos = _find_field_position(args_json, field_name)
    if field_pos < 0:
        return None

    array_start = args_json.find("[", field_pos)
    if array_start < 0:
        return None

    array_end = _find_matching_bracket(args_json, array_start)
    if array_end < 0:
        array_end = args_json.rfind("]")
        if array_end < array_start:
            return None

    inner = args_json[array_start + 1 : array_end]
    parts = _split_top_level(inner)
    recovered = [
        value
        for value in (
            _normalize_fragment(part, strip_wrapping_quotes=strip_wrapping_quotes)
            for part in parts
        )
        if value
    ]
    return recovered or None


def recover_string_field(args_json: str, field_name: str) -> str | None:
    """Recover a string field from malformed JSON-like text."""
    field_pos = _find_field_position(args_json, field_name)
    if field_pos < 0:
        return None

    colon_pos = args_json.find(":", field_pos)
    if colon_pos < 0:
        return None

    value_start = colon_pos + 1
    while value_start < len(args_json) and args_json[value_start].isspace():
        value_start += 1
    if value_start >= len(args_json):
        return None

    tail = args_json[value_start:]
    raw_value: str

    if tail[0] in {'"', "'"}:
        quote = tail[0]
        closing = _find_closing_quote(tail, quote)
        if closing >= 0:
            raw_value = tail[: closing + 1]
        else:
            raw_value = tail.rstrip().rstrip("}")
    else:
        next_field = re.search(r',\s*"[^"]+"\s*:', tail)
        if next_field:
            raw_value = tail[: next_field.start()]
        else:
            raw_value = tail.rstrip().rstrip("}")

    return _normalize_fragment(raw_value, strip_wrapping_quotes=True) or None


def _find_field_position(args_json: str, field_name: str) -> int:
    pattern = rf'"{re.escape(field_name)}"\s*:'
    match = re.search(pattern, args_json)
    return match.start() if match else -1


def _find_matching_bracket(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _split_top_level(inner: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    bracket_depth = 0
    brace_depth = 0

    for char in inner:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)

        if char == "," and bracket_depth == 0 and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_closing_quote(text: str, quote: str) -> int:
    escaped = False
    for index in range(1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return index
    return -1


def _normalize_fragment(fragment: str, *, strip_wrapping_quotes: bool) -> str:
    value = fragment.strip()
    if not value:
        return ""

    # Fix the most common model artifact: an accidental doubled leading/trailing
    # quote caused by mixing search-phrase quotes with JSON string quotes.
    if value.startswith('""'):
        value = value[1:]
    if value.endswith('""'):
        value = value[:-1]

    value = value.replace('\\"', '"').replace("\\'", "'")
    value = re.sub(r"\s+", " ", value).strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        if value.count(quote) % 2 == 1:
            value = value[:-1].rstrip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        inner = value[1:-1]
        if quote not in inner:
            if strip_wrapping_quotes or quote == "'":
                value = inner.strip()

    return value.strip()
