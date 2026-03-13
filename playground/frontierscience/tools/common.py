"""Shared helpers for FrontierScience local tools."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any


MAX_TOOL_TEXT = 20000
_MODULE_CACHE: dict[str, Any] = {}


def ensure_text(result: Any, max_len: int = MAX_TOOL_TEXT) -> str:
    text = str(result) if result is not None else ""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + "\n\n... [truncated due to length] ...\n\n" + text[-half:]


def get_external_script_path(env_key: str, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None:
        return explicit_path.resolve()
    override = os.getenv(env_key, "").strip()
    if not override:
        return None
    return Path(override).expanduser().resolve()


def get_serper_api_key() -> str:
    """Load Serper API key from SERPER_API_KEY."""
    return os.getenv("SERPER_API_KEY", "").strip()


def _load_module(script_path: Path) -> Any:
    resolved = str(script_path.resolve())
    if resolved in _MODULE_CACHE:
        return _MODULE_CACHE[resolved]

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    module_name = f"frontier_ext_{script_path.stem}_{abs(hash(resolved))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec for: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _MODULE_CACHE[resolved] = module
    return module


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: dict[str, Any] = {}

    def _runner() -> None:
        holder["value"] = asyncio.run(coro)

    import threading

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    return holder.get("value")


def call_external_function(script_path: Path, fn_name: str, *args: Any, **kwargs: Any) -> Any:
    module = _load_module(script_path)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"Function '{fn_name}' not found in {script_path.name}")

    if inspect.iscoroutinefunction(fn):
        return _run_async(fn(*args, **kwargs))

    value = fn(*args, **kwargs)
    if inspect.isawaitable(value):
        return _run_async(value)
    return value


def extract_text_from_html(text: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
