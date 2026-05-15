"""Runtime compatibility hooks for local PaperBench experiments.

This module is auto-imported by Python when playground/codex4paperbench is on
PYTHONPATH. Keep behavior opt-in so normal PaperBench runs are unchanged.
"""

from __future__ import annotations

import asyncio
import os


def _patch_openai_completer_client() -> None:
    try:
        import openai
        from preparedness_turn_completer.oai_completions_turn_completer import (
            OpenAICompletionsTurnCompleter,
        )
        from preparedness_turn_completer.utils import RetryConfig
    except Exception:
        return

    request_timeout = float(os.getenv("CODEX4PAPERBENCH_OPENAI_TIMEOUT", "90"))
    max_retries = int(os.getenv("CODEX4PAPERBENCH_OPENAI_CLIENT_MAX_RETRIES", "0"))
    retry_wait_min = float(os.getenv("CODEX4PAPERBENCH_JUDGE_RETRY_WAIT_MIN", "1"))
    retry_wait_max = float(os.getenv("CODEX4PAPERBENCH_JUDGE_RETRY_WAIT_MAX", "10"))
    retry_stop_after = float(os.getenv("CODEX4PAPERBENCH_JUDGE_RETRY_STOP_AFTER", "240"))

    def patched_client(self):
        cached = self.__dict__.get("_codex4paperbench_openai_client")
        if cached is not None:
            return cached

        kwargs = {
            "timeout": request_timeout,
            "max_retries": max_retries,
        }
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("GPT_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GRADER_OPENAI_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        client = openai.AsyncClient(**kwargs)
        self.__dict__["_codex4paperbench_openai_client"] = client
        return client

    OpenAICompletionsTurnCompleter._client = property(patched_client)

    original_init = OpenAICompletionsTurnCompleter.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.retry_config = RetryConfig(
            wait_min=retry_wait_min,
            wait_max=retry_wait_max,
            stop_after=retry_stop_after,
        )

    OpenAICompletionsTurnCompleter.__init__ = patched_init


def _patch_simple_judge_structured_model() -> None:
    model = os.getenv("CODEX4PAPERBENCH_JUDGE_STRUCTURED_MODEL") or os.getenv("GPT_CHAT_MODEL")
    if not model:
        return

    try:
        from preparedness_turn_completer import utils as completer_utils
        from preparedness_turn_completer.oai_completions_turn_completer import (
            OpenAICompletionsTurnCompleter,
        )
        from paperbench.judge.simple import SimpleJudge
    except Exception:
        return

    context_window = int(os.getenv("CODEX4PAPERBENCH_JUDGE_CONTEXT_WINDOW", "128000"))
    completer_utils.CONTEXT_WINDOW_LENGTHS.setdefault(model, context_window)

    original = SimpleJudge._init_structured_completer

    def patched_init_structured_completer(self, config, response_format):
        if config is None:
            config = OpenAICompletionsTurnCompleter.Config(
                model=model,
                response_format=response_format,
            )
        return original(self, config, response_format)

    SimpleJudge._init_structured_completer = patched_init_structured_completer

    leaf_concurrency = os.getenv("CODEX4PAPERBENCH_JUDGE_LEAF_CONCURRENCY")
    if leaf_concurrency:
        original_init = SimpleJudge.__init__
        leaf_limit = int(leaf_concurrency)

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.leaf_semaphore = asyncio.Semaphore(leaf_limit)

        SimpleJudge.__init__ = patched_init


_patch_openai_completer_client()
_patch_simple_judge_structured_model()
