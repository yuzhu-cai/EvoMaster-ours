from __future__ import annotations

import os
import re
from functools import cached_property

import openai
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.utils import CONTEXT_WINDOW_LENGTHS, RetryConfig


DEFAULT_VENDOR_CONTEXT_WINDOW = 400_000
DEFAULT_VENDOR_MODEL = "Vendor2/GPT-5.4"


def register_vendor_context_windows() -> None:
    """Teach PaperBench's judge about OpenAI-compatible vendor model aliases."""

    n_ctx = int(
        os.environ.get(
            "OPENHANDS4PAPERBENCH_VENDOR_CONTEXT_WINDOW",
            str(DEFAULT_VENDOR_CONTEXT_WINDOW),
        )
    )
    for model in (
        DEFAULT_VENDOR_MODEL,
        f"openai/{DEFAULT_VENDOR_MODEL}",
    ):
        CONTEXT_WINDOW_LENGTHS.setdefault(model, n_ctx)


def _structured_judge_model() -> str:
    model = os.environ.get("OPENHANDS4PAPERBENCH_STRUCTURED_JUDGE_MODEL")
    model = model or os.environ.get("GPT_CHAT_MODEL")
    model = model or DEFAULT_VENDOR_MODEL
    return model.removeprefix("openai/")


def patch_simple_judge_structured_completers() -> None:
    """Route SimpleJudge's response-parser calls away from the hard-coded gpt-4o."""

    from paperbench.judge.simple import SimpleJudge

    if getattr(SimpleJudge, "_openhands4paperbench_vendor_patch", False):
        return

    original = SimpleJudge._init_structured_completer

    def _init_structured_completer(self, config, response_format):
        if config is None:
            retry_stop_after = float(
                os.environ.get("OPENHANDS4PAPERBENCH_STRUCTURED_JUDGE_RETRY_STOP_AFTER", "600")
            )
            config = VendorOpenAICompletionsTurnCompleter.Config(
                model=_structured_judge_model(),
                reasoning_effort=os.environ.get(
                    "OPENHANDS4PAPERBENCH_STRUCTURED_JUDGE_REASONING_EFFORT",
                    "medium",
                ),
                response_format=response_format,
                retry_config=RetryConfig(wait_min=1, wait_max=30, stop_after=retry_stop_after),
            )
        return original(self, config, response_format)

    SimpleJudge._init_structured_completer = _init_structured_completer
    SimpleJudge._openhands4paperbench_vendor_patch = True


def patch_completer_client_timeout() -> None:
    timeout_raw = os.environ.get("OPENHANDS4PAPERBENCH_JUDGE_TIMEOUT_SECONDS")
    if not timeout_raw:
        return
    if getattr(OpenAICompletionsTurnCompleter, "_openhands4paperbench_timeout_patch", None) == timeout_raw:
        return

    timeout = float(timeout_raw)

    @cached_property
    def _client(self) -> openai.AsyncClient:
        return openai.AsyncClient(timeout=timeout)

    _client.__set_name__(OpenAICompletionsTurnCompleter, "_client")
    OpenAICompletionsTurnCompleter._client = _client
    OpenAICompletionsTurnCompleter._openhands4paperbench_timeout_patch = timeout_raw


def patch_token_usage_none_outputs() -> None:
    """Handle vendor usage payloads that omit completion token counts.

    Some OpenAI-compatible endpoints return a usage object with
    ``completion_tokens=None`` for structured-parser calls. PaperBench token
    accounting is reporting-only, so coerce missing counts to zero instead of
    letting the judge fail after producing a score.
    """

    from paperbench.judge.token_usage import TokenUsage

    if getattr(TokenUsage, "_openhands4paperbench_none_usage_patch", False):
        return

    original_add_usage = TokenUsage.add_usage

    def add_usage(self, model: str, input_tokens: int | None, output_tokens: int | None) -> None:
        original_add_usage(self, model, int(input_tokens or 0), int(output_tokens or 0))

    TokenUsage.add_usage = add_usage
    TokenUsage._openhands4paperbench_none_usage_patch = True


def patch_simple_judge_fallback_parser() -> None:
    """Add a local parser fallback for judge responses.

    The vendor endpoint can return a successful chat response for the structured
    parser call while leaving the SDK's parsed/content fields empty. The original
    judge treats that as an invalid leaf. The primary judge response is already a
    markdown answer with a ``# Score`` section, so parse that directly when the
    structured parser fails.
    """

    from paperbench.judge.simple import (
        ParsedJudgeResponseFloat,
        ParsedJudgeResponseInt,
        SimpleJudge,
    )

    if getattr(SimpleJudge, "_openhands4paperbench_fallback_parser_patch", False):
        return

    original_parse = SimpleJudge._parse_model_response

    def _extract_score(response: str) -> float | None:
        score_section = re.search(
            r"(?is)(?:^|\n)\s*#{0,6}\s*score\b(?P<body>.*?)(?:\n\s*#{1,6}\s+\S|\Z)",
            response,
        )
        candidates = score_section.group("body") if score_section else response[-1200:]
        match = re.search(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])", candidates)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    async def _parse_model_response(self, response: str | None, continuous: bool = False):
        try:
            return await original_parse(self, response, continuous=continuous)
        except Exception as exc:
            ParsedJudgeResponse = ParsedJudgeResponseFloat if continuous else ParsedJudgeResponseInt
            if response is None:
                parsed = ParsedJudgeResponse(
                    valid_score=True,
                    score=0.0 if continuous else 0,
                    explanation=f"No judge response received; defaulting score to 0. Parser error: {exc}",
                )
                return parsed, None

            score = _extract_score(response)
            if score is None:
                parsed = ParsedJudgeResponse(
                    valid_score=True,
                    score=0.0 if continuous else 0,
                    explanation=(
                        "Structured parser failed and fallback parser found no explicit score; "
                        f"defaulting score to 0. Parser error: {exc}. Response excerpt: "
                        f"{response[:500]}"
                    ),
                )
                return parsed, None

            score = max(0.0, min(1.0, score))
            if continuous:
                parsed = ParsedJudgeResponse(
                    valid_score=True,
                    score=score,
                    explanation=response[:4000],
                )
            else:
                parsed = ParsedJudgeResponse(
                    valid_score=True,
                    score=1 if score >= 0.5 else 0,
                    explanation=response[:4000],
                )
            return parsed, None

    SimpleJudge._parse_model_response = _parse_model_response
    SimpleJudge._openhands4paperbench_fallback_parser_patch = True


def patch_simple_judge_file_selection_fallback() -> None:
    """Avoid invalid leaves when the auxiliary file-ranker call is empty."""

    from paperbench.judge.simple import SimpleJudge
    from paperbench.judge.utils import format_file, read_file_content

    if getattr(SimpleJudge, "_openhands4paperbench_file_selection_patch", False):
        return

    original_prepare = SimpleJudge._prepare_relevant_files

    async def _prepare_relevant_files(self, task, max_files: int | None = 10) -> str:
        try:
            return await original_prepare(self, task, max_files=max_files)
        except Exception as exc:
            task_category = task.task_category or "Subtree"
            try:
                leaf_logger = self.get_logger(task)
                leaf_logger.info(f"File selection failed; using deterministic fallback: {exc}")
            except Exception:
                pass

            try:
                whitelisted_files = await self._get_whitelisted_files(
                    task_category, max_file_depth=self.max_file_depth
                )
            except Exception as list_exc:
                return (
                    "File selection failed and no deterministic file list could be built. "
                    f"file_selection_error={exc}; list_error={list_exc}"
                )

            max_tokens = self.avail_context_lens[task_category] - 2000
            selected_tokens: list[int] = []
            total_tokens = 0
            num_files = 0
            for path in whitelisted_files:
                if max_files and num_files >= max_files:
                    break
                try:
                    content = await read_file_content(path, self.computer)
                    file_content = format_file(path.relative_to(self.submission_dir), content)
                    content_tokens = self.token_encoder.encode(
                        file_content + "\n\n", disallowed_special=()
                    )
                    if total_tokens + len(content_tokens) > max_tokens:
                        selected_tokens.extend(content_tokens[: max(0, max_tokens - total_tokens)])
                        break
                    selected_tokens.extend(content_tokens)
                    total_tokens += len(content_tokens)
                    num_files += 1
                except Exception:
                    continue

            if not selected_tokens:
                return (
                    "File selection failed and deterministic fallback found no readable files. "
                    f"file_selection_error={exc}"
                )
            return self.token_encoder.decode(selected_tokens).rsplit("\n", 1)[0]

    SimpleJudge._prepare_relevant_files = _prepare_relevant_files
    SimpleJudge._openhands4paperbench_file_selection_patch = True


class VendorOpenAICompletionsTurnCompleter(OpenAICompletionsTurnCompleter):
    class Config(OpenAICompletionsTurnCompleter.Config):
        def build(self) -> OpenAICompletionsTurnCompleter:
            register_vendor_context_windows()
            patch_completer_client_timeout()
            patch_token_usage_none_outputs()
            patch_simple_judge_fallback_parser()
            patch_simple_judge_file_selection_fallback()
            patch_simple_judge_structured_completers()
            return super().build()
