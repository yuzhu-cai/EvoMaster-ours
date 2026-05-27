#!/usr/bin/env python3
"""Grade one PaperBench Code-Dev submission with a streaming Responses API judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import Field

from preparedness_turn_completer import utils as turn_completer_utils
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, get_model_context_window_length

from paperbench.grade import JudgeOutput
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.utils import get_timestamp

from playground.paperbench_codedev_agent.scripts.grade_submission import (
    DEFAULT_ENV,
    _format_failure_feedback,
    _run_judge_controlled,
)


class ResponsesTurnCompleter(TurnCompleter):
    """Minimal TurnCompleter backed by OpenAI-compatible streaming Responses API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str | None = "medium",
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = 4096,
        timeout: float = 240.0,
        retry_config: RetryConfig | None = None,
        context_window: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            self.encoding_name = "o200k_base"
        self.n_ctx = context_window or get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        model: str
        base_url: str
        api_key_env: str = "OPENAI_API_KEY"
        reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = "medium"
        reasoning_summary: str | None = None
        max_output_tokens: int | None = 4096
        timeout: float = 240.0
        retry_config: RetryConfig = Field(default_factory=RetryConfig)
        context_window: int | None = None

        def build(self) -> "ResponsesTurnCompleter":
            return ResponsesTurnCompleter(
                model=self.model,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                max_output_tokens=self.max_output_tokens,
                timeout=self.timeout,
                retry_config=self.retry_config,
                context_window=self.context_window,
            )

    class Completion(TurnCompleter.Completion):
        usage: Any | None = None

    @cached_property
    def _client(self) -> AsyncOpenAI:
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not configured")
        return AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def completion(self, conversation: TurnCompleter.RuntimeConversation, **params: Any):
        raise NotImplementedError("Use async_completion")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Any,
    ) -> "ResponsesTurnCompleter.Completion":
        del params
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._convert_messages(conversation),
            "stream": True,
        }
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        reasoning: dict[str, Any] = {}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning

        last_usage = None
        async for attempt in self.retry_config.build():
            with attempt:
                text_parts: list[str] = []
                stream = await self._client.responses.create(**payload)
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        text_parts.append(getattr(event, "delta", "") or "")
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        last_usage = getattr(response, "usage", None)
                        if not text_parts:
                            recovered = self._extract_output_text(response)
                            if recovered:
                                text_parts.append(recovered)
                    elif event_type == "response.failed":
                        response = getattr(event, "response", None)
                        err = getattr(response, "error", None)
                        raise RuntimeError(f"Responses API failed: {err}")
                text = "".join(text_parts).strip()
        msg = ChatCompletionMessage(role="assistant", content=text)
        return ResponsesTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[msg],
            usage=last_usage,
        )

    @staticmethod
    def _convert_messages(conversation: TurnCompleter.RuntimeConversation) -> list[dict[str, str]]:
        converted: list[dict[str, str]] = []
        for message in conversation:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)
            converted.append({"role": role, "content": "" if content is None else str(content)})
        return converted

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        if response is None:
            return ""
        output = getattr(response, "output", None) or []
        parts: list[str] = []
        for item in output:
            for content in getattr(item, "content", None) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://139.180.136.5:3000/openai")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--reasoning-summary", default="")
    parser.add_argument("--context-window", type=int, default=272000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--openai-timeout", type=float, default=240)
    parser.add_argument("--max-depth", type=int, default=999)
    parser.add_argument("--leaf-concurrency", type=int, default=4)
    parser.add_argument("--leaf-timeout", type=float, default=3600)
    parser.add_argument("--retry-stop-after", type=float, default=1800)
    parser.add_argument("--print-failures", type=int, default=0)
    return parser.parse_args()


async def grade(args: argparse.Namespace) -> None:
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    turn_completer_utils.CONTEXT_WINDOW_LENGTHS.setdefault(args.model, args.context_window)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completer_config = ResponsesTurnCompleter.Config(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary or None,
        max_output_tokens=args.max_output_tokens,
        timeout=args.openai_timeout,
        retry_config=RetryConfig(wait_min=2, wait_max=60, stop_after=args.retry_stop_after),
        context_window=args.context_window,
    )
    graded_task_tree = await _run_judge_controlled(
        submission_path=args.submission,
        paper_id=args.paper_id,
        completer_config=completer_config,
        out_dir=args.out_dir,
        max_depth=args.max_depth,
        leaf_concurrency=args.leaf_concurrency,
        leaf_timeout=args.leaf_timeout,
        local_parse=True,
    )
    token_usage = get_total_token_usage(graded_task_tree)
    judge_output = JudgeOutput(
        judge_type="simple-responses",
        completer_config=completer_config,
        score=graded_task_tree.score,
        num_leaf_nodes=len(graded_task_tree.get_leaf_nodes()),
        num_invalid_leaf_nodes=len(
            [node for node in graded_task_tree.get_leaf_nodes() if not node.valid_score]
        ),
        graded_at=get_timestamp(),
        graded_task_tree=graded_task_tree,
        token_usage=token_usage,
    )
    output = {
        "paper_id": args.paper_id,
        "submission": str(args.submission),
        "judge_output": judge_output.to_dict(),
        "score": judge_output.score,
    }
    path = args.out_dir / "grader_output.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"paper_id": args.paper_id, "score": judge_output.score, "out": str(path)}, indent=2))
    if args.print_failures > 0:
        print(_format_failure_feedback(judge_output.graded_task_tree.to_dict(), args.print_failures))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(grade(parse_args())))
