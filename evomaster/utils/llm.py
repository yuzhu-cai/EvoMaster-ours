"""EvoMaster LLM Interface Wrapper

Provides a unified LLM calling interface with support for multiple providers.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import random
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from evomaster.utils.types import AssistantMessage, Dialog, FunctionCall, ToolCall


# ---------------------------------------------------------------------------
# Context overflow detection (reference: OpenCode provider/error.ts)
# ---------------------------------------------------------------------------

class ContextOverflowError(Exception):
    """LLM API rejected the request: context too long. Not retryable; caller must compact."""
    pass


# Covers overflow error message patterns from mainstream LLM providers (lowercase matching)
_OVERFLOW_PATTERNS = [
    "prompt is too long",                # Anthropic
    "exceeds the context window",        # OpenAI
    "maximum context length",            # OpenRouter / DeepSeek
    "context_length_exceeded",           # Generic
    "token count exceeds",               # DeepSeek
    "too many tokens",                   # Generic
    "reduce the length of the messages", # Groq
    "request entity too large",          # HTTP 413
    "input is too long",                 # Bedrock
    "maximum prompt length",             # xAI (Grok)
    "context window exceeds limit",      # MiniMax
    "exceeded model token limit",        # Kimi / Moonshot
]


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string.

    Args:
        image_path: Image file path.

    Returns:
        Base64 encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the MIME type of an image based on its file extension.

    Args:
        image_path: Image file path.

    Returns:
        MIME type string.
    """
    suffix = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return media_types.get(suffix, "image/png")


def build_multimodal_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    """Build a list of multimodal content blocks containing text and images.

    Generates OpenAI-format content block lists, compatible with OpenAI / DeepSeek / OpenRouter APIs.

    Args:
        text: Text content.
        image_paths: List of image file paths.

    Returns:
        Content block list, for example:
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Please analyze these images"}
        ]
    """
    content_blocks: list[dict[str, Any]] = []

    # Add images first
    for img_path in image_paths:
        media_type = get_image_media_type(img_path)
        b64_data = encode_image_to_base64(img_path)
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64_data}"
            }
        })

    # Then add text
    content_blocks.append({
        "type": "text",
        "text": text,
    })

    return content_blocks


def truncate_content(content: str, max_length: int = 5000, head_length: int = 2500, tail_length: int = 2500) -> str:
    """Return the full content (no truncation), for log display of complete content.

    Args:
        content: The content to display.
        max_length: Unused, kept for parameter compatibility.
        head_length: Unused, kept for parameter compatibility.
        tail_length: Unused, kept for parameter compatibility.

    Returns:
        The full content.
    """
    return content


class _SDKRetryReasonFilter(logging.Filter):
    """Enrich the OpenAI / Anthropic SDK retry log with the underlying cause.

    Both SDKs emit two separate records around a retry:

        log.debug("Encountered httpx.TimeoutException", exc_info=True)  # or similar
        log.info("Retrying request to %s in %f seconds", url, timeout)

    The ``Encountered ...`` record carries the actual exception via
    ``exc_info`` but is only visible at DEBUG. Users running at INFO see the
    ``Retrying request ...`` line with no context and cannot tell whether
    it was a network timeout, a 5xx, a 429, etc.

    This filter attaches to the SDK's ``_base_client`` logger, captures the
    exception from the most recent ``Encountered`` record (per thread), and
    appends a short summary to the next ``Retrying request`` record on the
    same thread. To make the ``Encountered`` record reachable we have to
    raise the logger level to DEBUG — but we drop every DEBUG record here
    after inspecting it so the console is not flooded with the SDK's normal
    per-request chatter (``Request options``, ``Sending HTTP Request``,
    ``HTTP Response``, ``request_id``, etc.).
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-thread storage so concurrent requests don't scramble reasons.
        self._local = threading.local()

    def _get_last_reason(self) -> str | None:
        return getattr(self._local, "last_reason", None)

    def _set_last_reason(self, value: str | None) -> None:
        self._local.last_reason = value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            # Unknown record: if it's DEBUG, drop per our policy; else keep.
            return record.levelno > logging.DEBUG

        if msg.startswith("Encountered "):
            exc_info = record.exc_info
            if exc_info and exc_info[1] is not None:
                exc = exc_info[1]
                exc_str = str(exc).strip()
                summary = f"{type(exc).__module__}.{type(exc).__name__}"
                # Trim the redundant "builtins." prefix for stdlib errors.
                if summary.startswith("builtins."):
                    summary = summary[len("builtins."):]
                status = getattr(exc, "status_code", None)
                if status is not None:
                    summary += f" (status={status})"
                if exc_str:
                    summary += f": {exc_str}"
                self._set_last_reason(summary)
            else:
                # Fall back to the bare "Encountered <X>" string.
                self._set_last_reason(msg)
            # Captured — drop so it doesn't show up in the console.
            return False

        if msg.startswith("Retrying request to"):
            reason = self._get_last_reason() or "unknown"
            # Rewrite the record so the reason shows up wherever the original
            # line was going. Pre-render with the current args to keep
            # downstream formatting simple and avoid mutating args.
            try:
                rendered = record.msg % record.args if record.args else record.msg
            except Exception:
                rendered = record.msg
            record.msg = f"{rendered} (reason: {reason})"
            record.args = None
            self._set_last_reason(None)
            return True

        # Everything else: suppress the SDK's DEBUG chatter (Request options,
        # Sending HTTP Request, HTTP Response, request_id, ...), let INFO
        # and above through unchanged.
        return record.levelno > logging.DEBUG


_SDK_RETRY_LOGGER_NAMES = (
    "openai._base_client",
    "anthropic._base_client",
)

_sdk_retry_filter_installed = False


def _install_sdk_retry_reason_filter() -> None:
    """Attach :class:`_SDKRetryReasonFilter` to the OpenAI / Anthropic SDK loggers.

    We raise each logger's level to DEBUG so the ``Encountered ...`` record
    survives the level gate and reaches the filter; the filter then drops
    every DEBUG record itself so the extra chatter never reaches the root
    handlers. Safe to call multiple times — the filter is only installed
    once per logger.
    """
    global _sdk_retry_filter_installed
    if _sdk_retry_filter_installed:
        return
    filt = _SDKRetryReasonFilter()
    for name in _SDK_RETRY_LOGGER_NAMES:
        lg = logging.getLogger(name)
        # DEBUG so the "Encountered ..." record passes the logger-level gate
        # and reaches our filter. The filter drops DEBUG records itself so
        # this does not spam the console.
        if lg.level == logging.NOTSET or lg.level > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
        # Filter has to run on the logger (not a handler) so it sees records
        # before they propagate to root handlers.
        already = any(isinstance(f, _SDKRetryReasonFilter) for f in lg.filters)
        if not already:
            lg.addFilter(filt)
    _sdk_retry_filter_installed = True


_install_sdk_retry_reason_filter()


def _error_status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is not None:
        return int(status)
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if status is not None else None


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: Literal["openai", "anthropic","deepseek","openrouter"] = Field(description="LLM provider")
    model: str = Field(description="Model name")
    api_key: str = Field(description="API Key, must be provided in config")
    base_url: str | None = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum generation tokens")
    max_completion_tokens: int | None = Field(default=None, description="OpenAI reasoning-model generation token limit")
    max_tokens_param: Literal["auto", "max_tokens", "max_completion_tokens"] = Field(
        default="auto",
        description="Which Chat Completions token-limit parameter to send.",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None,
        description="Optional reasoning effort for OpenAI-compatible reasoning models.",
    )
    timeout: int = Field(default=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    use_completion_api: bool = Field(default=False, description="Use Completion API instead of Chat API")


class LLMResponse(BaseModel):
    """LLM response."""
    content: str | None = Field(default=None, description="Generated text content")
    tool_calls: list[ToolCall] | None = Field(default=None, description="Tool call list")
    reasoning_content: str | None = Field(
        default=None,
        description="DeepSeek thinking mode: echoed on the next multi-turn request",
    )
    finish_reason: str | None = Field(default=None, description="Finish reason")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    meta: dict[str, Any] = Field(default_factory=dict, description="Other metadata")

    def to_assistant_message(self) -> AssistantMessage:
        """Convert to an AssistantMessage.

        Returns:
            AssistantMessage instance.
        """
        return AssistantMessage(
            content=self.content,
            tool_calls=self.tool_calls,
            reasoning_content=self.reasoning_content,
            meta={
                "finish_reason": self.finish_reason,
                "usage": self.usage,
                **self.meta,
            }
        )


class BaseLLM(ABC):
    """LLM base class.

    Defines a unified LLM calling interface.
    """

    def __init__(self, config: LLMConfig, output_config: dict[str, Any] | None = None):
        """Initialize the LLM.

        Args:
            config: LLM configuration.
            output_config: Output display configuration, including:
                - show_in_console: Whether to display in terminal.
                - log_to_file: Whether to log to file.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_config = output_config or {}
        self.show_in_console = self.output_config.get("show_in_console", False)
        self.log_to_file = self.output_config.get("log_to_file", False)
        # Track logged message count to avoid duplicate logging of system messages and initial task descriptions
        self._logged_message_count = 0
        self._setup()

    def _setup(self) -> None:
        """Initialization setup, implemented by subclasses."""
        pass

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the LLM API (subclass implementation).

        Args:
            messages: Message list (API format).
            tools: Tool specification list (API format).
            **kwargs: Additional parameters.

        Returns:
            LLM response.
        """
        pass

    def query(
        self,
        dialog: Dialog,
        **kwargs: Any,
    ) -> AssistantMessage:
        """Query the LLM.

        Args:
            dialog: Dialog object.
            **kwargs: Additional parameters (override config).

        Returns:
            Assistant message.
        """
        # Convert to API format
        messages = dialog.get_messages_for_api()
        tools = self._convert_tools(dialog.tools) if dialog.tools else None

        # Log request (if logging enabled)
        if self.log_to_file:
            self._log_request(messages, tools)

        # Call API (with retry)
        # breakpoint()
        response = self._call_with_retry(messages, tools, **kwargs)
        # breakpoint()
        # Log response (if logging enabled)
        if self.log_to_file:
            self._log_response(response)

        # Convert to AssistantMessage
        return response.to_assistant_message()

    def _log_request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> None:
        """Log the LLM request.

        Optimization: only logs new messages to avoid duplicate logging of system
        messages and initial task descriptions. First request logs all messages;
        subsequent requests only log new messages. When message count decreases
        (e.g., after context reset), resets the counter and logs all messages.
        """
        self.logger.info("=" * 80)
        self.logger.info("LLM Request:")
        self.logger.info(f"Model: {self.config.model}")
        if tools:
            self.logger.info(f"Tools: {[t.get('function', {}).get('name', 'unknown') for t in tools]}")
        
        # Check if this is the start of a new conversation (message count decreased, usually after context reset)
        if len(messages) <= self._logged_message_count:
            # Message count decreased, indicating a new conversation; reset counter
            self.logger.info("New conversation detected (message count decreased), resetting log counter")
            self._logged_message_count = 0
        
        # Calculate messages to log
        new_messages = messages[self._logged_message_count:]

        if self._logged_message_count == 0:
            # First request, log all messages (including system messages and initial task description)
            self.logger.info("Messages:")
            for i, msg in enumerate(messages):
                self._log_single_message(i + 1, msg)
            self._logged_message_count = len(messages)
        else:
            # Subsequent requests, only log new messages
            if new_messages:
                self.logger.info(f"New Messages (continuing from message {self._logged_message_count + 1}):")
                for i, msg in enumerate(new_messages):
                    self._log_single_message(self._logged_message_count + i + 1, msg)
                self._logged_message_count = len(messages)
            else:
                # No new messages (possibly due to context truncation reducing message count)
                self.logger.info(f"Messages: (same as previous, total: {len(messages)})")
                # Update the logged message count to avoid future duplicates
                self._logged_message_count = len(messages)

        self.logger.info("=" * 80)

    def _log_single_message(self, index: int, msg: dict[str, Any]) -> None:
        """Log a single message, with special handling for tool call display.

        Args:
            index: Message sequence number.
            msg: Message dictionary.
        """
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # If this is an assistant message with tool calls
        if role == "assistant" and tool_calls:
            if content:
                # Has text content, display it first
                content_display = truncate_content(content) if isinstance(content, str) else f"[Multimodal content with {len(content)} blocks]"
                self.logger.info(f"  [{index}] {role}: {content_display}")
            else:
                # Only tool calls, display placeholder
                self.logger.info(f"  [{index}] {role}: [Calling {len(tool_calls)} tool(s)]")

            # Display details of each tool call
            for i, tc in enumerate(tool_calls):
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "")

                    # Format arguments (if JSON string, try to parse and pretty-print)
                    try:
                        import json
                        args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        args_display = json.dumps(args_dict, indent=2, ensure_ascii=False)
                    except:
                        args_display = str(tool_args)

                    self.logger.info(f"      Tool #{i+1}: {tool_name}")
                    self.logger.info(f"      Args: {args_display}")
        else:
            # Normal message (no tool calls)
            if isinstance(content, str):
                content = truncate_content(content)
            elif isinstance(content, list):
                # Multimodal content: display summary information
                text_blocks = [b for b in content if b.get("type") == "text"]
                image_blocks = [b for b in content if b.get("type") in ("image_url", "image")]
                text_preview = text_blocks[0].get("text", "")[:200] if text_blocks else ""
                content = f"[Multimodal: {len(image_blocks)} image(s)] {text_preview}..."
            self.logger.info(f"  [{index}] {role}: {content}")

    def _log_response(self, response: LLMResponse) -> None:
        """Log the LLM response."""
        self.logger.info("=" * 80)
        self.logger.info("LLM Response:")
        if response.content:
            # Truncate overly long content
            content = truncate_content(response.content)
            self.logger.info(f"Content: {content}")
        if response.tool_calls:
            self.logger.info(f"Tool Calls: {[tc.function.name for tc in response.tool_calls]}")
        if response.usage:
            self.logger.info(f"Usage: {response.usage}")
        self.logger.info("=" * 80)

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call with retry.

        Reference: OpenCode — context overflow and other non-retryable 4xx errors
        are not retried; instead they are raised immediately so the caller can
        perform compact recovery.

        Args:
            messages: Message list.
            tools: Tool list.
            **kwargs: Additional parameters.

        Returns:
            LLM response.
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                self._apply_client_rate_limit()
                return self._call(messages, tools, **kwargs)
            except Exception as e:
                last_error = e

                # Context overflow → do not retry; raise immediately for caller to compact
                if self._is_context_overflow_error(e):
                    self.logger.warning("Context overflow (non-retryable): %s", e)
                    raise ContextOverflowError(str(e)) from e

                # Other 4xx (non-429 rate-limit) are also not retried
                status = _error_status_code(e)
                if status and 400 <= status < 500 and status != 429:
                    self.logger.warning("Non-retryable %d error: %s", status, e)
                    raise
                if status == 429:
                    self._record_rate_limit_cooldown()

                # Retryable error: normal retry
                self.logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, self.config.max_retries, e,
                )

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)

        # All retries exhausted
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts") from last_error

    def _apply_client_rate_limit(self) -> None:
        """Optionally serialize LLM calls across worker processes.

        PaperBench batch runs can spawn many Python workers. If each worker
        sends a 30k-80k token request at once, OpenAI-compatible gateways often
        return TPM 429s and all workers retry in lock-step. Set
        EVOMASTER_LLM_MIN_INTERVAL_SECONDS to a positive value to enforce a
        host-wide minimum interval between calls for the same model/base URL.
        """
        try:
            min_interval = float(os.environ.get("EVOMASTER_LLM_MIN_INTERVAL_SECONDS", "0") or 0)
        except ValueError:
            min_interval = 0.0
        if min_interval <= 0:
            return

        try:
            jitter = float(os.environ.get("EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS", "0") or 0)
        except ValueError:
            jitter = 0.0

        lock_dir = Path(os.environ.get("EVOMASTER_LLM_RATE_LIMIT_DIR", "/tmp/evomaster_llm_rate_limits"))
        key_raw = os.environ.get("EVOMASTER_LLM_RATE_LIMIT_KEY") or f"{self.config.base_url or 'default'}::{self.config.model}"
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_raw).strip("._") or "default"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{key}.lock"
        stamp_path = lock_dir / f"{key}.last"

        try:
            import fcntl
        except ImportError:
            if jitter > 0:
                time.sleep(random.uniform(0, jitter))
            return

        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                last = 0.0
                with contextlib.suppress(OSError, ValueError):
                    last = float(stamp_path.read_text(encoding="utf-8").strip() or 0)
                wait = min_interval - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
                if jitter > 0:
                    time.sleep(random.uniform(0, jitter))
                stamp_path.write_text(str(time.monotonic()), encoding="utf-8")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _record_rate_limit_cooldown(self) -> None:
        """Push the shared rate-limit stamp into the future after a 429.

        A fixed inter-request interval prevents most bursts, but PaperBench
        prompts can grow to tens of thousands of tokens. When the gateway still
        returns TPM throttling, this host-wide cooldown keeps parallel workers
        from retrying in lock-step.
        """
        try:
            cooldown = float(os.environ.get("EVOMASTER_LLM_429_COOLDOWN_SECONDS", "0") or 0)
        except ValueError:
            cooldown = 0.0
        if cooldown <= 0:
            return

        lock_dir = Path(os.environ.get("EVOMASTER_LLM_RATE_LIMIT_DIR", "/tmp/evomaster_llm_rate_limits"))
        key_raw = os.environ.get("EVOMASTER_LLM_RATE_LIMIT_KEY") or f"{self.config.base_url or 'default'}::{self.config.model}"
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key_raw).strip("._") or "default"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{key}.lock"
        stamp_path = lock_dir / f"{key}.last"

        try:
            import fcntl
        except ImportError:
            time.sleep(cooldown)
            return

        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                current = 0.0
                with contextlib.suppress(OSError, ValueError):
                    current = float(stamp_path.read_text(encoding="utf-8").strip() or 0)
                stamp_path.write_text(str(max(current, time.monotonic() + cooldown)), encoding="utf-8")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _is_context_overflow_error(error: Exception) -> bool:
        """Check whether the error is a context overflow error.

        Reference: OpenCode OVERFLOW_PATTERNS, covering error messages from mainstream providers.
        """
        error_msg = str(error).lower()
        status = _error_status_code(error)
        # HTTP 413 (Request Entity Too Large) is always an overflow
        if status == 413:
            return True
        # 400 BadRequest + contains overflow keywords
        if status == 400:
            return any(p in error_msg for p in _OVERFLOW_PATTERNS)
        return False

    def _convert_tools(self, tool_specs: list) -> list[dict[str, Any]]:
        """Convert tool specifications to API format.

        Args:
            tool_specs: List of ToolSpec objects.

        Returns:
            List of tools in API format.
        """
        return [spec.model_dump() for spec in tool_specs]


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation.

    Supports the OpenAI API and compatible interfaces (e.g., vLLM, Ollama, etc.).
    """

    def _setup(self) -> None:
        """Set up the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        # API key must be provided in config
        if not self.config.api_key:
            raise ValueError("OpenAI API key must be provided in config")

        # Create the client
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        self.client = OpenAI(**client_kwargs)

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the OpenAI API."""
        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "timeout": kwargs.get("timeout", self.config.timeout)
        }

        token_limit = kwargs.get(
            "max_completion_tokens",
            kwargs.get("max_tokens", self.config.max_completion_tokens or self.config.max_tokens),
        )
        token_param = self._token_limit_param_name()
        if token_limit:
            request_params[token_param] = token_limit
        if self.config.reasoning_effort:
            request_params["reasoning_effort"] = self.config.reasoning_effort

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Call the API. Some local OpenAI SDK / Pydantic combinations raise
        # `TypeError: by_alias NoneType` while serializing tool requests; fall
        # back to a raw HTTP call so tool-enabled agents still run.
        try:
            response = self.client.chat.completions.create(**request_params)
        except TypeError as e:
            if not self._should_raw_fallback(e):
                raise
            try:
                return self._call_raw_chat(request_params)
            except Exception as raw_error:
                if self._is_unsupported_token_param(raw_error, token_param):
                    return self._call_raw_chat(
                        self._with_alternate_token_param(request_params, token_param)
                    )
                raise
        except Exception as e:
            if self._is_unsupported_token_param(e, token_param):
                response = self.client.chat.completions.create(
                    **self._with_alternate_token_param(request_params, token_param)
                )
            else:
                raise

        # Parse the response
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            reasoning_content=getattr(message, "reasoning_content", None),
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            meta={
                "model": response.model,
                "response_id": response.id,
            }
        )

    def _token_limit_param_name(self) -> str:
        """Return the token limit parameter compatible with this model/config."""
        if self.config.max_tokens_param != "auto":
            return self.config.max_tokens_param

        model = (self.config.model or "").lower()
        reasoning_markers = ("gpt-5", "o1", "o3", "o4")
        if any(marker in model for marker in reasoning_markers):
            return "max_completion_tokens"
        return "max_tokens"

    @staticmethod
    def _alternate_token_param(param: str) -> str:
        return "max_completion_tokens" if param == "max_tokens" else "max_tokens"

    def _with_alternate_token_param(self, request_params: dict[str, Any], current_param: str) -> dict[str, Any]:
        """Swap max_tokens <-> max_completion_tokens after gateway compatibility errors."""
        alternate = self._alternate_token_param(current_param)
        retry_params = request_params.copy()
        if current_param in retry_params:
            retry_params[alternate] = retry_params.pop(current_param)
        return retry_params

    @staticmethod
    def _should_raw_fallback(error: TypeError) -> bool:
        msg = str(error)
        return "by_alias" in msg or "unexpected keyword" in msg

    @staticmethod
    def _is_unsupported_token_param(error: Exception, token_param: str) -> bool:
        if _error_status_code(error) != 400:
            return False
        msg = str(error).lower()
        return (
            token_param.lower() in msg
            and "unsupported" in msg
            and ("max_tokens" in msg or "max_completion_tokens" in msg)
        )

    def _call_raw_chat(self, request_params: dict[str, Any]) -> LLMResponse:
        """Call a Chat Completions-compatible endpoint without the SDK."""
        import httpx

        body = request_params.copy()
        timeout = body.pop("timeout", self.config.timeout)
        url_base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{url_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if response.status_code >= 400:
            body_preview = response.text[:2000]
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} from {url}: {body_preview}",
                request=response.request,
                response=response,
            )
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message") or {}
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=(tc.get("function") or {}).get("name", ""),
                        arguments=(tc.get("function") or {}).get("arguments", ""),
                    ),
                )
                for tc in message.get("tool_calls", [])
            ]
        usage = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            reasoning_content=message.get("reasoning_content")
            or (message.get("provider_specific_fields") or {}).get("reasoning_content"),
            finish_reason=choice.get("finish_reason"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            meta={
                "model": data.get("model", self.config.model),
                "response_id": data.get("id", ""),
                "api_type": "raw_chat",
            },
        )

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation.

    Supports both the Chat Completion API and the Completion API.
    """

    def _setup(self) -> None:
        """Set up the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        # API key must be provided in config
        if not self.config.api_key:
            raise ValueError("OpenAI API key must be provided in config")

        # Create the client
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        self.client = OpenAI(**client_kwargs)

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert a message list to a single prompt string (for the Completion API).

        Format is consistent with the X-Master r1_tool.jinja template.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"<｜User｜> {content} <｜Assistant｜>")
            elif role == "assistant":
                parts.append(content)
            elif role == "tool":
                # Wrap tool results in execution_results tags
                parts.append(f"<execution_results>{content}</execution_results>")

        return "".join(parts)

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the DeepSeek API."""
        if self.config.use_completion_api:
            return self._call_completion(messages, **kwargs)
        else:
            return self._call_chat(messages, tools, **kwargs)

    def _call_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the Completion API."""
        prompt = self._messages_to_prompt(messages)

        request_params = {
            "model": self.config.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "timeout": kwargs.get("timeout", self.config.timeout),
        }

        if self.config.max_tokens:
            request_params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)

        # Call the Completion API
        response = self.client.completions.create(**request_params)

        # Parse the response
        choice = response.choices[0]

        return LLMResponse(
            content=choice.text,
            tool_calls=None,  # Completion API does not support native tool calls
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            meta={
                "model": response.model,
                "response_id": response.id,
                "api_type": "completion",
            }
        )

    def _call_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the Chat Completion API."""
        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "extra_body": {
                "chat_template_kwargs": {"thinking": True},
                "separate_reasoning": True
            }
        }

        if self.config.max_tokens:
            request_params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)

        if tools:
            # Clean None values from tools (e.g., strict=None); some APIs do not accept None
            cleaned_tools = []
            for tool in tools:
                cleaned_tool = tool.copy()
                if "function" in cleaned_tool and isinstance(cleaned_tool["function"], dict):
                    cleaned_function = cleaned_tool["function"].copy()
                    # Remove the strict=None field
                    if cleaned_function.get("strict") is None:
                        cleaned_function.pop("strict", None)
                    cleaned_tool["function"] = cleaned_function
                cleaned_tools.append(cleaned_tool)
            request_params["tools"] = cleaned_tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Call the API
        response = self.client.chat.completions.create(**request_params)

        # Parse the response
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                for tc in message.tool_calls
            ]

        # DeepSeek thinking mode (separate_reasoning): SDK keeps extra fields on the message model
        reasoning_content = getattr(message, "reasoning_content", None)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            meta={
                "model": response.model,
                "response_id": response.id,
                "api_type": "chat",
            }
        )


class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation.

    Supports Claude series models.
    """

    def _setup(self) -> None:
        """Set up the Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        # API key must be provided in config
        if not self.config.api_key:
            raise ValueError("Anthropic API key must be provided in config")

        # Create the client
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            # Set auth_token so the SDK sends the correct Bearer token.
            client_kwargs["auth_token"] = self.config.api_key

        self.client = Anthropic(**client_kwargs)

    @staticmethod
    def _convert_content_for_anthropic(content):
        """Convert OpenAI-format multimodal content to Anthropic format.

        OpenAI format:
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
             {"type": "text", "text": "..."}]

        Anthropic format:
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
             {"type": "text", "text": "..."}]
        """
        if not isinstance(content, list):
            return content

        converted = []
        for block in content:
            if block.get("type") == "image_url":
                # Parse data URI: "data:image/png;base64,<data>"
                url = block["image_url"]["url"]
                if url.startswith("data:"):
                    # Parse MIME type and base64 data
                    header, b64_data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        }
                    })
                else:
                    # URL-based image (also supported by Anthropic)
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        }
                    })
            elif block.get("type") == "text":
                converted.append(block)
            else:
                converted.append(block)
        return converted

    @staticmethod
    def _convert_tools_for_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Anthropic format.

        OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
        """
        converted = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
            converted.append(anthropic_tool)
        return converted

    @staticmethod
    def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-format message list to Anthropic format.

        Conversion rules:
        - system messages are extracted as a separate field
        - assistant + tool_calls -> content includes text + tool_use blocks
        - role:"tool" -> role:"user" + tool_result content blocks (consecutive ones are merged)
        """
        import json as _json

        system_message = None
        anthropic_messages: list[dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "system":
                system_message = msg.get("content", "")
                i += 1
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Build Anthropic-format content blocks
                    content_blocks = []
                    text = msg.get("content")
                    if text and str(text).strip():
                        content_blocks.append({"type": "text", "text": str(text)})
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        try:
                            input_data = _json.loads(func.get("arguments", "{}"))
                        except (_json.JSONDecodeError, TypeError):
                            input_data = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": input_data,
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    content = msg.get("content", "")
                    anthropic_messages.append({"role": "assistant", "content": content or " "})
                i += 1
                continue

            if role == "tool":
                # Collect consecutive tool messages and merge into a single user message
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    t = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": t.get("tool_call_id", ""),
                        "content": t.get("content", "") or " ",
                    })
                    i += 1
                anthropic_messages.append({"role": "user", "content": tool_results})
                continue

            # user or other
            content = msg.get("content", "")
            if isinstance(content, list):
                content = AnthropicLLM._convert_content_for_anthropic(content)
            anthropic_messages.append({"role": role, "content": content})
            i += 1

        return system_message, anthropic_messages

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the Anthropic API."""
        # Convert OpenAI-format messages to Anthropic format
        system_message, user_messages = self._convert_messages_for_anthropic(messages)

        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "timeout": kwargs.get("timeout", self.config.timeout),
        }

        if system_message:
            request_params["system"] = system_message

        if tools:
            request_params["tools"] = self._convert_tools_for_anthropic(tools)
            request_params["tool_choice"] = kwargs.get("tool_choice", {"type": "auto"})

        # Call the API
        response = self.client.messages.create(**request_params)

        # Parse the response
        content_text = None
        tool_calls = None

        for content in response.content:
            if content.type == "text":
                content_text = content.text
            elif content.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                # Anthropic tool call format needs conversion
                import json
                tool_calls.append(
                    ToolCall(
                        id=content.id,
                        type="function",
                        function=FunctionCall(
                            name=content.name,
                            arguments=json.dumps(content.input),
                        )
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            meta={
                "model": response.model,
                "response_id": response.id,
            }
        )


def create_llm(config: LLMConfig, output_config: dict[str, Any] | None = None) -> BaseLLM:
    """LLM factory function.

    Args:
        config: LLM configuration.
        output_config: Output display configuration.

    Returns:
        LLM instance.

    Raises:
        ValueError: Unsupported provider.
    """
    if config.provider == "openai" or config.provider == "openrouter":
        return OpenAILLM(config, output_config=output_config)
    elif config.provider == "anthropic":
        return AnthropicLLM(config, output_config=output_config)
    elif config.provider == "deepseek":
        return DeepSeekLLM(config, output_config=output_config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
