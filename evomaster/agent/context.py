"""EvoMaster Agent 上下文管理

提供上下文管理功能，包括对话历史管理、上下文窗口控制、历史压缩等。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from evomaster.utils.types import AssistantMessage, Dialog, Message, ToolMessage
else:
    from evomaster.utils.types import AssistantMessage, Dialog, Message, ToolMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compaction prompts
# ---------------------------------------------------------------------------

COMPACTION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Produce a detailed but concise summary of the conversation. "
    "Focus on information that would be helpful for continuing the conversation, including: "
    "what was done, what is currently being worked on, key user requests and constraints, "
    "important decisions and their reasons, and key facts or data. "
    "Do not respond to any questions in the conversation, only output the summary."
)

COMPACTION_USER_PROMPT = """\
Summarize the conversation above for handoff to a continuing agent.

Use this template:
---
## Goal
[What the user is trying to accomplish]

## Key Decisions & Discoveries
[Important findings, user preferences, constraints]

## Accomplished
[What was completed, what is in progress, what remains]

## Context
[Key facts, data, or state needed to continue naturally]
---"""

# Prune 阈值
_PRUNE_PROTECT_TOKENS = 40_000  # 保护最近的 tool 输出不被清除
_PRUNE_MINIMUM_TOKENS = 10_000  # 至少清除这么多才值得 prune
_RESERVED_OUTPUT_TOKENS = 20_000  # 预留给 LLM 回复的 token 空间


class TruncationStrategy(str, Enum):
    """历史截断策略"""
    NONE = "none"  # 不截断
    LATEST_HALF = "latest_half"  # 保留最新一半
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    SUMMARY = "summary"  # 摘要压缩


class ContextConfig(BaseModel):
    """上下文管理配置"""
    max_tokens: int = Field(default=128000, description="最大 token 数")
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.LATEST_HALF,
        description="截断策略"
    )
    preserve_system_messages: bool = Field(
        default=True,
        description="是否保留系统消息"
    )
    preserve_recent_turns: int = Field(
        default=5,
        description="保留最近的对话轮数"
    )


class ContextManager:
    """上下文管理器
    
    负责管理对话上下文，包括：
    - 上下文窗口大小控制
    - 历史消息截断和压缩
    - Token 计数（可扩展）
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        self._token_counter: TokenCounter | None = None
        self._summary_llm: BaseLLM | None = None
        self._last_prompt_tokens: int = 0
        self._last_prompt_msg_count: int = 0
        self.on_before_compaction: Callable[[list[Message]], None] | None = None

    def set_token_counter(self, counter: TokenCounter) -> None:
        """设置 token 计数器"""
        self._token_counter = counter

    def set_summary_llm(self, llm: BaseLLM) -> None:
        """设置用于 auto-compact 摘要的 LLM

        当截断策略为 SUMMARY 时，使用此 LLM 对旧消息进行摘要压缩。
        """
        self._summary_llm = llm

    def update_usage(self, usage: dict[str, int], msg_count: int = 0) -> None:
        """记录 LLM API 返回的真实 token 用量。

        仅关心 prompt_tokens（即实际 dialog 占用的 input token 数）。
        completion_tokens 包含 thinking tokens，但 thinking 不会存入 dialog，
        因此不能用 total_tokens 判断 dialog 大小。

        Args:
            usage: API 返回的 usage 字典
            msg_count: 发送给 API 的消息数量（用于增量估算）
        """
        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
        if msg_count > 0:
            self._last_prompt_msg_count = msg_count

    def estimate_tokens(self, dialog: Dialog) -> int:
        """估算对话的 token 数

        如果设置了 token 计数器，使用计数器；否则使用简单估算。
        """
        if self._token_counter:
            return self._token_counter.count_dialog(dialog)

        total_chars = self._count_messages_chars(dialog.messages)

        # Tool specs 随每次 API 请求发送，也占 context
        if dialog.tools:
            import json as _json
            for spec in dialog.tools:
                try:
                    total_chars += len(_json.dumps(spec.model_dump()))
                except Exception:
                    total_chars += 500  # fallback per tool

        return total_chars // 4

    def _count_messages_chars(self, messages: list[Message]) -> int:
        """统计一组消息的字符数（用于 token 估算）"""
        total_chars = 0
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # 多模态内容：只计算文本部分，图片按固定 token 数估算
                for block in content:
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") in ("image_url", "image"):
                        total_chars += 3000  # 图片约占 ~750 tokens，按 3000 字符估算
            # tool_calls 的 arguments 也占 token
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.function.name) + len(tc.function.arguments or "")
        return total_chars

    def _estimate_messages_tokens(self, messages: list[Message]) -> int:
        """估算一组消息的 token 数（不含 tool specs），用于增量计算。"""
        return self._count_messages_chars(messages) // 4

    def should_truncate(self, dialog: Dialog) -> bool:
        """判断是否需要截断"""
        return self.estimate_tokens(dialog) > self.config.max_tokens

    def truncate(self, dialog: Dialog) -> Dialog:
        """根据策略截断对话历史
        
        Returns:
            截断后的新 Dialog 对象
        """
        if self.config.truncation_strategy == TruncationStrategy.NONE:
            return dialog
        elif self.config.truncation_strategy == TruncationStrategy.LATEST_HALF:
            return self._truncate_latest_half(dialog)
        elif self.config.truncation_strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._truncate_sliding_window(dialog)
        elif self.config.truncation_strategy == TruncationStrategy.SUMMARY:
            return self._truncate_with_summary(dialog)
        else:
            return dialog

    def _truncate_latest_half(self, dialog: Dialog) -> Dialog:
        """保留最新一半的历史
        
        保留系统消息和用户初始消息，然后保留最近一半的对话。
        """
        messages = dialog.messages
        
        # 找到第一个 assistant 消息的位置
        assistant_start = 0
        for i, msg in enumerate(messages):
            if msg.role.value == "assistant":
                assistant_start = i
                break
        
        # 计算需要保留的消息数量
        num_messages = len(messages)
        num_to_truncate = num_messages - assistant_start
        num_to_preserve = num_to_truncate // 2
        preserve_start = num_messages - num_to_preserve
        
        # 确保从 assistant 消息开始
        while preserve_start < num_messages and messages[preserve_start].role.value != "assistant":
            preserve_start += 1
        
        if preserve_start >= num_messages:
            # 无法截断，返回原对话
            return dialog
        
        # 构建新对话
        new_messages = messages[:assistant_start] + messages[preserve_start:]
        
        return Dialog(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "truncated": True, "strategy": "latest_half"}
        )

    def _truncate_sliding_window(self, dialog: Dialog) -> Dialog:
        """滑动窗口截断

        保留系统消息和最近 N 轮对话。
        一轮 = 一个 assistant 消息及其关联的 tool 消息。
        """
        messages = dialog.messages
        preserve_turns = self.config.preserve_recent_turns

        # 分离系统消息和其他消息
        system_messages: list[Message] = []
        other_messages: list[Message] = []

        for msg in messages:
            if msg.role.value == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # 从后往前数 preserve_turns 个 assistant 消息，确定保留起点
        assistant_count = 0
        keep_from = len(other_messages)
        for i in range(len(other_messages) - 1, -1, -1):
            if other_messages[i].role.value == "assistant":
                assistant_count += 1
                if assistant_count >= preserve_turns:
                    keep_from = i
                    break

        if keep_from == 0:
            return dialog

        # 保留最近的消息
        new_messages = system_messages + other_messages[keep_from:]

        return Dialog(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "truncated": True, "strategy": "sliding_window"}
        )

    def _truncate_with_summary(self, dialog: Dialog) -> Dialog:
        """Auto-compact：用 LLM 摘要旧消息，替换为紧凑的上下文总结。

        将对话分为三部分：
        1. system_msgs: 系统消息 + 初始用户消息（保持不动）
        2. old_msgs: 需要被摘要的旧消息
        3. recent_msgs: 最近保留的消息（保持不动）

        把 old_msgs 作为完整对话（含 tool_calls 结构）发给摘要 LLM，
        摘要后的 dialog = system_msgs + [UserMessage(摘要)] + recent_msgs。
        如果 LLM 调用失败，回退到 latest_half 策略。
        """
        if self._summary_llm is None:
            logger.warning("Summary LLM not set, falling back to latest_half")
            return self._truncate_latest_half(dialog)

        from evomaster.utils.types import (
            AssistantMessage as AMsg,
            Dialog as DialogCls,
            SystemMessage,
            UserMessage,
        )

        messages = dialog.messages

        # 找到第一个 assistant 消息的位置（system + initial user 之后）
        assistant_start = 0
        for i, msg in enumerate(messages):
            if msg.role.value == "assistant":
                assistant_start = i
                break

        if assistant_start == 0:
            return dialog

        # 从后往前数 preserve_recent_turns 个 assistant 消息，确定 recent_start
        assistant_count = 0
        recent_start = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role.value == "assistant":
                assistant_count += 1
                if assistant_count >= self.config.preserve_recent_turns:
                    recent_start = i
                    break

        if recent_start >= len(messages) or recent_start <= assistant_start:
            return self._truncate_latest_half(dialog)

        # system_msgs 只保留 system role 的消息，initial user message 会被摘要覆盖
        system_msgs = [m for m in messages[:assistant_start] if m.role.value == "system"]
        # old_msgs 包含 initial user message 和之后到 recent_start 的所有消息
        old_msgs = messages[len(system_msgs):recent_start]
        recent_msgs = messages[recent_start:]

        if not old_msgs:
            return dialog

        # 触发 compaction 前钩子（用于记忆提取等）
        if self.on_before_compaction:
            try:
                self.on_before_compaction(list(old_msgs))
            except Exception:
                logger.exception("on_before_compaction hook failed")

        # 构造摘要对话：system prompt + old_msgs 完整结构 + 摘要指令
        # old_msgs 保持原样不截断，让摘要 LLM 看到完整上下文
        try:
            summary_dialog = DialogCls(
                messages=[
                    SystemMessage(content=COMPACTION_SYSTEM_PROMPT),
                    *old_msgs,
                    UserMessage(content=COMPACTION_USER_PROMPT),
                ],
                tools=[],
            )

            response = self._summary_llm.query(summary_dialog)
            summary_text = response.content or ""

            if not summary_text.strip():
                logger.warning("Empty summary from LLM, falling back to latest_half")
                return self._truncate_latest_half(dialog)

            logger.info(
                "Auto-compact: summarized %d messages -> %d chars summary",
                len(old_msgs),
                len(summary_text),
            )

            # 用 user ask + assistant answer 一对消息表示摘要
            compaction_request = UserMessage(
                content="What did we do so far?",
            )
            compaction_response = AMsg(
                content=summary_text,
                meta={"summary": True, "strategy": "compaction"},
            )
            new_messages = (
                list(system_msgs)
                + [compaction_request, compaction_response]
                + list(recent_msgs)
            )

            return DialogCls(
                messages=new_messages,
                tools=dialog.tools,
                meta={**dialog.meta, "truncated": True, "strategy": "summary"},
            )

        except Exception:
            logger.exception("Auto-compact failed, falling back to latest_half")
            return self._truncate_latest_half(dialog)

    def reset_prompt_tokens(self) -> None:
        """重置 prompt_tokens 记录，用于 compact 回写后强制重新估算。"""
        self._last_prompt_tokens = 0
        self._last_prompt_msg_count = 0

    def prepare_for_query(self, dialog: Dialog) -> tuple[Dialog, bool]:
        """为 LLM 查询准备对话

        参考 OpenCode 的 isOverflow 逻辑：
        - 用真实 prompt_tokens（来自上次 LLM API 响应）+ 增量估算判断是否溢出
        - 首次调用无 usage 数据时回退到 estimate_tokens 估算
        - usable = max_tokens - reserved_output_tokens

        两层策略：
        1. tokens >= usable → 完整摘要（truncate）— 永久压缩
        2. tokens >= 80% usable → 轻量 prune（清除旧 tool 输出）— 临时视图

        Returns:
            (dialog_for_query, compacted):
            - compacted=True: 执行了永久压缩（truncate/summary），调用者应回写到 current_dialog
            - compacted=False: 无变更或仅临时 prune，不应回写（保留完整 tool 输出供未来 summary 使用）
        """
        usable = self.config.max_tokens - _RESERVED_OUTPUT_TOKENS
        if usable <= 0:
            # max_tokens 小于 reserved，无法计算有效阈值，跳过截断
            logger.warning(
                "max_tokens (%d) <= _RESERVED_OUTPUT_TOKENS (%d), skipping truncation",
                self.config.max_tokens,
                _RESERVED_OUTPUT_TOKENS,
            )
            return dialog, False

        if self._last_prompt_tokens > 0 and self._last_prompt_msg_count > 0:
            # 增量估算：上次真实 tokens + 新增消息估算
            current_msg_count = len(dialog.messages)
            if current_msg_count > self._last_prompt_msg_count:
                new_msgs = dialog.messages[self._last_prompt_msg_count:]
                delta = self._estimate_messages_tokens(new_msgs)
                tokens = self._last_prompt_tokens + delta
            else:
                tokens = self._last_prompt_tokens
        else:
            # 首次调用无 usage 数据，用全量估算
            tokens = self.estimate_tokens(dialog)

        # 5% 安全余量
        tokens = int(tokens * 1.05)

        if tokens >= usable:
            return self.truncate(dialog), True
        if tokens >= int(usable * 0.8):
            return self._prune_old_tool_outputs(dialog), False
        return dialog, False

    def is_overflow(self, usage: dict[str, int]) -> bool:
        """用 API 返回的真实 token 数判断是否需要 compact。

        参考 OpenCode isOverflow: 每次 LLM 成功返回后调用，
        用真实 total_tokens 判断是否已接近上下文极限。
        如果是，调用者应立即执行 compact 以避免下次调用溢出。
        """
        total = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        usable = self.config.max_tokens - _RESERVED_OUTPUT_TOKENS
        return total >= usable

    def _prune_old_tool_outputs(self, dialog: Dialog) -> Dialog:
        """轻量 prune：清除旧的 tool 输出，保护最近的不动。

        参考 OpenCode 的 prune 策略：
        - 从最新消息往回扫描
        - 保护最近 2 个 user turn 的 tool 输出
        - 超出保护范围的 ToolMessage 内容替换为 "[Old tool output cleared]"
        - 只在可清除量超过阈值时执行
        """
        from evomaster.utils.types import Dialog as DialogCls, ToolMessage as TMsg

        messages = dialog.messages
        tool_token_total = 0
        prunable_indices: list[int] = []
        prunable_tokens: list[int] = []
        user_turns = 0

        # 从后往前扫描
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role.value == "user":
                user_turns += 1
            if user_turns < 2:
                continue  # 保护最近 2 个 user turn

            if isinstance(msg, TMsg):
                content = msg.content or ""
                if isinstance(content, str) and len(content) > 200:
                    tokens = len(content) // 4
                    tool_token_total += tokens
                    if tool_token_total > _PRUNE_PROTECT_TOKENS:
                        prunable_indices.append(i)
                        prunable_tokens.append(tokens)

        total_prunable = sum(prunable_tokens)
        if total_prunable < _PRUNE_MINIMUM_TOKENS:
            return dialog

        logger.info(
            "Prune: clearing %d old tool outputs (~%d tokens)",
            len(prunable_indices),
            total_prunable,
        )

        new_messages = list(messages)
        for idx in prunable_indices:
            old_msg = new_messages[idx]
            assert isinstance(old_msg, TMsg)
            new_messages[idx] = TMsg(
                content="[Old tool output cleared]",
                tool_call_id=old_msg.tool_call_id,
                name=old_msg.name,
                meta=old_msg.meta,
            )

        return DialogCls(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "pruned": True},
        )


class TokenCounter(ABC):
    """Token 计数器抽象基类"""

    @abstractmethod
    def count_text(self, text: str) -> int:
        """计算文本的 token 数"""
        pass

    @abstractmethod
    def count_message(self, message: Message) -> int:
        """计算单条消息的 token 数"""
        pass

    def count_dialog(self, dialog: Dialog) -> int:
        """计算对话的总 token 数"""
        return sum(self.count_message(msg) for msg in dialog.messages)


class SimpleTokenCounter(TokenCounter):
    """简单的 Token 计数器
    
    基于字符数的简单估算。
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def count_message(self, message: Message) -> int:
        content = message.content
        if isinstance(content, str):
            content_tokens = self.count_text(content)
        elif isinstance(content, list):
            content_tokens = 0
            for block in content:
                if block.get("type") == "text":
                    content_tokens += self.count_text(block.get("text", ""))
                elif block.get("type") in ("image_url", "image"):
                    content_tokens += 750  # 图片固定估算
        else:
            content_tokens = 0
        # 额外的 token 开销（role, 格式等）
        overhead = 4
        return content_tokens + overhead

