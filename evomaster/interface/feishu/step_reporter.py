"""飞书卡片实时进度报告

卡片仅显示执行进度和文档链接，完整轨迹写入飞书文档。
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lark_oapi as lark
    from .document_writer import FeishuDocumentWriter

logger = logging.getLogger(__name__)


class FeishuStepReporter:
    """维护一张飞书卡片显示执行进度，完整轨迹写入飞书文档。

    Usage::

        reporter = FeishuStepReporter(client, chat_id, reply_to, document_writer=writer)
        reporter.send_initial_card("计算 3+5")
        playground.run(task_description=text, on_step=reporter.on_step)
        reporter.finalize("completed", "3+5=8")
    """

    def __init__(
        self,
        client: lark.Client,
        chat_id: str,
        reply_to_message_id: str | None = None,
        document_writer: FeishuDocumentWriter | None = None,
        sender_open_id: str | None = None,
    ):
        self._client = client
        self._chat_id = chat_id
        self._reply_to = reply_to_message_id
        self._card_message_id: str | None = None
        self._task_text: str = ""
        self._start_time: float = 0.0
        self._step_count: int = 0

        # 飞书文档（完整轨迹）
        self._doc_writer = document_writer
        self._sender_open_id = sender_open_id
        self._document_id: str | None = None
        self._document_url: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_initial_card(self, task_text: str) -> bool:
        """发送初始 "正在处理" 卡片，捕获 message_id 用于后续 PATCH。"""
        from .sender import send_card_message

        self._task_text = task_text[:200]
        self._start_time = time.time()

        # 创建飞书文档（完整轨迹）
        self._create_trajectory_document(task_text)

        content = self._build_progress_content(0, 0, running=True)

        message_id = send_card_message(
            self._client,
            self._chat_id,
            title="🤖 Agent 执行中...",
            content=content,
            reply_to_message_id=self._reply_to,
            header_template="wathet",
        )
        if message_id:
            self._card_message_id = message_id
            return True
        return False

    def on_step(self, step_record: Any, step_number: int, max_steps: int) -> None:
        """每步回调：更新卡片进度 + 向文档写入完整内容。"""
        if self._card_message_id is None:
            return

        self._step_count = step_number

        # 卡片：仅更新进度
        content = self._build_progress_content(step_number, max_steps, running=True)
        self._patch(
            title=f"🤖 Agent 执行中... (Step {step_number}/{max_steps})",
            content=content,
            template="wathet",
        )

        # 文档：完整内容（无截断）
        if self._doc_writer and self._document_id:
            try:
                self._append_step_to_document(step_record, step_number, max_steps)
            except Exception:
                logger.exception("Failed to append step %d to document", step_number)

    def finalize(self, status: str, final_answer: str = "") -> None:
        """最终更新卡片（任务完成/失败）。"""
        if self._card_message_id is None:
            return

        elapsed = time.time() - self._start_time

        content = self._build_progress_content(
            self._step_count, self._step_count, running=False
        )
        content += (
            f"\n\n---\n"
            f"**状态:** {status} | "
            f"**耗时:** {elapsed:.1f}s | "
            f"**步数:** {self._step_count}"
        )
        if final_answer:
            content += f"\n\n**最终回答:**\n{final_answer[:3000]}"

        if status == "completed":
            template, title = "green", "✅ 任务完成"
        else:
            template, title = "red", f"❌ 任务{status}"

        self._patch(title=title, content=content, template=template)

        # 文档：追加总结
        self._finalize_document(status, elapsed, final_answer)

    # ------------------------------------------------------------------
    # Internal — Card
    # ------------------------------------------------------------------

    def _build_progress_content(
        self, current_step: int, max_steps: int, running: bool
    ) -> str:
        """构建卡片内容：任务信息 + 进度 + 文档链接。"""
        parts = [f"**任务:** {self._task_text}"]

        if self._document_url:
            parts.append(f"[📄 查看完整轨迹]({self._document_url})")

        parts.append("---")

        if running:
            elapsed = time.time() - self._start_time
            if current_step > 0:
                parts.append(f"> 正在执行 Step {current_step}/{max_steps}... ({elapsed:.0f}s)")
            else:
                parts.append("> 正在处理...")

        return "\n\n".join(parts)

    def _patch(self, title: str, content: str, template: str) -> None:
        """执行 PATCH 调用。"""
        from .sender import patch_card_message

        try:
            patch_card_message(
                self._client,
                self._card_message_id,
                title=title,
                content=content,
                header_template=template,
            )
        except Exception:
            logger.exception("Failed to patch card %s", self._card_message_id)

    # ------------------------------------------------------------------
    # Internal — Document
    # ------------------------------------------------------------------

    def _create_trajectory_document(self, task_text: str) -> None:
        """创建飞书文档用于存放完整轨迹。失败时静默降级。"""
        if not self._doc_writer:
            return

        try:
            doc_id = self._doc_writer.create_document(
                title=f"Agent Trajectory: {task_text[:100]}"
            )
            if not doc_id:
                return

            self._document_id = doc_id
            self._doc_writer.set_public_readable(doc_id)
            self._document_url = self._doc_writer.get_document_url(doc_id)

            # 转移文档所有权给发送消息的用户
            if self._sender_open_id:
                self._doc_writer.transfer_ownership(doc_id, self._sender_open_id)

            # 写入文档标题和任务描述
            self._doc_writer.append_heading(doc_id, f"Task: {task_text[:500]}", level=1)
            self._doc_writer.append_text(
                doc_id,
                f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self._doc_writer.append_divider(doc_id)
        except Exception:
            logger.exception("Failed to create trajectory document")

    def _append_step_to_document(
        self, step_record: Any, step_num: int, max_steps: int
    ) -> None:
        """向飞书文档追加完整步骤内容（无截断）。"""
        from .document_writer import (
            _build_code_block,
            _build_divider_block,
            _build_heading_block,
            _build_text_block,
        )

        blocks = []

        # Step heading
        blocks.append(_build_heading_block(f"Step {step_num}/{max_steps}", level=3))

        assistant_msg = getattr(step_record, "assistant_message", None)
        if assistant_msg is None:
            blocks.append(_build_text_block("(no assistant message)"))
            blocks.append(_build_divider_block())
            self._doc_writer.append_blocks(self._document_id, blocks)
            return

        # Thinking content (full)
        content = getattr(assistant_msg, "content", "") or ""
        if content.strip():
            blocks.append(_build_text_block("Thinking:", bold=True))
            blocks.append(_build_text_block(content))

        # Tool calls (full arguments)
        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "?")
            raw_args = getattr(func, "arguments", "")
            try:
                args_obj = json.loads(raw_args)
                args_str = json.dumps(args_obj, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                args_str = str(raw_args)
            blocks.append(_build_text_block(f"Tool Call: {name}", bold=True))
            blocks.append(_build_code_block(args_str, "json"))

        # Tool responses (full content)
        tool_responses = getattr(step_record, "tool_responses", None) or []
        for tr in tool_responses:
            tr_name = getattr(tr, "name", "?")
            tr_content = getattr(tr, "content", "") or ""
            blocks.append(_build_text_block(f"Result ({tr_name}):", bold=True))
            blocks.append(_build_code_block(tr_content))

        blocks.append(_build_divider_block())

        # 批量追加（单次 API 调用）
        self._doc_writer.append_blocks(self._document_id, blocks)

    def _finalize_document(
        self, status: str, elapsed: float, final_answer: str
    ) -> None:
        """向文档追加总结。"""
        if not self._doc_writer or not self._document_id:
            return

        try:
            self._doc_writer.append_divider(self._document_id)
            self._doc_writer.append_heading(self._document_id, "Summary", level=2)
            summary = (
                f"Status: {status}\n"
                f"Duration: {elapsed:.1f}s\n"
                f"Steps: {self._step_count}"
            )
            self._doc_writer.append_text(self._document_id, summary)

            if final_answer:
                self._doc_writer.append_heading(
                    self._document_id, "Final Answer", level=3
                )
                self._doc_writer.append_text(self._document_id, final_answer)
        except Exception:
            logger.exception("Failed to finalize trajectory document")
