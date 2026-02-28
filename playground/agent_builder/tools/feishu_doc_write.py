"""飞书文档写入工具

封装 FeishuDocumentWriter 为 Agent 可调用的 tool，
支持创建文档、追加标题/文本/代码/分割线等操作。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.interface.feishu.document_writer import FeishuDocumentWriter

logger = logging.getLogger(__name__)


class FeishuDocWriteToolParams(BaseToolParams):
    """Create or write content to a Feishu (Lark) document.

    Actions:
    - "create": Create a new document with the given title. Returns the document URL.
    - "append_heading": Append a heading (level 1-4) to the current document.
    - "append_text": Append a text paragraph to the current document.
    - "append_code": Append a code block to the current document.
    - "append_divider": Append a divider line to the current document.

    You must call "create" first before using any "append_*" actions.
    After creating and filling the document, share the URL with the user.
    """

    name: ClassVar[str] = "feishu_doc_write"

    action: Literal["create", "append_heading", "append_text", "append_code", "append_divider"] = Field(
        description='The action to perform: "create", "append_heading", "append_text", "append_code", or "append_divider"'
    )
    title: Optional[str] = Field(
        default=None,
        description='Document title (required for "create" action)'
    )
    content: Optional[str] = Field(
        default=None,
        description='Text content (required for "append_heading", "append_text", "append_code")'
    )
    level: Optional[int] = Field(
        default=2,
        description='Heading level 1-4 (only for "append_heading", default 2)'
    )
    language: Optional[str] = Field(
        default="plaintext",
        description='Code language for "append_code": "plaintext", "python", "json"'
    )


class FeishuDocWriteTool(BaseTool):
    """飞书文档写入工具

    在 agent 运行期间动态创建飞书文档并写入结构化内容。
    由 dispatcher 注入 FeishuDocumentWriter 实例。
    """

    name: ClassVar[str] = "feishu_doc_write"
    params_class: ClassVar[type[BaseToolParams]] = FeishuDocWriteToolParams

    def __init__(
        self,
        document_writer: FeishuDocumentWriter,
        sender_open_id: str | None = None,
    ):
        super().__init__()
        self._writer = document_writer
        self._sender_open_id = sender_open_id
        # 当前活跃文档（create 后设置）
        self._current_doc_id: str | None = None
        self._current_doc_url: str | None = None

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行文档写入操作"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, FeishuDocWriteToolParams)
        action = params.action

        try:
            if action == "create":
                return self._do_create(params)
            elif action == "append_heading":
                return self._do_append_heading(params)
            elif action == "append_text":
                return self._do_append_text(params)
            elif action == "append_code":
                return self._do_append_code(params)
            elif action == "append_divider":
                return self._do_append_divider()
            else:
                return f"Unknown action: {action}", {"error": "unknown_action"}
        except Exception as e:
            self.logger.error("feishu_doc_write failed: action=%s, error=%s", action, e)
            return f"Failed to execute {action}: {e}", {"error": str(e)}

    def _do_create(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """创建新文档"""
        title = params.title
        if not title:
            return "Title is required for 'create' action.", {"error": "missing_title"}

        doc_id = self._writer.create_document(title)
        if not doc_id:
            return "Failed to create document.", {"error": "create_failed"}

        # 设置公开可读
        self._writer.set_public_readable(doc_id)

        # 转移所有权给发送者
        if self._sender_open_id:
            self._writer.transfer_ownership(doc_id, self._sender_open_id)

        self._current_doc_id = doc_id
        self._current_doc_url = self._writer.get_document_url(doc_id)

        self.logger.info("Created document: %s -> %s", title, self._current_doc_url)
        return (
            f"Document created successfully.\n"
            f"URL: {self._current_doc_url}\n"
            f"Document ID: {doc_id}\n"
            f"You can now use append_* actions to add content.",
            {"document_id": doc_id, "url": self._current_doc_url},
        )

    def _require_doc(self) -> tuple[str, dict[str, Any]] | None:
        """检查是否已创建文档"""
        if not self._current_doc_id:
            return (
                "No active document. Please use 'create' action first.",
                {"error": "no_document"},
            )
        return None

    def _do_append_heading(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """追加标题"""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_heading'.", {"error": "missing_content"}

        level = max(1, min(4, params.level or 2))
        ok = self._writer.append_heading(self._current_doc_id, content, level=level)
        if not ok:
            return "Failed to append heading.", {"error": "append_failed"}
        return f"Heading (level {level}) appended.", {"action": "append_heading"}

    def _do_append_text(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """追加文本"""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_text'.", {"error": "missing_content"}

        ok = self._writer.append_text(self._current_doc_id, content)
        if not ok:
            return "Failed to append text.", {"error": "append_failed"}
        return "Text appended.", {"action": "append_text"}

    def _do_append_code(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """追加代码块"""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_code'.", {"error": "missing_content"}

        language = params.language or "plaintext"
        ok = self._writer.append_code_block(self._current_doc_id, content, language=language)
        if not ok:
            return "Failed to append code block.", {"error": "append_failed"}
        return f"Code block ({language}) appended.", {"action": "append_code"}

    def _do_append_divider(self) -> tuple[str, dict[str, Any]]:
        """追加分割线"""
        err = self._require_doc()
        if err:
            return err

        ok = self._writer.append_divider(self._current_doc_id)
        if not ok:
            return "Failed to append divider.", {"error": "append_failed"}
        return "Divider appended.", {"action": "append_divider"}
