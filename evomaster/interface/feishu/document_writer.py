"""飞书文档写入器

封装 Feishu docx.v1 API，提供创建文档、追加内容块的简洁接口。
用于将 Agent 完整轨迹写入飞书文档（无截断）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lark_oapi as lark

from lark_oapi.api.docx.v1 import (
    Block,
    CreateDocumentBlockChildrenRequest,
    CreateDocumentBlockChildrenRequestBody,
    CreateDocumentRequest,
    CreateDocumentRequestBody,
    Divider,
    Text,
    TextElement,
    TextElementStyle,
    TextRun,
    TextStyle,
)
from lark_oapi.api.drive.v1 import (
    Owner,
    PatchPermissionPublicRequest,
    PermissionPublic,
    PermissionPublicRequest,
    TransferOwnerPermissionMemberRequest,
)

logger = logging.getLogger(__name__)

# Block type constants (Feishu docx.v1)
_BT_TEXT = 2
_BT_HEADING1 = 3
_BT_HEADING2 = 4
_BT_HEADING3 = 5
_BT_HEADING4 = 6
_BT_CODE = 14
_BT_DIVIDER = 22

# Code language: 1 = PlainText, 49 = JSON, 15 = Python
_LANG_PLAIN = 1
_LANG_JSON = 49
_LANG_PYTHON = 15

# Max chars per code block (Feishu limit is ~64KB per block)
_MAX_CODE_BLOCK_CHARS = 30000

_LANG_MAP = {
    "plaintext": _LANG_PLAIN,
    "json": _LANG_JSON,
    "python": _LANG_PYTHON,
}


class FeishuDocumentWriter:
    """飞书文档写入器：创建文档并追加内容块"""

    def __init__(
        self,
        client: lark.Client,
        folder_token: str | None = None,
        domain: str = "https://open.feishu.cn",
    ):
        self._client = client
        self._folder_token = folder_token
        # Extract base domain for URL construction
        # "https://open.feishu.cn" -> "feishu.cn"
        # "https://open.larksuite.com" -> "larksuite.com"
        host = domain.replace("https://", "").replace("http://", "")
        host = host.removeprefix("open.")
        self._base_domain = host

    def create_document(self, title: str) -> str | None:
        """创建飞书文档

        Returns:
            document_id on success, None on failure
        """
        body_builder = CreateDocumentRequestBody.builder().title(title[:800])
        if self._folder_token:
            body_builder = body_builder.folder_token(self._folder_token)

        request = (
            CreateDocumentRequest.builder()
            .request_body(body_builder.build())
            .build()
        )

        try:
            response = self._client.docx.v1.document.create(request)
            if not response.success():
                logger.warning(
                    "Failed to create document: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return None
            doc_id = response.data.document.document_id
            logger.info("Created Feishu document: %s", doc_id)
            return doc_id
        except Exception:
            logger.exception("Exception creating document")
            return None

    def set_public_readable(self, document_id: str) -> bool:
        """设置文档链接可读权限"""
        request = (
            PatchPermissionPublicRequest.builder()
            .token(document_id)
            .type("docx")
            .request_body(
                PermissionPublicRequest.builder()
                .link_share_entity("anyone_readable")
                .external_access(True)
                .build()
            )
            .build()
        )

        try:
            response = self._client.drive.v1.permission_public.patch(request)
            if not response.success():
                logger.warning(
                    "Failed to set doc permission: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception("Exception setting document permission")
            return False

    def get_document_url(self, document_id: str) -> str:
        """构建文档 URL"""
        return f"https://{self._base_domain}/docx/{document_id}"

    def transfer_ownership(self, document_id: str, user_open_id: str) -> bool:
        """将文档所有权转移给指定用户

        转移后用户可以自行管理（编辑、删除）该文档。

        Args:
            document_id: 文档 ID
            user_open_id: 目标用户的 open_id
        """
        request = (
            TransferOwnerPermissionMemberRequest.builder()
            .token(document_id)
            .type("docx")
            .need_notification(False)
            .request_body(
                Owner.builder()
                .member_type("openid")
                .member_id(user_open_id)
                .build()
            )
            .build()
        )

        try:
            response = self._client.drive.v1.permission_member.transfer_owner(request)
            if not response.success():
                logger.warning(
                    "Failed to transfer doc ownership: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            logger.info(
                "Transferred doc %s ownership to %s", document_id, user_open_id
            )
            return True
        except Exception:
            logger.exception("Exception transferring document ownership")
            return False

    def append_blocks(self, document_id: str, blocks: list[Block]) -> bool:
        """批量追加 blocks 到文档末尾"""
        if not blocks:
            return True

        request = (
            CreateDocumentBlockChildrenRequest.builder()
            .document_id(document_id)
            .block_id(document_id)  # root block = document_id
            .request_body(
                CreateDocumentBlockChildrenRequestBody.builder()
                .children(blocks)
                .build()
            )
            .build()
        )

        try:
            response = self._client.docx.v1.document_block_children.create(request)
            if not response.success():
                logger.warning(
                    "Failed to append blocks: code=%s, msg=%s",
                    response.code, response.msg,
                )
                return False
            return True
        except Exception:
            logger.exception("Exception appending blocks to document %s", document_id)
            return False

    # ---- Convenience methods ----

    def append_heading(
        self, document_id: str, text: str, level: int = 3
    ) -> bool:
        """追加标题块 (level 1-9)"""
        block = _build_heading_block(text, level)
        return self.append_blocks(document_id, [block])

    def append_text(
        self, document_id: str, text: str, bold: bool = False
    ) -> bool:
        """追加文本段落"""
        block = _build_text_block(text, bold=bold)
        return self.append_blocks(document_id, [block])

    def append_code_block(
        self, document_id: str, code: str, language: str = "plaintext"
    ) -> bool:
        """追加代码块"""
        if len(code) > _MAX_CODE_BLOCK_CHARS:
            code = code[:_MAX_CODE_BLOCK_CHARS] + "\n... (content truncated)"
        block = _build_code_block(code, language)
        return self.append_blocks(document_id, [block])

    def append_divider(self, document_id: str) -> bool:
        """追加分割线"""
        block = _build_divider_block()
        return self.append_blocks(document_id, [block])


# ---- Block builder helpers ----

def _build_text_run(content: str, bold: bool = False) -> TextElement:
    """构建 TextElement (TextRun)"""
    style_builder = TextElementStyle.builder()
    if bold:
        style_builder = style_builder.bold(True)

    return (
        TextElement.builder()
        .text_run(
            TextRun.builder()
            .content(content)
            .text_element_style(style_builder.build())
            .build()
        )
        .build()
    )


def _build_text_block(content: str, bold: bool = False) -> Block:
    """构建文本段落 Block"""
    return (
        Block.builder()
        .block_type(_BT_TEXT)
        .text(
            Text.builder()
            .elements([_build_text_run(content, bold=bold)])
            .build()
        )
        .build()
    )


def _build_heading_block(content: str, level: int = 3) -> Block:
    """构建标题 Block (level 1-9)"""
    level = max(1, min(9, level))
    block_type = _BT_HEADING1 + level - 1  # heading1=3, heading2=4, ...

    heading_text = (
        Text.builder()
        .elements([_build_text_run(content)])
        .build()
    )

    builder = Block.builder().block_type(block_type)
    # Map level to the correct heading method
    heading_setter = getattr(builder, f"heading{level}", None)
    if heading_setter:
        builder = heading_setter(heading_text)
    else:
        builder = builder.heading3(heading_text)

    return builder.build()


def _build_code_block(code: str, language: str = "plaintext") -> Block:
    """构建代码块 Block"""
    lang_id = _LANG_MAP.get(language, _LANG_PLAIN)

    return (
        Block.builder()
        .block_type(_BT_CODE)
        .code(
            Text.builder()
            .elements([_build_text_run(code)])
            .style(TextStyle.builder().language(lang_id).build())
            .build()
        )
        .build()
    )


def _build_divider_block() -> Block:
    """构建分割线 Block"""
    return Block.builder().block_type(_BT_DIVIDER).divider(Divider.builder().build()).build()
