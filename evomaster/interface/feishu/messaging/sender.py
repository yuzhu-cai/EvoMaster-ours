"""飞书消息发送

发送文本消息、卡片消息和回复消息到飞书。支持卡片消息的原地更新（PATCH）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

logger = logging.getLogger(__name__)

# 卡片消息内容上限（飞书 interactive 消息约 30KB，保守取 15KB）
_MAX_CARD_CONTENT_LENGTH = 15000


def _build_card_json(
    title: str,
    content: str,
    header_template: str = "blue",
) -> str:
    """构建卡片 JSON 字符串"""
    if len(content) > _MAX_CARD_CONTENT_LENGTH:
        content = content[:_MAX_CARD_CONTENT_LENGTH] + "\n\n...(内容过长已截断)"

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template,
        },
        "elements": [
            {"tag": "markdown", "content": content},
        ],
    })


def build_card_with_actions(
    title: str,
    content: str,
    actions: list[dict],
    header_template: str = "blue",
) -> str:
    """构建包含按钮的卡片 JSON 字符串

    Args:
        title: 卡片标题
        content: 卡片 Markdown 内容
        actions: 按钮列表，每项格式:
            {"text": "确认", "type": "primary", "value": {"action": "confirm"}}
            type 可选: "default", "primary", "danger"
        header_template: 卡片标题栏颜色模板

    Returns:
        卡片 JSON 字符串
    """
    if len(content) > _MAX_CARD_CONTENT_LENGTH:
        content = content[:_MAX_CARD_CONTENT_LENGTH] + "\n\n...(内容过长已截断)"

    elements: list[dict] = [{"tag": "markdown", "content": content}]

    if actions:
        action_items = []
        for a in actions:
            action_items.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": a["text"]},
                "type": a.get("type", "default"),
                "value": a.get("value", {}),
            })
        elements.append({"tag": "action", "actions": action_items})

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template,
        },
        "elements": elements,
    })


def send_text_message(
    client: lark.Client,
    chat_id: str,
    text: str,
    reply_to_message_id: str | None = None,
) -> bool:
    """发送或回复文本消息

    Args:
        client: 飞书 Client 实例
        chat_id: 聊天 ID
        text: 消息文本
        reply_to_message_id: 要回复的消息 ID（可选）

    Returns:
        True 表示发送成功
    """
    content = json.dumps({"text": text})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return False

        logger.debug("Message sent to chat %s", chat_id)
        return True

    except Exception:
        logger.exception("Error sending message to chat %s", chat_id)
        return False


def send_card_message(
    client: lark.Client,
    chat_id: str,
    title: str,
    content: str,
    reply_to_message_id: str | None = None,
    header_template: str = "blue",
    card_json: str | None = None,
) -> str | None:
    """发送卡片消息（支持 Markdown 格式，适合较长内容）

    Args:
        client: 飞书 Client 实例
        chat_id: 聊天 ID
        title: 卡片标题
        content: 卡片内容（支持飞书 Markdown 子集）
        reply_to_message_id: 要回复的消息 ID（可选）
        header_template: 卡片标题栏颜色模板
        card_json: 预构建的卡片 JSON（可选，传入时忽略 title/content/header_template）

    Returns:
        发送成功返回新消息的 message_id，失败返回 None
    """
    card = card_json or _build_card_json(title, content, header_template)

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(card)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(card)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send card message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("Card message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending card message to chat %s", chat_id)
        return None


def upload_image(
    client: lark.Client,
    image_path: str,
) -> str | None:
    """上传图片到飞书 IM，返回 image_key

    Args:
        client: 飞书 Client 实例
        image_path: 本地图片文件路径

    Returns:
        成功返回 image_key，失败返回 None
    """
    from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

    try:
        with open(image_path, "rb") as f:
            request = (
                CreateImageRequest.builder()
                .request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(f)
                    .build()
                )
                .build()
            )
            response = client.im.v1.image.create(request)

        if not response.success():
            logger.error(
                "Failed to upload image: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        image_key = response.data.image_key
        logger.debug("Image uploaded: %s -> %s", image_path, image_key)
        return image_key

    except Exception:
        logger.exception("Error uploading image %s", image_path)
        return None


def send_image_message(
    client: lark.Client,
    chat_id: str,
    image_key: str,
    reply_to_message_id: str | None = None,
) -> str | None:
    """发送图片消息

    Args:
        client: 飞书 Client 实例
        chat_id: 聊天 ID
        image_key: 已上传的图片 key（从 upload_image 获取）
        reply_to_message_id: 要回复的消息 ID（可选）

    Returns:
        发送成功返回 message_id，失败返回 None
    """
    content = json.dumps({"image_key": image_key})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("image")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("image")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send image message: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("Image message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending image message to chat %s", chat_id)
        return None


# 文件后缀 → 飞书 file_type 映射
_FILE_TYPE_MAP = {
    ".pdf": "pdf",
    ".doc": "doc", ".docx": "doc",
    ".xls": "xls", ".xlsx": "xls",
    ".ppt": "ppt", ".pptx": "ppt",
    ".mp4": "mp4",
    ".opus": "opus", ".ogg": "opus",
}


def upload_file(
    client: lark.Client,
    file_path: str,
) -> str | None:
    """上传文件到飞书 IM，返回 file_key

    Args:
        client: 飞书 Client 实例
        file_path: 本地文件路径

    Returns:
        成功返回 file_key，失败返回 None
    """
    from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

    p = Path(file_path)
    suffix = p.suffix.lower()
    file_type = _FILE_TYPE_MAP.get(suffix, "stream")
    file_name = p.name

    try:
        with open(file_path, "rb") as f:
            request = (
                CreateFileRequest.builder()
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_type(file_type)
                    .file_name(file_name)
                    .file(f)
                    .build()
                )
                .build()
            )
            response = client.im.v1.file.create(request)

        if not response.success():
            logger.error(
                "Failed to upload file: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        file_key = response.data.file_key
        logger.debug("File uploaded: %s -> %s (type=%s)", file_path, file_key, file_type)
        return file_key

    except Exception:
        logger.exception("Error uploading file %s", file_path)
        return None


def send_file_message(
    client: lark.Client,
    chat_id: str,
    file_key: str,
    reply_to_message_id: str | None = None,
) -> str | None:
    """发送文件消息

    Args:
        client: 飞书 Client 实例
        chat_id: 聊天 ID
        file_key: 已上传的文件 key（从 upload_file 获取）
        reply_to_message_id: 要回复的消息 ID（可选）

    Returns:
        发送成功返回 message_id，失败返回 None
    """
    content = json.dumps({"file_key": file_key})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("file")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("file")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send file message: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("File message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending file message to chat %s", chat_id)
        return None


def patch_card_message(
    client: lark.Client,
    message_id: str,
    title: str = "",
    content: str = "",
    header_template: str = "blue",
    card_json: str | None = None,
) -> bool:
    """原地更新已发送的卡片消息（PATCH API）

    仅支持更新 bot 自己发送的 interactive 类型消息。

    Args:
        client: 飞书 Client 实例
        message_id: 要更新的消息 ID
        title: 更新后的卡片标题
        content: 更新后的卡片内容
        header_template: 卡片标题栏颜色模板
        card_json: 预构建的卡片 JSON（可选，传入时忽略 title/content/header_template）

    Returns:
        True 表示更新成功
    """
    card = card_json or _build_card_json(title, content, header_template)

    try:
        request = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                PatchMessageRequestBody.builder()
                .content(card)
                .build()
            )
            .build()
        )
        response = client.im.v1.message.patch(request)

        if not response.success():
            logger.error(
                "Failed to patch card message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return False

        logger.debug("Card message patched: %s", message_id)
        return True

    except Exception:
        logger.exception("Error patching card message %s", message_id)
        return False
