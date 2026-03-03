"""飞书 API 通信层

封装飞书 API 交互：客户端创建、消息发送、文档操作。
"""

from .client import create_feishu_client
from .sender import (
    send_text_message,
    send_card_message,
    patch_card_message,
    build_card_with_actions,
    _build_card_json,
)
from .document import FeishuDocumentWriter

__all__ = [
    "create_feishu_client",
    "send_text_message",
    "send_card_message",
    "patch_card_message",
    "build_card_with_actions",
    "_build_card_json",
    "FeishuDocumentWriter",
]
