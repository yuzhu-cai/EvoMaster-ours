"""跨 interface 共用的 Agent 工具

适用于所有交互式 interface（feishu、telegram 等），由 dispatcher 按需注入。
"""

from .ask_user import AskUserTool, AskUserToolParams

__all__ = [
    "AskUserTool",
    "AskUserToolParams",
]
