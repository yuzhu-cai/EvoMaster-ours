"""AskUser 工具 — 仅在交互式上下文中注入

允许 Agent 向用户提出结构化问题，暂停执行等待用户回答。
该工具不属于 builtin（非所有 agent 通用），由 dispatcher 按需注入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class AskUserToolParams(BaseToolParams):
    """Ask the user 1-2 clarification questions when critical information is missing.

    Use this tool when:
    - The user's request is ambiguous and you need to choose between fundamentally different approaches
    - Critical information is missing that would significantly change your design

    Do NOT use this tool when:
    - The request is clear enough to proceed with reasonable defaults
    - The missing details are minor and can be decided for the user

    Guidelines:
    - Ask at most 1-2 questions at a time
    - Each question should have 2-4 concise options
    - Only ask about decisions that fundamentally affect the architecture or approach
    - For minor details, make reasonable defaults and mention them in your plan
    """

    name: ClassVar[str] = "ask_user"

    questions: list[dict[str, Any]] = Field(
        description=(
            "List of questions to ask. Each question is an object with: "
            "'question' (the question text), "
            "'options' (list of objects with 'label' (1-5 words) and optional 'description')."
        ),
    )


class AskUserTool(BaseTool):
    """用户提问工具 — 调用后 agent 暂停执行，等待用户回答后继续"""

    name: ClassVar[str] = "ask_user"
    params_class: ClassVar[type[BaseToolParams]] = AskUserToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """解析问题参数（实际拦截在 agent.py 中，此方法作为 fallback）"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}

        assert isinstance(params, AskUserToolParams)

        self.logger.info(
            "ask_user called with %d question(s)", len(params.questions)
        )

        return "Questions sent to user. Waiting for response.", {
            "ask_user": True,
            "questions": params.questions,
        }
