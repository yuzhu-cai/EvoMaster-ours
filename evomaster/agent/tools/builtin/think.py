"""EvoMaster Think 工具

提供思考和推理的能力，不会对环境产生影响。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from ..base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class ThinkToolParams(BaseToolParams):
    """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

    Common use cases:
    1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
    2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
    3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
    4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
    5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

    The tool simply logs your thought process for better transparency and does not execute any code or make changes.
    """
    
    name: ClassVar[str] = "think"

    thought: str = Field(description="The thought to log.")


class ThinkTool(BaseTool):
    """思考工具"""
    
    name: ClassVar[str] = "think"
    params_class: ClassVar[type[BaseToolParams]] = ThinkToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """记录思考内容（不执行任何操作）"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, ThinkToolParams)
        
        # Think 工具只记录，不执行任何操作
        self.logger.debug(f"Agent thought: {params.thought[:100]}...")
        
        return "Your thought has been logged.", {"thought": params.thought}

