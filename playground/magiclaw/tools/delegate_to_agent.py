"""MagiClaw 委派工具

允许 magiclaw 将任务委派给专业 Agent（如 agent_builder）。
支持动态 agent 列表，由 dispatcher 在运行时注入。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

# 默认可委派 Agent（dispatcher 未注入时的 fallback）
_DEFAULT_AGENTS = {
    "agent_builder": "创建/设计/构建新的 AI Agent",
}


class DelegateToAgentParams(BaseToolParams):
    """Delegate a task to a specialized Agent.

    Available Agents:
    - agent_builder: Create/design/build new AI Agents. Use when user wants to create an agent.
      Examples: "Help me create an xxx agent", "I want to build a code review agent"

    Only delegate when the user explicitly needs specialized Agent capabilities.
    Handle normal conversation, search, and Q&A by yourself.
    """

    name: ClassVar[str] = "delegate_to_agent"

    agent_name: str = Field(
        description="委派目标 Agent 名称"
    )
    task: str = Field(
        description="任务描述，使用用户原始语言，包含完整上下文"
    )


class DelegateToAgentTool(BaseTool):
    """Delegation tool: forwards tasks to specialized Agents."""

    name: ClassVar[str] = "delegate_to_agent"
    params_class: ClassVar[type[BaseToolParams]] = DelegateToAgentParams

    def __init__(self, available_agents: dict[str, str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._available_agents: dict[str, str] = dict(available_agents or _DEFAULT_AGENTS)

    def set_available_agents(self, agents: dict[str, str]) -> None:
        """由 dispatcher 调用，注入运行时可用的 agent 列表。"""
        self._available_agents = dict(agents)

    def get_description(self) -> str:
        """动态生成工具描述，包含当前可用的 agent 列表。"""
        lines = ["将任务委派给专业 Agent 在后台执行。\n\n可用 Agent:"]
        for name, desc in self._available_agents.items():
            lines.append(f"- {name}: {desc}")
        lines.append(
            "\n只在用户明确需要专业 Agent 能力时委派。普通对话、搜索、问答自己处理。"
            "\n委派后任务将在后台运行，用户可继续与你对话。"
        )
        return "\n".join(lines)

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute delegation: validate agent name and store delegation info.

        The returned info dict contains a delegated=True marker.
        The dispatcher detects delegation by scanning the trajectory's ToolMessage.meta["info"].
        """
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数错误: {e}", {"error": str(e)}

        assert isinstance(params, DelegateToAgentParams)

        agent_name = params.agent_name
        task = params.task

        if agent_name not in self._available_agents:
            available = ", ".join(self._available_agents.keys())
            return (
                f"未知 Agent: '{agent_name}'。可用: {available}",
                {"error": "unknown_agent", "agent_name": agent_name},
            )

        self.logger.info("Delegation requested: agent=%s, task=%s", agent_name, task[:100])

        return (
            f"委派已接受。任务将由 '{agent_name}' 在后台处理。任务完成后系统会自动通知你审阅结果，请勿轮询 check_background_tasks，直接告知用户任务已启动即可结束本轮对话。",
            {"delegated": True, "agent_name": agent_name, "task": task},
        )
