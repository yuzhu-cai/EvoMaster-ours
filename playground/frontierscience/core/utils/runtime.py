"""Runtime helpers for FrontierScience playground."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def enable_frontier_tools_for_agents(
    agents: Any,
    frontier_tool_names: list[str],
) -> None:
    """Attach custom FrontierScience tools to every configured agent."""
    for _, agent in agents.items():
        if agent is None:
            continue
        enabled_tool_names = list(agent.enabled_tool_names or [])
        for tool_name in frontier_tool_names:
            if tool_name not in enabled_tool_names:
                enabled_tool_names.append(tool_name)
        agent.enabled_tool_names = enabled_tool_names


def create_worker_agents(playground: Any, worker_index: int) -> dict[str, Any]:
    """Clone worker agents for one parallel branch."""
    return {
        "draft": playground.copy_agent(
            playground.agents["draft_agent"],
            new_agent_name=f"draft_worker_{worker_index}",
        ),
        "improve": playground.copy_agent(
            playground.agents["improve_agent"],
            new_agent_name=f"improve_worker_{worker_index}",
        ),
        "metric": playground.copy_agent(
            playground.agents["metric_agent"],
            new_agent_name=f"metric_worker_{worker_index}",
        ),
    }


def resolve_worker_workspace(worker_index: int, main_workspace: Path, parallel_config: dict[str, Any]) -> Path:
    """Choose a worker workspace, optionally splitting one subdirectory per branch."""
    if bool(parallel_config.get("split_workspace_for_exp", False)):
        return main_workspace / f"exp_{worker_index}"
    return main_workspace
