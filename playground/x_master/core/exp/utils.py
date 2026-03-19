"""Utility functions for X-Master experiments."""

import json
from typing import Any


def strip_think_and_exec(text: str) -> str:
    """Remove trailing </think> and </execution_results> sections from text.

    Keep only the visible answer part and discard preceding internal reasoning or execution result tags.
    This is used to clean upstream agent outputs before passing them to downstream agents
    (e.g., Critic/Rewriter/Selector), ensuring downstream agents see the final answer and not
    intermediate reasoning steps.

    Args:
        text: the original text

    Returns:
        Cleaned text with only the visible answer portion.
    """
    if text is None:
        return ""
    out = text
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()


def extract_agent_response(trajectory: Any) -> str:
    """Extract the agent's final response from a trajectory object.

    Args:
        trajectory: execution trajectory object (expected to have a .dialogs list)

    Returns:
        The agent's response text if found, otherwise an empty string.
    """
    if not trajectory or not trajectory.dialogs:
        return ""

    # Get the last dialog
    last_dialog = trajectory.dialogs[-1]
    
    # Search for the last assistant message
    for message in reversed(last_dialog.messages):
        if hasattr(message, 'role') and message.role.value == 'assistant':
            # Standard assistant content
            if hasattr(message, 'content') and message.content:
                return message.content

            # Final answer might come as tool_calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if not hasattr(tool_call, 'function'):
                        continue
                    func = tool_call.function

                    if not hasattr(func, 'arguments'):
                        continue

                    args = func.arguments

                    if not args:
                        continue

                    # arguments might be a JSON string
                    try:
                        args_dict = json.loads(args)
                    except Exception:
                        continue

                    # Prefer the 'message' field if present
                    if "message" in args_dict and args_dict["message"]:
                        return args_dict["message"]
            
    return ""
