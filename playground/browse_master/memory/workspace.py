"""Workspace assembly for the BrowseMaster MDP loop."""

from __future__ import annotations

from dataclasses import dataclass

from evomaster.utils.types import Dialog, SystemMessage, ToolSpec, UserMessage


INITIAL_REPORT = """## Current Most Likely Answer Direction
No answer direction yet.

## Confirmed Facts
- None yet.

## Hypotheses to Verify
- None yet.

## Excluded Directions
- None yet.

## Information Gaps
- Need to identify decisive clues from the original question.

## Next Step Priorities
1. Extract hard clues from the question and search for candidate entities or facts.
"""


@dataclass
class ImmediateContext:
    """The only raw interaction carried into the next MDP state."""

    action: str
    observation: str


@dataclass
class WorkspaceState:
    """Markovian state: question, evolving report, and latest interaction."""

    question: str
    report: str = INITIAL_REPORT
    immediate: ImmediateContext | None = None


def build_workspace_dialog(
    *,
    system_prompt: str,
    state: WorkspaceState,
    tools: list[ToolSpec],
) -> Dialog:
    """Build a bounded dialog from the current MDP state."""

    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=f"## Original Question\n\n{state.question}"),
        UserMessage(content=f"## Current Evolving Report\n\n{state.report}"),
    ]

    if state.immediate is not None:
        messages.append(
            UserMessage(
                content=(
                    "## Previous Action\n"
                    f"{state.immediate.action}\n\n"
                    "## Previous Result\n"
                    f"{state.immediate.observation}"
                )
            )
        )
    else:
        messages.append(
            UserMessage(
                content=(
                    "## Previous Action\n"
                    "[Initial state: no previous action]\n\n"
                    "## Previous Result\n"
                    "[Initial state: no previous result]"
                )
            )
        )

    messages.append(
        UserMessage(
            content=(
                "Generate the next MDP decision now. Put the complete six-section "
                "evolving report directly in assistant message content as Markdown "
                "without XML tags. Use the native think tool for private reasoning "
                "when available, then make one native action tool call: "
                "google_search, web_fetch, or finish."
            )
        )
    )
    return Dialog(messages=messages, tools=tools)
