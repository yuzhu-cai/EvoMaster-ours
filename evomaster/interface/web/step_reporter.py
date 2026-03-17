"""Web SocketIO real-time step reporter.

Emits SocketIO events for each agent step, providing the same public API
as ``FeishuStepReporter`` but targeting a browser-based chat interface
instead of Feishu card messages.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class WebStepReporter:
    """Emits SocketIO events to report agent execution progress.

    Usage::

        reporter = WebStepReporter(socketio, room="session:abc", message_id="msg-1")
        reporter.send_initial_card("Calculate 3+5")
        playground.run(task_description=text, on_step=reporter.on_step)
        reporter.finalize("completed", "3+5=8")
    """

    _ARGS_PREVIEW_MAX = 200

    def __init__(self, socketio: Any, room: str, message_id: str) -> None:
        self._socketio = socketio
        self._room = room
        self._message_id = message_id

        self._task_text: str = ""
        self._start_time: float = 0.0
        self._step_count: int = 0

        # TODO progress checklist
        self._todo_items: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def card_message_id(self) -> str | None:
        """Return *message_id* for compatibility with the Feishu reporter."""
        return self._message_id

    def set_todo_items(self, items: list[str]) -> None:
        """Set TODO checklist items (for agent_builder progress tracking)."""
        self._todo_items = [{"label": item, "done": False} for item in items]

    def has_incomplete_todos(self) -> bool:
        """Return ``True`` if any TODO item is still pending."""
        if not self._todo_items:
            return False
        return any(not item["done"] for item in self._todo_items)

    def get_incomplete_todo_labels(self) -> list[str]:
        """Return labels of incomplete TODO items."""
        return [item["label"] for item in self._todo_items if not item["done"]]

    def send_initial_card(self, task_text: str) -> bool:
        """Emit a ``message_ack`` event to acknowledge the task."""
        self._task_text = task_text[:200]
        self._start_time = time.time()

        self._emit("message_ack", {
            "message_id": self._message_id,
            "status": "processing",
            "task_text": self._task_text,
        })
        return True

    def on_step(self, step_record: Any, step_number: int, max_steps: int) -> None:
        """Per-step callback: emit ``step_progress`` event."""
        self._step_count = step_number

        # Check TODO progress via think-tool PROGRESS markers
        self._check_todo_progress(step_record)

        elapsed = time.time() - self._start_time

        payload: dict[str, Any] = {
            "message_id": self._message_id,
            "step": step_number,
            "max_steps": max_steps,
            "elapsed": round(elapsed, 1),
            "step_text": f"Step {step_number}/{max_steps}",
            "thinking": self._extract_thinking(step_record),
        }

        # Frontend expects "tool_name" as a single string
        tool_calls = self._extract_tool_calls(step_record)
        if tool_calls:
            payload["tool_name"] = tool_calls[0]["name"]

        # Frontend expects "todo" as a summary string
        if self._todo_items:
            done = sum(1 for t in self._todo_items if t["done"])
            total = len(self._todo_items)
            payload["todo"] = f"Progress: {done}/{total}"

        self._emit("step_progress", payload)

    def finalize(
        self,
        status: str,
        final_answer: str = "",
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit ``agent_response`` event for task completion or failure."""
        elapsed = time.time() - self._start_time

        payload: dict[str, Any] = {
            "message_id": self._message_id,
            "status": status,
            "answer": final_answer,
            "steps": self._step_count,
            "elapsed": round(elapsed, 1),
        }

        if actions:
            payload["actions"] = actions

        if self._todo_items:
            payload["todos"] = self._serialize_todos()

        self._emit("agent_response", payload)

    def finalize_as_question(
        self,
        question_text: str,
        actions: list[dict[str, Any]] | None = None,
        options: list[dict[str, str]] | None = None,
        session_key: str = "",
        agent_name: str = "",
    ) -> None:
        """Emit ``question`` event when the agent needs user input."""
        elapsed = time.time() - self._start_time

        payload: dict[str, Any] = {
            "message_id": self._message_id,
            "status": "waiting_for_input",
            "question": question_text,
            "steps": self._step_count,
            "elapsed": round(elapsed, 1),
            "session_key": session_key,
            "agent_name": agent_name,
        }

        if options:
            payload["options"] = options
        if actions:
            payload["actions"] = actions

        self._emit("question", payload)

    # ------------------------------------------------------------------
    # Internal — SocketIO
    # ------------------------------------------------------------------

    def _serialize_todos(self) -> list[dict[str, Any]]:
        """Return a serializable snapshot of the TODO checklist."""
        return [{"label": t["label"], "done": t["done"]} for t in self._todo_items]

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Emit a SocketIO event to the room, logging failures."""
        try:
            self._socketio.emit(event, data, room=self._room)
        except Exception:
            logger.exception("Failed to emit '%s' to room %s", event, self._room)

    # ------------------------------------------------------------------
    # Internal — Step extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_tool_calls(cls, step_record: Any) -> list[dict[str, str]]:
        """Extract tool call summaries from a step record.

        Returns a list of ``{"name": str, "args_preview": str}`` dicts.
        """
        assistant_msg = getattr(step_record, "assistant_message", None)
        if assistant_msg is None:
            return []

        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        results: list[dict[str, str]] = []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "unknown")
            raw_args = getattr(func, "arguments", "")
            args_preview = cls._truncate_args(raw_args)
            results.append({"name": name, "args_preview": args_preview})
        return results

    @classmethod
    def _truncate_args(cls, raw_args: str) -> str:
        """Return a truncated, pretty-printed preview of tool arguments."""
        try:
            args_obj = json.loads(raw_args)
            formatted = json.dumps(args_obj, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted = str(raw_args)

        if len(formatted) > cls._ARGS_PREVIEW_MAX:
            return formatted[: cls._ARGS_PREVIEW_MAX] + "..."
        return formatted

    @staticmethod
    def _extract_thinking(step_record: Any) -> str | None:
        """Extract thinking/reasoning text from the assistant message.

        If the assistant message contains both text *content* and tool calls,
        the content is treated as thinking (the same convention used by the
        Feishu document writer).  Returns ``None`` when there is no thinking
        content to report.
        """
        assistant_msg = getattr(step_record, "assistant_message", None)
        if assistant_msg is None:
            return None

        content = getattr(assistant_msg, "content", None) or ""
        if isinstance(content, str) and content.strip():
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []
            # Content alongside tool calls is thinking; standalone content
            # is the final answer — only report the former.
            if tool_calls:
                return content.strip()
        return None

    # ------------------------------------------------------------------
    # Internal — TODO Progress
    # ------------------------------------------------------------------

    def _check_todo_progress(self, step_record: Any) -> None:
        """Detect PROGRESS markers in the ``think`` tool and update TODOs."""
        if not self._todo_items:
            return

        assistant_msg = getattr(step_record, "assistant_message", None)
        if not assistant_msg:
            return

        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func and getattr(func, "name", "") == "think":
                raw_args = getattr(func, "arguments", "")
                try:
                    args_obj = json.loads(raw_args)
                    thought = (
                        args_obj.get("thought", "")
                        or args_obj.get("content", "")
                        or ""
                    )
                except (json.JSONDecodeError, TypeError):
                    thought = str(raw_args)

                if "PROGRESS:" in thought and "[x]" in thought:
                    progress_text = thought.split("PROGRESS:", 1)[1].strip()
                    label = progress_text.replace("[x]", "").strip()
                    self._fuzzy_mark_done(label)

    def _fuzzy_mark_done(self, completed_label: str) -> None:
        """Fuzzy-match and mark a TODO item as done."""
        completed_lower = completed_label.lower()
        for item in self._todo_items:
            if not item["done"]:
                item_lower = item["label"].lower()
                if completed_lower in item_lower or item_lower in completed_lower:
                    item["done"] = True
                    break
