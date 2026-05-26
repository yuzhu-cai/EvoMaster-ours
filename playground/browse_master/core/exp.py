"""BrowseMaster experiment workflow using an IterResearch-style MDP workspace."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import (
    Dialog,
    FunctionCall,
    StepRecord,
    TaskInstance,
    ToolCall,
    ToolMessage,
    Trajectory,
    UserMessage,
)

from ..memory import EvidenceLog, ImmediateContext, WorkspaceState, build_workspace_dialog


BROWSE_TOOL_NAMES = {"google_search", "web_fetch", "think", "finish"}
ACTION_TOOL_NAMES = {"google_search", "web_fetch", "finish"}
REQUIRED_REPORT_SECTIONS = (
    ("current most likely answer direction", "当前最可能的答案方向"),
    ("confirmed facts", "已确认的事实"),
    ("hypotheses to verify", "待验证的假设"),
    ("excluded directions", "已排除的方向"),
    ("information gaps", "信息缺口"),
    ("next step priorities", "下一步优先级"),
)


class BrowseMasterExp(BaseExp):
    """Single-agent benchmark web search experiment with Markovian state reset."""

    FORCE_FINAL_FALLBACK = "UNKNOWN"

    def __init__(self, agent, config):
        super().__init__(agent, config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ground_truth = None
        self.evidence_log = EvidenceLog()
        self._mdp_records: list[dict[str, Any]] = []
        self._parser_retries = 0
        self._last_report = ""

    @property
    def exp_name(self) -> str:
        return "BrowseMaster"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self._log_task_start(task_id, task_description)
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)

        task = self._build_task(
            task_id=task_id,
            task_description=task_description,
            images=images,
        )

        try:
            trajectory = self._run_mdp_task(task, on_step=on_step)
        except Exception as exc:
            self.logger.error("Browse task crashed: %s", exc, exc_info=True)
            result = self._build_exception_result(task_id, exc)
            self.results.append(result)
            return result

        forced_answer = ""
        fallback_used = False
        if not self._normalize_answer(self._extract_answer(trajectory)):
            forced_answer = self._best_effort_answer_from_report(self._last_report)
            self._append_forced_finish_step(trajectory, forced_answer, on_step=on_step)
            fallback_used = True

        result = self._build_result(
            task_id,
            trajectory,
            forced_answer=forced_answer,
            fallback_used=fallback_used,
        )
        self.results.append(result)
        self._log_task_end(result)
        return result

    def save_results(self, output_file: str):
        """Save experiment results including diagnostics."""
        output_data = [self._serialize_result(result) for result in self.results]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info("Results saved to %s", output_file)

    def _run_mdp_task(self, task: TaskInstance, on_step=None) -> Trajectory:
        trajectory = Trajectory(
            task_id=task.task_id,
            meta={
                "agent_version": getattr(self.agent, "VERSION", "unknown"),
                "task_type": task.task_type,
                "workspace_mode": "iterative_mdp",
            },
        )
        self.agent.trajectory = trajectory
        self.agent._step_count = 0

        system_prompt = self.agent._get_system_prompt()
        question_prompt = self.agent._get_user_prompt(task)
        state = WorkspaceState(question=question_prompt)
        self._last_report = state.report

        max_turns = getattr(getattr(self.agent, "config", None), "max_turns", 100)
        for turn in range(max_turns):
            self.logger.info("=" * 80)
            self.logger.info("MDP Step [%s/%s]", turn + 1, max_turns)
            self.logger.info("=" * 80)

            tools = self.agent._get_tool_specs()
            dialog = build_workspace_dialog(
                system_prompt=system_prompt,
                state=state,
                tools=tools,
            )
            assistant_message = self.agent.llm.query(dialog)
            assistant_message, prefix_messages, prefix_tool_messages, prefix_think = (
                self._resolve_think_only_response(dialog, assistant_message)
            )
            tool_calls = assistant_message.tool_calls or []
            selected_tool = self._select_action_tool(tool_calls)
            think_messages = [*prefix_tool_messages, *self._execute_think_tools(tool_calls)]
            report = self._extract_report(assistant_message.content)
            think = "\n\n".join(
                part for part in [prefix_think, self._extract_think_from_tools(tool_calls)] if part
            )

            if not report:
                self._parser_retries += 1
                report, think = self._repair_missing_report(
                    dialog=dialog,
                    assistant_message=assistant_message,
                    selected_tool=selected_tool,
                    fallback_report=state.report,
                    fallback_think=think,
                )
            elif not self._report_has_required_sections(report):
                self._parser_retries += 1
                self.logger.warning("Assistant report is missing one or more required sections")

            if selected_tool is None:
                self._parser_retries += 1
                observation = (
                    "Invalid MDP decision: no valid native tool call was made. "
                    "Make exactly one google_search, web_fetch, or finish call."
                )
                step_record = self._record_step(
                    trajectory=trajectory,
                    prompt_dialog=dialog,
                    prefix_messages=prefix_messages,
                    assistant_message=assistant_message,
                    tool_messages=think_messages,
                    on_step=on_step,
                    max_steps=max_turns,
                )
                state = WorkspaceState(
                    question=question_prompt,
                    report=report,
                    immediate=ImmediateContext(
                        action="invalid_response",
                        observation=observation,
                    ),
                )
                self._record_mdp_round(
                    turn=turn + 1,
                    report=report,
                    think=think,
                    action="invalid_response",
                    observation=observation,
                    step=step_record,
                )
                self._last_report = report
                continue

            tool_name = selected_tool.function.name
            if tool_name == "finish":
                tool_message = self._finish_tool_message(selected_tool)
                step_record = self._record_step(
                    trajectory=trajectory,
                    prompt_dialog=dialog,
                    prefix_messages=prefix_messages,
                    assistant_message=assistant_message,
                    tool_messages=[*think_messages, tool_message],
                    on_step=on_step,
                    max_steps=max_turns,
                )
                trajectory.finish("completed", {"workspace_mode": "iterative_mdp"})
                self._last_report = report
                self._record_mdp_round(
                    turn=turn + 1,
                    report=report,
                    think=think,
                    action=self._format_tool_call(selected_tool),
                    observation=tool_message.content or "",
                    step=step_record,
                )
                self._save_mdp_artifacts()
                return trajectory

            observation, info = self.agent._execute_tool(selected_tool)
            evidence = self.evidence_log.add(
                tool_name=tool_name,
                arguments=selected_tool.function.arguments,
                content=observation,
                meta=info,
            )
            evidence_observation = (
                f"[{evidence.evidence_id}] Tool result from {tool_name}:\n"
                f"{observation}"
            )
            tool_message = ToolMessage(
                content=evidence_observation,
                tool_call_id=selected_tool.id,
                name=tool_name,
                meta={"info": info, "evidence_id": evidence.evidence_id},
            )
            step_record = self._record_step(
                trajectory=trajectory,
                prompt_dialog=dialog,
                prefix_messages=prefix_messages,
                assistant_message=assistant_message,
                tool_messages=[*think_messages, tool_message],
                on_step=on_step,
                max_steps=max_turns,
            )

            action = self._format_tool_call(selected_tool)
            state = WorkspaceState(
                question=question_prompt,
                report=report,
                immediate=ImmediateContext(
                    action=action,
                    observation=evidence_observation,
                ),
            )
            self._last_report = report
            self._record_mdp_round(
                turn=turn + 1,
                report=report,
                think=think,
                action=action,
                observation=evidence_observation,
                step=step_record,
                evidence_id=evidence.evidence_id,
            )
            self._save_mdp_artifacts()

        trajectory.finish(
            "failed",
            {
                "reason": "max_turns_exceeded",
                "workspace_mode": "iterative_mdp",
                "last_report": self._last_report,
            },
        )
        return trajectory

    def _record_step(
        self,
        *,
        trajectory: Trajectory,
        prompt_dialog: Dialog,
        prefix_messages: list | None = None,
        assistant_message,
        tool_messages: list[ToolMessage],
        on_step=None,
        max_steps: int,
    ) -> StepRecord:
        self.agent._step_count = len(trajectory.steps) + 1
        response_dialog = prompt_dialog.model_copy(deep=True)
        for message in prefix_messages or []:
            response_dialog.add_message(message)
        response_dialog.add_message(assistant_message)
        for tool_message in tool_messages:
            response_dialog.add_message(tool_message)
        self.agent.current_dialog = response_dialog
        trajectory.dialogs.append(response_dialog)

        step_record = StepRecord(
            step_id=self.agent._step_count,
            assistant_message=assistant_message,
            tool_responses=tool_messages,
        )
        trajectory.add_step(step_record)
        self.agent._append_trajectory_entry(prompt_dialog, step_record)

        if on_step:
            try:
                on_step(step_record, self.agent._step_count, max_steps)
            except Exception as exc:
                self.logger.warning("on_step callback failed: %s", exc)
        return step_record

    def _append_forced_finish_step(
        self,
        trajectory: Trajectory,
        answer: str,
        on_step=None,
    ) -> None:
        answer = self._normalize_answer(answer) or self.FORCE_FINAL_FALLBACK
        finish_call = ToolCall(
            id="forced_finish",
            type="function",
            function=FunctionCall(
                name="finish",
                arguments=json.dumps(
                    {"message": answer, "task_completed": "true"},
                    ensure_ascii=False,
                ),
            ),
        )
        from evomaster.utils.types import AssistantMessage

        assistant_message = AssistantMessage(
            content=(
                "## Current Most Likely Answer Direction\n"
                f"{answer}\n\n"
                f"{self._last_report}"
            ),
            tool_calls=[finish_call],
        )
        state = WorkspaceState(question="Final forced answer", report=self._last_report)
        dialog = build_workspace_dialog(
            system_prompt="Forced final answer.",
            state=state,
            tools=[],
        )
        tool_message = self._finish_tool_message(finish_call)
        self._record_step(
            trajectory=trajectory,
            prompt_dialog=dialog,
            prefix_messages=None,
            assistant_message=assistant_message,
            tool_messages=[tool_message],
            on_step=on_step,
            max_steps=len(trajectory.steps) + 1,
        )
        trajectory.finish(
            "completed",
            {
                "forced_final_answer": True,
                "forced_final_message": answer,
                "status_before_forced_final": "failed",
                "workspace_mode": "iterative_mdp",
            },
        )
        self._save_mdp_artifacts()

    def _select_action_tool(self, tool_calls: list[ToolCall]) -> ToolCall | None:
        if not tool_calls:
            return None

        valid = [tc for tc in tool_calls if tc.function.name in ACTION_TOOL_NAMES]
        if not valid:
            names = [tc.function.name for tc in tool_calls]
            if all(name == "think" for name in names):
                self.logger.info("Model emitted think-only tool call; requesting action continuation")
            else:
                self.logger.warning("No valid browse action in tool calls: %s", names)
            return None

        if len(valid) > 1:
            self.logger.warning(
                "Multiple valid tool calls were emitted; executing only the first: %s",
                [tc.function.name for tc in valid],
            )
        leading_calls = tool_calls[: tool_calls.index(valid[0])]
        if leading_calls and any(tc.function.name != "think" for tc in leading_calls):
            self.logger.warning(
                "Ignoring leading non-browse tool calls before %s", valid[0].function.name
            )
        return valid[0]

    def _resolve_think_only_response(
        self,
        dialog: Dialog,
        assistant_message,
        *,
        max_continuations: int = 2,
    ) -> tuple[Any, list, list[ToolMessage], str]:
        """Continue immediately when the provider allows only a think call first."""
        prefix_messages: list = []
        prefix_tool_messages: list[ToolMessage] = []
        thoughts: list[str] = []
        current = assistant_message

        for _ in range(max_continuations):
            tool_calls = current.tool_calls or []
            if self._select_action_tool(tool_calls) is not None:
                return current, prefix_messages, prefix_tool_messages, "\n\n".join(thoughts)
            if not tool_calls or any(tc.function.name != "think" for tc in tool_calls):
                return current, prefix_messages, prefix_tool_messages, "\n\n".join(thoughts)

            think_messages = self._execute_think_tools(tool_calls)
            thoughts_text = self._extract_think_from_tools(tool_calls)
            if thoughts_text:
                thoughts.append(thoughts_text)

            followup_dialog = dialog.model_copy(deep=True)
            for message in prefix_messages:
                followup_dialog.add_message(message)
            followup_dialog.add_message(current)
            for tool_message in think_messages:
                followup_dialog.add_message(tool_message)
            followup_dialog.add_message(
                UserMessage(
                    content=(
                        "You have completed the think step. Now write the complete "
                        "six-section Markdown evolving report in assistant content "
                        "and make exactly one native action tool call: google_search, "
                        "web_fetch, or finish. Do not call think alone again."
                    )
                )
            )
            prefix_messages.extend([current, *think_messages])
            prefix_tool_messages.extend(think_messages)
            current = self.agent.llm.query(followup_dialog)

        return current, prefix_messages, prefix_tool_messages, "\n\n".join(thoughts)

    def _execute_think_tools(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        """Execute native think calls before the external MDP action."""
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            if tool_call.function.name != "think":
                continue
            observation, info = self.agent._execute_tool(tool_call)
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call.id,
                    name="think",
                    meta={"info": info},
                )
            )
        return tool_messages

    @staticmethod
    def _extract_think_from_tools(tool_calls: list[ToolCall]) -> str:
        thoughts: list[str] = []
        for tool_call in tool_calls:
            if tool_call.function.name != "think":
                continue
            try:
                payload = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                payload = {}
            thought = payload.get("thought", "")
            if isinstance(thought, str) and thought.strip():
                thoughts.append(thought.strip())
        return "\n\n".join(thoughts)

    def _repair_missing_report(
        self,
        *,
        dialog: Dialog,
        assistant_message,
        selected_tool: ToolCall | None,
        fallback_report: str,
        fallback_think: str,
    ) -> tuple[str, str]:
        """Ask for a report-only repair when the model emits only a tool call.

        Some function-calling providers return an empty assistant content whenever
        a native tool call is present. The MDP loop still needs the report for the
        next state, so we run a small no-tools repair pass and merge its content
        into the recorded assistant message.
        """
        if selected_tool is None:
            self.logger.info("Assistant response did not contain report content or a valid action")
            return fallback_report, fallback_think

        self.logger.info(
            "Assistant emitted %s without report content; requesting report-only repair",
            selected_tool.function.name,
        )
        repair_dialog = dialog.model_copy(deep=True)
        repair_dialog.tools = []
        repair_dialog.add_message(
            UserMessage(
                content=(
                    "Your previous response selected this native tool call but did not "
                    "include the required MDP report in assistant content.\n\n"
                    f"Selected action: {self._format_tool_call(selected_tool)}\n\n"
                    "Do not call any tool now. Output only the complete six-section "
                    "Markdown evolving report, without XML tags. The report must "
                    "preserve all important information needed for the next Markovian state."
                )
            )
        )

        try:
            repair_message = self.agent.llm.query(repair_dialog)
        except Exception as exc:
            self.logger.warning("Report repair call failed: %s", exc, exc_info=True)
            return fallback_report, fallback_think

        repaired_report = self._extract_report(repair_message.content)
        repaired_think = self._extract_tag(repair_message.content, "think")
        if not repaired_report:
            self.logger.warning("Report repair response still did not contain a usable report")
            return fallback_report, fallback_think

        original_content = assistant_message.content or ""
        assistant_message.content = (
            f"{original_content}\n\n{repair_message.content}"
            if original_content.strip()
            else repair_message.content
        )
        meta = getattr(assistant_message, "meta", None)
        if isinstance(meta, dict):
            meta["report_repaired"] = True
        return repaired_report, repaired_think or fallback_think

    def _finish_tool_message(self, tool_call: ToolCall) -> ToolMessage:
        try:
            payload = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            payload = {}
        message = payload.get("message", "")
        task_completed = payload.get("task_completed", "true")
        self.logger.info("=" * 80)
        self.logger.info("Finish Tool Arguments:")
        self.logger.info("  message: %s", message)
        self.logger.info("  task_completed: %s", task_completed)
        self.logger.info("=" * 80)
        return ToolMessage(
            content="Task marked as finished.",
            tool_call_id=tool_call.id,
            name="finish",
            meta={"info": {"task_completed": task_completed, "message": message}},
        )

    def _record_mdp_round(
        self,
        *,
        turn: int,
        report: str,
        think: str,
        action: str,
        observation: str,
        step: StepRecord,
        evidence_id: str | None = None,
    ) -> None:
        self._mdp_records.append(
            {
                "turn": turn,
                "step_id": step.step_id,
                "think": think,
                "report": report,
                "action": action,
                "observation": observation,
                "evidence_id": evidence_id,
            }
        )

    def _save_mdp_artifacts(self) -> None:
        if self.run_dir is None:
            return
        run_dir = Path(self.run_dir)
        self.evidence_log.save(run_dir / "evidence_log.json")
        with (run_dir / "mdp_trajectory.json").open("w", encoding="utf-8") as handle:
            json.dump(self._mdp_records, handle, indent=2, ensure_ascii=False)

    def _build_task(
        self,
        task_id: str,
        task_description: str,
        images: list[str] | None = None,
    ) -> TaskInstance:
        return TaskInstance(
            task_id=task_id,
            task_type="search",
            description=task_description,
            input_data={"benchmark": "browsecomp"},
            images=images or [],
        )

    def _build_result(
        self,
        task_id: str,
        trajectory,
        forced_answer: str = "",
        fallback_used: bool = False,
    ) -> dict[str, Any]:
        agent_answer = self._normalize_answer(self._extract_answer(trajectory))
        if not agent_answer:
            agent_answer = self._normalize_answer(forced_answer) or self.FORCE_FINAL_FALLBACK

        analysis = self._analyze_trajectory(trajectory, agent_answer)
        if fallback_used:
            analysis["forced_final_answer"] = True
            if analysis.get("answer_source") == "none":
                analysis["answer_source"] = "forced_final"

        status = trajectory.status
        if fallback_used and agent_answer:
            status = "completed"
            trajectory.status = "completed"
            result_meta = getattr(trajectory, "result", {}) or {}
            if isinstance(result_meta, dict):
                result_meta["forced_final_answer"] = True
                result_meta["forced_final_message"] = agent_answer
                trajectory.result = result_meta

        return {
            "task_id": task_id,
            "status": status,
            "steps": len(trajectory.steps),
            "trajectory": trajectory,
            "agent_answer": agent_answer,
            "final_answer": agent_answer,
            "ground_truth": self.ground_truth,
            "analysis": analysis,
        }

    def _build_exception_result(self, task_id: str, exc: Exception) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "status": "failed",
            "steps": 0,
            "trajectory": None,
            "agent_answer": self.FORCE_FINAL_FALLBACK,
            "final_answer": self.FORCE_FINAL_FALLBACK,
            "ground_truth": self.ground_truth,
            "analysis": {
                "answer_found": True,
                "answer_source": "forced_exception_fallback",
                "finish_called": False,
                "failure_reason": str(exc),
                "forced_final_answer": True,
                "tool_call_counts": {},
                "browse_tool_counts": {},
                "non_browse_tool_counts": {},
                "stagnation_steps": 0,
                "last_tool_names": [],
                "mdp_rounds": len(self._mdp_records),
                "report_chars_last": len(self._last_report),
                "parser_retries": self._parser_retries,
                "evidence_count": len(self.evidence_log.entries),
            },
            "error": str(exc),
        }

    def _analyze_trajectory(self, trajectory, agent_answer: str) -> dict[str, Any]:
        tool_call_counts: Counter[str] = Counter()
        browse_tool_counts: Counter[str] = Counter()
        non_browse_tool_counts: Counter[str] = Counter()
        finish_called = False
        stagnation_steps = 0
        last_tool_names: list[str] = []

        for step in trajectory.steps:
            tool_names: list[str] = []
            assistant_message = step.assistant_message
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_names.append(tool_name)
                tool_call_counts[tool_name] += 1
                if tool_name in BROWSE_TOOL_NAMES:
                    browse_tool_counts[tool_name] += 1
                else:
                    non_browse_tool_counts[tool_name] += 1
                if tool_name == "finish":
                    finish_called = True

            if tool_names:
                last_tool_names = tool_names

            if self._step_has_stagnation_signal(step):
                stagnation_steps += 1

        answer_source = "none"
        if finish_called and agent_answer:
            answer_source = "finish"
        elif agent_answer:
            answer_source = "assistant_text"

        failure_reason = self._extract_failure_reason(trajectory)
        result_meta = getattr(trajectory, "result", {}) or {}

        return {
            "answer_found": bool(agent_answer),
            "answer_source": answer_source,
            "finish_called": finish_called,
            "failure_reason": failure_reason,
            "tool_call_counts": dict(tool_call_counts),
            "browse_tool_counts": dict(browse_tool_counts),
            "non_browse_tool_counts": dict(non_browse_tool_counts),
            "stagnation_steps": stagnation_steps,
            "last_tool_names": last_tool_names,
            "mdp_rounds": len(self._mdp_records),
            "report_chars_last": len(self._last_report),
            "parser_retries": self._parser_retries,
            "evidence_count": len(self.evidence_log.entries),
            "forced_final_answer": bool(
                isinstance(result_meta, dict) and result_meta.get("forced_final_answer")
            ),
        }

    def _extract_failure_reason(self, trajectory) -> str:
        result = getattr(trajectory, "result", {}) or {}
        if isinstance(result, dict):
            reason = result.get("reason")
            if isinstance(reason, str):
                return reason
        return ""

    def _step_has_stagnation_signal(self, step) -> bool:
        for tool_message in step.tool_responses:
            info = getattr(tool_message, "meta", {}).get("info", {})
            if not isinstance(info, dict):
                continue
            guard = info.get("guard", {})
            if isinstance(guard, dict) and guard.get("stagnation"):
                return True
        return False

    def _extract_answer(self, trajectory) -> str:
        answer = self._extract_agent_response(trajectory)
        if answer:
            return answer

        for step in reversed(getattr(trajectory, "steps", [])):
            assistant_message = step.assistant_message
            if assistant_message is None:
                continue

            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if tool_call.function.name != "finish":
                    continue
                try:
                    payload = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    payload = {}
                finish_message = payload.get("message", "")
                if isinstance(finish_message, str) and finish_message.strip():
                    return finish_message

        return ""

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        trajectory = result.get("trajectory")
        trajectory_dump = trajectory.model_dump() if trajectory is not None else None

        payload = {
            "task_id": result["task_id"],
            "status": result["status"],
            "steps": result["steps"],
            "agent_answer": result.get("agent_answer", ""),
            "final_answer": result.get("final_answer", result.get("agent_answer", "")),
            "ground_truth": result.get("ground_truth"),
            "analysis": result.get("analysis", {}),
            "trajectory": trajectory_dump,
        }
        if "error" in result:
            payload["error"] = result["error"]
        return payload

    def _log_task_start(self, task_id: str, task_description: str) -> None:
        self.logger.info("=" * 80)
        self.logger.info("Browse task start: %s", task_id)
        self.logger.info("Question: %s", task_description)
        self.logger.info("=" * 80)

    def _log_task_end(self, result: dict[str, Any]) -> None:
        analysis = result.get("analysis", {})
        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info(
            "Browse task end: status=%s steps=%s finish_called=%s answer_source=%s "
            "stagnation_steps=%s mdp_rounds=%s evidence_count=%s parser_retries=%s",
            result.get("status"),
            result.get("steps"),
            analysis.get("finish_called"),
            analysis.get("answer_source"),
            analysis.get("stagnation_steps"),
            analysis.get("mdp_rounds"),
            analysis.get("evidence_count"),
            analysis.get("parser_retries"),
        )
        if analysis.get("failure_reason"):
            self.logger.info("Failure reason: %s", analysis["failure_reason"])
        if analysis.get("non_browse_tool_counts"):
            self.logger.warning(
                "Non-browse tools were used during benchmark solving: %s",
                analysis["non_browse_tool_counts"],
            )
        self.logger.info("Tool call counts: %s", analysis.get("tool_call_counts", {}))
        self.logger.info("=" * 80)

    def _extract_report(self, content: str | None) -> str:
        if not content:
            return ""

        tagged_report = self._extract_tag(content, "report")
        if tagged_report:
            return tagged_report

        # New BrowseMaster prompt treats normal assistant text as the report.
        plain = re.sub(
            r"<think>\s*.*?\s*</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        return plain

    @staticmethod
    def _extract_tag(content: str | None, tag: str) -> str:
        if not content:
            return ""
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _report_has_required_sections(report: str) -> bool:
        normalized = report.lower()
        return all(
            any(section.lower() in normalized for section in alternatives)
            for alternatives in REQUIRED_REPORT_SECTIONS
        )

    @staticmethod
    def _format_tool_call(tool_call: ToolCall) -> str:
        return f"{tool_call.function.name}({tool_call.function.arguments})"

    def _best_effort_answer_from_report(self, report: str) -> str:
        if not report.strip():
            return self.FORCE_FINAL_FALLBACK
        match = re.search(
            r"##\s*(?:Current Most Likely Answer Direction|当前最可能的答案方向)\s*(.*?)(?:\n##|\Z)",
            report,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            answer = match.group(1).strip()
            if answer and "no answer direction" not in answer.lower():
                return answer
        return self.FORCE_FINAL_FALLBACK

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return answer.strip()
