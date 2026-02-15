"""PHY Master experiment with MCTS-style exploration."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from .visualization import write_mcts_html


@dataclass
class SearchNode:
    """A node in the MCTS-like subtask tree."""

    node_id: str
    subtask: str
    parent_id: str | None
    depth: int
    created_by: str = "clarifier"
    visits: int = 0
    value_sum: float = 0.0
    score: float | None = None
    reward: float = 0.0
    status: str = "open"
    theoretician_output: str = ""
    critic_feedback: dict[str, Any] = field(default_factory=dict)
    supervisor_feedback: dict[str, Any] = field(default_factory=dict)
    selected_round: int | None = None
    children: list[str] = field(default_factory=list)

    @property
    def average_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class PhyMasterExp(BaseExp):
    """Coordinate Clarifier/Supervisor/Theoretician/Critic/Summarizer agents."""

    def __init__(
        self,
        clarifier_agent,
        supervisor_agent,
        theoretician_agent,
        critic_agent,
        summarizer_agent,
        config,
    ):
        super().__init__(clarifier_agent, config)
        self.clarifier_agent = clarifier_agent
        self.supervisor_agent = supervisor_agent
        self.theoretician_agent = theoretician_agent
        self.critic_agent = critic_agent
        self.summarizer_agent = summarizer_agent
        self.logger = logging.getLogger(self.__class__.__name__)

        cfg = getattr(config, "phy_mcts", {})
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        elif cfg is None:
            cfg = {}

        self.max_rounds = int(cfg.get("max_rounds", 12))
        self.max_depth = int(cfg.get("max_depth", 4))
        self.max_children_per_node = int(cfg.get("max_children_per_node", 3))
        self.max_initial_subtasks = int(cfg.get("max_initial_subtasks", 5))
        self.exploration_constant = float(cfg.get("exploration_constant", 1.4))
        self.min_score = float(cfg.get("min_score", 0.0))
        self.max_score = float(cfg.get("max_score", 10.0))
        self.early_stop_score = float(cfg.get("early_stop_score", 9.0))

        self.root_id = "root"
        self.nodes: dict[str, SearchNode] = {}
        self.node_counter = 0

        self.round_records: list[dict[str, Any]] = []
        self.best_node_id = self.root_id
        self.best_score: float | None = None
        self.best_reward = -1.0
        self.contract: dict[str, Any] = {}

    @property
    def exp_name(self) -> str:
        return "PhyMaster"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        self.logger.info("Starting PHY Master workflow")
        self.logger.info("Task: %s", task_description)

        try:
            plan = self._run_clarifier(task_description, task_id)
            self._initialize_tree(task_description, plan)

            for round_idx in range(self.max_rounds):
                selected_id = self._select_node()
                if selected_id is None:
                    self.logger.info("No selectable node. Stop search.")
                    break

                node = self.nodes[selected_id]
                path_nodes = self._get_path_nodes(selected_id)
                path_summary = self._path_to_text(path_nodes)

                self.logger.info(
                    "Round %s selecting node=%s depth=%s subtask=%s",
                    round_idx,
                    selected_id,
                    node.depth,
                    node.subtask[:200],
                )

                theor_output = self._run_theoretician(
                    task_description=task_description,
                    subtask=node.subtask,
                    path_summary=path_summary,
                    round_idx=round_idx,
                    task_id=task_id,
                )

                critic_feedback = self._run_critic(
                    task_description=task_description,
                    subtask=node.subtask,
                    theoretician_output=theor_output,
                    path_summary=path_summary,
                    round_idx=round_idx,
                    task_id=task_id,
                )

                score = self._extract_score(critic_feedback)
                reward = self._score_to_reward(score)

                node.theoretician_output = theor_output
                node.critic_feedback = critic_feedback
                node.score = score
                node.reward = reward
                node.visits += 1
                node.value_sum += reward

                self._backpropagate(node.parent_id, reward)
                self._update_best(node)

                supervisor_feedback = self._run_supervisor(
                    task_description=task_description,
                    subtask=node.subtask,
                    theoretician_output=theor_output,
                    critic_feedback=critic_feedback,
                    round_idx=round_idx,
                    task_id=task_id,
                )
                node.supervisor_feedback = supervisor_feedback
                node.selected_round = round_idx

                new_children = self._expand_node(node.node_id, critic_feedback, supervisor_feedback)
                node.status = "expanded" if new_children else "closed"

                record = {
                    "round": round_idx,
                    "selected_node": node.node_id,
                    "score": score,
                    "reward": reward,
                    "new_children": new_children,
                    "critic_verdict": critic_feedback.get("verdict", ""),
                    "supervisor_action": supervisor_feedback.get("action", "continue"),
                }
                self.round_records.append(record)

                if score is not None and score >= self.early_stop_score:
                    self.logger.info("Early stop: score %s >= %s", score, self.early_stop_score)
                    break
                if str(supervisor_feedback.get("action", "")).lower() == "terminate":
                    self.logger.info("Supervisor requested terminate.")
                    break

            best_path_nodes = self._resolve_best_path()
            best_path_payload = self._serialize_path(best_path_nodes)

            summary = self._run_summarizer(
                task_description=task_description,
                best_path_payload=best_path_payload,
                task_id=task_id,
            )

            result = {
                "status": "completed",
                "steps": len(self.round_records),
                "best_score": self.best_score,
                "best_node_id": self.best_node_id,
                "contract": self.contract,
                "best_path": best_path_payload,
                "search_trace": self.round_records,
                "summary": summary,
            }

            vis_path = self._save_visualization(task_description)
            if vis_path is not None:
                result["visualization_html"] = str(vis_path)

            self.results.append(
                {
                    "task_id": task_id,
                    "status": "completed",
                    "steps": len(self.round_records),
                    "result": result,
                }
            )
            return result

        except Exception as e:
            self.logger.error("PHY Master execution failed: %s", e, exc_info=True)
            failed = {
                "status": "failed",
                "steps": len(self.round_records),
                "error": str(e),
                "search_trace": self.round_records,
            }
            vis_path = self._save_visualization(task_description)
            if vis_path is not None:
                failed["visualization_html"] = str(vis_path)
            return failed

    def _run_clarifier(self, task_description: str, task_id: str) -> dict[str, Any]:
        clarifier_schema = self._load_contract_schema()
        clarifier_cfg = self.config.model_dump().get("clarifier", {})
        max_keys = int(
            clarifier_cfg.get(
                "max_key_concpets",
                clarifier_cfg.get("max_key_concepts", 8),
            )
        )
        response = self._run_agent_once(
            agent=self.clarifier_agent,
            stage_name="clarifier",
            stage_index=0,
            task_id=f"{task_id}_clarifier",
            task_type="clarifier",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "user_query": task_description,
                "schema_json": json.dumps(clarifier_schema, ensure_ascii=False, indent=2),
                "max_keys": max_keys,
            },
        )
        payload = self._extract_json_object(response)
        return self._normalize_plan(payload, task_description)

    def _run_theoretician(
        self,
        task_description: str,
        subtask: str,
        path_summary: str,
        round_idx: int,
        task_id: str,
    ) -> str:
        node_metadata = {
            "round_index": round_idx,
            "current_subtask": subtask,
            "path_summary": path_summary,
        }
        return self._run_agent_once(
            agent=self.theoretician_agent,
            stage_name="theoretician",
            stage_index=round_idx,
            task_id=f"{task_id}_theoretician_{round_idx}",
            task_type="theoretician",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "current_subtask": subtask,
                "path_summary": path_summary,
                "round_index": round_idx,
                "subtask": subtask,
                "memory": path_summary,
                "structured": self._contract_for_prompt(),
                "node_metadata": json.dumps(node_metadata, ensure_ascii=False, indent=2),
                "path": self.theoretician_agent.session.config.workspace_path,
            },
        )

    def _run_critic(
        self,
        task_description: str,
        subtask: str,
        theoretician_output: str,
        path_summary: str,
        round_idx: int,
        task_id: str,
    ) -> dict[str, Any]:
        critic_context = {
            "task_description": task_description,
            "current_subtask": subtask,
            "path_summary": path_summary,
            "structured_contract": self.contract,
        }
        response = self._run_agent_once(
            agent=self.critic_agent,
            stage_name="critic",
            stage_index=round_idx,
            task_id=f"{task_id}_critic_{round_idx}",
            task_type="critic",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "current_subtask": subtask,
                "theoretician_output": theoretician_output,
                "path_summary": path_summary,
                "round_index": round_idx,
                "result": theoretician_output,
                "context": json.dumps(critic_context, ensure_ascii=False, indent=2),
            },
        )
        payload = self._extract_json_object(response)
        if not isinstance(payload, dict):
            payload = {}

        # Compatibility with PHY_Master critic schema.
        if "verdict" not in payload and "decision" in payload:
            decision = str(payload.get("decision", "")).lower()
            mapping = {
                "complete": "accept",
                "to_improve": "partial",
                "to_revise": "refine",
                "to_redraft": "reject",
            }
            payload["verdict"] = mapping.get(decision, "refine")
        if "analysis" not in payload:
            payload["analysis"] = payload.get("summary") or payload.get("opinion") or ""

        if "new_subtasks" not in payload:
            payload["new_subtasks"] = []
        payload.setdefault("score", None)
        payload.setdefault("verdict", "refine")
        payload.setdefault("analysis", "")
        payload.setdefault("new_subtasks", [])
        return payload

    def _run_supervisor(
        self,
        task_description: str,
        subtask: str,
        theoretician_output: str,
        critic_feedback: dict[str, Any],
        round_idx: int,
        task_id: str,
    ) -> dict[str, Any]:
        node_info = {
            "round_index": round_idx,
            "current_subtask": subtask,
            "best_score": self.best_score,
            "tree_snapshot": self._tree_snapshot(),
        }
        response = self._run_agent_once(
            agent=self.supervisor_agent,
            stage_name="supervisor",
            stage_index=round_idx,
            task_id=f"{task_id}_supervisor_{round_idx}",
            task_type="supervisor",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "current_subtask": subtask,
                "theoretician_output": theoretician_output,
                "critic_feedback_json": json.dumps(critic_feedback, ensure_ascii=False, indent=2),
                "tree_snapshot": self._tree_snapshot(),
                "best_score": self.best_score,
                "round_index": round_idx,
                "structured": self._contract_for_prompt(),
                "node": json.dumps(node_info, ensure_ascii=False, indent=2),
            },
        )
        payload = self._extract_json_object(response)
        if not isinstance(payload, dict):
            text = response.strip()
            payload = {
                "action": "continue",
                "reason": text[:400],
                "new_subtasks": [{"title": "followup", "objective": text, "priority": "medium"}] if text else [],
            }
        payload.setdefault("action", "continue")
        payload.setdefault("new_subtasks", [])
        payload.setdefault("reason", "")
        return payload

    def _run_summarizer(self, task_description: str, best_path_payload: list[dict[str, Any]], task_id: str) -> str:
        response = self._run_agent_once(
            agent=self.summarizer_agent,
            stage_name="summarizer",
            stage_index=len(self.round_records),
            task_id=f"{task_id}_summarizer",
            task_type="summarizer",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "best_score": self.best_score,
                "best_path_json": json.dumps(best_path_payload, ensure_ascii=False, indent=2),
                "search_trace_json": json.dumps(self.round_records, ensure_ascii=False, indent=2),
            },
        )
        return response

    def _run_agent_once(
        self,
        agent,
        stage_name: str,
        stage_index: int,
        task_id: str,
        task_type: str,
        description: str,
        prompt_kwargs: dict[str, Any],
    ) -> str:
        BaseAgent.set_exp_info(exp_name=f"{self.exp_name}_{stage_name}", exp_index=stage_index)

        original_kwargs = agent._prompt_format_kwargs.copy()
        agent._prompt_format_kwargs.update(prompt_kwargs)
        try:
            task = TaskInstance(
                task_id=task_id,
                task_type=task_type,
                description=description,
                input_data={},
            )
            trajectory = agent.run(task)
            response = self._extract_agent_response(trajectory)
            return response or ""
        finally:
            agent._prompt_format_kwargs = original_kwargs

    def _initialize_tree(self, task_description: str, plan: dict[str, Any]) -> None:
        self.contract = plan.get("contract", {}) if isinstance(plan, dict) else {}
        summary = str(plan.get("problem_summary") or task_description).strip()
        self.nodes = {
            self.root_id: SearchNode(
                node_id=self.root_id,
                subtask=summary,
                parent_id=None,
                depth=0,
                created_by="root",
                status="expanded",
            )
        }
        self.node_counter = 0
        self.best_node_id = self.root_id
        self.best_score = None
        self.best_reward = -1.0

        subtasks = self._normalize_subtasks(plan.get("subtasks", []))
        if not subtasks:
            subtasks = [task_description]

        for subtask in subtasks[: self.max_initial_subtasks]:
            self._add_child(self.root_id, subtask, "clarifier")

    def _select_node(self) -> str | None:
        candidates = [
            node
            for node in self.nodes.values()
            if node.node_id != self.root_id and node.status in {"open", "expanded"}
        ]
        if not candidates:
            return None

        def ucb(node: SearchNode) -> float:
            if node.visits == 0:
                return float("inf")
            if node.parent_id and node.parent_id in self.nodes:
                parent_visits = max(1, self.nodes[node.parent_id].visits)
            else:
                parent_visits = max(1, self.nodes[self.root_id].visits)

            exploit = node.average_value
            explore = self.exploration_constant * math.sqrt(math.log(parent_visits + 1) / node.visits)
            return exploit + explore

        selected = max(candidates, key=lambda node: (ucb(node), -node.depth))
        return selected.node_id

    def _expand_node(
        self,
        node_id: str,
        critic_feedback: dict[str, Any],
        supervisor_feedback: dict[str, Any],
    ) -> list[str]:
        node = self.nodes[node_id]
        if node.depth >= self.max_depth:
            return []

        candidates: list[str] = []
        candidates.extend(self._normalize_subtasks(critic_feedback.get("new_subtasks", [])))
        candidates.extend(self._normalize_subtasks(supervisor_feedback.get("new_subtasks", [])))

        if not candidates:
            verdict = str(critic_feedback.get("verdict", "")).lower()
            action = str(supervisor_feedback.get("action", "")).lower()
            reason = str(critic_feedback.get("analysis", "")).strip()
            if verdict in {"refine", "partial"} or action in {"refine", "continue", "branch"}:
                fallback = f"Refine this subtask: {node.subtask}"
                if reason:
                    fallback += f". Address critic concern: {reason}"
                candidates = [fallback]

        existing = {
            self.nodes[child_id].subtask.strip().lower()
            for child_id in node.children
            if child_id in self.nodes
        }

        created: list[str] = []
        for subtask in candidates:
            normalized = subtask.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in existing:
                continue
            if len(created) >= self.max_children_per_node:
                break
            child_id = self._add_child(node_id, normalized, "supervisor")
            created.append(child_id)
            existing.add(key)

        return created

    def _add_child(self, parent_id: str, subtask: str, created_by: str) -> str:
        self.node_counter += 1
        child_id = f"n{self.node_counter:03d}"
        parent = self.nodes[parent_id]
        child = SearchNode(
            node_id=child_id,
            subtask=subtask,
            parent_id=parent_id,
            depth=parent.depth + 1,
            created_by=created_by,
            status="open",
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        return child_id

    def _backpropagate(self, parent_id: str | None, reward: float) -> None:
        current = parent_id
        while current is not None:
            node = self.nodes[current]
            node.visits += 1
            node.value_sum += reward
            current = node.parent_id

    def _update_best(self, node: SearchNode) -> None:
        if node.reward > self.best_reward:
            self.best_reward = node.reward
            self.best_score = node.score
            self.best_node_id = node.node_id

    def _resolve_best_path(self) -> list[SearchNode]:
        if self.best_node_id == self.root_id:
            scored_nodes = [node for node in self.nodes.values() if node.node_id != self.root_id and node.score is not None]
            if scored_nodes:
                best = max(scored_nodes, key=lambda n: (n.score if n.score is not None else -1.0, n.average_value))
                self.best_node_id = best.node_id
                self.best_score = best.score
        return self._get_path_nodes(self.best_node_id)

    def _get_path_nodes(self, node_id: str) -> list[SearchNode]:
        path: list[SearchNode] = []
        current = node_id
        while current in self.nodes:
            node = self.nodes[current]
            path.append(node)
            if node.parent_id is None:
                break
            current = node.parent_id
        path.reverse()
        return path

    def _serialize_path(self, path_nodes: list[SearchNode]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.node_id,
                "depth": node.depth,
                "subtask": node.subtask,
                "score": node.score,
                "reward": node.reward,
                "visits": node.visits,
                "created_by": node.created_by,
                "critic_feedback": node.critic_feedback,
                "supervisor_feedback": node.supervisor_feedback,
                "theoretician_output": node.theoretician_output,
            }
            for node in path_nodes
        ]

    def _path_to_text(self, path_nodes: list[SearchNode]) -> str:
        lines: list[str] = []
        for node in path_nodes:
            lines.append(
                f"[{node.node_id}] depth={node.depth} score={node.score} subtask={node.subtask}"
            )
        return "\n".join(lines)

    def _tree_snapshot(self) -> str:
        nodes = [node for node in self.nodes.values() if node.node_id != self.root_id]
        if not nodes:
            return "(empty tree)"

        nodes = sorted(
            nodes,
            key=lambda n: (
                n.score if n.score is not None else -1.0,
                n.average_value,
                n.visits,
            ),
            reverse=True,
        )[:8]

        lines = []
        for node in nodes:
            lines.append(
                f"{node.node_id} depth={node.depth} score={node.score} avg={node.average_value:.3f} "
                f"visits={node.visits} status={node.status} subtask={node.subtask[:140]}"
            )
        return "\n".join(lines)

    def _extract_score(self, payload: dict[str, Any]) -> float | None:
        value = payload.get("score")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _score_to_reward(self, score: float | None) -> float:
        if score is None:
            return 0.0
        if self.max_score <= self.min_score:
            return 0.0
        clamped = max(self.min_score, min(self.max_score, score))
        return (clamped - self.min_score) / (self.max_score - self.min_score)

    def _normalize_plan(self, payload: Any, task_description: str) -> dict[str, Any]:
        payload_dict = payload if isinstance(payload, dict) else {}
        contract = self._normalize_contract(payload, task_description)
        summary = str(
            contract.get("task_description")
            or contract.get("task_background")
            or contract.get("topic")
            or task_description
        ).strip()

        candidate_lists = [
            contract.get("subtasks"),
            contract.get("sub_tasks"),
            payload_dict.get("initial_subtasks"),
            payload_dict.get("subtasks"),
            payload_dict.get("tasks"),
            payload_dict.get("plan"),
        ]

        subtasks_raw = []
        for candidate in candidate_lists:
            if isinstance(candidate, list):
                subtasks_raw = candidate
                break

        return {
            "problem_summary": summary,
            "subtasks": self._normalize_subtasks(subtasks_raw),
            "contract": contract,
        }

    def _normalize_contract(self, payload: Any, task_description: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}

        contract = dict(payload)
        contract.pop("instruction_filename", None)

        if "input" not in contract and isinstance(contract.get("input_data"), dict):
            contract["input"] = contract.get("input_data")

        if "subtasks" not in contract:
            alt_subtasks = contract.get("sub_tasks") or contract.get("initial_subtasks")
            if isinstance(alt_subtasks, list):
                contract["subtasks"] = alt_subtasks

        if "sub_tasks" not in contract:
            alt_subtasks = contract.get("subtasks") or contract.get("initial_subtasks")
            if isinstance(alt_subtasks, list):
                contract["sub_tasks"] = alt_subtasks

        contract.setdefault("topic", "None")
        contract.setdefault("domain", "None")
        contract.setdefault("subdomain", "None")
        contract.setdefault("related_knowledge", [])
        contract.setdefault("task_type", "None")
        contract.setdefault("task_background", "None")
        contract.setdefault("task_description", task_description or "None")
        contract.setdefault("constraints", [])
        contract.setdefault(
            "input",
            {
                "format": "None",
                "path": "none",
                "description": "None",
            },
        )
        contract.setdefault("expected_output", [])
        contract.setdefault("subtasks", [])
        contract.setdefault("sub_tasks", [])
        return contract

    def _normalize_subtasks(self, raw: Any) -> list[str]:
        results: list[str] = []

        if isinstance(raw, dict):
            iterable = list(raw.values())
        elif isinstance(raw, list):
            iterable = raw
        elif isinstance(raw, str):
            iterable = [raw]
        else:
            iterable = []

        for item in iterable:
            text = ""
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or "").strip()
                objective = str(
                    item.get("description")
                    or item.get("objective")
                    or item.get("task")
                    or ""
                ).strip()
                task_type = str(item.get("subtask_type") or "").strip()
                item_input = str(item.get("input") or "").strip()
                item_output = str(item.get("expected_output") or "").strip()
                task_id = str(item.get("id") or "").strip()

                parts = []
                if task_id:
                    parts.append(f"#{task_id}")
                if task_type:
                    parts.append(task_type)
                if title:
                    parts.append(title)
                if objective:
                    parts.append(objective)
                if item_input:
                    parts.append(f"input: {item_input}")
                if item_output:
                    parts.append(f"expected_output: {item_output}")
                text = " | ".join(parts)

            text = text.strip()
            if text:
                results.append(text)

        deduped: list[str] = []
        seen = set()
        for text in results:
            key = text.lower()
            if key in seen:
                continue
            deduped.append(text)
            seen.add(key)

        return deduped

    def _contract_for_prompt(self) -> str:
        if not self.contract:
            return "None"
        try:
            return json.dumps(self.contract, ensure_ascii=False, indent=2)
        except Exception:
            return str(self.contract)

    def _load_contract_schema(self) -> dict[str, Any]:
        schema_path = Path(__file__).resolve().parent.parent / "template" / "contract_template.json"
        if schema_path.exists():
            try:
                with schema_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                self.logger.warning("Failed to load contract schema from %s: %s", schema_path, e)

        # Fallback minimal schema to keep clarifier runnable.
        return {
            "topic": "string",
            "domain": "string",
            "subdomain": "string",
            "related_knowledge": ["string"],
            "task_type": "string",
            "task_background": "string",
            "task_description": "string",
            "input": {"format": "string", "path": "string", "description": "string"},
            "constraints": ["string"],
            "expected_output": [
                {"format": "string", "path": "string", "description": "string"}
            ],
            "subtasks": [
                {
                    "id": 1,
                    "subtask_type": "reasoning",
                    "input": "string",
                    "expected_output": "string",
                    "description": "string",
                }
            ],
            "sub_tasks": [
                {
                    "id": 1,
                    "subtask_type": "reasoning",
                    "input": "string",
                    "expected_output": "string",
                    "description": "string",
                }
            ],
        }

    def _extract_json_object(self, text: str) -> Any:
        if not text:
            return {}
        content = text.strip()

        # 1) direct JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 2) fenced block
        blocks = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", content)
        for block in blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        # 3) best effort: from first { to last }
        left = content.find("{")
        right = content.rfind("}")
        if left != -1 and right != -1 and right > left:
            candidate = content[left : right + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return {}

        # 4) best effort: from first [ to last ]
        left = content.find("[")
        right = content.rfind("]")
        if left != -1 and right != -1 and right > left:
            candidate = content[left : right + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return {}

        return {}

    def _serialize_nodes_for_visualization(self) -> list[dict[str, Any]]:
        node_list: list[dict[str, Any]] = []
        for node_id, node in self.nodes.items():
            node_list.append(
                {
                    "node_id": node_id,
                    "parent_id": node.parent_id,
                    "children": list(node.children),
                    "depth": node.depth,
                    "subtask": _safe_short(node.subtask, 4000),
                    "score": node.score,
                    "reward": node.reward,
                    "visits": node.visits,
                    "status": node.status,
                    "created_by": node.created_by,
                    "theoretician_output": _safe_short(node.theoretician_output, 12000),
                    "critic_feedback": node.critic_feedback,
                    "supervisor_feedback": node.supervisor_feedback,
                    "selected_round": node.selected_round,
                }
            )
        return node_list

    def _save_visualization(self, task_description: str) -> Path | None:
        if not self.run_dir:
            return None
        output_path = Path(self.run_dir) / "visualization.html"
        nodes = self._serialize_nodes_for_visualization()
        write_mcts_html(
            output_path,
            nodes=nodes,
            root_id=self.root_id,
            best_node_id=self.best_node_id,
            task_description=task_description,
        )
        return output_path


def _safe_short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]
