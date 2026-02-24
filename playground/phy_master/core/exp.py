"""PHY Master experiment with MCTS-style exploration."""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    subtask: Any
    parent_id: str | None
    depth: int
    description: str = ""
    node_type: str = "draft"
    created_by: str = "clarifier"
    visits: int = 0
    value_sum: float = 0.0
    score: float | None = None
    reward: float = 0.0
    status: str = "open"
    theoretician_output: str = ""
    memory: str = ""
    supervisor_dispatch: dict[str, Any] = field(default_factory=dict)
    critic_feedback: dict[str, Any] = field(default_factory=dict)
    supervisor_feedback: dict[str, Any] = field(default_factory=dict)
    selected_round: int | None = None
    subtask_index: int = -1
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
        config=None,
        supervisor_agent_pool=None,
        theoretician_agent_pool=None,
        critic_agent_pool=None,
        parallel_nodes: int = 1,
    ):
        super().__init__(clarifier_agent, config)
        self.clarifier_agent = clarifier_agent
        self.supervisor_agent = supervisor_agent
        self.theoretician_agent = theoretician_agent
        self.critic_agent = critic_agent
        self.summarizer_agent = summarizer_agent
        self.supervisor_agent_pool = supervisor_agent_pool or [supervisor_agent]
        self.theoretician_agent_pool = theoretician_agent_pool or [theoretician_agent]
        self.critic_agent_pool = critic_agent_pool or [critic_agent]
        self.logger = logging.getLogger(self.__class__.__name__)

        cfg = getattr(config, "phy_mcts", {})
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        elif cfg is None:
            cfg = {}

        self.max_rounds = int(cfg.get("max_rounds", 12))
        self.max_depth = int(cfg.get("max_depth", 4))
        self.max_children_per_node = int(cfg.get("max_children_per_node", 3))
        self.beam_width = max(0, int(cfg.get("beam_width", 0)))
        self.exploration_constant = float(cfg.get("exploration_constant", 1.4))
        self.min_score = float(cfg.get("min_score", 0.0))
        self.max_score = float(cfg.get("max_score", 10.0))
        cfg_parallel = cfg.get("parallel_nodes", cfg.get("parallel_processes", parallel_nodes))
        self.parallel_nodes = max(1, int(cfg_parallel if cfg_parallel is not None else 1))

        self.root_id = "root"
        self.nodes: dict[str, SearchNode] = {}
        self.node_counter = 0

        self.round_records: list[dict[str, Any]] = []
        self.best_node_id = self.root_id
        self.contract: dict[str, Any] = {}
        self.subtask_sequence: list[dict[str, Any]] = []
        self.beam_expandable_node_ids: set[str] = set()

    @property
    def exp_name(self) -> str:
        return "PhyMaster"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        self.logger.info("Starting PHY Master workflow")
        self.logger.info("Task: %s", task_description)

        try:
            plan = self._run_clarifier(task_description, task_id)
            self._initialize_tree(task_description, plan)
            completed_path_nodes: list[SearchNode] | None = self._find_full_completion_path()

            for round_idx in range(self.max_rounds):
                selected_ids = self._select_nodes(self.parallel_nodes)
                if not selected_ids:
                    self.logger.info("No selectable node. Stop search.")
                    break

                self.logger.info(
                    "Round %s selected nodes: %s (parallel_nodes=%s)",
                    round_idx,
                    selected_ids,
                    self.parallel_nodes,
                )

                eval_results: list[dict[str, Any]] = []
                workers = max(1, min(self.parallel_nodes, len(selected_ids)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            self._evaluate_selected_node,
                            node_id,
                            round_idx,
                            task_description,
                            task_id,
                            slot,
                        ): node_id
                        for slot, node_id in enumerate(selected_ids)
                    }
                    for future in as_completed(futures):
                        eval_results.append(future.result())

                for item in sorted(eval_results, key=lambda x: x["slot"]):
                    node = self.nodes[item["node_id"]]
                    supervisor_dispatch = item["supervisor_dispatch"]
                    theor_output = item["theoretician_output"]
                    critic_feedback = item["critic_feedback"]
                    score = item["score"]
                    reward = item["reward"]

                    node.supervisor_dispatch = supervisor_dispatch
                    node.node_type = supervisor_dispatch.get("node_type", node.node_type)
                    node.subtask = supervisor_dispatch.get("subtask", node.subtask)
                    node.description = supervisor_dispatch.get("description", node.description)

                    node.theoretician_output = theor_output
                    node.critic_feedback = critic_feedback
                    node.score = score
                    node.reward = reward
                    node.visits += 1
                    node.value_sum += reward

                    self._backpropagate(node.parent_id, reward)
                    node.memory = self._compose_node_memory(node, critic_feedback)

                    node.supervisor_feedback = supervisor_dispatch
                    node.selected_round = round_idx

                    # Once a full-completion path exists, stop creating new nodes.
                    has_full_completion = self._find_full_completion_path() is not None
                    if (not has_full_completion) and round_idx < self.max_rounds - 1:
                        new_children = self._expand_node(node.node_id, critic_feedback)
                        node.status = "expanded" if self._can_expand_node(node) else "closed"
                    else:
                        new_children = []
                        node.status = "closed"

                    record = {
                        "round": round_idx,
                        "selected_node": node.node_id,
                        "score": score,
                        "reward": reward,
                        "node_type": node.node_type,
                        "subtask": node.subtask,
                        "description": node.description,
                        "new_children": new_children,
                        "critic_verdict": critic_feedback.get("verdict", ""),
                    }
                    self.round_records.append(record)

                self._apply_beam_pruning()
                completed_path_nodes = self._find_full_completion_path()

                if completed_path_nodes is not None:
                    completed_count, _ = self._count_completed_subtasks_in_path(completed_path_nodes)
                    self.logger.info(
                        "Early stop: found a full-completion path (%s/%s subtasks).",
                        completed_count,
                        len(self.subtask_sequence),
                    )
                    break

            if completed_path_nodes is not None:
                best_path_nodes = completed_path_nodes
            else:
                best_path_nodes = self._resolve_best_path()

            if best_path_nodes:
                best_leaf = best_path_nodes[-1]
                self.best_node_id = best_leaf.node_id
            best_path_payload = self._serialize_path(best_path_nodes)

            summary = self._run_summarizer(
                task_description=task_description,
                best_path_payload=best_path_payload,
                task_id=task_id,
            )
            self._save_summary_markdown(
                task_description=task_description,
                summary=summary,
                best_path_payload=best_path_payload,
            )

            result = {
                "status": "completed",
                "steps": len(self.round_records),
                "best_node_id": self.best_node_id,
                "contract": self.contract,
                "best_path": best_path_payload,
                "search_trace": self.round_records,
                "summary": summary,
            }

            vis_path = self._save_visualization(task_description, summary=summary)
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
        plan = self._normalize_plan(payload, task_description)
        self._save_clarifier_contract(plan.get("contract", {}), task_id)
        return plan

    def _run_theoretician(
        self,
        task_description: str,
        node_id: str,
        node_type: str,
        subtask: Any,
        description: str,
        round_idx: int,
        task_id: str,
        agent=None,
    ) -> str:
        use_agent = agent or self.theoretician_agent
        node_workspace = self._node_workspace_path(node_id, use_agent)
        node_metadata = {
            "round_index": round_idx,
            "node_id": node_id,
            "node_type": node_type,
            "current_subtask": subtask,
            "node_workspace_path": str(node_workspace),
        }
        return self._run_agent_once(
            agent=use_agent,
            stage_name="theoretician",
            stage_index=round_idx,
            task_id=f"{task_id}_theoretician_{round_idx}",
            task_type="theoretician",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "subtask": self._subtask_to_prompt_text(subtask),
                "node_description": description,
                "node_metadata": json.dumps(node_metadata, ensure_ascii=False, indent=2),
                "path": str(node_workspace),
            },
        )

    def _run_critic(
        self,
        task_description: str,
        subtask: Any,
        theoretician_output: str,
        path_summary: str,
        round_idx: int,
        task_id: str,
        agent=None,
    ) -> dict[str, Any]:
        use_agent = agent or self.critic_agent
        critic_context = {
            "task_description": task_description,
            "current_subtask": self._subtask_to_prompt_text(subtask),
            "path_summary": path_summary,
            "structured_contract": self.contract,
        }
        response = self._run_agent_once(
            agent=use_agent,
            stage_name="critic",
            stage_index=round_idx,
            task_id=f"{task_id}_critic_{round_idx}",
            task_type="critic",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "current_subtask": self._subtask_to_prompt_text(subtask),
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

        payload["summary"] = self._to_natural_text(payload.get("summary"))
        payload["opinion"] = self._to_natural_text(payload.get("opinion"))
        if "analysis" not in payload:
            payload["analysis"] = payload.get("summary") or payload.get("opinion") or ""
        payload["analysis"] = self._to_natural_text(payload.get("analysis"))

        payload.setdefault("score", None)
        payload.setdefault("verdict", "refine")
        payload.setdefault("analysis", "")
        payload.pop("new_subtasks", None)
        return payload

    def _run_supervisor_dispatch(
        self,
        task_description: str,
        node: SearchNode,
        path_memory: list[dict[str, Any]],
        round_idx: int,
        task_id: str,
        agent=None,
    ) -> dict[str, Any]:
        use_agent = agent or self.supervisor_agent
        node_info = {
            "round_index": round_idx,
            "node_id": node.node_id,
            "current_node_type": node.node_type,
            "current_subtask": self._subtask_to_prompt_text(node.subtask),
            "current_description": node.description,
            "tree_snapshot": self._tree_snapshot(),
        }
        response = self._run_agent_once(
            agent=use_agent,
            stage_name="supervisor_dispatch",
            stage_index=round_idx,
            task_id=f"{task_id}_supervisor_dispatch_{round_idx}",
            task_type="supervisor_dispatch",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "path_memory_json": json.dumps(path_memory, ensure_ascii=False, indent=2),
                "tree_snapshot": self._tree_snapshot(),
                "round_index": round_idx,
                "structured": self._contract_for_prompt(),
                "node": json.dumps(node_info, ensure_ascii=False, indent=2),
            },
        )
        payload = self._extract_json_object(response)
        if not isinstance(payload, dict):
            payload = {}

        node_type = str(payload.get("node_type") or node.node_type or "draft").strip().lower()
        if node_type not in {"draft", "revise", "improve"}:
            node_type = "draft"

        # Subtask is rule-based: always sourced from clarifier contract sequence.
        new_subtask = self._get_contract_subtask_by_index(node.subtask_index, node.subtask)
        new_description = str(
            payload.get("description")
            or node.description
            or self._subtask_brief(new_subtask)
        ).strip() or self._subtask_brief(new_subtask)
        return {
            "node_type": node_type,
            "subtask": new_subtask,
            "description": new_description,
        }

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
                description=summary,
                node_type="draft",
                created_by="root",
                status="expanded",
            )
        }
        self.node_counter = 0
        self.best_node_id = self.root_id

        subtask_specs = self._normalize_subtask_specs(plan.get("subtasks", []))
        if not subtask_specs:
            subtask_specs = [{
                "node_type": "draft",
                "subtask": {
                    "id": 1,
                    "subtask_type": "reasoning",
                    "input": "None",
                    "expected_output": "None",
                    "description": task_description,
                },
                "description": task_description,
            }]
        self.subtask_sequence = subtask_specs

        initial_children = max(0, self.max_children_per_node)
        if initial_children <= 0:
            return
        if not self.subtask_sequence:
            return

        first_spec = self.subtask_sequence[0]
        for idx in range(initial_children):
            # Root parallel lanes always start from subtask #1.
            spec = first_spec
            subtask_index = 0
            lane_suffix = "" if idx == 0 else f" [parallel lane {idx + 1}]"
            base_desc = str(spec.get("description", "")).strip()
            self._add_child(
                self.root_id,
                spec.get("subtask", task_description),
                "clarifier",
                node_type=spec.get("node_type", "draft"),
                description=(base_desc + lane_suffix).strip(),
                subtask_index=subtask_index,
            )
        self._apply_beam_pruning()

    def _select_node(self) -> str | None:
        return self._select_node_with_exclude(set())

    def _select_nodes(self, count: int) -> list[str]:
        selected: list[str] = []
        excluded: set[str] = set()
        target = max(1, int(count))
        for _ in range(target):
            node_id = self._select_node_with_exclude(excluded)
            if node_id is None:
                break
            selected.append(node_id)
            excluded.add(node_id)
        return selected

    def _select_node_with_exclude(self, excluded_ids: set[str]) -> str | None:
        candidates = [
            node
            for node in self.nodes.values()
            if (
                node.node_id != self.root_id
                and node.node_id not in excluded_ids
                and self._can_select_node(node)
            )
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

    def _evaluate_selected_node(
        self,
        node_id: str,
        round_idx: int,
        task_description: str,
        task_id: str,
        slot: int,
    ) -> dict[str, Any]:
        node = self.nodes[node_id]
        path_nodes = self._get_path_nodes(node_id)
        path_summary = self._path_to_text(path_nodes)
        path_memory = self._build_path_memory(path_nodes)

        supervisor_agent = self.supervisor_agent_pool[slot % len(self.supervisor_agent_pool)]
        theoretician_agent = self.theoretician_agent_pool[slot % len(self.theoretician_agent_pool)]
        critic_agent = self.critic_agent_pool[slot % len(self.critic_agent_pool)]

        self.logger.info(
            "Round %s slot=%s evaluating node=%s depth=%s node_type=%s",
            round_idx,
            slot,
            node_id,
            node.depth,
            node.node_type,
        )

        supervisor_dispatch = self._run_supervisor_dispatch(
            task_description=task_description,
            node=node,
            path_memory=path_memory,
            round_idx=round_idx,
            task_id=task_id,
            agent=supervisor_agent,
        )

        dispatched_node_type = supervisor_dispatch.get("node_type", node.node_type)
        dispatched_subtask = supervisor_dispatch.get("subtask", node.subtask)
        dispatched_description = supervisor_dispatch.get("description", node.description)

        theor_output = self._run_theoretician(
            task_description=task_description,
            node_id=node.node_id,
            node_type=dispatched_node_type,
            subtask=dispatched_subtask,
            description=dispatched_description,
            round_idx=round_idx,
            task_id=task_id,
            agent=theoretician_agent,
        )

        critic_feedback = self._run_critic(
            task_description=task_description,
            subtask=dispatched_subtask,
            theoretician_output=theor_output,
            path_summary=path_summary,
            round_idx=round_idx,
            task_id=task_id,
            agent=critic_agent,
        )

        score = self._extract_score(critic_feedback)
        reward = self._score_to_reward(score)
        return {
            "slot": slot,
            "node_id": node_id,
            "supervisor_dispatch": supervisor_dispatch,
            "theoretician_output": theor_output,
            "critic_feedback": critic_feedback,
            "score": score,
            "reward": reward,
        }

    def _expand_node(
        self,
        node_id: str,
        critic_feedback: dict[str, Any],
    ) -> list[str]:
        node = self.nodes[node_id]
        if node.depth >= self.max_depth:
            return []

        verdict = str(critic_feedback.get("verdict", "")).lower()
        decision = str(critic_feedback.get("decision", "")).lower()
        reason = str(critic_feedback.get("analysis", "") or critic_feedback.get("summary", "")).strip()

        candidates: list[dict[str, Any]] = []
        if decision == "complete" or verdict == "accept":
            next_index = node.subtask_index + 1 if node.subtask_index >= 0 else 0
            if 0 <= next_index < len(self.subtask_sequence):
                nxt = self.subtask_sequence[next_index]
                candidates.append(
                    {
                        "node_type": str(nxt.get("node_type") or "draft"),
                        "subtask": nxt.get("subtask"),
                        "description": str(nxt.get("description") or self._subtask_brief(nxt.get("subtask")) or ""),
                        "subtask_index": str(next_index),
                    }
                )
        else:
            mapped_type = "revise"
            if decision == "to_improve" or verdict == "partial":
                mapped_type = "improve"
            elif decision == "to_redraft" or verdict == "reject":
                mapped_type = "draft"
            fallback_subtask = node.subtask
            fallback_description = node.description or self._subtask_brief(node.subtask)
            if reason:
                fallback_description += f" | Critic focus: {reason}"
            candidates.append(
                {
                    "node_type": mapped_type,
                    "subtask": fallback_subtask,
                    "description": fallback_description,
                    "subtask_index": str(node.subtask_index),
                }
            )

        existing = {
            f"{self.nodes[child_id].node_type}::{self._subtask_key(self.nodes[child_id].subtask)}"
            for child_id in node.children
            if child_id in self.nodes
        }

        remaining_slots = max(0, self.max_children_per_node - len(node.children))
        if remaining_slots <= 0:
            return []

        created: list[str] = []
        for spec in candidates:
            normalized = spec.get("subtask")
            if normalized is None or (isinstance(normalized, str) and not normalized.strip()):
                continue
            node_type = str(spec.get("node_type", "draft")).strip().lower() or "draft"
            key = f"{node_type}::{self._subtask_key(normalized)}"
            if key in existing:
                continue
            if len(created) >= remaining_slots:
                break
            child_id = self._add_child(
                node_id,
                normalized,
                "critic",
                node_type=spec.get("node_type", "draft"),
                description=str(spec.get("description", self._subtask_brief(normalized))),
                subtask_index=int(spec.get("subtask_index", node.subtask_index)),
            )
            created.append(child_id)
            existing.add(key)

        # If no new node was created and this node still has remaining capacity,
        # try alternative node types only when critic says this node is NOT complete.
        if (
            not created
            and len(node.children) < self.max_children_per_node
            and not (decision == "complete" or verdict == "accept")
        ):
            fallback_subtask = node.subtask
            fallback_description = node.description or self._subtask_brief(node.subtask)
            alt_types = ["draft", "revise", "improve"]
            for alt_type in alt_types:
                key = f"{alt_type}::{self._subtask_key(fallback_subtask)}"
                if key in existing:
                    continue
                child_id = self._add_child(
                    node_id,
                    fallback_subtask,
                    "critic",
                    node_type=alt_type,
                    description=fallback_description,
                    subtask_index=node.subtask_index,
                )
                created.append(child_id)
                existing.add(key)
                if len(created) >= remaining_slots:
                    break

        return created

    def _add_child(
        self,
        parent_id: str,
        subtask: Any,
        created_by: str,
        node_type: str = "draft",
        description: str = "",
        subtask_index: int = -1,
    ) -> str:
        self.node_counter += 1
        child_id = f"n{self.node_counter:03d}"
        parent = self.nodes[parent_id]
        clean_type = str(node_type or "draft").strip().lower()
        if clean_type not in {"draft", "revise", "improve"}:
            clean_type = "draft"
        child = SearchNode(
            node_id=child_id,
            subtask=subtask,
            parent_id=parent_id,
            depth=parent.depth + 1,
            description=(description or self._subtask_brief(subtask)),
            node_type=clean_type,
            created_by=created_by,
            status="open",
            subtask_index=subtask_index,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        if len(parent.children) >= self.max_children_per_node:
            parent.status = "closed"
        else:
            parent.status = "expanded"
        return child_id

    def _can_expand_node(self, node: SearchNode) -> bool:
        if node.node_id == self.root_id:
            return False
        if node.depth >= self.max_depth:
            return False
        if len(node.children) >= self.max_children_per_node:
            return False
        if self.beam_width > 0 and node.node_id not in self.beam_expandable_node_ids:
            return False
        return True

    def _can_select_node(self, node: SearchNode) -> bool:
        if node.node_id == self.root_id:
            return False
        # Terminal-depth nodes are still selectable once to avoid empty placeholder nodes.
        if node.depth >= self.max_depth:
            return node.visits == 0
        if node.status not in {"open", "expanded"}:
            return False
        if node.visits == 0:
            return True
        return self._can_expand_node(node)

    def _apply_beam_pruning(self) -> None:
        if self.beam_width <= 0:
            self.beam_expandable_node_ids = {
                node.node_id
                for node in self.nodes.values()
                if node.node_id != self.root_id and node.depth < self.max_depth and len(node.children) < self.max_children_per_node
            }
            return

        by_depth: dict[int, list[SearchNode]] = {}
        for node in self.nodes.values():
            if node.node_id == self.root_id:
                continue
            if node.depth >= self.max_depth:
                continue
            if len(node.children) >= self.max_children_per_node:
                continue
            by_depth.setdefault(node.depth, []).append(node)

        expandable_ids: set[str] = set()
        for depth_nodes in by_depth.values():
            ranked = sorted(
                depth_nodes,
                key=lambda n: (
                    -(n.reward if n.reward is not None else 0.0),
                    -((n.score if n.score is not None else self.min_score) if isinstance(n.score, (int, float)) else self.min_score),
                    -n.visits,
                    n.node_id,
                ),
            )
            for node in ranked[: self.beam_width]:
                expandable_ids.add(node.node_id)

        self.beam_expandable_node_ids = expandable_ids
        for node in self.nodes.values():
            if node.node_id == self.root_id:
                continue
            if node.depth >= self.max_depth or len(node.children) >= self.max_children_per_node:
                node.status = "closed"
                continue
            if node.node_id in self.beam_expandable_node_ids:
                node.status = "expanded" if node.children else "open"
            else:
                node.status = "closed"

    def _is_subtask_complete(self, critic_feedback: dict[str, Any]) -> bool:
        decision = str(critic_feedback.get("decision", "")).strip().lower()
        verdict = str(critic_feedback.get("verdict", "")).strip().lower()
        return decision == "complete" or verdict == "accept"

    def _to_natural_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                text = self._to_natural_text(item)
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                text = self._to_natural_text(v)
                if text:
                    parts.append(f"{k}: {text}")
            return "; ".join(parts).strip()
        return str(value).strip()

    def _backpropagate(self, parent_id: str | None, reward: float) -> None:
        current = parent_id
        while current is not None:
            node = self.nodes[current]
            node.visits += 1
            node.value_sum += reward
            current = node.parent_id

    def _count_completed_subtasks_in_path(self, path_nodes: list[SearchNode]) -> tuple[int, set[int]]:
        completed: set[int] = set()
        for node in path_nodes:
            if node.node_id == self.root_id:
                continue
            if node.subtask_index < 0:
                continue
            if self._is_subtask_complete(node.critic_feedback):
                completed.add(node.subtask_index)
        return len(completed), completed

    def _path_reward_sum(self, path_nodes: list[SearchNode]) -> float:
        total = 0.0
        for node in path_nodes:
            if node.node_id == self.root_id:
                continue
            total += float(node.reward or 0.0)
        return total

    def _find_full_completion_path(self) -> list[SearchNode] | None:
        total_subtasks = len(self.subtask_sequence)
        if total_subtasks <= 0:
            return None

        complete_candidates: list[list[SearchNode]] = []
        for node in self.nodes.values():
            if node.node_id == self.root_id or node.visits <= 0:
                continue
            path_nodes = self._get_path_nodes(node.node_id)
            completed_count, _ = self._count_completed_subtasks_in_path(path_nodes)
            if completed_count >= total_subtasks:
                complete_candidates.append(path_nodes)

        if not complete_candidates:
            return None

        return max(
            complete_candidates,
            key=lambda path: (
                self._path_reward_sum(path),
                len(path),
                path[-1].visits if path else 0,
                path[-1].node_id if path else "",
            ),
        )

    def _resolve_best_path(self) -> list[SearchNode]:
        candidates = [node for node in self.nodes.values() if node.node_id != self.root_id and node.visits > 0]
        if not candidates:
            return self._get_path_nodes(self.root_id)

        best_path = max(
            (self._get_path_nodes(node.node_id) for node in candidates),
            key=lambda path: (
                self._count_completed_subtasks_in_path(path)[0],
                self._path_reward_sum(path),
                len(path),
                path[-1].node_id if path else "",
            ),
        )
        return best_path

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
                "node_type": node.node_type,
                "subtask": node.subtask,
                "description": node.description,
                "score": node.score,
                "reward": node.reward,
                "visits": node.visits,
                "created_by": node.created_by,
                "memory": node.memory,
                "subtask_index": node.subtask_index,
                "supervisor_dispatch": node.supervisor_dispatch,
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
                f"[{node.node_id}] depth={node.depth} type={node.node_type} score={node.score} subtask={self._subtask_brief(node.subtask)}"
            )
        return "\n".join(lines)

    def _build_path_memory(self, path_nodes: list[SearchNode], include_current: bool = False) -> list[dict[str, Any]]:
        memory_chain: list[dict[str, Any]] = []
        nodes = path_nodes if include_current else path_nodes[:-1]
        for node in nodes:
            if node.node_id == self.root_id:
                continue
            memory_chain.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "node_type": node.node_type,
                    "subtask": node.subtask,
                    "description": node.description,
                    "memory": node.memory,
                    "score": node.score,
                }
            )
        return memory_chain

    def _compose_node_memory(self, node: SearchNode, critic_feedback: dict[str, Any]) -> str:
        summary = str(
            critic_feedback.get("summary")
            or critic_feedback.get("opinion")
            or critic_feedback.get("analysis")
            or ""
        ).strip()
        verdict = str(critic_feedback.get("verdict") or critic_feedback.get("decision") or "").strip()
        score = critic_feedback.get("score")
        score_text = f"{score}" if score is not None else "None"
        parts = [
            f"type={node.node_type}",
            f"subtask={self._subtask_brief(node.subtask)}",
            f"description={node.description}",
            f"verdict={verdict or 'unknown'}",
            f"score={score_text}",
        ]
        if summary:
            parts.append(f"summary={summary}")
        return " | ".join(parts)

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
                f"{node.node_id} depth={node.depth} type={node.node_type} score={node.score} avg={node.average_value:.3f} "
                f"visits={node.visits} status={node.status} subtask={self._subtask_brief(node.subtask)[:140]}"
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
            # Keep structured subtasks so downstream nodes can preserve
            # contract fields like id/subtask_type/input/expected_output.
            "subtasks": subtasks_raw if isinstance(subtasks_raw, list) else self._normalize_subtasks(subtasks_raw),
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

    def _normalize_subtask_specs(self, raw: Any) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        texts = self._normalize_subtasks(raw)
        for text in texts:
            specs.append(
                {
                    "node_type": "draft",
                    "subtask": {
                        "id": None,
                        "subtask_type": "draft",
                        "input": "None",
                        "expected_output": "None",
                        "description": text,
                    },
                    "description": text,
                }
            )

        # Prefer explicit structure when provided.
        iterable: list[Any]
        if isinstance(raw, dict):
            iterable = list(raw.values())
        elif isinstance(raw, list):
            iterable = raw
        else:
            iterable = []

        explicit_specs: list[dict[str, Any]] = []
        for item in iterable:
            if not isinstance(item, dict):
                continue

            node_type = str(item.get("node_type") or item.get("type") or item.get("subtask_type") or "draft").strip().lower()
            if node_type not in {"draft", "revise", "improve"}:
                node_type = "draft"
            subtask = str(item.get("subtask") or item.get("title") or item.get("name") or "").strip()
            description = str(
                item.get("description")
                or item.get("objective")
                or item.get("task")
                or ""
            ).strip()
            if not subtask:
                subtask = description
            if not subtask:
                continue
            if not description:
                description = subtask
            explicit_specs.append(
                {
                    "node_type": node_type,
                    "subtask": self._normalize_subtask_contract_item(item, fallback_description=description),
                    "description": description,
                }
            )

        if explicit_specs:
            specs = explicit_specs

        deduped: list[dict[str, Any]] = []
        seen = set()
        for spec in specs:
            key = f"{spec.get('node_type','draft').lower()}::{self._subtask_key(spec.get('subtask'))}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
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
                    "node_type": node.node_type,
                    "subtask": node.subtask,
                    "description": _safe_short(node.description, 4000),
                    "score": node.score,
                    "reward": node.reward,
                    "visits": node.visits,
                    "status": node.status,
                    "created_by": node.created_by,
                    "subtask_index": node.subtask_index,
                    "memory": _safe_short(node.memory, 4000),
                    "theoretician_output": _safe_short(node.theoretician_output, 12000),
                    "supervisor_dispatch": node.supervisor_dispatch,
                    "critic_feedback": node.critic_feedback,
                    "supervisor_feedback": node.supervisor_feedback,
                    "selected_round": node.selected_round,
                }
            )
        return node_list

    def _normalize_subtask_contract_item(self, item: dict[str, Any], fallback_description: str = "") -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "subtask_type": str(item.get("subtask_type") or item.get("type") or "draft"),
            "input": item.get("input") if item.get("input") is not None else "None",
            "expected_output": item.get("expected_output") if item.get("expected_output") is not None else "None",
            "description": str(item.get("description") or item.get("objective") or item.get("task") or fallback_description or "").strip(),
        }

    def _subtask_to_prompt_text(self, subtask: Any) -> str:
        if isinstance(subtask, (dict, list)):
            try:
                return json.dumps(subtask, ensure_ascii=False, indent=2)
            except Exception:
                return str(subtask)
        return "" if subtask is None else str(subtask)

    def _subtask_brief(self, subtask: Any) -> str:
        if isinstance(subtask, dict):
            sid = subtask.get("id")
            stype = subtask.get("subtask_type")
            desc = str(subtask.get("description") or "").strip()
            prefix = []
            if sid is not None:
                prefix.append(f"#{sid}")
            if stype:
                prefix.append(str(stype))
            if desc:
                prefix.append(desc)
            return " | ".join(prefix) if prefix else self._subtask_to_prompt_text(subtask)
        return self._subtask_to_prompt_text(subtask)

    def _subtask_key(self, subtask: Any) -> str:
        if isinstance(subtask, dict):
            sid = subtask.get("id")
            if sid is not None:
                return f"id::{sid}"
            try:
                return json.dumps(subtask, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(subtask).strip().lower()
        return str(subtask or "").strip().lower()

    def _get_contract_subtask_by_index(self, subtask_index: int, fallback: Any = None) -> Any:
        if 0 <= subtask_index < len(self.subtask_sequence):
            return self.subtask_sequence[subtask_index].get("subtask", fallback)
        return fallback

    def _node_workspace_path(self, node_id: str, agent) -> Path:
        base = Path(agent.session.config.workspace_path)
        m = re.match(r"^n(\d+)$", str(node_id).strip().lower())
        suffix = m.group(1).zfill(3) if m else re.sub(r"[^a-zA-Z0-9_-]+", "_", str(node_id)).strip("_") or "unknown"
        node_dir = base / f"node_{suffix}"
        node_dir.mkdir(parents=True, exist_ok=True)
        return node_dir

    def _save_clarifier_contract(self, contract: dict[str, Any], task_id: str) -> None:
        if not isinstance(contract, dict) or not contract:
            return

        paths: list[Path] = []
        if self.run_dir:
            paths.append(Path(self.run_dir) / "clarifier_contract.json")
            if task_id:
                paths.append(Path(self.run_dir) / "workspaces" / task_id / "clarifier_contract.json")

        try:
            workspace = self.clarifier_agent.session.config.workspace_path
            if workspace:
                p = Path(workspace) / "clarifier_contract.json"
                if p not in paths:
                    paths.append(p)
        except Exception:
            pass

        for path in paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                self.logger.warning("Failed to save clarifier contract to %s: %s", path, e)

    def _save_visualization(self, task_description: str, summary: str = "") -> Path | None:
        if not self.run_dir:
            return None
        output_path = Path(self.run_dir) / "visualization.html"
        nodes = self._serialize_nodes_for_visualization()
        subtasks: Any = []
        if isinstance(self.contract, dict):
            subtasks = self.contract.get("subtasks") or self.contract.get("sub_tasks") or []
        write_mcts_html(
            output_path,
            nodes=nodes,
            root_id=self.root_id,
            best_node_id=self.best_node_id,
            task_description=task_description,
            subtasks=subtasks,
            summary=summary,
        )
        return output_path

    def _save_summary_markdown(
        self,
        task_description: str,
        summary: str,
        best_path_payload: list[dict[str, Any]],
    ) -> Path | None:
        if not self.run_dir:
            return None
        output_path = Path(self.run_dir) / "summary.md"
        best_path_ids = [str(node.get("node_id", "")) for node in best_path_payload if node.get("node_id")]
        best_path_text = " -> ".join(best_path_ids) if best_path_ids else "(empty)"
        lines = [
            "# PHY Master Summary",
            "",
            "## Task",
            task_description or "",
            "",
            "## Best Path",
            best_path_text,
            "",
            "## Summary",
            summary or "",
            "",
        ]
        try:
            output_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            self.logger.warning("Failed to save summary markdown to %s: %s", output_path, e)
            return None
        return output_path


def _safe_short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]
