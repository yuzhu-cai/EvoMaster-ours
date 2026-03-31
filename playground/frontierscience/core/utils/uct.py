"""Unified UCT/MCTS manager for FrontierScience."""

from __future__ import annotations

import json
import logging
import math
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..exp import DraftExp, ImproveExp, NodeExp
from .runtime import create_worker_agents, resolve_worker_workspace
from .shared_context import FrontierScienceSharedContext

logger = logging.getLogger(__name__)


@dataclass
class FrontierScienceMCTSConfig:
    num_drafts: int = 2
    max_depth: int = 2
    max_total_nodes: int | None = None
    max_iterations: int = 4
    max_children_per_node: int = 2
    exploration_weight: float = 1.414
    allow_repeat_improve: bool = False
    use_light_eval_for_intermediate: bool = True
    use_full_eval_for_final: bool = True

    @property
    def total_node_budget(self) -> int:
        raw = self.max_total_nodes if self.max_total_nodes is not None else self.max_iterations
        return max(int(raw), int(self.num_drafts))


@dataclass
class FrontierScienceSearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    action_type: str | None
    state_summary: str = ""
    trajectory: Any = None
    current_answer: str = ""
    answer_path: str | None = None
    visits: int = 0
    value_sum: float = 0.0
    latest_metric_result: dict[str, Any] | None = None
    children: list["FrontierScienceSearchNode"] = field(default_factory=list)
    parent: "FrontierScienceSearchNode | None" = field(default=None, repr=False)
    shared_tool_memory: str = ""
    expected_child_count: int = 0
    pending_nonrepeat_actions: set[str] = field(default_factory=set, repr=False)
    locked: bool = False
    is_terminal: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def uct_score(self, exploration_weight: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(parent_visits, 1)
        return self.mean_value + exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)

    def search_stats(self, exploration_weight: float | None = None) -> dict[str, Any]:
        parent_visits = self.parent.visits if self.parent is not None else None
        stats = {
            "depth": self.depth,
            "visits": self.visits,
            "value_sum": self.value_sum,
            "mean_value": self.mean_value,
            "parent_visits": parent_visits,
            "expected_child_count": self.expected_child_count,
            "locked": self.locked,
            "is_terminal": self.is_terminal,
        }
        if exploration_weight is not None and self.parent is not None:
            stats["uct"] = self.uct_score(exploration_weight, parent_visits or 1)
        else:
            stats["uct"] = None
        return stats

    def to_dict(self, exploration_weight: float | None = None) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "action_type": self.action_type,
            "state_summary": self.state_summary,
            "current_answer": self.current_answer or None,
            "answer_path": self.answer_path,
            **self.search_stats(exploration_weight),
            "latest_metric_result": self.latest_metric_result,
            "children": [child.node_id for child in self.children],
        }


class _MetricEvaluatorAdapter:
    def __init__(self, evaluator: Any, *, force_final: bool) -> None:
        self._evaluator = evaluator
        self._force_final = force_final

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["is_final"] = self._force_final
        return self._evaluator.evaluate(**kwargs)


class FrontierScienceUCTSearchManager:
    """Unified UCT tree manager plus FrontierScience runtime executor."""

    def __init__(
        self,
        *,
        config: FrontierScienceMCTSConfig | None = None,
        action_runner: Callable[[FrontierScienceSearchNode, str], dict[str, Any]] | None = None,
        metric_evaluator: Callable[[FrontierScienceSearchNode, bool], dict[str, Any]] | None = None,
        playground: Any | None = None,
        problem: str = "",
        task_id: str = "",
        task_type: str = "",
        task_input_data: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> None:
        self.playground = playground
        self.problem = problem
        self.task_id = task_id
        self.task_type = task_type
        self.task_input_data = task_input_data or {}
        self.images = images
        self.action_runner = action_runner
        self.metric_evaluator = metric_evaluator

        self.frontierscience_cfg: dict[str, Any] = {}
        self.reference_rubric = ""
        if config is not None:
            self.config = config
        elif playground is not None:
            from .config import build_mcts_config, get_frontierscience_cfg, reference_rubric

            self.frontierscience_cfg = get_frontierscience_cfg(playground.config)
            self.config = build_mcts_config(self.frontierscience_cfg)
            self.reference_rubric = reference_rubric(self.task_input_data)
        else:
            raise ValueError("Either config or playground must be provided")

        self.root = FrontierScienceSearchNode(
            node_id="root",
            parent_id=None,
            depth=0,
            action_type=None,
            state_summary="virtual_root",
        )
        self.best_node: FrontierScienceSearchNode | None = None
        self.all_nodes: list[FrontierScienceSearchNode] = [self.root]

    def run(self) -> dict[str, Any]:
        if self.playground is None:
            return self._run_local_search()
        return self._run_playground_search()

    def serialize_tree(self) -> list[dict[str, Any]]:
        return [node.to_dict(self.config.exploration_weight) for node in self.all_nodes]

    def available_actions(self, node: FrontierScienceSearchNode) -> list[str]:
        return self._available_actions(node)

    def has_expandable_frontier(self, node: FrontierScienceSearchNode | None = None) -> bool:
        target = node or self.root
        return self._has_expandable_frontier(target)

    def is_search_exhausted(self) -> bool:
        return not self.has_expandable_frontier(self.root)

    def attach_child(
        self,
        parent: FrontierScienceSearchNode,
        action_type: str,
        result: dict[str, Any],
    ) -> FrontierScienceSearchNode:
        self._complete_expansion(parent, action_type)
        child = FrontierScienceSearchNode(
            node_id=result.get("node_id") or uuid.uuid4().hex,
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            action_type=action_type,
            state_summary=str(result.get("state_summary", "")),
            trajectory=result.get("trajectory"),
            current_answer=str(result.get("current_answer", "")),
            answer_path=result.get("answer_path"),
            parent=parent,
            shared_tool_memory=str(result.get("shared_tool_memory", "")),
        )
        parent.children.append(child)
        parent.is_terminal = False
        self.all_nodes.append(child)
        return child

    def save_tree(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize_tree(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _run_local_search(self) -> dict[str, Any]:
        while len(self.all_nodes) - 1 < self.config.total_node_budget and not self.is_search_exhausted():
            selected = self._select(self.root)
            if selected.depth >= self.config.max_depth:
                metric_result = selected.latest_metric_result or self._evaluate_node_metric(
                    selected,
                    is_final=self._intermediate_eval_is_final(),
                )
                self._ingest_metric(selected, metric_result)
                selected.is_terminal = True
                continue

            child = self._expand(selected)
            if child is None:
                selected.is_terminal = not self._has_expandable_frontier(selected)
                if selected is self.root:
                    break
                self._backup(selected, selected.mean_value)
                continue

            metric_result = self._evaluate_node_metric(child, is_final=self._intermediate_eval_is_final())
            self._ingest_metric(child, metric_result)

        self._finalize_best_node()
        best = self.best_node or self._best_valid_node() or self.root
        return {
            "best_node": best,
            "best_answer": best.current_answer,
            "best_metric_result": best.latest_metric_result,
            "tree": self.serialize_tree(),
        }

    def _run_playground_search(self) -> dict[str, Any]:
        main_workspace = self._resolve_main_workspace()
        shared_context = self._build_shared_context(main_workspace)
        worker_agents_map = {
            index: create_worker_agents(self.playground, index)
            for index in range(self.playground.max_workers)
        }
        worker_workspaces = {
            index: resolve_worker_workspace(index, main_workspace, self.playground._parallel_config)
            for index in range(self.playground.max_workers)
        }
        for workspace in worker_workspaces.values():
            workspace.mkdir(parents=True, exist_ok=True)

        expansions_done = 0
        while expansions_done < self.config.total_node_budget and not self.is_search_exhausted():
            batch_size = min(self.playground.max_workers, self.config.total_node_budget - expansions_done)
            batch = self._choose_batch(batch_size)
            if not batch:
                break

            shared_snapshot = shared_context.render_for_prompt()
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                future_map = {
                    executor.submit(
                        self._run_node_job,
                        worker_agents=worker_agents_map[index % self.playground.max_workers],
                        node=node,
                        action_type=action_type,
                        shared_tool_memory=shared_snapshot,
                        worker_workspace=worker_workspaces[index % self.playground.max_workers],
                    ): (node, action_type)
                    for index, (node, action_type) in enumerate(batch)
                }

                for future in as_completed(future_map):
                    parent, action_type = future_map[future]
                    try:
                        result = future.result()
                    except Exception:
                        self._cancel_expansion(parent, action_type)
                        raise
                    child = self.attach_child(parent, action_type, result)
                    metric_result = result.get("metric_result") or {}
                    self._ingest_metric(child, metric_result)
                    shared_context.add_trajectory(
                        node_id=child.node_id,
                        parent_id=parent.node_id if parent else None,
                        action_type=action_type,
                        trajectory=child.trajectory,
                        metric_result=metric_result,
                        search_node=child,
                        exploration_weight=self.config.exploration_weight,
                    )
                    child.shared_tool_memory = shared_context.render_for_prompt()
                    expansions_done += 1

        self._finalize_best_node()
        tree_root = Path(self.playground.run_dir) if self.playground.run_dir else main_workspace
        tree_path = tree_root / "mcts_tree.json"
        self.save_tree(tree_path)
        best = self.best_node or self._best_valid_node() or self.root
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": "completed" if best.current_answer else "failed",
            "steps": expansions_done,
            "initial_answer": next(
                (
                    node.current_answer
                    for node in self.all_nodes
                    if node.action_type == "draft" and node.current_answer
                ),
                "",
            ),
            "refined_answer": best.current_answer,
            "final_answer": best.current_answer,
            "solve_trajectory": best.trajectory,
            "metric_result": best.latest_metric_result,
            "mcts_tree_path": str(tree_path),
            "shared_tool_memory": shared_context.render_for_prompt(),
            "tool_trace_manifest": str(shared_context.manifest_path),
            "best_answer_path": getattr(best, "answer_path", None),
        }

    def _select(self, node: FrontierScienceSearchNode) -> FrontierScienceSearchNode:
        current = node
        while current.depth < self.config.max_depth and not current.is_terminal:
            if self._available_actions(current):
                return current
            selected = self._uct_select(current)
            if selected is current:
                return current
            current = selected
        return current

    def _uct_select(self, node: FrontierScienceSearchNode) -> FrontierScienceSearchNode:
        eligible_children = [child for child in node.children if self._has_expandable_frontier(child)]
        if node is self.root:
            eligible_children = [child for child in eligible_children if not child.locked]
        if not eligible_children:
            return node
        return max(
            eligible_children,
            key=lambda child: child.uct_score(self.config.exploration_weight, node.visits),
        )

    def _expand(self, node: FrontierScienceSearchNode) -> FrontierScienceSearchNode | None:
        actions = self._available_actions(node)
        if not actions or self.action_runner is None:
            return None
        action_type = actions[0]
        self._reserve_expansion(node, action_type)
        try:
            result = self.action_runner(node, action_type)
        except Exception:
            self._cancel_expansion(node, action_type)
            raise
        return self.attach_child(node, action_type, result)

    def _backup(self, node: FrontierScienceSearchNode, reward: float) -> None:
        current: FrontierScienceSearchNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def _ingest_metric(self, node: FrontierScienceSearchNode, metric_result: dict[str, Any] | None) -> None:
        node.latest_metric_result = metric_result or {}
        reward = float((node.latest_metric_result or {}).get("overall_score", 0.0))
        self._backup(node, reward)
        self._maybe_update_best(node)

    def _maybe_update_best(self, node: FrontierScienceSearchNode) -> None:
        metric_result = node.latest_metric_result or {}
        if not bool(metric_result.get("is_valid", True)):
            if self.best_node is node:
                self.best_node = self._best_valid_node(exclude=node)
            return

        if self.best_node is None:
            self.best_node = node
            return

        best_score = self._metric_score(self.best_node.latest_metric_result)
        node_score = self._metric_score(metric_result)
        if node_score > best_score:
            self.best_node = node

    def _best_valid_node(
        self,
        *,
        exclude: FrontierScienceSearchNode | None = None,
    ) -> FrontierScienceSearchNode | None:
        candidates = [
            node
            for node in self.all_nodes
            if node is not self.root
            and node is not exclude
            and node.latest_metric_result
            and bool(node.latest_metric_result.get("is_valid", True))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda node: self._metric_score(node.latest_metric_result))

    def _finalize_best_node(self) -> None:
        if self.best_node is None or not self.config.use_full_eval_for_final:
            if self.best_node is None:
                self.best_node = self._best_valid_node()
            return

        final_metric = self._evaluate_node_metric(self.best_node, is_final=True)
        self.best_node.latest_metric_result = final_metric
        if not bool(final_metric.get("is_valid", True)):
            self.best_node = self._best_valid_node(exclude=self.best_node)
            return
        self._maybe_update_best(self.best_node)

    def _has_expandable_frontier(self, node: FrontierScienceSearchNode) -> bool:
        if node.is_terminal:
            return False
        if node.depth < self.config.max_depth and self._available_actions(node):
            return True
        return any(self._has_expandable_frontier(child) for child in node.children)

    def _available_actions(self, node: FrontierScienceSearchNode) -> list[str]:
        if node.is_terminal:
            return []
        if node is self.root:
            return ["draft"] if len(node.children) + node.expected_child_count < self.config.num_drafts else []
        if node.depth >= self.config.max_depth:
            return []
        if not node.current_answer.strip():
            return []
        if len(node.children) + node.expected_child_count >= self.config.max_children_per_node:
            return []

        actions: list[str] = []
        child_actions = [child.action_type for child in node.children]
        improve_reserved = "improve" in node.pending_nonrepeat_actions
        if self.config.allow_repeat_improve or ("improve" not in child_actions and not improve_reserved):
            actions.append("improve")
        return actions

    def _choose_batch(self, batch_size: int) -> list[tuple[FrontierScienceSearchNode, str]]:
        batch: list[tuple[FrontierScienceSearchNode, str]] = []
        remaining_budget = max(self.config.total_node_budget - (len(self.all_nodes) - 1), 0)
        target_batch_size = min(batch_size, remaining_budget)
        attempts = 0
        while len(batch) < target_batch_size and attempts < max(target_batch_size * 8, 1):
            attempts += 1
            node = self._select(self.root)
            actions = self._available_actions(node)
            if not actions:
                if node is self.root:
                    break
                node.is_terminal = not self._has_expandable_frontier(node)
                continue
            action_type = actions[0]
            self._reserve_expansion(node, action_type)
            batch.append((node, action_type))
        return batch

    def _reserve_expansion(self, node: FrontierScienceSearchNode, action_type: str) -> None:
        node.expected_child_count += 1
        if not self.config.allow_repeat_improve:
            node.pending_nonrepeat_actions.add(action_type)
        branch = self._root_branch(node)
        if branch is not None:
            branch.locked = True

    def _complete_expansion(self, node: FrontierScienceSearchNode, action_type: str) -> None:
        node.expected_child_count = max(node.expected_child_count - 1, 0)
        node.pending_nonrepeat_actions.discard(action_type)
        branch = self._root_branch(node)
        if branch is not None:
            branch.locked = False

    def _cancel_expansion(self, node: FrontierScienceSearchNode, action_type: str) -> None:
        self._complete_expansion(node, action_type)

    def _root_branch(self, node: FrontierScienceSearchNode) -> FrontierScienceSearchNode | None:
        current = node
        while current.parent is not None and current.parent is not self.root:
            current = current.parent
        return current if current.parent is self.root else None

    def _intermediate_eval_is_final(self) -> bool:
        return not self.config.use_light_eval_for_intermediate

    def _evaluate_node_metric(self, node: FrontierScienceSearchNode, *, is_final: bool) -> dict[str, Any]:
        if self.playground is None:
            if self.metric_evaluator is None:
                return node.latest_metric_result or {}
            return self.metric_evaluator(node, is_final)

        evaluator = self._build_metric_evaluator(force_final=is_final)
        return evaluator.evaluate(
            problem=self.problem,
            final_answer=self._load_metric_answer(node),
            trajectory=node.trajectory,
            previous_answer=self._load_metric_answer(node.parent) if node.parent and node.parent is not self.root else "",
            reference_rubric=self.reference_rubric,
            action_type=node.action_type or "draft",
            is_final=is_final,
        )

    def _resolve_main_workspace(self) -> Path:
        workspace_path = getattr(self.playground.session.config, "workspace_path", None)
        if not workspace_path:
            workspace_path = self.playground.session.get_workspace_path()
        if not workspace_path:
            raise ValueError("Session workspace_path is not available")
        main_workspace = Path(workspace_path)
        main_workspace.mkdir(parents=True, exist_ok=True)
        return main_workspace

    def _build_shared_context(self, main_workspace: Path) -> FrontierScienceSharedContext:
        run_root = Path(self.playground.run_dir) if self.playground.run_dir else main_workspace
        return FrontierScienceSharedContext(
            run_root / "tool_trace",
            trajectory_dir=run_root / "trajectories",
        )

    def _build_metric_evaluator(self, metric_agent: Any | None = None, *, force_final: bool = False) -> Any:
        from .config import build_metric_evaluator

        base_evaluator = build_metric_evaluator(
            self.frontierscience_cfg,
            metric_agent=metric_agent or self.playground.agents.get("metric_agent"),
        )
        return _MetricEvaluatorAdapter(base_evaluator, force_final=force_final)

    def _run_node_job(
        self,
        *,
        worker_agents: dict[str, Any],
        node: FrontierScienceSearchNode,
        action_type: str,
        shared_tool_memory: str,
        worker_workspace: Path,
    ) -> dict[str, Any]:
        node_run_id = uuid.uuid4().hex[:8]
        solution_filename = f"solution_{action_type}_{node.node_id}_{node_run_id}.md"
        solution_path = worker_workspace / solution_filename
        metric_evaluator = self._build_metric_evaluator(
            worker_agents.get("metric"),
            force_final=self._intermediate_eval_is_final(),
        )
        if action_type == "draft":
            exp = DraftExp(worker_agents["draft"], metric_evaluator=metric_evaluator)
            result = exp.run(
                problem=self.problem,
                task_id=f"{self.task_id}_{node_run_id}",
                task_input_data=self.task_input_data,
                reference_rubric=self.reference_rubric,
                trajectory_summary=node.state_summary,
                shared_tool_memory=shared_tool_memory,
                worker_workspace=str(worker_workspace),
                solution_path=str(solution_path),
                solution_filename=solution_filename,
                images=self.images,
            )
        else:
            exp = ImproveExp(worker_agents["improve"], metric_evaluator=metric_evaluator)
            result = exp.run(
                problem=self.problem,
                task_id=f"{self.task_id}_{node_run_id}",
                task_input_data=self.task_input_data,
                reference_rubric=self.reference_rubric,
                trajectory_summary=node.state_summary,
                existing_answer=node.current_answer,
                improvement_focus=self._resolve_improvement_focus(node),
                shared_tool_memory=shared_tool_memory,
                worker_workspace=str(worker_workspace),
                solution_path=str(solution_path),
                solution_filename=solution_filename,
                images=self.images,
            )
        result["shared_tool_memory"] = shared_tool_memory
        result["worker_workspace"] = str(worker_workspace)
        result["solution_path"] = str(solution_path)
        return result

    @staticmethod
    def _resolve_improvement_focus(node: Any) -> str:
        improvement_focus = "Strengthen evidence grounding, completeness, and scientific precision."
        latest_metric_result = getattr(node, "latest_metric_result", None) or {}
        suggestions = latest_metric_result.get("improvement_suggestions") or []
        weaknesses = latest_metric_result.get("weaknesses") or []
        return "\n".join([*map(str, weaknesses[:2]), *map(str, suggestions[:3])]).strip() or improvement_focus

    @staticmethod
    def _metric_score(metric_result: dict[str, Any] | None) -> float:
        if not metric_result:
            return -1.0
        return float(metric_result.get("overall_score", -1.0))

    @staticmethod
    def _load_metric_answer(node: FrontierScienceSearchNode | None) -> str:
        if node is None:
            return ""
        return NodeExp._load_metric_answer(node.answer_path or "", node.current_answer)


class FrontierScienceMCTS(FrontierScienceUCTSearchManager):
    """Backward-compatible alias for the unified UCT manager."""


class FrontierScienceSearchRunner(FrontierScienceUCTSearchManager):
    """Backward-compatible alias for the unified UCT manager."""

