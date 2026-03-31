"""Configuration helpers for FrontierScience playground."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .uct import FrontierScienceMCTSConfig
from .metric import FrontierScienceMetricEvaluator, default_metric_weights


def get_frontierscience_cfg(config: Any) -> dict[str, Any]:
    """Read FrontierScience config safely from the playground config object."""
    return getattr(config, "frontierscience", {}) or {}


def resolve_mcts_limits(search_cfg: dict[str, Any]) -> tuple[int, int]:
    """Keep legacy max_iterations support while respecting user draft count."""
    num_drafts = max(1, int(search_cfg.get("num_drafts", 2)))
    max_total_nodes = max(int(search_cfg.get("max_total_nodes", search_cfg.get("max_iterations", 4))), num_drafts)
    return num_drafts, max_total_nodes


def reference_rubric(task_input_data: dict[str, Any]) -> str:
    """Resolve the scoring rubric used by the metric evaluator."""
    explicit = str(task_input_data.get("reference_rubric", "")).strip()
    if explicit:
        return explicit
    return (
        "Use a FrontierScience-style multi-dimensional rubric: partial credit by item, "
        "strong emphasis on evidence grounding, completeness, scientific correctness, formulas or derivations when needed, "
        "and explicit coverage of every requested sub-question."
    )


def build_metric_evaluator(
    frontierscience_cfg: dict[str, Any],
    metric_agent: Any | None = None,
) -> FrontierScienceMetricEvaluator:
    """Build the rubric evaluator from config and agent."""
    metric_cfg = frontierscience_cfg.get("metric", {}) or {}
    weights = deepcopy(metric_cfg.get("weights") or default_metric_weights())
    return FrontierScienceMetricEvaluator(metric_agent=metric_agent, weights=weights)


def build_mcts_config(frontierscience_cfg: dict[str, Any]) -> FrontierScienceMCTSConfig:
    """Build MCTS config from the FrontierScience section."""
    search_cfg = frontierscience_cfg.get("mcts", {}) or {}
    num_drafts, max_total_nodes = resolve_mcts_limits(search_cfg)
    return FrontierScienceMCTSConfig(
        num_drafts=num_drafts,
        max_depth=int(search_cfg.get("max_depth", 2)),
        max_total_nodes=max_total_nodes,
        max_iterations=max_total_nodes,
        max_children_per_node=int(search_cfg.get("max_children_per_node", 2)),
        exploration_weight=float(search_cfg.get("exploration_weight", 1.414)),
        allow_repeat_improve=bool(search_cfg.get("allow_repeat_improve", False)),
        use_light_eval_for_intermediate=bool(search_cfg.get("use_light_eval_for_intermediate", True)),
        use_full_eval_for_final=bool(search_cfg.get("use_full_eval_for_final", True)),
    )
