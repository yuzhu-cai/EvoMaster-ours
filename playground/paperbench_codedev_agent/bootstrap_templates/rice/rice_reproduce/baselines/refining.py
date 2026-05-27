"""Baseline refining methods compared with RICE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rice_reproduce.critical_states import random_critical_state, select_critical_state
from rice_reproduce.mask_network import MaskNetwork
from rice_reproduce.refinement import RICERefiner, RefinementConfig
from rice_reproduce.statemask import OriginalStateMaskExplanation
from rice_reproduce.types import Trajectory


@dataclass
class BaselineSpec:
    name: str
    description: str


BASELINES = {
    "ppo_finetune": BaselineSpec("PPO fine-tuning", "Continue PPO from the default initial distribution."),
    "statemask_r": BaselineSpec("StateMask-R", "Always reset to critical states selected by the explanation."),
    "jsrl": BaselineSpec("JSRL", "Roll in with a guide policy then train an exploration policy."),
    "random_explanation_rice": BaselineSpec("RICE with random explanation", "Use the RICE refiner with randomly chosen visited states."),
}


def make_ppo_finetune(env, policy, importance: Callable[[np.ndarray], np.ndarray], base: RefinementConfig) -> RICERefiner:
    cfg = RefinementConfig(
        **{
            **base.__dict__,
            "p_reset_to_critical": 0.0,
            "rnd_lambda": 0.0,
            "policy_learning_rate": min(base.policy_learning_rate, 3e-5),
        }
    )
    return RICERefiner(env, policy, importance, cfg)


def make_statemask_r(env, policy, importance: Callable[[np.ndarray], np.ndarray], base: RefinementConfig) -> RICERefiner:
    cfg = RefinementConfig(**{**base.__dict__, "p_reset_to_critical": 1.0, "rnd_lambda": 0.0})
    return RICERefiner(env, policy, importance, cfg)


def make_explanation_guided_ppo(
    env,
    policy,
    ours_importance: Callable[[np.ndarray], np.ndarray],
    base: RefinementConfig,
) -> RICERefiner:
    """Experiment II PPO fine-tuning row using the optimized StateMask explanation."""

    return make_ppo_finetune(env, policy, ours_importance, base)


def make_explanation_guided_jsrl_config(base: RefinementConfig) -> RefinementConfig:
    """JSRL uses the same optimized StateMask critical states for roll-in starts."""

    return RefinementConfig(
        **{
            **base.__dict__,
            "p_reset_to_critical": max(base.p_reset_to_critical, 0.25),
            "rnd_lambda": 0.0,
        }
    )


def random_importance(states: np.ndarray) -> np.ndarray:
    return np.random.default_rng().random(len(states)).astype(np.float32)


def select_explanation_method(name: str, mask: MaskNetwork) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve Experiment II explanation choices used during retraining.

    The original StateMask explanation can be selected with `name="statemask"`;
    the paper's optimized StateMask/RICE explanation is `name="ours"`.
    """
    if name == "ours":
        return mask.importance
    if name == "statemask":
        return OriginalStateMaskExplanation(mask).importance
    if name == "random":
        return random_importance
    raise ValueError(f"unknown explanation method: {name}")


def critical_from_method(trajectory: Trajectory, method: str, importance: np.ndarray):
    if method == "random":
        return random_critical_state(trajectory)
    return select_critical_state(trajectory, importance)
