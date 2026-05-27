"""Experiment II matrix: refinement method x environment x explanation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from rice_reproduce.baselines.refining import (
    make_explanation_guided_ppo,
    make_statemask_r,
    select_explanation_method,
)
from rice_reproduce.baselines.jsrl import JSRLRefiner
from rice_reproduce.envs.factory import make_env
from rice_reproduce.mask_network import MaskNetwork
from rice_reproduce.policies import make_actor_critic_for_env
from rice_reproduce.refinement import RICERefiner, RefinementConfig


@dataclass(frozen=True)
class ExperimentIIRow:
    environment: str
    method: str
    explanation: str
    uses_optimized_statemask: bool
    config: dict


def make_explanation(name: str, mask: MaskNetwork) -> Callable[[np.ndarray], np.ndarray]:
    return select_explanation_method(name, mask)


def build_experiment_ii_runner(
    env_id: str,
    method: str,
    explanation: str = "ours",
    base: RefinementConfig | None = None,
) -> tuple[RICERefiner, ExperimentIIRow]:
    built = make_env(env_id)
    env = built.env
    policy = make_actor_critic_for_env(env, built.key)
    mask = MaskNetwork(int(np.prod(env.observation_space.shape)))
    importance = make_explanation(explanation, mask)
    cfg = base or RefinementConfig()

    if method == "rice":
        refiner = RICERefiner(env, policy, importance, cfg)
    elif method == "statemask_r":
        refiner = make_statemask_r(env, policy, importance, cfg)
    elif method == "ppo_finetune":
        refiner = make_explanation_guided_ppo(env, policy, importance, cfg)
    elif method == "jsrl":
        refiner = JSRLRefiner(env, built.key, guide_policy=policy, importance_fn=importance)
    else:
        raise ValueError(f"unknown Experiment II method: {method}")

    row = ExperimentIIRow(
        environment=built.key,
        method=method,
        explanation=explanation,
        uses_optimized_statemask=explanation == "ours",
        config=asdict(refiner.config) if hasattr(refiner, "config") else {"jsrl": True, "pi_e_initialized_from_pi_g": True},
    )
    return refiner, row


def materialize_experiment_ii_plan() -> list[dict]:
    envs = [
        "Hopper-v3",
        "Walker2d-v3",
        "Reacher-v2",
        "HalfCheetah-v3",
        "SelfishMining-v0",
        "CybORG-CAGE2-v0",
        "MetaDrive-Macro-v1",
    ]
    methods = ["rice", "statemask_r", "ppo_finetune", "jsrl"]
    rows: list[dict] = []
    for env_id in envs:
        for method in methods:
            for explanation in ["ours", "statemask", "random"]:
                rows.append(
                    {
                        "env_id": env_id,
                        "method": method,
                        "explanation": explanation,
                        "optimized_StateMask_Ours_used": explanation == "ours",
                    }
                )
    return rows
