"""Experiment grids and hyper-parameters from the RICE paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    key: str
    gym_id: str
    p: float
    rnd_lambda: float
    alpha: float = 1e-4
    normalize_observation: bool = False
    sparse_threshold: float | None = None


MUPAPER_ENVS = {
    "hopper": EnvSpec("hopper", "Hopper-v3", 0.25, 0.001),
    "walker2d": EnvSpec("walker2d", "Walker2d-v3", 0.25, 0.01, normalize_observation=True),
    "reacher": EnvSpec("reacher", "Reacher-v2", 0.50, 0.001),
    "halfcheetah": EnvSpec("halfcheetah", "HalfCheetah-v3", 0.50, 0.01, normalize_observation=True),
}

REAL_WORLD_ENVS = {
    "selfish_mining": EnvSpec("selfish_mining", "SelfishMining-v0", 0.25, 0.001),
    "cage": EnvSpec("cage", "CybORG-CAGE2-v0", 0.50, 0.01),
    "autodriving": EnvSpec("autodriving", "MetaDrive-Macro-v1", 0.25, 0.01),
}

FIDELITY_K = [0.10, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]
ALPHA_GRID = [1e-2, 1e-3, 1e-4]
P_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
LAMBDA_GRID = [0.0, 0.1, 0.01, 0.001]

TABLE1_REFINERS = ["ppo_finetune", "jsrl", "statemask_r", "rice"]
TABLE1_EXPLANATIONS = ["random", "statemask", "ours"]
