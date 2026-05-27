"""Sparse MuJoCo wrappers matching the RICE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass
class SparseRewardSpec:
    x_threshold: float
    reward_scale: float = 1.0


class SparseXReward(gym.Wrapper):
    """Expose x-position reward only after a threshold."""

    def __init__(self, env: gym.Env, spec: SparseRewardSpec):
        super().__init__(env)
        self.spec = spec

    def step(self, action: Any):
        step = self.env.step(action)
        if len(step) == 5:
            obs, reward, terminated, truncated, info = step
        else:
            obs, reward, done, info = step
            terminated, truncated = bool(done), False
        x_position = float(info.get("x_position", info.get("x", 0.0)))
        sparse_reward = self.spec.reward_scale * x_position if x_position > self.spec.x_threshold else 0.0
        info = dict(info)
        info["dense_reward"] = reward
        info["sparse_x_position"] = x_position
        return obs, sparse_reward, terminated, truncated, info


def make_mujoco_env(gym_id: str, sparse: bool = False):
    env = gym.make(gym_id)
    if not sparse:
        return env
    threshold = 5.0 if "HalfCheetah" in gym_id else 0.6
    return SparseXReward(env, SparseRewardSpec(threshold))
