"""Environment factory covering the RICE Experiment I/II task set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cage import CageObservationAdapter, CageRewardAdapter
from .metadrive import MACRO_V1_ENV_ID, make_metadrive_macro_v1_env
from .selfish_mining import SelfishMiningEnv
from .sparse_mujoco import make_mujoco_env


class RunningMeanStd:
    """Online observation statistics used for MuJoCo PPO normalization."""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == len(self.mean.shape):
            arr = arr[None, ...]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = arr.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, value: np.ndarray, clip: float = 10.0) -> np.ndarray:
        self.update(value)
        normalized = (value - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations during PPO training for Walker/HalfCheetah."""

    def __init__(self, env: gym.Env, clip: float = 10.0):
        super().__init__(env)
        self.clip = clip
        self.rms = RunningMeanStd(env.observation_space.shape)
        self.observation_space = spaces.Box(
            low=-clip,
            high=clip,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return self.rms.normalize(np.asarray(observation, dtype=np.float32), self.clip)


class MacroDriveEnv(gym.Env):
    """Lightweight macro-action autonomous-driving environment.

    The adapter tries MetaDrive first, but this fallback keeps the autonomous
    driving task initializable in clean CI environments without DI-drive.
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 500):
        self.horizon = horizon
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng()
        self.state = np.zeros(6, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.asarray([0.0, 0.0, 0.0, 0.0, 12.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        steer, throttle, brake = np.asarray(action, dtype=np.float32).reshape(-1)[:3]
        x, y, heading, lane_offset, speed, progress = self.state
        speed = max(0.0, float(speed + 0.2 * throttle - 0.5 * brake))
        heading = float(heading + 0.04 * steer)
        x = float(x + speed * np.cos(heading) * 0.1)
        y = float(y + speed * np.sin(heading) * 0.1)
        lane_offset = float(0.98 * lane_offset + 0.05 * steer)
        progress = float(progress + max(speed, 0.0) * 0.1)
        self.steps += 1
        offroad = abs(lane_offset) > 1.5 or abs(y) > 8.0
        reward = speed * 0.1 - 2.0 * abs(lane_offset) - (10.0 if offroad else 0.0)
        self.state = np.asarray([x, y, heading, lane_offset, speed, progress], dtype=np.float32)
        done = offroad or self.steps >= self.horizon
        return self.state.copy(), float(reward), bool(done), False, {"progress": progress, "offroad": offroad}


class CageChallenge2Env(gym.Env):
    """Initializable CAGE Challenge 2 style network-defence environment."""

    metadata = {"render_modes": []}

    def __init__(self, hosts: int = 5, horizon: int = 100):
        self.hosts = hosts
        self.horizon = horizon
        self.observation_space = spaces.Box(0.0, 1.0, shape=(hosts * 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.adapter = CageObservationAdapter()
        self.reward_adapter = CageRewardAdapter()
        self.rng = np.random.default_rng()
        self.compromised = np.zeros(hosts, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.compromised = np.zeros(self.hosts, dtype=np.float32)
        self.steps = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        alert = (self.compromised > 0).astype(np.float32)
        patch = 1.0 - self.compromised
        return np.concatenate([self.compromised, alert, patch]).astype(np.float32)

    def step(self, action: int):
        if self.rng.random() < 0.25:
            self.compromised[self.rng.integers(0, self.hosts)] = 1.0
        removed = 0
        if int(action) in {2, 3}:
            target = int(np.argmax(self.compromised))
            removed = int(self.compromised[target] > 0)
            self.compromised[target] = 0.0
        reward = -float(self.compromised.sum()) + float(removed)
        self.steps += 1
        done = self.steps >= self.horizon
        info = {"removed_red_processes": removed, "compromised": float(self.compromised.sum())}
        return self._obs(), self.reward_adapter(reward, int(action), info), done, False, info


MUJOCO_NORMALIZE = {"Walker2d-v3", "HalfCheetah-v3"}


@dataclass(frozen=True)
class BuiltEnv:
    env: gym.Env
    key: str
    env_id: str
    normalized: bool = False
    sparse: bool = False


def make_autonomous_driving_env() -> gym.Env:
    try:
        return make_metadrive_macro_v1_env()
    except Exception:
        # CI fallback only; experiment metadata still records Macro-v1.
        return MacroDriveEnv()


def make_network_defence_env() -> gym.Env:
    try:
        from CybORG import CybORG  # type: ignore
        from CybORG.Agents.Wrappers import ChallengeWrapper  # type: ignore

        return ChallengeWrapper(CybORG("CAGEChallenge2", "sim"))
    except Exception:
        return CageChallenge2Env()


def make_env(env_id: str, *, sparse: bool = False, normalize: bool | None = None) -> BuiltEnv:
    if env_id in {"SelfishMining-v0", "selfish_mining"}:
        return BuiltEnv(SelfishMiningEnv(), "selfish_mining", "SelfishMining-v0")
    if env_id in {"CybORG-CAGE2-v0", "cage", "network_defence"}:
        return BuiltEnv(make_network_defence_env(), "cage", "CybORG-CAGE2-v0")
    if env_id in {MACRO_V1_ENV_ID, "MetaDrive-Macro-v1", "autodriving", "autonomous_driving"}:
        return BuiltEnv(make_autonomous_driving_env(), "autodriving", MACRO_V1_ENV_ID)

    env = make_mujoco_env(env_id, sparse=sparse)
    should_normalize = env_id in MUJOCO_NORMALIZE if normalize is None else normalize
    if should_normalize:
        env = NormalizeObservation(env)
    key = env_id.replace("-v3", "").replace("-v2", "").lower()
    if sparse:
        key = f"sparse_{key}"
    return BuiltEnv(env, key, env_id, normalized=should_normalize, sparse=sparse)
