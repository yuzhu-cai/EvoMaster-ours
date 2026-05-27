"""Selfish-mining environment used for RICE real-world evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SelfishMiningConfig:
    alpha: float = 0.35
    gamma: float = 0.5
    max_private: int = 8
    horizon: int = 200


class SelfishMiningEnv(gym.Env):
    """Three-action blockchain selfish-mining MDP: adopt, reveal, mine."""

    metadata = {"render_modes": []}

    def __init__(self, config: SelfishMiningConfig | None = None):
        self.config = config or SelfishMiningConfig()
        self.observation_space = spaces.Box(low=0, high=self.config.max_private, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.rng = np.random.default_rng()
        self.private = 0
        self.public = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.private = 0
        self.public = 0
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        lead = self.private - self.public
        return np.asarray([self.private, self.public, lead], dtype=np.float32)

    def snapshot_state(self) -> dict[str, object]:
        """Return a simulator snapshot for fidelity perturbation rollouts."""
        return {
            "private": int(self.private),
            "public": int(self.public),
            "steps": int(self.steps),
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, state: np.ndarray | dict[str, object], *, steps: int | None = None):
        """Restore from either an observation vector or a snapshot dictionary."""
        if isinstance(state, dict):
            self.private = int(state["private"])
            self.public = int(state["public"])
            self.steps = int(state["steps"])
            self.rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        else:
            obs = np.asarray(state, dtype=np.float32).reshape(-1)
            self.private = int(round(float(obs[0])))
            self.public = int(round(float(obs[1])))
            if steps is not None:
                self.steps = int(steps)
        return self._obs()

    def step(self, action: int):
        cfg = self.config
        reward = 0.0
        if action == 0:  # adopt public chain
            self.private = 0
            self.public = 0
        elif action == 1:  # reveal private branch
            if self.private > self.public:
                reward += float(self.private)
                self.private = 0
                self.public = 0
            else:
                reward -= 0.1
        elif action == 2:  # mine next block
            if self.rng.random() < cfg.alpha:
                self.private = min(cfg.max_private, self.private + 1)
            else:
                self.public = min(cfg.max_private, self.public + 1)
                if self.public > self.private:
                    self.private = 0
                    self.public = 0
                    reward -= 1.0 - cfg.alpha
        self.steps += 1
        done = self.steps >= cfg.horizon
        return self._obs(), reward, done, False, {"private": self.private, "public": self.public}
