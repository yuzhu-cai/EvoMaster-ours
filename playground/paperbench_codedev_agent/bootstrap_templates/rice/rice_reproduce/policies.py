"""Policy adapters used by explanation and refinement code."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


class PolicyLike(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Any: ...


class SB3PolicyAdapter:
    """Adapter for Stable-Baselines3 PPO/SAC policies."""

    def __init__(self, model: Any, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic

    def act(self, observation: np.ndarray) -> np.ndarray | int:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Any:
        return self.model.predict(observation, deterministic=deterministic or self.deterministic)


class MLPActorCritic(nn.Module):
    """Small actor-critic network compatible with PPO updates in this repo."""

    def __init__(self, obs_dim: int, action_dim: int, discrete: bool, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.Tanh()])
            last = width
        self.backbone = nn.Sequential(*layers)
        self.discrete = discrete
        self.actor = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor):
        h = self.backbone(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=self.actor(h))
        mean = self.actor(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(obs)).squeeze(-1)

    def act(self, observation: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.distribution(obs)
            action = dist.probs.argmax(-1) if deterministic and self.discrete else dist.sample()
            if deterministic and not self.discrete:
                action = dist.mean
        action_np = action.squeeze(0).cpu().numpy()
        return int(action_np) if self.discrete else action_np


def sample_random_action(env: Any) -> Any:
    return env.action_space.sample()


def default_hidden_sizes(env_key: str) -> tuple[int, ...]:
    if env_key in {"selfish_mining", "SelfishMining-v0"}:
        return (128, 128, 128, 128)
    return (64, 64)


def make_actor_critic_for_env(env: Any, env_key: str = "") -> MLPActorCritic:
    obs_dim = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, "n")
    action_dim = int(env.action_space.n if discrete else env.action_space.shape[0])
    return MLPActorCritic(obs_dim, action_dim, discrete, hidden_sizes=default_hidden_sizes(env_key))
