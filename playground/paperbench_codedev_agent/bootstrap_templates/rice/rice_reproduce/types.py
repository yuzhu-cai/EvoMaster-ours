"""Shared data structures for trajectory and rollout handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class Transition:
    state: Array
    action: Array | int
    reward: float
    next_state: Array
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    mask_action: int | None = None
    log_prob: float | None = None
    value: float | None = None
    intrinsic_reward: float = 0.0


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    @property
    def states(self) -> Array:
        return np.asarray([t.state for t in self.transitions])

    @property
    def actions(self) -> Array:
        return np.asarray([t.action for t in self.transitions])

    @property
    def rewards(self) -> Array:
        return np.asarray([t.reward for t in self.transitions], dtype=np.float32)

    @property
    def next_states(self) -> Array:
        return np.asarray([t.next_state for t in self.transitions])

    @property
    def dones(self) -> Array:
        return np.asarray([t.done for t in self.transitions], dtype=np.float32)

    def total_reward(self) -> float:
        return float(sum(t.reward for t in self.transitions))

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class RolloutBatch:
    states: Array
    actions: Array
    rewards: Array
    next_states: Array
    dones: Array
    log_probs: Array | None = None
    values: Array | None = None
    advantages: Array | None = None
    returns: Array | None = None


def discounted_returns(rewards: Array, dones: Array, gamma: float) -> Array:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running * (1.0 - float(dones[i]))
        out[i] = running
    return out


def generalized_advantage_estimate(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * nonterminal - float(values[t])
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[t] = last_advantage
        next_value = float(values[t])
    returns = advantages + values.astype(np.float32)
    return advantages, returns
