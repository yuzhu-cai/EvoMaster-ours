"""Critical-state identification and fidelity scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .types import Trajectory


@dataclass(frozen=True)
class CriticalState:
    index: int
    state: np.ndarray
    score: float
    window_start: int
    window_end: int


def sliding_window_scores(importance: Sequence[float], fraction: float) -> np.ndarray:
    values = np.asarray(importance, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("importance must be one-dimensional")
    if len(values) == 0:
        return np.asarray([], dtype=np.float32)
    width = max(1, int(round(len(values) * fraction)))
    cumsum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    means = (cumsum[width:] - cumsum[:-width]) / width
    return means.astype(np.float32)


def select_critical_state(trajectory: Trajectory, importance: Sequence[float], fraction: float = 0.1) -> CriticalState:
    if len(trajectory) == 0:
        raise ValueError("trajectory is empty")
    means = sliding_window_scores(importance, fraction)
    if len(means) == 0:
        idx = int(np.argmax(np.asarray(importance, dtype=np.float32)))
        return CriticalState(idx, trajectory.transitions[idx].state, float(importance[idx]), idx, idx + 1)
    start = int(np.argmax(means))
    width = max(1, int(round(len(trajectory) * fraction)))
    local_scores = np.asarray(importance[start:start + width], dtype=np.float32)
    local_idx = int(np.argmax(local_scores))
    idx = start + local_idx
    return CriticalState(idx, trajectory.transitions[idx].state, float(importance[idx]), start, start + width)


def random_critical_state(trajectory: Trajectory, rng: np.random.Generator | None = None) -> CriticalState:
    rng = rng or np.random.default_rng()
    idx = int(rng.integers(0, len(trajectory)))
    return CriticalState(idx, trajectory.transitions[idx].state, 0.0, idx, idx + 1)


def fidelity_score(reward_delta: float, max_reward_delta: float, window_fraction: float, eps: float = 1e-8) -> float:
    d = max(abs(float(reward_delta)), eps)
    dmax = max(abs(float(max_reward_delta)), eps)
    frac = max(float(window_fraction), eps)
    return float(np.log(d / dmax) - np.log(frac))


def perturb_segment_return(
    env,
    trajectory: Trajectory,
    policy: Callable[[np.ndarray], object],
    start: int,
    end: int,
    reset_to_state: Callable[[object, np.ndarray, int], np.ndarray],
    max_steps: int | None = None,
) -> float:
    """Evaluate reward after random actions in a critical segment.

    The reset_to_state callback restores the simulator to trajectory[start].state.
    This captures the addendum fidelity protocol: fast-forward, randomize the
    selected segment, then let the policy finish the episode.
    """
    obs = reset_to_state(env, trajectory.transitions[start].state, start)
    total = 0.0
    horizon = max_steps or len(trajectory)
    for t in range(start, horizon):
        action = env.action_space.sample() if t < end else policy(obs)
        step = env.step(action)
        if len(step) == 5:
            obs, reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            obs, reward, done, _info = step
        total += float(reward)
        if done:
            break
    return total


def topk_indices(importance: Sequence[float], k_fraction: float) -> list[int]:
    values = np.asarray(importance, dtype=np.float32)
    k = max(1, int(round(len(values) * k_fraction)))
    return list(np.argsort(values)[-k:][::-1])
