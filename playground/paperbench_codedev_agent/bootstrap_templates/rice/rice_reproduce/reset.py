"""Simulator state restoration helpers for RICE roll-in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ResetSnapshot:
    state: np.ndarray
    env_state: Any | None = None
    prefix_actions: list[Any] | None = None
    index: int = 0


class ResettableEnvWrapper:
    """Adds best-effort restore-to-critical-state support to simulator envs."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def capture(self, state: np.ndarray, index: int, prefix_actions: list[Any] | None = None) -> ResetSnapshot:
        env_state = None
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "snapshot_state"):
            env_state = unwrapped.snapshot_state()
        sim = getattr(unwrapped, "sim", None)
        if env_state is None and sim is not None and hasattr(sim, "get_state"):
            env_state = sim.get_state()
        elif env_state is None and hasattr(unwrapped, "state"):
            env_state = np.asarray(unwrapped.state).copy()
        return ResetSnapshot(np.asarray(state).copy(), env_state, list(prefix_actions or []), index)

    def restore(self, snapshot: ResetSnapshot):
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if snapshot.env_state is not None:
            if hasattr(unwrapped, "restore_state"):
                return unwrapped.restore_state(snapshot.env_state)
            sim = getattr(unwrapped, "sim", None)
            if sim is not None and hasattr(sim, "set_state"):
                sim.set_state(snapshot.env_state)
                if hasattr(sim, "forward"):
                    sim.forward()
                return snapshot.state.copy()
            if hasattr(unwrapped, "state"):
                unwrapped.state = np.asarray(snapshot.env_state).copy()
                return snapshot.state.copy()
        if hasattr(unwrapped, "restore_state"):
            return unwrapped.restore_state(snapshot.state, steps=snapshot.index)
        reset = self.env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        for action in snapshot.prefix_actions or []:
            step = self.env.step(action)
            obs = step[0]
        return obs


def replay_reset(env: Any, state: np.ndarray, _index: int) -> np.ndarray:
    reset = env.reset()
    return reset[0] if isinstance(reset, tuple) else reset


def make_reset_callback(wrapper: ResettableEnvWrapper, snapshot: ResetSnapshot) -> Callable[[Any, np.ndarray, int], np.ndarray]:
    def _restore(_env: Any, _state: np.ndarray, _index: int) -> np.ndarray:
        return wrapper.restore(snapshot)
    return _restore
