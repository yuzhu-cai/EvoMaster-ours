"""CAGE Challenge adapter definitions for cyber-defense experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CageActionSpec:
    monitor: int = 0
    analyse: int = 1
    remove: int = 2
    restore: int = 3


class CageObservationAdapter:
    """Flatten host/process/service observations for MLP policies."""

    def __call__(self, observation: Any) -> np.ndarray:
        values: list[float] = []
        if isinstance(observation, dict):
            for key in sorted(observation):
                values.extend(self._flatten(observation[key]))
        else:
            values.extend(self._flatten(observation))
        return np.asarray(values, dtype=np.float32)

    def _flatten(self, item: Any) -> list[float]:
        if isinstance(item, dict):
            out: list[float] = []
            for key in sorted(item):
                out.extend(self._flatten(item[key]))
            return out
        if isinstance(item, (list, tuple)):
            out: list[float] = []
            for value in item:
                out.extend(self._flatten(value))
            return out
        if isinstance(item, (int, float, bool)):
            return [float(item)]
        return [float(abs(hash(str(item))) % 997) / 997.0]


class CageRewardAdapter:
    """Tracks negative restore cost and defense rewards from CybORG info."""

    def __call__(self, reward: float, action: int, info: dict[str, Any]) -> float:
        adjusted = float(reward)
        if action == CageActionSpec().restore:
            adjusted -= 1.0
        adjusted += float(info.get("removed_red_processes", 0))
        return adjusted
