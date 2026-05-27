"""MetaDrive Macro-v1 adapter for autonomous-driving experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


MACRO_V1_ENV_ID = "Macro-v1"


@dataclass
class DriveAction:
    steering: float
    acceleration: float
    brake: float


def normalized_to_drive(action: np.ndarray) -> DriveAction:
    action = np.asarray(action, dtype=np.float32)
    steering = float(np.clip(action[0], -1.0, 1.0))
    throttle = float(np.clip(action[1], -1.0, 1.0))
    return DriveAction(steering=steering, acceleration=max(throttle, 0.0), brake=max(-throttle, 0.0))


def vectorize_metadrive_observation(obs) -> np.ndarray:
    if isinstance(obs, dict):
        chunks = []
        for key in sorted(obs):
            chunks.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class MetaDriveMacroV1Env:
    """Adapter for DI-drive/MetaDrive's exact `Macro-v1` task.

    RICE Appendix C.2 uses the Macro-v1 environment powered by MetaDrive, with
    normalized two-dimensional actions converted to steering, acceleration, and
    brake commands. This wrapper keeps that exact environment id explicit while
    presenting a Gymnasium-like API to the rest of the reproduction code.
    """

    env_id = MACRO_V1_ENV_ID

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {
            "use_render": False,
            "force_render": False,
            "horizon": 500,
            "vehicle_config": {"lidar": {"num_lasers": 72}},
            "agent_policy": None,
            **(config or {}),
        }
        self.env = self._build_macro_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _build_macro_env(self):
        import metadrive  # noqa: F401  # registers MetaDrive environments
        import gymnasium as gym

        try:
            return gym.make(MACRO_V1_ENV_ID, config=self.config)
        except Exception:
            # Some DI-drive installs register the task through classic gym.
            import gym as classic_gym  # type: ignore

            return classic_gym.make(MACRO_V1_ENV_ID, config=self.config)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        drive = normalized_to_drive(np.asarray(action, dtype=np.float32))
        command = np.asarray([drive.steering, drive.acceleration, drive.brake], dtype=np.float32)
        return self.env.step(command)

    def close(self):
        return self.env.close()


def make_metadrive_macro_v1_env(config: dict[str, Any] | None = None):
    """Construct the paper's Macro-v1 MetaDrive environment by exact id."""
    return MetaDriveMacroV1Env(config=config)
