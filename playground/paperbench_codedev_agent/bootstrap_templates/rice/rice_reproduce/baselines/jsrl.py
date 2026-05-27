"""Jump-Start Reinforcement Learning style curriculum for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Callable

import numpy as np

from rice_reproduce.critical_states import select_critical_state
from rice_reproduce.pretraining import PPOPretrainer, PretrainConfig, PretrainReport
from rice_reproduce.types import Trajectory, Transition


@dataclass
class JSRLCurriculum:
    guide_steps_start: int = 100
    guide_steps_end: int = 0
    total_iterations: int = 100

    def guide_steps(self, iteration: int) -> int:
        frac = min(max(iteration / max(self.total_iterations - 1, 1), 0.0), 1.0)
        return int(round((1.0 - frac) * self.guide_steps_start + frac * self.guide_steps_end))


def jsrl_rollin(env, guide_policy: Callable[[np.ndarray], object], steps: int):
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    for _ in range(steps):
        step = env.step(guide_policy(obs))
        if len(step) == 5:
            obs, _reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            obs, _reward, done, _info = step
        if done:
            reset = env.reset()
            obs = reset[0] if isinstance(reset, tuple) else reset
            break
    return obs


class ExplanationGuidedJSRL:
    """JSRL variant whose guide starts are chosen by an explanation method."""

    def __init__(self, guide_policy: Callable[[np.ndarray], object], importance_fn: Callable[[np.ndarray], np.ndarray]):
        self.guide_policy = guide_policy
        self.importance_fn = importance_fn

    def collect_guide_trajectory(self, env, steps: int) -> Trajectory:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        trajectory = Trajectory()
        for _ in range(steps):
            state = np.asarray(obs, dtype=np.float32)
            action = self.guide_policy(state)
            step = env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step
            trajectory.append(Transition(state, action, float(reward), np.asarray(next_obs, dtype=np.float32), bool(done), info))
            obs = next_obs
            if done:
                break
        return trajectory

    def explanation_start_state(self, env, steps: int = 100) -> np.ndarray:
        trajectory = self.collect_guide_trajectory(env, steps)
        if len(trajectory) == 0:
            reset = env.reset()
            return reset[0] if isinstance(reset, tuple) else reset
        scores = self.importance_fn(trajectory.states)
        return select_critical_state(trajectory, scores).state


class JSRLRefiner:
    """Jump-Start RL baseline with pi_e initialized from guide policy pi_g."""

    def __init__(
        self,
        env,
        env_key: str,
        guide_policy,
        importance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        curriculum: JSRLCurriculum | None = None,
        config: PretrainConfig | None = None,
    ):
        self.env = env
        self.env_key = env_key
        self.guide_policy = guide_policy
        self.exploration_policy = deepcopy(guide_policy)
        self.importance_fn = importance_fn
        self.curriculum = curriculum or JSRLCurriculum()
        self.trainer = PPOPretrainer(env, env_key, config or PretrainConfig())
        self.trainer.policy.load_state_dict(self.exploration_policy.state_dict())

    def initialize_exploration_from_guide(self) -> None:
        self.exploration_policy.load_state_dict(self.guide_policy.state_dict())
        self.trainer.policy.load_state_dict(self.guide_policy.state_dict())

    def refine(self, iterations: int) -> PretrainReport:
        self.initialize_exploration_from_guide()
        report = PretrainReport()
        for iteration in range(iterations):
            guide_steps = self.curriculum.guide_steps(iteration)
            if guide_steps > 0:
                jsrl_rollin(self.env, lambda obs: self.guide_policy.act(obs, deterministic=True), guide_steps)
            batch = self.trainer.collect()
            loss = self.trainer.update(batch)
            reward = float(np.sum(batch.rewards))
            prior = report.cumulative_reward[-1] if report.cumulative_reward else 0.0
            report.update_reward.append(reward)
            report.cumulative_reward.append(prior + reward)
            report.losses.append({**loss, "guide_steps": float(guide_steps)})
        self.exploration_policy.load_state_dict(self.trainer.policy.state_dict())
        return report
