"""Original StateMask explanation baseline with primal-dual optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mask_network import MaskNetwork
from .policies import sample_random_action
from .types import Trajectory, Transition


def _reset_observation(env: Any) -> Any:
    reset = env.reset()
    return reset[0] if isinstance(reset, tuple) else reset


def _step_env(env: Any, action: Any):
    step = env.step(action)
    if len(step) == 5:
        obs, reward, terminated, truncated, info = step
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = step
    return obs, float(reward), bool(done), info


@dataclass
class StateMaskConfig:
    """Hyper-parameters for the original StateMask objective."""

    learning_rate: float = 3e-4
    dual_learning_rate: float = 1e-2
    mask_budget: float = 0.25
    entropy_coef: float = 1e-3
    return_gap_coef: float = 1.0
    horizon: int = 1000
    batch_episodes: int = 4


@dataclass
class StateMaskHistory:
    return_gap: list[float] = field(default_factory=list)
    target_return: list[float] = field(default_factory=list)
    perturbed_return: list[float] = field(default_factory=list)
    mask_rate: list[float] = field(default_factory=list)
    dual_value: list[float] = field(default_factory=list)
    cumulative_reward: list[float] = field(default_factory=list)
    training_time_seconds: list[float] = field(default_factory=list)


class OriginalStateMaskExplanation:
    """Inference-time StateMask explainer selectable for rollouts/retraining."""

    def __init__(self, mask_network: MaskNetwork, threshold: float = 0.5):
        self.mask_network = mask_network
        self.threshold = threshold

    def importance(self, states: np.ndarray) -> np.ndarray:
        return self.mask_network.importance(states)

    def mask_decision(self, observation: np.ndarray) -> int:
        score = float(self.importance(np.asarray([observation], dtype=np.float32))[0])
        return 0 if score >= self.threshold else 1

    def generate_perturbed_rollout(
        self,
        env: Any,
        target_policy: Callable[[np.ndarray], Any],
        horizon: int,
    ) -> Trajectory:
        obs = _reset_observation(env)
        trajectory = Trajectory()
        for _ in range(horizon):
            state = np.asarray(obs, dtype=np.float32)
            action = sample_random_action(env) if self.mask_decision(state) == 1 else target_policy(state)
            next_obs, reward, done, info = _step_env(env, action)
            trajectory.append(Transition(state, action, reward, np.asarray(next_obs, dtype=np.float32), done, info))
            obs = _reset_observation(env) if done else next_obs
            if done:
                break
        return trajectory


class StateMaskPrimalDualTrainer:
    """Train StateMask with J(theta)=min |eta(pi)-eta(pi_bar)| and a dual mask-budget constraint."""

    def __init__(self, mask_network: MaskNetwork, config: StateMaskConfig | None = None, device: str = "cpu"):
        self.mask_network = mask_network.to(device)
        self.config = config or StateMaskConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(self.mask_network.parameters(), lr=self.config.learning_rate)
        self.dual_lambda = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.history = StateMaskHistory()

    def _policy_action(self, policy: Callable[[np.ndarray], Any], obs: np.ndarray) -> Any:
        return policy(obs)

    def _episode(self, env: Any, target_policy: Callable[[np.ndarray], Any], perturbed: bool):
        obs = _reset_observation(env)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        mask_actions: list[torch.Tensor] = []
        total = 0.0
        for _ in range(self.config.horizon):
            state = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.mask_network.distribution(obs_t)
            mask_action = dist.sample()
            use_random = perturbed and int(mask_action.item()) == 1
            action = sample_random_action(env) if use_random else self._policy_action(target_policy, state)
            next_obs, reward, done, _info = _step_env(env, action)
            total += reward
            log_probs.append(dist.log_prob(mask_action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            mask_actions.append(mask_action.float().squeeze(0))
            obs = _reset_observation(env) if done else next_obs
            if done:
                break
        return total, log_probs, entropies, mask_actions

    def train_step(self, env_factory: Callable[[], Any], target_policy: Callable[[np.ndarray], Any]) -> dict[str, float]:
        start = time.perf_counter()
        target_returns: list[float] = []
        perturbed_returns: list[float] = []
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        mask_terms: list[torch.Tensor] = []
        for _ in range(self.config.batch_episodes):
            target_ret, _, _, _ = self._episode(env_factory(), target_policy, perturbed=False)
            perturbed_ret, log_probs, entropies, masks = self._episode(env_factory(), target_policy, perturbed=True)
            gap = abs(target_ret - perturbed_ret)
            target_returns.append(target_ret)
            perturbed_returns.append(perturbed_ret)
            log_prob_terms.extend([lp * gap for lp in log_probs])
            entropy_terms.extend(entropies)
            mask_terms.extend(masks)

        target_mean = float(np.mean(target_returns))
        perturbed_mean = float(np.mean(perturbed_returns))
        return_gap = abs(target_mean - perturbed_mean)
        mask_rate_t = torch.stack(mask_terms).mean() if mask_terms else torch.tensor(0.0, device=self.device)
        policy_loss = torch.stack(log_prob_terms).mean() if log_prob_terms else torch.tensor(0.0, device=self.device)
        entropy = torch.stack(entropy_terms).mean() if entropy_terms else torch.tensor(0.0, device=self.device)
        constraint = mask_rate_t - self.config.mask_budget
        loss = self.config.return_gap_coef * policy_loss + self.dual_lambda.detach() * constraint - self.config.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.mask_network.parameters(), 0.5)
        self.optimizer.step()

        with torch.no_grad():
            self.dual_lambda = torch.clamp(
                self.dual_lambda + self.config.dual_learning_rate * constraint.detach(),
                min=0.0,
            )

        elapsed = time.perf_counter() - start
        metrics = {
            "return_gap": float(return_gap),
            "target_return": target_mean,
            "perturbed_return": perturbed_mean,
            "mask_rate": float(mask_rate_t.detach().cpu()),
            "dual_value": float(self.dual_lambda.detach().cpu()),
            "training_time_seconds": float(elapsed),
        }
        self.history.return_gap.append(metrics["return_gap"])
        self.history.target_return.append(metrics["target_return"])
        self.history.perturbed_return.append(metrics["perturbed_return"])
        self.history.mask_rate.append(metrics["mask_rate"])
        self.history.dual_value.append(metrics["dual_value"])
        self.history.cumulative_reward.append(metrics["perturbed_return"])
        self.history.training_time_seconds.append(metrics["training_time_seconds"])
        return metrics

    def train(self, env_factory: Callable[[], Any], target_policy: Callable[[np.ndarray], Any], iterations: int):
        for _ in range(iterations):
            self.train_step(env_factory, target_policy)
        return self.history

    def explanation(self) -> OriginalStateMaskExplanation:
        return OriginalStateMaskExplanation(self.mask_network)


def statemask_objective(return_target: float, return_perturbed: float) -> float:
    """Return J(theta)=|eta(pi)-eta(pi_bar)| for explicit tests and audits."""

    return abs(float(return_target) - float(return_perturbed))
