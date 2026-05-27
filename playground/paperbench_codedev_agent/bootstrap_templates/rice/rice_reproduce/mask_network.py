"""RICE mask network and training loop."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policies import sample_random_action
from .types import RolloutBatch, Trajectory, Transition, generalized_advantage_estimate


def _reset_observation(env: Any) -> Any:
    reset = env.reset()
    return reset[0] if isinstance(reset, tuple) else reset


class MaskNetwork(nn.Module):
    """Binary policy: 0 marks critical/keep-policy steps, 1 randomizes."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.Tanh()])
            last = width
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, 2)
        self.value_head = nn.Linear(last, 1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.policy_head(self.body(obs)))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.body(obs)).squeeze(-1)

    def importance(self, states: np.ndarray) -> np.ndarray:
        """Return P(mask=0), the probability that each state is critical."""
        obs = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            probs = self.distribution(obs).probs.cpu().numpy()
        return probs[:, 0]


@dataclass
class MaskTrainConfig:
    alpha: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 256


class PerturbedPolicy:
    """Combines the target policy with mask-controlled random actions."""

    def __init__(self, target_policy: Callable[[np.ndarray], Any], mask_network: MaskNetwork, env: Any):
        self.target_policy = target_policy
        self.mask_network = mask_network
        self.env = env

    def act(self, observation: np.ndarray) -> tuple[Any, int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self.mask_network.distribution(obs)
            mask_action = dist.sample()
            log_prob = dist.log_prob(mask_action)
            value = self.mask_network.value(obs)
        base_action = self.target_policy(observation)
        action = sample_random_action(self.env) if int(mask_action.item()) == 1 else base_action
        return action, int(mask_action.item()), float(log_prob.item()), float(value.item())


class OptimizedStateMaskExplanation:
    """Inference-time RICE/Ours explanation usable without retraining."""

    def __init__(self, mask_network: MaskNetwork, threshold: float = 0.5):
        self.mask_network = mask_network
        self.threshold = threshold

    @classmethod
    def from_checkpoint(cls, path: str, obs_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        mask = MaskNetwork(obs_dim, hidden_sizes)
        state = torch.load(path, map_location="cpu")
        mask.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        return cls(mask)

    def importance(self, states: np.ndarray) -> np.ndarray:
        return self.mask_network.importance(states)

    def mask_decision(self, observation: np.ndarray) -> int:
        importance = float(self.importance(np.asarray([observation], dtype=np.float32))[0])
        return 0 if importance >= self.threshold else 1

    def generate_perturbed_rollout(self, env: Any, target_policy: Callable[[np.ndarray], Any], horizon: int) -> Trajectory:
        obs = _reset_observation(env)
        trajectory = Trajectory()
        for _ in range(horizon):
            state = np.asarray(obs, dtype=np.float32)
            action = sample_random_action(env) if self.mask_decision(state) == 1 else target_policy(state)
            step = env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step
            trajectory.append(Transition(state, action, float(reward), np.asarray(next_obs, dtype=np.float32), bool(done), info))
            obs = _reset_observation(env) if done else next_obs
            if done:
                break
        return trajectory


class MaskTrainer:
    """PPO trainer for the RICE mask objective R + alpha * mask_action."""

    def __init__(self, mask_network: MaskNetwork, config: MaskTrainConfig | None = None, device: str = "cpu"):
        self.mask_network = mask_network.to(device)
        self.config = config or MaskTrainConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(self.mask_network.parameters(), lr=self.config.learning_rate)
        self.training_time_seconds: list[float] = []

    def collect(self, env: Any, target_policy: Callable[[np.ndarray], Any], horizon: int) -> RolloutBatch:
        perturbed = PerturbedPolicy(target_policy, self.mask_network, env)
        obs = _reset_observation(env)
        transitions: list[Transition] = []
        for _ in range(horizon):
            action, mask_action, log_prob, value = perturbed.act(np.asarray(obs, dtype=np.float32))
            step = env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step
            shaped_reward = float(reward) + self.config.alpha * float(mask_action)
            transitions.append(
                Transition(
                    state=np.asarray(obs, dtype=np.float32),
                    action=mask_action,
                    reward=shaped_reward,
                    next_state=np.asarray(next_obs, dtype=np.float32),
                    done=bool(done),
                    info=info,
                    mask_action=mask_action,
                    log_prob=log_prob,
                    value=value,
                )
            )
            obs = _reset_observation(env) if done else next_obs
        rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
        dones = np.asarray([t.done for t in transitions], dtype=np.float32)
        values = np.asarray([t.value for t in transitions], dtype=np.float32)
        advantages, returns = generalized_advantage_estimate(rewards, values, dones, self.config.gamma, self.config.gae_lambda)
        return RolloutBatch(
            states=np.asarray([t.state for t in transitions], dtype=np.float32),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            rewards=rewards,
            next_states=np.asarray([t.next_state for t in transitions], dtype=np.float32),
            dones=dones,
            log_probs=np.asarray([t.log_prob for t in transitions], dtype=np.float32),
            values=values,
            advantages=advantages,
            returns=returns,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = states.shape[0]
        last = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.config.epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                dist = self.mask_network.distribution(states[idx])
                log_probs = dist.log_prob(actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()
                values = self.mask_network.value(states[idx])
                value_loss = F.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mask_network.parameters(), 0.5)
                self.optimizer.step()
                last = {"policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy)}
        return last

    def train(self, env: Any, target_policy: Callable[[np.ndarray], Any], iterations: int, horizon: int) -> list[dict[str, float]]:
        history = []
        for _ in range(iterations):
            started_at = time.perf_counter()
            batch = self.collect(env, target_policy, horizon)
            metrics = self.update(batch)
            elapsed = time.perf_counter() - started_at
            self.training_time_seconds.append(float(elapsed))
            metrics.update(
                {
                    "iteration_training_time_seconds": float(elapsed),
                    "cumulative_training_time_seconds": float(sum(self.training_time_seconds)),
                    "samples_seen": float((len(history) + 1) * horizon),
                }
            )
            history.append(metrics)
        return history

    def score_trajectory(self, trajectory: Trajectory) -> np.ndarray:
        return self.mask_network.importance(trajectory.states)
