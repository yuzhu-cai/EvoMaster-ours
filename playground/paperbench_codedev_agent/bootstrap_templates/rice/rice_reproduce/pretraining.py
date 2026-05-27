"""Policy pretraining loops used before RICE refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policies import MLPActorCritic, make_actor_critic_for_env
from .types import RolloutBatch, Transition, generalized_advantage_estimate


@dataclass
class PretrainConfig:
    total_updates: int = 10
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "cpu"


@dataclass
class PretrainReport:
    update_reward: list[float] = field(default_factory=list)
    cumulative_reward: list[float] = field(default_factory=list)
    losses: list[dict[str, float]] = field(default_factory=list)
    training_time_seconds: list[float] = field(default_factory=list)


def _reset(env: Any):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env: Any, action: Any):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


class PPOPretrainer:
    """Generic PPO pretraining for MuJoCo, SelfishMining, CAGE, and driving policies."""

    def __init__(self, env: Any, env_key: str, config: PretrainConfig | None = None):
        self.env = env
        self.env_key = env_key
        self.config = config or PretrainConfig()
        self.policy = make_actor_critic_for_env(env, env_key).to(self.config.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    def collect(self) -> RolloutBatch:
        obs = _reset(self.env)
        transitions: list[Transition] = []
        for _ in range(self.config.rollout_steps):
            state = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.config.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.policy.distribution(obs_t)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t) if self.policy.discrete else dist.log_prob(action_t).sum(-1)
                value = self.policy.value(obs_t)
            action = int(action_t.item()) if self.policy.discrete else action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = _step(self.env, action)
            transitions.append(Transition(state, action, reward, np.asarray(next_obs, dtype=np.float32), done, info, log_prob=float(log_prob.item()), value=float(value.item())))
            obs = _reset(self.env) if done else next_obs
        rewards = np.asarray([t.reward for t in transitions], dtype=np.float32)
        dones = np.asarray([t.done for t in transitions], dtype=np.float32)
        values = np.asarray([t.value for t in transitions], dtype=np.float32)
        advantages, returns = generalized_advantage_estimate(rewards, values, dones, self.config.gamma, self.config.gae_lambda)
        return RolloutBatch(
            states=np.asarray([t.state for t in transitions], dtype=np.float32),
            actions=np.asarray([t.action for t in transitions]),
            rewards=rewards,
            next_states=np.asarray([t.next_state for t in transitions], dtype=np.float32),
            dones=dones,
            log_probs=np.asarray([t.log_prob for t in transitions], dtype=np.float32),
            values=values,
            advantages=advantages,
            returns=returns,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.config.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long if self.policy.discrete else torch.float32, device=self.config.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.config.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.config.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.config.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        last = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n = states.shape[0]
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(n, device=self.config.device)
            for start in range(0, n, self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                dist = self.policy.distribution(states[idx])
                log_probs = dist.log_prob(actions[idx]) if self.policy.discrete else dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().mean() if self.policy.discrete else dist.entropy().sum(-1).mean()
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()
                value_loss = F.mse_loss(self.policy.value(states[idx]), returns[idx])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                last = {"policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy)}
        return last

    def train(self) -> tuple[MLPActorCritic, PretrainReport]:
        report = PretrainReport()
        total_reward = 0.0
        for _ in range(self.config.total_updates):
            start = time.perf_counter()
            batch = self.collect()
            loss = self.update(batch)
            update_reward = float(np.sum(batch.rewards))
            total_reward += update_reward
            report.update_reward.append(update_reward)
            report.cumulative_reward.append(total_reward)
            report.losses.append(loss)
            report.training_time_seconds.append(time.perf_counter() - start)
        return self.policy, report
