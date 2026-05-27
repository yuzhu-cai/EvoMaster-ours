"""RICE refinement loop with mixed initial states and RND rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .critical_states import CriticalState, select_critical_state
from .policies import MLPActorCritic
from .reset import ResetSnapshot, ResettableEnvWrapper
from .rnd import RNDModel, RNDReward
from .types import RolloutBatch, Trajectory, Transition, generalized_advantage_estimate


@dataclass
class RefinementConfig:
    p_reset_to_critical: float = 0.25
    rnd_lambda: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    policy_learning_rate: float = 3e-4
    rnd_learning_rate: float = 1e-4
    rollout_steps: int = 2048
    ppo_epochs: int = 4
    batch_size: int = 256
    device: str = "cpu"
    critical_fraction: float = 0.1


@dataclass
class RefinementReport:
    rewards: list[float] = field(default_factory=list)
    intrinsic_means: list[float] = field(default_factory=list)
    critical_indices: list[int] = field(default_factory=list)
    losses: list[dict[str, float]] = field(default_factory=list)
    cumulative_reward: list[float] = field(default_factory=list)
    training_time_seconds: list[float] = field(default_factory=list)


class RICERefiner:
    """Implements Algorithm 2 from the paper."""

    def __init__(
        self,
        env: Any,
        policy: MLPActorCritic,
        mask_importance: Callable[[np.ndarray], np.ndarray],
        config: RefinementConfig | None = None,
    ):
        self.env = ResettableEnvWrapper(env)
        self.policy = policy
        self.mask_importance = mask_importance
        self.config = config or RefinementConfig()
        self.device = self.config.device
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.policy_learning_rate)
        obs_dim = int(np.prod(env.observation_space.shape))
        self.rnd = RNDReward(RNDModel(obs_dim), self.config.rnd_learning_rate, self.device)
        self.rng = np.random.default_rng()

    def sample_policy_trajectory(self, max_steps: int) -> tuple[Trajectory, list[Any]]:
        reset = self.env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        trajectory = Trajectory()
        actions: list[Any] = []
        for _ in range(max_steps):
            action = self.policy.act(np.asarray(obs, dtype=np.float32), deterministic=False)
            pre_snapshot = self.env.snapshot_state() if hasattr(self.env, "snapshot_state") else None
            step = self.env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step
            info = dict(info or {})
            if pre_snapshot is not None:
                info["pre_state_snapshot"] = pre_snapshot
            trajectory.append(Transition(np.asarray(obs, dtype=np.float32), action, float(reward), np.asarray(next_obs, dtype=np.float32), bool(done), info))
            actions.append(action)
            obs = next_obs
            if done:
                break
        return trajectory, actions

    def choose_initial_observation(self, report: RefinementReport) -> np.ndarray:
        if self.rng.random() >= self.config.p_reset_to_critical:
            reset = self.env.reset()
            return reset[0] if isinstance(reset, tuple) else reset
        trajectory, actions = self.sample_policy_trajectory(max_steps=min(1000, self.config.rollout_steps))
        if len(trajectory) == 0:
            reset = self.env.reset()
            return reset[0] if isinstance(reset, tuple) else reset
        scores = self.mask_importance(trajectory.states)
        critical = select_critical_state(trajectory, scores, self.config.critical_fraction)
        prefix = actions[:critical.index]
        transition_info = trajectory.transitions[critical.index].info
        env_state = transition_info.get("pre_state_snapshot") if isinstance(transition_info, dict) else None
        snapshot = self.env.capture(critical.state, critical.index, prefix)
        if env_state is not None:
            snapshot = ResetSnapshot(critical.state, env_state, prefix, critical.index)
        report.critical_indices.append(critical.index)
        return self.env.restore(snapshot)

    def collect_rollout(self, report: RefinementReport) -> RolloutBatch:
        obs = self.choose_initial_observation(report)
        transitions: list[Transition] = []
        episode_reward = 0.0
        for _ in range(self.config.rollout_steps):
            state = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.policy.distribution(obs_t)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).sum(-1) if not self.policy.discrete else dist.log_prob(action_t)
                value = self.policy.value(obs_t)
            action = int(action_t.item()) if self.policy.discrete else action_t.squeeze(0).cpu().numpy()
            step = self.env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step
            transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=np.asarray(next_obs, dtype=np.float32),
                    done=bool(done),
                    info=info,
                    log_prob=float(log_prob.item()),
                    value=float(value.item()),
                )
            )
            episode_reward += float(reward)
            obs = self.choose_initial_observation(report) if done else next_obs
            if done:
                report.rewards.append(episode_reward)
                episode_reward = 0.0
        next_states = np.asarray([t.next_state for t in transitions], dtype=np.float32)
        intrinsic = self.rnd.bonus(next_states, update_stats=True)
        report.intrinsic_means.append(float(np.mean(intrinsic)))
        rewards = np.asarray([t.reward for t in transitions], dtype=np.float32) + self.config.rnd_lambda * intrinsic
        dones = np.asarray([t.done for t in transitions], dtype=np.float32)
        values = np.asarray([t.value for t in transitions], dtype=np.float32)
        advantages, returns = generalized_advantage_estimate(rewards, values, dones, self.config.gamma, self.config.gae_lambda)
        return RolloutBatch(
            states=np.asarray([t.state for t in transitions], dtype=np.float32),
            actions=np.asarray([t.action for t in transitions]),
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=np.asarray([t.log_prob for t in transitions], dtype=np.float32),
            values=values,
            advantages=advantages,
            returns=returns,
        )

    def update_policy(self, batch: RolloutBatch) -> dict[str, float]:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions_dtype = torch.long if self.policy.discrete else torch.float32
        actions = torch.as_tensor(batch.actions, dtype=actions_dtype, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = states.shape[0]
        last = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "rnd_loss": 0.0}
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                dist = self.policy.distribution(states[idx])
                if self.policy.discrete:
                    log_probs = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()
                else:
                    log_probs = dist.log_prob(actions[idx]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()
                values = self.policy.value(states[idx])
                value_loss = F.mse_loss(values, returns[idx])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                last.update({"policy_loss": float(policy_loss), "value_loss": float(value_loss), "entropy": float(entropy)})
        last["rnd_loss"] = self.rnd.update(batch.next_states, epochs=1, batch_size=self.config.batch_size)
        return last

    def refine(self, iterations: int) -> RefinementReport:
        report = RefinementReport()
        for _ in range(iterations):
            start = time.perf_counter()
            batch = self.collect_rollout(report)
            report.losses.append(self.update_policy(batch))
            report.cumulative_reward.append(float(np.sum(batch.rewards)))
            report.training_time_seconds.append(float(time.perf_counter() - start))
        return report
