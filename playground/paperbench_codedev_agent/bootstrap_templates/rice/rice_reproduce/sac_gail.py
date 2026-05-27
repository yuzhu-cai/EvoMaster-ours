"""SAC warm-start, SAC pretraining, and GAIL-style policy distillation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policies import MLPActorCritic


@dataclass
class DistillationConfig:
    learning_rate: float = 3e-4
    epochs: int = 20
    batch_size: int = 256


@dataclass
class SACPretrainConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1000000
    warmup_steps: int = 1000
    updates_per_step: int = 1


@dataclass
class ReplayBuffer:
    max_size: int
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[np.ndarray] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    def add(self, state, action, reward, next_state, done) -> None:
        if len(self.states) >= self.max_size:
            self.states.pop(0); self.actions.pop(0); self.rewards.pop(0); self.next_states.pop(0); self.dones.pop(0)
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(float(done))

    def sample(self, batch_size: int):
        idx = np.random.default_rng().integers(0, len(self.states), size=batch_size)
        return (
            torch.as_tensor(np.asarray([self.states[i] for i in idx]), dtype=torch.float32),
            torch.as_tensor(np.asarray([self.actions[i] for i in idx]), dtype=torch.float32),
            torch.as_tensor(np.asarray([self.rewards[i] for i in idx]), dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(np.asarray([self.next_states[i] for i in idx]), dtype=torch.float32),
            torch.as_tensor(np.asarray([self.dones[i] for i in idx]), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.states)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(last, action_dim)
        self.log_std = nn.Linear(last, action_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.mean(h), torch.clamp(self.log_std(h), -20, 2)

    def sample(self, obs: torch.Tensor):
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor):
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim + action_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.net(torch.cat([obs, action], dim=-1))


class GAILDiscriminator(nn.Module):
    """Discriminator D(s,a) for SAC-to-PPO adversarial imitation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim + action_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.Tanh()])
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))

    def imitation_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self(states, actions)
        return -F.logsigmoid(-logits)


class SACAgent:
    """Minimal Soft Actor-Critic agent for dense MuJoCo pretraining."""

    def __init__(self, obs_dim: int, action_dim: int, config: SACPretrainConfig | None = None):
        self.config = config or SACPretrainConfig()
        self.actor = SquashedGaussianActor(obs_dim, action_dim)
        self.q1 = QNetwork(obs_dim, action_dim)
        self.q2 = QNetwork(obs_dim, action_dim)
        self.target_q1 = QNetwork(obs_dim, action_dim)
        self.target_q2 = QNetwork(obs_dim, action_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.config.critic_lr)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.deterministic(obs) if deterministic else self.actor.sample(obs)[0]
        return action.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        cfg = self.config
        states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)
        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_states)
            target_q = torch.min(self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions))
            backup = rewards + cfg.gamma * (1.0 - dones) * (target_q - cfg.alpha * next_logp)
        q1_loss = F.mse_loss(self.q1(states, actions), backup)
        q2_loss = F.mse_loss(self.q2(states, actions), backup)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        sampled_actions, logp = self.actor.sample(states)
        actor_loss = (cfg.alpha * logp - torch.min(self.q1(states, sampled_actions), self.q2(states, sampled_actions))).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * sp.data)
        return {"actor_loss": float(actor_loss), "q1_loss": float(q1_loss), "q2_loss": float(q2_loss)}


def collect_teacher_dataset(env, teacher_policy: Callable[[np.ndarray], object], steps: int):
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    states, actions = [], []
    for _ in range(steps):
        action = teacher_policy(obs)
        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action))
        step = env.step(action)
        if len(step) == 5:
            obs, _reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            obs, _reward, done, _info = step
        if done:
            reset = env.reset()
            obs = reset[0] if isinstance(reset, tuple) else reset
    return np.asarray(states, dtype=np.float32), np.asarray(actions)


def pretrain_sac(env, steps: int, config: SACPretrainConfig | None = None) -> tuple[SACAgent, list[dict[str, float]]]:
    """Pretrain a SAC teacher on a dense MuJoCo environment such as Hopper-v3."""

    cfg = config or SACPretrainConfig()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = SACAgent(obs_dim, action_dim, cfg)
    replay = ReplayBuffer(cfg.replay_size)
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    history: list[dict[str, float]] = []
    for step_i in range(steps):
        if step_i < cfg.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(np.asarray(obs, dtype=np.float32))
        step = env.step(action)
        if len(step) == 5:
            next_obs, reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            next_obs, reward, done, _info = step
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            reset = env.reset()
            obs = reset[0] if isinstance(reset, tuple) else reset
        if len(replay) >= cfg.batch_size and step_i >= cfg.warmup_steps:
            for _ in range(cfg.updates_per_step):
                history.append(agent.update(replay))
    return agent, history


def distill_policy(student: MLPActorCritic, states: np.ndarray, actions: np.ndarray, discrete: bool, config: DistillationConfig):
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(actions, dtype=torch.long if discrete else torch.float32)
    n = x.shape[0]
    losses = []
    for _ in range(config.epochs):
        order = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            dist = student.distribution(x[idx])
            if discrete:
                loss = F.cross_entropy(dist.logits, y[idx])
            else:
                loss = -dist.log_prob(y[idx]).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def train_gail_from_teacher(
    student: MLPActorCritic,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    policy_states: np.ndarray,
    policy_actions: np.ndarray,
    config: DistillationConfig,
) -> dict[str, list[float]]:
    """Apply GAIL: adversarially match SAC teacher state-action occupancy."""

    obs_dim = int(expert_states.shape[-1])
    action_dim = int(np.asarray(expert_actions).reshape(len(expert_actions), -1).shape[-1])
    discriminator = GAILDiscriminator(obs_dim, action_dim)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    pol_opt = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    expert_s = torch.as_tensor(expert_states, dtype=torch.float32)
    expert_a = torch.as_tensor(np.asarray(expert_actions).reshape(len(expert_actions), -1), dtype=torch.float32)
    policy_s = torch.as_tensor(policy_states, dtype=torch.float32)
    policy_a = torch.as_tensor(np.asarray(policy_actions).reshape(len(policy_actions), -1), dtype=torch.float32)
    losses = {"discriminator": [], "policy": []}
    n = min(len(expert_s), len(policy_s))
    for _ in range(config.epochs):
        order = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            exp_logits = discriminator(expert_s[idx], expert_a[idx])
            pol_logits = discriminator(policy_s[idx], policy_a[idx])
            disc_loss = F.binary_cross_entropy_with_logits(exp_logits, torch.ones_like(exp_logits))
            disc_loss = disc_loss + F.binary_cross_entropy_with_logits(pol_logits, torch.zeros_like(pol_logits))
            disc_opt.zero_grad(); disc_loss.backward(); disc_opt.step()

            dist = student.distribution(policy_s[idx])
            sampled = dist.sample()
            sampled_flat = sampled.reshape(sampled.shape[0], -1).float()
            imitation_reward = discriminator.imitation_reward(policy_s[idx], sampled_flat).detach().squeeze(-1)
            log_prob = dist.log_prob(sampled) if student.discrete else dist.log_prob(sampled).sum(-1)
            policy_loss = -(log_prob * imitation_reward).mean()
            pol_opt.zero_grad(); policy_loss.backward(); pol_opt.step()
            losses["discriminator"].append(float(disc_loss))
            losses["policy"].append(float(policy_loss))
    return losses
