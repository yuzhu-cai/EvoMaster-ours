"""Random Network Distillation exploration bonus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RNDModel(nn.Module):
    def __init__(self, obs_dim: int, feature_dim: int = 128, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.target = self._make_network(obs_dim, feature_dim, hidden_sizes)
        self.predictor = self._make_network(obs_dim, feature_dim, hidden_sizes)
        for param in self.target.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _make_network(obs_dim: int, feature_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        layers.append(nn.Linear(last, feature_dim))
        return nn.Sequential(*layers)

    def prediction_error(self, obs: torch.Tensor) -> torch.Tensor:
        target = self.target(obs)
        pred = self.predictor(obs)
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)


@dataclass
class RunningNormalizer:
    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = float(values.size)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (np.sqrt(self.var) + 1e-8)


class RNDReward:
    def __init__(self, model: RNDModel, learning_rate: float = 1e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.predictor.parameters(), lr=learning_rate)
        self.normalizer = RunningNormalizer()

    def bonus(self, next_states: np.ndarray, update_stats: bool = True) -> np.ndarray:
        obs = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            errors = self.model.prediction_error(obs).cpu().numpy()
        if update_stats:
            self.normalizer.update(errors)
        return np.maximum(self.normalizer.normalize(errors), 0.0).astype(np.float32)

    def update(self, next_states: np.ndarray, epochs: int = 1, batch_size: int = 256) -> float:
        obs = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        n = obs.shape[0]
        last = 0.0
        for _ in range(epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                loss = self.model.prediction_error(obs[idx]).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                last = float(loss.detach().cpu())
        return last
