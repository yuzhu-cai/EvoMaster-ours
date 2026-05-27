from __future__ import annotations

import numpy as np
import torch

from rice_reproduce.critical_states import fidelity_score, select_critical_state, sliding_window_scores
from rice_reproduce.envs.factory import CageChallenge2Env, MacroDriveEnv
from rice_reproduce.mask_network import MaskNetwork
from rice_reproduce.policies import make_actor_critic_for_env
from rice_reproduce.rnd import RNDModel, RNDReward
from rice_reproduce.statemask import statemask_objective
from rice_reproduce.types import Trajectory, Transition


def make_traj(n=5):
    t = Trajectory()
    for i in range(n):
        s = np.asarray([i, i + 1], dtype=np.float32)
        t.append(Transition(s, 0, float(i), s + 1, False))
    return t


def test_sliding_window_and_critical_state():
    traj = make_traj()
    scores = [0.1, 0.2, 0.9, 0.8, 0.1]
    windows = sliding_window_scores(scores, 0.4)
    assert windows.shape[0] == 4
    critical = select_critical_state(traj, scores, 0.4)
    assert critical.index in {2, 3}


def test_fidelity_score_is_finite():
    assert np.isfinite(fidelity_score(5.0, 20.0, 0.1))


def test_mask_importance_shape():
    net = MaskNetwork(obs_dim=2)
    importance = net.importance(np.zeros((4, 2), dtype=np.float32))
    assert importance.shape == (4,)


def test_rnd_bonus_shape():
    rnd = RNDReward(RNDModel(obs_dim=2))
    bonus = rnd.bonus(np.zeros((3, 2), dtype=np.float32))
    assert bonus.shape == (3,)


def test_statemask_objective():
    assert statemask_objective(10.0, 7.5) == 2.5


def test_real_world_envs_initialize():
    for env in [MacroDriveEnv(horizon=2), CageChallenge2Env(hosts=2, horizon=2)]:
        obs, _ = env.reset(seed=0)
        step = env.step(env.action_space.sample())
        assert np.asarray(obs).size > 0
        assert len(step) == 5


def test_selfish_mining_policy_architecture():
    env = CageChallenge2Env(hosts=2)
    policy = make_actor_critic_for_env(env, "selfish_mining")
    linear_layers = [m for m in policy.backbone if isinstance(m, torch.nn.Linear)]
    assert [layer.out_features for layer in linear_layers] == [128, 128, 128, 128]
