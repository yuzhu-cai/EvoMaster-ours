"""Evaluation entry points for fidelity and refining experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable
import json

import numpy as np

from rice_reproduce.critical_states import fidelity_score, perturb_segment_return, select_critical_state
from rice_reproduce.experiments.configs import FIDELITY_K
from rice_reproduce.policies import sample_random_action
from rice_reproduce.refinement import RICERefiner, RefinementConfig
from rice_reproduce.types import Trajectory


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def evaluate_fidelity(
    trajectories: list[Trajectory],
    importance_fn: Callable[[np.ndarray], np.ndarray],
    perturb_eval: Callable[[Trajectory, int, int], tuple[float, float]],
    max_reward_delta: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for k in FIDELITY_K:
        scores = []
        for trajectory in trajectories:
            importance = importance_fn(trajectory.states)
            critical = select_critical_state(trajectory, importance, k)
            original, perturbed = perturb_eval(trajectory, critical.window_start, critical.window_end)
            scores.append(fidelity_score(abs(perturbed - original), max_reward_delta, k))
        results[str(k)] = summarize(scores)
    return results


def rollout_return(env, policy, horizon: int) -> tuple[Trajectory, float]:
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    trajectory = Trajectory()
    total = 0.0
    from rice_reproduce.types import Transition

    for _ in range(horizon):
        state = np.asarray(obs, dtype=np.float32)
        action = policy(state)
        pre_snapshot = env.snapshot_state() if hasattr(env, "snapshot_state") else None
        step = env.step(action)
        if len(step) == 5:
            next_obs, reward, terminated, truncated, info = step
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step
        total += float(reward)
        info = dict(info or {})
        if pre_snapshot is not None:
            info["pre_state_snapshot"] = pre_snapshot
        trajectory.append(Transition(state, action, float(reward), np.asarray(next_obs, dtype=np.float32), bool(done), info))
        obs = next_obs
        if done:
            break
    return trajectory, total


def reset_env_to_trajectory_state(env, trajectory: Trajectory, step_index: int):
    """Restore env at the exact critical step when possible, otherwise replay."""
    transition = trajectory.transitions[step_index]
    snapshot = transition.info.get("pre_state_snapshot") if isinstance(transition.info, dict) else None
    if snapshot is not None and hasattr(env, "restore_state"):
        return env.restore_state(snapshot)
    if hasattr(env, "restore_state"):
        return env.restore_state(transition.state, steps=step_index)

    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    for t in range(step_index):
        step = env.step(trajectory.transitions[t].action)
        if len(step) == 5:
            obs, _reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            obs, _reward, done, _info = step
        if done:
            break
    return np.asarray(obs, dtype=np.float32)


def replay_with_randomized_segment(env, trajectory: Trajectory, policy, window_start: int, window_end: int) -> float:
    """Reset to the critical state, randomize the window, then finish normally."""
    return perturb_segment_return(
        env,
        trajectory,
        policy,
        window_start,
        window_end,
        lambda e, _state, start: reset_env_to_trajectory_state(e, trajectory, start),
        max_steps=len(trajectory),
    )


def replay_full_episode_with_randomized_segment(env, trajectory: Trajectory, policy, window_start: int, window_end: int) -> float:
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    total = 0.0
    horizon = len(trajectory)
    for t in range(horizon):
        state = np.asarray(obs, dtype=np.float32)
        action = sample_random_action(env) if window_start <= t < window_end else policy(state)
        step = env.step(action)
        if len(step) == 5:
            obs, reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            obs, reward, done, _info = step
        total += float(reward)
        if done:
            break
    return total


def measure_fidelity_reward_changes(env_factory, policy, importance_fn, trajectories: int = 50, horizon: int = 1000):
    rows = []
    max_delta = 1e-8
    for _ in range(trajectories):
        env = env_factory()
        trajectory, original_return = rollout_return(env, policy, horizon)
        if len(trajectory) == 0:
            continue
        scores = importance_fn(trajectory.states)
        for k in FIDELITY_K:
            critical = select_critical_state(trajectory, scores, k)
            prefix_return = float(sum(t.reward for t in trajectory.transitions[: critical.window_start]))
            perturbed = replay_with_randomized_segment(
                env_factory(),
                trajectory,
                policy,
                critical.window_start,
                critical.window_end,
            )
            perturbed_episode_return = prefix_return + perturbed
            delta = abs(perturbed_episode_return - original_return)
            max_delta = max(max_delta, delta)
            rows.append({"k": k, "delta": delta, "critical_index": critical.index})
    for row in rows:
        row["fidelity"] = fidelity_score(row["delta"], max_delta, row["k"])
    return {
        "average_reward_change": float(np.mean([r["delta"] for r in rows])) if rows else 0.0,
        "max_reward_change": float(max_delta),
        "by_k": rows,
    }


def run_refinement_grid(env_factory, policy_factory, mask_importance, configs: list[RefinementConfig], iterations: int):
    rows = []
    for cfg in configs:
        env = env_factory()
        policy = policy_factory(env)
        refiner = RICERefiner(env, policy, mask_importance, cfg)
        report = refiner.refine(iterations)
        rows.append({"config": asdict(cfg), "reward": summarize(report.rewards), "intrinsic": summarize(report.intrinsic_means)})
    return rows


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
