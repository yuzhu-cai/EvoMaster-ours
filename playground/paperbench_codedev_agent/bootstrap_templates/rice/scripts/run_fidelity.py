#!/usr/bin/env python3
"""Run fidelity evaluation by perturbing critical time windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rice_reproduce.critical_states import fidelity_score, select_critical_state
from rice_reproduce.envs.factory import make_env
from rice_reproduce.experiments.evaluate import measure_fidelity_reward_changes
from rice_reproduce.mask_network import MaskNetwork
from rice_reproduce.policies import make_actor_critic_for_env
from rice_reproduce.types import Trajectory, Transition


def load_trajectory(path: Path) -> Trajectory:
    data = json.loads(path.read_text())
    traj = Trajectory()
    for item in data["transitions"]:
        traj.append(Transition(np.asarray(item["state"], dtype=np.float32), item["action"], float(item["reward"]), np.asarray(item["next_state"], dtype=np.float32), bool(item["done"])))
    return traj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="", help="Run end-to-end fidelity in an environment.")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--importance", type=Path)
    parser.add_argument("--k", type=float, default=0.1)
    parser.add_argument("--reward-delta", type=float)
    parser.add_argument("--max-reward-delta", type=float)
    parser.add_argument("--trajectories", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1000)
    args = parser.parse_args()
    if args.env_id:
        built = make_env(args.env_id)
        policy = make_actor_critic_for_env(built.env, built.key)
        mask = MaskNetwork(int(np.prod(built.env.observation_space.shape)))
        result = measure_fidelity_reward_changes(
            lambda: make_env(args.env_id).env,
            lambda obs: policy.act(obs, deterministic=True),
            mask.importance,
            trajectories=args.trajectories,
            horizon=args.horizon,
        )
        print(json.dumps({"environment": built.key, **result}, indent=2))
        return 0
    if not args.trajectory or not args.importance or args.reward_delta is None or args.max_reward_delta is None:
        raise SystemExit("pass --env-id for measured fidelity or provide --trajectory, --importance, --reward-delta, and --max-reward-delta")
    trajectory = load_trajectory(args.trajectory)
    importance = np.load(args.importance)
    critical = select_critical_state(trajectory, importance, args.k)
    print(json.dumps({"critical_index": critical.index, "window": [critical.window_start, critical.window_end], "score": fidelity_score(args.reward_delta, args.max_reward_delta, args.k)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
