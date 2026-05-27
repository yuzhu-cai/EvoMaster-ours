"""Command-line interface for RICE reproduction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .critical_states import fidelity_score, sliding_window_scores
from .experiments.configs import MUPAPER_ENVS, P_GRID, LAMBDA_GRID


def cmd_describe(args: argparse.Namespace) -> None:
    payload = {
        "mujoco_envs": {k: v.__dict__ for k, v in MUPAPER_ENVS.items()},
        "p_grid": P_GRID,
        "lambda_grid": LAMBDA_GRID,
        "core_components": [
            "mask network with alpha blinding bonus",
            "mixed critical/default initial-state distribution",
            "RND intrinsic reward",
            "PPO refinement update",
            "fidelity sliding-window evaluation",
            "PPO, StateMask-R, JSRL, random-explanation baselines",
        ],
    }
    print(json.dumps(payload, indent=2))


def cmd_smoke(args: argparse.Namespace) -> None:
    scores = sliding_window_scores([0.1, 0.3, 0.2, 0.9, 0.8], 0.4)
    fid = fidelity_score(reward_delta=4.0, max_reward_delta=20.0, window_fraction=0.2)
    out = {"sliding_window": scores.tolist(), "fidelity": fid}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RICE reproduction CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.set_defaults(func=cmd_describe)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="outputs/smoke.json")
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
