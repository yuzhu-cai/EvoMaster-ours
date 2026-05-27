#!/usr/bin/env python3
"""Run a RICE refinement experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rice_reproduce.envs.factory import make_env
from rice_reproduce.experiments.experiment_ii import build_experiment_ii_runner
from rice_reproduce.mask_network import MaskNetwork
from rice_reproduce.policies import make_actor_critic_for_env
from rice_reproduce.refinement import RICERefiner, RefinementConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/refine_report.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    method = cfg.get("method", "rice")
    explanation = cfg.get("explanation", "ours")
    ref_cfg = RefinementConfig(**cfg.get("refinement", {}))
    if method == "rice" and explanation == "ours":
        built = make_env(cfg["env_id"], sparse=bool(cfg.get("sparse", False)), normalize=cfg.get("normalize_observation"))
        env = built.env
        obs_dim = int(env.observation_space.shape[0])
        policy = make_actor_critic_for_env(env, built.key)
        mask = MaskNetwork(obs_dim)
        refiner = RICERefiner(env, policy, mask.importance, ref_cfg)
        row = {"environment": built.key, "method": method, "explanation": explanation, "normalized": built.normalized}
    else:
        refiner, row = build_experiment_ii_runner(cfg["env_id"], method, explanation, ref_cfg)
    report = refiner.refine(int(cfg.get("iterations", 1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": row, "report": report.__dict__}, indent=2, default=float), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
