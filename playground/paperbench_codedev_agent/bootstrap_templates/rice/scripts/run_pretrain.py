#!/usr/bin/env python3
"""Pretrain PPO policies for RICE environments and record reward curves."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rice_reproduce.envs.factory import make_env
from rice_reproduce.pretraining import PPOPretrainer, PretrainConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SelfishMining-v0")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/pretrain_report.json"))
    args = parser.parse_args()
    built = make_env(args.env_id)
    trainer = PPOPretrainer(built.env, built.key, PretrainConfig(total_updates=args.updates))
    _policy, report = trainer.train()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env_meta = {"key": built.key, "env_id": built.env_id, "normalized": built.normalized, "sparse": built.sparse}
    args.out.write_text(json.dumps({"environment": env_meta, "pretraining": asdict(report)}, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
