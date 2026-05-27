#!/usr/bin/env python3
"""Materialize the Table 1 experiment plan."""

from __future__ import annotations

import json
from rice_reproduce.experiments.configs import MUPAPER_ENVS, REAL_WORLD_ENVS, TABLE1_EXPLANATIONS, TABLE1_REFINERS
from rice_reproduce.experiments.experiment_ii import materialize_experiment_ii_plan


def main() -> int:
    rows = []
    for spec in list(MUPAPER_ENVS.values()) + list(REAL_WORLD_ENVS.values()):
        rows.append({"environment": spec.__dict__, "refiners": TABLE1_REFINERS, "explanations": TABLE1_EXPLANATIONS})
    print(json.dumps({"table1_summary": rows, "experiment_ii_full_matrix": materialize_experiment_ii_plan()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
