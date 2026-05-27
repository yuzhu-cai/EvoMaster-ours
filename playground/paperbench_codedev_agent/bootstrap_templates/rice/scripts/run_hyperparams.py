#!/usr/bin/env python3
"""Print RICE hyper-parameter sweeps for Experiment V."""

from __future__ import annotations

import json
from rice_reproduce.experiments.configs import ALPHA_GRID, LAMBDA_GRID, P_GRID

print(json.dumps({"p": P_GRID, "lambda": LAMBDA_GRID, "alpha": ALPHA_GRID}, indent=2))
