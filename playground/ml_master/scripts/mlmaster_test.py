#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

PROJECT_DIR = Path("/data/chengwang/EvoMaster-ours")
RUN_PY = PROJECT_DIR / "run.py"
BASE_CONFIG = Path("/data/chengwang/EvoMaster-ours/configs/ml_master/config.yaml")

EXP_ROOT = Path("/data/exp_data")
AGENT = "ml_master"
MAX_PARALLEL = 1

COMPETITIONS = [
    # "aerial-cactus-identification",
    # "aptos2019-blindness-detection",
    # "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    # "dog-breed-identification",
    # "dogs-vs-cats-redux-kernels-edition",
    # "histopathologic-cancer-detection",
    # "jigsaw-toxic-comment-classification-challenge",
    # "leaf-classification",
    # "mlsp-2013-birds",
    # "new-york-city-taxi-fare-prediction",
    # "nomad2018-predict-transparent-conductors",
    # "plant-pathology-2020-fgvc7",
    # "random-acts-of-pizza",
    # "ranzcr-clip-catheter-line-classification",
    # "siim-isic-melanoma-classification",
    # "spooky-author-identification",
    # "tabular-playground-series-dec-2021",
    # "tabular-playground-series-may-2022",
    # "text-normalization-challenge-english-language",
    # "text-normalization-challenge-russian-language",
    # "the-icml-2013-whale-challenge-right-whale-redux",
]

# CPU binding settings for parallel workers.
CPU_SETS = ["0-127"]
THREADS_PER_JOB = 128


def task_path(comp_id: str) -> Path:
    """Build the task description path for a competition.

    Args:
        comp_id: Competition identifier.

    Returns:
        Path: Path to `description.md` for this competition.
    """
    return EXP_ROOT / comp_id / "prepared" / "public" / "description.md"


def make_tmp_config(comp_id: str) -> Path:
    """Create a temporary config file for one competition.

    Args:
        comp_id: Competition identifier.

    Returns:
        Path: Path to the generated temporary YAML config.
    """
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["exp_id"] = comp_id

    # Dynamically update input symlinks.
    public_dir = str(EXP_ROOT / comp_id / "prepared" / "public")
    cfg.setdefault("session", {}).setdefault("local", {}).setdefault("symlinks", {})
    cfg["session"]["local"]["symlinks"] = {public_dir: "input"}

    # Optional internal parallelism.
    # cfg["session"]["local"].setdefault("parallel", {})
    # cfg["session"]["local"]["parallel"]["enabled"] = True
    # cfg["session"]["local"]["parallel"]["max_parallel"] = 2

    # Override working directories for this competition.
    cfg["session"]["local"]["working_dir"] = f"./playground/ml_master/workspace/{comp_id}"
    cfg["session"]["local"]["workspace_path"] = f"./playground/workspace/{comp_id}"

    fix_prompt_paths(cfg)
    tmp_dir = PROJECT_DIR / ".tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / f"{comp_id}.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return tmp_cfg


def run_one(comp_id: str, slot: int) -> int:
    """Run one competition job in the selected worker slot.

    Args:
        comp_id: Competition identifier.
        slot: Worker slot index for CPU binding.

    Returns:
        int: Process return code.
    """
    tpath = task_path(comp_id)
    if not tpath.exists():
        print(f"[SKIP] {comp_id}: task not found: {tpath}", flush=True)
        return 2

    tmp_cfg = make_tmp_config(comp_id)

    unique_id = f"ml_master_{comp_id}_{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    runs_dir = PROJECT_DIR / "runs" / unique_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # out_dir = PROJECT_DIR / ".mlmaster_test_out"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_file = out_dir / f"{comp_id}.out.txt"

    # CPU/thread limits.
    cpu_set = CPU_SETS[slot % len(CPU_SETS)]
    env = {
        **dict(**__import__("os").environ),
        "OMP_NUM_THREADS": str(THREADS_PER_JOB),
        "MKL_NUM_THREADS": str(THREADS_PER_JOB),
        "OPENBLAS_NUM_THREADS": str(THREADS_PER_JOB),
        "NUMEXPR_NUM_THREADS": str(THREADS_PER_JOB),
        "VECLIB_MAXIMUM_THREADS": str(THREADS_PER_JOB),
    }

    cmd = [
        "taskset",
        "-c",
        cpu_set,
        sys.executable,
        str(RUN_PY),
        "--agent",
        AGENT,
        "--task",
        str(tpath),
        "--config",
        str(tmp_cfg),
        "--run-dir",
        str(runs_dir),
    ]

    print(f"[RUN] slot={slot} cpu={cpu_set} comp={comp_id}", flush=True)
    # with out_file.open("w", encoding="utf-8") as f:
    #     p = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env, stdout=f, stderr=subprocess.STDOUT)
    #
    # rc = p.returncode
    # if rc != 0:
    #     tail = out_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
    #     print(f"\n[FAIL] {comp_id} returncode={rc}  (tail of {out_file})", flush=True)
    #     print("\n".join(tail), flush=True)
    #
    # return rc
    p = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    return p.returncode


def abs_if_relative(p: str) -> str:
    """Convert a relative path to an absolute path under ml_master.

    Args:
        p: Original path string.

    Returns:
        str: Absolute path string.
    """
    pth = Path(p)
    return str(pth if pth.is_absolute() else (PROJECT_DIR / "playground" / "ml_master" / pth))


def fix_prompt_paths(cfg: dict) -> None:
    """Normalize prompt-related paths in config to absolute paths.

    Args:
        cfg: Config dictionary loaded from YAML.

    Returns:
        None.
    """
    # Top-level system prompt path.
    if "system_prompt_file" in cfg:
        cfg["system_prompt_file"] = abs_if_relative(cfg["system_prompt_file"])

    # MCP config path.
    if "mcp" in cfg and isinstance(cfg["mcp"], dict) and "config_file" in cfg["mcp"]:
        cfg["mcp"]["config_file"] = abs_if_relative(cfg["mcp"]["config_file"])

    # Agent prompt paths.
    agents = cfg.get("agents", {})
    for _name, agent_cfg in agents.items():
        if not isinstance(agent_cfg, dict):
            continue
        for key in ["system_prompt_file", "user_prompt_file"]:
            if key in agent_cfg:
                agent_cfg[key] = abs_if_relative(agent_cfg[key])


def main() -> None:
    """Run all configured competitions in batches.

    Returns:
        None.
    """
    assert RUN_PY.exists(), f"run.py not found: {RUN_PY}"
    assert BASE_CONFIG.exists(), f"base config not found: {BASE_CONFIG}"

    # Submit tasks in batches: each batch runs up to MAX_PARALLEL jobs.
    i = 0
    while i < len(COMPETITIONS):
        batch = COMPETITIONS[i : i + MAX_PARALLEL]
        print(f"\n=== BATCH {i // MAX_PARALLEL + 1}: {batch} ===", flush=True)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {}
            for slot, comp_id in enumerate(batch):
                futures[executor.submit(run_one, comp_id, slot)] = comp_id

            for future in as_completed(futures):
                comp_id = futures[future]
                rc = future.result()
                print(f"[DONE] {comp_id}: returncode={rc}", flush=True)

        i += MAX_PARALLEL


if __name__ == "__main__":
    main()
