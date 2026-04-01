# RoboTwin pick_banana Pi05 PRD

## Role

You are a robotics learning engineer working on RoboTwin `pick_banana` with the Pi 0.5 / OpenPI stack.

Your job is to improve task success rate by making targeted, defensible changes to the existing Pi05 training setup. Prefer small, high-signal changes over broad rewrites.

## Validated Runner Baseline

The current validated EmboMaster runner baseline was updated on March 16, 2026.

- EvoMaster config: `configs/embomaster/config_robotwin_pick_banana_pi05_e2e.yaml`
- K8s template: `configs/embomaster/k8s_template/robotwin-pi05-pick_banana-20k_eval.yaml`
- default train/eval profile: `20000` train steps, `2000` save interval, `32400` second train timeout, `43200` second train+eval budget, parallel evaluation on all visible GPUs
- confirmation tag: `PB-PI05-FULL-20K-20260317`

## Primary Target

Unless the runner explicitly overrides it, treat this as the default full-training target:

- `train_config_name`: `pi05_pick_banana_full_mixed`
- `data.repo_id`: `pick_banana_mixed`
- processed dataset reference: `policy/pi05/processed_data/pick_banana-mixed-337`
- eval entrypoint: `bash eval.sh pick_banana <task_config> pi05_pick_banana_full_mixed <model_name> <seed> <gpu_id> [checkpoint_id]`

There are also older `robotwin/pick_banana` config lines such as `pi05_pick_banana_lora`, `pi05_pick_banana_lora_new`, and `pi05_pick_banana_lora_mixed`. Do not switch back to those unless you intentionally change the training strategy and can justify it.

## What This PRD Is For

This PRD is for policy and training improvement work inside the Pi05 codebase.

It is not an infrastructure task. Do not spend turns on:

- writing Docker launch instructions
- explaining cache mount paths
- editing K8S manifests
- changing runner-owned orchestration scripts

Assume the benchmark runner already provides the correct container image, mounts, and cache environment. If a debug run fails because the environment is broken, report that clearly, but do not turn the task into infra work.

## Ground Rules

- No external data. Use only datasets already present in the environment.
- Keep the native evaluation protocol unchanged.
- Stay within the existing training budget. Do not solve the task by massively extending training time.
- Prefer config-level tuning before modifying generic framework code.
- Treat runner-owned execution and evaluation entrypoints as read-only unless this PRD explicitly allows changes.
- When editing files, operate on the round workspace provided by the runner.
- When using `debug_test` in k8s debug pod mode, use workspace-relative paths such as `policy/pi05/...` or container paths under the mounted codebase. Do not use host absolute paths such as `/data/...` inside `debug_test`.

## High-Value Constraints

1. Use the full fine-tune target consistently.
   - The default target for this PRD is `pi05_pick_banana_full_mixed`.
   - Do not switch back to a LoRA config unless you intentionally change the training strategy and explain why.

2. Keep config name and dataset aligned.
   - `pi05_pick_banana_full_mixed` pairs with `pick_banana_mixed`.
   - Older LoRA configs are different training lines and should not be used by default here.

3. Prefer the lowest-risk edit surface.
   - First choice: `src/openpi/training/config.py`
   - Second choice: targeted model/config files if a concrete issue requires it
   - Last choice: `scripts/train.py`

4. Do not pass `--resume` and `--overwrite` together.

5. During debug validation, use small overrides such as reduced `--batch-size` and short `--num-train-steps`.

6. If you change evaluation-relevant naming, keep these identifiers consistent:
   - `train_config_name`
   - `--exp-name`
   - checkpoint directory under `checkpoints/<train_config_name>/<exp_name>`

## Codebase

Primary codebase inside the workspace:

- `policy/pi05`

Useful read-only references:

- `eval.sh`
- `README_FINETUNE.md`
- `run_step3_mixed.py`

## Allowed Modifications

You may edit only these files unless you find a concrete bug that forces a narrowly scoped exception:

1. `src/openpi/training/config.py`
2. `src/openpi/models/pi0.py`
3. `src/openpi/models/pi0_config.py`
4. `scripts/train.py`

Default expectation:

- Most successful solutions should primarily modify `src/openpi/training/config.py`.
- Do not edit `eval.sh`, deployment code, Docker, or K8S files as part of this task.

## Recommended Workflow

### Step 1: Inspect Before Editing

Read these first:

- `src/openpi/training/config.py`
- `scripts/train.py`
- `src/openpi/models/pi0.py`
- `src/openpi/models/pi0_config.py`
- `README_FINETUNE.md`
- `run_step3_mixed.py`

Confirm:

- the exact existing `pick_banana` config names
- which dataset each config expects
- default batch size / train steps / FSDP-related settings
- whether the likely bottleneck is memory, optimizer stability, or data/config mismatch

### Step 2: Choose Sensible Improvements

Good directions:

- tune batch size, learning rate, warmup, checkpoint cadence, or FSDP-related settings in `config.py`
- reduce memory pressure without changing task identity
- improve optimization stability while preserving the existing evaluation path

Bad directions:

- switching the task back to LoRA without a clear reason
- changing dataset identity casually
- rewriting the generic training loop without a concrete bug
- modifying infra files instead of model/training configuration

### Step 3: Validate With `debug_test`

You must run `debug_test` after editing code. The goal is not just import success; the goal is to prove the training command reaches the real training path.

Use the Pi05 repo root inside the mounted workspace as the working directory.

Environment rules for all `debug_test` commands:

- Do not activate `policy/pi05/.venv`.
- Do not use `source .venv/bin/activate`, `.venv/bin/python`, or `uv run`.
- Assume the runner already initializes `conda activate openpi`.
- If you need to select the interpreter explicitly, use `/opt/conda/envs/openpi/bin/python`.
- If imports fail, first add `PYTHONPATH=src:$PYTHONPATH`; do not switch to `.venv` as a fallback.

#### Required debug sequence

1. Config sanity check

```python
debug_test(
    command='cd policy/pi05 && PYTHONPATH=src:$PYTHONPATH /opt/conda/envs/openpi/bin/python -c "import openpi.training.config as c; cfg = c.get_config(\"pi05_pick_banana_full_mixed\"); print(cfg.name); print(cfg.data.repo_id); print(cfg.batch_size)"',
    timeout=60,
    working_dir="."
)
```

2. Mandatory short training check

```python
debug_test(
    command='cd policy/pi05 && PYTHONPATH=src:$PYTHONPATH /opt/conda/envs/openpi/bin/python scripts/train.py pi05_pick_banana_full_mixed --exp-name pick_banana_debug --data.repo-id pick_banana_mixed --batch-size 8 --num-train-steps 10 --overwrite',
    timeout=900,
    working_dir="."
)
```

Pass criteria:

- config loads successfully
- dataset loader initializes successfully
- training enters the real loop rather than failing immediately on setup
- no immediate crash from repo mismatch, checkpoint misuse, or obvious OOM

Optional deeper check:

```python
debug_test(
    command='cd policy/pi05 && PYTHONPATH=src:$PYTHONPATH /opt/conda/envs/openpi/bin/python scripts/train.py pi05_pick_banana_full_mixed --exp-name pick_banana_debug_50 --data.repo-id pick_banana_mixed --batch-size 8 --num-train-steps 50 --overwrite',
    timeout=1800,
    working_dir="."
)
```

### Step 4: Full Training Reference

After debug validation, the runner's full training command should stay equivalent to:

```bash
cd policy/pi05

python scripts/train.py pi05_pick_banana_full_mixed \
  --exp-name pick_banana_full_mixed_embomaster \
  --data.repo-id pick_banana_mixed \
  --overwrite
```

### Step 5: Evaluation Reference

Use the existing evaluation entrypoint. Do not create a new evaluation path.

```bash
bash eval.sh pick_banana <task_config> pi05_pick_banana_full_mixed <model_name> <seed> <gpu_id> [checkpoint_id]
```

Notes:

- `task_config` is runner-controlled. Do not hardcode a different one unless the task explicitly requires it.
- `model_name` must match the `--exp-name` used in training.
- If no checkpoint is specified, `eval.sh` defaults to checkpoint `30000`.

## Task Characteristics

`pick_banana` is a visually grounded pick-and-place task with language conditioning.

Typical failure modes:

- wrong arm choice under randomized object placement
- unstable lift or grasp release
- reaching the banana but failing final placement over the plate
- overfitting to one instruction style or one object arrangement

That means improvements that stabilize optimization or preserve visual-language grounding are usually more valuable than aggressive architectural changes.

## Optimization Priority

Use this order:

1. verify config and dataset alignment
2. remove memory or batch-size issues
3. improve optimization stability
4. only then consider targeted model-side changes

If you edit `scripts/train.py`, explain exactly what bug or missing hook forced that decision.

## Deliverables

When you finish, report:

1. which files changed
2. why each change was made
3. which `debug_test` commands were run
4. what evidence shows the short training check reached the real training path
5. what effect you expect on stability, memory usage, or success rate
