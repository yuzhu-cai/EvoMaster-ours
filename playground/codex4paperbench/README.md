# Codex for PaperBench

This directory contains the local PaperBench + Codex experiment wrapper. It does
not modify the PaperBench checkout.

The current derived image name is:

```bash
pb-env-codex:0.125.0
```

Docker image build assets are under [`docker/`](docker/).

## Execution Model

PaperBench is controlled by the host conda environment:

```text
host conda env: paperbench
  runs paperbench.nano.entrypoint
    starts agent Docker image: pb-env-codex:0.125.0
      runs codex exec inside the PaperBench agent container
```

The custom solver is [`codex4paperbench/solver.py`](codex4paperbench/solver.py).
It is imported by the host PaperBench process via `PYTHONPATH`.

## Smoke Run

From the EvoMaster repo root:

```bash
playground/codex4paperbench/run-paperbench.sh
```

With no arguments, the wrapper runs:

- `paperbench.paper_split=debug`
- `paperbench.solver.time_limit=300`
- `paperbench.reproduction.skip_reproduction=True`
- `paperbench.judge.scaffold=dummy`
- `runner.max_retries=0`

This still invokes Codex for the `rice` debug paper.

## Custom Run

Pass any PaperBench/chz overrides after the wrapper:

```bash
playground/codex4paperbench/run-paperbench.sh \
  paperbench.paper_split=debug \
  paperbench.solver.time_limit=1800 \
  paperbench.reproduction.skip_reproduction=True \
  paperbench.judge.scaffold=dummy \
  runner.max_retries=0
```

Full reproduction and real judging are controlled by PaperBench flags. For
example, remove `paperbench.reproduction.skip_reproduction=True` and set a real
judge config when you want full scoring.

Useful environment overrides:

```bash
AGENT_IMAGE=pb-env-codex:0.125.0
REPRO_IMAGE=pb-reproducer:latest
CONDA_ENV=paperbench
FRONTIER_EVALS_ROOT=/data/yuzhu/Devs/third_party/frontier-evals
RUNS_DIR=/data/yuzhu/Devs/EvoMaster-ours/runs/codex4paperbench
```
