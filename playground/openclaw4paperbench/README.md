# OpenClaw for PaperBench

This directory contains the local PaperBench + OpenClaw experiment wrapper. It
does not modify the PaperBench checkout.

The default derived image name is:

```bash
pb-env-openclaw:2026.5.12
```

Docker image build assets are under [`docker/`](docker/).

## Execution Model

PaperBench is controlled by the host conda environment:

```text
host conda env: paperbench
  runs paperbench.nano.entrypoint
    starts agent Docker image: pb-env-openclaw:2026.5.12
      runs openclaw agent --local inside the PaperBench agent container
```

The custom solver is
[`openclaw4paperbench/solver.py`](openclaw4paperbench/solver.py). It is imported
by the host PaperBench process via `PYTHONPATH`.

The solver copies `/root/.openclaw/openclaw.json` to a per-run state directory
and rewrites only runtime execution settings:

- `agents.defaults.workspace` becomes `/home`
- `tools.exec.security` defaults to `full`
- `tools.exec.ask` defaults to `off`
- `tools.exec.host` defaults to `auto`

This keeps provider keys/model settings from the baked OpenClaw config while
making OpenClaw operate inside the PaperBench container workspace.

## Build Image

From the EvoMaster repo root:

```bash
IMAGE=pb-env-openclaw:2026.5.12 playground/openclaw4paperbench/docker/build-with-host-openclaw.sh
```

By default this copies only:

- `${HOME}/.openclaw/openclaw.json`
- `${HOME}/.openclaw/.env`, when present

Keep the resulting image private because the OpenClaw config may contain API
keys.

## Smoke Run

From the EvoMaster repo root:

```bash
playground/openclaw4paperbench/run-paperbench.sh
```

With no arguments, the wrapper runs:

- `paperbench.paper_split=debug`
- `paperbench.solver.time_limit=300`
- `paperbench.reproduction.skip_reproduction=True`
- `paperbench.judge.scaffold=dummy`
- `runner.max_retries=0`

This still invokes OpenClaw for the `rice` debug paper.

## Custom Run

Pass any PaperBench/chz overrides after the wrapper:

```bash
playground/openclaw4paperbench/run-paperbench.sh \
  paperbench.paper_split=debug \
  paperbench.solver.time_limit=1800 \
  paperbench.reproduction.skip_reproduction=True \
  paperbench.judge.scaffold=dummy \
  runner.max_retries=0
```

Useful solver overrides:

```bash
paperbench.solver.model=custom-api-gpugeek-com/Vendor2/GPT-5.4
paperbench.solver.thinking=medium
paperbench.solver.exec_security=full
paperbench.solver.exec_ask=off
paperbench.solver.web_search_provider=gemini
```

Useful environment overrides:

```bash
AGENT_IMAGE=pb-env-openclaw:2026.5.12
REPRO_IMAGE=pb-reproducer:latest
CONDA_ENV=paperbench
FRONTIER_EVALS_ROOT=/data/yuzhu/Devs/third_party/frontier-evals
RUNS_DIR=/data/yuzhu/Devs/EvoMaster-ours/runs/openclaw4paperbench
OPENCLAW_HOME_SRC=${HOME}/.openclaw
```

## Web Search

OpenClaw web search is controlled by the OpenClaw config baked into the image.
If the host config already has `tools.web.search.provider`, the solver preserves
it. You can also force a provider for a PaperBench run:

```bash
playground/openclaw4paperbench/run-paperbench.sh \
  paperbench.solver.web_search_provider=gemini \
  paperbench.paper_split=debug \
  paperbench.solver.time_limit=1800 \
  paperbench.reproduction.skip_reproduction=True \
  paperbench.judge.scaffold=dummy \
  runner.max_retries=0
```

The provider credentials must be available in `/root/.openclaw/openclaw.json`
or `/root/.openclaw/.env` inside the image.
