# Codex for PaperBench Docker Image

This directory builds a PaperBench agent image derived from `pb-env:latest` with
Codex CLI pinned to the version documented in `playground/codex`: `0.125.0`.

The Dockerfile can optionally bake the host Codex config into `/root/.codex`.
Only `${HOME}/.codex/config.toml` and `${HOME}/.codex/.env` are copied. The
small config bundle is passed through a BuildKit secret, so it is not copied
into the Docker build context or this repository.

## Build with host Codex config

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
IMAGE=pb-env-codex:0.125.0 playground/codex4paperbench/docker/build-with-host-codex.sh
```

By default this copies `${HOME}/.codex/config.toml` and `${HOME}/.codex/.env`
into the image. Keep the resulting image private because `.env` may contain API
keys.

## Build without baking config

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
BAKE_CODEX_CONFIG=0 IMAGE=pb-env-codex:0.125.0 playground/codex4paperbench/docker/build-with-host-codex.sh
```

Then mount the host config at runtime:

```bash
docker run --rm \
  -v "${HOME}/.codex:/root/.codex:ro" \
  pb-env-codex:0.125.0 \
  codex --version
```

## PaperBench usage

Use this as the PaperBench rollout image:

```bash
paperbench.docker_image=pb-env-codex:0.125.0
```

If grading is configured to run in a container instead of locally, set the judge
container image separately:

```bash
paperbench.judge.computer_config.docker_image=pb-env-codex:0.125.0
```
