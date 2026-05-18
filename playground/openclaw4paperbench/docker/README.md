# OpenClaw for PaperBench Docker Image

This directory builds a PaperBench agent image derived from `pb-env:latest`
with the OpenClaw CLI installed. By default the build pins OpenClaw to the
version installed on the host, currently expected to be `2026.5.12`, and uses
Node `24.15.0` to match the current host runtime.

The image can optionally bake the host OpenClaw config into `/root/.openclaw`.
Only `${HOME}/.openclaw/openclaw.json` and `${HOME}/.openclaw/.env` are copied.
The small config bundle is passed through a BuildKit secret, so it is not copied
into the Docker build context or this repository.

## Build with host OpenClaw config

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
IMAGE=pb-env-openclaw:2026.5.12 playground/openclaw4paperbench/docker/build-with-host-openclaw.sh
```

Keep the resulting image private because `openclaw.json` may contain provider
API keys.

## Build without baking config

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
BAKE_OPENCLAW_CONFIG=0 IMAGE=pb-env-openclaw:2026.5.12 playground/openclaw4paperbench/docker/build-with-host-openclaw.sh
```

Then mount the host config for ad-hoc inspection:

```bash
docker run --rm \
  -v "${HOME}/.openclaw:/root/.openclaw:ro" \
  pb-env-openclaw:2026.5.12 \
  openclaw --version
```

PaperBench runs normally use the baked config because the agent container is
created by PaperBench rather than by a hand-written `docker run`.

## PaperBench usage

Use this as the PaperBench rollout image:

```bash
paperbench.docker_image=pb-env-openclaw:2026.5.12
```

The custom solver rewrites the OpenClaw workspace to `/home` in a per-run
runtime config so the agent reads `/home/paper` and writes `/home/submission`
inside the PaperBench container instead of using the host workspace path stored
in `~/.openclaw/openclaw.json`.
