# OpenClaw StepTron Integration (swe-agents)

This directory adds an OpenClaw CLI pipeline (`SWE-OpenClaw`) to the swe-agents StepTron adaptor.
It follows the same pattern as `SWE-Claude-Code` / `SWE-Codex` / `SWE-KiloCode`:

- Allocate a SessionRouter container for the task image
- Start the local FastAPI proxy (`fastapi_server.py`) to provide an OpenAI-compatible `baseUrl`
- Run OpenClaw in embedded mode (`openclaw agent --local`) against that proxy
- Extract a small `meta.sft_data` transcript from `/tmp/fastapi_logs/*.jsonl`

## Task schema expectations

Required (in `task.swe_instance`):

- `instance_id`
- `docker_image`
- `workspace_root`
- `prompt`
- `openclaw_timeout_sec` (or `cc_timeout_sec` / `codex_timeout_sec`)
- `type`: `SWE-OpenClaw`

Optional:

- `openclaw_system_prompt` (prepended to `prompt`)
- `openclaw_model` (overrides `init_kwargs.model_names[0]`)
- `openclaw_max_tokens` (per-task output cap; forwarded to `OPENCLAW_MAX_TOKENS`)
- `keep_session_open`

## Environment variables

On the swe-agents host:

- `SESSION_ROUTER_USER_TOKEN` (or `SR_USER_TOKEN`)
- `MODEL_PROXY_API_KEY` (or pass `init_kwargs.api_key`)

Optional resource tuning:

- `SWE_OPENCLAW_REQUEST_CPU`, `SWE_OPENCLAW_REQUEST_MEMORY`
- `SWE_OPENCLAW_LIMIT_CPU`, `SWE_OPENCLAW_LIMIT_MEMORY`

Optional Stepcast host patching:

- `SWE_OPENCLAW_STEPCAST_HOST` (IP)
- `SWE_OPENCLAW_STEPCAST_HOST_ALIASES` (comma/space-separated, default `stepcast-router.basemind-core,stepcast-router`)
- `SWE_OPENCLAW_TRAIN_MODE=1` (forces /etc/hosts patching)

Transport policy knobs:

- `SWE_OPENCLAW_MODEL_API`:
  - `anthropic-messages` (default on Stepcast)
  - `openai-completions`
  - `openai-responses`
- `SWE_OPENCLAW_MAX_TOKENS` / `OPENCLAW_MAX_TOKENS`:
  - per-run generation max tokens (must be positive integer)
  - propagated to OpenClaw `agents.defaults.models.<provider/model>.params.maxTokens`
    so both OpenAI and Anthropic paths receive `options.maxTokens`
- `SWE_OPENCLAW_PROVIDER`:
  - `anthropic` (auto-selected when model API is `anthropic-messages`)
  - `openai`

Behavior:

- `anthropic-messages`: native `/v1/messages` path via the shared passthrough proxy.
- `openai-*`: OpenAI-compatible routing via local models-proxy + FastAPI.

## Node/OpenClaw install knobs

OpenClaw runtime is now strict:
- source bundle is resolved from repo-local `env_tools/openclaw/bin/` first
- those repo-local entries should stay as links into worker-mounted `/mnt/.../packages/...`
- pipeline uploads the resolved bundle to the fixed container path `/tmp/openclaw-runtime-bundle.tar.gz`
- `openclaw_init.sh` only accepts that fixed `/tmp` upload; env/path fallback is disabled

Optional:
- `SWE_OPENCLAW_RESUME_SKIP_EXISTING_UPLOADS`: default `1`; on resumed sessions skip re-uploading existing large tar artifacts in `/tmp`

## Logs

In the SessionRouter container:

- OpenClaw stdout/stderr: `/tmp/fastapi_logs/openclaw_logs_latest.txt`
- FastAPI proxy logs: `/tmp/fastapi_logs/*.jsonl`
- FastAPI server log: `/tmp/fastapi_log.txt`

## Batch runner

This directory now also contains `batch_openclaw_runner.py`, which can be used
to batch-submit tasks to the local OpenClaw CLI on the current machine and save
both trajectories and final answers.

### Input task file

Provide a JSON array. Each item may be either a string or an object.

Minimal object example:

```json
[
  {
    "id": "task_001",
    "prompt": "Solve the issue and explain the root cause.",
    "docker_image": "your-task-image",
    "workspace_root": "/workspace/task_001"
  },
  {
    "id": "task_002",
    "description": "Inspect the repository and summarize the bug."
  }
]
```

Supported per-task fields include:

- `id` / `instance_id`
- `prompt` / `description`
- `docker_image`
- `workspace_root`
- `openclaw_timeout_sec`
- `openclaw_system_prompt`
- `openclaw_model`
- `openclaw_max_tokens`
- `keep_session_open`

### Run example

```bash
python playground/openclaw/batch_openclaw_runner.py \
  --tasks playground/openclaw/tasks.json \
  --output-dir runs/openclaw_batch_001 \
  --endpoint https://your-endpoint.example/v1 \
  --model your-model-name \
  --parallel 4
```

If `--api-key` is omitted, the script reads:

- `MODEL_PROXY_API_KEY`, then
- `MODELPROXY_APIKEY`

### Output layout

Under `--output-dir`:

- `summary.json`: overall success/failure counts
- `results.jsonl`: one result row per task
- `<task_id>/trajectory.json`: full returned trajectory object
- `<task_id>/meta.json`: extracted pipeline meta
- `<task_id>/final_answer.txt`: last assistant answer found in dialogs
- `<task_id>/result.json`: per-task summary/error

### Important note

The batch runner no longer depends on `swe_agents`. It starts the local
`fastapi_server.py`, optionally starts `models_proxy.py`, runs:

```bash
openclaw agent --local --json
```

and then extracts dialogs from `fastapi_logs/*.jsonl` using
`openclaw_log_extract.py`'s logic.

## FastAPI bundle rebuild

If the private OpenClaw FastAPI proxy bundle ever needs to be rebuilt, use the
legacy image flow below:

```bash
sudo docker logout
sudo docker pull python:3.10-slim-buster
```

Builds should stay on `python:3.10-slim-buster`.
