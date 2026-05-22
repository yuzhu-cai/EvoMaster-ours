# OpenHands for PaperBench

This directory contains the local PaperBench + OpenHands experiment wrapper. It
does not modify the PaperBench checkout.

The default derived image name is:

```bash
pb-env-openhands:1.16.0
```

Docker image build assets are under [`docker/`](docker/).

## Execution Model

PaperBench is controlled by the host conda environment:

```text
host conda env: paperbench
  runs paperbench.nano.entrypoint
    starts agent Docker image: pb-env-openhands:1.16.0
      runs openhands --headless inside the PaperBench agent container
```

The custom solver is
[`openhands4paperbench/solver.py`](openhands4paperbench/solver.py). It is
imported by the host PaperBench process via `PYTHONPATH`.

The solver runs OpenHands with `RUNTIME=process` so it operates directly inside
the PaperBench agent container. This avoids a nested Docker sandbox whose bind
mounts would be interpreted relative to the Docker host instead of the
PaperBench container filesystem.

OpenHands CLI headless mode does not run the interactive critic/refinement
path. To avoid accepting the first shallow `Done` message as final, the wrapper
adds an explicit continuation loop: after the first `openhands --headless` run,
it extracts the conversation id and resumes the same conversation with
`openhands --resume <conversation-id> --headless -f <followup>`. The follow-up
prompt asks the agent to review the current `/home/submission` against the
paper, addendum, and rubric, then replace scaffold or README-only claims with
concrete paper-specific code. Configure this with
`OPENHANDS_CONTINUATION_PASSES` or
`paperbench.solver.continuation_passes`.

The image also includes a local MCP server at
`/opt/openhands4paperbench/google_search_mcp.py`. The solver writes
`/root/.openhands/mcp.json` before each OpenHands run, enabling:

- `google_search`: Google results through Serper (`SERPER_KEY_ID`)
- `web_search`: alias for `google_search`, for agents that look for a generic web-search tool
- `web_fetch`: webpage-to-Markdown fetch through Jina Reader (`JINA_API_KEY`)

The wrapper sources `${EVOMASTER_ROOT}/.env` by default, so the project-level
Serper/Jina keys are passed at runtime instead of being baked into the image.

For the current `Vendor2/GPT-5.4` endpoint, the wrapper defaults to:

- `OPENHANDS_FORCE_CHAT_COMPLETION=true`
- `OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=false`
- `LLM_NATIVE_TOOL_CALLING=true`
- `LLM_REASONING_EFFORT=medium`
- `OPENHANDS_AGENT_RETRIES=1` can be set to retry failed OpenHands CLI exits inside one PaperBench task.
- `OPENHANDS_CONTINUATION_PASSES=1` runs one extra resume/self-review pass after the initial OpenHands finish.

This keeps OpenHands on the chat-completions path while preserving native tool
calling for MCP tools. In local smoke tests this path successfully calls
`google_search` and writes files, while the Responses streaming path can still
terminate PaperBench rollouts with intermittent `ConversationErrorEvent` /
`APIError` from the vendor endpoint. The image still contains the Responses API
compatibility patch, so the old path can be re-enabled explicitly for debugging.

For PaperBench Code-Dev runs, the wrapper now keeps the original
`code_only_instructions.txt` semantics: Code-Dev changes the grading path only.
The prompt explicitly asks OpenHands to implement concrete paper-specific code
paths rather than README-only scaffolds, while still avoiding full expensive
training runs, heavyweight dependency downloads, and forbidden prior
implementations. The container exports `PIP_NO_DEPS=1` so validation installs do
not pull packages like torch/CUDA wheels during Code-Dev runs.

## Build Image

From the EvoMaster repo root:

```bash
playground/openhands4paperbench/docker/build-with-host-openhands.sh
```

If you want to bake private OpenHands settings into the image:

```bash
BAKE_OPENHANDS_CONFIG=1 \
  playground/openhands4paperbench/docker/build-with-host-openhands.sh
```

Only `${HOME}/.openhands/settings.json` and `${HOME}/.openhands/.env` are
copied. Keep the resulting image private if either file contains API keys.

## Smoke Run

From the EvoMaster repo root:

```bash
OPENAI_API_KEY=... \
LLM_MODEL=openai/Vendor2/GPT-5.4 \
LLM_BASE_URL=https://api.gpugeek.com/v1 \
LLM_REASONING_EFFORT=medium \
OPENHANDS_FORCE_CHAT_COMPLETION=true \
OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=false \
LLM_NATIVE_TOOL_CALLING=true \
playground/openhands4paperbench/run-paperbench.sh
```

With no PaperBench arguments, the wrapper runs:

- `paperbench.paper_split=debug`
- `paperbench.solver.time_limit=300`
- `paperbench.reproduction.skip_reproduction=True`
- `paperbench.judge.scaffold=dummy`
- `runner.max_retries=0`

This still invokes OpenHands for the `rice` debug paper.

## Custom Run

Pass PaperBench/chz overrides after the wrapper:

```bash
OPENAI_API_KEY=... \
LLM_MODEL=openai/Vendor2/GPT-5.4 \
LLM_BASE_URL=https://api.gpugeek.com/v1 \
LLM_REASONING_EFFORT=medium \
playground/openhands4paperbench/run-paperbench.sh \
  paperbench.paper_split=debug \
  paperbench.solver.time_limit=1800 \
  paperbench.reproduction.skip_reproduction=True \
  paperbench.judge.scaffold=dummy \
  runner.max_retries=0
```

Useful solver overrides:

```bash
paperbench.solver.model=openai/Vendor2/GPT-5.4
paperbench.solver.base_url=https://api.gpugeek.com/v1
paperbench.solver.reasoning_effort=medium
paperbench.solver.native_tool_calling=True
paperbench.solver.force_chat_completion=True
paperbench.solver.use_openai_sdk_responses_stream=False
paperbench.solver.agent_retries=1
paperbench.solver.continuation_passes=1
paperbench.solver.runtime=process
paperbench.solver.always_approve=True
paperbench.solver.override_with_envs=True
```

Useful judge overrides for grading with the same `Vendor2/GPT-5.4` endpoint:

```bash
paperbench.judge.completer_config=openhands4paperbench.judge:VendorOpenAICompletionsTurnCompleter.Config
paperbench.judge.completer_config.model=Vendor2/GPT-5.4
paperbench.judge.completer_config.reasoning_effort=medium
```

The wrapper exports `OPENAI_BASE_URL=${LLM_BASE_URL}` when it is not already
set, so the host-side PaperBench judge also points at `https://api.gpugeek.com/v1`.
The local judge config only registers the vendor model alias in PaperBench's
context-window table; it does not change the judge prompt or scoring logic.

Useful environment overrides:

```bash
AGENT_IMAGE=pb-env-openhands:1.16.0
REPRO_IMAGE=pb-reproducer:latest
CONDA_ENV=paperbench
FRONTIER_EVALS_ROOT=/data/yuzhu/Devs/third_party/frontier-evals
RUNS_DIR=/data/yuzhu/Devs/EvoMaster-ours/runs/openhands4paperbench
ENV_FILE=/data/yuzhu/Devs/EvoMaster-ours/.env
LLM_REASONING_EFFORT=medium
LLM_NATIVE_TOOL_CALLING=true
OPENHANDS_FORCE_CHAT_COMPLETION=true
OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=false
OPENHANDS_AGENT_RETRIES=1
OPENHANDS_CONTINUATION_PASSES=1
OPENHANDS_HOME_SRC=${HOME}/.openhands
```
