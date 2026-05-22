# OpenHands for PaperBench Docker Image

This image derives from `pb-env:latest` and installs the OpenHands CLI with
Python 3.12 via `uv tool install`.

It also installs a local MCP server that exposes `google_search` / `web_search`
via Serper and `web_fetch` via Jina Reader. The image contains the MCP server
code and a base `/root/.openhands/mcp.json`, but not the API keys. Pass keys
with `--env-file` or through `playground/openhands4paperbench/run-paperbench.sh`.

The image patches OpenHands' GPT-5-family routing in two ways:

- `OPENHANDS_FORCE_CHAT_COMPLETION=1` can still force the Chat Completions path.
- `OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=1` makes the Responses path call
  the official OpenAI SDK with `stream=True` instead of LiteLLM.

The image still contains the Responses API compatibility patch for debugging,
but the PaperBench wrapper defaults to `OPENHANDS_FORCE_CHAT_COMPLETION=true`
and `OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=false` for the current
`api.gpugeek.com` endpoint. In local PaperBench and direct OpenHands smoke runs,
the chat-completions path preserves MCP tool calling and avoids intermittent
Responses-path `ConversationErrorEvent` / `APIError` failures.

Build from the EvoMaster repo root:

```bash
playground/openhands4paperbench/docker/build-with-host-openhands.sh
```

If PyPI downloads need the host proxy on port 5890, build with host networking:

```bash
DOCKER_BUILD_NETWORK=host \
HTTP_PROXY=http://127.0.0.1:5890 \
HTTPS_PROXY=http://127.0.0.1:5890 \
  playground/openhands4paperbench/docker/build-with-host-openhands.sh
```

The default image name is:

```bash
pb-env-openhands:1.16.0
```

By default, host OpenHands settings are not baked into the image. For private
images only, you can copy `${HOME}/.openhands/settings.json` and `.env`:

```bash
BAKE_OPENHANDS_CONFIG=1 \
  playground/openhands4paperbench/docker/build-with-host-openhands.sh
```

For PaperBench runs, prefer passing model configuration via environment
variables on the host:

```bash
OPENAI_API_KEY=... \
LLM_MODEL=openai/Vendor2/GPT-5.4 \
LLM_BASE_URL=https://api.gpugeek.com/v1 \
LLM_REASONING_EFFORT=medium \
OPENHANDS_FORCE_CHAT_COMPLETION=true \
OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM=false \
LLM_NATIVE_TOOL_CALLING=true \
AGENT_IMAGE=pb-env-openhands:1.16.0 \
  playground/openhands4paperbench/run-paperbench.sh
```
