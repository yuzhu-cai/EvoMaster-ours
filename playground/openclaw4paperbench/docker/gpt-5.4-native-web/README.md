# OpenClaw GPT PaperBench Image

This directory builds a PaperBench agent image from
`pb-env-openclaw:2026.5.12`.

The image keeps the baked host OpenClaw credentials from the base image, but
rewires the selected model while preserving the gpugeek OpenAI-compatible
completions API mode:

```text
provider: custom-api-gpugeek-com
api: openai-completions
model: Vendor2/GPT-5.4 or Vendor2/GPT-5.5
timeoutSeconds: 1800
maxTokens: 8192
managed web_search: enabled, provider=duckduckgo, plugin enabled
```

This intentionally avoids the previous experimental `openai-responses` patch.
In PaperBench, that Responses payload returned HTTP 400 from the gpugeek
endpoint before the agent could produce code. The native Codex web-search config
is enabled in cached mode (`tools.web.search.openaiCodex.enabled=true`,
`mode=cached`), matching the host-style OpenClaw setting, while the chat traffic
continues to use the completions-compatible API.

Note: OpenClaw only activates native Codex search for Codex-capable providers.
For this custom completions-compatible provider, the image explicitly enables
the managed OpenClaw `web_search` tool with the keyless `duckduckgo` provider,
so the agent still has a real search tool available.

The image also sets `agents.defaults.skipBootstrap=true` so fresh PaperBench
workspaces do not spend the first turn asking the operator to define an
OpenClaw persona. It removes stale per-model default overrides inherited from
the host config, and sets `models.providers.custom-api-gpugeek-com.timeoutSeconds`
to `1800` so slow PaperBench-sized prompts are not cut off by OpenClaw's default
120-second model idle timeout. The image patches OpenClaw's embedded runner so
non-local OpenAI-compatible providers honor that configured provider timeout
instead of clamping the idle watchdog to 120 seconds. It also caps the model
`maxTokens` at `8192`, leaving more room for large PaperBench prompts and
avoiding compaction/overflow requests that the GPT-5.4-compatible endpoint can
reject with HTTP 400.

Build GPT-5.4:

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
playground/openclaw4paperbench/docker/gpt-5.4-native-web/build.sh
```

Build GPT-5.5:

```bash
cd /data/yuzhu/Devs/EvoMaster-ours
GPT_VERSION=5.5 playground/openclaw4paperbench/docker/gpt-5.4-native-web/build.sh
```

Default image names:

```text
pb-env-openclaw-gpt-5.4-web:2026.5.12
pb-env-openclaw-gpt-5.5-web:2026.5.12
```

PaperBench usage:

```bash
AGENT_IMAGE=pb-env-openclaw-gpt-5.4-web:2026.5.12 \
  playground/openclaw4paperbench/run-paperbench.sh ...
```
