#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-pb-env-openclaw:2026.5.12}"
OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.5.12}"
GPT_VERSION="${GPT_VERSION:-5.4}"
IMAGE="${IMAGE:-pb-env-openclaw-gpt-${GPT_VERSION}-web:${OPENCLAW_VERSION}}"
OPENCLAW_MODEL_ID="${OPENCLAW_MODEL_ID:-Vendor2/GPT-${GPT_VERSION}}"
OPENCLAW_SOURCE_PROVIDER="${OPENCLAW_SOURCE_PROVIDER:-custom-api-gpugeek-com}"
OPENCLAW_BASE_URL="${OPENCLAW_BASE_URL:-}"
OPENCLAW_PROVIDER_TIMEOUT_SECONDS="${OPENCLAW_PROVIDER_TIMEOUT_SECONDS:-1800}"
OPENCLAW_MODEL_MAX_TOKENS="${OPENCLAW_MODEL_MAX_TOKENS:-8192}"

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "Base image not found: ${BASE_IMAGE}" >&2
  exit 1
fi

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OPENCLAW_MODEL_ID=${OPENCLAW_MODEL_ID}" \
  --build-arg "OPENCLAW_SOURCE_PROVIDER=${OPENCLAW_SOURCE_PROVIDER}" \
  --build-arg "OPENCLAW_BASE_URL=${OPENCLAW_BASE_URL}" \
  --build-arg "OPENCLAW_PROVIDER_TIMEOUT_SECONDS=${OPENCLAW_PROVIDER_TIMEOUT_SECONDS}" \
  --build-arg "OPENCLAW_MODEL_MAX_TOKENS=${OPENCLAW_MODEL_MAX_TOKENS}" \
  -t "${IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm "${IMAGE}" bash -lc 'openclaw --version && node - <<'"'"'NODE'"'"'
const fs = require("fs");
const cfg = JSON.parse(fs.readFileSync(process.env.OPENCLAW_CONFIG_PATH, "utf8"));
const primary = cfg.agents.defaults.model.primary;
const [providerId, ...modelParts] = primary.split("/");
const modelId = modelParts.join("/");
const provider = cfg.models.providers[providerId];
console.log(JSON.stringify({
  primary,
  provider: providerId,
  api: provider.api,
  baseUrl: provider.baseUrl,
  timeoutSeconds: provider.timeoutSeconds,
  model: modelId,
  maxTokens: provider.models[0].maxTokens,
  reasoning: provider.models[0].reasoning,
  skipBootstrap: cfg.agents.defaults.skipBootstrap,
  hasApiKey: Boolean(provider.apiKey),
  managedWebSearch: {
    enabled: cfg.tools.web.search.enabled,
    provider: cfg.tools.web.search.provider,
    pluginEnabled: cfg.plugins.entries.duckduckgo.enabled,
  },
  nativeWebSearch: cfg.tools.web.search.openaiCodex,
}, null, 2));
NODE'

cat <<EOF
Built ${IMAGE}

This is a patched OpenClaw PaperBench image:
  base image: ${BASE_IMAGE}
  model: ${OPENCLAW_MODEL_ID}
  provider: ${OPENCLAW_SOURCE_PROVIDER}
  provider API: openai-completions
  provider timeout: ${OPENCLAW_PROVIDER_TIMEOUT_SECONDS}s
  model max tokens: ${OPENCLAW_MODEL_MAX_TOKENS}
  model idle timeout cap: patched to honor provider timeout
  workspace bootstrap prompt: disabled
  managed web_search provider: duckduckgo plugin enabled
  native Codex web search config: enabled, cached mode

Keep this image private because it inherits the baked OpenClaw config from
${BASE_IMAGE}, which may contain API keys.
EOF
