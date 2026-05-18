const fs = require("fs");

const configPath = process.env.OPENCLAW_CONFIG_PATH || "/root/.openclaw/openclaw.json";
if (!fs.existsSync(configPath)) {
  throw new Error(`OpenClaw config not found: ${configPath}`);
}

const cfg = JSON.parse(fs.readFileSync(configPath, "utf8"));
cfg.models ??= {};
cfg.models.providers ??= {};

const sourceProviderId = process.env.OPENCLAW_SOURCE_PROVIDER || "custom-api-gpugeek-com";
let providerId = sourceProviderId;
let sourceProvider = cfg.models.providers[sourceProviderId];
if (!sourceProvider) {
  const candidates = Object.entries(cfg.models.providers).filter(([, provider]) => {
    return typeof provider?.baseUrl === "string" && provider.baseUrl.includes("gpugeek.com");
  });
  if (candidates.length === 1) {
    providerId = candidates[0][0];
    sourceProvider = candidates[0][1];
  }
}
if (!sourceProvider) {
  throw new Error(`Source OpenClaw provider not found: ${sourceProviderId}`);
}
if (typeof sourceProvider.apiKey !== "string" || !sourceProvider.apiKey) {
  throw new Error(`Source OpenClaw provider is missing apiKey: ${sourceProviderId}`);
}

const modelId = process.env.OPENCLAW_MODEL_ID || "Vendor2/GPT-5.4";
const baseUrl = process.env.OPENCLAW_BASE_URL || sourceProvider.baseUrl;
if (typeof baseUrl !== "string" || !baseUrl) {
  throw new Error("OpenClaw provider baseUrl is empty");
}
const providerTimeoutSeconds = Number(process.env.OPENCLAW_PROVIDER_TIMEOUT_SECONDS || 1800);
if (!Number.isFinite(providerTimeoutSeconds) || providerTimeoutSeconds <= 0) {
  throw new Error(`Invalid OPENCLAW_PROVIDER_TIMEOUT_SECONDS: ${process.env.OPENCLAW_PROVIDER_TIMEOUT_SECONDS}`);
}
const modelMaxTokens = Number(process.env.OPENCLAW_MODEL_MAX_TOKENS || 8192);
if (!Number.isFinite(modelMaxTokens) || modelMaxTokens <= 0) {
  throw new Error(`Invalid OPENCLAW_MODEL_MAX_TOKENS: ${process.env.OPENCLAW_MODEL_MAX_TOKENS}`);
}

const sourceModels = Array.isArray(sourceProvider.models) ? sourceProvider.models : [];
const sourceModel = sourceModels.find((model) => model?.id === modelId) || sourceModels[0] || {};
const completionsModel = {
  ...sourceModel,
  id: modelId,
  name: sourceModel.name && sourceModel.id === modelId ? sourceModel.name : `${modelId} (Custom Provider)`,
  maxTokens: modelMaxTokens,
  reasoning: false,
};
delete completionsModel.api;
delete completionsModel.compat;

cfg.models.providers[providerId] = {
  ...sourceProvider,
  baseUrl,
  api: "openai-completions",
  timeoutSeconds: providerTimeoutSeconds,
  models: [completionsModel],
};

// Drop the stale Responses provider left by older experimental builds. The
// gpugeek endpoint used here accepts Chat/Completions-compatible traffic, but
// returned HTTP 400 for OpenClaw's Responses+tools payloads during PaperBench.
if (providerId !== "openai") {
  delete cfg.models.providers.openai;
}

cfg.agents ??= {};
cfg.agents.defaults ??= {};
cfg.agents.defaults.model ??= {};
cfg.agents.defaults.model.primary = `${providerId}/${modelId}`;
cfg.agents.defaults.skipBootstrap = true;
delete cfg.agents.defaults.thinkingDefault;
// The host config can carry per-model default overrides from a different GPT
// version; remove them so the baked primary model is the one OpenClaw dispatches.
delete cfg.agents.defaults.models;
if (Array.isArray(cfg.agents.list)) {
  for (const entry of cfg.agents.list) {
    if (!entry || typeof entry !== "object") continue;
    entry.model ??= {};
    entry.model.primary = `${providerId}/${modelId}`;
    delete entry.models;
    delete entry.thinkingDefault;
  }
}

cfg.tools ??= {};
cfg.tools.web ??= {};
cfg.tools.web.search ??= {};
cfg.tools.web.search.enabled = true;
cfg.tools.web.search.provider = "duckduckgo";
cfg.tools.web.search.openaiCodex = {
  ...(cfg.tools.web.search.openaiCodex || {}),
  enabled: true,
  mode: "cached",
};

cfg.plugins ??= {};
cfg.plugins.entries ??= {};
cfg.plugins.entries.duckduckgo = {
  ...(cfg.plugins.entries.duckduckgo || {}),
  enabled: true,
};

fs.writeFileSync(configPath, JSON.stringify(cfg, null, 2) + "\n", { mode: 0o600 });
console.log(`Configured OpenClaw primary model: ${providerId}/${modelId}`);
console.log("Configured OpenClaw provider API: openai-completions");
console.log(`Configured OpenClaw provider timeout: ${providerTimeoutSeconds}s`);
console.log(`Configured OpenClaw model max tokens: ${modelMaxTokens}`);
console.log("Enabled managed web_search provider: duckduckgo");
console.log("Enabled OpenClaw duckduckgo plugin");
console.log("Enabled native Codex web search config in cached mode");
