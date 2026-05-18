ARG BASE_IMAGE=pb-env-openclaw:2026.5.12
FROM ${BASE_IMAGE}

ARG OPENCLAW_MODEL_ID=Vendor2/GPT-5.4
ARG OPENCLAW_BASE_URL=
ARG OPENCLAW_SOURCE_PROVIDER=custom-api-gpugeek-com
ARG OPENCLAW_PROVIDER_TIMEOUT_SECONDS=1800
ARG OPENCLAW_MODEL_MAX_TOKENS=8192

ENV OPENCLAW_HOME=/root/.openclaw
ENV OPENCLAW_STATE_DIR=/root/.openclaw
ENV OPENCLAW_CONFIG_PATH=/root/.openclaw/openclaw.json

COPY configure-openclaw-gpt-web.js patch-openclaw-idle-timeout.js /tmp/

RUN set -eux; \
    node /tmp/patch-openclaw-idle-timeout.js; \
    rm -f /tmp/patch-openclaw-idle-timeout.js

RUN set -eux; \
    OPENCLAW_MODEL_ID="${OPENCLAW_MODEL_ID}" \
    OPENCLAW_BASE_URL="${OPENCLAW_BASE_URL}" \
    OPENCLAW_SOURCE_PROVIDER="${OPENCLAW_SOURCE_PROVIDER}" \
    OPENCLAW_PROVIDER_TIMEOUT_SECONDS="${OPENCLAW_PROVIDER_TIMEOUT_SECONDS}" \
    OPENCLAW_MODEL_MAX_TOKENS="${OPENCLAW_MODEL_MAX_TOKENS}" \
    node /tmp/configure-openclaw-gpt-web.js; \
    rm -f /tmp/configure-openclaw-gpt-web.js

RUN set -eux; \
    openclaw --version; \
    openclaw config validate; \
    OPENCLAW_PROVIDER_TIMEOUT_SECONDS="${OPENCLAW_PROVIDER_TIMEOUT_SECONDS}" \
    OPENCLAW_MODEL_MAX_TOKENS="${OPENCLAW_MODEL_MAX_TOKENS}" \
    node -e 'const fs=require("fs"); const cfg=JSON.parse(fs.readFileSync(process.env.OPENCLAW_CONFIG_PATH,"utf8")); const expectedTimeout=Number(process.env.OPENCLAW_PROVIDER_TIMEOUT_SECONDS||1800); const expectedMaxTokens=Number(process.env.OPENCLAW_MODEL_MAX_TOKENS||8192); const primary=cfg.agents.defaults.model.primary; const [providerId, ...modelParts]=primary.split("/"); const modelId=modelParts.join("/"); const provider=cfg.models.providers[providerId]; const model=provider&&provider.models&&provider.models[0]; const search=cfg.tools?.web?.search; if(!provider||provider.api!=="openai-completions") throw new Error("Expected provider with api=openai-completions"); if(provider.timeoutSeconds!==expectedTimeout) throw new Error(`Expected provider.timeoutSeconds=${expectedTimeout}`); if(cfg.agents.defaults.models) throw new Error("Expected agents.defaults.models to be removed"); if(!model||model.id!==modelId) throw new Error("OpenClaw primary model does not match provider model"); if(model.maxTokens!==expectedMaxTokens) throw new Error(`Expected model.maxTokens=${expectedMaxTokens}`); if(model.reasoning!==false) throw new Error("Expected model.reasoning=false"); if(cfg.agents.defaults.skipBootstrap!==true) throw new Error("Expected agents.defaults.skipBootstrap=true"); if(search?.enabled!==true) throw new Error("Expected managed web_search enabled"); if(search?.provider!=="duckduckgo") throw new Error("Expected managed web_search provider=duckduckgo"); if(cfg.plugins?.entries?.duckduckgo?.enabled!==true) throw new Error("Expected duckduckgo plugin enabled"); if(search?.openaiCodex?.enabled!==true) throw new Error("Expected native Codex web search enabled"); if(search?.openaiCodex?.mode!=="cached") throw new Error("Expected native Codex web search mode=cached"); console.log(JSON.stringify({primary,provider:providerId,api:provider.api,timeoutSeconds:provider.timeoutSeconds,model:model.id,maxTokens:model.maxTokens,reasoning:model.reasoning,skipBootstrap:cfg.agents.defaults.skipBootstrap,managedWebSearch:{enabled:search.enabled,provider:search.provider,pluginEnabled:cfg.plugins.entries.duckduckgo.enabled},nativeWebSearch:search.openaiCodex},null,2));'

WORKDIR /home

CMD ["bash"]
