ARG BASE_IMAGE=pb-env:latest
FROM ${BASE_IMAGE}

ARG OPENHANDS_PACKAGE=openhands
ARG OPENHANDS_VERSION=1.16.0
ARG OPENHANDS_PYTHON_VERSION=3.12
ARG BAKE_OPENHANDS_CONFIG=0
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV OPENHANDS_HOME=/root/.openhands
ENV PATH="/root/.local/bin:${PATH}"

COPY mcp/google_search_mcp.py /opt/openhands4paperbench/google_search_mcp.py

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git; \
    rm -rf /var/lib/apt/lists/*; \
    python -m pip install --no-cache-dir --timeout 120 --retries 10 --resume-retries 10 \
        uv \
        "mcp>=1.1.0" \
        "httpx>=0.27.0"; \
    chmod 755 /opt/openhands4paperbench/google_search_mcp.py; \
    if [ -n "${OPENHANDS_VERSION}" ]; then \
        install_spec="${OPENHANDS_PACKAGE}==${OPENHANDS_VERSION}"; \
    else \
        install_spec="${OPENHANDS_PACKAGE}"; \
    fi; \
    UV_HTTP_TIMEOUT=120 uv tool install "${install_spec}" --python "${OPENHANDS_PYTHON_VERSION}"; \
    openhands -v; \
    openhands --help | grep -q -- '--headless'

RUN /root/.local/share/uv/tools/openhands/bin/python - <<'PY'
from pathlib import Path

path = Path("/root/.local/share/uv/tools/openhands/lib/python3.12/site-packages/openhands/sdk/llm/llm.py")
text = path.read_text()
if "OPENHANDS_FORCE_CHAT_COMPLETION" not in text:
    old = '''    def uses_responses_api(self) -> bool:
        """Whether this model uses the OpenAI Responses API path."""

        # by default, uses = supports
        return get_features(self._model_name_for_capabilities()).supports_responses_api
'''
    new = '''    def uses_responses_api(self) -> bool:
        """Whether this model uses the OpenAI Responses API path."""

        force_chat = os.environ.get("OPENHANDS_FORCE_CHAT_COMPLETION", "").lower()
        if force_chat in {"1", "true", "yes", "on"}:
            return False

        # by default, uses = supports
        return get_features(self._model_name_for_capabilities()).supports_responses_api
'''
    if old not in text:
        raise RuntimeError("OpenHands LLM.uses_responses_api patch target not found")
    text = text.replace(old, new)

if "OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM" not in text:
    old = '''                    ret = litellm_responses(
                        model=self.model,
                        input=typed_input,
                        instructions=instructions,
                        tools=resp_tools,
                        api_key=api_key_value,
                        api_base=self.base_url,
                        api_version=self.api_version,
                        timeout=self.timeout,
                        drop_params=self.drop_params,
                        seed=self.seed,
                        **{**self._aws_kwargs(), **final_kwargs},
                    )
                    if isinstance(ret, ResponsesAPIResponse):
'''
    new = '''                    if os.environ.get("OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM", "").lower() in {"1", "true", "yes", "on"}:
                        from openai import OpenAI as _OpenAI

                        sdk_model = self.model.removeprefix("openai/")
                        sdk_tools = [
                            t.model_dump(exclude_none=True)
                            if hasattr(t, "model_dump")
                            else dict(t)
                            for t in (resp_tools or [])
                        ] or None
                        sdk_kwargs = {**final_kwargs}
                        sdk_kwargs.pop("stream", None)
                        if os.environ.get("OPENHANDS_RESPONSES_STORE", "true").lower() not in {"0", "false", "no", "off"}:
                            # OpenHands carries Responses reasoning items across turns.
                            # Vendor-compatible Responses endpoints reject those item
                            # references unless the prior turn was stored.
                            sdk_kwargs.setdefault("store", True)
                        sdk_client = _OpenAI(
                            api_key=api_key_value,
                            base_url=self.base_url,
                        )
                        sdk_stream = sdk_client.responses.create(
                            model=sdk_model,
                            input=typed_input,
                            instructions=instructions,
                            tools=sdk_tools,
                            stream=True,
                            timeout=self.timeout,
                            **sdk_kwargs,
                        )
                        collected_output_items = []
                        completed_resp = None
                        for event in sdk_stream:
                            evt_type = getattr(event, "type", None)
                            if evt_type == "response.output_item.done":
                                item = getattr(event, "item", None)
                                if item is not None:
                                    collected_output_items.append(item)
                            elif evt_type == "response.completed":
                                completed_resp = getattr(event, "response", None)

                        if completed_resp is None:
                            if not collected_output_items:
                                raise LLMNoResponseError(
                                    "OpenAI SDK Responses stream finished without a completed response"
                                )
                            import time as _time

                            completed_data = {
                                "id": "resp_openhands_sdk_stream_fallback",
                                "created_at": int(_time.time()),
                                "model": sdk_model,
                                "object": "response",
                                "output": [],
                                "status": "completed",
                            }
                        else:
                            completed_data = completed_resp.model_dump(exclude_none=False)
                        if collected_output_items:
                            completed_data["output"] = [
                                item.model_dump(exclude_none=False)
                                if hasattr(item, "model_dump")
                                else item
                                for item in collected_output_items
                            ]
                        completed_resp = ResponsesAPIResponse.model_validate(completed_data)
                        self._telemetry.on_response(completed_resp)
                        return completed_resp

                    ret = litellm_responses(
                        model=self.model,
                        input=typed_input,
                        instructions=instructions,
                        tools=resp_tools,
                        api_key=api_key_value,
                        api_base=self.base_url,
                        api_version=self.api_version,
                        timeout=self.timeout,
                        drop_params=self.drop_params,
                        seed=self.seed,
                        **{**self._aws_kwargs(), **final_kwargs},
                    )
                    if isinstance(ret, ResponsesAPIResponse):
'''
    if old not in text:
        raise RuntimeError("OpenHands LLM.responses LiteLLM call patch target not found")
    text = text.replace(old, new)

path.write_text(text)
PY

RUN --mount=type=secret,id=openhands_home_tar,target=/tmp/openhands_home.tar.gz,required=false \
    set -eux; \
    mkdir -p "${OPENHANDS_HOME}"; \
    if [ "${BAKE_OPENHANDS_CONFIG}" = "1" ]; then \
        test -s /tmp/openhands_home.tar.gz; \
        tar -xzf /tmp/openhands_home.tar.gz -C "${OPENHANDS_HOME}"; \
        chmod -R go-rwx "${OPENHANDS_HOME}"; \
    else \
        echo "OpenHands host config not baked. Mount \$HOME/.openhands to ${OPENHANDS_HOME} at runtime if needed."; \
    fi; \
    printf '%s\n' '{"mcpServers":{"google-search":{"command":"python3","args":["/opt/openhands4paperbench/google_search_mcp.py"]}}}' > "${OPENHANDS_HOME}/mcp.json"; \
    chmod 600 "${OPENHANDS_HOME}/mcp.json"; \
    skills_cache="${OPENHANDS_HOME}/cache/skills"; \
    skills_repo="${skills_cache}/public-skills"; \
    skills_origin="${skills_cache}/public-skills-origin.git"; \
    skills_work="${skills_cache}/public-skills-origin-work"; \
    rm -rf "${skills_repo}" "${skills_origin}" "${skills_work}"; \
    mkdir -p "${skills_work}/skills"; \
    git init -q "${skills_work}"; \
    git -C "${skills_work}" checkout -q -b main; \
    git -C "${skills_work}" config user.email "openhands4paperbench@example.invalid"; \
    git -C "${skills_work}" config user.name "openhands4paperbench"; \
    touch "${skills_work}/skills/.keep"; \
    git -C "${skills_work}" add skills/.keep; \
    git -C "${skills_work}" commit -q -m "Initialize empty public skills cache"; \
    git clone -q --bare "${skills_work}" "${skills_origin}"; \
    git clone -q "${skills_origin}" "${skills_repo}"; \
    rm -rf "${skills_work}"

WORKDIR /home

CMD ["bash"]
