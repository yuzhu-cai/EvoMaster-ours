ARG BASE_IMAGE=pb-env:latest
FROM ${BASE_IMAGE}

ARG NODE_VERSION=22.16.0
ARG CODEX_VERSION=0.125.0
ARG BAKE_CODEX_CONFIG=0

ENV CODEX_HOME=/root/.codex

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        xz-utils; \
    rm -rf /var/lib/apt/lists/*; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
        amd64) node_arch="x64" ;; \
        arm64) node_arch="arm64" ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSLo /tmp/node.tar.xz "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz"; \
    tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1; \
    rm /tmp/node.tar.xz; \
    node --version; \
    npm --version; \
    npm install -g "@openai/codex@${CODEX_VERSION}"; \
    npm cache clean --force; \
    codex --version

RUN --mount=type=secret,id=codex_home_tar,target=/tmp/codex_home.tar.gz,required=false \
    set -eux; \
    mkdir -p "${CODEX_HOME}"; \
    if [ "${BAKE_CODEX_CONFIG}" = "1" ]; then \
        test -s /tmp/codex_home.tar.gz; \
        tar -xzf /tmp/codex_home.tar.gz -C "${CODEX_HOME}"; \
        chmod -R go-rwx "${CODEX_HOME}"; \
    else \
        echo "Codex host config not baked. Mount \$HOME/.codex to ${CODEX_HOME} at runtime if needed."; \
    fi

WORKDIR /home

CMD ["bash"]
