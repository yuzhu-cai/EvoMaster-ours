ARG BASE_IMAGE=pb-env:latest
FROM ${BASE_IMAGE}

ARG NODE_VERSION=24.15.0
ARG OPENCLAW_VERSION=2026.5.12
ARG BAKE_OPENCLAW_CONFIG=0

ENV OPENCLAW_HOME=/root/.openclaw
ENV OPENCLAW_STATE_DIR=/root/.openclaw
ENV OPENCLAW_CONFIG_PATH=/root/.openclaw/openclaw.json

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
    npm install -g "openclaw@${OPENCLAW_VERSION}"; \
    npm cache clean --force; \
    openclaw --version

RUN --mount=type=secret,id=openclaw_home_tar,target=/tmp/openclaw_home.tar.gz,required=false \
    set -eux; \
    mkdir -p "${OPENCLAW_HOME}"; \
    if [ "${BAKE_OPENCLAW_CONFIG}" = "1" ]; then \
        test -s /tmp/openclaw_home.tar.gz; \
        tar -xzf /tmp/openclaw_home.tar.gz -C "${OPENCLAW_HOME}"; \
        chmod -R go-rwx "${OPENCLAW_HOME}"; \
    else \
        echo "OpenClaw host config not baked. Mount \$HOME/.openclaw to ${OPENCLAW_HOME} at runtime if needed."; \
    fi

WORKDIR /home

CMD ["bash"]
