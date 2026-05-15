#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-pb-env:latest}"
IMAGE="${IMAGE:-pb-env-codex:0.125.0}"
CODEX_VERSION="${CODEX_VERSION:-0.125.0}"
NODE_VERSION="${NODE_VERSION:-22.16.0}"
BAKE_CODEX_CONFIG="${BAKE_CODEX_CONFIG:-1}"
CODEX_HOME_SRC="${CODEX_HOME_SRC:-${HOME}/.codex}"

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "Base image not found: ${BASE_IMAGE}" >&2
  exit 1
fi

secret_args=()
tmp_tar=""
tmp_stage=""

cleanup() {
  if [[ -n "${tmp_tar}" && -f "${tmp_tar}" ]]; then
    rm -f "${tmp_tar}"
  fi
  if [[ -n "${tmp_stage}" && -d "${tmp_stage}" ]]; then
    rm -rf "${tmp_stage}"
  fi
}
trap cleanup EXIT

if [[ "${BAKE_CODEX_CONFIG}" == "1" ]]; then
  if [[ ! -d "${CODEX_HOME_SRC}" ]]; then
    echo "Codex config directory not found: ${CODEX_HOME_SRC}" >&2
    exit 1
  fi

  tmp_tar="$(mktemp -t codex-home.XXXXXX.tar.gz)"
  tmp_stage="$(mktemp -d -t codex-home-stage.XXXXXX)"
  chmod 600 "${tmp_tar}"

  for name in config.toml .env; do
    src="${CODEX_HOME_SRC}/${name}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${tmp_stage}/"
    fi
  done

  if [[ ! -f "${tmp_stage}/config.toml" ]]; then
    echo "Missing required Codex config file: ${CODEX_HOME_SRC}/config.toml" >&2
    exit 1
  fi

  tar -C "${tmp_stage}" -czf "${tmp_tar}" .
  secret_size="$(stat -c '%s' "${tmp_tar}")"
  if [[ "${secret_size}" -gt 512000 ]]; then
    echo "Packaged Codex config is too large for Docker BuildKit secrets: ${secret_size} bytes" >&2
    echo "Reduce files in ${CODEX_HOME_SRC} or build with BAKE_CODEX_CONFIG=0 and mount config at runtime." >&2
    exit 1
  fi
  secret_args=(--secret "id=codex_home_tar,src=${tmp_tar}")
fi

DOCKER_BUILDKIT=1 docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "CODEX_VERSION=${CODEX_VERSION}" \
  --build-arg "NODE_VERSION=${NODE_VERSION}" \
  --build-arg "BAKE_CODEX_CONFIG=${BAKE_CODEX_CONFIG}" \
  "${secret_args[@]}" \
  -t "${IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm "${IMAGE}" bash -lc 'codex --version && test -d "$CODEX_HOME" && ls -la "$CODEX_HOME" >/dev/null'

cat <<EOF
Built ${IMAGE}

Codex version:
  ${CODEX_VERSION}

Config mode:
  BAKE_CODEX_CONFIG=${BAKE_CODEX_CONFIG}

If BAKE_CODEX_CONFIG=1, the image contains ${CODEX_HOME_SRC}/config.toml
and ${CODEX_HOME_SRC}/.env when present.
Keep the image private.
EOF
