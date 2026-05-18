#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-pb-env:latest}"
OPENCLAW_VERSION="${OPENCLAW_VERSION:-}"
if [[ -z "${OPENCLAW_VERSION}" ]]; then
  if command -v openclaw >/dev/null 2>&1; then
    OPENCLAW_VERSION="$(openclaw --version | awk '{print $2}')"
  else
    OPENCLAW_VERSION="2026.5.12"
  fi
fi
IMAGE="${IMAGE:-pb-env-openclaw:${OPENCLAW_VERSION}}"
NODE_VERSION="${NODE_VERSION:-24.15.0}"
BAKE_OPENCLAW_CONFIG="${BAKE_OPENCLAW_CONFIG:-1}"
OPENCLAW_HOME_SRC="${OPENCLAW_HOME_SRC:-${HOME}/.openclaw}"

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

if [[ "${BAKE_OPENCLAW_CONFIG}" == "1" ]]; then
  if [[ ! -d "${OPENCLAW_HOME_SRC}" ]]; then
    echo "OpenClaw config directory not found: ${OPENCLAW_HOME_SRC}" >&2
    exit 1
  fi

  tmp_tar="$(mktemp -t openclaw-home.XXXXXX.tar.gz)"
  tmp_stage="$(mktemp -d -t openclaw-home-stage.XXXXXX)"
  chmod 600 "${tmp_tar}"

  for name in openclaw.json .env; do
    src="${OPENCLAW_HOME_SRC}/${name}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${tmp_stage}/"
    fi
  done

  if [[ ! -f "${tmp_stage}/openclaw.json" ]]; then
    echo "Missing required OpenClaw config file: ${OPENCLAW_HOME_SRC}/openclaw.json" >&2
    exit 1
  fi

  tar -C "${tmp_stage}" -czf "${tmp_tar}" .
  secret_size="$(stat -c '%s' "${tmp_tar}")"
  if [[ "${secret_size}" -gt 1048576 ]]; then
    echo "Packaged OpenClaw config is too large for Docker BuildKit secrets: ${secret_size} bytes" >&2
    echo "Only openclaw.json and .env are copied; inspect ${OPENCLAW_HOME_SRC} or build with BAKE_OPENCLAW_CONFIG=0." >&2
    exit 1
  fi
  secret_args=(--secret "id=openclaw_home_tar,src=${tmp_tar}")
fi

DOCKER_BUILDKIT=1 docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OPENCLAW_VERSION=${OPENCLAW_VERSION}" \
  --build-arg "NODE_VERSION=${NODE_VERSION}" \
  --build-arg "BAKE_OPENCLAW_CONFIG=${BAKE_OPENCLAW_CONFIG}" \
  "${secret_args[@]}" \
  -t "${IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm "${IMAGE}" bash -lc 'openclaw --version && test -d "$OPENCLAW_HOME" && ls -la "$OPENCLAW_HOME" >/dev/null'

cat <<EOF
Built ${IMAGE}

OpenClaw version:
  ${OPENCLAW_VERSION}

Config mode:
  BAKE_OPENCLAW_CONFIG=${BAKE_OPENCLAW_CONFIG}

If BAKE_OPENCLAW_CONFIG=1, the image contains ${OPENCLAW_HOME_SRC}/openclaw.json
and ${OPENCLAW_HOME_SRC}/.env when present. Keep the image private because the
OpenClaw config may contain provider API keys or gateway tokens.
EOF
