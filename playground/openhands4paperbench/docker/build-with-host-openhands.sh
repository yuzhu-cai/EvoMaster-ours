#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-pb-env:latest}"
OPENHANDS_PACKAGE="${OPENHANDS_PACKAGE:-openhands}"
OPENHANDS_VERSION="${OPENHANDS_VERSION:-1.16.0}"
OPENHANDS_PYTHON_VERSION="${OPENHANDS_PYTHON_VERSION:-3.12}"
IMAGE="${IMAGE:-pb-env-openhands:${OPENHANDS_VERSION}}"
BAKE_OPENHANDS_CONFIG="${BAKE_OPENHANDS_CONFIG:-0}"
OPENHANDS_HOME_SRC="${OPENHANDS_HOME_SRC:-${HOME}/.openhands}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-}"

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

if [[ "${BAKE_OPENHANDS_CONFIG}" == "1" ]]; then
  if [[ ! -d "${OPENHANDS_HOME_SRC}" ]]; then
    echo "OpenHands config directory not found: ${OPENHANDS_HOME_SRC}" >&2
    exit 1
  fi

  tmp_tar="$(mktemp -t openhands-home.XXXXXX.tar.gz)"
  tmp_stage="$(mktemp -d -t openhands-home-stage.XXXXXX)"
  chmod 600 "${tmp_tar}"

  for name in settings.json .env; do
    src="${OPENHANDS_HOME_SRC}/${name}"
    if [[ -f "${src}" ]]; then
      cp -a "${src}" "${tmp_stage}/"
    fi
  done

  if [[ ! -f "${tmp_stage}/settings.json" && ! -f "${tmp_stage}/.env" ]]; then
    echo "No settings.json or .env found under ${OPENHANDS_HOME_SRC}" >&2
    exit 1
  fi

  tar -C "${tmp_stage}" -czf "${tmp_tar}" .
  secret_size="$(stat -c '%s' "${tmp_tar}")"
  if [[ "${secret_size}" -gt 512000 ]]; then
    echo "Packaged OpenHands config is too large for Docker BuildKit secrets: ${secret_size} bytes" >&2
    echo "Only settings.json and .env are copied; inspect ${OPENHANDS_HOME_SRC} or build with BAKE_OPENHANDS_CONFIG=0." >&2
    exit 1
  fi
  secret_args=(--secret "id=openhands_home_tar,src=${tmp_tar}")
fi

network_args=()
if [[ -n "${DOCKER_BUILD_NETWORK}" ]]; then
  network_args=(--network "${DOCKER_BUILD_NETWORK}")
fi

proxy_args=()
for proxy_key in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_key:-}" ]]; then
    proxy_args+=(--build-arg "${proxy_key}=${!proxy_key}")
  fi
done

DOCKER_BUILDKIT=1 docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${network_args[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OPENHANDS_PACKAGE=${OPENHANDS_PACKAGE}" \
  --build-arg "OPENHANDS_VERSION=${OPENHANDS_VERSION}" \
  --build-arg "OPENHANDS_PYTHON_VERSION=${OPENHANDS_PYTHON_VERSION}" \
  --build-arg "BAKE_OPENHANDS_CONFIG=${BAKE_OPENHANDS_CONFIG}" \
  "${proxy_args[@]}" \
  "${secret_args[@]}" \
  -t "${IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm "${IMAGE}" bash -lc 'openhands -v && openhands --help | grep -q -- "--headless" && test -f /opt/openhands4paperbench/google_search_mcp.py && test -f "$OPENHANDS_HOME/mcp.json"'

cat <<EOF
Built ${IMAGE}

OpenHands package:
  ${OPENHANDS_PACKAGE} ${OPENHANDS_VERSION}

Config mode:
  BAKE_OPENHANDS_CONFIG=${BAKE_OPENHANDS_CONFIG}

If BAKE_OPENHANDS_CONFIG=1, the image contains ${OPENHANDS_HOME_SRC}/settings.json
and ${OPENHANDS_HOME_SRC}/.env when present. Keep the image private if either
file contains API keys.
EOF
