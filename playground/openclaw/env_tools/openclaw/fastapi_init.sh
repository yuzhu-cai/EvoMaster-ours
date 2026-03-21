#!/bin/bash

set -euo pipefail

# NOTE: This script runs in SessionRouter containers. Some images ship a
# non-interactive ~/.bashrc that assumes PS1 exists; under `set -u` that can
# crash the pipeline. Source it best-effort with nounset disabled.
if [[ -f "$HOME/.bashrc" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" || true
    set -u
fi

# Best-effort cleanup: if a previous run left a FastAPI process around,
# kill it before starting a new one. (This prevents stale /tmp/fastapi.ini
# from pointing at a dead or conflicting server.)
if [[ -f /tmp/fastapi.ini ]]; then
    old_pids=$(grep 'pid = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)
    for pi in $old_pids; do
        kill -9 "$pi" 2>/dev/null || true
    done
fi

rm -f /tmp/fastapi.ini
rm -rf /tmp/fastapi-3.10-bin
mkdir -p /tmp/fastapi-3.10-bin
mkdir -p /tmp/fastapi_logs

FASTAPI_STARTUP_TIMEOUT="${FASTAPI_STARTUP_TIMEOUT:-600}"

model_endpoint="${1:-}"
if [[ -z "${model_endpoint}" ]]; then
    echo "Error: missing model endpoint argument." >&2
    echo "Usage: $0 <model_endpoint>" >&2
    exit 1
fi

# Stepcast upstream runs on the cluster network and must not go through external proxies.
# Some packaged FastAPI binaries may still honor proxy env vars even if the Python
# implementation sets trust_env=False, so defensively drop proxies in stepcast mode.
if [[ "${model_endpoint}" == "stepcast" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
    # Ensure localhost + stepcast router hostnames bypass proxies for any subprocesses.
    _np="${NO_PROXY:-${no_proxy:-}}"
    _add="127.0.0.1,localhost,stepcast-router.basemind-core,stepcast-router"
    if [[ -n "${_np}" ]]; then
        export NO_PROXY="${_np},${_add}"
        export no_proxy="${_np},${_add}"
    else
        export NO_PROXY="${_add}"
        export no_proxy="${_add}"
    fi
fi

pkill -9 -f "/tmp/fastapi-3.10-bin/fastapi_server.dist/fastapi_server.bin" 2>/dev/null || true
pkill -9 -f "/tmp/fastapi_server.py --model-endpoint" 2>/dev/null || true

FASTAPI_BOOT_MODE="${FASTAPI_BOOT_MODE:-bundle}"
fastapi_py_path="/tmp/fastapi_server.py"
fastapi_bin_path="/tmp/fastapi-3.10-bin/fastapi_server.dist/fastapi_server.bin"
start_desc=""
start_cmd=()

case "${FASTAPI_BOOT_MODE}" in
    auto)
        if [[ -f "${fastapi_py_path}" ]]; then
            FASTAPI_BOOT_MODE="python"
        else
            FASTAPI_BOOT_MODE="bundle"
        fi
        ;;
    python|bundle)
        ;;
    *)
        echo "Error: unsupported FASTAPI_BOOT_MODE=${FASTAPI_BOOT_MODE} (expected auto|python|bundle)." >&2
        exit 2
        ;;
esac

if [[ "${FASTAPI_BOOT_MODE}" == "python" ]]; then
    if [[ ! -f "${fastapi_py_path}" ]]; then
        echo "Error: requested python fastapi bootstrap but source is missing: ${fastapi_py_path}" >&2
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: requested python fastapi bootstrap but python3 is unavailable." >&2
        exit 2
    fi
    start_desc="fastapi_server.py (uploaded source)"
    start_cmd=(python3 "${fastapi_py_path}" --model-endpoint "${model_endpoint}")
fi

if [[ "${FASTAPI_BOOT_MODE}" == "bundle" ]]; then
    fastapi_tarball=""
    for cand in \
        /tmp/fastapi-proxy-v14-3.10-bin-v2.tar.gz \
        /tmp/fastapi-proxy-v13-3.10-bin-v2.tar.gz; do
        if [[ -f "$cand" ]]; then
            fastapi_tarball="$cand"
            break
        fi
    done

    if [[ -z "$fastapi_tarball" ]]; then
        echo "Error: fastapi tarball not found in /tmp." >&2
        ls -la /tmp | head -n 80 >&2 || true
        exit 2
    fi

    if [[ "$fastapi_tarball" == *.tar.gz ]]; then
        tar -xzvf "$fastapi_tarball" -C /tmp/fastapi-3.10-bin
    else
        tar -xvf "$fastapi_tarball" -C /tmp/fastapi-3.10-bin
    fi

    if [[ ! -x "$fastapi_bin_path" ]]; then
        echo "Error: fastapi binary missing or not executable: $fastapi_bin_path" >&2
        exit 2
    fi

    start_desc="fastapi_server.bin ($(basename "$fastapi_tarball"))"
    start_cmd=("${fastapi_bin_path}" --model-endpoint "${model_endpoint}")
fi

nohup "${start_cmd[@]}" > /tmp/fastapi_log.txt 2>&1 & disown

for ((i=1; i<=FASTAPI_STARTUP_TIMEOUT; i++)); do
    if [[ -f "/tmp/fastapi.ini" ]]; then
        ports=$(grep 'port = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)
        port=$(echo "$ports" | tail -n 1)
        pids=$(grep 'pid = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)
        pid=$(echo "$pids" | tail -n 1)
        if [[ -z "$port" ]]; then
            echo "Error: fastapi server failed to get port." >&2
            exit 1
        fi
        if [[ -z "$pid" ]]; then
            echo "Error: fastapi server failed to get pid." >&2
            exit 1
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Error: fastapi server process is not running (pid=$pid)." >&2
            if [[ -f "/tmp/fastapi_log.txt" ]]; then
                echo "---- fastapi_log.txt (tail) ----" >&2
                tail -n 200 /tmp/fastapi_log.txt >&2 || true
            fi
            exit 1
        fi

        # /tmp/fastapi.ini is written very early by some proxy variants.
        # Ensure the port is actually accepting connections before proceeding.
        if python3 - <<PY
import socket, sys
port = int("${port}")
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(2)
finally:
    s.close()
PY
        then
            break
        fi
        # Not ready yet; keep waiting (until timeout).
        sleep 1
        continue
    fi
    if (( i == FASTAPI_STARTUP_TIMEOUT )); then
        echo "Error: fastapi server failed to start in ${FASTAPI_STARTUP_TIMEOUT}s (${start_desc})." >&2
        if [[ -f "/tmp/fastapi_log.txt" ]]; then
            echo "---- fastapi_log.txt (tail) ----" >&2
            tail -n 200 /tmp/fastapi_log.txt >&2 || true
        fi
        if [[ -f "/tmp/fastapi.ini" ]]; then
            pids=$(grep 'pid = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)
            for pi in $pids; do
                kill -9 "$pi" || true
            done
        fi
        ps -ef | grep -E "/tmp/fastapi-3.10-bin/fastapi_server.dist/fastapi_server.bin|/tmp/fastapi_server.py --model-endpoint" | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
        exit 1
    fi
    sleep 1
done
