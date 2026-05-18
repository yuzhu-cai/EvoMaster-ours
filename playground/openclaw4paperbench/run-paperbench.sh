#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOMASTER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-paperbench}"
FRONTIER_EVALS_ROOT="${FRONTIER_EVALS_ROOT:-/data/yuzhu/Devs/third_party/frontier-evals}"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-${FRONTIER_EVALS_ROOT}/project/paperbench}"
RUNS_DIR="${RUNS_DIR:-${EVOMASTER_ROOT}/runs/openclaw4paperbench}"
AGENT_IMAGE="${AGENT_IMAGE:-pb-env-openclaw:2026.5.12}"
REPRO_IMAGE="${REPRO_IMAGE:-pb-reproducer:latest}"
OPENCLAW_HOME_SRC="${OPENCLAW_HOME_SRC:-${HOME}/.openclaw}"

if [[ ! -d "${PAPERBENCH_ROOT}" ]]; then
  echo "PaperBench root not found: ${PAPERBENCH_ROOT}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if ! docker image inspect "${AGENT_IMAGE}" >/dev/null 2>&1; then
  echo "Agent image not found: ${AGENT_IMAGE}" >&2
  echo "Build it with: IMAGE=${AGENT_IMAGE} playground/openclaw4paperbench/docker/build-with-host-openclaw.sh" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PAPERBENCH_DATA_DIR="${PAPERBENCH_DATA_DIR:-${PAPERBENCH_ROOT}/data}"

if [[ "${OPENCLAW4PAPERBENCH_SOURCE_HOST_OPENCLAW_ENV:-1}" != "0" && -f "${OPENCLAW_HOME_SRC}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${OPENCLAW_HOME_SRC}/.env"
  set +a
fi

# PaperBench validates judge credentials early, even for some dry/debug paths.
# When the OpenClaw config stores an OpenAI-compatible provider directly, mirror
# the selected provider into the conventional env vars without printing secrets.
if [[ "${OPENCLAW4PAPERBENCH_SOURCE_OPENCLAW_MODEL_ENV:-1}" != "0" && -f "${OPENCLAW_HOME_SRC}/openclaw.json" ]]; then
  eval "$(
    OPENCLAW_CONFIG_JSON="${OPENCLAW_HOME_SRC}/openclaw.json" python3 - <<'PY'
import json
import os
import shlex

cfg_path = os.environ["OPENCLAW_CONFIG_JSON"]
try:
    cfg = json.load(open(cfg_path, encoding="utf-8"))
except Exception:
    raise SystemExit(0)

primary = (((cfg.get("agents") or {}).get("defaults") or {}).get("model") or {}).get("primary")
if not isinstance(primary, str) or "/" not in primary:
    raise SystemExit(0)

provider_id, model = primary.split("/", 1)
provider = ((cfg.get("models") or {}).get("providers") or {}).get(provider_id) or {}
api_key = provider.get("apiKey")
base_url = provider.get("baseUrl")
if not isinstance(api_key, str) or not api_key:
    raise SystemExit(0)

exports = {}
if not os.environ.get("OPENAI_API_KEY"):
    exports["OPENAI_API_KEY"] = api_key
if not os.environ.get("GRADER_OPENAI_API_KEY"):
    exports["GRADER_OPENAI_API_KEY"] = api_key
if isinstance(base_url, str) and base_url:
    if not os.environ.get("OPENAI_BASE_URL"):
        exports["OPENAI_BASE_URL"] = base_url
    if not os.environ.get("GPT_BASE_URL"):
        exports["GPT_BASE_URL"] = base_url
if not os.environ.get("GPT_CHAT_MODEL"):
    exports["GPT_CHAT_MODEL"] = model

for key, value in exports.items():
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi

if [[ -z "${GRADER_OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export GRADER_OPENAI_API_KEY="${OPENAI_API_KEY}"
fi

extra_args=(
  "paperbench.solver=openclaw4paperbench.solver:OpenClawCliSolver"
  "paperbench.docker_image=${AGENT_IMAGE}"
  "paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime"
  "paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig"
  "paperbench.solver.computer_runtime.env.pull_from_registry=false"
  "paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime"
  "paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig"
  "paperbench.reproduction.computer_runtime.env.pull_from_registry=false"
  "paperbench.reproduction.computer_config.docker_image=${REPRO_IMAGE}"
  "paperbench.runs_dir=${RUNS_DIR}"
  "runner.recorder=nanoeval.json_recorder:json_recorder"
)

if [[ $# -eq 0 ]]; then
  set -- \
    "paperbench.paper_split=debug" \
    "paperbench.solver.time_limit=300" \
    "paperbench.reproduction.skip_reproduction=True" \
    "paperbench.judge.scaffold=dummy" \
    "runner.max_retries=0"
fi

cd "${PAPERBENCH_ROOT}"
exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m paperbench.nano.entrypoint \
  "${extra_args[@]}" \
  "$@"
