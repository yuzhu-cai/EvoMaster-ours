#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOMASTER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-paperbench}"
FRONTIER_EVALS_ROOT="${FRONTIER_EVALS_ROOT:-/data/yuzhu/Devs/third_party/frontier-evals}"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-${FRONTIER_EVALS_ROOT}/project/paperbench}"
RUNS_DIR="${RUNS_DIR:-${EVOMASTER_ROOT}/runs/openhands4paperbench}"
AGENT_IMAGE="${AGENT_IMAGE:-pb-env-openhands:1.16.0}"
REPRO_IMAGE="${REPRO_IMAGE:-pb-reproducer:latest}"
OPENHANDS_HOME_SRC="${OPENHANDS_HOME_SRC:-${HOME}/.openhands}"
ENV_FILE="${ENV_FILE:-${EVOMASTER_ROOT}/.env}"

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
  echo "Build it with: IMAGE=${AGENT_IMAGE} playground/openhands4paperbench/docker/build-with-host-openhands.sh" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PAPERBENCH_DATA_DIR="${PAPERBENCH_DATA_DIR:-${PAPERBENCH_ROOT}/data}"

if [[ "${OPENHANDS4PAPERBENCH_SOURCE_ENV_FILE:-1}" != "0" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

if [[ "${OPENHANDS4PAPERBENCH_SOURCE_HOST_OPENHANDS_ENV:-1}" != "0" && -f "${OPENHANDS_HOME_SRC}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${OPENHANDS_HOME_SRC}/.env"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLM_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${LLM_API_KEY}"
fi

if [[ -z "${LLM_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export LLM_API_KEY="${OPENAI_API_KEY}"
fi

if [[ -z "${LLM_BASE_URL:-}" && -n "${GPT_BASE_URL:-}" ]]; then
  export LLM_BASE_URL="${GPT_BASE_URL}"
fi

if [[ -z "${OPENAI_BASE_URL:-}" && -n "${LLM_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${LLM_BASE_URL}"
fi

if [[ -z "${LLM_MODEL:-}" && -n "${GPT_CHAT_MODEL:-}" ]]; then
  export LLM_MODEL="openai/${GPT_CHAT_MODEL}"
fi

if [[ -z "${LLM_REASONING_EFFORT:-}" ]]; then
  export LLM_REASONING_EFFORT="medium"
fi

if [[ -z "${LLM_NATIVE_TOOL_CALLING:-}" ]]; then
  export LLM_NATIVE_TOOL_CALLING="true"
fi

if [[ -z "${OPENHANDS_FORCE_CHAT_COMPLETION:-}" ]]; then
  export OPENHANDS_FORCE_CHAT_COMPLETION="true"
fi

if [[ -z "${OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM:-}" ]]; then
  export OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM="false"
fi

if [[ -z "${OPENHANDS_CONTINUATION_PASSES:-}" ]]; then
  export OPENHANDS_CONTINUATION_PASSES="1"
fi

if [[ -z "${GRADER_OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export GRADER_OPENAI_API_KEY="${OPENAI_API_KEY}"
fi

extra_args=(
  "paperbench.solver=openhands4paperbench.solver:OpenHandsCliSolver"
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

if [[ -n "${LLM_MODEL:-}" ]]; then
  extra_args+=("paperbench.solver.model=${LLM_MODEL}")
fi

if [[ -n "${LLM_BASE_URL:-}" ]]; then
  extra_args+=("paperbench.solver.base_url=${LLM_BASE_URL}")
fi

if [[ -n "${LLM_REASONING_EFFORT:-}" ]]; then
  extra_args+=("paperbench.solver.reasoning_effort=${LLM_REASONING_EFFORT}")
fi

if [[ -n "${LLM_NATIVE_TOOL_CALLING:-}" ]]; then
  extra_args+=("paperbench.solver.native_tool_calling=${LLM_NATIVE_TOOL_CALLING}")
fi

if [[ -n "${OPENHANDS_FORCE_CHAT_COMPLETION:-}" ]]; then
  extra_args+=("paperbench.solver.force_chat_completion=${OPENHANDS_FORCE_CHAT_COMPLETION}")
fi

if [[ -n "${OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM:-}" ]]; then
  extra_args+=("paperbench.solver.use_openai_sdk_responses_stream=${OPENHANDS_USE_OPENAI_SDK_RESPONSES_STREAM}")
fi

if [[ -n "${OPENHANDS_AGENT_RETRIES:-}" ]]; then
  extra_args+=("paperbench.solver.agent_retries=${OPENHANDS_AGENT_RETRIES}")
fi

if [[ -n "${OPENHANDS_CONTINUATION_PASSES:-}" ]]; then
  extra_args+=("paperbench.solver.continuation_passes=${OPENHANDS_CONTINUATION_PASSES}")
fi

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
