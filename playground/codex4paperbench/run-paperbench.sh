#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOMASTER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-paperbench}"
FRONTIER_EVALS_ROOT="${FRONTIER_EVALS_ROOT:-/data/yuzhu/Devs/third_party/frontier-evals}"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-${FRONTIER_EVALS_ROOT}/project/paperbench}"
RUNS_DIR="${RUNS_DIR:-${EVOMASTER_ROOT}/runs/codex4paperbench}"
AGENT_IMAGE="${AGENT_IMAGE:-pb-env-codex:0.125.0}"
REPRO_IMAGE="${REPRO_IMAGE:-pb-reproducer:latest}"

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
  echo "Build it with: IMAGE=${AGENT_IMAGE} playground/codex4paperbench/docker/build-with-host-codex.sh" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PAPERBENCH_DATA_DIR="${PAPERBENCH_DATA_DIR:-${PAPERBENCH_ROOT}/data}"

if [[ "${CODEX4PAPERBENCH_SOURCE_HOST_CODEX_ENV:-1}" != "0" && -f "${HOME}/.codex/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${HOME}/.codex/.env"
  set +a
fi

if [[ -z "${GRADER_OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export GRADER_OPENAI_API_KEY="${OPENAI_API_KEY}"
fi

# Local Codex configs may use CRS_KEY instead of OPENAI_API_KEY. PaperBench checks
# GRADER_OPENAI_API_KEY before it even creates tasks, including dummy-judge runs.
if [[ -z "${OPENAI_API_KEY:-}" && -n "${CRS_KEY:-}" ]]; then
  export OPENAI_API_KEY="${CRS_KEY}"
fi

if [[ -z "${GRADER_OPENAI_API_KEY:-}" && -n "${CRS_KEY:-}" ]]; then
  export GRADER_OPENAI_API_KEY="${CRS_KEY}"
fi

extra_args=(
  "paperbench.solver=codex4paperbench.solver:CodexCliSolver"
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
