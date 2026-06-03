#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "${ROOT_DIR}"

exec conda run --no-capture-output -n evomaster python run.py \
  --agent embomaster \
  --config "${ROOT_DIR}/configs/embomaster/config_maniskill_pull_cube_ppofast_e2e.yaml" \
  --task "${ROOT_DIR}/playground/embomaster/PRDs/maniskill/instruction_pull_cube.md" \
  "$@"
