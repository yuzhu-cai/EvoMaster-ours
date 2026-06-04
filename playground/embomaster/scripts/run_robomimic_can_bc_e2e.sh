#!/usr/bin/env bash
set -euo pipefail

# Resolve workspace root from this script location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "${ROOT_DIR}"

exec conda run --no-capture-output -n evomaster python run.py \
  --agent embomaster \
  --config "${ROOT_DIR}/configs/embomaster/config_robomimic_can_bc_e2e.yaml" \
  --task "${ROOT_DIR}/playground/embomaster/PRDs/robomimic/robomimic_task_can_bc.md" \
  "$@"
