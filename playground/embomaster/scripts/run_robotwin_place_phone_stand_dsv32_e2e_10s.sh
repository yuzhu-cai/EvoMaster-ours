#!/usr/bin/env bash
set -euo pipefail

# Resolve workspace root from this script location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "${ROOT_DIR}"

exec conda run --no-capture-output -n embomaster python run.py \
  --agent embomaster \
  --config "${ROOT_DIR}/configs/embomaster/config_robotwin_place_phone_stand_dsv32_e2e_10s.yaml" \
  --task "${ROOT_DIR}/playground/embomaster/PRDs/robotwin_place_phone_stand_RDT.md" \
  "$@"
