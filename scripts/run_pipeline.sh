#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(realpath "${1:-${ROOT}/configs/full.yaml}")"
export PROGRESSFLIP_OUTPUT_ROOT="${PROGRESSFLIP_OUTPUT_ROOT:-${ROOT}/outputs/run_$(date +%Y%m%d_%H%M%S)}"
export OPENVLA_OFT_CHECKPOINT="${OPENVLA_OFT_CHECKPOINT:-${ROOT}/checkpoints/openvla-oft-libero10}"
mkdir -p "${PROGRESSFLIP_OUTPUT_ROOT}"
if [[ ! -f "${OPENVLA_OFT_CHECKPOINT}/dataset_statistics.json" ]]; then
  "${ROOT}/scripts/prefetch_checkpoint.sh" "${OPENVLA_OFT_CHECKPOINT}"
fi
progressflip preflight --config "${CONFIG}" --expect-gpus 8
"${ROOT}/scripts/run_collect.sh" "${CONFIG}"
progressflip manifest --config "${CONFIG}"
"${ROOT}/scripts/run_8gpu.sh" "${CONFIG}"
progressflip analyze --config "${CONFIG}"
echo "Report: ${PROGRESSFLIP_OUTPUT_ROOT}/analysis/report.md"
