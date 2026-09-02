#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(realpath "${1:-${ROOT}/configs/full.yaml}")"
export PROGRESSFLIP_OUTPUT_ROOT="${PROGRESSFLIP_OUTPUT_ROOT:-${ROOT}/outputs/run_$(date +%Y%m%d_%H%M%S)}"
export OPENVLA_OFT_CHECKPOINT="${OPENVLA_OFT_CHECKPOINT:-${ROOT}/checkpoints/openvla-oft-libero10}"
mkdir -p "${PROGRESSFLIP_OUTPUT_ROOT}"
if [[ ! -f "${OPENVLA_OFT_CHECKPOINT}/dataset_statistics.json" ]]; then
  bash "${ROOT}/scripts/prefetch_checkpoint.sh" "${OPENVLA_OFT_CHECKPOINT}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
progressflip preflight --config "${CONFIG}" --expect-gpus 8
bash "${ROOT}/scripts/run_collect.sh" "${CONFIG}"
progressflip manifest --config "${CONFIG}"
bash "${ROOT}/scripts/run_8gpu.sh" "${CONFIG}"
progressflip analyze --config "${CONFIG}"
echo "Report: ${PROGRESSFLIP_OUTPUT_ROOT}/analysis/report.md"
echo "Collection GPU utilization: ${PROGRESSFLIP_OUTPUT_ROOT}/runtime/collect_utilization/gpu_utilization_report.md"
echo "Rollout GPU utilization: ${PROGRESSFLIP_OUTPUT_ROOT}/runtime/run_utilization/gpu_utilization_report.md"
