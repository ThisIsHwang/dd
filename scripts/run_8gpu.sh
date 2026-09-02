#!/usr/bin/env bash
set -euo pipefail
CONFIG="$(realpath "${1:-configs/full.yaml}")"
WORLD_SIZE="${PF_WORLD_SIZE:-8}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS="${PF_CPU_THREADS:-6}"
export MKL_NUM_THREADS="${PF_CPU_THREADS:-6}"
mkdir -p "${PROGRESSFLIP_OUTPUT_ROOT}/worker_logs"
pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  (
    export CUDA_VISIBLE_DEVICES="${rank}"
    export PF_RANK="${rank}"
    export PF_WORLD_SIZE="${WORLD_SIZE}"
    progressflip run-worker --config "${CONFIG}" --rank "${rank}" --world-size "${WORLD_SIZE}"
  ) >"${PROGRESSFLIP_OUTPUT_ROOT}/worker_logs/rank$(printf '%03d' "${rank}").log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
