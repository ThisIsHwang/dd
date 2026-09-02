#!/usr/bin/env bash
set -euo pipefail
CONFIG="$(realpath "${1:-configs/full.yaml}")"
TASKS=(microwave_close drawer_close mugs_plates soup_cheese_basket)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p "${PROGRESSFLIP_OUTPUT_ROOT}"
pids=()
for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  (
    export CUDA_VISIBLE_DEVICES="${index}"
    progressflip collect --config "${CONFIG}" --task "${task}"
  ) >"${PROGRESSFLIP_OUTPUT_ROOT}/collect_${task}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
