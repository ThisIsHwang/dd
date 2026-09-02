#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <collect|run> <config.yaml>" >&2
  exit 2
fi

MODE="$1"
CONFIG="$(realpath "$2")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROGRESSFLIP_OUTPUT_ROOT:?Set PROGRESSFLIP_OUTPUT_ROOT to node-local storage}"
GPU_LIST="${PROGRESSFLIP_GPU_LIST:-0,1,2,3,4,5,6,7}"
REQUESTED_SLOTS="${PF_WORKERS_PER_GPU:-auto}"
RESET_FAILED="${PF_RESET_FAILED:-1}"
config_value() {
  python - "$CONFIG" "$1" <<'PYCFG'
import sys
from progressflip.config import load_config
cfg = load_config(sys.argv[1])
value = cfg
for part in sys.argv[2].split('.'):
    value = value[part]
print(value)
PYCFG
}
STARTUP_STAGGER="${PF_STARTUP_STAGGER_SECONDS:-$(config_value compute.startup_stagger_seconds)}"
MONITOR_INTERVAL="${PF_GPU_MONITOR_INTERVAL_SECONDS:-$(config_value compute.gpu_monitor_interval_seconds)}"
CPU_THREADS="${PF_CPU_THREADS:-$(config_value compute.cpu_threads_per_worker)}"

case "$MODE" in
  collect)
    INIT_COMMAND="collection-queue-init"
    WORKER_COMMAND="collect-dynamic-worker"
    QUEUE_KIND="collect"
    ;;
  run)
    INIT_COMMAND="run-queue-init"
    WORKER_COMMAND="run-dynamic-worker"
    QUEUE_KIND="run"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT/runtime" "$OUTPUT_ROOT/worker_logs/$MODE"
PLAN_PATH="$OUTPUT_ROOT/runtime/${MODE}_gpu_plan.json"
progressflip gpu-plan \
  --config "$CONFIG" \
  --gpu-ids "$GPU_LIST" \
  --workers-per-gpu "$REQUESTED_SLOTS" \
  --output "$PLAN_PATH"

SLOTS="$(python - "$PLAN_PATH" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["workers_per_gpu"])
PY
)"
TOTAL_WORKERS="$(python - "$PLAN_PATH" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["total_workers"])
PY
)"
IFS=',' read -r -a GPU_IDS <<<"$GPU_LIST"
for index in "${!GPU_IDS[@]}"; do
  GPU_IDS[$index]="${GPU_IDS[$index]//[[:space:]]/}"
done

init_args=(progressflip "$INIT_COMMAND" --config "$CONFIG")
if [[ "$RESET_FAILED" == "1" ]]; then
  init_args+=(--reset-failed)
fi
if [[ "${PF_RECLAIM_RUNNING:-1}" == "1" ]]; then
  init_args+=(--reclaim-running)
fi
"${init_args[@]}"

METRICS="$OUTPUT_ROOT/runtime/${MODE}_gpu_metrics.csv"
bash "$ROOT/scripts/monitor_gpus.sh" "$GPU_LIST" "$METRICS" "$MONITOR_INTERVAL" &
MONITOR_PID="$!"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS="${PF_TF_INTRA_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${PF_TF_INTER_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PF_CPU_THREADS="$CPU_THREADS"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

pids=()
labels=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

worker_index=0
# Start one replica on every GPU before adding a second / third replica. This
# avoids three checkpoint loads contending on GPU 0 while the other cards sit idle.
for ((slot=0; slot<SLOTS; slot++)); do
  for gpu in "${GPU_IDS[@]}"; do
    if [[ "$MODE" == "run" ]]; then
      worker_prefix="rank"
    else
      worker_prefix="collect"
    fi
    worker_id="${worker_prefix}-g$(printf '%02d' "$gpu")-s$(printf '%02d' "$slot")"
    log="$OUTPUT_ROOT/worker_logs/$MODE/${worker_id}.log"
    labels+=("$worker_id gpu=$gpu slot=$slot log=$log")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PF_PHYSICAL_GPU="$gpu"
      export PF_GPU_SLOT="$slot"
      export PF_WORKER_ID="$worker_id"
      export MUJOCO_EGL_DEVICE_ID="${PF_EGL_DEVICE_ID:-$gpu}"
      export HF_MODULES_CACHE="$OUTPUT_ROOT/cache/hf_modules/$worker_id"
      export CUDA_CACHE_PATH="$OUTPUT_ROOT/cache/cuda/gpu$(printf '%02d' "$gpu")"
      mkdir -p "$HF_MODULES_CACHE" "$CUDA_CACHE_PATH"
      ulimit -c 0 || true
      sleep $((worker_index * STARTUP_STAGGER))

      command=(progressflip "$WORKER_COMMAND" --config "$CONFIG" --worker-id "$worker_id")
      if [[ "${PF_NUMA_BIND:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
        bus="$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]')"
        sys_bus="${bus/00000000:/0000:}"
        numa_file="/sys/bus/pci/devices/$sys_bus/numa_node"
        if [[ -r "$numa_file" ]]; then
          numa="$(cat "$numa_file")"
          if [[ "$numa" =~ ^[0-9]+$ ]]; then
            exec numactl --cpunodebind="$numa" --membind="$numa" "${command[@]}"
          fi
        fi
      fi
      exec "${command[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
    last_label_index=$((${#labels[@]} - 1))
    echo "launched ${labels[$last_label_index]}"
    worker_index=$((worker_index + 1))
  done
done

echo "Launched $TOTAL_WORKERS workers across ${#GPU_IDS[@]} GPUs ($SLOTS workers/GPU)."
status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "worker failed: ${labels[$index]}" >&2
    tail -n 100 "${labels[$index]##*log=}" >&2 || true
    status=1
  fi
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - INT TERM EXIT

STATUS_PATH="$OUTPUT_ROOT/runtime/${MODE}_queue_status.json"
progressflip queue-status --config "$CONFIG" --kind "$QUEUE_KIND" >"$STATUS_PATH" || status=1
progressflip gpu-util-report \
  --config "$CONFIG" \
  --csv "$METRICS" \
  --plan "$PLAN_PATH" \
  --output-dir "$OUTPUT_ROOT/runtime/${MODE}_utilization" || true

if [[ "$MODE" == "collect" && "$status" == "0" ]]; then
  progressflip freeze-pairs --config "$CONFIG"
fi

if [[ "$status" != "0" ]]; then
  echo "Phase $MODE failed. Inspect $OUTPUT_ROOT/worker_logs/$MODE and $STATUS_PATH" >&2
  exit "$status"
fi

echo "Phase $MODE complete. GPU report: $OUTPUT_ROOT/runtime/${MODE}_utilization/gpu_utilization_report.md"
