#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu-list> <output.csv> [interval-seconds]" >&2
  exit 2
fi

GPU_LIST="$1"
OUTPUT="$2"
INTERVAL="${3:-5}"
mkdir -p "$(dirname "$OUTPUT")"
echo 'timestamp,gpu_index,gpu_util_percent,memory_util_percent,memory_used_mb,memory_total_mb,power_draw_w,sm_clock_mhz' >"$OUTPUT"

while true; do
  timestamp="$(date --iso-8601=seconds)"
  nvidia-smi -i "$GPU_LIST" \
    --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm \
    --format=csv,noheader,nounits 2>/dev/null |
  while IFS= read -r line; do
    [[ -n "$line" ]] && printf '%s,%s\n' "$timestamp" "$line" >>"$OUTPUT"
  done
  sleep "$INTERVAL"
done
