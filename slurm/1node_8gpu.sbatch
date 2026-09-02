#!/usr/bin/env bash
#SBATCH --job-name=progressflip-max8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${PROGRESSFLIP_CONDA_ENV:-progressflip}"

export PROGRESSFLIP_OUTPUT_ROOT="${PROGRESSFLIP_OUTPUT_ROOT:-${SLURM_TMPDIR:-/tmp}/progressflip_${SLURM_JOB_ID}}"
export OPENVLA_OFT_CHECKPOINT="${OPENVLA_OFT_CHECKPOINT:-${ROOT}/checkpoints/openvla-oft-libero10}"
export PROGRESSFLIP_GPU_LIST="${PROGRESSFLIP_GPU_LIST:-0,1,2,3,4,5,6,7}"
export PF_WORKERS_PER_GPU="${PF_WORKERS_PER_GPU:-auto}"
export PF_CPU_THREADS="${PF_CPU_THREADS:-2}"
export PF_NUMA_BIND="${PF_NUMA_BIND:-1}"
export PF_RESET_FAILED="${PF_RESET_FAILED:-1}"
export PF_RECLAIM_RUNNING="${PF_RECLAIM_RUNNING:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p logs

persist() {
  if [[ -n "${PROGRESSFLIP_PERSIST_ROOT:-}" && -d "${PROGRESSFLIP_OUTPUT_ROOT}" ]]; then
    mkdir -p "${PROGRESSFLIP_PERSIST_ROOT}"
    rsync -a --partial --exclude checkpoints --exclude third_party \
      "${PROGRESSFLIP_OUTPUT_ROOT}/" "${PROGRESSFLIP_PERSIST_ROOT}/"
  fi
}
trap persist EXIT TERM INT

bash scripts/run_pipeline.sh configs/full.yaml
