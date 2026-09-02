#!/usr/bin/env bash
#SBATCH --job-name=progressflip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${PROGRESSFLIP_CONDA_ENV:-progressflip}"
export PROGRESSFLIP_OUTPUT_ROOT="${PROGRESSFLIP_OUTPUT_ROOT:-${SLURM_TMPDIR:-/tmp}/progressflip_${SLURM_JOB_ID}}"
export OPENVLA_OFT_CHECKPOINT="${OPENVLA_OFT_CHECKPOINT:-${ROOT}/checkpoints/openvla-oft-libero10}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p logs
bash scripts/run_pipeline.sh configs/full.yaml
if [[ -n "${PROGRESSFLIP_PERSIST_ROOT:-}" ]]; then
  mkdir -p "${PROGRESSFLIP_PERSIST_ROOT}"
  rsync -a --exclude checkpoints --exclude third_party "${PROGRESSFLIP_OUTPUT_ROOT}/" "${PROGRESSFLIP_PERSIST_ROOT}/"
fi
