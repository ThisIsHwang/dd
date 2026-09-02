# Getting started on one 8×H100 node

## Install

```bash
git clone https://github.com/ThisIsHwang/dd.git
cd dd
conda create -n progressflip python=3.10 -y
conda activate progressflip
bash scripts/bootstrap.sh
bash scripts/prefetch_checkpoint.sh
```

Install the system package `numactl` when permitted. The launcher works without it, but cannot bind workers to the GPU-local NUMA node.

## Smoke run

```bash
export OPENVLA_OFT_CHECKPOINT="$PWD/checkpoints/openvla-oft-libero10"
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_smoke
export PROGRESSFLIP_GPU_LIST=0,1,2,3,4,5,6,7
export PF_WORKERS_PER_GPU=auto
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

progressflip preflight --config configs/smoke.yaml --expect-gpus 8
bash scripts/run_pipeline.sh configs/smoke.yaml
```

The smoke run uses all eight GPUs for both collection and causal rollouts. On an empty 80 GB H100 node, auto mode normally proposes up to three model workers per GPU if the memory estimate permits it.

Inspect:

```text
$PROGRESSFLIP_OUTPUT_ROOT/analysis/report.md
$PROGRESSFLIP_OUTPUT_ROOT/runtime/collect_utilization/gpu_utilization_report.md
$PROGRESSFLIP_OUTPUT_ROOT/runtime/run_utilization/gpu_utilization_report.md
```

Also inspect the representative videos for `TRUE_RECOMPOSED_K1`, `ROBOT_OLD_K1`, `NONROBOT_OLD_K1`, the action crossovers, and `STALE_K1_RESET`.

## Full frozen experiment

Use a new output directory:

```bash
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_full
# Override only after reading the smoke utilization report.
export PF_WORKERS_PER_GPU=3
bash scripts/run_pipeline.sh configs/full.yaml
```

Full collection dynamically screens all configured candidate initial states with all eight GPUs, then freezes the prospective first-K cohort. The rollout phase dynamically assigns whole pair bundles so every condition for one pair stays on one physical H100.

## Resume after preemption or failure

Run the same command with the same output root:

```bash
export PF_RESET_FAILED=1
export PF_RECLAIM_RUNNING=1
bash scripts/run_pipeline.sh configs/full.yaml
```

Completed candidate screens, frozen pairs, and completed condition jobs are reused. Failed or abandoned queue items are retried.

## Conservative fallback after CUDA OOM

```bash
export PF_WORKERS_PER_GPU=2
export PF_RESET_FAILED=1
export PF_RECLAIM_RUNNING=1
bash scripts/run_pipeline.sh configs/full.yaml
```

Do not delete the manifest or pair packs. The queue resumes only unfinished pair bundles.

## Slurm

```bash
mkdir -p logs
export PROGRESSFLIP_CONDA_ENV=progressflip
export OPENVLA_OFT_CHECKPOINT=$PWD/checkpoints/openvla-oft-libero10
export PROGRESSFLIP_PERSIST_ROOT=$HOME/progressflip_results/$SLURM_JOB_ID
sbatch slurm/1node_8gpu.sbatch
```

The Slurm wrapper requests eight GPUs, 128 CPU cores, all node memory, and 72 hours. It uses node-local temporary storage for the live queue and copies outputs to `PROGRESSFLIP_PERSIST_ROOT` on exit.

See [GPU utilization design](GPU_UTILIZATION.md) for the queue, cache, memory model, and tuning rules.
