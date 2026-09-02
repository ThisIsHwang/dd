# ProgressFlip VLA

A standalone research repository for controlled causal experiments on **visual self-state dominance** in vision-language-action policies.

The experiments hold the physical task state fixed and intervene on robot pixels, first actions, or instructions. They test whether a VLA follows the advanced scene state or a visually implied stale robot state.

## Main experiment families

- **Robot-pixel factorization:** true recomposition, robot-old, nonrobot-old, EEF-old, arm-old, masking, random robot pose, and phase interpolation.
- **First-action mediation:** query-only `K0`, true/stale action crossovers, post-action physical reset, and action-scale curves.
- **Instruction rescue:** original instruction, remaining subgoal, explicit progress, and an intentionally wrong previous subgoal.

Every condition within a pair uses the same initial state, action prefix, and physically advanced endpoint state.

## 8×H100 throughput architecture

The default launcher is designed for one node with eight H100 GPUs:

- all eight GPUs participate in **candidate collection** and **rollout execution**;
- a persisted SQLite lease queue dynamically assigns work instead of static rank sharding;
- all conditions for one causal pair remain on one worker and physical GPU;
- auto mode launches one to three independent model/environment workers per H100 based on free memory;
- pair-local caching removes repeated byte-identical first-decision queries;
- TensorFlow preprocessing uses the CPU-only wheel and cannot reserve H100 memory;
- full runs encode videos only for representative pairs while retaining all numeric traces;
- `nvidia-smi` monitoring generates a utilization and next-run slot recommendation.

OpenVLA-OFT's released continuous-action path has batch-size-one assumptions, so this repository uses tested independent replicas rather than an unvalidated inference microbatch patch.

## Install

```bash
git clone https://github.com/ThisIsHwang/dd.git
cd dd
conda create -n progressflip python=3.10 -y
conda activate progressflip
bash scripts/bootstrap.sh
bash scripts/prefetch_checkpoint.sh
```

The host may expose CUDA 12.9. The reproducibility environment uses the OpenVLA-OFT-compatible PyTorch 2.2.0 `cu121` wheel.

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

## Full run

Use a separate node-local output directory:

```bash
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_full
# Set this to the smoke report's recommendation; 3 is the H100-80GB target.
export PF_WORKERS_PER_GPU=3
bash scripts/run_pipeline.sh configs/full.yaml
```

The pipeline performs:

```text
8-GPU prospective candidate screening
  → deterministic first-K cohort freeze
  → pair × condition manifest freeze
  → dynamic multi-slot 8-GPU rollout
  → paired statistical analysis
```

## Resume

The queue and results are append-only. Re-run the same command with the same output root:

```bash
export PF_RESET_FAILED=1
export PF_RECLAIM_RUNNING=1
bash scripts/run_pipeline.sh configs/full.yaml
```

## Outputs

```text
pairs/                         frozen causal pair packs
cohort_lock.json               prospective cohort selection
manifest.jsonl                 frozen pair × condition jobs
queues/*.sqlite3               persistent collection and rollout queues
results/rank-*.jsonl           append-only outcomes
traces/                        per-condition numeric traces
videos/                        representative qualitative rollouts
analysis/report.md             scientific analysis
runtime/*_gpu_plan.json        resolved worker slots
runtime/*_gpu_metrics.csv      raw nvidia-smi samples
runtime/*_utilization/         utilization report and next-run recommendation
```

## Slurm

```bash
mkdir -p logs
export PROGRESSFLIP_CONDA_ENV=progressflip
export OPENVLA_OFT_CHECKPOINT=$PWD/checkpoints/openvla-oft-libero10
export PROGRESSFLIP_PERSIST_ROOT=$HOME/progressflip_results/$SLURM_JOB_ID
sbatch slurm/1node_8gpu.sbatch
```

## Documentation

- [8×H100 getting started](docs/GETTING_STARTED.md)
- [GPU utilization design and tuning](docs/GPU_UTILIZATION.md)

## Tests

```bash
python -m pytest -q
python -m compileall -q src tests
```

CPU tests do not replace the required H100 smoke test. The repository does not claim that EGL, checkpoint loading, or the multi-process H100 path was executed outside the target node.

## Scope

This code supports a controlled LIBERO simulation study. It can establish causal sensitivity for the evaluated policies and frozen screened population. It does not by itself establish the same mechanism for every VLA or for a physical robot.
