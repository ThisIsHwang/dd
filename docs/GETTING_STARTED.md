# Getting started on the 8×H100 node

```bash
git clone https://github.com/ThisIsHwang/dd.git
cd dd
conda create -n progressflip python=3.10 -y
conda activate progressflip
bash scripts/bootstrap.sh
bash scripts/prefetch_checkpoint.sh
```

Run the one-pair-per-task smoke cohort first:

```bash
export OPENVLA_OFT_CHECKPOINT="$PWD/checkpoints/openvla-oft-libero10"
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_smoke
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
progressflip preflight --config configs/smoke.yaml --expect-gpus 8
bash scripts/run_pipeline.sh configs/smoke.yaml
```

Inspect the generated videos, especially `TRUE_RECOMPOSED_K1`, `ROBOT_OLD_K1`, `NONROBOT_OLD_K1`, the action crossover conditions, and `STALE_K1_RESET`.

Run the full frozen cohort in a different directory:

```bash
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_full
bash scripts/run_pipeline.sh configs/full.yaml
```

The final report is written to:

```text
$PROGRESSFLIP_OUTPUT_ROOT/analysis/report.md
```

For Slurm, use `slurm/1node_8gpu.sbatch` and set `PROGRESSFLIP_PERSIST_ROOT` to durable storage before submission.
