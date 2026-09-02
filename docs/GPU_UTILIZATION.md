# Saturating one 8×H100 node

The upstream OpenVLA-OFT action path is effectively batch-size-one for this checkpoint: the multimodal generation code contains batch-size-one assumptions, and the continuous action result is reshaped to one `[chunk, action_dim]` tensor. ProgressFlip therefore does **not** add an unvalidated microbatch patch. It uses independent model replicas and overlaps GPU inference with MuJoCo stepping and rendering.

## Runtime architecture

```text
8 physical H100s
  └── auto 1–3 worker processes per H100
        ├── one local OpenVLA-OFT replica
        ├── one MuJoCo environment at a time
        ├── pair-local deterministic query cache
        └── dynamic SQLite lease queue on node-local NVMe
```

The default H100-80GB plan estimates 22 GB per process, keeps 10 GB reserve, and therefore selects three workers per GPU when the cards are otherwise empty. That creates 24 independent workers. Set an explicit value when the measured memory report recommends one:

```bash
export PF_WORKERS_PER_GPU=2   # conservative
export PF_WORKERS_PER_GPU=3   # default auto target on an empty H100-80GB node
```

All conditions for one causal pair remain on one worker and physical GPU. Idle workers claim the next unclaimed pair, so task-length variance does not create the static-sharding tail seen in the old launcher.

## Why several workers per GPU?

One rollout repeatedly alternates between:

1. a short H100 forward pass;
2. CPU MuJoCo stepping;
3. off-screen rendering and image preparation;
4. another H100 forward pass.

A single process leaves the H100 idle during steps 2 and 3. Two or three independent processes interleave these phases. No gradients or inter-GPU communication are used.

## Collection also uses all eight GPUs

The old collector assigned four tasks to four GPUs. The dynamic collector creates one work item for every task × candidate initial state and allows every H100 worker to screen candidates. After all screens finish, the cohort is frozen deterministically as the lowest initial-state IDs among accepted candidates. Parallel completion order therefore cannot alter the selected cohort.

## Query reuse

Within a pair, many conditions request byte-identical true or stale first-decision inputs. `PolicyQueryCache` hashes:

- instruction;
- agent image;
- wrist image;
- proprioceptive override;
- endpoint robot state.

Only identical inputs reuse an action chunk. Cache entries never cross pair boundaries.

## EGL placement across eight GPUs

robosuite 1.4.x can treat `CUDA_VISIBLE_DEVICES` as an index into EGL's separately ordered device list. That assumption is not portable across bare-metal, container, and MIG layouts. Each dynamic worker installs an in-memory patch **before the first render context is created**. It queries NVIDIA's `EGL_CUDA_DEVICE_NV` attribute and selects the EGL device matching the worker's process-local CUDA ordinal 0.

The launcher deliberately unsets `MUJOCO_EGL_DEVICE_ID`; a host physical GPU number and an EGL list index are not guaranteed to share a namespace. The patch modifies no files in site-packages and fails loudly if the NVIDIA query extension is unavailable. In that case, use the CPU rendering fallback:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

After the smoke run, confirm that peak memory and utilization are distributed across all eight physical GPUs in `runtime/*_utilization/gpu_utilization_report.md`.

## Video overhead

Full runs save videos only for the first two frozen pairs per task by default. All pairs still save numeric traces and results. This avoids blocking inference workers on repeated FFmpeg encoding while preserving representative qualitative evidence. Change `data.video_pairs_per_task` to increase or disable this limit.

## Node-local storage is required

Use local NVMe for the SQLite queues, pair packs, traces, and worker logs:

```bash
export PROGRESSFLIP_OUTPUT_ROOT=/local_nvme/$USER/progressflip_full
```

SQLite WAL mode is efficient on a local filesystem. Do not place the live queue on high-latency NFS. Copy results to durable storage after the job.

## Monitoring and tuning

Every phase writes:

```text
runtime/<phase>_gpu_plan.json
runtime/<phase>_gpu_metrics.csv
runtime/<phase>_utilization/gpu_utilization_report.md
runtime/<phase>_utilization/gpu_utilization_summary.json
```

The report contains utilization, idle fraction, peak memory, and a next-run slot recommendation.

Recommended tuning order:

1. Run `configs/smoke.yaml` with one worker per GPU to validate checkpoint loading and EGL placement safely.
2. Confirm that all eight workers loaded successfully and no EGL or CUDA OOM errors occurred.
3. Read the generated peak-memory and utilization report.
4. Use `PF_WORKERS_PER_GPU=2`, `3`, or full-config `auto` for the full frozen experiment.
5. Keep the setting whose report shows the best throughput without OOM or severe context contention.

Do not compare scientific outcomes across different manifests when tuning throughput. Launcher changes are operational; the causal unit and condition definitions must remain fixed.

## CPU and NUMA settings

The launcher limits PyTorch, OpenMP, MKL, OpenBLAS, NumExpr, and TensorFlow threads per process. When `numactl` is installed, each process is bound to the NUMA node closest to its physical GPU.

```bash
export PF_CPU_THREADS=2
export PF_NUMA_BIND=1
```

With 24 workers, two CPU threads per worker is a safe starting point on a 96–128 core node. Increase only after checking CPU utilization and render latency.

## TensorFlow must stay off the GPUs

OpenVLA-OFT uses TensorFlow only in image preprocessing. `scripts/bootstrap.sh` installs `tensorflow-cpu`, and preflight fails if TensorFlow sees a GPU. This prevents every process from reserving H100 memory outside PyTorch.

## Resume semantics

The SQLite queue uses leases. On a clean restart, the launcher:

- marks already completed pair bundles as done;
- resets failed bundles when `PF_RESET_FAILED=1`;
- reclaims stale running leases when `PF_RECLAIM_RUNNING=1`;
- retries only missing work.

Result files remain append-only. Analysis takes the latest record for each `job_id`.

## Optional CUDA MPS

CUDA MPS can reduce context-switch overhead for multiple processes, but cluster policies and device remapping differ. It is deliberately not enabled automatically. Establish a correct non-MPS smoke run first; enable MPS only with the cluster administrator's recommended setup and compare the utilization report on the same frozen manifest.
