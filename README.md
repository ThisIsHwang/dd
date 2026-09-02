# ProgressFlip: Visual Self-State Causality in VLA Policies

A standalone research repository for testing why vision-language-action (VLA) policies fail after a task is externally advanced.

The central experiment holds the physical robot and task state fixed, then changes only what the policy sees or which first action is executed. It separates four explanations:

1. **Robot-pixel shortcut:** visible robot pixels, rather than the advanced task state, drive the first decision.
2. **First-action mediation:** one action generated from a stale visual self-state causes the persistent failure.
3. **Task-phase versus self-localization:** a remaining-subgoal instruction can rescue high-level phase confusion, but not a low-level position-estimation error.
4. **Decoder-specific behavior:** the experiment is implemented for both OpenVLA-OFT and the original autoregressive OpenVLA checkpoint.

## Implemented interventions

### Pixel-level causal controls

All conditions begin from the same frozen prefix and the same physically advanced endpoint state.

| Condition family | What changes at the first policy query |
|---|---|
| True recomposition | Remove and paste the same endpoint robot; compositor control |
| Robot-old | Current scene, old robot pixels only |
| Non-robot-old | Current robot, old non-robot pixels |
| EEF-old | Old gripper/end-effector pixels only |
| Arm-old | Old arm pixels excluding the end effector |
| Robot-mask | Remove visible robot pixels |

Robot, arm, and end-effector masks are rendered from MuJoCo element segmentation. Every outcome run records mask coverage, image hashes, image deltas, and the exact generated composite.

### First-action mediation

| Condition | Query image | Actually executed first action |
|---|---|---|
| Query only (`K0`) | stale | none; immediately query the true image |
| Stale query / true action | stale | action generated from the true image |
| True query / stale action | true | action generated from the stale image |
| Stale action + reset | stale | execute one action, restore exact simulator state, then continue |
| Scale curve | stale | scale first 6-DoF motion by 10%, 25%, 50%, 75%, or 100% |

The analysis verifies action identity across crossover conditions and exact state restoration before reporting inferential statistics.

### Instruction rescue

At the first query only, the stale robot image is paired with one of:

- the original long-horizon instruction;
- the remaining subgoal only;
- an explicit statement that the first subgoal is already complete;
- the deliberately wrong previous subgoal.

Subsequent queries return to the original instruction and true observation.

## Experimental safeguards

- prospective, frozen cohort lock;
- policy-free intervention feasibility screening;
- identical initial state and action prefix within every pair;
- pinned model and source revisions;
- one GPU-isolated worker per H100;
- no post-freeze replacement;
- append-only results with restart support;
- exact replay and image-identity checks;
- smoke gate required before a confirmatory full run;
- paired bootstrap confidence intervals and exact discordant-pair tests;
- Holm correction for the preregistered directional family;
- 90% confidence-interval criterion for equivalence controls.

## Target hardware

The launcher targets one node with **8× NVIDIA H100**. The host may expose CUDA 12.9; the reproducibility environment intentionally uses the OpenVLA-OFT-compatible PyTorch 2.2.0 `cu121` runtime. No local FlashAttention build is required.

## Installation

```bash
git clone https://github.com/ThisIsHwang/dd.git
cd dd
bash scripts/bootstrap_conda.sh
conda activate progressflip
```

Validate the node before any scientific run:

```bash
export PROGRESSFLIP_GPU_LIST=0,1,2,3,4,5,6,7
export PROGRESSFLIP_RENDER_BACKEND=egl
progressflip-doctor --project-root "$PWD" --expect-gpus 8 --strict
```

Use `PROGRESSFLIP_RENDER_BACKEND=osmesa` only when EGL is unavailable and OSMesa is installed.

## Run the OpenVLA-OFT experiment

The smoke run uses one prefix per task and all 24 conditions. The full run uses 15 locked prefixes per task.

```bash
bash scripts/prepare_prospective_source.sh
bash scripts/run_visual_self_state_pipeline.sh \
  /local_nvme/$USER/self_state_oft_smoke \
  /local_nvme/$USER/self_state_oft_full
```

Expected outcome jobs:

```text
smoke: 4 tasks × 1 prefix × 24 conditions = 96
full:  4 tasks × 15 prefixes × 24 conditions = 1,440
```

## Run the original OpenVLA cross-decoder replication

```bash
bash scripts/run_visual_self_state_original_openvla_pipeline.sh \
  /local_nvme/$USER/self_state_original_smoke \
  /local_nvme/$USER/self_state_original_full
```

Expected outcome jobs:

```text
smoke: 4 × 1 × 18 = 72
full:  4 × 6 × 18 = 432
```

Run both stages sequentially:

```bash
bash scripts/run_visual_self_state_campaign.sh \
  /local_nvme/$USER/visual_self_state_campaign
```

For Slurm:

```bash
sbatch scripts/slurm_visual_self_state_1node_8gpu.sbatch
```

## Outputs

Each run directory contains:

```text
metadata/                 frozen config, source provenance, runtime and model metadata
prefixes/                 action prefixes and counterfactual checkpoints
screening/                policy-free feasibility results
manifests/                frozen cohort and condition jobs
results/rank_*.jsonl      append-only outcome records
traces/                   per-step actions, states, predicates and distances
videos/                   retained qualitative rollouts
analysis/report.md        original ProgressFlip analysis
analysis/self_state_report.md
analysis/self_state_summary.json
analysis/self_state_condition_summary.csv
analysis/self_state_contrasts.csv
analysis/self_state_scaling_curve.csv
```

The self-state inference gate fails if any requested condition is missing, any result is invalid, the pixel mask fails, recomposition exceeds the configured tolerance, crossover action identity fails, the scale intervention is wrong, the post-action reset is inexact, or an instruction override is not the preregistered one.

## Analysis only

```bash
progressflip-analyze-self-state \
  --config /path/to/run/metadata/config.yaml \
  --run-dir /path/to/run
```

The report also measures the first selected action relative to:

- the true end-effector position and next goal;
- the stale visually implied end-effector position and next goal;
- the true/stale end-effector position and the already completed progress object.

These direction diagnostics help distinguish high-level subgoal regression from low-level visual self-localization error.

## Tests

CPU-only unit and regression tests do not require MuJoCo rendering or model weights:

```bash
python -m pytest -q
python -m compileall -q progressflip tests
```

The final hardware smoke test must still be run on the target H100 node. This repository does not claim that EGL, the model checkpoint, or the 8-GPU end-to-end path was executed in the repository-build environment.

## Research documents

- [Experiment design](docs/EXPERIMENT_DESIGN.md)
- [Preregistration](docs/PREREGISTRATION.md)
- [Results motivating this experiment](docs/RESULTS_SO_FAR.md)
- [Implementation and validity notes](docs/IMPLEMENTATION_NOTES.md)
- [Environment profile](environment/README.md)
- [Third-party software](THIRD_PARTY.md)

## Scope of claims

This code supports a controlled simulation study. A successful run can establish causal sensitivity to visual self-state interventions for the evaluated policies and screened LIBERO population. It does **not** by itself establish the same mechanism for every VLA, every task, or a physical robot.

## License

ProgressFlip source code is released under the MIT License. OpenVLA-OFT, OpenVLA, LIBERO, robosuite, MuJoCo, and model checkpoints remain under their own licenses and terms.
