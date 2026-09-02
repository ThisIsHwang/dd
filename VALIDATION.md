# Validation scope

## GitHub Actions

The optimized 8×H100 branch is covered by the repository's CPU CI workflow:

```text
pytest
python -m compileall -q src tests
ruff check src tests
```

The most recent completed workflow before the final documentation-only adjustment passed all three stages with **11 tests passed**. New pushes rerun the same workflow automatically.

The tests cover configuration-independent condition registration, segmentation masks, image composition, action crossover selection, pair-pack checksums, atomic pair writes, persisted queue claim/resume behavior, deterministic query caching, H100 worker-slot planning, GPU telemetry aggregation, and logical CUDA-to-EGL device selection.

## Target-node validation still required

The following require the target node and are checked by `progressflip preflight` plus the smoke pipeline:

- visibility of all 8 H100 GPUs;
- complete local OpenVLA-OFT checkpoint;
- safe loading of the configured number of model replicas per H100;
- MuJoCo EGL rendering and segmentation on every physical GPU;
- LIBERO private predicate compatibility;
- robot geometry discovery;
- controller-goal resynchronization after direct state restoration;
- end-to-end VLA inference;
- measured GPU utilization and peak-memory headroom.

A successful CPU test suite is not evidence that simulator composites rendered correctly or that three model replicas fit the target node. Start with `configs/smoke.yaml`, which uses one worker per GPU, inspect representative videos and utilization reports, and only then use `auto` or three workers per GPU for the full run.
