# Validation scope

CPU checks cover configuration-independent condition registration, segmentation masks, image composition, action crossover selection, pair-pack checksums, and paired analysis.

The following require the target node and are checked by `progressflip preflight` plus the smoke pipeline:

- 8×H100 visibility;
- complete local OpenVLA-OFT checkpoint;
- MuJoCo EGL rendering and segmentation;
- LIBERO private predicate compatibility;
- robot geometry discovery;
- controller-goal resynchronization after direct state restoration;
- end-to-end VLA inference.

A successful CPU test suite is not evidence that simulator composites rendered correctly. Inspect smoke videos before the full cohort.
