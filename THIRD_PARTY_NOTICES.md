# Third-party notices

## OpenVLA-OFT

ProgressFlip integrates the public OpenVLA-OFT implementation and checkpoints under their respective upstream terms. The upstream repository is pinned by `scripts/bootstrap.sh`.

## LIBERO and robosuite

The simulator integration uses LIBERO and robosuite under their respective upstream licenses.

`src/progressflip/egl_affinity.py` is a runtime-only interoperability adaptation informed by:

- robosuite's Apache-2.0 EGL context implementation; and
- the Apache-2.0 `patch_robosuite_egl.py` implementation in `verl-project/verl-vla`.

The ProgressFlip implementation does not modify installed third-party source files. It queries NVIDIA's `EGL_CUDA_DEVICE_NV` attribute at process startup and replaces robosuite's in-memory EGL display-selection function before any rendering context is created.

The relevant upstream copyright and license notices remain with those projects.
