#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPENVLA_COMMIT="e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TRANSFORMERS_COMMIT="bc339d9ad707454c0c115970db43c260067c61ab"
DLIMP_COMMIT="040105d256bd28866cc6620621a3d5f7b6b91b46"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required; found {sys.version}")
PY

mkdir -p "${THIRD_PARTY}"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip uninstall -y torch torchvision torchaudio tensorflow >/dev/null 2>&1 || true
"${PYTHON_BIN}" -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# TensorFlow is used only for image preprocessing. The CPU wheel prevents every
# rollout process from reserving memory on the H100 that should be used by PyTorch.
"${PYTHON_BIN}" -m pip install \
  'numpy==1.26.4' 'protobuf<5' 'tensorflow-cpu==2.15.0' \
  'tensorflow-datasets==4.9.3' \
  'accelerate>=0.25,<1' 'peft==0.11.1' 'diffusers==0.30.3' \
  'draccus==0.8.0' 'timm==0.9.10' 'sentencepiece==0.1.99' \
  'tokenizers==0.19.1' safetensors einops rich jsonlines json-numpy wandb \
  'mujoco==2.3.7' 'robosuite==1.4.1' bddl easydict cloudpickle 'gym==0.25.2' \
  scipy pandas matplotlib 'imageio[ffmpeg]' Pillow PyYAML pytest
# Keep tensorflow-cpu as the provider; tensorflow-graphics itself is pure Python.
"${PYTHON_BIN}" -m pip install 'tensorflow-graphics==2021.12.3' --no-deps

clone_pinned() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then git clone "${url}" "${destination}"; fi
  git -C "${destination}" fetch --all --tags
  git -C "${destination}" checkout --detach "${commit}"
}
clone_pinned https://github.com/moojink/openvla-oft.git "${THIRD_PARTY}/openvla-oft" "${OPENVLA_COMMIT}"
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git "${THIRD_PARTY}/LIBERO" "${LIBERO_COMMIT}"
clone_pinned https://github.com/moojink/transformers-openvla-oft.git "${THIRD_PARTY}/transformers-openvla-oft" "${TRANSFORMERS_COMMIT}"
clone_pinned https://github.com/moojink/dlimp_openvla.git "${THIRD_PARTY}/dlimp_openvla" "${DLIMP_COMMIT}"

"${PYTHON_BIN}" -m pip uninstall -y transformers >/dev/null 2>&1 || true
"${PYTHON_BIN}" -m pip install -e "${THIRD_PARTY}/transformers-openvla-oft" --no-deps
"${PYTHON_BIN}" -m pip install -e "${THIRD_PARTY}/dlimp_openvla" --no-deps
"${PYTHON_BIN}" -m pip install -e "${THIRD_PARTY}/openvla-oft" --no-deps
"${PYTHON_BIN}" -m pip install -e "${THIRD_PARTY}/LIBERO" --no-deps
"${PYTHON_BIN}" -m pip install -e "${ROOT}[dev]"

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export LIBERO_CONFIG_PATH
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
benchmark_root: ${THIRD_PARTY}/LIBERO/libero/libero
bddl_files: ${THIRD_PARTY}/LIBERO/libero/libero/bddl_files
init_states: ${THIRD_PARTY}/LIBERO/libero/libero/init_files
datasets: ${THIRD_PARTY}/LIBERO/libero/datasets
assets: ${THIRD_PARTY}/LIBERO/libero/libero/assets
YAML

"${PYTHON_BIN}" - <<'PY'
import torch
import tensorflow as tf
print("torch", torch.__version__, "runtime", torch.version.cuda)
print("tensorflow-visible-gpus", tf.config.list_physical_devices("GPU"))
if tf.config.list_physical_devices("GPU"):
    raise SystemExit("tensorflow-cpu was not installed cleanly; TensorFlow sees a GPU")
PY

echo "Bootstrap complete. Host CUDA 12.9 is compatible with the pinned PyTorch cu121 wheel."
echo "Install the system package 'numactl' to enable optional GPU-local CPU/NUMA binding."
