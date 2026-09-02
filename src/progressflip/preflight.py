from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from typing import Any

from .config import atomic_json


def _package(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_report(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser()
    required = [
        "config.json",
        "dataset_statistics.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    action_heads = sorted(root.glob("*action_head*checkpoint*.pt"))
    proprio = sorted(root.glob("*proprio_projector*checkpoint*.pt"))
    if len(action_heads) != 1:
        missing.append(f"one action head (found {len(action_heads)})")
    if len(proprio) != 1:
        missing.append(f"one proprio projector (found {len(proprio)})")
    has_weights = (root / "model.safetensors").is_file() or (
        root / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors or model.safetensors.index.json")
    return {
        "path": str(root.resolve()) if root.exists() else str(root),
        "exists": root.is_dir(),
        "complete": root.is_dir() and not missing,
        "missing": missing,
        "action_heads": [item.name for item in action_heads],
        "proprio_projectors": [item.name for item in proprio],
    }


def run_preflight(cfg: dict[str, Any], expect_gpus: int = 8) -> dict[str, Any]:
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "memory_bytes": int(properties.total_memory),
                "capability": [int(properties.major), int(properties.minor)],
            }
        )
    checkpoint = checkpoint_report(cfg["model"]["checkpoint"])
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": len(devices) == int(expect_gpus),
        "all_h100": bool(devices) and all("H100" in device["name"] for device in devices),
        "checkpoint_complete": checkpoint["complete"],
        "mujoco_gl": os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"},
    }
    result = {
        "ok": all(checks.values()),
        "checks": checks,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "devices": devices,
        "checkpoint": checkpoint,
        "packages": {
            name: _package(name)
            for name in ("robosuite", "libero", "mujoco", "transformers", "peft")
        },
    }
    atomic_json(Path(cfg["data"]["output_root"]) / "preflight.json", result)
    return result
