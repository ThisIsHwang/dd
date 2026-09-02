from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import atomic_json


@dataclass(frozen=True)
class GPUDevice:
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    uuid: str


@dataclass(frozen=True)
class GPUWorkerPlan:
    gpu_ids: list[int]
    workers_per_gpu: int
    total_workers: int
    per_worker_memory_mb: int
    reserve_memory_mb: int
    max_workers_per_gpu: int
    devices: list[GPUDevice]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["devices"] = [asdict(device) for device in self.devices]
        return output


def parse_gpu_ids(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        ids = [int(item) for item in value]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"GPU IDs must be a non-empty unique list, got {ids}")
    return ids


def query_devices(gpu_ids: Iterable[int]) -> list[GPUDevice]:
    wanted = set(int(value) for value in gpu_ids)
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,uuid",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    devices = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) != 5:
            continue
        index = int(row[0].strip())
        if index not in wanted:
            continue
        devices.append(
            GPUDevice(
                index=index,
                name=row[1].strip(),
                memory_total_mb=int(float(row[2].strip())),
                memory_free_mb=int(float(row[3].strip())),
                uuid=row[4].strip(),
            )
        )
    devices.sort(key=lambda device: device.index)
    missing = wanted.difference(device.index for device in devices)
    if missing:
        raise RuntimeError(f"nvidia-smi did not return requested GPUs: {sorted(missing)}")
    return devices


def resolve_worker_plan(
    cfg: Mapping[str, Any],
    gpu_ids: str | Iterable[int],
    requested: str | int | None = None,
) -> GPUWorkerPlan:
    ids = parse_gpu_ids(gpu_ids)
    compute = dict(cfg.get("compute", {}))
    per_worker = int(compute.get("model_worker_memory_mb", 22000))
    reserve = int(compute.get("gpu_reserve_memory_mb", 10000))
    max_workers = int(compute.get("max_workers_per_gpu", 3))
    minimum = int(compute.get("min_workers_per_gpu", 1))
    requested_value: str | int = (
        compute.get("workers_per_gpu", "auto") if requested is None else requested
    )
    devices = query_devices(ids)
    if str(requested_value).lower() == "auto":
        capacity = min(
            max(0, (device.memory_free_mb - reserve) // max(1, per_worker))
            for device in devices
        )
        workers = max(minimum, min(max_workers, int(capacity)))
        rationale = (
            "auto: floor((minimum free GPU memory - reserve) / estimated model-worker memory), "
            f"clamped to [{minimum}, {max_workers}]"
        )
    else:
        workers = int(requested_value)
        if workers < 1 or workers > max_workers:
            raise ValueError(f"workers_per_gpu={workers} must be in [1, {max_workers}]")
        required = workers * per_worker + reserve
        undersized = [
            device.index for device in devices if device.memory_free_mb < required
        ]
        if undersized:
            raise RuntimeError(
                f"Requested {workers} workers/GPU needs about {required} MiB free; "
                f"insufficient GPUs={undersized}"
            )
        rationale = "explicit workers_per_gpu request"
    return GPUWorkerPlan(
        gpu_ids=ids,
        workers_per_gpu=workers,
        total_workers=len(ids) * workers,
        per_worker_memory_mb=per_worker,
        reserve_memory_mb=reserve,
        max_workers_per_gpu=max_workers,
        devices=devices,
        rationale=rationale,
    )


def write_worker_plan(
    cfg: Mapping[str, Any],
    gpu_ids: str | Iterable[int],
    requested: str | int | None,
    output: str | Path,
) -> dict[str, Any]:
    plan = resolve_worker_plan(cfg, gpu_ids, requested)
    atomic_json(output, plan.to_dict())
    return plan.to_dict()
