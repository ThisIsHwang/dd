from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from progressflip.config import load_config
from progressflip.egl_affinity import unique_device_for_cuda_ordinal
from progressflip.gpu_metrics import summarize_gpu_metrics
from progressflip.gpu_plan import GPUDevice, resolve_worker_plan
from progressflip.query_cache import PolicyQueryCache
from progressflip.workqueue import SQLiteWorkQueue


class FakePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, obs, instruction, *, agent_image=None, wrist_image=None, proprio=None):
        self.calls += 1
        value = float(self.calls)
        return np.full((8, 7), value, dtype=np.float32)


def test_pair_local_query_cache_uses_all_modalities_in_key():
    policy = FakePolicy()
    cache = PolicyQueryCache(enabled=True)
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    first = cache.query(policy, obs, "task", agent_image=image, wrist_image=image)
    second = cache.query(policy, obs, "task", agent_image=image.copy(), wrist_image=image.copy())
    np.testing.assert_array_equal(first, second)
    assert policy.calls == 1
    assert cache.hits == 1 and cache.misses == 1

    changed = image.copy()
    changed[0, 0, 0] = 1
    third = cache.query(policy, obs, "task", agent_image=changed, wrist_image=image)
    assert policy.calls == 2
    assert not np.array_equal(first, third)


def test_sqlite_queue_dynamic_claim_and_resume(tmp_path: Path):
    path = tmp_path / "queue.sqlite3"
    items = [
        {"work_id": "p0", "task_id": 1, "priority": 10, "payload": {"x": 0}},
        {"work_id": "p1", "task_id": 2, "priority": 20, "payload": {"x": 1}},
        {"work_id": "p2", "task_id": 1, "priority": 5, "payload": {"x": 2}},
    ]
    with SQLiteWorkQueue(path, lease_seconds=60) as queue:
        queue.initialize(items, queue_kind="run", source_fingerprint="abc")
        preferred = queue.claim("w0", preferred_task_id=1)
        assert preferred is not None and preferred.work_id == "p0"
        global_claim = queue.claim("w1")
        assert global_claim is not None and global_claim.work_id == "p1"
        queue.complete(preferred.work_id, "w0")
        queue.fail(global_claim.work_id, "w1", "synthetic")
        remaining = queue.claim("w2")
        assert remaining is not None and remaining.work_id == "p2"

    with SQLiteWorkQueue(path, lease_seconds=60) as queue:
        summary = queue.initialize(
            items,
            queue_kind="run",
            source_fingerprint="abc",
            reset_failed=True,
            reclaim_running=True,
        )
        assert summary["counts"] == {
            "pending": 2,
            "running": 0,
            "done": 1,
            "failed": 0,
            "retired": 0,
        }


def test_gpu_plan_auto_uses_minimum_free_memory(monkeypatch):
    devices = [
        GPUDevice(
            index=i,
            name="NVIDIA H100 80GB HBM3",
            memory_total_mb=81559,
            memory_free_mb=79000 - i * 100,
            uuid=f"GPU-{i}",
        )
        for i in range(8)
    ]
    monkeypatch.setattr("progressflip.gpu_plan.query_devices", lambda _: devices)
    cfg = {
        "compute": {
            "workers_per_gpu": "auto",
            "min_workers_per_gpu": 1,
            "max_workers_per_gpu": 3,
            "model_worker_memory_mb": 22000,
            "gpu_reserve_memory_mb": 10000,
        }
    }
    plan = resolve_worker_plan(cfg, range(8))
    assert plan.workers_per_gpu == 3
    assert plan.total_workers == 24


def test_egl_affinity_matches_logical_cuda_ordinal():
    devices = ["egl-a", "egl-b", "egl-c"]
    mapping = {"egl-a": 2, "egl-b": 0, "egl-c": 1}
    selected = unique_device_for_cuda_ordinal(devices, 0, mapping.get)
    assert selected == "egl-b"
    with pytest.raises(RuntimeError, match="exactly one EGL device"):
        unique_device_for_cuda_ordinal(devices, 4, mapping.get)


def test_gpu_metric_summary(tmp_path: Path):
    path = tmp_path / "gpu.csv"
    rows = [
        ["2026-09-02T00:00:00+09:00", 0, 0, 10, 1000, 80000, 100, 1200],
        ["2026-09-02T00:00:05+09:00", 0, 80, 50, 20000, 80000, 400, 1800],
        ["2026-09-02T00:00:00+09:00", 1, 50, 30, 15000, 80000, 300, 1700],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "gpu_index",
                "gpu_util_percent",
                "memory_util_percent",
                "memory_used_mb",
                "memory_total_mb",
                "power_draw_w",
                "sm_clock_mhz",
            ]
        )
        writer.writerows(rows)
    output = tmp_path / "report"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "workers_per_gpu": 1,
                "max_workers_per_gpu": 3,
                "per_worker_memory_mb": 22000,
                "reserve_memory_mb": 10000,
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_gpu_metrics(path, output, plan_path)
    assert len(summary["gpus"]) == 2
    assert summary["gpus"][0]["gpu_util_mean"] == 40.0
    assert (output / "gpu_utilization_report.md").is_file()


def test_compute_defaults_are_validated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENVLA_OFT_CHECKPOINT", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("PROGRESSFLIP_OUTPUT_ROOT", str(tmp_path / "output"))
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model: {checkpoint: '${OPENVLA_OFT_CHECKPOINT}'}
environment: {max_steps: 10}
data: {output_root: '${PROGRESSFLIP_OUTPUT_ROOT}'}
experiment: {conditions: [A]}
tasks:
  - key: task
    task_id: 1
    candidates:
      - {object: object_1, predicate: [in, object_1, region], remaining_instruction: finish}
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert cfg["compute"]["gpu_ids"] == list(range(8))
    assert cfg["compute"]["workers_per_gpu"] == "auto"
    assert cfg["compute"]["max_workers_per_gpu"] == 3
