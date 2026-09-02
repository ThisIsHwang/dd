from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .collector import _candidate_trigger, _numpy, _process_first
from .config import atomic_json, fingerprint
from .libero import (
    eval_predicate,
    flat_state,
    make_env,
    make_object_advanced_state,
    make_suite,
    object_qvel,
    reset_and_wait,
    set_flat_state,
)
from .policy import OpenVLAOFTPolicy
from .records import validate_pair, write_pair
from .workqueue import SQLiteWorkQueue, default_worker_id, queue_summary


LOGGER = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _collect_candidate(
    cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    initial_state_id: int,
    *,
    env: Any,
    instruction: str,
    initial_states: Any,
    policy: OpenVLAOFTPolicy,
    destination_root: Path,
) -> dict[str, Any]:
    task_key = str(task_cfg["key"])
    task_id = int(task_cfg["task_id"])
    pair_id = f"{task_key}--init{initial_state_id:03d}"
    pair_dir = destination_root / pair_id
    result: dict[str, Any] = {
        "pair_id": pair_id,
        "task_key": task_key,
        "task_id": task_id,
        "initial_state_id": int(initial_state_id),
        "accepted": False,
    }
    if pair_dir.is_dir():
        validate_pair(pair_dir)
        result.update({"accepted": True, "reason": "existing_valid_candidate"})
        return result
    if initial_state_id >= len(initial_states):
        result["reason"] = "initial_state_out_of_range"
        return result

    wait_steps = int(cfg["environment"].get("wait_steps", 10))
    max_prefix_steps = int(task_cfg.get("trigger", {}).get("max_prefix_steps", 300))
    max_steps = int(cfg["environment"]["max_steps"])
    stable_steps = int(cfg["experiment"].get("predicate_stable_steps", 4))
    stable_velocity = float(cfg["experiment"].get("stable_object_velocity", 0.15))

    initial_state = _numpy(initial_states[initial_state_id])
    obs = reset_and_wait(env, initial_state, wait_steps, policy.dummy_action)
    prefix_actions: list[np.ndarray] = []
    candidate = None
    for _ in range(max_prefix_steps):
        candidate = _candidate_trigger(env, obs, task_cfg)
        if candidate is not None:
            break
        action = _process_first(policy, obs, instruction)
        obs, _, done, _ = env.step(action.tolist())
        prefix_actions.append(np.asarray(action, dtype=np.float32))
        if done or env.check_success():
            candidate = None
            break
    if candidate is None:
        result["reason"] = "no_eligible_trigger"
        return result

    trigger_state = flat_state(env)
    trajectory_states: list[np.ndarray] = [trigger_state.copy()]
    stable_count = 0
    stable_state = None
    stable_index = None
    elapsed = wait_steps + len(prefix_actions)
    while elapsed < max_steps:
        action = _process_first(policy, obs, instruction)
        obs, _, done, _ = env.step(action.tolist())
        elapsed += 1
        trajectory_states.append(flat_state(env))
        predicate_true = eval_predicate(env, candidate["predicate"])
        velocity = float(np.linalg.norm(object_qvel(env, candidate["object"])))
        stable_count = stable_count + 1 if predicate_true and velocity <= stable_velocity else 0
        if stable_count >= stable_steps and stable_state is None:
            stable_state = flat_state(env)
            stable_index = len(trajectory_states) - 1
        if env.check_success() or done:
            break

    success = bool(env.check_success())
    if success and stable_state is None:
        for _ in range(stable_steps):
            obs, _, _, _ = env.step(policy.dummy_action.tolist())
            trajectory_states.append(flat_state(env))
            predicate_true = eval_predicate(env, candidate["predicate"])
            velocity = float(np.linalg.norm(object_qvel(env, candidate["object"])))
            stable_count = stable_count + 1 if predicate_true and velocity <= stable_velocity else 0
            if stable_count >= stable_steps:
                stable_state = flat_state(env)
                stable_index = len(trajectory_states) - 1
                break
    if not success or stable_state is None or stable_index is None:
        result["reason"] = "nominal_failure_or_no_stable_checkpoint"
        return result

    object_advanced_state = make_object_advanced_state(
        env, trigger_state, stable_state, candidate["object"]
    )
    set_flat_state(env, object_advanced_state)
    if not eval_predicate(env, candidate["predicate"]):
        result["reason"] = "advanced_predicate_invalid"
        return result
    if any(
        eval_predicate(env, other["predicate"])
        for other in task_cfg["candidates"]
        if other["object"] != candidate["object"]
    ):
        result["reason"] = "advanced_state_changes_another_target"
        return result

    fractions: dict[int, np.ndarray] = {}
    for fraction in (25, 50, 75):
        index = int(round((fraction / 100.0) * stable_index))
        index = max(0, min(index, stable_index))
        fractions[fraction] = trajectory_states[index]

    remaining = candidate["remaining_instruction"]
    explicit = candidate.get(
        "explicit_progress_instruction",
        f"The subgoal involving {candidate['object']} is already complete. {remaining}",
    )
    incorrect = candidate.get(
        "incorrect_instruction",
        f"Move {candidate['object']} again before doing anything else.",
    )
    metadata = {
        "pair_id": pair_id,
        "task_key": task_key,
        "task_id": task_id,
        "initial_state_id": int(initial_state_id),
        "instruction": instruction,
        "remaining_instruction": remaining,
        "explicit_progress_instruction": explicit,
        "incorrect_instruction": incorrect,
        "target_object": candidate["object"],
        "target_predicate": candidate["predicate"],
        "wait_steps": wait_steps,
        "prefix_steps": len(prefix_actions),
        "stable_index": int(stable_index),
        "source_success": success,
        "collection_mode": "dynamic_candidate_screen",
    }
    write_pair(
        pair_dir,
        metadata,
        initial_state=initial_state,
        prefix_actions=np.asarray(prefix_actions, dtype=np.float32).reshape(-1, 7),
        trigger_state=trigger_state,
        object_advanced_state=object_advanced_state,
        endpoint_state=stable_state,
        phase_states=fractions,
    )
    validate_pair(pair_dir)
    result.update(
        {
            "accepted": True,
            "reason": "accepted",
            "pair_dir": str(pair_dir.resolve()),
            "prefix_steps": len(prefix_actions),
            "stable_index": int(stable_index),
        }
    )
    return result


def run_collection_worker(
    cfg: dict[str, Any],
    worker_id: str | None = None,
) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    queue_path = output_root / "queues" / "collect.sqlite3"
    if not queue_path.is_file():
        raise FileNotFoundError(f"Collection queue is missing: {queue_path}")
    worker_id = worker_id or default_worker_id("collect")
    result_path = output_root / "collection_results" / f"{worker_id}.jsonl"
    candidates_root = output_root / "candidate_pairs"
    candidates_root.mkdir(parents=True, exist_ok=True)
    lease_seconds = int(cfg.get("compute", {}).get("lease_seconds", 3600))
    prefer_same_task = bool(cfg.get("compute", {}).get("prefer_same_task", True))

    suite = make_suite(cfg["environment"]["suite"])
    task_configs = {int(task["task_id"]): task for task in cfg["tasks"]}
    policy = OpenVLAOFTPolicy(cfg["model"], cfg["environment"]["suite"])
    torch_module = getattr(policy, "torch", None)
    if torch_module is not None:
        cpu_threads = max(1, int(cfg.get("compute", {}).get("cpu_threads_per_worker", 2)))
        torch_module.set_num_threads(cpu_threads)
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch_module.set_float32_matmul_precision("high")
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True
        torch_module.cuda.reset_peak_memory_stats(0)
    env = None
    current_task_id: int | None = None
    instruction = ""
    initial_states = None
    processed = 0
    accepted = 0
    failures = 0

    try:
        with SQLiteWorkQueue(queue_path, lease_seconds) as queue:
            while True:
                claimed = queue.claim(
                    worker_id,
                    preferred_task_id=current_task_id if prefer_same_task else None,
                )
                if claimed is None:
                    break
                started = time.time()
                payload = claimed.payload
                task_id = int(payload["task_id"])
                if task_id != current_task_id:
                    if env is not None:
                        env.close()
                    benchmark_task = suite.get_task(task_id)
                    env, instruction = make_env(benchmark_task, cfg["environment"])
                    initial_states = suite.get_task_init_states(task_id)
                    current_task_id = task_id
                assert env is not None and initial_states is not None
                row: dict[str, Any] = {
                    "record_type": "collection_candidate",
                    "work_id": claimed.work_id,
                    "worker_id": worker_id,
                    "physical_gpu": os.environ.get("PF_PHYSICAL_GPU"),
                    "gpu_slot": os.environ.get("PF_GPU_SLOT"),
                    "attempt": claimed.attempts,
                    "completed": False,
                }
                try:
                    outcome = _collect_candidate(
                        cfg,
                        task_configs[task_id],
                        int(payload["initial_state_id"]),
                        env=env,
                        instruction=instruction,
                        initial_states=initial_states,
                        policy=policy,
                        destination_root=candidates_root,
                    )
                    row.update(outcome)
                    row["completed"] = True
                    queue.complete(claimed.work_id, worker_id)
                    accepted += int(bool(outcome.get("accepted")))
                except Exception as exc:
                    failures += 1
                    row.update(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "retryable": True,
                        }
                    )
                    queue.fail(claimed.work_id, worker_id, row["error"])
                    LOGGER.exception("candidate collection failed: %s", claimed.work_id)
                row["wall_seconds"] = time.time() - started
                _append_jsonl(result_path, row)
                processed += 1
    finally:
        if env is not None:
            env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    if failures:
        raise RuntimeError(
            f"Collection worker {worker_id} recorded {failures} retryable failures"
        )
    return {
        "worker_id": worker_id,
        "processed": processed,
        "accepted": accepted,
        "result_path": str(result_path),
    }


def _latest_collection_rows(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return latest
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("work_id"):
                latest[str(row["work_id"])] = row
    return latest


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def freeze_candidate_pairs(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    status = queue_summary(cfg, "collect")
    counts = status["counts"]
    if counts["pending"] or counts["running"] or counts["failed"]:
        raise RuntimeError(f"Collection queue is not complete: {counts}")

    latest = _latest_collection_rows(output_root / "collection_results")
    candidate_count = int(cfg["data"].get("candidate_initial_states", 50))
    expected = len(cfg["tasks"]) * candidate_count
    completed = [row for row in latest.values() if row.get("completed")]
    if len(completed) != expected:
        raise RuntimeError(
            f"Expected {expected} completed candidate screens, found {len(completed)}"
        )

    required = int(cfg["data"].get("pairs_per_task", 10))
    selected: dict[str, list[dict[str, Any]]] = {}
    for task in cfg["tasks"]:
        task_key = str(task["key"])
        accepted = sorted(
            (
                row
                for row in completed
                if row.get("task_key") == task_key and row.get("accepted")
            ),
            key=lambda row: int(row["initial_state_id"]),
        )
        if len(accepted) < required:
            raise RuntimeError(
                f"Task {task_key} has only {len(accepted)} accepted candidates; required={required}"
            )
        selected[task_key] = accepted[:required]

    lock = {
        "config_fingerprint": fingerprint(cfg),
        "selection_rule": (
            "lowest initial_state_id among all completed nominal-policy candidate screens; "
            "selection frozen before intervention outcomes"
        ),
        "pairs_per_task": required,
        "selected": {
            task_key: [
                {
                    "pair_id": row["pair_id"],
                    "initial_state_id": int(row["initial_state_id"]),
                    "pair_dir": row["pair_dir"],
                }
                for row in rows
            ]
            for task_key, rows in selected.items()
        },
    }
    lock_path = output_root / "cohort_lock.json"
    pairs_root = output_root / "pairs"
    if lock_path.is_file() and pairs_root.is_dir() and not force:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing == lock:
            return existing
        raise RuntimeError(
            "A different cohort is already frozen in this output root. Use a new output root "
            "or pass --force-refreeze deliberately."
        )
    if pairs_root.exists():
        if not force:
            raise RuntimeError(
                f"{pairs_root} already exists without a matching cohort lock; refusing to overwrite"
            )
        backup = output_root / f"pairs.backup.{int(time.time())}"
        pairs_root.replace(backup)
    pairs_root.mkdir(parents=True, exist_ok=False)
    for rows in selected.values():
        for row in rows:
            source = Path(str(row["pair_dir"]))
            validate_pair(source)
            destination = pairs_root / str(row["pair_id"])
            shutil.copytree(source, destination, copy_function=_link_or_copy)
            validate_pair(destination)
    atomic_json(lock_path, lock)
    atomic_json(output_root / "collection_summary.json", lock)
    return lock
