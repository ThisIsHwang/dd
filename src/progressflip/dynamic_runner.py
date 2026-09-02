from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .actions import QueryBank, mean_absolute_error, scale_action, select_chunk
from .conditions import Condition, registry
from .libero import (
    eval_goals,
    eval_predicate,
    flat_state,
    make_env,
    make_suite,
    max_state_error,
    raw_camera_image,
    replay,
    reset_and_wait,
    set_flat_state,
    sync_controller_to_current,
)
from .policy import OpenVLAOFTPolicy
from .query_cache import PolicyQueryCache
from .records import PairRecord
from .runner import (
    _append_jsonl,
    _capture_donors,
    _digest,
    _save_video,
    _view_images,
)
from .vision import image_mae
from .workqueue import SQLiteWorkQueue, default_worker_id


LOGGER = logging.getLogger(__name__)


@dataclass
class PairContext:
    endpoint_obs: dict[str, np.ndarray]
    true_wrist: np.ndarray
    old_wrist: np.ndarray
    composites: Any
    image_diagnostics: dict[str, Any]
    baseline_state: np.ndarray
    baseline_goals: list[bool]
    query_cache: PolicyQueryCache = field(default_factory=PolicyQueryCache)


def _build_pair_context(
    cfg: dict[str, Any], env: Any, pair: PairRecord
) -> PairContext:
    endpoint_obs, true_wrist, old_wrist, composites, diagnostics = _capture_donors(
        env, pair
    )
    if not eval_predicate(env, pair.target_predicate):
        raise RuntimeError("Target progress predicate is false at endpoint baseline")
    return PairContext(
        endpoint_obs={key: np.asarray(value).copy() for key, value in endpoint_obs.items()},
        true_wrist=np.asarray(true_wrist, dtype=np.uint8).copy(),
        old_wrist=np.asarray(old_wrist, dtype=np.uint8).copy(),
        composites=composites,
        image_diagnostics=diagnostics,
        baseline_state=np.asarray(pair.endpoint_state).copy(),
        baseline_goals=list(eval_goals(env)),
        query_cache=PolicyQueryCache(
            enabled=bool(cfg.get("compute", {}).get("query_cache", True))
        ),
    )


def _execute_condition_cached(
    cfg: dict[str, Any],
    env: Any,
    policy: OpenVLAOFTPolicy,
    pair: PairRecord,
    condition: Condition,
    output_root: Path,
    context: PairContext,
    save_video_pair: bool,
) -> dict[str, Any]:
    instruction = pair.instruction_for(condition.instruction)
    condition_agent, condition_wrist = _view_images(
        condition,
        context.endpoint_obs,
        context.true_wrist,
        context.old_wrist,
        context.composites,
    )
    cache_hits_before = context.query_cache.hits
    cache_misses_before = context.query_cache.misses
    true_raw = context.query_cache.query(
        policy,
        context.endpoint_obs,
        instruction,
        agent_image=context.composites.true,
        wrist_image=context.true_wrist,
    )
    stale_raw = context.query_cache.query(
        policy,
        context.endpoint_obs,
        instruction,
        agent_image=context.composites.full_old,
        wrist_image=context.old_wrist,
    )
    query_raw = context.query_cache.query(
        policy,
        context.endpoint_obs,
        instruction,
        agent_image=condition_agent,
        wrist_image=condition_wrist,
    )
    bank = QueryBank(
        true_raw=true_raw,
        stale_raw=stale_raw,
        true_env=policy.process_chunk(true_raw),
        stale_env=policy.process_chunk(stale_raw),
    )
    query_env = policy.process_chunk(query_raw)
    selected = select_chunk(bank, condition.action_source, query_env)

    frames: list[np.ndarray] = []
    save_video = (
        bool(cfg["data"].get("save_videos", True))
        and save_video_pair
        and condition.name in set(cfg["data"].get("video_conditions", []))
    )
    if save_video:
        frames.append(np.asarray(condition_agent, dtype=np.uint8))

    first_actions: list[np.ndarray] = []
    obs = context.endpoint_obs
    for index in range(min(condition.execute_steps, len(selected))):
        action = scale_action(selected[index], condition.action_scale, scale_gripper=False)
        obs, _, done, _ = env.step(action.tolist())
        first_actions.append(action)
        if save_video:
            frames.append(raw_camera_image(obs, "agentview"))
        if done and not env.check_success():
            break

    state_after_first = flat_state(env)
    displacement_after_first = max_state_error(
        context.baseline_state, state_after_first
    )
    if condition.reset_after_first:
        obs = set_flat_state(env, context.baseline_state)
        sync_controller_to_current(env)

    goal_history = [eval_goals(env)]
    target_history = [eval_predicate(env, pair.target_predicate)]
    elapsed = len(first_actions)
    max_steps = int(cfg["environment"]["max_steps"])
    open_loop = int(cfg["model"].get("open_loop_steps", 8))
    queue: deque[np.ndarray] = deque()
    success = bool(env.check_success())
    while not success and elapsed < max_steps:
        if not queue:
            raw = policy.query(obs, pair.instruction)
            env_actions = policy.process_chunk(raw[:open_loop])
            queue.extend(env_actions)
        action = queue.popleft()
        obs, _, done, _ = env.step(action.tolist())
        elapsed += 1
        goals = eval_goals(env)
        goal_history.append(goals)
        target_history.append(eval_predicate(env, pair.target_predicate))
        if save_video:
            frames.append(raw_camera_image(obs, "agentview"))
        success = bool(env.check_success())
        if done and not success:
            break

    video_path = _save_video(
        output_root / "videos" / pair.task_key / pair.pair_id / f"{condition.name}.mp4",
        frames,
        fps=int(cfg["environment"].get("control_freq", 20)),
        stride=int(cfg["data"].get("video_stride", 2)),
    )
    trace_path = output_root / "traces" / f"{pair.pair_id}--{condition.name}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        first_actions=np.asarray(first_actions, dtype=np.float32).reshape(-1, 7),
        true_first_chunk=np.asarray(bank.true_env, dtype=np.float32),
        stale_first_chunk=np.asarray(bank.stale_env, dtype=np.float32),
        query_first_chunk=np.asarray(query_env, dtype=np.float32),
        goal_history=np.asarray(goal_history, dtype=bool),
        target_history=np.asarray(target_history, dtype=bool),
    )
    gpu_memory: dict[str, float] = {}
    torch_module = getattr(policy, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        gpu_memory = {
            "gpu_memory_allocated_mb": float(
                torch_module.cuda.memory_allocated(0) / (1024**2)
            ),
            "gpu_memory_reserved_mb": float(
                torch_module.cuda.memory_reserved(0) / (1024**2)
            ),
            "gpu_peak_memory_allocated_mb": float(
                torch_module.cuda.max_memory_allocated(0) / (1024**2)
            ),
        }
    return {
        "success": success,
        "timeout": not success and elapsed >= max_steps,
        "steps": elapsed,
        "target_progress_preserved": bool(all(target_history)),
        "any_new_goal_achieved": bool(
            any(
                any(current and not before for current, before in zip(row, context.baseline_goals))
                for row in goal_history
            )
        ),
        "first_action_count": len(first_actions),
        "first_action_scale": condition.action_scale,
        "first_action_state_displacement": displacement_after_first,
        "true_stale_chunk_mae": mean_absolute_error(bank.true_env, bank.stale_env),
        "query_to_true_chunk_mae": mean_absolute_error(query_env, bank.true_env),
        "query_to_stale_chunk_mae": mean_absolute_error(query_env, bank.stale_env),
        "true_chunk_digest": _digest(bank.true_env),
        "stale_chunk_digest": _digest(bank.stale_env),
        "query_chunk_digest": _digest(query_env),
        "instruction": instruction,
        "image_diagnostics": context.image_diagnostics,
        "condition_agent_mae_to_true": image_mae(condition_agent, context.composites.true),
        "condition_wrist_mae_to_true": image_mae(condition_wrist, context.true_wrist),
        "query_cache_hits_this_condition": context.query_cache.hits - cache_hits_before,
        "query_cache_misses_this_condition": context.query_cache.misses - cache_misses_before,
        "query_cache_hits_pair_total": context.query_cache.hits,
        "query_cache_misses_pair_total": context.query_cache.misses,
        "trace_path": str(trace_path.resolve()),
        "video_path": video_path,
        **gpu_memory,
    }


def _video_pair_ids(cfg: dict[str, Any], output_root: Path) -> set[str] | None:
    limit = cfg["data"].get("video_pairs_per_task")
    if limit is None:
        return None
    limit = max(0, int(limit))
    if limit == 0:
        return set()
    lock_path = output_root / "cohort_lock.json"
    if not lock_path.is_file():
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected: set[str] = set()
    for rows in lock.get("selected", {}).values():
        for row in list(rows)[:limit]:
            selected.add(str(row["pair_id"]))
    return selected


def _completed_across_workers(results_root: Path) -> set[str]:
    completed: set[str] = set()
    if not results_root.is_dir():
        return completed
    for path in sorted(results_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("completed") and row.get("valid") and row.get("job_id"):
                completed.add(str(row["job_id"]))
    return completed


def _run_pair_bundle(
    cfg: dict[str, Any],
    env: Any,
    policy: OpenVLAOFTPolicy,
    pair: PairRecord,
    jobs: list[dict[str, Any]],
    output_root: Path,
    result_path: Path,
    worker_id: str,
    queue: SQLiteWorkQueue,
    queue_work_id: str,
    completed_jobs: set[str],
    attempt: int,
    video_pair_ids: set[str] | None,
) -> int:
    condition_registry = registry()
    context: PairContext | None = None
    finished = 0
    for job in jobs:
        job_id = str(job["job_id"])
        if job_id in completed_jobs:
            continue
        started = time.time()
        row: dict[str, Any] = {
            **job,
            "record_type": "result",
            "worker_id": worker_id,
            "physical_gpu": os.environ.get("PF_PHYSICAL_GPU"),
            "gpu_slot": os.environ.get("PF_GPU_SLOT"),
            "queue_attempt": int(attempt),
            "completed": False,
            "valid": False,
        }
        try:
            obs = reset_and_wait(
                env,
                pair.initial_state,
                int(cfg["environment"].get("wait_steps", 10)),
                policy.dummy_action,
            )
            replayed_obs = replay(env, pair.prefix_actions)
            if replayed_obs is not None:
                obs = replayed_obs
            replay_error = max_state_error(flat_state(env), pair.trigger_state)
            row["replay_error"] = replay_error
            if replay_error > float(cfg["experiment"].get("replay_tolerance", 1e-5)):
                raise RuntimeError(f"Prefix replay mismatch: {replay_error}")
            set_flat_state(env, pair.endpoint_state)
            sync_controller_to_current(env)
            if context is None:
                context = _build_pair_context(cfg, env, pair)
                set_flat_state(env, pair.endpoint_state)
                sync_controller_to_current(env)
            outcome = _execute_condition_cached(
                cfg,
                env,
                policy,
                pair,
                condition_registry[str(job["condition"])],
                output_root,
                context,
                save_video_pair=(
                    video_pair_ids is None or pair.pair_id in video_pair_ids
                ),
            )
            recomposition_mae = outcome["image_diagnostics"]["true_recomposed_mae"]
            if recomposition_mae > float(
                cfg["experiment"].get("recomposition_mae_threshold", 3.0)
            ):
                raise RuntimeError(
                    f"True recomposition control failed: MAE={recomposition_mae}"
                )
            row.update(outcome)
            row.update({"completed": True, "valid": True})
        except Exception as exc:
            row.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "retryable": True,
                }
            )
            row["wall_seconds"] = time.time() - started
            _append_jsonl(result_path, row)
            raise
        row["wall_seconds"] = time.time() - started
        _append_jsonl(result_path, row)
        completed_jobs.add(job_id)
        finished += 1
        queue.heartbeat(queue_work_id, worker_id)
    return finished


def run_dynamic_worker(
    cfg: dict[str, Any],
    worker_id: str | None = None,
) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    queue_path = output_root / "queues" / "run.sqlite3"
    if not queue_path.is_file():
        raise FileNotFoundError(f"Run queue is missing: {queue_path}")
    worker_id = worker_id or default_worker_id("run")
    result_path = output_root / "results" / f"{worker_id}.jsonl"
    lease_seconds = int(cfg.get("compute", {}).get("lease_seconds", 3600))
    prefer_same_task = bool(cfg.get("compute", {}).get("prefer_same_task", True))
    completed_jobs = _completed_across_workers(output_root / "results")
    video_pair_ids = _video_pair_ids(cfg, output_root)

    suite = make_suite(cfg["environment"]["suite"])
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
    processed_pairs = 0
    processed_jobs = 0
    failures = 0
    idle_started = time.time()

    try:
        with SQLiteWorkQueue(queue_path, lease_seconds) as queue:
            while True:
                claimed = queue.claim(
                    worker_id,
                    preferred_task_id=current_task_id if prefer_same_task else None,
                )
                if claimed is None:
                    break
                idle_seconds = time.time() - idle_started
                jobs = list(claimed.payload.get("jobs", []))
                if not jobs:
                    queue.fail(claimed.work_id, worker_id, "pair bundle has no jobs")
                    failures += 1
                    continue
                task_id = int(jobs[0]["task_id"])
                if task_id != current_task_id:
                    if env is not None:
                        env.close()
                    benchmark_task = suite.get_task(task_id)
                    env, _ = make_env(benchmark_task, cfg["environment"])
                    current_task_id = task_id
                assert env is not None
                pair_started = time.time()
                try:
                    pair = PairRecord.load(jobs[0]["pair_dir"])
                    completed_now = _run_pair_bundle(
                        cfg,
                        env,
                        policy,
                        pair,
                        jobs,
                        output_root,
                        result_path,
                        worker_id,
                        queue,
                        claimed.work_id,
                        completed_jobs,
                        claimed.attempts,
                        video_pair_ids,
                    )
                    queue.complete(claimed.work_id, worker_id)
                    processed_pairs += 1
                    processed_jobs += completed_now
                    LOGGER.info(
                        "completed pair=%s jobs=%d pair_wall=%.1fs idle_before_claim=%.1fs",
                        claimed.work_id,
                        completed_now,
                        time.time() - pair_started,
                        idle_seconds,
                    )
                except Exception as exc:
                    failures += 1
                    queue.fail(
                        claimed.work_id,
                        worker_id,
                        f"{type(exc).__name__}: {exc}",
                    )
                    LOGGER.exception("pair bundle failed: %s", claimed.work_id)
                idle_started = time.time()
    finally:
        if env is not None:
            env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    if failures:
        raise RuntimeError(
            f"Dynamic worker {worker_id} recorded {failures} failed pair bundle(s)"
        )
    return {
        "worker_id": worker_id,
        "pairs": processed_pairs,
        "jobs": processed_jobs,
        "result_path": str(result_path),
    }
