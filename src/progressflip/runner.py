from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .actions import QueryBank, mean_absolute_error, scale_action, select_chunk
from .conditions import Condition, registry
from .config import stable_int
from .libero import (
    capture,
    discover_robot_geoms,
    eval_goals,
    eval_predicate,
    flat_state,
    make_env,
    make_object_advanced_state,
    make_suite,
    max_state_error,
    raw_camera_image,
    replay,
    reset_and_wait,
    set_flat_state,
    sync_controller_to_current,
)
from .manifest import read_manifest
from .policy import OpenVLAOFTPolicy
from .records import PairRecord
from .vision import build_composites, changed_fraction, geom_mask, image_mae

LOGGER = logging.getLogger(__name__)


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    output = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("completed"):
                output.add(str(row["job_id"]))
    return output


def _capture_donors(env: Any, pair: PairRecord):
    endpoint_obs = set_flat_state(env, pair.endpoint_state)
    current = capture(env, endpoint_obs, "agentview")
    current_wrist = raw_camera_image(endpoint_obs, "wrist")

    old_obs = set_flat_state(env, pair.object_advanced_state)
    old = capture(env, old_obs, "agentview")
    old_wrist = raw_camera_image(old_obs, "wrist")

    phase_captures = {}
    for fraction, phase_state in pair.phase_states.items():
        advanced_phase = make_object_advanced_state(
            env, phase_state, pair.endpoint_state, pair.target_object
        )
        phase_obs = set_flat_state(env, advanced_phase)
        phase_captures[fraction] = capture(env, phase_obs, "agentview")

    set_flat_state(env, pair.endpoint_state)
    sync_controller_to_current(env)
    groups = discover_robot_geoms(env)
    random_capture = None
    if phase_captures:
        choices = sorted(phase_captures)
        random_fraction = choices[stable_int((pair.pair_id, "random_robot")) % len(choices)]
        random_capture = phase_captures[random_fraction]
    composites = build_composites(
        current,
        old,
        groups,
        random_capture=random_capture,
        phase_captures=phase_captures,
        dilation_px=2,
        feather_px=0.0,
    )
    diagnostics = {
        "true_recomposed_mae": image_mae(current.rgb, composites.true_recomposed),
        "robot_old_changed_fraction": changed_fraction(current.rgb, composites.robot_old),
        "nonrobot_old_changed_fraction": changed_fraction(current.rgb, composites.nonrobot_old),
        "current_robot_pixel_fraction": float(
            np.mean(geom_mask(current.segmentation, groups.robot))
        ),
        "old_robot_pixel_fraction": float(np.mean(geom_mask(old.segmentation, groups.robot))),
        "current_eef_pixel_fraction": float(np.mean(geom_mask(current.segmentation, groups.eef))),
    }
    return endpoint_obs, current_wrist, old_wrist, composites, diagnostics


def _view_images(
    condition: Condition,
    endpoint_obs: dict[str, np.ndarray],
    true_wrist: np.ndarray,
    old_wrist: np.ndarray,
    composites,
):
    if condition.name == "TRUE_RECOMPOSED_K1":
        return composites.true_recomposed, true_wrist
    if condition.view == "true":
        return composites.true, true_wrist
    if condition.view == "full_old":
        return composites.full_old, old_wrist
    mapping = {
        "robot_old": "robot_old",
        "nonrobot_old": "nonrobot_old",
        "eef_old": "eef_old",
        "arm_old": "arm_old",
        "robot_mask": "robot_mask",
        "robot_random": "robot_random",
        "robot_phase25": "robot_phase25",
        "robot_phase50": "robot_phase50",
        "robot_phase75": "robot_phase75",
    }
    if condition.view not in mapping:
        raise KeyError(condition.view)
    return composites.get(mapping[condition.view]), true_wrist


def _save_video(path: Path, frames: list[np.ndarray], fps: int, stride: int) -> str | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=max(1, int(round(fps / max(1, stride)))), codec="libx264", quality=7
    )
    try:
        for frame in frames[:: max(1, stride)]:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    return str(path)


def _execute_condition(
    cfg: dict[str, Any],
    env: Any,
    policy: OpenVLAOFTPolicy,
    pair: PairRecord,
    condition: Condition,
    output_root: Path,
) -> dict[str, Any]:
    endpoint_obs, true_wrist, old_wrist, composites, image_diagnostics = _capture_donors(env, pair)
    baseline_state = pair.endpoint_state.copy()
    baseline_goals = eval_goals(env)
    if not eval_predicate(env, pair.target_predicate):
        raise RuntimeError("Target progress predicate is false at endpoint baseline")

    instruction = pair.instruction_for(condition.instruction)
    condition_agent, condition_wrist = _view_images(
        condition, endpoint_obs, true_wrist, old_wrist, composites
    )
    true_raw = policy.query(
        endpoint_obs,
        instruction,
        agent_image=composites.true,
        wrist_image=true_wrist,
    )
    stale_raw = policy.query(
        endpoint_obs,
        instruction,
        agent_image=composites.full_old,
        wrist_image=old_wrist,
    )
    query_raw = policy.query(
        endpoint_obs,
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
    save_video = bool(cfg["data"].get("save_videos", True)) and condition.name in set(
        cfg["data"].get("video_conditions", [])
    )
    if save_video:
        frames.append(np.asarray(condition_agent, dtype=np.uint8))

    first_actions: list[np.ndarray] = []
    obs = endpoint_obs
    for index in range(min(condition.execute_steps, len(selected))):
        action = scale_action(selected[index], condition.action_scale, scale_gripper=False)
        obs, _, done, _ = env.step(action.tolist())
        first_actions.append(action)
        if save_video:
            frames.append(raw_camera_image(obs, "agentview"))
        if done and not env.check_success():
            break

    state_after_first = flat_state(env)
    displacement_after_first = max_state_error(baseline_state, state_after_first)
    if condition.reset_after_first:
        obs = set_flat_state(env, baseline_state)
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
    return {
        "success": success,
        "timeout": not success and elapsed >= max_steps,
        "steps": elapsed,
        "target_progress_preserved": bool(all(target_history)),
        "any_new_goal_achieved": bool(
            any(
                any(current and not before for current, before in zip(row, baseline_goals))
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
        "image_diagnostics": image_diagnostics,
        "condition_agent_mae_to_true": image_mae(condition_agent, composites.true),
        "condition_wrist_mae_to_true": image_mae(condition_wrist, true_wrist),
        "trace_path": str(trace_path.resolve()),
        "video_path": video_path,
    }


def run_worker(cfg: dict[str, Any], rank: int, world_size: int) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    manifest = read_manifest(output_root / "manifest.jsonl")
    jobs = [row for row in manifest if int(row["shard_key"]) % world_size == rank]
    result_path = output_root / "results" / f"rank{rank:03d}.jsonl"
    completed = _completed(result_path)
    condition_registry = registry()
    suite = make_suite(cfg["environment"]["suite"])
    policy = OpenVLAOFTPolicy(cfg["model"], cfg["environment"]["suite"])
    total = 0
    failures = 0

    for task_id in sorted({int(row["task_id"]) for row in jobs}):
        benchmark_task = suite.get_task(task_id)
        env, _ = make_env(benchmark_task, cfg["environment"])
        try:
            for job in (row for row in jobs if int(row["task_id"]) == task_id):
                if job["job_id"] in completed:
                    continue
                started = time.time()
                row = {
                    **job,
                    "record_type": "result",
                    "rank": rank,
                    "completed": False,
                    "valid": False,
                }
                try:
                    pair = PairRecord.load(job["pair_dir"])
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
                    outcome = _execute_condition(
                        cfg,
                        env,
                        policy,
                        pair,
                        condition_registry[job["condition"]],
                        output_root,
                    )
                    recomposition_mae = outcome["image_diagnostics"]["true_recomposed_mae"]
                    if recomposition_mae > float(
                        cfg["experiment"].get("recomposition_mae_threshold", 3.0)
                    ):
                        raise RuntimeError(f"True recomposition control failed: MAE={recomposition_mae}")
                    row.update(outcome)
                    row.update({"completed": True, "valid": True})
                except Exception as exc:
                    failures += 1
                    row.update(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "retryable": True,
                        }
                    )
                    LOGGER.exception("job failed: %s", job["job_id"])
                row["wall_seconds"] = time.time() - started
                _append_jsonl(result_path, row)
                total += 1
        finally:
            env.close()

    if failures:
        raise RuntimeError(f"Rank {rank} recorded {failures} retryable job failures")
    return {"rank": rank, "jobs": total, "result_path": str(result_path)}
