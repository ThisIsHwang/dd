from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import atomic_json
from .libero import (
    eef_object_distance,
    eval_predicate,
    flat_state,
    goal_predicates,
    gripper_aperture,
    make_env,
    make_object_advanced_state,
    make_suite,
    object_qvel,
    reset_and_wait,
    set_flat_state,
)
from .policy import OpenVLAOFTPolicy
from .records import write_pair

LOGGER = logging.getLogger(__name__)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64).copy()


def _candidate_trigger(env: Any, obs: dict[str, np.ndarray], task: dict[str, Any]):
    trigger = task.get("trigger", {})
    if bool(trigger.get("require_all_goals_false", True)):
        if any(eval_predicate(env, predicate) for predicate in goal_predicates(env)):
            return None
    eligible: list[tuple[float, dict[str, Any]]] = []
    for candidate in task["candidates"]:
        predicate = candidate["predicate"]
        if eval_predicate(env, predicate):
            continue
        distance = eef_object_distance(env, obs, candidate["object"])
        velocity = float(np.linalg.norm(object_qvel(env, candidate["object"])))
        if velocity > float(trigger.get("max_object_velocity", 0.05)):
            continue
        eligible.append((distance, candidate))
    if not eligible:
        return None
    distance, candidate = min(eligible, key=lambda item: item[0])
    if distance > float(trigger.get("distance_m", 0.12)):
        return None
    if bool(trigger.get("require_gripper_open", True)):
        if gripper_aperture(obs) < float(trigger.get("gripper_aperture_min", 0.02)):
            return None
    return candidate


def _process_first(policy: OpenVLAOFTPolicy, obs: dict[str, np.ndarray], instruction: str):
    raw = policy.query(obs, instruction)
    return policy.process_chunk(raw[:1])[0]


def collect_task(cfg: dict[str, Any], task_key: str) -> dict[str, Any]:
    task_cfg = next(task for task in cfg["tasks"] if task["key"] == task_key)
    output_root = Path(cfg["data"]["output_root"])
    pairs_root = output_root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    target_count = int(cfg["data"].get("pairs_per_task", 10))
    candidate_count = int(cfg["data"].get("candidate_initial_states", 50))

    suite = make_suite(cfg["environment"]["suite"])
    task_id = int(task_cfg["task_id"])
    benchmark_task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    env, instruction = make_env(benchmark_task, cfg["environment"])
    policy = OpenVLAOFTPolicy(cfg["model"], cfg["environment"]["suite"])

    accepted = 0
    attempts: list[dict[str, Any]] = []
    wait_steps = int(cfg["environment"].get("wait_steps", 10))
    max_prefix_steps = int(task_cfg.get("trigger", {}).get("max_prefix_steps", 300))
    max_steps = int(cfg["environment"]["max_steps"])
    stable_steps = int(cfg["experiment"].get("predicate_stable_steps", 4))
    stable_velocity = float(cfg["experiment"].get("stable_object_velocity", 0.15))

    try:
        for initial_state_id in range(min(candidate_count, len(initial_states))):
            if accepted >= target_count:
                break
            pair_id = f"{task_key}--init{initial_state_id:03d}"
            pair_dir = pairs_root / pair_id
            if pair_dir.is_dir():
                accepted += 1
                continue
            attempt: dict[str, Any] = {
                "pair_id": pair_id,
                "initial_state_id": initial_state_id,
                "accepted": False,
            }
            try:
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
                    attempt["reason"] = "no_eligible_trigger"
                    attempts.append(attempt)
                    continue

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
                    if env.check_success():
                        break
                    if done:
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
                    attempt["reason"] = "nominal_failure_or_no_stable_checkpoint"
                    attempts.append(attempt)
                    continue

                object_advanced_state = make_object_advanced_state(
                    env, trigger_state, stable_state, candidate["object"]
                )
                set_flat_state(env, object_advanced_state)
                if not eval_predicate(env, candidate["predicate"]):
                    attempt["reason"] = "advanced_predicate_invalid"
                    attempts.append(attempt)
                    continue
                if any(
                    eval_predicate(env, other["predicate"])
                    for other in task_cfg["candidates"]
                    if other["object"] != candidate["object"]
                ):
                    attempt["reason"] = "advanced_state_changes_another_target"
                    attempts.append(attempt)
                    continue

                fractions = {}
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
                    "initial_state_id": initial_state_id,
                    "instruction": instruction,
                    "remaining_instruction": remaining,
                    "explicit_progress_instruction": explicit,
                    "incorrect_instruction": incorrect,
                    "target_object": candidate["object"],
                    "target_predicate": candidate["predicate"],
                    "wait_steps": wait_steps,
                    "prefix_steps": len(prefix_actions),
                    "stable_index": stable_index,
                    "source_success": success,
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
                accepted += 1
                attempt["accepted"] = True
                attempts.append(attempt)
                LOGGER.info("accepted %s (%d/%d)", pair_id, accepted, target_count)
            except Exception as exc:
                attempt["reason"] = f"exception:{type(exc).__name__}:{exc}"
                attempts.append(attempt)
                LOGGER.exception("collection failed for %s", pair_id)
    finally:
        env.close()

    summary = {
        "task_key": task_key,
        "accepted": accepted,
        "required": target_count,
        "attempts": attempts,
    }
    atomic_json(output_root / "collection" / f"{task_key}.json", summary)
    if accepted < target_count:
        raise RuntimeError(f"Collected only {accepted}/{target_count} pairs for {task_key}")
    return summary
