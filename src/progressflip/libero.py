from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .vision import Capture, GeomGroups


class LiberoError(RuntimeError):
    """Raised when the installed LIBERO / MuJoCo API is incompatible."""


def make_suite(name: str):
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if name not in suites:
        raise KeyError(f"Unknown benchmark {name!r}; available={sorted(suites)}")
    return suites[name]()


def make_env(task: Any, cfg: dict[str, Any]):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(path),
        camera_heights=int(cfg.get("image_size", 256)),
        camera_widths=int(cfg.get("image_size", 256)),
        render_gpu_device_id=int(cfg.get("render_gpu_device_id", 0)),
        control_freq=int(cfg.get("control_freq", 20)),
    )
    env.seed(0)
    return env, task.language


def base_env(env: Any) -> Any:
    return getattr(env, "env", env)


def flat_state(env: Any) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        return np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()


def set_flat_state(env: Any, state: np.ndarray) -> dict[str, np.ndarray]:
    vector = np.asarray(state, dtype=np.float64)
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(vector)
    env.sim.set_state_from_flattened(vector)
    env.sim.forward()
    underlying = base_env(env)
    if hasattr(underlying, "_post_process"):
        underlying._post_process()
    if hasattr(underlying, "_update_observables"):
        underlying._update_observables(force=True)
    return underlying._get_observations()


def reset_and_wait(env: Any, initial_state: np.ndarray, wait_steps: int, dummy_action: np.ndarray):
    env.reset()
    obs = env.set_init_state(np.asarray(initial_state))
    for _ in range(int(wait_steps)):
        obs, _, _, _ = env.step(np.asarray(dummy_action, dtype=np.float32).tolist())
    return obs


def replay(env: Any, actions: np.ndarray):
    obs = None
    for action in np.asarray(actions):
        obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
        if done and not bool(env.check_success()):
            break
    return obs


def max_state_error(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return float("inf")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def goal_predicates(env: Any) -> list[list[str]]:
    parsed = base_env(env).parsed_problem
    return [
        [str(item).lower() if index == 0 else str(item) for index, item in enumerate(predicate)]
        for predicate in parsed["goal_state"]
    ]


def eval_predicate(env: Any, predicate: Iterable[str]) -> bool:
    underlying = base_env(env)
    if not hasattr(underlying, "_eval_predicate"):
        raise LiberoError("LIBERO environment has no _eval_predicate")
    return bool(underlying._eval_predicate(list(predicate)))


def eval_goals(env: Any) -> list[bool]:
    return [eval_predicate(env, predicate) for predicate in goal_predicates(env)]


def _name_to_id(model: Any, kind: str, name: str) -> int:
    method = getattr(model, f"{kind}_name2id", None)
    if method is not None:
        return int(method(name))
    raise LiberoError(f"MuJoCo model cannot resolve {kind} name {name!r}")


def _joint_width(model: Any, joint_id: int, velocity: bool = False) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if velocity:
        return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def object_joint(env: Any, object_name: str) -> str:
    objects = base_env(env).objects_dict
    if object_name not in objects:
        raise KeyError(f"Unknown movable object {object_name!r}; available={sorted(objects)}")
    joints = list(objects[object_name].joints)
    if not joints:
        raise LiberoError(f"Object {object_name!r} has no joint")
    return str(joints[-1])


def joint_qpos(env: Any, joint_name: str) -> np.ndarray:
    data = env.sim.data
    if hasattr(data, "get_joint_qpos"):
        return np.asarray(data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    joint_id = _name_to_id(env.sim.model, "joint", joint_name)
    address = int(env.sim.model.jnt_qposadr[joint_id])
    width = _joint_width(env.sim.model, joint_id)
    return np.asarray(data.qpos[address : address + width], dtype=np.float64).copy()


def joint_qvel(env: Any, joint_name: str) -> np.ndarray:
    data = env.sim.data
    if hasattr(data, "get_joint_qvel"):
        return np.asarray(data.get_joint_qvel(joint_name), dtype=np.float64).copy()
    joint_id = _name_to_id(env.sim.model, "joint", joint_name)
    address = int(env.sim.model.jnt_dofadr[joint_id])
    width = _joint_width(env.sim.model, joint_id, velocity=True)
    return np.asarray(data.qvel[address : address + width], dtype=np.float64).copy()


def set_joint_qpos(env: Any, joint_name: str, values: np.ndarray) -> None:
    data = env.sim.data
    array = np.asarray(values, dtype=np.float64)
    if hasattr(data, "set_joint_qpos"):
        data.set_joint_qpos(joint_name, array)
        return
    joint_id = _name_to_id(env.sim.model, "joint", joint_name)
    address = int(env.sim.model.jnt_qposadr[joint_id])
    data.qpos[address : address + len(array)] = array


def set_joint_qvel(env: Any, joint_name: str, values: np.ndarray) -> None:
    data = env.sim.data
    array = np.asarray(values, dtype=np.float64)
    if hasattr(data, "set_joint_qvel"):
        data.set_joint_qvel(joint_name, array)
        return
    joint_id = _name_to_id(env.sim.model, "joint", joint_name)
    address = int(env.sim.model.jnt_dofadr[joint_id])
    data.qvel[address : address + len(array)] = array


def object_qpos(env: Any, object_name: str) -> np.ndarray:
    values = joint_qpos(env, object_joint(env, object_name))
    if values.shape != (7,):
        raise LiberoError(f"Expected free joint for {object_name}, got {values.shape}")
    return values


def object_qvel(env: Any, object_name: str) -> np.ndarray:
    return joint_qvel(env, object_joint(env, object_name))


def set_object_qpos(env: Any, object_name: str, values: np.ndarray) -> None:
    joint = object_joint(env, object_name)
    qpos = np.asarray(values, dtype=np.float64).reshape(7).copy()
    norm = float(np.linalg.norm(qpos[3:]))
    if norm <= 1e-12:
        raise ValueError("Object quaternion cannot be zero")
    qpos[3:] /= norm
    set_joint_qpos(env, joint, qpos)
    set_joint_qvel(env, joint, np.zeros(6, dtype=np.float64))
    env.sim.forward()


def make_object_advanced_state(
    env: Any,
    trigger_state: np.ndarray,
    stable_state: np.ndarray,
    object_name: str,
) -> np.ndarray:
    set_flat_state(env, stable_state)
    target_qpos = object_qpos(env, object_name)
    set_flat_state(env, trigger_state)
    set_object_qpos(env, object_name, target_qpos)
    return flat_state(env)


def eef_position(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)


def body_position(env: Any, name: str) -> np.ndarray:
    body_id = _name_to_id(env.sim.model, "body", name)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def eef_object_distance(env: Any, obs: dict[str, np.ndarray], object_name: str) -> float:
    return float(np.linalg.norm(eef_position(obs) - body_position(env, object_name)))


def gripper_aperture(obs: dict[str, np.ndarray]) -> float:
    return float(np.sum(np.abs(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64))))


def raw_camera_image(obs: dict[str, np.ndarray], camera: str) -> np.ndarray:
    key = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}[camera]
    return np.asarray(obs[key], dtype=np.uint8).copy()


def _render_candidates(image: np.ndarray) -> list[np.ndarray]:
    return [image, image[::-1], image[:, ::-1], image[::-1, ::-1]]


def calibrated_render(env: Any, obs: dict[str, np.ndarray], camera: str, segmentation: bool = False):
    name = "agentview" if camera == "agentview" else "robot0_eye_in_hand"
    height, width = raw_camera_image(obs, camera).shape[:2]
    rendered = env.sim.render(
        camera_name=name,
        height=height,
        width=width,
        segmentation=bool(segmentation),
    )
    reference = raw_camera_image(obs, camera)
    if segmentation:
        rgb_render = env.sim.render(camera_name=name, height=height, width=width)
        errors = [
            float(np.mean(np.abs(candidate.astype(float) - reference.astype(float))))
            for candidate in _render_candidates(rgb_render)
        ]
        return _render_candidates(rendered)[int(np.argmin(errors))].copy()
    candidates = _render_candidates(rendered)
    errors = [
        float(np.mean(np.abs(candidate.astype(float) - reference.astype(float))))
        for candidate in candidates
    ]
    return candidates[int(np.argmin(errors))].copy()


def capture(env: Any, obs: dict[str, np.ndarray], camera: str = "agentview") -> Capture:
    return Capture(
        rgb=raw_camera_image(obs, camera),
        segmentation=calibrated_render(env, obs, camera, segmentation=True),
    )


def discover_robot_geoms(env: Any) -> GeomGroups:
    model = env.sim.model
    robot_ids: list[int] = []
    eef_ids: list[int] = []
    eef_tokens = ("gripper", "hand", "finger", "eef", "wrist")
    robot_tokens = ("robot0", "panda", "gripper", "hand", "finger")
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) if hasattr(model, "geom_id2name") else None
        if not name:
            continue
        lowered = str(name).lower()
        if any(token in lowered for token in robot_tokens):
            robot_ids.append(geom_id)
            if any(token in lowered for token in eef_tokens):
                eef_ids.append(geom_id)
    if not robot_ids:
        raise LiberoError("Could not discover robot geometry IDs")
    return GeomGroups.from_ids(robot_ids, eef_ids)


@contextlib.contextmanager
def restored_state(env: Any) -> Iterator[None]:
    state = flat_state(env)
    try:
        yield
    finally:
        set_flat_state(env, state)


def capture_at_state(env: Any, state: np.ndarray, camera: str = "agentview") -> tuple[dict, Capture]:
    with restored_state(env):
        obs = set_flat_state(env, state)
        return obs, capture(env, obs, camera)


def sync_controller_to_current(env: Any) -> None:
    """Reset common robosuite controller goals after direct state restoration."""
    underlying = base_env(env)
    robots = getattr(underlying, "robots", None)
    if not robots:
        raise LiberoError("Environment exposes no robot controller")
    robot = robots[0]
    controller = getattr(robot, "controller", None)
    if controller is None:
        raise LiberoError("Robot exposes no controller")
    refreshed = False
    if hasattr(controller, "update_initial_joints"):
        controller.update_initial_joints(np.asarray(robot._joint_positions, dtype=np.float64))
        refreshed = True
    if hasattr(controller, "goal_pos") and hasattr(controller, "ee_pos"):
        controller.goal_pos = np.asarray(controller.ee_pos, dtype=np.float64).copy()
        refreshed = True
    if hasattr(controller, "goal_ori") and hasattr(controller, "ee_ori_mat"):
        controller.goal_ori = np.asarray(controller.ee_ori_mat, dtype=np.float64).copy()
        refreshed = True
    if hasattr(controller, "set_goal"):
        control_dim = int(getattr(controller, "control_dim", 6))
        try:
            controller.set_goal(np.zeros(control_dim, dtype=np.float64))
            refreshed = True
        except Exception:
            pass
    if not refreshed:
        raise LiberoError(f"Unsupported controller type: {type(controller)!r}")


def inspect_task(env: Any) -> dict[str, Any]:
    underlying = base_env(env)
    return {
        "objects": sorted(underlying.objects_dict),
        "fixtures": sorted(getattr(underlying, "fixtures_dict", {})),
        "goals": goal_predicates(env),
    }
