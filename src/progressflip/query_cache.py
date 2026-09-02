from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class QueryPolicy(Protocol):
    def query(
        self,
        obs: dict[str, np.ndarray],
        instruction: str,
        *,
        agent_image: np.ndarray | None = None,
        wrist_image: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> np.ndarray: ...


def array_digest(value: np.ndarray | None) -> str:
    if value is None:
        return "none"
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def observation_state_digest(obs: dict[str, np.ndarray]) -> str:
    pieces = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key in obs:
            pieces.append(np.asarray(obs[key]).reshape(-1))
    state = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return array_digest(state)


class PolicyQueryCache:
    """Pair-local cache for deterministic, byte-identical policy queries.

    The cache never crosses a causal pair. Its key includes the instruction, both
    images, the optional proprioceptive override, and the endpoint robot state.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.values: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def _key(
        self,
        obs: dict[str, np.ndarray],
        instruction: str,
        agent_image: np.ndarray | None,
        wrist_image: np.ndarray | None,
        proprio: np.ndarray | None,
    ) -> str:
        payload = "|".join(
            (
                instruction,
                observation_state_digest(obs),
                array_digest(agent_image),
                array_digest(wrist_image),
                array_digest(proprio),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def query(
        self,
        policy: QueryPolicy,
        obs: dict[str, np.ndarray],
        instruction: str,
        *,
        agent_image: np.ndarray | None = None,
        wrist_image: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.enabled:
            self.misses += 1
            return np.asarray(
                policy.query(
                    obs,
                    instruction,
                    agent_image=agent_image,
                    wrist_image=wrist_image,
                    proprio=proprio,
                ),
                dtype=np.float32,
            )
        key = self._key(obs, instruction, agent_image, wrist_image, proprio)
        cached = self.values.get(key)
        if cached is not None:
            self.hits += 1
            return cached.copy()
        self.misses += 1
        value = np.asarray(
            policy.query(
                obs,
                instruction,
                agent_image=agent_image,
                wrist_image=wrist_image,
                proprio=proprio,
            ),
            dtype=np.float32,
        )
        self.values[key] = value.copy()
        return value
