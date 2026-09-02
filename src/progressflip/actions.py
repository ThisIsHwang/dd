from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueryBank:
    true_raw: np.ndarray
    stale_raw: np.ndarray
    true_env: np.ndarray
    stale_env: np.ndarray


def scale_action(action: np.ndarray, scale: float, *, scale_gripper: bool = False) -> np.ndarray:
    output = np.asarray(action, dtype=np.float32).copy()
    if output.shape[-1] != 7:
        raise ValueError(f"Expected seven-dimensional action, got {output.shape}")
    output[..., :6] *= float(scale)
    if scale_gripper:
        output[..., 6] *= float(scale)
    return output


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Action shapes differ: {aa.shape} vs {bb.shape}")
    return float(np.mean(np.abs(aa - bb)))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denominator <= eps:
        return float("nan")
    return float(np.dot(aa, bb) / denominator)


def select_chunk(bank: QueryBank, source: str, query_env: np.ndarray) -> np.ndarray:
    if source == "query":
        return np.asarray(query_env, dtype=np.float32)
    if source == "true":
        return np.asarray(bank.true_env, dtype=np.float32)
    if source == "stale":
        return np.asarray(bank.stale_env, dtype=np.float32)
    raise KeyError(source)
