from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    task_key: str
    task_id: int
    instruction: str
    remaining_instruction: str
    explicit_progress_instruction: str
    incorrect_instruction: str
    initial_state: np.ndarray
    prefix_actions: np.ndarray
    trigger_state: np.ndarray
    object_advanced_state: np.ndarray
    endpoint_state: np.ndarray
    phase_states: dict[int, np.ndarray]
    target_object: str
    target_predicate: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path) -> PairRecord:
        root = Path(directory)
        metadata = json.loads((root / "pair.json").read_text(encoding="utf-8"))
        with np.load(root / "states.npz", allow_pickle=False) as arrays:
            phases = {
                fraction: arrays[f"phase_{fraction}_state"].copy()
                for fraction in (25, 50, 75)
                if f"phase_{fraction}_state" in arrays.files
            }
            return cls(
                pair_id=str(metadata["pair_id"]),
                task_key=str(metadata["task_key"]),
                task_id=int(metadata["task_id"]),
                instruction=str(metadata["instruction"]),
                remaining_instruction=str(metadata["remaining_instruction"]),
                explicit_progress_instruction=str(metadata["explicit_progress_instruction"]),
                incorrect_instruction=str(metadata["incorrect_instruction"]),
                initial_state=arrays["initial_state"].copy(),
                prefix_actions=arrays["prefix_actions"].copy(),
                trigger_state=arrays["trigger_state"].copy(),
                object_advanced_state=arrays["object_advanced_state"].copy(),
                endpoint_state=arrays["endpoint_state"].copy(),
                phase_states=phases,
                target_object=str(metadata["target_object"]),
                target_predicate=tuple(str(item) for item in metadata["target_predicate"]),
                metadata=metadata,
            )

    def instruction_for(self, mode: str) -> str:
        mapping = {
            "original": self.instruction,
            "remaining": self.remaining_instruction,
            "explicit_progress": self.explicit_progress_instruction,
            "incorrect": self.incorrect_instruction,
        }
        return mapping[mode]


def _write_pair_contents(
    root: Path,
    metadata: dict[str, Any],
    *,
    initial_state: np.ndarray,
    prefix_actions: np.ndarray,
    trigger_state: np.ndarray,
    object_advanced_state: np.ndarray,
    endpoint_state: np.ndarray,
    phase_states: dict[int, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "initial_state": np.asarray(initial_state),
        "prefix_actions": np.asarray(prefix_actions, dtype=np.float32),
        "trigger_state": np.asarray(trigger_state),
        "object_advanced_state": np.asarray(object_advanced_state),
        "endpoint_state": np.asarray(endpoint_state),
    }
    for fraction, state in phase_states.items():
        arrays[f"phase_{int(fraction)}_state"] = np.asarray(state)
    np.savez_compressed(root / "states.npz", **arrays)
    (root / "pair.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    checksums = {
        path.name: sha256_file(path) for path in (root / "pair.json", root / "states.npz")
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


def write_pair(
    directory: str | Path,
    metadata: dict[str, Any],
    *,
    initial_state: np.ndarray,
    prefix_actions: np.ndarray,
    trigger_state: np.ndarray,
    object_advanced_state: np.ndarray,
    endpoint_state: np.ndarray,
    phase_states: dict[int, np.ndarray],
) -> None:
    root = Path(directory)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(root)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.writing-", dir=root.parent))
    try:
        _write_pair_contents(
            temporary,
            metadata,
            initial_state=initial_state,
            prefix_actions=prefix_actions,
            trigger_state=trigger_state,
            object_advanced_state=object_advanced_state,
            endpoint_state=endpoint_state,
            phase_states=phase_states,
        )
        validate_pair(temporary)
        temporary.replace(root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_pair(directory: str | Path) -> None:
    root = Path(directory)
    expected = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, digest in expected.items():
        observed = sha256_file(root / name)
        if observed != digest:
            raise ValueError(f"Checksum mismatch for {root / name}: {observed} != {digest}")
    pair = PairRecord.load(root)
    if pair.prefix_actions.ndim != 2 or pair.prefix_actions.shape[1] != 7:
        raise ValueError(f"Invalid prefix action shape for {pair.pair_id}: {pair.prefix_actions.shape}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
