from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("The top-level YAML value must be a mapping")
    cfg = _expand(raw)
    for key in ("model", "environment", "data", "tasks", "experiment"):
        if key not in cfg:
            raise ConfigError(f"Missing top-level key: {key}")
    if not isinstance(cfg["tasks"], list) or not cfg["tasks"]:
        raise ConfigError("tasks must be a non-empty list")
    if not isinstance(cfg["experiment"].get("conditions"), list):
        raise ConfigError("experiment.conditions must be a list")
    cfg["_config_path"] = str(config_path)
    cfg["data"]["output_root"] = str(Path(cfg["data"]["output_root"]).expanduser().resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    names = [str(task["key"]) for task in cfg["tasks"]]
    if len(names) != len(set(names)):
        raise ConfigError("Task keys must be unique")
    for task in cfg["tasks"]:
        for required in ("key", "task_id", "candidates"):
            if required not in task:
                raise ConfigError(f"Task {task!r} is missing {required}")
        if not task["candidates"]:
            raise ConfigError(f"Task {task['key']} has no target candidates")
        for candidate in task["candidates"]:
            for required in ("object", "predicate", "remaining_instruction"):
                if required not in candidate:
                    raise ConfigError(
                        f"Candidate in task {task['key']} is missing {required}: {candidate!r}"
                    )
    conditions = cfg["experiment"]["conditions"]
    if len(conditions) != len(set(conditions)):
        raise ConfigError("Condition names must be unique")
    if int(cfg["environment"].get("max_steps", 0)) <= 0:
        raise ConfigError("environment.max_steps must be positive")


def fingerprint(cfg: dict[str, Any]) -> str:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_int(value: Any) -> int:
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(target)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")
