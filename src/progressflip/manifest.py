from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .conditions import resolve
from .config import atomic_json, fingerprint, stable_int
from .records import PairRecord, validate_pair


def build_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    pairs_root = output_root / "pairs"
    manifest_path = output_root / "manifest.jsonl"
    conditions = resolve(cfg["experiment"]["conditions"])
    task_keys = {task["key"] for task in cfg["tasks"]}
    pair_dirs = sorted(path for path in pairs_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {key: 0 for key in task_keys}
    for pair_dir in pair_dirs:
        validate_pair(pair_dir)
        pair = PairRecord.load(pair_dir)
        if pair.task_key not in task_keys:
            continue
        pair_counts[pair.task_key] += 1
        for condition in conditions:
            rows.append(
                {
                    "job_id": f"{pair.pair_id}--{condition.name}",
                    "pair_id": pair.pair_id,
                    "pair_dir": str(pair_dir.resolve()),
                    "task_key": pair.task_key,
                    "task_id": pair.task_id,
                    "condition": condition.name,
                    "shard_key": stable_int(pair.pair_id),
                }
            )
    required = int(cfg["data"].get("pairs_per_task", 10))
    underfilled = {key: count for key, count in pair_counts.items() if count < required}
    if underfilled:
        raise RuntimeError(f"Insufficient locked pairs: required={required}, observed={underfilled}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "config_fingerprint": fingerprint(cfg),
        "manifest": str(manifest_path.resolve()),
        "pairs": pair_counts,
        "conditions": [condition.name for condition in conditions],
        "jobs": len(rows),
    }
    atomic_json(output_root / "manifest_summary.json", summary)
    return summary


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
