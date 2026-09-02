from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import atomic_json


def _number(value: str) -> float:
    cleaned = value.strip().replace("%", "").replace(" MiB", "").replace(" W", "")
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return float("nan")
    return float(cleaned)


def _percentile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if value == value)
    if not finite:
        return float("nan")
    position = (len(finite) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _slot_recommendation(
    rows: list[dict[str, Any]], plan: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not plan or not rows:
        return None
    current = int(plan["workers_per_gpu"])
    maximum = int(plan.get("max_workers_per_gpu", current))
    per_worker = int(plan.get("per_worker_memory_mb", 0))
    reserve = int(plan.get("reserve_memory_mb", 0))
    cluster_util = mean(row["gpu_util_mean"] for row in rows)
    cluster_idle = mean(row["idle_fraction_below_10pct"] for row in rows)
    min_headroom = min(
        row["memory_total_mb"] - row["memory_used_peak_mb"] for row in rows
    )
    if current < maximum and cluster_util < 65.0 and cluster_idle > 0.15:
        if min_headroom >= per_worker + max(2000, reserve // 3):
            recommended = current + 1
            reason = "GPU kernels are frequently idle and measured memory headroom fits another model worker"
        else:
            recommended = current
            reason = "GPU utilization is low, but measured memory headroom is insufficient for another replica"
    elif any(row["memory_peak_fraction"] >= 0.94 for row in rows) and current > 1:
        recommended = current - 1
        reason = "peak device memory exceeded 94%; reduce replicas to avoid CUDA OOM"
    else:
        recommended = current
        reason = "keep the current slot count"
    return {
        "current_workers_per_gpu": current,
        "recommended_workers_per_gpu_next_run": recommended,
        "reason": reason,
        "cluster_mean_util_percent": cluster_util,
        "cluster_idle_fraction_below_10pct": cluster_idle,
        "minimum_measured_memory_headroom_mb": min_headroom,
    }


def summarize_gpu_metrics(
    csv_path: str | Path,
    output_root: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(csv_path)
    by_gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                gpu = int(row["gpu_index"].strip())
            except (KeyError, ValueError):
                continue
            for source, target in (
                ("gpu_util_percent", "gpu_util"),
                ("memory_util_percent", "memory_util"),
                ("memory_used_mb", "memory_used"),
                ("memory_total_mb", "memory_total"),
                ("power_draw_w", "power_draw"),
                ("sm_clock_mhz", "sm_clock"),
            ):
                try:
                    by_gpu[gpu][target].append(_number(row[source]))
                except KeyError:
                    pass
    rows = []
    for gpu, metrics in sorted(by_gpu.items()):
        util = [value for value in metrics.get("gpu_util", []) if value == value]
        memory = [value for value in metrics.get("memory_used", []) if value == value]
        totals = [value for value in metrics.get("memory_total", []) if value == value]
        total = max(totals) if totals else float("nan")
        peak = max(memory) if memory else float("nan")
        row = {
            "gpu_index": gpu,
            "samples": len(util),
            "gpu_util_mean": mean(util) if util else float("nan"),
            "gpu_util_median": median(util) if util else float("nan"),
            "gpu_util_p95": _percentile(util, 0.95),
            "idle_fraction_below_10pct": (
                sum(value < 10.0 for value in util) / len(util) if util else float("nan")
            ),
            "busy_fraction_above_70pct": (
                sum(value >= 70.0 for value in util) / len(util) if util else float("nan")
            ),
            "memory_used_peak_mb": peak,
            "memory_total_mb": total,
            "memory_peak_fraction": peak / total if total and total == total else float("nan"),
            "power_draw_mean_w": mean(
                value for value in metrics.get("power_draw", []) if value == value
            )
            if any(value == value for value in metrics.get("power_draw", []))
            else float("nan"),
        }
        rows.append(row)
    all_util = [
        value
        for metrics in by_gpu.values()
        for value in metrics.get("gpu_util", [])
        if value == value
    ]
    plan = None
    if plan_path is not None and Path(plan_path).is_file():
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    summary = {
        "source_csv": str(path.resolve()),
        "worker_plan": plan,
        "gpus": rows,
        "cluster_gpu_util_mean": mean(all_util) if all_util else float("nan"),
        "cluster_idle_fraction_below_10pct": (
            sum(value < 10.0 for value in all_util) / len(all_util)
            if all_util
            else float("nan")
        ),
        "slot_recommendation": _slot_recommendation(rows, plan),
    }
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "gpu_utilization_summary.json", summary)
        lines = [
            "# GPU utilization report",
            "",
            f"Source: `{path.resolve()}`",
            "",
            "| GPU | samples | mean util | p95 util | idle <10% | busy ≥70% | peak memory |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['gpu_index']} | {row['samples']} | {row['gpu_util_mean']:.1f}% | "
                f"{row['gpu_util_p95']:.1f}% | {100*row['idle_fraction_below_10pct']:.1f}% | "
                f"{100*row['busy_fraction_above_70pct']:.1f}% | "
                f"{row['memory_used_peak_mb']:.0f}/{row['memory_total_mb']:.0f} MiB |"
            )
        lines.extend(
            [
                "",
                f"Cluster mean GPU utilization: **{summary['cluster_gpu_util_mean']:.1f}%**",
            ]
        )
        recommendation = summary["slot_recommendation"]
        if recommendation:
            lines.extend(
                [
                    "",
                    "## Next-run slot recommendation",
                    "",
                    f"Current: **{recommendation['current_workers_per_gpu']} workers/GPU**",
                    f"Recommended: **{recommendation['recommended_workers_per_gpu_next_run']} workers/GPU**",
                    f"Reason: {recommendation['reason']}.",
                ]
            )
        lines.extend(
            [
                "",
                "Treat this as a throughput recommendation, not a scientific result. "
                "Always keep the same frozen manifest when comparing launcher settings.",
            ]
        )
        (root / "gpu_utilization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
