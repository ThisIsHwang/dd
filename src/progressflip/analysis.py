from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .config import atomic_json


def _latest_results(paths: list[Path]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                latest[str(row["job_id"])] = row
    return sorted(latest.values(), key=lambda row: (row["pair_id"], row["condition"]))


def _bootstrap_difference(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(outcomes_a) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(outcomes_a), size=(samples, len(outcomes_a)))
    differences = np.mean(outcomes_a[indices] - outcomes_b[indices], axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def _holm(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_contrast(
    index: dict[tuple[str, str], dict[str, Any]],
    a: str,
    b: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = []
    pair_ids = sorted({pair_id for pair_id, _ in index})
    for pair_id in pair_ids:
        row_a = index.get((pair_id, a))
        row_b = index.get((pair_id, b))
        if row_a is not None and row_b is not None:
            pairs.append((row_a, row_b))
    outcomes_a = np.asarray([float(row_a["success"]) for row_a, _ in pairs])
    outcomes_b = np.asarray([float(row_b["success"]) for _, row_b in pairs])
    a1_b0 = int(np.sum((outcomes_a == 1) & (outcomes_b == 0)))
    a0_b1 = int(np.sum((outcomes_a == 0) & (outcomes_b == 1)))
    discordant = a1_b0 + a0_b1
    p_greater = (
        float(binomtest(a1_b0, discordant, p=0.5, alternative="greater").pvalue)
        if discordant
        else 1.0
    )
    ci = _bootstrap_difference(
        outcomes_a,
        outcomes_b,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "a": a,
        "b": b,
        "n": len(pairs),
        "success_a": float(np.mean(outcomes_a)) if len(pairs) else float("nan"),
        "success_b": float(np.mean(outcomes_b)) if len(pairs) else float("nan"),
        "difference_a_minus_b": (
            float(np.mean(outcomes_a - outcomes_b)) if len(pairs) else float("nan")
        ),
        "bootstrap_95ci": list(ci),
        "a1_b0": a1_b0,
        "a0_b1": a0_b1,
        "one_sided_exact_p_a_gt_b": p_greater,
    }


def analyze(cfg: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(cfg["data"]["output_root"])
    result_paths = sorted((output_root / "results").glob("rank*.jsonl"))
    rows = _latest_results(result_paths)
    completed = [row for row in rows if row.get("completed")]
    valid = [row for row in completed if row.get("valid")]
    failures = [row for row in rows if not row.get("completed") or not row.get("valid")]
    index = {(str(row["pair_id"]), str(row["condition"])): row for row in valid}

    by_condition: list[dict[str, Any]] = []
    for condition in cfg["experiment"]["conditions"]:
        subset = [row for row in valid if row["condition"] == condition]
        by_condition.append(
            {
                "condition": condition,
                "n": len(subset),
                "success_rate": float(np.mean([row["success"] for row in subset]))
                if subset
                else float("nan"),
                "timeout_rate": float(np.mean([row.get("timeout", False) for row in subset]))
                if subset
                else float("nan"),
                "progress_preserved_rate": float(
                    np.mean([row.get("target_progress_preserved", False) for row in subset])
                )
                if subset
                else float("nan"),
                "new_goal_rate": float(
                    np.mean([row.get("any_new_goal_achieved", False) for row in subset])
                )
                if subset
                else float("nan"),
                "mean_query_to_true_mae": float(
                    np.mean([row.get("query_to_true_chunk_mae", math.nan) for row in subset])
                )
                if subset
                else float("nan"),
                "mean_query_to_stale_mae": float(
                    np.mean([row.get("query_to_stale_chunk_mae", math.nan) for row in subset])
                )
                if subset
                else float("nan"),
            }
        )

    contrasts = []
    for offset, specification in enumerate(cfg["experiment"].get("confirmatory_contrasts", [])):
        contrasts.append(
            paired_contrast(
                index,
                specification["a"],
                specification["b"],
                bootstrap_samples=int(cfg["experiment"].get("bootstrap_samples", 10000)),
                seed=int(cfg["experiment"].get("analysis_seed", 17)) + offset,
            )
        )
    adjusted = _holm([contrast["one_sided_exact_p_a_gt_b"] for contrast in contrasts])
    for contrast, pvalue in zip(contrasts, adjusted):
        contrast["holm_p"] = pvalue

    expected_jobs = sum(
        1 for _ in (output_root / "manifest.jsonl").open("r", encoding="utf-8")
    )
    summary = {
        "expected_jobs": expected_jobs,
        "observed_jobs": len(rows),
        "completed_jobs": len(completed),
        "valid_jobs": len(valid),
        "failed_or_invalid_jobs": len(failures),
        "failure_reasons": dict(Counter(str(row.get("error", "unknown")) for row in failures)),
        "condition_summary": by_condition,
        "confirmatory_contrasts": contrasts,
        "inference_gate_passed": len(rows) == expected_jobs and len(failures) == 0,
    }
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_json(analysis_root / "summary.json", summary)
    _write_csv(analysis_root / "condition_summary.csv", by_condition)
    _write_csv(analysis_root / "confirmatory_contrasts.csv", contrasts)
    _write_report(analysis_root / "report.md", summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def _pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ProgressFlip visual self-state experiment",
        "",
        f"Inference gate: **{'PASS' if summary['inference_gate_passed'] else 'FAIL'}**",
        "",
        f"Jobs: {summary['completed_jobs']}/{summary['expected_jobs']} completed; "
        f"{summary['failed_or_invalid_jobs']} failed or invalid.",
        "",
        "## Condition results",
        "",
        "| Condition | N | Success | Timeout | Progress preserved | New goal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["condition_summary"]:
        lines.append(
            f"| {row['condition']} | {row['n']} | {_pct(row['success_rate'])} | "
            f"{_pct(row['timeout_rate'])} | {_pct(row['progress_preserved_rate'])} | "
            f"{_pct(row['new_goal_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Confirmatory paired contrasts",
            "",
            "| A | B | N | SR(A) | SR(B) | A−B | 95% CI | Holm p |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in summary["confirmatory_contrasts"]:
        ci = row["bootstrap_95ci"]
        lines.append(
            f"| {row['a']} | {row['b']} | {row['n']} | {_pct(row['success_a'])} | "
            f"{_pct(row['success_b'])} | {_pct(row['difference_a_minus_b'])} | "
            f"[{_pct(ci[0])}, {_pct(ci[1])}] | {row['holm_p']:.4g} |"
        )
    if summary["failure_reasons"]:
        lines.extend(["", "## Failures", ""])
        for reason, count in summary["failure_reasons"].items():
            lines.append(f"- {count} × `{reason}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
