from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .analysis import analyze
from .collector import collect_task
from .collector_dynamic import freeze_candidate_pairs, run_collection_worker
from .config import load_config
from .dynamic_runner import run_dynamic_worker
from .gpu_metrics import summarize_gpu_metrics
from .gpu_plan import write_worker_plan
from .manifest import build_manifest
from .preflight import run_preflight
from .runner import run_worker
from .workqueue import (
    initialize_collection_queue,
    initialize_pair_queue,
    queue_summary,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="progressflip")
    root.add_argument("--log-level", default=os.environ.get("PF_LOG_LEVEL", "INFO"))
    commands = root.add_subparsers(dest="command", required=True)

    def config_command(name: str, help_text: str):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        return command

    preflight = config_command("preflight", "Check the H100/OpenVLA/LIBERO environment")
    preflight.add_argument("--expect-gpus", type=int, default=8)

    collect = config_command("collect", "Legacy: collect one task serially")
    collect.add_argument("--task", required=True)

    collect_queue = config_command(
        "collection-queue-init", "Freeze all task × initial-state collection jobs"
    )
    collect_queue.add_argument("--reset-failed", action="store_true")
    collect_queue.add_argument("--reclaim-running", action="store_true")

    collect_worker = config_command(
        "collect-dynamic-worker", "Run one dynamically scheduled collection worker"
    )
    collect_worker.add_argument("--worker-id", default=None)

    freeze = config_command(
        "freeze-pairs", "Select the prospective first-K accepted candidates per task"
    )
    freeze.add_argument("--force-refreeze", action="store_true")

    config_command("manifest", "Freeze the pair × condition manifest")

    run = config_command("run-worker", "Legacy: run one statically sharded GPU worker")
    run.add_argument("--rank", type=int, default=int(os.environ.get("PF_RANK", 0)))
    run.add_argument("--world-size", type=int, default=int(os.environ.get("PF_WORLD_SIZE", 1)))

    run_queue = config_command(
        "run-queue-init", "Build the dynamic pair-bundle execution queue"
    )
    run_queue.add_argument("--reset-failed", action="store_true")
    run_queue.add_argument("--reclaim-running", action="store_true")

    dynamic = config_command(
        "run-dynamic-worker", "Run one pair-preserving dynamic GPU worker"
    )
    dynamic.add_argument("--worker-id", default=None)

    status = config_command("queue-status", "Print persisted queue state")
    status.add_argument("--kind", choices=("collect", "run"), required=True)

    gpu_plan = config_command(
        "gpu-plan", "Resolve safe concurrent worker slots from current free GPU memory"
    )
    gpu_plan.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    gpu_plan.add_argument("--workers-per-gpu", default=None)
    gpu_plan.add_argument("--output", required=True)

    utilization = config_command(
        "gpu-util-report", "Summarize nvidia-smi samples from a launched phase"
    )
    utilization.add_argument("--csv", required=True)
    utilization.add_argument("--output-dir", required=True)
    utilization.add_argument("--plan", default=None)

    config_command("analyze", "Analyze completed paired outcomes")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(arguments.config)
    Path(cfg["data"]["output_root"]).mkdir(parents=True, exist_ok=True)

    if arguments.command == "preflight":
        result = run_preflight(cfg, arguments.expect_gpus)
        exit_code = 0 if result["ok"] else 2
    elif arguments.command == "collect":
        result = collect_task(cfg, arguments.task)
        exit_code = 0
    elif arguments.command == "collection-queue-init":
        result = initialize_collection_queue(
            cfg,
            reset_failed=arguments.reset_failed,
            reclaim_running=arguments.reclaim_running,
        )
        exit_code = 0
    elif arguments.command == "collect-dynamic-worker":
        result = run_collection_worker(cfg, arguments.worker_id)
        exit_code = 0
    elif arguments.command == "freeze-pairs":
        result = freeze_candidate_pairs(cfg, force=arguments.force_refreeze)
        exit_code = 0
    elif arguments.command == "manifest":
        result = build_manifest(cfg)
        exit_code = 0
    elif arguments.command == "run-worker":
        result = run_worker(cfg, arguments.rank, arguments.world_size)
        exit_code = 0
    elif arguments.command == "run-queue-init":
        result = initialize_pair_queue(
            cfg,
            reset_failed=arguments.reset_failed,
            reclaim_running=arguments.reclaim_running,
        )
        exit_code = 0
    elif arguments.command == "run-dynamic-worker":
        result = run_dynamic_worker(cfg, arguments.worker_id)
        exit_code = 0
    elif arguments.command == "queue-status":
        result = queue_summary(cfg, arguments.kind)
        counts = result["counts"]
        exit_code = 0 if not counts["failed"] else 3
    elif arguments.command == "gpu-plan":
        result = write_worker_plan(
            cfg,
            arguments.gpu_ids,
            arguments.workers_per_gpu,
            arguments.output,
        )
        exit_code = 0
    elif arguments.command == "gpu-util-report":
        result = summarize_gpu_metrics(
            arguments.csv, arguments.output_dir, arguments.plan
        )
        exit_code = 0
    elif arguments.command == "analyze":
        result = analyze(cfg)
        exit_code = 0
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
