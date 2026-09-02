from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .analysis import analyze
from .collector import collect_task
from .config import load_config
from .manifest import build_manifest
from .preflight import run_preflight
from .runner import run_worker


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

    collect = config_command("collect", "Collect one task's locked causal pair pack")
    collect.add_argument("--task", required=True)

    config_command("manifest", "Freeze the pair × condition manifest")

    run = config_command("run-worker", "Run one deterministic GPU worker")
    run.add_argument("--rank", type=int, default=int(os.environ.get("PF_RANK", 0)))
    run.add_argument("--world-size", type=int, default=int(os.environ.get("PF_WORLD_SIZE", 1)))

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
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 2
    if arguments.command == "collect":
        result = collect_task(cfg, arguments.task)
    elif arguments.command == "manifest":
        result = build_manifest(cfg)
    elif arguments.command == "run-worker":
        result = run_worker(cfg, arguments.rank, arguments.world_size)
    elif arguments.command == "analyze":
        result = analyze(cfg)
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
