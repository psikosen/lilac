#!/usr/bin/env python3
"""Run a small dense/MoE sweep with one command.

This intentionally launches subprocesses so each experiment gets an isolated
process, memory profile, metrics file, and checkpoint directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[
        "configs/dense_baseline.yaml", "configs/moe_e4.yaml", "configs/moe_e8.yaml", "configs/moe_e16.yaml"
    ])
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/sweep")
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--data-mode", choices=("synthetic", "text"), default="synthetic")
    parser.add_argument("--text-path", action="append", default=[])
    parser.add_argument("--task-mix", default="induction,shift,add,copy")
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    for config_arg in args.configs:
        config = Path(config_arg)
        if not config.is_absolute():
            config = ROOT / config
        run_dir = args.run_root / config.stem
        command = [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            "--config", str(config),
            "--run-dir", str(run_dir),
            "--device", args.device,
            "--dtype", args.dtype,
            "--data-mode", args.data_mode,
            "--task-mix", args.task_mix,
        ]
        if args.reference_dir:
            command += ["--reference-dir", str(args.reference_dir)]
        if args.total_tokens is not None:
            command += ["--total-tokens", str(args.total_tokens)]
        for path in args.text_path:
            command += ["--text-path", path]
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
