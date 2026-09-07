#!/usr/bin/env python3
"""Run the isolated synthetic capability probes on a Lilac training checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lilac.reference import load_stack
from lilac.tasks import padded_ids, synthetic_probe_examples


def latest_checkpoint(run_dir: Path) -> Path:
    choices = sorted(run_dir.glob("checkpoint_step*.pt"))
    if not choices:
        raise FileNotFoundError(f"no checkpoint_step*.pt found in {run_dir}")
    return choices[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    os.environ.setdefault("M2R_MOE_COMPILE_BODY", "0")
    run = args.run_dir.resolve()
    metadata = json.loads((run / "run.json").read_text())
    reference_dir = args.reference_dir or metadata["reference_dir"]
    config_path = run / "config.source.yaml"
    checkpoint = args.checkpoint or latest_checkpoint(run)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    stack = load_stack(
        reference_dir,
        device=device,
        dtype=dtype,
        weights=None,
        config_path=config_path,
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    stack.model.load_state_dict(payload["model"], strict=True)
    stack.model.eval()
    pad_to = max(stack.cfg.model.window, stack.cfg.model.route_block or 1, 256)
    results = []
    with torch.inference_mode():
        for task, ids, target in synthetic_probe_examples(stack.cfg.model.vocab_size):
            row, n = padded_ids(ids, pad_to)
            row = row.to(device)
            hidden = stack.model.body(row, stack.mask)[:, n - 1]
            logits = (hidden @ stack.model.emb.t().to(hidden.dtype)).float()[0]
            if stack.banned_ids.numel():
                logits[stack.banned_ids] = -1e30
            values, indices = torch.topk(logits, k=args.top_k)
            top = [int(x) for x in indices.cpu()]
            rank = int((logits > logits[target]).sum().item()) + 1
            record = {
                "task": task,
                "target": target,
                "top": top,
                "target_rank": rank,
                "hit": int(top[0] == target),
            }
            results.append(record)
            print(json.dumps(record))
    print(json.dumps({"count": len(results), "top1": sum(r["hit"] for r in results) / len(results)}))


if __name__ == "__main__":
    main()
