#!/usr/bin/env python3
"""Collect routing, hidden-state, and logit diagnostics for a run checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lilac.reference import load_stack
from lilac.tasks import SyntheticTaskBatcher
from scripts.train import prepare_moe


def participation_ratio(values: torch.Tensor) -> float:
    values = values.float().flatten()
    denom = values.square().sum()
    return float(values.sum().square() / denom.clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--route-mode", choices=("configured", "legacy"), default="configured")
    parser.add_argument("--moe-act", choices=("configured", "relu2", "silu"), default="configured")
    parser.add_argument("--ablate", choices=("none", "random", "hot", "shuffle", "ugate"), default="none")
    parser.add_argument("--fixed-experts", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("M2R_MOE_COMPILE_BODY", "0")
    if args.route_mode == "legacy":
        os.environ["M2R_MOE_ROUTE"] = "legacy"
    if args.moe_act != "configured":
        os.environ["M2R_MOE_ACT"] = args.moe_act
    if args.fixed_experts:
        os.environ["M2R_MOE_FIXED"] = "1"
    if args.ablate != "none":
        os.environ["M2R_MOE_ABLATE"] = args.ablate

    run = args.run_dir.resolve()
    metadata = json.loads((run / "run.json").read_text())
    reference_dir = args.reference_dir or metadata["reference_dir"]
    checkpoint = args.checkpoint
    if checkpoint is None:
        choices = sorted(run.glob("checkpoint_step*.pt"))
        if not choices:
            raise FileNotFoundError(f"no checkpoint_step*.pt found in {run}")
        checkpoint = choices[-1]
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    stack = load_stack(
        reference_dir,
        device=device,
        dtype=dtype,
        weights=None,
        config_path=run / "config.source.yaml",
    )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    stack.model.load_state_dict(payload["model"], strict=True)
    moe_layers, _ = prepare_moe(stack.model, device)
    stack.model.eval()
    batcher = SyntheticTaskBatcher(
        vocab_size=stack.cfg.model.vocab_size,
        sequence_length=stack.cfg.data.sequence_length,
        device=device,
        seed=args.seed,
        tasks_per_sequence=8,
    )
    batch = batcher.next(1)
    with torch.inference_mode():
        hidden_all = stack.model.body(batch.idx, stack.mask)
        hidden = hidden_all.reshape(-1, stack.cfg.model.d_model)
        logits = stack.model.tied_logits(hidden[-1:], chunk=1).float()[0]
    result = {
        "checkpoint": str(checkpoint),
        "step": int(payload.get("step", -1)),
        "tokens": int(payload.get("tokens", -1)),
        "hidden_rms": float(hidden.float().square().mean().sqrt().cpu()),
        "hidden_participation_ratio": participation_ratio(hidden.float().square().mean(0).cpu()),
        "logit_mean": float(logits.mean().cpu()),
        "logit_std": float(logits.std().cpu()),
        "top_token_ids": [int(x) for x in torch.topk(logits, k=10).indices.cpu()],
        "expert_layers": [],
    }
    for index, layer in enumerate(moe_layers):
        touched = layer.touched.detach().cpu()
        counts = torch.bincount(touched, minlength=layer.E)[: layer.E].float()
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).cpu())
        result["expert_layers"].append({
            "layer": index,
            "experts": int(layer.E),
            "touched": int(touched.numel()),
            "active_experts": int((counts > 0).sum()),
            "participation_ratio": participation_ratio(counts),
            "normalized_entropy": entropy / max(1e-12, float(torch.log(torch.tensor(layer.E)))),
            "counts": [int(x) for x in counts],
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
