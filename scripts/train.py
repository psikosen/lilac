#!/usr/bin/env python3
"""Train a small dense or fixed-E MoE model on text or isolated synthetic tasks.

Examples:
  python scripts/train.py --config configs/dense_baseline.yaml
  python scripts/train.py --config configs/moe_e8.yaml --total-tokens 10000000
  python scripts/train.py --config configs/dense_baseline.yaml --data-mode text \
      --text-path /path/to/corpus --run-dir runs/corpus-dense
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lilac.reference import add_reference_to_path, model_param_summary
from lilac.tasks import Batch, SyntheticTaskBatcher, TextBatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--data-mode", choices=("synthetic", "text"), default="synthetic")
    parser.add_argument("--text-path", action="append", default=[])
    parser.add_argument("--task-mix", default="induction,shift,add,copy")
    parser.add_argument("--tasks-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {value}, but CUDA is unavailable")
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(value: str, config_value: str, device: torch.device) -> torch.dtype:
    name = config_value if value == "auto" else value
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unknown dtype {name!r}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer | None, lr: float) -> None:
    if optimizer is not None:
        for group in optimizer.param_groups:
            group["lr"] = lr


def scheduled_lr(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    if warmup and step < warmup:
        return base_lr * max(step, 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def optimizer_for(params: list[torch.nn.Parameter], cfg: Any) -> torch.optim.Optimizer | None:
    if not params:
        return None
    if cfg.decay_norms:
        groups = [{"params": params, "weight_decay": cfg.weight_decay}]
    else:
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": cfg.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def prepare_moe(model: Any, device: torch.device) -> tuple[list[Any], list[torch.nn.Parameter]]:
    """Move the reference implementation's unregistered gradient banks to device."""
    layers = list(model.moe_layers())
    masters: list[torch.nn.Parameter] = []
    for layer in layers:
        for name in ("g_grad", "u_grad", "d_grad"):
            setattr(layer, name, getattr(layer, name).to(device))
        # The reference custom autograd function returns a CPU zero gradient
        # for this CPU-side anchor. Leave it where the reference creates it;
        # only the hand-scattered gradient banks need to follow the model.
        layer._anchor.requires_grad_(True)
        masters.extend(master for master, _ in layer.master_pairs())
    return layers, masters


def zero_moe_grads(layers: list[Any]) -> None:
    with torch.no_grad():
        for layer in layers:
            for grad in layer.grads():
                grad.zero_()


def copy_moe_masters(layers: list[Any]) -> None:
    with torch.no_grad():
        for layer in layers:
            for master, bank in layer.master_pairs():
                bank.copy_(master.to(dtype=bank.dtype, device=bank.device))


def sync_moe_masters_from_banks(layers: list[Any]) -> None:
    """Refresh optimizer masters after loading a checkpoint's model banks."""
    with torch.no_grad():
        for layer in layers:
            for master, bank in layer.master_pairs():
                master.copy_(bank.float())


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    moe_optimizer: torch.optim.Optimizer | None,
    step: int,
    tokens: int,
    seed: int,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "moe_optimizer": None if moe_optimizer is None else moe_optimizer.state_dict(),
        "step": step,
        "tokens": tokens,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    moe_optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, int, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if moe_optimizer is not None and payload.get("moe_optimizer") is not None:
        moe_optimizer.load_state_dict(payload["moe_optimizer"])
    return int(payload["step"]), int(payload["tokens"]), int(payload.get("seed", 0))


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir.expanduser().resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return (REPO_ROOT / "runs" / f"{args.config.stem}-{stamp}").resolve()


def make_batcher(args: argparse.Namespace, cfg: Any, reference_dir: Path,
                 device: torch.device, seed: int) -> Any:
    if args.data_mode == "text":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(reference_dir / "tokenizer.json"))
        return TextBatcher(args.text_path, tokenizer, device=device)
    return SyntheticTaskBatcher(
        vocab_size=cfg.model.vocab_size,
        sequence_length=cfg.data.sequence_length,
        device=device,
        seed=seed,
        task_mix=args.task_mix,
        tasks_per_sequence=args.tasks_per_sequence,
    )


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = (
        args.reference_dir
        or os.environ.get("LILAC_REFERENCE_DIR")
        or REPO_ROOT / ".cache/reference/amx-reasoning-v1-instruct"
    )
    add_reference_to_path(reference_dir)
    from m2r.config import load, to_dict

    cfg = load(args.config)
    if args.total_tokens is not None:
        cfg.train.total_tokens = args.total_tokens
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.ckpt_every is not None:
        cfg.train.ckpt_every = args.ckpt_every
    # The reference MoE module compiles its custom body at import time. Keep
    # the portable configs eager by default; users can opt into compilation in
    # the environment after validating a run.
    if not cfg.train.compile:
        os.environ.setdefault("M2R_MOE_COMPILE_BODY", "0")
    from m2r.model.torch_model import Model, swa_mask
    if args.batch_size is None and cfg.train.tokens_per_step % cfg.data.sequence_length:
        raise ValueError("train.tokens_per_step must be divisible by data.sequence_length")
    batch_size = args.batch_size or (cfg.train.tokens_per_step // cfg.data.sequence_length)
    tokens_per_step = batch_size * cfg.data.sequence_length

    dtype = choose_dtype(args.dtype, cfg.train.dtype, device)
    if dtype == torch.float16 and device.type == "cpu":
        raise ValueError("float16 training on CPU is unsupported; use float32 or bfloat16")
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    model = Model(cfg.model).to(device=device, dtype=dtype)
    moe_layers, moe_masters = prepare_moe(model, device)
    optimizer = optimizer_for(list(model.parameters()), cfg.optim)
    if optimizer is None:
        raise RuntimeError("model has no trainable parameters")
    moe_optimizer = optimizer_for(moe_masters, cfg.optim) if moe_masters else None
    mask = swa_mask(cfg.model, dtype=dtype, device=device)

    batcher: Any = make_batcher(args, cfg, Path(reference_dir), device, cfg.train.seed)
    val_batcher: Any = make_batcher(args, cfg, Path(reference_dir), device, cfg.train.seed + 1)

    source_config = run_dir / "config.source.yaml"
    shutil.copy2(args.config, source_config)
    metadata = {
        "config": to_dict(cfg),
        "reference_dir": str(Path(reference_dir).expanduser().resolve()),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "data_mode": args.data_mode,
        "batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
        "task_mix": args.task_mix if args.data_mode == "synthetic" else None,
        "tasks_per_sequence": args.tasks_per_sequence if args.data_mode == "synthetic" else None,
        **model_param_summary(cfg, model),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=list) + "\n")
    metrics_path = run_dir / "metrics.jsonl"
    total_steps = max(1, math.ceil(cfg.train.total_tokens / tokens_per_step))
    start_step, tokens_seen, resume_seed = 0, 0, cfg.train.seed
    if args.resume and not args.no_resume:
        start_step, tokens_seen, resume_seed = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            moe_optimizer=moe_optimizer,
            device=device,
        )
        sync_moe_masters_from_banks(moe_layers)
        print(f"resumed step={start_step} tokens={tokens_seen} seed={resume_seed}")
    if cfg.train.loss not in ("sampled", "full"):
        raise ValueError("this runner supports train.loss: sampled or full")
    if cfg.train.loss == "full" and moe_layers:
        raise ValueError("full loss with MoE is not supported by this portable runner")

    print(
        f"run={run_dir}\ndevice={device} dtype={dtype} batch={batch_size} "
        f"steps={total_steps} active={cfg.model.active_params():,} stored={cfg.model.stored_params():,}"
    )
    model.train()
    started = time.perf_counter()
    for step in range(start_step, total_steps):
        step_start = time.perf_counter()
        batch: Batch = (
            batcher.next(batch_size, cfg.data.sequence_length)
            if args.data_mode == "text"
            else batcher.next(batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        zero_moe_grads(moe_layers)
        if cfg.train.loss == "sampled":
            loss = model.sampled_loss(batch.idx, batch.tgt, mask, batch.pool, loss_mask=batch.loss_mask)
        else:
            loss = model.full_loss(batch.idx, batch.tgt, mask, loss_mask=batch.loss_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        lr = scheduled_lr(step + 1, total_steps, cfg.optim.lr, cfg.optim.warmup_steps, cfg.optim.min_lr_ratio)
        set_optimizer_lr(optimizer, lr)
        set_optimizer_lr(moe_optimizer, lr)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), cfg.optim.grad_clip)
        if moe_masters:
            torch.nn.utils.clip_grad_norm_(moe_masters, cfg.optim.grad_clip)
        optimizer.step()
        if moe_optimizer is not None:
            moe_optimizer.step()
            copy_moe_masters(moe_layers)
        step_tokens = batch_size * cfg.data.sequence_length
        tokens_seen += step_tokens
        elapsed = time.perf_counter() - step_start
        if (step + 1) % cfg.train.log_every == 0 or step == start_step:
            record = {
                "step": step + 1,
                "tokens": tokens_seen,
                "loss": float(loss.detach().float().cpu()),
                "lr": lr,
                "tokens_per_sec": step_tokens / max(elapsed, 1e-9),
                "elapsed_sec": time.perf_counter() - started,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step {step + 1:>7}/{total_steps} loss={record['loss']:.4f} "
                f"tok/s={record['tokens_per_sec']:.0f}"
            )
        if cfg.train.val_every and (step + 1) % cfg.train.val_every == 0:
            val_steps = max(1, math.ceil(cfg.train.val_tokens / tokens_per_step))
            model.eval()
            values = []
            with torch.inference_mode():
                for _ in range(val_steps):
                    val_batch: Batch = (
                        val_batcher.next(batch_size, cfg.data.sequence_length)
                        if args.data_mode == "text"
                        else val_batcher.next(batch_size)
                    )
                    if cfg.train.loss == "sampled":
                        val = model.sampled_loss(
                            val_batch.idx, val_batch.tgt, mask, val_batch.pool,
                            loss_mask=val_batch.loss_mask,
                        )
                    else:
                        val = model.full_loss(
                            val_batch.idx, val_batch.tgt, mask,
                            loss_mask=val_batch.loss_mask,
                        )
                    values.append(float(val.float().cpu()))
            model.train()
            val_record = {
                "step": step + 1,
                "tokens": tokens_seen,
                "val_loss": sum(values) / len(values),
                "val_steps": val_steps,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(val_record) + "\n")
            print(f"validation step {step + 1:>7} loss={val_record['val_loss']:.4f}")
        if (step + 1) % cfg.train.ckpt_every == 0 or step + 1 == total_steps:
            save_checkpoint(
                run_dir / f"checkpoint_step{step + 1:07d}.pt",
                model=model,
                optimizer=optimizer,
                moe_optimizer=moe_optimizer,
                step=step + 1,
                tokens=tokens_seen,
                seed=resume_seed,
            )

    print(f"finished: {run_dir}")


if __name__ == "__main__":
    main()
