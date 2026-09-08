#!/usr/bin/env python3
"""DEX-style hard exploration on teacher-failed QA rows.

This is a compact, dependency-light GRPO-style loop for Lilac. It starts from
a pure-distillation checkpoint, samples a group of answers for each failed QA
row, assigns containment rewards, standardizes rewards within each group, and
updates the student with a policy-gradient objective. The optional KL term is a
sampled-token approximation to the reference-policy regularizer; it is logged
explicitly rather than presented as a full-vocabulary KL.

Example:
  python scripts/hard_explore.py \
    --run-dir runs/dex-pd \
    --failed-jsonl data/dex_smoke/teacher_failed.jsonl \
    --steps 100 --group-size 4
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lilac.qa import containment_match
from lilac.reference import load_stack
from scripts.train import (
    copy_moe_masters,
    optimizer_for,
    prepare_moe,
    save_checkpoint,
    zero_moe_grads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--failed-jsonl", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def latest_checkpoint(run_dir: Path) -> Path:
    choices = sorted(run_dir.glob("checkpoint_step*.pt"))
    if not choices:
        raise FileNotFoundError(f"no checkpoint_step*.pt found in {run_dir}")
    return choices[-1]


def prompt_ids(stack: Any, row: dict) -> list[int]:
    from m2r.data.templates import render_prompt

    question = row.get("question") or row.get("prompt")
    passage = row.get("passage") or row.get("context") or ""
    prompt = render_prompt([f"{question}\n\n{passage}"], thinking=False)
    return stack.tokenizer.encode(prompt, add_special_tokens=False).ids


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    values = logits.float() / temperature
    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        cumulative = torch.softmax(sorted_values, -1).cumsum(-1)
        remove = cumulative > top_p
        remove[0] = False
        values[sorted_indices[remove]] = -float("inf")
    return int(torch.multinomial(torch.softmax(values, -1), 1))


def generate(
    stack: Any,
    ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    pad_to = max(stack.cfg.model.window, stack.cfg.model.route_block or 1, 256)
    output: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            n = len(ids)
            row = torch.tensor(
                [ids + [0] * ((-n) % pad_to)],
                device=stack.mask.device,
            )
            hidden = stack.model.body(row, stack.mask)[:, n - 1]
            if hasattr(stack.model, "_take_aux"):
                stack.model._take_aux()
            logits = (hidden @ stack.model.emb.t().to(hidden.dtype)).float()[0]
            if stack.banned_ids.numel():
                logits[stack.banned_ids] = -1e30
            token = sample_token(logits, temperature, top_p)
            output.append(token)
            ids.append(token)
            if token == stack.eot_id:
                break
    return output


def sequence_logprob(stack: Any, prompt: list[int], tokens: list[int]) -> torch.Tensor:
    """Mean log probability of sampled tokens, retaining student gradients."""
    if not tokens:
        return torch.zeros((), device=stack.mask.device, requires_grad=True)
    pad_to = max(stack.cfg.model.window, stack.cfg.model.route_block or 1, 256)
    context = prompt + tokens[:-1]
    n = len(context)
    row = torch.tensor(
        [context + [0] * ((-n) % pad_to)],
        device=stack.mask.device,
    )
    hidden = stack.model.body(row, stack.mask)[0]
    if hasattr(stack.model, "_take_aux"):
        stack.model._take_aux()
    positions = torch.arange(
        len(prompt) - 1,
        len(prompt) - 1 + len(tokens),
        device=stack.mask.device,
    )
    logits = stack.model.tied_logits(hidden[positions], chunk=512).float()
    target = torch.tensor(tokens, dtype=torch.long, device=stack.mask.device)
    return F.log_softmax(logits, dim=-1).gather(1, target[:, None])[:, 0].mean()


def reward(row: dict, prediction: str) -> float:
    answers = row.get("answers") or row.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    return float(containment_match(prediction, answers))


def decode(stack: Any, tokens: list[int]) -> str:
    return stack.tokenizer.decode(tokens).replace("<think></think>", "").strip()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.group_size < 2 or args.batch_size < 1:
        raise ValueError("steps >= 1, group-size >= 2, and batch-size >= 1 are required")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    os.environ.setdefault("M2R_MOE_COMPILE_BODY", "0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run = args.run_dir.resolve()
    metadata = json.loads((run / "run.json").read_text())
    reference_dir = Path(args.reference_dir or metadata["reference_dir"])
    checkpoint = args.checkpoint or latest_checkpoint(run)
    output_dir = (args.output_dir or run / "hard-explore").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "config.source.yaml", output_dir / "config.source.yaml")
    rows = [json.loads(line) for line in args.failed_jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("failed-jsonl contains no rows")
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
    stack.model.eval()
    # This is the frozen PD policy used for the sampled-token KL anchor.
    reference = load_stack(
        reference_dir,
        device=device,
        dtype=dtype,
        weights=None,
        config_path=run / "config.source.yaml",
    )
    reference.model.load_state_dict(payload["model"], strict=True)
    reference.model.eval()
    moe_layers, masters = prepare_moe(stack.model, device)
    stack.cfg.optim.lr = args.learning_rate
    optimizer = optimizer_for(list(stack.model.parameters()), stack.cfg.optim)
    moe_optimizer = optimizer_for(masters, stack.cfg.optim) if masters else None
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    tokens_seen = 0
    print(
        f"run={output_dir}\ncheckpoint={checkpoint}\nrows={len(rows)} "
        f"group={args.group_size} steps={args.steps}"
    )
    for step in range(1, args.steps + 1):
        selected = [rows[(step * args.batch_size + i) % len(rows)] for i in range(args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        zero_moe_grads(moe_layers)
        losses = []
        policy_losses = []
        kl_losses = []
        rewards_seen = []
        generated = 0
        for row in selected:
            prompt = prompt_ids(stack, row)
            group_logp = []
            group_ref_logp = []
            group_rewards = []
            for _ in range(args.group_size):
                sampled = generate(
                    stack, prompt.copy(), max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
                prediction = decode(stack, sampled)
                group_rewards.append(reward(row, prediction))
                generated += len(sampled)
                group_logp.append(sequence_logprob(stack, prompt, sampled))
                with torch.inference_mode():
                    group_ref_logp.append(float(sequence_logprob(reference, prompt, sampled).cpu()))
            rewards = torch.tensor(group_rewards, dtype=torch.float32, device=device)
            advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
            logp = torch.stack(group_logp)
            ref_logp = torch.tensor(group_ref_logp, dtype=logp.dtype, device=device)
            policy = -(advantages.detach() * logp).mean()
            kl = (logp - ref_logp.detach()).square().mean()
            losses.append(policy + args.kl_coef * kl)
            policy_losses.append(policy.detach())
            kl_losses.append(kl.detach())
            rewards_seen.extend(group_rewards)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite hard-exploration loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(stack.model.parameters()), stack.cfg.optim.grad_clip)
        if masters:
            torch.nn.utils.clip_grad_norm_(masters, stack.cfg.optim.grad_clip)
        optimizer.step()
        if moe_optimizer is not None:
            moe_optimizer.step()
            copy_moe_masters(moe_layers)
        tokens_seen += generated
        record = {
            "step": step,
            "tokens": tokens_seen,
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(torch.stack(policy_losses).mean().cpu()),
            "sampled_token_kl": float(torch.stack(kl_losses).mean().cpu()),
            "reward_mean": sum(rewards_seen) / max(1, len(rewards_seen)),
            "reward_successes": int(sum(rewards_seen)),
            "group_rollouts": len(rewards_seen),
            "tokens_per_sec": generated / max(1e-9, time.perf_counter() - started),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == 1 or step % 10 == 0:
            print(
                f"step {step:>5}/{args.steps} loss={record['loss']:.4f} "
                f"reward={record['reward_mean']:.3f} tok/s={record['tokens_per_sec']:.0f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                output_dir / f"checkpoint_he_step{step:07d}.pt",
                model=stack.model,
                optimizer=optimizer,
                moe_optimizer=moe_optimizer,
                step=step,
                tokens=tokens_seen,
                seed=args.seed,
            )
    print(f"finished: {output_dir}")


if __name__ == "__main__":
    main()
