"""Exact prompt and greedy decoding helpers for the released checkpoint."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def greedy_answer(stack: Any, question: str, passage: str, *, max_new_tokens: int = 32) -> str:
    import torch
    from m2r.data.templates import render_prompt

    prompt = render_prompt([f"{question}\n\n{passage}"], thinking=False)
    ids = stack.tokenizer.encode(prompt, add_special_tokens=False).ids
    pad_to = max(stack.cfg.model.window, stack.cfg.model.route_block or 1, 256)
    output: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            n = len(ids)
            row = torch.tensor([ids + [0] * ((-n) % pad_to)], device=stack.mask.device)
            hidden = stack.model.body(row, stack.mask)[:, n - 1]
            logits = (hidden @ stack.model.emb.t().to(hidden.dtype)).float()[0]
            if stack.banned_ids.numel():
                logits[stack.banned_ids] = -1e30
            token = int(logits.argmax())
            if token == stack.eot_id:
                break
            output.append(token)
            ids.append(token)
    return stack.tokenizer.decode(output).replace("<think></think>", "").strip()


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, answers: list[str]) -> float:
    return max((token_f1(prediction, answer) for answer in answers), default=0.0)
