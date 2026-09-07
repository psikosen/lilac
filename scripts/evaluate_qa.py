#!/usr/bin/env python3
"""Evaluate the released instruct checkpoint on a tiny JSONL QA file.

Each line should contain ``question`` and ``passage`` plus ``answers`` as a
list of acceptable strings. ``answerable`` is optional and enables abstention
reporting when present.

Example:
  python scripts/evaluate_qa.py --reference-dir .cache/reference/amx-reasoning-v1-instruct \
      --jsonl data/qa.jsonl --max-new-tokens 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lilac.qa import best_f1, exact_match, greedy_answer
from lilac.reference import load_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    weights = args.weights or args.reference_dir / "model.safetensors"
    dtype = getattr(torch, args.dtype)
    stack = load_stack(args.reference_dir, device=torch.device(args.device), dtype=dtype, weights=weights)

    rows = []
    if args.jsonl:
        for line in args.jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        rows = [{
            "question": "In what year was the Royal Dutch Petroleum Company founded?",
            "passage": (
                "In February 1907, the Royal Dutch Shell Group was created through the "
                "amalgamation of two rival companies: the Royal Dutch Petroleum Company, "
                "founded in 1890, and the Shell Transport and Trading Company of the United Kingdom."
            ),
            "answers": ["1890"],
        }]
    if args.limit:
        rows = rows[: args.limit]
    scores = []
    for index, row in enumerate(rows):
        question = row.get("question") or row.get("prompt")
        passage = row.get("passage") or row.get("context") or ""
        answers = row.get("answers") or row.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        if not question or not answers:
            raise ValueError(f"row {index} needs question and answers")
        prediction = greedy_answer(stack, question, passage, max_new_tokens=args.max_new_tokens)
        em = exact_match(prediction, answers)
        f1 = max((best_f1(prediction, [answer]) for answer in answers), default=0.0)
        scores.append((em, f1))
        print(json.dumps({"index": index, "prediction": prediction, "em": em, "f1": f1}))
    mean_em = sum(x[0] for x in scores) / max(len(scores), 1)
    mean_f1 = sum(x[1] for x in scores) / max(len(scores), 1)
    print(json.dumps({"count": len(scores), "exact_match": mean_em, "f1": mean_f1}))


if __name__ == "__main__":
    main()
