#!/usr/bin/env python3
"""Partition QA data into teacher-correct and teacher-failed DEX splits.

The split uses containment exact match, matching the CEM-style correctness
criterion used by the DEX-Comp paper. The original row is preserved and
teacher metadata is appended, so the files remain auditable.

Example:
  python scripts/build_teacher_splits.py \
    --reference-dir .cache/reference/amx-reasoning-v1-instruct \
    --input data/qa_smoke.jsonl \
    --output-dir data/dex_smoke --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lilac.qa import best_f1, containment_match
from lilac.qa import greedy_answer
from lilac.reference import load_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    weights = args.weights or args.reference_dir / "model.safetensors"
    stack = load_stack(
        args.reference_dir,
        device=torch.device(args.device),
        dtype=getattr(torch, args.dtype),
        weights=weights,
    )
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    correct_path = args.output_dir / "teacher_correct.jsonl"
    failed_path = args.output_dir / "teacher_failed.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct = failed = 0
    total_f1 = 0.0
    with correct_path.open("w") as correct_file, failed_path.open("w") as failed_file:
        for index, row in enumerate(rows):
            question = row.get("question") or row.get("prompt")
            passage = row.get("passage") or row.get("context") or ""
            answers = row.get("answers") or row.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            if not question or not answers:
                raise ValueError(f"row {index} needs question and answers")
            prediction = greedy_answer(
                stack, question, passage, max_new_tokens=args.max_new_tokens
            )
            is_correct = bool(containment_match(prediction, answers))
            record = dict(row)
            record["teacher_prediction"] = prediction
            record["teacher_correct"] = is_correct
            record["teacher_f1"] = best_f1(prediction, answers)
            (correct_file if is_correct else failed_file).write(json.dumps(record) + "\n")
            correct += int(is_correct)
            failed += int(not is_correct)
            total_f1 += record["teacher_f1"]
            print(json.dumps({"index": index, "correct": is_correct, "prediction": prediction}))
    manifest = {
        "input": str(args.input.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "weights": str(weights.resolve()),
        "count": correct + failed,
        "teacher_correct": correct,
        "teacher_failed": failed,
        "teacher_cem": correct / max(1, correct + failed),
        "teacher_f1": total_f1 / max(1, correct + failed),
        "criterion": "normalized containment exact match",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
