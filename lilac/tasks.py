"""Deterministic synthetic tasks and simple text-stream batches.

The synthetic stream is deliberately small and inspectable. Each example ends
with a query token whose next-token target is the answer. It is not intended to
be a benchmark; it gives architecture and optimizer changes a cheap causal
test before a large corpus run.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MARKERS = {
    "induction": 8,
    "shift": 11,
    "add": 14,
    "copy": 18,
}


@dataclass
class Batch:
    idx: Any
    tgt: Any
    loss_mask: Any
    pool: Any


def parse_mix(value: str) -> list[str]:
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    allowed = set(MARKERS)
    unknown = sorted(set(names) - allowed)
    if not names or unknown:
        raise ValueError(f"task mix must use {sorted(allowed)}; got {unknown or value!r}")
    return names


class SyntheticTaskBatcher:
    """Generate fixed-shape batches with multiple answer queries per sequence."""

    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        device: Any,
        seed: int = 0,
        task_mix: str = "induction,shift,add,copy",
        tasks_per_sequence: int = 8,
    ) -> None:
        import torch

        if sequence_length < 64:
            raise ValueError("sequence_length must be at least 64")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.device = device
        self.tasks = parse_mix(task_mix)
        self.tasks_per_sequence = tasks_per_sequence
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator(device="cpu").manual_seed(seed)

    def _task_block(self, task: str) -> tuple[list[int], int]:
        key = self.rng.randrange(32, 288)
        value = 512 + (key - 32)
        if task == "induction":
            # marker, key, value, separator, query-marker, key -> value
            return [MARKERS[task], key, value, 10, 9, key], value
        if task == "shift":
            # A positional-style relation with a separate query marker.
            return [MARKERS[task], key, value, 12, 13, key], value
        if task == "copy":
            return [MARKERS[task], key, 19, 20, 21, key], key
        if task == "add":
            a, b = self.rng.randrange(10), self.rng.randrange(10)
            answer = 768 + a + b
            return [MARKERS[task], 32 + a, 48 + b, 15, 16, 17], answer
        raise AssertionError(task)

    def next(self, batch_size: int) -> Batch:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        low, high = 24, min(self.vocab_size, 4096)
        idx = torch.randint(
            low, high, (batch_size, self.sequence_length), generator=self.torch_rng
        )
        tgt = torch.roll(idx, shifts=-1, dims=1)
        loss_mask = torch.zeros_like(idx, dtype=torch.float32)
        for row in range(batch_size):
            cursor = self.sequence_length - self.tasks_per_sequence * 6
            if cursor < 0:
                raise ValueError("tasks_per_sequence does not fit sequence_length")
            for offset in range(self.tasks_per_sequence):
                task = self.tasks[(row + offset) % len(self.tasks)]
                block, answer = self._task_block(task)
                start = cursor + offset * 6
                idx[row, start : start + 6] = torch.tensor(block, dtype=torch.long)
                tgt[row, start + 5] = answer
                loss_mask[row, start + 5] = 1.0
        pool_size = min(self.vocab_size, 4096)
        pool = torch.arange(pool_size, dtype=torch.long)
        return Batch(
            idx.to(self.device, non_blocking=True),
            tgt.to(self.device, non_blocking=True),
            loss_mask.to(self.device, non_blocking=True),
            pool.to(self.device, non_blocking=True),
        )


def _extract_text(path: Path) -> str:
    if path.suffix.lower() != ".jsonl":
        return path.read_text(errors="replace")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, str):
            rows.append(row)
        elif isinstance(row, dict):
            for key in ("text", "content", "document", "input", "question", "prompt"):
                if isinstance(row.get(key), str) and row[key].strip():
                    rows.append(row[key])
                    break
    return "\n".join(rows)


class TextBatcher:
    """Cycle a tokenized text corpus into next-token prediction batches."""

    def __init__(self, paths: Iterable[str | Path], tokenizer: Any, *, device: Any) -> None:
        import torch

        files: list[Path] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".jsonl"}))
            elif path.is_file():
                files.append(path)
            else:
                raise FileNotFoundError(path)
        if not files:
            raise ValueError("no .txt or .jsonl files found")
        ids: list[int] = []
        for path in files:
            ids.extend(tokenizer.encode(_extract_text(path), add_special_tokens=False).ids)
        if len(ids) < 2:
            raise ValueError("text corpus produced fewer than two tokens")
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.device = device
        self.cursor = 0

    def next(self, batch_size: int, sequence_length: int) -> Batch:
        import torch

        total = batch_size * sequence_length
        values = []
        for _ in range(total):
            values.append(int(self.tokens[self.cursor]))
            self.cursor = (self.cursor + 1) % self.tokens.numel()
        idx = torch.tensor(values, dtype=torch.long).reshape(batch_size, sequence_length)
        tgt = torch.roll(idx, shifts=-1, dims=1)
        mask = torch.ones_like(idx, dtype=torch.float32)
        pool = torch.arange(min(4096, int(self.tokens.max()) + 1), dtype=torch.long)
        return Batch(
            idx.to(self.device), tgt.to(self.device), mask.to(self.device), pool.to(self.device)
        )


def padded_ids(ids: list[int], pad_to: int) -> tuple[Any, int]:
    """Return one row of IDs padded for the reference model's block layout."""
    import torch

    n = len(ids)
    return torch.tensor([ids + [0] * ((-n) % pad_to)], dtype=torch.long), n


def synthetic_probe_examples(vocab_size: int) -> list[tuple[str, list[int], int]]:
    examples = []
    for task in ("induction", "shift", "copy", "add"):
        if task == "add":
            ids, target = [MARKERS[task], 35, 49, 15, 16, 17], 772
        elif task == "copy":
            ids, target = [MARKERS[task], 77, 19, 20, 21, 77], 77
        else:
            ids, target = [MARKERS[task], 77, 557, 10 if task == "induction" else 12,
                           9 if task == "induction" else 13, 77], 557
        if target >= vocab_size:
            raise ValueError("synthetic probe target exceeds vocabulary")
        examples.append((task, ids, target))
    return examples
