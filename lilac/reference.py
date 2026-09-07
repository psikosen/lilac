"""Load the pinned AMX reasoning reference implementation from a local cache.

The reference repository is intentionally fetched from the Hugging Face model
repository instead of copied into Lilac. That keeps this repo small while
ensuring that every run records the exact source revision it used.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DIR = Path(".cache/reference/amx-reasoning-v1-instruct")


@dataclass
class ReferenceStack:
    cfg: Any
    model: Any
    tokenizer: Any
    mask: Any
    eot_id: int
    banned_ids: Any
    generation: dict


def add_reference_to_path(reference_dir: str | Path) -> Path:
    """Make the downloaded ``m2r`` package importable and return its path."""
    path = Path(reference_dir).expanduser().resolve()
    if not (path / "m2r").is_dir():
        raise FileNotFoundError(
            f"reference source not found at {path}; run scripts/fetch_reference.sh"
        )
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path


def load_stack(
    reference_dir: str | Path,
    *,
    device: Any,
    dtype: Any,
    weights: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ReferenceStack:
    """Load the reference model, tokenizer, generation mask, and attention mask."""
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    path = add_reference_to_path(reference_dir)
    from m2r.config import load
    from m2r.data.templates import EOT
    from m2r.model.torch_model import Model, swa_mask

    cfg_path = Path(config_path) if config_path else path / "training_config.yaml"
    cfg = load(cfg_path)
    model = Model(cfg.model).to(device=device, dtype=dtype)
    if weights is not None:
        weight_path = Path(weights)
        if not weight_path.is_file():
            raise FileNotFoundError(f"weights not found at {weight_path}")
        state = load_file(str(weight_path), device=str(device))
        model.load_state_dict(state, strict=True)
    model.eval()

    tokenizer_path = path / "tokenizer.json"
    generation_path = path / "generation.json"
    if not tokenizer_path.is_file() or not generation_path.is_file():
        raise FileNotFoundError(
            f"reference cache at {path} is incomplete; fetch tokenizer.json and generation.json"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    generation = json.loads(generation_path.read_text())
    eot_tokens = tokenizer.encode(EOT, add_special_tokens=False).ids
    if len(eot_tokens) != 1:
        raise ValueError(f"expected EOT to be one token, got {eot_tokens}")
    banned = torch.tensor(
        generation.get("banned_token_ids", []), dtype=torch.long, device=device
    )
    mask = swa_mask(cfg.model, dtype=dtype, device=device)
    return ReferenceStack(cfg, model, tokenizer, mask, eot_tokens[0], banned, generation)


def model_param_summary(cfg: Any, model: Any | None = None) -> dict[str, int]:
    """Return active parameters plus an exact stored count when a model exists."""
    if model is None:
        stored = int(cfg.model.stored_params())
    else:
        # Count the serialized model, including MoE banks and frozen buffers,
        # instead of relying on a config-side estimate. This is the number
        # that should agree with a checkpoint and its memory footprint.
        stored = sum(int(t.numel()) for t in model.state_dict().values())
    return {"active_params": int(cfg.model.active_params()), "stored_params": stored}
