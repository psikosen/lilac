# Lilac

Small, isolated experiments for efficient reasoning models, starting from Greg Diamos's single-core AMX work.

The first research target is a tiny hybrid language model that can be measured honestly: capability probes, generated outputs, failure diagnostics, tokens per second, active parameters, and wall-clock cost all matter alongside loss.

## Starting point

The reference checkpoint is [`gdiamos/amx-reasoning-v1-instruct`](https://huggingface.co/gdiamos/amx-reasoning-v1-instruct).

Important distinction: the released checkpoint is not the paper's `E=128` MoE arm. It is a 3.315M-active-parameter, 7.49M-stored-parameter hybrid model with six layers:

```text
linear attention, sliding-window attention, sliding-window attention,
linear attention, sliding-window attention, linear attention
```

The paper's headline MoE result is a separate experimental arm: 3.65M active parameters, 128 experts, and block routing. Lilac will keep the dense hybrid baseline and MoE variants separate so results remain attributable.

## MLX compatibility

The idea is portable to [MLX](https://github.com/ml-explore/mlx) on Apple silicon, but the released checkpoint is not MLX-compatible out of the box.

Why:

- the checkpoint ships custom PyTorch `m2r` code rather than a standard Transformers architecture;
- its linear-attention state, sliding-window attention, document isolation, and decode loop need MLX implementations;
- its `safetensors` weights need a deliberate MLX conversion and numerical parity check;
- the Intel AMX single-core throughput numbers do not transfer directly to Apple silicon.

The safe path is:

1. reproduce the reference model with its supplied runner;
2. port one layer and compare PyTorch/MLX logits on identical inputs;
3. port the full dense baseline;
4. add MoE routing or recurrent state only after parity passes;
5. benchmark Apple silicon separately from the AMX reference.

MLX already provides examples for custom language models, MoE models, LoRA, and model conversion. Lilac should use those patterns while retaining the reference model's exact behavior as the test oracle.

## First experiment ladder

### 1. Measurement baseline

Run the reference checkpoint with its exact prompt template, greedy decoding, EOT stopping, and vocabulary mask. Measure:

- induction;
- positional shift;
- two-digit addition;
- extractive QA;
- abstention;
- junk-token frequency with and without the vocabulary mask.

### 2. Small MoE sweep

Keep the backbone fixed and test `E=4, 8, 16` before attempting a large expert bank. Use block routing over 256 tokens and compare windowed versus prefix routing. Log expert utilization, RMSNorm participation ratio, routing entropy, permutation ablations, validation loss, capability accuracy, tokens/sec, and wall-clock time.

### 3. Latent working state

Add one small persistent recurrent state and reuse a shared block for 2, 4, or 8 refinement steps. Compare it with the dense baseline on copying, variable-digit addition, and long-context retrieval. Freeze or ablate the state to test whether it is causally useful rather than merely decodable.

### 4. Data and post-training

Hold the architecture fixed while comparing raw data, curated data, generated reasoning traces, and verifiable synthetic tasks. Start with supervised or self-distilled data before attempting RL. Keep arithmetic, abstention, and retrieval as separate metrics; removing junk logits does not create arithmetic ability.

## Research principles

- Compare equal tokens and equal wall-clock budgets.
- Treat generated outputs as first-class evidence next to benchmark scores.
- Check expert collapse and dead vocabulary rows explicitly.
- Use held-out procedural variants to detect memorization and leakage.
- Change one architectural or data variable per experiment.

## Status

This repository currently contains the research plan. The next implementation milestone is a reference runner plus a PyTorch/MLX logit-parity test for the dense baseline.
