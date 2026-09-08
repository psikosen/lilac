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

## Scope

This first implementation is PyTorch-only. MLX conversion and parity are
intentionally deferred so the experiment variables can be tested on the other
computer without mixing a framework port into the results.

The reference implementation is fetched at a pinned Hugging Face revision by
`scripts/fetch_reference.sh`; Lilac does not vendor the custom `m2r` source or
the checkpoint weights.

## Quick start

From the repository root:

```bash
./scripts/setup.sh
PATH=.venv/bin:$PATH ./scripts/fetch_reference.sh --weights
```

Run the released checkpoint's exact QA smoke test:

```bash
python scripts/evaluate_qa.py \
  --reference-dir .cache/reference/amx-reasoning-v1-instruct \
  --jsonl data/qa_smoke.jsonl
```

Train a small synthetic capability model from scratch:

```bash
python scripts/train.py --config configs/dense_baseline.yaml \
  --run-dir runs/dense-baseline
python scripts/probe_ladder.py --run-dir runs/dense-baseline \
  --reference-dir .cache/reference/amx-reasoning-v1-instruct
```

The default synthetic run is deliberately short. Scale it with
`--total-tokens`, or point the trainer at `.txt`/`.jsonl` files with
`--data-mode text --text-path /path/to/corpus`.

Run the dense and E=4/8/16 comparison as isolated processes:

```bash
python scripts/run_sweep.py --total-tokens 10000000
```

Each run writes `run.json`, append-only `metrics.jsonl`, and resumable
`checkpoint_step*.pt` files. `scripts/diagnostics.py` reports hidden-state
participation, logit spread, and per-layer expert utilization. Routing ablations
are process-level controls, for example:

```bash
python scripts/diagnostics.py --run-dir runs/sweep/moe_e8 --route-mode legacy
python scripts/diagnostics.py --run-dir runs/sweep/moe_e8 --fixed-experts
```

For a corpus run, use the same tokenizer that ships with the reference cache;
the text loader is a transparent next-token stream, while the synthetic loader
is the controlled capability probe. Keep those results in separate run roots.

## DEX-style training track

This track adapts the training recipe from
[Compression Beyond the Uncompressed: A Two-Stage Training Recipe for Soft
Context Compression in RAG (DEX-Comp)](https://arxiv.org/html/2609.05152v1).
DEX-Comp first performs Pure Distillation on teacher-correct examples, then
performs Hard Exploration only on teacher-failed examples. Lilac uses those
ideas as an independent training-recipe axis; its compression-specific Mistral
architecture is not mixed into the tiny AMX model.

The Lilac comparison is:

```text
all-data SFT                 -> baseline training recipe
teacher-correct-only SFT     -> Pure Distillation analogue
teacher-correct-only + HE   -> targeted exploration analogue
```

Stage II uses grouped sampled rollouts, containment rewards, group-relative
advantages, and a sampled-token KL anchor to the Stage I checkpoint. The KL is
deliberately labeled as sampled-token rather than full-vocabulary KL.

Create the two auditable teacher splits:

```bash
python scripts/build_teacher_splits.py \
  --reference-dir .cache/reference/amx-reasoning-v1-instruct \
  --input data/qa_smoke.jsonl \
  --output-dir data/dex_smoke
```

Train Stage I on teacher-correct rows:

```bash
python scripts/train.py --config configs/dex_pd.yaml --data-mode qa \
  --qa-jsonl data/dex_smoke/teacher_correct.jsonl --run-dir runs/dex-pd
```

Run Stage II on teacher-failed rows. The script uses grouped sampled rollouts,
containment rewards, group-relative advantages, and a sampled-token KL anchor to
the Stage I checkpoint:

```bash
python scripts/hard_explore.py --run-dir runs/dex-pd \
  --failed-jsonl data/dex_smoke/teacher_failed.jsonl \
  --steps 100 --group-size 4
```

The one-row smoke fixture may produce an empty failure split if the released
teacher answers it correctly; in that case skip Stage II for the smoke run and
use a larger QA file to exercise hard exploration.

For a real experiment, keep held-out QA data separate and compare: all-data
SFT, teacher-correct-only SFT, pure distillation only, and pure distillation
plus hard exploration. Log results separately for teacher-correct and
teacher-failed slices.

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

The initial runnable experiment kit is present: pinned reference fetch,
released-checkpoint QA evaluation, synthetic induction/shift/add/copy probes,
PyTorch dense and fixed-E MoE training, checkpoint resume, sweeps, and routing
diagnostics, plus the DEX-style teacher-split and hard-exploration track.
