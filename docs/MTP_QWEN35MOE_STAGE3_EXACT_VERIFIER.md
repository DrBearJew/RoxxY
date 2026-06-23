# Qwen35MoE Stage3 exact verifier: state-safe tail batching

## Problem this patch targets

The fast causal-batched verifier can match sampled tokens but diverges in recurrent/KV state bytes before repair.  The serial oracle is byte-exact, but it is slow because it builds and runs the prefix verifier as a token-major serial graph.  Prior prefix layer-batching recovered batched shapes but let batched FFN/MoE numerics feed later recurrent-state writes, so it matched tokens while still producing `state_match=0` before repair.

Stage3 does **not** promote the old generic layer-batched prefix path.  It adds an opt-in `LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH=1` graph shape that keeps the semantic invariant needed for pre-repair state equality:

```text
for every verifier row:
  for every transformer layer:
    run attention/recurrent state-writing work row-serial
    run FFN/MoE row-serial until the final transformer layer

batch only:
  final layer post-attention FFN/MoE
  final output norm / LM head, when LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD=1
```

The key design rule is: **no generic batched op may feed a future recurrent/KV state write**.  Batching is allowed only after the last verifier state write has completed for every row.

## New environment switches

```bash
# Select the existing exact prefix backend.
export LLAMA_MTP_SERIAL_EQUIV_PREFIX=1

# Enable the Stage3 state-safe final-tail batched graph.
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH=1

# Keep the already-added batched output head enabled so output.weight sees ncols_dst > 1.
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD=1

# Optional graph-construction marker.
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH_LOG=1
```

When the server chooses the backend, the reason string changes to:

```text
exact_token_major_prefix_graph_tail_batch_requested
```

The graph emits callback names containing `prefix_exact_tail_*`, and a log marker when `LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH_LOG=1` is enabled.

## MMVQ / MUL_MAT_ID policy for the tail

The server activates `LLAMA_MTP_PREFIX_EXACT_TAIL_ACTIVE=1` only around target prefix verification decode.  During that scope, the existing MMVQ serial-columns mechanism gets an additional filter over the state-free tail tensors:

```text
ffn_gate_inp,ffn_gate_up,ffn_gate,ffn_up,ffn_down,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,output
```

It also forces:

```bash
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS=1
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH=1
```

This is intended to recover the useful batched shapes, especially `ncols_dst=4`, while keeping row-equivalent column handling for MMVQ/MUL_MAT_ID in the final tail.

## Validation gates

The promotion gate is deliberately strict:

```text
token_match=1
state_match=1 before repair
stable accepted length across repeated runs
speed > safe serial baseline
```

Run with compare logging enabled and reject the patch if any `MTP_VERIFY_COMPARE:` line has `token_match=0` or `state_match=0`.

## Why this is not the old unsafe layer-batched verifier

Old unsafe shape:

```text
(layer 0 batched FFN/MoE) -> layer 1 recurrent state write -> ...
```

That can perturb hidden activations before later state writes, which is enough to produce byte-level state divergence even when sampled tokens match.

Stage3 shape:

```text
all state writes row-serial -> final state-free tail batched
```

The final batched tail can affect logits and acceptance decisions, so token-match and acceptance stability still need validation.  It should not affect pre-repair committed recurrent/KV state bytes, because those state bytes are produced before the batched tail.

## About direct Q8_0 LM-head top1

For `n_rows` around 1/3/4, direct Q8_0 top1 can be slower than full-logits top1 on a large real model.  The direct path pays hidden-row quantization and top1 scan overhead on very small row counts and may launch many narrow kernels.  The full-logits path reuses the mature high-throughput matmul route and can amortize work better even though it materializes logits.  Direct top1 is still useful as a correctness/control surface, but the exact verifier blocker should be solved before additional direct-top1 kernel polishing.

## Next kernel work after this patch

If this candidate gives `state_match=1` but not enough speed, the next step is not generic layer batching.  The next step is custom row-equivalent small-N kernels for Qwen35MoE verifier tails/projections/MoE that expose batched launch geometry while internally preserving the single-row arithmetic order required by the serial oracle.
