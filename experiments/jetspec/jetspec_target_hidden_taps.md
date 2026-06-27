# JetSpec target-hidden tap contract draft

Status: inert design fixture. This file is not included by CMake and no llama.cpp
runtime reads it.

Purpose: define exactly how a future llama.cpp JetSpec integration should expose
target hidden states to the draft head while keeping normal greedy output
unchanged when JetSpec is disabled.

## Upstream reference semantics

Pinned upstream checkout:
`/home/mrtrent/.harness/tmp/jetspec-upstream-master`

Relevant source points:

- `jetspec/models/draft_head.py:53` `extract_context_feature(hidden_states, layer_ids)`
- `jetspec/core/model_runner.py:62` `_capture_target_hidden(model, target_layer_ids)`
- `jetspec/core/model_runner.py:146` hook path concatenates `[hidden_by_layer[layer_id] ...]`
- `jetspec/draft_head_adapter.py:52` `block_output_ids()` builds `[anchor, fill..., mask...]`
- `jetspec/inference_engine/compiled_verify_stack.py:262` post-layer output tap

Critical semantic detail:

- HF hidden-state tuple index `0` is embedding output.
- Target layer `L` tap is `hidden_states[L + 1]`.
- Hook/compiled paths capture the post-layer residual for target layer `L`.
- Taps are concatenated along feature dimension in `target_layer_ids` order.

For the Qwen3.6 JetSpec head staged here:

- `target_layer_ids = [1, 10, 19, 28, 37]`
- `target_hidden_size = 2048`
- concatenated width = `5 * 2048 = 10240`
- `draft.fc.weight` shape = `[2048, 10240]`

## Capture contract

A future target forward with JetSpec enabled may return an optional side-channel:

```text
(logits, kv_state, target_hidden)
```

Where:

```text
target_hidden.shape == [batch, new_tokens, len(target_layer_ids) * hidden_size]
```

The target logits and KV state must be the same values the target forward would
produce without hidden capture. Hidden capture is observational, not a mutation
point.

For each generated/verified target token position:

1. Run normal target decoder layers.
2. After each requested layer `L`, capture the post-layer output vector.
3. Do not alter the vector used by later layers.
4. Emit concatenated vector in exact order `[tap(1), tap(10), tap(19), tap(28), tap(37)]`.
5. Append only committed/accepted-token hidden rows to the persistent draft-head context.

Rejected tree branches must not be appended to the committed `target_hidden` cache.

## Normal greedy output invariant

When JetSpec is disabled:

- No hidden tap allocation is required.
- No hidden tap copy is allowed on the hot path.
- Target logits, sampler inputs, and KV cache updates must follow the existing path.

When JetSpec is enabled:

- Hidden tap capture must be side-channel only.
- It must not change target logits, target KV layout, sampling order, or accepted token ids.
- Any fast/hooked/compiled capture path must byte-match the fallback equivalent of
  concatenating HF `hidden_states[L + 1]` for each requested layer.

## Block input IDs for the draft head

The draft head also needs target embeddings for a synthetic block:

```text
block_output_ids = [anchor_token_id] + fill_tokens[:15] + mask_token_id padding
```

For normal tree proposal with no fill path:

```text
[anchor_token_id, mask_token_id, mask_token_id, ..., mask_token_id]
```

For this model:

- `block_size = 16`
- `draft_depth = 15`
- `mask_token_id = 248070`

These IDs are embedded with the target model's `embed_tokens`. The draft GGUF
must not carry its own token embeddings.

## Target-hidden cache contract

The draft head conditions on the committed target context hidden cache.

Required cache behavior:

- Prefill: store target hidden rows for prompt tokens when JetSpec draft head is active.
- Decode verify: store hidden rows for accepted target tokens only.
- Tree verify: if a batch contains accepted and rejected nodes, gather only accepted
  path rows into the committed hidden cache.
- Rollback: rejected branch rows are discarded together with any speculative KV/state.
- Shape: persistent hidden cache width is always `10240` for this Qwen3.6 head.

## Fail-closed checks

A future integration must reject or disable the JetSpec head if any condition is
true:

- A requested layer id is missing from the target model.
- `target_layer_ids` are unsorted, duplicated, or out of range.
- Captured vector width is not `2048`.
- Concatenated width is not `10240`.
- `draft.fc.weight` input width disagrees with `len(target_layer_ids) * hidden_size`.
- Target embeddings are unavailable.
- Target `lm_head` is unavailable or has incompatible vocab width.
- Hidden capture changes normal target logits in an A/B fixture.
- Rejected branch hidden rows can leak into committed draft-head context.

## Inert validation tools

- `target_hidden_taps.py`: stdlib helper for tap spec, block ID, capture order, HF `hidden_states[layer_id + 1]` extraction, concatenation checks, and A/B side-channel parity checks.
- `test_target_hidden_taps.py`: unit tests for plan-derived Qwen3.6 dimensions, deterministic fixture output, fail-closed missing/wrong-width layers, immutable side-channel capture, hidden-state index mapping, target mismatch, and unchanged logits/greedy output.
- `fixtures/target_hidden_taps_smoke.json`: small synthetic layer-output fixture.
- `fixtures/target_hidden_taps_smoke.out.json`: expected block IDs and concatenated hidden vector.
- `fixtures/target_hidden_tap_parity_smoke.json`: compact P3 A/B fixture that generates width-2048 HF hidden states for 40 target layers, captures `[1, 10, 19, 28, 37]`, verifies concatenated width `10240`, and compares baseline vs capture logits/greedy output.
- `fixtures/target_hidden_tap_parity_smoke.out.json`: expected P3 parity result with `fallback_matches_hook_capture=true` and `capture_is_side_channel_only=true`.

These fixtures are contracts only. They do not expose hidden states from llama.cpp
today and do not modify the original `llama-server` build.
