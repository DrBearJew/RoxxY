# Qwen35MoE MTP Stage4 exact row-equivalent verifier

Stage3 proved that the exact prefix verifier can preserve serial recurrent-state
bytes, but it was demoted because batching only the final state-free tail was
slower than both the serial-equivalent prefix baseline and the safe/default
serial verifier. The useful next target is not more LM-head or MoE dot4 polish;
it is recovering `ncols_dst=4` verifier shapes while preserving the serial state
contract.

Stage4 therefore moves the speed work back inside every transformer layer, but
only between state barriers.

## Contract

For each verifier prefix row and layer:

1. Attention/recurrent state-writing work remains token-major and row-serial.
2. The post-attention RMSNorm that feeds FFN/MoE is computed row-serial.
3. The layer FFN/MoE is batched across verifier rows.
4. Every batched FFN/MoE matmul that can feed a future state write is forced
   through row-equivalent small-N MMVQ kernels. Those kernels preserve the same
   `ncols_dst=1` arithmetic/reduction shape as the serial oracle for each
   verifier row, while flattening rows into a single launch when possible.
5. The resulting FFN/MoE rows are concatenated and used as the next layer input.

Promotion requires all of the following in a real-model run:

```text
token_match=1
state_match=1 before repair
stable acceptance
speed > safe/default serial verifier baseline
```

## New graph mode

```bash
LLAMA_MTP_SERIAL_EQUIV_PREFIX=1
LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH=1
LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG=1
LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD=1
```

Compatibility aliases are also accepted:

```bash
LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH=1
LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH=1
```

Expected graph marker:

```text
MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe)
```

Expected backend reason:

```text
exact_roweq_layer_ffn_prefix_requested
```

## Backend policy

During the target prefix verifier decode, the server scopes:

```bash
LLAMA_MTP_PREFIX_ROWEQ_LAYER_ACTIVE=1
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE=1
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS=1
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH=1
LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER=ffn_gate_inp,ffn_gate_up,ffn_gate,ffn_up,ffn_down,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,ffn_gate_inp_shexp
```

The filter intentionally covers router, routed expert, and shared expert FFN
weights. These outputs can feed later recurrent/KV state writes, so normal
multi-column MMVQ arithmetic is unsafe even when sampled tokens match.

Stage4 extends the serial-column path to fused GLU/bias MMVQ as well. Fused
`ffn_gate_up` projections no longer have to fall back to unsafe multi-column
geometry or N separate host launches; they can use:

```text
route=rdna3_mmvq_dot4_serial_columns
route=mmvq_serial_columns_fused_single_launch
```

The non-fused generic path keeps the previous markers:

```text
route=mmvq_serial_columns
route=mmvq_serial_columns_single_launch
```

## Validation

Use the helper wrapper:

```bash
scripts/run_qwen35moe_stage4_exact_row_equiv_env.sh ./llama-server ... 2>&1 | tee stage4_roweq.log
```

Then validate:

```bash
scripts/check_qwen35moe_stage4_exact_row_equiv_log.py stage4_roweq.log --require-ncols 4
```

A passing log must show the Stage4 graph marker, Stage4 backend reason,
`token_match=1`, `state_match=1` before repair, and selected serial-column
routes with `ncols_dst=4`.

## Bisect valves

These valves help isolate any remaining `state_match=0` without changing the
state-writing schedule:

```bash
LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST=0
LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST=15
```

Layers outside the range fall back to the fully row-serial per-layer path. The
safe promotion configuration leaves the range at the full transformer depth.

## Why this is different from Stage3

Stage3 batched only after all verifier state writes were complete. That was
state-exact but too little work to matter. Stage4 batches the expensive FFN/MoE
block at every layer, but it makes the batched block row-equivalent at the
backend level before its result can feed the next state write.

This is the narrow safe target: recover useful `ncols_dst=4` work without
returning to the old causal-batched failure mode where `token_match=1` but
`state_match=0` before repair.
