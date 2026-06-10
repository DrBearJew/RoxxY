# Qwen35MoE MTP Stage4.1 component-bisect verifier

Stage4 proved that backend route selection is not the remaining blocker.  A
layer-0-only run selected the generic `mmvq_serial_columns` /
`mmvq_serial_columns_single_launch` routes for the real tensor mix, but still
reported `token_match=1` and `state_match=0` before repair.  Because disabling
all roweq layers restored `state_match=1`, the remaining gap is inside the
batched FFN/MoE segment, not in prefix backend selection.

The real layer-0 tensor mix also explains why the RDNA3 Q4_K dot4 serial-column
route is not expected there:

```text
blk.0.ffn_gate_up_exps.weight : iq3_s
blk.0.ffn_down_exps.weight    : iq4_xs
blk.0.ffn_*_shexp.weight      : q8_0
blk.0.ffn_gate_inp.weight     : f32
blk.0.ffn_gate_inp_shexp      : f32
```

So Stage4.1 does not polish the RDNA3 dot4 path.  It adds a narrow diagnostic and
containment graph for the actual suspected source: dense router/shared-gate,
top-k/weight normalization, shared expert FFN, routed non-matmul glue, and expert
weighted aggregation were still batched in Stage4 and can perturb hidden bytes
that later feed recurrent-state writes.

## Design

The Stage4.1 candidate keeps the Stage4 state schedule:

```text
for each layer:
  run attention/recurrent state-writing work row-serial
  concatenate post-attention-normalized rows
  run a component-bisect MoE/FFN segment
  concatenate hidden rows before the next layer state write
```

Inside the component-bisect MoE segment, the default candidate batches only the
expensive routed expert projections:

```text
batched through serial-column backend policy:
  ffn_gate_up_exps
  ffn_down_exps

row-serial by default:
  router dense matmul
  softmax/top-k/expert weights
  routed activation/scale glue around the projections
  weighted expert aggregation
  shared expert dense gate
  shared expert FFN
  final MoE + shared-expert add
```

This is intentionally slower than a future optimized implementation.  Its job is
to answer the immediate exactness question:

```text
Can we keep ncols_dst=3/4 routed expert projections while restoring
state_match=1 before repair?
```

If yes, the next optimization target is custom fused row-equivalent small-N
kernels for the specific serial components that the bisection proves necessary.
If no, the hidden-row hashes identify the first layer/row/component where the
batched projection result itself diverges.

## Main activation

Use the wrapper:

```bash
scripts/run_qwen35moe_stage41_component_bisect_env.sh ./llama-server ... -v 2>&1 | tee stage41.log
```

The `-v` / high-verbosity requirement matters: serial-column route lines are
emitted at `GGML_LOG_INFO`, so route checks can be falsely absent in quiet logs.

Equivalent environment:

```bash
export LLAMA_MTP_SERIAL_EQUIV_PREFIX=1
export LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG=1
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH=1
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG=1
export LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH=0
export LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD=1

export LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE=1
export LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN=1

export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE=1
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS=1
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH=1
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX=4
export LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG=1

export LLAMA_MTP_VERIFY_TRACE=1
export LLAMA_MTP_VERIFY_COMPARE=1
export LLAMA_MTP_PREFIX_HIDDEN_TRACE=1
export LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER=0

export LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST=0
export LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST=0
```

Expected backend reason:

```text
exact_roweq_stage41_component_bisect_prefix_requested
```

Expected graph marker:

```text
MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe): ... stage41=1 ... contract=serial_attention_component_bisect_ffn
```

Expected hidden-row hash marker:

```text
MTP_PREFIX_HIDDEN_TRACE: layer=0 kind=h row=... node=prefix41_hidden_pre_next_state_row... hash=...
```

## Component bisection knobs

All are active by default when `LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG=1`, unless
`LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL=1` is set.

```bash
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN=1
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1
```

Suggested run order:

1. Candidate A, default wrapper: layer 0 only, all serial glue enabled, routed
   `gate_up/down` projections batched.
2. If exact, turn off one serial knob at a time to find which component requires
   row-serial semantics.
3. If not exact, keep all serial knobs enabled and disable
   `LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS`.  Stage4.2 fixes the
   Stage4.1 bug in this path: `batch_routed_proj=0` now builds the old FFN path
   one verifier row at a time instead of accidentally calling the generic MoE
   builder with a multi-row tensor.  This is a control/fallback path, not a speed
   candidate.
4. Compare `MTP_PREFIX_HIDDEN_TRACE` hashes between the exact and failing runs to
   locate the first hidden row/layer divergence before the next state write.

## Validation

Diagnostic mode, useful while isolating a failing component:

```bash
scripts/check_qwen35moe_stage41_component_bisect_log.py stage41.log \
  --allow-pre-repair-state-miss \
  --no-require-component-flags \
  --require-ncols 3
```

When intentionally turning component flags off, use
`--no-require-component-flags` or one of the explicit profiles below.  The old
safe-island default expects every component flag to be 1 and will reject minimal
or failing bisection logs by design.

```bash
# Latest exact minimal profile from the real-model bisection.
scripts/check_qwen35moe_stage41_component_bisect_log.py stage41-minimal.log \
  --component-profile minimal-router-topk \
  --no-require-hidden-trace \
  --require-ncols 4

# Full row-serial fallback control.  It should not require serial-column routes.
scripts/check_qwen35moe_stage41_component_bisect_log.py stage41-proj0.log \
  --component-profile full-serial-fallback \
  --no-require-serial-columns \
  --no-require-hidden-trace
```

Promotion mode, only after exactness is restored:

```bash
scripts/check_qwen35moe_stage41_component_bisect_log.py stage41.log \
  --require-ncols 4
```

Promotion criteria remain strict:

```text
token_match=1
state_match=1 before repair
stable acceptance
selected serial-column route with target ncols_dst=3/4
speed > current safe/default serial verifier baseline
```

## Important non-goals

Stage4.1 is not the final fast path.  It deliberately spends extra graph nodes to
preserve row semantics and expose per-component switches.  Once the exact subset
is known, the fast Stage5 target should replace that subset with custom
row-equivalent small-N kernels rather than keeping all Stage4.1 serial glue.


## Stage4.2 addendum: router/top-k finding and containment fixes

Initial real-model bisection showed that Stage4.1 was a useful exactness
containment patch but not a promotable speed path.  The original narrow minimal
subset was:

```text
serial_router=1
serial_topk_weights=1
serial_routed_glue=0
serial_expert_agg=0
serial_shared_gate=0
serial_shared_ffn=0
batch_routed_proj=1
```

Continuation correction: `batch_routed_proj=1` is not robust enough for safe
default use.  It can pass the original `n_tokens=3` gate but fail later
width/state sequences with pre-repair `state_match=0`.  The repaired
`batch_routed_proj=0` serial-row fallback is now the safe default, while batched
routed projections require explicit diagnostic opt-in with
`LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS`.

Stage4.2 therefore makes three containment changes:

1. `batch_routed_proj=0` is repaired.  It now builds the old FFN/MoE path
   verifier-row-by-verifier-row, so it is a real serial control path.
2. Router and top-k/weight bisection are split.  `SERIAL_ROUTER=1` now isolates
   the dense router matmul; `SERIAL_TOPK_WEIGHTS=1` now isolates softmax,
   grouping, top-k, and expert-weight normalization.  Setting one no longer
   silently serializes the other.
3. The checker documents and supports component-off bisection with
   `--no-require-component-flags`, `--component-profile minimal-router-topk`,
   and `--component-profile full-serial-fallback`.

The next performance patch should not start from router/top-k fusion: Stage4.3
already selected its fused route but failed `state_match=1`, and rocprofv3 showed
router/top-k work is tiny versus expert matvec work.  The next target is an
exactness-preserving expert matvec / MoE projection schedule that reduces the
serial-row fallback overhead without changing recurrent-state bytes.  It must
prove `token_match=1` and `state_match=1` before repair against the serial oracle
before any speed comparison or promotion discussion.
