# Qwen35MoE MTP Stage4.2 router/top-k containment

## Purpose

Stage4.1 initially restored verifier exactness for a narrow real-model gate when
router/top-k/weight processing stayed row-serial and routed expert projections
were batched through serial-column MMVQ.  Later width/state-sequence testing
showed that this was too optimistic: `batch_routed_proj=1` can pass some
`n_tokens=3` gates but still produce pre-repair `state_match=0` in other verifier
sequences.

Stage4.2 is therefore a containment and diagnostic patch, not a promotion claim.
The safe default is now the repaired serial-row MoE fallback
(`batch_routed_proj=0`).  Router dense MMVF and batched routed projections remain
opt-in diagnostic tools for finding a future exact fast path.

## What Stage4.2 fixes

### 1. `batch_routed_proj=0` is a real row-serial control path

Stage4.1 documented `LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=0` as a
fallback to the old fully row-serial MoE/FFN behavior inside the Stage4.1 layer
schedule.  The implementation did not actually do that: it called the generic
MoE builder with the full multi-row verifier tensor, so it could reproduce the
same hidden/state drift seen when router/top-k was batched.

Stage4.2 changes the fallback to build the old FFN one verifier row at a time:

```text
for row in verifier_rows:
  cur_row = view(cur, row)
  out_row = build_layer_ffn(cur_row, layer)
  out_all = concat(out_all, out_row)
```

The marker remains:

```text
batch_routed_proj=0
batch_routed_proj0_fallback=serial_rows
```

Continuation correction: this fallback is now the **safe default**.  Batched
routed projections are disabled by default with:

```bash
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS=0
```

For diagnostic reproduction of the old Stage4.2 candidate, explicitly opt back
in, for example:

```bash
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS=3
```

Do not promote `batch_routed_proj=1` unless it passes the full pre-repair
`state_match=1` gate across width/state sequences, not only a single narrow
`n_tokens=3` run.

and the row-serial fallback emits nodes named like:

```text
prefix42_full_serial_ffn_input_batch_routed_proj0_row0
prefix42_full_serial_ffn_out_batch_routed_proj0_row0
prefix42_full_serial_ffn_out_all_batch_routed_proj0
```

### 2. Router dense and top-k/weights are independently switchable

Stage4.1 bisection could not isolate dense router matmul from softmax/top-k/
weight normalization because any request to serialize one effectively serialized
both in the graph-building path.

Stage4.2 splits the graph into four cases:

```text
serial_router=1 serial_topk_weights=1:
  router row-serial, top-k/weights row-serial

serial_router=1 serial_topk_weights=0:
  router row-serial, logits concatenated, top-k/weights batched

serial_router=0 serial_topk_weights=1:
  router batched, logits row views, top-k/weights row-serial

serial_router=0 serial_topk_weights=0:
  router batched, top-k/weights batched
```

Aliases are accepted for local bisection scripts:

```bash
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK=1
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED=0
```

The canonical variables remain:

```bash
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=1
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1
```

### 3. Opt-in router dense small-N backend route

Stage4.2 adds a narrow CUDA route for dense router matmul:

```text
route=router_mmvf_serial_columns
```

The route is selected only when all of the following hold:

```text
LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE=1
src0 tensor name matches LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER
src0 type is f32/f16/bf16
src1 and dst are f32
src1 rows/columns are not transposed
2 <= ncols_dst <= LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS
```

Default filter:

```bash
LLAMA_MTP_ROWEQ_ROUTER_MMVF_FILTER=.ffn_gate_inp.weight
LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS=4
```

This is not a proof that router batching is exact on the target model.  It is a
candidate route for the next run: if `serial_router=0` and
`serial_topk_weights=1` is exact with this route, the remaining speed target is
softmax/top-k/weight normalization.  If it is not exact, the dense router matmul
still needs a stricter custom row-equivalent implementation.

### 4. The checker is bisection-aware

The Stage4.1 checker can still enforce the original all-serial safe-island, but
it no longer requires that profile by default.  Use explicit profiles when the
component flags matter:

```bash
# Latest exact minimal profile from Stage4.1 bisection.
scripts/check_qwen35moe_stage41_component_bisect_log.py stage42-minimal.log \
  --component-profile minimal-router-topk \
  --no-require-hidden-trace \
  --require-ncols 4

# Stage4.2 router dense candidate.
scripts/check_qwen35moe_stage42_router_topk_log.py stage42-router-topk.log \
  --require-ncols 4

# Full row-serial fallback control, no serial-column routes expected.
scripts/check_qwen35moe_stage41_component_bisect_log.py stage42-proj0.log \
  --component-profile full-serial-fallback \
  --no-require-serial-columns \
  --no-require-hidden-trace

# Intentional failing component-off bisection.
scripts/check_qwen35moe_stage41_component_bisect_log.py stage42-router0.log \
  --allow-pre-repair-state-miss \
  --no-require-component-flags \
  --require-ncols 3
```

Route checks still require high verbosity, for example `llama-cli -v` or
`llama-server -v`, because backend route selection lines are logged at
`GGML_LOG_INFO`.

## Run profiles

### A. Safe Stage4.2 router-dense/default control

Use this first to test the current safe containment path.  Router dense may still
use the small-N MMVF route, but routed MoE projections fall back to the repaired
serial-row path by default:

```bash
scripts/run_qwen35moe_stage42_router_topk_env.sh ./llama-server -v ... 2>&1 | tee stage42-router-topk.log

scripts/check_qwen35moe_stage42_router_topk_log.py stage42-router-topk.log \
  --require-ncols 4
```

Equivalent component settings:

```bash
LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK=1
LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN=0
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS=0
LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE=1
LLAMA_MTP_ROWEQ_ROUTER_MMVF_MAX_COLS=4
```

Expected backend reason:

```text
exact_roweq_stage42_router_topk_prefix_requested
```

Expected marker fields:

```text
stage42_router_topk=1
router_topk_split=1
serial_router=0
serial_topk_weights=1
batch_routed_proj=0
```

Expected router route:

```text
route=router_mmvf_serial_columns ... tensor=blk.N.ffn_gate_inp.weight ... ncols_dst=2|3|4
```

### B. Diagnostic batched-projection containment profile

This was the narrow exactness profile from the original reported bisection, but
it is no longer safe as a default.  Use only to reproduce or debug the demoted
batched routed-projection path:

```bash
scripts/run_qwen35moe_stage42_router_topk_minimal_env.sh ./llama-server -v ... 2>&1 | tee stage42-minimal.log

scripts/check_qwen35moe_stage41_component_bisect_log.py stage42-minimal.log \
  --component-profile minimal-router-topk \
  --require-ncols 4
```

Equivalent component settings:

```bash
LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS=1
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE=0
LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN=0
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS=1
LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS=3
```

This profile passed the original reported bisection but later failed another
real-model width/state sequence with pre-repair `state_match=0`.  Treat it as
unsafe/diagnostic only.

### C. Full row-serial fallback control

Use this to prove that `batch_routed_proj=0` now does what the Stage4.1 docs
promised:

```bash
scripts/run_qwen35moe_stage42_full_serial_fallback_env.sh ./llama-server -v ... 2>&1 | tee stage42-proj0.log

scripts/check_qwen35moe_stage41_component_bisect_log.py stage42-proj0.log \
  --component-profile full-serial-fallback \
  --no-require-serial-columns \
  --no-require-hidden-trace
```

## Next optimization boundary

Do not spend the next performance phase on broad router/top-k fusion.  Stage4.3
fused router/top-k failed exactness, and rocprofv3 characterization showed
router/top-k is tiny compared with expert matvec work.  The next fast path should
target the exactness-preserving expert matvec/MoE projection bottleneck that the
safe fallback exposes:

```text
routed gate/up projection
routed down projection
shared gate/up/down projection
serial-row MoE fallback overhead
```

Any candidate must prove:

```text
token_match=1
state_match=1 before repair
stable acceptance
speed > safe/default serial verifier baseline
```

before any promotion discussion.


## Stage4.3 follow-up note

Stage4.3 fused router/top-k was tested and demoted: it selected `router_topk_weights_roweq_fused_candidate` but failed the hard pre-repair state gate (`token_match=1,state_match=0`).  Do not promote it.  The current safe Stage4.2 default is the serial-row MoE fallback with router/top-k diagnostics kept opt-in.
