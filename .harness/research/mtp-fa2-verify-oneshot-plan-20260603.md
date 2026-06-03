# MTP FA2 Verify One-Shot Plan — 2026-06-03

## Correction

The previous plan was wrong.

This is not primarily a generic MMVQ problem, and it is not acceptable to hand-wave the long-context KV scan as an unavoidable decode limit.

For MTP verification, the target model should be doing a **small-Q FlashAttention-2 verification pass**:

```text
Q = 2..4 speculative target positions
K/V = full context + local speculative positions
mask = real MTP verification mask
```

A correct FA2-style verifier should reuse the K/V stream across all verification queries and should be materially faster than independent decode. MTP should be far faster if the target verification path is actually using an efficient multi-Q FA2 kernel and if the rest of the graph is not serializing the candidate tokens.

Current measured result:

```text
27B Q4_K_M, 7900 XTX, ~14k context
no-MTP reference:          ~32.4 tok/s
MTP n=3 batched verify:    ~25.6 tok/s
```

That result means our current path is still wrong or incomplete. It does **not** mean MTP cannot win.

## Actual goal

Make MTP target verification a true FA2 small-Q path:

```text
nq = 2..4
nk = long context, e.g. 14k+
head_dim = 128
GQA = 6 for 27B
V cache = q4_0
K/Q packed16 DOT4 path where applicable
real FA mask consumed
```

Target:

```text
MTP n=3 at 14k >= 45 tok/s
```

Minimum acceptable first win:

```text
MTP n=3 at 14k > no-MTP reference, i.e. >32 tok/s
```

## Core diagnosis

The current `small_verify_batched_splitk` route improved the catastrophic prefill/PWMMA fallback, but the measured speed says it is still behaving too much like decode or is losing the FA2 advantage somewhere.

Possible failure modes:

1. MTP verify is still not consistently routed through the optimized FA2-style path.
2. The current custom kernel is batched, but not FA2 enough: too much per-query work, poor Q reuse, poor V reuse, or expensive split-K reduce.
3. The verify graph may still be issuing multiple FA ops/graphs instead of one small-Q verify op per layer.
4. Mask handling may force slow generic behavior or inhibit FA2 tiling.
5. Q/K/V layout conversions or q8/q4 repacking may dominate around the FA op.
6. The kernel may process all Q in one CTA but still underutilize waves/CUs at long context.
7. Stage1/reduce split-K structure may be launching too much and spilling too much partial state.

The next work must prove and fix this at the FA2 verifier level.

## Non-negotiable constraints

Keep all experimental routing explicit:

```bash
LLAMA_MTP_FA_ROUTE=1
GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=small_verify_fa2
GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ=4
GGML_CUDA_ROCM_Q8K_DOT4_DECODE_MAX_NQ=4
```

Do not change default behavior.

Do not route `nq=1` into this kernel:

```text
nq=1 -> existing decode split-K path
nq=2..4 -> explicit MTP FA2 verify path only
```

Do not ignore the real FA mask.

Do not print per-call logs.

Do not use synthetic keyword prompts for acceptance testing.

Do not use 256-token runs for iteration. Use `max_tokens=16` or `32` until there is a clear win.

## Correct next implementation

Implement a new route:

```text
rocm_packed16_small_verify_fa2
```

Selected by:

```bash
GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=small_verify_fa2
```

This route is not just another decode split-K variant. It should be a dedicated FA2-style small-Q verifier.

## Kernel design

### Shape

Target shape:

```text
nq:        2, 3, 4
nk:        long context, especially >=8192
head_dim:  128
GQA:       6 query heads per KV head on 27B
V type:    q4_0
Q/K dot:   packed16/DOT4 path already proven for QK
mask:      real FA mask
```

### Grid

Use a grid that gives enough parallelism over long K while keeping all Q in the same K/V stream.

Preferred starting point:

```text
grid.x = K/V tile or split tile
grid.y = KV head
grid.z = batch
```

Inside each CTA/wave group:

```text
process all nq queries for the same KV head
process all GQA query heads for that KV head if register/shared-memory budget allows
stream K/V tile once
compute QK for all q
softmax update for all q
accumulate P×V for all q
```

The important invariant:

```text
K/V tile must be loaded once and consumed by all verification queries.
```

Bad:

```text
for q in nq:
    scan full K/V
```

Good:

```text
for K/V tile:
    load K/V once
    update q0/q1/q2/q3 accumulators
```

### FA2 online softmax

Use online softmax per query:

```cpp
m_q = max(m_q, score)
l_q = l_q * exp(old_m_q - new_m_q) + exp(score - new_m_q)
acc_q = acc_q * exp(old_m_q - new_m_q) + exp(score - new_m_q) * V
```

Maintain separate state for each query:

```text
m0/l0/acc0
m1/l1/acc1
m2/l2/acc2
m3/l3/acc3
```

Specialize by compile-time `NQ`:

```cpp
small_verify_fa2<NQ=2>
small_verify_fa2<NQ=3>
small_verify_fa2<NQ=4>
```

Do not use dynamic `for q < nq` inside the hot loop for the final optimized path.

### Mask semantics

The kernel must consume the real FA mask.

MTP verification is not simple decode causal masking. It has target positions and speculative positions. The kernel must preserve exact mask semantics already proven by the current mask-aware small-verify kernels.

Mask rule:

```text
if mask says disallow, score = -inf
```

Do not approximate mask bounds unless separately proven equivalent.

### Split-K / FA2 partitioning

Long context needs enough K parallelism. Use split-K, but reduce overhead must be kept minimal.

Stage1 should emit compact partials:

```text
for each split, kv_head, batch, query-head/group, q:
    m_partial
    l_partial
    acc_partial[head_dim]
```

Reduce should combine splits using online softmax merge:

```cpp
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
acc = acc_a * exp(m_a - m) + acc_b * exp(m_b - m)
```

Then normalize:

```cpp
out = acc / l
```

Critical optimization:

```text
partials for q0/q1/q2/q3 should be contiguous so reduce reads coalesced.
```

### Avoid current likely failure

If current `small_verify_batched_splitk` does:

```text
all-QK sweep
softmax
P×V
```

but still stores too much intermediate state or repeats V work per query/head inefficiently, replace it with true online FA2 accumulation so that P×V is fused into the K/V tile stream.

FA2 verify should not materialize full attention probabilities.

It should only materialize per-split partial softmax state and output accumulators.

## Route integration

Existing route issue:

MTP target verification arrives as:

```text
GGML_FATTN_INST_PREFILL_QK
```

not reliably as:

```text
GGML_FATTN_INST_MTP_VERIFY_QK
```

Therefore route must be keyed on explicit opt-in and small-Q shape, not only the instruction enum.

Route condition:

```cpp
const bool explicit_mtp_fa_route = getenv("LLAMA_MTP_FA_ROUTE") && atoi(getenv("LLAMA_MTP_FA_ROUTE")) != 0;
const bool small_q = nq >= 2 && nq <= max_nq && nq <= 4;
const bool supported_shape =
    explicit_mtp_fa_route &&
    small_q &&
    nk >= 1 &&
    head_dim == 128 &&
    gqa == 6 &&
    v_type == GGML_TYPE_Q4_0 &&
    qk_type_is_supported;

if (impl == "small_verify_fa2" && supported_shape) {
    launch_rocm_packed16_small_verify_fa2<NQ>(...);
    return;
}
```

`PREFILL_QK` must not exclude the route. The explicit env gate is the safety boundary.

## Files to modify

Primary:

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-decode.cuh
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
ggml/src/ggml-cuda/fattn.cu
```

Possible shared helpers:

```text
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/common.cuh
```

Tests:

```text
tests/test-packed16-decode-variants.cpp
benchmarks/mtp-fattn-correctness.sh
```

## Implementation sequence

### Step 1 — Add route name only

Add:

```text
small_verify_fa2
rocm_packed16_small_verify_fa2
```

Route it only under explicit env and `nq=2..4`.

At first it can call the existing batched split-K implementation but must log a deduped route line:

```text
packed16_decode_impl selected=small_verify_fa2 nq=3 nk=14336 hq=36 hk=6 gqa=6
```

Purpose: prove routing without changing math.

### Step 2 — Add profile counters around current route

Add sampled timing:

```bash
GGML_CUDA_ROCM_SMALL_VERIFY_FA2_PROFILE=1
GGML_CUDA_ROCM_SMALL_VERIFY_FA2_PROFILE_EVERY=128
```

Collect:

```text
stage1_ms
reduce_ms
nq
nk
split_size
n_splits
head_dim
gqa
launch count
```

No unconditional logs.

Purpose: identify whether stage1, reduce, or launch overhead is killing the FA2 advantage.

### Step 3 — Implement FA2 online stage1

Create a new stage1 kernel:

```cpp
template <int NQ, int HEAD_DIM, int GQA>
__global__ void small_verify_fa2_stage1_kernel(...)
```

Requirements:

- load K tile once;
- compute QK for all `NQ` queries;
- apply real mask;
- online softmax per q;
- accumulate P×V per q;
- emit compact per-split partials only.

Do not materialize probability matrix.

### Step 4 — Implement compact reduce

Create reduce kernel:

```cpp
template <int NQ, int HEAD_DIM>
__global__ void small_verify_fa2_reduce_kernel(...)
```

Merge split partials using online softmax merge.

Write final output in the exact layout expected by current FA op.

### Step 5 — Correctness harness

Add/extend packed16 decode variant tests:

```text
small_verify_fa2 nq=2 nk={256,4096,14336}
small_verify_fa2 nq=3 nk={256,4096,14336}
small_verify_fa2 nq=4 nk={256,4096,14336}
```

Compare against trusted reference/current mask-aware path.

Tolerance: same as existing packed16 decode correctness harness.

Must pass with real mask cases.

### Step 6 — Short live proof

Run only short generation:

```text
27B Q4_K_M
14k prompt
MTP n_max=3
max_tokens=16 or 32
temperature=0
```

Env:

```bash
LLAMA_MTP_FA_ROUTE=1
GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=small_verify_fa2
GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ=4
GGML_CUDA_ROCM_Q8K_DOT4_DECODE_MAX_NQ=4
```

Pass criteria:

```text
route hits nq=2..4 at nk≈14k
no-MTP unchanged
MTP n=3 > 32 tok/s
no ROCm teardown errors
no stderr spam
```

## Kernel performance strategy

### QK

Use existing packed16/DOT4 machinery. This part has already been proven valuable.

Do not regress to scalar QK.

### V

V is q4_0. P×V must be fused into FA2 online accumulation.

Prior WMMA P×V attempt was slower. Do not reuse that version blindly.

For v1:

```text
scalar/vectorized q4_0 P×V inside FA2 tile stream
```

Then optimize:

- vectorized q4_0 loads;
- reuse V scales across Q;
- avoid reload per q;
- keep `NQ` accumulators in registers;
- template `NQ=2/3/4`.

### Split size

Expose runtime knob:

```bash
GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE=256|384|512|768|1024
```

Use quick 14k/32-token tests only.

### Occupancy

If one CTA per split/head underutilizes the GPU, increase parallelism across:

```text
KV split
query head within GQA group
head_dim lanes
```

But do not split each query into separate K/V streams. That loses the MTP/FA2 benefit.

## Expected outcome

A correct FA2 MTP verify route should not be slower than no-MTP at 96% acceptance.

Expected target range after true FA2 verify:

```text
>=32 tok/s minimum
40-50 tok/s plausible target
```

If the route remains ~25 tok/s after true FA2 online accumulation, then the problem is likely not the attention math but one of:

```text
graph/launch churn
verification being invoked too many times
acceptance loop not actually yielding 3 tokens per target pass
non-FA ops dominating due to repeated target graph execution
```

At that point inspect the MTP scheduler/acceptance loop, not MMVQ.

## Hard stop conditions

Stop and reassess if:

```text
nq=1 routes to small_verify_fa2
mask correctness fails
route only works at nk=256 but not 14k
stage1/reduce timings show no FA2 reuse benefit
MTP accepted-run distribution shows mostly 1 token per target pass despite high aggregate acceptance
```

## Final one-line plan

Build a true explicit `small_verify_fa2` MTP verification route: packed16/DOT4 QK, real-mask online FA2 over `nq=2..4`, fused q4_0 P×V, compact split-K reduce, and prove it beats no-MTP at 14k with 32-token live tests.
