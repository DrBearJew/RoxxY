# Packed16-only K cache: root-cause fix instructions

Date: 2026-05-27
Repo: `/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github`

## Objective

Finish the packed16-only K cache path so the default f16 V cache remains unchanged, the f16 K cache allocation is removed, and packed16 K becomes primary storage.

Target result:

- `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` allocates packed16 K payload/scales instead of f16 K.
- c512 KV buffer is approximately `24.50 MiB` for the 27B Q4_K_M test model.
- c512 PPL matches the known-good DOT4 FA shadow-mode result: approximately `1.0128`.
- c65536 KV buffer drops from approximately `4096 MiB` to approximately `3136 MiB`.
- No global `ggml_can_mul_mat()` behavior change.

## Non-negotiables

1. **Do not modify `ggml_can_mul_mat()` globally.**
   - A previous global I32-K exception compiled and ran but regressed PPL to `1.156`.
   - Packed16 I32 K is only valid for the gated DOT4 flash-attention path, not for generic matmul.

2. **Keep V as normal f16 for this milestone.**
   - Do not quantize V.
   - Do not require `--cache-type-k` or `--cache-type-v` flags.

3. **Treat packed16 K as physical I32 storage.**
   - Logical K head dimension: `D`.
   - Physical payload dimension: `D / 4` I32 values.
   - Physical scale dimension: `D / 32` F16 scales.
   - Do not fake packed16 as an f16-compatible tensor.

4. **Route must stay gated.**
   - Continue using the DOT4 route env gates.
   - Do not alter generic llama.cpp runtime behavior outside this route.

5. **Every code edit must preserve baseline operation with the env gate off.**
   - Baseline f16 KV cache should still allocate `32.00 MiB` at c512.

## Current failure signatures

### Failure 1: graph reserve/non-flash path

Observed signature:

```text
ggml_mul_mat(k, q)
GGML_ASSERT(ggml_can_mul_mat(k, q)) failed
```

Meaning:

- Packed16 K is an I32 tensor with `ne[0] = D / 4`.
- The non-flash attention path calls `ggml_mul_mat(k, q)`.
- Generic matmul expects `k->ne[0] == q->ne[0]`.
- For packed16 K, that is intentionally false: `k->ne[0] * 4 == q->ne[0]`.

Correct fix:

- Force flash attention when K is packed16 I32.
- Do **not** loosen `ggml_can_mul_mat()`.

### Failure 2: global `ggml_can_mul_mat()` hack

Observed signature:

```text
PPL regressed from ~1.0128 to ~1.156
```

Meaning:

- Generic matmul accepted I32 packed16 K and some non-DOT4 graph path consumed it incorrectly.
- The shape assertion was bypassed globally, so invalid graph edges became legal.

Correct fix:

- Add a flash-attention-specific K/Q compatibility helper.
- Use that helper only in `ggml_flash_attn_ext()`.

### Failure 3: input tensor null-buffer cascade

Observed signatures:

```text
set_input_k_idxs: GGML_ASSERT(buffer) failed
set_input_v_idxs: GGML_ASSERT(buffer) failed
set_input_kq_mask: GGML_ASSERT(buffer) failed
llm_graph_input_out_ids::set_input: GGML_ASSERT(buffer) failed
```

Meaning:

- The scheduler only allocates buffers for graph-referenced tensors.
- The packed16-only graph changed which input tensors are consumed.
- `k_idxs` became orphaned when `GGML_OP_PACK_K_PACKED16` did not reference it.
- Some optional input tensors are still being populated by `set_input*()` even when their graph tensor has no allocated backend buffer.

Correct fix:

- Make `GGML_OP_PACK_K_PACKED16` reference `k_idxs` as a source dependency.
- Add defensive no-op guards for optional/unallocated input tensors.
- Verify that required tensors still have consumers and buffers.

## Phase 0: clean up the tree before editing

Start from a clean tracked tree and preserve any WIP stash.

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github

git status --short
git stash list | head -5
```

If tracked files are conflicted, stop and resolve before continuing:

```bash
git diff --name-only --diff-filter=U
```

Remove stale reject files only after confirming they are not needed:

```bash
# Example only; inspect first.
ls -l ggml/src/ggml-cuda/*.rej 2>/dev/null || true
```

Known WIP source:

- `stash@{0}` was named `current WIP` during the previous recovery attempt.
- It contains the packed16 GGML op and DOT4 route work.
- If applying stash creates conflicts, prefer the known-good packed16 op implementation but re-check signatures manually.

Suggested conflict strategy:

```bash
# Inspect, do not blindly pop.
git stash show --stat stash@{0}
git stash show -p stash@{0} -- ggml/include/ggml.h | less

# Apply only when ready.
git stash apply stash@{0}

# Check unresolved conflicts.
git diff --name-only --diff-filter=U
```

For files that conflict, resolve by semantic intent, not by blindly taking `ours` or `theirs`.

## Phase 1: add flash-attention-specific K/Q compatibility

Files:

- `ggml/src/ggml.c`
- optionally `ggml/include/ggml.h` only if the helper must be exported; prefer `static inline` in `ggml.c` if only used there.

### 1.1 Add helper near `ggml_can_mul_mat()` or near `ggml_flash_attn_ext()`

Do not change `ggml_can_mul_mat()`.

Add a helper with this behavior:

```c
static inline bool ggml_can_flash_attn_ext_kq(
        const struct ggml_tensor * k,
        const struct ggml_tensor * q) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    const bool normal_kq = k->ne[0] == q->ne[0];
    const bool packed16_kq =
        k->type == GGML_TYPE_I32 &&
        k->ne[0] * 4 == q->ne[0];

    return (normal_kq || packed16_kq) &&
           (q->ne[2] % k->ne[2] == 0) &&
           (q->ne[3] % k->ne[3] == 0);
}
```

Notes:

- This helper deliberately only handles K/Q compatibility for flash attention.
- It preserves the normal broadcast checks from `ggml_can_mul_mat()`.
- The packed16 exception is restricted to `GGML_TYPE_I32` K.

### 1.2 Use helper inside `ggml_flash_attn_ext()`

Replace this assertion:

```c
GGML_ASSERT(ggml_can_mul_mat(k, q));
```

with:

```c
GGML_ASSERT(ggml_can_flash_attn_ext_kq(k, q));
```

Do not touch the `ggml_flash_attn_back()` assertion unless the packed16 path actually uses FA backward, which it does not for inference.

### 1.3 Verification for Phase 1

With shadow mode still allocating f16 K, PPL must remain correct:

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j$(nproc)

export DOT4_ENV="GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1"

timeout 180 env $DOT4_ENV ./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt 2>&1 | tee /tmp/packed16-phase1.log
```

Expected:

- Build succeeds.
- No PPL regression.
- No global matmul change.

## Phase 2: force flash attention for I32 packed16 K

File:

- `src/llama-graph.cpp`

Function:

- `llm_graph_context::build_attn_mha()`

Current pattern:

```cpp
const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
```

Replace with the equivalent of:

```cpp
const bool k_is_packed16_i32 = k->type == GGML_TYPE_I32;
const bool use_flash_attn = (cparams.flash_attn || k_is_packed16_i32) && kq_b == nullptr;
```

Why:

- `graph_reserve()` and auto-FA probing can build a graph with FA disabled.
- I32 packed16 K cannot legally enter the non-flash `ggml_mul_mat(k, q)` path.
- When K is I32 packed16, flash attention is not optional; it is the only valid path.

### 2.1 Mask cast must follow effective `use_flash_attn`

Any mask conversion that currently uses `cparams.flash_attn` must use the same effective decision if the tensor may feed this forced-FA call.

Bad pattern:

```cpp
inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
```

Correct intent:

```cpp
const bool use_flash_attn_for_mask = cparams.flash_attn || packed16_active_for_this_graph;
inp->self_kq_mask_cnv = use_flash_attn_for_mask ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
```

Implementation detail:

- If the local code does not have an easy `packed16_active_for_this_graph` variable, derive it from the actual K tensor used by the layer: `k && k->type == GGML_TYPE_I32`.
- Keep this scoped to attention graph construction; do not globally force FA for unrelated contexts.

### 2.2 Verification for Phase 2

Run with packed16-only allocation enabled after Phase 3/4 are in place. Until then, at minimum build and confirm no `ggml_mul_mat(k,q)` assertion occurs when K is I32.

Expected failure should move from `ggml_mul_mat` to input-buffer allocation if later phases are not done yet. That is progress.

## Phase 3: make `GGML_OP_PACK_K_PACKED16` depend on `k_idxs`

Files:

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`
- `src/llama-kv-cache.cpp`

Problem:

- `k_idxs` is normally consumed by `ggml_set_rows()`.
- In packed16-only mode, K is packed directly into payload/scales.
- If `GGML_OP_PACK_K_PACKED16` does not reference `k_idxs`, the scheduler can treat `k_idxs` as unused and leave `k_idxs->buffer == nullptr`.
- Then `set_input_k_idxs()` crashes before execution.

### 3.1 Change public signature

In `ggml/include/ggml.h`, change:

```c
GGML_API struct ggml_tensor * ggml_pack_k_packed16(
        struct ggml_context * ctx,
        struct ggml_tensor  * k_cur,
        struct ggml_tensor  * payload,
        struct ggml_tensor  * scales);
```

to:

```c
GGML_API struct ggml_tensor * ggml_pack_k_packed16(
        struct ggml_context * ctx,
        struct ggml_tensor  * k_cur,
        struct ggml_tensor  * payload,
        struct ggml_tensor  * scales,
        struct ggml_tensor  * k_idxs);
```

### 3.2 Change implementation

In `ggml/src/ggml.c`, change the definition accordingly.

Set sources like this:

```c
result->op     = GGML_OP_PACK_K_PACKED16;
result->src[0] = k_cur;   // source f16/f32 K for current ubatch
result->src[1] = scales;  // secondary output tensor for scales
result->src[2] = k_idxs;  // row indices dependency; keeps scheduler buffer alive
```

Add assertions:

```c
GGML_ASSERT(k_idxs);
GGML_ASSERT(k_idxs->type == GGML_TYPE_I64 || k_idxs->type == GGML_TYPE_I32);
```

Do not use `k_idxs` only as a comment; it must be present in `src[2]`.

### 3.3 Change call site in `cpy_k()`

In `src/llama-kv-cache.cpp`, function `llama_kv_cache::cpy_k(...)`, packed16 mode should call:

```cpp
return ggml_pack_k_packed16(ctx, k_cur, k_payload, k_scales, k_idxs);
```

Do not drop `k_idxs`.

If shadow mode is still desired for A/B testing, use both operations in the graph:

```cpp
ggml_tensor * set_rows = ggml_set_rows(ctx, k, k_cur, k_idxs);
ggml_tensor * pack     = ggml_pack_k_packed16(ctx, k_cur, k_payload, k_scales, k_idxs);
GGML_UNUSED(set_rows); // only if the graph has another way to keep it alive; otherwise chain/use it deliberately
return pack;
```

For final packed16-only mode, do not require f16 `k` to exist.

### 3.4 Backend handler does not need semantic `k_idxs` yet

In `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`, read the new source for dependency/synchronization:

```cpp
ggml_tensor * k_idxs = dst->src[2];
GGML_ASSERT(k_idxs != nullptr);
GGML_UNUSED(k_idxs);
```

If the quant kernel writes compact current-token rows and the payload view already points at the correct rows, `k_idxs` can remain unused initially. If not, the kernel must scatter according to `k_idxs`. Verify this carefully:

- If `payload` passed to the op is the full cache tensor, then the kernel must use `k_idxs` to scatter into cache rows.
- If `payload` passed to the op is a pre-indexed/viewed destination, then direct writes are acceptable.

The safer final implementation is to use `k_idxs` in the kernel or pass a view whose offset/shape is already row-correct.

### 3.5 Update CUDA supports-op check

In `ggml/src/ggml-cuda/ggml-cuda.cu`, ensure `GGML_OP_PACK_K_PACKED16` is accepted exactly once in each relevant switch:

1. Dispatch switch:

```cpp
case GGML_OP_PACK_K_PACKED16:
    ggml_cuda_op_pack_k_packed16(ctx, dst);
    break;
```

2. Device `supports_op` switch:

```cpp
case GGML_OP_PACK_K_PACKED16:
    {
        return op->type == GGML_TYPE_I32 &&
               (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
               op->src[1]->type == GGML_TYPE_F16 &&
               op->src[2] != nullptr &&
               (op->src[2]->type == GGML_TYPE_I64 || op->src[2]->type == GGML_TYPE_I32);
    } break;
```

Important conflict note:

- Do not add duplicate `case GGML_OP_PACK_K_PACKED16:` labels inside the same switch.
- A previous stash merge produced duplicate-case compiler errors.
- Use `rg -n "case GGML_OP_PACK_K_PACKED16" ggml/src/ggml-cuda/ggml-cuda.cu` and inspect each location.
- There should be one dispatch case and one supports-op case, not two in the same switch.

### 3.6 Verification for Phase 3

Build error to avoid:

```text
conflicting types for 'ggml_pack_k_packed16'
```

Check all signatures agree:

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
rg -n "ggml_pack_k_packed16\(" ggml/include/ggml.h ggml/src/ggml.c src/llama-kv-cache.cpp ggml/src/ggml-cuda
```

Expected:

- Declaration has 5 args.
- Definition has 5 args.
- Call site has 5 args.
- Backend reads `dst->src[2]` or at least asserts it.

Then build:

```bash
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j$(nproc)
```

## Phase 4: packed16-only K allocation and physical strides

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

### 4.1 Allocation rules

When packed16 mode is enabled:

```cpp
const bool packed16_active = has_k && getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE") && atoi(getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE")) != 0;
```

Use the semantic equivalent of:

```cpp
ggml_tensor * k = (has_k && !packed16_active)
    ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
    : nullptr;

ggml_tensor * k_payload = nullptr;
ggml_tensor * k_scales  = nullptr;

if (has_k && packed16_active) {
    GGML_ASSERT(n_embd_k_gqa % 32 == 0);
    k_payload = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_embd_k_gqa / 4,  kv_size, n_stream);
    k_scales  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa / 32, kv_size, n_stream);
    ggml_format_name(k_payload, "cache_k_payload_l%d", il);
    ggml_format_name(k_scales,  "cache_k_scales_l%d",  il);
}
```

In shadow mode, keep f16 `k` allocated as before and allocate packed16 tensors additionally. In final packed16-only mode, do not allocate f16 `k`.

### 4.2 Stream views

If packed16 is active and `k == nullptr`, build stream views from `k_payload`, not from `k`.

Physical dimensions:

- Payload stream view `ne[0] = n_embd_k_gqa / 4`.
- Payload stream view `ne[1] = kv_size`.
- Offset uses `k_payload->nb[2]`, not `k->nb[2]`.

Example intent:

```cpp
if (k_payload) {
    k_payload_stream.push_back(ggml_view_2d(
        ctx, k_payload,
        n_embd_k_gqa / 4,
        kv_size,
        k_payload->nb[1],
        s * k_payload->nb[2]));
}
```

Do the same for `k_scales` with `n_embd_k_gqa / 32`.

### 4.3 `get_k()` behavior

Where the graph asks for K cache:

- If packed16 tensors exist, return `k_payload` or the appropriate payload stream view.
- Do not return a fake f16 view.
- Do not attempt to create `ggml_view_2d()` with f16 dimensions on an I32 payload.

The returned K tensor should have:

```text
type  = GGML_TYPE_I32
ne[0] = logical_D / 4
ne[1] = n_kv
ne[2] = n_head_kv or stream/head dimension as expected by graph
ne[3] = batch/stream dimension as expected by graph
```

The DOT4 FA route must interpret this as packed16.

### 4.4 `type_k()` and size accounting

For graph compatibility, if code paths expect `type_k()` to describe the user-facing/logical type, keep it returning F16 if necessary. But memory accounting must count the actual tensors.

`size_k_bytes()` must count:

```text
payload bytes + scales bytes
```

not f16-K bytes, when packed16-only is active.

At c512 for the 27B Q4_K_M model:

- Baseline f16 K + f16 V: `32.00 MiB` KV.
- Shadow f16 K + packed16 K + f16 V: `40.50 MiB` KV.
- Packed16-only K + f16 V: `24.50 MiB` KV.

## Phase 5: route gates for I32 packed16 K + f16 V

Files:

- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh`
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

Required route behavior:

- Q: `GGML_TYPE_F32`
- K: either default f16 shadow source or primary `GGML_TYPE_I32` packed16 payload
- V: `GGML_TYPE_F16`
- dst: `GGML_TYPE_F32`
- route alias accepted: `rocm_q8k_dot4_packed16_blockfa`

### 5.1 Do not require q8_0 K or q4_0 V for this route

Old route predicates required:

```cpp
K->type == GGML_TYPE_Q8_0
V->type == GGML_TYPE_Q4_0
```

For packed16 default-KV operation, accept:

```cpp
K->type == GGML_TYPE_I32
V->type == GGML_TYPE_F16
```

For shadow mode, f16 K may still be present, but the DOT4 path should use packed16 payload via registry or op output, not repack every call.

### 5.2 Shape check

Packed16 physical K dimension must satisfy:

```cpp
K->type == GGML_TYPE_I32 && K->ne[0] * 4 == Q->ne[0]
```

Do not require `K->ne[0] == Q->ne[0]` for this route.

### 5.3 Skip generic repack when K is already I32 packed16

In the DOT4 backend path:

- If `K->type == GGML_TYPE_I32`, do not repack K.
- Look up scales through the packed16 registry or the op-provided tensor.
- If scales are missing, fail the route clearly and fall back only if a safe fallback exists.

Failure should be explicit:

```text
packed16 I32 K route selected but no scales tensor found
```

not a silent wrong-output fallback.

## Phase 6: post-allocation registry for payload/scales

Files:

- `src/llama-kv-cache.cpp`
- DOT4 backend registry location in `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu` or shared helper.

Problem:

- Before backend allocation, `tensor->data` may be null or unstable.
- Registry keyed by `K->data` must be populated after backend allocation.

Correct timing:

- After `ggml_backend_alloc_ctx_tensors_from_buft()` succeeds.
- After `k_payload->data` and `k_scales->data` are valid.

Registry entries should map:

```text
k_payload->data -> { k_payload, k_scales }
k->data         -> { k_payload, k_scales }   # only in shadow mode, if f16 k exists
```

For packed16-only mode, `k` is null, so only map `k_payload->data`.

Do not use `hipMalloc` scratch buffers for final packed16-only mode if GGML tensors already own storage. The op should populate the GGML-managed payload/scales tensors.

## Phase 7: guard optional input setters without hiding real missing dependencies

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-graph.cpp`

Goal:

- Avoid crashes when an input tensor object exists but scheduler did not allocate it because the graph did not consume it.
- Do not silently skip a required input tensor.

### 7.1 Add a helper or local guard

Suggested helper pattern:

```cpp
static bool llama_input_tensor_has_host_buffer(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer) && t->data;
}
```

If no shared helper is desired, use local guards.

### 7.2 Guard `set_input_k_idxs()`

In `llama_kv_cache::set_input_k_idxs(...)`, before asserting host buffer:

```cpp
if (!dst || !dst->buffer) {
    return;
}
```

Then keep the existing assert:

```cpp
GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
```

Rationale:

- If `k_idxs` is unused by the actual graph, no input needs to be populated.
- If packed16 K packing needs `k_idxs`, Phase 3 should make it a graph dependency and the buffer will exist.
- Therefore a skipped `k_idxs` after Phase 3 is a sign that the op is not connected; investigate if correctness fails.

### 7.3 Guard `set_input_v_idxs()`

Same pattern:

```cpp
if (!dst || !dst->buffer) {
    return;
}
```

Keep the assert after the guard.

### 7.4 Guard `set_input_kq_mask()`

Same pattern:

```cpp
if (!dst || !dst->buffer) {
    return;
}
```

Important:

- If FA requires a mask for the current graph, this tensor should be consumed and allocated.
- If this guard fires and PPL changes, the mask path is broken. Fix graph references rather than keeping the guard as a workaround.

### 7.5 Guard `llm_graph_input_out_ids::set_input()`

In `src/llama-graph.cpp`, function `llm_graph_input_out_ids::set_input(...)`, before asserting host buffer:

```cpp
if (!out_ids || !out_ids->buffer) {
    return;
}
```

Keep the host-buffer assert after the guard.

Important:

- This is safe only when output IDs are not consumed by the graph.
- If `n_outputs != n_tokens`, `out_ids` is usually required. If guard fires in that case, prefer to keep `out_ids` referenced in the graph rather than silently skipping.

Suggested stricter guard:

```cpp
if (!out_ids || !out_ids->buffer) {
    GGML_ASSERT(n_outputs == ubatch->n_tokens && "out_ids missing buffer but output compaction is required");
    return;
}
```

Use the stricter version if it compiles with available local variables.

### 7.6 Add debug logging only if needed

Do not spam normal runs. If diagnosing, use a temporary debug log under the DOT4 env gate:

```cpp
if (getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE")) {
    LLAMA_LOG_DEBUG("%s: skipping unallocated optional input tensor\n", __func__);
}
```

Remove or downgrade logs before final commit.

## Phase 8: verification matrix

### 8.1 Build

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
ROCM_PATH=/opt/rocm-7.2.3 \
CC=/opt/rocm-7.2.3/llvm/bin/clang \
CXX=/opt/rocm-7.2.3/llvm/bin/clang++ \
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j$(nproc)
```

Expected:

```text
Built target llama-perplexity
```

### 8.2 Baseline with env gate off

```bash
timeout 180 ./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt 2>&1 | tee /tmp/p16-baseline-c512.log
```

Expected:

```text
llama_kv_cache:      ROCm0 KV buffer size =    32.00 MiB
```

PPL should match previous baseline for the same test file.

### 8.3 Packed16-only c512

```bash
export DOT4_ENV="GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1"

timeout 240 env $DOT4_ENV ./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt 2>&1 | tee /tmp/p16-packedonly-c512.log
```

Expected:

```text
llama_kv_cache:      ROCm0 KV buffer size =    24.50 MiB
Final estimate: PPL = ~1.0128
```

Acceptable:

- Minor throughput variation.
- PPL same as known-good DOT4 FA shadow mode within normal numeric noise.

Reject:

- NaN PPL.
- PPL around `1.156`.
- Any `GGML_ASSERT(buffer) failed`.
- Any fallback that reports route requirement failure under `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa`.

### 8.4 Shadow-mode cross-check

If packed16-only fails, temporarily re-enable f16 K allocation while keeping packed16 tensors.

Expected:

```text
llama_kv_cache:      ROCm0 KV buffer size =    40.50 MiB
Final estimate: PPL = ~1.0128
```

If shadow mode is correct but packed16-only fails:

- Allocation and graph dimension semantics are still wrong.
- Focus on `get_k()`, physical strides, and route shape handling.

If shadow mode also fails:

- The op, route gate, registry, or kernel has regressed.
- Do not debug packed16-only allocation until shadow mode is correct again.

### 8.5 c65536 VRAM verification

Run only after c512 PPL is correct.

Use the same DOT4 env and a long-context command with `-c 65536`.

Expected KV memory:

```text
baseline f16 K + f16 V:      ~4096 MiB
packed16 K + f16 V only:    ~3136 MiB
savings:                    ~960 MiB
```

Stop if c512 correctness is not already proven.

## Phase 9: troubleshooting map

### Build: duplicate `GGML_OP_PACK_K_PACKED16` case

Cause:

- Stash conflict or manual edit inserted the same case twice in one switch.

Fix:

```bash
rg -n "case GGML_OP_PACK_K_PACKED16" ggml/src/ggml-cuda/ggml-cuda.cu
```

There should be:

- One case in compute dispatch.
- One case in device supports-op.

No switch may contain duplicate case labels.

### Build: conflicting `ggml_pack_k_packed16` types

Cause:

- Header, C implementation, and call sites disagree on 4 vs 5 parameters.

Fix:

```bash
rg -n "ggml_pack_k_packed16\(" ggml/include/ggml.h ggml/src/ggml.c src/llama-kv-cache.cpp
```

All must use the 5-argument form:

```text
ctx, k_cur, payload, scales, k_idxs
```

### Runtime: `ggml_mul_mat(k,q)` assertion

Cause:

- I32 packed16 K entered non-flash path.

Fix:

- Re-check `build_attn_mha()` effective `use_flash_attn`.
- Re-check mask cast uses effective FA state.
- Re-check no graph reserve path still builds non-flash attention for I32 K.

### Runtime: `GGML_ASSERT(buffer) failed` in input setters

Cause:

- Tensor object exists but scheduler did not allocate a buffer because graph does not consume it.

Fix order:

1. Verify `GGML_OP_PACK_K_PACKED16` has `k_idxs` in `src[2]`.
2. Verify CUDA supports-op accepts `src[2]` index type.
3. Add/verify no-op guards in input setters.
4. Confirm PPL; if wrong, a required input was skipped and must be reconnected to the graph.

### Runtime: route requirement failure

Cause:

- DOT4 route predicates do not accept current K/V types or shape.

Fix:

- For packed16-only: accept `K->type == GGML_TYPE_I32` and `K->ne[0] * 4 == Q->ne[0]`.
- Accept `V->type == GGML_TYPE_F16`.
- Ensure route alias `rocm_q8k_dot4_packed16_blockfa` is recognized.

### Runtime: PPL ~1.156

Cause:

- Generic matmul likely accepted packed16 I32 K or an invalid fallback path ran.

Fix:

- Revert any `ggml_can_mul_mat()` global change.
- Require route alias during testing.
- Fail hard if packed16 route cannot run.

### Runtime: NaN PPL

Cause candidates:

- scales tensor not found or uninitialized.
- pack op writes wrong rows.
- kernel physical stride mismatch.
- mask not cast to F16 for forced FA.

Fix order:

1. Verify registry maps `k_payload->data` to `k_scales` after backend allocation.
2. Verify `GGML_OP_PACK_K_PACKED16` writes scales for every packed row.
3. Verify physical K strides use I32 row size and `D / 4` dimension.
4. Verify mask dtype is F16 for FA.

## Final acceptance checklist

- [ ] Tracked tree has no merge conflicts.
- [ ] `ggml_can_mul_mat()` unchanged from baseline.
- [ ] `ggml_can_flash_attn_ext_kq()` or equivalent helper exists and is used only by `ggml_flash_attn_ext()`.
- [ ] `build_attn_mha()` forces FA for `GGML_TYPE_I32` K.
- [ ] FA mask conversion follows effective FA state, not only `cparams.flash_attn`.
- [ ] `ggml_pack_k_packed16()` has 5-argument signature everywhere.
- [ ] `GGML_OP_PACK_K_PACKED16` sets `src[2] = k_idxs`.
- [ ] CUDA backend accepts `GGML_OP_PACK_K_PACKED16` once in dispatch and once in supports-op.
- [ ] Packed16-only allocation does not allocate f16 K.
- [ ] `get_k()` returns physical I32 payload tensor/view when packed16 is active.
- [ ] scales registry is populated after backend allocation.
- [ ] optional input setters no-op on null/unallocated tensors, with stricter checks where needed.
- [ ] Baseline env-off c512 still shows `32.00 MiB` KV.
- [ ] Packed16-only c512 shows `24.50 MiB` KV.
- [ ] Packed16-only c512 PPL is approximately `1.0128`.
- [ ] c65536 KV buffer is approximately `3136 MiB`.

## Recommended commit message

```text
ROCm DOT4: make packed16 K primary cache storage

- add flash-attn-specific K/Q compatibility for I32 packed16 K
- force FA path for packed16 K to avoid invalid generic matmul
- thread k_idxs through GGML_OP_PACK_K_PACKED16 for scheduler deps
- allocate packed16 payload/scales as primary K cache under env gate
- guard optional graph input setters for unallocated tensors
- preserve baseline f16 KV behavior when gate is off
```
