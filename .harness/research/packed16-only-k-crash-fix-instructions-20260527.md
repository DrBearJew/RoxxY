# Packed16-Only K Crash Fix Instructions

## Objective

Fix the packed16-only K path so llama.cpp HIP/RDNA3 can run Q8K DOT4 flash attention with:

- K cache stored only as packed16 I32 payload + F16 scales.
- No f16 K cache allocation.
- V cache still stored as normal F16.
- DOT4 FA route active for I32 K + F16 V.
- Perplexity smoke test completes without GPU segfault or invalid-argument launch failure.

The current failure happens during the first perplexity chunk after context construction and graph reserve succeed.

---

## Current Known State

Repo:

```text
/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
```

Main files involved:

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml.c
ggml/include/ggml.h
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-graph.cpp
ggml/src/ggml-cpu/ops.cpp
```

Runtime test command shape:

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github

export ROCM_PATH=/opt/rocm-7.2.3
export CC=/opt/rocm-7.2.3/llvm/bin/clang
export CXX=/opt/rocm-7.2.3/llvm/bin/clang++

export GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1
export GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
export GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1
export GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1

./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt
```

Observed status:

- Context construction succeeds.
- Graph reserve succeeds.
- KV buffer reports about 50 MiB at c512 in the current inflated layout.
- DOT4 route is selected.
- Registry lookup for packed16 tensors succeeds.
- Q quant launch succeeds.
- I32 K repack must be skipped.
- Failure remains during first compute chunk.

---

## High-Level Root Cause

The implementation currently has two different notions of packed16 K:

1. **GGML tensor identity and shape** used by graph/scheduler/route checks.
2. **Raw kernel payload layout** used by the DOT4 kernel.

The crash is most likely caused by one of these mismatches:

- The FA op receives an I32 K tensor with `ne[0] = D/4`, but some dispatch/selector/check path still assumes logical K width `D`.
- The DOT4 dispatch correctly receives `k_payload.ptr`, but derives `nk`, `n_heads_k`, `batch`, or mask dimensions from a K view that is not exactly the shape the kernel expects.
- The pack op writes the packed16 payload through one view, while FA reads it through another view/registry side channel; graph dependency may be invisible or incomplete.
- The standard non-DOT4 FA fallback must never run with I32 K; I32 K is only valid for the DOT4 path.

The fix should make these contracts explicit and fail fast if violated.

---

# Fix Plan

## Phase 1 — Make I32 K a strict DOT4-only format

### Goal

Never let an I32 packed16 K tensor enter generic CUDA FA, generic CPU FA, or generic `mul_mat` paths.

### Required behavior

If `K->type == GGML_TYPE_I32`, then:

- DOT4 route must be enabled.
- DOT4 route must be selected.
- `V->type` must be `GGML_TYPE_F16`.
- `Q->type` must be `GGML_TYPE_F32`.
- `Q->ne[0] == 256`.
- `K->ne[0] * 4 == Q->ne[0]`.
- `V->ne[0] == Q->ne[0]`.
- `Q->ne[2] % K->ne[2] == 0`.
- Any other route must return unsupported or abort with a clear message.

### File: `ggml/src/ggml-cuda/fattn.cu`

In `ggml_cuda_get_best_fattn_kernel()`, add an early I32 branch before the generic `switch (K->ne[0])` shape checks.

Pseudo-patch:

```cpp
#ifdef GGML_USE_HIP
    if (K->type == GGML_TYPE_I32) {
        const bool shape_ok =
            Q->type == GGML_TYPE_F32 &&
            V->type == GGML_TYPE_F16 &&
            dst->type == GGML_TYPE_F32 &&
            Q->ne[0] == 256 &&
            K->ne[0] * 4 == Q->ne[0] &&
            V->ne[0] == Q->ne[0] &&
            K->ne[1] > 0 &&
            K->ne[2] > 0 &&
            Q->ne[2] % K->ne[2] == 0;

        if (!shape_ok) {
            return BEST_FATTN_KERNEL_NONE;
        }

        if (!ggml_cuda_q8k_dot4_kq_enabled()) {
            return BEST_FATTN_KERNEL_NONE;
        }

        return BEST_FATTN_KERNEL_Q8K_DOT4_KQ;
    }
#endif
```

Important: do this before code like:

```cpp
switch (K->ne[0]) {
    case 64:
        if (V->ne[0] != K->ne[0]) return BEST_FATTN_KERNEL_NONE;
```

That generic check is wrong for I32 packed16 K because I32 K has physical width `D/4`, while V has logical width `D`.

### Also update route logging

If there is route logging for unsupported mixed KV, make sure I32 does not spam misleading mixed-KV rejection messages.

---

## Phase 2 — Add hard DOT4 dispatch contract checks

### Goal

Before launching any DOT4 kernel, validate all dimensions and pointers. Abort with a useful error instead of letting HIP segfault.

### File: `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

At the start of `ggml_cuda_flash_attn_ext_q8k_dot4_kq()`, immediately after reading `Q`, `K`, `V`, `mask`, `sinks`, add a contract block.

Pseudo-patch:

```cpp
    const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;

    if (k_is_i32_packed16) {
        GGML_ASSERT(Q->type == GGML_TYPE_F32);
        GGML_ASSERT(V->type == GGML_TYPE_F16);
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(Q->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D);
        GGML_ASSERT(K->ne[0] * 4 == Q->ne[0]);
        GGML_ASSERT(V->ne[0] == Q->ne[0]);
        GGML_ASSERT(K->ne[1] > 0);
        GGML_ASSERT(K->ne[2] > 0);
        GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
        GGML_ASSERT(K->data != nullptr);
        GGML_ASSERT(V->data != nullptr);
        GGML_ASSERT(dst->data != nullptr);
    }
```

Then after computing:

```cpp
    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;
```

add:

```cpp
    GGML_ASSERT(nq > 0);
    GGML_ASSERT(nk > 0);
    GGML_ASSERT(n_heads_q > 0);
    GGML_ASSERT(n_heads_k > 0);
    GGML_ASSERT(batch > 0);
    GGML_ASSERT(gqa_ratio > 0);
```

This turns silent GPU crashes into actionable CPU-side aborts if the graph shape is wrong.

---

## Phase 3 — Fix I32 K registry handling and skip repack

### Goal

When `K->type == GGML_TYPE_I32`, the K payload is already packed16. The DOT4 dispatch must use the registry payload/scales directly and must not run f16→packed16 or q8_0→packed16 repack kernels.

### File: `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

In the persistent packed16 cache block, enforce registry availability for I32 K.

Expected logic:

```cpp
    const bool use_packed16 = ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
    const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;
    bool skip_k_repack = false;

    if (use_packed16) {
        ggml_tensor * payload_tensor = nullptr;
        ggml_tensor * scales_tensor  = nullptr;
        llama_kv_cache_get_packed16_tensors(K->data, &payload_tensor, &scales_tensor);

        if (k_is_i32_packed16) {
            GGML_ASSERT(payload_tensor != nullptr);
            GGML_ASSERT(scales_tensor  != nullptr);
        }

        if (payload_tensor && scales_tensor) {
            k_payload.ptr = (int *) payload_tensor->data;
            k_scales.ptr  = (half *) scales_tensor->data;

            GGML_ASSERT(k_payload.ptr != nullptr);
            GGML_ASSERT(k_scales.ptr  != nullptr);

            if (k_is_i32_packed16) {
                skip_k_repack = true;
            } else {
                std::lock_guard<std::mutex> lock(s_cache_mutex);
                auto it_rows = s_cache_rows.find(K->data);
                const size_t prev_rows = it_rows == s_cache_rows.end() ? 0 : it_rows->second;
                skip_k_repack = prev_rows >= (size_t) k_rows;
            }
        } else {
            GGML_ASSERT(!k_is_i32_packed16);
            // existing hipMalloc fallback for f16/q8_0 K only
        }
    }
```

Then in the repack section:

```cpp
        if (!skip_k_repack) {
            GGML_ASSERT(K->type != GGML_TYPE_I32);
            if (K->type == GGML_TYPE_F16) {
                // f16 -> packed16
            } else {
                // q8_0 -> packed16
            }
        }
```

Never let I32 enter the old q8_0 repack branch.

---

## Phase 4 — Add explicit graph dependency from pack_k to FA

### Goal

Make the scheduler see that FA depends on the packed16 K write.

The registry alone is a side channel. Even if the raw pointers are correct, the scheduler may not know that the FA op must wait for the `GGML_OP_PACK_K_PACKED16` write.

### Preferred fix

Thread the result of `cpy_k()` into the K tensor used by attention when packed16-only K is active.

### File: `src/llama-graph.cpp`

Current pattern appears in several attention builders:

```cpp
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

const auto & kq_mask = inp->get_kq_mask();

ggml_tensor * q = q_cur;
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);

ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
```

Change to:

```cpp
ggml_tensor * k_pack_dep = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
ggml_build_forward_expand(gf, k_pack_dep);
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

const auto & kq_mask = inp->get_kq_mask();

ggml_tensor * q = q_cur;
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);

if (k != nullptr && k->type == GGML_TYPE_I32) {
    // Make FA depend on pack_k without changing the physical K view used by DOT4.
    // Use an explicit dependency op or attach k_pack_dep as an auxiliary FA source.
}
```

Now choose one of the two concrete dependency mechanisms below.

---

### Dependency mechanism A — Add auxiliary source to FA op

This is the cleanest long-term fix.

#### File: `ggml/src/ggml.c`

Modify `ggml_flash_attn_ext()` to accept an optional dependency tensor, or create a new helper:

```cpp
struct ggml_tensor * ggml_flash_attn_ext_with_dep(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * mask,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * dep,
        float scale,
        float max_bias,
        float logit_softcap) {

    struct ggml_tensor * result = ggml_flash_attn_ext(ctx, q, k, v, mask, sinks, scale, max_bias, logit_softcap);
    result->src[5] = dep;
    return result;
}
```

Only do this if `GGML_MAX_SRC` has enough slots and no existing FA code uses `src[5]` for something else.

Then in CUDA/CPU backends, ignore `src[5]` functionally. It exists only to express graph dependency.

#### File: `src/llama-graph.cpp`

When K is I32:

```cpp
if (k->type == GGML_TYPE_I32) {
    cur = build_attn_mha_with_k_dep(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il, k_pack_dep);
} else {
    cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
}
```

This is more invasive but architecturally correct.

---

### Dependency mechanism B — Use a no-op view/copy dependency

This is less invasive but more fragile.

Create a cheap no-op tensor that depends on `k_pack_dep`, and force `k` to depend on it before FA. Avoid changing FA op arity.

Possible idea:

```cpp
if (k->type == GGML_TYPE_I32) {
    ggml_tensor * dep = ggml_view_1d(ctx0, k_pack_dep, 1, 0);
    ggml_build_forward_expand(gf, dep);
    // Then use an op that forces dep to be in the graph before FA.
}
```

However, just adding `dep` to the graph may not force FA to wait unless FA consumes it. Therefore mechanism A is preferred.

---

## Phase 5 — Correct packed16 K view shape and strides

### Goal

The K tensor passed to FA must present the logical DOT4 shape:

```text
K logical view: [D/4, n_kv, n_head_kv, n_stream]
```

For Qwen 27B at c512:

```text
D/4       = 64
n_kv      = 512
n_head_kv = 4
n_stream  = 1
```

Physical payload allocation currently:

```text
payload: [64, n_head_kv * kv_size, n_stream]
       = [64, 2048, 1]
```

The 4D view should map:

```text
K[d4, kv, hk, s] -> payload[d4, hk * kv_size + kv, s]
```

So strides should be:

```text
nb[0] = sizeof(int32_t)
nb[1] = (D/4) * sizeof(int32_t)
nb[2] = kv_size * nb[1]
nb[3] = n_head_kv * kv_size * nb[1]
```

For c512:

```text
nb[0] = 4
nb[1] = 64 * 4 = 256
nb[2] = 512 * 256 = 131072
nb[3] = 4 * 512 * 256 = 524288
```

### File: `src/llama-kv-cache.cpp`

In `get_k()`, ensure I32 packed16 returns a view equivalent to:

```cpp
return ggml_view_4d(
    ctx,
    layer.k_payload,
    n_embd_head_k / 4,
    kv_size,
    n_head_kv,
    n_stream,
    row_bytes,
    row_bytes * kv_size,
    row_bytes * kv_size * n_head_kv,
    0);
```

Where:

```cpp
const size_t row_bytes = (n_embd_head_k / 4) * ggml_type_size(GGML_TYPE_I32);
```

Do not use the earlier interleaved-head stride variant:

```text
nb[1] = row_bytes * n_head_kv
nb[2] = row_bytes
```

That stride describes layout:

```text
payload row order: kv0/head0, kv0/head1, kv0/head2, kv0/head3, kv1/head0, ...
```

But the current payload row order from `pack_k` is intended to be:

```text
head0/kv0..kv511, head1/kv0..kv511, head2/kv0..kv511, head3/kv0..kv511
```

The DOT4 kernel row formula uses:

```cpp
row = (batch * n_heads_k + hk) * nk + kv;
```

That matches head-major row layout, not kv-major interleaved-head layout.

This is a critical likely bug: if `get_k()` returns strides for kv-major interleaved layout while payload is head-major, graph shape/registry may be correct but any generic view user or registry key assumptions become inconsistent.

---

## Phase 6 — Verify pack_k writes the same head-major layout DOT4 reads

### Goal

Confirm `ggml_cuda_q8k_dot4_quant_k_packed16_kernel` writes rows as:

```text
row = (batch * n_heads + head) * nk + kv
```

### File: `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

Open the kernel:

```cpp
static __global__ void ggml_cuda_q8k_dot4_quant_k_packed16_kernel(...)
```

Inside the kernel, the destination offset should be equivalent to:

```cpp
const int kv = blockIdx.x;
const int head = blockIdx.y;
const int batch = blockIdx.z;
const int row = (batch * n_heads + head) * nk + kv;

int * dst_payload_row = payload + row * (D / 4);
half * dst_scales_row = scales + row * (D / 32);
```

If the kernel instead writes:

```cpp
row = (batch * nk + kv) * n_heads + head;
```

then either:

1. Change the kernel to head-major row order, or
2. Change DOT4 FA read row formula to match kv-major interleaved-head order.

Do not mix them.

Recommended for now: **use head-major row order** because the DOT4 kernel already expects:

```cpp
row = (b * n_heads_k + hk) * nk + k;
```

---

## Phase 7 — Remove or gate temporary device sync

A `cudaDeviceSynchronize()` before FA launch is useful for diagnosis but too expensive for final code.

Keep it temporarily behind an env var:

```cpp
if (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC")) {
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

Default should be off after the graph dependency is fixed.

---

## Phase 8 — Add debug instrumentation env var

### Goal

When the crash happens, print exact dimensions/pointers once before kernel launch.

### File: `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

Add near DOT4 dispatch:

```cpp
static bool printed_i32_contract = false;
if (k_is_i32_packed16 && !printed_i32_contract &&
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
    printed_i32_contract = true;
    fprintf(stderr,
        "q8k_dot4_i32: Q type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n"
        "q8k_dot4_i32: K type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
        "q8k_dot4_i32: V type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
        "q8k_dot4_i32: payload=%p scales=%p nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d\n",
        Q->type, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3],
        K->type, K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[0], K->nb[1], K->nb[2], K->nb[3], K->data,
        V->type, V->ne[0], V->ne[1], V->ne[2], V->ne[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], V->data,
        k_payload.ptr, k_scales.ptr, nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio);
}
```

Expected print for c512 Qwen 27B:

```text
Q ne=[256,256,24,1]
K ne=[64,512,4,1]
V ne=[256,512,4,1]
nq=256 nk=512 hq=24 hk=4 batch=1 gqa=6
payload != null
scales != null
```

If any value differs, fix the graph/view before touching the kernel.

---

## Phase 9 — Verification sequence

### 1. Build

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github

ROCM_PATH=/opt/rocm-7.2.3 \
CC=/opt/rocm-7.2.3/llvm/bin/clang \
CXX=/opt/rocm-7.2.3/llvm/bin/clang++ \
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j4
```

### 2. Run debug smoke

```bash
export GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1
export GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
export GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1
export GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32=1
export GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC=1

./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt
```

### 3. Expected debug output

```text
llama_kv_cache: ROCm0 KV buffer size = 50.00 MiB
q8k_dot4_i32: Q type=F32 ne=[256,256,24,1]
q8k_dot4_i32: K type=I32 ne=[64,512,4,1]
q8k_dot4_i32: V type=F16 ne=[256,512,4,1]
payload != null
scales != null
```

### 4. Expected result

The run should reach a final PPL line. Target should be close to the previous DOT4/shadow result, roughly:

```text
Final estimate: PPL ~= 1.01 to 1.20 for the tiny smoke file
```

Do not over-interpret the tiny file PPL. The key gate is: no crash and no route fallback.

### 5. Then remove forced sync

```bash
unset GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC
```

Run again. If it crashes only without forced sync, the graph dependency is still missing.

If it crashes even with forced sync, the problem is not scheduling; it is layout, shape, pointer, or kernel indexing.

---

## Most Likely Concrete Fix

The highest-probability fix is this combination:

1. Add an early I32 branch in `ggml_cuda_get_best_fattn_kernel()` so I32 K bypasses generic `K->ne[0] == V->ne[0]` checks.
2. Enforce I32 K registry lookup and skip all K repack in DOT4 dispatch.
3. Fix `get_k()` I32 view strides to head-major layout:

```text
nb[1] = row_bytes
nb[2] = row_bytes * kv_size
nb[3] = row_bytes * kv_size * n_head_kv
```

not:

```text
nb[1] = row_bytes * n_head_kv
nb[2] = row_bytes
```

4. Add a true graph dependency from `GGML_OP_PACK_K_PACKED16` output to the FA op, preferably via an auxiliary FA source.

---

## Why the 50 MiB KV Cache Happens

At c512 with the current inflated layout:

```text
K payload: [64, 2048, 1] I32 = 64 * 2048 * 4 = 524,288 bytes = 0.50 MiB per layer
K scales : [8,  2048, 1] F16 = 8  * 2048 * 2 = 32,768 bytes  = 0.03 MiB per layer
V cache  : [256, 512, 4, 1] F16 = 256 * 512 * 4 * 2 = 1.00 MiB per layer
```

Approx per layer:

```text
1.53 MiB
```

For ~32 effective KV-bearing layers or alignment/pool effects, this reports about 50 MiB.

If V head width is actually 128 in the current model path, V is 0.50 MiB per layer and total accounting differs. Trust the tensor print from `GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32=1` over assumptions.

The current 50 MiB is acceptable for debugging. After correctness, shrink by removing the inflated `n_head_kv * kv_size` payload layout or reducing pool/view overhead.

---

## Stop Conditions

Stop and re-check assumptions if any of these occur:

1. `K->ne[2]` is not the KV head count.
2. `Q->ne[2] % K->ne[2] != 0`.
3. `K->data` is not exactly the registered payload pointer.
4. `payload_tensor->data != K->data` for I32 K.
5. `k_scales.ptr` is null.
6. DOT4 route is not selected for I32 K.
7. Generic FA, VEC FA, WMMA FA, or `mul_mat` runs with I32 K.
8. The crash persists with `cudaDeviceSynchronize()` enabled and debug dimensions are correct — then inspect the DOT4 kernel indexing directly.

---

## Final Acceptance Criteria

The fix is complete when all of these pass:

```bash
# Build
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j4

# Packed16-only DOT4 smoke
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 \
./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt
```

Required result:

- No segfault.
- No `ROCm error: invalid argument`.
- No fallback to non-DOT4 FA for I32 K.
- Final PPL line is printed.
- KV buffer remains packed16-only, with no f16 K allocation.
