# Packed16-Only K NaN Fix Instructions

## Objective

Fix the remaining packed16-only K correctness failure:

```text
DOT4 FA launches and completes, but perplexity prints NaN.
```

The previous crash is fixed. The current problem is **incorrect K cache layout/indexing**, not kernel launch validity.

Target final behavior:

- Packed16-only K remains active.
- Flash Attention remains active.
- DOT4 route is selected for I32 K + F16 V.
- Perplexity smoke completes with finite PPL.
- KV cache at c512 drops from current inflated ~50 MiB to expected ~24.5 MiB for the current 16 KV-cache layers.

---

## Current Working State

Confirmed debug output after the crash fix:

```text
get_k I32: kp ne=[256,2048,1,1] nb=1024 d4_per_head=64 n_kv=512 n_head_kv=4 ns=1 payload_row_bytes=1024
q8k_dot4_i32: Q type=0 ne=[256,256,24,1] nb=[4,24576,1024,6291456]
q8k_dot4_i32: K type=26 ne=[64,256,4,1] nb=[4,1024,262144,1048576]
q8k_dot4_i32: V type=1 ne=[256,256,4,1] nb=[2,2048,512,1048576]
q8k_dot4_i32: payload=... scales=... nq=256 nk=256 hq=24 hk=4 batch=1 gqa=6
```

This means:

- DOT4 FA is now reached.
- I32 K route works.
- The old segfault/invalid-argument issue is gone.
- Output is still wrong: `PPL = NaN`.

---

## Actual Root Cause

There are two related layout bugs.

### Bug 1 — Payload/scales are allocated too wide

Current allocation uses total GQA width:

```cpp
k_payload = ggml_new_tensor_3d(ctx, GGML_TYPE_I32,
        n_embd_k_gqa / 4,
        kv_size * n_head_kv,
        n_stream);

k_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
        n_embd_k_gqa / 32,
        kv_size * n_head_kv,
        n_stream);
```

For Qwen3.6-27B:

```text
n_embd_head_k = 256
n_head_kv     = 4
n_embd_k_gqa  = 1024
```

So current payload becomes:

```text
k_payload ne[0] = 1024 / 4 = 256 I32 values
k_scales  ne[0] = 1024 / 32 = 32 F16 values
```

But DOT4 per-head K needs:

```text
D/4  = 256 / 4  = 64 I32 values per KV head row
D/32 = 256 / 32 = 8  F16 scales per KV head row
```

So current payload/scales are **4x too wide**.

This explains the current ~50 MiB KV cache:

```text
K payload: 256 * 2048 * 4 bytes = 2.00 MiB/layer
K scales : 32  * 2048 * 2 bytes = 0.125 MiB/layer
V cache  : 1024 * 512 * 2 bytes = 1.00 MiB/layer
Total    : ~3.125 MiB/layer * 16 layers = ~50 MiB
```

Correct compact packed16-only K should be:

```text
K payload: 64 * 2048 * 4 bytes = 0.50 MiB/layer
K scales : 8  * 2048 * 2 bytes = 0.03125 MiB/layer
V cache  : 1024 * 512 * 2 bytes = 1.00 MiB/layer
Total    : ~1.53125 MiB/layer * 16 layers = ~24.5 MiB
```

---

### Bug 2 — Pack/read indexing uses active `nk`, not fixed KV cache capacity

The persistent K cache must use **absolute KV cache cell indices** and a **fixed per-head stride equal to `kv_size`**, not the current active `nk`.

Current pack kernel writes rows using:

```cpp
row = ((batch * n_heads_k + hk) * nk + k);
```

Current DOT4 FA kernel reads rows using the same active `nk` formula:

```cpp
k_head_base = ((b * n_heads_k + hk) * nk);
row = k_head_base + k;
```

This only works inside one isolated temporary K block.

It is wrong for persistent KV cache because `nk` changes over time:

- First ubatch: `nk = 256`
- Second ubatch: `nk = 512`
- Later context: `nk` grows further

So head base shifts as context grows. Head 1 data written when `nk=256` is later read as if it was written with `nk=512`.

Also, `k_idxs` was added as `src[2]`, but the CUDA pack kernel currently does **not** use it. It writes current-batch rows densely from zero instead of writing into absolute KV cache slots.

This is the most likely direct cause of NaN.

---

# Correct Canonical Layout

Use one simple persistent layout everywhere:

```text
logical packed16 K payload: [D/4,  kv_size * n_head_kv, n_stream]
logical packed16 K scales : [D/32, kv_size * n_head_kv, n_stream]
```

For Qwen3.6-27B at c512:

```text
payload: [64, 512 * 4 = 2048, 1]
scales : [8,  512 * 4 = 2048, 1]
```

Row mapping:

```text
row = stream * (n_head_kv * kv_size) + head * kv_size + kv_cell
```

or for current single-stream path:

```cpp
row = head * kv_size + kv_cell;
```

Where:

- `head` is KV head index, `0..n_head_kv-1`
- `kv_cell` is absolute cache slot from `k_idxs[k]`
- `kv_size` is cache capacity/cells, not active `nk`

Do not use active `nk` as the head stride for persistent storage.

---

# Required Fixes

## Fix 1 — Allocate compact per-head payload/scales

### File

```text
src/llama-kv-cache.cpp
```

Find the packed16 K allocation near the current lines around 260-275.

Current buggy pattern:

```cpp
k_payload = ggml_new_tensor_3d(ctx, GGML_TYPE_I32,
        n_embd_k_gqa / 4,
        kv_size * n_head_kv,
        n_stream);

k_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
        n_embd_k_gqa / 32,
        kv_size * n_head_kv,
        n_stream);
```

Replace with per-head dimensions:

```cpp
const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
const int64_t k_payload_d4  = n_embd_head_k / 4;
const int64_t k_scales_d32  = n_embd_head_k / 32;

GGML_ASSERT(n_embd_head_k % 32 == 0);

k_payload = ggml_new_tensor_3d(ctx, GGML_TYPE_I32,
        k_payload_d4,
        kv_size * n_head_kv,
        n_stream);

k_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
        k_scales_d32,
        kv_size * n_head_kv,
        n_stream);
```

Expected debug after this fix:

```text
get_k I32: kp ne=[64,2048,1,1]
```

not:

```text
get_k I32: kp ne=[256,2048,1,1]
```

---

## Fix 2 — Correct `size_k_bytes()` accounting

### File

```text
src/llama-kv-cache.cpp
```

Wherever `size_k_bytes()` or equivalent K-size accounting sums packed16 sizes, make sure it uses actual tensor sizes:

```cpp
if (layer.k_payload) size += ggml_nbytes(layer.k_payload);
if (layer.k_scales)  size += ggml_nbytes(layer.k_scales);
```

Do not compute packed16 K size from `n_embd_k_gqa / 4` manually after Fix 1.

Expected log at c512 should move from about 50 MiB to about 24.5 MiB.

---

## Fix 3 — Return a compact, head-major I32 K view

### File

```text
src/llama-kv-cache.cpp
```

In `llama_kv_cache::get_k()` packed16-only branch, after Fix 1, payload `kp->ne[0]` should already be `D/4` per head.

Use this view:

```cpp
if (!k && layers[ikv].k_payload) {
    auto * kp = layers[ikv].k_payload;
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const int64_t n_head_kv = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    const int64_t d4_per_head = n_embd_head_k / 4;

    GGML_ASSERT(kp->type == GGML_TYPE_I32);
    GGML_ASSERT(kp->ne[0] == d4_per_head);

    const size_t row_bytes = kp->nb[1]; // d4_per_head * sizeof(int32_t)

    return ggml_view_4d(ctx, kp,
            d4_per_head,
            n_kv,
            n_head_kv,
            ns,
            row_bytes,                                      // nb[1]: next KV cell in same head
            row_bytes * (size_t) get_size(),                // nb[2]: next head, fixed kv_size stride
            row_bytes * (size_t) get_size() * n_head_kv,    // nb[3]: next stream
            row_bytes * (size_t) get_size() * n_head_kv * (size_t) sinfo.s0);
}
```

Important details:

- `ne[0]` must be `D/4`, not `n_embd_k_gqa/4`.
- Head stride must use fixed cache capacity `get_size()`, not current `n_kv`.
- Stream offset should use the fixed stream stride.

For c512, expected K view:

```text
K ne=[64, active_n_kv, 4, 1]
K nb=[4, 256, 131072, 524288]
```

Where:

```text
nb[1] = 64 * 4 = 256 bytes
nb[2] = 256 * 512 = 131072 bytes
nb[3] = 131072 * 4 = 524288 bytes
```

---

## Fix 4 — Use `k_idxs` in f16→packed16 K pack kernel

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

Current kernel writes densely using current `nk`:

```cpp
const size_t row = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
```

This must be replaced with absolute cache slot indexing.

### Add a templated kernel variant

Use `k_idxs` as either I32 or I64.

Pseudo-code:

```cpp
template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel(
        const half * __restrict__ K,
        int        * __restrict__ k_payload,
        half       * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size) {

    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;

    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const int64_t cell = (int64_t) k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) {
        return;
    }

    const half * k_ptr = (const half *) ((const char *) K
            + int64_t(b)  * nb03
            + int64_t(hk) * nb02
            + int64_t(k_local) * nb01);

    const int d = q_block * QK8_0 + lane;
    const float x = __half2float(k_ptr[d]);

    float amax = fabsf(x);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset));
    }

    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);

    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;

    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }
}
```

### Update the CUDA op handler

In:

```cpp
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
```

Current handler ignores `k_idxs` except for lifetime:

```cpp
GGML_UNUSED(k_idxs);
```

Remove that and dispatch by index type.

Compute fixed cache capacity:

```cpp
const int nk_cur = (int) k_cur->ne[1];
const int n_heads = (int) k_cur->ne[2];
const int batch = (int) k_cur->ne[3];
const int kv_size = (int) (payload->ne[1] / n_heads);

GGML_ASSERT(payload->type == GGML_TYPE_I32);
GGML_ASSERT(scales->type  == GGML_TYPE_F16);
GGML_ASSERT(payload->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 4);
GGML_ASSERT(scales->ne[0]  == GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
GGML_ASSERT(kv_size >= nk_cur);
```

Then:

```cpp
dim3 grid(nk_cur, n_heads, batch);
dim3 block(256);
cudaStream_t stream = ctx.stream();

if (k_idxs->type == GGML_TYPE_I64) {
    ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
        (const half *) k_cur->data,
        (int *) payload->data,
        (half *) scales->data,
        (const int64_t *) k_idxs->data,
        k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
        nk_cur, n_heads, batch, kv_size);
} else {
    ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
        (const half *) k_cur->data,
        (int *) payload->data,
        (half *) scales->data,
        (const int32_t *) k_idxs->data,
        k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
        nk_cur, n_heads, batch, kv_size);
}
CUDA_CHECK(cudaGetLastError());
```

This is mandatory. Without it, second ubatch/chunk reads stale or wrong K rows.

---

## Fix 5 — Read DOT4 K with fixed head stride, not active `nk`

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

The v4 single kernel currently computes:

```cpp
const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
```

This must use fixed cache stride derived from the K view:

```cpp
const size_t k_head_base = size_t(b) * size_t(k_batch_stride_rows)
                         + size_t(hk) * size_t(k_head_stride_rows);
```

Then each K cell is read as:

```cpp
row = k_head_base + k;
```

where `k` is active absolute/visible KV position `0..nk-1`.

### Dispatch-side derivation

In `ggml_cuda_flash_attn_ext_q8k_dot4_kq()`, after `nk`, `n_heads_k`, etc. are computed, derive row strides from K view strides:

```cpp
const int k_payload_row_i32 = GGML_CUDA_Q8K_DOT4_KQ_D / 4;
GGML_ASSERT(K->nb[1] % sizeof(int) == 0);
GGML_ASSERT(K->nb[2] % K->nb[1] == 0);
GGML_ASSERT(K->nb[3] % K->nb[1] == 0);
GGML_ASSERT((int)(K->nb[1] / sizeof(int)) == k_payload_row_i32);

const int k_head_stride_rows  = (int)(K->nb[2] / K->nb[1]);
const int k_batch_stride_rows = (int)(K->nb[3] / K->nb[1]);

GGML_ASSERT(k_head_stride_rows >= nk);
GGML_ASSERT(k_batch_stride_rows >= k_head_stride_rows * n_heads_k);
```

For c512 first ubatch, expected:

```text
nk = 256
k_head_stride_rows = 512
k_batch_stride_rows = 2048
```

Pass these two new ints to `ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel`.

### Kernel signature change

Add parameters to the v4 single kernel:

```cpp
int k_head_stride_rows,
int k_batch_stride_rows,
```

Then replace:

```cpp
const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
```

with:

```cpp
const size_t k_head_base = size_t(b) * size_t(k_batch_stride_rows)
                         + size_t(hk) * size_t(k_head_stride_rows);
```

Do this for all v4 single launch variants:

```text
<true, 8, 16>
<false, 8, 16>
<true, 16, 8>
<false, 16, 8>
<true, 8, 8>
<false, 8, 8>
```

### Important

For I32 K, abort if a non-v4 DOT4 variant is selected until all variants are updated:

```cpp
if (k_is_i32_packed16 && !blockfa_recthist_v4_single) {
    GGML_ABORT("I32 packed16 K currently requires blockfa_recthist_v4_single");
}
```

---

## Fix 6 — Keep registry path strict for I32 K

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

For I32 packed16 K:

- Registry lookup must succeed.
- Do not allocate hipMalloc fallback.
- Do not repack in FA dispatch.
- `k_payload.ptr` and `k_scales.ptr` must point to the KV cache tensors.

Expected logic:

```cpp
const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;

llama_kv_cache_get_packed16_tensors(K->data, &payload_tensor, &scales_tensor);

if (k_is_i32_packed16) {
    GGML_ASSERT(payload_tensor != nullptr);
    GGML_ASSERT(scales_tensor  != nullptr);
    GGML_ASSERT(payload_tensor->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    GGML_ASSERT(scales_tensor->ne[0]  == GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    k_payload.ptr = (int *) payload_tensor->data;
    k_scales.ptr  = (half *) scales_tensor->data;
    skip_k_repack = true;
}
```

Never enter this branch for I32 K:

```cpp
// q8_0 -> packed16 repack
```

---

## Fix 7 — Keep temporary sync gated

A device sync is useful while debugging but should not be unconditional.

Use:

```cpp
if (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC")) {
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

Validation order:

1. First validate finite PPL with `GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC=1`.
2. Then unset it and test again.
3. If finite only with sync enabled, add explicit graph dependency from `GGML_OP_PACK_K_PACKED16` output to FA op.

---

## Fix 8 — Add/keep debug prints behind env vars only

Do not leave unconditional prints in final code.

Use env gate:

```cpp
GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32=1
```

Print once:

```text
get_k I32: kp ne=[64,2048,1,1]
q8k_dot4_i32: K ne=[64,256,4,1] nb=[4,256,131072,524288]
q8k_dot4_i32: nk=256 k_head_stride_rows=512 k_batch_stride_rows=2048
```

Expected c512 values after all fixes:

```text
payload tensor: ne=[64,2048,1,1]
scales tensor : ne=[8,2048,1,1]
K view first ubatch: ne=[64,256,4,1]
K view second ubatch: ne=[64,512,4,1]
K nb=[4,256,131072,524288]
```

---

# Verification Commands

## Build

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github

ROCM_PATH=/opt/rocm-7.2.3 \
CC=/opt/rocm-7.2.3/llvm/bin/clang \
CXX=/opt/rocm-7.2.3/llvm/bin/clang++ \
cmake --build build-rocm-rdna3-fa --target llama-perplexity -j4
```

## Debug smoke with forced sync

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github

GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC=1 \
./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 -fit off \
  -f /tmp/test_2048.txt
```

Expected:

```text
llama_kv_cache: ROCm0 KV buffer size = ~24.50 MiB
q8k_dot4_i32: ... finite dimensions ...
[1]<finite>,[2]<finite>,[3]<finite>
```

Failure states:

- If KV buffer is still ~50 MiB: Fix 1 is incomplete.
- If K view has `ne[0]=256`: Fix 3 is incomplete.
- If `k_head_stride_rows == nk` on first ubatch: Fix 5 is incomplete; should be fixed capacity 512.
- If PPL is still NaN: dump first few packed K scales/quants and compare against f16 shadow path.
- If only forced-sync works: add explicit graph dependency.

## Final smoke without forced sync

```bash
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

Expected:

- No crash.
- No `ROCm error: invalid argument`.
- No `Flash Attention was auto, set to disabled`.
- No NaN.
- Final PPL is finite.

---

# Minimal Patch Order

Apply in this order:

1. Compact payload/scales allocation: `[64, kv_size*n_head_kv]` and `[8, kv_size*n_head_kv]`.
2. Fix `get_k()` I32 view: `ne[0]=D/4`, fixed `kv_size` head stride.
3. Update pack_k kernel to use `k_idxs` absolute cache cells and fixed `kv_size` head stride.
4. Update v4 DOT4 kernel to read K with fixed `k_head_stride_rows`, not active `nk`.
5. Keep I32 registry strict and skip all FA-dispatch repack.
6. Run with forced sync.
7. Run without forced sync; if it regresses, add explicit graph dependency.

This sequence directly addresses the NaN-producing inconsistency.
