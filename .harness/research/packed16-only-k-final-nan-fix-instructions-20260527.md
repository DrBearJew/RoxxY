# Packed16-Only K Final NaN Fix Instructions

## Goal

Fix the remaining packed16-only K correctness issue:

```text
DOT4 FA launches and completes, KV cache is compact at 24.50 MiB, but perplexity prints NaN.
```

Current state after the last patch round:

- Crash is fixed.
- DOT4 route is active.
- K cache is compact: about **24.50 MiB** at c512.
- K payload allocation is now per-head compact: expected `[64, 2048, 1]` for Qwen3.6-27B.
- `get_k()` returns an I32 view with `K->ne[0] = 64`.
- `pack_k` uses `k_idxs` and fixed `kv_size` stride.
- v4 DOT4 kernel reads K using fixed `k_head_stride_rows`.
- Remaining failure: output logits become NaN.

The next likely bug is **source K layout handling in the pack_k CUDA op**.

---

## Short Diagnosis

The compact packed16 cache now stores per-head rows:

```text
payload row width = D/4 = 64 I32 values
scales row width  = D/32 = 8 F16 values
row = head * kv_size + kv_cell
```

But the source tensor `k_cur` passed into `GGML_OP_PACK_K_PACKED16` may not be shaped as:

```text
[D, n_tokens, n_head_kv, batch]
```

It is likely shaped as a **GQA-combined K vector**:

```text
[n_embd_k_gqa, n_tokens, 1, batch]
```

For Qwen3.6-27B:

```text
n_embd_k_gqa = 1024
D            = 256
n_head_kv    = 4
```

So `k_cur->ne[0]` may be `1024`, not `256`, and `k_cur->ne[2]` may be `1`, not `4`.

If the pack handler does this:

```cpp
const int n_heads = (int) k_cur->ne[2];
const int kv_size = (int) (payload->ne[1] / n_heads);
```

then it infers:

```text
n_heads = 1
kv_size = 2048
```

instead of:

```text
n_heads = 4
kv_size = 512
```

That means only one KV head is packed correctly, and heads 1-3 are stale/zero/uninitialized. DOT4 FA then reads four heads and produces invalid logits/NaN.

This is the highest-probability remaining root cause.

---

# Fix Plan

## Fix 1 — Make pack_k infer KV heads from source width, not only `k_cur->ne[2]`

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

### Function

```cpp
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
```

### Problematic logic

Do not rely on this alone:

```cpp
const int n_heads = (int) k_cur->ne[2];
const int kv_size = (int) (payload->ne[1] / n_heads);
```

This fails when `k_cur` stores all KV heads combined in `ne[0]`.

### Correct logic

Use `D = GGML_CUDA_Q8K_DOT4_KQ_D` and infer whether K source is split-head or combined-head:

```cpp
const int D = GGML_CUDA_Q8K_DOT4_KQ_D;
const int nk_cur = (int) k_cur->ne[1];
const int batch  = (int) k_cur->ne[3];

int n_heads = 0;
int64_t src_head_stride_bytes = 0;

if (k_cur->ne[0] == D) {
    // Split-head source layout: [D, nk_cur, n_heads, batch]
    n_heads = (int) k_cur->ne[2];
    src_head_stride_bytes = k_cur->nb[2];
} else {
    // Combined GQA source layout: [D * n_heads, nk_cur, 1, batch]
    GGML_ASSERT(k_cur->ne[0] % D == 0);
    n_heads = (int) (k_cur->ne[0] / D);
    src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(k_cur->type);
}

GGML_ASSERT(n_heads > 0);
GGML_ASSERT(payload->ne[1] % n_heads == 0);

const int kv_size = (int) (payload->ne[1] / n_heads);

GGML_ASSERT(kv_size >= nk_cur);
GGML_ASSERT(payload->ne[0] == D / 4);
GGML_ASSERT(scales->ne[0]  == D / QK8_0);
```

For Qwen3.6-27B, expected values:

```text
D = 256
k_cur->ne[0] = 1024
n_heads = 4
src_head_stride_bytes = 512 bytes
payload->ne[1] = 2048
kv_size = 512
```

`src_head_stride_bytes = D * sizeof(half) = 256 * 2 = 512`.

---

## Fix 2 — Pass source-head stride to indexed pack kernel

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

### Current indexed kernel source pointer pattern

If the kernel currently does this:

```cpp
const half * k_ptr = (const half *) ((const char *) K
        + int64_t(b)  * nb03
        + int64_t(hk) * nb02
        + int64_t(k_local) * nb01);
```

that is only correct for split-head layout.

### Required signature change

Add this parameter to the indexed kernel:

```cpp
int64_t src_head_stride_bytes
```

Full relevant signature:

```cpp
template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel(
        const half  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size)
```

`nb02` can remain in the signature for compatibility/debug, but the kernel should use `src_head_stride_bytes` for head indexing.

### Required source pointer

Replace the source pointer computation with:

```cpp
const half * k_ptr = (const half *) ((const char *) K
        + int64_t(b)       * nb03
        + int64_t(k_local) * nb01
        + int64_t(hk)      * src_head_stride_bytes);
```

This works for both layouts:

1. Split-head source:
   ```text
   src_head_stride_bytes = k_cur->nb[2]
   ```
2. Combined GQA source:
   ```text
   src_head_stride_bytes = D * sizeof(half)
   ```

Then keep the existing quantization loop:

```cpp
const int d = q_block * QK8_0 + lane;
const float x = __half2float(k_ptr[d]);
```

---

## Fix 3 — Update pack handler launches

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

In `ggml_cuda_op_pack_k_packed16()`, after inferring `n_heads`, `kv_size`, and `src_head_stride_bytes`, pass the new parameter to both I64 and I32 kernel launches.

### I64 launch

```cpp
ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
    (const half *) k_cur->data,
    (int *) payload->data,
    (half *) scales->data,
    (const int64_t *) k_idxs->data,
    k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
    src_head_stride_bytes,
    nk_cur, n_heads, batch, kv_size);
```

### I32 launch

```cpp
ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
    (const half *) k_cur->data,
    (int *) payload->data,
    (half *) scales->data,
    (const int32_t *) k_idxs->data,
    k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
    src_head_stride_bytes,
    nk_cur, n_heads, batch, kv_size);
```

---

## Fix 4 — Add one-shot pack_k debug print

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

In `ggml_cuda_op_pack_k_packed16()`, add an env-gated one-shot print.

```cpp
static bool pack_debug_printed = false;
if (!pack_debug_printed && ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
    pack_debug_printed = true;
    fprintf(stderr,
        "pack_k_i32: k_cur type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
        "payload ne=[%lld,%lld,%lld,%lld] scales ne=[%lld,%lld,%lld,%lld] "
        "n_heads=%d nk_cur=%d kv_size=%d src_head_stride=%lld\n",
        k_cur->type, k_cur->ne[0], k_cur->ne[1], k_cur->ne[2], k_cur->ne[3],
        k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
        payload->ne[0], payload->ne[1], payload->ne[2], payload->ne[3],
        scales->ne[0], scales->ne[1], scales->ne[2], scales->ne[3],
        n_heads, nk_cur, kv_size, (long long) src_head_stride_bytes);
}
```

Expected for Qwen3.6-27B compact path:

```text
pack_k_i32: k_cur ne=[1024,256,1,1] ... payload ne=[64,2048,1,1] scales ne=[8,2048,1,1] n_heads=4 nk_cur=256 kv_size=512 src_head_stride=512
```

If it prints:

```text
n_heads=1 kv_size=2048
```

then the bug is not fixed.

---

## Fix 5 — Keep `ggml.c` assertions layout-flexible but strict enough

### File

```text
ggml/src/ggml.c
```

Function:

```cpp
struct ggml_tensor * ggml_pack_k_packed16(...)
```

The old assertion:

```cpp
GGML_ASSERT(k_cur->ne[0] == payload->ne[0] * 4);
GGML_ASSERT(k_cur->ne[0] == scales->ne[0]  * 32);
```

is invalid for compact per-head payload with combined-GQA source.

Use:

```cpp
const int64_t per_head_k = payload->ne[0] * 4;
GGML_ASSERT(per_head_k > 0);
GGML_ASSERT(k_cur->ne[0] % per_head_k == 0);

const int64_t inferred_heads = k_cur->ne[0] / per_head_k;
GGML_ASSERT(inferred_heads > 0);
GGML_ASSERT(scales->ne[0] * 32 == per_head_k);
GGML_ASSERT(payload->ne[1] % inferred_heads == 0);
GGML_ASSERT(scales->ne[1] == payload->ne[1]);
```

This checks the correct invariant:

```text
k_cur width = n_heads * D
payload width = D/4
scales width = D/32
payload rows = kv_size * n_heads
```

---

## Fix 6 — Validate v4 read-side stride still uses fixed `kv_size`

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

The v4 DOT4 kernel must read K with:

```cpp
const size_t k_head_base = size_t(b) * size_t(k_batch_stride_rows)
                         + size_t(hk) * size_t(k_head_stride_rows);
```

not:

```cpp
const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
```

Dispatch-side expected values for c512 first ubatch:

```text
nk = 256
k_head_stride_rows = 512
k_batch_stride_rows = 2048
```

If debug shows:

```text
k_head_stride_rows = 256
```

then the view/stride is still wrong.

---

## Fix 7 — Add a NaN sentinel kernel after v4 output

This is optional but useful if NaN persists after Fixes 1-6.

### File

```text
ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
```

Add a tiny debug-only kernel:

```cpp
static __global__ void ggml_cuda_q8k_dot4_nan_check_kernel(
        const float * x,
        int64_t n,
        int * flag) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(x[i])) {
        atomicExch(flag, 1);
    }
}
```

After the v4 kernel launch, if env var is enabled:

```cpp
if (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_NAN_CHECK")) {
    ggml_cuda_pool_alloc<int> nan_flag(pool);
    nan_flag.alloc(1);
    CUDA_CHECK(cudaMemsetAsync(nan_flag.ptr, 0, sizeof(int), stream));
    const int64_t dst_ne = ggml_nelements(dst);
    ggml_cuda_q8k_dot4_nan_check_kernel<<<(dst_ne + 255) / 256, 256, 0, stream>>>(
        (const float *) dst->data, dst_ne, nan_flag.ptr);
    CUDA_CHECK(cudaGetLastError());
    int h_flag = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_flag, nan_flag.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_flag) {
        GGML_ABORT("q8k_dot4_i32: NaN detected in FA output");
    }
}
```

This confirms whether NaN is born inside FA or later in the model.

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

Expected debug lines:

```text
llama_kv_cache: ROCm0 KV buffer size = 24.50 MiB
get_k I32: kp ne=[64,2048,1,1]
pack_k_i32: k_cur ne=[1024,256,1,1] ... n_heads=4 nk_cur=256 kv_size=512 src_head_stride=512
q8k_dot4_i32: K ne=[64,256,4,1] nb=[4,256,131072,524288]
q8k_dot4_i32: ... nk=256 hk=4 ...
```

Expected result:

```text
[1]<finite>,[2]<finite>,[3]<finite>
```

Not:

```text
[1]nan,[2]nan,[3]nan,
```

## If PPL is still NaN

Enable the FA output sentinel:

```bash
GGML_CUDA_ROCM_Q8K_DOT4_KQ_NAN_CHECK=1
```

Interpretation:

- If sentinel aborts immediately after FA: NaN is born inside v4 FA; inspect mask/softmax path.
- If sentinel does not abort but PPL is NaN later: downstream model op receives finite but wrong values; compare FA output against f16 FA shadow.

---

# Minimal Patch Summary

The most likely final fix is this exact change:

1. In `ggml_cuda_op_pack_k_packed16`, infer `n_heads` as:

```cpp
if (k_cur->ne[0] == D) n_heads = k_cur->ne[2];
else n_heads = k_cur->ne[0] / D;
```

2. Compute:

```cpp
src_head_stride_bytes = (k_cur->ne[0] == D) ? k_cur->nb[2] : D * sizeof(half);
kv_size = payload->ne[1] / n_heads;
```

3. In the indexed kernel, read source K using:

```cpp
K + b*nb03 + k_local*nb01 + hk*src_head_stride_bytes
```

not always:

```cpp
K + b*nb03 + hk*nb02 + k_local*nb01
```

This fixes the likely cause: only head 0 was packed, while FA reads heads 0-3.
