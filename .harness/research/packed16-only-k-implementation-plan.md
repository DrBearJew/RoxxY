# Packed16-only K cache implementation plan

## Purpose

Ship the DOT4 flash-attention path with packed16 K as the primary K cache storage instead of a shadow next to the existing f16 K cache.

The target outcome is:

- default f16 V cache remains unchanged;
- f16 K cache allocation is removed when `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` is set;
- K cache storage becomes `GGML_TYPE_I32` packed INT8 payload plus `GGML_TYPE_F16` scales;
- DOT4 FA receives K as an I32 packed16 tensor with physical head dimension `D/4` and treats it as logical head dimension `D`;
- observed 64k KV buffer moves from current shadow mode `5184 MiB` to target packed16-only `3136 MiB`;
- correctness remains at the current verified PPL level: `PPL ~= 1.0128` on the Qwen3.6 27B Q4_K_M smoke.

## Current state to preserve

Known-good DOT4 FA behavior before packed16-only K:

- DOT4 FA v4 runs correctly with default f16 KV cache.
- Packed16 K shadow tensors exist: `k_payload` as `GGML_TYPE_I32`, `k_scales` as `GGML_TYPE_F16`.
- `GGML_OP_PACK_K_PACKED16` exists and has HIP backend dispatch.
- Route gates already accept f16 K + f16 V for the shadow mode.
- Route kernel accepts `K->type == GGML_TYPE_I32` in `ggml_cuda_q8k_dot4_kq_supported()`.
- Verified shadow-mode numbers:
  - c512 KV buffer: `40.50 MiB` = f16 K + f16 V + packed16 shadow.
  - c65536 KV buffer: `5184.00 MiB`.
  - PPL: `1.0128 +/- ~0.004`.

Expected packed16-only numbers:

- c512 target KV buffer: about `24.50 MiB` = f16 V + packed16 K.
- c65536 target KV buffer: `3136.00 MiB`.
- 64k saving vs baseline f16 K+V: `4096 - 3136 = 960 MiB`.
- 64k saving vs current shadow mode: `5184 - 3136 = 2048 MiB`.

## Non-negotiable constraints

1. Keep all behavior route-gated. Do not change default llama.cpp runtime behavior.
2. Do not make generic `ggml_can_mul_mat()` accept I32 compressed K globally. That would lie to every non-FA matmul caller.
3. Treat packed16 K as a physical tensor with `ne[0] = D/4`, not as f16-compatible storage.
4. Use logical dimension helpers only at flash-attention graph validation and CUDA selector points.
5. Keep V as normal f16 for this milestone.
6. If packed16 K is primary storage, K writes must honor KV cache row indices. Ignoring `k_idxs` is acceptable only for a throwaway contiguous-prefill experiment, not for server use.
7. `ggml_permute()` itself should not be modified unless a hard assertion proves otherwise. The correct approach is to feed it a valid physical I32 view and handle logical K dimension later in FA validation.

## Best implementation strategy

Use a narrow, route-specific logical-dimension adapter for flash attention:

- The graph still contains a normal `GGML_OP_FLASH_ATTN_EXT` node.
- Q remains physical/logical `[D, nq, n_head_q, batch]` after graph permutation.
- K becomes physical `[D/4, nkv, n_head_kv, batch]` after graph permutation when packed16-only mode is active.
- Flash-attention validation treats this K as logical `D` only for the K/Q compatibility checks.
- CUDA route selection treats `K->type == GGML_TYPE_I32 && K->ne[0] * 4 == Q->ne[0]` as valid only for the gated DOT4 packed16 route.
- The DOT4 backend receives I32 payload directly and obtains the matching scales tensor through the existing packed16 registry.

This avoids broad ggml semantics changes while allowing the HIP DOT4 FA route to consume a compressed K layout.

---

# Implementation phases

## Phase 0 — Start from a clean implementation branch

Before editing, inspect the worktree:

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
git status --short
git diff -- ggml/src/ggml.c ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh src/llama-kv-cache.cpp src/llama-kv-cache.h
```

Important: previous exploratory edits may have left `src/llama-kv-cache.cpp` in a half-primary state. In particular, check for these two bad patterns before implementing:

- `get_k()` using `k_payload` but computing strides with the logical f16 width;
- `cpy_k()` dereferencing `k->ne[...]` before handling the packed16-only branch.

If those are present, either revert only `src/llama-kv-cache.cpp` to the last known-good shadow-mode version, or carefully replace the affected blocks with the instructions below. Do not build on top of a half-applied sed patch.

Recommended safety checkpoint:

```bash
git diff > /tmp/packed16-before-primary-k.diff
```

## Phase 1 — Replace the generic `ggml_can_mul_mat` assertion with a flash-attention-specific helper

File:

- `ggml/src/ggml.c`

Current blocker:

```c
GGML_ASSERT(ggml_can_mul_mat(k, q));
```

or the current temporary one-liner:

```c
GGML_ASSERT(ggml_can_mul_mat(k, q) || (k->type == GGML_TYPE_I32 && k->ne[0] * 4 == q->ne[0]));
```

Best replacement: add a local helper near `ggml_flash_attn_ext()` instead of broadening `ggml_can_mul_mat()` globally.

Suggested helper:

```c
static inline bool ggml_can_flash_attn_ext_kq(const struct ggml_tensor * k, const struct ggml_tensor * q) {
    if (ggml_can_mul_mat(k, q)) {
        return true;
    }

    // Packed16 K cache for ROCm DOT4 FA:
    // - K is physical I32 packed INT8 payload, one I32 holds 4 K dimensions.
    // - Q remains physical F32/F16 logical head dimension D.
    // - The batch/head broadcast rules must remain identical to ggml_can_mul_mat().
    return k->type == GGML_TYPE_I32 &&
           q->ne[0] % 4 == 0 &&
           k->ne[0] * 4 == q->ne[0] &&
           q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] % k->ne[3] == 0;
}
```

Then use:

```c
GGML_ASSERT(ggml_can_flash_attn_ext_kq(k, q));
```

Do this in the live `ggml_flash_attn_ext()` function. The older aborted compatibility function can stay unchanged unless it is compiled through a test path; it aborts before meaningful use.

Why this is best:

- It does not teach generic matmul that I32 packed K and f16/f32 Q are multiply-compatible.
- It isolates the exception to FA graph construction, where the HIP backend knows how to consume the format.
- It preserves head/batch broadcast validation from `ggml_can_mul_mat()`.

## Phase 2 — Make `get_k()` return a physically correct packed16 K view

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

### 2.1 Allocation

In the KV cache constructor, detect packed16 mode once per layer before K allocation:

```c++
const bool packed16_active = has_k && ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
```

If using the env helper is not visible here, keep the local env read, but store it in a clearly named bool.

Allocate:

```c++
ggml_tensor * k = (has_k && !packed16_active)
    ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
    : nullptr;

ggml_tensor * k_payload = nullptr;
ggml_tensor * k_scales  = nullptr;

if (packed16_active) {
    GGML_ASSERT(n_embd_k_gqa % 4 == 0);
    GGML_ASSERT(n_embd_k_gqa % 32 == 0);

    k_payload = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_embd_k_gqa / 4,  kv_size, n_stream);
    k_scales  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa / 32, kv_size, n_stream);

    ggml_format_name(k_payload, "cache_k_payload_l%d", il);
    ggml_format_name(k_scales,  "cache_k_scales_l%d",  il);
}
```

Only call `ggml_format_name(k, ...)` if `k != nullptr`.

### 2.2 Stream views

Populate both existing `k_stream` and explicit packed16 stream vectors. In packed16-only mode `k_stream` should point to the payload view because callers use `layer.k_stream[...]` for K copies and state paths.

Suggested shape:

```c++
for (uint32_t s = 0; s < n_stream; ++s) {
    if (has_k && packed16_active) {
        ggml_tensor * payload_view = ggml_view_2d(ctx, k_payload,
                n_embd_k_gqa / 4, kv_size,
                k_payload->nb[1],
                s * k_payload->nb[2]);
        ggml_tensor * scales_view = ggml_view_2d(ctx, k_scales,
                n_embd_k_gqa / 32, kv_size,
                k_scales->nb[1],
                s * k_scales->nb[2]);

        k_stream.push_back(payload_view);
        k_payload_stream.push_back(payload_view);
        k_scales_stream.push_back(scales_view);
    } else {
        k_stream.push_back(has_k ? ggml_view_2d(ctx, k,
                n_embd_k_gqa, kv_size,
                k->nb[1],
                s * k->nb[2]) : nullptr);
        k_payload_stream.push_back(nullptr);
        k_scales_stream.push_back(nullptr);
    }

    v_stream.push_back(has_v ? ggml_view_2d(ctx, v,
            n_embd_v_gqa, kv_size,
            v->nb[1],
            s * v->nb[2]) : nullptr);
}
```

### 2.3 Correct `get_k()` dimensions and strides

This is the most important part. Do not compute I32 payload strides using logical f16 dimensions.

For f16/default K, keep the old code.

For packed16 primary K:

- physical full GQA dim = `n_embd_k_gqa / 4`;
- physical per-head dim = `hparams.n_embd_head_k(il) / 4`;
- physical token stride = row size of the packed full GQA dim, not logical dim;
- physical stream stride = row size of packed full GQA dim times `kv_size`.

Recommended structure:

```c++
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    const uint64_t kv_size = get_size();
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (layers[ikv].k_payload) {
        ggml_tensor * k = layers[ikv].k_payload;

        const uint64_t n_embd_k_gqa_logical = hparams.n_embd_k_gqa(il);
        const uint64_t n_embd_k_gqa_phys    = n_embd_k_gqa_logical / 4;
        const uint64_t n_embd_head_phys     = hparams.n_embd_head_k(il) / 4;

        GGML_ASSERT(k->type == GGML_TYPE_I32);
        GGML_ASSERT(n_embd_k_gqa_logical % 4 == 0);
        GGML_ASSERT(hparams.n_embd_head_k(il) % 4 == 0);
        GGML_ASSERT(k->ne[0] == (int64_t) n_embd_k_gqa_phys);

        return ggml_view_4d(ctx, k,
                n_embd_head_phys, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(k->type, n_embd_head_phys),
                ggml_row_size(k->type, n_embd_k_gqa_phys),
                ggml_row_size(k->type, n_embd_k_gqa_phys * kv_size),
                ggml_row_size(k->type, n_embd_k_gqa_phys * kv_size) * sinfo.s0);
    }

    // Existing f16/TBQ/default path remains unchanged below.
}
```

`ggml_permute(ctx0, k, 0, 2, 1, 3)` in `src/llama-graph.cpp` should then work without modification:

- before permute packed K is `[D/4, n_head_kv, n_kv, ns]`;
- after permute packed K is `[D/4, n_kv, n_head_kv, ns]`;
- flash-attention validation maps `D/4` back to logical `D`.

If an assertion later proves `ggml_permute()` itself rejects the tensor, fix only that assertion. Do not preemptively change generic permute semantics.

## Phase 3 — Make `cpy_k()` populate packed16 primary K safely

File group:

- `src/llama-kv-cache.cpp`
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

### 3.1 Do not dereference f16 `k` in packed16-only mode

When `k == nullptr`, this pattern is invalid:

```c++
const int64_t n_stream = k->ne[2];
```

The packed16 branch must happen before any `k` dereference.

Recommended `cpy_k()` flow:

1. Grab `k`, `k_payload`, `k_scales`.
2. Read `n_embd_head`, `n_head`, `n_tokens` from the original 3D `k_cur`.
3. If `k_payload && k_scales`, call packed16 op using the original 3D K layout and return.
4. Only then execute the old f16 `ggml_set_rows()` path.

Pseudo-code:

```c++
ggml_tensor * llama_kv_cache::cpy_k(... ) const {
    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k         = layers[ikv].k;
    ggml_tensor * k_payload = layers[ikv].k_payload;
    ggml_tensor * k_scales  = layers[ikv].k_scales;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head * n_head;

    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    if (k_payload && k_scales) {
        return ggml_pack_k_packed16(ctx, k_cur, k_payload, k_scales, k_idxs);
    }

    // old f16 path: reshape to 2D and ggml_set_rows
}
```

### 3.2 Extend `GGML_OP_PACK_K_PACKED16` to accept `k_idxs`

This is the best server-safe path. The current packed op shape writes source token `i` to destination row `i`. That works only for trivial contiguous prefill starting at cache row 0. It is not safe as primary storage for:

- decode;
- cache reuse;
- shifted/ring-buffer slots;
- multi-sequence server traffic;
- any non-zero cache head.

Change the public function signature from:

```c
struct ggml_tensor * ggml_pack_k_packed16(ctx, k_cur, payload, scales);
```

to:

```c
struct ggml_tensor * ggml_pack_k_packed16(ctx, k_cur, payload, scales, k_idxs);
```

Set:

```c
result->src[0] = k_cur;
result->src[1] = scales;
result->src[2] = k_idxs;
```

Backend kernel should use `k_idxs[token]` as the destination KV row. If the op must support multi-stream later, include stream offset explicitly or use the same global index convention as `ggml_set_rows()`.

### 3.3 Kernel write layout

The pack kernel should interpret source K as the original 3D tensor:

- `k_cur->ne[0] == D` per-head dimension;
- `k_cur->ne[1] == n_head_kv`;
- `k_cur->ne[2] == n_tokens`;
- `k_cur->ne[3]` usually 1.

Destination payload tensor is:

- base shape `[n_embd_k_gqa/4, kv_size, n_stream]`;
- physical per-head payload width `D/4` I32 values;
- scales physical per-head width `D/32` f16 values.

For each source token `t`, head `hk`, and q_block:

```c++
const int dst_token = ((const int32_t *) k_idxs->data)[t];
const size_t dst_row = ((size_t) dst_token * n_heads_k + hk);
// or, if payload is flattened by [all_heads_payload, kv], compute:
// payload offset = dst_token * (n_heads_k * D/4) + hk * (D/4)
```

Use the destination layout already used by the FA kernel. Verify row order matches the KQ kernel's `k_base` calculation. Do not assume row order from memory names; inspect the actual indexing in `ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel` and keep the same flattened convention.

Minimum acceptable smoke-only fallback:

- temporarily assert `sinfo.is_contiguous()`;
- write to `sinfo.head() + t`;
- mark this as prefill-only and do not ship server decode with it.

Best implementation: use `k_idxs` now.

## Phase 4 — Register packed16 tensors after backend allocation

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- registry functions currently declared near the packed16 code and defined in `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`.

The DOT4 backend needs both payload and scales. If FA receives `K` as the I32 payload view, there is no direct graph edge from K to `k_scales`. Continue using the registry, but register after backend allocation because `tensor->data` is not final until the buffer is allocated.

After each `ggml_backend_alloc_ctx_tensors_from_buft()` succeeds and before leaving the KV cache constructor, register aliases for every packed16 layer:

```c++
for (auto & layer : layers) {
    if (!layer.k_payload || !layer.k_scales) {
        continue;
    }

    // Primary packed16 mode: FA K->data is payload data.
    llama_kv_cache_register_packed16(layer.k_payload->data, layer.k_payload, layer.k_scales);

    // Shadow mode compatibility: if f16 K still exists, FA K->data may be f16 K data.
    if (layer.k) {
        llama_kv_cache_register_packed16(layer.k->data, layer.k_payload, layer.k_scales);
    }
}
```

If there are per-stream views whose `data` pointer includes an offset, register those too:

```c++
for (size_t s = 0; s < layer.k_payload_stream.size(); ++s) {
    if (layer.k_payload_stream[s]) {
        llama_kv_cache_register_packed16(layer.k_payload_stream[s]->data, layer.k_payload_stream[s], layer.k_scales_stream[s]);
    }
}
```

The backend lookup path in `ggml_cuda_flash_attn_ext_q8k_dot4_kq()` should handle three cases:

1. `K->type == GGML_TYPE_I32` and registry lookup succeeds: use K payload plus registered scales; skip repack.
2. shadow mode with f16 K and registry lookup succeeds: use registered payload/scales; skip repack if rows are current.
3. no registry: fall back to hipMalloc/repack path for compatibility.

For primary packed16 mode, skip-repack should be unconditional after successful registry lookup. The `GGML_OP_PACK_K_PACKED16` op is responsible for keeping payload/scales current.

## Phase 5 — Fix CUDA flash-attention route validation for I32 packed16 K

Files:

- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh`

### 5.1 Normalize route contract aliases

Current code has partial awareness of `rocm_q8k_dot4_packed16_blockfa`. Make it complete.

Update all route-contract helpers so these names are equivalent for the DOT4 packed16/blockfa route:

- `rocm_q8k_dot4_kq`
- `rocm_q8k_dot4_packed16_blockfa`

Required places:

1. `ggml_cuda_q8k_dot4_kq_route_required()` in `fattn-dot4-q8k-kq.cuh`.
2. `ggml_cuda_fattn_route_contract_matches()` in `fattn.cu`.
3. `ggml_cuda_fattn_route_contract_is_i8()` in `fattn.cu`.
4. `ggml_cuda_fattn_route_contract_applicable()` in `fattn.cu`.

Suggested helper to avoid drift:

```c++
static inline bool ggml_cuda_q8k_dot4_kq_route_name(const char * required) {
    return required &&
        (strcmp(required, "rocm_q8k_dot4_kq") == 0 ||
         strcmp(required, "rocm_q8k_dot4_packed16_blockfa") == 0);
}
```

Use that helper in both `.cuh` and `.cu` if visibility permits; otherwise duplicate the exact same predicate with a comment.

### 5.2 Add a CUDA-side packed16 logical-K predicate

Inside `ggml_cuda_get_best_fattn_kernel()`, compute this after `Q/K/V` and `cc` are known:

```c++
#ifdef GGML_USE_HIP
const bool q8k_dot4_packed16_k =
    ggml_cuda_q8k_dot4_kq_supported(cc, dst) &&
    K->type == GGML_TYPE_I32 &&
    Q->ne[0] == 256 &&
    K->ne[0] == 256 / 4 &&
    V->ne[0] == 256 &&
    dst->type == GGML_TYPE_F32;
#else
const bool q8k_dot4_packed16_k = false;
#endif
```

Do not use this predicate to bless any other route.

### 5.3 Fix the `K->ne[0]` switch

Current selector rejects case `K->ne[0] == 64` if `V->ne[0] != K->ne[0]`. For packed16 K, that is expected: physical K is 64, logical V is 256.

In the `case 64:` path, allow:

```c++
if (q8k_dot4_packed16_k) {
    break;
}
```

before the generic `V->ne[0] != K->ne[0]` rejection.

Alternative cleaner version:

```c++
const int64_t K_ne0_logical = q8k_dot4_packed16_k ? K->ne[0] * 4 : K->ne[0];
```

and then compare `V->ne[0]` to `K_ne0_logical` in the generic cases. Keep this local to selector validation.

### 5.4 Fix mixed-KV validation

`ggml_cuda_fattn_mixed_kv_supported()` currently allows only selected quantized mixtures. It should not globally allow I32/F16. Add a route-gated path, e.g. either pass `dst`/`cc` to it or check a narrow local predicate before calling it.

Best local minimal pattern:

```c++
if (!q8k_dot4_packed16_k && !ggml_cuda_fattn_mixed_kv_supported(Q, K, V)) {
    ggml_cuda_fattn_log_mixed_kv_reject(Q, K, V);
    return BEST_FATTN_KERNEL_NONE;
}
```

This keeps generic mixed-KV policy unchanged.

### 5.5 Permit `GGML_TYPE_I32` in the K type switch only for this route

Add to the `switch (K->type)` block:

```c++
case GGML_TYPE_I32:
    if (!q8k_dot4_packed16_k) {
        return BEST_FATTN_KERNEL_NONE;
    }
    break;
```

### 5.6 Ensure the route-selection block runs for I32 K

The current quantized-route block is entered when K/V is quantized or a required i8 contract is applicable. I32 is not a ggml quantized type. After route alias fixes, `i8_applicable` should become true for `rocm_q8k_dot4_packed16_blockfa`; still, make this explicit and robust:

```c++
if ((((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) || i8_applicable || q8k_dot4_packed16_k) && can_use_vector_kernel)) {
    ...
    if (!require_f16_route && ggml_cuda_q8k_dot4_kq_supported(cc, dst)) {
        return return_quantized_route(BEST_FATTN_KERNEL_Q8K_DOT4_KQ);
    }
}
```

Do not let I32 fall through to generic TILE/MMA f16 FA.

## Phase 6 — Backend direct-primary handling

File:

- `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu`

In `ggml_cuda_flash_attn_ext_q8k_dot4_kq()`, when `K->type == GGML_TYPE_I32`:

1. Look up `(payload, scales)` by `K->data` in the registry.
2. If found:
   - set `k_payload.ptr = (int *) payload_tensor->data` or `(int *) K->data`;
   - set `k_scales.ptr = (half *) scales_tensor->data`;
   - set `skip_k_repack = true`.
3. If not found, abort with a clear message in primary mode. Do not silently reinterpret I32 K without scales.

Suggested abort:

```c++
if (K->type == GGML_TYPE_I32 && (!payload_tensor || !scales_tensor)) {
    GGML_ABORT("packed16 primary K reached DOT4 FA without registered scales tensor");
}
```

When `K->type == GGML_TYPE_I32`, never run the f16-to-packed16 or q8_0-to-packed16 repack kernels. The K tensor is already the payload.

## Phase 7 — KV cache accounting and auxiliary paths

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

### 7.1 `type_k()`

If `layers[0].k == nullptr`, return `layers[0].k_payload->type`.

```c++
ggml_type llama_kv_cache::type_k() const {
    return layers[0].k ? layers[0].k->type : layers[0].k_payload->type;
}
```

### 7.2 `size_k_bytes()`

If packed16 exists, count payload + scales. If f16 K also exists in shadow mode, include it too.

```c++
for (const auto & layer : layers) {
    if (layer.k) {
        size_k_bytes += ggml_nbytes(layer.k);
    }
    if (layer.k_payload) {
        size_k_bytes += ggml_nbytes(layer.k_payload);
    }
    if (layer.k_scales) {
        size_k_bytes += ggml_nbytes(layer.k_scales);
    }
}
```

### 7.3 Stream copies

Where stream copies currently do:

```c++
ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);
```

ensure packed16 scales are also copied:

```c++
if (layer.k_stream[ssrc]) {
    ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);
}
if (layer.k_scales_stream[ssrc]) {
    ggml_backend_tensor_copy(layer.k_scales_stream[ssrc], layer.k_scales_stream[sdst]);
}
```

### 7.4 State save/load and shift paths

Before calling packed16-only mode production-ready, inspect state save/load and K-shift paths:

```bash
rg -n "state_write|state_read|build_rope_shift|k_stream|layer\.k" src/llama-kv-cache.cpp
```

Minimum safe milestone:

- disable or abort state save/load for packed16-only K until scales serialization is implemented;
- verify K-shift is not used for the target Qwen workload, or implement packed16-aware shift/copy for payload and scales.

Do not claim general llama.cpp KV state compatibility until these are handled.

## Phase 8 — Verification plan

Use the exact ROCm build environment:

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
ROCM_PATH=/opt/rocm-7.2.3 \
CC=/opt/rocm-7.2.3/llvm/bin/clang \
CXX=/opt/rocm-7.2.3/llvm/bin/clang++ \
cmake --build build-rocm-rdna3-fa --target llama-perplexity llama-server -j$(nproc)
```

### 8.1 c512 smoke: allocation and PPL

```bash
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 \
./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 512 -b 256 -ub 256 \
  -f /tmp/test_2048.txt 2>&1 | tee /tmp/packed16-primary-c512.log
```

Pass criteria:

- KV buffer is about `24.50 MiB`, not `40.50 MiB`.
- No `ggml_can_mul_mat` assertion.
- Route contract selects DOT4 route.
- PPL remains about `1.0128`.

Useful grep:

```bash
grep -E "KV buffer|fa_route_contract|route=|Final estimate|PPL|packed16" /tmp/packed16-primary-c512.log
```

### 8.2 c65536 allocation check

```bash
timeout 90 bash -c 'GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single \
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 \
GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 \
./build-rocm-rdna3-fa/bin/llama-perplexity \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M.gguf \
  --no-warmup -ngl 99 -c 65536 -b 256 -ub 256 --parallel 1 -fit off \
  -f /tmp/test_2048.txt 2>&1 | grep "KV buffer"'
```

Pass criteria:

- KV buffer reports about `3136.00 MiB`.

### 8.3 Route fail-fast check

Run with the required route and no packed16 cache:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_packed16_blockfa \
./build-rocm-rdna3-fa/bin/llama-perplexity ...
```

Expected behavior:

- Either route selects shadow/f16-compatible DOT4 mode if allowed by policy, or logs a clear not-applicable/rejected route.
- It must not silently select baseline FA while claiming packed16-only success.

### 8.4 Server decode smoke

After c512 PPL passes, run a short server decode with packed16-only K and a prompt that forces both prefill and decode.

Pass criteria:

- No cache-row corruption.
- No crash after the prompt phase.
- Decode output is coherent enough for a smoke.
- If `GGML_OP_PACK_K_PACKED16` does not yet use `k_idxs`, this test is not sufficient for production; fix `k_idxs` first.

## Phase 9 — Acceptance criteria

The implementation is done when all of these are true:

1. Build succeeds for `llama-perplexity` and `llama-server`.
2. c512 packed16-only KV buffer is about `24.50 MiB`.
3. c65536 packed16-only KV buffer is about `3136.00 MiB`.
4. PPL remains `~1.0128` on the known smoke file.
5. Route logs prove `BEST_FATTN_KERNEL_Q8K_DOT4_KQ` / DOT4 blockfa path is selected.
6. There is no f16 K allocation when packed16-only mode is active.
7. `cpy_k()` writes packed16 payload/scales using `k_idxs` or an explicitly documented contiguous-only assertion.
8. No generic ggml matmul behavior is changed.
9. Disabling `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE` restores the known shadow/default behavior.

## Likely failure modes and fixes

### Failure: `ggml_can_mul_mat(k, q)` assertion

Fix:

- confirm `ggml_flash_attn_ext()` uses `ggml_can_flash_attn_ext_kq()`;
- confirm the helper preserves broadcast checks;
- print `q->ne` and `k->ne` if needed.

Expected packed16 FA dimensions after graph permute:

- Q: `[256, nq, n_head_q, ns]`
- K: `[64, nkv, n_head_kv, ns]`
- V: `[256, nkv, n_head_kv, ns]`

### Failure: selector returns `BEST_FATTN_KERNEL_NONE`

Fix in order:

1. Check route alias support in `.cuh` and `fattn.cu`.
2. Check `K->ne[0] == 64` and `V->ne[0] == 256` passes the selector shape gate.
3. Check mixed-KV rejection bypass for `q8k_dot4_packed16_k`.
4. Check `GGML_TYPE_I32` is allowed in K type switch only under the route predicate.
5. Check quantized/i8 route block is entered for I32 K.

### Failure: backend aborts missing scales

Fix:

- register `k_payload->data -> (k_payload, k_scales)` after backend buffer allocation;
- register per-stream view data pointers if FA sees a stream-offset K view;
- in backend, for `K->type == GGML_TYPE_I32`, require registry success and skip repack.

### Failure: PPL is NaN or wrong

Likely causes:

1. `get_k()` packed strides used logical width instead of physical packed width.
2. `GGML_OP_PACK_K_PACKED16` packed from reshaped 2D K instead of original 3D K.
3. Pack kernel ignored `k_idxs` and wrote rows to wrong KV positions.
4. K payload row order does not match the DOT4 FA kernel's `k_base` calculation.
5. Scales tensor was not copied/registered for the same stream as payload.

First debug print/check:

- dump `K->ne`, `K->nb`, `V->ne`, route, and registry hit/miss in the DOT4 backend for layer 0 only.

### Failure: c512 KV is still `40.50 MiB`

Fix:

- f16 `k` is still allocated. Confirm constructor uses `k = nullptr` when packed16 active.
- Check `size_k_bytes()` is not double-counting old K.
- Check there is not a hidden stream view allocating f16 K.

### Failure: c512 KV is `24.50 MiB` but server decode corrupts

Fix:

- implement `k_idxs` in `GGML_OP_PACK_K_PACKED16`.
- copy `k_scales_stream` during stream copies.
- audit state/shift paths before production.

## Recommended commit structure

1. `ggml: allow packed16 K logical dim in flash_attn_ext validation`
   - only `ggml/src/ggml.c` helper and assertion.

2. `cuda: route DOT4 FA selector for I32 packed16 K`
   - route alias fixes;
   - selector shape/mixed/type gates;
   - backend direct-primary registry lookup.

3. `kv-cache: make packed16 K primary storage under env gate`
   - allocation;
   - stream views;
   - correct `get_k()` strides;
   - size/type helpers;
   - registry after allocation.

4. `ggml: make PACK_K_PACKED16 honor KV row indices`
   - signature change;
   - CUDA kernel update;
   - `cpy_k()` call update.

5. `docs/tests: record packed16-only VRAM and PPL smoke`
   - update master plan with c512/c65536 table and command logs.

## Short answer

The best implementation is not to teach all of ggml that I32 K is matmul-compatible. Instead, keep packed16 K as a real physical I32 tensor, add a flash-attention-only logical dimension helper, and make the CUDA DOT4 route selector explicitly accept `K I32 D/4 + V F16 D` only under the DOT4 packed16 route contract. The most important correctness detail is that `get_k()` must use physical packed strides and `GGML_OP_PACK_K_PACKED16` must honor `k_idxs` before packed16 becomes the only K storage.
