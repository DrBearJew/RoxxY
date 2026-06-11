#include "fattn-dot4-q8k-kq.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

// Packed16 K cache tensor registry (shared with llama-kv-cache).
struct packed16_registry_entry {
    ggml_tensor * payload  = nullptr;
    ggml_tensor * scales   = nullptr;
    ggml_tensor * shadow_k = nullptr;
};

static std::mutex s_packed16_mutex;
static std::unordered_map<const void *, packed16_registry_entry> s_packed16_registry;

extern "C" {
void llama_kv_cache_register_packed16(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    packed16_registry_entry & entry = s_packed16_registry[k_view_data];
    entry.payload = payload;
    entry.scales  = scales;
}

void llama_kv_cache_register_packed16_shadow(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales, ggml_tensor * shadow_k) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    s_packed16_registry[k_view_data] = {payload, scales, shadow_k};
}

void llama_kv_cache_get_packed16_tensors(const void * k_view_data, ggml_tensor ** payload, ggml_tensor ** scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        *payload = it->second.payload;
        *scales  = it->second.scales;
    } else {
        *payload = nullptr;
        *scales  = nullptr;
    }
}

void llama_kv_cache_get_packed16_shadow_k(const void * k_view_data, ggml_tensor ** shadow_k) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    *shadow_k = it != s_packed16_registry.end() ? it->second.shadow_k : nullptr;
}
}

#ifdef GGML_USE_HIP

static constexpr int GGML_CUDA_Q8K_DOT4_KQ_D = 256;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_BLOCKS = GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0;

static inline bool ggml_cuda_pack_k_packed16_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

// Packed16 K quality knobs for acceptance/drift experiments. Defaults preserve
// existing behavior: persistent indexed K-cache packing keeps the current
// one-step MSE scale.
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE=maxabs|mse|0|1
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL=<positive float>
// LLAMA_MTP_* aliases are accepted for MTP-specific sweeps.
static inline int ggml_cuda_pack_k_packed16_scale_mode(int fallback) {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE");
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_MODE");
    }
    if (!env || env[0] == '\0') {
        return fallback;
    }
    const std::string mode(env);
    if (mode == "maxabs" || mode == "amax" || mode == "0") {
        return 0;
    }
    if (mode == "mse" || mode == "mse1" || mode == "1") {
        return 1;
    }
    return fallback;
}

static inline float ggml_cuda_pack_k_packed16_scale_mul() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL");
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_MUL");
    }
    if (!env || env[0] == '\0') {
        return 1.0f;
    }
    const float v = strtof(env, nullptr);
    return (v > 0.0f && v < 100.0f) ? v : 1.0f;
}

static __device__ __forceinline__ int ggml_cuda_pack_k_packed16_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_pack_k_packed16_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_k_packed16_from_q8_indexed_kernel(
        const char     * __restrict__ K,       // q8_0 K cache (block_q8_0 rows)
        int            * __restrict__ k_payload,
        half           * __restrict__ k_scales,
        const int64_t  * __restrict__ k_idxs,
        int64_t nb11,     // K row stride in bytes
        int kv_size,
        int n_heads_k,
        int batch,
        int nk_cur) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int qblk = tid;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || qblk >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const int64_t cell = k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) {
        return;
    }

    const size_t q8_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    const char * src = K + q8_row * nb11 + qblk * sizeof(block_q8_0);

    const size_t packed_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    k_scales[packed_row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_pack_k_packed16_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        k_payload[packed_row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4) + i] =
            ggml_cuda_pack_k_packed16_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

// Indexed variant: writes to absolute KV cache slots using k_idxs.
// Uses fixed kv_size head stride so persistent cache survives chunk growth.
template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_quant_k_packed16_indexed_kernel(
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
        int kv_size,
        int scale_mode,
        float scale_mul) {
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

    // Preserve the existing packer contract: graph code may pass f32 or f16
    // storage, and existing callers provide byte strides describing the row.
    const float * k_ptr = (const float *) ((const char *) K
            + int64_t(b)       * nb03
            + int64_t(k_local) * nb01
            + int64_t(hk)      * src_head_stride_bytes);

    const int d = q_block * QK8_0 + lane;
    const float x = k_ptr[d];

    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale0 = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi0 = max(-128, min(127, int(lrintf(x / scale0))));

    float scale = scale0;
    if (scale_mode != 0) {
        const float xi = (float) qi0;
        float num = x * xi;
        float den = xi * xi;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            num += __shfl_xor(num, mask, 32);
            den += __shfl_xor(den, mask, 32);
        }
        scale = (den > 0.0f) ? (num / den) : scale0;
    }
    scale *= scale_mul;
    if (!(scale > 0.0f)) {
        scale = scale0;
    }
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("q8k DOT4 attention route has been removed; use packed16 DOT4-MMQ/PDMQ");
}

// ggml OP_PACK_K_PACKED16 backend: quantize Kcur -> packed16 (I32 payload + F16 scales).
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k_cur   = dst->src[0];  // source K rows
    ggml_tensor * scales  = dst->src[1];  // F16 scales output
    ggml_tensor * k_idxs  = dst->src[2];  // row indices (absolute cache slots)
    ggml_tensor * payload = dst;          // I32 payload (dst is view of payload)

    GGML_ASSERT(k_idxs != nullptr);
    GGML_ASSERT(payload->type == GGML_TYPE_I32);
    GGML_ASSERT(scales->type == GGML_TYPE_F16);
    GGML_ASSERT(payload->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    GGML_ASSERT(scales->ne[0]  == GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    const int D = GGML_CUDA_Q8K_DOT4_KQ_D;
    const int batch = (int) k_cur->ne[3];

    int n_heads = 0;
    int nk_cur  = 0;
    int64_t src_head_stride_bytes = 0;

    if (k_cur->ne[0] == D) {
        // Split-head source: [D, n_heads, nk_cur, batch]
        n_heads = (int) k_cur->ne[1];
        nk_cur  = (int) k_cur->ne[2];
        src_head_stride_bytes = k_cur->nb[1];
    } else {
        // Combined GQA source: [D * n_heads, nk_cur, 1, batch]
        GGML_ASSERT(k_cur->ne[0] % D == 0);
        n_heads = (int) (k_cur->ne[0] / D);
        nk_cur  = (int) k_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(k_cur->type);
    }

    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(payload->ne[1] % n_heads == 0);

    const int kv_size = (int) (payload->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);

    const bool is_q8_k = k_cur->type == GGML_TYPE_Q8_0;
    if (is_q8_k) {
        // Active token count for q8_0 source comes from row indices.
        nk_cur = (int) k_idxs->ne[0];
        GGML_ASSERT(nk_cur <= kv_size);
    }

    const int64_t src_token_stride_bytes = (k_cur->ne[0] == D)
        ? k_cur->nb[2]
        : k_cur->nb[1];

    const int k_scale_mode = ggml_cuda_pack_k_packed16_scale_mode(1);
    const float k_scale_mul = ggml_cuda_pack_k_packed16_scale_mul();

    static bool pack_debug_printed = false;
    if (!pack_debug_printed && ggml_cuda_pack_k_packed16_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
        pack_debug_printed = true;
        fprintf(stderr,
            "pack_k_i32: k_cur type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
            "payload ne=[%lld,%lld,%lld,%lld] scales ne=[%lld,%lld,%lld,%lld] "
            "n_heads=%d nk_cur=%d kv_size=%d src_head_stride=%lld scale_mode=%d scale_mul=%g\n",
            k_cur->type, (long long) k_cur->ne[0], (long long) k_cur->ne[1], (long long) k_cur->ne[2], (long long) k_cur->ne[3],
            k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
            (long long) payload->ne[0], (long long) payload->ne[1], (long long) payload->ne[2], (long long) payload->ne[3],
            (long long) scales->ne[0], (long long) scales->ne[1], (long long) scales->ne[2], (long long) scales->ne[3],
            n_heads, nk_cur, kv_size, (long long) src_head_stride_bytes, k_scale_mode, (double) k_scale_mul);
    }

    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(256);
    cudaStream_t stream = ctx.stream();

    if (is_q8_k) {
        // Indexed pack from q8_0 cache blocks preserves calibrated quantization.
        ggml_cuda_pack_k_packed16_from_q8_indexed_kernel<<<grid, block, 0, stream>>>(
            (const char *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int64_t *) k_idxs->data,
            k_cur->nb[1], kv_size, n_heads, batch, nk_cur);
    } else if (k_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int64_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul);
    } else {
        GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
        ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int32_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_USE_HIP
