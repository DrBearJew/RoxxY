#include "fattn-dot4-q8k-kq.cuh"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

// Packed16 K cache tensor registry (shared with llama-kv-cache)
static std::mutex s_packed16_mutex;
static std::unordered_map<const void *, std::pair<ggml_tensor *, ggml_tensor *>> s_packed16_registry;

extern "C" {
void llama_kv_cache_register_packed16(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    s_packed16_registry[k_view_data] = {payload, scales};
}
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, ggml_tensor ** payload, ggml_tensor ** scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        *payload = it->second.first;
        *scales  = it->second.second;
    } else {
        *payload = nullptr;
        *scales  = nullptr;
    }
}
}

#ifdef GGML_USE_HIP

static constexpr int GGML_CUDA_Q8K_DOT4_KQ_D = 256;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_BLOCKS = GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_PACKED16_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 16;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_M = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_N = 16;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_GQA6 = 6;

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

static inline int ggml_cuda_q8k_dot4_kq_env_int(const char * name, int fallback) {
    const char * env = getenv(name);
    return env ? atoi(env) : fallback;
}

static inline void ggml_cuda_q8k_dot4_kq_event_create(hipEvent_t * event) {
    CUDA_CHECK(hipEventCreate(event));
}

static inline void ggml_cuda_q8k_dot4_kq_event_destroy(hipEvent_t event) {
    if (event != nullptr) {
        CUDA_CHECK(hipEventDestroy(event));
    }
}

static inline float ggml_cuda_q8k_dot4_kq_event_elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return ggml_cuda_dp4a(a, b, c);
#endif
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_q8k_dot4_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_4chain(
        const int * __restrict__ q_payload,
        const int * __restrict__ k_payload,
        int idx) {
    int acc = 0;
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 0], k_payload[idx + 0], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 1], k_payload[idx + 1], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 2], k_payload[idx + 2], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 3], k_payload[idx + 3], acc);
    return acc;
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_block(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ kq_sums,
        int buf) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    float partial = 0.0f;

    if (tid < 32) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = lane + i * 32;
            const int qb = idx / (QK8_0 / 4);
            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx], k_payload[idx], 0);
            partial += float(acc) * q_scales[qb] * __half2float(k_scales[qb]);
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset, 32);
        }
        if (lane == 0) {
            kq_sums[buf] = partial;
        }
    }
    __syncthreads();
    return kq_sums[buf];
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_pack_k_packed16_from_q8_indexed_kernel(
        const char   * __restrict__ K,       // q8_0 K cache (block_q8_0 rows)
        int          * __restrict__ k_payload,
        half         * __restrict__ k_scales,
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
    if (cell < 0 || cell >= kv_size) return;

    // Read q8_0 block: [d (half), qs[32] (int8)]
    const size_t q8_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    const char * src = K + q8_row * nb11 + qblk * sizeof(block_q8_0);
    // block_q8_0 layout: d (half, 2 bytes) + qs[32] (int8, 32 bytes) = 34 bytes per block
    // Write to packed16: k_scales = d, k_payload = qs as i32
    const size_t packed_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    k_scales[packed_row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_q8k_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        k_payload[packed_row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4) + i] =
            ggml_cuda_q8k_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_k_packed16_kernel(
        const half  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nk,
        int n_heads_k,
        int batch) {
    const int tid = threadIdx.x;
    const int k = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (k >= nk || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const half * k_ptr = (const half *) ((const char *) K + int64_t(b) * nb03 + int64_t(hk) * nb02 + int64_t(k) * nb01);
    const int d = q_block * QK8_0 + lane;
    float x = __half2float(k_ptr[d]);
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }
}

// Indexed variant: writes to absolute KV cache slots using k_idxs.
// Uses fixed kv_size head stride so persistent cache survives chunk growth.
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

    // Read K source as float — k_cur may be f32 (model output) or f16.
    // Always use 4-byte float stride for head offset.
    const float * k_ptr = (const float *) ((const char *) K
            + int64_t(b)       * nb03
            + int64_t(k_local) * nb01
            + int64_t(hk)      * src_head_stride_bytes);
    const int d = q_block * QK8_0 + lane;
    const float x = k_ptr[d];

    // Pass 1: block-wise amax for initial scale.
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale0 = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi0 = max(-128, min(127, int(lrintf(x / scale0))));

    // Pass 2: MSE-optimal scale = sum(x * qi) / sum(qi^2)
    const float xi = (float) qi0;
    float num = x * xi;
    float den = xi * xi;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        num += __shfl_xor(num, mask, 32);
        den += __shfl_xor(den, mask, 32);
    }
    const float scale = (den > 0.0f) ? (num / den) : scale0;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    // Fixed kv_size head stride — row = head * kv_size + cell
    const size_t row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_q_packed16_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ q_payload,
        float       * __restrict__ q_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nq,
        int n_heads_q,
        int batch) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (q >= nq || hq >= n_heads_q || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q) * nb01);
    const int d = q_block * QK8_0 + lane;
    const float x = q_ptr[d];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q);
    ((int8_t *)(q_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        q_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = scale;
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_pack_k_packed16_kernel(
        const char * __restrict__ K,
        int        * __restrict__ k_payload,
        half       * __restrict__ k_scales,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nk,
        int n_heads_k,
        int batch) {
    const int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = batch * n_heads_k * nk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    if (linear >= total) {
        return;
    }

    const int qblk = linear % GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int t = linear / GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int k = t % nk;
    const int hk_b = t / nk;
    const int hk = hk_b % n_heads_k;
    const int b = hk_b / n_heads_k;
    const char * src = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k) * nb11 + int64_t(qblk) * nb10;
    int * dst = k_payload + (size_t(t) * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4));

    k_scales[size_t(t) * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_q8k_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        dst[i] = ggml_cuda_q8k_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ logits,
        float scale,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int half_tile = (tid >> 4) & 1;
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const int idx = qb * (QK8_0 / 4) + half_tile * 4;
        const int acc = ggml_cuda_q8k_dot4_4chain(q_row_payload, k_row_payload, idx);
        sum += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
    }
    sum += __shfl_xor(sum, 16, 32);
    if (half_tile == 0) {
        logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    }
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_q8block_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const char  * __restrict__ K,
        float       * __restrict__ logits,
        float scale,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int half_tile = (tid >> 4) & 1;
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const char * k_blk = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k_row) * nb11 + int64_t(qb) * nb10;
        const half ks_h = ggml_cuda_q8k_dot4_load_half_unaligned(k_blk);
        const int * q_blk = q_row_payload + qb * (QK8_0 / 4);
        const int idx = half_tile * 4;
        int k_i32[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            k_i32[i] = ggml_cuda_q8k_dot4_load_i32_unaligned(k_blk + sizeof(half) + 4 * (idx + i));
        }
        int acc = 0;
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 0], k_i32[0], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 1], k_i32[1], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 2], k_i32[2], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 3], k_i32[3], acc);
        sum += float(acc) * q_row_scales[qb] * __half2float(ks_h);
    }
    sum += __shfl_xor(sum, 16, 32);
    if (half_tile == 0) {
        logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    }
    GGML_UNUSED(n_heads_k);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_q4_0(
        const char * __restrict__ V,
        int64_t nb20,
        int i) {
    const int ib = i / QK4_0;
    const int iq = i & 15;
    const int shift = (i & 31) >= 16 ? 4 : 0;
    const block_q4_0 * v = (const block_q4_0 *) (V + int64_t(ib) * nb20);
    const int q = (v->qs[iq] >> shift) & 0x0f;
    return (float(q) - 8.0f) * __half2float(v->d);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_q8_0(
        const char * __restrict__ V,
        int64_t nb20,
        int i) {
    const int ib = i / QK8_0;
    const int iq = i % QK8_0;
    const block_q8_0 * v = (const block_q8_0 *) (V + int64_t(ib) * nb20);
    return float(v->qs[iq]) * __half2float(v->d);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_mask_value(
        const char * __restrict__ mask,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int q,
        int k,
        int b) {
    if (!mask) {
        return 0.0f;
    }
    const char * p = mask + int64_t(b % ne33) * nb33 + int64_t(q) * nb31 + int64_t(k) * nb30;
    return __half2float(*(const half *) p);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel(
        const float * __restrict__ logits,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const float s = logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k] +
            ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out / denom;
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_fused_online_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    __shared__ float kq_sums[2];
    __shared__ float row_max_shared;
    __shared__ float denom_shared;
    __shared__ float old_scale_shared;
    __shared__ float p_shared;
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }

    const int hk = hq / gqa_ratio;
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    if (tid == 0) {
        row_max_shared = -FLT_MAX;
        denom_shared = 0.0f;
    }
    __syncthreads();

    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
        const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

        const float kq = ggml_cuda_q8k_dot4_kq_dot_block(
            q_row_payload, q_row_scales, k_row_payload, k_row_scales, kq_sums, k & 1);
        if (tid == 0) {
            const float s = kq * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
            const float next_max = fmaxf(row_max_shared, s);
            old_scale_shared = denom_shared > 0.0f ? expf(row_max_shared - next_max) : 0.0f;
            p_shared = expf(s - next_max);
            denom_shared = denom_shared * old_scale_shared + p_shared;
            row_max_shared = next_max;
        }
        __syncthreads();
        out = out * old_scale_shared + p_shared * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out / denom_shared;
    GGML_UNUSED(n_heads_k);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_tile8_parallel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ logits,
        int tile_n,
        int tile_lane) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int k_lane = tile_lane;
    float partial = 0.0f;

    if (k_lane < tile_n && lane < 8) {
        const int * k_row_payload = k_payload + k_lane * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
        const half * k_row_scales = k_scales + k_lane * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
        for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
            const int idx = qb * (QK8_0 / 4) + lane;
            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx], k_row_payload[idx], 0);
            partial += float(acc) * q_scales[qb] * __half2float(k_row_scales[qb]);
        }
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        partial += __shfl_down(partial, offset, 32);
    }
    if (lane == 0 && k_lane < tile_n) {
        logits[k_lane] = partial;
    }
    __syncthreads();
    return k_lane < tile_n ? logits[k_lane] : -INFINITY;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_fused_tile8_parallel_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int tile_lane = tid >> 5;
    __shared__ float logits[GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs[GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared;
    __shared__ float denom_shared;
    __shared__ float old_scale_shared;
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }

    const int hk = hq / gqa_ratio;
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    if (tid == 0) {
        row_max_shared = -FLT_MAX;
        denom_shared = 0.0f;
    }
    __syncthreads();

    float out = 0.0f;
    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
        const int k = k0 + tile_lane;
        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0);
        const int * k_tile_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
        const half * k_tile_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
        const float kq = ggml_cuda_q8k_dot4_kq_dot_tile8_parallel(
            q_row_payload, q_row_scales, k_tile_payload, k_tile_scales, logits, tile_n, tile_lane);
        if (tile_lane < tile_n && (tid & 31) == 0) {
            logits[tile_lane] = kq * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        }
        __syncthreads();

        if (tid == 0) {
            float tile_max = -FLT_MAX;
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                tile_max = fmaxf(tile_max, kk < tile_n ? logits[kk] : -INFINITY);
            }
            const float next_max = fmaxf(row_max_shared, tile_max);
            old_scale_shared = denom_shared > 0.0f ? expf(row_max_shared - next_max) : 0.0f;
            float tile_sum = 0.0f;
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                const float p = kk < tile_n ? expf(logits[kk] - next_max) : 0.0f;
                probs[kk] = p;
                tile_sum += p;
            }
            denom_shared = denom_shared * old_scale_shared + tile_sum;
            row_max_shared = next_max;
        }
        __syncthreads();

        out *= old_scale_shared;
#pragma unroll
        for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
            if (kk < tile_n) {
                out += probs[kk] * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, tid);
            }
        }
        __syncthreads();
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out / denom_shared;
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_grouped_gqa_online_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int hq_base = hk * gqa_ratio;
    __shared__ float kq_sums[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs_shared[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float denom_shared[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float old_scale_shared[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    if (q_row >= nq || hk >= n_heads_k || b >= batch) {
        return;
    }

    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    float out[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
#pragma unroll
    for (int h = 0; h < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++h) {
        out[h] = 0.0f;
    }
    if (tid == 0) {
#pragma unroll
        for (int h = 0; h < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++h) {
            row_max_shared[h] = -FLT_MAX;
            denom_shared[h] = 0.0f;
            old_scale_shared[h] = 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
        const int h = tid >> 5;
        const int lane = tid & 31;
        if (h < gqa_ratio) {
            const int hq = hq_base + h;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
            const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
            const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                float partial = 0.0f;
                if (kk < tile_n) {
                    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0 + kk);
                    const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                        partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                    }
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_down(partial, offset, 32);
                }
                if (lane == 0) {
                    kq_sums[h][kk] = kk < tile_n ? partial : -INFINITY;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
#pragma unroll
            for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
                if (hh < gqa_ratio) {
                    float tile_max = -FLT_MAX;
#pragma unroll
                    for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                        const float s = kk < tile_n ? kq_sums[hh][kk] * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k0 + kk, b) : -INFINITY;
                        kq_sums[hh][kk] = s;
                        tile_max = fmaxf(tile_max, s);
                    }
                    const float next_max = fmaxf(row_max_shared[hh], tile_max);
                    old_scale_shared[hh] = denom_shared[hh] > 0.0f ? expf(row_max_shared[hh] - next_max) : 0.0f;
                    float tile_sum = 0.0f;
#pragma unroll
                    for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                        const float p = kk < tile_n ? expf(kq_sums[hh][kk] - next_max) : 0.0f;
                        probs_shared[hh][kk] = p;
                        tile_sum += p;
                    }
                    denom_shared[hh] = denom_shared[hh] * old_scale_shared[hh] + tile_sum;
                    row_max_shared[hh] = next_max;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int h = 0; h < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++h) {
            if (h < gqa_ratio) {
                out[h] *= old_scale_shared[h];
            }
        }
#pragma unroll
        for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
            if (kk < tile_n) {
                const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, tid);
#pragma unroll
                for (int h = 0; h < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++h) {
                    if (h < gqa_ratio) {
                        out[h] += probs_shared[h][kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = 0; h < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++h) {
        if (h < gqa_ratio) {
            const int hq = hq_base + h;
            dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[h] / denom_shared[h];
        }
    }
}


static __global__ __launch_bounds__(512, 1) void ggml_cuda_q8k_dot4_grouped_gqa_qtile2_tile8_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_base_row = blockIdx.x * 2;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_lane = tid >> 8;
    const int q_row = q_base_row + q_lane;
    const int h = (tid >> 5) & 7;
    const int lane = tid & 31;
    const int hq_base = hk * gqa_ratio;
    __shared__ float kq_sums[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs_shared[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float denom_shared[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float old_scale_shared[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    if (hk >= n_heads_k || b >= batch) {
        return;
    }

    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    float out[GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
#pragma unroll
    for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
        out[hh] = 0.0f;
    }
    if (tid < 2 * GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA) {
        const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
        const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
        row_max_shared[qr][hh] = -FLT_MAX;
        denom_shared[qr][hh] = 0.0f;
        old_scale_shared[qr][hh] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
        if (q_row < nq && h < gqa_ratio) {
            const int hq = hq_base + h;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
            const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
            const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                float partial = 0.0f;
                if (kk < tile_n) {
                    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0 + kk);
                    const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                        partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                    }
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_down(partial, offset, 32);
                }
                if (lane == 0) {
                    kq_sums[q_lane][h][kk] = kk < tile_n ? partial : -INFINITY;
                }
            }
        }
        __syncthreads();

        if (tid < 2 * GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA) {
            const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
            const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
            const int qr_row = q_base_row + qr;
            if (qr_row < nq && hh < gqa_ratio) {
                float tile_max = -FLT_MAX;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float s = kk < tile_n ? kq_sums[qr][hh][kk] * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, qr_row, k0 + kk, b) : -INFINITY;
                    kq_sums[qr][hh][kk] = s;
                    tile_max = fmaxf(tile_max, s);
                }
                const float next_max = fmaxf(row_max_shared[qr][hh], tile_max);
                old_scale_shared[qr][hh] = denom_shared[qr][hh] > 0.0f ? expf(row_max_shared[qr][hh] - next_max) : 0.0f;
                float tile_sum = 0.0f;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float p = kk < tile_n ? expf(kq_sums[qr][hh][kk] - next_max) : 0.0f;
                    probs_shared[qr][hh][kk] = p;
                    tile_sum += p;
                }
                denom_shared[qr][hh] = denom_shared[qr][hh] * old_scale_shared[qr][hh] + tile_sum;
                row_max_shared[qr][hh] = next_max;
            }
        }
        __syncthreads();

#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
            if (q_row < nq && hh < gqa_ratio) {
                out[hh] *= old_scale_shared[q_lane][hh];
            }
        }
#pragma unroll
        for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
            if (q_row < nq && kk < tile_n) {
                const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, tid & 255);
#pragma unroll
                for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
                    if (hh < gqa_ratio) {
                        out[hh] += probs_shared[q_lane][hh][kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
        if (q_row < nq && hh < gqa_ratio) {
            const int hq = hq_base + hh;
            dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + (tid & 255)] = out[hh] / denom_shared[q_lane][hh];
        }
    }
}

static __global__ __launch_bounds__(512, 1) void ggml_cuda_q8k_dot4_grouped_gqa6_qtile4_tile8_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int batch) {
    const int tid = threadIdx.x;
    const int q_base_row = blockIdx.x * 4;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_pair_lane = tid >> 8;
    const int h = (tid >> 5) & 7;
    const int lane = tid & 31;
    const int dim = tid & 255;
    const int hq_base = hk * GGML_CUDA_Q8K_DOT4_KQ_GQA6;
    __shared__ float kq_sums[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    __shared__ float denom_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    __shared__ float old_scale_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    if (hk >= n_heads_k || b >= batch) {
        return;
    }

    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    float out[2][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
            out[phase][hh] = 0.0f;
        }
    }
    if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
        const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_GQA6;
        const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_GQA6;
        row_max_shared[qr][hh] = -FLT_MAX;
        denom_shared[qr][hh] = 0.0f;
        old_scale_shared[qr][hh] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
            if (q_row < nq && h < GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
                const int hq = hq_base + h;
                const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
                const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    float partial = 0.0f;
                    if (kk < tile_n) {
                        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0 + kk);
                        const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const int idx = lane + i * 32;
                            const int qb = idx / (QK8_0 / 4);
                            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                            partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                        }
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_down(partial, offset, 32);
                    }
                    if (lane == 0) {
                        kq_sums[q_lane][h][kk] = kk < tile_n ? partial : -INFINITY;
                    }
                }
            }
            __syncthreads();
        }

        if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
            const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_GQA6;
            const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_GQA6;
            const int qr_row = q_base_row + qr;
            if (qr_row < nq) {
                float tile_max = -FLT_MAX;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float s = kk < tile_n ? kq_sums[qr][hh][kk] * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, qr_row, k0 + kk, b) : -INFINITY;
                    kq_sums[qr][hh][kk] = s;
                    tile_max = fmaxf(tile_max, s);
                }
                const float next_max = fmaxf(row_max_shared[qr][hh], tile_max);
                old_scale_shared[qr][hh] = denom_shared[qr][hh] > 0.0f ? expf(row_max_shared[qr][hh] - next_max) : 0.0f;
                float tile_sum = 0.0f;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float p = kk < tile_n ? expf(kq_sums[qr][hh][kk] - next_max) : 0.0f;
                    probs_shared[qr][hh][kk] = p;
                    tile_sum += p;
                }
                denom_shared[qr][hh] = denom_shared[qr][hh] * old_scale_shared[qr][hh] + tile_sum;
                row_max_shared[qr][hh] = next_max;
            }
        }
        __syncthreads();

#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
#pragma unroll
            for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
                if (q_row < nq) {
                    out[phase][hh] *= old_scale_shared[q_lane][hh];
                }
            }
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                if (q_row < nq && kk < tile_n) {
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, dim);
#pragma unroll
                    for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
                        out[phase][hh] += probs_shared[q_lane][hh][kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const int q_lane = phase * 2 + q_pair_lane;
        const int q_row = q_base_row + q_lane;
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
            if (q_row < nq) {
                const int hq = hq_base + hh;
                dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + dim] = out[phase][hh] / denom_shared[q_lane][hh];
            }
        }
    }
}

static __global__ __launch_bounds__(512, 1) void ggml_cuda_q8k_dot4_grouped_gqa6_qtile4_vshared_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int batch) {
    const int tid = threadIdx.x;
    const int q_base_row = blockIdx.x * 4;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_pair_lane = tid >> 8;
    const int h = (tid >> 5) & 7;
    const int lane = tid & 31;
    const int dim = tid & 255;
    const int hq_base = hk * GGML_CUDA_Q8K_DOT4_KQ_GQA6;
    __shared__ float kq_sums[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    __shared__ float denom_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    __shared__ float old_scale_shared[4][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
    __shared__ float v_tile[GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K][GGML_CUDA_Q8K_DOT4_KQ_D];
    if (hk >= n_heads_k || b >= batch) {
        return;
    }

    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    float out[2][GGML_CUDA_Q8K_DOT4_KQ_GQA6];
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
            out[phase][hh] = 0.0f;
        }
    }
    if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
        const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_GQA6;
        const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_GQA6;
        row_max_shared[qr][hh] = -FLT_MAX;
        denom_shared[qr][hh] = 0.0f;
        old_scale_shared[qr][hh] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
            if (q_row < nq && h < GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
                const int hq = hq_base + h;
                const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
                const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    float partial = 0.0f;
                    if (kk < tile_n) {
                        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0 + kk);
                        const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const int idx = lane + i * 32;
                            const int qb = idx / (QK8_0 / 4);
                            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                            partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                        }
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_down(partial, offset, 32);
                    }
                    if (lane == 0) {
                        kq_sums[q_lane][h][kk] = kk < tile_n ? partial : -INFINITY;
                    }
                }
            }
            __syncthreads();
        }

        if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
            const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_GQA6;
            const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_GQA6;
            const int qr_row = q_base_row + qr;
            if (qr_row < nq) {
                float tile_max = -FLT_MAX;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float s = kk < tile_n ? kq_sums[qr][hh][kk] * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, qr_row, k0 + kk, b) : -INFINITY;
                    kq_sums[qr][hh][kk] = s;
                    tile_max = fmaxf(tile_max, s);
                }
                const float next_max = fmaxf(row_max_shared[qr][hh], tile_max);
                old_scale_shared[qr][hh] = denom_shared[qr][hh] > 0.0f ? expf(row_max_shared[qr][hh] - next_max) : 0.0f;
                float tile_sum = 0.0f;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float p = kk < tile_n ? expf(kq_sums[qr][hh][kk] - next_max) : 0.0f;
                    probs_shared[qr][hh][kk] = p;
                    tile_sum += p;
                }
                denom_shared[qr][hh] = denom_shared[qr][hh] * old_scale_shared[qr][hh] + tile_sum;
                row_max_shared[qr][hh] = next_max;
            }
        }
        __syncthreads();

for (int v_idx = tid; v_idx < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K * GGML_CUDA_Q8K_DOT4_KQ_D; v_idx += blockDim.x) {
            const int kk = v_idx / GGML_CUDA_Q8K_DOT4_KQ_D;
            const int v_dim = v_idx - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
            if (kk < tile_n) {
                v_tile[kk][v_dim] = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, v_dim);
            }
        }
        __syncthreads();

#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
#pragma unroll
            for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
                if (q_row < nq) {
                    out[phase][hh] *= old_scale_shared[q_lane][hh];
                }
            }
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                if (q_row < nq && kk < tile_n) {
                    const float v = v_tile[kk][dim];
#pragma unroll
                    for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
                        out[phase][hh] += probs_shared[q_lane][hh][kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const int q_lane = phase * 2 + q_pair_lane;
        const int q_row = q_base_row + q_lane;
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_GQA6; ++hh) {
            if (q_row < nq) {
                const int hq = hq_base + hh;
                dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + dim] = out[phase][hh] / denom_shared[q_lane][hh];
            }
        }
    }
}

static __global__ __launch_bounds__(512, 1) void ggml_cuda_q8k_dot4_grouped_gqa_qtile4_tile8_fattn_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_base_row = blockIdx.x * 4;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_pair_lane = tid >> 8;
    const int h = (tid >> 5) & 7;
    const int lane = tid & 31;
    const int dim = tid & 255;
    const int hq_base = hk * gqa_ratio;
    __shared__ float kq_sums[4][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float probs_shared[4][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA][GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K];
    __shared__ float row_max_shared[4][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float denom_shared[4][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    __shared__ float old_scale_shared[4][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
    if (hk >= n_heads_k || b >= batch) {
        return;
    }

    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    float out[2][GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA];
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
            out[phase][hh] = 0.0f;
        }
    }
    if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA) {
        const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
        const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
        row_max_shared[qr][hh] = -FLT_MAX;
        denom_shared[qr][hh] = 0.0f;
        old_scale_shared[qr][hh] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K) {
        const int tile_n = min(GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K, nk - k0);
#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
            if (q_row < nq && h < gqa_ratio) {
                const int hq = hq_base + h;
                const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
                const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    float partial = 0.0f;
                    if (kk < tile_n) {
                        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k0 + kk);
                        const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const int idx = lane + i * 32;
                            const int qb = idx / (QK8_0 / 4);
                            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                            partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                        }
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_down(partial, offset, 32);
                    }
                    if (lane == 0) {
                        kq_sums[q_lane][h][kk] = kk < tile_n ? partial : -INFINITY;
                    }
                }
            }
            __syncthreads();
        }

        if (tid < 4 * GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA) {
            const int qr = tid / GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
            const int hh = tid % GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA;
            const int qr_row = q_base_row + qr;
            if (qr_row < nq && hh < gqa_ratio) {
                float tile_max = -FLT_MAX;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float s = kk < tile_n ? kq_sums[qr][hh][kk] * scale + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, qr_row, k0 + kk, b) : -INFINITY;
                    kq_sums[qr][hh][kk] = s;
                    tile_max = fmaxf(tile_max, s);
                }
                const float next_max = fmaxf(row_max_shared[qr][hh], tile_max);
                old_scale_shared[qr][hh] = denom_shared[qr][hh] > 0.0f ? expf(row_max_shared[qr][hh] - next_max) : 0.0f;
                float tile_sum = 0.0f;
#pragma unroll
                for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                    const float p = kk < tile_n ? expf(kq_sums[qr][hh][kk] - next_max) : 0.0f;
                    probs_shared[qr][hh][kk] = p;
                    tile_sum += p;
                }
                denom_shared[qr][hh] = denom_shared[qr][hh] * old_scale_shared[qr][hh] + tile_sum;
                row_max_shared[qr][hh] = next_max;
            }
        }
        __syncthreads();

#pragma unroll
        for (int phase = 0; phase < 2; ++phase) {
            const int q_lane = phase * 2 + q_pair_lane;
            const int q_row = q_base_row + q_lane;
#pragma unroll
            for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
                if (q_row < nq && hh < gqa_ratio) {
                    out[phase][hh] *= old_scale_shared[q_lane][hh];
                }
            }
#pragma unroll
            for (int kk = 0; kk < GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K; ++kk) {
                if (q_row < nq && kk < tile_n) {
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k0 + kk) * nb21, nb20, dim);
#pragma unroll
                    for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
                        if (hh < gqa_ratio) {
                            out[phase][hh] += probs_shared[q_lane][hh][kk] * v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const int q_lane = phase * 2 + q_pair_lane;
        const int q_row = q_base_row + q_lane;
#pragma unroll
        for (int hh = 0; hh < GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA; ++hh) {
            if (q_row < nq && hh < gqa_ratio) {
                const int hq = hq_base + hh;
                dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + dim] = out[phase][hh] / denom_shared[q_lane][hh];
            }
        }
    }
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_direct_d64(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales) {
    // 4 independent D64 accumulators — shorter dependency chains = better ILP on RDNA3
    static constexpr int CHUNK_BLOCKS = 2;  // 2 blocks × 32 = D64
    float sum = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS / CHUNK_BLOCKS; ++chunk) {
        int acc = 0;
#pragma unroll
        for (int qb = 0; qb < CHUNK_BLOCKS; ++qb) {
            const int block = chunk * CHUNK_BLOCKS + qb;
            const int base = block * (QK8_0 / 4);
#pragma unroll
            for (int i = 0; i < QK8_0 / 4; ++i) {
                const int idx = base + i;
                acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx], k_payload[idx], acc);
            }
            sum += float(acc) * q_scales[block] * __half2float(k_scales[block]);
            acc = 0;
        }
    }
    return sum;
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_direct(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales) {
    return ggml_cuda_q8k_dot4_kq_dot_direct_d64(q_payload, q_scales, k_payload, k_scales);
}

template <int BN, bool DIRECT_OUT, bool USE_MASK, bool CAUSAL_MASK>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float       * __restrict__ partial_o,
        float       * __restrict__ partial_m,
        float       * __restrict__ partial_l,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int split_k,
        int q_offset,
        int v_reuse) {
    static constexpr int BM = 8;
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * row_m  = logits + BM * BN;
    float * row_l  = row_m + BM;
    float * old_s  = row_l + BM;
    float * v_tile = old_s + BM;
    const bool use_v_reuse = v_reuse != 0;

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM;
    const int split = int(blockIdx.y);
    const int hq = int(blockIdx.z) % n_heads_q;
    const int b = int(blockIdx.z) / n_heads_q;
    const int hk = hq / gqa_ratio;
    if (b >= batch || hk >= n_heads_k) {
        return;
    }

    const int k_begin = (int64_t(nk) * split) / split_k;
    const int k_end   = (int64_t(nk) * (split + 1)) / split_k;
    const size_t q_head_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    if (tid < BM) {
        row_m[tid] = -FLT_MAX / 2.0f;
        row_l[tid] = 0.0f;
        old_s[tid] = 0.0f;
    }

    float out[BM];
#pragma unroll
    for (int qr = 0; qr < BM; ++qr) {
        out[qr] = 0.0f;
    }
    __syncthreads();

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = min(BN, k_end - k0);
        const bool full_q = q0 + BM <= nq;
        const bool full_k = tile_n == BN && k0 + BN <= nk;
        bool exact_tile = full_q && full_k && (!USE_MASK || CAUSAL_MASK);
        if constexpr (CAUSAL_MASK) {
            if (k0 > q_offset + min(q0 + BM - 1, nq - 1)) {
                continue;
            }
            exact_tile = exact_tile && (k0 + BN - 1 <= q_offset + q0);
        }
        if constexpr (USE_MASK && !CAUSAL_MASK) {
            exact_tile = false;
        }

        if (exact_tile) {
            if (tid < BM * BN) {
                const int qr = tid / BN;
                const int kk = tid - qr * BN;
                const int q = q0 + qr;
                const int k = k0 + kk;
                const size_t q_base = q_head_base + q;
                const size_t k_base = k_head_base + k;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
            }
            __syncthreads();

            if (tid < BM) {
                float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
                }
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_l += expf(logits[tid * BN + kk] - m_new);
                }
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new;
                old_s[tid] = old_scale;
            }
            __syncthreads();

            if (use_v_reuse) {
                for (int v_idx = tid; v_idx < BN * GGML_CUDA_Q8K_DOT4_KQ_D; v_idx += blockDim.x) {
                    const int kk = v_idx / GGML_CUDA_Q8K_DOT4_KQ_D;
                    const int dim = v_idx - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
                    const int k = k0 + kk;
                    v_tile[v_idx] = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim);
                }
                __syncthreads();
            }

            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    float acc = out[qr] * old_s[qr];
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const int k = k0 + kk;
                        const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                        const float v = use_v_reuse ? v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid] :
                            ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                        acc += p * v;
                    }
                    out[qr] = acc;
                }
            }
            __syncthreads();
            continue;
        }

        if (tid < BM * BN) {
            const int qr = tid / BN;
            const int kk = tid - qr * BN;
            const int q = q0 + qr;
            const int k = k0 + kk;
            float s = -FLT_MAX / 2.0f;
            bool valid = q < nq && kk < tile_n && k < nk;
            if constexpr (CAUSAL_MASK) {
                valid = valid && k <= q_offset + q;
            }
            if (valid) {
                const size_t q_base = q_head_base + q;
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
                if constexpr (USE_MASK && !CAUSAL_MASK) {
                    s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                }
            }
            logits[tid] = s;
        }
        __syncthreads();

        if (tid < BM) {
            float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
            }
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_l += expf(logits[tid * BN + kk] - m_new);
            }
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new;
            old_s[tid] = old_scale;
        }
        __syncthreads();

        if (use_v_reuse) {
            for (int v_idx = tid; v_idx < tile_n * GGML_CUDA_Q8K_DOT4_KQ_D; v_idx += blockDim.x) {
                const int kk = v_idx / GGML_CUDA_Q8K_DOT4_KQ_D;
                const int dim = v_idx - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
                const int k = k0 + kk;
                v_tile[v_idx] = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim);
            }
            __syncthreads();
        }

        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) {
                    continue;
                }
                float acc = out[qr] * old_s[qr];
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    bool valid = kk < tile_n && k < nk;
                    if constexpr (CAUSAL_MASK) {
                        valid = valid && k <= q_offset + q;
                    }
                    if (!valid) {
                        continue;
                    }
                    const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                    const float v = use_v_reuse ? v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid] :
                        ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                    acc += p * v;
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
    }

    if constexpr (DIRECT_OUT) {
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q < nq) {
                    dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[qr] / row_l[qr];
                }
            }
        }
    } else {
        const size_t slot_base = ((size_t(b) * n_heads_q + hq) * size_t(split_k) + split) * size_t(nq);
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q < nq) {
                    partial_o[(slot_base + q) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[qr];
                }
            }
        }
        if (tid < BM) {
            const int q = q0 + tid;
            if (q < nq) {
                partial_m[slot_base + q] = row_m[tid];
                partial_l[slot_base + q] = row_l[tid];
            }
        }
    }
}

// Recthist-v3 chunked FA: hoist Q once, hoist V tile per k-iteration,
// then reuse for all BM q-rows.  Same DOT4 QK dot, softmax, and V
// accumulation as recthist_v2, but eliminates repeated global reads
// for Q payload/scales and repeated per-element q4_0 V dequant.
template <int BN, bool CAUSAL_TAIL>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_vhoist_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ partial_o,
        float       * __restrict__ partial_m,
        float       * __restrict__ partial_l,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_begin_arg,
        int k_end_arg,
        int partial_slots,
        int partial_slot) {
    static constexpr int BM = 8;
    static constexpr int I32_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 4;   // 64
    static constexpr int N_BLOCKS   = GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;   // 8
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits          = reinterpret_cast<float *>(smem);
    float * row_m           = logits + BM * BN;
    float * row_l           = row_m + BM;
    float * old_s           = row_l + BM;
    float * v_tile          = old_s + BM;
    int   * q_payload_tile  = reinterpret_cast<int  *>(v_tile + BN * GGML_CUDA_Q8K_DOT4_KQ_D);
    float * q_scales_tile   = reinterpret_cast<float *>(q_payload_tile + BM * I32_PER_ROW);

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM;
    const int hq = int(blockIdx.y);
    const int b = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) {
        return;
    }

    const int k_begin = max(0, min(k_begin_arg, nk));
    const int k_end   = max(k_begin, min(k_end_arg, nk));
    const size_t q_head_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    // Hoist Q payload + scales once into shared memory
    for (int off = tid; off < BM * I32_PER_ROW; off += blockDim.x) {
        const int qr = off / I32_PER_ROW;
        const int col = off - qr * I32_PER_ROW;
        const int q = q0 + qr;
        q_payload_tile[off] = (q < nq) ? q_payload[(q_head_base + q) * I32_PER_ROW + col] : 0;
    }
    for (int off = tid; off < BM * N_BLOCKS; off += blockDim.x) {
        const int qr = off / N_BLOCKS;
        const int blk = off - qr * N_BLOCKS;
        const int q = q0 + qr;
        q_scales_tile[off] = (q < nq) ? q_scales[(q_head_base + q) * N_BLOCKS + blk] : 1.0f;
    }

    if (tid < BM) {
        row_m[tid] = -FLT_MAX / 2.0f;
        row_l[tid] = 0.0f;
        old_s[tid] = 0.0f;
    }

    float out[BM];
#pragma unroll
    for (int qr = 0; qr < BM; ++qr) {
        out[qr] = 0.0f;
    }
    __syncthreads();

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = min(BN, k_end - k0);

        // Hoist V tile: dequant q4_0 → f32 once for this k-tile
        for (int off = tid; off < BN * GGML_CUDA_Q8K_DOT4_KQ_D; off += blockDim.x) {
            const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
            const int d  = off - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
            const int k = k0 + kk;
            v_tile[off] = (kk < tile_n && k < nk)
                ? ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d)
                : 0.0f;
        }
        __syncthreads();

        const bool full_q = q0 + BM <= nq;
        const bool full_k = tile_n == BN && k0 + BN <= k_end;
        bool exact_tile = full_q && full_k;
        if constexpr (CAUSAL_TAIL) {
            if (k0 > q_offset + min(q0 + BM - 1, nq - 1)) {
                continue;
            }
            exact_tile = exact_tile && (k0 + BN - 1 <= q_offset + q0);
        }

        // QK dot: read Q from hoisted smem, K from global (streamed)
        if (exact_tile) {
            if (tid < BM * BN) {
                const int qr = tid / BN;
                const int kk = tid - qr * BN;
                const int k = k0 + kk;
                const size_t k_base = k_head_base + k;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW,
                    q_scales_tile  + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW,
                    k_scales + k_base * N_BLOCKS) * scale;
            }
            __syncthreads();

            if (tid < BM) {
                float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
                }
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_l += expf(logits[tid * BN + kk] - m_new);
                }
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new;
                old_s[tid] = old_scale;
            }
            __syncthreads();

            // V accumulation: read pre-dequantised v_tile
            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    float acc = out[qr] * old_s[qr];
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                        acc += p * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                    }
                    out[qr] = acc;
                }
            }
            __syncthreads();
            continue;
        }

        // Edge tile: Q from smem, K from global, V from smem
        if (tid < BM * BN) {
            const int qr = tid / BN;
            const int kk = tid - qr * BN;
            const int q = q0 + qr;
            const int k = k0 + kk;
            float s = -FLT_MAX / 2.0f;
            bool valid = q < nq && kk < tile_n && k < k_end;
            if constexpr (CAUSAL_TAIL) {
                valid = valid && (k - q_offset <= q);
            }
            if (valid) {
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW,
                    q_scales_tile  + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW,
                    k_scales + k_base * N_BLOCKS) * scale;
            }
            logits[tid] = s;
        }
        __syncthreads();

        if (tid < BM) {
            float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
            }
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_l += expf(logits[tid * BN + kk] - m_new);
            }
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new;
            old_s[tid] = old_scale;
        }
        __syncthreads();

        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) {
                    continue;
                }
                float acc = out[qr] * old_s[qr];
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    bool valid = kk < tile_n && k < k_end;
                    if constexpr (CAUSAL_TAIL) {
                        valid = valid && (k - q_offset <= q);
                    }
                    if (!valid) {
                        continue;
                    }
                    const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                    acc += p * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
    }

    const size_t slot_base = ((size_t(b) * n_heads_q + hq) * size_t(partial_slots) + partial_slot) * size_t(nq);
    if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
        for (int qr = 0; qr < BM; ++qr) {
            const int q = q0 + qr;
            if (q < nq) {
                partial_o[(slot_base + q) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[qr];
            }
        }
    }
    if (tid < BM) {
        const int q = q0 + tid;
        if (q < nq) {
            partial_m[slot_base + q] = row_m[tid];
            partial_l[slot_base + q] = row_l[tid];
        }
    }
}

template <bool USE_F16_V, bool USE_Q8_V, int BN, int BM>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {
    static constexpr int BM_VAL = BM;
    static constexpr bool F16_V = USE_F16_V;
    static constexpr bool Q8_V  = USE_Q8_V;
    static constexpr int I32_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 4;
    static constexpr int N_BLOCKS   = GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    static constexpr int V_TILE_SIZE = BN * GGML_CUDA_Q8K_DOT4_KQ_D;
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits          = reinterpret_cast<float *>(smem);
    float * row_m           = logits + BM_VAL * BN;
    float * row_l           = row_m + BM_VAL;
    float * old_s           = row_l + BM_VAL;
    float * v_tile          = old_s + BM_VAL;
    float * v_tile_next     = v_tile + V_TILE_SIZE;
    int   * q_payload_tile  = reinterpret_cast<int  *>(v_tile_next + V_TILE_SIZE);
    float * q_scales_tile   = reinterpret_cast<float *>(q_payload_tile + BM_VAL * I32_PER_ROW);

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM_VAL;
    const int hq = int(blockIdx.y);
    const int b = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;

    const size_t q_head_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    const size_t k_head_base = size_t(b) * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int off = tid; off < BM_VAL * I32_PER_ROW; off += blockDim.x) {
        const int qr = off / I32_PER_ROW;
        const int q = q0 + qr;
        q_payload_tile[off] = (q < nq) ? q_payload[(q_head_base + q) * I32_PER_ROW + (off - qr * I32_PER_ROW)] : 0;
    }
    for (int off = tid; off < BM_VAL * N_BLOCKS; off += blockDim.x) {
        const int qr = off / N_BLOCKS;
        const int q = q0 + qr;
        q_scales_tile[off] = (q < nq) ? q_scales[(q_head_base + q) * N_BLOCKS + (off - qr * N_BLOCKS)] : 1.0f;
    }

    if (tid < BM_VAL) { row_m[tid] = -FLT_MAX / 2.0f; row_l[tid] = 0.0f; old_s[tid] = 0.0f; }
    float out[BM_VAL]; for (int qr = 0; qr < BM_VAL; ++qr) out[qr] = 0.0f;

    // Pre-fetch first V tile
    {
        const int k0 = 0;
        const int tile_n = min(BN, nk - k0);
        for (int off = tid; off < V_TILE_SIZE; off += blockDim.x) {
            const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
            const int k = k0 + kk;
            v_tile[off] = (kk < tile_n && k < nk)
                ? (F16_V
                    ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                    : (Q8_V
                        ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                        : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = min(BN, nk - k0);
        const bool have_next = k0 + BN < nk;
        const int next_k0 = k0 + BN;
        const int next_tile_n = min(BN, nk - next_k0);

        if (k0 > q_offset + min(q0 + BM_VAL - 1, nq - 1)) continue;
        const bool exact_tile = (q0 + BM_VAL <= nq) && (tile_n == BN) && (k0 + BN - 1 <= q_offset + q0);

        if (exact_tile) {
            // QK dot on threads 0-63, V pre-fetch on 64-255 (if have_next)
            if (tid < BM_VAL * BN) {
                const int qr = tid / BN, kk = tid - qr * BN;
                const size_t k_base = k_head_base + k0 + kk;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW, q_scales_tile + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW, k_scales + k_base * N_BLOCKS) * scale;
            } else if (have_next && tid >= BM_VAL * BN) {
                // Pre-fetch next V tile while QK dot runs
                const int pf_start = BM_VAL * BN;
                const int num_pf = blockDim.x - pf_start;
                for (int off = tid - pf_start; off < V_TILE_SIZE; off += num_pf) {
                    const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
                    const int k = next_k0 + kk;
                    v_tile_next[off] = (kk < next_tile_n && k < nk)
                        ? (F16_V
                            ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                            : (Q8_V
                                ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                                : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
                }
            }
            __syncthreads();
            // Softmax
            if (tid < BM_VAL) {
                float tile_m = -FLT_MAX / 2.0f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) tile_l += expf(logits[tid * BN + kk] - m_new);
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new; old_s[tid] = old_scale;
            }
            __syncthreads();
            // V accumulation from current v_tile
            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
                for (int qr = 0; qr < BM_VAL; ++qr) {
                    float acc = out[qr] * old_s[qr];
                    for (int kk = 0; kk < BN; ++kk)
                        acc += expf(logits[qr * BN + kk] - row_m[qr]) * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                    out[qr] = acc;
                }
            }
            __syncthreads();
            // Swap V tiles for next iteration
            if (have_next) {
                float * tmp = v_tile;
                v_tile = v_tile_next;
                v_tile_next = tmp;
            }
            __syncthreads();
            continue;
        }

        // Edge tile: V pre-fetch on threads 64-255 in parallel with QK dot
        if (tid < BM_VAL * BN) {
            const int qr = tid / BN, kk = tid - qr * BN, q = q0 + qr, k = k0 + kk;
            float s = -FLT_MAX / 2.0f;
            if (q < nq && kk < tile_n && k < nk && (k - q_offset <= q)) {
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW, q_scales_tile + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW, k_scales + k_base * N_BLOCKS) * scale;
            }
            logits[tid] = s;
        } else if (have_next && tid >= BM_VAL * BN) {
            const int pf_start = BM_VAL * BN;
            const int num_pf = blockDim.x - pf_start;
            for (int off = tid - pf_start; off < V_TILE_SIZE; off += num_pf) {
                const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
                const int k = next_k0 + kk;
                v_tile_next[off] = (kk < next_tile_n && k < nk)
                    ? (F16_V
                        ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                        : (Q8_V
                            ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                            : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
            }
        }
        __syncthreads();
        if (tid < BM_VAL) {
            float tile_m = -FLT_MAX / 2.0f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) tile_l += expf(logits[tid * BN + kk] - m_new);
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new; old_s[tid] = old_scale;
        }
        __syncthreads();
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
            for (int qr = 0; qr < BM_VAL; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) continue;
                float acc = out[qr] * old_s[qr];
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    if (kk >= tile_n || k >= nk || (k - q_offset > q)) continue;
                    acc += expf(logits[qr * BN + kk] - row_m[qr]) * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
        if (have_next) {
            float * tmp = v_tile;
            v_tile = v_tile_next;
            v_tile_next = tmp;
        }
        __syncthreads();
    }

    if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
        for (int qr = 0; qr < BM_VAL; ++qr) {
            const int q = q0 + qr;
            if (q < nq)
                dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] =
                    out[qr] / fmaxf(row_l[qr], 1.0e-20f);
        }
    }
}

template <int BN, bool CAUSAL_TAIL>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ partial_o,
        float       * __restrict__ partial_m,
        float       * __restrict__ partial_l,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_begin_arg,
        int k_end_arg,
        int partial_slots,
        int partial_slot) {
    static constexpr int BM = 8;
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * row_m  = logits + BM * BN;
    float * row_l  = row_m + BM;
    float * old_s  = row_l + BM;

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM;
    const int hq = int(blockIdx.y);
    const int b = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) {
        return;
    }

    const int k_begin = max(0, min(k_begin_arg, nk));
    const int k_end   = max(k_begin, min(k_end_arg, nk));
    const size_t q_head_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    if (tid < BM) {
        row_m[tid] = -FLT_MAX / 2.0f;
        row_l[tid] = 0.0f;
        old_s[tid] = 0.0f;
    }

    float out[BM];
#pragma unroll
    for (int qr = 0; qr < BM; ++qr) {
        out[qr] = 0.0f;
    }
    __syncthreads();

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = min(BN, k_end - k0);
        const bool full_q = q0 + BM <= nq;
        const bool full_k = tile_n == BN && k0 + BN <= k_end;
        bool exact_tile = full_q && full_k;
        if constexpr (CAUSAL_TAIL) {
            if (k0 > q_offset + min(q0 + BM - 1, nq - 1)) {
                continue;
            }
            exact_tile = exact_tile && (k0 + BN - 1 <= q_offset + q0);
        }

        if (exact_tile) {
            if (tid < BM * BN) {
                const int qr = tid / BN;
                const int kk = tid - qr * BN;
                const int q = q0 + qr;
                const int k = k0 + kk;
                const size_t q_base = q_head_base + q;
                const size_t k_base = k_head_base + k;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
            }
            __syncthreads();

            if (tid < BM) {
                float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
                }
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_l += expf(logits[tid * BN + kk] - m_new);
                }
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new;
                old_s[tid] = old_scale;
            }
            __syncthreads();

            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    float acc = out[qr] * old_s[qr];
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const int k = k0 + kk;
                        const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                        const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                        acc += p * v;
                    }
                    out[qr] = acc;
                }
            }
            __syncthreads();
            continue;
        }

        if (tid < BM * BN) {
            const int qr = tid / BN;
            const int kk = tid - qr * BN;
            const int q = q0 + qr;
            const int k = k0 + kk;
            float s = -FLT_MAX / 2.0f;
            bool valid = q < nq && kk < tile_n && k < k_end;
            if constexpr (CAUSAL_TAIL) {
                valid = valid && (k - q_offset <= q);
            }
            if (valid) {
                const size_t q_base = q_head_base + q;
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
            }
            logits[tid] = s;
        }
        __syncthreads();

        if (tid < BM) {
            float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
            }
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_l += expf(logits[tid * BN + kk] - m_new);
            }
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new;
            old_s[tid] = old_scale;
        }
        __syncthreads();

        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) {
                    continue;
                }
                float acc = out[qr] * old_s[qr];
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    bool valid = kk < tile_n && k < k_end;
                    if constexpr (CAUSAL_TAIL) {
                        valid = valid && (k - q_offset <= q);
                    }
                    if (!valid) {
                        continue;
                    }
                    const float p = expf(logits[qr * BN + kk] - row_m[qr]);
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                    acc += p * v;
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
    }

    const size_t slot_base = ((size_t(b) * n_heads_q + hq) * size_t(partial_slots) + partial_slot) * size_t(nq);
    if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
        for (int qr = 0; qr < BM; ++qr) {
            const int q = q0 + qr;
            if (q < nq) {
                partial_o[(slot_base + q) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[qr];
            }
        }
    }
    if (tid < BM) {
        const int q = q0 + tid;
        if (q < nq) {
            partial_m[slot_base + q] = row_m[tid];
            partial_l[slot_base + q] = row_l[tid];
        }
    }
}

template <int BN, bool DIRECT_OUT, bool USE_MASK, bool CAUSAL_MASK, int GH, bool K_REUSE, bool WARP_KQ, bool WARP_ROW_KQ>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_hpair_q4_0_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float       * __restrict__ partial_o,
        float       * __restrict__ partial_m,
        float       * __restrict__ partial_l,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int split_k,
        int q_offset,
        int head_groups_per_batch,
        int head_group_stride,
        int head_group_offset) {
    static constexpr int BM = 8;
    static constexpr int ROWS = GH * BM;
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * row_m  = logits + ROWS * BN;
    float * row_l  = row_m + ROWS;
    float * old_s  = row_l + ROWS;
    float * v_tile = old_s + ROWS;
    int   * k_tile_payload = reinterpret_cast<int *>(v_tile + BN * GGML_CUDA_Q8K_DOT4_KQ_D);
    half  * k_tile_scales  = reinterpret_cast<half *>(k_tile_payload + BN * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM;
    const int split = int(blockIdx.y);
    const int hgroup = int(blockIdx.z) % head_groups_per_batch;
    const int b = int(blockIdx.z) / head_groups_per_batch;
    const int hq0 = hgroup * head_group_stride + head_group_offset;
    const int hk = hq0 / gqa_ratio;
    if (b >= batch || hk >= n_heads_k || hq0 + GH > n_heads_q || (hq0 + GH - 1) / gqa_ratio != hk) {
        return;
    }

    const int k_begin = (int64_t(nk) * split) / split_k;
    const int k_end   = (int64_t(nk) * (split + 1)) / split_k;
    const size_t k_head_base = ((size_t(b) * n_heads_k + hk) * size_t(nk));
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    if (tid < ROWS) {
        row_m[tid] = -FLT_MAX / 2.0f;
        row_l[tid] = 0.0f;
        old_s[tid] = 0.0f;
    }

    float out[GH][BM];
#pragma unroll
    for (int gh = 0; gh < GH; ++gh) {
#pragma unroll
        for (int qr = 0; qr < BM; ++qr) {
            out[gh][qr] = 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = min(BN, k_end - k0);
        const bool full_q = q0 + BM <= nq;
        const bool full_k = tile_n == BN && k0 + BN <= nk;
        bool exact_tile = full_q && full_k && (!USE_MASK || CAUSAL_MASK);
        if constexpr (CAUSAL_MASK) {
            if (k0 > q_offset + min(q0 + BM - 1, nq - 1)) {
                continue;
            }
            exact_tile = exact_tile && (k0 + BN - 1 <= q_offset + q0);
        }
        if constexpr (USE_MASK && !CAUSAL_MASK) {
            exact_tile = false;
        }

        if constexpr (K_REUSE) {
            for (int idx = tid; idx < BN * (GGML_CUDA_Q8K_DOT4_KQ_D / 4); idx += blockDim.x) {
                const int kk = idx / (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                const int col = idx - kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                const int k = k0 + kk;
                if (kk < tile_n) {
                    const size_t k_base = k_head_base + k;
                    k_tile_payload[idx] = k_payload[k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + col];
                }
            }
            for (int idx = tid; idx < BN * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; idx += blockDim.x) {
                const int kk = idx / GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                const int qb = idx - kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                const int k = k0 + kk;
                if (kk < tile_n) {
                    const size_t k_base = k_head_base + k;
                    k_tile_scales[idx] = k_scales[k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qb];
                }
            }
            __syncthreads();
        }

        if (exact_tile) {
            if constexpr (WARP_ROW_KQ) {
                const int lane = tid & 31;
                const int warp = tid >> 5;
                for (int ri = warp; ri < ROWS; ri += 8) {
                    const int gh = ri / BM;
                    const int qr = ri - gh * BM;
                    const int q = q0 + qr;
                    const int hq = hq0 + gh;
                    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                    float partial[BN];
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        partial[kk] = 0.0f;
                    }
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int q_word = q_row_payload[idx];
                        const float q_scale = q_row_scales[qb];
#pragma unroll
                        for (int kk = 0; kk < BN; ++kk) {
                            const int k = k0 + kk;
                            const size_t k_base = k_head_base + k;
                            const int * k_row_payload = K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                            const half * k_row_scales = K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_word, k_row_payload[idx], 0);
                            partial[kk] += float(acc) * q_scale * __half2float(k_row_scales[qb]);
                        }
                    }
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        float s = partial[kk];
#pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1) {
                            s += __shfl_down(s, offset, 32);
                        }
                        if (lane == 0) {
                            logits[ri * BN + kk] = s * scale;
                        }
                    }
                }
            } else if constexpr (WARP_KQ) {
                const int lane = tid & 31;
                const int warp = tid >> 5;
                for (int li = warp; li < ROWS * BN; li += 8) {
                    const int gh = li / (BM * BN);
                    const int rem = li - gh * BM * BN;
                    const int qr = rem / BN;
                    const int kk = rem - qr * BN;
                    const int q = q0 + qr;
                    const int k = k0 + kk;
                    const int hq = hq0 + gh;
                    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                    const size_t k_base = k_head_base + k;
                    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                    const int * k_row_payload = K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const half * k_row_scales = K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                        partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_down(partial, offset, 32);
                    }
                    if (lane == 0) {
                        logits[li] = partial * scale;
                    }
                }
            } else if (tid < ROWS * BN) {
                const int gh = tid / (BM * BN);
                const int rem = tid - gh * BM * BN;
                const int qr = rem / BN;
                const int kk = rem - qr * BN;
                const int q = q0 + qr;
                const int k = k0 + kk;
                const int hq = hq0 + gh;
                const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                const size_t k_base = k_head_base + k;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
            }
            __syncthreads();

            if (tid < ROWS) {
                const int gh = tid / BM;
                const int qr = tid - gh * BM;
                float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_m = fmaxf(tile_m, logits[(gh * BM + qr) * BN + kk]);
                }
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_l += expf(logits[(gh * BM + qr) * BN + kk] - m_new);
                }
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new;
                old_s[tid] = old_scale;
            }
            __syncthreads();

            for (int v_idx = tid; v_idx < BN * GGML_CUDA_Q8K_DOT4_KQ_D; v_idx += blockDim.x) {
                const int kk = v_idx / GGML_CUDA_Q8K_DOT4_KQ_D;
                const int dim = v_idx - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
                const int k = k0 + kk;
                v_tile[v_idx] = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim);
            }
            __syncthreads();

            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
                for (int gh = 0; gh < GH; ++gh) {
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int ri = gh * BM + qr;
                        float acc = out[gh][qr] * old_s[ri];
#pragma unroll
                        for (int kk = 0; kk < BN; ++kk) {
                            const float p = expf(logits[ri * BN + kk] - row_m[ri]);
                            acc += p * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                        }
                        out[gh][qr] = acc;
                    }
                }
            }
            __syncthreads();
            continue;
        }

        if constexpr (WARP_ROW_KQ) {
            const int lane = tid & 31;
            const int warp = tid >> 5;
            for (int ri = warp; ri < ROWS; ri += 8) {
                const int gh = ri / BM;
                const int qr = ri - gh * BM;
                const int q = q0 + qr;
                const int hq = hq0 + gh;
                const bool q_valid = q < nq;
                const int * q_row_payload = q_payload;
                const float * q_row_scales = q_scales;
                if (q_valid) {
                    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                    q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                }
                float partial[BN];
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    partial[kk] = 0.0f;
                }
                if (q_valid) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int q_word = q_row_payload[idx];
                        const float q_scale = q_row_scales[qb];
#pragma unroll
                        for (int kk = 0; kk < BN; ++kk) {
                            const int k = k0 + kk;
                            bool valid = kk < tile_n && k < nk;
                            if constexpr (CAUSAL_MASK) {
                                valid = valid && k <= q_offset + q;
                            }
                            if (valid) {
                                const size_t k_base = k_head_base + k;
                                const int * k_row_payload = K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                                const half * k_row_scales = K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                                const int acc = ggml_cuda_q8k_dot4_i8_i8(q_word, k_row_payload[idx], 0);
                                partial[kk] += float(acc) * q_scale * __half2float(k_row_scales[qb]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    bool valid = q_valid && kk < tile_n && k < nk;
                    if constexpr (CAUSAL_MASK) {
                        valid = valid && k <= q_offset + q;
                    }
                    float s = partial[kk];
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        s += __shfl_down(s, offset, 32);
                    }
                    if (lane == 0) {
                        if (valid) {
                            s *= scale;
                            if constexpr (USE_MASK && !CAUSAL_MASK) {
                                s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                            }
                        } else {
                            s = -FLT_MAX / 2.0f;
                        }
                        logits[ri * BN + kk] = s;
                    }
                }
            }
        } else if constexpr (WARP_KQ) {
            const int lane = tid & 31;
            const int warp = tid >> 5;
            for (int li = warp; li < ROWS * BN; li += 8) {
                const int gh = li / (BM * BN);
                const int rem = li - gh * BM * BN;
                const int qr = rem / BN;
                const int kk = rem - qr * BN;
                const int q = q0 + qr;
                const int k = k0 + kk;
                const int hq = hq0 + gh;
                float s = -FLT_MAX / 2.0f;
                bool valid = q < nq && kk < tile_n && k < nk;
                if constexpr (CAUSAL_MASK) {
                    valid = valid && k <= q_offset + q;
                }
                if (valid) {
                    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                    const size_t k_base = k_head_base + k;
                    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                    const int * k_row_payload = K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
                    const half * k_row_scales = K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int idx = lane + i * 32;
                        const int qb = idx / (QK8_0 / 4);
                        const int acc = ggml_cuda_q8k_dot4_i8_i8(q_row_payload[idx], k_row_payload[idx], 0);
                        partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        partial += __shfl_down(partial, offset, 32);
                    }
                    if (lane == 0) {
                        s = partial * scale;
                        if constexpr (USE_MASK && !CAUSAL_MASK) {
                            s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                        }
                    }
                }
                if (lane == 0) {
                    logits[li] = s;
                }
            }
        } else if (tid < ROWS * BN) {
            const int gh = tid / (BM * BN);
            const int rem = tid - gh * BM * BN;
            const int qr = rem / BN;
            const int kk = rem - qr * BN;
            const int q = q0 + qr;
            const int k = k0 + kk;
            const int hq = hq0 + gh;
            float s = -FLT_MAX / 2.0f;
            bool valid = q < nq && kk < tile_n && k < nk;
            if constexpr (CAUSAL_MASK) {
                valid = valid && k <= q_offset + q;
            }
            if (valid) {
                const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq)) + q;
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS,
                    K_REUSE ? k_tile_payload + kk * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) : k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4),
                    K_REUSE ? k_tile_scales + kk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS : k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) * scale;
                if constexpr (USE_MASK && !CAUSAL_MASK) {
                    s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                }
            }
            logits[tid] = s;
        }
        __syncthreads();

        if (tid < ROWS) {
            const int gh = tid / BM;
            const int qr = tid - gh * BM;
            float tile_m = -FLT_MAX / 2.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_m = fmaxf(tile_m, logits[(gh * BM + qr) * BN + kk]);
            }
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                tile_l += expf(logits[(gh * BM + qr) * BN + kk] - m_new);
            }
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new;
            old_s[tid] = old_scale;
        }
        __syncthreads();

        for (int v_idx = tid; v_idx < tile_n * GGML_CUDA_Q8K_DOT4_KQ_D; v_idx += blockDim.x) {
            const int kk = v_idx / GGML_CUDA_Q8K_DOT4_KQ_D;
            const int dim = v_idx - kk * GGML_CUDA_Q8K_DOT4_KQ_D;
            const int k = k0 + kk;
            v_tile[v_idx] = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim);
        }
        __syncthreads();

        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int gh = 0; gh < GH; ++gh) {
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    if (q >= nq) {
                        continue;
                    }
                    const int ri = gh * BM + qr;
                    float acc = out[gh][qr] * old_s[ri];
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const int k = k0 + kk;
                        bool valid = kk < tile_n && k < nk;
                        if constexpr (CAUSAL_MASK) {
                            valid = valid && k <= q_offset + q;
                        }
                        if (!valid) {
                            continue;
                        }
                        const float p = expf(logits[ri * BN + kk] - row_m[ri]);
                        acc += p * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                    }
                    out[gh][qr] = acc;
                }
            }
        }
        __syncthreads();
    }

    if constexpr (DIRECT_OUT) {
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int gh = 0; gh < GH; ++gh) {
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    if (q < nq) {
                        const int hq = hq0 + gh;
                        const int ri = gh * BM + qr;
                        dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[gh][qr] / row_l[ri];
                    }
                }
            }
        }
    } else {
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
#pragma unroll
            for (int gh = 0; gh < GH; ++gh) {
                const int hq = hq0 + gh;
                const size_t slot_base = ((size_t(b) * n_heads_q + hq) * size_t(split_k) + split) * size_t(nq);
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    if (q < nq) {
                        partial_o[(slot_base + q) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out[gh][qr];
                    }
                }
            }
        }
        if (tid < ROWS) {
            const int gh = tid / BM;
            const int qr = tid - gh * BM;
            const int q = q0 + qr;
            if (q < nq) {
                const int hq = hq0 + gh;
                const size_t slot_base = ((size_t(b) * n_heads_q + hq) * size_t(split_k) + split) * size_t(nq);
                partial_m[slot_base + q] = row_m[tid];
                partial_l[slot_base + q] = row_l[tid];
            }
        }
    }
}

static __global__ void ggml_cuda_q8k_dot4_blockfa_reset_partial_kernel(
        float  * __restrict__ partial_m,
        float  * __restrict__ partial_l,
        size_t rows) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    partial_m[i] = -FLT_MAX / 2.0f;
    partial_l[i] = 0.0f;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_combine_kernel(
        const float * __restrict__ partial_o,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float       * __restrict__ dst,
        int nq,
        int n_heads_q,
        int split_k) {
    const int q = int(blockIdx.x);
    const int hq = int(blockIdx.y);
    const int b = int(blockIdx.z);
    const int dim = int(threadIdx.x);
    const size_t slot0 = ((size_t(b) * n_heads_q + hq) * size_t(split_k)) * size_t(nq) + q;
    __shared__ float m_shared;
    __shared__ float l_shared;

    if (threadIdx.x == 0) {
        float m = -FLT_MAX / 2.0f;
        for (int s = 0; s < split_k; ++s) {
            const float l_s = partial_l[slot0 + size_t(s) * nq];
            if (l_s > 0.0f) {
                m = fmaxf(m, partial_m[slot0 + size_t(s) * nq]);
            }
        }
        float l = 0.0f;
        for (int s = 0; s < split_k; ++s) {
            const float l_s = partial_l[slot0 + size_t(s) * nq];
            if (l_s > 0.0f) {
                l += l_s * expf(partial_m[slot0 + size_t(s) * nq] - m);
            }
        }
        m_shared = m;
        l_shared = fmaxf(l, 1.0e-20f);
    }
    __syncthreads();

    if (dim < GGML_CUDA_Q8K_DOT4_KQ_D) {
        float out = 0.0f;
        for (int s = 0; s < split_k; ++s) {
            const float l_s = partial_l[slot0 + size_t(s) * nq];
            if (l_s > 0.0f) {
                out += partial_o[(slot0 + size_t(s) * nq) * GGML_CUDA_Q8K_DOT4_KQ_D + dim] *
                    expf(partial_m[slot0 + size_t(s) * nq] - m_shared);
            }
        }
        dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + dim] = out / l_shared;
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_reference_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ K,
        float       * __restrict__ logits,
        float scale,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q_row) * nb01);

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const char * k_blk = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k_row) * nb11 + int64_t(qb) * nb10;
        const half ks_h = ggml_cuda_q8k_dot4_load_half_unaligned(k_blk);
        const float ks = __half2float(ks_h);
#pragma unroll
        for (int lane = 0; lane < QK8_0; ++lane) {
            const int8_t kval = *(const int8_t *) (k_blk + sizeof(half) + lane);
            sum += q_ptr[qb * QK8_0 + lane] * float(kval) * ks;
        }
    }
    logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    GGML_UNUSED(n_heads_k);
}

static __global__ void ggml_cuda_q8k_dot4_kq_error_kernel(
        const float * __restrict__ candidate,
        const float * __restrict__ reference,
        float       * __restrict__ metrics,
        size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float diff = fabsf(candidate[i] - reference[i]);
    atomicAdd(metrics + 0, diff * diff);
    atomicAdd(metrics + 1, diff);
    atomicAdd(metrics + 2, diff > 0.5f ? 1.0f : 0.0f);
}

#include "fattn-dot4-q8k-decode.cuh"

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];

    // ── Launch proof: DOT4 actually dispatched ───────────────────
    if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
        if (log_env && atoi(log_env) != 0) {
            GGML_LOG_INFO(
                "fa_dot4_launch: fa_inst=%d nq=%lld nk=%lld d=%lld K=%s V=%s\n",
                fa_inst_i32,
                (long long) Q->ne[1],
                (long long) K->ne[1],
                (long long) Q->ne[0],
                ggml_type_name(K->type),
                ggml_type_name(V->type));
        }
    }

    // I32 packed16 K contract: DOT4-dispatch path only.
    const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;

    if (k_is_i32_packed16) {
        GGML_ASSERT(Q->type == GGML_TYPE_F32);
        GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0);
        if (V->type != GGML_TYPE_F16) {
            GGML_ASSERT(k_is_i32_packed16 && "quantized V with DOT4 FA requires packed16 K (I32 type)");
        }
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(K->ne[0] * 4 == Q->ne[0]);  // D/4 * 4 == D
        GGML_ASSERT(V->ne[0] == Q->ne[0]);
        GGML_ASSERT(K->ne[1] > 0);
        GGML_ASSERT(K->ne[2] > 0);
        GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
        GGML_ASSERT(K->data != nullptr);
        GGML_ASSERT(V->data != nullptr);
        GGML_ASSERT(dst->data != nullptr);
    }

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_payload(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);
    ggml_cuda_pool_alloc<int>   k_payload(pool);
    ggml_cuda_pool_alloc<half>  k_scales(pool);
    ggml_cuda_pool_alloc<float> logits(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_o(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_m(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_l(pool);
    ggml_cuda_pool_alloc<float> ref_logits(pool);
    ggml_cuda_pool_alloc<float> metrics(pool);

    const size_t q_rows = (size_t) batch * n_heads_q * nq;
    const size_t k_rows = (size_t) batch * n_heads_k * nk;
    const size_t logits_ne = (size_t) batch * n_heads_q * nq * nk;
    const char * check_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_CHECK");
    const bool check = check_env && atoi(check_env) != 0;
    const char * variant_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT");
    const bool q8block_variant = variant_env && strcmp(variant_env, "q8block") == 0;
    const bool fused_online = variant_env && strcmp(variant_env, "fused_online") == 0;
    const bool fused_tile8_parallel = variant_env && strcmp(variant_env, "fused_tile8_parallel") == 0;
    const bool grouped_gqa_online = variant_env && strcmp(variant_env, "grouped_gqa_online") == 0;
    const bool grouped_gqa_qtile2_tile8 = variant_env && strcmp(variant_env, "grouped_gqa_qtile2_tile8") == 0;
    const bool grouped_gqa_qtile4_tile8 = variant_env && strcmp(variant_env, "grouped_gqa_qtile4_tile8") == 0;
    const bool grouped_gqa6_qtile4_tile8 = variant_env && strcmp(variant_env, "grouped_gqa6_qtile4_tile8") == 0;
    const bool grouped_gqa6_qtile4_vshared = variant_env && strcmp(variant_env, "grouped_gqa6_qtile4_vshared") == 0;
    const bool blockfa_hybrid_bm8 = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8") == 0;
    const bool blockfa_hybrid_bm8_packed16_scalar = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_packed16_scalar") == 0;
    const bool blockfa_hybrid_bm8_vreuse = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_vreuse") == 0;
    const bool blockfa_hybrid_bm8_hpair = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_hpair") == 0;
    const bool blockfa_hybrid_bm8_hpair_kreuse = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_hpair_kreuse") == 0;
    const bool blockfa_hybrid_bm8_hpair_warpkq = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_hpair_warpkq") == 0;
    const bool blockfa_hybrid_bm8_hpair_rowwarpkq = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_hpair_rowwarpkq") == 0;
    const bool blockfa_hybrid_bm8_htriad = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_htriad") == 0;
    const bool blockfa_hybrid_bm8_htriad_kreuse = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_htriad_kreuse") == 0;
    const bool blockfa_hybrid_bm8_hquad = variant_env && strcmp(variant_env, "blockfa_hybrid_bm8_hquad") == 0;
    const bool blockfa_recthist_v2 = variant_env && strcmp(variant_env, "blockfa_recthist_v2") == 0;
    const bool blockfa_recthist_v2_bn16 = variant_env && strcmp(variant_env, "blockfa_recthist_v2_bn16") == 0;
    const bool blockfa_recthist_v3_vhoist = variant_env && strcmp(variant_env, "blockfa_recthist_v3_vhoist") == 0;
    const bool blockfa_recthist_v4_single = variant_env && strcmp(variant_env, "blockfa_recthist_v4_single") == 0;
    const bool blockfa_recthist_v2_any = blockfa_recthist_v2 || blockfa_recthist_v2_bn16;
    const bool blockfa_recthist_any_explicit = blockfa_recthist_v2_any || blockfa_recthist_v3_vhoist || blockfa_recthist_v4_single;
    const bool blockfa_hybrid_bm8_hgroup = blockfa_hybrid_bm8_hpair || blockfa_hybrid_bm8_hpair_kreuse || blockfa_hybrid_bm8_hpair_warpkq || blockfa_hybrid_bm8_hpair_rowwarpkq || blockfa_hybrid_bm8_htriad || blockfa_hybrid_bm8_htriad_kreuse || blockfa_hybrid_bm8_hquad;
    const bool blockfa_hybrid_bm8_kreuse = blockfa_hybrid_bm8_hpair_kreuse || blockfa_hybrid_bm8_htriad_kreuse;
    const bool blockfa_hybrid_bm8_any = blockfa_hybrid_bm8 || blockfa_hybrid_bm8_packed16_scalar || blockfa_hybrid_bm8_vreuse || blockfa_hybrid_bm8_hgroup;
    const bool blockfa_runtime_any_explicit = blockfa_hybrid_bm8_any || blockfa_recthist_any_explicit;
    const bool fused_variant_explicit = fused_online || fused_tile8_parallel || grouped_gqa_online || grouped_gqa_qtile2_tile8 || grouped_gqa_qtile4_tile8 || grouped_gqa6_qtile4_tile8 || grouped_gqa6_qtile4_vshared || blockfa_runtime_any_explicit;
    const bool full_fa = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA");

    // Auto-select blockfa_recthist_v4_single for f16/q8_0/q4_0 V when FULL_FA=1
    // and no explicit variant is set. The default packed16 path for f16/q8_0 would
    // read V data as q4_0 blocks (incorrect); and the default q4_0 path uses the
    // legacy fattn_from_logits kernel which has poor occupancy at scale.
    // blockfa_recthist_v4_single handles all three V types natively.
    const bool blockfa_recthist_v4_single_auto =
        full_fa && !fused_variant_explicit &&
        (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0);

    const bool blockfa_recthist_v4_single_effective = blockfa_recthist_v4_single || blockfa_recthist_v4_single_auto;
    const bool blockfa_recthist_any_effective = blockfa_recthist_v2_any || blockfa_recthist_v3_vhoist || blockfa_recthist_v4_single_effective;
    const bool blockfa_runtime_any_effective = blockfa_hybrid_bm8_any || blockfa_recthist_any_effective;
    const bool fused_variant_effective = fused_online || fused_tile8_parallel || grouped_gqa_online || grouped_gqa_qtile2_tile8 || grouped_gqa_qtile4_tile8 || grouped_gqa6_qtile4_tile8 || grouped_gqa6_qtile4_vshared || blockfa_runtime_any_effective;

    const char * variant_name = blockfa_recthist_v4_single_auto && !blockfa_recthist_v4_single ? "blockfa_recthist_v4_single_auto" :
            (blockfa_recthist_v4_single ? "blockfa_recthist_v4_single" :
            (blockfa_recthist_v2_bn16 ? "blockfa_recthist_v2_bn16" :
            (blockfa_recthist_v2 ? "blockfa_recthist_v2" :
            (blockfa_hybrid_bm8_hpair_rowwarpkq ? "blockfa_hybrid_bm8_hpair_rowwarpkq" :
            (blockfa_hybrid_bm8_hpair_warpkq ? "blockfa_hybrid_bm8_hpair_warpkq" :
            (blockfa_hybrid_bm8_htriad_kreuse ? "blockfa_hybrid_bm8_htriad_kreuse" :
            (blockfa_hybrid_bm8_hpair_kreuse ? "blockfa_hybrid_bm8_hpair_kreuse" :
            (blockfa_hybrid_bm8_hquad ? "blockfa_hybrid_bm8_hquad" :
            (blockfa_hybrid_bm8_htriad ? "blockfa_hybrid_bm8_htriad" :
            (blockfa_hybrid_bm8_hpair ? "blockfa_hybrid_bm8_hpair" :
            (blockfa_hybrid_bm8_vreuse ? "blockfa_hybrid_bm8_vreuse" :
            (blockfa_hybrid_bm8_packed16_scalar ? "blockfa_hybrid_bm8_packed16_scalar" :
            (blockfa_hybrid_bm8 ? "blockfa_hybrid_bm8" :
            (grouped_gqa6_qtile4_vshared ? "grouped_gqa6_qtile4_vshared" :
            (grouped_gqa6_qtile4_tile8 ? "grouped_gqa6_qtile4_tile8" :
            (grouped_gqa_qtile4_tile8 ? "grouped_gqa_qtile4_tile8" :
            (grouped_gqa_qtile2_tile8 ? "grouped_gqa_qtile2_tile8" :
            (grouped_gqa_online ? "grouped_gqa_online" :
            (fused_tile8_parallel ? "fused_tile8_parallel" :
            (fused_online ? "fused_online" :
            (q8block_variant ? "q8block" : "packed16")))))))))))))))))))));
    const char * variant_note = blockfa_recthist_v4_single_effective ? "blockfa_recthist_v4_single_probe" :
            (blockfa_recthist_v2_bn16 ? "blockfa_recthist_v2_bn16_probe" :
            (blockfa_recthist_v2 ? "blockfa_recthist_v2_probe" :
            (blockfa_hybrid_bm8_hpair_rowwarpkq ? "blockfa_hybrid_bm8_hpair_rowwarpkq_probe" :
            (blockfa_hybrid_bm8_hpair_warpkq ? "blockfa_hybrid_bm8_hpair_warpkq_probe" :
            (blockfa_hybrid_bm8_htriad_kreuse ? "blockfa_hybrid_bm8_htriad_kreuse_probe" :
            (blockfa_hybrid_bm8_hpair_kreuse ? "blockfa_hybrid_bm8_hpair_kreuse_probe" :
            (blockfa_hybrid_bm8_hquad ? "blockfa_hybrid_bm8_hquad_probe" :
            (blockfa_hybrid_bm8_htriad ? "blockfa_hybrid_bm8_htriad_probe" :
            (blockfa_hybrid_bm8_hpair ? "blockfa_hybrid_bm8_hpair_probe" :
            (blockfa_hybrid_bm8_vreuse ? "blockfa_hybrid_bm8_vreuse_probe" :
            (blockfa_hybrid_bm8_packed16_scalar ? "blockfa_hybrid_bm8_packed16_scalar_probe" :
            (blockfa_hybrid_bm8 ? "blockfa_hybrid_bm8_probe" :
            (grouped_gqa6_qtile4_vshared ? "grouped_gqa6_qtile4_vshared_probe" :
            (grouped_gqa6_qtile4_tile8 ? "grouped_gqa6_qtile4_tile8_probe" :
            (grouped_gqa_qtile4_tile8 ? "grouped_gqa_qtile4_tile8_probe" :
            (grouped_gqa_qtile2_tile8 ? "grouped_gqa_qtile2_tile8_probe" :
            (grouped_gqa_online ? "grouped_gqa_online_probe" :
            (fused_tile8_parallel ? "fused_tile8_parallel_probe" :
            (fused_online ? "fused_online_probe" :
            (full_fa ? "full_fa_from_logits_probe" : "kq_only_zero_output"))))))))))))))))))));
    if (fused_variant_effective && !full_fa) {
        GGML_ABORT("q8k_dot4_kq fused variants require GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1");
    }

    if (full_fa) {
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
            GGML_ABORT("q8k_dot4_kq full FA probe does not support sinks, max_bias, or logit_softcap");
        }
        if (fused_variant_effective && q8block_variant) {
            GGML_ABORT("q8k_dot4_kq fused variants require packed16 K");
        }
        if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
                mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
            GGML_ABORT("q8k_dot4_kq full FA probe mask shape/type unsupported");
        }
        if ((grouped_gqa_online || grouped_gqa_qtile2_tile8 || grouped_gqa_qtile4_tile8) && gqa_ratio > GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA) {
            GGML_ABORT("q8k_dot4_kq grouped_gqa variants support GQA ratio <= %d, got %d",
                    GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA, gqa_ratio);
        }
        if ((grouped_gqa6_qtile4_tile8 || grouped_gqa6_qtile4_vshared) && gqa_ratio != GGML_CUDA_Q8K_DOT4_KQ_GQA6) {
            GGML_ABORT("q8k_dot4_kq grouped_gqa6 qtile4 variants require GQA ratio %d, got %d",
                    GGML_CUDA_Q8K_DOT4_KQ_GQA6, gqa_ratio);
        }
        if (blockfa_runtime_any_effective && mask && getenv("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL") == nullptr) {
            // Auto-selected blockfa_recthist_v4_single: assume causal mask is
            // correctly supplied (standard attention with causal mask).
            if (blockfa_recthist_v4_single_auto) {
                // OK — auto path implies causal mask is intentional.
            } else {
                GGML_ABORT("q8k_dot4_kq blockFA variants require no mask or GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1");
            }
        }
    }

    const bool timing_requested = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING");
    bool stream_is_capturing = false;
#ifdef USE_CUDA_GRAPH
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &capture_status));
    stream_is_capturing = capture_status != hipStreamCaptureStatusNone;
#endif // USE_CUDA_GRAPH
    const int timing_every = max(1, ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING_EVERY", 1));
    static unsigned long long timing_counter = 0;
    const unsigned long long timing_call = timing_requested ? ++timing_counter : 0;
    const bool timing = timing_requested && !stream_is_capturing && (timing_call % (unsigned long long) timing_every) == 0;

    q_payload.alloc(q_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
    q_scales.alloc(q_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    // Persistent packed16 K cache: hipMalloc once per K tensor, reuse across passes.
    // Skips re-pack when k_rows hasn't grown since last repack.
    static std::mutex s_cache_mutex;
    static std::unordered_map<const void *, std::pair<int *, half *>> s_cache;
    static std::unordered_map<const void *, size_t> s_cache_rows;  // k_rows at last repack
    const bool use_packed16 = ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
    bool skip_k_repack = false;
    if (use_packed16) {
        // Prefer GGML tensors from registry (allocated by KV cache). Fall back to hipMalloc.
        ggml_tensor * payload_tensor = nullptr;
        ggml_tensor * scales_tensor  = nullptr;
        llama_kv_cache_get_packed16_tensors(K->data, &payload_tensor, &scales_tensor);
        // Packed16 I32 K MUST have registry entries (pre-quantized in cache).
        if (K->type == GGML_TYPE_I32) {
            GGML_ASSERT(payload_tensor && scales_tensor && "I32 K requires packed16 registry (populated in kv_cache init)");
        }
        if (payload_tensor && scales_tensor) {
            // Use GGML tensors directly — no hipMalloc needed.
            k_payload.ptr = (int *) payload_tensor->data;
            k_scales.ptr  = (half *) scales_tensor->data;
            // Check if cache unchanged since last repack.
            std::lock_guard<std::mutex> lock(s_cache_mutex);
            size_t & prev_rows = s_cache_rows[K->data];
            if (prev_rows >= (size_t)k_rows) {
                skip_k_repack = true;
            }
        } else {
            // Fallback: hipMalloc persistent buffers.
            const size_t need_payload = (size_t)k_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
            const size_t need_scales  = (size_t)k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
            std::lock_guard<std::mutex> lock(s_cache_mutex);
            auto & e = s_cache[K->data];
            size_t & prev_rows = s_cache_rows[K->data];
            if (e.first == nullptr || prev_rows < (size_t)k_rows) {
                if (e.first && prev_rows < (size_t)k_rows) {
                    CUDA_CHECK(hipFree(e.first));
                    CUDA_CHECK(hipFree(e.second));
                }
                CUDA_CHECK(hipMalloc(&e.first,  need_payload * sizeof(int)));
                CUDA_CHECK(hipMalloc(&e.second, need_scales  * sizeof(half)));
            } else {
                skip_k_repack = true;
            }
            k_payload.ptr = e.first;
            k_scales.ptr  = e.second;
        }
    } else {
        k_payload.alloc(k_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
        k_scales.alloc(k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
    }
    if (!fused_variant_effective) {
        logits.alloc(logits_ne);
    }
    const char * blockfa_split_k_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_SPLIT_K");
    const int blockfa_split_k_requested = blockfa_split_k_env ? atoi(blockfa_split_k_env) : (blockfa_recthist_any_effective ? 1 : 4);
    const int blockfa_split_k = blockfa_runtime_any_effective ? max(1, blockfa_split_k_requested) : 1;
    const int blockfa_bn_default = blockfa_recthist_v2_bn16 ? 16 : 8;
    const int blockfa_bn = blockfa_runtime_any_effective ? ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BN", blockfa_bn_default) : 8;
    const int blockfa_v_reuse = (blockfa_hybrid_bm8_vreuse || blockfa_hybrid_bm8_hgroup) ? 1 : 0;
    const int blockfa_q_offset = (full_fa && blockfa_runtime_any_effective && mask) ? max(0, nk - nq) : 0;
    const int blockfa_recthist_prefix_k = blockfa_recthist_any_effective ? blockfa_q_offset : 0;
    const int blockfa_recthist_tail_k = blockfa_recthist_any_effective ? nk - blockfa_q_offset : 0;
    if (blockfa_recthist_any_effective) {
        if (blockfa_split_k_requested != 1) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist variants support splitK=1 only, got %d", blockfa_split_k_requested);
        }
        if ((blockfa_recthist_v2 || blockfa_recthist_v3_vhoist) && blockfa_bn != 8) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v2/v3 requires BN8, got %d", blockfa_bn);
        }
        if (blockfa_recthist_v4_single_effective) {
            if (blockfa_bn != 8 && blockfa_bn != 16) {
                GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single_effective requires BN8 or BN16, got %d", blockfa_bn);
            }
            const int v4_bm = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BM", 8);
            if (v4_bm != 8 && v4_bm != 16) {
                GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single_effective requires BM8 or BM16, got %d", v4_bm);
            }
            if (v4_bm == 16 && blockfa_bn == 16) {
                GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single_effective BM16+BN16 not supported (no idle threads for V prefetch)");
            }
        }
        if (blockfa_recthist_v2_bn16 && blockfa_bn != 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v2_bn16 requires BN16, got %d", blockfa_bn);
        }
    }
    if (blockfa_hybrid_bm8_packed16_scalar) {
        if (blockfa_bn != 8) {
            GGML_ABORT("q8k_dot4_kq blockfa_hybrid_bm8_packed16_scalar requires BN8");
        }
        if (blockfa_v_reuse != 0 || blockfa_hybrid_bm8_kreuse) {
            GGML_ABORT("q8k_dot4_kq blockfa_hybrid_bm8_packed16_scalar excludes V/K reuse variants");
        }
    }
    if ((blockfa_recthist_any_effective && !blockfa_recthist_v4_single_effective) || (blockfa_hybrid_bm8_any && blockfa_split_k > 1)) {
        const size_t partial_rows = q_rows * size_t(blockfa_recthist_any_effective ? 2 : blockfa_split_k);
        blockfa_partial_o.alloc(partial_rows * GGML_CUDA_Q8K_DOT4_KQ_D);
        blockfa_partial_m.alloc(partial_rows);
        blockfa_partial_l.alloc(partial_rows);
    }

    cudaStream_t stream = ctx.stream();
    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_q_quant = nullptr;
    hipEvent_t ev_k_pack = nullptr;
    hipEvent_t ev_kq = nullptr;
    hipEvent_t ev_ref = nullptr;
    hipEvent_t ev_err = nullptr;
    hipEvent_t ev_blockfa = nullptr;
    hipEvent_t ev_recthist_ready = nullptr;
    hipEvent_t ev_recthist_prefix = nullptr;
    hipEvent_t ev_recthist_tail = nullptr;
    hipEvent_t ev_recthist_merge = nullptr;
    hipEvent_t ev_combine = nullptr;
    hipEvent_t ev_zero = nullptr;
    if (timing) {
        ggml_cuda_q8k_dot4_kq_event_create(&ev_start);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_kq);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_ref);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_err);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_recthist_ready);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_recthist_prefix);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_recthist_tail);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_recthist_merge);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_combine);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_zero);
        CUDA_CHECK(hipEventRecord(ev_start, stream));
    }

    dim3 q_grid(nq, n_heads_q, batch);
    dim3 block(256);
    ggml_cuda_q8k_dot4_quant_q_packed16_kernel<<<q_grid, block, 0, stream>>>(
        (const float *) Q->data, q_payload.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, batch);
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_q_quant, stream));
    }

    if (!q8block_variant || fused_variant_effective) {
        if (!skip_k_repack) {
            // Packed16 I32 K is pre-quantized; skip repack and use registry data.
            const bool k_is_already_packed16 = (K->type == GGML_TYPE_I32);
            if (k_is_already_packed16) {
                skip_k_repack = true; // already in k_payload.ptr from registry above
            } else if (K->type == GGML_TYPE_F16) {
                // Direct f16→packed16 quantization (one kernel, no intermediate)
                dim3 k_quant_grid(nk, n_heads_k, batch);
                ggml_cuda_q8k_dot4_quant_k_packed16_kernel<<<k_quant_grid, block, 0, stream>>>(
                    (const half *) K->data, k_payload.ptr, k_scales.ptr,
                    K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // q8_0→packed16 repack (existing path)
                dim3 pack_grid((k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + 255) / 256);
                ggml_cuda_q8k_dot4_pack_k_packed16_kernel<<<pack_grid, block, 0, stream>>>(
                    (const char *) K->data, k_payload.ptr, k_scales.ptr,
                    K->nb[0], K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch);
                CUDA_CHECK(cudaGetLastError());
            }
            if (use_packed16) {
                std::lock_guard<std::mutex> lock(s_cache_mutex);
                s_cache_rows[K->data] = (size_t)k_rows;
            }
        }
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_k_pack, stream));
        }
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    dim3 kq_grid((nk + GGML_CUDA_Q8K_DOT4_KQ_TILE_N - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_N,
                 (nq + GGML_CUDA_Q8K_DOT4_KQ_TILE_M - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_M,
                 n_heads_q * batch);
    if (!fused_variant_effective) {
        if (q8block_variant) {
            ggml_cuda_q8k_dot4_kq_q8block_kernel<<<kq_grid, block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, (const char *) K->data, logits.ptr, scale,
                K->nb[0], K->nb[1], K->nb[2], K->nb[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else {
            ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, logits.ptr, scale,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_kq, stream));
    }

    if (check && !fused_variant_effective) {
        ref_logits.alloc(logits_ne);
        metrics.alloc(3);
        CUDA_CHECK(cudaMemsetAsync(metrics.ptr, 0, 3 * sizeof(float), stream));
        ggml_cuda_q8k_dot4_kq_reference_kernel<<<kq_grid, block, 0, stream>>>(
            (const float *) Q->data, (const char *) K->data, ref_logits.ptr, scale,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[0], K->nb[1], K->nb[2], K->nb[3],
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_ref, stream));
        }
        ggml_cuda_q8k_dot4_kq_error_kernel<<<(logits_ne + 255) / 256, 256, 0, stream>>>(
            logits.ptr, ref_logits.ptr, metrics.ptr, logits_ne);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_err, stream));
        }
        float h_metrics[3] = {0.0f, 0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(h_metrics, metrics.ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double rms = sqrt(double(h_metrics[0]) / double(logits_ne));
        const double mean_abs = double(h_metrics[1]) / double(logits_ne);
        GGML_LOG_INFO("%s: q8k_dot4_kq_check logits=%zu rms=%.6g mean_abs=%.6g gt0.5=%.0f\n",
            __func__, logits_ne, rms, mean_abs, double(h_metrics[2]));
    }

    if (full_fa) {
        dim3 fa_grid(nq, n_heads_q, batch);
        if (blockfa_runtime_any_effective) {
            const int q_offset = blockfa_q_offset;
            const bool use_causal = mask && (blockfa_recthist_v4_single_auto ||
                ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL"));
            const dim3 blockfa_grid((nq + 7) / 8, blockfa_split_k, n_heads_q * batch);
            const int blockfa_head_group = blockfa_hybrid_bm8_hquad ? 4 :
                ((blockfa_hybrid_bm8_htriad || blockfa_hybrid_bm8_htriad_kreuse) ? 3 :
                ((blockfa_hybrid_bm8_hpair || blockfa_hybrid_bm8_hpair_kreuse || blockfa_hybrid_bm8_hpair_warpkq || blockfa_hybrid_bm8_hpair_rowwarpkq) ? 2 : 1));
            const size_t smem_bn8 = size_t(8 * 8 + 3 * 8 + (blockfa_v_reuse ? 8 * GGML_CUDA_Q8K_DOT4_KQ_D : 0)) * sizeof(float);
            const size_t smem_bn16 = size_t(8 * 16 + 3 * 8 + (blockfa_v_reuse ? 16 * GGML_CUDA_Q8K_DOT4_KQ_D : 0)) * sizeof(float);
            const size_t smem_hgroup_bn8 = size_t(blockfa_head_group * 8 * 8 + 3 * blockfa_head_group * 8 + 8 * GGML_CUDA_Q8K_DOT4_KQ_D) * sizeof(float);
            const size_t smem_hpair_tail_bn8 = size_t(2 * 8 * 8 + 3 * 2 * 8 + 8 * GGML_CUDA_Q8K_DOT4_KQ_D) * sizeof(float);
            if (blockfa_hybrid_bm8_hgroup && (gqa_ratio != 6 || blockfa_bn != 8)) {
                GGML_ABORT("q8k_dot4_kq blockfa_hybrid_bm8 hgroup variants require GQA ratio 6 and BN8");
            }
            if (blockfa_recthist_v4_single_effective) {
                if (!use_causal) {
                    GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires causal-tail mask and GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1");
                }
                // Debug instrumentation for I32 packed16 contracts.
                static bool debug_i32_contract_printed = false;
                if (k_is_i32_packed16 && !debug_i32_contract_printed &&
                        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
                    debug_i32_contract_printed = true;
                    fprintf(stderr,
                        "q8k_dot4_i32: Q type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: K type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: V type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: payload=%p scales=%p nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d\n",
                        Q->type, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                        Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3], Q->data,
                        K->type, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3], K->data,
                        V->type, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        V->nb[0], V->nb[1], V->nb[2], V->nb[3], V->data,
                        k_payload.ptr, k_scales.ptr, nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio);
                }
                // Ensure all prior GPU work (pack_k on any stream) finished before FA reads registry data.
                const bool force_sync = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC");
                if (force_sync) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                const int q_offset = blockfa_q_offset;
                const int v4_bn = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BN", 8);
                const int v4_bm = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BM", 8);
                const size_t smem_v4 = (size_t(v4_bm * v4_bn + 3 * v4_bm + 2 * v4_bn * GGML_CUDA_Q8K_DOT4_KQ_D) + size_t(v4_bm * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)) + size_t(v4_bm * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS)) * sizeof(float);
                const dim3 recthist_grid((nq + v4_bm - 1) / v4_bm, n_heads_q, batch);
                const bool use_f16_v = (V->type == GGML_TYPE_F16);
                const bool use_q8_v  = (V->type == GGML_TYPE_Q8_0);
                // ---- Packed16 decode fast-path (nq=1) ----
                const int decode_bn = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN", 0);
                const int32_t fa_inst = ((const int32_t *)dst->op_params)[4];
                const bool is_mtp_draft_decode = (fa_inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK);

                if (decode_bn > 0 && nq == 1 && (K->type == GGML_TYPE_I32 || is_mtp_draft_decode)) {
                    // For f16-source K, packed16 is already materialized in k_payload/k_scales.
                    // Strides are computed from packed16 layout, not from K tensor strides.
                    const int packed16_per_row = 64; // D=256 / 4
                    const int k_head_stride_rows  = nk * packed16_per_row;
                    const int k_batch_stride_rows = n_heads_k * k_head_stride_rows;
                    const int decode_vsub = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_VSUB", 8);
                    const bool inline_q4 = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_INLINE_Q4");
                    const bool q4pair = V->type == GGML_TYPE_Q4_0 && ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR");
                    const bool splitk = (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK") || nk >= 2048) && V->type == GGML_TYPE_Q4_0;
                    if (splitk) {
                        const int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512);
                        const int n_splits = (nk + split_size - 1) / split_size;
                        blockfa_partial_o.alloc((size_t) batch * n_heads_q * n_splits * 256);
                        blockfa_partial_m.alloc((size_t) batch * n_heads_q * n_splits);
                        blockfa_partial_l.alloc((size_t) batch * n_heads_q * n_splits);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SPLITK(64, 8) }
                        else { GGML_ABORT("q8k_dot4_kq split-K decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (decode_bn == 64 && decode_vsub == 8) {
                        if (q4pair)      LAUNCH_DECODE_Q4PAIR(64, 8)
                        else if (inline_q4) LAUNCH_DECODE_INLINE_Q4(64, 8)
                        else             LAUNCH_DECODE_VSUB(64, 8)
                    } else {
                        GGML_ABORT("q8k_dot4_kq decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub);
                    }
                    return;
                }
                // Fixed KV cache head strides for persistent packed16 K.
                GGML_ASSERT(K->nb[1] % sizeof(int) == 0);
                const int k_head_stride_rows  = (int)(K->nb[2] / K->nb[1]);
                const int k_batch_stride_rows = (int)(K->nb[3] / K->nb[1]);
                if (v4_bm == 16 && v4_bn == 8) {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    }
                } else if (v4_bm == 8 && v4_bn == 16) {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    }
                } else {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_recthist_prefix, stream));
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
            } else if (blockfa_recthist_v2_any || blockfa_recthist_v3_vhoist) {
                if (!use_causal) {
                    GGML_ABORT("q8k_dot4_kq blockfa_recthist variants require causal-tail mask and GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1");
                }
                const int recthist_slots = 2;
                const int prefix_k = blockfa_recthist_prefix_k;
                const int tail_k = blockfa_recthist_tail_k;
                const size_t partial_rows_local = q_rows * size_t(recthist_slots);
                const bool vhoist = blockfa_recthist_v3_vhoist;
                const size_t smem_bn8_vhoist = (size_t(8 * 8 + 3 * 8 + 8 * GGML_CUDA_Q8K_DOT4_KQ_D) + size_t(8 * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)) + size_t(8 * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS)) * sizeof(float);
                ggml_cuda_q8k_dot4_blockfa_reset_partial_kernel<<<(partial_rows_local + 255) / 256, 256, 0, stream>>>(
                    blockfa_partial_m.ptr, blockfa_partial_l.ptr, partial_rows_local);
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_recthist_ready, stream));
                }
                const dim3 recthist_grid((nq + 7) / 8, n_heads_q, batch);
                if (prefix_k > 0) {
                    if (vhoist) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_vhoist_kernel<8, false><<<recthist_grid, block, smem_bn8_vhoist, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, 0, prefix_k, recthist_slots, 0);
                    } else if (blockfa_bn == 16) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_kernel<16, false><<<recthist_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, 0, prefix_k, recthist_slots, 0);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_kernel<8, false><<<recthist_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, 0, prefix_k, recthist_slots, 0);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_recthist_prefix, stream));
                }
                if (tail_k > 0) {
                    if (vhoist) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_vhoist_kernel<8, true><<<recthist_grid, block, smem_bn8_vhoist, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, prefix_k, nk, recthist_slots, 1);
                    } else if (blockfa_bn == 16) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_kernel<16, true><<<recthist_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, prefix_k, nk, recthist_slots, 1);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_kernel<8, true><<<recthist_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, prefix_k, nk, recthist_slots, 1);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_recthist_tail, stream));
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
                ggml_cuda_q8k_dot4_blockfa_combine_kernel<<<fa_grid, block, 0, stream>>>(
                    blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, nq, n_heads_q, recthist_slots);
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_recthist_merge, stream));
                    CUDA_CHECK(hipEventRecord(ev_combine, stream));
                }
            } else if (blockfa_hybrid_bm8_hgroup) {
#define GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(GH, KREUSE, WARPKQ, ROWWARPKQ, GRID, SMEM, HEAD_GROUPS_PER_BATCH, HEAD_GROUP_STRIDE, HEAD_GROUP_OFFSET) \
                do { \
                    if (blockfa_split_k == 1) { \
                        if (use_causal) { \
                            ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_hpair_q4_0_kernel<8, true, true, true, GH, KREUSE, WARPKQ, ROWWARPKQ><<<(GRID), block, (SMEM), stream>>>( \
                                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data, \
                                nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
                                mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, \
                                (HEAD_GROUPS_PER_BATCH), (HEAD_GROUP_STRIDE), (HEAD_GROUP_OFFSET)); \
                        } else { \
                            ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_hpair_q4_0_kernel<8, true, false, false, GH, KREUSE, WARPKQ, ROWWARPKQ><<<(GRID), block, (SMEM), stream>>>( \
                                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data, \
                                nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
                                0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, \
                                (HEAD_GROUPS_PER_BATCH), (HEAD_GROUP_STRIDE), (HEAD_GROUP_OFFSET)); \
                        } \
                    } else { \
                        if (use_causal) { \
                            ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_hpair_q4_0_kernel<8, false, true, true, GH, KREUSE, WARPKQ, ROWWARPKQ><<<(GRID), block, (SMEM), stream>>>( \
                                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data, \
                                blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
                                mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, \
                                (HEAD_GROUPS_PER_BATCH), (HEAD_GROUP_STRIDE), (HEAD_GROUP_OFFSET)); \
                        } else { \
                            ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_hpair_q4_0_kernel<8, false, false, false, GH, KREUSE, WARPKQ, ROWWARPKQ><<<(GRID), block, (SMEM), stream>>>( \
                                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data, \
                                blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
                                0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, \
                                (HEAD_GROUPS_PER_BATCH), (HEAD_GROUP_STRIDE), (HEAD_GROUP_OFFSET)); \
                        } \
                    } \
                } while (0)

                if (blockfa_hybrid_bm8_hpair) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 2) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(2, false, false, false, blockfa_hgroup_grid, smem_hgroup_bn8, n_heads_q / 2, 2, 0);
                } else if (blockfa_hybrid_bm8_hpair_kreuse) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 2) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(2, true, false, false, blockfa_hgroup_grid, smem_hgroup_bn8 + size_t(8 * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) * sizeof(int) + 8 * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS * sizeof(half)), n_heads_q / 2, 2, 0);
                } else if (blockfa_hybrid_bm8_hpair_warpkq) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 2) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(2, false, true, false, blockfa_hgroup_grid, smem_hgroup_bn8, n_heads_q / 2, 2, 0);
                } else if (blockfa_hybrid_bm8_hpair_rowwarpkq) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 2) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(2, false, false, true, blockfa_hgroup_grid, smem_hgroup_bn8, n_heads_q / 2, 2, 0);
                } else if (blockfa_hybrid_bm8_htriad) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 3) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(3, false, false, false, blockfa_hgroup_grid, smem_hgroup_bn8, n_heads_q / 3, 3, 0);
                } else if (blockfa_hybrid_bm8_htriad_kreuse) {
                    const dim3 blockfa_hgroup_grid((nq + 7) / 8, blockfa_split_k, (n_heads_q / 3) * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(3, true, false, false, blockfa_hgroup_grid, smem_hgroup_bn8 + size_t(8 * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) * sizeof(int) + 8 * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS * sizeof(half)), n_heads_q / 3, 3, 0);
                } else {
                    const dim3 blockfa_hquad_grid((nq + 7) / 8, blockfa_split_k, n_heads_k * batch);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(4, false, false, false, blockfa_hquad_grid, smem_hgroup_bn8, n_heads_k, gqa_ratio, 0);
                    GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP(2, false, false, false, blockfa_hquad_grid, smem_hpair_tail_bn8, n_heads_k, gqa_ratio, 4);
                }
#undef GGML_CUDA_Q8K_DOT4_BLOCKFA_LAUNCH_HGROUP
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
                if (blockfa_split_k > 1) {
                    ggml_cuda_q8k_dot4_blockfa_combine_kernel<<<fa_grid, block, 0, stream>>>(
                        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, nq, n_heads_q, blockfa_split_k);
                    if (timing) {
                        CUDA_CHECK(hipEventRecord(ev_combine, stream));
                    }
                }
            } else if (blockfa_split_k == 1) {
                if (blockfa_bn == 16) {
                    if (use_causal) {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<16, true, true, true><<<blockfa_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data,
                            nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<16, true, false, false><<<blockfa_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data,
                            nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    }
                } else {
                    if (use_causal) {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<8, true, true, true><<<blockfa_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data,
                            nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<8, true, false, false><<<blockfa_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data,
                            nullptr, nullptr, nullptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
            } else {
                if (blockfa_bn == 16) {
                    if (use_causal) {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<16, false, true, true><<<blockfa_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<16, false, false, false><<<blockfa_grid, block, smem_bn16, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    }
                } else {
                    if (use_causal) {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<8, false, true, true><<<blockfa_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, (const char *) mask->data, (float *) dst->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_hybrid_bm8_q4_0_kernel<8, false, false, false><<<blockfa_grid, block, smem_bn8, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, nullptr, (float *) dst->data,
                            blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            0, 0, 0, 1, nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, blockfa_split_k, q_offset, blockfa_v_reuse);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
                ggml_cuda_q8k_dot4_blockfa_combine_kernel<<<fa_grid, block, 0, stream>>>(
                    blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, nq, n_heads_q, blockfa_split_k);
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_combine, stream));
                }
            }
        } else if (grouped_gqa6_qtile4_vshared) {
            dim3 grouped_grid((nq + 3) / 4, n_heads_k, batch);
            dim3 qtile_block(512);
            ggml_cuda_q8k_dot4_grouped_gqa6_qtile4_vshared_fattn_q4_0_kernel<<<grouped_grid, qtile_block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, batch);
        } else if (grouped_gqa6_qtile4_tile8) {
            dim3 grouped_grid((nq + 3) / 4, n_heads_k, batch);
            dim3 qtile_block(512);
            ggml_cuda_q8k_dot4_grouped_gqa6_qtile4_tile8_fattn_q4_0_kernel<<<grouped_grid, qtile_block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, batch);
        } else if (grouped_gqa_qtile4_tile8) {
            dim3 grouped_grid((nq + 3) / 4, n_heads_k, batch);
            dim3 qtile_block(512);
            ggml_cuda_q8k_dot4_grouped_gqa_qtile4_tile8_fattn_q4_0_kernel<<<grouped_grid, qtile_block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else if (grouped_gqa_qtile2_tile8) {
            dim3 grouped_grid((nq + 1) / 2, n_heads_k, batch);
            dim3 qtile_block(512);
            ggml_cuda_q8k_dot4_grouped_gqa_qtile2_tile8_fattn_q4_0_kernel<<<grouped_grid, qtile_block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else if (grouped_gqa_online) {
            dim3 grouped_grid(nq, n_heads_k, batch);
            ggml_cuda_q8k_dot4_grouped_gqa_online_fattn_q4_0_kernel<<<grouped_grid, block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else if (fused_tile8_parallel) {
            ggml_cuda_q8k_dot4_fused_tile8_parallel_fattn_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else if (fused_online) {
            ggml_cuda_q8k_dot4_fused_online_fattn_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr,
                (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        } else {
            ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        // KQ-only probe: deliberately leave FA output unchanged/zeroed so this route
        // cannot masquerade as production attention. It exists to validate runtime
        // packed-DOT4 KQ layout, route contracts, and ISA only.
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_zero, stream));
        CUDA_CHECK(hipEventSynchronize(ev_zero));
        const float q_quant_ms = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_q_quant);
        const float k_pack_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_q_quant, ev_k_pack);
        const float kq_ms      = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_k_pack, ev_kq);
        const float ref_ms     = check && !fused_variant_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_kq, ev_ref) : 0.0f;
        const float err_ms     = check && !fused_variant_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_ref, ev_err) : 0.0f;
        const float output_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(check && !fused_variant_effective ? ev_err : ev_kq, ev_zero);
        const float zero_ms    = full_fa ? 0.0f : output_ms;
        const float fa_ms      = full_fa ? output_ms : 0.0f;
        const float blockfa_ms = full_fa && blockfa_runtime_any_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_kq, ev_blockfa) : 0.0f;
        const float combine_ms = full_fa && ((blockfa_hybrid_bm8_any && blockfa_split_k > 1) || blockfa_recthist_v2_any) ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_blockfa, ev_combine) : 0.0f;
        const float prefix_ms  = full_fa && blockfa_recthist_v2_any && blockfa_recthist_prefix_k > 0 ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_recthist_ready, ev_recthist_prefix) : 0.0f;
        const float tail_ms    = full_fa && blockfa_recthist_v2_any && blockfa_recthist_tail_k > 0 ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_recthist_prefix, ev_recthist_tail) : 0.0f;
        const float merge_ms   = full_fa && blockfa_recthist_v2_any ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_recthist_tail, ev_recthist_merge) : 0.0f;
        const float total_ms   = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_zero);
        GGML_LOG_INFO("%s: q8k_dot4_kq_timing call=%llu variant=%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d check=%d q_quant_ms=%.6f k_pack_ms=%.6f kq_ms=%.6f ref_ms=%.6f err_ms=%.6f zero_ms=%.6f total_ms=%.6f fa_ms=%.6f blockfa_ms=%.6f combine_ms=%.6f prefix_ms=%.6f tail_ms=%.6f merge_ms=%.6f q_offset=%d prefix_k=%d tail_k=%d full_fa=%d blockfa_split_k=%d blockfa_bn=%d\n",
            __func__, timing_call, variant_name, nq, nk, n_heads_q, n_heads_k, batch, check && !fused_variant_effective ? 1 : 0,
            q_quant_ms, k_pack_ms, kq_ms, ref_ms, err_ms, zero_ms, total_ms, fa_ms, blockfa_ms, combine_ms, prefix_ms, tail_ms, merge_ms,
            blockfa_recthist_v2_any ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k,
            full_fa ? 1 : 0, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_start);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_kq);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_ref);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_err);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_recthist_ready);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_recthist_prefix);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_recthist_tail);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_recthist_merge);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_combine);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_zero);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double(q_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double k_mib = double(k_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double logits_mib = fused_variant_effective ? 0.0 : double(logits_ne * sizeof(float)) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=rocm_q8k_dot4_kq variant=%s full_fa=%d nq=%d nk=%d heads_q=%d heads_k=%d batch=%d q_payload=%.3fMiB k_payload=%.3fMiB logits=%.3fMiB note=%s blockfa_split_k=%d blockfa_bn=%d q_offset=%d prefix_k=%d tail_k=%d\n",
            __func__, variant_name, full_fa ? 1 : 0, nq, nk, n_heads_q, n_heads_k, batch, q_mib, k_mib, logits_mib, variant_note, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0,
            blockfa_recthist_v2_any ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k);
    }

    // Detach hipMalloc'd persistent buffers from pool allocator destructors.
    if (ggml_cuda_q8k_dot4_packed16_k_cache_enabled()) {
        k_payload.ptr = nullptr;
        k_scales.ptr  = nullptr;
    }

    GGML_UNUSED(sinks);
}

// ggml OP_PACK_K_PACKED16 backend: quantize f16 Kcur → packed16 (I32 payload + F16 scales)
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k_cur   = dst->src[0];  // f16 source
    ggml_tensor * scales  = dst->src[1];  // F16 scales output
    ggml_tensor * k_idxs  = dst->src[2];  // row indices (absolute cache slots)
    ggml_tensor * payload = dst;          // I32 payload (dst is view of payload)

    GGML_ASSERT(k_idxs != nullptr);
    GGML_ASSERT(payload->type == GGML_TYPE_I32);
    GGML_ASSERT(scales->type == GGML_TYPE_F16);
    GGML_ASSERT(payload->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    GGML_ASSERT(scales->ne[0]  == GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    const int D = GGML_CUDA_Q8K_DOT4_KQ_D;
    const int batch    = (int) k_cur->ne[3];

    // Infer KV head count from source K shape.  k_cur may be split-head [D, n_heads, nk, B]
    // or combined-GQA [D*nh, nk, 1, B].  The DOT4 kernel needs per-head packed16 rows.
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

    bool is_q8_k = k_cur->type == GGML_TYPE_Q8_0;
    // For q8_0 source: override nk_cur to active token count from k_idxs.
    if (is_q8_k) {
        nk_cur = (int) k_idxs->ne[0];  // active tokens in this batch
        GGML_ASSERT(nk_cur <= kv_size);
    }

    // Token stride for source K: nb[2] for split-head, nb[1] for combined-GQA.
    const int64_t src_token_stride_bytes = (k_cur->ne[0] == D)
        ? k_cur->nb[2]   // nb[2] advances across tokens in [D, nh, nk, B]
        : k_cur->nb[1];  // nb[1] advances across tokens in [D*nh, nk, 1, B]

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

    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(256);
    cudaStream_t stream = ctx.stream();

    if (is_q8_k) {
        // Indexed pack from q8_0 cache blocks — preserves calibrated quantization.
        ggml_cuda_q8k_dot4_pack_k_packed16_from_q8_indexed_kernel<<<grid, block, 0, stream>>>(
            (const char *) k_cur->data, (int *) payload->data, (half *) scales->data,
            (const int64_t *) k_idxs->data,
            k_cur->nb[1], kv_size, n_heads, batch, nk_cur);
    } else if (k_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int64_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size);
    } else {
        GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
        ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int32_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_USE_HIP
