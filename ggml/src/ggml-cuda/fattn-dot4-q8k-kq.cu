#include "fattn-dot4-q8k-kq.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Packed16 K cache tensor registry (shared with llama-kv-cache).
struct packed16_registry_entry {
    ggml_tensor * payload  = nullptr;
    ggml_tensor * scales   = nullptr;
};

struct v4_k16d16_registry_entry {
    ggml_tensor * v_cache = nullptr;
    ggml_tensor * v_tail  = nullptr;
};

static std::mutex s_packed16_mutex;
static std::unordered_map<const void *, packed16_registry_entry> s_packed16_registry;
static std::mutex s_v4_k16d16_mutex;
static std::unordered_map<const void *, v4_k16d16_registry_entry> s_v4_k16d16_registry;

extern "C" {
void llama_kv_cache_register_packed16(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    packed16_registry_entry & entry = s_packed16_registry[k_view_data];
    entry.payload = payload;
    entry.scales  = scales;
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

void llama_kv_cache_register_v4_k16d16(const void * v_view_data, ggml_tensor * v_cache, ggml_tensor * v_tail) {
    std::lock_guard<std::mutex> lock(s_v4_k16d16_mutex);
    s_v4_k16d16_registry[v_view_data] = {v_cache, v_tail};
}

void llama_kv_cache_get_v4_k16d16_tensors(const void * v_view_data, ggml_tensor ** v_cache, ggml_tensor ** v_tail) {
    std::lock_guard<std::mutex> lock(s_v4_k16d16_mutex);
    auto it = s_v4_k16d16_registry.find(v_view_data);
    if (it != s_v4_k16d16_registry.end()) {
        *v_cache = it->second.v_cache;
        *v_tail  = it->second.v_tail;
    } else {
        *v_cache = nullptr;
        *v_tail  = nullptr;
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

static inline const char * ggml_cuda_q8k_dot4_k_physical_name(const ggml_tensor * K) {
    if (!K) {
        return "unknown";
    }
    if (K->type == GGML_TYPE_Q8_0) {
        return "q8_0_block32";
    }
    if (K->type == GGML_TYPE_I32) {
        return "packed16_q8_sidechannel_i32_f16scales";
    }
    if (K->type == GGML_TYPE_F16) {
        return "f16_source_op_local_packed16";
    }
    return "unknown";
}

static bool ggml_cuda_q8k_dot4_packed16_sidecar_valid(
        const ggml_tensor * K,
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const bool verbose,
        const char * context) {
    const char * reject = nullptr;
    int64_t head_stride_payload = 0;
    int64_t head_stride_scales = 0;

    if (!K || K->type != GGML_TYPE_I32) {
        reject = "not_i32_packed16_k";
    } else if (!payload || !scales) {
        reject = "missing_packed16_k_sidecar";
    } else if (payload->type != GGML_TYPE_I32 || scales->type != GGML_TYPE_F16) {
        reject = "bad_packed16_k_sidecar_type";
    } else if (K->ne[2] <= 0 || payload->ne[1] % K->ne[2] != 0 || scales->ne[1] % K->ne[2] != 0) {
        reject = "bad_packed16_k_head_stride";
    } else {
        head_stride_payload = payload->ne[1] / K->ne[2];
        head_stride_scales = scales->ne[1] / K->ne[2];
        if (payload->ne[0] != GGML_CUDA_Q8K_DOT4_KQ_D / 4 ||
                scales->ne[0] != GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0 ||
                head_stride_payload < K->ne[1] || head_stride_scales < K->ne[1] ||
                payload->ne[1] < K->ne[1] * K->ne[2] ||
                scales->ne[1] < K->ne[1] * K->ne[2] ||
                payload->nb[0] != (int64_t) sizeof(int) ||
                scales->nb[0] != (int64_t) sizeof(half) ||
                payload->nb[1] != (GGML_CUDA_Q8K_DOT4_KQ_D / 4) * (int64_t) sizeof(int) ||
                scales->nb[1] != (GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0) * (int64_t) sizeof(half) ||
                payload->nb[2] != payload->ne[1] * payload->nb[1] ||
                scales->nb[2] != scales->ne[1] * scales->nb[1] ||
                payload->nb[3] != payload->ne[2] * payload->nb[2] ||
                scales->nb[3] != scales->ne[2] * scales->nb[2]) {
            reject = "bad_packed16_k_sidecar_shape";
        }
    }

    if (!reject) {
        return true;
    }

    if (verbose) {
        fprintf(stderr,
            "q8k_dot4_kq reject route=rocm_fa2_packed16_dot4_decode "
            "reject=%s context=%s K_data=%p K=[%lld,%lld,%lld,%lld] "
            "payload=%p scales=%p head_stride_payload=%lld head_stride_scales=%lld\n",
            reject, context ? context : "unknown", K ? K->data : nullptr,
            K ? (long long) K->ne[0] : 0, K ? (long long) K->ne[1] : 0,
            K ? (long long) K->ne[2] : 0, K ? (long long) K->ne[3] : 0,
            (const void *) payload, (const void *) scales,
            (long long) head_stride_payload, (long long) head_stride_scales);
    }
    return false;
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

static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_D32 = 32;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_PAYLOAD_BYTES = 2048;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_WORDS_PER_D = 2;

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_v4_k16d16_144(
        const char * __restrict__ V_head,
        const int64_t nb21,
        const int k,
        const int d) {
    const int k16_base = k & ~(GGML_CUDA_V4_K16D16_144_DECODE_K - 1);
    const int slot = k & (GGML_CUDA_V4_K16D16_144_DECODE_K - 1);
    const char * block = V_head + int64_t(k16_base) * nb21;
    const uint32_t word = ((const uint32_t *) block)[d * GGML_CUDA_V4_K16D16_144_DECODE_WORDS_PER_D + (slot >> 3)];
    const int q = int((word >> (4 * (slot & 7))) & 0x0fu) - 8;
    const int d32 = d / GGML_CUDA_V4_K16D16_144_DECODE_D32;
    const half * scales = (const half *) (block + GGML_CUDA_V4_K16D16_144_DECODE_PAYLOAD_BYTES);
    return float(q) * __half2float(scales[d32 * GGML_CUDA_V4_K16D16_144_DECODE_K + slot]);
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
        int k_batch_stride_rows,
        int debug) {
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
            if (debug && b == 0 && hq == 0 && q0 == 0 && k0 == 0 && tid < min(8, BN)) {
                printf("dot4_debug_qk: q=0 h=0 k=%d logit=%.8e scale=%.8e q_offset=%d\n", tid, logits[tid], scale, q_offset);
            }
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
            if (debug && b == 0 && hq == 0 && q0 == 0 && (k0 == 0 || !have_next) && tid == 0) {
                printf("dot4_debug_ml: q=0 h=0 k0=%d m=%.8e l=%.8e\n", k0, row_m[0], row_l[0]);
            }
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
        if (debug && b == 0 && hq == 0 && q0 == 0 && k0 == 0 && tid < min(8, BN)) {
            printf("dot4_debug_qk: q=0 h=0 k=%d logit=%.8e scale=%.8e q_offset=%d\n", tid, logits[tid], scale, q_offset);
        }
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
        if (debug && b == 0 && hq == 0 && q0 == 0 && (k0 == 0 || !have_next) && tid == 0) {
            printf("dot4_debug_ml: q=0 h=0 k0=%d m=%.8e l=%.8e\n", k0, row_m[0], row_l[0]);
        }
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
            if (q < nq) {
                const float denom = fmaxf(row_l[qr], 1.0e-20f);
                const float val = out[qr] / denom;
                if (debug && b == 0 && hq == 0 && q == 0 && tid < 8) {
                    const bool bad = !isfinite(row_m[qr]) || !isfinite(row_l[qr]) || row_l[qr] <= 0.0f || !isfinite(val) || fabsf(val) > 1.0e6f;
                    printf("dot4_debug_o: q=0 h=0 d=%d m=%.8e l=%.8e dst=%.8e bad=%d\n", tid, row_m[qr], row_l[qr], val, (int) bad);
                }
                dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = val;
            }
        }
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

#include "fattn-packed16-wmma-builtin.cuh"
#include "fattn-dot4-q8k-decode.cuh"

static inline bool ggml_cuda_pack_k_packed16_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

// Packed16 K quality knobs for acceptance/drift experiments. Defaults preserve
// existing behavior: persistent indexed K-cache packing keeps the current
// one-step MSE scale at block32 granularity.
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE=maxabs|mse|0|1
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL=<positive float>
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS=1|2|4|8
// Coarser scale groups are diagnostics for the "F16 scale mask" hypothesis:
// they requantize K with one shared scale over multiple 32-dim blocks and write
// that scale back into each existing scale slot. This preserves the tensor ABI
// and tests quality sensitivity; it does not reduce scale-memory traffic yet.
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

static inline int ggml_cuda_pack_k_packed16_scale_group_qblocks() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP_QBLOCKS");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP");
    }
    if (!env || env[0] == '\0') {
        return 1;
    }
    const int v = atoi(env);
    if (v <= 1) return 1;
    if (v <= 2) return 2;
    if (v <= 4) return 4;
    return GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
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
        float scale_mul,
        int scale_group_qblocks) {
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

    // Preserve the original block32 packer exactly unless the diagnostic asks
    // for coarser F16 scale-mask groups.
    if (scale_group_qblocks <= 1) {
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
        return;
    }

    __shared__ float block_amax[GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    __shared__ float block_num [GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    __shared__ float block_den [GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    if (lane == 0) {
        block_amax[q_block] = amax;
    }
    __syncthreads();

    const int group_qblocks = scale_group_qblocks <= 1 ? 1 : (scale_group_qblocks <= 2 ? 2 : (scale_group_qblocks <= 4 ? 4 : GGML_CUDA_Q8K_DOT4_KQ_BLOCKS));
    const int group_start = (q_block / group_qblocks) * group_qblocks;
    const int group_end = min(group_start + group_qblocks, GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    float group_amax = 0.0f;
#pragma unroll
    for (int s = 0; s < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++s) {
        if (s >= group_start && s < group_end) {
            group_amax = fmaxf(group_amax, block_amax[s]);
        }
    }
    const float scale0 = group_amax > 0.0f ? group_amax / 127.0f : 1.0f;
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
        if (lane == 0) {
            block_num[q_block] = num;
            block_den[q_block] = den;
        }
        __syncthreads();

        float group_num = 0.0f;
        float group_den = 0.0f;
#pragma unroll
        for (int s = 0; s < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++s) {
            if (s >= group_start && s < group_end) {
                group_num += block_num[s];
                group_den += block_den[s];
            }
        }
        scale = (group_den > 0.0f) ? (group_num / group_den) : scale0;
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
                "fa_dot4_launch: fa_inst=%d nq=%lld nk=%lld d=%lld K=%s k_phys=%s V=%s\n",
                fa_inst_i32,
                (long long) Q->ne[1],
                (long long) K->ne[1],
                (long long) Q->ne[0],
                ggml_type_name(K->type),
                ggml_cuda_q8k_dot4_k_physical_name(K),
                ggml_type_name(V->type));
        }
    }

    const bool v4_144_nomtp_decode =
        V->type == GGML_TYPE_V4_K16D16_144 &&
        (fa_inst_i32 == GGML_FATTN_INST_NONE || fa_inst_i32 == GGML_FATTN_INST_DECODE_QK);
    const bool v4_144_decode_diag_env =
        V->type == GGML_TYPE_V4_K16D16_144 &&
        (v4_144_nomtp_decode ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT"));

    // I32 packed16 K contract: DOT4-dispatch path only.
    const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;

    if (k_is_i32_packed16) {
        GGML_ASSERT(Q->type == GGML_TYPE_F32);
        GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || v4_144_decode_diag_env);
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
    const bool blockfa_recthist_v4_single = variant_env && strcmp(variant_env, "blockfa_recthist_v4_single") == 0;
    if (variant_env && variant_env[0] != '\0' && strcmp(variant_env, "packed16") != 0 && !blockfa_recthist_v4_single) {
        GGML_ABORT("unsupported q8k_dot4_kq variant '%s'; supported variants: packed16, blockfa_recthist_v4_single", variant_env);
    }
    const bool blockfa_runtime_any_explicit = blockfa_recthist_v4_single;
    const bool full_fa_env = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA");
    // Packed16 scalar decode is a real attention lane, not the legacy KQ-only
    // probe. Auto-enable FULL_FA for the exact persistent packed16/I32 + q4_0
    // scalar decode contracts so restored no-MTP decode cannot zero dst.  The
    // V4_144 diagnostic adapter must pass through the same gate or the selector
    // can say rocm_packed16_decode while compute falls back to KQ-only zero output.
    const bool auto_full_fa_packed16_decode =
        !full_fa_env &&
        (fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK ||
         fa_inst_i32 == GGML_FATTN_INST_DECODE_QK ||
         fa_inst_i32 == GGML_FATTN_INST_NONE) &&
        Q->ne[1] == 1 &&
        K->type == GGML_TYPE_I32 &&
        (V->type == GGML_TYPE_Q4_0 || v4_144_decode_diag_env);
    const bool full_fa = full_fa_env || auto_full_fa_packed16_decode;

    // Auto-select blockfa_recthist_v4_single for f16/q8_0/q4_0 V when FULL_FA=1
    // and no explicit variant is set. The default packed16 path for f16/q8_0 would
    // read V data as q4_0 blocks (incorrect); and the default q4_0 path uses the
    // legacy fattn_from_logits kernel which has poor occupancy at scale.
    // V4_144 is only included for nq==1 diagnostic decode so it reaches the
    // packed16 decode fast-path below and returns before any q4_0 blockFA fallback.
    const bool blockfa_recthist_v4_single_auto =
        full_fa && !blockfa_runtime_any_explicit &&
        (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0 ||
         (v4_144_decode_diag_env && Q->ne[1] == 1));

    const bool blockfa_recthist_v4_single_effective = blockfa_recthist_v4_single || blockfa_recthist_v4_single_auto;
    const bool blockfa_recthist_any_effective = blockfa_recthist_v4_single_effective;
    const bool blockfa_runtime_any_effective = blockfa_recthist_any_effective;
    const bool fused_variant_effective = blockfa_runtime_any_effective;

    const char * variant_name = blockfa_recthist_v4_single_auto && !blockfa_recthist_v4_single ? "blockfa_recthist_v4_single_auto" :
            (blockfa_recthist_v4_single ? "blockfa_recthist_v4_single" : "packed16");
    const char * variant_note = blockfa_recthist_v4_single_effective ? "blockfa_recthist_v4_single" :
            (full_fa ? "full_fa_from_logits" : "kq_only_diagnostic_zero_output");
    if (fused_variant_effective && !full_fa) {
        GGML_ABORT("q8k_dot4_kq blockFA variant requires GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1");
    }

    if (full_fa) {
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
            GGML_ABORT("q8k_dot4_kq full FA path does not support sinks, max_bias, or logit_softcap");
        }
        if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
                mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
            GGML_ABORT("q8k_dot4_kq full FA path mask shape/type unsupported");
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
    const int dot4_debug = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_DOT4_DEBUG") ? 1 : 0;
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
    if (k_is_i32_packed16 && !use_packed16) {
        GGML_ABORT("q8k_dot4_kq packed16-q8 side-channel K requires packed16 K cache sidecars; K=%s physical=%s",
                ggml_type_name(K->type), ggml_cuda_q8k_dot4_k_physical_name(K));
    }
    bool skip_k_repack = false;
    if (use_packed16) {
        // Prefer GGML tensors from registry (allocated by KV cache). Fall back to hipMalloc.
        ggml_tensor * payload_tensor = nullptr;
        ggml_tensor * scales_tensor  = nullptr;
        llama_kv_cache_get_packed16_tensors(K->data, &payload_tensor, &scales_tensor);
        // Packed16 I32 K MUST have registry entries (pre-quantized in cache)
        // with flat physical rows: row = hk * head_stride + k.
        if (k_is_i32_packed16 && !ggml_cuda_q8k_dot4_packed16_sidecar_valid(
                    K, payload_tensor, scales_tensor, true, "launch")) {
            GGML_ABORT("q8k_dot4_kq packed16 sidecar contract failed for physical I32+F16-scale K");
        }
        if (payload_tensor && scales_tensor) {
            // Use GGML tensors directly — no hipMalloc needed.
            k_payload.ptr = (int *) payload_tensor->data;
            k_scales.ptr  = (half *) scales_tensor->data;
            // One-time DOT4 packed16 registry dump
            static bool dot4_registry_printed = false;
            if (!dot4_registry_printed) {
                dot4_registry_printed = true;
                const int64_t head_stride = (payload_tensor->ne[1] > 0 && K->ne[2] > 0) ? payload_tensor->ne[1] / K->ne[2] : 0;
                fprintf(stderr, "DOT4 packed16 registry: payload=%p scales=%p "
                        "payload_ne=(%lld,%lld,%lld,%lld) scales_ne=(%lld,%lld,%lld,%lld) "
                        "physical=packed16_q8_sidechannel_i32_f16scales head_stride=%lld row=hk*head_stride+k\n",
                        (void*)payload_tensor->data, (void*)scales_tensor->data,
                        (long long)payload_tensor->ne[0], (long long)payload_tensor->ne[1],
                        (long long)payload_tensor->ne[2], (long long)payload_tensor->ne[3],
                        (long long)scales_tensor->ne[0], (long long)scales_tensor->ne[1],
                        (long long)scales_tensor->ne[2], (long long)scales_tensor->ne[3],
                        (long long)head_stride);
            }
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
    const int blockfa_split_k_requested = blockfa_split_k_env ? atoi(blockfa_split_k_env) : 1;
    const int blockfa_split_k = blockfa_runtime_any_effective ? max(1, blockfa_split_k_requested) : 1;
    const int blockfa_bn = blockfa_runtime_any_effective ? ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BN", 8) : 8;
    const int blockfa_q_offset = (full_fa && blockfa_runtime_any_effective && mask) ? max(0, nk - nq) : 0;
    const int blockfa_recthist_prefix_k = blockfa_recthist_any_effective ? blockfa_q_offset : 0;
    const int blockfa_recthist_tail_k = blockfa_recthist_any_effective ? nk - blockfa_q_offset : 0;
    if (blockfa_recthist_any_effective) {
        if (blockfa_split_k_requested != 1) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single supports splitK=1 only, got %d", blockfa_split_k_requested);
        }
        if (blockfa_bn != 8 && blockfa_bn != 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires BN8 or BN16, got %d", blockfa_bn);
        }
        const int v4_bm = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BM", 8);
        if (v4_bm != 8 && v4_bm != 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires BM8 or BM16, got %d", v4_bm);
        }
        if (v4_bm == 16 && blockfa_bn == 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single BM16+BN16 not supported (no idle threads for V prefetch)");
        }
    }

    cudaStream_t stream = ctx.stream();
    dim3 q_grid(nq, n_heads_q, batch);
    dim3 block(256);

    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_q_quant = nullptr;
    hipEvent_t ev_k_pack = nullptr;
    hipEvent_t ev_kq = nullptr;
    hipEvent_t ev_ref = nullptr;
    hipEvent_t ev_err = nullptr;
    hipEvent_t ev_blockfa = nullptr;
    hipEvent_t ev_zero = nullptr;
    if (timing) {
        ggml_cuda_q8k_dot4_kq_event_create(&ev_start);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_kq);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_ref);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_err);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_zero);
        CUDA_CHECK(hipEventRecord(ev_start, stream));
    }

    ggml_cuda_q8k_dot4_quant_q_packed16_kernel<<<q_grid, block, 0, stream>>>(
        (const float *) Q->data, q_payload.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, batch);
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_q_quant, stream));
    }

    {
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
        ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, logits.ptr, scale,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
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
                // ---- Packed16 decode / small verify fast-path ----
                const int32_t fa_inst = ((const int32_t *)dst->op_params)[4];
                const bool is_mtp_draft_decode = (fa_inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK);
                const bool is_standard_decode =
                    fa_inst == GGML_FATTN_INST_DECODE_QK || fa_inst == GGML_FATTN_INST_NONE;
                const char * packed16_decode_impl_env = getenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
                const char * packed16_decode_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
                const bool packed16_small_verify_requested =
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_batched_splitk") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa2") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pvwmma") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv_dot4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_dot4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_wmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_batched_splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa2") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
                const bool packed16_small_verify_splitk_requested =
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_splitk") == 0);
                const bool packed16_decode_splitk_requested =
                    packed16_small_verify_splitk_requested ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_splitk") == 0);
                const bool v4_144_nomtp_decode =
                    V->type == GGML_TYPE_V4_K16D16_144 && is_standard_decode;
                const bool v4_144_decode_diag =
                    V->type == GGML_TYPE_V4_K16D16_144 &&
                    (v4_144_nomtp_decode ||
                     ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT"));
                const bool use_default_packed16_decode =
                    nq == 1 && K->type == GGML_TYPE_I32 && (V->type == GGML_TYPE_Q4_0 || v4_144_decode_diag) &&
                    (is_mtp_draft_decode || is_standard_decode);
                const int decode_bn = ggml_cuda_q8k_dot4_kq_env_int(
                    "GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN", (use_default_packed16_decode || packed16_small_verify_requested || packed16_small_verify_splitk_requested) ? 64 : 0);
                const int decode_max_nq = (packed16_small_verify_requested || packed16_small_verify_splitk_requested) ?
                    ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", 4) :
                    (packed16_decode_splitk_requested ? ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_MAX_NQ", 1) : 1);

                if (decode_bn > 0 && nq <= decode_max_nq && (K->type == GGML_TYPE_I32 || is_mtp_draft_decode)) {
                    // For f16-source K, packed16 is already materialized in k_payload/k_scales.
                    // Strides are computed from packed16 layout, not from K tensor strides.
                    const int k_head_stride_rows  = nk;
                    const int k_batch_stride_rows = n_heads_k * k_head_stride_rows;
                    const int decode_vsub = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_VSUB", 8);
                    const char * packed16_decode_impl =
                        (packed16_decode_impl_env && packed16_decode_impl_env[0]) ? packed16_decode_impl_env :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_q4pair") == 0) ? "q4pair" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_scalar") == 0) ? "gqa_scalar" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_waveqk") == 0) ? "waveqk" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_waveqk_q4pair") == 0) ? "waveqk_q4pair" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_pvwmma") == 0) ? "pvwmma" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_pvwmma") == 0) ? "gqa_pvwmma" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_wmma_full") == 0) ? "wmma_full" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_wmma_full") == 0) ? "gqa_wmma_full" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_dsplit") == 0) ? "dsplit" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_logits_debug") == 0) ? "logits_debug" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify") == 0) ? (nq == 1 ? "splitk" : "small_verify") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_splitk") == 0) ? (nq == 1 ? "splitk" : "small_verify_splitk") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_batched_splitk") == 0) ? (nq == 1 ? "splitk" : "small_verify_batched_splitk") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa2") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa2") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa4_pvwmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pvwmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pint8pv") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pint8pv_dot4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_intflash_vfrag_wmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_intflash_vfrag_dot4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_splitk") == 0) ? "splitk" :
                        "scalar";
                    const bool impl_gqa_scalar = strcmp(packed16_decode_impl, "gqa_scalar") == 0;
                    const bool impl_scalar = strcmp(packed16_decode_impl, "scalar") == 0;
                    const bool impl_inline_q4 = strcmp(packed16_decode_impl, "inline_q4") == 0;
                    const bool impl_q4pair = strcmp(packed16_decode_impl, "q4pair") == 0;
                    const bool impl_waveqk = strcmp(packed16_decode_impl, "waveqk") == 0 || strcmp(packed16_decode_impl, "gqa_waveqk") == 0;
                    const bool impl_waveqk_q4pair = strcmp(packed16_decode_impl, "waveqk_q4pair") == 0;
                    const bool impl_pvwmma = strcmp(packed16_decode_impl, "pvwmma") == 0 || strcmp(packed16_decode_impl, "gqa_pvwmma") == 0;
                    const bool impl_wmma_full = strcmp(packed16_decode_impl, "wmma_full") == 0 || strcmp(packed16_decode_impl, "gqa_wmma_full") == 0;
                    const bool impl_dsplit = strcmp(packed16_decode_impl, "dsplit") == 0;
                    const bool impl_logits_debug = strcmp(packed16_decode_impl, "logits_debug") == 0;
                    const bool impl_splitk = strcmp(packed16_decode_impl, "splitk") == 0;
                    bool impl_small_verify = strcmp(packed16_decode_impl, "small_verify") == 0;
                    bool impl_small_verify_splitk = strcmp(packed16_decode_impl, "small_verify_splitk") == 0;
                    bool impl_small_verify_batched_splitk = strcmp(packed16_decode_impl, "small_verify_batched_splitk") == 0;
                    bool impl_small_verify_fa2 = strcmp(packed16_decode_impl, "small_verify_fa2") == 0;
                    bool impl_small_verify_fa4 = strcmp(packed16_decode_impl, "small_verify_fa4") == 0;
                    bool impl_small_verify_fa4_pvwmma = strcmp(packed16_decode_impl, "small_verify_fa4_pvwmma") == 0;
                    bool impl_bm_dot4_pages = strcmp(packed16_decode_impl, "bm_dot4_pages") == 0;
                    bool impl_bm_dot4_pages_pvwmma = strcmp(packed16_decode_impl, "bm_dot4_pages_pvwmma") == 0;
                    bool impl_bm_dot4_pages_pint8pv = strcmp(packed16_decode_impl, "bm_dot4_pages_pint8pv") == 0;
                    bool impl_bm_dot4_pages_pint8pv_dot4 = strcmp(packed16_decode_impl, "bm_dot4_pages_pint8pv_dot4") == 0;
                    bool impl_bm_dot4_pages_intflash_vfrag_dot4 = strcmp(packed16_decode_impl, "bm_dot4_pages_intflash_vfrag_dot4") == 0;
                    bool impl_bm_dot4_pages_intflash_vfrag_wmma = strcmp(packed16_decode_impl, "bm_dot4_pages_intflash_vfrag_wmma") == 0;
                    const bool impl_supported = impl_scalar || impl_gqa_scalar || impl_inline_q4 || impl_q4pair || impl_waveqk || impl_waveqk_q4pair || impl_pvwmma || impl_wmma_full || impl_dsplit || impl_logits_debug || impl_splitk || impl_small_verify || impl_small_verify_splitk || impl_small_verify_batched_splitk || impl_small_verify_fa2 || impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma || impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma;
                    if (!impl_supported) {
                        GGML_ABORT("packed16 decode impl '%s' is not implemented yet; supported in this build: scalar, gqa_scalar, inline_q4, q4pair, waveqk, waveqk_q4pair, splitk, small_verify, small_verify_splitk, small_verify_batched_splitk, small_verify_fa2, small_verify_fa4, small_verify_fa4_pvwmma, bm_dot4_pages, bm_dot4_pages_pvwmma, bm_dot4_pages_pint8pv, bm_dot4_pages_pint8pv_dot4, bm_dot4_pages_intflash_vfrag_dot4, bm_dot4_pages_intflash_vfrag_wmma, dsplit, pvwmma, gqa_pvwmma, wmma_full, gqa_wmma_full, logits_debug", packed16_decode_impl);
                    }
                    const bool packed16_decode_impl_explicit = packed16_decode_impl_env && packed16_decode_impl_env[0];
                    const bool packed16_decode_log =
                        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_PACKED16_DECODE_LOG") ||
                        ggml_cuda_q8k_dot4_kq_env_enabled("COMPRESSED_KV_FATTN_LOG");
                    if (packed16_decode_log) {
                        static std::unordered_set<std::string> packed16_decode_impl_seen;
                        std::string shape_key = std::string(packed16_decode_impl) + "_nq" + std::to_string(nq) + "_nk" + std::to_string(nk) + "_hq" + std::to_string(n_heads_q) + "_hk" + std::to_string(n_heads_k) + "_gqa" + std::to_string(gqa_ratio);
                        if (packed16_decode_impl_explicit || packed16_decode_impl_seen.find(shape_key) == packed16_decode_impl_seen.end()) {
                            packed16_decode_impl_seen.insert(shape_key);
                            fprintf(stderr,
                                "packed16_decode_impl selected=%s route=%s nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d V=%s\n",
                                packed16_decode_impl, packed16_decode_route ? packed16_decode_route : "",
                                nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio, ggml_type_name(V->type));
                        }
                    }
                    const int decode_v_debug = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG") ? 1 : 0;
                    const int decode_v_debug_hq = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_HQ", 0);
                    const int decode_v_debug_k  = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_K", 0);
                    const int decode_v_debug_d0 = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_D0", 0);
                    const int decode_v_debug_count = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_COUNT", 8);
                    if (v4_144_decode_diag && !(impl_scalar || impl_splitk)) {
                        GGML_ABORT("V4_K16D16_144 q8k decode diagnostic currently supports packed16 decode impl=scalar or splitk, got %s", packed16_decode_impl);
                    }
                    const bool inline_q4 = !v4_144_decode_diag && (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_INLINE_Q4") || impl_inline_q4);
                    const bool q4pair = !v4_144_decode_diag && V->type == GGML_TYPE_Q4_0 &&
                        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR") || impl_q4pair);
                    const int splitk_threshold = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD", 2048);
                    const bool v4_144_splitk = v4_144_decode_diag &&
                        (v4_144_nomtp_decode || impl_splitk || ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_DECODE_SPLITK"));
                    const bool splitk = !impl_logits_debug && !impl_dsplit &&
                        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK") || impl_splitk || impl_small_verify_splitk || nk >= splitk_threshold) &&
                        (V->type == GGML_TYPE_Q4_0 || v4_144_splitk);
                    // small_verify/batched_splitk only apply for nq >= 2; nq == 1 falls through
                    // to regular decode (splitk/BN64/etc)
                    if (impl_small_verify && nq < 2) impl_small_verify = false;
                    if (impl_small_verify_splitk && nq < 2) impl_small_verify_splitk = false;
                    if (impl_small_verify_batched_splitk && nq < 2) impl_small_verify_batched_splitk = false;
                    if (impl_small_verify_fa2 && nq < 2) impl_small_verify_fa2 = false;
                    if (impl_small_verify_fa4 && nq < 2) impl_small_verify_fa4 = false;
                    if (impl_small_verify_fa4_pvwmma && nq < 2) impl_small_verify_fa4_pvwmma = false;
                    if (impl_bm_dot4_pages && nq < 2) impl_bm_dot4_pages = false;
                    if (impl_bm_dot4_pages_pvwmma && nq < 2) impl_bm_dot4_pages_pvwmma = false;
                    if (impl_bm_dot4_pages_pint8pv && nq < 2) impl_bm_dot4_pages_pint8pv = false;
                    if (impl_bm_dot4_pages_pint8pv_dot4 && nq < 2) impl_bm_dot4_pages_pint8pv_dot4 = false;
                    if (impl_bm_dot4_pages_intflash_vfrag_dot4 && nq < 2) impl_bm_dot4_pages_intflash_vfrag_dot4 = false;
                    if (impl_bm_dot4_pages_intflash_vfrag_wmma && nq < 2) impl_bm_dot4_pages_intflash_vfrag_wmma = false;
                    if ((impl_small_verify || impl_small_verify_splitk || impl_small_verify_batched_splitk || impl_small_verify_fa2 || impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma || impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma) && !(nq >= 2 && nq <= decode_max_nq && K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0)) {
                        GGML_ABORT("packed16 %s requires I32 K, q4_0 V, and 2 <= nq <= %d; got nq=%d K=%s V=%s",
                            packed16_decode_impl, decode_max_nq, nq, ggml_type_name(K->type), ggml_type_name(V->type));
                    }
                    if (impl_small_verify) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 small_verify supports gqa_ratio <= 8, got %d", gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SMALL_VERIFY(64, 8, 8) }
                        else { GGML_ABORT("packed16 small_verify: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_splitk) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 small_verify_splitk supports gqa_ratio <= 8, got %d", gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SMALL_VERIFY_SPLITK(64, 8, 8) }
                        else { GGML_ABORT("packed16 small_verify_splitk: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_batched_splitk) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 2, 512) }
                            else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 3, 512) }
                            else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 4, 512) }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_fa2) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            // Qwen 27B MTP verify is GQA6.  Specialize the qtile state to
                            // GH_MAX=6 instead of the generic GH_MAX=8 path: less LDS,
                            // fewer unrolled PV/softmax lanes, and lower register pressure
                            // without changing route semantics or adding another PV variant.
                            if (gqa_ratio == 6) {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 2, 224) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 3, 224) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 4, 224) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 2, 112) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 3, 112) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 4, 112) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_vsub == 8) {
                            if (impl_bm_dot4_pages_pvwmma) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                            } else if (impl_bm_dot4_pages_pint8pv) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                            } else if (impl_bm_dot4_pages_pint8pv_dot4) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                            } else if (impl_bm_dot4_pages_intflash_vfrag_dot4) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                            } else if (impl_bm_dot4_pages_intflash_vfrag_wmma) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected VSUB=8, got VSUB=%d", packed16_decode_impl, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            if (impl_small_verify_fa4_pvwmma) {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 2, 128, true) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 3, 128, true) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 4, 128, true) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 2, 128, false) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 3, 128, false) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 4, 128, false) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_logits_debug) {
                        ggml_cuda_pool_alloc<float> decode_logits(pool);
                        decode_logits.alloc(logits_ne);
                        ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, decode_logits.ptr, scale,
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
                        CUDA_CHECK(cudaGetLastError());
                        dim3 fa_grid(nq, n_heads_q, batch);
                        ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                            decode_logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
                        CUDA_CHECK(cudaGetLastError());
                        return;
                    }
                    if (splitk) {
                        const int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512);
                        const int n_splits = (nk + split_size - 1) / split_size;
                        blockfa_partial_o.alloc((size_t) batch * nq * n_heads_q * n_splits * 256);
                        blockfa_partial_m.alloc((size_t) batch * nq * n_heads_q * n_splits);
                        blockfa_partial_l.alloc((size_t) batch * nq * n_heads_q * n_splits);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SPLITK(64, 8) }
                        else { GGML_ABORT("q8k_dot4_kq split-K decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_dsplit) {
                        if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 dsplit decode currently requires V=q4_0");
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_DSPLIT(64, 8, 64) }
                        else { GGML_ABORT("q8k_dot4_kq D-split decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (decode_bn == 64 && decode_vsub == 8) {
                        if (impl_wmma_full) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 gqa_wmma_full decode currently requires V=q4_0");
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_wmma_full decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA_WMMA_FULL(64, 8)
                        } else if (impl_waveqk_q4pair) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 waveqk_q4pair decode currently requires V=q4_0");
                            LAUNCH_DECODE_WAVEQK_Q4PAIR(64, 8)
                        } else if (impl_waveqk) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 waveqk decode currently requires V=q4_0");
                            LAUNCH_DECODE_WAVEQK(64, 8)
                        } else if (impl_pvwmma) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 gqa_pvwmma decode currently requires V=q4_0");
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_pvwmma decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA_PVWMMA(64, 8)
                        } else if (impl_gqa_scalar) {
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_scalar decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA(64, 8, 8)
                        } else if (q4pair)      LAUNCH_DECODE_Q4PAIR(64, 8)
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
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                } else if (v4_bm == 8 && v4_bn == 16) {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                } else {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
            }
        } else {
            ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Diagnostic KQ-only mode: deliberately leave FA output zeroed so this route
        // cannot masquerade as production attention. It exists only to validate
        // runtime packed-DOT4 KQ layout, route contracts, and ISA.
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
        const float total_ms   = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_zero);
        GGML_LOG_INFO("%s: q8k_dot4_kq_timing call=%llu variant=%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d check=%d q_quant_ms=%.6f k_pack_ms=%.6f kq_ms=%.6f ref_ms=%.6f err_ms=%.6f zero_ms=%.6f total_ms=%.6f fa_ms=%.6f blockfa_ms=%.6f q_offset=%d prefix_k=%d tail_k=%d full_fa=%d blockfa_split_k=%d blockfa_bn=%d\n",
            __func__, timing_call, variant_name, nq, nk, n_heads_q, n_heads_k, batch, check && !fused_variant_effective ? 1 : 0,
            q_quant_ms, k_pack_ms, kq_ms, ref_ms, err_ms, zero_ms, total_ms, fa_ms, blockfa_ms,
            blockfa_recthist_v4_single_effective ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k,
            full_fa ? 1 : 0, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_start);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_kq);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_ref);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_err);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_zero);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double(q_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double k_mib = double(k_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double logits_mib = fused_variant_effective ? 0.0 : double(logits_ne * sizeof(float)) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=rocm_q8k_dot4_kq variant=%s full_fa=%d nq=%d nk=%d heads_q=%d heads_k=%d batch=%d q_payload=%.3fMiB k_payload=%.3fMiB logits=%.3fMiB note=%s blockfa_split_k=%d blockfa_bn=%d q_offset=%d prefix_k=%d tail_k=%d\n",
            __func__, variant_name, full_fa ? 1 : 0, nq, nk, n_heads_q, n_heads_k, batch, q_mib, k_mib, logits_mib, variant_note, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0,
            blockfa_recthist_v4_single_effective ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k);
    }

    // Detach hipMalloc'd persistent buffers from pool allocator destructors.
    if (ggml_cuda_q8k_dot4_packed16_k_cache_enabled()) {
        k_payload.ptr = nullptr;
        k_scales.ptr  = nullptr;
    }

    GGML_UNUSED(sinks);
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
    const int k_scale_group_qblocks = ggml_cuda_pack_k_packed16_scale_group_qblocks();

    static bool pack_debug_printed = false;
    if (!pack_debug_printed && ggml_cuda_pack_k_packed16_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
        pack_debug_printed = true;
        fprintf(stderr,
            "pack_k_i32: k_cur type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
            "payload ne=[%lld,%lld,%lld,%lld] scales ne=[%lld,%lld,%lld,%lld] "
            "n_heads=%d nk_cur=%d kv_size=%d src_head_stride=%lld scale_mode=%d scale_mul=%g scale_group_qblocks=%d\n",
            k_cur->type, (long long) k_cur->ne[0], (long long) k_cur->ne[1], (long long) k_cur->ne[2], (long long) k_cur->ne[3],
            k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
            (long long) payload->ne[0], (long long) payload->ne[1], (long long) payload->ne[2], (long long) payload->ne[3],
            (long long) scales->ne[0], (long long) scales->ne[1], (long long) scales->ne[2], (long long) scales->ne[3],
            n_heads, nk_cur, kv_size, (long long) src_head_stride_bytes, k_scale_mode, (double) k_scale_mul, k_scale_group_qblocks);
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
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul, k_scale_group_qblocks);
    } else {
        GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
        ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int32_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul, k_scale_group_qblocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

static constexpr int GGML_CUDA_V4_K16D16_D = 256;
static constexpr int GGML_CUDA_V4_K16D16_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_D_TILE = 16;
static constexpr int GGML_CUDA_V4_K16D16_WORDS_PER_D = 2;
static constexpr int GGML_CUDA_V4_K16D16_ROW_BYTES = sizeof(block_v4_k16d16);
static constexpr int GGML_CUDA_V4_K16D16_BLOCK_BYTES = GGML_CUDA_V4_K16D16_ROW_BYTES * GGML_CUDA_V4_K16D16_K;
static constexpr int GGML_CUDA_V4_K16D16_PAYLOAD_BYTES = GGML_CUDA_V4_K16D16_D * GGML_CUDA_V4_K16D16_WORDS_PER_D * (int) sizeof(uint32_t);
static_assert(GGML_CUDA_V4_K16D16_PAYLOAD_BYTES + (GGML_CUDA_V4_K16D16_D / GGML_CUDA_V4_K16D16_D_TILE) * (int) sizeof(half) == GGML_CUDA_V4_K16D16_BLOCK_BYTES, "bad V4_K16D16 byte math");

static __device__ __forceinline__ float ggml_cuda_v4_load_src(
        const char * __restrict__ src,
        const bool src_f16,
        const int64_t nb01,
        const int64_t nb02,
        const int64_t nb03,
        const int64_t src_head_stride_bytes,
        const int token,
        const int head,
        const int batch,
        const int d) {
    const char * p = src + int64_t(batch) * nb03 + int64_t(token) * nb01 + int64_t(head) * src_head_stride_bytes;
    return src_f16 ? __half2float(((const half *) p)[d]) : ((const float *) p)[d];
}

template<typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_v4_k16d16_indexed_kernel(
        const char  * __restrict__ Vcur,
        char        * __restrict__ v4_cache,
        half        * __restrict__ v_tail,
        const idx_t * __restrict__ v_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int64_t v4_nb1,
        int64_t v4_nb2,
        int64_t tail_nb1,
        int64_t tail_nb2,
        int nk_cur,
        int n_heads_v,
        int batch,
        int kv_size,
        bool src_f16) {
    const int d = int(threadIdx.x);
    const int token = int(blockIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    if (d >= GGML_CUDA_V4_K16D16_D || token >= nk_cur || hv >= n_heads_v || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) v_idxs[token];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    const int tail_slot = cell & (GGML_CUDA_V4_K16D16_K - 1);

    const float x = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d);
    half * tail_row = (half *) ((char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + tail_slot) * tail_nb1);
    tail_row[d] = __float2half(x);

    if (tail_slot != GGML_CUDA_V4_K16D16_K - 1) {
        return;
    }

    const int block_start = cell - tail_slot;
    char * block = v4_cache + int64_t(b) * v4_nb2 + int64_t(hv * kv_size + block_start) * v4_nb1;
    const int d_tile = d / GGML_CUDA_V4_K16D16_D_TILE;
    const int d0 = d_tile * GGML_CUDA_V4_K16D16_D_TILE;

    float amax = 0.0f;
    float maxv = 0.0f;
#pragma unroll
    for (int kk = 0; kk < GGML_CUDA_V4_K16D16_K; ++kk) {
        const half * row = (const half *) ((const char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + kk) * tail_nb1);
#pragma unroll
        for (int dd = 0; dd < GGML_CUDA_V4_K16D16_D_TILE; ++dd) {
            const float v = __half2float(row[d0 + dd]);
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
                maxv = v;
            }
        }
    }
    float scale = maxv / -8.0f;
    if (!(scale != 0.0f) || !isfinite(scale)) {
        scale = 1.0f;
    }
    const float inv_scale = 1.0f / scale;

#pragma unroll
    for (int g = 0; g < GGML_CUDA_V4_K16D16_WORDS_PER_D; ++g) {
        uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kk = g * 8 + j;
            const half * row = (const half *) ((const char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + kk) * tail_nb1);
            const float v = __half2float(row[d]);
            int q = (int) (v * inv_scale + 8.5f);
            q = q < 0 ? 0 : (q > 15 ? 15 : q);
            word |= (uint32_t(q) & 0x0fu) << (4 * j);
        }
        ((uint32_t *) (block + d * GGML_CUDA_V4_K16D16_WORDS_PER_D * (int) sizeof(uint32_t)))[g] = word;
    }
    if ((d & (GGML_CUDA_V4_K16D16_D_TILE - 1)) == 0) {
        ((half *) (block + GGML_CUDA_V4_K16D16_PAYLOAD_BYTES))[d_tile] = __float2half(scale);
    }
}

void ggml_cuda_op_pack_v4_k16d16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * v_cur  = dst->src[0];
    ggml_tensor * v_tail = dst->src[1];
    ggml_tensor * v_idxs = dst->src[2];
    ggml_tensor * v4     = dst;

    GGML_ASSERT(v4->type == GGML_TYPE_V4_K16D16);
    GGML_ASSERT(v_tail->type == GGML_TYPE_F16);
    GGML_ASSERT(v_idxs->type == GGML_TYPE_I64 || v_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(v4->ne[0] == GGML_CUDA_V4_K16D16_D);
    GGML_ASSERT(v_tail->ne[0] == GGML_CUDA_V4_K16D16_D);
    GGML_ASSERT(v_tail->ne[1] % GGML_CUDA_V4_K16D16_K == 0);

    const int D = GGML_CUDA_V4_K16D16_D;
    const bool src_f16 = v_cur->type == GGML_TYPE_F16;
    const int batch = (int) v_cur->ne[3];
    int n_heads = 0;
    int nk_cur = 0;
    int64_t src_head_stride_bytes = 0;
    if (v_cur->ne[0] == D) {
        n_heads = (int) v_cur->ne[1];
        nk_cur = (int) v_cur->ne[2];
        src_head_stride_bytes = v_cur->nb[1];
    } else {
        GGML_ASSERT(v_cur->ne[0] % D == 0);
        n_heads = (int) (v_cur->ne[0] / D);
        nk_cur = (int) v_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(v_cur->type);
    }
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(v4->ne[1] % n_heads == 0);
    GGML_ASSERT(v_tail->ne[1] == GGML_CUDA_V4_K16D16_K * n_heads);
    const int kv_size = (int) (v4->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);

    const int64_t src_token_stride_bytes = (v_cur->ne[0] == D) ? v_cur->nb[2] : v_cur->nb[1];
    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(D);
    cudaStream_t stream = ctx.stream();
    if (v_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_pack_v4_k16d16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v4->data, (half *) v_tail->data, (const int64_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v4->nb[1], v4->nb[2], v_tail->nb[1], v_tail->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    } else {
        ggml_cuda_pack_v4_k16d16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v4->data, (half *) v_tail->data, (const int32_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v4->nb[1], v4->nb[2], v_tail->nb[1], v_tail->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    }
    CUDA_CHECK(cudaGetLastError());
}

static constexpr int GGML_CUDA_V4_K16D16_144_D = 256;
static constexpr int GGML_CUDA_V4_K16D16_144_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_144_D32 = 32;
static constexpr int GGML_CUDA_V4_K16D16_144_D32_BLOCKS = GGML_CUDA_V4_K16D16_144_D / GGML_CUDA_V4_K16D16_144_D32;
static constexpr int GGML_CUDA_V4_K16D16_144_ROW_BYTES = sizeof(block_v4_k16d16_144);
static constexpr int GGML_CUDA_V4_K16D16_144_BLOCK_BYTES = GGML_CUDA_V4_K16D16_144_ROW_BYTES * GGML_CUDA_V4_K16D16_144_K;
static constexpr int GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES = GGML_CUDA_V4_K16D16_144_D * 2 * (int) sizeof(uint32_t);
static constexpr int GGML_CUDA_V4_K16D16_144_SCALE_BYTES = GGML_CUDA_V4_K16D16_144_D32_BLOCKS * GGML_CUDA_V4_K16D16_144_K * (int) sizeof(half);
static_assert(GGML_CUDA_V4_K16D16_144_BLOCK_BYTES == 2304, "bad V4_K16D16_144 block bytes");
static_assert(GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES == 2048, "bad V4_K16D16_144 payload bytes");
static_assert(GGML_CUDA_V4_K16D16_144_SCALE_BYTES == 256, "bad V4_K16D16_144 scale bytes");
static_assert(GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES + GGML_CUDA_V4_K16D16_144_SCALE_BYTES == GGML_CUDA_V4_K16D16_144_BLOCK_BYTES, "bad V4_K16D16_144 byte math");

static __device__ __forceinline__ void ggml_cuda_v4_k16d16_144_store_nibble(
        uint32_t * __restrict__ wordp,
        const int slot,
        const uint32_t q_unsigned) {
    const uint32_t shift = uint32_t(4 * (slot & 7));
    const uint32_t mask  = uint32_t(0x0fu) << shift;
    uint32_t old = *wordp;
    uint32_t assumed;
    do {
        assumed = old;
        const uint32_t desired = (assumed & ~mask) | ((q_unsigned & 0x0fu) << shift);
        old = atomicCAS((unsigned int *) wordp, (unsigned int) assumed, (unsigned int) desired);
    } while (old != assumed);
}

static __device__ __forceinline__ uint16_t ggml_cuda_v4_k16d16_144_half_bits(const half h) {
    union {
        half h;
        uint16_t u;
    } cvt;
    cvt.h = h;
    return cvt.u;
}

static __device__ __forceinline__ void ggml_cuda_v4_k16d16_144_store_scale(
        half * __restrict__ scales,
        const int scale_idx,
        const half scale) {
    uint32_t * wordp = (uint32_t *) (scales + (scale_idx & ~1));
    const uint32_t shift = uint32_t(16 * (scale_idx & 1));
    const uint32_t mask  = uint32_t(0xffffu) << shift;
    const uint32_t bits  = uint32_t(ggml_cuda_v4_k16d16_144_half_bits(scale)) << shift;
    uint32_t old = *wordp;
    uint32_t assumed;
    do {
        assumed = old;
        const uint32_t desired = (assumed & ~mask) | bits;
        old = atomicCAS((unsigned int *) wordp, (unsigned int) assumed, (unsigned int) desired);
    } while (old != assumed);
}

template<typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_v4_k16d16_144_indexed_kernel(
        const char  * __restrict__ Vcur,
        char        * __restrict__ v144_cache,
        const idx_t * __restrict__ v_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int64_t v144_nb1,
        int64_t v144_nb2,
        int nk_cur,
        int n_heads_v,
        int batch,
        int kv_size,
        bool src_f16) {
    const int d = int(threadIdx.x);
    const int token = int(blockIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    if (d >= GGML_CUDA_V4_K16D16_144_D || token >= nk_cur || hv >= n_heads_v || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) v_idxs[token];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    const int slot = cell & (GGML_CUDA_V4_K16D16_144_K - 1);
    const int k16_base = cell & ~(GGML_CUDA_V4_K16D16_144_K - 1);

    char * block = v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + k16_base) * v144_nb1;
    const int d32 = d >> 5;
    const int d0 = d32 * GGML_CUDA_V4_K16D16_144_D32;

    float amax = 0.0f;
    float maxv = 0.0f;
#pragma unroll
    for (int dd = 0; dd < GGML_CUDA_V4_K16D16_144_D32; ++dd) {
        const float v = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d0 + dd);
        const float av = fabsf(v);
        if (av > amax) {
            amax = av;
            maxv = v;
        }
    }
    const float scale = maxv / -8.0f;
    const float inv_scale = scale ? 1.0f / scale : 0.0f;

    const float x = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d);
    int q = (int8_t) (x * inv_scale + 8.5f);
    q = q > 15 ? 15 : q;

    uint32_t * payload = (uint32_t *) block;
    ggml_cuda_v4_k16d16_144_store_nibble(&payload[d * 2 + (slot >> 3)], slot, (uint32_t) q);

    if ((d & (GGML_CUDA_V4_K16D16_144_D32 - 1)) == 0) {
        half * scales = (half *) (block + GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES);
        const int scale_idx = d32 * GGML_CUDA_V4_K16D16_144_K + slot;
        ggml_cuda_v4_k16d16_144_store_scale(scales, scale_idx, __float2half(scale));
    }
}

void ggml_cuda_op_pack_v4_k16d16_144(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * v_cur  = dst->src[0];
    ggml_tensor * v_idxs = dst->src[1];
    ggml_tensor * v144   = dst;

    GGML_ASSERT(v144->type == GGML_TYPE_V4_K16D16_144);
    GGML_ASSERT(v_idxs->type == GGML_TYPE_I64 || v_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(v144->ne[0] == GGML_CUDA_V4_K16D16_144_D);

    const int D = GGML_CUDA_V4_K16D16_144_D;
    const bool src_f16 = v_cur->type == GGML_TYPE_F16;
    const int batch = (int) v_cur->ne[3];
    int n_heads = 0;
    int nk_cur = 0;
    int64_t src_head_stride_bytes = 0;
    if (v_cur->ne[0] == D) {
        n_heads = (int) v_cur->ne[1];
        nk_cur = (int) v_cur->ne[2];
        src_head_stride_bytes = v_cur->nb[1];
    } else {
        GGML_ASSERT(v_cur->ne[0] % D == 0);
        n_heads = (int) (v_cur->ne[0] / D);
        nk_cur = (int) v_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(v_cur->type);
    }
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(v144->ne[1] % n_heads == 0);
    const int kv_size = (int) (v144->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);
    GGML_ASSERT((kv_size % GGML_CUDA_V4_K16D16_144_K) == 0);
    GGML_ASSERT(v144->nb[1] == GGML_CUDA_V4_K16D16_144_ROW_BYTES);

    const int64_t src_token_stride_bytes = (v_cur->ne[0] == D) ? v_cur->nb[2] : v_cur->nb[1];
    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(D);
    cudaStream_t stream = ctx.stream();
    if (v_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v144->data, (const int64_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    } else {
        ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v144->data, (const int32_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_USE_HIP
