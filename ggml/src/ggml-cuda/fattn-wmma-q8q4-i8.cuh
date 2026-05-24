#pragma once

#include "common.cuh"

#include <cstdlib>
#include <cstring>

static inline bool ggml_cuda_q8q4_wmma_i8_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_wmma_i8_unsafe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_wmma_i8_qscale16_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_QSCALE16");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_wmma_i8_flash_prefill_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_FLASH_PREFILL");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_wmma_i8_allow_gqa6_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_flash_prefill_min_nq() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_FLASH_PREFILL_MIN_NQ");
    return env ? atoll(env) : 256;
#else
    return 0;
#endif
}

static constexpr int64_t GGML_CUDA_Q8Q4_WMMA_I8_KV_CHUNK = 256;

static inline bool ggml_cuda_q8q4_wmma_i8_has_attn_sinks(const ggml_tensor * dst) {
    return dst->src[4] != nullptr;
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_real_kv_len(const ggml_tensor * dst) {
    return dst->src[1]->ne[1];
}

static inline bool ggml_cuda_q8q4_wmma_i8_real_kv_aligned(const ggml_tensor * dst) {
    return ggml_cuda_q8q4_wmma_i8_real_kv_len(dst) % GGML_CUDA_Q8Q4_WMMA_I8_KV_CHUNK == 0;
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_n_chunks_for_kv(const int64_t nk) {
    return (nk + GGML_CUDA_Q8Q4_WMMA_I8_KV_CHUNK - 1) / GGML_CUDA_Q8Q4_WMMA_I8_KV_CHUNK;
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_scratch_alloc_chunks(const int64_t n_chunks) {
    // During long prefill nk grows by micro-batch. Allocating exact-size
    // partial buffers for every new nk makes the CUDA pool retain a staircase
    // of old partial_acc buffers. Bucket the allocation so one cached buffer is
    // reused instead of accumulating O(sum(nk)) scratch across the request.
    const char * stable_nkv_env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_STABLE_NKV");
    if (!stable_nkv_env) {
        // Reuse the existing stable-NKV knob used by the ROCm quant-prefill path
        // when callers already set it for long-context server runs.
        stable_nkv_env = getenv("GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV");
    }
    if (stable_nkv_env) {
        const int64_t stable_nkv = atoll(stable_nkv_env);
        if (stable_nkv > 0) {
            const int64_t stable_chunks = ggml_cuda_q8q4_wmma_i8_n_chunks_for_kv(stable_nkv);
            return stable_chunks > n_chunks ? stable_chunks : n_chunks;
        }
    }

    int64_t bucket = 1;
    while (bucket < n_chunks) {
        bucket <<= 1;
    }
    return bucket;
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head_chunks(const ggml_tensor * dst, const int64_t n_chunks) {
    const ggml_tensor * Q = dst->src[0];
    const int64_t nq = Q->ne[1];
    const int64_t d = Q->ne[0];
    const int64_t q_scale_width = ggml_cuda_q8q4_wmma_i8_qscale16_enabled() ? 16 : 32;
    return n_chunks * nq * d * (int64_t) sizeof(float) + 2 * n_chunks * nq * (int64_t) sizeof(float) +
           nq * d * (int64_t) sizeof(int8_t) + nq * (d / q_scale_width) * (int64_t) sizeof(float);
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head(const ggml_tensor * dst) {
    const int64_t n_chunks = ggml_cuda_q8q4_wmma_i8_n_chunks_for_kv(ggml_cuda_q8q4_wmma_i8_real_kv_len(dst));
    return ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head_chunks(dst, ggml_cuda_q8q4_wmma_i8_scratch_alloc_chunks(n_chunks));
}

static inline int64_t ggml_cuda_q8q4_wmma_i8_max_mib() {
    const char * max_mib_env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_MAX_MIB");
    // Route is opt-in; default scratch cap is high enough to batch all Q heads
    // for the important pp4096/D256 case (~1 GiB active scratch on Qwen3.6-35B),
    // while still forcing very large shapes back to the lower-scratch per-head path.
    return max_mib_env ? atoll(max_mib_env) : 2048;
}

static inline int ggml_cuda_q8q4_wmma_i8_layer_id(const ggml_tensor * dst) {
    const char * name = dst->name;
    const char * dash = name ? strrchr(name, '-') : nullptr;
    if (!dash || !dash[1]) {
        return -1;
    }
    char * end = nullptr;
    const long id = strtol(dash + 1, &end, 10);
    return end && *end == '\0' ? (int) id : -1;
}

static inline bool ggml_cuda_q8q4_wmma_i8_layer_list_contains(const char * list, const int id) {
    if (!list || id < 0) {
        return false;
    }

    const char * p = list;
    while (*p) {
        while (*p == ',' || *p == ';' || *p == ' ' || *p == '\t') {
            ++p;
        }
        if (!*p) {
            break;
        }

        char * end = nullptr;
        const long first = strtol(p, &end, 10);
        if (end == p) {
            while (*p && *p != ',' && *p != ';' && *p != ' ' && *p != '\t') {
                ++p;
            }
            continue;
        }

        long last = first;
        if (*end == '-') {
            char * range_end = nullptr;
            const long parsed_last = strtol(end + 1, &range_end, 10);
            if (range_end != end + 1) {
                last = parsed_last;
                end = range_end;
            }
        }
        if ((first <= id && id <= last) || (last <= id && id <= first)) {
            return true;
        }
        p = end;
    }
    return false;
}

static inline bool ggml_cuda_q8q4_wmma_i8_layer_filter_allows(const ggml_tensor * dst) {
    const char * one_env       = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER");
    const char * min_env       = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN");
    const char * max_env       = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MAX");
    const char * skip_one_env  = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYER");
    const char * skip_list_env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYERS");

    const bool has_include_filter = one_env || min_env || max_env;
    const bool has_skip_filter = skip_one_env || skip_list_env;
    if (!has_include_filter && !has_skip_filter) {
        return true;
    }

    const int id = ggml_cuda_q8q4_wmma_i8_layer_id(dst);
    if (id < 0) {
        return !has_include_filter;
    }
    if (one_env && id != atoi(one_env)) {
        return false;
    }
    if (min_env && id < atoi(min_env)) {
        return false;
    }
    if (max_env && id > atoi(max_env)) {
        return false;
    }
    if (skip_one_env && id == atoi(skip_one_env)) {
        return false;
    }
    if (ggml_cuda_q8q4_wmma_i8_layer_list_contains(skip_list_env, id)) {
        return false;
    }
    return true;
}

static inline bool ggml_cuda_q8q4_wmma_i8_supported(const int cc, const ggml_tensor * dst, const float max_bias, const float logit_softcap) {
#ifdef GGML_USE_HIP
    // Additional unsafe lab flag required until backend-op parity is fixed.
    if (!ggml_cuda_q8q4_wmma_i8_enabled() || !ggml_cuda_q8q4_wmma_i8_unsafe_enabled() || !GGML_CUDA_CC_IS_RDNA3(cc) ||
            !ggml_cuda_q8q4_wmma_i8_layer_filter_allows(dst)) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = ggml_cuda_q8q4_wmma_i8_has_attn_sinks(dst) ? dst->src[4] : nullptr;

    if (!Q || !K || !V) {
        return false;
    }

    const bool dtype_ok = Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 && dst->type == GGML_TYPE_F32;
    const bool d_ok = Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 && dst->ne[0] == 256;
    const bool seq_ok = Q->ne[1] > 2 && K->ne[1] >= Q->ne[1] && V->ne[1] == K->ne[1];
    const bool head_ok = K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3];
    const int64_t gqa_ratio = head_ok ? Q->ne[2] / K->ne[2] : 0;
    // GQA=6 is the Qwen3.6-27B lab-reopen shape. Keep it behind an extra
    // explicit gate until it has the same coverage as the original GQA=4/8 set.
    const bool gqa_ok = gqa_ratio == 4 || gqa_ratio == 8 ||
        (gqa_ratio == 6 && ggml_cuda_q8q4_wmma_i8_allow_gqa6_enabled());
    const bool dst_ok = dst->ne[0] == V->ne[0] && dst->ne[1] == Q->ne[2] && dst->ne[2] == Q->ne[1] && dst->ne[3] == Q->ne[3];
    const bool mask_ok = mask && mask->type == GGML_TYPE_F16 && mask->ne[0] == K->ne[1] && mask->ne[1] == Q->ne[1] &&
        mask->ne[2] == 1 && mask->ne[3] == Q->ne[3] && max_bias == 0.0f;
    const bool softcap_ok = logit_softcap == 0.0f;
    const bool sinks_ok = !sinks || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] >= Q->ne[2]);
    const bool kv_aligned_ok = ggml_cuda_q8q4_wmma_i8_real_kv_aligned(dst);

    if (!(dtype_ok && d_ok && seq_ok && head_ok && gqa_ok && dst_ok && mask_ok && softcap_ok && sinks_ok && kv_aligned_ok)) {
        return false;
    }

    if (ggml_cuda_q8q4_wmma_i8_flash_prefill_enabled() && Q->ne[1] >= ggml_cuda_q8q4_wmma_i8_flash_prefill_min_nq()) {
        return true;
    }

    const int64_t max_mib = ggml_cuda_q8q4_wmma_i8_max_mib();
    const int64_t scratch = ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head(dst);
    return max_mib <= 0 || scratch <= max_mib * 1024LL * 1024LL;
#else
    GGML_UNUSED(cc); GGML_UNUSED(dst); GGML_UNUSED(max_bias); GGML_UNUSED(logit_softcap);
    return false;
#endif
}

void ggml_cuda_flash_attn_ext_q8q4_wmma_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_USE_HIP)
#include <rocwmma/rocwmma.hpp>

static constexpr int GGML_CUDA_Q8Q4_I8_D = 256;
static constexpr int GGML_CUDA_Q8Q4_I8_QK = 32;
static constexpr int GGML_CUDA_Q8Q4_I8_QSCALE16_QK = 16;
static constexpr int GGML_CUDA_Q8Q4_I8_BLOCKS = GGML_CUDA_Q8Q4_I8_D / GGML_CUDA_Q8Q4_I8_QK;
static constexpr int GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS = GGML_CUDA_Q8Q4_I8_D / GGML_CUDA_Q8Q4_I8_QSCALE16_QK;
static constexpr int GGML_CUDA_Q8Q4_I8_WMMA_M = 16;
static constexpr int GGML_CUDA_Q8Q4_I8_WMMA_N = 16;
static constexpr int GGML_CUDA_Q8Q4_I8_WMMA_K = 16;
static constexpr int GGML_CUDA_Q8Q4_I8_CHUNK = (int) GGML_CUDA_Q8Q4_WMMA_I8_KV_CHUNK;

static __device__ __forceinline__ int8_t ggml_cuda_q8q4_i8_clamp(float x) {
    int v = (int) lrintf(x);
    v = v < -128 ? -128 : (v > 127 ? 127 : v);
    return (int8_t) v;
}

static __device__ __forceinline__ float ggml_cuda_q8q4_i8_mask_value(
        const char * __restrict__ mask,
        int64_t nb30, int64_t nb31, int64_t nb33,
        int64_t ne33,
        int q,
        int k,
        int ib) {
    if (!mask) {
        return 0.0f;
    }
    const char * p = mask + int64_t(ib % ne33) * nb33 + int64_t(q) * nb31 + int64_t(k) * nb30;
    return __half2float(*(const half *) p);
}

// Sink is a softmax denominator-only pseudo-token. It has no V contribution.
// For split-K, call this once after all real K/V chunks have been merged.
static __device__ __forceinline__ void ggml_cuda_q8q4_i8_apply_attn_sink(
        const float sink,
        float & row_max,
        float & denom,
        float & acc) {
    const float next_max  = fmaxf(row_max, sink);
    const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
    const float sink_exp  = expf(sink - next_max);

    acc *= old_scale;
    denom = denom * old_scale + sink_exp;
    row_max = next_max;
}

static __global__ __launch_bounds__(32, 4) void ggml_cuda_q8q4_i8_quant_q_kernel(
        const char * __restrict__ Q,
        int8_t     * __restrict__ q_i8,
        float      * __restrict__ q_scales,
        int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
        int nq,
        int hq0,
        int ib0,
        int n_heads_q,
        int q_scale_blocks) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int qb = blockIdx.y;
    const int instance = blockIdx.z;
    const int linear_head = hq0 + instance;
    const int hq = linear_head % n_heads_q;
    const int ib = ib0 + linear_head / n_heads_q;
    if (q >= nq) {
        return;
    }

    const int q_scale_width = GGML_CUDA_Q8Q4_I8_D / q_scale_blocks;
    const char * q_row = Q + int64_t(q) * nb01 + int64_t(hq) * nb02 + int64_t(ib) * nb03;
    const int d = qb * q_scale_width + tid;
    const float x = tid < q_scale_width ? *(const float *) (q_row + int64_t(d) * nb00) : 0.0f;
    float amax = tid < q_scale_width ? fabsf(x) : 0.0f;
    for (int mask = q_scale_width / 2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, q_scale_width));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const size_t q_scale_offset = size_t(instance) * nq * q_scale_blocks;
    const size_t q_i8_offset = size_t(instance) * nq * GGML_CUDA_Q8Q4_I8_D;
    if (tid == 0) {
        q_scales[q_scale_offset + size_t(q) * q_scale_blocks + qb] = scale;
    }
    if (tid < q_scale_width) {
        q_i8[q_i8_offset + size_t(q) * GGML_CUDA_Q8Q4_I8_D + d] = ggml_cuda_q8q4_i8_clamp(x / scale);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_i8_splitk_stage1_kernel(
        const int8_t * __restrict__ q_i8,
        const float  * __restrict__ q_scales,
        const char   * __restrict__ K,
        const char   * __restrict__ V,
        const char   * __restrict__ mask,
        float        * __restrict__ partial_acc,
        float        * __restrict__ partial_m,
        float        * __restrict__ partial_l,
        int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
        int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
        int64_t nb30, int64_t nb31, int64_t nb33, int64_t ne33,
        int nq,
        int nk,
        int hq0,
        int ib0,
        int n_heads_q,
        int gqa_ratio,
        float softmax_scale,
        int q_scale_blocks) {
    __shared__ int8_t  q_tile[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K];
    __shared__ int8_t  k_tile[GGML_CUDA_Q8Q4_I8_WMMA_K * GGML_CUDA_Q8Q4_I8_WMMA_N];
    __shared__ int32_t partial[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N];
    __shared__ float   logits[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N];

    const int tid = threadIdx.x;
    const int q_base = blockIdx.x * GGML_CUDA_Q8Q4_I8_WMMA_M;
    const int chunk = blockIdx.y;
    const int instance = blockIdx.z;
    const int linear_head = hq0 + instance;
    const int hq = linear_head % n_heads_q;
    const int ib = ib0 + linear_head / n_heads_q;
    const int k_begin = chunk * GGML_CUDA_Q8Q4_I8_CHUNK;
    const int k_end = min(nk, k_begin + GGML_CUDA_Q8Q4_I8_CHUNK);
    const int d = tid;
    const int v_scale_block = d >> 5;
    const int hkv = hq / gqa_ratio;
    const size_t q_i8_offset = size_t(instance) * nq * GGML_CUDA_Q8Q4_I8_D;
    const size_t q_scale_offset = size_t(instance) * nq * q_scale_blocks;
    const size_t partial_base = size_t(instance) * size_t(gridDim.y) * nq;

    float row_max[GGML_CUDA_Q8Q4_I8_WMMA_M];
    float denom[GGML_CUDA_Q8Q4_I8_WMMA_M];
    float acc[GGML_CUDA_Q8Q4_I8_WMMA_M];
#pragma unroll
    for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
        row_max[r] = -3.402823466e+38F;
        denom[r] = 0.0f;
        acc[r] = 0.0f;
    }

    for (int k_base = k_begin; k_base < k_end; k_base += GGML_CUDA_Q8Q4_I8_WMMA_N) {
        const int lane_row = tid >> 4;
        const int lane_col = tid & 15;
        const int q_row = q_base + lane_row;
        const int k_row = k_base + lane_col;
        float logit = 0.0f;

        if (q_scale_blocks == GGML_CUDA_Q8Q4_I8_BLOCKS) {
#pragma unroll
            for (int kb = 0; kb < GGML_CUDA_Q8Q4_I8_BLOCKS; ++kb) {
                rocwmma::fragment<rocwmma::accumulator, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int32_t> acc_i32;
                if (tid < 32) {
                    rocwmma::fill_fragment(acc_i32, 0);
                }

#pragma unroll
                for (int inner = 0; inner < 2; ++inner) {
                    const int step = kb * 2 + inner;
                    if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K) {
                        const int d_q = step * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col;
                        q_tile[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col] = (q_row < nq) ? q_i8[q_i8_offset + size_t(q_row) * GGML_CUDA_Q8Q4_I8_D + d_q] : 0;

                        const int d_k = step * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_row;
                        const block_q8_0 * kb_ptr = nullptr;
                        if (k_row < nk) {
                            const char * k_row_ptr = K + int64_t(k_row) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                            kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(kb) * nb10);
                        }
                        k_tile[lane_row + lane_col * GGML_CUDA_Q8Q4_I8_WMMA_K] = (k_row < nk) ? kb_ptr->qs[d_k - kb * GGML_CUDA_Q8Q4_I8_QK] : 0;
                    }
                    __syncthreads();

                    if (tid < 32) {
                        rocwmma::fragment<rocwmma::matrix_a, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::row_major> q_frag;
                        rocwmma::fragment<rocwmma::matrix_b, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::col_major> k_frag;
                        rocwmma::load_matrix_sync(q_frag, q_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                        rocwmma::load_matrix_sync(k_frag, k_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                        rocwmma::mma_sync(acc_i32, q_frag, k_frag, acc_i32);
                    }
                    if (inner == 0) {
                        __syncthreads();
                    }
                }

                if (tid < 32) {
                    rocwmma::store_matrix_sync(partial, acc_i32, GGML_CUDA_Q8Q4_I8_WMMA_N, rocwmma::mem_row_major);
                }
                __syncthreads();

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N && q_row < nq && k_row < nk) {
                    const char * k_row_ptr = K + int64_t(k_row) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                    const block_q8_0 * kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(kb) * nb10);
                    const float qs = q_scales[q_scale_offset + size_t(q_row) * q_scale_blocks + kb];
                    const float ks = float(kb_ptr->d);
                    logit += float(partial[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_N + lane_col]) * qs * ks;
                }
                __syncthreads();
            }
        } else {
#pragma unroll
            for (int qsb = 0; qsb < GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS; ++qsb) {
                const int kb = qsb >> 1;
                rocwmma::fragment<rocwmma::accumulator, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int32_t> acc_i32;
                if (tid < 32) {
                    rocwmma::fill_fragment(acc_i32, 0);
                }

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K) {
                    const int d_q = qsb * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col;
                    q_tile[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col] = (q_row < nq) ? q_i8[q_i8_offset + size_t(q_row) * GGML_CUDA_Q8Q4_I8_D + d_q] : 0;

                    const int d_k = qsb * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_row;
                    const block_q8_0 * kb_ptr = nullptr;
                    if (k_row < nk) {
                        const char * k_row_ptr = K + int64_t(k_row) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                        kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(kb) * nb10);
                    }
                    k_tile[lane_row + lane_col * GGML_CUDA_Q8Q4_I8_WMMA_K] = (k_row < nk) ? kb_ptr->qs[d_k - kb * GGML_CUDA_Q8Q4_I8_QK] : 0;
                }
                __syncthreads();

                if (tid < 32) {
                    rocwmma::fragment<rocwmma::matrix_a, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::row_major> q_frag;
                    rocwmma::fragment<rocwmma::matrix_b, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::col_major> k_frag;
                    rocwmma::load_matrix_sync(q_frag, q_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                    rocwmma::load_matrix_sync(k_frag, k_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                    rocwmma::mma_sync(acc_i32, q_frag, k_frag, acc_i32);
                    rocwmma::store_matrix_sync(partial, acc_i32, GGML_CUDA_Q8Q4_I8_WMMA_N, rocwmma::mem_row_major);
                }
                __syncthreads();

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N && q_row < nq && k_row < nk) {
                    const char * k_row_ptr = K + int64_t(k_row) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                    const block_q8_0 * kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(kb) * nb10);
                    const float qs = q_scales[q_scale_offset + size_t(q_row) * q_scale_blocks + qsb];
                    const float ks = float(kb_ptr->d);
                    logit += float(partial[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_N + lane_col]) * qs * ks;
                }
                __syncthreads();
            }
        }

        if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N) {
            logits[tid] = logit;
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < GGML_CUDA_Q8Q4_I8_WMMA_N; ++c) {
            const int k_row = k_base + c;
            if (k_row >= k_end) {
                continue;
            }
            const char * v_row_ptr = V + int64_t(k_row) * nb21 + int64_t(hkv) * nb22 + int64_t(ib) * nb23;
            const block_q4_0 * vb = (const block_q4_0 *) (v_row_ptr + int64_t(v_scale_block) * nb20);
            const int iqs = d & 15;
            const int shift = (d & 31) >= 16 ? 4 : 0;
            const uint8_t packed = vb->qs[iqs];
            const int nibble = int((packed >> shift) & 0x0f);
            const float v = float(nibble - 8) * float(vb->d);
#pragma unroll
            for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
                const int q_row_r = q_base + r;
                if (q_row_r >= nq) {
                    continue;
                }
                const float x = logits[r * GGML_CUDA_Q8Q4_I8_WMMA_N + c] * softmax_scale +
                    ggml_cuda_q8q4_i8_mask_value(mask, nb30, nb31, nb33, ne33, q_row_r, k_row, ib);
                const float next_max = fmaxf(row_max[r], x);
                const float old_scale = expf(row_max[r] - next_max);
                const float p = expf(x - next_max);
                acc[r] = acc[r] * old_scale + p * v;
                denom[r] = denom[r] * old_scale + p;
                row_max[r] = next_max;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
        const int q_row = q_base + r;
        if (q_row < nq) {
            partial_acc[(partial_base + size_t(chunk) * nq + q_row) * GGML_CUDA_Q8Q4_I8_D + d] = acc[r];
            if (tid == r) {
                partial_m[partial_base + size_t(chunk) * nq + q_row] = row_max[r];
                partial_l[partial_base + size_t(chunk) * nq + q_row] = denom[r];
            }
        }
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_i8_reduce_kernel(
        const float * __restrict__ partial_acc,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        const float * __restrict__ sinks,
        char        * __restrict__ dst,
        int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
        int nq,
        int n_chunks,
        int hq0,
        int ib0,
        int n_heads_q) {
    const int d = threadIdx.x;
    const int q = blockIdx.x;
    const int instance = blockIdx.z;
    const int linear_head = hq0 + instance;
    const int hq = linear_head % n_heads_q;
    const int ib = ib0 + linear_head / n_heads_q;
    const size_t partial_base = size_t(instance) * n_chunks * nq;
    if (d >= GGML_CUDA_Q8Q4_I8_D || q >= nq) {
        return;
    }

    float row_max = -3.402823466e+38F;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float cm = partial_m[partial_base + size_t(c) * nq + q];
        const float cl = partial_l[partial_base + size_t(c) * nq + q];
        if (cl <= 0.0f) {
            continue;
        }
        const float ca = partial_acc[(partial_base + size_t(c) * nq + q) * GGML_CUDA_Q8Q4_I8_D + d];
        const float next_max = fmaxf(row_max, cm);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float chunk_scale = expf(cm - next_max);
        acc = acc * old_scale + ca * chunk_scale;
        denom = denom * old_scale + cl * chunk_scale;
        row_max = next_max;
    }
    if (sinks) {
        ggml_cuda_q8q4_i8_apply_attn_sink(sinks[hq], row_max, denom, acc);
    }
    // ggml_flash_attn_ext creates dst with logical shape [d_v, head_q, q, batch].
    // Keep output indexing in that order; using [d_v, q, head_q, batch] corrupts
    // multi-head backend-op comparisons even when the per-head math is correct.
    float * out = (float *) (dst + int64_t(d) * nb0 + int64_t(hq) * nb1 + int64_t(q) * nb2 + int64_t(ib) * nb3);
    *out = denom > 0.0f ? acc / denom : 0.0f;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_i8_flash_kernel(
        const int8_t * __restrict__ q_i8,
        const float  * __restrict__ q_scales,
        const char   * __restrict__ K,
        const char   * __restrict__ V,
        const char   * __restrict__ mask,
        const float  * __restrict__ sinks,
        char         * __restrict__ dst,
        int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
        int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
        int64_t nb30, int64_t nb31, int64_t nb33, int64_t ne33,
        int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
        int nq,
        int nk,
        int hq0,
        int ib0,
        int n_heads_q,
        int gqa_ratio,
        float softmax_scale,
        int q_scale_blocks) {
    __shared__ int8_t  q_hoist[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_D];
    __shared__ float   q_scale_hoist[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS];
    __shared__ int8_t  k_hoist[GGML_CUDA_Q8Q4_I8_WMMA_N * GGML_CUDA_Q8Q4_I8_D];
    __shared__ float   k_scale_hoist[GGML_CUDA_Q8Q4_I8_WMMA_N * GGML_CUDA_Q8Q4_I8_BLOCKS];
    __shared__ int8_t  q_tile[GGML_CUDA_Q8Q4_I8_BLOCKS * GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K];
    __shared__ int8_t  k_tile[GGML_CUDA_Q8Q4_I8_BLOCKS * GGML_CUDA_Q8Q4_I8_WMMA_K * GGML_CUDA_Q8Q4_I8_WMMA_N];
    __shared__ int32_t partial[GGML_CUDA_Q8Q4_I8_BLOCKS * GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N];
    // After QK, this tile is reused for masked logits and then for the
    // unnormalized softmax probabilities shared by all output-D threads.
    __shared__ float   logits[GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N];
    __shared__ float   row_max_s[GGML_CUDA_Q8Q4_I8_WMMA_M];
    __shared__ float   denom_s[GGML_CUDA_Q8Q4_I8_WMMA_M];
    __shared__ float   old_scale_s[GGML_CUDA_Q8Q4_I8_WMMA_M];

    const int tid = threadIdx.x;
    const int q_base = blockIdx.x * GGML_CUDA_Q8Q4_I8_WMMA_M;
    const int instance = blockIdx.z;
    const int linear_head = hq0 + instance;
    const int hq = linear_head % n_heads_q;
    const int ib = ib0 + linear_head / n_heads_q;
    const int d = tid;
    const int v_scale_block = d >> 5;
    const int hkv = hq / gqa_ratio;
    const size_t q_i8_offset = size_t(instance) * nq * GGML_CUDA_Q8Q4_I8_D;
    const size_t q_scale_offset = size_t(instance) * nq * q_scale_blocks;

    // Hoist the whole Q tile once. The previous flash prototype rebuilt q_tile
    // from global memory for every K tile and every 16-wide D step. Flash-style
    // attention should keep Q resident while streaming K/V.
    for (int off = tid; off < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_D; off += blockDim.x) {
        const int r = off / GGML_CUDA_Q8Q4_I8_D;
        const int dd = off - r * GGML_CUDA_Q8Q4_I8_D;
        const int q_row = q_base + r;
        q_hoist[off] = q_row < nq ? q_i8[q_i8_offset + size_t(q_row) * GGML_CUDA_Q8Q4_I8_D + dd] : 0;
    }
    for (int off = tid; off < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS; off += blockDim.x) {
        const int r = off / GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS;
        const int b = off - r * GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS;
        const int q_row = q_base + r;
        q_scale_hoist[off] = (q_row < nq && b < q_scale_blocks) ? q_scales[q_scale_offset + size_t(q_row) * q_scale_blocks + b] : 1.0f;
    }
    __syncthreads();

    if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M) {
        row_max_s[tid] = -3.402823466e+38F;
        denom_s[tid] = 0.0f;
        old_scale_s[tid] = 0.0f;
    }

    float acc[GGML_CUDA_Q8Q4_I8_WMMA_M];
#pragma unroll
    for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
        acc[r] = 0.0f;
    }
    __syncthreads();

    for (int k_base = 0; k_base < nk; k_base += GGML_CUDA_Q8Q4_I8_WMMA_N) {
        const int lane_row = tid >> 4;
        const int lane_col = tid & 15;
        const int q_row = q_base + lane_row;
        const int k_row = k_base + lane_col;

        // Hoist one K tile (16 rows x D) and its q8 scales. This keeps the
        // WMMA inner loop on LDS/register data instead of redoing row pointer
        // arithmetic and scattered global reads for every 16-wide D step.
        for (int off = tid; off < GGML_CUDA_Q8Q4_I8_WMMA_N * GGML_CUDA_Q8Q4_I8_D; off += blockDim.x) {
            const int c = off / GGML_CUDA_Q8Q4_I8_D;
            const int dd = off - c * GGML_CUDA_Q8Q4_I8_D;
            const int k_row_load = k_base + c;
            if (k_row_load < nk) {
                const char * k_row_ptr = K + int64_t(k_row_load) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                const block_q8_0 * kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(dd >> 5) * nb10);
                k_hoist[off] = kb_ptr->qs[dd & 31];
            } else {
                k_hoist[off] = 0;
            }
        }
        for (int off = tid; off < GGML_CUDA_Q8Q4_I8_WMMA_N * GGML_CUDA_Q8Q4_I8_BLOCKS; off += blockDim.x) {
            const int c = off / GGML_CUDA_Q8Q4_I8_BLOCKS;
            const int kb = off - c * GGML_CUDA_Q8Q4_I8_BLOCKS;
            const int k_row_load = k_base + c;
            if (k_row_load < nk) {
                const char * k_row_ptr = K + int64_t(k_row_load) * nb11 + int64_t(hkv) * nb12 + int64_t(ib) * nb13;
                const block_q8_0 * kb_ptr = (const block_q8_0 *) (k_row_ptr + int64_t(kb) * nb10);
                k_scale_hoist[off] = float(kb_ptr->d);
            } else {
                k_scale_hoist[off] = 1.0f;
            }
        }
        __syncthreads();

        float logit = 0.0f;
        if (q_scale_blocks == GGML_CUDA_Q8Q4_I8_BLOCKS) {
            // Keep the q8_0 scale-width-32 path as the known-good single-wave
            // WMMA sequence. A previous 8-wave-per-CTA variant improved QK
            // parallelism but regressed end-to-end prefill because the flash
            // prototype is dominated by the softmax/PV tail, not by QK.
#pragma unroll
            for (int kb = 0; kb < GGML_CUDA_Q8Q4_I8_BLOCKS; ++kb) {
                rocwmma::fragment<rocwmma::accumulator, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int32_t> acc_i32;
                if (tid < 32) {
                    rocwmma::fill_fragment(acc_i32, 0);
                }

#pragma unroll
                for (int inner = 0; inner < 2; ++inner) {
                    const int step = kb * 2 + inner;
                    if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K) {
                        const int d_q = step * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col;
                        q_tile[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col] = q_hoist[lane_row * GGML_CUDA_Q8Q4_I8_D + d_q];

                        const int d_k = step * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_row;
                        k_tile[lane_row + lane_col * GGML_CUDA_Q8Q4_I8_WMMA_K] = k_hoist[lane_col * GGML_CUDA_Q8Q4_I8_D + d_k];
                    }
                    __syncthreads();

                    if (tid < 32) {
                        rocwmma::fragment<rocwmma::matrix_a, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::row_major> q_frag;
                        rocwmma::fragment<rocwmma::matrix_b, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::col_major> k_frag;
                        rocwmma::load_matrix_sync(q_frag, q_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                        rocwmma::load_matrix_sync(k_frag, k_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                        rocwmma::mma_sync(acc_i32, q_frag, k_frag, acc_i32);
                    }
                    if (inner == 0) {
                        __syncthreads();
                    }
                }

                if (tid < 32) {
                    rocwmma::store_matrix_sync(partial, acc_i32, GGML_CUDA_Q8Q4_I8_WMMA_N, rocwmma::mem_row_major);
                }
                __syncthreads();

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N && q_row < nq && k_row < nk) {
                    const float qs = q_scale_hoist[lane_row * GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS + kb];
                    const float ks = k_scale_hoist[lane_col * GGML_CUDA_Q8Q4_I8_BLOCKS + kb];
                    logit += float(partial[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_N + lane_col]) * qs * ks;
                }
                __syncthreads();
            }
        } else {
#pragma unroll
            for (int qsb = 0; qsb < GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS; ++qsb) {
                const int kb = qsb >> 1;
                rocwmma::fragment<rocwmma::accumulator, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int32_t> acc_i32;
                if (tid < 32) {
                    rocwmma::fill_fragment(acc_i32, 0);
                }

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_K) {
                    const int d_q = qsb * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col;
                    q_tile[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_col] = q_hoist[lane_row * GGML_CUDA_Q8Q4_I8_D + d_q];

                    const int d_k = qsb * GGML_CUDA_Q8Q4_I8_WMMA_K + lane_row;
                    k_tile[lane_row + lane_col * GGML_CUDA_Q8Q4_I8_WMMA_K] = k_hoist[lane_col * GGML_CUDA_Q8Q4_I8_D + d_k];
                }
                __syncthreads();

                if (tid < 32) {
                    rocwmma::fragment<rocwmma::matrix_a, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::row_major> q_frag;
                    rocwmma::fragment<rocwmma::matrix_b, GGML_CUDA_Q8Q4_I8_WMMA_M, GGML_CUDA_Q8Q4_I8_WMMA_N, GGML_CUDA_Q8Q4_I8_WMMA_K, int8_t, rocwmma::col_major> k_frag;
                    rocwmma::load_matrix_sync(q_frag, q_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                    rocwmma::load_matrix_sync(k_frag, k_tile, GGML_CUDA_Q8Q4_I8_WMMA_K);
                    rocwmma::mma_sync(acc_i32, q_frag, k_frag, acc_i32);
                    rocwmma::store_matrix_sync(partial, acc_i32, GGML_CUDA_Q8Q4_I8_WMMA_N, rocwmma::mem_row_major);
                }
                __syncthreads();

                if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N && q_row < nq && k_row < nk) {
                    const float qs = q_scale_hoist[lane_row * GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS + qsb];
                    const float ks = k_scale_hoist[lane_col * GGML_CUDA_Q8Q4_I8_BLOCKS + kb];
                    logit += float(partial[lane_row * GGML_CUDA_Q8Q4_I8_WMMA_N + lane_col]) * qs * ks;
                }
                __syncthreads();
            }
        }

        if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M * GGML_CUDA_Q8Q4_I8_WMMA_N) {
            const int r = tid >> 4;
            const int c = tid & 15;
            const int q_row_r = q_base + r;
            const int k_row_c = k_base + c;
            logits[tid] = (q_row_r < nq && k_row_c < nk) ?
                logit * softmax_scale + ggml_cuda_q8q4_i8_mask_value(mask, nb30, nb31, nb33, ne33, q_row_r, k_row_c, ib) :
                -3.402823466e+38F;
        }
        __syncthreads();

        // Compute row max/denominator and the current tile probabilities once
        // per Q row, then share them across all 256 output-D threads. The first
        // flash prototype updated online softmax independently in every D
        // thread, paying 256x redundant expf/max work for identical row stats.
        if (tid < GGML_CUDA_Q8Q4_I8_WMMA_M) {
            const int r = tid;
            const int q_row_r = q_base + r;
            float tile_max = -3.402823466e+38F;
#pragma unroll
            for (int c = 0; c < GGML_CUDA_Q8Q4_I8_WMMA_N; ++c) {
                tile_max = fmaxf(tile_max, logits[r * GGML_CUDA_Q8Q4_I8_WMMA_N + c]);
            }

            const float next_max = fmaxf(row_max_s[r], tile_max);
            const float old_scale = denom_s[r] > 0.0f ? expf(row_max_s[r] - next_max) : 0.0f;
            float tile_sum = 0.0f;
#pragma unroll
            for (int c = 0; c < GGML_CUDA_Q8Q4_I8_WMMA_N; ++c) {
                const int k_row_c = k_base + c;
                const float p = (q_row_r < nq && k_row_c < nk) ? expf(logits[r * GGML_CUDA_Q8Q4_I8_WMMA_N + c] - next_max) : 0.0f;
                logits[r * GGML_CUDA_Q8Q4_I8_WMMA_N + c] = p;
                tile_sum += p;
            }
            old_scale_s[r] = old_scale;
            denom_s[r] = denom_s[r] * old_scale + tile_sum;
            row_max_s[r] = next_max;
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
            acc[r] *= old_scale_s[r];
        }
#pragma unroll
        for (int c = 0; c < GGML_CUDA_Q8Q4_I8_WMMA_N; ++c) {
            const int k_row_c = k_base + c;
            if (k_row_c >= nk) {
                continue;
            }
            const char * v_row_ptr = V + int64_t(k_row_c) * nb21 + int64_t(hkv) * nb22 + int64_t(ib) * nb23;
            const block_q4_0 * vb = (const block_q4_0 *) (v_row_ptr + int64_t(v_scale_block) * nb20);
            const int iqs = d & 15;
            const int shift = (d & 31) >= 16 ? 4 : 0;
            const uint8_t packed = vb->qs[iqs];
            const int nibble = int((packed >> shift) & 0x0f);
            const float v = float(nibble - 8) * float(vb->d);
#pragma unroll
            for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
                acc[r] += logits[r * GGML_CUDA_Q8Q4_I8_WMMA_N + c] * v;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < GGML_CUDA_Q8Q4_I8_WMMA_M; ++r) {
        const int q_row = q_base + r;
        if (q_row < nq) {
            float row_max_r = row_max_s[r];
            float denom_r = denom_s[r];
            float acc_r = acc[r];
            if (sinks) {
                ggml_cuda_q8q4_i8_apply_attn_sink(sinks[hq], row_max_r, denom_r, acc_r);
            }
            float * out = (float *) (dst + int64_t(d) * nb0 + int64_t(hq) * nb1 + int64_t(q_row) * nb2 + int64_t(ib) * nb3);
            *out = denom_r > 0.0f ? acc_r / denom_r : 0.0f;
        }
    }
}

inline void ggml_cuda_flash_attn_ext_q8q4_wmma_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = ggml_cuda_q8q4_wmma_i8_has_attn_sinks(dst) ? dst->src[4] : nullptr;

    GGML_ASSERT(mask != nullptr && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!sinks || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] >= Q->ne[2]));
    GGML_ASSERT(Q->ne[0] == GGML_CUDA_Q8Q4_I8_D && K->ne[0] == GGML_CUDA_Q8Q4_I8_D && V->ne[0] == GGML_CUDA_Q8Q4_I8_D);
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int nq = Q->ne[1];
    const int nk = K->ne[1];
    const int n_heads_q = Q->ne[2];
    const int n_batch = Q->ne[3];
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int n_chunks = (nk + GGML_CUDA_Q8Q4_I8_CHUNK - 1) / GGML_CUDA_Q8Q4_I8_CHUNK;
    const bool flash_prefill = ggml_cuda_q8q4_wmma_i8_flash_prefill_enabled() && nq >= ggml_cuda_q8q4_wmma_i8_flash_prefill_min_nq();
    const int alloc_chunks = flash_prefill ? 0 : (int) ggml_cuda_q8q4_wmma_i8_scratch_alloc_chunks(n_chunks);
    const int q_scale_blocks = ggml_cuda_q8q4_wmma_i8_qscale16_enabled() ? GGML_CUDA_Q8Q4_I8_QSCALE16_BLOCKS : GGML_CUDA_Q8Q4_I8_BLOCKS;

    const int n_instances_total = n_heads_q * n_batch;
    const int64_t scratch_per_head = flash_prefill ?
        (nq * GGML_CUDA_Q8Q4_I8_D * (int64_t) sizeof(int8_t) + nq * q_scale_blocks * (int64_t) sizeof(float)) :
        ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head_chunks(dst, alloc_chunks);
    const int64_t exact_scratch_per_head = flash_prefill ? scratch_per_head : ggml_cuda_q8q4_wmma_i8_scratch_bytes_per_head_chunks(dst, n_chunks);
    const int64_t max_mib = ggml_cuda_q8q4_wmma_i8_max_mib();
    const int64_t batch_scratch = scratch_per_head * int64_t(n_instances_total);
    const bool batch_heads = max_mib <= 0 || batch_scratch <= max_mib * 1024LL * 1024LL;
    const int active_instances = batch_heads ? n_instances_total : 1;
    const float * sinks_data = sinks ? (const float *) sinks->data : nullptr;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int8_t> q_i8(pool, size_t(active_instances) * nq * GGML_CUDA_Q8Q4_I8_D);
    ggml_cuda_pool_alloc<float> q_scales(pool, size_t(active_instances) * nq * q_scale_blocks);
    ggml_cuda_pool_alloc<float> partial_acc(pool, size_t(active_instances) * alloc_chunks * nq * GGML_CUDA_Q8Q4_I8_D);
    ggml_cuda_pool_alloc<float> partial_m(pool, size_t(active_instances) * alloc_chunks * nq);
    ggml_cuda_pool_alloc<float> partial_l(pool, size_t(active_instances) * alloc_chunks * nq);

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double exact_scratch_mib = double(exact_scratch_per_head) / (1024.0 * 1024.0);
        const double alloc_scratch_mib = double(scratch_per_head) / (1024.0 * 1024.0);
        const double active_scratch_mib = double(scratch_per_head * int64_t(active_instances)) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=q8q4_wmma_i8%s nq=%d nk=%d heads=%d batch=%d gqa=%d chunks=%d alloc_chunks=%d q_scale_width=%d exact_scratch_per_head=%.3f MiB alloc_scratch_per_head=%.3f MiB active_scratch=%.3f MiB batch_heads=%d mask=1 sinks=%d\n",
            __func__, flash_prefill ? "_flash" : "", nq, nk, n_heads_q, n_batch, gqa_ratio, n_chunks, alloc_chunks, GGML_CUDA_Q8Q4_I8_D / q_scale_blocks, exact_scratch_mib, alloc_scratch_mib, active_scratch_mib, batch_heads, sinks != nullptr);
    }

    const dim3 quant_block(32);
    const dim3 block(256);
    cudaStream_t stream = ctx.stream();

    if (batch_heads) {
        const dim3 quant_grid(nq, q_scale_blocks, active_instances);
        const dim3 stage_grid((nq + GGML_CUDA_Q8Q4_I8_WMMA_M - 1) / GGML_CUDA_Q8Q4_I8_WMMA_M, n_chunks, active_instances);
        const dim3 reduce_grid(nq, 1, active_instances);
        ggml_cuda_q8q4_i8_quant_q_kernel<<<quant_grid, quant_block, 0, stream>>>(
            (const char *) Q->data, q_i8.ptr, q_scales.ptr,
            Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3], nq, 0, 0, n_heads_q, q_scale_blocks);
        if (flash_prefill) {
            const dim3 flash_grid((nq + GGML_CUDA_Q8Q4_I8_WMMA_M - 1) / GGML_CUDA_Q8Q4_I8_WMMA_M, 1, active_instances);
            ggml_cuda_q8q4_i8_flash_kernel<<<flash_grid, block, 0, stream>>>(
                q_i8.ptr, q_scales.ptr, (const char *) K->data, (const char *) V->data, (const char *) mask->data, sinks_data, (char *) dst->data,
                K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                nq, nk, 0, 0, n_heads_q, gqa_ratio, scale, q_scale_blocks);
        } else {
            ggml_cuda_q8q4_i8_splitk_stage1_kernel<<<stage_grid, block, 0, stream>>>(
                q_i8.ptr, q_scales.ptr, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
                partial_acc.ptr, partial_m.ptr, partial_l.ptr,
                K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3],
                nq, nk, 0, 0, n_heads_q, gqa_ratio, scale, q_scale_blocks);
            ggml_cuda_q8q4_i8_reduce_kernel<<<reduce_grid, block, 0, stream>>>(
                partial_acc.ptr, partial_m.ptr, partial_l.ptr, sinks_data, (char *) dst->data,
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], nq, n_chunks, 0, 0, n_heads_q);
        }
    } else {
        const dim3 quant_grid(nq, q_scale_blocks, 1);
        const dim3 stage_grid((nq + GGML_CUDA_Q8Q4_I8_WMMA_M - 1) / GGML_CUDA_Q8Q4_I8_WMMA_M, n_chunks, 1);
        const dim3 reduce_grid(nq, 1, 1);
        for (int ib = 0; ib < n_batch; ++ib) {
            for (int hq = 0; hq < n_heads_q; ++hq) {
                ggml_cuda_q8q4_i8_quant_q_kernel<<<quant_grid, quant_block, 0, stream>>>(
                    (const char *) Q->data, q_i8.ptr, q_scales.ptr,
                    Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3], nq, hq, ib, n_heads_q, q_scale_blocks);
                if (flash_prefill) {
                    const dim3 flash_grid((nq + GGML_CUDA_Q8Q4_I8_WMMA_M - 1) / GGML_CUDA_Q8Q4_I8_WMMA_M, 1, 1);
                    ggml_cuda_q8q4_i8_flash_kernel<<<flash_grid, block, 0, stream>>>(
                        q_i8.ptr, q_scales.ptr, (const char *) K->data, (const char *) V->data, (const char *) mask->data, sinks_data, (char *) dst->data,
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                        V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                        mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3],
                        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                        nq, nk, hq, ib, n_heads_q, gqa_ratio, scale, q_scale_blocks);
                } else {
                    ggml_cuda_q8q4_i8_splitk_stage1_kernel<<<stage_grid, block, 0, stream>>>(
                        q_i8.ptr, q_scales.ptr, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
                        partial_acc.ptr, partial_m.ptr, partial_l.ptr,
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                        V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                        mask->nb[0], mask->nb[1], mask->nb[3], mask->ne[3],
                        nq, nk, hq, ib, n_heads_q, gqa_ratio, scale, q_scale_blocks);
                    ggml_cuda_q8q4_i8_reduce_kernel<<<reduce_grid, block, 0, stream>>>(
                        partial_acc.ptr, partial_m.ptr, partial_l.ptr, sinks_data, (char *) dst->data,
                        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], nq, n_chunks, hq, ib, n_heads_q);
                }
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
#else
inline void ggml_cuda_flash_attn_ext_q8q4_wmma_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("q8q4_wmma_i8 requires HIP");
}
#endif
