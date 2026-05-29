#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cstdlib>

static inline bool ggml_cuda_dot4_prefill_unsafe_enabled() {
#ifdef GGML_USE_HIP
    // Default-enabled. Explicitly disabled by EXPERIMENTAL_UNSAFE=0.
    const char * v = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
    if (!v) v = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
    return !(v && atoi(v) == 0);
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_dot4_prefill_enabled() {
#ifdef GGML_USE_HIP
    // Default-enabled. Explicitly disabled by EXPERIMENTAL_UNSAFE=0 or Q8Q4_DOT4_PREFILL=0.
    const char * v = getenv("GGML_CUDA_ROCM_Q8Q4_DOT4_PREFILL");
    return ggml_cuda_dot4_prefill_unsafe_enabled() && !(v && atoi(v) == 0);
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8q4_dot4_prefill_probe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_DOT4_PREFILL_PROBE_ONLY");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline int64_t ggml_cuda_q8q4_dot4_prefill_min_nq() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_DOT4_PREFILL_MIN_NQ");
    return env ? atoll(env) : 0;
#else
    return 0;
#endif
}

static inline bool ggml_cuda_q8q4_dot4_prefill_supported(const int cc, const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!ggml_cuda_q8q4_dot4_prefill_enabled()) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || dst->ne[0] != 256 || Q->ne[1] <= 2) {
        return false;
    }
    if (Q->ne[1] < ggml_cuda_q8q4_dot4_prefill_min_nq()) {
        return false;
    }
    if (K->ne[1] < Q->ne[1] || V->ne[1] != K->ne[1] || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }
    if (dst->ne[0] != V->ne[0] || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->ne[3] != Q->ne[3]) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // Runtime path supports either full-attention (mask=null) diagnostics or the
    // normal F16 causal mask layout used by server prefill. Keep all other mask
    // layouts on the existing TILE/F16/WMMA paths until separately validated.
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
#endif
}

static inline bool ggml_cuda_q8tbq4_dot4_prefill_enabled() {
#ifdef GGML_USE_HIP
    // Default-enabled. Explicitly disabled by EXPERIMENTAL_UNSAFE=0 or Q8TBQ4_DOT4_PREFILL=0.
    const char * v = getenv("GGML_CUDA_ROCM_Q8TBQ4_DOT4_PREFILL");
    return ggml_cuda_dot4_prefill_unsafe_enabled() && !(v && atoi(v) == 0);
#else
    return false;
#endif
}

static inline bool ggml_cuda_q8tbq4_dot4_prefill_probe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8TBQ4_DOT4_PREFILL_PROBE_ONLY");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline int64_t ggml_cuda_q8tbq4_dot4_prefill_min_nq() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8TBQ4_DOT4_PREFILL_MIN_NQ");
    return env ? atoll(env) : 1024;
#else
    return 0;
#endif
}

static inline bool ggml_cuda_q8tbq4_dot4_prefill_supported(const int cc, const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!ggml_cuda_q8tbq4_dot4_prefill_enabled()) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_TBQ4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || dst->ne[0] != 256 || Q->ne[1] <= 2) {
        return false;
    }
    if (Q->ne[1] < ggml_cuda_q8tbq4_dot4_prefill_min_nq()) {
        return false;
    }
    if (K->ne[1] < Q->ne[1] || V->ne[1] != K->ne[1] || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }
    if (dst->ne[0] != V->ne[0] || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->ne[3] != Q->ne[3]) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
#endif
}

static inline bool ggml_cuda_tbq4_dot4_prefill_enabled() {
#ifdef GGML_USE_HIP
    // Default-enabled. Explicitly disabled by EXPERIMENTAL_UNSAFE=0 or TBQ4_DOT4_PREFILL=0.
    const char * v = getenv("GGML_CUDA_ROCM_TBQ4_DOT4_PREFILL");
    return ggml_cuda_dot4_prefill_unsafe_enabled() && !(v && atoi(v) == 0);
#else
    return false;
#endif
}

static inline bool ggml_cuda_tbq4_dot4_prefill_probe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_TBQ4_DOT4_PREFILL_PROBE_ONLY");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static inline int64_t ggml_cuda_tbq4_dot4_prefill_min_nq() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_TBQ4_DOT4_PREFILL_MIN_NQ");
    return env ? atoll(env) : 1024;
#else
    return 0;
#endif
}

static inline bool ggml_cuda_tbq4_dot4_prefill_supported(const int cc, const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!ggml_cuda_tbq4_dot4_prefill_enabled()) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_TBQ4_0 || V->type != GGML_TYPE_TBQ4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || dst->ne[0] != 256 || Q->ne[1] <= 2) {
        return false;
    }
    if (Q->ne[1] < ggml_cuda_tbq4_dot4_prefill_min_nq()) {
        return false;
    }
    if (K->ne[1] < Q->ne[1] || V->ne[1] != K->ne[1] || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }
    if (dst->ne[0] != V->ne[0] || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->ne[3] != Q->ne[3]) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
#endif
}

#ifdef GGML_USE_HIP
static constexpr int GGML_CUDA_Q8Q4_DOT4_D = 256;
static constexpr int GGML_CUDA_Q8Q4_DOT4_QK8 = 32;
static constexpr int GGML_CUDA_Q8Q4_DOT4_BLOCKS = GGML_CUDA_Q8Q4_DOT4_D / GGML_CUDA_Q8Q4_DOT4_QK8;
static constexpr int GGML_CUDA_Q8Q4_DOT4_Q8BLOCK_BYTES = int(sizeof(block_q8_0));
static constexpr int GGML_CUDA_Q8Q4_DOT4_TILE_M = 16;
static constexpr int GGML_CUDA_Q8Q4_DOT4_TILE_N = 16;
static constexpr int GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK = GGML_CUDA_Q8Q4_DOT4_QK8 / 4;

static __device__ __forceinline__ int ggml_cuda_q8q4_dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return ggml_cuda_dp4a(a, b, c);
#endif
}

static __device__ __forceinline__ int ggml_cuda_q8q4_dot4_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_q8q4_dot4_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_dot4_quant_q_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ q_i32,
        float       * __restrict__ q_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nq,
        int n_heads_q,
        int ib) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (q >= nq || hq >= n_heads_q || q_block >= GGML_CUDA_Q8Q4_DOT4_BLOCKS) {
        return;
    }

    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(ib + q) * nb01);
    const int d = q_block * GGML_CUDA_Q8Q4_DOT4_QK8 + lane;
    const float x = q_ptr[d];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t base = (size_t(b) * n_heads_q + hq) * (size_t)nq + q;
    ((int8_t *)(q_i32 + base * (GGML_CUDA_Q8Q4_DOT4_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        q_scales[base * GGML_CUDA_Q8Q4_DOT4_BLOCKS + q_block] = scale;
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_dot4_pack_k_kernel(
        const char * __restrict__ K,
        int        * __restrict__ k_i32,
        half       * __restrict__ k_scales,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nk,
        int n_heads_k,
        int batch,
        int ib) {
    const int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    if (linear >= total) {
        return;
    }

    const int qblk = linear % GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const int t = linear / GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const int k = t % nk;
    const int hk_b = t / nk;
    const int hk = hk_b % n_heads_k;
    const int b = hk_b / n_heads_k;
    const char * src = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(ib + k) * nb11 + int64_t(qblk) * nb10;
    int * dst = k_i32 + (size_t(t) * (GGML_CUDA_Q8Q4_DOT4_D / 4) + qblk * GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK);

    k_scales[size_t(t) * GGML_CUDA_Q8Q4_DOT4_BLOCKS + qblk] = ggml_cuda_q8q4_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK; ++i) {
        dst[i] = ggml_cuda_q8q4_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __device__ __forceinline__ float ggml_cuda_q8q4_dot4_kq_dot(
        const int   * __restrict__ q_row_i32,
        const float * __restrict__ q_row_scales,
        const int   * __restrict__ k_row_i32,
        const half  * __restrict__ k_row_scales) {
    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8Q4_DOT4_BLOCKS; ++qb) {
        const int4 * q4 = (const int4 *) (q_row_i32 + qb * GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK);
        const int4 * k4 = (const int4 *) (k_row_i32 + qb * GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK);
        const int4 q0 = q4[0];
        const int4 q1 = q4[1];
        const int4 k0 = k4[0];
        const int4 k1 = k4[1];
        int acc0 = 0;
        acc0 = ggml_cuda_q8q4_dot4_i8_i8(q0.x, k0.x, acc0);
        acc0 = ggml_cuda_q8q4_dot4_i8_i8(q0.y, k0.y, acc0);
        acc0 = ggml_cuda_q8q4_dot4_i8_i8(q0.z, k0.z, acc0);
        acc0 = ggml_cuda_q8q4_dot4_i8_i8(q0.w, k0.w, acc0);
        int acc1 = 0;
        acc1 = ggml_cuda_q8q4_dot4_i8_i8(q1.x, k1.x, acc1);
        acc1 = ggml_cuda_q8q4_dot4_i8_i8(q1.y, k1.y, acc1);
        acc1 = ggml_cuda_q8q4_dot4_i8_i8(q1.z, k1.z, acc1);
        acc1 = ggml_cuda_q8q4_dot4_i8_i8(q1.w, k1.w, acc1);
        sum += float(acc0 + acc1) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
    }
    return sum;
}

static __device__ __forceinline__ float ggml_cuda_q8q4_dot4_kq_dot_block(
        const int   * __restrict__ q_row_i32,
        const float * __restrict__ q_row_scales,
        const int   * __restrict__ k_row_i32,
        const half  * __restrict__ k_row_scales,
        float       * __restrict__ kq_sums,
        int buf) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    float partial = 0.0f;

    if (tid < 32) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = lane + i * 32;
            const int qb = idx / GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK;
            const int acc = ggml_cuda_q8q4_dot4_i8_i8(q_row_i32[idx], k_row_i32[idx], 0);
            partial += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
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

static __device__ __forceinline__ float ggml_cuda_q8q4_dot4_dequant_q4_0(
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

static __device__ __forceinline__ float ggml_cuda_tbq4_dot4_dequant(
        const char * __restrict__ X,
        int64_t nb0,
        int i) {
    const int ib = i / QK_TBQ4;
    const int j  = i % QK_TBQ4;
    const block_tbq4_0 * x = (const block_tbq4_0 *) (X + int64_t(ib) * nb0);
    const uint8_t packed = x->qs[j / 2];
    const uint8_t idx = (j & 1) ? (packed >> 4) : (packed & 0x0f);
    return d_tbq4_centroids[idx] * __half2float(x->d);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_tbq4_dot4_quant_q_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ q_i32,
        float       * __restrict__ q_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nq,
        int n_heads_q,
        int ib) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 7;
    const int lane = tid & 127;
    if (q >= nq || hq >= n_heads_q || q_block >= GGML_CUDA_Q8Q4_DOT4_D / QK_TBQ4) {
        return;
    }

    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(ib + q) * nb01);
    const int d = q_block * QK_TBQ4 + lane;
    const float x = q_ptr[d];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    amax = fmaxf(amax, __shfl(amax, lane ^ 32, 32));
    amax = fmaxf(amax, __shfl(amax, lane ^ 64, 32));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t base = (size_t(b) * n_heads_q + hq) * (size_t)nq + q;
    ((int8_t *)(q_i32 + base * (GGML_CUDA_Q8Q4_DOT4_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        q_scales[base * (GGML_CUDA_Q8Q4_DOT4_D / QK_TBQ4) + q_block] = scale;
    }
}

static __device__ __forceinline__ float ggml_cuda_tbq4_dot4_kq_dot(
        const int   * __restrict__ q_row_i32,
        const float * __restrict__ q_row_scales,
        const char  * __restrict__ k_row) {
    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8Q4_DOT4_BLOCKS; ++qb) {
        const int * q_blk_i32 = q_row_i32 + qb * GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK;
        const float q_scale = q_row_scales[qb];
#pragma unroll
        for (int d4 = 0; d4 < GGML_CUDA_Q8Q4_DOT4_DOT4_PER_BLOCK; ++d4) {
            const int q_packed = q_blk_i32[d4];
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int8_t q = ((const int8_t *) &q_packed)[lane];
                const float k = ggml_cuda_tbq4_dot4_dequant(k_row, sizeof(block_tbq4_0), qb * GGML_CUDA_Q8Q4_DOT4_QK8 + d4 * 4 + lane);
                sum += float(q) * q_scale * k;
            }
        }
    }
    return sum;
}

static __device__ __forceinline__ float ggml_cuda_q8q4_dot4_mask_value(
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

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_dot4_kq_probe_kernel(
        const int   * __restrict__ q_i32,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_i32,
        const half  * __restrict__ k_scales,
        float       * __restrict__ logits,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8Q4_DOT4_TILE_M + (tid >> 4);
    const int k_row = blockIdx.x * GGML_CUDA_Q8Q4_DOT4_TILE_N + (tid & 15);
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k_row);
    const int * q_row_i32 = q_i32 + q_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
    const int * k_row_i32 = k_i32 + k_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;

    logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] =
        ggml_cuda_q8q4_dot4_kq_dot(q_row_i32, q_row_scales, k_row_i32, k_row_scales);
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8q4_dot4_fattn_kernel(
        const int   * __restrict__ q_i32,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_i32,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float2      * __restrict__ dst_meta,
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
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const int dst_d = tid;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_i32 = q_i32 + q_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    __shared__ float kq_sums[2];

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
        const int * k_row_i32 = k_i32 + k_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
        const float s = ggml_cuda_q8q4_dot4_kq_dot_block(q_row_i32, q_row_scales, k_row_i32, k_row_scales, kq_sums, k & 1) * scale +
            ggml_cuda_q8q4_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_q8q4_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dst_d);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8Q4_DOT4_D + dst_d] = out / denom;
    if (dst_meta && tid == 0) {
        dst_meta[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq)] = make_float2(row_max, denom);
    }
    GGML_UNUSED(n_heads_k);
}

inline void ggml_cuda_flash_attn_ext_q8q4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_i32(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);
    ggml_cuda_pool_alloc<int>   k_i32(pool);
    ggml_cuda_pool_alloc<half>  k_scales(pool);
    ggml_cuda_pool_alloc<float> logits(pool);

    q_i32.alloc((size_t) batch * n_heads_q * nq * (GGML_CUDA_Q8Q4_DOT4_D / 4));
    q_scales.alloc((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_BLOCKS);
    k_i32.alloc((size_t) batch * n_heads_k * nk * (GGML_CUDA_Q8Q4_DOT4_D / 4));
    k_scales.alloc((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_BLOCKS);
    if (ggml_cuda_q8q4_dot4_prefill_probe_enabled()) {
        logits.alloc((size_t) batch * n_heads_q * nq * nk);
    }

    cudaStream_t stream = ctx.stream();
    dim3 q_grid(nq, n_heads_q, batch);
    dim3 q_block(256);
    ggml_cuda_q8q4_dot4_quant_q_kernel<<<q_grid, q_block, 0, stream>>>(
        (const float *) Q->data, q_i32.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, 0);
    CUDA_CHECK(cudaGetLastError());

    dim3 pack_grid(((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_BLOCKS + 255) / 256);
    ggml_cuda_q8q4_dot4_pack_k_kernel<<<pack_grid, q_block, 0, stream>>>(
        (const char *) K->data, k_i32.ptr, k_scales.ptr,
        K->nb[0], K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch, 0);
    CUDA_CHECK(cudaGetLastError());

    if (ggml_cuda_q8q4_dot4_prefill_probe_enabled()) {
        dim3 kq_grid((nk + GGML_CUDA_Q8Q4_DOT4_TILE_N - 1) / GGML_CUDA_Q8Q4_DOT4_TILE_N,
                     (nq + GGML_CUDA_Q8Q4_DOT4_TILE_M - 1) / GGML_CUDA_Q8Q4_DOT4_TILE_M,
                     n_heads_q * batch);
        ggml_cuda_q8q4_dot4_kq_probe_kernel<<<kq_grid, q_block, 0, stream>>>(
            q_i32.ptr, q_scales.ptr, k_i32.ptr, k_scales.ptr, logits.ptr,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
    } else {
        float scale = 1.0f;
        memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
        dim3 fa_grid(nq, n_heads_q, batch);
        ggml_cuda_q8q4_dot4_fattn_kernel<<<fa_grid, q_block, 0, stream>>>(
            q_i32.ptr, q_scales.ptr, k_i32.ptr, k_scales.ptr,
            (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, nullptr, scale,
            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8Q4_DOT4_PREFILL_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_D) / (1024.0 * 1024.0);
        const double k_mib = double((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_D) / (1024.0 * 1024.0);
        const double logits_mib = ggml_cuda_q8q4_dot4_prefill_probe_enabled() ? double((size_t) batch * n_heads_q * nq * nk * sizeof(float)) / (1024.0 * 1024.0) : 0.0;
        GGML_LOG_INFO("%s: route=q8q4_dot4_prefill%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d mask=%d q_payload=%.3fMiB k_payload=%.3fMiB logits=%.3fMiB\n",
            __func__, ggml_cuda_q8q4_dot4_prefill_probe_enabled() ? "_probe" : "", nq, nk, n_heads_q, n_heads_k, batch, mask ? 1 : 0, q_mib, k_mib, logits_mib);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8tbq4_dot4_fattn_kernel(
        const int   * __restrict__ q_i32,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_i32,
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
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const int dst_d = tid;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_i32 = q_i32 + q_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    __shared__ float kq_sums[2];

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
        const int * k_row_i32 = k_i32 + k_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
        const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
        const float s = ggml_cuda_q8q4_dot4_kq_dot_block(q_row_i32, q_row_scales, k_row_i32, k_row_scales, kq_sums, k & 1) * scale +
            ggml_cuda_q8q4_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_tbq4_dot4_dequant(v_head + int64_t(k) * nb21, nb20, dst_d);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8Q4_DOT4_D + dst_d] = out / denom;
    GGML_UNUSED(n_heads_k);
}

inline void ggml_cuda_flash_attn_ext_q8tbq4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_i32(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);
    ggml_cuda_pool_alloc<int>   k_i32(pool);
    ggml_cuda_pool_alloc<half>  k_scales(pool);

    q_i32.alloc((size_t) batch * n_heads_q * nq * (GGML_CUDA_Q8Q4_DOT4_D / 4));
    q_scales.alloc((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_BLOCKS);
    k_i32.alloc((size_t) batch * n_heads_k * nk * (GGML_CUDA_Q8Q4_DOT4_D / 4));
    k_scales.alloc((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_BLOCKS);

    cudaStream_t stream = ctx.stream();
    dim3 q_grid(nq, n_heads_q, batch);
    dim3 q_block(256);
    ggml_cuda_q8q4_dot4_quant_q_kernel<<<q_grid, q_block, 0, stream>>>(
        (const float *) Q->data, q_i32.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, 0);
    CUDA_CHECK(cudaGetLastError());

    dim3 pack_grid(((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_BLOCKS + 255) / 256);
    ggml_cuda_q8q4_dot4_pack_k_kernel<<<pack_grid, q_block, 0, stream>>>(
        (const char *) K->data, k_i32.ptr, k_scales.ptr,
        K->nb[0], K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch, 0);
    CUDA_CHECK(cudaGetLastError());

    if (!ggml_cuda_q8tbq4_dot4_prefill_probe_enabled()) {
        float scale = 1.0f;
        memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
        dim3 fa_grid(nq, n_heads_q, batch);
        ggml_cuda_q8tbq4_dot4_fattn_kernel<<<fa_grid, q_block, 0, stream>>>(
            q_i32.ptr, q_scales.ptr, k_i32.ptr, k_scales.ptr,
            (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
        const int64_t nrows = Q->ne[1] * Q->ne[2] * Q->ne[3];
        tbq4_rotate_output_cuda((float *) dst->data, nrows, V->ne[0], stream);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8TBQ4_DOT4_PREFILL_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_D) / (1024.0 * 1024.0);
        const double k_mib = double((size_t) batch * n_heads_k * nk * GGML_CUDA_Q8Q4_DOT4_D) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=q8tbq4_dot4_prefill%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d mask=%d q_payload=%.3fMiB k_payload=%.3fMiB v_native=1\n",
            __func__, ggml_cuda_q8tbq4_dot4_prefill_probe_enabled() ? "_probe" : "", nq, nk, n_heads_q, n_heads_k, batch, mask ? 1 : 0, q_mib, k_mib);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_tbq4_dot4_fattn_kernel(
        const int   * __restrict__ q_i32,
        const float * __restrict__ q_scales,
        const char  * __restrict__ K,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
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
    const int dst_d = tid;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_i32 = q_i32 + q_base * (GGML_CUDA_Q8Q4_DOT4_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8Q4_DOT4_BLOCKS;
    const char * k_head = K + int64_t(b) * nb13 + int64_t(hk) * nb12;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const char * k_row_ptr = k_head + int64_t(k) * nb11;
        const float s = ggml_cuda_tbq4_dot4_kq_dot(q_row_i32, q_row_scales, k_row_ptr) * scale +
            ggml_cuda_q8q4_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_tbq4_dot4_dequant(v_head + int64_t(k) * nb21, nb20, dst_d);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8Q4_DOT4_D + dst_d] = out / denom;
    GGML_UNUSED(nb10); GGML_UNUSED(n_heads_k);
}

inline void ggml_cuda_flash_attn_ext_tbq4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_i32(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);

    q_i32.alloc((size_t) batch * n_heads_q * nq * (GGML_CUDA_Q8Q4_DOT4_D / 4));
    q_scales.alloc((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_BLOCKS);

    cudaStream_t stream = ctx.stream();
    const int64_t nrows = Q->ne[1] * Q->ne[2] * Q->ne[3];
    tbq4_rotate_input_cuda((float *) Q->data, nrows, Q->ne[0], stream);

    dim3 q_grid(nq, n_heads_q, batch);
    dim3 q_block(256);
    ggml_cuda_q8q4_dot4_quant_q_kernel<<<q_grid, q_block, 0, stream>>>(
        (const float *) Q->data, q_i32.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, 0);
    CUDA_CHECK(cudaGetLastError());

    if (!ggml_cuda_tbq4_dot4_prefill_probe_enabled()) {
        float scale = 1.0f;
        memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
        dim3 fa_grid(nq, n_heads_q, batch);
        ggml_cuda_tbq4_dot4_fattn_kernel<<<fa_grid, q_block, 0, stream>>>(
            q_i32.ptr, q_scales.ptr,
            (const char *) K->data, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale,
            K->nb[0], K->nb[1], K->nb[2], K->nb[3],
            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
        tbq4_rotate_output_cuda((float *) dst->data, nrows, V->ne[0], stream);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_TBQ4_DOT4_PREFILL_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double((size_t) batch * n_heads_q * nq * GGML_CUDA_Q8Q4_DOT4_D) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=tbq4_dot4_prefill%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d mask=%d q_payload=%.3fMiB k_native=1 v_native=1\n",
            __func__, ggml_cuda_tbq4_dot4_prefill_probe_enabled() ? "_probe" : "", nq, nk, n_heads_q, n_heads_k, batch, mask ? 1 : 0, q_mib);
    }
}
#else
inline void ggml_cuda_flash_attn_ext_q8q4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("q8q4_dot4_prefill requires HIP");
}
inline void ggml_cuda_flash_attn_ext_q8tbq4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("q8tbq4_dot4_prefill requires HIP");
}
inline void ggml_cuda_flash_attn_ext_tbq4_dot4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("tbq4_dot4_prefill requires HIP");
}
#endif
