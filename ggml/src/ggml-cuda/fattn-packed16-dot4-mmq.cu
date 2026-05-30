// ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cu
//
// DOT4-MMQ packed16 FlashAttention backend.
//
// Design:
//   - Q tile is quantized to q8_0-like signed int8 blocks.
//   - K cache stays packed int8 words, 4 x int8 per int32.
//   - QK uses dot4 / sudot4 / dp4a style packed integer dot.
//   - Accumulates int32 per 32-d block, then applies q_scale * k_scale.
//   - Then normal FlashAttention:
//         causal/additive mask
//         online softmax
//         P @ V
//         final dst write
//
// Initial production-shaped kernel:
//   - head_dim = 256 only.
//   - BM = 4 Q rows per CTA.
//   - BN = 64 K/V tokens per tile.
//   - 256 threads per CTA.
//   - One CTA handles {batch, q_head, q_tile}.
//   - Supports GQA by mapping q_head -> kv_head through gqa_ratio.
//   - Supports V as f16, f32, or q8 packed32 with per-32 scale.
//   - Supports dst as f16 or f32.
//   - Supports optional additive float mask.
//   - Supports causal future-tile skip.
//
// This file is intentionally self-contained. In the tree, you should later
// replace the load/store helpers with shared packed16 helpers from:
//   fattn-common.cuh
//   fattn-compressed-kv.cuh
//
// Do not wire this as a fallback for the scalar DOT4-Q8K probe.
// This is the sibling backend to packed16 WMMA.

#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define DOT4_MMQ_BACKEND_HIP 1
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define DOT4_MMQ_BACKEND_HIP 0
#endif

#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#ifndef DOT4_MMQ_USE_SCALAR_DOT4
#define DOT4_MMQ_USE_SCALAR_DOT4 0
#endif

#ifndef DOT4_MMQ_AMDGPU_SUDOT4_A_SIGNED
#define DOT4_MMQ_AMDGPU_SUDOT4_A_SIGNED true
#endif

#ifndef DOT4_MMQ_AMDGPU_SUDOT4_B_SIGNED
#define DOT4_MMQ_AMDGPU_SUDOT4_B_SIGNED true
#endif

#ifndef DOT4_MMQ_BM
#define DOT4_MMQ_BM 4
#endif

#ifndef DOT4_MMQ_BN
#define DOT4_MMQ_BN 64
#endif

#ifndef DOT4_MMQ_D
#define DOT4_MMQ_D 256
#endif

#ifndef DOT4_MMQ_THREADS
#define DOT4_MMQ_THREADS 256
#endif

#if DOT4_MMQ_BACKEND_HIP
using dot4_mmq_stream_t = hipStream_t;
using dot4_mmq_error_t  = hipError_t;
#define DOT4_MMQ_SUCCESS             hipSuccess
#define DOT4_MMQ_ERROR_INVALID_VALUE hipErrorInvalidValue
#define DOT4_MMQ_GET_LAST_ERROR      hipGetLastError
#else
using dot4_mmq_stream_t = cudaStream_t;
using dot4_mmq_error_t  = cudaError_t;
#define DOT4_MMQ_SUCCESS             cudaSuccess
#define DOT4_MMQ_ERROR_INVALID_VALUE cudaErrorInvalidValue
#define DOT4_MMQ_GET_LAST_ERROR      cudaGetLastError
#endif

enum ggml_cuda_dot4_mmq_dtype : int {
    DOT4_MMQ_DTYPE_F16          = 0,
    DOT4_MMQ_DTYPE_F32          = 1,
    DOT4_MMQ_DTYPE_Q8_PACKED32  = 2,
};

struct ggml_cuda_dot4_mmq_params {
    // Q:
    //   dtype: f16 or f32
    //   logical layout through strides:
    //     q[b, hq, iq, d]
    const void * q;

    // K:
    //   int8 packed into int32 words.
    //   D = 256 => 8 blocks of 32 dims.
    //   each 32-d block => 8 int32 words.
    //
    //   logical:
    //     k_words[b, hkv, ik, qblock, word]
    //
    //   k_scales:
    //     k_scales[b, hkv, ik, qblock]
    const int32_t * k_words;
    const void    * k_scales;

    // V:
    //   dtype:
    //     f16
    //     f32
    //     q8 packed32 with v_scales
    //
    //   f16/f32 logical:
    //     v[b, hkv, ik, d]
    //
    //   q8 packed logical:
    //     v_words[b, hkv, ik, qblock, word]
    //     v_scales[b, hkv, ik, qblock]
    const void * v;
    const void * v_scales;

    // Output:
    //   dtype: f16 or f32
    //   logical:
    //     dst[b, hq, iq, d]
    void * dst;

    // Optional additive mask.
    //
    // If mask != nullptr:
    //   logit += mask[b, hq, iq, ik]
    //
    // Causal mask is applied before additive mask.
    // If causal-masked, logit is forced to -inf and additive mask is skipped.
    const float * mask;

    int batch_count;
    int n_q;
    int n_kv;
    int n_heads_q;
    int n_heads_kv;
    int gqa_ratio;
    int head_dim;

    int q_dtype;
    int k_scale_dtype;
    int v_dtype;
    int v_scale_dtype;
    int dst_dtype;

    int causal;
    int causal_diagonal;

    // Absolute position mapping for causal mask.
    //
    // q_abs  = q_pos_base  + iq
    // kv_abs = kv_pos_base + ik
    //
    // causal masks kv_abs > q_abs + causal_diagonal.
    int64_t q_pos_base;
    int64_t kv_pos_base;

    float attn_scale;

    // Q strides, in q scalar elements.
    // Used for f16/f32 Q.
    int64_t q_batch_stride;
    int64_t q_head_stride;
    int64_t q_row_stride;
    int64_t q_dim_stride;

    // K word strides, in int32 words.
    int64_t k_batch_stride_words;
    int64_t k_head_stride_words;
    int64_t k_row_stride_words;
    int64_t k_block_stride_words;
    int64_t k_word_stride_words;

    // K scale strides, in k_scale scalar elements.
    int64_t k_scale_batch_stride;
    int64_t k_scale_head_stride;
    int64_t k_scale_row_stride;
    int64_t k_scale_block_stride;

    // V f16/f32 scalar strides, in v scalar elements.
    int64_t v_batch_stride;
    int64_t v_head_stride;
    int64_t v_row_stride;
    int64_t v_dim_stride;

    // V q8 packed word strides, in int32 words.
    int64_t v_word_batch_stride;
    int64_t v_word_head_stride;
    int64_t v_word_row_stride;
    int64_t v_word_block_stride;
    int64_t v_word_stride;

    // V q8 scale strides, in v_scale scalar elements.
    int64_t v_scale_batch_stride;
    int64_t v_scale_head_stride;
    int64_t v_scale_row_stride;
    int64_t v_scale_block_stride;

    // Output strides, in dst scalar elements.
    int64_t dst_batch_stride;
    int64_t dst_head_stride;
    int64_t dst_row_stride;
    int64_t dst_dim_stride;

    // Additive mask strides, in float elements.
    int64_t mask_batch_stride;
    int64_t mask_head_stride;
    int64_t mask_q_stride;
    int64_t mask_k_stride;
};

static __device__ __forceinline__ float dot4_mmq_neg_inf() {
    return -3.40282346638528859812e38f;
}

static __device__ __forceinline__ bool dot4_mmq_valid_logit(const float x) {
    return x > -3.0e38f;
}

static __device__ __forceinline__ float dot4_mmq_exp(const float x) {
    return expf(x);
}

static __device__ __forceinline__ float dot4_mmq_half_to_float(const __half h) {
    return __half2float(h);
}

static __device__ __forceinline__ __half dot4_mmq_float_to_half(const float x) {
    return __float2half(x);
}

static __device__ __forceinline__ float dot4_mmq_load_scalar(
        const void * base,
        const int64_t idx,
        const int dtype) {
    if (dtype == DOT4_MMQ_DTYPE_F32) {
        return reinterpret_cast<const float *>(base)[idx];
    }

    if (dtype == DOT4_MMQ_DTYPE_F16) {
        return dot4_mmq_half_to_float(reinterpret_cast<const __half *>(base)[idx]);
    }

    return 0.0f;
}

static __device__ __forceinline__ void dot4_mmq_store_scalar(
        void * base,
        const int64_t idx,
        const int dtype,
        const float x) {
    if (dtype == DOT4_MMQ_DTYPE_F32) {
        reinterpret_cast<float *>(base)[idx] = x;
        return;
    }

    if (dtype == DOT4_MMQ_DTYPE_F16) {
        reinterpret_cast<__half *>(base)[idx] = dot4_mmq_float_to_half(x);
        return;
    }
}

static __device__ __forceinline__ int dot4_mmq_clamp_i8(const int x) {
    return x < -127 ? -127 : (x > 127 ? 127 : x);
}

static __device__ __forceinline__ int dot4_mmq_round_to_i8(const float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const int xi = __float2int_rn(x);
#else
    const int xi = (int) nearbyintf(x);
#endif
    return dot4_mmq_clamp_i8(xi);
}

static __device__ __forceinline__ int dot4_mmq_extract_s8(
        const int32_t word,
        const int lane) {
    const uint32_t u = static_cast<uint32_t>(word);
    const uint32_t b = (u >> (lane * 8)) & 0xffu;
    return static_cast<int>(static_cast<int8_t>(b));
}

static __device__ __forceinline__ int32_t dot4_mmq_pack_s8x4(
        const int x0,
        const int x1,
        const int x2,
        const int x3) {
    const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(x0));
    const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(x1));
    const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(x2));
    const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(x3));

    const uint32_t packed =
        (b0 <<  0) |
        (b1 <<  8) |
        (b2 << 16) |
        (b3 << 24);

    return static_cast<int32_t>(packed);
}

static __device__ __forceinline__ int dot4_mmq_dot4_scalar_i8_i8(
        const int32_t a,
        const int32_t b,
        int acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc += dot4_mmq_extract_s8(a, i) * dot4_mmq_extract_s8(b, i);
    }
    return acc;
}

static __device__ __forceinline__ int dot4_mmq_dot4_i8_i8(
        const int32_t q_word,
        const int32_t k_word,
        const int acc) {
#if DOT4_MMQ_USE_SCALAR_DOT4
    return dot4_mmq_dot4_scalar_i8_i8(q_word, k_word, acc);
#else

#if defined(__HIP_PLATFORM_AMD__) && ( \
        defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
        defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))

    return __builtin_amdgcn_sudot4(
        DOT4_MMQ_AMDGPU_SUDOT4_A_SIGNED,
        q_word,
        DOT4_MMQ_AMDGPU_SUDOT4_B_SIGNED,
        k_word,
        acc,
        false);

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)

    return __dp4a(q_word, k_word, acc);

#else

    return dot4_mmq_dot4_scalar_i8_i8(q_word, k_word, acc);

#endif

#endif
}

static __device__ __forceinline__ float dot4_mmq_load_q(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hq,
        const int iq,
        const int d) {
    const int64_t idx =
        static_cast<int64_t>(b)  * p.q_batch_stride +
        static_cast<int64_t>(hq) * p.q_head_stride  +
        static_cast<int64_t>(iq) * p.q_row_stride   +
        static_cast<int64_t>(d)  * p.q_dim_stride;

    return dot4_mmq_load_scalar(p.q, idx, p.q_dtype);
}

static __device__ __forceinline__ int32_t dot4_mmq_load_k_word(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hkv,
        const int ik,
        const int qblock,
        const int word) {
    const int64_t idx =
        static_cast<int64_t>(b)      * p.k_batch_stride_words +
        static_cast<int64_t>(hkv)    * p.k_head_stride_words  +
        static_cast<int64_t>(ik)     * p.k_row_stride_words   +
        static_cast<int64_t>(qblock) * p.k_block_stride_words +
        static_cast<int64_t>(word)   * p.k_word_stride_words;

    return p.k_words[idx];
}

static __device__ __forceinline__ float dot4_mmq_load_k_scale(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hkv,
        const int ik,
        const int qblock) {
    const int64_t idx =
        static_cast<int64_t>(b)      * p.k_scale_batch_stride +
        static_cast<int64_t>(hkv)    * p.k_scale_head_stride  +
        static_cast<int64_t>(ik)     * p.k_scale_row_stride   +
        static_cast<int64_t>(qblock) * p.k_scale_block_stride;

    return dot4_mmq_load_scalar(p.k_scales, idx, p.k_scale_dtype);
}

static __device__ __forceinline__ float dot4_mmq_load_v_q8_packed32(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hkv,
        const int ik,
        const int d) {
    const int qblock = d >> 5;
    const int in_blk = d & 31;
    const int word   = in_blk >> 2;
    const int lane   = in_blk & 3;

    const int64_t widx =
        static_cast<int64_t>(b)      * p.v_word_batch_stride +
        static_cast<int64_t>(hkv)    * p.v_word_head_stride  +
        static_cast<int64_t>(ik)     * p.v_word_row_stride   +
        static_cast<int64_t>(qblock) * p.v_word_block_stride +
        static_cast<int64_t>(word)   * p.v_word_stride;

    const int32_t packed = reinterpret_cast<const int32_t *>(p.v)[widx];
    const int vi = dot4_mmq_extract_s8(packed, lane);

    const int64_t sidx =
        static_cast<int64_t>(b)      * p.v_scale_batch_stride +
        static_cast<int64_t>(hkv)    * p.v_scale_head_stride  +
        static_cast<int64_t>(ik)     * p.v_scale_row_stride   +
        static_cast<int64_t>(qblock) * p.v_scale_block_stride;

    const float scale = dot4_mmq_load_scalar(p.v_scales, sidx, p.v_scale_dtype);
    return static_cast<float>(vi) * scale;
}

static __device__ __forceinline__ float dot4_mmq_load_v(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hkv,
        const int ik,
        const int d) {
    if (p.v_dtype == DOT4_MMQ_DTYPE_Q8_PACKED32) {
        return dot4_mmq_load_v_q8_packed32(p, b, hkv, ik, d);
    }

    const int64_t idx =
        static_cast<int64_t>(b)   * p.v_batch_stride +
        static_cast<int64_t>(hkv) * p.v_head_stride  +
        static_cast<int64_t>(ik)  * p.v_row_stride   +
        static_cast<int64_t>(d)   * p.v_dim_stride;

    return dot4_mmq_load_scalar(p.v, idx, p.v_dtype);
}

static __device__ __forceinline__ void dot4_mmq_store_dst(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hq,
        const int iq,
        const int d,
        const float x) {
    const int64_t idx =
        static_cast<int64_t>(b)  * p.dst_batch_stride +
        static_cast<int64_t>(hq) * p.dst_head_stride  +
        static_cast<int64_t>(iq) * p.dst_row_stride   +
        static_cast<int64_t>(d)  * p.dst_dim_stride;

    dot4_mmq_store_scalar(p.dst, idx, p.dst_dtype, x);
}

static __device__ __forceinline__ float dot4_mmq_load_mask(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hq,
        const int iq,
        const int ik) {
    if (p.mask == nullptr) {
        return 0.0f;
    }

    const int64_t idx =
        static_cast<int64_t>(b)  * p.mask_batch_stride +
        static_cast<int64_t>(hq) * p.mask_head_stride  +
        static_cast<int64_t>(iq) * p.mask_q_stride     +
        static_cast<int64_t>(ik) * p.mask_k_stride;

    return p.mask[idx];
}

static __device__ __forceinline__ bool dot4_mmq_is_causal_masked(
        const ggml_cuda_dot4_mmq_params & p,
        const int iq,
        const int ik) {
    if (!p.causal) {
        return false;
    }

    const int64_t q_abs  = p.q_pos_base  + static_cast<int64_t>(iq);
    const int64_t kv_abs = p.kv_pos_base + static_cast<int64_t>(ik);

    return kv_abs > q_abs + static_cast<int64_t>(p.causal_diagonal);
}

template<int BM, int BN, int D>
static __device__ __forceinline__ void dot4_mmq_quantize_q_tile(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hq,
        const int q0,
        const int tid,
        int32_t (&s_q_words)[BM][D / 4],
        float   (&s_q_scales)[BM][D / 32]) {
    constexpr int QBLOCKS = D / 32;
    constexpr int WORDS_PER_BLOCK = 8;

    if (tid >= BM * QBLOCKS) {
        return;
    }

    const int r  = tid / QBLOCKS;
    const int qb = tid - r * QBLOCKS;
    const int iq = q0 + r;

    if (iq >= p.n_q) {
        s_q_scales[r][qb] = 0.0f;

#pragma unroll
        for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
            s_q_words[r][qb * WORDS_PER_BLOCK + w] = 0;
        }

        return;
    }

    float amax = 0.0f;

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int d = qb * 32 + i;
        const float x = dot4_mmq_load_q(p, b, hq, iq, d);
        amax = fmaxf(amax, fabsf(x));
    }

    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv   = amax > 0.0f ? 127.0f / amax : 0.0f;

    s_q_scales[r][qb] = scale;

#pragma unroll
    for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
        const int d0 = qb * 32 + w * 4 + 0;
        const int d1 = qb * 32 + w * 4 + 1;
        const int d2 = qb * 32 + w * 4 + 2;
        const int d3 = qb * 32 + w * 4 + 3;

        const int q0i = amax > 0.0f ? dot4_mmq_round_to_i8(dot4_mmq_load_q(p, b, hq, iq, d0) * inv) : 0;
        const int q1i = amax > 0.0f ? dot4_mmq_round_to_i8(dot4_mmq_load_q(p, b, hq, iq, d1) * inv) : 0;
        const int q2i = amax > 0.0f ? dot4_mmq_round_to_i8(dot4_mmq_load_q(p, b, hq, iq, d2) * inv) : 0;
        const int q3i = amax > 0.0f ? dot4_mmq_round_to_i8(dot4_mmq_load_q(p, b, hq, iq, d3) * inv) : 0;

        s_q_words[r][qb * WORDS_PER_BLOCK + w] =
            dot4_mmq_pack_s8x4(q0i, q1i, q2i, q3i);
    }
}

template<int BM, int BN, int D>
static __device__ __forceinline__ float dot4_mmq_compute_qk_logit(
        const ggml_cuda_dot4_mmq_params & p,
        const int b,
        const int hkv,
        const int ik,
        const int q_row,
        const int32_t (&s_q_words)[BM][D / 4],
        const float   (&s_q_scales)[BM][D / 32]) {
    constexpr int QBLOCKS = D / 32;
    constexpr int WORDS_PER_BLOCK = 8;

    float qk = 0.0f;

#pragma unroll
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        int acc = 0;

#pragma unroll
        for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
            const int32_t qw = s_q_words[q_row][qb * WORDS_PER_BLOCK + w];
            const int32_t kw = dot4_mmq_load_k_word(p, b, hkv, ik, qb, w);
            acc = dot4_mmq_dot4_i8_i8(qw, kw, acc);
        }

        const float qs = s_q_scales[q_row][qb];
        const float ks = dot4_mmq_load_k_scale(p, b, hkv, ik, qb);
        qk += static_cast<float>(acc) * qs * ks;
    }

    return qk;
}

template<int BM, int BN, int D>
__global__ __launch_bounds__(DOT4_MMQ_THREADS, 2)
void ggml_cuda_fattn_packed16_dot4_mmq_kernel(
        ggml_cuda_dot4_mmq_params p) {
    static_assert(D == 256, "DOT4-MMQ initial kernel is D=256 only");
    static_assert(BM > 0, "BM must be positive");
    static_assert(BN > 0, "BN must be positive");
    static_assert(BM * BN <= DOT4_MMQ_THREADS, "BM * BN must fit in one CTA for logit compute");
    static_assert(D <= DOT4_MMQ_THREADS, "D must fit one dimension thread per CTA");

    constexpr int QBLOCKS = D / 32;
    constexpr int WORDS_TOTAL = D / 4;

    const int tid = threadIdx.x;

    const int q_tile = static_cast<int>(blockIdx.x);
    const int hq     = static_cast<int>(blockIdx.y);
    const int b      = static_cast<int>(blockIdx.z);

    const int q0 = q_tile * BM;

    if (b >= p.batch_count || hq >= p.n_heads_q || q0 >= p.n_q) {
        return;
    }

    int gqa = p.gqa_ratio;
    if (gqa <= 0) {
        gqa = p.n_heads_kv > 0 ? (p.n_heads_q / p.n_heads_kv) : 1;
    }
    if (gqa <= 0) {
        gqa = 1;
    }

    const int hkv = hq / gqa;

    if (hkv >= p.n_heads_kv) {
        return;
    }

    __shared__ int32_t s_q_words[BM][WORDS_TOTAL];
    __shared__ float   s_q_scales[BM][QBLOCKS];

    // Reused:
    //   before softmax: logits[r][n]
    //   after softmax:  p[r][n]
    __shared__ float s_logits[BM][BN];

    __shared__ float s_m[BM];
    __shared__ float s_l[BM];
    __shared__ float s_m_new[BM];
    __shared__ float s_l_new[BM];

    if (tid < BM) {
        s_m[tid] = dot4_mmq_neg_inf();
        s_l[tid] = 0.0f;
        s_m_new[tid] = dot4_mmq_neg_inf();
        s_l_new[tid] = 0.0f;
    }

    dot4_mmq_quantize_q_tile<BM, BN, D>(
        p,
        b,
        hq,
        q0,
        tid,
        s_q_words,
        s_q_scales);

    __syncthreads();

    float o[BM];

#pragma unroll
    for (int r = 0; r < BM; ++r) {
        o[r] = 0.0f;
    }

    const int last_q_valid = min(q0 + BM - 1, p.n_q - 1);

    for (int kv0 = 0; kv0 < p.n_kv; kv0 += BN) {
        // Causal future-tile skip.
        // Since K/V tiles are processed in increasing ik order, if the first
        // kv token in this tile is beyond the max allowed q position for all
        // Q rows in this CTA, all later tiles are also future-only.
        if (p.causal && last_q_valid >= q0) {
            const int64_t q_abs_max =
                p.q_pos_base +
                static_cast<int64_t>(last_q_valid) +
                static_cast<int64_t>(p.causal_diagonal);

            const int64_t kv_abs_first =
                p.kv_pos_base + static_cast<int64_t>(kv0);

            if (kv_abs_first > q_abs_max) {
                break;
            }
        }

        if (tid < BM * BN) {
            const int r = tid / BN;
            const int n = tid - r * BN;

            const int iq = q0 + r;
            const int ik = kv0 + n;

            float logit = dot4_mmq_neg_inf();

            if (iq < p.n_q && ik < p.n_kv) {
                const bool causal_masked = dot4_mmq_is_causal_masked(p, iq, ik);

                if (!causal_masked) {
                    logit = dot4_mmq_compute_qk_logit<BM, BN, D>(
                        p,
                        b,
                        hkv,
                        ik,
                        r,
                        s_q_words,
                        s_q_scales);

                    logit *= p.attn_scale;
                    logit += dot4_mmq_load_mask(p, b, hq, iq, ik);
                }
            }

            s_logits[r][n] = logit;
        }

        __syncthreads();

        // Per-Q-row online softmax update.
        //
        // This reduction is intentionally simple for the first backend version.
        // The expensive part remains QK + PV; replace this with warp/block
        // reductions after correctness is locked.
        if (tid < BM) {
            const int r = tid;
            const int iq = q0 + r;

            float tile_max = dot4_mmq_neg_inf();

            if (iq < p.n_q) {
#pragma unroll
                for (int n = 0; n < BN; ++n) {
                    tile_max = fmaxf(tile_max, s_logits[r][n]);
                }
            }

            const float old_m = s_m[r];
            const float old_l = s_l[r];
            const float m_new = fmaxf(old_m, tile_max);

            s_m_new[r] = m_new;

            float l_new = 0.0f;

            if (dot4_mmq_valid_logit(m_new)) {
                const float old_scale =
                    (old_l > 0.0f && dot4_mmq_valid_logit(old_m))
                        ? dot4_mmq_exp(old_m - m_new)
                        : 0.0f;

                l_new = old_l * old_scale;

#pragma unroll
                for (int n = 0; n < BN; ++n) {
                    const float x = s_logits[r][n];
                    const float pv =
                        dot4_mmq_valid_logit(x)
                            ? dot4_mmq_exp(x - m_new)
                            : 0.0f;

                    s_logits[r][n] = pv;
                    l_new += pv;
                }
            } else {
#pragma unroll
                for (int n = 0; n < BN; ++n) {
                    s_logits[r][n] = 0.0f;
                }

                l_new = 0.0f;
            }

            s_l_new[r] = l_new;
        }

        __syncthreads();

        // P @ V.
        //
        // One thread owns one output dimension d for all BM rows.
        if (tid < D) {
            const int d = tid;

#pragma unroll
            for (int r = 0; r < BM; ++r) {
                const int iq = q0 + r;

                if (iq >= p.n_q) {
                    continue;
                }

                const float old_m = s_m[r];
                const float old_l = s_l[r];
                const float m_new = s_m_new[r];

                const float old_scale =
                    (old_l > 0.0f &&
                     dot4_mmq_valid_logit(old_m) &&
                     dot4_mmq_valid_logit(m_new))
                        ? dot4_mmq_exp(old_m - m_new)
                        : 0.0f;

                float pv_sum = 0.0f;

#pragma unroll
                for (int n = 0; n < BN; ++n) {
                    const int ik = kv0 + n;

                    if (ik >= p.n_kv) {
                        continue;
                    }

                    const float prob = s_logits[r][n];

                    if (prob != 0.0f) {
                        const float vv = dot4_mmq_load_v(p, b, hkv, ik, d);
                        pv_sum += prob * vv;
                    }
                }

                o[r] = o[r] * old_scale + pv_sum;
            }
        }

        __syncthreads();

        if (tid < BM) {
            s_m[tid] = s_m_new[tid];
            s_l[tid] = s_l_new[tid];
        }

        __syncthreads();
    }

    if (tid < D) {
        const int d = tid;

#pragma unroll
        for (int r = 0; r < BM; ++r) {
            const int iq = q0 + r;

            if (iq >= p.n_q) {
                continue;
            }

            const float denom = s_l[r];
            const float out = denom > 0.0f ? (o[r] / denom) : 0.0f;

            dot4_mmq_store_dst(p, b, hq, iq, d, out);
        }
    }
}

static inline bool ggml_cuda_dot4_mmq_dtype_is_scalar_q(const int dtype) {
    return dtype == DOT4_MMQ_DTYPE_F16 || dtype == DOT4_MMQ_DTYPE_F32;
}

static inline bool ggml_cuda_dot4_mmq_dtype_is_scalar_dst(const int dtype) {
    return dtype == DOT4_MMQ_DTYPE_F16 || dtype == DOT4_MMQ_DTYPE_F32;
}

static inline bool ggml_cuda_dot4_mmq_dtype_is_scale(const int dtype) {
    return dtype == DOT4_MMQ_DTYPE_F16 || dtype == DOT4_MMQ_DTYPE_F32;
}

extern "C" int ggml_cuda_fattn_packed16_dot4_mmq_can_run(
        ggml_cuda_dot4_mmq_params p) {
    if (p.head_dim != DOT4_MMQ_D) {
        return 0;
    }

    if (p.batch_count <= 0 || p.n_q <= 0 || p.n_kv <= 0) {
        return 0;
    }

    if (p.n_heads_q <= 0 || p.n_heads_kv <= 0) {
        return 0;
    }

    if (p.q == nullptr || p.k_words == nullptr || p.k_scales == nullptr ||
        p.v == nullptr || p.dst == nullptr) {
        return 0;
    }

    if (!ggml_cuda_dot4_mmq_dtype_is_scalar_q(p.q_dtype)) {
        return 0;
    }

    if (!ggml_cuda_dot4_mmq_dtype_is_scale(p.k_scale_dtype)) {
        return 0;
    }

    if (p.v_dtype == DOT4_MMQ_DTYPE_Q8_PACKED32) {
        if (p.v_scales == nullptr) {
            return 0;
        }

        if (!ggml_cuda_dot4_mmq_dtype_is_scale(p.v_scale_dtype)) {
            return 0;
        }
    } else if (!ggml_cuda_dot4_mmq_dtype_is_scalar_q(p.v_dtype)) {
        return 0;
    }

    if (!ggml_cuda_dot4_mmq_dtype_is_scalar_dst(p.dst_dtype)) {
        return 0;
    }

    int gqa = p.gqa_ratio;
    if (gqa <= 0) {
        if (p.n_heads_kv <= 0) {
            return 0;
        }

        if ((p.n_heads_q % p.n_heads_kv) != 0) {
            return 0;
        }

        gqa = p.n_heads_q / p.n_heads_kv;
    }

    if (gqa <= 0) {
        return 0;
    }

    if ((p.n_heads_q + gqa - 1) / gqa > p.n_heads_kv) {
        return 0;
    }

    return 1;
}

extern "C" dot4_mmq_error_t ggml_cuda_fattn_packed16_dot4_mmq_launch(
        ggml_cuda_dot4_mmq_params p,
        dot4_mmq_stream_t stream) {
    if (!ggml_cuda_fattn_packed16_dot4_mmq_can_run(p)) {
        return DOT4_MMQ_ERROR_INVALID_VALUE;
    }

    if (p.gqa_ratio <= 0) {
        p.gqa_ratio = p.n_heads_q / p.n_heads_kv;
    }

    const dim3 block(DOT4_MMQ_THREADS, 1, 1);
    const dim3 grid(
        static_cast<unsigned int>((p.n_q + DOT4_MMQ_BM - 1) / DOT4_MMQ_BM),
        static_cast<unsigned int>(p.n_heads_q),
        static_cast<unsigned int>(p.batch_count));

    ggml_cuda_fattn_packed16_dot4_mmq_kernel<
        DOT4_MMQ_BM,
        DOT4_MMQ_BN,
        DOT4_MMQ_D><<<grid, block, 0, stream>>>(p);

    return DOT4_MMQ_GET_LAST_ERROR();
}

// Optional helper for route naming/debug prints.
extern "C" const char * ggml_cuda_fattn_packed16_dot4_mmq_name() {
    return "rocm_packed16_dot4_mmq";
}
