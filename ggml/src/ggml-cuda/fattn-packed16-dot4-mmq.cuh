// fattn-packed16-dot4-mmq.cuh — packed16 K cache + DOT4/MMQ QK + FlashAttention
//
// rocm_packed16_dot4_mmq is the integer sibling of packed16_wmma_tile:
//   - Q tile: f32 -> q8_0-like int8 payload + float scale per 32 dims
//   - K tile: persistent packed16 int32 payload + half scale per 32 dims
//   - QK: sudot4/dp4a int8xint8 -> int32, then q_scale*k_scale
//   - FA: online softmax + P@V, with the same packed16 row convention as WMMA
//
// Initial target: D=256, Q=f32, K=I32 packed16, V=q4_0/q8_0/f16, dst=f32.
//
// Shapes:
//   M16N16: prefill/MMQ-oriented default for nq >= 16. More Q rows per CTA, lower V LDS.
//   M8N32:  small-prefill fallback. Fewer Q rows, wider K tile.
//   M4N64:  optional long-K/small-Q debug shape; higher V LDS, not default.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#include <cfloat>
#include <cstdlib>
#include <cstring>

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data,
                                          ggml_tensor ** payload,
                                          ggml_tensor ** scales);
}

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

#define PACKED16_DOT4_MMQ_VERSION 20260531

static constexpr int PDMQ_D       = 256;
static constexpr int PDMQ_THREADS = 256;

static constexpr int PDMQ_BM_PREFILL = 16;
static constexpr int PDMQ_BN_PREFILL = 16;
static constexpr int PDMQ_BM_SMALL   = 8;
static constexpr int PDMQ_BN_SMALL   = 32;
static constexpr int PDMQ_BM_LONGK   = 4;
static constexpr int PDMQ_BN_LONGK   = 64;

static_assert(PDMQ_D % QK8_0 == 0, "packed16_dot4_mmq requires D multiple of QK8_0");
static_assert(PDMQ_BM_PREFILL * PDMQ_BN_PREFILL == PDMQ_THREADS, "M16N16 maps one QK logit per thread");
static_assert(PDMQ_BM_SMALL   * PDMQ_BN_SMALL   == PDMQ_THREADS, "M8N32 maps one QK logit per thread");
static_assert(PDMQ_BM_LONGK   * PDMQ_BN_LONGK   == PDMQ_THREADS, "M4N64 maps one QK logit per thread");

// +1 row padding mirrors the useful MMQ/RDNA LDS trick: avoid perfect
// power-of-two row strides in shared memory without materially increasing LDS.
// Keep this local to DOT4-MMQ; do not reuse normal MMQ env knobs here.
static constexpr int PDMQ_Q_WORDS       = PDMQ_D / 4;
static constexpr int PDMQ_Q_BLOCKS      = PDMQ_D / QK8_0;
static constexpr int PDMQ_Q_WORDS_PAD   = PDMQ_Q_WORDS + 1;
static constexpr int PDMQ_Q_BLOCKS_PAD  = PDMQ_Q_BLOCKS + 1;
static constexpr int PDMQ_D_PAD         = PDMQ_D + 1;

// ── Experiment envs ──────────────────────────────────────────────
// PDMQ_IMPL: baseline | kshared | kshared_stagev | kshared_directv
static const char * pdmq_impl_env() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL");
    return v ? v : "";
}
static inline bool pdmq_kshared_enabled() {
    const char * v = pdmq_impl_env();
    return strcmp(v, "kshared") == 0 || strcmp(v, "kshared_stagev") == 0 || strcmp(v, "kshared_directv") == 0;
}
static inline bool pdmq_stagev_enabled() {
    const char * v = pdmq_impl_env();
    // stagev is default; directv is opt-in
    return strcmp(v, "kshared_directv") != 0;
}

enum packed16_dot4_mmq_v_type {
    PACKED16_DOT4_MMQ_V_Q4_0,
    PACKED16_DOT4_MMQ_V_Q8_0,
    PACKED16_DOT4_MMQ_V_F16,
};

// Helpers shared by host (probe validation) and device (kernel). Pure math, no HIP deps.
#if defined(__HIPCC__) || defined(__CUDACC__)
static __host__ __device__ __forceinline__ int8_t pdmq_i8_from_i32(const int packed, const int byte) {
#else
static inline int8_t pdmq_i8_from_i32(const int packed, const int byte) {
#endif
    const uint32_t u = static_cast<uint32_t>(packed);
    return static_cast<int8_t>((u >> (8 * byte)) & 0xffu);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
static __host__ __device__ __forceinline__ int pdmq_pack_i8x4(const int q0, const int q1, const int q2, const int q3) {
#else
static inline int pdmq_pack_i8x4(const int q0, const int q1, const int q2, const int q3) {
#endif
    const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(q0));
    const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(q1));
    const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(q2));
    const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(q3));
    return static_cast<int>((b0 << 0) | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

#if defined(__HIPCC__) || defined(__CUDACC__)
static __host__ __device__ __forceinline__ int pdmq_clamp_i8(const int x) {
    return x < -127 ? -127 : (x > 127 ? 127 : x);
}
#else
static inline int pdmq_clamp_i8(const int x) {
    return x < -127 ? -127 : (x > 127 ? 127 : x);
}
#endif

#ifdef GGML_USE_HIP

static inline bool ggml_cuda_packed16_dot4_mmq_route_required() {
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    return required &&
        (strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
         strcmp(required, "packed16_dot4_mmq") == 0);
}

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    // Auto-enabled when packed16 K cache is active.
    // Disable: GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0
    {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ");
        if (v && atoi(v) == 0) return false;
    }
    return ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
}

static inline bool ggml_cuda_packed16_dot4_mmq_supported(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!ggml_cuda_packed16_dot4_mmq_enabled()) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != PDMQ_D || K->ne[0] * 4 != PDMQ_D || V->ne[0] != PDMQ_D || dst->ne[0] != PDMQ_D) {
        return false;
    }
    if (Q->ne[1] <= 1) {
        // Keep decode on the existing BN64/split-K DOT4 route for now.
        return false;
    }
    if (Q->ne[2] <= 0 || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] < K->ne[1] || V->ne[2] != K->ne[2] || (V->ne[3] != Q->ne[3] && V->ne[3] != 1)) {
        return false;
    }
    if (V->type != GGML_TYPE_Q4_0 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1)) {
        return false;
    }
    return true;
}

static __device__ __forceinline__ int pdmq_min_i(const int a, const int b) {
    return a < b ? a : b;
}

static __device__ __forceinline__ int pdmq_max_i(const int a, const int b) {
    return a > b ? a : b;
}

static __device__ __forceinline__ float pdmq_neg() {
    return -FLT_MAX / 2.0f;
}

static __device__ __forceinline__ bool pdmq_logit_is_live(const float x) {
    return x > -FLT_MAX / 4.0f;
}

static __device__ __forceinline__ int pdmq_quant_i8(const float x, const float inv_scale) {
    return pdmq_clamp_i8((int) lrintf(x * inv_scale));
}

static __device__ __forceinline__ int pdmq_dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return ggml_cuda_dp4a(a, b, c);
#endif
}

static __device__ __forceinline__ float pdmq_load_q_f32(
        const float * __restrict__ Q,
        const int64_t q_nb01,
        const int64_t q_nb02,
        const int64_t q_nb03,
        const int q,
        const int hq,
        const int b,
        const int d) {
    const char * ptr = (const char *) Q + int64_t(b) * q_nb03 + int64_t(hq) * q_nb02 + int64_t(q) * q_nb01;
    return ((const float *) ptr)[d];
}

static __device__ __forceinline__ float pdmq_decode_v_q4_0(
        const char * __restrict__ V,
        const int64_t v_nb10,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k) * v_nb11;
    const int blk = d / QK4_0;
    const int in  = d & (QK4_0 - 1);
    const block_q4_0 * bq = (const block_q4_0 *) (ptr + int64_t(blk) * v_nb10);
    const int iq = in & 15;
    const int shift = in >= 16 ? 4 : 0;
    return float((int(bq->qs[iq] >> shift) & 0x0f) - 8) * __half2float(bq->d);
}

static __device__ __forceinline__ float pdmq_decode_v_q8_0(
        const char * __restrict__ V,
        const int64_t v_nb10,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k) * v_nb11;
    const int blk = d / QK8_0;
    const block_q8_0 * bq = (const block_q8_0 *) (ptr + int64_t(blk) * v_nb10);
    return float(bq->qs[d & (QK8_0 - 1)]) * __half2float(bq->d);
}

static __device__ __forceinline__ float pdmq_decode_v_f16(
        const char * __restrict__ V,
        const int64_t v_nb10,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k) * v_nb11 + int64_t(d) * v_nb10;
    return __half2float(*(const half *) ptr);
}

template<packed16_dot4_mmq_v_type V_TYPE>
static __device__ __forceinline__ float pdmq_decode_v(
        const char * __restrict__ V,
        const int64_t v_nb10,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d) {
    if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0) {
        return pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_Q8_0) {
        return pdmq_decode_v_q8_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else {
        return pdmq_decode_v_f16(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    }
}

static __device__ __forceinline__ float pdmq_mask_val(
        const char * __restrict__ mask,
        const int64_t mask_ne00,
        const int64_t mask_ne01,
        const int64_t mask_ne03,
        const int64_t mask_nb00,
        const int64_t mask_nb01,
        const int64_t mask_nb03,
        const int q,
        const int k,
        const int b) {
    if (!mask) return 0.0f;
    if (q < 0 || q >= mask_ne01 || k < 0 || k >= mask_ne00) return pdmq_neg();
    const int mb = mask_ne03 > 1 ? (b % mask_ne03) : 0;
    const half hv = *(const half *) (mask + int64_t(mb) * mask_nb03 + int64_t(q) * mask_nb01 + int64_t(k) * mask_nb00);
    const float v = __half2float(hv);
    return isfinite(v) ? v : pdmq_neg();
}

template<int BM, int D>
static __device__ __forceinline__ void pdmq_quantize_q_tile(
        const float * __restrict__ Q,
        const int64_t q_nb01,
        const int64_t q_nb02,
        const int64_t q_nb03,
        const int q0,
        const int hq,
        const int b,
        const int nq,
        int   (&q_payload)[BM][D / 4 + 1],
        float (&q_scales) [BM][D / QK8_0 + 1]) {
    constexpr int QBLOCKS = D / QK8_0;
    constexpr int WORDS_PER_BLOCK = QK8_0 / 4;

    for (int linear = int(threadIdx.x); linear < BM * QBLOCKS; linear += int(blockDim.x)) {
        const int qr = linear / QBLOCKS;
        const int qb = linear - qr * QBLOCKS;
        const int q  = q0 + qr;

        float amax = 0.0f;
        if (q < nq) {
#pragma unroll
            for (int i = 0; i < QK8_0; ++i) {
                const float x = pdmq_load_q_f32(Q, q_nb01, q_nb02, q_nb03, q, hq, b, qb * QK8_0 + i);
                amax = fmaxf(amax, fabsf(x));
            }
        }

        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        q_scales[qr][qb] = scale;

#pragma unroll
        for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
            int qs[4] = {0, 0, 0, 0};
            if (q < nq && amax > 0.0f) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = qb * QK8_0 + w * 4 + j;
                    qs[j] = pdmq_quant_i8(pdmq_load_q_f32(Q, q_nb01, q_nb02, q_nb03, q, hq, b, d), inv_scale);
                }
            }
            q_payload[qr][qb * WORDS_PER_BLOCK + w] = pdmq_pack_i8x4(qs[0], qs[1], qs[2], qs[3]);
        }
    }
    __syncthreads();
}

template<int BM, int D>
static __device__ __forceinline__ float pdmq_qk_dot(
        const int   (&q_payload)[BM][D / 4 + 1],
        const float (&q_scales) [BM][D / QK8_0 + 1],
        const int qr,
        const int * __restrict__ k_row_payload,
        const half * __restrict__ k_row_scales) {
    constexpr int QBLOCKS = D / QK8_0;
    constexpr int WORDS_PER_BLOCK = QK8_0 / 4;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        int acc = 0;
#pragma unroll
        for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
            const int idx = qb * WORDS_PER_BLOCK + w;
            acc = pdmq_dot4_i8_i8(q_payload[qr][idx], k_row_payload[idx], acc);
        }
        sum += float(acc) * q_scales[qr][qb] * __half2float(k_row_scales[qb]);
    }
    return sum;
}

// ── GQA2 Kernel (2 Q heads per CTA, M16N16 only) ──────────────
template<packed16_dot4_mmq_v_type V_TYPE, bool CAUSAL_MASK, bool STAGE_V, bool KSHARED>
static __global__ __launch_bounds__(PDMQ_THREADS, 1) void packed16_dot4_mmq_gqa2_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows, int q_offset, float attention_scale) {

    static constexpr int GQA_GROUP = 2;
    static constexpr int BM = 16;
    static constexpr int BN = 16;
    static constexpr int D  = PDMQ_D;

    const int tid = int(threadIdx.x);
    const int q_tile = int(blockIdx.x);
    const int gy     = int(blockIdx.y);
    const int b      = int(blockIdx.z);

    const int groups_per_kv = CEIL_DIV(gqa_ratio, GQA_GROUP);
    const int hk            = gy / groups_per_kv;
    const int local_group   = gy - hk * groups_per_kv;
    const int hq0 = hk * gqa_ratio + local_group * GQA_GROUP + 0;
    const int hq1 = hk * gqa_ratio + local_group * GQA_GROUP + 1;
    const bool slot_valid_0 = hq0 < n_heads_q && hq0 < (hk + 1) * gqa_ratio;
    const bool slot_valid_1 = hq1 < n_heads_q && hq1 < (hk + 1) * gqa_ratio;
    const int q0 = q_tile * BM;

    if (q0 >= nq) return;

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ int   q_tile_i32[GQA_GROUP][BM][D / 4 + 1];
    __shared__ float q_tile_scales[GQA_GROUP][BM][D / QK8_0 + 1];
    __shared__ float logits[BM][BN + 1];
    __shared__ float row_m[GQA_GROUP][BM + 1];
    __shared__ float row_l[GQA_GROUP][BM + 1];
    __shared__ float old_s[GQA_GROUP][BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ int   k_payload_s[KSHARED ? BN * (PDMQ_D / 4 + 1) : 1];
    __shared__ half  k_scales_s [KSHARED ? BN * (PDMQ_D / QK8_0 + 1) : 1];

    for (int s = 0; s < GQA_GROUP; ++s) {
        if (tid < BM) { row_m[s][tid] = pdmq_neg(); row_l[s][tid] = 0.0f; old_s[s][tid] = 0.0f; }
    }

    float out[GQA_GROUP][BM];
    for (int s = 0; s < GQA_GROUP; ++s)
        for (int qr = 0; qr < BM; ++qr) out[s][qr] = 0.0f;
    __syncthreads();

    // Quantize Q for both slots
    if (slot_valid_0)
        pdmq_quantize_q_tile<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq0, b, nq, q_tile_i32[0], q_tile_scales[0]);
    if (slot_valid_1)
        pdmq_quantize_q_tile<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq1, b, nq, q_tile_i32[1], q_tile_scales[1]);

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = pdmq_min_i(BN, nk - k0);

        if constexpr (CAUSAL_MASK) {
            const int q_last = pdmq_min_i(q0 + BM - 1, nq - 1);
            if (k0 > q_offset + q_last) break;
        }

        // V load: once per K tile, shared by both slots
        if constexpr (STAGE_V) {
            for (int v_idx = tid; v_idx < BN * D; v_idx += int(blockDim.x)) {
                const int kk = v_idx / D, d = v_idx - kk * D, k = k0 + kk;
                v_tile[kk * (D + 1) + d] = kk < tile_n
                    ? pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d) : 0.0f;
            }
            __syncthreads();
        }

        // KSHARED: stage K payload + scales into shared memory once per K tile
        if constexpr (KSHARED) {
            constexpr int KP_PER_ROW = PDMQ_D / 4;
            constexpr int KS_PER_ROW = PDMQ_D / QK8_0;
            for (int idx = tid; idx < tile_n * KP_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KP_PER_ROW, w = idx - kk * KP_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_payload_s[kk * (KP_PER_ROW + 1) + w] = k_payload[k_row + w];
            }
            for (int idx = tid; idx < tile_n * KS_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KS_PER_ROW, s_val = idx - kk * KS_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_scales_s[kk * (KS_PER_ROW + 1) + s_val] = k_scales[k_row + s_val];
            }
            __syncthreads();
        }

        // Process both Q slots
        for (int s = 0; s < GQA_GROUP; ++s) {
            const bool slot_valid = (s == 0) ? slot_valid_0 : slot_valid_1;
            if (!slot_valid) continue;

            const int hq_slot = (s == 0) ? hq0 : hq1;

            // QK compute
            for (int idx = tid; idx < BM * BN; idx += int(blockDim.x)) {
                const int qr = idx / BN, kk = idx - qr * BN, q = q0 + qr, k = k0 + kk;
                float s_val = pdmq_neg();
                bool valid = q < nq && kk < tile_n && k < nk;
                if constexpr (CAUSAL_MASK) valid = valid && (k <= q_offset + q);
                if (valid) {
                    const int  * kp_row;
                    const half * ks_row;
                    if constexpr (KSHARED) {
                        kp_row = &k_payload_s[kk * (PDMQ_D / 4 + 1)];
                        ks_row = &k_scales_s [kk * (PDMQ_D / QK8_0 + 1)];
                    } else {
                        const size_t k_row = k_head_base + size_t(k);
                        kp_row = k_payload + k_row * (D / 4);
                        ks_row = k_scales  + k_row * (D / QK8_0);
                    }
                    s_val = pdmq_qk_dot<BM, D>(q_tile_i32[s], q_tile_scales[s], qr, kp_row, ks_row) * attention_scale;
                    if (!CAUSAL_MASK && mask)
                        s_val += pdmq_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b);
                }
                logits[qr][kk] = s_val;
            }
            __syncthreads();

            // Online softmax
            if (tid < BM) {
                const int qr = tid, q = q0 + qr;
                float tile_m = pdmq_neg();
                if (q < nq) {
                    for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[qr][kk]);
                }
                const float m_new = fmaxf(row_m[s][qr], tile_m);
                const float alpha = row_l[s][qr] > 0.0f && pdmq_logit_is_live(row_m[s][qr]) && pdmq_logit_is_live(m_new)
                    ? expf(row_m[s][qr] - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) {
                    const float lv = logits[qr][kk];
                    const float p = pdmq_logit_is_live(lv) && pdmq_logit_is_live(m_new) ? expf(lv - m_new) : 0.0f;
                    logits[qr][kk] = p; tile_l += p;
                }
                row_l[s][qr] = row_l[s][qr] * alpha + tile_l;
                row_m[s][qr] = m_new;
                old_s[s][qr] = alpha;
            }
            __syncthreads();

            // PV accumulate
            if (tid < D) {
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    if (q >= nq) continue;
                    float acc = out[s][qr] * old_s[s][qr];
                    for (int kk = 0; kk < BN; ++kk) {
                        const float p = logits[qr][kk];
                        if (p != 0.0f)
                            acc += p * (STAGE_V ? v_tile[kk * (D + 1) + tid]
                                               : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid));
                    }
                    out[s][qr] = acc;
                }
            }
            __syncthreads();
        }
    }

    // Final write: slot 0 → hq0, slot 1 → hq1
    if (tid < D) {
        if (slot_valid_0) for (int qr = 0; qr < BM; ++qr) {
            const int q = q0 + qr;
            if (q >= nq) continue;
            const float denom = row_l[0][qr];
            dst[((size_t(b) * nq + q) * n_heads_q + hq0) * D + tid] = denom > 0.0f ? out[0][qr] / denom : 0.0f;
        }
        if (slot_valid_1) for (int qr = 0; qr < BM; ++qr) {
            const int q = q0 + qr;
            if (q >= nq) continue;
            const float denom = row_l[1][qr];
            dst[((size_t(b) * nq + q) * n_heads_q + hq1) * D + tid] = denom > 0.0f ? out[1][qr] / denom : 0.0f;
        }
    }
}

template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool KSHARED>
static __global__ __launch_bounds__(PDMQ_THREADS, 1) void packed16_dot4_mmq_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        int64_t q_nb01,
        int64_t q_nb02,
        int64_t q_nb03,
        int64_t v_nb10,
        int64_t v_nb11,
        int64_t v_nb12,
        int64_t v_nb13,
        int64_t v_ne13,
        const char * __restrict__ mask,
        int64_t mask_ne00,
        int64_t mask_ne01,
        int64_t mask_ne03,
        int64_t mask_nb00,
        int64_t mask_nb01,
        int64_t mask_nb03,
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int packed_rows,
        int q_offset,
        float attention_scale) {

    static_assert(D == PDMQ_D, "packed16_dot4_mmq is D256-only");
    
    const int tid = int(threadIdx.x);
    const int q_tile = int(blockIdx.x);
    const int hq = int(blockIdx.y);
    const int b  = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    const int q0 = q_tile * BM;

    if (hq >= n_heads_q || hk >= n_heads_k || q0 >= nq) {
        return;
    }

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ int   q_tile_i32[BM][D / 4 + 1];
    __shared__ float q_tile_scales[BM][D / QK8_0 + 1];
    __shared__ float logits[BM][BN + 1];
    __shared__ float row_m[BM + 1];
    __shared__ float row_l[BM + 1];
    __shared__ float old_s[BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ int   k_payload_s[KSHARED ? BN * (PDMQ_D / 4 + 1) : 1];
    __shared__ half  k_scales_s [KSHARED ? BN * (PDMQ_D / QK8_0 + 1) : 1];

    if (tid < BM) {
        row_m[tid] = pdmq_neg();
        row_l[tid] = 0.0f;
        old_s[tid] = 0.0f;
    }

    float out[BM];
#pragma unroll
    for (int qr = 0; qr < BM; ++qr) {
        out[qr] = 0.0f;
    }

    pdmq_quantize_q_tile<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq, b, nq, q_tile_i32, q_tile_scales);

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = pdmq_min_i(BN, nk - k0);

        if constexpr (CAUSAL_MASK) {
            const int q_last = pdmq_min_i(q0 + BM - 1, nq - 1);
            if (k0 > q_offset + q_last) {
                break;
            }
        }

        if constexpr (STAGE_V) {
        for (int v_idx = tid; v_idx < BN * D; v_idx += int(blockDim.x)) {
            const int kk = v_idx / D;
            const int d  = v_idx - kk * D;
            const int k  = k0 + kk;
            v_tile[kk * (D + 1) + d] = kk < tile_n
                ? pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d)
                : 0.0f;
        }
        __syncthreads();
        }

        // KSHARED: stage K payload + scales into shared memory once per K tile
        if constexpr (KSHARED) {
            constexpr int KP_PER_ROW = PDMQ_D / 4;
            constexpr int KS_PER_ROW = PDMQ_D / QK8_0;
            for (int idx = tid; idx < tile_n * KP_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KP_PER_ROW, w = idx - kk * KP_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_payload_s[kk * (KP_PER_ROW + 1) + w] = k_payload[k_row + w];
            }
            for (int idx = tid; idx < tile_n * KS_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KS_PER_ROW, s = idx - kk * KS_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_scales_s[kk * (KS_PER_ROW + 1) + s] = k_scales[k_row + s];
            }
            __syncthreads();
        }

        const bool full_q = q0 + BM <= nq;
        const bool full_k = tile_n == BN;
        bool exact_tile = full_q && full_k && mask == nullptr;
        if constexpr (CAUSAL_MASK) {
            exact_tile = exact_tile && (k0 + BN - 1 <= q_offset + q0);
        }

        if (tid < BM * BN) {
            const int qr = tid / BN;
            const int kk = tid - qr * BN;
            const int q  = q0 + qr;
            const int k  = k0 + kk;

            float s = pdmq_neg();
            bool valid = exact_tile || (q < nq && kk < tile_n && k < nk);
            if constexpr (CAUSAL_MASK) {
                valid = valid && (k <= q_offset + q);
            }

            if (valid) {
                const int  * k_row_payload;
                const half * k_row_scales;
                if constexpr (KSHARED) {
                    k_row_payload = &k_payload_s[kk * (PDMQ_D / 4 + 1)];
                    k_row_scales  = &k_scales_s [kk * (PDMQ_D / QK8_0 + 1)];
                } else {
                    const size_t k_row = k_head_base + size_t(k);
                    k_row_payload = k_payload + k_row * (D / 4);
                    k_row_scales  = k_scales  + k_row * (D / QK8_0);
                }
                s = pdmq_qk_dot<BM, D>(q_tile_i32, q_tile_scales, qr, k_row_payload, k_row_scales) * attention_scale;
                if (!exact_tile && mask && !CAUSAL_MASK) {
                    s += pdmq_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b);
                }
            }
            logits[qr][kk] = s;
        }
        __syncthreads();

        if (tid < BM) {
            const int qr = tid;
            const int q  = q0 + qr;
            float tile_m = pdmq_neg();
            if (q < nq) {
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const float s = logits[qr][kk];
                    tile_m = fmaxf(tile_m, s);
                }
            }

            const float m_new = fmaxf(row_m[qr], tile_m);
            const float alpha = row_l[qr] > 0.0f && pdmq_logit_is_live(row_m[qr]) && pdmq_logit_is_live(m_new)
                ? expf(row_m[qr] - m_new)
                : 0.0f;
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                const float s = logits[qr][kk];
                const float p = pdmq_logit_is_live(s) && pdmq_logit_is_live(m_new) ? expf(s - m_new) : 0.0f;
                logits[qr][kk] = p;
                tile_l += p;
            }
            row_l[qr] = row_l[qr] * alpha + tile_l;
            row_m[qr] = m_new;
            old_s[qr] = alpha;
        }
        __syncthreads();

        if (tid < D) {
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) {
                    continue;
                }
                float acc = out[qr] * old_s[qr];
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const float p = logits[qr][kk];
                    if (p != 0.0f) {
                        const float vv = STAGE_V
                            ? v_tile[kk * (D + 1) + tid]
                            : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid);
                        acc += p * vv;
                    }
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
    }

    if (tid < D) {
#pragma unroll
        for (int qr = 0; qr < BM; ++qr) {
            const int q = q0 + qr;
            if (q >= nq) {
                continue;
            }
            const float denom = row_l[qr];
            const float val = denom > 0.0f ? out[qr] / denom : 0.0f;
            dst[((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * size_t(D) + size_t(tid)] = val;
        }
    }
}

// ── QK probe ────────────────────────────────────────────────────
//
// Validates DOT4-MMQ QK math against same-quantized scalar reference.
// Q tile: [16][256] f32, quantized per 32-D block to int8.
// K tile: [16][256] represented as packed int8 payload + half scales.
// Output: [16][16] QK = int8xint8 DOT4 + scale application.
//
// Launch: <<<1, PDMQ_THREADS, 0, stream>>>

static __global__ void pdmq_qk_probe_kernel(
        const float * __restrict__ q_probe,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ out) {

    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int D  = PDMQ_D;

    __shared__ int   q_tile_i32[BM][D / 4 + 1];
    __shared__ float q_tile_scales[BM][D / QK8_0 + 1];

    pdmq_quantize_q_tile<BM, D>(q_probe, sizeof(float) * D, 0, 0, 0, 0, 0, BM, q_tile_i32, q_tile_scales);

    for (int idx = int(threadIdx.x); idx < BM * BN; idx += int(blockDim.x)) {
        const int qr = idx / BN;
        const int kk = idx - qr * BN;

        const int  * k_row_payload = k_payload + size_t(kk) * (D / 4);
        const half * k_row_scales  = k_scales  + size_t(kk) * (D / QK8_0);
        float s = pdmq_qk_dot<BM, D>(q_tile_i32, q_tile_scales, qr, k_row_payload, k_row_scales);
        out[qr * BN + kk] = s;
    }
}

static bool pdmq_qk_probe_pass(hipStream_t stream) {
    constexpr int NQ   = 16;
    constexpr int NK   = 16;
    constexpr int D    = PDMQ_D;
    constexpr int QBLOCKS = D / QK8_0;
    constexpr int WORDS_PER_BLOCK = QK8_0 / 4;

    constexpr size_t qk_bytes    = NQ  * D * sizeof(float);
    constexpr size_t kp_bytes    = NK  * (D / 4) * sizeof(int);
    constexpr size_t ks_bytes    = NK  * (D / QK8_0) * sizeof(half);
    constexpr size_t out_bytes   = NQ  * NK * sizeof(float);

    float * h_q  = (float *) malloc(qk_bytes);
    int   * h_kp = (int   *) malloc(kp_bytes);
    half  * h_ks = (half  *) malloc(ks_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    srand(87123);
    for (size_t i = 0; i < NQ * D; i++) {
        h_q[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 4.0f;
    }

    srand(45678);
    for (int kk = 0; kk < NK; ++kk) {
        for (int qb = 0; qb < QBLOCKS; ++qb) {
            float k_amax = 0.0f;
            int8_t k_raw[QK8_0];
            for (int i = 0; i < QK8_0; ++i) {
                k_raw[i] = (int8_t)((rand() % 255) - 127);
                k_amax = fmaxf(k_amax, fabsf((float)k_raw[i]));
            }
            float ks = k_amax > 0.0f ? k_amax / 127.0f : 0.0f;
            h_ks[kk * QBLOCKS + qb] = __float2half(ks);
            for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
                h_kp[kk * (D / 4) + qb * WORDS_PER_BLOCK + w] =
                    pdmq_pack_i8x4(k_raw[w*4+0], k_raw[w*4+1], k_raw[w*4+2], k_raw[w*4+3]);
            }
        }
    }

    // Scalar reference: same quantization, same DOT4 math
    for (int qr = 0; qr < NQ; ++qr) {
        float q_scales[QBLOCKS];
        float q_invscales[QBLOCKS];
        int8_t q_i8[QBLOCKS * QK8_0];

        for (int qb = 0; qb < QBLOCKS; ++qb) {
            float amax = 0.0f;
            for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(h_q[qr * D + qb * QK8_0 + i]));
            q_scales[qb] = amax > 0.0f ? amax / 127.0f : 0.0f;
            q_invscales[qb] = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int i = 0; i < QK8_0; ++i)
                q_i8[qb * QK8_0 + i] = (int8_t) pdmq_clamp_i8((int) lrintf(h_q[qr * D + qb * QK8_0 + i] * q_invscales[qb]));
        }

        for (int kk = 0; kk < NK; ++kk) {
            float sum = 0.0f;
            for (int qb = 0; qb < QBLOCKS; ++qb) {
                int acc = 0;
                for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
                    int q_packed = pdmq_pack_i8x4(
                        q_i8[qb * QK8_0 + w*4 + 0], q_i8[qb * QK8_0 + w*4 + 1],
                        q_i8[qb * QK8_0 + w*4 + 2], q_i8[qb * QK8_0 + w*4 + 3]);
                    int dot = 0;
                    for (int b = 0; b < 4; ++b) {
                        dot += (int)pdmq_i8_from_i32(q_packed, b) *
                               (int)pdmq_i8_from_i32(h_kp[kk * (D / 4) + qb * WORDS_PER_BLOCK + w], b);
                    }
                    acc += dot;
                }
                sum += float(acc) * q_scales[qb] * __half2float(h_ks[kk * QBLOCKS + qb]);
            }
            h_ref[qr * NK + kk] = sum;
        }
    }

    float * d_q = nullptr;
    int   * d_kp = nullptr;
    half  * d_ks = nullptr;
    float * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_kp,  kp_bytes));
    CUDA_CHECK(hipMalloc(&d_ks,  ks_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_kp,  h_kp, kp_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_ks,  h_ks, ks_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pdmq_qk_probe_kernel<<<1, PDMQ_THREADS, 0, stream>>>(d_q, d_kp, d_ks, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_kp));
    CUDA_CHECK(hipFree(d_ks));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_kp); free(h_ks); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PDMQ QK probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    fprintf(stderr, "PDMQ QK probe PASSED: max_err=%f\n", max_err);
    return true;
}

// ── GQA group selector ────────────────────────────────────────────
static inline int ggml_cuda_rocm_packed16_dot4_mmq_gqa_group() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP");
    if (!s || !*s) return 1;
    const int g = atoi(s);
    if (g == 1 || g == 2) return g;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP=%s; expected 1 or 2", s);
}

// ── GQA2 QK probe ───────────────────────────────────────────────
// Validates GQA2 QK mapping: two Q slots, same K tile.
// Launch: <<<1, PDMQ_THREADS, 0, stream>>>

static __global__ void pdmq_qk_probe_gqa2_kernel(
        const float * __restrict__ q_probe,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ out) {

    constexpr int GQA_GROUP = 2;
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int D  = PDMQ_D;

    __shared__ int   q_tile_i32[GQA_GROUP][BM][D / 4 + 1];
    __shared__ float q_tile_scales[GQA_GROUP][BM][D / QK8_0 + 1];

    for (int slot = 0; slot < GQA_GROUP; ++slot) {
        pdmq_quantize_q_tile<BM, D>(q_probe + slot * BM * D, sizeof(float) * D, 0, 0, 0, 0, 0, BM,
                                     q_tile_i32[slot], q_tile_scales[slot]);
    }

    for (int idx = int(threadIdx.x); idx < GQA_GROUP * BM * BN; idx += int(blockDim.x)) {
        const int slot = idx / (BM * BN);
        const int qr   = (idx / BN) % BM;
        const int kk   = idx % BN;

        const int  * k_row_payload = k_payload + size_t(kk) * (D / 4);
        const half * k_row_scales  = k_scales  + size_t(kk) * (D / QK8_0);
        float s = pdmq_qk_dot<BM, D>(q_tile_i32[slot], q_tile_scales[slot], qr, k_row_payload, k_row_scales);
        out[(slot * BM + qr) * BN + kk] = s;
    }
}

static bool pdmq_qk_probe_gqa2_pass(hipStream_t stream) {
    constexpr int GQA_GROUP = 2;
    constexpr int NQ   = PDMQ_BM_PREFILL * GQA_GROUP;
    constexpr int NK   = PDMQ_BN_PREFILL;
    constexpr int D    = PDMQ_D;
    constexpr int QBLOCKS = D / QK8_0;
    constexpr int WORDS_PER_BLOCK = QK8_0 / 4;

    constexpr size_t qk_bytes    = NQ  * D * sizeof(float);
    constexpr size_t kp_bytes    = NK  * (D / 4) * sizeof(int);
    constexpr size_t ks_bytes    = NK  * (D / QK8_0) * sizeof(half);
    constexpr size_t out_bytes   = NQ  * NK * sizeof(float);

    float * h_q  = (float *) malloc(qk_bytes);
    int   * h_kp = (int   *) malloc(kp_bytes);
    half  * h_ks = (half  *) malloc(ks_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    // Fill slot 0 and slot 1 with DIFFERENT random values to catch aliasing
    srand(615243);
    for (int slot = 0; slot < GQA_GROUP; ++slot) {
        srand(615243 + slot * 7919);
        for (size_t i = 0; i < PDMQ_BM_PREFILL * D; i++) {
            h_q[slot * PDMQ_BM_PREFILL * D + i] =
                ((float)rand() / RAND_MAX * 2.0f - 1.0f) * (slot == 0 ? 3.0f : 5.0f);
        }
    }

    srand(45678);
    for (int kk = 0; kk < NK; ++kk) {
        for (int qb = 0; qb < QBLOCKS; ++qb) {
            int8_t k_raw[QK8_0];
            float k_amax = 0.0f;
            for (int i = 0; i < QK8_0; ++i) {
                k_raw[i] = (int8_t)((rand() % 255) - 127);
                k_amax = fmaxf(k_amax, fabsf((float)k_raw[i]));
            }
            float ks = k_amax > 0.0f ? k_amax / 127.0f : 0.0f;
            h_ks[kk * QBLOCKS + qb] = __float2half(ks);
            for (int w = 0; w < WORDS_PER_BLOCK; ++w)
                h_kp[kk * (D / 4) + qb * WORDS_PER_BLOCK + w] =
                    pdmq_pack_i8x4(k_raw[w*4+0], k_raw[w*4+1], k_raw[w*4+2], k_raw[w*4+3]);
        }
    }

    // Scalar reference: same-quantized DOT4-MMQ
    for (int slot = 0; slot < GQA_GROUP; ++slot) {
        for (int qr = 0; qr < PDMQ_BM_PREFILL; ++qr) {
            float q_scales[QBLOCKS], q_invscales[QBLOCKS];
            int8_t q_i8[QBLOCKS * QK8_0];
            for (int qb = 0; qb < QBLOCKS; ++qb) {
                float amax = 0.0f;
                for (int i = 0; i < QK8_0; ++i)
                    amax = fmaxf(amax, fabsf(h_q[(slot * PDMQ_BM_PREFILL + qr) * D + qb * QK8_0 + i]));
                q_scales[qb] = amax > 0.0f ? amax / 127.0f : 0.0f;
                q_invscales[qb] = amax > 0.0f ? 127.0f / amax : 0.0f;
                for (int i = 0; i < QK8_0; ++i)
                    q_i8[qb * QK8_0 + i] = (int8_t) pdmq_clamp_i8(
                        (int) lrintf(h_q[(slot * PDMQ_BM_PREFILL + qr) * D + qb * QK8_0 + i] * q_invscales[qb]));
            }
            for (int kk = 0; kk < NK; ++kk) {
                float sum = 0.0f;
                for (int qb = 0; qb < QBLOCKS; ++qb) {
                    int acc = 0;
                    for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
                        int q_packed = pdmq_pack_i8x4(
                            q_i8[qb * QK8_0 + w*4 + 0], q_i8[qb * QK8_0 + w*4 + 1],
                            q_i8[qb * QK8_0 + w*4 + 2], q_i8[qb * QK8_0 + w*4 + 3]);
                        int dot = 0;
                        for (int b = 0; b < 4; ++b)
                            dot += (int)pdmq_i8_from_i32(q_packed, b) *
                                   (int)pdmq_i8_from_i32(h_kp[kk * (D / 4) + qb * WORDS_PER_BLOCK + w], b);
                        acc += dot;
                    }
                    sum += float(acc) * q_scales[qb] * __half2float(h_ks[kk * QBLOCKS + qb]);
                }
                h_ref[(slot * PDMQ_BM_PREFILL + qr) * NK + kk] = sum;
            }
        }
    }

    float * d_q = nullptr; int * d_kp = nullptr; half * d_ks = nullptr; float * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_kp,  kp_bytes));
    CUDA_CHECK(hipMalloc(&d_ks,  ks_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_kp,  h_kp, kp_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_ks,  h_ks, ks_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pdmq_qk_probe_gqa2_kernel<<<1, PDMQ_THREADS, 0, stream>>>(d_q, d_kp, d_ks, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    // Check for slot aliasing
    bool dup_found = false;
    for (int r1 = 0; r1 < PDMQ_BM_PREFILL && !dup_found; r1++) {
        for (int r2 = 0; r2 < PDMQ_BM_PREFILL && !dup_found; r2++) {
            bool same = true;
            for (int c = 0; c < NK && same; c++)
                if (fabsf(h_gpu[r1 * NK + c] - h_gpu[(PDMQ_BM_PREFILL + r2) * NK + c]) > 1e-4f) same = false;
            if (same) { fprintf(stderr, "PDMQ GQA2 slot ALIAS: slot0_r=%d slot1_r=%d\n", r1, r2); dup_found = true; }
        }
    }

    CUDA_CHECK(hipFree(d_q)); CUDA_CHECK(hipFree(d_kp)); CUDA_CHECK(hipFree(d_ks)); CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_kp); free(h_ks); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PDMQ GQA2 QK probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    if (dup_found) {
        fprintf(stderr, "PDMQ GQA2 QK probe FAILED: slot aliasing\n");
        return false;
    }
    fprintf(stderr, "PDMQ GQA2 QK probe PASSED: max_err=%f\n", max_err);
    return true;
}

enum pdmq_shape { PDMQ_SHAPE_M16N16, PDMQ_SHAPE_M8N32, PDMQ_SHAPE_M4N64 };

static inline const char * pdmq_shape_name(const pdmq_shape shape) {
    switch (shape) {
        case PDMQ_SHAPE_M16N16: return "16x16";
        case PDMQ_SHAPE_M8N32:  return "8x32";
        case PDMQ_SHAPE_M4N64:  return "4x64";
        default:                return "unknown";
    }
}

static inline pdmq_shape pdmq_parse_shape_env(const char * env) {
    if (strcmp(env, "16x16") == 0 || strcmp(env, "m16n16") == 0) {
        return PDMQ_SHAPE_M16N16;
    }
    if (strcmp(env, "8x32") == 0 || strcmp(env, "m8n32") == 0) {
        return PDMQ_SHAPE_M8N32;
    }
    if (strcmp(env, "4x64") == 0 || strcmp(env, "m4n64") == 0) {
        return PDMQ_SHAPE_M4N64;
    }
    GGML_ABORT("packed16_dot4_mmq: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE=%s, expected 16x16, 8x32, or 4x64", env);
}

static inline pdmq_shape pdmq_select_shape_auto(
        const int nq,
        const int nk,
        const ggml_type v_type) {
    // Attention equivalent of the MMQ max-x lesson: do not maximize tile width
    // on gfx1100 just because it exists. M4N64 is env-only/debug because even
    // without V staging it tends to add pressure and has worse Q-row reuse.
    //
    // V quantized paths have extra decode work; give them more Q rows to
    // amortize K/QK setup. F16 V is cheaper to decode, so 8x32 is more useful
    // for very small nq and long-K verification.
    if (nq >= 16) {
        return PDMQ_SHAPE_M16N16;
    }
    if (nq >= 8 && v_type != GGML_TYPE_F16) {
        return PDMQ_SHAPE_M16N16;
    }
    if (nk <= 32 && nq >= 4) {
        return PDMQ_SHAPE_M16N16;
    }
    return PDMQ_SHAPE_M8N32;
}

static void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == PDMQ_D && K->ne[0] * 4 == PDMQ_D && V->ne[0] == PDMQ_D && dst->ne[0] == PDMQ_D);
    GGML_ASSERT(Q->ne[1] > 1 && Q->ne[2] % K->ne[2] == 0);
    GGML_ASSERT(V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16);

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && logit_softcap == 0.0f);

    if (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(mask->ne[0] >= K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1);
    }

    ggml_tensor * packed16_payload = nullptr;
    ggml_tensor * packed16_scales  = nullptr;
    llama_kv_cache_get_packed16_tensors(K->data, &packed16_payload, &packed16_scales);
    GGML_ASSERT(packed16_payload && packed16_scales);
    GGML_ASSERT(packed16_payload->ne[0] == PDMQ_D / 4 && packed16_scales->ne[0] == PDMQ_D / QK8_0);
    GGML_ASSERT(packed16_payload->ne[1] >= K->ne[1] * K->ne[2] && packed16_scales->ne[1] >= K->ne[1] * K->ne[2]);
    GGML_ASSERT(packed16_payload->nb[0] == (int64_t) sizeof(int) && packed16_scales->nb[0] == (int64_t) sizeof(half));
    GGML_ASSERT(packed16_payload->nb[1] == (PDMQ_D / 4) * (int64_t) sizeof(int));
    GGML_ASSERT(packed16_scales->nb[1]  == (PDMQ_D / QK8_0) * (int64_t) sizeof(half));
    GGML_ASSERT(packed16_payload->nb[2] == packed16_payload->ne[1] * packed16_payload->nb[1]);
    GGML_ASSERT(packed16_scales->nb[2]  == packed16_scales->ne[1]  * packed16_scales->nb[1]);
    GGML_ASSERT(packed16_payload->nb[3] == packed16_payload->ne[2] * packed16_payload->nb[2]);
    GGML_ASSERT(packed16_scales->nb[3]  == packed16_scales->ne[2]  * packed16_scales->nb[2]);

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int gqa_ratio = n_heads_q / n_heads_k;
    const int batch = (int) Q->ne[3];
    const int packed_rows = (int) packed16_payload->ne[1];
    const int64_t v_ne13 = V->ne[3] > 0 ? V->ne[3] : 1;
    const float attention_scale = ((const float *) dst->op_params)[0];

    const int64_t mask_ne00 = mask ? mask->ne[0] : 0;
    const int64_t mask_ne01 = mask ? mask->ne[1] : 0;
    const int64_t mask_ne03 = mask ? mask->ne[3] : 1;
    const int64_t mask_nb00 = mask ? mask->nb[0] : 0;
    const int64_t mask_nb01 = mask ? mask->nb[1] : 0;
    const int64_t mask_nb03 = mask ? mask->nb[3] : 0;

    const bool assume_causal = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_ASSUME_CAUSAL");
        return v && atoi(v) != 0;
    }();
    const int q_offset = assume_causal ? (nk > nq ? nk - nq : 0) : 0;

    const bool shape_auto = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE_AUTO");
        return !v || atoi(v) != 0;
    }();

    pdmq_shape shape = shape_auto
        ? pdmq_select_shape_auto(nq, nk, V->type)
        : (nq >= 16 ? PDMQ_SHAPE_M16N16 : PDMQ_SHAPE_M8N32);

    if (const char * env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE")) {
        shape = pdmq_parse_shape_env(env);
    }

    const int launch_bm = shape == PDMQ_SHAPE_M16N16 ? PDMQ_BM_PREFILL : (shape == PDMQ_SHAPE_M8N32 ? PDMQ_BM_SMALL : PDMQ_BM_LONGK);
    const int launch_bn = shape == PDMQ_SHAPE_M16N16 ? PDMQ_BN_PREFILL : (shape == PDMQ_SHAPE_M8N32 ? PDMQ_BN_SMALL : PDMQ_BN_LONGK);

    const int gqa_group = ggml_cuda_rocm_packed16_dot4_mmq_gqa_group();
    const bool is_gqa2 = (gqa_group == 2);
    const bool gqa2_supported = is_gqa2 && (shape == PDMQ_SHAPE_M16N16) && (gqa_ratio >= 2) && (nq > 1);
    if (is_gqa2 && !gqa2_supported) {
        GGML_ABORT("PDMQ GQA2 requested but unsupported: shape=%s nq=%d hq=%d hk=%d gqa_ratio=%d",
                   pdmq_shape_name(shape), nq, n_heads_q, n_heads_k, gqa_ratio);
    }

    const int groups_per_kv = CEIL_DIV(gqa_ratio, 2);
    const int grid_y_gqa2  = n_heads_k * groups_per_kv;
    const int grid_y_old   = n_heads_q;

    dim3 grid(is_gqa2
        ? CEIL_DIV(nq, PDMQ_BM_PREFILL)
        : CEIL_DIV(nq, launch_bm),
        is_gqa2 ? grid_y_gqa2 : n_heads_q,
        batch);
    dim3 block(PDMQ_THREADS);
    hipStream_t stream = ctx.stream();

    const bool kshared = pdmq_kshared_enabled();
    const bool directv = kshared && !pdmq_stagev_enabled();  // direct-V = kshared with V staging off

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr,
            "PDMQ v%d route=rocm_packed16_dot4_mmq variant=%s shape=%s(%dx%d) nq=%d nk=%d hq=%d hk=%d "
            "gqa_ratio=%d group=%d grid_y_old=%d grid_y_new=%d b=%d sc=%g "
            "V=%s packed_rows=%d head_stride=%d causal=%d stage_v=%d kshared=%d directv=%d\n",
            PACKED16_DOT4_MMQ_VERSION,
            is_gqa2 ? "GQA2" : "GQA1",
            pdmq_shape_name(shape), launch_bm, launch_bn,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, gqa_group,
            grid_y_old, is_gqa2 ? grid_y_gqa2 : grid_y_old,
            batch, (double) attention_scale,
            ggml_type_name(V->type), packed_rows, packed_rows / n_heads_k, assume_causal ? 1 : 0,
            directv ? 0 : 1, kshared ? 1 : 0, directv ? 1 : 0);
        if (!pdmq_qk_probe_pass(stream)) GGML_ABORT("PDMQ QK probe failed");
        if (is_gqa2 && !pdmq_qk_probe_gqa2_pass(stream)) GGML_ABORT("PDMQ GQA2 QK probe failed");
    }}

#define PDMQ_LAUNCH_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V, KSHARED) \
    packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, KSHARED><<<grid, block, 0, stream>>>( \
        (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale)

#define PDMQ_LAUNCH_GQA2(VT, CAUSAL, kshared_var, directv_var) do { \
    if (kshared_var && !(directv_var)) { \
        packed16_dot4_mmq_gqa2_kernel<VT, CAUSAL, true, true><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } else if (kshared_var) { \
        packed16_dot4_mmq_gqa2_kernel<VT, CAUSAL, false, true><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } else if (!(directv_var)) { \
        packed16_dot4_mmq_gqa2_kernel<VT, CAUSAL, true, false><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } else { \
        packed16_dot4_mmq_gqa2_kernel<VT, CAUSAL, false, false><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, kshared_val, stagev_val) do { \
    if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, stagev_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, stagev_val, kshared_val); \
    } else { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_LONGK, PDMQ_BN_LONGK, CAUSAL, false, kshared_val); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED(VT, CAUSAL, kshared_var, directv_var) do { \
    if (kshared_var && directv_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, false); \
    } else if (kshared_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, true); \
    } else { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, true); \
    } \
} while (0)

if (is_gqa2) {
    if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0: PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0, true, kshared, directv); break;
            case GGML_TYPE_Q8_0: PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q8_0, true, kshared, directv); break;
            case GGML_TYPE_F16:  PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_F16,  true, kshared, directv); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0: PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0, false, kshared, directv); break;
            case GGML_TYPE_Q8_0: PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q8_0, false, kshared, directv); break;
            case GGML_TYPE_F16:  PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_F16,  false, kshared, directv); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    }
} else {
#define PDMQ_SWITCH_V(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0: PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL, kshared, directv); break; \
        case GGML_TYPE_Q8_0: PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q8_0, CAUSAL, kshared, directv); break; \
        case GGML_TYPE_F16:  PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_F16,  CAUSAL, kshared, directv); break; \
        default: GGML_ABORT("packed16_dot4_mmq: unsupported V type"); \
    }

    if (assume_causal) {
        PDMQ_SWITCH_V(true);
    } else {
        PDMQ_SWITCH_V(false);
    }

#undef PDMQ_SWITCH_V
}
#undef PDMQ_LAUNCH_TYPED
#undef PDMQ_LAUNCH_GQA2
#undef PDMQ_LAUNCH_SHAPE

    CUDA_CHECK(hipGetLastError());
}

#else

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    return false;
}

static inline bool ggml_cuda_packed16_dot4_mmq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
}

static void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("packed16_dot4_mmq requires HIP");
}

#endif // GGML_USE_HIP
