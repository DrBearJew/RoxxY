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

enum packed16_dot4_mmq_v_type {
    PACKED16_DOT4_MMQ_V_Q4_0,
    PACKED16_DOT4_MMQ_V_Q8_0,
    PACKED16_DOT4_MMQ_V_F16,
};

#ifdef GGML_USE_HIP

static inline bool ggml_cuda_packed16_dot4_mmq_route_required() {
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    return required &&
        (strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
         strcmp(required, "packed16_dot4_mmq") == 0);
}

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    // DOT4-MMQ is opt-in unless explicitly route-required. This prevents it from
    // stealing packed16-WMMA while the WMMA mask route is still being stabilized.
    {
        const char * v = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
        if (!v) v = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
        if (v && atoi(v) == 0) return false;
    }
    {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ");
        if (v && atoi(v) == 0) return false;
        if (v && atoi(v) != 0) return true;
    }
    return ggml_cuda_packed16_dot4_mmq_route_required();
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

static __device__ __forceinline__ int8_t pdmq_i8_from_i32(const int packed, const int byte) {
    const uint32_t u = static_cast<uint32_t>(packed);
    return static_cast<int8_t>((u >> (8 * byte)) & 0xffu);
}

static __device__ __forceinline__ int pdmq_pack_i8x4(const int q0, const int q1, const int q2, const int q3) {
    const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(q0));
    const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(q1));
    const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(q2));
    const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(q3));
    return static_cast<int>((b0 << 0) | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

static __device__ __forceinline__ int pdmq_clamp_i8(const int x) {
    return x < -127 ? -127 : (x > 127 ? 127 : x);
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

template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V>
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
                const size_t k_row = k_head_base + size_t(k);
                const int  * k_row_payload = k_payload + k_row * (D / 4);
                const half * k_row_scales  = k_scales  + k_row * (D / QK8_0);
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

    dim3 grid(CEIL_DIV(nq, launch_bm), n_heads_q, batch);
    dim3 block(PDMQ_THREADS);
    hipStream_t stream = ctx.stream();

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr,
            "PDMQ v%d route=rocm_packed16_dot4_mmq shape=%s(%dx%d) auto=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g "
            "V=%s packed_rows=%d head_stride=%d causal=%d stage_v=%d lds_pad=1\n",
            PACKED16_DOT4_MMQ_VERSION,
            pdmq_shape_name(shape), launch_bm, launch_bn, shape_auto ? 1 : 0,
            nq, nk, n_heads_q, n_heads_k, batch, (double) attention_scale,
            ggml_type_name(V->type), packed_rows, packed_rows / n_heads_k, assume_causal ? 1 : 0,
            shape == PDMQ_SHAPE_M4N64 ? 0 : 1);
    }}

#define PDMQ_LAUNCH_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V) \
    packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V><<<grid, block, 0, stream>>>( \
        (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale)

#define PDMQ_LAUNCH_TYPED(VT, CAUSAL) do { \
    if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, true); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, true); \
    } else { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_LONGK, PDMQ_BN_LONGK, CAUSAL, false); \
    } \
} while (0)

#define PDMQ_SWITCH_V(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0: PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL); break; \
        case GGML_TYPE_Q8_0: PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q8_0, CAUSAL); break; \
        case GGML_TYPE_F16:  PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_F16,  CAUSAL); break; \
        default: GGML_ABORT("packed16_dot4_mmq: unsupported V type"); \
    }

    if (assume_causal) {
        PDMQ_SWITCH_V(true);
    } else {
        PDMQ_SWITCH_V(false);
    }

#undef PDMQ_SWITCH_V
#undef PDMQ_LAUNCH_TYPED
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
