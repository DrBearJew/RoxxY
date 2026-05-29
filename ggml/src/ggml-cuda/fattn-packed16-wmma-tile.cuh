// fattn-packed16-wmma-tile.cuh — v0.3 diagnostics
//
// Packed16 I32 K → raw RDNA3 WMMA builtin → FlashAttention.
// True tiled online-softmax FA. No full f16 K cache, no nq×nk logits.
//
// Q4_0 layout: ggml standard — low nibbles qs[0..15]=d0..d15, high nibbles=d16..d31
#define PWMMA_Q4_LAYOUT_FIXED 20260529

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-packed16-wmma-builtin.cuh"

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data,
                                          ggml_tensor ** payload,
                                          ggml_tensor ** scales);
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PWMMA_D  = 256;
static constexpr int PWMMA_BM = 16;
static constexpr int PWMMA_BN = 16;

enum packed16_wmma_v_type {
    PACKED16_WMMA_V_Q4_0,
    PACKED16_WMMA_V_Q8_0,
    PACKED16_WMMA_V_F16,
};

#if defined(GGML_USE_HIP) && defined(GGML_HIP_ROCWMMA_FATTN)
#include <rocwmma/rocwmma.hpp>

// ── Explicit packed-byte extraction (no pointer-pun on int) ────
static __device__ __forceinline__ int8_t pwmma_i8_from_i32(const int packed, const int byte) {
    const uint32_t u = static_cast<uint32_t>(packed);
    return static_cast<int8_t>((u >> (8 * byte)) & 0xffu);
}

// ── Reference decode helpers (for debug checks) ──────────────────

static __device__ __forceinline__ float pwmma_decode_k(
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        size_t row, int d) {
    constexpr int I32_PER_ROW = PWMMA_D / 4;
    constexpr int SCALES_PER_ROW = PWMMA_D / QK8_0;
    const int qb = d / QK8_0, inner = d & 31;
    const int word = inner >> 2, byte = inner & 3;
    const int   packed = k_payload[row * I32_PER_ROW + qb * 8 + word];
    const int8_t q     = pwmma_i8_from_i32(packed, byte);
    return float(q) * __half2float(k_scales[row * SCALES_PER_ROW + qb]);
}

static __device__ __forceinline__ float pwmma_decode_v_q4_0(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k, int hk, int b, int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
    const int blk = d / QK4_0, in = d & (QK4_0 - 1);
    const block_q4_0 * bq = (const block_q4_0 *)(ptr + int64_t(blk)*v_nb10);
    const int iq = in & 15, shift = in >= 16 ? 4 : 0;
    return float((int(bq->qs[iq] >> shift) & 0x0f) - 8) * __half2float(bq->d);
}

static __device__ __forceinline__ float pwmma_decode_v_q8_0(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k, int hk, int b, int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
    const int blk = d / QK8_0;
    const block_q8_0 * bq = (const block_q8_0 *)(ptr + int64_t(blk)*v_nb10);
    return float(bq->qs[d & (QK8_0 - 1)]) * __half2float(bq->d);
}

static __device__ __forceinline__ float pwmma_decode_v_f16(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k, int hk, int b, int d) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
    return __half2float(*(const half *)(ptr + int64_t(d)*v_nb10));
}

// ── Mask helper ──────────────────────────────────────────────────
static __device__ __forceinline__ float pwmma_mask_val(
        const char * __restrict__ mask,
        int64_t mnb30, int64_t mnb31, int64_t mnb33, int64_t mne33,
        int q, int k, int b) {
    if (!mask) return 0.0f;
    const int mb = mne33 > 1 ? (b % mne33) : 0;
    const char * p = mask + int64_t(mb)*mnb33 + int64_t(q)*mnb31 + int64_t(k)*mnb30;
    return __half2float(*(const half *) p);
}

// ── Materializers ─────────────────────────────────────────────────

template<int BM, int D>
static __device__ __forceinline__ void pwmma_q_load(
        const float * __restrict__ Q, half * __restrict__ q_smem,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int q_tile, int hq, int b, int nq,
        float attention_scale) {
    const int tid = threadIdx.x;
#ifdef GGML_CUDA_PWMMA_DEBUG
    if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        float max_abs_q = 0.0f;
        for (int r = 0; r < BM; ++r) {
            const int q = q_tile * BM + r;
            if (q >= nq) continue;
            const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq)*q_nb02 + int64_t(q)*q_nb01;
            for (int dd = 0; dd < D; ++dd)
                max_abs_q = fmaxf(max_abs_q, fabsf(((const float *) ptr)[dd]));
        }
        if (hq == 5 || hq == 12)
            printf("PWMMA Q range hq=%d max_abs_raw=%f max_abs_scaled=%f\n",
                   hq, max_abs_q, max_abs_q * attention_scale);
    }
#endif
    for (int r = 0; r < BM; ++r) {
        const int q = q_tile * BM + r;
        for (int d = tid; d < D; d += blockDim.x) {
            half v = __float2half(0.0f);
            if (q < nq) {
                const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq)*q_nb02 + int64_t(q)*q_nb01;
                const float qf = ((const float *) ptr)[d] * attention_scale;
#ifdef GGML_CUDA_PWMMA_DEBUG
                if (!isfinite(qf)) {
                    printf("PWMMA bad Q before half q=%d hq=%d d=%d qf=%f scale=%f\n",
                           q, hq, d, qf, attention_scale);
                    asm volatile("s_trap 20");
                }
#endif
                v = __float2half(qf);
#ifdef GGML_CUDA_PWMMA_DEBUG
                if (!isfinite(__half2float(v))) {
                    printf("PWMMA bad Q half q=%d hq=%d d=%d qf=%f half=%f scale=%f\n",
                           q, hq, d, qf, __half2float(v), attention_scale);
                    asm volatile("s_trap 21");
                }
#endif
            }
            q_smem[r * D + d] = v;
        }
    }
    __syncthreads();
}

template<int BN, int D>
static __device__ __forceinline__ void pwmma_v_q4_0_load(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k_tile, int valid_rows, int hk, int b,
        half * __restrict__ v_tile) {
    const int tid = threadIdx.x, vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    for (int r = 0; r < BN; ++r) {
        if (r >= valid_rows) { for (int d = tid; d < D; d += blockDim.x) v_tile[r*D+d] = __float2half(0.0f); continue; }
        const int k = k_tile*BN + r;
        const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
        for (int d = tid; d < D; d += blockDim.x) {
            const int blk = d / QK4_0, in = d & (QK4_0 - 1);
            const block_q4_0 * bq = (const block_q4_0 *)(ptr + int64_t(blk)*v_nb10);
            const int iq = in & 15, shift = in >= 16 ? 4 : 0;
            v_tile[r*D + d] = __float2half(float((int(bq->qs[iq]>>shift)&0x0f)-8) * __half2float(bq->d));
        }
    }
    __syncthreads();
}

template<int BN, int D>
static __device__ __forceinline__ void pwmma_v_q8_0_load(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k_tile, int valid_rows, int hk, int b,
        half * __restrict__ v_tile) {
    const int tid = threadIdx.x, vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    for (int r = 0; r < BN; ++r) {
        if (r >= valid_rows) { for (int d = tid; d < D; d += blockDim.x) v_tile[r*D+d] = __float2half(0.0f); continue; }
        const int k = k_tile*BN + r;
        const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
        for (int d = tid; d < D; d += blockDim.x) {
            const int blk = d / QK8_0;
            const block_q8_0 * bq = (const block_q8_0 *)(ptr + int64_t(blk)*v_nb10);
            v_tile[r*D + d] = __float2half(float(bq->qs[d & (QK8_0 - 1)]) * __half2float(bq->d));
        }
    }
    __syncthreads();
}

template<int BN, int D>
static __device__ __forceinline__ void pwmma_v_f16_load(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k_tile, int valid_rows, int hk, int b,
        half * __restrict__ v_tile) {
    const int tid = threadIdx.x, vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    for (int r = 0; r < BN; ++r) {
        if (r >= valid_rows) { for (int d = tid; d < D; d += blockDim.x) v_tile[r*D+d] = __float2half(0.0f); continue; }
        const int k = k_tile*BN + r;
        const char * ptr = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
        for (int d = tid; d < D; d += blockDim.x)
            v_tile[r*D + d] = *(const half *)(ptr + int64_t(d)*v_nb10);
    }
    __syncthreads();
}

// ── Kernel ────────────────────────────────────────────────────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_kernel(
        const float * __restrict__ Q, const char * __restrict__ V, float * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        const char * __restrict__ mask,
        int64_t mask_nb30, int64_t mask_nb31, int64_t mask_nb33, int64_t mask_ne33,
        const int  * __restrict__ k_payload, const half * __restrict__ k_scales,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        float attention_scale) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    // k_head_base: packed16 payload has all heads flat in ne[1].
    // Registry key maps to layer.k which covers all heads.
    // Head offset = hk * nk (nk is rows per head, from K->ne[1]).
    const size_t k_head_base = size_t(hk) * size_t(nk);

#ifdef GGML_CUDA_PWMMA_ROUTE_TRAP
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("PWMMA DEVICE KERNEL ENTERED\n");
        asm volatile("s_trap 7");
    }
#endif

    __shared__ half  q_tile_f16[PWMMA_BM][PWMMA_D];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM], row_l_smem[PWMMA_BM], alpha_smem[PWMMA_BM];
    __shared__ float out_smem[PWMMA_BM * PWMMA_D];

    if (threadIdx.x < PWMMA_BM) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    for (int i = threadIdx.x; i < PWMMA_BM * PWMMA_D; i += blockDim.x) out_smem[i] = 0.0f;
    __syncthreads();

    pwmma_q_load<PWMMA_BM, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq, attention_scale);

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        // V load
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);

#ifdef GGML_CUDA_PWMMA_DEBUG
        // Diagnostic 2: V tile reference
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            float max_v_err = 0.0f;
            for (int c = 0; c < valid_k; ++c) {
                const int kk = k0 + c;
                for (int d = 0; d < PWMMA_D; ++d) {
                    float ref = 0.0f;
                    if ((V_TYPE) == PACKED16_WMMA_V_Q4_0)
                        ref = pwmma_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kk, hk, b, d);
                    else if ((V_TYPE) == PACKED16_WMMA_V_Q8_0)
                        ref = pwmma_decode_v_q8_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kk, hk, b, d);
                    else
                        ref = pwmma_decode_v_f16(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kk, hk, b, d);
                    const float got = __half2float(v_tile_f16[c][d]);
                    max_v_err = fmaxf(max_v_err, fabsf(ref - got));
                }
            }
            if (max_v_err > 1e-3f) {
                printf("PWMMA V TILE MISMATCH: max_v_err=%f block=(%d,%d,%d) k0=%d\n",
                       max_v_err, blockIdx.x, blockIdx.y, blockIdx.z, k0);
                asm volatile("s_trap 13");
            }
        }
        __syncthreads();
#endif

        // WMMA QK: raw RDNA3 builtins, B fragment from packed16 directly
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x, lane_lo = lane & 15, lane_hi = lane >> 4;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    a_frag[i] = (_Float16) q_tile_f16[lane_lo][d];
#ifdef GGML_CUDA_PWMMA_DEBUG
                    if (!isfinite(float(a_frag[i]))) {
                        printf("PWMMA AFRAG NaN hq=%d q_tile=%d lane=%d lane_lo=%d i=%d d=%d af=%f\n",
                               hq, q_tile, lane, lane_lo, i, d, float(a_frag[i]));
                        asm volatile("s_trap 33");
                    }
#endif
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
#ifdef GGML_CUDA_PWMMA_DEBUG
                        if (!isfinite(float(b_frag[i]))) {
                            printf("PWMMA BFRAG NaN hq=%d hk=%d k0=%d lane=%d lane_lo=%d lane_hi=%d i=%d d=%d row=%llu val=%f\n",
                                   hq, hk, k0, lane, lane_lo, lane_hi, i, d,
                                   (unsigned long long)row, float(b_frag[i]));
                            asm volatile("s_trap 32");
                        }
                        // K payload debug dump for odd lanes on affected heads
                        if ((hq == 9 || hq == 11) && blockIdx.x == 0 && blockIdx.z == 0 && k0 == 0 && d0 == 0 && i == 0)
                            printf("PWMMA KDBG hq=%d hk=%d lane_lo=%d row=%llu packed0=%08x scale0=%f packed=%08x byte=%d i8=%d\n",
                                   hq, hk, lane_lo, (unsigned long long)row,
                                   k_payload[row * (PWMMA_D / 4) + 0],
                                   __half2float(k_scales[row * (PWMMA_D / QK8_0) + 0]),
                                   packed, byte, (int)pwmma_i8_from_i32(packed, byte));
#endif
                    } else {
                        b_frag[i] = (_Float16)0.0f;
                    }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
#ifdef GGML_CUDA_PWMMA_DEBUG
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (!isfinite(acc[i])) {
                        printf("PWMMA ACC NaN hq=%d hk=%d k0=%d d0=%d lane=%d lane_lo=%d lane_hi=%d slot=%d cf=%f\n",
                               hq, hk, k0, d0, lane, lane_lo, lane_hi, i, acc[i]);
                        asm volatile("s_trap 34");
                    }
                }
#endif
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

#ifdef GGML_CUDA_PWMMA_DEBUG
        // PREMASK NaN trap: raw QK logits from WMMA, before scale/mask
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN;
            if (!isfinite(logits_f32[r][c])) {
                printf("PWMMA PREMASK QK NaN block=(%d,%d,%d) hq=%d hk=%d r=%d c=%d x=%f\n",
                       blockIdx.x, blockIdx.y, blockIdx.z, hq, hk, r, c, logits_f32[r][c]);
                asm volatile("s_trap 30");
            }
        }
        // QK scalar reference on affected heads + per-column print
        const bool debug_head = (hq == 9 || hq == 11);
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.z == 0 && debug_head) {
            float max_abs_err = 0.0f, max_ref_abs = 0.0f;
            for (int r = 0; r < PWMMA_BM; ++r) {
                const int qq = q_tile * PWMMA_BM + r;
                if (qq >= nq) continue;
                for (int c = 0; c < valid_k; ++c) {
                    const size_t row = k_head_base + size_t(k0 + c);
                    float ref = 0.0f;
                    for (int d = 0; d < PWMMA_D; ++d) {
                        ref += __half2float(q_tile_f16[r][d]) *
                               pwmma_decode_k(k_payload, k_scales, row, d);
                    }
                    const float got = logits_f32[r][c];
                    printf("PWMMA QK hq=%d hk=%d c=%d ref=%f got=%f err=%f row=%llu\n",
                           hq, hk, c, ref, got, fabsf(ref - got),
                           (unsigned long long)row);
                    max_abs_err = fmaxf(max_abs_err, fabsf(ref - got));
                    max_ref_abs = fmaxf(max_ref_abs, fabsf(ref));
                }
            }
            const float rel = max_abs_err / fmaxf(max_ref_abs, 1e-6f);
            if (max_abs_err > 0.25f && rel > 5e-3f) {
                printf("PWMMA REAL QK MISMATCH: abs=%f rel=%f ref_abs=%f block=(%d,%d,%d) k0=%d\n",
                       max_abs_err, rel, max_ref_abs, blockIdx.x, blockIdx.y, blockIdx.z, k0);
                asm volatile("s_trap 9");
            }
        }
        __syncthreads();
#endif

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else {
                if (mask) {
                    float mv = pwmma_mask_val(mask, mask_nb30, mask_nb31, mask_nb33, mask_ne33, qq, k0+c, b);
#ifdef GGML_CUDA_PWMMA_DEBUG
                    if (!isfinite(mv)) {
                        printf("PWMMA MASK NaN block=(%d,%d,%d) hq=%d hk=%d qq=%d k=%d r=%d c=%d mv=%f "
                               "mnb30=%lld mnb31=%lld mnb33=%lld mne33=%lld\n",
                               blockIdx.x, blockIdx.y, blockIdx.z,
                               hq, hk, qq, k0+c, r, c, mv,
                               (long long)mask_nb30, (long long)mask_nb31,
                               (long long)mask_nb33, (long long)mask_ne33);
                        asm volatile("s_trap 31");
                    }
#endif
                    v += mv;
                }
            }
            logits_f32[r][c] = v;
        }
        __syncthreads();

#ifdef GGML_CUDA_PWMMA_DEBUG
        // Range trap: bad logits (valid rows only)
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM + r;
            if (qq >= nq) continue;
            const float x = logits_f32[r][c];
            if (!isfinite(x) || fabsf(x) > 1.0e4f) {
                printf("PWMMA bad logit block=(%d,%d,%d) r=%d c=%d x=%f (qq=%d valid_k=%d)\n",
                       blockIdx.x, blockIdx.y, blockIdx.z, r, c, x, qq, valid_k);
                asm volatile("s_trap 10");
            }
        }
        __syncthreads();

        // Row state trap after softmax
        for (int r = threadIdx.x; r < PWMMA_BM; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM + r;
            if (qq < nq) {
                const float m = row_m_smem[r], l = row_l_smem[r];
                if (!isfinite(m) || !isfinite(l) || l <= 0.0f || l > float(nk) * 4.0f) {
                    printf("PWMMA bad row state block=(%d,%d,%d) r=%d m=%f l=%f\n",
                           blockIdx.x, blockIdx.y, blockIdx.z, r, m, l);
                    asm volatile("s_trap 11");
                }
            }
        }
        __syncthreads();
#endif

        // Alpha scale + PV
        for (int r = threadIdx.x; r < PWMMA_BM; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM + r;
            if (qq >= nq) { alpha_smem[r] = 0.0f; continue; }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) { const float p = expf(logits_f32[r][c] - new_m); probs_f32[r][c] = p; p_sum += p; }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();

#ifdef GGML_CUDA_PWMMA_DEBUG
        // Row state trap after softmax
        for (int r = threadIdx.x; r < PWMMA_BM; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM + r;
            if (qq < nq) {
                const float m = row_m_smem[r], l = row_l_smem[r];
                if (!isfinite(m) || !isfinite(l) || l <= 0.0f || l > float(nk) * 4.0f) {
                    printf("PWMMA bad row state block=(%d,%d,%d) r=%d m=%f l=%f\n",
                           blockIdx.x, blockIdx.y, blockIdx.z, r, m, l);
                    asm volatile("s_trap 11");
                }
            }
        }
        __syncthreads();
#endif

        // Alpha scale + PV
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_D; idx += blockDim.x) out_smem[idx] *= alpha_smem[idx / PWMMA_D];
        __syncthreads();
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_D; idx += blockDim.x) {
            const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM + r;
            if (qq >= nq) continue;
            float acc = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) acc += probs_f32[r][c] * __half2float(v_tile_f16[c][d]);
            out_smem[idx] += acc;
        }
        __syncthreads();
    }

    // Final write
    for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_D; idx += blockDim.x) {
        const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM + r;
        if (qq >= nq) continue;
        const float l = row_l_smem[r]; if (l <= 0.0f) continue;
        const float val = out_smem[r * PWMMA_D + d] / l;
        dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = val;
    }
}

// ── Host launcher ─────────────────────────────────────────────────
static void ggml_cuda_flash_attn_ext_packed16_wmma_tile(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];

    GGML_ASSERT(Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == 256 && K->ne[0]*4 == Q->ne[0] && V->ne[0] == Q->ne[0]);
    GGML_ASSERT(Q->ne[1] > 1 && Q->ne[2] % K->ne[2] == 0);
    GGML_ASSERT(V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16);

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias, (const float*)dst->op_params+1, sizeof(float));
    memcpy(&logit_softcap, (const float*)dst->op_params+2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && logit_softcap == 0.0f);

    ggml_tensor * packed16_payload = nullptr, * packed16_scales = nullptr;
    llama_kv_cache_get_packed16_tensors(K->data, &packed16_payload, &packed16_scales);
    GGML_ASSERT(packed16_payload && packed16_scales);
    GGML_ASSERT(packed16_payload->ne[0] == PWMMA_D/4 && packed16_scales->ne[0] == PWMMA_D/QK8_0);
    GGML_ASSERT(packed16_payload->ne[1] >= K->ne[1] * K->ne[2] && packed16_scales->ne[1] >= K->ne[1] * K->ne[2]);
    GGML_ASSERT(packed16_payload->nb[0] == (int64_t)sizeof(int) && packed16_scales->nb[0] == (int64_t)sizeof(half));
    GGML_ASSERT(packed16_payload->nb[1] == (PWMMA_D/4)*(int64_t)sizeof(int));
    GGML_ASSERT(packed16_scales->nb[1] == (PWMMA_D/QK8_0)*(int64_t)sizeof(half));
    // Assert compact head/batch layout (required for k_head_base arithmetic)
    GGML_ASSERT(packed16_payload->nb[2] == packed16_payload->ne[1] * packed16_payload->nb[1]);
    GGML_ASSERT(packed16_scales->nb[2]  == packed16_scales->ne[1]  * packed16_scales->nb[1]);
    GGML_ASSERT(packed16_payload->nb[3] == packed16_payload->ne[2] * packed16_payload->nb[2]);
    GGML_ASSERT(packed16_scales->nb[3]  == packed16_scales->ne[2]  * packed16_scales->nb[2]);
    // Packed16 stores all heads flat in ne[1]: rows = kv_size * n_heads_k
    // ne[2] = n_stream (batch), not n_heads_k
    (void)0;

    const int * k_payload = (const int*)packed16_payload->data;
    const half * k_scales = (const half*)packed16_scales->data;

    const int nq = (int)Q->ne[1], nk = (int)K->ne[1], n_heads_q = (int)Q->ne[2], n_heads_k = (int)K->ne[2];
    const int gqa_ratio = n_heads_q / n_heads_k, batch = (int)Q->ne[3];
    const float attention_scale = ((const float*)dst->op_params)[0];
    const int packed_kv_size = (int)packed16_payload->ne[1]; // total flat rows (all heads)
    const int packed_batch   = (int)packed16_payload->ne[3];
    const int64_t v_ne13 = V->ne[3] > 0 ? V->ne[3] : 1;
    const int64_t mask_nb30 = mask ? mask->nb[0] : 0, mask_nb31 = mask ? mask->nb[1] : 0;
    const int64_t mask_nb33 = mask ? mask->nb[3] : 0, mask_ne33 = mask ? mask->ne[3] : 1;

    dim3 grid(CEIL_DIV(nq, PWMMA_BM), n_heads_q, batch);
    dim3 block(256);
    hipStream_t stream = ctx.stream();

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr, "PWMMA v0.3 Q4fix=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g "
                "payload_ne1=%lld payload_ne2=%lld packed_kv_size=%d\n",
                PWMMA_Q4_LAYOUT_FIXED, nq, nk, n_heads_q, n_heads_k, batch, (double)attention_scale,
                (long long)packed16_payload->ne[1], (long long)packed16_payload->ne[2], packed_kv_size);
        if (!pbwmma_qk_probe_pass(stream)) GGML_ABORT("PBWMMA QK probe failed");
    }}

#define LAUNCH(VT) \
    packed16_wmma_tile_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        mask ? (const char*)mask->data : nullptr, mask_nb30, mask_nb31, mask_nb33, mask_ne33, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, attention_scale)

    switch (V->type) {
        case GGML_TYPE_Q4_0: LAUNCH(PACKED16_WMMA_V_Q4_0); break;
        case GGML_TYPE_Q8_0: LAUNCH(PACKED16_WMMA_V_Q8_0); break;
        case GGML_TYPE_F16:  LAUNCH(PACKED16_WMMA_V_F16);  break;
        default: GGML_ABORT("pwmma: unsupported V type");
    }
#undef LAUNCH
    CUDA_CHECK(hipGetLastError());
}

#else
static void ggml_cuda_flash_attn_ext_packed16_wmma_tile(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("packed16_wmma_tile requires HIP + GGML_HIP_ROCWMMA_FATTN");
}
#endif
