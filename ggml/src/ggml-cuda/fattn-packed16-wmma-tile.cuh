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

// Layout modes for V tensor: FA = [D,n_kv,heads,batch], TRANS = [n_kv,heads,D,batch]
enum pwmma_v_layout {
    PWMMA_V_LAYOUT_FA    = 0,
    PWMMA_V_LAYOUT_TRANS = 1,
};

#if defined(GGML_USE_HIP) && defined(GGML_HIP_ROCWMMA_FATTN)
#include <rocwmma/rocwmma.hpp>

// ── Packed16 debug dump kernel ──────────────────────────────────
static __global__ void pwmma_packed16_dump_kernel(
        const int  * __restrict__ payload,
        const half * __restrict__ scales,
        int packed_rows, int nk, int n_heads_k) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    printf("PWMMA DEVICE DUMP: payload=%p scales=%p packed_rows=%d nk=%d n_heads_k=%d\n",
           payload, scales, packed_rows, nk, n_heads_k);
    const int head_stride = packed_rows / n_heads_k;
    for (int hk = 0; hk < n_heads_k; ++hk) {
        const int r_nk    = hk * nk;
        const int r_stride = hk * head_stride;
        printf("PWMMA hk=%d row_by_nk=%d p=%08x s=%g | row_by_stride=%d p=%08x s=%g\n",
               hk, r_nk,
               r_nk    < packed_rows ? payload[r_nk    * (PWMMA_D/4) + 0] : 0,
               r_nk    < packed_rows ? (double)__half2float(scales[r_nk    * (PWMMA_D/QK8_0) + 0]) : 0.0,
               r_stride,
               r_stride < packed_rows ? payload[r_stride * (PWMMA_D/4) + 0] : 0,
               r_stride < packed_rows ? (double)__half2float(scales[r_stride * (PWMMA_D/QK8_0) + 0]) : 0.0);
    }
}

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
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        int q, int k, int b) {
    if (!mask) return 0.0f;
    if (q < 0 || q >= mask_ne01 || k < 0 || k >= mask_ne00) return -INFINITY;
    const int mb = mask_ne03 > 1 ? (b % mask_ne03) : 0;
    const half hv = *(const half *)(mask + int64_t(mb)*mask_nb03 + int64_t(q)*mask_nb01 + int64_t(k)*mask_nb00);
    const float v = __half2float(hv);
    return isfinite(v) ? v : -INFINITY;
}

// ── Materializers ─────────────────────────────────────────────────

template<int BM, int D>
static __device__ __forceinline__ void pwmma_q_load(
        const float * __restrict__ Q, half * __restrict__ q_smem,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int q_tile, int hq, int b, int nq,
        float attention_scale) {
    const int tid = threadIdx.x;
    for (int r = 0; r < BM; ++r) {
        const int q = q_tile * BM + r;
        for (int d = tid; d < D; d += blockDim.x) {
            half v = __float2half(0.0f);
            if (q < nq) {
                const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq)*q_nb02 + int64_t(q)*q_nb01;
                const float qf = ((const float *) ptr)[d] * attention_scale;
                v = __float2half(qf);
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
        int64_t v_nb13, int64_t v_ne13, int v_layout, int k_tile, int valid_rows, int hk, int b,
        half * __restrict__ v_tile) {
    const int tid = threadIdx.x, vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    for (int r = 0; r < BN; ++r) {
        if (r >= valid_rows) { for (int d = tid; d < D; d += blockDim.x) v_tile[r*D+d] = __float2half(0.0f); continue; }
        const int k = k_tile*BN + r;
        for (int d = tid; d < D; d += blockDim.x) {
            const char * p;
            if (v_layout == PWMMA_V_LAYOUT_FA) {
                p = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11 + int64_t(d)*v_nb10;
            } else {
                p = V + int64_t(vb)*v_nb13 + int64_t(d)*v_nb12  + int64_t(hk)*v_nb11 + int64_t(k)*v_nb10;
            }
            v_tile[r*D + d] = *(const half *)p;
        }
    }
    __syncthreads();
}

// ── Kernel ────────────────────────────────────────────────────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_kernel(
        const float * __restrict__ Q, const char * __restrict__ V, float * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int v_layout,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload, const half * __restrict__ k_scales,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows,
        float attention_scale) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    // packed16 payload has all heads flat in ne[1].
    // Physical stride per head = packed_rows / n_heads_k (may differ from logical nk).
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);



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
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);


        // WMMA QK: raw RDNA3 builtins, B fragment from packed16 directly
        // Zero-init logits before WMMA to prevent stale shared-mem NaN
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
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
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else {
                        b_frag[i] = (_Float16)0.0f;
                    }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask — track NaN birth
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM + r;
            const float before = logits_f32[r][c];
            float v = before;
            float mv = 0.0f;
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) {
                mv = pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
                v += mv;
            }
            logits_f32[r][c] = v;
        }
        __syncthreads();

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

    // V layout detection
    const bool v_layout_fa =
        V->ne[0] == Q->ne[0] &&
        V->ne[1] == K->ne[1] &&
        V->ne[2] == K->ne[2];
    const bool v_layout_trans =
        V->type == GGML_TYPE_F16 &&
        V->ne[0] == K->ne[1] &&
        V->ne[1] == K->ne[2] &&
        V->ne[2] == Q->ne[0];

    int v_layout = -1;
    if (v_layout_fa) {
        v_layout = PWMMA_V_LAYOUT_FA;
        GGML_ASSERT(V->nb[0] == (int64_t)ggml_type_size(V->type));
    } else if (v_layout_trans) {
        v_layout = PWMMA_V_LAYOUT_TRANS;
        GGML_ASSERT(V->type == GGML_TYPE_F16);
        GGML_ASSERT(V->nb[0] == (int64_t)ggml_type_size(V->type));
        GGML_ASSERT(V->ne[2] == Q->ne[0]);
    } else {
        GGML_ABORT("PWMMA unsupported V layout: ne=(%lld,%lld,%lld,%lld) nb=(%lld,%lld,%lld,%lld) QD=%lld nk=%lld hk=%lld",
            (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2], (long long)V->ne[3],
            (long long)V->nb[0], (long long)V->nb[1], (long long)V->nb[2], (long long)V->nb[3],
            (long long)Q->ne[0], (long long)K->ne[1], (long long)K->ne[2]);
    }
    fprintf(stderr, "PWMMA V layout=%s ne=(%lld,%lld,%lld,%lld) nb=(%lld,%lld,%lld,%lld)\n",
        v_layout == PWMMA_V_LAYOUT_FA ? "FA" : "TRANS",
        (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2], (long long)V->ne[3],
        (long long)V->nb[0], (long long)V->nb[1], (long long)V->nb[2], (long long)V->nb[3]);

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
    const int64_t mask_ne00 = mask ? mask->ne[0] : 0;
    const int64_t mask_ne01 = mask ? mask->ne[1] : 0;
    const int64_t mask_ne03 = mask ? mask->ne[3] : 1;
    const int64_t mask_nb00 = mask ? mask->nb[0] : 0;
    const int64_t mask_nb01 = mask ? mask->nb[1] : 0;
    const int64_t mask_nb03 = mask ? mask->nb[3] : 0;

    dim3 grid(CEIL_DIV(nq, PWMMA_BM), n_heads_q, batch);
    dim3 block(256);
    hipStream_t stream = ctx.stream();

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr, "PWMMA v0.3 Q4fix=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g "
                "payload_ne1=%lld payload_ne2=%lld packed_kv_size=%d head_stride=%d\n",
                PWMMA_Q4_LAYOUT_FIXED, nq, nk, n_heads_q, n_heads_k, batch, (double)attention_scale,
                (long long)packed16_payload->ne[1], (long long)packed16_payload->ne[2],
                packed_kv_size, packed_kv_size / n_heads_k);
        // Dump packed16 via device kernel (host can't deref device ptrs)
        if (getenv("GGML_CUDA_PWMMA_DUMP_PACKED16")) {
            pwmma_packed16_dump_kernel<<<1, 1, 0, stream>>>(
                k_payload, k_scales, packed_kv_size, nk, n_heads_k);
            CUDA_CHECK(hipGetLastError());
            CUDA_CHECK(hipStreamSynchronize(stream));
        }
        if (!pbwmma_qk_probe_pass(stream)) GGML_ABORT("PBWMMA QK probe failed");
    }}

#define LAUNCH(VT) \
    packed16_wmma_tile_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale)

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
