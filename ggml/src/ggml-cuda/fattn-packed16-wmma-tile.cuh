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
    const int8_t q     = ((const int8_t *) &packed)[byte];
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
        int q_tile, int hq, int b, int nq) {
    const int tid = threadIdx.x;
    for (int r = 0; r < BM; ++r) {
        const int q = q_tile * BM + r;
        for (int d = tid; d < D; d += blockDim.x) {
            half v = __float2half(0.0f);
            if (q < nq) {
                const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq)*q_nb02 + int64_t(q)*q_nb01;
                v = __float2half(((const float *) ptr)[d]);
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

    __shared__ half  q_tile_f16[PWMMA_BM][PWMMA_D];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM], row_l_smem[PWMMA_BM], alpha_smem[PWMMA_BM];
    __shared__ float out_smem[PWMMA_BM * PWMMA_D];

    if (threadIdx.x < PWMMA_BM) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    for (int i = threadIdx.x; i < PWMMA_BM * PWMMA_D; i += blockDim.x) out_smem[i] = 0.0f;
    __syncthreads();

    pwmma_q_load<PWMMA_BM, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq);

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
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(((const int8_t*)&packed)[byte]) * s);
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

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else { v *= attention_scale; if (mask) v += pwmma_mask_val(mask, mask_nb30, mask_nb31, mask_nb33, mask_ne33, qq, k0+c, b); }
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
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
    GGML_ASSERT(packed16_payload->ne[1] >= K->ne[1] && packed16_scales->ne[1] >= K->ne[1]);
    GGML_ASSERT(packed16_payload->nb[0] == (int64_t)sizeof(int) && packed16_scales->nb[0] == (int64_t)sizeof(half));
    GGML_ASSERT(packed16_payload->nb[1] == (PWMMA_D/4)*(int64_t)sizeof(int));
    GGML_ASSERT(packed16_scales->nb[1] == (PWMMA_D/QK8_0)*(int64_t)sizeof(half));
    // Assert compact head/batch layout
    GGML_ASSERT(packed16_payload->nb[2] == packed16_payload->ne[1] * packed16_payload->nb[1]);
    GGML_ASSERT(packed16_scales->nb[2]  == packed16_scales->ne[1]  * packed16_scales->nb[1]);
    GGML_ASSERT(packed16_payload->nb[3] == packed16_payload->ne[2] * packed16_payload->nb[2]);
    GGML_ASSERT(packed16_scales->nb[3]  == packed16_scales->ne[2]  * packed16_scales->nb[2]);
    // v0.3: single-batch only. Packed16 rows are flat [head][kv].
    GGML_ASSERT(Q->ne[3] == 1);
    GGML_ASSERT(packed16_payload->ne[1] >= K->ne[1] * K->ne[2]);
    GGML_ASSERT(packed16_scales->ne[1]  >= K->ne[1] * K->ne[2]);
    GGML_ASSERT(packed16_payload->ne[2] == 1);
    GGML_ASSERT(packed16_scales->ne[2]  == 1);
    GGML_ASSERT(packed16_payload->ne[3] == 1);
    GGML_ASSERT(packed16_scales->ne[3]  == 1);
    // Packed16 stores all heads flat in ne[1]: rows = kv_size * n_heads_k
    // ne[2] = n_stream (batch), not n_heads_k
    (void)0;

    const int * k_payload = (const int*)packed16_payload->data;
    const half * k_scales = (const half*)packed16_scales->data;

    const int nq = (int)Q->ne[1], nk = (int)K->ne[1], n_heads_q = (int)Q->ne[2], n_heads_k = (int)K->ne[2];
    const int gqa_ratio = n_heads_q / n_heads_k, batch = (int)Q->ne[3];
    const float attention_scale = ((const float*)dst->op_params)[0];
    const int packed_kv_size = (int)packed16_payload->ne[1]; // unused in v0.3 single-batch, kept for future multi-batch
    const int64_t v_ne13 = V->ne[3] > 0 ? V->ne[3] : 1;
    const int64_t mask_nb30 = mask ? mask->nb[0] : 0, mask_nb31 = mask ? mask->nb[1] : 0;
    const int64_t mask_nb33 = mask ? mask->nb[3] : 0, mask_ne33 = mask ? mask->ne[3] : 1;

    dim3 grid(CEIL_DIV(nq, PWMMA_BM), n_heads_q, batch);
    dim3 block(256);
    hipStream_t stream = ctx.stream();

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr, "PWMMA v0.3 Q4fix=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g\n",
                PWMMA_Q4_LAYOUT_FIXED, nq, nk, n_heads_q, n_heads_k, batch, (double)attention_scale);
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
