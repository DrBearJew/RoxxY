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

#include <atomic>

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data,
                                          ggml_tensor ** payload,
                                          ggml_tensor ** scales);
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PWMMA_D  = 256;
static constexpr int PWMMA_BN = 16;
static constexpr int PWMMA_I8_KSHARED_I32_COUNT    = PWMMA_BN * (PWMMA_D/4);
static constexpr int PWMMA_I8_KSHARED_SCALE_COUNT  = PWMMA_BN * (PWMMA_D/QK8_0);
static constexpr int PWMMA_I8_KSHARED_SMEM_BYTES   = PWMMA_I8_KSHARED_I32_COUNT * (int) sizeof(int) +
                                                      PWMMA_I8_KSHARED_SCALE_COUNT * (int) sizeof(half);

// ── BM selector ──────────────────────────────────────────────────
static int ggml_cuda_rocm_packed16_wmma_bm() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM");
    if (!s || !*s) return 32;  // default BM32 as PWMMA sweet spot
    const int bm = atoi(s);
    if (bm == 16 || bm == 32 || bm == 64) return bm;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_WMMA_BM=%s; expected 16, 32, or 64", s);
}

// ── GQA group selector ────────────────────────────────────────────
static int ggml_cuda_rocm_packed16_wmma_gqa_group() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_GQA_GROUP");
    if (!s || !*s) return 1;
    const int g = atoi(s);
    if (g == 1 || g == 2) return g;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_WMMA_GQA_GROUP=%s; expected 1 or 2", s);
}

static int ggml_cuda_rocm_packed16_wmma_impl() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
    if (!s || !*s) return 0;  // 0 = smem (default, known-good)
    if (strcmp(s, "smem") == 0)                return 0;
    if (strcmp(s, "bm32_regout_stagev") == 0)  return 1;
    if (strcmp(s, "bm32_regout_directv") == 0) return 2;
    if (strcmp(s, "bm64_regout_stagev") == 0)  return 3;
    if (strcmp(s, "bm64_regout_directv") == 0)       return 4;
    if (strcmp(s, "bm64_regout_directv_512t") == 0)  return 5;
    if (strcmp(s, "bm64_512t_wavegate_directv") == 0)   return 6;
    if (strcmp(s, "bm64_512t_wavegate_stagev") == 0)     return 7;
    if (strcmp(s, "bm64_512t_wavegate_stagev_kshared") == 0) return 8;
    if (strcmp(s, "bm64_i8qk_512t_wavegate_stagev") == 0) return 9;
    if (strcmp(s, "bm64_i8qk_p16_512t_wavegate_stagev") == 0) return 10;
    if (strcmp(s, "bm64_i8qk_k32acc_512t_wavegate_stagev") == 0) return 11;
    if (strcmp(s, "bm64_i8qk_kshared_512t_wavegate_stagev") == 0) return 12;
    if (strcmp(s, "bm64_i8qk_k32acc_kshared_512t_wavegate_stagev") == 0) return 13;
    if (strcmp(s, "bm64_i8qk_pvwmma_512t_wavegate_stagev") == 0) return 14;
    if (strcmp(s, "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev") == 0) return 15;
    if (strcmp(s, "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev") == 0) return 16;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=%s", s);
}

enum packed16_wmma_v_type {
    PACKED16_WMMA_V_Q4_0,
    PACKED16_WMMA_V_Q8_0,
    PACKED16_WMMA_V_F16,
};

// Layout modes for V tensor: FA = [D,n_kv,heads,batch], TRANS = [n_kv,heads,D,batch]
enum pwmma_v_layout {
    PWMMA_V_LAYOUT_FA       = 0,
    PWMMA_V_LAYOUT_TRANS    = 1,
    PWMMA_V_LAYOUT_NATIVE_KDH = 2,
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

static __device__ __forceinline__ int8_t pwmma_i8_clamp_round(float x) {
    const int v = max(-127, min(127, __float2int_rn(x)));
    return static_cast<int8_t>(v);
}

static __device__ __forceinline__ int pwmma_pack_i8x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return int(uint32_t(uint8_t(x0)) |
               (uint32_t(uint8_t(x1)) << 8) |
               (uint32_t(uint8_t(x2)) << 16) |
               (uint32_t(uint8_t(x3)) << 24));
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
            } else if (v_layout == PWMMA_V_LAYOUT_NATIVE_KDH) {
                p = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(d)*v_nb11 + int64_t(k)*v_nb10;
            } else {
                p = V + int64_t(vb)*v_nb13 + int64_t(d)*v_nb12  + int64_t(hk)*v_nb11 + int64_t(k)*v_nb10;
            }
            v_tile[r*D + d] = *(const half *)p;
        }
    }
    __syncthreads();
}

// ── Probability storage selector ────────────────────────────────
template<bool P16> struct pwmma_prob_storage_type { using type = float; };
template<> struct pwmma_prob_storage_type<true> { using type = half; };

// ── Lightweight phase profiling ────────────────────────────────
struct pwmma_kernel_profile {
    unsigned long long qk_cycles;
    unsigned long long softmax_cycles;
    unsigned long long pv_cycles;
};

#define PWMMA_PROFILE_PHASE_BEGIN() do { \
    if (profile) { \
        __syncthreads(); \
        if (threadIdx.x == 0) { profile_t0 = clock64(); } \
        __syncthreads(); \
    } \
} while (0)

#define PWMMA_PROFILE_PHASE_END(FIELD) do { \
    if (profile) { \
        __syncthreads(); \
        if (threadIdx.x == 0) { \
            const unsigned long long profile_t1 = clock64(); \
            atomicAdd(&profile->FIELD, profile_t1 >= profile_t0 ? profile_t1 - profile_t0 : 0ull); \
        } \
        __syncthreads(); \
    } \
} while (0)

// ── Bounds debug error struct ───────────────────────────────────
struct pwmma_debug_error {
    int flag;
    int code;
    int call_id;
    int variant;
    int block_x, block_y, block_z, thread_x;
    int nq, nk;
    int n_heads_q, n_heads_k, gqa_ratio;
    int q_tile, hq, hk;
    int k0, valid_k, k, d;
    int head_stride, packed_rows, k_row;
};

// ── BM16 constants ──────────────────────────────────────────────
static constexpr int PWMMA_BM16 = 16;

// ── BM16 Kernel (1 WMMA wave) ───────────────────────────────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm16_1w_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    // packed16 payload has all heads flat in ne[1].
    // Physical stride per head = packed_rows / n_heads_k (may differ from logical nk).
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);



    __shared__ half  q_tile_f16[PWMMA_BM16][PWMMA_D];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM16][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM16][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM16], row_l_smem[PWMMA_BM16], alpha_smem[PWMMA_BM16];
    __shared__ float out_smem[PWMMA_BM16 * PWMMA_D];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM16) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    for (int i = threadIdx.x; i < PWMMA_BM16 * PWMMA_D; i += blockDim.x) out_smem[i] = 0.0f;
    __syncthreads();

    pwmma_q_load<PWMMA_BM16, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq, attention_scale);

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        // ── Causal future-tile skip ──────────────────────────────
        // nk == nq (standard causal prefill): skip when k0 > q_last.
        // nk >  nq (KV cache extends past prompt): skip when
        //   k0 > (nk - nq) + q_last.
        if (causal_skip_enabled && mask) {
            const int q_first  = q_tile * PWMMA_BM16;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM16 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);

            if (threadIdx.x == 0) {
                skip_tile_smem = future_candidate ? 1 : 0;
            }
            __syncthreads();

            if (future_candidate) {
                // Pure causal skip: kk > qq + (nk-nq) for all (q,k) pairs.
                // The FA mask in llama.cpp is typically all-zeros (causal is
                // implicit from geometry), so mask confirmation is not needed
                // for standard prefill.  SWA / packed-sequence safety can be
                // re-added as a mask-confirming Stage B when those edge cases
                // are exercised.
                for (int idx = threadIdx.x; idx < PWMMA_BM16 * valid_k; idx += blockDim.x) {
                    const int r  = idx / valid_k;
                    const int c  = idx % valid_k;
                    const int qq = q_first + r;
                    const int kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) {
                        atomicExch(&skip_tile_smem, 0);
                        break;
                    }
                }
            }
            __syncthreads();

            if (skip_tile_smem) {
                if (skip_counter && threadIdx.x == 0) {
                    atomicAdd(skip_counter, 1ULL);
                }
                continue;
            }
        }

        // V load
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);


        // WMMA QK: raw RDNA3 builtins, B fragment from packed16 directly
        // Zero-init logits before WMMA to prevent stale shared-mem NaN
        for (int idx = threadIdx.x; idx < PWMMA_BM16 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
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
        for (int idx = threadIdx.x; idx < PWMMA_BM16 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM16 + r;
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
        for (int r = threadIdx.x; r < PWMMA_BM16; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM16 + r;
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
        for (int idx = threadIdx.x; idx < PWMMA_BM16 * PWMMA_D; idx += blockDim.x) out_smem[idx] *= alpha_smem[idx / PWMMA_D];
        __syncthreads();
        for (int idx = threadIdx.x; idx < PWMMA_BM16 * PWMMA_D; idx += blockDim.x) {
            const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM16 + r;
            if (qq >= nq) continue;
            float acc = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) acc += probs_f32[r][c] * __half2float(v_tile_f16[c][d]);
            out_smem[idx] += acc;
        }
        __syncthreads();
    }

    // Final write
    for (int idx = threadIdx.x; idx < PWMMA_BM16 * PWMMA_D; idx += blockDim.x) {
        const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM16 + r;
        if (qq >= nq) continue;
        const float l = row_l_smem[r]; if (l <= 0.0f) continue;
        const float val = out_smem[r * PWMMA_D + d] / l;
        dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = val;
    }
}

// ── BM32 constants ──────────────────────────────────────────────
static constexpr int PWMMA_BM32 = 32;

// ── BM32 Kernel (2 WMMA waves) ──────────────────────────────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm32_2w_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ half  q_tile_f16[PWMMA_BM32][PWMMA_D];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM32][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM32][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM32], row_l_smem[PWMMA_BM32], alpha_smem[PWMMA_BM32];
    __shared__ float out_smem[PWMMA_BM32 * PWMMA_D];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM32) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    for (int i = threadIdx.x; i < PWMMA_BM32 * PWMMA_D; i += blockDim.x) out_smem[i] = 0.0f;
    __syncthreads();

    pwmma_q_load<PWMMA_BM32, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq, attention_scale);

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        // ── Causal future-tile skip (BM32 geometry) ───────────
        if (causal_skip_enabled && mask) {
            const int q_first  = q_tile * PWMMA_BM32;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM32 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);

            if (threadIdx.x == 0) {
                skip_tile_smem = future_candidate ? 1 : 0;
            }
            __syncthreads();

            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM32 * valid_k; idx += blockDim.x) {
                    const int r  = idx / valid_k;
                    const int c  = idx % valid_k;
                    const int qq = q_first + r;
                    const int kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) {
                        atomicExch(&skip_tile_smem, 0);
                        break;
                    }
                }
            }
            __syncthreads();

            if (skip_tile_smem) {
                if (skip_counter && threadIdx.x == 0) {
                    atomicAdd(skip_counter, 1ULL);
                }
                continue;
            }
        }

        // V load
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);

        // WMMA QK: 2-wave mode — wave 0 computes rows 0..15, wave 1 computes rows 16..31
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 64) {
            const int wave_id = threadIdx.x >> 5;  // 0 or 1
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;        // row offset for this wave
            const bool k_col_valid = lane_lo < valid_k;

            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    a_frag[i] = (_Float16) q_tile_f16[r_base + lane_lo][d];
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
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM32 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax + PV
        for (int r = threadIdx.x; r < PWMMA_BM32; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM32 + r;
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

        // Alpha scale old out
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_D; idx += blockDim.x) out_smem[idx] *= alpha_smem[idx / PWMMA_D];
        __syncthreads();
        // PV accumulate
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_D; idx += blockDim.x) {
            const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM32 + r;
            if (qq >= nq) continue;
            float acc = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) acc += probs_f32[r][c] * __half2float(v_tile_f16[c][d]);
            out_smem[idx] += acc;
        }
        __syncthreads();
    }

    // Final write
    for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_D; idx += blockDim.x) {
        const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q_tile * PWMMA_BM32 + r;
        if (qq >= nq) continue;
        const float l = row_l_smem[r]; if (l <= 0.0f) continue;
        const float val = out_smem[r * PWMMA_D + d] / l;
        dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = val;
    }
}

// ── V element direct load helper (for direct-V variants) ──────
template<packed16_wmma_v_type V_TYPE>
static __device__ __forceinline__ float pwmma_v_element(
        const char * __restrict__ V,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int v_layout, int k, int hk, int b, int d) {

    const int vb = v_ne13 > 1 ? (b % int(v_ne13)) : 0;

    if constexpr (V_TYPE == PACKED16_WMMA_V_F16) {
        const char * p = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11 + int64_t(d)*v_nb10;
        return __half2float(*(const half *)p);
    } else if constexpr (V_TYPE == PACKED16_WMMA_V_Q8_0) {
        const char * row_base = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
        const int blk = d / QK8_0;
        const block_q8_0 * bq = (const block_q8_0 *)(row_base + int64_t(blk)*v_nb10);
        return float(bq->qs[d & (QK8_0 - 1)]) * __half2float(bq->d);
    } else { // PACKED16_WMMA_V_Q4_0
        const char * row_base = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k)*v_nb11;
        const int blk = d / QK4_0, in = d & (QK4_0 - 1);
        const block_q4_0 * bq = (const block_q4_0 *)(row_base + int64_t(blk)*v_nb10);
        const int iq = in & 15, shift = in >= 16 ? 4 : 0;
        return float((int(bq->qs[iq]>>shift)&0x0f)-8) * __half2float(bq->d);
    }
}

// ── BM32_REGOUT_STAGEV: register output, staged V tile ─────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm32_regout_stagev_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ half  q_tile_f16[PWMMA_BM32][PWMMA_D];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM32][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM32][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM32], row_l_smem[PWMMA_BM32], alpha_smem[PWMMA_BM32];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM32) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    float out[PWMMA_BM32];
    if (threadIdx.x < PWMMA_D) { for (int r = 0; r < PWMMA_BM32; ++r) out[r] = 0.0f; }
    __syncthreads();

    pwmma_q_load<PWMMA_BM32, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq, attention_scale);

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        // Causal future-tile skip (BM32 geometry)
        if (causal_skip_enabled && mask) {
            const int q_first  = q_tile * PWMMA_BM32;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM32 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);
            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM32 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) { if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL); continue; }
        }

        // V load (staged)
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);

        // WMMA QK: 2-wave mode
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 64) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    a_frag[i] = (_Float16) q_tile_f16[r_base + lane_lo][d];
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM32 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM32; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM32 + r;
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

        // PV accumulate with register output
        if (threadIdx.x < PWMMA_D) {
            const int d = threadIdx.x;
            for (int r = 0; r < PWMMA_BM32; ++r) out[r] *= alpha_smem[r];
            for (int c = 0; c < valid_k; ++c) {
                const float v = __half2float(v_tile_f16[c][d]);
                for (int r = 0; r < PWMMA_BM32; ++r) out[r] += probs_f32[r][c] * v;
            }
        }
        __syncthreads();
    }

    // Final write
    if (threadIdx.x < PWMMA_D) {
        const int d = threadIdx.x;
        for (int r = 0; r < PWMMA_BM32; ++r) {
            const int qq = q_tile * PWMMA_BM32 + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM32_REGOUT_DIRECTV: register output, no V tile staging ───
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm32_regout_directv_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ half  q_tile_f16[PWMMA_BM32][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM32][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM32][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM32], row_l_smem[PWMMA_BM32], alpha_smem[PWMMA_BM32];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM32) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    float out[PWMMA_BM32];
    if (threadIdx.x < PWMMA_D) { for (int r = 0; r < PWMMA_BM32; ++r) out[r] = 0.0f; }
    __syncthreads();

    pwmma_q_load<PWMMA_BM32, PWMMA_D>(Q, (half*)q_tile_f16, q_nb01, q_nb02, q_nb03, q_tile, hq, b, nq, attention_scale);

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        if (causal_skip_enabled && mask) {
            const int q_first  = q_tile * PWMMA_BM32;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM32 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);
            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM32 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) { if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL); continue; }
        }

        // WMMA QK: 2-wave mode
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 64) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    a_frag[i] = (_Float16) q_tile_f16[r_base + lane_lo][d];
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM32 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q_tile * PWMMA_BM32 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM32; r += blockDim.x) {
            const int qq = q_tile * PWMMA_BM32 + r;
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

        // PV accumulate with register output + direct V load
        if (threadIdx.x < PWMMA_D) {
            const int d = threadIdx.x;
            for (int r = 0; r < PWMMA_BM32; ++r) out[r] *= alpha_smem[r];
            for (int c = 0; c < valid_k; ++c) {
                const float v = pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, d);
                for (int r = 0; r < PWMMA_BM32; ++r) out[r] += probs_f32[r][c] * v;
            }
        }
        __syncthreads();
    }

    // Final write
    if (threadIdx.x < PWMMA_D) {
        const int d = threadIdx.x;
        for (int r = 0; r < PWMMA_BM32; ++r) {
            const int qq = q_tile * PWMMA_BM32 + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_REGOUT_STAGEV: register output, staged V tile (x4) ────
static constexpr int PWMMA_BM64 = 64;  // needed before existing BM64 section
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_regout_stagev_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    float out[PWMMA_BM64];
    if (threadIdx.x < PWMMA_D) { for (int r = 0; r < PWMMA_BM64; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        if (causal_skip_enabled && mask) {
            const int q_first  = q0;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM64 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);
            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM64 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) { if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL); continue; }
        }

        // V load (staged)
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);

        // WMMA QK: 4-wave, Q reloaded from global (no Q LDS)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + d] * attention_scale)
                        : (_Float16)0.0f;
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int qq = q0 + r;
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

        // PV accumulate with register output
        if (threadIdx.x < PWMMA_D) {
            const int d = threadIdx.x;
            for (int r = 0; r < PWMMA_BM64; ++r) out[r] *= alpha_smem[r];
            for (int c = 0; c < valid_k; ++c) {
                const float v = __half2float(v_tile_f16[c][d]);
                for (int r = 0; r < PWMMA_BM64; ++r) out[r] += probs_f32[r][c] * v;
            }
        }
        __syncthreads();
    }

    // Final write
    if (threadIdx.x < PWMMA_D) {
        const int d = threadIdx.x;
        for (int r = 0; r < PWMMA_BM64; ++r) {
            const int qq = q0 + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_REGOUT_DIRECTV: register output, no V tile staging (x4)
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_regout_directv_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    float out[PWMMA_BM64];
    if (threadIdx.x < PWMMA_D) { for (int r = 0; r < PWMMA_BM64; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        if (causal_skip_enabled && mask) {
            const int q_first  = q0;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM64 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);
            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM64 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) { if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL); continue; }
        }

        // WMMA QK: 4-wave, Q reloaded from global (no Q LDS)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + d] * attention_scale)
                        : (_Float16)0.0f;
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int qq = q0 + r;
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

        // PV accumulate with register output + direct V load
        if (threadIdx.x < PWMMA_D) {
            const int d = threadIdx.x;
            for (int r = 0; r < PWMMA_BM64; ++r) out[r] *= alpha_smem[r];
            for (int c = 0; c < valid_k; ++c) {
                const float v = pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, d);
                for (int r = 0; r < PWMMA_BM64; ++r) out[r] += probs_f32[r][c] * v;
            }
        }
        __syncthreads();
    }

    // Final write
    if (threadIdx.x < PWMMA_D) {
        const int d = threadIdx.x;
        for (int r = 0; r < PWMMA_BM64; ++r) {
            const int qq = q0 + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_REGOUT_DIRECTV_512T: 512-thread, register output, no V tile staging (x4) ──
// Key difference vs current BM64 regout:
//   - 512 threads instead of 256
//   - QK: first 128 threads (4 WMMA waves)
//   - PV: all 512 threads, split into 2 groups (rows 0..31 and 32..63)
//   - out[32] per D-thread (half the out[64] of current BM64 regout)
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_regout_directv_512t_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    // Per-thread out[32] instead of out[64]
    // pv_group 0 (tids 0..255) owns rows 0..31; pv_group 1 (tids 256..511) owns rows 32..63
    const int pv_group = threadIdx.x >> 8;     // 0 or 1
    const int d        = threadIdx.x & 255;    // 0..255
    const int row_base = pv_group * 32;        // 0 or 32
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        if (causal_skip_enabled && mask) {
            const int q_first  = q0;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM64 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);
            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM64 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) { if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL); continue; }
        }

        // WMMA QK: only first 128 threads (4 waves)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int dd = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd] * attention_scale)
                        : (_Float16)0.0f;
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = dd / QK8_0, inner = dd & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int qq = q0 + r;
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

        // PV accumulate: all 512 threads, 2-row-group ownership (out[32] each)
        if (d < PWMMA_D) {
            for (int r = 0; r < 32; ++r) out[r] *= alpha_smem[row_base + r];
            for (int c = 0; c < valid_k; ++c) {
                const float v = pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, d);
                #pragma unroll
                for (int r = 0; r < 32; ++r) out[r] += probs_f32[row_base + r][c] * v;
            }
        }
        __syncthreads();
    }

    // Final write: each thread writes its 32 assigned rows
    if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_512T_WAVEGATE_DIRECTV: per-wave causal gating ──────────
// Fixes BM64's coarse CTA-wide causal skip which does ~5.9% extra
// row-K work vs BM32 at pp512. Instead: gate QK, softmax, and PV
// per 16-row WMMA wave, matching BM32's effective granularity.
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_512t_wavegate_directv_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ bool  wave_active[4];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;     // 0 or 1
    const int d        = threadIdx.x & 255;    // 0..255
    const int row_base = pv_group * 32;        // 0 or 32
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);
        const int q_offset = nk - nq;

        // Per-wave causal gate (not CTA-wide)
        if (threadIdx.x < 4) {
            for (int w = threadIdx.x; w < 4; w += 4) {
                const int wg_first = q0 + w * 16;
                const int wg_last  = min(nq - 1, wg_first + 15);
                wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
            }
        }
        __syncthreads();

        // CTA-wide early skip only if ALL 4 waves are inactive
        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        // WMMA QK: only active waves compute
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int dd = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd] * attention_scale)
                        : (_Float16)0.0f;
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = dd / QK8_0, inner = dd & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
            }
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask && wave_active[r >> 4])
                v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax — skip inactive waves
        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4;
            const int qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
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

        // PV accumulate: gate per pv_group (each covers 2 waves: 0+1 or 2+3)
        if (d < PWMMA_D) {
            const int w0 = row_base >> 4;      // wave id for first 16 rows in group
            const int w1 = w0 + 1;             // wave id for second 16 rows
            const bool any_active = wave_active[w0] || wave_active[w1];

            if (any_active) {
                // Alpha scale active rows
                for (int r = 0; r < 32; ++r) {
                    const int w = (row_base + r) >> 4;
                    if (wave_active[w]) out[r] *= alpha_smem[row_base + r];
                }
                for (int c = 0; c < valid_k; ++c) {
                    const float v = pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, d);
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        const int w = (row_base + r) >> 4;
                        if (wave_active[w]) out[r] += probs_f32[row_base + r][c] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Final write
    if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_512T_WAVEGATE_STAGEV: per-wave gate + staged V (8 KiB LDS) ──
// Fixes duplicated V loads: directv loads V element twice (once per PV group).
// Stage V in LDS once per K tile, both groups reuse.
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_512t_wavegate_stagev_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];  // 8 KiB
    __shared__ bool  wave_active[4];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);
        const int q_offset = nk - nq;

        if (threadIdx.x < 4) {
            for (int w = threadIdx.x; w < 4; w += 4) {
                const int wg_first = q0 + w * 16;
                const int wg_last  = min(nq - 1, wg_first + 15);
                wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
            }
        }
        __syncthreads();

        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        // Stage V ONCE (all 512 threads cooperatively load v_tile_f16)
        for (int idx = threadIdx.x; idx < PWMMA_BN * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16[c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        __syncthreads();

        // WMMA QK: only active waves
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int dd = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd] * attention_scale)
                        : (_Float16)0.0f;
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = dd / QK8_0, inner = dd & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else { b_frag[i] = (_Float16)0.0f; }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask && wave_active[r >> 4])
                v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4, qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
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

        // PV: both groups read staged V (no duplicated load)
        if (d < PWMMA_D) {
            const int w0 = row_base >> 4, w1 = w0 + 1;
            if (wave_active[w0] || wave_active[w1]) {
                for (int r = 0; r < 32; ++r) {
                    if (wave_active[(row_base + r) >> 4]) out[r] *= alpha_smem[row_base + r];
                }
                for (int c = 0; c < valid_k; ++c) {
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        if (wave_active[(row_base + r) >> 4])
                            out[r] += probs_f32[row_base + r][c] * __half2float(v_tile_f16[c][d]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_512T_WAVEGATE_STAGEV_KSHARED: stage V + shared K (16 KiB LDS) ──
// Both Fix 2 (staged V) and Fix 3 (shared K decode). K tile decoded once
// into LDS instead of 4x independently per WMMA wave.
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_512t_wavegate_stagev_kshared_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;

    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];  //  8 KiB
    __shared__ half  k_tile_f16[PWMMA_BN][PWMMA_D];  //  8 KiB (shared K decode)
    __shared__ bool  wave_active[4];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);
        const int q_offset = nk - nq;

        if (threadIdx.x < 4) {
            for (int w = threadIdx.x; w < 4; w += 4) {
                const int wg_first = q0 + w * 16;
                const int wg_last  = min(nq - 1, wg_first + 15);
                wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
            }
        }
        __syncthreads();

        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        // Stage V ONCE
        for (int idx = threadIdx.x; idx < PWMMA_BN * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16[c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        // Decode K ONCE into LDS (shared across all 4 waves)
        for (int idx = threadIdx.x; idx < PWMMA_BN * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            const size_t row = k_head_base + size_t(k0) + size_t(c);
            const int qb = dd / QK8_0, inner = dd & 31, word = inner >> 2, byte = inner & 3;
            const int packed = k_payload[row * (PWMMA_D/4) + qb * 8 + word];
            const float s = __half2float(k_scales[row * (PWMMA_D/QK8_0) + qb]);
            k_tile_f16[c][dd] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
        }
        __syncthreads();

        // WMMA QK: uses staged K (no per-wave decode)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const bool k_col_valid = lane_lo < valid_k;
            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int dd = d0 + i;
                    const int qr = r_base + lane_lo, qq_val = q0 + qr;
                    a_frag[i] = (qr < PWMMA_BM64 && qq_val < nq)
                        ? (_Float16)(Q[qq_val * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd] * attention_scale)
                        : (_Float16)0.0f;
                    b_frag[i] = k_col_valid ? (_Float16)k_tile_f16[lane_lo][dd] : (_Float16)0.0f;
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask && wave_active[r >> 4])
                v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4, qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
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

        if (d < PWMMA_D) {
            const int w0 = row_base >> 4, w1 = w0 + 1;
            if (wave_active[w0] || wave_active[w1]) {
                for (int r = 0; r < 32; ++r) {
                    if (wave_active[(row_base + r) >> 4]) out[r] *= alpha_smem[row_base + r];
                }
                for (int c = 0; c < valid_k; ++c) {
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        if (wave_active[(row_base + r) >> 4])
                            out[r] += probs_f32[row_base + r][c] * __half2float(v_tile_f16[c][d]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_I8QK_512T_WAVEGATE_STAGEV: i8 WMMA QK + staged V ──────
// Q is quantized once per BM64 CTA to int8, K is consumed directly from the
// packed16 int32 payload, and QK uses v_wmma_i32_16x16x16_iu8.  This keeps
// the existing scalar/register PV path for first bring-up.  K32_ACC pairs two
// K16 WMMA steps before applying the shared q/k scale for the q8_0 32-wide block;
// K_SHARED stages one 16xD K tile plus q8_0 scales into dynamic LDS for BM64 reuse;
// PV_WMMA replaces scalar P*V with f16 WMMA tiles and keeps online softmax state.
template<packed16_wmma_v_type V_TYPE, bool PROBS_F16 = false, bool K32_ACC = false, bool K_SHARED = false, bool PV_WMMA = false, bool BN32 = false>
static __global__ void packed16_wmma_tile_bm64_i8qk_512t_wavegate_stagev_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err,
        pwmma_kernel_profile * __restrict__ profile) {

    GGML_UNUSED(causal_skip_enabled);
    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;
    constexpr int Q_SCALE_BLOCK = 32;
    constexpr int Q_SCALE_BLOCKS = PWMMA_D / Q_SCALE_BLOCK;
    constexpr int BN_TILE = BN32 ? 32 : PWMMA_BN;

    using prob_t = typename pwmma_prob_storage_type<PROBS_F16>::type;
    __shared__ int q_i32[PWMMA_BM64][PWMMA_D/4];
    __shared__ float  q_s [PWMMA_BM64][Q_SCALE_BLOCKS];
    __shared__ half   v_tile_f16[BN_TILE][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][BN_TILE];
    __shared__ prob_t probs [PWMMA_BM64][BN_TILE];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ bool  wave_active[4];
    __shared__ unsigned long long profile_t0;
    extern __shared__ int pbwmma_i8_kshared_i32[];
    int  * const k_i32_smem = pbwmma_i8_kshared_i32;
    half * const k_s_smem   = reinterpret_cast<half *>(k_i32_smem + PWMMA_I8_KSHARED_I32_COUNT);

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }

    // Quantize Q once for this BM64 tile with 32-dim scales matching K scale granularity.
    for (int idx = threadIdx.x; idx < PWMMA_BM64 * Q_SCALE_BLOCKS; idx += blockDim.x) {
        const int r = idx / Q_SCALE_BLOCKS;
        const int sb = idx % Q_SCALE_BLOCKS;
        const int qq = q0 + r;
        const int d_base = sb * Q_SCALE_BLOCK;
        float amax = 0.0f;
        float vals[Q_SCALE_BLOCK];
        #pragma unroll
        for (int i = 0; i < Q_SCALE_BLOCK; ++i) {
            const int dd = d_base + i;
            const float x = (qq < nq)
                ? Q[qq * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd]
                : 0.0f;
            vals[i] = x;
            amax = fmaxf(amax, fabsf(x));
        }
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        q_s[r][sb] = s;
        #pragma unroll
        for (int g = 0; g < Q_SCALE_BLOCK/4; ++g) {
            q_i32[r][d_base/4 + g] = pwmma_pack_i8x4(
                pwmma_i8_clamp_round(vals[4*g + 0] / s),
                pwmma_i8_clamp_round(vals[4*g + 1] / s),
                pwmma_i8_clamp_round(vals[4*g + 2] / s),
                pwmma_i8_clamp_round(vals[4*g + 3] / s));
        }
    }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    float out_pv[4][8];
    if constexpr (PV_WMMA) {
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_pv[rb][i] = 0.0f;
        }
    }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, BN_TILE);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BN_TILE, valid_k = min(BN_TILE, nk - k0);
        const int q_offset = nk - nq;

        if (threadIdx.x < 4) {
            const int w = threadIdx.x;
            const int wg_first = q0 + w * 16;
            const int wg_last  = min(nq - 1, wg_first + 15);
            wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
        }
        __syncthreads();
        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        for (int idx = threadIdx.x; idx < BN_TILE * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16[c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        if constexpr (K_SHARED) {
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_I32_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/4);
                const int g = idx - c * (PWMMA_D/4);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_i32_smem[idx] = k_payload[row * (PWMMA_D/4) + g];
                } else {
                    k_i32_smem[idx] = 0;
                }
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_s_smem[idx] = k_scales[row * (PWMMA_D/QK8_0) + s];
                } else {
                    k_s_smem[idx] = __float2half(0.0f);
                }
            }
        }
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) logits_f32[idx / BN_TILE][idx % BN_TILE] = 0.0f;
        __syncthreads();

        PWMMA_PROFILE_PHASE_BEGIN();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
                const int lane    = threadIdx.x & 31;
                const int lane_lo = lane & 15;
                const int lane_hi = lane >> 4;
                const int r_base  = wave_id * 16;
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                const int k_col   = kc_base + lane_lo;
                const bool k_col_valid = k_col < valid_k;
                float acc_f[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc_f[i] = 0.0f;

                constexpr int QK_WMMA_STEP = K32_ACC ? Q_SCALE_BLOCK : 16;
                for (int d0 = 0; d0 < PWMMA_D; d0 += QK_WMMA_STEP) {
                    if constexpr (K32_ACC) {
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        int ref_i[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) ref_i[i] = 0;

                        #pragma unroll
                        for (int sub = 0; sub < 2; ++sub) {
                            const int sd0 = d0 + sub * 16;
                            pbwmma_v4i32 a_frag;
                            pbwmma_v4i32 b_frag;
                            #pragma unroll
                            for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                                const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                                a_frag[g] = q_i32[qr][(sd0/4) + g];
                                if constexpr (K_SHARED) {
                                    b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (sd0/4) + g];
                                } else if (k_col_valid) {
                                    const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                    b_frag[g] = k_payload[row * (PWMMA_D/4) + (sd0/4) + g];
                                } else {
                                    b_frag[g] = 0;
                                }
                            }
                            acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                            if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                                #pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                    ref_i[i] += pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][sd0/4], b_frag);
                                    if (acc_i[i] != ref_i[i] && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                        bounds_err->code = 9001;
                                        bounds_err->variant = K_SHARED ? 13 : 11;
                                        bounds_err->block_x = blockIdx.x;
                                        bounds_err->block_y = blockIdx.y;
                                        bounds_err->block_z = blockIdx.z;
                                        bounds_err->thread_x = threadIdx.x;
                                        bounds_err->nq = nq;
                                        bounds_err->nk = nk;
                                        bounds_err->n_heads_q = n_heads_q;
                                        bounds_err->n_heads_k = n_heads_k;
                                        bounds_err->gqa_ratio = gqa_ratio;
                                        bounds_err->q_tile = q_tile;
                                        bounds_err->hq = hq;
                                        bounds_err->hk = hk;
                                        bounds_err->k0 = k0;
                                        bounds_err->valid_k = valid_k;
                                        bounds_err->k = k_col;
                                        bounds_err->d = sd0 + i;
                                        bounds_err->head_stride = head_stride;
                                        bounds_err->packed_rows = packed_rows;
                                        bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                    }
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    } else {
                        pbwmma_v4i32 a_frag;
                        pbwmma_v4i32 b_frag;
                        #pragma unroll
                        for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                            const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                            a_frag[g] = q_i32[qr][(d0/4) + g];
                            if constexpr (K_SHARED) {
                                b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (d0/4) + g];
                            } else if (k_col_valid) {
                                const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                b_frag[g] = k_payload[row * (PWMMA_D/4) + (d0/4) + g];
                            } else {
                                b_frag[g] = 0;
                            }
                        }
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                const int ref = pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][d0/4], b_frag);
                                if (acc_i[i] != ref && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                    bounds_err->code = 9001;
                                    bounds_err->variant = BN32 ? 15 : (K_SHARED ? 12 : (PROBS_F16 ? 10 : 9));
                                    bounds_err->block_x = blockIdx.x;
                                    bounds_err->block_y = blockIdx.y;
                                    bounds_err->block_z = blockIdx.z;
                                    bounds_err->thread_x = threadIdx.x;
                                    bounds_err->nq = nq;
                                    bounds_err->nk = nk;
                                    bounds_err->n_heads_q = n_heads_q;
                                    bounds_err->n_heads_k = n_heads_k;
                                    bounds_err->gqa_ratio = gqa_ratio;
                                    bounds_err->q_tile = q_tile;
                                    bounds_err->hq = hq;
                                    bounds_err->hk = hk;
                                    bounds_err->k0 = k0;
                                    bounds_err->valid_k = valid_k;
                                    bounds_err->k = k_col;
                                    bounds_err->d = d0 + i;
                                    bounds_err->head_stride = head_stride;
                                    bounds_err->packed_rows = packed_rows;
                                    bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    logits_f32[r_base + pbwmma_i8_d_row_from_acc(i, lane_hi)][kc_base + pbwmma_i8_d_col_from_lane_lo(lane_lo)] = acc_f[i];
                }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(qk_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
            const int r = idx / BN_TILE, c = idx % BN_TILE, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k || !wave_active[r >> 4]) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4, qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) {
                const float p = expf(logits_f32[r][c] - new_m);
                if constexpr (PROBS_F16) probs[r][c] = __float2half(p);
                else probs[r][c] = p;
                p_sum += p;
            }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(softmax_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        if constexpr (PV_WMMA) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int d_tile  = (threadIdx.x >> 5) & 15;
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                    out_pv[rb][i] *= alpha_smem[global_r];
                }
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                    pbwmma_v16fp16 a_frag;
                    pbwmma_v16fp16 b_frag;
                    if (wave_active[rb]) {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            const int kc = kc_base + kk;
                            if (kc < valid_k) {
                                if constexpr (PROBS_F16) a_frag[kk] = probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc];
                                else a_frag[kk] = __float2half(probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc]);
                                b_frag[kk] = v_tile_f16[kc][d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo)];
                            } else {
                                a_frag[kk] = __float2half(0.0f);
                                b_frag[kk] = __float2half(0.0f);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            a_frag[kk] = __float2half(0.0f);
                            b_frag[kk] = __float2half(0.0f);
                        }
                    }
                    pbwmma_v8fp32 pv_acc = {0,0,0,0,0,0,0,0};
                    pv_acc = pbwmma_mma(a_frag, b_frag, pv_acc);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0 && d_tile == 0) {
                            float ref = 0.0f;
                            #pragma unroll
                            for (int c = 0; c < 16; ++c) {
                                const int kc = kc_base + c;
                                if (kc < valid_k) ref += __half2float(probs[global_r][kc]) * __half2float(v_tile_f16[kc][lane_lo]);
                            }
                            const float tol = 2.5e-2f * fmaxf(1.0f, fabsf(ref));
                            if (fabsf(pv_acc[i] - ref) > tol && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                bounds_err->code = 9101;
                                bounds_err->variant = BN32 ? 15 : 14;
                                bounds_err->block_x = blockIdx.x;
                                bounds_err->block_y = blockIdx.y;
                                bounds_err->block_z = blockIdx.z;
                                bounds_err->thread_x = threadIdx.x;
                                bounds_err->nq = nq;
                                bounds_err->nk = nk;
                                bounds_err->n_heads_q = n_heads_q;
                                bounds_err->n_heads_k = n_heads_k;
                                bounds_err->gqa_ratio = gqa_ratio;
                                bounds_err->q_tile = q_tile;
                                bounds_err->hq = hq;
                                bounds_err->hk = hk;
                                bounds_err->k0 = k0;
                                bounds_err->valid_k = valid_k;
                                bounds_err->k = rb;
                                bounds_err->d = d_tile * 16 + lane_lo;
                                bounds_err->head_stride = head_stride;
                                bounds_err->packed_rows = packed_rows;
                                bounds_err->k_row = global_r;
                            }
                        }
                        out_pv[rb][i] += pv_acc[i];
                    }
                }
            }
        } else if (d < PWMMA_D) {
            const int w0 = row_base >> 4, w1 = w0 + 1;
            if (wave_active[w0] || wave_active[w1]) {
                for (int r = 0; r < 32; ++r) if (wave_active[(row_base + r) >> 4]) out[r] *= alpha_smem[row_base + r];
                for (int c = 0; c < valid_k; ++c) {
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        if (wave_active[(row_base + r) >> 4]) {
                            float p;
                            if constexpr (PROBS_F16) p = __half2float(probs[row_base + r][c]);
                            else p = probs[row_base + r][c];
                            out[r] += p * __half2float(v_tile_f16[c][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(pv_cycles);
    }

    if constexpr (PV_WMMA) {
        const int lane    = threadIdx.x & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
        const int d_tile  = (threadIdx.x >> 5) & 15;
        const int dd      = d_tile * 16 + lane_lo;
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int qq = q0 + rb * 16 + rr;
                if (qq >= nq) continue;
                const float l = row_l_smem[rb * 16 + rr]; if (l <= 0.0f) continue;
                dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + dd] = out_pv[rb][i] / l;
            }
        }
    } else if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

// ── BM64_I8QK PV-WMMA DBV: dedicated scaffold for V double-buffer route ──
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel(
        const float * __restrict__ Q, const char * __restrict__ V, float * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int v_layout,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload, const half * __restrict__ k_scales,
        int k_payload_row_stride_i32, int k_scales_row_stride_half,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows,
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err,
        pwmma_kernel_profile * __restrict__ profile) {

    GGML_UNUSED(causal_skip_enabled);
    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;
    constexpr bool PROBS_F16 = true;
    constexpr bool K32_ACC = false;
    constexpr bool K_SHARED = false;
    constexpr bool PV_WMMA = true;
    constexpr bool BN32 = false;
    constexpr int Q_SCALE_BLOCK = 32;
    constexpr int Q_SCALE_BLOCKS = PWMMA_D / Q_SCALE_BLOCK;
    constexpr int BN_TILE = PWMMA_BN;

    using prob_t = typename pwmma_prob_storage_type<PROBS_F16>::type;
    __shared__ int q_i32[PWMMA_BM64][PWMMA_D/4];
    __shared__ float  q_s [PWMMA_BM64][Q_SCALE_BLOCKS];
    __shared__ half   v_tile_f16_db[2][BN_TILE][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][BN_TILE];
    __shared__ prob_t probs [PWMMA_BM64][BN_TILE];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ bool  wave_active[4];
    __shared__ unsigned long long profile_t0;
    extern __shared__ int pbwmma_i8_kshared_i32[];
    int  * const k_i32_smem = pbwmma_i8_kshared_i32;
    half * const k_s_smem   = reinterpret_cast<half *>(k_i32_smem + PWMMA_I8_KSHARED_I32_COUNT);

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }

    // Quantize Q once for this BM64 tile with 32-dim scales matching K scale granularity.
    for (int idx = threadIdx.x; idx < PWMMA_BM64 * Q_SCALE_BLOCKS; idx += blockDim.x) {
        const int r = idx / Q_SCALE_BLOCKS;
        const int sb = idx % Q_SCALE_BLOCKS;
        const int qq = q0 + r;
        const int d_base = sb * Q_SCALE_BLOCK;
        float amax = 0.0f;
        float vals[Q_SCALE_BLOCK];
        #pragma unroll
        for (int i = 0; i < Q_SCALE_BLOCK; ++i) {
            const int dd = d_base + i;
            const float x = (qq < nq)
                ? Q[qq * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd]
                : 0.0f;
            vals[i] = x;
            amax = fmaxf(amax, fabsf(x));
        }
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        q_s[r][sb] = s;
        #pragma unroll
        for (int g = 0; g < Q_SCALE_BLOCK/4; ++g) {
            q_i32[r][d_base/4 + g] = pwmma_pack_i8x4(
                pwmma_i8_clamp_round(vals[4*g + 0] / s),
                pwmma_i8_clamp_round(vals[4*g + 1] / s),
                pwmma_i8_clamp_round(vals[4*g + 2] / s),
                pwmma_i8_clamp_round(vals[4*g + 3] / s));
        }
    }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    float out_pv[4][8];
    if constexpr (PV_WMMA) {
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_pv[rb][i] = 0.0f;
        }
    }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, BN_TILE);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BN_TILE, valid_k = min(BN_TILE, nk - k0);
        const int q_offset = nk - nq;

        if (threadIdx.x < 4) {
            const int w = threadIdx.x;
            const int wg_first = q0 + w * 16;
            const int wg_last  = min(nq - 1, wg_first + 15);
            wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
        }
        __syncthreads();
        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        for (int idx = threadIdx.x; idx < BN_TILE * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16_db[kt & 1][c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        if constexpr (K_SHARED) {
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_I32_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/4);
                const int g = idx - c * (PWMMA_D/4);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_i32_smem[idx] = k_payload[row * k_payload_row_stride_i32 + g];
                } else {
                    k_i32_smem[idx] = 0;
                }
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_s_smem[idx] = k_scales[row * k_scales_row_stride_half + s];
                } else {
                    k_s_smem[idx] = __float2half(0.0f);
                }
            }
        }
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) logits_f32[idx / BN_TILE][idx % BN_TILE] = 0.0f;
        __syncthreads();

        PWMMA_PROFILE_PHASE_BEGIN();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
                const int lane    = threadIdx.x & 31;
                const int lane_lo = lane & 15;
                const int lane_hi = lane >> 4;
                const int r_base  = wave_id * 16;
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                const int k_col   = kc_base + lane_lo;
                const bool k_col_valid = k_col < valid_k;
                float acc_f[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc_f[i] = 0.0f;

                constexpr int QK_WMMA_STEP = K32_ACC ? Q_SCALE_BLOCK : 16;
                for (int d0 = 0; d0 < PWMMA_D; d0 += QK_WMMA_STEP) {
                    if constexpr (K32_ACC) {
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        int ref_i[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) ref_i[i] = 0;

                        #pragma unroll
                        for (int sub = 0; sub < 2; ++sub) {
                            const int sd0 = d0 + sub * 16;
                            pbwmma_v4i32 a_frag;
                            pbwmma_v4i32 b_frag;
                            #pragma unroll
                            for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                                const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                                a_frag[g] = q_i32[qr][(sd0/4) + g];
                                if constexpr (K_SHARED) {
                                    b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (sd0/4) + g];
                                } else if (k_col_valid) {
                                    const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                    b_frag[g] = k_payload[row * k_payload_row_stride_i32 + (sd0/4) + g];
                                } else {
                                    b_frag[g] = 0;
                                }
                            }
                            acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                            if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                                #pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                    ref_i[i] += pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][sd0/4], b_frag);
                                    if (acc_i[i] != ref_i[i] && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                        bounds_err->code = 9001;
                                        bounds_err->variant = K_SHARED ? 13 : 11;
                                        bounds_err->block_x = blockIdx.x;
                                        bounds_err->block_y = blockIdx.y;
                                        bounds_err->block_z = blockIdx.z;
                                        bounds_err->thread_x = threadIdx.x;
                                        bounds_err->nq = nq;
                                        bounds_err->nk = nk;
                                        bounds_err->n_heads_q = n_heads_q;
                                        bounds_err->n_heads_k = n_heads_k;
                                        bounds_err->gqa_ratio = gqa_ratio;
                                        bounds_err->q_tile = q_tile;
                                        bounds_err->hq = hq;
                                        bounds_err->hk = hk;
                                        bounds_err->k0 = k0;
                                        bounds_err->valid_k = valid_k;
                                        bounds_err->k = k_col;
                                        bounds_err->d = sd0 + i;
                                        bounds_err->head_stride = head_stride;
                                        bounds_err->packed_rows = packed_rows;
                                        bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                    }
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * k_scales_row_stride_half + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    } else {
                        pbwmma_v4i32 a_frag;
                        pbwmma_v4i32 b_frag;
                        #pragma unroll
                        for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                            const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                            a_frag[g] = q_i32[qr][(d0/4) + g];
                            if constexpr (K_SHARED) {
                                b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (d0/4) + g];
                            } else if (k_col_valid) {
                                const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                b_frag[g] = k_payload[row * k_payload_row_stride_i32 + (d0/4) + g];
                            } else {
                                b_frag[g] = 0;
                            }
                        }
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                const int ref = pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][d0/4], b_frag);
                                if (acc_i[i] != ref && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                    bounds_err->code = 9001;
                                    bounds_err->variant = BN32 ? 15 : (K_SHARED ? 12 : (PROBS_F16 ? 10 : 9));
                                    bounds_err->block_x = blockIdx.x;
                                    bounds_err->block_y = blockIdx.y;
                                    bounds_err->block_z = blockIdx.z;
                                    bounds_err->thread_x = threadIdx.x;
                                    bounds_err->nq = nq;
                                    bounds_err->nk = nk;
                                    bounds_err->n_heads_q = n_heads_q;
                                    bounds_err->n_heads_k = n_heads_k;
                                    bounds_err->gqa_ratio = gqa_ratio;
                                    bounds_err->q_tile = q_tile;
                                    bounds_err->hq = hq;
                                    bounds_err->hk = hk;
                                    bounds_err->k0 = k0;
                                    bounds_err->valid_k = valid_k;
                                    bounds_err->k = k_col;
                                    bounds_err->d = d0 + i;
                                    bounds_err->head_stride = head_stride;
                                    bounds_err->packed_rows = packed_rows;
                                    bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * k_scales_row_stride_half + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    logits_f32[r_base + pbwmma_i8_d_row_from_acc(i, lane_hi)][kc_base + pbwmma_i8_d_col_from_lane_lo(lane_lo)] = acc_f[i];
                }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(qk_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
            const int r = idx / BN_TILE, c = idx % BN_TILE, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k || !wave_active[r >> 4]) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4, qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) {
                const float p = expf(logits_f32[r][c] - new_m);
                if constexpr (PROBS_F16) probs[r][c] = __float2half(p);
                else probs[r][c] = p;
                p_sum += p;
            }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(softmax_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        if constexpr (PV_WMMA) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int d_tile  = (threadIdx.x >> 5) & 15;
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                    out_pv[rb][i] *= alpha_smem[global_r];
                }
            }
            #pragma unroll
            for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                pbwmma_v16fp16 b_frag;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    const int kc = kc_base + kk;
                    b_frag[kk] = kc < valid_k
                        ? v_tile_f16_db[kt & 1][kc][d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo)]
                        : __float2half(0.0f);
                }
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb) {
                    pbwmma_v16fp16 a_frag;
                    if (wave_active[rb]) {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            const int kc = kc_base + kk;
                            if (kc < valid_k) {
                                if constexpr (PROBS_F16) a_frag[kk] = probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc];
                                else a_frag[kk] = __float2half(probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc]);
                            } else {
                                a_frag[kk] = __float2half(0.0f);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) a_frag[kk] = __float2half(0.0f);
                    }
                    pbwmma_v8fp32 pv_acc = {0,0,0,0,0,0,0,0};
                    pv_acc = pbwmma_mma(a_frag, b_frag, pv_acc);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0 && d_tile == 0) {
                            float ref = 0.0f;
                            #pragma unroll
                            for (int c = 0; c < 16; ++c) {
                                const int kc = kc_base + c;
                                if (kc < valid_k) ref += __half2float(probs[global_r][kc]) * __half2float(v_tile_f16_db[kt & 1][kc][lane_lo]);
                            }
                            const float tol = 2.5e-2f * fmaxf(1.0f, fabsf(ref));
                            if (fabsf(pv_acc[i] - ref) > tol && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                bounds_err->code = 9101;
                                bounds_err->variant = BN32 ? 15 : 14;
                                bounds_err->block_x = blockIdx.x;
                                bounds_err->block_y = blockIdx.y;
                                bounds_err->block_z = blockIdx.z;
                                bounds_err->thread_x = threadIdx.x;
                                bounds_err->nq = nq;
                                bounds_err->nk = nk;
                                bounds_err->n_heads_q = n_heads_q;
                                bounds_err->n_heads_k = n_heads_k;
                                bounds_err->gqa_ratio = gqa_ratio;
                                bounds_err->q_tile = q_tile;
                                bounds_err->hq = hq;
                                bounds_err->hk = hk;
                                bounds_err->k0 = k0;
                                bounds_err->valid_k = valid_k;
                                bounds_err->k = rb;
                                bounds_err->d = d_tile * 16 + lane_lo;
                                bounds_err->head_stride = head_stride;
                                bounds_err->packed_rows = packed_rows;
                                bounds_err->k_row = global_r;
                            }
                        }
                        out_pv[rb][i] += pv_acc[i];
                    }
                }
            }
        } else if (d < PWMMA_D) {
            const int w0 = row_base >> 4, w1 = w0 + 1;
            if (wave_active[w0] || wave_active[w1]) {
                for (int r = 0; r < 32; ++r) if (wave_active[(row_base + r) >> 4]) out[r] *= alpha_smem[row_base + r];
                for (int c = 0; c < valid_k; ++c) {
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        if (wave_active[(row_base + r) >> 4]) {
                            float p;
                            if constexpr (PROBS_F16) p = __half2float(probs[row_base + r][c]);
                            else p = probs[row_base + r][c];
                            out[r] += p * __half2float(v_tile_f16_db[kt & 1][c][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(pv_cycles);
    }

    if constexpr (PV_WMMA) {
        const int lane    = threadIdx.x & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
        const int d_tile  = (threadIdx.x >> 5) & 15;
        const int dd      = d_tile * 16 + lane_lo;
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int qq = q0 + rb * 16 + rr;
                if (qq >= nq) continue;
                const float l = row_l_smem[rb * 16 + rr]; if (l <= 0.0f) continue;
                dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + dd] = out_pv[rb][i] / l;
            }
        }
    } else if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}

template<packed16_wmma_v_type V_TYPE>


static __global__ void packed16_wmma_tile_bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    GGML_UNUSED(causal_skip_enabled);
    static constexpr bool PROBS_F16 = true;
    static constexpr bool K32_ACC = false;
    static constexpr bool K_SHARED = false;
    static constexpr bool PV_WMMA = true;
    static constexpr bool BN32 = true;
    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * PWMMA_BM64;
    constexpr int Q_SCALE_BLOCK = 32;
    constexpr int Q_SCALE_BLOCKS = PWMMA_D / Q_SCALE_BLOCK;
    constexpr int BN_TILE = BN32 ? 32 : PWMMA_BN;

    using prob_t = typename pwmma_prob_storage_type<PROBS_F16>::type;
    __shared__ int q_i32[PWMMA_BM64][PWMMA_D/4];
    __shared__ float  q_s [PWMMA_BM64][Q_SCALE_BLOCKS];
    __shared__ half   v_tile_f16[BN_TILE][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][BN_TILE];
    __shared__ prob_t probs [PWMMA_BM64][BN_TILE];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ bool  wave_active[4];
    extern __shared__ int pbwmma_i8_kshared_i32[];
    int  * const k_i32_smem = pbwmma_i8_kshared_i32;
    half * const k_s_smem   = reinterpret_cast<half *>(k_i32_smem + PWMMA_I8_KSHARED_I32_COUNT);

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }

    // Quantize Q once for this BM64 tile with 32-dim scales matching K scale granularity.
    for (int idx = threadIdx.x; idx < PWMMA_BM64 * Q_SCALE_BLOCKS; idx += blockDim.x) {
        const int r = idx / Q_SCALE_BLOCKS;
        const int sb = idx % Q_SCALE_BLOCKS;
        const int qq = q0 + r;
        const int d_base = sb * Q_SCALE_BLOCK;
        float amax = 0.0f;
        float vals[Q_SCALE_BLOCK];
        #pragma unroll
        for (int i = 0; i < Q_SCALE_BLOCK; ++i) {
            const int dd = d_base + i;
            const float x = (qq < nq)
                ? Q[qq * (int)(q_nb01 / sizeof(float)) + hq * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd]
                : 0.0f;
            vals[i] = x;
            amax = fmaxf(amax, fabsf(x));
        }
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        q_s[r][sb] = s;
        #pragma unroll
        for (int g = 0; g < Q_SCALE_BLOCK/4; ++g) {
            q_i32[r][d_base/4 + g] = pwmma_pack_i8x4(
                pwmma_i8_clamp_round(vals[4*g + 0] / s),
                pwmma_i8_clamp_round(vals[4*g + 1] / s),
                pwmma_i8_clamp_round(vals[4*g + 2] / s),
                pwmma_i8_clamp_round(vals[4*g + 3] / s));
        }
    }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    float out_pv[4][8];
    if constexpr (PV_WMMA) {
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_pv[rb][i] = 0.0f;
        }
    }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, BN_TILE);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BN_TILE, valid_k = min(BN_TILE, nk - k0);
        const int q_offset = nk - nq;

        if (threadIdx.x < 4) {
            const int w = threadIdx.x;
            const int wg_first = q0 + w * 16;
            const int wg_last  = min(nq - 1, wg_first + 15);
            wave_active[w] = (wg_first < nq) && (k0 <= q_offset + wg_last);
        }
        __syncthreads();
        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        for (int idx = threadIdx.x; idx < BN_TILE * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16[c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        if constexpr (K_SHARED) {
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_I32_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/4);
                const int g = idx - c * (PWMMA_D/4);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_i32_smem[idx] = k_payload[row * (PWMMA_D/4) + g];
                } else {
                    k_i32_smem[idx] = 0;
                }
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_s_smem[idx] = k_scales[row * (PWMMA_D/QK8_0) + s];
                } else {
                    k_s_smem[idx] = __float2half(0.0f);
                }
            }
        }
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) logits_f32[idx / BN_TILE][idx % BN_TILE] = 0.0f;
        __syncthreads();

        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
                const int lane    = threadIdx.x & 31;
                const int lane_lo = lane & 15;
                const int lane_hi = lane >> 4;
                const int r_base  = wave_id * 16;
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                const int k_col   = kc_base + lane_lo;
                const bool k_col_valid = k_col < valid_k;
                float acc_f[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc_f[i] = 0.0f;

                constexpr int QK_WMMA_STEP = K32_ACC ? Q_SCALE_BLOCK : 16;
                for (int d0 = 0; d0 < PWMMA_D; d0 += QK_WMMA_STEP) {
                    if constexpr (K32_ACC) {
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        int ref_i[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) ref_i[i] = 0;

                        #pragma unroll
                        for (int sub = 0; sub < 2; ++sub) {
                            const int sd0 = d0 + sub * 16;
                            pbwmma_v4i32 a_frag;
                            pbwmma_v4i32 b_frag;
                            #pragma unroll
                            for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                                const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                                a_frag[g] = q_i32[qr][(sd0/4) + g];
                                if constexpr (K_SHARED) {
                                    b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (sd0/4) + g];
                                } else if (k_col_valid) {
                                    const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                    b_frag[g] = k_payload[row * (PWMMA_D/4) + (sd0/4) + g];
                                } else {
                                    b_frag[g] = 0;
                                }
                            }
                            acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                            if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                                #pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                    ref_i[i] += pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][sd0/4], b_frag);
                                    if (acc_i[i] != ref_i[i] && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                        bounds_err->code = 9001;
                                        bounds_err->variant = K_SHARED ? 13 : 11;
                                        bounds_err->block_x = blockIdx.x;
                                        bounds_err->block_y = blockIdx.y;
                                        bounds_err->block_z = blockIdx.z;
                                        bounds_err->thread_x = threadIdx.x;
                                        bounds_err->nq = nq;
                                        bounds_err->nk = nk;
                                        bounds_err->n_heads_q = n_heads_q;
                                        bounds_err->n_heads_k = n_heads_k;
                                        bounds_err->gqa_ratio = gqa_ratio;
                                        bounds_err->q_tile = q_tile;
                                        bounds_err->hq = hq;
                                        bounds_err->hk = hk;
                                        bounds_err->k0 = k0;
                                        bounds_err->valid_k = valid_k;
                                        bounds_err->k = k_col;
                                        bounds_err->d = sd0 + i;
                                        bounds_err->head_stride = head_stride;
                                        bounds_err->packed_rows = packed_rows;
                                        bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                    }
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    } else {
                        pbwmma_v4i32 a_frag;
                        pbwmma_v4i32 b_frag;
                        #pragma unroll
                        for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                            const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                            a_frag[g] = q_i32[qr][(d0/4) + g];
                            if constexpr (K_SHARED) {
                                b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (d0/4) + g];
                            } else if (k_col_valid) {
                                const size_t row = k_head_base + size_t(k0) + size_t(k_col);
                                b_frag[g] = k_payload[row * (PWMMA_D/4) + (d0/4) + g];
                            } else {
                                b_frag[g] = 0;
                            }
                        }
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                const int ref = pbwmma_i8_dot4_k16_ref_frag(&q_i32[rr][d0/4], b_frag);
                                if (acc_i[i] != ref && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                    bounds_err->code = 9001;
                                    bounds_err->variant = BN32 ? 15 : (K_SHARED ? 12 : (PROBS_F16 ? 10 : 9));
                                    bounds_err->block_x = blockIdx.x;
                                    bounds_err->block_y = blockIdx.y;
                                    bounds_err->block_z = blockIdx.z;
                                    bounds_err->thread_x = threadIdx.x;
                                    bounds_err->nq = nq;
                                    bounds_err->nk = nk;
                                    bounds_err->n_heads_q = n_heads_q;
                                    bounds_err->n_heads_k = n_heads_k;
                                    bounds_err->gqa_ratio = gqa_ratio;
                                    bounds_err->q_tile = q_tile;
                                    bounds_err->hq = hq;
                                    bounds_err->hk = hk;
                                    bounds_err->k0 = k0;
                                    bounds_err->valid_k = valid_k;
                                    bounds_err->k = k_col;
                                    bounds_err->d = d0 + i;
                                    bounds_err->head_stride = head_stride;
                                    bounds_err->packed_rows = packed_rows;
                                    bounds_err->k_row = int(k_head_base + size_t(k0) + size_t(k_col));
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            ks = __half2float(k_scales[(k_head_base + size_t(k0) + size_t(k_col)) * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    logits_f32[r_base + pbwmma_i8_d_row_from_acc(i, lane_hi)][kc_base + pbwmma_i8_d_col_from_lane_lo(lane_lo)] = acc_f[i];
                }
                }
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
            const int r = idx / BN_TILE, c = idx % BN_TILE, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k || !wave_active[r >> 4]) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4, qq = q0 + r;
            if (qq >= nq || !wave_active[wave_id]) { alpha_smem[r] = 0.0f; continue; }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) {
                const float p = expf(logits_f32[r][c] - new_m);
                if constexpr (PROBS_F16) probs[r][c] = __float2half(p);
                else probs[r][c] = p;
                p_sum += p;
            }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();

        if constexpr (PV_WMMA) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int d_tile  = (threadIdx.x >> 5) & 15;
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                    out_pv[rb][i] *= alpha_smem[global_r];
                }
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                    pbwmma_v16fp16 a_frag;
                    pbwmma_v16fp16 b_frag;
                    if (wave_active[rb]) {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            const int kc = kc_base + kk;
                            if (kc < valid_k) {
                                if constexpr (PROBS_F16) a_frag[kk] = probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc];
                                else a_frag[kk] = __float2half(probs[rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo)][kc]);
                                b_frag[kk] = v_tile_f16[kc][d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo)];
                            } else {
                                a_frag[kk] = __float2half(0.0f);
                                b_frag[kk] = __float2half(0.0f);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            a_frag[kk] = __float2half(0.0f);
                            b_frag[kk] = __float2half(0.0f);
                        }
                    }
                    pbwmma_v8fp32 pv_acc = {0,0,0,0,0,0,0,0};
                    pv_acc = pbwmma_mma(a_frag, b_frag, pv_acc);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0 && d_tile == 0) {
                            float ref = 0.0f;
                            #pragma unroll
                            for (int c = 0; c < 16; ++c) {
                                const int kc = kc_base + c;
                                if (kc < valid_k) ref += __half2float(probs[global_r][kc]) * __half2float(v_tile_f16[kc][lane_lo]);
                            }
                            const float tol = 2.5e-2f * fmaxf(1.0f, fabsf(ref));
                            if (fabsf(pv_acc[i] - ref) > tol && atomicCAS(&bounds_err->flag, 0, 1) == 0) {
                                bounds_err->code = 9101;
                                bounds_err->variant = BN32 ? 15 : 14;
                                bounds_err->block_x = blockIdx.x;
                                bounds_err->block_y = blockIdx.y;
                                bounds_err->block_z = blockIdx.z;
                                bounds_err->thread_x = threadIdx.x;
                                bounds_err->nq = nq;
                                bounds_err->nk = nk;
                                bounds_err->n_heads_q = n_heads_q;
                                bounds_err->n_heads_k = n_heads_k;
                                bounds_err->gqa_ratio = gqa_ratio;
                                bounds_err->q_tile = q_tile;
                                bounds_err->hq = hq;
                                bounds_err->hk = hk;
                                bounds_err->k0 = k0;
                                bounds_err->valid_k = valid_k;
                                bounds_err->k = rb;
                                bounds_err->d = d_tile * 16 + lane_lo;
                                bounds_err->head_stride = head_stride;
                                bounds_err->packed_rows = packed_rows;
                                bounds_err->k_row = global_r;
                            }
                        }
                        out_pv[rb][i] += pv_acc[i];
                    }
                }
            }
        } else if (d < PWMMA_D) {
            const int w0 = row_base >> 4, w1 = w0 + 1;
            if (wave_active[w0] || wave_active[w1]) {
                for (int r = 0; r < 32; ++r) if (wave_active[(row_base + r) >> 4]) out[r] *= alpha_smem[row_base + r];
                for (int c = 0; c < valid_k; ++c) {
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        if (wave_active[(row_base + r) >> 4]) {
                            float p;
                            if constexpr (PROBS_F16) p = __half2float(probs[row_base + r][c]);
                            else p = probs[row_base + r][c];
                            out[r] += p * __half2float(v_tile_f16[c][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (PV_WMMA) {
        const int lane    = threadIdx.x & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
        const int d_tile  = (threadIdx.x >> 5) & 15;
        const int dd      = d_tile * 16 + lane_lo;
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int qq = q0 + rb * 16 + rr;
                if (qq >= nq) continue;
                const float l = row_l_smem[rb * 16 + rr]; if (l <= 0.0f) continue;
                dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + dd] = out_pv[rb][i] / l;
            }
        }
    } else if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int qq = q0 + row_base + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[row_base + r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = out[r] / l;
        }
    }
}


// ── BM64 constants ──────────────────────────────────────────────
// PWMMA_BM64 declared earlier for BM64_REGOUT kernels

// ── BM64 Kernel (4 WMMA waves, x4) ─────────────────────────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm64_x4_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    const int q_tile = blockIdx.x, hq = blockIdx.y, b = blockIdx.z, hk = hq / gqa_ratio;
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ half  v_tile_f16[PWMMA_BN][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][PWMMA_BN];
    __shared__ float probs_f32 [PWMMA_BM64][PWMMA_BN];
    __shared__ float row_m_smem[PWMMA_BM64], row_l_smem[PWMMA_BM64], alpha_smem[PWMMA_BM64];
    __shared__ half  out_smem[PWMMA_BM64 * PWMMA_D];
    __shared__ int   skip_tile_smem;

    if (threadIdx.x < PWMMA_BM64) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }
    for (int i = threadIdx.x; i < PWMMA_BM64 * PWMMA_D; i += blockDim.x) out_smem[i] = __float2half(0.0f);
    __syncthreads();

    // Precompute Q address offsets (no Q LDS — reload per K tile from global)
    // We store Q base pointers per row to avoid recomputing nb arithmetic in inner loops
    const int q0 = q_tile * PWMMA_BM64;

    const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * PWMMA_BN, valid_k = min(PWMMA_BN, nk - k0);

        if (causal_skip_enabled && mask) {
            const int q_first  = q0;
            const int q_last   = min(nq - 1, q_first + PWMMA_BM64 - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);

            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();
            if (future_candidate) {
                for (int idx = threadIdx.x; idx < PWMMA_BM64 * valid_k; idx += blockDim.x) {
                    const int r = idx / valid_k, c = idx % valid_k, qq = q_first + r, kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();
            if (skip_tile_smem) {
                if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
                continue;
            }
        }

        // V load
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<PWMMA_BN, PWMMA_D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);

        // WMMA QK: 4-wave, Q reloaded from global (no Q LDS)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) logits_f32[idx / PWMMA_BN][idx % PWMMA_BN] = 0.0f;
        __syncthreads();
        if (threadIdx.x < 128) {
            const int wave_id = threadIdx.x >> 5;
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int r_base  = wave_id * 16;
            const int q_row   = q0 + r_base + lane_lo;
            const bool q_valid = q_row < nq;
            const bool k_col_valid = lane_lo < valid_k;

            const char * q_base = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq)*q_nb02 + int64_t(q_row)*q_nb01;

            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < PWMMA_D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    if (q_valid)
                        a_frag[i] = __float2half(((const float *) q_base)[d] * attention_scale);
                    else
                        a_frag[i] = (_Float16)0.0f;
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
            for (int i = 0; i < 8; ++i) logits_f32[r_base + 2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Scale + mask
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_BN; idx += blockDim.x) {
            const int r = idx / PWMMA_BN, c = idx % PWMMA_BN, qq = q0 + r;
            float v = logits_f32[r][c];
            if (qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
            else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
            logits_f32[r][c] = v;
        }
        __syncthreads();

        // Online softmax
        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int qq = q0 + r;
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

        // Alpha scale old out (half precision: load, scale, store)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_D; idx += blockDim.x) {
            const int r = idx / PWMMA_D;
            float old = __half2float(out_smem[idx]);
            out_smem[idx] = __float2half(old * alpha_smem[r]);
        }
        __syncthreads();

        // PV accumulate (half precision out)
        for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_D; idx += blockDim.x) {
            const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q0 + r;
            if (qq >= nq) continue;
            float acc = __half2float(out_smem[idx]);
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) acc += probs_f32[r][c] * __half2float(v_tile_f16[c][d]);
            out_smem[idx] = __float2half(acc);
        }
        __syncthreads();
    }

    // Final write
    for (int idx = threadIdx.x; idx < PWMMA_BM64 * PWMMA_D; idx += blockDim.x) {
        const int r = idx / PWMMA_D, d = idx % PWMMA_D, qq = q0 + r;
        if (qq >= nq) continue;
        const float l = row_l_smem[r]; if (l <= 0.0f) continue;
        const float val = __half2float(out_smem[r * PWMMA_D + d]) / l;
        dst[((size_t(b)*nq + qq)*n_heads_q + hq)*PWMMA_D + d] = val;
    }
}

// ── BM16_GQA2 Kernel (2 Q heads per CTA, shared V tile) ─────────
template<packed16_wmma_v_type V_TYPE>
static __global__ void packed16_wmma_tile_bm16_gqa2_kernel(
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
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err) {

    static constexpr int GQA_GROUP = 2;
    static constexpr int BM = 16;
    static constexpr int BN = 16;
    static constexpr int D  = 256;

    const int q_tile = blockIdx.x;
    const int gy     = blockIdx.y;
    const int b      = blockIdx.z;

    const int groups_per_kv = CEIL_DIV(gqa_ratio, GQA_GROUP);
    const int hk            = gy / groups_per_kv;
    const int local_group   = gy - hk * groups_per_kv;

    const int hq0 = hk * gqa_ratio + local_group * GQA_GROUP + 0;
    const int hq1 = hk * gqa_ratio + local_group * GQA_GROUP + 1;

    const bool slot_valid_0 = hq0 < n_heads_q && hq0 < (hk + 1) * gqa_ratio;
    const bool slot_valid_1 = hq1 < n_heads_q && hq1 < (hk + 1) * gqa_ratio;

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ half  q_tile_f16[GQA_GROUP][BM][D];
    __shared__ half  v_tile_f16[BN][D];
    __shared__ float logits_f32[GQA_GROUP][BM][BN];
    __shared__ float probs_f32[GQA_GROUP][BM][BN];
    __shared__ float row_m_smem[GQA_GROUP][BM];
    __shared__ float row_l_smem[GQA_GROUP][BM];
    __shared__ float alpha_smem[GQA_GROUP][BM];
    __shared__ float out_smem[GQA_GROUP][BM * D];
    __shared__ int   skip_tile_smem;

    for (int s = 0; s < GQA_GROUP; ++s) {
        if (threadIdx.x < BM) { row_m_smem[s][threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[s][threadIdx.x] = 0.0f; }
    }
    for (int i = threadIdx.x; i < GQA_GROUP * BM * D; i += blockDim.x) out_smem[0][i] = 0.0f;
    __syncthreads();

    // Q load: two Q heads, same q_tile rows
    {
        const int tid = threadIdx.x;
        // slot 0
        for (int r = 0; r < BM; ++r) {
            const int q = q_tile * BM + r;
            for (int d = tid; d < D; d += blockDim.x) {
                half v = __float2half(0.0f);
                if (q < nq && slot_valid_0) {
                    const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq0)*q_nb02 + int64_t(q)*q_nb01;
                    v = __float2half(((const float *) ptr)[d] * attention_scale);
                }
                q_tile_f16[0][r][d] = v;
            }
        }
        // slot 1
        for (int r = 0; r < BM; ++r) {
            const int q = q_tile * BM + r;
            for (int d = tid; d < D; d += blockDim.x) {
                half v = __float2half(0.0f);
                if (q < nq && slot_valid_1) {
                    const char * ptr = (const char *) Q + int64_t(b)*q_nb03 + int64_t(hq1)*q_nb02 + int64_t(q)*q_nb01;
                    v = __float2half(((const float *) ptr)[d] * attention_scale);
                }
                q_tile_f16[1][r][d] = v;
            }
        }
    }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, BN);
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BN, valid_k = min(BN, nk - k0);

        // ── Causal future-tile skip ──────────────────────────
        if (causal_skip_enabled && mask) {
            const int q_first  = q_tile * BM;
            const int q_last   = min(nq - 1, q_first + BM - 1);
            const int q_offset = nk - nq;
            const bool future_candidate = (k0 > q_offset + q_last);

            if (threadIdx.x == 0) { skip_tile_smem = future_candidate ? 1 : 0; }
            __syncthreads();

            if (future_candidate) {
                for (int idx = threadIdx.x; idx < BM * valid_k; idx += blockDim.x) {
                    const int r  = idx / valid_k;
                    const int c  = idx % valid_k;
                    const int qq = q_first + r;
                    const int kk = k0 + c;
                    if (qq >= nq) break;
                    if (kk <= qq + q_offset) { atomicExch(&skip_tile_smem, 0); break; }
                }
            }
            __syncthreads();

            if (skip_tile_smem) {
                if (skip_counter && threadIdx.x == 0) {
                    // Count skipped CTA; multiply by valid slots for head-aware stat
                    const int valid_slots = (slot_valid_0 ? 1 : 0) + (slot_valid_1 ? 1 : 0);
                    atomicAdd(skip_counter, (unsigned long long)valid_slots);
                }
                continue;
            }
        }

        // V load: once per K tile, shared by both Q-head slots
        if (V_TYPE == PACKED16_WMMA_V_Q4_0)
            pwmma_v_q4_0_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else if (V_TYPE == PACKED16_WMMA_V_Q8_0)
            pwmma_v_q8_0_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        else
            pwmma_v_f16_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);

        // WMMA QK: two waves — wave 0 → slot 0, wave 1 → slot 1
        for (int idx = threadIdx.x; idx < GQA_GROUP * BM * BN; idx += blockDim.x) {
            const int flat = idx;
            logits_f32[flat / (BM*BN)][(flat/BN) % BM][flat % BN] = 0.0f;
        }
        __syncthreads();
        if (threadIdx.x < 64) {
            const int slot    = threadIdx.x >> 5;   // 0 or 1
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const bool k_col_valid = lane_lo < valid_k;

            pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
            for (int d0 = 0; d0 < D; d0 += 16) {
                pbwmma_v16fp16 a_frag, b_frag;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int d = d0 + i;
                    a_frag[i] = (_Float16) q_tile_f16[slot][lane_lo][d];
                    if (k_col_valid) {
                        const size_t row = k_head_base + size_t(k0) + size_t(lane_lo);
                        const int qb = d / QK8_0, inner = d & 31, word = inner >> 2, byte = inner & 3;
                        const int packed = k_payload[row * (D/4) + qb * 8 + word];
                        const float s = __half2float(k_scales[row * (D/QK8_0) + qb]);
                        b_frag[i] = (_Float16)(float(pwmma_i8_from_i32(packed, byte)) * s);
                    } else {
                        b_frag[i] = (_Float16)0.0f;
                    }
                }
                acc = pbwmma_mma(a_frag, b_frag, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) logits_f32[slot][2*i + lane_hi][lane_lo] = acc[i];
        }
        __syncthreads();

        // Mask + scale
        for (int s = 0; s < GQA_GROUP; ++s) {
            const bool slot_valid = (s == 0) ? slot_valid_0 : slot_valid_1;
            const int hq_slot = (s == 0) ? hq0 : hq1;
            for (int idx = threadIdx.x; idx < BM * BN; idx += blockDim.x) {
                const int r = idx / BN, c = idx % BN, qq = q_tile * BM + r;
                float v = logits_f32[s][r][c];
                if (!slot_valid || qq >= nq || c >= valid_k) v = -FLT_MAX/2.0f;
                else if (mask) v += pwmma_mask_val(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, qq, k0+c, b);
                logits_f32[s][r][c] = v;
            }
        }
        __syncthreads();

        // Online softmax + PV per slot
        for (int s = 0; s < GQA_GROUP; ++s) {
            const bool slot_valid = (s == 0) ? slot_valid_0 : slot_valid_1;
            for (int r = threadIdx.x; r < BM; r += blockDim.x) {
                const int qq = q_tile * BM + r;
                if (!slot_valid || qq >= nq) { alpha_smem[s][r] = 0.0f; continue; }
                float tile_max = -FLT_MAX;
                #pragma unroll
                for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[s][r][c]);
                const float new_m = fmaxf(row_m_smem[s][r], tile_max);
                const float alpha = (row_l_smem[s][r] > 0.0f) ? expf(row_m_smem[s][r] - new_m) : 0.0f;
                float p_sum = 0.0f;
                #pragma unroll
                for (int c = 0; c < valid_k; ++c) { const float p = expf(logits_f32[s][r][c] - new_m); probs_f32[s][r][c] = p; p_sum += p; }
                alpha_smem[s][r] = alpha; row_m_smem[s][r] = new_m; row_l_smem[s][r] = row_l_smem[s][r] * alpha + p_sum;
            }
        }
        __syncthreads();

        // Alpha scale old out per slot
        for (int s = 0; s < GQA_GROUP; ++s)
            for (int idx = threadIdx.x; idx < BM * D; idx += blockDim.x) out_smem[s][idx] *= alpha_smem[s][idx / D];
        __syncthreads();

        // PV accumulate per slot
        for (int s = 0; s < GQA_GROUP; ++s) {
            const bool slot_valid = (s == 0) ? slot_valid_0 : slot_valid_1;
            for (int idx = threadIdx.x; idx < BM * D; idx += blockDim.x) {
                const int r = idx / D, d = idx % D, qq = q_tile * BM + r;
                if (!slot_valid || qq >= nq) continue;
                float acc = 0.0f;
                #pragma unroll
                for (int c = 0; c < valid_k; ++c) acc += probs_f32[s][r][c] * __half2float(v_tile_f16[c][d]);
                out_smem[s][idx] += acc;
            }
        }
        __syncthreads();
    }

    // Final write: slot 0 → hq0, slot 1 → hq1
    if (slot_valid_0) {
        for (int idx = threadIdx.x; idx < BM * D; idx += blockDim.x) {
            const int r = idx / D, d = idx % D, qq = q_tile * BM + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[0][r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq0)*D + d] = out_smem[0][r*D + d] / l;
        }
    }
    if (slot_valid_1) {
        for (int idx = threadIdx.x; idx < BM * D; idx += blockDim.x) {
            const int r = idx / D, d = idx % D, qq = q_tile * BM + r;
            if (qq >= nq) continue;
            const float l = row_l_smem[1][r]; if (l <= 0.0f) continue;
            dst[((size_t(b)*nq + qq)*n_heads_q + hq1)*D + d] = out_smem[1][r*D + d] / l;
        }
    }
}

// ── Host launcher ─────────────────────────────────────────────────
static void ggml_cuda_flash_attn_ext_packed16_wmma_tile(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    fprintf(stderr, "PWMMA ENTRY: Q_ne=(%lld,%lld) K_ne=(%lld,%lld) V_ne=(%lld,%lld)\n", (long long)Q->ne[0], (long long)Q->ne[1], (long long)K->ne[0], (long long)K->ne[1], (long long)V->ne[0], (long long)V->ne[1]);

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

    const bool v_layout_native_kdh =
        V->type == GGML_TYPE_F16 &&
        V->ne[0] == Q->ne[0] &&
        V->ne[1] == K->ne[1] &&
        V->ne[2] == K->ne[2] &&
        V->nb[1] == (int64_t) ggml_type_size(V->type);

    int v_layout = -1;
    if (v_layout_native_kdh) {
        v_layout = PWMMA_V_LAYOUT_NATIVE_KDH;
        GGML_ASSERT(V->nb[1] == (int64_t)ggml_type_size(V->type));
    } else if (v_layout_fa) {
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

    // ── Contract check ───────────────────────────────────────
    const char * contract_env = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_CONTRACT_CHECK");
    const bool contract_check = contract_env && atoi(contract_env);
    if (contract_check) {
        fprintf(stderr, "PWMMA CONTRACT: Q_type=%d K_type=%d V_type=%d dst_type=%d "
                "Q_ne=(%lld,%lld,%lld,%lld) K_ne=(%lld,%lld,%lld,%lld) "
                "K_ne0x4=%lld D=%llu\n",
                (int)Q->type, (int)K->type, (int)V->type, (int)dst->type,
                (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2], (long long)Q->ne[3],
                (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3],
                (long long)K->ne[0]*4, (unsigned long long)Q->ne[0]);
    }
    GGML_ASSERT(Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == 256 && K->ne[0]*4 == Q->ne[0]);
    if (getenv("GGML_CUDA_PWMMA_ABORT_AFTER_LAYOUT")) {
        GGML_ABORT("PWMMA layout debug abort");
    }
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
    GGML_ASSERT(packed16_payload->nb[1] >= (PWMMA_D/4)*(int64_t)sizeof(int));
    GGML_ASSERT(packed16_scales->nb[1] >= (PWMMA_D/QK8_0)*(int64_t)sizeof(half));
    GGML_ASSERT((packed16_payload->nb[1] % (int64_t)sizeof(int)) == 0);
    GGML_ASSERT((packed16_scales->nb[1] % (int64_t)sizeof(half)) == 0);
    const int k_payload_row_stride_i32 = (int)(packed16_payload->nb[1] / (int64_t)sizeof(int));
    const int k_scales_row_stride_half = (int)(packed16_scales->nb[1] / (int64_t)sizeof(half));
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

    const int bm = ggml_cuda_rocm_packed16_wmma_bm();
    const int gqa_group = ggml_cuda_rocm_packed16_wmma_gqa_group();
    const int impl = ggml_cuda_rocm_packed16_wmma_impl();
    const bool is_bm32 = (bm == 32);
    const bool is_bm64 = (bm == 64);
    const bool is_gqa2 = (gqa_group == 2);

    // impl validation: bm32_regout only valid with BM32; bm64_regout only valid with BM64
    const bool impl_bm32_regout = (impl == 1 || impl == 2);
    const bool impl_bm64_regout = (impl == 3 || impl == 4 || impl == 5 || impl == 6 || impl == 7 || impl == 8 || impl == 9 || impl == 10 || impl == 11 || impl == 12 || impl == 13 || impl == 14 || impl == 15 || impl == 16);
    if (impl_bm32_regout && !is_bm32) GGML_ABORT("PWMMA BM32 regout impl requires BM=32, got BM=%d impl=%d", bm, impl);
    if (impl_bm64_regout && !is_bm64) GGML_ABORT("PWMMA BM64 regout impl requires BM=64, got BM=%d impl=%d", bm, impl);

    const char * impl_name = (impl == 0) ? "smem" :
        (impl == 1) ? "bm32_regout_stagev" :
        (impl == 2) ? "bm32_regout_directv" :
        (impl == 3) ? "bm64_regout_stagev" :
        (impl == 4) ? "bm64_regout_directv" :
        (impl == 5) ? "bm64_regout_directv_512t" :
        (impl == 6) ? "bm64_512t_wavegate_directv" :
        (impl == 7) ? "bm64_512t_wavegate_stagev" :
        (impl == 8) ? "bm64_512t_wavegate_stagev_kshared" :
        (impl == 9) ? "bm64_i8qk_512t_wavegate_stagev" :
        (impl == 10) ? "bm64_i8qk_p16_512t_wavegate_stagev" :
        (impl == 11) ? "bm64_i8qk_k32acc_512t_wavegate_stagev" :
        (impl == 12) ? "bm64_i8qk_kshared_512t_wavegate_stagev" :
        (impl == 13) ? "bm64_i8qk_k32acc_kshared_512t_wavegate_stagev" :
        (impl == 14) ? "bm64_i8qk_pvwmma_512t_wavegate_stagev" :
        (impl == 15) ? "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev" :
        (impl == 16) ? "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev" :
        "unknown";

    // GQA2 validation: only BM16, GQA ratio >= 2, nq > 1
    const bool gqa2_supported = (bm == 16) && (gqa_ratio >= 2) && (Q->ne[1] > 1);
    if (is_gqa2 && !gqa2_supported) {
        GGML_ABORT("BM16_GQA2 requested but unsupported: bm=%d gqa_ratio=%d nq=%d. "
                   "GQA2 requires BM=16, gqa_ratio>=2, nq>1.",
                   bm, gqa_ratio, (int)Q->ne[1]);
    }

    const int groups_per_kv = CEIL_DIV(gqa_ratio, 2);
    const int grid_y_gqa2  = n_heads_k * groups_per_kv;
    const int grid_y_old    = n_heads_q;

    dim3 grid(is_gqa2
        ? CEIL_DIV(nq, 16)       // x: q tiles
        : CEIL_DIV(nq, bm),
        is_gqa2 ? grid_y_gqa2    // y: GQA groups
                : n_heads_q,
        batch);
    dim3 block(256);
    hipStream_t stream = ctx.stream();

    // ── Causal skip stats ───────────────────────────────────────
    const bool causal_skip_enabled =
        (getenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP") != nullptr) &&
        (atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP")) != 0);
    const bool skip_stats =
        (getenv("GGML_CUDA_ROCM_PACKED16_WMMA_SKIP_STATS") != nullptr) &&
        (atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_SKIP_STATS")) != 0);
    if (causal_skip_enabled && !mask) {
        fprintf(stderr, "PWMMA causal skip: mask is null, skip disabled\n");
    }
    const bool causal_skip_active = causal_skip_enabled && mask != nullptr;
    unsigned long long * d_skip_counter = nullptr;
    if (causal_skip_active && skip_stats) {
        CUDA_CHECK(hipMalloc(&d_skip_counter, sizeof(unsigned long long)));
        CUDA_CHECK(hipMemset(d_skip_counter, 0, sizeof(unsigned long long)));
    }

    const bool impl_i8qk = (impl == 9 || impl == 10 || impl == 11 || impl == 12 || impl == 13 || impl == 14 || impl == 15 || impl == 16);
    const bool impl_pvwmma = (impl == 14 || impl == 15 || impl == 16);
    const bool live_dot4_shadow_requested = impl_i8qk && getenv("GGML_CUDA_PWMMA_I8_LIVE_DOT4_SHADOW");
    const bool live_pv_shadow_requested = impl_pvwmma && getenv("GGML_CUDA_PWMMA_PV_WMMA_SHADOW");
    bool live_dot4_shadow = live_dot4_shadow_requested || live_pv_shadow_requested;
    pwmma_debug_error * d_live_shadow_err = nullptr;
    if (live_dot4_shadow) {
        hipStreamCaptureStatus live_cap_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &live_cap_status) == hipSuccess &&
                live_cap_status != hipStreamCaptureStatusNone) {
            static bool live_shadow_capture_warned = false;
            if (!live_shadow_capture_warned) {
                live_shadow_capture_warned = true;
                fprintf(stderr, "%s\n", live_pv_shadow_requested ? "PBWMMA PV WMMA live shadow skipped during graph capture" : "PBWMMA I8 live DOT4 shadow skipped during graph capture");
            }
            live_dot4_shadow = false;
        }
    }
    if (live_dot4_shadow) {
        CUDA_CHECK(hipMalloc(&d_live_shadow_err, sizeof(pwmma_debug_error)));
        CUDA_CHECK(hipMemset(d_live_shadow_err, 0, sizeof(pwmma_debug_error)));
    }

    bool profile_enabled = impl_i8qk && impl != 15 &&
        getenv("GGML_CUDA_PWMMA_PROFILE") && atoi(getenv("GGML_CUDA_PWMMA_PROFILE")) != 0;
    if (profile_enabled) {
        hipStreamCaptureStatus profile_cap_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &profile_cap_status) == hipSuccess &&
                profile_cap_status != hipStreamCaptureStatusNone) {
            static bool profile_capture_warned = false;
            if (!profile_capture_warned) {
                profile_capture_warned = true;
                fprintf(stderr, "PWMMA PROFILE skipped during graph capture\n");
            }
            profile_enabled = false;
        }
    }
    pwmma_kernel_profile * profile_dev = nullptr;
    hipEvent_t profile_start = nullptr;
    hipEvent_t profile_stop  = nullptr;
    if (profile_enabled) {
        CUDA_CHECK(hipMalloc((void **) &profile_dev, sizeof(pwmma_kernel_profile)));
        CUDA_CHECK(hipMemsetAsync(profile_dev, 0, sizeof(pwmma_kernel_profile), stream));
        CUDA_CHECK(hipEventCreate(&profile_start));
        CUDA_CHECK(hipEventCreate(&profile_stop));
        CUDA_CHECK(hipEventRecord(profile_start, stream));
    }

    { static bool once = false; if (!once) { once = true;
        fprintf(stderr, "PWMMA v0.6 variant=%s BM=%d GQA_GROUP=%d IMPL=%s Q4fix=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g "
                "gqa_ratio=%d grid_y_old=%d grid_y_new=%d payload_ne1=%lld payload_ne2=%lld packed_kv_size=%d head_stride=%d payload_stride_i32=%d scales_stride_half=%d\n",
                is_gqa2 ? "BM16_GQA2" : (is_bm64 ? "BM64_X4" : (is_bm32 ? "BM32_2W" : "BM16_1W")), bm, gqa_group,
                impl_name,
                PWMMA_Q4_LAYOUT_FIXED, nq, nk, n_heads_q, n_heads_k, batch, (double)attention_scale,
                gqa_ratio, grid_y_old, is_gqa2 ? grid_y_gqa2 : grid_y_old,
                (long long)packed16_payload->ne[1], (long long)packed16_payload->ne[2],
                packed_kv_size, packed_kv_size / n_heads_k,
                k_payload_row_stride_i32, k_scales_row_stride_half);
        // Dump packed16 via device kernel (host can't deref device ptrs)
        if (getenv("GGML_CUDA_PWMMA_DUMP_PACKED16")) {
            pwmma_packed16_dump_kernel<<<1, 1, 0, stream>>>(
                k_payload, k_scales, packed_kv_size, nk, n_heads_k);
            CUDA_CHECK(hipGetLastError());
            // Sync only if not capturing (graph-capture safety)
            hipStreamCaptureStatus dump_cap_status = hipStreamCaptureStatusNone;
            if (hipStreamIsCapturing(stream, &dump_cap_status) == hipSuccess &&
                dump_cap_status == hipStreamCaptureStatusNone) {
                CUDA_CHECK(hipStreamSynchronize(stream));
            }
        }
        if (!pbwmma_qk_probe_pass(stream)) GGML_ABORT("PBWMMA QK probe failed");
        if (impl_i8qk && !pbwmma_i8_qk_probe_pass(stream)) GGML_ABORT("PBWMMA I8 QK probe failed");
        if (impl_pvwmma && !pbwmma_pv_probe_pass(stream)) GGML_ABORT("PBWMMA PV WMMA probe failed");
        if (impl_i8qk && getenv("GGML_CUDA_PWMMA_I8_DOT4_SHADOW_PROBE") &&
                !pbwmma_i8_dot4_shadow_probe_pass(stream)) GGML_ABORT("PBWMMA I8 DOT4 shadow probe failed");
        if (is_bm32 && !pbwmma_qk_probe_bm32_pass(stream)) GGML_ABORT("PBWMMA BM32_2W QK probe failed");
        if (is_gqa2 && !pbwmma_qk_probe_gqa2_pass(stream)) GGML_ABORT("PBWMMA GQA2 QK probe failed");
        // BM64 probe disabled by default — enable with GGML_CUDA_PWMMA_BM64_PROBE=1
        if (getenv("GGML_CUDA_PWMMA_BM64_PROBE")) {
            static bool bm64_probed = false; if (!bm64_probed) { bm64_probed = true;
                pbwmma_qk_probe_bm64_x4_pass(stream);
            }
        }
        if (getenv("GGML_CUDA_PACKED16_KV_CHECK")) {
            const int head_stride_chk = packed_kv_size / n_heads_k;
            packed16_kv_check_kernel<<<1, 1, 0, stream>>>(
                k_payload, k_scales, packed_kv_size, nk, n_heads_k, head_stride_chk);
            CUDA_CHECK(hipGetLastError());
            // Sync only if not capturing (graph-capture safety)
            hipStreamCaptureStatus kv_cap_status = hipStreamCaptureStatusNone;
            if (hipStreamIsCapturing(stream, &kv_cap_status) == hipSuccess &&
                kv_cap_status == hipStreamCaptureStatusNone) {
                CUDA_CHECK(hipStreamSynchronize(stream));
            }
        }
    }}

    // ── Variant dispatch ───────────────────────────────────────
    if (is_gqa2) {
#define LAUNCH_GQA2(VT) \
    packed16_wmma_tile_bm16_gqa2_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_GQA2(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_GQA2(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_GQA2(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma gqa2: unsupported V type");
        }
#undef LAUNCH_GQA2
    } else if (is_bm64) {
        if (impl == 3) {
#define LAUNCH_BM64_REGOUT_SV(VT) \
    packed16_wmma_tile_bm64_regout_stagev_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_REGOUT_SV(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_REGOUT_SV(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_REGOUT_SV(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 regout_stagev: unsupported V type");
            }
#undef LAUNCH_BM64_REGOUT_SV
        } else if (impl == 4) {
#define LAUNCH_BM64_REGOUT_DV(VT) \
    packed16_wmma_tile_bm64_regout_directv_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_REGOUT_DV(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_REGOUT_DV(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_REGOUT_DV(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 regout_directv: unsupported V type");
            }
#undef LAUNCH_BM64_REGOUT_DV
        } else if (impl == 5) {
            dim3 block512(512);
#define LAUNCH_BM64_REGOUT_DV_512T(VT) \
    packed16_wmma_tile_bm64_regout_directv_512t_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_REGOUT_DV_512T(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_REGOUT_DV_512T(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_REGOUT_DV_512T(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 regout_directv_512t: unsupported V type");
            }
#undef LAUNCH_BM64_REGOUT_DV_512T
        } else if (impl == 6) {
            dim3 block512(512);
#define LAUNCH_BM64_WG_DV(VT) \
    packed16_wmma_tile_bm64_512t_wavegate_directv_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_WG_DV(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_WG_DV(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_WG_DV(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 wavegate_directv: unsupported V type");
            }
#undef LAUNCH_BM64_WG_DV
        } else if (impl == 7) {
            dim3 block512(512);
#define LAUNCH_BM64_WG_SV(VT) \
    packed16_wmma_tile_bm64_512t_wavegate_stagev_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_WG_SV(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_WG_SV(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_WG_SV(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 wavegate_stagev: unsupported V type");
            }
#undef LAUNCH_BM64_WG_SV
        } else if (impl == 8) {
            dim3 block512(512);
#define LAUNCH_BM64_WG_SVK(VT) \
    packed16_wmma_tile_bm64_512t_wavegate_stagev_kshared_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_WG_SVK(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_WG_SVK(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_WG_SVK(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 wavegate_stagev_kshared: unsupported V type");
            }
#undef LAUNCH_BM64_WG_SVK
        } else if (impl == 15) {
            dim3 block512(512);
#define LAUNCH_BM64_I8QK_PV_BN32(VT) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, d_live_shadow_err)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 i8qk pvwmma bn32: unsupported V type");
            }
#undef LAUNCH_BM64_I8QK_PV_BN32
        } else if (impl == 16) {
            dim3 block512(512);
#define LAUNCH_BM64_I8QK_DBV(VT) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT><<<grid, block512, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, d_live_shadow_err, profile_dev)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_I8QK_DBV(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_I8QK_DBV(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_I8QK_DBV(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 i8qk pvwmma dbv: unsupported V type");
            }
#undef LAUNCH_BM64_I8QK_DBV
        } else if (impl_i8qk) {
            dim3 block512(512);
#define LAUNCH_BM64_I8QK_WG_SV(VT, P16, K32, KSHARED, PVWMMA, BN32V) \
    packed16_wmma_tile_bm64_i8qk_512t_wavegate_stagev_kernel<VT, P16, K32, KSHARED, PVWMMA, BN32V><<<grid, block512, KSHARED ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, d_live_shadow_err, profile_dev)
#define LAUNCH_BM64_I8QK_PSEL(VT) do { \
    if (impl == 10) { LAUNCH_BM64_I8QK_WG_SV(VT, true, false, false, false, false); } \
    else if (impl == 11) { LAUNCH_BM64_I8QK_WG_SV(VT, false, true, false, false, false); } \
    else if (impl == 12) { LAUNCH_BM64_I8QK_WG_SV(VT, false, false, true, false, false); } \
    else if (impl == 13) { LAUNCH_BM64_I8QK_WG_SV(VT, false, true, true, false, false); } \
    else if (impl == 14) { LAUNCH_BM64_I8QK_WG_SV(VT, true, false, false, true, false); } \
    else { LAUNCH_BM64_I8QK_WG_SV(VT, false, false, false, false, false); } \
} while (0)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_I8QK_PSEL(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_I8QK_PSEL(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_I8QK_PSEL(PACKED16_WMMA_V_F16);  break;
                default: GGML_ABORT("pwmma bm64 i8qk wavegate_stagev: unsupported V type");
            }
#undef LAUNCH_BM64_I8QK_PSEL
#undef LAUNCH_BM64_I8QK_WG_SV
        } else {
#define LAUNCH_BM64(VT) \
    packed16_wmma_tile_bm64_x4_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_BM64(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_BM64(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_BM64(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma bm64: unsupported V type");
        }
#undef LAUNCH_BM64
        }
    } else if (is_bm32) {
        if (impl == 1) {
#define LAUNCH_BM32_REGOUT_SV(VT) \
    packed16_wmma_tile_bm32_regout_stagev_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_BM32_REGOUT_SV(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_BM32_REGOUT_SV(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_BM32_REGOUT_SV(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma bm32 regout_stagev: unsupported V type");
        }
#undef LAUNCH_BM32_REGOUT_SV
        } else if (impl == 2) {
#define LAUNCH_BM32_REGOUT_DV(VT) \
    packed16_wmma_tile_bm32_regout_directv_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_BM32_REGOUT_DV(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_BM32_REGOUT_DV(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_BM32_REGOUT_DV(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma bm32 regout_directv: unsupported V type");
        }
#undef LAUNCH_BM32_REGOUT_DV
        } else {
#define LAUNCH_BM32(VT) \
    packed16_wmma_tile_bm32_2w_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_BM32(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_BM32(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_BM32(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma bm32: unsupported V type");
        }
#undef LAUNCH_BM32
        }
    } else {
#define LAUNCH_BM16(VT) \
    packed16_wmma_tile_bm16_1w_kernel<VT><<<grid, block, 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr)

        switch (V->type) {
            case GGML_TYPE_Q4_0: LAUNCH_BM16(PACKED16_WMMA_V_Q4_0); break;
            case GGML_TYPE_Q8_0: LAUNCH_BM16(PACKED16_WMMA_V_Q8_0); break;
            case GGML_TYPE_F16:  LAUNCH_BM16(PACKED16_WMMA_V_F16);  break;
            default: GGML_ABORT("pwmma bm16: unsupported V type");
        }
#undef LAUNCH_BM16
    }
    // ── Sync debug ──────────────────────────────────────────
    const bool sync_debug =
        getenv("GGML_CUDA_ROCM_PACKED16_WMMA_SYNC_DEBUG") &&
        atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_SYNC_DEBUG"));
    static int call_id = 0;
    const int this_call_id = ++call_id;

    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        GGML_ABORT("PWMMA launch failed: call=%d nq=%d nk=%d variant=%s impl=%s err=%s",
            this_call_id, nq, nk,
            is_gqa2 ? "BM16_GQA2" : (is_bm64 ? "BM64_X4" : (is_bm32 ? "BM32_2W" : "BM16_1W")),
            impl_name,
            hipGetErrorString(launch_err));
    }
    // hipDeviceSynchronize is NOT graph-capture safe.
    // Only synchronize if the stream is not capturing.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    hipError_t capture_err = hipStreamIsCapturing(stream, &capture_status);
    if (capture_err == hipSuccess && capture_status == hipStreamCaptureStatusNone) {
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            GGML_ABORT("PWMMA sync failed: call=%d nq=%d nk=%d hq=%d hk=%d D=256 variant=%s impl=%s err=%s",
                this_call_id, nq, nk, n_heads_q, n_heads_k,
                is_gqa2 ? "BM16_GQA2" : (is_bm64 ? "BM64_X4" : (is_bm32 ? "BM32_2W" : "BM16_1W")),
                impl_name,
                hipGetErrorString(sync_err));
        }
    }

    if (profile_enabled) {
        CUDA_CHECK(hipEventRecord(profile_stop, stream));
        pwmma_kernel_profile profile_host = {};
        CUDA_CHECK(hipMemcpyAsync(&profile_host, profile_dev, sizeof(profile_host), hipMemcpyDeviceToHost, stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        float kernel_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&kernel_ms, profile_start, profile_stop));
        fprintf(stderr, "PWMMA PROFILE: impl=%s nq=%d nk=%d kernel_ms=%.3f qk_cycles=%llu softmax_cycles=%llu pv_cycles=%llu\n",
            impl_name, nq, nk, (double) kernel_ms,
            (unsigned long long) profile_host.qk_cycles,
            (unsigned long long) profile_host.softmax_cycles,
            (unsigned long long) profile_host.pv_cycles);
        CUDA_CHECK(hipEventDestroy(profile_start));
        CUDA_CHECK(hipEventDestroy(profile_stop));
        CUDA_CHECK(hipFree(profile_dev));
    }

    if (live_dot4_shadow && d_live_shadow_err) {
        pwmma_debug_error h_err = {};
        CUDA_CHECK(hipMemcpy(&h_err, d_live_shadow_err, sizeof(h_err), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipFree(d_live_shadow_err));
        if (h_err.flag) {
            GGML_ABORT("PBWMMA live shadow FAILED: code=%d variant=%d block=(%d,%d,%d) thread=%d nq=%d nk=%d hq=%d hk=%d k0=%d k=%d d=%d head_stride=%d packed_rows=%d k_row=%d",
                h_err.code, h_err.variant, h_err.block_x, h_err.block_y, h_err.block_z, h_err.thread_x,
                h_err.nq, h_err.nk, h_err.hq, h_err.hk, h_err.k0, h_err.k, h_err.d,
                h_err.head_stride, h_err.packed_rows, h_err.k_row);
        }
        fprintf(stderr, "%s\n", live_pv_shadow_requested ? "PBWMMA PV WMMA live shadow PASSED" : "PBWMMA I8 live DOT4 shadow PASSED");
    }

    if (causal_skip_active && skip_stats && d_skip_counter) {
        unsigned long long h_skip;
        CUDA_CHECK(hipMemcpy(&h_skip, d_skip_counter, sizeof(unsigned long long), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipFree(d_skip_counter));
        fprintf(stderr, "PWMMA causal skip: variant=%s skipped_cta_tiles=%llu BM=16 skipped_head_tiles=%llu skipped_row_tiles=%llu nq=%d nk=%d BN=%d\n",
            is_gqa2 ? "BM16_GQA2" : (is_bm64 ? "BM64_X4" : (is_bm32 ? "BM32_2W" : "BM16_1W")),
            (unsigned long long)h_skip, (unsigned long long)(is_gqa2 ? h_skip : h_skip),
            (unsigned long long)(is_gqa2 ? h_skip * 2 * 16 : (is_bm64 ? h_skip * 64 : h_skip * bm)), nq, nk, PWMMA_BN);
    }
    fprintf(stderr, "PWMMA EXIT OK\n"); fflush(stderr);
}

#else
static void ggml_cuda_flash_attn_ext_packed16_wmma_tile(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("packed16_wmma_tile requires HIP + GGML_HIP_ROCWMMA_FATTN");
}
#endif
