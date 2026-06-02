// fattn-packed16-dot4-mmq.cuh — packed16 K cache + DOT4/MMQ QK + FlashAttention
//
// rocm_packed16_dot4_mmq is the integer sibling of packed16_wmma_tile:
//   - Q tile: f32 -> q8_0-like int8 payload + float scale per 32 dims
//   - K tile: persistent packed16 int32 payload + half scale per 32 dims
//   - QK: sudot4/dp4a int8xint8 -> int32, then q_scale*k_scale
//   - FA: online softmax + P@V, with the same packed16 row convention as WMMA
//
// Initial target: D=256, Q=f32, K=I32 packed16, V=q4_0/q8_0/f16, dst=f32.
// Optional experiment: V=tbq4_0 via GGML_CUDA_ROCM_PACKED16_TBQ4_V=1.
// Planar/Iso 3-bit V formats are decoded with their inverse local rotations.
//
// Shapes:
//   M16N16: prefill/MMQ-oriented default for nq >= 16. More Q rows per CTA, lower V LDS.
//   M8N32:  small-prefill fallback. Fewer Q rows, wider K tile.
//   M4N64:  optional long-K/small-Q debug shape; higher V LDS, not default.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-dot4-q8k-kq.cuh"
#include "fattn-mma-tbq4.cuh"
#include "dot4-packed16/dp16-trace.cuh"

#include <atomic>
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

static constexpr int PDMQ_BM_DECODE32  = 1;
static constexpr int PDMQ_BN_DECODE32  = 32;
static constexpr int PDMQ_BM_VERIFY2_32 = 2;
static constexpr int PDMQ_BN_VERIFY2_32 = 32;
static constexpr int PDMQ_BM_VERIFY4_32 = 4;
static constexpr int PDMQ_BN_VERIFY4_32 = 32;
static constexpr int PDMQ_BM_DECODE  = 1;
static constexpr int PDMQ_BN_DECODE  = 64;
static constexpr int PDMQ_BM_VERIFY2 = 2;
static constexpr int PDMQ_BN_VERIFY2 = 64;
static constexpr int PDMQ_BM_PREFILL = 16;
static constexpr int PDMQ_BN_PREFILL = 16;
static constexpr int PDMQ_BM_SMALL   = 8;
static constexpr int PDMQ_BN_SMALL   = 32;
static constexpr int PDMQ_BM_LONGK   = 4;
static constexpr int PDMQ_BN_LONGK   = 64;

static_assert(PDMQ_D % QK8_0 == 0, "packed16_dot4_mmq requires D multiple of QK8_0");
static_assert(PDMQ_BM_DECODE32  * PDMQ_BN_DECODE32  <= PDMQ_THREADS, "M1N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_VERIFY2_32 * PDMQ_BN_VERIFY2_32 <= PDMQ_THREADS, "M2N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_VERIFY4_32 * PDMQ_BN_VERIFY4_32 <= PDMQ_THREADS, "M4N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_DECODE  * PDMQ_BN_DECODE  <= PDMQ_THREADS, "M1N64 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_VERIFY2 * PDMQ_BN_VERIFY2 <= PDMQ_THREADS, "M2N64 maps one QK logit subset per CTA");
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
static constexpr int PDMQ_V_Q4_BLOCKS   = PDMQ_D / QK4_0;

// ── Experiment envs ──────────────────────────────────────────────
// PDMQ_IMPL:  baseline | kshared | kshared_stagev | kshared_directv
// PDMQ_VPATH: stage_f32 | direct_pv_q4 | raw_lds_q4
// Explicit legacy PDMQ_IMPL stage/direct spellings remain honored when PDMQ_VPATH is unset.
static const char * pdmq_impl_env() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL");
    return v ? v : "";
}
static const char * pdmq_vpath_env() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH");
    return v ? v : "";
}
static inline bool pdmq_kshared_enabled() {
    const char * v = pdmq_impl_env();
    return strcmp(v, "kshared") == 0 || strcmp(v, "kshared_stagev") == 0 || strcmp(v, "kshared_directv") == 0;
}

enum packed16_dot4_mmq_v_type {
    PACKED16_DOT4_MMQ_V_Q4_0,
    PACKED16_DOT4_MMQ_V_Q8_0,
    PACKED16_DOT4_MMQ_V_F16,
    PACKED16_DOT4_MMQ_V_TBQ4_0,
    PACKED16_DOT4_MMQ_V_PLANAR3_0,
    PACKED16_DOT4_MMQ_V_ISO3_0,
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

bool ggml_cuda_packed16_dot4_mmq_enabled() {
    // Auto-enabled when packed16 K cache is active.
    // Disable: GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0
    {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ");
        if (v && atoi(v) == 0) return false;
    }
    return ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
}

bool ggml_cuda_packed16_tbq4_v_enabled() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_TBQ4_V");
    return v && atoi(v) != 0;
}

static inline bool pdmq_packed16_sidecar_valid(
        const ggml_tensor * K,
        const ggml_tensor * packed16_payload,
        const ggml_tensor * packed16_scales,
        const bool verbose,
        const char * context) {
    const char * reject = nullptr;
    if (!packed16_payload || !packed16_scales) {
        reject = "missing_packed16_k_sidecar";
    } else if (packed16_payload->ne[0] != PDMQ_D / 4 || packed16_scales->ne[0] != PDMQ_D / QK8_0 ||
               packed16_payload->ne[1] < K->ne[1] * K->ne[2] || packed16_scales->ne[1] < K->ne[1] * K->ne[2] ||
               packed16_payload->nb[0] != (int64_t) sizeof(int) || packed16_scales->nb[0] != (int64_t) sizeof(half) ||
               packed16_payload->nb[1] != (PDMQ_D / 4) * (int64_t) sizeof(int) ||
               packed16_scales->nb[1] != (PDMQ_D / QK8_0) * (int64_t) sizeof(half) ||
               packed16_payload->nb[2] != packed16_payload->ne[1] * packed16_payload->nb[1] ||
               packed16_scales->nb[2] != packed16_scales->ne[1] * packed16_scales->nb[1] ||
               packed16_payload->nb[3] != packed16_payload->ne[2] * packed16_payload->nb[2] ||
               packed16_scales->nb[3] != packed16_scales->ne[2] * packed16_scales->nb[2]) {
        reject = "bad_packed16_k_sidecar_shape";
    }

    if (!reject) {
        return true;
    }

    if (verbose) {
        fprintf(stderr,
            "PDMQ2 reject route=rocm_packed16_dot4_mmq backend=dot4_packed16_fa "
            "reject=%s context=%s fallback_disallowed=%d K_data=%p "
            "K=[%lld,%lld,%lld,%lld] payload=%p scales=%p\n",
            reject, context ? context : "unknown",
            ggml_cuda_packed16_dot4_mmq_route_required() ? 1 : 0,
            K ? K->data : nullptr,
            K ? (long long) K->ne[0] : 0, K ? (long long) K->ne[1] : 0,
            K ? (long long) K->ne[2] : 0, K ? (long long) K->ne[3] : 0,
            (const void *) packed16_payload, (const void *) packed16_scales);
    }
    return false;
}

static inline bool ggml_cuda_packed16_dot4_mmq_sidecar_ready(const ggml_tensor * K, const bool verbose) {
    ggml_tensor * packed16_payload = nullptr;
    ggml_tensor * packed16_scales  = nullptr;
    llama_kv_cache_get_packed16_tensors(K ? K->data : nullptr, &packed16_payload, &packed16_scales);
    return K && pdmq_packed16_sidecar_valid(K, packed16_payload, packed16_scales, verbose, "support");
}

bool ggml_cuda_packed16_dot4_mmq_v_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_F16  ||
           type == GGML_TYPE_PLANAR3_0 ||
           type == GGML_TYPE_ISO3_0 ||
           (type == GGML_TYPE_TBQ4_0 && ggml_cuda_packed16_tbq4_v_enabled());
}

bool ggml_cuda_packed16_dot4_mmq_supported(const int cc, const ggml_tensor * dst) {
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
    const bool v_needs_mmq_decode = V->type == GGML_TYPE_TBQ4_0 || V->type == GGML_TYPE_PLANAR3_0 || V->type == GGML_TYPE_ISO3_0;
    if (Q->ne[1] <= 1 && !v_needs_mmq_decode && !ggml_cuda_packed16_dot4_mmq_route_required()) {
        // Keep default decode on the existing BN64/split-K DOT4 route for established V types.
        // A route-require opt-in is a contract and may force this MMQ kernel for q4_0/q8_0/f16 decode too.
        // The 3-bit/TBQ V experiments use this kernel for nq==1 because q8k_dot4_kq's V loaders do not support them.
        return false;
    }
    if (Q->ne[2] <= 0 || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] < K->ne[1] || V->ne[2] != K->ne[2] || (V->ne[3] != Q->ne[3] && V->ne[3] != 1)) {
        return false;
    }
    if (!ggml_cuda_packed16_dot4_mmq_v_supported(V->type)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1)) {
        return false;
    }
    if (!ggml_cuda_packed16_dot4_mmq_sidecar_ready(K, ggml_cuda_packed16_dot4_mmq_route_required())) {
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

static __device__ __forceinline__ const block_q4_0 * pdmq_v_q4_0_block_ptr(
        const char * __restrict__ V,
        const int64_t v_nb10,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int blk) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const char * ptr = V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k) * v_nb11;
    return (const block_q4_0 *) (ptr + int64_t(blk) * v_nb10);
}

static __device__ __forceinline__ float pdmq_decode_v_q4_0_block(
        const block_q4_0 & bq,
        const int in) {
    const int iq = in & 15;
    const int shift = in >= 16 ? 4 : 0;
    return float((int(bq.qs[iq] >> shift) & 0x0f) - 8) * __half2float(bq.d);
}

template<int D>
static __device__ __forceinline__ float pdmq_decode_v_q4_0_raw_lds(
        const block_q4_0 * __restrict__ v_raw_tile,
        const int kk,
        const int d) {
    static_assert(D % QK4_0 == 0, "raw q4 LDS V decode requires full q4 blocks");
    const int blk = d / QK4_0;
    const int in  = d & (QK4_0 - 1);
    return pdmq_decode_v_q4_0_block(v_raw_tile[kk * (D / QK4_0) + blk], in);
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
    const int blk = d / QK4_0;
    const int in  = d & (QK4_0 - 1);
    const block_q4_0 * bq = pdmq_v_q4_0_block_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, blk);
    return pdmq_decode_v_q4_0_block(*bq, in);
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

static __device__ __forceinline__ uint8_t pdmq_unpack_3bit_planar_iso(
        const uint8_t * __restrict__ qs,
        const uint8_t * __restrict__ signs,
        const int j) {
    const uint8_t low = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi  = (signs[j / 8] >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

static __device__ __forceinline__ float pdmq_decode_v_planar3_0(
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
    const int ib = d / QK_PLANAR3;
    const int j  = d % QK_PLANAR3;
    const int p  = j / 2;
    const block_planar3_0 * bp = (const block_planar3_0 *) (ptr + int64_t(ib) * v_nb10);
    const int j0 = p * 2;
    const float q0 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bp->qs, bp->signs, j0 + 0)];
    const float q1 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bp->qs, bp->signs, j0 + 1)];
    const float c = PI_COS[p];
    const float s = PI_SIN[p];
    const float v0 =  c * q0 + s * q1;
    const float v1 = -s * q0 + c * q1;
    return (j & 1 ? v1 : v0) * __half2float(bp->d);
}

static __device__ __forceinline__ float pdmq_decode_v_iso3_0(
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
    const int ib = d / QK_ISO3;
    const int j  = d % QK_ISO3;
    const int g  = j / 4;
    const int off = j & 3;
    const block_iso3_0 * bi = (const block_iso3_0 *) (ptr + int64_t(ib) * v_nb10);
    const int j0 = g * 4;
    const float q0 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bi->qs, bi->signs, j0 + 0)];
    const float q1 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bi->qs, bi->signs, j0 + 1)];
    const float q2 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bi->qs, bi->signs, j0 + 2)];
    const float q3 = PI_CENTROIDS_3BIT[pdmq_unpack_3bit_planar_iso(bi->qs, bi->signs, j0 + 3)];
    const float qw = PI_QW[g];
    const float qx = -PI_QX[g];
    const float qy = -PI_QY[g];
    const float qz = -PI_QZ[g];
    const float r0 = qw*q0 - qx*q1 - qy*q2 - qz*q3;
    const float r1 = qw*q1 + qx*q0 + qy*q3 - qz*q2;
    const float r2 = qw*q2 - qx*q3 + qy*q0 + qz*q1;
    const float r3 = qw*q3 + qx*q2 - qy*q1 + qz*q0;
    const float norm = __half2float(bi->d);
    return (off == 0 ? r0 : off == 1 ? r1 : off == 2 ? r2 : r3) * norm;
}

static __device__ __forceinline__ float pdmq_decode_v_tbq4_0(
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
    const int ib = d / QK_TBQ4;
    const int j  = d % QK_TBQ4;
    const block_tbq4_0 * bt = (const block_tbq4_0 *) (ptr + int64_t(ib) * v_nb10);
    const uint8_t packed = bt->qs[j / 2];
    const uint8_t idx = (j & 1) ? (packed >> 4) : (packed & 0x0f);
    return d_tbq4_centroids[idx] * __half2float(bt->d);
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
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_TBQ4_0) {
        return pdmq_decode_v_tbq4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_PLANAR3_0) {
        return pdmq_decode_v_planar3_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_ISO3_0) {
        return pdmq_decode_v_iso3_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
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

struct pdmq_kernel_profile {
    unsigned long long qk_cycles;
    unsigned long long softmax_cycles;
    unsigned long long pv_cycles;
};

#define PDMQ_PROFILE_PHASE_BEGIN() do { \
    if (profile) { \
        __syncthreads(); \
        if (tid == 0) { profile_t0 = clock64(); } \
        __syncthreads(); \
    } \
} while (0)

#define PDMQ_PROFILE_PHASE_END(FIELD) do { \
    if (profile) { \
        __syncthreads(); \
        if (tid == 0) { \
            const unsigned long long profile_t1 = clock64(); \
            atomicAdd(&profile->FIELD, profile_t1 >= profile_t0 ? profile_t1 - profile_t0 : 0ull); \
        } \
        __syncthreads(); \
    } \
} while (0)

// ── GQA2 Kernel (2 Q heads per CTA, M16N16 or M8N32) ──────────────
template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool RAW_LDS_Q4, bool KSHARED>
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
        int packed_rows, int q_offset, float attention_scale,
        pdmq_kernel_profile * __restrict__ profile) {

    static_assert(D == PDMQ_D, "packed16_dot4_mmq_gqa2 is D256-only");
    static_assert(BM * BN == PDMQ_THREADS, "packed16_dot4_mmq_gqa2 maps one QK logit per thread");
    static constexpr int GQA_GROUP = 2;

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
    static constexpr bool RAW_Q4_LDS = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0;

    __shared__ float row_m[GQA_GROUP][BM + 1];
    __shared__ float row_l[GQA_GROUP][BM + 1];
    __shared__ float old_s[GQA_GROUP][BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ block_q4_0 v_raw_tile[RAW_Q4_LDS ? BN * (D / QK4_0) : 1];
    __shared__ int   k_payload_s[KSHARED ? BN * (PDMQ_D / 4 + 1) : 1];
    __shared__ half  k_scales_s [KSHARED ? BN * (PDMQ_D / QK8_0 + 1) : 1];
    __shared__ unsigned long long profile_t0;

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
        } else if constexpr (RAW_Q4_LDS) {
            constexpr int V_BLOCKS = D / QK4_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                v_raw_tile[kk * V_BLOCKS + blk] = *pdmq_v_q4_0_block_ptr(
                    V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
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

            PDMQ_PROFILE_PHASE_BEGIN();

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
            PDMQ_PROFILE_PHASE_END(qk_cycles);

            PDMQ_PROFILE_PHASE_BEGIN();

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
            PDMQ_PROFILE_PHASE_END(softmax_cycles);

            PDMQ_PROFILE_PHASE_BEGIN();

            // PV accumulate. For q4 direct-V, run kk-major so each V[k,d]
            // is decoded once per thread lane and reused across query rows.
            if (tid < D) {
                if constexpr (!STAGE_V && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0) {
                    float acc[BM];
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        acc[qr] = q < nq ? out[s][qr] * old_s[s][qr] : 0.0f;
                    }
                    for (int kk = 0; kk < BN; ++kk) {
                        const float vv = kk < tile_n
                            ? (RAW_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kk, tid)
                                : pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid))
                            : 0.0f;
                        for (int qr = 0; qr < BM; ++qr) {
                            const int q = q0 + qr;
                            if (q >= nq) continue;
                            const float p = logits[qr][kk];
                            if (p != 0.0f) acc[qr] += p * vv;
                        }
                    }
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        if (q < nq) out[s][qr] = acc[qr];
                    }
                } else {
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
            }
            __syncthreads();
            PDMQ_PROFILE_PHASE_END(pv_cycles);
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

template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool RAW_LDS_Q4, bool KSHARED>
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
        float attention_scale,
        pdmq_kernel_profile * __restrict__ profile) {

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
    static constexpr bool RAW_Q4_LDS = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0;

    __shared__ float row_m[BM + 1];
    __shared__ float row_l[BM + 1];
    __shared__ float old_s[BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ block_q4_0 v_raw_tile[RAW_Q4_LDS ? BN * (D / QK4_0) : 1];
    __shared__ int   k_payload_s[KSHARED ? BN * (PDMQ_D / 4 + 1) : 1];
    __shared__ half  k_scales_s [KSHARED ? BN * (PDMQ_D / QK8_0 + 1) : 1];
    __shared__ unsigned long long profile_t0;

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
        } else if constexpr (RAW_Q4_LDS) {
            constexpr int V_BLOCKS = D / QK4_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                v_raw_tile[kk * V_BLOCKS + blk] = *pdmq_v_q4_0_block_ptr(
                    V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
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

        PDMQ_PROFILE_PHASE_BEGIN();

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
        PDMQ_PROFILE_PHASE_END(qk_cycles);

        PDMQ_PROFILE_PHASE_BEGIN();

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
        PDMQ_PROFILE_PHASE_END(softmax_cycles);

        PDMQ_PROFILE_PHASE_BEGIN();

        if (tid < D) {
            if constexpr (!STAGE_V && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0) {
                float acc[BM];
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    acc[qr] = q < nq ? out[qr] * old_s[qr] : 0.0f;
                }
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const float vv = kk < tile_n
                        ? (RAW_Q4_LDS
                            ? pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kk, tid)
                            : pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid))
                        : 0.0f;
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        if (q >= nq) {
                            continue;
                        }
                        const float p = logits[qr][kk];
                        if (p != 0.0f) {
                            acc[qr] += p * vv;
                        }
                    }
                }
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = q0 + qr;
                    if (q < nq) {
                        out[qr] = acc[qr];
                    }
                }
            } else {
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
        }
        __syncthreads();
        PDMQ_PROFILE_PHASE_END(pv_cycles);
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

enum pdmq_shape {
    PDMQ_SHAPE_M1N32,
    PDMQ_SHAPE_M2N32,
    PDMQ_SHAPE_M4N32,
    PDMQ_SHAPE_M1N64,
    PDMQ_SHAPE_M2N64,
    PDMQ_SHAPE_M16N16,
    PDMQ_SHAPE_M8N32,
    PDMQ_SHAPE_M4N64,
};

enum pdmq_v_path {
    PDMQ_V_STAGE_F32,
    PDMQ_V_DIRECT_PV_Q4,
    PDMQ_V_RAW_LDS_Q4,
    PDMQ_V_DIRECT_LEGACY,
};

enum pdmq_role {
    PDMQ_ROLE_VERIFY,
    PDMQ_ROLE_DECODE,
    PDMQ_ROLE_PREFILL,
};

struct pdmq_plan {
    pdmq_shape  shape;
    pdmq_v_path v_path;
    pdmq_role   role;

    int  gqa_group;
    int  requested_gqa_group;
    bool causal;
    bool kshared;
    bool split_k;

    ggml_type v_type;
    int nq;
    int nk;
    int hq;
    int hk;
    int d;
};

static inline const char * pdmq_shape_name(const pdmq_shape shape) {
    switch (shape) {
        case PDMQ_SHAPE_M1N32:  return "1x32";
        case PDMQ_SHAPE_M2N32:  return "2x32";
        case PDMQ_SHAPE_M4N32:  return "4x32";
        case PDMQ_SHAPE_M1N64:  return "1x64";
        case PDMQ_SHAPE_M2N64:  return "2x64";
        case PDMQ_SHAPE_M16N16: return "16x16";
        case PDMQ_SHAPE_M8N32:  return "8x32";
        case PDMQ_SHAPE_M4N64:  return "4x64";
        default:                return "unknown";
    }
}

static inline const char * pdmq_v_path_name(const pdmq_v_path v_path) {
    switch (v_path) {
        case PDMQ_V_STAGE_F32:     return "stage_f32";
        case PDMQ_V_DIRECT_PV_Q4:  return "direct_pv_q4";
        case PDMQ_V_RAW_LDS_Q4:    return "raw_lds_q4";
        case PDMQ_V_DIRECT_LEGACY: return "direct_legacy";
        default:                   return "unknown";
    }
}

static inline const char * pdmq_role_name(const pdmq_role role) {
    switch (role) {
        case PDMQ_ROLE_VERIFY:  return "verify";
        case PDMQ_ROLE_DECODE:  return "decode";
        case PDMQ_ROLE_PREFILL: return "prefill";
        default:                return "unknown";
    }
}

static inline bool pdmq_profile_enabled() {
    const char * v = getenv("GGML_CUDA_PDMQ_PROFILE");
    return v && atoi(v) != 0;
}

struct pdmq_role_profile_slot {
    std::atomic<unsigned long long> calls;
    std::atomic<unsigned long long> kernel_us;
    std::atomic<unsigned long long> qk_cycles;
    std::atomic<unsigned long long> softmax_cycles;
    std::atomic<unsigned long long> pv_cycles;
};

static inline pdmq_role_profile_slot * pdmq_role_profile_slots() {
    static pdmq_role_profile_slot slots[5] = {};
    return slots;
}

static inline int pdmq_role_profile_index(const int nq) {
    if (nq <= 1) return 0;
    if (nq == 2) return 1;
    if (nq == 3) return 2;
    if (nq == 4) return 3;
    return 4;
}

static inline const char * pdmq_role_profile_label(const int idx) {
    switch (idx) {
        case 0: return "decode nq=1";
        case 1: return "verify nq=2";
        case 2: return "verify nq=3";
        case 3: return "verify nq=4";
        default: return "prefill/verify-other";
    }
}

static inline void pdmq_profile_record_and_emit(
        const pdmq_plan & plan,
        const pdmq_kernel_profile & kernel_profile,
        const float kernel_ms) {
    pdmq_role_profile_slot * slots = pdmq_role_profile_slots();
    const int idx = pdmq_role_profile_index(plan.nq);
    slots[idx].calls.fetch_add(1, std::memory_order_relaxed);
    slots[idx].kernel_us.fetch_add((unsigned long long) (kernel_ms * 1000.0f), std::memory_order_relaxed);
    slots[idx].qk_cycles.fetch_add(kernel_profile.qk_cycles, std::memory_order_relaxed);
    slots[idx].softmax_cycles.fetch_add(kernel_profile.softmax_cycles, std::memory_order_relaxed);
    slots[idx].pv_cycles.fetch_add(kernel_profile.pv_cycles, std::memory_order_relaxed);

    fprintf(stderr,
        "PDMQ2 summary route=rocm_packed16_dot4_mmq K=i32 V=%s last_role=%s last_nq=%d last_kernel_ms=%.3f "
        "last_qk_cycles=%llu last_softmax_cycles=%llu last_pv_cycles=%llu\n",
        ggml_type_name(plan.v_type), pdmq_role_name(plan.role), plan.nq, (double) kernel_ms,
        (unsigned long long) kernel_profile.qk_cycles,
        (unsigned long long) kernel_profile.softmax_cycles,
        (unsigned long long) kernel_profile.pv_cycles);
    for (int i = 0; i < 5; ++i) {
        const unsigned long long calls = slots[i].calls.load(std::memory_order_relaxed);
        if (!calls) continue;
        fprintf(stderr,
            "PDMQ2 summary route=rocm_packed16_dot4_mmq K=i32 V=%s %s calls=%llu time_ms=%.3f "
            "qk_cycles=%llu softmax_cycles=%llu pv_cycles=%llu\n",
            ggml_type_name(plan.v_type), pdmq_role_profile_label(i), calls,
            (double) slots[i].kernel_us.load(std::memory_order_relaxed) / 1000.0,
            (unsigned long long) slots[i].qk_cycles.load(std::memory_order_relaxed),
            (unsigned long long) slots[i].softmax_cycles.load(std::memory_order_relaxed),
            (unsigned long long) slots[i].pv_cycles.load(std::memory_order_relaxed));
    }
}

static inline pdmq_shape pdmq_parse_shape_env(const char * env) {
    if (strcmp(env, "1x32") == 0 || strcmp(env, "m1n32") == 0) {
        return PDMQ_SHAPE_M1N32;
    }
    if (strcmp(env, "2x32") == 0 || strcmp(env, "m2n32") == 0) {
        return PDMQ_SHAPE_M2N32;
    }
    if (strcmp(env, "4x32") == 0 || strcmp(env, "m4n32") == 0) {
        return PDMQ_SHAPE_M4N32;
    }
    if (strcmp(env, "1x64") == 0 || strcmp(env, "m1n64") == 0) {
        return PDMQ_SHAPE_M1N64;
    }
    if (strcmp(env, "2x64") == 0 || strcmp(env, "m2n64") == 0) {
        return PDMQ_SHAPE_M2N64;
    }
    if (strcmp(env, "16x16") == 0 || strcmp(env, "m16n16") == 0) {
        return PDMQ_SHAPE_M16N16;
    }
    if (strcmp(env, "8x32") == 0 || strcmp(env, "m8n32") == 0) {
        return PDMQ_SHAPE_M8N32;
    }
    if (strcmp(env, "4x64") == 0 || strcmp(env, "m4n64") == 0) {
        return PDMQ_SHAPE_M4N64;
    }
    GGML_ABORT("packed16_dot4_mmq: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE=%s, expected 1x32, 2x32, 4x32, 1x64, 2x64, 16x16, 8x32, or 4x64", env);
}

static inline pdmq_v_path pdmq_parse_v_path_env(const char * env) {
    if (strcmp(env, "stage_f32") == 0 || strcmp(env, "stage") == 0 || strcmp(env, "f32") == 0) {
        return PDMQ_V_STAGE_F32;
    }
    if (strcmp(env, "direct_pv_q4") == 0 || strcmp(env, "direct_q4") == 0 || strcmp(env, "directv") == 0) {
        return PDMQ_V_DIRECT_PV_Q4;
    }
    if (strcmp(env, "raw_lds_q4") == 0 || strcmp(env, "raw_q4") == 0) {
        return PDMQ_V_RAW_LDS_Q4;
    }
    GGML_ABORT("packed16_dot4_mmq: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH=%s, expected stage_f32, direct_pv_q4, or raw_lds_q4", env);
}

static inline pdmq_shape pdmq_select_shape_auto(
        const int nq,
        const int nk,
        const ggml_type v_type) {
    // Role-specialized MTP shapes. Decode and tiny verify should not share the
    // prefill/small-prefill policy: they have one to four Q rows and benefit
    // from wider K tiles with less inactive-M overhead.
    if (nq <= 1) {
        return PDMQ_SHAPE_M1N32;
    }
    if (nq == 2) {
        return PDMQ_SHAPE_M2N32;
    }
    if (nq <= 4) {
        // M4N32 remains available via SHAPE env, but the runtime canary showed
        // higher verify-nq4 time than the proven 8x32 policy.
        return PDMQ_SHAPE_M8N32;
    }

    // Attention equivalent of the MMQ max-x lesson: do not maximize tile width
    // on gfx1100 just because it exists. Wider N64 shapes are reserved for the
    // decode/verify roles above or explicit env overrides.
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

static inline pdmq_role pdmq_select_role(const int nq) {
    if (nq <= 1) {
        return PDMQ_ROLE_DECODE;
    }
    if (nq <= 8) {
        return PDMQ_ROLE_VERIFY;
    }
    return PDMQ_ROLE_PREFILL;
}

static inline pdmq_v_path pdmq_select_v_path(
        const int nq,
        const int nk,
        const ggml_type v_type) {
    GGML_UNUSED(nk);

    if (const char * env = pdmq_vpath_env(); env && *env) {
        const pdmq_v_path requested = pdmq_parse_v_path_env(env);
        if (requested == PDMQ_V_RAW_LDS_Q4 && v_type != GGML_TYPE_Q4_0) {
            GGML_ABORT("packed16_dot4_mmq: raw_lds_q4 requires V=q4_0, got V=%s", ggml_type_name(v_type));
        }
        if (requested == PDMQ_V_DIRECT_PV_Q4 && v_type != GGML_TYPE_Q4_0) {
            // direct_pv_q4 is a q4 policy; keep other V types on the proven f32-stage path.
            return PDMQ_V_STAGE_F32;
        }
        return requested;
    }

    const char * impl = pdmq_impl_env();
    if (impl && *impl) {
        if (strcmp(impl, "kshared_directv") == 0) {
            return v_type == GGML_TYPE_Q4_0 ? PDMQ_V_DIRECT_PV_Q4 : PDMQ_V_DIRECT_LEGACY;
        }
        // Preserve explicit legacy baseline/kshared/kshared_stagev semantics for A/B tests.
        return PDMQ_V_STAGE_F32;
    }

    // Patch 4: with the production-constrained ubatch=1024 top case, raw LDS q4
    // wins over f32 V staging for MTP small-Q roles. Keep direct_pv_q4 explicit;
    // it remains slower in the same A/B.
    if (v_type == GGML_TYPE_Q4_0 && nq <= 8) {
        return PDMQ_V_RAW_LDS_Q4;
    }

    return PDMQ_V_STAGE_F32;
}

static inline pdmq_plan pdmq_make_plan(
        const int nq,
        const int nk,
        const int n_heads_q,
        const int n_heads_k,
        const int d,
        const ggml_type v_type,
        const bool causal,
        const pdmq_shape shape,
        const int requested_gqa_group) {
    pdmq_plan plan = {};
    plan.shape = shape;
    plan.v_path = pdmq_select_v_path(nq, nk, v_type);
    plan.role = pdmq_select_role(nq);
    plan.gqa_group = requested_gqa_group;
    plan.requested_gqa_group = requested_gqa_group;
    plan.causal = causal;
    plan.kshared = pdmq_kshared_enabled();
    plan.split_k = false;
    plan.v_type = v_type;
    plan.nq = nq;
    plan.nk = nk;
    plan.hq = n_heads_q;
    plan.hk = n_heads_k;
    plan.d = d;
    return plan;
}

void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == PDMQ_D && K->ne[0] * 4 == PDMQ_D && V->ne[0] == PDMQ_D && dst->ne[0] == PDMQ_D);
    GGML_ASSERT(Q->ne[1] > 0 && Q->ne[2] % K->ne[2] == 0);
    GGML_ASSERT(ggml_cuda_packed16_dot4_mmq_v_supported(V->type));

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
    if (!pdmq_packed16_sidecar_valid(K, packed16_payload, packed16_scales, true, "launch")) {
        GGML_ABORT("required rocm_packed16_dot4_mmq route missing valid packed16 K sidecar");
    }

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

    pdmq_plan plan = pdmq_make_plan(
        nq, nk, n_heads_q, n_heads_k, PDMQ_D, V->type, assume_causal, shape,
        ggml_cuda_rocm_packed16_dot4_mmq_gqa_group());
    shape = plan.shape;

    const int launch_bm = shape == PDMQ_SHAPE_M1N32  ? PDMQ_BM_DECODE32  :
                          shape == PDMQ_SHAPE_M2N32  ? PDMQ_BM_VERIFY2_32 :
                          shape == PDMQ_SHAPE_M4N32  ? PDMQ_BM_VERIFY4_32 :
                          shape == PDMQ_SHAPE_M1N64  ? PDMQ_BM_DECODE     :
                          shape == PDMQ_SHAPE_M2N64  ? PDMQ_BM_VERIFY2    :
                          shape == PDMQ_SHAPE_M16N16 ? PDMQ_BM_PREFILL    :
                          shape == PDMQ_SHAPE_M8N32  ? PDMQ_BM_SMALL      : PDMQ_BM_LONGK;
    const int launch_bn = shape == PDMQ_SHAPE_M1N32  ? PDMQ_BN_DECODE32  :
                          shape == PDMQ_SHAPE_M2N32  ? PDMQ_BN_VERIFY2_32 :
                          shape == PDMQ_SHAPE_M4N32  ? PDMQ_BN_VERIFY4_32 :
                          shape == PDMQ_SHAPE_M1N64  ? PDMQ_BN_DECODE     :
                          shape == PDMQ_SHAPE_M2N64  ? PDMQ_BN_VERIFY2    :
                          shape == PDMQ_SHAPE_M16N16 ? PDMQ_BN_PREFILL    :
                          shape == PDMQ_SHAPE_M8N32  ? PDMQ_BN_SMALL      : PDMQ_BN_LONGK;

    const bool request_gqa2 = (plan.requested_gqa_group == 2);
    const bool gqa2_supported = request_gqa2 &&
        (shape == PDMQ_SHAPE_M16N16 || shape == PDMQ_SHAPE_M8N32) &&
        (gqa_ratio >= 2) && (nq > 1);
    const bool gqa_require = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_REQUIRE");
        return v && atoi(v) != 0;
    }();
    if (request_gqa2 && !gqa2_supported && gqa_require) {
        GGML_ABORT("PDMQ GQA2 requested but unsupported: shape=%s nq=%d hq=%d hk=%d gqa_ratio=%d",
                   pdmq_shape_name(shape), nq, n_heads_q, n_heads_k, gqa_ratio);
    }
    const bool is_gqa2 = request_gqa2 && gqa2_supported;
    plan.gqa_group = is_gqa2 ? 2 : 1;

    const int groups_per_kv = CEIL_DIV(gqa_ratio, plan.gqa_group);
    const int grid_y_gqa2  = n_heads_k * groups_per_kv;
    const int grid_y_old   = n_heads_q;

    dim3 grid(CEIL_DIV(nq, launch_bm),
        is_gqa2 ? grid_y_gqa2 : n_heads_q,
        batch);
    dim3 block(PDMQ_THREADS);
    hipStream_t stream = ctx.stream();

    const bool kshared = plan.kshared;
    const bool n64_shape = shape == PDMQ_SHAPE_M1N64 || shape == PDMQ_SHAPE_M2N64 || shape == PDMQ_SHAPE_M4N64;
    const bool stage_v_requested = plan.v_path == PDMQ_V_STAGE_F32;
    // N64 tiles would exceed/pressure LDS with staged f32 V. Keep these role
    // specializations on direct V loads while preserving the explicit vpath label.
    const bool stage_v = stage_v_requested && !n64_shape;
    const bool raw_lds_q4 = plan.v_path == PDMQ_V_RAW_LDS_Q4;
    const bool directv = !stage_v;

    if (dp16_trace_enabled()) {
        dp16_problem dp_problem = dp16_problem_init(DP16_OP_FA_QKPV);
        dp_problem.m = nq;
        dp_problem.n = nk;
        dp_problem.k = PDMQ_D;
        dp_problem.batch = batch;
        dp_problem.heads_q = n_heads_q;
        dp_problem.heads_kv = n_heads_k;
        dp_problem.head_dim = PDMQ_D;
        dp_problem.src0_type = Q->type;
        dp_problem.src1_type = K->type;
        dp_problem.src2_type = V->type;
        dp_problem.dst_type = dst->type;
        dp_problem.is_decode = nq == 1;
        dp_problem.cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        dp_problem.a = dp16_operand_desc_from_type(DP16_OPERAND_ACTIVATION, DP16_STORAGE_TRANSIENT_TILE,
                Q->type, nq, PDMQ_D, Q->nb[1], Q->nb[0]);
        dp_problem.b = dp16_operand_desc_from_type(DP16_OPERAND_K_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
                K->type, nk, PDMQ_D, K->nb[1], K->nb[0]);
        dp_problem.v = dp16_operand_desc_from_type(DP16_OPERAND_V_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
                V->type, nk, PDMQ_D, V->nb[1], V->nb[0]);
        dp_problem.dst = dp16_operand_desc_from_type(DP16_OPERAND_OUTPUT, DP16_STORAGE_OUTPUT,
                dst->type, nq, PDMQ_D, dst->nb[1], dst->nb[0]);

        const dp16_plan dp_plan = dp16_plan_fa_mmq(dp_problem,
                launch_bm, launch_bn, plan.gqa_group, plan.requested_gqa_group,
                pdmq_v_path_name(plan.v_path), assume_causal, kshared);
        dp16_trace_emit_plan(dp_problem, dp_plan);
    }

    GGML_ASSERT(!raw_lds_q4 || V->type == GGML_TYPE_Q4_0);
    const bool pdmq_verbose =
        (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")) != 0) ||
        (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE")) != 0);

    { static bool once = false; if (pdmq_verbose || !once) {
        fprintf(stderr,
            "PDMQ2 route=rocm_packed16_dot4_mmq backend=dot4_packed16_fa variant=%s "
            "shape=%s(%dx%d) vpath=%s gqa_group=%d gqa_request=%d role=%s "
            "nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d grid_y_old=%d grid_y_new=%d b=%d sc=%g "
            "K=i32 V=%s packed_rows=%d head_stride=%d causal=%d stage_v=%d raw_lds_q4=%d kshared=%d directv=%d split_k=%d\n",
            is_gqa2 ? "GQA2" : "GQA1",
            pdmq_shape_name(plan.shape), launch_bm, launch_bn,
            pdmq_v_path_name(plan.v_path), plan.gqa_group, plan.requested_gqa_group, pdmq_role_name(plan.role),
            nq, nk, n_heads_q, n_heads_k, gqa_ratio,
            grid_y_old, is_gqa2 ? grid_y_gqa2 : grid_y_old,
            batch, (double) attention_scale,
            ggml_type_name(V->type), packed_rows, packed_rows / n_heads_k, assume_causal ? 1 : 0,
            stage_v ? 1 : 0, raw_lds_q4 ? 1 : 0, kshared ? 1 : 0, directv ? 1 : 0, plan.split_k ? 1 : 0);
    }
    if (!once) { once = true;
        if (!pdmq_qk_probe_pass(stream)) GGML_ABORT("PDMQ QK probe failed");
        if (is_gqa2 && !pdmq_qk_probe_gqa2_pass(stream)) GGML_ABORT("PDMQ GQA2 QK probe failed");
    }}

    bool profile_enabled = pdmq_profile_enabled();
    if (profile_enabled) {
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        const hipError_t capture_err = hipStreamIsCapturing(stream, &capture_status);
        if (capture_err == hipSuccess && capture_status != hipStreamCaptureStatusNone) {
            // hipStreamSynchronize/event timing is illegal during CUDA/HIP graph capture.
            // Keep profiling debug-only and skip captured replay/capture calls instead of
            // perturbing normal graph execution.
            profile_enabled = false;
        }
    }
    pdmq_kernel_profile * profile_dev = nullptr;
    hipEvent_t profile_start = nullptr;
    hipEvent_t profile_stop  = nullptr;
    if (profile_enabled) {
        CUDA_CHECK(hipMalloc((void **) &profile_dev, sizeof(pdmq_kernel_profile)));
        CUDA_CHECK(hipMemsetAsync(profile_dev, 0, sizeof(pdmq_kernel_profile), stream));
        CUDA_CHECK(hipEventCreate(&profile_start));
        CUDA_CHECK(hipEventCreate(&profile_stop));
        CUDA_CHECK(hipEventRecord(profile_start, stream));
    }

#define PDMQ_LAUNCH_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED) \
    packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED><<<grid, block, 0, stream>>>( \
        (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, profile_dev)

#define PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL) \
    packed16_dot4_mmq_gqa2_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL><<<grid, block, 0, stream>>>( \
        (const float *) Q->data, (const char *) V->data, (float *) dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, profile_dev)

#define PDMQ_LAUNCH_GQA2_IMPL(VT, BM_VAL, BN_VAL, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) do { \
    if (raw_lds_q4_var && kshared_var) { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, false, true, true); \
    } else if (raw_lds_q4_var) { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, false, true, false); \
    } else if (kshared_var && directv_var) { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, false, false, true); \
    } else if (kshared_var) { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, true, false, true); \
    } else if (directv_var) { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, false, false, false); \
    } else { \
        PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, true, false, false); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQA2(VT, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) do { \
    if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_GQA2_IMPL(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, kshared_var, directv_var, raw_lds_q4_var); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_GQA2_IMPL(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, kshared_var, directv_var, raw_lds_q4_var); \
    } else { \
        GGML_ABORT("PDMQ GQA2 launch requested for unsupported shape=%s", pdmq_shape_name(shape)); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQA2_NO_RAW(VT, CAUSAL, kshared_var, directv_var) do { \
    if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_GQA2_IMPL(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, kshared_var, directv_var, false); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_GQA2_IMPL(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, kshared_var, directv_var, false); \
    } else { \
        GGML_ABORT("PDMQ GQA2 launch requested for unsupported shape=%s", pdmq_shape_name(shape)); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, kshared_val, stagev_val, raw_lds_q4_val) do { \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M1N64) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_DECODE, PDMQ_BN_DECODE, CAUSAL, false, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M2N64) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY2, PDMQ_BN_VERIFY2, CAUSAL, false, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_LONGK, PDMQ_BN_LONGK, CAUSAL, false, raw_lds_q4_val, kshared_val); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED(VT, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) do { \
    if (raw_lds_q4_var && kshared_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, false, true); \
    } else if (raw_lds_q4_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, true); \
    } else if (kshared_var && directv_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, false, false); \
    } else if (kshared_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, true, false); \
    } else if (directv_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, false); \
    } else { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, true, false); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED_NO_RAW(VT, CAUSAL, kshared_var, directv_var) do { \
    if (kshared_var && directv_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, false, false); \
    } else if (kshared_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, true, true, false); \
    } else if (directv_var) { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, false); \
    } else { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, true, false); \
    } \
} while (0)

if (is_gqa2) {
    if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0,      true, kshared, directv, raw_lds_q4); break;
            case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_Q8_0,      true, kshared, directv); break;
            case GGML_TYPE_F16:       PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_F16,       true, kshared, directv); break;
            case GGML_TYPE_TBQ4_0:    PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_TBQ4_0,    true, kshared, directv); break;
            case GGML_TYPE_PLANAR3_0: PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_PLANAR3_0, true, kshared, directv); break;
            case GGML_TYPE_ISO3_0:    PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_ISO3_0,    true, kshared, directv); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0,      false, kshared, directv, raw_lds_q4); break;
            case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_Q8_0,      false, kshared, directv); break;
            case GGML_TYPE_F16:       PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_F16,       false, kshared, directv); break;
            case GGML_TYPE_TBQ4_0:    PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_TBQ4_0,    false, kshared, directv); break;
            case GGML_TYPE_PLANAR3_0: PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_PLANAR3_0, false, kshared, directv); break;
            case GGML_TYPE_ISO3_0:    PDMQ_LAUNCH_GQA2_NO_RAW(PACKED16_DOT4_MMQ_V_ISO3_0,    false, kshared, directv); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    }
} else {
#define PDMQ_SWITCH_V(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q4_0,      CAUSAL, kshared, directv, raw_lds_q4); break; \
        case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_TYPED_NO_RAW(PACKED16_DOT4_MMQ_V_Q8_0,      CAUSAL, kshared, directv); break; \
        case GGML_TYPE_F16:       PDMQ_LAUNCH_TYPED_NO_RAW(PACKED16_DOT4_MMQ_V_F16,       CAUSAL, kshared, directv); break; \
        case GGML_TYPE_TBQ4_0:    PDMQ_LAUNCH_TYPED_NO_RAW(PACKED16_DOT4_MMQ_V_TBQ4_0,    CAUSAL, kshared, directv); break; \
        case GGML_TYPE_PLANAR3_0: PDMQ_LAUNCH_TYPED_NO_RAW(PACKED16_DOT4_MMQ_V_PLANAR3_0, CAUSAL, kshared, directv); break; \
        case GGML_TYPE_ISO3_0:    PDMQ_LAUNCH_TYPED_NO_RAW(PACKED16_DOT4_MMQ_V_ISO3_0,    CAUSAL, kshared, directv); break; \
        default: GGML_ABORT("packed16_dot4_mmq: unsupported V type"); \
    }

    if (assume_causal) {
        PDMQ_SWITCH_V(true);
    } else {
        PDMQ_SWITCH_V(false);
    }

#undef PDMQ_SWITCH_V
}
#undef PDMQ_LAUNCH_TYPED_NO_RAW
#undef PDMQ_LAUNCH_TYPED
#undef PDMQ_LAUNCH_GQA2_NO_RAW
#undef PDMQ_LAUNCH_GQA2
#undef PDMQ_LAUNCH_GQA2_IMPL
#undef PDMQ_LAUNCH_GQA2_SHAPE
#undef PDMQ_LAUNCH_SHAPE

    CUDA_CHECK(hipGetLastError());
    if (profile_enabled) {
        CUDA_CHECK(hipEventRecord(profile_stop, stream));
        pdmq_kernel_profile profile_host = {};
        CUDA_CHECK(hipMemcpyAsync(&profile_host, profile_dev, sizeof(profile_host), hipMemcpyDeviceToHost, stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        float kernel_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&kernel_ms, profile_start, profile_stop));
        pdmq_profile_record_and_emit(plan, profile_host, kernel_ms);
        CUDA_CHECK(hipEventDestroy(profile_start));
        CUDA_CHECK(hipEventDestroy(profile_stop));
        CUDA_CHECK(hipFree(profile_dev));
    }

    if (V->type == GGML_TYPE_TBQ4_0) {
        const int64_t nrows = Q->ne[1] * Q->ne[2] * Q->ne[3];
        tbq4_rotate_output_cuda((float *) dst->data, nrows, (int) V->ne[0], stream);
    }

    CUDA_CHECK(hipGetLastError());
}

#else

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    return false;
}

static inline bool ggml_cuda_packed16_tbq4_v_enabled() {
    return false;
}

static inline bool ggml_cuda_packed16_dot4_mmq_v_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_F16  ||
           type == GGML_TYPE_PLANAR3_0 ||
           type == GGML_TYPE_ISO3_0;
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
