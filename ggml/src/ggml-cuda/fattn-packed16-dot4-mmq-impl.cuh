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

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-dot4-q8k-kq.cuh"
#include "fattn-packed16-wmma-builtin.cuh"
#include "dot4-packed16/dp16-fa-qpack.cuh"
#include "dot4-packed16/dp16-trace.cuh"

#include <atomic>
#include <cfloat>
#include <climits>
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

static inline const char * pdmq_fattn_node_name(const ggml_tensor * dst) {
    return dst && dst->name[0] ? dst->name : "-";
}

static inline int pdmq_fattn_layer_from_node_name(const ggml_tensor * dst) {
    const char * name = pdmq_fattn_node_name(dst);
    constexpr const char * prefix = "__fattn__-";
    constexpr size_t prefix_len = sizeof("__fattn__-") - 1;
    if (strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long layer = strtol(name + prefix_len, &end, 10);
    return end && end != name + prefix_len ? (int) layer : -1;
}

static constexpr int PDMQ_BM_DECODE32  = 1;
static constexpr int PDMQ_BN_DECODE32  = 32;
static constexpr int PDMQ_BM_VERIFY2_32 = 2;
static constexpr int PDMQ_BN_VERIFY2_32 = 32;
static constexpr int PDMQ_BM_VERIFY4_32 = 4;
static constexpr int PDMQ_BN_VERIFY4_32 = 32;
static constexpr int PDMQ_BM_PREFILL = 16;
static constexpr int PDMQ_BN_PREFILL = 16;
static constexpr int PDMQ_BM_SMALL   = 8;
static constexpr int PDMQ_BN_SMALL   = 32;

static_assert(PDMQ_D % QK8_0 == 0, "packed16_dot4_mmq requires D multiple of QK8_0");
static_assert(PDMQ_BM_DECODE32  * PDMQ_BN_DECODE32  <= PDMQ_THREADS, "M1N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_VERIFY2_32 * PDMQ_BN_VERIFY2_32 <= PDMQ_THREADS, "M2N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_VERIFY4_32 * PDMQ_BN_VERIFY4_32 <= PDMQ_THREADS, "M4N32 maps one QK logit subset per CTA");
static_assert(PDMQ_BM_PREFILL * PDMQ_BN_PREFILL == PDMQ_THREADS, "M16N16 maps one QK logit per thread");
static_assert(PDMQ_BM_SMALL   * PDMQ_BN_SMALL   == PDMQ_THREADS, "M8N32 maps one QK logit per thread");

#ifndef GGML_HIP_PDMQ_QWEN35_DEBUG_ONLY
#define GGML_HIP_PDMQ_QWEN35_DEBUG_ONLY 1
#endif
#ifndef GGML_HIP_PDMQ_COMPILE_PVBLOCK_EXACT
#define GGML_HIP_PDMQ_COMPILE_PVBLOCK_EXACT 0
#endif
#ifndef GGML_HIP_PDMQ_COMPILE_V4_144_PV4
#define GGML_HIP_PDMQ_COMPILE_V4_144_PV4 0
#endif

#define PDMQ_QWEN35_DEBUG_ONLY GGML_HIP_PDMQ_QWEN35_DEBUG_ONLY
#define PDMQ_COMPILE_PVBLOCK_EXACT GGML_HIP_PDMQ_COMPILE_PVBLOCK_EXACT
#define PDMQ_COMPILE_V4_144_PV4 GGML_HIP_PDMQ_COMPILE_V4_144_PV4
#define PDMQ_COMPILE_V4_144 PDMQ_COMPILE_V4_144_PV4
#define PDMQ_COMPILE_V4_ANY PDMQ_COMPILE_V4_144

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
// PDMQ_VPATH is retained only for compact/raw diagnostics:
// stage_f32 | raw_lds_q4 | raw_lds_q4_k16d16_oracle | v4_k16d16 | v4_k16d16_144 | raw_lds_q8_0 | raw_lds_f16.
static const char * pdmq_vpath_env() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH");
    return v ? v : "";
}
static inline bool pdmq_pv_wmma_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_WMMA");
    return v && atoi(v) != 0;
}

static inline bool pdmq_pv_i8_wmma_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I8_WMMA");
    return v && atoi(v) != 0;
}

static inline bool pdmq_pv_i4_wmma_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I4_WMMA");
    return v && atoi(v) != 0;
}

static inline bool pdmq_pv_i4_cache_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I4_CACHE");
    return v && atoi(v) != 0;
}

static inline bool pdmq_v4_144_pv4_requested() {
    const char * old_v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_Q4_LDS");
    if (old_v && atoi(old_v) != 0) {
        GGML_ABORT("GGML_CUDA_ROCM_V4_K16D16_144_Q4_LDS is renamed/demoted; use GGML_CUDA_ROCM_V4_K16D16_144_PV4=1 for the exact PV4 path");
    }
    const char * v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4");
#if !PDMQ_COMPILE_V4_144_PV4
    if (v && atoi(v) != 0) {
        GGML_ABORT("GGML_CUDA_ROCM_V4_K16D16_144_PV4 requires -DGGML_HIP_PDMQ_COMPILE_V4_144_PV4=ON; direct scalar V4 remains the default A path");
    }
    return false;
#else
    return v && atoi(v) != 0;
#endif
}

static inline bool pdmq_pvblock_exact_enabled_by_env() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PVBLOCK_EXACT");
#if !PDMQ_COMPILE_PVBLOCK_EXACT
    if (v && atoi(v) != 0) {
        GGML_ABORT("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PVBLOCK_EXACT requires -DGGML_HIP_PDMQ_COMPILE_PVBLOCK_EXACT=ON in this fast dev build");
    }
    return false;
#else
    return !v || atoi(v) != 0;
#endif
}

static inline int pdmq_pv_i4_cache_refresh_blocks() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I4_CACHE_REFRESH_BLOCKS");
    const int n = v && *v ? atoi(v) : 64;
    return n < 1 ? 1 : (n > 4096 ? 4096 : n);
}

static inline int pdmq_pv_i4_cache_scope() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I4_CACHE_SCOPE");
    if (!v || !*v || strcmp(v, "draft") == 0) return 0;
    if (strcmp(v, "mtp") == 0) return 1;
    if (strcmp(v, "all") == 0 || strcmp(v, "decode_verify") == 0) return 2;
    GGML_ABORT("invalid GGML_CUDA_DP16_FA_PV_I4_CACHE_SCOPE=%s; expected draft, mtp, or all", v);
}

static inline int pdmq_k_scale_group_qblocks() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS");
    if (!v) v = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP");
    if (!v) v = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP_QBLOCKS");
    if (!v) v = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP");
    if (!v || !*v) return 1;
    const int n = atoi(v);
    if (n <= 1) return 1;
    if (n <= 2) return 2;
    if (n <= 4) return 4;
    return PDMQ_D / QK8_0;
}

static inline bool pdmq_pv_i8_wmma_coherent_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I8_WMMA_COHERENT");
    return v && atoi(v) != 0;
}

static inline bool pdmq_pv_i8_wmma_tiny_scalar_requested() {
    const char * v = getenv("GGML_CUDA_DP16_FA_PV_I8_WMMA_TINY_SCALAR");
    return v && atoi(v) != 0;
}

enum packed16_dot4_mmq_v_type {
    PACKED16_DOT4_MMQ_V_Q4_0,
    PACKED16_DOT4_MMQ_V_Q8_0,
    PACKED16_DOT4_MMQ_V_F16,
    PACKED16_DOT4_MMQ_V4_K16D16_144,
};

// Helpers shared by host and device. Pure math, no HIP deps.
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

#if defined(__HIPCC__) || defined(__CUDACC__)
static __host__ __device__ __forceinline__ int pdmq_clamp_i4_signed(const int x) {
    return x < -8 ? -8 : (x > 7 ? 7 : x);
}
static __host__ __device__ __forceinline__ int pdmq_pack_i4x8_unsigned(
        const int q0, const int q1, const int q2, const int q3,
        const int q4, const int q5, const int q6, const int q7) {
#else
static inline int pdmq_clamp_i4_signed(const int x) {
    return x < -8 ? -8 : (x > 7 ? 7 : x);
}
static inline int pdmq_pack_i4x8_unsigned(
        const int q0, const int q1, const int q2, const int q3,
        const int q4, const int q5, const int q6, const int q7) {
#endif
    const uint32_t n0 = uint32_t(q0) & 0x0fu;
    const uint32_t n1 = uint32_t(q1) & 0x0fu;
    const uint32_t n2 = uint32_t(q2) & 0x0fu;
    const uint32_t n3 = uint32_t(q3) & 0x0fu;
    const uint32_t n4 = uint32_t(q4) & 0x0fu;
    const uint32_t n5 = uint32_t(q5) & 0x0fu;
    const uint32_t n6 = uint32_t(q6) & 0x0fu;
    const uint32_t n7 = uint32_t(q7) & 0x0fu;
    return int((n0 << 0) | (n1 << 4) | (n2 << 8) | (n3 << 12) |
               (n4 << 16) | (n5 << 20) | (n6 << 24) | (n7 << 28));
}

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
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_V4_K16D16 ||
           type == GGML_TYPE_V4_K16D16_144;
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
    const bool k_persistent_i32 = K->type == GGML_TYPE_I32 && K->ne[0] * 4 == PDMQ_D;
    const bool k_f16_hotcold_sidecar = K->type == GGML_TYPE_F16 && K->ne[0] == PDMQ_D;
    if (Q->type != GGML_TYPE_F32 || !(k_persistent_i32 || k_f16_hotcold_sidecar) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != PDMQ_D || V->ne[0] != PDMQ_D || dst->ne[0] != PDMQ_D) {
        return false;
    }
    if (Q->ne[1] <= 0 && !ggml_cuda_packed16_dot4_mmq_route_required()) {
        // nq==1 MTP q4 decode is now a standard PDMQ path; route-require is no
        // longer needed just to make the support check accept decode.
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

static __device__ __forceinline__ int pdmq_decode_v_q4_0_block_i8(
        const block_q4_0 & bq,
        const int in) {
    const int iq = in & 15;
    const int shift = in >= 16 ? 4 : 0;
    return int((bq.qs[iq] >> shift) & 0x0f) - 8;
}

static __device__ __forceinline__ float pdmq_decode_v_q4_0_block(
        const block_q4_0 & bq,
        const int in) {
    const float scale = __half2float(bq.d);
    return float(pdmq_decode_v_q4_0_block_i8(bq, in)) * scale;
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


template<int D>
static __device__ __forceinline__ int pdmq_decode_v_q4_0_raw_lds_i8(
        const block_q4_0 * __restrict__ v_raw_tile,
        const int kk,
        const int d) {
    static_assert(D % QK4_0 == 0, "raw q4 LDS V decode requires full q4 blocks");
    const int blk = d / QK4_0;
    const int in  = d & (QK4_0 - 1);
    return pdmq_decode_v_q4_0_block_i8(v_raw_tile[kk * (D / QK4_0) + blk], in);
}

template<int D>
static __device__ __forceinline__ float pdmq_v_q4_0_raw_lds_scale(
        const block_q4_0 * __restrict__ v_raw_tile,
        const int kk,
        const int d) {
    static_assert(D % QK4_0 == 0, "raw q4 LDS V scale requires full q4 blocks");
    const int blk = d / QK4_0;
    return __half2float(v_raw_tile[kk * (D / QK4_0) + blk].d);
}

template<int D, int BN>
static __device__ __forceinline__ void pdmq_pack_v_q4_0_k16d16_oracle_tile(
        const block_q4_0 * __restrict__ v_raw_tile,
        int * __restrict__ v4_k16d16_words,
        const int tile_n) {
    static_assert(D % QK4_0 == 0, "K16D16 q4 oracle requires full q4 blocks");
    static_assert(BN % 16 == 0 || BN == 32, "K16D16 q4 oracle expects a K tile compatible with K16 groups");
    constexpr int GROUPS = (BN + 7) / 8;
    for (int idx = int(threadIdx.x); idx < D * GROUPS; idx += int(blockDim.x)) {
        const int d = idx / GROUPS;
        const int g = idx - d * GROUPS;
        int word = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kk = g * 8 + j;
            const int q = kk < tile_n ? pdmq_decode_v_q4_0_raw_lds_i8<D>(v_raw_tile, kk, d) + 8 : 0;
            word |= (q & 0x0f) << (4 * j);
        }
        v4_k16d16_words[d * GROUPS + g] = word;
    }
    __syncthreads();
}

template<int D, int BN>
static __device__ __forceinline__ int pdmq_decode_v_q4_0_k16d16_oracle_i8(
        const int * __restrict__ v4_k16d16_words,
        const int kk,
        const int d) {
    constexpr int GROUPS = (BN + 7) / 8;
    const int g = kk >> 3;
    const int j = kk & 7;
    const int word = v4_k16d16_words[d * GROUPS + g];
    return int((uint32_t(word) >> (4 * j)) & 0x0fu) - 8;
}

template<int D, int BN, bool V4_K16D16_ORACLE>
static __device__ __forceinline__ int pdmq_decode_v_q4_0_raw_or_k16d16_i8(
        const block_q4_0 * __restrict__ v_raw_tile,
        const int * __restrict__ v4_k16d16_words,
        const int kk,
        const int d) {
    if constexpr (V4_K16D16_ORACLE) {
        return pdmq_decode_v_q4_0_k16d16_oracle_i8<D, BN>(v4_k16d16_words, kk, d);
    } else {
        GGML_UNUSED(v4_k16d16_words);
        return pdmq_decode_v_q4_0_raw_lds_i8<D>(v_raw_tile, kk, d);
    }
}

template<int D, int BN, bool V4_K16D16_ORACLE>
static __device__ __forceinline__ float pdmq_decode_v_q4_0_raw_or_k16d16(
        const block_q4_0 * __restrict__ v_raw_tile,
        const int * __restrict__ v4_k16d16_words,
        const int kk,
        const int d) {
    const int q = pdmq_decode_v_q4_0_raw_or_k16d16_i8<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, d);
    return float(q) * pdmq_v_q4_0_raw_lds_scale<D>(v_raw_tile, kk, d);
}

static __device__ __host__ __forceinline__ size_t pdmq_v_i4_cache_index(
        const int vb,
        const int hk,
        const int k16,
        const int d,
        const int n_heads_k,
        const int cache_k_blocks) {
    return (((size_t(vb) * size_t(n_heads_k) + size_t(hk)) * size_t(cache_k_blocks) + size_t(k16)) * size_t(PDMQ_D)) + size_t(d);
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

static __global__ void pdmq_pack_v_i4_cache_kernel(
        const char * __restrict__ V,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int nk, int n_heads_k, int batch, int block_begin, int block_end, int cache_k_blocks,
        int * __restrict__ cache_words,
        float * __restrict__ cache_scales) {
    const int k16_rel = int(blockIdx.x);
    const int hk      = int(blockIdx.y);
    const int b       = int(blockIdx.z);
    const int d       = int(threadIdx.x);
    if (d >= PDMQ_D || hk >= n_heads_k || b >= batch) {
        return;
    }
    const int k16 = block_begin + k16_rel;
    if (k16 >= block_end || k16 >= cache_k_blocks) {
        return;
    }
    const int k_base = k16 * 16;
    const int tile_n = pdmq_min_i(16, nk - k_base);
    const int vb = v_ne13 > 1 ? (b % int(v_ne13)) : 0;
    const size_t ci = pdmq_v_i4_cache_index(vb, hk, k16, d, n_heads_k, cache_k_blocks);
    float vmax = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        if (kk < tile_n) {
            const float vv = pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k_base + kk, hk, b, d);
            vmax = fmaxf(vmax, fabsf(vv));
        }
    }
    const float v_scale = vmax > 0.0f ? vmax / 7.0f : 1.0f;
    cache_scales[ci] = v_scale;
#pragma unroll
    for (int g = 0; g < PBWMMA_I4_WORDS_PER_K16; ++g) {
        int q[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kk = 8 * g + j;
            if (kk < tile_n) {
                const float vv = pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k_base + kk, hk, b, d);
                q[j] = pdmq_clamp_i4_signed((int) lrintf(vv / v_scale));
            } else {
                q[j] = 0;
            }
        }
        cache_words[ci * size_t(PBWMMA_I4_WORDS_PER_K16) + size_t(g)] =
            pdmq_pack_i4x8_unsigned(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
    }
}

static __device__ __forceinline__ const block_q8_0 * pdmq_v_q8_0_block_ptr(
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
    return (const block_q8_0 *) (ptr + int64_t(blk) * v_nb10);
}

static __device__ __forceinline__ float pdmq_decode_v_q8_0_block(
        const block_q8_0 & bq,
        const int in) {
    return float(bq.qs[in]) * __half2float(bq.d);
}

template<int D>
static __device__ __forceinline__ float pdmq_decode_v_q8_0_raw_lds(
        const block_q8_0 * __restrict__ v_raw_tile,
        const int kk,
        const int d) {
    static_assert(D % QK8_0 == 0, "raw q8 LDS V decode requires full q8 blocks");
    const int blk = d / QK8_0;
    const int in  = d & (QK8_0 - 1);
    return pdmq_decode_v_q8_0_block(v_raw_tile[kk * (D / QK8_0) + blk], in);
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
    const int blk = d / QK8_0;
    const int in  = d & (QK8_0 - 1);
    const block_q8_0 * bq = pdmq_v_q8_0_block_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, blk);
    return pdmq_decode_v_q8_0_block(*bq, in);
}

static __device__ __forceinline__ const half * pdmq_v_f16_ptr(
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
    return (const half *) (V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k) * v_nb11 + int64_t(d) * v_nb10);
}

template<int D>
static __device__ __forceinline__ float pdmq_decode_v_f16_raw_lds(
        const half * __restrict__ v_raw_tile,
        const int kk,
        const int d) {
    return __half2float(v_raw_tile[kk * D + d]);
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
    return __half2float(*pdmq_v_f16_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d));
}

static constexpr int PDMQ_V4_K16D16_144_K = 16;
static constexpr int PDMQ_V4_K16D16_144_D32 = 32;
static constexpr int PDMQ_V4_K16D16_144_WORDS_PER_D = 2;
static constexpr int PDMQ_V4_K16D16_144_PAYLOAD_BYTES = PDMQ_D * PDMQ_V4_K16D16_144_WORDS_PER_D * (int) sizeof(uint32_t);

static __device__ __forceinline__ const char * pdmq_v_v4_k16d16_144_block_ptr(
        const char * __restrict__ V,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k16_base,
        const int hk,
        const int b) {
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    return V + int64_t(vb) * v_nb13 + int64_t(hk) * v_nb12 + int64_t(k16_base) * v_nb11;
}

static __device__ __forceinline__ uint32_t pdmq_v_v4_k16d16_144_payload_word(
        const char * __restrict__ block,
        const int slot,
        const int d) {
    return ((const uint32_t *) block)[d * PDMQ_V4_K16D16_144_WORDS_PER_D + (slot >> 3)];
}

static __device__ __forceinline__ int pdmq_decode_v_v4_k16d16_144_word_i4(
        const uint32_t word,
        const int slot_lo3) {
    return int((word >> (4 * (slot_lo3 & 7))) & 0x0fu) - 8;
}

static __device__ __forceinline__ int pdmq_decode_v_v4_k16d16_144_block_i4(
        const char * __restrict__ block,
        const int slot,
        const int d) {
    return pdmq_decode_v_v4_k16d16_144_word_i4(
        pdmq_v_v4_k16d16_144_payload_word(block, slot, d), slot);
}

static __device__ __forceinline__ float pdmq_v_v4_k16d16_144_block_scale(
        const char * __restrict__ block,
        const int slot,
        const int d) {
    const int d32 = d / PDMQ_V4_K16D16_144_D32;
    const half * scales = (const half *) (block + PDMQ_V4_K16D16_144_PAYLOAD_BYTES);
    return __half2float(scales[d32 * PDMQ_V4_K16D16_144_K + slot]);
}

static __device__ __forceinline__ float pdmq_decode_v_v4_k16d16_144(
        const char * __restrict__ V,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d) {
    const int k16_base = k & ~(PDMQ_V4_K16D16_144_K - 1);
    const int slot = k & (PDMQ_V4_K16D16_144_K - 1);
    const char * block = pdmq_v_v4_k16d16_144_block_ptr(V, v_nb11, v_nb12, v_nb13, v_ne13, k16_base, hk, b);
    const int q = pdmq_decode_v_v4_k16d16_144_block_i4(block, slot, d);
    return float(q) * pdmq_v_v4_k16d16_144_block_scale(block, slot, d);
}

static __device__ __forceinline__ block_q4_0 pdmq_load_v4_k16d16_144_as_q4_0_block(
        const char * __restrict__ V,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int blk) {
    const int k16_base = k & ~(PDMQ_V4_K16D16_144_K - 1);
    const int slot = k & (PDMQ_V4_K16D16_144_K - 1);
    const char * block = pdmq_v_v4_k16d16_144_block_ptr(V, v_nb11, v_nb12, v_nb13, v_ne13, k16_base, hk, b);
    block_q4_0 out;
    out.d = ((const half *) (block + PDMQ_V4_K16D16_144_PAYLOAD_BYTES))[blk * PDMQ_V4_K16D16_144_K + slot];
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int q0 = pdmq_decode_v_v4_k16d16_144_block_i4(block, slot, blk * QK4_0 + j) + 8;
        const int q1 = pdmq_decode_v_v4_k16d16_144_block_i4(block, slot, blk * QK4_0 + QK4_0 / 2 + j) + 8;
        out.qs[j] = uint8_t((q0 & 0x0f) | ((q1 & 0x0f) << 4));
    }
    return out;
}

template<int D>
static __device__ __forceinline__ float pdmq_decode_v_v4_k16d16_144_scale_lds(
        const char * __restrict__ V,
        const int64_t v_nb11,
        const int64_t v_nb12,
        const int64_t v_nb13,
        const int64_t v_ne13,
        const int k,
        const int hk,
        const int b,
        const int d,
        const half * __restrict__ scale_tile,
        const int kk_tile) {
    static_assert(D % PDMQ_V4_K16D16_144_D32 == 0, "V4_144 PV4 scale LDS requires D divisible by 32");
    const int k16_base = k & ~(PDMQ_V4_K16D16_144_K - 1);
    const int slot = k & (PDMQ_V4_K16D16_144_K - 1);
    const char * block = pdmq_v_v4_k16d16_144_block_ptr(V, v_nb11, v_nb12, v_nb13, v_ne13, k16_base, hk, b);
    const int q = pdmq_decode_v_v4_k16d16_144_block_i4(block, slot, d);
    const int d32 = d / PDMQ_V4_K16D16_144_D32;
    return float(q) * __half2float(scale_tile[kk_tile * (D / PDMQ_V4_K16D16_144_D32) + d32]);
}

template<int D>
static __device__ __forceinline__ float pdmq_decode_v_v4_k16d16_144_pv4_cached(
        const uint32_t * __restrict__ payload_words,
        const half * __restrict__ scale_tile,
        const int kk_tile,
        const int d) {
    static_assert(D % PDMQ_V4_K16D16_144_D32 == 0, "V4_144 PV4 cached decode requires D divisible by 32");
    const int q = pdmq_decode_v_v4_k16d16_144_word_i4(payload_words[kk_tile >> 3], kk_tile);
    const int d32 = d / PDMQ_V4_K16D16_144_D32;
    return float(q) * __half2float(scale_tile[kk_tile * (D / PDMQ_V4_K16D16_144_D32) + d32]);
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
        const int d,
        const half * __restrict__ V4_tail = nullptr,
        const int64_t v4_tail_nb10 = 0,
        const int64_t v4_tail_nb11 = 0,
        const int64_t v4_tail_nb12 = 0,
        const int v4_live_nk = INT_MAX) {
    if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0) {
        GGML_UNUSED(V4_tail); GGML_UNUSED(v4_tail_nb10); GGML_UNUSED(v4_tail_nb11); GGML_UNUSED(v4_tail_nb12); GGML_UNUSED(v4_live_nk);
        return pdmq_decode_v_q4_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V_Q8_0) {
        GGML_UNUSED(V4_tail); GGML_UNUSED(v4_tail_nb10); GGML_UNUSED(v4_tail_nb11); GGML_UNUSED(v4_tail_nb12); GGML_UNUSED(v4_live_nk);
        return pdmq_decode_v_q8_0(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else if constexpr (V_TYPE == PACKED16_DOT4_MMQ_V4_K16D16_144) {
        GGML_UNUSED(v_nb10); GGML_UNUSED(V4_tail); GGML_UNUSED(v4_tail_nb10); GGML_UNUSED(v4_tail_nb11); GGML_UNUSED(v4_tail_nb12); GGML_UNUSED(v4_live_nk);
        return pdmq_decode_v_v4_k16d16_144(V, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    } else {
        GGML_UNUSED(V4_tail); GGML_UNUSED(v4_tail_nb10); GGML_UNUSED(v4_tail_nb11); GGML_UNUSED(v4_tail_nb12); GGML_UNUSED(v4_live_nk);
        return pdmq_decode_v_f16(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
    }
}

static __device__ __forceinline__ bool pdmq_mask_keep_and_bias(
        const char * __restrict__ mask,
        const int64_t mask_ne00,
        const int64_t mask_ne01,
        const int64_t mask_ne03,
        const int64_t mask_nb00,
        const int64_t mask_nb01,
        const int64_t mask_nb03,
        const int q,
        const int k,
        const int b,
        float * __restrict__ bias) {
    *bias = 0.0f;
    if (!mask) return true;
    if (q < 0 || q >= mask_ne01 || k < 0 || k >= mask_ne00) return false;
    const int mb = mask_ne03 > 1 ? (b % mask_ne03) : 0;
    const half hv = *(const half *) (mask + int64_t(mb) * mask_nb03 + int64_t(q) * mask_nb01 + int64_t(k) * mask_nb00);
    const float v = __half2float(hv);
    // Explicit masks in FA are row-local keep/drop decisions plus an optional
    // finite bias.  Treat -inf (and fp16-saturated large negative masks) as a
    // dropped cell before softmax so masked probabilities are exactly zero and
    // PV cannot consume a stale/broadcast probability for another query row.
    if (!isfinite(v) || v <= -60000.0f) {
        return false;
    }
    *bias = v;
    return true;
}

static __device__ __forceinline__ long long pdmq_mask_debug_byte_offset(
        const int64_t mask_ne00,
        const int64_t mask_ne01,
        const int64_t mask_ne03,
        const int64_t mask_nb00,
        const int64_t mask_nb01,
        const int64_t mask_nb03,
        const int q,
        const int k,
        const int b) {
    if (q < 0 || q >= mask_ne01 || k < 0 || k >= mask_ne00) return -1ll;
    const int mb = mask_ne03 > 1 ? (b % mask_ne03) : 0;
    return (long long) (int64_t(mb) * mask_nb03 + int64_t(q) * mask_nb01 + int64_t(k) * mask_nb00);
}

static __device__ __forceinline__ float pdmq_mask_debug_raw_val(
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
    const long long off = pdmq_mask_debug_byte_offset(mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b);
    if (off < 0) return -INFINITY;
    const half hv = *(const half *) (mask + off);
    return __half2float(hv);
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
static __device__ __forceinline__ void pdmq_quantize_q_program_rows(
        const float * __restrict__ Q,
        const int64_t q_nb01,
        const int64_t q_nb02,
        const int64_t q_nb03,
        const int q0,
        const int hq,
        const int b,
        const int nq,
        const dp16_fa_qblock_program & qblock_program,
        const int row_kind_filter,
        int   (&q_payload)[BM][D / 4 + 1],
        float (&q_scales) [BM][D / QK8_0 + 1]) {
    constexpr int QBLOCKS = D / QK8_0;
    constexpr int WORDS_PER_BLOCK = QK8_0 / 4;

    for (int linear = int(threadIdx.x); linear < BM * QBLOCKS; linear += int(blockDim.x)) {
        const int qr = linear / QBLOCKS;
        const int qb = linear - qr * QBLOCKS;
        const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
        const bool kind_ok = row_kind_filter < 0 || !qblock_program.enabled || qblock_program.row_kind[qr] == row_kind_filter;
        const bool row_ok  = kind_ok && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq);

        if (!row_ok) {
            continue;
        }

        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            const float x = pdmq_load_q_f32(Q, q_nb01, q_nb02, q_nb03, q, hq, b, qb * QK8_0 + i);
            amax = fmaxf(amax, fabsf(x));
        }

        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        q_scales[qr][qb] = scale;

#pragma unroll
        for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
            int qs[4] = {0, 0, 0, 0};
            if (amax > 0.0f) {
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
        const float k_scale = __half2float(k_row_scales[qb]);
        sum += float(acc) * q_scales[qr][qb] * k_scale;
    }
    return sum;
}

static __device__ __forceinline__ unsigned int pdmq_f32_bits(const float x) {
    union {
        float f;
        unsigned int u;
    } v;
    v.f = x;
    return v.u;
}

// ── GQA2 Kernel (2 Q heads per CTA, M16N16 or M8N32) ──────────────
template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool RAW_LDS_Q4, bool KSHARED, bool V4_K16D16_ORACLE = false>
static __global__ __launch_bounds__(PDMQ_THREADS, 1) void packed16_dot4_mmq_gqa2_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ V,
        const half  * __restrict__ V4_tail,
        float       * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int64_t v4_tail_nb10, int64_t v4_tail_nb11, int64_t v4_tail_nb12,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows, int q_offset, float attention_scale) {

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
    const int v4_live_nk = CAUSAL_MASK ? pdmq_min_i(nk, q_offset + nq) : nk;

    if (q0 >= nq) return;

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ int   q_tile_i32[GQA_GROUP][BM][D / 4 + 1];
    __shared__ float q_tile_scales[GQA_GROUP][BM][D / QK8_0 + 1];
    __shared__ float logits[BM][BN + 1];
    static constexpr bool RAW_Q4_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0;
    static constexpr bool RAW_Q8_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q8_0;
    static constexpr bool RAW_F16_LDS = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_F16;
    static_assert(!V4_K16D16_ORACLE || RAW_Q4_LDS, "V4_K16D16 oracle requires raw q4 LDS input");

    __shared__ float row_m[GQA_GROUP][BM + 1];
    __shared__ float row_l[GQA_GROUP][BM + 1];
    __shared__ float old_s[GQA_GROUP][BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ block_q4_0 v_raw_tile[RAW_Q4_LDS ? BN * (D / QK4_0) : 1];
    __shared__ int   v4_k16d16_words[V4_K16D16_ORACLE ? D * ((BN + 7) / 8) : 1];
    __shared__ block_q8_0 v_raw_q8_tile[RAW_Q8_LDS ? BN * (D / QK8_0) : 1];
    __shared__ half  v_raw_f16_tile[RAW_F16_LDS ? BN * D : 1];
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
                    ? pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk) : 0.0f;
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
            if constexpr (V4_K16D16_ORACLE) {
                pdmq_pack_v_q4_0_k16d16_oracle_tile<D, BN>(v_raw_tile, v4_k16d16_words, tile_n);
            }
        } else if constexpr (RAW_Q8_LDS) {
            constexpr int V_BLOCKS = D / QK8_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                v_raw_q8_tile[kk * V_BLOCKS + blk] = *pdmq_v_q8_0_block_ptr(
                    V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
            }
            __syncthreads();
        } else if constexpr (RAW_F16_LDS) {
            constexpr int HALF_PER_VEC = int(sizeof(int4) / sizeof(half));
            constexpr int VECS_PER_ROW = D / HALF_PER_VEC;
            static_assert(D % HALF_PER_VEC == 0, "raw f16 LDS vector copy requires int4-aligned rows");
            if (v_nb10 == int64_t(sizeof(half))) {
                for (int v_idx = tid; v_idx < tile_n * VECS_PER_ROW; v_idx += int(blockDim.x)) {
                    const int kk  = v_idx / VECS_PER_ROW;
                    const int vec = v_idx - kk * VECS_PER_ROW;
                    const int d   = vec * HALF_PER_VEC;
                    const int k   = k0 + kk;
                    const half * src = pdmq_v_f16_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
                    ((int4 *) (v_raw_f16_tile + kk * D))[vec] = *((const int4 *) src);
                }
            } else {
                for (int v_idx = tid; v_idx < tile_n * D; v_idx += int(blockDim.x)) {
                    const int kk = v_idx / D;
                    const int d  = v_idx - kk * D;
                    v_raw_f16_tile[kk * D + d] = *pdmq_v_f16_ptr(
                        V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, d);
                }
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
                k_payload_s[kk * (KP_PER_ROW + 1) + w] = k_payload[k_row * KP_PER_ROW + w];
            }
            for (int idx = tid; idx < tile_n * KS_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KS_PER_ROW, s_val = idx - kk * KS_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_scales_s[kk * (KS_PER_ROW + 1) + s_val] = k_scales[k_row * KS_PER_ROW + s_val];
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
                float mask_bias = 0.0f;
                if (valid && mask) {
                    valid = pdmq_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b, &mask_bias);
                }
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
                    s_val = pdmq_qk_dot<BM, D>(q_tile_i32[s], q_tile_scales[s], qr, kp_row, ks_row) * attention_scale + mask_bias;
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


            // PV accumulate. For direct-V, run kk-major so each V[k,d]
            // is decoded/loaded once per thread lane and reused across query rows.
            // q4_0 can optionally source the raw block from LDS; q8_0/f16 use the
            // same reuse pattern through their typed direct decoders.
            if (tid < D) {
                if constexpr (!STAGE_V) {
                    float acc[BM];
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        acc[qr] = q < nq ? out[s][qr] * old_s[s][qr] : 0.0f;
                    }
                    for (int kk = 0; kk < BN; ++kk) {
                        const float vv = kk < tile_n
                            ? (RAW_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                : (RAW_Q8_LDS
                                    ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                    : (RAW_F16_LDS
                                        ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                        : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk))))
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
                            if (p != 0.0f) {
                                acc += p * v_tile[kk * (D + 1) + tid];
                            }
                        }
                        out[s][qr] = acc;
                    }
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


// ── GQAX/XQA Kernel (9B grouped-Q-head verifier experiment) ───────
// Unlike the older GQA2 canary, this keeps all slot probabilities live until
// PV so a q4 V value is decoded once per K/V tile element and reused across
// grouped Q heads. This is deliberately narrow: it is a verifier dataflow
// experiment for hq=16,hk=4,V=q4_0, not a generic policy.
template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool RAW_LDS_Q4, bool KSHARED, int GQA_GROUP, bool PV_WMMA = false, bool PV_I8_WMMA = false, bool PV_I8_FULL_TILES_ONLY = true, bool PV_I8_TINY_SCALAR = false, bool PV_I4_WMMA = false, bool PV_BLOCK_EXACT = false, bool V4_K16D16_ORACLE = false>
static __global__ __launch_bounds__(PDMQ_THREADS, 1) void packed16_dot4_mmq_gqax_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ V,
        const half  * __restrict__ V4_tail,
        float       * __restrict__ dst,
        float       * __restrict__ partial_m,
        float       * __restrict__ partial_l,
        float       * __restrict__ partial_out,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int64_t v4_tail_nb10, int64_t v4_tail_nb11, int64_t v4_tail_nb12,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows, int q_offset, float attention_scale,
        int batch, int split_k_factor,
        const int   * __restrict__ qpack_payload,
        const float * __restrict__ qpack_scales,
        const dp16_fa_qblock_program qblock_program,
        const int   * __restrict__ v_i4_cache_words,
        const float * __restrict__ v_i4_cache_scales,
        int v_i4_cache_k_blocks) {

    static_assert(D == PDMQ_D, "packed16_dot4_mmq_gqax is D256-only");
    static_assert(BM * BN <= PDMQ_THREADS, "packed16_dot4_mmq_gqax maps one QK logit subset per CTA");
    static_assert(GQA_GROUP == 1 || GQA_GROUP == 2 || GQA_GROUP == 3 || GQA_GROUP == 4 || GQA_GROUP == 6, "packed16_dot4_mmq_gqax supports GQA groups 1, 2, 3, 4, or 6");
    static constexpr int PV_WMMA_ROWS = GQA_GROUP * BM;
    static_assert(!PV_WMMA || (PV_WMMA_ROWS <= 32), "PDMQ PV-WMMA row-blocked micro-GEMM supports at most two 16-row P tiles per CTA");
    static_assert(!PV_WMMA || (BN % 16 == 0), "PDMQ PV-WMMA requires 16-token K tiles");
    static_assert(!PV_WMMA || (D % 16 == 0), "PDMQ PV-WMMA requires 16-column D tiles");

    const int tid = int(threadIdx.x);
    const int q_tile = int(blockIdx.x);
    const int gy     = int(blockIdx.y);
    const int bz     = int(blockIdx.z);
    const bool split_mode = partial_out != nullptr;
    const int split_k     = split_mode ? split_k_factor : 1;
    const int b           = split_mode ? bz / split_k : bz;
    const int split_idx   = split_mode ? (bz - b * split_k) : 0;

    const int groups_per_kv = CEIL_DIV(gqa_ratio, GQA_GROUP);
    const int hk            = gy / groups_per_kv;
    const int local_group   = gy - hk * groups_per_kv;
    const int hq_base       = hk * gqa_ratio + local_group * GQA_GROUP;
    const int q0            = q_tile * BM;
    const int v4_live_nk    = CAUSAL_MASK ? pdmq_min_i(nk, q_offset + nq) : nk;

    if (hk >= n_heads_k || q0 >= nq || b >= batch) {
        return;
    }

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    const int k_blocks_total = CEIL_DIV(nk, BN);
    const int k_block_begin  = split_mode ? (k_blocks_total * split_idx) / split_k : 0;
    const int k_block_end    = split_mode ? (k_blocks_total * (split_idx + 1)) / split_k : k_blocks_total;
    const int k_begin        = k_block_begin * BN;
    const int k_end          = pdmq_min_i(nk, k_block_end * BN);

    const int qblock_invalid_mask = qblock_program.enabled
        ? (dp16_fa_qblock_full_mask(qblock_program.rows_per_cta) & ~qblock_program.row_valid_mask)
        : 0;
    if (qblock_invalid_mask != 0) {
        const size_t rows_per_split = size_t(batch) * size_t(nq) * size_t(n_heads_q);
        for (int idx = tid; idx < GQA_GROUP * BM * D; idx += int(blockDim.x)) {
            const int d    = idx % D;
            const int prow = idx / D;
            const int slot = prow / BM;
            const int qr   = prow - slot * BM;
            if ((qblock_invalid_mask & (1 << qr)) == 0) {
                continue;
            }
            const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) || q < 0 || q >= nq) {
                continue;
            }
            const size_t row = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq));
            if (split_mode) {
                const size_t partial_row = size_t(split_idx) * rows_per_split + row;
                if (d == 0) {
                    partial_m[partial_row] = pdmq_neg();
                    partial_l[partial_row] = 0.0f;
                }
                partial_out[partial_row * size_t(D) + size_t(d)] = 0.0f;
            } else {
                dst[row * size_t(D) + size_t(d)] = 0.0f;
            }
        }
        __syncthreads();
    }

    // Fast path for intentionally oversubscribed split-K CTAs.  Raw split8
    // restores occupancy for GQA2X, but for nk=256/M4N64 only four K tiles
    // contain work.  Empty split shards must still publish m/l sentinels so
    // the merge can ignore them; they do not need Q quantization, V staging,
    // QK/PV work, or partial_out writes.
    if (split_mode && k_begin >= k_end) {
        const size_t rows_per_split = size_t(batch) * size_t(nq) * size_t(n_heads_q);
        if (tid == 0) {
#pragma unroll
            for (int slot = 0; slot < GQA_GROUP; ++slot) {
                const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio)) continue;
#pragma unroll
                for (int qr = 0; qr < BM; ++qr) {
                    const int q = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                    if (!dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) continue;
                    const size_t row = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq));
                    const size_t partial_row = size_t(split_idx) * rows_per_split + row;
                    partial_m[partial_row] = pdmq_neg();
                    partial_l[partial_row] = 0.0f;
                }
            }
        }
        return;
    }

    __shared__ int   q_tile_i32[GQA_GROUP][BM][D / 4 + 1];
    __shared__ float q_tile_scales[GQA_GROUP][BM][D / QK8_0 + 1];
    __shared__ float logits[GQA_GROUP][BM][BN + 1];
    static constexpr bool RAW_Q4_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0;
    static constexpr bool RAW_Q8_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q8_0;
    static constexpr bool RAW_F16_LDS = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_F16;
    static constexpr bool V4_144_PV4_COMPILED = PDMQ_COMPILE_V4_144_PV4 && !STAGE_V && !PV_WMMA && V_TYPE == PACKED16_DOT4_MMQ_V4_K16D16_144;
    static_assert(!V4_K16D16_ORACLE || RAW_Q4_LDS, "V4_K16D16 oracle requires raw q4 LDS input");
    static_assert(!V4_K16D16_ORACLE || !PV_WMMA, "V4_K16D16 oracle is scalar-PV only");

    __shared__ float row_m[GQA_GROUP][BM + 1];
    __shared__ float row_l[GQA_GROUP][BM + 1];
    __shared__ float old_s[GQA_GROUP][BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ block_q4_0 v_raw_tile[RAW_Q4_LDS ? BN * (D / QK4_0) : 1];
    __shared__ half  v4_144_scale_tile[V4_144_PV4_COMPILED ? BN * (D / PDMQ_V4_K16D16_144_D32) : 1];
    __shared__ int   v4_k16d16_words[V4_K16D16_ORACLE ? D * ((BN + 7) / 8) : 1];
    __shared__ block_q8_0 v_raw_q8_tile[RAW_Q8_LDS ? BN * (D / QK8_0) : 1];
    __shared__ half  v_raw_f16_tile[RAW_F16_LDS ? BN * D : 1];
    __shared__ half  v_wmma_tile[PV_WMMA ? BN * D : 1];
    __shared__ int   k_payload_s[KSHARED ? BN * (PDMQ_D / 4 + 1) : 1];
    __shared__ half  k_scales_s [KSHARED ? BN * (PDMQ_D / QK8_0 + 1) : 1];
    __shared__ float out_wmma[PV_WMMA ? PV_WMMA_ROWS * D : 1];

    const bool v4_144_pv4_active = V4_144_PV4_COMPILED &&
        qblock_program.v_decode_mode == DP16_FA_QBLOCK_V_DECODE_RAW_LDS;

    for (int idx = tid; idx < GQA_GROUP * BM; idx += int(blockDim.x)) {
        const int s = idx / BM;
        const int qr = idx - s * BM;
        row_m[s][qr] = pdmq_neg();
        row_l[s][qr] = 0.0f;
        old_s[s][qr] = 0.0f;
    }

    float out[GQA_GROUP][BM];
#pragma unroll
    for (int s = 0; s < GQA_GROUP; ++s) {
#pragma unroll
        for (int qr = 0; qr < BM; ++qr) {
            out[s][qr] = 0.0f;
        }
    }
    if constexpr (PV_WMMA) {
        for (int idx = tid; idx < PV_WMMA_ROWS * D; idx += int(blockDim.x)) {
            out_wmma[idx] = 0.0f;
        }
    }
    __syncthreads();

    if (qpack_payload && qpack_scales) {
        constexpr int Q_WORDS  = D / 4;
        constexpr int Q_BLOCKS = D / QK8_0;
        for (int idx = tid; idx < GQA_GROUP * BM * Q_WORDS; idx += int(blockDim.x)) {
            const int slot = idx / (BM * Q_WORDS);
            const int rem  = idx - slot * BM * Q_WORDS;
            const int qr   = rem / Q_WORDS;
            const int w    = rem - qr * Q_WORDS;
            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) ||
                    !dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                continue;
            }
            const size_t row = (size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q);
            q_tile_i32[slot][qr][w] = qpack_payload[row * size_t(Q_WORDS) + size_t(w)];
        }
        for (int idx = tid; idx < GQA_GROUP * BM * Q_BLOCKS; idx += int(blockDim.x)) {
            const int slot = idx / (BM * Q_BLOCKS);
            const int rem  = idx - slot * BM * Q_BLOCKS;
            const int qr   = rem / Q_BLOCKS;
            const int qb   = rem - qr * Q_BLOCKS;
            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) ||
                    !dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                continue;
            }
            const size_t row = (size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q);
            q_tile_scales[slot][qr][qb] = qpack_scales[row * size_t(Q_BLOCKS) + size_t(qb)];
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < GQA_GROUP; ++s) {
            const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, s);
            const bool slot_valid = dp16_fa_qblock_program_head_live(qblock_program, hq, s, hk, n_heads_q, gqa_ratio);
            if (!slot_valid) {
                continue;
            }
            if (qblock_program.q_precision_mode == DP16_FA_QBLOCK_Q_PRECISION_INLINE_ALL) {
                pdmq_quantize_q_program_rows<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq, b, nq,
                        qblock_program, -1, q_tile_i32[s], q_tile_scales[s]);
            } else if (qblock_program.q_precision_mode == DP16_FA_QBLOCK_Q_PRECISION_TARGET_INLINE) {
                pdmq_quantize_q_program_rows<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq, b, nq,
                        qblock_program, DP16_FA_QBLOCK_ROW_TARGET, q_tile_i32[s], q_tile_scales[s]);
            } else if (qblock_program.q_precision_mode == DP16_FA_QBLOCK_Q_PRECISION_DRAFT_INLINE) {
                pdmq_quantize_q_program_rows<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq, b, nq,
                        qblock_program, DP16_FA_QBLOCK_ROW_DRAFT, q_tile_i32[s], q_tile_scales[s]);
            }
        }
    } else {
#pragma unroll
        for (int s = 0; s < GQA_GROUP; ++s) {
            const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, s);
            const bool slot_valid = dp16_fa_qblock_program_head_live(qblock_program, hq, s, hk, n_heads_q, gqa_ratio);
            if (slot_valid) {
                pdmq_quantize_q_program_rows<BM, D>(Q, q_nb01, q_nb02, q_nb03, q0, hq, b, nq,
                        qblock_program, -1, q_tile_i32[s], q_tile_scales[s]);
            }
        }
    }

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = pdmq_min_i(BN, nk - k0);

        if constexpr (CAUSAL_MASK) {
            const int q_last = dp16_fa_qblock_program_q_last_live(qblock_program, q0, BM, nq);
            if (k0 > q_offset + q_last) break;
        }

        static constexpr bool V4_144_PV4_WORD_CACHE = V4_144_PV4_COMPILED && (BN % 8 == 0);
        uint32_t v4_144_pv4_payload_words[V4_144_PV4_WORD_CACHE ? (BN / 8) : 1];
        const bool v4_144_pv4_words_active = v4_144_pv4_active && ((k0 & 7) == 0);


        if constexpr (STAGE_V) {
            for (int v_idx = tid; v_idx < BN * D; v_idx += int(blockDim.x)) {
                const int kk = v_idx / D;
                const int d  = v_idx - kk * D;
                const int k  = k0 + kk;
                v_tile[kk * (D + 1) + d] = kk < tile_n
                    ? pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk)
                    : 0.0f;
            }
            __syncthreads();
        } else if constexpr (RAW_Q4_LDS || V4_144_PV4_COMPILED) {
            constexpr int V_BLOCKS = D / QK4_0;
            if constexpr (RAW_Q4_LDS) {
                if (!(PV_I4_WMMA && v_i4_cache_words != nullptr && v_i4_cache_scales != nullptr)) {
                    for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                        const int kk = v_idx / V_BLOCKS;
                        const int blk = v_idx - kk * V_BLOCKS;
                        v_raw_tile[kk * V_BLOCKS + blk] = *pdmq_v_q4_0_block_ptr(
                            V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
                    }
                    __syncthreads();
                    if constexpr (V4_K16D16_ORACLE) {
                        pdmq_pack_v_q4_0_k16d16_oracle_tile<D, BN>(v_raw_tile, v4_k16d16_words, tile_n);
                    }
                }
            } else if (v4_144_pv4_active) {
                constexpr int V_SCALE_BLOCKS = D / PDMQ_V4_K16D16_144_D32;
                for (int v_idx = tid; v_idx < tile_n * V_SCALE_BLOCKS; v_idx += int(blockDim.x)) {
                    const int kk = v_idx / V_SCALE_BLOCKS;
                    const int d32 = v_idx - kk * V_SCALE_BLOCKS;
                    const int k = k0 + kk;
                    const int k16_base = k & ~(PDMQ_V4_K16D16_144_K - 1);
                    const int slot = k & (PDMQ_V4_K16D16_144_K - 1);
                    const char * block = pdmq_v_v4_k16d16_144_block_ptr(V, v_nb11, v_nb12, v_nb13, v_ne13, k16_base, hk, b);
                    v4_144_scale_tile[kk * V_SCALE_BLOCKS + d32] = ((const half *) (block + PDMQ_V4_K16D16_144_PAYLOAD_BYTES))[d32 * PDMQ_V4_K16D16_144_K + slot];
                }
                if constexpr (V4_144_PV4_WORD_CACHE) {
#pragma unroll
                    for (int kg = 0; kg < BN / 8; ++kg) {
                        const int kk = kg * 8;
                        if (v4_144_pv4_words_active && kk < tile_n) {
                            const int k = k0 + kk;
                            const int k16_base = k & ~(PDMQ_V4_K16D16_144_K - 1);
                            const int slot = k & (PDMQ_V4_K16D16_144_K - 1);
                            const char * block = pdmq_v_v4_k16d16_144_block_ptr(V, v_nb11, v_nb12, v_nb13, v_ne13, k16_base, hk, b);
                            v4_144_pv4_payload_words[kg] = pdmq_v_v4_k16d16_144_payload_word(block, slot, tid);
                        } else {
                            v4_144_pv4_payload_words[kg] = 0;
                        }
                    }
                }
                __syncthreads();
            }
        } else if constexpr (RAW_Q8_LDS) {
            constexpr int V_BLOCKS = D / QK8_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                v_raw_q8_tile[kk * V_BLOCKS + blk] = *pdmq_v_q8_0_block_ptr(
                    V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
            }
            __syncthreads();
        } else if constexpr (RAW_F16_LDS) {
            constexpr int HALF_PER_VEC = int(sizeof(int4) / sizeof(half));
            constexpr int VECS_PER_ROW = D / HALF_PER_VEC;
            static_assert(D % HALF_PER_VEC == 0, "raw f16 LDS vector copy requires int4-aligned rows");
            if (v_nb10 == int64_t(sizeof(half))) {
                for (int v_idx = tid; v_idx < tile_n * VECS_PER_ROW; v_idx += int(blockDim.x)) {
                    const int kk  = v_idx / VECS_PER_ROW;
                    const int vec = v_idx - kk * VECS_PER_ROW;
                    const int d   = vec * HALF_PER_VEC;
                    const int k   = k0 + kk;
                    const half * src = pdmq_v_f16_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
                    ((int4 *) (v_raw_f16_tile + kk * D))[vec] = *((const int4 *) src);
                }
            } else {
                for (int v_idx = tid; v_idx < tile_n * D; v_idx += int(blockDim.x)) {
                    const int kk = v_idx / D;
                    const int d  = v_idx - kk * D;
                    v_raw_f16_tile[kk * D + d] = *pdmq_v_f16_ptr(
                        V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, d);
                }
            }
            __syncthreads();
        }


        if constexpr (PV_WMMA) {
            // Experimental/reference PV-WMMA path: decode V to one f16 tile per
            // K block, then feed WMMA fragments from that tile. This remains a
            // diagnostic backend; the current promotion candidate is the native
            // QBlock/DOT4 scheduling path, not direct PV arithmetic.
            for (int v_idx = tid; v_idx < BN * D; v_idx += int(blockDim.x)) {
                const int kk = v_idx / D;
                const int d  = v_idx - kk * D;
                if (kk < tile_n) {
                    const float vv = STAGE_V
                        ? v_tile[kk * (D + 1) + d]
                        : (RAW_Q4_LDS
                            ? pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kk, d)
                            : (RAW_Q8_LDS
                                ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, d)
                                : (RAW_F16_LDS
                                    ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, d)
                                    : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, d, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk))));
                    v_wmma_tile[v_idx] = __float2half(vv);
                } else {
                    v_wmma_tile[v_idx] = __float2half(0.0f);
                }
            }
            __syncthreads();
        }


        if constexpr (KSHARED) {
            constexpr int KP_PER_ROW = PDMQ_D / 4;
            constexpr int KS_PER_ROW = PDMQ_D / QK8_0;
            for (int idx = tid; idx < tile_n * KP_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KP_PER_ROW, w = idx - kk * KP_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_payload_s[kk * (KP_PER_ROW + 1) + w] = k_payload[k_row * KP_PER_ROW + w];
            }
            for (int idx = tid; idx < tile_n * KS_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KS_PER_ROW, s_val = idx - kk * KS_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_scales_s[kk * (KS_PER_ROW + 1) + s_val] = k_scales[k_row * KS_PER_ROW + s_val];
            }
            __syncthreads();
        }


        for (int idx = tid; idx < GQA_GROUP * BM * BN; idx += int(blockDim.x)) {
            const int slot = idx / (BM * BN);
            const int rem  = idx - slot * BM * BN;
            const int qr   = rem / BN;
            const int kk   = rem - qr * BN;
            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            const int k    = k0 + kk;

            float s_val = pdmq_neg();
            bool valid = dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq) && kk < tile_n && k < nk;
            if constexpr (CAUSAL_MASK) {
                valid = valid && (k <= q_offset + q);
            }
            float mask_bias = 0.0f;
            if (valid && mask) {
                valid = pdmq_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b, &mask_bias);
            }
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
                s_val = pdmq_qk_dot<BM, D>(q_tile_i32[slot], q_tile_scales[slot], qr, kp_row, ks_row) * attention_scale + mask_bias;
            }
            logits[slot][qr][kk] = s_val;
        }
        __syncthreads();


        for (int idx = tid; idx < GQA_GROUP * BM; idx += int(blockDim.x)) {
            const int slot = idx / BM;
            const int qr   = idx - slot * BM;
            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            float tile_m = pdmq_neg();
            if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                    dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    tile_m = fmaxf(tile_m, logits[slot][qr][kk]);
                }
            }
            const float m_new = fmaxf(row_m[slot][qr], tile_m);
            const float alpha = row_l[slot][qr] > 0.0f && pdmq_logit_is_live(row_m[slot][qr]) && pdmq_logit_is_live(m_new)
                ? expf(row_m[slot][qr] - m_new) : 0.0f;
            float tile_l = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BN; ++kk) {
                const float lv = logits[slot][qr][kk];
                const float p = pdmq_logit_is_live(lv) && pdmq_logit_is_live(m_new) ? expf(lv - m_new) : 0.0f;
                logits[slot][qr][kk] = p;
                tile_l += p;
            }
            row_l[slot][qr] = row_l[slot][qr] * alpha + tile_l;
            row_m[slot][qr] = m_new;
            old_s[slot][qr] = alpha;
        }
        __syncthreads();


        if constexpr (PV_WMMA) {
            const int lane    = tid & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int wave    = tid >> 5;
            constexpr int D_TILES = D / 16;
            constexpr int WAVES_PER_CTA = PDMQ_THREADS / 32;

            if constexpr (PV_I8_WMMA && PV_I8_TINY_SCALAR && !PV_I8_FULL_TILES_ONLY && RAW_Q4_LDS && PV_WMMA_ROWS < 16) {
                // Coherent tiny-M decode path: mirror the INT-FA contract without
                // launching a mostly padded 16-row WMMA tile for M=GQA_GROUP*BM < 16:
                //   P_i8 = round(127 * softmax(P)), V_i8 ~= V / V_scale,
                //   O += (P_i8 @ V_i8) * V_scale / 127.
                if (tid < D) {
#pragma unroll
                    for (int p_row = 0; p_row < PV_WMMA_ROWS; ++p_row) {
                        const int slot = p_row / BM;
                        const int qr   = p_row - slot * BM;
                        const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                        const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                        if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) || !dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                            continue;
                        }
                        float pv_acc = 0.0f;
#pragma unroll
                        for (int kc_base = 0; kc_base < BN; kc_base += 16) {
                            float vmax = 0.0f;
#pragma unroll
                            for (int kk = 0; kk < 16; ++kk) {
                                const int kc = kc_base + kk;
                                if (kc < tile_n) {
                                    const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, tid);
                                    vmax = fmaxf(vmax, fabsf(vv));
                                }
                            }
                            const float v_scale = vmax > 0.0f ? vmax / 127.0f : 1.0f;
                            int pv_acc_i = 0;
#pragma unroll
                            for (int kk = 0; kk < 16; ++kk) {
                                const int kc = kc_base + kk;
                                if (kc < tile_n) {
                                    const int av = pdmq_clamp_i8((int) lrintf(127.0f * logits[slot][qr][kc]));
                                    const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, tid);
                                    const int bv = pdmq_clamp_i8((int) lrintf(vv / v_scale));
                                    pv_acc_i += av * bv;
                                }
                            }
                            pv_acc += float(pv_acc_i) * (v_scale / 127.0f);
                        }
                        const int out_idx = p_row * D + tid;
                        out_wmma[out_idx] = out_wmma[out_idx] * old_s[slot][qr] + pv_acc;
                    }
                }
            } else {

            // Treat PV as a small GEMM: P[M x BN] * V[BN x D]. RDNA WMMA is
            // 16x16x16, so row-block M in chunks of 16. This keeps GQA4/M4
            // as one full tile and lets 27B GQA6/M4 run as 16+8 padded rows.
#pragma unroll
            for (int row_block = 0; row_block < PV_WMMA_ROWS; row_block += 16) {
#pragma unroll
                for (int d_tile_base = 0; d_tile_base < D_TILES; d_tile_base += WAVES_PER_CTA) {
                    const int d_tile = d_tile_base + wave;
                    if (d_tile >= D_TILES) {
                        continue;
                    }
                    if constexpr (PV_I4_WMMA && RAW_Q4_LDS) {
                        // Native RDNA3 i4 PV in INT-FA form for raw q4 LDS.
                        if (!PV_I8_FULL_TILES_ONLY || row_block + 16 <= PV_WMMA_ROWS) {
                            const int d_col = d_tile * 16 + pbwmma_i8_b_col_from_lane_lo(lane_lo);
                            float pv_acc_f[8] = {0,0,0,0,0,0,0,0};
#pragma unroll
                            for (int kc_base = 0; kc_base < BN; kc_base += 16) {
                                pbwmma_v2i32 a_frag;
                                pbwmma_v2i32 b_frag;
                                const int a_p_row = row_block + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                                const int vb = v_ne13 > 1 ? (b % int(v_ne13)) : 0;
                                const int k16_abs = (k0 + kc_base) >> 4;
                                const bool use_v_i4_cache = v_i4_cache_words != nullptr && v_i4_cache_scales != nullptr &&
                                    v_i4_cache_k_blocks > 0 && k16_abs < v_i4_cache_k_blocks;
                                const size_t vci = use_v_i4_cache
                                    ? pdmq_v_i4_cache_index(vb, hk, k16_abs, d_col, n_heads_k, v_i4_cache_k_blocks)
                                    : size_t(0);
                                float vmax_lane = 0.0f;
                                if (!use_v_i4_cache) {
#pragma unroll
                                    for (int kk = 0; kk < 16; ++kk) {
                                        const int kc = kc_base + kk;
                                        if (kc < tile_n) {
                                            const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, d_col);
                                            vmax_lane = fmaxf(vmax_lane, fabsf(vv));
                                        }
                                    }
                                }
                                const float v_scale_lane = use_v_i4_cache ? v_i4_cache_scales[vci] : (vmax_lane > 0.0f ? vmax_lane / 7.0f : 1.0f);
#pragma unroll
                                for (int g = 0; g < PBWMMA_I4_WORDS_PER_K16; ++g) {
                                    int av[8];
                                    int bv[8];
#pragma unroll
                                    for (int j = 0; j < 8; ++j) {
                                        const int kk = 8 * g + j;
                                        const int kc = kc_base + kk;
                                        if (a_p_row < PV_WMMA_ROWS && kc < tile_n) {
                                            const int slot = a_p_row / BM;
                                            const int qr   = a_p_row - slot * BM;
                                            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                            const float af = (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                                    dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq))
                                                ? logits[slot][qr][kc] : 0.0f;
                                            const int aq = (int) lrintf(15.0f * af);
                                            av[j] = aq < 0 ? 0 : (aq > 15 ? 15 : aq);
                                        } else {
                                            av[j] = 0;
                                        }
                                        if (!use_v_i4_cache) {
                                            if (kc < tile_n) {
                                                const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, d_col);
                                                bv[j] = pdmq_clamp_i4_signed((int) lrintf(vv / v_scale_lane)) & 0x0f;
                                            } else {
                                                bv[j] = 0;
                                            }
                                        }
                                    }
                                    a_frag[g] = pdmq_pack_i4x8_unsigned(av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7]);
                                    b_frag[g] = use_v_i4_cache
                                        ? v_i4_cache_words[vci * size_t(PBWMMA_I4_WORDS_PER_K16) + size_t(g)]
                                        : pdmq_pack_i4x8_unsigned(bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], bv[6], bv[7]);
                                }
                                pbwmma_v8i32 pv_acc_i = {0,0,0,0,0,0,0,0};
                                pv_acc_i = pbwmma_mma_u4s4(a_frag, b_frag, pv_acc_i);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const int p_row = row_block + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                    if (p_row < PV_WMMA_ROWS) {
                                        const int slot = p_row / BM;
                                        const int qr   = p_row - slot * BM;
                                        const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                        const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                        if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                                dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                                            pv_acc_f[i] += float(pv_acc_i[i]) * (v_scale_lane / 15.0f);
                                        }
                                    }
                                }
                            }
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int p_row = row_block + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                if (p_row >= PV_WMMA_ROWS) {
                                    continue;
                                }
                                const int slot = p_row / BM;
                                const int qr   = p_row - slot * BM;
                                const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                        dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                                    const int out_idx = p_row * D + d_col;
                                    out_wmma[out_idx] = out_wmma[out_idx] * old_s[slot][qr] + pv_acc_f[i];
                                }
                            }
                        }
                    }  else if constexpr (PV_I8_WMMA && RAW_Q4_LDS) {
                        // INT-FA-style PV: P_i8 = round(127 * softmax(P)); V is
                        // quantized per K16/output-column tile to int8 and rescaled as
                        // O += (P_i8 @ V_i8) * V_scale / 127. Use complete 16-row PV
                        // tiles unless coherent mode explicitly asks for padded tails.
                        if (!PV_I8_FULL_TILES_ONLY || row_block + 16 <= PV_WMMA_ROWS) {
                            const int d_col = d_tile * 16 + pbwmma_i8_b_col_from_lane_lo(lane_lo);
                            float pv_acc_f[8] = {0,0,0,0,0,0,0,0};
#pragma unroll
                            for (int kc_base = 0; kc_base < BN; kc_base += 16) {
                                pbwmma_v4i32 a_frag;
                                pbwmma_v4i32 b_frag;
                                const int a_p_row = row_block + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                                float vmax_lane = 0.0f;
#pragma unroll
                                for (int kk = 0; kk < 16; ++kk) {
                                    const int kc = kc_base + kk;
                                    if (kc < tile_n) {
                                        const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, d_col);
                                        vmax_lane = fmaxf(vmax_lane, fabsf(vv));
                                    }
                                }
                                const float v_scale_lane = vmax_lane > 0.0f ? vmax_lane / 127.0f : 1.0f;
#pragma unroll
                                for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                                    int av[4];
                                    int bv[4];
#pragma unroll
                                    for (int j = 0; j < 4; ++j) {
                                        const int kk = 4 * g + j;
                                        const int kc = kc_base + kk;
                                        if (a_p_row < PV_WMMA_ROWS && kc < tile_n) {
                                            const int slot = a_p_row / BM;
                                            const int qr   = a_p_row - slot * BM;
                                            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                            const float af = (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq))
                                                ? logits[slot][qr][kc] : 0.0f;
                                            av[j] = pdmq_clamp_i8((int) lrintf(127.0f * af));
                                        } else {
                                            av[j] = 0;
                                        }
                                        if (kc < tile_n) {
                                            const float vv = pdmq_decode_v_q4_0_raw_lds<D>(v_raw_tile, kc, d_col);
                                            bv[j] = pdmq_clamp_i8((int) lrintf(vv / v_scale_lane));
                                        } else {
                                            bv[j] = 0;
                                        }
                                    }
                                    a_frag[g] = pdmq_pack_i8x4(av[0], av[1], av[2], av[3]);
                                    b_frag[g] = pdmq_pack_i8x4(bv[0], bv[1], bv[2], bv[3]);
                                }
                                pbwmma_v8i32 pv_acc_i = {0,0,0,0,0,0,0,0};
                                pv_acc_i = pbwmma_mma_i8(a_frag, b_frag, pv_acc_i);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    const int p_row = row_block + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                    if (p_row < PV_WMMA_ROWS) {
                                        const int slot = p_row / BM;
                                        const int qr   = p_row - slot * BM;
                                        const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                        const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                        if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                                            pv_acc_f[i] += float(pv_acc_i[i]) * (v_scale_lane / 127.0f);
                                        }
                                    }
                                }
                            }
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int p_row = row_block + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                if (p_row >= PV_WMMA_ROWS) {
                                    continue;
                                }
                                const int slot = p_row / BM;
                                const int qr   = p_row - slot * BM;
                                const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                                    const int out_idx = p_row * D + d_col;
                                    out_wmma[out_idx] = out_wmma[out_idx] * old_s[slot][qr] + pv_acc_f[i];
                                }
                            }
                        } else {
                            const int d_col = d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo);
                        pbwmma_v8fp32 pv_acc = {0,0,0,0,0,0,0,0};
#pragma unroll
                        for (int kc_base = 0; kc_base < BN; kc_base += 16) {
                            pbwmma_v16fp16 a_frag;
                            pbwmma_v16fp16 b_frag;
#pragma unroll
                            for (int kk = 0; kk < 16; ++kk) {
                                const int kc = kc_base + kk;
                                const int p_row = row_block + pbwmma_f16_a_row_from_lane_lo(lane_lo);
                                if (p_row < PV_WMMA_ROWS && kc < tile_n) {
                                    const int slot = p_row / BM;
                                    const int qr   = p_row - slot * BM;
                                    const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                    const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                    a_frag[kk] = (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq))
                                        ? __float2half(logits[slot][qr][kc]) : __float2half(0.0f);
                                } else {
                                    a_frag[kk] = __float2half(0.0f);
                                }
                                b_frag[kk] = kc < tile_n ? v_wmma_tile[kc * D + d_col] : __float2half(0.0f);
                            }
                            pv_acc = pbwmma_mma(a_frag, b_frag, pv_acc);
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int p_row = row_block + pbwmma_f16_d_row_from_acc(i, lane_hi);
                            if (p_row >= PV_WMMA_ROWS) {
                                continue;
                            }
                            const int slot = p_row / BM;
                            const int qr   = p_row - slot * BM;
                            const int hq   = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                            const int q    = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                            if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) && dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                                const int out_idx = p_row * D + d_col;
                                out_wmma[out_idx] = out_wmma[out_idx] * old_s[slot][qr] + pv_acc[i];
                            }
                        }
                        }
                    }
                }
            }
            }
        } else if (tid < D) {
            if constexpr (PV_BLOCK_EXACT) {
                bool live[GQA_GROUP][BM];
#pragma unroll
                for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                        const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                        live[slot][qr] = dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                            dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq);
                        out[slot][qr] = live[slot][qr] ? out[slot][qr] * old_s[slot][qr] : 0.0f;
                    }
                }
#pragma unroll
                for (int kk = 0; kk < BN; ++kk) {
                    const float vv = kk < tile_n
                        ? (v4_144_pv4_active
                            ? (v4_144_pv4_words_active ? pdmq_decode_v_v4_k16d16_144_pv4_cached<D>(v4_144_pv4_payload_words, v4_144_scale_tile, kk, tid) : pdmq_decode_v_v4_k16d16_144_scale_lds<D>(V, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, v4_144_scale_tile, kk))
                            : (RAW_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                : (RAW_Q8_LDS
                                    ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                    : (RAW_F16_LDS
                                        ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                        : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk)))))
                        : 0.0f;
#pragma unroll
                    for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                        for (int qr = 0; qr < BM; ++qr) {
                            if (!live[slot][qr]) {
                                continue;
                            }
                            const float p = logits[slot][qr][kk];
                            if (p != 0.0f) out[slot][qr] += p * vv;
                        }
                    }
                }
            } else {
                float acc[GQA_GROUP][BM];
#pragma unroll
                for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                        const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                        acc[slot][qr] = (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq))
                            ? out[slot][qr] * old_s[slot][qr] : 0.0f;
                    }
                }





                    if (qblock_program.pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR) {
                        bool live[GQA_GROUP][BM];
#pragma unroll
                        for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                            for (int qr = 0; qr < BM; ++qr) {
                                const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                live[slot][qr] = dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                    dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq);
                            }
                        }
#pragma unroll
                        for (int kk = 0; kk < BN; ++kk) {
                            const float vv = kk < tile_n
                                ? (v4_144_pv4_active
                                    ? (v4_144_pv4_words_active ? pdmq_decode_v_v4_k16d16_144_pv4_cached<D>(v4_144_pv4_payload_words, v4_144_scale_tile, kk, tid) : pdmq_decode_v_v4_k16d16_144_scale_lds<D>(V, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, v4_144_scale_tile, kk))
                                    : (RAW_Q4_LDS
                                        ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                        : (RAW_Q8_LDS
                                            ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                            : (RAW_F16_LDS
                                                ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                                : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk)))))
                                : 0.0f;
#pragma unroll
                            for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                                for (int qr = 0; qr < BM; ++qr) {
                                    if (!live[slot][qr]) {
                                        continue;
                                    }
                                    const float p = logits[slot][qr][kk];
                                    if (p != 0.0f) acc[slot][qr] += p * vv;
                                }
                            }
                        }
                    } else {
#pragma unroll
                        for (int kk = 0; kk < BN; ++kk) {
                            const float vv = kk < tile_n
                                ? (v4_144_pv4_active
                                    ? (v4_144_pv4_words_active ? pdmq_decode_v_v4_k16d16_144_pv4_cached<D>(v4_144_pv4_payload_words, v4_144_scale_tile, kk, tid) : pdmq_decode_v_v4_k16d16_144_scale_lds<D>(V, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, v4_144_scale_tile, kk))
                                    : (RAW_Q4_LDS
                                        ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                        : (RAW_Q8_LDS
                                            ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                            : (RAW_F16_LDS
                                                ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                                : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk)))))
                                : 0.0f;
#pragma unroll
                            for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                                for (int qr = 0; qr < BM; ++qr) {
                                    const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                                    const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                                    if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) ||
                                            !dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) continue;
                                    const float p = logits[slot][qr][kk];
                                    if (p != 0.0f) acc[slot][qr] += p * vv;
                                }
                            }
                        }
                    }

#pragma unroll
                for (int slot = 0; slot < GQA_GROUP; ++slot) {
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
                        const int q  = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                        if (dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) &&
                                dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                            out[slot][qr] = acc[slot][qr];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (PV_WMMA) {
        for (int idx = tid; idx < PV_WMMA_ROWS * D; idx += int(blockDim.x)) {
            const int p_row = idx / D;
            const int d     = idx - p_row * D;
            const int slot  = p_row / BM;
            const int qr    = p_row - slot * BM;
            const int hq    = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            const int q     = dp16_fa_qblock_program_q(qblock_program, q0, qr);
            if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio) ||
                    !dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) {
                continue;
            }
            const float denom = row_l[slot][qr];
            const size_t row = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq));
            if (split_mode) {
                const size_t partial_row = size_t(split_idx) * size_t(batch) * size_t(nq) * size_t(n_heads_q) + row;
                if (d == 0) {
                    partial_m[partial_row] = row_m[slot][qr];
                    partial_l[partial_row] = denom;
                }
                if (pdmq_logit_is_live(row_m[slot][qr])) {
                    partial_out[partial_row * size_t(D) + size_t(d)] = out_wmma[idx];
                }
            } else {
                dst[row * size_t(D) + size_t(d)] = denom > 0.0f ? out_wmma[idx] / denom : 0.0f;
            }
        }
    } else if (tid < D) {
#pragma unroll
        for (int slot = 0; slot < GQA_GROUP; ++slot) {
            const int hq = dp16_fa_qblock_program_hq(qblock_program, hq_base, slot);
            if (!dp16_fa_qblock_program_head_live(qblock_program, hq, slot, hk, n_heads_q, gqa_ratio)) continue;
#pragma unroll
            for (int qr = 0; qr < BM; ++qr) {
                const int q = dp16_fa_qblock_program_q(qblock_program, q0, qr);
                if (!dp16_fa_qblock_program_row_live(qblock_program, q, qr, nq)) continue;
                const float denom = row_l[slot][qr];
                const float final = denom > 0.0f ? out[slot][qr] / denom : 0.0f;
                const size_t row = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq));
                if (split_mode) {
                    const size_t partial_row = size_t(split_idx) * size_t(batch) * size_t(nq) * size_t(n_heads_q) + row;
                    if (tid == 0) {
                        partial_m[partial_row] = row_m[slot][qr];
                        partial_l[partial_row] = denom;
                    }
                    if (pdmq_logit_is_live(row_m[slot][qr])) {
                        partial_out[partial_row * size_t(D) + size_t(tid)] = out[slot][qr];
                    }
                } else {
                    dst[row * size_t(D) + size_t(tid)] = final;
                }
            }
        }
    }
}

static __global__ void packed16_dot4_mmq_gqax_splitk_merge_kernel(
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        const float * __restrict__ partial_out,
        float       * __restrict__ dst,
        int nq, int nk, int n_heads_q, int batch, int split_k) {
    const size_t total = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
    const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (idx >= total) {
        return;
    }

    const int d = int(idx % PDMQ_D);
    const size_t row = idx / size_t(PDMQ_D);
    const size_t rows_per_split = size_t(batch) * size_t(nq) * size_t(n_heads_q);
    const int hq = int(row % size_t(n_heads_q));
    const size_t bq = row / size_t(n_heads_q);
    const int q = int(bq % size_t(nq));
    const int b = int(bq / size_t(nq));

    float m = pdmq_neg();
    for (int s = 0; s < split_k; ++s) {
        const float ms = partial_m[size_t(s) * rows_per_split + row];
        m = fmaxf(m, ms);
    }

    float l = 0.0f;
    float out = 0.0f;
    if (pdmq_logit_is_live(m)) {
        for (int s = 0; s < split_k; ++s) {
            const size_t prow = size_t(s) * rows_per_split + row;
            const float ms = partial_m[prow];
            if (!pdmq_logit_is_live(ms)) {
                continue;
            }
            const float scale = expf(ms - m);
            l += partial_l[prow] * scale;
            out += partial_out[prow * size_t(PDMQ_D) + size_t(d)] * scale;
        }
    }
    const float final = l > 0.0f ? out / l : 0.0f;
    dst[idx] = final;
}

template<packed16_dot4_mmq_v_type V_TYPE, int BM, int BN, int D, bool CAUSAL_MASK, bool STAGE_V, bool RAW_LDS_Q4, bool KSHARED, bool PV_BLOCK_EXACT = false, bool V4_K16D16_ORACLE = false>
static __global__ __launch_bounds__(PDMQ_THREADS, 1) void packed16_dot4_mmq_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ V,
        const half  * __restrict__ V4_tail,
        float       * __restrict__ dst,
        int64_t q_nb01,
        int64_t q_nb02,
        int64_t q_nb03,
        int64_t v_nb10,
        int64_t v_nb11,
        int64_t v_nb12,
        int64_t v_nb13,
        int64_t v_ne13,
        int64_t v4_tail_nb10,
        int64_t v4_tail_nb11,
        int64_t v4_tail_nb12,
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
        int pvblock_mode) {

    static_assert(D == PDMQ_D, "packed16_dot4_mmq is D256-only");

    const int tid = int(threadIdx.x);
    const int q_tile = int(blockIdx.x);
    const int hq = int(blockIdx.y);
    const int b  = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    const int q0 = q_tile * BM;
    const int v4_live_nk = CAUSAL_MASK ? pdmq_min_i(nk, q_offset + nq) : nk;

    if (hq >= n_heads_q || hk >= n_heads_k || q0 >= nq) {
        return;
    }

    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);

    __shared__ int   q_tile_i32[BM][D / 4 + 1];
    __shared__ float q_tile_scales[BM][D / QK8_0 + 1];
    __shared__ float logits[BM][BN + 1];
    static constexpr bool RAW_Q4_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q4_0;
    static constexpr bool RAW_Q8_LDS  = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_Q8_0;
    static constexpr bool RAW_F16_LDS = RAW_LDS_Q4 && V_TYPE == PACKED16_DOT4_MMQ_V_F16;
    static constexpr bool V4_144_Q4_LDS = !STAGE_V && V_TYPE == PACKED16_DOT4_MMQ_V4_K16D16_144;
    static_assert(!V4_K16D16_ORACLE || RAW_Q4_LDS, "V4_K16D16 oracle requires raw q4 LDS input");

    __shared__ float row_m[BM + 1];
    __shared__ float row_l[BM + 1];
    __shared__ float old_s[BM + 1];
    __shared__ float v_tile[STAGE_V ? BN * (D + 1) : 1];
    __shared__ block_q4_0 v_raw_tile[(RAW_Q4_LDS || V4_144_Q4_LDS) ? BN * (D / QK4_0) : 1];
    __shared__ int   v4_k16d16_words[V4_K16D16_ORACLE ? D * ((BN + 7) / 8) : 1];
    __shared__ block_q8_0 v_raw_q8_tile[RAW_Q8_LDS ? BN * (D / QK8_0) : 1];
    __shared__ half  v_raw_f16_tile[RAW_F16_LDS ? BN * D : 1];
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
                    ? pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk)
                    : 0.0f;
            }
            __syncthreads();
        } else if constexpr (RAW_Q4_LDS || V4_144_Q4_LDS) {
            constexpr int V_BLOCKS = D / QK4_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                if constexpr (RAW_Q4_LDS) {
                    v_raw_tile[kk * V_BLOCKS + blk] = *pdmq_v_q4_0_block_ptr(
                        V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
                } else {
                    v_raw_tile[kk * V_BLOCKS + blk] = pdmq_load_v4_k16d16_144_as_q4_0_block(
                        V, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
                }
            }
            __syncthreads();
            if constexpr (V4_K16D16_ORACLE) {
                pdmq_pack_v_q4_0_k16d16_oracle_tile<D, BN>(v_raw_tile, v4_k16d16_words, tile_n);
            }
        } else if constexpr (RAW_Q8_LDS) {
            constexpr int V_BLOCKS = D / QK8_0;
            for (int v_idx = tid; v_idx < tile_n * V_BLOCKS; v_idx += int(blockDim.x)) {
                const int kk = v_idx / V_BLOCKS;
                const int blk = v_idx - kk * V_BLOCKS;
                v_raw_q8_tile[kk * V_BLOCKS + blk] = *pdmq_v_q8_0_block_ptr(
                    V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, blk);
            }
            __syncthreads();
        } else if constexpr (RAW_F16_LDS) {
            constexpr int HALF_PER_VEC = int(sizeof(int4) / sizeof(half));
            constexpr int VECS_PER_ROW = D / HALF_PER_VEC;
            static_assert(D % HALF_PER_VEC == 0, "raw f16 LDS vector copy requires int4-aligned rows");
            if (v_nb10 == int64_t(sizeof(half))) {
                for (int v_idx = tid; v_idx < tile_n * VECS_PER_ROW; v_idx += int(blockDim.x)) {
                    const int kk  = v_idx / VECS_PER_ROW;
                    const int vec = v_idx - kk * VECS_PER_ROW;
                    const int d   = vec * HALF_PER_VEC;
                    const int k   = k0 + kk;
                    const half * src = pdmq_v_f16_ptr(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
                    ((int4 *) (v_raw_f16_tile + kk * D))[vec] = *((const int4 *) src);
                }
            } else {
                for (int v_idx = tid; v_idx < tile_n * D; v_idx += int(blockDim.x)) {
                    const int kk = v_idx / D;
                    const int d  = v_idx - kk * D;
                    v_raw_f16_tile[kk * D + d] = *pdmq_v_f16_ptr(
                        V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, d);
                }
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
                k_payload_s[kk * (KP_PER_ROW + 1) + w] = k_payload[k_row * KP_PER_ROW + w];
            }
            for (int idx = tid; idx < tile_n * KS_PER_ROW; idx += int(blockDim.x)) {
                const int kk = idx / KS_PER_ROW, s = idx - kk * KS_PER_ROW;
                const size_t k_row = k_head_base + size_t(k0 + kk);
                k_scales_s[kk * (KS_PER_ROW + 1) + s] = k_scales[k_row * KS_PER_ROW + s];
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
                float mask_bias = 0.0f;
                if (!exact_tile && valid && mask) {
                    valid = pdmq_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, q, k, b, &mask_bias);
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
                    s = pdmq_qk_dot<BM, D>(q_tile_i32, q_tile_scales, qr, k_row_payload, k_row_scales) * attention_scale + mask_bias;
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
            if constexpr (!STAGE_V) {
                if constexpr (PV_BLOCK_EXACT) {
                    unsigned live_mask = 0u;
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        if (q < nq) {
                            live_mask |= (1u << qr);
                            out[qr] *= old_s[qr];
                        } else {
                            out[qr] = 0.0f;
                        }
                    }
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const float vv = kk < tile_n
                            ? (RAW_Q4_LDS || V4_144_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                : (RAW_Q8_LDS
                                    ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                    : (RAW_F16_LDS
                                        ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                        : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk))))
                            : 0.0f;
#pragma unroll
                        for (int qr = 0; qr < BM; ++qr) {
                            if ((live_mask & (1u << qr)) == 0u) {
                                continue;
                            }
                            const float p = logits[qr][kk];
                            if (p != 0.0f) {
                                out[qr] += p * vv;
                            }
                        }
                    }
                } else if (pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR) {
                    float acc[BM];
                    unsigned live_mask = 0u;
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        if (q < nq) {
                            live_mask |= (1u << qr);
                            acc[qr] = out[qr] * old_s[qr];
                        } else {
                            acc[qr] = 0.0f;
                        }
                    }
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const float vv = kk < tile_n
                            ? (RAW_Q4_LDS || V4_144_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                : (RAW_Q8_LDS
                                    ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                    : (RAW_F16_LDS
                                        ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                        : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk))))
                            : 0.0f;
#pragma unroll
                        for (int qr = 0; qr < BM; ++qr) {
                            if ((live_mask & (1u << qr)) == 0u) {
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
                        if ((live_mask & (1u << qr)) != 0u) {
                            out[qr] = acc[qr];
                        }
                    }
                } else {
                    float acc[BM];
#pragma unroll
                    for (int qr = 0; qr < BM; ++qr) {
                        const int q = q0 + qr;
                        acc[qr] = q < nq ? out[qr] * old_s[qr] : 0.0f;
                    }
#pragma unroll
                    for (int kk = 0; kk < BN; ++kk) {
                        const float vv = kk < tile_n
                            ? (RAW_Q4_LDS || V4_144_Q4_LDS
                                ? pdmq_decode_v_q4_0_raw_or_k16d16<D, BN, V4_K16D16_ORACLE>(v_raw_tile, v4_k16d16_words, kk, tid)
                                : (RAW_Q8_LDS
                                    ? pdmq_decode_v_q8_0_raw_lds<D>(v_raw_q8_tile, kk, tid)
                                    : (RAW_F16_LDS
                                        ? pdmq_decode_v_f16_raw_lds<D>(v_raw_f16_tile, kk, tid)
                                        : pdmq_decode_v<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k0 + kk, hk, b, tid, V4_tail, v4_tail_nb10, v4_tail_nb11, v4_tail_nb12, v4_live_nk))))
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
                            acc += p * v_tile[kk * (D + 1) + tid];
                        }
                    }
                    out[qr] = acc;
                }
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
            const size_t row = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq));
            dst[row * size_t(D) + size_t(tid)] = val;
        }
    }
}

//
// ── GQA group selector ────────────────────────────────────────────
static inline int ggml_cuda_rocm_packed16_dot4_mmq_gqa_group() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP");
    if (!s || !*s) return 1;
    const int g = atoi(s);
    if (g == 1 || g == 2 || g == 4 || g == 6) return g;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP=%s; expected 1, 2, 4, or 6", s);
}

static inline int ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK");
    if (!s || !*s) return 1;
    const int split_k = atoi(s);
    if (split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16 || split_k == 32 || split_k == 64) return split_k;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK=%s; expected 1, 2, 4, 8, 16, 32, or 64", s);
}

static inline bool ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk_roof_cap() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_ROOF_CAP");
    return s && atoi(s) != 0;
}

static inline bool ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk_compact_empty() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_COMPACT_EMPTY");
    return s && atoi(s) != 0;
}

static inline int pdmq_splitk_roof_pow2(int nk) {
    // Keep each split shard at or above the DOT4-tile roof observed for the
    // 9B GQA2X route.  For nk=256 this caps split-K at 4; split-K=8 creates
    // 32-token shards and changed the speculative trajectory even though final
    // SHA and serial-oracle tokens stayed stable.
    constexpr int min_tokens_per_split = 64;
    if (nk >= 8 * min_tokens_per_split) return 8;
    if (nk >= 4 * min_tokens_per_split) return 4;
    if (nk >= 2 * min_tokens_per_split) return 2;
    return 1;
}

enum pdmq_shape {
    PDMQ_SHAPE_M1N32,
    PDMQ_SHAPE_M2N32,
    PDMQ_SHAPE_M4N32,
    PDMQ_SHAPE_M16N16,
    PDMQ_SHAPE_M8N32,
};

enum pdmq_v_path {
    PDMQ_V_STAGE_F32,
    PDMQ_V_RAW_LDS_Q4,
    PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE,
    PDMQ_V_V4_K16D16_144,
    PDMQ_V_RAW_LDS_Q8_0,
    PDMQ_V_RAW_LDS_F16,
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

    ggml_type k_type;
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
        case PDMQ_SHAPE_M16N16: return "16x16";
        case PDMQ_SHAPE_M8N32:  return "8x32";
        default:                return "unknown";
    }
}

static inline const char * pdmq_v_path_name(const pdmq_v_path v_path) {
    switch (v_path) {
        case PDMQ_V_STAGE_F32:     return "stage_f32";
        case PDMQ_V_RAW_LDS_Q4:                return "raw_lds_q4";
        case PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE:  return "raw_lds_q4_k16d16_oracle";
        case PDMQ_V_V4_K16D16_144:             return "v4_k16d16_144";
        case PDMQ_V_RAW_LDS_Q8_0:              return "raw_lds_q8_0";
        case PDMQ_V_RAW_LDS_F16:   return "raw_lds_f16";
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
    if (strcmp(env, "16x16") == 0 || strcmp(env, "m16n16") == 0) {
        return PDMQ_SHAPE_M16N16;
    }
    if (strcmp(env, "8x32") == 0 || strcmp(env, "m8n32") == 0) {
        return PDMQ_SHAPE_M8N32;
    }
    GGML_ABORT("packed16_dot4_mmq: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE=%s, expected 1x32, 2x32, 4x32, 8x32, or 16x16", env);
}

static inline pdmq_v_path pdmq_parse_v_path_env(const char * env) {
    if (strcmp(env, "stage_f32") == 0 || strcmp(env, "stage") == 0 || strcmp(env, "f32") == 0) {
        return PDMQ_V_STAGE_F32;
    }
    if (strcmp(env, "raw_lds_q4") == 0 || strcmp(env, "raw_q4") == 0) {
        return PDMQ_V_RAW_LDS_Q4;
    }
    if (strcmp(env, "raw_lds_q4_k16d16_oracle") == 0 || strcmp(env, "v4_k16d16_oracle") == 0) {
        return PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE;
    }
    if (strcmp(env, "v4_k16d16_144") == 0 || strcmp(env, "persistent_v4_k16d16_144") == 0 || strcmp(env, "q4_exact_k16d16") == 0) {
        return PDMQ_V_V4_K16D16_144;
    }
    if (strcmp(env, "raw_lds_q8_0") == 0 || strcmp(env, "raw_q8_0") == 0 || strcmp(env, "raw_q8") == 0) {
        return PDMQ_V_RAW_LDS_Q8_0;
    }
    if (strcmp(env, "raw_lds_f16") == 0 || strcmp(env, "raw_f16") == 0) {
        return PDMQ_V_RAW_LDS_F16;
    }
    GGML_ABORT("packed16_dot4_mmq: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH=%s, expected stage_f32, raw_lds_q4, raw_lds_q4_k16d16_oracle, v4_k16d16_144, raw_lds_q8_0, or raw_lds_f16", env);
}

static inline pdmq_shape pdmq_select_qblock_shape_auto(const int nq) {
    // Standard QBlock verifier policy: use the validated M8N32 small-Q shape
    // for verifier spans up to the current QBlock max.  M4N32 remains an
    // explicit shape experiment; it is not part of the Qwen35 standard build
    // matrix and must not be selected silently by the default route.
    if (nq <= 8) {
        return PDMQ_SHAPE_M8N32;
    }
    return PDMQ_SHAPE_M16N16;
}

static inline bool pdmq_qblock_tetris_shape_policy_requested() {
    const char * env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE_POLICY");
    if (!env || !*env) {
        return false;
    }
    return strcmp(env, "tetris") == 0 || strcmp(env, "compact_tetris") == 0 || strcmp(env, "fit") == 0 || atoi(env) != 0;
}

static inline pdmq_shape pdmq_select_qwen35_gqa6_qblock_shape(const int nq, const bool tetris) {
    if (nq <= 1) {
        return PDMQ_SHAPE_M1N32;
    }
    if (nq <= 2 || !tetris) {
        return PDMQ_SHAPE_M2N32;
    }
    return PDMQ_SHAPE_M4N32;
}

static inline pdmq_shape pdmq_select_shape_auto(
        const int nq,
        const int nk,
        const int n_heads_q,
        const int n_heads_k,
        const ggml_type v_type) {
    // Role-specialized MTP shapes. Decode and tiny verify should not share the
    // prefill/small-prefill policy: they have one to four Q rows and benefit
    // from wider K tiles with less inactive-M overhead.
    GGML_UNUSED(n_heads_q);
    GGML_UNUSED(n_heads_k);
    if (nq <= 1) {
        return PDMQ_SHAPE_M1N32;
    }
    if (nq <= 4) {
        if (nq == 2) {
            return PDMQ_SHAPE_M2N32;
        }
        // M4N32 remains available via SHAPE env, but the runtime canary showed
        // higher verify-nq4 time than the proven 8x32 policy.
        return PDMQ_SHAPE_M8N32;
    }

    // Attention equivalent of the MMQ max-x lesson: do not maximize tile width
    // on gfx1100 just because it exists.
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

static inline pdmq_v_path pdmq_default_v_path(const int nq, const ggml_type v_type) {
    // Raw compressed-q4 LDS wins over f32 V staging for both small-Q verify/decode
    // and long-context prefill. Keeping q4 compressed in LDS avoids expanding the
    // full V tile to float before PV; q8_0/f16 keep their raw LDS paths limited to
    // small-Q roles until their large-Q staging is separately proven.
    if (v_type == GGML_TYPE_Q4_0) {
        return PDMQ_V_RAW_LDS_Q4;
    }
    if (v_type == GGML_TYPE_Q8_0 && nq <= 8) {
        return PDMQ_V_RAW_LDS_Q8_0;
    }
    if (v_type == GGML_TYPE_F16 && nq <= 8) {
        return PDMQ_V_RAW_LDS_F16;
    }
    if (v_type == GGML_TYPE_V4_K16D16_144) {
        return PDMQ_V_V4_K16D16_144;
    }
    return PDMQ_V_STAGE_F32;
}

static inline pdmq_v_path pdmq_select_v_path(
        const int nq,
        const int nk,
        const ggml_type v_type) {
    GGML_UNUSED(nk);

    const pdmq_v_path default_path = pdmq_default_v_path(nq, v_type);

    if (const char * env = pdmq_vpath_env(); env && *env) {
        const pdmq_v_path requested = pdmq_parse_v_path_env(env);
        if ((requested == PDMQ_V_RAW_LDS_Q4 || requested == PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE) && v_type != GGML_TYPE_Q4_0) {
            GGML_ABORT("packed16_dot4_mmq: %s requires V=q4_0, got V=%s", pdmq_v_path_name(requested), ggml_type_name(v_type));
        }
        if (requested == PDMQ_V_V4_K16D16_144 && v_type != GGML_TYPE_V4_K16D16_144) {
            GGML_ABORT("packed16_dot4_mmq: v4_k16d16_144 requires V=v4_k16d16_144, got V=%s", ggml_type_name(v_type));
        }
        if (requested == PDMQ_V_RAW_LDS_Q8_0 && v_type != GGML_TYPE_Q8_0) {
            GGML_ABORT("packed16_dot4_mmq: raw_lds_q8_0 requires V=q8_0, got V=%s", ggml_type_name(v_type));
        }
        if (requested == PDMQ_V_RAW_LDS_F16 && v_type != GGML_TYPE_F16) {
            GGML_ABORT("packed16_dot4_mmq: raw_lds_f16 requires V=f16, got V=%s", ggml_type_name(v_type));
        }
        if (requested != default_path && requested != PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE && requested != PDMQ_V_V4_K16D16_144) {
            GGML_ABORT("packed16_dot4_mmq: VPATH=%s for V=%s nq=%d is not part of the compact PDMQ route set; default vpath=%s",
                       env, ggml_type_name(v_type), nq, pdmq_v_path_name(default_path));
        }
        return requested;
    }

    return default_path;
}

static inline pdmq_plan pdmq_make_plan(
        const int nq,
        const int nk,
        const int n_heads_q,
        const int n_heads_k,
        const int d,
        const ggml_type k_type,
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
    plan.kshared = false;
    plan.split_k = false;
    plan.k_type = k_type;
    plan.v_type = v_type;
    plan.nq = nq;
    plan.nk = nk;
    plan.hq = n_heads_q;
    plan.hk = n_heads_k;
    plan.d = d;
    return plan;
}

struct pdmq_v_i4_cache_state {
    const void * v_data = nullptr;
    int64_t v_nb10 = 0, v_nb11 = 0, v_nb12 = 0, v_nb13 = 0, v_ne13 = 0;
    int n_heads_k = 0;
    int batch = 0;
    int capacity_blocks = 0;
    int packed_full_blocks = 0;
    int last_nk = 0;
    int * words = nullptr;
    float * scales = nullptr;

    void release() {
        if (words)  CUDA_CHECK(hipFree(words));
        if (scales) CUDA_CHECK(hipFree(scales));
        *this = {};
    }
};

static inline bool pdmq_v_i4_cache_state_matches(
        const pdmq_v_i4_cache_state & c,
        const ggml_tensor * V,
        const int n_heads_k,
        const int batch) {
    return c.v_data == V->data && c.v_nb10 == V->nb[0] && c.v_nb11 == V->nb[1] &&
        c.v_nb12 == V->nb[2] && c.v_nb13 == V->nb[3] && c.v_ne13 == (V->ne[3] > 0 ? V->ne[3] : 1) &&
        c.n_heads_k == n_heads_k && c.batch == batch;
}

static inline pdmq_v_i4_cache_state & pdmq_v_i4_cache_for(
        const ggml_tensor * V,
        const int n_heads_k,
        const int batch) {
    static pdmq_v_i4_cache_state caches[32];
    static int victim = 0;
    int free_idx = -1;
    for (int i = 0; i < 32; ++i) {
        if (caches[i].v_data && pdmq_v_i4_cache_state_matches(caches[i], V, n_heads_k, batch)) {
            return caches[i];
        }
        if (!caches[i].v_data && free_idx < 0) {
            free_idx = i;
        }
    }
    const int idx = free_idx >= 0 ? free_idx : victim;
    victim = (idx + 1) & 31;
    caches[idx].release();
    return caches[idx];
}

static inline void pdmq_prepare_v_i4_cache(
        const ggml_tensor * V,
        const int nk,
        const int n_heads_k,
        const int batch,
        cudaStream_t stream,
        int ** words_out,
        float ** scales_out,
        int * k_blocks_out) {
    *words_out = nullptr;
    *scales_out = nullptr;
    *k_blocks_out = 0;
    const int total_blocks = CEIL_DIV(nk, 16);
    if (total_blocks <= 0 || V->type != GGML_TYPE_Q4_0 || V->ne[0] != PDMQ_D) {
        return;
    }
    pdmq_v_i4_cache_state & c = pdmq_v_i4_cache_for(V, n_heads_k, batch);
    const bool reset = !pdmq_v_i4_cache_state_matches(c, V, n_heads_k, batch) || nk < c.last_nk || c.capacity_blocks < total_blocks;
    if (reset) {
        c.release();
        c.v_data = V->data;
        c.v_nb10 = V->nb[0];
        c.v_nb11 = V->nb[1];
        c.v_nb12 = V->nb[2];
        c.v_nb13 = V->nb[3];
        c.v_ne13 = V->ne[3] > 0 ? V->ne[3] : 1;
        c.n_heads_k = n_heads_k;
        c.batch = batch;
        c.capacity_blocks = total_blocks;
        const size_t entries = size_t(c.v_ne13) * size_t(n_heads_k) * size_t(c.capacity_blocks) * size_t(PDMQ_D);
        CUDA_CHECK(hipMalloc((void **) &c.words, entries * size_t(PBWMMA_I4_WORDS_PER_K16) * sizeof(int)));
        CUDA_CHECK(hipMalloc((void **) &c.scales, entries * sizeof(float)));
    }

    const int refresh_blocks = pdmq_pv_i4_cache_refresh_blocks();
    const int tail_begin = total_blocks > refresh_blocks ? total_blocks - refresh_blocks : 0;
    if (c.packed_full_blocks > tail_begin) {
        c.packed_full_blocks = tail_begin;
    }
    if (c.packed_full_blocks < tail_begin) {
        const int begin = c.packed_full_blocks;
        const int end = tail_begin;
        dim3 grid(end - begin, n_heads_k, batch);
        dim3 block(PDMQ_D, 1, 1);
        pdmq_pack_v_i4_cache_kernel<<<grid, block, 0, stream>>>(
            (const char *) V->data, V->nb[0], V->nb[1], V->nb[2], V->nb[3], V->ne[3] > 0 ? V->ne[3] : 1,
            nk, n_heads_k, batch, begin, end, c.capacity_blocks, c.words, c.scales);
        c.packed_full_blocks = tail_begin;
    }
    if (tail_begin < total_blocks) {
        dim3 grid(total_blocks - tail_begin, n_heads_k, batch);
        dim3 block(PDMQ_D, 1, 1);
        pdmq_pack_v_i4_cache_kernel<<<grid, block, 0, stream>>>(
            (const char *) V->data, V->nb[0], V->nb[1], V->nb[2], V->nb[3], V->ne[3] > 0 ? V->ne[3] : 1,
            nk, n_heads_k, batch, tail_begin, total_blocks, c.capacity_blocks, c.words, c.scales);
    }
    c.last_nk = nk;
    *words_out = c.words;
    *scales_out = c.scales;
    *k_blocks_out = c.capacity_blocks;
}

void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool k_persistent_i32 = K->type == GGML_TYPE_I32 && K->ne[0] * 4 == PDMQ_D;
    const bool k_f16_hotcold_sidecar = K->type == GGML_TYPE_F16 && K->ne[0] == PDMQ_D;
    GGML_ASSERT(Q->type == GGML_TYPE_F32 && (k_persistent_i32 || k_f16_hotcold_sidecar) && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == PDMQ_D && V->ne[0] == PDMQ_D && dst->ne[0] == PDMQ_D);
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

    ggml_tensor * v4_cache = nullptr;
    ggml_tensor * v4_tail  = nullptr;


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

    const int32_t fa_inst_i32 = ((const int32_t *) dst->op_params)[4];
    const bool qblock_inst = fa_inst_i32 == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK;
    const bool shape_auto = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE_AUTO");
        return !v || atoi(v) != 0;
    }();

    pdmq_shape shape = shape_auto
        ? (qblock_inst ? pdmq_select_qblock_shape_auto(nq) : pdmq_select_shape_auto(nq, nk, n_heads_q, n_heads_k, V->type))
        : (nq >= 16 ? PDMQ_SHAPE_M16N16 : PDMQ_SHAPE_M8N32);

    const char * qblock_shape_env = qblock_inst ? getenv("GGML_CUDA_ROCM_MTP_QBLOCK_SHAPE") : nullptr;
    const char * shape_env = qblock_shape_env && *qblock_shape_env ? qblock_shape_env : getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE");
    const char * pv_rows_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PV_ROWS");
    const bool shape_env_set = shape_env && *shape_env;
    const bool pv_rows_env_set = pv_rows_env && *pv_rows_env;
    const char * qblock_gqa_group_env = qblock_inst ? getenv("GGML_CUDA_ROCM_MTP_QBLOCK_GQA_GROUP") : nullptr;
    const bool qblock_gqa_group_env_set = qblock_gqa_group_env && *qblock_gqa_group_env;
    const bool requested_gqa_group_env_set = qblock_gqa_group_env_set || []() {
        const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP");
        return s && *s;
    }();
    int requested_gqa_group_raw = ggml_cuda_rocm_packed16_dot4_mmq_gqa_group();
    if (qblock_gqa_group_env_set) {
        const int qblock_gqa_group = atoi(qblock_gqa_group_env);
        if (qblock_gqa_group == 1 || qblock_gqa_group == 2 || qblock_gqa_group == 4 || qblock_gqa_group == 6) {
            requested_gqa_group_raw = qblock_gqa_group;
        } else {
            GGML_ABORT("invalid GGML_CUDA_ROCM_MTP_QBLOCK_GQA_GROUP=%s; expected 1, 2, 4, or 6", qblock_gqa_group_env);
        }
    }
    const bool pdmq_pv_wmma_policy_requested = pdmq_pv_wmma_requested();
    const bool qblock_tetris_shape_policy_requested = qblock_inst && pdmq_qblock_tetris_shape_policy_requested();
    const bool qwen27b_gqa6_topology =
        n_heads_q == 24 && n_heads_k == 4 && (n_heads_q / n_heads_k) == 6;
    const bool qwen27b_gqa6_q4_smallq =
        V->type == GGML_TYPE_Q4_0 && nq <= 4 && qwen27b_gqa6_topology;
    const int qwen27b_gqa6_splitk_min_nk = []() {
        const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_MIN_NK");
        return s && *s ? atoi(s) : 12288;
    }();
    const bool qwen27b_gqa6_policy_requested = qwen27b_gqa6_q4_smallq && nk >= qwen27b_gqa6_splitk_min_nk &&
        (pdmq_pv_wmma_policy_requested || requested_gqa_group_raw == 6);
    if (shape_env_set) {
        shape = pdmq_parse_shape_env(shape_env);
    } else if (pv_rows_env_set && nq <= 4) {
        const int pv_rows = atoi(pv_rows_env);
        if (pv_rows != 1 && pv_rows != 2 && pv_rows != 4) {
            GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PV_ROWS=%s; expected 1, 2, or 4", pv_rows_env);
        }
        // For 27B GQA6, PV_ROWS is a max/verify tile hint, not a reason to
        // run token decode as mostly-zero M4. Match active M rows: 6/12/24.
        if (qwen27b_gqa6_q4_smallq) {
            shape = nq <= 1 ? PDMQ_SHAPE_M1N32 : (nq <= 2 ? PDMQ_SHAPE_M2N32 : PDMQ_SHAPE_M4N32);
        } else if (pv_rows == 1) {
            shape = PDMQ_SHAPE_M1N32;
        } else if (pv_rows == 2) {
            shape = PDMQ_SHAPE_M2N32;
        } else {
            shape = PDMQ_SHAPE_M4N32;
        }
    } else if (qwen27b_gqa6_policy_requested) {
        // Historical/debug GQA6 split-K policy. M4/M8 no-split grouped routes
        // later passed exactness but failed speed gates, so GQA6 remains an
        // opt-in diagnostic path rather than the QBlock/DOT4 promotion route.
        shape = pdmq_select_qwen35_gqa6_qblock_shape(nq, qblock_tetris_shape_policy_requested);
    }

    const bool v4_144_pv4_requested = V->type == GGML_TYPE_V4_K16D16_144 && pdmq_v4_144_pv4_requested();
    const bool v4_144_draft_v_cache_requested = []() {
        const char * v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_MTP_DRAFT_V_CACHE");
        return v && atoi(v) != 0;
    }();
    const bool qwen27b_gqa6_row_serial_pdmq =
        qwen27b_gqa6_topology &&
        (V->type == GGML_TYPE_Q4_0 || v4_144_pv4_requested) &&
        !shape_env_set && !pv_rows_env_set && !qwen27b_gqa6_policy_requested &&
        (ggml_cuda_packed16_dot4_mmq_route_required() || v4_144_pv4_requested);
    if (qwen27b_gqa6_row_serial_pdmq) {
        // Normal q4 uses PWMMA/DBV for these rows. When q4 is explicitly forced
        // through PDMQ, or V4_144 PV4 must use PDMQ, avoid auto M8/M16 because
        // it changes long-context MTP output by n46/n128. Draft-V cache has more
        // all-V4 rows, and clean n128 gates show M2N32 preserves the q4 hash while
        // reducing V4 draft overhead; target-only and forced q4 stay on M1N32.
        shape = v4_144_pv4_requested && v4_144_draft_v_cache_requested ? PDMQ_SHAPE_M2N32 : PDMQ_SHAPE_M1N32;
    }

    // m8n32 is a useful shallow/promptfill GQA2 shape, but deep pp2048
    // becomes K/V reread dominated.  At long K, switch grouped prefill back to
    // BM16 so each K/V tile is reused across twice as many live Q rows.
    if (requested_gqa_group_raw == 2 && V->type == GGML_TYPE_Q4_0 && nq >= 16 && nk >= 8192) {
        shape = PDMQ_SHAPE_M16N16;
    }

    // A prefill-oriented shape override such as m8n32 must not poison true
    // token-generation decode. For Qwen3.6-27B GQA6, keep small-Q decode on
    // the proven row-safe M1N32 shape unless the caller explicitly overrides.
    if (qwen27b_gqa6_q4_smallq && !qwen27b_gqa6_policy_requested && !shape_env_set && !pv_rows_env_set && shape != PDMQ_SHAPE_M1N32 && nq <= 1) {
        shape = PDMQ_SHAPE_M1N32;
    }

    int requested_gqa_group = requested_gqa_group_raw;
    // Qwen3.6-27B q4 small-Q has hq=24,hk=4,GQA=6. When PVMMA is requested,
    // use the grouped GQA6 dataflow by default; otherwise keep scalar/control
    // runs on the proven GQA1 path unless explicitly requested.
    if (qwen27b_gqa6_policy_requested && (pdmq_pv_wmma_policy_requested || requested_gqa_group == 6)) {
        requested_gqa_group = 6;
    }

    pdmq_plan plan = pdmq_make_plan(
        nq, nk, n_heads_q, n_heads_k, PDMQ_D, K->type, V->type, assume_causal, shape,
        requested_gqa_group);
    shape = plan.shape;

    const int launch_bm = shape == PDMQ_SHAPE_M1N32  ? PDMQ_BM_DECODE32  :
                          shape == PDMQ_SHAPE_M2N32  ? PDMQ_BM_VERIFY2_32 :
                          shape == PDMQ_SHAPE_M4N32  ? PDMQ_BM_VERIFY4_32 :
                          shape == PDMQ_SHAPE_M16N16 ? PDMQ_BM_PREFILL    : PDMQ_BM_SMALL;
    const int launch_bn = shape == PDMQ_SHAPE_M1N32  ? PDMQ_BN_DECODE32  :
                          shape == PDMQ_SHAPE_M2N32  ? PDMQ_BN_VERIFY2_32 :
                          shape == PDMQ_SHAPE_M4N32  ? PDMQ_BN_VERIFY4_32 :
                          shape == PDMQ_SHAPE_M16N16 ? PDMQ_BN_PREFILL    : PDMQ_BN_SMALL;

    const bool request_gqa2 = (plan.requested_gqa_group == 2);
    const bool request_gqa4 = (plan.requested_gqa_group == 4);
    const bool request_gqa6 = (plan.requested_gqa_group == 6);
    const bool request_gqa1 = !request_gqa2 && !request_gqa4 && !request_gqa6;
    const bool gqax_splitk_env_set = []() {
        const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK");
        return s && *s;
    }();
    int gqax_splitk_requested = ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk();
    const char * qblock_splitk_env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK");
    const bool qblock_splitk_env_set = qblock_inst && qblock_splitk_env && *qblock_splitk_env;
    if (qblock_splitk_env_set) {
        const int qblock_split_k = atoi(qblock_splitk_env);
        if (qblock_split_k == 1 || qblock_split_k == 2 || qblock_split_k == 4 || qblock_split_k == 8 ||
                qblock_split_k == 16 || qblock_split_k == 32 || qblock_split_k == 64) {
            gqax_splitk_requested = qblock_split_k;
        } else {
            GGML_ABORT("invalid GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK=%s; expected 1, 2, 4, 8, 16, 32, or 64", qblock_splitk_env);
        }
    }
    const int gqax_splitk_min_nk = [qblock_inst]() {
        const char * s = qblock_inst ? getenv("GGML_CUDA_ROCM_MTP_QBLOCK_SPLITK_MIN_NK") : nullptr;
        if (!s || !*s) {
            s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_MIN_NK");
        }
        return s && *s ? atoi(s) : 12288;
    }();
    if (gqax_splitk_requested > 1 && nk < gqax_splitk_min_nk) {
        gqax_splitk_requested = 1;
    }
    const bool gqa2_xqa_supported = request_gqa2 &&
        shape == PDMQ_SHAPE_M8N32 && V->type == GGML_TYPE_Q4_0 &&
        n_heads_q == 16 && n_heads_k == 4 && gqa_ratio == 4;
    const bool gqa2_legacy_supported = request_gqa2 &&
        (shape == PDMQ_SHAPE_M16N16 || shape == PDMQ_SHAPE_M8N32) &&
        (gqa_ratio >= 2) && (nq > 1);
    const bool gqa2_splitk_plan_v =
        (V->type == GGML_TYPE_Q4_0 && plan.v_path == PDMQ_V_RAW_LDS_Q4) ||
        (V->type == GGML_TYPE_Q8_0 && plan.v_path == PDMQ_V_RAW_LDS_Q8_0) ||
        (V->type == GGML_TYPE_F16  && plan.v_path == PDMQ_V_RAW_LDS_F16);
    const bool gqa2_splitk_plan_supported = request_gqa2 && gqax_splitk_requested > 1 &&
        gqa_ratio >= 2 && nq <= 8 && !plan.kshared && gqa2_splitk_plan_v;
    const bool gqa2_supported = gqa2_xqa_supported || gqa2_legacy_supported || gqa2_splitk_plan_supported;
    const bool gqa4_non_split_supported = false;
    const bool gqa4_splitk_plan_v =
        V->type == GGML_TYPE_Q4_0 && plan.v_path == PDMQ_V_RAW_LDS_Q4;
    const bool gqa4_splitk_plan_supported = request_gqa4 && gqax_splitk_requested > 1 &&
        (shape == PDMQ_SHAPE_M1N32 || shape == PDMQ_SHAPE_M2N32 || shape == PDMQ_SHAPE_M4N32 ||
         shape == PDMQ_SHAPE_M8N32) &&
        gqa4_splitk_plan_v && !plan.kshared &&
        n_heads_q == 16 && n_heads_k == 4 && gqa_ratio == 4 && nq <= 8;
    const bool gqa4_supported = gqa4_non_split_supported || gqa4_splitk_plan_supported;
    const bool gqa6_decode_supported = request_gqa6 &&
        shape == PDMQ_SHAPE_M1N32 &&
        V->type == GGML_TYPE_Q4_0 &&
        n_heads_q == 24 && n_heads_k == 4 && gqa_ratio == 6 && nq <= 1;
    const bool gqa6_m4_candidate_allowed = shape == PDMQ_SHAPE_M4N32 &&
        (pdmq_pv_wmma_policy_requested || qblock_tetris_shape_policy_requested);
    const bool gqa6_splitk_plan_v =
        V->type == GGML_TYPE_Q4_0 && plan.v_path == PDMQ_V_RAW_LDS_Q4;
    const bool gqa6_splitk_plan_supported = request_gqa6 && gqax_splitk_requested > 1 &&
        (shape == PDMQ_SHAPE_M1N32 || shape == PDMQ_SHAPE_M2N32 || gqa6_m4_candidate_allowed) &&
        gqa6_splitk_plan_v && !plan.kshared &&
        n_heads_q == 24 && n_heads_k == 4 && gqa_ratio == 6 && nq <= 4;
    const bool gqa6_supported = gqa6_decode_supported || gqa6_splitk_plan_supported;
    const bool gqa_require = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_REQUIRE");
        return v && atoi(v) != 0;
    }();
    if (request_gqa2 && !gqa2_supported && gqa_require) {
        GGML_ABORT("PDMQ GQA2 requested but unsupported: shape=%s nq=%d hq=%d hk=%d gqa_ratio=%d",
                   pdmq_shape_name(shape), nq, n_heads_q, n_heads_k, gqa_ratio);
    }
    if (request_gqa4 && !gqa4_supported && gqa_require) {
        GGML_ABORT("PDMQ GQA4 requested but unsupported: shape=%s nq=%d hq=%d hk=%d gqa_ratio=%d V=%s",
                   pdmq_shape_name(shape), nq, n_heads_q, n_heads_k, gqa_ratio, ggml_type_name(V->type));
    }
    if (request_gqa6 && !gqa6_supported && gqa_require) {
        GGML_ABORT("PDMQ GQA6 requested but unsupported: shape=%s nq=%d hq=%d hk=%d gqa_ratio=%d V=%s split_k=%d",
                   pdmq_shape_name(shape), nq, n_heads_q, n_heads_k, gqa_ratio, ggml_type_name(V->type), gqax_splitk_requested);
    }
    const bool is_gqa2_xqa = request_gqa2 && gqa2_xqa_supported;
    const bool is_gqa2 = request_gqa2 && gqa2_supported && !is_gqa2_xqa;
    const int pdmq_layer_index = pdmq_fattn_layer_from_node_name(dst);
    const bool is_gqa4 = request_gqa4 && gqa4_supported;
    const bool is_gqa6 = request_gqa6 && gqa6_supported;
    const bool effective_gqa1 = !is_gqa2 && !is_gqa2_xqa && !is_gqa4 && !is_gqa6;
    const bool is_gqa1 = request_gqa1 && effective_gqa1;
    const bool gqa1_splitk_auto_enabled = []() {
        const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA1_SPLITK_AUTO");
        return !s || atoi(s) != 0;
    }();
    const int gqa1_splitk_min_nk = []() {
        const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA1_SPLITK_MIN_NK");
        return s && *s ? atoi(s) : 12288;
    }();
    const bool gqa1_splitk_auto_v =
        (V->type == GGML_TYPE_Q4_0 && plan.v_path == PDMQ_V_RAW_LDS_Q4) ||
        (V->type == GGML_TYPE_Q8_0 && plan.v_path == PDMQ_V_RAW_LDS_Q8_0) ||
        (V->type == GGML_TYPE_F16  && plan.v_path == PDMQ_V_RAW_LDS_F16);
    const bool gqa1_splitk_auto = gqa1_splitk_auto_enabled && !gqax_splitk_env_set && request_gqa1 &&
        gqa1_splitk_auto_v && !plan.kshared && nq <= 8 && nk >= gqa1_splitk_min_nk;
    const bool gqa6_splitk_auto = gqa1_splitk_auto_enabled && !gqax_splitk_env_set && request_gqa6 &&
        qwen27b_gqa6_policy_requested && gqa1_splitk_auto_v && !plan.kshared && nq <= 4 && nk >= gqa1_splitk_min_nk;
    if (gqa1_splitk_auto || gqa6_splitk_auto) {
        gqax_splitk_requested = 32;
    }
    const bool gqax_splitk_roof_cap = ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk_roof_cap();
    const bool gqax_splitk_compact_empty = ggml_cuda_rocm_packed16_dot4_mmq_gqax_splitk_compact_empty();
    const int gqax_splitk_roof = pdmq_splitk_roof_pow2(nk);
    const int gqax_splitk_effective = gqax_splitk_roof_cap
        ? (gqax_splitk_requested < gqax_splitk_roof ? gqax_splitk_requested : gqax_splitk_roof)
        : gqax_splitk_requested;
    const int gqax_k_blocks_total_raw = CEIL_DIV(nk, launch_bn);
    const int gqax_k_blocks_total = gqax_k_blocks_total_raw > 0 ? gqax_k_blocks_total_raw : 1;
    const int gqax_splitk_active_raw = (gqax_splitk_compact_empty && gqax_splitk_effective > gqax_k_blocks_total)
        ? gqax_k_blocks_total
        : gqax_splitk_effective;
    plan.gqa_group = is_gqa6 ? 6 : (is_gqa4 ? 4 : ((is_gqa2 || is_gqa2_xqa) ? 2 : 1));

    const int groups_per_kv = CEIL_DIV(gqa_ratio, plan.gqa_group);
    const int grid_y_grouped = n_heads_k * groups_per_kv;
    const int grid_y_old   = n_heads_q;

    dim3 block(PDMQ_THREADS);
    hipStream_t stream = ctx.stream();

    const bool kshared = plan.kshared;
    const int pdmq_k_scale_group = pdmq_k_scale_group_qblocks();
    const bool stage_v_requested = plan.v_path == PDMQ_V_STAGE_F32;
    const bool v4_k16d16_oracle = plan.v_path == PDMQ_V_RAW_LDS_Q4_K16D16_ORACLE;
    const bool v4_k16d16_144_persistent = plan.v_path == PDMQ_V_V4_K16D16_144;
#if !PDMQ_COMPILE_V4_ANY
    if (v4_k16d16_oracle || v4_k16d16_144_persistent || V->type == GGML_TYPE_V4_K16D16_144) {
        GGML_ABORT("PDMQ V4_144 exact PV4 requires -DGGML_HIP_PDMQ_COMPILE_V4_144_PV4=ON in this fast dev build");
    }
#endif
    const bool stage_v = stage_v_requested;
    const bool raw_lds_q4  = plan.v_path == PDMQ_V_RAW_LDS_Q4 || v4_k16d16_oracle;
    const bool raw_lds_q8  = plan.v_path == PDMQ_V_RAW_LDS_Q8_0;
    const bool raw_lds_f16 = plan.v_path == PDMQ_V_RAW_LDS_F16;
    const bool raw_lds = raw_lds_q4 || raw_lds_q8 || raw_lds_f16;
    const bool directv = !stage_v;
    const bool pdmq_pv_wmma_req = pdmq_pv_wmma_requested();
    const bool pdmq_pv_i8_wmma_req = pdmq_pv_i8_wmma_requested();
    const bool pdmq_pv_i4_wmma_req = pdmq_pv_i4_wmma_requested();
    const bool pdmq_pv_i4_cache_req = pdmq_pv_i4_cache_requested();
    const bool pdmq_pv_i8_wmma_coherent = pdmq_pv_i8_wmma_coherent_requested();
    const bool pdmq_pv_i8_wmma_tiny_scalar = pdmq_pv_i8_wmma_tiny_scalar_requested();
    const bool pdmq_pv_i4_padded_rows = pdmq_pv_i8_wmma_coherent;
    const bool gqa1_splitk_supported = effective_gqa1 && !kshared && nq <= 8 &&
        raw_lds && (V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16);
    const bool gqa2_splitk_supported = is_gqa2 && !kshared && nq <= 8 &&
        raw_lds && (V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16);
    const bool gqa2x_splitk_supported = is_gqa2_xqa && raw_lds_q4 && !kshared;
    const bool gqa4_splitk_supported = is_gqa4 && !kshared && raw_lds_q4 && V->type == GGML_TYPE_Q4_0 &&
        n_heads_q == 16 && n_heads_k == 4 && gqa_ratio == 4 && nq <= 8;
    const bool gqa6_splitk_supported = is_gqa6 && !kshared && raw_lds_q4 && V->type == GGML_TYPE_Q4_0 &&
        n_heads_q == 24 && n_heads_k == 4 && gqa_ratio == 6 &&
        nq <= 4 && (shape == PDMQ_SHAPE_M1N32 || shape == PDMQ_SHAPE_M2N32 || gqa6_m4_candidate_allowed);
    const bool gqax_splitk_supported = gqa1_splitk_supported || gqa2_splitk_supported || gqa2x_splitk_supported || gqa4_splitk_supported || gqa6_splitk_supported;
    const bool gqax_splitk_require = []() {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_REQUIRE");
        return v && atoi(v) != 0;
    }();
    if (gqax_splitk_requested > 1 && !gqax_splitk_supported && gqax_splitk_require) {
        GGML_ABORT("PDMQ GQAX split-K requested but unsupported: split_k=%d variant=%s shape=%s V=%s vpath=%s kshared=%d nq=%d hq=%d hk=%d",
                   gqax_splitk_requested,
                   is_gqa6 ? "GQA6" : (is_gqa4 ? "GQA4" : (is_gqa2_xqa ? "GQA2X" : (is_gqa2 ? "GQA2" : "GQA1"))),
                   pdmq_shape_name(shape), ggml_type_name(V->type), pdmq_v_path_name(plan.v_path), kshared ? 1 : 0,
                   nq, n_heads_q, n_heads_k);
    }
    const bool gqax_splitk_active_raw_supported = gqax_splitk_active_raw > 1 && gqax_splitk_supported;
    const int dp16_k_shards_per_q_stage = (gqax_splitk_active_raw_supported && (gqa1_splitk_supported || gqa4_splitk_supported || gqa6_splitk_supported)) ? dp16_fa_k_shards_per_q_stage() : 1;
    const int gqax_splitk_coarsened = CEIL_DIV(gqax_splitk_active_raw, dp16_k_shards_per_q_stage);
    const int gqax_splitk_active = (dp16_k_shards_per_q_stage > 1 && gqax_splitk_active_raw > 1)
        ? (gqax_splitk_coarsened > 1 ? gqax_splitk_coarsened : 1)
        : gqax_splitk_active_raw;
    const bool is_gqax_splitk = gqax_splitk_active > 1 && gqax_splitk_supported;
    const bool is_gqa1_splitk = is_gqax_splitk && gqa1_splitk_supported;
    const bool is_gqa2_splitk = is_gqax_splitk && gqa2_splitk_supported;
    const bool is_gqa2_xqa_splitk = is_gqax_splitk && gqa2x_splitk_supported;
    const bool is_gqa4_splitk = is_gqax_splitk && gqa4_splitk_supported;
    const bool is_gqa6_splitk = is_gqax_splitk && gqa6_splitk_supported;
    // QPACK is a split-K organizer contract, not just an nq<=4 verifier fast path.
    // PDMQ split-K supports small-Q verify/decode up to nq<=8; keep all of those
    // rows on the same prepared-Q path so late MTP verify groups do not silently
    // fall back to per-shard inline quantization.
    const dp16_fa_q_stage q_stage = (dp16_fa_qpack_i8_enabled() && (is_gqa1_splitk || is_gqa4_splitk || is_gqa6_splitk) && nq <= 8 && nk >= dp16_fa_qpack_i8_min_nk()) ?
        DP16_FA_Q_STAGE_QPACK_I8_BLOCK32 : DP16_FA_Q_STAGE_INLINE;
    const bool dp16_qpack_stage_active = q_stage == DP16_FA_Q_STAGE_QPACK_I8_BLOCK32;
    const bool pdmq_pv_wmma_route_supported = raw_lds && !kshared && (launch_bn % 16 == 0) &&
        ((is_gqa1_splitk && (V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16)) ||
         (is_gqa4_splitk && raw_lds_q4 && V->type == GGML_TYPE_Q4_0) ||
         (is_gqa6_splitk && raw_lds_q4 && V->type == GGML_TYPE_Q4_0));
    // PV-WMMA is a small-GEMM microkernel over P[M x BN] * V[BN x D].
    // It row-blocks M in 16-row chunks, so GQA4/M4 is one full tile and
    // GQA6/M4 is 16+8 padded rows. Larger M should wait for a dedicated
    // multi-CTA row-blocked design.
    const bool pdmq_pv_wmma_shape_supported = plan.gqa_group * launch_bm <= 32;
    const bool pdmq_pv_wmma_supported = pdmq_pv_wmma_route_supported && pdmq_pv_wmma_shape_supported;
    if (pdmq_pv_wmma_req && is_gqax_splitk && !pdmq_pv_wmma_route_supported) {
        GGML_ABORT("PDMQ PV-WMMA requested but unsupported on split-K route: variant=%s shape=%s(%dx%d) V=%s vpath=%s split_k=%d gqa_group=%d rows=%d kshared=%d",
                   is_gqa1_splitk ? "GQA1_SPLITK" : (is_gqa2_splitk ? "GQA2_SPLITK" : (is_gqa2_xqa_splitk ? "GQA2X_SPLITK" : (is_gqa4_splitk ? "GQA4_SPLITK" : (is_gqa6_splitk ? "GQA6_SPLITK" : "non_splitk")))),
                   pdmq_shape_name(shape), launch_bm, launch_bn, ggml_type_name(V->type), pdmq_v_path_name(plan.v_path),
                   is_gqax_splitk ? gqax_splitk_active : 0, plan.gqa_group, plan.gqa_group * launch_bm, kshared ? 1 : 0);
    }
    const bool pdmq_pv_wmma_active = pdmq_pv_wmma_req && pdmq_pv_wmma_supported;
    const bool pdmq_pv_i8_wmma_full_tile_available = plan.role == PDMQ_ROLE_VERIFY && plan.gqa_group * launch_bm >= 16;
    const bool pdmq_pv_i4_wmma_active = pdmq_pv_wmma_active && pdmq_pv_i4_wmma_req && !pdmq_pv_i8_wmma_req &&
        (pdmq_pv_i4_padded_rows || pdmq_pv_i8_wmma_full_tile_available) && raw_lds_q4 && V->type == GGML_TYPE_Q4_0;
    const bool pdmq_pv_i8_wmma_active = !pdmq_pv_i4_wmma_active && pdmq_pv_wmma_active && pdmq_pv_i8_wmma_req &&
        (pdmq_pv_i8_wmma_coherent || pdmq_pv_i8_wmma_full_tile_available) && raw_lds_q4 && V->type == GGML_TYPE_Q4_0;
    const bool pdmq_pvblock_exact_env_enabled = pdmq_pvblock_exact_enabled_by_env();
    const bool pdmq_pvblock_exact_supported = raw_lds_q4 && V->type == GGML_TYPE_Q4_0 && !kshared && !stage_v &&
        !pdmq_pv_wmma_active && !pdmq_pv_i8_wmma_active && !pdmq_pv_i4_wmma_active;
    const bool pdmq_pvblock_exact_active = pdmq_pvblock_exact_env_enabled && pdmq_pvblock_exact_supported;
    const int pdmq_pvblock_mode = pdmq_pvblock_exact_active ? DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR : DP16_FA_QBLOCK_PVBLOCK_NONE;
    int * v_i4_cache_words = nullptr;
    float * v_i4_cache_scales = nullptr;
    int v_i4_cache_k_blocks = 0;
    const int pdmq_pv_i4_cache_scope_val = pdmq_pv_i4_cache_scope();
    const bool pdmq_pv_i4_cache_inst_draft = fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK && nq == 1;
    const bool pdmq_pv_i4_cache_inst_mtp = pdmq_pv_i4_cache_inst_draft ||
        ((fa_inst_i32 == GGML_FATTN_INST_MTP_VERIFY_QK || fa_inst_i32 == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK) && nq <= 4);
    const bool pdmq_pv_i4_cache_inst_all = nq <= 4;
    const bool pdmq_pv_i4_cache_scope_ok = pdmq_pv_i4_cache_scope_val == 0 ? pdmq_pv_i4_cache_inst_draft :
        (pdmq_pv_i4_cache_scope_val == 1 ? pdmq_pv_i4_cache_inst_mtp : pdmq_pv_i4_cache_inst_all);
    const bool pdmq_pv_i4_cache_active = pdmq_pv_i4_wmma_active && pdmq_pv_i4_cache_req &&
        pdmq_pv_i4_cache_scope_ok && raw_lds_q4 && V->type == GGML_TYPE_Q4_0;
    if (pdmq_pv_i4_cache_active) {
        pdmq_prepare_v_i4_cache(V, nk, n_heads_k, batch, stream, &v_i4_cache_words, &v_i4_cache_scales, &v_i4_cache_k_blocks);
    }
    plan.split_k = is_gqax_splitk;
    const int grid_z = batch * (is_gqax_splitk ? gqax_splitk_active : 1);
    dim3 grid(CEIL_DIV(nq, launch_bm),
        (is_gqa2 || is_gqa2_xqa || is_gqa4 || is_gqa6 || is_gqa1_splitk) ? grid_y_grouped : n_heads_q,
        grid_z);

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

    dp16_fa_qblock_program qblock_program = dp16_fa_qblock_program_make(
        qblock_inst,
        launch_bm,
        plan.gqa_group,
        launch_bn,
        is_gqax_splitk ? gqax_splitk_active : 1,
        q_stage,
        stage_v,
        raw_lds,
        directv);
    qblock_program.pvblock_mode = pdmq_pvblock_mode;
    const bool pdmq_v4_144_pv4_req = v4_k16d16_144_persistent && pdmq_v4_144_pv4_requested();
    const bool pdmq_v4_144_pv4_supported = v4_k16d16_144_persistent && directv && !stage_v &&
        !pdmq_pv_wmma_active && !pdmq_pv_i8_wmma_active && !pdmq_pv_i4_wmma_active;
    if (pdmq_v4_144_pv4_req && !pdmq_v4_144_pv4_supported) {
        GGML_ABORT("GGML_CUDA_ROCM_V4_K16D16_144_PV4 requested but unsupported: V=%s vpath=%s stage_v=%d directv=%d pv_wmma=%d pv_i8=%d pv_i4=%d",
                   ggml_type_name(V->type), pdmq_v_path_name(plan.v_path), stage_v ? 1 : 0, directv ? 1 : 0,
                   pdmq_pv_wmma_active ? 1 : 0, pdmq_pv_i8_wmma_active ? 1 : 0,
                   pdmq_pv_i4_wmma_active ? 1 : 0);
    }
    if (pdmq_v4_144_pv4_req) {
        qblock_program.v_decode_mode = DP16_FA_QBLOCK_V_DECODE_RAW_LDS;
    }
    dp16_fa_qblock_program_apply_debug_rowmap(qblock_program);

    GGML_ASSERT(!raw_lds_q4  || V->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(!raw_lds_q8  || V->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(!raw_lds_f16 || V->type == GGML_TYPE_F16);
    const bool pdmq_verbose =
        (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")) != 0) ||
        (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE")) != 0) ||
        (getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VERBOSE")) != 0);
    const bool qblock_route_trace = qblock_inst && (
        (getenv("LLAMA_MTP_QBLOCK_TRACE") && atoi(getenv("LLAMA_MTP_QBLOCK_TRACE")) != 0) ||
        pdmq_verbose);
    const char * pdmq_variant_name =
        is_gqa1_splitk ? "GQA1_SPLITK" : (is_gqa2_splitk ? "GQA2_SPLITK" : (is_gqa2_xqa_splitk ? "GQA2X_SPLITK" : (is_gqa4_splitk ? "GQA4_SPLITK" : (is_gqa6_splitk ? "GQA6_SPLITK" : (is_gqa4 ? "GQA4" : (is_gqa6 ? "GQA6" : (is_gqa2_xqa ? "GQA2X" : (is_gqa2 ? "GQA2" : "GQA1"))))))));
    if (qblock_route_trace) {
        const char * qblock_nq_env = getenv("LLAMA_MTP_QBLOCK_NQ");
        fprintf(stderr,
                "MTP_QBLOCK_ROUTE: backend=rocm_packed16_dot4_mmq node=%s layer=%d graph_inst=%d variant=%s shape=%s bm=%d bn=%d nq=%d qblock_nq_env=%s nk=%d hq=%d hk=%d gqa_ratio=%d gqa_group=%d gqa_request=%d split_k=%d split_k_requested=%d split_k_effective=%d k_shards_per_q_stage=%d q_stage=%s vpath=%s mask=%d causal=%d stage_v=%d raw_lds_q4=%d v4_k16d16_oracle=%d v4_k16d16_144_persistent=%d directv=%d pv_wmma=%d pv_i8_wmma=%d pv_i4_wmma=%d pv_i4_cache=%d pvblock_exact=%d qprog=%d qprog_mode=%s qprog_qprec=%s qprog_rows=%d qprog_rowmask=0x%x qprog_kind0=%d qprog_kind1=%d qprog_qmap0=%d qprog_qmap1=%d qprog_hmap0=%d qprog_pv_inline=%d qprog_vdecode=%d qprog_o_res=%d grid=(%u,%u,%u) qblock_shape_env=%d qblock_shape_policy=%s qblock_gqa_env=%d qblock_splitk_env=%d\n",
                pdmq_fattn_node_name(dst),
                pdmq_fattn_layer_from_node_name(dst),
                fa_inst_i32,
                pdmq_variant_name,
                pdmq_shape_name(plan.shape),
                launch_bm,
                launch_bn,
                nq,
                qblock_nq_env && *qblock_nq_env ? qblock_nq_env : "none",
                nk,
                n_heads_q,
                n_heads_k,
                gqa_ratio,
                plan.gqa_group,
                plan.requested_gqa_group,
                is_gqax_splitk ? gqax_splitk_active : 0,
                gqax_splitk_requested,
                gqax_splitk_effective,
                dp16_k_shards_per_q_stage,
                dp16_fa_q_stage_name(q_stage),
                pdmq_v_path_name(plan.v_path),
                mask ? 1 : 0,
                assume_causal ? 1 : 0,
                stage_v ? 1 : 0,
                raw_lds_q4 ? 1 : 0,
                v4_k16d16_oracle ? 1 : 0,
                v4_k16d16_144_persistent ? 1 : 0,
                directv ? 1 : 0,
                pdmq_pv_wmma_active ? 1 : 0,
                pdmq_pv_i8_wmma_active ? 1 : 0,
                pdmq_pv_i4_wmma_active ? 1 : 0,
                pdmq_pv_i4_cache_active && v_i4_cache_words && v_i4_cache_scales ? 1 : 0,
                qblock_program.pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR ? 1 : 0,
                qblock_program.enabled,
                dp16_fa_qblock_rowmap_mode_name(qblock_program.rowmap_mode),
                dp16_fa_qblock_q_precision_mode_name(qblock_program.q_precision_mode),
                qblock_program.rows_per_cta,
                qblock_program.row_valid_mask,
                qblock_program.row_kind[0],
                qblock_program.row_kind[1],
                qblock_program.row_q_delta[0],
                qblock_program.row_q_delta[1],
                qblock_program.head_slot_delta[0],
                qblock_program.pv_consume_mode == DP16_FA_QBLOCK_PV_CONSUME_INLINE ? 1 : 0,
                qblock_program.v_decode_mode,
                qblock_program.o_residency,
                grid.x, grid.y, grid.z,
                qblock_shape_env && *qblock_shape_env ? 1 : 0,
                qblock_tetris_shape_policy_requested ? "tetris" : "safe",
                qblock_gqa_group_env_set ? 1 : 0,
                qblock_splitk_env_set ? 1 : 0);
    }


    { static bool once = false; if (pdmq_verbose || !once) {
        fprintf(stderr,
            "PDMQ2 route=rocm_packed16_dot4_mmq backend=dot4_packed16_fa node=%s layer=%d graph_inst=%d variant=%s "
            "shape=%s(%dx%d) vpath=%s gqa_group=%d gqa_request=%d role=%s "
            "nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d grid_y_old=%d grid_y_new=%d effective_grid_y=%d b=%d sc=%g "
            "K=%s V=%s mask=%d f16_sidecar=%d packed_rows=%d head_stride=%d causal=%d stage_v=%d raw_lds_q4=%d v4_k16d16_oracle=%d v4_k16d16_144_persistent=%d raw_lds_q8=%d raw_lds_f16=%d kshared=%d directv=%d qblock=%d qprog=%d qprog_mode=%s qprog_qprec=%s qprog_rows=%d qprog_rowmask=0x%x qprog_kind0=%d qprog_kind1=%d qprog_qmap0=%d qprog_qmap1=%d qprog_hmap0=%d qprog_vdecode=%d qblock_shape_env=%d qblock_gqa_env=%d qblock_splitk_env=%d split_k=%d split_k_uncoarsened=%d k_shards_per_q_stage=%d q_stage=%s pv_wmma=%d pv_i8_wmma=%d pv_i4_wmma=%d pv_i4_cache=%d pvblock_exact=%d k_scale_group=%d k_block_tokens=%d split_k_requested=%d split_k_effective=%d split_k_compact_empty=%d split_k_roof_cap=%d split_k_roof=%d k_blocks_total=%d\n",
            pdmq_fattn_node_name(dst), pdmq_layer_index, fa_inst_i32,
            is_gqa1_splitk ? "GQA1_SPLITK" : (is_gqa2_splitk ? "GQA2_SPLITK" : (is_gqa2_xqa_splitk ? "GQA2X_SPLITK" : (is_gqa4_splitk ? "GQA4_SPLITK" : (is_gqa6_splitk ? "GQA6_SPLITK" : (is_gqa4 ? "GQA4" : (is_gqa6 ? "GQA6" : (is_gqa2_xqa ? "GQA2X" : (is_gqa2 ? "GQA2" : "GQA1")))))))),
            pdmq_shape_name(plan.shape), launch_bm, launch_bn,
            pdmq_v_path_name(plan.v_path), plan.gqa_group, plan.requested_gqa_group, pdmq_role_name(plan.role),
            nq, nk, n_heads_q, n_heads_k, gqa_ratio,
            grid_y_old, (is_gqa2 || is_gqa2_xqa || is_gqa4 || is_gqa6) ? grid_y_grouped : grid_y_old,
            ((is_gqa2 || is_gqa2_xqa || is_gqa4 || is_gqa6 || is_gqa1_splitk) ? grid_y_grouped : grid_y_old) * (is_gqax_splitk ? gqax_splitk_active : 1),
            batch, (double) attention_scale,
            ggml_type_name(K->type), ggml_type_name(V->type), mask ? 1 : 0, k_f16_hotcold_sidecar ? 1 : 0, packed_rows, packed_rows / n_heads_k, assume_causal ? 1 : 0,
            stage_v ? 1 : 0, raw_lds_q4 ? 1 : 0, v4_k16d16_oracle ? 1 : 0, v4_k16d16_144_persistent ? 1 : 0, raw_lds_q8 ? 1 : 0, raw_lds_f16 ? 1 : 0, kshared ? 1 : 0, directv ? 1 : 0,
            qblock_inst ? 1 : 0, qblock_program.enabled, dp16_fa_qblock_rowmap_mode_name(qblock_program.rowmap_mode), dp16_fa_qblock_q_precision_mode_name(qblock_program.q_precision_mode), qblock_program.rows_per_cta, qblock_program.row_valid_mask,
            qblock_program.row_kind[0], qblock_program.row_kind[1], qblock_program.row_q_delta[0], qblock_program.row_q_delta[1], qblock_program.head_slot_delta[0],
            qblock_program.v_decode_mode,
            qblock_shape_env && *qblock_shape_env ? 1 : 0, qblock_gqa_group_env_set ? 1 : 0, qblock_splitk_env_set ? 1 : 0,
            is_gqax_splitk ? gqax_splitk_active : 0,
            gqax_splitk_active_raw_supported ? gqax_splitk_active_raw : 0,
            dp16_k_shards_per_q_stage,
            dp16_fa_q_stage_name(q_stage),
            pdmq_pv_wmma_active ? 1 : 0,
            pdmq_pv_i8_wmma_active ? 1 : 0,
            pdmq_pv_i4_wmma_active ? 1 : 0,
            pdmq_pv_i4_cache_active && v_i4_cache_words && v_i4_cache_scales ? 1 : 0,
            qblock_program.pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR ? 1 : 0,
            pdmq_k_scale_group,
            is_gqax_splitk ? CEIL_DIV(nk, gqax_splitk_active) : 0,
            gqax_splitk_requested, gqax_splitk_effective, gqax_splitk_compact_empty ? 1 : 0, gqax_splitk_roof_cap ? 1 : 0, gqax_splitk_roof, gqax_k_blocks_total);
    }
    if (!once) { once = true; }}

    if (plan.role == PDMQ_ROLE_VERIFY && nq >= 5 && nq <= 8) {
        static bool hot_verify_once = false;
        if (!hot_verify_once || pdmq_verbose) {
            hot_verify_once = true;
            fprintf(stderr,
                "PDMQ2_HOT_VERIFY_ROUTE: variant=%s node=%s layer=%d graph_inst=%d shape=%s bm=%d bn=%d nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d gqa_group=%d gqa_request=%d grid=(%u,%u,%u) split_k=%d split_k_requested=%d split_k_effective=%d k_blocks_total=%d k_shards_per_q_stage=%d q_stage=%s vpath=%s V=%s raw_lds_q4=%d v4_k16d16_oracle=%d v4_k16d16_144_persistent=%d qblock=%d pvblock_exact=%d pv_wmma=%d pv_i8_wmma=%d pv_i4_wmma=%d\n",
                pdmq_variant_name, pdmq_fattn_node_name(dst), pdmq_fattn_layer_from_node_name(dst), fa_inst_i32,
                pdmq_shape_name(plan.shape), launch_bm, launch_bn, nq, nk, n_heads_q, n_heads_k,
                gqa_ratio, plan.gqa_group, plan.requested_gqa_group, grid.x, grid.y, grid.z,
                is_gqax_splitk ? gqax_splitk_active : 0, gqax_splitk_requested, gqax_splitk_effective,
                gqax_k_blocks_total, dp16_k_shards_per_q_stage, dp16_fa_q_stage_name(q_stage),
                pdmq_v_path_name(plan.v_path), ggml_type_name(V->type), raw_lds_q4 ? 1 : 0, v4_k16d16_oracle ? 1 : 0, v4_k16d16_144_persistent ? 1 : 0,
                qblock_inst ? 1 : 0,
                qblock_program.pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR ? 1 : 0,
                pdmq_pv_wmma_active ? 1 : 0, pdmq_pv_i8_wmma_active ? 1 : 0, pdmq_pv_i4_wmma_active ? 1 : 0);
        }
    }
    if (plan.role == PDMQ_ROLE_DECODE && nq == 1) {
        static bool decode_once = false;
        if (!decode_once || pdmq_verbose) {
            decode_once = true;
            fprintf(stderr,
                "PDMQ2_DECODE_ROUTE: variant=%s node=%s layer=%d graph_inst=%d shape=%s bm=%d bn=%d nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d gqa_group=%d gqa_request=%d grid=(%u,%u,%u) split_k=%d q_stage=%s vpath=%s V=%s raw_lds_q4=%d v4_k16d16_oracle=%d v4_k16d16_144_persistent=%d qblock=%d pvblock_exact=%d pv_wmma=%d pv_i8_wmma=%d pv_i4_wmma=%d\n",
                pdmq_variant_name, pdmq_fattn_node_name(dst), pdmq_fattn_layer_from_node_name(dst), fa_inst_i32,
                pdmq_shape_name(plan.shape), launch_bm, launch_bn, nq, nk, n_heads_q, n_heads_k,
                gqa_ratio, plan.gqa_group, plan.requested_gqa_group, grid.x, grid.y, grid.z,
                is_gqax_splitk ? gqax_splitk_active : 0, dp16_fa_q_stage_name(q_stage),
                pdmq_v_path_name(plan.v_path), ggml_type_name(V->type), raw_lds_q4 ? 1 : 0, v4_k16d16_oracle ? 1 : 0, v4_k16d16_144_persistent ? 1 : 0,
                qblock_inst ? 1 : 0,
                qblock_program.pvblock_mode == DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR ? 1 : 0,
                pdmq_pv_wmma_active ? 1 : 0, pdmq_pv_i8_wmma_active ? 1 : 0, pdmq_pv_i4_wmma_active ? 1 : 0);
        }
    }


    dp16_fa_q_view q_view = dp16_fa_make_inline_q_view(Q, PDMQ_D);
    q_view.qblock_program = qblock_program;

#define PDMQ_LAUNCH_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED) do { \
    if constexpr (VT == PACKED16_DOT4_MMQ_V_Q4_0) { \
        if (pdmq_pvblock_exact_active && !(STAGE_V)) { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, PDMQ_COMPILE_PVBLOCK_EXACT, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
            } else { \
                packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
            } \
        } else { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, false, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
            } else { \
                packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, false><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
            } \
        } \
    } else { \
        if (pdmq_pvblock_exact_active && !(STAGE_V)) { \
            packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
        } else { \
            packed16_dot4_mmq_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V, RAW_LDS_Q4, KSHARED, false><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, pdmq_pvblock_mode); \
        } \
    } \
} while (0)

#define PDMQ_LAUNCH_GQA2_SHAPE(VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL) do { \
    if (v4_k16d16_oracle) { \
        packed16_dot4_mmq_gqa2_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL, true><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } else { \
        packed16_dot4_mmq_gqa2_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL><<<grid, block, 0, stream>>>( \
            (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, \
            Q->nb[1], Q->nb[2], Q->nb[3], \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
            v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
            mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
            (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale); \
    } \
} while (0)

#define PDMQ_COMPACT_MATRIX_ABORT(WHAT) \
    GGML_ABORT("packed16_dot4_mmq compact build does not compile " WHAT)

#define PDMQ_LAUNCH_GQAX_SHAPE(GROUP_VAL, VT, BM_VAL, BN_VAL, CAUSAL, STAGE_V_VAL, RAW_LDS_Q4_VAL, KSHARED_VAL) do { \
    if constexpr (GROUP_VAL == 6 && VT == PACKED16_DOT4_MMQ_V_Q4_0) { \
        if constexpr (STAGE_V_VAL || KSHARED_VAL || !RAW_LDS_Q4_VAL) { \
            PDMQ_COMPACT_MATRIX_ABORT("non-raw/kshared compact GQA6 kernels"); \
        } else if (pdmq_pvblock_exact_active) { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, true, false, 6, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
            } else { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, true, false, 6, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
            } \
        } else { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, true, false, 6, false, false, true, false, false, false, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
            } else { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, true, false, 6><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
            } \
        } \
    } else if constexpr (GROUP_VAL == 6 && (VT == PACKED16_DOT4_MMQ_V4_K16D16_144)) { \
        if constexpr (STAGE_V_VAL || KSHARED_VAL || RAW_LDS_Q4_VAL) { \
            PDMQ_COMPACT_MATRIX_ABORT("non-direct persistent V4 compact GQA6 kernels"); \
        } else if (pdmq_pvblock_exact_active) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, false, false, 6, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
        } else { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, false, false, 6><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, nullptr, nullptr, nullptr, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, 1, nullptr, nullptr, qblock_program, nullptr, nullptr, 0); \
        } \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("experimental grouped-GQA non-split kernels"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, BM_VAL, BN_VAL, CAUSAL, RAW_LDS_Q4_VAL, PM, PL, PO) do { \
    if constexpr (GROUP_VAL == 1 || ((GROUP_VAL == 2 || GROUP_VAL == 4 || GROUP_VAL == 6) && VT == PACKED16_DOT4_MMQ_V_Q4_0)) { \
        if (pdmq_pvblock_exact_active) { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
            } else { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
            } \
        } else { \
            if (v4_k16d16_oracle) { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, false, false, true, false, false, false, true><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
            } else { \
                packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL><<<grid, block, 0, stream>>>( \
                    (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                    Q->nb[1], Q->nb[2], Q->nb[3], \
                    V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                    v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                    mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                    (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                    nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
            } \
        } \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("experimental grouped-GQA split-K kernels"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(GROUP_VAL, VT, CAUSAL, PM, PL, PO) do { \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, true, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, true, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, true, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, true, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(GROUP_VAL, VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, true, PM, PL, PO); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("non-standard split-K shapes"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, BM_VAL, BN_VAL, CAUSAL, PM, PL, PO) do { \
    if constexpr (GROUP_VAL == 1) { \
        if (pdmq_pvblock_exact_active) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, false, false, GROUP_VAL, false, false, true, false, false, PDMQ_COMPILE_PVBLOCK_EXACT><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } else { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, false, false, GROUP_VAL><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("direct-V grouped split-K kernels"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_DIRECTV(GROUP_VAL, VT, CAUSAL, PM, PL, PO) do { \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_DIRECTV(GROUP_VAL, VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, PM, PL, PO); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("non-standard direct-V split-K shapes"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, BM_VAL, BN_VAL, CAUSAL, RAW_LDS_Q4_VAL, PM, PL, PO) do { \
    if constexpr (GROUP_VAL == 1 || ((GROUP_VAL == 4 || GROUP_VAL == 6) && VT == PACKED16_DOT4_MMQ_V_Q4_0)) { \
        if (pdmq_pv_i4_wmma_active && pdmq_pv_i4_padded_rows) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, true, false, false, false, true><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } else if (pdmq_pv_i4_wmma_active) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, true, false, true, false, true><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } else if (pdmq_pv_i8_wmma_active && pdmq_pv_i8_wmma_coherent) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, true, true, false><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } else if (pdmq_pv_i8_wmma_active) { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, true, true, true><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } else { \
            packed16_dot4_mmq_gqax_kernel<VT, BM_VAL, BN_VAL, PDMQ_D, CAUSAL, false, RAW_LDS_Q4_VAL, false, GROUP_VAL, true, false><<<grid, block, 0, stream>>>( \
                (const float *) Q->data, (const char *) V->data, (const half *) (v4_tail ? v4_tail->data : nullptr), (float *) dst->data, PM, PL, PO, \
                Q->nb[1], Q->nb[2], Q->nb[3], \
                V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
                v4_tail ? v4_tail->nb[0] : 0, v4_tail ? v4_tail->nb[1] : 0, v4_tail ? v4_tail->nb[2] : 0, \
                mask ? (const char *) mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
                (const int *) packed16_payload->data, (const half *) packed16_scales->data, \
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_rows, q_offset, attention_scale, batch, gqax_splitk_active, q_view.qpack_payload, q_view.qpack_scales, qblock_program, v_i4_cache_words, v_i4_cache_scales, v_i4_cache_k_blocks); \
        } \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("experimental grouped-GQA PV-WMMA split-K kernels"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA(GROUP_VAL, VT, CAUSAL, PM, PL, PO) do { \
    constexpr bool pvwmma_raw_lds_q4 = true; \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("non-standard PV-WMMA split-K shapes"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA_ROWS16(GROUP_VAL, VT, CAUSAL, PM, PL, PO) do { \
    constexpr bool pvwmma_raw_lds_q4 = true; \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA(GROUP_VAL, VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, pvwmma_raw_lds_q4, PM, PL, PO); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("compact PV-WMMA split-K rows16 shapes"); \
    } \
} while (0)

#define PDMQ_LAUNCH_GQA4_IMPL(VT, BM_VAL, BN_VAL, CAUSAL, kshared_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA4 kernels")
#define PDMQ_LAUNCH_GQA4(VT, CAUSAL, kshared_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA4 kernels")
#define PDMQ_LAUNCH_GQA6_IMPL(VT, BM_VAL, BN_VAL, CAUSAL, kshared_var, raw_lds_q4_var) do { \
    if constexpr (VT == PACKED16_DOT4_MMQ_V4_K16D16_144) { \
        if (kshared_var || raw_lds_q4_var) { \
            PDMQ_COMPACT_MATRIX_ABORT("non-direct persistent V4 GQA6 kernels"); \
        } else { \
            PDMQ_LAUNCH_GQAX_SHAPE(6, VT, BM_VAL, BN_VAL, CAUSAL, false, false, false); \
        } \
    } else if (kshared_var || !raw_lds_q4_var) { \
        PDMQ_COMPACT_MATRIX_ABORT("non-raw/kshared GQA6 kernels"); \
    } else { \
        PDMQ_LAUNCH_GQAX_SHAPE(6, VT, BM_VAL, BN_VAL, CAUSAL, false, true, false); \
    } \
} while (0)
#define PDMQ_LAUNCH_GQA6(VT, CAUSAL, kshared_var, raw_lds_q4_var) do { \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_GQA6_IMPL(VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, kshared_var, raw_lds_q4_var); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("non-M1N32 compact GQA6 kernels"); \
    } \
} while (0)
#define PDMQ_LAUNCH_GQA2X_IMPL(VT, BM_VAL, BN_VAL, CAUSAL, kshared_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA2X kernels")
#define PDMQ_LAUNCH_GQA2X(VT, CAUSAL, kshared_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA2X kernels")
#define PDMQ_LAUNCH_GQA2_IMPL(VT, BM_VAL, BN_VAL, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA2 kernels")
#define PDMQ_LAUNCH_GQA2(VT, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA2 kernels")
#define PDMQ_LAUNCH_GQA2_NO_RAW(VT, CAUSAL, kshared_var, directv_var) \
    PDMQ_COMPACT_MATRIX_ABORT("GQA2 kernels")

#define PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, kshared_val, stagev_val, raw_lds_q4_val) do { \
    if (shape == PDMQ_SHAPE_M1N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M2N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M4N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M16N16) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else if (shape == PDMQ_SHAPE_M8N32) { \
        PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, stagev_val, raw_lds_q4_val, kshared_val); \
    } else { \
        PDMQ_COMPACT_MATRIX_ABORT("non-standard GQA1 shapes"); \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED(VT, CAUSAL, kshared_var, directv_var, raw_lds_q4_var) do { \
    if (kshared_var) { \
        PDMQ_COMPACT_MATRIX_ABORT("kshared PDMQ variants"); \
    } else if constexpr (VT == PACKED16_DOT4_MMQ_V_Q4_0) { \
        if (raw_lds_q4_var) { \
            PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, true); \
        } else { \
            PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, false); \
        } \
    } else if constexpr (VT == PACKED16_DOT4_MMQ_V4_K16D16_144) { \
        if (raw_lds_q4_var || !directv_var) { \
            PDMQ_COMPACT_MATRIX_ABORT("staged/raw persistent V4 K16D16 V paths"); \
        } else { \
            PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, false); \
        } \
    } else { \
        if (raw_lds_q4_var) { \
            PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, false, true); \
        } else if (!directv_var) { \
            PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, true, false); \
        } else { \
            PDMQ_COMPACT_MATRIX_ABORT("direct non-raw q8/f16 V paths"); \
        } \
    } \
} while (0)

#define PDMQ_LAUNCH_TYPED_NO_RAW(VT, CAUSAL, kshared_var, directv_var) do { \
    if (kshared_var || directv_var) { \
        PDMQ_COMPACT_MATRIX_ABORT("non-standard no-raw PDMQ variants"); \
    } else { \
        PDMQ_LAUNCH_TYPED_IMPL(VT, CAUSAL, false, true, false); \
    } \
} while (0)

#if !PDMQ_QWEN35_DEBUG_ONLY
if (is_gqa1_splitk) {
    GGML_ASSERT(raw_lds && !kshared);
    const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
    ggml_cuda_pool & pool = ctx.pool();
    dp16_fa_q_workspace q_workspace(pool);
    if (dp16_qpack_stage_active) {
        q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
        q_view.qblock_program = qblock_program;
    }
    ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
    ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
    ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
#define PDMQ_LAUNCH_GQA1_SPLITK_SCALAR(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0: PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(1, PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        case GGML_TYPE_Q8_0: PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(1, PACKED16_DOT4_MMQ_V_Q8_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        case GGML_TYPE_F16:  PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(1, PACKED16_DOT4_MMQ_V_F16,  CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        default: GGML_ABORT("packed16_dot4_mmq gqa1 split-K: unsupported V type"); \
    }
#define PDMQ_LAUNCH_GQA1_SPLITK_PVWMMA(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0:          PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA(1, PACKED16_DOT4_MMQ_V_Q4_0,        CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        case GGML_TYPE_Q8_0:          PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA(1, PACKED16_DOT4_MMQ_V_Q8_0,        CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        case GGML_TYPE_F16:           PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA(1, PACKED16_DOT4_MMQ_V_F16,         CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
        default: GGML_ABORT("packed16_dot4_mmq gqa1 split-K PV-WMMA: unsupported V type"); \
    }
    if (assume_causal) {
        if (pdmq_pv_wmma_active) {
            PDMQ_LAUNCH_GQA1_SPLITK_PVWMMA(true);
        } else {
            PDMQ_LAUNCH_GQA1_SPLITK_SCALAR(true);
        }
    } else {
        if (pdmq_pv_wmma_active) {
            PDMQ_LAUNCH_GQA1_SPLITK_PVWMMA(false);
        } else {
            PDMQ_LAUNCH_GQA1_SPLITK_SCALAR(false);
        }
    }
#undef PDMQ_LAUNCH_GQA1_SPLITK_PVWMMA
#undef PDMQ_LAUNCH_GQA1_SPLITK_SCALAR
    const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
    const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
    packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
        partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
        nq, nk, n_heads_q, batch, gqax_splitk_active);
} else if (is_gqa4) {
    if (is_gqa4_splitk) {
        GGML_ASSERT(V->type == GGML_TYPE_Q4_0 && raw_lds_q4 && !kshared);
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        dp16_fa_q_workspace q_workspace(pool);
        if (dp16_qpack_stage_active) {
            q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
        q_view.qblock_program = qblock_program;
        }
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
#define PDMQ_LAUNCH_GQA4_SPLITK_SCALAR(CAUSAL) \
        PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(4, PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get())
#define PDMQ_LAUNCH_GQA4_SPLITK_PVWMMA(CAUSAL) \
        switch (V->type) { \
            case GGML_TYPE_Q4_0:          PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA_ROWS16(4, PACKED16_DOT4_MMQ_V_Q4_0,        CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
            default: GGML_ABORT("packed16_dot4_mmq gqa4 split-K PV-WMMA: unsupported V type"); \
        }
        if (assume_causal) {
            if (pdmq_pv_wmma_active) {
                PDMQ_LAUNCH_GQA4_SPLITK_PVWMMA(true);
            } else {
                PDMQ_LAUNCH_GQA4_SPLITK_SCALAR(true);
            }
        } else {
            if (pdmq_pv_wmma_active) {
                PDMQ_LAUNCH_GQA4_SPLITK_PVWMMA(false);
            } else {
                PDMQ_LAUNCH_GQA4_SPLITK_SCALAR(false);
            }
        }
#undef PDMQ_LAUNCH_GQA4_SPLITK_PVWMMA
#undef PDMQ_LAUNCH_GQA4_SPLITK_SCALAR
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA4(PACKED16_DOT4_MMQ_V_Q4_0, true, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa4: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA4(PACKED16_DOT4_MMQ_V_Q4_0, false, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa4: unsupported V type");
        }
    }
} else if (is_gqa6) {
    if (is_gqa6_splitk) {
        GGML_ASSERT(V->type == GGML_TYPE_Q4_0 && raw_lds_q4 && !kshared);
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        dp16_fa_q_workspace q_workspace(pool);
        if (dp16_qpack_stage_active) {
            q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
        q_view.qblock_program = qblock_program;
        }
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
#define PDMQ_LAUNCH_GQA6_SPLITK_SCALAR(CAUSAL) \
        PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get())
#define PDMQ_LAUNCH_GQA6_SPLITK_PVWMMA(CAUSAL) \
        switch (V->type) { \
            case GGML_TYPE_Q4_0:          PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA_ROWS16(6, PACKED16_DOT4_MMQ_V_Q4_0,        CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
            default: GGML_ABORT("packed16_dot4_mmq gqa6 split-K PV-WMMA: unsupported V type"); \
        }
        if (assume_causal) {
            if (pdmq_pv_wmma_active) {
                PDMQ_LAUNCH_GQA6_SPLITK_PVWMMA(true);
            } else {
                PDMQ_LAUNCH_GQA6_SPLITK_SCALAR(true);
            }
        } else {
            if (pdmq_pv_wmma_active) {
                PDMQ_LAUNCH_GQA6_SPLITK_PVWMMA(false);
            } else {
                PDMQ_LAUNCH_GQA6_SPLITK_SCALAR(false);
            }
        }
#undef PDMQ_LAUNCH_GQA6_SPLITK_PVWMMA
#undef PDMQ_LAUNCH_GQA6_SPLITK_SCALAR
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V_Q4_0, true, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa6: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V_Q4_0, false, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa6: unsupported V type");
        }
    }
} else if (is_gqa2_xqa) {
    if (is_gqa2_xqa_splitk) {
        GGML_ASSERT(V->type == GGML_TYPE_Q4_0 && raw_lds_q4 && !kshared);
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
        if (shape != PDMQ_SHAPE_M8N32) {
            GGML_ABORT("PDMQ GQA2X split-K launch requested for unsupported shape=%s", pdmq_shape_name(shape));
        }
        if (assume_causal) {
            PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(2, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_SMALL, PDMQ_BN_SMALL, true, true, partial_m.get(), partial_l.get(), partial_out.get());
        } else {
            PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(2, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_SMALL, PDMQ_BN_SMALL, false, true, partial_m.get(), partial_l.get(), partial_out.get());
        }
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2X(PACKED16_DOT4_MMQ_V_Q4_0, true, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2x: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2X(PACKED16_DOT4_MMQ_V_Q4_0, false, kshared, raw_lds_q4); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2x: unsupported V type");
        }
    }
} else if (is_gqa2) {
    if (is_gqa2_splitk) {
        GGML_ASSERT(raw_lds && !kshared);
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
#define PDMQ_LAUNCH_GQA2_SPLITK(CAUSAL) \
        switch (V->type) { \
            case GGML_TYPE_Q4_0: PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(2, PACKED16_DOT4_MMQ_V_Q4_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
            case GGML_TYPE_Q8_0: PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(2, PACKED16_DOT4_MMQ_V_Q8_0, CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
            case GGML_TYPE_F16:  PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(2, PACKED16_DOT4_MMQ_V_F16,  CAUSAL, partial_m.get(), partial_l.get(), partial_out.get()); break; \
            default: GGML_ABORT("packed16_dot4_mmq gqa2 split-K: unsupported V type"); \
        }
        if (assume_causal) {
            PDMQ_LAUNCH_GQA2_SPLITK(true);
        } else {
            PDMQ_LAUNCH_GQA2_SPLITK(false);
        }
#undef PDMQ_LAUNCH_GQA2_SPLITK
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (assume_causal) {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0,      true, kshared, directv, raw_lds); break;
            case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q8_0,      true, kshared, directv, raw_lds); break;
            case GGML_TYPE_F16:       PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_F16,       true, kshared, directv, raw_lds); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    } else {
        switch (V->type) {
            case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q4_0,      false, kshared, directv, raw_lds); break;
            case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_Q8_0,      false, kshared, directv, raw_lds); break;
            case GGML_TYPE_F16:       PDMQ_LAUNCH_GQA2(PACKED16_DOT4_MMQ_V_F16,       false, kshared, directv, raw_lds); break;
            default: GGML_ABORT("packed16_dot4_mmq gqa2: unsupported V type");
        }
    }
} else {
#define PDMQ_SWITCH_V(CAUSAL) \
    switch (V->type) { \
        case GGML_TYPE_Q4_0:      PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q4_0,      CAUSAL, kshared, directv, raw_lds); break; \
        case GGML_TYPE_Q8_0:      PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_Q8_0,      CAUSAL, kshared, directv, raw_lds); break; \
        case GGML_TYPE_F16:            PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V_F16,            CAUSAL, kshared, directv, raw_lds); break; \
        case GGML_TYPE_V4_K16D16_144:  PDMQ_LAUNCH_TYPED(PACKED16_DOT4_MMQ_V4_K16D16_144,    CAUSAL, kshared, directv, raw_lds); break; \
        default: GGML_ABORT("packed16_dot4_mmq: unsupported V type"); \
    }

    if (assume_causal) {
        PDMQ_SWITCH_V(true);
    } else {
        PDMQ_SWITCH_V(false);
    }

#undef PDMQ_SWITCH_V
}
#else
{
    // Qwen35 investigation build: compile only the cells used by the current
    // failing boundary artifact instead of expanding the full compact matrix.
    // Required cells:
    //   - GQA6 split-K, V=q4_0, raw_lds_q4, M2N32 (qblock nq=2/4)
    //   - GQA6 split-K, V=q4_0, raw_lds_q4, M1N32 (row-serial decode oracle)
    //   - GQA6 non-split, V=q4_0, raw_lds_q4, M1N32 (decode fallback/oracle)
    //   - GQA1 non-split, V=q4_0, raw_lds_q4, M1/M2/M8N32 and M16N16 (warmup / normal prefill)
#if PDMQ_COMPILE_V4_ANY
    const bool debug_v4_144_cell = PDMQ_COMPILE_V4_144 && V->type == GGML_TYPE_V4_K16D16_144 && v4_k16d16_144_persistent && !raw_lds && !kshared;
    const bool debug_v4_cell = debug_v4_144_cell;
#else
    const bool debug_v4_144_cell = false;
    const bool debug_v4_cell = false;
#endif
    if (!debug_v4_cell && (V->type != GGML_TYPE_Q4_0 || !raw_lds_q4 || kshared)) {
        GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY supports only V=q4_0 raw_lds_q4 kshared=0 or scalar persistent V4 K16D16 GQA1/GQA6; variant=%s V=%s raw_lds_q4=%d kshared=%d vpath=%s",
                   pdmq_variant_name, ggml_type_name(V->type), raw_lds_q4 ? 1 : 0, kshared ? 1 : 0, pdmq_v_path_name(plan.v_path));
    }
    if (pdmq_pv_wmma_active || pdmq_pv_i8_wmma_active || pdmq_pv_i4_wmma_active || pdmq_pv_i4_cache_active) {
        GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY disables PV-WMMA/PV-I8/PV-I4/i4-cache debug cells");
    }

    const bool debug_gqa1_cell = !is_gqa2 && !is_gqa2_xqa && !is_gqa4 && !is_gqa6 && !is_gqax_splitk;
    if (debug_gqa1_cell) {
        if (shape != PDMQ_SHAPE_M1N32 && shape != PDMQ_SHAPE_M2N32 && shape != PDMQ_SHAPE_M8N32 && shape != PDMQ_SHAPE_M16N16) {
            GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY GQA1 supports only M1N32/M2N32/M8N32/M16N16, got shape=%s", pdmq_shape_name(shape));
        }
#define PDMQ_LAUNCH_DEBUG_GQA1(VT, CAUSAL, RAW_Q4) do { \
            if (shape == PDMQ_SHAPE_M1N32) { \
                PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, CAUSAL, false, RAW_Q4, false); \
            } else if (shape == PDMQ_SHAPE_M2N32) { \
                PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, CAUSAL, false, RAW_Q4, false); \
            } else if (shape == PDMQ_SHAPE_M8N32) { \
                PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_SMALL, PDMQ_BN_SMALL, CAUSAL, false, RAW_Q4, false); \
            } else { \
                PDMQ_LAUNCH_SHAPE(VT, PDMQ_BM_PREFILL, PDMQ_BN_PREFILL, CAUSAL, false, RAW_Q4, false); \
            } \
        } while (0)
#if PDMQ_COMPILE_V4_ANY
        if (debug_v4_144_cell) {
            if (assume_causal) {
                PDMQ_LAUNCH_DEBUG_GQA1(PACKED16_DOT4_MMQ_V4_K16D16_144, true, false);
            } else {
                PDMQ_LAUNCH_DEBUG_GQA1(PACKED16_DOT4_MMQ_V4_K16D16_144, false, false);
            }
        } else
#endif
        {
            if (assume_causal) {
                PDMQ_LAUNCH_DEBUG_GQA1(PACKED16_DOT4_MMQ_V_Q4_0, true, true);
            } else {
                PDMQ_LAUNCH_DEBUG_GQA1(PACKED16_DOT4_MMQ_V_Q4_0, false, true);
            }
        }
#undef PDMQ_LAUNCH_DEBUG_GQA1
    } else if (is_gqa1_splitk) {
        GGML_ASSERT(V->type == GGML_TYPE_Q4_0 && raw_lds_q4 && !kshared);
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        dp16_fa_q_workspace q_workspace(pool);
        if (dp16_qpack_stage_active) {
            q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
            q_view.qblock_program = qblock_program;
        }
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
        if (assume_causal) {
            PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(1, PACKED16_DOT4_MMQ_V_Q4_0, true, partial_m.get(), partial_l.get(), partial_out.get());
        } else {
            PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE(1, PACKED16_DOT4_MMQ_V_Q4_0, false, partial_m.get(), partial_l.get(), partial_out.get());
        }
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (is_gqa4_splitk) {
        if (shape != PDMQ_SHAPE_M1N32 && shape != PDMQ_SHAPE_M2N32 && shape != PDMQ_SHAPE_M4N32 && shape != PDMQ_SHAPE_M8N32) {
            GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY GQA4 split-K supports only M1/M2/M4/M8N32, got shape=%s", pdmq_shape_name(shape));
        }
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        dp16_fa_q_workspace q_workspace(pool);
        if (dp16_qpack_stage_active) {
            q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
            q_view.qblock_program = qblock_program;
        }
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
        {
            GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY GQA4 split-K is enabled only for persistent V4_144 PV-WMMA candidates");
        }
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (is_gqa6_splitk) {
        if (shape != PDMQ_SHAPE_M1N32 && shape != PDMQ_SHAPE_M2N32 && shape != PDMQ_SHAPE_M4N32) {
            GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY split-K supports only M1N32/M2N32/M4N32, got shape=%s", pdmq_shape_name(shape));
        }
        const size_t partial_rows = size_t(gqax_splitk_active) * size_t(batch) * size_t(nq) * size_t(n_heads_q);
        ggml_cuda_pool & pool = ctx.pool();
        dp16_fa_q_workspace q_workspace(pool);
        if (dp16_qpack_stage_active) {
            q_view = dp16_fa_prepare_q_stage<PDMQ_D, PDMQ_THREADS>(stream, q_stage, Q, q_workspace);
            q_view.qblock_program = qblock_program;
        }
        ggml_cuda_pool_alloc<float> partial_m(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_l(pool, partial_rows);
        ggml_cuda_pool_alloc<float> partial_out(pool, partial_rows * size_t(PDMQ_D));
        if (shape == PDMQ_SHAPE_M1N32) {
            if (assume_causal) {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, true, true, partial_m.get(), partial_l.get(), partial_out.get());
            } else {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_DECODE32, PDMQ_BN_DECODE32, false, true, partial_m.get(), partial_l.get(), partial_out.get());
            }
        } else if (shape == PDMQ_SHAPE_M2N32) {
            if (assume_causal) {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, true, true, partial_m.get(), partial_l.get(), partial_out.get());
            } else {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_VERIFY2_32, PDMQ_BN_VERIFY2_32, false, true, partial_m.get(), partial_l.get(), partial_out.get());
            }
        } else {
            if (assume_causal) {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, true, true, partial_m.get(), partial_l.get(), partial_out.get());
            } else {
                PDMQ_LAUNCH_GQAX_SPLITK_SHAPE(6, PACKED16_DOT4_MMQ_V_Q4_0, PDMQ_BM_VERIFY4_32, PDMQ_BN_VERIFY4_32, false, true, partial_m.get(), partial_l.get(), partial_out.get());
            }
        }
        const size_t merge_elems = size_t(batch) * size_t(nq) * size_t(n_heads_q) * size_t(PDMQ_D);
        const int merge_blocks = (int) ((merge_elems + size_t(PDMQ_THREADS) - 1) / size_t(PDMQ_THREADS));
        packed16_dot4_mmq_gqax_splitk_merge_kernel<<<merge_blocks, PDMQ_THREADS, 0, stream>>>(
            partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
            nq, nk, n_heads_q, batch, gqax_splitk_active);
    } else if (is_gqa6) {
        if (shape != PDMQ_SHAPE_M1N32) {
            GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY GQA6 non-split supports only M1N32, got shape=%s", pdmq_shape_name(shape));
        }
#if PDMQ_COMPILE_V4_ANY
        if (debug_v4_144_cell) {
            if (assume_causal) {
                PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V4_K16D16_144, true, false, false);
            } else {
                PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V4_K16D16_144, false, false, false);
            }
        } else
#endif
        if (assume_causal) {
            PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V_Q4_0, true, kshared, raw_lds_q4);
        } else {
            PDMQ_LAUNCH_GQA6(PACKED16_DOT4_MMQ_V_Q4_0, false, kshared, raw_lds_q4);
        }
    } else {
        GGML_ABORT("PDMQ QWEN35_DEBUG_ONLY supports only GQA1 warmup, V4 GQA4 split-K candidates, or GQA6 qwen35 cells; variant=%s V=%s vpath=%s", pdmq_variant_name, ggml_type_name(V->type), pdmq_v_path_name(plan.v_path));
    }
}
#endif
#undef PDMQ_LAUNCH_TYPED_NO_RAW
#undef PDMQ_LAUNCH_TYPED
#undef PDMQ_LAUNCH_GQA2_NO_RAW
#undef PDMQ_LAUNCH_GQA2
#undef PDMQ_LAUNCH_GQA2_IMPL
#undef PDMQ_LAUNCH_GQA2_SHAPE
#undef PDMQ_LAUNCH_GQA2X
#undef PDMQ_LAUNCH_GQA2X_IMPL
#undef PDMQ_LAUNCH_GQA6
#undef PDMQ_LAUNCH_GQA6_IMPL
#undef PDMQ_LAUNCH_GQA4
#undef PDMQ_LAUNCH_GQA4_IMPL
#undef PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA_ROWS16
#undef PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE_PVWMMA
#undef PDMQ_LAUNCH_GQAX_SPLITK_SHAPE_PVWMMA
#undef PDMQ_LAUNCH_GQAX_SPLITK_BY_SHAPE
#undef PDMQ_LAUNCH_GQAX_SPLITK_SHAPE
#undef PDMQ_LAUNCH_GQAX_SHAPE
#undef PDMQ_LAUNCH_SHAPE
#undef PDMQ_COMPACT_MATRIX_ABORT

    CUDA_CHECK(hipGetLastError());
}

#else

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    return false;
}

static inline bool ggml_cuda_packed16_dot4_mmq_v_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_V4_K16D16 ||
           type == GGML_TYPE_V4_K16D16_144;
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
