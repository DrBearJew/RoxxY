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
#include "fattn-dot4-q8k-kq.cuh"
#include "fattn-packed16-wmma-builtin.cuh"
#include "dot4-packed16/dp16-packed-i8-desc.cuh"
#include "dot4-packed16/dp16-fa-qpack.cuh"
#include "dot4-packed16/fa-block-meta.cuh"

#include <atomic>
#include <string>
#include <vector>

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data,
                                          ggml_tensor ** payload,
                                          ggml_tensor ** scales);
void llama_kv_cache_get_packed16_packed_i8_desc(const void * k_view_data,
                                                 dp16_packed_i8_desc_v1 * desc);
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PWMMA_D  = 256;
static constexpr int PWMMA_BN = 16;
static constexpr int PWMMA_IMPLICIT_FLAG_BLOCK_META_SKIP_CLAMP = 1 << 29;
static constexpr int PWMMA_IMPLICIT_FLAG_BLOCK_META_FULL_TILE  = 1 << 30;
static constexpr int PWMMA_I8_KSHARED_I32_COUNT    = PWMMA_BN * (PWMMA_D/4);
static constexpr int PWMMA_I8_KSHARED_SCALE_COUNT  = PWMMA_BN * (PWMMA_D/QK8_0);
static constexpr int PWMMA_I8_KSHARED_SMEM_BYTES   = PWMMA_I8_KSHARED_I32_COUNT * (int) sizeof(int) +
                                                      PWMMA_I8_KSHARED_SCALE_COUNT * (int) sizeof(half);

struct pwmma_dbv_page_map_v1 {
    uint32_t active = 0;
    uint32_t logical_base_token = 0;
    uint32_t valid_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    uint32_t non_identity_page_begin = 0;
    uint32_t non_identity_page_end = 0;
    uint32_t flags = 0;
    const int32_t * block_table = nullptr;
};

static __host__ __device__ __forceinline__ int pwmma_dbv_page_map_physical_k(
        const pwmma_dbv_page_map_v1 & map,
        const int logical_k,
        const int physical_capacity_tokens) {
    if (!map.active) {
        return logical_k;
    }
    if (map.page_tokens == 0 || map.valid_tokens == 0 || map.block_table_pages == 0 ||
            map.block_table == nullptr || map.physical_pages == 0 || logical_k < (int) map.logical_base_token) {
        return -1;
    }
    const uint32_t rel = (uint32_t) logical_k - map.logical_base_token;
    if (rel >= map.valid_tokens) {
        return logical_k;
    }
    const uint32_t logical_page = rel / map.page_tokens;
    if (logical_page >= map.block_table_pages) {
        return -1;
    }
    if ((map.flags & GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_FLAG_IDENTITY) != 0 &&
            (logical_page < map.non_identity_page_begin || logical_page >= map.non_identity_page_end)) {
        return logical_k;
    }
    const int32_t physical_page = map.block_table[logical_page];
    if (physical_page < 0 || (uint32_t) physical_page >= map.physical_pages) {
        return -1;
    }
    const uint32_t slot = rel - logical_page * map.page_tokens;
    const uint64_t physical = uint64_t((uint32_t) physical_page) * uint64_t(map.page_tokens) + uint64_t(slot);
    if (physical >= uint64_t(physical_capacity_tokens)) {
        return -1;
    }
    return (int) physical;
}

static __host__ __device__ __forceinline__ int pwmma_dbv_page_map_physical_k_tile_base(
        const pwmma_dbv_page_map_v1 & map,
        const int logical_k0,
        const int valid_k,
        const int physical_capacity_tokens) {
    if (!map.active) {
        return logical_k0;
    }
    if (valid_k <= 0 || map.page_tokens == 0 || map.valid_tokens == 0 || map.block_table_pages == 0 ||
            map.block_table == nullptr || map.physical_pages == 0 || logical_k0 < (int) map.logical_base_token) {
        return -1;
    }
    const uint32_t rel = (uint32_t) logical_k0 - map.logical_base_token;
    if (rel >= map.valid_tokens || uint64_t(rel) + uint64_t(valid_k) > uint64_t(map.valid_tokens)) {
        return -1;
    }
    if (map.page_tokens != PWMMA_BN || (rel % map.page_tokens) != 0 || (uint32_t) valid_k > map.page_tokens) {
        return -1;
    }
    const uint32_t logical_page = rel / map.page_tokens;
    if (logical_page >= map.block_table_pages) {
        return -1;
    }
    if ((map.flags & GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_FLAG_IDENTITY) != 0 &&
            (logical_page < map.non_identity_page_begin || logical_page >= map.non_identity_page_end)) {
        return logical_k0;
    }
    const int32_t physical_page = map.block_table[logical_page];
    if (physical_page < 0 || (uint32_t) physical_page >= map.physical_pages) {
        return -1;
    }
    const uint64_t physical = uint64_t((uint32_t) physical_page) * uint64_t(map.page_tokens);
    if (physical + uint64_t(valid_k) > uint64_t(physical_capacity_tokens)) {
        return -1;
    }
    return (int) physical;
}

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

static inline void ggml_cuda_pwmma_self_probes_fail_closed_if_requested() {
    const char * v = getenv("GGML_CUDA_PWMMA_SELF_PROBES");
    if (!v) {
        v = getenv("GGML_CUDA_PWMMA_DEBUG_SELF_PROBES");
    }
    if (v && atoi(v) != 0) {
        GGML_ABORT("GGML_CUDA_PWMMA_SELF_PROBES was archived out-of-tree by 2026-06-23 build-trim");
    }
}

static inline bool ggml_cuda_pwmma_log_enabled() {
    const char * v = getenv("COMPRESSED_KV_FATTN_LOG");
    return v && atoi(v) != 0;
}

static inline bool ggml_cuda_pwmma_consumer_read_mode_trace_enabled() {
    const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_CONSUMER_READ_MODE_TRACE");
    if (v && atoi(v) != 0) {
        return true;
    }
    v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_CONSUMER_READ_MODE_TRACE");
    return v && atoi(v) != 0;
}

static int ggml_cuda_rocm_packed16_wmma_streamk_splits() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_STREAMK_SPLITS");
    const int splits = (!s || !*s) ? 2 : atoi(s);
    if (splits >= 2 && splits <= 64) {
        return splits;
    }
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_WMMA_STREAMK_SPLITS=%s; expected 2..64", s ? s : "");
}

static inline bool ggml_cuda_rocm_packed16_fa_debug_qk_enabled() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_QK");
    return s && atoi(s) != 0;
}

static inline int ggml_cuda_rocm_env_i32(const char * name, int def) {
    const char * s = getenv(name);
    return s && *s ? atoi(s) : def;
}

struct ggml_cuda_rocm_packed16_wmma_timing_event {
    bool active = false;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

static inline void ggml_cuda_rocm_packed16_wmma_timing_begin(
        ggml_cuda_rocm_packed16_wmma_timing_event & ev,
        cudaStream_t stream,
        const char * phase) {
    ev = {};
    if (ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_TIMING_TRACE", 0) == 0) {
        return;
    }
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture_status) == hipSuccess && capture_status != hipStreamCaptureStatusNone) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "PACKED16_TIMING_TRACE skipped during graph capture phase=%s\n", phase ? phase : "-");
        }
        return;
    }
    CUDA_CHECK(hipEventCreate(&ev.start));
    CUDA_CHECK(hipEventCreate(&ev.stop));
    CUDA_CHECK(hipEventRecord(ev.start, stream));
    ev.active = true;
}

static inline float ggml_cuda_rocm_packed16_wmma_timing_end(
        ggml_cuda_rocm_packed16_wmma_timing_event & ev,
        cudaStream_t stream) {
    if (!ev.active) {
        return -1.0f;
    }
    CUDA_CHECK(hipEventRecord(ev.stop, stream));
    CUDA_CHECK(hipEventSynchronize(ev.stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, ev.start, ev.stop));
    CUDA_CHECK(hipEventDestroy(ev.start));
    CUDA_CHECK(hipEventDestroy(ev.stop));
    ev = {};
    return elapsed_ms;
}

static inline int ggml_cuda_rocm_packed16_debug_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    const char * dash = strrchr(name, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long v = strtol(dash + 1, &end, 10);
    return end && *end == '\0' ? (int) v : -1;
}

static inline bool ggml_cuda_rocm_packed16_debug_layer_matches(const ggml_tensor * dst) {
    const int want = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_LAYER", -1);
    if (want < 0) {
        return true;
    }
    return ggml_cuda_rocm_packed16_debug_layer_from_name(dst ? dst->name : nullptr) == want;
}

static inline uint64_t ggml_cuda_rocm_packed16_debug_fnv1a64(const float * data, size_t n) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(data);
    const size_t nbytes = n * sizeof(float);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t) bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline void ggml_cuda_rocm_packed16_debug_write_f32(
        const std::string & path,
        const float * data,
        size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "PWMMA DEBUG QK: failed to open %s\n", path.c_str());
        return;
    }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
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
    if (strcmp(s, "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev") == 0) return 17;
    if (strcmp(s, "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev") == 0) return 18;
    if (strcmp(s, "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev") == 0) return 19;
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=%s", s);
}

enum packed16_wmma_v_type {
    PACKED16_WMMA_V_Q4_0,
    PACKED16_WMMA_V_Q8_0,
    PACKED16_WMMA_V_F16,
    PACKED16_WMMA_V4_K16D16_144,
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

static __device__ __forceinline__ bool pwmma_dbv_q_row_info(
        const int r, const int rows_per_head, const int dbv_gqa_group,
        const int hq_base, const int hk, const int gqa_ratio,
        const int n_heads_q, const int nq, const int q0,
        int & qq, int & hq_row) {
    const int slot = r / rows_per_head;
    const int local_r = r - slot * rows_per_head;
    hq_row = hq_base + slot;
    qq = q0 + local_r;
    return slot < dbv_gqa_group && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio && qq < nq;
}

static __device__ __forceinline__ float pwmma_dbv_q_load_or_zero(
        const float * __restrict__ Q,
        const int64_t q_nb01, const int64_t q_nb02, const int64_t q_nb03,
        const int r, const int d,
        const int rows_per_head, const int dbv_gqa_group,
        const int hq_base, const int hk, const int gqa_ratio,
        const int n_heads_q, const int nq, const int q0, const int b) {
    int qq = 0;
    int hq_row = 0;
    if (!pwmma_dbv_q_row_info(r, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, qq, hq_row)) {
        return 0.0f;
    }
    const int64_t s0 = q_nb01 / (int64_t) sizeof(float);
    const int64_t s1 = q_nb02 / (int64_t) sizeof(float);
    const int64_t s2 = q_nb03 / (int64_t) sizeof(float);
    return Q[(int64_t) qq * s0 + (int64_t) hq_row * s1 + (int64_t) b * s2 + d];
}

static __device__ __forceinline__ float pwmma_dbv_q_scale32_recompute(
        const float * __restrict__ Q,
        const int64_t q_nb01, const int64_t q_nb02, const int64_t q_nb03,
        const int r, const int d0,
        const int rows_per_head, const int dbv_gqa_group,
        const int hq_base, const int hk, const int gqa_ratio,
        const int n_heads_q, const int nq, const int q0, const int b) {
    const int d_base = (d0 / 32) * 32;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(pwmma_dbv_q_load_or_zero(Q, q_nb01, q_nb02, q_nb03,
            r, d_base + i, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b)));
    }
    return amax > 0.0f ? amax / 127.0f : 1.0f;
}

static __device__ __forceinline__ int pwmma_dbv_q_pack_i8x4_recompute(
        const float * __restrict__ Q,
        const int64_t q_nb01, const int64_t q_nb02, const int64_t q_nb03,
        const int r, const int d_word, const float q_scale,
        const int rows_per_head, const int dbv_gqa_group,
        const int hq_base, const int hk, const int gqa_ratio,
        const int n_heads_q, const int nq, const int q0, const int b) {
    const int d_base = d_word * 4;
    return pwmma_pack_i8x4(
        pwmma_i8_clamp_round(pwmma_dbv_q_load_or_zero(Q, q_nb01, q_nb02, q_nb03,
            r, d_base + 0, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) / q_scale),
        pwmma_i8_clamp_round(pwmma_dbv_q_load_or_zero(Q, q_nb01, q_nb02, q_nb03,
            r, d_base + 1, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) / q_scale),
        pwmma_i8_clamp_round(pwmma_dbv_q_load_or_zero(Q, q_nb01, q_nb02, q_nb03,
            r, d_base + 2, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) / q_scale),
        pwmma_i8_clamp_round(pwmma_dbv_q_load_or_zero(Q, q_nb01, q_nb02, q_nb03,
            r, d_base + 3, rows_per_head, dbv_gqa_group, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) / q_scale));
}

static __device__ __forceinline__ int pwmma_i8qk_k_word(
        const int * __restrict__ k_payload, const size_t row, const int row_stride_i32, const int d_word) {
    return k_payload[row * size_t(row_stride_i32) + size_t(d_word)];
}

static __device__ __forceinline__ half pwmma_i8qk_k_scale(
        const half * __restrict__ k_scales, const size_t row, const int row_stride_half, const int qblock) {
    return k_scales[row * size_t(row_stride_half) + size_t(qblock)];
}

static __device__ __forceinline__ void pwmma_i8qk_row_to_head_token(
        const size_t row, const int head_stride_rows, uint32_t & head, uint32_t & token) {
    head = head_stride_rows > 0 ? uint32_t(row / size_t(head_stride_rows)) : 0u;
    token = head_stride_rows > 0 ? uint32_t(row - size_t(head) * size_t(head_stride_rows)) : uint32_t(row);
}

static __device__ __forceinline__ bool pwmma_i8qk_desc_d16_vector_layout(
        const dp16_packed_i8_desc_v1 & k_desc) {
    return (k_desc.layout_kind == DP16_PACKED_I8_LAYOUT_D16_PLANAR ||
            k_desc.layout_kind == DP16_PACKED_I8_LAYOUT_PAGE16_D16) &&
        dp16_i8x16_desc_has_vector_abi(k_desc);
}

static __device__ __forceinline__ int pwmma_i8qk_k_word_desc(
        const int * __restrict__ k_payload,
        const dp16_packed_i8_desc_v1 & k_desc,
        const size_t row,
        const int row_stride_i32,
        const int head_stride_rows,
        const int d_word) {
    if (!pwmma_i8qk_desc_d16_vector_layout(k_desc)) {
        return pwmma_i8qk_k_word(k_payload, row, row_stride_i32, d_word);
    }
    uint32_t head, token;
    pwmma_i8qk_row_to_head_token(row, head_stride_rows, head, token);
    const uint32_t d16 = uint32_t(d_word / int(DP16_PACKED_I8X16_WORDS));
    const uint32_t word = uint32_t(d_word - int(d16) * int(DP16_PACKED_I8X16_WORDS));
    return k_payload[dp16_packed_i8_payload_word_index(k_desc, head, token, d16, word)];
}

static __device__ __forceinline__ half pwmma_i8qk_k_scale_desc(
        const half * __restrict__ k_scales,
        const dp16_packed_i8_desc_v1 & k_desc,
        const size_t row,
        const int row_stride_half,
        const int head_stride_rows,
        const int qblock) {
    if (!pwmma_i8qk_desc_d16_vector_layout(k_desc)) {
        return pwmma_i8qk_k_scale(k_scales, row, row_stride_half, qblock);
    }
    uint32_t head, token;
    pwmma_i8qk_row_to_head_token(row, head_stride_rows, head, token);
    return k_scales[dp16_packed_i8_scale_byte_offset(k_desc, head, token, uint32_t(qblock)) / sizeof(half)];
}

static __device__ __forceinline__ int4 pwmma_i8qk_k_d16x4_desc(
        const int * __restrict__ k_payload,
        const dp16_packed_i8_desc_v1 & k_desc,
        const size_t row,
        const int row_stride_i32,
        const int head_stride_rows,
        const int d16) {
    const int d_word = d16 * int(DP16_PACKED_I8X16_WORDS);
    if (!pwmma_i8qk_desc_d16_vector_layout(k_desc)) {
        return *((const int4 *) (k_payload + row * size_t(row_stride_i32) + size_t(d_word)));
    }
    uint32_t head, token;
    pwmma_i8qk_row_to_head_token(row, head_stride_rows, head, token);
    const size_t off = dp16_packed_i8_payload_word_index(k_desc, head, token, uint32_t(d16), 0u);
    return *((const int4 *) (k_payload + off));
}

static __device__ __forceinline__ int pwmma_i8qk_packed8_word_as_i8x4(
        const int * __restrict__ k_payload, const size_t row, const int row_stride_i32, const int d_word) {
    const int d_base = d_word * 4;
    uint32_t packed_i8 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dd = d_base + i;
        const uint32_t src = static_cast<uint32_t>(k_payload[row * size_t(row_stride_i32) + size_t(dd >> 3)]);
        const int code = int((src >> (4 * (dd & 7))) & 0x0fu);
        const int8_t q = static_cast<int8_t>(code - 8);
        packed_i8 |= uint32_t(uint8_t(q)) << (8 * i);
    }
    return static_cast<int>(packed_i8);
}

static __device__ __forceinline__ void pwmma_i8qk_packed8_word_as_i8x4_pair(
        const int * __restrict__ k_payload, const size_t row, const int row_stride_i32, const int src_word,
        int & lo_i8x4, int & hi_i8x4) {
    const uint32_t src = static_cast<uint32_t>(k_payload[row * size_t(row_stride_i32) + size_t(src_word)]);
    uint32_t lo = 0;
    uint32_t hi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int code = int((src >> (4 * i)) & 0x0fu);
        const int8_t q = static_cast<int8_t>(code - 8);
        lo |= uint32_t(uint8_t(q)) << (8 * i);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int code = int((src >> (4 * (i + 4))) & 0x0fu);
        const int8_t q = static_cast<int8_t>(code - 8);
        hi |= uint32_t(uint8_t(q)) << (8 * i);
    }
    lo_i8x4 = static_cast<int>(lo);
    hi_i8x4 = static_cast<int>(hi);
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

static constexpr int PWMMA_V4_K16D16_144_K = 16;
static constexpr int PWMMA_V4_K16D16_144_D32 = 32;
static constexpr int PWMMA_V4_K16D16_144_PAYLOAD_BYTES = 2048;
static constexpr int PWMMA_V4_K16D16_144_WORDS_PER_D = 2;

static __device__ __forceinline__ float pwmma_decode_v_v4_k16d16_144(
        const char * __restrict__ V, int64_t v_nb10, int64_t v_nb11, int64_t v_nb12,
        int64_t v_nb13, int64_t v_ne13, int k, int hk, int b, int d) {
    GGML_UNUSED(v_nb10);
    const int vb = v_ne13 > 1 ? (b % v_ne13) : 0;
    const int k16_base = k & ~(PWMMA_V4_K16D16_144_K - 1);
    const int slot = k & (PWMMA_V4_K16D16_144_K - 1);
    const char * block = V + int64_t(vb)*v_nb13 + int64_t(hk)*v_nb12 + int64_t(k16_base)*v_nb11;
    const uint32_t word = ((const uint32_t *) block)[d * PWMMA_V4_K16D16_144_WORDS_PER_D + (slot >> 3)];
    const int q = int((word >> (4 * (slot & 7))) & 0x0fu) - 8;
    const half * scales = (const half *) (block + PWMMA_V4_K16D16_144_PAYLOAD_BYTES);
    return float(q) * __half2float(scales[(d / PWMMA_V4_K16D16_144_D32) * PWMMA_V4_K16D16_144_K + slot]);
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

static __device__ __forceinline__ bool pwmma_mask_keep_and_bias(
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        int q, int k, int b,
        float * __restrict__ bias,
        int implicit_n_kv,
        int implicit_q_offset,
        int implicit_flags) {
    *bias = 0.0f;
    if ((implicit_flags & 1) != 0) {
        if (k >= implicit_n_kv || k > q + implicit_q_offset) {
            return false;
        }
    }
    if (!mask) {
        return true;
    }
    if (q < 0 || q >= mask_ne01 || k < 0 || k >= mask_ne00) {
        return false;
    }
    const int mb = mask_ne03 > 1 ? (b % mask_ne03) : 0;
    const half hv = *(const half *)(mask + int64_t(mb)*mask_nb03 + int64_t(q)*mask_nb01 + int64_t(k)*mask_nb00);
    const float v = __half2float(hv);
    if (!isfinite(v) || v <= -60000.0f) {
        return false;
    }
    *bias = v;
    return true;
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
    if constexpr (PWMMA_PROFILE_COMPILE_ENABLED) { \
        if (profile) { \
            __syncthreads(); \
            if (threadIdx.x == 0) { profile_t0 = clock64(); } \
            __syncthreads(); \
        } \
    } \
} while (0)

#define PWMMA_PROFILE_PHASE_END(FIELD) do { \
    if constexpr (PWMMA_PROFILE_COMPILE_ENABLED) { \
        if (profile) { \
            __syncthreads(); \
            if (threadIdx.x == 0) { \
                const unsigned long long profile_t1 = clock64(); \
                atomicAdd(&profile->FIELD, profile_t1 >= profile_t0 ? profile_t1 - profile_t0 : 0ull); \
            } \
            __syncthreads(); \
        } \
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
    float got, ref;
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
    } else if constexpr (V_TYPE == PACKED16_WMMA_V4_K16D16_144) {
        return pwmma_decode_v_v4_k16d16_144(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, k, hk, b, d);
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
            if (qq >= nq) {
                alpha_smem[r] = 0.0f;
                #pragma unroll
                for (int c = 0; c < PWMMA_BN; ++c) probs_f32[r][c] = 0.0f;
                continue;
            }
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
        const dp16_packed_i8_desc_v1 packed16_desc,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows,
        float attention_scale,
        unsigned long long * __restrict__ skip_counter,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err,
        pwmma_kernel_profile * __restrict__ profile) {

    GGML_UNUSED(causal_skip_enabled);
    constexpr bool PWMMA_PROFILE_COMPILE_ENABLED = true;
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
            for (int idx = threadIdx.x; idx < BN_TILE * (PWMMA_D/16); idx += blockDim.x) {
                const int c = idx / (PWMMA_D/16);
                const int d16 = idx - c * (PWMMA_D/16);
                const int g = d16 * 4;
                int4 kw4 = {0, 0, 0, 0};
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    kw4 = pwmma_i8qk_k_d16x4_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, d16);
                }
                int * dst = &k_i32_smem[c * (PWMMA_D/4) + g];
                dst[0] = kw4.x;
                dst[1] = kw4.y;
                dst[2] = kw4.z;
                dst[3] = kw4.w;
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_s_smem[idx] = pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, row, PWMMA_D/QK8_0, head_stride, s);
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
                                    b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, (sd0/4) + g);
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
                            ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k0) + size_t(k_col), PWMMA_D/QK8_0, head_stride, d0 / QK8_0));
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
                                b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, (d0/4) + g);
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
                            ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k0) + size_t(k_col), PWMMA_D/QK8_0, head_stride, d0 / QK8_0));
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
                if (wave_active[rb]) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        out_pv[rb][i] *= alpha_smem[global_r];
                    }
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
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                            float ref = 0.0f;
                            if (wave_active[rb]) {
                                #pragma unroll
                                for (int c = 0; c < 16; ++c) {
                                    const int kc = kc_base + c;
                                    if (kc < valid_k) ref += __half2float(probs[global_r][kc]) * __half2float(v_tile_f16[kc][d_tile * 16 + lane_lo]);
                                }
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
                                bounds_err->got = pv_acc[i];
                                bounds_err->ref = ref;
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
// ── BM64_I8QK PV-WMMA DBV row-layout lean restore ──
template<packed16_wmma_v_type V_TYPE, bool PACKED8_K = false, bool FORCE_K_SHARED = false,
         bool ENABLE_STREAMK = true, bool ENABLE_BOUNDS = true, bool ENABLE_PROFILE = true, bool ENABLE_DEBUG_QK = true,
         bool DOUBLE_BUFFER_V = true, bool PROBS_OVERLAY = false, bool DBV_GQA2 = false, bool SKIP_LOGITS_INIT = false>
static __global__ void packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_row_lean_kernel(
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
        pwmma_kernel_profile * __restrict__ profile,
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        float * __restrict__ partial_out,
        int streamk_splits,
        float * __restrict__ debug_qk_raw,
        float * __restrict__ debug_qk_masked,
        float * __restrict__ debug_probs,
        float * __restrict__ debug_m,
        float * __restrict__ debug_l,
        float * __restrict__ debug_alpha,
        int debug_qtile,
        int debug_hq,
        int debug_b,
        int debug_kt,
        int implicit_n_kv_arg,
        int implicit_q_offset_arg,
        int implicit_flags_arg) {

    GGML_UNUSED(causal_skip_enabled);
    constexpr bool PWMMA_PROFILE_COMPILE_ENABLED = ENABLE_PROFILE;
    const bool streamk_mode = ENABLE_STREAMK && partial_out != nullptr;
    const int split = streamk_mode ? (int(blockIdx.z) % streamk_splits) : 0;
    const int q_tile = blockIdx.x;
    const int gy = blockIdx.y;
    const int b = streamk_mode ? (int(blockIdx.z) / streamk_splits) : int(blockIdx.z);
    constexpr int DBV_GQA_GROUP = DBV_GQA2 ? 2 : 1;
    constexpr int ROWS_PER_HEAD = DBV_GQA2 ? (PWMMA_BM64 / 2) : PWMMA_BM64;
    const int groups_per_kv_dbv = DBV_GQA2 ? CEIL_DIV(gqa_ratio, DBV_GQA_GROUP) : 1;
    const int hk = DBV_GQA2 ? (gy / groups_per_kv_dbv) : (gy / gqa_ratio);
    const int local_group = DBV_GQA2 ? (gy - hk * groups_per_kv_dbv) : 0;
    const int hq_base = DBV_GQA2 ? (hk * gqa_ratio + local_group * DBV_GQA_GROUP) : gy;
    const int hq = hq_base;
    const int batch_eff = streamk_mode ? (int(gridDim.z) / streamk_splits) : int(gridDim.z);
    const size_t rows_per_split = size_t(batch_eff) * size_t(nq) * size_t(n_heads_q);
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * ROWS_PER_HEAD;
    constexpr bool PROBS_F16 = true;
    constexpr bool K32_ACC = false;
    constexpr bool K_SHARED = PACKED8_K || FORCE_K_SHARED;
    constexpr bool PV_WMMA = true;
    constexpr bool BN32 = false;
    constexpr int Q_SCALE_BLOCK = 32;
    constexpr int Q_SCALE_BLOCKS = PWMMA_D / Q_SCALE_BLOCK;
    constexpr int BN_TILE = PWMMA_BN;

    using prob_t = typename pwmma_prob_storage_type<PROBS_F16>::type;
    constexpr int V_BUFFER_COUNT = DOUBLE_BUFFER_V ? 2 : 1;
    constexpr int PROBS_STORAGE_ROWS = PROBS_OVERLAY ? 1 : PWMMA_BM64;
    __shared__ int q_i32[PWMMA_BM64][PWMMA_D/4];
    __shared__ float  q_s [PWMMA_BM64][Q_SCALE_BLOCKS];
    __shared__ half   v_tile_f16_db[V_BUFFER_COUNT][BN_TILE][PWMMA_D];
    __shared__ float logits_f32[PWMMA_BM64][BN_TILE];
    __shared__ prob_t probs [PROBS_STORAGE_ROWS][BN_TILE];
    prob_t (* const probs_overlay)[BN_TILE * 2] = reinterpret_cast<prob_t (*)[BN_TILE * 2]>(logits_f32);
#define PWMMA_DBV_PROB_SET(R, C, V) do { \
    if constexpr (PROBS_OVERLAY) { probs_overlay[(R)][(C)] = (V); } \
    else { probs[(R)][(C)] = (V); } \
} while (0)
#define PWMMA_DBV_PROB_GET(R, C) (PROBS_OVERLAY ? probs_overlay[(R)][(C)] : probs[(R)][(C)])
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
        const int slot = DBV_GQA2 ? (r / ROWS_PER_HEAD) : 0;
        const int local_r = DBV_GQA2 ? (r - slot * ROWS_PER_HEAD) : r;
        const int hq_row = hq_base + slot;
        const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
        const int qq = q0 + local_r;
        const int d_base = sb * Q_SCALE_BLOCK;
        float amax = 0.0f;
        float vals[Q_SCALE_BLOCK];
        #pragma unroll
        for (int i = 0; i < Q_SCALE_BLOCK; ++i) {
            const int dd = d_base + i;
            const float x = (slot_valid && qq < nq)
                ? Q[qq * (int)(q_nb01 / sizeof(float)) + hq_row * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd]
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
    const int kt_begin = streamk_mode ? (split * num_k_tiles) / streamk_splits : 0;
    const int kt_end   = streamk_mode ? ((split + 1) * num_k_tiles) / streamk_splits : num_k_tiles;
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const bool debug_tile = ENABLE_DEBUG_QK && debug_qk_raw != nullptr &&
            q_tile == debug_qtile && hq == debug_hq && b == debug_b && kt == debug_kt;
        const int k0 = kt * BN_TILE, valid_k = min(BN_TILE, nk - k0);
        const bool implicit_causal = (implicit_flags_arg & 1) != 0;
        const int implicit_n_kv = implicit_causal ? implicit_n_kv_arg : nk;
        const int q_offset = implicit_causal ? implicit_q_offset_arg : (nk - nq);
        const int mask_flags = implicit_causal ? implicit_flags_arg : 0;
        if (implicit_causal && k0 >= implicit_n_kv) {
            break;
        }

        if (threadIdx.x < 4) {
            const int w = threadIdx.x;
            const int slot = DBV_GQA2 ? ((w * 16) / ROWS_PER_HEAD) : 0;
            const int local_first = DBV_GQA2 ? ((w * 16) - slot * ROWS_PER_HEAD) : (w * 16);
            const int hq_wave = hq_base + slot;
            const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_wave < n_heads_q && hq_wave < (hk + 1) * gqa_ratio);
            const int wg_first = q0 + local_first;
            const int wg_last  = min(nq - 1, wg_first + 15);
            wave_active[w] = slot_valid && (wg_first < nq) && (k0 <= q_offset + wg_last);
        }
        __syncthreads();
        if (!wave_active[0] && !wave_active[1] && !wave_active[2] && !wave_active[3]) {
            if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
            continue;
        }

        const int vbuf = DOUBLE_BUFFER_V ? (kt & 1) : 0;
        for (int idx = threadIdx.x; idx < BN_TILE * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            v_tile_f16_db[vbuf][c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
        }
        if constexpr (K_SHARED) {
            if constexpr (PACKED8_K) {
                for (int idx = threadIdx.x; idx < BN_TILE * (PWMMA_D/8); idx += blockDim.x) {
                    const int c = idx / (PWMMA_D/8);
                    const int src_word = idx - c * (PWMMA_D/8);
                    int lo_i8x4 = 0;
                    int hi_i8x4 = 0;
                    if (c < valid_k) {
                        const size_t row = k_head_base + size_t(k0) + size_t(c);
                        pwmma_i8qk_packed8_word_as_i8x4_pair(k_payload, row, k_payload_row_stride_i32, src_word, lo_i8x4, hi_i8x4);
                    }
                    int * dst_i8x4 = &k_i32_smem[c * (PWMMA_D/4) + src_word * 2];
                    dst_i8x4[0] = lo_i8x4;
                    dst_i8x4[1] = hi_i8x4;
                }
            } else {
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
        if constexpr (!SKIP_LOGITS_INIT) {
            for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) logits_f32[idx / BN_TILE][idx % BN_TILE] = 0.0f;
        }
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
                            if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
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
                        if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
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
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile) {
            for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
                debug_qk_raw[idx] = logits_f32[idx / BN_TILE][idx % BN_TILE];
            }
            __syncthreads();
        }
        PWMMA_PROFILE_PHASE_END(qk_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
            const int r = idx / BN_TILE, c = idx % BN_TILE;
            const int slot = DBV_GQA2 ? (r / ROWS_PER_HEAD) : 0;
            const int local_r = DBV_GQA2 ? (r - slot * ROWS_PER_HEAD) : r;
            const int hq_row = hq_base + slot;
            const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            const int kk = k0 + c;
            float v;
            if (!slot_valid || qq >= nq || c >= valid_k || !wave_active[r >> 4]) {
                v = -FLT_MAX/2.0f;
            } else {
                v = logits_f32[r][c];
                if (mask || implicit_causal) {
                    float mask_bias = 0.0f;
                    if (!pwmma_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03,
                            qq, kk, b, &mask_bias, implicit_n_kv, q_offset, mask_flags)) {
                        v = -INFINITY;
                    } else {
                        v += mask_bias;
                    }
                }
            }
            logits_f32[r][c] = v;
        }
        __syncthreads();
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile && debug_qk_masked != nullptr) {
            for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
                debug_qk_masked[idx] = logits_f32[idx / BN_TILE][idx % BN_TILE];
            }
        }

        for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
            const int wave_id = r >> 4;
            const int slot = DBV_GQA2 ? (r / ROWS_PER_HEAD) : 0;
            const int local_r = DBV_GQA2 ? (r - slot * ROWS_PER_HEAD) : r;
            const int hq_row = hq_base + slot;
            const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            if (!slot_valid || qq >= nq || !wave_active[wave_id]) {
                alpha_smem[r] = 0.0f;
                #pragma unroll
                for (int c = 0; c < BN_TILE; ++c) {
                    if constexpr (PROBS_F16) PWMMA_DBV_PROB_SET(r, c, __float2half(0.0f));
                    else PWMMA_DBV_PROB_SET(r, c, 0.0f);
                }
                continue;
            }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) {
                const float p = expf(logits_f32[r][c] - new_m);
                if constexpr (PROBS_F16) PWMMA_DBV_PROB_SET(r, c, __float2half(p));
                else PWMMA_DBV_PROB_SET(r, c, p);
                p_sum += p;
            }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile) {
            for (int idx = threadIdx.x; idx < PWMMA_BM64 * BN_TILE; idx += blockDim.x) {
                float p;
                if constexpr (PROBS_F16) p = __half2float(PWMMA_DBV_PROB_GET(idx / BN_TILE, idx % BN_TILE));
                else p = PWMMA_DBV_PROB_GET(idx / BN_TILE, idx % BN_TILE);
                if (debug_probs != nullptr) debug_probs[idx] = p;
            }
            for (int r = threadIdx.x; r < PWMMA_BM64; r += blockDim.x) {
                if (debug_m != nullptr) debug_m[r] = row_m_smem[r];
                if (debug_l != nullptr) debug_l[r] = row_l_smem[r];
                if (debug_alpha != nullptr) debug_alpha[r] = alpha_smem[r];
            }
        }
        PWMMA_PROFILE_PHASE_END(softmax_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        if constexpr (PV_WMMA) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int d_tile  = (threadIdx.x >> 5) & 15;
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                if (wave_active[rb]) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        out_pv[rb][i] *= alpha_smem[global_r];
                    }
                }
            }
            #pragma unroll
            for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                pbwmma_v16fp16 b_frag;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    const int kc = kc_base + kk;
                    b_frag[kk] = kc < valid_k
                        ? v_tile_f16_db[vbuf][kc][d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo)]
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
                                if constexpr (PROBS_F16) a_frag[kk] = PWMMA_DBV_PROB_GET(rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo), kc);
                                else a_frag[kk] = __float2half(PWMMA_DBV_PROB_GET(rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo), kc));
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
                        if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                            float ref = 0.0f;
                            if (wave_active[rb]) {
                                #pragma unroll
                                for (int c = 0; c < 16; ++c) {
                                    const int kc = kc_base + c;
                                    if (kc < valid_k) ref += __half2float(PWMMA_DBV_PROB_GET(global_r, kc)) * __half2float(v_tile_f16_db[vbuf][kc][d_tile * 16 + lane_lo]);
                                }
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
                                bounds_err->got = pv_acc[i];
                                bounds_err->ref = ref;
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
                            if constexpr (PROBS_F16) p = __half2float(PWMMA_DBV_PROB_GET(row_base + r, c));
                            else p = PWMMA_DBV_PROB_GET(row_base + r, c);
                            out[r] += p * __half2float(v_tile_f16_db[vbuf][c][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(pv_cycles);
    }

#undef PWMMA_DBV_PROB_GET
#undef PWMMA_DBV_PROB_SET

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
                const int global_r = rb * 16 + rr;
                const int slot = DBV_GQA2 ? (global_r / ROWS_PER_HEAD) : 0;
                const int local_r = DBV_GQA2 ? (global_r - slot * ROWS_PER_HEAD) : global_r;
                const int hq_out = hq_base + slot;
                const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_out < n_heads_q && hq_out < (hk + 1) * gqa_ratio);
                const int qq = q0 + local_r;
                if (!slot_valid || qq >= nq) continue;
                const float l = row_l_smem[global_r];
                if (ENABLE_STREAMK && streamk_mode) {
                    const size_t dst_row = (size_t(b) * size_t(nq) + size_t(qq)) * size_t(n_heads_q) + size_t(hq_out);
                    const size_t partial_row = size_t(split) * rows_per_split + dst_row;
                    if (dd == 0) {
                        partial_m[partial_row] = row_m_smem[global_r];
                        partial_l[partial_row] = l;
                    }
                    partial_out[partial_row * size_t(PWMMA_D) + size_t(dd)] = l > 0.0f ? out_pv[rb][i] : 0.0f;
                } else {
                    if (l <= 0.0f) continue;
                    dst[((size_t(b)*nq + qq)*n_heads_q + hq_out)*PWMMA_D + dd] = out_pv[rb][i] / l;
                }
            }
        }
    } else if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int global_r = row_base + r;
            const int slot = DBV_GQA2 ? (global_r / ROWS_PER_HEAD) : 0;
            const int local_r = DBV_GQA2 ? (global_r - slot * ROWS_PER_HEAD) : global_r;
            const int hq_out = hq_base + slot;
            const bool slot_valid = !DBV_GQA2 || (slot < DBV_GQA_GROUP && hq_out < n_heads_q && hq_out < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            if (!slot_valid || qq >= nq) continue;
            const float l = row_l_smem[global_r];
            if (ENABLE_STREAMK && streamk_mode) {
                const size_t dst_row = (size_t(b) * size_t(nq) + size_t(qq)) * size_t(n_heads_q) + size_t(hq_out);
                const size_t partial_row = size_t(split) * rows_per_split + dst_row;
                if (d == 0) {
                    partial_m[partial_row] = row_m_smem[global_r];
                    partial_l[partial_row] = l;
                }
                partial_out[partial_row * size_t(PWMMA_D) + size_t(d)] = l > 0.0f ? out[r] : 0.0f;
            } else {
                if (l <= 0.0f) continue;
                dst[((size_t(b)*nq + qq)*n_heads_q + hq_out)*PWMMA_D + d] = out[r] / l;
            }
        }
    }
}


template<packed16_wmma_v_type V_TYPE, bool PACKED8_K = false, bool FORCE_K_SHARED = false,
         bool ENABLE_STREAMK = true, bool ENABLE_BOUNDS = true, bool ENABLE_PROFILE = true, bool ENABLE_DEBUG_QK = true,
         bool DOUBLE_BUFFER_V = true, bool PROBS_OVERLAY = false, int DBV_GQA_GROUP = 1, int DBV_TOTAL_ROWS = PWMMA_BM64,
         bool SKIP_LOGITS_INIT = false, bool BLOCK_META_PLAN = false, bool RECOMPUTE_Q = false, bool USE_QPACK = false,
         bool BLOCK_META_SKIP_CLAMP = false, bool DIRECT_ROW_KV = false>
static __global__ void packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel(
        const float * __restrict__ Q,
        const int * __restrict__ qpack_payload,
        const float * __restrict__ qpack_scales,
        int qpack_payload_stride_i32,
        int qpack_scales_stride_f32,
        const char * __restrict__ V, float * __restrict__ dst,
        int64_t q_nb01, int64_t q_nb02, int64_t q_nb03,
        int64_t v_nb10, int64_t v_nb11, int64_t v_nb12, int64_t v_nb13, int64_t v_ne13,
        int v_layout,
        const char * __restrict__ mask,
        int64_t mask_ne00, int64_t mask_ne01, int64_t mask_ne03,
        int64_t mask_nb00, int64_t mask_nb01, int64_t mask_nb03,
        const int  * __restrict__ k_payload, const half * __restrict__ k_scales,
        const dp16_packed_i8_desc_v1 packed16_desc,
        int k_payload_row_stride_i32, int k_scales_row_stride_half,
        int nq, int nk, int n_heads_q, int n_heads_k, int gqa_ratio,
        int packed_rows,
        float attention_scale,
        const pwmma_dbv_page_map_v1 page_map,
        unsigned long long * __restrict__ skip_counter,
        unsigned long long * __restrict__ block_meta_effect_counts,
        bool causal_skip_enabled,
        pwmma_debug_error * __restrict__ bounds_err,
        pwmma_kernel_profile * __restrict__ profile,
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        float * __restrict__ partial_out,
        int streamk_splits,
        int streamk_prefix_tiles,
        float * __restrict__ debug_qk_raw,
        float * __restrict__ debug_qk_masked,
        float * __restrict__ debug_probs,
        float * __restrict__ debug_m,
        float * __restrict__ debug_l,
        float * __restrict__ debug_alpha,
        int debug_qtile,
        int debug_hq,
        int debug_b,
        int debug_kt,
        int implicit_n_kv_arg,
        int implicit_q_offset_arg,
        int implicit_flags_arg) {

    GGML_UNUSED(causal_skip_enabled);
    constexpr bool PWMMA_PROFILE_COMPILE_ENABLED = ENABLE_PROFILE;
    const bool streamk_mode = ENABLE_STREAMK && partial_out != nullptr;
    const int split = streamk_mode ? (int(blockIdx.z) % streamk_splits) : 0;
    const int q_tile = blockIdx.x;
    const int gy = blockIdx.y;
    const int b = streamk_mode ? (int(blockIdx.z) / streamk_splits) : int(blockIdx.z);
    constexpr int DBV_WAVES = CEIL_DIV(DBV_TOTAL_ROWS, 16);
    constexpr int ROWS_PER_HEAD = DBV_TOTAL_ROWS / DBV_GQA_GROUP;
    constexpr bool DBV_SINGLE_GROUP = DBV_GQA_GROUP == 1 && DBV_TOTAL_ROWS == PWMMA_BM64;
    static_assert(DBV_GQA_GROUP >= 1, "DBV_GQA_GROUP must be positive");
    static_assert(DBV_TOTAL_ROWS % DBV_GQA_GROUP == 0, "DBV_TOTAL_ROWS must divide by DBV_GQA_GROUP");
    static_assert((DBV_TOTAL_ROWS / DBV_GQA_GROUP) % 16 == 0, "DBV rows/head must be a 16-row WMMA wave multiple");
    static_assert(DBV_TOTAL_ROWS >= PWMMA_BM64 && (DBV_TOTAL_ROWS <= 96 || ((RECOMPUTE_Q || USE_QPACK) && DBV_TOTAL_ROWS <= 192)),
        "DBV_TOTAL_ROWS > 96 requires RECOMPUTE_Q or USE_QPACK min-LDS mode and is capped at 192");
    static_assert(DBV_WAVES <= ((RECOMPUTE_Q || USE_QPACK) ? 12 : 6), "DBV kernel wave_active exceeds supported wave count");
    const int groups_per_kv_dbv = CEIL_DIV(gqa_ratio, DBV_GQA_GROUP);
    const int hk = gy / groups_per_kv_dbv;
    const int local_group = gy - hk * groups_per_kv_dbv;
    const int hq_base = hk * gqa_ratio + local_group * DBV_GQA_GROUP;
    const int hq = hq_base;
    const int batch_eff = streamk_mode ? (int(gridDim.z) / streamk_splits) : int(gridDim.z);
    const size_t rows_per_split = size_t(batch_eff) * size_t(nq) * size_t(n_heads_q);
    const int head_stride = packed_rows / n_heads_k;
    const size_t k_head_base = size_t(hk) * size_t(head_stride);
    const int q0 = q_tile * ROWS_PER_HEAD;
    constexpr bool PROBS_F16 = true;
    constexpr bool K32_ACC = false;
    constexpr bool K_SHARED = PACKED8_K || FORCE_K_SHARED;
    constexpr bool PV_WMMA = true;
    constexpr bool BN32 = false;
    constexpr int Q_SCALE_BLOCK = 32;
    constexpr int Q_SCALE_BLOCKS = PWMMA_D / Q_SCALE_BLOCK;
    constexpr int BN_TILE = PWMMA_BN;

    using prob_t = typename pwmma_prob_storage_type<PROBS_F16>::type;
    constexpr bool STAGED_QPACK = USE_QPACK && DBV_TOTAL_ROWS == 192;
    constexpr int Q_STAGE_ROWS = STAGED_QPACK ? 96 : DBV_TOTAL_ROWS;
    constexpr int Q_STAGE_COUNT = STAGED_QPACK ? 2 : 1;
    constexpr int V_BUFFER_COUNT = DOUBLE_BUFFER_V ? 2 : 1;
    constexpr int PROBS_STORAGE_ROWS = PROBS_OVERLAY ? 1 : DBV_TOTAL_ROWS;
    constexpr int Q_STORAGE_ROWS = RECOMPUTE_Q ? 1 : (STAGED_QPACK ? Q_STAGE_ROWS : DBV_TOTAL_ROWS);
    __shared__ int q_i32[Q_STORAGE_ROWS][PWMMA_D/4];
    __shared__ float  q_s [Q_STORAGE_ROWS][Q_SCALE_BLOCKS];
    __shared__ half   v_tile_f16_db[V_BUFFER_COUNT][BN_TILE][PWMMA_D];
    __shared__ float logits_f32[DBV_TOTAL_ROWS][BN_TILE];
    __shared__ prob_t probs [PROBS_STORAGE_ROWS][BN_TILE];
    prob_t (* const probs_overlay)[BN_TILE * 2] = reinterpret_cast<prob_t (*)[BN_TILE * 2]>(logits_f32);
#define PWMMA_DBV_PROB_SET(R, C, V) do { \
    if constexpr (PROBS_OVERLAY) { probs_overlay[(R)][(C)] = (V); } \
    else { probs[(R)][(C)] = (V); } \
} while (0)
#define PWMMA_DBV_PROB_GET(R, C) (PROBS_OVERLAY ? probs_overlay[(R)][(C)] : probs[(R)][(C)])
    __shared__ float row_m_smem[DBV_TOTAL_ROWS], row_l_smem[DBV_TOTAL_ROWS], alpha_smem[DBV_TOTAL_ROWS];
    __shared__ bool  wave_active[DBV_WAVES];
    __shared__ int   k_phys_tile[DIRECT_ROW_KV ? 1 : BN_TILE];
    __shared__ int   k_phys_tile_base;
    __shared__ unsigned long long profile_t0;
    extern __shared__ int pbwmma_i8_kshared_i32[];
    int  * const k_i32_smem = pbwmma_i8_kshared_i32;
    half * const k_s_smem   = reinterpret_cast<half *>(k_i32_smem + PWMMA_I8_KSHARED_I32_COUNT);

    if (threadIdx.x < DBV_TOTAL_ROWS) { row_m_smem[threadIdx.x] = -FLT_MAX/2.0f; row_l_smem[threadIdx.x] = 0.0f; }

    // Quantize Q once for normal DBV. 192-row min-LDS candidates recompute the
    // per-row Q fragments/scales on demand to avoid a full 192 x 256 i8 slab in LDS.
    if constexpr (!RECOMPUTE_Q && !STAGED_QPACK) {
        for (int idx = threadIdx.x; idx < DBV_TOTAL_ROWS * Q_SCALE_BLOCKS; idx += blockDim.x) {
            const int r = idx / Q_SCALE_BLOCKS;
            const int sb = idx % Q_SCALE_BLOCKS;
            const int slot = DBV_SINGLE_GROUP ? 0 : (r / ROWS_PER_HEAD);
            const int local_r = DBV_SINGLE_GROUP ? r : (r - slot * ROWS_PER_HEAD);
            const int hq_row = hq_base + slot;
            const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            const int d_base = sb * Q_SCALE_BLOCK;
            float amax = 0.0f;
            float vals[Q_SCALE_BLOCK];
            #pragma unroll
            for (int i = 0; i < Q_SCALE_BLOCK; ++i) {
                const int dd = d_base + i;
                const float x = (slot_valid && qq < nq)
                    ? Q[qq * (int)(q_nb01 / sizeof(float)) + hq_row * (int)(q_nb02 / sizeof(float)) + b * (int)(q_nb03 / sizeof(float)) + dd]
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
    }
    __syncthreads();

    const int pv_group = threadIdx.x >> 8;
    const int d        = threadIdx.x & 255;
    const int row_base = pv_group * 32;
    float out[32];
    if (d < PWMMA_D) { for (int r = 0; r < 32; ++r) out[r] = 0.0f; }
    float out_pv[DBV_WAVES][8];
    if constexpr (PV_WMMA) {
        #pragma unroll
        for (int rb = 0; rb < DBV_WAVES; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_pv[rb][i] = 0.0f;
        }
    }
    __syncthreads();

    const int num_k_tiles = CEIL_DIV(nk, BN_TILE);
    const bool streamk_prefix_split = streamk_mode && streamk_splits == 2 && streamk_prefix_tiles > 0 && streamk_prefix_tiles < num_k_tiles;
    const int kt_begin = streamk_prefix_split ? (split == 0 ? 0 : streamk_prefix_tiles) :
        (streamk_mode ? (split * num_k_tiles) / streamk_splits : 0);
    const int kt_end = streamk_prefix_split ? (split == 0 ? streamk_prefix_tiles : num_k_tiles) :
        (streamk_mode ? ((split + 1) * num_k_tiles) / streamk_splits : num_k_tiles);
    int kt_loop_end = kt_end;
    uint32_t block_meta_planned_skipped_tiles = 0;
    bool block_meta_q_plan_valid = false;
    uint32_t block_meta_full_begin = 0;
    uint32_t block_meta_full_end = 0;
    uint32_t block_meta_skipped_begin = 0;
    uint32_t block_meta_skipped_end = 0;
    constexpr bool BLOCK_META_ANY = BLOCK_META_PLAN || BLOCK_META_SKIP_CLAMP;
    const bool block_meta_plan_enabled = BLOCK_META_ANY &&
        (implicit_flags_arg & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0 &&
        mask == nullptr;
    if constexpr (BLOCK_META_ANY) {
        if (block_meta_plan_enabled) {
            const ggml_cuda_fa_block_meta_v1 block_meta = ggml_cuda_fa_block_meta_make_legacy_causal(
                implicit_n_kv_arg > 0 ? (uint32_t) implicit_n_kv_arg : 0u,
                nq > 0 ? (uint32_t) nq : 0u,
                implicit_q_offset_arg,
                (uint32_t) (implicit_flags_arg & ~(PWMMA_IMPLICIT_FLAG_BLOCK_META_FULL_TILE | PWMMA_IMPLICIT_FLAG_BLOCK_META_SKIP_CLAMP)),
                (uint32_t) ROWS_PER_HEAD,
                (uint32_t) BN_TILE);
            ggml_cuda_fa_block_plan_v1 q_plan = {};
            block_meta_q_plan_valid = ggml_cuda_fa_block_meta_plan_pure_causal(
                block_meta, (uint32_t) q0, (uint32_t) ROWS_PER_HEAD, &q_plan) == GGML_CUDA_FA_BLOCK_META_OK;
            if (block_meta_q_plan_valid) {
                block_meta_full_begin = q_plan.full_begin;
                block_meta_full_end = q_plan.full_end;
                block_meta_skipped_begin = q_plan.skipped_begin;
                block_meta_skipped_end = q_plan.skipped_end;
                if constexpr (BLOCK_META_SKIP_CLAMP) {
                    uint32_t clipped_end = (uint32_t) kt_end;
                    block_meta_planned_skipped_tiles = ggml_cuda_fa_block_meta_clip_k_range_to_visible(
                        q_plan, (uint32_t) kt_begin, (uint32_t) kt_end, &clipped_end);
                    kt_loop_end = (int) clipped_end;
                }
            }
        }
    }
    if constexpr (BLOCK_META_ANY) {
        if (block_meta_planned_skipped_tiles != 0 && threadIdx.x == 0) {
            if (skip_counter) {
                atomicAdd(skip_counter, (unsigned long long) block_meta_planned_skipped_tiles);
            }
            if (block_meta_effect_counts) {
                atomicAdd(&block_meta_effect_counts[3], (unsigned long long) block_meta_planned_skipped_tiles);
            }
        }
    }
    for (int kt = kt_begin; kt < kt_loop_end; ++kt) {
        const bool debug_tile = ENABLE_DEBUG_QK && debug_qk_raw != nullptr &&
            q_tile == debug_qtile && hq == debug_hq && b == debug_b && kt == debug_kt;
        const int k0 = kt * BN_TILE, valid_k = min(BN_TILE, nk - k0);
        const bool implicit_causal = (implicit_flags_arg & 1) != 0;
        const int implicit_n_kv = implicit_causal ? implicit_n_kv_arg : nk;
        const int q_offset = implicit_causal ? implicit_q_offset_arg : (nk - nq);
        const int mask_flags = implicit_causal ? implicit_flags_arg : 0;
        if (implicit_causal && k0 >= implicit_n_kv) {
            break;
        }
        bool block_meta_full_tile = false;
        bool block_meta_skipped_tile = false;
        if constexpr (BLOCK_META_ANY) {
            if (block_meta_q_plan_valid && valid_k == BN_TILE) {
                const uint32_t k_block = (uint32_t) kt;
                if constexpr (BLOCK_META_PLAN) {
                    block_meta_full_tile = ggml_cuda_fa_block_meta_range_contains(block_meta_full_begin, block_meta_full_end, k_block);
                }
                block_meta_skipped_tile = ggml_cuda_fa_block_meta_range_contains(block_meta_skipped_begin, block_meta_skipped_end, k_block);
            }
            if (block_meta_skipped_tile) {
                if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
                if (block_meta_effect_counts && threadIdx.x == 0) atomicAdd(&block_meta_effect_counts[3], 1ULL);
                continue;
            }
        }

        if constexpr (!DIRECT_ROW_KV) {
            if (threadIdx.x == 0) {
                k_phys_tile_base = pwmma_dbv_page_map_physical_k_tile_base(page_map, k0, valid_k, head_stride);
            }
            __syncthreads();
            for (int c = threadIdx.x; c < BN_TILE; c += int(blockDim.x)) {
                k_phys_tile[c] = c < valid_k ?
                    (k_phys_tile_base >= 0 ? k_phys_tile_base + c : pwmma_dbv_page_map_physical_k(page_map, k0 + c, head_stride)) : -1;
            }
            __syncthreads();
        }

        const int vbuf = DOUBLE_BUFFER_V ? (kt & 1) : 0;
        for (int idx = threadIdx.x; idx < BN_TILE * PWMMA_D; idx += blockDim.x) {
            const int c = idx / PWMMA_D, dd = idx % PWMMA_D;
            if constexpr (DIRECT_ROW_KV) {
                v_tile_f16_db[vbuf][c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
            } else {
                const int k_phys = k_phys_tile[c];
                v_tile_f16_db[vbuf][c][dd] = k_phys >= 0 ? (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k_phys, hk, b, dd) : (_Float16)0.0f;
            }
        }
        if constexpr (K_SHARED) {
            if constexpr (PACKED8_K) {
                for (int idx = threadIdx.x; idx < BN_TILE * (PWMMA_D/8); idx += blockDim.x) {
                    const int c = idx / (PWMMA_D/8);
                    const int src_word = idx - c * (PWMMA_D/8);
                    const bool k_valid = c < valid_k;
                    const int k_phys = DIRECT_ROW_KV ? (k0 + c) : (k_valid ? k_phys_tile[c] : -1);
                    int lo_i8x4 = 0;
                    int hi_i8x4 = 0;
                    if (k_valid && (DIRECT_ROW_KV || k_phys >= 0)) {
                        const size_t row = k_head_base + size_t(k_phys);
                        pwmma_i8qk_packed8_word_as_i8x4_pair(k_payload, row, k_payload_row_stride_i32, src_word, lo_i8x4, hi_i8x4);
                    }
                    int * dst_i8x4 = &k_i32_smem[c * (PWMMA_D/4) + src_word * 2];
                    dst_i8x4[0] = lo_i8x4;
                    dst_i8x4[1] = hi_i8x4;
                }
            } else {
                for (int idx = threadIdx.x; idx < BN_TILE * (PWMMA_D/16); idx += blockDim.x) {
                    const int c = idx / (PWMMA_D/16);
                    const int d16 = idx - c * (PWMMA_D/16);
                    const int g = d16 * 4;
                    int4 kw4 = {0, 0, 0, 0};
                    const bool k_valid = c < valid_k;
                    const int k_phys = DIRECT_ROW_KV ? (k0 + c) : (k_valid ? k_phys_tile[c] : -1);
                    if (k_valid && (DIRECT_ROW_KV || k_phys >= 0)) {
                        const size_t row = k_head_base + size_t(k_phys);
                        if constexpr (DIRECT_ROW_KV) {
                            kw4 = *((const int4 *) (k_payload + row * size_t(k_payload_row_stride_i32) + size_t(d16 * int(DP16_PACKED_I8X16_WORDS))));
                        } else {
                            kw4 = pwmma_i8qk_k_d16x4_desc(k_payload, packed16_desc, row, k_payload_row_stride_i32, head_stride, d16);
                        }
                    }
                    int * dst = &k_i32_smem[c * (PWMMA_D/4) + g];
                    dst[0] = kw4.x;
                    dst[1] = kw4.y;
                    dst[2] = kw4.z;
                    dst[3] = kw4.w;
                }
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                const bool k_valid = c < valid_k;
                const int k_phys = DIRECT_ROW_KV ? (k0 + c) : (k_valid ? k_phys_tile[c] : -1);
                if (k_valid && (DIRECT_ROW_KV || k_phys >= 0)) {
                    const size_t row = k_head_base + size_t(k_phys);
                    if constexpr (DIRECT_ROW_KV) {
                        k_s_smem[idx] = k_scales[row * size_t(k_scales_row_stride_half) + size_t(s)];
                    } else {
                        k_s_smem[idx] = pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, row, k_scales_row_stride_half, head_stride, s);
                    }
                } else {
                    k_s_smem[idx] = __float2half(0.0f);
                }
            }
        }
        for (int q_stage_idx = 0; q_stage_idx < Q_STAGE_COUNT; ++q_stage_idx) {
            const int q_stage_row0 = STAGED_QPACK ? q_stage_idx * Q_STAGE_ROWS : 0;
            if constexpr (STAGED_QPACK) {
                for (int idx = threadIdx.x; idx < Q_STORAGE_ROWS * Q_SCALE_BLOCKS; idx += blockDim.x) {
                    const int local_r = idx / Q_SCALE_BLOCKS;
                    const int sb = idx - local_r * Q_SCALE_BLOCKS;
                    const int global_r = q_stage_row0 + local_r;
                    int qq = 0;
                    int hq_row = 0;
                    const bool valid = pwmma_dbv_q_row_info(global_r, ROWS_PER_HEAD, DBV_GQA_GROUP,
                        hq_base, hk, gqa_ratio, n_heads_q, nq, q0, qq, hq_row);
                    const size_t qpack_row = (size_t(b) * size_t(n_heads_q) + size_t(hq_row)) * size_t(nq) + size_t(qq);
                    q_s[local_r][sb] = valid ? qpack_scales[qpack_row * size_t(qpack_scales_stride_f32) + size_t(sb)] : 0.0f;
                    #pragma unroll
                    for (int g = 0; g < Q_SCALE_BLOCK/4; ++g) {
                        q_i32[local_r][sb * (Q_SCALE_BLOCK/4) + g] = valid ?
                            qpack_payload[qpack_row * size_t(qpack_payload_stride_i32) + size_t(sb * (Q_SCALE_BLOCK/4) + g)] : 0;
                    }
                }
            }
            if (threadIdx.x < DBV_WAVES) {
                const int w = threadIdx.x;
                const int slot = DBV_SINGLE_GROUP ? 0 : ((w * 16) / ROWS_PER_HEAD);
                const int local_first = DBV_SINGLE_GROUP ? (w * 16) : ((w * 16) - slot * ROWS_PER_HEAD);
                const int hq_wave = hq_base + slot;
                const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_wave < n_heads_q && hq_wave < (hk + 1) * gqa_ratio);
                const int wg_first = q0 + local_first;
                const bool wave_in_q_stage = !STAGED_QPACK || ((w * 16) >= q_stage_row0 && (w * 16) < q_stage_row0 + Q_STAGE_ROWS);
                if (block_meta_full_tile) {
                    wave_active[w] = wave_in_q_stage && slot_valid && (wg_first < nq);
                } else {
                    const int wg_last  = min(nq - 1, wg_first + 15);
                    wave_active[w] = wave_in_q_stage && slot_valid && (wg_first < nq) && (k0 <= q_offset + wg_last);
                }
            }
            __syncthreads();
            bool any_wave_active = false;
            #pragma unroll
            for (int w = 0; w < DBV_WAVES; ++w) {
                any_wave_active = any_wave_active || wave_active[w];
            }
            if (!any_wave_active) {
                if (skip_counter && threadIdx.x == 0) atomicAdd(skip_counter, 1ULL);
                if (block_meta_effect_counts && threadIdx.x == 0) atomicAdd(&block_meta_effect_counts[3], 1ULL);
                continue;
            }
            if constexpr (BLOCK_META_ANY) {
                if (block_meta_effect_counts && threadIdx.x == 0) {
                    atomicAdd(&block_meta_effect_counts[0], 1ULL);
                    if (block_meta_full_tile) {
                        atomicAdd(&block_meta_effect_counts[1], 1ULL);
                    } else if (implicit_causal) {
                        atomicAdd(&block_meta_effect_counts[2], 1ULL);
                    }
                }
            }
            const int stage_row_base = STAGED_QPACK ? q_stage_row0 : 0;
            const int stage_rows = STAGED_QPACK ? Q_STAGE_ROWS : DBV_TOTAL_ROWS;
            const int stage_wave_base = STAGED_QPACK ? (q_stage_row0 >> 4) : 0;
            const int stage_waves = STAGED_QPACK ? ((Q_STAGE_ROWS + 15) / 16) : DBV_WAVES;
            if constexpr (!SKIP_LOGITS_INIT) {
                for (int idx = threadIdx.x; idx < stage_rows * BN_TILE; idx += blockDim.x) {
                    const int r = stage_row_base + idx / BN_TILE;
                    const int c = idx % BN_TILE;
                    logits_f32[r][c] = 0.0f;
                }
            }
            __syncthreads();

        PWMMA_PROFILE_PHASE_BEGIN();
        if (threadIdx.x < DBV_WAVES * 32) {
            const int wave_id = threadIdx.x >> 5;
            if (wave_active[wave_id]) {
                const int lane    = threadIdx.x & 31;
                const int lane_lo = lane & 15;
                const int lane_hi = lane >> 4;
                const int r_base  = wave_id * 16;
                #pragma unroll
                for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                const int k_col   = kc_base + lane_lo;
                const int k_phys_col = DIRECT_ROW_KV ? (k0 + k_col) : (k_col < BN_TILE ? k_phys_tile[k_col] : -1);
                const bool k_col_valid = k_col < valid_k && (DIRECT_ROW_KV || k_phys_col >= 0);
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
                                    const size_t row = k_head_base + size_t(k_phys_col);
                                    if constexpr (DIRECT_ROW_KV) {
                                        b_frag[g] = k_payload[row * size_t(k_payload_row_stride_i32) + size_t((sd0/4) + g)];
                                    } else {
                                        b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, k_payload_row_stride_i32, head_stride, (sd0/4) + g);
                                    }
                                } else {
                                    b_frag[g] = 0;
                                }
                            }
                            acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                            if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
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
                                        bounds_err->k_row = k_col_valid ? int(k_head_base + size_t(k_phys_col)) : -1;
                                    }
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            if constexpr (DIRECT_ROW_KV) {
                                ks = __half2float(k_scales[(k_head_base + size_t(k_phys_col)) * size_t(k_scales_row_stride_half) + size_t(d0 / QK8_0)]);
                            } else {
                                ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k_phys_col), k_scales_row_stride_half, head_stride, d0 / QK8_0));
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            acc_f[i] += float(acc_i[i]) * q_s[rr][d0 / Q_SCALE_BLOCK] * ks * attention_scale;
                        }
                    } else {
                        pbwmma_v4i32 a_frag;
                        pbwmma_v4i32 b_frag;
                        const int qr = r_base + pbwmma_i8_a_row_from_lane_lo(lane_lo);
                        const int qlr = STAGED_QPACK ? (qr - q_stage_row0) : qr;
                        const float q_scale_frag = RECOMPUTE_Q ? pwmma_dbv_q_scale32_recompute(Q, q_nb01, q_nb02, q_nb03,
                            qr, d0, ROWS_PER_HEAD, DBV_GQA_GROUP, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) : 1.0f;
                        #pragma unroll
                        for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
                            if constexpr (RECOMPUTE_Q) {
                                a_frag[g] = pwmma_dbv_q_pack_i8x4_recompute(Q, q_nb01, q_nb02, q_nb03,
                                    qr, (d0/4) + g, q_scale_frag, ROWS_PER_HEAD, DBV_GQA_GROUP, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b);
                            } else {
                                a_frag[g] = q_i32[qlr][(d0/4) + g];
                            }
                            if constexpr (K_SHARED) {
                                b_frag[g] = k_i32_smem[k_col * (PWMMA_D/4) + (d0/4) + g];
                            } else if (k_col_valid) {
                                const size_t row = k_head_base + size_t(k_phys_col);
                                if constexpr (DIRECT_ROW_KV) {
                                    b_frag[g] = k_payload[row * size_t(k_payload_row_stride_i32) + size_t((d0/4) + g)];
                                } else {
                                    b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, k_payload_row_stride_i32, head_stride, (d0/4) + g);
                                }
                            } else {
                                b_frag[g] = 0;
                            }
                        }
                        pbwmma_v8i32 acc_i = {0,0,0,0,0,0,0,0};
                        acc_i = pbwmma_mma_i8(a_frag, b_frag, acc_i);
                        if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && kt == 0) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                                const int rlr = STAGED_QPACK ? (rr - q_stage_row0) : rr;
                                const int ref = pbwmma_i8_dot4_k16_ref_frag(&q_i32[rlr][d0/4], b_frag);
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
                                    bounds_err->k_row = k_col_valid ? int(k_head_base + size_t(k_phys_col)) : -1;
                                }
                            }
                        }
                        float ks = 0.0f;
                        if constexpr (K_SHARED) {
                            ks = __half2float(k_s_smem[k_col * (PWMMA_D/QK8_0) + (d0 / QK8_0)]);
                        } else if (k_col_valid) {
                            if constexpr (DIRECT_ROW_KV) {
                                ks = __half2float(k_scales[(k_head_base + size_t(k_phys_col)) * size_t(k_scales_row_stride_half) + size_t(d0 / QK8_0)]);
                            } else {
                                ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k_phys_col), k_scales_row_stride_half, head_stride, d0 / QK8_0));
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int rr = r_base + pbwmma_i8_d_row_from_acc(i, lane_hi);
                            const int rlr = STAGED_QPACK ? (rr - q_stage_row0) : rr;
                            const float q_scale_rr = RECOMPUTE_Q ? pwmma_dbv_q_scale32_recompute(Q, q_nb01, q_nb02, q_nb03,
                                rr, d0, ROWS_PER_HEAD, DBV_GQA_GROUP, hq_base, hk, gqa_ratio, n_heads_q, nq, q0, b) : q_s[rlr][d0 / Q_SCALE_BLOCK];
                            acc_f[i] += float(acc_i[i]) * q_scale_rr * ks * attention_scale;
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
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile) {
            for (int idx = threadIdx.x; idx < DBV_TOTAL_ROWS * BN_TILE; idx += blockDim.x) {
                debug_qk_raw[idx] = logits_f32[idx / BN_TILE][idx % BN_TILE];
            }
            __syncthreads();
        }
        PWMMA_PROFILE_PHASE_END(qk_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        if constexpr (BLOCK_META_PLAN) {
            if (block_meta_full_tile) {
                for (int idx = threadIdx.x; idx < stage_rows * BN_TILE; idx += blockDim.x) {
                    const int r = stage_row_base + idx / BN_TILE, c = idx % BN_TILE;
                    const int slot = DBV_SINGLE_GROUP ? 0 : (r / ROWS_PER_HEAD);
                    const int local_r = DBV_SINGLE_GROUP ? r : (r - slot * ROWS_PER_HEAD);
                    const int hq_row = hq_base + slot;
                    const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
                    const int qq = q0 + local_r;
                    logits_f32[r][c] = (!slot_valid || qq >= nq || c >= valid_k || (!DIRECT_ROW_KV && k_phys_tile[c] < 0) || !wave_active[r >> 4]) ? -FLT_MAX/2.0f : logits_f32[r][c];
                }
            } else {
                for (int idx = threadIdx.x; idx < stage_rows * BN_TILE; idx += blockDim.x) {
                    const int r = stage_row_base + idx / BN_TILE, c = idx % BN_TILE;
                    const int slot = DBV_SINGLE_GROUP ? 0 : (r / ROWS_PER_HEAD);
                    const int local_r = DBV_SINGLE_GROUP ? r : (r - slot * ROWS_PER_HEAD);
                    const int hq_row = hq_base + slot;
                    const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
                    const int qq = q0 + local_r;
                    const int kk = k0 + c;
                    float v;
                    if (!slot_valid || qq >= nq || c >= valid_k || (!DIRECT_ROW_KV && k_phys_tile[c] < 0) || !wave_active[r >> 4]) {
                        v = -FLT_MAX/2.0f;
                    } else {
                        v = logits_f32[r][c];
                        if (mask || implicit_causal) {
                            float mask_bias = 0.0f;
                            if (!pwmma_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03,
                                    qq, kk, b, &mask_bias, implicit_n_kv, q_offset, mask_flags)) {
                                v = -INFINITY;
                            } else {
                                v += mask_bias;
                            }
                        }
                    }
                    logits_f32[r][c] = v;
                }
            }
        } else {
            for (int idx = threadIdx.x; idx < stage_rows * BN_TILE; idx += blockDim.x) {
                const int r = stage_row_base + idx / BN_TILE, c = idx % BN_TILE;
                const int slot = DBV_SINGLE_GROUP ? 0 : (r / ROWS_PER_HEAD);
                const int local_r = DBV_SINGLE_GROUP ? r : (r - slot * ROWS_PER_HEAD);
                const int hq_row = hq_base + slot;
                const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
                const int qq = q0 + local_r;
                const int kk = k0 + c;
                float v;
                if (!slot_valid || qq >= nq || c >= valid_k || (!DIRECT_ROW_KV && k_phys_tile[c] < 0) || !wave_active[r >> 4]) {
                    v = -FLT_MAX/2.0f;
                } else {
                    v = logits_f32[r][c];
                    if (mask || implicit_causal) {
                        float mask_bias = 0.0f;
                        if (!pwmma_mask_keep_and_bias(mask, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03,
                                qq, kk, b, &mask_bias, implicit_n_kv, q_offset, mask_flags)) {
                            v = -INFINITY;
                        } else {
                            v += mask_bias;
                        }
                    }
                }
                logits_f32[r][c] = v;
            }
        }
        __syncthreads();
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile && debug_qk_masked != nullptr) {
            for (int idx = threadIdx.x; idx < DBV_TOTAL_ROWS * BN_TILE; idx += blockDim.x) {
                debug_qk_masked[idx] = logits_f32[idx / BN_TILE][idx % BN_TILE];
            }
        }

        for (int rr_idx = threadIdx.x; rr_idx < stage_rows; rr_idx += blockDim.x) {
            const int r = stage_row_base + rr_idx;
            const int wave_id = r >> 4;
            const int slot = DBV_SINGLE_GROUP ? 0 : (r / ROWS_PER_HEAD);
            const int local_r = DBV_SINGLE_GROUP ? r : (r - slot * ROWS_PER_HEAD);
            const int hq_row = hq_base + slot;
            const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_row < n_heads_q && hq_row < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            if (!slot_valid || qq >= nq || !wave_active[wave_id]) {
                alpha_smem[r] = 0.0f;
                #pragma unroll
                for (int c = 0; c < BN_TILE; ++c) {
                    if constexpr (PROBS_F16) PWMMA_DBV_PROB_SET(r, c, __float2half(0.0f));
                    else PWMMA_DBV_PROB_SET(r, c, 0.0f);
                }
                continue;
            }
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) tile_max = fmaxf(tile_max, logits_f32[r][c]);
            const float new_m = fmaxf(row_m_smem[r], tile_max);
            const float alpha = (row_l_smem[r] > 0.0f) ? expf(row_m_smem[r] - new_m) : 0.0f;
            float p_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < valid_k; ++c) {
                const float p = expf(logits_f32[r][c] - new_m);
                if constexpr (PROBS_F16) PWMMA_DBV_PROB_SET(r, c, __float2half(p));
                else PWMMA_DBV_PROB_SET(r, c, p);
                p_sum += p;
            }
            alpha_smem[r] = alpha; row_m_smem[r] = new_m; row_l_smem[r] = row_l_smem[r] * alpha + p_sum;
        }
        __syncthreads();
        if constexpr (ENABLE_DEBUG_QK) if (debug_tile) {
            for (int idx = threadIdx.x; idx < DBV_TOTAL_ROWS * BN_TILE; idx += blockDim.x) {
                float p;
                if constexpr (PROBS_F16) p = __half2float(PWMMA_DBV_PROB_GET(idx / BN_TILE, idx % BN_TILE));
                else p = PWMMA_DBV_PROB_GET(idx / BN_TILE, idx % BN_TILE);
                if (debug_probs != nullptr) debug_probs[idx] = p;
            }
            for (int r = threadIdx.x; r < DBV_TOTAL_ROWS; r += blockDim.x) {
                if (debug_m != nullptr) debug_m[r] = row_m_smem[r];
                if (debug_l != nullptr) debug_l[r] = row_l_smem[r];
                if (debug_alpha != nullptr) debug_alpha[r] = alpha_smem[r];
            }
        }
        PWMMA_PROFILE_PHASE_END(softmax_cycles);
        PWMMA_PROFILE_PHASE_BEGIN();

        if constexpr (PV_WMMA) {
            const int lane    = threadIdx.x & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
            const int d_tile  = (threadIdx.x >> 5) & 15;
            for (int rb_it = 0; rb_it < stage_waves; ++rb_it) {
                const int rb = stage_wave_base + rb_it;
                if (wave_active[rb]) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        out_pv[rb][i] *= alpha_smem[global_r];
                    }
                }
            }
            #pragma unroll
            for (int kc_base = 0; kc_base < BN_TILE; kc_base += 16) {
                pbwmma_v16fp16 b_frag;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    const int kc = kc_base + kk;
                    b_frag[kk] = kc < valid_k
                        ? v_tile_f16_db[vbuf][kc][d_tile * 16 + pbwmma_f16_b_col_from_lane_lo(lane_lo)]
                        : __float2half(0.0f);
                }
                for (int rb_it = 0; rb_it < stage_waves; ++rb_it) {
                    const int rb = stage_wave_base + rb_it;
                    pbwmma_v16fp16 a_frag;
                    if (wave_active[rb]) {
                        #pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            const int kc = kc_base + kk;
                            if (kc < valid_k) {
                                if constexpr (PROBS_F16) a_frag[kk] = PWMMA_DBV_PROB_GET(rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo), kc);
                                else a_frag[kk] = __float2half(PWMMA_DBV_PROB_GET(rb * 16 + pbwmma_f16_a_row_from_lane_lo(lane_lo), kc));
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
                        if constexpr (ENABLE_BOUNDS) if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                            float ref = 0.0f;
                            if (wave_active[rb]) {
                                #pragma unroll
                                for (int c = 0; c < 16; ++c) {
                                    const int kc = kc_base + c;
                                    if (kc < valid_k) ref += __half2float(PWMMA_DBV_PROB_GET(global_r, kc)) * __half2float(v_tile_f16_db[vbuf][kc][d_tile * 16 + lane_lo]);
                                }
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
                                bounds_err->got = pv_acc[i];
                                bounds_err->ref = ref;
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
                            if constexpr (PROBS_F16) p = __half2float(PWMMA_DBV_PROB_GET(row_base + r, c));
                            else p = PWMMA_DBV_PROB_GET(row_base + r, c);
                            out[r] += p * __half2float(v_tile_f16_db[vbuf][c][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        PWMMA_PROFILE_PHASE_END(pv_cycles);
    }
    }

#undef PWMMA_DBV_PROB_GET
#undef PWMMA_DBV_PROB_SET

    if constexpr (PV_WMMA) {
        const int lane    = threadIdx.x & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
        const int d_tile  = (threadIdx.x >> 5) & 15;
        const int dd      = d_tile * 16 + lane_lo;
        #pragma unroll
        for (int rb = 0; rb < DBV_WAVES; ++rb) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int global_r = rb * 16 + rr;
                const int slot = DBV_SINGLE_GROUP ? 0 : (global_r / ROWS_PER_HEAD);
                const int local_r = DBV_SINGLE_GROUP ? global_r : (global_r - slot * ROWS_PER_HEAD);
                const int hq_out = hq_base + slot;
                const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_out < n_heads_q && hq_out < (hk + 1) * gqa_ratio);
                const int qq = q0 + local_r;
                if (!slot_valid || qq >= nq) continue;
                const float l = row_l_smem[global_r];
                if (ENABLE_STREAMK && streamk_mode) {
                    const size_t dst_row = (size_t(b) * size_t(nq) + size_t(qq)) * size_t(n_heads_q) + size_t(hq_out);
                    const size_t partial_row = size_t(split) * rows_per_split + dst_row;
                    if (dd == 0) {
                        partial_m[partial_row] = row_m_smem[global_r];
                        partial_l[partial_row] = l;
                    }
                    partial_out[partial_row * size_t(PWMMA_D) + size_t(dd)] = l > 0.0f ? out_pv[rb][i] : 0.0f;
                } else {
                    if (l <= 0.0f) continue;
                    dst[((size_t(b)*nq + qq)*n_heads_q + hq_out)*PWMMA_D + dd] = out_pv[rb][i] / l;
                }
            }
        }
    } else if (d < PWMMA_D) {
        for (int r = 0; r < 32; ++r) {
            const int global_r = row_base + r;
            const int slot = DBV_SINGLE_GROUP ? 0 : (global_r / ROWS_PER_HEAD);
            const int local_r = DBV_SINGLE_GROUP ? global_r : (global_r - slot * ROWS_PER_HEAD);
            const int hq_out = hq_base + slot;
            const bool slot_valid = DBV_SINGLE_GROUP ? true : (slot < DBV_GQA_GROUP && hq_out < n_heads_q && hq_out < (hk + 1) * gqa_ratio);
            const int qq = q0 + local_r;
            if (!slot_valid || qq >= nq) continue;
            const float l = row_l_smem[global_r];
            if (ENABLE_STREAMK && streamk_mode) {
                const size_t dst_row = (size_t(b) * size_t(nq) + size_t(qq)) * size_t(n_heads_q) + size_t(hq_out);
                const size_t partial_row = size_t(split) * rows_per_split + dst_row;
                if (d == 0) {
                    partial_m[partial_row] = row_m_smem[global_r];
                    partial_l[partial_row] = l;
                }
                partial_out[partial_row * size_t(PWMMA_D) + size_t(d)] = l > 0.0f ? out[r] : 0.0f;
            } else {
                if (l <= 0.0f) continue;
                dst[((size_t(b)*nq + qq)*n_heads_q + hq_out)*PWMMA_D + d] = out[r] / l;
            }
        }
    }
}

static __global__ void packed16_wmma_streamk_combine_kernel(
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        const float * __restrict__ partial_out,
        float       * __restrict__ dst,
        int rows_per_split,
        int streamk_splits) {
    const size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    const size_t total = size_t(rows_per_split) * size_t(PWMMA_D);
    if (idx >= total) {
        return;
    }

    const int d = int(idx % PWMMA_D);
    const size_t row = idx / size_t(PWMMA_D);

    float m = -FLT_MAX/2.0f;
    for (int s = 0; s < streamk_splits; ++s) {
        const size_t prow = size_t(s) * size_t(rows_per_split) + row;
        const float ls = partial_l[prow];
        if (ls > 0.0f) {
            m = fmaxf(m, partial_m[prow]);
        }
    }

    float l = 0.0f;
    float out = 0.0f;
    if (m > -FLT_MAX/4.0f) {
        for (int s = 0; s < streamk_splits; ++s) {
            const size_t prow = size_t(s) * size_t(rows_per_split) + row;
            const float ms = partial_m[prow];
            const float ls = partial_l[prow];
            if (ms <= -FLT_MAX/4.0f || ls <= 0.0f) {
                continue;
            }
            const float scale = expf(ms - m);
            l += ls * scale;
            out += partial_out[prow * size_t(PWMMA_D) + size_t(d)] * scale;
        }
    }

    dst[row * size_t(PWMMA_D) + size_t(d)] = l > 0.0f ? out / l : 0.0f;
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
        const dp16_packed_i8_desc_v1 packed16_desc,
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
                    k_i32_smem[idx] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, g);
                } else {
                    k_i32_smem[idx] = 0;
                }
            }
            for (int idx = threadIdx.x; idx < PWMMA_I8_KSHARED_SCALE_COUNT; idx += blockDim.x) {
                const int c = idx / (PWMMA_D/QK8_0);
                const int s = idx - c * (PWMMA_D/QK8_0);
                if (c < valid_k) {
                    const size_t row = k_head_base + size_t(k0) + size_t(c);
                    k_s_smem[idx] = pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, row, PWMMA_D/QK8_0, head_stride, s);
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
                                    b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, (sd0/4) + g);
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
                            ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k0) + size_t(k_col), PWMMA_D/QK8_0, head_stride, d0 / QK8_0));
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
                                b_frag[g] = pwmma_i8qk_k_word_desc(k_payload, packed16_desc, row, PWMMA_D/4, head_stride, (d0/4) + g);
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
                            ks = __half2float(pwmma_i8qk_k_scale_desc(k_scales, packed16_desc, k_head_base + size_t(k0) + size_t(k_col), PWMMA_D/QK8_0, head_stride, d0 / QK8_0));
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
                if (wave_active[rb]) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int global_r = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        out_pv[rb][i] *= alpha_smem[global_r];
                    }
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
                        if (bounds_err && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                            float ref = 0.0f;
                            if (wave_active[rb]) {
                                #pragma unroll
                                for (int c = 0; c < 16; ++c) {
                                    const int kc = kc_base + c;
                                    if (kc < valid_k) ref += __half2float(probs[global_r][kc]) * __half2float(v_tile_f16[kc][d_tile * 16 + lane_lo]);
                                }
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
                                bounds_err->got = pv_acc[i];
                                bounds_err->ref = ref;
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

        // V load: once per K tile, shared by both Q-head slots. Use the generic
        // element loader for V4_144 because it is not a plain F16 tensor layout.
        if constexpr (V_TYPE == PACKED16_WMMA_V4_K16D16_144) {
            for (int idx = threadIdx.x; idx < BN * D; idx += blockDim.x) {
                const int c = idx / D, dd = idx % D;
                v_tile_f16[c][dd] = (_Float16)pwmma_v_element<V_TYPE>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, k0 + c, hk, b, dd);
            }
        } else if (V_TYPE == PACKED16_WMMA_V_Q4_0) {
            pwmma_v_q4_0_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        } else if (V_TYPE == PACKED16_WMMA_V_Q8_0) {
            pwmma_v_q8_0_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, kt, valid_k, hk, b, (half*)v_tile_f16);
        } else {
            pwmma_v_f16_load<BN, D>(V, v_nb10, v_nb11, v_nb12, v_nb13, v_ne13, v_layout, kt, valid_k, hk, b, (half*)v_tile_f16);
        }

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
    const bool pwmma_log = ggml_cuda_pwmma_log_enabled();
    if (pwmma_log) {
        fprintf(stderr, "PWMMA ENTRY: Q_ne=(%lld,%lld) K_ne=(%lld,%lld) V_ne=(%lld,%lld)\n", (long long)Q->ne[0], (long long)Q->ne[1], (long long)K->ne[0], (long long)K->ne[1], (long long)V->ne[0], (long long)V->ne[1]);
    }

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
    if (pwmma_log) {
        fprintf(stderr, "PWMMA V layout=%s ne=(%lld,%lld,%lld,%lld) nb=(%lld,%lld,%lld,%lld)\n",
            v_layout == PWMMA_V_LAYOUT_FA ? "FA" : (v_layout == PWMMA_V_LAYOUT_TRANS ? "TRANS" : "NATIVE_KDH"),
            (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2], (long long)V->ne[3],
            (long long)V->nb[0], (long long)V->nb[1], (long long)V->nb[2], (long long)V->nb[3]);
    }

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
    const bool k_view_packed16 = K->ne[0] * 4 == Q->ne[0];
    const bool k_view_packed8  = K->ne[0] * 8 == Q->ne[0];
    GGML_ASSERT(Q->ne[0] == 256 && (k_view_packed16 || k_view_packed8));
    if (getenv("GGML_CUDA_PWMMA_ABORT_AFTER_LAYOUT")) {
        GGML_ABORT("PWMMA layout debug abort");
    }
    GGML_ASSERT(Q->ne[1] > 1 && Q->ne[2] % K->ne[2] == 0);
    GGML_ASSERT(V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_V4_K16D16_144);

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias, (const float*)dst->op_params+1, sizeof(float));
    memcpy(&logit_softcap, (const float*)dst->op_params+2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && logit_softcap == 0.0f);

    ggml_tensor * packed16_payload = nullptr, * packed16_scales = nullptr;
    llama_kv_cache_get_packed16_tensors(K->data, &packed16_payload, &packed16_scales);
    GGML_ASSERT(packed16_payload && packed16_scales);
    const bool k_payload_packed16 = k_view_packed16 && packed16_payload->ne[0] == PWMMA_D/4;
    const bool k_payload_packed8  = k_view_packed8  && packed16_payload->ne[0] == PWMMA_D/8;
    const int k_payload_i32_per_row = k_payload_packed8 ? PWMMA_D/8 : PWMMA_D/4;
    GGML_ASSERT((k_payload_packed16 || k_payload_packed8) && packed16_scales->ne[0] == PWMMA_D/QK8_0);
    GGML_ASSERT(packed16_payload->ne[1] >= K->ne[1] * K->ne[2] && packed16_scales->ne[1] >= K->ne[1] * K->ne[2]);
    GGML_ASSERT(packed16_payload->nb[0] == (int64_t)sizeof(int) && packed16_scales->nb[0] == (int64_t)sizeof(half));
    GGML_ASSERT(packed16_payload->nb[1] >= k_payload_i32_per_row*(int64_t)sizeof(int));
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
    dp16_packed_i8_desc_v1 packed16_desc = {};
    llama_kv_cache_get_packed16_packed_i8_desc(K->data, &packed16_desc);
    if (k_payload_packed16 && packed16_desc.layout_kind == DP16_PACKED_I8_LAYOUT_D16_PLANAR &&
            !dp16_i8x16_desc_has_vector_abi(packed16_desc)) {
        GGML_ABORT("PWMMA packed16 D16-planar K descriptor has invalid vector ABI");
    }

    const int nq = (int)Q->ne[1];
    const int nk_storage = (int)K->ne[1];
    int nk = nk_storage;
    const int n_heads_q = (int)Q->ne[2], n_heads_k = (int)K->ne[2];
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
    const int implicit_n_kv = ggml_get_op_params_i32(dst, 5);
    const int implicit_q_offset = ggml_get_op_params_i32(dst, 6);
    const int implicit_flags = ggml_get_op_params_i32(dst, 7);

    const int bm = ggml_cuda_rocm_packed16_wmma_bm();
    const int gqa_group = ggml_cuda_rocm_packed16_wmma_gqa_group();
    const int impl = ggml_cuda_rocm_packed16_wmma_impl();
    const bool is_bm32 = (bm == 32);
    const bool is_bm64 = (bm == 64);
    const bool is_gqa2 = (gqa_group == 2);

    // impl validation: bm32_regout only valid with BM32; bm64_regout only valid with BM64
    const bool impl_bm32_regout = (impl == 1 || impl == 2);
    const bool impl_bm64_regout = (impl == 3 || impl == 4 || impl == 5 || impl == 6 || impl == 7 || impl == 8 || impl == 9 || impl == 10 || impl == 11 || impl == 12 || impl == 13 || impl == 14 || impl == 15 || impl == 16 || impl == 17 || impl == 18 || impl == 19);
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
        (impl == 17) ? "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev" :
        (impl == 18) ? "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev" :
        (impl == 19) ? "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev" :
        "unknown";
    const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    const bool require_i8mma_qk = required_route &&
        (strcmp(required_route, "rocm_packed16_i8mma_qk") == 0 ||
         strcmp(required_route, "packed16_i8mma_qk") == 0 ||
         strcmp(required_route, "rocm_packed16_i8qk_wmma") == 0 ||
         strcmp(required_route, "packed16_i8qk_wmma") == 0);
    const bool strict_i8qk_impl = (impl >= 9 && impl <= 19 && impl != 17);
    if (require_i8mma_qk) {
        if (!k_view_packed16 || k_view_packed8) {
            GGML_ABORT("required %s route needs packed16 q8 I32 K payload, got k_view_packed16=%d k_view_packed8=%d impl=%s",
                required_route, k_view_packed16 ? 1 : 0, k_view_packed8 ? 1 : 0, impl_name);
        }
        if (!strict_i8qk_impl) {
            GGML_ABORT("required %s route needs bm64_i8qk WMMA QK impl over packed16 q8 K, got impl=%s", required_route, impl_name);
        }
        if (nq <= 1) {
            GGML_ABORT("required %s route is prefill/verify only and rejects nq=%d decode", required_route, nq);
        }
    }
    if (ggml_cuda_pwmma_consumer_read_mode_trace_enabled()) {
        const int fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        fprintf(stderr,
                "MTP_QBLOCK_CONSUMER_READ_MODE: backend=rocm_packed16_wmma_tile read_mode=canonical node=%s layer=%d graph_inst=%d impl=%s bm=%d nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d gqa_group=%d K=%s V=%s v_layout=%d k_view_packed16=%d k_view_packed8=%d desc_layout=%d physical_rows=%d\n",
                dst && dst->name[0] ? dst->name : "-",
                ggml_cuda_rocm_packed16_debug_layer_from_name(dst ? dst->name : nullptr),
                fa_inst_i32,
                impl_name,
                bm,
                nq,
                nk,
                n_heads_q,
                n_heads_k,
                gqa_ratio,
                gqa_group,
                ggml_type_name(K->type),
                ggml_type_name(V->type),
                v_layout,
                k_view_packed16 ? 1 : 0,
                k_view_packed8 ? 1 : 0,
                (int) packed16_desc.layout_kind,
                packed_kv_size);
    }

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
    unsigned long long * d_block_meta_effect_counts = nullptr;

    if (k_payload_packed8 && impl != 17) {
        GGML_ABORT("packed8_q4 PWMMA requires GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev, got %s", impl_name);
    }
    if (!k_payload_packed8 && impl == 17) {
        GGML_ABORT("packed8_q4 PWMMA impl requires packed8_q4_144 K payload");
    }
    if (k_payload_packed8 && impl == 18) {
        GGML_ABORT("PWMMA stream-K skeleton currently supports packed16_q8_272 K only, got packed8_q4_144");
    }
    const bool impl_i8qk = (impl == 9 || impl == 10 || impl == 11 || impl == 12 || impl == 13 || impl == 14 || impl == 15 || impl == 16 || impl == 17 || impl == 18 || impl == 19);
    const bool impl_pvwmma = (impl == 14 || impl == 15 || impl == 16 || impl == 17 || impl == 18 || impl == 19);
    const bool impl_dbv = (impl == 16 || impl == 17 || impl == 18 || impl == 19);
    if ((implicit_flags & 1) && !impl_dbv) {
        GGML_ABORT("packed16 implicit causal mask metadata requires DBV impl, got %s", impl_name);
    }

    const bool active_pages_prefill_env = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_ACTIVE_PAGES", 0) != 0;
    const bool active_pages_prefill_trace = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_ACTIVE_PAGES_TRACE", 0) != 0;
    const bool active_pages_prefill_candidate =
        active_pages_prefill_env &&
        impl_dbv &&
        nq >= 16 &&
        mask == nullptr &&
        (implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0 &&
        implicit_n_kv > 0 &&
        implicit_n_kv <= nk_storage;
    const char * active_pages_prefill_reason = "ok";
    if (active_pages_prefill_env && !active_pages_prefill_candidate) {
        active_pages_prefill_reason = !impl_dbv ? "not_dbv" :
            (nq < 16 ? "not_prefill" :
            (mask != nullptr ? "dense_mask" :
            ((implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) == 0 ? "no_causal_meta" :
            (implicit_n_kv <= 0 ? "bad_active_n_kv" :
            (implicit_n_kv > nk_storage ? "active_gt_storage" : "unknown")))));
    }
    if (active_pages_prefill_candidate) {
        nk = implicit_n_kv;
    }

    pwmma_dbv_page_map_v1 dbv_page_map = {};
    const int dbv_paged_attention_min_nq_env =
        ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION_MIN_NQ", 16);
    const int dbv_paged_attention_min_nq = dbv_paged_attention_min_nq_env < 1 ? 1 : dbv_paged_attention_min_nq_env;
    const bool dbv_paged_attention_requested = impl_dbv && nq >= dbv_paged_attention_min_nq &&
        ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION", 0) != 0;
    const bool dbv_paged_attention_require_full_map = dbv_paged_attention_requested &&
        ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION_REQUIRE_FULL_MAP", 1) != 0;
    if (dbv_paged_attention_requested) {
        ggml_cuda_mtp_qblock_full_page_map_v1 registered_map = {};
        llama_kv_cache_get_mtp_qblock_full_page_map(K->data, &registered_map);
        const int packed_head_capacity = n_heads_k > 0 ? packed_kv_size / n_heads_k : 0;
        const uint32_t active_logical_n_kv = ((implicit_flags & 1) != 0 && implicit_n_kv > 0) ? (uint32_t) implicit_n_kv : (uint32_t) nk;
        const uint32_t required_pages = registered_map.page_tokens == 0 ? 0u :
            (active_logical_n_kv + registered_map.page_tokens - 1u) / registered_map.page_tokens;
        const uint64_t map_end = uint64_t(registered_map.logical_base_token) + uint64_t(registered_map.valid_tokens);
        if (registered_map.active) {
            const char * reject = nullptr;
            if (!k_payload_packed16 || k_payload_packed8) {
                reject = "not_packed16_q8_272";
            } else if (registered_map.version != GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_VERSION ||
                    registered_map.abi_bytes != sizeof(ggml_cuda_mtp_qblock_full_page_map_v1)) {
                reject = "bad_full_map_abi";
            } else if (registered_map.page_tokens != GGML_CUDA_PACKED16_K_PAGE16_TOKENS) {
                reject = "bad_page_tokens";
            } else if (registered_map.logical_base_token != 0 || registered_map.valid_tokens < active_logical_n_kv) {
                reject = "requires_full_current_k_page_table";
            } else if (registered_map.valid_tokens == 0 || required_pages == 0 || map_end <= registered_map.logical_base_token) {
                reject = "empty_full_map";
            } else if (registered_map.block_table == nullptr || registered_map.block_table_pages < required_pages) {
                reject = "bad_full_block_table";
            } else if (registered_map.non_identity_page_begin > registered_map.non_identity_page_end ||
                    registered_map.non_identity_page_end > registered_map.block_table_pages) {
                reject = "bad_non_identity_range";
            } else if (packed_head_capacity <= 0 || packed_kv_size % n_heads_k != 0) {
                reject = "bad_packed_head_capacity";
            } else if (registered_map.physical_pages == 0 ||
                    uint64_t(registered_map.physical_pages) * uint64_t(registered_map.page_tokens) > uint64_t(packed_head_capacity)) {
                reject = "bad_physical_pages";
            }
            if (reject != nullptr) {
                GGML_ABORT("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION rejected: %s", reject);
            }
            dbv_page_map.active = 1;
            dbv_page_map.logical_base_token = registered_map.logical_base_token;
            dbv_page_map.valid_tokens = registered_map.valid_tokens;
            dbv_page_map.page_tokens = registered_map.page_tokens;
            dbv_page_map.physical_pages = registered_map.physical_pages;
            dbv_page_map.block_table_pages = registered_map.block_table_pages;
            dbv_page_map.non_identity_page_begin = registered_map.non_identity_page_begin;
            dbv_page_map.non_identity_page_end = registered_map.non_identity_page_end;
            dbv_page_map.flags = registered_map.flags;
            dbv_page_map.block_table = registered_map.block_table;
            if (ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION_TRACE", 0) != 0) {
                fprintf(stderr,
                        "PWMMA DBV PAGED_ATTENTION: active=1 full_map=1 flags=0x%x logical_base=%u valid_tokens=%u page_tokens=%u physical_pages=%u block_table_pages=%u table_ptr=%p first_pages=[%d,%d,%d,%d] nk=%d active_n_kv=%u capacity=%d\n",
                        registered_map.flags,
                        dbv_page_map.logical_base_token, dbv_page_map.valid_tokens, dbv_page_map.page_tokens,
                        dbv_page_map.physical_pages, dbv_page_map.block_table_pages,
                        (const void *) dbv_page_map.block_table,
                        registered_map.debug_first_pages[0], registered_map.debug_first_pages[1], registered_map.debug_first_pages[2], registered_map.debug_first_pages[3],
                        nk, active_logical_n_kv, packed_head_capacity);
            }
        } else {
            if (dbv_paged_attention_require_full_map) {
                GGML_ABORT("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION rejected: requires_full_current_k_page_table");
            }
            if (ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PAGED_ATTENTION_TRACE", 0) != 0) {
                fprintf(stderr,
                        "PWMMA DBV PAGE_ATTENTION_SKIP: reason=no_registered_full_map nk=%d active_n_kv=%u capacity=%d\n",
                        nk, active_logical_n_kv, packed_head_capacity);
            }
        }
    }

    const bool block_meta_effect_stats = impl_dbv && ggml_cuda_rocm_env_i32("GGML_CUDA_FA_BLOCK_META_EFFECT_STATS", 0) != 0;
    const bool work_trace = impl_dbv && ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_WMMA_WORK_TRACE", 0) != 0;
    if (block_meta_effect_stats || work_trace) {
        CUDA_CHECK(hipMalloc(&d_block_meta_effect_counts, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(hipMemset(d_block_meta_effect_counts, 0, 4 * sizeof(unsigned long long)));
    }
    const bool dbv_force_implicit_causal = impl_dbv && mask != nullptr && batch == 1 &&
        ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_FORCE_IMPLICIT_CAUSAL", 0) != 0;
    const char * dbv_mask_data = dbv_force_implicit_causal ? nullptr : (mask ? (const char *) mask->data : nullptr);
    const int dbv_implicit_n_kv = dbv_force_implicit_causal ? nk : implicit_n_kv;
    const int dbv_implicit_q_offset = dbv_force_implicit_causal ? (nk - nq) : implicit_q_offset;
    int dbv_implicit_flags = dbv_force_implicit_causal ? (implicit_flags | 1) : implicit_flags;
    const bool dbv_block_meta_full_tile_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_FA_BLOCK_META_FULL_TILE", 0) != 0;
    const bool dbv_block_meta_skip_clamp_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_FA_BLOCK_META_SKIP_CLAMP", 0) != 0;
    if (dbv_block_meta_full_tile_requested) {
        dbv_implicit_flags |= PWMMA_IMPLICIT_FLAG_BLOCK_META_FULL_TILE;
    }
    if (dbv_block_meta_skip_clamp_requested) {
        dbv_implicit_flags |= PWMMA_IMPLICIT_FLAG_BLOCK_META_SKIP_CLAMP;
    }
    const bool dbv_block_meta_full_tile_plan_active = impl_dbv && dbv_mask_data == nullptr &&
        (dbv_implicit_flags & PWMMA_IMPLICIT_FLAG_BLOCK_META_FULL_TILE) != 0 &&
        (dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0;
    const bool dbv_block_meta_skip_clamp_plan_active = impl_dbv && dbv_mask_data == nullptr &&
        (dbv_implicit_flags & PWMMA_IMPLICIT_FLAG_BLOCK_META_SKIP_CLAMP) != 0 &&
        (dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0;
    if (active_pages_prefill_env && active_pages_prefill_trace) {
        const int skipped_capacity_tokens = active_pages_prefill_candidate ? nk_storage - nk : 0;
        const int skipped_capacity_blocks = active_pages_prefill_candidate ?
            (CEIL_DIV(nk_storage, PWMMA_BN) - CEIL_DIV(nk, PWMMA_BN)) : 0;
        fprintf(stderr,
            "GGML_CUDA_ROCM_PACKED16_PREFILL_ACTIVE_PAGES_TRACE: backend=pwmma_dbv active=%d reason=%s impl=%s node=%s "
            "nq=%d nk_storage=%d nk_launch=%d valid_n_kv=%d q_offset=%d implicit_q_offset=%d flags=%d "
            "packed_rows=%d skipped_capacity_tokens=%d skipped_capacity_blocks=%d grid=(%u,%u,%u)\n",
            active_pages_prefill_candidate ? 1 : 0, active_pages_prefill_reason, impl_name,
            (dst && dst->name[0]) ? dst->name : "-", nq, nk_storage, nk, dbv_implicit_n_kv,
            dbv_implicit_q_offset, implicit_q_offset, dbv_implicit_flags, packed_kv_size,
            skipped_capacity_tokens, skipped_capacity_blocks, (unsigned) grid.x, (unsigned) grid.y, (unsigned) grid.z);
    }
    if (ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT", 0) != 0) {
        GGML_ABORT("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT is planner-only in this tree; use "
                   "GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT_TRACE=1 for non-routing diagnostics");
    }
    if (ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT_TRACE", 0) != 0) {
        const bool prefix_split_pure_causal = impl_dbv && dbv_mask_data == nullptr &&
            (dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0 &&
            dbv_implicit_n_kv > 0 && dbv_implicit_n_kv <= nk;
        const bool prefix_split_row_layout = packed16_desc.layout_kind == DP16_PACKED_I8_LAYOUT_ROW &&
            packed16_desc.scale_layout == DP16_PACKED_I8_SCALE_LAYOUT_ROW;
        const bool prefix_split_candidate = prefix_split_pure_causal && prefix_split_row_layout &&
            k_payload_packed16 && !k_payload_packed8 && nq >= 128 && dbv_implicit_q_offset > 0;
        const char * prefix_split_reason = prefix_split_candidate ? "ok" :
            (!impl_dbv ? "not_dbv" :
            (dbv_mask_data != nullptr ? "dense_mask" :
            ((dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) == 0 ? "no_causal_meta" :
            (dbv_implicit_n_kv <= 0 ? "bad_active_n_kv" :
            (dbv_implicit_n_kv > nk ? "active_gt_launch_nk" :
            (!prefix_split_row_layout ? "not_row_layout" :
            (!k_payload_packed16 || k_payload_packed8 ? "not_q8_272" :
            (nq < 128 ? "small_nq" :
            (dbv_implicit_q_offset <= 0 ? "no_prefix" : "unknown")))))))));
        const int active_n_kv = prefix_split_pure_causal ? dbv_implicit_n_kv : nk;
        const int prefix_tokens_raw = prefix_split_pure_causal ? dbv_implicit_q_offset : 0;
        const int prefix_tokens = prefix_tokens_raw < 0 ? 0 : (prefix_tokens_raw > active_n_kv ? active_n_kv : prefix_tokens_raw);
        const int causal_tail_tokens = active_n_kv > prefix_tokens ? active_n_kv - prefix_tokens : 0;
        const int prefix_tiles = CEIL_DIV(prefix_tokens, PWMMA_BN);
        const int causal_tail_tiles = CEIL_DIV(causal_tail_tokens, PWMMA_BN);
        const int q_tiles_bm64 = CEIL_DIV(nq, 64);
        const int qtiles_per_stage_raw = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_QTILES", 4);
        const int qtiles_per_stage = qtiles_per_stage_raw > 0 ? qtiles_per_stage_raw : 4;
        const int gqa_fuse_raw = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_GQA_FUSE", gqa_ratio);
        const int gqa_fuse = gqa_fuse_raw > 0 ? (gqa_fuse_raw > gqa_ratio ? gqa_ratio : gqa_fuse_raw) : 1;
        const int q_stage_groups = CEIL_DIV(q_tiles_bm64, qtiles_per_stage);
        const int gqa_stage_groups = CEIL_DIV(gqa_ratio, gqa_fuse);
        const size_t rows = size_t(batch) * size_t(nq) * size_t(n_heads_q);
        const size_t two_state_f32 = rows * size_t(2) * size_t(PWMMA_D + 2);
        const double two_state_mib = double(two_state_f32 * sizeof(float)) / (1024.0 * 1024.0);
        const long long baseline_prefix_tile_refs = (long long) q_tiles_bm64 * (long long) prefix_tiles *
            (long long) n_heads_q * (long long) batch;
        const long long baseline_tail_tile_refs = (long long) q_tiles_bm64 * (long long) causal_tail_tiles *
            (long long) n_heads_q * (long long) batch;
        const long long baseline_tile_refs = baseline_prefix_tile_refs + baseline_tail_tile_refs;
        const long long prefix_tail_tile_refs = baseline_tile_refs;
        const long long kmajor_prefix_k_load_refs = (long long) q_stage_groups * (long long) prefix_tiles *
            (long long) n_heads_k * (long long) gqa_stage_groups * (long long) batch;
        const double prefix_k_reuse_ratio = kmajor_prefix_k_load_refs > 0 ?
            (double) baseline_prefix_tile_refs / (double) kmajor_prefix_k_load_refs : 0.0;
        fprintf(stderr,
            "GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT_TRACE: backend=pwmma_dbv active=%d reason=%s impl=%s node=%s "
            "nq=%d nk_launch=%d active_n_kv=%d q_offset=%d prefix_tokens=%d causal_tail_tokens=%d "
            "prefix_tiles=%d causal_tail_tiles=%d q_tiles_bm64=%d hq=%d hk=%d gqa=%d batch=%d "
            "desc_layout=%d desc_scale_layout=%d two_state_workspace_mib=%.1f baseline_tile_refs=%lld "
            "prefix_tail_tile_refs=%lld baseline_prefix_tile_refs=%lld baseline_tail_tile_refs=%lld "
            "qtiles_per_stage=%d gqa_fuse=%d kmajor_prefix_k_load_refs=%lld prefix_k_reuse_ratio=%.2f\n",
            prefix_split_candidate ? 1 : 0, prefix_split_reason, impl_name,
            (dst && dst->name[0]) ? dst->name : "-", nq, nk, active_n_kv, dbv_implicit_q_offset,
            prefix_tokens, causal_tail_tokens, prefix_tiles, causal_tail_tiles, q_tiles_bm64,
            n_heads_q, n_heads_k, gqa_ratio, batch, (int) packed16_desc.layout_kind,
            (int) packed16_desc.scale_layout, two_state_mib, baseline_tile_refs, prefix_tail_tile_refs,
            baseline_prefix_tile_refs, baseline_tail_tile_refs, qtiles_per_stage, gqa_fuse,
            kmajor_prefix_k_load_refs, prefix_k_reuse_ratio);
    }
    if (dbv_force_implicit_causal) {
        static bool dbv_force_implicit_logged = false;
        if (!dbv_force_implicit_logged) {
            dbv_force_implicit_logged = true;
            fprintf(stderr, "PWMMA DBV FORCE_IMPLICIT_CAUSAL: enabled impl=%s nq=%d nk=%d q_offset=%d\n",
                impl_name, nq, nk, dbv_implicit_q_offset);
        }
    }
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

    ggml_cuda_rocm_packed16_wmma_timing_event fa_timing;
    ggml_cuda_rocm_packed16_wmma_timing_begin(fa_timing, stream, "fa_pwmma");

    { static bool once = false; if (!once) { once = true;
        if (pwmma_log) {
            fprintf(stderr, "PWMMA v0.6 variant=%s BM=%d GQA_GROUP=%d IMPL=%s Q4fix=%d nq=%d nk=%d hq=%d hk=%d b=%d sc=%g "
                    "gqa_ratio=%d grid_y_old=%d grid_y_new=%d k_format=%s payload_ne0=%lld payload_ne1=%lld payload_ne2=%lld packed_kv_size=%d head_stride=%d payload_stride_i32=%d scales_stride_half=%d\n",
                    is_gqa2 ? "BM16_GQA2" : (is_bm64 ? "BM64_X4" : (is_bm32 ? "BM32_2W" : "BM16_1W")), bm, gqa_group,
                    impl_name,
                    PWMMA_Q4_LAYOUT_FIXED, nq, nk, n_heads_q, n_heads_k, batch, (double)attention_scale,
                    gqa_ratio, grid_y_old, is_gqa2 ? grid_y_gqa2 : grid_y_old,
                    k_payload_packed8 ? "packed8_q4_144" : "packed16_q8_272",
                    (long long)packed16_payload->ne[0], (long long)packed16_payload->ne[1], (long long)packed16_payload->ne[2],
                    packed_kv_size, packed_kv_size / n_heads_k,
                    k_payload_row_stride_i32, k_scales_row_stride_half);
        }
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
        ggml_cuda_pwmma_self_probes_fail_closed_if_requested();
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
            case GGML_TYPE_V4_K16D16_144: LAUNCH_GQA2(PACKED16_WMMA_V4_K16D16_144); break;
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
        k_payload, k_scales, packed16_desc, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, d_live_shadow_err)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V_F16);  break;
                case GGML_TYPE_V4_K16D16_144: LAUNCH_BM64_I8QK_PV_BN32(PACKED16_WMMA_V4_K16D16_144); break;
                default: GGML_ABORT("pwmma bm64 i8qk pvwmma bn32: unsupported V type");
            }
#undef LAUNCH_BM64_I8QK_PV_BN32
        } else if (impl == 16 || impl == 17 || impl == 18 || impl == 19) {
            dim3 block512(512);
            const bool streamk_active = impl == 18;
            const int num_k_tiles_dbv = CEIL_DIV(nk, PWMMA_BN);
            const int prefix_split_exact_tiles = CEIL_DIV(dbv_implicit_q_offset > 0 ? dbv_implicit_q_offset : 0, PWMMA_BN);
            const bool prefix_split_exact_requested = (impl == 16 || impl == 19) &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_PREFIX_SPLIT_EXACT", 0) != 0;
            const bool prefix_split_exact_active = prefix_split_exact_requested && !streamk_active &&
                dbv_mask_data == nullptr && (dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0 &&
                dbv_implicit_n_kv > 0 && dbv_implicit_n_kv <= nk && prefix_split_exact_tiles > 0 &&
                prefix_split_exact_tiles < num_k_tiles_dbv;
            const bool partial_mode_active = streamk_active || prefix_split_exact_active;
            int streamk_splits = 1;
            int streamk_prefix_tiles = 0;
            ggml_cuda_pool_alloc<float> partial_m;
            ggml_cuda_pool_alloc<float> partial_l;
            ggml_cuda_pool_alloc<float> partial_out;
            dim3 dbv_grid = grid;
            if (partial_mode_active) {
                if (streamk_active) {
                    streamk_splits = std::min(ggml_cuda_rocm_packed16_wmma_streamk_splits(), num_k_tiles_dbv);
                    if (streamk_splits < 2) {
                        GGML_ABORT("PWMMA stream-K requested but nk=%d gives only %d K tile(s)", nk, num_k_tiles_dbv);
                    }
                } else {
                    streamk_splits = 2;
                    streamk_prefix_tiles = prefix_split_exact_tiles;
                }
                const size_t rows_per_split = size_t(batch) * size_t(nq) * size_t(n_heads_q);
                const size_t partial_rows = rows_per_split * size_t(streamk_splits);
                ggml_cuda_pool & pool = ctx.pool();
                partial_m.alloc(pool, partial_rows);
                partial_l.alloc(pool, partial_rows);
                partial_out.alloc(pool, partial_rows * size_t(PWMMA_D));
                CUDA_CHECK(hipMemsetAsync(partial_l.get(), 0, partial_rows * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(partial_out.get(), 0, partial_rows * size_t(PWMMA_D) * sizeof(float), stream));
                dbv_grid.z = batch * streamk_splits;
                if (pwmma_log || prefix_split_exact_active) {
                    const double mib = double((partial_rows * 2 + partial_rows * size_t(PWMMA_D)) * sizeof(float)) / (1024.0 * 1024.0);
                    fprintf(stderr, "%s: splits=%d prefix_tiles=%d rows_per_split=%zu partial_rows=%zu workspace_mib=%.1f grid=(%u,%u,%u)\n",
                        prefix_split_exact_active ? "PWMMA PREFIX_SPLIT_EXACT" : "PWMMA STREAMK",
                        streamk_splits, streamk_prefix_tiles, rows_per_split, partial_rows, mib,
                        (unsigned)dbv_grid.x, (unsigned)dbv_grid.y, (unsigned)dbv_grid.z);
                }
            }

            const bool require_i32_source_f16_qk_route = required_route &&
                (strcmp(required_route, "rocm_quant_prefill_f16") == 0 || strcmp(required_route, "f16_temp") == 0);
            const bool dbv_i32_source_f16_qk_requested = nq > 2 &&
                (require_i32_source_f16_qk_route || ggml_cuda_fattn_rocm_quant_prefill_f16_enabled() || ggml_cuda_fattn_rocm_quant_prefill_f16_auto_enabled());
            const bool dbv_i32_source_f16_qk_candidate = dbv_i32_source_f16_qk_requested && impl == 16 && gqa_ratio == 6 &&
                n_heads_q == 24 && n_heads_k == 4 && V->type == GGML_TYPE_V4_K16D16_144 &&
                !partial_mode_active && dbv_page_map.active == 0;
            if (dbv_i32_source_f16_qk_requested && !dbv_i32_source_f16_qk_candidate) {
                GGML_ABORT("packed16 DBV I32-source f16-QK route rejected: requires plain impl16 GQA6_BM96, V4_144, no streamK/prefix split, no paged map; impl=%s gqa=%d hq=%d hk=%d V=%s partial=%d paged=%d",
                    impl_name, gqa_ratio, n_heads_q, n_heads_k, ggml_type_name(V->type), partial_mode_active ? 1 : 0, dbv_page_map.active ? 1 : 0);
            }
            if (dbv_i32_source_f16_qk_candidate &&
                    (pwmma_log || getenv("COMPRESSED_KV_FATTN_LOG") || getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE"))) {
                fprintf(stderr,
                    "PWMMA DBV GQA6_I32SRC_F16QK: active=1 route=rocm_quant_prefill_f16 impl=%s nq=%d nk=%d gqa=%d k_source=i32_packed16 k_stage=tile_smem_f16 k_operand=f16_fragment_from_i8_scale k_temp=none v_temp=none qk_math=f16_wmma dbv_shape=GQA6_BM96\n",
                    impl_name, nq, nk, gqa_ratio);
            }
            constexpr size_t PWMMA_DEBUG_QK_ELEMS = size_t(PWMMA_BM64) * size_t(PWMMA_BN);
            constexpr size_t PWMMA_DEBUG_ROW_ELEMS = size_t(PWMMA_BM64);
            const int debug_nk_filter = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_NK", -1);
            bool debug_qk_active = ggml_cuda_rocm_packed16_fa_debug_qk_enabled() && !partial_mode_active &&
                ggml_cuda_rocm_packed16_debug_layer_matches(dst) &&
                (debug_nk_filter < 0 || debug_nk_filter == nk);
            const int debug_qtile = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_QTILE", 0);
            const int debug_hq    = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_HEAD", 0);
            const int debug_b     = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_BATCH", 0);
            const int debug_kt    = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_KT", 0);
            ggml_cuda_pool_alloc<float> debug_qk_raw;
            ggml_cuda_pool_alloc<float> debug_qk_masked;
            ggml_cuda_pool_alloc<float> debug_probs;
            ggml_cuda_pool_alloc<float> debug_m;
            ggml_cuda_pool_alloc<float> debug_l;
            ggml_cuda_pool_alloc<float> debug_alpha;
            if (debug_qk_active) {
                hipStreamCaptureStatus debug_cap_status = hipStreamCaptureStatusNone;
                if (hipStreamIsCapturing(stream, &debug_cap_status) == hipSuccess && debug_cap_status != hipStreamCaptureStatusNone) {
                    static bool debug_capture_warned = false;
                    if (!debug_capture_warned) {
                        debug_capture_warned = true;
                        fprintf(stderr, "PWMMA DEBUG QK skipped during graph capture\n");
                    }
                    debug_qk_active = false;
                }
            }
            if (debug_qk_active) {
                const int num_k_tiles = CEIL_DIV(nk, PWMMA_BN);
                if (debug_qtile < 0 || debug_qtile >= (int) grid.x || debug_hq < 0 || debug_hq >= n_heads_q ||
                        debug_b < 0 || debug_b >= batch || debug_kt < 0) {
                    GGML_ABORT("PWMMA DEBUG QK invalid tile qtile=%d/%u head=%d/%d batch=%d/%d kt=%d/%d",
                        debug_qtile, (unsigned) grid.x, debug_hq, n_heads_q, debug_b, batch, debug_kt, num_k_tiles);
                }
                if (debug_kt >= num_k_tiles) {
                    debug_qk_active = false;
                }
            }
            if (debug_qk_active) {
                ggml_cuda_pool & pool = ctx.pool();
                debug_qk_raw.alloc(pool, PWMMA_DEBUG_QK_ELEMS);
                debug_qk_masked.alloc(pool, PWMMA_DEBUG_QK_ELEMS);
                debug_probs.alloc(pool, PWMMA_DEBUG_QK_ELEMS);
                debug_m.alloc(pool, PWMMA_DEBUG_ROW_ELEMS);
                debug_l.alloc(pool, PWMMA_DEBUG_ROW_ELEMS);
                debug_alpha.alloc(pool, PWMMA_DEBUG_ROW_ELEMS);
                CUDA_CHECK(hipMemsetAsync(debug_qk_raw.get(),    0, PWMMA_DEBUG_QK_ELEMS  * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(debug_qk_masked.get(), 0, PWMMA_DEBUG_QK_ELEMS  * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(debug_probs.get(),     0, PWMMA_DEBUG_QK_ELEMS  * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(debug_m.get(),         0, PWMMA_DEBUG_ROW_ELEMS * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(debug_l.get(),         0, PWMMA_DEBUG_ROW_ELEMS * sizeof(float), stream));
                CUDA_CHECK(hipMemsetAsync(debug_alpha.get(),     0, PWMMA_DEBUG_ROW_ELEMS * sizeof(float), stream));
                fprintf(stderr, "PWMMA DEBUG QK: armed node=%s impl=%s nq=%d nk=%d qtile=%d head=%d batch=%d kt=%d q0=%d k0=%d\n",
                    dst->name, impl_name, nq, nk, debug_qtile, debug_hq, debug_b, debug_kt, debug_qtile * PWMMA_BM64, debug_kt * PWMMA_BN);
            }
            const bool dbv_kshared_active = impl == 19;
            const bool dbv_gqa6_kshared_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_GQA6_KSHARED", 0) != 0;
            const bool dbv_gqa6_rph32_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_GQA6_RPH32", 0) != 0;
            const bool dbv_kblock_cascade_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_PREFILL_KBLOCK_CASCADE", 0) != 0;
            const bool dbv_gqa3_rph32_requested = ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_GQA3_RPH32", 0) != 0;
            const bool dbv_prod_specialization_ok = (impl == 16 || dbv_kshared_active) && !streamk_active && !debug_qk_active &&
                !profile_enabled && !live_dot4_shadow;
            const bool dbv_direct_row_specialization_ok = (impl == 16 || impl == 17) && !streamk_active && !debug_qk_active &&
                !profile_enabled && !live_dot4_shadow;
            const bool dbv_row_layout = packed16_desc.layout_kind == DP16_PACKED_I8_LAYOUT_ROW &&
                packed16_desc.scale_layout == DP16_PACKED_I8_SCALE_LAYOUT_ROW;
            const bool dbv_direct_row_kv_active = dbv_direct_row_specialization_ok && !dbv_kshared_active && !partial_mode_active &&
                dbv_page_map.active == 0 && ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_DIRECT_ROW_KV", 1) != 0 &&
                ((impl == 16 && k_payload_packed16 && !k_payload_packed8 && dbv_row_layout) ||
                 (impl == 17 && k_payload_packed8));
            const bool dbv_gqa2_active = dbv_prod_specialization_ok && !dbv_gqa6_rph32_requested && !dbv_kblock_cascade_requested && !dbv_gqa3_rph32_requested && !dbv_gqa6_kshared_requested && !dbv_kshared_active && gqa_ratio >= 2 &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_GQA2", 0) != 0;
            // Demoted 2026-06-27: total192 attempts are correctness-clean but too slow.
            // - recompute-Q min-LDS: pp32k/n16 still in prefill after >230s.
            // - QPACK-staged retry: pp32k/n16 324.63 tok/s vs matched GQA6 547.58.
            // Keep fail-closed until a real cascade/staged-Q design avoids the double K pass.
            const bool dbv_gqa6_rph32_active = false;
            if (dbv_gqa6_rph32_requested) {
                GGML_ABORT("GGML_CUDA_ROCM_PACKED16_DBV_GQA6_RPH32 is demoted: total192 candidates were too slow; use GQA3_RPH32_TOTAL96 diagnostics or redesign with cascade/staged-Q before retry");
            }
            // Demoted 2026-06-27: K-outer total192 cascade is correctness-clean but slower.
            // Stage-local version: pp32k/n16 394.56 tok/s vs matched GQA6 541.66 (-27.2%).
            // The saved K/V load does not overcome QPACK prep + larger CTA/state overhead.
            const bool dbv_kblock_cascade_active = false;
            if (dbv_kblock_cascade_requested) {
                GGML_ABORT("GGML_CUDA_ROCM_PACKED16_PREFILL_KBLOCK_CASCADE is demoted: pp32k/n16 was 394.56 tok/s vs GQA6 541.66; do not use without a new prefill scheduler design");
            }
            const bool dbv_gqa3_rph32_active = dbv_prod_specialization_ok && dbv_gqa3_rph32_requested &&
                impl == 16 && !dbv_kshared_active && !dbv_gqa6_rph32_active && !dbv_kblock_cascade_active && !dbv_gqa2_active && gqa_ratio >= 3;
            if (dbv_gqa3_rph32_requested && !dbv_gqa3_rph32_active) {
                GGML_ABORT("GGML_CUDA_ROCM_PACKED16_DBV_GQA3_RPH32 requires impl=16 DBV, no kshared/streamk/profile/debug/live-shadow, and gqa_ratio>=3");
            }
            const bool dbv_gqa6_kshared_active = dbv_prod_specialization_ok && dbv_gqa6_kshared_requested &&
                (impl == 16 || dbv_kshared_active) && !dbv_gqa6_rph32_active && !dbv_kblock_cascade_active && !dbv_gqa3_rph32_active && !dbv_gqa2_active && gqa_ratio >= 6;
            const char * dbv_gqa6_env = getenv("GGML_CUDA_ROCM_PACKED16_DBV_GQA6");
            const bool dbv_gqa6_27b_default = !dbv_gqa6_env &&
                V->type == GGML_TYPE_V4_K16D16_144 && n_heads_q == 24 && n_heads_k == 4 && gqa_ratio == 6;
            const bool dbv_gqa6_requested = dbv_gqa6_27b_default ||
                (dbv_gqa6_env && *dbv_gqa6_env && atoi(dbv_gqa6_env) != 0);
            const bool dbv_gqa6_active = dbv_prod_specialization_ok && impl == 16 && !dbv_gqa6_rph32_active && !dbv_kblock_cascade_active && !dbv_gqa3_rph32_active && !dbv_gqa6_kshared_active && !dbv_gqa2_active && !dbv_kshared_active && gqa_ratio >= 6 &&
                dbv_gqa6_requested;
            const size_t kshared_smem = (impl == 17 || impl == 19 || dbv_gqa6_kshared_active) ? PWMMA_I8_KSHARED_SMEM_BYTES : 0;
            if (dbv_gqa2_active) {
                dbv_grid.x = CEIL_DIV(nq, PWMMA_BM64 / 2);
                dbv_grid.y = n_heads_k * CEIL_DIV(gqa_ratio, 2);
            } else if (dbv_kblock_cascade_active || dbv_gqa6_rph32_active) {
                dbv_grid.x = CEIL_DIV(nq, 32);
                dbv_grid.y = n_heads_k * CEIL_DIV(gqa_ratio, 6);
            } else if (dbv_gqa3_rph32_active) {
                dbv_grid.x = CEIL_DIV(nq, 32);
                dbv_grid.y = n_heads_k * CEIL_DIV(gqa_ratio, 3);
            } else if (dbv_gqa6_active || dbv_gqa6_kshared_active) {
                dbv_grid.x = CEIL_DIV(nq, 16);
                dbv_grid.y = n_heads_k * CEIL_DIV(gqa_ratio, 6);
            }
            const bool dbv_grouped_active = dbv_gqa2_active || dbv_gqa6_rph32_active || dbv_kblock_cascade_active || dbv_gqa3_rph32_active || dbv_gqa6_active || dbv_gqa6_kshared_active;
            const bool dbv_no_logits_init_active = dbv_prod_specialization_ok && !dbv_grouped_active &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_NO_LOGITS_INIT", 0) != 0;
            const bool dbv_probs_overlay_active = dbv_prod_specialization_ok && !dbv_grouped_active && !dbv_no_logits_init_active &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_PROBS_OVERLAY", 0) != 0;
            const bool dbv_single_vbuf_active = dbv_prod_specialization_ok && !dbv_grouped_active && !dbv_no_logits_init_active && !dbv_probs_overlay_active &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_SINGLE_VBUF", 0) != 0;
            const bool dbv_lean_active = dbv_prod_specialization_ok && !dbv_grouped_active && !dbv_no_logits_init_active && !dbv_probs_overlay_active && !dbv_single_vbuf_active &&
                ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_LEAN", 0) != 0;
            dp16_fa_q_view qpack_view = dp16_fa_make_inline_q_view(Q, PWMMA_D);
            dp16_fa_q_workspace qpack_workspace(ctx.pool());
            if (dbv_kblock_cascade_active || dbv_gqa6_rph32_active) {
                qpack_view = dp16_fa_prepare_q_stage<PWMMA_D, 256>(stream, DP16_FA_Q_STAGE_QPACK_I8_BLOCK32, Q, qpack_workspace);
            }
            if ((dbv_direct_row_kv_active || dbv_lean_active || dbv_single_vbuf_active || dbv_probs_overlay_active || dbv_grouped_active || dbv_no_logits_init_active) && pwmma_log) {
                static bool dbv_prod_specialization_logged = false;
                if (!dbv_prod_specialization_logged) {
                    dbv_prod_specialization_logged = true;
                    const char * dbv_route = dbv_gqa6_kshared_active ? "GQA6_BM96_KSHARED" : (dbv_kblock_cascade_active ? "GQA6_RPH32_TOTAL192_KBLOCK_CASCADE" : (dbv_gqa6_rph32_active ? "GQA6_RPH32_TOTAL192_QPACK_STAGED" : (dbv_gqa6_active ? "GQA6_BM96" : (dbv_gqa3_rph32_active ? "GQA3_RPH32_TOTAL96" : (dbv_gqa2_active ? "GQA2" :
                        (dbv_direct_row_kv_active ? "DIRECT_ROW_KV" : (dbv_no_logits_init_active ? "NO_LOGITS_INIT" : (dbv_probs_overlay_active ? "PROBS_OVERLAY" : (dbv_single_vbuf_active ? "SINGLE_VBUF" : "LEAN")))))))));
                    fprintf(stderr, "PWMMA DBV %s: enabled impl=%d%s nq=%d nk=%d gqa_ratio=%d grid=(%u,%u,%u) rows_per_head=%d total_rows=%d\n",
                        dbv_route, impl, (dbv_kshared_active || dbv_gqa6_kshared_active) ? " kshared" : "",
                        nq, nk, gqa_ratio, (unsigned) dbv_grid.x, (unsigned) dbv_grid.y, (unsigned) dbv_grid.z,
                        (dbv_kblock_cascade_active || dbv_gqa6_rph32_active || dbv_gqa3_rph32_active) ? 32 : ((dbv_gqa6_active || dbv_gqa6_kshared_active) ? 16 : (dbv_gqa2_active ? (PWMMA_BM64 / 2) : PWMMA_BM64)),
                        (dbv_kblock_cascade_active || dbv_gqa6_rph32_active) ? 192 : ((dbv_gqa3_rph32_active || dbv_gqa6_active || dbv_gqa6_kshared_active) ? 96 : PWMMA_BM64));
                }
            }
            if (ggml_cuda_rocm_env_i32("GGML_CUDA_FA_BLOCK_META_TRACE", 0) != 0 &&
                    (dbv_implicit_flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0) {
                static int block_meta_trace_count = 0;
                const int block_meta_trace_limit = ggml_cuda_rocm_env_i32("GGML_CUDA_FA_BLOCK_META_TRACE_LIMIT", 64);
                if (block_meta_trace_count < block_meta_trace_limit) {
                    block_meta_trace_count++;
                    const int dbv_group = (dbv_kblock_cascade_active || dbv_gqa6_rph32_active || dbv_gqa6_active || dbv_gqa6_kshared_active) ? 6 : (dbv_gqa3_rph32_active ? 3 : (dbv_gqa2_active ? 2 : 1));
                    const int rows_per_head = (dbv_kblock_cascade_active || dbv_gqa6_rph32_active || dbv_gqa3_rph32_active) ? 32 : ((dbv_gqa6_active || dbv_gqa6_kshared_active) ? 16 : (dbv_gqa2_active ? (PWMMA_BM64 / 2) : bm));
                    const uint32_t valid_n_kv_u32 = dbv_implicit_n_kv > 0 ? (uint32_t) dbv_implicit_n_kv : 0u;
                    const uint32_t q_tokens_u32 = nq > 0 ? (uint32_t) nq : 0u;
                    const ggml_cuda_fa_block_meta_v1 meta = ggml_cuda_fa_block_meta_make_legacy_causal(
                        valid_n_kv_u32, q_tokens_u32, dbv_implicit_q_offset,
                        (uint32_t) (dbv_implicit_flags & ~(PWMMA_IMPLICIT_FLAG_BLOCK_META_FULL_TILE | PWMMA_IMPLICIT_FLAG_BLOCK_META_SKIP_CLAMP)),
                        (uint32_t) rows_per_head, (uint32_t) PWMMA_BN);
                    ggml_cuda_fa_block_plan_counts_v1 counts = {};
                    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_plan_all_counts(meta, &counts);
                    const unsigned long long cta_groups = (unsigned long long) batch *
                        (unsigned long long) n_heads_k * (unsigned long long) CEIL_DIV(gqa_ratio, dbv_group);
                    fprintf(stderr,
                        "GGML_CUDA_FA_BLOCK_META_TRACE: backend=pwmma_dbv route=%s status=%u node=%s nq=%d nk=%d valid_n_kv=%d q_offset=%d flags=%d "
                        "q_block=%d k_block=%d q_blocks=%u k_blocks=%u head_groups=%llu per_group_full=%u per_group_edge_front=%u "
                        "per_group_edge_back=%u per_group_skipped=%u per_group_padded_refs=%u total_full=%llu total_edge=%llu total_skipped=%llu\n",
                        dbv_gqa6_kshared_active ? "GQA6_BM96_KSHARED" : (dbv_kblock_cascade_active ? "GQA6_RPH32_TOTAL192_KBLOCK_CASCADE" : (dbv_gqa6_rph32_active ? "GQA6_RPH32_TOTAL192_QPACK_STAGED" : (dbv_gqa6_active ? "GQA6_BM96" : (dbv_gqa3_rph32_active ? "GQA3_RPH32_TOTAL96" : (dbv_gqa2_active ? "GQA2" : impl_name))))),
                        (unsigned) st, (dst && dst->name[0]) ? dst->name : "-", nq, nk, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags,
                        rows_per_head, PWMMA_BN, counts.q_blocks, counts.k_blocks, cta_groups,
                        counts.full_tiles, counts.edge_front_tiles, counts.edge_back_tiles, counts.skipped_tiles, counts.padded_tile_refs,
                        (unsigned long long) counts.full_tiles * cta_groups,
                        (unsigned long long) (counts.edge_front_tiles + counts.edge_back_tiles) * cta_groups,
                        (unsigned long long) counts.skipped_tiles * cta_groups);
                }
            }
#define LAUNCH_BM64_I8QK_DBV(VT, PACKED8V, FORCE_KSHAREDV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, PACKED8V, FORCE_KSHAREDV><<<dbv_grid, block512, kshared_smem, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, d_live_shadow_err, profile_dev, \
        partial_m.get(), partial_l.get(), partial_out.get(), streamk_splits, streamk_prefix_tiles, \
        debug_qk_active ? debug_qk_raw.get() : nullptr, \
        debug_qk_active ? debug_qk_masked.get() : nullptr, \
        debug_qk_active ? debug_probs.get() : nullptr, \
        debug_qk_active ? debug_m.get() : nullptr, \
        debug_qk_active ? debug_l.get() : nullptr, \
        debug_qk_active ? debug_alpha.get() : nullptr, \
        debug_qtile, debug_hq, debug_b, debug_kt, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_DIRECT_ROW(VT, PACKED8V) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_row_lean_kernel<VT, PACKED8V, false, false, false, false, false><<<dbv_grid, block512, (PACKED8V ? PWMMA_I8_KSHARED_SMEM_BYTES : 0), stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, \
        d_skip_counter, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_LEAN(VT, FORCE_KSHAREDV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, FORCE_KSHAREDV, false, false, false, false><<<grid, block512, FORCE_KSHAREDV ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_SINGLE_VBUF(VT, FORCE_KSHAREDV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, FORCE_KSHAREDV, false, false, false, false, false><<<grid, block512, FORCE_KSHAREDV ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_PROBS_OVERLAY(VT, FORCE_KSHAREDV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, FORCE_KSHAREDV, false, false, false, false, false, true><<<grid, block512, FORCE_KSHAREDV ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_GQA2(VT) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, false, false, false, false, false, false, false, 2, PWMMA_BM64><<<dbv_grid, block512, 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE_PLAN(VT, PLANV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, false, false, false, false, false, false, true, 6, 192, false, PLANV, false, true><<<dbv_grid, block512, 0, stream>>>( \
        (const float*)Q->data, qpack_view.qpack_payload, qpack_view.qpack_scales, (int) qpack_view.payload_stride_row_i32, (int) qpack_view.scales_stride_row_f32, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE(VT) do { \
    if (dbv_block_meta_full_tile_plan_active) { LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE_PLAN(VT, true); } \
    else { LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE_PLAN(VT, false); } \
} while (0)
#define LAUNCH_BM64_I8QK_DBV_GQA3_RPH32_PLAN(VT, PLANV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, false, false, false, false, false, false, false, 3, 96, false, PLANV><<<dbv_grid, block512, 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_GQA3_RPH32(VT) do { \
    if (dbv_block_meta_full_tile_plan_active) { LAUNCH_BM64_I8QK_DBV_GQA3_RPH32_PLAN(VT, true); } \
    else { LAUNCH_BM64_I8QK_DBV_GQA3_RPH32_PLAN(VT, false); } \
} while (0)
#define LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, PLANV, STREAMV, SKIPCLAMPV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, false, STREAMV, false, false, false, false, false, 6, 96, false, PLANV, false, false, SKIPCLAMPV><<<dbv_grid, block512, 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        partial_m.get(), partial_l.get(), partial_out.get(), streamk_splits, streamk_prefix_tiles, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_GQA6(VT) do { \
    if (dbv_block_meta_full_tile_plan_active) { \
        if (dbv_block_meta_skip_clamp_plan_active) { \
            if (prefix_split_exact_active) { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, true, true, true); } \
            else { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, true, false, true); } \
        } else { \
            if (prefix_split_exact_active) { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, true, true, false); } \
            else { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, true, false, false); } \
        } \
    } else { \
        if (dbv_block_meta_skip_clamp_plan_active) { \
            if (prefix_split_exact_active) { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, false, true, true); } \
            else { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, false, false, true); } \
        } else { \
            if (prefix_split_exact_active) { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, false, true, false); } \
            else { LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM(VT, false, false, false); } \
        } \
    } \
} while (0)
#define LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED_PLAN(VT, PLANV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, true, false, false, false, false, false, false, 6, 96, false, PLANV><<<dbv_grid, block512, PWMMA_I8_KSHARED_SMEM_BYTES, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED(VT) do { \
    if (dbv_block_meta_full_tile_plan_active) { LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED_PLAN(VT, true); } \
    else { LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED_PLAN(VT, false); } \
} while (0)
#define LAUNCH_BM64_I8QK_DBV_NO_LOGITS_INIT(VT, FORCE_KSHAREDV) \
    packed16_wmma_tile_bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev_kernel<VT, false, FORCE_KSHAREDV, false, false, false, false, false, false, 1, PWMMA_BM64, true><<<grid, block512, FORCE_KSHAREDV ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, nullptr, nullptr, 0, 0, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        dbv_mask_data, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, k_payload_row_stride_i32, k_scales_row_stride_half, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, packed_kv_size, attention_scale, dbv_page_map, \
        d_skip_counter, d_block_meta_effect_counts, causal_skip_active, nullptr, nullptr, \
        nullptr, nullptr, nullptr, 1, 0, \
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
        0, 0, 0, 0, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags)
#define LAUNCH_BM64_I8QK_DBV_SELECT(VT) do { \
    if (dbv_gqa2_active) { LAUNCH_BM64_I8QK_DBV_GQA2(VT); } \
    else if (dbv_kblock_cascade_active) { LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE(VT); } \
    else if (dbv_gqa3_rph32_active) { LAUNCH_BM64_I8QK_DBV_GQA3_RPH32(VT); } \
    else if (dbv_gqa6_kshared_active) { LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED(VT); } \
    else if (dbv_gqa6_active) { LAUNCH_BM64_I8QK_DBV_GQA6(VT); } \
    else if (dbv_no_logits_init_active && dbv_kshared_active) { LAUNCH_BM64_I8QK_DBV_NO_LOGITS_INIT(VT, true); } \
    else if (dbv_no_logits_init_active) { LAUNCH_BM64_I8QK_DBV_NO_LOGITS_INIT(VT, false); } \
    else if (dbv_probs_overlay_active && dbv_kshared_active) { LAUNCH_BM64_I8QK_DBV_PROBS_OVERLAY(VT, true); } \
    else if (dbv_probs_overlay_active) { LAUNCH_BM64_I8QK_DBV_PROBS_OVERLAY(VT, false); } \
    else if (dbv_single_vbuf_active && dbv_kshared_active) { LAUNCH_BM64_I8QK_DBV_SINGLE_VBUF(VT, true); } \
    else if (dbv_single_vbuf_active) { LAUNCH_BM64_I8QK_DBV_SINGLE_VBUF(VT, false); } \
    else if (dbv_lean_active && dbv_kshared_active) { LAUNCH_BM64_I8QK_DBV_LEAN(VT, true); } \
    else if (dbv_lean_active) { LAUNCH_BM64_I8QK_DBV_LEAN(VT, false); } \
    else if (dbv_direct_row_kv_active && impl == 17) { LAUNCH_BM64_I8QK_DBV_DIRECT_ROW(VT, true); } \
    else if (dbv_direct_row_kv_active) { LAUNCH_BM64_I8QK_DBV_DIRECT_ROW(VT, false); } \
    else if (impl == 17) { LAUNCH_BM64_I8QK_DBV(VT, true, false); } \
    else if (impl == 19) { LAUNCH_BM64_I8QK_DBV(VT, false, true); } \
    else { LAUNCH_BM64_I8QK_DBV(VT, false, false); } \
} while (0)
            switch (V->type) {
                case GGML_TYPE_Q4_0: LAUNCH_BM64_I8QK_DBV_SELECT(PACKED16_WMMA_V_Q4_0); break;
                case GGML_TYPE_Q8_0: LAUNCH_BM64_I8QK_DBV_SELECT(PACKED16_WMMA_V_Q8_0); break;
                case GGML_TYPE_F16:  LAUNCH_BM64_I8QK_DBV_SELECT(PACKED16_WMMA_V_F16);  break;
                case GGML_TYPE_V4_K16D16_144: LAUNCH_BM64_I8QK_DBV_SELECT(PACKED16_WMMA_V4_K16D16_144); break;
                default: GGML_ABORT("pwmma bm64 i8qk pvwmma dbv: unsupported V type");
            }
#undef LAUNCH_BM64_I8QK_DBV_SELECT
#undef LAUNCH_BM64_I8QK_DBV_NO_LOGITS_INIT
#undef LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED
#undef LAUNCH_BM64_I8QK_DBV_GQA6_KSHARED_PLAN
#undef LAUNCH_BM64_I8QK_DBV_GQA6
#undef LAUNCH_BM64_I8QK_DBV_GQA6_PLAN_STREAM
#undef LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE
#undef LAUNCH_BM64_I8QK_DBV_KBLOCK_CASCADE_PLAN
#undef LAUNCH_BM64_I8QK_DBV_GQA3_RPH32
#undef LAUNCH_BM64_I8QK_DBV_GQA3_RPH32_PLAN
#undef LAUNCH_BM64_I8QK_DBV_GQA2
#undef LAUNCH_BM64_I8QK_DBV_PROBS_OVERLAY
#undef LAUNCH_BM64_I8QK_DBV_SINGLE_VBUF
#undef LAUNCH_BM64_I8QK_DBV_LEAN
#undef LAUNCH_BM64_I8QK_DBV_DIRECT_ROW
#undef LAUNCH_BM64_I8QK_DBV
            if (partial_mode_active) {
                const size_t rows_per_split = size_t(batch) * size_t(nq) * size_t(n_heads_q);
                const size_t merge_elems = rows_per_split * size_t(PWMMA_D);
                const int merge_threads = 256;
                const int merge_blocks = (int)((merge_elems + size_t(merge_threads) - 1) / size_t(merge_threads));
                packed16_wmma_streamk_combine_kernel<<<merge_blocks, merge_threads, 0, stream>>>(
                    partial_m.get(), partial_l.get(), partial_out.get(), (float *) dst->data,
                    (int) rows_per_split, streamk_splits);
            }
            if (debug_qk_active) {
                CUDA_CHECK(hipGetLastError());
                std::vector<float> h_qk_raw(PWMMA_DEBUG_QK_ELEMS);
                std::vector<float> h_qk_masked(PWMMA_DEBUG_QK_ELEMS);
                std::vector<float> h_probs(PWMMA_DEBUG_QK_ELEMS);
                std::vector<float> h_m(PWMMA_DEBUG_ROW_ELEMS);
                std::vector<float> h_l(PWMMA_DEBUG_ROW_ELEMS);
                std::vector<float> h_alpha(PWMMA_DEBUG_ROW_ELEMS);
                CUDA_CHECK(hipMemcpyAsync(h_qk_raw.data(),    debug_qk_raw.get(),    h_qk_raw.size()    * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipMemcpyAsync(h_qk_masked.data(), debug_qk_masked.get(), h_qk_masked.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipMemcpyAsync(h_probs.data(),     debug_probs.get(),     h_probs.size()     * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipMemcpyAsync(h_m.data(),         debug_m.get(),         h_m.size()         * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipMemcpyAsync(h_l.data(),         debug_l.get(),         h_l.size()         * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipMemcpyAsync(h_alpha.data(),     debug_alpha.get(),     h_alpha.size()     * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipStreamSynchronize(stream));
                const char * dump_env = getenv("GGML_CUDA_ROCM_PACKED16_FA_DEBUG_DUMP");
                const std::string base = (dump_env && *dump_env) ? dump_env : "/tmp/packed16-fa-debug";
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".qk_raw.f32",    h_qk_raw.data(),    h_qk_raw.size());
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".qk_masked.f32", h_qk_masked.data(), h_qk_masked.size());
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".probs.f32",     h_probs.data(),     h_probs.size());
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".m.f32",         h_m.data(),         h_m.size());
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".l.f32",         h_l.data(),         h_l.size());
                ggml_cuda_rocm_packed16_debug_write_f32(base + ".alpha.f32",     h_alpha.data(),     h_alpha.size());
                FILE * meta = fopen((base + ".meta").c_str(), "w");
                if (meta) {
                    fprintf(meta, "node=%s\nimpl=%s\nshape_qk=64,16\nshape_row=64\nnq=%d\nnk=%d\nhead=%d\nbatch=%d\nqtile=%d\nkt=%d\nq0=%d\nk0=%d\nraw_hash=%016llx\nmasked_hash=%016llx\nprobs_hash=%016llx\n",
                        dst->name, impl_name, nq, nk, debug_hq, debug_b, debug_qtile, debug_kt,
                        debug_qtile * PWMMA_BM64, debug_kt * PWMMA_BN,
                        (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_qk_raw.data(), h_qk_raw.size()),
                        (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_qk_masked.data(), h_qk_masked.size()),
                        (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_probs.data(), h_probs.size()));
                    fclose(meta);
                }
                fprintf(stderr, "PWMMA DEBUG QK: wrote %s.{qk_raw,qk_masked,probs,m,l,alpha}.f32 raw_hash=%016llx masked_hash=%016llx probs_hash=%016llx first_raw=%g first_masked=%g first_prob=%g\n",
                    base.c_str(),
                    (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_qk_raw.data(), h_qk_raw.size()),
                    (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_qk_masked.data(), h_qk_masked.size()),
                    (unsigned long long) ggml_cuda_rocm_packed16_debug_fnv1a64(h_probs.data(), h_probs.size()),
                    (double) h_qk_raw[0], (double) h_qk_masked[0], (double) h_probs[0]);
            }
        } else if (impl_i8qk) {
            dim3 block512(512);
#define LAUNCH_BM64_I8QK_WG_SV(VT, P16, K32, KSHARED, PVWMMA, BN32V) \
    packed16_wmma_tile_bm64_i8qk_512t_wavegate_stagev_kernel<VT, P16, K32, KSHARED, PVWMMA, BN32V><<<grid, block512, KSHARED ? PWMMA_I8_KSHARED_SMEM_BYTES : 0, stream>>>( \
        (const float*)Q->data, (const char*)V->data, (float*)dst->data, \
        Q->nb[1], Q->nb[2], Q->nb[3], V->nb[0], V->nb[1], V->nb[2], V->nb[3], v_ne13, \
        v_layout, \
        mask ? (const char*)mask->data : nullptr, mask_ne00, mask_ne01, mask_ne03, mask_nb00, mask_nb01, mask_nb03, \
        k_payload, k_scales, packed16_desc, \
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
            case GGML_TYPE_V4_K16D16_144: LAUNCH_BM32_REGOUT_DV(PACKED16_WMMA_V4_K16D16_144); break;
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
            case GGML_TYPE_V4_K16D16_144: LAUNCH_BM16(PACKED16_WMMA_V4_K16D16_144); break;
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
    const float fa_timing_ms = ggml_cuda_rocm_packed16_wmma_timing_end(fa_timing, stream);
    if (fa_timing_ms >= 0.0f) {
        fprintf(stderr,
            "PACKED16_TIMING_TRACE phase=fa_pwmma ms=%.3f call=%d node=%s impl=%s impl_id=%d nq=%d nk=%d hq=%d hk=%d gqa=%d batch=%d desc_layout=%d desc_scale_layout=%d vtype=%s bm=%d dbv=%d kshared=%d\n",
            (double) fa_timing_ms,
            this_call_id,
            (dst && dst->name[0]) ? dst->name : "-",
            impl_name,
            impl,
            nq,
            nk,
            n_heads_q,
            n_heads_k,
            gqa_ratio,
            batch,
            (int) packed16_desc.layout_kind,
            (int) packed16_desc.scale_layout,
            ggml_type_name(V->type),
            bm,
            impl_dbv ? 1 : 0,
            (impl == 17 || impl == 19 || ggml_cuda_rocm_env_i32("GGML_CUDA_ROCM_PACKED16_DBV_GQA6_KSHARED", 0) != 0) ? 1 : 0);
    }
    if (sync_debug) {
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
            GGML_ABORT("PBWMMA live shadow FAILED: code=%d variant=%d block=(%d,%d,%d) thread=%d nq=%d nk=%d hq=%d hk=%d k0=%d k=%d d=%d head_stride=%d packed_rows=%d k_row=%d got=%g ref=%g",
                h_err.code, h_err.variant, h_err.block_x, h_err.block_y, h_err.block_z, h_err.thread_x,
                h_err.nq, h_err.nk, h_err.hq, h_err.hk, h_err.k0, h_err.k, h_err.d,
                h_err.head_stride, h_err.packed_rows, h_err.k_row, (double) h_err.got, (double) h_err.ref);
        }
        fprintf(stderr, "%s\n", live_pv_shadow_requested ? "PBWMMA PV WMMA live shadow PASSED" : "PBWMMA I8 live DOT4 shadow PASSED");
    }

    if ((block_meta_effect_stats || work_trace) && d_block_meta_effect_counts) {
        CUDA_CHECK(hipStreamSynchronize(stream));
        unsigned long long h_counts[4] = {0, 0, 0, 0};
        CUDA_CHECK(hipMemcpy(h_counts, d_block_meta_effect_counts, 4 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipFree(d_block_meta_effect_counts));
        if (block_meta_effect_stats) {
            fprintf(stderr,
                "GGML_CUDA_FA_BLOCK_META_EFFECT: backend=pwmma_dbv route=%s processed_tiles=%llu full_tile_bypass=%llu edge_or_masked_tiles=%llu skipped_tiles=%llu nq=%d nk=%d flags=%d\n",
                impl_name,
                (unsigned long long) h_counts[0],
                (unsigned long long) h_counts[1],
                (unsigned long long) h_counts[2],
                (unsigned long long) h_counts[3],
                nq, nk, dbv_implicit_flags);
        }
        if (work_trace) {
            const int bm_trace = is_bm64 ? 64 : (is_bm32 ? 32 : 16);
            const int bn_trace = impl_dbv ? 64 : PWMMA_BN;
            const long long q_tiles = (long long) CEIL_DIV(nq, bm_trace);
            const long long k_tiles = (long long) CEIL_DIV(nk, bn_trace);
            const long long active_tiles = (long long) h_counts[0];
            const long long candidate_tiles = active_tiles + (long long) h_counts[3];
            fprintf(stderr,
                "GGML_CUDA_ROCM_PACKED16_WMMA_WORK_TRACE: backend=pwmma_dbv route=%s "
                "nq=%d nk=%d valid_n_kv=%d q_offset=%d flags=%d bm=%d bn=%d q_tiles=%lld k_tiles=%lld "
                "candidate_tiles=%lld processed_tiles=%llu full_tile_bypass=%llu edge_or_masked_tiles=%llu skipped_tiles=%llu\n",
                impl_name, nq, nk, dbv_implicit_n_kv, dbv_implicit_q_offset, dbv_implicit_flags,
                bm_trace, bn_trace, q_tiles, k_tiles, candidate_tiles,
                (unsigned long long) h_counts[0],
                (unsigned long long) h_counts[1],
                (unsigned long long) h_counts[2],
                (unsigned long long) h_counts[3]);
        }
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
    if (pwmma_log) {
        fprintf(stderr, "PWMMA EXIT OK\n");
        fflush(stderr);
    }
}

#else
static void ggml_cuda_flash_attn_ext_packed16_wmma_tile(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("packed16_wmma_tile requires HIP + GGML_HIP_ROCWMMA_FATTN");
}
#endif
