#pragma once

#include "common.cuh"
#include "dot4-packed16/dp16-packed-i8-desc.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

struct ggml_backend_cuda_context;
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_op_pack_v4_k16d16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_op_pack_v4_k16d16_144(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

static constexpr int GGML_CUDA_PACKED16_K_LAYOUT_UNKNOWN    = -1;
static constexpr int GGML_CUDA_PACKED16_K_LAYOUT_ROW        = 0;
static constexpr int GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR = 1;

static constexpr uint32_t GGML_CUDA_PACKED16_K_FORMAT_VERSION = 1;
static constexpr uint32_t GGML_CUDA_PDMQ_K_FORMAT_VERSION = 2;
static constexpr int GGML_CUDA_PACKED16_K_SCALE_LAYOUT_UNKNOWN       = -1;
static constexpr int GGML_CUDA_PACKED16_K_SCALE_LAYOUT_ROW           = 0;
static constexpr int GGML_CUDA_PACKED16_K_SCALE_LAYOUT_QBLOCK_PLANAR = 1;

enum ggml_cuda_pdmq_k_format : uint8_t {
    GGML_CUDA_PDMQ_K_FORMAT_NONE             = 0,
    GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272  = 1,
    GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144   = 2,
    GGML_CUDA_PDMQ_K_FORMAT_PACKED4_Q2_80_V1 = 3,
};

static inline const char * ggml_cuda_pdmq_k_format_name(const int format) {
    switch (format) {
        case GGML_CUDA_PDMQ_K_FORMAT_NONE:             return "none";
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272:  return "packed16_q8_272";
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144:   return "packed8_q4_144";
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED4_Q2_80_V1: return "packed4_q2_80_v1";
        default:                                       return "unknown";
    }
}

static inline uint32_t ggml_cuda_pdmq_k_code_bits(const int format) {
    switch (format) {
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272:  return 8;
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144:   return 4;
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED4_Q2_80_V1: return 2;
        default:                                       return 0;
    }
}

static inline uint32_t ggml_cuda_pdmq_k_zero_point(const int format) {
    switch (format) {
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272:  return 0;
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144:   return 8;
        case GGML_CUDA_PDMQ_K_FORMAT_PACKED4_Q2_80_V1: return 2;
        default:                                       return 0;
    }
}

static inline uint32_t ggml_cuda_pdmq_k_payload_words_per_token(const int format, const uint32_t d) {
    const uint32_t bits = ggml_cuda_pdmq_k_code_bits(format);
    return bits ? (d * bits) / 32u : 0u;
}

#ifndef GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_V1_DEFINED
#define GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_V1_DEFINED
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_VERSION = 1;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES = 4;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY = 1u << 0;

struct ggml_cuda_mtp_qblock_tail_page_map_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t active = 0;
    uint32_t flags = 0;
    uint32_t logical_base_token = 0;
    uint32_t valid_tail_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    int32_t block_table[GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES] = {};
    uint64_t generation = 0;
};
#endif

#ifndef GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_V1_DEFINED
#define GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_V1_DEFINED
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_VERSION = 1;
static constexpr uint32_t GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_NODE_NAME_MAX = 96;
struct ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t active = 0;
    uint32_t reserved = 0;
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint64_t bind_count = 0;
    int32_t layer = -1;
    int32_t graph_inst = -1;
    int32_t nk = 0;
    char node_name[GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_NODE_NAME_MAX] = {};
};
#endif

struct ggml_cuda_packed16_sidecar_meta {
    uint32_t format_version = 0;
    uint64_t generation = 0;
    uint32_t kv_capacity = 0;
    uint16_t d = 0;
    uint16_t block_size = 0;
    int32_t layout_kind = GGML_CUDA_PACKED16_K_LAYOUT_UNKNOWN;
    int32_t scale_layout = GGML_CUDA_PACKED16_K_SCALE_LAYOUT_UNKNOWN;
    size_t payload_head_stride = 0;
    size_t scale_head_stride = 0;
    size_t payload_token_stride = 0;
    size_t scale_token_stride = 0;
    size_t payload_d16_plane_stride = 0;
    size_t scale_qblock_plane_stride = 0;

    uint8_t k_format = GGML_CUDA_PDMQ_K_FORMAT_NONE;
    uint8_t code_bits = 0;
    uint8_t zero_point = 0;
    uint8_t code_order = 0;
    uint32_t payload_words_per_token = 0;

    dp16_packed_i8_desc_v1 packed_i8_desc = {};
};

extern "C" {
void llama_kv_cache_register_packed16_with_layout_info(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales, int layout_kind, uint32_t kv_capacity, uint32_t d);
void llama_kv_cache_register_pdmq_k_with_layout_info(const void * k_view_data, struct ggml_tensor * payload, struct ggml_tensor * scales, int k_format, int layout_kind, uint32_t kv_capacity, uint32_t d);
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, struct ggml_tensor ** payload, struct ggml_tensor ** scales);
void llama_kv_cache_get_packed16_metadata(const void * k_view_data, int * layout_kind, unsigned long long * generation);
void llama_kv_cache_get_packed16_sidecar_meta(const void * k_view_data, struct ggml_cuda_packed16_sidecar_meta * meta);
void llama_kv_cache_get_pdmq_k_sidecar_meta(const void * k_view_data, struct ggml_cuda_packed16_sidecar_meta * meta);
void llama_kv_cache_get_packed16_packed_i8_desc(const void * k_view_data, struct dp16_packed_i8_desc_v1 * desc);
void llama_kv_cache_get_packed16_shadow_k(const void * k_view_data, struct ggml_tensor ** shadow_k);
void llama_kv_cache_register_mtp_qblock_tail_page_map(const void * k_view_data, const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
void llama_kv_cache_clear_mtp_qblock_tail_page_map(const void * k_view_data);
void llama_kv_cache_get_mtp_qblock_tail_page_map(const void * k_view_data, struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
bool llama_kv_cache_get_mtp_qblock_tail_page_published_map(struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map);
void llama_kv_cache_record_mtp_qblock_tail_page_dispatch_bind(const void * k_view_data, const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map, const char * node_name, int layer, int graph_inst, int nk);
void llama_kv_cache_note_mtp_qblock_tail_page_pending_dispatch_bind(const struct ggml_cuda_mtp_qblock_tail_page_map_v1 * map, uint64_t req_begin, uint64_t req_end, int slot);
bool llama_kv_cache_get_mtp_qblock_tail_page_last_dispatch_bind(struct ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 * out);
void llama_kv_cache_record_mtp_qblock_tail_page_consumer_miss(uint32_t nk, const char * reason);
uint32_t llama_kv_cache_get_mtp_qblock_tail_page_consumer_no_map_nk_max(void);
void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle(bool data_invalidates);
void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle_preserve_snapshot(bool data_invalidates, bool keep_producer_snapshot);
bool llama_kv_cache_mtp_qblock_tail_page_published_scratch_map_covers(uint32_t logical_base, uint32_t n_tokens);
}

static constexpr int GGML_CUDA_PACKED16_K_TILE_D = 256;
static constexpr int GGML_CUDA_PACKED16_K_WORDS = GGML_CUDA_PACKED16_K_TILE_D / 4;
static constexpr int GGML_CUDA_PACKED16_K_QBLOCKS = GGML_CUDA_PACKED16_K_TILE_D / QK8_0;
static constexpr int GGML_CUDA_PACKED16_K_D16 = 16;
static constexpr int GGML_CUDA_PACKED16_K_WORDS_PER_D16 = GGML_CUDA_PACKED16_K_D16 / 4;

static inline int ggml_cuda_packed16_k_layout_kind_from_env() {
    const char * layout = getenv("GGML_CUDA_ROCM_PACKED16_K_LAYOUT");
    if (layout && (strcmp(layout, "tile16") == 0 || strcmp(layout, "native") == 0 || strcmp(layout, "v2") == 0 || strcmp(layout, "d16_planar") == 0)) {
        return GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR;
    }
    const char * native = getenv("GGML_CUDA_ROCM_PACKED16_NATIVE_K");
    return native && atoi(native) != 0 ? GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR : GGML_CUDA_PACKED16_K_LAYOUT_ROW;
}

static inline bool ggml_cuda_packed16_k_layout_kind_tile16(const int layout_kind) {
    return layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR;
}

static inline const char * ggml_cuda_packed16_k_layout_kind_name(const int layout_kind) {
    switch (layout_kind) {
        case GGML_CUDA_PACKED16_K_LAYOUT_ROW:        return "row";
        case GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR: return "d16_planar";
        default:                                     return "unknown";
    }
}

static inline bool ggml_cuda_packed16_k_layout_tile16() {
    return ggml_cuda_packed16_k_layout_kind_tile16(ggml_cuda_packed16_k_layout_kind_from_env());
}

static inline const char * ggml_cuda_packed16_k_layout_name() {
    return ggml_cuda_packed16_k_layout_tile16() ? "tile16" : "row";
}

static __host__ __device__ __forceinline__ size_t ggml_cuda_packed16_k_payload_index(
        const size_t head_base_rows,
        const int head_stride_rows,
        const int k,
        const int d_word,
        const bool tile16) {
    if (!tile16) {
        return (head_base_rows + size_t(k)) * size_t(GGML_CUDA_PACKED16_K_WORDS) + size_t(d_word);
    }
    // In-place native layout, no row padding: [head][D16 plane][K row][4xi32 word].
    // This preserves exactly 272 B/token/KV-head while making each D16 plane a
    // contiguous 16B packed-vector scratch plane over K rows.
    const int d16 = d_word / GGML_CUDA_PACKED16_K_WORDS_PER_D16;
    const int w4  = d_word - d16 * GGML_CUDA_PACKED16_K_WORDS_PER_D16;
    return head_base_rows * size_t(GGML_CUDA_PACKED16_K_WORDS)
        + size_t(d16) * size_t(head_stride_rows) * size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16)
        + size_t(k) * size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16)
        + size_t(w4);
}

static __host__ __device__ __forceinline__ size_t ggml_cuda_packed16_k_scale_index(
        const size_t head_base_rows,
        const int head_stride_rows,
        const int k,
        const int qblock,
        const bool tile16) {
    if (!tile16) {
        return (head_base_rows + size_t(k)) * size_t(GGML_CUDA_PACKED16_K_QBLOCKS) + size_t(qblock);
    }
    return head_base_rows * size_t(GGML_CUDA_PACKED16_K_QBLOCKS)
        + size_t(qblock) * size_t(head_stride_rows)
        + size_t(k);
}

static __host__ __device__ __forceinline__ size_t ggml_cuda_packed16_k_payload_index_from_desc(
        const dp16_packed_i8_desc_v1 & desc,
        const uint32_t head,
        const uint32_t token,
        const uint32_t d_word) {
    const uint32_t d16 = d_word / GGML_CUDA_PACKED16_K_WORDS_PER_D16;
    const uint32_t word = d_word - d16 * GGML_CUDA_PACKED16_K_WORDS_PER_D16;
    return dp16_packed_i8_payload_word_index(desc, head, token, d16, word);
}

static __host__ __device__ __forceinline__ size_t ggml_cuda_packed16_k_scale_index_from_desc(
        const dp16_packed_i8_desc_v1 & desc,
        const uint32_t head,
        const uint32_t token,
        const uint32_t qblock) {
    return (size_t) (dp16_packed_i8_scale_byte_offset(desc, head, token, qblock) / sizeof(uint16_t));
}

#ifdef GGML_USE_HIP

// The old q8k DOT4 attention route was a lab/probe path and is intentionally
// demoted. Keep these helpers as disabled stubs so legacy selection code and
// route-contract checks compile, while packed16 K packing/registry remains live.
static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(ggml_fattn_instruction inst) {
    GGML_UNUSED(inst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_allow_source_q4_0() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_legal_kv(const ggml_tensor * K, const ggml_tensor * V) {
    GGML_UNUSED(K);
    GGML_UNUSED(V);
    return false;
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(ggml_fattn_instruction inst) {
    GGML_UNUSED(inst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_allow_source_q4_0() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_legal_kv(const ggml_tensor * K, const ggml_tensor * V) {
    GGML_UNUSED(K);
    GGML_UNUSED(V);
    return false;
}

inline void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("q8k DOT4 attention route has been removed");
}

#endif // GGML_USE_HIP
