// fa-block-meta.cuh — versioned FlashAttention block/page metadata ABI
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_CUDA_FA_BLOCK_META_HD __host__ __device__ __forceinline__
#else
#define GGML_CUDA_FA_BLOCK_META_HD inline
#endif

static constexpr uint32_t GGML_CUDA_FA_BLOCK_META_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_PLAN_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_META_CAPS_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_PLAN_COUNTS_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_TILE_DESC_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_TILE_DESC_COUNTS_VERSION = 1;
static constexpr uint32_t GGML_CUDA_FA_BLOCK_INVALID = 0xffffffffu;

enum ggml_cuda_fa_block_meta_mode : uint32_t {
    GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK     = 0,
    GGML_CUDA_FA_BLOCK_META_MODE_PURE_CAUSAL_CONTIG = 1,
    GGML_CUDA_FA_BLOCK_META_MODE_VARLEN_CAUSAL      = 2,
    GGML_CUDA_FA_BLOCK_META_MODE_PAGED_CAUSAL       = 3,
    GGML_CUDA_FA_BLOCK_META_MODE_PREFIX_PLUS_EXTEND = 4,
};

enum ggml_cuda_fa_block_meta_flags : uint32_t {
    GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL               = 1u << 0,
    GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE      = 1u << 1,
    GGML_CUDA_FA_BLOCK_META_FLAG_HAS_KV_INDPTR        = 1u << 2,
    GGML_CUDA_FA_BLOCK_META_FLAG_HAS_KV_INDICES       = 1u << 3,
    GGML_CUDA_FA_BLOCK_META_FLAG_HAS_LAST_PAGE_TOKENS = 1u << 4,
    GGML_CUDA_FA_BLOCK_META_FLAG_PREFIX_CONTEXT       = 1u << 5,
};

enum ggml_cuda_fa_block_page_kind : uint32_t {
    GGML_CUDA_FA_BLOCK_PAGE_KIND_NONE        = 0,
    GGML_CUDA_FA_BLOCK_PAGE_KIND_BLOCK_TABLE = 1,
    GGML_CUDA_FA_BLOCK_PAGE_KIND_CSR         = 2,
};

enum ggml_cuda_fa_block_page_kind_bits : uint32_t {
    GGML_CUDA_FA_BLOCK_PAGE_KIND_BIT_NONE        = 1u << GGML_CUDA_FA_BLOCK_PAGE_KIND_NONE,
    GGML_CUDA_FA_BLOCK_PAGE_KIND_BIT_BLOCK_TABLE = 1u << GGML_CUDA_FA_BLOCK_PAGE_KIND_BLOCK_TABLE,
    GGML_CUDA_FA_BLOCK_PAGE_KIND_BIT_CSR         = 1u << GGML_CUDA_FA_BLOCK_PAGE_KIND_CSR,
};

enum ggml_cuda_fa_block_meta_status : uint32_t {
    GGML_CUDA_FA_BLOCK_META_OK = 0,
    GGML_CUDA_FA_BLOCK_META_BAD_VERSION,
    GGML_CUDA_FA_BLOCK_META_BAD_ABI_BYTES,
    GGML_CUDA_FA_BLOCK_META_BAD_MODE,
    GGML_CUDA_FA_BLOCK_META_MISSING_CAUSAL,
    GGML_CUDA_FA_BLOCK_META_BAD_Q_TOKENS,
    GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS,
    GGML_CUDA_FA_BLOCK_META_BAD_PAGE_KIND,
    GGML_CUDA_FA_BLOCK_META_BAD_PAGE_TOKENS,
    GGML_CUDA_FA_BLOCK_META_BAD_LOGICAL_TOKENS,
    GGML_CUDA_FA_BLOCK_META_MISSING_BLOCK_TABLE,
    GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TABLE_PAGES,
    GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGES,
    GGML_CUDA_FA_BLOCK_META_BAD_LAST_PAGE_TOKENS,
    GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGE,
    GGML_CUDA_FA_BLOCK_META_DUPLICATE_PHYSICAL_PAGE,
    GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE,
    GGML_CUDA_FA_BLOCK_META_BAD_Q_RANGE,
    GGML_CUDA_FA_BLOCK_META_BAD_CAPS_VERSION,
    GGML_CUDA_FA_BLOCK_META_BAD_CAPS_ABI_BYTES,
    GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_MODE,
    GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_FLAGS,
    GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_PAGE_KIND,
    GGML_CUDA_FA_BLOCK_META_BELOW_MIN_N_KV,
};

struct ggml_cuda_fa_block_meta_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t flags = 0;
    uint32_t mode = GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK;

    // Logical attention shape. q_offset is the logical K offset for Q row 0:
    // visible causal K for row q is k <= q_offset + q.
    uint32_t valid_n_kv = 0;
    uint32_t q_tokens = 0;
    int32_t  q_offset = 0;
    uint32_t batch = 0;

    // Planner granularity. k_block_tokens is the K tile/page width used for
    // full/edge/skip classification. q_block_tokens is advisory and normally
    // matches the backend row tile.
    uint32_t q_block_tokens = 0;
    uint32_t k_block_tokens = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;

    // Optional physical page/block mapping. Causal semantics are always in the
    // logical K coordinate space; page mapping only translates logical K to a
    // physical K/V cache page and page slot.
    uint32_t page_kind = GGML_CUDA_FA_BLOCK_PAGE_KIND_NONE;
    uint32_t page_tokens = 0;
    uint32_t logical_base_token = 0;
    uint32_t logical_tokens = 0;

    const int32_t * block_table = nullptr;
    uint32_t block_table_pages = 0;
    uint32_t physical_pages = 0;
    uint32_t last_page_tokens = 0;

    const int32_t * kv_indptr = nullptr;
    const int32_t * kv_indices = nullptr;
    uint64_t reserved2 = 0;
    uint64_t reserved3 = 0;
};

struct ggml_cuda_fa_block_plan_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t flags = 0;
    uint32_t mode = 0;

    uint32_t q_begin = 0;
    uint32_t q_end = 0;
    uint32_t k_block_begin = 0;
    uint32_t k_block_end = 0;

    // Half-open K-block ranges. Pure causal contiguous uses only full/back-edge/skip.
    uint32_t full_begin = 0;
    uint32_t full_end = 0;
    uint32_t edge_front_begin = 0;
    uint32_t edge_front_end = 0;
    uint32_t edge_back_begin = 0;
    uint32_t edge_back_end = 0;
    uint32_t skipped_begin = 0;
    uint32_t skipped_end = 0;

    uint32_t padded_last_block = GGML_CUDA_FA_BLOCK_INVALID;
    uint32_t reserved0 = 0;
};

struct ggml_cuda_fa_block_meta_caps_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t supported_modes = 0;
    uint32_t supported_flags = 0;
    uint32_t page_kinds = GGML_CUDA_FA_BLOCK_PAGE_KIND_BIT_NONE;
    uint32_t min_n_kv_for_null_mask = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

struct ggml_cuda_fa_block_plan_counts_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t flags = 0;
    uint32_t mode = 0;

    uint32_t q_block_tokens = 0;
    uint32_t k_block_tokens = 0;
    uint32_t q_blocks = 0;
    uint32_t k_blocks = 0;

    uint32_t full_tiles = 0;
    uint32_t edge_front_tiles = 0;
    uint32_t edge_back_tiles = 0;
    uint32_t skipped_tiles = 0;
    uint32_t padded_tile_refs = 0;
    uint32_t reserved0 = 0;
};

enum ggml_cuda_fa_block_tile_class : uint32_t {
    GGML_CUDA_FA_BLOCK_TILE_INVALID = 0,
    GGML_CUDA_FA_BLOCK_TILE_FULL    = 1,
    GGML_CUDA_FA_BLOCK_TILE_EDGE    = 2,
    GGML_CUDA_FA_BLOCK_TILE_SKIPPED = 3,
};

struct ggml_cuda_fa_block_tile_desc_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t tile_class = GGML_CUDA_FA_BLOCK_TILE_INVALID;
    uint32_t flags = 0;

    uint32_t q_begin = 0;
    uint32_t q_rows = 0;
    uint32_t logical_k_begin = 0;
    uint32_t k_rows = 0;

    uint32_t physical_k_begin = GGML_CUDA_FA_BLOCK_INVALID;
    uint32_t physical_page = GGML_CUDA_FA_BLOCK_INVALID;
    uint32_t page_offset = GGML_CUDA_FA_BLOCK_INVALID;
    uint32_t page_valid_tokens = 0;
};

struct ggml_cuda_fa_block_tile_desc_counts_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t described_tiles = 0;
    uint32_t full_tiles = 0;
    uint32_t edge_tiles = 0;
    uint32_t skipped_tiles = 0;
    uint32_t physical_tiles = 0;
    uint32_t physical_rejects = 0;
};

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_div_round_up(
        const uint32_t n,
        const uint32_t d) {
    return d == 0 ? 0 : (n + d - 1u) / d;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_v1 ggml_cuda_fa_block_meta_make_pure_causal_contig(
        const uint32_t valid_n_kv,
        const uint32_t q_tokens,
        const int32_t  q_offset,
        const uint32_t q_block_tokens,
        const uint32_t k_block_tokens) {
    ggml_cuda_fa_block_meta_v1 meta = {};
    meta.version = GGML_CUDA_FA_BLOCK_META_VERSION;
    meta.abi_bytes = sizeof(ggml_cuda_fa_block_meta_v1);
    meta.flags = GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL;
    meta.mode = GGML_CUDA_FA_BLOCK_META_MODE_PURE_CAUSAL_CONTIG;
    meta.valid_n_kv = valid_n_kv;
    meta.q_tokens = q_tokens;
    meta.q_offset = q_offset;
    meta.q_block_tokens = q_block_tokens;
    meta.k_block_tokens = k_block_tokens;
    return meta;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_v1 ggml_cuda_fa_block_meta_make_legacy_causal(
        const uint32_t valid_n_kv,
        const uint32_t q_tokens,
        const int32_t  q_offset,
        const uint32_t flags,
        const uint32_t q_block_tokens,
        const uint32_t k_block_tokens) {
    ggml_cuda_fa_block_meta_v1 meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(
        valid_n_kv, q_tokens, q_offset, q_block_tokens, k_block_tokens);
    meta.flags = flags;
    meta.mode = (flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) != 0 ?
        GGML_CUDA_FA_BLOCK_META_MODE_PURE_CAUSAL_CONTIG :
        GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK;
    return meta;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_validate_static(
        const ggml_cuda_fa_block_meta_v1 & meta) {
    if (meta.version != GGML_CUDA_FA_BLOCK_META_VERSION) {
        return GGML_CUDA_FA_BLOCK_META_BAD_VERSION;
    }
    if (meta.abi_bytes != sizeof(ggml_cuda_fa_block_meta_v1)) {
        return GGML_CUDA_FA_BLOCK_META_BAD_ABI_BYTES;
    }
    if (meta.mode > GGML_CUDA_FA_BLOCK_META_MODE_PREFIX_PLUS_EXTEND) {
        return GGML_CUDA_FA_BLOCK_META_BAD_MODE;
    }
    if (meta.mode != GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK &&
            (meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) == 0) {
        return GGML_CUDA_FA_BLOCK_META_MISSING_CAUSAL;
    }
    if (meta.mode != GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK && meta.q_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_Q_TOKENS;
    }
    if (meta.mode != GGML_CUDA_FA_BLOCK_META_MODE_DENSE_FALLBACK &&
            (meta.q_block_tokens == 0 || meta.k_block_tokens == 0)) {
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }

    const bool paged = meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PAGED_CAUSAL ||
        meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PREFIX_PLUS_EXTEND ||
        (meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE) != 0;
    if (!paged) {
        return GGML_CUDA_FA_BLOCK_META_OK;
    }

    if (meta.page_kind != GGML_CUDA_FA_BLOCK_PAGE_KIND_BLOCK_TABLE &&
            meta.page_kind != GGML_CUDA_FA_BLOCK_PAGE_KIND_CSR) {
        return GGML_CUDA_FA_BLOCK_META_BAD_PAGE_KIND;
    }
    if (meta.page_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_PAGE_TOKENS;
    }
    if (meta.logical_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_LOGICAL_TOKENS;
    }
    if (meta.page_kind == GGML_CUDA_FA_BLOCK_PAGE_KIND_BLOCK_TABLE ||
            (meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE) != 0) {
        if (meta.block_table == nullptr) {
            return GGML_CUDA_FA_BLOCK_META_MISSING_BLOCK_TABLE;
        }
        const uint32_t expected_pages = ggml_cuda_fa_block_meta_div_round_up(meta.logical_tokens, meta.page_tokens);
        if (meta.block_table_pages < expected_pages) {
            return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TABLE_PAGES;
        }
        if (meta.physical_pages == 0) {
            return GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGES;
        }
        const uint32_t expected_last = (meta.logical_tokens % meta.page_tokens) != 0 ?
            (meta.logical_tokens % meta.page_tokens) : meta.page_tokens;
        if (meta.last_page_tokens != expected_last) {
            return GGML_CUDA_FA_BLOCK_META_BAD_LAST_PAGE_TOKENS;
        }
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_validate_block_table(
        const ggml_cuda_fa_block_meta_v1 & meta) {
    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_validate_static(meta);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (meta.block_table == nullptr || meta.page_tokens == 0 || meta.logical_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_OK;
    }
    const uint32_t pages = ggml_cuda_fa_block_meta_div_round_up(meta.logical_tokens, meta.page_tokens);
    for (uint32_t i = 0; i < pages; ++i) {
        const int32_t pi = meta.block_table[i];
        if (pi < 0 || (uint32_t) pi >= meta.physical_pages) {
            return GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGE;
        }
        for (uint32_t j = i + 1; j < pages; ++j) {
            if (meta.block_table[j] == pi) {
                return GGML_CUDA_FA_BLOCK_META_DUPLICATE_PHYSICAL_PAGE;
            }
        }
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_mode_bit(const uint32_t mode) {
    return mode < 32u ? (1u << mode) : 0u;
}

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_page_kind_bit(const uint32_t page_kind) {
    return page_kind < 32u ? (1u << page_kind) : 0u;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_validate_caps(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const ggml_cuda_fa_block_meta_caps_v1 & caps,
        const bool null_dense_mask) {
    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_validate_static(meta);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (caps.version != GGML_CUDA_FA_BLOCK_META_CAPS_VERSION) {
        return GGML_CUDA_FA_BLOCK_META_BAD_CAPS_VERSION;
    }
    if (caps.abi_bytes != sizeof(ggml_cuda_fa_block_meta_caps_v1)) {
        return GGML_CUDA_FA_BLOCK_META_BAD_CAPS_ABI_BYTES;
    }
    if ((caps.supported_modes & ggml_cuda_fa_block_meta_mode_bit(meta.mode)) == 0) {
        return GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_MODE;
    }
    if ((meta.flags & ~caps.supported_flags) != 0) {
        return GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_FLAGS;
    }
    if ((meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE) != 0 ||
            meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PAGED_CAUSAL ||
            meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PREFIX_PLUS_EXTEND) {
        if ((caps.page_kinds & ggml_cuda_fa_block_meta_page_kind_bit(meta.page_kind)) == 0) {
            return GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_PAGE_KIND;
        }
    }
    if (null_dense_mask && meta.valid_n_kv < caps.min_n_kv_for_null_mask) {
        return GGML_CUDA_FA_BLOCK_META_BELOW_MIN_N_KV;
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_plan_pure_causal(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t q_begin,
        const uint32_t q_rows,
        ggml_cuda_fa_block_plan_v1 * out_plan) {
    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_validate_static(meta);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if ((meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL) == 0 || q_rows == 0 || q_begin >= meta.q_tokens) {
        return GGML_CUDA_FA_BLOCK_META_BAD_Q_RANGE;
    }

    ggml_cuda_fa_block_plan_v1 plan = {};
    plan.version = GGML_CUDA_FA_BLOCK_PLAN_VERSION;
    plan.abi_bytes = sizeof(ggml_cuda_fa_block_plan_v1);
    plan.flags = meta.flags;
    plan.mode = meta.mode;
    plan.q_begin = q_begin;
    const uint32_t q_end = q_begin + q_rows < q_begin ? meta.q_tokens : q_begin + q_rows;
    plan.q_end = q_end < meta.q_tokens ? q_end : meta.q_tokens;
    plan.k_block_begin = 0;
    plan.k_block_end = ggml_cuda_fa_block_meta_div_round_up(meta.valid_n_kv, meta.k_block_tokens);
    plan.padded_last_block = (meta.valid_n_kv != 0 && (meta.valid_n_kv % meta.k_block_tokens) != 0) ?
        (plan.k_block_end - 1u) : GGML_CUDA_FA_BLOCK_INVALID;

    const int64_t q_last = (int64_t) plan.q_end - 1;
    int64_t last_visible = (int64_t) meta.q_offset + q_last;
    if (meta.valid_n_kv == 0 || last_visible < 0) {
        plan.skipped_begin = 0;
        plan.skipped_end = plan.k_block_end;
        if (out_plan) {
            *out_plan = plan;
        }
        return GGML_CUDA_FA_BLOCK_META_OK;
    }
    const int64_t max_k = (int64_t) meta.valid_n_kv - 1;
    if (last_visible > max_k) {
        last_visible = max_k;
    }

    const uint32_t visible_blocks = ggml_cuda_fa_block_meta_div_round_up((uint32_t) last_visible + 1u, meta.k_block_tokens);
    uint32_t edge_blocks = ggml_cuda_fa_block_meta_div_round_up(plan.q_end - plan.q_begin, meta.k_block_tokens) + 1u;
    if (edge_blocks > visible_blocks) {
        edge_blocks = visible_blocks;
    }

    plan.full_begin = 0;
    plan.full_end = visible_blocks - edge_blocks;
    plan.edge_front_begin = 0;
    plan.edge_front_end = 0;
    plan.edge_back_begin = plan.full_end;
    plan.edge_back_end = visible_blocks;
    plan.skipped_begin = visible_blocks;
    plan.skipped_end = plan.k_block_end;

    if (out_plan) {
        *out_plan = plan;
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_sub_u32(
        const uint32_t end,
        const uint32_t begin) {
    return end > begin ? end - begin : 0u;
}

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_add_sat_u32(
        const uint32_t a,
        const uint32_t b) {
    const uint32_t c = a + b;
    return c < a ? GGML_CUDA_FA_BLOCK_INVALID : c;
}

GGML_CUDA_FA_BLOCK_META_HD bool ggml_cuda_fa_block_meta_range_contains(
        const uint32_t begin,
        const uint32_t end,
        const uint32_t value) {
    return begin <= value && value < end;
}

GGML_CUDA_FA_BLOCK_META_HD uint32_t ggml_cuda_fa_block_meta_clip_k_range_to_visible(
        const ggml_cuda_fa_block_plan_v1 & plan,
        const uint32_t range_begin,
        const uint32_t range_end,
        uint32_t * out_clipped_end) {
    uint32_t clipped_end = range_end;
    if (plan.skipped_begin < clipped_end) {
        clipped_end = plan.skipped_begin;
    }
    if (clipped_end < range_begin) {
        clipped_end = range_begin;
    }
    if (out_clipped_end) {
        *out_clipped_end = clipped_end;
    }
    return range_end > clipped_end ? range_end - clipped_end : 0u;
}

GGML_CUDA_FA_BLOCK_META_HD bool ggml_cuda_fa_block_meta_tile_plan_for_exact_k_block(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t q_begin,
        const uint32_t q_rows,
        const uint32_t k_begin,
        const uint32_t k_rows,
        ggml_cuda_fa_block_plan_v1 * out_plan,
        uint32_t * out_k_block) {
    if (q_rows == 0 || k_rows == 0 || meta.k_block_tokens == 0) {
        return false;
    }
    if (k_begin % meta.k_block_tokens != 0 || k_rows != meta.k_block_tokens) {
        return false;
    }
    ggml_cuda_fa_block_plan_v1 plan = {};
    if (ggml_cuda_fa_block_meta_plan_pure_causal(meta, q_begin, q_rows, &plan) != GGML_CUDA_FA_BLOCK_META_OK) {
        return false;
    }
    if (out_plan) {
        *out_plan = plan;
    }
    if (out_k_block) {
        *out_k_block = k_begin / meta.k_block_tokens;
    }
    return true;
}

GGML_CUDA_FA_BLOCK_META_HD bool ggml_cuda_fa_block_meta_tile_is_full_visible(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t q_begin,
        const uint32_t q_rows,
        const uint32_t k_begin,
        const uint32_t k_rows) {
    ggml_cuda_fa_block_plan_v1 plan = {};
    uint32_t k_block = 0;
    if (!ggml_cuda_fa_block_meta_tile_plan_for_exact_k_block(meta, q_begin, q_rows, k_begin, k_rows, &plan, &k_block)) {
        return false;
    }
    return ggml_cuda_fa_block_meta_range_contains(plan.full_begin, plan.full_end, k_block);
}

GGML_CUDA_FA_BLOCK_META_HD bool ggml_cuda_fa_block_meta_tile_is_skipped(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t q_begin,
        const uint32_t q_rows,
        const uint32_t k_begin,
        const uint32_t k_rows) {
    ggml_cuda_fa_block_plan_v1 plan = {};
    uint32_t k_block = 0;
    if (!ggml_cuda_fa_block_meta_tile_plan_for_exact_k_block(meta, q_begin, q_rows, k_begin, k_rows, &plan, &k_block)) {
        return false;
    }
    return ggml_cuda_fa_block_meta_range_contains(plan.skipped_begin, plan.skipped_end, k_block);
}

GGML_CUDA_FA_BLOCK_META_HD void ggml_cuda_fa_block_meta_accum_plan_counts(
        const ggml_cuda_fa_block_plan_v1 & plan,
        ggml_cuda_fa_block_plan_counts_v1 * counts) {
    if (counts == nullptr) {
        return;
    }
    counts->full_tiles = ggml_cuda_fa_block_meta_add_sat_u32(
        counts->full_tiles,
        ggml_cuda_fa_block_meta_sub_u32(plan.full_end, plan.full_begin));
    counts->edge_front_tiles = ggml_cuda_fa_block_meta_add_sat_u32(
        counts->edge_front_tiles,
        ggml_cuda_fa_block_meta_sub_u32(plan.edge_front_end, plan.edge_front_begin));
    counts->edge_back_tiles = ggml_cuda_fa_block_meta_add_sat_u32(
        counts->edge_back_tiles,
        ggml_cuda_fa_block_meta_sub_u32(plan.edge_back_end, plan.edge_back_begin));
    counts->skipped_tiles = ggml_cuda_fa_block_meta_add_sat_u32(
        counts->skipped_tiles,
        ggml_cuda_fa_block_meta_sub_u32(plan.skipped_end, plan.skipped_begin));

    if (plan.padded_last_block != GGML_CUDA_FA_BLOCK_INVALID &&
            (ggml_cuda_fa_block_meta_range_contains(plan.full_begin, plan.full_end, plan.padded_last_block) ||
             ggml_cuda_fa_block_meta_range_contains(plan.edge_front_begin, plan.edge_front_end, plan.padded_last_block) ||
             ggml_cuda_fa_block_meta_range_contains(plan.edge_back_begin, plan.edge_back_end, plan.padded_last_block))) {
        counts->padded_tile_refs = ggml_cuda_fa_block_meta_add_sat_u32(counts->padded_tile_refs, 1u);
    }
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_plan_all_counts(
        const ggml_cuda_fa_block_meta_v1 & meta,
        ggml_cuda_fa_block_plan_counts_v1 * out_counts) {
    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_validate_static(meta);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (out_counts == nullptr || meta.q_tokens == 0 || meta.q_block_tokens == 0 || meta.k_block_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }

    ggml_cuda_fa_block_plan_counts_v1 counts = {};
    counts.version = GGML_CUDA_FA_BLOCK_PLAN_COUNTS_VERSION;
    counts.abi_bytes = sizeof(ggml_cuda_fa_block_plan_counts_v1);
    counts.flags = meta.flags;
    counts.mode = meta.mode;
    counts.q_block_tokens = meta.q_block_tokens;
    counts.k_block_tokens = meta.k_block_tokens;
    counts.q_blocks = ggml_cuda_fa_block_meta_div_round_up(meta.q_tokens, meta.q_block_tokens);
    counts.k_blocks = ggml_cuda_fa_block_meta_div_round_up(meta.valid_n_kv, meta.k_block_tokens);

    for (uint32_t q_begin = 0; q_begin < meta.q_tokens; q_begin += meta.q_block_tokens) {
        const uint32_t rows_left = meta.q_tokens - q_begin;
        const uint32_t q_rows = rows_left < meta.q_block_tokens ? rows_left : meta.q_block_tokens;
        ggml_cuda_fa_block_plan_v1 plan = {};
        const ggml_cuda_fa_block_meta_status pst = ggml_cuda_fa_block_meta_plan_pure_causal(meta, q_begin, q_rows, &plan);
        if (pst != GGML_CUDA_FA_BLOCK_META_OK) {
            return pst;
        }
        ggml_cuda_fa_block_meta_accum_plan_counts(plan, &counts);
    }

    *out_counts = counts;
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_logical_to_physical(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t logical_token,
        uint32_t * out_physical_page,
        uint32_t * out_page_offset,
        uint32_t * out_page_valid_tokens) {
    const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_validate_static(meta);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (meta.block_table == nullptr || meta.page_tokens == 0 || meta.logical_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_MISSING_BLOCK_TABLE;
    }
    if (logical_token < meta.logical_base_token || logical_token >= meta.logical_base_token + meta.logical_tokens) {
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }
    const uint32_t rel = logical_token - meta.logical_base_token;
    const uint32_t lp = rel / meta.page_tokens;
    const uint32_t off = rel - lp * meta.page_tokens;
    const uint32_t pages = ggml_cuda_fa_block_meta_div_round_up(meta.logical_tokens, meta.page_tokens);
    if (lp >= pages || lp >= meta.block_table_pages) {
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }
    const uint32_t valid = (lp + 1u == pages) ? meta.last_page_tokens : meta.page_tokens;
    if (off >= valid) {
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }
    const int32_t pp = meta.block_table[lp];
    if (pp < 0 || (uint32_t) pp >= meta.physical_pages) {
        return GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGE;
    }
    if (out_physical_page) {
        *out_physical_page = (uint32_t) pp;
    }
    if (out_page_offset) {
        *out_page_offset = off;
    }
    if (out_page_valid_tokens) {
        *out_page_valid_tokens = valid;
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t logical_begin,
        const uint32_t tile_tokens,
        uint32_t * out_physical_begin,
        uint32_t * out_physical_page,
        uint32_t * out_page_offset) {
    if (tile_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }
    uint32_t first_page = 0;
    uint32_t first_off = 0;
    uint32_t first_valid = 0;
    ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_logical_to_physical(
        meta, logical_begin, &first_page, &first_off, &first_valid);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (first_off + tile_tokens > first_valid) {
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }
    const uint32_t logical_last = logical_begin + tile_tokens - 1u;
    if (logical_last < logical_begin) {
        return GGML_CUDA_FA_BLOCK_META_BAD_LOGICAL_TOKENS;
    }
    uint32_t last_page = 0;
    uint32_t last_off = 0;
    uint32_t last_valid = 0;
    st = ggml_cuda_fa_block_meta_logical_to_physical(meta, logical_last, &last_page, &last_off, &last_valid);
    if (st != GGML_CUDA_FA_BLOCK_META_OK) {
        return st;
    }
    if (last_page != first_page || last_valid != first_valid || last_off + 1u != first_off + tile_tokens) {
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }
    if (out_physical_begin) {
        *out_physical_begin = first_page * meta.page_tokens + first_off;
    }
    if (out_physical_page) {
        *out_physical_page = first_page;
    }
    if (out_page_offset) {
        *out_page_offset = first_off;
    }
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_describe_logical_tile(
        const ggml_cuda_fa_block_meta_v1 & meta,
        const uint32_t q_begin,
        const uint32_t q_rows,
        const uint32_t logical_k_begin,
        const uint32_t k_rows,
        ggml_cuda_fa_block_tile_desc_v1 * out_desc) {
    if (out_desc == nullptr) {
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }
    ggml_cuda_fa_block_tile_desc_v1 desc = {};
    desc.version = GGML_CUDA_FA_BLOCK_TILE_DESC_VERSION;
    desc.abi_bytes = sizeof(ggml_cuda_fa_block_tile_desc_v1);
    desc.flags = meta.flags;
    desc.q_begin = q_begin;
    desc.q_rows = q_rows;
    desc.logical_k_begin = logical_k_begin;
    desc.k_rows = k_rows;

    ggml_cuda_fa_block_plan_v1 plan = {};
    uint32_t k_block = 0;
    if (!ggml_cuda_fa_block_meta_tile_plan_for_exact_k_block(meta, q_begin, q_rows, logical_k_begin, k_rows, &plan, &k_block)) {
        *out_desc = desc;
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }
    if (ggml_cuda_fa_block_meta_range_contains(plan.skipped_begin, plan.skipped_end, k_block)) {
        desc.tile_class = GGML_CUDA_FA_BLOCK_TILE_SKIPPED;
        *out_desc = desc;
        return GGML_CUDA_FA_BLOCK_META_OK;
    }
    if (ggml_cuda_fa_block_meta_range_contains(plan.full_begin, plan.full_end, k_block)) {
        desc.tile_class = GGML_CUDA_FA_BLOCK_TILE_FULL;
    } else if (ggml_cuda_fa_block_meta_range_contains(plan.edge_front_begin, plan.edge_front_end, k_block) ||
            ggml_cuda_fa_block_meta_range_contains(plan.edge_back_begin, plan.edge_back_end, k_block)) {
        desc.tile_class = GGML_CUDA_FA_BLOCK_TILE_EDGE;
    } else {
        desc.tile_class = GGML_CUDA_FA_BLOCK_TILE_INVALID;
        *out_desc = desc;
        return GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE;
    }

    if ((meta.flags & GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE) != 0 ||
            meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PAGED_CAUSAL ||
            meta.mode == GGML_CUDA_FA_BLOCK_META_MODE_PREFIX_PLUS_EXTEND) {
        uint32_t physical_begin = 0;
        uint32_t physical_page = 0;
        uint32_t page_offset = 0;
        const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(
            meta, logical_k_begin, k_rows, &physical_begin, &physical_page, &page_offset);
        if (st != GGML_CUDA_FA_BLOCK_META_OK) {
            *out_desc = desc;
            return st;
        }
        desc.physical_k_begin = physical_begin;
        desc.physical_page = physical_page;
        desc.page_offset = page_offset;
        uint32_t unused_page = 0;
        uint32_t unused_off = 0;
        uint32_t page_valid = 0;
        const ggml_cuda_fa_block_meta_status vst = ggml_cuda_fa_block_meta_logical_to_physical(
            meta, logical_k_begin, &unused_page, &unused_off, &page_valid);
        if (vst != GGML_CUDA_FA_BLOCK_META_OK) {
            *out_desc = desc;
            return vst;
        }
        desc.page_valid_tokens = page_valid;
    } else {
        desc.physical_k_begin = logical_k_begin;
        desc.physical_page = logical_k_begin / meta.k_block_tokens;
        desc.page_offset = logical_k_begin - desc.physical_page * meta.k_block_tokens;
        desc.page_valid_tokens = meta.k_block_tokens;
    }

    *out_desc = desc;
    return GGML_CUDA_FA_BLOCK_META_OK;
}

GGML_CUDA_FA_BLOCK_META_HD ggml_cuda_fa_block_meta_status ggml_cuda_fa_block_meta_describe_all_counts(
        const ggml_cuda_fa_block_meta_v1 & meta,
        ggml_cuda_fa_block_tile_desc_counts_v1 * out_counts) {
    if (out_counts == nullptr || meta.q_tokens == 0 || meta.q_block_tokens == 0 || meta.k_block_tokens == 0) {
        return GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS;
    }
    ggml_cuda_fa_block_tile_desc_counts_v1 counts = {};
    counts.version = GGML_CUDA_FA_BLOCK_TILE_DESC_COUNTS_VERSION;
    counts.abi_bytes = sizeof(ggml_cuda_fa_block_tile_desc_counts_v1);
    const uint32_t logical_begin = (meta.logical_tokens != 0) ? meta.logical_base_token : 0u;
    const uint32_t logical_end = (meta.logical_tokens != 0) ? (meta.logical_base_token + meta.logical_tokens) : meta.valid_n_kv;
    if (logical_end < logical_begin) {
        return GGML_CUDA_FA_BLOCK_META_BAD_LOGICAL_TOKENS;
    }
    for (uint32_t q_begin = 0; q_begin < meta.q_tokens; q_begin += meta.q_block_tokens) {
        const uint32_t q_rows_left = meta.q_tokens - q_begin;
        const uint32_t q_rows = q_rows_left < meta.q_block_tokens ? q_rows_left : meta.q_block_tokens;
        for (uint32_t k_begin = logical_begin; k_begin < logical_end; k_begin += meta.k_block_tokens) {
            ggml_cuda_fa_block_tile_desc_v1 desc = {};
            const ggml_cuda_fa_block_meta_status st = ggml_cuda_fa_block_meta_describe_logical_tile(
                meta, q_begin, q_rows, k_begin, meta.k_block_tokens, &desc);
            if (st == GGML_CUDA_FA_BLOCK_META_OK) {
                counts.described_tiles = ggml_cuda_fa_block_meta_add_sat_u32(counts.described_tiles, 1u);
                if (desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_FULL) {
                    counts.full_tiles = ggml_cuda_fa_block_meta_add_sat_u32(counts.full_tiles, 1u);
                } else if (desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_EDGE) {
                    counts.edge_tiles = ggml_cuda_fa_block_meta_add_sat_u32(counts.edge_tiles, 1u);
                } else if (desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_SKIPPED) {
                    counts.skipped_tiles = ggml_cuda_fa_block_meta_add_sat_u32(counts.skipped_tiles, 1u);
                }
                if (desc.physical_k_begin != GGML_CUDA_FA_BLOCK_INVALID) {
                    counts.physical_tiles = ggml_cuda_fa_block_meta_add_sat_u32(counts.physical_tiles, 1u);
                }
            } else if (st == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE) {
                counts.physical_rejects = ggml_cuda_fa_block_meta_add_sat_u32(counts.physical_rejects, 1u);
            } else {
                return st;
            }
        }
    }
    *out_counts = counts;
    return GGML_CUDA_FA_BLOCK_META_OK;
}

#undef GGML_CUDA_FA_BLOCK_META_HD
