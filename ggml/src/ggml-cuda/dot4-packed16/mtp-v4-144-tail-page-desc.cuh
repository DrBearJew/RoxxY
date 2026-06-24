// mtp-v4-144-tail-page-desc.cuh — narrow transactional tail-page ABI for QBlock/MTP
#pragma once

#include "dp16-packed-i8-desc.cuh"

#include <cstddef>
#include <cstdint>

// This ABI intentionally describes one narrow path only:
//   K: existing packed16 I32 sidecar, addressed by dp16_packed_i8_desc_v1
//   V: GGML_TYPE_V4_K16D16_144
//   D: 256
//   page: 16 tokens
// The tail-page descriptor is consumed only by proof/diagnostic paths here.
// Runtime dispatch must still perform tensor type checks before constructing it.
// Partial-page producer staging uses a separate descriptor below to avoid mixing
// committed-page visibility with scratch read/merge/write accounting.
static constexpr uint32_t MTP_V4_144_TAIL_PAGE_ABI_VERSION = 1;
static constexpr uint32_t MTP_V4_144_PAGE_TOKENS = 16;
static constexpr uint32_t MTP_V4_144_D = 256;
static constexpr uint32_t MTP_V4_144_D32 = 32;
static constexpr uint32_t MTP_V4_144_D32_BLOCKS = MTP_V4_144_D / MTP_V4_144_D32;
static constexpr uint32_t MTP_V4_144_WORDS_PER_D = 2;
static constexpr uint32_t MTP_V4_144_PAYLOAD_WORDS = MTP_V4_144_D * MTP_V4_144_WORDS_PER_D;
static constexpr uint32_t MTP_V4_144_PAYLOAD_BYTES_PER_PAGE = MTP_V4_144_PAYLOAD_WORDS * sizeof(uint32_t);
static constexpr uint32_t MTP_V4_144_SCALE_BYTES_PER_PAGE = MTP_V4_144_D32_BLOCKS * MTP_V4_144_PAGE_TOKENS * sizeof(uint16_t);
static constexpr uint32_t MTP_V4_144_PAGE_BYTES = MTP_V4_144_PAYLOAD_BYTES_PER_PAGE + MTP_V4_144_SCALE_BYTES_PER_PAGE;
static constexpr uint32_t MTP_V4_144_ROW_BYTES = MTP_V4_144_PAGE_BYTES / MTP_V4_144_PAGE_TOKENS;

static constexpr uint32_t MTP_PACKED16_K_D = 256;
static constexpr uint32_t MTP_PACKED16_K_WORDS = MTP_PACKED16_K_D / 4;
static constexpr uint32_t MTP_PACKED16_K_QBLOCKS = MTP_PACKED16_K_D / 32;
static constexpr uint32_t MTP_PACKED16_K_ROW_BYTES = MTP_PACKED16_K_WORDS * sizeof(uint32_t) + MTP_PACKED16_K_QBLOCKS * sizeof(uint16_t);
static constexpr uint32_t MTP_PACKED16_K_PAGE_BYTES = MTP_PACKED16_K_ROW_BYTES * MTP_V4_144_PAGE_TOKENS;
static constexpr uint32_t MTP_V4_144_KV_PAGE_BYTES = MTP_PACKED16_K_PAGE_BYTES + MTP_V4_144_PAGE_BYTES;

static_assert(MTP_V4_144_PAYLOAD_BYTES_PER_PAGE == 2048, "bad V4_144 payload bytes");
static_assert(MTP_V4_144_SCALE_BYTES_PER_PAGE == 256, "bad V4_144 scale bytes");
static_assert(MTP_V4_144_PAGE_BYTES == 2304, "bad V4_144 page bytes");
static_assert(MTP_V4_144_ROW_BYTES == 144, "bad V4_144 row bytes");
static_assert(MTP_PACKED16_K_ROW_BYTES == 272, "bad packed16 K row bytes");
static_assert(MTP_PACKED16_K_PAGE_BYTES == 4352, "bad packed16 K page bytes");
static_assert(MTP_V4_144_KV_PAGE_BYTES == 6656, "bad packed16 K + V4_144 page bytes");

enum mtp_v4_144_tail_page_flags : uint32_t {
    MTP_V4_144_TAIL_FLAG_TAIL_ONLY     = 1u << 0,
    MTP_V4_144_TAIL_FLAG_BOUNDARY_COPY = 1u << 1,
    MTP_V4_144_TAIL_FLAG_ALIGNED_ONLY  = 1u << 2,
    MTP_V4_144_TAIL_FLAG_DEBUG_ABORTS  = 1u << 3,
};

enum mtp_v4_144_tail_page_status : uint32_t {
    MTP_V4_144_TAIL_PAGE_OK = 0,
    MTP_V4_144_TAIL_PAGE_BAD_VERSION,
    MTP_V4_144_TAIL_PAGE_BAD_ABI_BYTES,
    MTP_V4_144_TAIL_PAGE_MISSING_TAIL_ONLY,
    MTP_V4_144_TAIL_PAGE_BAD_PAGE_TOKENS,
    MTP_V4_144_TAIL_PAGE_BAD_D,
    MTP_V4_144_TAIL_PAGE_MISSING_BLOCK_TABLE,
    MTP_V4_144_TAIL_PAGE_MISSING_K_BASE,
    MTP_V4_144_TAIL_PAGE_MISSING_V_BASE,
    MTP_V4_144_TAIL_PAGE_BAD_V4_STRIDE,
    MTP_V4_144_TAIL_PAGE_BAD_LOGICAL_TOKENS,
    MTP_V4_144_TAIL_PAGE_BAD_VALID_TOKENS,
    MTP_V4_144_TAIL_PAGE_BAD_BATCH,
    MTP_V4_144_TAIL_PAGE_BAD_BOUNDARY,
    MTP_V4_144_TAIL_PAGE_BAD_K_DESC,
    MTP_V4_144_TAIL_PAGE_BAD_K_DESC_AXES,
    MTP_V4_144_TAIL_PAGE_BAD_K_DESC_SHAPE,
    MTP_V4_144_TAIL_PAGE_BAD_PHYSICAL_PAGE,
    MTP_V4_144_TAIL_PAGE_TOKEN_NOT_VISIBLE,
};

struct mtp_v4_144_tail_page_desc_v1 {
    // ABI header
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;

    // Fixed geometry
    uint32_t page_tokens = 0;
    uint32_t d = 0;
    uint32_t logical_base_token = 0;
    uint32_t logical_tokens = 0;

    // Page table
    const int32_t * block_table = nullptr;
    uint32_t block_table_pages = 0;
    uint32_t physical_pages = 0;
    uint32_t valid_tail_tokens = 0;

    // Existing contiguous prefix boundary
    uint32_t prefix_tokens = 0;
    uint32_t boundary_slot = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;

    // Physical K page bank / sidecar bases
    const void * k_payload_base = nullptr;
    const void * k_scale_base = nullptr;
    uint64_t k_head_stride_bytes = 0;
    uint64_t k_page_stride_bytes = 0;

    // Physical V4_144 page bank
    const void * v4_base = nullptr;
    uint64_t v4_head_stride_bytes = 0;
    uint64_t v4_page_stride_bytes = 0;
    uint64_t v4_batch_stride_bytes = 0;

    // Shape
    uint32_t kv_heads = 0;
    uint32_t batch = 0;
    uint32_t gqa_ratio = 0;
    uint32_t reserved3 = 0;

    // Existing packed16 K layout contract, embedded by value.
    dp16_packed_i8_desc_v1 k_desc = {};
};

static constexpr uint32_t MTP_V4_144_TAIL_STAGE_ABI_VERSION = 1;

enum mtp_v4_144_tail_stage_kind : uint32_t {
    MTP_V4_144_TAIL_STAGE_KIND_NONE       = 0,
    MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K = 1,
    MTP_V4_144_TAIL_STAGE_KIND_V4_144     = 2,
};

enum mtp_v4_144_tail_stage_flags : uint32_t {
    MTP_V4_144_TAIL_STAGE_FLAG_READ_MERGE_WRITE = 1u << 0,
    MTP_V4_144_TAIL_STAGE_FLAG_BOUNDARY_COPY    = 1u << 1,
    MTP_V4_144_TAIL_STAGE_FLAG_FULL_PAGE_PAYLOAD = 1u << 2,
};

enum mtp_v4_144_tail_stage_status : uint32_t {
    MTP_V4_144_TAIL_STAGE_OK = 0,
    MTP_V4_144_TAIL_STAGE_BAD_VERSION,
    MTP_V4_144_TAIL_STAGE_BAD_ABI_BYTES,
    MTP_V4_144_TAIL_STAGE_BAD_KIND,
    MTP_V4_144_TAIL_STAGE_MISSING_READ_MERGE_WRITE,
    MTP_V4_144_TAIL_STAGE_BAD_PAGE_TOKENS,
    MTP_V4_144_TAIL_STAGE_BAD_D,
    MTP_V4_144_TAIL_STAGE_BAD_WRITE_TOKENS,
    MTP_V4_144_TAIL_STAGE_BAD_PAGE_BASE,
    MTP_V4_144_TAIL_STAGE_BAD_PAGE_END,
    MTP_V4_144_TAIL_STAGE_BAD_SLOT_RANGE,
    MTP_V4_144_TAIL_STAGE_SPANS_PAGE,
    MTP_V4_144_TAIL_STAGE_BAD_CACHE_TOKENS,
    MTP_V4_144_TAIL_STAGE_BAD_MERGE_FLAGS,
    MTP_V4_144_TAIL_STAGE_BAD_ROW_BYTES,
    MTP_V4_144_TAIL_STAGE_BAD_PAGE_BYTES,
    MTP_V4_144_TAIL_STAGE_BAD_PAYLOAD_BYTES,
    MTP_V4_144_TAIL_STAGE_BAD_MERGE_COPY_BYTES,
};

struct mtp_v4_144_tail_stage_desc_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t kind = 0;
    uint32_t flags = 0;

    uint32_t page_tokens = 0;
    uint32_t d = 0;
    uint32_t page_base_token = 0;
    uint32_t page_end_token = 0;

    uint32_t write_start_token = 0;
    uint32_t write_tokens = 0;
    uint32_t slot_begin = 0;
    uint32_t slot_end_excl = 0;

    uint32_t slots_before = 0;
    uint32_t slots_after = 0;
    uint32_t merge_copy_slots = 0;
    uint32_t cache_tokens = 0;

    uint32_t row_bytes = 0;
    uint32_t page_bytes = 0;
    uint32_t payload_bytes = 0;
    uint32_t merge_copy_bytes = 0;
};

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_div_round_up(
        const uint32_t n,
        const uint32_t d) {
    return (n + d - 1u) / d;
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_count_for_tokens(
        const uint32_t tokens) {
    return mtp_v4_144_tail_page_div_round_up(tokens, MTP_V4_144_PAGE_TOKENS);
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_stage_expected_row_bytes(
        const uint32_t kind) {
    if (kind == MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K) {
        return MTP_PACKED16_K_ROW_BYTES;
    }
    if (kind == MTP_V4_144_TAIL_STAGE_KIND_V4_144) {
        return MTP_V4_144_ROW_BYTES;
    }
    return 0;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_stage_desc_v1 mtp_v4_144_tail_stage_make(
        const uint32_t kind,
        const uint32_t cache_tokens,
        const uint32_t write_start_token,
        const uint32_t write_tokens,
        const uint32_t row_bytes) {
    mtp_v4_144_tail_stage_desc_v1 desc = {};
    desc.version = MTP_V4_144_TAIL_STAGE_ABI_VERSION;
    desc.abi_bytes = sizeof(mtp_v4_144_tail_stage_desc_v1);
    desc.kind = kind;
    desc.page_tokens = MTP_V4_144_PAGE_TOKENS;
    desc.d = MTP_V4_144_D;
    desc.page_base_token = write_start_token & ~(MTP_V4_144_PAGE_TOKENS - 1u);
    desc.page_end_token = desc.page_base_token + MTP_V4_144_PAGE_TOKENS;
    desc.write_start_token = write_start_token;
    desc.write_tokens = write_tokens;
    desc.slot_begin = write_start_token - desc.page_base_token;
    desc.slot_end_excl = desc.slot_begin + write_tokens;
    desc.slots_before = desc.slot_begin;
    desc.slots_after = desc.slot_end_excl > MTP_V4_144_PAGE_TOKENS ? 0u : MTP_V4_144_PAGE_TOKENS - desc.slot_end_excl;
    desc.merge_copy_slots = desc.slots_before + desc.slots_after;
    desc.cache_tokens = cache_tokens;
    desc.row_bytes = row_bytes;
    desc.page_bytes = row_bytes * MTP_V4_144_PAGE_TOKENS;
    desc.payload_bytes = row_bytes * write_tokens;
    desc.merge_copy_bytes = row_bytes * desc.merge_copy_slots;
    desc.flags = MTP_V4_144_TAIL_STAGE_FLAG_READ_MERGE_WRITE |
        (desc.merge_copy_slots != 0 ? MTP_V4_144_TAIL_STAGE_FLAG_BOUNDARY_COPY : 0u) |
        ((desc.slot_begin == 0 && write_tokens == MTP_V4_144_PAGE_TOKENS) ? MTP_V4_144_TAIL_STAGE_FLAG_FULL_PAGE_PAYLOAD : 0u);
    return desc;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_stage_status mtp_v4_144_tail_stage_validate_static(
        const mtp_v4_144_tail_stage_desc_v1 & desc) {
    if (desc.version != MTP_V4_144_TAIL_STAGE_ABI_VERSION) {
        return MTP_V4_144_TAIL_STAGE_BAD_VERSION;
    }
    if (desc.abi_bytes != sizeof(mtp_v4_144_tail_stage_desc_v1)) {
        return MTP_V4_144_TAIL_STAGE_BAD_ABI_BYTES;
    }
    if (desc.kind != MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K && desc.kind != MTP_V4_144_TAIL_STAGE_KIND_V4_144) {
        return MTP_V4_144_TAIL_STAGE_BAD_KIND;
    }
    if ((desc.flags & MTP_V4_144_TAIL_STAGE_FLAG_READ_MERGE_WRITE) == 0) {
        return MTP_V4_144_TAIL_STAGE_MISSING_READ_MERGE_WRITE;
    }
    if (desc.page_tokens != MTP_V4_144_PAGE_TOKENS) {
        return MTP_V4_144_TAIL_STAGE_BAD_PAGE_TOKENS;
    }
    if (desc.d != MTP_V4_144_D) {
        return MTP_V4_144_TAIL_STAGE_BAD_D;
    }
    if (desc.write_tokens == 0 || desc.write_tokens > desc.page_tokens) {
        return MTP_V4_144_TAIL_STAGE_BAD_WRITE_TOKENS;
    }
    if ((desc.page_base_token & (MTP_V4_144_PAGE_TOKENS - 1u)) != 0) {
        return MTP_V4_144_TAIL_STAGE_BAD_PAGE_BASE;
    }
    if (desc.page_end_token != desc.page_base_token + desc.page_tokens) {
        return MTP_V4_144_TAIL_STAGE_BAD_PAGE_END;
    }
    if (desc.write_start_token < desc.page_base_token || desc.slot_begin != desc.write_start_token - desc.page_base_token) {
        return MTP_V4_144_TAIL_STAGE_BAD_SLOT_RANGE;
    }
    if (desc.slot_end_excl != desc.slot_begin + desc.write_tokens || desc.slot_begin >= desc.page_tokens) {
        return MTP_V4_144_TAIL_STAGE_BAD_SLOT_RANGE;
    }
    if (desc.slot_end_excl > desc.page_tokens) {
        return MTP_V4_144_TAIL_STAGE_SPANS_PAGE;
    }
    if (desc.cache_tokens < desc.page_end_token) {
        return MTP_V4_144_TAIL_STAGE_BAD_CACHE_TOKENS;
    }
    const uint32_t slots_before = desc.slot_begin;
    const uint32_t slots_after = desc.page_tokens - desc.slot_end_excl;
    const uint32_t merge_copy_slots = slots_before + slots_after;
    const bool boundary_copy_required = merge_copy_slots != 0;
    const bool full_page_payload = desc.slot_begin == 0 && desc.write_tokens == desc.page_tokens;
    if (desc.slots_before != slots_before || desc.slots_after != slots_after || desc.merge_copy_slots != merge_copy_slots ||
            (((desc.flags & MTP_V4_144_TAIL_STAGE_FLAG_BOUNDARY_COPY) != 0) != boundary_copy_required) ||
            (((desc.flags & MTP_V4_144_TAIL_STAGE_FLAG_FULL_PAGE_PAYLOAD) != 0) != full_page_payload)) {
        return MTP_V4_144_TAIL_STAGE_BAD_MERGE_FLAGS;
    }
    const uint32_t expected_row_bytes = mtp_v4_144_tail_stage_expected_row_bytes(desc.kind);
    if (desc.row_bytes != expected_row_bytes) {
        return MTP_V4_144_TAIL_STAGE_BAD_ROW_BYTES;
    }
    if (desc.page_bytes != expected_row_bytes * desc.page_tokens) {
        return MTP_V4_144_TAIL_STAGE_BAD_PAGE_BYTES;
    }
    if (desc.payload_bytes != expected_row_bytes * desc.write_tokens) {
        return MTP_V4_144_TAIL_STAGE_BAD_PAYLOAD_BYTES;
    }
    if (desc.merge_copy_bytes != expected_row_bytes * desc.merge_copy_slots) {
        return MTP_V4_144_TAIL_STAGE_BAD_MERGE_COPY_BYTES;
    }
    return MTP_V4_144_TAIL_STAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_page_status mtp_v4_144_tail_page_validate_k_desc(
        const mtp_v4_144_tail_page_desc_v1 & desc) {
    if (!dp16_i8x16_desc_has_vector_abi(desc.k_desc)) {
        return MTP_V4_144_TAIL_PAGE_BAD_K_DESC;
    }
    if (desc.k_desc.axis_x != DP16_PACKED_I8_AXIS_D16 ||
            desc.k_desc.axis_y != DP16_PACKED_I8_AXIS_TOKEN ||
            desc.k_desc.axis_z != DP16_PACKED_I8_AXIS_HEAD ||
            desc.k_desc.scale_axis_x != DP16_PACKED_I8_AXIS_QBLOCK ||
            desc.k_desc.scale_axis_y != DP16_PACKED_I8_AXIS_TOKEN ||
            desc.k_desc.scale_axis_z != DP16_PACKED_I8_AXIS_HEAD) {
        return MTP_V4_144_TAIL_PAGE_BAD_K_DESC_AXES;
    }
    if (desc.k_desc.logical_x != MTP_PACKED16_K_D / DP16_PACKED_I8X16_LANES ||
            desc.k_desc.physical_x != desc.k_desc.logical_x ||
            desc.k_desc.logical_y == 0 ||
            desc.k_desc.physical_y != desc.k_desc.logical_y ||
            desc.k_desc.logical_z == 0 ||
            desc.k_desc.physical_z != desc.k_desc.logical_z ||
            desc.k_desc.halo_x_before != 0 || desc.k_desc.halo_y_before != 0 ||
            desc.k_desc.halo_x_after != 0 || desc.k_desc.halo_y_after != 0) {
        return MTP_V4_144_TAIL_PAGE_BAD_K_DESC_SHAPE;
    }
    return MTP_V4_144_TAIL_PAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_page_status mtp_v4_144_tail_page_validate_static(
        const mtp_v4_144_tail_page_desc_v1 & desc) {
    if (desc.version != MTP_V4_144_TAIL_PAGE_ABI_VERSION) {
        return MTP_V4_144_TAIL_PAGE_BAD_VERSION;
    }
    if (desc.abi_bytes != sizeof(mtp_v4_144_tail_page_desc_v1)) {
        return MTP_V4_144_TAIL_PAGE_BAD_ABI_BYTES;
    }
    if ((desc.flags & MTP_V4_144_TAIL_FLAG_TAIL_ONLY) == 0) {
        return MTP_V4_144_TAIL_PAGE_MISSING_TAIL_ONLY;
    }
    if (desc.page_tokens != MTP_V4_144_PAGE_TOKENS) {
        return MTP_V4_144_TAIL_PAGE_BAD_PAGE_TOKENS;
    }
    if (desc.d != MTP_V4_144_D) {
        return MTP_V4_144_TAIL_PAGE_BAD_D;
    }
    if (desc.block_table == nullptr || desc.block_table_pages == 0) {
        return MTP_V4_144_TAIL_PAGE_MISSING_BLOCK_TABLE;
    }
    if (desc.k_payload_base == nullptr || desc.k_scale_base == nullptr) {
        return desc.k_payload_base == nullptr ? MTP_V4_144_TAIL_PAGE_MISSING_K_BASE : MTP_V4_144_TAIL_PAGE_MISSING_K_BASE;
    }
    if (desc.v4_base == nullptr) {
        return MTP_V4_144_TAIL_PAGE_MISSING_V_BASE;
    }
    if (desc.v4_page_stride_bytes != MTP_V4_144_PAGE_BYTES) {
        return MTP_V4_144_TAIL_PAGE_BAD_V4_STRIDE;
    }
    if (desc.logical_tokens > desc.block_table_pages * MTP_V4_144_PAGE_TOKENS) {
        return MTP_V4_144_TAIL_PAGE_BAD_LOGICAL_TOKENS;
    }
    if (desc.valid_tail_tokens > desc.logical_tokens) {
        return MTP_V4_144_TAIL_PAGE_BAD_VALID_TOKENS;
    }
    if (desc.batch != 1) {
        return MTP_V4_144_TAIL_PAGE_BAD_BATCH;
    }
    if (desc.boundary_slot != (desc.logical_base_token & (MTP_V4_144_PAGE_TOKENS - 1u))) {
        return MTP_V4_144_TAIL_PAGE_BAD_BOUNDARY;
    }
    if ((desc.flags & MTP_V4_144_TAIL_FLAG_ALIGNED_ONLY) != 0 && desc.boundary_slot != 0) {
        return MTP_V4_144_TAIL_PAGE_BAD_BOUNDARY;
    }
    const mtp_v4_144_tail_page_status k_status = mtp_v4_144_tail_page_validate_k_desc(desc);
    if (k_status != MTP_V4_144_TAIL_PAGE_OK) {
        return k_status;
    }
    return MTP_V4_144_TAIL_PAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_page_status mtp_v4_144_tail_page_validate_block_table_entry(
        const mtp_v4_144_tail_page_desc_v1 & desc,
        const uint32_t logical_page) {
    if (logical_page >= desc.block_table_pages) {
        return MTP_V4_144_TAIL_PAGE_BAD_LOGICAL_TOKENS;
    }
    const int32_t physical_page = desc.block_table[logical_page];
    if (physical_page < 0 || uint32_t(physical_page) >= desc.physical_pages) {
        return MTP_V4_144_TAIL_PAGE_BAD_PHYSICAL_PAGE;
    }
    return MTP_V4_144_TAIL_PAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_page_status mtp_v4_144_tail_page_logical_to_physical(
        const mtp_v4_144_tail_page_desc_v1 & desc,
        const uint32_t logical_token,
        uint32_t * physical_page,
        uint32_t * slot) {
    const mtp_v4_144_tail_page_status status = mtp_v4_144_tail_page_validate_static(desc);
    if (status != MTP_V4_144_TAIL_PAGE_OK) {
        return status;
    }
    if (logical_token < desc.logical_base_token) {
        return MTP_V4_144_TAIL_PAGE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t relative = logical_token - desc.logical_base_token;
    if (relative >= desc.valid_tail_tokens) {
        return MTP_V4_144_TAIL_PAGE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t logical_page = relative / MTP_V4_144_PAGE_TOKENS;
    const mtp_v4_144_tail_page_status table_status = mtp_v4_144_tail_page_validate_block_table_entry(desc, logical_page);
    if (table_status != MTP_V4_144_TAIL_PAGE_OK) {
        return table_status;
    }
    if (physical_page != nullptr) {
        *physical_page = uint32_t(desc.block_table[logical_page]);
    }
    if (slot != nullptr) {
        *slot = relative & (MTP_V4_144_PAGE_TOKENS - 1u);
    }
    return MTP_V4_144_TAIL_PAGE_OK;
}

static __host__ __device__ __forceinline__ uint64_t mtp_v4_144_tail_page_v4_page_base_byte_offset(
        const mtp_v4_144_tail_page_desc_v1 & desc,
        const uint32_t physical_page,
        const uint32_t kv_head,
        const uint32_t batch_index) {
    return uint64_t(batch_index) * desc.v4_batch_stride_bytes +
        uint64_t(kv_head) * desc.v4_head_stride_bytes +
        uint64_t(physical_page) * desc.v4_page_stride_bytes;
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_v4_payload_word_index(
        const uint32_t slot,
        const uint32_t d) {
    return d * MTP_V4_144_WORDS_PER_D + (slot >> 3);
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_v4_scale_index(
        const uint32_t slot,
        const uint32_t d) {
    return (d / MTP_V4_144_D32) * MTP_V4_144_PAGE_TOKENS + slot;
}

static __host__ __device__ __forceinline__ uint64_t mtp_v4_144_tail_page_v4_payload_byte_offset(
        const mtp_v4_144_tail_page_desc_v1 & desc,
        const uint32_t physical_page,
        const uint32_t slot,
        const uint32_t d,
        const uint32_t kv_head,
        const uint32_t batch_index) {
    return mtp_v4_144_tail_page_v4_page_base_byte_offset(desc, physical_page, kv_head, batch_index) +
        uint64_t(mtp_v4_144_tail_page_v4_payload_word_index(slot, d)) * sizeof(uint32_t);
}

static __host__ __device__ __forceinline__ uint64_t mtp_v4_144_tail_page_v4_scale_byte_offset(
        const mtp_v4_144_tail_page_desc_v1 & desc,
        const uint32_t physical_page,
        const uint32_t slot,
        const uint32_t d,
        const uint32_t kv_head,
        const uint32_t batch_index) {
    return mtp_v4_144_tail_page_v4_page_base_byte_offset(desc, physical_page, kv_head, batch_index) +
        MTP_V4_144_PAYLOAD_BYTES_PER_PAGE +
        uint64_t(mtp_v4_144_tail_page_v4_scale_index(slot, d)) * sizeof(uint16_t);
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_k_physical_token(
        const uint32_t physical_page,
        const uint32_t slot) {
    return physical_page * MTP_V4_144_PAGE_TOKENS + slot;
}
