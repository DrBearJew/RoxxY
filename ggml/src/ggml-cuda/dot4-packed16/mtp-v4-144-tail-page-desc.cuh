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
// It is an unused descriptor in this patch. Runtime dispatch must still perform
// tensor type checks before constructing it.
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

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_div_round_up(
        const uint32_t n,
        const uint32_t d) {
    return (n + d - 1u) / d;
}

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_page_count_for_tokens(
        const uint32_t tokens) {
    return mtp_v4_144_tail_page_div_round_up(tokens, MTP_V4_144_PAGE_TOKENS);
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
