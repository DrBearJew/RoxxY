// mtp-qblock-txn-lineage.cuh — unused QBlock transaction lineage ABI
#pragma once

#include "mtp-v4-144-tail-page-desc.cuh"

#include <cstddef>
#include <cstdint>

// This ABI intentionally describes only the small QBlock verifier transaction
// surface needed by the packed16-K + V4_144 tail pager. It is header-only and
// unused by dispatch in this patch.
//
// v1 policy:
//   - max rows is tiny and fixed: root + up to 7 verifier/draft rows;
//   - implicit chain fast path needs no parent upload;
//   - non-chain branches use bounded uint8_t sidecar arrays;
//   - commit is valid only when accepted rows are chain-contiguous and slots are
//     canonical order inside the current K16 page window;
//   - recurrent state is represented by slot ids, not tensor ownership.
static constexpr uint32_t MTP_QBLOCK_TXN_LINEAGE_ABI_VERSION = 1;
static constexpr uint32_t MTP_QBLOCK_TXN_MAX_ROWS = 8;
static constexpr uint32_t MTP_QBLOCK_TXN_ROOT_ROW = 0;
static constexpr uint8_t  MTP_QBLOCK_TXN_INVALID_U8 = 0xffu;

static_assert(MTP_QBLOCK_TXN_MAX_ROWS <= 16, "accepted_mask is uint16_t");
static_assert(MTP_QBLOCK_TXN_MAX_ROWS <= MTP_V4_144_PAGE_TOKENS, "v1 lineage fits one K16 page");

enum mtp_qblock_txn_lineage_flags : uint32_t {
    // parent(row i) is implicitly i-1, root points to itself. Parent/depth
    // arrays may be left at their zero defaults when this flag is set.
    MTP_QBLOCK_TXN_LINEAGE_FLAG_IMPLICIT_CHAIN = 1u << 0,

    // accepted rows must be root + first accepted_len chain rows. This is the
    // only promotable v1 fast path.
    MTP_QBLOCK_TXN_LINEAGE_FLAG_CONTIGUOUS_COMMIT = 1u << 1,

    // kv_slot(row) must equal boundary_slot + row_index - 1 for accepted rows.
    MTP_QBLOCK_TXN_LINEAGE_FLAG_CANONICAL_SLOT_ORDER = 1u << 2,

    // Abort/fail closed on unsupported lineage instead of silently accepting.
    MTP_QBLOCK_TXN_LINEAGE_FLAG_DEBUG_ABORTS = 1u << 3,
};

enum mtp_qblock_txn_lineage_status : uint32_t {
    MTP_QBLOCK_TXN_LINEAGE_OK = 0,
    MTP_QBLOCK_TXN_LINEAGE_BAD_VERSION,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ABI_BYTES,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ROW_COUNT,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ROOT,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEN,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEAF,
    MTP_QBLOCK_TXN_LINEAGE_BAD_PARENT,
    MTP_QBLOCK_TXN_LINEAGE_CYCLE,
    MTP_QBLOCK_TXN_LINEAGE_DEPTH_OVERFLOW,
    MTP_QBLOCK_TXN_LINEAGE_BAD_KV_SLOT,
    MTP_QBLOCK_TXN_LINEAGE_BAD_STATE_SLOT,
    MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_MASK,
    MTP_QBLOCK_TXN_LINEAGE_NOT_CONTIGUOUS,
    MTP_QBLOCK_TXN_LINEAGE_NOT_CANONICAL_SLOT_ORDER,
    MTP_QBLOCK_TXN_LINEAGE_TAIL_DESC_REJECTED,
};

struct mtp_qblock_txn_lineage_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t n_rows = 0;
    uint32_t root_row = 0;

    uint32_t flags = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;

    uint8_t parent[MTP_QBLOCK_TXN_MAX_ROWS] = {};
    uint8_t depth[MTP_QBLOCK_TXN_MAX_ROWS] = {};
    uint8_t kv_slot[MTP_QBLOCK_TXN_MAX_ROWS] = {};
    uint8_t state_slot[MTP_QBLOCK_TXN_MAX_ROWS] = {};

    uint8_t accepted_leaf = 0;
    uint8_t accepted_len = 0;
    uint16_t accepted_mask = 0;
    uint32_t reserved3 = 0;
};

static_assert(sizeof(mtp_qblock_txn_lineage_v1) <= 96, "lineage sidecar must stay tiny");

static __host__ __device__ __forceinline__ bool mtp_qblock_txn_lineage_flag(
        const mtp_qblock_txn_lineage_v1 & lin,
        const mtp_qblock_txn_lineage_flags flag) {
    return (lin.flags & uint32_t(flag)) != 0;
}

static __host__ __device__ __forceinline__ uint16_t mtp_qblock_txn_lineage_chain_mask(
        const uint32_t accepted_len) {
    // accepted_len counts committed non-root rows. The accepted mask includes
    // root plus accepted_len rows: len 0 -> bit0 only, len 1 -> bits0..1.
    if (accepted_len + 1u >= 16u) {
        return 0xffffu;
    }
    return uint16_t((1u << (accepted_len + 1u)) - 1u);
}

static __host__ __device__ __forceinline__ uint8_t mtp_qblock_txn_lineage_parent_of(
        const mtp_qblock_txn_lineage_v1 & lin,
        const uint32_t row) {
    if (mtp_qblock_txn_lineage_flag(lin, MTP_QBLOCK_TXN_LINEAGE_FLAG_IMPLICIT_CHAIN)) {
        return row == lin.root_row ? uint8_t(lin.root_row) : uint8_t(row - 1u);
    }
    return lin.parent[row];
}

static __host__ __device__ __forceinline__ mtp_qblock_txn_lineage_status mtp_qblock_txn_lineage_validate_header(
        const mtp_qblock_txn_lineage_v1 & lin) {
    if (lin.version != MTP_QBLOCK_TXN_LINEAGE_ABI_VERSION) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_VERSION;
    }
    if (lin.abi_bytes != sizeof(mtp_qblock_txn_lineage_v1)) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ABI_BYTES;
    }
    if (lin.n_rows == 0 || lin.n_rows > MTP_QBLOCK_TXN_MAX_ROWS) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ROW_COUNT;
    }
    if (lin.root_row != MTP_QBLOCK_TXN_ROOT_ROW || lin.root_row >= lin.n_rows) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ROOT;
    }
    if (uint32_t(lin.accepted_len) + 1u > lin.n_rows) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEN;
    }
    if (lin.accepted_leaf >= lin.n_rows) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEAF;
    }
    return MTP_QBLOCK_TXN_LINEAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_qblock_txn_lineage_status mtp_qblock_txn_lineage_build_accepted_mask(
        const mtp_qblock_txn_lineage_v1 & lin,
        uint16_t * out_mask,
        uint8_t * out_depth) {
    mtp_qblock_txn_lineage_status status = mtp_qblock_txn_lineage_validate_header(lin);
    if (status != MTP_QBLOCK_TXN_LINEAGE_OK) {
        return status;
    }

    uint16_t mask = 0;
    uint32_t row = lin.accepted_leaf;
    uint32_t depth = 0;
    for (;;) {
        if (row >= lin.n_rows) {
            return MTP_QBLOCK_TXN_LINEAGE_BAD_PARENT;
        }
        if ((mask & uint16_t(1u << row)) != 0) {
            return MTP_QBLOCK_TXN_LINEAGE_CYCLE;
        }
        mask = uint16_t(mask | uint16_t(1u << row));
        if (row == lin.root_row) {
            break;
        }
        if (++depth >= MTP_QBLOCK_TXN_MAX_ROWS) {
            return MTP_QBLOCK_TXN_LINEAGE_DEPTH_OVERFLOW;
        }
        row = uint32_t(mtp_qblock_txn_lineage_parent_of(lin, row));
    }

    if (depth != uint32_t(lin.accepted_len)) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEN;
    }
    if (out_mask != nullptr) {
        *out_mask = mask;
    }
    if (out_depth != nullptr) {
        *out_depth = uint8_t(depth);
    }
    return MTP_QBLOCK_TXN_LINEAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_qblock_txn_lineage_status mtp_qblock_txn_lineage_validate_slots(
        const mtp_qblock_txn_lineage_v1 & lin,
        const mtp_v4_144_tail_page_desc_v1 & tail) {
    const mtp_v4_144_tail_page_status tail_status = mtp_v4_144_tail_page_validate_static(tail);
    if (tail_status != MTP_V4_144_TAIL_PAGE_OK) {
        return MTP_QBLOCK_TXN_LINEAGE_TAIL_DESC_REJECTED;
    }

    for (uint32_t row = 0; row < lin.n_rows; ++row) {
        if (lin.kv_slot[row] >= MTP_V4_144_PAGE_TOKENS) {
            return MTP_QBLOCK_TXN_LINEAGE_BAD_KV_SLOT;
        }
        if (lin.state_slot[row] == MTP_QBLOCK_TXN_INVALID_U8) {
            return MTP_QBLOCK_TXN_LINEAGE_BAD_STATE_SLOT;
        }
    }

    if (mtp_qblock_txn_lineage_flag(lin, MTP_QBLOCK_TXN_LINEAGE_FLAG_CANONICAL_SLOT_ORDER)) {
        const uint32_t boundary = tail.boundary_slot;
        for (uint32_t row = 1; row <= uint32_t(lin.accepted_len); ++row) {
            const uint32_t expected = boundary + row - 1u;
            if (expected >= MTP_V4_144_PAGE_TOKENS || uint32_t(lin.kv_slot[row]) != expected) {
                return MTP_QBLOCK_TXN_LINEAGE_NOT_CANONICAL_SLOT_ORDER;
            }
        }
    }
    return MTP_QBLOCK_TXN_LINEAGE_OK;
}

static __host__ __device__ __forceinline__ mtp_qblock_txn_lineage_status mtp_qblock_txn_lineage_validate_commit(
        const mtp_qblock_txn_lineage_v1 & lin,
        const mtp_v4_144_tail_page_desc_v1 & tail) {
    uint16_t mask = 0;
    uint8_t depth = 0;
    mtp_qblock_txn_lineage_status status = mtp_qblock_txn_lineage_build_accepted_mask(lin, &mask, &depth);
    if (status != MTP_QBLOCK_TXN_LINEAGE_OK) {
        return status;
    }
    if (lin.accepted_mask != 0 && lin.accepted_mask != mask) {
        return MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_MASK;
    }

    if (mtp_qblock_txn_lineage_flag(lin, MTP_QBLOCK_TXN_LINEAGE_FLAG_CONTIGUOUS_COMMIT)) {
        const uint16_t chain_mask = mtp_qblock_txn_lineage_chain_mask(lin.accepted_len);
        if (mask != chain_mask || lin.accepted_leaf != lin.accepted_len) {
            return MTP_QBLOCK_TXN_LINEAGE_NOT_CONTIGUOUS;
        }
    }

    status = mtp_qblock_txn_lineage_validate_slots(lin, tail);
    if (status != MTP_QBLOCK_TXN_LINEAGE_OK) {
        return status;
    }
    return MTP_QBLOCK_TXN_LINEAGE_OK;
}

static __host__ __device__ __forceinline__ uint8_t mtp_qblock_txn_lineage_final_state_slot(
        const mtp_qblock_txn_lineage_v1 & lin) {
    return lin.accepted_leaf < lin.n_rows ? lin.state_slot[lin.accepted_leaf] : MTP_QBLOCK_TXN_INVALID_U8;
}

static __host__ __device__ __forceinline__ uint32_t mtp_qblock_txn_lineage_new_valid_tail_tokens(
        const mtp_v4_144_tail_page_desc_v1 & tail,
        const mtp_qblock_txn_lineage_v1 & lin) {
    return tail.valid_tail_tokens + uint32_t(lin.accepted_len);
}
