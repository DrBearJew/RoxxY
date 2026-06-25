// mtp-v4-144-tail-page-state.cuh — tiny tail-page commit/free-list model
#pragma once

#include "mtp-qblock-txn-lineage.cuh"

#include <cstddef>
#include <cstdint>

// Header-only, dispatch-inert state model for the narrow QBlock/MTP
// packed16-K + V4_144 transactional tail pager.  This is intentionally not a
// generic PagedAttention cache.  It models only:
//   - a canonical tail block table;
//   - single-owner physical pages;
//   - transaction-owned pages that commit by metadata promotion;
//   - rollback/free-list behavior for rejected/sibling pages;
//   - the accepted final recurrent state slot chosen by lineage.
static constexpr uint32_t MTP_V4_144_TAIL_STATE_ABI_VERSION = 1;
static constexpr uint32_t MTP_V4_144_TAIL_STATE_MAX_PAGES = 32;
static constexpr int32_t  MTP_V4_144_TAIL_PAGE_INVALID = -1;

static_assert(MTP_V4_144_TAIL_STATE_MAX_PAGES <= 32, "state bit masks are uint32_t");

enum mtp_v4_144_tail_state_flags : uint32_t {
    MTP_V4_144_TAIL_STATE_FLAG_SINGLE_OWNER            = 1u << 0,
    MTP_V4_144_TAIL_STATE_FLAG_FREE_REJECTED_ON_COMMIT = 1u << 1,
    MTP_V4_144_TAIL_STATE_FLAG_DEBUG_ABORTS            = 1u << 2,
};

enum mtp_v4_144_tail_page_owner : uint8_t {
    MTP_V4_144_TAIL_PAGE_OWNER_FREE      = 0,
    MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL = 1,
    MTP_V4_144_TAIL_PAGE_OWNER_TXN       = 2,
    MTP_V4_144_TAIL_PAGE_OWNER_RESERVED  = 3,
};

enum mtp_v4_144_tail_state_status : uint32_t {
    MTP_V4_144_TAIL_STATE_OK = 0,
    MTP_V4_144_TAIL_STATE_BAD_VERSION,
    MTP_V4_144_TAIL_STATE_BAD_ABI_BYTES,
    MTP_V4_144_TAIL_STATE_BAD_PAGE_TOKENS,
    MTP_V4_144_TAIL_STATE_BAD_PHYSICAL_PAGES,
    MTP_V4_144_TAIL_STATE_BAD_LOGICAL_BASE,
    MTP_V4_144_TAIL_STATE_BAD_LOGICAL_PAGES,
    MTP_V4_144_TAIL_STATE_BAD_VALID_TOKENS,
    MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE,
    MTP_V4_144_TAIL_STATE_BAD_OWNER,
    MTP_V4_144_TAIL_STATE_BAD_REFCOUNT,
    MTP_V4_144_TAIL_STATE_BAD_PAGE_VALID_TOKENS,
    MTP_V4_144_TAIL_STATE_BAD_FREE_COUNT,
    MTP_V4_144_TAIL_STATE_BAD_FREE_PAGE,
    MTP_V4_144_TAIL_STATE_DUP_FREE_PAGE,
    MTP_V4_144_TAIL_STATE_NO_FREE_PAGE,
    MTP_V4_144_TAIL_STATE_BAD_PAGE,
    MTP_V4_144_TAIL_STATE_PAGE_NOT_FREE,
    MTP_V4_144_TAIL_STATE_PAGE_NOT_TXN,
    MTP_V4_144_TAIL_STATE_TAIL_DESC_REJECTED,
    MTP_V4_144_TAIL_STATE_LINEAGE_REJECTED,
    MTP_V4_144_TAIL_STATE_ACCEPTED_OVERFLOW,
    MTP_V4_144_TAIL_STATE_BAD_ACCEPTED_PAGE,
    MTP_V4_144_TAIL_STATE_BAD_STATE_SLOT,
    MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE,
};

struct mtp_v4_144_tail_page_state_v1 {
    uint32_t version = 0;
    uint32_t abi_bytes = 0;
    uint32_t flags = 0;
    uint32_t page_tokens = 0;

    uint32_t logical_base_token = 0;
    uint32_t valid_tail_tokens = 0;
    uint32_t logical_pages = 0;
    uint32_t physical_pages = 0;

    uint32_t free_count = 0;
    uint32_t final_state_slot = MTP_QBLOCK_TXN_INVALID_U8;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;

    // Canonical logical tail page -> physical page. Only entries below
    // logical_pages are visible; stale slots inside the final page are masked by
    // valid_tail_tokens and page_valid_tokens.
    int32_t block_table[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};

    // Physical-page metadata. v1 is single-owner and refcount is only 0/1.
    uint8_t owner[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
    uint8_t refcount[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
    uint8_t page_valid_tokens[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
    uint8_t free_stack[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
};

static_assert(sizeof(mtp_v4_144_tail_page_state_v1) <= 320, "tail-page state sidecar should stay tiny");

static __host__ __device__ __forceinline__ uint32_t mtp_v4_144_tail_state_page_count_for_valid_tokens(
        const uint32_t valid_tail_tokens) {
    return mtp_v4_144_tail_page_count_for_tokens(valid_tail_tokens);
}

static __host__ __device__ __forceinline__ uint8_t mtp_v4_144_tail_state_expected_page_valid_tokens(
        const uint32_t valid_tail_tokens,
        const uint32_t logical_page) {
    const uint32_t page_start = logical_page * MTP_V4_144_PAGE_TOKENS;
    if (valid_tail_tokens >= page_start + MTP_V4_144_PAGE_TOKENS) {
        return uint8_t(MTP_V4_144_PAGE_TOKENS);
    }
    if (valid_tail_tokens <= page_start) {
        return 0;
    }
    return uint8_t(valid_tail_tokens - page_start);
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_init(
        mtp_v4_144_tail_page_state_v1 & state,
        const uint32_t logical_base_token,
        const uint32_t physical_pages) {
    if (physical_pages == 0 || physical_pages > MTP_V4_144_TAIL_STATE_MAX_PAGES) {
        return MTP_V4_144_TAIL_STATE_BAD_PHYSICAL_PAGES;
    }

    state = mtp_v4_144_tail_page_state_v1{};
    state.version = MTP_V4_144_TAIL_STATE_ABI_VERSION;
    state.abi_bytes = sizeof(mtp_v4_144_tail_page_state_v1);
    state.flags = MTP_V4_144_TAIL_STATE_FLAG_SINGLE_OWNER |
        MTP_V4_144_TAIL_STATE_FLAG_FREE_REJECTED_ON_COMMIT;
    state.page_tokens = MTP_V4_144_PAGE_TOKENS;
    state.logical_base_token = logical_base_token;
    state.physical_pages = physical_pages;
    state.free_count = physical_pages;
    state.final_state_slot = MTP_QBLOCK_TXN_INVALID_U8;

    for (uint32_t i = 0; i < MTP_V4_144_TAIL_STATE_MAX_PAGES; ++i) {
        state.block_table[i] = MTP_V4_144_TAIL_PAGE_INVALID;
        state.owner[i] = i < physical_pages ? uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE) : uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_RESERVED);
        state.refcount[i] = 0;
        state.page_valid_tokens[i] = 0;
        state.free_stack[i] = i < physical_pages ? uint8_t(physical_pages - 1u - i) : MTP_QBLOCK_TXN_INVALID_U8;
    }
    return MTP_V4_144_TAIL_STATE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_validate_static(
        const mtp_v4_144_tail_page_state_v1 & state) {
    if (state.version != MTP_V4_144_TAIL_STATE_ABI_VERSION) {
        return MTP_V4_144_TAIL_STATE_BAD_VERSION;
    }
    if (state.abi_bytes != sizeof(mtp_v4_144_tail_page_state_v1)) {
        return MTP_V4_144_TAIL_STATE_BAD_ABI_BYTES;
    }
    if (state.page_tokens != MTP_V4_144_PAGE_TOKENS) {
        return MTP_V4_144_TAIL_STATE_BAD_PAGE_TOKENS;
    }
    if (state.physical_pages == 0 || state.physical_pages > MTP_V4_144_TAIL_STATE_MAX_PAGES) {
        return MTP_V4_144_TAIL_STATE_BAD_PHYSICAL_PAGES;
    }
    const uint32_t expected_logical_pages = mtp_v4_144_tail_state_page_count_for_valid_tokens(state.valid_tail_tokens);
    if (state.logical_pages != expected_logical_pages || state.logical_pages > MTP_V4_144_TAIL_STATE_MAX_PAGES) {
        return MTP_V4_144_TAIL_STATE_BAD_LOGICAL_PAGES;
    }
    if (state.valid_tail_tokens > state.logical_pages * MTP_V4_144_PAGE_TOKENS) {
        return MTP_V4_144_TAIL_STATE_BAD_VALID_TOKENS;
    }

    for (uint32_t lp = 0; lp < MTP_V4_144_TAIL_STATE_MAX_PAGES; ++lp) {
        const int32_t pp = state.block_table[lp];
        if (lp < state.logical_pages) {
            if (pp < 0 || uint32_t(pp) >= state.physical_pages) {
                return MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE;
            }
        } else if (pp != MTP_V4_144_TAIL_PAGE_INVALID) {
            return MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE;
        }
    }

    for (uint32_t pp = 0; pp < MTP_V4_144_TAIL_STATE_MAX_PAGES; ++pp) {
        const uint8_t owner = state.owner[pp];
        if (pp < state.physical_pages) {
            if (owner > uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_TXN)) {
                return MTP_V4_144_TAIL_STATE_BAD_OWNER;
            }
            if (state.page_valid_tokens[pp] > MTP_V4_144_PAGE_TOKENS) {
                return MTP_V4_144_TAIL_STATE_BAD_PAGE_VALID_TOKENS;
            }
            if (owner == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE)) {
                if (state.refcount[pp] != 0 || state.page_valid_tokens[pp] != 0) {
                    return MTP_V4_144_TAIL_STATE_BAD_REFCOUNT;
                }
            } else {
                if ((state.flags & MTP_V4_144_TAIL_STATE_FLAG_SINGLE_OWNER) != 0 && state.refcount[pp] != 1) {
                    return MTP_V4_144_TAIL_STATE_BAD_REFCOUNT;
                }
            }
        } else if (owner != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_RESERVED) || state.refcount[pp] != 0 || state.page_valid_tokens[pp] != 0) {
            return MTP_V4_144_TAIL_STATE_BAD_OWNER;
        }
    }

    for (uint32_t lp = 0; lp < state.logical_pages; ++lp) {
        const uint32_t pp = uint32_t(state.block_table[lp]);
        if (state.owner[pp] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL)) {
            return MTP_V4_144_TAIL_STATE_BAD_OWNER;
        }
        const uint8_t expected = mtp_v4_144_tail_state_expected_page_valid_tokens(state.valid_tail_tokens, lp);
        if (state.page_valid_tokens[pp] != expected) {
            return MTP_V4_144_TAIL_STATE_BAD_PAGE_VALID_TOKENS;
        }
    }

    if (state.free_count > state.physical_pages) {
        return MTP_V4_144_TAIL_STATE_BAD_FREE_COUNT;
    }
    uint32_t free_mask = 0;
    for (uint32_t i = 0; i < state.free_count; ++i) {
        const uint32_t pp = uint32_t(state.free_stack[i]);
        if (pp >= state.physical_pages) {
            return MTP_V4_144_TAIL_STATE_BAD_FREE_PAGE;
        }
        const uint32_t bit = 1u << pp;
        if ((free_mask & bit) != 0) {
            return MTP_V4_144_TAIL_STATE_DUP_FREE_PAGE;
        }
        free_mask |= bit;
        if (state.owner[pp] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE)) {
            return MTP_V4_144_TAIL_STATE_BAD_FREE_PAGE;
        }
    }
    for (uint32_t pp = 0; pp < state.physical_pages; ++pp) {
        const bool listed_free = (free_mask & (1u << pp)) != 0;
        const bool owner_free = state.owner[pp] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE);
        if (listed_free != owner_free) {
            return MTP_V4_144_TAIL_STATE_BAD_FREE_PAGE;
        }
    }
    return MTP_V4_144_TAIL_STATE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_alloc_page(
        mtp_v4_144_tail_page_state_v1 & state,
        const mtp_v4_144_tail_page_owner owner,
        uint32_t * out_page) {
    if (owner != MTP_V4_144_TAIL_PAGE_OWNER_TXN && owner != MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL) {
        return MTP_V4_144_TAIL_STATE_BAD_OWNER;
    }
    const mtp_v4_144_tail_state_status status = mtp_v4_144_tail_state_validate_static(state);
    if (status != MTP_V4_144_TAIL_STATE_OK) {
        return status;
    }
    if (state.free_count == 0) {
        return MTP_V4_144_TAIL_STATE_NO_FREE_PAGE;
    }
    const uint32_t pp = uint32_t(state.free_stack[--state.free_count]);
    if (pp >= state.physical_pages || state.owner[pp] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE)) {
        return MTP_V4_144_TAIL_STATE_PAGE_NOT_FREE;
    }
    state.owner[pp] = uint8_t(owner);
    state.refcount[pp] = 1;
    state.page_valid_tokens[pp] = 0;
    if (out_page != nullptr) {
        *out_page = pp;
    }
    return MTP_V4_144_TAIL_STATE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_release_txn_page(
        mtp_v4_144_tail_page_state_v1 & state,
        const uint32_t physical_page) {
    if (physical_page >= state.physical_pages) {
        return MTP_V4_144_TAIL_STATE_BAD_PAGE;
    }
    if (state.owner[physical_page] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_TXN)) {
        return MTP_V4_144_TAIL_STATE_PAGE_NOT_TXN;
    }
    if (state.free_count >= state.physical_pages) {
        return MTP_V4_144_TAIL_STATE_BAD_FREE_COUNT;
    }
    state.owner[physical_page] = uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE);
    state.refcount[physical_page] = 0;
    state.page_valid_tokens[physical_page] = 0;
    state.free_stack[state.free_count++] = uint8_t(physical_page);
    return MTP_V4_144_TAIL_STATE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_rollback_txn_pages(
        mtp_v4_144_tail_page_state_v1 & state) {
    for (uint32_t pp = 0; pp < state.physical_pages; ++pp) {
        if (state.owner[pp] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_TXN)) {
            const mtp_v4_144_tail_state_status status = mtp_v4_144_tail_state_release_txn_page(state, pp);
            if (status != MTP_V4_144_TAIL_STATE_OK) {
                return status;
            }
        }
    }
    return mtp_v4_144_tail_state_validate_static(state);
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_logical_to_physical(
        const mtp_v4_144_tail_page_state_v1 & state,
        const uint32_t logical_token,
        uint32_t * physical_page,
        uint32_t * slot) {
    const mtp_v4_144_tail_state_status status = mtp_v4_144_tail_state_validate_static(state);
    if (status != MTP_V4_144_TAIL_STATE_OK) {
        return status;
    }
    if (logical_token < state.logical_base_token) {
        return MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t relative = logical_token - state.logical_base_token;
    if (relative >= state.valid_tail_tokens) {
        return MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t lp = relative / MTP_V4_144_PAGE_TOKENS;
    if (lp >= state.logical_pages || state.block_table[lp] < 0) {
        return MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE;
    }
    if (physical_page != nullptr) {
        *physical_page = uint32_t(state.block_table[lp]);
    }
    if (slot != nullptr) {
        *slot = relative & (MTP_V4_144_PAGE_TOKENS - 1u);
    }
    return MTP_V4_144_TAIL_STATE_OK;
}

static __host__ __device__ __forceinline__ mtp_v4_144_tail_state_status mtp_v4_144_tail_state_commit_lineage(
        mtp_v4_144_tail_page_state_v1 & state,
        const mtp_v4_144_tail_page_desc_v1 & tail,
        const mtp_qblock_txn_lineage_v1 & lin) {
    mtp_v4_144_tail_state_status state_status = mtp_v4_144_tail_state_validate_static(state);
    if (state_status != MTP_V4_144_TAIL_STATE_OK) {
        return state_status;
    }
    if (tail.logical_base_token != state.logical_base_token) {
        return MTP_V4_144_TAIL_STATE_BAD_LOGICAL_BASE;
    }
    const mtp_v4_144_tail_page_status tail_status = mtp_v4_144_tail_page_validate_static(tail);
    if (tail_status != MTP_V4_144_TAIL_PAGE_OK) {
        return MTP_V4_144_TAIL_STATE_TAIL_DESC_REJECTED;
    }
    const mtp_qblock_txn_lineage_status lineage_status = mtp_qblock_txn_lineage_validate_commit(lin, tail);
    if (lineage_status != MTP_QBLOCK_TXN_LINEAGE_OK) {
        return MTP_V4_144_TAIL_STATE_LINEAGE_REJECTED;
    }

    const uint32_t new_valid_tail_tokens = mtp_qblock_txn_lineage_new_valid_tail_tokens(tail, lin);
    if (new_valid_tail_tokens > tail.logical_tokens) {
        return MTP_V4_144_TAIL_STATE_ACCEPTED_OVERFLOW;
    }
    const uint32_t new_logical_pages = mtp_v4_144_tail_state_page_count_for_valid_tokens(new_valid_tail_tokens);
    if (new_logical_pages > MTP_V4_144_TAIL_STATE_MAX_PAGES || new_logical_pages > tail.block_table_pages) {
        return MTP_V4_144_TAIL_STATE_BAD_LOGICAL_PAGES;
    }

    uint32_t accepted_page_mask = 0;
    int32_t new_block_table[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
    uint8_t new_page_valid[MTP_V4_144_TAIL_STATE_MAX_PAGES] = {};
    for (uint32_t lp = 0; lp < MTP_V4_144_TAIL_STATE_MAX_PAGES; ++lp) {
        new_block_table[lp] = MTP_V4_144_TAIL_PAGE_INVALID;
    }

    for (uint32_t lp = 0; lp < new_logical_pages; ++lp) {
        const mtp_v4_144_tail_page_status table_status = mtp_v4_144_tail_page_validate_block_table_entry(tail, lp);
        if (table_status != MTP_V4_144_TAIL_PAGE_OK) {
            return MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE;
        }
        const uint32_t pp = uint32_t(tail.block_table[lp]);
        if (pp >= state.physical_pages ||
                (state.owner[pp] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_TXN) &&
                 state.owner[pp] != uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL))) {
            return MTP_V4_144_TAIL_STATE_BAD_ACCEPTED_PAGE;
        }
        accepted_page_mask |= 1u << pp;
        new_block_table[lp] = int32_t(pp);
        new_page_valid[pp] = mtp_v4_144_tail_state_expected_page_valid_tokens(new_valid_tail_tokens, lp);
    }

    const uint8_t final_state_slot = mtp_qblock_txn_lineage_final_state_slot(lin);
    if (final_state_slot == MTP_QBLOCK_TXN_INVALID_U8) {
        return MTP_V4_144_TAIL_STATE_BAD_STATE_SLOT;
    }

    for (uint32_t lp = 0; lp < MTP_V4_144_TAIL_STATE_MAX_PAGES; ++lp) {
        state.block_table[lp] = new_block_table[lp];
    }
    for (uint32_t pp = 0; pp < state.physical_pages; ++pp) {
        if ((accepted_page_mask & (1u << pp)) != 0) {
            state.owner[pp] = uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL);
            state.refcount[pp] = 1;
            state.page_valid_tokens[pp] = new_page_valid[pp];
        }
    }
    state.valid_tail_tokens = new_valid_tail_tokens;
    state.logical_pages = new_logical_pages;
    state.final_state_slot = final_state_slot;

    if ((state.flags & MTP_V4_144_TAIL_STATE_FLAG_FREE_REJECTED_ON_COMMIT) != 0) {
        for (uint32_t pp = 0; pp < state.physical_pages; ++pp) {
            if (state.owner[pp] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_TXN)) {
                state_status = mtp_v4_144_tail_state_release_txn_page(state, pp);
                if (state_status != MTP_V4_144_TAIL_STATE_OK) {
                    return state_status;
                }
            }
        }
    }

    return mtp_v4_144_tail_state_validate_static(state);
}
