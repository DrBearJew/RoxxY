#pragma once

#include <cstdint>

static constexpr uint32_t LLAMA_MTP_QBLOCK_PAGED_STATE_VERSION = 1;
static constexpr uint32_t LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES = 32;
static constexpr uint32_t LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS = 16;
static constexpr uint32_t LLAMA_MTP_QBLOCK_PAGED_CONSUMER_MAP_MAX_PAGES = 4;
static constexpr uint32_t LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_VERSION = 1;
static constexpr int32_t  LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE = -1;
static constexpr uint32_t LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT = 0xffu;

enum llama_mtp_qblock_paged_page_owner : uint8_t {
    LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE      = 0,
    LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL = 1,
    LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN       = 2,
    LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED  = 3,
};

enum llama_mtp_qblock_paged_state_status : uint32_t {
    LLAMA_MTP_QBLOCK_PAGED_STATE_OK = 0,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_VERSION,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_ABI_BYTES,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_TOKENS,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_BASE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_PAGES,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_VALID_TOKENS,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_REFCOUNT,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_VALID_TOKENS,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_COUNT,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_FREE_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_MANAGED_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_NO_FREE_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_NOT_FREE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_NOT_TXN,
    LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES,
    LLAMA_MTP_QBLOCK_PAGED_STATE_MAP_TOO_LARGE,
    LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE,
};

struct llama_mtp_qblock_paged_state_v1 {
    uint32_t version = LLAMA_MTP_QBLOCK_PAGED_STATE_VERSION;
    uint32_t abi_bytes = sizeof(llama_mtp_qblock_paged_state_v1);
    uint32_t active = 0;
    uint32_t page_tokens = LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS;

    uint32_t logical_base_token = 0;
    uint32_t valid_tail_tokens = 0;
    uint32_t logical_pages = 0;
    uint32_t physical_pages = 0; // total KV-cache physical pages visible to consumers
    uint32_t managed_pages = 0;  // local owner/free metadata slots; <= MAX_PAGES

    uint32_t free_count = 0;
    uint32_t final_state_slot = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    uint64_t generation = 0;

    int32_t block_table[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    int32_t managed_physical_page[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    uint8_t owner[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    uint8_t refcount[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    uint8_t page_valid_tokens[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    uint8_t free_stack[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
};

struct llama_mtp_qblock_paged_consumer_map_v1 {
    uint32_t logical_base_token = 0;
    uint32_t valid_tail_tokens = 0;
    uint32_t page_tokens = 0;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    int32_t block_table[LLAMA_MTP_QBLOCK_PAGED_CONSUMER_MAP_MAX_PAGES] = {};
    uint64_t generation = 0;
};

enum llama_mtp_qblock_tail_txn_commit_status : uint32_t {
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK = 0,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_VERSION,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_ABI_BYTES,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PAGE_TOKENS,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_ACCEPTED_TOKENS,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PHYSICAL_PAGES,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_LOGICAL_PAGES,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_BLOCK_TABLE,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_FINAL_STATE_SLOT,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_SAMPLER_COMMIT,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_NO_VISIBLE_PAGES,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_MAP_TOO_LARGE,
    LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PAGE_STATE,
};

struct llama_mtp_qblock_tail_txn_commit_v1 {
    uint32_t version = LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_VERSION;
    uint32_t abi_bytes = sizeof(llama_mtp_qblock_tail_txn_commit_v1);
    uint32_t logical_base_token = 0;
    uint32_t accepted_tokens = 0;
    uint32_t page_tokens = LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS;
    uint32_t physical_pages = 0;
    uint32_t block_table_pages = 0;
    uint32_t final_state_slot = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    uint32_t recurrent_slot_count = 0;
    uint32_t sampler_commit_tokens = 0;
    uint64_t generation = 0;
    int32_t block_table[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
};

static inline uint32_t llama_mtp_qblock_paged_state_page_count_for_valid_tokens(const uint32_t valid_tail_tokens) {
    return valid_tail_tokens == 0 ? 0 : (valid_tail_tokens + LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS - 1u) / LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_consumer_map_logical_to_physical(
        const llama_mtp_qblock_paged_consumer_map_v1 & map,
        const uint32_t logical_token,
        uint32_t * physical_page,
        uint32_t * slot,
        uint64_t * physical_slot) {
    if (map.page_tokens != LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS || map.page_tokens == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_TOKENS;
    }
    if (map.block_table_pages > LLAMA_MTP_QBLOCK_PAGED_CONSUMER_MAP_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_MAP_TOO_LARGE;
    }
    if (map.valid_tail_tokens == 0 || map.block_table_pages == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES;
    }
    if (logical_token < map.logical_base_token) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t rel = logical_token - map.logical_base_token;
    if (rel >= map.valid_tail_tokens) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t logical_page = rel / map.page_tokens;
    if (logical_page >= map.block_table_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
    }
    const int32_t pp_i32 = map.block_table[logical_page];
    if (pp_i32 < 0 || uint32_t(pp_i32) >= map.physical_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
    }
    const uint32_t slot_u32 = rel % map.page_tokens;
    const uint32_t pp = uint32_t(pp_i32);
    if (physical_page != nullptr) {
        *physical_page = pp;
    }
    if (slot != nullptr) {
        *slot = slot_u32;
    }
    if (physical_slot != nullptr) {
        *physical_slot = uint64_t(pp) * uint64_t(map.page_tokens) + uint64_t(slot_u32);
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_consumer_map_validate_range(
        const llama_mtp_qblock_paged_consumer_map_v1 & map,
        const uint32_t logical_base,
        const uint32_t n_tokens) {
    if (n_tokens == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint64_t req_begin = uint64_t(logical_base);
    const uint64_t req_end = req_begin + uint64_t(n_tokens);
    if (req_end <= req_begin || req_end - 1u > uint64_t(UINT32_MAX)) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_consumer_map_logical_to_physical(
        map, logical_base, nullptr, nullptr, nullptr);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    return llama_mtp_qblock_paged_consumer_map_logical_to_physical(
        map, uint32_t(req_end - 1u), nullptr, nullptr, nullptr);
}

static inline uint32_t llama_mtp_qblock_tail_txn_commit_page_count(
        const llama_mtp_qblock_tail_txn_commit_v1 & desc) {
    return llama_mtp_qblock_paged_state_page_count_for_valid_tokens(desc.accepted_tokens);
}

static inline llama_mtp_qblock_tail_txn_commit_status llama_mtp_qblock_tail_txn_commit_validate_static(
        const llama_mtp_qblock_tail_txn_commit_v1 & desc) {
    if (desc.version != LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_VERSION) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_VERSION;
    }
    if (desc.abi_bytes != sizeof(llama_mtp_qblock_tail_txn_commit_v1)) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_ABI_BYTES;
    }
    if (desc.page_tokens != LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PAGE_TOKENS;
    }
    if (desc.accepted_tokens == 0) {
        if (desc.sampler_commit_tokens != 0) {
            return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_SAMPLER_COMMIT;
        }
        if (desc.final_state_slot != LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT) {
            return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_FINAL_STATE_SLOT;
        }
        if (desc.block_table_pages != 0) {
            return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_BLOCK_TABLE;
        }
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK;
    }
    if (desc.sampler_commit_tokens != desc.accepted_tokens) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_SAMPLER_COMMIT;
    }
    if (desc.final_state_slot == LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT ||
            desc.recurrent_slot_count == 0 || desc.final_state_slot >= desc.recurrent_slot_count) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_FINAL_STATE_SLOT;
    }
    if (desc.physical_pages == 0) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PHYSICAL_PAGES;
    }
    const uint32_t logical_pages = llama_mtp_qblock_tail_txn_commit_page_count(desc);
    if (logical_pages == 0 || logical_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES ||
            logical_pages > desc.block_table_pages) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_LOGICAL_PAGES;
    }
    for (uint32_t lp = 0; lp < logical_pages; ++lp) {
        const int32_t physical_page = desc.block_table[lp];
        if (physical_page < 0 || uint32_t(physical_page) >= desc.physical_pages) {
            return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_BLOCK_TABLE;
        }
        for (uint32_t prior = 0; prior < lp; ++prior) {
            if (desc.block_table[prior] == physical_page) {
                return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_BLOCK_TABLE;
            }
        }
    }
    return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK;
}

static inline llama_mtp_qblock_tail_txn_commit_status llama_mtp_qblock_tail_txn_commit_export_consumer_map(
        const llama_mtp_qblock_tail_txn_commit_v1 & desc,
        llama_mtp_qblock_paged_consumer_map_v1 * out_map) {
    if (out_map != nullptr) {
        *out_map = llama_mtp_qblock_paged_consumer_map_v1{};
    }
    const llama_mtp_qblock_tail_txn_commit_status status = llama_mtp_qblock_tail_txn_commit_validate_static(desc);
    if (status != LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK) {
        return status;
    }
    if (desc.accepted_tokens == 0) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_NO_VISIBLE_PAGES;
    }
    const uint32_t logical_pages = llama_mtp_qblock_tail_txn_commit_page_count(desc);
    if (logical_pages > LLAMA_MTP_QBLOCK_PAGED_CONSUMER_MAP_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_MAP_TOO_LARGE;
    }
    if (out_map == nullptr) {
        return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK;
    }
    out_map->logical_base_token = desc.logical_base_token;
    out_map->valid_tail_tokens = desc.accepted_tokens;
    out_map->page_tokens = desc.page_tokens;
    out_map->physical_pages = desc.physical_pages;
    out_map->block_table_pages = logical_pages;
    out_map->generation = desc.generation;
    for (uint32_t lp = 0; lp < logical_pages; ++lp) {
        out_map->block_table[lp] = desc.block_table[lp];
    }
    return LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK;
}

static inline uint8_t llama_mtp_qblock_paged_state_expected_page_valid_tokens(
        const uint32_t valid_tail_tokens,
        const uint32_t logical_page) {
    const uint32_t page_start = logical_page * LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS;
    if (valid_tail_tokens >= page_start + LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS) {
        return uint8_t(LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS);
    }
    if (valid_tail_tokens <= page_start) {
        return 0;
    }
    return uint8_t(valid_tail_tokens - page_start);
}

static inline void llama_mtp_qblock_paged_state_clear(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint64_t generation) {
    state = llama_mtp_qblock_paged_state_v1{};
    state.generation = generation;
    for (uint32_t i = 0; i < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++i) {
        state.block_table[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.managed_physical_page[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.owner[i] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE);
        state.refcount[i] = 0;
        state.page_valid_tokens[i] = 0;
        state.free_stack[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    }
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_init(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t logical_base_token,
        const uint32_t physical_pages,
        const uint64_t generation) {
    if ((logical_base_token & (LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS - 1u)) != 0) {
        llama_mtp_qblock_paged_state_clear(state, generation);
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_BASE;
    }
    if (physical_pages == 0 || physical_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES) {
        llama_mtp_qblock_paged_state_clear(state, generation);
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES;
    }

    state = llama_mtp_qblock_paged_state_v1{};
    state.active = 1;
    state.logical_base_token = logical_base_token;
    state.physical_pages = physical_pages;
    state.managed_pages = physical_pages;
    state.free_count = physical_pages;
    state.generation = generation;
    for (uint32_t i = 0; i < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++i) {
        state.block_table[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.managed_physical_page[i] = i < physical_pages ? int32_t(i) : LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.owner[i] = i < physical_pages ? uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE) : uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED);
        state.refcount[i] = 0;
        state.page_valid_tokens[i] = 0;
        state.free_stack[i] = i < physical_pages ? uint8_t(physical_pages - 1u - i) : LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline uint32_t llama_mtp_qblock_paged_state_find_managed_slot(
        const llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t physical_page) {
    for (uint32_t slot = 0; slot < state.managed_pages && slot < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++slot) {
        if (state.managed_physical_page[slot] == int32_t(physical_page)) {
            return slot;
        }
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_validate_static(
        const llama_mtp_qblock_paged_state_v1 & state) {
    if (!state.active) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
    }
    if (state.version != LLAMA_MTP_QBLOCK_PAGED_STATE_VERSION) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_VERSION;
    }
    if (state.abi_bytes != sizeof(llama_mtp_qblock_paged_state_v1)) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_ABI_BYTES;
    }
    if (state.page_tokens != LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_TOKENS;
    }
    if (state.physical_pages == 0 || state.managed_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES ||
            state.managed_pages > state.physical_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES;
    }
    if (state.managed_pages == 0 && (state.valid_tail_tokens != 0 || state.logical_pages != 0 || state.free_count != 0)) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES;
    }
    const uint32_t expected_logical_pages = llama_mtp_qblock_paged_state_page_count_for_valid_tokens(state.valid_tail_tokens);
    if (state.logical_pages != expected_logical_pages || state.logical_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_PAGES;
    }
    if (state.valid_tail_tokens > state.logical_pages * state.page_tokens) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_VALID_TOKENS;
    }

    for (uint32_t lp = 0; lp < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++lp) {
        const int32_t pp = state.block_table[lp];
        if (lp < state.logical_pages) {
            if (pp < 0 || uint32_t(pp) >= state.physical_pages) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
            }
        } else if (pp != LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
        }
    }

    const bool direct_owner_index = state.physical_pages <= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES &&
        state.managed_pages == state.physical_pages;
    for (uint32_t slot = 0; slot < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++slot) {
        const int32_t physical_page = state.managed_physical_page[slot];
        if (slot < state.managed_pages) {
            if (physical_page < 0 || uint32_t(physical_page) >= state.physical_pages) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE;
            }
            if (direct_owner_index && uint32_t(physical_page) != slot) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE;
            }
            for (uint32_t prior = 0; prior < slot; ++prior) {
                if (state.managed_physical_page[prior] == physical_page) {
                    return LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_MANAGED_PAGE;
                }
            }
        } else if (physical_page != LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE;
        }
    }

    for (uint32_t slot = 0; slot < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++slot) {
        const uint8_t owner = state.owner[slot];
        if (slot < state.managed_pages) {
            if (owner > uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN)) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER;
            }
            if (state.page_valid_tokens[slot] > state.page_tokens) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_VALID_TOKENS;
            }
            if (owner == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE)) {
                if (state.refcount[slot] != 0 || state.page_valid_tokens[slot] != 0) {
                    return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_REFCOUNT;
                }
            } else if (state.refcount[slot] != 1) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_REFCOUNT;
            }
        } else if (owner != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED) || state.refcount[slot] != 0 || state.page_valid_tokens[slot] != 0) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER;
        }
    }

    for (uint32_t lp = 0; lp < state.logical_pages; ++lp) {
        const uint32_t pp = uint32_t(state.block_table[lp]);
        const uint32_t meta = llama_mtp_qblock_paged_state_find_managed_slot(state, pp);
        if (meta >= state.managed_pages || state.owner[meta] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL)) {
            return meta >= state.managed_pages ? LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE : LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER;
        }
        const uint8_t expected = llama_mtp_qblock_paged_state_expected_page_valid_tokens(state.valid_tail_tokens, lp);
        if (state.page_valid_tokens[meta] != expected) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE_VALID_TOKENS;
        }
    }

    if (direct_owner_index) {
        if (state.free_count > state.managed_pages) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_COUNT;
        }
        uint64_t free_mask = 0;
        for (uint32_t i = 0; i < state.free_count; ++i) {
            const uint32_t slot = uint32_t(state.free_stack[i]);
            if (slot >= state.managed_pages) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE;
            }
            const uint64_t bit = uint64_t(1) << slot;
            if ((free_mask & bit) != 0) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_FREE_PAGE;
            }
            free_mask |= bit;
            if (state.owner[slot] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE)) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE;
            }
        }
        for (uint32_t slot = 0; slot < state.managed_pages; ++slot) {
            const bool listed_free = (free_mask & (uint64_t(1) << slot)) != 0;
            const bool owner_free = state.owner[slot] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE);
            if (listed_free != owner_free) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE;
            }
        }
    } else {
        if (state.free_count != 0) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_COUNT;
        }
        for (uint32_t slot = 0; slot < state.managed_pages; ++slot) {
            if (state.owner[slot] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE)) {
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE;
            }
        }
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_init_absolute_empty(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t logical_base_token,
        const uint32_t physical_pages,
        const uint64_t generation) {
    if (physical_pages == 0) {
        llama_mtp_qblock_paged_state_clear(state, generation);
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES;
    }

    state = llama_mtp_qblock_paged_state_v1{};
    state.active = 1;
    state.logical_base_token = logical_base_token;
    state.physical_pages = physical_pages;
    state.managed_pages = 0;
    state.free_count = 0;
    state.final_state_slot = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    state.generation = generation;
    for (uint32_t i = 0; i < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++i) {
        state.block_table[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.managed_physical_page[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.owner[i] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED);
        state.refcount[i] = 0;
        state.page_valid_tokens[i] = 0;
        state.free_stack[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    }
    return llama_mtp_qblock_paged_state_validate_static(state);
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_alloc_page(
        llama_mtp_qblock_paged_state_v1 & state,
        const llama_mtp_qblock_paged_page_owner owner,
        uint32_t * out_page) {
    if (owner != LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN && owner != LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER;
    }
    const llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_validate_static(state);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    if (!state.active || state.physical_pages != state.managed_pages || state.managed_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES || state.free_count == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_FREE_PAGE;
    }
    const uint32_t slot = uint32_t(state.free_stack[--state.free_count]);
    if (slot >= state.managed_pages || state.owner[slot] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE) ||
            state.managed_physical_page[slot] < 0 || uint32_t(state.managed_physical_page[slot]) >= state.physical_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_NOT_FREE;
    }
    state.owner[slot] = uint8_t(owner);
    state.refcount[slot] = 1;
    state.page_valid_tokens[slot] = 0;
    if (out_page != nullptr) {
        *out_page = uint32_t(state.managed_physical_page[slot]);
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_claim_txn_page(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t physical_page,
        uint32_t * out_slot) {
    llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_validate_static(state);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    if (!state.active) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_FREE_PAGE;
    }
    if (physical_page >= state.physical_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE;
    }

    const bool direct_owner_index = state.physical_pages <= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES &&
        state.managed_pages == state.physical_pages;
    const uint32_t existing = llama_mtp_qblock_paged_state_find_managed_slot(state, physical_page);
    if (existing < state.managed_pages) {
        if (state.owner[existing] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE)) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_MANAGED_PAGE;
        }
        if (!direct_owner_index) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_NOT_FREE;
        }
        uint32_t free_index = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
        for (uint32_t i = 0; i < state.free_count; ++i) {
            if (uint32_t(state.free_stack[i]) == existing) {
                free_index = i;
                break;
            }
        }
        if (free_index >= state.free_count) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_PAGE;
        }
        for (uint32_t i = free_index; i + 1 < state.free_count; ++i) {
            state.free_stack[i] = state.free_stack[i + 1];
        }
        state.free_stack[--state.free_count] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
        state.owner[existing] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN);
        state.refcount[existing] = 1;
        state.page_valid_tokens[existing] = 0;
        if (out_slot != nullptr) {
            *out_slot = existing;
        }
        return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
    }

    if (state.managed_pages >= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_FREE_PAGE;
    }
    const uint32_t slot = state.managed_pages++;
    state.managed_physical_page[slot] = int32_t(physical_page);
    state.owner[slot] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN);
    state.refcount[slot] = 1;
    state.page_valid_tokens[slot] = 0;
    state.free_stack[slot] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    if (out_slot != nullptr) {
        *out_slot = slot;
    }
    return llama_mtp_qblock_paged_state_validate_static(state);
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_release_txn_page(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t physical_page) {
    const uint32_t slot = llama_mtp_qblock_paged_state_find_managed_slot(state, physical_page);
    if (slot >= state.managed_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PAGE;
    }
    if (state.owner[slot] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN)) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_NOT_TXN;
    }

    const bool direct_owner_index = state.physical_pages <= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES &&
        state.managed_pages == state.physical_pages;
    if (direct_owner_index) {
        if (state.free_count >= state.managed_pages) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_FREE_COUNT;
        }
        state.owner[slot] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE);
        state.refcount[slot] = 0;
        state.page_valid_tokens[slot] = 0;
        state.free_stack[state.free_count++] = uint8_t(slot);
        return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
    }

    for (uint32_t i = slot; i + 1 < state.managed_pages; ++i) {
        state.managed_physical_page[i] = state.managed_physical_page[i + 1];
        state.owner[i] = state.owner[i + 1];
        state.refcount[i] = state.refcount[i + 1];
        state.page_valid_tokens[i] = state.page_valid_tokens[i + 1];
    }
    const uint32_t last = --state.managed_pages;
    state.managed_physical_page[last] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
    state.owner[last] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED);
    state.refcount[last] = 0;
    state.page_valid_tokens[last] = 0;
    state.free_stack[last] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_rollback_txn_pages(
        llama_mtp_qblock_paged_state_v1 & state) {
    const bool direct_owner_index = state.physical_pages <= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES &&
        state.managed_pages == state.physical_pages;
    for (uint32_t slot = 0; slot < state.managed_pages;) {
        if (state.owner[slot] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN)) {
            const uint32_t physical_page = uint32_t(state.managed_physical_page[slot]);
            const llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_release_txn_page(state, physical_page);
            if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
                return status;
            }
            if (direct_owner_index) {
                ++slot;
            }
        } else {
            ++slot;
        }
    }
    return llama_mtp_qblock_paged_state_validate_static(state);
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_commit_pages(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t valid_tail_tokens,
        const int32_t * block_table,
        const uint32_t block_table_pages,
        const uint32_t final_state_slot) {
    llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_validate_static(state);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    if (!state.active || valid_tail_tokens == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES;
    }
    if (state.managed_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES;
    }
    const uint32_t new_logical_pages = llama_mtp_qblock_paged_state_page_count_for_valid_tokens(valid_tail_tokens);
    if (new_logical_pages == 0 || new_logical_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES || new_logical_pages > block_table_pages) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_PAGES;
    }

    uint64_t accepted_slot_mask = 0;
    int32_t new_block_table[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    uint8_t new_page_valid[LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES] = {};
    for (uint32_t lp = 0; lp < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++lp) {
        new_block_table[lp] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
    }
    for (uint32_t lp = 0; lp < new_logical_pages; ++lp) {
        const int32_t physical_page = block_table ? block_table[lp] : LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        if (physical_page < 0 || uint32_t(physical_page) >= state.physical_pages) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
        }
        const uint32_t slot = llama_mtp_qblock_paged_state_find_managed_slot(state, uint32_t(physical_page));
        if (slot >= state.managed_pages) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_MANAGED_PAGE;
        }
        const uint64_t bit = uint64_t(1) << slot;
        if ((accepted_slot_mask & bit) != 0) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
        }
        if (state.owner[slot] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN) &&
                state.owner[slot] != uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL)) {
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_OWNER;
        }
        accepted_slot_mask |= bit;
        new_block_table[lp] = physical_page;
        new_page_valid[slot] = llama_mtp_qblock_paged_state_expected_page_valid_tokens(valid_tail_tokens, lp);
    }

    for (uint32_t lp = 0; lp < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++lp) {
        state.block_table[lp] = new_block_table[lp];
    }
    for (uint32_t slot = 0; slot < state.managed_pages; ++slot) {
        if ((accepted_slot_mask & (uint64_t(1) << slot)) != 0) {
            state.owner[slot] = uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL);
            state.refcount[slot] = 1;
            state.page_valid_tokens[slot] = new_page_valid[slot];
        }
    }
    state.valid_tail_tokens = valid_tail_tokens;
    state.logical_pages = new_logical_pages;
    state.final_state_slot = final_state_slot;
    ++state.generation;

    const bool direct_owner_index = state.physical_pages <= LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES &&
        state.managed_pages == state.physical_pages;
    for (uint32_t slot = 0; slot < state.managed_pages;) {
        if (state.owner[slot] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN)) {
            const uint32_t physical_page = uint32_t(state.managed_physical_page[slot]);
            status = llama_mtp_qblock_paged_state_release_txn_page(state, physical_page);
            if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
                return status;
            }
            if (direct_owner_index) {
                ++slot;
            }
        } else {
            ++slot;
        }
    }
    return llama_mtp_qblock_paged_state_validate_static(state);
}

static inline llama_mtp_qblock_tail_txn_commit_status llama_mtp_qblock_tail_txn_commit_apply(
        llama_mtp_qblock_paged_state_v1 & state,
        const llama_mtp_qblock_tail_txn_commit_v1 & desc) {
    llama_mtp_qblock_tail_txn_commit_status status = llama_mtp_qblock_tail_txn_commit_validate_static(desc);
    if (status != LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK) {
        return status;
    }
    const llama_mtp_qblock_paged_state_status state_status = desc.accepted_tokens == 0 ?
        llama_mtp_qblock_paged_state_rollback_txn_pages(state) :
        llama_mtp_qblock_paged_state_commit_pages(
            state, desc.accepted_tokens, desc.block_table, desc.block_table_pages, desc.final_state_slot);
    return state_status == LLAMA_MTP_QBLOCK_PAGED_STATE_OK ?
        LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_OK : LLAMA_MTP_QBLOCK_TAIL_TXN_COMMIT_BAD_PAGE_STATE;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_import_committed_pages(
        llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t logical_base_token,
        const uint32_t physical_pages,
        const uint32_t valid_tail_tokens,
        const int32_t * block_table,
        const uint32_t block_table_pages,
        const uint32_t final_state_slot,
        const uint64_t generation) {
    if (physical_pages == 0 || valid_tail_tokens == 0) {
        llama_mtp_qblock_paged_state_clear(state, generation);
        return physical_pages == 0 ? LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_PHYSICAL_PAGES : LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES;
    }
    const uint32_t logical_pages = llama_mtp_qblock_paged_state_page_count_for_valid_tokens(valid_tail_tokens);
    if (logical_pages == 0 || logical_pages > LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES || logical_pages > block_table_pages) {
        llama_mtp_qblock_paged_state_clear(state, generation);
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_LOGICAL_PAGES;
    }

    state = llama_mtp_qblock_paged_state_v1{};
    state.active = 1;
    state.logical_base_token = logical_base_token;
    state.valid_tail_tokens = valid_tail_tokens;
    state.logical_pages = logical_pages;
    state.physical_pages = physical_pages;
    state.managed_pages = logical_pages;
    state.free_count = 0;
    state.final_state_slot = final_state_slot;
    state.generation = generation;
    for (uint32_t i = 0; i < LLAMA_MTP_QBLOCK_PAGED_STATE_MAX_PAGES; ++i) {
        state.block_table[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.managed_physical_page[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        state.owner[i] = i < state.managed_pages ? uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL) : uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_RESERVED);
        state.refcount[i] = i < state.managed_pages ? 1 : 0;
        state.page_valid_tokens[i] = i < state.managed_pages ? llama_mtp_qblock_paged_state_expected_page_valid_tokens(valid_tail_tokens, i) : 0;
        state.free_stack[i] = LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT;
    }
    for (uint32_t lp = 0; lp < logical_pages; ++lp) {
        const int32_t physical_page = block_table ? block_table[lp] : LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE;
        if (physical_page < 0 || uint32_t(physical_page) >= physical_pages) {
            llama_mtp_qblock_paged_state_clear(state, generation + 1);
            return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
        }
        for (uint32_t prior = 0; prior < lp; ++prior) {
            if (state.block_table[prior] == physical_page) {
                llama_mtp_qblock_paged_state_clear(state, generation + 1);
                return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
            }
        }
        state.block_table[lp] = physical_page;
        state.managed_physical_page[lp] = physical_page;
    }
    return llama_mtp_qblock_paged_state_validate_static(state);
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_logical_to_physical(
        const llama_mtp_qblock_paged_state_v1 & state,
        const uint32_t logical_token,
        uint32_t * physical_page,
        uint32_t * slot) {
    llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_validate_static(state);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    if (!state.active || logical_token < state.logical_base_token) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t relative = logical_token - state.logical_base_token;
    if (relative >= state.valid_tail_tokens) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE;
    }
    const uint32_t lp = relative / state.page_tokens;
    if (lp >= state.logical_pages || state.block_table[lp] < 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE;
    }
    if (physical_page != nullptr) {
        *physical_page = uint32_t(state.block_table[lp]);
    }
    if (slot != nullptr) {
        *slot = relative & (state.page_tokens - 1u);
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}

static inline llama_mtp_qblock_paged_state_status llama_mtp_qblock_paged_state_export_consumer_map(
        const llama_mtp_qblock_paged_state_v1 & state,
        llama_mtp_qblock_paged_consumer_map_v1 * out_map) {
    if (out_map != nullptr) {
        *out_map = llama_mtp_qblock_paged_consumer_map_v1{};
    }
    llama_mtp_qblock_paged_state_status status = llama_mtp_qblock_paged_state_validate_static(state);
    if (status != LLAMA_MTP_QBLOCK_PAGED_STATE_OK) {
        return status;
    }
    if (!state.active || state.valid_tail_tokens == 0 || state.logical_pages == 0) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES;
    }
    if (state.logical_pages > LLAMA_MTP_QBLOCK_PAGED_CONSUMER_MAP_MAX_PAGES) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_MAP_TOO_LARGE;
    }
    if (out_map == nullptr) {
        return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
    }
    out_map->logical_base_token = state.logical_base_token;
    out_map->valid_tail_tokens = state.valid_tail_tokens;
    out_map->page_tokens = state.page_tokens;
    out_map->physical_pages = state.physical_pages;
    out_map->block_table_pages = state.logical_pages;
    out_map->generation = state.generation;
    for (uint32_t lp = 0; lp < state.logical_pages; ++lp) {
        out_map->block_table[lp] = state.block_table[lp];
    }
    return LLAMA_MTP_QBLOCK_PAGED_STATE_OK;
}
