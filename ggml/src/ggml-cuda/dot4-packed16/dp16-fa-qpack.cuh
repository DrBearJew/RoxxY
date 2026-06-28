// dp16-fa-qpack.cuh — DP16 FA organizer stages for transient Q
#pragma once

#include "../common.cuh"
#include "dp16-common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static constexpr int DP16_FA_QBLOCK_MAX_ROWS            = 16;
static constexpr int DP16_FA_QBLOCK_MAX_HEAD_SLOTS      = 8;
static constexpr int DP16_FA_QBLOCK_TAIL_PAGE_MAX_PAGES = 4;

// QBlock execution metadata shared by QPACK and DOT4/PV kernels.  This is the
// narrow verifier program contract: QPACK still owns transient-Q layout, while
// DOT4/PV consumes explicit row/head/tile policy instead of rediscovering it
// from route envs inside the kernel.  The first implementation keeps the row
// map contiguous, but every row/head validity check goes through this contract
// so later accepted-row maps, V reuse scopes, and verifier epilogues have a
// stable ABI surface.
struct dp16_fa_qblock_program {
    int enabled;

    int rows_per_cta;
    int row_valid_mask;
    int gqa_group;

    // Explicit local program maps. q_delta/h_delta are relative to the CTA's
    // q0/hq_base, so the first program is identity without baking identity into
    // DOT4/PV.  Future verifier programs can permute/drop rows by changing only
    // these fields and row_valid_mask.
    int row_q_delta[DP16_FA_QBLOCK_MAX_ROWS];
    int row_kind[DP16_FA_QBLOCK_MAX_ROWS];

    // Branch/tree metadata for QBlock as a verifier block IR. Current production
    // programs are a single linear branch: row 0 is the target/sample row and
    // row i>0 has parent i-1. Keeping this in the backend-visible program lets
    // later DFlash-style schedulers add sibling rows, tree masks, and per-row
    // output policies without inventing a parallel metadata channel.
    int row_parent[DP16_FA_QBLOCK_MAX_ROWS];
    int row_branch_id[DP16_FA_QBLOCK_MAX_ROWS];
    int row_candidate_rank[DP16_FA_QBLOCK_MAX_ROWS];
    int row_output_policy[DP16_FA_QBLOCK_MAX_ROWS];

    int head_slot_delta[DP16_FA_QBLOCK_MAX_HEAD_SLOTS];
    int rowmap_mode;
    int q_precision_mode;
    // Plain MTP verifier rows use the graph/mask tensor contract. Only enable
    // backend-local tree visibility when explicit QBlock op metadata supplies
    // row parent/tree data; default linear rows must match legacy GQA1/GQA6.
    int local_tree_mask_mode;

    int k_tile;
    int split_k;

    int qpack_layout;
    int qpack_scale_mode;
    int qpack_reuse_scope;

    int dot4_accum_mode;
    int score_output_mode;
    int softmax_mode;

    int pv_consume_mode;
    int pvblock_mode;
    int v_decode_mode;
    int v_reuse_scope;
    int o_residency;

    int output_mode;
    int commit_mode;

    // Optional read-side tail-page consumer. Logical K positions remain the
    // attention/mask contract; only physical K/V cache addresses are remapped.
    // This is fail-closed and inactive unless a host path binds a validated
    // page table into the program before launch.
    int tail_page_mode;
    int tail_page_logical_base_token;
    int tail_page_valid_tail_tokens;
    int tail_page_page_tokens;
    int tail_page_physical_pages;
    int tail_page_block_table_pages;
    int tail_page_block_table[DP16_FA_QBLOCK_TAIL_PAGE_MAX_PAGES];
};

enum dp16_fa_qblock_row_kind {
    DP16_FA_QBLOCK_ROW_INVALID = 0,
    DP16_FA_QBLOCK_ROW_TARGET,
    DP16_FA_QBLOCK_ROW_DRAFT,
};

enum dp16_fa_qblock_row_output_policy {
    DP16_FA_QBLOCK_ROW_OUTPUT_NONE = 0,
    DP16_FA_QBLOCK_ROW_OUTPUT_FULL_LOGITS,
    DP16_FA_QBLOCK_ROW_OUTPUT_VERIFY_TOP1,
    DP16_FA_QBLOCK_ROW_OUTPUT_VERIFY_TOPK,
    DP16_FA_QBLOCK_ROW_OUTPUT_ATTENTION_ONLY,
};

enum dp16_fa_qblock_rowmap_mode {
    DP16_FA_QBLOCK_ROWMAP_IDENTITY = 0,
    DP16_FA_QBLOCK_ROWMAP_TARGET_ONLY,
    DP16_FA_QBLOCK_ROWMAP_DROP_DRAFT1,
    DP16_FA_QBLOCK_ROWMAP_SWAP01,
    DP16_FA_QBLOCK_ROWMAP_DUP_TARGET,
};

enum dp16_fa_qblock_q_precision_mode {
    DP16_FA_QBLOCK_Q_PRECISION_QPACK = 0,
    DP16_FA_QBLOCK_Q_PRECISION_INLINE_ALL,
    DP16_FA_QBLOCK_Q_PRECISION_TARGET_INLINE,
    DP16_FA_QBLOCK_Q_PRECISION_DRAFT_INLINE,
};

enum dp16_fa_qblock_qpack_layout {
    DP16_FA_QBLOCK_QPACK_LAYOUT_NONE = 0,
    DP16_FA_QBLOCK_QPACK_LAYOUT_QBLOCK_MAJOR,
};

enum dp16_fa_qblock_scale_mode {
    DP16_FA_QBLOCK_SCALE_NONE = 0,
    DP16_FA_QBLOCK_SCALE_I8_BLOCK32,
};

enum dp16_fa_qblock_reuse_scope {
    DP16_FA_QBLOCK_REUSE_NONE = 0,
    DP16_FA_QBLOCK_REUSE_K_SHARD,
    DP16_FA_QBLOCK_REUSE_KV_TILE,
    DP16_FA_QBLOCK_REUSE_QBLOCK,
};

enum dp16_fa_qblock_dot4_mode {
    DP16_FA_QBLOCK_DOT4_NONE = 0,
    DP16_FA_QBLOCK_DOT4_I32_TRANSIENT,
};

enum dp16_fa_qblock_score_mode {
    DP16_FA_QBLOCK_SCORE_NONE = 0,
    DP16_FA_QBLOCK_SCORE_TILE_RESIDENT,
};

enum dp16_fa_qblock_softmax_mode {
    DP16_FA_QBLOCK_SOFTMAX_NONE = 0,
    DP16_FA_QBLOCK_SOFTMAX_ONLINE_ML,
};

enum dp16_fa_qblock_pv_mode {
    DP16_FA_QBLOCK_PV_NONE = 0,
    DP16_FA_QBLOCK_PV_CONSUME_INLINE,
};

enum dp16_fa_qblock_pvblock_mode {
    DP16_FA_QBLOCK_PVBLOCK_NONE = 0,
    DP16_FA_QBLOCK_PVBLOCK_EXACT_SCALAR,
};

enum dp16_fa_qblock_v_decode_mode {
    DP16_FA_QBLOCK_V_DECODE_NONE = 0,
    DP16_FA_QBLOCK_V_DECODE_DIRECT,
    DP16_FA_QBLOCK_V_DECODE_RAW_LDS,
    DP16_FA_QBLOCK_V_DECODE_STAGE_F32,
};

enum dp16_fa_qblock_o_residency {
    DP16_FA_QBLOCK_O_NONE = 0,
    DP16_FA_QBLOCK_O_REGISTER_UNTIL_TILE_DONE,
};

enum dp16_fa_qblock_output_mode {
    DP16_FA_QBLOCK_OUTPUT_NONE = 0,
    DP16_FA_QBLOCK_OUTPUT_ATTENTION_O,
};

enum dp16_fa_qblock_commit_mode {
    DP16_FA_QBLOCK_COMMIT_NONE = 0,
    DP16_FA_QBLOCK_COMMIT_VERIFIER_FAIL_CLOSED,
};

enum dp16_fa_qblock_tail_page_mode {
    DP16_FA_QBLOCK_TAIL_PAGE_NONE = 0,
    DP16_FA_QBLOCK_TAIL_PAGE_TABLE,
};

static __host__ __device__ __forceinline__ int dp16_fa_qblock_full_mask(const int n) {
    return n >= 31 ? 0x7fffffff : ((1 << n) - 1);
}

static inline const char * dp16_fa_qblock_rowmap_mode_name(const int mode) {
    switch (mode) {
        case DP16_FA_QBLOCK_ROWMAP_IDENTITY:    return "identity";
        case DP16_FA_QBLOCK_ROWMAP_TARGET_ONLY: return "target_only";
        case DP16_FA_QBLOCK_ROWMAP_DROP_DRAFT1: return "drop_draft1";
        case DP16_FA_QBLOCK_ROWMAP_SWAP01:      return "swap01";
        case DP16_FA_QBLOCK_ROWMAP_DUP_TARGET:  return "dup_target";
        default:                                return "unknown";
    }
}

static inline const char * dp16_fa_qblock_q_precision_mode_name(const int mode) {
    switch (mode) {
        case DP16_FA_QBLOCK_Q_PRECISION_QPACK:         return "qpack";
        case DP16_FA_QBLOCK_Q_PRECISION_INLINE_ALL:    return "inline_all";
        case DP16_FA_QBLOCK_Q_PRECISION_TARGET_INLINE: return "target_inline";
        case DP16_FA_QBLOCK_Q_PRECISION_DRAFT_INLINE:  return "draft_inline";
        default:                                      return "unknown";
    }
}

static inline const char * dp16_fa_qblock_row_output_policy_name(const int mode) {
    switch (mode) {
        case DP16_FA_QBLOCK_ROW_OUTPUT_NONE:           return "none";
        case DP16_FA_QBLOCK_ROW_OUTPUT_FULL_LOGITS:    return "full_logits";
        case DP16_FA_QBLOCK_ROW_OUTPUT_VERIFY_TOP1:    return "verify_top1";
        case DP16_FA_QBLOCK_ROW_OUTPUT_VERIFY_TOPK:    return "verify_topk";
        case DP16_FA_QBLOCK_ROW_OUTPUT_ATTENTION_ONLY: return "attention_only";
        default:                                      return "unknown";
    }
}

static inline void dp16_fa_qblock_program_invalidate_row(dp16_fa_qblock_program & p, const int qr) {
    if (qr < 0 || qr >= DP16_FA_QBLOCK_MAX_ROWS) {
        return;
    }
    p.row_valid_mask &= ~(1 << qr);
    p.row_kind[qr] = DP16_FA_QBLOCK_ROW_INVALID;
    p.row_parent[qr] = -1;
    p.row_output_policy[qr] = DP16_FA_QBLOCK_ROW_OUTPUT_NONE;
}

static inline void dp16_fa_qblock_program_apply_debug_rowmap(dp16_fa_qblock_program & p) {
    if (!p.enabled) {
        return;
    }

    const char * qprec = getenv("GGML_CUDA_DP16_FA_QBLOCK_Q_PRECISION");
    if (qprec == nullptr || qprec[0] == '\0') {
        qprec = getenv("LLAMA_MTP_QBLOCK_Q_PRECISION");
    }
    if (qprec != nullptr && qprec[0] != '\0') {
        if (strcmp(qprec, "qpack") == 0 || strcmp(qprec, "default") == 0) {
            p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_QPACK;
        } else if (strcmp(qprec, "inline_all") == 0) {
            p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_INLINE_ALL;
        } else if (strcmp(qprec, "target_inline") == 0) {
            p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_TARGET_INLINE;
        } else if (strcmp(qprec, "draft_inline") == 0) {
            p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_DRAFT_INLINE;
        } else {
            static bool qprec_warned = false;
            if (!qprec_warned) {
                fprintf(stderr, "DP16_FA_QBLOCK_Q_PRECISION: unknown mode '%s', using qpack\n", qprec);
                qprec_warned = true;
            }
            p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_QPACK;
        }
    }

    const char * mode = getenv("GGML_CUDA_DP16_FA_QBLOCK_ROWMAP_MODE");
    if (mode == nullptr || mode[0] == '\0') {
        mode = getenv("LLAMA_MTP_QBLOCK_ROWMAP_MODE");
    }
    if (mode == nullptr || mode[0] == '\0' || strcmp(mode, "identity") == 0 || strcmp(mode, "default") == 0) {
        p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_IDENTITY;
        return;
    }

    if (strcmp(mode, "target_only") == 0) {
        p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_TARGET_ONLY;
        for (int qr = 1; qr < DP16_FA_QBLOCK_MAX_ROWS; ++qr) {
            dp16_fa_qblock_program_invalidate_row(p, qr);
        }
        return;
    }

    if (strcmp(mode, "drop_draft1") == 0 || strcmp(mode, "disable_row1") == 0) {
        p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_DROP_DRAFT1;
        dp16_fa_qblock_program_invalidate_row(p, 1);
        return;
    }

    if (strcmp(mode, "swap01") == 0) {
        p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_SWAP01;
        if (p.rows_per_cta >= 2) {
            p.row_q_delta[0] = 1;
            p.row_q_delta[1] = 0;
            p.row_kind[0] = DP16_FA_QBLOCK_ROW_DRAFT;
            p.row_kind[1] = DP16_FA_QBLOCK_ROW_TARGET;
        }
        return;
    }

    if (strcmp(mode, "dup_target") == 0) {
        p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_DUP_TARGET;
        if (p.rows_per_cta >= 2) {
            p.row_q_delta[0] = 0;
            p.row_q_delta[1] = 0;
            p.row_kind[0] = DP16_FA_QBLOCK_ROW_TARGET;
            p.row_kind[1] = DP16_FA_QBLOCK_ROW_TARGET;
        }
        return;
    }

    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "DP16_FA_QBLOCK_ROWMAP_MODE: unknown mode '%s', using identity\n", mode);
        warned = true;
    }
    p.rowmap_mode = DP16_FA_QBLOCK_ROWMAP_IDENTITY;
}

static __host__ __device__ __forceinline__ dp16_fa_qblock_program dp16_fa_qblock_program_disabled() {
    dp16_fa_qblock_program p = {};
    return p;
}

static __host__ __device__ __forceinline__ dp16_fa_qblock_program dp16_fa_qblock_program_make(
        const bool enabled,
        const int rows_per_cta,
        const int gqa_group,
        const int k_tile,
        const int split_k,
        const dp16_fa_q_stage q_stage,
        const bool stage_v,
        const bool raw_lds_v,
        const bool direct_v) {
    dp16_fa_qblock_program p = {};
    if (!enabled) {
        return p;
    }

    p.enabled        = 1;
    p.rows_per_cta   = rows_per_cta;
    p.row_valid_mask = dp16_fa_qblock_full_mask(rows_per_cta);
    p.gqa_group      = gqa_group;
    p.rowmap_mode    = DP16_FA_QBLOCK_ROWMAP_IDENTITY;
    p.q_precision_mode = DP16_FA_QBLOCK_Q_PRECISION_QPACK;
    p.local_tree_mask_mode = 0;
    for (int i = 0; i < DP16_FA_QBLOCK_MAX_ROWS; ++i) {
        const bool live = i < rows_per_cta;
        p.row_q_delta[i] = i;
        p.row_kind[i] = live ? (i == 0 ? DP16_FA_QBLOCK_ROW_TARGET : DP16_FA_QBLOCK_ROW_DRAFT) : DP16_FA_QBLOCK_ROW_INVALID;
        p.row_parent[i] = live ? (i == 0 ? -1 : i - 1) : -1;
        p.row_branch_id[i] = 0;
        p.row_candidate_rank[i] = live ? (i == 0 ? -1 : 0) : -1;
        p.row_output_policy[i] = live ? DP16_FA_QBLOCK_ROW_OUTPUT_FULL_LOGITS : DP16_FA_QBLOCK_ROW_OUTPUT_NONE;
    }
    for (int i = 0; i < DP16_FA_QBLOCK_MAX_HEAD_SLOTS; ++i) {
        p.head_slot_delta[i] = i;
    }
    p.k_tile         = k_tile;
    p.split_k        = split_k;

    p.qpack_layout      = DP16_FA_QBLOCK_QPACK_LAYOUT_QBLOCK_MAJOR;
    p.qpack_scale_mode  = q_stage == DP16_FA_Q_STAGE_QPACK_I8_BLOCK32 ? DP16_FA_QBLOCK_SCALE_I8_BLOCK32 : DP16_FA_QBLOCK_SCALE_NONE;
    p.qpack_reuse_scope = q_stage == DP16_FA_Q_STAGE_QPACK_I8_BLOCK32 ? DP16_FA_QBLOCK_REUSE_K_SHARD : DP16_FA_QBLOCK_REUSE_NONE;

    p.dot4_accum_mode   = DP16_FA_QBLOCK_DOT4_I32_TRANSIENT;
    p.score_output_mode = DP16_FA_QBLOCK_SCORE_TILE_RESIDENT;
    p.softmax_mode      = DP16_FA_QBLOCK_SOFTMAX_ONLINE_ML;

    p.pv_consume_mode = DP16_FA_QBLOCK_PV_CONSUME_INLINE;
    p.pvblock_mode    = DP16_FA_QBLOCK_PVBLOCK_NONE;
    p.v_decode_mode   = stage_v ? DP16_FA_QBLOCK_V_DECODE_STAGE_F32 : (raw_lds_v ? DP16_FA_QBLOCK_V_DECODE_RAW_LDS : (direct_v ? DP16_FA_QBLOCK_V_DECODE_DIRECT : DP16_FA_QBLOCK_V_DECODE_NONE));
    p.v_reuse_scope   = DP16_FA_QBLOCK_REUSE_KV_TILE;
    p.o_residency     = DP16_FA_QBLOCK_O_REGISTER_UNTIL_TILE_DONE;

    p.output_mode = DP16_FA_QBLOCK_OUTPUT_ATTENTION_O;
    p.commit_mode = DP16_FA_QBLOCK_COMMIT_VERIFIER_FAIL_CLOSED;
    p.tail_page_mode = DP16_FA_QBLOCK_TAIL_PAGE_NONE;
    return p;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_q(
        const dp16_fa_qblock_program & program,
        const int q0,
        const int qr) {
    if (program.enabled && qr >= 0 && qr < DP16_FA_QBLOCK_MAX_ROWS) {
        return q0 + program.row_q_delta[qr];
    }
    return q0 + qr;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_hq(
        const dp16_fa_qblock_program & program,
        const int hq_base,
        const int slot) {
    if (program.enabled && slot >= 0 && slot < DP16_FA_QBLOCK_MAX_HEAD_SLOTS) {
        return hq_base + program.head_slot_delta[slot];
    }
    return hq_base + slot;
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_program_row_enabled(
        const dp16_fa_qblock_program & program,
        const int qr) {
    if (!program.enabled) {
        return true;
    }
    if (qr < 0 || qr >= DP16_FA_QBLOCK_MAX_ROWS || qr >= program.rows_per_cta) {
        return false;
    }
    return (program.row_valid_mask & (1 << qr)) != 0;
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_program_row_live(
        const dp16_fa_qblock_program & program,
        const int q,
        const int qr,
        const int nq) {
    return q >= 0 && q < nq && dp16_fa_qblock_program_row_enabled(program, qr);
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_row_parent(
        const dp16_fa_qblock_program & program,
        const int qr) {
    if (program.enabled && qr >= 0 && qr < DP16_FA_QBLOCK_MAX_ROWS) {
        return program.row_parent[qr];
    }
    return qr == 0 ? -1 : qr - 1;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_row_branch_id(
        const dp16_fa_qblock_program & program,
        const int qr) {
    if (program.enabled && qr >= 0 && qr < DP16_FA_QBLOCK_MAX_ROWS) {
        return program.row_branch_id[qr];
    }
    return 0;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_row_candidate_rank(
        const dp16_fa_qblock_program & program,
        const int qr) {
    if (program.enabled && qr >= 0 && qr < DP16_FA_QBLOCK_MAX_ROWS) {
        return program.row_candidate_rank[qr];
    }
    return qr == 0 ? -1 : 0;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_row_output_policy(
        const dp16_fa_qblock_program & program,
        const int qr) {
    if (program.enabled && qr >= 0 && qr < DP16_FA_QBLOCK_MAX_ROWS) {
        return program.row_output_policy[qr];
    }
    return DP16_FA_QBLOCK_ROW_OUTPUT_FULL_LOGITS;
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_program_row_writes_attention(
        const dp16_fa_qblock_program & program,
        const int qr) {
    return dp16_fa_qblock_program_row_output_policy(program, qr) != DP16_FA_QBLOCK_ROW_OUTPUT_NONE;
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_program_tree_allows_local_k(
        const dp16_fa_qblock_program & program,
        const int qr,
        const int local_k) {
    if (!program.enabled || local_k < 0) {
        return true;
    }
    if (qr < 0 || qr >= DP16_FA_QBLOCK_MAX_ROWS) {
        return false;
    }

    int cur = qr;
    for (int depth = 0; depth < DP16_FA_QBLOCK_MAX_ROWS; ++depth) {
        if (cur < 0) {
            return false;
        }
        if (cur >= DP16_FA_QBLOCK_MAX_ROWS || !dp16_fa_qblock_program_row_enabled(program, cur)) {
            return false;
        }
        if (program.row_q_delta[cur] == local_k) {
            return true;
        }
        const int parent = dp16_fa_qblock_program_row_parent(program, cur);
        if (parent == cur) {
            return false;
        }
        cur = parent;
    }
    return false;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_program_q_last_live(
        const dp16_fa_qblock_program & program,
        const int q0,
        const int fallback_rows_per_cta,
        const int nq) {
    if (!program.enabled) {
        return q0 + fallback_rows_per_cta - 1 < nq ? q0 + fallback_rows_per_cta - 1 : nq - 1;
    }

    int q_last = -1;
    const int rows = program.rows_per_cta < DP16_FA_QBLOCK_MAX_ROWS ? program.rows_per_cta : DP16_FA_QBLOCK_MAX_ROWS;
    for (int qr = 0; qr < rows; ++qr) {
        const int q = dp16_fa_qblock_program_q(program, q0, qr);
        if (dp16_fa_qblock_program_row_live(program, q, qr, nq) && q > q_last) {
            q_last = q;
        }
    }
    return q_last >= 0 ? q_last : (q0 < nq ? q0 : nq - 1);
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_tail_page_consumer_active(
        const dp16_fa_qblock_program & program) {
    return program.tail_page_mode == DP16_FA_QBLOCK_TAIL_PAGE_TABLE &&
        program.tail_page_page_tokens > 0 &&
        program.tail_page_valid_tail_tokens > 0 &&
        program.tail_page_block_table_pages > 0 &&
        program.tail_page_block_table_pages <= DP16_FA_QBLOCK_TAIL_PAGE_MAX_PAGES;
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_tail_page_covers_k(
        const dp16_fa_qblock_program & program,
        const int k) {
    return dp16_fa_qblock_tail_page_consumer_active(program) &&
        k >= program.tail_page_logical_base_token &&
        k < program.tail_page_logical_base_token + program.tail_page_valid_tail_tokens;
}

static __host__ __device__ __forceinline__ int dp16_fa_qblock_tail_page_physical_k(
        const dp16_fa_qblock_program & program,
        const int k) {
    if (!dp16_fa_qblock_tail_page_covers_k(program, k)) {
        return k;
    }
    const int rel = k - program.tail_page_logical_base_token;
    const int logical_page = rel / program.tail_page_page_tokens;
    if (logical_page < 0 || logical_page >= program.tail_page_block_table_pages ||
            logical_page >= DP16_FA_QBLOCK_TAIL_PAGE_MAX_PAGES) {
        return -1;
    }
    const int physical_page = program.tail_page_block_table[logical_page];
    if (physical_page < 0 || physical_page >= program.tail_page_physical_pages) {
        return -1;
    }
    return physical_page * program.tail_page_page_tokens + (rel - logical_page * program.tail_page_page_tokens);
}

static constexpr uint32_t DP16_FA_QBLOCK_META_MAGIC   = 0x51424d01u;
static constexpr int      DP16_FA_QBLOCK_META_WORD0   = 5;
static constexpr int      DP16_FA_QBLOCK_META_N_WORDS = 9;
static constexpr int      DP16_FA_QBLOCK_META_HEADER  = 14;

static inline uint32_t dp16_fa_qblock_meta_get_bits(const uint32_t words[DP16_FA_QBLOCK_META_N_WORDS], uint32_t bit, uint32_t width) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < width; ++i) {
        const uint32_t src_bit = bit + i;
        const uint32_t word = src_bit / 32u;
        const uint32_t off  = src_bit % 32u;
        if (word < DP16_FA_QBLOCK_META_N_WORDS && ((words[word] >> off) & 1u) != 0) {
            value |= 1u << i;
        }
    }
    return value;
}

static inline int dp16_fa_qblock_meta_decode_signed5(const uint32_t v) {
    return (int) (v & 0x1fu) - 1;
}

static inline int dp16_fa_qblock_op_metadata_n_rows(const int32_t * op_params) {
    if (op_params == nullptr) {
        return 0;
    }
    const uint32_t header = (uint32_t) op_params[DP16_FA_QBLOCK_META_HEADER];
    for (int n = 1; n <= DP16_FA_QBLOCK_MAX_ROWS; ++n) {
        if (header == (DP16_FA_QBLOCK_META_MAGIC ^ (uint32_t) n)) {
            return n;
        }
    }
    return 0;
}

static inline void dp16_fa_qblock_program_apply_op_metadata(
        dp16_fa_qblock_program & program,
        const int32_t * op_params) {
    if (!program.enabled || op_params == nullptr) {
        return;
    }

    const int n_meta_rows = dp16_fa_qblock_op_metadata_n_rows(op_params);
    if (n_meta_rows <= 0) {
        return;
    }

    uint32_t words[DP16_FA_QBLOCK_META_N_WORDS];
    for (int i = 0; i < DP16_FA_QBLOCK_META_N_WORDS; ++i) {
        words[i] = (uint32_t) op_params[DP16_FA_QBLOCK_META_WORD0 + i];
    }

    const int rows = n_meta_rows < program.rows_per_cta ? n_meta_rows : program.rows_per_cta;
    program.local_tree_mask_mode = rows > 0 ? 1 : program.local_tree_mask_mode;
    for (int r = 0; r < rows && r < DP16_FA_QBLOCK_MAX_ROWS; ++r) {
        const uint32_t base = (uint32_t) r * 17u;
        program.row_parent[r]         = dp16_fa_qblock_meta_decode_signed5(dp16_fa_qblock_meta_get_bits(words, base +  0u, 5u));
        program.row_branch_id[r]      = (int) dp16_fa_qblock_meta_get_bits(words, base +  5u, 4u);
        program.row_candidate_rank[r] = dp16_fa_qblock_meta_decode_signed5(dp16_fa_qblock_meta_get_bits(words, base +  9u, 5u));
        program.row_output_policy[r]  = (int) dp16_fa_qblock_meta_get_bits(words, base + 14u, 3u);
    }
}

static __host__ __device__ __forceinline__ bool dp16_fa_qblock_program_head_live(
        const dp16_fa_qblock_program & program,
        const int hq,
        const int slot,
        const int hk,
        const int n_heads_q,
        const int gqa_ratio) {
    const bool slot_ok = !program.enabled || (slot >= 0 && slot < program.gqa_group && slot < DP16_FA_QBLOCK_MAX_HEAD_SLOTS);
    return slot_ok && hq < n_heads_q && hq < (hk + 1) * gqa_ratio;
}

struct dp16_fa_q_view {
    dp16_fa_q_stage stage;

    const float * q_f32;
    const int   * qpack_payload;
    const float * qpack_scales;

    int nq;
    int n_heads_q;
    int batch;
    int d;

    int64_t q_nb01;
    int64_t q_nb02;
    int64_t q_nb03;

    size_t payload_stride_row_i32;
    size_t scales_stride_row_f32;

    dp16_fa_qblock_program qblock_program;
};

struct dp16_fa_q_workspace {
    ggml_cuda_pool_alloc<int>   payload;
    ggml_cuda_pool_alloc<float> scales;

    explicit dp16_fa_q_workspace(ggml_cuda_pool & pool) : payload(pool), scales(pool) {}

    dp16_fa_q_workspace(const dp16_fa_q_workspace &) = delete;
    dp16_fa_q_workspace & operator=(const dp16_fa_q_workspace &) = delete;
};

static inline dp16_fa_q_view dp16_fa_make_inline_q_view(const ggml_tensor * Q, const int d) {
    dp16_fa_q_view view = {};
    view.stage = DP16_FA_Q_STAGE_INLINE;
    view.q_f32 = (const float *) Q->data;
    view.qpack_payload = nullptr;
    view.qpack_scales = nullptr;
    view.nq = (int) Q->ne[1];
    view.n_heads_q = (int) Q->ne[2];
    view.batch = (int) Q->ne[3];
    view.d = d;
    view.q_nb01 = Q->nb[1];
    view.q_nb02 = Q->nb[2];
    view.q_nb03 = Q->nb[3];
    view.payload_stride_row_i32 = 0;
    view.scales_stride_row_f32 = 0;
    view.qblock_program = dp16_fa_qblock_program_disabled();
    return view;
}

// Pack four signed int8 lanes into the packed16/DOT4 i32 payload convention.
static __device__ __forceinline__ int dp16_fa_qpack_pack_i8x4(
        const int q0, const int q1, const int q2, const int q3) {
    const uint32_t b0 = static_cast<uint8_t>(static_cast<int8_t>(q0));
    const uint32_t b1 = static_cast<uint8_t>(static_cast<int8_t>(q1));
    const uint32_t b2 = static_cast<uint8_t>(static_cast<int8_t>(q2));
    const uint32_t b3 = static_cast<uint8_t>(static_cast<int8_t>(q3));
    return static_cast<int>((b0 << 0) | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

static __device__ __forceinline__ int dp16_fa_qpack_clamp_i8(const int x) {
    return x < -127 ? -127 : (x > 127 ? 127 : x);
}

static __device__ __forceinline__ int dp16_fa_qpack_quant_i8(const float x, const float inv_scale) {
    return dp16_fa_qpack_clamp_i8((int) lrintf(x * inv_scale));
}

static __device__ __forceinline__ float dp16_fa_qpack_load_q_f32(
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

// DP16 FA QPACK organizer. It materializes transient f32 Q as q8/block32
// payload+float-scale rows keyed by (batch, head_q, q). DOT4 attention lanes
// can consume the prepared view across K shards/stages. The row-parallel kernel
// maps one CTA to one Q row and one warp to each block32 scale group.
template<int D, int THREADS>
static __global__ __launch_bounds__(THREADS, 1) void dp16_fa_qpack_i8_block32_row_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ qpack_payload,
        float       * __restrict__ qpack_scales,
        int64_t q_nb01,
        int64_t q_nb02,
        int64_t q_nb03,
        int nq,
        int n_heads_q) {
    static_assert(D % QK8_0 == 0, "DP16 FA QPACK requires D multiple of QK8_0");
    static_assert(QK8_0 == WARP_SIZE, "DP16 FA QPACK row kernel maps one warp to one block32");
    static_assert(THREADS >= D, "DP16 FA QPACK row kernel expects at least one thread per D lane");

    constexpr int Q_BLOCKS = D / QK8_0;
    constexpr int Q_WORDS  = D / 4;

    const int tid = int(threadIdx.x);
    if (tid >= D) {
        return;
    }

    const int q  = int(blockIdx.x);
    const int hq = int(blockIdx.y);
    const int b  = int(blockIdx.z);
    if (q >= nq || hq >= n_heads_q) {
        return;
    }

    const int qb   = tid / QK8_0;
    const int lane = tid - qb * QK8_0;
    const float x = dp16_fa_qpack_load_q_f32(Q, q_nb01, q_nb02, q_nb03, q, hq, b, tid);
    const float amax = warp_reduce_max<WARP_SIZE>(fabsf(x));
    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;

    const size_t row = (size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q);
    if (lane == 0) {
        qpack_scales[row * size_t(Q_BLOCKS) + size_t(qb)] = scale;
    }
    if ((lane & 3) == 0) {
        int qs[4] = {0, 0, 0, 0};
        if (amax > 0.0f) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = qb * QK8_0 + lane + j;
                qs[j] = dp16_fa_qpack_quant_i8(
                    dp16_fa_qpack_load_q_f32(Q, q_nb01, q_nb02, q_nb03, q, hq, b, d), inv_scale);
            }
        }
        qpack_payload[row * size_t(Q_WORDS) + size_t(tid / 4)] = dp16_fa_qpack_pack_i8x4(qs[0], qs[1], qs[2], qs[3]);
    }
}

template<int D, int THREADS>
static inline dp16_fa_q_view dp16_fa_prepare_q_stage(
        hipStream_t stream,
        const dp16_fa_q_stage stage,
        const ggml_tensor * Q,
        dp16_fa_q_workspace & workspace) {
    dp16_fa_q_view view = dp16_fa_make_inline_q_view(Q, D);
    if (stage == DP16_FA_Q_STAGE_INLINE) {
        return view;
    }
    if (stage != DP16_FA_Q_STAGE_QPACK_I8_BLOCK32) {
        GGML_ABORT("unsupported DP16 FA Q stage=%s", dp16_fa_q_stage_name(stage));
    }

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(Q->ne[0] == D);

    const size_t qpack_rows = size_t(view.batch) * size_t(view.n_heads_q) * size_t(view.nq);
    int   * payload = workspace.payload.alloc(qpack_rows * size_t(D / 4));
    float * scales  = workspace.scales.alloc (qpack_rows * size_t(D / QK8_0));

    const dim3 block(THREADS);
    const dim3 grid(view.nq, view.n_heads_q, view.batch);
    dp16_fa_qpack_i8_block32_row_kernel<D, THREADS><<<grid, block, 0, stream>>>(
        view.q_f32, payload, scales,
        view.q_nb01, view.q_nb02, view.q_nb03,
        view.nq, view.n_heads_q);

    view.stage = DP16_FA_Q_STAGE_QPACK_I8_BLOCK32;
    view.qpack_payload = payload;
    view.qpack_scales = scales;
    view.payload_stride_row_i32 = D / 4;
    view.scales_stride_row_f32 = D / QK8_0;
    return view;
}
