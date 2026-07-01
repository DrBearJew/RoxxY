#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_pre_norm / llama_get_embeddings_pre_norm_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <limits>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE}, // local backwards-compatible alias
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3}, // local backwards-compatible alias
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"mtp",           COMMON_SPECULATIVE_TYPE_DRAFT_MTP},    // local backwards-compatible alias
    {"draft-jetspec", COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC},
    {"jetspec",       COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC}, // local experimental alias
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}, // local backwards-compatible alias

    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},   // local backwards-compatible alias
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V}, // local backwards-compatible alias
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},     // local backwards-compatible alias
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}    // local backwards-compatible alias
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static uint64_t common_speculative_fnv1a64(const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool common_speculative_hidden_shift_trace_enabled() {
    const char * env = getenv("LLAMA_MTP_HIDDEN_SHIFT_TRACE");
    return env && atoi(env) != 0;
}

static bool common_speculative_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

static constexpr int32_t JETSPEC_QWEN36_DRAFT_BLOCK_SIZE  = 16;
static constexpr int32_t JETSPEC_QWEN36_TARGET_TAP_COUNT  = 5;
static constexpr int32_t JETSPEC_QWEN36_TARGET_HIDDEN     = 2048;
static constexpr int32_t JETSPEC_QWEN36_TARGET_LAYERS     = 40;
static constexpr int32_t JETSPEC_QWEN36_TARGET_TAP_WIDTH  = JETSPEC_QWEN36_TARGET_TAP_COUNT * JETSPEC_QWEN36_TARGET_HIDDEN;
static constexpr int32_t JETSPEC_QWEN36_DRAFT_LAYERS      = 8;
static constexpr int32_t JETSPEC_QWEN36_DRAFT_HEADS       = 32;
static constexpr int32_t JETSPEC_QWEN36_DRAFT_HEADS_KV    = 4;
static constexpr int32_t JETSPEC_QWEN36_VOCAB_SIZE        = 248320;
static constexpr int32_t JETSPEC_TRANSACTION_PHASE_COUNT  = 9;
static constexpr int32_t JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT = 7;
static constexpr const char * JETSPEC_TRANSACTION_PHASE_ORDER =
    "snapshot_pre_round|reserve_transient_tree_pages|build_tree|build_verify_mask|accept_path|commit_tokens|commit_hidden_kv_survivors|discard_rejected_branches|publish_post_commit_state";
static constexpr const char * JETSPEC_TRANSACTION_ROLLBACK_POINTS =
    "after_reserve|after_build_tree|after_verify_mask|after_accept|after_token_commit|after_hidden_kv_commit|after_rejected_discard";
static constexpr const char * JETSPEC_PRE_ROUND_SNAPSHOT_PHASE = "snapshot_pre_round";
static constexpr const char * JETSPEC_TRANSIENT_RESERVATION_PHASE = "reserve_transient_tree_pages";
static constexpr const char * JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT = "after_reserve";
static constexpr const char * JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR = "transient_reservation_descriptor_only";
static constexpr const char * JETSPEC_TREE_BUILD_PHASE = "build_tree";
static constexpr const char * JETSPEC_TREE_BUILD_ROLLBACK_POINT = "after_build_tree";
static constexpr const char * JETSPEC_TREE_BUILD_DESCRIPTOR = "tree_build_descriptor_only";
static constexpr const char * JETSPEC_ROOT_TREE_RUNTIME_PHASE = "root_tree_runtime";
static constexpr const char * JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE = "root_verify_mask_runtime";
static constexpr const char * JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_RUNTIME_PHASE = "root_anchor_accept_path_runtime";
static constexpr const char * JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE = "root_token_commit_noop_runtime";
static constexpr const char * JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE = "root_hidden_kv_commit_noop_runtime";
static constexpr const char * JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE = "root_rejected_branch_discard_noop_runtime";
static constexpr const char * JETSPEC_ROOT_PUBLISH_GATE_NOOP_RUNTIME_PHASE = "root_publish_gate_noop_runtime";
static constexpr const char * JETSPEC_TOPK_TREE_RUNTIME_PHASE = "topk_tree_runtime";
static constexpr const char * JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE = "topk_verify_mask_runtime";
static constexpr const char * JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE = "topk_accept_boundary_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_CANARY_RUNTIME_PHASE = "real_draft_head_canary";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY_RUNTIME_PHASE = "real_draft_head_logits_canary";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE = "real_draft_head_topk_candidate_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_RUNTIME_PHASE = "real_draft_head_topk_tree_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_RUNTIME_PHASE = "real_draft_head_topk_verify_mask_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE = "real_draft_head_topk_accept_boundary_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_RUNTIME_PHASE = "real_draft_head_topk_accept_path_descriptor_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_RUNTIME_PHASE = "real_draft_head_topk_token_commit_noop_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE = "real_draft_head_topk_hidden_kv_commit_noop_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE = "real_draft_head_topk_rejected_branch_discard_noop_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_RUNTIME_PHASE = "real_draft_head_topk_publish_gate_noop_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY_RUNTIME_PHASE = "real_draft_head_topk_target_logits_walk_canary_runtime";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE = "draft_head_full_vocab_logits";
static constexpr const char * JETSPEC_TARGET_LOGITS_SOURCE = "target_model_full_vocab_logits";
static constexpr const char * JETSPEC_TARGET_LOGITS_WALK_ROW_SEMANTICS = "parent_position_scores_candidate_children";
static constexpr const char * JETSPEC_ACCEPT_DECISION_SOURCE_TARGET_LOGITS_CANARY_ONLY = "target_logits_canary_only_no_accept";
static constexpr const char * JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS = "rank_stable_descending_logit";
static constexpr const char * JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX = "synthetic_full_vocab_softmax";
static constexpr const char * JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS = "none_no_logits";
static constexpr const char * JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS = "none_no_target_logits";
static constexpr int32_t JETSPEC_TOPK_ABI_WIDTH = 2;
static constexpr int32_t JETSPEC_TOPK_ABI_DEPTH = 1;
static constexpr int32_t JETSPEC_TOPK_ABI_NODES = 3;
static constexpr int32_t JETSPEC_TOPK_ABI_NON_ROOT_NODES = 2;
static constexpr int32_t JETSPEC_TOPK_ABI_MASK_ENTRIES = 5;
static constexpr const char * JETSPEC_VERIFY_MASK_PHASE = "build_verify_mask";
static constexpr const char * JETSPEC_VERIFY_MASK_ROLLBACK_POINT = "after_verify_mask";
static constexpr const char * JETSPEC_VERIFY_MASK_DESCRIPTOR = "verify_mask_descriptor_only";
static constexpr const char * JETSPEC_ACCEPT_PATH_PHASE = "accept_path";
static constexpr const char * JETSPEC_ACCEPT_PATH_ROLLBACK_POINT = "after_accept";
static constexpr const char * JETSPEC_ACCEPT_PATH_DESCRIPTOR = "accept_path_descriptor_only";
static constexpr const char * JETSPEC_TOKEN_COMMIT_PHASE = "commit_tokens";
static constexpr const char * JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT = "after_token_commit";
static constexpr const char * JETSPEC_TOKEN_COMMIT_DESCRIPTOR = "token_commit_descriptor_only";
static constexpr const char * JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE = "commit_hidden_kv_survivors";
static constexpr const char * JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT = "after_hidden_kv_commit";
static constexpr const char * JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_DESCRIPTOR = "hidden_kv_survivor_commit_descriptor_only";
static constexpr const char * JETSPEC_REJECTED_BRANCH_DISCARD_PHASE = "discard_rejected_branches";
static constexpr const char * JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT = "after_rejected_discard";
static constexpr const char * JETSPEC_REJECTED_BRANCH_DISCARD_DESCRIPTOR = "rejected_branch_discard_descriptor_only";
static constexpr const char * JETSPEC_PUBLISH_GATE_PHASE = "publish_post_commit_state";
static constexpr const char * JETSPEC_PUBLISH_GATE_DESCRIPTOR = "publish_gate_descriptor_only";
static constexpr int32_t JETSPEC_TREE_ROOT_PARENT = -1;
static constexpr int32_t JETSPEC_TREE_ROOT_DEPTH = 0;

static bool common_speculative_jetspec_expect_i32(const char * what, int32_t actual, int32_t expected, std::string & reason) {
    if (actual == expected) {
        return true;
    }
    reason = std::string(what) + "=" + std::to_string(actual) + " expected=" + std::to_string(expected);
    return false;
}

static bool common_speculative_jetspec_expect_meta_str(
        const llama_model * model, const char * key, const char * expected, std::string & reason) {
    char value[128] = {};
    const int32_t n = llama_model_meta_val_str(model, key, value, sizeof(value));
    if (n < 0) {
        reason = std::string("missing metadata ") + key;
        return false;
    }
    if (std::strcmp(value, expected) != 0) {
        reason = std::string(key) + "=" + value + " expected=" + expected;
        return false;
    }
    return true;
}

static bool common_speculative_jetspec_meta_i32(
        const llama_model * model, const char * key, int32_t & out, std::string & reason, bool required = true) {
    char value[128] = {};
    const int32_t n = llama_model_meta_val_str(model, key, value, sizeof(value));
    if (n < 0) {
        if (!required) {
            return false;
        }
        reason = std::string("missing metadata ") + key;
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0') || parsed < INT32_MIN || parsed > INT32_MAX) {
        reason = std::string("invalid integer metadata ") + key + "=" + value;
        return false;
    }
    out = (int32_t) parsed;
    return true;
}

static bool common_speculative_jetspec_preflight(
        const common_params_speculative_draft & params, int32_t tap_count, int32_t tap_width, std::string & reason) {
    if (params.ctx_tgt == nullptr) {
        reason = "missing target context";
        return false;
    }

    const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
    const llama_model * model_dft = params.ctx_dft != nullptr ? llama_get_model(params.ctx_dft) : params.model;
    if (model_tgt == nullptr || model_dft == nullptr) {
        reason = "missing target or draft model";
        return false;
    }

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    if (vocab_tgt == nullptr) {
        reason = "missing target vocab";
        return false;
    }

    if (!common_speculative_jetspec_expect_meta_str(model_dft, "general.architecture", "jetspec_qwen3_draft_head", reason) ||
        !common_speculative_jetspec_expect_meta_str(model_dft, "jetspec.architecture", "qwen3_draft_head", reason) ||
        !common_speculative_jetspec_expect_meta_str(model_dft, "jetspec.source_architecture", "DFlashDraftModel", reason) ||
        !common_speculative_jetspec_expect_meta_str(model_dft, "jetspec.tensor_data_dtype", "bfloat16", reason)) {
        return false;
    }

    int32_t draft_vocab = 0;
    if (!common_speculative_jetspec_meta_i32(model_dft, "jetspec.vocab_size", draft_vocab, reason)) {
        return false;
    }

    int32_t target_layers = llama_model_n_layer(model_tgt);
    int32_t target_nextn = 0;
    std::string optional_reason;
    if (common_speculative_jetspec_meta_i32(model_tgt, "qwen35moe.nextn_predict_layers", target_nextn, optional_reason, false) &&
            target_nextn > 0 && target_layers - target_nextn == JETSPEC_QWEN36_TARGET_LAYERS) {
        target_layers -= target_nextn;
    }

    const int32_t target_hidden = llama_model_n_embd(model_tgt);
    if (!common_speculative_jetspec_expect_i32("target.n_embd", target_hidden, JETSPEC_QWEN36_TARGET_HIDDEN, reason) ||
        !common_speculative_jetspec_expect_i32("target.effective_n_layer", target_layers, JETSPEC_QWEN36_TARGET_LAYERS, reason) ||
        !common_speculative_jetspec_expect_i32("target.vocab", llama_vocab_n_tokens(vocab_tgt), JETSPEC_QWEN36_VOCAB_SIZE, reason) ||
        !common_speculative_jetspec_expect_i32("draft.n_ctx_train", llama_model_n_ctx_train(model_dft), JETSPEC_QWEN36_DRAFT_BLOCK_SIZE, reason) ||
        !common_speculative_jetspec_expect_i32("draft.n_embd", llama_model_n_embd(model_dft), JETSPEC_QWEN36_TARGET_HIDDEN, reason) ||
        !common_speculative_jetspec_expect_i32("draft.n_layer", llama_model_n_layer(model_dft), JETSPEC_QWEN36_DRAFT_LAYERS, reason) ||
        !common_speculative_jetspec_expect_i32("draft.n_head", llama_model_n_head(model_dft), JETSPEC_QWEN36_DRAFT_HEADS, reason) ||
        !common_speculative_jetspec_expect_i32("draft.n_head_kv", llama_model_n_head_kv(model_dft), JETSPEC_QWEN36_DRAFT_HEADS_KV, reason) ||
        !common_speculative_jetspec_expect_i32("draft.vocab", draft_vocab, JETSPEC_QWEN36_VOCAB_SIZE, reason) ||
        !common_speculative_jetspec_expect_i32("target_tap_count", tap_count, JETSPEC_QWEN36_TARGET_TAP_COUNT, reason) ||
        !common_speculative_jetspec_expect_i32("target_tap_width", tap_width, JETSPEC_QWEN36_TARGET_TAP_WIDTH, reason) ||
        !common_speculative_jetspec_expect_i32("target_tap_width_vs_target", tap_width, tap_count * target_hidden, reason)) {
        return false;
    }

    return true;
}

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // Per-depth generated/accepted token counts. Index 0 is draft depth 1.
    std::vector<size_t> n_gen_tokens_by_depth;
    std::vector<size_t> n_acc_tokens_by_depth;

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    void record_generated_depths(size_t n_tokens) {
        if (n_gen_tokens_by_depth.size() < n_tokens) {
            n_gen_tokens_by_depth.resize(n_tokens, 0);
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            n_gen_tokens_by_depth[i]++;
        }
    }

    void record_accepted_depths(size_t n_tokens) {
        if (n_acc_tokens_by_depth.size() < n_tokens) {
            n_acc_tokens_by_depth.resize(n_tokens, 0);
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            n_acc_tokens_by_depth[i]++;
        }
    }

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual bool process_with_pre_norm(const llama_batch & batch, const float * h_pre_norm) {
        (void) h_pre_norm;
        return process(batch);
    }

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_pre_norm() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & /*params*/, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

static int32_t common_speculative_jetspec_real_draft_head_canary_eval(llama_context * ctx, llama_batch batch) {
    return llama_decode(ctx, batch);
}

struct common_speculative_impl_draft_jetspec : public common_speculative_impl {
    common_params_speculative_draft params; // reuses draft ctx_tgt/ctx_dft binding; P5AI can create ctx_dft behind an explicit canary gate

    enum class jetspec_runtime_phase {
        waiting_for_target_taps,
        waiting_for_pre_round_snapshot,
        pre_round_snapshot_ready,
        target_taps_captured,
        transaction_plan_scaffold_ready,
        transient_reservation_descriptor_ready,
        tree_build_descriptor_ready,
        tree_build_runtime_ready,
        verify_mask_runtime_ready,
        verify_mask_descriptor_ready,
        real_draft_head_canary_ready,
        real_draft_head_logits_canary_ready,
        real_draft_head_topk_candidate_ready,
        real_draft_head_topk_tree_ready,
        real_draft_head_topk_verify_mask_ready,
        real_draft_head_topk_accept_boundary_ready,
        real_draft_head_topk_accept_path_descriptor_ready,
        real_draft_head_topk_token_commit_noop_ready,
        real_draft_head_topk_hidden_kv_commit_noop_ready,
        real_draft_head_topk_rejected_branch_discard_noop_ready,
        real_draft_head_topk_publish_gate_noop_ready,
        real_draft_head_topk_target_logits_walk_canary_ready,
        accept_path_runtime_ready,
        token_commit_runtime_ready,
        hidden_kv_commit_runtime_ready,
        rejected_branch_discard_runtime_ready,
        publish_gate_runtime_ready,
        accept_path_descriptor_ready,
        token_commit_descriptor_ready,
        hidden_kv_survivor_commit_descriptor_ready,
        rejected_branch_discard_descriptor_ready,
        publish_gate_descriptor_ready,
        disabled,
    };

    enum class jetspec_runtime_failure {
        none,
        missing_target_context,
        invalid_binding,
        invalid_target_taps,
        invalid_pre_round_snapshot,
        invalid_transaction_plan,
        invalid_transient_reservation_descriptor,
        invalid_tree_build_descriptor,
        invalid_root_tree_runtime,
        invalid_root_verify_mask_runtime,
        invalid_verify_mask_descriptor,
        invalid_root_anchor_accept_path_runtime,
        invalid_root_token_commit_noop_runtime,
        invalid_root_hidden_kv_commit_noop_runtime,
        invalid_root_rejected_branch_discard_noop_runtime,
        invalid_root_publish_gate_noop_runtime,
        invalid_topk_tree_runtime,
        invalid_topk_verify_mask_runtime,
        invalid_topk_accept_boundary_runtime,
        invalid_real_draft_head_canary_runtime,
        invalid_real_draft_head_logits_canary_runtime,
        invalid_real_draft_head_topk_candidate_runtime,
        invalid_real_draft_head_topk_tree_runtime,
        invalid_real_draft_head_topk_verify_mask_runtime,
        invalid_real_draft_head_topk_accept_boundary_runtime,
        invalid_real_draft_head_topk_accept_path_descriptor_runtime,
        invalid_real_draft_head_topk_token_commit_noop_runtime,
        invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime,
        invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime,
        invalid_real_draft_head_topk_publish_gate_noop_runtime,
        invalid_real_draft_head_topk_target_logits_walk_canary_runtime,
        invalid_accept_path_descriptor,
        invalid_token_commit_descriptor,
        invalid_hidden_kv_survivor_commit_descriptor,
        invalid_rejected_branch_discard_descriptor,
        invalid_publish_gate_descriptor,
    };

    struct jetspec_target_tap_row_state {
        int32_t      batch_index = -1;
        llama_pos    pos         = -1;
        llama_seq_id seq_id      = -1;
    };

    std::vector<float> target_tap_rows;
    std::vector<jetspec_target_tap_row_state> target_tap_row_state;
    size_t n_target_tap_rows_total = 0;
    size_t n_target_tap_rows_cached = 0;
    size_t n_target_tap_process = 0;
    size_t n_runtime_state_resets = 0;
    size_t n_runtime_draft_calls = 0;
    size_t n_transaction_plan_scaffolds = 0;
    size_t n_transient_reservation_descriptors = 0;
    size_t n_tree_build_descriptors = 0;
    size_t n_root_tree_runtime_builds = 0;
    size_t n_root_verify_mask_runtime_builds = 0;
    size_t n_root_anchor_accept_path_runtime_builds = 0;
    size_t n_root_token_commit_noop_runtime_builds = 0;
    size_t n_root_hidden_kv_commit_noop_runtime_builds = 0;
    size_t n_root_rejected_branch_discard_noop_runtime_builds = 0;
    size_t n_root_publish_gate_noop_runtime_builds = 0;
    size_t n_topk_tree_runtime_builds = 0;
    size_t n_topk_verify_mask_runtime_builds = 0;
    size_t n_topk_accept_boundary_runtime_builds = 0;
    size_t n_real_draft_head_canary_decodes = 0;
    size_t n_real_draft_head_topk_candidate_runtime_builds = 0;
    size_t n_real_draft_head_topk_tree_runtime_builds = 0;
    size_t n_real_draft_head_topk_verify_mask_runtime_builds = 0;
    size_t n_real_draft_head_topk_accept_boundary_runtime_builds = 0;
    size_t n_real_draft_head_topk_accept_path_descriptor_runtime_builds = 0;
    size_t n_real_draft_head_topk_token_commit_noop_runtime_builds = 0;
    size_t n_real_draft_head_topk_hidden_kv_commit_noop_runtime_builds = 0;
    size_t n_real_draft_head_topk_rejected_branch_discard_noop_runtime_builds = 0;
    size_t n_real_draft_head_topk_publish_gate_noop_runtime_builds = 0;
    size_t n_real_draft_head_topk_target_logits_walk_canary_builds = 0;
    size_t n_verify_mask_descriptors = 0;
    size_t n_accept_path_descriptors = 0;
    size_t n_token_commit_descriptors = 0;
    size_t n_hidden_kv_survivor_commit_descriptors = 0;
    size_t n_rejected_branch_discard_descriptors = 0;
    size_t n_publish_gate_descriptors = 0;
    size_t n_pre_round_snapshots = 0;
    uint64_t target_tap_hash_last = 0;
    uint64_t pre_round_snapshot_hash_last = 0;
    uint64_t pre_round_prompt_hash_last = 0;
    uint64_t transaction_plan_hash_last = 0;
    uint64_t transient_reservation_hash_last = 0;
    uint64_t tree_build_descriptor_hash_last = 0;
    uint64_t root_tree_runtime_hash_last = 0;
    uint64_t root_verify_mask_runtime_hash_last = 0;
    uint64_t root_anchor_accept_path_runtime_hash_last = 0;
    uint64_t root_token_commit_noop_runtime_hash_last = 0;
    uint64_t root_hidden_kv_commit_noop_runtime_hash_last = 0;
    uint64_t root_rejected_branch_discard_noop_runtime_hash_last = 0;
    uint64_t root_publish_gate_noop_runtime_hash_last = 0;
    uint64_t topk_tree_runtime_hash_last = 0;
    uint64_t topk_verify_mask_runtime_hash_last = 0;
    uint64_t topk_accept_boundary_runtime_hash_last = 0;
    uint64_t real_draft_head_canary_hash_last = 0;
    uint64_t real_draft_head_topk_candidate_hash_last = 0;
    uint64_t real_draft_head_topk_tree_hash_last = 0;
    uint64_t real_draft_head_topk_verify_mask_hash_last = 0;
    uint64_t real_draft_head_topk_accept_boundary_hash_last = 0;
    uint64_t real_draft_head_topk_accept_path_descriptor_hash_last = 0;
    uint64_t real_draft_head_topk_token_commit_noop_hash_last = 0;
    uint64_t real_draft_head_topk_hidden_kv_commit_noop_hash_last = 0;
    uint64_t real_draft_head_topk_rejected_branch_discard_noop_hash_last = 0;
    uint64_t real_draft_head_topk_publish_gate_noop_hash_last = 0;
    uint64_t real_draft_head_topk_target_logits_walk_canary_hash_last = 0;
    uint64_t verify_mask_descriptor_hash_last = 0;
    uint64_t accept_path_descriptor_hash_last = 0;
    uint64_t token_commit_descriptor_hash_last = 0;
    uint64_t hidden_kv_survivor_commit_descriptor_hash_last = 0;
    uint64_t rejected_branch_discard_descriptor_hash_last = 0;
    uint64_t publish_gate_descriptor_hash_last = 0;
    int32_t target_tap_count_last = 0;
    int32_t target_tap_width_last = 0;
    int32_t pre_round_seq_id_last = -1;
    int32_t transient_reservation_seq_id_last = -1;
    int32_t tree_build_seq_id_last = -1;
    int32_t root_tree_runtime_seq_id_last = -1;
    int32_t root_verify_mask_runtime_seq_id_last = -1;
    int32_t root_anchor_accept_path_runtime_seq_id_last = -1;
    int32_t root_token_commit_noop_runtime_seq_id_last = -1;
    int32_t root_hidden_kv_commit_noop_runtime_seq_id_last = -1;
    int32_t root_rejected_branch_discard_noop_runtime_seq_id_last = -1;
    int32_t root_publish_gate_noop_runtime_seq_id_last = -1;
    int32_t topk_tree_runtime_seq_id_last = -1;
    int32_t topk_verify_mask_runtime_seq_id_last = -1;
    int32_t topk_accept_boundary_runtime_seq_id_last = -1;
    int32_t real_draft_head_topk_candidate_seq_id_last = -1;
    int32_t real_draft_head_topk_tree_seq_id_last = -1;
    int32_t real_draft_head_topk_verify_mask_seq_id_last = -1;
    int32_t real_draft_head_topk_accept_boundary_seq_id_last = -1;
    int32_t real_draft_head_topk_accept_path_descriptor_seq_id_last = -1;
    int32_t real_draft_head_topk_token_commit_noop_seq_id_last = -1;
    int32_t real_draft_head_topk_hidden_kv_commit_noop_seq_id_last = -1;
    int32_t real_draft_head_topk_rejected_branch_discard_noop_seq_id_last = -1;
    int32_t real_draft_head_topk_publish_gate_noop_seq_id_last = -1;
    int32_t real_draft_head_topk_target_logits_walk_canary_seq_id_last = -1;
    int32_t verify_mask_seq_id_last = -1;
    int32_t accept_path_seq_id_last = -1;
    int32_t token_commit_seq_id_last = -1;
    int32_t hidden_kv_survivor_commit_seq_id_last = -1;
    int32_t rejected_branch_discard_seq_id_last = -1;
    int32_t publish_gate_seq_id_last = -1;
    size_t pre_round_prompt_tokens_last = 0;
    llama_token pre_round_root_token_last = -1;
    int32_t transient_reservation_node_budget_last = 0;
    int32_t transient_reservation_actual_pages_last = 0;
    int32_t tree_build_node_budget_last = 0;
    int32_t tree_build_root_parent_last = JETSPEC_TREE_ROOT_PARENT;
    int32_t tree_build_root_depth_last = JETSPEC_TREE_ROOT_DEPTH;
    int32_t tree_build_actual_nodes_last = 0;
    int32_t topk_tree_width_last = 0;
    int32_t topk_tree_depth_last = 0;
    int32_t topk_tree_non_root_nodes_last = 0;
    int32_t topk_accept_candidate_nodes_last = 0;
    int32_t topk_accept_boundary_verified_edges_last = 0;
    int32_t topk_actual_verified_logits_rows_last = 0;
    int32_t real_draft_head_canary_ctx_present_last = 0;
    int32_t real_draft_head_canary_decode_rc_last = 0;
    int32_t real_draft_head_canary_input_rows_last = 0;
    int32_t real_draft_head_canary_input_width_last = 0;
    int32_t real_draft_head_canary_output_rows_last = 0;
    int32_t real_draft_head_canary_output_width_last = 0;
    int32_t real_draft_head_canary_logits_rows_last = 0;
    int32_t real_draft_head_canary_topk_rows_last = 0;
    int32_t real_draft_head_canary_topk_k_last = 0;
    int32_t real_draft_head_canary_top1_id_last = -1;
    int32_t real_draft_head_canary_top2_id_last = -1;
    float real_draft_head_canary_top1_logit_last = 0.0f;
    float real_draft_head_canary_top2_logit_last = 0.0f;
    int32_t real_draft_head_topk_parent_node_last = -1;
    int32_t real_draft_head_topk_candidate_nodes_last = 0;
    int32_t real_draft_head_topk_verified_logits_rows_last = 0;
    std::array<llama_token, JETSPEC_TOPK_ABI_WIDTH> real_draft_head_topk_candidate_ids = {};
    std::array<float, JETSPEC_TOPK_ABI_WIDTH> real_draft_head_topk_candidate_logits = {};
    int32_t real_draft_head_topk_tree_nodes_last = 0;
    std::array<llama_token, JETSPEC_TOPK_ABI_NODES> real_draft_head_topk_tree_token_ids = {};
    std::array<int32_t, JETSPEC_TOPK_ABI_NODES> real_draft_head_topk_tree_parent_indices = {};
    std::array<int32_t, JETSPEC_TOPK_ABI_NODES> real_draft_head_topk_tree_depth = {};
    std::array<int32_t, JETSPEC_TOPK_ABI_NODES> real_draft_head_topk_tree_rank = {};
    std::array<float, JETSPEC_TOPK_ABI_NODES> real_draft_head_topk_tree_cum_logit = {};
    int32_t real_draft_head_topk_verify_mask_entries_last = 0;
    std::array<int32_t, JETSPEC_TOPK_ABI_MASK_ENTRIES> real_draft_head_topk_verify_mask_rows = {};
    std::array<int32_t, JETSPEC_TOPK_ABI_MASK_ENTRIES> real_draft_head_topk_verify_mask_cols = {};
    std::array<uint8_t, JETSPEC_TOPK_ABI_MASK_ENTRIES> real_draft_head_topk_verify_mask_values = {};
    int32_t real_draft_head_topk_accept_candidate_nodes_last = 0;
    int32_t real_draft_head_topk_accept_boundary_verified_edges_last = 0;
    int32_t real_draft_head_topk_actual_verified_logits_rows_last = 0;
    int32_t real_draft_head_topk_accept_path_len_last = 0;
    int32_t real_draft_head_topk_actual_accepted_nodes_last = 0;
    int32_t real_draft_head_topk_correction_token_present_last = 0;
    int32_t real_draft_head_topk_accept_path_descriptor_candidate_nodes_last = 0;
    int32_t real_draft_head_topk_accept_path_descriptor_verified_edges_last = 0;
    int32_t real_draft_head_topk_accept_path_descriptor_len_last = 0;
    int32_t real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last = 0;
    int32_t real_draft_head_topk_accept_path_descriptor_correction_token_present_last = 0;
    int32_t real_draft_head_topk_target_logits_planned_rows_last = 0;
    int32_t real_draft_head_topk_target_logits_actual_rows_walked_last = 0;
    int32_t real_draft_head_topk_target_logits_batch_index_last = -1;
    int32_t real_draft_head_topk_target_logits_pos_last = -1;
    int32_t real_draft_head_topk_target_logits_seq_id_last = -1;
    int32_t real_draft_head_topk_target_logits_width_last = 0;
    int32_t real_draft_head_topk_target_logits_candidate_nodes_last = 0;
    std::array<float, JETSPEC_TOPK_ABI_WIDTH> real_draft_head_topk_target_logits_candidate_scores = {};
    std::array<llama_token, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> tree_token_ids = {};
    std::array<int32_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> tree_parent_indices = {};
    std::array<int32_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> tree_depth = {};
    std::array<int32_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> tree_rank = {};
    std::array<float, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> tree_cum_logprob = {};
    std::array<int32_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE * JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> root_verify_mask_rows = {};
    std::array<int32_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE * JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> root_verify_mask_cols = {};
    std::array<uint8_t, JETSPEC_QWEN36_DRAFT_BLOCK_SIZE * JETSPEC_QWEN36_DRAFT_BLOCK_SIZE> root_verify_mask_values = {};
    int32_t actual_verify_mask_entries_last = 0;
    int32_t root_verified_anchor_last = 0;
    int32_t accept_path_len_last = 0;
    int32_t actual_accepted_nodes_last = 0;
    int32_t correction_token_present_last = 0;
    int32_t actual_committed_tokens_last = 0;
    int32_t actual_survivor_pages_committed_last = 0;
    int32_t actual_pages_discarded_last = 0;
    int32_t rejected_branch_pages_reachable_after_discard_last = 0;
    int32_t actual_publish_visible_state_last = 0;
    int32_t root_runtime_ready_for_real_test_last = 0;
    int32_t transaction_plan_phase_count_last = 0;
    int32_t transaction_plan_rollback_count_last = 0;
    jetspec_runtime_phase runtime_phase = jetspec_runtime_phase::waiting_for_target_taps;
    jetspec_runtime_failure runtime_failure = jetspec_runtime_failure::none;
    bool target_taps_active = true;
    bool pre_round_snapshot_ready = false;
    bool transaction_plan_ready = false;
    bool transient_reservation_ready = false;
    bool tree_build_descriptor_ready = false;
    bool root_tree_runtime_ready = false;
    bool root_verify_mask_runtime_ready = false;
    bool root_anchor_accept_path_runtime_ready = false;
    bool root_token_commit_noop_runtime_ready = false;
    bool root_hidden_kv_commit_noop_runtime_ready = false;
    bool root_rejected_branch_discard_noop_runtime_ready = false;
    bool root_publish_gate_noop_runtime_ready = false;
    bool topk_tree_runtime_ready = false;
    bool topk_verify_mask_runtime_ready = false;
    bool topk_accept_boundary_runtime_ready = false;
    bool real_draft_head_canary_ready = false;
    bool real_draft_head_logits_canary_ready = false;
    bool real_draft_head_topk_candidate_runtime_ready = false;
    bool real_draft_head_topk_tree_runtime_ready = false;
    bool real_draft_head_topk_verify_mask_runtime_ready = false;
    bool real_draft_head_topk_accept_boundary_runtime_ready = false;
    bool real_draft_head_topk_accept_path_descriptor_runtime_ready = false;
    bool real_draft_head_topk_token_commit_noop_runtime_ready = false;
    bool real_draft_head_topk_hidden_kv_commit_noop_runtime_ready = false;
    bool real_draft_head_topk_rejected_branch_discard_noop_runtime_ready = false;
    bool real_draft_head_topk_publish_gate_noop_runtime_ready = false;
    bool real_draft_head_topk_target_logits_walk_canary_ready = false;
    bool verify_mask_descriptor_ready = false;
    bool accept_path_descriptor_ready = false;
    bool token_commit_descriptor_ready = false;
    bool hidden_kv_survivor_commit_descriptor_ready = false;
    bool rejected_branch_discard_descriptor_ready = false;
    bool publish_gate_descriptor_ready = false;
    bool trace_taps = false;
    bool p5x_root_tree_enabled = false;
    bool p5y_root_verify_mask_enabled = false;
    bool p5z_root_anchor_accept_path_enabled = false;
    bool p5aa_root_token_commit_noop_enabled = false;
    bool p5ab_root_hidden_kv_commit_noop_enabled = false;
    bool p5ac_root_rejected_branch_discard_noop_enabled = false;
    bool p5ad_root_publish_gate_noop_enabled = false;
    bool p5ae_topk_tree_enabled = false;
    bool p5af_topk_verify_mask_enabled = false;
    bool p5ag_topk_accept_boundary_enabled = false;
    bool p5ai_real_draft_head_canary_enabled = false;
    bool p5aj_real_draft_head_logits_canary_enabled = false;
    bool p5ak_real_draft_head_topk_candidate_enabled = false;
    bool p5al_real_draft_head_topk_tree_enabled = false;
    bool p5am_real_draft_head_topk_verify_mask_enabled = false;
    bool p5an_real_draft_head_topk_accept_boundary_enabled = false;
    bool p5ao_real_draft_head_topk_accept_path_descriptor_enabled = false;
    bool p5ap_real_draft_head_topk_token_commit_noop_enabled = false;
    bool p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled = false;
    bool p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled = false;
    bool p5as_real_draft_head_topk_publish_gate_noop_enabled = false;
    bool p5av_real_draft_head_topk_target_logits_walk_canary_enabled = false;

    static const char * jetspec_runtime_phase_name(jetspec_runtime_phase phase) {
        switch (phase) {
            case jetspec_runtime_phase::waiting_for_target_taps:        return "waiting_for_target_taps";
            case jetspec_runtime_phase::waiting_for_pre_round_snapshot:  return "waiting_for_pre_round_snapshot";
            case jetspec_runtime_phase::pre_round_snapshot_ready:        return "pre_round_snapshot_ready";
            case jetspec_runtime_phase::target_taps_captured:           return "target_taps_captured";
            case jetspec_runtime_phase::transaction_plan_scaffold_ready:             return "transaction_plan_scaffold_ready";
            case jetspec_runtime_phase::transient_reservation_descriptor_ready:      return "transient_reservation_descriptor_ready";
            case jetspec_runtime_phase::tree_build_descriptor_ready:                 return "tree_build_descriptor_ready";
            case jetspec_runtime_phase::tree_build_runtime_ready:                    return "tree_build_runtime_ready";
            case jetspec_runtime_phase::verify_mask_runtime_ready:                   return "verify_mask_runtime_ready";
            case jetspec_runtime_phase::verify_mask_descriptor_ready:                return "verify_mask_descriptor_ready";
            case jetspec_runtime_phase::real_draft_head_canary_ready:                return "real_draft_head_canary_ready";
            case jetspec_runtime_phase::real_draft_head_logits_canary_ready:         return "real_draft_head_logits_canary_ready";
            case jetspec_runtime_phase::real_draft_head_topk_candidate_ready:        return "real_draft_head_topk_candidate_ready";
            case jetspec_runtime_phase::real_draft_head_topk_tree_ready:             return "real_draft_head_topk_tree_ready";
            case jetspec_runtime_phase::real_draft_head_topk_verify_mask_ready:      return "real_draft_head_topk_verify_mask_ready";
            case jetspec_runtime_phase::real_draft_head_topk_accept_boundary_ready:  return "real_draft_head_topk_accept_boundary_ready";
            case jetspec_runtime_phase::real_draft_head_topk_accept_path_descriptor_ready: return "real_draft_head_topk_accept_path_descriptor_ready";
            case jetspec_runtime_phase::real_draft_head_topk_token_commit_noop_ready: return "real_draft_head_topk_token_commit_noop_ready";
            case jetspec_runtime_phase::real_draft_head_topk_hidden_kv_commit_noop_ready: return "real_draft_head_topk_hidden_kv_commit_noop_ready";
            case jetspec_runtime_phase::real_draft_head_topk_rejected_branch_discard_noop_ready: return "real_draft_head_topk_rejected_branch_discard_noop_ready";
            case jetspec_runtime_phase::real_draft_head_topk_publish_gate_noop_ready: return "real_draft_head_topk_publish_gate_noop_ready";
            case jetspec_runtime_phase::real_draft_head_topk_target_logits_walk_canary_ready: return "real_draft_head_topk_target_logits_walk_canary_ready";
            case jetspec_runtime_phase::accept_path_runtime_ready:                   return "accept_path_runtime_ready";
            case jetspec_runtime_phase::token_commit_runtime_ready:                  return "token_commit_runtime_ready";
            case jetspec_runtime_phase::hidden_kv_commit_runtime_ready:              return "hidden_kv_commit_runtime_ready";
            case jetspec_runtime_phase::rejected_branch_discard_runtime_ready:        return "rejected_branch_discard_runtime_ready";
            case jetspec_runtime_phase::publish_gate_runtime_ready:                   return "publish_gate_runtime_ready";
            case jetspec_runtime_phase::accept_path_descriptor_ready:                return "accept_path_descriptor_ready";
            case jetspec_runtime_phase::token_commit_descriptor_ready:               return "token_commit_descriptor_ready";
            case jetspec_runtime_phase::hidden_kv_survivor_commit_descriptor_ready:  return "hidden_kv_survivor_commit_descriptor_ready";
            case jetspec_runtime_phase::rejected_branch_discard_descriptor_ready:    return "rejected_branch_discard_descriptor_ready";
            case jetspec_runtime_phase::publish_gate_descriptor_ready:               return "publish_gate_descriptor_ready";
            case jetspec_runtime_phase::disabled:                                   return "disabled";
        }
        return "unknown";
    }

    static const char * jetspec_runtime_failure_name(jetspec_runtime_failure failure) {
        switch (failure) {
            case jetspec_runtime_failure::none:                     return "none";
            case jetspec_runtime_failure::missing_target_context:   return "missing_target_context";
            case jetspec_runtime_failure::invalid_binding:          return "invalid_binding";
            case jetspec_runtime_failure::invalid_target_taps:        return "invalid_target_taps";
            case jetspec_runtime_failure::invalid_pre_round_snapshot: return "invalid_pre_round_snapshot";
            case jetspec_runtime_failure::invalid_transaction_plan:                         return "invalid_transaction_plan";
            case jetspec_runtime_failure::invalid_transient_reservation_descriptor:          return "invalid_transient_reservation_descriptor";
            case jetspec_runtime_failure::invalid_tree_build_descriptor:                     return "invalid_tree_build_descriptor";
            case jetspec_runtime_failure::invalid_root_tree_runtime:                         return "invalid_root_tree_runtime";
            case jetspec_runtime_failure::invalid_root_verify_mask_runtime:                  return "invalid_root_verify_mask_runtime";
            case jetspec_runtime_failure::invalid_verify_mask_descriptor:                    return "invalid_verify_mask_descriptor";
            case jetspec_runtime_failure::invalid_root_anchor_accept_path_runtime:            return "invalid_root_anchor_accept_path_runtime";
            case jetspec_runtime_failure::invalid_root_token_commit_noop_runtime:             return "invalid_root_token_commit_noop_runtime";
            case jetspec_runtime_failure::invalid_root_hidden_kv_commit_noop_runtime:         return "invalid_root_hidden_kv_commit_noop_runtime";
            case jetspec_runtime_failure::invalid_root_rejected_branch_discard_noop_runtime:    return "invalid_root_rejected_branch_discard_noop_runtime";
            case jetspec_runtime_failure::invalid_root_publish_gate_noop_runtime:               return "invalid_root_publish_gate_noop_runtime";
            case jetspec_runtime_failure::invalid_topk_tree_runtime:                         return "invalid_topk_tree_runtime";
            case jetspec_runtime_failure::invalid_topk_verify_mask_runtime:                  return "invalid_topk_verify_mask_runtime";
            case jetspec_runtime_failure::invalid_topk_accept_boundary_runtime:              return "invalid_topk_accept_boundary_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_canary_runtime:            return "invalid_real_draft_head_canary_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_logits_canary_runtime:     return "invalid_real_draft_head_logits_canary_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_candidate_runtime:          return "invalid_real_draft_head_topk_candidate_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_tree_runtime:               return "invalid_real_draft_head_topk_tree_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_verify_mask_runtime:        return "invalid_real_draft_head_topk_verify_mask_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_accept_boundary_runtime:    return "invalid_real_draft_head_topk_accept_boundary_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_accept_path_descriptor_runtime: return "invalid_real_draft_head_topk_accept_path_descriptor_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_token_commit_noop_runtime: return "invalid_real_draft_head_topk_token_commit_noop_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime: return "invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime: return "invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_publish_gate_noop_runtime: return "invalid_real_draft_head_topk_publish_gate_noop_runtime";
            case jetspec_runtime_failure::invalid_real_draft_head_topk_target_logits_walk_canary_runtime: return "invalid_real_draft_head_topk_target_logits_walk_canary_runtime";
            case jetspec_runtime_failure::invalid_accept_path_descriptor:                          return "invalid_accept_path_descriptor";
            case jetspec_runtime_failure::invalid_token_commit_descriptor:                   return "invalid_token_commit_descriptor";
            case jetspec_runtime_failure::invalid_hidden_kv_survivor_commit_descriptor:      return "invalid_hidden_kv_survivor_commit_descriptor";
            case jetspec_runtime_failure::invalid_rejected_branch_discard_descriptor:        return "invalid_rejected_branch_discard_descriptor";
            case jetspec_runtime_failure::invalid_publish_gate_descriptor:                   return "invalid_publish_gate_descriptor";
        }
        return "unknown";
    }

    common_speculative_impl_draft_jetspec(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC, n_seq)
        , params(params.draft) {
        trace_taps = common_speculative_env_enabled("LLAMA_JETSPEC_TRACE") ||
                     common_speculative_env_enabled("LLAMA_JETSPEC_TAP_TRACE") ||
                     common_speculative_env_enabled("LLAMA_JETSPEC_STATE_TRACE");
        p5x_root_tree_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_TREE_BUILD_ROOT_ONLY");
        p5y_root_verify_mask_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_VERIFY_MASK_ROOT_ONLY");
        p5z_root_anchor_accept_path_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_ONLY");
        p5aa_root_token_commit_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_TOKEN_COMMIT_NOOP_ONLY");
        p5ab_root_hidden_kv_commit_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_ONLY");
        p5ac_root_rejected_branch_discard_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_ONLY");
        p5ad_root_publish_gate_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ROOT_PUBLISH_GATE_NOOP_ONLY");
        p5ae_topk_tree_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_TREE_BUILD_TOPK_ABI_ONLY");
        p5af_topk_verify_mask_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_VERIFY_MASK_TOPK_ABI_ONLY");
        p5ag_topk_accept_boundary_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_ACCEPT_PATH_TOPK_ABI_ONLY");
        p5aj_real_draft_head_logits_canary_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY");
        p5ak_real_draft_head_topk_candidate_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ABI_ONLY");
        p5al_real_draft_head_topk_tree_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_ABI_ONLY");
        p5am_real_draft_head_topk_verify_mask_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_ABI_ONLY");
        p5an_real_draft_head_topk_accept_boundary_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_ABI_ONLY");
        p5ao_real_draft_head_topk_accept_path_descriptor_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_ABI_ONLY");
        p5ap_real_draft_head_topk_token_commit_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_ABI_ONLY");
        p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_ABI_ONLY");
        p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_ABI_ONLY");
        p5as_real_draft_head_topk_publish_gate_noop_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_ABI_ONLY");
        p5av_real_draft_head_topk_target_logits_walk_canary_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY");
        p5ai_real_draft_head_canary_enabled = common_speculative_env_enabled("LLAMA_JETSPEC_REAL_DRAFT_HEAD_CANARY") || p5aj_real_draft_head_logits_canary_enabled;
        if (this->params.ctx_tgt != nullptr) {
            llama_set_jetspec_target_hidden_taps(this->params.ctx_tgt, true, true);
        } else {
            disable_runtime_state(jetspec_runtime_failure::missing_target_context, 0, 0, nullptr);
        }
        LOG_WRN("%s: draft-jetspec accepted as P5F binding-preflight plus P5N-P5W descriptor route; P5X root tree=%d P5Y root verify mask=%d P5Z root anchor accept path=%d P5AA root token commit noop=%d P5AB root hidden/KV noop=%d P5AC root discard noop=%d P5AD root publish noop=%d P5AE topk tree ABI=%d P5AF topk verify mask ABI=%d P5AG topk accept boundary ABI=%d P5AI real draft-head canary=%d P5AJ real draft-head logits canary=%d P5AK real draft-head topk ABI=%d P5AL real draft-head topk tree ABI=%d P5AM real draft-head topk verify mask ABI=%d P5AN real draft-head topk accept boundary ABI=%d P5AO real draft-head topk accept path descriptor ABI=%d P5AP real draft-head topk token commit noop ABI=%d P5AQ real draft-head topk hidden/KV commit noop ABI=%d P5AR real draft-head topk rejected-discard noop ABI=%d P5AS real draft-head topk publish-gate noop ABI=%d P5AV target-logits walk canary=%d; runtime_supported=false, no draft tokens will be generated\n", __func__, p5x_root_tree_enabled ? 1 : 0, p5y_root_verify_mask_enabled ? 1 : 0, p5z_root_anchor_accept_path_enabled ? 1 : 0, p5aa_root_token_commit_noop_enabled ? 1 : 0, p5ab_root_hidden_kv_commit_noop_enabled ? 1 : 0, p5ac_root_rejected_branch_discard_noop_enabled ? 1 : 0, p5ad_root_publish_gate_noop_enabled ? 1 : 0, p5ae_topk_tree_enabled ? 1 : 0, p5af_topk_verify_mask_enabled ? 1 : 0, p5ag_topk_accept_boundary_enabled ? 1 : 0, p5ai_real_draft_head_canary_enabled ? 1 : 0, p5aj_real_draft_head_logits_canary_enabled ? 1 : 0, p5ak_real_draft_head_topk_candidate_enabled ? 1 : 0, p5al_real_draft_head_topk_tree_enabled ? 1 : 0, p5am_real_draft_head_topk_verify_mask_enabled ? 1 : 0, p5an_real_draft_head_topk_accept_boundary_enabled ? 1 : 0, p5ao_real_draft_head_topk_accept_path_descriptor_enabled ? 1 : 0, p5ap_real_draft_head_topk_token_commit_noop_enabled ? 1 : 0, p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled ? 1 : 0, p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled ? 1 : 0, p5as_real_draft_head_topk_publish_gate_noop_enabled ? 1 : 0, p5av_real_draft_head_topk_target_logits_walk_canary_enabled ? 1 : 0);
    }

    ~common_speculative_impl_draft_jetspec() override {
        if (params.ctx_tgt != nullptr) {
            llama_set_jetspec_target_hidden_taps(params.ctx_tgt, false, true);
        }
    }

    void reset_tree_arrays() {
        for (int32_t i = 0; i < JETSPEC_QWEN36_DRAFT_BLOCK_SIZE; ++i) {
            tree_token_ids[i] = -1;
            tree_parent_indices[i] = JETSPEC_TREE_ROOT_PARENT;
            tree_depth[i] = JETSPEC_TREE_ROOT_DEPTH;
            tree_rank[i] = -1;
            tree_cum_logprob[i] = 0.0f;
        }
    }

    void reset_verify_mask_arrays() {
        for (int32_t i = 0; i < JETSPEC_QWEN36_DRAFT_BLOCK_SIZE * JETSPEC_QWEN36_DRAFT_BLOCK_SIZE; ++i) {
            root_verify_mask_rows[i] = 0;
            root_verify_mask_cols[i] = 0;
            root_verify_mask_values[i] = 0;
        }
    }

    void reset_real_draft_head_topk_tree_arrays() {
        real_draft_head_topk_tree_token_ids.fill(-1);
        real_draft_head_topk_tree_parent_indices.fill(JETSPEC_TREE_ROOT_PARENT);
        real_draft_head_topk_tree_depth.fill(JETSPEC_TREE_ROOT_DEPTH);
        real_draft_head_topk_tree_rank.fill(-1);
        real_draft_head_topk_tree_cum_logit.fill(0.0f);
    }

    void reset_real_draft_head_topk_verify_mask_arrays() {
        real_draft_head_topk_verify_mask_rows.fill(0);
        real_draft_head_topk_verify_mask_cols.fill(0);
        real_draft_head_topk_verify_mask_values.fill(0);
        reset_real_draft_head_topk_accept_boundary_metadata();
    }

    void reset_real_draft_head_topk_accept_boundary_metadata() {
        real_draft_head_topk_accept_boundary_runtime_ready = false;
        real_draft_head_topk_accept_boundary_hash_last = 0;
        real_draft_head_topk_accept_boundary_seq_id_last = -1;
        real_draft_head_topk_accept_candidate_nodes_last = 0;
        real_draft_head_topk_accept_boundary_verified_edges_last = 0;
        real_draft_head_topk_actual_verified_logits_rows_last = 0;
        real_draft_head_topk_accept_path_len_last = 0;
        real_draft_head_topk_actual_accepted_nodes_last = 0;
        real_draft_head_topk_correction_token_present_last = 0;
        reset_real_draft_head_topk_accept_path_descriptor_metadata();
    }

    void reset_real_draft_head_topk_accept_path_descriptor_metadata() {
        real_draft_head_topk_accept_path_descriptor_runtime_ready = false;
        real_draft_head_topk_accept_path_descriptor_hash_last = 0;
        real_draft_head_topk_accept_path_descriptor_seq_id_last = -1;
        real_draft_head_topk_accept_path_descriptor_candidate_nodes_last = 0;
        real_draft_head_topk_accept_path_descriptor_verified_edges_last = 0;
        real_draft_head_topk_accept_path_descriptor_len_last = 0;
        real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last = 0;
        real_draft_head_topk_accept_path_descriptor_correction_token_present_last = 0;
        reset_real_draft_head_topk_token_commit_noop_metadata();
    }

    void reset_real_draft_head_topk_token_commit_noop_metadata() {
        real_draft_head_topk_token_commit_noop_runtime_ready = false;
        real_draft_head_topk_token_commit_noop_hash_last = 0;
        real_draft_head_topk_token_commit_noop_seq_id_last = -1;
        reset_real_draft_head_topk_hidden_kv_commit_noop_metadata();
    }

    void reset_real_draft_head_topk_hidden_kv_commit_noop_metadata() {
        real_draft_head_topk_hidden_kv_commit_noop_runtime_ready = false;
        real_draft_head_topk_hidden_kv_commit_noop_hash_last = 0;
        real_draft_head_topk_hidden_kv_commit_noop_seq_id_last = -1;
        reset_real_draft_head_topk_rejected_branch_discard_noop_metadata();
    }

    void reset_real_draft_head_topk_rejected_branch_discard_noop_metadata() {
        real_draft_head_topk_rejected_branch_discard_noop_runtime_ready = false;
        real_draft_head_topk_rejected_branch_discard_noop_hash_last = 0;
        real_draft_head_topk_rejected_branch_discard_noop_seq_id_last = -1;
        reset_real_draft_head_topk_publish_gate_noop_metadata();
    }

    void reset_real_draft_head_topk_publish_gate_noop_metadata() {
        real_draft_head_topk_publish_gate_noop_runtime_ready = false;
        real_draft_head_topk_publish_gate_noop_hash_last = 0;
        real_draft_head_topk_publish_gate_noop_seq_id_last = -1;
        reset_real_draft_head_topk_target_logits_walk_canary_metadata();
    }

    void reset_real_draft_head_topk_target_logits_walk_canary_metadata() {
        real_draft_head_topk_target_logits_walk_canary_ready = false;
        real_draft_head_topk_target_logits_walk_canary_hash_last = 0;
        real_draft_head_topk_target_logits_walk_canary_seq_id_last = -1;
        real_draft_head_topk_target_logits_planned_rows_last = 0;
        real_draft_head_topk_target_logits_actual_rows_walked_last = 0;
        real_draft_head_topk_target_logits_batch_index_last = -1;
        real_draft_head_topk_target_logits_pos_last = -1;
        real_draft_head_topk_target_logits_seq_id_last = -1;
        real_draft_head_topk_target_logits_width_last = 0;
        real_draft_head_topk_target_logits_candidate_nodes_last = 0;
        real_draft_head_topk_target_logits_candidate_scores.fill(0.0f);
    }

    bool topk_abi_root_tail_conflict() const {
        return p5y_root_verify_mask_enabled || p5z_root_anchor_accept_path_enabled ||
               p5aa_root_token_commit_noop_enabled || p5ab_root_hidden_kv_commit_noop_enabled ||
               p5ac_root_rejected_branch_discard_noop_enabled || p5ad_root_publish_gate_noop_enabled;
    }

    void reset_runtime_state() {
        target_tap_hash_last = 0;
        pre_round_snapshot_hash_last = 0;
        pre_round_prompt_hash_last = 0;
        transaction_plan_hash_last = 0;
        transient_reservation_hash_last = 0;
        tree_build_descriptor_hash_last = 0;
        root_tree_runtime_hash_last = 0;
        root_verify_mask_runtime_hash_last = 0;
        root_anchor_accept_path_runtime_hash_last = 0;
        root_token_commit_noop_runtime_hash_last = 0;
        root_hidden_kv_commit_noop_runtime_hash_last = 0;
        root_rejected_branch_discard_noop_runtime_hash_last = 0;
        root_publish_gate_noop_runtime_hash_last = 0;
        topk_tree_runtime_hash_last = 0;
        topk_verify_mask_runtime_hash_last = 0;
        topk_accept_boundary_runtime_hash_last = 0;
        real_draft_head_canary_hash_last = 0;
        real_draft_head_topk_candidate_hash_last = 0;
        real_draft_head_topk_tree_hash_last = 0;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_accept_boundary_hash_last = 0;
        real_draft_head_topk_accept_path_descriptor_hash_last = 0;
        real_draft_head_topk_token_commit_noop_hash_last = 0;
        real_draft_head_topk_hidden_kv_commit_noop_hash_last = 0;
        real_draft_head_topk_rejected_branch_discard_noop_hash_last = 0;
        real_draft_head_topk_publish_gate_noop_hash_last = 0;
        real_draft_head_topk_target_logits_walk_canary_hash_last = 0;
        verify_mask_descriptor_hash_last = 0;
        accept_path_descriptor_hash_last = 0;
        token_commit_descriptor_hash_last = 0;
        hidden_kv_survivor_commit_descriptor_hash_last = 0;
        rejected_branch_discard_descriptor_hash_last = 0;
        publish_gate_descriptor_hash_last = 0;
        target_tap_count_last = 0;
        target_tap_width_last = 0;
        pre_round_seq_id_last = -1;
        transient_reservation_seq_id_last = -1;
        tree_build_seq_id_last = -1;
        root_tree_runtime_seq_id_last = -1;
        root_verify_mask_runtime_seq_id_last = -1;
        root_anchor_accept_path_runtime_seq_id_last = -1;
        root_token_commit_noop_runtime_seq_id_last = -1;
        root_hidden_kv_commit_noop_runtime_seq_id_last = -1;
        root_rejected_branch_discard_noop_runtime_seq_id_last = -1;
        root_publish_gate_noop_runtime_seq_id_last = -1;
        topk_tree_runtime_seq_id_last = -1;
        topk_verify_mask_runtime_seq_id_last = -1;
        topk_accept_boundary_runtime_seq_id_last = -1;
        real_draft_head_topk_candidate_seq_id_last = -1;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_target_logits_walk_canary_seq_id_last = -1;
        verify_mask_seq_id_last = -1;
        accept_path_seq_id_last = -1;
        token_commit_seq_id_last = -1;
        hidden_kv_survivor_commit_seq_id_last = -1;
        rejected_branch_discard_seq_id_last = -1;
        publish_gate_seq_id_last = -1;
        pre_round_prompt_tokens_last = 0;
        pre_round_root_token_last = -1;
        transient_reservation_node_budget_last = 0;
        transient_reservation_actual_pages_last = 0;
        tree_build_node_budget_last = 0;
        tree_build_root_parent_last = JETSPEC_TREE_ROOT_PARENT;
        tree_build_root_depth_last = JETSPEC_TREE_ROOT_DEPTH;
        tree_build_actual_nodes_last = 0;
        topk_tree_width_last = 0;
        topk_tree_depth_last = 0;
        topk_tree_non_root_nodes_last = 0;
        topk_accept_candidate_nodes_last = 0;
        topk_accept_boundary_verified_edges_last = 0;
        topk_actual_verified_logits_rows_last = 0;
        real_draft_head_canary_ctx_present_last = params.ctx_dft != nullptr ? 1 : 0;
        real_draft_head_canary_decode_rc_last = 0;
        real_draft_head_canary_input_rows_last = 0;
        real_draft_head_canary_input_width_last = 0;
        real_draft_head_canary_output_rows_last = 0;
        real_draft_head_canary_output_width_last = 0;
        real_draft_head_canary_logits_rows_last = 0;
        real_draft_head_canary_topk_rows_last = 0;
        real_draft_head_canary_topk_k_last = 0;
        real_draft_head_canary_top1_id_last = -1;
        real_draft_head_canary_top2_id_last = -1;
        real_draft_head_canary_top1_logit_last = 0.0f;
        real_draft_head_canary_top2_logit_last = 0.0f;
        real_draft_head_topk_parent_node_last = -1;
        real_draft_head_topk_candidate_nodes_last = 0;
        real_draft_head_topk_verified_logits_rows_last = 0;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_tree_nodes_last = 0;
        real_draft_head_topk_verify_mask_entries_last = 0;
        real_draft_head_topk_target_logits_planned_rows_last = 0;
        real_draft_head_topk_target_logits_actual_rows_walked_last = 0;
        real_draft_head_topk_target_logits_batch_index_last = -1;
        real_draft_head_topk_target_logits_pos_last = -1;
        real_draft_head_topk_target_logits_seq_id_last = -1;
        real_draft_head_topk_target_logits_width_last = 0;
        real_draft_head_topk_target_logits_candidate_nodes_last = 0;
        real_draft_head_topk_target_logits_candidate_scores.fill(0.0f);
        real_draft_head_topk_candidate_ids.fill(-1);
        real_draft_head_topk_candidate_logits.fill(0.0f);
        reset_real_draft_head_topk_tree_arrays();
        reset_real_draft_head_topk_verify_mask_arrays();
        reset_tree_arrays();
        reset_verify_mask_arrays();
        actual_verify_mask_entries_last = 0;
        root_verified_anchor_last = 0;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;
        root_runtime_ready_for_real_test_last = 0;
        transaction_plan_phase_count_last = 0;
        transaction_plan_rollback_count_last = 0;
        pre_round_snapshot_ready = false;
        transaction_plan_ready = false;
        transient_reservation_ready = false;
        tree_build_descriptor_ready = false;
        root_tree_runtime_ready = false;
        root_verify_mask_runtime_ready = false;
        root_anchor_accept_path_runtime_ready = false;
        root_token_commit_noop_runtime_ready = false;
        root_hidden_kv_commit_noop_runtime_ready = false;
        root_rejected_branch_discard_noop_runtime_ready = false;
        root_publish_gate_noop_runtime_ready = false;
        topk_tree_runtime_ready = false;
        topk_verify_mask_runtime_ready = false;
        topk_accept_boundary_runtime_ready = false;
        real_draft_head_canary_ready = false;
        real_draft_head_logits_canary_ready = false;
        real_draft_head_topk_candidate_runtime_ready = false;
        real_draft_head_topk_tree_runtime_ready = false;
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_target_logits_walk_canary_ready = false;
        verify_mask_descriptor_ready = false;
        accept_path_descriptor_ready = false;
        token_commit_descriptor_ready = false;
        hidden_kv_survivor_commit_descriptor_ready = false;
        rejected_branch_discard_descriptor_ready = false;
        publish_gate_descriptor_ready = false;
        n_target_tap_rows_cached = 0;
        runtime_phase = target_taps_active ? jetspec_runtime_phase::waiting_for_target_taps : jetspec_runtime_phase::disabled;
        runtime_failure = target_taps_active ? jetspec_runtime_failure::none : runtime_failure;
        target_tap_rows.clear();
        target_tap_row_state.clear();
        n_runtime_state_resets++;
    }

    void disable_runtime_state(jetspec_runtime_failure failure, int32_t tap_count, int32_t tap_width, const float * taps) {
        runtime_phase = jetspec_runtime_phase::disabled;
        runtime_failure = failure;
        target_tap_count_last = tap_count;
        target_tap_width_last = tap_width;
        target_taps_active = false;
        n_target_tap_rows_cached = 0;
        pre_round_snapshot_hash_last = 0;
        pre_round_prompt_hash_last = 0;
        transaction_plan_hash_last = 0;
        transient_reservation_hash_last = 0;
        tree_build_descriptor_hash_last = 0;
        root_tree_runtime_hash_last = 0;
        root_verify_mask_runtime_hash_last = 0;
        root_anchor_accept_path_runtime_hash_last = 0;
        root_token_commit_noop_runtime_hash_last = 0;
        root_hidden_kv_commit_noop_runtime_hash_last = 0;
        root_rejected_branch_discard_noop_runtime_hash_last = 0;
        root_publish_gate_noop_runtime_hash_last = 0;
        topk_tree_runtime_hash_last = 0;
        topk_verify_mask_runtime_hash_last = 0;
        topk_accept_boundary_runtime_hash_last = 0;
        real_draft_head_canary_hash_last = 0;
        real_draft_head_topk_candidate_hash_last = 0;
        real_draft_head_topk_tree_hash_last = 0;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_accept_boundary_hash_last = 0;
        real_draft_head_topk_accept_path_descriptor_hash_last = 0;
        real_draft_head_topk_token_commit_noop_hash_last = 0;
        real_draft_head_topk_hidden_kv_commit_noop_hash_last = 0;
        real_draft_head_topk_rejected_branch_discard_noop_hash_last = 0;
        real_draft_head_topk_publish_gate_noop_hash_last = 0;
        real_draft_head_topk_target_logits_walk_canary_hash_last = 0;
        verify_mask_descriptor_hash_last = 0;
        accept_path_descriptor_hash_last = 0;
        token_commit_descriptor_hash_last = 0;
        hidden_kv_survivor_commit_descriptor_hash_last = 0;
        rejected_branch_discard_descriptor_hash_last = 0;
        publish_gate_descriptor_hash_last = 0;
        pre_round_seq_id_last = -1;
        transient_reservation_seq_id_last = -1;
        tree_build_seq_id_last = -1;
        root_tree_runtime_seq_id_last = -1;
        root_verify_mask_runtime_seq_id_last = -1;
        root_anchor_accept_path_runtime_seq_id_last = -1;
        root_token_commit_noop_runtime_seq_id_last = -1;
        root_hidden_kv_commit_noop_runtime_seq_id_last = -1;
        root_rejected_branch_discard_noop_runtime_seq_id_last = -1;
        root_publish_gate_noop_runtime_seq_id_last = -1;
        topk_tree_runtime_seq_id_last = -1;
        topk_verify_mask_runtime_seq_id_last = -1;
        topk_accept_boundary_runtime_seq_id_last = -1;
        real_draft_head_topk_candidate_seq_id_last = -1;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_target_logits_walk_canary_seq_id_last = -1;
        verify_mask_seq_id_last = -1;
        accept_path_seq_id_last = -1;
        token_commit_seq_id_last = -1;
        hidden_kv_survivor_commit_seq_id_last = -1;
        rejected_branch_discard_seq_id_last = -1;
        publish_gate_seq_id_last = -1;
        pre_round_prompt_tokens_last = 0;
        pre_round_root_token_last = -1;
        transient_reservation_node_budget_last = 0;
        transient_reservation_actual_pages_last = 0;
        tree_build_node_budget_last = 0;
        tree_build_root_parent_last = JETSPEC_TREE_ROOT_PARENT;
        tree_build_root_depth_last = JETSPEC_TREE_ROOT_DEPTH;
        tree_build_actual_nodes_last = 0;
        topk_tree_width_last = 0;
        topk_tree_depth_last = 0;
        topk_tree_non_root_nodes_last = 0;
        topk_accept_candidate_nodes_last = 0;
        topk_accept_boundary_verified_edges_last = 0;
        topk_actual_verified_logits_rows_last = 0;
        real_draft_head_canary_ctx_present_last = params.ctx_dft != nullptr ? 1 : 0;
        real_draft_head_canary_decode_rc_last = 0;
        real_draft_head_canary_input_rows_last = 0;
        real_draft_head_canary_input_width_last = 0;
        real_draft_head_canary_output_rows_last = 0;
        real_draft_head_canary_output_width_last = 0;
        real_draft_head_canary_logits_rows_last = 0;
        real_draft_head_canary_topk_rows_last = 0;
        real_draft_head_canary_topk_k_last = 0;
        real_draft_head_canary_top1_id_last = -1;
        real_draft_head_canary_top2_id_last = -1;
        real_draft_head_canary_top1_logit_last = 0.0f;
        real_draft_head_canary_top2_logit_last = 0.0f;
        real_draft_head_topk_parent_node_last = -1;
        real_draft_head_topk_candidate_nodes_last = 0;
        real_draft_head_topk_verified_logits_rows_last = 0;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_tree_nodes_last = 0;
        real_draft_head_topk_verify_mask_entries_last = 0;
        real_draft_head_topk_target_logits_planned_rows_last = 0;
        real_draft_head_topk_target_logits_actual_rows_walked_last = 0;
        real_draft_head_topk_target_logits_batch_index_last = -1;
        real_draft_head_topk_target_logits_pos_last = -1;
        real_draft_head_topk_target_logits_seq_id_last = -1;
        real_draft_head_topk_target_logits_width_last = 0;
        real_draft_head_topk_target_logits_candidate_nodes_last = 0;
        real_draft_head_topk_target_logits_candidate_scores.fill(0.0f);
        real_draft_head_topk_candidate_ids.fill(-1);
        real_draft_head_topk_candidate_logits.fill(0.0f);
        reset_real_draft_head_topk_tree_arrays();
        reset_real_draft_head_topk_verify_mask_arrays();
        reset_tree_arrays();
        reset_verify_mask_arrays();
        actual_verify_mask_entries_last = 0;
        root_verified_anchor_last = 0;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;
        root_runtime_ready_for_real_test_last = 0;
        transaction_plan_phase_count_last = 0;
        transaction_plan_rollback_count_last = 0;
        pre_round_snapshot_ready = false;
        transaction_plan_ready = false;
        transient_reservation_ready = false;
        tree_build_descriptor_ready = false;
        root_tree_runtime_ready = false;
        root_verify_mask_runtime_ready = false;
        root_anchor_accept_path_runtime_ready = false;
        root_token_commit_noop_runtime_ready = false;
        root_hidden_kv_commit_noop_runtime_ready = false;
        root_rejected_branch_discard_noop_runtime_ready = false;
        root_publish_gate_noop_runtime_ready = false;
        topk_tree_runtime_ready = false;
        topk_verify_mask_runtime_ready = false;
        topk_accept_boundary_runtime_ready = false;
        real_draft_head_canary_ready = false;
        real_draft_head_logits_canary_ready = false;
        real_draft_head_topk_candidate_runtime_ready = false;
        real_draft_head_topk_tree_runtime_ready = false;
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_target_logits_walk_canary_ready = false;
        verify_mask_descriptor_ready = false;
        accept_path_descriptor_ready = false;
        token_commit_descriptor_ready = false;
        hidden_kv_survivor_commit_descriptor_ready = false;
        rejected_branch_discard_descriptor_ready = false;
        publish_gate_descriptor_ready = false;
        target_tap_rows.clear();
        target_tap_row_state.clear();
        if (params.ctx_tgt != nullptr) {
            llama_set_jetspec_target_hidden_taps(params.ctx_tgt, false, true);
        }
        LOG_WRN("%s: draft-jetspec runtime state disabled reason=%s taps=%d width=%d ptr=%p\n",
                __func__, jetspec_runtime_failure_name(failure), tap_count, tap_width, (const void *) taps);
    }

    int32_t capture_target_tap_row_state(const llama_batch & batch) {
        target_tap_row_state.clear();
        if (batch.logits == nullptr) {
            return 0;
        }
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                continue;
            }
            jetspec_target_tap_row_state row;
            row.batch_index = i;
            row.pos = batch.pos != nullptr ? batch.pos[i] : -1;
            if (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr) {
                row.seq_id = batch.seq_id[i][0];
            }
            target_tap_row_state.push_back(row);
        }
        return (int32_t) target_tap_row_state.size();
    }

    bool run_real_draft_head_canary_decode(int32_t tap_width) {
        real_draft_head_canary_ready = false;
        real_draft_head_logits_canary_ready = false;
        real_draft_head_canary_hash_last = 0;
        real_draft_head_canary_ctx_present_last = params.ctx_dft != nullptr ? 1 : 0;
        real_draft_head_canary_decode_rc_last = 0;
        real_draft_head_canary_input_rows_last = 0;
        real_draft_head_canary_input_width_last = 0;
        real_draft_head_canary_output_rows_last = 0;
        real_draft_head_canary_output_width_last = 0;
        real_draft_head_canary_logits_rows_last = 0;
        real_draft_head_canary_topk_rows_last = 0;
        real_draft_head_canary_topk_k_last = 0;
        real_draft_head_canary_top1_id_last = -1;
        real_draft_head_canary_top2_id_last = -1;
        real_draft_head_canary_top1_logit_last = 0.0f;
        real_draft_head_canary_top2_logit_last = 0.0f;
        real_draft_head_topk_candidate_runtime_ready = false;
        real_draft_head_topk_candidate_hash_last = 0;
        real_draft_head_topk_tree_runtime_ready = false;
        real_draft_head_topk_tree_hash_last = 0;
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_candidate_seq_id_last = -1;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_parent_node_last = -1;
        real_draft_head_topk_candidate_nodes_last = 0;
        real_draft_head_topk_verified_logits_rows_last = 0;
        real_draft_head_topk_tree_nodes_last = 0;
        real_draft_head_topk_verify_mask_entries_last = 0;
        real_draft_head_topk_candidate_ids.fill(-1);
        real_draft_head_topk_candidate_logits.fill(0.0f);
        reset_real_draft_head_topk_tree_arrays();
        reset_real_draft_head_topk_verify_mask_arrays();

        if (!p5ai_real_draft_head_canary_enabled) {
            return true;
        }
        if (params.ctx_dft == nullptr || n_target_tap_rows_cached == 0 || target_tap_rows.empty()) {
            return false;
        }
        if (tap_width != JETSPEC_QWEN36_TARGET_TAP_WIDTH || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        const int32_t n_rows = (int32_t) n_target_tap_rows_cached;
        const int32_t output_width = p5aj_real_draft_head_logits_canary_enabled ? JETSPEC_QWEN36_VOCAB_SIZE : JETSPEC_QWEN36_TARGET_HIDDEN;
        llama_batch batch = llama_batch_init(n_rows, tap_width, 1);
        batch.n_tokens = n_rows;
        for (int32_t i = 0; i < n_rows; ++i) {
            std::memcpy(batch.embd + (size_t) i * (size_t) tap_width,
                    target_tap_rows.data() + (size_t) i * (size_t) tap_width,
                    (size_t) tap_width * sizeof(float));
            batch.pos[i] = target_tap_row_state[i].pos >= 0 ? target_tap_row_state[i].pos : i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = target_tap_row_state[i].seq_id >= 0 ? target_tap_row_state[i].seq_id : 0;
            batch.logits[i] = 1;
        }

        const int32_t rc = common_speculative_jetspec_real_draft_head_canary_eval(params.ctx_dft, batch);
        real_draft_head_canary_decode_rc_last = rc;
        if (rc != 0) {
            llama_batch_free(batch);
            return false;
        }

        uint64_t hash = common_speculative_fnv1a64(
                p5aj_real_draft_head_logits_canary_enabled ? JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY_RUNTIME_PHASE : JETSPEC_REAL_DRAFT_HEAD_CANARY_RUNTIME_PHASE,
                std::strlen(p5aj_real_draft_head_logits_canary_enabled ? JETSPEC_REAL_DRAFT_HEAD_LOGITS_CANARY_RUNTIME_PHASE : JETSPEC_REAL_DRAFT_HEAD_CANARY_RUNTIME_PHASE));
        for (int32_t i = 0; i < n_rows; ++i) {
            const float * embd = llama_get_embeddings_ith(params.ctx_dft, i);
            if (embd == nullptr) {
                llama_batch_free(batch);
                return false;
            }
            hash ^= common_speculative_fnv1a64(embd, (size_t) output_width * sizeof(float));
            if (p5aj_real_draft_head_logits_canary_enabled && i == 0) {
                int32_t top1 = -1;
                int32_t top2 = -1;
                float top1_v = -std::numeric_limits<float>::infinity();
                float top2_v = -std::numeric_limits<float>::infinity();
                for (int32_t j = 0; j < output_width; ++j) {
                    const float v = embd[j];
                    if (v > top1_v) {
                        top2_v = top1_v;
                        top2 = top1;
                        top1_v = v;
                        top1 = j;
                    } else if (v > top2_v) {
                        top2_v = v;
                        top2 = j;
                    }
                }
                real_draft_head_canary_top1_id_last = top1;
                real_draft_head_canary_top2_id_last = top2;
                real_draft_head_canary_top1_logit_last = top1_v;
                real_draft_head_canary_top2_logit_last = top2_v;
            }
        }

        llama_batch_free(batch);

        real_draft_head_canary_hash_last = hash;
        real_draft_head_canary_input_rows_last = n_rows;
        real_draft_head_canary_input_width_last = tap_width;
        real_draft_head_canary_output_rows_last = n_rows;
        real_draft_head_canary_output_width_last = output_width;
        real_draft_head_canary_logits_rows_last = p5aj_real_draft_head_logits_canary_enabled ? n_rows : 0;
        real_draft_head_canary_topk_rows_last = p5aj_real_draft_head_logits_canary_enabled ? n_rows : 0;
        real_draft_head_canary_topk_k_last = p5aj_real_draft_head_logits_canary_enabled ? 2 : 0;
        real_draft_head_canary_ready = true;
        real_draft_head_logits_canary_ready = p5aj_real_draft_head_logits_canary_enabled;
        n_real_draft_head_canary_decodes++;
        runtime_phase = p5aj_real_draft_head_logits_canary_enabled ? jetspec_runtime_phase::real_draft_head_logits_canary_ready : jetspec_runtime_phase::real_draft_head_canary_ready;
        return true;
    }

    bool build_real_draft_head_topk_candidate_runtime() {
        real_draft_head_topk_candidate_runtime_ready = false;
        real_draft_head_topk_candidate_hash_last = 0;
        real_draft_head_topk_tree_runtime_ready = false;
        real_draft_head_topk_tree_hash_last = 0;
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_candidate_seq_id_last = -1;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_parent_node_last = -1;
        real_draft_head_topk_candidate_nodes_last = 0;
        real_draft_head_topk_verified_logits_rows_last = 0;
        real_draft_head_topk_verify_mask_entries_last = 0;
        real_draft_head_topk_candidate_ids.fill(-1);
        real_draft_head_topk_candidate_logits.fill(0.0f);
        reset_real_draft_head_topk_verify_mask_arrays();

        if (!p5ak_real_draft_head_topk_candidate_enabled) {
            return false;
        }
        if (!p5aj_real_draft_head_logits_canary_enabled || !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0) {
            return false;
        }
        if (!p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (!topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0) {
            return false;
        }
        if (real_draft_head_canary_logits_rows_last <= 0 || real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_canary_topk_rows_last != real_draft_head_canary_logits_rows_last || real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_canary_top1_id_last < 0 || real_draft_head_canary_top2_id_last < 0 ||
                real_draft_head_canary_top1_id_last >= JETSPEC_QWEN36_VOCAB_SIZE || real_draft_head_canary_top2_id_last >= JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_canary_top1_id_last == real_draft_head_canary_top2_id_last) {
            return false;
        }
        if (!std::isfinite(real_draft_head_canary_top1_logit_last) || !std::isfinite(real_draft_head_canary_top2_logit_last) ||
                real_draft_head_canary_top1_logit_last < real_draft_head_canary_top2_logit_last) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES || topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0) {
            return false;
        }

        real_draft_head_topk_candidate_seq_id_last = topk_accept_boundary_runtime_seq_id_last;
        real_draft_head_topk_parent_node_last = 0;
        real_draft_head_topk_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES;
        real_draft_head_topk_verified_logits_rows_last = real_draft_head_canary_logits_rows_last;
        real_draft_head_topk_candidate_ids[0] = real_draft_head_canary_top1_id_last;
        real_draft_head_topk_candidate_ids[1] = real_draft_head_canary_top2_id_last;
        real_draft_head_topk_candidate_logits[0] = real_draft_head_canary_top1_logit_last;
        real_draft_head_topk_candidate_logits[1] = real_draft_head_canary_top2_logit_last;

        const int64_t candidate_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_seq_id_last,
            (int64_t) real_draft_head_topk_parent_node_last,
            (int64_t) real_draft_head_topk_candidate_nodes_last,
            (int64_t) real_draft_head_topk_verified_logits_rows_last,
            (int64_t) real_draft_head_canary_output_width_last,
            (int64_t) real_draft_head_canary_topk_k_last,
            (int64_t) real_draft_head_topk_candidate_ids[0],
            (int64_t) real_draft_head_topk_candidate_ids[1],
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_candidate_hash_last = common_speculative_fnv1a64(candidate_words, sizeof(candidate_words));
        real_draft_head_topk_candidate_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_candidate_logits.data(), real_draft_head_topk_candidate_logits.size() * sizeof(real_draft_head_topk_candidate_logits[0]));
        real_draft_head_topk_candidate_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_CANDIDATE_RUNTIME_PHASE));
        real_draft_head_topk_candidate_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE));
        real_draft_head_topk_candidate_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS));
        real_draft_head_topk_candidate_runtime_ready = true;
        n_real_draft_head_topk_candidate_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_candidate_ready;
        return true;
    }

    bool build_real_draft_head_topk_tree_runtime() {
        real_draft_head_topk_tree_runtime_ready = false;
        real_draft_head_topk_tree_hash_last = 0;
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_tree_seq_id_last = -1;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_tree_nodes_last = 0;
        real_draft_head_topk_verify_mask_entries_last = 0;
        reset_real_draft_head_topk_tree_arrays();
        reset_real_draft_head_topk_verify_mask_arrays();

        if (!p5al_real_draft_head_topk_tree_enabled) {
            return false;
        }
        if (!p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (!real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0) {
            return false;
        }
        if (!real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0) {
            return false;
        }
        if (real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_canary_topk_rows_last != 1 || real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH ||
                real_draft_head_topk_verified_logits_rows_last != 1) {
            return false;
        }
        if (real_draft_head_topk_parent_node_last != 0 || real_draft_head_topk_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES) {
            return false;
        }
        if (real_draft_head_topk_candidate_ids[0] < 0 || real_draft_head_topk_candidate_ids[1] < 0 ||
                real_draft_head_topk_candidate_ids[0] >= JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_candidate_ids[1] >= JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_candidate_ids[0] == real_draft_head_topk_candidate_ids[1]) {
            return false;
        }
        if (!std::isfinite(real_draft_head_topk_candidate_logits[0]) || !std::isfinite(real_draft_head_topk_candidate_logits[1]) ||
                real_draft_head_topk_candidate_logits[0] < real_draft_head_topk_candidate_logits[1]) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || topk_tree_width_last != JETSPEC_TOPK_ABI_WIDTH ||
                topk_tree_depth_last != JETSPEC_TOPK_ABI_DEPTH || topk_tree_non_root_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES) {
            return false;
        }
        if (pre_round_root_token_last < 0 || tree_token_ids[0] != pre_round_root_token_last ||
                tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_parent_indices[1] != 0 || tree_parent_indices[2] != 0 ||
                tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH || tree_depth[1] != 1 || tree_depth[2] != 1 ||
                tree_rank[0] != -1 || tree_rank[1] != 0 || tree_rank[2] != 1) {
            return false;
        }
        if (actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0 ||
                actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_tree_seq_id_last = real_draft_head_topk_candidate_seq_id_last;
        real_draft_head_topk_tree_nodes_last = JETSPEC_TOPK_ABI_NODES;
        real_draft_head_topk_tree_token_ids[0] = tree_token_ids[0];
        real_draft_head_topk_tree_token_ids[1] = real_draft_head_topk_candidate_ids[0];
        real_draft_head_topk_tree_token_ids[2] = real_draft_head_topk_candidate_ids[1];
        real_draft_head_topk_tree_parent_indices[0] = JETSPEC_TREE_ROOT_PARENT;
        real_draft_head_topk_tree_parent_indices[1] = 0;
        real_draft_head_topk_tree_parent_indices[2] = 0;
        real_draft_head_topk_tree_depth[0] = JETSPEC_TREE_ROOT_DEPTH;
        real_draft_head_topk_tree_depth[1] = 1;
        real_draft_head_topk_tree_depth[2] = 1;
        real_draft_head_topk_tree_rank[0] = -1;
        real_draft_head_topk_tree_rank[1] = 0;
        real_draft_head_topk_tree_rank[2] = 1;
        real_draft_head_topk_tree_cum_logit[0] = 0.0f;
        real_draft_head_topk_tree_cum_logit[1] = real_draft_head_topk_candidate_logits[0];
        real_draft_head_topk_tree_cum_logit[2] = real_draft_head_topk_candidate_logits[1];
        if (real_draft_head_topk_tree_token_ids[0] != tree_token_ids[0] ||
                real_draft_head_topk_tree_token_ids[1] != real_draft_head_topk_candidate_ids[0] ||
                real_draft_head_topk_tree_token_ids[2] != real_draft_head_topk_candidate_ids[1]) {
            return false;
        }
        if (real_draft_head_topk_tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT ||
                real_draft_head_topk_tree_parent_indices[1] != 0 || real_draft_head_topk_tree_parent_indices[2] != 0 ||
                real_draft_head_topk_tree_depth[0] != 0 || real_draft_head_topk_tree_depth[1] != 1 || real_draft_head_topk_tree_depth[2] != 1 ||
                real_draft_head_topk_tree_rank[0] != -1 || real_draft_head_topk_tree_rank[1] != 0 || real_draft_head_topk_tree_rank[2] != 1) {
            return false;
        }
        if (real_draft_head_topk_tree_token_ids[1] < 0 || real_draft_head_topk_tree_token_ids[2] < 0 ||
                real_draft_head_topk_tree_token_ids[1] == real_draft_head_topk_tree_token_ids[2]) {
            return false;
        }

        const int64_t real_tree_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) topk_tree_width_last,
            (int64_t) topk_tree_depth_last,
            (int64_t) topk_tree_non_root_nodes_last,
            (int64_t) real_draft_head_topk_tree_token_ids[0],
            (int64_t) real_draft_head_topk_tree_token_ids[1],
            (int64_t) real_draft_head_topk_tree_token_ids[2],
            (int64_t) real_draft_head_topk_tree_parent_indices[0],
            (int64_t) real_draft_head_topk_tree_parent_indices[1],
            (int64_t) real_draft_head_topk_tree_parent_indices[2],
            (int64_t) real_draft_head_topk_tree_depth[0],
            (int64_t) real_draft_head_topk_tree_depth[1],
            (int64_t) real_draft_head_topk_tree_depth[2],
            (int64_t) real_draft_head_topk_tree_rank[0],
            (int64_t) real_draft_head_topk_tree_rank[1],
            (int64_t) real_draft_head_topk_tree_rank[2],
            (int64_t) real_draft_head_topk_verified_logits_rows_last,
            (int64_t) real_draft_head_canary_output_width_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_tree_hash_last = common_speculative_fnv1a64(real_tree_words, sizeof(real_tree_words));
        real_draft_head_topk_tree_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_tree_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_TREE_RUNTIME_PHASE));
        real_draft_head_topk_tree_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE));
        real_draft_head_topk_tree_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS));
        real_draft_head_topk_tree_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_PHASE, std::strlen(JETSPEC_TREE_BUILD_PHASE));
        real_draft_head_topk_tree_runtime_ready = true;
        n_real_draft_head_topk_tree_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_tree_ready;
        return true;
    }

    bool build_real_draft_head_topk_verify_mask_runtime() {
        real_draft_head_topk_verify_mask_runtime_ready = false;
        real_draft_head_topk_verify_mask_hash_last = 0;
        real_draft_head_topk_verify_mask_seq_id_last = -1;
        real_draft_head_topk_verify_mask_entries_last = 0;
        reset_real_draft_head_topk_verify_mask_arrays();

        if (!p5am_real_draft_head_topk_verify_mask_enabled) {
            return false;
        }
        if (!p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled ||
                !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled ||
                !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (!real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_verified_logits_rows_last != 1 || real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_tree_token_ids[0] != tree_token_ids[0] ||
                real_draft_head_topk_tree_token_ids[1] != real_draft_head_topk_candidate_ids[0] ||
                real_draft_head_topk_tree_token_ids[2] != real_draft_head_topk_candidate_ids[1] ||
                real_draft_head_topk_tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT ||
                real_draft_head_topk_tree_parent_indices[1] != 0 || real_draft_head_topk_tree_parent_indices[2] != 0 ||
                real_draft_head_topk_tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH ||
                real_draft_head_topk_tree_depth[1] != 1 || real_draft_head_topk_tree_depth[2] != 1 ||
                real_draft_head_topk_tree_rank[0] != -1 || real_draft_head_topk_tree_rank[1] != 0 || real_draft_head_topk_tree_rank[2] != 1) {
            return false;
        }
        if (real_draft_head_topk_tree_cum_logit[0] != 0.0f ||
                real_draft_head_topk_tree_cum_logit[1] != real_draft_head_topk_candidate_logits[0] ||
                real_draft_head_topk_tree_cum_logit[2] != real_draft_head_topk_candidate_logits[1]) {
            return false;
        }
        if (actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                root_verify_mask_rows[0] != 0 || root_verify_mask_cols[0] != 0 || root_verify_mask_values[0] != 1 ||
                root_verify_mask_rows[1] != 1 || root_verify_mask_cols[1] != 0 || root_verify_mask_values[1] != 1 ||
                root_verify_mask_rows[2] != 1 || root_verify_mask_cols[2] != 1 || root_verify_mask_values[2] != 1 ||
                root_verify_mask_rows[3] != 2 || root_verify_mask_cols[3] != 0 || root_verify_mask_values[3] != 1 ||
                root_verify_mask_rows[4] != 2 || root_verify_mask_cols[4] != 2 || root_verify_mask_values[4] != 1) {
            return false;
        }
        if (accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0 ||
                actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_verify_mask_seq_id_last = real_draft_head_topk_tree_seq_id_last;
        real_draft_head_topk_verify_mask_entries_last = JETSPEC_TOPK_ABI_MASK_ENTRIES;
        real_draft_head_topk_verify_mask_rows[0] = 0;
        real_draft_head_topk_verify_mask_cols[0] = 0;
        real_draft_head_topk_verify_mask_values[0] = 1;
        real_draft_head_topk_verify_mask_rows[1] = 1;
        real_draft_head_topk_verify_mask_cols[1] = 0;
        real_draft_head_topk_verify_mask_values[1] = 1;
        real_draft_head_topk_verify_mask_rows[2] = 1;
        real_draft_head_topk_verify_mask_cols[2] = 1;
        real_draft_head_topk_verify_mask_values[2] = 1;
        real_draft_head_topk_verify_mask_rows[3] = 2;
        real_draft_head_topk_verify_mask_cols[3] = 0;
        real_draft_head_topk_verify_mask_values[3] = 1;
        real_draft_head_topk_verify_mask_rows[4] = 2;
        real_draft_head_topk_verify_mask_cols[4] = 2;
        real_draft_head_topk_verify_mask_values[4] = 1;
        for (int32_t i = 0; i < real_draft_head_topk_verify_mask_entries_last; ++i) {
            if (real_draft_head_topk_verify_mask_rows[i] < 0 || real_draft_head_topk_verify_mask_rows[i] >= real_draft_head_topk_tree_nodes_last ||
                    real_draft_head_topk_verify_mask_cols[i] < 0 || real_draft_head_topk_verify_mask_cols[i] >= real_draft_head_topk_tree_nodes_last ||
                    real_draft_head_topk_verify_mask_values[i] != 1) {
                return false;
            }
        }
        if (real_draft_head_topk_verify_mask_rows[1] != 1 || real_draft_head_topk_verify_mask_cols[1] != real_draft_head_topk_tree_parent_indices[1] ||
                real_draft_head_topk_verify_mask_rows[3] != 2 || real_draft_head_topk_verify_mask_cols[3] != real_draft_head_topk_tree_parent_indices[2]) {
            return false;
        }

        const int64_t real_mask_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_tree_token_ids[0],
            (int64_t) real_draft_head_topk_tree_token_ids[1],
            (int64_t) real_draft_head_topk_tree_token_ids[2],
            (int64_t) real_draft_head_topk_tree_parent_indices[0],
            (int64_t) real_draft_head_topk_tree_parent_indices[1],
            (int64_t) real_draft_head_topk_tree_parent_indices[2],
            (int64_t) real_draft_head_topk_tree_depth[0],
            (int64_t) real_draft_head_topk_tree_depth[1],
            (int64_t) real_draft_head_topk_tree_depth[2],
            (int64_t) real_draft_head_topk_tree_rank[0],
            (int64_t) real_draft_head_topk_tree_rank[1],
            (int64_t) real_draft_head_topk_tree_rank[2],
            (int64_t) real_draft_head_topk_verified_logits_rows_last,
            (int64_t) real_draft_head_canary_output_width_last,
            (int64_t) real_draft_head_canary_topk_k_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_verify_mask_hash_last = common_speculative_fnv1a64(real_mask_words, sizeof(real_mask_words));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_VERIFY_MASK_RUNTIME_PHASE));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS));
        real_draft_head_topk_verify_mask_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_PHASE, std::strlen(JETSPEC_VERIFY_MASK_PHASE));
        real_draft_head_topk_verify_mask_runtime_ready = true;
        n_real_draft_head_topk_verify_mask_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_verify_mask_ready;
        return true;
    }

    bool build_real_draft_head_topk_accept_boundary_runtime() {
        reset_real_draft_head_topk_accept_boundary_metadata();

        if (!p5an_real_draft_head_topk_accept_boundary_enabled) {
            return false;
        }
        if (!p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled ||
                !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled ||
                !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (!real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_verified_logits_rows_last != 1 || real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_tree_token_ids[0] != tree_token_ids[0] ||
                real_draft_head_topk_tree_token_ids[1] != real_draft_head_topk_candidate_ids[0] ||
                real_draft_head_topk_tree_token_ids[2] != real_draft_head_topk_candidate_ids[1] ||
                real_draft_head_topk_tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT ||
                real_draft_head_topk_tree_parent_indices[1] != 0 || real_draft_head_topk_tree_parent_indices[2] != 0 ||
                real_draft_head_topk_tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH ||
                real_draft_head_topk_tree_depth[1] != 1 || real_draft_head_topk_tree_depth[2] != 1 ||
                real_draft_head_topk_tree_rank[0] != -1 || real_draft_head_topk_tree_rank[1] != 0 ||
                real_draft_head_topk_tree_rank[2] != 1) {
            return false;
        }
        if (real_draft_head_topk_verify_mask_rows[0] != 0 || real_draft_head_topk_verify_mask_cols[0] != 0 || real_draft_head_topk_verify_mask_values[0] != 1 ||
                real_draft_head_topk_verify_mask_rows[1] != 1 || real_draft_head_topk_verify_mask_cols[1] != 0 || real_draft_head_topk_verify_mask_values[1] != 1 ||
                real_draft_head_topk_verify_mask_rows[2] != 1 || real_draft_head_topk_verify_mask_cols[2] != 1 || real_draft_head_topk_verify_mask_values[2] != 1 ||
                real_draft_head_topk_verify_mask_rows[3] != 2 || real_draft_head_topk_verify_mask_cols[3] != 0 || real_draft_head_topk_verify_mask_values[3] != 1 ||
                real_draft_head_topk_verify_mask_rows[4] != 2 || real_draft_head_topk_verify_mask_cols[4] != 2 || real_draft_head_topk_verify_mask_values[4] != 1) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_actual_verified_logits_rows_last != 0 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 ||
                correction_token_present_last != 0 || actual_committed_tokens_last != 0 ||
                actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_accept_boundary_seq_id_last = real_draft_head_topk_verify_mask_seq_id_last;
        real_draft_head_topk_accept_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES;
        real_draft_head_topk_accept_boundary_verified_edges_last = JETSPEC_TOPK_ABI_MASK_ENTRIES;
        real_draft_head_topk_actual_verified_logits_rows_last = real_draft_head_topk_verified_logits_rows_last;
        real_draft_head_topk_accept_path_len_last = 0;
        real_draft_head_topk_actual_accepted_nodes_last = 0;
        real_draft_head_topk_correction_token_present_last = 0;
        if (real_draft_head_topk_accept_candidate_nodes_last != 2 || real_draft_head_topk_accept_boundary_verified_edges_last != 5 ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 || real_draft_head_topk_accept_path_len_last != 0 ||
                real_draft_head_topk_actual_accepted_nodes_last != 0 || real_draft_head_topk_correction_token_present_last != 0) {
            return false;
        }

        const int64_t real_accept_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_boundary_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_len_last,
            (int64_t) real_draft_head_topk_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_correction_token_present_last,
            (int64_t) real_draft_head_topk_tree_token_ids[0],
            (int64_t) real_draft_head_topk_tree_token_ids[1],
            (int64_t) real_draft_head_topk_tree_token_ids[2],
            (int64_t) real_draft_head_topk_tree_parent_indices[0],
            (int64_t) real_draft_head_topk_tree_parent_indices[1],
            (int64_t) real_draft_head_topk_tree_parent_indices[2],
            (int64_t) real_draft_head_topk_verified_logits_rows_last,
            (int64_t) real_draft_head_canary_output_width_last,
            (int64_t) real_draft_head_canary_topk_k_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_accept_boundary_hash_last = common_speculative_fnv1a64(real_accept_words, sizeof(real_accept_words));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_PHASE, std::strlen(JETSPEC_ACCEPT_PATH_PHASE));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS));
        real_draft_head_topk_accept_boundary_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_accept_boundary_runtime_ready = true;
        n_real_draft_head_topk_accept_boundary_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_accept_boundary_ready;
        return true;
    }

    bool build_real_draft_head_topk_accept_path_descriptor_runtime() {
        reset_real_draft_head_topk_accept_path_descriptor_metadata();

        if (!p5ao_real_draft_head_topk_accept_path_descriptor_enabled) {
            return false;
        }
        if (!p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled ||
                !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled ||
                !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled ||
                !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 ||
                real_draft_head_topk_verified_logits_rows_last != 1 ||
                real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_accept_boundary_seq_id_last != real_draft_head_topk_verify_mask_seq_id_last ||
                real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_len_last != 0 || real_draft_head_topk_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_correction_token_present_last != 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_actual_verified_logits_rows_last != 0 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 ||
                correction_token_present_last != 0 || actual_committed_tokens_last != 0 ||
                actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_accept_path_descriptor_seq_id_last = real_draft_head_topk_accept_boundary_seq_id_last;
        real_draft_head_topk_accept_path_descriptor_candidate_nodes_last = real_draft_head_topk_accept_candidate_nodes_last;
        real_draft_head_topk_accept_path_descriptor_verified_edges_last = real_draft_head_topk_accept_boundary_verified_edges_last;
        real_draft_head_topk_accept_path_descriptor_len_last = real_draft_head_topk_accept_path_len_last;
        real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last = real_draft_head_topk_actual_accepted_nodes_last;
        real_draft_head_topk_accept_path_descriptor_correction_token_present_last = real_draft_head_topk_correction_token_present_last;
        if (real_draft_head_topk_accept_path_descriptor_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_path_descriptor_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }

        const int64_t real_accept_descriptor_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_hash_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_accept_path_descriptor_hash_last = common_speculative_fnv1a64(real_accept_descriptor_words, sizeof(real_accept_descriptor_words));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_ACCEPT_PATH_DESCRIPTOR_RUNTIME_PHASE));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_PHASE, std::strlen(JETSPEC_ACCEPT_PATH_PHASE));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_ROLLBACK_POINT, std::strlen(JETSPEC_ACCEPT_PATH_ROLLBACK_POINT));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_DESCRIPTOR, std::strlen(JETSPEC_ACCEPT_PATH_DESCRIPTOR));
        real_draft_head_topk_accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_accept_path_descriptor_runtime_ready = true;
        n_real_draft_head_topk_accept_path_descriptor_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_accept_path_descriptor_ready;
        return true;
    }

    bool build_real_draft_head_topk_token_commit_noop_runtime() {
        reset_real_draft_head_topk_token_commit_noop_metadata();
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        actual_publish_visible_state_last = 0;

        if (!p5ap_real_draft_head_topk_token_commit_noop_enabled) {
            return false;
        }
        if (!p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled ||
                !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled ||
                !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled ||
                !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_accept_path_descriptor_runtime_ready || real_draft_head_topk_accept_path_descriptor_hash_last == 0 ||
                !real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 ||
                real_draft_head_topk_verified_logits_rows_last != 1 ||
                real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_accept_path_descriptor_seq_id_last != real_draft_head_topk_accept_boundary_seq_id_last ||
                real_draft_head_topk_accept_boundary_seq_id_last != real_draft_head_topk_verify_mask_seq_id_last ||
                real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_accept_path_descriptor_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_path_descriptor_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_accept_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                topk_accept_boundary_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                topk_actual_verified_logits_rows_last != 0 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 ||
                correction_token_present_last != 0 || actual_committed_tokens_last != 0 ||
                actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_token_commit_noop_seq_id_last = real_draft_head_topk_accept_path_descriptor_seq_id_last;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        actual_publish_visible_state_last = 0;
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        const int64_t real_commit_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_hash_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_hash_last,
            (int64_t) real_draft_head_topk_token_commit_noop_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_token_commit_noop_hash_last = common_speculative_fnv1a64(real_commit_words, sizeof(real_commit_words));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_TOKEN_COMMIT_NOOP_RUNTIME_PHASE));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_PHASE, std::strlen(JETSPEC_TOKEN_COMMIT_PHASE));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT));
        real_draft_head_topk_token_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_token_commit_noop_runtime_ready = true;
        n_real_draft_head_topk_token_commit_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_token_commit_noop_ready;
        return true;
    }

    bool build_real_draft_head_topk_hidden_kv_commit_noop_runtime() {
        reset_real_draft_head_topk_hidden_kv_commit_noop_metadata();
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        actual_publish_visible_state_last = 0;

        if (!p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled) {
            return false;
        }
        if (!p5ap_real_draft_head_topk_token_commit_noop_enabled || !p5ao_real_draft_head_topk_accept_path_descriptor_enabled ||
                !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled ||
                !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled ||
                !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled ||
                !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_token_commit_noop_runtime_ready || real_draft_head_topk_token_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_accept_path_descriptor_runtime_ready || real_draft_head_topk_accept_path_descriptor_hash_last == 0 ||
                !real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 ||
                real_draft_head_topk_verified_logits_rows_last != 1 ||
                real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_token_commit_noop_seq_id_last != real_draft_head_topk_accept_path_descriptor_seq_id_last ||
                real_draft_head_topk_accept_path_descriptor_seq_id_last != real_draft_head_topk_accept_boundary_seq_id_last ||
                real_draft_head_topk_accept_boundary_seq_id_last != real_draft_head_topk_verify_mask_seq_id_last ||
                real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_accept_path_descriptor_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_path_descriptor_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_hidden_kv_commit_noop_seq_id_last = real_draft_head_topk_token_commit_noop_seq_id_last;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        actual_publish_visible_state_last = 0;
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        const int64_t real_hidden_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_hash_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_hash_last,
            (int64_t) real_draft_head_topk_token_commit_noop_hash_last,
            (int64_t) real_draft_head_topk_hidden_kv_commit_noop_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_hidden_kv_commit_noop_hash_last = common_speculative_fnv1a64(real_hidden_words, sizeof(real_hidden_words));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT));
        real_draft_head_topk_hidden_kv_commit_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_hidden_kv_commit_noop_runtime_ready = true;
        n_real_draft_head_topk_hidden_kv_commit_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_hidden_kv_commit_noop_ready;
        return true;
    }

    bool build_real_draft_head_topk_rejected_branch_discard_noop_runtime() {
        reset_real_draft_head_topk_rejected_branch_discard_noop_metadata();
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;

        if (!p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled) {
            return false;
        }
        if (!p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled ||
                !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled ||
                !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled ||
                !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled ||
                !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_hidden_kv_commit_noop_runtime_ready || real_draft_head_topk_hidden_kv_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_token_commit_noop_runtime_ready || real_draft_head_topk_token_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_accept_path_descriptor_runtime_ready || real_draft_head_topk_accept_path_descriptor_hash_last == 0 ||
                !real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 ||
                real_draft_head_topk_verified_logits_rows_last != 1 ||
                real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_hidden_kv_commit_noop_seq_id_last != real_draft_head_topk_token_commit_noop_seq_id_last ||
                real_draft_head_topk_token_commit_noop_seq_id_last != real_draft_head_topk_accept_path_descriptor_seq_id_last ||
                real_draft_head_topk_accept_path_descriptor_seq_id_last != real_draft_head_topk_accept_boundary_seq_id_last ||
                real_draft_head_topk_accept_boundary_seq_id_last != real_draft_head_topk_verify_mask_seq_id_last ||
                real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_accept_path_descriptor_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_path_descriptor_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_rejected_branch_discard_noop_seq_id_last = real_draft_head_topk_hidden_kv_commit_noop_seq_id_last;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        const int64_t real_discard_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_hash_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_hash_last,
            (int64_t) real_draft_head_topk_token_commit_noop_hash_last,
            (int64_t) real_draft_head_topk_hidden_kv_commit_noop_hash_last,
            (int64_t) real_draft_head_topk_rejected_branch_discard_noop_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_rejected_branch_discard_noop_hash_last = common_speculative_fnv1a64(real_discard_words, sizeof(real_discard_words));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT));
        real_draft_head_topk_rejected_branch_discard_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_rejected_branch_discard_noop_runtime_ready = true;
        n_real_draft_head_topk_rejected_branch_discard_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_rejected_branch_discard_noop_ready;
        return true;
    }

    bool build_real_draft_head_topk_publish_gate_noop_runtime() {
        reset_real_draft_head_topk_publish_gate_noop_metadata();
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;

        if (!p5as_real_draft_head_topk_publish_gate_noop_enabled) {
            return false;
        }
        if (!p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled ||
                !p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled ||
                !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled ||
                !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled ||
                !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled ||
                !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_rejected_branch_discard_noop_runtime_ready || real_draft_head_topk_rejected_branch_discard_noop_hash_last == 0 ||
                !real_draft_head_topk_hidden_kv_commit_noop_runtime_ready || real_draft_head_topk_hidden_kv_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_token_commit_noop_runtime_ready || real_draft_head_topk_token_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_accept_path_descriptor_runtime_ready || real_draft_head_topk_accept_path_descriptor_hash_last == 0 ||
                !real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0 ||
                !real_draft_head_logits_canary_ready || real_draft_head_canary_hash_last == 0 ||
                !topk_accept_boundary_runtime_ready || topk_accept_boundary_runtime_hash_last == 0 ||
                !topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0 ||
                !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0 ||
                !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (real_draft_head_canary_ctx_present_last != 1 || real_draft_head_canary_decode_rc_last != 0 ||
                real_draft_head_canary_logits_rows_last != 1 || real_draft_head_canary_output_rows_last != 1 ||
                real_draft_head_canary_output_width_last != JETSPEC_QWEN36_VOCAB_SIZE ||
                real_draft_head_topk_actual_verified_logits_rows_last != 1 ||
                real_draft_head_topk_verified_logits_rows_last != 1 ||
                real_draft_head_canary_topk_k_last != JETSPEC_TOPK_ABI_WIDTH) {
            return false;
        }
        if (real_draft_head_topk_rejected_branch_discard_noop_seq_id_last != real_draft_head_topk_hidden_kv_commit_noop_seq_id_last ||
                real_draft_head_topk_hidden_kv_commit_noop_seq_id_last != real_draft_head_topk_token_commit_noop_seq_id_last ||
                real_draft_head_topk_token_commit_noop_seq_id_last != real_draft_head_topk_accept_path_descriptor_seq_id_last ||
                real_draft_head_topk_accept_path_descriptor_seq_id_last != real_draft_head_topk_accept_boundary_seq_id_last ||
                real_draft_head_topk_accept_boundary_seq_id_last != real_draft_head_topk_verify_mask_seq_id_last ||
                real_draft_head_topk_tree_seq_id_last != real_draft_head_topk_candidate_seq_id_last ||
                real_draft_head_topk_verify_mask_seq_id_last != real_draft_head_topk_tree_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (real_draft_head_topk_accept_path_descriptor_candidate_nodes_last != JETSPEC_TOPK_ABI_NON_ROOT_NODES ||
                real_draft_head_topk_accept_path_descriptor_verified_edges_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        real_draft_head_topk_publish_gate_noop_seq_id_last = real_draft_head_topk_rejected_branch_discard_noop_seq_id_last;
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }

        const int64_t real_publish_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_hash_last,
            (int64_t) real_draft_head_canary_hash_last,
            (int64_t) real_draft_head_topk_candidate_hash_last,
            (int64_t) real_draft_head_topk_tree_hash_last,
            (int64_t) real_draft_head_topk_verify_mask_hash_last,
            (int64_t) real_draft_head_topk_accept_boundary_hash_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_hash_last,
            (int64_t) real_draft_head_topk_token_commit_noop_hash_last,
            (int64_t) real_draft_head_topk_hidden_kv_commit_noop_hash_last,
            (int64_t) real_draft_head_topk_rejected_branch_discard_noop_hash_last,
            (int64_t) real_draft_head_topk_publish_gate_noop_seq_id_last,
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_verified_edges_last,
            (int64_t) real_draft_head_topk_actual_verified_logits_rows_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        real_draft_head_topk_publish_gate_noop_hash_last = common_speculative_fnv1a64(real_publish_words, sizeof(real_publish_words));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_tree_cum_logit.data(), real_draft_head_topk_tree_cum_logit.size() * sizeof(real_draft_head_topk_tree_cum_logit[0]));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_rows.data(), real_draft_head_topk_verify_mask_rows.size() * sizeof(real_draft_head_topk_verify_mask_rows[0]));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_cols.data(), real_draft_head_topk_verify_mask_cols.size() * sizeof(real_draft_head_topk_verify_mask_cols[0]));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_verify_mask_values.data(), real_draft_head_topk_verify_mask_values.size() * sizeof(real_draft_head_topk_verify_mask_values[0]));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_PUBLISH_GATE_NOOP_RUNTIME_PHASE));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_PUBLISH_GATE_PHASE, std::strlen(JETSPEC_PUBLISH_GATE_PHASE));
        real_draft_head_topk_publish_gate_noop_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS));
        real_draft_head_topk_publish_gate_noop_runtime_ready = true;
        n_real_draft_head_topk_publish_gate_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_publish_gate_noop_ready;
        return true;
    }

    bool build_real_draft_head_topk_target_logits_walk_canary_runtime(const llama_batch & batch) {
        reset_real_draft_head_topk_target_logits_walk_canary_metadata();
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;

        if (!p5av_real_draft_head_topk_target_logits_walk_canary_enabled) {
            return false;
        }
        if (!p5as_real_draft_head_topk_publish_gate_noop_enabled || !p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled ||
                !p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled ||
                !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled ||
                !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled ||
                !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled ||
                !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled ||
                !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (accept_path_descriptor_ready || token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready ||
                rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready ||
                root_token_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_ready ||
                root_rejected_branch_discard_noop_runtime_ready || root_publish_gate_noop_runtime_ready) {
            return false;
        }
        if (!real_draft_head_topk_publish_gate_noop_runtime_ready || real_draft_head_topk_publish_gate_noop_hash_last == 0 ||
                !real_draft_head_topk_rejected_branch_discard_noop_runtime_ready || real_draft_head_topk_rejected_branch_discard_noop_hash_last == 0 ||
                !real_draft_head_topk_hidden_kv_commit_noop_runtime_ready || real_draft_head_topk_hidden_kv_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_token_commit_noop_runtime_ready || real_draft_head_topk_token_commit_noop_hash_last == 0 ||
                !real_draft_head_topk_accept_path_descriptor_runtime_ready || real_draft_head_topk_accept_path_descriptor_hash_last == 0 ||
                !real_draft_head_topk_accept_boundary_runtime_ready || real_draft_head_topk_accept_boundary_hash_last == 0 ||
                !real_draft_head_topk_verify_mask_runtime_ready || real_draft_head_topk_verify_mask_hash_last == 0 ||
                !real_draft_head_topk_tree_runtime_ready || real_draft_head_topk_tree_hash_last == 0 ||
                !real_draft_head_topk_candidate_runtime_ready || real_draft_head_topk_candidate_hash_last == 0) {
            return false;
        }
        if (real_draft_head_topk_publish_gate_noop_seq_id_last != real_draft_head_topk_rejected_branch_discard_noop_seq_id_last ||
                real_draft_head_topk_tree_nodes_last != JETSPEC_TOPK_ABI_NODES ||
                real_draft_head_topk_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES ||
                real_draft_head_topk_accept_path_descriptor_len_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last != 0 ||
                real_draft_head_topk_accept_path_descriptor_correction_token_present_last != 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 ||
                actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 ||
                actual_publish_visible_state_last != 0) {
            return false;
        }
        for (int32_t i = 0; i < JETSPEC_TOPK_ABI_WIDTH; ++i) {
            if (real_draft_head_topk_candidate_ids[i] < 0 || real_draft_head_topk_candidate_ids[i] >= JETSPEC_QWEN36_VOCAB_SIZE) {
                return false;
            }
        }

        int32_t batch_index = -1;
        if (batch.logits != nullptr) {
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (batch.logits[i] != 0) {
                    batch_index = i;
                    break;
                }
            }
        }
        if (batch_index < 0 || params.ctx_tgt == nullptr) {
            return false;
        }
        const float * target_logits = llama_get_logits_ith(params.ctx_tgt, batch_index);
        if (target_logits == nullptr) {
            return false;
        }

        real_draft_head_topk_target_logits_walk_canary_seq_id_last = real_draft_head_topk_publish_gate_noop_seq_id_last;
        real_draft_head_topk_target_logits_planned_rows_last = 1;
        real_draft_head_topk_target_logits_actual_rows_walked_last = 1;
        real_draft_head_topk_target_logits_batch_index_last = batch_index;
        real_draft_head_topk_target_logits_pos_last = batch.pos != nullptr ? (int32_t) batch.pos[batch_index] : -1;
        real_draft_head_topk_target_logits_seq_id_last = -1;
        if (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[batch_index] > 0 && batch.seq_id[batch_index] != nullptr) {
            real_draft_head_topk_target_logits_seq_id_last = (int32_t) batch.seq_id[batch_index][0];
        }
        real_draft_head_topk_target_logits_width_last = JETSPEC_QWEN36_VOCAB_SIZE;
        real_draft_head_topk_target_logits_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES;
        for (int32_t i = 0; i < JETSPEC_TOPK_ABI_WIDTH; ++i) {
            real_draft_head_topk_target_logits_candidate_scores[i] = target_logits[real_draft_head_topk_candidate_ids[i]];
        }
        actual_committed_tokens_last = 0;
        actual_survivor_pages_committed_last = 0;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        actual_publish_visible_state_last = 0;

        const int64_t target_walk_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) real_draft_head_topk_publish_gate_noop_hash_last,
            (int64_t) real_draft_head_topk_target_logits_walk_canary_seq_id_last,
            (int64_t) real_draft_head_topk_target_logits_planned_rows_last,
            (int64_t) real_draft_head_topk_target_logits_actual_rows_walked_last,
            (int64_t) real_draft_head_topk_target_logits_batch_index_last,
            (int64_t) real_draft_head_topk_target_logits_pos_last,
            (int64_t) real_draft_head_topk_target_logits_seq_id_last,
            (int64_t) real_draft_head_topk_target_logits_width_last,
            (int64_t) real_draft_head_topk_target_logits_candidate_nodes_last,
            (int64_t) real_draft_head_topk_candidate_ids[0],
            (int64_t) real_draft_head_topk_candidate_ids[1],
            (int64_t) real_draft_head_topk_tree_nodes_last,
            (int64_t) real_draft_head_topk_verify_mask_entries_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_len_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
            (int64_t) real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) actual_publish_visible_state_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
        };
        real_draft_head_topk_target_logits_walk_canary_hash_last = common_speculative_fnv1a64(target_walk_words, sizeof(target_walk_words));
        real_draft_head_topk_target_logits_walk_canary_hash_last ^= common_speculative_fnv1a64(real_draft_head_topk_target_logits_candidate_scores.data(), real_draft_head_topk_target_logits_candidate_scores.size() * sizeof(real_draft_head_topk_target_logits_candidate_scores[0]));
        real_draft_head_topk_target_logits_walk_canary_hash_last ^= common_speculative_fnv1a64(JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY_RUNTIME_PHASE, std::strlen(JETSPEC_REAL_DRAFT_HEAD_TOPK_TARGET_LOGITS_WALK_CANARY_RUNTIME_PHASE));
        real_draft_head_topk_target_logits_walk_canary_hash_last ^= common_speculative_fnv1a64(JETSPEC_TARGET_LOGITS_SOURCE, std::strlen(JETSPEC_TARGET_LOGITS_SOURCE));
        real_draft_head_topk_target_logits_walk_canary_hash_last ^= common_speculative_fnv1a64(JETSPEC_TARGET_LOGITS_WALK_ROW_SEMANTICS, std::strlen(JETSPEC_TARGET_LOGITS_WALK_ROW_SEMANTICS));
        real_draft_head_topk_target_logits_walk_canary_ready = true;
        n_real_draft_head_topk_target_logits_walk_canary_builds++;
        runtime_phase = jetspec_runtime_phase::real_draft_head_topk_target_logits_walk_canary_ready;
        return true;
    }

    bool build_pre_round_snapshot(llama_seq_id seq_id, const llama_tokens & prompt) {
        pre_round_snapshot_ready = false;
        pre_round_snapshot_hash_last = 0;
        pre_round_prompt_hash_last = 0;
        pre_round_seq_id_last = -1;
        pre_round_prompt_tokens_last = 0;
        pre_round_root_token_last = -1;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq) {
            return false;
        }
        if (p5x_root_tree_enabled && prompt.empty()) {
            return false;
        }

        pre_round_seq_id_last = (int32_t) seq_id;
        pre_round_prompt_tokens_last = prompt.size();
        pre_round_root_token_last = prompt.empty() ? -1 : prompt.back();
        pre_round_prompt_hash_last = common_speculative_fnv1a64(prompt.data(), prompt.size() * sizeof(prompt[0]));

        const int64_t snapshot_words[] = {
            (int64_t) pre_round_seq_id_last,
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) pre_round_root_token_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
        };
        pre_round_snapshot_hash_last = common_speculative_fnv1a64(snapshot_words, sizeof(snapshot_words));
        pre_round_snapshot_hash_last ^= common_speculative_fnv1a64(JETSPEC_PRE_ROUND_SNAPSHOT_PHASE, std::strlen(JETSPEC_PRE_ROUND_SNAPSHOT_PHASE));
        pre_round_snapshot_ready = true;
        n_pre_round_snapshots++;
        runtime_phase = jetspec_runtime_phase::pre_round_snapshot_ready;
        return true;
    }

    bool build_transaction_plan_scaffold() {
        transaction_plan_ready = false;
        transaction_plan_hash_last = 0;
        transaction_plan_phase_count_last = 0;
        transaction_plan_rollback_count_last = 0;
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        std::vector<int64_t> plan_words;
        plan_words.reserve(8 + target_tap_row_state.size() * 3);
        plan_words.push_back((int64_t) pre_round_snapshot_hash_last);
        plan_words.push_back((int64_t) pre_round_seq_id_last);
        plan_words.push_back((int64_t) pre_round_prompt_tokens_last);
        plan_words.push_back((int64_t) pre_round_prompt_hash_last);
        plan_words.push_back(JETSPEC_TRANSACTION_PHASE_COUNT);
        plan_words.push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT);
        plan_words.push_back((int64_t) n_target_tap_rows_cached);
        plan_words.push_back((int64_t) target_tap_hash_last);
        for (const auto & row : target_tap_row_state) {
            plan_words.push_back((int64_t) row.batch_index);
            plan_words.push_back((int64_t) row.pos);
            plan_words.push_back((int64_t) row.seq_id);
        }

        transaction_plan_hash_last = common_speculative_fnv1a64(plan_words.data(), plan_words.size() * sizeof(plan_words[0]));
        transaction_plan_hash_last ^= common_speculative_fnv1a64(JETSPEC_TRANSACTION_PHASE_ORDER, std::strlen(JETSPEC_TRANSACTION_PHASE_ORDER));
        transaction_plan_hash_last ^= common_speculative_fnv1a64(JETSPEC_TRANSACTION_ROLLBACK_POINTS, std::strlen(JETSPEC_TRANSACTION_ROLLBACK_POINTS));
        transaction_plan_phase_count_last = JETSPEC_TRANSACTION_PHASE_COUNT;
        transaction_plan_rollback_count_last = JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT;
        transaction_plan_ready = true;
        n_transaction_plan_scaffolds++;
        runtime_phase = jetspec_runtime_phase::transaction_plan_scaffold_ready;
        return true;
    }

    bool build_transient_reservation_descriptor() {
        transient_reservation_ready = false;
        transient_reservation_hash_last = 0;
        transient_reservation_seq_id_last = -1;
        transient_reservation_node_budget_last = 0;
        transient_reservation_actual_pages_last = 0;
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        transient_reservation_seq_id_last = pre_round_seq_id_last;
        transient_reservation_node_budget_last = (int32_t) target_tap_row_state.size();
        if (transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE) {
            return false;
        }
        transient_reservation_actual_pages_last = 0;

        std::vector<int64_t> reservation_words;
        reservation_words.reserve(11 + target_tap_row_state.size() * 3);
        reservation_words.push_back((int64_t) transaction_plan_hash_last);
        reservation_words.push_back((int64_t) pre_round_snapshot_hash_last);
        reservation_words.push_back((int64_t) pre_round_seq_id_last);
        reservation_words.push_back((int64_t) pre_round_prompt_tokens_last);
        reservation_words.push_back((int64_t) transient_reservation_seq_id_last);
        reservation_words.push_back((int64_t) transient_reservation_node_budget_last);
        reservation_words.push_back((int64_t) transient_reservation_actual_pages_last);
        reservation_words.push_back((int64_t) n_target_tap_rows_cached);
        reservation_words.push_back((int64_t) target_tap_hash_last);
        reservation_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE);
        reservation_words.push_back(JETSPEC_TRANSACTION_PHASE_COUNT);
        for (const auto & row : target_tap_row_state) {
            reservation_words.push_back((int64_t) row.batch_index);
            reservation_words.push_back((int64_t) row.pos);
            reservation_words.push_back((int64_t) row.seq_id);
        }

        transient_reservation_hash_last = common_speculative_fnv1a64(reservation_words.data(), reservation_words.size() * sizeof(reservation_words[0]));
        transient_reservation_hash_last ^= common_speculative_fnv1a64(JETSPEC_TRANSIENT_RESERVATION_PHASE, std::strlen(JETSPEC_TRANSIENT_RESERVATION_PHASE));
        transient_reservation_hash_last ^= common_speculative_fnv1a64(JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT, std::strlen(JETSPEC_TRANSIENT_RESERVATION_ROLLBACK_POINT));
        transient_reservation_hash_last ^= common_speculative_fnv1a64(JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR, std::strlen(JETSPEC_TRANSIENT_RESERVATION_DESCRIPTOR));
        transient_reservation_ready = true;
        n_transient_reservation_descriptors++;
        runtime_phase = jetspec_runtime_phase::transient_reservation_descriptor_ready;
        return true;
    }

    bool build_tree_build_descriptor() {
        tree_build_descriptor_ready = false;
        tree_build_descriptor_hash_last = 0;
        tree_build_seq_id_last = -1;
        tree_build_node_budget_last = 0;
        tree_build_root_parent_last = JETSPEC_TREE_ROOT_PARENT;
        tree_build_root_depth_last = JETSPEC_TREE_ROOT_DEPTH;
        tree_build_actual_nodes_last = 0;
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (transient_reservation_actual_pages_last != 0) {
            return false;
        }
        if (transient_reservation_node_budget_last <= 0 || transient_reservation_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        tree_build_seq_id_last = pre_round_seq_id_last;
        tree_build_node_budget_last = transient_reservation_node_budget_last;
        if (p5ae_topk_tree_enabled) {
            tree_build_node_budget_last = std::max(tree_build_node_budget_last, JETSPEC_TOPK_ABI_NODES);
        }
        tree_build_root_parent_last = JETSPEC_TREE_ROOT_PARENT;
        tree_build_root_depth_last = JETSPEC_TREE_ROOT_DEPTH;
        tree_build_actual_nodes_last = 0;

        std::vector<int64_t> tree_words;
        tree_words.reserve(15 + target_tap_row_state.size() * 3);
        tree_words.push_back((int64_t) transaction_plan_hash_last);
        tree_words.push_back((int64_t) pre_round_snapshot_hash_last);
        tree_words.push_back((int64_t) transient_reservation_hash_last);
        tree_words.push_back((int64_t) pre_round_prompt_tokens_last);
        tree_words.push_back((int64_t) pre_round_prompt_hash_last);
        tree_words.push_back((int64_t) tree_build_seq_id_last);
        tree_words.push_back((int64_t) tree_build_node_budget_last);
        tree_words.push_back((int64_t) tree_build_root_parent_last);
        tree_words.push_back((int64_t) tree_build_root_depth_last);
        tree_words.push_back((int64_t) tree_build_actual_nodes_last);
        tree_words.push_back((int64_t) n_target_tap_rows_cached);
        tree_words.push_back((int64_t) target_tap_hash_last);
        tree_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE);
        tree_words.push_back(JETSPEC_TRANSACTION_PHASE_COUNT);
        tree_words.push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT);
        for (const auto & row : target_tap_row_state) {
            tree_words.push_back((int64_t) row.batch_index);
            tree_words.push_back((int64_t) row.pos);
            tree_words.push_back((int64_t) row.seq_id);
        }

        tree_build_descriptor_hash_last = common_speculative_fnv1a64(tree_words.data(), tree_words.size() * sizeof(tree_words[0]));
        tree_build_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_PHASE, std::strlen(JETSPEC_TREE_BUILD_PHASE));
        tree_build_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_ROLLBACK_POINT, std::strlen(JETSPEC_TREE_BUILD_ROLLBACK_POINT));
        tree_build_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_DESCRIPTOR, std::strlen(JETSPEC_TREE_BUILD_DESCRIPTOR));
        tree_build_descriptor_ready = true;
        n_tree_build_descriptors++;
        runtime_phase = jetspec_runtime_phase::tree_build_descriptor_ready;
        return true;
    }

    bool build_root_only_runtime_tree() {
        root_tree_runtime_ready = false;
        root_tree_runtime_hash_last = 0;
        root_tree_runtime_seq_id_last = -1;
        tree_build_actual_nodes_last = 0;
        topk_tree_width_last = 0;
        topk_tree_depth_last = 0;
        topk_tree_non_root_nodes_last = 0;
        reset_tree_arrays();
        if (!p5x_root_tree_enabled) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (transient_reservation_actual_pages_last != 0) {
            return false;
        }
        if (tree_build_node_budget_last <= 0 || tree_build_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE) {
            return false;
        }
        if (pre_round_root_token_last < 0) {
            return false;
        }

        root_tree_runtime_seq_id_last = pre_round_seq_id_last;
        tree_build_actual_nodes_last = 1;
        tree_token_ids[0] = pre_round_root_token_last;
        tree_parent_indices[0] = JETSPEC_TREE_ROOT_PARENT;
        tree_depth[0] = JETSPEC_TREE_ROOT_DEPTH;
        tree_rank[0] = -1;
        tree_cum_logprob[0] = 0.0f;
        if (tree_build_actual_nodes_last > tree_build_node_budget_last) {
            return false;
        }
        if (tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH) {
            return false;
        }

        const int64_t root_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) tree_token_ids[0],
            (int64_t) tree_parent_indices[0],
            (int64_t) tree_depth[0],
            (int64_t) tree_rank[0],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_tree_runtime_hash_last = common_speculative_fnv1a64(root_words, sizeof(root_words));
        root_tree_runtime_hash_last ^= common_speculative_fnv1a64(&tree_cum_logprob[0], sizeof(tree_cum_logprob[0]));
        root_tree_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_TREE_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_TREE_RUNTIME_PHASE));
        root_tree_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_PHASE, std::strlen(JETSPEC_TREE_BUILD_PHASE));
        root_tree_runtime_ready = true;
        n_root_tree_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::tree_build_runtime_ready;
        return true;
    }

    bool build_topk_tree_runtime() {
        topk_tree_runtime_ready = false;
        topk_tree_runtime_hash_last = 0;
        topk_tree_runtime_seq_id_last = -1;
        topk_tree_width_last = 0;
        topk_tree_depth_last = 0;
        topk_tree_non_root_nodes_last = 0;
        if (!p5ae_topk_tree_enabled) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (pre_round_root_token_last < 0 || tree_token_ids[0] != pre_round_root_token_last) {
            return false;
        }
        if (tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH || tree_rank[0] != -1) {
            return false;
        }
        if (tree_build_node_budget_last < JETSPEC_TOPK_ABI_NODES || tree_build_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        topk_tree_runtime_seq_id_last = root_tree_runtime_seq_id_last;
        topk_tree_width_last = JETSPEC_TOPK_ABI_WIDTH;
        topk_tree_depth_last = JETSPEC_TOPK_ABI_DEPTH;
        topk_tree_non_root_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES;
        tree_build_actual_nodes_last = JETSPEC_TOPK_ABI_NODES;
        tree_token_ids[0] = pre_round_root_token_last;
        tree_token_ids[1] = (pre_round_root_token_last + 1) % JETSPEC_QWEN36_VOCAB_SIZE;
        tree_token_ids[2] = (pre_round_root_token_last + 2) % JETSPEC_QWEN36_VOCAB_SIZE;
        tree_parent_indices[0] = JETSPEC_TREE_ROOT_PARENT;
        tree_parent_indices[1] = 0;
        tree_parent_indices[2] = 0;
        tree_depth[0] = JETSPEC_TREE_ROOT_DEPTH;
        tree_depth[1] = 1;
        tree_depth[2] = 1;
        tree_rank[0] = -1;
        tree_rank[1] = 0;
        tree_rank[2] = 1;
        tree_cum_logprob[0] = 0.0f;
        tree_cum_logprob[1] = -0.1f;
        tree_cum_logprob[2] = -0.3f;
        if (tree_build_actual_nodes_last > tree_build_node_budget_last) {
            return false;
        }
        if (tree_parent_indices[1] >= 1 || tree_parent_indices[2] >= 2 || tree_depth[1] != tree_depth[0] + 1 || tree_depth[2] != tree_depth[0] + 1) {
            return false;
        }
        if (tree_token_ids[1] < 0 || tree_token_ids[2] < 0 || tree_token_ids[1] == tree_token_ids[2]) {
            return false;
        }

        const int64_t topk_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) topk_tree_width_last,
            (int64_t) topk_tree_depth_last,
            (int64_t) topk_tree_non_root_nodes_last,
            (int64_t) tree_token_ids[0],
            (int64_t) tree_token_ids[1],
            (int64_t) tree_token_ids[2],
            (int64_t) tree_parent_indices[0],
            (int64_t) tree_parent_indices[1],
            (int64_t) tree_parent_indices[2],
            (int64_t) tree_depth[0],
            (int64_t) tree_depth[1],
            (int64_t) tree_depth[2],
            (int64_t) tree_rank[0],
            (int64_t) tree_rank[1],
            (int64_t) tree_rank[2],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        topk_tree_runtime_hash_last = common_speculative_fnv1a64(topk_words, sizeof(topk_words));
        topk_tree_runtime_hash_last ^= common_speculative_fnv1a64(&tree_cum_logprob[0], JETSPEC_TOPK_ABI_NODES * sizeof(tree_cum_logprob[0]));
        topk_tree_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOPK_TREE_RUNTIME_PHASE, std::strlen(JETSPEC_TOPK_TREE_RUNTIME_PHASE));
        topk_tree_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX, std::strlen(JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX));
        topk_tree_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TREE_BUILD_PHASE, std::strlen(JETSPEC_TREE_BUILD_PHASE));
        topk_tree_runtime_ready = true;
        n_topk_tree_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::tree_build_runtime_ready;
        return true;
    }

    bool build_topk_verify_mask_runtime() {
        topk_verify_mask_runtime_ready = false;
        topk_verify_mask_runtime_hash_last = 0;
        topk_verify_mask_runtime_seq_id_last = -1;
        actual_verify_mask_entries_last = 0;
        reset_verify_mask_arrays();
        if (!p5af_topk_verify_mask_enabled) {
            return false;
        }
        if (!p5ae_topk_tree_enabled || !topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || topk_tree_width_last != JETSPEC_TOPK_ABI_WIDTH || topk_tree_depth_last != JETSPEC_TOPK_ABI_DEPTH) {
            return false;
        }
        if (tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_parent_indices[1] != 0 || tree_parent_indices[2] != 0) {
            return false;
        }
        if (tree_depth[0] != 0 || tree_depth[1] != 1 || tree_depth[2] != 1 || tree_rank[0] != -1 || tree_rank[1] != 0 || tree_rank[2] != 1) {
            return false;
        }

        topk_verify_mask_runtime_seq_id_last = topk_tree_runtime_seq_id_last;
        actual_verify_mask_entries_last = JETSPEC_TOPK_ABI_MASK_ENTRIES;
        root_verify_mask_rows[0] = 0;
        root_verify_mask_cols[0] = 0;
        root_verify_mask_values[0] = 1;
        root_verify_mask_rows[1] = 1;
        root_verify_mask_cols[1] = 0;
        root_verify_mask_values[1] = 1;
        root_verify_mask_rows[2] = 1;
        root_verify_mask_cols[2] = 1;
        root_verify_mask_values[2] = 1;
        root_verify_mask_rows[3] = 2;
        root_verify_mask_cols[3] = 0;
        root_verify_mask_values[3] = 1;
        root_verify_mask_rows[4] = 2;
        root_verify_mask_cols[4] = 2;
        root_verify_mask_values[4] = 1;
        if (actual_verify_mask_entries_last > tree_build_actual_nodes_last * tree_build_actual_nodes_last) {
            return false;
        }
        for (int32_t i = 0; i < actual_verify_mask_entries_last; ++i) {
            if (root_verify_mask_rows[i] < 0 || root_verify_mask_rows[i] >= tree_build_actual_nodes_last ||
                    root_verify_mask_cols[i] < 0 || root_verify_mask_cols[i] >= tree_build_actual_nodes_last ||
                    root_verify_mask_values[i] != 1) {
                return false;
            }
        }
        for (int32_t i = actual_verify_mask_entries_last; i < JETSPEC_QWEN36_DRAFT_BLOCK_SIZE * JETSPEC_QWEN36_DRAFT_BLOCK_SIZE; ++i) {
            if (root_verify_mask_values[i] != 0) {
                return false;
            }
        }

        const int64_t mask_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) topk_tree_width_last,
            (int64_t) topk_tree_depth_last,
            (int64_t) tree_token_ids[0],
            (int64_t) tree_token_ids[1],
            (int64_t) tree_token_ids[2],
            (int64_t) root_verify_mask_rows[0],
            (int64_t) root_verify_mask_cols[0],
            (int64_t) root_verify_mask_rows[1],
            (int64_t) root_verify_mask_cols[1],
            (int64_t) root_verify_mask_rows[2],
            (int64_t) root_verify_mask_cols[2],
            (int64_t) root_verify_mask_rows[3],
            (int64_t) root_verify_mask_cols[3],
            (int64_t) root_verify_mask_rows[4],
            (int64_t) root_verify_mask_cols[4],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        topk_verify_mask_runtime_hash_last = common_speculative_fnv1a64(mask_words, sizeof(mask_words));
        topk_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(&root_verify_mask_values[0], JETSPEC_TOPK_ABI_MASK_ENTRIES * sizeof(root_verify_mask_values[0]));
        topk_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE, std::strlen(JETSPEC_TOPK_VERIFY_MASK_RUNTIME_PHASE));
        topk_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_PHASE, std::strlen(JETSPEC_VERIFY_MASK_PHASE));
        topk_verify_mask_runtime_ready = true;
        n_topk_verify_mask_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::verify_mask_runtime_ready;
        return true;
    }

    bool build_topk_accept_boundary_runtime() {
        topk_accept_boundary_runtime_ready = false;
        topk_accept_boundary_runtime_hash_last = 0;
        topk_accept_boundary_runtime_seq_id_last = -1;
        topk_accept_candidate_nodes_last = 0;
        topk_accept_boundary_verified_edges_last = 0;
        topk_actual_verified_logits_rows_last = 0;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        if (!p5ag_topk_accept_boundary_enabled) {
            return false;
        }
        if (!p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled) {
            return false;
        }
        if (!topk_verify_mask_runtime_ready || topk_verify_mask_runtime_hash_last == 0) {
            return false;
        }
        if (!topk_tree_runtime_ready || topk_tree_runtime_hash_last == 0) {
            return false;
        }
        if (topk_abi_root_tail_conflict()) {
            return false;
        }
        if (tree_build_actual_nodes_last != JETSPEC_TOPK_ABI_NODES || actual_verify_mask_entries_last != JETSPEC_TOPK_ABI_MASK_ENTRIES) {
            return false;
        }
        if (root_verify_mask_rows[0] != 0 || root_verify_mask_cols[0] != 0 || root_verify_mask_values[0] != 1 ||
                root_verify_mask_rows[1] != 1 || root_verify_mask_cols[1] != 0 || root_verify_mask_values[1] != 1 ||
                root_verify_mask_rows[2] != 1 || root_verify_mask_cols[2] != 1 || root_verify_mask_values[2] != 1 ||
                root_verify_mask_rows[3] != 2 || root_verify_mask_cols[3] != 0 || root_verify_mask_values[3] != 1 ||
                root_verify_mask_rows[4] != 2 || root_verify_mask_cols[4] != 2 || root_verify_mask_values[4] != 1) {
            return false;
        }
        if (tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_parent_indices[1] != 0 || tree_parent_indices[2] != 0) {
            return false;
        }
        if (tree_depth[0] != 0 || tree_depth[1] != 1 || tree_depth[2] != 1 || tree_rank[0] != -1 || tree_rank[1] != 0 || tree_rank[2] != 1) {
            return false;
        }

        topk_accept_boundary_runtime_seq_id_last = topk_verify_mask_runtime_seq_id_last;
        topk_accept_candidate_nodes_last = JETSPEC_TOPK_ABI_NON_ROOT_NODES;
        topk_accept_boundary_verified_edges_last = actual_verify_mask_entries_last;
        topk_actual_verified_logits_rows_last = 0;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        if (topk_accept_candidate_nodes_last != 2 || topk_accept_boundary_verified_edges_last != 5 ||
                topk_actual_verified_logits_rows_last != 0 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 ||
                correction_token_present_last != 0) {
            return false;
        }

        const int64_t accept_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) topk_tree_runtime_hash_last,
            (int64_t) topk_verify_mask_runtime_hash_last,
            (int64_t) topk_accept_boundary_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) topk_accept_candidate_nodes_last,
            (int64_t) topk_accept_boundary_verified_edges_last,
            (int64_t) topk_actual_verified_logits_rows_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) tree_token_ids[0],
            (int64_t) tree_token_ids[1],
            (int64_t) tree_token_ids[2],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        topk_accept_boundary_runtime_hash_last = common_speculative_fnv1a64(accept_words, sizeof(accept_words));
        topk_accept_boundary_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE, std::strlen(JETSPEC_TOPK_ACCEPT_BOUNDARY_RUNTIME_PHASE));
        topk_accept_boundary_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_PHASE, std::strlen(JETSPEC_ACCEPT_PATH_PHASE));
        topk_accept_boundary_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS, std::strlen(JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS));
        topk_accept_boundary_runtime_ready = true;
        n_topk_accept_boundary_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::accept_path_runtime_ready;
        return true;
    }

    bool build_root_only_verify_mask_runtime() {
        root_verify_mask_runtime_ready = false;
        root_verify_mask_runtime_hash_last = 0;
        root_verify_mask_runtime_seq_id_last = -1;
        actual_verify_mask_entries_last = 0;
        reset_verify_mask_arrays();
        if (!p5y_root_verify_mask_enabled) {
            return false;
        }
        if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != 1) {
            return false;
        }
        if (tree_build_actual_nodes_last > tree_build_node_budget_last) {
            return false;
        }
        if (tree_token_ids[0] < 0 || tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH || tree_rank[0] != -1) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        root_verify_mask_runtime_seq_id_last = root_tree_runtime_seq_id_last;
        actual_verify_mask_entries_last = 1;
        root_verify_mask_rows[0] = 0;
        root_verify_mask_cols[0] = 0;
        root_verify_mask_values[0] = 1;
        if (actual_verify_mask_entries_last > tree_build_actual_nodes_last * tree_build_actual_nodes_last) {
            return false;
        }
        if (root_verify_mask_values[0] != 1 || root_verify_mask_rows[0] != 0 || root_verify_mask_cols[0] != 0) {
            return false;
        }

        const int64_t mask_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) root_verify_mask_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) tree_token_ids[0],
            (int64_t) tree_parent_indices[0],
            (int64_t) tree_depth[0],
            (int64_t) tree_rank[0],
            (int64_t) root_verify_mask_rows[0],
            (int64_t) root_verify_mask_cols[0],
            (int64_t) root_verify_mask_values[0],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_verify_mask_runtime_hash_last = common_speculative_fnv1a64(mask_words, sizeof(mask_words));
        root_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(&tree_cum_logprob[0], sizeof(tree_cum_logprob[0]));
        root_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_VERIFY_MASK_RUNTIME_PHASE));
        root_verify_mask_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_PHASE, std::strlen(JETSPEC_VERIFY_MASK_PHASE));
        root_verify_mask_runtime_ready = true;
        n_root_verify_mask_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::verify_mask_runtime_ready;
        return true;
    }

    bool build_verify_mask_descriptor() {
        verify_mask_descriptor_ready = false;
        verify_mask_descriptor_hash_last = 0;
        verify_mask_seq_id_last = -1;
        actual_verify_mask_entries_last = 0;
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != 0) {
            return false;
        }
        if (tree_build_node_budget_last <= 0 || tree_build_node_budget_last > JETSPEC_QWEN36_DRAFT_BLOCK_SIZE) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        verify_mask_seq_id_last = pre_round_seq_id_last;
        actual_verify_mask_entries_last = 0;

        std::vector<int64_t> mask_words;
        mask_words.reserve(17 + target_tap_row_state.size() * 3);
        mask_words.push_back((int64_t) transaction_plan_hash_last);
        mask_words.push_back((int64_t) pre_round_snapshot_hash_last);
        mask_words.push_back((int64_t) transient_reservation_hash_last);
        mask_words.push_back((int64_t) tree_build_descriptor_hash_last);
        mask_words.push_back((int64_t) pre_round_prompt_tokens_last);
        mask_words.push_back((int64_t) pre_round_prompt_hash_last);
        mask_words.push_back((int64_t) verify_mask_seq_id_last);
        mask_words.push_back((int64_t) tree_build_node_budget_last);
        mask_words.push_back((int64_t) tree_build_actual_nodes_last);
        mask_words.push_back((int64_t) actual_verify_mask_entries_last);
        mask_words.push_back((int64_t) n_target_tap_rows_cached);
        mask_words.push_back((int64_t) target_tap_hash_last);
        mask_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE);
        mask_words.push_back(JETSPEC_TRANSACTION_PHASE_COUNT);
        mask_words.push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT);
        mask_words.push_back(JETSPEC_TREE_ROOT_PARENT);
        mask_words.push_back(JETSPEC_TREE_ROOT_DEPTH);
        for (const auto & row : target_tap_row_state) {
            mask_words.push_back((int64_t) row.batch_index);
            mask_words.push_back((int64_t) row.pos);
            mask_words.push_back((int64_t) row.seq_id);
        }

        verify_mask_descriptor_hash_last = common_speculative_fnv1a64(mask_words.data(), mask_words.size() * sizeof(mask_words[0]));
        verify_mask_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_PHASE, std::strlen(JETSPEC_VERIFY_MASK_PHASE));
        verify_mask_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_ROLLBACK_POINT, std::strlen(JETSPEC_VERIFY_MASK_ROLLBACK_POINT));
        verify_mask_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_VERIFY_MASK_DESCRIPTOR, std::strlen(JETSPEC_VERIFY_MASK_DESCRIPTOR));
        verify_mask_descriptor_ready = true;
        n_verify_mask_descriptors++;
        runtime_phase = jetspec_runtime_phase::verify_mask_descriptor_ready;
        return true;
    }

    bool build_root_anchor_accept_path_runtime() {
        root_anchor_accept_path_runtime_ready = false;
        root_anchor_accept_path_runtime_hash_last = 0;
        root_anchor_accept_path_runtime_seq_id_last = -1;
        root_verified_anchor_last = 0;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        if (!p5z_root_anchor_accept_path_enabled) {
            return false;
        }
        if (!p5y_root_verify_mask_enabled || !root_verify_mask_runtime_ready || root_verify_mask_runtime_hash_last == 0) {
            return false;
        }
        if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != 1 || actual_verify_mask_entries_last != 1) {
            return false;
        }
        if (root_verify_mask_values[0] != 1 || root_verify_mask_rows[0] != 0 || root_verify_mask_cols[0] != 0) {
            return false;
        }
        if (tree_token_ids[0] < 0 || tree_parent_indices[0] != JETSPEC_TREE_ROOT_PARENT || tree_depth[0] != JETSPEC_TREE_ROOT_DEPTH || tree_rank[0] != -1) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        root_anchor_accept_path_runtime_seq_id_last = root_verify_mask_runtime_seq_id_last;
        root_verified_anchor_last = 1;
        accept_path_len_last = 0;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        if (root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0) {
            return false;
        }

        const int64_t accept_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) root_verify_mask_runtime_hash_last,
            (int64_t) root_anchor_accept_path_runtime_seq_id_last,
            (int64_t) tree_build_node_budget_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) root_verified_anchor_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) tree_token_ids[0],
            (int64_t) root_verify_mask_values[0],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_anchor_accept_path_runtime_hash_last = common_speculative_fnv1a64(accept_words, sizeof(accept_words));
        root_anchor_accept_path_runtime_hash_last ^= common_speculative_fnv1a64(&tree_cum_logprob[0], sizeof(tree_cum_logprob[0]));
        root_anchor_accept_path_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_ANCHOR_ACCEPT_PATH_RUNTIME_PHASE));
        root_anchor_accept_path_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_PHASE, std::strlen(JETSPEC_ACCEPT_PATH_PHASE));
        root_anchor_accept_path_runtime_ready = true;
        n_root_anchor_accept_path_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::accept_path_runtime_ready;
        return true;
    }

    bool build_root_token_commit_noop_runtime() {
        root_token_commit_noop_runtime_ready = false;
        root_token_commit_noop_runtime_hash_last = 0;
        root_token_commit_noop_runtime_seq_id_last = -1;
        actual_committed_tokens_last = 0;
        if (!p5aa_root_token_commit_noop_enabled) {
            return false;
        }
        if (!p5z_root_anchor_accept_path_enabled || !root_anchor_accept_path_runtime_ready || root_anchor_accept_path_runtime_hash_last == 0) {
            return false;
        }
        if (!p5y_root_verify_mask_enabled || !root_verify_mask_runtime_ready || root_verify_mask_runtime_hash_last == 0) {
            return false;
        }
        if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (!transient_reservation_ready || transient_reservation_hash_last == 0) {
            return false;
        }
        if (!transaction_plan_ready || transaction_plan_hash_last == 0) {
            return false;
        }
        if (!pre_round_snapshot_ready || pre_round_snapshot_hash_last == 0) {
            return false;
        }
        if (tree_build_actual_nodes_last != 1 || actual_verify_mask_entries_last != 1) {
            return false;
        }
        if (root_verified_anchor_last != 1 || accept_path_len_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0) {
            return false;
        }
        if (token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready || rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready) {
            return false;
        }
        if (actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        root_token_commit_noop_runtime_seq_id_last = root_anchor_accept_path_runtime_seq_id_last;
        actual_committed_tokens_last = 0;
        if (actual_committed_tokens_last != 0) {
            return false;
        }

        const int64_t commit_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) transient_reservation_hash_last,
            (int64_t) tree_build_descriptor_hash_last,
            (int64_t) root_tree_runtime_hash_last,
            (int64_t) root_verify_mask_runtime_hash_last,
            (int64_t) root_anchor_accept_path_runtime_hash_last,
            (int64_t) root_token_commit_noop_runtime_seq_id_last,
            (int64_t) root_verified_anchor_last,
            (int64_t) accept_path_len_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) tree_build_actual_nodes_last,
            (int64_t) actual_verify_mask_entries_last,
            (int64_t) tree_token_ids[0],
            (int64_t) pre_round_prompt_tokens_last,
            (int64_t) pre_round_prompt_hash_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_QWEN36_DRAFT_BLOCK_SIZE,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_token_commit_noop_runtime_hash_last = common_speculative_fnv1a64(commit_words, sizeof(commit_words));
        root_token_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_TOKEN_COMMIT_NOOP_RUNTIME_PHASE));
        root_token_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_PHASE, std::strlen(JETSPEC_TOKEN_COMMIT_PHASE));
        root_token_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT));
        root_token_commit_noop_runtime_ready = true;
        n_root_token_commit_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::token_commit_runtime_ready;
        return true;
    }

    bool build_root_hidden_kv_commit_noop_runtime() {
        root_hidden_kv_commit_noop_runtime_ready = false;
        root_hidden_kv_commit_noop_runtime_hash_last = 0;
        root_hidden_kv_commit_noop_runtime_seq_id_last = -1;
        actual_survivor_pages_committed_last = 0;
        if (!p5ab_root_hidden_kv_commit_noop_enabled) {
            return false;
        }
        if (!p5aa_root_token_commit_noop_enabled || !root_token_commit_noop_runtime_ready || root_token_commit_noop_runtime_hash_last == 0) {
            return false;
        }
        if (!p5z_root_anchor_accept_path_enabled || !root_anchor_accept_path_runtime_ready || root_anchor_accept_path_runtime_hash_last == 0) {
            return false;
        }
        if (!p5y_root_verify_mask_enabled || !root_verify_mask_runtime_ready || root_verify_mask_runtime_hash_last == 0) {
            return false;
        }
        if (!p5x_root_tree_enabled || !root_tree_runtime_ready || root_tree_runtime_hash_last == 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_accepted_nodes_last != 0 || correction_token_present_last != 0) {
            return false;
        }
        if (token_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_ready || rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready) {
            return false;
        }
        if (actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0 || actual_publish_visible_state_last != 0) {
            return false;
        }

        root_hidden_kv_commit_noop_runtime_seq_id_last = root_token_commit_noop_runtime_seq_id_last;
        actual_survivor_pages_committed_last = 0;
        if (actual_survivor_pages_committed_last != 0) {
            return false;
        }

        const int64_t hidden_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) root_token_commit_noop_runtime_hash_last,
            (int64_t) root_hidden_kv_commit_noop_runtime_seq_id_last,
            (int64_t) root_verified_anchor_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_hidden_kv_commit_noop_runtime_hash_last = common_speculative_fnv1a64(hidden_words, sizeof(hidden_words));
        root_hidden_kv_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_HIDDEN_KV_COMMIT_NOOP_RUNTIME_PHASE));
        root_hidden_kv_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE));
        root_hidden_kv_commit_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT));
        root_hidden_kv_commit_noop_runtime_ready = true;
        n_root_hidden_kv_commit_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::hidden_kv_commit_runtime_ready;
        return true;
    }

    bool build_root_rejected_branch_discard_noop_runtime() {
        root_rejected_branch_discard_noop_runtime_ready = false;
        root_rejected_branch_discard_noop_runtime_hash_last = 0;
        root_rejected_branch_discard_noop_runtime_seq_id_last = -1;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        if (!p5ac_root_rejected_branch_discard_noop_enabled) {
            return false;
        }
        if (!p5ab_root_hidden_kv_commit_noop_enabled || !root_hidden_kv_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_hash_last == 0) {
            return false;
        }
        if (!p5aa_root_token_commit_noop_enabled || !root_token_commit_noop_runtime_ready || root_token_commit_noop_runtime_hash_last == 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0) {
            return false;
        }
        if (hidden_kv_survivor_commit_descriptor_ready || rejected_branch_discard_descriptor_ready || publish_gate_descriptor_ready) {
            return false;
        }
        if (actual_publish_visible_state_last != 0) {
            return false;
        }

        root_rejected_branch_discard_noop_runtime_seq_id_last = root_hidden_kv_commit_noop_runtime_seq_id_last;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        if (actual_pages_discarded_last != 0 || rejected_branch_pages_reachable_after_discard_last != 0) {
            return false;
        }

        const int64_t discard_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) root_hidden_kv_commit_noop_runtime_hash_last,
            (int64_t) root_rejected_branch_discard_noop_runtime_seq_id_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_rejected_branch_discard_noop_runtime_hash_last = common_speculative_fnv1a64(discard_words, sizeof(discard_words));
        root_rejected_branch_discard_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_REJECTED_BRANCH_DISCARD_NOOP_RUNTIME_PHASE));
        root_rejected_branch_discard_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE));
        root_rejected_branch_discard_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT));
        root_rejected_branch_discard_noop_runtime_ready = true;
        n_root_rejected_branch_discard_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::rejected_branch_discard_runtime_ready;
        return true;
    }

    bool build_root_publish_gate_noop_runtime() {
        root_publish_gate_noop_runtime_ready = false;
        root_publish_gate_noop_runtime_hash_last = 0;
        root_publish_gate_noop_runtime_seq_id_last = -1;
        actual_publish_visible_state_last = 0;
        root_runtime_ready_for_real_test_last = 0;
        if (!p5ad_root_publish_gate_noop_enabled) {
            return false;
        }
        if (!p5ac_root_rejected_branch_discard_noop_enabled || !root_rejected_branch_discard_noop_runtime_ready || root_rejected_branch_discard_noop_runtime_hash_last == 0) {
            return false;
        }
        if (!p5ab_root_hidden_kv_commit_noop_enabled || !root_hidden_kv_commit_noop_runtime_ready || root_hidden_kv_commit_noop_runtime_hash_last == 0) {
            return false;
        }
        if (!p5aa_root_token_commit_noop_enabled || !root_token_commit_noop_runtime_ready || root_token_commit_noop_runtime_hash_last == 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0) {
            return false;
        }
        if (rejected_branch_pages_reachable_after_discard_last != 0) {
            return false;
        }
        if (publish_gate_descriptor_ready) {
            return false;
        }

        root_publish_gate_noop_runtime_seq_id_last = root_rejected_branch_discard_noop_runtime_seq_id_last;
        actual_publish_visible_state_last = 0;
        root_runtime_ready_for_real_test_last = 1;
        if (actual_publish_visible_state_last != 0 || root_runtime_ready_for_real_test_last != 1) {
            return false;
        }

        const int64_t publish_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) root_rejected_branch_discard_noop_runtime_hash_last,
            (int64_t) root_publish_gate_noop_runtime_seq_id_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) actual_publish_visible_state_last,
            (int64_t) root_runtime_ready_for_real_test_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        root_publish_gate_noop_runtime_hash_last = common_speculative_fnv1a64(publish_words, sizeof(publish_words));
        root_publish_gate_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_ROOT_PUBLISH_GATE_NOOP_RUNTIME_PHASE, std::strlen(JETSPEC_ROOT_PUBLISH_GATE_NOOP_RUNTIME_PHASE));
        root_publish_gate_noop_runtime_hash_last ^= common_speculative_fnv1a64(JETSPEC_PUBLISH_GATE_PHASE, std::strlen(JETSPEC_PUBLISH_GATE_PHASE));
        root_publish_gate_noop_runtime_ready = true;
        n_root_publish_gate_noop_runtime_builds++;
        runtime_phase = jetspec_runtime_phase::publish_gate_runtime_ready;
        return true;
    }

    bool build_accept_path_descriptor() {
        accept_path_descriptor_ready = false;
        accept_path_descriptor_hash_last = 0;
        accept_path_seq_id_last = -1;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;
        if (!verify_mask_descriptor_ready || verify_mask_descriptor_hash_last == 0) {
            return false;
        }
        if (!tree_build_descriptor_ready || tree_build_descriptor_hash_last == 0) {
            return false;
        }
        if (actual_verify_mask_entries_last != 0 || tree_build_actual_nodes_last != 0) {
            return false;
        }
        if (n_target_tap_rows_cached == 0 || target_tap_row_state.size() != n_target_tap_rows_cached) {
            return false;
        }

        accept_path_seq_id_last = pre_round_seq_id_last;
        actual_accepted_nodes_last = 0;
        correction_token_present_last = 0;

        std::vector<int64_t> accept_words;
        accept_words.reserve(18 + target_tap_row_state.size() * 3);
        accept_words.push_back((int64_t) transaction_plan_hash_last);
        accept_words.push_back((int64_t) pre_round_snapshot_hash_last);
        accept_words.push_back((int64_t) transient_reservation_hash_last);
        accept_words.push_back((int64_t) tree_build_descriptor_hash_last);
        accept_words.push_back((int64_t) verify_mask_descriptor_hash_last);
        accept_words.push_back((int64_t) pre_round_prompt_tokens_last);
        accept_words.push_back((int64_t) pre_round_prompt_hash_last);
        accept_words.push_back((int64_t) accept_path_seq_id_last);
        accept_words.push_back((int64_t) tree_build_node_budget_last);
        accept_words.push_back((int64_t) tree_build_actual_nodes_last);
        accept_words.push_back((int64_t) actual_verify_mask_entries_last);
        accept_words.push_back((int64_t) actual_accepted_nodes_last);
        accept_words.push_back((int64_t) correction_token_present_last);
        accept_words.push_back((int64_t) n_target_tap_rows_cached);
        accept_words.push_back((int64_t) target_tap_hash_last);
        accept_words.push_back(JETSPEC_QWEN36_DRAFT_BLOCK_SIZE);
        accept_words.push_back(JETSPEC_TRANSACTION_PHASE_COUNT);
        accept_words.push_back(JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT);
        for (const auto & row : target_tap_row_state) {
            accept_words.push_back((int64_t) row.batch_index);
            accept_words.push_back((int64_t) row.pos);
            accept_words.push_back((int64_t) row.seq_id);
        }

        accept_path_descriptor_hash_last = common_speculative_fnv1a64(accept_words.data(), accept_words.size() * sizeof(accept_words[0]));
        accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_PHASE, std::strlen(JETSPEC_ACCEPT_PATH_PHASE));
        accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_ROLLBACK_POINT, std::strlen(JETSPEC_ACCEPT_PATH_ROLLBACK_POINT));
        accept_path_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_ACCEPT_PATH_DESCRIPTOR, std::strlen(JETSPEC_ACCEPT_PATH_DESCRIPTOR));
        accept_path_descriptor_ready = true;
        n_accept_path_descriptors++;
        runtime_phase = jetspec_runtime_phase::accept_path_descriptor_ready;
        return true;
    }

    bool build_token_commit_descriptor() {
        token_commit_descriptor_ready = false;
        token_commit_descriptor_hash_last = 0;
        token_commit_seq_id_last = -1;
        actual_committed_tokens_last = 0;
        if (!accept_path_descriptor_ready || accept_path_descriptor_hash_last == 0) {
            return false;
        }
        if (actual_accepted_nodes_last != 0 || correction_token_present_last != 0) {
            return false;
        }

        token_commit_seq_id_last = pre_round_seq_id_last;
        actual_committed_tokens_last = 0;

        const int64_t commit_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) accept_path_descriptor_hash_last,
            (int64_t) token_commit_seq_id_last,
            (int64_t) actual_accepted_nodes_last,
            (int64_t) correction_token_present_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        token_commit_descriptor_hash_last = common_speculative_fnv1a64(commit_words, sizeof(commit_words));
        token_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_PHASE, std::strlen(JETSPEC_TOKEN_COMMIT_PHASE));
        token_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_TOKEN_COMMIT_ROLLBACK_POINT));
        token_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_TOKEN_COMMIT_DESCRIPTOR, std::strlen(JETSPEC_TOKEN_COMMIT_DESCRIPTOR));
        token_commit_descriptor_ready = true;
        n_token_commit_descriptors++;
        runtime_phase = jetspec_runtime_phase::token_commit_descriptor_ready;
        return true;
    }

    bool build_hidden_kv_survivor_commit_descriptor() {
        hidden_kv_survivor_commit_descriptor_ready = false;
        hidden_kv_survivor_commit_descriptor_hash_last = 0;
        hidden_kv_survivor_commit_seq_id_last = -1;
        actual_survivor_pages_committed_last = 0;
        if (!token_commit_descriptor_ready || token_commit_descriptor_hash_last == 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0) {
            return false;
        }

        hidden_kv_survivor_commit_seq_id_last = pre_round_seq_id_last;
        actual_survivor_pages_committed_last = 0;

        const int64_t hidden_kv_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) token_commit_descriptor_hash_last,
            (int64_t) hidden_kv_survivor_commit_seq_id_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        hidden_kv_survivor_commit_descriptor_hash_last = common_speculative_fnv1a64(hidden_kv_words, sizeof(hidden_kv_words));
        hidden_kv_survivor_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_PHASE));
        hidden_kv_survivor_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_ROLLBACK_POINT));
        hidden_kv_survivor_commit_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_DESCRIPTOR, std::strlen(JETSPEC_HIDDEN_KV_SURVIVOR_COMMIT_DESCRIPTOR));
        hidden_kv_survivor_commit_descriptor_ready = true;
        n_hidden_kv_survivor_commit_descriptors++;
        runtime_phase = jetspec_runtime_phase::hidden_kv_survivor_commit_descriptor_ready;
        return true;
    }

    bool build_rejected_branch_discard_descriptor() {
        rejected_branch_discard_descriptor_ready = false;
        rejected_branch_discard_descriptor_hash_last = 0;
        rejected_branch_discard_seq_id_last = -1;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;
        if (!hidden_kv_survivor_commit_descriptor_ready || hidden_kv_survivor_commit_descriptor_hash_last == 0) {
            return false;
        }
        if (actual_survivor_pages_committed_last != 0) {
            return false;
        }

        rejected_branch_discard_seq_id_last = pre_round_seq_id_last;
        actual_pages_discarded_last = 0;
        rejected_branch_pages_reachable_after_discard_last = 0;

        const int64_t discard_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) hidden_kv_survivor_commit_descriptor_hash_last,
            (int64_t) rejected_branch_discard_seq_id_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        rejected_branch_discard_descriptor_hash_last = common_speculative_fnv1a64(discard_words, sizeof(discard_words));
        rejected_branch_discard_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_PHASE));
        rejected_branch_discard_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_ROLLBACK_POINT));
        rejected_branch_discard_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_REJECTED_BRANCH_DISCARD_DESCRIPTOR, std::strlen(JETSPEC_REJECTED_BRANCH_DISCARD_DESCRIPTOR));
        rejected_branch_discard_descriptor_ready = true;
        n_rejected_branch_discard_descriptors++;
        runtime_phase = jetspec_runtime_phase::rejected_branch_discard_descriptor_ready;
        return true;
    }

    bool build_publish_gate_descriptor() {
        publish_gate_descriptor_ready = false;
        publish_gate_descriptor_hash_last = 0;
        publish_gate_seq_id_last = -1;
        actual_publish_visible_state_last = 0;
        if (!rejected_branch_discard_descriptor_ready || rejected_branch_discard_descriptor_hash_last == 0) {
            return false;
        }
        if (actual_committed_tokens_last != 0 || actual_survivor_pages_committed_last != 0 || actual_pages_discarded_last != 0) {
            return false;
        }
        if (rejected_branch_pages_reachable_after_discard_last != 0) {
            return false;
        }

        publish_gate_seq_id_last = pre_round_seq_id_last;
        actual_publish_visible_state_last = 0;

        const int64_t publish_words[] = {
            (int64_t) transaction_plan_hash_last,
            (int64_t) pre_round_snapshot_hash_last,
            (int64_t) token_commit_descriptor_hash_last,
            (int64_t) hidden_kv_survivor_commit_descriptor_hash_last,
            (int64_t) rejected_branch_discard_descriptor_hash_last,
            (int64_t) publish_gate_seq_id_last,
            (int64_t) actual_committed_tokens_last,
            (int64_t) actual_survivor_pages_committed_last,
            (int64_t) actual_pages_discarded_last,
            (int64_t) rejected_branch_pages_reachable_after_discard_last,
            (int64_t) actual_publish_visible_state_last,
            (int64_t) n_target_tap_rows_cached,
            (int64_t) target_tap_hash_last,
            JETSPEC_TRANSACTION_PHASE_COUNT,
            JETSPEC_TRANSACTION_ROLLBACK_POINT_COUNT,
        };
        publish_gate_descriptor_hash_last = common_speculative_fnv1a64(publish_words, sizeof(publish_words));
        publish_gate_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_PUBLISH_GATE_PHASE, std::strlen(JETSPEC_PUBLISH_GATE_PHASE));
        publish_gate_descriptor_hash_last ^= common_speculative_fnv1a64(JETSPEC_PUBLISH_GATE_DESCRIPTOR, std::strlen(JETSPEC_PUBLISH_GATE_DESCRIPTOR));
        publish_gate_descriptor_ready = true;
        n_publish_gate_descriptors++;
        runtime_phase = jetspec_runtime_phase::publish_gate_descriptor_ready;
        return true;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        reset_runtime_state();
        runtime_phase = jetspec_runtime_phase::waiting_for_pre_round_snapshot;
        if (!build_pre_round_snapshot(seq_id, prompt)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_pre_round_snapshot, 0, 0, nullptr);
            return;
        }
        if (trace_taps) {
            LOG_INF("%s: draft-jetspec pre_round_snapshot_ready=%d pre_round_snapshot_hash=%016" PRIx64 " pre_round_seq_id=%d pre_round_prompt_tokens=%zu pre_round_prompt_hash=%016" PRIx64 " transaction_phase=%s no_reserve=1 no_tree_build=1 no_verify_mask=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, pre_round_snapshot_ready ? 1 : 0, pre_round_snapshot_hash_last,
                    pre_round_seq_id_last, pre_round_prompt_tokens_last, pre_round_prompt_hash_last,
                    JETSPEC_PRE_ROUND_SNAPSHOT_PHASE);
        }
    }

    bool process(const llama_batch & batch) override {
        if (!target_taps_active) {
            return true;
        }

        int32_t n_rows = 0;
        if (batch.logits != nullptr) {
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                n_rows += batch.logits[i] != 0;
            }
        }
        if (n_rows == 0) {
            n_target_tap_rows_cached = 0;
            transaction_plan_hash_last = 0;
            transient_reservation_hash_last = 0;
            tree_build_descriptor_hash_last = 0;
            root_tree_runtime_hash_last = 0;
            root_verify_mask_runtime_hash_last = 0;
            root_anchor_accept_path_runtime_hash_last = 0;
            root_token_commit_noop_runtime_hash_last = 0;
            root_hidden_kv_commit_noop_runtime_hash_last = 0;
            root_rejected_branch_discard_noop_runtime_hash_last = 0;
            root_publish_gate_noop_runtime_hash_last = 0;
            topk_tree_runtime_hash_last = 0;
            topk_verify_mask_runtime_hash_last = 0;
            topk_accept_boundary_runtime_hash_last = 0;
            real_draft_head_canary_hash_last = 0;
            real_draft_head_topk_candidate_hash_last = 0;
            real_draft_head_topk_tree_hash_last = 0;
            real_draft_head_topk_verify_mask_hash_last = 0;
            real_draft_head_topk_target_logits_walk_canary_hash_last = 0;
            verify_mask_descriptor_hash_last = 0;
            accept_path_descriptor_hash_last = 0;
            token_commit_descriptor_hash_last = 0;
            hidden_kv_survivor_commit_descriptor_hash_last = 0;
            rejected_branch_discard_descriptor_hash_last = 0;
            publish_gate_descriptor_hash_last = 0;
            root_tree_runtime_seq_id_last = -1;
            root_verify_mask_runtime_seq_id_last = -1;
            root_anchor_accept_path_runtime_seq_id_last = -1;
            root_token_commit_noop_runtime_seq_id_last = -1;
            root_hidden_kv_commit_noop_runtime_seq_id_last = -1;
            root_rejected_branch_discard_noop_runtime_seq_id_last = -1;
            root_publish_gate_noop_runtime_seq_id_last = -1;
            topk_tree_runtime_seq_id_last = -1;
            topk_verify_mask_runtime_seq_id_last = -1;
            topk_accept_boundary_runtime_seq_id_last = -1;
            real_draft_head_topk_candidate_seq_id_last = -1;
            real_draft_head_topk_tree_seq_id_last = -1;
            real_draft_head_topk_verify_mask_seq_id_last = -1;
            real_draft_head_topk_target_logits_walk_canary_seq_id_last = -1;
            transaction_plan_phase_count_last = 0;
            transaction_plan_rollback_count_last = 0;
            tree_build_actual_nodes_last = 0;
            topk_tree_width_last = 0;
            topk_tree_depth_last = 0;
            topk_tree_non_root_nodes_last = 0;
            topk_accept_candidate_nodes_last = 0;
            topk_accept_boundary_verified_edges_last = 0;
            topk_actual_verified_logits_rows_last = 0;
            real_draft_head_canary_ctx_present_last = params.ctx_dft != nullptr ? 1 : 0;
            real_draft_head_canary_decode_rc_last = 0;
            real_draft_head_canary_input_rows_last = 0;
            real_draft_head_canary_input_width_last = 0;
            real_draft_head_canary_output_rows_last = 0;
            real_draft_head_canary_output_width_last = 0;
            real_draft_head_canary_logits_rows_last = 0;
            real_draft_head_canary_topk_rows_last = 0;
            real_draft_head_canary_topk_k_last = 0;
            real_draft_head_canary_top1_id_last = -1;
            real_draft_head_canary_top2_id_last = -1;
            real_draft_head_canary_top1_logit_last = 0.0f;
            real_draft_head_canary_top2_logit_last = 0.0f;
            real_draft_head_topk_parent_node_last = -1;
            real_draft_head_topk_candidate_nodes_last = 0;
            real_draft_head_topk_verified_logits_rows_last = 0;
            real_draft_head_topk_tree_seq_id_last = -1;
            real_draft_head_topk_verify_mask_seq_id_last = -1;
            real_draft_head_topk_tree_nodes_last = 0;
            real_draft_head_topk_verify_mask_entries_last = 0;
            real_draft_head_topk_target_logits_planned_rows_last = 0;
            real_draft_head_topk_target_logits_actual_rows_walked_last = 0;
            real_draft_head_topk_target_logits_batch_index_last = -1;
            real_draft_head_topk_target_logits_pos_last = -1;
            real_draft_head_topk_target_logits_seq_id_last = -1;
            real_draft_head_topk_target_logits_width_last = 0;
            real_draft_head_topk_target_logits_candidate_nodes_last = 0;
            real_draft_head_topk_target_logits_candidate_scores.fill(0.0f);
            real_draft_head_topk_candidate_ids.fill(-1);
            real_draft_head_topk_candidate_logits.fill(0.0f);
            reset_real_draft_head_topk_tree_arrays();
            reset_real_draft_head_topk_verify_mask_arrays();
            reset_tree_arrays();
            reset_verify_mask_arrays();
            actual_verify_mask_entries_last = 0;
            root_verified_anchor_last = 0;
            accept_path_len_last = 0;
            actual_accepted_nodes_last = 0;
            correction_token_present_last = 0;
            actual_committed_tokens_last = 0;
            actual_survivor_pages_committed_last = 0;
            actual_pages_discarded_last = 0;
            rejected_branch_pages_reachable_after_discard_last = 0;
            actual_publish_visible_state_last = 0;
            root_runtime_ready_for_real_test_last = 0;
            transaction_plan_ready = false;
            transient_reservation_ready = false;
            tree_build_descriptor_ready = false;
            root_tree_runtime_ready = false;
            root_verify_mask_runtime_ready = false;
            root_anchor_accept_path_runtime_ready = false;
            root_token_commit_noop_runtime_ready = false;
            root_hidden_kv_commit_noop_runtime_ready = false;
            root_rejected_branch_discard_noop_runtime_ready = false;
            root_publish_gate_noop_runtime_ready = false;
            topk_tree_runtime_ready = false;
            topk_verify_mask_runtime_ready = false;
            topk_accept_boundary_runtime_ready = false;
            real_draft_head_canary_ready = false;
            real_draft_head_logits_canary_ready = false;
            real_draft_head_topk_candidate_runtime_ready = false;
            real_draft_head_topk_tree_runtime_ready = false;
            real_draft_head_topk_verify_mask_runtime_ready = false;
            real_draft_head_topk_target_logits_walk_canary_ready = false;
            verify_mask_descriptor_ready = false;
            accept_path_descriptor_ready = false;
            token_commit_descriptor_ready = false;
            hidden_kv_survivor_commit_descriptor_ready = false;
            rejected_branch_discard_descriptor_ready = false;
            publish_gate_descriptor_ready = false;
            target_tap_rows.clear();
            target_tap_row_state.clear();
            runtime_phase = jetspec_runtime_phase::waiting_for_target_taps;
            return true;
        }

        if (!pre_round_snapshot_ready) {
            runtime_phase = jetspec_runtime_phase::waiting_for_pre_round_snapshot;
            return true;
        }

        const int32_t tap_count = llama_get_jetspec_target_hidden_tap_count(params.ctx_tgt);
        const int32_t tap_width = llama_get_jetspec_target_hidden_tap_width(params.ctx_tgt);
        float * taps = llama_get_jetspec_target_hidden_taps(params.ctx_tgt);
        if (tap_count != 5 || tap_width != 10240 || taps == nullptr) {
            LOG_WRN("%s: draft-jetspec target taps unavailable after target decode (%d taps, width %d, ptr %p); disabling JetSpec tap ingestion\n",
                    __func__, tap_count, tap_width, (void *) taps);
            disable_runtime_state(jetspec_runtime_failure::invalid_target_taps, tap_count, tap_width, taps);
            return true;
        }

        const size_t n_values = (size_t) n_rows * (size_t) tap_width;
        target_tap_rows.assign(taps, taps + n_values);
        target_tap_hash_last = common_speculative_fnv1a64(target_tap_rows.data(), n_values * sizeof(float));
        target_tap_count_last = tap_count;
        target_tap_width_last = tap_width;
        n_target_tap_rows_cached = (size_t) n_rows;
        n_target_tap_rows_total += (size_t) n_rows;
        n_target_tap_process++;
        runtime_phase = jetspec_runtime_phase::target_taps_captured;
        runtime_failure = jetspec_runtime_failure::none;

        const int32_t n_row_state = capture_target_tap_row_state(batch);
        if (n_row_state != n_rows) {
            disable_runtime_state(jetspec_runtime_failure::invalid_target_taps, tap_count, tap_width, taps);
            return true;
        }
        if (!run_real_draft_head_canary_decode(tap_width)) {
            disable_runtime_state(p5aj_real_draft_head_logits_canary_enabled ?
                    jetspec_runtime_failure::invalid_real_draft_head_logits_canary_runtime :
                    jetspec_runtime_failure::invalid_real_draft_head_canary_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (trace_taps && p5aj_real_draft_head_logits_canary_enabled) {
            LOG_INF("%s: draft-jetspec p5aj_real_draft_head_logits_canary phase=%s logits_canary_enabled=1 ctx_dft_present=%d decode_rc=%d input_rows=%d input_width=%d logits_rows=%d logits_width=%d logits_hash=%016" PRIx64 " topk_rows=%d topk_k=%d top1_id=%d top2_id=%d top1_logit=%.6g top2_logit=%.6g logits_extract_source=private_embeddings_output real_draft_head_tensors_bound=1 hidden_taps_source=target_tap_capture no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                    __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_canary_ctx_present_last,
                    real_draft_head_canary_decode_rc_last, real_draft_head_canary_input_rows_last,
                    real_draft_head_canary_input_width_last, real_draft_head_canary_logits_rows_last,
                    real_draft_head_canary_output_width_last, real_draft_head_canary_hash_last,
                    real_draft_head_canary_topk_rows_last, real_draft_head_canary_topk_k_last,
                    real_draft_head_canary_top1_id_last, real_draft_head_canary_top2_id_last,
                    (double) real_draft_head_canary_top1_logit_last,
                    (double) real_draft_head_canary_top2_logit_last);
        } else if (trace_taps && p5ai_real_draft_head_canary_enabled) {
            LOG_INF("%s: draft-jetspec p5ai_real_draft_head_canary phase=%s canary_enabled=1 ctx_dft_present=%d decode_rc=%d input_rows=%d input_width=%d output_rows=%d output_width=%d graph_hash=%016" PRIx64 " actual_draft_head_graph_rows=%d actual_draft_head_logits_rows=%d actual_topk_rows=0 real_draft_head_tensors_bound=1 hidden_taps_source=target_tap_capture no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                    __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_canary_ctx_present_last,
                    real_draft_head_canary_decode_rc_last, real_draft_head_canary_input_rows_last,
                    real_draft_head_canary_input_width_last, real_draft_head_canary_output_rows_last,
                    real_draft_head_canary_output_width_last, real_draft_head_canary_hash_last,
                    real_draft_head_canary_output_rows_last, real_draft_head_canary_logits_rows_last);
        }
        if (!build_transaction_plan_scaffold()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_transaction_plan, tap_count, tap_width, taps);
            return true;
        }
        if (!build_transient_reservation_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_transient_reservation_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_tree_build_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_tree_build_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (p5av_real_draft_head_topk_target_logits_walk_canary_enabled && (!p5as_real_draft_head_topk_publish_gate_noop_enabled || !p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled || !p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled || !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_target_logits_walk_canary_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5as_real_draft_head_topk_publish_gate_noop_enabled && (!p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled || !p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled || !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_publish_gate_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled && (!p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled || !p5ap_real_draft_head_topk_token_commit_noop_enabled || !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled && (!p5ap_real_draft_head_topk_token_commit_noop_enabled || !p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ap_real_draft_head_topk_token_commit_noop_enabled && (!p5ao_real_draft_head_topk_accept_path_descriptor_enabled || !p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_token_commit_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ao_real_draft_head_topk_accept_path_descriptor_enabled && (!p5an_real_draft_head_topk_accept_boundary_enabled || !p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_accept_path_descriptor_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5an_real_draft_head_topk_accept_boundary_enabled && (!p5am_real_draft_head_topk_verify_mask_enabled || !p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_accept_boundary_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5am_real_draft_head_topk_verify_mask_enabled && (!p5al_real_draft_head_topk_tree_enabled || !p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_verify_mask_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5al_real_draft_head_topk_tree_enabled && (!p5ak_real_draft_head_topk_candidate_enabled || !p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_tree_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ak_real_draft_head_topk_candidate_enabled && (!p5aj_real_draft_head_logits_canary_enabled || !p5ag_topk_accept_boundary_enabled || !p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_candidate_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ag_topk_accept_boundary_enabled && (!p5af_topk_verify_mask_enabled || !p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_topk_accept_boundary_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5af_topk_verify_mask_enabled && (!p5ae_topk_tree_enabled || !p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_topk_verify_mask_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ae_topk_tree_enabled && (!p5x_root_tree_enabled || topk_abi_root_tail_conflict())) {
            disable_runtime_state(jetspec_runtime_failure::invalid_topk_tree_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ad_root_publish_gate_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled || !p5aa_root_token_commit_noop_enabled || !p5ab_root_hidden_kv_commit_noop_enabled || !p5ac_root_rejected_branch_discard_noop_enabled)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_publish_gate_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ac_root_rejected_branch_discard_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled || !p5aa_root_token_commit_noop_enabled || !p5ab_root_hidden_kv_commit_noop_enabled)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_rejected_branch_discard_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5ab_root_hidden_kv_commit_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled || !p5aa_root_token_commit_noop_enabled)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_hidden_kv_commit_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5aa_root_token_commit_noop_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled || !p5z_root_anchor_accept_path_enabled)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_token_commit_noop_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5z_root_anchor_accept_path_enabled && (!p5x_root_tree_enabled || !p5y_root_verify_mask_enabled)) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_anchor_accept_path_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5y_root_verify_mask_enabled && !p5x_root_tree_enabled) {
            disable_runtime_state(jetspec_runtime_failure::invalid_root_verify_mask_runtime, tap_count, tap_width, taps);
            return true;
        }
        if (p5x_root_tree_enabled) {
            if (!build_root_only_runtime_tree()) {
                disable_runtime_state(jetspec_runtime_failure::invalid_root_tree_runtime, tap_count, tap_width, taps);
                return true;
            }
            if (p5ae_topk_tree_enabled) {
                if (!build_topk_tree_runtime()) {
                    disable_runtime_state(jetspec_runtime_failure::invalid_topk_tree_runtime, tap_count, tap_width, taps);
                    return true;
                }
                if (p5af_topk_verify_mask_enabled) {
                    if (!build_topk_verify_mask_runtime()) {
                        disable_runtime_state(jetspec_runtime_failure::invalid_topk_verify_mask_runtime, tap_count, tap_width, taps);
                        return true;
                    }
                    if (p5ag_topk_accept_boundary_enabled) {
                        if (!build_topk_accept_boundary_runtime()) {
                            disable_runtime_state(jetspec_runtime_failure::invalid_topk_accept_boundary_runtime, tap_count, tap_width, taps);
                            return true;
                        }
                        if (p5ak_real_draft_head_topk_candidate_enabled) {
                            if (!build_real_draft_head_topk_candidate_runtime()) {
                                disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_candidate_runtime, tap_count, tap_width, taps);
                                return true;
                            }
                            if (p5al_real_draft_head_topk_tree_enabled) {
                                if (!build_real_draft_head_topk_tree_runtime()) {
                                    disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_tree_runtime, tap_count, tap_width, taps);
                                    return true;
                                }
                                if (p5am_real_draft_head_topk_verify_mask_enabled) {
                                    if (!build_real_draft_head_topk_verify_mask_runtime()) {
                                        disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_verify_mask_runtime, tap_count, tap_width, taps);
                                        return true;
                                    }
                                    if (p5an_real_draft_head_topk_accept_boundary_enabled) {
                                        if (!build_real_draft_head_topk_accept_boundary_runtime()) {
                                            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_accept_boundary_runtime, tap_count, tap_width, taps);
                                            return true;
                                        }
                                        if (p5ao_real_draft_head_topk_accept_path_descriptor_enabled) {
                                            if (!build_real_draft_head_topk_accept_path_descriptor_runtime()) {
                                                disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_accept_path_descriptor_runtime, tap_count, tap_width, taps);
                                                return true;
                                            }
                                            if (p5ap_real_draft_head_topk_token_commit_noop_enabled) {
                                                if (!build_real_draft_head_topk_token_commit_noop_runtime()) {
                                                    disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_token_commit_noop_runtime, tap_count, tap_width, taps);
                                                    return true;
                                                }
                                                if (p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled) {
                                                    if (!build_real_draft_head_topk_hidden_kv_commit_noop_runtime()) {
                                                        disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_hidden_kv_commit_noop_runtime, tap_count, tap_width, taps);
                                                        return true;
                                                    }
                                                    if (p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled) {
                                                        if (!build_real_draft_head_topk_rejected_branch_discard_noop_runtime()) {
                                                            disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_rejected_branch_discard_noop_runtime, tap_count, tap_width, taps);
                                                            return true;
                                                        }
                                                        if (p5as_real_draft_head_topk_publish_gate_noop_enabled) {
                                                            if (!build_real_draft_head_topk_publish_gate_noop_runtime()) {
                                                                disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_publish_gate_noop_runtime, tap_count, tap_width, taps);
                                                                return true;
                                                            }
                                                            if (trace_taps) {
                                                                LOG_INF("%s: draft-jetspec p5as_real_draft_head_topk_publish_gate_noop_runtime phase=%s real_topk_publish_gate_noop_runtime_ready=%d real_topk_publish_gate_noop_runtime_hash=%016" PRIx64 " real_topk_publish_gate_noop_seq_id=%d real_topk_publish_gate_noop_builds=%zu p5as_real_topk_publish_gate_noop_env=%d real_topk_rejected_branch_discard_noop_runtime_ready=%d real_topk_hidden_kv_commit_noop_runtime_ready=%d real_topk_token_commit_noop_runtime_ready=%d real_topk_accept_path_descriptor_runtime_ready=%d real_topk_accept_boundary_runtime_ready=%d real_topk_verify_mask_runtime_ready=%d real_topk_tree_runtime_ready=%d real_topk_candidate_runtime_ready=%d logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] candidate_nodes=%d candidate_ids=[%d,%d] rank_semantics=%s actual_verify_mask_entries=%d allowed_edges=[0:0,1:0,1:1,2:0,2:2] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_path_descriptor_len=%d actual_accepted_nodes=%d correction_token_present=%d accept_decision_source=%s token_commit_noop=1 hidden_kv_commit_noop=1 rejected_branch_discard_noop=1 reuse_p5ar_rejected_branch_discard_noop=1 publish_gate_noop=1 publish_after_commit_and_discard_only=1 actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d rejected_branch_pages_reachable_after_discard=%d actual_publish_visible_state=%d no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 no_visible_token_publish=1 no_real_hidden_kv_commit=1 no_hidden_kv_commit=1 no_real_rejected_branch_discard=1 no_rejected_branch_discard=1 no_real_publish=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                                        __func__, jetspec_runtime_phase_name(runtime_phase),
                                                                        real_draft_head_topk_publish_gate_noop_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_publish_gate_noop_hash_last,
                                                                        real_draft_head_topk_publish_gate_noop_seq_id_last,
                                                                        n_real_draft_head_topk_publish_gate_noop_runtime_builds,
                                                                        p5as_real_draft_head_topk_publish_gate_noop_enabled ? 1 : 0,
                                                                        real_draft_head_topk_rejected_branch_discard_noop_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_token_commit_noop_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_accept_path_descriptor_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                                        real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                                        JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                                        real_draft_head_canary_ctx_present_last,
                                                                        real_draft_head_canary_decode_rc_last,
                                                                        real_draft_head_canary_logits_rows_last,
                                                                        real_draft_head_canary_output_width_last,
                                                                        real_draft_head_topk_actual_verified_logits_rows_last,
                                                                        real_draft_head_canary_topk_k_last,
                                                                        real_draft_head_topk_tree_nodes_last,
                                                                        real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                                        real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
                                                                        real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                                        JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS,
                                                                        real_draft_head_topk_verify_mask_entries_last,
                                                                        real_draft_head_topk_accept_candidate_nodes_last,
                                                                        real_draft_head_topk_accept_boundary_verified_edges_last,
                                                                        real_draft_head_topk_accept_path_descriptor_len_last,
                                                                        real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                                        real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                                        JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS,
                                                                        actual_committed_tokens_last,
                                                                        actual_survivor_pages_committed_last,
                                                                        actual_pages_discarded_last,
                                                                        rejected_branch_pages_reachable_after_discard_last,
                                                                        actual_publish_visible_state_last);
                                                            }
                                                            if (p5av_real_draft_head_topk_target_logits_walk_canary_enabled) {
                                                                if (!build_real_draft_head_topk_target_logits_walk_canary_runtime(batch)) {
                                                                    disable_runtime_state(jetspec_runtime_failure::invalid_real_draft_head_topk_target_logits_walk_canary_runtime, tap_count, tap_width, taps);
                                                                    return true;
                                                                }
                                                                if (trace_taps) {
                                                                    LOG_INF("%s: draft-jetspec p5av_real_draft_head_topk_target_logits_walk_canary_runtime phase=%s target_logits_walk_canary_ready=%d target_logits_walk_canary_hash=%016" PRIx64 " target_logits_walk_canary_seq_id=%d target_logits_walk_canary_builds=%zu p5av_target_logits_walk_canary_env=%d real_topk_publish_gate_noop_runtime_ready=%d real_topk_publish_gate_noop_runtime_hash=%016" PRIx64 " planned_target_logits_rows=%d actual_target_logits_rows_walked=%d target_logits_source=%s target_logits_width=%d target_logits_batch_index=%d target_logits_pos=%d target_logits_seq_id=%d planned_parent_nodes=[0] planned_candidate_nodes=[1,2] candidate_ids=[%d,%d] target_candidate_logits=[%.6g,%.6g] row_semantics=%s accept_decision_source=%s actual_target_accept_steps=0 actual_accepted_nodes=%d correction_token_present=%d actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d rejected_branch_pages_reachable_after_discard=%d actual_publish_visible_state=%d target_logits_walk_canary_only=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 no_visible_token_publish=1 no_real_hidden_kv_commit=1 no_hidden_kv_commit=1 no_real_rejected_branch_discard=1 no_rejected_branch_discard=1 no_real_publish=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                                            __func__, jetspec_runtime_phase_name(runtime_phase),
                                                                            real_draft_head_topk_target_logits_walk_canary_ready ? 1 : 0,
                                                                            real_draft_head_topk_target_logits_walk_canary_hash_last,
                                                                            real_draft_head_topk_target_logits_walk_canary_seq_id_last,
                                                                            n_real_draft_head_topk_target_logits_walk_canary_builds,
                                                                            p5av_real_draft_head_topk_target_logits_walk_canary_enabled ? 1 : 0,
                                                                            real_draft_head_topk_publish_gate_noop_runtime_ready ? 1 : 0,
                                                                            real_draft_head_topk_publish_gate_noop_hash_last,
                                                                            real_draft_head_topk_target_logits_planned_rows_last,
                                                                            real_draft_head_topk_target_logits_actual_rows_walked_last,
                                                                            JETSPEC_TARGET_LOGITS_SOURCE,
                                                                            real_draft_head_topk_target_logits_width_last,
                                                                            real_draft_head_topk_target_logits_batch_index_last,
                                                                            real_draft_head_topk_target_logits_pos_last,
                                                                            real_draft_head_topk_target_logits_seq_id_last,
                                                                            real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                                            (double) real_draft_head_topk_target_logits_candidate_scores[0],
                                                                            (double) real_draft_head_topk_target_logits_candidate_scores[1],
                                                                            JETSPEC_TARGET_LOGITS_WALK_ROW_SEMANTICS,
                                                                            JETSPEC_ACCEPT_DECISION_SOURCE_TARGET_LOGITS_CANARY_ONLY,
                                                                            real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                                            real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                                            actual_committed_tokens_last,
                                                                            actual_survivor_pages_committed_last,
                                                                            actual_pages_discarded_last,
                                                                            rejected_branch_pages_reachable_after_discard_last,
                                                                            actual_publish_visible_state_last);
                                                                }
                                                            }
                                                            return true;
                                                        }
                                                        if (trace_taps) {
                                                            LOG_INF("%s: draft-jetspec p5ar_real_draft_head_topk_rejected_branch_discard_noop_runtime phase=%s real_topk_rejected_branch_discard_noop_runtime_ready=%d real_topk_rejected_branch_discard_noop_runtime_hash=%016" PRIx64 " real_topk_rejected_branch_discard_noop_seq_id=%d real_topk_rejected_branch_discard_noop_builds=%zu p5ar_real_topk_rejected_branch_discard_noop_env=%d real_topk_hidden_kv_commit_noop_runtime_ready=%d real_topk_token_commit_noop_runtime_ready=%d real_topk_accept_path_descriptor_runtime_ready=%d real_topk_accept_boundary_runtime_ready=%d real_topk_verify_mask_runtime_ready=%d real_topk_tree_runtime_ready=%d real_topk_candidate_runtime_ready=%d logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] candidate_nodes=%d candidate_ids=[%d,%d] rank_semantics=%s actual_verify_mask_entries=%d allowed_edges=[0:0,1:0,1:1,2:0,2:2] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_path_descriptor_len=%d actual_accepted_nodes=%d correction_token_present=%d accept_decision_source=%s token_commit_noop=1 hidden_kv_commit_noop=1 reuse_p5aq_hidden_kv_commit_noop=1 rejected_branch_discard_noop=1 actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d rejected_branch_pages_reachable_after_discard=%d actual_publish_visible_state=%d no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 no_visible_token_publish=1 no_real_hidden_kv_commit=1 no_hidden_kv_commit=1 no_real_rejected_branch_discard=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                                    __func__, jetspec_runtime_phase_name(runtime_phase),
                                                                    real_draft_head_topk_rejected_branch_discard_noop_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_rejected_branch_discard_noop_hash_last,
                                                                    real_draft_head_topk_rejected_branch_discard_noop_seq_id_last,
                                                                    n_real_draft_head_topk_rejected_branch_discard_noop_runtime_builds,
                                                                    p5ar_real_draft_head_topk_rejected_branch_discard_noop_enabled ? 1 : 0,
                                                                    real_draft_head_topk_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_token_commit_noop_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_accept_path_descriptor_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                                    real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                                    JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                                    real_draft_head_canary_ctx_present_last,
                                                                    real_draft_head_canary_decode_rc_last,
                                                                    real_draft_head_canary_logits_rows_last,
                                                                    real_draft_head_canary_output_width_last,
                                                                    real_draft_head_topk_actual_verified_logits_rows_last,
                                                                    real_draft_head_canary_topk_k_last,
                                                                    real_draft_head_topk_tree_nodes_last,
                                                                    real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                                    real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
                                                                    real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                                    JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS,
                                                                    real_draft_head_topk_verify_mask_entries_last,
                                                                    real_draft_head_topk_accept_candidate_nodes_last,
                                                                    real_draft_head_topk_accept_boundary_verified_edges_last,
                                                                    real_draft_head_topk_accept_path_descriptor_len_last,
                                                                    real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                                    real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                                    JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS,
                                                                    actual_committed_tokens_last,
                                                                    actual_survivor_pages_committed_last,
                                                                    actual_pages_discarded_last,
                                                                    rejected_branch_pages_reachable_after_discard_last,
                                                                    actual_publish_visible_state_last);
                                                        }
                                                        return true;
                                                    }
                                                    if (trace_taps) {
                                                        LOG_INF("%s: draft-jetspec p5aq_real_draft_head_topk_hidden_kv_commit_noop_runtime phase=%s real_topk_hidden_kv_commit_noop_runtime_ready=%d real_topk_hidden_kv_commit_noop_runtime_hash=%016" PRIx64 " real_topk_hidden_kv_commit_noop_seq_id=%d real_topk_hidden_kv_commit_noop_builds=%zu p5aq_real_topk_hidden_kv_commit_noop_env=%d real_topk_token_commit_noop_runtime_ready=%d real_topk_accept_path_descriptor_runtime_ready=%d real_topk_accept_boundary_runtime_ready=%d real_topk_verify_mask_runtime_ready=%d real_topk_tree_runtime_ready=%d real_topk_candidate_runtime_ready=%d logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] candidate_nodes=%d candidate_ids=[%d,%d] rank_semantics=%s actual_verify_mask_entries=%d allowed_edges=[0:0,1:0,1:1,2:0,2:2] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_path_descriptor_len=%d actual_accepted_nodes=%d correction_token_present=%d accept_decision_source=%s token_commit_noop=1 reuse_p5ap_token_commit_noop=1 hidden_kv_commit_noop=1 actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d actual_publish_visible_state=%d no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 no_visible_token_publish=1 no_real_hidden_kv_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                                __func__, jetspec_runtime_phase_name(runtime_phase),
                                                                real_draft_head_topk_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_hidden_kv_commit_noop_hash_last,
                                                                real_draft_head_topk_hidden_kv_commit_noop_seq_id_last,
                                                                n_real_draft_head_topk_hidden_kv_commit_noop_runtime_builds,
                                                                p5aq_real_draft_head_topk_hidden_kv_commit_noop_enabled ? 1 : 0,
                                                                real_draft_head_topk_token_commit_noop_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_accept_path_descriptor_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                                real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                                JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                                real_draft_head_canary_ctx_present_last,
                                                                real_draft_head_canary_decode_rc_last,
                                                                real_draft_head_canary_logits_rows_last,
                                                                real_draft_head_canary_output_width_last,
                                                                real_draft_head_topk_actual_verified_logits_rows_last,
                                                                real_draft_head_canary_topk_k_last,
                                                                real_draft_head_topk_tree_nodes_last,
                                                                real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                                real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
                                                                real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                                JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS,
                                                                real_draft_head_topk_verify_mask_entries_last,
                                                                real_draft_head_topk_accept_candidate_nodes_last,
                                                                real_draft_head_topk_accept_boundary_verified_edges_last,
                                                                real_draft_head_topk_accept_path_descriptor_len_last,
                                                                real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                                real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                                JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS,
                                                                actual_committed_tokens_last,
                                                                actual_survivor_pages_committed_last,
                                                                actual_pages_discarded_last,
                                                                actual_publish_visible_state_last);
                                                    }
                                                    return true;
                                                }
                                                if (trace_taps) {
                                                    LOG_INF("%s: draft-jetspec p5ap_real_draft_head_topk_token_commit_noop_runtime phase=%s real_topk_token_commit_noop_runtime_ready=%d real_topk_token_commit_noop_runtime_hash=%016" PRIx64 " real_topk_token_commit_noop_seq_id=%d real_topk_token_commit_noop_builds=%zu p5ap_real_topk_token_commit_noop_env=%d real_topk_accept_path_descriptor_runtime_ready=%d real_topk_accept_boundary_runtime_ready=%d real_topk_verify_mask_runtime_ready=%d real_topk_tree_runtime_ready=%d real_topk_candidate_runtime_ready=%d logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] candidate_nodes=%d candidate_ids=[%d,%d] rank_semantics=%s actual_verify_mask_entries=%d allowed_edges=[0:0,1:0,1:1,2:0,2:2] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_path_descriptor_len=%d actual_accepted_nodes=%d correction_token_present=%d accept_decision_source=%s token_commit_noop=1 reuse_p5ao_accept_path_descriptor=1 actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d actual_publish_visible_state=%d no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_real_token_commit=1 no_visible_token_publish=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                            __func__, jetspec_runtime_phase_name(runtime_phase),
                                                            real_draft_head_topk_token_commit_noop_runtime_ready ? 1 : 0,
                                                            real_draft_head_topk_token_commit_noop_hash_last,
                                                            real_draft_head_topk_token_commit_noop_seq_id_last,
                                                            n_real_draft_head_topk_token_commit_noop_runtime_builds,
                                                            p5ap_real_draft_head_topk_token_commit_noop_enabled ? 1 : 0,
                                                            real_draft_head_topk_accept_path_descriptor_runtime_ready ? 1 : 0,
                                                            real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                            real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                            real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                            real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                            JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                            real_draft_head_canary_ctx_present_last,
                                                            real_draft_head_canary_decode_rc_last,
                                                            real_draft_head_canary_logits_rows_last,
                                                            real_draft_head_canary_output_width_last,
                                                            real_draft_head_topk_actual_verified_logits_rows_last,
                                                            real_draft_head_canary_topk_k_last,
                                                            real_draft_head_topk_tree_nodes_last,
                                                            real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                            real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
                                                            real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                            JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS,
                                                            real_draft_head_topk_verify_mask_entries_last,
                                                            real_draft_head_topk_accept_candidate_nodes_last,
                                                            real_draft_head_topk_accept_boundary_verified_edges_last,
                                                            real_draft_head_topk_accept_path_descriptor_len_last,
                                                            real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                            real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                            JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS,
                                                            actual_committed_tokens_last,
                                                            actual_survivor_pages_committed_last,
                                                            actual_pages_discarded_last,
                                                            actual_publish_visible_state_last);
                                                }
                                                return true;
                                            }
                                            if (trace_taps) {
                                                LOG_INF("%s: draft-jetspec p5ao_real_draft_head_topk_accept_path_descriptor_runtime phase=%s real_topk_accept_path_descriptor_runtime_ready=%d real_topk_accept_path_descriptor_runtime_hash=%016" PRIx64 " real_topk_accept_path_descriptor_seq_id=%d real_topk_accept_path_descriptor_builds=%zu p5ao_real_topk_accept_path_descriptor_env=%d real_topk_accept_boundary_runtime_ready=%d real_topk_accept_boundary_runtime_hash=%016" PRIx64 " real_topk_verify_mask_runtime_ready=%d real_topk_verify_mask_runtime_hash=%016" PRIx64 " real_topk_tree_runtime_ready=%d real_topk_tree_runtime_hash=%016" PRIx64 " real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] candidate_nodes=%d candidate_ids=[%d,%d] rank_semantics=%s actual_verify_mask_entries=%d allowed_edges=[0:0,1:0,1:1,2:0,2:2] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_path_descriptor_len=%d actual_accepted_nodes=%d correction_token_present=%d accept_decision_source=%s descriptor_only=1 reuse_p5an_accept_boundary_metadata=1 actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d actual_publish_visible_state=%d no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                        __func__, jetspec_runtime_phase_name(runtime_phase),
                                                        real_draft_head_topk_accept_path_descriptor_runtime_ready ? 1 : 0,
                                                        real_draft_head_topk_accept_path_descriptor_hash_last,
                                                        real_draft_head_topk_accept_path_descriptor_seq_id_last,
                                                        n_real_draft_head_topk_accept_path_descriptor_runtime_builds,
                                                        p5ao_real_draft_head_topk_accept_path_descriptor_enabled ? 1 : 0,
                                                        real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                        real_draft_head_topk_accept_boundary_hash_last,
                                                        real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                        real_draft_head_topk_verify_mask_hash_last,
                                                        real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                        real_draft_head_topk_tree_hash_last,
                                                        real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                        real_draft_head_topk_candidate_hash_last,
                                                        topk_accept_boundary_runtime_ready ? 1 : 0,
                                                        topk_accept_boundary_runtime_hash_last,
                                                        topk_verify_mask_runtime_ready ? 1 : 0,
                                                        topk_verify_mask_runtime_hash_last,
                                                        topk_tree_runtime_ready ? 1 : 0,
                                                        topk_tree_runtime_hash_last,
                                                        JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                        real_draft_head_canary_ctx_present_last,
                                                        real_draft_head_canary_decode_rc_last,
                                                        real_draft_head_canary_logits_rows_last,
                                                        real_draft_head_canary_output_width_last,
                                                        real_draft_head_topk_actual_verified_logits_rows_last,
                                                        real_draft_head_canary_topk_k_last,
                                                        real_draft_head_topk_tree_nodes_last,
                                                        real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                        real_draft_head_topk_accept_path_descriptor_candidate_nodes_last,
                                                        real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                        JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS,
                                                        real_draft_head_topk_verify_mask_entries_last,
                                                        real_draft_head_topk_accept_candidate_nodes_last,
                                                        real_draft_head_topk_accept_boundary_verified_edges_last,
                                                        real_draft_head_topk_accept_path_descriptor_len_last,
                                                        real_draft_head_topk_accept_path_descriptor_actual_accepted_nodes_last,
                                                        real_draft_head_topk_accept_path_descriptor_correction_token_present_last,
                                                        JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS,
                                                        actual_committed_tokens_last,
                                                        actual_survivor_pages_committed_last,
                                                        actual_pages_discarded_last,
                                                        actual_publish_visible_state_last);
                                            }
                                            return true;
                                        }
                                        if (trace_taps) {
                                            LOG_INF("%s: draft-jetspec p5an_real_draft_head_topk_accept_boundary_runtime phase=%s real_topk_accept_boundary_runtime_ready=%d real_topk_accept_boundary_runtime_hash=%016" PRIx64 " real_topk_accept_boundary_seq_id=%d real_topk_verify_mask_runtime_ready=%d real_topk_verify_mask_runtime_hash=%016" PRIx64 " real_topk_tree_runtime_ready=%d real_topk_tree_runtime_hash=%016" PRIx64 " real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] real_tree_parent_indices=[%d,%d,%d] real_tree_depth=[%d,%d,%d] real_tree_rank=[%d,%d,%d] real_tree_logits=[%.6g,%.6g,%.6g] parent_node=%d candidate_nodes=%d candidate_ids=[%d,%d] candidate_logits=[%.6g,%.6g] rank_semantics=%s actual_verify_mask_entries=%d verify_mask_rows=3 verify_mask_cols=3 allowed_edges=[0:0,1:0,1:1,2:0,2:2] real_verify_mask_rows=[%d,%d,%d,%d,%d] real_verify_mask_cols=[%d,%d,%d,%d,%d] real_verify_mask_values=[%d,%d,%d,%d,%d] accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d accept_decision_source=%s accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d actual_publish_visible_state=%d reuse_p5am_real_tree_mask_metadata=1 no_target_logits_walk=1 no_target_accept_walk=1 no_sampler=1 no_accept=1 no_mask_tensor=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                    __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_topk_accept_boundary_runtime_ready ? 1 : 0,
                                                    real_draft_head_topk_accept_boundary_hash_last, real_draft_head_topk_accept_boundary_seq_id_last,
                                                    real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0, real_draft_head_topk_verify_mask_hash_last,
                                                    real_draft_head_topk_tree_runtime_ready ? 1 : 0, real_draft_head_topk_tree_hash_last,
                                                    real_draft_head_topk_candidate_runtime_ready ? 1 : 0, real_draft_head_topk_candidate_hash_last,
                                                    topk_accept_boundary_runtime_ready ? 1 : 0, topk_accept_boundary_runtime_hash_last,
                                                    topk_verify_mask_runtime_ready ? 1 : 0, topk_verify_mask_runtime_hash_last,
                                                    topk_tree_runtime_ready ? 1 : 0, topk_tree_runtime_hash_last,
                                                    JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE, real_draft_head_canary_ctx_present_last,
                                                    real_draft_head_canary_decode_rc_last, real_draft_head_canary_logits_rows_last,
                                                    real_draft_head_canary_output_width_last, real_draft_head_topk_actual_verified_logits_rows_last,
                                                    real_draft_head_canary_topk_k_last, real_draft_head_topk_tree_nodes_last,
                                                    real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                    real_draft_head_topk_tree_parent_indices[0], real_draft_head_topk_tree_parent_indices[1], real_draft_head_topk_tree_parent_indices[2],
                                                    real_draft_head_topk_tree_depth[0], real_draft_head_topk_tree_depth[1], real_draft_head_topk_tree_depth[2],
                                                    real_draft_head_topk_tree_rank[0], real_draft_head_topk_tree_rank[1], real_draft_head_topk_tree_rank[2],
                                                    (double) real_draft_head_topk_tree_cum_logit[0], (double) real_draft_head_topk_tree_cum_logit[1], (double) real_draft_head_topk_tree_cum_logit[2],
                                                    real_draft_head_topk_parent_node_last, real_draft_head_topk_accept_candidate_nodes_last,
                                                    real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                    (double) real_draft_head_topk_candidate_logits[0], (double) real_draft_head_topk_candidate_logits[1],
                                                    JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, real_draft_head_topk_verify_mask_entries_last,
                                                    real_draft_head_topk_verify_mask_rows[0], real_draft_head_topk_verify_mask_rows[1], real_draft_head_topk_verify_mask_rows[2], real_draft_head_topk_verify_mask_rows[3], real_draft_head_topk_verify_mask_rows[4],
                                                    real_draft_head_topk_verify_mask_cols[0], real_draft_head_topk_verify_mask_cols[1], real_draft_head_topk_verify_mask_cols[2], real_draft_head_topk_verify_mask_cols[3], real_draft_head_topk_verify_mask_cols[4],
                                                    (int) real_draft_head_topk_verify_mask_values[0], (int) real_draft_head_topk_verify_mask_values[1], (int) real_draft_head_topk_verify_mask_values[2], (int) real_draft_head_topk_verify_mask_values[3], (int) real_draft_head_topk_verify_mask_values[4],
                                                    real_draft_head_topk_accept_candidate_nodes_last, real_draft_head_topk_accept_boundary_verified_edges_last,
                                                    JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_TARGET_LOGITS, real_draft_head_topk_accept_path_len_last,
                                                    real_draft_head_topk_actual_accepted_nodes_last, real_draft_head_topk_correction_token_present_last,
                                                    actual_committed_tokens_last, actual_survivor_pages_committed_last, actual_pages_discarded_last,
                                                    actual_publish_visible_state_last);
                                        }
                                        return true;
                                    }
                                    if (trace_taps) {
                                        LOG_INF("%s: draft-jetspec p5am_real_draft_head_topk_verify_mask_runtime phase=%s real_topk_verify_mask_runtime_ready=%d real_topk_verify_mask_runtime_hash=%016" PRIx64 " real_topk_tree_runtime_ready=%d real_topk_tree_runtime_hash=%016" PRIx64 " real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] real_tree_parent_indices=[%d,%d,%d] real_tree_depth=[%d,%d,%d] real_tree_rank=[%d,%d,%d] real_tree_logits=[%.6g,%.6g,%.6g] parent_node=%d candidate_nodes=%d candidate_ids=[%d,%d] candidate_logits=[%.6g,%.6g] rank_semantics=%s accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d actual_verify_mask_entries=%d verify_mask_rows=3 verify_mask_cols=3 allowed_edges=[0:0,1:0,1:1,2:0,2:2] real_verify_mask_rows=[%d,%d,%d,%d,%d] real_verify_mask_cols=[%d,%d,%d,%d,%d] real_verify_mask_values=[%d,%d,%d,%d,%d] prefix_visible=1 ancestor_only=1 root_attends_self=1 child_attends_root=1 child_attends_self=1 sibling_visible=0 descendant_visible=0 no_mask_tensor=1 no_synthetic_mask_mutation=1 no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                                __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_topk_verify_mask_runtime_ready ? 1 : 0,
                                                real_draft_head_topk_verify_mask_hash_last, real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                                real_draft_head_topk_tree_hash_last, real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                                real_draft_head_topk_candidate_hash_last, topk_accept_boundary_runtime_ready ? 1 : 0,
                                                topk_accept_boundary_runtime_hash_last, topk_verify_mask_runtime_ready ? 1 : 0,
                                                topk_verify_mask_runtime_hash_last, topk_tree_runtime_ready ? 1 : 0,
                                                topk_tree_runtime_hash_last, JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                                real_draft_head_canary_ctx_present_last, real_draft_head_canary_decode_rc_last,
                                                real_draft_head_canary_logits_rows_last, real_draft_head_canary_output_width_last,
                                                real_draft_head_topk_verified_logits_rows_last, real_draft_head_canary_topk_k_last,
                                                real_draft_head_topk_tree_nodes_last,
                                                real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                                real_draft_head_topk_tree_parent_indices[0], real_draft_head_topk_tree_parent_indices[1], real_draft_head_topk_tree_parent_indices[2],
                                                real_draft_head_topk_tree_depth[0], real_draft_head_topk_tree_depth[1], real_draft_head_topk_tree_depth[2],
                                                real_draft_head_topk_tree_rank[0], real_draft_head_topk_tree_rank[1], real_draft_head_topk_tree_rank[2],
                                                (double) real_draft_head_topk_tree_cum_logit[0], (double) real_draft_head_topk_tree_cum_logit[1], (double) real_draft_head_topk_tree_cum_logit[2],
                                                real_draft_head_topk_parent_node_last, real_draft_head_topk_candidate_nodes_last,
                                                real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                                (double) real_draft_head_topk_candidate_logits[0], (double) real_draft_head_topk_candidate_logits[1],
                                                JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, accept_path_len_last,
                                                actual_accepted_nodes_last, correction_token_present_last, real_draft_head_topk_verify_mask_entries_last,
                                                real_draft_head_topk_verify_mask_rows[0], real_draft_head_topk_verify_mask_rows[1], real_draft_head_topk_verify_mask_rows[2], real_draft_head_topk_verify_mask_rows[3], real_draft_head_topk_verify_mask_rows[4],
                                                real_draft_head_topk_verify_mask_cols[0], real_draft_head_topk_verify_mask_cols[1], real_draft_head_topk_verify_mask_cols[2], real_draft_head_topk_verify_mask_cols[3], real_draft_head_topk_verify_mask_cols[4],
                                                (int) real_draft_head_topk_verify_mask_values[0], (int) real_draft_head_topk_verify_mask_values[1], (int) real_draft_head_topk_verify_mask_values[2], (int) real_draft_head_topk_verify_mask_values[3], (int) real_draft_head_topk_verify_mask_values[4]);
                                    }
                                    return true;
                                }
                                if (trace_taps) {
                                    LOG_INF("%s: draft-jetspec p5al_real_draft_head_topk_tree_runtime phase=%s real_topk_tree_runtime_ready=%d real_topk_tree_runtime_hash=%016" PRIx64 " real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d actual_tree_nodes=%d real_tree_token_ids=[%d,%d,%d] real_tree_parent_indices=[%d,%d,%d] real_tree_depth=[%d,%d,%d] real_tree_rank=[%d,%d,%d] real_tree_logits=[%.6g,%.6g,%.6g] parent_node=%d candidate_nodes=%d candidate_ids=[%d,%d] candidate_logits=[%.6g,%.6g] rank_semantics=%s accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d no_synthetic_token_ids=1 no_external_logits_walk=1 no_target_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                            __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_topk_tree_runtime_ready ? 1 : 0,
                                            real_draft_head_topk_tree_hash_last, real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                            real_draft_head_topk_candidate_hash_last, topk_accept_boundary_runtime_ready ? 1 : 0,
                                            topk_accept_boundary_runtime_hash_last, topk_verify_mask_runtime_ready ? 1 : 0, topk_verify_mask_runtime_hash_last,
                                            topk_tree_runtime_ready ? 1 : 0, topk_tree_runtime_hash_last, JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                            real_draft_head_canary_ctx_present_last, real_draft_head_canary_decode_rc_last,
                                            real_draft_head_canary_logits_rows_last, real_draft_head_canary_output_width_last,
                                            real_draft_head_topk_verified_logits_rows_last, real_draft_head_canary_topk_k_last,
                                            real_draft_head_topk_tree_nodes_last,
                                            real_draft_head_topk_tree_token_ids[0], real_draft_head_topk_tree_token_ids[1], real_draft_head_topk_tree_token_ids[2],
                                            real_draft_head_topk_tree_parent_indices[0], real_draft_head_topk_tree_parent_indices[1], real_draft_head_topk_tree_parent_indices[2],
                                            real_draft_head_topk_tree_depth[0], real_draft_head_topk_tree_depth[1], real_draft_head_topk_tree_depth[2],
                                            real_draft_head_topk_tree_rank[0], real_draft_head_topk_tree_rank[1], real_draft_head_topk_tree_rank[2],
                                            (double) real_draft_head_topk_tree_cum_logit[0], (double) real_draft_head_topk_tree_cum_logit[1], (double) real_draft_head_topk_tree_cum_logit[2],
                                            real_draft_head_topk_parent_node_last, real_draft_head_topk_candidate_nodes_last,
                                            real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                            (double) real_draft_head_topk_candidate_logits[0], (double) real_draft_head_topk_candidate_logits[1],
                                            JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, accept_path_len_last,
                                            actual_accepted_nodes_last, correction_token_present_last);
                                }
                                return true;
                            }
                            if (trace_taps) {
                                LOG_INF("%s: draft-jetspec p5ak_real_draft_head_topk_candidate_runtime phase=%s real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " logits_source=%s ctx_dft_present=%d decode_rc=%d logits_rows=%d logits_width=%d actual_verified_logits_rows=%d topk_k=%d parent_node=%d candidate_nodes=%d candidate_ids=[%d,%d] candidate_logits=[%.6g,%.6g] rank_semantics=%s accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d no_external_logits_walk=1 no_target_accept_walk=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                                        __func__, jetspec_runtime_phase_name(runtime_phase), real_draft_head_topk_candidate_runtime_ready ? 1 : 0,
                                        real_draft_head_topk_candidate_hash_last, topk_accept_boundary_runtime_ready ? 1 : 0,
                                        topk_accept_boundary_runtime_hash_last, topk_verify_mask_runtime_ready ? 1 : 0, topk_verify_mask_runtime_hash_last,
                                        topk_tree_runtime_ready ? 1 : 0, topk_tree_runtime_hash_last, JETSPEC_REAL_DRAFT_HEAD_TOPK_LOGITS_SOURCE,
                                        real_draft_head_canary_ctx_present_last, real_draft_head_canary_decode_rc_last,
                                        real_draft_head_canary_logits_rows_last, real_draft_head_canary_output_width_last,
                                        real_draft_head_topk_verified_logits_rows_last, real_draft_head_canary_topk_k_last,
                                        real_draft_head_topk_parent_node_last, real_draft_head_topk_candidate_nodes_last,
                                        real_draft_head_topk_candidate_ids[0], real_draft_head_topk_candidate_ids[1],
                                        (double) real_draft_head_topk_candidate_logits[0], (double) real_draft_head_topk_candidate_logits[1],
                                        JETSPEC_REAL_DRAFT_HEAD_TOPK_RANK_SEMANTICS, accept_path_len_last,
                                        actual_accepted_nodes_last, correction_token_present_last);
                            }
                            return true;
                        }
                        if (trace_taps) {
                            LOG_INF("%s: draft-jetspec p5ag_topk_accept_boundary_runtime phase=%s topk_accept_boundary_runtime_ready=%d topk_accept_boundary_runtime_hash=%016" PRIx64 " topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " actual_tree_nodes=%d actual_verify_mask_entries=%d accept_boundary_candidate_nodes=%d accept_boundary_verified_edges=%d actual_verified_logits_rows=%d accept_decision_source=%s accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d no_target_logits_walk=1 no_target_accept_walk=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                    __func__, jetspec_runtime_phase_name(runtime_phase), topk_accept_boundary_runtime_ready ? 1 : 0,
                                    topk_accept_boundary_runtime_hash_last, topk_verify_mask_runtime_ready ? 1 : 0, topk_verify_mask_runtime_hash_last,
                                    topk_tree_runtime_ready ? 1 : 0, topk_tree_runtime_hash_last, tree_build_actual_nodes_last,
                                    actual_verify_mask_entries_last, topk_accept_candidate_nodes_last, topk_accept_boundary_verified_edges_last,
                                    topk_actual_verified_logits_rows_last, JETSPEC_ACCEPT_DECISION_SOURCE_NONE_NO_LOGITS, accept_path_len_last,
                                    actual_accepted_nodes_last, correction_token_present_last);
                        }
                        return true;
                    }
                    if (trace_taps) {
                        LOG_INF("%s: draft-jetspec p5af_topk_verify_mask_runtime phase=%s topk_verify_mask_runtime_ready=%d topk_verify_mask_runtime_hash=%016" PRIx64 " topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " actual_tree_nodes=%d actual_verify_mask_entries=%d verify_mask_rows=3 verify_mask_cols=3 allowed_edges=[0:0,1:0,1:1,2:0,2:2] prefix_visible=1 ancestor_only=1 root_attends_self=1 child_attends_root=1 child_attends_self=1 sibling_visible=0 descendant_visible=0 no_mask_tensor=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                __func__, jetspec_runtime_phase_name(runtime_phase), topk_verify_mask_runtime_ready ? 1 : 0,
                                topk_verify_mask_runtime_hash_last, topk_tree_runtime_ready ? 1 : 0, topk_tree_runtime_hash_last,
                                tree_build_actual_nodes_last, actual_verify_mask_entries_last);
                    }
                    return true;
                }
                if (trace_taps) {
                    LOG_INF("%s: draft-jetspec p5ae_topk_tree_runtime phase=%s topk_tree_runtime_ready=%d topk_tree_runtime_hash=%016" PRIx64 " topk_logprob_source=%s topk_width=%d topk_depth=%d actual_tree_nodes=%d tree_token_ids=[%d,%d,%d] tree_parent_indices=[%d,%d,%d] tree_depth=[%d,%d,%d] tree_rank=[%d,%d,%d] tree_cum_logprob=[%.1f,%.1f,%.1f] parent_before_child=1 num_nodes_lte_budget=1 non_root_nodes=%d no_draft_head_graph=1 no_draft_logits=1 no_verify_mask=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                            __func__, jetspec_runtime_phase_name(runtime_phase), topk_tree_runtime_ready ? 1 : 0,
                            topk_tree_runtime_hash_last, JETSPEC_SYNTHETIC_FULL_VOCAB_SOFTMAX, topk_tree_width_last,
                            topk_tree_depth_last, tree_build_actual_nodes_last, tree_token_ids[0], tree_token_ids[1], tree_token_ids[2],
                            tree_parent_indices[0], tree_parent_indices[1], tree_parent_indices[2], tree_depth[0], tree_depth[1], tree_depth[2],
                            tree_rank[0], tree_rank[1], tree_rank[2], (double) tree_cum_logprob[0], (double) tree_cum_logprob[1],
                            (double) tree_cum_logprob[2], topk_tree_non_root_nodes_last);
                }
                return true;
            }
            if (p5y_root_verify_mask_enabled) {
                if (!build_root_only_verify_mask_runtime()) {
                    disable_runtime_state(jetspec_runtime_failure::invalid_root_verify_mask_runtime, tap_count, tap_width, taps);
                    return true;
                }
                if (p5z_root_anchor_accept_path_enabled) {
                    if (!build_root_anchor_accept_path_runtime()) {
                        disable_runtime_state(jetspec_runtime_failure::invalid_root_anchor_accept_path_runtime, tap_count, tap_width, taps);
                        return true;
                    }
                    if (p5aa_root_token_commit_noop_enabled) {
                        if (!build_root_token_commit_noop_runtime()) {
                            disable_runtime_state(jetspec_runtime_failure::invalid_root_token_commit_noop_runtime, tap_count, tap_width, taps);
                            return true;
                        }
                        if (p5ab_root_hidden_kv_commit_noop_enabled) {
                            if (!build_root_hidden_kv_commit_noop_runtime()) {
                                disable_runtime_state(jetspec_runtime_failure::invalid_root_hidden_kv_commit_noop_runtime, tap_count, tap_width, taps);
                                return true;
                            }
                            if (p5ac_root_rejected_branch_discard_noop_enabled) {
                                if (!build_root_rejected_branch_discard_noop_runtime()) {
                                    disable_runtime_state(jetspec_runtime_failure::invalid_root_rejected_branch_discard_noop_runtime, tap_count, tap_width, taps);
                                    return true;
                                }
                                if (p5ad_root_publish_gate_noop_enabled) {
                                    if (!build_root_publish_gate_noop_runtime()) {
                                        disable_runtime_state(jetspec_runtime_failure::invalid_root_publish_gate_noop_runtime, tap_count, tap_width, taps);
                                        return true;
                                    }
                                    if (trace_taps) {
                                        LOG_INF("%s: draft-jetspec p5ad_root_publish_gate_noop_runtime phase=%s root_publish_gate_noop_runtime_ready=%d root_publish_gate_noop_runtime_hash=%016" PRIx64 " root_rejected_branch_discard_noop_runtime_ready=%d root_rejected_branch_discard_noop_runtime_hash=%016" PRIx64 " root_hidden_kv_commit_noop_runtime_ready=%d root_hidden_kv_commit_noop_runtime_hash=%016" PRIx64 " root_token_commit_noop_runtime_ready=%d root_token_commit_noop_runtime_hash=%016" PRIx64 " root_runtime_ready_for_real_test=%d actual_committed_tokens=%d actual_survivor_pages_committed=%d actual_pages_discarded=%d rejected_branch_pages_reachable_after_discard=%d actual_publish_visible_state=%d publish_after_commit_and_discard_only=1 no_real_token_commit=1 no_real_hidden_kv_commit=1 no_real_rejected_branch_discard=1 no_real_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                                __func__, jetspec_runtime_phase_name(runtime_phase), root_publish_gate_noop_runtime_ready ? 1 : 0,
                                                root_publish_gate_noop_runtime_hash_last, root_rejected_branch_discard_noop_runtime_ready ? 1 : 0,
                                                root_rejected_branch_discard_noop_runtime_hash_last, root_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                                root_hidden_kv_commit_noop_runtime_hash_last, root_token_commit_noop_runtime_ready ? 1 : 0,
                                                root_token_commit_noop_runtime_hash_last, root_runtime_ready_for_real_test_last, actual_committed_tokens_last,
                                                actual_survivor_pages_committed_last, actual_pages_discarded_last,
                                                rejected_branch_pages_reachable_after_discard_last, actual_publish_visible_state_last);
                                    }
                                    return true;
                                }
                                if (trace_taps) {
                                    LOG_INF("%s: draft-jetspec p5ac_root_rejected_branch_discard_noop_runtime phase=%s root_rejected_branch_discard_noop_runtime_ready=%d root_rejected_branch_discard_noop_runtime_hash=%016" PRIx64 " root_hidden_kv_commit_noop_runtime_ready=%d root_hidden_kv_commit_noop_runtime_hash=%016" PRIx64 " actual_survivor_pages_committed=%d actual_pages_discarded=%d rejected_branch_pages_reachable_after_discard=%d no_real_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                            __func__, jetspec_runtime_phase_name(runtime_phase), root_rejected_branch_discard_noop_runtime_ready ? 1 : 0,
                                            root_rejected_branch_discard_noop_runtime_hash_last, root_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                            root_hidden_kv_commit_noop_runtime_hash_last, actual_survivor_pages_committed_last,
                                            actual_pages_discarded_last, rejected_branch_pages_reachable_after_discard_last);
                                }
                                return true;
                            }
                            if (trace_taps) {
                                LOG_INF("%s: draft-jetspec p5ab_root_hidden_kv_commit_noop_runtime phase=%s root_hidden_kv_commit_noop_runtime_ready=%d root_hidden_kv_commit_noop_runtime_hash=%016" PRIx64 " root_token_commit_noop_runtime_ready=%d root_token_commit_noop_runtime_hash=%016" PRIx64 " actual_committed_tokens=%d actual_survivor_pages_committed=%d no_real_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                        __func__, jetspec_runtime_phase_name(runtime_phase), root_hidden_kv_commit_noop_runtime_ready ? 1 : 0,
                                        root_hidden_kv_commit_noop_runtime_hash_last, root_token_commit_noop_runtime_ready ? 1 : 0,
                                        root_token_commit_noop_runtime_hash_last, actual_committed_tokens_last, actual_survivor_pages_committed_last);
                            }
                            return true;
                        }
                        if (trace_taps) {
                            LOG_INF("%s: draft-jetspec p5aa_root_token_commit_noop_runtime phase=%s root_token_commit_noop_runtime_ready=%d root_token_commit_noop_runtime_hash=%016" PRIx64 " root_anchor_accept_path_runtime_ready=%d root_anchor_accept_path_runtime_hash=%016" PRIx64 " root_verify_mask_runtime_ready=%d root_verify_mask_runtime_hash=%016" PRIx64 " root_tree_runtime_ready=%d root_tree_runtime_hash=%016" PRIx64 " root_verified_anchor=%d accept_path_len=%d actual_tree_nodes=%d actual_verify_mask_entries=%d actual_accepted_nodes=%d correction_token_present=%d actual_committed_tokens=%d no_real_token_commit=1 no_visible_token_publish=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                    __func__, jetspec_runtime_phase_name(runtime_phase), root_token_commit_noop_runtime_ready ? 1 : 0,
                                    root_token_commit_noop_runtime_hash_last, root_anchor_accept_path_runtime_ready ? 1 : 0,
                                    root_anchor_accept_path_runtime_hash_last, root_verify_mask_runtime_ready ? 1 : 0,
                                    root_verify_mask_runtime_hash_last, root_tree_runtime_ready ? 1 : 0, root_tree_runtime_hash_last,
                                    root_verified_anchor_last, accept_path_len_last, tree_build_actual_nodes_last, actual_verify_mask_entries_last,
                                    actual_accepted_nodes_last, correction_token_present_last, actual_committed_tokens_last);
                        }
                        return true;
                    }
                    if (trace_taps) {
                        LOG_INF("%s: draft-jetspec p5z_root_anchor_accept_path_runtime phase=%s root_anchor_accept_path_runtime_ready=%d root_anchor_accept_path_runtime_hash=%016" PRIx64 " root_verify_mask_runtime_ready=%d root_verify_mask_runtime_hash=%016" PRIx64 " root_tree_runtime_ready=%d root_tree_runtime_hash=%016" PRIx64 " root_verified_anchor=%d accept_path_len=%d actual_tree_nodes=%d actual_verify_mask_entries=%d actual_accepted_nodes=%d correction_token_present=%d no_target_logits_walk=1 no_target_accept_walk=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_head_graph=1 no_draft_tokens=1\n",
                                __func__, jetspec_runtime_phase_name(runtime_phase), root_anchor_accept_path_runtime_ready ? 1 : 0,
                                root_anchor_accept_path_runtime_hash_last, root_verify_mask_runtime_ready ? 1 : 0, root_verify_mask_runtime_hash_last,
                                root_tree_runtime_ready ? 1 : 0, root_tree_runtime_hash_last, root_verified_anchor_last, accept_path_len_last,
                                tree_build_actual_nodes_last, actual_verify_mask_entries_last, actual_accepted_nodes_last, correction_token_present_last);
                    }
                    return true;
                }
                if (trace_taps) {
                    LOG_INF("%s: draft-jetspec p5y_root_verify_mask_runtime phase=%s root_verify_mask_runtime_ready=%d root_verify_mask_runtime_hash=%016" PRIx64 " root_tree_runtime_ready=%d root_tree_runtime_hash=%016" PRIx64 " actual_tree_nodes=%d actual_verify_mask_entries=1 verify_mask_rows=1 verify_mask_cols=1 root_attends_self=%d root_mask_row=%d root_mask_col=%d prefix_visible=1 ancestor_only=1 sibling_visible=0 descendant_visible=0 no_draft_head_graph=1 no_mask_tensor=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_kv_mutation=1 no_draft_tokens=1\n",
                            __func__, jetspec_runtime_phase_name(runtime_phase), root_verify_mask_runtime_ready ? 1 : 0,
                            root_verify_mask_runtime_hash_last, root_tree_runtime_ready ? 1 : 0, root_tree_runtime_hash_last,
                            tree_build_actual_nodes_last, (int) root_verify_mask_values[0], root_verify_mask_rows[0], root_verify_mask_cols[0]);
                }
                return true;
            }
            if (trace_taps) {
                LOG_INF("%s: draft-jetspec p5x_root_tree_runtime phase=%s root_tree_runtime_ready=%d root_tree_runtime_hash=%016" PRIx64 " actual_tree_nodes=1 tree_token_ids=[%d] tree_parent_indices=[%d] tree_depth=[%d] tree_rank=[%d] tree_cum_logprob=[%.1f] root_parent=%d root_depth=%d parent_before_child=1 num_nodes_lte_budget=1 tree_build_node_budget=%d no_draft_head_graph=1 no_verify_mask=1 no_accept=1 no_token_commit=1 no_hidden_kv_commit=1 no_rejected_branch_discard=1 no_publish=1 no_visible_state_change=1 no_draft_tokens=1\n",
                        __func__, jetspec_runtime_phase_name(runtime_phase), root_tree_runtime_ready ? 1 : 0,
                        root_tree_runtime_hash_last, tree_token_ids[0], tree_parent_indices[0], tree_depth[0], tree_rank[0],
                        (double) tree_cum_logprob[0], tree_parent_indices[0], tree_depth[0], tree_build_node_budget_last);
            }
            return true;
        }
        if (!build_verify_mask_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_verify_mask_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_accept_path_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_accept_path_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_token_commit_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_token_commit_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_hidden_kv_survivor_commit_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_hidden_kv_survivor_commit_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_rejected_branch_discard_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_rejected_branch_discard_descriptor, tap_count, tap_width, taps);
            return true;
        }
        if (!build_publish_gate_descriptor()) {
            disable_runtime_state(jetspec_runtime_failure::invalid_publish_gate_descriptor, tap_count, tap_width, taps);
            return true;
        }

        if (trace_taps) {
            LOG_INF("%s: draft-jetspec runtime_state phase=%s failure=%s pre_round_snapshot_ready=%d pre_round_snapshot_hash=%016" PRIx64 " pre_round_seq_id=%d pre_round_prompt_tokens=%zu pre_round_prompt_hash=%016" PRIx64 " captured_rows=%zu row_state=%zu taps=%d width=%d hash=%016" PRIx64 " transaction_plan_ready=%d transaction_plan_hash=%016" PRIx64 " transaction_plan_phases=%d rollback_points=%d transient_reservation_ready=%d transient_reservation_hash=%016" PRIx64 " transient_tree_node_budget=%d actual_pages_reserved=0 tree_build_descriptor_ready=%d tree_build_descriptor_hash=%016" PRIx64 " planned_tree_node_budget=%d actual_tree_nodes=0 total_rows=%zu process=%zu resets=%zu\n",
                    __func__, jetspec_runtime_phase_name(runtime_phase), jetspec_runtime_failure_name(runtime_failure),
                    pre_round_snapshot_ready ? 1 : 0, pre_round_snapshot_hash_last, pre_round_seq_id_last,
                    pre_round_prompt_tokens_last, pre_round_prompt_hash_last,
                    n_target_tap_rows_cached, target_tap_row_state.size(), tap_count, tap_width, target_tap_hash_last,
                    transaction_plan_ready ? 1 : 0, transaction_plan_hash_last, transaction_plan_phase_count_last,
                    transaction_plan_rollback_count_last, transient_reservation_ready ? 1 : 0, transient_reservation_hash_last,
                    transient_reservation_node_budget_last, tree_build_descriptor_ready ? 1 : 0, tree_build_descriptor_hash_last,
                    tree_build_node_budget_last,
                    n_target_tap_rows_total, n_target_tap_process, n_runtime_state_resets);
            LOG_INF("%s: draft-jetspec transaction_plan_scaffold order=%s failpoints=%s transaction_phase=%s transient_reservation_phase=reserve_transient_tree_pages rollback_point=after_reserve transient_reservation_descriptor_only=1 transient_reservation_ready=%d transient_reservation_hash=%016" PRIx64 " actual_pages_reserved=0 pre_publish_visible_state_unmodified=1 no_reserve=1 no_real_reserve=1 no_page_map_write=1 no_tree_build=1 no_verify_mask=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, JETSPEC_TRANSACTION_PHASE_ORDER, JETSPEC_TRANSACTION_ROLLBACK_POINTS, JETSPEC_PRE_ROUND_SNAPSHOT_PHASE,
                    transient_reservation_ready ? 1 : 0, transient_reservation_hash_last);
            LOG_INF("%s: draft-jetspec tree_build_descriptor tree_build_phase=build_tree rollback_point=after_build_tree tree_build_descriptor_only=1 tree_build_descriptor_ready=%d tree_build_descriptor_hash=%016" PRIx64 " planned_tree_node_budget=%d actual_tree_nodes=0 root_parent=%d root_depth=%d pre_publish_visible_state_unmodified=1 no_real_tree_build=1 no_tree_arrays=1 no_verify_mask=1 no_accept=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, tree_build_descriptor_ready ? 1 : 0, tree_build_descriptor_hash_last,
                    tree_build_node_budget_last, tree_build_root_parent_last, tree_build_root_depth_last);
            LOG_INF("%s: draft-jetspec verify_mask_descriptor verify_mask_phase=build_verify_mask rollback_point=after_verify_mask verify_mask_descriptor_only=1 verify_mask_descriptor_ready=%d verify_mask_descriptor_hash=%016" PRIx64 " actual_verify_mask_entries=0 pre_publish_visible_state_unmodified=1 no_real_verify_mask=1 no_verify_mask=1 no_accept=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, verify_mask_descriptor_ready ? 1 : 0, verify_mask_descriptor_hash_last);
            LOG_INF("%s: draft-jetspec accept_path_descriptor accept_path_phase=accept_path rollback_point=after_accept accept_path_descriptor_only=1 accept_path_descriptor_ready=%d accept_path_descriptor_hash=%016" PRIx64 " actual_accepted_nodes=0 correction_token_present=0 pre_publish_visible_state_unmodified=1 no_real_accept=1 no_commit_tokens=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, accept_path_descriptor_ready ? 1 : 0, accept_path_descriptor_hash_last);
            LOG_INF("%s: draft-jetspec token_commit_descriptor token_commit_phase=commit_tokens rollback_point=after_token_commit token_commit_descriptor_only=1 token_commit_descriptor_ready=%d token_commit_descriptor_hash=%016" PRIx64 " actual_committed_tokens=0 pre_publish_visible_state_unmodified=1 no_real_token_commit=1 no_visible_token_publish=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, token_commit_descriptor_ready ? 1 : 0, token_commit_descriptor_hash_last);
            LOG_INF("%s: draft-jetspec hidden_kv_survivor_commit_descriptor hidden_kv_survivor_commit_phase=commit_hidden_kv_survivors rollback_point=after_hidden_kv_commit hidden_kv_survivor_commit_descriptor_only=1 hidden_kv_survivor_commit_descriptor_ready=%d hidden_kv_survivor_commit_descriptor_hash=%016" PRIx64 " actual_survivor_pages_committed=0 pre_publish_visible_state_unmodified=1 no_real_hidden_kv_commit=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, hidden_kv_survivor_commit_descriptor_ready ? 1 : 0, hidden_kv_survivor_commit_descriptor_hash_last);
            LOG_INF("%s: draft-jetspec rejected_branch_discard_descriptor rejected_branch_discard_phase=discard_rejected_branches rollback_point=after_rejected_discard rejected_branch_discard_descriptor_only=1 rejected_branch_discard_descriptor_ready=%d rejected_branch_discard_descriptor_hash=%016" PRIx64 " actual_pages_discarded=0 rejected_branch_pages_reachable_after_discard=0 pre_publish_visible_state_unmodified=1 no_real_rejected_branch_discard=1 no_kv_mutation=1 no_publish=1 no_draft_tokens=1\n",
                    __func__, rejected_branch_discard_descriptor_ready ? 1 : 0, rejected_branch_discard_descriptor_hash_last);
            LOG_INF("%s: draft-jetspec publish_gate_descriptor publish_gate_phase=publish_post_commit_state publish_gate_descriptor_only=1 publish_gate_descriptor_ready=%d publish_gate_descriptor_hash=%016" PRIx64 " actual_publish_visible_state=0 publish_after_commit_and_discard_only=1 no_real_publish=1 no_visible_state_change=1 no_draft_tokens=1\n",
                    __func__, publish_gate_descriptor_ready ? 1 : 0, publish_gate_descriptor_hash_last);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        n_runtime_draft_calls++;
        if (trace_taps) {
            LOG_INF("%s: draft-jetspec runtime_state phase=%s failure=%s pre_round_snapshot_ready=%d pre_round_snapshot_hash=%016" PRIx64 " captured_rows=%zu hash=%016" PRIx64 " transaction_plan_ready=%d transaction_plan_hash=%016" PRIx64 " transient_reservation_ready=%d transient_reservation_hash=%016" PRIx64 " actual_pages_reserved=0 tree_build_descriptor_ready=%d tree_build_descriptor_hash=%016" PRIx64 " root_tree_runtime_ready=%d root_tree_runtime_hash=%016" PRIx64 " root_verify_mask_runtime_ready=%d root_verify_mask_runtime_hash=%016" PRIx64 " root_anchor_accept_path_runtime_ready=%d root_anchor_accept_path_runtime_hash=%016" PRIx64 " root_token_commit_noop_runtime_ready=%d root_token_commit_noop_runtime_hash=%016" PRIx64 " real_draft_head_canary_ready=%d real_draft_head_canary_hash=%016" PRIx64 " real_topk_candidate_runtime_ready=%d real_topk_candidate_runtime_hash=%016" PRIx64 " real_topk_candidate_runtime_builds=%zu real_topk_tree_runtime_ready=%d real_topk_tree_runtime_hash=%016" PRIx64 " real_topk_tree_runtime_builds=%zu real_draft_head_canary_rows=%d real_draft_head_logits_rows=%d actual_tree_nodes=%d actual_verify_mask_entries=%d root_verified_anchor=%d accept_path_len=%d actual_accepted_nodes=%d correction_token_present=%d actual_committed_tokens=%d draft_call=%zu no_draft=1 no_kv_mutation=1 no_draft_token_emit=1 no_reserve=1 no_real_reserve=1 no_page_map_write=1 no_tree_expand=1 no_full_verify_mask=1 no_target_logits_walk=1 no_real_token_commit=1 no_visible_token_publish=1\n",
                    __func__, jetspec_runtime_phase_name(runtime_phase), jetspec_runtime_failure_name(runtime_failure),
                    pre_round_snapshot_ready ? 1 : 0, pre_round_snapshot_hash_last,
                    n_target_tap_rows_cached, target_tap_hash_last, transaction_plan_ready ? 1 : 0,
                    transaction_plan_hash_last, transient_reservation_ready ? 1 : 0, transient_reservation_hash_last,
                    tree_build_descriptor_ready ? 1 : 0, tree_build_descriptor_hash_last,
                    root_tree_runtime_ready ? 1 : 0, root_tree_runtime_hash_last,
                    root_verify_mask_runtime_ready ? 1 : 0, root_verify_mask_runtime_hash_last,
                    root_anchor_accept_path_runtime_ready ? 1 : 0, root_anchor_accept_path_runtime_hash_last,
                    root_token_commit_noop_runtime_ready ? 1 : 0, root_token_commit_noop_runtime_hash_last,
                    real_draft_head_canary_ready ? 1 : 0, real_draft_head_canary_hash_last,
                    real_draft_head_topk_candidate_runtime_ready ? 1 : 0, real_draft_head_topk_candidate_hash_last,
                    n_real_draft_head_topk_candidate_runtime_builds,
                    real_draft_head_topk_tree_runtime_ready ? 1 : 0, real_draft_head_topk_tree_hash_last,
                    n_real_draft_head_topk_tree_runtime_builds,
                    real_draft_head_canary_output_rows_last, real_draft_head_canary_logits_rows_last,
                    tree_build_actual_nodes_last, actual_verify_mask_entries_last, root_verified_anchor_last,
                    accept_path_len_last, actual_accepted_nodes_last, correction_token_present_last, actual_committed_tokens_last, n_runtime_draft_calls);
            LOG_INF("%s: draft-jetspec descriptor_gate verify_mask_descriptor_ready=%d verify_mask_descriptor_hash=%016" PRIx64 " root_verify_mask_runtime_ready=%d root_verify_mask_runtime_hash=%016" PRIx64 " root_anchor_accept_path_runtime_ready=%d root_anchor_accept_path_runtime_hash=%016" PRIx64 " root_token_commit_noop_runtime_ready=%d root_token_commit_noop_runtime_hash=%016" PRIx64 " actual_verify_mask_entries=%d accept_path_descriptor_ready=%d accept_path_descriptor_hash=%016" PRIx64 " actual_accepted_nodes=%d token_commit_descriptor_ready=%d token_commit_descriptor_hash=%016" PRIx64 " actual_committed_tokens=%d hidden_kv_survivor_commit_descriptor_ready=%d hidden_kv_survivor_commit_descriptor_hash=%016" PRIx64 " actual_survivor_pages_committed=0 rejected_branch_discard_descriptor_ready=%d rejected_branch_discard_descriptor_hash=%016" PRIx64 " actual_pages_discarded=0 publish_gate_descriptor_ready=%d publish_gate_descriptor_hash=%016" PRIx64 " actual_publish_visible_state=0 no_real_token_commit=1 no_real_hidden_kv_commit=1 no_real_rejected_branch_discard=1 no_real_publish=1 no_visible_state_change=1 no_draft_tokens=1\n",
                    __func__, verify_mask_descriptor_ready ? 1 : 0, verify_mask_descriptor_hash_last,
                    root_verify_mask_runtime_ready ? 1 : 0, root_verify_mask_runtime_hash_last,
                    root_anchor_accept_path_runtime_ready ? 1 : 0, root_anchor_accept_path_runtime_hash_last,
                    root_token_commit_noop_runtime_ready ? 1 : 0, root_token_commit_noop_runtime_hash_last,
                    actual_verify_mask_entries_last,
                    accept_path_descriptor_ready ? 1 : 0, accept_path_descriptor_hash_last,
                    actual_accepted_nodes_last,
                    token_commit_descriptor_ready ? 1 : 0, token_commit_descriptor_hash_last,
                    actual_committed_tokens_last,
                    hidden_kv_survivor_commit_descriptor_ready ? 1 : 0, hidden_kv_survivor_commit_descriptor_hash_last,
                    rejected_branch_discard_descriptor_ready ? 1 : 0, rejected_branch_discard_descriptor_hash_last,
                    publish_gate_descriptor_ready ? 1 : 0, publish_gate_descriptor_hash_last);
        }
        // fail closed: do not emit draft tokens before draft-head graph, tree verify, and rollback runtime exist.
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop until tree commit/rollback state is implemented.
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    std::vector<llama_sampler *> backend_topk_smpls;
    std::vector<bool> backend_topk_attached;

    bool backend_topk_enabled = false;
    bool backend_topk_require = false;
    bool backend_topk_verify = false;
    int32_t backend_topk_k = 16;

    bool sidecar_enabled = false;
    bool sidecar_require = false;
    int32_t sidecar_k = 256;

    bool quality_trace_enabled = false;
    bool teacher_probe_enabled = false;

    // Default-off position-fragility gate for noisy MTP draft paths (for example packed16 K).
    // It stops drafting before low-margin later positions instead of forcing a fixed max depth.
    bool margin_gate_enabled = false;
    float margin_gate_min = 0.0f;
    int32_t margin_gate_depth_min = 1;

    struct backend_topk_entry {
        llama_token id = LLAMA_TOKEN_NULL;
        float logit = -INFINITY;
    };

    struct quality_trace_entry {
        bool valid = false;
        int depth = 0;
        llama_token draft_top1 = LLAMA_TOKEN_NULL;
        llama_token target_top1 = LLAMA_TOKEN_NULL;
        float draft_p_top1 = 0.0f;
        float draft_margin = 0.0f;
        float target_margin = 0.0f;
        bool top1_match = false;
        const char * source = "unknown";
    };

    static void copy_branch_candidates(
            std::vector<common_speculative_branch_candidate> & out,
            const std::vector<backend_topk_entry> & top,
            float p_top1) {
        out.clear();
        out.reserve(top.size());
        for (size_t i = 0; i < top.size(); ++i) {
            common_speculative_branch_candidate cand;
            cand.id = top[i].id;
            cand.logit = top[i].logit;
            cand.p = i == 0 ? p_top1 : 0.0f;
            cand.rank = (int32_t) i;
            out.push_back(cand);
        }
    }

    static void copy_selected_branch_candidate(
            std::vector<common_speculative_branch_candidate> & out,
            llama_token id,
            float p_top1) {
        out.clear();
        if (id == LLAMA_TOKEN_NULL) {
            return;
        }
        common_speculative_branch_candidate cand;
        cand.id = id;
        cand.logit = 0.0f;
        cand.p = p_top1;
        cand.rank = 0;
        out.push_back(cand);
    }

    int32_t n_embd = 0;

    bool kv_shared_with_target = false;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<std::vector<backend_topk_entry>> target_sidecar;
    std::vector<int32_t> verify_h_rows;
    std::vector<std::vector<quality_trace_entry>> last_quality_trace;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_pre_norm width");

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 1;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        backend_topk_enabled = []() {
            const char * env = getenv("LLAMA_MTP_BACKEND_TOPK");
            return env ? atoi(env) != 0 : true;
        }();
        backend_topk_require = [this]() {
            const char * env = getenv("LLAMA_MTP_BACKEND_TOPK_REQUIRE");
            return env ? atoi(env) != 0 : backend_topk_enabled;
        }();
        backend_topk_verify = []() {
            const char * env = getenv("LLAMA_MTP_TOPK_VERIFY");
            return env && atoi(env) != 0;
        }();
        quality_trace_enabled = []() {
            const char * env = getenv("LLAMA_MTP_PACKED16_QUALITY_TRACE");
            if (env && atoi(env) != 0) {
                return true;
            }
            env = getenv("LLAMA_MTP_QUALITY_TRACE");
            return env && atoi(env) != 0;
        }();
        teacher_probe_enabled = []() {
            const char * env = getenv("LLAMA_MTP_TEACHER_PROBE");
            return env && atoi(env) != 0;
        }();
        if (teacher_probe_enabled && backend_topk_enabled) {
            LOG_WRN("%s: disabling MTP backend top-k while LLAMA_MTP_TEACHER_PROBE=1 because probe catch-up emits multiple logits rows per sequence\n", __func__);
            backend_topk_enabled = false;
            backend_topk_require = false;
        }
        const bool branch_candidates_requested = common_speculative_env_enabled("LLAMA_MTP_DRAFT_CANDIDATES_TRACE") ||
            common_speculative_env_enabled("LLAMA_MTP_DRAFT_BRANCH_CANDIDATES") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_SIBLING_TXN_PROOF") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_SAMPLER_COMMIT") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_RECURRENT_COMMIT") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_KV_SPLIT_PROOF") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_KV_SPLIT_COMMIT") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_KV_ATTENTION_IMPORT_COMMIT") ||
            common_speculative_env_enabled("LLAMA_MTP_QBLOCK_BRANCH_TXN_KV_PHYSICAL_IMPORT_COMMIT");
        if (const char * env = getenv("LLAMA_MTP_DRAFT_MARGIN_MIN")) {
            char * end = nullptr;
            const float v = std::strtof(env, &end);
            if (end != env && std::isfinite(v) && v > 0.0f) {
                margin_gate_enabled = true;
                margin_gate_min = v;
            }
        }
        if (const char * env = getenv("LLAMA_MTP_DRAFT_MARGIN_DEPTH_MIN")) {
            margin_gate_depth_min = std::max(1, atoi(env));
        }
        bool backend_topk_k_user = false;
        if (const char * env = getenv("LLAMA_MTP_BACKEND_TOPK_K")) {
            backend_topk_k = std::max(1, std::min(16, atoi(env)));
            backend_topk_k_user = true;
        }
        if (!backend_topk_k_user && (quality_trace_enabled || branch_candidates_requested)) {
            backend_topk_k = 16;
        } else if (!backend_topk_k_user && this->params.p_min <= 0.0f) {
            backend_topk_k = 1;
        }
        if (this->params.p_min > 0.0f && backend_topk_k < 16) {
            LOG_WRN("%s: MTP backend top-k k=%d is insufficient for p_min confidence gate; using k=16\n",
                    __func__, backend_topk_k);
            backend_topk_k = 16;
        }
        if (margin_gate_enabled && backend_topk_k < 2) {
            backend_topk_k = 2;
        }

        sidecar_enabled = []() {
            const char * env = getenv("LLAMA_MTP_SIDECAR_CANDIDATES");
            return env && atoi(env) != 0;
        }();
        sidecar_require = []() {
            const char * env = getenv("LLAMA_MTP_SIDECAR_REQUIRE");
            return env && atoi(env) != 0;
        }();
        if (const char * env = getenv("LLAMA_MTP_SIDECAR_K")) {
            sidecar_k = std::max(1, std::min(1024, atoi(env)));
        }
        if (quality_trace_enabled && sidecar_k < 2) {
            sidecar_k = 2;
        }
        if (quality_trace_enabled) {
            LOG_INF("%s: MTP packed16 quality trace enabled sidecar_k=%d backend_topk_k=%d direct_sidecar=%d\n",
                    __func__, sidecar_k, backend_topk_k, sidecar_enabled ? 1 : 0);
        }
        if (branch_candidates_requested) {
            LOG_INF("%s: MTP draft branch-candidate capture enabled backend_topk_k=%d\n",
                    __func__, backend_topk_k);
        }
        if (margin_gate_enabled) {
            LOG_INF("%s: MTP draft margin gate enabled min=%.6g depth_min=%d backend_topk_k=%d\n",
                    __func__, (double) margin_gate_min, margin_gate_depth_min, backend_topk_k);
        }
        if (sidecar_enabled) {
            LOG_INF("%s: MTP target-topK sidecar enabled k=%d require=%d source=target_topk direct_top1=1\n",
                    __func__, sidecar_k, sidecar_require ? 1 : 0);
        }

        backend_topk_smpls.assign(n_seq, nullptr);
        backend_topk_attached.assign(n_seq, false);
        if (backend_topk_enabled) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
                llama_sampler * chain = llama_sampler_chain_init(cparams);
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(backend_topk_k));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: failed to attach backend top-k sampler for seq_id=%d; falling back to CPU sampler path\n",
                            __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    if (backend_topk_require) {
                        throw std::runtime_error("LLAMA_MTP_BACKEND_TOPK_REQUIRE=1 but backend top-k sampler attachment failed");
                    }
                    continue;
                }

                backend_topk_smpls[seq_id] = chain;
                backend_topk_attached[seq_id] = true;
            }

            LOG_INF("%s: MTP backend top-k consumer enabled k=%d require=%d verify=%d\n",
                    __func__, backend_topk_k, backend_topk_require ? 1 : 0, backend_topk_verify ? 1 : 0);
        }

        llama_set_embeddings_pre_norm(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);
        llama_set_mtp_source(ctx_dft, ctx_tgt);

        kv_shared_with_target = llama_model_n_layer_kv(llama_get_model(ctx_dft)) == 0;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        target_sidecar.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        last_quality_trace.assign(n_seq, {});

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_topk_smpls.size(); ++seq_id) {
            if (backend_topk_smpls[seq_id] != nullptr) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
                llama_sampler_free(backend_topk_smpls[seq_id]);
                backend_topk_smpls[seq_id] = nullptr;
            }
        }

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    bool draft_backend_topk(struct llama_context * ctx, int idx, llama_seq_id seq_id, llama_token & id, float & p_top1,
            std::vector<backend_topk_entry> & top, std::string & reason) const {
        top.clear();

        if (!backend_topk_enabled) {
            reason = "disabled";
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) backend_topk_attached.size() || !backend_topk_attached[seq_id]) {
            reason = "not_attached";
            return false;
        }

        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        if (backend_topk_k == 1 && params.p_min <= 0.0f && !backend_topk_verify) {
            const llama_token sampled = llama_get_sampled_token_ith(ctx, idx);
            if (sampled != LLAMA_TOKEN_NULL) {
                if (sampled < 0 || sampled >= nv) {
                    reason = "sampled_id";
                    return false;
                }
                id = sampled;
                p_top1 = 1.0f;
                top.push_back({ sampled, 0.0f });
                if (getenv("LLAMA_MTP_TOPK_TRACE")) {
                    fprintf(stderr, "MTP_TOPK_TRACE: idx=%d sampled=%d source=fused_lm_head_top1\n", idx, (int) id);
                }
                reason.clear();
                return true;
            }
        }

        const uint32_t n_logits = llama_get_sampled_logits_count_ith(ctx, idx);
        const uint32_t n_ids    = llama_get_sampled_candidates_count_ith(ctx, idx);
        float * logits = llama_get_sampled_logits_ith(ctx, idx);
        llama_token * ids = llama_get_sampled_candidates_ith(ctx, idx);

        if (logits == nullptr) {
            reason = "logits_null";
            return false;
        }
        if (ids == nullptr) {
            reason = "candidates_null";
            return false;
        }
        if (n_logits == 0 || n_logits > (uint32_t) backend_topk_k) {
            reason = "logits_count";
            return false;
        }
        if (n_ids != n_logits) {
            reason = "count_mismatch";
            return false;
        }

        top.reserve(n_logits);
        for (uint32_t k = 0; k < n_logits; ++k) {
            if (ids[k] < 0 || ids[k] >= nv) {
                reason = "candidate_id";
                return false;
            }
            if (!std::isfinite(logits[k])) {
                reason = "candidate_logit";
                return false;
            }
            top.push_back({ ids[k], logits[k] });
        }

        std::sort(top.begin(), top.end(), [](const backend_topk_entry & a, const backend_topk_entry & b) {
            if (a.logit == b.logit) {
                return a.id < b.id;
            }
            return a.logit > b.logit;
        });

        if (top.empty()) {
            reason = "empty";
            return false;
        }

        float denom = 0.0f;
        for (const auto & e : top) {
            denom += std::exp(e.logit - top[0].logit);
        }

        id = top[0].id;
        p_top1 = denom > 0.0f ? 1.0f / denom : 0.0f;

        if (getenv("LLAMA_MTP_CONF_TRACE")) {
            const float margin = top.size() > 1 ? top[0].logit - top[1].logit : INFINITY;
            const llama_token top2_id = top.size() > 1 ? top[1].id : LLAMA_TOKEN_NULL;
            const float top2_logit = top.size() > 1 ? top[1].logit : -INFINITY;
            fprintf(stderr, "MTP_CONF_TRACE: idx=%d sampled=%d top1=%d p_top16=%.6g margin=%.6g top1_logit=%.6g top2=%d top2_logit=%.6g source=backend count=%zu\n",
                    idx, (int) id, (int) top[0].id, p_top1, margin, top[0].logit, (int) top2_id, top2_logit, top.size());
        }

        if (getenv("LLAMA_MTP_TOPK_TRACE")) {
            fprintf(stderr, "MTP_TOPK_TRACE: idx=%d sampled=%d source=backend", idx, (int) id);
            for (const auto & e : top) {
                fprintf(stderr, " %d:%.8g", (int) e.id, e.logit);
            }
            fprintf(stderr, "\n");
        }

        reason.clear();
        return true;
    }

    bool verify_backend_topk(struct llama_context * ctx, int idx, llama_seq_id seq_id,
            const std::vector<backend_topk_entry> & top, float p_top1) const {
        if (!backend_topk_verify) {
            return true;
        }

        const float * logits = llama_get_logits_raw_ith(ctx, idx);
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

        std::string mismatch;
        auto add_mismatch = [&mismatch](const char * what) {
            if (!mismatch.empty()) {
                mismatch += ",";
            }
            mismatch += what;
        };

        if (logits == nullptr) {
            fprintf(stderr, "MTP_TOPK_VERIFY: idx=%d seq_id=%d ok=0 mismatch=raw_logits_null\n", idx, (int) seq_id);
            return false;
        }
        if (top.empty()) {
            fprintf(stderr, "MTP_TOPK_VERIFY: idx=%d seq_id=%d ok=0 mismatch=backend_empty\n", idx, (int) seq_id);
            return false;
        }

        const int k_conf = std::max(1, std::min(16, backend_topk_k));
        std::vector<backend_topk_entry> cpu_top;
        cpu_top.reserve(k_conf);

        for (int j = 0; j < nv; ++j) {
            const float v = logits[j];
            if (!std::isfinite(v)) {
                continue;
            }

            backend_topk_entry cand { j, v };
            auto better = [](const backend_topk_entry & a, const backend_topk_entry & b) {
                if (a.logit == b.logit) {
                    return a.id < b.id;
                }
                return a.logit > b.logit;
            };

            auto it = cpu_top.begin();
            for (; it != cpu_top.end(); ++it) {
                if (better(cand, *it)) {
                    break;
                }
            }
            if (it != cpu_top.end() || (int) cpu_top.size() < k_conf) {
                cpu_top.insert(it, cand);
                if ((int) cpu_top.size() > k_conf) {
                    cpu_top.pop_back();
                }
            }
        }

        if (cpu_top.empty()) {
            fprintf(stderr, "MTP_TOPK_VERIFY: idx=%d seq_id=%d ok=0 mismatch=cpu_empty\n", idx, (int) seq_id);
            return false;
        }

        const auto contains_id = [](const std::vector<backend_topk_entry> & entries, llama_token id) {
            for (const auto & e : entries) {
                if (e.id == id) {
                    return true;
                }
            }
            return false;
        };

        const float logit_tol = 1.0e-4f;
        const float tie_tol   = 1.0e-6f;
        const float p_tol     = 5.0e-4f;

        bool logits_match = true;
        for (const auto & e : top) {
            if (e.id < 0 || e.id >= nv) {
                logits_match = false;
                continue;
            }
            const float raw = logits[e.id];
            const float tol = logit_tol * std::max(1.0f, std::max(std::fabs(raw), std::fabs(e.logit)));
            if (!std::isfinite(raw) || std::fabs(raw - e.logit) > tol) {
                logits_match = false;
            }
        }
        if (!logits_match) {
            add_mismatch("logits");
        }

        const bool top1_match = top[0].id == cpu_top[0].id;
        if (!top1_match) {
            add_mismatch("top1");
        }

        const size_t k_cmp = std::min(top.size(), cpu_top.size());
        bool set_exact = top.size() == k_cmp;
        for (size_t k = 0; k < k_cmp && set_exact; ++k) {
            if (!contains_id(top, cpu_top[k].id)) {
                set_exact = false;
            }
        }

        bool set_tie_permitted = true;
        if (k_cmp == 0 || top.size() > cpu_top.size()) {
            set_tie_permitted = false;
        } else {
            const float boundary = cpu_top[k_cmp - 1].logit;
            for (const auto & e : top) {
                if (e.id < 0 || e.id >= nv || logits[e.id] + tie_tol < boundary) {
                    set_tie_permitted = false;
                    break;
                }
            }
            for (size_t k = 0; k < k_cmp; ++k) {
                if (cpu_top[k].logit > boundary + tie_tol && !contains_id(top, cpu_top[k].id)) {
                    set_tie_permitted = false;
                    break;
                }
            }
        }

        const bool candidates_match = set_exact || set_tie_permitted;
        if (!candidates_match) {
            add_mismatch("candidates");
        }

        float p_cpu = 0.0f;
        {
            float denom = 0.0f;
            for (int k = 0; k < k_conf && k < (int) cpu_top.size(); ++k) {
                denom += std::exp(cpu_top[k].logit - cpu_top[0].logit);
            }
            p_cpu = denom > 0.0f ? 1.0f / denom : 0.0f;
        }

        bool p_close = top.size() >= (size_t) k_conf;
        if (p_close) {
            const float tol = p_tol * std::max(1.0f, std::max(std::fabs(p_top1), std::fabs(p_cpu)));
            p_close = std::fabs(p_top1 - p_cpu) <= tol;
        }
        if (!p_close) {
            add_mismatch("p_top16");
        }

        const bool ok = top1_match && candidates_match && logits_match && p_close;
        fprintf(stderr,
                "MTP_TOPK_VERIFY: idx=%d seq_id=%d ok=%d mismatch=%s top1_backend=%d top1_cpu=%d count_backend=%zu count_cpu=%zu p_top16_backend=%.9g p_top16_cpu=%.9g candidates=%d logits=%d p_close=%d\n",
                idx, (int) seq_id, ok ? 1 : 0, ok ? "none" : mismatch.c_str(),
                (int) top[0].id, (int) cpu_top[0].id, top.size(), cpu_top.size(),
                p_top1, p_cpu, candidates_match ? 1 : 0, logits_match ? 1 : 0, p_close ? 1 : 0);

        return ok;
    }

    static float draft_margin_from_top(const std::vector<backend_topk_entry> & top) {
        return top.size() > 1 ? top[0].logit - top[1].logit : INFINITY;
    }

    bool draft_margin_gate_reject(int depth, const std::vector<backend_topk_entry> & top, float & margin, const char *& reason) const {
        margin = draft_margin_from_top(top);
        reason = "disabled";
        if (!margin_gate_enabled || depth < margin_gate_depth_min) {
            return false;
        }
        if (top.size() < 2 || !std::isfinite(margin)) {
            reason = "missing_top2";
            return true;
        }
        if (margin < margin_gate_min) {
            reason = "low_margin";
            return true;
        }
        reason = "ok";
        return false;
    }

    float draft_confidence_topk(struct llama_context * ctx, int idx, llama_token id,
            std::vector<backend_topk_entry> * top_out = nullptr) const {
        constexpr int k_conf = 16;

        const float * logits = llama_get_logits_raw_ith(ctx, idx);
        if (logits == nullptr) {
            return 0.0f;
        }

        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        float top_val[k_conf];
        llama_token top_id[k_conf];
        for (int k = 0; k < k_conf; ++k) {
            top_val[k] = -INFINITY;
            top_id[k]  = LLAMA_TOKEN_NULL;
        }

        for (int j = 0; j < nv; ++j) {
            const float v = logits[j];
            if (!std::isfinite(v)) {
                continue;
            }
            for (int k = 0; k < k_conf; ++k) {
                if (v > top_val[k]) {
                    for (int u = k_conf - 1; u > k; --u) {
                        top_val[u] = top_val[u - 1];
                        top_id[u]  = top_id[u - 1];
                    }
                    top_val[k] = v;
                    top_id[k]  = j;
                    break;
                }
            }
        }

        if (top_out != nullptr) {
            top_out->clear();
            for (int k = 0; k < k_conf && top_id[k] != LLAMA_TOKEN_NULL; ++k) {
                top_out->push_back({ top_id[k], top_val[k] });
            }
        }

        if (top_id[0] == LLAMA_TOKEN_NULL || !std::isfinite(top_val[0])) {
            return 0.0f;
        }

        // MTP samples greedily via top_k=1, so cur_p[0].p is always 1.0 and
        // cannot calibrate --spec-draft-p-min. Use a stable top-16 local
        // softmax over raw logits instead; this keeps token selection unchanged
        // while making p_min an actual confidence gate.
        float denom = 0.0f;
        for (int k = 0; k < k_conf && top_id[k] != LLAMA_TOKEN_NULL; ++k) {
            denom += std::exp(top_val[k] - top_val[0]);
        }
        const float p_top1 = denom > 0.0f ? 1.0f / denom : 0.0f;

        if (getenv("LLAMA_MTP_CONF_TRACE")) {
            const float margin = top_id[1] != LLAMA_TOKEN_NULL ? top_val[0] - top_val[1] : INFINITY;
            fprintf(stderr, "MTP_CONF_TRACE: idx=%d sampled=%d top1=%d p_top16=%.6g margin=%.6g top1_logit=%.6g top2=%d top2_logit=%.6g\n",
                    idx, (int) id, (int) top_id[0], p_top1, margin, top_val[0], (int) top_id[1], top_val[1]);
        }

        if (getenv("LLAMA_MTP_TOPK_TRACE")) {
            fprintf(stderr, "MTP_TOPK_TRACE: idx=%d sampled=%d", idx, (int) id);
            for (int k = 0; k < k_conf && top_id[k] != LLAMA_TOKEN_NULL; ++k) {
                fprintf(stderr, " %d:%.8g", (int) top_id[k], top_val[k]);
            }
            fprintf(stderr, "\n");
        }

        return id == top_id[0] ? p_top1 : 0.0f;
    }

    void validate_real_mtp_batch(const char * phase, const llama_batch & b) const {
        if (!getenv("LLAMA_MTP_VALIDATE_INPUTS")) {
            return;
        }

        int n_outputs = 0;
        if (b.logits) {
            for (int32_t i = 0; i < b.n_tokens; ++i) {
                n_outputs += b.logits[i] != 0;
            }
        }

        int bad_h = 0;
        int64_t n_check = 0;
        if (b.embd) {
            n_check = (int64_t) b.n_tokens * n_embd;
            for (int64_t i = 0; i < n_check; ++i) {
                bad_h += !std::isfinite(b.embd[i]);
            }
        }

        fprintf(stderr, "MTP_INPUT_REAL: phase=%s token=%d embd=%d h_required=1 n_tokens=%d n_outputs=%d bad_h=%d checked_h=%lld mode=token_path_hidden_state\n",
            phase, b.token != nullptr, b.embd != nullptr, (int) b.n_tokens, n_outputs, bad_h, (long long) n_check);

        if (b.token == nullptr || b.embd == nullptr) {
            GGML_ABORT("real MTP decode requires both token ids and ubatch.embd hidden state");
        }
        if (bad_h != 0) {
            GGML_ABORT("real MTP decode hidden state contains NaN/Inf before graph execution");
        }
    }

    void capture_target_sidecar(struct llama_context * ctx, int idx, llama_seq_id seq_id) {
        if ((!sidecar_enabled && !quality_trace_enabled) || seq_id < 0 || seq_id >= (llama_seq_id) target_sidecar.size()) {
            return;
        }

        auto & out = target_sidecar[seq_id];
        out.clear();

        const float * logits = llama_get_logits_raw_ith(ctx, idx);
        if (logits == nullptr) {
            logits = llama_get_logits_ith(ctx, idx);
        }
        if (logits == nullptr) {
            if (getenv("LLAMA_MTP_SIDECAR_TRACE") || quality_trace_enabled) {
                fprintf(stderr, "MTP_SIDECAR_TRACE: phase=capture idx=%d seq_id=%d status=missing_logits quality=%d\n", idx, (int) seq_id, quality_trace_enabled ? 1 : 0);
            }
            return;
        }

        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const int k = std::max(1, std::min(sidecar_k, nv));
        out.reserve(k);

        auto better = [](const backend_topk_entry & a, const backend_topk_entry & b) {
            if (a.logit == b.logit) {
                return a.id < b.id;
            }
            return a.logit > b.logit;
        };
        auto heap_cmp = [&](const backend_topk_entry & a, const backend_topk_entry & b) {
            return better(b, a);
        };

        for (int j = 0; j < nv; ++j) {
            const float v = logits[j];
            if (!std::isfinite(v)) {
                continue;
            }
            backend_topk_entry cand { j, v };
            if ((int) out.size() < k) {
                out.push_back(cand);
                std::push_heap(out.begin(), out.end(), heap_cmp);
            } else if (better(cand, out.front())) {
                std::pop_heap(out.begin(), out.end(), heap_cmp);
                out.back() = cand;
                std::push_heap(out.begin(), out.end(), heap_cmp);
            }
        }

        std::sort(out.begin(), out.end(), better);

        if (getenv("LLAMA_MTP_SIDECAR_TRACE")) {
            fprintf(stderr, "MTP_SIDECAR_TRACE: phase=capture idx=%d seq_id=%d count=%zu", idx, (int) seq_id, out.size());
            for (int i = 0; i < std::min<int>(5, out.size()); ++i) {
                fprintf(stderr, " %d:%.8g", (int) out[i].id, out[i].logit);
            }
            fprintf(stderr, "\n");
        }
    }

    bool draft_sidecar_direct(llama_seq_id seq_id, llama_token & id, float & p_top1) const {
        if (!sidecar_enabled || seq_id < 0 || seq_id >= (llama_seq_id) target_sidecar.size()) {
            return false;
        }
        const auto & top = target_sidecar[seq_id];
        if (top.empty()) {
            return false;
        }
        id = top[0].id;
        if (top.size() == 1) {
            p_top1 = 1.0f;
        } else {
            float denom = 0.0f;
            for (const auto & e : top) {
                denom += std::exp(e.logit - top[0].logit);
            }
            p_top1 = denom > 0.0f ? 1.0f / denom : 0.0f;
        }
        return id != LLAMA_TOKEN_NULL;
    }

    const char * quality_lane_name() const {
        const char * disable_p16 = getenv("LLAMA_MTP_DISABLE_PACKED16_FA");
        if (!disable_p16 || atoi(disable_p16) == 0) {
            return "packed16_dot4_default";
        }
        return "f16_exact_control";
    }

    static const char * quality_env_value(const char * primary, const char * alias, const char * fallback) {
        const char * v = getenv(primary);
        if (!v || v[0] == '\0') {
            v = alias ? getenv(alias) : nullptr;
        }
        return (v && v[0] != '\0') ? v : fallback;
    }

    void record_quality_trace(
            llama_seq_id seq_id,
            int depth,
            llama_token draft_id,
            float p_draft,
            const std::vector<backend_topk_entry> & draft_top,
            const char * source) {
        if (!quality_trace_enabled || seq_id < 0 || seq_id >= (llama_seq_id) target_sidecar.size()) {
            return;
        }

        quality_trace_entry stored = {};
        stored.depth = depth;
        stored.draft_top1 = draft_top.empty() ? draft_id : draft_top[0].id;
        stored.draft_p_top1 = p_draft;
        stored.source = source ? source : "unknown";
        stored.draft_margin = draft_top.size() > 1 ? draft_top[0].logit - draft_top[1].logit : 0.0f;

        const auto & target_top = target_sidecar[seq_id];
        const llama_token draft_top2 = draft_top.size() > 1 ? draft_top[1].id : LLAMA_TOKEN_NULL;

        if (depth != 1) {
            last_quality_trace[seq_id].push_back(stored);
            fprintf(stderr,
                    "MTP_PACKED16_QUALITY: phase=draft lane=%s seq_id=%d depth=%d status=target_sidecar_depth1_only "
                    "draft_top1=%d draft_top2=%d draft_p_top1=%.8g draft_margin=%.8g draft_source=%s k_scale_mode=%s k_scale_mul=%s\n",
                    quality_lane_name(), (int) seq_id, depth, (int) stored.draft_top1, (int) draft_top2,
                    stored.draft_p_top1, stored.draft_margin, stored.source,
                    quality_env_value("GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE", "LLAMA_MTP_PACKED16_K_SCALE_MODE", "default"),
                    quality_env_value("GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL", "LLAMA_MTP_PACKED16_K_SCALE_MUL", "1"));
            return;
        }

        if (target_top.empty()) {
            last_quality_trace[seq_id].push_back(stored);
            fprintf(stderr,
                    "MTP_PACKED16_QUALITY: phase=draft lane=%s seq_id=%d depth=%d status=missing_target_sidecar draft_top1=%d draft_source=%s\n",
                    quality_lane_name(), (int) seq_id, depth, (int) stored.draft_top1, stored.source);
            return;
        }

        stored.valid = true;
        stored.target_top1 = target_top[0].id;
        stored.target_margin = target_top.size() > 1 ? target_top[0].logit - target_top[1].logit : 0.0f;
        stored.top1_match = stored.draft_top1 == stored.target_top1;
        last_quality_trace[seq_id].push_back(stored);

        const llama_token target_top2 = target_top.size() > 1 ? target_top[1].id : LLAMA_TOKEN_NULL;
        fprintf(stderr,
                "MTP_PACKED16_QUALITY: phase=draft lane=%s seq_id=%d depth=%d draft_top1=%d target_raw_top1=%d raw_top1_match=%d "
                "draft_top2=%d target_raw_top2=%d draft_p_top1=%.8g draft_margin=%.8g target_raw_margin=%.8g "
                "draft_source=%s k_scale_mode=%s k_scale_mul=%s\n",
                quality_lane_name(), (int) seq_id, depth, (int) stored.draft_top1, (int) stored.target_top1,
                stored.top1_match ? 1 : 0, (int) draft_top2, (int) target_top2,
                stored.draft_p_top1, stored.draft_margin, stored.target_margin, stored.source,
                quality_env_value("GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE", "LLAMA_MTP_PACKED16_K_SCALE_MODE", "default"),
                quality_env_value("GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL", "LLAMA_MTP_PACKED16_K_SCALE_MUL", "1"));
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1 && !kv_shared_with_target) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        return process_impl(batch_in, nullptr);
    }

    bool process_with_pre_norm(const llama_batch & batch_in, const float * h_pre_norm) override {
        if (h_pre_norm == nullptr || sidecar_enabled || quality_trace_enabled || teacher_probe_enabled) {
            return process(batch_in);
        }
        return process_impl(batch_in, h_pre_norm);
    }

    bool process_impl(const llama_batch & batch_in, const float * h_pre_norm) {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        const bool hidden_shift_trace = common_speculative_hidden_shift_trace_enabled();
        const bool hidden_shift_abort = common_speculative_env_enabled("LLAMA_MTP_HIDDEN_SHIFT_REQUIRE");
        std::vector<uint64_t> hidden_pending_before_hash(hidden_shift_trace || hidden_shift_abort ? n_seq : 0, 0);
        std::vector<uint64_t> hidden_catchup_row0_hash(hidden_shift_trace || hidden_shift_abort ? n_seq : 0, 0);
        std::vector<uint64_t> hidden_catchup_row1_hash(hidden_shift_trace || hidden_shift_abort ? n_seq : 0, 0);
        if (hidden_shift_trace || hidden_shift_abort) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                hidden_pending_before_hash[seq_id] = common_speculative_fnv1a64(pending_h[seq_id].data(), row_bytes);
            }
        }

        // If the target context is wired to stream MTP hidden states directly
        // into ctx_dft during llama_decode(), do not replay the same target
        // batch here. Replaying collides with the already-advanced draft KV and
        // can fail as an invalid input batch. We still capture target hidden
        // rows below for pending_h / verification bookkeeping.
        const bool mtp_hook_wired = getenv("LLAMA_MTP_HOOK_WIRE") != nullptr;

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!kv_shared_with_target && !mtp_hook_wired) {
            common_batch_clear(batch);

            std::vector<llama_token> teacher_expected;
            const bool teacher_probe = teacher_probe_enabled;
            for (int k = 0; k < n_tokens; ++k) {
                const bool want_logits = teacher_probe && k + 1 < n_tokens;
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, want_logits);
                if (want_logits) {
                    teacher_expected.push_back(batch_in.token[k + 1]);
                }
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = h_pre_norm ? h_pre_norm : llama_get_embeddings_pre_norm(ctx_tgt);
                GGML_ASSERT(h_tgt != nullptr);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            if (hidden_shift_trace || hidden_shift_abort) {
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (i_batch_beg[seq_id] < 0) {
                        continue;
                    }
                    hidden_catchup_row0_hash[seq_id] = common_speculative_fnv1a64(
                            batch.embd + (size_t) i_batch_beg[seq_id] * n_embd, row_bytes);
                    if (i_batch_beg[seq_id] + 1 <= i_batch_end[seq_id]) {
                        hidden_catchup_row1_hash[seq_id] = common_speculative_fnv1a64(
                                batch.embd + (size_t) (i_batch_beg[seq_id] + 1) * n_embd, row_bytes);
                    }
                }
            }

            validate_real_mtp_batch("process_catchup", batch);
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
                return false;
            }

            if (teacher_probe && !teacher_expected.empty()) {
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
                for (int32_t oi = 0; oi < (int32_t) teacher_expected.size(); ++oi) {
                    const float * logits = llama_get_logits_ith(ctx_dft, oi);
                    const llama_token expected = teacher_expected[oi];
                    int rank = 1;
                    int bad = 0;
                    float expected_val = logits ? logits[expected] : NAN;
                    float top_val[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
                    int   top_id [5] = {-1, -1, -1, -1, -1};
                    if (logits) {
                        for (int j = 0; j < nv; ++j) {
                            const float v = logits[j];
                            if (!std::isfinite(v)) {
                                ++bad;
                                continue;
                            }
                            if (v > expected_val) {
                                ++rank;
                            }
                            for (int t = 0; t < 5; ++t) {
                                if (v > top_val[t]) {
                                    for (int u = 4; u > t; --u) { top_val[u] = top_val[u - 1]; top_id[u] = top_id[u - 1]; }
                                    top_val[t] = v; top_id[t] = j;
                                    break;
                                }
                            }
                        }
                    }
                    fprintf(stderr, "MTP_TEACHER_PROBE: phase=process_catchup head=0 output=%d expected_offset=1 expected_token=%d expected_rank=%d expected_logit=%.6g bad_logits=%d top5=%d:%.6g,%d:%.6g,%d:%.6g,%d:%.6g,%d:%.6g\n",
                        (int) oi, (int) expected, rank, expected_val, bad,
                        top_id[0], top_val[0], top_id[1], top_val[1], top_id[2], top_val[2], top_id[3], top_val[3], top_id[4], top_val[4]);
                }
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = h_pre_norm ? h_pre_norm + (size_t) (i_batch_beg[seq_id] + i) * n_embd :
                    llama_get_embeddings_pre_norm_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                GGML_ASSERT(h != nullptr);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);

            if (hidden_shift_trace || hidden_shift_abort) {
                const uint64_t verify_first_hash = common_speculative_fnv1a64(verify_h[seq_id].data(), row_bytes);
                const uint64_t verify_last_hash = common_speculative_fnv1a64(
                        verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
                const uint64_t pending_after_hash = common_speculative_fnv1a64(pending_h[seq_id].data(), row_bytes);
                const bool catchup_ran = !kv_shared_with_target && !mtp_hook_wired;
                const bool row0_match = !catchup_ran || hidden_pending_before_hash[seq_id] == hidden_catchup_row0_hash[seq_id];
                const bool row1_match = !catchup_ran || n_rows <= 1 || hidden_catchup_row1_hash[seq_id] == verify_first_hash;
                const bool pending_after_match = pending_after_hash == verify_last_hash;
                if (hidden_shift_trace) {
                    fprintf(stderr,
                            "MTP_HIDDEN_SHIFT_TRACE: seq_id=%d n_rows=%d i_batch_beg=%d i_batch_end=%d catchup=%d pending_before_hash=%016" PRIx64 " catchup_row0_hash=%016" PRIx64 " row0_match=%d catchup_row1_hash=%016" PRIx64 " verify_first_hash=%016" PRIx64 " row1_shift_match=%d verify_last_hash=%016" PRIx64 " pending_after_hash=%016" PRIx64 " pending_after_match=%d\n",
                            (int) seq_id,
                            (int) n_rows,
                            (int) i_batch_beg[seq_id],
                            (int) i_batch_end[seq_id],
                            catchup_ran ? 1 : 0,
                            hidden_pending_before_hash[seq_id],
                            hidden_catchup_row0_hash[seq_id],
                            row0_match ? 1 : 0,
                            hidden_catchup_row1_hash[seq_id],
                            verify_first_hash,
                            row1_match ? 1 : 0,
                            verify_last_hash,
                            pending_after_hash,
                            pending_after_match ? 1 : 0);
                }
                if (hidden_shift_abort && (!row0_match || !row1_match || !pending_after_match)) {
                    GGML_ABORT("MTP hidden-shift trace invariant failed: seq_id=%d row0_match=%d row1_match=%d pending_after_match=%d",
                            (int) seq_id, row0_match ? 1 : 0, row1_match ? 1 : 0, pending_after_match ? 1 : 0);
                }
            }

            if (h_pre_norm == nullptr) {
                capture_target_sidecar(ctx_tgt, i_batch_end[seq_id], seq_id);
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (seq_id >= 0 && seq_id < (llama_seq_id) last_quality_trace.size()) {
                last_quality_trace[seq_id].clear();
            }

            if (!dp.drafting) {
                continue;
            }

            if (dp.branch_candidates != nullptr) {
                dp.branch_candidates->clear();
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        if (sidecar_enabled) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting) {
                    last_n_drafted[seq_id] = 0;
                    continue;
                }

                llama_token id = LLAMA_TOKEN_NULL;
                float p_draft = 0.0f;
                if (!draft_sidecar_direct(seq_id, id, p_draft)) {
                    if (sidecar_require) {
                        GGML_ABORT("LLAMA_MTP_SIDECAR_REQUIRE=1 but target-topK sidecar is unavailable/invalid");
                    }
                    if (getenv("LLAMA_MTP_SIDECAR_TRACE")) {
                        fprintf(stderr, "MTP_SIDECAR_TRACE: phase=draft seq_id=%d status=missing no_draft=1\n", (int) seq_id);
                    }
                    last_n_drafted[seq_id] = 0;
                    continue;
                }

                if (getenv("LLAMA_MTP_SIDECAR_TRACE")) {
                    fprintf(stderr, "MTP_SIDECAR_TRACE: phase=draft seq_id=%d sampled=%d p=%.8g source=target_topk_direct\n",
                            (int) seq_id, (int) id, p_draft);
                }

                if (p_draft < params.p_min) {
                    last_n_drafted[seq_id] = 0;
                    continue;
                }

                common_sampler_accept(smpls[seq_id].get(), id, true);
                dp.result->push_back(id);
                if (dp.branch_candidates != nullptr) {
                    dp.branch_candidates->emplace_back();
                    copy_selected_branch_candidate(dp.branch_candidates->back(), id, p_draft);
                }
                last_n_drafted[seq_id] = 1;
            }

            return;
        }

        validate_real_mtp_batch("draft_initial", batch);
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        // LLAMA_MTP_TRACE: dump draft logit diagnostics
        if (getenv("LLAMA_MTP_TRACE")) {
            const float * logits = llama_get_logits(ctx_dft);
            if (logits) {
                int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_dft)));
                float mx = -1e30f, mn = 1e30f, sum = 0; int mi = -1;
                float top3_val[3] = {-1e30f, -1e30f, -1e30f};
                int   top3_id[3]  = {-1, -1, -1};
                for (int j = 0; j < std::min(nv, 256000); j++) {
                    float v = logits[j];
                    if (v > mx) { mx = v; mi = j; }
                    if (v < mn) mn = v;
                    sum += v;
                    if (v > top3_val[0]) { top3_val[2]=top3_val[1]; top3_id[2]=top3_id[1]; top3_val[1]=top3_val[0]; top3_id[1]=top3_id[0]; top3_val[0]=v; top3_id[0]=j; }
                    else if (v > top3_val[1]) { top3_val[2]=top3_val[1]; top3_id[2]=top3_id[1]; top3_val[1]=v; top3_id[1]=j; }
                    else if (v > top3_val[2]) { top3_val[2]=v; top3_id[2]=j; }
                }
                fprintf(stderr, "MTP_TRACE logits top=%d(%.2f) top3=%d(%.2f) %d(%.2f) %d(%.2f) min=%.2f mean=%.2f nv=%d s0=%.2f\n",
                    mi, mx, top3_id[0], top3_val[0], top3_id[1], top3_val[1], top3_id[2], top3_val[2],
                    mn, sum/std::max(nv,1), nv, logits[0]);
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const int sample_idx = i_batch;
                llama_token id = LLAMA_TOKEN_NULL;
                float p_draft = 0.0f;
                bool used_backend_topk = false;
                std::vector<backend_topk_entry> top_backend;
                std::vector<backend_topk_entry> top_cpu;

                if (backend_topk_enabled) {
                    std::string reason;
                    used_backend_topk = draft_backend_topk(ctx_dft, sample_idx, seq_id, id, p_draft, top_backend, reason);
                    if (!used_backend_topk) {
                        if (backend_topk_require) {
                            GGML_ABORT("LLAMA_MTP_BACKEND_TOPK_REQUIRE=1 but backend top-k output is unavailable/invalid");
                        }
                        if (getenv("LLAMA_MTP_TOPK_TRACE")) {
                            fprintf(stderr, "MTP_TOPK_TRACE: idx=%d seq_id=%d source=backend invalid=%s fallback=cpu\n",
                                    sample_idx, (int) seq_id, reason.c_str());
                        }
                    }
                }

                if (used_backend_topk && backend_topk_verify) {
                    const bool verify_ok = verify_backend_topk(ctx_dft, sample_idx, seq_id, top_backend, p_draft);
                    if (!verify_ok && backend_topk_require) {
                        GGML_ABORT("LLAMA_MTP_BACKEND_TOPK_REQUIRE=1 but backend top-k verification failed");
                    }
                }

                if (!used_backend_topk) {
                    common_sampler_sample(smpl, ctx_dft, sample_idx, true);
                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    // Add drafted token for each sequence if the MTP head is confident enough.
                    // The sampler itself remains greedy (top_k=1); p_min is gated by a
                    // separate top-k logit confidence so it is not permanently 1.0.
                    id = cur_p->data[0].id;
                    const bool need_branch_candidates = dparams[seq_id].branch_candidates != nullptr;
                    const bool need_confidence = need_branch_candidates || params.p_min > 0.0f || quality_trace_enabled || margin_gate_enabled || getenv("LLAMA_MTP_CONF_TRACE") || getenv("LLAMA_MTP_TOPK_TRACE");
                    p_draft = need_confidence ? draft_confidence_topk(ctx_dft, sample_idx, id, &top_cpu) : 1.0f;
                } else {
                    for (int k = 0; k < std::min(3, (int) top_backend.size()); ++k) {
                        LOG_DBG(" - seq_id %d, draft backend candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, top_backend[k].id, top_backend[k].logit,
                                common_token_to_piece(ctx_dft, top_backend[k].id).c_str());
                    }
                }

                const auto & top_for_gate = used_backend_topk ? top_backend : top_cpu;

                if (quality_trace_enabled) {
                    record_quality_trace(seq_id, i + 1, id, p_draft, top_for_gate,
                            used_backend_topk ? "backend_topk" : "cpu_sampler");
                }

                ++i_batch;

                if (p_draft < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                float draft_margin = 0.0f;
                const char * margin_reason = nullptr;
                if (draft_margin_gate_reject(i + 1, top_for_gate, draft_margin, margin_reason)) {
                    if (getenv("LLAMA_MTP_DRAFT_MARGIN_TRACE") || getenv("LLAMA_MTP_CONF_TRACE")) {
                        fprintf(stderr,
                                "MTP_DRAFT_MARGIN_GATE: seq_id=%d depth=%d stop=1 reason=%s margin=%.8g min=%.8g p_top1=%.8g topk=%zu token=%d\n",
                                (int) seq_id, i + 1, margin_reason ? margin_reason : "unknown", draft_margin,
                                (double) margin_gate_min, p_draft, top_for_gate.size(), (int) id);
                    }
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);
                if (dp.branch_candidates != nullptr) {
                    dp.branch_candidates->emplace_back();
                    copy_branch_candidates(dp.branch_candidates->back(), top_for_gate, p_draft);
                }

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, sample_idx);
                GGML_ASSERT(h_row != nullptr);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            validate_real_mtp_batch("draft_iter", batch);
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        if (getenv("LLAMA_MTP_PROFILE")) {
            const uint16_t n_drafted = seq_id < (llama_seq_id) last_n_drafted.size() ? last_n_drafted[seq_id] : 0;
            const uint16_t n_rejected = n_drafted > n_accepted ? (uint16_t) (n_drafted - n_accepted) : 0;
            fprintf(stderr,
                    "MTP_PROFILE_ACCEPT seq_id=%d drafted=%u accepted=%u rejected=%u "
                    "total_generated=%zu total_accepted=%zu calls_accept=%zu p_min=%.6g n_max=%d backend_topk=%d sidecar=%d\n",
                    (int) seq_id, (unsigned) n_drafted, (unsigned) n_accepted, (unsigned) n_rejected,
                    n_gen_tokens, n_acc_tokens, n_call_accept,
                    (double) params.p_min, params.n_max, backend_topk_enabled ? 1 : 0, sidecar_enabled ? 1 : 0);
        }

        if (quality_trace_enabled && seq_id >= 0 && seq_id < (llama_seq_id) last_quality_trace.size()) {
            for (const auto & q : last_quality_trace[seq_id]) {
                const bool accepted = q.depth > 0 && q.depth <= (int) n_accepted;
                fprintf(stderr,
                        "MTP_PACKED16_QUALITY: phase=accept lane=%s seq_id=%d depth=%d accepted=%d draft_top1=%d target_raw_top1=%d raw_top1_match=%d "
                        "draft_p_top1=%.8g draft_margin=%.8g target_raw_margin=%.8g draft_source=%s\n",
                        quality_lane_name(), (int) seq_id, q.depth, accepted ? 1 : 0,
                        (int) q.draft_top1, (int) q.target_top1, q.top1_match ? 1 : 0,
                        q.draft_p_top1, q.draft_margin, q.target_margin, q.source);
            }
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_pre_norm() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_params_speculative & params,
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
        , params(params.ngram_map_k) {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n‑gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                this->params.n_match, mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.5) {
                sinfo.n_low++;
                if (sinfo.n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC: return "draft-jetspec";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;
        bool has_draft_jetspec = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC));

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            if (!has_draft_model_path) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft_simple = false;
            }
        } else if (has_draft_model_path && !has_mtp && !has_draft_eagle3 && !has_draft_jetspec) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
        }

        if (has_draft_jetspec) {
            const bool has_draft_head_model = params.draft.ctx_dft != nullptr || params.draft.model != nullptr;
            if (!common_speculative_env_enabled("LLAMA_JETSPEC_EXPERIMENTAL")) {
                LOG_WRN("%s: draft-jetspec requires LLAMA_JETSPEC_EXPERIMENTAL=1; disabling JetSpec before runtime execution\n", __func__);
                has_draft_jetspec = false;
            } else if (params.draft.ctx_tgt == nullptr || !has_draft_head_model) {
                LOG_WRN("%s: draft-jetspec requires a target context and a loaded draft-head model; disabling JetSpec before runtime execution\n", __func__);
                has_draft_jetspec = false;
            } else {
                llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, true, true);
                const int32_t tap_count = llama_get_jetspec_target_hidden_tap_count(params.draft.ctx_tgt);
                const int32_t tap_width = llama_get_jetspec_target_hidden_tap_width(params.draft.ctx_tgt);
                std::string preflight_reason;
                if (!common_speculative_jetspec_preflight(params.draft, tap_count, tap_width, preflight_reason)) {
                    llama_set_jetspec_target_hidden_taps(params.draft.ctx_tgt, false, true);
                    LOG_WRN("%s: draft-jetspec preflight failed (%s; taps=%d width=%d); disabling JetSpec before runtime execution\n",
                            __func__, preflight_reason.c_str(), tap_count, tap_width);
                    has_draft_jetspec = false;
                }
            }
        }

        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_jetspec) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_JETSPEC: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_jetspec>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            config.params, get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_process_with_pre_norm(common_speculative * spec, const llama_batch & batch, const float * h_pre_norm) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process_with_pre_norm(batch, h_pre_norm);
    }

    return result;
}

bool common_speculative_probe_descendants(
        common_speculative * spec,
        llama_seq_id seq_id,
        uint16_t reject_depth,
        llama_pos sibling_pos,
        llama_token sampled,
        int n_max,
        llama_tokens & result,
        std::vector<std::vector<common_speculative_branch_candidate>> * branch_candidates) {
    (void) spec;
    (void) seq_id;
    (void) reject_depth;
    (void) sibling_pos;
    (void) sampled;
    (void) n_max;
    result.clear();
    if (branch_candidates != nullptr) {
        branch_candidates->clear();
    }
    return false;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_pre_norm(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_pre_norm()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                if (dp.branch_candidates != nullptr) {
                    dp.branch_candidates->clear();
                }
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }
                if (dp.branch_candidates != nullptr && dp.branch_candidates->size() != result.size()) {
                    dp.branch_candidates->resize(result.size());
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                    impl->record_generated_depths(result.size());
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    // TODO: currently only the implementation that generated the draft is used to accept it
    //       however, some implementations (such as MTP) need to also "see" the accepted tokens
    //       extend `common_speculative_impl::accept()` with an extra argument `bool is_other` to
    //       inform the implementation if the accepted tokens are from another implementation and
    //       pass the accepted tokens to all remaining implementations using `is_other == true`
    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
            impl->record_accepted_depths(n_accepted);
        }

        impl->accept(seq_id, n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());

        if (!impl->n_gen_tokens_by_depth.empty()) {
            std::ostringstream oss;
            const size_t n_depth = impl->n_gen_tokens_by_depth.size();
            for (size_t i = 0; i < n_depth; ++i) {
                const size_t gen = impl->n_gen_tokens_by_depth[i];
                const size_t acc = i < impl->n_acc_tokens_by_depth.size() ? impl->n_acc_tokens_by_depth[i] : 0;
                if (i > 0) {
                    oss << ", ";
                }
                oss << "d" << (i + 1) << "=" << acc << "/" << gen;
            }
            LOG_INF("statistics %s-depth: %s\n",
                    common_speculative_type_to_str(impl->type).c_str(), oss.str().c_str());
        }
    }
}
