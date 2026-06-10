
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: llama_set_mtp_source
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static constexpr const char * MTP_EXACT_SINGLE_SLOT_VERIFY_MMVQ_FILTER = "ssm_out,attn_q,attn_v,ffn_down";
static constexpr const char * MTP_DENSE_FFN_SERIAL_COLUMNS_FILTER     = "ffn_up,ffn_gate,ffn_down";
static constexpr const char * MTP_PREFIX_EXACT_TAIL_MMVQ_FILTER      =
    "ffn_gate_inp,ffn_gate_up,ffn_gate,ffn_up,ffn_down,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,output";
static constexpr const char * MTP_PREFIX_ROWEQ_LAYER_FFN_MMVQ_FILTER  =
    "ffn_gate_inp,ffn_gate_up,ffn_gate,ffn_up,ffn_down,ffn_up_shexp,ffn_gate_shexp,ffn_down_shexp,ffn_gate_inp_shexp";

static bool mtp_env_like_value_disabled(const char * env) {
    return env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0;
}

static bool mtp_env_like_enabled(const char * name) {
    return !mtp_env_like_value_disabled(getenv(name));
}

static bool mtp_dense_ffn_serial_columns_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_DENSE_FFN_SERIAL_COLUMNS");
}

static bool mtp_prefix_exact_tail_batch_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH");
}

static bool mtp_prefix_exact_tail_filter_active() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_TAIL_ACTIVE");
}

static bool mtp_prefix_roweq_stage43_fused_router_topk_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_FUSED") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_WEIGHTS") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_FUSED_ROUTER_TOPK_WEIGHTS") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_TOPK_FUSED") ||
           mtp_env_like_enabled("LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_ROWEQ_ROUTER_TOPK_WEIGHT_FUSION_ACTIVE");
}

static bool mtp_prefix_roweq_stage42_router_topk_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_BISECT") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_TOPK_BISECT") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_MMVF") ||
           mtp_env_like_enabled("LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE") ||
           mtp_prefix_roweq_stage43_fused_router_topk_enabled();
}

static bool mtp_prefix_roweq_stage41_diag_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_DIAG") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_COMPONENT_BISECT") ||
           mtp_prefix_roweq_stage42_router_topk_enabled();
}

static bool mtp_prefix_roweq_layer_ffn_batch_enabled() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH") ||
           mtp_prefix_roweq_stage41_diag_enabled() ||
           mtp_prefix_roweq_stage42_router_topk_enabled();
}

static bool mtp_prefix_roweq_layer_filter_active() {
    return mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_LAYER_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_EXACT_ROWEQ_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_ACTIVE") ||
           mtp_env_like_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK_ACTIVE");
}

static bool mtp_filter_has_token(const std::string & filter, const std::string & token) {
    size_t pos = 0;
    while ((pos = filter.find(token, pos)) != std::string::npos) {
        const bool left_ok  = pos == 0 || filter[pos - 1] == ',';
        const size_t end    = pos + token.size();
        const bool right_ok = end == filter.size() || filter[end] == ',';
        if (left_ok && right_ok) {
            return true;
        }
        pos = end;
    }
    return false;
}

static void mtp_append_unique_filter_tokens(std::string & dst, const char * tokens) {
    if (tokens == nullptr || tokens[0] == '\0') {
        return;
    }

    const char * cur = tokens;
    while (*cur != '\0') {
        const char * comma = strchr(cur, ',');
        const size_t len = comma != nullptr ? (size_t) (comma - cur) : strlen(cur);
        if (len > 0) {
            const std::string token(cur, len);
            if (!mtp_filter_has_token(dst, token)) {
                if (!dst.empty()) {
                    dst += ',';
                }
                dst += token;
            }
        }
        if (comma == nullptr) {
            break;
        }
        cur = comma + 1;
    }
}

static const char * mtp_append_mmvq_serial_columns_filters(const char * base) {
    static thread_local std::string combined;
    combined = mtp_env_like_value_disabled(base) ? std::string() : std::string(base);

    if (mtp_dense_ffn_serial_columns_enabled()) {
        mtp_append_unique_filter_tokens(combined, MTP_DENSE_FFN_SERIAL_COLUMNS_FILTER);
    }

    if (mtp_prefix_exact_tail_filter_active()) {
        mtp_append_unique_filter_tokens(combined, MTP_PREFIX_EXACT_TAIL_MMVQ_FILTER);
    }

    if (mtp_prefix_roweq_layer_filter_active()) {
        mtp_append_unique_filter_tokens(combined, MTP_PREFIX_ROWEQ_LAYER_FFN_MMVQ_FILTER);
    }

    return combined.empty() ? nullptr : combined.c_str();
}

static bool mtp_exact_single_slot_verify_enabled() {
    const char * env = getenv("LLAMA_MTP_EXACT_SINGLE_SLOT_VERIFY");
    if (env != nullptr) {
        return atoi(env) != 0;
    }

    // Standard operating mode: one active user / one active speculative slot.
    // Default to the proven exact single-slot verifier policy without requiring
    // an env flag; set LLAMA_MTP_EXACT_SINGLE_SLOT_VERIFY=0 only for explicit
    // legacy/unsafe experiments.
    return true;
}

static bool mtp_exact_multi_slot_verify_per_slot_enabled() {
    const char * env = getenv("LLAMA_MTP_EXACT_MULTI_SLOT_VERIFY");
    return env != nullptr && (strcmp(env, "per_slot") == 0 || atoi(env) != 0);
}

static bool mtp_verify_compare_enabled() {
    const char * env = getenv("LLAMA_MTP_VERIFY_COMPARE");
    return env && atoi(env) != 0;
}

static bool mtp_verify_logit_compare_enabled() {
    const char * env = getenv("LLAMA_MTP_VERIFY_LOGIT_COMPARE");
    return env && atoi(env) != 0;
}

static void mtp_trace_verify_logits(
        const char * label,
        int slot_id,
        const char * scope,
        struct llama_context * ctx,
        int idx,
        int depth,
        llama_token draft_id,
        llama_token sampled_id,
        bool accepted,
        const std::vector<llama_token> & watch_tokens) {
    if (!mtp_verify_logit_compare_enabled()) {
        return;
    }

    const float * logits = llama_get_logits_ith(ctx, idx);
    if (!logits) {
        fprintf(stderr,
                "MTP_VERIFY_LOGITS: label=%s slot=%d scope=%s depth=%d status=missing_logits idx=%d draft=%d sampled=%d accepted=%d\n",
                label, slot_id, scope && scope[0] ? scope : "-", depth, idx,
                (int) draft_id, (int) sampled_id, accepted ? 1 : 0);
        return;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token top1_id = LLAMA_TOKEN_NULL;
    llama_token top2_id = LLAMA_TOKEN_NULL;
    float top1_logit = -INFINITY;
    float top2_logit = -INFINITY;

    auto better = [](float a_logit, llama_token a_id, float b_logit, llama_token b_id) {
        return a_logit > b_logit || (a_logit == b_logit && a_id < b_id);
    };

    auto consider_top2 = [&](llama_token id, float logit) {
        if (!std::isfinite(logit)) {
            return;
        }
        if (better(logit, id, top1_logit, top1_id)) {
            top2_id = top1_id;
            top2_logit = top1_logit;
            top1_id = id;
            top1_logit = logit;
        } else if (better(logit, id, top2_logit, top2_id)) {
            top2_id = id;
            top2_logit = logit;
        }
    };

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        consider_top2(token_id, logits[token_id]);
    }

    auto token_logit = [&](llama_token id) -> float {
        if (id < 0 || id >= n_vocab) {
            return NAN;
        }
        return logits[id];
    };

    const float draft_logit = token_logit(draft_id);
    const float sampled_logit = token_logit(sampled_id);
    int draft_rank = std::isfinite(draft_logit) ? 1 : -1;
    int sampled_rank = std::isfinite(sampled_logit) ? 1 : -1;
    int bad_logits = 0;
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        const float v = logits[token_id];
        if (!std::isfinite(v)) {
            ++bad_logits;
            continue;
        }
        if (draft_rank > 0 && better(v, token_id, draft_logit, draft_id)) {
            ++draft_rank;
        }
        if (sampled_rank > 0 && better(v, token_id, sampled_logit, sampled_id)) {
            ++sampled_rank;
        }
    }
    const float margin = (top1_id != LLAMA_TOKEN_NULL && top2_id != LLAMA_TOKEN_NULL) ? top1_logit - top2_logit : 0.0f;
    const float sampled_delta = std::isfinite(sampled_logit) && std::isfinite(top1_logit) ? top1_logit - sampled_logit : 0.0f;
    const float draft_delta = std::isfinite(draft_logit) && std::isfinite(top1_logit) ? top1_logit - draft_logit : 0.0f;

    fprintf(stderr,
            "MTP_VERIFY_LOGITS: label=%s slot=%d scope=%s depth=%d idx=%d draft=%d sampled=%d accepted=%d top1=%d top1_logit=%.8g top2=%d top2_logit=%.8g margin=%.8g draft_logit=%.8g draft_delta=%.8g draft_rank=%d sampled_logit=%.8g sampled_delta=%.8g sampled_rank=%d bad_logits=%d watch=[",
            label,
            slot_id,
            scope && scope[0] ? scope : "-",
            depth,
            idx,
            (int) draft_id,
            (int) sampled_id,
            accepted ? 1 : 0,
            (int) top1_id,
            top1_logit,
            (int) top2_id,
            top2_logit,
            margin,
            draft_logit,
            draft_delta,
            draft_rank,
            sampled_logit,
            sampled_delta,
            sampled_rank,
            bad_logits);

    std::vector<llama_token> printed;
    auto print_watch = [&](llama_token id) {
        if (id < 0 || id >= n_vocab) {
            return;
        }
        if (std::find(printed.begin(), printed.end(), id) != printed.end()) {
            return;
        }
        fprintf(stderr, "%s%d:%.8g", printed.empty() ? "" : ",", (int) id, logits[id]);
        printed.push_back(id);
    };
    print_watch(draft_id);
    print_watch(sampled_id);
    print_watch(top1_id);
    print_watch(top2_id);
    for (llama_token id : watch_tokens) {
        print_watch(id);
    }
    fprintf(stderr, "]\n");
}

static bool mtp_multi_slot_verify_compare_enabled() {
    const char * env = getenv("LLAMA_MTP_MULTI_SLOT_VERIFY_COMPARE");
    return mtp_verify_compare_enabled() || (env && atoi(env) != 0);
}

static bool mtp_multi_slot_state_compare_enabled() {
    const char * env = getenv("LLAMA_MTP_MULTI_SLOT_STATE_COMPARE");
    return env && atoi(env) != 0;
}

static bool mtp_multi_slot_live_rollback_enabled() {
    const char * env = getenv("LLAMA_MTP_MULTI_SLOT_LIVE_ROLLBACK");
    return env && atoi(env) != 0;
}

static bool mtp_target_serial_verify_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_SERIAL_VERIFY");
    return env && atoi(env) != 0;
}

static bool mtp_serial_verify_batch_dft_process_enabled() {
    const char * env = getenv("LLAMA_MTP_SERIAL_VERIFY_BATCH_DFT_PROCESS");
    return env && atoi(env) != 0;
}

static bool mtp_serial_verify_batch_dft_process_trace_enabled() {
    const char * env = getenv("LLAMA_MTP_SERIAL_VERIFY_BATCH_DFT_PROCESS_TRACE");
    return env && atoi(env) != 0;
}

static bool mtp_recurrent_prefix_v0_enabled() {
    const char * env = getenv("LLAMA_MTP_RECURRENT_PREFIX_V0");
    return env && atoi(env) != 0;
}

static bool mtp_serial_equiv_prefix_enabled() {
    const char * env = getenv("LLAMA_MTP_SERIAL_EQUIV_PREFIX");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_unsafe_requested() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_UNSAFE");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_unsafe_enabled() {
    if (mtp_target_batch_verify_unsafe_requested()) {
        return true;
    }

    if (mtp_exact_single_slot_verify_enabled() || mtp_exact_multi_slot_verify_per_slot_enabled()) {
        return true;
    }

    return false;
}

static bool mtp_target_batch_verify_replay_accepted_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_ACCEPTED");
    return env && atoi(env) != 0;
}

static bool mtp_prefix_accepted_row_only_commit_enabled() {
    const char * env = getenv("LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT");
    return env && atoi(env) != 0;
}

static int mtp_prefix_accepted_row_commit_verify_slots() {
    if (!mtp_prefix_accepted_row_only_commit_enabled()) {
        return 0;
    }
    const char * env = getenv("LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS");
    if (!env || env[0] == '\0') {
        return 1;
    }
    const int slots = atoi(env);
    return slots < 0 ? 0 : slots;
}

static bool mtp_target_batch_verify_replay_skip_dft_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_SKIP_DFT");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_replay_no_logits_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_NO_LOGITS");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_replay_from_prefix_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_FROM_PREFIX");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_replay_partial_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_REPLAY_PARTIAL");
    return env && atoi(env) != 0;
}

static bool mtp_target_batch_verify_ubatch1_enabled() {
    const char * env = getenv("LLAMA_MTP_TARGET_BATCH_VERIFY_UBATCH1");
    return env && atoi(env) != 0;
}

static bool mtp_spec_single_slot_only_enabled() {
    if (mtp_exact_single_slot_verify_enabled() && !mtp_exact_multi_slot_verify_per_slot_enabled()) {
        return true;
    }

    const char * env = getenv("LLAMA_MTP_SPEC_SINGLE_SLOT_ONLY");
    return env && atoi(env) != 0;
}

static bool mtp_rs_state_trace_enabled() {
    const char * env = getenv("LLAMA_MTP_RS_STATE_TRACE");
    return env && atoi(env) != 0;
}

static bool mtp_rs_window_trace_enabled() {
    const char * env = getenv("LLAMA_MTP_RS_WINDOW_TRACE");
    return env && atoi(env) != 0;
}

static bool mtp_rs_float_trace_all_enabled() {
    const char * env = getenv("LLAMA_MTP_RS_FLOAT_TRACE_ALL");
    return env && atoi(env) != 0;
}

static bool mtp_sync_after_target_decode_enabled() {
    const char * env = getenv("LLAMA_MTP_SYNC_AFTER_TARGET_DECODE");
    return env && atoi(env) != 0;
}

static bool mtp_roctx_enabled() {
    const char * env = getenv("LLAMA_MTP_ROCTX");
    return env && atoi(env) != 0;
}

static bool mtp_env_value_disabled(const char * env) {
    return env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0;
}

static bool mtp_model_arch_is(const llama_model * model, const char * arch) {
    if (model == nullptr || arch == nullptr) {
        return false;
    }

    char buf[64] = {};
    const int32_t len = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
    return len > 0 && strcmp(buf, arch) == 0;
}

enum mtp_verify_backend {
    MTP_VERIFY_BACKEND_NONE = 0,
    MTP_VERIFY_BACKEND_SERIAL_ORACLE,
    MTP_VERIFY_BACKEND_CAUSAL_BATCHED,
    MTP_VERIFY_BACKEND_RECURRENT_PREFIX,
    MTP_VERIFY_BACKEND_SERIAL_EQUIV_UBATCH1,
    MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX,
    MTP_VERIFY_BACKEND_PER_SLOT_BATCHED_COMMIT,
    MTP_VERIFY_BACKEND_REPLAY_ACCEPTED_COMMIT,
};

static const char * mtp_verify_backend_name(mtp_verify_backend backend) {
    switch (backend) {
        case MTP_VERIFY_BACKEND_NONE:                 return "none";
        case MTP_VERIFY_BACKEND_SERIAL_ORACLE:        return "serial_oracle";
        case MTP_VERIFY_BACKEND_CAUSAL_BATCHED:       return "causal_batched";
        case MTP_VERIFY_BACKEND_RECURRENT_PREFIX:     return "recurrent_prefix";
        case MTP_VERIFY_BACKEND_SERIAL_EQUIV_UBATCH1: return "serial_equiv_ubatch1";
        case MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX:  return "serial_equiv_prefix";
        case MTP_VERIFY_BACKEND_PER_SLOT_BATCHED_COMMIT: return "per_slot_batched_commit";
        case MTP_VERIFY_BACKEND_REPLAY_ACCEPTED_COMMIT:  return "replay_accepted_commit";
    }

    return "unknown";
}

struct mtp_verify_backend_choice {
    mtp_verify_backend backend = MTP_VERIFY_BACKEND_NONE;
    const char * reason = "none";
    bool serial_verify = false;
    bool per_slot_verify = false;
    bool recurrent_prefix_required = false;
};

static mtp_verify_backend_choice mtp_select_verify_backend(
        const llama_model * model,
        llama_context * ctx,
        bool per_slot_requested,
        bool target_serial_requested,
        bool replay_repair_requested,
        bool unsafe_requested,
        bool exact_verify,
        size_t n_draft) {
    const uint32_t n_rs_seq = ctx != nullptr ? llama_n_rs_seq(ctx) : 0;
    const bool has_recurrent_state = n_rs_seq > 0;
    const bool arch_qwen35moe = mtp_model_arch_is(model, "qwen35moe");
    const bool recurrent_prefix_required =
        exact_verify && has_recurrent_state && (arch_qwen35moe || n_draft > 1);

    mtp_verify_backend_choice choice;
    choice.recurrent_prefix_required = recurrent_prefix_required;

    if (per_slot_requested) {
        choice.backend = MTP_VERIFY_BACKEND_PER_SLOT_BATCHED_COMMIT;
        choice.reason = "explicit_multi_slot_per_slot";
        choice.per_slot_verify = true;
        return choice;
    }

    if (target_serial_requested) {
        choice.backend = MTP_VERIFY_BACKEND_SERIAL_ORACLE;
        choice.reason = "forced_serial_env";
        choice.serial_verify = true;
        return choice;
    }

    if (recurrent_prefix_required && mtp_serial_equiv_prefix_enabled() && !replay_repair_requested && !unsafe_requested && !mtp_target_batch_verify_ubatch1_enabled()) {
        // This backend is opt-in and fail-closed until the reserved
        // LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY graph has a real token-major
        // qwen35/qwen35moe implementation. Do not alias it to causal_batched.
        choice.backend = MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX;
        choice.reason = mtp_prefix_roweq_stage43_fused_router_topk_enabled() ?
            "exact_roweq_stage43_fused_router_topk_prefix_requested" :
            (mtp_prefix_roweq_stage42_router_topk_enabled() ?
            "exact_roweq_stage42_router_topk_prefix_requested" :
            (mtp_prefix_roweq_stage41_diag_enabled() ?
            "exact_roweq_stage41_component_bisect_prefix_requested" :
            (mtp_prefix_roweq_layer_ffn_batch_enabled() ?
            "exact_roweq_layer_ffn_prefix_requested" :
            (mtp_prefix_exact_tail_batch_enabled() ?
                "exact_token_major_prefix_graph_tail_batch_requested" :
                "exact_token_major_prefix_graph_requested"))));
        return choice;
    }

    const bool ubatch1_exact_requested = mtp_target_batch_verify_ubatch1_enabled() && mtp_target_batch_verify_replay_partial_enabled();
    if (recurrent_prefix_required && ubatch1_exact_requested) {
        choice.backend = MTP_VERIFY_BACKEND_SERIAL_EQUIV_UBATCH1;
        choice.reason = "ubatch1_partial_replay_serial_equiv";
        return choice;
    }

    if (recurrent_prefix_required && !replay_repair_requested && !unsafe_requested) {
        if (mtp_recurrent_prefix_v0_enabled()) {
            // v0 is intentionally serial-equivalent: it runs the existing token-by-token
            // verifier under the recurrent_prefix contract so later optimized kernels
            // can replace the implementation without changing scheduling/commit semantics.
            choice.backend = MTP_VERIFY_BACKEND_RECURRENT_PREFIX;
            choice.reason = "recurrent_prefix_v0_serial_equiv";
            choice.serial_verify = true;
            return choice;
        }

        choice.backend = MTP_VERIFY_BACKEND_SERIAL_ORACLE;
        choice.reason = "recurrent_prefix_required_serial_fallback";
        choice.serial_verify = true;
        return choice;
    }

    if (!replay_repair_requested && !mtp_target_batch_verify_unsafe_enabled() && has_recurrent_state) {
        choice.backend = MTP_VERIFY_BACKEND_SERIAL_ORACLE;
        choice.reason = "recurrent_batch_verify_disabled";
        choice.serial_verify = true;
        return choice;
    }

    if (replay_repair_requested) {
        choice.backend = MTP_VERIFY_BACKEND_REPLAY_ACCEPTED_COMMIT;
        choice.reason = recurrent_prefix_required ? "diagnostic_batched_then_serial_repair_recurrent" : "diagnostic_batched_then_serial_repair";
        return choice;
    }

    choice.backend = MTP_VERIFY_BACKEND_CAUSAL_BATCHED;
    choice.reason = unsafe_requested ? "unsafe_batched_env" : "causal_batched";
    return choice;
}

static bool mtp_verify_trace_enabled() {
    const char * env = getenv("LLAMA_MTP_VERIFY_TRACE");
    return mtp_verify_compare_enabled() || (env && atoi(env) != 0);
}

static const char * mtp_mmvq_serial_columns_filter_for_decode(int verifier_slots) {
    const bool exact_single_slot = mtp_exact_single_slot_verify_enabled() || mtp_exact_multi_slot_verify_per_slot_enabled();
    const char * base = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS");
    if (mtp_env_value_disabled(base)) {
        base = exact_single_slot ? MTP_EXACT_SINGLE_SLOT_VERIFY_MMVQ_FILTER : nullptr;
    }

    const char * selected = base;
    if (verifier_slots > 1) {
        const char * multi = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MULTI");
        if (!mtp_env_value_disabled(multi)) {
            selected = multi;
        }
    }

    if (mtp_env_value_disabled(selected) && !mtp_dense_ffn_serial_columns_enabled() &&
            !mtp_prefix_exact_tail_filter_active() && !mtp_prefix_roweq_layer_filter_active()) {
        return nullptr;
    }

    return mtp_append_mmvq_serial_columns_filters(selected);
}

struct mtp_env_flag_scope {
    const char * name   = nullptr;
    bool         active = false;
    bool         had    = false;
    std::string  old;

    mtp_env_flag_scope(const char * name, bool active) : name(name), active(active) {
        if (!active) {
            return;
        }

        if (const char * cur = getenv(name)) {
            had = true;
            old = cur;
        }

#if defined(_WIN32)
        _putenv_s(name, "1");
#else
        setenv(name, "1", 1);
#endif
    }

    ~mtp_env_flag_scope() {
        if (!active) {
            return;
        }

#if defined(_WIN32)
        _putenv_s(name, had ? old.c_str() : "");
#else
        if (had) {
            setenv(name, old.c_str(), 1);
        } else {
            unsetenv(name);
        }
#endif
    }
};

struct mtp_env_var_scope {
    const char * name   = nullptr;
    bool         active = false;
    bool         had    = false;
    std::string  old;

    mtp_env_var_scope(const char * name, const char * value, bool active) : name(name), active(active && value != nullptr) {
        if (!this->active) {
            return;
        }

        if (const char * cur = getenv(name)) {
            had = true;
            old = cur;
        }

#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    ~mtp_env_var_scope() {
        if (!active) {
            return;
        }

#if defined(_WIN32)
        _putenv_s(name, had ? old.c_str() : "");
#else
        if (had) {
            setenv(name, old.c_str(), 1);
        } else {
            unsetenv(name);
        }
#endif
    }
};

struct mtp_llama_batch_scope {
    llama_batch batch;

    mtp_llama_batch_scope(int32_t n_tokens_alloc, int32_t n_seq_max = 1) : batch(llama_batch_init(n_tokens_alloc, 0, n_seq_max)) {}

    ~mtp_llama_batch_scope() {
        llama_batch_free(batch);
    }
};

#if !defined(_WIN32)
struct mtp_roctx_api {
    using push_fn = int (*)(const char *);
    using pop_fn  = int (*)();

    void *  handle = nullptr;
    push_fn push   = nullptr;
    pop_fn  pop    = nullptr;
    bool    tried  = false;
};

static mtp_roctx_api & mtp_roctx_get_api() {
    static mtp_roctx_api api;
    if (api.tried) {
        return api;
    }
    api.tried = true;

    const char * libs[] = {
        "librocprofiler-sdk-roctx.so",
        "libroctx64.so",
    };
    for (const char * lib : libs) {
        api.handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
        if (api.handle != nullptr) {
            api.push = reinterpret_cast<mtp_roctx_api::push_fn>(dlsym(api.handle, "roctxRangePushA"));
            api.pop  = reinterpret_cast<mtp_roctx_api::pop_fn >(dlsym(api.handle, "roctxRangePop"));
            if (api.push != nullptr && api.pop != nullptr) {
                break;
            }
            dlclose(api.handle);
            api.handle = nullptr;
            api.push = nullptr;
            api.pop = nullptr;
        }
    }

    if (api.push == nullptr || api.pop == nullptr) {
        fprintf(stderr, "MTP_ROCTX: failed to load ROCTX range API; markers disabled\n");
    }
    return api;
}
#endif

class mtp_roctx_range {
public:
    explicit mtp_roctx_range(const char * name) {
#if !defined(_WIN32)
        if (!mtp_roctx_enabled()) {
            return;
        }
        auto & api = mtp_roctx_get_api();
        if (api.push != nullptr && api.pop != nullptr) {
            active = true;
            api.push(name);
        }
#else
        (void) name;
#endif
    }

    mtp_roctx_range(const mtp_roctx_range &) = delete;
    mtp_roctx_range & operator=(const mtp_roctx_range &) = delete;

    ~mtp_roctx_range() {
#if !defined(_WIN32)
        if (active) {
            auto & api = mtp_roctx_get_api();
            if (api.pop != nullptr) {
                api.pop();
            }
        }
#endif
    }

private:
    bool active = false;
};

struct mtp_rs_state_digest {
    size_t   size = 0;
    uint64_t hash = 1469598103934665603ULL;
};

static uint64_t mtp_fnv1a64(const uint8_t * data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= (uint64_t) data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static mtp_rs_state_digest mtp_digest_bytes(const std::vector<uint8_t> & data) {
    mtp_rs_state_digest res;
    res.size = data.size();
    res.hash = mtp_fnv1a64(data.data(), data.size());
    return res;
}

static std::vector<uint8_t> mtp_get_partial_seq_state_data(llama_context * ctx, llama_seq_id seq_id) {
    if (ctx == nullptr) {
        return {};
    }

    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
    if (size == 0) {
        return {};
    }

    std::vector<uint8_t> data(size);
    const size_t n = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq_id, flags);
    if (n != data.size()) {
        fprintf(stderr,
                "MTP_RS_STATE_TRACE: state_capture_error seq=%d expected_size=%zu got=%zu\n",
                (int) seq_id, data.size(), n);
        data.resize(std::min(n, data.size()));
    }
    return data;
}

static mtp_rs_state_digest mtp_digest_partial_seq_state(llama_context * ctx, llama_seq_id seq_id) {
    return mtp_digest_bytes(mtp_get_partial_seq_state_data(ctx, seq_id));
}

static int64_t mtp_first_diff_offset(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return (int64_t) i;
        }
    }
    return a.size() == b.size() ? -1 : (int64_t) n;
}

struct mtp_rs_state_cell_meta {
    llama_pos pos = 0;
    std::vector<llama_seq_id> seq_ids;
};

struct mtp_rs_state_span {
    char     kind = '?';
    uint32_t layer = 0;
    size_t   offset = 0;
    size_t   size = 0;
    size_t   row_size = 0;
};

struct mtp_rs_state_layout {
    bool ok = false;
    uint32_t cell_count = 0;
    uint32_t n_layer = 0;
    std::vector<mtp_rs_state_cell_meta> cells;
    std::vector<mtp_rs_state_span> spans;
    std::string error;
};

template <typename T>
static bool mtp_read_le(const std::vector<uint8_t> & data, size_t & off, T & out) {
    if (off + sizeof(T) > data.size()) {
        return false;
    }
    memcpy(&out, data.data() + off, sizeof(T));
    off += sizeof(T);
    return true;
}

static mtp_rs_state_layout mtp_parse_rs_state_layout(const std::vector<uint8_t> & data) {
    mtp_rs_state_layout layout;
    size_t off = 0;

    uint32_t magic = 0;
    llama_seq_id seq_id = 0;
    uint32_t cell_count = 0;
    if (!mtp_read_le(data, off, magic) || !mtp_read_le(data, off, seq_id) || !mtp_read_le(data, off, cell_count)) {
        layout.error = "short_header";
        return layout;
    }
    layout.cell_count = cell_count;

    layout.cells.reserve(cell_count);
    for (uint32_t i = 0; i < cell_count; ++i) {
        llama_pos pos = 0;
        uint32_t n_seq_id = 0;
        if (!mtp_read_le(data, off, pos) || !mtp_read_le(data, off, n_seq_id)) {
            layout.error = "short_cell_meta";
            return layout;
        }
        mtp_rs_state_cell_meta cell;
        cell.pos = pos;
        cell.seq_ids.reserve(n_seq_id);
        for (uint32_t j = 0; j < n_seq_id; ++j) {
            llama_seq_id seq_id_i = 0;
            if (!mtp_read_le(data, off, seq_id_i)) {
                layout.error = "short_cell_seq_ids";
                return layout;
            }
            cell.seq_ids.push_back(seq_id_i);
        }
        layout.cells.push_back(std::move(cell));
    }

    uint32_t s_trans = 0;
    uint32_t n_layer = 0;
    if (!mtp_read_le(data, off, s_trans) || !mtp_read_le(data, off, n_layer)) {
        layout.error = "short_data_header";
        return layout;
    }
    layout.n_layer = n_layer;

    struct raw_row_span {
        size_t row_size = 0;
        size_t offset = 0;
        size_t size = 0;
    };
    std::vector<raw_row_span> rows;
    while (off < data.size()) {
        if (data.size() - off < sizeof(int32_t) + sizeof(uint64_t)) {
            layout.error = "short_row_trailer";
            return layout;
        }
        int32_t type_i = 0;
        uint64_t row_size = 0;
        if (!mtp_read_le(data, off, type_i) || !mtp_read_le(data, off, row_size)) {
            layout.error = "short_row_header";
            return layout;
        }
        GGML_UNUSED(type_i);
        const size_t row_size_bytes = (size_t) row_size;
        const size_t span_size = row_size_bytes * (size_t) cell_count;
        if (off + span_size > data.size()) {
            layout.error = "short_row_data";
            return layout;
        }
        rows.push_back({row_size_bytes, off, span_size});
        off += span_size;
    }

    if (s_trans != 0) {
        layout.error = "s_trans_unsupported";
        return layout;
    }

    size_t split = rows.size();
    if (!rows.empty()) {
        const size_t r_row_size = rows[0].row_size;
        for (size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].row_size != r_row_size) {
                split = i;
                break;
            }
        }
        if (split == rows.size() && rows.size() > n_layer) {
            split = n_layer;
        }
    }

    for (size_t i = 0; i < rows.size(); ++i) {
        const char kind = i < split ? 'R' : 'S';
        const uint32_t layer = (uint32_t) (i < split ? i : i - split);
        layout.spans.push_back({kind, layer, rows[i].offset, rows[i].size, rows[i].row_size});
    }

    layout.ok = true;
    return layout;
}

static std::string mtp_rs_cells_summary(const mtp_rs_state_layout & layout) {
    std::string out;
    out += "[";
    for (size_t i = 0; i < layout.cells.size(); ++i) {
        const auto & cell = layout.cells[i];
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(i);
        out += ":pos=";
        out += std::to_string((long long) cell.pos);
        out += ":seqs=";
        if (cell.seq_ids.empty()) {
            out += "-";
        } else {
            for (size_t j = 0; j < cell.seq_ids.size(); ++j) {
                if (j != 0) {
                    out += "/";
                }
                out += std::to_string((int) cell.seq_ids[j]);
            }
        }
    }
    out += "]";
    return out;
}

static size_t mtp_rs_span_cell_for_float_index(const mtp_rs_state_span & span, size_t float_i) {
    const size_t row_floats = span.row_size / sizeof(float);
    if (row_floats == 0) {
        return (size_t) -1;
    }
    return float_i / row_floats;
}

static llama_pos mtp_rs_layout_cell_pos(const mtp_rs_state_layout & layout, size_t cell_i) {
    return cell_i < layout.cells.size() ? layout.cells[cell_i].pos : (llama_pos) -1;
}

static void mtp_trace_rs_window_layer0_component(
        int slot_id,
        size_t n_draft,
        size_t n_accepted,
        uint32_t n_rollback,
        uint32_t row_rollback,
        size_t prefix_tokens,
        llama_pos commit_pos,
        char kind,
        const mtp_rs_state_layout & la,
        const mtp_rs_state_layout & lb,
        const std::vector<uint8_t> & a,
        const std::vector<uint8_t> & b) {
    const mtp_rs_state_span * sa = nullptr;
    const mtp_rs_state_span * sb = nullptr;
    for (size_t i = 0; i < la.spans.size() && i < lb.spans.size(); ++i) {
        if (la.spans[i].kind == kind && lb.spans[i].kind == kind && la.spans[i].layer == 0 && lb.spans[i].layer == 0) {
            sa = &la.spans[i];
            sb = &lb.spans[i];
            break;
        }
    }
    if (sa == nullptr || sb == nullptr || sa->size != sb->size) {
        fprintf(stderr,
                "MTP_RS_WINDOW_FLOAT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d component=%c layer=0 unavailable\n",
                slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) commit_pos, kind);
        return;
    }

    const uint8_t * pa = a.data() + sa->offset;
    const uint8_t * pb = b.data() + sb->offset;
    const size_t n_float = sa->size / sizeof(float);
    size_t first_float_diff = (size_t) -1;
    float first_a = 0.0f;
    float first_b = 0.0f;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_abs_i = 0;
    for (size_t j = 0; j < n_float; ++j) {
        float fa;
        float fb;
        memcpy(&fa, pa + j * sizeof(float), sizeof(float));
        memcpy(&fb, pb + j * sizeof(float), sizeof(float));
        const float ad = fabsf(fa - fb);
        const float rd = ad / fmaxf(fmaxf(fabsf(fa), fabsf(fb)), 1.0e-9f);
        if (ad > max_abs) {
            max_abs = ad;
            max_abs_i = j;
        }
        if (rd > max_rel) {
            max_rel = rd;
        }
        if (first_float_diff == (size_t) -1 && fa != fb) {
            first_float_diff = j;
            first_a = fa;
            first_b = fb;
        }
    }

    const size_t first_report_i = first_float_diff == (size_t) -1 ? n_float : first_float_diff;
    const size_t first_cell_a = first_float_diff == (size_t) -1 ? (size_t) -1 : mtp_rs_span_cell_for_float_index(*sa, first_float_diff);
    const size_t max_cell_a = mtp_rs_span_cell_for_float_index(*sa, max_abs_i);
    const llama_pos first_cell_pos_a = mtp_rs_layout_cell_pos(la, first_cell_a);
    const llama_pos first_cell_pos_b = mtp_rs_layout_cell_pos(lb, first_cell_a);
    const llama_pos max_cell_pos_a = mtp_rs_layout_cell_pos(la, max_cell_a);
    const llama_pos max_cell_pos_b = mtp_rs_layout_cell_pos(lb, max_cell_a);

    fprintf(stderr,
            "MTP_RS_WINDOW_FLOAT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d component=%c layer=0 n_float=%zu first_float_diff=%zu first_cell=%zu first_cell_pos_a=%d first_cell_pos_b=%d first_a=%.9g first_b=%.9g max_abs=%.9g max_rel=%.9g max_abs_i=%zu max_cell=%zu max_cell_pos_a=%d max_cell_pos_b=%d\n",
            slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) commit_pos,
            kind, n_float, first_report_i, first_cell_a, (int) first_cell_pos_a, (int) first_cell_pos_b,
            first_a, first_b, max_abs, max_rel, max_abs_i, max_cell_a, (int) max_cell_pos_a, (int) max_cell_pos_b);
}

static void mtp_trace_rs_window_row(
        int slot_id,
        size_t n_draft,
        size_t n_accepted,
        uint32_t n_rollback,
        uint32_t row_rollback,
        size_t prefix_tokens,
        llama_pos commit_pos,
        const std::vector<uint8_t> & unsafe_data,
        const std::vector<uint8_t> & serial_data) {
    const mtp_rs_state_digest unsafe_digest = mtp_digest_bytes(unsafe_data);
    const mtp_rs_state_digest serial_digest = mtp_digest_bytes(serial_data);
    const bool match = unsafe_digest.size == serial_digest.size && unsafe_digest.hash == serial_digest.hash;
    const int64_t first_diff = mtp_first_diff_offset(unsafe_data, serial_data);
    const int unsafe_byte = first_diff >= 0 && (size_t) first_diff < unsafe_data.size() ? unsafe_data[(size_t) first_diff] : -1;
    const int serial_byte = first_diff >= 0 && (size_t) first_diff < serial_data.size() ? serial_data[(size_t) first_diff] : -1;

    fprintf(stderr,
            "MTP_RS_WINDOW_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d unsafe_size=%zu unsafe_hash=%016" PRIx64 " serial_size=%zu serial_hash=%016" PRIx64 " match=%d first_diff=%lld unsafe_byte=%d serial_byte=%d\n",
            slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) commit_pos,
            unsafe_digest.size, unsafe_digest.hash, serial_digest.size, serial_digest.hash,
            match ? 1 : 0, (long long) first_diff, unsafe_byte, serial_byte);

    const mtp_rs_state_layout la = mtp_parse_rs_state_layout(unsafe_data);
    const mtp_rs_state_layout lb = mtp_parse_rs_state_layout(serial_data);
    if (!la.ok || !lb.ok || la.spans.size() != lb.spans.size()) {
        fprintf(stderr,
                "MTP_RS_WINDOW_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d parse_error a_ok=%d b_ok=%d a_err=%s b_err=%s a_spans=%zu b_spans=%zu\n",
                slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) commit_pos,
                la.ok ? 1 : 0, lb.ok ? 1 : 0, la.error.c_str(), lb.error.c_str(), la.spans.size(), lb.spans.size());
        return;
    }

    const std::string cells_a = mtp_rs_cells_summary(la);
    const std::string cells_b = mtp_rs_cells_summary(lb);
    fprintf(stderr,
            "MTP_RS_WINDOW_CELLS_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d cell_count_a=%u cell_count_b=%u cells_a=%s cells_b=%s\n",
            slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) commit_pos,
            la.cell_count, lb.cell_count, cells_a.c_str(), cells_b.c_str());

    mtp_trace_rs_window_layer0_component(slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, commit_pos, 'R', la, lb, unsafe_data, serial_data);
    mtp_trace_rs_window_layer0_component(slot_id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, commit_pos, 'S', la, lb, unsafe_data, serial_data);
}

static void mtp_trace_rs_state_components(
        int slot_id,
        size_t n_draft,
        size_t n_accepted,
        uint32_t n_rollback,
        const char * label,
        const std::vector<uint8_t> & a,
        const std::vector<uint8_t> & b) {
    const mtp_rs_state_layout la = mtp_parse_rs_state_layout(a);
    const mtp_rs_state_layout lb = mtp_parse_rs_state_layout(b);
    if (!la.ok || !lb.ok || la.spans.size() != lb.spans.size()) {
        fprintf(stderr,
                "MTP_RS_COMPONENT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u label=%s parse_error a_ok=%d b_ok=%d a_err=%s b_err=%s a_spans=%zu b_spans=%zu\n",
                slot_id, n_draft, n_accepted, n_rollback, label,
                la.ok ? 1 : 0, lb.ok ? 1 : 0, la.error.c_str(), lb.error.c_str(), la.spans.size(), lb.spans.size());
        return;
    }

    int n_diff = 0;
    int n_diff_r = 0;
    int n_diff_s = 0;
    int n_print_r = 0;
    int n_print_s = 0;
    for (size_t i = 0; i < la.spans.size(); ++i) {
        const auto & sa = la.spans[i];
        const auto & sb = lb.spans[i];
        if (sa.kind != sb.kind || sa.layer != sb.layer || sa.size != sb.size) {
            fprintf(stderr,
                    "MTP_RS_COMPONENT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u label=%s span_mismatch index=%zu a=%c%u/%zu b=%c%u/%zu\n",
                    slot_id, n_draft, n_accepted, n_rollback, label, i,
                    sa.kind, sa.layer, sa.size, sb.kind, sb.layer, sb.size);
            return;
        }

        const uint8_t * pa = a.data() + sa.offset;
        const uint8_t * pb = b.data() + sb.offset;
        const uint64_t ha = mtp_fnv1a64(pa, sa.size);
        const uint64_t hb = mtp_fnv1a64(pb, sb.size);
        if (ha == hb) {
            continue;
        }

        n_diff++;
        int & n_diff_kind = sa.kind == 'R' ? n_diff_r : n_diff_s;
        int & n_print_kind = sa.kind == 'R' ? n_print_r : n_print_s;
        n_diff_kind++;
        if (n_print_kind < 8) {
            int64_t first_diff = -1;
            for (size_t j = 0; j < sa.size; ++j) {
                if (pa[j] != pb[j]) {
                    first_diff = (int64_t) j;
                    break;
                }
            }
            const int a_byte = first_diff >= 0 ? pa[first_diff] : -1;
            const int b_byte = first_diff >= 0 ? pb[first_diff] : -1;
            fprintf(stderr,
                    "MTP_RS_COMPONENT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u label=%s component=%c layer=%u size=%zu hash_a=%016" PRIx64 " hash_b=%016" PRIx64 " first_diff=%lld a_byte=%d b_byte=%d\n",
                    slot_id, n_draft, n_accepted, n_rollback, label,
                    sa.kind, sa.layer, sa.size, ha, hb, (long long) first_diff, a_byte, b_byte);
            if (sa.size >= sizeof(float) && (sa.layer == 0 || mtp_rs_float_trace_all_enabled())) {
                size_t first_float_diff = (size_t) -1;
                float first_a = 0.0f;
                float first_b = 0.0f;
                float max_abs = 0.0f;
                float max_rel = 0.0f;
                size_t max_abs_i = 0;
                const size_t n_float = sa.size / sizeof(float);
                for (size_t j = 0; j < n_float; ++j) {
                    float fa;
                    float fb;
                    memcpy(&fa, pa + j * sizeof(float), sizeof(float));
                    memcpy(&fb, pb + j * sizeof(float), sizeof(float));
                    const float ad = fabsf(fa - fb);
                    const float rd = ad / fmaxf(fmaxf(fabsf(fa), fabsf(fb)), 1.0e-9f);
                    if (ad > max_abs) {
                        max_abs = ad;
                        max_abs_i = j;
                    }
                    if (rd > max_rel) {
                        max_rel = rd;
                    }
                    if (first_float_diff == (size_t) -1 && fa != fb) {
                        first_float_diff = j;
                        first_a = fa;
                        first_b = fb;
                    }
                }
                fprintf(stderr,
                        "MTP_RS_FLOAT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u label=%s component=%c layer=%u n_float=%zu first_float_diff=%zu first_a=%.9g first_b=%.9g max_abs=%.9g max_rel=%.9g max_abs_i=%zu first12_a=[",
                        slot_id, n_draft, n_accepted, n_rollback, label, sa.kind, sa.layer, n_float,
                        first_float_diff == (size_t) -1 ? n_float : first_float_diff,
                        first_a, first_b, max_abs, max_rel, max_abs_i);
                const size_t n_print = std::min<size_t>(12, n_float);
                for (size_t j = 0; j < n_print; ++j) {
                    float fa;
                    memcpy(&fa, pa + j * sizeof(float), sizeof(float));
                    fprintf(stderr, "%s%.9g", j == 0 ? "" : ",", fa);
                }
                fprintf(stderr, "] first12_b=[");
                for (size_t j = 0; j < n_print; ++j) {
                    float fb;
                    memcpy(&fb, pb + j * sizeof(float), sizeof(float));
                    fprintf(stderr, "%s%.9g", j == 0 ? "" : ",", fb);
                }
                fprintf(stderr, "]\n");
            }
            n_print_kind++;
        }
    }

    fprintf(stderr,
            "MTP_RS_COMPONENT_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u label=%s components=%zu diff_components=%d diff_r=%d diff_s=%d printed_r=%d printed_s=%d cell_count=%u n_layer=%u\n",
            slot_id, n_draft, n_accepted, n_rollback, label,
            la.spans.size(), n_diff, n_diff_r, n_diff_s, n_print_r, n_print_s, la.cell_count, la.n_layer);
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    common_speculative * spec;

    bool spec_target_serial_verify   = false;
    bool spec_target_per_slot_verify = false;
    mtp_verify_backend spec_verify_backend = MTP_VERIFY_BACKEND_NONE;
    const char * spec_verify_backend_reason = "none";
    bool spec_verify_recurrent_prefix_required = false;
    std::string spec_gdn_compare_scope;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        spec_target_serial_verify = false;
        spec_target_per_slot_verify = false;
        spec_verify_backend = MTP_VERIFY_BACKEND_NONE;
        spec_verify_backend_reason = "none";
        spec_verify_recurrent_prefix_required = false;
        spec_gdn_compare_scope.clear();

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_pre_norm() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_pre_norm(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            const bool serial_verify = spec_target_serial_verify || spec_target_per_slot_verify || mtp_target_serial_verify_enabled();
            if (const char * env = getenv("LLAMA_MTP_SERIAL_VERIFY_BATCH_DFT_PROCESS_TRACE"); env && atoi(env) != 0) {
                fprintf(stderr,
                        "MTP_SERIAL_BATCH_DFT_TRACE: phase=update_batch_begin slot=%d batch_tokens_before=%d prompt_next=%d sampled=%d draft=%zu serial_verify=%d slot_serial=%d slot_per_slot=%d env_serial=%d\n",
                        id, (int) batch.n_tokens, (int) prompt.tokens.pos_next(), (int) sampled, spec_draft.size(),
                        serial_verify ? 1 : 0, spec_target_serial_verify ? 1 : 0, spec_target_per_slot_verify ? 1 : 0,
                        mtp_target_serial_verify_enabled() ? 1 : 0);
            }
            spec_i_batch.push_back(batch.n_tokens);
            if (!serial_verify) {
                for (size_t i = 0; i < spec_draft.size(); i++) {
                    spec_i_batch.push_back(batch.n_tokens + i + 1);
                }
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            if (!serial_verify) {
                for (auto token : spec_draft) {
                    common_batch_add(batch, token, pos0++, { this->id }, true);
                }
            }
            if (const char * env = getenv("LLAMA_MTP_SERIAL_VERIFY_BATCH_DFT_PROCESS_TRACE"); env && atoi(env) != 0) {
                fprintf(stderr,
                        "MTP_SERIAL_BATCH_DFT_TRACE: phase=update_batch_end slot=%d batch_tokens_after=%d first_added=%d last_added=%d prompt_next_before_push=%d spec_i_batch=%zu\n",
                        id, (int) batch.n_tokens,
                        batch.n_tokens > 0 ? (int) batch.pos[std::max(0, batch.n_tokens - (int) (serial_verify ? 1 : spec_draft.size() + 1))] : -1,
                        batch.n_tokens > 0 ? (int) batch.pos[batch.n_tokens - 1] : -1,
                        (int) prompt.tokens.pos_next(), spec_i_batch.size());
            }
        }

        prompt.tokens.push_back(sampled);
        if (!(spec_target_serial_verify || spec_target_per_slot_verify || mtp_target_serial_verify_enabled())) {
            prompt.tokens.insert(spec_draft);
        }
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total
            );
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // Prompt cache mode derived from seq_rm capability
    enum slot_cache_mode {
        SLOT_CACHE_PARTIAL_SEQ_RM,
        SLOT_CACHE_CHECKPOINT_FULL,
        SLOT_CACHE_FULL_REPROCESS,
    } cache_mode = SLOT_CACHE_FULL_REPROCESS;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    void destroy() {
        // Speculative state owns samplers and non-owning context pointers. Drop it
        // before freeing ctx_dft/ctx_tgt; MTP teardown can otherwise observe stale
        // target/draft contexts during shutdown or sleep/resume.
        for (server_slot & slot : slots) {
            slot.spec = nullptr;
            slot.ctx_tgt = nullptr;
            slot.ctx_dft = nullptr;
            slot.spec_draft.clear();
            slot.spec_prompt.clear();
            slot.spec_i_batch.clear();
            slot.spec_ckpt.clear();
        }
        spec.reset();
        params_base.speculative.draft.ctx_tgt = nullptr;
        params_base.speculative.draft.ctx_dft = nullptr;

        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            // TODO: find a better way to expose that the cache is shared
            if (spec_mtp) {
                params_dft.cache_type_k = params_base.cache_type_k;
                params_dft.cache_type_v = params_base.cache_type_v;
            } else {
                params_dft.cache_type_k = params_spec.cache_type_k;
                params_dft.cache_type_v = params_spec.cache_type_v;
            }

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq = 0;
            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

            if (spec_mtp) {
                // MTP draft must know its target before the first decode
                llama_set_mtp_source(ctx_dft.get(), ctx_tgt);
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end()) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            auto cparams_mtp = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.n_rs_seq = 0;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            // wire the source before any decode (the seq-rm probe below
            // triggers sched_reserve which needs src for Gemma4-style MTP)
            llama_set_mtp_source(ctx_dft.get(), ctx_tgt);

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // Resolve prompt cache mode
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
            ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            cache_mode = SLOT_CACHE_PARTIAL_SEQ_RM;
        } else if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            cache_mode = SLOT_CACHE_CHECKPOINT_FULL;
        } else {
            cache_mode = SLOT_CACHE_FULL_REPROCESS;
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s", "--cache-idle-slots requires --kv-unified, disabling\n");
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        int mtp_active_spec_slots_for_policy = 0;
        if (spec && (mtp_spec_single_slot_only_enabled() || mtp_exact_multi_slot_verify_per_slot_enabled())) {
            for (const auto & slot : slots) {
                if (slot.is_processing() && slot.can_speculate()) {
                    mtp_active_spec_slots_for_policy++;
                }
            }
        }
        const bool mtp_disable_spec_multi_slot = mtp_spec_single_slot_only_enabled() && mtp_active_spec_slots_for_policy > 1;
        const bool mtp_per_slot_verify_multi = mtp_exact_multi_slot_verify_per_slot_enabled() && mtp_active_spec_slots_for_policy > 1;

        // determine which slots are generating and drafting
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                if (mtp_disable_spec_multi_slot && slot.can_speculate()) {
                    slot.spec_draft.clear();
                    slot.spec_i_batch.clear();
                    slot.spec_ckpt.clear();
                    continue;
                }

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        {
            const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
            const int64_t mtp_cycle_draft_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
            mtp_roctx_range roctx_mtp_draft("MTP:draft");
            common_speculative_draft(spec.get());
            if (mtp_cycle_trace && !drafting.empty()) {
                size_t mtp_cycle_draft_tokens = 0;
                for (const auto * slot_ptr : drafting) {
                    mtp_cycle_draft_tokens += slot_ptr->spec_draft.size();
                }
                const double draft_wall_ms = double(ggml_time_us() - mtp_cycle_draft_t0) / 1000.0;
                fprintf(stderr,
                        "MTP_CYCLE_TRACE: phase=draft_wall slots=%zu draft_tokens=%zu draft_wall_ms=%.3f\n",
                        drafting.size(), mtp_cycle_draft_tokens, draft_wall_ms);
            }
        }

        // make checkpoints if needed
        for (auto * slot_ptr : drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }

                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt)) ||
                    mtp_target_batch_verify_replay_accepted_enabled() ||
                    mtp_target_batch_verify_replay_partial_enabled() ||
                    mtp_prefix_accepted_row_only_commit_enabled() ||
                    mtp_verify_compare_enabled() ||
                    mtp_per_slot_verify_multi;

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
                    const int64_t mtp_cycle_ckpt_t0 = mtp_cycle_trace ? ggml_time_us() : 0;

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                    if (mtp_cycle_trace) {
                        const double ckpt_ms = double(ggml_time_us() - mtp_cycle_ckpt_t0) / 1000.0;
                        fprintf(stderr,
                                "MTP_CYCLE_TRACE: phase=checkpoint_tgt slot=%d draft=%zu ckpt_size=%zu ckpt_ms=%.3f\n",
                                slot.id, draft.size(), ckpt.data_tgt.size(), ckpt_ms);
                    }

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
            }
        }

        // update the batch with the sampled/drafted tokens
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;

            // One verifier contract, multiple backends:
            //   - serial_oracle: universal exact fallback/oracle;
            //   - causal_batched: fast path for ordinary KV-only target verification;
            //   - recurrent_prefix: required future fast path for qwen35/qwen35moe-style recurrent state;
            //   - serial_equiv_ubatch1: experimental exact recurrent verifier that forces verifier decode
            //     through one-token ubatches and serial-replays only rejected/partial prefixes;
            //   - serial_equiv_prefix: opt-in, fail-closed placeholder for the future token-major graph.
            // Until a faster recurrent-prefix backend exists, exact recurrent qwen35moe / multi-draft verification
            // deliberately selects serial_oracle unless an explicit experimental backend is requested.
            const bool replay_accepted = mtp_target_batch_verify_replay_accepted_enabled();
            const bool replay_partial = mtp_target_batch_verify_replay_partial_enabled();
            const bool replay_repair_requested = replay_accepted || (replay_partial && mtp_target_batch_verify_ubatch1_enabled());
            const bool unsafe_requested = mtp_target_batch_verify_unsafe_requested();
            const bool exact_verify = mtp_exact_single_slot_verify_enabled() || mtp_exact_multi_slot_verify_per_slot_enabled();
            const mtp_verify_backend_choice verify_choice = mtp_select_verify_backend(
                    model_tgt,
                    ctx_tgt,
                    mtp_per_slot_verify_multi && !slot.spec_draft.empty(),
                    mtp_target_serial_verify_enabled(),
                    replay_repair_requested,
                    unsafe_requested,
                    exact_verify,
                    slot.spec_draft.size());
            slot.spec_verify_backend = verify_choice.backend;
            slot.spec_verify_backend_reason = verify_choice.reason;
            slot.spec_verify_recurrent_prefix_required = verify_choice.recurrent_prefix_required;
            slot.spec_target_per_slot_verify = verify_choice.per_slot_verify;
            slot.spec_target_serial_verify = verify_choice.serial_verify;

            if (mtp_verify_trace_enabled() && !slot.spec_draft.empty()) {
                fprintf(stderr,
                        "MTP_VERIFY_BACKEND: slot=%d backend=%s reason=%s draft=%zu n_rs_seq=%u qwen35moe=%d exact=%d replay_accepted=%d replay_partial=%d replay_repair=%d unsafe_requested=%d compare=%d recurrent_prefix_required=%d recurrent_prefix_v0=%d serial_equiv_prefix=%d prefix_exact_tail_batch=%d prefix_roweq_layer_ffn_batch=%d\n",
                        slot.id,
                        mtp_verify_backend_name(slot.spec_verify_backend),
                        slot.spec_verify_backend_reason,
                        slot.spec_draft.size(),
                        llama_n_rs_seq(ctx_tgt),
                        mtp_model_arch_is(model_tgt, "qwen35moe") ? 1 : 0,
                        exact_verify ? 1 : 0,
                        replay_accepted ? 1 : 0,
                        replay_partial ? 1 : 0,
                        replay_repair_requested ? 1 : 0,
                        unsafe_requested ? 1 : 0,
                        mtp_verify_compare_enabled() ? 1 : 0,
                        slot.spec_verify_recurrent_prefix_required ? 1 : 0,
                        mtp_recurrent_prefix_v0_enabled() ? 1 : 0,
                        mtp_serial_equiv_prefix_enabled() ? 1 : 0,
                        mtp_prefix_exact_tail_batch_enabled() ? 1 : 0,
                        mtp_prefix_roweq_layer_ffn_batch_enabled() ? 1 : 0);
            }

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }

                                // For non-PARTIAL cache modes: only safe to reuse cache
                                // when the new prompt exactly extends the cached prefix.
                                // If we'd need to trim the suffix, force full reprocessing.
                                if (cache_mode != SLOT_CACHE_PARTIAL_SEQ_RM) {
                                    if (n_past > slot.task->n_tokens()) {
                                        SLT_WRN(slot, "%s", "non-PARTIAL cache mode: cached suffix would need trimming, forcing n_past=0\n");
                                        n_past = 0;
                                        slot.n_prompt_tokens_cache = 0;
                                        slot.n_prompt_tokens_processed = 0;
                                        // Clear KV cache state so batch init starts from empty
                                        common_context_seq_rm(ctx_tgt, slot.id, 0, -1);
                                        if (ctx_dft) common_context_seq_rm(ctx_dft.get(), slot.id, 0, -1);
                                    }
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa);

                            if (n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                        if (cache_mode != SLOT_CACHE_PARTIAL_SEQ_RM) {
                                            slot.n_prompt_tokens_cache = 0;
                                            slot.n_prompt_tokens_processed = 0;
                                        }
                                        // p0=0 is safe for all seq_rm types
                                        common_context_seq_rm(ctx_tgt, slot.id, 0, -1);
                                        if (ctx_dft) common_context_seq_rm(ctx_dft.get(), slot.id, 0, -1);
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            if (cache_mode == SLOT_CACHE_PARTIAL_SEQ_RM) {
                                n_past--;
                                SLT_WRN(slot, "n_past was set to %d\n", n_past);
                            } else {
                                // For CHECKPOINT_FULL/FULL_REPROCESS: cannot do partial removal.
                                // Force full reprocessing when we'd need to trim the suffix.
                                SLT_WRN(slot, "%s", "forcing full re-processing (context does not support partial seq_rm)\n");
                                n_past = 0;
                                slot.n_prompt_tokens_cache = 0;
                                slot.n_prompt_tokens_processed = 0;
                            }
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end) cache_mode=%d\n", slot.prompt.n_tokens(), p0, (int)cache_mode);

                    // Only call seq_rm for PARTIAL mode. CHECKPOINT_FULL and
                    // FULL_REPROCESS manage state through checkpoints or full reprocessing.
                    if (cache_mode == SLOT_CACHE_PARTIAL_SEQ_RM) {
                        common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                        if (ctx_dft) {
                            common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx_tgt, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            res = input_tokens.process_chunk(ctx_dft.get(), mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_pre_norm into ctx_dft.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed =
                            !n_before_user_known ||
                            is_on_user ||
                            (is_after_user && near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        int mtp_target_verify_slots_total = 0;
        int mtp_active_speculative_slots_total = 0;
        for (const auto & slot : slots) {
            if (slot.is_processing() && slot.can_speculate()) {
                mtp_active_speculative_slots_total++;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.can_speculate() && !slot.spec_target_serial_verify && !slot.spec_target_per_slot_verify && !slot.spec_i_batch.empty()) {
                mtp_target_verify_slots_total++;
            }
        }

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
            int mtp_target_verify_slots = 0;
            int mtp_target_prefix_verify_slots = 0;
            int mtp_cycle_spec_slots = 0;
            size_t mtp_cycle_draft_tokens = 0;
            std::vector<server_slot *> mtp_target_verify_slot_ptrs;
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_target_serial_verify || slot.spec_target_per_slot_verify || slot.spec_i_batch.empty()) {
                    continue;
                }

                bool slot_in_batch_view = false;
                for (const int32_t i_batch : slot.spec_i_batch) {
                    if (i_batch >= i && i_batch < i + n_tokens) {
                        slot_in_batch_view = true;
                        break;
                    }
                }
                if (!slot_in_batch_view) {
                    continue;
                }

                mtp_target_verify_slots++;
                if (slot.spec_verify_backend == MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX) {
                    mtp_target_prefix_verify_slots++;
                }
                mtp_target_verify_slot_ptrs.push_back(&slot);
                if (mtp_cycle_trace) {
                    mtp_cycle_spec_slots++;
                    mtp_cycle_draft_tokens += slot.spec_draft.size();
                }
            }
            static uint64_t mtp_gdn_compare_scope_seq = 1;
            std::string mtp_gdn_compare_scope;
            if (!mtp_target_verify_slot_ptrs.empty()) {
                mtp_gdn_compare_scope = "mtp_verify_" + std::to_string(mtp_gdn_compare_scope_seq++);
                for (auto * slot_ptr : mtp_target_verify_slot_ptrs) {
                    slot_ptr->spec_gdn_compare_scope = mtp_gdn_compare_scope;
                }
            }
            const int64_t mtp_cycle_decode_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
            int ret = 0;
            {
                mtp_roctx_range roctx_mtp_target_decode("MTP:target_decode");
                const bool mtp_roweq_layer_prefix_active = mtp_target_prefix_verify_slots > 0 && mtp_prefix_roweq_layer_ffn_batch_enabled();
                const bool mtp_exact_tail_prefix_active = mtp_target_prefix_verify_slots > 0 && mtp_prefix_exact_tail_batch_enabled() && !mtp_roweq_layer_prefix_active;
                const bool mtp_roweq_prefix_policy_active = mtp_exact_tail_prefix_active || mtp_roweq_layer_prefix_active;
                mtp_env_flag_scope mtp_exact_tail_prefix_active_scope("LLAMA_MTP_PREFIX_EXACT_TAIL_ACTIVE", mtp_exact_tail_prefix_active);
                mtp_env_flag_scope mtp_roweq_layer_prefix_active_scope("LLAMA_MTP_PREFIX_ROWEQ_LAYER_ACTIVE", mtp_roweq_layer_prefix_active);
                mtp_env_flag_scope mtp_roweq_layer_prefix_compat_scope("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_ACTIVE", mtp_roweq_layer_prefix_active);
                mtp_env_flag_scope mtp_roweq_stage41_prefix_active_scope("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_ACTIVE", mtp_roweq_layer_prefix_active && mtp_prefix_roweq_stage41_diag_enabled());
                mtp_env_flag_scope mtp_roweq_stage42_router_topk_active_scope("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_ACTIVE", mtp_roweq_layer_prefix_active && mtp_prefix_roweq_stage42_router_topk_enabled());
                mtp_env_flag_scope mtp_roweq_stage43_fused_router_topk_active_scope("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK_ACTIVE", mtp_roweq_layer_prefix_active && mtp_prefix_roweq_stage43_fused_router_topk_enabled());
                mtp_env_flag_scope mtp_roweq_router_mmvf_active_scope("LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE", mtp_roweq_layer_prefix_active && mtp_prefix_roweq_stage42_router_topk_enabled());
                mtp_env_flag_scope mtp_roweq_router_topk_fused_active_scope("LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE", mtp_roweq_layer_prefix_active && mtp_prefix_roweq_stage43_fused_router_topk_enabled());
                const int mtp_mmvq_serial_columns_policy_slots = std::max(
                        std::max(std::max(mtp_target_verify_slots, mtp_target_verify_slots_total), mtp_active_speculative_slots_total),
                        params_base.n_parallel > 1 ? 2 : 0);
                const char * mtp_mmvq_serial_columns_filter = mtp_mmvq_serial_columns_filter_for_decode(mtp_mmvq_serial_columns_policy_slots);
                const bool mtp_mmvq_serial_columns_active = mtp_target_verify_slots > 0 && !mtp_env_value_disabled(mtp_mmvq_serial_columns_filter);
                mtp_env_flag_scope mtp_mmvq_serial_columns_active_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE", mtp_mmvq_serial_columns_active);
                mtp_env_var_scope mtp_mmvq_serial_columns_filter_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER", mtp_mmvq_serial_columns_filter, mtp_mmvq_serial_columns_active);
                mtp_env_flag_scope mtp_mmvq_serial_columns_ids_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS", mtp_roweq_prefix_policy_active);
                mtp_env_flag_scope mtp_mmvq_serial_columns_single_launch_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH", mtp_roweq_prefix_policy_active);
                mtp_env_flag_scope mtp_decode_ubatch1_scope("LLAMA_MTP_DECODE_FORCE_UBATCH_ONE", mtp_target_verify_slots > 0 && mtp_target_batch_verify_ubatch1_enabled());
                mtp_env_var_scope mtp_gdn_compare_scope_env("LLAMA_MTP_GDN_INPUT_TRACE_COMPARE_SCOPE", mtp_gdn_compare_scope.c_str(), !mtp_gdn_compare_scope.empty());
                ret = mtp_target_prefix_verify_slots > 0
                    ? llama_decode_prefix_verify(ctx_tgt, batch_view)
                    : llama_decode(ctx_tgt, batch_view);
                if (mtp_sync_after_target_decode_enabled()) {
                    llama_synchronize(ctx_tgt);
                }
            }
            if (mtp_cycle_trace && mtp_cycle_spec_slots > 0) {
                const double decode_ms = double(ggml_time_us() - mtp_cycle_decode_t0) / 1000.0;
                fprintf(stderr,
                        "MTP_CYCLE_TRACE: phase=target_decode batch_tokens=%d batch_offset=%d spec_slots=%d draft_tokens=%zu decode_ms=%.3f\n",
                        (int) batch_view.n_tokens, (int) i, mtp_cycle_spec_slots, mtp_cycle_draft_tokens, decode_ms);
            }

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            {
                mtp_roctx_range roctx_mtp_spec_process("MTP:spec_process");
                if (!common_speculative_process(spec.get(), batch_view)) {
                    SRV_ERR("%s", "failed to process speculative batch\n");

                    // TODO: handle error
                    break;
                }
            }


            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            // For serial speculative verification with multiple active slots, sample the
            // first target token for every serial-verifier slot before any per-slot serial
            // replay calls llama_decode() again. Each replay overwrites the context output
            // rows, so deferring the second slot's initial sample can make its original
            // batch index unavailable (for example get_logits_ith(1) after a one-token
            // serial replay). Batched verifier acceptance samples all rows before replay;
            // do the same for the safe serial path.
            std::vector<llama_token> mtp_serial_verify_initial_ids(slots.size(), LLAMA_TOKEN_NULL);
            std::vector<uint8_t> mtp_serial_verify_initial_sampled(slots.size(), 0);
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }
                if (!(slot.spec_target_serial_verify || mtp_target_serial_verify_enabled())) {
                    continue;
                }
                if (slot.id < 0 || (size_t) slot.id >= mtp_serial_verify_initial_ids.size()) {
                    continue;
                }

                GGML_ASSERT(slot.spec_i_batch.size() == 1);
                const int32_t i_batch = slot.spec_i_batch[0];
                if (i_batch < i || i_batch >= i + n_tokens) {
                    continue;
                }

                const int tok_idx = i_batch - i;
                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
                common_sampler_accept(slot.smpl.get(), id, true);
                mtp_serial_verify_initial_ids[slot.id] = id;
                mtp_serial_verify_initial_sampled[slot.id] = 1;
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                if (slot.spec_target_per_slot_verify) {
                    GGML_ASSERT(slot.spec_i_batch.size() == 1);

                    const auto & ckpt = slot.spec_ckpt;
                    if (ckpt.data_tgt.empty()) {
                        SRV_ERR("%s", "MTP per-slot verifier requested but target checkpoint is empty\n");
                        slot.spec_i_batch.clear();
                        continue;
                    }

                    const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
                    const bool mtp_compare_verify = mtp_multi_slot_verify_compare_enabled();
                    const bool mtp_state_compare = mtp_multi_slot_state_compare_enabled();
                    const bool mtp_live_rollback = mtp_multi_slot_live_rollback_enabled() && !mtp_compare_verify && !mtp_state_compare;
                    double mtp_cycle_per_slot_decode_ms = 0.0;
                    double mtp_cycle_per_slot_process_ms = 0.0;
                    mtp_llama_batch_scope mtp_per_slot_batch_scope((int32_t) n_draft + 1);
                    llama_batch & mtp_batch = mtp_per_slot_batch_scope.batch;

                    ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);

                    if (slot.ctx_dft) {
                        if (!ckpt.data_dft.empty()) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }
                        common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                    }

                    common_batch_clear(mtp_batch);
                    std::vector<int32_t> mtp_per_slot_i_batch;
                    mtp_per_slot_i_batch.reserve(n_draft + 1);
                    llama_pos mtp_per_slot_pos = (llama_pos) ckpt.n_tokens;
                    mtp_per_slot_i_batch.push_back(mtp_batch.n_tokens);
                    common_batch_add(mtp_batch, slot.sampled, mtp_per_slot_pos++, { slot.id }, true);
                    for (llama_token tok : slot.spec_draft) {
                        mtp_per_slot_i_batch.push_back(mtp_batch.n_tokens);
                        common_batch_add(mtp_batch, tok, mtp_per_slot_pos++, { slot.id }, true);
                    }

                    int ret_per_slot = 0;
                    const int64_t mtp_cycle_per_slot_decode_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    {
                        mtp_roctx_range roctx_mtp_per_slot_decode("MTP:target_decode_per_slot");
                        const char * mtp_mmvq_serial_columns_filter = mtp_mmvq_serial_columns_filter_for_decode(1);
                        const bool mtp_mmvq_serial_columns_active = !mtp_env_value_disabled(mtp_mmvq_serial_columns_filter);
                        mtp_env_flag_scope mtp_mmvq_serial_columns_active_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE", mtp_mmvq_serial_columns_active);
                        mtp_env_var_scope mtp_mmvq_serial_columns_filter_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER", mtp_mmvq_serial_columns_filter, mtp_mmvq_serial_columns_active);
                        ret_per_slot = llama_decode(slot.ctx_tgt, mtp_batch);
                    }
                    if (mtp_cycle_trace) {
                        mtp_cycle_per_slot_decode_ms = double(ggml_time_us() - mtp_cycle_per_slot_decode_t0) / 1000.0;
                    }
                    metrics.on_decoded(slots);
                    if (ret_per_slot != 0) {
                        SRV_ERR("MTP per-slot target decode failed, ret = %d\n", ret_per_slot);
                        slot.spec_i_batch.clear();
                        continue;
                    }

                    const int64_t mtp_cycle_per_slot_process_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    {
                        mtp_roctx_range roctx_mtp_per_slot_process("MTP:spec_process_per_slot");
                        if (!common_speculative_process(spec.get(), mtp_batch)) {
                            SRV_ERR("%s", "failed to process MTP per-slot speculative batch\n");
                            slot.spec_i_batch.clear();
                            continue;
                        }
                    }
                    if (mtp_cycle_trace) {
                        mtp_cycle_per_slot_process_ms = double(ggml_time_us() - mtp_cycle_per_slot_process_t0) / 1000.0;
                    }

                    slot.spec_i_batch = std::move(mtp_per_slot_i_batch);

                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    const int64_t mtp_cycle_sample_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    llama_tokens accepted;
                    {
                        mtp_roctx_range roctx_mtp_sample_per_slot("MTP:sample_accept_per_slot");
                        accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    }
                    const double mtp_cycle_sample_ms = mtp_cycle_trace ? double(ggml_time_us() - mtp_cycle_sample_t0) / 1000.0 : 0.0;
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
                    const size_t n_accepted = accepted.size() - 1;
                    if (mtp_cycle_trace) {
                        fprintf(stderr,
                                "MTP_CYCLE_TRACE: phase=sample_accept_per_slot slot=%d draft=%zu accepted=%zu rollback=%u decode_ms=%.3f process_ms=%.3f sample_accept_ms=%.3f\n",
                                slot.id, n_draft, n_accepted, n_rollback,
                                mtp_cycle_per_slot_decode_ms, mtp_cycle_per_slot_process_ms, mtp_cycle_sample_ms);
                    }

                    std::vector<uint8_t> mtp_state_compare_live_data;
                    mtp_rs_state_digest mtp_state_compare_live_digest;
                    bool mtp_state_compare_live_ok = false;
                    if (mtp_state_compare) {
                        const llama_pos live_commit_pos = (llama_pos) ckpt.n_tokens + 1 + (llama_pos) n_accepted;
                        if (n_rollback > 0) {
                            common_context_seq_rm(slot.ctx_tgt, slot.id, live_commit_pos, -1);
                        }
                        mtp_state_compare_live_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                        mtp_state_compare_live_digest = mtp_digest_bytes(mtp_state_compare_live_data);
                        mtp_state_compare_live_ok = !mtp_state_compare_live_data.empty();
                    }

                    if (mtp_compare_verify) {
                        llama_tokens oracle;
                        oracle.reserve(n_draft + 1);
                        common_sampler_ptr smpl_oracle(common_sampler_clone(smpl_save.get()));
                        mtp_llama_batch_scope mtp_oracle_batch_scope(1);
                        llama_batch & oracle_batch = mtp_oracle_batch_scope.batch;

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);

                        llama_pos oracle_pos = (llama_pos) ckpt.n_tokens;
                        auto oracle_decode_one = [&](llama_token tok) -> bool {
                            common_batch_clear(oracle_batch);
                            common_batch_add(oracle_batch, tok, oracle_pos++, { slot.id }, true);
                            const int ret_oracle = llama_decode(slot.ctx_tgt, oracle_batch);
                            metrics.on_decoded(slots);
                            if (ret_oracle != 0) {
                                SRV_ERR("MTP per-slot serial-compare decode failed, ret = %d\n", ret_oracle);
                                return false;
                            }
                            return true;
                        };

                        bool oracle_ok = oracle_decode_one(slot.sampled);
                        size_t oracle_accepted = 0;
                        llama_token oracle_id = LLAMA_TOKEN_NULL;
                        if (oracle_ok) {
                            oracle_id = common_sampler_sample(smpl_oracle.get(), slot.ctx_tgt, 0);
                            common_sampler_accept(smpl_oracle.get(), oracle_id, true);
                            oracle.push_back(oracle_id);
                        }
                        while (oracle_ok && oracle_accepted < n_draft && oracle_id == slot.spec_draft[oracle_accepted]) {
                            oracle_ok = oracle_decode_one(oracle_id);
                            if (!oracle_ok) {
                                break;
                            }
                            oracle_accepted++;
                            oracle_id = common_sampler_sample(smpl_oracle.get(), slot.ctx_tgt, 0);
                            common_sampler_accept(smpl_oracle.get(), oracle_id, true);
                            oracle.push_back(oracle_id);
                        }

                        const bool oracle_match = oracle_ok && accepted.size() == oracle.size() && std::equal(accepted.begin(), accepted.end(), oracle.begin());
                        fprintf(stderr,
                                "MTP_MULTI_SLOT_VERIFY_COMPARE: slot=%d draft=%zu per_slot_accepted=%zu oracle_accepted=%zu match=%d ok=%d per_slot_tokens=[",
                                slot.id, n_draft, accepted.size() - 1, oracle_accepted, oracle_match ? 1 : 0, oracle_ok ? 1 : 0);
                        for (size_t j = 0; j < accepted.size(); ++j) {
                            fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) accepted[j]);
                        }
                        fprintf(stderr, "] oracle_tokens=[");
                        for (size_t j = 0; j < oracle.size(); ++j) {
                            fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) oracle[j]);
                        }
                        fprintf(stderr, "]\n");
                    }

                    if (n_rollback > 0 && mtp_live_rollback) {
                        const llama_pos live_commit_pos = (llama_pos) ckpt.n_tokens + 1 + (llama_pos) n_accepted;
                        common_context_seq_rm(slot.ctx_tgt, slot.id, live_commit_pos, -1);
                        if (slot.ctx_dft) {
                            common_context_seq_rm(slot.ctx_dft, slot.id, live_commit_pos, -1);
                        }
                        if (mtp_cycle_trace) {
                            fprintf(stderr,
                                    "MTP_CYCLE_TRACE: phase=live_rollback_per_slot slot=%d accepted=%zu rollback=%u commit_pos=%d\n",
                                    slot.id, n_accepted, n_rollback, (int) live_commit_pos);
                        }
                    } else if (n_rollback > 0 || mtp_compare_verify || mtp_state_compare) {
                    // The full per-slot verifier batch leaves recurrent rollback rows in
                    // the target context. Materialize the accepted prefix from the exact
                    // checkpoint instead of relying on multi-row rollback after rejection.
                    // This keeps the next generated token's starting position aligned with
                    // slot.prompt while still sampling/accepting from the per-slot verifier.
                    double mtp_cycle_per_slot_commit_ms = 0.0;
                    ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                    if (slot.ctx_dft) {
                        if (!ckpt.data_dft.empty()) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }
                        common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                    }

                    common_batch_clear(mtp_batch);
                    llama_pos mtp_per_slot_commit_pos = (llama_pos) ckpt.n_tokens;
                    common_batch_add(mtp_batch, slot.sampled, mtp_per_slot_commit_pos++, { slot.id }, true);
                    for (size_t j = 0; j < n_accepted; ++j) {
                        common_batch_add(mtp_batch, accepted[j], mtp_per_slot_commit_pos++, { slot.id }, true);
                    }

                    const int64_t mtp_cycle_per_slot_commit_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    int ret_commit = 0;
                    {
                        mtp_roctx_range roctx_mtp_per_slot_commit("MTP:target_commit_per_slot");
                        const char * mtp_mmvq_serial_columns_filter = mtp_mmvq_serial_columns_filter_for_decode(1);
                        const bool mtp_mmvq_serial_columns_active = !mtp_env_value_disabled(mtp_mmvq_serial_columns_filter);
                        mtp_env_flag_scope mtp_mmvq_serial_columns_active_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE", mtp_mmvq_serial_columns_active);
                        mtp_env_var_scope mtp_mmvq_serial_columns_filter_scope("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER", mtp_mmvq_serial_columns_filter, mtp_mmvq_serial_columns_active);
                        ret_commit = llama_decode(slot.ctx_tgt, mtp_batch);
                    }
                    if (mtp_cycle_trace) {
                        mtp_cycle_per_slot_commit_ms = double(ggml_time_us() - mtp_cycle_per_slot_commit_t0) / 1000.0;
                    }
                    metrics.on_decoded(slots);
                    if (ret_commit != 0) {
                        SRV_ERR("MTP per-slot accepted-prefix commit failed, ret = %d\n", ret_commit);
                        slot.smpl = std::move(smpl_save);
                        continue;
                    }
                    {
                        mtp_roctx_range roctx_mtp_per_slot_commit_process("MTP:spec_process_per_slot_commit");
                        if (!common_speculative_process(spec.get(), mtp_batch)) {
                            SRV_ERR("%s", "failed to process MTP per-slot accepted-prefix commit batch\n");
                            slot.smpl = std::move(smpl_save);
                            continue;
                        }
                    }
                    if (mtp_state_compare) {
                        auto serial_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                        const mtp_rs_state_digest serial_digest = mtp_digest_bytes(serial_data);
                        const bool state_match = mtp_state_compare_live_ok &&
                            mtp_state_compare_live_digest.size == serial_digest.size &&
                            mtp_state_compare_live_digest.hash == serial_digest.hash;
                        const int64_t first_diff = mtp_first_diff_offset(mtp_state_compare_live_data, serial_data);
                        fprintf(stderr,
                                "MTP_MULTI_SLOT_STATE_COMPARE: slot=%d draft=%zu accepted=%zu rollback=%u live_ok=%d match=%d live_size=%zu live_hash=%016" PRIx64 " serial_size=%zu serial_hash=%016" PRIx64 " first_diff=%lld\n",
                                slot.id, n_draft, n_accepted, n_rollback, mtp_state_compare_live_ok ? 1 : 0, state_match ? 1 : 0,
                                mtp_state_compare_live_digest.size, mtp_state_compare_live_digest.hash,
                                serial_digest.size, serial_digest.hash, (long long) first_diff);
                    }
                    if (mtp_cycle_trace) {
                        fprintf(stderr,
                                "MTP_CYCLE_TRACE: phase=commit_per_slot slot=%d accepted=%zu commit_tokens=%zu commit_ms=%.3f\n",
                                slot.id, n_accepted, n_accepted + 1, mtp_cycle_per_slot_commit_ms);
                    }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens (per-slot)\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);
                    slot.spec_draft = std::move(accepted);

                    const int64_t t_current = ggml_time_us();
                    const auto ids = std::move(slot.spec_draft);

                    slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;
                    slot.n_draft_accepted += ids.size() - 1;

                    slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                    slot.sampled = ids.back();
                    SLT_DBG(slot, "add per-slot accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                    common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                    if (slot.ctx_dft) {
                        common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                    }

                    for (size_t j = 0; j < ids.size(); ++j) {
                        completion_token_output result;

                        result.tok          = ids[j];
                        result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                        result.prob         = 1.0f;

                        slot.n_decoded += 1;

                        if (!process_token(result, slot)) {
                            slot.print_timings();
                            send_final_response(slot);
                            metrics.on_prediction(slot);
                            slot.release();

                            break;
                        }
                    }

                    slot.print_timings_tg();

                    SLT_DBG(slot, "accepted %d/%d draft tokens (per-slot), new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());

                    continue;
                }

                if (slot.spec_target_serial_verify || mtp_target_serial_verify_enabled()) {
                    GGML_ASSERT(slot.spec_i_batch.size() == 1);

                    std::vector<llama_token> ids;
                    ids.reserve(n_draft + 1);

                    const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
                    const int64_t mtp_cycle_sample_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    mtp_roctx_range roctx_mtp_sample_serial("MTP:sample_accept_serial");

                    llama_token id = LLAMA_TOKEN_NULL;
                    if (slot.id >= 0 && (size_t) slot.id < mtp_serial_verify_initial_ids.size() && mtp_serial_verify_initial_sampled[slot.id]) {
                        id = mtp_serial_verify_initial_ids[slot.id];
                    } else {
                        // Fallback for unexpected chunking; keep the old behavior rather
                        // than failing before the serial verifier can report its own error.
                        id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch[0]);
                        common_sampler_accept(slot.smpl.get(), id, true);
                    }
                    ids.push_back(id);
                    slot.spec_i_batch.clear();

                    size_t n_accepted = 0;
                    double mtp_cycle_serial_decode_ms = 0.0;
                    double mtp_cycle_serial_sample_ms = 0.0;
                    double mtp_cycle_serial_h_capture_ms = 0.0;
                    double mtp_cycle_serial_process_ms = 0.0;
                    const bool mtp_serial_batch_dft_process = mtp_serial_verify_batch_dft_process_enabled() &&
                        common_speculative_need_embd_pre_norm(spec.get());
                    const bool mtp_serial_batch_dft_trace = mtp_serial_verify_batch_dft_process_trace_enabled();
                    if (mtp_serial_batch_dft_trace) {
                        fprintf(stderr,
                                "MTP_SERIAL_BATCH_DFT_TRACE: phase=serial_begin slot=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d batch_process=%d\n",
                                slot.id, (int) slot.prompt.tokens.pos_next(),
                                (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                slot.ctx_dft ? (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id) : -999,
                                mtp_serial_batch_dft_process ? 1 : 0);
                    }
                    const int32_t mtp_serial_h_dim = mtp_serial_batch_dft_process ? llama_model_n_embd(llama_get_model(slot.ctx_tgt)) : 0;
                    std::vector<llama_token> mtp_serial_process_tokens;
                    std::vector<llama_pos>   mtp_serial_process_pos;
                    std::vector<float>       mtp_serial_process_h;
                    if (mtp_serial_batch_dft_process) {
                        mtp_serial_process_tokens.reserve(n_draft);
                        mtp_serial_process_pos.reserve(n_draft);
                        mtp_serial_process_h.reserve((size_t) n_draft * mtp_serial_h_dim);
                    }
                    while (n_accepted < n_draft && id == slot.spec_draft[n_accepted]) {
                        const llama_pos pos = slot.prompt.tokens.pos_next();
                        common_batch_clear(batch);
                        common_batch_add(batch, id, pos, { slot.id }, true);
                        slot.prompt.tokens.push_back(id);

                        const int64_t mtp_cycle_serial_decode_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                        int ret_serial = 0;
                        {
                            mtp_roctx_range roctx_mtp_serial_decode("MTP:serial_verify_decode");
                            ret_serial = llama_decode(slot.ctx_tgt, batch);
                        }
                        if (mtp_cycle_trace) {
                            mtp_cycle_serial_decode_ms += double(ggml_time_us() - mtp_cycle_serial_decode_t0) / 1000.0;
                        }
                        metrics.on_decoded(slots);
                        if (ret_serial != 0) {
                            SRV_ERR("serial speculative target decode failed, ret = %d\n", ret_serial);
                            break;
                        }
                        if (mtp_serial_batch_dft_process) {
                            const int64_t mtp_cycle_serial_h_capture_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                            const float * h = llama_get_embeddings_pre_norm_ith(slot.ctx_tgt, 0);
                            if (mtp_cycle_trace) {
                                mtp_cycle_serial_h_capture_ms += double(ggml_time_us() - mtp_cycle_serial_h_capture_t0) / 1000.0;
                            }
                            if (h == nullptr) {
                                SRV_ERR("%s", "failed to capture target pre-norm row for batched serial MTP process\n");
                                break;
                            }
                            mtp_serial_process_tokens.push_back(id);
                            mtp_serial_process_pos.push_back(pos);
                            mtp_serial_process_h.insert(mtp_serial_process_h.end(), h, h + mtp_serial_h_dim);
                            if (mtp_serial_batch_dft_trace) {
                                fprintf(stderr,
                                        "MTP_SERIAL_BATCH_DFT_TRACE: phase=serial_row slot=%d accepted_next=%zu row_pos=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d\n",
                                        slot.id, n_accepted + 1, (int) pos, (int) slot.prompt.tokens.pos_next(),
                                        (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                        slot.ctx_dft ? (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id) : -999);
                            }
                        } else {
                            const int64_t mtp_cycle_serial_process_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                            bool serial_process_ok = false;
                            {
                                mtp_roctx_range roctx_mtp_serial_process("MTP:serial_verify_process");
                                serial_process_ok = common_speculative_process(spec.get(), batch);
                            }
                            if (!serial_process_ok) {
                                SRV_ERR("%s", "failed to process serial speculative batch\n");
                                break;
                            }
                            if (mtp_cycle_trace) {
                                mtp_cycle_serial_process_ms += double(ggml_time_us() - mtp_cycle_serial_process_t0) / 1000.0;
                            }
                        }

                        n_accepted++;
                        const int64_t mtp_cycle_serial_sample_one_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                        id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, 0);
                        if (mtp_cycle_trace) {
                            mtp_cycle_serial_sample_ms += double(ggml_time_us() - mtp_cycle_serial_sample_one_t0) / 1000.0;
                        }
                        common_sampler_accept(slot.smpl.get(), id, true);
                        ids.push_back(id);
                    }

                    if (mtp_serial_batch_dft_process && !mtp_serial_process_tokens.empty()) {
                        GGML_ASSERT(mtp_serial_process_tokens.size() == mtp_serial_process_pos.size());
                        GGML_ASSERT(mtp_serial_process_h.size() == mtp_serial_process_tokens.size() * (size_t) mtp_serial_h_dim);
                        mtp_llama_batch_scope mtp_serial_process_batch_scope((int32_t) mtp_serial_process_tokens.size());
                        llama_batch & serial_process_batch = mtp_serial_process_batch_scope.batch;
                        common_batch_clear(serial_process_batch);
                        for (size_t j = 0; j < mtp_serial_process_tokens.size(); ++j) {
                            common_batch_add(serial_process_batch, mtp_serial_process_tokens[j], mtp_serial_process_pos[j], { slot.id }, true);
                        }

                        if (mtp_serial_batch_dft_trace) {
                            fprintf(stderr,
                                    "MTP_SERIAL_BATCH_DFT_TRACE: phase=before_process slot=%d n_tokens=%d first_pos=%d last_pos=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d\n",
                                    slot.id, (int) serial_process_batch.n_tokens,
                                    serial_process_batch.n_tokens > 0 ? (int) serial_process_batch.pos[0] : -1,
                                    serial_process_batch.n_tokens > 0 ? (int) serial_process_batch.pos[serial_process_batch.n_tokens - 1] : -1,
                                    (int) slot.prompt.tokens.pos_next(),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                    slot.ctx_dft ? (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id) : -999);
                        }
                        const int64_t mtp_cycle_serial_process_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                        bool serial_process_ok = false;
                        {
                            mtp_roctx_range roctx_mtp_serial_process("MTP:serial_verify_process_batched");
                            serial_process_ok = common_speculative_process_with_pre_norm(spec.get(), serial_process_batch, mtp_serial_process_h.data());
                        }
                        if (!serial_process_ok) {
                            SRV_ERR("%s", "failed to process batched serial speculative batch\n");
                        }
                        if (mtp_serial_batch_dft_trace) {
                            fprintf(stderr,
                                    "MTP_SERIAL_BATCH_DFT_TRACE: phase=after_process slot=%d ok=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d\n",
                                    slot.id, serial_process_ok ? 1 : 0, (int) slot.prompt.tokens.pos_next(),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                    slot.ctx_dft ? (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id) : -999);
                        }
                        if (mtp_cycle_trace) {
                            mtp_cycle_serial_process_ms += double(ggml_time_us() - mtp_cycle_serial_process_t0) / 1000.0;
                        }
                    }

                    const double mtp_cycle_sample_ms = mtp_cycle_trace ? double(ggml_time_us() - mtp_cycle_sample_t0) / 1000.0 : 0.0;
                    if (mtp_cycle_trace) {
                        fprintf(stderr,
                                "MTP_CYCLE_TRACE: phase=sample_accept_serial slot=%d draft=%zu accepted=%zu rollback=%zu sample_accept_ms=%.3f serial_decode_ms=%.3f serial_sample_ms=%.3f serial_h_capture_ms=%.3f serial_process_ms=%.3f\n",
                                slot.id, n_draft, n_accepted, n_draft - n_accepted, mtp_cycle_sample_ms,
                                mtp_cycle_serial_decode_ms, mtp_cycle_serial_sample_ms, mtp_cycle_serial_h_capture_ms, mtp_cycle_serial_process_ms);
                    }
                    if (mtp_verify_trace_enabled()) {
                        fprintf(stderr,
                                "MTP_VERIFY_SERIAL: slot=%d backend=%s reason=%s draft=%zu accepted=%zu recurrent_prefix_required=%d\n",
                                slot.id,
                                mtp_verify_backend_name(slot.spec_verify_backend),
                                slot.spec_verify_backend_reason,
                                n_draft,
                                n_accepted,
                                slot.spec_verify_recurrent_prefix_required ? 1 : 0);
                    }

                    common_speculative_accept(spec.get(), slot.id, (uint16_t) n_accepted);
                    slot.spec_draft.clear();

                    const int64_t t_current = ggml_time_us();
                    slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;
                    slot.n_draft_accepted += n_accepted;
                    slot.sampled = ids.back();
                    if (mtp_serial_batch_dft_process && slot.ctx_dft) {
                        const llama_pos rm_from = slot.prompt.tokens.pos_next();
                        if (mtp_serial_batch_dft_trace) {
                            fprintf(stderr,
                                    "MTP_SERIAL_BATCH_DFT_TRACE: phase=before_cleanup slot=%d rm_from=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d\n",
                                    slot.id, (int) rm_from, (int) slot.prompt.tokens.pos_next(),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id));
                        }
                        common_context_seq_rm(slot.ctx_dft, slot.id, rm_from, -1);
                        if (mtp_serial_batch_dft_trace) {
                            fprintf(stderr,
                                    "MTP_SERIAL_BATCH_DFT_TRACE: phase=after_cleanup slot=%d rm_from=%d prompt_next=%d tgt_pos_max=%d dft_pos_max=%d\n",
                                    slot.id, (int) rm_from, (int) slot.prompt.tokens.pos_next(),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_tgt), slot.id),
                                    (int) llama_memory_seq_pos_max(llama_get_memory(slot.ctx_dft), slot.id));
                        }
                    }

                    bool released = false;
                    for (llama_token tok : ids) {
                        completion_token_output result;
                        result.tok          = tok;
                        result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                        result.prob         = 1.0f;

                        slot.n_decoded += 1;

                        if (!process_token(result, slot)) {
                            slot.print_timings();
                            send_final_response(slot);
                            metrics.on_prediction(slot);
                            slot.release();
                            released = true;
                            break;
                        }
                    }

                    if (!released) {
                        slot.print_timings_tg();
                    }

                    continue;
                }

                bool prefix_accepted_row_commit_done = false;
                uint32_t prefix_accepted_row_commit_idx = 0;

                // verify and try to accept the draft
                {
                    // save the sampler sampler state in case we need to restore it
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    const bool mtp_cycle_trace = getenv("LLAMA_MTP_CYCLE_TRACE") && atoi(getenv("LLAMA_MTP_CYCLE_TRACE")) != 0;
                    const int64_t mtp_cycle_sample_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                    llama_tokens accepted;
                    const std::vector<int> candidate_i_batch = slot.spec_i_batch;
                    {
                        mtp_roctx_range roctx_mtp_sample_batched("MTP:sample_accept_batched");
                        accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    }
                    const double mtp_cycle_sample_ms = mtp_cycle_trace ? double(ggml_time_us() - mtp_cycle_sample_t0) / 1000.0 : 0.0;
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const bool mtp_compare_verify = mtp_verify_compare_enabled();
                    const bool replay_accepted_requested = mtp_target_batch_verify_replay_accepted_enabled() || mtp_compare_verify;
                    std::vector<uint8_t> verify_compare_oracle_state_data;
                    mtp_rs_state_digest verify_compare_oracle_state_digest;
                    bool verify_compare_oracle_state_ok = false;
                    if (mtp_compare_verify) {
                        const llama_tokens candidate = accepted;
                        for (size_t j = 0; j < candidate.size() && j < candidate_i_batch.size(); ++j) {
                            const llama_token draft_id = j < slot.spec_draft.size() ? slot.spec_draft[j] : LLAMA_TOKEN_NULL;
                            const bool decision_accepted = draft_id == LLAMA_TOKEN_NULL || candidate[j] == draft_id;
                            mtp_trace_verify_logits("candidate", slot.id, slot.spec_gdn_compare_scope.c_str(), slot.ctx_tgt,
                                    candidate_i_batch[j], (int) j + 1, draft_id, candidate[j], decision_accepted, {});
                            if (draft_id != LLAMA_TOKEN_NULL && candidate[j] != draft_id) {
                                break;
                            }
                        }
                        const size_t candidate_accepted = candidate.size() - 1;
                        const llama_pos candidate_commit_pos = (llama_pos) slot.spec_ckpt.n_tokens + 1 + (llama_pos) candidate_accepted;
                        std::vector<uint8_t> candidate_state_data;
                        mtp_rs_state_digest candidate_state_digest;
                        bool candidate_state_ok = false;
                        if (!slot.spec_ckpt.data_tgt.empty() && llama_n_rs_seq(slot.ctx_tgt) > 0) {
                            common_context_seq_rm(slot.ctx_tgt, slot.id, candidate_commit_pos, -1);
                            if (slot.spec_verify_backend == MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX &&
                                    !llama_context_recurrent_commit_pending_rs_rollback(slot.ctx_tgt, slot.id)) {
                                SRV_ERR("MTP serial_equiv_prefix candidate recurrent commit failed for slot %d\n", slot.id);
                            }
                            candidate_state_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                            candidate_state_digest = mtp_digest_bytes(candidate_state_data);
                            candidate_state_ok = !candidate_state_data.empty();
                        }

                        llama_tokens oracle;
                        oracle.reserve(n_draft + 1);
                        common_sampler_ptr smpl_oracle(common_sampler_clone(smpl_save.get()));
                        bool oracle_ok = false;
                        size_t oracle_accepted = 0;
                        std::vector<uint8_t> oracle_state_data;
                        mtp_rs_state_digest oracle_state_digest;
                        bool oracle_state_ok = false;

                        const auto & ckpt = slot.spec_ckpt;
                        if (ckpt.data_tgt.empty()) {
                            SRV_ERR("%s", "MTP verify-compare requested but target checkpoint is empty\n");
                        } else {
                            mtp_llama_batch_scope mtp_oracle_batch_scope(1);
                            llama_batch & oracle_batch = mtp_oracle_batch_scope.batch;

                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                            common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);

                            llama_pos oracle_pos = (llama_pos) ckpt.n_tokens;
                            auto oracle_decode_one = [&](llama_token tok) -> bool {
                                common_batch_clear(oracle_batch);
                                common_batch_add(oracle_batch, tok, oracle_pos++, { slot.id }, true);
                                const int ret_oracle = llama_decode(slot.ctx_tgt, oracle_batch);
                                metrics.on_decoded(slots);
                                if (ret_oracle != 0) {
                                    SRV_ERR("MTP verify-compare serial decode failed, ret = %d\n", ret_oracle);
                                    return false;
                                }
                                return true;
                            };

                            oracle_ok = oracle_decode_one(slot.sampled);
                            llama_token oracle_id = LLAMA_TOKEN_NULL;
                            if (oracle_ok) {
                                oracle_id = common_sampler_sample(smpl_oracle.get(), slot.ctx_tgt, 0);
                                common_sampler_accept(smpl_oracle.get(), oracle_id, true);
                                oracle.push_back(oracle_id);
                                const llama_token draft_id = n_draft > 0 ? slot.spec_draft[0] : LLAMA_TOKEN_NULL;
                                const bool decision_accepted = draft_id == LLAMA_TOKEN_NULL || oracle_id == draft_id;
                                const std::vector<llama_token> watch = candidate.empty() ? std::vector<llama_token>{} : std::vector<llama_token>{ candidate[0] };
                                mtp_trace_verify_logits("oracle", slot.id, slot.spec_gdn_compare_scope.c_str(), slot.ctx_tgt,
                                        0, 1, draft_id, oracle_id, decision_accepted, watch);
                            }
                            while (oracle_ok && oracle_accepted < n_draft && oracle_id == slot.spec_draft[oracle_accepted]) {
                                oracle_ok = oracle_decode_one(oracle_id);
                                if (!oracle_ok) {
                                    break;
                                }
                                oracle_accepted++;
                                oracle_id = common_sampler_sample(smpl_oracle.get(), slot.ctx_tgt, 0);
                                common_sampler_accept(smpl_oracle.get(), oracle_id, true);
                                oracle.push_back(oracle_id);
                                const llama_token draft_id = oracle_accepted < n_draft ? slot.spec_draft[oracle_accepted] : LLAMA_TOKEN_NULL;
                                const bool decision_accepted = draft_id == LLAMA_TOKEN_NULL || oracle_id == draft_id;
                                const std::vector<llama_token> watch = oracle_accepted < candidate.size() ? std::vector<llama_token>{ candidate[oracle_accepted] } : std::vector<llama_token>{};
                                mtp_trace_verify_logits("oracle", slot.id, slot.spec_gdn_compare_scope.c_str(), slot.ctx_tgt,
                                        0, (int) oracle_accepted + 1, draft_id, oracle_id, decision_accepted, watch);
                            }

                            if (oracle_ok && llama_n_rs_seq(slot.ctx_tgt) > 0) {
                                oracle_state_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                                oracle_state_digest = mtp_digest_bytes(oracle_state_data);
                                oracle_state_ok = !oracle_state_data.empty();
                            }
                        }

                        size_t first_mismatch = (size_t) -1;
                        const size_t n_cmp = std::min(candidate.size(), oracle.size());
                        for (size_t j = 0; j < n_cmp; ++j) {
                            if (candidate[j] != oracle[j]) {
                                first_mismatch = j;
                                break;
                            }
                        }
                        if (first_mismatch == (size_t) -1 && candidate.size() != oracle.size()) {
                            first_mismatch = n_cmp;
                        }
                        const bool token_match = oracle_ok && candidate.size() == oracle.size() && std::equal(candidate.begin(), candidate.end(), oracle.begin());
                        const bool state_match = candidate_state_ok && oracle_state_ok &&
                            candidate_state_digest.size == oracle_state_digest.size &&
                            candidate_state_digest.hash == oracle_state_digest.hash;
                        const int64_t state_first_diff = (candidate_state_ok && oracle_state_ok) ? mtp_first_diff_offset(candidate_state_data, oracle_state_data) : -1;
                        const uint32_t candidate_n_rollback = (uint32_t) (slot.spec_draft.size() + 1 - candidate.size());
                        if (mtp_rs_state_trace_enabled() && candidate_state_ok && oracle_state_ok && !state_match) {
                            mtp_trace_rs_state_components(slot.id, n_draft, candidate_accepted, candidate_n_rollback,
                                    "verify_compare_candidate_vs_oracle", candidate_state_data, oracle_state_data);
                        }
                        fprintf(stderr,
                                "MTP_VERIFY_COMPARE: slot=%d backend=%s reason=%s draft=%zu candidate_accepted=%zu oracle_accepted=%zu token_match=%d oracle_ok=%d first_mismatch=%lld candidate_state_ok=%d oracle_state_ok=%d state_match=%d candidate_state_size=%zu candidate_state_hash=%016" PRIx64 " oracle_state_size=%zu oracle_state_hash=%016" PRIx64 " state_first_diff=%lld candidate_tokens=[",
                                slot.id,
                                mtp_verify_backend_name(slot.spec_verify_backend),
                                slot.spec_verify_backend_reason,
                                n_draft,
                                candidate_accepted,
                                oracle_accepted,
                                token_match ? 1 : 0,
                                oracle_ok ? 1 : 0,
                                first_mismatch == (size_t) -1 ? -1LL : (long long) first_mismatch,
                                candidate_state_ok ? 1 : 0,
                                oracle_state_ok ? 1 : 0,
                                state_match ? 1 : 0,
                                candidate_state_digest.size,
                                candidate_state_digest.hash,
                                oracle_state_digest.size,
                                oracle_state_digest.hash,
                                (long long) state_first_diff);
                        for (size_t j = 0; j < candidate.size(); ++j) {
                            fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) candidate[j]);
                        }
                        fprintf(stderr, "] oracle_tokens=[");
                        for (size_t j = 0; j < oracle.size(); ++j) {
                            fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) oracle[j]);
                        }
                        fprintf(stderr, "]\n");

                        if (oracle_state_ok) {
                            verify_compare_oracle_state_data = oracle_state_data;
                            verify_compare_oracle_state_digest = oracle_state_digest;
                            verify_compare_oracle_state_ok = true;
                        }

                        if (oracle_ok) {
                            accepted = std::move(oracle);
                            slot.smpl = std::move(smpl_oracle);
                        }
                    }

                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
                    if (mtp_cycle_trace) {
                        fprintf(stderr,
                                "MTP_CYCLE_TRACE: phase=sample_accept slot=%d draft=%zu accepted=%zu rollback=%u sample_accept_ms=%.3f\n",
                                slot.id, n_draft, accepted.size() - 1, n_rollback, mtp_cycle_sample_ms);
                    }

                    const bool accepted_row_only_commit =
                        mtp_prefix_accepted_row_only_commit_enabled() &&
                        slot.spec_verify_backend == MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX &&
                        !replay_accepted_requested;
                    if (accepted_row_only_commit) {
                        const size_t n_accepted = accepted.size() - 1;
                        const int fast_commit_slots = mtp_prefix_accepted_row_commit_verify_slots();
                        if (n_rollback < (uint32_t) fast_commit_slots) {
                            // The verifier graph materializes a small accepted-prefix suffix
                            // into matching rollback slots even when other prefix snapshots
                            // are skipped. Covered accept lengths can therefore commit
                            // directly without a checkpoint restore or second prefix pass.
                            prefix_accepted_row_commit_done = true;
                            prefix_accepted_row_commit_idx = n_rollback;
                            if (mtp_cycle_trace) {
                                fprintf(stderr,
                                        "MTP_CYCLE_TRACE: phase=accepted_row_only_commit_fast slot=%d draft=%zu accepted=%zu rollback=%u fast_slots=%d commit_tokens=%zu restore_ms=0.000 commit_ms=0.000\n",
                                        slot.id, n_draft, n_accepted, n_rollback, fast_commit_slots, accepted.size());
                            }
                        } else {
                            const auto & ckpt = slot.spec_ckpt;
                            if (ckpt.data_tgt.empty()) {
                                SRV_ERR("%s", "MTP accepted-row-only prefix commit requested but target checkpoint is empty\n");
                                continue;
                            }

                            const int64_t mtp_cycle_commit_restore_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                            common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            const double mtp_cycle_commit_restore_ms = mtp_cycle_trace ? double(ggml_time_us() - mtp_cycle_commit_restore_t0) / 1000.0 : 0.0;

                            mtp_llama_batch_scope mtp_prefix_commit_batch_scope((int32_t) accepted.size());
                            llama_batch & prefix_commit_batch = mtp_prefix_commit_batch_scope.batch;
                            llama_pos pos = (llama_pos) slot.spec_ckpt.n_tokens;
                            common_batch_add(prefix_commit_batch, slot.sampled, pos++, { slot.id }, false);
                            for (size_t j = 0; j < n_accepted; ++j) {
                                common_batch_add(prefix_commit_batch, accepted[j], pos++, { slot.id }, false);
                            }

                            const int64_t mtp_cycle_commit_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                            int ret_commit = 0;
                            {
                                mtp_roctx_range roctx_mtp_prefix_commit("MTP:prefix_accepted_row_commit");
                                mtp_env_var_scope mtp_gdn_compare_scope_env("LLAMA_MTP_GDN_INPUT_TRACE_COMPARE_SCOPE", slot.spec_gdn_compare_scope.c_str(), !slot.spec_gdn_compare_scope.empty());
                                ret_commit = llama_decode_prefix_commit(slot.ctx_tgt, prefix_commit_batch, n_rollback);
                            }
                            metrics.on_decoded(slots);
                            if (ret_commit != 0) {
                                SRV_ERR("MTP accepted-row-only prefix commit failed, ret = %d\n", ret_commit);
                                continue;
                            }
                            prefix_accepted_row_commit_done = true;
                            prefix_accepted_row_commit_idx = n_rollback;
                            if (mtp_cycle_trace) {
                                const double commit_ms = double(ggml_time_us() - mtp_cycle_commit_t0) / 1000.0;
                                fprintf(stderr,
                                        "MTP_CYCLE_TRACE: phase=accepted_row_only_commit slot=%d draft=%zu accepted=%zu rollback=%u commit_tokens=%zu restore_ms=%.3f commit_ms=%.3f\n",
                                        slot.id, n_draft, n_accepted, n_rollback, accepted.size(), mtp_cycle_commit_restore_ms, commit_ms);
                            }
                        }
                    }

                    const bool replay_accepted = replay_accepted_requested ||
                        (mtp_target_batch_verify_replay_partial_enabled() && n_rollback > 0);
                    if (replay_accepted) {
                        const auto & ckpt = slot.spec_ckpt;
                        if (ckpt.data_tgt.empty()) {
                            SRV_ERR("%s", "MTP replay-accepted verifier requested but target checkpoint is empty\n");
                            continue;
                        }

                        const size_t n_accepted = accepted.size() - 1;

                        const bool rs_state_trace = !mtp_compare_verify && mtp_rs_state_trace_enabled();
                        const bool rs_window_trace = rs_state_trace && mtp_rs_window_trace_enabled();
                        mtp_rs_state_digest rs_trace_batched_full;
                        mtp_rs_state_digest rs_trace_unsafe_commit;
                        std::vector<uint8_t> rs_trace_batched_full_data;
                        std::vector<uint8_t> rs_trace_unsafe_data;
                        std::vector<std::vector<uint8_t>> rs_trace_unsafe_window_data;
                        std::vector<llama_token> rs_trace_verify_tokens;
                        const llama_pos rs_trace_commit_pos = (llama_pos) ckpt.n_tokens + 1 + (llama_pos) n_accepted;
                        if (rs_state_trace) {
                            fprintf(stderr,
                                    "MTP_RS_TOKEN_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u ckpt_tokens=%lld ckpt_pos_min=%d ckpt_pos_max=%d commit_pos=%d sampled=%d draft_tokens=[",
                                    slot.id, n_draft, n_accepted, n_rollback,
                                    (long long) ckpt.n_tokens, ckpt.pos_min, ckpt.pos_max, rs_trace_commit_pos, (int) slot.sampled);
                            for (size_t j = 0; j < slot.spec_draft.size(); ++j) {
                                fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) slot.spec_draft[j]);
                            }
                            fprintf(stderr, "] accepted_tokens=[");
                            for (size_t j = 0; j < accepted.size(); ++j) {
                                fprintf(stderr, "%s%d", j == 0 ? "" : ",", (int) accepted[j]);
                            }
                            fprintf(stderr, "] replay_tokens=[%d", (int) slot.sampled);
                            for (size_t j = 0; j < n_accepted; ++j) {
                                fprintf(stderr, ",%d", (int) accepted[j]);
                            }
                            fprintf(stderr, "]\n");

                            // Compare the state that the old unsafe batched verifier would
                            // commit after bounded rollback against the exact serial replay
                            // state. This is diagnostic only; the checkpoint restore below
                            // discards the temporary seq_rm/materialization side effects.
                            if (rs_window_trace) {
                                rs_trace_verify_tokens.reserve(slot.spec_draft.size() + 1);
                                rs_trace_verify_tokens.push_back(slot.sampled);
                                rs_trace_verify_tokens.insert(rs_trace_verify_tokens.end(), slot.spec_draft.begin(), slot.spec_draft.end());
                            }
                            rs_trace_batched_full_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                            rs_trace_batched_full = mtp_digest_bytes(rs_trace_batched_full_data);

                            uint32_t max_window_rollback = 0;
                            if (rs_window_trace && !rs_trace_verify_tokens.empty()) {
                                const size_t n_verify_tokens = rs_trace_verify_tokens.size();
                                max_window_rollback = std::min<uint32_t>(llama_n_rs_seq(slot.ctx_tgt), (uint32_t) n_verify_tokens - 1);
                                rs_trace_unsafe_window_data.reserve((size_t) max_window_rollback + 1);
                                rs_trace_unsafe_window_data.push_back(rs_trace_batched_full_data);
                                // Walk the live batched context backward in increasing rollback order.
                                // The serialized seq state only contains the currently materialized row,
                                // so reloading a checkpoint here would lose the extra rs_seq snapshot rows.
                                for (uint32_t row_rollback = 1; row_rollback <= max_window_rollback; ++row_rollback) {
                                    const size_t prefix_tokens = n_verify_tokens - row_rollback;
                                    const llama_pos row_commit_pos = (llama_pos) ckpt.n_tokens + (llama_pos) prefix_tokens;
                                    common_context_seq_rm(slot.ctx_tgt, slot.id, row_commit_pos, -1);
                                    rs_trace_unsafe_window_data.push_back(mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id));
                                }
                                const uint32_t unsafe_row = std::min<uint32_t>(n_rollback, max_window_rollback);
                                rs_trace_unsafe_data = rs_trace_unsafe_window_data[unsafe_row];
                            } else {
                                common_context_seq_rm(slot.ctx_tgt, slot.id, rs_trace_commit_pos, -1);
                                rs_trace_unsafe_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                            }
                            rs_trace_unsafe_commit = mtp_digest_bytes(rs_trace_unsafe_data);

                            if (rs_window_trace && !rs_trace_verify_tokens.empty() && !rs_trace_unsafe_window_data.empty()) {
                                const llama_state_seq_flags rs_trace_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
                                const size_t n_verify_tokens = rs_trace_verify_tokens.size();
                                for (uint32_t row_rollback = 0; row_rollback <= max_window_rollback; ++row_rollback) {
                                    const size_t prefix_tokens = n_verify_tokens - row_rollback;
                                    const llama_pos row_commit_pos = (llama_pos) ckpt.n_tokens + (llama_pos) prefix_tokens;

                                    ckpt.load_tgt(slot.ctx_tgt, slot.id, rs_trace_flags);
                                    common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                                    llama_pos row_pos = (llama_pos) ckpt.n_tokens;
                                    bool row_replay_ok = true;
                                    for (size_t j = 0; j < prefix_tokens; ++j) {
                                        common_batch_clear(batch);
                                        common_batch_add(batch, rs_trace_verify_tokens[j], row_pos++, { slot.id }, false);
                                        const int ret_trace_replay = llama_decode(slot.ctx_tgt, batch);
                                        if (ret_trace_replay != 0) {
                                            fprintf(stderr,
                                                    "MTP_RS_WINDOW_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u row_rollback=%u prefix_tokens=%zu commit_pos=%d serial_replay_failed ret=%d\n",
                                                    slot.id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, (int) row_commit_pos, ret_trace_replay);
                                            row_replay_ok = false;
                                            break;
                                        }
                                    }
                                    if (row_replay_ok) {
                                        std::vector<uint8_t> serial_row_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                                        mtp_trace_rs_window_row(slot.id, n_draft, n_accepted, n_rollback, row_rollback, prefix_tokens, row_commit_pos,
                                                rs_trace_unsafe_window_data[row_rollback], serial_row_data);
                                    }
                                }
                            }
                        }

                        const bool replay_from_prefix = mtp_target_batch_verify_replay_from_prefix_enabled();
                        const size_t replay_prefix_tokens = replay_from_prefix ? 1u : 0u;

                        const int64_t mtp_cycle_replay_restore_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                        if (replay_from_prefix) {
                            const llama_pos prefix_pos = (llama_pos) ckpt.n_tokens + (llama_pos) replay_prefix_tokens;
                            common_context_seq_rm(slot.ctx_tgt, slot.id, prefix_pos, -1);
                        } else {
                            ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                            common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                        }
                        const double mtp_cycle_replay_restore_ms = mtp_cycle_trace ? double(ggml_time_us() - mtp_cycle_replay_restore_t0) / 1000.0 : 0.0;

                        const bool replay_skip_dft = mtp_target_batch_verify_replay_skip_dft_enabled();
                        const bool replay_outputs = !replay_skip_dft || !mtp_target_batch_verify_replay_no_logits_enabled();
                        if (!replay_skip_dft && slot.ctx_dft) {
                            if (!ckpt.data_dft.empty()) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                            }
                            common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                        }

                        llama_pos pos = (llama_pos) ckpt.n_tokens + (llama_pos) replay_prefix_tokens;
                        size_t n_replayed = 0;
                        double mtp_cycle_replay_decode_ms = 0.0;
                        auto replay_one = [&](llama_token tok) -> bool {
                            common_batch_clear(batch);
                            common_batch_add(batch, tok, pos++, { slot.id }, replay_outputs);

                            const int64_t mtp_cycle_replay_decode_t0 = mtp_cycle_trace ? ggml_time_us() : 0;
                            int ret_replay = 0;
                            {
                                mtp_env_var_scope mtp_gdn_compare_scope_env("LLAMA_MTP_GDN_INPUT_TRACE_COMPARE_SCOPE", slot.spec_gdn_compare_scope.c_str(), !slot.spec_gdn_compare_scope.empty());
                                ret_replay = llama_decode(slot.ctx_tgt, batch);
                            }
                            if (mtp_cycle_trace) {
                                mtp_cycle_replay_decode_ms += double(ggml_time_us() - mtp_cycle_replay_decode_t0) / 1000.0;
                            }
                            metrics.on_decoded(slots);
                            if (ret_replay != 0) {
                                SRV_ERR("MTP replay-accepted target decode failed, ret = %d\n", ret_replay);
                                return false;
                            }
                            if (!replay_skip_dft) {
                                if (!common_speculative_process(spec.get(), batch)) {
                                    SRV_ERR("%s", "failed to process MTP replay-accepted batch\n");
                                    return false;
                                }
                            }
                            n_replayed++;
                            return true;
                        };

                        bool replay_ok = true;
                        if (!replay_from_prefix) {
                            replay_ok = replay_one(slot.sampled);
                        }
                        for (size_t j = 0; replay_ok && j < n_accepted; ++j) {
                            replay_ok = replay_one(accepted[j]);
                        }
                        if (!replay_ok) {
                            continue;
                        }

                        if (mtp_cycle_trace) {
                            fprintf(stderr,
                                    "MTP_CYCLE_TRACE: phase=replay_accepted_serial slot=%d draft=%zu accepted=%zu replay_tokens=%zu from_prefix=%d prefix_tokens=%zu skip_dft=%d outputs=%d restore_ms=%.3f replay_decode_ms=%.3f\n",
                                    slot.id, n_draft, n_accepted, n_replayed, replay_from_prefix ? 1 : 0, replay_prefix_tokens,
                                    replay_skip_dft ? 1 : 0, replay_outputs ? 1 : 0,
                                    mtp_cycle_replay_restore_ms, mtp_cycle_replay_decode_ms);
                        }

                        if (mtp_compare_verify && verify_compare_oracle_state_ok && llama_n_rs_seq(slot.ctx_tgt) > 0) {
                            std::vector<uint8_t> post_repair_state_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                            const mtp_rs_state_digest post_repair_state_digest = mtp_digest_bytes(post_repair_state_data);
                            const bool post_repair_state_ok = !post_repair_state_data.empty();
                            const bool post_repair_state_match = post_repair_state_ok &&
                                post_repair_state_digest.size == verify_compare_oracle_state_digest.size &&
                                post_repair_state_digest.hash == verify_compare_oracle_state_digest.hash;
                            const int64_t post_repair_first_diff = post_repair_state_ok ? mtp_first_diff_offset(post_repair_state_data, verify_compare_oracle_state_data) : -1;
                            fprintf(stderr,
                                    "MTP_VERIFY_COMPARE_POST_REPAIR: slot=%d backend=%s reason=%s draft=%zu accepted=%zu rollback=%u post_state_ok=%d oracle_state_ok=%d state_match=%d post_state_size=%zu post_state_hash=%016" PRIx64 " oracle_state_size=%zu oracle_state_hash=%016" PRIx64 " state_first_diff=%lld\n",
                                    slot.id,
                                    mtp_verify_backend_name(slot.spec_verify_backend),
                                    slot.spec_verify_backend_reason,
                                    n_draft,
                                    n_accepted,
                                    n_rollback,
                                    post_repair_state_ok ? 1 : 0,
                                    verify_compare_oracle_state_ok ? 1 : 0,
                                    post_repair_state_match ? 1 : 0,
                                    post_repair_state_digest.size,
                                    post_repair_state_digest.hash,
                                    verify_compare_oracle_state_digest.size,
                                    verify_compare_oracle_state_digest.hash,
                                    (long long) post_repair_first_diff);
                        }

                        if (rs_state_trace) {
                            auto rs_trace_serial_data = mtp_get_partial_seq_state_data(slot.ctx_tgt, slot.id);
                            const mtp_rs_state_digest rs_trace_serial_replay = mtp_digest_bytes(rs_trace_serial_data);
                            const bool rs_match =
                                rs_trace_unsafe_commit.size == rs_trace_serial_replay.size &&
                                rs_trace_unsafe_commit.hash == rs_trace_serial_replay.hash;
                            const int64_t first_diff = mtp_first_diff_offset(rs_trace_unsafe_data, rs_trace_serial_data);
                            const int unsafe_byte = first_diff >= 0 && (size_t) first_diff < rs_trace_unsafe_data.size() ? rs_trace_unsafe_data[(size_t) first_diff] : -1;
                            const int serial_byte = first_diff >= 0 && (size_t) first_diff < rs_trace_serial_data.size() ? rs_trace_serial_data[(size_t) first_diff] : -1;
                            fprintf(stderr,
                                    "MTP_RS_STATE_TRACE: slot=%d draft=%zu accepted=%zu rollback=%u n_rs_seq=%u ckpt_tokens=%lld ckpt_pos_min=%d ckpt_pos_max=%d commit_pos=%d "
                                    "batched_full_size=%zu batched_full_hash=%016" PRIx64 " unsafe_size=%zu unsafe_hash=%016" PRIx64 " serial_size=%zu serial_hash=%016" PRIx64 " match=%d first_diff=%lld unsafe_byte=%d serial_byte=%d\n",
                                    slot.id, n_draft, n_accepted, n_rollback, llama_n_rs_seq(slot.ctx_tgt),
                                    (long long) ckpt.n_tokens, ckpt.pos_min, ckpt.pos_max, rs_trace_commit_pos,
                                    rs_trace_batched_full.size, rs_trace_batched_full.hash,
                                    rs_trace_unsafe_commit.size, rs_trace_unsafe_commit.hash,
                                    rs_trace_serial_replay.size, rs_trace_serial_replay.hash,
                                    rs_match ? 1 : 0, (long long) first_diff, unsafe_byte, serial_byte);
                            mtp_trace_rs_state_components(slot.id, n_draft, n_accepted, n_rollback, "unsafe_vs_serial", rs_trace_unsafe_data, rs_trace_serial_data);
                            if (rs_trace_batched_full.hash != rs_trace_unsafe_commit.hash || rs_trace_batched_full.size != rs_trace_unsafe_commit.size) {
                                mtp_trace_rs_state_components(slot.id, n_draft, n_accepted, n_rollback, "batched_full_vs_serial", rs_trace_batched_full_data, rs_trace_serial_data);
                            }
                        }

                        common_speculative_accept(spec.get(), slot.id, (uint16_t) n_accepted);
                        slot.spec_draft = std::move(accepted);
                    } else {
                        const bool use_ckpt_tgt =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                       (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                    // check for partial draft acceptance
                    if (n_rollback > 0) {
                        if (use_ckpt_tgt) {
                            if (trace > 0) {
                                SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                            }

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            {
                                ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            }

                            if (slot.ctx_dft) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                                common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                            }

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);

                            continue;
                        }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                        slot.spec_draft = std::move(accepted);
                    }
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                if (slot.spec_verify_backend == MTP_VERIFY_BACKEND_SERIAL_EQUIV_PREFIX) {
                    if (prefix_accepted_row_commit_done &&
                            !llama_context_recurrent_set_pending_rs_rollback(slot.ctx_tgt, slot.id, prefix_accepted_row_commit_idx)) {
                        SRV_ERR("MTP accepted-row-only prefix commit failed to set recurrent rollback row %u for slot %d\n",
                                prefix_accepted_row_commit_idx, slot.id);
                    }
                    if (!llama_context_recurrent_commit_pending_rs_rollback(slot.ctx_tgt, slot.id)) {
                        SRV_ERR("MTP serial_equiv_prefix recurrent commit failed for slot %d\n", slot.id);
                    }
                }
                if (slot.ctx_dft) {
                    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    slot.n_decoded += 1;

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

void server_context::unload_model() {
    if (!impl->sleeping) {
        impl->destroy();
        impl->sleeping = true;
    }
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
