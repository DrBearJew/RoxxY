#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

bool qwen35moe_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0' && atoi(env) != 0;
}


bool qwen35moe_env_bool_default(const char * name, bool def) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    return atoi(env) != 0;
}

bool qwen35moe_env_has_value(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0';
}

bool qwen35moe_env_bool_default2(const char * primary, const char * alias, bool def) {
    if (qwen35moe_env_has_value(primary)) {
        return qwen35moe_env_bool_default(primary, def);
    }
    if (qwen35moe_env_has_value(alias)) {
        return qwen35moe_env_bool_default(alias, def);
    }
    return def;
}

int qwen35moe_env_i32(const char * name, int def) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env ? (int) v : def;
}

bool qwen35moe_prefix_state_candidate_trace_enabled(int il) {
    if (!qwen35moe_env_enabled("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE")) {
        return false;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35moe_prefix_state_reconstruct_copy_enabled(int il) {
    if (!qwen35moe_env_enabled("LLAMA_MTP_PREFIX_STATE_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_STATE_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35moe_prefix_state_source_trace_enabled(int il) {
    if (!qwen35moe_env_enabled("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE")) {
        return false;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35moe_prefix_r_direct_reconstruct_copy_enabled(int il) {
    if (!qwen35moe_env_enabled("LLAMA_MTP_PREFIX_R_DIRECT_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_R_DIRECT_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35moe_prefix_s_state_only_reconstruct_copy_enabled(int il) {
    if (!qwen35moe_env_enabled("LLAMA_MTP_PREFIX_S_STATE_ONLY_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_S_STATE_ONLY_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35moe_prefix_accepted_row_only_commit_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT");
}

bool qwen35moe_prefix_layer_batch_enabled() {
    // Disabled: generic layer-batching changes recurrent-state bytes vs the
    // serial oracle.  Stage3 uses LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH instead,
    // which keeps every state-writing recurrent/attention step token-major and
    // batches only the final state-free verifier tail.
    return false;
}

bool qwen35moe_prefix_exact_tail_batch_log_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH_LOG");
}

bool qwen35moe_prefix_stage43_router_topk_fused_requested() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_FUSED") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_WEIGHTS") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_FUSED_ROUTER_TOPK_WEIGHTS") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_TOPK_FUSED") ||
           qwen35moe_env_enabled("LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE") ||
           qwen35moe_env_enabled("LLAMA_MTP_ROWEQ_ROUTER_TOPK_WEIGHT_FUSION_ACTIVE");
}

bool qwen35moe_prefix_stage42_router_topk_requested() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_BISECT") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_TOPK_BISECT") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_MMVF") ||
           qwen35moe_env_enabled("LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE") ||
           qwen35moe_prefix_stage43_router_topk_fused_requested();
}

bool qwen35moe_prefix_roweq_stage41_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_COMPONENT_BISECT") ||
           qwen35moe_prefix_stage42_router_topk_requested();
}

bool qwen35moe_prefix_roweq_layer_ffn_batch_log_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH_LOG") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH_LOG") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH_LOG") ||
           qwen35moe_prefix_roweq_stage41_diag_enabled();
}

bool qwen35moe_prefix_stage41_serial_default() {
    // Stage4.1 keeps its conservative default: all MoE/FFN component valves are
    // row-serial unless the operator explicitly opts out. Stage4.2 is a narrower
    // router/top-k follow-up, so it defaults only top-k/weights row-serial and
    // leaves router dense eligible for the small-N backend route.
    return !qwen35moe_prefix_stage42_router_topk_requested() &&
           qwen35moe_prefix_roweq_stage41_diag_enabled() &&
           !qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL");
}

bool qwen35moe_prefix_stage42_router_topk_default() {
    return qwen35moe_prefix_stage42_router_topk_requested() &&
           !qwen35moe_prefix_stage43_router_topk_fused_requested() &&
           !qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL");
}

bool qwen35moe_prefix_stage43_router_topk_fused_default() {
    return qwen35moe_prefix_stage43_router_topk_fused_requested() &&
           !qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_NO_DEFAULT_SERIAL");
}

bool qwen35moe_prefix_stage41_serial_router_enabled() {
    const bool def = qwen35moe_prefix_stage41_serial_default();
    if (qwen35moe_prefix_stage43_router_topk_fused_default()) {
        return qwen35moe_env_bool_default2("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE",
                "LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER", false);
    }
    return qwen35moe_env_bool_default2("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER_DENSE",
            "LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTER", def);
}

bool qwen35moe_prefix_stage41_serial_topk_weights_enabled() {
    const bool def = qwen35moe_prefix_stage41_serial_default() || qwen35moe_prefix_stage42_router_topk_default();
    if (qwen35moe_prefix_stage43_router_topk_fused_default()) {
        return qwen35moe_env_bool_default2("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS",
                "LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK", false);
    }
    return qwen35moe_env_bool_default2("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK_WEIGHTS",
            "LLAMA_MTP_PREFIX_ROWEQ_SERIAL_TOPK", def);
}

bool qwen35moe_prefix_stage41_serial_routed_glue_enabled() {
    return qwen35moe_env_bool_default("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_ROUTED_GLUE",
            qwen35moe_prefix_stage41_serial_default());
}

bool qwen35moe_prefix_stage41_serial_expert_agg_enabled() {
    const bool def = qwen35moe_prefix_stage41_serial_default();
    return qwen35moe_env_bool_default("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_AGG", def) ||
           qwen35moe_env_bool_default("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_EXPERT_WEIGHT_AGG", def);
}

bool qwen35moe_prefix_stage41_serial_shared_gate_enabled() {
    return qwen35moe_env_bool_default("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_GATE",
            qwen35moe_prefix_stage41_serial_default());
}

bool qwen35moe_prefix_stage41_serial_shared_ffn_enabled() {
    return qwen35moe_env_bool_default("LLAMA_MTP_PREFIX_ROWEQ_SERIAL_SHARED_FFN",
            qwen35moe_prefix_stage41_serial_default());
}

bool qwen35moe_prefix_stage41_batch_routed_projections_enabled() {
    return qwen35moe_env_bool_default2("LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS",
            "LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED", true);
}

int qwen35moe_prefix_stage41_batch_routed_projections_max_rows() {
    // Batched routed projections are a diagnostic-only optimization.  They pass
    // some narrow n_tokens=3 gates but are not robust across verifier width/state
    // sequences; keep the repaired serial-row fallback as the safe default.
    return qwen35moe_env_i32("LLAMA_MTP_PREFIX_ROWEQ_BATCH_ROUTED_PROJECTIONS_MAX_ROWS", 0);
}

bool qwen35moe_prefix_stage41_batch_routed_projections_enabled_for_rows(int64_t n_rows) {
    if (!qwen35moe_prefix_stage41_batch_routed_projections_enabled()) {
        return false;
    }
    const int max_rows = qwen35moe_prefix_stage41_batch_routed_projections_max_rows();
    return max_rows > 0 && n_rows <= max_rows;
}

bool qwen35moe_prefix_moe_routed_lanes_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_gather_scatter_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_GATHER_SCATTER_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_GATHER_SCATTER_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_row_slot_map_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_ROW_SLOT_MAP_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_ROW_SLOT_MAP_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_expert_bounds_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_EXPERT_BOUNDS_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_EXPERT_BOUNDS_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_f32_projection_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_F32_PROJ_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_F32_PROJ_SHADOW_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_quant_projection_shadow_diag_enabled(enum ggml_type type) {
    const bool supported = type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S;
    if (!supported) {
        return false;
    }

    if (qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_QUANT_PROJ_SHADOW_DIAG") ||
        qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_QUANT_PROJ_SHADOW_DIAG")) {
        return true;
    }

    switch (type) {
        case GGML_TYPE_Q8_0:
            return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_Q8_0_PROJ_SHADOW_DIAG") ||
                   qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_Q8_0_PROJ_SHADOW_DIAG");
        case GGML_TYPE_IQ4_XS:
            return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_IQ4_XS_PROJ_SHADOW_DIAG") ||
                   qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_IQ4_XS_PROJ_SHADOW_DIAG");
        case GGML_TYPE_IQ3_S:
            return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_IQ3_S_PROJ_SHADOW_DIAG") ||
                   qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_IQ3_S_PROJ_SHADOW_DIAG");
        default:
            return false;
    }
}

bool qwen35moe_prefix_moe_routed_lanes_slot_projection_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_SLOT_PROJ_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_SLOT_PROJ_SHADOW_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_down_projection_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_DOWN_PROJ_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_DOWN_PROJ_SHADOW_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_quant_down_projection_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_QUANT_DOWN_PROJ_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_QUANT_DOWN_PROJ_SHADOW_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_down_aggregation_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_DOWN_AGG_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_DOWN_AGG_SHADOW_DIAG");
}

bool qwen35moe_prefix_moe_routed_lanes_full_branch_shadow_diag_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_ROUTED_LANES_FULL_BRANCH_SHADOW_DIAG") ||
           qwen35moe_env_enabled("LLAMA_MTP_MOE_ROUTED_LANES_FULL_BRANCH_SHADOW_DIAG");
}

std::string qwen35moe_prefix_stage41_row_suffix(int64_t row_for_name) {
    if (row_for_name >= 0) {
        return "_row" + std::to_string((long long) row_for_name);
    }
    if (row_for_name == -2) {
        return "_row_serial_logits_all";
    }
    if (row_for_name == -3) {
        return "_batched_logits_serial_topk_all";
    }
    if (row_for_name == -4) {
        return "_stage43_fused_router_topk_rows_all";
    }
    if (row_for_name == -5) {
        return "_stage43_fused_router_topk_not_allowed_rows_all";
    }
    return "_all";
}

bool qwen35moe_prefix_hidden_trace_enabled() {
    return qwen35moe_env_enabled("LLAMA_MTP_PREFIX_HIDDEN_TRACE") ||
           qwen35moe_env_enabled("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE");
}

thread_local bool qwen35moe_prefix_moe_row_shadow_active = false;
thread_local int64_t qwen35moe_prefix_moe_row_shadow_source_row = -1;

int qwen35moe_prefix_roweq_first_layer(int n_transformer_layers) {
    int first = qwen35moe_env_i32("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_FIRST_LAYER",
            qwen35moe_env_i32("LLAMA_MTP_PREFIX_EXACT_ROWEQ_FIRST_LAYER",
                qwen35moe_env_i32("LLAMA_MTP_PREFIX_ROWEQ_LAYER_FIRST", 0)));
    if (first < 0) {
        first = 0;
    }
    if (first >= n_transformer_layers) {
        first = n_transformer_layers - 1;
    }
    return first;
}

int qwen35moe_prefix_roweq_last_layer(int n_transformer_layers) {
    int last = qwen35moe_env_i32("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_LAST_LAYER",
            qwen35moe_env_i32("LLAMA_MTP_PREFIX_EXACT_ROWEQ_LAST_LAYER",
                qwen35moe_env_i32("LLAMA_MTP_PREFIX_ROWEQ_LAYER_LAST", n_transformer_layers - 1)));
    if (last < 0) {
        last = 0;
    }
    if (last >= n_transformer_layers) {
        last = n_transformer_layers - 1;
    }
    return last;
}

bool qwen35moe_prefix_roweq_layer_enabled(int il, int n_transformer_layers) {
    if (n_transformer_layers <= 0) {
        return false;
    }
    const int first = qwen35moe_prefix_roweq_first_layer(n_transformer_layers);
    const int last  = qwen35moe_prefix_roweq_last_layer(n_transformer_layers);
    return first <= last && il >= first && il <= last;
}

void qwen35moe_lm_head_top1_apply_eog_mask(const llama_model & model, ggml_tensor * top1) {
    top1->op_params[0] = 0;
    const char * eog_mask_env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_EOG_MASK");
    if (eog_mask_env && atoi(eog_mask_env) == 0) {
        return;
    }

    int32_t n_ban = 0;
    const int32_t n_vocab = model.vocab.n_tokens();
    for (llama_token tok = 0; tok < n_vocab && n_ban < 8; ++tok) {
        if (model.vocab.is_eog(tok)) {
            top1->op_params[1 + n_ban] = tok;
            ++n_ban;
        }
    }
    top1->op_params[0] = n_ban;

    if (qwen35moe_env_enabled("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
        fprintf(stderr, "MTP_TARGET_TOP1_EOG_MASK: tensor=%s n_ban=%d", top1->name, (int) n_ban);
        for (int32_t i = 0; i < n_ban; ++i) {
            fprintf(stderr, " %d", top1->op_params[1 + i]);
        }
        fprintf(stderr, "\n");
    }
}

ggml_tensor * qwen35moe_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;
    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

}

void llama_model_qwen35moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");
    hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    {
        const uint32_t n_main = hparams.n_layer - hparams.nextn_predict_layers;
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
            hparams.recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer - hparams.nextn_predict_layers) {
        case 40: type = LLM_TYPE_35B_A3B; break;
        case 48: type = LLM_TYPE_122B_A10B; break;
        case 60: type = LLM_TYPE_397B_A17B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_main = n_layer - hparams.nextn_predict_layers;
    const bool mtp_only   = (hparams.nextn_predict_layers > 0) &&
                            (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recurrent(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, 0);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, 0);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, 0);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, 0);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              0);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              0);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < (int) n_main; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = (int) n_main; i < n_layer; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY ||
        params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_COMMIT) {
        return std::make_unique<graph_prefix_verify>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids && cparams.embeddings_pre_norm_masked) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cb(cur, "h_pre_norm", -1);
    res->t_h_pre_norm = cur;

    if (cparams.embeddings_pre_norm) {
        ggml_tensor * h_mtp_capture = ggml_dup(ctx0, cur);
        cb(h_mtp_capture, "mtp_h_capture", -1);
        res->t_mtp_h_capture = h_mtp_capture;
        ggml_build_forward_expand(gf, h_mtp_capture);
    }

    if (!cparams.embeddings_pre_norm_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_tensor * lm_head_input = cur;

    const char * target_top1_active = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE");
    const char * target_top1_active_raw = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE");
    const char * target_top1_active_logits = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_LOGITS");
    const bool lm_head_top1_direct_supported =
            (model.output->type == GGML_TYPE_Q6_K || model.output->type == GGML_TYPE_Q8_0) &&
            (loras == nullptr || loras->empty());
    const bool target_top1_active_flag = target_top1_active ? atoi(target_top1_active) != 0 : cparams.embeddings_pre_norm;
    const bool target_top1_raw_flag = target_top1_active_raw ? atoi(target_top1_active_raw) != 0 : cparams.embeddings_pre_norm;
    const bool target_top1_logits_flag = target_top1_active_logits ? atoi(target_top1_active_logits) != 0 : cparams.embeddings_pre_norm;
    const bool target_top1_active_enabled = target_top1_active_flag && target_top1_raw_flag && lm_head_top1_direct_supported;
    const bool target_top1_active_from_logits = target_top1_active_enabled && target_top1_logits_flag;
    if (target_top1_active_enabled && !target_top1_active_from_logits) {
        ggml_tensor * sampled = ggml_lm_head_top_k(ctx0, model.output, lm_head_input, 1);
        cb(sampled, "target_lm_head_top1_active", -1);
        qwen35moe_lm_head_top1_apply_eog_mask(model, sampled);
        res->t_mtp_target_top1_fused_all = sampled;
        ggml_build_forward_expand(gf, sampled);
        return;
    }

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    if (target_top1_active_from_logits) {
        ggml_tensor * sampled = ggml_top_k(ctx0, cur, 1);
        cb(sampled, "target_logits_top1_active", -1);
        qwen35moe_lm_head_top1_apply_eog_mask(model, sampled);
        res->t_mtp_target_top1_fused_all = sampled;
        ggml_build_forward_expand(gf, sampled);
    }

    const char * target_top1_shadow = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW");
    if (target_top1_shadow && atoi(target_top1_shadow) != 0 && lm_head_top1_direct_supported) {
        const char * target_top1_shadow_no_cont = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_NO_CONT");
        ggml_tensor * lm_head_input_for_top1 = lm_head_input;
        if (!(target_top1_shadow_no_cont && atoi(target_top1_shadow_no_cont) != 0)) {
            lm_head_input_for_top1 = ggml_cont(ctx0, lm_head_input);
            cb(lm_head_input_for_top1, "target_lm_head_input_cont_shadow", -1);
        }
        ggml_tensor * fused_top1 = ggml_lm_head_top_k(ctx0, model.output, lm_head_input_for_top1, 1);
        cb(fused_top1, "target_lm_head_top1_fused", -1);
        res->t_mtp_target_top1_fused_all = fused_top1;
        ggml_build_forward_expand(gf, fused_top1);
    }

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35moe::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present - following Qwen3Next reference implementation
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating as in the reference implementation
        // The shared expert has its own gate that is sigmoided
        // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        // Apply sigmoid to the gate
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);


        // Apply the gate to the shared expert output
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

llama_model_qwen35moe::graph_prefix_verify::graph_prefix_verify(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const bool commit_only = params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_COMMIT;
    GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY || commit_only);
    GGML_ASSERT(!commit_only || params.mtp_prefix_commit_slot >= 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_seqs == 1);
    GGML_ASSERT(ubatch.n_seq_tokens == ubatch.n_tokens);
    GGML_ASSERT(cparams.n_rs_seq > 0);
    GGML_ASSERT(ubatch.n_tokens <= 1 + cparams.n_rs_seq);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "prefix_model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos = build_inp_pos();

    const int n_transformer_layers = n_layer - (int) hparams.nextn_predict_layers;
    std::vector<ggml_tensor *> conv_state(n_transformer_layers, nullptr);
    std::vector<ggml_tensor *> ssm_state (n_transformer_layers, nullptr);

    std::vector<ggml_tensor *> inp_pos_rows((size_t) ubatch.n_tokens, nullptr);
    for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
        ggml_tensor * inp_pos_row = nullptr;
        for (uint32_t p = 0; p < ubatch.n_pos; ++p) {
            ggml_tensor * pos_comp = ggml_view_1d(ctx0, inp_pos, 1, (p * (int64_t) ubatch.n_tokens + row) * inp_pos->nb[0]);
            inp_pos_row = inp_pos_row ? ggml_concat(ctx0, inp_pos_row, pos_comp, 0) : pos_comp;
        }
        cb(inp_pos_row, "prefix_pos_row", -1);
        inp_pos_rows[(size_t) row] = inp_pos_row;
    }

    const bool skip_verify_snapshots = !commit_only && qwen35moe_prefix_accepted_row_only_commit_enabled();
    const bool batch_output_head = !commit_only && params.mtp_prefix_batch_output_head;
    // Accepted-row-only mode cannot know the accepted length until after
    // sampling.  Materialize a small suffix of verifier rows into rollback
    // slots during verifier decode; covered rollback values can commit without
    // a checkpoint restore or second prefix pass, and uncovered partial accepts
    // fall back to the commit graph.
    const int64_t snapshot_slot_limit    = skip_verify_snapshots ? params.mtp_prefix_accepted_commit_verify_slots : 0;
    const int64_t snapshot_slot_override = commit_only ? params.mtp_prefix_commit_slot : -1;

    // Stage3 only batches the state-free verifier tail and keeps the prefix
    // output-head path in full-logits mode.  Direct top1 is a separate diagnostic
    // path in the decode graph; enabling it here would remove logits needed by
    // the serial verifier sampler/compare gates.
    const bool target_top1_active_direct = false;

    const bool roweq_layer_ffn_batch =
        !commit_only && params.mtp_prefix_roweq_layer_ffn_batch && n_transformer_layers > 0 && ubatch.n_tokens > 1;
    const bool exact_tail_batch =
        !roweq_layer_ffn_batch &&
        !commit_only && params.mtp_prefix_exact_tail_batch && n_transformer_layers > 0 && ubatch.n_tokens > 1;

    if (roweq_layer_ffn_batch && qwen35moe_prefix_roweq_layer_ffn_batch_log_enabled()) {
        fprintf(stderr,
                "MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH(qwen35moe): n_tokens=%u layers=%d..%d batch_output_head=%d direct_top1=%d stage41=%d stage42_router_topk=%d stage43_router_topk_fused=%d router_topk_split=1 batch_routed_proj0_fallback=serial_rows serial_router=%d serial_router_dense=%d serial_topk_weights=%d serial_routed_glue=%d serial_expert_agg=%d serial_shared_gate=%d serial_shared_ffn=%d batch_routed_proj=%d router_topk_mode=%s contract=serial_attention_router_topk_split_component_bisect_ffn\n",
                (unsigned) ubatch.n_tokens,
                qwen35moe_prefix_roweq_first_layer(n_transformer_layers),
                qwen35moe_prefix_roweq_last_layer(n_transformer_layers),
                batch_output_head ? 1 : 0, target_top1_active_direct ? 1 : 0,
                qwen35moe_prefix_roweq_stage41_diag_enabled() ? 1 : 0,
                qwen35moe_prefix_stage42_router_topk_requested() ? 1 : 0,
                qwen35moe_prefix_stage43_router_topk_fused_requested() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_router_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_router_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_topk_weights_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_routed_glue_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_expert_agg_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_shared_gate_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_serial_shared_ffn_enabled() ? 1 : 0,
                qwen35moe_prefix_stage41_batch_routed_projections_enabled_for_rows(ubatch.n_tokens) ? 1 : 0,
                qwen35moe_prefix_stage43_router_topk_fused_requested() ? "router_mmvf_roweq_fused_topk_weights_candidate" :
                    (qwen35moe_prefix_stage42_router_topk_requested() ? "router_dense_mmvf_topk_serial_candidate" : "component_bisect"));
    }

    if (exact_tail_batch && qwen35moe_prefix_exact_tail_batch_log_enabled()) {
        fprintf(stderr,
                "MTP_PREFIX_EXACT_TAIL_BATCH(qwen35moe): n_tokens=%u tail_layer=%d batch_output_head=%d direct_top1=%d\n",
                (unsigned) ubatch.n_tokens, n_transformer_layers - 1, batch_output_head ? 1 : 0,
                target_top1_active_direct ? 1 : 0);
    }

    ggml_tensor * h_pre_norm_all = nullptr;
    ggml_tensor * embd_all       = nullptr;
    ggml_tensor * logits_all     = nullptr;

    if (roweq_layer_ffn_batch) {
        // Stage4 exact verifier candidate.
        //
        // State writes remain token-major/row-serial.  The only per-layer work
        // batched before later state writes is the post-attention FFN/MoE, and
        // the server installs row-equivalent small-N MMVQ policy for those
        // projections.  In other words, this graph batches the expensive
        // state-free segment between two state barriers, but every column that
        // can influence a future recurrent/KV write is computed with the same
        // ncols_dst=1 arithmetic shape as the serial oracle.
        //
        // Layer range envs are a diagnostic bisect valve.  Layers outside the
        // enabled range use the layer-major serial row path, so a target run can
        // narrow any remaining state_match=0 source without changing scheduling
        // or commit semantics.
        ggml_tensor * cur_all = inpL;
        cb(cur_all, "prefix_roweq_input_batched", -1);

        for (int il = 0; il < n_transformer_layers; ++il) {
            const bool batch_this_layer = qwen35moe_prefix_roweq_layer_enabled(il, n_transformer_layers);

            if (batch_this_layer) {
                ggml_tensor * inpSA_all = cur_all;
                ggml_tensor * ffn_residual_all = nullptr;
                ggml_tensor * attn_post_norm_all = nullptr;

                for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                    ggml_tensor * inpSA = ggml_view_2d(ctx0, inpSA_all, n_embd, 1, inpSA_all->nb[1], row * inpSA_all->nb[1]);
                    cb(inpSA, "prefix_roweq_input_row_state_barrier", il);

                    ggml_tensor * cur = build_norm(inpSA, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                    cb(cur, "prefix_roweq_attn_norm_row_state_barrier", il);

                    ggml_build_forward_expand(gf, cur);

                    if (hparams.is_recurrent(il)) {
                        cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                                commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                    } else {
                        cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                    }

                    cur = ggml_add(ctx0, cur, inpSA);
                    cb(cur, "prefix_roweq_attn_residual_row_state_barrier", il);
                    ffn_residual_all = ffn_residual_all ? ggml_concat(ctx0, ffn_residual_all, cur, 1) : cur;

                    ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                    cb(attn_post_norm, "prefix_roweq_attn_post_norm_row_state_barrier", il);
                    attn_post_norm_all = attn_post_norm_all ? ggml_concat(ctx0, attn_post_norm_all, attn_post_norm, 1) : attn_post_norm;
                }
                cb(ffn_residual_all, "prefix_roweq_attn_residual_batched_input", il);
                cb(attn_post_norm_all, "prefix_roweq_attn_post_norm_batched_input", il);

                cur_all = qwen35moe_prefix_roweq_stage41_diag_enabled() ?
                    build_layer_ffn_stage41(attn_post_norm_all, il) :
                    build_layer_ffn(attn_post_norm_all, il);
                cb(cur_all, qwen35moe_prefix_roweq_stage41_diag_enabled() ? "prefix41_roweq_ffn_out_component_bisect" : "prefix_roweq_ffn_out_roweq_batched", il);

                cur_all = ggml_add(ctx0, cur_all, ffn_residual_all);
                cb(cur_all, "prefix_roweq_post_moe_roweq_batched", il);

                cur_all = build_cvec(cur_all, il);
                cb(cur_all, qwen35moe_prefix_roweq_stage41_diag_enabled() ? "prefix41_roweq_l_out_component_bisect" : "prefix_roweq_l_out_roweq_batched", il);
                trace_prefix_hidden_rows(cur_all, qwen35moe_prefix_roweq_stage41_diag_enabled() ? "prefix41_hidden_pre_next_state" : "prefix_roweq_hidden_pre_next_state", il);
            } else {
                ggml_tensor * next_all = nullptr;
                for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                    ggml_tensor * cur = ggml_view_2d(ctx0, cur_all, n_embd, 1, cur_all->nb[1], row * cur_all->nb[1]);
                    cb(cur, "prefix_roweq_input_row_serial_layer", il);

                    ggml_tensor * inpSA = cur;

                    cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                    cb(cur, "prefix_roweq_attn_norm_serial_layer", il);

                    ggml_build_forward_expand(gf, cur);

                    if (hparams.is_recurrent(il)) {
                        cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                                commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                    } else {
                        cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                    }

                    cur = ggml_add(ctx0, cur, inpSA);
                    cb(cur, "prefix_roweq_attn_residual_serial_layer", il);

                    ggml_tensor * ffn_residual = cur;

                    ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                    cb(attn_post_norm, "prefix_roweq_attn_post_norm_serial_layer", il);

                    cur = build_layer_ffn(attn_post_norm, il);
                    cb(cur, "prefix_roweq_ffn_out_serial_layer", il);

                    cur = ggml_add(ctx0, cur, ffn_residual);
                    cb(cur, "prefix_roweq_post_moe_serial_layer", il);

                    cur = build_cvec(cur, il);
                    cb(cur, "prefix_roweq_l_out_serial_layer", il);

                    next_all = next_all ? ggml_concat(ctx0, next_all, cur, 1) : cur;
                }
                cur_all = next_all;
                cb(cur_all, "prefix_roweq_l_out_serial_layer_all", il);
                trace_prefix_hidden_rows(cur_all, "prefix_roweq_hidden_pre_next_state_serial_layer", il);
            }
        }

        h_pre_norm_all = cur_all;
        cb(h_pre_norm_all, "prefix_roweq_h_pre_norm_batched", -1);

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_roweq_h_pre_norm_row", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active_direct) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
            }
        }
    } else if (exact_tail_batch) {
        // Stage3 exact verifier candidate.
        //
        // The previous fully layer-batched prefix graph produced matching tokens but
        // state_match=0 because batched FFN/MoE numerics fed later recurrent-state
        // writes.  This graph preserves serial recurrent-state semantics by running
        // every layer attention/recurrent state-writing step and every non-tail FFN
        // token-major.  Only the last layer's post-attention FFN/MoE and optional
        // output head are batched, after all verifier state writes have already
        // happened.  Therefore hidden/logit numerics may still differ from the
        // byte-for-byte serial oracle, but committed recurrent/KV state bytes must
        // not depend on that batched tail.
        ggml_tensor * ffn_residual_tail_all = nullptr;
        const int il_tail = n_transformer_layers - 1;

        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], row * inpL->nb[1]);
            cb(cur, "prefix_exact_tail_input_row", -1);

            for (int il = 0; il < n_transformer_layers; ++il) {
                ggml_tensor * inpSA = cur;

                cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "prefix_exact_tail_attn_norm", il);

                ggml_build_forward_expand(gf, cur);

                if (hparams.is_recurrent(il)) {
                    cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                            commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                } else {
                    cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                }

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "prefix_exact_tail_attn_residual", il);

                if (il == il_tail) {
                    ffn_residual_tail_all = ffn_residual_tail_all ? ggml_concat(ctx0, ffn_residual_tail_all, cur, 1) : cur;
                    break;
                }

                ggml_tensor * ffn_residual = cur;

                ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_post_norm, "prefix_exact_tail_attn_post_norm_serial", il);

                cur = build_layer_ffn(attn_post_norm, il);
                cb(cur, "prefix_exact_tail_ffn_out_serial", il);

                cur = ggml_add(ctx0, cur, ffn_residual);
                cb(cur, "prefix_exact_tail_post_moe_serial", il);

                cur = build_cvec(cur, il);
                cb(cur, "prefix_exact_tail_l_out_serial", il);
            }
        }

        GGML_ASSERT(ffn_residual_tail_all != nullptr);
        cb(ffn_residual_tail_all, "prefix_exact_tail_attn_residual_batched_input", il_tail);

        ggml_tensor * attn_post_norm_tail = build_norm(ffn_residual_tail_all, model.layers[il_tail].attn_post_norm, nullptr, LLM_NORM_RMS, il_tail);
        cb(attn_post_norm_tail, "prefix_exact_tail_attn_post_norm_batched", il_tail);

        ggml_tensor * cur_tail_all = build_layer_ffn(attn_post_norm_tail, il_tail);
        cb(cur_tail_all, "prefix_exact_tail_ffn_out_batched", il_tail);

        cur_tail_all = ggml_add(ctx0, cur_tail_all, ffn_residual_tail_all);
        cb(cur_tail_all, "prefix_exact_tail_post_moe_batched", il_tail);

        h_pre_norm_all = build_cvec(cur_tail_all, il_tail);
        cb(h_pre_norm_all, "prefix_exact_tail_h_pre_norm_batched", -1);

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_exact_tail_h_pre_norm_row_batched", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active_direct) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
            }
        }
    } else if (qwen35moe_prefix_layer_batch_enabled()) {
        ggml_tensor * cur_all = inpL;
        cb(cur_all, "prefix_input_batched", -1);

        for (int il = 0; il < n_transformer_layers; ++il) {
            ggml_tensor * inpSA_all = cur_all;

            ggml_tensor * attn_norm_all = build_norm(cur_all, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(attn_norm_all, "prefix_attn_norm_batched", il);
            ggml_build_forward_expand(gf, attn_norm_all);

            ggml_tensor * ffn_residual_all = nullptr;
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * inpSA = ggml_view_2d(ctx0, inpSA_all, n_embd, 1, inpSA_all->nb[1], row * inpSA_all->nb[1]);
                cb(inpSA, "prefix_input_row_batched_layer", il);

                ggml_tensor * cur = ggml_view_2d(ctx0, attn_norm_all, n_embd, 1, attn_norm_all->nb[1], row * attn_norm_all->nb[1]);
                cb(cur, "prefix_attn_norm_row_batched_layer", il);

                if (hparams.is_recurrent(il)) {
                    cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                            commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                } else {
                    cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                }

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "prefix_attn_residual_row_batched_layer", il);
                ffn_residual_all = ffn_residual_all ? ggml_concat(ctx0, ffn_residual_all, cur, 1) : cur;
            }
            cb(ffn_residual_all, "prefix_attn_residual_batched", il);

            ggml_tensor * attn_post_norm = build_norm(ffn_residual_all, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(attn_post_norm, "prefix_attn_post_norm_batched", il);

            cur_all = build_layer_ffn(attn_post_norm, il);
            cb(cur_all, "prefix_ffn_out_batched", il);

            cur_all = ggml_add(ctx0, cur_all, ffn_residual_all);
            cb(cur_all, "prefix_post_moe_batched", il);

            cur_all = build_cvec(cur_all, il);
            cb(cur_all, "prefix_l_out_batched", il);
        }

        h_pre_norm_all = cur_all;
        cb(h_pre_norm_all, "prefix_h_pre_norm_batched", -1);

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_h_pre_norm_row_batched", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                cb(logits_row, "prefix_result_output_row", -1);
                logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
            }
        }
    } else {
        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], row * inpL->nb[1]);
            cb(cur, "prefix_input_row", -1);

            for (int il = 0; il < n_transformer_layers; ++il) {
                ggml_tensor * inpSA = cur;

                cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "prefix_attn_norm", il);

                ggml_build_forward_expand(gf, cur);

                if (hparams.is_recurrent(il)) {
                    cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                            commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                } else {
                    cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                }

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "prefix_attn_residual", il);

                ggml_tensor * ffn_residual = cur;

                ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_post_norm, "prefix_attn_post_norm", il);

                cur = build_layer_ffn(attn_post_norm, il);
                cb(cur, "prefix_ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_residual);
                cb(cur, "prefix_post_moe", il);

                cur = build_cvec(cur, il);
                cb(cur, "prefix_l_out", il);
            }

            cb(cur, "prefix_h_pre_norm_row", -1);
            h_pre_norm_all = h_pre_norm_all ? ggml_concat(ctx0, h_pre_norm_all, cur, 1) : cur;

            if (!commit_only && !batch_output_head) {
                ggml_tensor * embd_row = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                cb(logits_row, "prefix_result_output_row", -1);
                logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
            }
        }
    }

    if (batch_output_head) {
        embd_all = build_norm(h_pre_norm_all, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(embd_all, "prefix_result_norm_batched", -1);

        logits_all = build_lora_mm(model.output, embd_all);
        cb(logits_all, "prefix_result_output_batched", -1);
    }

    res->t_h_pre_norm = h_pre_norm_all;
    res->t_embd       = embd_all;
    res->t_logits     = logits_all;

    ggml_build_forward_expand(gf, h_pre_norm_all);
    if (!commit_only) {
        ggml_build_forward_expand(gf, embd_all);
        ggml_build_forward_expand(gf, logits_all);
    }
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_attn_prefix_row(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        float                     kq_scale,
        int                       il,
        int64_t                   row) {
    if (inp_attn->self_k_rot) {
        q_cur = qwen35moe_mul_mat_aux(ctx0, q_cur, inp_attn->self_k_rot);
        k_cur = qwen35moe_mul_mat_aux(ctx0, k_cur, inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        v_cur = qwen35moe_mul_mat_aux(ctx0, v_cur, inp_attn->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp_attn->mctx;

    ggml_tensor * k_idxs = ggml_view_1d(ctx0, inp_attn->get_k_idxs(), 1, row * inp_attn->get_k_idxs()->nb[0]);

    const int64_t v_idx_span = inp_attn->get_v_idxs()->ne[0] / n_tokens;
    GGML_ASSERT(v_idx_span * n_tokens == inp_attn->get_v_idxs()->ne[0]);
    ggml_tensor * v_idxs = ggml_view_1d(ctx0, inp_attn->get_v_idxs(), v_idx_span, row * v_idx_span * inp_attn->get_v_idxs()->nb[0]);

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    kq_mask = ggml_view_4d(ctx0, kq_mask,
            kq_mask->ne[0], 1, kq_mask->ne[2], kq_mask->ne[3],
            kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], row * kq_mask->nb[1]);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    const bool use_fa = (cparams.flash_attn || k->type == GGML_TYPE_I32);
    ggml_tensor * v = mctx_cur->get_v(
            ctx0,
            il,
            use_fa ? LLAMA_KV_V_LAYOUT_FOR_FA
                   : LLAMA_KV_V_LAYOUT_FOR_NON_FA);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, il);
    cb(cur, "prefix_kqv_out", il);

    if (inp_attn->self_v_rot) {
        cur = qwen35moe_mul_mat_aux(ctx0, cur, inp_attn->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_layer_attn_prefix_row(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos_row,
        int *                     sections,
        int                       il,
        int64_t                   row) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    cb(Qcur_full, "prefix_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, 1,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "prefix_Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "prefix_Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "prefix_Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "prefix_Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, 1);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "prefix_Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, 1,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, 1);
    cb(gate, "prefix_gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, 1);

    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos_row, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos_row, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "prefix_Qcur", il);
    cb(Kcur, "prefix_Kcur", il);
    cb(Vcur, "prefix_Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn_prefix_row(inp_attn, Qcur, Kcur, Vcur, kq_scale, il, row);
    cb(cur, "prefix_attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "prefix_gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "prefix_attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "prefix_attn_output", il);

    return cur;
}

void llama_model_qwen35moe::graph_prefix_verify::copy_prefix_snapshot(
        llm_graph_input_rs * inp,
        ggml_tensor *        snapshot,
        ggml_tensor *        states_all,
        int64_t              state_size,
        int                  il,
        int64_t              row,
        int64_t              n_prefix,
        const char *         name,
        int64_t              slot_override) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();
    const uint32_t slot     = (uint32_t) (slot_override >= 0 ? slot_override : (n_prefix - 1 - row));

    const size_t row_size = state_size * ggml_element_size(states_all);
    ggml_tensor * dst = ggml_view_2d(ctx0, states_all, state_size, 1, states_all->nb[1], ((size_t) slot * mem_size + kv_head) * row_size);
    ggml_tensor * cpy = ggml_cpy(ctx0, snapshot, dst);
    const std::string copy_name = std::string("prefix_") + name + "_row" + std::to_string(row) + "_slot" + std::to_string(slot);
    cb(cpy, copy_name.c_str(), il);
    ggml_build_forward_expand(gf, cpy);
}

void llama_model_qwen35moe::graph_prefix_verify::copy_prefix_conv_reconstruct_snapshot_direct(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_state,
        ggml_tensor *        qkv_mixed,
        ggml_tensor *        conv_states_all,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il,
        int64_t              row,
        int64_t              n_prefix,
        int64_t              slot_override) {
    GGML_ASSERT(conv_kernel_size > 1);

    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();
    const uint32_t slot     = (uint32_t) (slot_override >= 0 ? slot_override : (n_prefix - 1 - row));

    const size_t elem_size = ggml_element_size(conv_states_all);
    const int64_t state_size = (conv_kernel_size - 1) * conv_channels;
    const size_t row_size = state_size * elem_size;
    const size_t base_off = ((size_t) slot * mem_size + kv_head) * row_size;
    const size_t dst_channel_stride = (size_t) (conv_kernel_size - 1) * elem_size;

    ggml_tensor * src_tail = ggml_view_3d(ctx0, conv_state, conv_kernel_size - 2, conv_channels, 1,
            conv_state->nb[1], conv_state->nb[2], ggml_element_size(conv_state));
    ggml_tensor * dst_tail = ggml_view_2d(ctx0, conv_states_all, conv_kernel_size - 2, conv_channels,
            dst_channel_stride, base_off);
    ggml_tensor * tail_cpy = ggml_cpy(ctx0, src_tail, dst_tail);
    const std::string tail_name = "prefix_conv_state_direct_tail_row" + std::to_string(row) + "_slot" + std::to_string(slot);
    cb(tail_cpy, tail_name.c_str(), il);
    ggml_build_forward_expand(gf, tail_cpy);

    ggml_tensor * dst_qkv = ggml_view_2d(ctx0, conv_states_all, 1, conv_channels,
            dst_channel_stride, base_off + (size_t) (conv_kernel_size - 2) * elem_size);
    // Use the pre-transpose [channel, 1] qkv tensor here.  Copying from the
    // [1, channel] transpose into a strided destination can hit CUDA's
    // transpose-specialized CPY path, which assumes a contiguous destination and
    // corrupts the channel-strided conv-state layout.
    ggml_tensor * qkv_cpy = ggml_cpy(ctx0, qkv_mixed, dst_qkv);
    const std::string qkv_name = "prefix_conv_state_direct_qkv_row" + std::to_string(row) + "_slot" + std::to_string(slot);
    cb(qkv_cpy, qkv_name.c_str(), il);
    ggml_build_forward_expand(gf, qkv_cpy);
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_layer_attn_linear_prefix_row(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        ggml_tensor *&       conv_state,
        ggml_tensor *&       ssm_state,
        int                  il,
        int64_t              row,
        int64_t              n_prefix,
        bool                 snapshot_selected_only,
        int64_t              snapshot_slot_limit,
        int64_t              snapshot_slot_override) {
    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], 1, 1);
    cb(qkv_mixed, "prefix_linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, cur, model.layers[il].wqkv_gate_s);
    cb(z, "prefix_z", il);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, 1, 1);
    cb(beta, "prefix_beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "prefix_beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, 1, 1);
    cb(alpha, "prefix_alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "prefix_a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "prefix_gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, 1, 1);

    ggml_tensor * conv_states_all = inp->mctx->get_r_l(il);
    ggml_tensor * ssm_states_all  = inp->mctx->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    if (conv_state == nullptr) {
        conv_state = build_rs(inp, conv_states_all, hparams.n_embd_r(), 1);
        cb(conv_state, "prefix_conv_states", il);
        conv_state = ggml_reshape_3d(ctx0, conv_state, conv_kernel_size - 1, conv_channels, 1);
        cb(conv_state, "prefix_conv_states_reshaped", il);
    }

    ggml_tensor * qkv_mixed_t = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed_t, "prefix_qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, qkv_mixed_t, 0);
    cb(conv_input, "prefix_conv_input", il);

    ggml_tensor * new_conv_state = ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, 1,
            conv_input->nb[1], conv_input->nb[2], ggml_element_size(conv_input));
    cb(new_conv_state, "prefix_last_conv_states", il);
    const bool prefix_state_candidate_trace    = qwen35moe_prefix_state_candidate_trace_enabled(il);
    const bool prefix_state_reconstruct_copy   = qwen35moe_prefix_state_reconstruct_copy_enabled(il);
    const bool prefix_state_source_trace       = qwen35moe_prefix_state_source_trace_enabled(il);
    const bool prefix_r_direct_reconstruct_copy =
        prefix_state_reconstruct_copy &&
        qwen35moe_prefix_r_direct_reconstruct_copy_enabled(il) &&
        !qwen35moe_env_enabled("LLAMA_MTP_PREFIX_SNAPSHOT_TRACE");
    const int64_t natural_snapshot_slot = n_prefix - 1 - row;
    const bool copy_snapshot_this_row = !snapshot_selected_only ||
        (snapshot_slot_override >= 0 ? row == n_prefix - 1 : natural_snapshot_slot < snapshot_slot_limit);
    const bool reconstruct_snapshot_this_row = prefix_state_reconstruct_copy && copy_snapshot_this_row;
    const uint32_t prefix_state_candidate_slot = (uint32_t) (snapshot_slot_override >= 0 ? snapshot_slot_override : natural_snapshot_slot);
    if (prefix_state_source_trace && reconstruct_snapshot_this_row) {
        ggml_build_forward_expand(gf, new_conv_state);
    }

    ggml_tensor * r_snapshot = new_conv_state;
    if (prefix_state_candidate_trace || reconstruct_snapshot_this_row) {
        ggml_tensor * r_candidate_prev_tail = ggml_view_3d(ctx0, conv_state, conv_kernel_size - 2, conv_channels, 1,
                conv_state->nb[1], conv_state->nb[2], ggml_element_size(conv_state));
        if (prefix_state_candidate_trace) {
            cb(r_candidate_prev_tail, "prefix_state_candidate_r_prev_tail", il);
        }
        ggml_tensor * r_candidate_reconstructed = ggml_concat(ctx0, r_candidate_prev_tail, qkv_mixed_t, 0);
        if (prefix_state_candidate_trace) {
            cb(r_candidate_reconstructed, "prefix_state_candidate_r_reconstructed", il);
            ggml_tensor * r_candidate = ggml_cont_2d(ctx0, r_candidate_reconstructed, (conv_kernel_size - 1) * conv_channels, 1);
            const std::string r_candidate_name = "prefix_state_candidate_r_row" + std::to_string(row) + "_slot" + std::to_string(prefix_state_candidate_slot);
            cb(r_candidate, r_candidate_name.c_str(), il);
            ggml_build_forward_expand(gf, r_candidate);
        }
        if (reconstruct_snapshot_this_row) {
            r_snapshot = r_candidate_reconstructed;
        }
    }
    if (copy_snapshot_this_row) {
        if (prefix_r_direct_reconstruct_copy) {
            copy_prefix_conv_reconstruct_snapshot_direct(inp, conv_state, qkv_mixed, conv_states_all, conv_kernel_size, conv_channels, il, row, n_prefix, snapshot_slot_override);
        } else {
            copy_prefix_snapshot(inp, r_snapshot, conv_states_all, (conv_kernel_size - 1) * conv_channels, il, row, n_prefix, "conv_state_copy", snapshot_slot_override);
        }
    }

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "prefix_conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "prefix_conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, 1, 1,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv, 0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, 1, 1,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, 1, 1,
            ggml_row_size(conv_qkv_mix->type, head_v_dim), nb1_qkv, nb1_qkv,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "prefix_q_conv", il);
    cb(k_conv, "prefix_k_conv", il);
    cb(v_conv, "prefix_v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, 1, 1);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, 1, 1);
    }

    cb(q_conv, "prefix_q_conv_predelta", il);
    cb(k_conv, "prefix_k_conv_predelta", il);
    cb(v_conv, "prefix_v_conv_predelta", il);

    if (ssm_state == nullptr) {
        ssm_state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), 1);
        ssm_state = ggml_reshape_4d(ctx0, ssm_state, head_v_dim, head_v_dim, num_v_heads, 1);
        cb(ssm_state, "prefix_state_predelta", il);
    }

    auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, ssm_state, il);
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "prefix_attn_output", il);
    cb(new_state, "prefix_new_state", il);
    if (prefix_state_source_trace && reconstruct_snapshot_this_row) {
        ggml_build_forward_expand(gf, new_state);
    }

    ggml_tensor * s_snapshot = new_state;
    if (prefix_state_candidate_trace || reconstruct_snapshot_this_row) {
        ggml_tensor * s_candidate_state = nullptr;
        const bool s_state_only_reconstruct_copy =
            reconstruct_snapshot_this_row &&
            qwen35moe_prefix_s_state_only_reconstruct_copy_enabled(il) &&
            cparams.fused_gdn_ar;
        if (s_state_only_reconstruct_copy) {
            s_candidate_state = build_delta_net_fused_state_only(q_conv, k_conv, v_conv, gate, beta, ssm_state, il);
        } else {
            auto s_candidate_attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, ssm_state, il);
            s_candidate_state = s_candidate_attn_out.second;
        }
        if (prefix_state_candidate_trace) {
            cb(s_candidate_state, "prefix_state_candidate_s_reconstructed", il);
            ggml_tensor * s_candidate = ggml_cont_2d(ctx0, s_candidate_state, hparams.n_embd_s(), 1);
            const std::string s_candidate_name = "prefix_state_candidate_s_row" + std::to_string(row) + "_slot" + std::to_string(prefix_state_candidate_slot);
            cb(s_candidate, s_candidate_name.c_str(), il);
            ggml_build_forward_expand(gf, s_candidate);
        }
        if (reconstruct_snapshot_this_row) {
            s_snapshot = s_candidate_state;
        }
    }
    if (copy_snapshot_this_row) {
        copy_prefix_snapshot(inp, s_snapshot, ssm_states_all, hparams.n_embd_s(), il, row, n_prefix, "ssm_state_copy", snapshot_slot_override);
    }

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, 1, 1);

    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, 1, 1);
    cb(final_output, "prefix_final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "prefix_linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, 1);

    conv_state = new_conv_state;
    ssm_state  = new_state;

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}


void llama_model_qwen35moe::graph_prefix_verify::trace_prefix_hidden_rows(ggml_tensor * cur, const char * stem, const int il) {
    if (!qwen35moe_prefix_hidden_trace_enabled() || cur == nullptr) {
        return;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER",
            qwen35moe_env_i32("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE_LAYER", -1));
    if (layer_filter >= 0 && layer_filter != il) {
        return;
    }

    const int64_t n_rows = cur->ne[1];
    for (int64_t row = 0; row < n_rows; ++row) {
        ggml_tensor * h_row = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], row * cur->nb[1]);
        const std::string name = std::string(stem) + "_row" + std::to_string((long long) row);
        cb(h_row, name.c_str(), il);
        ggml_build_forward_expand(gf, h_row);
    }
}

void llama_model_qwen35moe::graph_prefix_verify::trace_prefix_hidden_token_rows(ggml_tensor * cur, const char * stem, const int il) {
    if (!qwen35moe_prefix_hidden_trace_enabled() || cur == nullptr) {
        return;
    }
    const int layer_filter = qwen35moe_env_i32("LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER",
            qwen35moe_env_i32("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE_LAYER", -1));
    if (layer_filter >= 0 && layer_filter != il) {
        return;
    }

    const int64_t n_rows = cur->ne[2];
    for (int64_t row = 0; row < n_rows; ++row) {
        ggml_tensor * h_row = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1], 1,
                cur->nb[1], cur->nb[2], row * cur->nb[2]);
        const std::string name = std::string(stem) + "_row" + std::to_string((long long) row);
        cb(h_row, name.c_str(), il);
        ggml_build_forward_expand(gf, h_row);
    }
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_moe_ffn_stage41(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // Stage4.1/4.2/4.3 are deliberately narrow and target-shaped: preserve the
    // Stage4 row-serial attention/state barrier, then experiment only inside
    // the MoE/FFN segment.  Stage4.2 fixes the Stage4.1 fallback bug: when
    // routed projection batching is disabled, the fallback below really builds
    // the old MoE path one verifier row at a time instead of accidentally
    // feeding the generic MoE builder a multi-row tensor.  Stage4.3 keeps the
    // routed projection batching from the correctness-valid Stage4.2 path and
    // reopens the router/top-k/weights island only when the backend selects the
    // row-equivalent small-N fused route.
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(arch != LLM_ARCH_LLAMA4);

    auto build_old_moe = [&](ggml_tensor * x, const int64_t row_for_name) -> ggml_tensor * {
        ggml_tensor * out = build_moe_ffn(x,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
        cb(out, ("prefix42_ffn_moe_old_serial" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        return out;
    };

    const bool batch_routed_projections = qwen35moe_prefix_stage41_batch_routed_projections_enabled_for_rows(n_tokens);
    if (!batch_routed_projections || model.layers[il].ffn_gate_up_exps == nullptr) {
        ggml_tensor * moe_out_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_ffn_moe_old_serial_input_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * row_out = build_old_moe(cur_row, row);
            moe_out_all = moe_out_all ? ggml_concat(ctx0, moe_out_all, row_out, 1) : row_out;
        }
        cb(moe_out_all, "prefix42_ffn_moe_old_serial_all_batch_routed_proj0", il);
        trace_prefix_hidden_rows(moe_out_all, "prefix41_hidden_moe_out", il);
        return moe_out_all;
    }

    if (!qwen35moe_prefix_moe_row_shadow_active && qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_SHADOW_OLD_TRACE")) {
        ggml_tensor * old_shadow_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_ffn_moe_old_shadow_input_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * row_out = build_old_moe(cur_row, row);
            old_shadow_all = old_shadow_all ? ggml_concat(ctx0, old_shadow_all, row_out, 1) : row_out;
        }
        cb(old_shadow_all, "prefix42_ffn_moe_old_shadow_all", il);
        trace_prefix_hidden_rows(old_shadow_all, "prefix41_hidden_moe_old_shadow", il);
    }

    if (!qwen35moe_prefix_moe_row_shadow_active && qwen35moe_env_enabled("LLAMA_MTP_PREFIX_MOE_SHADOW_ROW_TRACE")) {
        ggml_tensor * row_shadow_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_ffn_moe_custom_row_shadow_input_row" + std::to_string((long long) row)).c_str(), il);
            qwen35moe_prefix_moe_row_shadow_active = true;
            qwen35moe_prefix_moe_row_shadow_source_row = row;
            ggml_tensor * row_out = build_moe_ffn_stage41(cur_row, il);
            qwen35moe_prefix_moe_row_shadow_source_row = -1;
            qwen35moe_prefix_moe_row_shadow_active = false;
            row_shadow_all = row_shadow_all ? ggml_concat(ctx0, row_shadow_all, row_out, 1) : row_out;
        }
        cb(row_shadow_all, "prefix42_ffn_moe_custom_row_shadow_all", il);
        trace_prefix_hidden_rows(row_shadow_all, "prefix41_hidden_moe_rowshadow_all", il);
    }

    const bool serial_router       = qwen35moe_prefix_stage41_serial_router_enabled();
    const bool serial_topk_weights = qwen35moe_prefix_stage41_serial_topk_weights_enabled();
    const bool serial_routed_glue  = qwen35moe_prefix_stage41_serial_routed_glue_enabled();
    const bool serial_expert_agg   = qwen35moe_prefix_stage41_serial_expert_agg_enabled();

    auto custom_trace_stem = [&](const char * suffix) -> std::string {
        if (qwen35moe_prefix_moe_row_shadow_active) {
            return std::string("prefix41_hidden_moe_rowshadow") +
                   std::to_string((long long) qwen35moe_prefix_moe_row_shadow_source_row) +
                   "_" + suffix;
        }
        return std::string("prefix41_hidden_moe_custom_") + suffix;
    };
    auto trace_custom_rows = [&](ggml_tensor * t, const char * suffix) {
        const std::string stem = custom_trace_stem(suffix);
        trace_prefix_hidden_rows(t, stem.c_str(), il);
    };
    auto trace_custom_token_rows = [&](ggml_tensor * t, const char * suffix) {
        const std::string stem = custom_trace_stem(suffix);
        trace_prefix_hidden_token_rows(t, stem.c_str(), il);
    };
    auto trace_custom_moe_out = [&](ggml_tensor * t) {
        if (qwen35moe_prefix_moe_row_shadow_active) {
            trace_custom_rows(t, "out");
        } else {
            trace_prefix_hidden_rows(t, "prefix41_hidden_moe_out", il);
        }
    };

    ggml_tensor * selected_experts = nullptr; // [n_expert_used, n_tokens]
    ggml_tensor * weights          = nullptr; // [1, n_expert_used, n_tokens]

    auto build_router_logits = [&](ggml_tensor * x, const int64_t row_for_name) -> ggml_tensor * {
        ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, x); // [n_expert, rows]
        cb(logits, ("prefix42_ffn_moe_router_logits" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        return logits;
    };

    auto build_topk_weights_from_logits = [&](ggml_tensor * logits, const int64_t row_for_name) -> std::pair<ggml_tensor *, ggml_tensor *> {
        const int64_t n_rows = logits->ne[1];
        GGML_ASSERT(n_rows > 0);

        ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        cb(probs, ("prefix42_ffn_moe_probs" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        ggml_tensor * selection_probs = probs;
        if (hparams.n_expert_groups > 1) {
            const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;
            ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_rows);
            ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2);
            group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores);
            group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3]));
            group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]);

            ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used);
            cb(expert_groups, ("prefix42_ffn_moe_group_topk" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

            selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups);
            selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups);
            selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_rows);
            cb(selection_probs, ("prefix42_ffn_moe_probs_masked" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        }

        ggml_tensor * selected = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
        cb(selected->src[0], ("prefix42_ffn_moe_argsort" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        cb(selected, ("prefix42_ffn_moe_topk" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_rows);
        ggml_tensor * w = ggml_get_rows(ctx0, probs, selected);
        cb(w, ("prefix42_ffn_moe_weights" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        w = ggml_reshape_2d(ctx0, w, n_expert_used, n_rows);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, w);
        cb(weights_sum, ("prefix42_ffn_moe_weights_sum" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, ("prefix42_ffn_moe_weights_sum_clamped" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        w = ggml_div(ctx0, w, weights_sum);
        cb(w, ("prefix42_ffn_moe_weights_norm" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);

        if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
            w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
            cb(w, ("prefix42_ffn_moe_weights_scaled" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        }

        w = ggml_reshape_3d(ctx0, w, 1, n_expert_used, n_rows);
        cb(w, ("prefix42_ffn_moe_weights_final" + qwen35moe_prefix_stage41_row_suffix(row_for_name)).c_str(), il);
        return std::make_pair(selected, w);
    };

    if (serial_router && serial_topk_weights) {
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_ffn_moe_router_input_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * logits_row = build_router_logits(cur_row, row);
            auto rw = build_topk_weights_from_logits(logits_row, row);
            selected_experts = selected_experts ? ggml_concat(ctx0, selected_experts, rw.first, 1) : rw.first;
            weights          = weights          ? ggml_concat(ctx0, weights,          rw.second, 2) : rw.second;
        }
        cb(selected_experts, "prefix42_ffn_moe_topk_serial_router_serial_topk_all", il);
        cb(weights,          "prefix42_ffn_moe_weights_serial_router_serial_topk_all", il);
    } else if (serial_router && !serial_topk_weights) {
        ggml_tensor * logits_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, n_embd, 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_ffn_moe_router_input_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * logits_row = build_router_logits(cur_row, row);
            logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
        }
        cb(logits_all, "prefix42_ffn_moe_router_logits_serial_all_topk_batched", il);
        auto rw = build_topk_weights_from_logits(logits_all, -2);
        selected_experts = rw.first;
        weights          = rw.second;
    } else if (!serial_router && serial_topk_weights) {
        ggml_tensor * logits_all = build_router_logits(cur, -1);
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * logits_row = ggml_view_2d(ctx0, logits_all, logits_all->ne[0], 1, logits_all->nb[1], row * logits_all->nb[1]);
            cb(logits_row, ("prefix42_ffn_moe_router_logits_batched_view_row" + std::to_string((long long) row)).c_str(), il);
            auto rw = build_topk_weights_from_logits(logits_row, row);
            selected_experts = selected_experts ? ggml_concat(ctx0, selected_experts, rw.first, 1) : rw.first;
            weights          = weights          ? ggml_concat(ctx0, weights,          rw.second, 2) : rw.second;
        }
        cb(selected_experts, "prefix42_ffn_moe_topk_batched_router_serial_topk_all", il);
        cb(weights,          "prefix42_ffn_moe_weights_batched_router_serial_topk_all", il);
    } else {
        const bool stage43_requested = qwen35moe_prefix_stage43_router_topk_fused_requested();
        const bool stage43_allowed   = stage43_requested && n_tokens <= 4;

        if (stage43_allowed) {
            ggml_tensor * logits_all = build_router_logits(cur, -4);
            cb(logits_all, "prefix43_router_topk_weights_fused_logits_all", il);

            const int router_topk_n_groups = hparams.n_expert_groups > 1 ? (int) hparams.n_expert_groups : 1;
            const int router_topk_n_group_used = hparams.n_expert_groups > 1 ? (int) hparams.n_group_used : 0;
            ggml_tensor * router_topk_payload = ggml_router_topk_weights(ctx0, logits_all, (int) n_expert_used,
                    router_topk_n_groups, router_topk_n_group_used, hparams.expert_weights_scale);
            cb(router_topk_payload, "prefix43_router_topk_weights_fused_payload", il);

            ggml_tensor * selected_f32 = ggml_view_2d(ctx0, router_topk_payload, n_expert_used, n_tokens,
                    router_topk_payload->nb[1], 0);
            cb(selected_f32, "prefix43_router_topk_weights_fused_selected_f32", il);
            selected_experts = ggml_cast(ctx0, selected_f32, GGML_TYPE_I32);
            cb(selected_experts, "prefix43_router_topk_weights_fused_selected_ids", il);

            ggml_tensor * weights_2d = ggml_view_2d(ctx0, router_topk_payload, n_expert_used, n_tokens,
                    router_topk_payload->nb[1], n_expert_used * router_topk_payload->nb[0]);
            cb(weights_2d, "prefix43_router_topk_weights_fused_weights_2d", il);
            weights_2d = ggml_cont(ctx0, weights_2d);
            cb(weights_2d, "prefix43_router_topk_weights_fused_weights_2d_cont", il);
            weights = ggml_reshape_3d(ctx0, weights_2d, 1, n_expert_used, n_tokens);
            cb(weights, "prefix43_router_topk_weights_fused_weights", il);
        } else {
            ggml_tensor * logits_all = build_router_logits(cur, stage43_requested ? -5 : -1);
            if (stage43_requested) {
                cb(logits_all, "prefix43_router_topk_weights_fused_not_allowed_logits_all", il);
            }
            auto rw = build_topk_weights_from_logits(logits_all, stage43_requested ? -5 : -1);
            selected_experts = rw.first;
            weights          = rw.second;
            if (stage43_requested) {
                cb(selected_experts, "prefix43_router_topk_weights_fused_not_allowed_selected", il);
                cb(weights,          "prefix43_router_topk_weights_fused_not_allowed_weights",  il);
            }
        }
    }

    // Match generic build_moe_ffn(): expand weights early for top-k MoE routing,
    // but do not expand selected_experts as a standalone graph output.  Expanding
    // selected_experts here perturbs the routed MoE graph and changes recurrent
    // state bytes even for a one-row custom shadow path.
    ggml_build_forward_expand(gf, weights);
    trace_custom_rows(selected_experts, "selected");
    trace_custom_token_rows(weights, "weights");

    ggml_tensor * routed_lanes_gate_up_shadow_source = nullptr;
    ggml_tensor * routed_lanes_routed_act_shadow_source = nullptr;
    ggml_tensor * routed_lanes_down_shadow_source = nullptr;
    ggml_tensor * routed_lanes_moe_out_shadow_source = nullptr;
    auto maybe_build_routed_lanes_diag = [&]() {
        const bool metadata_diag = qwen35moe_prefix_moe_routed_lanes_diag_enabled();
        const bool gather_scatter_diag = qwen35moe_prefix_moe_routed_lanes_gather_scatter_diag_enabled();
        const bool row_slot_map_diag = qwen35moe_prefix_moe_routed_lanes_row_slot_map_diag_enabled();
        const bool expert_bounds_diag = qwen35moe_prefix_moe_routed_lanes_expert_bounds_diag_enabled();
        const bool lora_empty = loras == nullptr || loras->empty();
        const ggml_type gate_up_exps_type = model.layers[il].ffn_gate_up_exps != nullptr ?
                model.layers[il].ffn_gate_up_exps->type : GGML_TYPE_COUNT;
        const bool f32_proj_shadow_requested = qwen35moe_prefix_moe_routed_lanes_f32_projection_shadow_diag_enabled();
        const bool f32_proj_shadow_diag = f32_proj_shadow_requested && model.layers[il].ffn_gate_up_exps != nullptr &&
                gate_up_exps_type == GGML_TYPE_F32;
        const bool quant_proj_shadow_diag = lora_empty && model.layers[il].ffn_gate_up_exps != nullptr &&
                qwen35moe_prefix_moe_routed_lanes_quant_projection_shadow_diag_enabled(gate_up_exps_type);
        const bool slot_proj_shadow_diag = qwen35moe_prefix_moe_routed_lanes_slot_projection_shadow_diag_enabled();
        const bool down_proj_shadow_diag = qwen35moe_prefix_moe_routed_lanes_down_projection_shadow_diag_enabled();
        const ggml_type down_exps_type = model.layers[il].ffn_down_exps != nullptr ?
                model.layers[il].ffn_down_exps->type : GGML_TYPE_COUNT;
        const bool quant_down_proj_shadow_requested = qwen35moe_prefix_moe_routed_lanes_quant_down_projection_shadow_diag_enabled();
        const bool quant_down_proj_shadow_diag = lora_empty && quant_down_proj_shadow_requested && model.layers[il].ffn_down_exps != nullptr &&
                (down_exps_type == GGML_TYPE_Q8_0 || down_exps_type == GGML_TYPE_IQ4_XS || down_exps_type == GGML_TYPE_IQ3_S);
        const bool down_agg_shadow_diag = qwen35moe_prefix_moe_routed_lanes_down_aggregation_shadow_diag_enabled();
        const bool full_branch_shadow_diag = qwen35moe_prefix_moe_routed_lanes_full_branch_shadow_diag_enabled();
        if ((!metadata_diag && !gather_scatter_diag && !row_slot_map_diag && !expert_bounds_diag && !f32_proj_shadow_diag && !quant_proj_shadow_diag && !slot_proj_shadow_diag && !down_proj_shadow_diag && !quant_down_proj_shadow_diag && !down_agg_shadow_diag && !full_branch_shadow_diag) || qwen35moe_prefix_moe_row_shadow_active) {
            return;
        }
        ggml_tensor * lanes = ggml_moe_routed_lanes(ctx0, selected_experts, weights, (int) n_expert);
        cb(lanes, "prefix42_ffn_moe_routed_lanes_meta", il);
        ggml_build_forward_expand(gf, lanes);

        ggml_tensor * expert_bounds = nullptr;
        if (expert_bounds_diag || f32_proj_shadow_diag || quant_proj_shadow_diag || quant_down_proj_shadow_diag) {
            expert_bounds = ggml_moe_routed_lanes_expert_bounds(ctx0, lanes);
            cb(expert_bounds, "prefix42_ffn_moe_routed_lanes_expert_bounds", il);
            if (expert_bounds_diag) {
                ggml_build_forward_expand(gf, expert_bounds);
            }
        }

        ggml_tensor * row_slot_map = nullptr;
        if (row_slot_map_diag || slot_proj_shadow_diag || down_proj_shadow_diag || quant_down_proj_shadow_diag || down_agg_shadow_diag || full_branch_shadow_diag) {
            row_slot_map = ggml_moe_routed_lanes_row_slot_map(ctx0, weights, lanes);
            cb(row_slot_map, "prefix42_ffn_moe_routed_lanes_row_slot_to_lane", il);
            ggml_build_forward_expand(gf, row_slot_map);
        }

        ggml_tensor * compact_x = nullptr;
        if (gather_scatter_diag || f32_proj_shadow_diag || quant_proj_shadow_diag || slot_proj_shadow_diag || full_branch_shadow_diag) {
            compact_x = ggml_moe_routed_lanes_gather(ctx0, cur, lanes);
            cb(compact_x, "prefix42_ffn_moe_routed_lanes_compact_x", il);
            ggml_build_forward_expand(gf, compact_x);
        }

        const int64_t n_lanes = n_expert_used*n_tokens;
        ggml_tensor * lane_experts = nullptr;
        if (f32_proj_shadow_diag || quant_proj_shadow_diag || slot_proj_shadow_diag || down_proj_shadow_diag || quant_down_proj_shadow_diag || down_agg_shadow_diag || full_branch_shadow_diag) {
            lane_experts = ggml_view_2d(ctx0, lanes, 1, n_lanes, lanes->nb[1], 0);
            cb(lane_experts, "prefix42_ffn_moe_routed_lanes_lane_experts", il);
        }

        if (f32_proj_shadow_diag && compact_x != nullptr && lane_experts != nullptr && expert_bounds != nullptr) {
            ggml_tensor * compact_x_3d = ggml_reshape_3d(ctx0, compact_x, n_embd, 1, n_lanes);
            cb(compact_x_3d, "prefix42_ffn_moe_routed_lanes_f32_proj_compact_x_3d", il);
            ggml_tensor * ref = build_lora_mm_id(model.layers[il].ffn_gate_up_exps, compact_x_3d, lane_experts);
            cb(ref, "prefix42_ffn_moe_routed_lanes_f32_proj_gate_up_ref", il);
            ggml_tensor * ref_2d = ggml_reshape_2d(ctx0, ref, ref->ne[0], n_lanes);
            cb(ref_2d, "prefix42_ffn_moe_routed_lanes_f32_proj_gate_up_ref_2d", il);
            ggml_tensor * shadow = ggml_moe_routed_lanes_projection(ctx0, model.layers[il].ffn_gate_up_exps, compact_x, lanes, expert_bounds);
            cb(shadow, "prefix42_ffn_moe_routed_lanes_f32_proj_gate_up_shadow", il);
            ggml_tensor * delta = ggml_sub(ctx0, shadow, ref_2d);
            cb(delta, "prefix42_ffn_moe_routed_lanes_f32_proj_gate_up_shadow_delta", il);
            ggml_build_forward_expand(gf, delta);
        }

        if (quant_proj_shadow_diag && compact_x != nullptr && lane_experts != nullptr && expert_bounds != nullptr) {
            ggml_tensor * compact_x_3d = ggml_reshape_3d(ctx0, compact_x, n_embd, 1, n_lanes);
            cb(compact_x_3d, "prefix42_ffn_moe_routed_lanes_quant_proj_compact_x_3d", il);
            ggml_tensor * ref = build_lora_mm_id(model.layers[il].ffn_gate_up_exps, compact_x_3d, lane_experts);
            cb(ref, "prefix42_ffn_moe_routed_lanes_quant_proj_gate_up_ref", il);
            ggml_tensor * ref_2d = ggml_reshape_2d(ctx0, ref, ref->ne[0], n_lanes);
            cb(ref_2d, "prefix42_ffn_moe_routed_lanes_quant_proj_gate_up_ref_2d", il);
            ggml_tensor * shadow = ggml_moe_routed_lanes_projection(ctx0, model.layers[il].ffn_gate_up_exps, compact_x, lanes, expert_bounds);
            cb(shadow, "prefix42_ffn_moe_routed_lanes_quant_proj_gate_up_shadow", il);
            ggml_tensor * delta = ggml_sub(ctx0, shadow, ref_2d);
            cb(delta, "prefix42_ffn_moe_routed_lanes_quant_proj_gate_up_shadow_delta", il);
            ggml_build_forward_expand(gf, delta);
        }

        if (slot_proj_shadow_diag && routed_lanes_gate_up_shadow_source != nullptr && row_slot_map != nullptr && compact_x != nullptr && lane_experts != nullptr) {
            ggml_tensor * compact_x_3d = ggml_reshape_3d(ctx0, compact_x, n_embd, 1, n_lanes);
            cb(compact_x_3d, "prefix42_ffn_moe_routed_lanes_compact_x_3d", il);
            ggml_tensor * compact_gate_up = build_lora_mm_id(model.layers[il].ffn_gate_up_exps, compact_x_3d, lane_experts);
            cb(compact_gate_up, "prefix42_ffn_moe_routed_lanes_gate_up_compact_shadow", il);
            ggml_tensor * compact_gate_up_2d = ggml_reshape_2d(ctx0, compact_gate_up, compact_gate_up->ne[0], n_lanes);
            cb(compact_gate_up_2d, "prefix42_ffn_moe_routed_lanes_gate_up_compact_shadow_2d", il);
            ggml_tensor * slot_gate_up = ggml_moe_routed_lanes_unpack_slots(ctx0, compact_gate_up_2d, row_slot_map);
            cb(slot_gate_up, "prefix42_ffn_moe_routed_lanes_gate_up_slot_shadow", il);
            ggml_build_forward_expand(gf, slot_gate_up);
        }

        if (full_branch_shadow_diag && compact_x != nullptr && row_slot_map != nullptr && lane_experts != nullptr) {
            ggml_tensor * compact_x_3d = ggml_reshape_3d(ctx0, compact_x, n_embd, 1, n_lanes);
            cb(compact_x_3d, "prefix42_ffn_moe_routed_lanes_full_compact_x_3d", il);
            ggml_tensor * compact_gate_up = build_lora_mm_id(model.layers[il].ffn_gate_up_exps, compact_x_3d, lane_experts);
            cb(compact_gate_up, "prefix42_ffn_moe_routed_lanes_full_gate_up_compact_shadow", il);

            if (model.layers[il].ffn_up_exps_s) {
                ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_up_exps_s, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_lanes, 1);
                s = ggml_get_rows(ctx0, s, lane_experts);
                cb(s, "prefix42_ffn_moe_routed_lanes_full_gate_up_compact_scale", il);
                compact_gate_up = ggml_mul(ctx0, compact_gate_up, s);
                cb(compact_gate_up, "prefix42_ffn_moe_routed_lanes_full_gate_up_compact_scaled_shadow", il);
            }

            const int64_t n_ff_compact = compact_gate_up->ne[0] / 2;
            ggml_tensor * compact_gate = ggml_view_3d(ctx0, compact_gate_up, n_ff_compact, compact_gate_up->ne[1], compact_gate_up->ne[2],
                    compact_gate_up->nb[1], compact_gate_up->nb[2], 0);
            cb(compact_gate, "prefix42_ffn_moe_routed_lanes_full_gate_compact_shadow", il);
            ggml_tensor * compact_up = ggml_view_3d(ctx0, compact_gate_up, n_ff_compact, compact_gate_up->ne[1], compact_gate_up->ne[2],
                    compact_gate_up->nb[1], compact_gate_up->nb[2], n_ff_compact * compact_gate_up->nb[0]);
            cb(compact_up, "prefix42_ffn_moe_routed_lanes_full_up_compact_shadow", il);
            ggml_tensor * compact_routed_act = ggml_swiglu_split(ctx0, compact_gate, compact_up);
            cb(compact_routed_act, "prefix42_ffn_moe_routed_lanes_full_swiglu_compact_shadow", il);

            ggml_tensor * compact_down = build_lora_mm_id(model.layers[il].ffn_down_exps, compact_routed_act, lane_experts);
            cb(compact_down, "prefix42_ffn_moe_routed_lanes_full_down_compact_shadow", il);

            if (model.layers[il].ffn_down_exps_s) {
                ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_down_exps_s, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_lanes, 1);
                s = ggml_get_rows(ctx0, s, lane_experts);
                cb(s, "prefix42_ffn_moe_routed_lanes_full_down_compact_scale", il);
                compact_down = ggml_mul(ctx0, compact_down, s);
                cb(compact_down, "prefix42_ffn_moe_routed_lanes_full_down_compact_scaled_shadow", il);
            }

            ggml_tensor * compact_down_2d = ggml_reshape_2d(ctx0, compact_down, compact_down->ne[0], n_lanes);
            cb(compact_down_2d, "prefix42_ffn_moe_routed_lanes_full_down_compact_shadow_2d", il);
            ggml_tensor * compact_moe_out = ggml_moe_routed_lanes_scatter_reduce(ctx0, compact_down_2d, weights, lanes);
            cb(compact_moe_out, "prefix42_ffn_moe_routed_lanes_full_moe_out_shadow", il);
            ggml_build_forward_expand(gf, compact_moe_out);
            if (routed_lanes_moe_out_shadow_source != nullptr) {
                ggml_tensor * delta = ggml_sub(ctx0, compact_moe_out, routed_lanes_moe_out_shadow_source);
                cb(delta, "prefix42_ffn_moe_routed_lanes_full_moe_out_shadow_delta", il);
                ggml_build_forward_expand(gf, delta);
            }
        }

        if ((down_proj_shadow_diag || quant_down_proj_shadow_diag || down_agg_shadow_diag) && routed_lanes_routed_act_shadow_source != nullptr && routed_lanes_down_shadow_source != nullptr && row_slot_map != nullptr && lane_experts != nullptr) {
            ggml_tensor * compact_routed_act = ggml_moe_routed_lanes_pack_slots(ctx0, routed_lanes_routed_act_shadow_source, row_slot_map);
            cb(compact_routed_act, "prefix42_ffn_moe_routed_lanes_routed_act_compact", il);
            ggml_tensor * compact_routed_act_3d = ggml_reshape_3d(ctx0, compact_routed_act, compact_routed_act->ne[0], 1, n_lanes);
            cb(compact_routed_act_3d, "prefix42_ffn_moe_routed_lanes_routed_act_compact_3d", il);
            ggml_tensor * compact_down = build_lora_mm_id(model.layers[il].ffn_down_exps, compact_routed_act_3d, lane_experts);
            cb(compact_down, "prefix42_ffn_moe_routed_lanes_down_compact_shadow", il);

            if (quant_down_proj_shadow_diag && expert_bounds != nullptr) {
                ggml_tensor * compact_down_ref_2d = ggml_reshape_2d(ctx0, compact_down, compact_down->ne[0], n_lanes);
                cb(compact_down_ref_2d, "prefix58_ffn_moe_routed_lanes_quant_down_proj_ref_2d", il);
                ggml_tensor * compact_down_shadow = ggml_moe_routed_lanes_projection(ctx0, model.layers[il].ffn_down_exps, compact_routed_act, lanes, expert_bounds);
                cb(compact_down_shadow, "prefix58_ffn_moe_routed_lanes_quant_down_proj_shadow", il);
                ggml_tensor * delta = ggml_sub(ctx0, compact_down_shadow, compact_down_ref_2d);
                cb(delta, "prefix58_ffn_moe_routed_lanes_quant_down_proj_shadow_delta", il);
                ggml_build_forward_expand(gf, delta);
            }

            // Stage58 quant down projection shadow compares only the raw ffn_down_exps matmul.
            // Keep ffn_down_exps_s after the raw projection comparison so both sides of
            // prefix58_*_delta are unscaled; existing down slot/aggregation shadows below still
            // apply the per-expert scale before unpack/scatter.
            if (model.layers[il].ffn_down_exps_s) {
                ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_down_exps_s, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_lanes, 1);
                s = ggml_get_rows(ctx0, s, lane_experts);
                cb(s, "prefix42_ffn_moe_routed_lanes_down_compact_scale", il);
                compact_down = ggml_mul(ctx0, compact_down, s);
                cb(compact_down, "prefix42_ffn_moe_routed_lanes_down_compact_scaled_shadow", il);
            }

            ggml_tensor * compact_down_2d = ggml_reshape_2d(ctx0, compact_down, compact_down->ne[0], n_lanes);
            cb(compact_down_2d, "prefix42_ffn_moe_routed_lanes_down_compact_shadow_2d", il);

            if (down_proj_shadow_diag) {
                ggml_tensor * slot_down = ggml_moe_routed_lanes_unpack_slots(ctx0, compact_down_2d, row_slot_map);
                cb(slot_down, "prefix42_ffn_moe_routed_lanes_down_slot_shadow", il);
                ggml_build_forward_expand(gf, slot_down);
            }

            if (down_agg_shadow_diag) {
                ggml_tensor * compact_moe_out = ggml_moe_routed_lanes_scatter_reduce(ctx0, compact_down_2d, weights, lanes);
                cb(compact_moe_out, "prefix42_ffn_moe_routed_lanes_moe_out_shadow", il);
                ggml_build_forward_expand(gf, compact_moe_out);
                if (routed_lanes_moe_out_shadow_source != nullptr) {
                    ggml_tensor * delta = ggml_sub(ctx0, compact_moe_out, routed_lanes_moe_out_shadow_source);
                    cb(delta, "prefix42_ffn_moe_routed_lanes_moe_out_shadow_delta", il);
                    ggml_build_forward_expand(gf, delta);
                }
            }
        }

        if (!gather_scatter_diag) {
            return;
        }
        ggml_tensor * roundtrip = ggml_moe_routed_lanes_scatter_reduce(ctx0, compact_x, weights, lanes);
        cb(roundtrip, "prefix42_ffn_moe_routed_lanes_weighted_roundtrip", il);
        ggml_build_forward_expand(gf, roundtrip);
    };

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    ggml_tensor * gate_up = build_lora_mm_id(model.layers[il].ffn_gate_up_exps, cur_3d, selected_experts);
    routed_lanes_gate_up_shadow_source = gate_up;
    cb(gate_up, "prefix42_ffn_moe_gate_up_batched_proj", il);
    trace_custom_token_rows(gate_up, "gate_up_proj");

    if (model.layers[il].ffn_up_exps_s) {
        if (serial_routed_glue) {
            ggml_tensor * gate_up_scaled_all = nullptr;
            for (int64_t row = 0; row < n_tokens; ++row) {
                ggml_tensor * gate_up_row = ggml_view_3d(ctx0, gate_up, gate_up->ne[0], gate_up->ne[1], 1,
                        gate_up->nb[1], gate_up->nb[2], row * gate_up->nb[2]);
                ggml_tensor * selected_row = ggml_view_2d(ctx0, selected_experts, selected_experts->ne[0], 1,
                        selected_experts->nb[1], row * selected_experts->nb[1]);
                ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_up_exps_s, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx0, s, 1, n_expert, 1, 1);
                s = ggml_get_rows(ctx0, s, selected_row);
                gate_up_row = ggml_mul(ctx0, gate_up_row, s);
                cb(gate_up_row, ("prefix42_ffn_moe_gate_up_scaled_row" + std::to_string((long long) row)).c_str(), il);
                gate_up_scaled_all = gate_up_scaled_all ? ggml_concat(ctx0, gate_up_scaled_all, gate_up_row, 2) : gate_up_row;
            }
            gate_up = gate_up_scaled_all;
            cb(gate_up, "prefix42_ffn_moe_gate_up_scaled_serial_all", il);
        } else {
            ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts);
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb(gate_up, "prefix42_ffn_moe_gate_up_scaled", il);
        }
    }

    trace_custom_token_rows(gate_up, "gate_up_scaled");

    const int64_t n_ff = gate_up->ne[0] / 2;
    ggml_tensor * routed_act = nullptr;

    if (serial_routed_glue) {
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * gate_up_row = ggml_view_3d(ctx0, gate_up, gate_up->ne[0], gate_up->ne[1], 1,
                    gate_up->nb[1], gate_up->nb[2], row * gate_up->nb[2]);
            ggml_tensor * gate_row = ggml_view_3d(ctx0, gate_up_row, n_ff, gate_up_row->ne[1], 1,
                    gate_up_row->nb[1], gate_up_row->nb[2], 0);
            cb(gate_row, ("prefix42_ffn_moe_gate_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * up_row = ggml_view_3d(ctx0, gate_up_row, n_ff, gate_up_row->ne[1], 1,
                    gate_up_row->nb[1], gate_up_row->nb[2], n_ff * gate_up_row->nb[0]);
            cb(up_row, ("prefix42_ffn_moe_up_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * act_row = ggml_swiglu_split(ctx0, gate_row, up_row);
            cb(act_row, ("prefix42_ffn_moe_swiglu_row" + std::to_string((long long) row)).c_str(), il);
            routed_act = routed_act ? ggml_concat(ctx0, routed_act, act_row, 2) : act_row;
        }
        cb(routed_act, "prefix42_ffn_moe_swiglu_serial_all", il);
    } else {
        ggml_tensor * gate = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(gate, "prefix42_ffn_moe_gate", il);
        ggml_tensor * up = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "prefix42_ffn_moe_up", il);
        routed_act = ggml_swiglu_split(ctx0, gate, up);
        cb(routed_act, "prefix42_ffn_moe_swiglu", il);
    }
    trace_custom_token_rows(routed_act, "routed_act");
    routed_lanes_routed_act_shadow_source = routed_act;

    ggml_tensor * experts = build_lora_mm_id(model.layers[il].ffn_down_exps, routed_act, selected_experts);
    routed_lanes_down_shadow_source = experts;
    cb(experts, "prefix42_ffn_moe_down_batched_proj", il);
    trace_custom_token_rows(experts, "down_proj");

    if (model.layers[il].ffn_down_exps_s) {
        if (serial_routed_glue || serial_expert_agg) {
            ggml_tensor * experts_scaled_all = nullptr;
            for (int64_t row = 0; row < n_tokens; ++row) {
                ggml_tensor * experts_row = ggml_view_3d(ctx0, experts, experts->ne[0], experts->ne[1], 1,
                        experts->nb[1], experts->nb[2], row * experts->nb[2]);
                ggml_tensor * selected_row = ggml_view_2d(ctx0, selected_experts, selected_experts->ne[0], 1,
                        selected_experts->nb[1], row * selected_experts->nb[1]);
                ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_down_exps_s, 1, n_expert, 1);
                s = ggml_repeat_4d(ctx0, s, 1, n_expert, 1, 1);
                s = ggml_get_rows(ctx0, s, selected_row);
                experts_row = ggml_mul(ctx0, experts_row, s);
                cb(experts_row, ("prefix42_ffn_moe_down_scaled_row" + std::to_string((long long) row)).c_str(), il);
                experts_scaled_all = experts_scaled_all ? ggml_concat(ctx0, experts_scaled_all, experts_row, 2) : experts_row;
            }
            experts = experts_scaled_all;
            cb(experts, "prefix42_ffn_moe_down_scaled_serial_all", il);
        } else {
            ggml_tensor * s = ggml_reshape_3d(ctx0, model.layers[il].ffn_down_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts);
            experts = ggml_mul(ctx0, experts, s);
            cb(experts, "prefix42_ffn_moe_down_scaled", il);
        }
    }

    trace_custom_token_rows(experts, "down_scaled");

    if (serial_expert_agg) {
        ggml_tensor * moe_out_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * experts_row = ggml_view_3d(ctx0, experts, experts->ne[0], experts->ne[1], 1,
                    experts->nb[1], experts->nb[2], row * experts->nb[2]);
            ggml_tensor * weights_row = ggml_view_3d(ctx0, weights, 1, n_expert_used, 1,
                    weights->nb[1], weights->nb[2], row * weights->nb[2]);
            experts_row = ggml_mul(ctx0, experts_row, weights_row);
            cb(experts_row, ("prefix42_ffn_moe_weighted_row" + std::to_string((long long) row)).c_str(), il);
            ggml_build_forward_expand(gf, experts_row);

            ggml_tensor * row_out = nullptr;
            for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
                ggml_tensor * expert_i = ggml_view_2d(ctx0, experts_row, n_embd, 1, experts_row->nb[2], i * experts_row->nb[1]);
                ggml_build_forward_expand(gf, expert_i);
                row_out = row_out ? ggml_add(ctx0, row_out, expert_i) : expert_i;
                ggml_build_forward_expand(gf, row_out);
            }
            if (hparams.n_expert_used == 1) {
                row_out = ggml_cont(ctx0, row_out);
            }
            cb(row_out, ("prefix42_ffn_moe_out_row" + std::to_string((long long) row)).c_str(), il);
            moe_out_all = moe_out_all ? ggml_concat(ctx0, moe_out_all, row_out, 1) : row_out;
        }
        cb(moe_out_all, "prefix42_ffn_moe_out_serial_agg", il);
        trace_custom_moe_out(moe_out_all);
        routed_lanes_moe_out_shadow_source = moe_out_all;
        maybe_build_routed_lanes_diag();
        return moe_out_all;
    }

    experts = ggml_mul(ctx0, experts, weights);
    cb(experts, "prefix42_ffn_moe_weighted", il);
    ggml_build_forward_expand(gf, experts);
    trace_custom_token_rows(experts, "weighted");

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    ggml_tensor * moe_out = cur_experts[0];
    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "prefix42_ffn_moe_out", il);
    trace_custom_moe_out(moe_out);
    routed_lanes_moe_out_shadow_source = moe_out;
    maybe_build_routed_lanes_diag();
    return moe_out;
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_layer_ffn_stage41(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    const int64_t n_tokens = cur->ne[1];
    GGML_ASSERT(n_tokens > 0);

    if (!qwen35moe_prefix_stage41_batch_routed_projections_enabled()) {
        ggml_tensor * out_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix42_full_serial_ffn_input_batch_routed_proj0" + qwen35moe_prefix_stage41_row_suffix(row)).c_str(), il);
            ggml_tensor * out_row = build_layer_ffn(cur_row, il);
            cb(out_row, ("prefix42_full_serial_ffn_out_batch_routed_proj0" + qwen35moe_prefix_stage41_row_suffix(row)).c_str(), il);
            out_all = out_all ? ggml_concat(ctx0, out_all, out_row, 1) : out_row;
        }
        cb(out_all, "prefix42_full_serial_ffn_out_all_batch_routed_proj0", il);
        return out_all;
    }

    ggml_tensor * moe_out = build_moe_ffn_stage41(cur, il);
    cb(moe_out, "prefix42_ffn_moe_out_layer", il);

    if (model.layers[il].ffn_up_shexp == nullptr) {
        return moe_out;
    }

    ggml_tensor * ffn_shexp = nullptr;

    if (qwen35moe_prefix_stage41_serial_shared_ffn_enabled()) {
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], row * cur->nb[1]);
            cb(cur_row, ("prefix41_shared_ffn_input_row" + std::to_string((long long) row)).c_str(), il);
            ggml_tensor * row_ffn = build_ffn(cur_row,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(row_ffn, ("prefix41_ffn_shexp_row" + std::to_string((long long) row)).c_str(), il);
            ffn_shexp = ffn_shexp ? ggml_concat(ctx0, ffn_shexp, row_ffn, 1) : row_ffn;
        }
        cb(ffn_shexp, "prefix41_ffn_shexp_serial_all", il);
    } else {
        ffn_shexp = build_ffn(cur,
            model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
            model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
            model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "prefix41_ffn_shexp", il);
    }

    ggml_tensor * shared_gate = nullptr;
    if (qwen35moe_prefix_stage41_serial_shared_gate_enabled()) {
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * cur_row = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], row * cur->nb[1]);
            ggml_tensor * gate_row = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur_row);
            cb(gate_row, ("prefix41_shared_expert_gate_row" + std::to_string((long long) row)).c_str(), il);
            gate_row = ggml_sigmoid(ctx0, gate_row);
            cb(gate_row, ("prefix41_shared_expert_gate_sigmoid_row" + std::to_string((long long) row)).c_str(), il);
            shared_gate = shared_gate ? ggml_concat(ctx0, shared_gate, gate_row, 1) : gate_row;
        }
        cb(shared_gate, "prefix41_shared_expert_gate_serial_all", il);
    } else {
        shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "prefix41_shared_expert_gate", il);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "prefix41_shared_expert_gate_sigmoid", il);
    }

    ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
    cb(ffn_shexp, "prefix41_ffn_shexp_gated", il);
    trace_prefix_hidden_rows(ffn_shexp, "prefix41_hidden_shexp", il);

    if (qwen35moe_prefix_stage41_serial_expert_agg_enabled()) {
        ggml_tensor * out_all = nullptr;
        for (int64_t row = 0; row < n_tokens; ++row) {
            ggml_tensor * moe_row = ggml_view_2d(ctx0, moe_out, moe_out->ne[0], 1, moe_out->nb[1], row * moe_out->nb[1]);
            ggml_tensor * sh_row  = ggml_view_2d(ctx0, ffn_shexp, ffn_shexp->ne[0], 1, ffn_shexp->nb[1], row * ffn_shexp->nb[1]);
            ggml_tensor * out_row = ggml_add(ctx0, moe_row, sh_row);
            cb(out_row, ("prefix41_ffn_out_row" + std::to_string((long long) row)).c_str(), il);
            out_all = out_all ? ggml_concat(ctx0, out_all, out_row, 1) : out_row;
        }
        cb(out_all, "prefix41_ffn_out_serial_all", il);
        trace_prefix_hidden_rows(out_all, "prefix41_hidden_ffn_out", il);
        return out_all;
    }

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "prefix41_ffn_out", il);
    trace_prefix_hidden_rows(cur, "prefix41_hidden_ffn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph_prefix_verify::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "prefix_ffn_moe_out", il);

    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "prefix_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "prefix_shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "prefix_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "prefix_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "prefix_ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35MOE MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "MTP block missing ffn_gate_inp");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    GGML_ASSERT(hparams.n_embd_inp() == hparams.n_embd && "MTP embd/h input dimensions must match until llm_graph_input_embd_h tracks both dimensions");
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    ggml_tensor * attn_pregate = cur;
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    ggml_tensor * attn_gated = cur;
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    ggml_tensor * attn_out = cur;
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    if (layer.ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "mtp_shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "mtp_ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    } else {
        cur = moe_out;
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    if (const char * bypass = getenv("LLAMA_MTP_BYPASS_CORE")) {
        if (strcmp(bypass, "h") == 0) {
            cur = h_embd;
        } else if (strcmp(bypass, "h_norm") == 0) {
            cur = h_norm;
        } else if (strcmp(bypass, "eh_proj") == 0) {
            cur = inpSA;
        } else if (strcmp(bypass, "attn_pregate") == 0) {
            cur = attn_pregate;
        } else if (strcmp(bypass, "attn_gated") == 0) {
            cur = attn_gated;
        } else if (strcmp(bypass, "attn_out") == 0) {
            cur = attn_out;
        } else if (strcmp(bypass, "attn_residual") == 0) {
            cur = ffn_residual;
        }
        fprintf(stderr, "MTP_BYPASS(qwen35moe): stage=%s cur=[%lld,%lld]\n",
            bypass, (long long) cur->ne[0], (long long) cur->ne[1]);
    }

    cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
    // Pre-norm hidden state: used by the AR draft loop to seed the next MTP step.
    cb(cur, "h_pre_norm", -1);
    res->t_h_pre_norm = cur;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    // LLAMA_MTP_HEAD_PROBE: dump output tensor dimensions
    if (getenv("LLAMA_MTP_HEAD_PROBE")) {
        fprintf(stderr, "MTP_HEAD_PROBE output: head_w=[%lld,%lld] cur_before=[%lld,%lld] cur_after=[%lld,%lld] n_vocab=%d\n",
            (long long)head_w->ne[0], (long long)head_w->ne[1],
            (long long)cur->ne[0], (long long)cur->ne[1],
            0LL, 0LL,
            llama_vocab_n_tokens(llama_model_get_vocab(&model)));
    }
    cur = build_lora_mm(head_w, cur, head_s);
    if (getenv("LLAMA_MTP_HEAD_PROBE")) {
        fprintf(stderr, "MTP_HEAD_PROBE after_mm: cur=[%lld,%lld]\n",
            (long long)cur->ne[0], (long long)cur->ne[1]);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
