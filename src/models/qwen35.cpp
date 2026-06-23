#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

#include <cstdlib>

namespace {

bool qwen35_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0' && atoi(env) != 0;
}

int qwen35_env_i32(const char * name, int def) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env ? (int) v : def;
}

bool qwen35_prefix_state_candidate_trace_enabled(int il) {
    if (!qwen35_env_enabled("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE")) {
        return false;
    }
    const int layer_filter = qwen35_env_i32("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35_prefix_state_reconstruct_copy_enabled(int il) {
    if (!qwen35_env_enabled("LLAMA_MTP_PREFIX_STATE_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35_env_i32("LLAMA_MTP_PREFIX_STATE_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35_prefix_state_source_trace_enabled(int il) {
    if (!qwen35_env_enabled("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE")) {
        return false;
    }
    const int layer_filter = qwen35_env_i32("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35_prefix_r_direct_reconstruct_copy_enabled(int il) {
    if (!qwen35_env_enabled("LLAMA_MTP_PREFIX_R_DIRECT_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35_env_i32("LLAMA_MTP_PREFIX_R_DIRECT_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35_prefix_s_state_only_reconstruct_copy_enabled(int il) {
    if (!qwen35_env_enabled("LLAMA_MTP_PREFIX_S_STATE_ONLY_RECONSTRUCT_COPY")) {
        return false;
    }
    const int layer_filter = qwen35_env_i32("LLAMA_MTP_PREFIX_S_STATE_ONLY_RECONSTRUCT_COPY_LAYER", 0);
    return layer_filter < 0 || layer_filter == il;
}

bool qwen35_prefix_accepted_row_only_commit_enabled() {
    return qwen35_env_enabled("LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT");
}

bool qwen35_prefix_qblock_fused_verify_enabled() {
    return qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY");
}

bool qwen35_prefix_qblock_fused_verify_layer_enabled(int il, int n_layers) {
    if (!qwen35_prefix_qblock_fused_verify_enabled()) {
        return false;
    }
    if (getenv("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MIN") || getenv("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MAX")) {
        GGML_ABORT("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MIN/MAX are deprecated unsafe aliases; use LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MIN/MAX");
    }
    const int layer_min = qwen35_env_i32("LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MIN", 0);
    const int layer_max = qwen35_env_i32("LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MAX", n_layers - 1);
    return il >= layer_min && il <= layer_max;
}

void qwen35_lm_head_top1_apply_eog_mask(const llama_model & model, ggml_tensor * top1) {
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

    if (qwen35_env_enabled("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
        fprintf(stderr, "MTP_TARGET_TOP1_EOG_MASK: tensor=%s n_ban=%d", top1->name, (int) n_ban);
        for (int32_t i = 0; i < n_ban; ++i) {
            fprintf(stderr, " %d", top1->op_params[1 + i]);
        }
        fprintf(stderr, "\n");
    }
}

ggml_tensor * qwen35_mul_mat_aux(
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

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
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
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
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

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, 0);

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

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY ||
        params.gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_COMMIT) {
        return std::make_unique<graph_prefix_verify>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
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

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

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
        qwen35_lm_head_top1_apply_eog_mask(model, sampled);
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
        qwen35_lm_head_top1_apply_eog_mask(model, sampled);
        res->t_mtp_target_top1_fused_all = sampled;
        ggml_build_forward_expand(gf, sampled);
    }

    const char * target_top1_shadow = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW");
    if (target_top1_shadow && atoi(target_top1_shadow) != 0 &&
            lm_head_top1_direct_supported) {
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

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
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

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
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

    // Apply MRoPE
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

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
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

    const uint32_t mem_size  = mctx_cur->get_size();
    const bool keep_intermediates   = (cparams.n_rs_seq > 0)
                            && (n_seq_tokens > 1)
                            && ((uint32_t) n_seq_tokens <= 1 + cparams.n_rs_seq);

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

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

llama_model_qwen35::graph_prefix_verify::graph_prefix_verify(const llama_model & model, const llm_graph_params & params) :
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

    const bool skip_verify_snapshots = !commit_only && qwen35_prefix_accepted_row_only_commit_enabled();
    const bool batch_output_head = !commit_only && params.mtp_prefix_batch_output_head;
    const char * target_top1_active_env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE");
    const char * target_top1_active_raw_env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE");
    const char * target_top1_active_logits_env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_LOGITS");
    const bool lm_head_top1_direct_supported =
            (model.output->type == GGML_TYPE_Q6_K || model.output->type == GGML_TYPE_Q8_0) &&
            (loras == nullptr || loras->empty());
    const bool target_top1_active_flag = target_top1_active_env ? atoi(target_top1_active_env) != 0 : true;
    const bool target_top1_raw_flag = target_top1_active_raw_env ? atoi(target_top1_active_raw_env) != 0 : true;
    const bool target_top1_logits_flag = target_top1_active_logits_env ? atoi(target_top1_active_logits_env) != 0 : true;
    const bool target_top1_active = !commit_only &&
            target_top1_active_flag && target_top1_raw_flag &&
            lm_head_top1_direct_supported;
    const bool target_top1_active_from_logits = target_top1_active && target_top1_logits_flag;
    // Accepted-row-only mode cannot know the accepted length until after
    // sampling.  Materialize a small suffix of verifier rows into rollback
    // slots during verifier decode; covered rollback values can commit without
    // a checkpoint restore or second prefix pass, and uncovered partial accepts
    // fall back to the commit graph.
    const int64_t snapshot_slot_limit    = skip_verify_snapshots ? params.mtp_prefix_accepted_commit_verify_slots : 0;
    const int64_t snapshot_slot_override = commit_only ? params.mtp_prefix_commit_slot : -1;

    ggml_tensor * h_pre_norm_all = nullptr;
    ggml_tensor * embd_all       = nullptr;
    ggml_tensor * logits_all     = nullptr;
    const bool qblock_prefix_fused_verify = !commit_only && ubatch.n_tokens > 1 && qwen35_prefix_qblock_fused_verify_enabled();
    const int qblock_prefix_fused_tail_layer = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_TAIL_LAYER", -1);
    const bool qblock_prefix_fused_tail = qblock_prefix_fused_verify && qblock_prefix_fused_tail_layer >= 0;
    if (getenv("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MAJOR_UNSAFE")) {
        GGML_ABORT("LLAMA_MTP_QBLOCK_PREFIX_FUSED_VERIFY_LAYER_MAJOR_UNSAFE is a deprecated unsafe alias; use LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MAJOR_UNSAFE");
    }
    const bool qblock_prefix_fused_layer_major = qblock_prefix_fused_verify && !qblock_prefix_fused_tail &&
            qwen35_env_enabled("LLAMA_MTP_QBLOCK_FUSED_VERIFY_LAYER_MAJOR_UNSAFE");
    const bool qblock_prefix_layer64_scratch_probe = !commit_only && ubatch.n_tokens > 1 &&
            qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_PROBE");

    if (qblock_prefix_layer64_scratch_probe) {
        const int scratch_src_layer = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_SRC_LAYER", n_transformer_layers - 1);
        const int scratch_label_layer = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_LABEL_LAYER", n_transformer_layers);
        GGML_ASSERT(scratch_src_layer == n_transformer_layers - 1);
        GGML_ASSERT(!hparams.is_recurrent(scratch_src_layer));

        auto trace_rows = [&](ggml_tensor * t, const char * stem, int il) {
            if (!qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_TRACE_ROWS")) {
                return;
            }
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * row_view = ggml_view_2d(ctx0, t, n_embd, 1, t->nb[1], row * t->nb[1]);
                const std::string name = std::string("prefix_roweq_hidden_") + stem + "_row" + std::to_string(row);
                cb(row_view, name.c_str(), il);
                ggml_build_forward_expand(gf, row_view);
            }
        };

        ggml_tensor * tail_input_all = nullptr;
        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], row * inpL->nb[1]);
            cb(cur, "prefix_layer64_scratch_input_row", -1);

            for (int il = 0; il < scratch_src_layer; ++il) {
                ggml_tensor * inpSA = cur;

                cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "prefix_layer64_scratch_pre_attn_norm", il);

                ggml_build_forward_expand(gf, cur);

                if (hparams.is_recurrent(il)) {
                    cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                            commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                } else {
                    cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                }

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "prefix_layer64_scratch_pre_attn_residual", il);

                ggml_tensor * ffn_residual = cur;

                ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_post_norm, "prefix_layer64_scratch_pre_attn_post_norm", il);

                cur = build_layer_ffn(attn_post_norm, il);
                cb(cur, "prefix_layer64_scratch_pre_ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_residual);
                cb(cur, "prefix_layer64_scratch_pre_post_ffn", il);

                cur = build_cvec(cur, il);
                cb(cur, "prefix_layer64_scratch_pre_l_out", il);
            }

            tail_input_all = tail_input_all ? ggml_concat(ctx0, tail_input_all, cur, 1) : cur;
        }
        cb(tail_input_all, "prefix_layer64_scratch_tail_input_all", scratch_src_layer);

        const bool qblock_prefix_layer64_inline_row_compare =
                qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_INLINE_ROW_COMPARE");
        ggml_tensor * serial_attn_all = nullptr;
        ggml_tensor * serial_q_all    = nullptr;
        ggml_tensor * serial_gate_all = nullptr;
        ggml_tensor * inline_scratch_attn_all = nullptr;
        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, tail_input_all, n_embd, 1, tail_input_all->nb[1], row * tail_input_all->nb[1]);
            cb(cur, "prefix_layer64_scratch_tail_input_ref_row", scratch_src_layer);

            ggml_tensor * inpSA = cur;

            cur = build_norm(cur, model.layers[scratch_src_layer].attn_norm, nullptr, LLM_NORM_RMS, scratch_src_layer);
            cb(cur, "prefix_layer64_scratch_ref_attn_norm", scratch_src_layer);

            ggml_build_forward_expand(gf, cur);

            ggml_tensor * serial_Qcur_full = build_lora_mm(model.layers[scratch_src_layer].wq, cur, model.layers[scratch_src_layer].wq_s);
            cb(serial_Qcur_full, "prefix_layer64_scratch_ref_Qcur_full", scratch_label_layer);

            ggml_tensor * serial_Qcur = ggml_view_3d(ctx0, serial_Qcur_full, n_embd_head, n_head, 1,
                ggml_element_size(serial_Qcur_full) * n_embd_head * 2,
                ggml_element_size(serial_Qcur_full) * n_embd_head * 2 * n_head, 0);
            cb(serial_Qcur, "prefix_layer64_scratch_ref_Qcur_reshaped", scratch_label_layer);

            serial_Qcur = build_norm(serial_Qcur, model.layers[scratch_src_layer].attn_q_norm, nullptr, LLM_NORM_RMS, scratch_src_layer);
            cb(serial_Qcur, "prefix_layer64_scratch_ref_Qcur_normed", scratch_label_layer);

            ggml_tensor * serial_gate = ggml_view_3d(ctx0, serial_Qcur_full, n_embd_head, n_head, 1,
                ggml_element_size(serial_Qcur_full) * n_embd_head * 2,
                ggml_element_size(serial_Qcur_full) * n_embd_head * 2 * n_head,
                ggml_element_size(serial_Qcur_full) * n_embd_head);
            serial_gate = ggml_cont_2d(ctx0, serial_gate, n_embd_head * n_head, 1);
            cb(serial_gate, "prefix_layer64_scratch_ref_gate", scratch_label_layer);

            serial_Qcur = ggml_rope_multi(
                    ctx0, serial_Qcur, inp_pos_rows[(size_t) row], nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            if (inp->get_attn()->self_k_rot) {
                serial_Qcur = qwen35_mul_mat_aux(ctx0, serial_Qcur, inp->get_attn()->self_k_rot);
            }
            cb(serial_Qcur, "prefix_layer64_scratch_ref_Qcur", scratch_label_layer);
            ggml_build_forward_expand(gf, serial_Qcur);

            serial_q_all = serial_q_all ? ggml_concat(ctx0, serial_q_all, serial_Qcur, 2) : serial_Qcur;
            serial_gate_all = serial_gate_all ? ggml_concat(ctx0, serial_gate_all, serial_gate, 1) : serial_gate;

            ggml_tensor * serial_attn = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, scratch_src_layer, row);
            cb(serial_attn, "prefix_layer64_scratch_ref_attn_output", scratch_src_layer);
            serial_attn_all = serial_attn_all ? ggml_concat(ctx0, serial_attn_all, serial_attn, 1) : serial_attn;

            if (qblock_prefix_layer64_inline_row_compare) {
                const auto * mctx_cur = inp->get_attn()->mctx;
                ggml_tensor * k = mctx_cur->get_k(ctx0, scratch_src_layer);
                const bool use_fa = (cparams.flash_attn || k->type == GGML_TYPE_I32);
                ggml_tensor * v = mctx_cur->get_v(
                        ctx0,
                        scratch_src_layer,
                        use_fa ? LLAMA_KV_V_LAYOUT_FOR_FA
                               : LLAMA_KV_V_LAYOUT_FOR_NON_FA);

                ggml_tensor * kq_mask = inp->get_attn()->get_kq_mask();
                kq_mask = ggml_view_4d(ctx0, kq_mask,
                        kq_mask->ne[0], 1, kq_mask->ne[2], kq_mask->ne[3],
                        kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], row * kq_mask->nb[1]);

                const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
                ggml_tensor * scratch_row = nullptr;
                if (qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_INLINE_DUP_ROW_PDMQ")) {
                    const int dup_n_raw = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_DUP_ROW_N", 2);
                    const int dup_n = std::max(2, std::min(4, dup_n_raw));
                    const int dup_n_eff = row + dup_n <= (int64_t) ubatch.n_tokens ? dup_n : 1;
                    ggml_tensor * q_dup = serial_Qcur;
                    for (int i = 1; i < dup_n_eff; ++i) {
                        q_dup = ggml_concat(ctx0, q_dup, serial_Qcur, 2);
                    }
                    ggml_tensor * mask_dup = inp->get_attn()->get_kq_mask();
                    mask_dup = ggml_view_4d(ctx0, mask_dup,
                            mask_dup->ne[0], dup_n_eff, mask_dup->ne[2], mask_dup->ne[3],
                            mask_dup->nb[1], mask_dup->nb[2], mask_dup->nb[3], row * mask_dup->nb[1]);
                    ggml_tensor * scratch_dup = build_attn_mha(q_dup, k, v, nullptr, mask_dup, nullptr, nullptr, kq_scale, scratch_label_layer);
                    cb(scratch_dup, "prefix_layer64_scratch_inline_dup_row_pregate_raw", scratch_label_layer);
                    scratch_row = ggml_view_2d(ctx0, scratch_dup, scratch_dup->ne[0], 1, scratch_dup->nb[1], 0);
                } else {
                    scratch_row = build_attn_mha(serial_Qcur, k, v, nullptr, kq_mask, nullptr, nullptr, kq_scale, scratch_label_layer);
                }
                cb(scratch_row, "prefix_layer64_scratch_inline_row_pregate_raw", scratch_label_layer);

                if (inp->get_attn()->self_v_rot) {
                    scratch_row = qwen35_mul_mat_aux(ctx0, scratch_row, inp->get_attn()->self_v_rot);
                }

                std::string trace_name = std::string("prefix_roweq_hidden_layer64_pdmq_pregate_row") + std::to_string(row);
                cb(scratch_row, trace_name.c_str(), scratch_label_layer);
                ggml_build_forward_expand(gf, scratch_row);

                ggml_tensor * scratch_gate_sigmoid = ggml_sigmoid(ctx0, serial_gate);
                cb(scratch_gate_sigmoid, "prefix_layer64_scratch_inline_row_gate_sigmoid", scratch_label_layer);
                ggml_tensor * scratch_gated = ggml_mul(ctx0, scratch_row, scratch_gate_sigmoid);
                trace_name = std::string("prefix_roweq_hidden_layer64_pdmq_gated_row") + std::to_string(row);
                cb(scratch_gated, trace_name.c_str(), scratch_label_layer);
                ggml_build_forward_expand(gf, scratch_gated);

                ggml_tensor * scratch_attn = build_lora_mm(model.layers[scratch_src_layer].wo, scratch_gated, model.layers[scratch_src_layer].wo_s);
                trace_name = std::string("prefix_roweq_hidden_layer64_pdmq_attn_row") + std::to_string(row);
                cb(scratch_attn, trace_name.c_str(), scratch_label_layer);
                ggml_build_forward_expand(gf, scratch_attn);
                inline_scratch_attn_all = inline_scratch_attn_all ? ggml_concat(ctx0, inline_scratch_attn_all, scratch_attn, 1) : scratch_attn;
            }

            cur = ggml_add(ctx0, serial_attn, inpSA);
            cb(cur, "prefix_layer64_scratch_ref_attn_residual", scratch_src_layer);

            ggml_tensor * ffn_residual = cur;

            ggml_tensor * attn_post_norm = build_norm(cur, model.layers[scratch_src_layer].attn_post_norm, nullptr, LLM_NORM_RMS, scratch_src_layer);
            cb(attn_post_norm, "prefix_layer64_scratch_ref_attn_post_norm", scratch_src_layer);

            cur = build_layer_ffn(attn_post_norm, scratch_src_layer);
            cb(cur, "prefix_layer64_scratch_ref_ffn_out", scratch_src_layer);

            cur = ggml_add(ctx0, cur, ffn_residual);
            cb(cur, "prefix_layer64_scratch_ref_post_ffn", scratch_src_layer);

            cur = build_cvec(cur, scratch_src_layer);
            cb(cur, "prefix_layer64_scratch_ref_l_out", scratch_src_layer);

            h_pre_norm_all = h_pre_norm_all ? ggml_concat(ctx0, h_pre_norm_all, cur, 1) : cur;
        }
        cb(serial_attn_all, "prefix_layer64_scratch_ref_attn_all", scratch_src_layer);
        cb(h_pre_norm_all, "prefix_layer64_scratch_h_pre_norm_all", -1);
        trace_rows(serial_attn_all, "layer64_ref_attn", scratch_label_layer);

        if (qblock_prefix_layer64_inline_row_compare) {
            cb(inline_scratch_attn_all, "prefix_layer64_scratch_pdmq_attn_all", scratch_label_layer);
            trace_rows(inline_scratch_attn_all, "layer64_pdmq_attn", scratch_label_layer);
        } else {
            // Force the scratch Q-only PDMQ node to depend on the serial layer-63
            // output without changing its input values. This keeps the scratch read
            // of layer-63 K/V after the real serial K/V writes in the graph.
            ggml_tensor * zero_dep = ggml_scale(ctx0, h_pre_norm_all, 0.0f);
            cb(zero_dep, "prefix_layer64_scratch_zero_dep", scratch_label_layer);
            ggml_tensor * scratch_input_all = ggml_add(ctx0, tail_input_all, zero_dep);
            cb(scratch_input_all, "prefix_layer64_scratch_input_all_dep", scratch_label_layer);

            ggml_tensor * scratch_attn_all = build_layer_attn_prefix_all_q_only_scratch(
                    inp->get_attn(), scratch_input_all, inp_pos, sections, scratch_src_layer, scratch_label_layer,
                    serial_q_all, serial_gate_all);
            cb(scratch_attn_all, "prefix_layer64_scratch_pdmq_attn_all", scratch_label_layer);
            trace_rows(scratch_attn_all, "layer64_pdmq_attn", scratch_label_layer);
        }

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_layer64_scratch_h_pre_norm_row", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active || target_top1_active_from_logits) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
            }
        }
    } else if (qblock_prefix_fused_tail) {
        GGML_ASSERT(qblock_prefix_fused_tail_layer == n_transformer_layers - 1);
        GGML_ASSERT(!hparams.is_recurrent(qblock_prefix_fused_tail_layer));

        ggml_tensor * tail_input_all = nullptr;
        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], row * inpL->nb[1]);
            cb(cur, "prefix_qblock_fused_tail_input_row", -1);

            for (int il = 0; il < qblock_prefix_fused_tail_layer; ++il) {
                ggml_tensor * inpSA = cur;

                cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(cur, "prefix_qblock_fused_tail_attn_norm", il);

                ggml_build_forward_expand(gf, cur);

                if (hparams.is_recurrent(il)) {
                    cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                            commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                } else {
                    cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                }

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "prefix_qblock_fused_tail_attn_residual", il);

                ggml_tensor * ffn_residual = cur;

                ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_post_norm, "prefix_qblock_fused_tail_attn_post_norm", il);

                cur = build_layer_ffn(attn_post_norm, il);
                cb(cur, "prefix_qblock_fused_tail_ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_residual);
                cb(cur, "prefix_qblock_fused_tail_post_ffn", il);

                cur = build_cvec(cur, il);
                cb(cur, "prefix_qblock_fused_tail_l_out", il);
            }

            tail_input_all = tail_input_all ? ggml_concat(ctx0, tail_input_all, cur, 1) : cur;
        }
        cb(tail_input_all, "prefix_qblock_fused_tail_input_all", qblock_prefix_fused_tail_layer);

        ggml_tensor * inpSA_all = tail_input_all;
        ggml_tensor * attn_norm_all = build_norm(tail_input_all, model.layers[qblock_prefix_fused_tail_layer].attn_norm, nullptr, LLM_NORM_RMS, qblock_prefix_fused_tail_layer);
        cb(attn_norm_all, "prefix_qblock_fused_tail_attn_norm_all", qblock_prefix_fused_tail_layer);
        ggml_build_forward_expand(gf, attn_norm_all);

        ggml_tensor * cur_all = build_layer_attn_prefix_all(inp->get_attn(), attn_norm_all, inp_pos, sections, qblock_prefix_fused_tail_layer);
        cb(cur_all, "prefix_qblock_fused_tail_attn_output_all", qblock_prefix_fused_tail_layer);

        cur_all = ggml_add(ctx0, cur_all, inpSA_all);
        cb(cur_all, "prefix_qblock_fused_tail_attn_residual_all", qblock_prefix_fused_tail_layer);

        for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
            ggml_tensor * cur = ggml_view_2d(ctx0, cur_all, n_embd, 1, cur_all->nb[1], row * cur_all->nb[1]);
            cb(cur, "prefix_qblock_fused_tail_ffn_input_row", qblock_prefix_fused_tail_layer);

            ggml_tensor * ffn_residual = cur;

            ggml_tensor * attn_post_norm = build_norm(cur, model.layers[qblock_prefix_fused_tail_layer].attn_post_norm, nullptr, LLM_NORM_RMS, qblock_prefix_fused_tail_layer);
            cb(attn_post_norm, "prefix_qblock_fused_tail_attn_post_norm_row", qblock_prefix_fused_tail_layer);

            cur = build_layer_ffn(attn_post_norm, qblock_prefix_fused_tail_layer);
            cb(cur, "prefix_qblock_fused_tail_ffn_out_row", qblock_prefix_fused_tail_layer);

            cur = ggml_add(ctx0, cur, ffn_residual);
            cb(cur, "prefix_qblock_fused_tail_post_ffn_row", qblock_prefix_fused_tail_layer);

            cur = build_cvec(cur, qblock_prefix_fused_tail_layer);
            cb(cur, "prefix_qblock_fused_tail_l_out_row", qblock_prefix_fused_tail_layer);

            h_pre_norm_all = h_pre_norm_all ? ggml_concat(ctx0, h_pre_norm_all, cur, 1) : cur;
        }
        cb(h_pre_norm_all, "prefix_qblock_fused_tail_h_pre_norm_all", -1);

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_qblock_fused_tail_h_pre_norm_row", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active || target_top1_active_from_logits) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
            }
        }
    } else if (qblock_prefix_fused_layer_major) {
        ggml_tensor * cur_all = inpL;
        cb(cur_all, "prefix_qblock_fused_input_all", -1);

        for (int il = 0; il < n_transformer_layers; ++il) {
            const bool fused_attn_layer = !hparams.is_recurrent(il) &&
                    qwen35_prefix_qblock_fused_verify_layer_enabled(il, n_transformer_layers);

            if (fused_attn_layer) {
                ggml_tensor * inpSA_all = cur_all;

                ggml_tensor * attn_norm_all = build_norm(cur_all, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_norm_all, "prefix_qblock_fused_attn_norm_all", il);
                ggml_build_forward_expand(gf, attn_norm_all);

                cur_all = build_layer_attn_prefix_all(inp->get_attn(), attn_norm_all, inp_pos, sections, il);
                cb(cur_all, "prefix_qblock_fused_attn_output_all", il);

                cur_all = ggml_add(ctx0, cur_all, inpSA_all);
                cb(cur_all, "prefix_qblock_fused_attn_residual_all", il);

                ggml_tensor * next_all = nullptr;
                for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                    ggml_tensor * cur = ggml_view_2d(ctx0, cur_all, n_embd, 1, cur_all->nb[1], row * cur_all->nb[1]);
                    cb(cur, "prefix_qblock_fused_ffn_input_row", il);

                    ggml_tensor * ffn_residual = cur;

                    ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                    cb(attn_post_norm, "prefix_qblock_fused_attn_post_norm_row", il);

                    cur = build_layer_ffn(attn_post_norm, il);
                    cb(cur, "prefix_qblock_fused_ffn_out_row", il);

                    cur = ggml_add(ctx0, cur, ffn_residual);
                    cb(cur, "prefix_qblock_fused_post_ffn_row", il);

                    cur = build_cvec(cur, il);
                    cb(cur, "prefix_qblock_fused_l_out_row", il);

                    next_all = next_all ? ggml_concat(ctx0, next_all, cur, 1) : cur;
                }
                cur_all = next_all;
                cb(cur_all, "prefix_qblock_fused_l_out_all", il);
            } else {
                ggml_tensor * next_all = nullptr;
                for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                    ggml_tensor * cur = ggml_view_2d(ctx0, cur_all, n_embd, 1, cur_all->nb[1], row * cur_all->nb[1]);
                    cb(cur, "prefix_qblock_fused_serial_input_row", il);

                    ggml_tensor * inpSA = cur;

                    cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
                    cb(cur, "prefix_qblock_fused_serial_attn_norm", il);

                    ggml_build_forward_expand(gf, cur);

                    if (hparams.is_recurrent(il)) {
                        cur = build_layer_attn_linear_prefix_row(inp->get_recr(), cur, conv_state[il], ssm_state[il], il, row, ubatch.n_tokens,
                                commit_only || skip_verify_snapshots, snapshot_slot_limit, snapshot_slot_override);
                    } else {
                        cur = build_layer_attn_prefix_row(inp->get_attn(), cur, inp_pos_rows[(size_t) row], sections, il, row);
                    }

                    cur = ggml_add(ctx0, cur, inpSA);
                    cb(cur, "prefix_qblock_fused_serial_attn_residual", il);

                    ggml_tensor * ffn_residual = cur;

                    ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
                    cb(attn_post_norm, "prefix_qblock_fused_serial_attn_post_norm", il);

                    cur = build_layer_ffn(attn_post_norm, il);
                    cb(cur, "prefix_qblock_fused_serial_ffn_out", il);

                    cur = ggml_add(ctx0, cur, ffn_residual);
                    cb(cur, "prefix_qblock_fused_serial_post_ffn", il);

                    cur = build_cvec(cur, il);
                    cb(cur, "prefix_qblock_fused_serial_l_out", il);

                    next_all = next_all ? ggml_concat(ctx0, next_all, cur, 1) : cur;
                }
                cur_all = next_all;
                cb(cur_all, "prefix_qblock_fused_serial_l_out_all", il);
            }
        }

        h_pre_norm_all = cur_all;
        cb(h_pre_norm_all, "prefix_qblock_fused_h_pre_norm_all", -1);

        if (!commit_only && !batch_output_head) {
            for (int64_t row = 0; row < (int64_t) ubatch.n_tokens; ++row) {
                ggml_tensor * h_row = ggml_view_2d(ctx0, h_pre_norm_all, n_embd, 1, h_pre_norm_all->nb[1], row * h_pre_norm_all->nb[1]);
                cb(h_row, "prefix_qblock_fused_h_pre_norm_row", -1);

                ggml_tensor * embd_row = build_norm(h_row, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active || target_top1_active_from_logits) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
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
                cb(cur, "prefix_post_ffn", il);

                cur = build_cvec(cur, il);
                cb(cur, "prefix_l_out", il);
            }

            cb(cur, "prefix_h_pre_norm_row", -1);
            h_pre_norm_all = h_pre_norm_all ? ggml_concat(ctx0, h_pre_norm_all, cur, 1) : cur;

            if (!commit_only && !batch_output_head) {
                ggml_tensor * embd_row = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
                cb(embd_row, "prefix_result_norm_row", -1);
                embd_all = embd_all ? ggml_concat(ctx0, embd_all, embd_row, 1) : embd_row;

                if (!target_top1_active || target_top1_active_from_logits) {
                    ggml_tensor * logits_row = build_lora_mm(model.output, embd_row);
                    cb(logits_row, "prefix_result_output_row", -1);
                    logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_row, 1) : logits_row;
                }
            }
        }
    }

    if (batch_output_head) {
        embd_all = build_norm(h_pre_norm_all, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(embd_all, "prefix_result_norm_batched", -1);

        if (!target_top1_active || target_top1_active_from_logits) {
            logits_all = build_lora_mm(model.output, embd_all);
            cb(logits_all, "prefix_result_output_batched", -1);
        }
    }

    if (target_top1_active) {
        ggml_tensor * sampled = nullptr;
        if (target_top1_active_from_logits) {
            sampled = ggml_top_k(ctx0, logits_all, 1);
            cb(sampled, "prefix_target_logits_top1_active", -1);
        } else {
            sampled = ggml_lm_head_top_k(ctx0, model.output, embd_all, 1);
            cb(sampled, "prefix_target_lm_head_top1_active", -1);
        }
        qwen35_lm_head_top1_apply_eog_mask(model, sampled);
        res->t_mtp_target_top1_fused_all = sampled;
    } else if (!commit_only) {
        const char * target_top1_shadow = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW");
        if (target_top1_shadow && atoi(target_top1_shadow) != 0 &&
                lm_head_top1_direct_supported) {
            ggml_tensor * fused_top1 = ggml_lm_head_top_k(ctx0, model.output, embd_all, 1);
            cb(fused_top1, "prefix_target_lm_head_top1_fused", -1);
            res->t_mtp_target_top1_fused_all = fused_top1;
        }
    }

    res->t_h_pre_norm = h_pre_norm_all;
    res->t_embd       = embd_all;
    res->t_logits     = logits_all;

    ggml_build_forward_expand(gf, h_pre_norm_all);
    if (!commit_only) {
        ggml_build_forward_expand(gf, embd_all);
        if (res->t_mtp_target_top1_fused_all != nullptr) {
            ggml_build_forward_expand(gf, res->t_mtp_target_top1_fused_all);
        }
        if (!target_top1_active || target_top1_active_from_logits) {
            ggml_build_forward_expand(gf, logits_all);
        }
    }
}

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_layer_attn_prefix_all(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    const int64_t n_rows = cur->ne[1];
    GGML_ASSERT(n_rows == (int64_t) ubatch.n_tokens);

    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    cb(Qcur_full, "prefix_fused_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_rows,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1], 0);
    cb(Qcur, "prefix_fused_Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "prefix_fused_Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "prefix_fused_Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "prefix_fused_Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_rows);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "prefix_fused_Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_rows,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1],
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_rows);
    cb(gate, "prefix_fused_gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_rows);

    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "prefix_fused_Qcur", il);
    cb(Kcur, "prefix_fused_Kcur", il);
    cb(Vcur, "prefix_fused_Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "prefix_fused_attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "prefix_fused_gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "prefix_fused_attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "prefix_fused_attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_layer_attn_prefix_all_q_only_scratch(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il_src,
        int                       il_label,
        ggml_tensor *             qcur_override,
        ggml_tensor *             gate_override) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    const int64_t n_rows = cur->ne[1];
    GGML_ASSERT(n_rows == (int64_t) ubatch.n_tokens);

    ggml_tensor * Qcur = qcur_override;
    ggml_tensor * gate = gate_override;
    if (Qcur != nullptr || gate != nullptr) {
        GGML_ASSERT(Qcur != nullptr && gate != nullptr);
        GGML_ASSERT(Qcur->ne[0] == n_embd_head && Qcur->ne[1] == n_head && Qcur->ne[2] == n_rows);
        GGML_ASSERT(gate->ne[0] == n_embd_head * n_head && gate->ne[1] == n_rows);
        cb(Qcur, "prefix_layer64_scratch_Qcur_serial_override", il_label);
        cb(gate, "prefix_layer64_scratch_gate_serial_override", il_label);
    } else {
        ggml_tensor * Qcur_full = build_lora_mm(model.layers[il_src].wq, cur, model.layers[il_src].wq_s);
        cb(Qcur_full, "prefix_layer64_scratch_Qcur_full", il_label);

        Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_rows,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1], 0);
        cb(Qcur, "prefix_layer64_scratch_Qcur_reshaped", il_label);

        Qcur = build_norm(Qcur, model.layers[il_src].attn_q_norm, nullptr, LLM_NORM_RMS, il_src);
        cb(Qcur, "prefix_layer64_scratch_Qcur_normed", il_label);

        gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_rows,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1],
            ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_rows);
        cb(gate, "prefix_layer64_scratch_gate_reshaped", il_label);

        Qcur = ggml_rope_multi(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "prefix_layer64_scratch_Qcur", il_label);

        if (inp_attn->self_k_rot) {
            Qcur = qwen35_mul_mat_aux(ctx0, Qcur, inp_attn->self_k_rot);
        }
    }
    if (qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_TRACE_INTERNALS")) {
        for (int64_t row = 0; row < n_rows; ++row) {
            ggml_tensor * row_view = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, 1,
                    Qcur->nb[1], Qcur->nb[2], row * Qcur->nb[2]);
            const std::string name = std::string("prefix_roweq_hidden_layer64_pdmq_q_row") + std::to_string(row);
            cb(row_view, name.c_str(), il_label);
            ggml_build_forward_expand(gf, row_view);
        }
    }
    ggml_build_forward_expand(gf, Qcur);

    const auto * mctx_cur = inp_attn->mctx;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il_src);

    const bool use_fa = (cparams.flash_attn || k->type == GGML_TYPE_I32);
    ggml_tensor * v = mctx_cur->get_v(
            ctx0,
            il_src,
            use_fa ? LLAMA_KV_V_LAYOUT_FOR_FA
                   : LLAMA_KV_V_LAYOUT_FOR_NON_FA);

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    auto trace_layer64_pdmq_stage = [&](ggml_tensor * t, const char * stage) {
        if (!qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_TRACE_INTERNALS")) {
            return;
        }
        for (int64_t row = 0; row < n_rows; ++row) {
            ggml_tensor * row_view = ggml_view_2d(ctx0, t, t->ne[0], 1, t->nb[1], row * t->nb[1]);
            const std::string name = std::string("prefix_roweq_hidden_layer64_pdmq_") + stage + "_row" + std::to_string(row);
            cb(row_view, name.c_str(), il_label);
            ggml_build_forward_expand(gf, row_view);
        }
    };

    auto build_layer64_scratch_attn_mha = [&](ggml_tensor * q_arg, ggml_tensor * mask_arg) -> ggml_tensor * {
        const bool override_qblock_active = qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_NO_QBLOCK_ACTIVE");
        const bool force_pdmq = qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_FORCE_PDMQ");

        const char * old_qblock_env = getenv("LLAMA_MTP_QBLOCK_ACTIVE");
        const bool had_old_qblock = old_qblock_env != nullptr;
        const std::string old_qblock_value = had_old_qblock ? old_qblock_env : "";
        const char * old_require_env = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        const bool had_old_require = old_require_env != nullptr;
        const std::string old_require_value = had_old_require ? old_require_env : "";

        if (override_qblock_active) {
            setenv("LLAMA_MTP_QBLOCK_ACTIVE", "0", 1);
        }
        if (force_pdmq) {
            setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_dot4_mmq", 1);
        }

        ggml_tensor * out = build_attn_mha(q_arg, k, v, nullptr, mask_arg, nullptr, nullptr, kq_scale, il_label);

        if (force_pdmq) {
            if (had_old_require) {
                setenv("GGML_CUDA_FA_ROUTE_REQUIRE", old_require_value.c_str(), 1);
            } else {
                unsetenv("GGML_CUDA_FA_ROUTE_REQUIRE");
            }
        }
        if (override_qblock_active) {
            if (had_old_qblock) {
                setenv("LLAMA_MTP_QBLOCK_ACTIVE", old_qblock_value.c_str(), 1);
            } else {
                unsetenv("LLAMA_MTP_QBLOCK_ACTIVE");
            }
        }
        return out;
    };

    if (qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_ROW_SLICED")) {
        ggml_tensor * pregate_all = nullptr;
        ggml_tensor * gated_all   = nullptr;
        ggml_tensor * attn_all    = nullptr;

        for (int64_t row = 0; row < n_rows; ++row) {
            ggml_tensor * q_row = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, 1,
                    Qcur->nb[1], Qcur->nb[2], row * Qcur->nb[2]);
            cb(q_row, "prefix_layer64_scratch_row_sliced_q", il_label);

            ggml_tensor * kq_mask_row = ggml_view_4d(ctx0, kq_mask,
                    kq_mask->ne[0], 1, kq_mask->ne[2], kq_mask->ne[3],
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], row * kq_mask->nb[1]);
            cb(kq_mask_row, "prefix_layer64_scratch_row_sliced_kq_mask", il_label);

            ggml_tensor * row_cur = nullptr;
            if (qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_DUP_ROW_PDMQ")) {
                const int dup_n_raw = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_DUP_ROW_N", 2);
                const int dup_n = std::max(2, std::min(4, dup_n_raw));
                const int dup_n_eff = row + dup_n <= n_rows ? dup_n : 1;
                ggml_tensor * q_dup = q_row;
                for (int i = 1; i < dup_n_eff; ++i) {
                    q_dup = ggml_concat(ctx0, q_dup, q_row, 2);
                }
                ggml_tensor * mask_dup = ggml_view_4d(ctx0, kq_mask,
                        kq_mask->ne[0], dup_n_eff, kq_mask->ne[2], kq_mask->ne[3],
                        kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], row * kq_mask->nb[1]);
                cb(q_dup, "prefix_layer64_scratch_dup_row_q", il_label);
                cb(mask_dup, "prefix_layer64_scratch_dup_row_kq_mask", il_label);
                ggml_tensor * row_dup = build_layer64_scratch_attn_mha(q_dup, mask_dup);
                cb(row_dup, "prefix_layer64_scratch_dup_row_pregate_raw", il_label);
                row_cur = ggml_view_2d(ctx0, row_dup, row_dup->ne[0], 1, row_dup->nb[1], 0);
                cb(row_cur, "prefix_layer64_scratch_dup_row_pregate_first", il_label);
            } else {
                row_cur = build_layer64_scratch_attn_mha(q_row, kq_mask_row);
            }
            cb(row_cur, "prefix_layer64_scratch_row_sliced_pregate_raw", il_label);

            if (inp_attn->self_v_rot) {
                row_cur = qwen35_mul_mat_aux(ctx0, row_cur, inp_attn->self_v_rot);
            }
            pregate_all = pregate_all ? ggml_concat(ctx0, pregate_all, row_cur, 1) : row_cur;

            ggml_tensor * gate_row = ggml_view_2d(ctx0, gate, gate->ne[0], 1, gate->nb[1], row * gate->nb[1]);
            ggml_tensor * gate_sigmoid_row = ggml_sigmoid(ctx0, gate_row);
            cb(gate_sigmoid_row, "prefix_layer64_scratch_row_sliced_gate_sigmoid", il_label);

            row_cur = ggml_mul(ctx0, row_cur, gate_sigmoid_row);
            cb(row_cur, "prefix_layer64_scratch_row_sliced_gated", il_label);
            gated_all = gated_all ? ggml_concat(ctx0, gated_all, row_cur, 1) : row_cur;

            row_cur = build_lora_mm(model.layers[il_src].wo, row_cur, model.layers[il_src].wo_s);
            cb(row_cur, "prefix_layer64_scratch_row_sliced_attn", il_label);
            attn_all = attn_all ? ggml_concat(ctx0, attn_all, row_cur, 1) : row_cur;
        }

        cb(pregate_all, "prefix_layer64_scratch_attn_pregate", il_label);
        trace_layer64_pdmq_stage(pregate_all, "pregate");
        cb(gated_all, "prefix_layer64_scratch_attn_gated", il_label);
        trace_layer64_pdmq_stage(gated_all, "gated");
        cb(attn_all, "prefix_layer64_scratch_attn_output", il_label);
        trace_layer64_pdmq_stage(attn_all, "attn");
        return attn_all;
    }

    cur = build_layer64_scratch_attn_mha(Qcur, kq_mask);
    cb(cur, "prefix_layer64_scratch_attn_pregate", il_label);

    if (inp_attn->self_v_rot) {
        cur = qwen35_mul_mat_aux(ctx0, cur, inp_attn->self_v_rot);
    }
    trace_layer64_pdmq_stage(cur, "pregate");

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "prefix_layer64_scratch_gate_sigmoid", il_label);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "prefix_layer64_scratch_attn_gated", il_label);
    trace_layer64_pdmq_stage(cur, "gated");

    cur = build_lora_mm(model.layers[il_src].wo, cur, model.layers[il_src].wo_s);
    cb(cur, "prefix_layer64_scratch_attn_output", il_label);
    trace_layer64_pdmq_stage(cur, "attn");

    return cur;
}

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_attn_prefix_row(
        llm_graph_input_attn_kv * inp_attn,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        float                     kq_scale,
        int                       il,
        int64_t                   row) {
    if (inp_attn->self_k_rot) {
        q_cur = qwen35_mul_mat_aux(ctx0, q_cur, inp_attn->self_k_rot);
        k_cur = qwen35_mul_mat_aux(ctx0, k_cur, inp_attn->self_k_rot);
    }

    if (qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_TRACE_INTERNALS") &&
            qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_PROBE")) {
        const int scratch_src_tail = n_layer - (int) hparams.nextn_predict_layers - 1;
        if (il == scratch_src_tail) {
            const int scratch_label_layer = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_LABEL_LAYER", n_layer - (int) hparams.nextn_predict_layers);
            const std::string name = std::string("prefix_roweq_hidden_layer64_ref_q_row") + std::to_string(row);
            cb(q_cur, name.c_str(), scratch_label_layer);
            ggml_build_forward_expand(gf, q_cur);
        }
    }

    if (inp_attn->self_v_rot) {
        v_cur = qwen35_mul_mat_aux(ctx0, v_cur, inp_attn->self_v_rot);
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
        cur = qwen35_mul_mat_aux(ctx0, cur, inp_attn->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_layer_attn_prefix_row(
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

    auto trace_layer64_ref_stage = [&](ggml_tensor * t, const char * stage) {
        const int scratch_src_tail = n_layer - (int) hparams.nextn_predict_layers - 1;
        if (!qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_TRACE_INTERNALS") ||
                !qwen35_env_enabled("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_PROBE") ||
                il != scratch_src_tail) {
            return;
        }
        const int scratch_label_layer = qwen35_env_i32("LLAMA_MTP_QBLOCK_PREFIX_LAYER64_SCRATCH_LABEL_LAYER", n_layer - (int) hparams.nextn_predict_layers);
        const std::string name = std::string("prefix_roweq_hidden_layer64_ref_") + stage + "_row" + std::to_string(row);
        cb(t, name.c_str(), scratch_label_layer);
        ggml_build_forward_expand(gf, t);
    };

    cur = build_attn_prefix_row(inp_attn, Qcur, Kcur, Vcur, kq_scale, il, row);
    cb(cur, "prefix_attn_pregate", il);
    trace_layer64_ref_stage(cur, "pregate");

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "prefix_gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "prefix_attn_gated", il);
    trace_layer64_ref_stage(cur, "gated");

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "prefix_attn_output", il);
    trace_layer64_ref_stage(cur, "attn");

    return cur;
}

void llama_model_qwen35::graph_prefix_verify::copy_prefix_snapshot(
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

void llama_model_qwen35::graph_prefix_verify::copy_prefix_conv_reconstruct_snapshot_direct(
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

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_layer_attn_linear_prefix_row(
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
    const bool prefix_state_candidate_trace    = qwen35_prefix_state_candidate_trace_enabled(il);
    const bool prefix_state_reconstruct_copy   = qwen35_prefix_state_reconstruct_copy_enabled(il);
    const bool prefix_state_source_trace       = qwen35_prefix_state_source_trace_enabled(il);
    const bool prefix_r_direct_reconstruct_copy =
        prefix_state_reconstruct_copy &&
        qwen35_prefix_r_direct_reconstruct_copy_enabled(il) &&
        !qwen35_env_enabled("LLAMA_MTP_PREFIX_SNAPSHOT_TRACE");
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
            qwen35_prefix_s_state_only_reconstruct_copy_enabled(il) &&
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

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph_prefix_verify::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "prefix_ffn_dense_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35 MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // The MTP block lives at the source file's original layer index.
    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common build_inp_embd_h()
    GGML_ASSERT(hparams.n_embd_inp() == hparams.n_embd && "MTP embd/h input dimensions must match until llm_graph_input_embd_h tracks both dimensions");
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

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

    auto * inp_attn       = build_attn_inp_kv();

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

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
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
        fprintf(stderr, "MTP_BYPASS(qwen35): stage=%s cur=[%lld,%lld]\n",
            bypass, (long long) cur->ne[0], (long long) cur->ne[1]);
    }

    cur   = ggml_get_rows(ctx0, cur, inp_out_ids);

    // Pre-norm hidden state: used by the AR draft loop to seed the next MTP step.
    // (In the trunk graph this is `t_h_pre_norm`; the MTP head reuses the same slot.)
    cb(cur, "h_pre_norm", -1);
    res->t_h_pre_norm = cur;
    res->t_mtp_out    = cur;

    const char * mtp_block_only = getenv("LLAMA_MTP_BLOCK_ONLY");
    if (mtp_block_only && atoi(mtp_block_only) != 0) {
        ggml_build_forward_expand(gf, res->t_mtp_out);
        return;
    }

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    // LLAMA_MTP_HEAD_PROBE: dump output tensor dimensions
    if (getenv("LLAMA_MTP_HEAD_PROBE")) {
        fprintf(stderr, "MTP_HEAD_PROBE(qwen35) output: head_w=[%lld,%lld] cur=[%lld,%lld] n_vocab=%d\n",
            (long long)head_w->ne[0], (long long)head_w->ne[1],
            (long long)cur->ne[0], (long long)cur->ne[1],
            llama_vocab_n_tokens(llama_model_get_vocab(&model)));
    }

    const char * fused_lm_head_topk = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK");
    if (fused_lm_head_topk && atoi(fused_lm_head_topk) != 0 && head_w->type == GGML_TYPE_Q6_K && head_s == nullptr) {
        ggml_tensor * sampled = ggml_lm_head_top_k(ctx0, head_w, cur, 1);
        cb(sampled, "mtp_lm_head_top1", -1);

        int32_t out_idx = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (!ubatch.output[i]) {
                continue;
            }
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            ggml_tensor * sampled_seq = ggml_view_1d(ctx0, sampled, 1, out_idx * sampled->nb[1]);
            sampled_seq = ggml_cont(ctx0, sampled_seq);
            ggml_format_name(sampled_seq, "mtp_lm_head_top1_seq_%d", seq_id);
            res->t_sampled[seq_id] = sampled_seq;
            ggml_build_forward_expand(gf, sampled_seq);
            ++out_idx;
        }

        ggml_build_forward_expand(gf, sampled);
        return;
    }

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
