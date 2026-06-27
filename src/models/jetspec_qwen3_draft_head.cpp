#include "models.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct jetspec_meta {
    std::string name;
    std::string head_arch;
    std::string source_arch;
    std::string dtype;
    uint32_t block_size = 0;
    uint32_t draft_depth = 0;
    uint32_t mask_token_id = 0;
    uint32_t num_target_layers = 0;
    uint32_t embedding_length = 0;
    uint32_t feed_forward_length = 0;
    uint32_t block_count = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t key_length = 0;
    uint32_t value_length = 0;
    uint32_t vocab_size = 0;
    float rope_freq_base = 0.0f;
    float rms_eps = 0.0f;
    bool causal_head = false;
    bool requires_target_embeddings = false;
    bool requires_target_lm_head = false;
    bool preview = false;
    bool metadata_only = false;
    bool runtime_supported = false;
    std::vector<uint32_t> target_layer_ids;
};

static bool jetspec_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::atoi(value) != 0;
}

static std::string jetspec_join(const std::vector<std::string> & values) {
    std::string out;
    for (const auto & value : values) {
        if (!out.empty()) {
            out += "; ";
        }
        out += value;
    }
    return out;
}

static int jetspec_find_required_key(const gguf_context * ctx, const char * key) {
    const int id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("missing JetSpec metadata key: ") + key);
    }
    return id;
}

static std::string jetspec_read_string(const gguf_context * ctx, const char * key, bool required = true) {
    const int id = required ? jetspec_find_required_key(ctx, key) : gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("JetSpec metadata key has wrong type: ") + key);
    }
    return gguf_get_val_str(ctx, id);
}

static uint32_t jetspec_read_u32(const gguf_context * ctx, const char * key) {
    const int id = jetspec_find_required_key(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("JetSpec metadata key has wrong type: ") + key);
    }
    return gguf_get_val_u32(ctx, id);
}

static float jetspec_read_f32(const gguf_context * ctx, const char * key) {
    const int id = jetspec_find_required_key(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("JetSpec metadata key has wrong type: ") + key);
    }
    return gguf_get_val_f32(ctx, id);
}

static bool jetspec_read_bool(const gguf_context * ctx, const char * key) {
    const int id = jetspec_find_required_key(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(std::string("JetSpec metadata key has wrong type: ") + key);
    }
    return gguf_get_val_bool(ctx, id);
}

static std::vector<uint32_t> jetspec_read_u32_array(const gguf_context * ctx, const char * key) {
    const int id = jetspec_find_required_key(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("JetSpec metadata key has wrong type: ") + key);
    }

    const size_t n = gguf_get_arr_n(ctx, id);
    const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, id));
    return std::vector<uint32_t>(data, data + n);
}

static jetspec_meta jetspec_load_meta(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;
    jetspec_meta meta;

    meta.name = jetspec_read_string(ctx, "general.name", false);
    meta.head_arch = jetspec_read_string(ctx, "jetspec.architecture");
    meta.source_arch = jetspec_read_string(ctx, "jetspec.source_architecture");
    meta.dtype = jetspec_read_string(ctx, "jetspec.tensor_data_dtype");
    meta.block_size = jetspec_read_u32(ctx, "jetspec.block_size");
    meta.draft_depth = jetspec_read_u32(ctx, "jetspec.draft_depth");
    meta.causal_head = jetspec_read_bool(ctx, "jetspec.causal_head");
    meta.mask_token_id = jetspec_read_u32(ctx, "jetspec.mask_token_id");
    meta.target_layer_ids = jetspec_read_u32_array(ctx, "jetspec.target_layer_ids");
    meta.num_target_layers = jetspec_read_u32(ctx, "jetspec.num_target_layers");
    meta.requires_target_embeddings = jetspec_read_bool(ctx, "jetspec.requires_target_embeddings");
    meta.requires_target_lm_head = jetspec_read_bool(ctx, "jetspec.requires_target_lm_head");
    meta.embedding_length = jetspec_read_u32(ctx, "jetspec.embedding_length");
    meta.feed_forward_length = jetspec_read_u32(ctx, "jetspec.feed_forward_length");
    meta.block_count = jetspec_read_u32(ctx, "jetspec.block_count");
    meta.head_count = jetspec_read_u32(ctx, "jetspec.attention.head_count");
    meta.head_count_kv = jetspec_read_u32(ctx, "jetspec.attention.head_count_kv");
    meta.key_length = jetspec_read_u32(ctx, "jetspec.attention.key_length");
    meta.value_length = jetspec_read_u32(ctx, "jetspec.attention.value_length");
    meta.rope_freq_base = jetspec_read_f32(ctx, "jetspec.rope.freq_base");
    meta.rms_eps = jetspec_read_f32(ctx, "jetspec.attention.layer_norm_rms_epsilon");
    meta.vocab_size = jetspec_read_u32(ctx, "jetspec.vocab_size");
    meta.preview = jetspec_read_bool(ctx, "jetspec.experimental.preview");
    meta.metadata_only = jetspec_read_bool(ctx, "jetspec.experimental.metadata_only");
    meta.runtime_supported = jetspec_read_bool(ctx, "jetspec.experimental.runtime_supported");

    return meta;
}

static void jetspec_expect(bool condition, std::vector<std::string> & errors, const std::string & message) {
    if (!condition) {
        errors.push_back(message);
    }
}

static void jetspec_validate_meta(const jetspec_meta & meta, int n_tensors) {
    std::vector<std::string> errors;
    const std::vector<uint32_t> expected_target_layers = { 1, 10, 19, 28, 37 };

    jetspec_expect(meta.head_arch == "qwen3_draft_head", errors, "jetspec.architecture must be qwen3_draft_head");
    jetspec_expect(meta.source_arch == "DFlashDraftModel", errors, "jetspec.source_architecture must be DFlashDraftModel");
    jetspec_expect(meta.block_size == 16, errors, "jetspec.block_size must be 16");
    jetspec_expect(meta.draft_depth == 15, errors, "jetspec.draft_depth must be 15");
    jetspec_expect(meta.causal_head, errors, "jetspec.causal_head must be true");
    jetspec_expect(meta.mask_token_id == 248070, errors, "jetspec.mask_token_id must be 248070");
    jetspec_expect(meta.target_layer_ids == expected_target_layers, errors, "jetspec.target_layer_ids must be [1,10,19,28,37]");
    jetspec_expect(meta.num_target_layers == 40, errors, "jetspec.num_target_layers must be 40");
    jetspec_expect(meta.requires_target_embeddings, errors, "jetspec.requires_target_embeddings must be true");
    jetspec_expect(meta.requires_target_lm_head, errors, "jetspec.requires_target_lm_head must be true");
    jetspec_expect(meta.embedding_length == 2048, errors, "jetspec.embedding_length must be 2048");
    jetspec_expect(meta.feed_forward_length == 6144, errors, "jetspec.feed_forward_length must be 6144");
    jetspec_expect(meta.block_count == 8, errors, "jetspec.block_count must be 8");
    jetspec_expect(meta.head_count == 32, errors, "jetspec.attention.head_count must be 32");
    jetspec_expect(meta.head_count_kv == 4, errors, "jetspec.attention.head_count_kv must be 4");
    jetspec_expect(meta.key_length == 128, errors, "jetspec.attention.key_length must be 128");
    jetspec_expect(meta.value_length == 128, errors, "jetspec.attention.value_length must be 128");
    jetspec_expect(meta.rope_freq_base == 10000000.0f, errors, "jetspec.rope.freq_base must be 10000000.0");
    jetspec_expect(meta.rms_eps > 0.0f && meta.rms_eps <= 0.0000011f, errors, "jetspec.attention.layer_norm_rms_epsilon must be 1e-6");
    jetspec_expect(meta.vocab_size == 248320, errors, "jetspec.vocab_size must be 248320");
    jetspec_expect(meta.dtype == "bfloat16", errors, "jetspec.tensor_data_dtype must be bfloat16");
    jetspec_expect(meta.preview, errors, "jetspec.experimental.preview must be true for P5A files");
    jetspec_expect(n_tensors == 0 || n_tensors == 91, errors, "JetSpec P5A expects 0 metadata-only tensors or 91 BF16 payload tensors");
    jetspec_expect(n_tensors != 0 || meta.metadata_only, errors, "zero-tensor preview must set jetspec.experimental.metadata_only=true");
    jetspec_expect(n_tensors != 91 || !meta.metadata_only, errors, "91-tensor payload must set jetspec.experimental.metadata_only=false");

    if (!errors.empty()) {
        throw std::runtime_error("jetspec_qwen3_draft_head: metadata validation failed: " + jetspec_join(errors));
    }
}

static void jetspec_check_tensor(const llama_model_loader & ml, std::vector<std::string> & errors, const std::string & name, std::initializer_list<int64_t> ne) {
    const auto it = ml.weights_map.find(name);
    if (it == ml.weights_map.end()) {
        errors.push_back("missing tensor " + name);
        return;
    }

    const ggml_tensor * tensor = it->second.tensor;
    if (tensor->type != GGML_TYPE_BF16) {
        errors.push_back("tensor " + name + " must be BF16");
    }

    int i = 0;
    for (const int64_t expected : ne) {
        if (tensor->ne[i] != expected) {
            errors.push_back("tensor " + name + " has unexpected shape");
            return;
        }
        ++i;
    }
    for (; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != 1) {
            errors.push_back("tensor " + name + " has unexpected rank");
            return;
        }
    }
}

static void jetspec_validate_tensor_inventory(const llama_model_loader & ml, const jetspec_meta & meta) {
    if (ml.n_tensors == 0) {
        return;
    }

    std::vector<std::string> errors;
    if (ml.n_tensors != 91 || ml.weights_map.size() != 91) {
        errors.push_back("JetSpec tensor payload must contain exactly 91 tensor infos");
    }

    const int64_t hidden = meta.embedding_length;
    const int64_t concat = meta.embedding_length * static_cast<int64_t>(meta.target_layer_ids.size());
    const int64_t ffn = meta.feed_forward_length;
    const int64_t q_dim = meta.head_count * meta.key_length;
    const int64_t kv_dim = meta.head_count_kv * meta.key_length;

    jetspec_check_tensor(ml, errors, "draft.fc.weight", { concat, hidden });
    jetspec_check_tensor(ml, errors, "draft.hidden_norm.weight", { hidden });
    jetspec_check_tensor(ml, errors, "draft.norm.weight", { hidden });

    for (uint32_t il = 0; il < meta.block_count; ++il) {
        const std::string prefix = "draft.layers." + std::to_string(il) + ".";
        jetspec_check_tensor(ml, errors, prefix + "input_layernorm.weight", { hidden });
        jetspec_check_tensor(ml, errors, prefix + "mlp.down_proj.weight", { ffn, hidden });
        jetspec_check_tensor(ml, errors, prefix + "mlp.gate_proj.weight", { hidden, ffn });
        jetspec_check_tensor(ml, errors, prefix + "mlp.up_proj.weight", { hidden, ffn });
        jetspec_check_tensor(ml, errors, prefix + "post_attention_layernorm.weight", { hidden });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.k_norm.weight", { meta.key_length });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.k_proj.weight", { hidden, kv_dim });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.o_proj.weight", { q_dim, hidden });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.q_norm.weight", { meta.key_length });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.q_proj.weight", { hidden, q_dim });
        jetspec_check_tensor(ml, errors, prefix + "self_attn.v_proj.weight", { hidden, kv_dim });
    }

    if (!errors.empty()) {
        throw std::runtime_error("jetspec_qwen3_draft_head: tensor inventory validation failed: " + jetspec_join(errors));
    }
}

} // namespace

void llama_model_jetspec_qwen3_draft_head::load_hparams(llama_model_loader & ml) {
    const jetspec_meta meta = jetspec_load_meta(ml);
    jetspec_validate_meta(meta, ml.n_tensors);
    jetspec_validate_tensor_inventory(ml, meta);

    name = meta.name.empty() ? "JetSpec Qwen3 draft head" : meta.name;
    type = LLM_TYPE_UNKNOWN;

    hparams.n_ctx_train = meta.block_size;
    hparams.n_embd = meta.embedding_length;
    hparams.n_layer = meta.block_count;
    hparams.causal_attn = true;
    hparams.n_embd_head_k_full = meta.key_length;
    hparams.n_embd_head_v_full = meta.value_length;
    hparams.n_rot_full = meta.key_length;
    hparams.rope_freq_base_train = meta.rope_freq_base;
    hparams.rope_freq_scale_train = 1.0f;
    hparams.f_norm_rms_eps = meta.rms_eps;
    std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.end(), meta.head_count);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), meta.head_count_kv);
    std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.end(), meta.feed_forward_length);

    if (meta.preview && !jetspec_env_enabled("LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD")) {
        throw std::runtime_error("preview_not_allowed: JetSpec preview GGUFs require LLAMA_JETSPEC_ALLOW_PREVIEW_LOAD=1 for explicit loader inspection");
    }

    if (!meta.runtime_supported) {
        throw std::runtime_error("unsupported_runtime: JetSpec P5A preserves runtime_supported=false and stops before graph execution");
    }

    if (!jetspec_env_enabled("LLAMA_JETSPEC_EXPERIMENTAL")) {
        throw std::runtime_error("jetspec_experimental_gate_disabled: set LLAMA_JETSPEC_EXPERIMENTAL=1 only for explicit JetSpec development");
    }

    throw std::runtime_error("unsupported_runtime: JetSpec P5A loader registration is validation-only and has no executable graph");
}

void llama_model_jetspec_qwen3_draft_head::load_arch_hparams(llama_model_loader &) {
    throw std::runtime_error("unsupported_runtime: JetSpec P5A should fail before load_arch_hparams");
}

void llama_model_jetspec_qwen3_draft_head::load_arch_tensors(llama_model_loader &) {
    throw std::runtime_error("unsupported_runtime: JetSpec P5A should fail before load_arch_tensors");
}

std::unique_ptr<llm_graph_context> llama_model_jetspec_qwen3_draft_head::build_arch_graph(const llm_graph_params &) const {
    throw std::runtime_error("unsupported_runtime: JetSpec P5A has no graph execution path");
}
