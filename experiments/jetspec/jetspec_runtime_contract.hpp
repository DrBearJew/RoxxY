#pragma once

// Experimental JetSpec runtime/loader state contract sketch.
//
// This header is intentionally staged under experiments/jetspec/ and is
// not included by any production llama.cpp source file. Do not add it to CMake
// or include it from src/common/tools until the GGUF draft-head loader, target tap
// capture, tree verifier, and rollback contracts have compiled parity tests.

#include "jetspec_tree_contract.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace llama_jetspec_experiment {

// Qwen3.6 JetSpec draft-head constants observed from
// JetSpec/jetspec-Qwen3.6-35B-A3B and validated by the inert GGUF plan.
constexpr int32_t jetspec_qwen36_block_size         = 16;
constexpr int32_t jetspec_qwen36_draft_depth        = 15;
constexpr int32_t jetspec_qwen36_mask_token_id      = 248070;
constexpr int32_t jetspec_qwen36_target_layers      = 40;
constexpr int32_t jetspec_qwen36_target_tap_count   = 5;
constexpr int32_t jetspec_qwen36_hidden_size        = 2048;
constexpr int32_t jetspec_qwen36_concat_width       = 10240;
constexpr int32_t jetspec_qwen36_draft_layers       = 8;
constexpr int32_t jetspec_qwen36_attention_heads    = 32;
constexpr int32_t jetspec_qwen36_attention_heads_kv = 4;
constexpr int32_t jetspec_qwen36_head_dim           = 128;
constexpr int32_t jetspec_qwen36_ffn_size           = 6144;
constexpr int32_t jetspec_qwen36_vocab_size         = 248320;
constexpr int32_t jetspec_qwen36_tensor_count       = 91;

static_assert(jetspec_qwen36_draft_depth == jetspec_qwen36_block_size - 1,
        "JetSpec block_size must be draft_depth + 1");
static_assert(jetspec_qwen36_concat_width == jetspec_qwen36_target_tap_count * jetspec_qwen36_hidden_size,
        "JetSpec fc input width must equal tap_count * hidden_size");

enum class tensor_payload_mode : uint8_t {
    metadata_only = 0,
    bf16_payload  = 1,
};

enum class runtime_phase : uint8_t {
    idle = 0,
    load_metadata,
    bind_target,
    draft,
    build_tree,
    verify_tree,
    accept_path,
    commit_round,
    failed,
};

enum class runtime_failure : uint8_t {
    none = 0,
    invalid_metadata,
    preview_not_allowed,
    missing_tensor,
    tensor_shape_mismatch,
    missing_target_embeddings,
    missing_target_lm_head,
    missing_target_hidden_tap,
    hidden_width_mismatch,
    verify_mask_violation,
    rejected_branch_leak,
    unsupported_runtime,
};

struct draft_head_metadata {
    std::string gguf_arch = "jetspec_qwen3_draft_head";
    std::string head_arch = "qwen3_draft_head";
    std::string source_arch = "DFlashDraftModel";

    int32_t block_size = jetspec_qwen36_block_size;
    int32_t draft_depth = jetspec_qwen36_draft_depth;
    int32_t mask_token_id = jetspec_qwen36_mask_token_id;
    int32_t target_layers = jetspec_qwen36_target_layers;
    std::vector<int32_t> target_layer_ids = {1, 10, 19, 28, 37};

    int32_t hidden_size = jetspec_qwen36_hidden_size;
    int32_t concat_width = jetspec_qwen36_concat_width;
    int32_t draft_layers = jetspec_qwen36_draft_layers;
    int32_t attention_heads = jetspec_qwen36_attention_heads;
    int32_t attention_heads_kv = jetspec_qwen36_attention_heads_kv;
    int32_t head_dim = jetspec_qwen36_head_dim;
    int32_t ffn_size = jetspec_qwen36_ffn_size;
    int32_t vocab_size = jetspec_qwen36_vocab_size;

    bool causal_head = true;
    bool requires_target_embeddings = true;
    bool requires_target_lm_head = true;
    bool runtime_supported = false; // must remain false until loader/graph parity exists
};

struct draft_head_tensor_info {
    std::string gguf_name;
    std::string hf_name;
    std::vector<int64_t> shape;
    int32_t ggml_type = 30; // GGML_TYPE_BF16 in current llama.cpp constants
    uint64_t nbytes = 0;
    uint64_t offset = 0;
};

struct draft_head_loader_plan {
    draft_head_metadata metadata;
    tensor_payload_mode payload_mode = tensor_payload_mode::metadata_only;
    std::vector<draft_head_tensor_info> tensors;

    // Set only by an explicit experimental runtime, never by preview files.
    bool preview_file = true;
    bool allow_preview_runtime = false;
};

struct target_model_bindings {
    bool has_token_embeddings = false;
    bool has_lm_head = false;
    bool can_capture_hidden_taps = false;

    int32_t hidden_size = 0;
    int32_t vocab_size = 0;
    int32_t layer_count = 0;
    std::vector<int32_t> captured_layer_ids;

    // Optional handles are intentionally abstract here; production code must own
    // them in llama.cpp-specific structs, not in this inert contract header.
    uint64_t token_embedding_handle = 0;
    uint64_t lm_head_handle = 0;
};

struct target_hidden_cache_state {
    int32_t width = jetspec_qwen36_concat_width;
    int32_t committed_token_count = 0;
    int32_t hidden_row_count = 0;

    // The invariant before and after every tree round is:
    // hidden_row_count == committed_token_count - 1.
    bool trails_committed_by_one = true;

    // Abstract row storage handle; no ownership implied in this inert contract.
    uint64_t row_storage_handle = 0;
};

struct tree_verify_plan {
    int32_t past_len = 0;
    int32_t node_count = 0;
    int32_t bucket_node_count = 0;

    // Dense node-only ancestor block, row-major N x N, bool-as-u8.
    std::vector<uint8_t> ancestor;

    // Optional bucketed additive bias over tree columns, row-major B x B.
    // Encoded as 0 for allowed and -inf in the future runtime representation;
    // this inert struct stores only the bool-equivalent visibility contract.
    std::vector<uint8_t> bucket_visibility;
};

struct round_commit_plan {
    accepted_path accepted;
    std::vector<int32_t> accepted_draft_tokens;
    int32_t correction_token = -1;

    // Node indices whose hidden/KV rows survive this round. Must equal
    // accepted.path and therefore include root node 0.
    std::vector<int32_t> appended_node_indices;

    // Node indices whose hidden/KV rows must be unreachable after commit.
    std::vector<int32_t> discarded_node_indices;

    bool correction_hidden_appended = false;
};

struct round_state {
    runtime_phase phase = runtime_phase::idle;
    runtime_failure failure = runtime_failure::none;

    draft_head_loader_plan loader;
    target_model_bindings target;
    target_hidden_cache_state hidden_cache;

    std::vector<int32_t> committed_tokens;
    draft_tree tree;
    tree_verify_plan verify;
    round_commit_plan commit;
};

} // namespace llama_jetspec_experiment
