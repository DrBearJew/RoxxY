// Compile-only JetSpec runtime contract smoke.
// This file is intentionally not wired into CMake or any production target.

#include "jetspec_runtime_contract.hpp"

#include <cstdint>
#include <type_traits>

using namespace llama_jetspec_experiment;

static_assert(jetspec_qwen36_block_size == 16, "staged block size drift");
static_assert(jetspec_qwen36_draft_depth == jetspec_qwen36_block_size - 1, "draft depth drift");
static_assert(jetspec_qwen36_concat_width == 10240, "target tap concat width drift");
static_assert(jetspec_qwen36_tensor_count == 91, "draft-head tensor count drift");
static_assert(static_cast<uint8_t>(tensor_payload_mode::metadata_only) == 0, "metadata-only enum drift");
static_assert(static_cast<uint8_t>(runtime_failure::unsupported_runtime) > 0, "failure enum drift");

int main() {
    round_state state;
    state.phase = runtime_phase::load_metadata;
    state.failure = runtime_failure::unsupported_runtime;
    state.loader.metadata.runtime_supported = false;
    state.loader.metadata.requires_target_embeddings = true;
    state.loader.metadata.requires_target_lm_head = true;
    state.loader.payload_mode = tensor_payload_mode::metadata_only;
    state.target.has_token_embeddings = false;
    state.target.has_lm_head = false;
    state.verify.past_len = 0;
    state.verify.node_count = 0;
    state.commit.correction_hidden_appended = false;

    draft_head_tensor_info tensor;
    tensor.gguf_name = "draft.fc.weight";
    tensor.hf_name = "fc.weight";
    tensor.shape = {2048, 10240};
    tensor.ggml_type = 30;
    tensor.nbytes = 41943040;
    state.loader.tensors.push_back(tensor);

    return state.loader.metadata.runtime_supported ? 1 : 0;
}
