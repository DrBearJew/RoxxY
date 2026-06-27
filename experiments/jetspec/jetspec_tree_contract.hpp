#pragma once

// Experimental JetSpec tree contract sketch.
//
// This header is intentionally staged under experiments/jetspec/ and is not
// included by any production llama.cpp source file. Do not add it to CMake or
// include it from src/common/tools until the GGUF draft-head loader and tree
// verifier are ready.

#include <cstdint>
#include <vector>

namespace llama_jetspec_experiment {

struct draft_candidate {
    int32_t token = -1;
    int32_t rank = -1;
    float logit = -3.4028234663852886e38F;
    float logprob = -3.4028234663852886e38F;
    float p = 0.0F;
};

struct draft_tree_node {
    int32_t token = -1;
    int32_t parent = -1; // -1 only for root
    int32_t depth = 0;   // root = 0
    int32_t rank = -1;   // candidate rank under parent, root = -1
    float cum_logprob = 0.0F;
};

struct draft_tree {
    std::vector<draft_tree_node> nodes;

    // Dense row-major ancestor matrix, N x N, bool-as-u8.
    // ancestor[row * N + col] != 0 means query node `row` may attend key node `col`.
    std::vector<uint8_t> ancestor;

    int32_t max_depth = 0;

    int32_t size() const {
        return (int32_t) nodes.size();
    }
};

struct accepted_path {
    // Root-inclusive node path. path.size() - 1 is accepted draft-token count.
    std::vector<int32_t> path;

    // Target argmax sampled at the final accepted node. This becomes the next
    // anchor/correction token when no child matches it.
    int32_t correction_token = -1;
};

// Planned first builder: JetSpec upstream `accum_logp`.
// Input topk_by_depth[d][rank] corresponds to draft logits for depth d+1.
// Output nodes are parent-before-child and root is node 0.
//
// This is a declaration-only contract for now; implementation stays out of the
// compiled server until tree verify/commit ownership is agreed.
draft_tree build_accum_logp_tree(
        int32_t root_token,
        const std::vector<std::vector<draft_candidate>> & topk_by_depth,
        int32_t budget);

// Planned verifier accept rule: same as JetSpec `tree_accept`.
// `target_argmax_by_node[i]` is the target greedy token for verifier row/node i.
// Walk from root while a child token matches the current node's target argmax.
accepted_path accept_greedy_path(
        const draft_tree & tree,
        const std::vector<int32_t> & target_argmax_by_node);

// Planned tree-causal mask builder: same as JetSpec `build_ancestor_matrix`.
// Prefix tokens are always visible to all tree rows and are not represented in
// this N x N node-only matrix.
std::vector<uint8_t> build_ancestor_matrix(const draft_tree & tree);

} // namespace llama_jetspec_experiment
