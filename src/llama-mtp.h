#pragma once

#include "llama.h"

#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // views the storage below; do not free with llama_batch_free()

    std::vector<llama_token>    hook_token;
    std::vector<float>          hook_embd;
    std::vector<llama_pos>      hook_pos;
    std::vector<int32_t>        hook_n_seq_id;
    std::vector<llama_seq_id>   hook_seq_id_storage;
    std::vector<llama_seq_id *> hook_seq_id_ptrs;
    std::vector<int8_t>         hook_logits;

    // Cross-ubatch shift state: pair (h_p, x_{p+1}) at MTP pos p+1. The last
    // h-row of one ubatch needs the first token of the NEXT ubatch to pair
    // with, so it's stashed here until that next ubatch fires. Resets when
    // pos_start of the new ubatch != pending_pos+1 (new prompt or seq_rm gap).
    std::vector<float> pending_h;
    llama_pos          pending_pos = -1;
};
