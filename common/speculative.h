#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_branch_candidate {
    llama_token id = LLAMA_TOKEN_NULL;
    float logit = -1.0e30f;
    float p = 0.0f;
    int32_t rank = -1;
};

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // Optional per-depth branch candidates for tree/block verifiers. This does
    // not affect token selection: result remains the accepted linear draft path.
    // When populated, branch_candidates[depth][rank] describes the draft model's
    // alternate candidates for result[depth].
    std::vector<std::vector<common_speculative_branch_candidate>> * branch_candidates = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// process the batch using externally captured target pre-norm embedding rows.
// h_pre_norm must contain batch.n_tokens contiguous rows of llama_model_n_embd(ctx_tgt) floats.
bool common_speculative_process_with_pre_norm(common_speculative * spec, const llama_batch & batch, const float * h_pre_norm);

// diagnostic-only descendant probe for branch/sibling experiments.
// Returns false unless a concrete speculative implementation supports the probe.
bool common_speculative_probe_descendants(
        common_speculative * spec,
        llama_seq_id seq_id,
        uint16_t reject_depth,
        llama_pos sibling_pos,
        llama_token sampled,
        int n_max,
        llama_tokens & result,
        std::vector<std::vector<common_speculative_branch_candidate>> * branch_candidates);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target pre-norm embeddings to be extracted
bool common_speculative_need_embd_pre_norm(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
