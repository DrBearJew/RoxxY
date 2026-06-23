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
#include <cassert>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <cmath>
#include <cstdlib>

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
            common_speculative_env_enabled("LLAMA_MTP_DRAFT_BRANCH_CANDIDATES");
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

                h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, sample_idx);
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

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 9);

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
        } else if (has_draft_model_path && !has_mtp && !has_draft_eagle3) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
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
