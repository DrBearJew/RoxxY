#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

static bool llama_mtp_decode_prefix_verify_requested() {
    const char * env = getenv("LLAMA_MTP_DECODE_PREFIX_VERIFY");
    return env != nullptr && atoi(env) != 0;
}

static uint32_t ctx_type_to_embd_inp(const llama_hparams & hparams, llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return hparams.n_embd_inp();
        case LLAMA_CONTEXT_TYPE_MTP    : return hparams.n_embd_out();
    }
    throw std::runtime_error("Unsupported ctx type");
}

namespace {
struct src_mctx_reset_on_exit {
    llama_memory_context_ptr * slot;
    ~src_mctx_reset_on_exit() { if (slot) slot->reset(); }
};

static void llama_assert_gemma4_mtp_source_placement(
        const llama_context * ctx,
        const llama_context * src) {
    if (!ctx || !src) {
        return;
    }

    const auto & model_dft = ctx->get_model();
    const auto & model_tgt = src->get_model();

    if (model_dft.arch != LLM_ARCH_GEMMA4_ASSISTANT || model_tgt.arch != LLM_ARCH_GEMMA4) {
        return;
    }

    if (model_tgt.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        return;
    }

    const auto & hparams_dft = model_dft.hparams;
    const auto & hparams_tgt = model_tgt.hparams;

    const int32_t il_tgt_full = (int32_t) hparams_tgt.n_layer - 1;
    const int32_t il_tgt_swa  = (int32_t) hparams_tgt.n_layer - 2;

    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev_cpu) {
        throw std::runtime_error("Gemma 4 assistant MTP placement check failed: no CPU backend found");
    }

    const bool kv_offload = src->get_cparams().offload_kqv;

    for (uint32_t il_dft = 0; il_dft < hparams_dft.n_layer; ++il_dft) {
        const int32_t il_tgt = hparams_dft.is_swa(il_dft) ? il_tgt_swa : il_tgt_full;

        ggml_backend_dev_t dev_dft = model_dft.dev_layer(il_dft);
        ggml_backend_dev_t dev_kv  = kv_offload ? model_tgt.dev_layer(il_tgt) : dev_cpu;

        if (dev_dft != dev_kv) {
            throw std::runtime_error(format(
                    "Gemma 4 assistant MTP placement mismatch: draft layer %d is on %s, "
                    "but shared target KV layer %d is on %s",
                    (int) il_dft,
                    ggml_backend_dev_name(dev_dft),
                    (int) il_tgt,
                    ggml_backend_dev_name(dev_kv)));
        }
    }
}

struct llama_mtp_node_profile_entry {
    std::string name;
    const char * op = nullptr;
    int64_t time_us = 0;
    int64_t ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    int64_t elements = 0;
    int count = 0;
};

struct llama_mtp_node_profile_state {
    bool all = false;
    const char * graph = "mtp";
    int32_t n_tokens = 0;
    int64_t t0_us = 0;
    std::string pending_name;
    const char * pending_op = nullptr;
    int64_t pending_ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    int64_t pending_elements = 0;
    std::map<std::string, llama_mtp_node_profile_entry> entries;
};

static bool llama_mtp_node_profile_is_disabled(const char * env) {
    return env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0;
}

static bool llama_mtp_node_profile_want(const char * name, bool all) {
    if (all) {
        return true;
    }
    return strstr(name, "mtp_eh_proj")       ||
           strstr(name, "mtp_Qcur_full")     ||
           strstr(name, "mtp_Kcur_normed")   ||
           strstr(name, "mtp_Vcur")          ||
           strstr(name, "mtp_attn_pregate")  ||
           strstr(name, "mtp_attn_out")      ||
           strstr(name, "mtp_ffn_out")       ||
           strstr(name, "mtp_shared_head_norm") ||
           strstr(name, "prefix_Qcur_full")  ||
           strstr(name, "prefix_Kcur")       ||
           strstr(name, "prefix_Vcur")       ||
           strstr(name, "prefix_kqv_out")    ||
           strstr(name, "prefix_attn_output") ||
           strstr(name, "prefix_linear_attn_out") ||
           strstr(name, "prefix_conv_output_raw") ||
           strstr(name, "prefix_new_state")  ||
           strstr(name, "prefix_ffn_moe_out") ||
           strstr(name, "prefix_ffn_shexp")  ||
           strstr(name, "prefix_ffn_out")    ||
           strstr(name, "prefix_result_output_row") ||
           strstr(name, "result_output");
}

static bool llama_mtp_node_profile_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<llama_mtp_node_profile_state *>(user_data);
    if (st == nullptr || t == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr) {
        name = "";
    }

    if (ask) {
        if (!llama_mtp_node_profile_want(name, st->all)) {
            return false;
        }
        st->pending_name = name;
        st->pending_op   = ggml_op_name(t->op);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            st->pending_ne[i] = t->ne[i];
        }
        st->pending_elements = ggml_nelements(t);
        st->t0_us = ggml_time_us();
        return true;
    }

    const int64_t dt_us = std::max<int64_t>(0, ggml_time_us() - st->t0_us);
    const std::string key = st->pending_name.empty() ? std::string(name) : st->pending_name;
    auto & e = st->entries[key];
    if (e.count == 0) {
        e.name     = key;
        e.op       = st->pending_op ? st->pending_op : ggml_op_name(t->op);
        e.elements = st->pending_elements > 0 ? st->pending_elements : ggml_nelements(t);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            e.ne[i] = st->pending_ne[i] > 0 ? st->pending_ne[i] : t->ne[i];
        }
    }
    e.time_us += dt_us;
    e.count++;

    st->pending_name.clear();
    st->pending_op = nullptr;
    st->pending_elements = 0;
    return true;
}

static bool llama_mtp_node_profile_parse_layer_suffix(const std::string & name, std::string & family, int & layer) {
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) {
        family = name;
        layer = -1;
        return false;
    }
    for (size_t i = dash + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            family = name;
            layer = -1;
            return false;
        }
    }
    family = name.substr(0, dash);
    layer = std::atoi(name.c_str() + dash + 1);
    return true;
}

static std::string llama_mtp_node_profile_normalize_family(const std::string & name) {
    std::string family;
    int layer = -1;
    llama_mtp_node_profile_parse_layer_suffix(name, family, layer);

    auto normalize_after = [&](const char * marker) {
        size_t pos = 0;
        const size_t marker_len = std::strlen(marker);
        while ((pos = family.find(marker, pos)) != std::string::npos) {
            size_t i = pos + marker_len;
            if (i >= family.size() || family[i] < '0' || family[i] > '9') {
                pos = i;
                continue;
            }
            size_t j = i + 1;
            while (j < family.size() && family[j] >= '0' && family[j] <= '9') {
                ++j;
            }
            family.replace(i, j - i, "#");
            pos = i + 1;
        }
    };
    normalize_after("_row");
    normalize_after("_slot");
    return family;
}

static uint64_t llama_mtp_fnv1a64(const uint8_t * data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= (uint64_t) data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct llama_prefix_snapshot_trace_state {
    int layer = 0;
    int source_layer = 0;
    int candidate_layer = 0;
    int max_print = 0;
    int64_t n_tokens = 0;
    bool source_trace = false;
    bool candidate_trace = false;
    bool hidden_trace = false;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> pos;
    std::string pending_name;
    std::map<std::pair<char, int>, int64_t> source_rows;
};

static bool llama_prefix_snapshot_trace_parse(const char * name, char & kind, long long & row, long long & slot, int & layer) {
    if (sscanf(name, "prefix_conv_state_copy_row%lld_slot%lld-%d", &row, &slot, &layer) == 3) {
        kind = 'r';
        return true;
    }
    if (sscanf(name, "prefix_ssm_state_copy_row%lld_slot%lld-%d", &row, &slot, &layer) == 3) {
        kind = 's';
        return true;
    }
    return false;
}

static bool llama_prefix_state_candidate_trace_parse(const char * name, char & kind, long long & row, long long & slot, int & layer) {
    if (sscanf(name, "prefix_state_candidate_r_row%lld_slot%lld-%d", &row, &slot, &layer) == 3) {
        kind = 'r';
        return true;
    }
    if (sscanf(name, "prefix_state_candidate_s_row%lld_slot%lld-%d", &row, &slot, &layer) == 3) {
        kind = 's';
        return true;
    }
    return false;
}

static bool llama_prefix_state_source_trace_parse(const char * name, char & kind, int & layer) {
    int parsed_layer = -1;
    int n_read = 0;
    if (sscanf(name, "prefix_last_conv_states-%d%n", &parsed_layer, &n_read) == 1 && name[n_read] == '\0') {
        kind = 'r';
        layer = parsed_layer;
        return true;
    }
    if (sscanf(name, "prefix_new_state-%d%n", &parsed_layer, &n_read) == 1 && name[n_read] == '\0') {
        kind = 's';
        layer = parsed_layer;
        return true;
    }
    return false;
}

static bool llama_prefix_hidden_trace_parse_prefixed_row_layer(const char * name, const char * prefix, long long & row, int & layer) {
    const size_t prefix_len = strlen(prefix);
    if (strncmp(name, prefix, prefix_len) != 0) {
        return false;
    }
    const char * row_part = nullptr;
    const char * scan = name + prefix_len;
    while ((scan = strstr(scan, "_row")) != nullptr) {
        row_part = scan;
        scan += 4;
    }
    if (row_part == nullptr) {
        return false;
    }
    row_part += 4;
    char * end_row = nullptr;
    const long long parsed_row = strtoll(row_part, &end_row, 10);
    if (end_row == row_part || end_row == nullptr || *end_row != '-') {
        return false;
    }
    char * end_layer = nullptr;
    const long parsed_layer = strtol(end_row + 1, &end_layer, 10);
    if (end_layer == end_row + 1 || end_layer == nullptr || *end_layer != '\0') {
        return false;
    }
    row = parsed_row;
    layer = (int) parsed_layer;
    return true;
}

static bool llama_prefix_hidden_trace_parse(const char * name, long long & row, int & layer) {
    if (llama_prefix_hidden_trace_parse_prefixed_row_layer(name, "prefix41_hidden_", row, layer) ||
        llama_prefix_hidden_trace_parse_prefixed_row_layer(name, "prefix_roweq_hidden_", row, layer)) {
        return true;
    }
    if (sscanf(name, "prefix41_hidden_pre_next_state_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    if (sscanf(name, "prefix_roweq_hidden_pre_next_state_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    if (sscanf(name, "prefix_roweq_hidden_pre_next_state_serial_layer_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    if (sscanf(name, "prefix41_hidden_moe_out_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    if (sscanf(name, "prefix41_hidden_shexp_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    if (sscanf(name, "prefix41_hidden_ffn_out_row%lld-%d", &row, &layer) == 2) {
        return true;
    }
    return false;
}

static bool llama_trace_tensor_logical_bytes(ggml_tensor * t, std::vector<uint8_t> & out) {
    if (t == nullptr || t->buffer == nullptr) {
        return false;
    }
    const size_t elem_size = ggml_element_size(t);
    const size_t row_bytes = (size_t) t->ne[0] * elem_size;
    const size_t logical_bytes = (size_t) ggml_nelements(t) * elem_size;
    out.resize(logical_bytes);
    size_t dst = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const size_t off = (size_t) i3 * t->nb[3] + (size_t) i2 * t->nb[2] + (size_t) i1 * t->nb[1];
                ggml_backend_tensor_get(t, out.data() + dst, off, row_bytes);
                dst += row_bytes;
            }
        }
    }
    return true;
}

static bool llama_prefix_snapshot_trace_want(const llama_prefix_snapshot_trace_state * st, const char * name) {
    char kind = 0;
    long long row = -1;
    long long slot = -1;
    int layer = -1;
    if (llama_prefix_snapshot_trace_parse(name, kind, row, slot, layer)) {
        return st->layer < 0 || layer == st->layer;
    }
    if (st->source_trace && llama_prefix_state_source_trace_parse(name, kind, layer)) {
        return st->source_layer < 0 || layer == st->source_layer;
    }
    if (st->candidate_trace && llama_prefix_state_candidate_trace_parse(name, kind, row, slot, layer)) {
        return st->candidate_layer < 0 || layer == st->candidate_layer;
    }
    if (st->hidden_trace && llama_prefix_hidden_trace_parse(name, row, layer)) {
        return st->layer < 0 || layer == st->layer;
    }
    return false;
}

static bool llama_prefix_snapshot_trace_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<llama_prefix_snapshot_trace_state *>(user_data);
    if (st == nullptr || t == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr) {
        name = "";
    }

    if (ask) {
        if (!llama_prefix_snapshot_trace_want(st, name)) {
            return false;
        }
        st->pending_name = name;
        return true;
    }

    const std::string node_name = st->pending_name.empty() ? std::string(name) : st->pending_name;
    st->pending_name.clear();

    char kind = 0;
    long long row = -1;
    long long slot = -1;
    int layer = -1;
    bool is_snapshot = llama_prefix_snapshot_trace_parse(node_name.c_str(), kind, row, slot, layer);
    bool is_source = false;
    bool is_candidate = false;
    if (!is_snapshot) {
        is_candidate = llama_prefix_state_candidate_trace_parse(node_name.c_str(), kind, row, slot, layer);
    }
    bool is_hidden = false;
    if (!is_snapshot && !is_candidate) {
        is_source = llama_prefix_state_source_trace_parse(node_name.c_str(), kind, layer);
        if (!is_source) {
            is_hidden = llama_prefix_hidden_trace_parse(node_name.c_str(), row, layer);
            if (!is_hidden) {
                return true;
            }
            kind = 'h';
            slot = -1;
        } else {
            auto & next_row = st->source_rows[std::make_pair(kind, layer)];
            row = next_row++;
            slot = st->n_tokens > 0 ? st->n_tokens - 1 - row : -1;
        }
    }

    const llama_token tok = row >= 0 && (size_t) row < st->tokens.size() ? st->tokens[(size_t) row] : LLAMA_TOKEN_NULL;
    const llama_pos pos = row >= 0 && (size_t) row < st->pos.size() ? st->pos[(size_t) row] : -1;
    const char * trace_name = is_hidden ? "MTP_PREFIX_HIDDEN_TRACE" : (is_source ? "MTP_PREFIX_STATE_SOURCE_TRACE" : (is_candidate ? "MTP_PREFIX_STATE_CANDIDATE_TRACE" : "MTP_PREFIX_SNAPSHOT_TRACE"));

    if (t->buffer == nullptr) {
        fprintf(stderr,
                "%s: layer=%d kind=%c row=%lld slot=%lld token=%d pos=%d node=%s buffer=0\n",
                trace_name, layer, kind, row, slot, (int) tok, (int) pos, node_name.c_str());
        return true;
    }

    std::vector<uint8_t> tmp;
    if (!llama_trace_tensor_logical_bytes(t, tmp)) {
        return true;
    }
    const size_t nbytes = tmp.size();
    const uint64_t hash = llama_mtp_fnv1a64(tmp.data(), tmp.size());
    const size_t nfloat = t->type == GGML_TYPE_F32 ? nbytes / sizeof(float) : 0;

    if (is_hidden && t->type == GGML_TYPE_F32 && getenv("LLAMA_MTP_PREFIX_HIDDEN_TRACE_COMPARE")) {
        auto layer64_stage = [](const std::string & name, const char * prefix, std::string & stage) -> bool {
            const size_t begin = name.find(prefix);
            if (begin == std::string::npos) {
                return false;
            }
            const size_t stage_begin = begin + strlen(prefix);
            const size_t stage_end = name.find("_row", stage_begin);
            if (stage_end == std::string::npos || stage_end == stage_begin) {
                return false;
            }
            stage = name.substr(stage_begin, stage_end - stage_begin);
            return true;
        };
        std::string ref_stage;
        std::string pdmq_stage;
        const bool is_layer64_ref  = layer64_stage(node_name, "layer64_ref_",  ref_stage);
        const bool is_layer64_pdmq = layer64_stage(node_name, "layer64_pdmq_", pdmq_stage);
        if ((is_layer64_ref || is_layer64_pdmq) && nfloat > 0) {
            static std::map<std::string, std::vector<float>> refs;
            const std::string & stage = is_layer64_ref ? ref_stage : pdmq_stage;
            const std::string key = std::string("layer64_") + stage + ":row=" + std::to_string(row) +
                ":token=" + std::to_string((long long) tok) +
                ":pos=" + std::to_string((long long) pos);
            const float * vals = reinterpret_cast<const float *>(tmp.data());
            std::vector<float> cur(vals, vals + nfloat);
            if (is_layer64_ref) {
                refs[key] = std::move(cur);
            } else {
                auto it = refs.find(key);
                if (it != refs.end()) {
                    if (it->second.size() != cur.size()) {
                        fprintf(stderr,
                                "MTP_PREFIX_HIDDEN_COMPARE: key=%s ref_n=%zu cur_n=%zu size_mismatch=1\n",
                                key.c_str(), it->second.size(), cur.size());
                    } else {
                        double sum_abs = 0.0;
                        double sumsq = 0.0;
                        float max_abs = 0.0f;
                        size_t max_i = 0;
                        for (size_t i = 0; i < cur.size(); ++i) {
                            const float d = cur[i] - it->second[i];
                            const float ad = std::fabs(d);
                            sum_abs += (double) ad;
                            sumsq += (double) d * (double) d;
                            if (ad > max_abs) {
                                max_abs = ad;
                                max_i = i;
                            }
                        }
                        const double mean_abs = sum_abs / (double) cur.size();
                        const double rms = std::sqrt(sumsq / (double) cur.size());
                        fprintf(stderr,
                                "MTP_PREFIX_HIDDEN_COMPARE: key=%s n_float=%zu max_abs=%.9g mean_abs=%.9g rms=%.9g max_i=%zu ref=%.9g cur=%.9g\n",
                                key.c_str(), cur.size(), max_abs, mean_abs, rms, max_i, it->second[max_i], cur[max_i]);
                    }
                }
            }
        }
    }

    fprintf(stderr,
            "%s: layer=%d kind=%c row=%lld slot=%lld token=%d pos=%d node=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] n_bytes=%zu n_float=%zu hash=%016" PRIx64 " first=[",
            trace_name, layer, kind, row, slot, (int) tok, (int) pos, node_name.c_str(), ggml_op_name(t->op), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            nbytes, nfloat, hash);
    if (t->type == GGML_TYPE_F32 && st->max_print > 0) {
        const float * vals = reinterpret_cast<const float *>(tmp.data());
        const size_t n_print = std::min<size_t>((size_t) st->max_print, nfloat);
        for (size_t i = 0; i < n_print; ++i) {
            fprintf(stderr, "%s%.9g", i == 0 ? "" : ",", vals[i]);
        }
    }
    fprintf(stderr, "]\n");
    return true;
}

struct llama_gdn_input_trace_state {
    int layer = 0;
    int max_print = 8;
    int max_n_tokens = 8;
    bool compare = true;
    int64_t gdn_sv = 0;
    int64_t n_tokens = 0;
    int64_t n_seq_tokens = 0;
    int64_t n_seqs = 0;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> pos;
    std::string pending_name;
};

static bool llama_gdn_input_trace_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0' && atoi(env) != 0;
}

static const char * llama_gdn_input_trace_dump_dir() {
    const char * dir = getenv("LLAMA_MTP_GDN_INPUT_TRACE_DUMP_DIR");
    return (dir != nullptr && dir[0] != '\0') ? dir : nullptr;
}

static bool llama_gdn_input_trace_dump_filter_matches(const std::string & node_name) {
    const char * filter = getenv("LLAMA_MTP_GDN_INPUT_TRACE_DUMP_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }

    const char * begin = filter;
    while (*begin != '\0') {
        while (*begin == ',' || *begin == ' ' || *begin == '\t' || *begin == '\n') {
            ++begin;
        }
        const char * end = begin;
        while (*end != '\0' && *end != ',') {
            ++end;
        }
        const size_t len = (size_t) (end - begin);
        if (len > 0 && node_name.find(std::string(begin, len)) != std::string::npos) {
            return true;
        }
        begin = end;
    }

    return false;
}

static std::string llama_gdn_input_trace_sanitize_file_component(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        out.push_back(ok ? c : '_');
    }
    return out.empty() ? std::string("node") : out;
}

static bool llama_gdn_input_trace_mkdir_p(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }

    std::string cur;
    cur.reserve(dir.size());
    for (size_t i = 0; i < dir.size(); ++i) {
        cur.push_back(dir[i]);
        if (dir[i] != '/' || cur.size() == 1) {
            continue;
        }
        if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

static void llama_gdn_input_trace_dump_bytes(
        const llama_gdn_input_trace_state * st,
        const std::string & node_name,
        const ggml_tensor * t,
        const char * suffix,
        const char * payload_type,
        const void * data,
        size_t nbytes,
        int64_t token_index = -1,
        llama_token token = LLAMA_TOKEN_NULL,
        llama_pos pos = -1) {
    const char * dir = llama_gdn_input_trace_dump_dir();
    if (dir == nullptr || st == nullptr || t == nullptr || data == nullptr || nbytes == 0) {
        return;
    }
    if (!llama_gdn_input_trace_dump_filter_matches(node_name)) {
        return;
    }

    const int64_t max_bytes = []() -> int64_t {
        const char * env = getenv("LLAMA_MTP_GDN_INPUT_TRACE_DUMP_MAX_BYTES");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        char * end = nullptr;
        const long long v = std::strtoll(env, &end, 10);
        return end != env ? (int64_t) v : 0;
    }();
    if (max_bytes > 0 && nbytes > (size_t) max_bytes) {
        return;
    }

    if (!llama_gdn_input_trace_mkdir_p(dir)) {
        fprintf(stderr, "MTP_GDN_DUMP: mkdir_failed dir=%s errno=%d\n", dir, errno);
        return;
    }

    static uint64_t dump_seq = 0;
    const uint64_t seq = dump_seq++;

    char prefix[512];
    snprintf(prefix, sizeof(prefix),
             "%s/%06" PRIu64 "_layer%d_%s",
             dir, seq, st->layer, llama_gdn_input_trace_sanitize_file_component(node_name).c_str());
    std::string base = prefix;
    if (suffix != nullptr && suffix[0] != '\0') {
        base += "_";
        base += llama_gdn_input_trace_sanitize_file_component(suffix);
    }

    const std::string bin_path  = base + ".bin";
    const std::string meta_path = base + ".meta";

    {
        std::ofstream out(bin_path, std::ios::binary);
        if (!out) {
            fprintf(stderr, "MTP_GDN_DUMP: write_failed path=%s\n", bin_path.c_str());
            return;
        }
        out.write(reinterpret_cast<const char *>(data), (std::streamsize) nbytes);
    }

    {
        std::ofstream meta(meta_path);
        if (!meta) {
            fprintf(stderr, "MTP_GDN_DUMP: meta_write_failed path=%s\n", meta_path.c_str());
            return;
        }
        meta << "node=" << node_name << "\n";
        meta << "layer=" << st->layer << "\n";
        meta << "op=" << ggml_op_name(t->op) << "\n";
        meta << "ggml_type=" << ggml_type_name(t->type) << "\n";
        meta << "payload_type=" << (payload_type ? payload_type : "raw") << "\n";
        meta << "nbytes=" << nbytes << "\n";
        meta << "ne=" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "\n";
        meta << "nb=" << t->nb[0] << "," << t->nb[1] << "," << t->nb[2] << "," << t->nb[3] << "\n";
        meta << "trace_n_tokens=" << st->n_tokens << "\n";
        meta << "trace_n_seq_tokens=" << st->n_seq_tokens << "\n";
        meta << "trace_n_seqs=" << st->n_seqs << "\n";
        meta << "token_index=" << token_index << "\n";
        meta << "token=" << (int) token << "\n";
        meta << "pos=" << (int) pos << "\n";
        meta << "bin=" << bin_path << "\n";
    }

    fprintf(stderr, "MTP_GDN_DUMP: node=%s suffix=%s nbytes=%zu bin=%s meta=%s\n",
            node_name.c_str(), suffix ? suffix : "", nbytes, bin_path.c_str(), meta_path.c_str());
}

static int llama_env_i32(const char * name, int def) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long v = std::strtol(env, &end, 10);
    return end != env ? (int) v : def;
}

static bool llama_gdn_input_trace_starts_layer_name(const char * name, const char * base) {
    const size_t n = strlen(base);
    return strncmp(name, base, n) == 0 && name[n] == '-';
}

static bool llama_gdn_input_trace_is_gdn_gate(const char * name) {
    return llama_gdn_input_trace_starts_layer_name(name, "gate");
}

static bool llama_gdn_input_trace_is_extra_per_token(const char * name) {
    return llama_gdn_input_trace_starts_layer_name(name, "z") ||
           llama_gdn_input_trace_starts_layer_name(name, "linear_attn_qkv_mixed") ||
           llama_gdn_input_trace_starts_layer_name(name, "linear_attn_out") ||
           llama_gdn_input_trace_starts_layer_name(name, "attn_residual") ||
           llama_gdn_input_trace_starts_layer_name(name, "attn_post_norm") ||
           llama_gdn_input_trace_starts_layer_name(name, "Qcur_full") ||
           llama_gdn_input_trace_starts_layer_name(name, "Qcur_reshaped") ||
           llama_gdn_input_trace_starts_layer_name(name, "Qcur_normed") ||
           llama_gdn_input_trace_starts_layer_name(name, "Qcur") ||
           llama_gdn_input_trace_starts_layer_name(name, "Kcur_normed") ||
           llama_gdn_input_trace_starts_layer_name(name, "Kcur") ||
           llama_gdn_input_trace_starts_layer_name(name, "Vcur") ||
           llama_gdn_input_trace_starts_layer_name(name, "gate_reshaped") ||
           llama_gdn_input_trace_starts_layer_name(name, "gate_sigmoid") ||
           llama_gdn_input_trace_starts_layer_name(name, "attn_pregate") ||
           llama_gdn_input_trace_starts_layer_name(name, "attn_gated") ||
           llama_gdn_input_trace_starts_layer_name(name, "attn_output") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_up") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_gate") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_swiglu") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_out") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_logits") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_probs") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_topk") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_weights") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_gate_up") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_gate") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_up") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_swiglu") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_down") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_weighted") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_moe_out") ||
           llama_gdn_input_trace_starts_layer_name(name, "shared_expert_gate") ||
           llama_gdn_input_trace_starts_layer_name(name, "shared_expert_gate_sigmoid") ||
           llama_gdn_input_trace_starts_layer_name(name, "shared_expert_out") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_shexp") ||
           llama_gdn_input_trace_starts_layer_name(name, "ffn_shexp_gated") ||
           llama_gdn_input_trace_starts_layer_name(name, "post_moe") ||
           llama_gdn_input_trace_starts_layer_name(name, "post_ffn") ||
           llama_gdn_input_trace_starts_layer_name(name, "l_out");
}

static bool llama_gdn_input_trace_parse_state_copy(const char * name, int & k_i, int & k_total) {
    return sscanf(name, "ssm_state_copy_k%d_of_%d-", &k_i, &k_total) == 2;
}

static bool llama_gdn_input_trace_parse_conv_state_copy(const char * name, int & k_i, int & k_total) {
    return sscanf(name, "conv_state_copy_k%d_of_%d-", &k_i, &k_total) == 2;
}

static int llama_gdn_input_trace_token_dim(const char * name) {
    if (llama_gdn_input_trace_is_gdn_gate(name)) {
        return 1;
    }
    if (strstr(name, "q_conv_predelta") || strstr(name, "k_conv_predelta") ||
        strstr(name, "v_conv_predelta") || strstr(name, "beta_sigmoid")) {
        return 2;
    }
    return -1;
}

static void llama_gdn_input_trace_compare_slice(
        const llama_gdn_input_trace_state * st,
        const char * kind,
        const char * node_name,
        llama_token token,
        llama_pos pos,
        const std::vector<float> & logical) {
    if (!st || !st->compare || logical.empty()) {
        return;
    }

    static std::map<std::string, std::vector<float>> refs;
    std::string key;
    std::string scope_label = "-";
    if (const char * scope = getenv("LLAMA_MTP_GDN_INPUT_TRACE_COMPARE_SCOPE")) {
        scope_label = scope;
        key += "scope=";
        key += scope;
        key += ":";
    }
    key += std::string(kind) + ":layer=" + std::to_string(st->layer) +
        ":tok=" + std::to_string((long long) token) + ":pos=" + std::to_string((long long) pos);
    if (strcmp(kind, "input") == 0) {
        key += ":node=";
        key += node_name;
    }
    auto it = refs.find(key);
    if (it == refs.end()) {
        refs.emplace(key, logical);
        return;
    }

    if (it->second.size() != logical.size()) {
        fprintf(stderr,
                "MTP_GDN_COMPARE: kind=%s layer=%d node=%s token=%d pos=%d scope=%s ref_n_float=%zu cur_n_float=%zu size_mismatch=1\n",
                kind, st->layer, node_name, (int) token, (int) pos, scope_label.c_str(), it->second.size(), logical.size());
        return;
    }

    double sumsq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    size_t max_i = 0;
    for (size_t i = 0; i < logical.size(); ++i) {
        const float d = logical[i] - it->second[i];
        const float ad = std::fabs(d);
        sum_abs += (double) ad;
        sumsq += (double) d * (double) d;
        if (ad > max_abs) {
            max_abs = ad;
            max_i = i;
        }
    }
    const double rms = std::sqrt(sumsq / (double) logical.size());
    const double mean_abs = sum_abs / (double) logical.size();
    fprintf(stderr,
            "MTP_GDN_COMPARE: kind=%s layer=%d node=%s token=%d pos=%d scope=%s n_float=%zu max_abs=%.9g mean_abs=%.9g rms=%.9g max_i=%zu ref=%.9g cur=%.9g\n",
            kind, st->layer, node_name, (int) token, (int) pos, scope_label.c_str(), logical.size(),
            max_abs, mean_abs, rms, max_i, it->second[max_i], logical[max_i]);
}

static bool llama_gdn_input_trace_want(const char * name, int layer) {
    char suffix[32];
    snprintf(suffix, sizeof(suffix), "-%d", layer);
    const size_t name_len = strlen(name);
    const size_t suffix_len = strlen(suffix);
    if (name_len < suffix_len || strcmp(name + name_len - suffix_len, suffix) != 0) {
        return false;
    }

    return strstr(name, "q_conv_predelta") ||
           strstr(name, "k_conv_predelta") ||
           strstr(name, "v_conv_predelta") ||
           llama_gdn_input_trace_is_gdn_gate(name) ||
           strstr(name, "beta_sigmoid")     ||
           strstr(name, "state_predelta")   ||
           strstr(name, "ssm_state_copy")   ||
           strstr(name, "conv_state_copy")  ||
           llama_gdn_input_trace_is_extra_per_token(name) ||
           strstr(name, LLAMA_TENSOR_NAME_FGDN_CH) ||
           strstr(name, LLAMA_TENSOR_NAME_FGDN_AR);
}

static bool llama_gdn_input_trace_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<llama_gdn_input_trace_state *>(user_data);
    if (st == nullptr || t == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr) {
        name = "";
    }

    if (ask) {
        if (st->n_tokens > st->max_n_tokens || !llama_gdn_input_trace_want(name, st->layer)) {
            return false;
        }
        st->pending_name = name;
        return true;
    }

    const std::string node_name = st->pending_name.empty() ? std::string(name) : st->pending_name;
    st->pending_name.clear();

    fprintf(stderr,
            "MTP_GDN_INPUT_TRACE: layer=%d n_tokens=%lld n_seq_tokens=%lld n_seqs=%lld node=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] contiguous=%d rows_contiguous=%d tokens=[",
            st->layer,
            (long long) st->n_tokens, (long long) st->n_seq_tokens, (long long) st->n_seqs,
            node_name.c_str(), ggml_op_name(t->op), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            ggml_is_contiguous(t) ? 1 : 0, ggml_is_contiguous_rows(t) ? 1 : 0);
    for (size_t i = 0; i < st->tokens.size(); ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int) st->tokens[i]);
    }
    fprintf(stderr, "] pos=[");
    for (size_t i = 0; i < st->pos.size(); ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int) st->pos[i]);
    }
    fprintf(stderr, "]");

    if (t->buffer == nullptr) {
        fprintf(stderr, " buffer=0\n");
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, " non_f32=1\n");
        if (llama_gdn_input_trace_dump_dir() != nullptr) {
            const size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> raw(nbytes);
            ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
            llama_gdn_input_trace_dump_bytes(st, node_name, t, "raw", ggml_type_name(t->type), raw.data(), raw.size());
        }
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const size_t nfloat = nbytes / sizeof(float);
    std::vector<float> tmp(nfloat);
    ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
    llama_gdn_input_trace_dump_bytes(st, node_name, t, "raw", "f32_raw", tmp.data(), nbytes);

    const uint64_t hash = llama_mtp_fnv1a64(reinterpret_cast<const uint8_t *>(tmp.data()), nbytes);
    float mn = INFINITY;
    float mx = -INFINITY;
    int bad = 0;
    for (float v : tmp) {
        if (!std::isfinite(v)) {
            ++bad;
            continue;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }

    fprintf(stderr,
            " n_float=%zu hash=%016" PRIx64 " bad=%d min=%.9g max=%.9g first=[",
            nfloat, hash, bad, mn, mx);
    const size_t n_print = std::min<size_t>((size_t) std::max(0, st->max_print), nfloat);
    for (size_t i = 0; i < n_print; ++i) {
        fprintf(stderr, "%s%.9g", i == 0 ? "" : ",", tmp[i]);
    }
    fprintf(stderr, "]\n");

    int state_copy_k = -1;
    int state_copy_k_total = -1;
    if (llama_gdn_input_trace_parse_state_copy(node_name.c_str(), state_copy_k, state_copy_k_total) && state_copy_k_total > 0) {
        const int64_t ti = (int64_t) state_copy_k + st->n_tokens - (int64_t) state_copy_k_total;
        if (ti >= 0 && ti < st->n_tokens) {
            const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
            const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
            llama_gdn_input_trace_compare_slice(st, "state_copy", node_name.c_str(), tok, pos, tmp);
        }
    }

    int conv_state_copy_k = -1;
    int conv_state_copy_k_total = -1;
    if (llama_gdn_input_trace_parse_conv_state_copy(node_name.c_str(), conv_state_copy_k, conv_state_copy_k_total) && conv_state_copy_k_total > 0) {
        const int64_t ti = (int64_t) conv_state_copy_k + st->n_tokens - (int64_t) conv_state_copy_k_total;
        if (ti >= 0 && ti < st->n_tokens) {
            const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
            const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
            llama_gdn_input_trace_compare_slice(st, "conv_state_copy", node_name.c_str(), tok, pos, tmp);
        }
    }

    int token_dim = llama_gdn_input_trace_token_dim(node_name.c_str());
    if (token_dim < 0 && llama_gdn_input_trace_is_extra_per_token(node_name.c_str())) {
        if (t->ne[1] == st->n_tokens) {
            token_dim = 1;
        } else if (t->ne[2] == st->n_tokens) {
            token_dim = 2;
        }
    }
    if (token_dim >= 0 && token_dim < GGML_MAX_DIMS && t->ne[token_dim] == st->n_tokens) {
        const size_t row_bytes = (size_t) t->ne[0] * sizeof(float);
        for (int64_t ti = 0; ti < st->n_tokens; ++ti) {
            std::vector<float> logical;
            if (token_dim == 2) {
                logical.reserve((size_t) t->ne[0] * (size_t) t->ne[1] * (size_t) t->ne[3]);
                std::vector<float> row((size_t) t->ne[0]);
                for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        const size_t off = (size_t) i3 * t->nb[3] + (size_t) ti * t->nb[2] + (size_t) i1 * t->nb[1];
                        ggml_backend_tensor_get(t, row.data(), off, row_bytes);
                        logical.insert(logical.end(), row.begin(), row.end());
                    }
                }
            } else if (token_dim == 1) {
                logical.reserve((size_t) t->ne[0] * (size_t) t->ne[2] * (size_t) t->ne[3]);
                std::vector<float> row((size_t) t->ne[0]);
                for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                        const size_t off = (size_t) i3 * t->nb[3] + (size_t) i2 * t->nb[2] + (size_t) ti * t->nb[1];
                        ggml_backend_tensor_get(t, row.data(), off, row_bytes);
                        logical.insert(logical.end(), row.begin(), row.end());
                    }
                }
            }
            if (logical.empty()) {
                continue;
            }

            const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
            const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
            char dump_suffix[96];
            snprintf(dump_suffix, sizeof(dump_suffix), "tok%lld_token%d_pos%d_f32", (long long) ti, (int) tok, (int) pos);
            llama_gdn_input_trace_dump_bytes(st, node_name, t, dump_suffix, "f32_logical_token", logical.data(), logical.size() * sizeof(float), ti, tok, pos);

            const uint64_t tok_hash = llama_mtp_fnv1a64(reinterpret_cast<const uint8_t *>(logical.data()), logical.size() * sizeof(float));
            float tok_mn = INFINITY;
            float tok_mx = -INFINITY;
            int tok_bad = 0;
            for (float v : logical) {
                if (!std::isfinite(v)) {
                    ++tok_bad;
                    continue;
                }
                tok_mn = std::min(tok_mn, v);
                tok_mx = std::max(tok_mx, v);
            }
            fprintf(stderr,
                    "MTP_GDN_TOKEN_TRACE: layer=%d node=%s token_index=%lld token=%d pos=%d n_float=%zu hash=%016" PRIx64 " bad=%d min=%.9g max=%.9g first=[",
                    st->layer, node_name.c_str(), (long long) ti, (int) tok, (int) pos,
                    logical.size(), tok_hash, tok_bad, tok_mn, tok_mx);
            const size_t n_tok_print = std::min<size_t>((size_t) std::max(0, st->max_print), logical.size());
            for (size_t i = 0; i < n_tok_print; ++i) {
                fprintf(stderr, "%s%.9g", i == 0 ? "" : ",", logical[i]);
            }
            fprintf(stderr, "]\n");

            llama_gdn_input_trace_compare_slice(st, "input", node_name.c_str(), tok, pos, logical);
        }
    }

    if (llama_gdn_input_trace_starts_layer_name(node_name.c_str(), "ffn_moe_weighted") &&
            t->src[1] != nullptr && t->src[1]->buffer != nullptr && t->src[1]->type == GGML_TYPE_F32) {
        const ggml_tensor * src_weights = t->src[1];
        std::string weights_node_name = node_name;
        const size_t name_pos = weights_node_name.find("ffn_moe_weighted");
        if (name_pos != std::string::npos) {
            weights_node_name.replace(name_pos, strlen("ffn_moe_weighted"), "ffn_moe_weights_final_src");
        } else {
            weights_node_name += "_weights_final_src";
        }

        const size_t src_nbytes = ggml_nbytes(src_weights);
        std::vector<float> src_tmp(src_nbytes / sizeof(float));
        ggml_backend_tensor_get(src_weights, src_tmp.data(), 0, src_nbytes);
        llama_gdn_input_trace_dump_bytes(st, weights_node_name, src_weights, "raw", "f32_raw", src_tmp.data(), src_nbytes);

        int src_token_dim = -1;
        if (src_weights->ne[1] == st->n_tokens) {
            src_token_dim = 1;
        } else if (src_weights->ne[2] == st->n_tokens) {
            src_token_dim = 2;
        }
        if (src_token_dim >= 0) {
            const size_t row_bytes = (size_t) src_weights->ne[0] * sizeof(float);
            for (int64_t ti = 0; ti < st->n_tokens; ++ti) {
                std::vector<float> logical;
                if (src_token_dim == 2) {
                    logical.reserve((size_t) src_weights->ne[0] * (size_t) src_weights->ne[1] * (size_t) src_weights->ne[3]);
                    std::vector<float> row((size_t) src_weights->ne[0]);
                    for (int64_t i3 = 0; i3 < src_weights->ne[3]; ++i3) {
                        for (int64_t i1 = 0; i1 < src_weights->ne[1]; ++i1) {
                            const size_t off = (size_t) i3 * src_weights->nb[3] + (size_t) ti * src_weights->nb[2] + (size_t) i1 * src_weights->nb[1];
                            ggml_backend_tensor_get(src_weights, row.data(), off, row_bytes);
                            logical.insert(logical.end(), row.begin(), row.end());
                        }
                    }
                } else if (src_token_dim == 1) {
                    logical.reserve((size_t) src_weights->ne[0] * (size_t) src_weights->ne[2] * (size_t) src_weights->ne[3]);
                    std::vector<float> row((size_t) src_weights->ne[0]);
                    for (int64_t i3 = 0; i3 < src_weights->ne[3]; ++i3) {
                        for (int64_t i2 = 0; i2 < src_weights->ne[2]; ++i2) {
                            const size_t off = (size_t) i3 * src_weights->nb[3] + (size_t) i2 * src_weights->nb[2] + (size_t) ti * src_weights->nb[1];
                            ggml_backend_tensor_get(src_weights, row.data(), off, row_bytes);
                            logical.insert(logical.end(), row.begin(), row.end());
                        }
                    }
                }
                if (logical.empty()) {
                    continue;
                }

                const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
                const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
                char dump_suffix[96];
                snprintf(dump_suffix, sizeof(dump_suffix), "tok%lld_token%d_pos%d_f32", (long long) ti, (int) tok, (int) pos);
                llama_gdn_input_trace_dump_bytes(st, weights_node_name, src_weights, dump_suffix, "f32_logical_token", logical.data(), logical.size() * sizeof(float), ti, tok, pos);
            }
        }
    }

    if (st->gdn_sv > 0 &&
            (strstr(node_name.c_str(), LLAMA_TENSOR_NAME_FGDN_CH) || strstr(node_name.c_str(), LLAMA_TENSOR_NAME_FGDN_AR)) &&
            t->ne[0] % st->gdn_sv == 0 && t->ne[1] > st->n_tokens) {
        const int64_t S_v = st->gdn_sv;
        const int64_t H   = t->ne[0] / S_v;
        const int64_t state_rows = t->ne[1] - st->n_tokens;
        if (H > 0 && state_rows >= S_v && state_rows % S_v == 0) {
            const int64_t K = state_rows / S_v;
            const int64_t attn_n_float = S_v * H;
            const int64_t state_n_float = S_v * S_v * H * t->ne[2] * t->ne[3];
            const int64_t attn_score_elems = attn_n_float * st->n_tokens * t->ne[2] * t->ne[3];
            const size_t n_print_result = (size_t) std::max(0, st->max_print);

            for (int64_t ti = 0; ti < st->n_tokens; ++ti) {
                std::vector<float> logical((size_t) attn_n_float);
                ggml_backend_tensor_get(t, logical.data(), (size_t) ti * attn_n_float * sizeof(float), logical.size() * sizeof(float));
                const uint64_t hash = llama_mtp_fnv1a64(reinterpret_cast<const uint8_t *>(logical.data()), logical.size() * sizeof(float));
                const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
                const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
                fprintf(stderr,
                        "MTP_GDN_RESULT_TRACE: kind=attn layer=%d node=%s token_index=%lld token=%d pos=%d n_float=%zu hash=%016" PRIx64 " first=[",
                        st->layer, node_name.c_str(), (long long) ti, (int) tok, (int) pos, logical.size(), hash);
                for (size_t i = 0; i < std::min(n_print_result, logical.size()); ++i) {
                    fprintf(stderr, "%s%.9g", i == 0 ? "" : ",", logical[i]);
                }
                fprintf(stderr, "]\n");
                llama_gdn_input_trace_compare_slice(st, "attn", node_name.c_str(), tok, pos, logical);
            }

            const int64_t shift = st->n_tokens - K;
            for (int64_t slot = 0; slot < K; ++slot) {
                const int64_t ti = slot + shift;
                if (ti < 0 || ti >= st->n_tokens) {
                    continue;
                }
                std::vector<float> logical((size_t) state_n_float);
                const size_t off = ((size_t) attn_score_elems + (size_t) slot * (size_t) state_n_float) * sizeof(float);
                ggml_backend_tensor_get(t, logical.data(), off, logical.size() * sizeof(float));
                const uint64_t hash = llama_mtp_fnv1a64(reinterpret_cast<const uint8_t *>(logical.data()), logical.size() * sizeof(float));
                const llama_token tok = (size_t) ti < st->tokens.size() ? st->tokens[(size_t) ti] : LLAMA_TOKEN_NULL;
                const llama_pos pos = (size_t) ti < st->pos.size() ? st->pos[(size_t) ti] : -1;
                fprintf(stderr,
                        "MTP_GDN_RESULT_TRACE: kind=state layer=%d node=%s slot=%lld token_index=%lld token=%d pos=%d n_float=%zu hash=%016" PRIx64 " first=[",
                        st->layer, node_name.c_str(), (long long) slot, (long long) ti, (int) tok, (int) pos, logical.size(), hash);
                for (size_t i = 0; i < std::min(n_print_result, logical.size()); ++i) {
                    fprintf(stderr, "%s%.9g", i == 0 ? "" : ",", logical[i]);
                }
                fprintf(stderr, "]\n");
                llama_gdn_input_trace_compare_slice(st, "state", node_name.c_str(), tok, pos, logical);
            }
        }
    }
    return true;
}

static void llama_mtp_node_profile_print(const llama_mtp_node_profile_state & st) {
    std::vector<llama_mtp_node_profile_entry> entries;
    entries.reserve(st.entries.size());
    int64_t total_us = 0;
    int count = 0;
    for (const auto & kv : st.entries) {
        entries.push_back(kv.second);
        total_us += kv.second.time_us;
        count    += kv.second.count;
    }
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
        return a.time_us > b.time_us;
    });

    struct profile_aggregate {
        int64_t time_us = 0;
        int calls = 0;
        int nodes = 0;
    };
    std::map<std::string, profile_aggregate> op_aggs;
    std::map<std::string, profile_aggregate> family_aggs;
    std::map<int, profile_aggregate> layer_aggs;
    for (const auto & e : entries) {
        auto & op_agg = op_aggs[e.op ? e.op : "?"];
        op_agg.time_us += e.time_us;
        op_agg.calls   += e.count;
        op_agg.nodes++;

        const std::string family_name = llama_mtp_node_profile_normalize_family(e.name);
        auto & family_agg = family_aggs[family_name.empty() ? std::string("?") : family_name];
        family_agg.time_us += e.time_us;
        family_agg.calls   += e.count;
        family_agg.nodes++;

        std::string family_unused;
        int layer = -1;
        if (llama_mtp_node_profile_parse_layer_suffix(e.name, family_unused, layer) && layer >= 0) {
            auto & layer_agg = layer_aggs[layer];
            layer_agg.time_us += e.time_us;
            layer_agg.calls   += e.count;
            layer_agg.nodes++;
        }
    }
    std::vector<std::pair<std::string, profile_aggregate>> op_entries(op_aggs.begin(), op_aggs.end());
    std::sort(op_entries.begin(), op_entries.end(), [](const auto & a, const auto & b) {
        return a.second.time_us > b.second.time_us;
    });
    std::vector<std::pair<std::string, profile_aggregate>> family_entries(family_aggs.begin(), family_aggs.end());
    std::sort(family_entries.begin(), family_entries.end(), [](const auto & a, const auto & b) {
        return a.second.time_us > b.second.time_us;
    });
    std::vector<std::pair<int, profile_aggregate>> layer_entries(layer_aggs.begin(), layer_aggs.end());
    std::sort(layer_entries.begin(), layer_entries.end(), [](const auto & a, const auto & b) {
        return a.second.time_us > b.second.time_us;
    });

    int limit = st.all ? 24 : 32;
    if (const char * env = getenv("LLAMA_MTP_NODE_PROFILE_LIMIT")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end != env && v > 0) {
            limit = (int) v;
        }
    }

    fprintf(stderr,
            "MTP_NODE_PROFILE: graph=%s mode=%s n_tokens=%d observed_nodes=%zu observed_calls=%d observed_total=%.3f_ms%s\n",
            st.graph ? st.graph : "?",
            st.all ? "all" : "targets",
            (int) st.n_tokens,
            entries.size(),
            count,
            (double) total_us / 1000.0,
            st.all ? "" : " note=stage-timing-use-LLAMA_MTP_NODE_PROFILE=all-for-per-node");

    const int n_op_print = std::min<int>(limit, (int) op_entries.size());
    for (int i = 0; i < n_op_print; ++i) {
        const auto & [op, agg] = op_entries[i];
        fprintf(stderr,
                "MTP_NODE_PROFILE_OP: graph=%s rank=%02d time=%.3f_ms nodes=%d calls=%d avg_call=%.3f_ms op=%s\n",
                st.graph ? st.graph : "?",
                i + 1,
                (double) agg.time_us / 1000.0,
                agg.nodes,
                agg.calls,
                agg.calls > 0 ? (double) agg.time_us / (1000.0 * agg.calls) : 0.0,
                op.c_str());
    }

    const int n_family_print = std::min<int>(limit, (int) family_entries.size());
    for (int i = 0; i < n_family_print; ++i) {
        const auto & [family, agg] = family_entries[i];
        fprintf(stderr,
                "MTP_NODE_PROFILE_FAMILY: graph=%s rank=%02d time=%.3f_ms nodes=%d calls=%d avg_call=%.3f_ms family=%s\n",
                st.graph ? st.graph : "?",
                i + 1,
                (double) agg.time_us / 1000.0,
                agg.nodes,
                agg.calls,
                agg.calls > 0 ? (double) agg.time_us / (1000.0 * agg.calls) : 0.0,
                family.c_str());
    }

    const int n_layer_print = std::min<int>(limit, (int) layer_entries.size());
    for (int i = 0; i < n_layer_print; ++i) {
        const auto & [layer, agg] = layer_entries[i];
        fprintf(stderr,
                "MTP_NODE_PROFILE_LAYER: graph=%s rank=%02d time=%.3f_ms nodes=%d calls=%d avg_call=%.3f_ms layer=%d\n",
                st.graph ? st.graph : "?",
                i + 1,
                (double) agg.time_us / 1000.0,
                agg.nodes,
                agg.calls,
                agg.calls > 0 ? (double) agg.time_us / (1000.0 * agg.calls) : 0.0,
                layer);
    }

    const int n_print = std::min<int>(limit, (int) entries.size());
    for (int i = 0; i < n_print; ++i) {
        const auto & e = entries[i];
        fprintf(stderr,
                "MTP_NODE_PROFILE: rank=%02d time=%.3f_ms count=%d avg=%.3f_ms op=%s ne=[%lld,%lld,%lld,%lld] elems=%lld name=%s\n",
                i + 1,
                (double) e.time_us / 1000.0,
                e.count,
                e.count > 0 ? (double) e.time_us / (1000.0 * e.count) : 0.0,
                e.op ? e.op : "?",
                (long long) e.ne[0], (long long) e.ne[1], (long long) e.ne[2], (long long) e.ne[3],
                (long long) e.elements,
                e.name.c_str());
    }
}
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.embeddings_pre_norm = false;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_type          = params.ctx_type;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    if (const char * env = getenv("LLAMA_DISABLE_FUSED_GDN_AR"); env && atoi(env) != 0) {
        cparams.fused_gdn_ar = false;
        LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) disabled by LLAMA_DISABLE_FUSED_GDN_AR\n", __func__);
    }
    if (const char * env = getenv("LLAMA_DISABLE_FUSED_GDN_CH"); env && atoi(env) != 0) {
        cparams.fused_gdn_ch = false;
        LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) disabled by LLAMA_DISABLE_FUSED_GDN_CH\n", __func__);
    }

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n",   __func__, cparams.n_rs_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
            /*.ctx_type= */ cparams.ctx_type,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // MTP draft contexts can't reserve until the source context is wired
        // via llama_set_mtp_source — defer to the first decode.
        if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
            sched_reserve();
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                // PDMQ compressed K cache with DOT4 FA supports quantized V without cparams.flash_attn.
#ifdef GGML_USE_HIP
                const bool pdmq_default_enabled = true;
#else
                const bool pdmq_default_enabled = false;
#endif
                const char * pdmq_env = getenv("GGML_CUDA_ROCM_PDMQ_K_CACHE");
                if (!pdmq_env) {
                    pdmq_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE"); // legacy alias
                }
                const bool pdmq_active = pdmq_env ? atoi(pdmq_env) != 0 : pdmq_default_enabled;
                if (!pdmq_active) {
                    throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
                }
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    mtp.hook_batch = llama_batch{};
    mtp.hook_token.clear();
    mtp.hook_embd.clear();
    mtp.hook_pos.clear();
    mtp.hook_n_seq_id.clear();
    mtp.hook_seq_id_storage.clear();
    mtp.hook_seq_id_ptrs.clear();
    mtp.hook_logits.clear();
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // When called from decode(), src_mctx_for_decode is already populated and
    // we must not drop it on exit (process_ubatch still needs it). Snapshot
    // only when sched_reserve runs standalone (e.g. lazy first-decode reserve
    // when set_mtp_source flipped sched_need_reserve).
    const bool owns_src_snapshot = src_ctx && !src_mctx_for_decode;
    if (owns_src_snapshot) {
        auto * src_memory = src_ctx->get_memory();
        if (!src_memory) {
            throw std::runtime_error("MTP source context has no memory module");
        }
        src_mctx_for_decode = src_memory->init_full();
        if (!src_mctx_for_decode) {
            throw std::runtime_error("failed to initialize MTP source memory snapshot");
        }
    }
    src_mctx_reset_on_exit reserve_src_drop{owns_src_snapshot ? &src_mctx_for_decode : nullptr};

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    const bool skip_redundant_sync = []() {
        const char * env = getenv("LLAMA_SKIP_REDUNDANT_SYNCHRONIZE");
        return env && atoi(env) != 0;
    }();

    // Do not skip synchronization when backend sampling output buffers exist:
    // sampled-token D2H copies are async and may be the only pending work before
    // llama_get_sampled_token_ith() reads the host buffer. Skipping here can make
    // the server consume a stale/invalid token on prompt-only completions.
    if (skip_redundant_sync && !sampling.sampled.has_data() && n_queued_tokens == 0 && t_compute_start_us == 0) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_pre_norm() {
    output_reorder();

    return embd_pre_norm.data;
}

float * llama_context::get_embeddings_pre_norm_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_pre_norm.data == nullptr) {
            throw std::runtime_error("no pre-norm embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_pre_norm_masked) {
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_pre_norm.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_pre_norm.size / n_embd));
            }
            return embd_pre_norm.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_pre_norm.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid pre-norm embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_pre_norm(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    const bool changed = cparams.embeddings_pre_norm        != value ||
                         cparams.embeddings_pre_norm_masked != masked;

    cparams.embeddings_pre_norm        = value;
    cparams.embeddings_pre_norm_masked = masked;

    if (changed) {
        sched_need_reserve = true;
    }
}

void llama_context::set_mtp_source(llama_context * src) {
    if (src_ctx == src) {
        return;
    }
    llama_assert_gemma4_mtp_source_placement(this, src);
    src_ctx = src;
    src_mctx_for_decode.reset();

    const char * mtp_disable_fa = getenv("LLAMA_MTP_DISABLE_FA");
    if (!mtp_disable_fa) {
        mtp_disable_fa = getenv("GGML_CUDA_ROCM_MTP_DISABLE_FA");
    }
    const char * mtp_enable_fa = getenv("LLAMA_MTP_ENABLE_FA");
    const bool mtp_fa_disabled = (mtp_disable_fa && atoi(mtp_disable_fa) != 0) ||
        (mtp_enable_fa && atoi(mtp_enable_fa) == 0);

    if (cparams.flash_attn && mtp_fa_disabled) {
        // Explicit opt-out only. Packed16 K MTP needs FA by default.
        cparams.flash_attn = false;
        cparams.auto_fa    = false;
        LLAMA_LOG_WARN("%s: disabling Flash Attention in the MTP draft context by explicit env override\n", __func__);
    }

    // worst-case compute buffers were reserved without knowing about the source
    // memory or draft FA policy; force a re-reserve so the next decode sees them
    sched_need_reserve = true;

    // Wire reverse hook: source (target) knows about this (draft) so that
    // handle_mtp_for_ubatch can feed hidden states to the MTP draft during decode.
    // Guarded by LLAMA_MTP_HOOK_WIRE=1 until GPU synchronization issue is resolved.
    if (getenv("LLAMA_MTP_HOOK_WIRE") && !src->mtp.ctx_mtp) {
        src->set_mtp(this);
    }
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

static void llama_mtp_target_lm_head_top1_shadow_check(const llm_graph_result * res);

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    llama_mtp_node_profile_state mtp_node_profile;
    bool mtp_node_profile_active = false;
    if (gtype == LLM_GRAPH_TYPE_DECODER_MTP || gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY) {
        const char * env = getenv("LLAMA_MTP_NODE_PROFILE");
        if (!llama_mtp_node_profile_is_disabled(env)) {
            if (cparams.cb_eval != nullptr) {
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("%s: LLAMA_MTP_NODE_PROFILE disabled because a user eval callback is already installed\n", __func__);
                    warned = true;
                }
            } else {
                mtp_node_profile.all      = strcmp(env, "all") == 0;
                mtp_node_profile.graph    = gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY ? "prefix_verify" : "mtp";
                mtp_node_profile.n_tokens = (int32_t) ubatch.n_tokens;
                mtp_node_profile_active   = true;
                ggml_backend_sched_set_eval_callback(sched.get(), llama_mtp_node_profile_cb, &mtp_node_profile);
            }
        }
    }

    llama_prefix_snapshot_trace_state prefix_snapshot_trace;
    if (gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY) {
        const char * env = getenv("LLAMA_MTP_PREFIX_SNAPSHOT_TRACE");
        const char * hidden_env = getenv("LLAMA_MTP_PREFIX_HIDDEN_TRACE");
        const char * hidden_env_compat = getenv("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE");
        if (!llama_mtp_node_profile_is_disabled(env) ||
                !llama_mtp_node_profile_is_disabled(hidden_env) ||
                !llama_mtp_node_profile_is_disabled(hidden_env_compat)) {
            if (cparams.cb_eval != nullptr || mtp_node_profile_active) {
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("%s: LLAMA_MTP_PREFIX_SNAPSHOT_TRACE disabled because another eval callback is already installed\n", __func__);
                    warned = true;
                }
            } else {
                prefix_snapshot_trace.layer        = llama_env_i32("LLAMA_MTP_PREFIX_HIDDEN_TRACE_LAYER",
                        llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE_LAYER",
                            llama_env_i32("LLAMA_MTP_PREFIX_SNAPSHOT_TRACE_LAYER", 0)));
                prefix_snapshot_trace.source_layer = llama_env_i32("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE_LAYER", prefix_snapshot_trace.layer);
                prefix_snapshot_trace.candidate_layer = llama_env_i32("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE_LAYER", prefix_snapshot_trace.layer);
                prefix_snapshot_trace.max_print    = llama_env_i32("LLAMA_MTP_PREFIX_SNAPSHOT_TRACE_MAX_PRINT", 0);
                prefix_snapshot_trace.source_trace    = llama_env_i32("LLAMA_MTP_PREFIX_STATE_SOURCE_TRACE", 0) != 0;
                prefix_snapshot_trace.candidate_trace = llama_env_i32("LLAMA_MTP_PREFIX_STATE_CANDIDATE_TRACE", 0) != 0;
                prefix_snapshot_trace.hidden_trace    = llama_env_i32("LLAMA_MTP_PREFIX_HIDDEN_TRACE",
                        llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_HIDDEN_TRACE", 0)) != 0;
                prefix_snapshot_trace.n_tokens        = ubatch.n_tokens;
                prefix_snapshot_trace.tokens.reserve(ubatch.n_tokens);
                prefix_snapshot_trace.pos.reserve(ubatch.n_tokens);
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    prefix_snapshot_trace.tokens.push_back(ubatch.token ? ubatch.token[i] : LLAMA_TOKEN_NULL);
                    prefix_snapshot_trace.pos.push_back(ubatch.pos ? ubatch.pos[i] : -1);
                }
                ggml_backend_sched_set_eval_callback(sched.get(), llama_prefix_snapshot_trace_cb, &prefix_snapshot_trace);
            }
        }
    }

    llama_gdn_input_trace_state gdn_input_trace;
    bool gdn_input_trace_active = false;
    if (gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER) {
        const char * env = getenv("LLAMA_MTP_GDN_INPUT_TRACE");
        if (!llama_mtp_node_profile_is_disabled(env)) {
            if (cparams.cb_eval != nullptr) {
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("%s: LLAMA_MTP_GDN_INPUT_TRACE disabled because a user eval callback is already installed\n", __func__);
                    warned = true;
                }
            } else {
                gdn_input_trace.layer        = llama_env_i32("LLAMA_MTP_GDN_INPUT_TRACE_LAYER", 0);
                gdn_input_trace.max_print    = llama_env_i32("LLAMA_MTP_GDN_INPUT_TRACE_MAX_PRINT", 8);
                gdn_input_trace.max_n_tokens = llama_env_i32("LLAMA_MTP_GDN_INPUT_TRACE_MAX_TOKENS", 8);
                gdn_input_trace.compare      = llama_env_i32("LLAMA_MTP_GDN_INPUT_TRACE_COMPARE", 1) != 0;
                if (model.hparams.ssm_d_inner > 0 && model.hparams.ssm_dt_rank > 0) {
                    gdn_input_trace.gdn_sv = model.hparams.ssm_d_inner / model.hparams.ssm_dt_rank;
                } else {
                    gdn_input_trace.gdn_sv = model.hparams.ssm_d_state;
                }
                gdn_input_trace.n_tokens     = ubatch.n_tokens;
                gdn_input_trace.n_seq_tokens = ubatch.n_seq_tokens;
                gdn_input_trace.n_seqs       = ubatch.n_seqs;
                gdn_input_trace.tokens.reserve(ubatch.n_tokens);
                gdn_input_trace.pos.reserve(ubatch.n_tokens);
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    gdn_input_trace.tokens.push_back(ubatch.token ? ubatch.token[i] : LLAMA_TOKEN_NULL);
                    gdn_input_trace.pos.push_back(ubatch.pos ? ubatch.pos[i] : -1);
                }
                gdn_input_trace_active = true;
                ggml_backend_sched_set_eval_callback(sched.get(), llama_gdn_input_trace_cb, &gdn_input_trace);
            }
        }
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);

    if (mtp_node_profile_active) {
        llama_mtp_node_profile_print(mtp_node_profile);
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
    }
    if (gdn_input_trace_active) {
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
    }

    if (getenv("LLAMA_MTP_FINITE_PROBE") && gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        ggml_cgraph * gf_probe = res->get_gf();
        const char * names_env = getenv("LLAMA_MTP_FINITE_PROBE");
        const bool all_names = strcmp(names_env, "1") == 0 || strcmp(names_env, "all") == 0;
        for (int i = 0; i < gf_probe->n_nodes; ++i) {
            ggml_tensor * t = gf_probe->nodes[i];
            const char * tn = t->name;
            const bool want = all_names ||
                strstr(tn, "mtp_Qcur_full") || strstr(tn, "mtp_Qcur_normed") ||
                strstr(tn, "mtp_Kcur_normed") || strstr(tn, "mtp_Vcur") ||
                strstr(tn, "mtp_gate") || strstr(tn, "mtp_attn_pregate") ||
                strstr(tn, "mtp_attn_out") || strstr(tn, "mtp_attn_residual") ||
                strstr(tn, "mtp_ffn_out") || strstr(tn, "mtp_post_ffn") ||
                strstr(tn, "h_pre_norm") || strstr(tn, "result_output");
            if (!want || t->buffer == nullptr || t->type != GGML_TYPE_F32) {
                continue;
            }
            const int64_t n = ggml_nelements(t);
            const int64_t n_check = std::min<int64_t>(n, 4096);
            std::vector<float> tmp(n_check);
            ggml_backend_tensor_get(t, tmp.data(), 0, n_check*sizeof(float));
            int bad = 0;
            float mn = INFINITY, mx = -INFINITY;
            for (int64_t j = 0; j < n_check; ++j) {
                const float v = tmp[j];
                if (!std::isfinite(v)) { ++bad; continue; }
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            fprintf(stderr, "MTP_FINITE: node=%d name=%s type=%d ne=[%lld,%lld,%lld,%lld] checked=%lld bad=%d min=%.6g max=%.6g\n",
                i, tn, t->type,
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                (long long)n_check, bad, mn, mx);
        }
    }

    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    const bool mtp_target_top1_shadow_enabled = []() {
        const char * env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW");
        return env && atoi(env) != 0;
    }();
    if (mtp_target_top1_shadow_enabled && res->t_logits != nullptr && res->t_mtp_target_top1_fused_all != nullptr) {
        // graph_compute() is async; the shadow verifier reads graph outputs synchronously on the host.
        // Synchronize only when the default-off verifier has both full logits and fused top1 to compare.
        ggml_backend_sched_synchronize(sched.get());
    }
    llama_mtp_target_lm_head_top1_shadow_check(res);

    if (mtp.ctx_mtp) {
        handle_mtp_for_ubatch(
                (int32_t) ubatch.n_tokens,
                ubatch.token,
                ubatch.pos,
                res->t_h_pre_norm);
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_pre_norm row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = ctx_type_to_embd_inp(hparams, cparams.ctx_type);
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits        = res->get_logits();
    auto * t_embd          = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_pre_norm    = cparams.embeddings_pre_norm ? (res->get_mtp_h_capture() ? res->get_mtp_h_capture() : res->get_h_pre_norm()) : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract pre-norm embeddings (hidden state before the final output norm)
    if (embd_pre_norm.data && t_h_pre_norm && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_pre_norm);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_pre_norm.size);
        ggml_backend_tensor_get_async(backend_h, t_h_pre_norm, embd_pre_norm.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static void llama_mtp_target_lm_head_top1_shadow_check(const llm_graph_result * res) {
    const char * shadow_env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW");
    if (!(shadow_env && atoi(shadow_env) != 0)) {
        return;
    }
    if (res == nullptr || res->t_logits == nullptr || res->t_mtp_target_top1_fused_all == nullptr) {
        return;
    }

    const ggml_tensor * logits_tensor = res->t_logits;
    const ggml_tensor * fused_tensor  = res->t_mtp_target_top1_fused_all;
    GGML_ASSERT(logits_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(fused_tensor->type == GGML_TYPE_I32);

    const int64_t n_vocab = logits_tensor->ne[0];
    const int64_t n_rows  = ggml_nrows(logits_tensor);
    if (n_rows <= 0) {
        return;
    }
    GGML_ASSERT(n_vocab > 0);
    GGML_ASSERT(ggml_nelements(fused_tensor) == n_rows);

    std::vector<float> logits((size_t) n_vocab * (size_t) n_rows);
    std::vector<int32_t> fused_ids(n_rows, -1);
    ggml_backend_tensor_get(logits_tensor, logits.data(), 0, logits.size()*sizeof(float));
    ggml_backend_tensor_get(fused_tensor,  fused_ids.data(), 0, fused_ids.size()*sizeof(int32_t));

    int checked = 0;
    int mismatch = 0;
    int printed = 0;
    for (int64_t row = 0; row < n_rows; ++row) {
        const float * row_logits = logits.data() + (size_t) row * (size_t) n_vocab;
        int32_t full_id = 0;
        float best = row_logits[0];
        for (int64_t i = 1; i < n_vocab; ++i) {
            const float v = row_logits[i];
            if (v > best) {
                best = v;
                full_id = (int32_t) i;
            }
        }

        ++checked;
        if (full_id != fused_ids[row]) {
            ++mismatch;
            if (printed < 16) {
                const int32_t fused_id = fused_ids[row];
                const float fused_logit = fused_id >= 0 && fused_id < n_vocab ? row_logits[fused_id] : -INFINITY;
                fprintf(stderr, "MTP_TARGET_TOP1_SHADOW: row=%lld ok=0 full=%d fused=%d full_logit=%.9g fused_logit=%.9g\n",
                        (long long) row, (int) full_id, (int) fused_id, (double) best, (double) fused_logit);
                ++printed;
            }
        }
    }

    static int64_t total_checked = 0;
    static int64_t total_mismatch = 0;
    total_checked += checked;
    total_mismatch += mismatch;

    const bool verbose = []() {
        const char * env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_LOG");
        return env && atoi(env) != 0;
    }();
    if (verbose || mismatch != 0) {
        fprintf(stderr,
                "MTP_TARGET_TOP1_SHADOW: summary checked=%d mismatch=%d total_checked=%lld total_mismatch=%lld\n",
                checked, mismatch,
                (long long) total_checked, (long long) total_mismatch);
    }

    const char * require = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE");
    if (require && atoi(require) != 0 && mismatch != 0) {
        GGML_ABORT("LLAMA_MTP_TARGET_LM_HEAD_TOPK_SHADOW_REQUIRE=1 but fused target top1 mismatched full logits top1");
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    if (const char * env = getenv("LLAMA_MTP_TOPK_VERIFY")) {
        if (atoi(env) != 0) {
            return true;
        }
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    const llm_graph_type gtype = llama_mtp_decode_prefix_verify_requested()
        ? LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY
        : ctx_type_to_graph_type(cparams.ctx_type);

    return decode(batch_inp, gtype);
}

int llama_context::decode(const llama_batch & batch_inp, llm_graph_type gtype) {
    // MTP hook batches carry both token (next-token id) and embd (h_pre_norm row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = ctx_type_to_embd_inp(hparams, cparams.ctx_type);

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    const char * force_ubatch_one_env = getenv("LLAMA_MTP_DECODE_FORCE_UBATCH_ONE");
    const uint32_t decode_n_ubatch = force_ubatch_one_env && atoi(force_ubatch_one_env) != 0 ? 1u : cparams.n_ubatch;

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    src_mctx_reset_on_exit decode_src_drop{&src_mctx_for_decode};
    if (src_ctx) {
        auto * src_memory = src_ctx->get_memory();
        if (!src_memory) {
            LLAMA_LOG_ERROR("%s: MTP source context has no memory module\n", __func__);
            return -2;
        }
        src_mctx_for_decode = src_memory->init_full();
        if (!src_mctx_for_decode) {
            LLAMA_LOG_ERROR("%s: failed to snapshot MTP source memory\n", __func__);
            return -2;
        }
    }

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, decode_n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, gtype, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits        = res->get_logits();
        auto * t_embd          = cparams.embeddings          ? res->get_embd()        : nullptr;
        auto * t_h_pre_norm    = cparams.embeddings_pre_norm ? (res->get_mtp_h_capture() ? res->get_mtp_h_capture() : res->get_h_pre_norm()) : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        // Direct target-top1 active routes keep logits on device for GPU top-k / shadow work but
        // return sampled-token rows directly.  In that lab mode, avoid the large full-vocab D2H
        // copy; common_sampler_sample() will consume sampling.sampled instead.
        const bool direct_target_top1_sampled = res->t_mtp_target_top1_fused_all != nullptr && sampling.sampled.has_data();
        if (logits.data && t_logits && n_outputs > 0 && !direct_target_top1_sampled && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // extract pre-norm embeddings (hidden state before the final output norm)
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_pre_norm_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_pre_norm.data && t_h_pre_norm && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_pre_norm);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd = hparams.n_embd_out();
                float * embd_pre_norm_out = embd_pre_norm.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_pre_norm.size);
                ggml_backend_tensor_get_async(backend_h, t_h_pre_norm, embd_pre_norm_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        // Copy direct fused target-top1 outputs, if present, into sampled-token rows.
        // This path is independent of backend sampler maps and supports multiple output rows per sequence.
        if (res->t_mtp_target_top1_fused_all != nullptr && sampling.sampled.has_data()) {
            ggml_tensor * t_top1 = res->t_mtp_target_top1_fused_all;
            GGML_ASSERT(t_top1->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(t_top1));
            GGML_ASSERT(ggml_nelements(t_top1) == n_outputs);
            GGML_ASSERT((size_t) (n_outputs_prev + n_outputs) <= sampling.sampled.size);

            ggml_backend_t backend_top1 = ggml_backend_sched_get_tensor_backend(sched.get(), t_top1);
            GGML_ASSERT(backend_top1 != nullptr);
            ggml_backend_tensor_get_async(backend_top1, t_top1, sampling.sampled.data + n_outputs_prev, 0, n_outputs*sizeof(llama_token));
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints      (res->t_sampled,        sampling.sampled,    seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits        = true;
    bool has_embd          = cparams.embeddings;
    bool has_embd_pre_norm = cparams.embeddings_pre_norm;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size        = has_logits        ? n_vocab*n_outputs_max     : 0;
    embd.size          = has_embd          ? n_embd_out*n_outputs_max  : 0;
    embd_pre_norm.size = has_embd_pre_norm ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_pre_norm && !cparams.embeddings_pre_norm_masked) {
        embd_pre_norm.size = (size_t) n_embd_out * n_batch;
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    const bool has_target_top1_sampled = [this]() {
        const char * env = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE");
        const char * raw = getenv("LLAMA_MTP_TARGET_LM_HEAD_TOPK_ACTIVE_RAW_UNSAFE");
        const bool active = env ? atoi(env) != 0 : cparams.embeddings_pre_norm;
        const bool raw_active = raw ? atoi(raw) != 0 : cparams.embeddings_pre_norm;
        return active && raw_active;
    }();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    } else if (has_target_top1_sampled) {
        backend_token_count = n_outputs_max;                    // sampled tokens only
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_pre_norm.size + backend_float_count) * sizeof(float) +
        (                                               backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_pre_norm.data = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_pre_norm = has_embd_pre_norm ? buffer_view<float>{(float *) (base + offset), embd_pre_norm.size} : buffer_view<float>{nullptr, 0};
    offset += embd_pre_norm.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else if (has_target_top1_sampled) {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_pre_norm.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd_pre_norm.data[i0*n_embd + k], embd_pre_norm.data[i1*n_embd + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        } else if (sampling.sampled.has_data()) {
            // Direct target-top1 sampled-token buffers are also indexed by output row.
            // They have no backend sampler maps/counts, but must still follow the same
            // lazy output-row swaps as logits/embeddings for recurrent prefix verifier batches.
            std::swap(sampling.sampled.data[i0], sampling.sampled.data[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // MTP draft graphs consume both the token id and the target hidden state
    // through llm_graph_input_embd_h. Reservation/support probes are synthetic
    // token-mode batches, so provide a zeroed hidden-state backing store here;
    // otherwise FA-enabled MTP probe graphs can leave/read mtp_h_input without a
    // valid ubatch.embd, then poison/reuse the topology before real MTP decode.
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        GGML_ASSERT(ubatch.data);
        const uint32_t n_embd_mtp = model.hparams.n_embd;
        ubatch.data->embd.assign((size_t) ubatch.n_tokens * n_embd_mtp, 0.0f);
        ubatch.embd = ubatch.data->embd.data();
    }

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    int32_t mtp_prefix_accepted_commit_verify_slots = 0;
    if (gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY && llama_env_i32("LLAMA_MTP_PREFIX_ACCEPTED_ROW_ONLY_COMMIT", 0) != 0) {
        mtp_prefix_accepted_commit_verify_slots = llama_env_i32("LLAMA_MTP_PREFIX_ACCEPTED_ROW_COMMIT_VERIFY_SLOTS", 1);
        if (mtp_prefix_accepted_commit_verify_slots < 0) {
            mtp_prefix_accepted_commit_verify_slots = 0;
        }
    }
    const bool mtp_prefix_batch_output_head =
        gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY && llama_env_i32("LLAMA_MTP_PREFIX_BATCH_OUTPUT_HEAD", 0) != 0;
    const bool mtp_prefix_exact_tail_batch =
        gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY && llama_env_i32("LLAMA_MTP_PREFIX_EXACT_TAIL_BATCH", 0) != 0;
    const bool mtp_prefix_roweq_layer_ffn_batch =
        gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY &&
        (llama_env_i32("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_BATCH", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_EXACT_ROWEQ_BATCH", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_LAYER_FFN_BATCH", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE41_DIAG", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_EXACT_ROW_EQUIV_DIAG", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_COMPONENT_BISECT", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_ACTIVE", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE42_ROUTER_TOPK_BISECT", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_TOPK_BISECT", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_ROUTER_MMVF", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_ROWEQ_ROUTER_MMVF_ACTIVE", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_FUSED_ROUTER_TOPK", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_FUSED", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_STAGE43_ROUTER_TOPK_WEIGHTS", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_PREFIX_ROWEQ_FUSED_ROUTER_TOPK_WEIGHTS", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_ACTIVE", 0) != 0 ||
         llama_env_i32("LLAMA_MTP_ROWEQ_ROUTER_TOPK_WEIGHT_FUSION_ACTIVE", 0) != 0);

    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.src_mctx    =*/ src_mctx_for_decode.get(),
        /*.src_model   =*/ src_ctx ? &src_ctx->get_model() : nullptr,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.mtp_prefix_commit_slot =*/ gtype == LLM_GRAPH_TYPE_DECODER_PREFIX_COMMIT ? llama_env_i32("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT", -1) : -1,
        /*.mtp_prefix_accepted_commit_verify_slots =*/ mtp_prefix_accepted_commit_verify_slots,
        /*.mtp_prefix_batch_output_head =*/ mtp_prefix_batch_output_head,
        /*.mtp_prefix_exact_tail_batch =*/ mtp_prefix_exact_tail_batch,
        /*.mtp_prefix_roweq_layer_ffn_batch =*/ mtp_prefix_roweq_layer_ffn_batch,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            if (!mbuf_cur.buf || mbuf_cur.org.size() != mbuf.org.size() || mbuf_cur.total_size != mbuf.total_size) {
                mbuf_cur = std::move(mbuf);

                mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf_cur.org[i]);
            }
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        const bool need_seq_match = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
            !(flags & LLAMA_STATE_SEQ_FLAGS_ALLOW_SEQ_REMAP);

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));
        if (need_seq_match && seq_id != seq_id_read) {
            throw std::runtime_error("wrong sequence id");
        }

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
            LLAMA_LOG_ERROR("%s: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        const bool is_tbq_k = params.type_k == GGML_TYPE_TBQ3_0 || params.type_k == GGML_TYPE_TBQ4_0;

        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            const uint32_t n_embd_k = is_tbq_k ? model->hparams.n_embd_k_gqa(il) : model->hparams.n_embd_head_k(il);

            if (n_embd_k % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide %s=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size,
                    is_tbq_k ? "n_embd_k_gqa" : "n_embd_head_k", n_embd_k);
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        const bool is_tbq_v = params.type_v == GGML_TYPE_TBQ3_0 || params.type_v == GGML_TYPE_TBQ4_0;

        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            const uint32_t n_embd_v = is_tbq_v ? model->hparams.n_embd_v_gqa(il) : model->hparams.n_embd_head_v(il);

            if (n_embd_v % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide %s=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size,
                    is_tbq_v ? "n_embd_v_gqa" : "n_embd_head_v", n_embd_v);
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        model->hparams.nextn_predict_layers == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }


    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

ggml_tensor * llama_context::get_t_h_pre_norm() const {
    return gf_res_prev ? gf_res_prev->t_h_pre_norm : nullptr;
}

ggml_tensor * llama_context_get_t_h_pre_norm(struct llama_context * ctx) {
    return ctx ? ctx->get_t_h_pre_norm() : nullptr;
}

ggml_tensor * llama_context::get_t_mtp_out() const {
    return gf_res_prev ? gf_res_prev->t_mtp_out : nullptr;
}

ggml_tensor * llama_context_get_t_mtp_out(struct llama_context * ctx) {
    return ctx ? ctx->get_t_mtp_out() : nullptr;
}

void llama_set_mtp(struct llama_context * ctx_target, struct llama_context * ctx_mtp) {
    if (!ctx_target) return;
    ctx_target->set_mtp(ctx_mtp);
}

void llama_context::set_mtp(llama_context * ctx_mtp_in) {
    if (mtp.ctx_mtp == ctx_mtp_in) return;

    mtp.hook_batch = llama_batch{};
    mtp.hook_token.clear();
    mtp.hook_embd.clear();
    mtp.hook_pos.clear();
    mtp.hook_n_seq_id.clear();
    mtp.hook_seq_id_storage.clear();
    mtp.hook_seq_id_ptrs.clear();
    mtp.hook_logits.clear();

    mtp.ctx_mtp     = ctx_mtp_in;
    mtp.pending_pos = -1;

    if (mtp.ctx_mtp) {
        const int32_t n_ub   = (int32_t) cparams.n_ubatch;
        const int32_t n_embd = (int32_t) model.hparams.n_embd;

        mtp.hook_token.resize(n_ub);
        mtp.hook_embd.resize((size_t) n_ub * n_embd);
        mtp.hook_pos.resize(n_ub);
        mtp.hook_n_seq_id.resize(n_ub);
        mtp.hook_seq_id_storage.resize(n_ub);
        mtp.hook_seq_id_ptrs.resize((size_t) n_ub + 1);
        mtp.hook_logits.resize(n_ub);
        for (int32_t i = 0; i < n_ub; ++i) {
            mtp.hook_seq_id_ptrs[i] = &mtp.hook_seq_id_storage[i];
        }
        mtp.hook_seq_id_ptrs[n_ub] = nullptr;

        mtp.hook_batch.token    = mtp.hook_token.data();
        mtp.hook_batch.embd     = mtp.hook_embd.data();
        mtp.hook_batch.pos      = mtp.hook_pos.data();
        mtp.hook_batch.n_seq_id = mtp.hook_n_seq_id.data();
        mtp.hook_batch.seq_id   = mtp.hook_seq_id_ptrs.data();
        mtp.hook_batch.logits   = mtp.hook_logits.data();

        mtp.pending_h.assign(n_embd, 0.0f);
        LLAMA_LOG_INFO("%s: MTP draft head registered (ctx_mtp=%p, n_ubatch=%d, n_embd=%d)\n",
                       __func__, (const void *) mtp.ctx_mtp, n_ub, n_embd);
    } else {
        mtp.pending_h.clear();
        mtp.pending_h.shrink_to_fit();
        LLAMA_LOG_INFO("%s: MTP draft head unregistered\n", __func__);
    }
}

void llama_context::handle_mtp_for_ubatch(
        int32_t                n_tokens,
        const llama_token    * tokens,
        const llama_pos      * positions,
        struct ggml_tensor   * t) {
    if (n_tokens == 0 || t == nullptr) {
        return;
    }
    if (t->ne[1] != (int64_t) n_tokens) {
        return;
    }
    const int64_t n_embd = model.hparams.n_embd;
    GGML_ASSERT(t->ne[0] == n_embd);

    const int       n_rows    = (int) n_tokens;
    const llama_pos pos_start = positions[0];

    const llama_pos pos_max_mtp = llama_memory_seq_pos_max(llama_get_memory(mtp.ctx_mtp), 0);
    if (pos_start <= pos_max_mtp) {
        return;
    }

    const bool pending_continues = mtp.pending_pos >= 0 && mtp.pending_pos + 1 == pos_start;
    if (mtp.pending_pos >= 0 && !pending_continues) {
        mtp.pending_pos = -1;
    }

    synchronize();

    const size_t row_bytes = (size_t) n_embd * sizeof(float);
    const int    n_out     = (pending_continues ? 1 : 0) + (n_rows - 1);

    if (n_out > 0) {
        int chunk_max = n_out;
        if (const char * env = getenv("LLAMA_MTP_PREFILL_CHUNK")) {
            char * end = nullptr;
            const long env_chunk = std::strtol(env, &end, 10);
            if (end != env && env_chunk > 0) {
                chunk_max = (int) std::min<long>(env_chunk, n_out);
            } else {
                LLAMA_LOG_WARN("%s: ignoring invalid LLAMA_MTP_PREFILL_CHUNK='%s'\n", __func__, env);
            }
        }
        chunk_max = std::max(1, std::min(chunk_max, (int) cparams.n_ubatch));
        if (chunk_max < n_out) {
            LLAMA_LOG_INFO("%s: chunking MTP prefill n_out=%d chunk=%d\n", __func__, n_out, chunk_max);
        }

        int out_idx = 0;
        int decoded = 0;
        auto decode_chunk = [&]() -> bool {
            if (out_idx == 0) {
                return true;
            }
            mtp.hook_batch.n_tokens = out_idx;
            const int32_t rc_dec = llama_decode(mtp.ctx_mtp, mtp.hook_batch);
            mtp.ctx_mtp->synchronize();
            if (rc_dec != 0) {
                LLAMA_LOG_ERROR("%s: llama_decode(ctx_mtp) failed rc=%d (pos=%d, n=%d)\n",
                                __func__, (int) rc_dec, (int) mtp.hook_batch.pos[0], out_idx);
                return false;
            }
            decoded += out_idx;
            out_idx = 0;
            return true;
        };

        auto finish_row = [&](llama_token token, llama_pos pos) -> bool {
            mtp.hook_batch.token[out_idx]     = token;
            mtp.hook_batch.pos[out_idx]       = pos;
            mtp.hook_batch.n_seq_id[out_idx]  = 1;
            mtp.hook_batch.seq_id[out_idx][0] = 0;
            mtp.hook_batch.logits[out_idx]    = 0;
            ++out_idx;
            return out_idx < chunk_max || decode_chunk();
        };

        if (pending_continues) {
            std::memcpy(mtp.hook_batch.embd + (size_t) out_idx * n_embd,
                        mtp.pending_h.data(), row_bytes);
            if (!finish_row(tokens[0], pos_start)) {
                return;
            }
        }

        for (int k = 0; k + 1 < n_rows; ++k) {
            ggml_backend_tensor_get(t,
                mtp.hook_batch.embd + (size_t) out_idx * n_embd,
                (size_t) k * row_bytes,
                row_bytes);
            if (!finish_row(tokens[k + 1], positions[k + 1])) {
                return;
            }
        }
        if (!decode_chunk()) {
            return;
        }
        GGML_ASSERT(decoded == n_out);
    }

    // Stash the last h-row as the new pending (for the next ubatch's first
    // token to pair with).
    ggml_backend_tensor_get(t, mtp.pending_h.data(),
        (size_t) (n_rows - 1) * row_bytes, row_bytes);
    mtp.pending_pos = pos_start + n_rows - 1;
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_logits_raw_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_pre_norm(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_pre_norm(value, masked);
}

void llama_set_mtp_source(llama_context * ctx, llama_context * src) {
    ctx->set_mtp_source(src);
}

float * llama_get_embeddings_pre_norm(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_pre_norm();
}

float * llama_get_embeddings_pre_norm_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_pre_norm_ith(i);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

bool llama_context_seq_rm(
    struct llama_context * ctx,
            llama_seq_id   seq_id,
               llama_pos   p0,
               llama_pos   p1) {
    if (!ctx) {
        return true;
    }
    const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);

    if (llama_context * ctx_mtp = ctx->get_mtp()) {
        llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, p0, p1);
    }
    return ok;
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

static bool llama_prefix_batch_validate(
        const char *    fn,
        llama_batch     batch,
        bool            require_logits,
        bool            require_no_logits) {
    if (batch.n_tokens <= 0 || (!batch.token && !batch.embd)) {
        LLAMA_LOG_ERROR("%s: invalid prefix batch: n_tokens=%d token=%p embd=%p\n",
                fn, batch.n_tokens, (const void *) batch.token, (const void *) batch.embd);
        return false;
    }

    const bool has_explicit_seq_ids = batch.n_seq_id != nullptr && batch.seq_id != nullptr;
    llama_seq_id seq_id = 0;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (has_explicit_seq_ids) {
            if (batch.n_seq_id[i] != 1) {
                LLAMA_LOG_ERROR("%s: prefix graph requires exactly one seq_id per row, row %d has %d\n",
                        fn, i, batch.n_seq_id[i]);
                return false;
            }
            if (i == 0) {
                seq_id = batch.seq_id[i][0];
            } else if (batch.seq_id[i][0] != seq_id) {
                LLAMA_LOG_ERROR("%s: prefix graph requires one sequence, row %d has seq_id %d expected %d\n",
                        fn, i, batch.seq_id[i][0], seq_id);
                return false;
            }
        }

        if (batch.pos != nullptr && i > 0 && batch.pos[i] != batch.pos[i - 1] + 1) {
            LLAMA_LOG_ERROR("%s: prefix graph requires contiguous positions, row %d has pos %d after %d\n",
                    fn, i, batch.pos[i], batch.pos[i - 1]);
            return false;
        }

        const bool has_logits = batch.logits != nullptr && batch.logits[i] != 0;
        if (require_logits && !has_logits) {
            LLAMA_LOG_ERROR("%s: prefix verify requires logits/output for every row, row %d is disabled\n",
                    fn, i);
            return false;
        }
        if (require_no_logits && has_logits) {
            LLAMA_LOG_ERROR("%s: prefix commit is logits-free, row %d has logits enabled\n", fn, i);
            return false;
        }
    }

    return true;
}

int32_t llama_decode_prefix_verify(
        llama_context * ctx,
          llama_batch   batch) {
    if (ctx == nullptr) {
        return -1;
    }

    // The prefix verifier contract is intentionally narrower than llama_decode():
    // one real sequence, contiguous prefix rows, and logits for every row.
    if (!llama_prefix_batch_validate(__func__, batch, /*require_logits=*/true, /*require_no_logits=*/false)) {
        return -1;
    }

    return ctx->decode(batch, LLM_GRAPH_TYPE_DECODER_PREFIX_VERIFY);
}

int32_t llama_decode_prefix_commit(
        llama_context * ctx,
          llama_batch   batch,
          uint32_t      dst_slot) {
    if (ctx == nullptr) {
        return -1;
    }

    if (!llama_prefix_batch_validate(__func__, batch, /*require_logits=*/false, /*require_no_logits=*/true)) {
        return -1;
    }

    struct env_scope {
        bool had = false;
        std::string old;
        env_scope(uint32_t slot) {
            if (const char * cur = getenv("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT")) {
                had = true;
                old = cur;
            }
            const std::string value = std::to_string(slot);
#if defined(_WIN32)
            _putenv_s("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT", value.c_str());
#else
            setenv("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT", value.c_str(), 1);
#endif
        }
        ~env_scope() {
#if defined(_WIN32)
            _putenv_s("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT", had ? old.c_str() : "");
#else
            if (had) {
                setenv("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT", old.c_str(), 1);
            } else {
                unsetenv("LLAMA_MTP_PREFIX_COMMIT_DST_SLOT");
            }
#endif
        }
    } scope(dst_slot);

    return ctx->decode(batch, LLM_GRAPH_TYPE_DECODER_PREFIX_COMMIT);
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

bool llama_memory_recurrent_commit_pending_rs_rollback(llama_memory_t mem, llama_seq_id seq_id) {
    if (mem == nullptr) {
        return true;
    }

    if (auto * recr = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recr->commit_pending_rs_rollback(seq_id);
    }

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        llama_memory_recurrent * recr = hybrid->get_mem_recr();
        return recr != nullptr && recr->commit_pending_rs_rollback(seq_id);
    }

    return false;
}

bool llama_context_recurrent_commit_pending_rs_rollback(struct llama_context * ctx, llama_seq_id seq_id) {
    if (ctx == nullptr) {
        return true;
    }

    return llama_memory_recurrent_commit_pending_rs_rollback(llama_get_memory(ctx), seq_id);
}

bool llama_memory_recurrent_set_pending_rs_rollback(llama_memory_t mem, llama_seq_id seq_id, uint32_t idx) {
    if (mem == nullptr) {
        return true;
    }

    if (auto * recr = dynamic_cast<llama_memory_recurrent *>(mem)) {
        recr->set_rs_idx(seq_id, idx);
        return true;
    }

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        llama_memory_recurrent * recr = hybrid->get_mem_recr();
        if (recr == nullptr) {
            return false;
        }
        recr->set_rs_idx(seq_id, idx);
        return true;
    }

    return false;
}

bool llama_context_recurrent_set_pending_rs_rollback(struct llama_context * ctx, llama_seq_id seq_id, uint32_t idx) {
    if (ctx == nullptr) {
        return true;
    }

    return llama_memory_recurrent_set_pending_rs_rollback(llama_get_memory(ctx), seq_id, idx);
}
