// dp16-plan.cuh — packed16/DOT4 backend-family problem and plan helpers
#pragma once

#include "dp16-operand.cuh"

struct dp16_problem {
    dp16_op op;

    int64_t m;
    int64_t n;
    int64_t k;

    int batch;
    int heads_q;
    int heads_kv;
    int head_dim;

    ggml_type src0_type;
    ggml_type src1_type;
    ggml_type src2_type;
    ggml_type dst_type;

    // Canonical DP16 operand contract. The src*_type fields remain as compact
    // compatibility inputs for existing call sites; role/storage/layout live here.
    dp16_operand_desc a;
    dp16_operand_desc b;
    dp16_operand_desc v;
    dp16_operand_desc dst;

    bool is_decode;
    bool is_mtp_verify;
    bool is_speculative;

    bool has_fusion;
    bool fusion_x_bias_only;
    bool has_ids;

    bool has_packed16_b;
    dp16_packed16_weight_view b_packed16;

    int cc;
};

struct dp16_fa_problem {
    dp16_fa_inst inst;

    int nq;
    int nk_bucket;
    int d_head;

    int n_heads_q;
    int n_heads_kv;
    int gqa_ratio;
    int batch;

    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;

    dp16_layout q_layout;
    dp16_layout k_layout;
    dp16_layout v_layout;

    bool causal;
    bool has_mask;
    bool has_sliding_window;
    bool has_sink;

    bool is_mtp;
    bool is_draft_decode;
    bool is_verify;
    bool is_prefill;
    bool packed16_k_ready;
    bool capture;
    bool route_required;
};

struct dp16_fa_plan {
    bool valid;
    bool selected;
    bool default_allowed;
    bool capture_safe;
    bool experimental;
    bool fallback;

    dp16_fa_plane plane;
    dp16_backend backend;
    dp16_k_repr k_repr;
    dp16_fa_shape shape;
    dp16_fa_vpath vpath;
    dp16_fa_q_stage q_stage;

    int qtok_tile;
    int gqa_tile;
    int logical_q;
    int k_tile;
    int k_shards_per_q_stage;

    const char * route;
    const char * reason;
};

struct dp16_fa_graph_key {
    dp16_fa_inst inst;
    dp16_fa_plane plane;
    dp16_backend backend;
    dp16_k_repr k_repr;
    dp16_fa_shape shape;
    dp16_fa_vpath vpath;
    dp16_fa_q_stage q_stage;

    int nq;
    int nk_bucket;
    int d_head;
    int n_heads_q;
    int n_heads_kv;
    int gqa_ratio;
    int batch;

    ggml_type q_type;
    ggml_type k_type;
    ggml_type v_type;

    dp16_layout q_layout;
    dp16_layout k_layout;
    dp16_layout v_layout;

    int qtok_tile;
    int gqa_tile;
    int logical_q;
    int k_tile;
    int k_shards_per_q_stage;

    bool causal;
    bool has_mask;
    bool has_sliding_window;
    bool has_sink;

    bool capture_safe;
    bool experimental;
    bool fallback;
};

struct dp16_fa_mmq_subplan {
    int bm;
    int bn;
    int gqa_group;
    int requested_gqa_group;

    const char * v_path;
    bool causal;
    bool kshared;
};

struct dp16_mmvq_plan {
    int n_tile;
    int k_tile;
    bool split_k;
    bool use_sideband_sums;
};

struct dp16_plan {
    dp16_backend backend;
    dp16_reject_reason reject;

    dp16_operand_desc a;
    dp16_operand_desc b;
    dp16_operand_desc v;
    dp16_operand_desc dst;

    // Legacy compact format labels retained for simple predicates/logs. The
    // operand descriptors above are the canonical substrate representation.
    dp16_format a_format;
    dp16_format b_format;
    dp16_format v_format;
    dp16_correction_policy correction;

    int tile_m;
    int tile_n;
    int tile_k;

    bool use_dot4;
    bool use_packed16;
    bool requires_k_multiple_256;

    const char * route_name;
    const char * kernel_name;

    dp16_fa_mmq_subplan fa;
    dp16_mmvq_plan mmvq;
};

static inline dp16_problem dp16_problem_init(const dp16_op op) {
    dp16_problem p = {};
    p.op = op;
    p.a = dp16_operand_unknown();
    p.b = dp16_operand_unknown();
    p.v = dp16_operand_unknown();
    p.dst = dp16_operand_unknown();
    return p;
}

static inline dp16_plan dp16_plan_reject(const dp16_reject_reason reason) {
    dp16_plan plan = {};
    plan.backend = DP16_BACKEND_NONE;
    plan.reject = reason;
    plan.a = dp16_operand_unknown();
    plan.b = dp16_operand_unknown();
    plan.v = dp16_operand_unknown();
    plan.dst = dp16_operand_unknown();
    plan.correction = DP16_CORR_UNSUPPORTED;
    return plan;
}

static inline bool dp16_plan_accepted(const dp16_plan & plan) {
    return plan.reject == DP16_REJECT_NONE && plan.backend != DP16_BACKEND_NONE;
}

static inline dp16_fa_plan dp16_fa_plan_reject(const char * route, const char * reason) {
    dp16_fa_plan plan = {};
    plan.valid = false;
    plan.selected = false;
    plan.plane = DP16_FA_PLANE_NONE;
    plan.backend = DP16_BACKEND_NONE;
    plan.k_repr = DP16_K_REPR_UNKNOWN;
    plan.shape = DP16_FA_SHAPE_NONE;
    plan.vpath = DP16_FA_VPATH_NONE;
    plan.route = route;
    plan.reason = reason;
    return plan;
}

static inline dp16_fa_plan dp16_make_fa1_vec_fallback(const char * reason) {
    dp16_fa_plan plan = {};
    plan.valid = true;
    plan.selected = true;
    plan.default_allowed = true;
    plan.capture_safe = true;
    plan.experimental = false;
    plan.fallback = true;
    plan.plane = DP16_FA_PLANE_FA1_VEC_FALLBACK;
    plan.backend = DP16_BACKEND_FA1_VEC_FALLBACK;
    plan.k_repr = DP16_K_REPR_EXACT_F16_FALLBACK;
    plan.shape = DP16_FA_SHAPE_NONE;
    plan.vpath = DP16_FA_VPATH_NONE;
    plan.route = DP16_ROUTE_FA1_VEC_FALLBACK;
    plan.reason = reason;
    return plan;
}

static inline dp16_fa_shape dp16_parse_pdmq_shape_env(const char * env) {
    if (!env || !*env) {
        return DP16_FA_SHAPE_NONE;
    }
    if (strcmp(env, "1x32") == 0 || strcmp(env, "m1n32") == 0) {
        return DP16_FA_SHAPE_1X32;
    }
    if (strcmp(env, "2x32") == 0 || strcmp(env, "m2n32") == 0) {
        return DP16_FA_SHAPE_2X32;
    }
    if (strcmp(env, "4x32") == 0 || strcmp(env, "m4n32") == 0) {
        return DP16_FA_SHAPE_4X32;
    }
    if (strcmp(env, "8x32") == 0 || strcmp(env, "m8n32") == 0) {
        return DP16_FA_SHAPE_8X32;
    }
    if (strcmp(env, "16x16") == 0 || strcmp(env, "m16n16") == 0) {
        return DP16_FA_SHAPE_16X16;
    }
    GGML_ABORT("DP16 FA planner: bad GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE=%s", env);
}

static inline dp16_fa_shape dp16_pdmq_shape_for_problem(const dp16_fa_problem & p) {
    // Keep the planner/graph-key shape contract aligned with the PDMQ launcher.
    // Explicit shape knobs must win first because they choose a different kernel
    // specialization even when nq/gqa/vpath are otherwise identical.
    if (const char * env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE"); env && *env) {
        return dp16_parse_pdmq_shape_env(env);
    }

    if (const char * env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PV_ROWS"); env && *env && p.nq <= 4) {
        const int pv_rows = atoi(env);
        if (pv_rows != 1 && pv_rows != 2 && pv_rows != 4) {
            GGML_ABORT("DP16 FA planner: invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_PV_ROWS=%s; expected 1, 2, or 4", env);
        }
        // For 27B GQA6, PV_ROWS is a max/verify tile hint, not a reason to
        // run decode as mostly-zero M4. Match active M rows: 6/12/24.
        if (p.v_type == GGML_TYPE_Q4_0 && p.n_heads_q == 24 && p.n_heads_kv == 4 && p.gqa_ratio == 6) {
            return p.nq <= 1 ? DP16_FA_SHAPE_1X32 : (p.nq <= 2 ? DP16_FA_SHAPE_2X32 : DP16_FA_SHAPE_4X32);
        }
        if (pv_rows == 1) {
            return DP16_FA_SHAPE_1X32;
        }
        if (pv_rows == 2) {
            return DP16_FA_SHAPE_2X32;
        }
        return DP16_FA_SHAPE_4X32;
    }

    const char * shape_auto_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE_AUTO");
    const bool shape_auto = !shape_auto_env || atoi(shape_auto_env) != 0;
    if (!shape_auto) {
        return p.nq >= 16 ? DP16_FA_SHAPE_16X16 : DP16_FA_SHAPE_8X32;
    }

    const char * requested_gqa_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP");
    const int requested_gqa_group = requested_gqa_env && *requested_gqa_env ? atoi(requested_gqa_env) : 1;
    const char * qwen27b_min_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_MIN_NK");
    const int qwen27b_gqa6_min_nk = qwen27b_min_env && *qwen27b_min_env ? atoi(qwen27b_min_env) : 12288;
    const bool qwen27b_gqa6_pvmma = p.nq <= 4 && p.nk_bucket >= qwen27b_gqa6_min_nk && p.v_type == GGML_TYPE_Q4_0 &&
        p.n_heads_q == 24 && p.n_heads_kv == 4 && p.gqa_ratio == 6 &&
        (dp16_env_enabled("GGML_CUDA_DP16_FA_PV_WMMA") || requested_gqa_group == 6);
    if (qwen27b_gqa6_pvmma) {
        return p.nq <= 1 ? DP16_FA_SHAPE_1X32 : DP16_FA_SHAPE_2X32;
    }

    if (p.nq <= 1) {
        return DP16_FA_SHAPE_1X32;
    }
    if (p.nq <= 4) {
        return p.nq == 2 ? DP16_FA_SHAPE_2X32 : DP16_FA_SHAPE_8X32;
    }
    if (p.nq >= 16) {
        return DP16_FA_SHAPE_16X16;
    }
    if (p.nq >= 8 && p.v_type != GGML_TYPE_F16) {
        return DP16_FA_SHAPE_16X16;
    }
    if (p.nk_bucket <= 32 && p.nq >= 4) {
        return DP16_FA_SHAPE_16X16;
    }
    return DP16_FA_SHAPE_8X32;
}

static inline int dp16_fa_shape_qtok_tile(const dp16_fa_shape shape) {
    switch (shape) {
        case DP16_FA_SHAPE_1X32:
        case DP16_FA_SHAPE_1X64:        return 1;
        case DP16_FA_SHAPE_2X32:
        case DP16_FA_SHAPE_2X64:
        case DP16_FA_SHAPE_Q2_GQA2_K32: return 2;
        case DP16_FA_SHAPE_4X32:
        case DP16_FA_SHAPE_4X64:        return 4;
        case DP16_FA_SHAPE_8X32:        return 8;
        case DP16_FA_SHAPE_16X16:       return 16;
        default:                        return 0;
    }
}

static inline int dp16_fa_shape_k_tile(const dp16_fa_shape shape) {
    switch (shape) {
        case DP16_FA_SHAPE_1X64:
        case DP16_FA_SHAPE_2X64:
        case DP16_FA_SHAPE_4X64: return 64;
        case DP16_FA_SHAPE_16X16:return 16;
        case DP16_FA_SHAPE_1X32:
        case DP16_FA_SHAPE_2X32:
        case DP16_FA_SHAPE_4X32:
        case DP16_FA_SHAPE_8X32:
        case DP16_FA_SHAPE_Q2_GQA2_K32:
            return 32;
        default:
            return 0;
    }
}

struct dp16_pdmq_q_stage_plan {
    dp16_fa_q_stage q_stage;
    int k_shards_per_q_stage;
    int gqa_tile;
};

static inline int dp16_pdmq_splitk_roof_pow2(const int nk) {
    constexpr int min_tokens_per_split = 64;
    if (nk >= 8 * min_tokens_per_split) return 8;
    if (nk >= 4 * min_tokens_per_split) return 4;
    if (nk >= 2 * min_tokens_per_split) return 2;
    return 1;
}

static inline int dp16_pdmq_gqax_splitk_requested() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK");
    if (!s || !*s) {
        return 1;
    }
    const int split_k = atoi(s);
    if (split_k == 1 || split_k == 2 || split_k == 4 || split_k == 8 || split_k == 16 || split_k == 32 || split_k == 64) {
        return split_k;
    }
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK=%s; expected 1, 2, 4, 8, 16, 32, or 64", s);
}

static inline int dp16_pdmq_requested_gqa_group() {
    const char * s = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP");
    if (!s || !*s) {
        return 1;
    }
    const int g = atoi(s);
    if (g == 1 || g == 2 || g == 4 || g == 6) {
        return g;
    }
    GGML_ABORT("invalid GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP=%s; expected 1, 2, 4, or 6", s);
}

static inline bool dp16_pdmq_kshared_requested() {
    return false;
}

static inline dp16_pdmq_q_stage_plan dp16_pdmq_q_stage_for_problem(
        const dp16_fa_problem & p,
        const dp16_fa_shape shape) {
    dp16_pdmq_q_stage_plan out = {};
    out.q_stage = DP16_FA_Q_STAGE_INLINE;
    out.k_shards_per_q_stage = 1;
    out.gqa_tile = 1;

    if (!dp16_fa_qpack_i8_enabled()) {
        return out;
    }
    if (p.nq > 8 || p.nk_bucket < dp16_fa_qpack_i8_min_nk()) {
        return out;
    }
    const int requested_gqa_group = dp16_pdmq_requested_gqa_group();
    const bool requested_gqa4_9b = requested_gqa_group == 4 &&
        p.n_heads_q == 16 && p.n_heads_kv == 4 && p.gqa_ratio == 4;
    const bool requested_gqa6_27b =
        (requested_gqa_group == 6 ||
         (requested_gqa_group == 1 && dp16_env_enabled("GGML_CUDA_DP16_FA_PV_WMMA"))) &&
        p.n_heads_q == 24 && p.n_heads_kv == 4 && p.gqa_ratio == 6;
    const bool gqa4_split_shape = requested_gqa4_9b &&
        (shape == DP16_FA_SHAPE_1X32 || shape == DP16_FA_SHAPE_2X32 || shape == DP16_FA_SHAPE_4X32 ||
         shape == DP16_FA_SHAPE_8X32 || shape == DP16_FA_SHAPE_16X16);
    const bool gqa6_split_shape = requested_gqa6_27b &&
        (shape == DP16_FA_SHAPE_1X32 || shape == DP16_FA_SHAPE_2X32);
    const bool gqa1_qpack_shape = requested_gqa_group == 1 && !requested_gqa6_27b;
    if (!(gqa1_qpack_shape || gqa4_split_shape || gqa6_split_shape) || dp16_pdmq_kshared_requested()) {
        return out;
    }
    if (!(p.v_type == GGML_TYPE_Q4_0 && p.v_layout == DP16_LAYOUT_Q4_0_BLOCK32)) {
        return out;
    }

    const int k_tile = dp16_fa_shape_k_tile(shape);
    if (k_tile <= 0) {
        return out;
    }

    const char * split_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK");
    const bool split_env_set = split_env && *split_env;
    int split_requested = dp16_pdmq_gqax_splitk_requested();

    const char * gqax_min_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_MIN_NK");
    const int gqax_splitk_min_nk = gqax_min_env && *gqax_min_env ? atoi(gqax_min_env) : 12288;
    if (split_requested > 1 && p.nk_bucket < gqax_splitk_min_nk) {
        split_requested = 1;
    }

    const char * auto_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA1_SPLITK_AUTO");
    const bool auto_enabled = !auto_env || atoi(auto_env) != 0;
    const char * gqa1_min_env = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA1_SPLITK_MIN_NK");
    const int gqa1_splitk_min_nk = gqa1_min_env && *gqa1_min_env ? atoi(gqa1_min_env) : 12288;
    if (((requested_gqa_group == 1 && !requested_gqa6_27b) || requested_gqa6_27b) && auto_enabled && !split_env_set && p.nk_bucket >= gqa1_splitk_min_nk) {
        split_requested = 32;
    }

    const bool roof_cap = dp16_env_enabled("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_ROOF_CAP");
    const int roof = dp16_pdmq_splitk_roof_pow2(p.nk_bucket);
    const int split_effective = roof_cap && split_requested > roof ? roof : split_requested;
    const int k_blocks_total_raw = (p.nk_bucket + k_tile - 1) / k_tile;
    const int k_blocks_total = k_blocks_total_raw > 0 ? k_blocks_total_raw : 1;
    const bool compact_empty = dp16_env_enabled("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK_COMPACT_EMPTY");
    const int split_active_raw = (compact_empty && split_effective > k_blocks_total) ? k_blocks_total : split_effective;
    if (split_active_raw <= 1) {
        return out;
    }

    out.k_shards_per_q_stage = dp16_fa_k_shards_per_q_stage();
    const int split_coarsened = (split_active_raw + out.k_shards_per_q_stage - 1) / out.k_shards_per_q_stage;
    const int split_active = out.k_shards_per_q_stage > 1 ? (split_coarsened > 1 ? split_coarsened : 1) : split_active_raw;
    if (split_active > 1) {
        out.q_stage = DP16_FA_Q_STAGE_QPACK_I8_BLOCK32;
        out.gqa_tile = gqa6_split_shape ? 6 : (gqa4_split_shape ? 4 : 1);
    }
    return out;
}

static inline dp16_fa_plan dp16_plan_mtp_fa(const dp16_fa_problem & p) {
    if (dp16_mtp_force_vec_fallback()) {
        return dp16_make_fa1_vec_fallback("forced_vec_safety_fallback");
    }

    const bool is_decode = p.is_draft_decode || p.inst == DP16_FA_INST_MTP_DRAFT_DECODE || p.inst == DP16_FA_INST_DECODE;
    const bool is_verify = p.is_verify || p.inst == DP16_FA_INST_MTP_VERIFY;
    const bool is_prefill = p.is_prefill || p.inst == DP16_FA_INST_PREFILL || p.inst == DP16_FA_INST_SHORT_EXTEND;
    const bool dot4_enabled = p.route_required || dp16_mtp_enable_dot4_fa2();

    // MTP draft decode nq==1: DOT4/packed16 is the target fast lane. FA1/VEC
    // remains only the fallback/control route when DOT4 is disabled or unsafe.
    if (p.is_mtp && is_decode && p.nq == 1 && p.d_head == 256) {
        if (dot4_enabled &&
                p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED &&
                p.packed16_k_ready &&
                p.v_type == GGML_TYPE_Q4_0 &&
                p.v_layout == DP16_LAYOUT_Q4_0_BLOCK32) {
            dp16_fa_plan plan = {};
            plan.valid = true;
            plan.selected = true;
            plan.capture_safe = true;
            plan.experimental = true;
            plan.fallback = false;
            plan.plane = DP16_FA_PLANE_FA2_PDMQ;
            plan.backend = DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY;
            plan.k_repr = DP16_K_REPR_PACKED16_I32_PERSISTENT;
            plan.shape = dp16_pdmq_shape_for_problem(p);
            plan.vpath = DP16_FA_VPATH_RAW_LDS_Q4;
            const dp16_pdmq_q_stage_plan q_stage_plan = dp16_pdmq_q_stage_for_problem(p, plan.shape);
            plan.q_stage = q_stage_plan.q_stage;
            plan.route = DP16_ROUTE_FA_PACKED16_MMQ;
            plan.reason = "mtp_draft_decode_pdmq_q4";
            plan.qtok_tile = dp16_fa_shape_qtok_tile(plan.shape);
            plan.gqa_tile = q_stage_plan.gqa_tile;
            plan.logical_q = plan.qtok_tile * plan.gqa_tile;
            plan.k_tile = dp16_fa_shape_k_tile(plan.shape);
            plan.k_shards_per_q_stage = q_stage_plan.k_shards_per_q_stage;
            return plan;
        }

        if (dot4_enabled &&
                p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED &&
                p.packed16_k_ready) {
            dp16_fa_plan plan = {};
            plan.valid = true;
            plan.selected = true;
            plan.capture_safe = true;
            plan.experimental = true;
            plan.fallback = false;
            plan.plane = DP16_FA_PLANE_FA2_DOT4;
            plan.backend = DP16_BACKEND_FA2_PACKED16_DOT4_DECODE;
            plan.k_repr = DP16_K_REPR_PACKED16_I32_PERSISTENT;
            plan.shape = DP16_FA_SHAPE_1X64;
            plan.vpath = DP16_FA_VPATH_DIRECT_PV;
            plan.route = DP16_ROUTE_FA2_PACKED16_DOT4_DECODE;
            plan.reason = "mtp_draft_decode_packed16_dot4";
            plan.k_tile = 64;
            return plan;
        }

        if (dot4_enabled && p.k_layout == DP16_LAYOUT_Q8_BLOCK32) {
            dp16_fa_plan plan = {};
            plan.valid = true;
            plan.selected = true;
            plan.capture_safe = true;
            plan.experimental = true;
            plan.fallback = false;
            plan.plane = DP16_FA_PLANE_FA2_DOT4;
            plan.backend = DP16_BACKEND_FA2_Q8K_DOT4_DECODE;
            plan.k_repr = DP16_K_REPR_Q8_BLOCK32;
            plan.shape = DP16_FA_SHAPE_1X64;
            plan.vpath = DP16_FA_VPATH_DIRECT_PV;
            plan.route = DP16_ROUTE_FA_Q8K_DOT4_KQ;
            plan.reason = "mtp_draft_decode_q8k_dot4";
            plan.k_tile = 64;
            return plan;
        }

        // Experimental source-f16 adaptation is a fast FA2/DOT4 lane, not an
        // exact owner. It must be explicit in logs and kept out of capture until
        // a prepared sidecar/capture policy is wired by the launcher.
        if (dot4_enabled &&
                dp16_mtp_enable_f16_adapt_dot4() &&
                p.k_type == GGML_TYPE_F16 &&
                p.v_type == GGML_TYPE_Q4_0 &&
                !p.capture) {
            dp16_fa_plan plan = {};
            plan.valid = true;
            plan.selected = true;
            plan.capture_safe = false;
            plan.experimental = true;
            plan.fallback = false;
            plan.plane = DP16_FA_PLANE_FA2_DOT4;
            plan.backend = DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE;
            plan.k_repr = DP16_K_REPR_F16_TO_PACKED16_EXPERIMENTAL;
            plan.shape = DP16_FA_SHAPE_1X64;
            plan.vpath = DP16_FA_VPATH_DIRECT_PV;
            plan.route = DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE;
            plan.reason = "mtp_draft_decode_f16k_adapt_dot4_experimental";
            plan.k_tile = 64;
            return plan;
        }

        return dp16_make_fa1_vec_fallback("dot4_decode_not_available_fallback_vec");
    }

    // MTP verify nq=2..8: PDMQ/DOT4-MMQ owns the fast lane for persistent
    // packed16 K + q4_0 V. VEC is only fallback/control. Keep this planner
    // contract aligned with the launcher, which routes Qwen3.5 GQA4 and late
    // MTP verify groups (including nq=5) through GQA1 split-K.
    if (p.is_mtp &&
            is_verify &&
            p.nq >= 2 && p.nq <= 8 &&
            p.d_head == 256 &&
            (p.gqa_ratio == 4 || p.gqa_ratio == 6 || p.gqa_ratio == 8) &&
            p.v_type == GGML_TYPE_Q4_0 &&
            p.v_layout == DP16_LAYOUT_Q4_0_BLOCK32 &&
            p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED &&
            p.packed16_k_ready &&
            dot4_enabled) {
        dp16_fa_plan plan = {};
        plan.valid = true;
        plan.selected = true;
        plan.capture_safe = true;
        plan.experimental = true;
        plan.fallback = false;
        plan.plane = DP16_FA_PLANE_FA2_PDMQ;
        plan.backend = DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY;
        plan.k_repr = DP16_K_REPR_PACKED16_I32_PERSISTENT;
        plan.shape = dp16_pdmq_shape_for_problem(p);
        plan.vpath = DP16_FA_VPATH_RAW_LDS_Q4;
        const dp16_pdmq_q_stage_plan q_stage_plan = dp16_pdmq_q_stage_for_problem(p, plan.shape);
        plan.q_stage = q_stage_plan.q_stage;
        plan.route = DP16_ROUTE_FA_PACKED16_MMQ;
        plan.reason = "mtp_verify_smallq_pdmq_q4";
        plan.qtok_tile = dp16_fa_shape_qtok_tile(plan.shape);
        plan.gqa_tile = q_stage_plan.gqa_tile;
        plan.logical_q = plan.qtok_tile * plan.gqa_tile;
        plan.k_tile = dp16_fa_shape_k_tile(plan.shape);
        plan.k_shards_per_q_stage = q_stage_plan.k_shards_per_q_stage;
        return plan;
    }

    // Prefill/larger-Q: packed16 WMMA/PWMMA is the fast-tile lane when legal.
    if (is_prefill &&
            p.d_head == 256 &&
            p.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED &&
            p.packed16_k_ready &&
            dot4_enabled) {
        dp16_fa_plan plan = {};
        plan.valid = true;
        plan.selected = true;
        plan.capture_safe = true;
        plan.experimental = true;
        plan.fallback = false;
        plan.plane = DP16_FA_PLANE_FA2_PWMMA;
        plan.backend = DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL;
        plan.k_repr = DP16_K_REPR_PACKED16_I32_PERSISTENT;
        plan.shape = DP16_FA_SHAPE_16X16;
        plan.vpath = DP16_FA_VPATH_DIRECT_PV;
        plan.route = DP16_ROUTE_FA_PACKED16_WMMA_TILE;
        plan.reason = "prefill_packed16_wmma";
        plan.k_tile = 64;
        return plan;
    }

    return dp16_make_fa1_vec_fallback("unsupported_fallback_vec");
}

static inline dp16_fa_plan dp16_plan_fa_qkpv(const dp16_fa_problem & p) {
    return dp16_plan_mtp_fa(p);
}

static inline dp16_fa_graph_key dp16_make_fa_graph_key(
        const dp16_fa_problem & p,
        const dp16_fa_plan & plan) {
    dp16_fa_graph_key key = {};
    key.inst = p.inst;
    key.plane = plan.plane;
    key.backend = plan.backend;
    key.k_repr = plan.k_repr;
    key.shape = plan.shape;
    key.vpath = plan.vpath;
    key.q_stage = plan.q_stage;
    key.nq = p.nq;
    key.nk_bucket = p.nk_bucket;
    key.d_head = p.d_head;
    key.n_heads_q = p.n_heads_q;
    key.n_heads_kv = p.n_heads_kv;
    key.gqa_ratio = p.gqa_ratio;
    key.batch = p.batch;
    key.q_type = p.q_type;
    key.k_type = p.k_type;
    key.v_type = p.v_type;
    key.q_layout = p.q_layout;
    key.k_layout = p.k_layout;
    key.v_layout = p.v_layout;
    key.qtok_tile = plan.qtok_tile;
    key.gqa_tile = plan.gqa_tile;
    key.logical_q = plan.logical_q;
    key.k_tile = plan.k_tile;
    key.k_shards_per_q_stage = plan.k_shards_per_q_stage;
    key.causal = p.causal;
    key.has_mask = p.has_mask;
    key.has_sliding_window = p.has_sliding_window;
    key.has_sink = p.has_sink;
    key.capture_safe = plan.capture_safe;
    key.experimental = plan.experimental;
    key.fallback = plan.fallback;
    return key;
}

static inline uint64_t dp16_hash_fa_graph_key(const dp16_fa_graph_key & key) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&](const uint64_t x) {
        h ^= x;
        h *= 1099511628211ull;
    };

    mix((uint64_t) key.inst);
    mix((uint64_t) key.plane);
    mix((uint64_t) key.backend);
    mix((uint64_t) key.k_repr);
    mix((uint64_t) key.shape);
    mix((uint64_t) key.vpath);
    mix((uint64_t) key.q_stage);
    mix((uint64_t) key.nq);
    mix((uint64_t) key.nk_bucket);
    mix((uint64_t) key.d_head);
    mix((uint64_t) key.n_heads_q);
    mix((uint64_t) key.n_heads_kv);
    mix((uint64_t) key.gqa_ratio);
    mix((uint64_t) key.batch);
    mix((uint64_t) key.q_type);
    mix((uint64_t) key.k_type);
    mix((uint64_t) key.v_type);
    mix((uint64_t) key.q_layout);
    mix((uint64_t) key.k_layout);
    mix((uint64_t) key.v_layout);
    mix((uint64_t) key.qtok_tile);
    mix((uint64_t) key.gqa_tile);
    mix((uint64_t) key.logical_q);
    mix((uint64_t) key.k_tile);
    mix((uint64_t) key.k_shards_per_q_stage);
    mix(key.causal ? 1u : 0u);
    mix(key.has_mask ? 1u : 0u);
    mix(key.has_sliding_window ? 1u : 0u);
    mix(key.has_sink ? 1u : 0u);
    mix(key.capture_safe ? 1u : 0u);
    mix(key.experimental ? 1u : 0u);
    mix(key.fallback ? 1u : 0u);
    return h;
}

static inline dp16_plan dp16_plan_fa_mmq(
        const dp16_problem & p,
        const int bm,
        const int bn,
        const int gqa_group,
        const int requested_gqa_group,
        const char * v_path,
        const bool causal,
        const bool kshared) {
    if (p.op != DP16_OP_FA_QKPV) {
        return dp16_plan_reject(DP16_REJECT_OP_UNSUPPORTED);
    }
    if (p.src0_type != GGML_TYPE_F32 || p.src1_type != GGML_TYPE_I32 || p.dst_type != GGML_TYPE_F32) {
        return dp16_plan_reject(DP16_REJECT_TYPE_UNSUPPORTED);
    }
    if (p.head_dim != 256 || p.k != 256) {
        return dp16_plan_reject(DP16_REJECT_K_NOT_ALIGNED);
    }

    dp16_plan plan = {};
    plan.backend = DP16_BACKEND_FA_MMQ;
    plan.reject = DP16_REJECT_NONE;
    plan.a = dp16_operand_desc_make(DP16_OPERAND_ACTIVATION, DP16_STORAGE_TRANSIENT_TILE,
            DP16_LAYOUT_Q8_BLOCK32, p.src0_type, p.m, p.head_dim);
    plan.b = dp16_operand_desc_make(DP16_OPERAND_K_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
            DP16_LAYOUT_PACKED16_I32_SCALED, p.src1_type, p.n, p.head_dim);
    plan.v = dp16_operand_desc_from_type(DP16_OPERAND_V_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
            p.src2_type, p.n, p.head_dim);
    plan.dst = dp16_operand_desc_from_type(DP16_OPERAND_OUTPUT, DP16_STORAGE_OUTPUT,
            p.dst_type, p.m, p.head_dim);
    plan.a_format = DP16_FMT_F32;
    plan.b_format = DP16_FMT_PACKED16_I32_SCALED;
    plan.v_format = dp16_format_from_ggml_type(p.src2_type);
    plan.correction = DP16_CORR_NONE;
    plan.tile_m = bm;
    plan.tile_n = bn;
    plan.tile_k = p.head_dim;
    plan.use_dot4 = true;
    plan.use_packed16 = true;
    plan.requires_k_multiple_256 = true;
    plan.route_name = DP16_ROUTE_FA_PACKED16_MMQ;
    plan.kernel_name = "dot4_packed16_fa_mmq";
    plan.fa.bm = bm;
    plan.fa.bn = bn;
    plan.fa.gqa_group = gqa_group;
    plan.fa.requested_gqa_group = requested_gqa_group;
    plan.fa.v_path = v_path ? v_path : "unknown";
    plan.fa.causal = causal;
    plan.fa.kshared = kshared;
    return plan;
}

static inline bool dp16_is_mmvq_q8_compatible(const dp16_problem & p) {
    return p.src0_type == GGML_TYPE_Q8_0 && p.src1_type == GGML_TYPE_Q8_1;
}

static inline bool dp16_is_mmvq_packed16_compatible(const dp16_problem & p) {
    return p.has_packed16_b &&
        p.b.layout == DP16_LAYOUT_PACKED16_I32_SCALED &&
        p.b.storage == DP16_STORAGE_PERSISTENT_WEIGHT &&
        p.src1_type == GGML_TYPE_Q8_1;
}

static inline dp16_plan dp16_plan_decode_proj_mmvq_q8(const dp16_problem & p) {
    dp16_plan plan = {};
    plan.backend = DP16_BACKEND_MMVQ_Q8_DOT4;
    plan.reject = DP16_REJECT_NONE;
    plan.a = p.a;
    plan.b = p.b;
    plan.v = dp16_operand_unknown();
    plan.dst = p.dst;
    plan.a_format = DP16_FMT_Q8_SIGNED_I8;
    plan.b_format = DP16_FMT_Q8_BLOCK32;
    plan.v_format = DP16_FMT_UNKNOWN;
    plan.correction = DP16_CORR_NONE;
    plan.tile_m = 1;
    plan.tile_n = 4;
    plan.tile_k = 256;
    plan.use_dot4 = true;
    plan.use_packed16 = false;
    plan.requires_k_multiple_256 = true;
    plan.route_name = DP16_ROUTE_MMVQ_Q8_DOT4;
    plan.kernel_name = p.has_fusion ? "dp16_mmvq_q8_dot4_fusion_n1_k256" : "dp16_mmvq_q8_dot4_n1_4_k256";
    plan.mmvq.n_tile = p.has_fusion ? 1 : 4;
    plan.mmvq.k_tile = 256;
    plan.mmvq.split_k = false;
    plan.mmvq.use_sideband_sums = false;
    return plan;
}

static inline dp16_plan dp16_plan_decode_proj_mmvq_packed16(const dp16_problem & p) {
    dp16_plan plan = {};
    plan.backend = DP16_BACKEND_MMVQ_PACKED16_I32_DOT4;
    plan.reject = DP16_REJECT_NONE;
    plan.a = p.a;
    plan.b = p.b;
    plan.v = dp16_operand_unknown();
    plan.dst = p.dst;
    plan.a_format = DP16_FMT_Q8_SIGNED_I8;
    plan.b_format = DP16_FMT_PACKED16_I32_SCALED;
    plan.v_format = DP16_FMT_UNKNOWN;
    plan.correction = DP16_CORR_NONE;
    plan.tile_m = 1;
    plan.tile_n = dp16_mmvq_packed16_runtime_max_n();
    plan.tile_k = 256;
    plan.use_dot4 = true;
    plan.use_packed16 = true;
    plan.requires_k_multiple_256 = true;
    plan.route_name = DP16_ROUTE_MMVQ_PACKED16_DOT4;
    plan.kernel_name = "dp16_mmvq_packed16_i32_b32_n1_16_k256";
    plan.mmvq.n_tile = dp16_mmvq_packed16_runtime_max_n();
    plan.mmvq.k_tile = 256;
    plan.mmvq.split_k = false;
    plan.mmvq.use_sideband_sums = false;
    return plan;
}

static inline dp16_plan dp16_plan_decode_proj_mmvq_q4_0_packed16(const dp16_problem & p) {
    dp16_plan plan = dp16_plan_decode_proj_mmvq_packed16(p);
    plan.backend = DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4;
    plan.b_format = DP16_FMT_PACKED16_I32_SCALED;
    plan.correction = DP16_CORR_PREPACK_SIGNED_I8;
    plan.route_name = DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4;
    plan.kernel_name = "dp16_mmvq_packed16_i32_b32_n1_16_k256";
    return plan;
}

static inline dp16_plan dp16_plan_decode_proj_mmvq(
        const dp16_problem & p,
        const char * required_route = nullptr) {
    if (p.op != DP16_OP_DECODE_PROJ_GEMV) {
        return dp16_plan_reject(DP16_REJECT_OP_UNSUPPORTED);
    }
    const bool require_q8 = dp16_route_name_is_mmvq_q8_dot4(required_route);
    const bool require_p16 = dp16_route_name_is_mmvq_packed16_dot4(required_route);
    const bool require_q4_0 = dp16_route_name_is_mmvq_q4_0_packed16_dot4(required_route);
    const bool packed16_fusion_ok = !p.has_fusion || p.fusion_x_bias_only;
    const bool allow_packed16_wide_n = p.n >= 1 && p.n <= dp16_mmvq_packed16_runtime_max_n() &&
        packed16_fusion_ok && !p.has_ids && !require_q8 &&
        (require_p16 || require_q4_0 || dp16_is_mmvq_packed16_compatible(p));
    if (!(p.n >= 1 && (p.n <= 4 || allow_packed16_wide_n))) {
        return dp16_plan_reject(DP16_REJECT_N_TOO_LARGE);
    }
    if ((p.k % 256) != 0) {
        return dp16_plan_reject(DP16_REJECT_K_NOT_ALIGNED);
    }
    if (p.dst_type != GGML_TYPE_F32) {
        return dp16_plan_reject(DP16_REJECT_DST_UNSUPPORTED);
    }

    if (p.has_ids) {
        return dp16_plan_reject(DP16_REJECT_IDS_UNSUPPORTED);
    }
    if (p.has_fusion) {
        if (p.fusion_x_bias_only && !require_q8 && p.n >= 1 && p.n <= dp16_mmvq_packed16_runtime_max_n()) {
            if ((require_p16 || require_q4_0 || dp16_is_mmvq_packed16_compatible(p)) && dp16_is_mmvq_packed16_compatible(p)) {
                return require_q4_0 ? dp16_plan_decode_proj_mmvq_q4_0_packed16(p) : dp16_plan_decode_proj_mmvq_packed16(p);
            }
        }
        if (p.n != 1 || require_p16 || require_q4_0) {
            return dp16_plan_reject(DP16_REJECT_FUSION_UNSUPPORTED);
        }
        if (require_q8) {
            if (!dp16_is_mmvq_q8_compatible(p)) {
                dp16_plan plan = dp16_plan_reject(DP16_REJECT_TYPE_UNSUPPORTED);
                plan.route_name = DP16_ROUTE_MMVQ_Q8_DOT4;
                plan.a = p.a;
                plan.b = p.b;
                plan.dst = p.dst;
                return plan;
            }
            return dp16_plan_decode_proj_mmvq_q8(p);
        }
        if (dp16_is_mmvq_q8_compatible(p)) {
            return dp16_plan_decode_proj_mmvq_q8(p);
        }
        return dp16_plan_reject(DP16_REJECT_FUSION_UNSUPPORTED);
    }

    if (p.src0_type == GGML_TYPE_Q4_0) {
        if (require_q4_0 && dp16_is_mmvq_packed16_compatible(p)) {
            return dp16_plan_decode_proj_mmvq_q4_0_packed16(p);
        }
        dp16_plan plan = dp16_plan_reject(DP16_REJECT_Q4_CORRECTION_UNDEFINED);
        plan.backend = DP16_BACKEND_MMVQ_Q4_REJECT_ONLY;
        plan.a = p.a;
        plan.b = p.b;
        plan.dst = p.dst;
        plan.a_format = dp16_format_from_ggml_type(p.src1_type);
        plan.b_format = dp16_format_from_ggml_type(p.src0_type);
        plan.correction = DP16_CORR_UNSUPPORTED;
        plan.route_name = require_q4_0 ? DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4 : "fallback";
        plan.kernel_name = "q4_reject_only";
        return plan;
    }

    if (dp16_type_is_q4_candidate(p.src0_type) || dp16_type_is_q4_candidate(p.src1_type)) {
        dp16_plan plan = dp16_plan_reject(DP16_REJECT_Q4_CORRECTION_UNDEFINED);
        plan.backend = DP16_BACKEND_MMVQ_Q4_REJECT_ONLY;
        plan.a = p.a;
        plan.b = p.b;
        plan.dst = p.dst;
        plan.a_format = dp16_format_from_ggml_type(p.src1_type);
        plan.b_format = dp16_format_from_ggml_type(p.src0_type);
        plan.correction = DP16_CORR_UNSUPPORTED;
        plan.route_name = "fallback";
        plan.kernel_name = "q4_reject_only";
        return plan;
    }

    if (require_p16) {
        if (!dp16_is_mmvq_packed16_compatible(p)) {
            dp16_plan plan = dp16_plan_reject(DP16_REJECT_PACKED16_WEIGHT_MISSING);
            plan.route_name = DP16_ROUTE_MMVQ_PACKED16_DOT4;
            plan.a = p.a;
            plan.b = p.b;
            plan.dst = p.dst;
            return plan;
        }
        return dp16_plan_decode_proj_mmvq_packed16(p);
    }

    if (require_q8) {
        if (!dp16_is_mmvq_q8_compatible(p)) {
            dp16_plan plan = dp16_plan_reject(DP16_REJECT_TYPE_UNSUPPORTED);
            plan.route_name = DP16_ROUTE_MMVQ_Q8_DOT4;
            plan.a = p.a;
            plan.b = p.b;
            plan.dst = p.dst;
            return plan;
        }
        return dp16_plan_decode_proj_mmvq_q8(p);
    }

    if (dp16_is_mmvq_packed16_compatible(p)) {
        return dp16_plan_decode_proj_mmvq_packed16(p);
    }

    if (dp16_is_mmvq_q8_compatible(p)) {
        return dp16_plan_decode_proj_mmvq_q8(p);
    }

    return dp16_plan_reject(DP16_REJECT_TYPE_UNSUPPORTED);
}
