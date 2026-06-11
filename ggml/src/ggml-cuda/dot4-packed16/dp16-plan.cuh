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

    int qtok_tile;
    int gqa_tile;
    int logical_q;
    int k_tile;

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
            plan.shape = DP16_FA_SHAPE_1X64;
            plan.vpath = DP16_FA_VPATH_RAW_LDS_Q4;
            plan.route = DP16_ROUTE_FA_PACKED16_MMQ;
            plan.reason = "mtp_draft_decode_pdmq_q4";
            plan.k_tile = 64;
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

    // MTP verify nq=2..4: PDMQ/DOT4-MMQ owns the fast lane for persistent
    // packed16 K + q4_0 V. VEC is only fallback/control.
    if (p.is_mtp &&
            is_verify &&
            p.nq >= 2 && p.nq <= 4 &&
            p.d_head == 256 &&
            (p.gqa_ratio == 6 || p.gqa_ratio == 8) &&
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
        plan.shape = DP16_FA_SHAPE_Q2_GQA2_K32;
        plan.vpath = DP16_FA_VPATH_RAW_LDS_Q4;
        plan.route = DP16_ROUTE_FA_PACKED16_MMQ;
        plan.reason = "mtp_verify_smallq_pdmq_q4";
        plan.qtok_tile = 2;
        plan.gqa_tile = 2;
        plan.logical_q = 4;
        plan.k_tile = 32;
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
