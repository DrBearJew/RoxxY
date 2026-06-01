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
    bool has_ids;

    bool has_packed16_b;
    dp16_packed16_weight_view b_packed16;

    int cc;
};

struct dp16_fa_plan {
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

    dp16_fa_plan fa;
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
    plan.tile_n = 4;
    plan.tile_k = 256;
    plan.use_dot4 = true;
    plan.use_packed16 = true;
    plan.requires_k_multiple_256 = true;
    plan.route_name = DP16_ROUTE_MMVQ_PACKED16_DOT4;
    plan.kernel_name = "dp16_mmvq_packed16_i32_b32_n1_4_k256";
    plan.mmvq.n_tile = 4;
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
    plan.kernel_name = "dp16_mmvq_packed16_i32_b32_n1_4_k256";
    return plan;
}

static inline dp16_plan dp16_plan_decode_proj_mmvq(
        const dp16_problem & p,
        const char * required_route = nullptr) {
    if (p.op != DP16_OP_DECODE_PROJ_GEMV) {
        return dp16_plan_reject(DP16_REJECT_OP_UNSUPPORTED);
    }
    if (!(p.n >= 1 && p.n <= 4)) {
        return dp16_plan_reject(DP16_REJECT_N_TOO_LARGE);
    }
    if ((p.k % 256) != 0) {
        return dp16_plan_reject(DP16_REJECT_K_NOT_ALIGNED);
    }
    if (p.dst_type != GGML_TYPE_F32) {
        return dp16_plan_reject(DP16_REJECT_DST_UNSUPPORTED);
    }
    const bool require_q8 = dp16_route_name_is_mmvq_q8_dot4(required_route);
    const bool require_p16 = dp16_route_name_is_mmvq_packed16_dot4(required_route);
    const bool require_q4_0 = dp16_route_name_is_mmvq_q4_0_packed16_dot4(required_route);

    if (p.has_ids) {
        return dp16_plan_reject(DP16_REJECT_IDS_UNSUPPORTED);
    }
    if (p.has_fusion) {
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
