// dp16-trace.cuh — guarded trace helpers for packed16/DOT4 operand plans
#pragma once

#include "dp16-plan.cuh"

#include <cstdio>

static inline const char * dp16_cstr_or_none(const char * s) {
    return s && *s ? s : "none";
}

static inline const char * dp16_ggml_type_name_safe(const ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT ? ggml_type_name(type) : "unknown";
}

static inline void dp16_trace_emit_operand(const char * label, const dp16_operand_desc & desc) {
    fprintf(stderr,
        "DP16 operand=%s role=%s storage=%s layout=%s ggml_type=%s rows=%lld cols=%lld "
        "stride_row=%lld stride_col=%lld scale_block=%d pack_lanes=%d signed_i8=%d zero_point=%d min_bias=%d sideband_sums=%d\n",
        dp16_cstr_or_none(label),
        dp16_operand_role_name(desc.role),
        dp16_storage_name(desc.storage),
        dp16_layout_name(desc.layout),
        dp16_ggml_type_name_safe(desc.ggml_type),
        (long long) desc.rows,
        (long long) desc.cols,
        (long long) desc.stride_row,
        (long long) desc.stride_col,
        desc.scale_block,
        desc.pack_lanes,
        desc.signed_i8 ? 1 : 0,
        desc.has_zero_point ? 1 : 0,
        desc.has_min_bias ? 1 : 0,
        desc.has_sideband_sums ? 1 : 0);
}

static inline void dp16_trace_emit_plan(const dp16_problem & problem, const dp16_plan & plan) {
    if (!dp16_trace_enabled()) {
        return;
    }

    const bool accepted = dp16_plan_accepted(plan);
    fprintf(stderr,
        "DP16 %s op=%s backend=%s route=%s kernel=%s reject=%s correction=%s "
        "m=%lld n=%lld k=%lld batch=%d heads_q=%d heads_kv=%d head_dim=%d cc=%d "
        "tile=%dx%dx%d dot4=%d packed16=%d k_multiple_256=%d\n",
        accepted ? "accept" : "reject",
        dp16_op_name(problem.op),
        dp16_backend_name(plan.backend),
        dp16_cstr_or_none(plan.route_name),
        dp16_cstr_or_none(plan.kernel_name),
        dp16_reject_name(plan.reject),
        dp16_correction_name(plan.correction),
        (long long) problem.m,
        (long long) problem.n,
        (long long) problem.k,
        problem.batch,
        problem.heads_q,
        problem.heads_kv,
        problem.head_dim,
        problem.cc,
        plan.tile_m,
        plan.tile_n,
        plan.tile_k,
        plan.use_dot4 ? 1 : 0,
        plan.use_packed16 ? 1 : 0,
        plan.requires_k_multiple_256 ? 1 : 0);

    if (accepted || plan.a.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("A", plan.a);
    } else if (problem.a.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("A", problem.a);
    }

    if (accepted || plan.b.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("B", plan.b);
    } else if (problem.b.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("B", problem.b);
    }

    if (accepted || plan.v.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("V", plan.v);
    } else if (problem.v.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("V", problem.v);
    }

    if (accepted || plan.dst.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("DST", plan.dst);
    } else if (problem.dst.role != DP16_OPERAND_UNKNOWN) {
        dp16_trace_emit_operand("DST", problem.dst);
    }
}

static inline void dp16_trace_emit_fa_plan(const dp16_fa_problem & problem, const dp16_fa_plan & plan) {
    if (!dp16_trace_enabled()) {
        return;
    }

    const dp16_fa_graph_key key = dp16_make_fa_graph_key(problem, plan);
    const uint64_t graph_hash = dp16_hash_fa_graph_key(key);
    fprintf(stderr,
        "dp16_fa_plan status=%s op=FA_QKPV inst=%s plane=%s backend=%s route=%s reason=%s "
        "nq=%d nk_bucket=%d d=%d heads_q=%d heads_kv=%d gqa=%d batch=%d "
        "q_type=%s k_type=%s v_type=%s q_layout=%s k_layout=%s v_layout=%s k_repr=%s "
        "shape=%s vpath=%s qtok_tile=%d gqa_tile=%d logical_q=%d k_tile=%d "
        "experimental=%d fallback=%d capture=%d capture_safe=%d default_allowed=%d route_required=%d "
        "causal=%d mask=%d sliding_window=%d sink=%d graph_key=0x%llx\n",
        plan.selected ? "selected" : "reject",
        dp16_fa_inst_name(problem.inst),
        dp16_fa_plane_name(plan.plane),
        dp16_backend_name(plan.backend),
        dp16_cstr_or_none(plan.route),
        dp16_cstr_or_none(plan.reason),
        problem.nq,
        problem.nk_bucket,
        problem.d_head,
        problem.n_heads_q,
        problem.n_heads_kv,
        problem.gqa_ratio,
        problem.batch,
        dp16_ggml_type_name_safe(problem.q_type),
        dp16_ggml_type_name_safe(problem.k_type),
        dp16_ggml_type_name_safe(problem.v_type),
        dp16_layout_name(problem.q_layout),
        dp16_layout_name(problem.k_layout),
        dp16_layout_name(problem.v_layout),
        dp16_k_repr_name(plan.k_repr),
        dp16_fa_shape_name(plan.shape),
        dp16_fa_vpath_name(plan.vpath),
        plan.qtok_tile,
        plan.gqa_tile,
        plan.logical_q,
        plan.k_tile,
        plan.experimental ? 1 : 0,
        plan.fallback ? 1 : 0,
        problem.capture ? 1 : 0,
        plan.capture_safe ? 1 : 0,
        plan.default_allowed ? 1 : 0,
        problem.route_required ? 1 : 0,
        problem.causal ? 1 : 0,
        problem.has_mask ? 1 : 0,
        problem.has_sliding_window ? 1 : 0,
        problem.has_sink ? 1 : 0,
        (unsigned long long) graph_hash);
}

static inline void dp16_trace_emit_reject(const dp16_problem & problem, const dp16_reject_reason reason, const char * route_name) {
    if (!dp16_trace_enabled()) {
        return;
    }
    dp16_plan plan = dp16_plan_reject(reason);
    plan.route_name = route_name;
    plan.a = problem.a;
    plan.b = problem.b;
    plan.v = problem.v;
    plan.dst = problem.dst;
    dp16_trace_emit_plan(problem, plan);
}

static inline void dp16_trace_emit_sidecar(
        const char * route_name,
        const ggml_type source_type,
        const dp16_layout layout,
        const dp16_correction_policy correction,
        const char * cache_state,
        const bool pack_launch,
        const size_t packed_bytes_payload,
        const size_t packed_bytes_scales) {
    if (!dp16_sidecar_trace_enabled()) {
        return;
    }

    fprintf(stderr,
        "DP16 sidecar route=%s source=%s layout=%s correction=%s cache=%s pack_launch=%d "
        "packed_bytes_payload=%zu packed_bytes_scales=%zu\n",
        dp16_cstr_or_none(route_name),
        dp16_ggml_type_name_safe(source_type),
        dp16_layout_name(layout),
        dp16_correction_name(correction),
        dp16_cstr_or_none(cache_state),
        pack_launch ? 1 : 0,
        packed_bytes_payload,
        packed_bytes_scales);
}
