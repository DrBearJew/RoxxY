// dp16-common.cuh — shared packed16/DOT4 backend-family contracts
#pragma once

#include "../common.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// DP16 = packed16/DOT4 backend family. This is intentionally a small
// host-side planning/trace contract, not a kernel implementation.

enum dp16_op {
    DP16_OP_NONE,
    DP16_OP_FA_QKPV,
    DP16_OP_DECODE_PROJ_GEMV,
    DP16_OP_DECODE_PROJ_GEMM_SMALL_N,
    DP16_OP_FFN_GEMV,
};

#define DP16_ROUTE_FA_PACKED16_MMQ        "rocm_packed16_dot4_mmq"
#define DP16_ROUTE_MMVQ_Q8_DOT4           "rocm_q8_dot4_mmvq"
#define DP16_ROUTE_MMVQ_PACKED16_DOT4     "rocm_packed16_dot4_mmvq"
#define DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4 "rocm_q4_0_packed16_dot4_mmvq"

struct dp16_packed16_weight_view {
    const int32_t * payload;
    const half    * scales;

    int64_t rows;
    int64_t cols;

    int64_t payload_stride_row_i32;
    int64_t scale_stride_row_half;

    int64_t payload_stride_channel_i32;
    int64_t scale_stride_channel_half;

    int64_t payload_stride_sample_i32;
    int64_t scale_stride_sample_half;

    ggml_type source_type;
    uint32_t flags;
};

enum dp16_backend {
    DP16_BACKEND_NONE,
    DP16_BACKEND_FA_MMQ,
    DP16_BACKEND_MMVQ_Q8_DOT4,
    DP16_BACKEND_MMVQ_PACKED16_I32_DOT4,
    DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4,
    DP16_BACKEND_MMVQ_Q4_REJECT_ONLY,
};

enum dp16_format {
    DP16_FMT_UNKNOWN,
    DP16_FMT_F16,
    DP16_FMT_F32,
    DP16_FMT_Q8_SIGNED_I8,
    DP16_FMT_Q8_BLOCK32,
    DP16_FMT_Q4_0,
    DP16_FMT_Q4_K,
    DP16_FMT_PACKED16_I32,
    DP16_FMT_PACKED16_I32_SCALED,
};

enum dp16_correction_policy {
    DP16_CORR_NONE,
    DP16_CORR_PREPACK_SIGNED_I8,
    DP16_CORR_RUNTIME_SIDEBAND_SUMS,
    DP16_CORR_UNSUPPORTED,
};

enum dp16_reject_reason {
    DP16_REJECT_NONE,
    DP16_REJECT_OP_UNSUPPORTED,
    DP16_REJECT_TYPE_UNSUPPORTED,
    DP16_REJECT_K_NOT_ALIGNED,
    DP16_REJECT_N_TOO_LARGE,
    DP16_REJECT_Q4_CORRECTION_UNDEFINED,
    DP16_REJECT_ARCH_UNSUPPORTED,
    DP16_REJECT_ROUTE_REQUIRE_MISMATCH,
    DP16_REJECT_RUNTIME_DISABLED,
    DP16_REJECT_PACKED16_WEIGHT_MISSING,
    DP16_REJECT_FUSION_UNSUPPORTED,
    DP16_REJECT_IDS_UNSUPPORTED,
    DP16_REJECT_DST_UNSUPPORTED,
};

static inline const char * dp16_op_name(const dp16_op op) {
    switch (op) {
        case DP16_OP_NONE:                   return "none";
        case DP16_OP_FA_QKPV:                return "fa_qkpv";
        case DP16_OP_DECODE_PROJ_GEMV:       return "decode_proj_gemv";
        case DP16_OP_DECODE_PROJ_GEMM_SMALL_N:return "decode_proj_gemm_small_n";
        case DP16_OP_FFN_GEMV:               return "ffn_gemv";
        default:                             return "unknown";
    }
}

static inline const char * dp16_backend_name(const dp16_backend backend) {
    switch (backend) {
        case DP16_BACKEND_NONE:               return "none";
        case DP16_BACKEND_FA_MMQ:                    return "fa_mmq";
        case DP16_BACKEND_MMVQ_Q8_DOT4:              return "mmvq_q8_dot4";
        case DP16_BACKEND_MMVQ_PACKED16_I32_DOT4:    return "mmvq_packed16_i32_dot4";
        case DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4:return "mmvq_q4_0_packed16_i32_dot4";
        case DP16_BACKEND_MMVQ_Q4_REJECT_ONLY:       return "mmvq_q4_reject_only";
        default:                                     return "unknown";
    }
}

static inline const char * dp16_format_name(const dp16_format format) {
    switch (format) {
        case DP16_FMT_UNKNOWN:       return "unknown";
        case DP16_FMT_F16:           return "f16";
        case DP16_FMT_F32:                   return "f32";
        case DP16_FMT_Q8_SIGNED_I8:          return "q8_signed_i8";
        case DP16_FMT_Q8_BLOCK32:            return "q8_block32";
        case DP16_FMT_Q4_0:                  return "q4_0";
        case DP16_FMT_Q4_K:                  return "q4_K";
        case DP16_FMT_PACKED16_I32:          return "packed16_i32";
        case DP16_FMT_PACKED16_I32_SCALED:   return "packed16_i32_scaled";
        default:                             return "unknown";
    }
}

static inline const char * dp16_correction_name(const dp16_correction_policy correction) {
    switch (correction) {
        case DP16_CORR_NONE:                 return "none";
        case DP16_CORR_PREPACK_SIGNED_I8:    return "prepack_signed_i8";
        case DP16_CORR_RUNTIME_SIDEBAND_SUMS:return "runtime_sideband_sums";
        case DP16_CORR_UNSUPPORTED:          return "unsupported";
        default:                             return "unknown";
    }
}

static inline const char * dp16_reject_name(const dp16_reject_reason reason) {
    switch (reason) {
        case DP16_REJECT_NONE:                   return "none";
        case DP16_REJECT_OP_UNSUPPORTED:         return "op_unsupported";
        case DP16_REJECT_TYPE_UNSUPPORTED:       return "type_unsupported";
        case DP16_REJECT_K_NOT_ALIGNED:          return "k_not_aligned";
        case DP16_REJECT_N_TOO_LARGE:            return "n_too_large";
        case DP16_REJECT_Q4_CORRECTION_UNDEFINED:return "q4_correction_undefined";
        case DP16_REJECT_ARCH_UNSUPPORTED:       return "arch_unsupported";
        case DP16_REJECT_ROUTE_REQUIRE_MISMATCH: return "route_require_mismatch";
        case DP16_REJECT_RUNTIME_DISABLED:       return "runtime_disabled";
        case DP16_REJECT_PACKED16_WEIGHT_MISSING:return "packed16_weight_missing";
        case DP16_REJECT_FUSION_UNSUPPORTED:     return "fusion_unsupported";
        case DP16_REJECT_IDS_UNSUPPORTED:        return "ids_unsupported";
        case DP16_REJECT_DST_UNSUPPORTED:        return "dst_unsupported";
        default:                                 return "unknown";
    }
}

static inline bool dp16_trace_enabled() {
    const char * v = getenv("GGML_CUDA_DP16_TRACE");
    return v && atoi(v) != 0;
}

static inline bool dp16_sidecar_trace_enabled() {
    const char * v = getenv("GGML_CUDA_DP16_SIDECAR_TRACE");
    return v && atoi(v) != 0;
}

static inline const char * dp16_route_require_env() {
    const char * v = getenv("GGML_CUDA_DP16_ROUTE_REQUIRE");
    return v ? v : "";
}

static inline bool dp16_route_name_is_mmvq_q8_dot4(const char * route_name) {
    return route_name && strcmp(route_name, DP16_ROUTE_MMVQ_Q8_DOT4) == 0;
}

static inline bool dp16_route_name_is_mmvq_packed16_dot4(const char * route_name) {
    return route_name &&
        (strcmp(route_name, DP16_ROUTE_MMVQ_PACKED16_DOT4) == 0 ||
         strcmp(route_name, "packed16_dot4_mmvq") == 0);
}

static inline bool dp16_route_name_is_mmvq_q4_0_packed16_dot4(const char * route_name) {
    return route_name && strcmp(route_name, DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4) == 0;
}

static inline bool dp16_route_required(const char * route_name) {
    const char * required = dp16_route_require_env();
    if (!required || !*required || !route_name) {
        return false;
    }
    if (dp16_route_name_is_mmvq_packed16_dot4(route_name)) {
        return dp16_route_name_is_mmvq_packed16_dot4(required);
    }
    if (dp16_route_name_is_mmvq_q4_0_packed16_dot4(route_name)) {
        return dp16_route_name_is_mmvq_q4_0_packed16_dot4(required);
    }
    return strcmp(required, route_name) == 0;
}

static inline bool dp16_type_is_q4_candidate(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K;
}

static inline bool dp16_type_is_q8_candidate(const ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q8_1;
}

static inline dp16_format dp16_format_from_ggml_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return DP16_FMT_F16;
        case GGML_TYPE_F32:  return DP16_FMT_F32;
        case GGML_TYPE_Q8_0: return DP16_FMT_Q8_BLOCK32;
        case GGML_TYPE_Q8_1: return DP16_FMT_Q8_BLOCK32;
        case GGML_TYPE_Q4_0: return DP16_FMT_Q4_0;
        case GGML_TYPE_Q4_K: return DP16_FMT_Q4_K;
        default:             return DP16_FMT_UNKNOWN;
    }
}
