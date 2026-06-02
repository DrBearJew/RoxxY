// dp16-common.cuh — shared packed16/DOT4 backend-family contracts
#pragma once

#include "../common.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// DP16 = packed16/DOT4 backend family. This is intentionally a small
// host-side planning/trace contract, not a kernel implementation.

static constexpr int DP16_MMVQ_PACKED16_COMPILE_MAX_N = 16;
static constexpr int DP16_MMVQ_PACKED16_PROD_MAX_N = 8;
static constexpr int DP16_MMVQ_PACKED16_REUSE_N_COMPILE_MAX_N = 8;
// The no-LDS acc[N] reuse-N kernel is a correctness/safety-passing negative
// prototype, not a production candidate: tight real-MTP profiling regressed
// N=3/4 and increased VGPR. Keep templates only for forced generic/harness
// experiments; MTP ffn_down/attn_out across-N routing is killed.
static constexpr int DP16_MMVQ_PACKED16_REUSE_N_ROUTE_MAX_N = 4;
static constexpr int DP16_MMVQ_PACKED16_REUSE_N_MAX_N = DP16_MMVQ_PACKED16_REUSE_N_COMPILE_MAX_N;
static constexpr int DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK = 4;
// LDS M2xN was the one expert-approved replacement prototype for across-N
// routing. It passed route/correctness/safety but failed real-MTP performance
// badly for N=2..5, so it is retained only as a no-go reference contract.
static constexpr int DP16_MMVQ_PACKED16_LDS_M2N_ROWS_TILE = 2;
static constexpr int DP16_MMVQ_PACKED16_LDS_M2N_QB_TILE = 32;
static constexpr int DP16_MMVQ_PACKED16_LDS_M2N_ROUTE_MAX_N = 4;
static constexpr int DP16_MMVQ_PACKED16_LDS_M2N_FORCE_MAX_N = 5;
// Template coverage stays at 16 for harness/prototype validation. Production
// route selection is capped at 8 unless the explicit wide-N prototype flag is set.
static constexpr int DP16_MMVQ_PACKED16_MAX_N = DP16_MMVQ_PACKED16_COMPILE_MAX_N;

enum dp16_op {
    DP16_OP_NONE,
    DP16_OP_FA_QKPV,
    DP16_OP_DECODE_PROJ_GEMV,
    DP16_OP_DECODE_PROJ_GEMM_SMALL_N,
    DP16_OP_FFN_GEMV,
};

#define DP16_ROUTE_FA_PACKED16_MMQ                 "rocm_packed16_dot4_mmq"
#define DP16_ROUTE_FA2_PACKED16_DOT4_DECODE        "rocm_fa2_packed16_dot4_decode"
#define DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE      "rocm_mtp_f16k_to_packed16_dot4_decode"
#define DP16_ROUTE_FA_Q8K_DOT4_KQ                  "rocm_q8k_dot4_kq"
#define DP16_ROUTE_FA_Q8K_DOT4_PACKED16_VEC        "rocm_q8k_dot4_packed16_vec"
#define DP16_ROUTE_FA_PACKED16_WMMA_TILE           "rocm_packed16_wmma_tile"
#define DP16_ROUTE_FA1_VEC_FALLBACK                "rocm_fattn_vec"
#define DP16_ROUTE_MMVQ_Q8_DOT4                    "rocm_q8_dot4_mmvq"
#define DP16_ROUTE_MMVQ_PACKED16_DOT4              "rocm_packed16_dot4_mmvq"
#define DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4         "rocm_q4_0_packed16_dot4_mmvq"

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

    // Existing/current DP16-family backends.
    DP16_BACKEND_FA_MMQ,
    DP16_BACKEND_MMVQ_Q8_DOT4,
    DP16_BACKEND_MMVQ_PACKED16_I32_DOT4,
    DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4,
    DP16_BACKEND_MMVQ_Q4_REJECT_ONLY,

    // Durable FA lane labels. DOT4/packed16/FA2 owns the experimental fast MTP
    // memory lanes; FA1/VEC is a fallback/control plane, not the architecture owner.
    DP16_BACKEND_FA2_Q8K_DOT4_DECODE,
    DP16_BACKEND_FA2_PACKED16_DOT4_DECODE,
    DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE,
    DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY,
    DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL,
    DP16_BACKEND_FA1_VEC_FALLBACK,
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
        case DP16_BACKEND_NONE:                              return "none";
        case DP16_BACKEND_FA_MMQ:                            return "fa_mmq";
        case DP16_BACKEND_MMVQ_Q8_DOT4:                      return "mmvq_q8_dot4";
        case DP16_BACKEND_MMVQ_PACKED16_I32_DOT4:            return "mmvq_packed16_i32_dot4";
        case DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4:       return "mmvq_q4_0_packed16_i32_dot4";
        case DP16_BACKEND_MMVQ_Q4_REJECT_ONLY:               return "mmvq_q4_reject_only";
        case DP16_BACKEND_FA2_Q8K_DOT4_DECODE:               return "fa2_q8k_dot4_decode";
        case DP16_BACKEND_FA2_PACKED16_DOT4_DECODE:          return "fa2_packed16_dot4_decode";
        case DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE:        return "fa2_f16k_adapt_dot4_decode";
        case DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY:      return "fa2_packed16_dot4_mmq_verify";
        case DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL:         return "fa2_packed16_wmma_prefill";
        case DP16_BACKEND_FA1_VEC_FALLBACK:                  return "fa1_vec_fallback";
        default:                                             return "unknown";
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

enum dp16_fa_plane {
    DP16_FA_PLANE_NONE = 0,
    DP16_FA_PLANE_FA2_DOT4,
    DP16_FA_PLANE_FA2_PDMQ,
    DP16_FA_PLANE_FA2_PWMMA,
    DP16_FA_PLANE_FA1_VEC_FALLBACK,
};

enum dp16_k_repr {
    DP16_K_REPR_UNKNOWN = 0,
    DP16_K_REPR_Q8_BLOCK32,
    DP16_K_REPR_PACKED16_I32_PERSISTENT,
    DP16_K_REPR_F16_TO_PACKED16_EXPERIMENTAL,
    DP16_K_REPR_EXACT_F16_FALLBACK,
};

static inline const char * dp16_fa_plane_name(const dp16_fa_plane plane) {
    switch (plane) {
        case DP16_FA_PLANE_NONE:             return "none";
        case DP16_FA_PLANE_FA2_DOT4:         return "fa2_dot4";
        case DP16_FA_PLANE_FA2_PDMQ:         return "fa2_pdmq";
        case DP16_FA_PLANE_FA2_PWMMA:        return "fa2_pwmma";
        case DP16_FA_PLANE_FA1_VEC_FALLBACK: return "fa1_vec_fallback";
        default:                             return "unknown";
    }
}

static inline const char * dp16_k_repr_name(const dp16_k_repr repr) {
    switch (repr) {
        case DP16_K_REPR_UNKNOWN:                         return "unknown";
        case DP16_K_REPR_Q8_BLOCK32:                      return "q8_block32";
        case DP16_K_REPR_PACKED16_I32_PERSISTENT:         return "packed16_i32_persistent";
        case DP16_K_REPR_F16_TO_PACKED16_EXPERIMENTAL:    return "f16_to_packed16_experimental";
        case DP16_K_REPR_EXACT_F16_FALLBACK:              return "exact_f16_fallback";
        default:                                          return "unknown";
    }
}

enum dp16_fa_inst {
    DP16_FA_INST_UNKNOWN = 0,
    DP16_FA_INST_MTP_DRAFT_DECODE,
    DP16_FA_INST_MTP_VERIFY,
    DP16_FA_INST_DECODE,
    DP16_FA_INST_SHORT_EXTEND,
    DP16_FA_INST_PREFILL,
};

enum dp16_fa_vpath {
    DP16_FA_VPATH_NONE = 0,
    DP16_FA_VPATH_RAW_LDS_Q4,
    DP16_FA_VPATH_STAGE_F32,
    DP16_FA_VPATH_DIRECT_PV,
};

enum dp16_fa_shape {
    DP16_FA_SHAPE_NONE = 0,
    DP16_FA_SHAPE_1X32,
    DP16_FA_SHAPE_2X32,
    DP16_FA_SHAPE_4X32,
    DP16_FA_SHAPE_8X32,
    DP16_FA_SHAPE_1X64,
    DP16_FA_SHAPE_2X64,
    DP16_FA_SHAPE_4X64,
    DP16_FA_SHAPE_16X16,
    DP16_FA_SHAPE_Q2_GQA2_K32,
};

static inline const char * dp16_fa_inst_name(const dp16_fa_inst inst) {
    switch (inst) {
        case DP16_FA_INST_UNKNOWN:          return "unknown";
        case DP16_FA_INST_MTP_DRAFT_DECODE: return "mtp_draft_decode";
        case DP16_FA_INST_MTP_VERIFY:       return "mtp_verify";
        case DP16_FA_INST_DECODE:           return "decode";
        case DP16_FA_INST_SHORT_EXTEND:     return "short_extend";
        case DP16_FA_INST_PREFILL:          return "prefill";
        default:                            return "unknown";
    }
}

static inline const char * dp16_fa_vpath_name(const dp16_fa_vpath vpath) {
    switch (vpath) {
        case DP16_FA_VPATH_NONE:       return "none";
        case DP16_FA_VPATH_RAW_LDS_Q4: return "raw_lds_q4";
        case DP16_FA_VPATH_STAGE_F32:  return "stage_f32";
        case DP16_FA_VPATH_DIRECT_PV:  return "direct_pv";
        default:                       return "unknown";
    }
}

static inline const char * dp16_fa_shape_name(const dp16_fa_shape shape) {
    switch (shape) {
        case DP16_FA_SHAPE_NONE:        return "none";
        case DP16_FA_SHAPE_1X32:        return "1x32";
        case DP16_FA_SHAPE_2X32:        return "2x32";
        case DP16_FA_SHAPE_4X32:        return "4x32";
        case DP16_FA_SHAPE_8X32:        return "8x32";
        case DP16_FA_SHAPE_1X64:        return "1x64";
        case DP16_FA_SHAPE_2X64:        return "2x64";
        case DP16_FA_SHAPE_4X64:        return "4x64";
        case DP16_FA_SHAPE_16X16:       return "16x16";
        case DP16_FA_SHAPE_Q2_GQA2_K32: return "q2_gqa2_k32";
        default:                        return "unknown";
    }
}

static inline bool dp16_env_enabled(const char * name) {
    const char * v = getenv(name);
    return v && atoi(v) != 0;
}

static inline bool dp16_mtp_force_vec_fallback() {
    return dp16_env_enabled("LLAMA_MTP_FORCE_FA1_VEC") ||
           dp16_env_enabled("LLAMA_MTP_FORCE_VEC_FALLBACK");
}

static inline bool dp16_mtp_enable_dot4_fa2() {
    return dp16_env_enabled("LLAMA_MTP_ENABLE_DOT4_FA2") ||
           dp16_env_enabled("LLAMA_MTP_ENABLE_PACKED16_FA") ||
           dp16_env_enabled("GGML_CUDA_FA_ROUTE_REQUIRE_DOT4");
}

static inline bool dp16_mtp_enable_f16_adapt_dot4() {
    return dp16_env_enabled("LLAMA_MTP_ENABLE_F16K_ADAPT_DOT4_FA2");
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

static inline int dp16_mmvq_packed16_runtime_max_n() {
    const char * wide_n = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_WIDE_N");
    return wide_n && atoi(wide_n) != 0 ? DP16_MMVQ_PACKED16_COMPILE_MAX_N : DP16_MMVQ_PACKED16_PROD_MAX_N;
}

static inline bool dp16_mmvq_packed16_reuse_n_enabled() {
    const char * reuse_n = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_REUSE_N");
    return reuse_n && atoi(reuse_n) != 0;
}

static inline bool dp16_mmvq_packed16_generic_reuse_n_enabled() {
    const char * reuse_n = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_GENERIC_REUSE_N");
    return reuse_n && atoi(reuse_n) != 0;
}

static inline bool dp16_mmvq_packed16_reuse_n_supported_n(const int ncols_dst) {
    return ncols_dst >= 2 && ncols_dst <= DP16_MMVQ_PACKED16_REUSE_N_ROUTE_MAX_N;
}

static inline bool dp16_mmvq_packed16_lds_m2n_enabled() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_LDS_M2N");
    return v && atoi(v) != 0;
}

static inline bool dp16_mmvq_packed16_lds_m2n_allow_n5() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_LDS_M2N_N5");
    return v && atoi(v) != 0;
}

static inline bool dp16_mmvq_packed16_lds_m2n_supported_n(const int ncols_dst) {
    if (ncols_dst >= 2 && ncols_dst <= DP16_MMVQ_PACKED16_LDS_M2N_ROUTE_MAX_N) {
        return true;
    }
    return ncols_dst == DP16_MMVQ_PACKED16_LDS_M2N_FORCE_MAX_N && dp16_mmvq_packed16_lds_m2n_allow_n5();
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
