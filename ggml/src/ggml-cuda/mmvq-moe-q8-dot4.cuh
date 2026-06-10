#pragma once

#include "common.cuh"

#include <cstdlib>
#include <cstring>

#define GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_ENV            "LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4"
#define GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG_ENV        "LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG"
#define GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES_ENV "LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES"
#define GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_ROWS_ENV       "LLAMA_MTP_MMVQ_MOE_Q8_0_DOT4_ROWS_PER_BLOCK"

static inline bool ggml_cuda_mtp_mmvq_moe_q8_0_dot4_env_truthy(const char * env) {
    return env != nullptr && env[0] != '\0' &&
        strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static inline bool ggml_cuda_mtp_mmvq_moe_q8_0_dot4_enabled() {
    return ggml_cuda_mtp_mmvq_moe_q8_0_dot4_env_truthy(getenv(GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_ENV));
}

static inline bool ggml_cuda_mtp_mmvq_moe_q8_0_dot4_log_enabled() {
    return ggml_cuda_mtp_mmvq_moe_q8_0_dot4_env_truthy(getenv(GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_LOG_ENV));
}

static inline int ggml_cuda_mtp_mmvq_moe_q8_0_dot4_parse_i32_env(const char * name, const int def) {
    const char * env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env ? (int) v : def;
}

static inline int ggml_cuda_mtp_mmvq_moe_q8_0_dot4_max_routes() {
    const int max_routes = ggml_cuda_mtp_mmvq_moe_q8_0_dot4_parse_i32_env(
            GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_MAX_ROUTES_ENV, 64);
    return max_routes < 1 ? 1 : max_routes;
}

static inline int ggml_cuda_mtp_mmvq_moe_q8_0_dot4_rows_per_block(
        const bool has_fusion, const uint32_t ncols_x, const uint32_t nrows_x) {
    const int requested = ggml_cuda_mtp_mmvq_moe_q8_0_dot4_parse_i32_env(
            GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_ROWS_ENV, -1);
    if (requested == 1 || requested == 2 || requested == 4 || requested == 8) {
        return requested;
    }

    // Qwen3.5/3.6 MoE down projection: K≈512, rows≈2048.  Reuse each token activation
    // load across four rows by default.  Gate/up is usually K≈2048, rows≈1024 and stays at
    // two rows to avoid raising register pressure on the fused GLU path.
    if (!has_fusion && ncols_x <= 1024 && nrows_x >= 1024) {
        return 4;
    }
    return 2;
}

static inline const char * ggml_cuda_mtp_mmvq_moe_q8_0_dot4_reject_reason(
        const int cc,
        const int warp_size,
        const bool has_ids,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t ncols_dst,
        const int nchannels_dst) {
    if (!ggml_cuda_mtp_mmvq_moe_q8_0_dot4_enabled()) {
        return "env_disabled";
    }
#if defined(GGML_USE_HIP)
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return "not_rdna3_0";
    }
    if (warp_size != 32) {
        return "warp_size";
    }
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(warp_size);
    return "not_hip_rocm";
#endif
    if (!has_ids) {
        return "no_ids";
    }
    if (ncols_dst < 1 || ncols_dst > 4) {
        return "ncols_dst";
    }
    if (ncols_x == 0 || nrows_x == 0) {
        return "empty_shape";
    }
    if (ncols_x % QK8_0 != 0) {
        return "k_not_q8_0_aligned";
    }
    if ((int64_t) ncols_dst * (int64_t) nchannels_dst > ggml_cuda_mtp_mmvq_moe_q8_0_dot4_max_routes()) {
        return "too_many_routes";
    }
    return nullptr;
}

static inline bool ggml_cuda_mtp_mmvq_moe_q8_0_dot4_supported(
        const int cc,
        const int warp_size,
        const bool has_ids,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t ncols_dst,
        const int nchannels_dst) {
    return ggml_cuda_mtp_mmvq_moe_q8_0_dot4_reject_reason(
            cc, warp_size, has_ids, ncols_x, nrows_x, ncols_dst, nchannels_dst) == nullptr;
}
