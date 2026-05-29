#pragma once

#include "common.cuh"

#include <cstdlib>
#include <cstring>

struct ggml_backend_cuda_context;
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

#ifdef GGML_USE_HIP

// Lab-only ROCm q8K DOT4 KQ/FA route. Keep this behind both the generic unsafe
// gate and an explicit route contract; standalone fused/tile8 probes currently
// underperform stable tiled/GQA FA. See
// docs/rocm-tbq4-paths/23-q8k-dot4-fa-root-cause-and-next-plan.md before
// changing these gates or promoting variants.
// Route-contract family names:
//   rocm_q8k_dot4_kq                          — broad/backcompat
//   rocm_q8k_dot4_recthist_mtp_verify         — MTP verify recthist-v4
//   rocm_q8k_dot4_decode_mtp_draft            — MTP draft decode (any)
//   rocm_q8k_dot4_decode_bn64_mtp_draft       — BN64 decode
//   rocm_q8k_dot4_decode_splitk_mtp_draft     — split-K decode
static inline bool ggml_cuda_q8k_dot4_kq_route_required() {
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (!required) return false;
    // Broad contract plus all precise family names.
    return strcmp(required, "rocm_q8k_dot4_kq") == 0
        || strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0;
}

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    const char * unsafe = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
    if (!unsafe) {
        unsafe = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
    }
    const char * env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ");
    return unsafe && atoi(unsafe) != 0 && env && atoi(env) != 0 && ggml_cuda_q8k_dot4_kq_route_required();
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!ggml_cuda_q8k_dot4_kq_enabled()) {
        return false;
    }
    const bool k_is_packed16_i32 = (K->type == GGML_TYPE_I32);

    // Defense-in-depth: MTP context must never use packed16 I32 K.
    // The primary gate is in llama_kv_cache (is_mtp_draft), but if an MTP
    // FA op somehow sees I32 K, reject here with a log.
    {
        const int32_t fa_inst = ((const int32_t *)dst->op_params)[4];
        const bool is_mtp = (fa_inst == GGML_FATTN_INST_MTP_DRAFT ||
                             fa_inst == GGML_FATTN_INST_MTP_VERIFY_QK);
        if (is_mtp && k_is_packed16_i32) {
            if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
                if (log_env && atoi(log_env) != 0) {
                    GGML_LOG_INFO("%s: q8k_dot4_kq reject=mtp_packed16_k inst=%s Q=[%lld,%lld,%lld,%lld]\n",
                            __func__,
                            fa_inst == GGML_FATTN_INST_MTP_DRAFT ? "mtp_draft" : "mtp_verify_qk",
                            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3]);
                }
            }
            return false;
        }
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }
    // Accept: q8_0 K (original path) or I32 packed16 K (packed16-only path)
    if (!k_is_packed16_i32 && (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0)) {
        return false;
    }
    if (k_is_packed16_i32 && V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q4_0) {
        return false;  // packed16 path requires f16 V
    }
    // Packed16 I32 K has ne[0] = D_per_head/4 vs Q's ne[0] = D_per_head
    const bool k_shape_ok = k_is_packed16_i32
        ? (K->ne[0] * 4 == Q->ne[0])
        : (K->ne[0] == Q->ne[0] && K->ne[0] == 256);
    if (!k_shape_ok) return false;
    if (!k_is_packed16_i32 && (V->ne[0] != 256 || dst->ne[0] != 256)) return false;
    if (k_is_packed16_i32 && (dst->ne[0] != Q->ne[0])) return false;
    if (Q->ne[1] <= 2) {
        return false;
    }
    if (K->ne[1] < Q->ne[1] || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] < K->ne[1] || V->ne[2] != K->ne[2] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    return true;
}

static inline bool ggml_cuda_q8k_dot4_packed16_k_cache_enabled() {
    const char * env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE");
    return env && atoi(env) != 0;
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
}

inline void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("q8k_dot4_kq requires HIP");
}

#endif // GGML_USE_HIP
