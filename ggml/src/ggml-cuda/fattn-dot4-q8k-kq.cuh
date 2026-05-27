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
static inline bool ggml_cuda_q8k_dot4_kq_route_required() {
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    return required && strcmp(required, "rocm_q8k_dot4_kq") == 0;
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
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 ||
            V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || dst->ne[0] != 256 || Q->ne[1] <= 2) {
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
