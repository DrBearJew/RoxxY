// fattn-packed16-dot4-mmq.cuh — declarations for packed16 DOT4/MMQ FlashAttention
//
// The heavy HIP implementation lives in fattn-packed16-dot4-mmq-impl.cuh and is
// included only by fattn-packed16-dot4-mmq.cu. Keep this header small so edits to
// the DOT4/MMQ implementation do not force recompilation of fattn.cu.

#pragma once

#include "common.cuh"
#include "dot4-packed16/dp16-trace.cuh"

#if defined(GGML_USE_HIP)

bool ggml_cuda_packed16_dot4_mmq_enabled();
bool ggml_cuda_packed16_dot4_mmq_v_supported(ggml_type type);
bool ggml_cuda_packed16_dot4_mmq_supported(int cc, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

#else

static inline bool ggml_cuda_packed16_dot4_mmq_enabled() {
    return false;
}

static inline bool ggml_cuda_packed16_dot4_mmq_v_supported(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_F16;
}

static inline bool ggml_cuda_packed16_dot4_mmq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
}

static inline void ggml_cuda_flash_attn_ext_packed16_dot4_mmq(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("packed16_dot4_mmq requires HIP");
}

#endif // GGML_USE_HIP
