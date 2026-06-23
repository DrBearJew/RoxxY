#pragma once

#include "common.cuh"

#include <cstdlib>

static inline bool ggml_cuda_q8k_dot4_packed16_k_cache_enabled() {
    // PDMQ compressed-K cache is default-enabled on HIP. Disable with
    // GGML_CUDA_ROCM_PDMQ_K_CACHE=0 or legacy GGML_CUDA_ROCM_PACKED16_DISABLE=1.
    {
        const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DISABLE");
        if (v && atoi(v) != 0) return false;
    }
    {
        const char * v = getenv("GGML_CUDA_ROCM_PDMQ_K_CACHE");
        if (!v) {
            v = getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE"); // legacy alias
        }
        if (v && atoi(v) == 0) return false;
    }
#ifdef GGML_USE_HIP
    return true;
#else
    return false;
#endif
}
