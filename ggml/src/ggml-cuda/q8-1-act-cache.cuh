#pragma once

#include "common.cuh"

void ggml_cuda_q8_1_act_cache_reset();

bool ggml_cuda_q8_1_act_cache_try_get(
        const ggml_tensor * src1,
        int64_t ne0_padded,
        size_t bytes,
        cudaStream_t stream,
        const char * route,
        const char * tensor_name,
        block_q8_1 ** out,
        bool * needs_quantize);

void ggml_cuda_q8_1_act_cache_mark_valid(
        const char * route,
        const char * tensor_name);
