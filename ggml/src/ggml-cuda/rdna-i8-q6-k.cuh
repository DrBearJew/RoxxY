#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_rdna_i8_q6_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_mul_mat_rdna_i8_q6_K(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst);
