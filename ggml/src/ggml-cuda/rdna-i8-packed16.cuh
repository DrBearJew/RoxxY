#pragma once

#include "common.cuh"

bool ggml_cuda_should_use_rdna_i8_packed16_gemm(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_mul_mat_rdna_i8_packed16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_should_use_rdna_i8_packed16_gemm_id(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_mul_mat_id_rdna_i8_packed16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

bool ggml_cuda_should_use_rdna_i8_packed16_moe_projection(
        const ggml_tensor * weights,
        const ggml_tensor * compact,
        const ggml_tensor * lanes,
        const ggml_tensor * bounds,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_moe_routed_lanes_projection_rdna_i8_packed16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
