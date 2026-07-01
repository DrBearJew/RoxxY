#pragma once

#include "common.cuh"

#include <cstddef>
#include <cstdint>

#define GGML_CUDA_HIPBLASLT_I8_CHUNK_K 32
#define GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS 8

enum ggml_cuda_hipblaslt_i8_status {
    GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS = 0,
    GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_ENABLED,
    GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED,
    GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT,
    GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_IMPLEMENTED,
    GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW,
    GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR,
};

enum ggml_cuda_hipblaslt_i8_route_kind {
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_0_PREFILL = 0,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q6_K_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q6_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ3_S_MOE_PREFILL,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_0_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q2_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_Q3_K_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_XS_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_NL_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ3_XXS_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XXS_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ1_S_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XS_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_S_MOE_GROUPED,
    GGML_CUDA_HIPBLASLT_I8_ROUTE_DENSE_FFN_GATE_UP_GROUPED,
};

enum ggml_cuda_hipblaslt_i8_weight_type {
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0 = 0,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS,
    GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S,
};

struct ggml_cuda_hipblaslt_i8_plan {
    int m = 0; // src1 columns / tokens
    int n = 0; // dst rows / output columns
    int k = 0; // reduction dim
    int chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
    int stride_a_blocks = 0; // q8_1 blocks per activation row
    int stride_b_blocks = 0; // q8_0 blocks per weight row
    int stride_d = 0;        // dst float stride between token rows
    int batch_count = 1;
    ggml_cuda_hipblaslt_i8_weight_type weight_type = GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0;
    int preferred_algo_ord = -1; // hipBLASLt heuristic ordinal, -1 keeps first returned heuristic
    int heuristic_request_count = 1;
    bool log_heuristic = false;
    bool full_k_stage_k32_exact = false;
    size_t hipblaslt_workspace_bytes = 64ull << 20;
};

struct ggml_cuda_hipblaslt_i8_scratch {
    int8_t  * a_i8 = nullptr; // dense A[M,32] or A[M,K] when full_k_stage_k32_exact
    int8_t  * b_i8 = nullptr; // dense B[32,N] or B[K,N] when full_k_stage_k32_exact
    int32_t * c_i32 = nullptr; // dense C[M,N]
    void    * hipblaslt_workspace = nullptr;
    void    * hipblaslt_user_args = nullptr; // device UserArguments for grouped GEMM

    size_t a_i8_bytes = 0;
    size_t b_i8_bytes = 0;
    size_t c_i32_bytes = 0;
    size_t hipblaslt_workspace_bytes = 0;
    size_t hipblaslt_user_args_bytes = 0;
};

struct ggml_cuda_hipblaslt_i8_args {
    const block_q8_1 * activations_q8_1 = nullptr;
    const block_q8_0 * weights_q8_0 = nullptr;
    const block_q4_0 * weights_q4_0 = nullptr;
    const block_q4_1 * weights_q4_1 = nullptr;
    const block_q5_0 * weights_q5_0 = nullptr;
    const block_q5_1 * weights_q5_1 = nullptr;
    const block_q4_K * weights_q4_K = nullptr;
    const block_q5_K * weights_q5_K = nullptr;
    const block_q8_K * weights_q8_K = nullptr;
    const block_q2_K * weights_q2_K = nullptr;
    const block_q3_K * weights_q3_K = nullptr;
    const block_iq4_xs * weights_iq4_xs = nullptr;
    const block_iq4_nl * weights_iq4_nl = nullptr;
    const block_iq3_xxs * weights_iq3_xxs = nullptr;
    const block_iq2_xxs * weights_iq2_xxs = nullptr;
    const block_iq1_s * weights_iq1_s = nullptr;
    const block_iq2_xs * weights_iq2_xs = nullptr;
    const block_iq2_s * weights_iq2_s = nullptr;
    const block_q6_K * weights_q6_K = nullptr;
    float * dst = nullptr;
    ggml_cuda_hipblaslt_i8_scratch scratch;
    cudaStream_t stream = nullptr;
};

const char * ggml_cuda_hipblaslt_i8_status_name(ggml_cuda_hipblaslt_i8_status status);
const char * ggml_cuda_hipblaslt_i8_route_name(ggml_cuda_hipblaslt_i8_route_kind route);

bool ggml_cuda_hipblaslt_i8_build_enabled();
bool ggml_cuda_hipblaslt_i8_supported();

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_get_scratch_size(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        ggml_cuda_hipblaslt_i8_scratch * scratch_size);

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_validate(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args);

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q8_0(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args);

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q6_K(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args);

bool ggml_cuda_should_use_hipblaslt_i8_q8_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q4_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q4_1_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q5_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q5_1_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q8_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q6_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q6_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q8_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq3s_moe_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q8_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q4_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q4_1_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q5_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q5_1_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q4_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q5_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q2_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_q3_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq4_xs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq4_nl_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq3_xxs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq2_xxs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq1_s_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq2_xs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_should_use_hipblaslt_i8_iq2_s_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc);

bool ggml_cuda_mul_mat_hipblaslt_i8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst);

bool ggml_cuda_mul_mat_hipblaslt_i8_dense_gate_up(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        ggml_tensor * dst);
