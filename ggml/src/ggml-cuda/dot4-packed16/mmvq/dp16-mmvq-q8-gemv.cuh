// dp16-mmvq-q8-gemv.cuh — first real DP16 MMVQ q8/signed-i8 DOT4 GEMV backend
#pragma once

#include "../../common.cuh"

// Narrow Stage5 backend:
//   B/weights: persistent q8_0 blocks
//   A/activation: transient q8_1 blocks produced by the existing MMVQ quantizer
//   N: 1..4 columns/tokens
//   K: multiple of 256 (enforced by the host planner)
//   Fusion: supported for N=1 decode FFN/GEMV (bias/gate/GLU)
//   IDs: not supported in this first lane

static constexpr int DP16_MMVQ_Q8_DOT4_WARP_SIZE = 32;

static __device__ __forceinline__ float dp16_mmvq_q8_dot4_accumulate_row(
        const block_q8_0 * x,
        const block_q8_1 * y,
        const uint32_t ncols_x,
        const int lane) {
    const int blocks_per_row = ncols_x / QK8_0;
    float acc = 0.0f;

    for (int kbx = lane; kbx < blocks_per_row; kbx += DP16_MMVQ_Q8_DOT4_WARP_SIZE) {
        const block_q8_0 & bx = x[kbx];
        const block_q8_1 & by = y[kbx];
        const int * qx = (const int *) bx.qs;
        const int * qy = (const int *) by.qs;

        int dot = 0;
#pragma unroll
        for (int i = 0; i < QK8_0 / 4; ++i) {
            dot = ggml_cuda_dp4a(qx[i], qy[i], dot);
        }
        acc += __half2float(bx.d) * __low2float(by.ds) * (float) dot;
    }

    return acc;
}

template<int NCOLS_DST>
__launch_bounds__(NCOLS_DST*DP16_MMVQ_Q8_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_q8_dot4_n1_4_k256_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint32_t stride_channel_x,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint32_t stride_sample_x,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ q8 DOT4 is N=1..4 only");

    const int lane = threadIdx.x;
    const int col  = threadIdx.y;
    if (col >= NCOLS_DST) {
        return;
    }

    const uint32_t row = blockIdx.x;
    if (row >= nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const block_q8_0 * x = ((const block_q8_0 *) vx) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y + col*stride_col_y;

    float acc = dp16_mmvq_q8_dot4_accumulate_row(x, y, ncols_x, lane);
    acc = warp_reduce_sum<DP16_MMVQ_Q8_DOT4_WARP_SIZE>(acc);
    if (lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] = acc;
    }
}

template<int NCOLS_DST>
__launch_bounds__(DP16_MMVQ_Q8_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_q8_dot4_fusion_n1_k256_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint32_t stride_channel_x,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint32_t stride_sample_x,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(NCOLS_DST == 1, "DP16 MMVQ q8 DOT4 fusion is N=1 only");

    const int lane = threadIdx.x;

    const uint32_t row = blockIdx.x;
    if (row >= nrows_x) {
        return;
    }

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const block_q8_0 * x = ((const block_q8_0 *) vx) +
        sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;

    float acc = dp16_mmvq_q8_dot4_accumulate_row(x, y, ncols_x, lane);

    const bool use_gate      = fusion.gate      != nullptr;
    const bool use_bias      = fusion.x_bias    != nullptr;
    const bool use_gate_bias = fusion.gate_bias != nullptr && use_gate;

    float gate_acc = 0.0f;
    if (use_gate) {
        const block_q8_0 * gate = ((const block_q8_0 *) fusion.gate) +
            sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
        gate_acc = dp16_mmvq_q8_dot4_accumulate_row(gate, y, ncols_x, lane);
    }

    acc = warp_reduce_sum<DP16_MMVQ_Q8_DOT4_WARP_SIZE>(acc);
    if (use_gate) {
        gate_acc = warp_reduce_sum<DP16_MMVQ_Q8_DOT4_WARP_SIZE>(gate_acc);
    }

    if (lane == 0) {
        float result = acc;
        const uint32_t out_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row;
        if (use_bias) {
            result += ((const float *) fusion.x_bias)[out_offset];
        }
        if (use_gate) {
            float gate_value = gate_acc;
            if (use_gate_bias) {
                gate_value += ((const float *) fusion.gate_bias)[out_offset];
            }
            switch (fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                    break;
                default:
                    result *= gate_value;
                    break;
            }
        }
        dst[out_offset] = result;
    }
}

template<int NCOLS_DST>
static inline void dp16_mmvq_q8_dot4_fusion_n1_k256_launch(
        const void * vx,
        const void * vy,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * dst,
        const int ncols_x,
        const int nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const int stride_row_x,
        const int stride_col_y,
        const int stride_col_dst,
        const int nchannels_dst,
        const int stride_channel_x,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int nsamples_dst,
        const int stride_sample_x,
        const int stride_sample_y,
        const int stride_sample_dst,
        cudaStream_t stream) {
    static_assert(NCOLS_DST == 1, "DP16 MMVQ q8 DOT4 fusion is N=1 only");
    GGML_UNUSED(stride_col_dst);
    GGML_UNUSED(stride_col_y);
    const dim3 block(DP16_MMVQ_Q8_DOT4_WARP_SIZE, 1, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_q8_dot4_fusion_n1_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        vx, vy, fusion, dst, ncols_x, nrows_x, channel_ratio, sample_ratio,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        stride_sample_x, stride_sample_y, stride_sample_dst);
}

template<int NCOLS_DST>
static inline void dp16_mmvq_q8_dot4_n1_4_k256_launch(
        const void * vx,
        const void * vy,
        float * dst,
        const int ncols_x,
        const int nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const int stride_row_x,
        const int stride_col_y,
        const int stride_col_dst,
        const int nchannels_dst,
        const int stride_channel_x,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int nsamples_dst,
        const int stride_sample_x,
        const int stride_sample_y,
        const int stride_sample_dst,
        cudaStream_t stream) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ q8 DOT4 is N=1..4 only");
    const dim3 block(DP16_MMVQ_Q8_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_q8_dot4_n1_4_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, channel_ratio, sample_ratio,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        stride_sample_x, stride_sample_y, stride_sample_dst);
}
