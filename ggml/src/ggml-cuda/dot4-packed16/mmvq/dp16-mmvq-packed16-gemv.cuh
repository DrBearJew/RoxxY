// dp16-mmvq-packed16-gemv.cuh — persistent packed16/I32 scaled DOT4 MMVQ backend
#pragma once

#include "../dp16-common.cuh"

// Narrow first packed16 MMVQ backend:
//   B/weights: persistent packed16_i32_scaled sidecar
//   A/activation: transient q8_1 blocks produced by the existing MMVQ quantizer
//   N: 1..4 columns/tokens
//   K: multiple of 256 (enforced by the host planner)
//   Fusion/IDs: not supported in this first lane

static constexpr int DP16_MMVQ_PACKED16_DOT4_WARP_SIZE = 32;

static inline bool dp16_mmvq_packed16_use_i32_lane_kernel() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_KERNEL");
    return v && strcmp(v, "i32_lane") == 0;
}

static inline const char * dp16_mmvq_packed16_kernel_name() {
    return dp16_mmvq_packed16_use_i32_lane_kernel()
        ? "dp16_mmvq_packed16_i32_n1_4_k256"
        : "dp16_mmvq_packed16_i32_b32_n1_4_k256";
}

template<int NCOLS_DST>
__launch_bounds__(NCOLS_DST*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_packed16_i32_n1_4_k256_kernel(
        const int32_t * __restrict__ w_payload,
        const half    * __restrict__ w_scales,
        const block_q8_1 * __restrict__ y,
        float * __restrict__ dst,
        const uint32_t k_cols,
        const uint32_t nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const uint32_t w_payload_stride_row_i32,
        const uint32_t w_scale_stride_row_half,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint32_t w_payload_stride_channel_i32,
        const uint32_t w_scale_stride_channel_half,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint32_t w_payload_stride_sample_i32,
        const uint32_t w_scale_stride_sample_half,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ packed16 DOT4 is N=1..4 only");

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

    const int32_t * w_row = w_payload +
        sample_x*w_payload_stride_sample_i32 +
        channel_x*w_payload_stride_channel_i32 +
        row*w_payload_stride_row_i32;
    const half * s_row = w_scales +
        sample_x*w_scale_stride_sample_half +
        channel_x*w_scale_stride_channel_half +
        row*w_scale_stride_row_half;
    const block_q8_1 * y_col = y +
        sample_y*stride_sample_y + channel_y*stride_channel_y + col*stride_col_y;

    const uint32_t n_i32 = k_cols >> 2;
    float acc = 0.0f;

    // Each lane owns one int32 = 4 signed int8 values. A full wave32 covers 128 K values.
    for (uint32_t ki4 = lane; ki4 < n_i32; ki4 += DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        const uint32_t qb = ki4 >> 3; // 8 int32 packs per q8 block / scale block
        const uint32_t qi = ki4 & 7;

        const int32_t wv = w_row[ki4];
        const block_q8_1 & by = y_col[qb];
        const int32_t av = ((const int32_t *) by.qs)[qi];
        const int dot = ggml_cuda_dp4a(wv, av, 0);

        acc += (float) dot * __half2float(s_row[qb]) * __low2float(by.ds);
    }

    acc = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(acc);
    if (lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] = acc;
    }
}

template<int NCOLS_DST>
__launch_bounds__(NCOLS_DST*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_packed16_i32_b32_n1_4_k256_kernel(
        const int32_t * __restrict__ w_payload,
        const half    * __restrict__ w_scales,
        const block_q8_1 * __restrict__ y,
        float * __restrict__ dst,
        const uint32_t k_cols,
        const uint32_t nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const uint32_t w_payload_stride_row_i32,
        const uint32_t w_scale_stride_row_half,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint32_t w_payload_stride_channel_i32,
        const uint32_t w_scale_stride_channel_half,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint32_t w_payload_stride_sample_i32,
        const uint32_t w_scale_stride_sample_half,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ packed16 DOT4 is N=1..4 only");

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

    const int32_t * w_row = w_payload +
        sample_x*w_payload_stride_sample_i32 +
        channel_x*w_payload_stride_channel_i32 +
        row*w_payload_stride_row_i32;
    const half * s_row = w_scales +
        sample_x*w_scale_stride_sample_half +
        channel_x*w_scale_stride_channel_half +
        row*w_scale_stride_row_half;
    const block_q8_1 * y_col = y +
        sample_y*stride_sample_y + channel_y*stride_channel_y + col*stride_col_y;

    const uint32_t blocks_per_row = k_cols >> 5;
    float acc = 0.0f;

    // Each lane owns one 32-value block: 8 packed int32 DOT4 ops + one scale pair.
    for (uint32_t qb = lane; qb < blocks_per_row; qb += DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        const int32_t * wp = w_row + qb*(QK8_0/4);
        const block_q8_1 & by = y_col[qb];
        const int32_t * ap = (const int32_t *) by.qs;

        int dot = 0;
#pragma unroll
        for (int i = 0; i < QK8_0/4; ++i) {
            dot = ggml_cuda_dp4a(wp[i], ap[i], dot);
        }

        acc += (float) dot * __half2float(s_row[qb]) * __low2float(by.ds);
    }

    acc = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(acc);
    if (lane == 0) {
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row] = acc;
    }
}

template<int NCOLS_DST>
static inline void dp16_mmvq_packed16_i32_n1_4_k256_launch(
        const dp16_packed16_weight_view & w16,
        const void * vy_q8_1,
        float * dst,
        const int ncols_x,
        const int nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const int stride_col_y,
        const int stride_col_dst,
        const int nchannels_dst,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int nsamples_dst,
        const int stride_sample_y,
        const int stride_sample_dst,
        cudaStream_t stream) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ packed16 DOT4 is N=1..4 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_n1_4_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        w16.payload,
        w16.scales,
        (const block_q8_1 *) vy_q8_1,
        dst,
        (uint32_t) ncols_x,
        (uint32_t) nrows_x,
        channel_ratio,
        sample_ratio,
        (uint32_t) w16.payload_stride_row_i32,
        (uint32_t) w16.scale_stride_row_half,
        (uint32_t) stride_col_y,
        (uint32_t) stride_col_dst,
        (uint32_t) w16.payload_stride_channel_i32,
        (uint32_t) w16.scale_stride_channel_half,
        (uint32_t) stride_channel_y,
        (uint32_t) stride_channel_dst,
        (uint32_t) w16.payload_stride_sample_i32,
        (uint32_t) w16.scale_stride_sample_half,
        (uint32_t) stride_sample_y,
        (uint32_t) stride_sample_dst);
}

template<int NCOLS_DST>
static inline void dp16_mmvq_packed16_i32_b32_n1_4_k256_launch(
        const dp16_packed16_weight_view & w16,
        const void * vy_q8_1,
        float * dst,
        const int ncols_x,
        const int nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const int stride_col_y,
        const int stride_col_dst,
        const int nchannels_dst,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int nsamples_dst,
        const int stride_sample_y,
        const int stride_sample_dst,
        cudaStream_t stream) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= 4, "DP16 MMVQ packed16 DOT4 is N=1..4 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_b32_n1_4_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        w16.payload,
        w16.scales,
        (const block_q8_1 *) vy_q8_1,
        dst,
        (uint32_t) ncols_x,
        (uint32_t) nrows_x,
        channel_ratio,
        sample_ratio,
        (uint32_t) w16.payload_stride_row_i32,
        (uint32_t) w16.scale_stride_row_half,
        (uint32_t) stride_col_y,
        (uint32_t) stride_col_dst,
        (uint32_t) w16.payload_stride_channel_i32,
        (uint32_t) w16.scale_stride_channel_half,
        (uint32_t) stride_channel_y,
        (uint32_t) stride_channel_dst,
        (uint32_t) w16.payload_stride_sample_i32,
        (uint32_t) w16.scale_stride_sample_half,
        (uint32_t) stride_sample_y,
        (uint32_t) stride_sample_dst);
}
