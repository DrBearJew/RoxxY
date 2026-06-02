// dp16-mmvq-packed16-gemv.cuh — persistent packed16/I32 scaled DOT4 MMVQ backend
#pragma once

#include "../dp16-common.cuh"

// Narrow first packed16 MMVQ backend:
//   B/weights: persistent packed16_i32_scaled sidecar
//   A/activation: transient q8_1 blocks produced by the existing MMVQ quantizer
//   N: 1..16 columns/tokens (wide-N path is used by opt-in packed QKV/FFN routes)
//   K: multiple of 256 (enforced by the host planner)
//   Fusion/IDs: not supported in this first lane

static constexpr int DP16_MMVQ_PACKED16_DOT4_WARP_SIZE = 32;

static inline bool dp16_mmvq_packed16_use_i32_lane_kernel() {
    const char * v = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ_KERNEL");
    return v && strcmp(v, "i32_lane") == 0;
}

static inline const char * dp16_mmvq_packed16_kernel_name() {
    return dp16_mmvq_packed16_use_i32_lane_kernel()
        ? "dp16_mmvq_packed16_i32_n1_16_k256"
        : "dp16_mmvq_packed16_i32_b32_n1_16_k256";
}

template<int NCOLS_DST>
__launch_bounds__(NCOLS_DST*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_packed16_i32_n1_16_k256_kernel(
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 DOT4 is N=1..16 only");

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
static __global__ void dp16_mmvq_packed16_i32_b32_n1_16_k256_kernel(
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 DOT4 is N=1..16 only");

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
__launch_bounds__(NCOLS_DST*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_packed16_i32_b32_fusion_n1_16_k256_kernel(
        const int32_t * __restrict__ w_payload,
        const half    * __restrict__ w_scales,
        const block_q8_1 * __restrict__ y,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 fusion is N=1..16 only");

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
        const uint32_t out_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row;
        float result = acc;
        if (fusion.x_bias) {
            result += ((const float *) fusion.x_bias)[out_offset];
        }
        dst[out_offset] = result;
    }
}

template<int NCOLS_DST>
static inline void dp16_mmvq_packed16_i32_b32_fusion_n1_16_k256_launch(
        const dp16_packed16_weight_view & w16,
        const void * vy_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 fusion is N=1..16 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_b32_fusion_n1_16_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        w16.payload,
        w16.scales,
        (const block_q8_1 *) vy_q8_1,
        fusion,
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

template<int NCOLS_DST, int ROWS_TILE, bool HAS_XBIAS, int QB_TILE = DP16_MMVQ_PACKED16_LDS_M2N_QB_TILE>
__launch_bounds__(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE*ROWS_TILE*NCOLS_DST, 1)
static __global__ void dp16_mmvq_packed16_i32_b32_lds_mtile_n_k256_kernel(
        const int32_t * __restrict__ w_payload,
        const half    * __restrict__ w_scales,
        const block_q8_1 * __restrict__ y,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 2 && NCOLS_DST <= DP16_MMVQ_PACKED16_LDS_M2N_FORCE_MAX_N,
            "DP16 MMVQ packed16 LDS M2xN prototype is N=2..5 only");
    static_assert(ROWS_TILE == DP16_MMVQ_PACKED16_LDS_M2N_ROWS_TILE,
            "DP16 MMVQ packed16 LDS M2xN prototype uses ROWS_TILE=2");
    static_assert(QB_TILE == DP16_MMVQ_PACKED16_LDS_M2N_QB_TILE,
            "DP16 MMVQ packed16 LDS M2xN prototype uses QB_TILE=32");

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int row_local = wave / NCOLS_DST;
    const int col = wave - row_local*NCOLS_DST;

    const uint32_t row = blockIdx.x*ROWS_TILE + row_local;
    const bool row_valid = row < nrows_x;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const uint32_t blocks_per_row = k_cols >> 5;

    extern __shared__ unsigned char smem_raw[];
    int32_t * s_w  = (int32_t *) smem_raw;
    float   * s_ws = (float *) (s_w + ROWS_TILE*QB_TILE*(QK8_0/4));
    int32_t * s_a  = (int32_t *) (s_ws + ROWS_TILE*QB_TILE);
    float   * s_as = (float *) (s_a + NCOLS_DST*QB_TILE*(QK8_0/4));

    const int32_t * w_row = w_payload +
        sample_x*w_payload_stride_sample_i32 +
        channel_x*w_payload_stride_channel_i32 +
        row*w_payload_stride_row_i32;
    const half * scale_row = w_scales +
        sample_x*w_scale_stride_sample_half +
        channel_x*w_scale_stride_channel_half +
        row*w_scale_stride_row_half;
    const block_q8_1 * y_base = y +
        sample_y*stride_sample_y + channel_y*stride_channel_y;

    float acc = 0.0f;

    // LDS M2xN: stage two weight rows and N activation columns per 32-qblock K tile.
    // Each wave still owns one output (row,col), so accumulator pressure is one float.
    for (uint32_t qb0 = 0; qb0 < blocks_per_row; qb0 += QB_TILE) {
        const uint32_t qb = qb0 + lane;
        const bool q_valid = qb < blocks_per_row;

        if (col == 0 && row_valid && q_valid) {
            const int32_t * wp = w_row + qb*(QK8_0/4);
            const int base = (row_local*QB_TILE + lane)*(QK8_0/4);
#pragma unroll
            for (int i = 0; i < QK8_0/4; ++i) {
                s_w[base + i] = wp[i];
            }
            s_ws[row_local*QB_TILE + lane] = __half2float(scale_row[qb]);
        }

        if (row_local == 0 && q_valid) {
            const block_q8_1 & by = y_base[col*stride_col_y + qb];
            const int32_t * ap = (const int32_t *) by.qs;
            const int base = (col*QB_TILE + lane)*(QK8_0/4);
#pragma unroll
            for (int i = 0; i < QK8_0/4; ++i) {
                s_a[base + i] = ap[i];
            }
            s_as[col*QB_TILE + lane] = __low2float(by.ds);
        }

        __syncthreads();

        if (row_valid && q_valid) {
            const int w_base = (row_local*QB_TILE + lane)*(QK8_0/4);
            const int a_base = (col*QB_TILE + lane)*(QK8_0/4);
            int dot = 0;
#pragma unroll
            for (int i = 0; i < QK8_0/4; ++i) {
                dot = ggml_cuda_dp4a(s_w[w_base + i], s_a[a_base + i], dot);
            }
            acc += (float) dot * s_ws[row_local*QB_TILE + lane] * s_as[col*QB_TILE + lane];
        }

        __syncthreads();
    }

    const float sum = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(acc);
    if (lane == 0 && row_valid) {
        const uint32_t out_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row;
        float result = sum;
        if constexpr (HAS_XBIAS) {
            if (fusion.x_bias) {
                result += ((const float *) fusion.x_bias)[out_offset];
            }
        }
        dst[out_offset] = result;
    }
}

template<int NCOLS_DST, bool HAS_XBIAS>
static inline void dp16_mmvq_packed16_i32_b32_lds_m2n_k256_launch(
        const dp16_packed16_weight_view & w16,
        const void * vy_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    constexpr int ROWS_TILE = DP16_MMVQ_PACKED16_LDS_M2N_ROWS_TILE;
    constexpr int QB_TILE = DP16_MMVQ_PACKED16_LDS_M2N_QB_TILE;
    static_assert(NCOLS_DST >= 2 && NCOLS_DST <= DP16_MMVQ_PACKED16_LDS_M2N_FORCE_MAX_N,
            "DP16 MMVQ packed16 LDS M2xN prototype is N=2..5 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, ROWS_TILE*NCOLS_DST, 1);
    const dim3 grid((nrows_x + ROWS_TILE - 1) / ROWS_TILE, nchannels_dst, nsamples_dst);
    const size_t lds_bytes =
        ROWS_TILE*QB_TILE*(QK8_0/4)*sizeof(int32_t) +
        ROWS_TILE*QB_TILE*sizeof(float) +
        NCOLS_DST*QB_TILE*(QK8_0/4)*sizeof(int32_t) +
        NCOLS_DST*QB_TILE*sizeof(float);

    dp16_mmvq_packed16_i32_b32_lds_mtile_n_k256_kernel<NCOLS_DST, ROWS_TILE, HAS_XBIAS, QB_TILE><<<grid, block, lds_bytes, stream>>>(
        w16.payload,
        w16.scales,
        (const block_q8_1 *) vy_q8_1,
        fusion,
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

template<int NCOLS_DST, bool HAS_XBIAS>
__launch_bounds__(DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 2)
static __global__ void dp16_mmvq_packed16_i32_b32_reuse_n_k256_kernel(
        const int32_t * __restrict__ w_payload,
        const half    * __restrict__ w_scales,
        const block_q8_1 * __restrict__ y,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_REUSE_N_MAX_N,
            "DP16 MMVQ packed16 reuse-N DOT4 is N=1..8 only");

    const int lane = threadIdx.x;
    const int row_local = threadIdx.y;
    if (row_local >= DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK) {
        return;
    }

    const uint32_t row = blockIdx.x*DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK + row_local;
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
    const block_q8_1 * y_base = y +
        sample_y*stride_sample_y + channel_y*stride_channel_y;

    const uint32_t blocks_per_row = k_cols >> 5;
    float acc[NCOLS_DST];
#pragma unroll
    for (int col = 0; col < NCOLS_DST; ++col) {
        acc[col] = 0.0f;
    }

    // One wave owns one output row and streams all N activation columns against
    // the same packed weight row. This removes the repeated per-column weight
    // loads used by the baseline block=(32,N,1) packed16 kernel.
    for (uint32_t qb = lane; qb < blocks_per_row; qb += DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        const int32_t * wp = w_row + qb*(QK8_0/4);
        const float w_scale = __half2float(s_row[qb]);

#pragma unroll
        for (int col = 0; col < NCOLS_DST; ++col) {
            const block_q8_1 & by = y_base[col*stride_col_y + qb];
            const int32_t * ap = (const int32_t *) by.qs;

            int dot = 0;
#pragma unroll
            for (int i = 0; i < QK8_0/4; ++i) {
                dot = ggml_cuda_dp4a(wp[i], ap[i], dot);
            }

            acc[col] += (float) dot * w_scale * __low2float(by.ds);
        }
    }

#pragma unroll
    for (int col = 0; col < NCOLS_DST; ++col) {
        const float sum = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(acc[col]);
        if (lane == 0) {
            const uint32_t out_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row;
            float result = sum;
            if constexpr (HAS_XBIAS) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[out_offset];
                }
            }
            dst[out_offset] = result;
        }
    }
}

template<int NCOLS_DST, bool HAS_XBIAS>
static inline void dp16_mmvq_packed16_i32_b32_reuse_n_k256_launch(
        const dp16_packed16_weight_view & w16,
        const void * vy_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_REUSE_N_MAX_N,
            "DP16 MMVQ packed16 reuse-N DOT4 is N=1..8 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK, 1);
    const dim3 grid((nrows_x + DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK - 1) /
            DP16_MMVQ_PACKED16_REUSE_N_ROWS_PER_BLOCK, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_b32_reuse_n_k256_kernel<NCOLS_DST, HAS_XBIAS><<<grid, block, 0, stream>>>(
        w16.payload,
        w16.scales,
        (const block_q8_1 *) vy_q8_1,
        fusion,
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
__launch_bounds__(NCOLS_DST*DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, 1)
static __global__ void dp16_mmvq_packed16_i32_b32_fused_glu_n1_16_k256_kernel(
        const int32_t * __restrict__ up_payload,
        const half    * __restrict__ up_scales,
        const int32_t * __restrict__ gate_payload,
        const half    * __restrict__ gate_scales,
        const block_q8_1 * __restrict__ y,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t k_cols,
        const uint32_t nrows_x,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const uint32_t up_payload_stride_row_i32,
        const uint32_t up_scale_stride_row_half,
        const uint32_t gate_payload_stride_row_i32,
        const uint32_t gate_scale_stride_row_half,
        const uint32_t stride_col_y,
        const uint32_t stride_col_dst,
        const uint32_t up_payload_stride_channel_i32,
        const uint32_t up_scale_stride_channel_half,
        const uint32_t gate_payload_stride_channel_i32,
        const uint32_t gate_scale_stride_channel_half,
        const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst,
        const uint32_t up_payload_stride_sample_i32,
        const uint32_t up_scale_stride_sample_half,
        const uint32_t gate_payload_stride_sample_i32,
        const uint32_t gate_scale_stride_sample_half,
        const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst) {
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 fused GLU is N=1..16 only");

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

    const int32_t * up_row = up_payload +
        sample_x*up_payload_stride_sample_i32 +
        channel_x*up_payload_stride_channel_i32 +
        row*up_payload_stride_row_i32;
    const half * up_s_row = up_scales +
        sample_x*up_scale_stride_sample_half +
        channel_x*up_scale_stride_channel_half +
        row*up_scale_stride_row_half;
    const int32_t * gate_row = gate_payload +
        sample_x*gate_payload_stride_sample_i32 +
        channel_x*gate_payload_stride_channel_i32 +
        row*gate_payload_stride_row_i32;
    const half * gate_s_row = gate_scales +
        sample_x*gate_scale_stride_sample_half +
        channel_x*gate_scale_stride_channel_half +
        row*gate_scale_stride_row_half;
    const block_q8_1 * y_col = y +
        sample_y*stride_sample_y + channel_y*stride_channel_y + col*stride_col_y;

    const uint32_t blocks_per_row = k_cols >> 5;
    float up_acc = 0.0f;
    float gate_acc = 0.0f;

    // Persistent packed-i32 sidecars: one wave lane handles one 32-value K block
    // for both FFN branches, reusing the same q8_1 activation block.
    for (uint32_t qb = lane; qb < blocks_per_row; qb += DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        const int32_t * up = up_row + qb*(QK8_0/4);
        const int32_t * gt = gate_row + qb*(QK8_0/4);
        const block_q8_1 & by = y_col[qb];
        const int32_t * ap = (const int32_t *) by.qs;

        int up_dot = 0;
        int gt_dot = 0;
#pragma unroll
        for (int i = 0; i < QK8_0/4; ++i) {
            up_dot = ggml_cuda_dp4a(up[i], ap[i], up_dot);
            gt_dot = ggml_cuda_dp4a(gt[i], ap[i], gt_dot);
        }

        const float a_scale = __low2float(by.ds);
        up_acc   += (float) up_dot * __half2float(up_s_row[qb]) * a_scale;
        gate_acc += (float) gt_dot * __half2float(gate_s_row[qb]) * a_scale;
    }

    up_acc = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(up_acc);
    gate_acc = warp_reduce_sum<DP16_MMVQ_PACKED16_DOT4_WARP_SIZE>(gate_acc);
    if (lane == 0) {
        const uint32_t out_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + col*stride_col_dst + row;
        float up_value = up_acc;
        float gate_value = gate_acc;
        if (fusion.x_bias) {
            up_value += ((const float *) fusion.x_bias)[out_offset];
        }
        if (fusion.gate_bias) {
            gate_value += ((const float *) fusion.gate_bias)[out_offset];
        }

        float result;
        switch (fusion.glu_op) {
            case GGML_GLU_OP_SWIGLU:     result = up_value * ggml_cuda_op_silu_single(gate_value); break;
            case GGML_GLU_OP_GEGLU:      result = up_value * ggml_cuda_op_gelu_single(gate_value); break;
            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, up_value); break;
            default:                     result = up_value * gate_value; break;
        }
        dst[out_offset] = result;
    }
}

template<int NCOLS_DST>
static inline void dp16_mmvq_packed16_i32_b32_fused_glu_n1_16_k256_launch(
        const dp16_packed16_weight_view & up16,
        const dp16_packed16_weight_view & gate16,
        const void * vy_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion,
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 fused GLU is N=1..16 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_b32_fused_glu_n1_16_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
        up16.payload,
        up16.scales,
        gate16.payload,
        gate16.scales,
        (const block_q8_1 *) vy_q8_1,
        fusion,
        dst,
        (uint32_t) ncols_x,
        (uint32_t) nrows_x,
        channel_ratio,
        sample_ratio,
        (uint32_t) up16.payload_stride_row_i32,
        (uint32_t) up16.scale_stride_row_half,
        (uint32_t) gate16.payload_stride_row_i32,
        (uint32_t) gate16.scale_stride_row_half,
        (uint32_t) stride_col_y,
        (uint32_t) stride_col_dst,
        (uint32_t) up16.payload_stride_channel_i32,
        (uint32_t) up16.scale_stride_channel_half,
        (uint32_t) gate16.payload_stride_channel_i32,
        (uint32_t) gate16.scale_stride_channel_half,
        (uint32_t) stride_channel_y,
        (uint32_t) stride_channel_dst,
        (uint32_t) up16.payload_stride_sample_i32,
        (uint32_t) up16.scale_stride_sample_half,
        (uint32_t) gate16.payload_stride_sample_i32,
        (uint32_t) gate16.scale_stride_sample_half,
        (uint32_t) stride_sample_y,
        (uint32_t) stride_sample_dst);
}

template<int NCOLS_DST>
static inline void dp16_mmvq_packed16_i32_n1_16_k256_launch(
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 DOT4 is N=1..16 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_n1_16_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
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
static inline void dp16_mmvq_packed16_i32_b32_n1_16_k256_launch(
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
    static_assert(NCOLS_DST >= 1 && NCOLS_DST <= DP16_MMVQ_PACKED16_MAX_N, "DP16 MMVQ packed16 DOT4 is N=1..16 only");

    const dim3 block(DP16_MMVQ_PACKED16_DOT4_WARP_SIZE, NCOLS_DST, 1);
    const dim3 grid(nrows_x, nchannels_dst, nsamples_dst);
    dp16_mmvq_packed16_i32_b32_n1_16_k256_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
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
