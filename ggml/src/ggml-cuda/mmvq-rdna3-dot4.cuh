#pragma once

#include "common.cuh"
#include "unary.cuh"

#include <cstdint>
#include <cstdlib>

#define GGML_CUDA_RDNA3_MMVQ_DOT4_ENV          "GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4"
#define GGML_CUDA_RDNA3_MMVQ_DOT4_SIDEBAND_ENV "GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_SIDEBAND"
#define GGML_CUDA_RDNA3_MMVQ_DOT4_IDS_ENV      "GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_IDS"
#define GGML_CUDA_RDNA3_MMVQ_DOT4_LOG_ENV      "GGML_CUDA_ROCM_RDNA3_MMVQ_DOT4_LOG"

static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE = 32;
static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_WARPS_PER_ROW = 2;
static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_ROWS_PER_BLOCK = 1;
static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_Q8_BLOCKS = QK_K / QK8_1;
static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS = 16;
static constexpr int GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS_PAIR = 4;

static inline bool ggml_cuda_rdna3_mmvq_dot4_env_enabled() {
    static const bool enabled = [](){ const char * e = getenv(GGML_CUDA_RDNA3_MMVQ_DOT4_ENV); return e && atoi(e) != 0; }();
    return enabled;
}

static inline bool ggml_cuda_rdna3_mmvq_dot4_sideband_enabled() {
    static const bool enabled = [](){ const char * e = getenv(GGML_CUDA_RDNA3_MMVQ_DOT4_SIDEBAND_ENV); return e == nullptr || atoi(e) != 0; }();
    return enabled;
}

static inline bool ggml_cuda_rdna3_mmvq_dot4_ids_enabled() {
    static const bool enabled = [](){ const char * e = getenv(GGML_CUDA_RDNA3_MMVQ_DOT4_IDS_ENV); return e && atoi(e) != 0; }();
    return enabled;
}

static inline bool ggml_cuda_rdna3_mmvq_dot4_log_enabled() {
    static const bool enabled = [](){ const char * e = getenv(GGML_CUDA_RDNA3_MMVQ_DOT4_LOG_ENV); return e && atoi(e) != 0; }();
    return enabled;
}

static inline bool ggml_cuda_rdna3_mmvq_dot4_wants_q8sum4(
        const ggml_type type, const int cc, const int warp_size, const int ncols_dst,
        const bool has_ids, const int ncols_x, const int nrows_x) {
#if defined(GGML_USE_HIP)
    return ggml_cuda_rdna3_mmvq_dot4_env_enabled() && ggml_cuda_rdna3_mmvq_dot4_sideband_enabled() &&
           GGML_CUDA_CC_IS_RDNA3_0(cc) && warp_size == GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE &&
           type == GGML_TYPE_Q4_K && ncols_dst == 1 && (!has_ids || ggml_cuda_rdna3_mmvq_dot4_ids_enabled()) &&
           ncols_x % QK_K == 0 && nrows_x > 0;
#else
    GGML_UNUSED_VARS(type, cc, warp_size, ncols_dst, has_ids, ncols_x, nrows_x);
    return false;
#endif
}

static inline bool ggml_cuda_rdna3_mmvq_dot4_supported(
        const ggml_type type, const int cc, const int warp_size, const int ncols_dst,
        const bool has_fusion, const bool has_ids, const int ncols_x, const int nrows_x,
        const int32_t * q8sum4) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(has_fusion);
    const bool base = ggml_cuda_rdna3_mmvq_dot4_env_enabled() && GGML_CUDA_CC_IS_RDNA3_0(cc) &&
           warp_size == GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE && type == GGML_TYPE_Q4_K && ncols_dst == 1 &&
           (!has_ids || ggml_cuda_rdna3_mmvq_dot4_ids_enabled()) && ncols_x % QK_K == 0 && nrows_x > 0;
    return base && (!ggml_cuda_rdna3_mmvq_dot4_sideband_enabled() || q8sum4 != nullptr);
#else
    GGML_UNUSED_VARS(type, cc, warp_size, ncols_dst, has_fusion, has_ids, ncols_x, nrows_x, q8sum4);
    return false;
#endif
}

#if defined(GGML_USE_HIP)

static_assert(QK_K == 256, "RDNA3 MMVQ DOT4 V3 assumes Q4_K blocks with 256 elements");
static_assert(QK8_1 == 32, "RDNA3 MMVQ DOT4 V3 assumes Q8_1 blocks with 32 elements");

static __global__ void ggml_cuda_rdna3_q8_1_sum4_kernel(const block_q8_1 * y, int32_t * q8sum4, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int * q8 = (const int *) y[i].qs;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        q8sum4[4*i + l] = ggml_cuda_dp4a(0x01010101, q8[l + 4], ggml_cuda_dp4a(0x01010101, q8[l], 0));
    }
}

static inline void ggml_cuda_rdna3_q8_1_sum4_precompute(const void * vy_q8_1, int32_t * q8sum4, const int64_t n, cudaStream_t stream) {
    if (q8sum4 == nullptr || n <= 0) return;
    constexpr int block_size = 256;
    ggml_cuda_rdna3_q8_1_sum4_kernel<<<(n + block_size - 1) / block_size, block_size, 0, stream>>>(
        (const block_q8_1 *) vy_q8_1, q8sum4, n);
}

static __device__ __forceinline__ void ggml_cuda_rdna3_q4k_get_scale_min_pair(
        const block_q4_K * bx, const int pair, uint8_t (&sc)[2], uint8_t (&m)[2]) {
    const uint16_t * scales = (const uint16_t *) bx->scales;
    uint16_t aux[2];
    if (pair < 2) {
        aux[0] = scales[pair + 0] & 0x3f3f;
        aux[1] = scales[pair + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[pair + 2] >> 0) & 0x0f0f) | ((scales[pair - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[pair + 2] >> 4) & 0x0f0f) | ((scales[pair - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * a = (const uint8_t *) aux;
    sc[0] = a[0]; sc[1] = a[1]; m[0] = a[2]; m[1] = a[3];
}

template <bool USE_Q8SUM4>
static __device__ __forceinline__ float ggml_cuda_rdna3_q4k_q8_1_dot4_seg(
        const void * vx, const block_q8_1 * by, const int32_t * q8sum4, const int kbx, const int seg) {
    const block_q4_K * bx = (const block_q4_K *) vx + kbx;
    const int pair       = seg / GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS_PAIR;
    const int lane4      = seg % GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS_PAIR;
    const int bq8_offset = 2 * pair;

    uint8_t sc[2], m[2];
    ggml_cuda_rdna3_q4k_get_scale_min_pair(bx, pair, sc, m);

    const int * q4 = (const int *) (bx->qs + 16*bq8_offset + 4*lane4);
    const int v0 = q4[0];
    const int v1 = q4[4];

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int q8_block = bq8_offset + i;
        const block_q8_1 * byi = by + q8_block;
        const int * q8 = (const int *) byi->qs + lane4;
        const int u0 = q8[0];
        const int u1 = q8[4];
        const int vi0 = (v0 >> (4*i)) & 0x0f0f0f0f;
        const int vi1 = (v1 >> (4*i)) & 0x0f0f0f0f;
        const int dot_q = ggml_cuda_dp4a(vi1, u1, ggml_cuda_dp4a(vi0, u0, 0));
        const float d8 = __half2float(((const half *) &byi->ds)[0]);
        sumf_d += d8 * (dot_q * sc[i]);

        int dot_m;
        if constexpr (USE_Q8SUM4) {
            dot_m = q8sum4[4*q8_block + lane4];
        } else {
            dot_m = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
        }
        sumf_m += d8 * (dot_m * m[i]);
    }
    const float2 dm = __half22float2(bx->dm);
    return dm.x*sumf_d - dm.y*sumf_m;
}

template <bool HAS_IDS, bool HAS_GATE, bool HAS_X_BIAS, bool HAS_GATE_BIAS, bool USE_Q8SUM4>
__launch_bounds__(GGML_CUDA_RDNA3_MMVQ_DOT4_ROWS_PER_BLOCK*GGML_CUDA_RDNA3_MMVQ_DOT4_WARPS_PER_ROW*GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE, 1)
static __global__ void ggml_cuda_rdna3_mmvq_q4_k_dot4_kernel(
        const void * vx, const void * vy, const int32_t * y_q8sum4, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst, const uint32_t ncols_x, const uint32_t nrows_x,
        const uint3 nchannels_y, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    GGML_UNUSED_VARS(ids_stride, stride_col_y, stride_col_dst);
    constexpr int ROWS_PER_BLOCK = GGML_CUDA_RDNA3_MMVQ_DOT4_ROWS_PER_BLOCK;
    constexpr int WARPS_PER_ROW  = GGML_CUDA_RDNA3_MMVQ_DOT4_WARPS_PER_ROW;
    constexpr int blocks_per_iter = WARPS_PER_ROW * GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE / GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row_local = warp / WARPS_PER_ROW;
    const int warp_in_row = warp - row_local*WARPS_PER_ROW;
    const int row_tid = warp_in_row*GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE + lane;
    const uint32_t row = blockIdx.x*ROWS_PER_BLOCK + row_local;
    const bool active = row < nrows_x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;

    uint32_t channel_x = 0, channel_y = 0, channel_bias = 0, sample_x = 0;
    if (active) {
        if constexpr (HAS_IDS) {
            channel_x = (uint32_t) ids[channel_dst];
            channel_y = fastmodulo(channel_dst, nchannels_y);
            channel_bias = channel_x;
        } else {
            channel_x = fastdiv(channel_dst, channel_ratio);
            channel_y = channel_dst;
            channel_bias = channel_dst;
        }
        sample_x = fastdiv(sample_dst, sample_ratio);
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_dst*stride_sample_y + channel_y*stride_channel_y;
    const int32_t * ysum4 = nullptr;
    if constexpr (USE_Q8SUM4) ysum4 = y_q8sum4 + 4*(sample_dst*stride_sample_y + channel_y*stride_channel_y);

    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;
    const int blocks_per_row = ncols_x / QK_K;
    const int seg = row_tid % GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS;
    const int kbx_start = row_tid / GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_SEGMENTS;

    float acc = 0.0f, acc_gate = 0.0f;
    if (active) {
        for (int kbx = kbx_start; kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kby = kbx * GGML_CUDA_RDNA3_MMVQ_DOT4_Q4K_Q8_BLOCKS;
            const int xk = kbx_offset + kbx;
            const int32_t * sb = USE_Q8SUM4 ? &ysum4[4*kby] : nullptr;
            acc += ggml_cuda_rdna3_q4k_q8_1_dot4_seg<USE_Q8SUM4>(vx, &y[kby], sb, xk, seg);
            if constexpr (HAS_GATE) acc_gate += ggml_cuda_rdna3_q4k_q8_1_dot4_seg<USE_Q8SUM4>(fusion.gate, &y[kby], sb, xk, seg);
        }
    }

    __shared__ float row_acc_shared[1][GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE];
    __shared__ float row_gate_shared[1][GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE];
    if (warp_in_row > 0) {
        row_acc_shared[0][lane] = acc;
        if constexpr (HAS_GATE) row_gate_shared[0][lane] = acc_gate;
    }
    __syncthreads();
    if (warp_in_row != 0) return;

    acc += row_acc_shared[0][lane];
    if constexpr (HAS_GATE) acc_gate += row_gate_shared[0][lane];
    acc = warp_reduce_sum<GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE>(acc);
    if constexpr (HAS_GATE) acc_gate = warp_reduce_sum<GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE>(acc_gate);
    if (lane != 0 || !active) return;

    float result = acc;
    if constexpr (HAS_X_BIAS) {
        const float * x_bias = (const float *) fusion.x_bias;
        result += x_bias[sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row];
    }
    if constexpr (HAS_GATE) {
        float gate_value = acc_gate;
        if constexpr (HAS_GATE_BIAS) {
            const float * gate_bias = (const float *) fusion.gate_bias;
            gate_value += gate_bias[sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row];
        }
        switch (fusion.glu_op) {
            case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(gate_value); break;
            case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(gate_value); break;
            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
            default:                     result *= gate_value; break;
        }
    }
    dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row] = result;
}

template <bool HAS_IDS, bool HAS_GATE, bool HAS_X_BIAS, bool HAS_GATE_BIAS, bool USE_Q8SUM4>
static inline void ggml_cuda_rdna3_mmvq_dot4_launch_kernel(
        const void * vx, const void * vy, const int32_t * y_q8sum4, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst, const int ncols_x, const int nrows_x,
        const uint3 nchannels_y, const uint3 channel_ratio, const uint3 sample_ratio,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, const int ids_stride,
        cudaStream_t stream) {
    const dim3 block_dims(GGML_CUDA_RDNA3_MMVQ_DOT4_WARP_SIZE, GGML_CUDA_RDNA3_MMVQ_DOT4_WARPS_PER_ROW, 1);
    const dim3 block_nums((nrows_x + GGML_CUDA_RDNA3_MMVQ_DOT4_ROWS_PER_BLOCK - 1) / GGML_CUDA_RDNA3_MMVQ_DOT4_ROWS_PER_BLOCK,
                          nchannels_dst, nsamples_dst);
    ggml_cuda_rdna3_mmvq_q4_k_dot4_kernel<HAS_IDS, HAS_GATE, HAS_X_BIAS, HAS_GATE_BIAS, USE_Q8SUM4>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, nchannels_y,
            channel_ratio, sample_ratio, stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y,
            stride_channel_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <bool HAS_IDS, bool USE_Q8SUM4>
static inline void ggml_cuda_rdna3_mmvq_dot4_launch_fusion(
        const void * vx, const void * vy, const int32_t * y_q8sum4, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst, const int ncols_x, const int nrows_x,
        const uint3 nchannels_y, const uint3 channel_ratio, const uint3 sample_ratio,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, const int ids_stride,
        cudaStream_t stream) {
    const bool has_gate = fusion.gate != nullptr;
    const bool has_x_bias = fusion.x_bias != nullptr;
    const bool has_gate_bias = has_gate && fusion.gate_bias != nullptr;
#define GGML_RDNA3_LAUNCH(G, XB, GB) \
    ggml_cuda_rdna3_mmvq_dot4_launch_kernel<HAS_IDS, G, XB, GB, USE_Q8SUM4>(vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, \
        nchannels_y, channel_ratio, sample_ratio, stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, \
        stride_channel_y, stride_channel_dst, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream)
    if (has_gate) {
        if (has_x_bias) {
            if (has_gate_bias) GGML_RDNA3_LAUNCH(true, true, true); else GGML_RDNA3_LAUNCH(true, true, false);
        } else {
            if (has_gate_bias) GGML_RDNA3_LAUNCH(true, false, true); else GGML_RDNA3_LAUNCH(true, false, false);
        }
    } else {
        if (has_x_bias) GGML_RDNA3_LAUNCH(false, true, false); else GGML_RDNA3_LAUNCH(false, false, false);
    }
#undef GGML_RDNA3_LAUNCH
}

template <ggml_type type>
static inline void ggml_cuda_rdna3_mmvq_dot4_launch(
        const void * vx, const void * vy, const int32_t * y_q8sum4, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst, const int ncols_x, const int nrows_x,
        const uint3 nchannels_y, const uint3 channel_ratio, const uint3 sample_ratio,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, const int ids_stride,
        cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        const bool use_sideband = y_q8sum4 != nullptr && ggml_cuda_rdna3_mmvq_dot4_sideband_enabled();
        const bool has_ids = ids != nullptr;
        if (ggml_cuda_rdna3_mmvq_dot4_log_enabled()) {
            GGML_LOG_INFO("%s: route=rdna3_mmvq_dot4_novec_v3 mode=%s ids=%d rows=%d cols=%d channels=%d samples=%d\n",
                __func__, use_sideband ? "q8sum4_sideband" : "exact_min_fallback", has_ids ? 1 : 0, nrows_x, ncols_x, nchannels_dst, nsamples_dst);
        }
#define GGML_RDNA3_TOP(IDS, SB) \
    ggml_cuda_rdna3_mmvq_dot4_launch_fusion<IDS, SB>(vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, channel_ratio, \
        sample_ratio, stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, \
        nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream)
        if (has_ids) { if (use_sideband) GGML_RDNA3_TOP(true, true); else GGML_RDNA3_TOP(true, false); }
        else         { if (use_sideband) GGML_RDNA3_TOP(false, true); else GGML_RDNA3_TOP(false, false); }
#undef GGML_RDNA3_TOP
    } else {
        GGML_UNUSED_VARS(vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, channel_ratio, sample_ratio,
                         stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y,
                         stride_channel_dst, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
    }
}

#else

static inline void ggml_cuda_rdna3_q8_1_sum4_precompute(const void * vy_q8_1, int32_t * q8sum4, const int64_t n, cudaStream_t stream) {
    GGML_UNUSED_VARS(vy_q8_1, q8sum4, n, stream);
}

template <ggml_type type>
static inline void ggml_cuda_rdna3_mmvq_dot4_launch(const void * vx, const void * vy, const int32_t * y_q8sum4, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst, const int ncols_x, const int nrows_x, const uint3 nchannels_y,
        const uint3 channel_ratio, const uint3 sample_ratio, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    GGML_UNUSED_VARS(vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, channel_ratio, sample_ratio,
        stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

#endif
