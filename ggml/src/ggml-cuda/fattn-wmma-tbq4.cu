// ROCm/rocWMMA fused TBQ4 flash attention — specialized for gfx1100 / RDNA3.
// Reads raw TBQ4_0 K/V directly, dequants multiple rows into warp-local shared half buffer,
// then calls existing rocwmma::load_matrix_sync / mma_sync for the GEMM.
//
// Supports D=128 and D=256. AMD gfx1100 only. V_is_K_view=false.
// Q is pre-rotated (tbq4_rotate_input_cuda), output post-inverse-rotated.
//
// Based on fattn-wmma-f16.cu plus TBQ4 dequant from fattn-mma-tbq4.cuh.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn-mma-tbq4.cuh"
#include <rocwmma/rocwmma.hpp>

namespace wmma = rocwmma;

// get_max_power_of_2 and get_VKQ_stride — duplicated from fattn-wmma-f16.cu
constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

template<int D, int ncols, int nwarps, int VKQ_stride, typename KQ_acc_t, bool use_logit_softcap>
__launch_bounds__(nwarps * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void flash_attn_ext_wmma_tbq4(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_HIP_ROCWMMA_FATTN) && defined(GGML_USE_WMMA_FATTN)

    static_assert(D == 128 || D == 256, "TBQ4 rocWMMA: only D=128 or D=256 supported");

    if (use_logit_softcap) return;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int ic0 = ncols * blockIdx.x;

    static_assert(D <= FATTN_KQ_STRIDE);
    static_assert(ncols == 8 || ncols % 16 == 0);
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0);

    using frag_a_K  = wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, _Float16, wmma::row_major>;
    using frag_a_V  = wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, _Float16, wmma::col_major>;
    using frag_b    = wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, _Float16, wmma::col_major>;
    using frag_c_KQ  = wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>;
    using frag_c_VKQ = wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, _Float16>;

    constexpr int KQ_stride_tc = nwarps * frag_m;
    constexpr int VKQ_ratio = KQ_stride_tc / VKQ_stride;
    static_assert(VKQ_ratio <= nwarps);

    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t) / sizeof(half);

    // TBQ4: each block_tbq4_0 covers 128 values. D/128 blocks per row.
    constexpr int tbq4_blocks_per_row = D / 128;

    const int sequence  = blockIdx.z / ne02;
    const int head      = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const float * Q_f    = (const float *)(Q + nb03 * sequence + nb02 * head + nb01 * ic0);
    const char  * K_b    = K + nb13 * sequence + nb12 * (head / gqa_ratio);
    const char  * V_b    = V + nb13 * sequence + nb12 * (head / gqa_ratio);
    const half  * maskh  = (const half  *)(mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_K_bytes = nb11;
    const int stride_V_bytes = nb21;

    frag_b Q_b[D / 16][ncols / frag_n];

    // ── shared memory ──
    constexpr int mem_KQ = ncols * kqs_padded * kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio * ncols * D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f  = (float *)KQ;
    half2 * KQ2   = (half2 *)KQ;

    __shared__ half VKQ[ncols * D_padded];
    half2 * VKQ2 = (half2 *)VKQ;

    // TBQ4 dequant buffer: one frag_m × D_padded section per warp
    constexpr int tbq4_warp_stride = frag_m * D_padded;
    __shared__ _Float16 tbq4_buf[nwarps * tbq4_warp_stride];

    float KQ_rowsum_f[ncols / nwarps] = {0.0f};
    float KQ_max_f[ncols / nwarps];
#pragma unroll
    for (int j = 0; j < ncols / nwarps; ++j) KQ_max_f[j] = -FLT_MAX / 2.0f;
    float KQ_max_scale_f[ncols / nwarps] = {0.0f};

    // init VKQ accumulator
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D / 2 && i >= D / 2) break;
            VKQ2[j * (D_padded / 2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // ── TBQ4 dequant: frag_m rows → warp-local section of tbq4_buf ──
    // D=128: 1 block_tbq4_0 per row (66 bytes → 128 half)
    // D=256: 2 block_tbq4_0 per row (132 bytes → 256 half)
    auto dequant_tbq4_frag_m_rows = [&](const char * base_ptr, int64_t stride_bytes, int warp_idx, int oob_flag) {
        _Float16 * my_buf = tbq4_buf + warp_idx * tbq4_warp_stride;
        for (int r = 0; r < frag_m; ++r) {
            _Float16 * row_buf = my_buf + r * D_padded;
            if (oob_flag) {
                // Zero-fill entire row for out-of-bounds access
                for (int i = threadIdx.x; i < D; i += warp_size) row_buf[i] = (_Float16)0.0f;
                continue;
            }
            const char * raw_row = base_ptr + int64_t(r) * stride_bytes;
            const block_tbq4_0 * blks = (const block_tbq4_0 *)raw_row;
            for (int b = 0; b < tbq4_blocks_per_row; ++b) {
                const block_tbq4_0 * blk = blks + b;
                const float norm = __half2float(blk->d);
                const int col_offset = b * 128;
                for (int i = threadIdx.x; i < 64; i += warp_size) {
                    const uint8_t byte = blk->qs[i];
                    row_buf[col_offset + 2 * i]     = __float2half(d_tbq4_centroids[byte & 0xF] * norm);
                    row_buf[col_offset + 2 * i + 1] = __float2half(d_tbq4_centroids[byte >> 4] * norm);
                }
            }
        }
        __syncthreads();
    };

    // ── load Q to shared then to fragments ──
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) break;
            KQ[j * D_padded + i] = __float2half((ic0 + j < int(ne01.z)) ? Q_f[j * stride_Q + i] * scale : 0.0f);
        }
    }
    __syncthreads();

#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            wmma::load_matrix_sync(Q_b[i0 / 16][j0 / frag_n],
                (const _Float16 *)(KQ + j0 * D_padded + i0), D_padded);
        }
    }
    __syncthreads();

    // ── main K/V iteration loop ──
    const int k_VKQ_max = KV_max ? KV_max[sequence * gridDim.x + blockIdx.x] : ne11;
    for (int k_VKQ_0 = blockIdx.y * FATTN_KQ_STRIDE; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y * FATTN_KQ_STRIDE) {

        // === KQ ===
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols / frag_n];
#pragma unroll
            for (int j = 0; j < ncols / frag_n; ++j)
                wmma::fill_fragment(KQ_c[j], static_cast<KQ_acc_t>(0.0f));

            const int k_row_first = k_VKQ_0 + i_KQ_0 + frag_m * threadIdx.y;
            const int k_row_oob = (k_row_first >= k_VKQ_max);
            dequant_tbq4_frag_m_rows(
                K_b + int64_t((k_row_oob ? 0 : k_row_first)) * stride_K_bytes,
                stride_K_bytes, threadIdx.y, k_row_oob);

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                _Float16 * my_buf = tbq4_buf + threadIdx.y * tbq4_warp_stride;
                wmma::load_matrix_sync(K_a, my_buf + k_KQ_0, D_padded);
#pragma unroll
                for (int j = 0; j < ncols / frag_n; ++j) {
                    wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0 / 16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync((KQ_acc_t *)KQ + j0 * kqs_padded + i_KQ_0 + frag_m * threadIdx.y,
                                        KQ_c[j0 / frag_n], kqs_padded, wmma::mem_col_major);
            }
        }
        __syncthreads();

        // === softmax (float acc) ===
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            float KQ_f_tmp[FATTN_KQ_STRIDE / warp_size];
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                KQ_f_tmp[k0 / warp_size] = KQ_f[j * kqs_padded + k0 + threadIdx.x];
            }
            float KQ_max_new = KQ_max_f[j0 / nwarps];
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                const int k = k0 + threadIdx.x;
                KQ_f_tmp[k0 / warp_size] += mask && ic0 + j < int(ne01.z) ?
                    __half2float(maskh[j * (nb31 / sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0 / warp_size] + FATTN_KQ_MAX_OFFSET);
            }
            KQ_max_new = warp_reduce_max<warp_size>(KQ_max_new);
            const float diff = KQ_max_f[j0 / nwarps] - KQ_max_new;
            KQ_max_scale_f[j0 / nwarps] = expf(diff);
            if (diff <= SOFTMAX_FTZ_THRESHOLD) KQ_max_scale_f[j0 / nwarps] = 0.0f;
            KQ_max_f[j0 / nwarps] = KQ_max_new;
            float KQ_rowsum_add = 0.0f;
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                const float d = KQ_f_tmp[k0 / warp_size] - KQ_max_f[j0 / nwarps];
                KQ_f_tmp[k0 / warp_size] = expf(d);
                if (d <= SOFTMAX_FTZ_THRESHOLD) KQ_f_tmp[k0 / warp_size] = 0.0f;
                KQ_rowsum_add += KQ_f_tmp[k0 / warp_size];
                KQ[j * (kqar * kqs_padded) + k0 + threadIdx.x] = KQ_f_tmp[k0 / warp_size];
            }
            KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);
            KQ_rowsum_f[j0 / nwarps] = KQ_max_scale_f[j0 / nwarps] * KQ_rowsum_f[j0 / nwarps] + KQ_rowsum_add;
        }
        __syncthreads();

        // === KQ load for VKQ ===
        frag_b KQ_b[FATTN_KQ_STRIDE / (VKQ_ratio * 16)][ncols / frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio * 16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio) * 16;
                wmma::load_matrix_sync(KQ_b[k0 / (VKQ_ratio * 16)][j0 / frag_n],
                    (const _Float16 *)(KQ + j0 * (kqar * kqs_padded) + k), kqar * kqs_padded);
            }
        }

        // === VKQ ===
        frag_c_VKQ VKQ_c[D / VKQ_stride][ncols / frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols / frag_n; ++j)
                wmma::fill_fragment(VKQ_c[i_VKQ_0 / VKQ_stride][j], (_Float16)0.0f);

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio * 16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio) * 16;

                const int v_row_first = k_VKQ_0 + k;
                const int v_row_oob = (v_row_first >= k_VKQ_max);
                dequant_tbq4_frag_m_rows(
                    V_b + int64_t((v_row_oob ? 0 : v_row_first)) * stride_V_bytes,
                    stride_V_bytes, threadIdx.y, v_row_oob);

                frag_a_V v_a;
                _Float16 * my_v_buf = tbq4_buf + threadIdx.y * tbq4_warp_stride;
                wmma::load_matrix_sync(v_a,
                    my_v_buf + i_VKQ_0 + frag_m * (threadIdx.y / VKQ_ratio),
                    D_padded);
#pragma unroll
                for (int j = 0; j < ncols / frag_n; ++j) {
                    wmma::mma_sync(VKQ_c[i_VKQ_0 / VKQ_stride][j], v_a,
                                   KQ_b[k0 / (VKQ_ratio * 16)][j],
                                   VKQ_c[i_VKQ_0 / VKQ_stride][j]);
                }
            }
        }
        __syncthreads();

        // === store VKQ ===
        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols * D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync(
                    (_Float16 *)(KQ + offset_k + j0 * D_padded + i_KQ_0 + frag_m * (threadIdx.y / VKQ_ratio)),
                    VKQ_c[i_KQ_0 / VKQ_stride][j0 / frag_n], D_padded, wmma::mem_col_major);
            }
        }
        __syncthreads();

        // === accumulate VKQ ===
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            const half2 VKQ_scale = make_half2(KQ_max_scale_f[j0 / nwarps], KQ_max_scale_f[j0 / nwarps]);
#pragma unroll
            for (int i0 = 0; i0 < D / 2; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                if (i0 + warp_size > D / 2 && i >= D / 2) break;
                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l)
                    VKQ_add += KQ2[l * (ncols * D_padded / 2) + j * (D_padded / 2) + i];
                VKQ2[j * (D_padded / 2) + i] = VKQ_scale * VKQ2[j * (D_padded / 2) + i] + VKQ_add;
            }
        }
        __syncthreads();
    }

    // ── write output ──
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= int(ne01.z)) return;
        const float KQ_rowsum_j = KQ_rowsum_f[j0 / nwarps];
        const int j_dst_unrolled = ((sequence * int(ne01.z) + ic0 + j_VKQ) * ne02 + head) * gridDim.y + blockIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) break;
            float dst_val = VKQ[j_VKQ * D_padded + i];
            if (gridDim.y == 1) dst_val /= KQ_rowsum_j;
            dst[j_dst_unrolled * D + i] = dst_val;
        }
        if (gridDim.y == 1 || threadIdx.x != 0) continue;
        float2 dst_meta_val;
        dst_meta_val.x = KQ_max_f[j0 / nwarps];
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[j_dst_unrolled] = dst_meta_val;
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

// ── ROCm-only TBQ4 launcher helper ──
template <int D, int cols_per_block>
static void ggml_cuda_flash_attn_ext_wmma_tbq4_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static_assert(D == 128 || D == 256);

    const ggml_tensor * KQV = dst;

    constexpr int nwarps = 4;
    constexpr int frag_m = cols_per_block == 8 && D % 32 == 0 ? 32 : 16;
    constexpr int VKQ = get_VKQ_stride(D, nwarps, frag_m);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        fattn_kernel = (fattn_kernel_t)flash_attn_ext_wmma_tbq4<D, cols_per_block, nwarps, VKQ, float, false>;
    } else {
        fattn_kernel = (fattn_kernel_t)flash_attn_ext_wmma_tbq4<D, cols_per_block, nwarps, VKQ, float, true>;
    }
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, 0, FATTN_KQ_STRIDE, false, false, false, warp_size);
}

// ── public entry point ──
void ggml_cuda_flash_attn_ext_wmma_tbq4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED_VARS(ctx, dst);
    return;
#else
    const ggml_tensor * Q = dst->src[0];

    tbq4_rotate_input_cuda((float *)Q->data, Q->ne[1], Q->ne[0], ctx.stream());

    if (Q->ne[1] <= 32 || Q->ne[0] == 256) {
        switch (Q->ne[0]) {
            case 128: ggml_cuda_flash_attn_ext_wmma_tbq4_case<128, 16>(ctx, dst); break;
            case 256: ggml_cuda_flash_attn_ext_wmma_tbq4_case<256, 16>(ctx, dst); break;
            default: GGML_ABORT("TBQ4 rocWMMA FA: only D=128,256 supported");
        }
    } else {
        switch (Q->ne[0]) {
            case 128: ggml_cuda_flash_attn_ext_wmma_tbq4_case<128, 32>(ctx, dst); break;
            default: GGML_ABORT("TBQ4 rocWMMA FA: only D=128 supported");
        }
    }

    tbq4_rotate_output_cuda((float *)dst->data, Q->ne[1], Q->ne[0], ctx.stream());
#endif
}
