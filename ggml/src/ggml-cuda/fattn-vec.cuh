#include "common.cuh"
#include "fattn-common.cuh"

#include <mutex>
#include <vector>

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, struct ggml_tensor ** payload, struct ggml_tensor ** scales);
}

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool q8k_dot4_packed16_vec = false> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
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
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int k_tile_rows = nthreads;
    constexpr bool KQ_uses_Q_reg = type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16;
    constexpr int nthreads_KQ = KQ_uses_Q_reg ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ_selected<type_K, D, nthreads_KQ, q8k_dot4_packed16_vec>();
    constexpr bool Q_q8_1 = !KQ_uses_Q_reg;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE
    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    const char * K_aux = sinks;
    if constexpr (q8k_dot4_packed16_vec) {
        if (K_aux != nullptr) {
            constexpr int K_aux_row_bytes = (D / QK8_0) * int(sizeof(half));
            K_aux += (int64_t(sequence) * ne12 + head / gqa_ratio) * ne11 * K_aux_row_bytes;
            K_aux += int64_t(blockIdx.y) * k_tile_rows * K_aux_row_bytes;
        }
    }
    K     += blockIdx.y*k_tile_rows * nb11;
    V     += blockIdx.y*k_tile_rows * nb21;
    maskh += blockIdx.y*k_tile_rows;
    for (int k_VKQ_0 = blockIdx.y*k_tile_rows; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*k_tile_rows,
             // Increment pointers after each loop:
             K += gridDim.y*k_tile_rows*nb11, V += gridDim.y*k_tile_rows*nb21, maskh += gridDim.y*k_tile_rows,
             K_aux += (q8k_dot4_packed16_vec && K_aux != nullptr) ? int64_t(gridDim.y)*k_tile_rows*(D / QK8_0)*int(sizeof(half)) : 0) {

        const int rows_this_tile = k_VKQ_max - k_VKQ_0 < k_tile_rows ? k_VKQ_max - k_VKQ_0 : k_tile_rows;

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;
            const bool active_k = i_KQ < rows_this_tile;
            const char * K_aux_row = nullptr;
            if constexpr (q8k_dot4_packed16_vec) {
                constexpr int K_aux_row_bytes = (D / QK8_0) * int(sizeof(half));
                K_aux_row = (K_aux != nullptr && active_k) ? K_aux + int64_t(i_KQ) * K_aux_row_bytes : nullptr;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (q8k_dot4_packed16_vec) {
                    sum = active_k ? vec_dot_fattn_vec_KQ_q8_0_packed16<D, nthreads_KQ>(
                        K + i_KQ*nb11, K_aux_row, Q_reg[j], Q_i32[j], Q_ds[j]) : 0.0f;
                } else {
                    sum = active_k ? vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]) : 0.0f;
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (!active_k) {
                    sum = -FLT_MAX/2.0f;
                }

                if (use_logit_softcap && active_k) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (active_k && mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*(nb31/(int32_t)sizeof(half)) + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);
            if (k >= rows_this_tile) {
                continue;
            }

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && !q8k_dot4_packed16_vec && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_pack_k_packed16_vec_kernel(
        const char * __restrict__ K,
        char       * __restrict__ K_packed,
        const int64_t nb10,
        const int64_t nb11,
        const int64_t nb12,
        const int64_t nb13,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t packed_ne11,
        const int64_t k_start) {
    static_assert(D == 256, "q8K packed16 VEC shadow is currently D256-only");
    constexpr int nblocks = D / QK8_0;
    constexpr int row_bytes = D + nblocks * int(sizeof(half));
    const int64_t n_k = ne11 - k_start;
    if (n_k <= 0) {
        return;
    }
    const int64_t total = ne13 * ne12 * n_k * nblocks;
    const int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int qblk = int(linear % nblocks);
    const int64_t t = linear / nblocks;
    const int64_t k = k_start + t % n_k;
    const int64_t h_b = t / n_k;
    const int64_t h = h_b % ne12;
    const int64_t b = h_b / ne12;

    const char * src = K + b*nb13 + h*nb12 + k*nb11 + int64_t(qblk)*nb10;
    char * dst_row = K_packed + ((b*ne12 + h)*packed_ne11 + k) * row_bytes;

    half d;
    memcpy(&d, src, sizeof(d));
    ((half *) (dst_row + D))[qblk] = d;

    int * dst_q = (int *) dst_row;
#pragma unroll
    for (int i = 0; i < QK8_0 / int(sizeof(int)); ++i) {
        int v;
        memcpy(&v, src + sizeof(half) + i*int(sizeof(int)), sizeof(v));
        dst_q[qblk * (QK8_0 / int(sizeof(int))) + i] = v;
    }
}

#ifdef GGML_USE_HIP
struct ggml_cuda_q8k_packed16_vec_cache_entry {
    int device = -1;
    const void * owner = nullptr;
    int stream_no = -1;
    const void * k_data = nullptr;
    int64_t nb10 = 0;
    int64_t nb11 = 0;
    int64_t nb12 = 0;
    int64_t nb13 = 0;
    int64_t ne12 = 0;
    int64_t ne13 = 0;
    int64_t packed_ne11 = 0;
    int64_t packed_upto = 0;
    char * data = nullptr;
    size_t bytes = 0;
};

static inline bool ggml_cuda_q8k_dot4_packed16_vec_cache_enabled() {
    const char * env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_CACHE");
    if (!env || atoi(env) == 0) {
        return false;
    }
#ifdef USE_CUDA_GRAPH
    // This unsafe lab cache updates host-side append state between graph launches.
    // Keep it out of CUDA/HIP graph capture unless graphs are explicitly disabled.
    return getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr;
#else
    return true;
#endif
}

static std::mutex & ggml_cuda_q8k_packed16_vec_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::vector<ggml_cuda_q8k_packed16_vec_cache_entry> & ggml_cuda_q8k_packed16_vec_cache_entries() {
    static std::vector<ggml_cuda_q8k_packed16_vec_cache_entry> entries;
    return entries;
}

static inline int64_t ggml_cuda_q8k_packed16_vec_cache_stride_ne11(const ggml_tensor * K) {
    int64_t packed_ne11 = K->ne[1];
    if (K->nb[2] > 0 && K->nb[3] > 0) {
        const int64_t capacity = int64_t(K->nb[3] / K->nb[2]);
        if (capacity >= K->ne[1]) {
            packed_ne11 = capacity;
        }
    }
    return packed_ne11;
}

template <int D>
static const char * ggml_cuda_q8k_packed16_vec_cache_get(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K,
        size_t & nb11_override,
        size_t & nb12_override,
        size_t & nb13_override) {
    static_assert(D == 256, "q8K packed16 VEC cache is currently D256-only");
    constexpr size_t row_bytes = D + (D / QK8_0) * sizeof(half);
    const int64_t packed_ne11 = ggml_cuda_q8k_packed16_vec_cache_stride_ne11(K);
    const void * owner = (const void *) &ctx;
    const size_t bytes = size_t(K->ne[3]) * size_t(K->ne[2]) * size_t(packed_ne11) * row_bytes;
    if (bytes == 0 || K->ne[1] > packed_ne11) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_q8k_packed16_vec_cache_mutex());
    auto & entries = ggml_cuda_q8k_packed16_vec_cache_entries();
    ggml_cuda_q8k_packed16_vec_cache_entry * entry = nullptr;
    for (auto & e : entries) {
        if (e.device == ctx.device && e.owner == owner && e.stream_no == ctx.curr_stream_no && e.k_data == K->data &&
                e.nb10 == int64_t(K->nb[0]) && e.nb11 == int64_t(K->nb[1]) && e.nb12 == int64_t(K->nb[2]) && e.nb13 == int64_t(K->nb[3]) &&
                e.ne12 == K->ne[2] && e.ne13 == K->ne[3] && e.packed_ne11 == packed_ne11) {
            entry = &e;
            break;
        }
    }

    if (!entry) {
        ggml_cuda_set_device(ctx.device);
        char * ptr = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &ptr, bytes));
        entries.push_back({ctx.device, owner, ctx.curr_stream_no, K->data,
            int64_t(K->nb[0]), int64_t(K->nb[1]), int64_t(K->nb[2]), int64_t(K->nb[3]),
            K->ne[2], K->ne[3], packed_ne11, 0, ptr, bytes});
        entry = &entries.back();
    }

    if (K->ne[1] < entry->packed_upto) {
        entry->packed_upto = 0;
    }

    if (K->ne[1] > entry->packed_upto) {
        const int64_t k_start = entry->packed_upto;
        const int64_t n_k = K->ne[1] - k_start;
        const dim3 pack_grid((size_t(n_k) * size_t(K->ne[2]) * size_t(K->ne[3]) * (D / QK8_0) + 255) / 256);
        ggml_cuda_q8k_pack_k_packed16_vec_kernel<D><<<pack_grid, 256, 0, ctx.stream()>>>(
            (const char *) K->data, entry->data,
            K->nb[0], K->nb[1], K->nb[2], K->nb[3],
            K->ne[1], K->ne[2], K->ne[3], packed_ne11, k_start);
        CUDA_CHECK(cudaGetLastError());
        entry->packed_upto = K->ne[1];
    }

    nb11_override = row_bytes;
    nb12_override = size_t(packed_ne11) * row_bytes;
    nb13_override = size_t(K->ne[2]) * nb12_override;
    return entry->data;
}
#endif // GGML_USE_HIP

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    const bool q8k_dot4_packed16_vec =
        (type_K == GGML_TYPE_Q8_0 && D == 256 && ggml_cuda_q8k_dot4_packed16_vec_enabled()) ||
        (type_K == GGML_TYPE_I32  && D == 256 && ggml_cuda_packed16_fa2_vec_enabled());
    fattn_kernel_t fattn_kernel = nullptr;
    if constexpr (type_K == GGML_TYPE_I32 && D == 256) {
        if (!q8k_dot4_packed16_vec) {
            GGML_ABORT("packed16 I32 VEC route requires GGML_CUDA_ROCM_PACKED16_FA2_VEC=1 or GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_fa2_vec");
        }
        fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, true>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0 && D == 256) {
        fattn_kernel = q8k_dot4_packed16_vec ?
            flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, true> :
            flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, false>;
    } else {
        fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, false>;
    }
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    constexpr int nbatch_fa = D;

    ggml_cuda_pool_alloc<char> K_packed(ctx.pool());
    const char * K_data_override = nullptr;
    const char * K_sinks_override = nullptr;
    size_t nb11_override = 0;
    size_t nb12_override = 0;
    size_t nb13_override = 0;
    if constexpr (type_K == GGML_TYPE_I32 && D == 256) {
        if (q8k_dot4_packed16_vec) {
#ifdef GGML_USE_HIP
            ggml_tensor * payload_tensor = nullptr;
            ggml_tensor * scales_tensor = nullptr;
            const void * lookup = (K->view_src != nullptr) ? K->view_src->data : K->data;
            llama_kv_cache_get_packed16_tensors(lookup, &payload_tensor, &scales_tensor);
            if (!payload_tensor || !scales_tensor) {
                GGML_ABORT("packed16 FA2/VEC sidecar route requires registered K payload/scales tensors");
            }
            const ptrdiff_t payload_offset = (const char *) K->data - (const char *) payload_tensor->data;
            const ptrdiff_t payload_row_bytes = (ptrdiff_t) payload_tensor->nb[1];
            const ptrdiff_t scales_row_bytes  = (ptrdiff_t) scales_tensor->nb[1];
            ptrdiff_t row_offset = 0;
            if (payload_offset >= 0 && payload_offset % payload_row_bytes == 0) {
                row_offset = payload_offset / payload_row_bytes;
            }
            // Always feed the VEC kernel from the registered packed16 payload. In
            // live KV this is usually identical to K->data (a payload view), but
            // tests and some graph views register a separate I32 tensor that only
            // describes the packed16 K shape. The sidecar registry is authoritative.
            K_data_override = (const char *) payload_tensor->data + row_offset * payload_row_bytes;
            nb11_override = (size_t) payload_row_bytes;
            nb12_override = (size_t) K->ne[1] * nb11_override;
            nb13_override = (size_t) K->ne[2] * nb12_override;
            K_sinks_override = (const char *) scales_tensor->data + row_offset * scales_row_bytes;
#else
            GGML_ABORT("packed16 FA2/VEC sidecar route is HIP-only");
#endif
        }
    }
    if constexpr (type_K == GGML_TYPE_Q8_0 && D == 256) {
        if (q8k_dot4_packed16_vec) {
#ifdef GGML_USE_HIP
            if (ggml_cuda_q8k_dot4_packed16_vec_cache_enabled()) {
                K_data_override = ggml_cuda_q8k_packed16_vec_cache_get<D>(ctx, K, nb11_override, nb12_override, nb13_override);
            }
#endif // GGML_USE_HIP
            if (!K_data_override) {
                constexpr size_t row_bytes = D + (D / QK8_0) * sizeof(half);
                const size_t nrows = size_t(K->ne[1]) * size_t(K->ne[2]) * size_t(K->ne[3]);
                K_packed.alloc(nrows * row_bytes);
                const dim3 pack_grid((nrows * (D / QK8_0) + 255) / 256);
                ggml_cuda_q8k_pack_k_packed16_vec_kernel<D><<<pack_grid, 256, 0, ctx.stream()>>>(
                    (const char *) K->data, K_packed.ptr,
                    K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                    K->ne[1], K->ne[2], K->ne[3], K->ne[1], 0);
                CUDA_CHECK(cudaGetLastError());
                K_data_override = K_packed.ptr;
                nb11_override = row_bytes;
                nb12_override = size_t(K->ne[1]) * row_bytes;
                nb13_override = size_t(K->ne[2]) * nb12_override;
            }
        }
    }

    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, nbatch_fa, need_f16_K, need_f16_V, false,
        WARP_SIZE, K_data_override, nb11_override, nb12_override, nb13_override, K_sinks_override);
}

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

#ifdef GGML_USE_HIP
    // Unsafe A/B probe for D256 q8_0/packed16-K + q4_0-V VEC that keeps the
    // stable launch_fattn/parallel-block/combine path but groups more Q columns
    // per block. This is intentionally opt-in so default quantized-KV behavior
    // and long-context serving stay on the validated cols=2 VEC route.
    if constexpr (D == 256 && (type_K == GGML_TYPE_Q8_0 || type_K == GGML_TYPE_I32) && type_V == GGML_TYPE_Q4_0) {
        const char * unsafe = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
        if (!unsafe) {
            unsafe = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
        }
        const char * cols_env = nullptr;
        if constexpr (type_K == GGML_TYPE_I32) {
            cols_env = getenv("GGML_CUDA_ROCM_PACKED16_FA2_VEC_COLS");
        } else {
            cols_env = getenv("GGML_CUDA_ROCM_Q8K_Q4V_VEC_COLS");
        }
        const int cols_override = (unsafe && atoi(unsafe) != 0 && cols_env) ? atoi(cols_env) : 0;
        if (cols_override == 16) {
            constexpr int cols_per_block = 16;
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
            return;
        }
        if (cols_override == 8) {
            constexpr int cols_per_block = 8;
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
            return;
        }
        if (cols_override == 4) {
            constexpr int cols_per_block = 4;
            if (logit_softcap == 0.0f) {
                constexpr bool use_logit_softcap = false;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            } else {
                constexpr bool use_logit_softcap = true;
                ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
            return;
        }
    }
#endif // GGML_USE_HIP

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_dispatch<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)
