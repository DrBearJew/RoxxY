// rdna-i8-packed16-iq4xs-producer-smoke.hip
// Test-only IQ4_XS producer proof: GPU dequant/pack synthetic block_iq4_xs expert weights
// into packed16 I32 payload + F16 K32 scales, then consume with existing ggml
// MUL_MAT_ID bounds WMMA route. No public ggml op/API/model graph wiring.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

struct iq4_xs_raw {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
static_assert(sizeof(iq4_xs_raw) == 136, "iq4_xs_raw must match block_iq4_xs byte layout for QK_K=256");

static constexpr int IQ4_K = 256;
static constexpr int IQ4_K32_BLOCKS = 8;
static constexpr int IQ4_QS_BYTES_PER_K32 = 16;
static const int8_t k_iq4nl_host[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static ggml_fp16_t half_bits_to_ggml_fp16(half h) {
    ggml_fp16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

static int8_t lane_i8(int32_t w, int lane) {
    return (int8_t) (((uint32_t) w >> (8 * lane)) & 0xffu);
}

static int32_t pack4_host(int a, int b, int c, int d) {
    uint32_t w = 0;
    const int vals[4] = {a, b, c, d};
    for (int i = 0; i < 4; ++i) {
        w |= ((uint32_t) (uint8_t) (int8_t) vals[i]) << (8 * i);
    }
    return (int32_t) w;
}

static __device__ __forceinline__ int iq4nl_device(int q) {
    constexpr int8_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return (int) table[q & 15];
}

static __device__ __forceinline__ int32_t pack4_device(int a, int b, int c, int d) {
    uint32_t w = 0;
    w |= ((uint32_t) (uint8_t) (int8_t) a) << 0;
    w |= ((uint32_t) (uint8_t) (int8_t) b) << 8;
    w |= ((uint32_t) (uint8_t) (int8_t) c) << 16;
    w |= ((uint32_t) (uint8_t) (int8_t) d) << 24;
    return (int32_t) w;
}

static __device__ __forceinline__ int quant_i8(float x, float scale) {
    int qi = (int) nearbyintf(x / scale);
    qi = max(-127, min(127, qi));
    return qi;
}

static __device__ __forceinline__ int iq4xs_ls(const iq4_xs_raw & block, int ib) {
    return ((block.scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((block.scales_h >> (2*ib)) & 0x03) << 4);
}

static __device__ __forceinline__ float iq4xs_dequant_lane(const iq4_xs_raw & block, int ib, int lane) {
    const uint8_t qbyte = block.qs[ib * IQ4_QS_BYTES_PER_K32 + (lane & 15)];
    const int q = lane < 16 ? (qbyte & 0x0f) : (qbyte >> 4);
    const float dl = __half2float(block.d) * (float) (iq4xs_ls(block, ib) - 32);
    return dl * (float) iq4nl_device(q);
}

static __global__ __launch_bounds__(32, 8) void iq4xs_experts_to_packed16_sidecars_kernel(
        const iq4_xs_raw * __restrict__ src,      // [expert,out]
        int32_t          * __restrict__ payload,  // ggml [K/4,out,expert], K=256 => K/4=64
        half             * __restrict__ scales,   // ggml [K/32,out,expert], K=256 => 8
        int n_out,
        int n_expert) {
    const int lane = (int) threadIdx.x & 31;
    const int ib = (int) blockIdx.x;
    const int out = (int) blockIdx.y;
    const int expert = (int) blockIdx.z;
    if (ib >= IQ4_K32_BLOCKS || out >= n_out || expert >= n_expert) {
        return;
    }

    const iq4_xs_raw & block = src[(size_t) expert * n_out + out];
    const float x = iq4xs_dequant_lane(block, ib, lane);
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;

    if (lane == 0) {
        scales[((size_t) expert * n_out + out) * IQ4_K32_BLOCKS + ib] = __float2half(scale);
    }
    if ((lane & 3) == 0) {
        payload[((size_t) expert * n_out + out) * (IQ4_K / 4) + ib * 8 + (lane >> 2)] = pack4_device(
                quant_i8(iq4xs_dequant_lane(block, ib, lane + 0), scale),
                quant_i8(iq4xs_dequant_lane(block, ib, lane + 1), scale),
                quant_i8(iq4xs_dequant_lane(block, ib, lane + 2), scale),
                quant_i8(iq4xs_dequant_lane(block, ib, lane + 3), scale));
    }
}

static __global__ __launch_bounds__(32, 8) void f32_rows_to_packed16_sidecars_kernel(
        const float * __restrict__ src,
        int32_t     * __restrict__ payload,
        half        * __restrict__ scales,
        int rows,
        int k_words) {
    const int lane = (int) threadIdx.x & 31;
    const int kb = (int) blockIdx.x;
    const int row = (int) blockIdx.y;
    const int k_blocks = k_words / 8;
    const int k = k_words * 4;
    if (row >= rows || kb >= k_blocks) return;

    const float x = src[(size_t) row * k + kb * 32 + lane];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    if (lane == 0) {
        scales[(size_t) row * k_blocks + kb] = __float2half(scale);
    }
    if ((lane & 3) == 0) {
        const int base = kb * 32 + lane;
        payload[(size_t) row * k_words + kb * 8 + (lane >> 2)] = pack4_device(
                quant_i8(src[(size_t) row * k + base + 0], scale),
                quant_i8(src[(size_t) row * k + base + 1], scale),
                quant_i8(src[(size_t) row * k + base + 2], scale),
                quant_i8(src[(size_t) row * k + base + 3], scale));
    }
}

static int iq4xs_ls_host(const iq4_xs_raw & block, int ib) {
    return ((block.scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((block.scales_h >> (2*ib)) & 0x03) << 4);
}

static float iq4xs_dequant_lane_host(const iq4_xs_raw & block, int ib, int lane) {
    const uint8_t qbyte = block.qs[ib * IQ4_QS_BYTES_PER_K32 + (lane & 15)];
    const int q = lane < 16 ? (qbyte & 0x0f) : (qbyte >> 4);
    const float dl = __half2float(block.d) * (float) (iq4xs_ls_host(block, ib) - 32);
    return dl * (float) k_iq4nl_host[q];
}

static void pack_iq4xs_weights_ref(
        const std::vector<iq4_xs_raw> & src,
        int n_out,
        int n_expert,
        std::vector<int32_t> & payload,
        std::vector<ggml_fp16_t> & scales) {
    payload.assign((size_t) n_expert * n_out * (IQ4_K / 4), 0);
    scales.assign((size_t) n_expert * n_out * IQ4_K32_BLOCKS, ggml_fp32_to_fp16(1.0f));
    for (int e = 0; e < n_expert; ++e) {
        for (int out = 0; out < n_out; ++out) {
            const iq4_xs_raw & block = src[(size_t) e * n_out + out];
            for (int ib = 0; ib < IQ4_K32_BLOCKS; ++ib) {
                float amax = 0.0f;
                for (int lane = 0; lane < 32; ++lane) {
                    amax = fmaxf(amax, fabsf(iq4xs_dequant_lane_host(block, ib, lane)));
                }
                const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
                scales[((size_t) e * n_out + out) * IQ4_K32_BLOCKS + ib] = ggml_fp32_to_fp16(scale);
                for (int w = 0; w < 8; ++w) {
                    int q[4];
                    for (int lane = 0; lane < 4; ++lane) {
                        int qi = (int) nearbyintf(iq4xs_dequant_lane_host(block, ib, w * 4 + lane) / scale);
                        if (qi < -127) qi = -127;
                        if (qi >  127) qi =  127;
                        q[lane] = qi;
                    }
                    payload[((size_t) e * n_out + out) * (IQ4_K / 4) + ib * 8 + w] = pack4_host(q[0], q[1], q[2], q[3]);
                }
            }
        }
    }
}

static void pack_f32_rows_ref(
        const std::vector<float> & src,
        int rows,
        int k_words,
        std::vector<int32_t> & payload,
        std::vector<ggml_fp16_t> & scales) {
    const int k = k_words * 4;
    const int k_blocks = k_words / 8;
    payload.assign((size_t) rows * k_words, 0);
    scales.assign((size_t) rows * k_blocks, ggml_fp32_to_fp16(1.0f));
    for (int row = 0; row < rows; ++row) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(src[(size_t) row * k + kb * 32 + i]));
            }
            const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            scales[(size_t) row * k_blocks + kb] = ggml_fp32_to_fp16(scale);
            for (int w = 0; w < 8; ++w) {
                int q[4];
                for (int lane = 0; lane < 4; ++lane) {
                    int qi = (int) nearbyintf(src[(size_t) row * k + kb * 32 + w * 4 + lane] / scale);
                    if (qi < -127) qi = -127;
                    if (qi >  127) qi =  127;
                    q[lane] = qi;
                }
                payload[(size_t) row * k_words + kb * 8 + w] = pack4_host(q[0], q[1], q[2], q[3]);
            }
        }
    }
}

int main() {
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_BOUNDS", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG", "1", 1);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend\n");
        return 77;
    }

    constexpr int K_WORDS = IQ4_K / 4;
    constexpr int K_BLOCKS = IQ4_K32_BLOCKS;
    constexpr int N_OUT = 16;
    constexpr int N_EXPERT = 3;
    constexpr int N_LANES = 7;

    const size_t meta = ggml_tensor_overhead() * 20 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 2;
    }

    ggml_tensor * W  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, N_OUT, N_EXPERT);
    ggml_tensor * A  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, 1, N_LANES);
    ggml_tensor * WS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, N_OUT, N_EXPERT);
    ggml_tensor * AS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, 1, N_LANES);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, N_LANES);
    ggml_tensor * bounds = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, N_EXPERT);
    ggml_set_name(W, "packed16_iq4xs_prod_weight_payload");
    ggml_set_name(A, "packed16_iq4xs_prod_act_payload");
    ggml_set_name(WS, "packed16_iq4xs_prod_weight_scales");
    ggml_set_name(AS, "packed16_iq4xs_prod_act_scales");
    ggml_set_name(ids, "packed16_iq4xs_prod_ids");
    ggml_set_name(bounds, "packed16_iq4xs_prod_bounds");

    ggml_tensor * C = ggml_mul_mat_id(ctx, W, A, ids);
    ggml_set_name(C, "packed16_iq4xs_prod_output");
    C->src[3] = WS;
    C->src[4] = AS;
    C->src[5] = bounds;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "alloc failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 3;
    }

    std::vector<iq4_xs_raw> hIQ4((size_t) N_EXPERT * N_OUT);
    for (int e = 0; e < N_EXPERT; ++e) {
        for (int out = 0; out < N_OUT; ++out) {
            iq4_xs_raw & block = hIQ4[(size_t) e * N_OUT + out];
            block.d = __float2half(0.00025f * (1 + e + out));
            block.scales_h = 0;
            for (int i = 0; i < 4; ++i) block.scales_l[i] = 0;
            for (int ib = 0; ib < IQ4_K32_BLOCKS; ++ib) {
                const int ls = 33 + ((e * 7 + out * 3 + ib * 5) % 24); // 33..56, non-zero dl
                block.scales_l[ib/2] |= (uint8_t) ((ls & 0x0f) << (4*(ib%2)));
                block.scales_h |= (uint16_t) (((ls >> 4) & 0x03) << (2*ib));
            }
            for (int i = 0; i < 128; ++i) {
                const uint8_t lo = (uint8_t) ((e * 5 + out * 7 + i * 3) & 15);
                const uint8_t hi = (uint8_t) ((e * 11 + out * 13 + i * 5 + 1) & 15);
                block.qs[i] = (uint8_t) (lo | (hi << 4));
            }
        }
    }

    std::vector<float> hA_f32((size_t) N_LANES * IQ4_K);
    for (int lane = 0; lane < N_LANES; ++lane) {
        for (int k = 0; k < IQ4_K; ++k) {
            float x = 0.03125f * (float) (((lane * 37 + k * 13) % 63) - 31);
            if (((lane + k) % 41) == 0) x = 0.0f;
            if (((lane * 7 + k) % 53) == 0) x = 4.0f;
            if (((lane * 11 + k) % 59) == 0) x = -4.0f;
            hA_f32[(size_t)lane*IQ4_K+k] = x;
        }
    }

    std::vector<int32_t> hW_ref;
    std::vector<ggml_fp16_t> hWS_ref;
    std::vector<int32_t> hA_ref;
    std::vector<ggml_fp16_t> hAS_ref;
    pack_iq4xs_weights_ref(hIQ4, N_OUT, N_EXPERT, hW_ref, hWS_ref);
    pack_f32_rows_ref(hA_f32, N_LANES, K_WORDS, hA_ref, hAS_ref);

    const int starts[N_EXPERT] = {0, 2, 5};
    const int counts[N_EXPERT] = {2, 3, 2};
    std::vector<int32_t> hIds((size_t) N_LANES);
    std::vector<int32_t> hBounds((size_t) N_EXPERT * 2);
    auto bounds_index = [](int row, int expert) { return (size_t) expert*2 + row; };
    for (int e = 0; e < N_EXPERT; ++e) {
        hBounds[bounds_index(0,e)] = starts[e];
        hBounds[bounds_index(1,e)] = counts[e];
        for (int lane = starts[e]; lane < starts[e] + counts[e]; ++lane) hIds[lane] = e;
    }
    ggml_backend_tensor_set(ids, hIds.data(), 0, hIds.size()*sizeof(int32_t));
    ggml_backend_tensor_set(bounds, hBounds.data(), 0, hBounds.size()*sizeof(int32_t));

    iq4_xs_raw * dIQ4 = nullptr;
    float * dA_f32 = nullptr;
    HIP_CHECK(hipMalloc(&dIQ4, hIQ4.size() * sizeof(iq4_xs_raw)));
    HIP_CHECK(hipMalloc(&dA_f32, hA_f32.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dIQ4, hIQ4.data(), hIQ4.size() * sizeof(iq4_xs_raw), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA_f32, hA_f32.data(), hA_f32.size() * sizeof(float), hipMemcpyHostToDevice));

    iq4xs_experts_to_packed16_sidecars_kernel<<<dim3(K_BLOCKS, N_OUT, N_EXPERT), dim3(32, 1, 1)>>>(
            dIQ4, (int32_t *) W->data, (half *) WS->data, N_OUT, N_EXPERT);
    HIP_CHECK(hipGetLastError());
    f32_rows_to_packed16_sidecars_kernel<<<dim3(K_BLOCKS, N_LANES, 1), dim3(32, 1, 1)>>>(
            dA_f32, (int32_t *) A->data, (half *) AS->data, N_LANES, K_WORDS);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int32_t> hW_gpu(hW_ref.size());
    std::vector<ggml_fp16_t> hWS_gpu(hWS_ref.size());
    std::vector<int32_t> hA_gpu(hA_ref.size());
    std::vector<ggml_fp16_t> hAS_gpu(hAS_ref.size());
    ggml_backend_tensor_get(W, hW_gpu.data(), 0, hW_gpu.size()*sizeof(int32_t));
    ggml_backend_tensor_get(WS, hWS_gpu.data(), 0, hWS_gpu.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(A, hA_gpu.data(), 0, hA_gpu.size()*sizeof(int32_t));
    ggml_backend_tensor_get(AS, hAS_gpu.data(), 0, hAS_gpu.size()*sizeof(ggml_fp16_t));
    for (size_t i = 0; i < hW_ref.size(); ++i) {
        if (hW_ref[i] != hW_gpu[i]) {
            std::fprintf(stderr, "iq4 weight payload mismatch i=%zu got=%d ref=%d\n", i, hW_gpu[i], hW_ref[i]);
            return 4;
        }
    }
    for (size_t i = 0; i < hWS_ref.size(); ++i) {
        if (hWS_ref[i] != hWS_gpu[i]) {
            std::fprintf(stderr, "iq4 weight scale mismatch i=%zu got=%u ref=%u\n", i, (unsigned) hWS_gpu[i], (unsigned) hWS_ref[i]);
            return 4;
        }
    }
    for (size_t i = 0; i < hA_ref.size(); ++i) {
        if (hA_ref[i] != hA_gpu[i]) {
            std::fprintf(stderr, "act payload mismatch i=%zu got=%d ref=%d\n", i, hA_gpu[i], hA_ref[i]);
            return 4;
        }
    }
    for (size_t i = 0; i < hAS_ref.size(); ++i) {
        if (hAS_ref[i] != hAS_gpu[i]) {
            std::fprintf(stderr, "act scale mismatch i=%zu got=%u ref=%u\n", i, (unsigned) hAS_gpu[i], (unsigned) hAS_ref[i]);
            return 4;
        }
    }

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(st));
        return 5;
    }

    std::vector<float> out((size_t) N_LANES * N_OUT);
    ggml_backend_tensor_get(C, out.data(), 0, out.size()*sizeof(float));

    float max_abs = 0.0f;
    for (int lane = 0; lane < N_LANES; ++lane) {
        const int e = hIds[lane];
        for (int row = 0; row < N_OUT; ++row) {
            float ref = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                int dot = 0;
                for (int w = 0; w < 8; ++w) {
                    const int32_t aw = hA_ref[(size_t)lane*K_WORDS + kb*8 + w];
                    const int32_t ww = hW_ref[((size_t)e*N_OUT + row)*K_WORDS + kb*8 + w];
                    for (int l = 0; l < 4; ++l) dot += (int) lane_i8(aw,l) * (int) lane_i8(ww,l);
                }
                ref += (float) dot * ggml_fp16_to_fp32(hAS_ref[(size_t)lane*K_BLOCKS+kb]) *
                                  ggml_fp16_to_fp32(hWS_ref[((size_t)e*N_OUT + row)*K_BLOCKS+kb]);
            }
            const float got = out[(size_t)lane*N_OUT + row];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                std::fprintf(stderr, "mismatch lane=%d row=%d e=%d got=%g ref=%g diff=%g\n", lane, row, e, got, ref, got-ref);
                return 6;
            }
        }
    }

    std::printf("PASS iq4xs_gpu_producer max_abs=%g weight_words=%zu act_words=%zu\n", max_abs, hW_gpu.size(), hA_gpu.size());

    HIP_CHECK(hipFree(dA_f32));
    HIP_CHECK(hipFree(dIQ4));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
