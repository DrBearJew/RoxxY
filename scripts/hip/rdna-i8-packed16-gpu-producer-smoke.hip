// rdna-i8-packed16-gpu-producer-smoke.hip
// Test-only HIP producer proof: F32 activations -> packed16 I32 payload + F16 K32 scales
// on device, then consume those tensors through the existing ggml RDNA packed16 route.
// This is intentionally not a ggml public op and not wired into model graph dispatch.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

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

static __global__ __launch_bounds__(32, 8) void f32_rows_to_packed16_sidecars_kernel(
        const float * __restrict__ src,      // row-major [rows,K]
        int32_t     * __restrict__ payload,  // row-major [rows,K/4]
        half        * __restrict__ scales,   // row-major [rows,K/32]
        int rows,
        int k_words) {
    const int lane = (int) threadIdx.x & 31;
    const int kb = (int) blockIdx.x;
    const int row = (int) blockIdx.y;
    const int k_blocks = k_words / 8;
    const int k = k_words * 4;
    if (row >= rows || kb >= k_blocks) {
        return;
    }

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
        const float x0 = src[(size_t) row * k + base + 0];
        const float x1 = src[(size_t) row * k + base + 1];
        const float x2 = src[(size_t) row * k + base + 2];
        const float x3 = src[(size_t) row * k + base + 3];
        payload[(size_t) row * k_words + kb * 8 + (lane >> 2)] = pack4_device(
                quant_i8(x0, scale), quant_i8(x1, scale), quant_i8(x2, scale), quant_i8(x3, scale));
    }
}

static void pack_f32_rows_to_packed16_ref(
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
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_LOG", "1", 1);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend\n");
        return 77;
    }

    constexpr int K_WORDS = 16; // 64 values, 2 K32 blocks
    constexpr int K_BLOCKS = K_WORDS / 8;
    constexpr int K = K_WORDS * 4;
    constexpr int M_ACT = 17;
    constexpr int N_OUT = 19;

    const size_t meta = ggml_tensor_overhead() * 10 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 2;
    }

    ggml_tensor * W  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K_WORDS, N_OUT);
    ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K_WORDS, M_ACT);
    ggml_tensor * WS = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K_BLOCKS, N_OUT);
    ggml_tensor * AS = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K_BLOCKS, M_ACT);
    ggml_set_name(W, "packed16_gpu_prod_weight_payload");
    ggml_set_name(A, "packed16_gpu_prod_act_payload");
    ggml_set_name(WS, "packed16_gpu_prod_weight_scales");
    ggml_set_name(AS, "packed16_gpu_prod_act_scales");
    W->src[1] = WS;
    A->src[1] = AS;

    ggml_tensor * C = ggml_mul_mat(ctx, W, A);
    ggml_set_name(C, "packed16_gpu_prod_output");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "alloc failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 3;
    }

    std::vector<int32_t> hW((size_t) N_OUT * K_WORDS);
    std::vector<ggml_fp16_t> hWS((size_t) N_OUT * K_BLOCKS);
    for (int n = 0; n < N_OUT; ++n) {
        for (int w = 0; w < K_WORDS; ++w) {
            hW[(size_t)n*K_WORDS+w] = pack4_host((n+w)%7-3, (2*n+w)%9-4, (n+3*w)%11-5, (3*n+2*w)%13-6);
        }
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            hWS[(size_t)n*K_BLOCKS+kb] = ggml_fp32_to_fp16(0.01f * (1 + n + kb));
        }
    }

    std::vector<float> hA_f32((size_t) M_ACT * K);
    for (int m = 0; m < M_ACT; ++m) {
        for (int k = 0; k < K; ++k) {
            float x = 0.125f * (float) (((m * 37 + k * 13) % 31) - 15);
            if (((m + k) % 23) == 0) x = 0.0f;
            if (((m * 7 + k) % 29) == 0) x = 12.7f;
            if (((m * 11 + k) % 31) == 0) x = -12.8f;
            hA_f32[(size_t)m*K+k] = x;
        }
    }
    std::vector<int32_t> hA_ref;
    std::vector<ggml_fp16_t> hAS_ref;
    pack_f32_rows_to_packed16_ref(hA_f32, M_ACT, K_WORDS, hA_ref, hAS_ref);

    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(int32_t));
    ggml_backend_tensor_set(WS, hWS.data(), 0, hWS.size()*sizeof(ggml_fp16_t));

    float * dA_f32 = nullptr;
    HIP_CHECK(hipMalloc(&dA_f32, hA_f32.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA_f32, hA_f32.data(), hA_f32.size() * sizeof(float), hipMemcpyHostToDevice));
    f32_rows_to_packed16_sidecars_kernel<<<dim3(K_BLOCKS, M_ACT, 1), dim3(32, 1, 1)>>>(
            dA_f32, (int32_t *) A->data, (half *) AS->data, M_ACT, K_WORDS);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int32_t> hA_gpu((size_t) M_ACT * K_WORDS);
    std::vector<ggml_fp16_t> hAS_gpu((size_t) M_ACT * K_BLOCKS);
    ggml_backend_tensor_get(A, hA_gpu.data(), 0, hA_gpu.size()*sizeof(int32_t));
    ggml_backend_tensor_get(AS, hAS_gpu.data(), 0, hAS_gpu.size()*sizeof(ggml_fp16_t));
    for (size_t i = 0; i < hA_ref.size(); ++i) {
        if (hA_ref[i] != hA_gpu[i]) {
            std::fprintf(stderr, "producer payload mismatch i=%zu got=%d ref=%d\n", i, hA_gpu[i], hA_ref[i]);
            return 4;
        }
    }
    for (size_t i = 0; i < hAS_ref.size(); ++i) {
        if (hAS_ref[i] != hAS_gpu[i]) {
            std::fprintf(stderr, "producer scale mismatch i=%zu got=%u ref=%u\n", i, (unsigned) hAS_gpu[i], (unsigned) hAS_ref[i]);
            return 4;
        }
    }

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(st));
        return 5;
    }

    std::vector<float> out((size_t) M_ACT * N_OUT);
    ggml_backend_tensor_get(C, out.data(), 0, out.size()*sizeof(float));

    float max_abs = 0.0f;
    for (int m = 0; m < M_ACT; ++m) {
        for (int n = 0; n < N_OUT; ++n) {
            float ref = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                int dot = 0;
                for (int w = 0; w < 8; ++w) {
                    const int32_t aw = hA_ref[(size_t)m*K_WORDS + kb*8 + w];
                    const int32_t ww = hW[(size_t)n*K_WORDS + kb*8 + w];
                    for (int lane = 0; lane < 4; ++lane) {
                        dot += (int) lane_i8(aw,lane) * (int) lane_i8(ww,lane);
                    }
                }
                ref += (float) dot * ggml_fp16_to_fp32(hAS_ref[(size_t)m*K_BLOCKS+kb]) * ggml_fp16_to_fp32(hWS[(size_t)n*K_BLOCKS+kb]);
            }
            const float got = out[(size_t)m*N_OUT+n];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                std::fprintf(stderr, "mismatch m=%d n=%d got=%g ref=%g diff=%g\n", m, n, got, ref, got-ref);
                return 6;
            }
        }
    }

    std::printf("PASS gpu_f32_producer max_abs=%g payload_words=%zu scale_halfs=%zu\n", max_abs, hA_gpu.size(), hAS_gpu.size());

    HIP_CHECK(hipFree(dA_f32));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
