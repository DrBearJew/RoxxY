// rdna-i8-packed16-blockscaled-proof.hip
// Standalone RDNA3/RDNA3.5 W8A8 packed16/I32 + F16 sidecar block-scaled GEMM proof.
// This is intentionally not wired into llama.cpp dispatch or hipBLASLt.
// It proves the target custom TensileLite math contract:
//   per-K32 I32 dot -> F16 sidecar scale -> F32 accumulate -> one F32 write.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

using rdna_i8_p16_v4i32 = int __attribute__((ext_vector_type(4)));
using rdna_i8_p16_v8i32 = int __attribute__((ext_vector_type(8)));

static constexpr int K32_WORDS = 8; // 32 int8 values / 4 lanes per int32 word.

static int32_t pack4_host(int a, int b, int c, int d) {
    const int vals[4] = {a, b, c, d};
    uint32_t w = 0;
    for (int i = 0; i < 4; ++i) {
        w |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(vals[i]))) << (8 * i);
    }
    return static_cast<int32_t>(w);
}

static int8_t lane_i8_host(int32_t w, int lane) {
    return static_cast<int8_t>((static_cast<uint32_t>(w) >> (8 * lane)) & 0xffu);
}

static __device__ __forceinline__ rdna_i8_p16_v8i32 rdna_i8_wmma_i32_16x16x16(
        rdna_i8_p16_v4i32 a_frag,
        rdna_i8_p16_v4i32 b_frag,
        rdna_i8_p16_v8i32 acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, b_frag, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
# error "rdna-i8-packed16-blockscaled-proof keeps gfx12 as future work; implement v2i32 WMMA fragments before enabling"
#elif defined(__HIP_DEVICE_COMPILE__)
# error "rdna-i8-packed16-blockscaled-proof requires gfx11/RDNA3/RDNA3.5 today, gfx12 is future TODO"
#else
    return acc;
#endif
}

static __global__ __launch_bounds__(32, 8) void packed16_blockscaled_kernel(
        const int32_t * __restrict__ A_words,   // row-major [M,K/4]
        const half    * __restrict__ A_scales,  // row-major [M,K/32], nullable if use_a_scale == 0
        const int32_t * __restrict__ B_words,   // row-major [N,K/4]
        const half    * __restrict__ B_scales,  // row-major [N,K/32]
        float         * __restrict__ C,         // row-major [M,N]
        int M,
        int N,
        int K_words,
        int use_a_scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int row_a  = tile_m + lane_lo;
    const int col_b  = tile_n + lane_lo;
    const int K_blocks = K_words / K32_WORDS;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < K_blocks; ++kb) {
        const int32_t * a_row = row_a < M ? A_words + static_cast<size_t>(row_a) * K_words + static_cast<size_t>(kb) * K32_WORDS : nullptr;
        const int32_t * b_row = col_b < N ? B_words + static_cast<size_t>(col_b) * K_words + static_cast<size_t>(kb) * K32_WORDS : nullptr;
        rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            rdna_i8_p16_v4i32 a_frag;
            rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int word = half_k32 * 4 + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                b_frag[g] = b_row != nullptr ? b_row[word] : 0;
            }
            acc_i = rdna_i8_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int out_m = tile_m + 2 * r + lane_hi;
            const int out_n = tile_n + lane_lo;
            if (out_m < M && out_n < N) {
                const float as = use_a_scale ? __half2float(A_scales[static_cast<size_t>(out_m) * K_blocks + kb]) : 1.0f;
                const float bs = __half2float(B_scales[static_cast<size_t>(out_n) * K_blocks + kb]);
                acc_f[r] += static_cast<float>(acc_i[r]) * as * bs;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        if (out_m < M && out_n < N) {
            C[static_cast<size_t>(out_m) * N + out_n] = acc_f[r];
        }
    }
}

static void cpu_ref(
        const std::vector<int32_t> & A,
        const std::vector<half> & As,
        const std::vector<int32_t> & B,
        const std::vector<half> & Bs,
        std::vector<float> & C,
        int M,
        int N,
        int K_words,
        bool use_a_scale) {
    const int K_blocks = K_words / K32_WORDS;
    std::fill(C.begin(), C.end(), 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int kb = 0; kb < K_blocks; ++kb) {
                int dot = 0;
                for (int w = 0; w < K32_WORDS; ++w) {
                    const int32_t aw = A[static_cast<size_t>(m) * K_words + kb * K32_WORDS + w];
                    const int32_t bw = B[static_cast<size_t>(n) * K_words + kb * K32_WORDS + w];
                    for (int lane = 0; lane < 4; ++lane) {
                        dot += static_cast<int>(lane_i8_host(aw, lane)) * static_cast<int>(lane_i8_host(bw, lane));
                    }
                }
                const float as = use_a_scale ? __half2float(As[static_cast<size_t>(m) * K_blocks + kb]) : 1.0f;
                const float bs = __half2float(Bs[static_cast<size_t>(n) * K_blocks + kb]);
                acc += static_cast<float>(dot) * as * bs;
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

int main(int argc, char ** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 64;
    const int N = argc > 2 ? std::atoi(argv[2]) : 128;
    const int K = argc > 3 ? std::atoi(argv[3]) : 512;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 100;
    const bool use_a_scale = argc > 5 ? std::atoi(argv[5]) != 0 : true;

    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s [M>0] [N>0] [K multiple of 32] [iters>0] [use_a_scale=0|1]\n", argv[0]);
        return 2;
    }

    const int K_words = K / 4;
    const int K_blocks = K / 32;
    std::printf("shape M=%d N=%d K=%d K_words=%d K32=%d use_a_scale=%d a_scale_ptr=%s\n", M, N, K, K_words, K_blocks, use_a_scale ? 1 : 0, use_a_scale ? "device" : "null");

    std::vector<int32_t> hA(static_cast<size_t>(M) * K_words);
    std::vector<int32_t> hB(static_cast<size_t>(N) * K_words);
    std::vector<half> hAs(static_cast<size_t>(M) * K_blocks);
    std::vector<half> hBs(static_cast<size_t>(N) * K_blocks);
    std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> hRef(static_cast<size_t>(M) * N, 0.0f);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> qdist(-7, 7);
    for (int m = 0; m < M; ++m) {
        for (int w = 0; w < K_words; ++w) {
            hA[static_cast<size_t>(m) * K_words + w] = pack4_host(qdist(rng), qdist(rng), qdist(rng), qdist(rng));
        }
        for (int kb = 0; kb < K_blocks; ++kb) {
            const float s = 0.003f * static_cast<float>(1 + ((m + 3 * kb) % 17));
            hAs[static_cast<size_t>(m) * K_blocks + kb] = __float2half(s);
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int w = 0; w < K_words; ++w) {
            hB[static_cast<size_t>(n) * K_words + w] = pack4_host(qdist(rng), qdist(rng), qdist(rng), qdist(rng));
        }
        for (int kb = 0; kb < K_blocks; ++kb) {
            const float s = 0.004f * static_cast<float>(1 + ((n + 5 * kb) % 19));
            hBs[static_cast<size_t>(n) * K_blocks + kb] = __float2half(s);
        }
    }

    int32_t * dA = nullptr;
    int32_t * dB = nullptr;
    half * dAs = nullptr;
    half * dBs = nullptr;
    float * dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&dAs, hAs.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&dBs, hBs.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&dC, hC.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAs, hAs.data(), hAs.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBs, hBs.data(), hBs.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, hC.size() * sizeof(float)));

    const dim3 block(32, 1, 1);
    const dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    half * dAs_arg = use_a_scale ? dAs : nullptr;

    packed16_blockscaled_kernel<<<grid, block>>>(dA, dAs_arg, dB, dBs, dC, M, N, K_words, use_a_scale ? 1 : 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        packed16_blockscaled_kernel<<<grid, block>>>(dA, dAs_arg, dB, dBs, dC, M, N, K_words, use_a_scale ? 1 : 0);
    }
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms_total = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_total, ev0, ev1));

    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    cpu_ref(hA, hAs, hB, hBs, hRef, M, N, K_words, use_a_scale);

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        const float diff = std::fabs(hC[i] - hRef[i]);
        const float rel = diff / std::max(1.0e-7f, std::fabs(hRef[i]));
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        if (diff > 1.0e-4f && rel > 1.0e-4f) {
            if (bad < 8) {
                std::fprintf(stderr, "mismatch i=%zu got=%g ref=%g diff=%g rel=%g\n", i, hC[i], hRef[i], diff, rel);
            }
            ++bad;
        }
    }

    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double avg_ms = ms_total / static_cast<double>(iters);
    const double tops = ops / (avg_ms * 1.0e-3) / 1.0e12;
    std::printf("%s max_abs=%g max_rel=%g avg_ms=%g tops=%g grid=(%u,%u,%u) iters=%d\n",
            bad == 0 ? "PASS" : "FAIL", max_abs, max_rel, avg_ms, tops, grid.x, grid.y, grid.z, iters);

    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(dBs));
    HIP_CHECK(hipFree(dAs));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dA));
    return bad == 0 ? 0 : 1;
}
