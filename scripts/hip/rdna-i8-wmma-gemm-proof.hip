// rdna-i8-wmma-gemm-proof.hip
// Standalone RDNA3 W8A8 -> I32 GEMM proof using v_wmma_i32_16x16x16_iu8_w32.
// This is intentionally not wired into llama.cpp dispatch. It proves the native
// instruction, lane layout, and exact I32 accumulation contract for M/N/K multiples of 16.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

using rdna_i8_v4i32 = int __attribute__((ext_vector_type(4)));
using rdna_i8_v8i32 = int __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ uint32_t pack_i8x4(const int8_t * base, int stride, int k0) {
    uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t b = static_cast<uint8_t>(base[(k0 + i) * stride]);
        w |= static_cast<uint32_t>(b) << (8 * i);
    }
    return w;
}

static __device__ __forceinline__ rdna_i8_v8i32 rdna_i8_wmma(
        rdna_i8_v4i32 a_frag, rdna_i8_v4i32 b_frag, rdna_i8_v8i32 acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
    // Integer NEG bits select signedness on RDNA WMMA: true,true means signed i8 A/B.
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, b_frag, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_frag, true, b_frag, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__)
# error "rdna-i8-wmma-gemm-proof requires RDNA3/RDNA3.5/RDNA4 WMMA target"
#else
    return acc;
#endif
}

static __global__ __launch_bounds__(32, 8) void rdna_i8_wmma_gemm_kernel(
        const int8_t * __restrict__ A,  // row-major [M,K]
        const int8_t * __restrict__ B,  // row-major [K,N]
        int32_t      * __restrict__ C,  // row-major [M,N]
        int M, int N, int K) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int row_a  = tile_m + lane_lo;  // duplicated by lanes row and row+16.
    const int col_b  = tile_n + lane_lo;  // duplicated by lanes col and col+16.

    rdna_i8_v8i32 acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k_base = 0; k_base < K; k_base += 16) {
        rdna_i8_v4i32 a_frag;
        rdna_i8_v4i32 b_frag;

#pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint32_t aw = 0;
            uint32_t bw = 0;
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int k = k_base + 4 * g + byte;
                const int8_t av = A[row_a * K + k];
                const int8_t bv = B[k * N + col_b];
                aw |= static_cast<uint32_t>(static_cast<uint8_t>(av)) << (8 * byte);
                bw |= static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * byte);
            }
            a_frag[g] = static_cast<int>(aw);
            b_frag[g] = static_cast<int>(bw);
        }

        acc = rdna_i8_wmma(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        C[out_m * N + out_n] = acc[r];
    }
}

static __global__ __launch_bounds__(32, 8) void rdna_i8_wmma_q8scale_gemm_kernel(
        const int8_t * __restrict__ A,        // row-major int8 activation [M,K]
        const float  * __restrict__ A_scales, // [M,K/32]
        const int8_t * __restrict__ B,        // row-major int8 weight [K,N]
        const float  * __restrict__ B_scales, // [K/32,N]
        float        * __restrict__ C,        // row-major f32 [M,N]
        int M, int N, int K) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int row_a  = tile_m + lane_lo;
    const int col_b  = tile_n + lane_lo;
    const int nblk32 = K >> 5;

    float acc_f[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int kb = 0; kb < nblk32; ++kb) {
        rdna_i8_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int kk16 = 0; kk16 < 32; kk16 += 16) {
            rdna_i8_v4i32 a_frag;
            rdna_i8_v4i32 b_frag;

#pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint32_t aw = 0;
                uint32_t bw = 0;
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int k = 32 * kb + kk16 + 4 * g + byte;
                    const int8_t av = A[row_a * K + k];
                    const int8_t bv = B[k * N + col_b];
                    aw |= static_cast<uint32_t>(static_cast<uint8_t>(av)) << (8 * byte);
                    bw |= static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * byte);
                }
                a_frag[g] = static_cast<int>(aw);
                b_frag[g] = static_cast<int>(bw);
            }

            acc_i = rdna_i8_wmma(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int out_m = tile_m + 2 * r + lane_hi;
            const int out_n = tile_n + lane_lo;
            const float scale = A_scales[out_m * nblk32 + kb] * B_scales[kb * N + out_n];
            acc_f[r] += static_cast<float>(acc_i[r]) * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        C[out_m * N + out_n] = acc_f[r];
    }
}

static __device__ __forceinline__ int8_t q4_0_get_signed_device(
        const uint8_t * __restrict__ B_q4, int kb, int n, int kk, int N) {
    const int byte_idx = kk & 15;
    const uint8_t packed = B_q4[(static_cast<size_t>(kb) * N + n) * 16 + byte_idx];
    const uint8_t q = kk < 16 ? (packed & 0x0fu) : (packed >> 4);
    return static_cast<int8_t>(static_cast<int>(q) - 8);
}

static __global__ __launch_bounds__(32, 8) void quantize_f32_to_i8_k32_kernel(
        const float * __restrict__ X,         // row-major f32 activation [M,K]
        int8_t      * __restrict__ A,         // row-major int8 activation [M,K]
        float       * __restrict__ A_scales,  // [M,K/32]
        int M, int K) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int m = static_cast<int>(blockIdx.y);
    const int kb = static_cast<int>(blockIdx.x);
    const int k = 32 * kb + lane;
    float v = X[m * K + k];
    float max_abs = fabsf(v);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, mask, 32));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    int q = static_cast<int>(nearbyintf(v / scale));
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    A[m * K + k] = static_cast<int8_t>(q);
    if (lane == 0) {
        A_scales[m * (K >> 5) + kb] = scale;
    }
}

static __global__ __launch_bounds__(32, 8) void rdna_i8_wmma_q4_0_gemm_kernel(
        const int8_t * __restrict__ A,         // row-major int8 activation [M,K]
        const float  * __restrict__ A_scales,  // [M,K/32]
        const uint8_t * __restrict__ B_q4,     // Q4_0-style [K/32,N,16 bytes]
        const float  * __restrict__ B_scales,  // [K/32,N]
        float        * __restrict__ C,         // row-major f32 [M,N]
        int M, int N, int K) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = static_cast<int>(blockIdx.y) * 16;
    const int tile_n = static_cast<int>(blockIdx.x) * 16;
    const int row_a  = tile_m + lane_lo;
    const int col_b  = tile_n + lane_lo;
    const int nblk32 = K >> 5;

    float acc_f[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int kb = 0; kb < nblk32; ++kb) {
        rdna_i8_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int kk16 = 0; kk16 < 32; kk16 += 16) {
            rdna_i8_v4i32 a_frag;
            rdna_i8_v4i32 b_frag;

#pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint32_t aw = 0;
                uint32_t bw = 0;
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int kk = kk16 + 4 * g + byte;
                    const int k = 32 * kb + kk;
                    const int8_t av = A[row_a * K + k];
                    const int8_t bv = q4_0_get_signed_device(B_q4, kb, col_b, kk, N);
                    aw |= static_cast<uint32_t>(static_cast<uint8_t>(av)) << (8 * byte);
                    bw |= static_cast<uint32_t>(static_cast<uint8_t>(bv)) << (8 * byte);
                }
                a_frag[g] = static_cast<int>(aw);
                b_frag[g] = static_cast<int>(bw);
            }

            acc_i = rdna_i8_wmma(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int out_m = tile_m + 2 * r + lane_hi;
            const int out_n = tile_n + lane_lo;
            const float scale = A_scales[out_m * nblk32 + kb] * B_scales[kb * N + out_n];
            acc_f[r] += static_cast<float>(acc_i[r]) * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        C[out_m * N + out_n] = acc_f[r];
    }
}

static void cpu_gemm_i8_i32(const std::vector<int8_t> & A, const std::vector<int8_t> & B, std::vector<int32_t> & C,
                            int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
}

static void cpu_gemm_i8_q8scale_f32(const std::vector<int8_t> & A, const std::vector<float> & A_scales,
                                     const std::vector<int8_t> & B, const std::vector<float> & B_scales,
                                     std::vector<float> & C, int M, int N, int K) {
    const int nblk32 = K >> 5;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int kb = 0; kb < nblk32; ++kb) {
                int32_t dot = 0;
                for (int kk = 0; kk < 32; ++kk) {
                    const int k = 32 * kb + kk;
                    dot += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
                }
                acc += static_cast<float>(dot) * A_scales[m * nblk32 + kb] * B_scales[kb * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

static void cpu_quantize_f32_to_i8_k32(const std::vector<float> & X, std::vector<int8_t> & A,
                                       std::vector<float> & A_scales, int M, int K) {
    const int nblk32 = K >> 5;
    for (int m = 0; m < M; ++m) {
        for (int kb = 0; kb < nblk32; ++kb) {
            float max_abs = 0.0f;
            for (int kk = 0; kk < 32; ++kk) {
                max_abs = fmaxf(max_abs, fabsf(X[m * K + 32 * kb + kk]));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            A_scales[m * nblk32 + kb] = scale;
            for (int kk = 0; kk < 32; ++kk) {
                int q = static_cast<int>(nearbyintf(X[m * K + 32 * kb + kk] / scale));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                A[m * K + 32 * kb + kk] = static_cast<int8_t>(q);
            }
        }
    }
}

static int8_t q4_0_get_signed_host(const std::vector<uint8_t> & B_q4, int kb, int n, int kk, int N) {
    const int byte_idx = kk & 15;
    const uint8_t packed = B_q4[(static_cast<size_t>(kb) * N + n) * 16 + byte_idx];
    const uint8_t q = kk < 16 ? (packed & 0x0fu) : (packed >> 4);
    return static_cast<int8_t>(static_cast<int>(q) - 8);
}

static void cpu_gemm_i8_q4_0_f32(const std::vector<int8_t> & A, const std::vector<float> & A_scales,
                                 const std::vector<uint8_t> & B_q4, const std::vector<float> & B_scales,
                                 std::vector<float> & C, int M, int N, int K) {
    const int nblk32 = K >> 5;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int kb = 0; kb < nblk32; ++kb) {
                int32_t dot = 0;
                for (int kk = 0; kk < 32; ++kk) {
                    const int k = 32 * kb + kk;
                    dot += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(q4_0_get_signed_host(B_q4, kb, n, kk, N));
                }
                acc += static_cast<float>(dot) * A_scales[m * nblk32 + kb] * B_scales[kb * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

static int parse_arg(const char * s, int fallback) {
    if (s == nullptr) {
        return fallback;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    return end != s && v > 0 ? static_cast<int>(v) : fallback;
}

static int parse_arg_nonnegative(const char * s, int fallback) {
    if (s == nullptr) {
        return fallback;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    return end != s && v >= 0 ? static_cast<int>(v) : fallback;
}

int main(int argc, char ** argv) {
    const int M = parse_arg(argc > 1 ? argv[1] : nullptr, 32);
    const int N = parse_arg(argc > 2 ? argv[2] : nullptr, 48);
    const int K = parse_arg(argc > 3 ? argv[3] : nullptr, 64);
    const int warmup = parse_arg_nonnegative(argc > 4 ? argv[4] : nullptr, 10);
    const int reps = parse_arg_nonnegative(argc > 5 ? argv[5] : nullptr, 50);
    const bool check = parse_arg_nonnegative(argc > 6 ? argv[6] : nullptr, 1) != 0;

    if ((M % 16) != 0 || (N % 16) != 0 || (K % 32) != 0) {
        std::fprintf(stderr, "M,N must be positive multiples of 16 and K a positive multiple of 32; got M=%d N=%d K=%d\n", M, N, K);
        return 2;
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("[rdna-i8-wmma] device=%d name=%s arch=%s M=%d N=%d K=%d warmup=%d reps=%d check=%d\n", device, prop.name, prop.gcnArchName, M, N, K, warmup, reps, check ? 1 : 0);

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(K) * N);
    std::vector<int32_t> C(static_cast<size_t>(M) * N, 0);
    std::vector<int32_t> Ref(static_cast<size_t>(M) * N, 0);

    std::mt19937 rng(0x52444e41u); // "RDNA"
    std::uniform_int_distribution<int> dist(-7, 7);
    for (auto & x : A) {
        x = static_cast<int8_t>(dist(rng));
    }
    for (auto & x : B) {
        x = static_cast<int8_t>(dist(rng));
    }

    int8_t * dA = nullptr;
    int8_t * dB = nullptr;
    int32_t * dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, A.size() * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dB, B.size() * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dC, C.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dA, A.data(), A.size() * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), B.size() * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, C.size() * sizeof(int32_t)));

    dim3 block(32, 1, 1);
    dim3 grid(static_cast<unsigned>(N / 16), static_cast<unsigned>(M / 16), 1);
    rdna_i8_wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if (check) {
        HIP_CHECK(hipMemcpy(C.data(), dC, C.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
        cpu_gemm_i8_i32(A, B, Ref, M, N, K);

        int mismatches = 0;
        int max_abs = 0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const int32_t got = C[m * N + n];
                const int32_t exp = Ref[m * N + n];
                const int diff = std::abs(static_cast<int>(got - exp));
                if (diff > max_abs) {
                    max_abs = diff;
                }
                if (got != exp) {
                    if (mismatches < 16) {
                        std::fprintf(stderr, "mismatch[%d,%d]: got=%d expected=%d\n", m, n, got, exp);
                    }
                    ++mismatches;
                }
            }
        }

        long long checksum = 0;
        for (int32_t v : C) {
            checksum += static_cast<long long>(v);
        }

        std::printf("[rdna-i8-wmma] dense_i32 checksum=%lld mismatches=%d max_abs_diff=%d\n", checksum, mismatches, max_abs);
        if (mismatches != 0) {
            HIP_CHECK(hipFree(dA));
            HIP_CHECK(hipFree(dB));
            HIP_CHECK(hipFree(dC));
            std::fprintf(stderr, "[rdna-i8-wmma] dense_i32 FAIL\n");
            return 1;
        }
    } else {
        std::printf("[rdna-i8-wmma] dense_i32 check=skipped\n");
    }

    const int nblk32 = K >> 5;
    std::vector<float> A_scales(static_cast<size_t>(M) * nblk32);
    std::vector<float> B_scales(static_cast<size_t>(nblk32) * N);
    std::vector<float> C_scaled(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> Ref_scaled(static_cast<size_t>(M) * N, 0.0f);
    std::uniform_real_distribution<float> sdist(0.001f, 0.03125f);
    for (auto & x : A_scales) {
        x = sdist(rng);
    }
    for (auto & x : B_scales) {
        x = sdist(rng);
    }

    float * dA_scales = nullptr;
    float * dB_scales = nullptr;
    float * dC_scaled = nullptr;
    HIP_CHECK(hipMalloc(&dA_scales, A_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB_scales, B_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC_scaled, C_scaled.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA_scales, A_scales.data(), A_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB_scales, B_scales.data(), B_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC_scaled, 0, C_scaled.size() * sizeof(float)));

    rdna_i8_wmma_q8scale_gemm_kernel<<<grid, block>>>(dA, dA_scales, dB, dB_scales, dC_scaled, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if (check) {
        HIP_CHECK(hipMemcpy(C_scaled.data(), dC_scaled, C_scaled.size() * sizeof(float), hipMemcpyDeviceToHost));
        cpu_gemm_i8_q8scale_f32(A, A_scales, B, B_scales, Ref_scaled, M, N, K);

        int scaled_mismatches = 0;
        float scaled_max_abs = 0.0f;
        double scaled_checksum = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const float got = C_scaled[m * N + n];
                const float exp = Ref_scaled[m * N + n];
                const float diff = std::fabs(got - exp);
                scaled_checksum += static_cast<double>(got);
                if (diff > scaled_max_abs) {
                    scaled_max_abs = diff;
                }
                if (diff > 1.0e-5f) {
                    if (scaled_mismatches < 16) {
                        std::fprintf(stderr, "scaled mismatch[%d,%d]: got=%.9g expected=%.9g diff=%.9g\n", m, n, got, exp, diff);
                    }
                    ++scaled_mismatches;
                }
            }
        }

        std::printf("[rdna-i8-wmma] q8scale_f32 checksum=%.9g mismatches=%d max_abs_diff=%.9g\n",
                scaled_checksum, scaled_mismatches, scaled_max_abs);
        if (scaled_mismatches != 0) {
            HIP_CHECK(hipFree(dA));
            HIP_CHECK(hipFree(dB));
            HIP_CHECK(hipFree(dC));
            HIP_CHECK(hipFree(dA_scales));
            HIP_CHECK(hipFree(dB_scales));
            HIP_CHECK(hipFree(dC_scaled));
            std::fprintf(stderr, "[rdna-i8-wmma] q8scale_f32 FAIL\n");
            return 1;
        }
    } else {
        std::printf("[rdna-i8-wmma] q8scale_f32 check=skipped\n");
    }

    std::vector<uint8_t> B_q4(static_cast<size_t>(nblk32) * N * 16, 0);
    std::vector<float> B_q4_scales(static_cast<size_t>(nblk32) * N);
    std::vector<float> C_q4(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> Ref_q4(static_cast<size_t>(M) * N, 0.0f);
    std::uniform_int_distribution<int> q4dist(-8, 7);
    for (int kb = 0; kb < nblk32; ++kb) {
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < 16; ++i) {
                const uint8_t lo = static_cast<uint8_t>(q4dist(rng) + 8) & 0x0fu;
                const uint8_t hi = static_cast<uint8_t>(q4dist(rng) + 8) & 0x0fu;
                B_q4[(static_cast<size_t>(kb) * N + n) * 16 + i] = static_cast<uint8_t>(lo | (hi << 4));
            }
            B_q4_scales[kb * N + n] = sdist(rng);
        }
    }

    uint8_t * dB_q4 = nullptr;
    float * dB_q4_scales = nullptr;
    float * dC_q4 = nullptr;
    HIP_CHECK(hipMalloc(&dB_q4, B_q4.size() * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&dB_q4_scales, B_q4_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC_q4, C_q4.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dB_q4, B_q4.data(), B_q4.size() * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB_q4_scales, B_q4_scales.data(), B_q4_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC_q4, 0, C_q4.size() * sizeof(float)));

    rdna_i8_wmma_q4_0_gemm_kernel<<<grid, block>>>(dA, dA_scales, dB_q4, dB_q4_scales, dC_q4, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if (check) {
        HIP_CHECK(hipMemcpy(C_q4.data(), dC_q4, C_q4.size() * sizeof(float), hipMemcpyDeviceToHost));
        cpu_gemm_i8_q4_0_f32(A, A_scales, B_q4, B_q4_scales, Ref_q4, M, N, K);

        int q4_mismatches = 0;
        float q4_max_abs = 0.0f;
        double q4_checksum = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const float got = C_q4[m * N + n];
                const float exp = Ref_q4[m * N + n];
                const float diff = std::fabs(got - exp);
                q4_checksum += static_cast<double>(got);
                if (diff > q4_max_abs) {
                    q4_max_abs = diff;
                }
                if (diff > 1.0e-5f) {
                    if (q4_mismatches < 16) {
                        std::fprintf(stderr, "q4 mismatch[%d,%d]: got=%.9g expected=%.9g diff=%.9g\n", m, n, got, exp, diff);
                    }
                    ++q4_mismatches;
                }
            }
        }

        std::printf("[rdna-i8-wmma] q4_0_f32 checksum=%.9g mismatches=%d max_abs_diff=%.9g\n",
                q4_checksum, q4_mismatches, q4_max_abs);
        if (q4_mismatches != 0) {
            HIP_CHECK(hipFree(dA));
            HIP_CHECK(hipFree(dB));
            HIP_CHECK(hipFree(dC));
            HIP_CHECK(hipFree(dA_scales));
            HIP_CHECK(hipFree(dB_scales));
            HIP_CHECK(hipFree(dC_scaled));
            HIP_CHECK(hipFree(dB_q4));
            HIP_CHECK(hipFree(dB_q4_scales));
            HIP_CHECK(hipFree(dC_q4));
            std::fprintf(stderr, "[rdna-i8-wmma] q4_0_f32 FAIL\n");
            return 1;
        }
    } else {
        std::printf("[rdna-i8-wmma] q4_0_f32 check=skipped\n");
    }

    std::vector<float> A_f32(static_cast<size_t>(M) * K);
    std::vector<int8_t> A_from_f32(static_cast<size_t>(M) * K, 0);
    std::vector<float> A_from_f32_scales(static_cast<size_t>(M) * nblk32, 0.0f);
    std::vector<float> C_q4_from_f32(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> Ref_q4_from_f32(static_cast<size_t>(M) * N, 0.0f);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    for (auto & x : A_f32) {
        x = fdist(rng);
    }

    float * dA_f32 = nullptr;
    int8_t * dA_from_f32 = nullptr;
    float * dA_from_f32_scales = nullptr;
    float * dC_q4_from_f32 = nullptr;
    HIP_CHECK(hipMalloc(&dA_f32, A_f32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dA_from_f32, A_from_f32.size() * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dA_from_f32_scales, A_from_f32_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC_q4_from_f32, C_q4_from_f32.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA_f32, A_f32.data(), A_f32.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dA_from_f32, 0, A_from_f32.size() * sizeof(int8_t)));
    HIP_CHECK(hipMemset(dA_from_f32_scales, 0, A_from_f32_scales.size() * sizeof(float)));
    HIP_CHECK(hipMemset(dC_q4_from_f32, 0, C_q4_from_f32.size() * sizeof(float)));

    dim3 qblock(32, 1, 1);
    dim3 qgrid(static_cast<unsigned>(nblk32), static_cast<unsigned>(M), 1);
    quantize_f32_to_i8_k32_kernel<<<qgrid, qblock>>>(dA_f32, dA_from_f32, dA_from_f32_scales, M, K);
    HIP_CHECK(hipGetLastError());
    rdna_i8_wmma_q4_0_gemm_kernel<<<grid, block>>>(dA_from_f32, dA_from_f32_scales, dB_q4, dB_q4_scales, dC_q4_from_f32, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if (check) {
        HIP_CHECK(hipMemcpy(C_q4_from_f32.data(), dC_q4_from_f32, C_q4_from_f32.size() * sizeof(float), hipMemcpyDeviceToHost));
        cpu_quantize_f32_to_i8_k32(A_f32, A_from_f32, A_from_f32_scales, M, K);
        cpu_gemm_i8_q4_0_f32(A_from_f32, A_from_f32_scales, B_q4, B_q4_scales, Ref_q4_from_f32, M, N, K);

        int f32_q4_mismatches = 0;
        float f32_q4_max_abs = 0.0f;
        double f32_q4_checksum = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const float got = C_q4_from_f32[m * N + n];
                const float exp = Ref_q4_from_f32[m * N + n];
                const float diff = std::fabs(got - exp);
                f32_q4_checksum += static_cast<double>(got);
                if (diff > f32_q4_max_abs) {
                    f32_q4_max_abs = diff;
                }
                if (diff > 1.0e-5f) {
                    if (f32_q4_mismatches < 16) {
                        std::fprintf(stderr, "f32_q4 mismatch[%d,%d]: got=%.9g expected=%.9g diff=%.9g\n", m, n, got, exp, diff);
                    }
                    ++f32_q4_mismatches;
                }
            }
        }

        std::printf("[rdna-i8-wmma] f32_quant_q4_0_f32 checksum=%.9g mismatches=%d max_abs_diff=%.9g\n",
                f32_q4_checksum, f32_q4_mismatches, f32_q4_max_abs);
        if (f32_q4_mismatches != 0) {
            std::fprintf(stderr, "[rdna-i8-wmma] f32_quant_q4_0_f32 FAIL\n");
            return 1;
        }
    } else {
        std::printf("[rdna-i8-wmma] f32_quant_q4_0_f32 check=skipped\n");
    }

    auto time_kernel = [&](const char * label, double work_items, auto launch) {
        for (int i = 0; i < warmup; ++i) {
            launch();
        }
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < reps; ++i) {
            launch();
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        const double ms = static_cast<double>(elapsed_ms) / static_cast<double>(reps);
        const double work_gop_s = work_items / (ms * 1.0e6);
        std::printf("[rdna-i8-wmma] timing_%s avg_ms=%.6f work_gop_s=%.3f warmup=%d reps=%d\n",
                label, ms, work_gop_s, warmup, reps);
    };

    if (reps > 0) {
        const double gemm_work = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        const double quant_work = static_cast<double>(M) * static_cast<double>(K);
        time_kernel("dense_i32", gemm_work, [&]() {
            rdna_i8_wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            HIP_CHECK(hipGetLastError());
        });
        time_kernel("q8scale_f32", gemm_work, [&]() {
            rdna_i8_wmma_q8scale_gemm_kernel<<<grid, block>>>(dA, dA_scales, dB, dB_scales, dC_scaled, M, N, K);
            HIP_CHECK(hipGetLastError());
        });
        time_kernel("q4_0_f32", gemm_work, [&]() {
            rdna_i8_wmma_q4_0_gemm_kernel<<<grid, block>>>(dA, dA_scales, dB_q4, dB_q4_scales, dC_q4, M, N, K);
            HIP_CHECK(hipGetLastError());
        });
        time_kernel("quant_f32_i8_k32", quant_work, [&]() {
            quantize_f32_to_i8_k32_kernel<<<qgrid, qblock>>>(dA_f32, dA_from_f32, dA_from_f32_scales, M, K);
            HIP_CHECK(hipGetLastError());
        });
        time_kernel("quant_f32_plus_q4_0_f32", gemm_work, [&]() {
            quantize_f32_to_i8_k32_kernel<<<qgrid, qblock>>>(dA_f32, dA_from_f32, dA_from_f32_scales, M, K);
            HIP_CHECK(hipGetLastError());
            rdna_i8_wmma_q4_0_gemm_kernel<<<grid, block>>>(dA_from_f32, dA_from_f32_scales, dB_q4, dB_q4_scales, dC_q4_from_f32, M, N, K);
            HIP_CHECK(hipGetLastError());
        });
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(dA_scales));
    HIP_CHECK(hipFree(dB_scales));
    HIP_CHECK(hipFree(dC_scaled));
    HIP_CHECK(hipFree(dB_q4));
    HIP_CHECK(hipFree(dB_q4_scales));
    HIP_CHECK(hipFree(dC_q4));
    HIP_CHECK(hipFree(dA_f32));
    HIP_CHECK(hipFree(dA_from_f32));
    HIP_CHECK(hipFree(dA_from_f32_scales));
    HIP_CHECK(hipFree(dC_q4_from_f32));

    if (check) {
        std::printf("[rdna-i8-wmma] PASS native W8A8/I32 WMMA GEMM, Q8-style, and Q4_0-style scale epilogues match CPU reference\n");
    } else {
        std::printf("[rdna-i8-wmma] PASS bench-only native W8A8/I32 WMMA GEMM, Q8-style, and Q4_0-style scale epilogues completed\n");
    }
    return 0;
}
