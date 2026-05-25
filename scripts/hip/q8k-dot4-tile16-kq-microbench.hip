// Standalone ROCm/RDNA3 q8_0-K DOT4 KQ microbench.
//
// Purpose: prototype the combined design from
// docs/rocm-tbq4-paths/24-wmma-sized-dot4-tile-plan.md:
//   - WMMA-sized 16x16 KQ macro tile / work decomposition
//   - packed INT8 DOT4 inner product over D=256
//   - compare ggml q8_0 34B block layout vs packed16 payload + separate scales
//
// This is intentionally not a full FlashAttention route.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static constexpr int D = 256;
static constexpr int QK8 = 32;
static constexpr int BLOCKS = D / QK8;
static constexpr int I32_PER_ROW = D / 4;
static constexpr int I32_PER_Q8_BLOCK = QK8 / 4;
static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int Q8_BLOCK_BYTES = int(sizeof(uint16_t) + QK8);
static constexpr int Q8_ROW_BYTES = BLOCKS * Q8_BLOCK_BYTES;

static __device__ __forceinline__ int dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const int8_t * aa = reinterpret_cast<const int8_t *>(&a);
    const int8_t * bb = reinterpret_cast<const int8_t *>(&b);
    return c + int(aa[0])*int(bb[0]) + int(aa[1])*int(bb[1]) + int(aa[2])*int(bb[2]) + int(aa[3])*int(bb[3]);
#endif
}

static __device__ __forceinline__ int load_i32_unaligned(const uint8_t * p) {
    int v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ __half load_half_unaligned(const uint8_t * p) {
    __half v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

__global__ __launch_bounds__(256, 1) void tile16_q8block_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const uint8_t * __restrict__ k_q8block,
        float       * __restrict__ out,
        int nq,
        int nk) {
    const int tid = int(threadIdx.x);
    const int qm = tid >> 4;
    const int kn = tid & 15;
    const int q = int(blockIdx.y) * TILE_M + qm;
    const int k = int(blockIdx.x) * TILE_N + kn;
    if (q >= nq || k >= nk) {
        return;
    }

    const int * q_row = q_payload + size_t(q) * I32_PER_ROW;
    const float * qs = q_scales + size_t(q) * BLOCKS;
    const uint8_t * k_row = k_q8block + size_t(k) * Q8_ROW_BYTES;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < BLOCKS; ++qb) {
        const uint8_t * k_blk = k_row + qb * Q8_BLOCK_BYTES;
        const float ks = __half2float(load_half_unaligned(k_blk));
        int acc = 0;
#pragma unroll
        for (int i = 0; i < I32_PER_Q8_BLOCK; ++i) {
            const int qv = q_row[qb * I32_PER_Q8_BLOCK + i];
            const int kv = load_i32_unaligned(k_blk + sizeof(uint16_t) + 4 * i);
            acc = dot4_i8_i8(qv, kv, acc);
        }
        sum += float(acc) * qs[qb] * ks;
    }
    out[size_t(q) * nk + k] = sum;
}

__global__ __launch_bounds__(256, 1) void pack_q8block_to_packed16_kernel(
        const uint8_t * __restrict__ k_q8block,
        int           * __restrict__ k_payload,
        __half        * __restrict__ k_scales,
        int nk) {
    const int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = nk * BLOCKS;
    if (linear >= total) {
        return;
    }
    const int qb = linear % BLOCKS;
    const int k  = linear / BLOCKS;
    const uint8_t * src = k_q8block + size_t(k) * Q8_ROW_BYTES + qb * Q8_BLOCK_BYTES;
    int * dst = k_payload + size_t(k) * I32_PER_ROW + qb * I32_PER_Q8_BLOCK;
    k_scales[size_t(k) * BLOCKS + qb] = load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < I32_PER_Q8_BLOCK; ++i) {
        dst[i] = load_i32_unaligned(src + sizeof(uint16_t) + 4 * i);
    }
}

__global__ __launch_bounds__(256, 1) void tile16_packed16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const __half * __restrict__ k_scales,
        float       * __restrict__ out,
        int nq,
        int nk) {
    const int tid = int(threadIdx.x);
    const int qm = tid >> 4;
    const int kn = tid & 15;
    const int q = int(blockIdx.y) * TILE_M + qm;
    const int k = int(blockIdx.x) * TILE_N + kn;
    if (q >= nq || k >= nk) {
        return;
    }

    const int * q_row = q_payload + size_t(q) * I32_PER_ROW;
    const float * qs = q_scales + size_t(q) * BLOCKS;
    const int * k_row = k_payload + size_t(k) * I32_PER_ROW;
    const __half * ks_row = k_scales + size_t(k) * BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < BLOCKS; ++qb) {
        int acc = 0;
#pragma unroll
        for (int i = 0; i < I32_PER_Q8_BLOCK; ++i) {
            const int qv = q_row[qb * I32_PER_Q8_BLOCK + i];
            const int kv = k_row[qb * I32_PER_Q8_BLOCK + i];
            acc = dot4_i8_i8(qv, kv, acc);
        }
        sum += float(acc) * qs[qb] * __half2float(ks_row[qb]);
    }
    out[size_t(q) * nk + k] = sum;
}

static int pack4_i8(const int8_t * p) {
    int v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static float elapsed_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
}

template <typename Fn>
static float time_kernel(Fn && fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop) / float(reps);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
}

static double rel_rms(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d*d;
        den += double(b[i]) * double(b[i]);
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(double(a[i]) - double(b[i])));
    }
    return m;
}

int main(int argc, char ** argv) {
    int nq = 256;
    int nk = 1024;
    int warmup = 20;
    int reps = 100;
    uint32_t seed = 12345;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_int = [&](int & dst) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", arg.c_str()); std::exit(2); }
            dst = std::atoi(argv[++i]);
        };
        if (arg == "--nq") next_int(nq);
        else if (arg == "--nk") next_int(nk);
        else if (arg == "--warmup") next_int(warmup);
        else if (arg == "--reps") next_int(reps);
        else if (arg == "--seed") { int tmp = int(seed); next_int(tmp); seed = uint32_t(tmp); }
        else if (arg == "--help" || arg == "-h") {
            printf("usage: %s [--nq N] [--nk N] [--warmup N] [--reps N] [--seed N]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (nq <= 0 || nk <= 0) {
        fprintf(stderr, "nq/nk must be positive\n");
        return 2;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qi(-127, 127);
    std::uniform_real_distribution<float> qs_dist(0.002f, 0.02f);

    std::vector<int> h_q_payload(size_t(nq) * I32_PER_ROW);
    std::vector<float> h_q_scales(size_t(nq) * BLOCKS);
    std::vector<uint8_t> h_k_q8block(size_t(nk) * Q8_ROW_BYTES);
    std::vector<int> h_k_payload(size_t(nk) * I32_PER_ROW);
    std::vector<__half> h_k_scales(size_t(nk) * BLOCKS);
    std::vector<float> h_ref(size_t(nq) * nk);

    int8_t tmp[4];
    for (int q = 0; q < nq; ++q) {
        for (int qb = 0; qb < BLOCKS; ++qb) {
            h_q_scales[size_t(q) * BLOCKS + qb] = qs_dist(rng);
        }
        for (int i = 0; i < I32_PER_ROW; ++i) {
            for (int b = 0; b < 4; ++b) tmp[b] = int8_t(qi(rng));
            h_q_payload[size_t(q) * I32_PER_ROW + i] = pack4_i8(tmp);
        }
    }

    for (int k = 0; k < nk; ++k) {
        for (int qb = 0; qb < BLOCKS; ++qb) {
            const float ksf = qs_dist(rng);
            const __half kh = __float2half(ksf);
            h_k_scales[size_t(k) * BLOCKS + qb] = kh;
            uint8_t * blk = h_k_q8block.data() + size_t(k) * Q8_ROW_BYTES + qb * Q8_BLOCK_BYTES;
            std::memcpy(blk, &kh, sizeof(uint16_t));
            for (int i = 0; i < I32_PER_Q8_BLOCK; ++i) {
                for (int b = 0; b < 4; ++b) tmp[b] = int8_t(qi(rng));
                const int v = pack4_i8(tmp);
                h_k_payload[size_t(k) * I32_PER_ROW + qb * I32_PER_Q8_BLOCK + i] = v;
                std::memcpy(blk + sizeof(uint16_t) + 4 * i, &v, sizeof(v));
            }
        }
    }

    for (int q = 0; q < nq; ++q) {
        for (int k = 0; k < nk; ++k) {
            float sum = 0.0f;
            for (int qb = 0; qb < BLOCKS; ++qb) {
                int acc = 0;
                for (int i = 0; i < I32_PER_Q8_BLOCK; ++i) {
                    const int qv = h_q_payload[size_t(q) * I32_PER_ROW + qb * I32_PER_Q8_BLOCK + i];
                    const int kv = h_k_payload[size_t(k) * I32_PER_ROW + qb * I32_PER_Q8_BLOCK + i];
                    const int8_t * qa = reinterpret_cast<const int8_t *>(&qv);
                    const int8_t * ka = reinterpret_cast<const int8_t *>(&kv);
                    for (int b = 0; b < 4; ++b) acc += int(qa[b]) * int(ka[b]);
                }
                sum += float(acc) * h_q_scales[size_t(q) * BLOCKS + qb] * __half2float(h_k_scales[size_t(k) * BLOCKS + qb]);
            }
            h_ref[size_t(q) * nk + k] = sum;
        }
    }

    int * d_q_payload = nullptr;
    float * d_q_scales = nullptr;
    uint8_t * d_k_q8block = nullptr;
    int * d_k_payload = nullptr;
    __half * d_k_scales = nullptr;
    float * d_out_q8block = nullptr;
    float * d_out_packed = nullptr;
    HIP_CHECK(hipMalloc(&d_q_payload, h_q_payload.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_q_scales, h_q_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k_q8block, h_k_q8block.size() * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&d_k_payload, h_k_payload.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_k_scales, h_k_scales.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out_q8block, h_ref.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_packed, h_ref.size() * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_q_payload, h_q_payload.data(), h_q_payload.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q_scales, h_q_scales.data(), h_q_scales.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k_q8block, h_k_q8block.data(), h_k_q8block.size() * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k_payload, h_k_payload.data(), h_k_payload.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k_scales, h_k_scales.data(), h_k_scales.size() * sizeof(__half), hipMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 tile_grid((nk + TILE_N - 1) / TILE_N, (nq + TILE_M - 1) / TILE_M);
    const dim3 pack_grid((nk * BLOCKS + 255) / 256);

    auto launch_pack = [&]() {
        pack_q8block_to_packed16_kernel<<<pack_grid, block>>>(d_k_q8block, d_k_payload, d_k_scales, nk);
        HIP_CHECK(hipGetLastError());
    };
    auto launch_q8block = [&]() {
        tile16_q8block_kernel<<<tile_grid, block>>>(d_q_payload, d_q_scales, d_k_q8block, d_out_q8block, nq, nk);
        HIP_CHECK(hipGetLastError());
    };
    auto launch_packed = [&]() {
        tile16_packed16_kernel<<<tile_grid, block>>>(d_q_payload, d_q_scales, d_k_payload, d_k_scales, d_out_packed, nq, nk);
        HIP_CHECK(hipGetLastError());
    };
    auto launch_pack_plus_packed = [&]() {
        pack_q8block_to_packed16_kernel<<<pack_grid, block>>>(d_k_q8block, d_k_payload, d_k_scales, nk);
        tile16_packed16_kernel<<<tile_grid, block>>>(d_q_payload, d_q_scales, d_k_payload, d_k_scales, d_out_packed, nq, nk);
        HIP_CHECK(hipGetLastError());
    };

    launch_pack();
    launch_q8block();
    launch_packed();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_out_q8block(h_ref.size());
    std::vector<float> h_out_packed(h_ref.size());
    HIP_CHECK(hipMemcpy(h_out_q8block.data(), d_out_q8block, h_out_q8block.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_out_packed.data(), d_out_packed, h_out_packed.size() * sizeof(float), hipMemcpyDeviceToHost));

    const float pack_ms = time_kernel(launch_pack, warmup, reps);
    const float q8block_ms = time_kernel(launch_q8block, warmup, reps);
    const float packed_ms = time_kernel(launch_packed, warmup, reps);
    const float pack_plus_packed_ms = time_kernel(launch_pack_plus_packed, warmup, reps);

    const double dot4_ops = double(nq) * double(nk) * double(D / 4);
    const double q8block_gops = dot4_ops / (double(q8block_ms) * 1.0e6);
    const double packed_gops = dot4_ops / (double(packed_ms) * 1.0e6);

    printf("{\n");
    printf("  \"kind\": \"q8k_dot4_tile16_kq_microbench\",\n");
    printf("  \"nq\": %d, \"nk\": %d, \"d\": %d, \"tile_m\": %d, \"tile_n\": %d,\n", nq, nk, D, TILE_M, TILE_N);
    printf("  \"warmup\": %d, \"reps\": %d, \"seed\": %u,\n", warmup, reps, seed);
    printf("  \"dot4_per_pair\": %d, \"dot4_per_16x16_tile\": %d,\n", D / 4, TILE_M * TILE_N * (D / 4));
    printf("  \"layouts\": {\n");
    printf("    \"q8_0_block_row_bytes\": %d,\n", Q8_ROW_BYTES);
    printf("    \"packed16_payload_row_bytes\": %d,\n", D);
    printf("    \"separate_scale_row_bytes\": %zu\n", size_t(BLOCKS) * sizeof(__half));
    printf("  },\n");
    printf("  \"timings\": {\n");
    printf("    \"pack_q8block_to_packed16_ms\": %.9g,\n", pack_ms);
    printf("    \"tile16_q8block_ms\": %.9g,\n", q8block_ms);
    printf("    \"tile16_packed16_ms\": %.9g,\n", packed_ms);
    printf("    \"pack_plus_tile16_packed16_ms\": %.9g,\n", pack_plus_packed_ms);
    printf("    \"speedup_packed16_vs_q8block\": %.9g,\n", double(q8block_ms) / double(packed_ms));
    printf("    \"speedup_pack_plus_packed16_vs_q8block\": %.9g,\n", double(q8block_ms) / double(pack_plus_packed_ms));
    printf("    \"q8block_dot4_gops\": %.9g,\n", q8block_gops);
    printf("    \"packed16_dot4_gops\": %.9g\n", packed_gops);
    printf("  },\n");
    printf("  \"correctness\": {\n");
    printf("    \"q8block_rel_rms_gpu_vs_ref\": %.9g,\n", rel_rms(h_out_q8block, h_ref));
    printf("    \"packed16_rel_rms_gpu_vs_ref\": %.9g,\n", rel_rms(h_out_packed, h_ref));
    printf("    \"q8block_max_abs_gpu_vs_ref\": %.9g,\n", max_abs_diff(h_out_q8block, h_ref));
    printf("    \"packed16_max_abs_gpu_vs_ref\": %.9g\n", max_abs_diff(h_out_packed, h_ref));
    printf("  },\n");
    printf("  \"contract\": {\n");
    printf("    \"macro_tile\": \"16x16 output tile over D256, WMMA-sized work decomposition\",\n");
    printf("    \"inner_product\": \"64 packed int8 dot4 ops per Q/K pair\",\n");
    printf("    \"q8block_meaning\": \"ggml q8_0-like 34B blocks: half scale plus int8 payload\",\n");
    printf("    \"packed16_meaning\": \"K int8 payload-only rows, 16B-friendly contiguous payload plus separate scales\"\n");
    printf("  }\n");
    printf("}\n");

    HIP_CHECK(hipFree(d_q_payload));
    HIP_CHECK(hipFree(d_q_scales));
    HIP_CHECK(hipFree(d_k_q8block));
    HIP_CHECK(hipFree(d_k_payload));
    HIP_CHECK(hipFree(d_k_scales));
    HIP_CHECK(hipFree(d_out_q8block));
    HIP_CHECK(hipFree(d_out_packed));
    return 0;
}
