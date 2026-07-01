// Standalone HIP/hipBLASLt dense int8 GEMM probe for RDNA/ROCm.
// Computes C[M,N] = A[M,K] * B[K,N] with int8 inputs and int32 output via column-major transposed view.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); }} while (0)
#define HBLASLT_CHECK(x) do { hipblasStatus_t s = (x); if (s != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLASLt error %s:%d: %d\n", __FILE__, __LINE__, (int)s); std::exit(1); }} while (0)

static int arg_int(int argc, char ** argv, const char * name, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return std::atoi(argv[i + 1]);
    }
    return def;
}

static bool arg_has(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == name) return true;
    }
    return false;
}

static void print_usage(const char * argv0) {
    std::printf("usage: %s [--m M] [--n N] [--k K] [--warmup N] [--reps N] [--algos N]\n", argv0);
    std::printf("  Dense int8 hipBLASLt GEMM probe: C[M,N] = A[M,K] * B[K,N], int32 accumulation.\n");
    std::printf("  Emits legacy [hipblaslt-i8] lines plus parseable [hipblaslt-i8-result] and [hipblaslt-i8-best] lines.\n");
}

static void print_result_line(int M, int N, int K, int algo_idx, size_t workspace, int state,
        const char * status, double avg_ms, double tmac_s) {
    std::printf("[hipblaslt-i8-result] M=%d N=%d K=%d algo_ord=%d solution_index=NA workspace=%zu state=%d status=%s",
            M, N, K, algo_idx, workspace, state, status);
    if (avg_ms >= 0.0) {
        std::printf(" avg_ms=%.6f tmac_s=%.3f", avg_ms, tmac_s);
    }
    std::printf("\n");
}

int main(int argc, char ** argv) {
    if (arg_has(argc, argv, "--help") || arg_has(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const int M = arg_int(argc, argv, "--m", 128);
    const int N = arg_int(argc, argv, "--n", 4096);
    const int K = arg_int(argc, argv, "--k", 4096);
    const int warmup = arg_int(argc, argv, "--warmup", 10);
    const int reps = arg_int(argc, argv, "--reps", 50);
    const int requested = arg_int(argc, argv, "--algos", 8);
    if (M <= 0 || N <= 0 || K <= 0 || warmup < 0 || reps <= 0 || requested <= 0) {
        std::fprintf(stderr, "invalid arguments: M/N/K/reps/algos must be positive and warmup must be non-negative\n");
        return 1;
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("[hipblaslt-i8] device=%s arch=%s M=%d N=%d K=%d warmup=%d reps=%d\n", prop.name, prop.gcnArchName, M, N, K, warmup, reps);

    std::vector<int8_t> hA((size_t) M*K), hB((size_t) K*N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t) ((int)(i*13 + 7) % 255 - 127);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t) ((int)(i*17 + 3) % 255 - 127);

    int8_t * dA = nullptr;
    int8_t * dB = nullptr;
    int32_t * dC = nullptr;
    void * workspace = nullptr;
    const size_t workspace_bytes = 64ull << 20;
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dB, hB.size()));
    HIP_CHECK(hipMalloc(&dC, (size_t) M*N*sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, (size_t) M*N*sizeof(int32_t)));

    hipblasLtHandle_t handle = nullptr;
    HBLASLT_CHECK(hipblasLtCreate(&handle));

    hipblasLtMatmulDesc_t op = nullptr;
    HBLASLT_CHECK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));

    // Column-major GEMM: C_col(N x M) = B_col(N x K) * A_col(K x M).
    hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    HBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_8I,  N, K, N));
    HBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_8I,  K, M, K));
    HBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_32I, N, M, N));
    HBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Ddesc, HIP_R_32I, N, M, N));

    hipblasLtMatmulPreference_t pref = nullptr;
    HBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t ws = workspace_bytes;
    HBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    std::vector<hipblasLtMatmulHeuristicResult_t> heur(std::max(1, requested));
    int returned = 0;
    hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(handle, op, Adesc, Bdesc, Cdesc, Ddesc, pref,
            (int) heur.size(), heur.data(), &returned);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
        std::fprintf(stderr, "hipBLASLt heuristic failed status=%d returned=%d\n", (int) hs, returned);
        return 2;
    }
    std::printf("[hipblaslt-i8] heuristic_returned=%d workspace0=%zu state0=%d\n", returned, heur[0].workspaceSize, (int) heur[0].state);

    const int32_t alpha = 1;
    const int32_t beta = 0;
    double best_ms = 1.0e30;
    int best_algo = -1;
    for (int algo_idx = 0; algo_idx < returned; ++algo_idx) {
        auto launch = [&]() -> hipblasStatus_t {
            return hipblasLtMatmul(handle, op,
                    &alpha,
                    dB, Adesc,
                    dA, Bdesc,
                    &beta,
                    dC, Cdesc,
                    dC, Ddesc,
                    &heur[algo_idx].algo,
                    workspace, workspace_bytes,
                    0);
        };

        bool ok = true;
        for (int i = 0; i < warmup; ++i) {
            hipblasStatus_t st = launch();
            if (st != HIPBLAS_STATUS_SUCCESS) {
                std::printf("[hipblaslt-i8] algo=%d warmup_status=%d skip\n", algo_idx, (int) st);
                print_result_line(M, N, K, algo_idx, heur[algo_idx].workspaceSize, (int) heur[algo_idx].state, "warmup_failed", -1.0, -1.0);
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t start = nullptr, stop = nullptr;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < reps; ++i) {
            hipblasStatus_t st = launch();
            if (st != HIPBLAS_STATUS_SUCCESS) {
                std::printf("[hipblaslt-i8] algo=%d run_status=%d skip\n", algo_idx, (int) st);
                print_result_line(M, N, K, algo_idx, heur[algo_idx].workspaceSize, (int) heur[algo_idx].state, "run_failed", -1.0, -1.0);
                ok = false;
                break;
            }
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        if (!ok) {
            continue;
        }
        const double avg_ms = (double) elapsed_ms / reps;
        const double tmac_s = ((double) M * N * K) / (avg_ms * 1.0e9);
        std::printf("[hipblaslt-i8] algo=%d workspace=%zu avg_ms=%.6f tmac_s=%.3f\n", algo_idx, heur[algo_idx].workspaceSize, avg_ms, tmac_s);
        print_result_line(M, N, K, algo_idx, heur[algo_idx].workspaceSize, (int) heur[algo_idx].state, "ok", avg_ms, tmac_s);
        if (avg_ms < best_ms) {
            best_ms = avg_ms;
            best_algo = algo_idx;
        }
    }
    if (best_algo < 0) {
        std::fprintf(stderr, "no working hipBLASLt algorithms\n");
        return 3;
    }
    const double best_tmac_s = ((double) M * N * K) / (best_ms * 1.0e9);
    std::printf("[hipblaslt-i8] best_algo=%d best_avg_ms=%.6f best_tmac_s=%.3f\n", best_algo, best_ms, best_tmac_s);
    std::printf("[hipblaslt-i8-best] M=%d N=%d K=%d algo_ord=%d solution_index=NA avg_ms=%.6f tmac_s=%.3f\n",
            M, N, K, best_algo, best_ms, best_tmac_s);

    HBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    HBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(Adesc));
    HBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(Bdesc));
    HBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(Cdesc));
    HBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(Ddesc));
    HBLASLT_CHECK(hipblasLtMatmulDescDestroy(op));
    HBLASLT_CHECK(hipblasLtDestroy(handle));
    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
