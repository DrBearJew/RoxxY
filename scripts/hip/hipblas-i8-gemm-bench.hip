// Standalone HIP/hipBLAS dense int8 GEMM probe for RDNA/ROCm.
// Computes C[M,N] = A[M,K] * B[K,N] with int8 inputs and int32 output.
// Uses column-major hipBLAS by storing A_col as KxM transposed view and B_col as NxK transposed view,
// then requests C_col[N,M] = B_col[N,K] * A_col[K,M].
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); }} while (0)
#define HBLAS_CHECK(x) do { hipblasStatus_t s = (x); if (s != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); std::exit(1); }} while (0)

static int arg_int(int argc, char ** argv, const char * name, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return std::atoi(argv[i + 1]);
    }
    return def;
}

int main(int argc, char ** argv) {
    const int M = arg_int(argc, argv, "--m", 128);
    const int N = arg_int(argc, argv, "--n", 4096);
    const int K = arg_int(argc, argv, "--k", 4096);
    const int warmup = arg_int(argc, argv, "--warmup", 10);
    const int reps = arg_int(argc, argv, "--reps", 50);

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("[hipblas-i8] device=%s arch=%s M=%d N=%d K=%d warmup=%d reps=%d\n", prop.name, prop.gcnArchName, M, N, K, warmup, reps);

    std::vector<int8_t> hA((size_t) M*K), hB((size_t) K*N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t) ((int)(i*13 + 7) % 255 - 127);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t) ((int)(i*17 + 3) % 255 - 127);

    int8_t * dA = nullptr;
    int8_t * dB = nullptr;
    int32_t * dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, hA.size()));
    HIP_CHECK(hipMalloc(&dB, hB.size()));
    HIP_CHECK(hipMalloc(&dC, (size_t) M*N*sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, (size_t) M*N*sizeof(int32_t)));

    hipblasHandle_t handle = nullptr;
    HBLAS_CHECK(hipblasCreate(&handle));

    // Column-major GEMM: C_col(N x M) = B_col(N x K) * A_col(K x M).
    // dB is row-major [K,N], identical memory to column-major [N,K].
    // dA is row-major [M,K], identical memory to column-major [K,M].
    // dC becomes column-major [N,M], identical memory to row-major [M,N].
    const int m_col = N;
    const int n_col = M;
    const int k_col = K;
    const int lda = N;
    const int ldb = K;
    const int ldc = N;
    const int32_t alpha = 1;
    const int32_t beta = 0;

    auto launch = [&]() {
        HBLAS_CHECK(hipblasGemmEx(handle,
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    m_col, n_col, k_col,
                    &alpha,
                    dB, HIP_R_8I, lda,
                    dA, HIP_R_8I, ldb,
                    &beta,
                    dC, HIP_R_32I, ldc,
                    HIPBLAS_COMPUTE_32I,
                    HIPBLAS_GEMM_DEFAULT));
    };

    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr, stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < reps; ++i) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    const double avg_ms = (double) elapsed_ms / reps;
    const double tmac_s = ((double) M * N * K) / (avg_ms * 1.0e9);
    std::printf("[hipblas-i8] avg_ms=%.6f tmac_s=%.3f\n", avg_ms, tmac_s);

    HBLAS_CHECK(hipblasDestroy(handle));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
