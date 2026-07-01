#include "hipblaslt-i8.cuh"

#include "mmid.cuh"
#include "q8-1-act-cache.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

namespace {

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)
struct ggml_cuda_hipblaslt_i8_tls_handle {
    hipblasLtHandle_t handle = nullptr;
    int device = -1;

    ~ggml_cuda_hipblaslt_i8_tls_handle() {
        if (handle != nullptr) {
            (void) hipblasLtDestroy(handle);
        }
    }
};

static hipblasStatus_t ggml_cuda_hipblaslt_i8_get_tls_handle(hipblasLtHandle_t * out) {
    if (out == nullptr) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int device = 0;
    const cudaError_t device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    static thread_local ggml_cuda_hipblaslt_i8_tls_handle tls;
    if (tls.handle != nullptr && tls.device != device) {
        (void) hipblasLtDestroy(tls.handle);
        tls.handle = nullptr;
        tls.device = -1;
    }
    if (tls.handle == nullptr) {
        const hipblasStatus_t status = hipblasLtCreate(&tls.handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            return status;
        }
        tls.device = device;
    }

    *out = tls.handle;
    return HIPBLAS_STATUS_SUCCESS;
}
#endif

static bool ggml_cuda_hipblaslt_i8_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0;
}

static int ggml_cuda_hipblaslt_i8_env_int(const char * name, const int fallback) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env) {
        return fallback;
    }
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return (int) value;
}

static bool ggml_cuda_hipblaslt_i8_diag_enabled() {
    return ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_DIAG") ||
           ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_PREFILL_DIAG") ||
           ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_PREFILL_ALGO_LOG");
}

static bool ggml_cuda_hipblaslt_i8_prefill_env_enabled(const char * typed_name, const char * legacy_name) {
    return ggml_cuda_hipblaslt_i8_env_enabled(typed_name) || ggml_cuda_hipblaslt_i8_env_enabled(legacy_name);
}

struct ggml_cuda_hipblaslt_i8_shape_algo {
    int m;
    int n;
    int k;
    int algo_ord;
};

struct ggml_cuda_hipblaslt_i8_grouped_shape_algo {
    int groups;
    int m;
    int n;
    int k;
    int algo_ord;
};

static int ggml_cuda_hipblaslt_i8_lookup_shape_algo_ord(const int m, const int n, const int k) {
    static constexpr ggml_cuda_hipblaslt_i8_shape_algo table[] = {
        // Q8_0 full-K dense-I8 tuner, P0/P1 shapes, 20260629.
        {128, 17408, 5120, 2},
        {256, 17408, 5120, 0},
        {512, 17408, 5120, 2},
        {1024, 17408, 5120, 2},
        {2048, 17408, 5120, 2},
        {128, 5120, 17408, 3},
        {256, 5120, 17408, 2},
        {512, 5120, 17408, 2},
        {1024, 5120, 17408, 2},
        {2048, 5120, 17408, 2},
        {128, 12288, 5120, 2},
        {256, 12288, 5120, 2},
        {512, 12288, 5120, 2},
        {1024, 12288, 5120, 0},
        {128, 10240, 5120, 2},
        {256, 10240, 5120, 2},
        {512, 10240, 5120, 2},
        {1024, 10240, 5120, 0},
        {128, 6144, 5120, 0},
        {256, 6144, 5120, 0},
        {512, 6144, 5120, 2},
        {1024, 6144, 5120, 2},
        {128, 5120, 6144, 3},
        {256, 5120, 6144, 2},
        {512, 5120, 6144, 0},
        {1024, 5120, 6144, 0},
        {128, 8192, 2048, 2},
        {256, 8192, 2048, 2},
        {512, 8192, 2048, 2},
        {1024, 8192, 2048, 0},
        {2048, 8192, 2048, 0},
        {128, 10240, 2048, 2},
        {256, 10240, 2048, 2},
        {512, 10240, 2048, 2},
        {1024, 10240, 2048, 2},
        {2048, 10240, 2048, 2},
        {128, 6144, 2048, 2},
        {256, 6144, 2048, 2},
        {512, 6144, 2048, 2},
        {1024, 6144, 2048, 0},
        {128, 2048, 6144, 4},
        {256, 2048, 6144, 1},
        {512, 2048, 6144, 0},
        {1024, 2048, 6144, 1},
        {128, 2048, 4096, 3},
        {256, 2048, 4096, 1},
        {512, 2048, 4096, 3},
        {1024, 2048, 4096, 1},
    };
    for (const ggml_cuda_hipblaslt_i8_shape_algo & entry : table) {
        if (entry.m == m && entry.n == n && entry.k == k) {
            return entry.algo_ord;
        }
    }
    return -1;
}

static int ggml_cuda_hipblaslt_i8_lookup_q6_K_shape_algo_ord(const int m, const int n, const int k) {
    static constexpr ggml_cuda_hipblaslt_i8_shape_algo table[] = {
        {128, 1024, 5120, 4}, // Q6_K split-K16 inner bench 20260629: M=128,N=1024,K16 best algo 4
        { 76, 1024, 5120, 5}, // Q6_K split-K16 inner bench 20260629: M=76,N=1024,K16 best algo 5
    };
    for (const ggml_cuda_hipblaslt_i8_shape_algo & entry : table) {
        if (entry.m == m && entry.n == n && entry.k == k) {
            return entry.algo_ord;
        }
    }
    return -1;
}

static int ggml_cuda_hipblaslt_i8_lookup_q4q5_k32_shape_algo_ord(const int m, const int n) {
    static constexpr ggml_cuda_hipblaslt_i8_shape_algo table[] = {
        // Q4_0/Q5_0 exact K32 core tuner, 20260629. k field is the hipBLASLt core K.
        {128,  512, 32, 5}, {512,  512, 32, 5}, {1024,  512, 32, 1},
        {128, 1024, 32, 4}, {512, 1024, 32, 3}, {1024, 1024, 32, 2},
        {128, 2048, 32, 3}, {512, 2048, 32, 0}, {1024, 2048, 32, 3},
        {128, 5120, 32, 2}, {512, 5120, 32, 2}, {1024, 5120, 32, 2},
        {128, 8192, 32, 1}, {512, 8192, 32, 2}, {1024, 8192, 32, 2},
        {128,10240, 32, 2}, {512,10240, 32, 2}, {1024,10240, 32, 2},
        {128,17408, 32, 0}, {512,17408, 32, 0}, {1024,17408, 32, 2},
    };
    for (const ggml_cuda_hipblaslt_i8_shape_algo & entry : table) {
        if (entry.m == m && entry.n == n) {
            return entry.algo_ord;
        }
    }
    return -1;
}

static int ggml_cuda_hipblaslt_i8_lookup_dense_gate_up_grouped_algo_ord(
        const int out_cols, const int rows, const int core_k) {
    static constexpr ggml_cuda_hipblaslt_i8_grouped_shape_algo table[] = {
        // Dense FFN gate+up groups=2 all-algo sweep, 20260630.
        // Shape key is the grouped hipBLASLt core [M,N,groups,K] = [out_cols, rows, 2, K32/K16].
        {2, 12288,  128, 32, 2},
        {2, 12288,  256, 32, 2},
        {2, 12288,  512, 32, 1},
        {2, 12288, 1024, 32, 1},
        {2, 17408,  128, 32, 1},
        {2, 17408,  256, 32, 2},
        {2, 17408,  512, 32, 0},
        {2, 17408, 1024, 32, 2},
        {2, 17408, 2048, 32, 1},
        {2, 12288,  128, 16, 2},
        {2, 12288,  256, 16, 2},
        {2, 12288,  512, 16, 1},
        {2, 12288, 1024, 16, 0},
        {2, 17408,  128, 16, 1},
        {2, 17408,  256, 16, 1},
        {2, 17408,  512, 16, 1},
        {2, 17408, 1024, 16, 2},
        {2, 17408, 2048, 16, 0},
    };
    for (const ggml_cuda_hipblaslt_i8_grouped_shape_algo & entry : table) {
        if (entry.groups == 2 && entry.m == out_cols && entry.n == rows && entry.k == core_k) {
            return entry.algo_ord;
        }
    }
    return -1;
}

static int ggml_cuda_hipblaslt_i8_lookup_q6_K_moe_grouped_algo_ord(
        const int groups, const int m, const int n, const int k) {
    static constexpr ggml_cuda_hipblaslt_i8_grouped_shape_algo table[] = {
        // Grouped-I8 v8 exact multi-solution library, 20260629.
        // Shape key is the grouped hipBLASLt core [M, N, groups, K] = [output_cols, max_rows_per_expert, active_experts, K16].
        {128, 768, 65, 16, 2},
    };
    for (const ggml_cuda_hipblaslt_i8_grouped_shape_algo & entry : table) {
        if (entry.groups == groups && entry.m == m && entry.n == n && entry.k == k) {
            return entry.algo_ord;
        }
    }
    return -1;
}

static bool ggml_cuda_hipblaslt_i8_mul_overflows_size(const size_t a, const size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return true;
    }
    out = a * b;
    return false;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_validate_plan_only(
        const ggml_cuda_hipblaslt_i8_plan & plan) {
    if (plan.m <= 0 || plan.n <= 0 || plan.k <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.stride_a_blocks < plan.k / QK8_1 || plan.stride_b_blocks < plan.k / QK8_0 || plan.stride_d < plan.n) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    const int64_t mn = (int64_t) plan.m * (int64_t) plan.n;
    const int64_t mk = (int64_t) plan.m * (int64_t) plan.k;
    const int64_t nk = (int64_t) plan.n * (int64_t) plan.k;
    if (mn > std::numeric_limits<int>::max() || mk > std::numeric_limits<int>::max() || nk > std::numeric_limits<int>::max()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    if (plan.batch_count != 1) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_IMPLEMENTED;
    }
    if (plan.heuristic_request_count <= 0 || plan.heuristic_request_count > GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.preferred_algo_ord < -1 || plan.preferred_algo_ord >= plan.heuristic_request_count) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_validate_q6_K_plan_only(
        const ggml_cuda_hipblaslt_i8_plan & plan) {
    if (plan.m <= 0 || plan.n <= 0 || plan.k <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.k % QK_K != 0 || plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K || plan.full_k_stage_k32_exact) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.stride_a_blocks < plan.k / QK8_1 || plan.stride_b_blocks < plan.k / QK_K || plan.stride_d < plan.n) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    const int64_t mn = (int64_t) plan.m * (int64_t) plan.n;
    const int64_t mk = (int64_t) plan.m * (int64_t) plan.k;
    const int64_t nk = (int64_t) plan.n * (int64_t) plan.k;
    if (mn > std::numeric_limits<int>::max() || mk > std::numeric_limits<int>::max() || nk > std::numeric_limits<int>::max()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    if (plan.batch_count != 1) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_IMPLEMENTED;
    }
    if (plan.heuristic_request_count <= 0 || plan.heuristic_request_count > GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.preferred_algo_ord < -1 || plan.preferred_algo_ord >= plan.heuristic_request_count) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)

static const char * ggml_cuda_hipblaslt_status_name(const hipblasStatus_t status) {
    switch (status) {
        case HIPBLAS_STATUS_SUCCESS:         return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED:    return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE:   return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_MAPPING_ERROR:   return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED:return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR:  return "HIPBLAS_STATUS_INTERNAL_ERROR";
        case HIPBLAS_STATUS_NOT_SUPPORTED:   return "HIPBLAS_STATUS_NOT_SUPPORTED";
        default:                             return "unknown hipBLASLt error";
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_quantize_q8_1_ids(
        const float * __restrict__ x,
        const int32_t * __restrict__ ids,
        block_q8_1 * __restrict__ y,
        float * __restrict__ sums_k16,
        int k,
        int stride_x,
        int nrows) {
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    const int row = (int) blockIdx.y;
    if (i0 >= k || row >= nrows) {
        return;
    }

    const int src_row = ids != nullptr ? ids[row] : row;
    const int64_t ib  = (int64_t) row * (k / QK8_1) + i0 / QK8_1;
    const int iqs = (int) (i0 % QK8_1);

    const float xi = x[(int64_t) src_row * stride_x + i0];
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    float sum16 = 0.0f;
    if (sums_k16 != nullptr) {
        sum16 = warp_reduce_sum<QK8_1/2>(xi);
    }

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;

    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
    if (sums_k16 != nullptr && (iqs & 15) == 0) {
        sums_k16[(int64_t) row * (k / 16) + i0 / 16] = sum16;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_a_q8_1(
        const block_q8_1 * __restrict__ a,
        int8_t * __restrict__ a_i8,
        int k_block,
        int m,
        int stride_a_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * QK8_1;
    if (i >= total) {
        return;
    }

    const int row = i / QK8_1;
    const int kk  = i - row * QK8_1;
    a_i8[row * QK8_1 + kk] = a[row * stride_a_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_0(
        const block_q8_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = QK8_0 * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    b_i8[kk * n + col] = b[col * stride_b_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_0_grouped(
        const block_q8_0 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = QK8_0 * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_q8_0 * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = bx[col * stride_b_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_ab_q8_1_q8_0_grouped(
        const block_q8_1 * __restrict__ a,
        const block_q8_0 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ a_i8,
        int8_t * __restrict__ b_i8,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total_a = m * QK8_1;
    const int group_size = QK8_0 * n;
    const int total_b = group_count * group_size;

    if (i < total_a) {
        const int row = i / QK8_1;
        const int kk = i - row * QK8_1;
        a_i8[row * QK8_1 + kk] = a[(size_t) row * stride_a_blocks + k_block].qs[kk];
    }

    if (i < total_b) {
        const int group = i / group_size;
        const int rem = i - group * group_size;
        const int kk = rem / n;
        const int col = rem - kk * n;
        const int expert = active_experts[group];
        const block_q8_0 * bx = b + (size_t) expert * stride_expert_blocks;
        b_i8[(size_t) group * group_size + (size_t) kk * n + col] = bx[(size_t) col * stride_b_blocks + k_block].qs[kk];
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_0_grouped_tiled(
        const block_q8_0 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int group = (int) blockIdx.y;
    const int col_base = (int) blockIdx.x * 8;
    const int load_kk = (int) threadIdx.x;
    const int load_col_lane = (int) threadIdx.y;
    if (group >= group_count || load_kk >= QK8_0 || load_col_lane >= 8) {
        return;
    }

    __shared__ int8_t tile[8][QK8_0];
    const int col = col_base + load_col_lane;
    int8_t v = 0;
    if (col < n) {
        const int expert = active_experts[group];
        const block_q8_0 * bx = b + (size_t) expert * stride_expert_blocks;
        v = bx[(size_t) col * stride_b_blocks + k_block].qs[load_kk];
    }
    tile[load_col_lane][load_kk] = v;
    __syncthreads();

    const int tid = (int) threadIdx.y * QK8_0 + (int) threadIdx.x;
    const int store_kk = tid / 8;
    const int store_col_lane = tid - store_kk * 8;
    const int store_col = col_base + store_col_lane;
    if (store_kk < QK8_0 && store_col < n) {
        const size_t group_size = (size_t) QK8_0 * n;
        b_i8[(size_t) group * group_size + (size_t) store_kk * n + store_col] = tile[store_col_lane][store_kk];
    }
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q4_0_value(
        const block_q4_0 & b,
        const int kk) {
    const uint8_t byte = b.qs[kk & 15];
    const int q = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    return (int8_t) (q - 8);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q4_1_value(
        const block_q4_1 & b,
        const int kk) {
    const uint8_t byte = b.qs[kk & 15];
    const int q = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    return (int8_t) q;
}

static __device__ __forceinline__ uint32_t ggml_cuda_hipblaslt_i8_q5_0_qh(
        const block_q5_0 & b) {
    return ((uint32_t) b.qh[0]) |
           ((uint32_t) b.qh[1] << 8) |
           ((uint32_t) b.qh[2] << 16) |
           ((uint32_t) b.qh[3] << 24);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q5_0_value(
        const block_q5_0 & b,
        const int kk) {
    const uint8_t byte = b.qs[kk & 15];
    const int ql = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    const int qh = (ggml_cuda_hipblaslt_i8_q5_0_qh(b) >> kk) & 1;
    return (int8_t) ((ql | (qh << 4)) - 16);
}

static __device__ __forceinline__ uint32_t ggml_cuda_hipblaslt_i8_q5_1_qh(
        const block_q5_1 & b) {
    return ((uint32_t) b.qh[0]) |
           ((uint32_t) b.qh[1] << 8) |
           ((uint32_t) b.qh[2] << 16) |
           ((uint32_t) b.qh[3] << 24);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q5_1_value(
        const block_q5_1 & b,
        const int kk) {
    const uint8_t byte = b.qs[kk & 15];
    const int ql = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    const int qh = (ggml_cuda_hipblaslt_i8_q5_1_qh(b) >> kk) & 1;
    return (int8_t) (ql | (qh << 4));
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_0(
        const block_q4_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_q4_0_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_1(
        const block_q4_1 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_q4_1_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_0(
        const block_q5_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_q5_0_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_1(
        const block_q5_1 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_q5_1_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_0_grouped(
        const block_q4_0 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_q4_0 * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_0_value(bx[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_1_grouped(
        const block_q4_1 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_q4_1 * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_1_value(bx[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_0_grouped(
        const block_q5_0 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_q5_0 * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_0_value(bx[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_1_grouped(
        const block_q5_1 * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_q5_1 * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_1_value(bx[col * stride_b_blocks + k_block], kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q1_0_value(
        const block_q1_0 & b,
        const int ib32,
        const int kk) {
    const uint8_t bits = b.qs[4*ib32 + (kk >> 3)];
    return (bits & (1u << (kk & 7))) ? (int8_t) 1 : (int8_t) -1;
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q1_0_dense_pair(
        const block_q1_0 * __restrict__ up,
        const block_q1_0 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k128 = k_block / (QK1_0 / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k128 * (QK1_0 / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_q1_0 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q1_0_value(weights[col * stride_b_blocks + k128], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_0_dense_pair(
        const block_q8_0 * __restrict__ up,
        const block_q8_0 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_q8_0 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = weights[col * stride_b_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_0_dense_pair(
        const block_q4_0 * __restrict__ up,
        const block_q4_0 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_q4_0 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_0_value(weights[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_1_dense_pair(
        const block_q4_1 * __restrict__ up,
        const block_q4_1 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_q4_1 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_1_value(weights[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_0_dense_pair(
        const block_q5_0 * __restrict__ up,
        const block_q5_0 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_q5_0 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_0_value(weights[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_1_dense_pair(
        const block_q5_1 * __restrict__ up,
        const block_q5_1 * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_q5_1 * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_1_value(weights[col * stride_b_blocks + k_block], kk);
}

static __device__ __forceinline__ void ggml_cuda_hipblaslt_i8_get_scale_min_k4(
        const int j,
        const uint8_t * __restrict__ q,
        uint8_t & d,
        uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q4_K_value(
        const block_q4_K & b,
        const int ib32,
        const int kk) {
    const int pair = ib32 >> 1;
    const uint8_t byte = b.qs[32 * pair + kk];
    const int q = (ib32 & 1) ? (byte >> 4) : (byte & 0x0f);
    return (int8_t) q;
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q5_K_value(
        const block_q5_K & b,
        const int ib32,
        const int kk) {
    const int pair = ib32 >> 1;
    const uint8_t byte = b.qs[32 * pair + kk];
    const int ql = (ib32 & 1) ? (byte >> 4) : (byte & 0x0f);
    const int qh = (b.qh[kk] >> ib32) & 1;
    return (int8_t) (ql | (qh << 4));
}

static __device__ __forceinline__ int ggml_cuda_hipblaslt_i8_q2q3_K_qidx(
        const int k_in_block,
        int & shift,
        uint8_t & hmask_bit) {
    const int half128 = k_in_block >> 7;
    const int within128 = k_in_block & 127;
    const int pair16 = within128 >> 4;
    shift = (pair16 >> 1) << 1;
    hmask_bit = (uint8_t) (1u << (k_in_block >> 5));
    return half128 * 32 + ((pair16 & 1) << 4) + (within128 & 15);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q2_K_value(
        const block_q2_K & b,
        const int k_in_block) {
    int shift = 0;
    uint8_t hmask_bit = 0;
    const int qidx = ggml_cuda_hipblaslt_i8_q2q3_K_qidx(k_in_block, shift, hmask_bit);
    GGML_UNUSED(hmask_bit);
    return (int8_t) ((b.qs[qidx] >> shift) & 0x03);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q3_K_value(
        const block_q3_K & b,
        const int k_in_block) {
    int shift = 0;
    uint8_t hmask_bit = 0;
    const int qidx = ggml_cuda_hipblaslt_i8_q2q3_K_qidx(k_in_block, shift, hmask_bit);
    const int within128 = k_in_block & 127;
    const int pair16 = within128 >> 4;
    const int hidx = ((pair16 & 1) << 4) + (within128 & 15);
    const int q = (b.qs[qidx] >> shift) & 0x03;
    return (int8_t) (q - ((b.hmask[hidx] & hmask_bit) ? 0 : 4));
}

static __device__ __forceinline__ int ggml_cuda_hipblaslt_i8_q3_K_scale(
        const block_q3_K & b,
        const int scale_idx) {
    const int isc_low = scale_idx & 7;
    const int sc_shift_low = 4 * (scale_idx >> 3);
    const int sc_low = (b.scales[isc_low] >> sc_shift_low) & 0x0f;
    const int isc_high = scale_idx & 3;
    const int sc_shift_high = 2 * (scale_idx >> 2);
    const int sc_high = ((b.scales[8 + isc_high] >> sc_shift_high) & 0x03) << 4;
    return (sc_low | sc_high) - 32;
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_K_grouped(
        const block_q4_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_q4_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_K_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_K_grouped(
        const block_q5_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_q5_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_K_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q4_K_dense_pair(
        const block_q4_K * __restrict__ up,
        const block_q4_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_q4_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q4_K_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q5_K_dense_pair(
        const block_q5_K * __restrict__ up,
        const block_q5_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_q5_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q5_K_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q2_K_grouped(
        const block_q2_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256 = k_full / QK_K;
    const int kin = k_full - k256 * QK_K;
    const block_q2_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q2_K_value(bx[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q2_K_dense_pair(
        const block_q2_K * __restrict__ up,
        const block_q2_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256 = k_full / QK_K;
    const int kin = k_full - k256 * QK_K;
    const block_q2_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q2_K_value(weights[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q3_K_grouped(
        const block_q3_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256 = k_full / QK_K;
    const int kin = k_full - k256 * QK_K;
    const block_q3_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q3_K_value(bx[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q3_K_dense_pair(
        const block_q3_K * __restrict__ up,
        const block_q3_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256 = k_full / QK_K;
    const int kin = k_full - k256 * QK_K;
    const block_q3_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q3_K_value(weights[col * stride_b_blocks + k256], kin);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_q6_K_get_i8(
        const block_q6_K & block,
        const int kk) {
    const int ip     = kk >> 7;      // 0 for [0,127], 1 for [128,255]
    const int within = kk & 127;
    const int seg    = within >> 5;  // four 32-value groups per 128-value half
    const int il     = within & 31;

    const uint8_t ql_byte = block.ql[64 * ip + il + ((seg & 1) ? 32 : 0)];
    const uint8_t qh_byte = block.qh[32 * ip + il];
    const int low4  = seg >= 2 ? (ql_byte >> 4) : (ql_byte & 0x0f);
    const int high2 = (qh_byte >> (2 * seg)) & 0x03;
    return (int8_t) ((low4 | (high2 << 4)) - 32);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q6_K(
        const block_q6_K * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_q6_K_get_i8(b[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q6_K_grouped(
        const block_q6_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_q6_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q6_K_get_i8(bx[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q6_K_dense_pair(
        const block_q6_K * __restrict__ up,
        const block_q6_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_q6_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_q6_K_get_i8(weights[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_K_grouped(
        const block_q8_K * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_q8_K * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = bx[col * stride_b_blocks + k256].qs[kin];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_q8_K_dense_pair(
        const block_q8_K * __restrict__ up,
        const block_q8_K * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_q8_K * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = weights[col * stride_b_blocks + k256].qs[kin];
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq3_s_grid_value(
        const block_iq3_s & b,
        const int k_in_block) {
    const int ib32 = k_in_block / 32;
    const int in32 = k_in_block - ib32 * 32;
    const int pair = ib32 / 2;
    const int half = ib32 - pair * 2;
    const int l    = in32 / 8;
    const int lane = in32 - l * 8;

    const int qs_base    = pair * 16 + half * 8;
    const int signs_base = pair * 8  + half * 4;
    const int qh         = b.qh[pair * 2 + half];
    const bool high_half = lane >= 4;
    const int qsi        = qs_base + 2*l + (high_half ? 1 : 0);
    const int grid_idx   = b.qs[qsi] | ((qh << ((high_half ? 7 : 8) - 2*l)) & 0x100);
    const uint32_t grid  = iq3s_grid[grid_idx];
    const int lane4      = high_half ? lane - 4 : lane;
    int value            = (grid >> (8 * lane4)) & 0xff;
    if (b.signs[signs_base + l] & kmask_iq2xs[lane]) {
        value = -value;
    }
    return (int8_t) value;
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq3_s(
        const block_iq3_s * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    if (i >= total) {
        return;
    }

    const int kk  = i / n;
    const int col = i - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    b_i8[kk * n + col] = ggml_cuda_hipblaslt_i8_iq3_s_grid_value(b[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq3_s_grouped(
        const block_iq3_s * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_iq3_s * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq3_s_grid_value(bx[col * stride_b_blocks + k256], kin);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq3_s_dense_pair(
        const block_iq3_s * __restrict__ up,
        const block_iq3_s * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + kk;
    const int k256   = k_full / QK_K;
    const int kin    = k_full - k256 * QK_K;
    const block_iq3_s * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq3_s_grid_value(weights[col * stride_b_blocks + k256], kin);
}

static __device__ __forceinline__ int ggml_cuda_hipblaslt_i8_iq4_xs_scale(
        const block_iq4_xs & b,
        const int ib32) {
    return ((b.scales_l[ib32/2] >> (4*(ib32%2))) & 0x0f) | (((b.scales_h >> (2*ib32)) & 0x03) << 4);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq4_xs_value(
        const block_iq4_xs & b,
        const int ib32,
        const int kk) {
    const uint8_t byte = b.qs[ib32 * 16 + (kk & 15)];
    const int q = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    return kvalues_iq4nl[q];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq4_xs_grouped(
        const block_iq4_xs * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq4_xs * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq4_xs_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq4_xs_dense_pair(
        const block_iq4_xs * __restrict__ up,
        const block_iq4_xs * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq4_xs * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq4_xs_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq4_nl_value(
        const block_iq4_nl & b,
        const int kk) {
    const uint8_t byte = b.qs[kk & 15];
    const int q = kk < 16 ? (byte & 0x0f) : (byte >> 4);
    return kvalues_iq4nl[q];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq4_nl_grouped(
        const block_iq4_nl * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const block_iq4_nl * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq4_nl_value(bx[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq4_nl_dense_pair(
        const block_iq4_nl * __restrict__ up,
        const block_iq4_nl * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const block_iq4_nl * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq4_nl_value(weights[col * stride_b_blocks + k_block], kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq3_xxs_value(
        const block_iq3_xxs & b,
        const int ib32,
        const int kk) {
    const int in32 = kk & 31;
    const int l    = in32 >> 3;
    const int lane = in32 & 7;
    const uint32_t aux32 = ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 0]) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 1] <<  8) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 2] << 16) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 3] << 24);
    const uint32_t grid = lane < 4 ? iq3xxs_grid[b.qs[8*ib32 + 2*l + 0]] : iq3xxs_grid[b.qs[8*ib32 + 2*l + 1]];
    int value = (grid >> (8 * (lane & 3))) & 0xff;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
    if (signs & kmask_iq2xs[lane]) {
        value = -value;
    }
    return (int8_t) value;
}

static __device__ __forceinline__ int ggml_cuda_hipblaslt_i8_iq3_xxs_scale(
        const block_iq3_xxs & b,
        const int ib32) {
    const uint32_t aux32 = ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 0]) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 1] <<  8) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 2] << 16) |
                           ((uint32_t) b.qs[QK_K/4 + 4*ib32 + 3] << 24);
    return (int) (aux32 >> 28);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq3_xxs_grouped(
        const block_iq3_xxs * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq3_xxs * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq3_xxs_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq3_xxs_dense_pair(
        const block_iq3_xxs * __restrict__ up,
        const block_iq3_xxs * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq3_xxs * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq3_xxs_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq2_xxs_value(
        const block_iq2_xxs & b,
        const int ib32,
        const int kk) {
    const int in32 = kk & 31;
    const int l    = in32 >> 3;
    const int lane = in32 & 7;
    const uint32_t aux0 = ((uint32_t) b.qs[4*ib32 + 0]) | ((uint32_t) b.qs[4*ib32 + 1] << 16);
    const uint32_t aux1 = ((uint32_t) b.qs[4*ib32 + 2]) | ((uint32_t) b.qs[4*ib32 + 3] << 16);
    const uint8_t grid_idx = (aux0 >> (8*l)) & 0xff;
    const uint64_t grid = iq2xxs_grid[grid_idx];
    int value = (grid >> (8 * lane)) & 0xff;
    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7*l)) & 127];
    if (signs & kmask_iq2xs[lane]) {
        value = -value;
    }
    return (int8_t) value;
}

static __device__ __forceinline__ int ggml_cuda_hipblaslt_i8_iq2_xxs_scale(
        const block_iq2_xxs & b,
        const int ib32) {
    const uint32_t aux1 = ((uint32_t) b.qs[4*ib32 + 2]) | ((uint32_t) b.qs[4*ib32 + 3] << 16);
    return (int) (aux1 >> 28);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_xxs_grouped(
        const block_iq2_xxs * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_xxs * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_xxs_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_xxs_dense_pair(
        const block_iq2_xxs * __restrict__ up,
        const block_iq2_xxs * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_xxs * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_xxs_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq1_s_value(
        const block_iq1_s & b,
        const int ib32,
        const int kk) {
    const int in32 = kk & 31;
    const int l    = in32 >> 3;
    const int lane = in32 & 7;
    const uint16_t qh = b.qh[ib32];
    const uint16_t grid_idx = b.qs[4*ib32 + l] | (((qh >> (3*l)) & 7) << 8);
    const uint32_t grid = iq1s_grid_gpu[grid_idx];
    return lane < 4 ? (int8_t) ((grid >> (8*lane)) & 0x0f) : (int8_t) ((grid >> (8*(lane - 4) + 4)) & 0x0f);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq1_s_grouped(
        const block_iq1_s * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq1_s * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq1_s_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq1_s_dense_pair(
        const block_iq1_s * __restrict__ up,
        const block_iq1_s * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq1_s * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq1_s_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq2_xs_value(
        const block_iq2_xs & b,
        const int ib32,
        const int kk) {
    const int in32 = kk & 31;
    const int l    = in32 >> 3;
    const int lane = in32 & 7;
    const uint16_t q2 = b.qs[4*ib32 + l];
    const uint64_t grid = iq2xs_grid[q2 & 0x1ff];
    int value = (grid >> (8 * lane)) & 0xff;
    const uint8_t signs = ksigns_iq2xs[q2 >> 9];
    if (signs & kmask_iq2xs[lane]) {
        value = -value;
    }
    return (int8_t) value;
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_xs_grouped(
        const block_iq2_xs * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_xs * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_xs_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_xs_dense_pair(
        const block_iq2_xs * __restrict__ up,
        const block_iq2_xs * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_xs * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_xs_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __device__ __forceinline__ int8_t ggml_cuda_hipblaslt_i8_iq2_s_value(
        const block_iq2_s & b,
        const int ib32,
        const int kk) {
    const int in32 = kk & 31;
    const int l    = in32 >> 3;
    const int lane = in32 & 7;
    const uint8_t qh = b.qh[ib32];
    const uint16_t grid_idx = b.qs[4*ib32 + l] | ((qh << (8 - 2*l)) & 0x300);
    const uint64_t grid = iq2s_grid[grid_idx];
    int value = (grid >> (8 * lane)) & 0xff;
    const uint8_t signs = b.qs[QK_K/8 + 4*ib32 + l];
    if (signs & kmask_iq2xs[lane]) {
        value = -value;
    }
    return (int8_t) value;
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_s_grouped(
        const block_iq2_s * __restrict__ b,
        const int32_t * __restrict__ active_experts,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks,
        int stride_expert_blocks,
        int group_count) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = group_count * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int expert = active_experts[group];
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_s * bx = b + (size_t) expert * stride_expert_blocks;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_s_value(bx[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_b_iq2_s_dense_pair(
        const block_iq2_s * __restrict__ up,
        const block_iq2_s * __restrict__ gate,
        int8_t * __restrict__ b_i8,
        int k_block,
        int n,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int group_size = GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n;
    const int total = 2 * group_size;
    if (i >= total) {
        return;
    }

    const int group = i / group_size;
    const int rem = i - group * group_size;
    const int kk  = rem / n;
    const int col = rem - kk * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const block_iq2_s * weights = group == 0 ? up : gate;
    b_i8[(size_t) group * group_size + kk * n + col] = ggml_cuda_hipblaslt_i8_iq2_s_value(weights[col * stride_b_blocks + k256], ib32, kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_a_q8_1(
        const block_q8_1 * __restrict__ a,
        int8_t * __restrict__ a_i8,
        int m,
        int k,
        int stride_a_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * k;
    if (i >= total) {
        return;
    }

    const int row = i / k;
    const int kk_full = i - row * k;
    const int k_block = kk_full / QK8_1;
    const int kk = kk_full - k_block * QK8_1;
    a_i8[row * k + kk_full] = a[row * stride_a_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_b_q8_0(
        const block_q8_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int n,
        int k,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = k * n;
    if (i >= total) {
        return;
    }

    const int kk_full = i / n;
    const int col = i - kk_full * n;
    const int k_block = kk_full / QK8_0;
    const int kk = kk_full - k_block * QK8_0;
    b_i8[kk_full * n + col] = b[col * stride_b_blocks + k_block].qs[kk];
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_b_q4_0(
        const block_q4_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int n,
        int k,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = k * n;
    if (i >= total) {
        return;
    }

    const int kk_full = i / n;
    const int col = i - kk_full * n;
    const int k_block = kk_full / QK4_0;
    const int kk = kk_full - k_block * QK4_0;
    b_i8[kk_full * n + col] = ggml_cuda_hipblaslt_i8_q4_0_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_b_q4_1(
        const block_q4_1 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int n,
        int k,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = k * n;
    if (i >= total) {
        return;
    }

    const int kk_full = i / n;
    const int col = i - kk_full * n;
    const int k_block = kk_full / QK4_1;
    const int kk = kk_full - k_block * QK4_1;
    b_i8[kk_full * n + col] = ggml_cuda_hipblaslt_i8_q4_1_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_b_q5_0(
        const block_q5_0 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int n,
        int k,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = k * n;
    if (i >= total) {
        return;
    }

    const int kk_full = i / n;
    const int col = i - kk_full * n;
    const int k_block = kk_full / QK5_0;
    const int kk = kk_full - k_block * QK5_0;
    b_i8[kk_full * n + col] = ggml_cuda_hipblaslt_i8_q5_0_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_stage_full_b_q5_1(
        const block_q5_1 * __restrict__ b,
        int8_t * __restrict__ b_i8,
        int n,
        int k,
        int stride_b_blocks) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = k * n;
    if (i >= total) {
        return;
    }

    const int kk_full = i / n;
    const int col = i - kk_full * n;
    const int k_block = kk_full / QK5_1;
    const int kk = kk_full - k_block * QK5_1;
    b_i8[kk_full * n + col] = ggml_cuda_hipblaslt_i8_q5_1_value(b[col * stride_b_blocks + k_block], kk);
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q8_0 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q8_0 & wb = b[col * stride_b_blocks + k_block];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q1_0(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q1_0 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k128 = k_block / (QK1_0 / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q1_0 & wb = b[col * stride_b_blocks + k128];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q8_0 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q8_0 & wb = b[col * stride_b_blocks + k_block];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0_ids_grouped(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q8_0 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ active_experts,
        const int32_t * __restrict__ expert_bounds,
        float * __restrict__ dst,
        int k_block,
        int max_rows,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_expert_blocks,
        int stride_d,
        int group_count) {
    const int group = (int) blockIdx.y;
    if (group >= group_count) {
        return;
    }

    const int expert = active_experts[group];
    const int row_low = expert_bounds[expert];
    const int row_high = expert_bounds[expert + 1];
    const int rows = row_high - row_low;
    if (rows <= 0) {
        return;
    }

    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * n || i >= max_rows * n) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int global_row = row_low + row;

    const block_q8_1 & ab = a[(size_t) global_row * stride_a_blocks + k_block];
    const block_q8_0 & wb = b[(size_t) expert * stride_expert_blocks + (size_t) col * stride_b_blocks + k_block];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[(size_t) global_row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[global_row] : global_row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_0_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_0 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_0 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_1_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_1 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_1 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm));
    const float wm = __half2float(__high2half(wb.dm));
    // Q4_1 is unsigned: w = d_w*q + m_w. Correct the I8 dot with the activation block sum.
    const float v = (float) c_i32[row * n + col] * ad * wd + wm * asum;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_0_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_0 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_0 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_1_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_1 * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_1 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm));
    const float wm = __half2float(__high2half(wb.dm));
    const float v = (float) c_i32[row * n + col] * ad * wd + wm * asum;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_K & wb = b[col * stride_b_blocks + k256];

    uint8_t sc = 0;
    uint8_t mn = 0;
    ggml_cuda_hipblaslt_i8_get_scale_min_k4(ib32, wb.scales, sc, mn);
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm)) * (float) sc;
    const float wm = __half2float(__high2half(wb.dm)) * (float) mn;
    // Q4_K dequantization subtracts the per-K32 min: w = d*sc*q - dmin*min.
    const float v = (float) c_i32[row * n + col] * ad * wd - wm * asum;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_K & wb = b[col * stride_b_blocks + k256];

    uint8_t sc = 0;
    uint8_t mn = 0;
    ggml_cuda_hipblaslt_i8_get_scale_min_k4(ib32, wb.scales, sc, mn);
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm)) * (float) sc;
    const float wm = __half2float(__high2half(wb.dm)) * (float) mn;
    // Q5_K dequantization subtracts the per-K32 min: w = d*sc*q - dmin*min.
    const float v = (float) c_i32[row * n + col] * ad * wd - wm * asum;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_K(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_K * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_K & wb = b[col * stride_b_blocks + k256];

    uint8_t sc = 0;
    uint8_t mn = 0;
    ggml_cuda_hipblaslt_i8_get_scale_min_k4(ib32, wb.scales, sc, mn);
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm)) * (float) sc;
    const float wm = __half2float(__high2half(wb.dm)) * (float) mn;
    const float v = (float) c_i32[row * n + col] * ad * wd - wm * asum;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_K(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_K * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_K & wb = b[col * stride_b_blocks + k256];

    uint8_t sc = 0;
    uint8_t mn = 0;
    ggml_cuda_hipblaslt_i8_get_scale_min_k4(ib32, wb.scales, sc, mn);
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm)) * (float) sc;
    const float wm = __half2float(__high2half(wb.dm)) * (float) mn;
    const float v = (float) c_i32[row * n + col] * ad * wd - wm * asum;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_dense_gate_up_swiglu(
        const float * __restrict__ up,
        const float * __restrict__ gate,
        float * __restrict__ dst,
        int m,
        int n,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }
    const int row = i / n;
    const int col = i - row * n;
    const float g = gate[i];
    const float silu = g / (1.0f + expf(-g));
    dst[row * stride_d + col] = up[i] * silu;
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_0(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_0 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_0 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_1(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q4_1 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q4_1 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm));
    const float wm = __half2float(__high2half(wb.dm));
    const float v = (float) c_i32[row * n + col] * ad * wd + wm * asum;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_0(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_0 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_0 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_1(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q5_1 * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q5_1 & wb = b[col * stride_b_blocks + k_block];
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(__low2half(wb.dm));
    const float wm = __half2float(__high2half(wb.dm));
    const float v = (float) c_i32[row * n + col] * ad * wd + wm * asum;

    float * out = dst + row * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q6_K(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q6_K * __restrict__ b,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int scale_idx = (k_full - k256 * QK_K) >> 4;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q6_K & wb = b[col * stride_b_blocks + k256];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) wb.scales[scale_idx];
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + row * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q6_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q6_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int scale_idx = (k_full - k256 * QK_K) >> 4;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q6_K & wb = b[col * stride_b_blocks + k256];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) wb.scales[scale_idx];
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q8_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
    const int k256 = k_full / QK_K;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q8_K & wb = b[col * stride_b_blocks + k256];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = wb.d;
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q2_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const float * __restrict__ a_k16_sums,
        const block_q2_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int scale_idx = (k_full - k256 * QK_K) >> 4;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q2_K & wb = b[col * stride_b_blocks + k256];
    const uint8_t sc = wb.scales[scale_idx];
    const float ad = __half2float(__low2half(ab.ds));
    const float asum = a_k16_sums[row * (stride_a_blocks * 2) + k_block * 2 + half16];
    const float wd = __half2float(__low2half(wb.dm)) * (float) (sc & 0x0f);
    const float wm = __half2float(__high2half(wb.dm)) * (float) (sc >> 4);
    const float v = (float) c_i32[row * n + col] * ad * wd - wm * asum;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q3_K_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_q3_K * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int scale_idx = (k_full - k256 * QK_K) >> 4;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_q3_K & wb = b[col * stride_b_blocks + k256];
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) ggml_cuda_hipblaslt_i8_q3_K_scale(wb, scale_idx);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_s_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq3_s * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = (k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K) / QK_K;
    const int ib32 = k_block & (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K - 1);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq3_s & wb = b[col * stride_b_blocks + k256];

    const int scale_byte = wb.scales[ib32 / 2];
    const int scale_nib  = (ib32 & 1) ? (scale_byte >> 4) : (scale_byte & 0x0f);
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) (1 + 2*scale_nib);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_xs_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq4_xs * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq4_xs & wb = b[col * stride_b_blocks + k256];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) (ggml_cuda_hipblaslt_i8_iq4_xs_scale(wb, ib32) - 32);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_nl_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq4_nl * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq4_nl & wb = b[col * stride_b_blocks + k_block];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d);
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_xxs_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq3_xxs * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq3_xxs & wb = b[col * stride_b_blocks + k256];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (0.5f + (float) ggml_cuda_hipblaslt_i8_iq3_xxs_scale(wb, ib32)) * 0.5f;
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xxs_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq2_xxs * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq2_xxs & wb = b[col * stride_b_blocks + k256];

    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (0.5f + (float) ggml_cuda_hipblaslt_i8_iq2_xxs_scale(wb, ib32)) * 0.25f;
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq1_s_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq1_s * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k256 = k_block / (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    const int ib32 = k_block - k256 * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K);

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq1_s & wb = b[col * stride_b_blocks + k256];
    const uint16_t qh = wb.qh[ib32];

    const float ad = __half2float(__low2half(ab.ds));
    const float asum = __half2float(__high2half(ab.ds));
    const float wd = __half2float(wb.d) * (float) (2*((qh >> 12) & 7) + 1);
    const float delta = -1.0f + ((qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA);
    const float v = (float) c_i32[row * n + col] * ad * wd + asum * wd * delta;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xs_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq2_xs * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int ib32 = (k_full - k256 * QK_K) >> 5;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq2_xs & wb = b[col * stride_b_blocks + k256];
    const int scale = (wb.scales[ib32] >> (4 * half16)) & 0x0f;
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (0.5f + (float) scale) * 0.25f;
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static __global__ void ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_s_ids(
        const int32_t * __restrict__ c_i32,
        const block_q8_1 * __restrict__ a,
        const block_iq2_s * __restrict__ b,
        const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst,
        int k_block,
        int half16,
        int m,
        int n,
        int stride_a_blocks,
        int stride_b_blocks,
        int stride_d) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = m * n;
    if (i >= total) {
        return;
    }

    const int row = i / n;
    const int col = i - row * n;
    const int k_full = k_block * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + half16 * 16;
    const int k256 = k_full / QK_K;
    const int ib32 = (k_full - k256 * QK_K) >> 5;

    const block_q8_1 & ab = a[row * stride_a_blocks + k_block];
    const block_iq2_s & wb = b[col * stride_b_blocks + k256];
    const int scale = (wb.scales[ib32] >> (4 * half16)) & 0x0f;
    const float ad = __half2float(__low2half(ab.ds));
    const float wd = __half2float(wb.d) * (0.5f + (float) scale) * 0.25f;
    const float v = (float) c_i32[row * n + col] * ad * wd;

    float * out = dst + (int64_t) (ids_dst != nullptr ? ids_dst[row] : row) * stride_d + col;
    if (k_block == 0 && half16 == 0) {
        *out = v;
    } else {
        *out += v;
    }
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args) {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a_desc = nullptr;
    hipblasLtMatrixLayout_t b_desc = nullptr;
    hipblasLtMatrixLayout_t c_desc = nullptr;
    hipblasLtMatrixLayout_t d_desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::array<hipblasLtMatmulHeuristicResult_t, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS> heuristics{};
    int returned = 0;
    int selected_algo_ord = 0;
    const int a_slice_ld = plan.full_k_stage_k32_exact ? plan.k : plan.chunk_k;
    ggml_cuda_hipblaslt_i8_status result = GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;

#define GGML_HIPBLASLT_I8_TRY(expr) do { \
        const hipblasStatus_t hblt_status = (expr); \
        if (hblt_status != HIPBLAS_STATUS_SUCCESS) { \
            GGML_LOG_WARN("%s: hipBLASLt call failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(hblt_status)); \
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR; \
            goto cleanup; \
        } \
    } while (0)

    GGML_HIPBLASLT_I8_TRY(hipblasLtCreate(&handle));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));

    // Same column-major transposed view as scripts/hip/hipblaslt-i8-gemm-bench.hip:
    // C_col(N,M) = B_col(N,32) * A_col(32,M), equivalent to row-major C[M,N] = A[M,32] * B[32,N].
    // The full-K staging variant keeps exact per-K32 scale accumulation but stages A[M,K]/B[K,N] once.
    // For A_col(32,M) slices inside A[M,K], the leading dimension is K instead of 32.
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_8I,  plan.n, plan.chunk_k, plan.n));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_8I,  plan.chunk_k, plan.m, a_slice_ld));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_32I, plan.n, plan.m, plan.n));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32I, plan.n, plan.m, plan.n));

    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceCreate(&pref));
    {
        uint64_t workspace_bytes = args.scratch.hipblaslt_workspace_bytes;
        GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceSetAttribute(
                    pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
    }

    {
        const int requested = std::max(1, std::min(plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        const hipblasStatus_t h = hipblasLtMatmulAlgoGetHeuristic(
                handle, op, a_desc, b_desc, c_desc, d_desc, pref, requested, heuristics.data(), &returned);
        if (h != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
            GGML_LOG_WARN("%s: hipBLASLt heuristic failed status=%s returned=%d\n",
                    __func__, ggml_cuda_hipblaslt_status_name(h), returned);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        selected_algo_ord = 0;
        if (plan.preferred_algo_ord >= 0) {
            if (plan.preferred_algo_ord < returned && heuristics[plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS) {
                selected_algo_ord = plan.preferred_algo_ord;
            } else if (plan.log_heuristic) {
                GGML_LOG_WARN("%s: hipBLASLt preferred algo_ord=%d unavailable state=%s returned=%d; falling back to first successful heuristic\n",
                        __func__, plan.preferred_algo_ord,
                        plan.preferred_algo_ord < returned ? ggml_cuda_hipblaslt_status_name(heuristics[plan.preferred_algo_ord].state) : "not_returned",
                        returned);
            }
        }
        if (heuristics[selected_algo_ord].state != HIPBLAS_STATUS_SUCCESS) {
            selected_algo_ord = -1;
            for (int i = 0; i < returned; ++i) {
                if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS) {
                    selected_algo_ord = i;
                    break;
                }
            }
        }
        if (selected_algo_ord < 0) {
            GGML_LOG_WARN("%s: hipBLASLt heuristic returned no successful algo returned=%d requested=%d\n", __func__, returned, requested);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        if (heuristics[selected_algo_ord].workspaceSize > args.scratch.hipblaslt_workspace_bytes) {
            GGML_LOG_WARN("%s: hipBLASLt selected algo workspace too large selected=%d need=%zu have=%zu\n",
                    __func__, selected_algo_ord, heuristics[selected_algo_ord].workspaceSize, args.scratch.hipblaslt_workspace_bytes);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (plan.log_heuristic) {
            const char * route = "hipblaslt_i8_q8_0_k32_stage";
            if (plan.weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0) {
                route = "hipblaslt_i8_q4_0_k32_stage";
            } else if (plan.weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1) {
                route = "hipblaslt_i8_q4_1_k32_stage";
            } else if (plan.weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0) {
                route = "hipblaslt_i8_q5_0_k32_stage";
            } else if (plan.weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1) {
                route = "hipblaslt_i8_q5_1_k32_stage";
            } else if (plan.full_k_stage_k32_exact) {
                route = "hipblaslt_i8_q8_0_fullk_stage_k32_exact";
            }
            GGML_LOG_WARN("%s: hipBLASLt heuristic route=%s m=%d n=%d k=%d chunk_k=%d stride_a_blocks=%d stride_b_blocks=%d stride_d=%d requested=%d returned=%d preferred=%d selected=%d workspace=%zu state=%s scratch_a=%zu scratch_b=%zu scratch_c=%zu\n",
                    __func__, route,
                    plan.m, plan.n, plan.k, plan.chunk_k, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d,
                    requested, returned, plan.preferred_algo_ord, selected_algo_ord,
                    heuristics[selected_algo_ord].workspaceSize,
                    ggml_cuda_hipblaslt_status_name(heuristics[selected_algo_ord].state),
                    args.scratch.a_i8_bytes, args.scratch.b_i8_bytes, args.scratch.c_i32_bytes);
        }
    }

    {
        const int32_t alpha = 1;
        const int32_t beta = 0;
        const int k_blocks = plan.k / plan.chunk_k;
        const int threads = 256;
        const dim3 grid_a((plan.m * plan.chunk_k + threads - 1) / threads);
        const dim3 grid_b((plan.chunk_k * plan.n + threads - 1) / threads);
        const dim3 grid_full_a((plan.m * plan.k + threads - 1) / threads);
        const dim3 grid_full_b((plan.k * plan.n + threads - 1) / threads);
        const dim3 grid_c((plan.m * plan.n + threads - 1) / threads);

        if (plan.full_k_stage_k32_exact) {
            ggml_cuda_hipblaslt_i8_stage_full_a_q8_1<<<grid_full_a, threads, 0, args.stream>>>(
                    args.activations_q8_1, args.scratch.a_i8, plan.m, plan.k, plan.stride_a_blocks);
            switch (plan.weight_type) {
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0:
                    ggml_cuda_hipblaslt_i8_stage_full_b_q8_0<<<grid_full_b, threads, 0, args.stream>>>(
                            args.weights_q8_0, args.scratch.b_i8, plan.n, plan.k, plan.stride_b_blocks);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0:
                    ggml_cuda_hipblaslt_i8_stage_full_b_q4_0<<<grid_full_b, threads, 0, args.stream>>>(
                            args.weights_q4_0, args.scratch.b_i8, plan.n, plan.k, plan.stride_b_blocks);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1:
                    ggml_cuda_hipblaslt_i8_stage_full_b_q4_1<<<grid_full_b, threads, 0, args.stream>>>(
                            args.weights_q4_1, args.scratch.b_i8, plan.n, plan.k, plan.stride_b_blocks);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0:
                    ggml_cuda_hipblaslt_i8_stage_full_b_q5_0<<<grid_full_b, threads, 0, args.stream>>>(
                            args.weights_q5_0, args.scratch.b_i8, plan.n, plan.k, plan.stride_b_blocks);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1:
                    ggml_cuda_hipblaslt_i8_stage_full_b_q5_1<<<grid_full_b, threads, 0, args.stream>>>(
                            args.weights_q5_1, args.scratch.b_i8, plan.n, plan.k, plan.stride_b_blocks);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S:
                    result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
                    goto cleanup;
            }
            CUDA_CHECK(cudaGetLastError());
        }

        for (int kb = 0; kb < k_blocks; ++kb) {
            int8_t * a_block = args.scratch.a_i8;
            int8_t * b_block = args.scratch.b_i8;
            if (plan.full_k_stage_k32_exact) {
                a_block += (size_t) kb * (size_t) plan.chunk_k;
                b_block += (size_t) kb * (size_t) plan.chunk_k * (size_t) plan.n;
            } else {
                ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, args.stream>>>(
                        args.activations_q8_1, a_block, kb, plan.m, plan.stride_a_blocks);
                switch (plan.weight_type) {
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0:
                        ggml_cuda_hipblaslt_i8_stage_b_q8_0<<<grid_b, threads, 0, args.stream>>>(
                                args.weights_q8_0, b_block, kb, plan.n, plan.stride_b_blocks);
                        break;
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0:
                        ggml_cuda_hipblaslt_i8_stage_b_q4_0<<<grid_b, threads, 0, args.stream>>>(
                                args.weights_q4_0, b_block, kb, plan.n, plan.stride_b_blocks);
                        break;
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1:
                        ggml_cuda_hipblaslt_i8_stage_b_q4_1<<<grid_b, threads, 0, args.stream>>>(
                                args.weights_q4_1, b_block, kb, plan.n, plan.stride_b_blocks);
                        break;
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0:
                        ggml_cuda_hipblaslt_i8_stage_b_q5_0<<<grid_b, threads, 0, args.stream>>>(
                                args.weights_q5_0, b_block, kb, plan.n, plan.stride_b_blocks);
                        break;
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1:
                        ggml_cuda_hipblaslt_i8_stage_b_q5_1<<<grid_b, threads, 0, args.stream>>>(
                                args.weights_q5_1, b_block, kb, plan.n, plan.stride_b_blocks);
                        break;
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS:
                    case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S:
                        result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
                        goto cleanup;
                }
                CUDA_CHECK(cudaGetLastError());
            }

            GGML_HIPBLASLT_I8_TRY(hipblasLtMatmul(
                        handle, op,
                        &alpha,
                        b_block, a_desc,
                        a_block, b_desc,
                        &beta,
                        args.scratch.c_i32, c_desc,
                        args.scratch.c_i32, d_desc,
                        &heuristics[selected_algo_ord].algo,
                        args.scratch.hipblaslt_workspace,
                        args.scratch.hipblaslt_workspace_bytes,
                        args.stream));

            switch (plan.weight_type) {
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0<<<grid_c, threads, 0, args.stream>>>(
                            args.scratch.c_i32, args.activations_q8_1, args.weights_q8_0, args.dst,
                            kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_0<<<grid_c, threads, 0, args.stream>>>(
                            args.scratch.c_i32, args.activations_q8_1, args.weights_q4_0, args.dst,
                            kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_1<<<grid_c, threads, 0, args.stream>>>(
                            args.scratch.c_i32, args.activations_q8_1, args.weights_q4_1, args.dst,
                            kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_0<<<grid_c, threads, 0, args.stream>>>(
                            args.scratch.c_i32, args.activations_q8_1, args.weights_q5_0, args.dst,
                            kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_1<<<grid_c, threads, 0, args.stream>>>(
                            args.scratch.c_i32, args.activations_q8_1, args.weights_q5_1, args.dst,
                            kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S:
                    result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
                    goto cleanup;
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }

cleanup:
    if (pref   != nullptr) { (void) hipblasLtMatmulPreferenceDestroy(pref); }
    if (a_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(a_desc); }
    if (b_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(b_desc); }
    if (c_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(c_desc); }
    if (d_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(d_desc); }
    if (op     != nullptr) { (void) hipblasLtMatmulDescDestroy(op); }
    if (handle != nullptr) { (void) hipblasLtDestroy(handle); }
    return result;

#undef GGML_HIPBLASLT_I8_TRY
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_iq3_s_range(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const block_q8_1 * activations_q8_1,
        const block_iq3_s * weights_iq3_s,
        const int32_t * ids_dst,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream) {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a_desc = nullptr;
    hipblasLtMatrixLayout_t b_desc = nullptr;
    hipblasLtMatrixLayout_t c_desc = nullptr;
    hipblasLtMatrixLayout_t d_desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::array<hipblasLtMatmulHeuristicResult_t, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS> heuristics{};
    int returned = 0;
    int selected_algo_ord = 0;
    ggml_cuda_hipblaslt_i8_status result = GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;

#define GGML_HIPBLASLT_I8_TRY(expr) do { \
        const hipblasStatus_t hblt_status = (expr); \
        if (hblt_status != HIPBLAS_STATUS_SUCCESS) { \
            GGML_LOG_WARN("%s: hipBLASLt call failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(hblt_status)); \
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR; \
            goto cleanup; \
        } \
    } while (0)

    if (plan.m <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if (plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0 || plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K ||
            activations_q8_1 == nullptr || weights_iq3_s == nullptr || ids_dst == nullptr || dst == nullptr || stream == nullptr ||
            scratch.a_i8 == nullptr || scratch.b_i8 == nullptr || scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    GGML_HIPBLASLT_I8_TRY(hipblasLtCreate(&handle));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));

    // Column-major transposed view: D_col(N,M) = B_col(N,K32) * A_col(K32,M),
    // equivalent to row-major D[M,N] = A[M,K32] * B[K32,N].
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_8I,  plan.n, plan.chunk_k, plan.n));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_8I,  plan.chunk_k, plan.m, plan.chunk_k));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_32I, plan.n, plan.m, plan.n));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32I, plan.n, plan.m, plan.n));

    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceCreate(&pref));
    {
        uint64_t workspace_bytes = scratch.hipblaslt_workspace_bytes;
        GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceSetAttribute(
                    pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
    }

    {
        const int requested = std::max(1, std::min(plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        const hipblasStatus_t h = hipblasLtMatmulAlgoGetHeuristic(
                handle, op, a_desc, b_desc, c_desc, d_desc, pref, requested, heuristics.data(), &returned);
        if (h != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
            if (plan.log_heuristic) {
                GGML_LOG_WARN("%s: hipBLASLt IQ3_S heuristic failed status=%s returned=%d m=%d n=%d k=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(h), returned, plan.m, plan.n, plan.k);
            }
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        selected_algo_ord = 0;
        if (plan.preferred_algo_ord >= 0 && plan.preferred_algo_ord < returned &&
                heuristics[plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS) {
            selected_algo_ord = plan.preferred_algo_ord;
        } else if (heuristics[selected_algo_ord].state != HIPBLAS_STATUS_SUCCESS) {
            selected_algo_ord = -1;
            for (int i = 0; i < returned; ++i) {
                if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS) {
                    selected_algo_ord = i;
                    break;
                }
            }
        }
        if (selected_algo_ord < 0) {
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        if (heuristics[selected_algo_ord].workspaceSize > scratch.hipblaslt_workspace_bytes) {
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (plan.log_heuristic) {
            GGML_LOG_WARN("%s: hipBLASLt IQ3_S route=mmeid_k32_forced m=%d n=%d k=%d selected=%d returned=%d workspace=%zu\n",
                    __func__, plan.m, plan.n, plan.k, selected_algo_ord, returned, heuristics[selected_algo_ord].workspaceSize);
        }
    }

    {
        const int32_t alpha = 1;
        const int32_t beta = 0;
        const int k_blocks = plan.k / plan.chunk_k;
        const int threads = 256;
        const dim3 grid_a((plan.m * plan.chunk_k + threads - 1) / threads);
        const dim3 grid_b((plan.chunk_k * plan.n + threads - 1) / threads);
        const dim3 grid_c((plan.m * plan.n + threads - 1) / threads);

        for (int kb = 0; kb < k_blocks; ++kb) {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                    activations_q8_1, scratch.a_i8, kb, plan.m, plan.stride_a_blocks);
            ggml_cuda_hipblaslt_i8_stage_b_iq3_s<<<grid_b, threads, 0, stream>>>(
                    weights_iq3_s, scratch.b_i8, kb, plan.n, plan.stride_b_blocks);
            CUDA_CHECK(cudaGetLastError());

            GGML_HIPBLASLT_I8_TRY(hipblasLtMatmul(
                        handle, op,
                        &alpha,
                        scratch.b_i8, a_desc,
                        scratch.a_i8, b_desc,
                        &beta,
                        scratch.c_i32, c_desc,
                        scratch.c_i32, d_desc,
                        &heuristics[selected_algo_ord].algo,
                        scratch.hipblaslt_workspace,
                        scratch.hipblaslt_workspace_bytes,
                        stream));

            ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_s_ids<<<grid_c, threads, 0, stream>>>(
                    scratch.c_i32, activations_q8_1, weights_iq3_s, ids_dst, dst,
                    kb, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
            CUDA_CHECK(cudaGetLastError());
        }
    }

cleanup:
    if (pref   != nullptr) { (void) hipblasLtMatmulPreferenceDestroy(pref); }
    if (a_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(a_desc); }
    if (b_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(b_desc); }
    if (c_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(c_desc); }
    if (d_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(d_desc); }
    if (op     != nullptr) { (void) hipblasLtMatmulDescDestroy(op); }
    if (handle != nullptr) { (void) hipblasLtDestroy(handle); }
    return result;

#undef GGML_HIPBLASLT_I8_TRY
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_q8_0_grouped_direct_qs(
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const block_q8_1 * activations_q8_1,
        const block_q8_0 * weights_q8_0,
        const int32_t * ids_dst,
        const std::vector<int32_t> & active_experts,
        const std::vector<int32_t> & expert_bounds_host,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        int stride_expert_blocks) {
    if (active_experts.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if ((base_plan.k != 0 && base_plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) ||
            base_plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K || base_plan.stride_b_blocks <= 0 ||
            activations_q8_1 == nullptr || weights_q8_0 == nullptr || ids_dst == nullptr || dst == nullptr ||
            stream == nullptr || scratch.c_i32 == nullptr || scratch.hipblaslt_user_args == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int group_count = (int) active_experts.size();
    const int k_blocks = base_plan.stride_b_blocks;
    const int n = base_plan.n;
    int ne_get_rows = 0;
    for (const int expert : active_experts) {
        ne_get_rows = std::max(ne_get_rows, expert_bounds_host[expert + 1]);
    }
    if (k_blocks <= 0 || n <= 0 || ne_get_rows <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    if (scratch.hipblaslt_user_args_bytes < user_args_bytes) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t handle_status = ggml_cuda_hipblaslt_i8_get_tls_handle(&handle);
    if (handle_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt cached handle failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(handle_status));
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    try {
        const int64_t lda_direct = (int64_t) base_plan.stride_b_blocks * (int64_t) sizeof(block_q8_0);
        const int64_t ldb_direct = (int64_t) base_plan.stride_a_blocks * (int64_t) sizeof(block_q8_1);
        std::vector<int64_t> m(group_count, n);
        std::vector<int64_t> n_vec(group_count, 0);
        std::vector<int64_t> k(group_count, base_plan.chunk_k);
        std::vector<int64_t> batch_count(group_count, 1);
        std::vector<int64_t> lda(group_count, lda_direct);
        std::vector<int64_t> ldb(group_count, ldb_direct);
        std::vector<int64_t> ldc(group_count, n);
        std::vector<int64_t> ldd(group_count, n);
        std::vector<int64_t> stride_a(group_count, 0);
        std::vector<int64_t> stride_b(group_count, 0);
        std::vector<int64_t> stride_c(group_count, 0);
        std::vector<int64_t> stride_d(group_count, 0);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(group_count);
        std::vector<hipblaslt_ext::GemmInputs> inputs(group_count);
        std::vector<int32_t> alpha(group_count, 1);
        std::vector<int32_t> beta(group_count, 0);

        for (int g = 0; g < group_count; ++g) {
            const int expert = active_experts[g];
            const int row_low = expert_bounds_host[expert];
            const int row_high = expert_bounds_host[expert + 1];
            const int rows = row_high - row_low;
            if (rows <= 0) {
                return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            }

            const block_q8_0 * wx = weights_q8_0 + (size_t) expert * stride_expert_blocks;
            const block_q8_1 * ax = activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks;
            n_vec[g] = rows;
            inputs[g].setA(wx[0].qs);
            inputs[g].setB(ax[0].qs);
            inputs[g].setC(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setD(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setAlpha(&alpha[g]);
            inputs[g].setBeta(&beta[g]);
        }

        hipblaslt_ext::GroupedGemm grouped(
                handle,
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblaslt_ext::GemmProblemType problem_type(
                HIPBLAS_OP_T, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblasStatus_t status = grouped.setProblem(
                m, n_vec, k, batch_count,
                lda, ldb, ldc, ldd,
                stride_a, stride_b, stride_c, stride_d,
                epilogue, inputs, problem_type);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: direct_qs Q8_0 setProblem failed: %s groups=%d lda=%" PRId64 " ldb=%" PRId64 "\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, lda_direct, ldb_direct);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(scratch.hipblaslt_workspace_bytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        const int requested = std::max(1, std::min(base_plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        status = grouped.algoGetHeuristic(requested, pref, heuristics);
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: direct_qs Q8_0 heuristic empty/fail: %s groups=%d returned=%zu; trying getAllAlgos\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            heuristics.clear();
            status = hipblaslt_ext::getAllAlgos(
                    handle,
                    hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                    HIPBLAS_OP_T, HIPBLAS_OP_N,
                    HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I,
                    heuristics);
        }
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: direct_qs Q8_0 getAllAlgos failed/empty: %s groups=%d returned=%zu\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }

        int selected = -1;
        size_t selected_workspace = 0;
        if (base_plan.preferred_algo_ord >= 0 && base_plan.preferred_algo_ord < (int) heuristics.size()) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[base_plan.preferred_algo_ord].algo;
            if (heuristics[base_plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = base_plan.preferred_algo_ord;
                selected_workspace = need;
            }
        }
        for (int i = 0; selected < 0 && i < (int) heuristics.size(); ++i) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[i].algo;
            if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = i;
                selected_workspace = need;
            }
        }
        if (selected < 0) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: direct_qs Q8_0 no supported algo groups=%d candidates=%zu\n",
                        __func__, group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }
        (void) cudaGetLastError();
        grouped.setMaxWorkspaceBytes(selected_workspace);

        std::vector<hipblaslt_ext::UserArguments> user_args(group_count);
        status = grouped.getDefaultValueForDeviceUserArguments(user_args.data());
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: direct_qs Q8_0 getDefaultValueForDeviceUserArguments failed: %s groups=%d selected=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, selected);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: direct_qs Q8_0 setup groups=%d n=%d k32=%d kblocks=%d selected=%d workspace=%zu candidates=%zu lda=%" PRId64 " ldb=%" PRId64 "\n",
                    __func__, group_count, n, base_plan.chunk_k, k_blocks, selected, selected_workspace, heuristics.size(), lda_direct, ldb_direct);
        }

        hipblaslt_ext::UserArguments * user_args_dev =
            reinterpret_cast<hipblaslt_ext::UserArguments *>(scratch.hipblaslt_user_args);
        const int threads = 256;
        for (int kb = 0; kb < k_blocks; ++kb) {
            for (int g = 0; g < group_count; ++g) {
                const int expert = active_experts[g];
                const int row_low = expert_bounds_host[expert];
                const block_q8_0 * wx = weights_q8_0 + (size_t) expert * stride_expert_blocks;
                const block_q8_1 * ax = activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks;
                user_args[g].a = const_cast<int8_t *>(wx[(size_t) kb].qs);
                user_args[g].b = const_cast<int8_t *>(ax[(size_t) kb].qs);
            }
            cudaError_t cuda_status = cudaMemcpyAsync(user_args_dev, user_args.data(), user_args_bytes, cudaMemcpyHostToDevice, stream);
            if (cuda_status != cudaSuccess) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: direct_qs Q8_0 user-args cudaMemcpyAsync failed: %s groups=%d bytes=%zu kb=%d\n",
                            __func__, cudaGetErrorString(cuda_status), group_count, user_args_bytes, kb);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }

            status = grouped.initialize(heuristics[selected].algo, scratch.hipblaslt_workspace, true, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: direct_qs Q8_0 initialize failed: %s groups=%d kb=%d selected=%d\n",
                            __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }

            status = grouped.run(user_args_dev, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: direct_qs Q8_0 run failed: %s groups=%d kb=%d selected=%d\n",
                            __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }

            for (int g = 0; g < group_count; ++g) {
                const int expert = active_experts[g];
                const int row_low = expert_bounds_host[expert];
                const int rows = expert_bounds_host[expert + 1] - row_low;
                const dim3 grid_c(((int64_t) rows * n + threads - 1) / threads);
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0_ids<<<grid_c, threads, 0, stream>>>(
                        scratch.c_i32 + (size_t) row_low * n,
                        activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                        weights_q8_0 + (size_t) expert * stride_expert_blocks,
                        ids_dst + row_low,
                        dst,
                        kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    } catch (...) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_k32_grouped(
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const block_q8_1 * activations_q8_1,
        const void * weights_k32,
        ggml_cuda_hipblaslt_i8_weight_type weight_type,
        const char * route_label,
        const int32_t * ids_dst,
        const std::vector<int32_t> & active_experts,
        const int32_t * active_experts_dev,
        const int32_t * expert_bounds_dev,
        const std::vector<int32_t> & expert_bounds_host,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        int stride_expert_blocks) {
    if (active_experts.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if ((base_plan.k != 0 && base_plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) ||
            base_plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K || base_plan.stride_b_blocks <= 0 ||
            activations_q8_1 == nullptr || weights_k32 == nullptr || ids_dst == nullptr || active_experts_dev == nullptr ||
            expert_bounds_dev == nullptr || dst == nullptr || stream == nullptr ||
            scratch.a_i8 == nullptr || scratch.b_i8 == nullptr || scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int group_count = (int) active_experts.size();
    const bool weight_stride_is_k256 = weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS ||
        weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S;
    const int k_blocks = weight_stride_is_k256 ?
        base_plan.stride_b_blocks * (QK_K / GGML_CUDA_HIPBLASLT_I8_CHUNK_K) : base_plan.stride_b_blocks;
    if (k_blocks <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    const int n = base_plan.n;
    int ne_get_rows = 0;
    int max_rows = 0;
    for (const int expert : active_experts) {
        const int row_low = expert_bounds_host[expert];
        const int row_high = expert_bounds_host[expert + 1];
        ne_get_rows = std::max(ne_get_rows, row_high);
        max_rows = std::max(max_rows, row_high - row_low);
    }
    if (ne_get_rows <= 0 || max_rows <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    const int threads = 256;
    const dim3 grid_a((ne_get_rows * base_plan.chunk_k + threads - 1) / threads);
    const dim3 grid_b(((int64_t) group_count * base_plan.chunk_k * n + threads - 1) / threads);
    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    if (scratch.hipblaslt_user_args == nullptr || scratch.hipblaslt_user_args_bytes < user_args_bytes) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t handle_status = ggml_cuda_hipblaslt_i8_get_tls_handle(&handle);
    if (handle_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt cached handle failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(handle_status));
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    try {
        std::vector<int64_t> m(group_count, n);
        std::vector<int64_t> n_vec(group_count, 0);
        std::vector<int64_t> k(group_count, base_plan.chunk_k);
        std::vector<int64_t> batch_count(group_count, 1);
        std::vector<int64_t> lda(group_count, n);
        std::vector<int64_t> ldb(group_count, base_plan.chunk_k);
        std::vector<int64_t> ldc(group_count, n);
        std::vector<int64_t> ldd(group_count, n);
        std::vector<int64_t> stride_a(group_count, 0);
        std::vector<int64_t> stride_b(group_count, 0);
        std::vector<int64_t> stride_c(group_count, 0);
        std::vector<int64_t> stride_d(group_count, 0);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(group_count);
        std::vector<hipblaslt_ext::GemmInputs> inputs(group_count);
        std::vector<int32_t> alpha(group_count, 1);
        std::vector<int32_t> beta(group_count, 0);

        for (int g = 0; g < group_count; ++g) {
            const int expert = active_experts[g];
            const int row_low = expert_bounds_host[expert];
            const int row_high = expert_bounds_host[expert + 1];
            const int rows = row_high - row_low;
            if (rows <= 0) {
                return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            }

            // Same column-major transposed layout as the working non-grouped path:
            // D_col(n,rows) = W_col(n,K32) * A_col(K32,rows).
            n_vec[g] = rows;
            inputs[g].setA(scratch.b_i8 + (size_t) g * base_plan.chunk_k * n);
            inputs[g].setB(scratch.a_i8 + (size_t) row_low * base_plan.chunk_k);
            inputs[g].setC(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setD(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setAlpha(&alpha[g]);
            inputs[g].setBeta(&beta[g]);
        }

        hipblaslt_ext::GroupedGemm grouped(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblaslt_ext::GemmProblemType problem_type(
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblasStatus_t status = grouped.setProblem(
                m, n_vec, k, batch_count,
                lda, ldb, ldc, ldd,
                stride_a, stride_b, stride_c, stride_d,
                epilogue, inputs, problem_type);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext setProblem failed: %s groups=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(scratch.hipblaslt_workspace_bytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        const int requested = std::max(1, std::min(base_plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        status = grouped.algoGetHeuristic(requested, pref, heuristics);
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext heuristic empty/fail: %s groups=%d returned=%zu; trying getAllAlgos\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            heuristics.clear();
            status = hipblaslt_ext::getAllAlgos(
                    handle,
                    hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I,
                    heuristics);
        }
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext getAllAlgos failed/empty: %s groups=%d returned=%zu\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }

        int selected = -1;
        size_t selected_workspace = 0;
        if (base_plan.preferred_algo_ord >= 0 && base_plan.preferred_algo_ord < (int) heuristics.size()) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[base_plan.preferred_algo_ord].algo;
            if (heuristics[base_plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = base_plan.preferred_algo_ord;
                selected_workspace = need;
            }
        }
        for (int i = 0; selected < 0 && i < (int) heuristics.size(); ++i) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[i].algo;
            if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = i;
                selected_workspace = need;
            }
        }
        if (selected < 0) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext no supported algo groups=%d candidates=%zu\n",
                        __func__, group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }

        // hipBLASLt grouped isAlgoSupported() can leave a stale HIP last-error even when it returns success.
        // Clear it before our staged-copy kernels so CUDA_CHECK reports only this route's launches.
        (void) cudaGetLastError();

        grouped.setMaxWorkspaceBytes(selected_workspace);

        std::vector<hipblaslt_ext::UserArguments> user_args(group_count);
        status = grouped.getDefaultValueForDeviceUserArguments(user_args.data());
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext getDefaultValueForDeviceUserArguments failed: %s groups=%d selected=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, selected);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        hipblaslt_ext::UserArguments * user_args_dev =
            reinterpret_cast<hipblaslt_ext::UserArguments *>(scratch.hipblaslt_user_args);
        cudaError_t cuda_status = cudaMemcpyAsync(user_args_dev, user_args.data(), user_args_bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext user-args cudaMemcpyAsync failed: %s groups=%d bytes=%zu\n",
                        __func__, cudaGetErrorString(cuda_status), group_count, user_args_bytes);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s setup_once groups=%d n=%d k32=%d kblocks=%d selected=%d workspace=%zu candidates=%zu user_args=pool init=per_kblock\n",
                    __func__, route_label, group_count, n, base_plan.chunk_k, k_blocks, selected, selected_workspace, heuristics.size());
        }

        // Some hipBLASLt-ext grouped setup helpers leave HIP's sticky last-error set despite success.
        // Do not attribute that host-side probe state to the following staging kernels.
        (void) cudaGetLastError();

    const bool q8_0_fused_scale = weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0 &&
        !ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_NO_FUSED_SCALE");
    const bool q8_0_tiled_stage_b = weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0 &&
        ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_TILED_STAGE_B");
    const bool q8_0_fused_stage_ab = weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0 &&
        ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_FUSED_STAGE_AB");
    const bool init_once = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_GROUPED_INIT_ONCE");
    if (init_once) {
        status = grouped.initialize(heuristics[selected].algo, scratch.hipblaslt_workspace, true, stream);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext initialize_once failed: %s groups=%d selected=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, selected);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }
    }

    for (int kb = 0; kb < k_blocks; ++kb) {
        if (q8_0_fused_stage_ab) {
            const int total_a = ne_get_rows * QK8_1;
            const int total_b = group_count * QK8_0 * n;
            const int total = std::max(total_a, total_b);
            const dim3 grid_ab((total + threads - 1) / threads);
            ggml_cuda_hipblaslt_i8_stage_ab_q8_1_q8_0_grouped<<<grid_ab, threads, 0, stream>>>(
                    activations_q8_1,
                    (const block_q8_0 *) weights_k32,
                    active_experts_dev,
                    scratch.a_i8,
                    scratch.b_i8,
                    kb, ne_get_rows, n,
                    base_plan.stride_a_blocks,
                    base_plan.stride_b_blocks,
                    stride_expert_blocks,
                    group_count);
        } else {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                    activations_q8_1, scratch.a_i8, kb, ne_get_rows, base_plan.stride_a_blocks);
        }
        switch (weight_type) {
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0:
                if (q8_0_fused_stage_ab) {
                    // A and B were staged together above.
                } else if (q8_0_tiled_stage_b) {
                    const dim3 grid_b_tiled((n + 7) / 8, group_count, 1);
                    const dim3 block_b_tiled(QK8_0, 8, 1);
                    ggml_cuda_hipblaslt_i8_stage_b_q8_0_grouped_tiled<<<grid_b_tiled, block_b_tiled, 0, stream>>>(
                            (const block_q8_0 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                            base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                } else {
                    ggml_cuda_hipblaslt_i8_stage_b_q8_0_grouped<<<grid_b, threads, 0, stream>>>(
                            (const block_q8_0 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                            base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                }
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0:
                ggml_cuda_hipblaslt_i8_stage_b_q4_0_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q4_0 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1:
                ggml_cuda_hipblaslt_i8_stage_b_q4_1_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q4_1 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0:
                ggml_cuda_hipblaslt_i8_stage_b_q5_0_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q5_0 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1:
                ggml_cuda_hipblaslt_i8_stage_b_q5_1_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q5_1 *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K:
                ggml_cuda_hipblaslt_i8_stage_b_q4_K_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q4_K *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K:
                ggml_cuda_hipblaslt_i8_stage_b_q5_K_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q5_K *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K:
                ggml_cuda_hipblaslt_i8_stage_b_q8_K_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q8_K *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS:
                ggml_cuda_hipblaslt_i8_stage_b_iq4_xs_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq4_xs *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL:
                ggml_cuda_hipblaslt_i8_stage_b_iq4_nl_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq4_nl *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS:
                ggml_cuda_hipblaslt_i8_stage_b_iq3_xxs_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq3_xxs *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS:
                ggml_cuda_hipblaslt_i8_stage_b_iq2_xxs_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq2_xxs *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S:
                ggml_cuda_hipblaslt_i8_stage_b_iq1_s_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq1_s *) weights_k32, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
                break;
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS:
            case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S:
                return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            default:
                return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
        }
        CUDA_CHECK(cudaGetLastError());

        if (!init_once) {
            status = grouped.initialize(heuristics[selected].algo, scratch.hipblaslt_workspace, true, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext initialize failed: %s groups=%d kb=%d selected=%d\n",
                            __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }
        }

        status = grouped.run(user_args_dev, stream);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext user-args run failed: %s groups=%d kb=%d selected=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        if (q8_0_fused_scale) {
            const dim3 grid_c(((int64_t) max_rows * n + threads - 1) / threads, group_count, 1);
            ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0_ids_grouped<<<grid_c, threads, 0, stream>>>(
                    scratch.c_i32,
                    activations_q8_1,
                    (const block_q8_0 *) weights_k32,
                    ids_dst,
                    active_experts_dev,
                    expert_bounds_dev,
                    dst,
                    kb, max_rows, n,
                    base_plan.stride_a_blocks,
                    base_plan.stride_b_blocks,
                    stride_expert_blocks,
                    base_plan.stride_d,
                    group_count);
            CUDA_CHECK(cudaGetLastError());
            continue;
        }

        for (int g = 0; g < group_count; ++g) {
            const int expert = active_experts[g];
            const int row_low = expert_bounds_host[expert];
            const int rows = expert_bounds_host[expert + 1] - row_low;
            const dim3 grid_c(((int64_t) rows * n + threads - 1) / threads);
            switch (weight_type) {
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q8_0 *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_0_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q4_0 *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_1_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q4_1 *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_0_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q5_0 *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_1_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q5_1 *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_K_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q4_K *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_K_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q5_K *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_K_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_q8_K *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_xs_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_iq4_xs *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_nl_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_iq4_nl *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_xxs_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_iq3_xxs *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xxs_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_iq2_xxs *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S:
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq1_s_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            (const block_iq1_s *) weights_k32 + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    break;
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS:
                case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S:
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
                default:
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    } catch (...) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}



struct ggml_cuda_hipblaslt_i8_grouped_k16_half_state {
    std::unique_ptr<hipblaslt_ext::GroupedGemm> grouped;
    std::unique_ptr<hipblaslt_ext::GemmProblemType> problem_type;
    std::vector<int64_t> m;
    std::vector<int64_t> n_vec;
    std::vector<int64_t> k;
    std::vector<int64_t> batch_count;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;
    std::vector<int64_t> ldd;
    std::vector<int64_t> stride_a;
    std::vector<int64_t> stride_b;
    std::vector<int64_t> stride_c;
    std::vector<int64_t> stride_d;
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue;
    std::vector<hipblaslt_ext::GemmInputs> inputs;
    std::vector<int32_t> alpha;
    std::vector<int32_t> beta;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    hipblaslt_ext::UserArguments * user_args_dev = nullptr;
    int selected = -1;
    size_t selected_workspace = 0;
};

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_prepare_grouped_k16_half(
        hipblasLtHandle_t handle,
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const char * route_label,
        const int half16,
        const std::vector<int32_t> & active_experts,
        const std::vector<int32_t> & expert_bounds_host,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        const bool init_once,
        ggml_cuda_hipblaslt_i8_grouped_k16_half_state & state) {
    const int group_count = (int) active_experts.size();
    const int n = base_plan.n;
    if (handle == nullptr || group_count <= 0 || (half16 != 0 && half16 != 1) || scratch.hipblaslt_user_args == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    state.m.assign(group_count, n);
    state.n_vec.assign(group_count, 0);
    state.k.assign(group_count, 16);
    state.batch_count.assign(group_count, 1);
    state.lda.assign(group_count, n);
    state.ldb.assign(group_count, base_plan.chunk_k);
    state.ldc.assign(group_count, n);
    state.ldd.assign(group_count, n);
    state.stride_a.assign(group_count, 0);
    state.stride_b.assign(group_count, 0);
    state.stride_c.assign(group_count, 0);
    state.stride_d.assign(group_count, 0);
    state.epilogue.assign(group_count, hipblaslt_ext::GemmEpilogue{});
    state.inputs.assign(group_count, hipblaslt_ext::GemmInputs{});
    state.alpha.assign(group_count, 1);
    state.beta.assign(group_count, 0);

    for (int g = 0; g < group_count; ++g) {
        const int expert = active_experts[g];
        const int row_low = expert_bounds_host[expert];
        const int row_high = expert_bounds_host[expert + 1];
        const int rows = row_high - row_low;
        if (rows <= 0) {
            return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
        }

        // D_col(n,rows) = W_col(n,K16) * A_col(K16,rows), with K16 selected from the K32 staging buffer.
        state.n_vec[g] = rows;
        state.inputs[g].setA(scratch.b_i8 + (size_t) g * base_plan.chunk_k * n + (size_t) half16 * 16 * n);
        state.inputs[g].setB(scratch.a_i8 + (size_t) row_low * base_plan.chunk_k + half16 * 16);
        state.inputs[g].setC(scratch.c_i32 + (size_t) row_low * n);
        state.inputs[g].setD(scratch.c_i32 + (size_t) row_low * n);
        state.inputs[g].setAlpha(&state.alpha[g]);
        state.inputs[g].setBeta(&state.beta[g]);
    }

    state.grouped.reset(new hipblaslt_ext::GroupedGemm(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I));
    state.problem_type.reset(new hipblaslt_ext::GemmProblemType(
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I));

    hipblasStatus_t status = state.grouped->setProblem(
            state.m, state.n_vec, state.k, state.batch_count,
            state.lda, state.ldb, state.ldc, state.ldd,
            state.stride_a, state.stride_b, state.stride_c, state.stride_d,
            state.epilogue, state.inputs, *state.problem_type);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s setProblem failed: %s groups=%d half=%d\n",
                    __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, half16);
        }
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(scratch.hipblaslt_workspace_bytes);
    const int requested = std::max(1, std::min(base_plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
    status = state.grouped->algoGetHeuristic(requested, pref, state.heuristics);
    if (status != HIPBLAS_STATUS_SUCCESS || state.heuristics.empty()) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s heuristic empty/fail: %s groups=%d half=%d returned=%zu; trying getAllAlgos\n",
                    __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, half16, state.heuristics.size());
        }
        state.heuristics.clear();
        status = hipblaslt_ext::getAllAlgos(
                handle,
                hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I,
                state.heuristics);
    }
    if (status != HIPBLAS_STATUS_SUCCESS || state.heuristics.empty()) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s getAllAlgos failed/empty: %s groups=%d half=%d returned=%zu\n",
                    __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, half16, state.heuristics.size());
        }
        return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
    }

    state.selected = -1;
    state.selected_workspace = 0;
    if (base_plan.preferred_algo_ord >= 0 && base_plan.preferred_algo_ord < (int) state.heuristics.size()) {
        size_t need = 0;
        hipblasLtMatmulAlgo_t algo = state.heuristics[base_plan.preferred_algo_ord].algo;
        if (state.heuristics[base_plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS &&
                state.grouped->isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                need <= scratch.hipblaslt_workspace_bytes) {
            state.selected = base_plan.preferred_algo_ord;
            state.selected_workspace = need;
        }
    }
    for (int i = 0; state.selected < 0 && i < (int) state.heuristics.size(); ++i) {
        size_t need = 0;
        hipblasLtMatmulAlgo_t algo = state.heuristics[i].algo;
        if (state.heuristics[i].state == HIPBLAS_STATUS_SUCCESS &&
                state.grouped->isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                need <= scratch.hipblaslt_workspace_bytes) {
            state.selected = i;
            state.selected_workspace = need;
        }
    }
    if (state.selected < 0) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s no supported algo groups=%d half=%d candidates=%zu\n",
                    __func__, route_label, group_count, half16, state.heuristics.size());
        }
        return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
    }

    (void) cudaGetLastError();
    state.grouped->setMaxWorkspaceBytes(state.selected_workspace);

    std::vector<hipblaslt_ext::UserArguments> user_args(group_count);
    status = state.grouped->getDefaultValueForDeviceUserArguments(user_args.data());
    if (status != HIPBLAS_STATUS_SUCCESS) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s getDefaultValueForDeviceUserArguments failed: %s groups=%d half=%d selected=%d\n",
                    __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, half16, state.selected);
        }
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    state.user_args_dev = reinterpret_cast<hipblaslt_ext::UserArguments *>(
            (char *) scratch.hipblaslt_user_args + (size_t) half16 * user_args_bytes);
    cudaError_t cuda_status = cudaMemcpyAsync(state.user_args_dev, user_args.data(), user_args_bytes, cudaMemcpyHostToDevice, stream);
    if (cuda_status != cudaSuccess) {
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext %s user-args cudaMemcpyAsync failed: %s groups=%d half=%d bytes=%zu\n",
                    __func__, route_label, cudaGetErrorString(cuda_status), group_count, half16, user_args_bytes);
        }
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    if (base_plan.log_heuristic) {
        GGML_LOG_WARN("%s: grouped ext %s setup_half groups=%d n=%d k16=16 half=%d selected=%d workspace=%zu candidates=%zu user_args=pool init=%s\n",
                __func__, route_label, group_count, n, half16, state.selected, state.selected_workspace, state.heuristics.size(),
                init_once ? "once" : "per_kblock");
    }
    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_k16_grouped(
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const block_q8_1 * activations_q8_1,
        const float * activations_k16_sums,
        const void * weights_k16,
        ggml_cuda_hipblaslt_i8_weight_type weight_type,
        const char * route_label,
        const int32_t * ids_dst,
        const std::vector<int32_t> & active_experts,
        const int32_t * active_experts_dev,
        const std::vector<int32_t> & expert_bounds_host,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        int stride_expert_blocks) {
    if (active_experts.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if (base_plan.k % QK_K != 0 || base_plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0 ||
            base_plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K ||
            activations_q8_1 == nullptr || weights_k16 == nullptr || ids_dst == nullptr || active_experts_dev == nullptr ||
            dst == nullptr || stream == nullptr || scratch.a_i8 == nullptr || scratch.b_i8 == nullptr || scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K && activations_k16_sums == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (weight_type != GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K && weight_type != GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K &&
            weight_type != GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS && weight_type != GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int group_count = (int) active_experts.size();
    const int k_blocks = base_plan.k / base_plan.chunk_k;
    const int n = base_plan.n;
    const int ne_get_rows = expert_bounds_host.back();
    const int threads = 256;
    const dim3 grid_a((ne_get_rows * base_plan.chunk_k + threads - 1) / threads);
    const dim3 grid_b(((int64_t) group_count * base_plan.chunk_k * n + threads - 1) / threads);
    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    if (scratch.hipblaslt_user_args == nullptr || scratch.hipblaslt_user_args_bytes < user_args_bytes * 2) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t handle_status = ggml_cuda_hipblaslt_i8_get_tls_handle(&handle);
    if (handle_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt cached handle failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(handle_status));
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }
    const bool init_once = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_GROUPED_INIT_ONCE");

    try {
        std::array<ggml_cuda_hipblaslt_i8_grouped_k16_half_state, 2> half_states;
        for (int half16 = 0; half16 < 2; ++half16) {
            const ggml_cuda_hipblaslt_i8_status setup_status = ggml_cuda_hipblaslt_i8_prepare_grouped_k16_half(
                    handle, base_plan, route_label, half16, active_experts, expert_bounds_host, scratch, stream, init_once, half_states[half16]);
            if (setup_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return setup_status;
            }
        }
        if (init_once) {
            for (int half16 = 0; half16 < 2; ++half16) {
                auto & half = half_states[half16];
                hipblasStatus_t status = half.grouped->initialize(half.heuristics[half.selected].algo, scratch.hipblaslt_workspace, true, stream);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (base_plan.log_heuristic) {
                        GGML_LOG_WARN("%s: grouped ext %s initialize_once failed: %s groups=%d half=%d selected=%d\n",
                                __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, half16, half.selected);
                    }
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                }
            }
        }

        (void) cudaGetLastError();
        for (int kb = 0; kb < k_blocks; ++kb) {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                    activations_q8_1, scratch.a_i8, kb, ne_get_rows, base_plan.stride_a_blocks);
            if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K) {
                ggml_cuda_hipblaslt_i8_stage_b_q2_K_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q2_K *) weights_k16, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            } else if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K) {
                ggml_cuda_hipblaslt_i8_stage_b_q3_K_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_q3_K *) weights_k16, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            } else if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS) {
                ggml_cuda_hipblaslt_i8_stage_b_iq2_xs_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq2_xs *) weights_k16, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            } else {
                ggml_cuda_hipblaslt_i8_stage_b_iq2_s_grouped<<<grid_b, threads, 0, stream>>>(
                        (const block_iq2_s *) weights_k16, active_experts_dev, scratch.b_i8, kb, n,
                        base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            }
            CUDA_CHECK(cudaGetLastError());

            for (int half16 = 0; half16 < 2; ++half16) {
                auto & half = half_states[half16];
                hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
                if (!init_once) {
                    status = half.grouped->initialize(half.heuristics[half.selected].algo, scratch.hipblaslt_workspace, true, stream);
                    if (status != HIPBLAS_STATUS_SUCCESS) {
                        if (base_plan.log_heuristic) {
                            GGML_LOG_WARN("%s: grouped ext %s initialize failed: %s groups=%d kb=%d half=%d selected=%d\n",
                                    __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, kb, half16, half.selected);
                        }
                        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                    }
                }

                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext %s route groups=%d n=%d k16=16 kb=%d half=%d selected=%d workspace=%zu candidates=%zu user_args=pool\n",
                            __func__, route_label, group_count, n, kb, half16, half.selected, half.selected_workspace, half.heuristics.size());
                }
                status = half.grouped->run(half.user_args_dev, stream);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (base_plan.log_heuristic) {
                        GGML_LOG_WARN("%s: grouped ext %s user-args run failed: %s groups=%d kb=%d half=%d selected=%d\n",
                                __func__, route_label, ggml_cuda_hipblaslt_status_name(status), group_count, kb, half16, half.selected);
                    }
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                }

                for (int g = 0; g < group_count; ++g) {
                    const int expert = active_experts[g];
                    const int row_low = expert_bounds_host[expert];
                    const int rows = expert_bounds_host[expert + 1] - row_low;
                    const dim3 grid_c(((int64_t) rows * n + threads - 1) / threads);
                    if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K) {
                        ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q2_K_ids<<<grid_c, threads, 0, stream>>>(
                                scratch.c_i32 + (size_t) row_low * n,
                                activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                                activations_k16_sums + (size_t) row_low * base_plan.stride_a_blocks * 2,
                                (const block_q2_K *) weights_k16 + (size_t) expert * stride_expert_blocks,
                                ids_dst + row_low,
                                dst,
                                kb, half16, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    } else if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K) {
                        ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q3_K_ids<<<grid_c, threads, 0, stream>>>(
                                scratch.c_i32 + (size_t) row_low * n,
                                activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                                (const block_q3_K *) weights_k16 + (size_t) expert * stride_expert_blocks,
                                ids_dst + row_low,
                                dst,
                                kb, half16, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    } else if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS) {
                        ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xs_ids<<<grid_c, threads, 0, stream>>>(
                                scratch.c_i32 + (size_t) row_low * n,
                                activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                                (const block_iq2_xs *) weights_k16 + (size_t) expert * stride_expert_blocks,
                                ids_dst + row_low,
                                dst,
                                kb, half16, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    } else {
                        ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_s_ids<<<grid_c, threads, 0, stream>>>(
                                scratch.c_i32 + (size_t) row_low * n,
                                activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                                (const block_iq2_s *) weights_k16 + (size_t) expert * stride_expert_blocks,
                                ids_dst + row_low,
                                dst,
                                kb, half16, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
            }
        }
    } catch (...) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_q6_K_grouped(
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const block_q8_1 * activations_q8_1,
        const block_q6_K * weights_q6_K,
        const int32_t * ids_dst,
        const std::vector<int32_t> & active_experts,
        const int32_t * active_experts_dev,
        const std::vector<int32_t> & expert_bounds_host,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        int stride_expert_blocks) {
    if (active_experts.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if (base_plan.k % QK_K != 0 || base_plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0 ||
            base_plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K ||
            activations_q8_1 == nullptr || weights_q6_K == nullptr || ids_dst == nullptr || active_experts_dev == nullptr ||
            dst == nullptr || stream == nullptr || scratch.a_i8 == nullptr || scratch.b_i8 == nullptr || scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int group_count = (int) active_experts.size();
    const int k_blocks = base_plan.k / base_plan.chunk_k;
    const int n = base_plan.n;
    const int ne_get_rows = expert_bounds_host.back();
    const int threads = 256;
    const dim3 grid_a((ne_get_rows * base_plan.chunk_k + threads - 1) / threads);
    const dim3 grid_b(((int64_t) group_count * base_plan.chunk_k * n + threads - 1) / threads);
    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    if (scratch.hipblaslt_user_args == nullptr || scratch.hipblaslt_user_args_bytes < user_args_bytes * 2) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t handle_status = ggml_cuda_hipblaslt_i8_get_tls_handle(&handle);
    if (handle_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt cached handle failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(handle_status));
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }
    const bool init_once = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_GROUPED_INIT_ONCE");

    try {
        std::array<ggml_cuda_hipblaslt_i8_grouped_k16_half_state, 2> half_states;
        for (int half16 = 0; half16 < 2; ++half16) {
            const ggml_cuda_hipblaslt_i8_status setup_status = ggml_cuda_hipblaslt_i8_prepare_grouped_k16_half(
                    handle, base_plan, "Q6_K", half16, active_experts, expert_bounds_host, scratch, stream, init_once, half_states[half16]);
            if (setup_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return setup_status;
            }
        }
        if (init_once) {
            for (int half16 = 0; half16 < 2; ++half16) {
                auto & half = half_states[half16];
                hipblasStatus_t status = half.grouped->initialize(half.heuristics[half.selected].algo, scratch.hipblaslt_workspace, true, stream);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (base_plan.log_heuristic) {
                        GGML_LOG_WARN("%s: grouped ext Q6_K initialize_once failed: %s groups=%d half=%d selected=%d\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), group_count, half16, half.selected);
                    }
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                }
            }
        }

        (void) cudaGetLastError();
        for (int kb = 0; kb < k_blocks; ++kb) {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                    activations_q8_1, scratch.a_i8, kb, ne_get_rows, base_plan.stride_a_blocks);
            ggml_cuda_hipblaslt_i8_stage_b_q6_K_grouped<<<grid_b, threads, 0, stream>>>(
                    weights_q6_K, active_experts_dev, scratch.b_i8, kb, n,
                    base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            CUDA_CHECK(cudaGetLastError());

            for (int half16 = 0; half16 < 2; ++half16) {
                auto & half = half_states[half16];
                hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
                if (!init_once) {
                    status = half.grouped->initialize(half.heuristics[half.selected].algo, scratch.hipblaslt_workspace, true, stream);
                    if (status != HIPBLAS_STATUS_SUCCESS) {
                        if (base_plan.log_heuristic) {
                            GGML_LOG_WARN("%s: grouped ext Q6_K initialize failed: %s groups=%d kb=%d half=%d selected=%d\n",
                                    __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, half16, half.selected);
                        }
                        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                    }
                }

                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext Q6_K route groups=%d n=%d k16=16 kb=%d half=%d selected=%d workspace=%zu candidates=%zu user_args=pool\n",
                            __func__, group_count, n, kb, half16, half.selected, half.selected_workspace, half.heuristics.size());
                }
                status = half.grouped->run(half.user_args_dev, stream);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (base_plan.log_heuristic) {
                        GGML_LOG_WARN("%s: grouped ext Q6_K user-args run failed: %s groups=%d kb=%d half=%d selected=%d\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, half16, half.selected);
                    }
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                }

                for (int g = 0; g < group_count; ++g) {
                    const int expert = active_experts[g];
                    const int row_low = expert_bounds_host[expert];
                    const int rows = expert_bounds_host[expert + 1] - row_low;
                    const dim3 grid_c(((int64_t) rows * n + threads - 1) / threads);
                    ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q6_K_ids<<<grid_c, threads, 0, stream>>>(
                            scratch.c_i32 + (size_t) row_low * n,
                            activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                            weights_q6_K + (size_t) expert * stride_expert_blocks,
                            ids_dst + row_low,
                            dst,
                            kb, half16, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
                }
                CUDA_CHECK(cudaGetLastError());
            }
        }
    } catch (...) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

static ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_run_hipblaslt_iq3_s_grouped(
        const ggml_cuda_hipblaslt_i8_plan & base_plan,
        const block_q8_1 * activations_q8_1,
        const block_iq3_s * weights_iq3_s,
        const int32_t * ids_dst,
        const std::vector<int32_t> & active_experts,
        const int32_t * active_experts_dev,
        const std::vector<int32_t> & expert_bounds_host,
        float * dst,
        const ggml_cuda_hipblaslt_i8_scratch & scratch,
        cudaStream_t stream,
        int stride_expert_blocks) {
    if (active_experts.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    }
    if (base_plan.k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0 || base_plan.chunk_k != GGML_CUDA_HIPBLASLT_I8_CHUNK_K ||
            activations_q8_1 == nullptr || weights_iq3_s == nullptr || ids_dst == nullptr || active_experts_dev == nullptr ||
            dst == nullptr || stream == nullptr || scratch.a_i8 == nullptr || scratch.b_i8 == nullptr || scratch.c_i32 == nullptr ||
            expert_bounds_host.empty()) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int group_count = (int) active_experts.size();
    const int k_blocks = base_plan.k / base_plan.chunk_k;
    const int n = base_plan.n;
    const int ne_get_rows = expert_bounds_host.back();
    if (k_blocks <= 0 || n <= 0 || ne_get_rows <= 0) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    const int threads = 256;
    const int stage_b_threads = 64;
    const dim3 grid_a((ne_get_rows * base_plan.chunk_k + threads - 1) / threads);
    const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
    if (scratch.hipblaslt_user_args == nullptr || scratch.hipblaslt_user_args_bytes < user_args_bytes) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t handle_status = ggml_cuda_hipblaslt_i8_get_tls_handle(&handle);
    if (handle_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt cached handle failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(handle_status));
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }

    try {
        std::vector<int64_t> m(group_count, n);
        std::vector<int64_t> n_vec(group_count, 0);
        std::vector<int64_t> k(group_count, base_plan.chunk_k);
        std::vector<int64_t> batch_count(group_count, 1);
        std::vector<int64_t> lda(group_count, n);
        std::vector<int64_t> ldb(group_count, base_plan.chunk_k);
        std::vector<int64_t> ldc(group_count, n);
        std::vector<int64_t> ldd(group_count, n);
        std::vector<int64_t> stride_a(group_count, 0);
        std::vector<int64_t> stride_b(group_count, 0);
        std::vector<int64_t> stride_c(group_count, 0);
        std::vector<int64_t> stride_d(group_count, 0);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(group_count);
        std::vector<hipblaslt_ext::GemmInputs> inputs(group_count);
        std::vector<int32_t> alpha(group_count, 1);
        std::vector<int32_t> beta(group_count, 0);

        for (int g = 0; g < group_count; ++g) {
            const int expert = active_experts[g];
            const int row_low = expert_bounds_host[expert];
            const int row_high = expert_bounds_host[expert + 1];
            const int rows = row_high - row_low;
            if (rows <= 0) {
                return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            }

            // Same column-major transposed layout as the working non-grouped path:
            // D_col(n,rows) = W_col(n,K32) * A_col(K32,rows).
            n_vec[g] = rows;
            inputs[g].setA(scratch.b_i8 + (size_t) g * base_plan.chunk_k * n);
            inputs[g].setB(scratch.a_i8 + (size_t) row_low * base_plan.chunk_k);
            inputs[g].setC(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setD(scratch.c_i32 + (size_t) row_low * n);
            inputs[g].setAlpha(&alpha[g]);
            inputs[g].setBeta(&beta[g]);
        }

        hipblaslt_ext::GroupedGemm grouped(
                handle,
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblaslt_ext::GemmProblemType problem_type(
                HIPBLAS_OP_N, HIPBLAS_OP_N,
                HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
        hipblasStatus_t status = grouped.setProblem(
                m, n_vec, k, batch_count,
                lda, ldb, ldc, ldd,
                stride_a, stride_b, stride_c, stride_d,
                epilogue, inputs, problem_type);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext setProblem failed: %s groups=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(scratch.hipblaslt_workspace_bytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        const int requested = std::max(1, std::min(base_plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        status = grouped.algoGetHeuristic(requested, pref, heuristics);
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext heuristic empty/fail: %s groups=%d returned=%zu; trying getAllAlgos\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            heuristics.clear();
            status = hipblaslt_ext::getAllAlgos(
                    handle,
                    hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I,
                    heuristics);
        }
        if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext getAllAlgos failed/empty: %s groups=%d returned=%zu\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }

        int selected = -1;
        size_t selected_workspace = 0;
        if (base_plan.preferred_algo_ord >= 0 && base_plan.preferred_algo_ord < (int) heuristics.size()) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[base_plan.preferred_algo_ord].algo;
            if (heuristics[base_plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = base_plan.preferred_algo_ord;
                selected_workspace = need;
            }
        }
        for (int i = 0; selected < 0 && i < (int) heuristics.size(); ++i) {
            size_t need = 0;
            hipblasLtMatmulAlgo_t algo = heuristics[i].algo;
            if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS &&
                    grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= scratch.hipblaslt_workspace_bytes) {
                selected = i;
                selected_workspace = need;
            }
        }
        if (selected < 0) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext no supported algo groups=%d candidates=%zu\n",
                        __func__, group_count, heuristics.size());
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
        }

        // hipBLASLt grouped isAlgoSupported() can leave a stale HIP last-error even when it returns success.
        // Clear it before our staged-copy kernels so CUDA_CHECK reports only this route's launches.
        (void) cudaGetLastError();

        grouped.setMaxWorkspaceBytes(selected_workspace);

        std::vector<hipblaslt_ext::UserArguments> user_args(group_count);
        status = grouped.getDefaultValueForDeviceUserArguments(user_args.data());
        if (status != HIPBLAS_STATUS_SUCCESS) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext getDefaultValueForDeviceUserArguments failed: %s groups=%d selected=%d\n",
                        __func__, ggml_cuda_hipblaslt_status_name(status), group_count, selected);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        hipblaslt_ext::UserArguments * user_args_dev =
            reinterpret_cast<hipblaslt_ext::UserArguments *>(scratch.hipblaslt_user_args);
        cudaError_t cuda_status = cudaMemcpyAsync(user_args_dev, user_args.data(), user_args_bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext user-args cudaMemcpyAsync failed: %s groups=%d bytes=%zu\n",
                        __func__, cudaGetErrorString(cuda_status), group_count, user_args_bytes);
            }
            return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
        }

        const bool init_once = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_GROUPED_INIT_ONCE");
        if (base_plan.log_heuristic) {
            GGML_LOG_WARN("%s: grouped ext IQ3_S setup_once groups=%d n=%d k32=%d kblocks=%d selected=%d workspace=%zu candidates=%zu user_args=pool init=%s\n",
                    __func__, group_count, n, base_plan.chunk_k, k_blocks, selected, selected_workspace, heuristics.size(),
                    init_once ? "once" : "per_kblock");
        }
        if (init_once) {
            status = grouped.initialize(heuristics[selected].algo, scratch.hipblaslt_workspace, true, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext initialize_once failed: %s groups=%d selected=%d\n",
                            __func__, ggml_cuda_hipblaslt_status_name(status), group_count, selected);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }
        }

        // Some hipBLASLt-ext grouped setup helpers leave HIP's sticky last-error set despite success.
        // Do not attribute that host-side probe state to the following staging kernels.
        (void) cudaGetLastError();

        for (int kb = 0; kb < k_blocks; ++kb) {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                    activations_q8_1, scratch.a_i8, kb, ne_get_rows, base_plan.stride_a_blocks);
            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext IQ3_S stage_a launch failed: %s groups=%d kb=%d grid=%u threads=%d\n",
                            __func__, cudaGetErrorString(cuda_status), group_count, kb, grid_a.x, threads);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }
            const int64_t stage_b_elems = (int64_t) group_count * base_plan.chunk_k * n;
            const int grid_b_x = (int) ((stage_b_elems + stage_b_threads - 1) / stage_b_threads);
            const dim3 grid_b(grid_b_x);
            ggml_cuda_hipblaslt_i8_stage_b_iq3_s_grouped<<<grid_b, stage_b_threads, 0, stream>>>(
                    weights_iq3_s, active_experts_dev, scratch.b_i8, kb, n,
                    base_plan.stride_b_blocks, stride_expert_blocks, group_count);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext IQ3_S stage_b launch failed: %s groups=%d kb=%d elems=%" PRId64 " grid=%d threads=%d\n",
                            __func__, cudaGetErrorString(cuda_status), group_count, kb, stage_b_elems, grid_b_x, stage_b_threads);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }

            if (!init_once) {
                status = grouped.initialize(heuristics[selected].algo, scratch.hipblaslt_workspace, true, stream);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (base_plan.log_heuristic) {
                        GGML_LOG_WARN("%s: grouped ext initialize failed: %s groups=%d kb=%d selected=%d\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
                    }
                    return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
                }
            }

            if (base_plan.log_heuristic) {
                GGML_LOG_WARN("%s: grouped ext IQ3_S route groups=%d n=%d k32=%d kb=%d selected=%d workspace=%zu candidates=%zu user_args=pool\n",
                        __func__, group_count, n, base_plan.chunk_k, kb, selected, selected_workspace, heuristics.size());
            }

            status = grouped.run(user_args_dev, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                if (base_plan.log_heuristic) {
                    GGML_LOG_WARN("%s: grouped ext user-args run failed: %s groups=%d kb=%d selected=%d\n",
                            __func__, ggml_cuda_hipblaslt_status_name(status), group_count, kb, selected);
                }
                return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
            }

            for (int g = 0; g < group_count; ++g) {
                const int expert = active_experts[g];
                const int row_low = expert_bounds_host[expert];
                const int rows = expert_bounds_host[expert + 1] - row_low;
                const dim3 grid_c(((int64_t) rows * n + threads - 1) / threads);
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_s_ids<<<grid_c, threads, 0, stream>>>(
                        scratch.c_i32 + (size_t) row_low * n,
                        activations_q8_1 + (size_t) row_low * base_plan.stride_a_blocks,
                        weights_iq3_s + (size_t) expert * stride_expert_blocks,
                        ids_dst + row_low,
                        dst,
                        kb, rows, n, base_plan.stride_a_blocks, base_plan.stride_b_blocks, base_plan.stride_d);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
    } catch (...) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR;
    }
}

#endif // defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)

} // namespace

const char * ggml_cuda_hipblaslt_i8_status_name(const ggml_cuda_hipblaslt_i8_status status) {
    switch (status) {
        case GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS:          return "success";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_ENABLED:      return "not_enabled";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED:    return "not_supported";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_IMPLEMENTED:  return "not_implemented";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW:    return "size_overflow";
        case GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR:  return "hipblaslt_error";
    }
    return "unknown";
}

const char * ggml_cuda_hipblaslt_i8_route_name(const ggml_cuda_hipblaslt_i8_route_kind route) {
    switch (route) {
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_0_PREFILL:      return "hipblaslt_i8_q8_0_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_PREFILL:      return "hipblaslt_i8_q4_0_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_PREFILL:      return "hipblaslt_i8_q4_1_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_PREFILL:      return "hipblaslt_i8_q5_0_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_PREFILL:      return "hipblaslt_i8_q5_1_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_PREFILL:      return "hipblaslt_i8_q8_K_prefill_probe";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q6_K_PREFILL:      return "hipblaslt_i8_q6_K_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q6_K_MOE_GROUPED:  return "hipblaslt_i8_q6_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_MOE_GROUPED:  return "hipblaslt_i8_q8_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ3_S_MOE_PREFILL: return "hipblaslt_i8_iq3_s_moe_prefill";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_0_MOE_GROUPED:  return "hipblaslt_i8_q8_0_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_MOE_GROUPED:  return "hipblaslt_i8_q4_0_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_MOE_GROUPED:  return "hipblaslt_i8_q4_1_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_MOE_GROUPED:  return "hipblaslt_i8_q5_0_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_MOE_GROUPED:  return "hipblaslt_i8_q5_1_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_K_MOE_GROUPED:  return "hipblaslt_i8_q4_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_K_MOE_GROUPED:  return "hipblaslt_i8_q5_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q2_K_MOE_GROUPED:  return "hipblaslt_i8_q2_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_Q3_K_MOE_GROUPED:  return "hipblaslt_i8_q3_K_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_XS_MOE_GROUPED: return "hipblaslt_i8_iq4_xs_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_NL_MOE_GROUPED: return "hipblaslt_i8_iq4_nl_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ3_XXS_MOE_GROUPED:return "hipblaslt_i8_iq3_xxs_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XXS_MOE_GROUPED:return "hipblaslt_i8_iq2_xxs_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ1_S_MOE_GROUPED:  return "hipblaslt_i8_iq1_s_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XS_MOE_GROUPED: return "hipblaslt_i8_iq2_xs_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_S_MOE_GROUPED:  return "hipblaslt_i8_iq2_s_moe_grouped";
        case GGML_CUDA_HIPBLASLT_I8_ROUTE_DENSE_FFN_GATE_UP_GROUPED: return "hipblaslt_i8_dense_ffn_gate_up_grouped";
    }
    return "hipblaslt_i8_unknown";
}

bool ggml_cuda_hipblaslt_i8_build_enabled() {
#if defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)
    return true;
#else
    return false;
#endif
}

bool ggml_cuda_hipblaslt_i8_supported() {
#if defined(GGML_USE_HIP) && defined(GGML_HIP_HAS_HIPBLASLT)
    static std::once_flag once;
    static bool supported = false;

    std::call_once(once, []() {
        hipblasLtHandle_t handle = nullptr;
        hipblasLtMatmulDesc_t op = nullptr;
        hipblasLtMatrixLayout_t a_desc = nullptr;
        hipblasLtMatrixLayout_t b_desc = nullptr;
        hipblasLtMatrixLayout_t c_desc = nullptr;
        hipblasLtMatrixLayout_t d_desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;

#define GGML_HIPBLASLT_I8_PROBE_TRY(expr) do { \
            if ((expr) != HIPBLAS_STATUS_SUCCESS) { \
                goto cleanup; \
            } \
        } while (0)

        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtCreate(&handle));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_8I, 128, 32, 128));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_8I, 32, 128, 32));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_32I, 128, 128, 128));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32I, 128, 128, 128));
        GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatmulPreferenceCreate(&pref));
        {
            uint64_t workspace_bytes = 0;
            GGML_HIPBLASLT_I8_PROBE_TRY(hipblasLtMatmulPreferenceSetAttribute(
                        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        }
        supported = hipblasLtMatmulAlgoGetHeuristic(
                handle, op, a_desc, b_desc, c_desc, d_desc, pref, 1, &heuristic, &returned) == HIPBLAS_STATUS_SUCCESS &&
            returned > 0 && heuristic.state == HIPBLAS_STATUS_SUCCESS;

cleanup:
        if (pref   != nullptr) { (void) hipblasLtMatmulPreferenceDestroy(pref); }
        if (a_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(a_desc); }
        if (b_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(b_desc); }
        if (c_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(c_desc); }
        if (d_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(d_desc); }
        if (op     != nullptr) { (void) hipblasLtMatmulDescDestroy(op); }
        if (handle != nullptr) { (void) hipblasLtDestroy(handle); }
        if (!supported && ggml_cuda_hipblaslt_i8_diag_enabled()) {
            GGML_LOG_WARN("%s: hipBLASLt I8 runtime probe failed; disabling tuned dense-I8 routes fail-closed\n", __func__);
        }

#undef GGML_HIPBLASLT_I8_PROBE_TRY
    });

    return supported;
#else
    return false;
#endif
}

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_get_scratch_size(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        ggml_cuda_hipblaslt_i8_scratch * scratch_size) {
    if (scratch_size == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    *scratch_size = ggml_cuda_hipblaslt_i8_scratch{};

    const ggml_cuda_hipblaslt_i8_status plan_status = ggml_cuda_hipblaslt_i8_validate_plan_only(plan);
    if (plan_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        return plan_status;
    }

    size_t tmp = 0;
    const size_t a_k = plan.full_k_stage_k32_exact ? (size_t) plan.k : (size_t) plan.chunk_k;
    const size_t b_k = plan.full_k_stage_k32_exact ? (size_t) plan.k : (size_t) plan.chunk_k;
    if (ggml_cuda_hipblaslt_i8_mul_overflows_size((size_t) plan.m, a_k, tmp)) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    scratch_size->a_i8_bytes = tmp * sizeof(int8_t);

    if (ggml_cuda_hipblaslt_i8_mul_overflows_size(b_k, (size_t) plan.n, tmp)) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    scratch_size->b_i8_bytes = tmp * sizeof(int8_t);

    if (ggml_cuda_hipblaslt_i8_mul_overflows_size((size_t) plan.m, (size_t) plan.n, tmp)) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    if (tmp > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_SIZE_OVERFLOW;
    }
    scratch_size->c_i32_bytes = tmp * sizeof(int32_t);
    scratch_size->hipblaslt_workspace_bytes = plan.hipblaslt_workspace_bytes;

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_validate(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args) {
    const ggml_cuda_hipblaslt_i8_status plan_status = ggml_cuda_hipblaslt_i8_validate_plan_only(plan);
    if (plan_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        return plan_status;
    }

    bool has_weight = false;
    switch (plan.weight_type) {
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0: has_weight = args.weights_q8_0 != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0: has_weight = args.weights_q4_0 != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1: has_weight = args.weights_q4_1 != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0: has_weight = args.weights_q5_0 != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1: has_weight = args.weights_q5_1 != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K: has_weight = args.weights_q4_K != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K: has_weight = args.weights_q5_K != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K: has_weight = args.weights_q8_K != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K: has_weight = args.weights_q2_K != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K: has_weight = args.weights_q3_K != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS: has_weight = args.weights_iq4_xs != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL: has_weight = args.weights_iq4_nl != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS: has_weight = args.weights_iq3_xxs != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS: has_weight = args.weights_iq2_xxs != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S: has_weight = args.weights_iq1_s != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS: has_weight = args.weights_iq2_xs != nullptr; break;
        case GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S: has_weight = args.weights_iq2_s != nullptr; break;
    }
    if (args.activations_q8_1 == nullptr || !has_weight || args.dst == nullptr || args.stream == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (args.scratch.a_i8 == nullptr || args.scratch.b_i8 == nullptr || args.scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.hipblaslt_workspace_bytes != 0 && args.scratch.hipblaslt_workspace == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    ggml_cuda_hipblaslt_i8_scratch required{};
    const ggml_cuda_hipblaslt_i8_status scratch_status = ggml_cuda_hipblaslt_i8_get_scratch_size(plan, &required);
    if (scratch_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        return scratch_status;
    }
    if (args.scratch.a_i8_bytes < required.a_i8_bytes ||
            args.scratch.b_i8_bytes < required.b_i8_bytes ||
            args.scratch.c_i32_bytes < required.c_i32_bytes ||
            args.scratch.hipblaslt_workspace_bytes < required.hipblaslt_workspace_bytes) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    return GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;
}

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q8_0(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(plan);
    GGML_UNUSED(args);
    return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_ENABLED;
#else
    const ggml_cuda_hipblaslt_i8_status validation = ggml_cuda_hipblaslt_i8_validate(plan, args);
    if (validation != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        return validation;
    }
    return ggml_cuda_hipblaslt_i8_run_hipblaslt(plan, args);
#endif
}

ggml_cuda_hipblaslt_i8_status ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q6_K(
        const ggml_cuda_hipblaslt_i8_plan & plan,
        const ggml_cuda_hipblaslt_i8_args & args) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(plan);
    GGML_UNUSED(args);
    return GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_ENABLED;
#else
    const ggml_cuda_hipblaslt_i8_status validation = ggml_cuda_hipblaslt_i8_validate_q6_K_plan_only(plan);
    if (validation != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        return validation;
    }
    if (args.activations_q8_1 == nullptr || args.weights_q6_K == nullptr || args.dst == nullptr || args.stream == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (args.scratch.a_i8 == nullptr || args.scratch.b_i8 == nullptr || args.scratch.c_i32 == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }
    if (plan.hipblaslt_workspace_bytes != 0 && args.scratch.hipblaslt_workspace == nullptr) {
        return GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
    }

    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a_desc = nullptr;
    hipblasLtMatrixLayout_t b_desc = nullptr;
    hipblasLtMatrixLayout_t c_desc = nullptr;
    hipblasLtMatrixLayout_t d_desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::array<hipblasLtMatmulHeuristicResult_t, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS> heuristics{};
    int returned = 0;
    int selected_algo_ord = 0;
    ggml_cuda_hipblaslt_i8_status result = GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS;

#define GGML_HIPBLASLT_I8_TRY(expr) do { \
        const hipblasStatus_t hblt_status = (expr); \
        if (hblt_status != HIPBLAS_STATUS_SUCCESS) { \
            GGML_LOG_WARN("%s: hipBLASLt Q6_K call failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(hblt_status)); \
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_HIPBLASLT_ERROR; \
            goto cleanup; \
        } \
    } while (0)

    GGML_HIPBLASLT_I8_TRY(hipblasLtCreate(&handle));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));

    // Q6_K has per-16 K scales, so each staged K32 tile is executed as two K16 I8 GEMMs.
    // A operand is staged B weights [16,N] in the same column-major transposed view as Q8_0.
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_8I,  plan.n, 16, plan.n));
    // B operand is activation rows [M,32], viewed as column-major [16,M] with ld=32 for each half.
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_8I,  16, plan.m, plan.chunk_k));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_32I, plan.n, plan.m, plan.n));
    GGML_HIPBLASLT_I8_TRY(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32I, plan.n, plan.m, plan.n));

    GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceCreate(&pref));
    {
        uint64_t workspace_bytes = args.scratch.hipblaslt_workspace_bytes;
        GGML_HIPBLASLT_I8_TRY(hipblasLtMatmulPreferenceSetAttribute(
                    pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
    }

    {
        const int requested = std::max(1, std::min(plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
        const hipblasStatus_t h = hipblasLtMatmulAlgoGetHeuristic(
                handle, op, a_desc, b_desc, c_desc, d_desc, pref, requested, heuristics.data(), &returned);
        if (h != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
            GGML_LOG_WARN("%s: hipBLASLt Q6_K K16 heuristic failed status=%s returned=%d\n",
                    __func__, ggml_cuda_hipblaslt_status_name(h), returned);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        selected_algo_ord = 0;
        if (plan.preferred_algo_ord >= 0) {
            if (plan.preferred_algo_ord < returned && heuristics[plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS) {
                selected_algo_ord = plan.preferred_algo_ord;
            } else if (plan.log_heuristic) {
                GGML_LOG_WARN("%s: hipBLASLt Q6_K preferred algo_ord=%d unavailable state=%s returned=%d; falling back to first successful heuristic\n",
                        __func__, plan.preferred_algo_ord,
                        plan.preferred_algo_ord < returned ? ggml_cuda_hipblaslt_status_name(heuristics[plan.preferred_algo_ord].state) : "not_returned",
                        returned);
            }
        }
        if (heuristics[selected_algo_ord].state != HIPBLAS_STATUS_SUCCESS) {
            selected_algo_ord = -1;
            for (int i = 0; i < returned; ++i) {
                if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS) {
                    selected_algo_ord = i;
                    break;
                }
            }
        }
        if (selected_algo_ord < 0) {
            GGML_LOG_WARN("%s: hipBLASLt Q6_K K16 heuristic returned no successful algo returned=%d requested=%d\n", __func__, returned, requested);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        if (heuristics[selected_algo_ord].workspaceSize > args.scratch.hipblaslt_workspace_bytes) {
            GGML_LOG_WARN("%s: hipBLASLt Q6_K selected algo workspace too large selected=%d need=%zu have=%zu\n",
                    __func__, selected_algo_ord, heuristics[selected_algo_ord].workspaceSize, args.scratch.hipblaslt_workspace_bytes);
            result = GGML_CUDA_HIPBLASLT_I8_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (plan.log_heuristic) {
            GGML_LOG_WARN("%s: hipBLASLt heuristic route=hipblaslt_i8_q6_K_k32_stage_split16_exact m=%d n=%d k=%d chunk_k=%d requested=%d returned=%d preferred=%d selected=%d workspace=%zu state=%s scratch_a=%zu scratch_b=%zu scratch_c=%zu\n",
                    __func__, plan.m, plan.n, plan.k, plan.chunk_k, requested, returned, plan.preferred_algo_ord, selected_algo_ord,
                    heuristics[selected_algo_ord].workspaceSize,
                    ggml_cuda_hipblaslt_status_name(heuristics[selected_algo_ord].state),
                    args.scratch.a_i8_bytes, args.scratch.b_i8_bytes, args.scratch.c_i32_bytes);
        }
    }

    {
        const int32_t alpha = 1;
        const int32_t beta = 0;
        const int k_blocks = plan.k / plan.chunk_k;
        const int threads = 256;
        const dim3 grid_a((plan.m * plan.chunk_k + threads - 1) / threads);
        const dim3 grid_b((plan.chunk_k * plan.n + threads - 1) / threads);
        const dim3 grid_c((plan.m * plan.n + threads - 1) / threads);

        for (int kb = 0; kb < k_blocks; ++kb) {
            ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, args.stream>>>(
                    args.activations_q8_1, args.scratch.a_i8, kb, plan.m, plan.stride_a_blocks);
            ggml_cuda_hipblaslt_i8_stage_b_q6_K<<<grid_b, threads, 0, args.stream>>>(
                    args.weights_q6_K, args.scratch.b_i8, kb, plan.n, plan.stride_b_blocks);
            CUDA_CHECK(cudaGetLastError());

            for (int half16 = 0; half16 < 2; ++half16) {
                int8_t * a_half = args.scratch.a_i8 + half16 * 16;
                int8_t * b_half = args.scratch.b_i8 + (size_t) half16 * 16 * (size_t) plan.n;

                GGML_HIPBLASLT_I8_TRY(hipblasLtMatmul(
                            handle, op,
                            &alpha,
                            b_half, a_desc,
                            a_half, b_desc,
                            &beta,
                            args.scratch.c_i32, c_desc,
                            args.scratch.c_i32, d_desc,
                            &heuristics[selected_algo_ord].algo,
                            args.scratch.hipblaslt_workspace,
                            args.scratch.hipblaslt_workspace_bytes,
                            args.stream));

                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q6_K<<<grid_c, threads, 0, args.stream>>>(
                        args.scratch.c_i32, args.activations_q8_1, args.weights_q6_K, args.dst,
                        kb, half16, plan.m, plan.n, plan.stride_a_blocks, plan.stride_b_blocks, plan.stride_d);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

cleanup:
    if (pref   != nullptr) { (void) hipblasLtMatmulPreferenceDestroy(pref); }
    if (a_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(a_desc); }
    if (b_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(b_desc); }
    if (c_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(c_desc); }
    if (d_desc != nullptr) { (void) hipblasLtMatrixLayoutDestroy(d_desc); }
    if (op     != nullptr) { (void) hipblasLtMatmulDescDestroy(op); }
    if (handle != nullptr) { (void) hipblasLtDestroy(handle); }
    return result;

#undef GGML_HIPBLASLT_I8_TRY
#endif
}

static bool ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc,
        ggml_type expected_type,
        ggml_cuda_hipblaslt_i8_route_kind route,
        const char * enable_env,
        const char * legacy_enable_env,
        const char * tensor_env,
        const char * legacy_tensor_env,
        const char * min_m_env,
        const char * min_n_env,
        const char * min_k_env,
        int default_min_m,
        int default_min_n,
        int default_min_k) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    GGML_UNUSED(expected_type);
    GGML_UNUSED(route);
    GGML_UNUSED(enable_env);
    GGML_UNUSED(legacy_enable_env);
    GGML_UNUSED(tensor_env);
    GGML_UNUSED(legacy_tensor_env);
    GGML_UNUSED(min_m_env);
    GGML_UNUSED(min_n_env);
    GGML_UNUSED(min_k_env);
    GGML_UNUSED(default_min_m);
    GGML_UNUSED(default_min_n);
    GGML_UNUSED(default_min_k);
    return false;
#else
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: route=%s reject=%s cc=%d src0=%s src1=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_route_name(route), reason, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    const bool enabled = legacy_enable_env != nullptr ?
        ggml_cuda_hipblaslt_i8_prefill_env_enabled(enable_env, legacy_enable_env) :
        ggml_cuda_hipblaslt_i8_env_enabled(enable_env);
    if (!enabled) {
        return reject("env_disabled");
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return reject("null_tensor");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    if (!ggml_cuda_hipblaslt_i8_supported()) {
        return reject("runtime_probe_failed");
    }
    if (src0->type != expected_type || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1]) {
        return reject("shape_mismatch");
    }
    if (src0->ne[0] % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return reject("k_not_multiple_of_32");
    }
    const int min_m = std::max(1, ggml_cuda_hipblaslt_i8_env_int(min_m_env, default_min_m));
    const int min_n = std::max(1, ggml_cuda_hipblaslt_i8_env_int(min_n_env, default_min_n));
    const int min_k = std::max(GGML_CUDA_HIPBLASLT_I8_CHUNK_K, ggml_cuda_hipblaslt_i8_env_int(min_k_env, default_min_k));
    if (src1->ne[1] < min_m || src0->ne[1] < min_n || src0->ne[0] < min_k) {
        return reject("speed_gate");
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return reject("batched_tensor_not_supported");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }

    const char * only = std::getenv(tensor_env);
    if ((only == nullptr || only[0] == '\0') && legacy_tensor_env != nullptr) {
        only = std::getenv(legacy_tensor_env);
    }
    if (only != nullptr && only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
        return reject("tensor_filter");
    }
    if (diag) {
        GGML_LOG_WARN("%s: route=%s accept m=%" PRId64 " n=%" PRId64 " k=%" PRId64 " min_m=%d min_n=%d min_k=%d tensor=%s dst=%s\n",
                __func__, ggml_cuda_hipblaslt_i8_route_name(route), src1->ne[1], src0->ne[1], src0->ne[0],
                min_m, min_n, min_k, src0->name, dst->name);
    }
    return true;
#endif
}

bool ggml_cuda_should_use_hipblaslt_i8_q8_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
            src0, src1, dst, cc, GGML_TYPE_Q8_0, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_0_PREFILL,
            "GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL", "GGML_CUDA_HIPBLASLT_I8_PREFILL",
            "GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_TENSOR", "GGML_CUDA_HIPBLASLT_I8_PREFILL_TENSOR",
            "GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_MIN_M",
            "GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_MIN_N",
            "GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_MIN_K",
            128, 2048, 2048);
}

bool ggml_cuda_should_use_hipblaslt_i8_q4_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
            src0, src1, dst, cc, GGML_TYPE_Q4_0, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_PREFILL,
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL_TENSOR", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL_MIN_M",
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL_MIN_N",
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL_MIN_K",
            128, 2048, 512);
}

bool ggml_cuda_should_use_hipblaslt_i8_q4_1_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
            src0, src1, dst, cc, GGML_TYPE_Q4_1, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_PREFILL,
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL_TENSOR", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL_MIN_M",
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL_MIN_N",
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL_MIN_K",
            128, 2048, 512);
}

bool ggml_cuda_should_use_hipblaslt_i8_q5_0_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
            src0, src1, dst, cc, GGML_TYPE_Q5_0, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_PREFILL,
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL_TENSOR", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL_MIN_M",
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL_MIN_N",
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL_MIN_K",
            128, 2048, 512);
}

bool ggml_cuda_should_use_hipblaslt_i8_q5_1_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_plain_prefill(
            src0, src1, dst, cc, GGML_TYPE_Q5_1, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_PREFILL,
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL_TENSOR", nullptr,
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL_MIN_M",
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL_MIN_N",
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL_MIN_K",
            128, 2048, 512);
}

bool ggml_cuda_should_use_hipblaslt_i8_q8_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    if (!ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_K_PREFILL")) {
        return false;
    }
    if (ggml_cuda_hipblaslt_i8_diag_enabled()) {
        GGML_LOG_WARN("%s: route=%s reject=q8_k_not_implemented_requires_scale_bsum_epilogue cc=%d src0=%s src1=%s dst=%s\n",
                __func__, ggml_cuda_hipblaslt_i8_route_name(GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_PREFILL), cc,
                src0 != nullptr ? src0->name : "<null>",
                src1 != nullptr ? src1->name : "<null>",
                dst  != nullptr ? dst->name  : "<null>");
    }
    return false;
#endif
}

bool ggml_cuda_should_use_hipblaslt_i8_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_q8_0_prefill(src0, src1, dst, cc);
}

bool ggml_cuda_should_use_hipblaslt_i8_q6_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: hipblaslt_i8_q6_K_prefill reject=%s cc=%d src0=%s src1=%s dst=%s\n",
                    __func__, reason, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    static const bool enabled = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q6_K_PREFILL");
    if (!enabled) {
        return reject("env_disabled");
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return reject("null_tensor");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    if (src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (src1->ne[1] <= 8 || src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1]) {
        return reject("shape_mismatch");
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[0] % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return reject("k_not_q6_or_k32_aligned");
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return reject("batched_tensor_not_supported");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }

    if (const char * only = std::getenv("GGML_CUDA_HIPBLASLT_I8_Q6_K_PREFILL_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return reject("tensor_filter");
        }
    }
    if (diag) {
        GGML_LOG_WARN("%s: hipblaslt_i8_q6_K_prefill accept m=%" PRId64 " n=%" PRId64 " k=%" PRId64 " tensor=%s dst=%s\n",
                __func__, src1->ne[1], src0->ne[1], src0->ne[0], src0->name, dst->name);
    }
    return true;
#endif
}

bool ggml_cuda_should_use_hipblaslt_i8_iq3s_moe_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: hipblaslt_i8_iq3s_moe_prefill reject=%s cc=%d src0=%s src1=%s ids=%s dst=%s\n",
                    __func__, reason, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    ids  != nullptr ? ids->name  : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    static const bool enabled = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_PREFILL");
    if (!enabled) {
        return reject("env_disabled");
    }
    static const bool force_mmeid_k32 = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_PREFILL_FORCE_MMEID_K32");
    if (!force_mmeid_k32) {
        return reject("mmeid_k32_force_disabled");
    }
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr) {
        return reject("null_tensor");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    if (src0->type != GGML_TYPE_IQ3_S || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) {
        return reject("shape_mismatch");
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[0] % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return reject("k_not_supported");
    }
    if (src0->ne[2] <= 0 || ids->ne[1] != src1->ne[2]) {
        return reject("ids_shape_mismatch");
    }
    if (src1->ne[2] <= 8 && !ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_PREFILL_ALLOW_SMALL")) {
        return reject("small_batch_disabled");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src0->ne[2] > std::numeric_limits<int>::max() || src1->ne[2] > std::numeric_limits<int>::max() ||
            ids->ne[0] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }
    if (const char * only = std::getenv("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return reject("tensor_filter");
        }
    }
    if (diag) {
        GGML_LOG_WARN("%s: hipblaslt_i8_iq3s_moe_prefill accept tokens=%" PRId64 " used=%" PRId64 " experts=%" PRId64 " n=%" PRId64 " k=%" PRId64 " tensor=%s dst=%s\n",
                __func__, src1->ne[2], ids->ne[0], src0->ne[2], src0->ne[1], src0->ne[0], src0->name, dst->name);
    }
    return true;
#endif
}

bool ggml_cuda_should_use_hipblaslt_i8_q8_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: hipblaslt_i8_q8_0_moe_grouped reject=%s cc=%d src0=%s src1=%s ids=%s dst=%s\n",
                    __func__, reason, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    ids  != nullptr ? ids->name  : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    static const bool enabled = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED");
    if (!enabled) {
        return reject("env_disabled");
    }
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr) {
        return reject("null_tensor");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) {
        return reject("shape_mismatch");
    }
    if (src0->ne[0] % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return reject("k_not_supported");
    }
    if (src0->ne[2] <= 0 || ids->ne[1] != src1->ne[2]) {
        return reject("ids_shape_mismatch");
    }
    if (src1->ne[2] <= 8 && !ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_ALLOW_SMALL")) {
        return reject("small_batch_disabled");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src0->ne[2] > std::numeric_limits<int>::max() || src1->ne[2] > std::numeric_limits<int>::max() ||
            ids->ne[0] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }
    if (const char * only = std::getenv("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return reject("tensor_filter");
        }
    }
    if (diag) {
        GGML_LOG_WARN("%s: hipblaslt_i8_q8_0_moe_grouped accept tokens=%" PRId64 " used=%" PRId64 " experts=%" PRId64 " n=%" PRId64 " k=%" PRId64 " tensor=%s dst=%s\n",
                __func__, src1->ne[2], ids->ne[0], src0->ne[2], src0->ne[1], src0->ne[0], src0->name, dst->name);
    }
    return true;
#endif
}

static bool ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc,
        ggml_type weight_type,
        ggml_cuda_hipblaslt_i8_route_kind route,
        const char * enable_env,
        const char * allow_small_env,
        const char * tensor_filter_env,
        int k_multiple) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    GGML_UNUSED(weight_type);
    GGML_UNUSED(route);
    GGML_UNUSED(enable_env);
    GGML_UNUSED(allow_small_env);
    GGML_UNUSED(tensor_filter_env);
    GGML_UNUSED(k_multiple);
    return false;
#else
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: %s reject=%s cc=%d src0=%s src1=%s ids=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_route_name(route), reason, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    ids  != nullptr ? ids->name  : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    if (!ggml_cuda_hipblaslt_i8_env_enabled(enable_env)) {
        return reject("env_disabled");
    }
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr) {
        return reject("null_tensor");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    if (src0->type != weight_type || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) {
        return reject("shape_mismatch");
    }
    if (src0->ne[0] % k_multiple != 0) {
        return reject("k_not_supported");
    }
    if (src0->ne[2] <= 0 || ids->ne[1] != src1->ne[2]) {
        return reject("ids_shape_mismatch");
    }
    if (src1->ne[2] <= 8 && !ggml_cuda_hipblaslt_i8_env_enabled(allow_small_env)) {
        return reject("small_batch_disabled");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src0->ne[2] > std::numeric_limits<int>::max() || src1->ne[2] > std::numeric_limits<int>::max() ||
            ids->ne[0] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }
    if (const char * only = std::getenv(tensor_filter_env)) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return reject("tensor_filter");
        }
    }
    if (diag) {
        GGML_LOG_WARN("%s: %s accept tokens=%" PRId64 " used=%" PRId64 " experts=%" PRId64 " n=%" PRId64 " k=%" PRId64 " tensor=%s dst=%s\n",
                __func__, ggml_cuda_hipblaslt_i8_route_name(route), src1->ne[2], ids->ne[0], src0->ne[2], src0->ne[1], src0->ne[0], src0->name, dst->name);
    }
    return true;
#endif
}

bool ggml_cuda_should_use_hipblaslt_i8_q6_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q6_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q6_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q6_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q6_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q6_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q8_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q8_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q8_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q8_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q8_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q8_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q4_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q4_0, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_0_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q4_0_MOE_GROUPED_TENSOR",
            GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q4_1_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q4_1, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_1_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q4_1_MOE_GROUPED_TENSOR",
            GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q5_0_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q5_0, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_0_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q5_0_MOE_GROUPED_TENSOR",
            GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q5_1_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q5_1, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_1_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q5_1_MOE_GROUPED_TENSOR",
            GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q4_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q4_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q4_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q4_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q4_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q4_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q5_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q5_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q5_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q5_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q5_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q5_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q2_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q2_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q2_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q2_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q2_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q2_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_q3_K_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_Q3_K, GGML_CUDA_HIPBLASLT_I8_ROUTE_Q3_K_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_Q3_K_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_Q3_K_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_Q3_K_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq4_xs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    if (ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED") &&
            !ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED_FORCE")) {
        if (ggml_cuda_hipblaslt_i8_diag_enabled()) {
            GGML_LOG_WARN("%s: hipblaslt_i8_iq4_xs_moe_grouped reject=force_disabled cc=%d src0=%s src1=%s ids=%s dst=%s\n",
                    __func__, cc,
                    src0 != nullptr ? src0->name : "<null>",
                    src1 != nullptr ? src1->name : "<null>",
                    ids  != nullptr ? ids->name  : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    }
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ4_XS, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_XS_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq4_nl_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ4_NL, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ4_NL_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ4_NL_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ4_NL_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ4_NL_MOE_GROUPED_TENSOR",
            GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq3_xxs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ3_XXS, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ3_XXS_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ3_XXS_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ3_XXS_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ3_XXS_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq2_xxs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ2_XXS, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XXS_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XXS_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XXS_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XXS_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq1_s_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ1_S, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ1_S_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq2_xs_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ2_XS, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_XS_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XS_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XS_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_XS_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_should_use_hipblaslt_i8_iq2_s_moe_grouped(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
    return ggml_cuda_should_use_hipblaslt_i8_k32_moe_grouped(
            src0, src1, ids, dst, cc, GGML_TYPE_IQ2_S, GGML_CUDA_HIPBLASLT_I8_ROUTE_IQ2_S_MOE_GROUPED,
            "GGML_CUDA_HIPBLASLT_I8_IQ2_S_MOE_GROUPED",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_S_MOE_GROUPED_ALLOW_SMALL",
            "GGML_CUDA_HIPBLASLT_I8_IQ2_S_MOE_GROUPED_TENSOR",
            QK_K);
}

bool ggml_cuda_mul_mat_hipblaslt_i8_dense_gate_up(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(ctx);
    GGML_UNUSED(up);
    GGML_UNUSED(gate);
    GGML_UNUSED(dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool diag = ggml_cuda_hipblaslt_i8_diag_enabled();
    auto reject = [&](const char * reason) {
        if (diag) {
            GGML_LOG_WARN("%s: route=%s reject=%s cc=%d up=%s gate=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_route_name(GGML_CUDA_HIPBLASLT_I8_ROUTE_DENSE_FFN_GATE_UP_GROUPED), reason, cc,
                    up   != nullptr ? up->name   : "<null>",
                    gate != nullptr ? gate->name : "<null>",
                    dst  != nullptr ? dst->name  : "<null>");
        }
        return false;
    };

    auto dense_supported_type = [](ggml_type type) {
        switch (type) {
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_K:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                return true;
            default:
                return false;
        }
    };
    auto dense_split_k16_type = [](ggml_type type) {
        return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ||
               type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
    };
    auto dense_stage_pair = [](ggml_type type, const void * up_data, const void * gate_data, int8_t * b_i8,
            int kb, int n, int stride_b_blocks, dim3 grid, int threads, cudaStream_t stream) {
        switch (type) {
            case GGML_TYPE_Q1_0:
                ggml_cuda_hipblaslt_i8_stage_b_q1_0_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q1_0 *) up_data, (const block_q1_0 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q8_0:
                ggml_cuda_hipblaslt_i8_stage_b_q8_0_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q8_0 *) up_data, (const block_q8_0 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q4_0:
                ggml_cuda_hipblaslt_i8_stage_b_q4_0_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q4_0 *) up_data, (const block_q4_0 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q4_1:
                ggml_cuda_hipblaslt_i8_stage_b_q4_1_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q4_1 *) up_data, (const block_q4_1 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q5_0:
                ggml_cuda_hipblaslt_i8_stage_b_q5_0_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q5_0 *) up_data, (const block_q5_0 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q5_1:
                ggml_cuda_hipblaslt_i8_stage_b_q5_1_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q5_1 *) up_data, (const block_q5_1 *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q2_K:
                ggml_cuda_hipblaslt_i8_stage_b_q2_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q2_K *) up_data, (const block_q2_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q3_K:
                ggml_cuda_hipblaslt_i8_stage_b_q3_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q3_K *) up_data, (const block_q3_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q4_K:
                ggml_cuda_hipblaslt_i8_stage_b_q4_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q4_K *) up_data, (const block_q4_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q5_K:
                ggml_cuda_hipblaslt_i8_stage_b_q5_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q5_K *) up_data, (const block_q5_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q6_K:
                ggml_cuda_hipblaslt_i8_stage_b_q6_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q6_K *) up_data, (const block_q6_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_Q8_K:
                ggml_cuda_hipblaslt_i8_stage_b_q8_K_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_q8_K *) up_data, (const block_q8_K *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ2_XXS:
                ggml_cuda_hipblaslt_i8_stage_b_iq2_xxs_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq2_xxs *) up_data, (const block_iq2_xxs *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ2_XS:
                ggml_cuda_hipblaslt_i8_stage_b_iq2_xs_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq2_xs *) up_data, (const block_iq2_xs *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ2_S:
                ggml_cuda_hipblaslt_i8_stage_b_iq2_s_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq2_s *) up_data, (const block_iq2_s *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ3_XXS:
                ggml_cuda_hipblaslt_i8_stage_b_iq3_xxs_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq3_xxs *) up_data, (const block_iq3_xxs *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ1_S:
                ggml_cuda_hipblaslt_i8_stage_b_iq1_s_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq1_s *) up_data, (const block_iq1_s *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ3_S:
                ggml_cuda_hipblaslt_i8_stage_b_iq3_s_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq3_s *) up_data, (const block_iq3_s *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ4_NL:
                ggml_cuda_hipblaslt_i8_stage_b_iq4_nl_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq4_nl *) up_data, (const block_iq4_nl *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            case GGML_TYPE_IQ4_XS:
                ggml_cuda_hipblaslt_i8_stage_b_iq4_xs_dense_pair<<<grid, threads, 0, stream>>>(
                        (const block_iq4_xs *) up_data, (const block_iq4_xs *) gate_data, b_i8, kb, n, stride_b_blocks);
                break;
            default:
                break;
        }
    };
    auto dense_scale_accum = [](ggml_type type, const int32_t * c_i32, const block_q8_1 * a, const float * a_k16_sums,
            const void * b, float * out, int kb, int half16, int m, int n, int stride_a_blocks, int stride_b_blocks,
            dim3 grid, int threads, cudaStream_t stream) {
        switch (type) {
            case GGML_TYPE_Q1_0:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q1_0<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q1_0 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q8_0:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_0<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q8_0 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q4_0:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_0<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q4_0 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q4_1:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_1<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q4_1 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q5_0:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_0<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q5_0 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q5_1:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_1<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q5_1 *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q2_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q2_K_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, a_k16_sums, (const block_q2_K *) b, nullptr, out, kb, half16, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q3_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q3_K_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q3_K *) b, nullptr, out, kb, half16, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q4_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q4_K<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q4_K *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q5_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q5_K<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q5_K *) b, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q6_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q6_K<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q6_K *) b, out, kb, half16, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_Q8_K:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_q8_K_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_q8_K *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ2_XXS:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xxs_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq2_xxs *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ2_XS:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_xs_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq2_xs *) b, nullptr, out, kb, half16, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ2_S:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq2_s_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq2_s *) b, nullptr, out, kb, half16, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ3_XXS:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_xxs_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq3_xxs *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ1_S:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq1_s_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq1_s *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ3_S:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq3_s_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq3_s *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ4_NL:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_nl_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq4_nl *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            case GGML_TYPE_IQ4_XS:
                ggml_cuda_hipblaslt_i8_scale_accum_q8_1_iq4_xs_ids<<<grid, threads, 0, stream>>>(
                        c_i32, a, (const block_iq4_xs *) b, nullptr, out, kb, m, n, stride_a_blocks, stride_b_blocks, n);
                break;
            default:
                break;
        }
    };

    if (!ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED")) {
        return reject("env_disabled");
    }
    if (up == nullptr || gate == nullptr || dst == nullptr || up->src[0] == nullptr || up->src[1] == nullptr || gate->src[0] == nullptr || gate->src[1] == nullptr) {
        return reject("null_tensor");
    }
    if (up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT || dst->op != GGML_OP_GLU) {
        return reject("unsupported_op");
    }
    if (ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU) {
        return reject("unsupported_glu_op");
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return reject("unsupported_arch");
    }
    const ggml_tensor * up_w = up->src[0];
    const ggml_tensor * gate_w = gate->src[0];
    const ggml_tensor * src1 = up->src[1];
    if (gate->src[1] != src1) {
        return reject("different_activation");
    }
    if (up_w->type != gate_w->type || !dense_supported_type(up_w->type)) {
        return reject("unsupported_weight_type");
    }
    if (src1->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return reject("unsupported_type");
    }
    if (up_w->ne[0] != gate_w->ne[0] || up_w->ne[1] != gate_w->ne[1] || src1->ne[0] != up_w->ne[0] ||
            up->ne[0] != up_w->ne[1] || gate->ne[0] != gate_w->ne[1] || dst->ne[0] != up_w->ne[1] ||
            up->ne[1] != src1->ne[1] || gate->ne[1] != src1->ne[1] || dst->ne[1] != src1->ne[1]) {
        return reject("shape_mismatch");
    }
    if (up_w->ne[2] != 1 || up_w->ne[3] != 1 || gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return reject("batched_tensor_not_supported");
    }
    if (!ggml_is_contiguous(up_w) || !ggml_is_contiguous(gate_w) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return reject("non_contiguous");
    }
    if (ggml_is_transposed(up_w) || ggml_is_transposed(gate_w) || ggml_is_transposed(src1)) {
        return reject("transposed_tensor");
    }
    if (up_w->ne[0] > std::numeric_limits<int>::max() || up_w->ne[1] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max()) {
        return reject("dimension_overflow");
    }

    const int m = (int) src1->ne[1];
    const int n = (int) up_w->ne[1];
    const int k = (int) up_w->ne[0];
    const int weight_block = (int) ggml_blck_size(up_w->type);
    if (weight_block <= 0 || k % weight_block != 0 || k % GGML_CUDA_HIPBLASLT_I8_CHUNK_K != 0) {
        return reject("k_not_supported");
    }
    const int k32_blocks = k / GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
    const int stride_b_blocks = k / weight_block;
    const bool split_k16 = dense_split_k16_type(up_w->type);
    const bool allow_small = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_ALLOW_SMALL");
    const int min_m = allow_small ? 1 : std::max(1, ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_MIN_M", 128));
    const int min_n = allow_small ? 1 : std::max(1, ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_MIN_N", 4096));
    const int min_k = allow_small ? GGML_CUDA_HIPBLASLT_I8_CHUNK_K : std::max(GGML_CUDA_HIPBLASLT_I8_CHUNK_K, ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_MIN_K", 2048));
    if (m < min_m || n < min_n || k < min_k) {
        return reject("speed_gate");
    }
    if (const char * only = std::getenv("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_TENSOR")) {
        if (only[0] != '\0' && std::strstr(up_w->name, only) == nullptr && std::strstr(gate_w->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return reject("tensor_filter");
        }
    }

    ggml_cuda_hipblaslt_i8_plan plan{};
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
    plan.stride_a_blocks = k32_blocks;
    plan.stride_b_blocks = stride_b_blocks;
    plan.stride_d = (int) (dst->nb[1] / sizeof(float));
    plan.heuristic_request_count = 1;
    if (const char * env_ws = std::getenv("GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")) {
        const long mb = std::strtol(env_ws, nullptr, 10);
        if (mb >= 0) {
            plan.hipblaslt_workspace_bytes = (size_t) mb << 20;
        }
    }
    const int tuned_algo_ord = ggml_cuda_hipblaslt_i8_lookup_dense_gate_up_grouped_algo_ord(n, m, split_k16 ? 16 : GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    if (tuned_algo_ord >= 0 && tuned_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
        plan.preferred_algo_ord = tuned_algo_ord;
        plan.heuristic_request_count = tuned_algo_ord + 1;
    }
    const int forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_DENSE_FFN_GATE_UP_GROUPED_ALGO_ORD", -1);
    if (forced_algo_ord >= 0 && forced_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
        plan.preferred_algo_ord = forced_algo_ord;
        plan.heuristic_request_count = forced_algo_ord + 1;
    }
    plan.log_heuristic = diag;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const size_t mn = (size_t) m * (size_t) n;
    ggml_cuda_pool_alloc<block_q8_1> activations_q8_1(pool, (size_t) m * (size_t) k32_blocks);
    ggml_cuda_pool_alloc<float> activations_k16_sums;
    float * activations_k16_sums_ptr = nullptr;
    if (up_w->type == GGML_TYPE_Q2_K) {
        activations_k16_sums_ptr = activations_k16_sums.alloc(pool, (size_t) m * (size_t) k32_blocks * 2);
        const int threads = 256;
        const dim3 grid((k + threads - 1) / threads, m, 1);
        ggml_cuda_hipblaslt_i8_quantize_q8_1_ids<<<grid, threads, 0, stream>>>(
                (const float *) src1->data, nullptr, activations_q8_1.get(), activations_k16_sums_ptr,
                k, src1->nb[1] / sizeof(float), m);
    } else {
        quantize_row_q8_1_cuda(
                (const float *) src1->data, nullptr, activations_q8_1.get(), up_w->type, k,
                src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                k, m, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int8_t>  a_i8(pool, (size_t) m * (size_t) GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    ggml_cuda_pool_alloc<int8_t>  b_i8(pool, 2ull * (size_t) n * (size_t) GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
    ggml_cuda_pool_alloc<int32_t> c_i32(pool, 2ull * mn);
    ggml_cuda_pool_alloc<float>   up_tmp(pool, mn);
    ggml_cuda_pool_alloc<float>   gate_tmp(pool, mn);
    ggml_cuda_pool_alloc<char> workspace;
    void * workspace_ptr = nullptr;
    if (plan.hipblaslt_workspace_bytes != 0) {
        workspace_ptr = workspace.alloc(pool, plan.hipblaslt_workspace_bytes);
    }

    hipblasLtHandle_t handle = nullptr;
    const hipblasStatus_t create_status = hipblasLtCreate(&handle);
    if (create_status != HIPBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt create failed: %s\n", __func__, ggml_cuda_hipblaslt_status_name(create_status));
        return false;
    }

    const int threads = 256;
    const dim3 grid_a(((int64_t) m * GGML_CUDA_HIPBLASLT_I8_CHUNK_K + threads - 1) / threads);
    const dim3 grid_b((2ll * GGML_CUDA_HIPBLASLT_I8_CHUNK_K * n + threads - 1) / threads);
    const dim3 grid_c(((int64_t) m * n + threads - 1) / threads);
    bool ok = true;

    for (int kb = 0; ok && kb < k32_blocks; ++kb) {
        ggml_cuda_hipblaslt_i8_stage_a_q8_1<<<grid_a, threads, 0, stream>>>(
                activations_q8_1.get(), a_i8.get(), kb, m, k32_blocks);
        dense_stage_pair(up_w->type, up_w->data, gate_w->data, b_i8.get(), kb, n, stride_b_blocks, grid_b, threads, stream);
        CUDA_CHECK(cudaGetLastError());

        const int half_count = split_k16 ? 2 : 1;
        const int gemm_k = split_k16 ? 16 : GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        for (int half16 = 0; ok && half16 < half_count; ++half16) {
            try {
                constexpr int group_count = 2;
                std::vector<int64_t> gm(group_count, n);
                std::vector<int64_t> gn(group_count, m);
                std::vector<int64_t> gk(group_count, gemm_k);
                std::vector<int64_t> batch_count(group_count, 1);
                std::vector<int64_t> lda(group_count, n);
                std::vector<int64_t> ldb(group_count, GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
                std::vector<int64_t> ldc(group_count, n);
                std::vector<int64_t> ldd(group_count, n);
                std::vector<int64_t> stride_a(group_count, 0);
                std::vector<int64_t> stride_b(group_count, 0);
                std::vector<int64_t> stride_c(group_count, 0);
                std::vector<int64_t> stride_d(group_count, 0);
                std::vector<hipblaslt_ext::GemmEpilogue> epilogue(group_count);
                std::vector<hipblaslt_ext::GemmInputs> inputs(group_count);
                std::vector<int32_t> alpha(group_count, 1);
                std::vector<int32_t> beta(group_count, 0);

                const size_t half_b_offset = (size_t) half16 * 16ull * (size_t) n;
                const size_t half_a_offset = (size_t) half16 * 16ull;
                inputs[0].setA(b_i8.get() + half_b_offset);
                inputs[0].setB(a_i8.get() + half_a_offset);
                inputs[0].setC(c_i32.get());
                inputs[0].setD(c_i32.get());
                inputs[0].setAlpha(&alpha[0]);
                inputs[0].setBeta(&beta[0]);
                inputs[1].setA(b_i8.get() + (size_t) GGML_CUDA_HIPBLASLT_I8_CHUNK_K * (size_t) n + half_b_offset);
                inputs[1].setB(a_i8.get() + half_a_offset);
                inputs[1].setC(c_i32.get() + mn);
                inputs[1].setD(c_i32.get() + mn);
                inputs[1].setAlpha(&alpha[1]);
                inputs[1].setBeta(&beta[1]);

                hipblaslt_ext::GroupedGemm grouped(
                        handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
                hipblaslt_ext::GemmProblemType problem_type(
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I);
                hipblasStatus_t status = grouped.setProblem(
                        gm, gn, gk, batch_count,
                        lda, ldb, ldc, ldd,
                        stride_a, stride_b, stride_c, stride_d,
                        epilogue, inputs, problem_type);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (diag) {
                        GGML_LOG_WARN("%s: dense grouped setProblem failed: %s type=%s m=%d n=%d kcore=%d kb=%d half=%d\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), ggml_type_name(up_w->type), m, n, gemm_k, kb, half16);
                    }
                    ok = false;
                    break;
                }

                hipblaslt_ext::GemmPreference pref;
                pref.setMaxWorkspaceBytes(plan.hipblaslt_workspace_bytes);
                std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
                const int requested = std::max(1, std::min(plan.heuristic_request_count, GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS));
                status = grouped.algoGetHeuristic(requested, pref, heuristics);
                if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
                    heuristics.clear();
                    status = hipblaslt_ext::getAllAlgos(
                            handle,
                            hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                            HIPBLAS_OP_N, HIPBLAS_OP_N,
                            HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I, HIPBLAS_COMPUTE_32I,
                            heuristics);
                }
                if (status != HIPBLAS_STATUS_SUCCESS || heuristics.empty()) {
                    if (diag) {
                        GGML_LOG_WARN("%s: dense grouped getAllAlgos failed/empty: %s type=%s m=%d n=%d kb=%d half=%d returned=%zu\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), ggml_type_name(up_w->type), m, n, kb, half16, heuristics.size());
                    }
                    ok = false;
                    break;
                }

                int selected = -1;
                size_t selected_workspace = 0;
                if (plan.preferred_algo_ord >= 0 && plan.preferred_algo_ord < (int) heuristics.size()) {
                    size_t need = 0;
                    hipblasLtMatmulAlgo_t algo = heuristics[plan.preferred_algo_ord].algo;
                    if (heuristics[plan.preferred_algo_ord].state == HIPBLAS_STATUS_SUCCESS &&
                            grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                            need <= plan.hipblaslt_workspace_bytes) {
                        selected = plan.preferred_algo_ord;
                        selected_workspace = need;
                    }
                }
                for (int hi = 0; selected < 0 && hi < (int) heuristics.size(); ++hi) {
                    size_t need = 0;
                    hipblasLtMatmulAlgo_t algo = heuristics[hi].algo;
                    if (heuristics[hi].state == HIPBLAS_STATUS_SUCCESS &&
                            grouped.isAlgoSupported(algo, need) == HIPBLAS_STATUS_SUCCESS &&
                            need <= plan.hipblaslt_workspace_bytes) {
                        selected = hi;
                        selected_workspace = need;
                    }
                }
                if (selected < 0) {
                    if (diag) {
                        GGML_LOG_WARN("%s: dense grouped no supported algo type=%s m=%d n=%d kb=%d half=%d candidates=%zu\n",
                                __func__, ggml_type_name(up_w->type), m, n, kb, half16, heuristics.size());
                    }
                    ok = false;
                    break;
                }
                grouped.setMaxWorkspaceBytes(selected_workspace);

                std::vector<hipblaslt_ext::UserArguments> user_args(group_count);
                status = grouped.getDefaultValueForDeviceUserArguments(user_args.data());
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
                hipblaslt_ext::UserArguments * user_args_dev = nullptr;
                const size_t user_args_bytes = sizeof(hipblaslt_ext::UserArguments) * (size_t) group_count;
                cudaError_t cuda_status = cudaMalloc((void **) &user_args_dev, user_args_bytes);
                if (cuda_status != cudaSuccess) {
                    ok = false;
                    break;
                }
                cuda_status = cudaMemcpyAsync(user_args_dev, user_args.data(), user_args_bytes, cudaMemcpyHostToDevice, stream);
                if (cuda_status != cudaSuccess) {
                    (void) cudaFree(user_args_dev);
                    ok = false;
                    break;
                }
                status = grouped.initialize(heuristics[selected].algo, workspace_ptr, true, stream);
                if (status == HIPBLAS_STATUS_SUCCESS) {
                    if (diag) {
                        GGML_LOG_WARN("%s: dense grouped ext %s route groups=2 n=%d rows=%d k%d=%d kb=%d half=%d selected=%d workspace=%zu candidates=%zu user_args=1\n",
                                __func__, ggml_type_name(up_w->type), n, m, gemm_k, gemm_k, kb, half16, selected, selected_workspace, heuristics.size());
                    }
                    status = grouped.run(user_args_dev, stream);
                }
                (void) cudaFree(user_args_dev);
                if (status != HIPBLAS_STATUS_SUCCESS) {
                    if (diag) {
                        GGML_LOG_WARN("%s: dense grouped run failed: %s type=%s m=%d n=%d kb=%d half=%d selected=%d\n",
                                __func__, ggml_cuda_hipblaslt_status_name(status), ggml_type_name(up_w->type), m, n, kb, half16, selected);
                    }
                    ok = false;
                    break;
                }
            } catch (...) {
                ok = false;
                break;
            }

            dense_scale_accum(up_w->type, c_i32.get(), activations_q8_1.get(), activations_k16_sums_ptr, up_w->data, up_tmp.get(),
                    kb, half16, m, n, k32_blocks, stride_b_blocks, grid_c, threads, stream);
            dense_scale_accum(up_w->type, c_i32.get() + mn, activations_q8_1.get(), activations_k16_sums_ptr, gate_w->data, gate_tmp.get(),
                    kb, half16, m, n, k32_blocks, stride_b_blocks, grid_c, threads, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    (void) hipblasLtDestroy(handle);
    if (!ok) {
        return false;
    }

    ggml_cuda_hipblaslt_i8_dense_gate_up_swiglu<<<grid_c, threads, 0, stream>>>(
            up_tmp.get(), gate_tmp.get(), (float *) dst->data, m, n, plan.stride_d);
    CUDA_CHECK(cudaGetLastError());

    return true;
#endif
}

bool ggml_cuda_mul_mat_hipblaslt_i8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) || !defined(GGML_HIP_HAS_HIPBLASLT)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#else
    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_mtp_mmq_stream_is_capturing(stream)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (ids != nullptr) {
        const bool use_iq3_s_moe = ggml_cuda_should_use_hipblaslt_i8_iq3s_moe_prefill(src0, src1, ids, dst, cc);
        const bool iq3_s_moe_grouped_env = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_GROUPED");
        const bool iq3_s_moe_grouped_force = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_GROUPED_FORCE");
        const bool use_iq3_s_moe_grouped = !use_iq3_s_moe && iq3_s_moe_grouped_env && iq3_s_moe_grouped_force &&
            src0 != nullptr && src1 != nullptr && ids != nullptr && dst != nullptr &&
            GGML_CUDA_CC_IS_AMD(cc) && cc >= GGML_CUDA_CC_RDNA3 &&
            src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32;
        if (ggml_cuda_hipblaslt_i8_diag_enabled() && iq3_s_moe_grouped_env) {
            GGML_LOG_WARN("%s: hipblaslt_i8_iq3s_moe_grouped %s cc=%d src0_type=%s src1_type=%s ids_type=%s dst_type=%s\n",
                    __func__, iq3_s_moe_grouped_force ? (use_iq3_s_moe_grouped ? "accept" : "reject=predicate") : "reject=force_disabled",
                    cc,
                    src0 != nullptr ? ggml_type_name(src0->type) : "<null>",
                    src1 != nullptr ? ggml_type_name(src1->type) : "<null>",
                    ids  != nullptr ? ggml_type_name(ids->type)  : "<null>",
                    dst  != nullptr ? ggml_type_name(dst->type)  : "<null>");
        }
        const bool use_iq4_xs_moe_grouped = !use_iq3_s_moe && ggml_cuda_should_use_hipblaslt_i8_iq4_xs_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq4_nl_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq4_nl_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq3_xxs_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq3_xxs_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq2_xxs_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq2_xxs_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq1_s_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq1_s_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq2_xs_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq2_xs_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_iq2_s_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_iq2_s_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q8_0_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q8_0_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q6_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q6_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q8_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q8_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q2_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q2_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q3_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q3_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q4_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q4_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q5_K_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && !use_q4_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q5_K_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q4_0_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && !use_q4_K_moe_grouped && !use_q5_K_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q4_0_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q4_1_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && !use_q4_K_moe_grouped && !use_q5_K_moe_grouped && !use_q4_0_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q4_1_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q5_0_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && !use_q4_K_moe_grouped && !use_q5_K_moe_grouped && !use_q4_0_moe_grouped && !use_q4_1_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q5_0_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_q5_1_moe_grouped = !use_iq3_s_moe && !use_iq4_xs_moe_grouped && !use_iq4_nl_moe_grouped && !use_iq3_xxs_moe_grouped && !use_iq2_xxs_moe_grouped && !use_iq1_s_moe_grouped && !use_iq2_xs_moe_grouped && !use_iq2_s_moe_grouped && !use_q8_0_moe_grouped && !use_q6_K_moe_grouped && !use_q8_K_moe_grouped && !use_q2_K_moe_grouped && !use_q3_K_moe_grouped && !use_q4_K_moe_grouped && !use_q5_K_moe_grouped && !use_q4_0_moe_grouped && !use_q4_1_moe_grouped && !use_q5_0_moe_grouped && ggml_cuda_should_use_hipblaslt_i8_q5_1_moe_grouped(src0, src1, ids, dst, cc);
        const bool use_k16_moe_grouped = use_q2_K_moe_grouped || use_q3_K_moe_grouped || use_iq2_xs_moe_grouped || use_iq2_s_moe_grouped;
        const bool use_k32_moe_grouped = use_iq4_xs_moe_grouped || use_iq4_nl_moe_grouped || use_iq3_xxs_moe_grouped || use_iq2_xxs_moe_grouped || use_iq1_s_moe_grouped || use_q8_0_moe_grouped || use_q8_K_moe_grouped || use_q4_K_moe_grouped || use_q5_K_moe_grouped || use_q4_0_moe_grouped || use_q4_1_moe_grouped || use_q5_0_moe_grouped || use_q5_1_moe_grouped;
        if (!use_iq3_s_moe && !use_iq3_s_moe_grouped && !use_q6_K_moe_grouped && !use_k16_moe_grouped && !use_k32_moe_grouped) {
            return false;
        }

        const int k = (int) src0->ne[0];
        const int n = (int) src0->ne[1];
        const int n_experts = (int) src0->ne[2];
        const int n_tokens = (int) src1->ne[2];
        const int n_expert_used = (int) ids->ne[0];
        const int k32_blocks = k / GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        const int k256_blocks = k / QK_K;
        const int64_t ne_get_rows = (int64_t) n_tokens * n_expert_used;
        const int si1  = (int) (ids->nb[1] / ggml_element_size(ids));
        const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
        const int stride_src1 = (int) (src1->nb[1] / sizeof(float));
        const int stride_b_blocks = (int) (src0->nb[1] / ggml_type_size(src0->type));
        const int stride_expert_blocks = (int) (src0->nb[2] / ggml_type_size(src0->type));
        const int stride_d = (int) (dst->nb[1] / sizeof(float));

        const int weight_k_blocks = (use_q6_K_moe_grouped || use_k16_moe_grouped || use_q8_K_moe_grouped || use_q4_K_moe_grouped || use_q5_K_moe_grouped || use_iq4_xs_moe_grouped || use_iq3_xxs_moe_grouped || use_iq2_xxs_moe_grouped || use_iq1_s_moe_grouped || (!use_k32_moe_grouped && !use_k16_moe_grouped)) ? k256_blocks : k32_blocks;
        if (k32_blocks <= 0 || weight_k_blocks <= 0 || ne_get_rows <= 0 || stride_b_blocks < weight_k_blocks) {
            return false;
        }

        ggml_cuda_pool & pool = ctx.pool();
        ggml_cuda_pool_alloc<int32_t> ids_src1(pool, ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(pool, ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(pool, (size_t) n_experts + 1);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                n_experts, n_tokens, n_expert_used, (int) src1->ne[1], si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int32_t> expert_bounds_host((size_t) n_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(expert_bounds_host.data(), expert_bounds.get(),
                    ((size_t) n_experts + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int max_rows = 0;
        std::vector<int32_t> active_experts;
        active_experts.reserve(n_experts);
        for (int expert = 0; expert < n_experts; ++expert) {
            const int rows = expert_bounds_host[expert + 1] - expert_bounds_host[expert];
            if (rows < 0) {
                return false;
            }
            if (rows > 0) {
                active_experts.push_back(expert);
            }
            max_rows = std::max(max_rows, rows);
        }
        if (max_rows == 0) {
            return true;
        }

        ggml_cuda_pool_alloc<block_q8_1> activations_q8_1(pool, (size_t) ne_get_rows * (size_t) k32_blocks);
        ggml_cuda_pool_alloc<float> activations_k16_sums;
        float * activations_k16_sums_ptr = nullptr;
        if (use_q2_K_moe_grouped) {
            activations_k16_sums_ptr = activations_k16_sums.alloc(pool, (size_t) ne_get_rows * (size_t) k32_blocks * 2);
        }
        {
            const int threads = 256;
            const dim3 grid((k + threads - 1) / threads, ne_get_rows, 1);
            ggml_cuda_hipblaslt_i8_quantize_q8_1_ids<<<grid, threads, 0, stream>>>(
                    (const float *) src1->data, ids_src1.get(), activations_q8_1.get(), activations_k16_sums_ptr, k, stride_src1, (int) ne_get_rows);
            CUDA_CHECK(cudaGetLastError());
        }

        size_t workspace_bytes = 64ull << 20;
        if (const char * env_ws = std::getenv("GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")) {
            const long mb = std::strtol(env_ws, nullptr, 10);
            if (mb >= 0) {
                workspace_bytes = (size_t) mb << 20;
            }
        }
        int forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ3S_MOE_ALGO_ORD", -1);
        if (use_iq4_xs_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ4_XS_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq4_nl_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ4_NL_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq3_xxs_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ3_XXS_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq2_xxs_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ2_XXS_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq1_s_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ1_S_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq2_xs_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ2_XS_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_iq2_s_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_IQ2_S_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q8_0_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q6_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q6_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q8_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q8_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q2_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q2_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q3_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q3_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q4_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q4_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q5_K_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q5_K_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q4_0_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q4_0_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q4_1_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q4_1_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q5_0_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q5_0_MOE_GROUPED_ALGO_ORD", -1);
        } else if (use_q5_1_moe_grouped) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q5_1_MOE_GROUPED_ALGO_ORD", -1);
        }

        const bool grouped_requested = use_q6_K_moe_grouped || use_k16_moe_grouped || use_k32_moe_grouped || use_iq3_s_moe_grouped;
        ggml_cuda_pool_alloc<int32_t> active_experts_dev;
        int32_t * active_experts_dev_ptr = nullptr;
        if (grouped_requested) {
            active_experts_dev_ptr = active_experts_dev.alloc(pool, active_experts.size());
            CUDA_CHECK(cudaMemcpyAsync(active_experts_dev_ptr, active_experts.data(),
                        active_experts.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }

        const size_t a_rows_alloc = grouped_requested ? (size_t) ne_get_rows : (size_t) max_rows;
        const size_t b_groups_alloc = grouped_requested ? active_experts.size() : 1;
        const size_t c_rows_alloc = grouped_requested ? (size_t) ne_get_rows : (size_t) max_rows;
        ggml_cuda_pool_alloc<int8_t>  a_i8(pool, a_rows_alloc * GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
        ggml_cuda_pool_alloc<int8_t>  b_i8(pool, b_groups_alloc * (size_t) n * GGML_CUDA_HIPBLASLT_I8_CHUNK_K);
        ggml_cuda_pool_alloc<int32_t> c_i32(pool, c_rows_alloc * (size_t) n);
        ggml_cuda_pool_alloc<char> workspace;
        void * workspace_ptr = nullptr;
        if (workspace_bytes != 0) {
            workspace_ptr = workspace.alloc(pool, workspace_bytes);
        }
        const size_t grouped_user_args_multiplier = (use_q6_K_moe_grouped || use_k16_moe_grouped) ? 2u : 1u;
        const size_t user_args_bytes = grouped_requested ?
            sizeof(hipblaslt_ext::UserArguments) * active_experts.size() * grouped_user_args_multiplier : 0;
        ggml_cuda_pool_alloc<char> user_args;
        void * user_args_ptr = nullptr;
        if (user_args_bytes != 0) {
            user_args_ptr = user_args.alloc(pool, user_args_bytes);
        }

        ggml_cuda_hipblaslt_i8_scratch scratch{};
        scratch.a_i8 = a_i8.get();
        scratch.b_i8 = b_i8.get();
        scratch.c_i32 = c_i32.get();
        scratch.hipblaslt_workspace = workspace_ptr;
        scratch.hipblaslt_user_args = user_args_ptr;
        scratch.a_i8_bytes = a_rows_alloc * GGML_CUDA_HIPBLASLT_I8_CHUNK_K * sizeof(int8_t);
        scratch.b_i8_bytes = b_groups_alloc * (size_t) n * GGML_CUDA_HIPBLASLT_I8_CHUNK_K * sizeof(int8_t);
        scratch.c_i32_bytes = c_rows_alloc * (size_t) n * sizeof(int32_t);
        scratch.hipblaslt_workspace_bytes = workspace_bytes;
        scratch.hipblaslt_user_args_bytes = user_args_bytes;

        ggml_cuda_hipblaslt_i8_plan plan{};
        plan.n = n;
        plan.k = k;
        plan.chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        plan.stride_a_blocks = k32_blocks;
        plan.stride_b_blocks = stride_b_blocks;
        plan.stride_d = stride_d;
        plan.batch_count = 1;
        plan.heuristic_request_count = 1;
        plan.preferred_algo_ord = -1;
        if (forced_algo_ord >= 0 && forced_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
            plan.preferred_algo_ord = forced_algo_ord;
        } else if (use_q6_K_moe_grouped &&
                ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q6_K_MOE_GROUPED_EXACT_ALGO")) {
            const int grouped_algo_ord = ggml_cuda_hipblaslt_i8_lookup_q6_K_moe_grouped_algo_ord(
                    (int) active_experts.size(), n, max_rows, 16);
            if (grouped_algo_ord >= 0 && grouped_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
                plan.preferred_algo_ord = grouped_algo_ord;
            }
        }
        if (plan.preferred_algo_ord >= 0) {
            plan.heuristic_request_count = plan.preferred_algo_ord + 1;
        }
        plan.log_heuristic = ggml_cuda_hipblaslt_i8_diag_enabled();
        plan.hipblaslt_workspace_bytes = workspace_bytes;

        const block_iq3_s * weights = (const block_iq3_s *) src0->data;
        float * dst_data = (float *) dst->data;
        if (use_q6_K_moe_grouped) {
            const ggml_cuda_hipblaslt_i8_status grouped_status = ggml_cuda_hipblaslt_i8_run_hipblaslt_q6_K_grouped(
                    plan,
                    activations_q8_1.get(),
                    (const block_q6_K *) src0->data,
                    ids_dst.get(),
                    active_experts,
                    active_experts_dev_ptr,
                    expert_bounds_host,
                    dst_data,
                    scratch,
                    stream,
                    stride_expert_blocks);
            if (grouped_status == GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return true;
            }
            GGML_LOG_WARN("%s: grouped Q6_K MoE hipBLASLt-I8 failed status=%s; falling back to standard MUL_MAT_ID route tensor=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_status_name(grouped_status), src0->name, dst->name);
            return false;
        }
        if (use_k16_moe_grouped) {
            const ggml_cuda_hipblaslt_i8_weight_type weight_type = use_q2_K_moe_grouped ?
                GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q2_K : (use_q3_K_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q3_K :
                        (use_iq2_xs_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XS : GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_S));
            const char * route_label = use_q2_K_moe_grouped ? "Q2_K" : (use_q3_K_moe_grouped ? "Q3_K" :
                    (use_iq2_xs_moe_grouped ? "IQ2_XS" : "IQ2_S"));
            const ggml_cuda_hipblaslt_i8_status grouped_status = ggml_cuda_hipblaslt_i8_run_hipblaslt_k16_grouped(
                    plan,
                    activations_q8_1.get(),
                    activations_k16_sums_ptr,
                    src0->data,
                    weight_type,
                    route_label,
                    ids_dst.get(),
                    active_experts,
                    active_experts_dev_ptr,
                    expert_bounds_host,
                    dst_data,
                    scratch,
                    stream,
                    stride_expert_blocks);
            if (grouped_status == GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return true;
            }
            GGML_LOG_WARN("%s: grouped %s MoE hipBLASLt-I8 failed status=%s; falling back to standard MUL_MAT_ID route tensor=%s dst=%s\n",
                    __func__, route_label, ggml_cuda_hipblaslt_i8_status_name(grouped_status), src0->name, dst->name);
            return false;
        }
        if (use_k32_moe_grouped) {
            const void * weights_k32 = src0->data;
            const ggml_cuda_hipblaslt_i8_weight_type weight_type = use_iq4_xs_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_XS :
                (use_iq4_nl_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ4_NL :
                 (use_iq3_xxs_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ3_XXS :
                  (use_iq2_xxs_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ2_XXS :
                   (use_iq1_s_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_IQ1_S :
                   (use_q8_K_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_K :
                    (use_q4_K_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_K :
                     (use_q5_K_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_K :
                      (use_q4_0_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0 :
                      (use_q4_1_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1 :
                       (use_q5_0_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0 :
                         (use_q5_1_moe_grouped ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1 : GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0)))))))))));
            const char * route_label = "Q8_0";
            if (use_iq4_xs_moe_grouped) {
                route_label = "IQ4_XS";
            } else if (use_iq4_nl_moe_grouped) {
                route_label = "IQ4_NL";
            } else if (use_iq3_xxs_moe_grouped) {
                route_label = "IQ3_XXS";
            } else if (use_iq2_xxs_moe_grouped) {
                route_label = "IQ2_XXS";
            } else if (use_iq1_s_moe_grouped) {
                route_label = "IQ1_S";
            } else if (use_q8_K_moe_grouped) {
                route_label = "Q8_K";
            } else if (use_q4_K_moe_grouped) {
                route_label = "Q4_K";
            } else if (use_q5_K_moe_grouped) {
                route_label = "Q5_K";
            } else if (use_q4_0_moe_grouped) {
                route_label = "Q4_0";
            } else if (use_q4_1_moe_grouped) {
                route_label = "Q4_1";
            } else if (use_q5_0_moe_grouped) {
                route_label = "Q5_0";
            } else if (use_q5_1_moe_grouped) {
                route_label = "Q5_1";
            }
            if (weight_type == GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0 &&
                    ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_MOE_GROUPED_DIRECT_QS")) {
                const ggml_cuda_hipblaslt_i8_status direct_status = ggml_cuda_hipblaslt_i8_run_hipblaslt_q8_0_grouped_direct_qs(
                        plan,
                        activations_q8_1.get(),
                        (const block_q8_0 *) weights_k32,
                        ids_dst.get(),
                        active_experts,
                        expert_bounds_host,
                        dst_data,
                        scratch,
                        stream,
                        stride_expert_blocks);
                if (direct_status == GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                    return true;
                }
                if (plan.log_heuristic) {
                    GGML_LOG_WARN("%s: direct_qs Q8_0 MoE hipBLASLt-I8 failed status=%s; trying staged grouped path tensor=%s dst=%s\n",
                            __func__, ggml_cuda_hipblaslt_i8_status_name(direct_status), src0->name, dst->name);
                }
            }
            const ggml_cuda_hipblaslt_i8_status grouped_status = ggml_cuda_hipblaslt_i8_run_hipblaslt_k32_grouped(
                    plan,
                    activations_q8_1.get(),
                    weights_k32,
                    weight_type,
                    route_label,
                    ids_dst.get(),
                    active_experts,
                    active_experts_dev_ptr,
                    expert_bounds.get(),
                    expert_bounds_host,
                    dst_data,
                    scratch,
                    stream,
                    stride_expert_blocks);
            if (grouped_status == GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return true;
            }
            GGML_LOG_WARN("%s: grouped %s MoE hipBLASLt-I8 failed status=%s; falling back to standard MUL_MAT_ID route tensor=%s dst=%s\n",
                    __func__, route_label, ggml_cuda_hipblaslt_i8_status_name(grouped_status), src0->name, dst->name);
            return false;
        }
        if (grouped_requested) {
            const ggml_cuda_hipblaslt_i8_status grouped_status = ggml_cuda_hipblaslt_i8_run_hipblaslt_iq3_s_grouped(
                    plan,
                    activations_q8_1.get(),
                    weights,
                    ids_dst.get(),
                    active_experts,
                    active_experts_dev_ptr,
                    expert_bounds_host,
                    dst_data,
                    scratch,
                    stream,
                    stride_expert_blocks);
            if (grouped_status == GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                return true;
            }
            GGML_LOG_WARN("%s: grouped IQ3_S MoE hipBLASLt-I8 failed status=%s; falling back to standard MUL_MAT_ID route tensor=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_status_name(grouped_status), src0->name, dst->name);
            if (!use_iq3_s_moe) {
                return false;
            }
        }

        for (int expert = 0; expert < n_experts; ++expert) {
            const int row_low = expert_bounds_host[expert];
            const int row_high = expert_bounds_host[expert + 1];
            const int rows = row_high - row_low;
            if (rows == 0) {
                continue;
            }
            plan.m = rows;
            const ggml_cuda_hipblaslt_i8_status status = ggml_cuda_hipblaslt_i8_run_hipblaslt_iq3_s_range(
                    plan,
                    activations_q8_1.get() + (size_t) row_low * k32_blocks,
                    weights + (size_t) expert * stride_expert_blocks,
                    ids_dst.get() + row_low,
                    dst_data,
                    scratch,
                    stream);
            if (status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
                GGML_LOG_WARN("%s: IQ3_S MoE hipBLASLt-I8 route failed status=%s expert=%d rows=%d tensor=%s dst=%s\n",
                        __func__, ggml_cuda_hipblaslt_i8_status_name(status), expert, rows, src0->name, dst->name);
                return false;
            }
        }
        return true;
    }

    if (src0->type == GGML_TYPE_Q6_K) {
        if (!ggml_cuda_should_use_hipblaslt_i8_q6_K_prefill(src0, src1, dst, cc)) {
            return false;
        }

        const int k = (int) src0->ne[0];
        const int n = (int) src0->ne[1];
        const int m = (int) src1->ne[1];
        const int k32_blocks = k / GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        const int k256_blocks = k / QK_K;

        ggml_cuda_hipblaslt_i8_plan plan{};
        plan.m = m;
        plan.n = n;
        plan.k = k;
        plan.chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        plan.stride_a_blocks = k32_blocks;
        plan.stride_b_blocks = k256_blocks;
        plan.stride_d = (int) (dst->nb[1] / sizeof(float));
        plan.batch_count = 1;
        plan.heuristic_request_count = 1;
        if (const char * env_ws = std::getenv("GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")) {
            const long mb = std::strtol(env_ws, nullptr, 10);
            if (mb >= 0) {
                plan.hipblaslt_workspace_bytes = (size_t) mb << 20;
            }
        }
        plan.preferred_algo_ord = ggml_cuda_hipblaslt_i8_lookup_q6_K_shape_algo_ord(m, n, k);
        if (plan.preferred_algo_ord >= 0) {
            plan.heuristic_request_count = std::min(GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS, plan.preferred_algo_ord + 1);
        }
        const int forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q6_K_PREFILL_ALGO_ORD", -1);
        if (forced_algo_ord >= 0 && forced_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
            plan.preferred_algo_ord = forced_algo_ord;
            plan.heuristic_request_count = forced_algo_ord + 1;
        }
        plan.log_heuristic = ggml_cuda_hipblaslt_i8_diag_enabled();

        ggml_cuda_pool & pool = ctx.pool();
        const size_t activations_q8_1_bytes = (size_t) m * (size_t) k32_blocks * sizeof(block_q8_1);
        ggml_cuda_pool_alloc<block_q8_1> activations_q8_1_pool;
        block_q8_1 * activations_q8_1 = nullptr;
        bool activations_q8_1_needs_quantize = true;
        const bool activations_q8_1_cache_used = ggml_cuda_q8_1_act_cache_try_get(
                src1, k, activations_q8_1_bytes, stream, "hipblaslt_i8_q6_K", src0->name,
                &activations_q8_1, &activations_q8_1_needs_quantize);
        if (!activations_q8_1_cache_used) {
            activations_q8_1 = activations_q8_1_pool.alloc(pool, (size_t) m * (size_t) k32_blocks);
            activations_q8_1_needs_quantize = true;
        }
        if (activations_q8_1_needs_quantize) {
            quantize_row_q8_1_cuda(
                    (const float *) src1->data, nullptr, activations_q8_1, src0->type, k,
                    src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                    k, m, 1, 1, stream);
            CUDA_CHECK(cudaGetLastError());
            if (activations_q8_1_cache_used) {
                ggml_cuda_q8_1_act_cache_mark_valid("hipblaslt_i8_q6_K", src0->name);
            }
        }

        const ggml_cuda_hipblaslt_i8_status plan_status = ggml_cuda_hipblaslt_i8_validate_q6_K_plan_only(plan);
        if (plan_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
            GGML_LOG_WARN("%s: q6_K hipBLASLt-I8 scratch plan rejected: %s\n", __func__, ggml_cuda_hipblaslt_i8_status_name(plan_status));
            return false;
        }

        ggml_cuda_hipblaslt_i8_scratch required{};
        required.a_i8_bytes = (size_t) m * (size_t) plan.chunk_k * sizeof(int8_t);
        required.b_i8_bytes = (size_t) plan.chunk_k * (size_t) n * sizeof(int8_t);
        required.c_i32_bytes = (size_t) m * (size_t) n * sizeof(int32_t);
        required.hipblaslt_workspace_bytes = plan.hipblaslt_workspace_bytes;

        ggml_cuda_pool_alloc<int8_t>  a_i8(pool, required.a_i8_bytes / sizeof(int8_t));
        ggml_cuda_pool_alloc<int8_t>  b_i8(pool, required.b_i8_bytes / sizeof(int8_t));
        ggml_cuda_pool_alloc<int32_t> c_i32(pool, required.c_i32_bytes / sizeof(int32_t));
        ggml_cuda_pool_alloc<char> workspace;
        void * workspace_ptr = nullptr;
        if (required.hipblaslt_workspace_bytes != 0) {
            workspace_ptr = workspace.alloc(pool, required.hipblaslt_workspace_bytes);
        }

        ggml_cuda_hipblaslt_i8_args args{};
        args.activations_q8_1 = activations_q8_1;
        args.weights_q6_K = (const block_q6_K *) src0->data;
        args.dst = (float *) dst->data;
        args.scratch.a_i8 = a_i8.get();
        args.scratch.b_i8 = b_i8.get();
        args.scratch.c_i32 = c_i32.get();
        args.scratch.hipblaslt_workspace = workspace_ptr;
        args.scratch.a_i8_bytes = required.a_i8_bytes;
        args.scratch.b_i8_bytes = required.b_i8_bytes;
        args.scratch.c_i32_bytes = required.c_i32_bytes;
        args.scratch.hipblaslt_workspace_bytes = required.hipblaslt_workspace_bytes;
        args.stream = stream;

        const ggml_cuda_hipblaslt_i8_status status = ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q6_K(plan, args);
        if (status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
            GGML_LOG_WARN("%s: q6_K hipBLASLt-I8 route failed status=%s tensor=%s dst=%s\n",
                    __func__, ggml_cuda_hipblaslt_i8_status_name(status), src0->name, dst->name);
            return false;
        }
        return true;
    }

    if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 || src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1) {
        const bool is_q4_0 = src0->type == GGML_TYPE_Q4_0;
        const bool is_q4_1 = src0->type == GGML_TYPE_Q4_1;
        const bool is_q5_1 = src0->type == GGML_TYPE_Q5_1;
        if (is_q4_0) {
            if (!ggml_cuda_should_use_hipblaslt_i8_q4_0_prefill(src0, src1, dst, cc)) {
                return false;
            }
        } else if (is_q4_1) {
            if (!ggml_cuda_should_use_hipblaslt_i8_q4_1_prefill(src0, src1, dst, cc)) {
                return false;
            }
        } else if (is_q5_1) {
            if (!ggml_cuda_should_use_hipblaslt_i8_q5_1_prefill(src0, src1, dst, cc)) {
                return false;
            }
        } else if (!ggml_cuda_should_use_hipblaslt_i8_q5_0_prefill(src0, src1, dst, cc)) {
            return false;
        }

        const int k = (int) src0->ne[0];
        const int n = (int) src0->ne[1];
        const int m = (int) src1->ne[1];
        const int k_blocks = k / GGML_CUDA_HIPBLASLT_I8_CHUNK_K;

        ggml_cuda_hipblaslt_i8_plan plan{};
        plan.m = m;
        plan.n = n;
        plan.k = k;
        plan.chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
        plan.stride_a_blocks = k_blocks;
        plan.stride_b_blocks = k_blocks;
        plan.stride_d = (int) (dst->nb[1] / sizeof(float));
        plan.batch_count = 1;
        plan.weight_type = is_q4_0 ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_0 :
            (is_q4_1 ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q4_1 :
             (is_q5_1 ? GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_1 : GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q5_0));
        plan.heuristic_request_count = 1;
        if (const char * env_ws = std::getenv("GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")) {
            const long mb = std::strtol(env_ws, nullptr, 10);
            if (mb >= 0) {
                plan.hipblaslt_workspace_bytes = (size_t) mb << 20;
            }
        }
        if (!is_q4_1 && !is_q5_1) {
            plan.preferred_algo_ord = ggml_cuda_hipblaslt_i8_lookup_q4q5_k32_shape_algo_ord(m, n);
            if (plan.preferred_algo_ord >= 0) {
                plan.heuristic_request_count = std::min(GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS, plan.preferred_algo_ord + 1);
            }
        }
        const char * forced_env = is_q4_0 ? "GGML_CUDA_HIPBLASLT_I8_Q4_0_PREFILL_ALGO_ORD" :
            (is_q4_1 ? "GGML_CUDA_HIPBLASLT_I8_Q4_1_PREFILL_ALGO_ORD" :
             (is_q5_1 ? "GGML_CUDA_HIPBLASLT_I8_Q5_1_PREFILL_ALGO_ORD" : "GGML_CUDA_HIPBLASLT_I8_Q5_0_PREFILL_ALGO_ORD"));
        int forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int(forced_env, -1);
        if (!is_q4_1 && !is_q5_1 && forced_algo_ord < 0) {
            forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q4Q5_PREFILL_ALGO_ORD", -1);
        }
        if (forced_algo_ord >= 0 && forced_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
            plan.preferred_algo_ord = forced_algo_ord;
            plan.heuristic_request_count = forced_algo_ord + 1;
        }
        plan.log_heuristic = ggml_cuda_hipblaslt_i8_diag_enabled();

        ggml_cuda_pool & pool = ctx.pool();
        const size_t activations_q8_1_bytes = (size_t) m * (size_t) k_blocks * sizeof(block_q8_1);
        ggml_cuda_pool_alloc<block_q8_1> activations_q8_1_pool;
        block_q8_1 * activations_q8_1 = nullptr;
        bool activations_q8_1_needs_quantize = true;
        const char * cache_route = is_q4_0 ? "hipblaslt_i8_q4_0" :
            (is_q4_1 ? "hipblaslt_i8_q4_1" : (is_q5_1 ? "hipblaslt_i8_q5_1" : "hipblaslt_i8_q5_0"));
        const bool activations_q8_1_cache_used = ggml_cuda_q8_1_act_cache_try_get(
                src1, k, activations_q8_1_bytes, stream, cache_route, src0->name,
                &activations_q8_1, &activations_q8_1_needs_quantize);
        if (!activations_q8_1_cache_used) {
            activations_q8_1 = activations_q8_1_pool.alloc(pool, (size_t) m * (size_t) k_blocks);
            activations_q8_1_needs_quantize = true;
        }
        if (activations_q8_1_needs_quantize) {
            quantize_row_q8_1_cuda(
                    (const float *) src1->data, nullptr, activations_q8_1, src0->type, k,
                    src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                    k, m, 1, 1, stream);
            CUDA_CHECK(cudaGetLastError());
            if (activations_q8_1_cache_used) {
                ggml_cuda_q8_1_act_cache_mark_valid(cache_route, src0->name);
            }
        }

        ggml_cuda_hipblaslt_i8_scratch required{};
        const ggml_cuda_hipblaslt_i8_status scratch_status = ggml_cuda_hipblaslt_i8_get_scratch_size(plan, &required);
        if (scratch_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
            GGML_LOG_WARN("%s: %s hipBLASLt-I8 scratch plan rejected: %s\n", __func__, is_q4_0 ? "q4_0" : (is_q4_1 ? "q4_1" : (is_q5_1 ? "q5_1" : "q5_0")), ggml_cuda_hipblaslt_i8_status_name(scratch_status));
            return false;
        }

        ggml_cuda_pool_alloc<int8_t>  a_i8(pool, required.a_i8_bytes / sizeof(int8_t));
        ggml_cuda_pool_alloc<int8_t>  b_i8(pool, required.b_i8_bytes / sizeof(int8_t));
        ggml_cuda_pool_alloc<int32_t> c_i32(pool, required.c_i32_bytes / sizeof(int32_t));
        ggml_cuda_pool_alloc<char> workspace;
        void * workspace_ptr = nullptr;
        if (required.hipblaslt_workspace_bytes != 0) {
            workspace_ptr = workspace.alloc(pool, required.hipblaslt_workspace_bytes);
        }

        ggml_cuda_hipblaslt_i8_args args{};
        args.activations_q8_1 = activations_q8_1;
        args.weights_q4_0 = is_q4_0 ? (const block_q4_0 *) src0->data : nullptr;
        args.weights_q4_1 = is_q4_1 ? (const block_q4_1 *) src0->data : nullptr;
        args.weights_q5_0 = (!is_q4_0 && !is_q4_1 && !is_q5_1) ? (const block_q5_0 *) src0->data : nullptr;
        args.weights_q5_1 = is_q5_1 ? (const block_q5_1 *) src0->data : nullptr;
        args.dst = (float *) dst->data;
        args.scratch.a_i8 = a_i8.get();
        args.scratch.b_i8 = b_i8.get();
        args.scratch.c_i32 = c_i32.get();
        args.scratch.hipblaslt_workspace = workspace_ptr;
        args.scratch.a_i8_bytes = required.a_i8_bytes;
        args.scratch.b_i8_bytes = required.b_i8_bytes;
        args.scratch.c_i32_bytes = required.c_i32_bytes;
        args.scratch.hipblaslt_workspace_bytes = required.hipblaslt_workspace_bytes;
        args.stream = stream;

        const ggml_cuda_hipblaslt_i8_status status = ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q8_0(plan, args);
        if (status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
            GGML_LOG_WARN("%s: %s hipBLASLt-I8 route failed status=%s tensor=%s dst=%s\n",
                    __func__, is_q4_0 ? "q4_0" : (is_q4_1 ? "q4_1" : (is_q5_1 ? "q5_1" : "q5_0")), ggml_cuda_hipblaslt_i8_status_name(status), src0->name, dst->name);
            return false;
        }
        return true;
    }

    const int k = (int) src0->ne[0];
    const int n = (int) src0->ne[1];
    const int m = (int) src1->ne[1];
    const int k_blocks = k / GGML_CUDA_HIPBLASLT_I8_CHUNK_K;

    ggml_cuda_hipblaslt_i8_plan plan{};
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.chunk_k = GGML_CUDA_HIPBLASLT_I8_CHUNK_K;
    plan.stride_a_blocks = k_blocks;
    plan.stride_b_blocks = k_blocks;
    plan.stride_d = (int) (dst->nb[1] / sizeof(float));
    plan.batch_count = 1;
    plan.weight_type = GGML_CUDA_HIPBLASLT_I8_WEIGHT_Q8_0;
    plan.heuristic_request_count = 1;
    plan.full_k_stage_k32_exact = ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_STAGE_K32_EXACT") ||
        ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_FULLK_STAGE_K32_EXACT");
    if (const char * env_ws = std::getenv("GGML_CUDA_HIPBLASLT_I8_WORKSPACE_MB")) {
        const long mb = std::strtol(env_ws, nullptr, 10);
        if (mb >= 0) {
            plan.hipblaslt_workspace_bytes = (size_t) mb << 20;
        }
    }
    if (ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_PREFILL_FULLK_ARTIFACT_ALGO_PREFS") ||
            ggml_cuda_hipblaslt_i8_env_enabled("GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_FULLK_ARTIFACT_ALGO_PREFS")) {
        // Diagnostic only unless full-K staging is also enabled: the table was swept on full-K dense-I8 GEMMs.
        // Keep the env explicit and fall back to the first successful heuristic.
        plan.preferred_algo_ord = ggml_cuda_hipblaslt_i8_lookup_shape_algo_ord(m, n, k);
        if (plan.preferred_algo_ord >= 0) {
            plan.heuristic_request_count = std::min(GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS, plan.preferred_algo_ord + 1);
        }
    }
    int forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_Q8_0_PREFILL_ALGO_ORD", -1);
    if (forced_algo_ord < 0) {
        forced_algo_ord = ggml_cuda_hipblaslt_i8_env_int("GGML_CUDA_HIPBLASLT_I8_PREFILL_ALGO_ORD", -1);
    }
    if (forced_algo_ord >= 0 && forced_algo_ord < GGML_CUDA_HIPBLASLT_I8_MAX_HEURISTICS) {
        plan.preferred_algo_ord = forced_algo_ord;
        plan.heuristic_request_count = forced_algo_ord + 1;
    }
    plan.log_heuristic = ggml_cuda_hipblaslt_i8_diag_enabled();

    ggml_cuda_pool & pool = ctx.pool();
    const size_t activations_q8_1_bytes = (size_t) m * (size_t) k_blocks * sizeof(block_q8_1);
    ggml_cuda_pool_alloc<block_q8_1> activations_q8_1_pool;
    block_q8_1 * activations_q8_1 = nullptr;
    bool activations_q8_1_needs_quantize = true;
    const bool activations_q8_1_cache_used = ggml_cuda_q8_1_act_cache_try_get(
            src1, k, activations_q8_1_bytes, stream, "hipblaslt_i8_q8_0", src0->name,
            &activations_q8_1, &activations_q8_1_needs_quantize);
    if (!activations_q8_1_cache_used) {
        activations_q8_1 = activations_q8_1_pool.alloc(pool, (size_t) m * (size_t) k_blocks);
        activations_q8_1_needs_quantize = true;
    }
    if (activations_q8_1_needs_quantize) {
        quantize_row_q8_1_cuda(
                (const float *) src1->data, nullptr, activations_q8_1, src0->type, k,
                src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                k, m, 1, 1, stream);
        CUDA_CHECK(cudaGetLastError());
        if (activations_q8_1_cache_used) {
            ggml_cuda_q8_1_act_cache_mark_valid("hipblaslt_i8_q8_0", src0->name);
        }
    }

    ggml_cuda_hipblaslt_i8_scratch required{};
    const ggml_cuda_hipblaslt_i8_status scratch_status = ggml_cuda_hipblaslt_i8_get_scratch_size(plan, &required);
    if (scratch_status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: q8_0 hipBLASLt-I8 scratch plan rejected: %s\n", __func__, ggml_cuda_hipblaslt_i8_status_name(scratch_status));
        return false;
    }

    ggml_cuda_pool_alloc<int8_t>  a_i8(pool, required.a_i8_bytes / sizeof(int8_t));
    ggml_cuda_pool_alloc<int8_t>  b_i8(pool, required.b_i8_bytes / sizeof(int8_t));
    ggml_cuda_pool_alloc<int32_t> c_i32(pool, required.c_i32_bytes / sizeof(int32_t));
    ggml_cuda_pool_alloc<char> workspace;
    void * workspace_ptr = nullptr;
    if (required.hipblaslt_workspace_bytes != 0) {
        workspace_ptr = workspace.alloc(pool, required.hipblaslt_workspace_bytes);
    }

    ggml_cuda_hipblaslt_i8_args args{};
    args.activations_q8_1 = activations_q8_1;
    args.weights_q8_0 = (const block_q8_0 *) src0->data;
    args.dst = (float *) dst->data;
    args.scratch.a_i8 = a_i8.get();
    args.scratch.b_i8 = b_i8.get();
    args.scratch.c_i32 = c_i32.get();
    args.scratch.hipblaslt_workspace = workspace_ptr;
    args.scratch.a_i8_bytes = required.a_i8_bytes;
    args.scratch.b_i8_bytes = required.b_i8_bytes;
    args.scratch.c_i32_bytes = required.c_i32_bytes;
    args.scratch.hipblaslt_workspace_bytes = required.hipblaslt_workspace_bytes;
    args.stream = stream;

    const ggml_cuda_hipblaslt_i8_status status = ggml_cuda_hipblaslt_i8_mul_mat_q8_1_q8_0(plan, args);
    if (status != GGML_CUDA_HIPBLASLT_I8_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: q8_0 hipBLASLt-I8 route failed status=%s tensor=%s dst=%s\n",
                __func__, ggml_cuda_hipblaslt_i8_status_name(status), src0->name, dst->name);
        return false;
    }

    return true;
#endif
}
