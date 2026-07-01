#include "rdna-i8-q6-k.cuh"

#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(GGML_USE_HIP)
#include "quantize.cuh"
#endif

#if defined(GGML_USE_HIP)

namespace {

static_assert(QK_K == 256, "rdna-i8-q6-k assumes q6_K super-blocks with 256 elements");
static_assert(QK8_1 == 32, "rdna-i8-q6-k assumes q8_1 activation blocks with 32 elements");

static bool ggml_cuda_rdna_i8_q6_K_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' &&
           std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "off") != 0 && std::strcmp(env, "OFF") != 0 &&
           std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0;
}

static bool ggml_cuda_rdna_i8_q6_K_log_enabled() {
    return ggml_cuda_rdna_i8_q6_K_env_enabled("GGML_CUDA_RDNA_I8_Q6_K_PREFILL_LOG");
}

static __device__ __forceinline__ int8_t ggml_cuda_rdna_i8_q6_K_get_i8(
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

static __device__ __forceinline__ int8_t ggml_cuda_rdna_i8_q6_K_get_scale(
        const block_q6_K & block,
        const int kk) {
    return block.scales[kk >> 4]; // one q6_K subscale per 16 weights
}

static __device__ __forceinline__ float ggml_cuda_rdna_i8_q6_K_dot_block(
        const block_q8_1_mmq * __restrict__ a,
        const int row,
        const int m,
        const block_q6_K & wb,
        const int kb6) {
    float acc = 0.0f;
    const float d6 = __half2float(wb.d);

#pragma unroll
    for (int q8_sub = 0; q8_sub < QK_K / QK8_1; ++q8_sub) {
        const int kk_base = q8_sub * QK8_1;
        const int mmq_k_block = kb6 * 2 + q8_sub / 4;
        const block_q8_1_mmq & ab = a[(size_t) mmq_k_block * (size_t) m + row];
        const int8_t * aq = ab.qs + (q8_sub & 3) * QK8_1;
        const float d8 = ab.d4[q8_sub & 3];

        int sumi_lo = 0;
        int sumi_hi = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int kk_lo = kk_base + i;
            const int kk_hi = kk_base + 16 + i;
            sumi_lo += (int) aq[i]      * (int) ggml_cuda_rdna_i8_q6_K_get_i8(wb, kk_lo);
            sumi_hi += (int) aq[16 + i] * (int) ggml_cuda_rdna_i8_q6_K_get_i8(wb, kk_hi);
        }

        const int sc_lo = (int) ggml_cuda_rdna_i8_q6_K_get_scale(wb, kk_base);
        const int sc_hi = (int) ggml_cuda_rdna_i8_q6_K_get_scale(wb, kk_base + 16);
        acc += (float) (sc_lo * sumi_lo) * d6 * d8;
        acc += (float) (sc_hi * sumi_hi) * d6 * d8;
    }

    return acc;
}

static __global__ __launch_bounds__(256, 2) void ggml_cuda_rdna_i8_q6_K_prefill_kernel(
        const block_q8_1_mmq * __restrict__ a, // MMQ q8_1 activations [M,K/128]
        const block_q6_K * __restrict__ b,      // q6_K weights [N,K/256]
        float * __restrict__ dst,               // F32 [M,N]
        int m,
        int n,
        int k_mmq_blocks,
        int k_q6_blocks,
        int stride_d) {
    const int row = (int) blockIdx.y * 16 + (int) threadIdx.y;
    const int col = (int) blockIdx.x * 16 + (int) threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    GGML_UNUSED(k_mmq_blocks);
    const block_q6_K * brow = b + (size_t) col * (size_t) k_q6_blocks;

    float acc = 0.0f;
    for (int kb6 = 0; kb6 < k_q6_blocks; ++kb6) {
        acc += ggml_cuda_rdna_i8_q6_K_dot_block(a, row, m, brow[kb6], kb6);
    }

    dst[(size_t) row * (size_t) stride_d + col] = acc;
}

} // namespace

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_rdna_i8_q6_K_prefill(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    static const bool enabled = ggml_cuda_rdna_i8_q6_K_env_enabled("GGML_CUDA_RDNA_I8_Q6_K_PREFILL");
    static const bool unsafe_hash_drift = ggml_cuda_rdna_i8_q6_K_env_enabled("GGML_CUDA_RDNA_I8_Q6_K_PREFILL_UNSAFE_HASH_DRIFT");
    if (!enabled || !unsafe_hash_drift) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] <= 8 || src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1]) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src1->ne[1] <= 0 || src0->ne[0] % QK_K != 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return false;
    }
    if (src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float) || (dst->nb[1] % (int64_t) sizeof(float)) != 0) {
        return false;
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max()) {
        return false;
    }

    if (const char * only = std::getenv("GGML_CUDA_RDNA_I8_Q6_K_PREFILL_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return false;
        }
    }
    if (ggml_cuda_rdna_i8_q6_K_log_enabled()) {
        GGML_LOG_INFO("%s: route=rdna_i8_q6_K_prefill status=selected tensor=%s dst=%s m=%lld n=%lld k=%lld\n",
                __func__, src0->name, dst->name, (long long) src1->ne[1], (long long) src0->ne[1], (long long) src0->ne[0]);
    }
    return true;
#endif
}

bool ggml_cuda_mul_mat_rdna_i8_q6_K(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#else
    if (ids != nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_mtp_mmq_stream_is_capturing(stream)) {
        return false;
    }

    const int k = (int) src0->ne[0];
    const int n = (int) src0->ne[1];
    const int m = (int) src1->ne[1];
    const int k_q8_blocks = k / QK8_1;
    const int k_mmq_blocks = k / (4 * QK8_1);
    const int k_q6_blocks = k / QK_K;
    const int stride_d = (int) (dst->nb[1] / sizeof(float));

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<block_q8_1_mmq> activations_q8_1_pool;
    block_q8_1_mmq * activations_q8_1 = activations_q8_1_pool.alloc(pool, (size_t) m * (size_t) k_mmq_blocks);
    quantize_mmq_q8_1_cuda(
            (const float *) src1->data, nullptr, activations_q8_1, src0->type, k,
            src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            k, m, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());

    const dim3 block(16, 16, 1);
    const dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
    if (ggml_cuda_rdna_i8_q6_K_log_enabled()) {
        GGML_LOG_INFO("%s: route=rdna_i8_q6_K_prefill status=launch tensor=%s m=%d n=%d k=%d q8_blocks=%d q8_mmq_blocks=%d q6_blocks=%d cache=%d quantize=%d grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                __func__, src0->name, m, n, k, k_q8_blocks, k_mmq_blocks, k_q6_blocks,
                0, 1, grid.x, grid.y, grid.z, block.x, block.y, block.z);
    }
    ggml_cuda_rdna_i8_q6_K_prefill_kernel<<<grid, block, 0, stream>>>(
            activations_q8_1, (const block_q6_K *) src0->data, (float *) dst->data,
            m, n, k_mmq_blocks, k_q6_blocks, stride_d);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}
