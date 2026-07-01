#include "rdna-i8-q8-0.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(GGML_USE_HIP)
#include "q8-1-act-cache.cuh"
#include "quantize.cuh"
#endif

#if defined(GGML_USE_HIP)

namespace {

using ggml_cuda_rdna_i8_v4i32 = int __attribute__((ext_vector_type(4)));
using ggml_cuda_rdna_i8_v8i32 = int __attribute__((ext_vector_type(8)));

static_assert(QK8_0 == 32 && QK8_1 == 32, "RDNA I8 q8_0 WMMA route assumes K32 q8_0/q8_1 blocks");

static bool ggml_cuda_rdna_i8_q8_0_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0;
}

static bool ggml_cuda_rdna_i8_q8_0_log_enabled() {
    static const bool enabled = ggml_cuda_rdna_i8_q8_0_env_enabled("GGML_CUDA_RDNA_I8_Q8_0_PREFILL_LOG");
    return enabled;
}

static bool ggml_cuda_rdna_i8_q8_0_mul_overflows_size(const size_t a, const size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return true;
    }
    out = a * b;
    return false;
}

static __device__ __forceinline__ ggml_cuda_rdna_i8_v8i32 ggml_cuda_rdna_i8_wmma_i32_16x16x16(
        ggml_cuda_rdna_i8_v4i32 a_frag,
        ggml_cuda_rdna_i8_v4i32 b_frag,
        ggml_cuda_rdna_i8_v8i32 acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, b_frag, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_frag, true, b_frag, acc, false);
#else
    return acc;
#endif
}

static __device__ __forceinline__ uint32_t ggml_cuda_rdna_i8_q8_0_pack_i8x4(
        const int8_t * __restrict__ qs,
        const int kk) {
    uint32_t w = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        w |= (uint32_t) (uint8_t) qs[kk + byte] << (8 * byte);
    }
    return w;
}

static __device__ __forceinline__ float ggml_cuda_rdna_i8_q8_1_scale(const block_q8_1 & block) {
    return __low2float(block.ds);
}

static __device__ __forceinline__ float ggml_cuda_rdna_i8_q8_0_scale(const block_q8_0 & block) {
    return __half2float(block.d);
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_q8_0_prefill_kernel(
        const block_q8_1 * __restrict__ a, // q8_1 activations [M,K/32]
        const block_q8_0 * __restrict__ b, // q8_0 weights [N,K/32]
        float * __restrict__ dst,          // F32 [M,N] in ggml dst row-major token stride
        int m,
        int n,
        int k_blocks,
        int stride_d) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = (int) blockIdx.y * 16;
    const int tile_n = (int) blockIdx.x * 16;
    const int row_a  = tile_m + lane_lo;
    const int col_b  = tile_n + lane_lo;

    // Wave32 RDNA WMMA layout, proven by scripts/hip/rdna-i8-wmma-gemm-proof.hip:
    // lanes 0..15 provide one A row and one B column for the 16x16 tile, lanes 16..31
    // duplicate operand rows/cols and receive the odd output rows via acc element mapping below.
    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const block_q8_1 * ab_lane = row_a < m ? a + row_a * k_blocks + kb : nullptr;
        const block_q8_0 * wb_lane = col_b < n ? b + col_b * k_blocks + kb : nullptr;
        ggml_cuda_rdna_i8_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int kk16 = 0; kk16 < QK8_0; kk16 += 16) {
            ggml_cuda_rdna_i8_v4i32 a_frag;
            ggml_cuda_rdna_i8_v4i32 b_frag;

#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int kk = kk16 + 4 * g;
                a_frag[g] = ab_lane != nullptr ? (int) ggml_cuda_rdna_i8_q8_0_pack_i8x4(ab_lane->qs, kk) : 0;
                b_frag[g] = wb_lane != nullptr ? (int) ggml_cuda_rdna_i8_q8_0_pack_i8x4(wb_lane->qs, kk) : 0;
            }

            acc_i = ggml_cuda_rdna_i8_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int out_m = tile_m + 2 * r + lane_hi;
            const int out_n = tile_n + lane_lo;
            if (out_m < m && out_n < n) {
                const block_q8_1 & ab = a[out_m * k_blocks + kb];
                const block_q8_0 & wb = b[out_n * k_blocks + kb];
                const float ad = ggml_cuda_rdna_i8_q8_1_scale(ab);
                const float wd = ggml_cuda_rdna_i8_q8_0_scale(wb);
                acc_f[r] += (float) acc_i[r] * ad * wd;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        if (out_m < m && out_n < n) {
            dst[out_m * stride_d + out_n] = acc_f[r];
        }
    }
}

} // namespace

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_rdna_i8_q8_0_prefill(
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
    static const bool enabled = ggml_cuda_rdna_i8_q8_0_env_enabled("GGML_CUDA_RDNA_I8_Q8_0_PREFILL");
    if (!enabled) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src1->ne[1] <= 8) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1]) {
        return false;
    }
    if (src0->ne[0] % QK8_0 != 0) {
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
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max()) {
        return false;
    }
    if (src1->nb[1] % (int64_t) sizeof(float) != 0 || src1->nb[2] % (int64_t) sizeof(float) != 0 ||
            src1->nb[3] % (int64_t) sizeof(float) != 0 || dst->nb[1] % (int64_t) sizeof(float) != 0) {
        return false;
    }
    if (dst->nb[1] / (int64_t) sizeof(float) > std::numeric_limits<int>::max()) {
        return false;
    }

    if (const char * only = std::getenv("GGML_CUDA_RDNA_I8_Q8_0_PREFILL_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return false;
        }
    }
    return true;
#endif
}

bool ggml_cuda_mul_mat_rdna_i8_q8_0(
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
    if (src0 == nullptr || src1 == nullptr || dst == nullptr || ids != nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src1->ne[1] <= 0 || src0->ne[0] != src1->ne[0] ||
            dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || src0->ne[0] % QK8_0 != 0) {
        return false;
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src1->ne[1] > std::numeric_limits<int>::max() || dst->nb[1] % (int64_t) sizeof(float) != 0 ||
            dst->nb[1] / (int64_t) sizeof(float) > std::numeric_limits<int>::max()) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_mtp_mmq_stream_is_capturing(stream)) {
        return false;
    }

    const int k = (int) src0->ne[0];
    const int n = (int) src0->ne[1];
    const int m = (int) src1->ne[1];
    const int k_blocks = k / QK8_0;
    const int stride_d = (int) (dst->nb[1] / sizeof(float));

    size_t activation_blocks = 0;
    size_t activations_q8_1_bytes = 0;
    if (ggml_cuda_rdna_i8_q8_0_mul_overflows_size((size_t) m, (size_t) k_blocks, activation_blocks) ||
            ggml_cuda_rdna_i8_q8_0_mul_overflows_size(activation_blocks, sizeof(block_q8_1), activations_q8_1_bytes)) {
        if (ggml_cuda_rdna_i8_q8_0_log_enabled()) {
            GGML_LOG_WARN("%s: rdna_i8_q8_0 route rejected: activation q8_1 allocation overflows m=%d k_blocks=%d tensor=%s\n",
                    __func__, m, k_blocks, src0->name);
        }
        return false;
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<block_q8_1> activations_q8_1_pool;
    block_q8_1 * activations_q8_1 = nullptr;
    bool activations_q8_1_needs_quantize = true;
    const bool activations_q8_1_cache_used = ggml_cuda_q8_1_act_cache_try_get(
            src1, k, activations_q8_1_bytes, stream, "rdna_i8_q8_0", src0->name,
            &activations_q8_1, &activations_q8_1_needs_quantize);
    if (!activations_q8_1_cache_used) {
        activations_q8_1 = activations_q8_1_pool.alloc(pool, activation_blocks);
        activations_q8_1_needs_quantize = true;
    }
    if (activations_q8_1_needs_quantize) {
        quantize_row_q8_1_cuda(
                (const float *) src1->data, nullptr, activations_q8_1, src0->type, k,
                src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
                k, m, 1, 1, stream);
        CUDA_CHECK(cudaGetLastError());
        if (activations_q8_1_cache_used) {
            ggml_cuda_q8_1_act_cache_mark_valid("rdna_i8_q8_0", src0->name);
        }
    }

    const dim3 block(32, 1, 1);
    const dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
    if (ggml_cuda_rdna_i8_q8_0_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=rdna_i8_q8_0 tensor=%s dst=%s status=selected m=%d n=%d k=%d k_blocks=%d cache=%d quantize=%d stride_d=%d grid=(%u,%u,%u)\n",
                __func__, src0->name, dst->name, m, n, k, k_blocks, activations_q8_1_cache_used ? 1 : 0,
                activations_q8_1_needs_quantize ? 1 : 0, stride_d, grid.x, grid.y, grid.z);
    }
    ggml_cuda_rdna_i8_q8_0_prefill_kernel<<<grid, block, 0, stream>>>(
            activations_q8_1, (const block_q8_0 *) src0->data, (float *) dst->data,
            m, n, k_blocks, stride_d);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}
