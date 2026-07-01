#include "rdna-i8-packed16.cuh"

#include "dot4-packed16/dp16-packed-i8-desc.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(GGML_USE_HIP)

namespace {

using ggml_cuda_rdna_i8_p16_v4i32 = int __attribute__((ext_vector_type(4)));
using ggml_cuda_rdna_i8_p16_v8i32 = int __attribute__((ext_vector_type(8)));

static constexpr int GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS = 8;

static bool ggml_cuda_rdna_i8_packed16_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0;
}

static bool ggml_cuda_rdna_i8_packed16_log_enabled() {
    static const bool enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_GEMM_LOG");
    return enabled;
}

static bool ggml_cuda_rdna_i8_packed16_id_log_enabled() {
    static const bool enabled = ggml_cuda_rdna_i8_packed16_log_enabled() ||
        ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG");
    return enabled;
}

static bool ggml_cuda_rdna_i8_packed16_id_bounds_enabled() {
    static const bool enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_BOUNDS");
    return enabled;
}

static const ggml_tensor * ggml_cuda_rdna_i8_packed16_scales(const ggml_tensor * payload) {
    if (payload == nullptr || payload->type != GGML_TYPE_I32) {
        return nullptr;
    }
    const ggml_tensor * scales = payload->src[1];
    if (scales == nullptr || scales->type != GGML_TYPE_F16) {
        return nullptr;
    }
    return scales;
}

static bool ggml_cuda_rdna_i8_packed16_valid_operand(
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const int64_t rows,
        const int64_t k_words) {
    if (payload == nullptr || scales == nullptr || payload->type != GGML_TYPE_I32 || scales->type != GGML_TYPE_F16) {
        return false;
    }
    if (k_words <= 0 || rows <= 0 || (k_words % GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS) != 0) {
        return false;
    }
    if (payload->ne[0] != k_words || payload->ne[1] != rows || payload->ne[2] != 1 || payload->ne[3] != 1) {
        return false;
    }
    const int64_t k_blocks = k_words / GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;
    if (scales->ne[0] != k_blocks || scales->ne[1] != rows || scales->ne[2] != 1 || scales->ne[3] != 1) {
        return false;
    }
    if (payload->nb[0] != (int64_t) sizeof(int32_t) || scales->nb[0] != (int64_t) sizeof(half)) {
        return false;
    }
    if (payload->nb[1] % (int64_t) sizeof(int32_t) != 0 || scales->nb[1] % (int64_t) sizeof(half) != 0) {
        return false;
    }
    if (ggml_is_transposed(payload) || ggml_is_transposed(scales)) {
        return false;
    }
    return true;
}

static const ggml_tensor * ggml_cuda_rdna_i8_packed16_id_scales(
        const ggml_tensor * payload,
        const ggml_tensor * dst,
        const int sidecar_index) {
    const ggml_tensor * scales = nullptr;
    if (dst != nullptr && sidecar_index >= 0 && sidecar_index < GGML_MAX_SRC) {
        scales = dst->src[sidecar_index];
    }
    if (scales == nullptr) {
        scales = ggml_cuda_rdna_i8_packed16_scales(payload);
    }
    if (scales == nullptr || scales->type != GGML_TYPE_F16) {
        return nullptr;
    }
    return scales;
}

static bool ggml_cuda_rdna_i8_packed16_valid_operand_3d(
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const int64_t rows,
        const int64_t planes,
        const int64_t k_words) {
    if (payload == nullptr || scales == nullptr || payload->type != GGML_TYPE_I32 || scales->type != GGML_TYPE_F16) {
        return false;
    }
    if (k_words <= 0 || rows <= 0 || planes <= 0 || (k_words % GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS) != 0) {
        return false;
    }
    if (payload->ne[0] != k_words || payload->ne[1] != rows || payload->ne[2] != planes || payload->ne[3] != 1) {
        return false;
    }
    const int64_t k_blocks = k_words / GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;
    if (scales->ne[0] != k_blocks || scales->ne[1] != rows || scales->ne[2] != planes || scales->ne[3] != 1) {
        return false;
    }
    if (payload->nb[0] != (int64_t) sizeof(int32_t) || scales->nb[0] != (int64_t) sizeof(half)) {
        return false;
    }
    if (payload->nb[1] % (int64_t) sizeof(int32_t) != 0 || payload->nb[2] % (int64_t) sizeof(int32_t) != 0 ||
            scales->nb[1] % (int64_t) sizeof(half) != 0 || scales->nb[2] % (int64_t) sizeof(half) != 0) {
        return false;
    }
    if (ggml_is_transposed(payload) || ggml_is_transposed(scales)) {
        return false;
    }
    return true;
}

static bool ggml_cuda_rdna_i8_packed16_valid_id_bounds(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        const ggml_tensor * bounds) {
    if (!ggml_cuda_rdna_i8_packed16_id_bounds_enabled() || src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr || bounds == nullptr) {
        return false;
    }
    if (bounds->type != GGML_TYPE_I32 || bounds->ne[0] != 2 || bounds->ne[1] != src0->ne[2] || bounds->ne[2] != 1 || bounds->ne[3] != 1) {
        return false;
    }
    if (src1->ne[1] != 1 || ids->ne[0] != 1 || dst->ne[1] != 1 || ids->ne[1] != src1->ne[2] || dst->ne[2] != src1->ne[2]) {
        return false;
    }
    if (bounds->nb[0] % (int64_t) sizeof(int32_t) != 0 || bounds->nb[1] % (int64_t) sizeof(int32_t) != 0) {
        return false;
    }
    if (src1->nb[2] % (int64_t) sizeof(int32_t) != 0 || dst->nb[2] % (int64_t) sizeof(float) != 0) {
        return false;
    }
    return true;
}

static __device__ __forceinline__ ggml_cuda_rdna_i8_p16_v8i32 ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(
        ggml_cuda_rdna_i8_p16_v4i32 a_frag,
        ggml_cuda_rdna_i8_p16_v4i32 b_frag,
        ggml_cuda_rdna_i8_p16_v8i32 acc) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_frag, true, b_frag, acc, false);
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_frag, true, b_frag, acc, false);
#else
    return acc;
#endif
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_kernel(
        const int32_t * __restrict__ a_words,  // packed A rows [M,K/4]
        const half    * __restrict__ a_scales, // A scales [M,K/32]
        const int32_t * __restrict__ b_words,  // packed B rows [N,K/4]
        const half    * __restrict__ b_scales, // B scales [N,K/32]
        float         * __restrict__ dst,      // F32 [M,N], ggml dst stride
        int m,
        int n,
        int k_blocks,
        int a_word_stride,
        int b_word_stride,
        int a_scale_stride,
        int b_scale_stride,
        int dst_stride) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int tile_m = (int) blockIdx.y * 16;
    const int tile_n = (int) blockIdx.x * 16;
    const int row_a  = tile_m + lane_lo;
    const int col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const int32_t * a_row = row_a < m ? a_words + (size_t) row_a * a_word_stride + (size_t) kb * GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        const int32_t * b_row = col_b < n ? b_words + (size_t) col_b * b_word_stride + (size_t) kb * GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                b_frag[g] = b_row != nullptr ? b_row[word] : 0;
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int out_m = tile_m + 2 * r + lane_hi;
            const int out_n = tile_n + lane_lo;
            if (out_m < m && out_n < n) {
                const float as = __half2float(a_scales[(size_t) out_m * a_scale_stride + kb]);
                const float bs = __half2float(b_scales[(size_t) out_n * b_scale_stride + kb]);
                acc_f[r] += (float) acc_i[r] * as * bs;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int out_m = tile_m + 2 * r + lane_hi;
        const int out_n = tile_n + lane_lo;
        if (out_m < m && out_n < n) {
            dst[(size_t) out_m * dst_stride + out_n] = acc_f[r];
        }
    }
}

static __global__ void ggml_cuda_rdna_i8_packed16_gemm_id_kernel(
        const int32_t * __restrict__ weight_words,  // [K/4, n_out, n_expert]
        const half    * __restrict__ weight_scales, // [K/32, n_out, n_expert]
        const int32_t * __restrict__ act_words,     // [K/4, n_src1_slots, n_tokens]
        const half    * __restrict__ act_scales,    // [K/32, n_src1_slots, n_tokens]
        const int32_t * __restrict__ ids,           // [n_ids, n_tokens]
        float         * __restrict__ dst,           // [n_out, n_ids, n_tokens]
        int64_t total,
        int k_blocks,
        int64_t n_out,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_src1_slots,
        int64_t n_expert,
        int64_t weight_word_s1,
        int64_t weight_word_s2,
        int64_t act_word_s1,
        int64_t act_word_s2,
        int64_t weight_scale_s1,
        int64_t weight_scale_s2,
        int64_t act_scale_s1,
        int64_t act_scale_s2,
        int64_t ids_s0,
        int64_t ids_s1,
        int64_t dst_s1,
        int64_t dst_s2) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int64_t out   = idx % n_out;
    const int64_t slot  = (idx / n_out) % n_ids;
    const int64_t token = idx / (n_out*n_ids);
    if (token >= n_tokens) {
        return;
    }

    const int32_t expert = ids[slot*ids_s0 + token*ids_s1];
    if (expert < 0 || (int64_t) expert >= n_expert) {
        dst[out + slot*dst_s1 + token*dst_s2] = 0.0f;
        return;
    }

    const int64_t act_slot = slot % n_src1_slots;
    const int32_t * weight_row = weight_words + out*weight_word_s1 + (int64_t) expert*weight_word_s2;
    const int32_t * act_row    = act_words    + act_slot*act_word_s1    + token*act_word_s2;
    const half    * weight_s   = weight_scales + out*weight_scale_s1 + (int64_t) expert*weight_scale_s2;
    const half    * act_s      = act_scales    + act_slot*act_scale_s1    + token*act_scale_s2;

    float acc = 0.0f;
    for (int kb = 0; kb < k_blocks; ++kb) {
        int dot = 0;
#pragma unroll
        for (int w = 0; w < GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS; ++w) {
            dot = ggml_cuda_dp4a(act_row[(int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS + w],
                    weight_row[(int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS + w], dot);
        }
        acc += (float) dot * __half2float(act_s[kb]) * __half2float(weight_s[kb]);
    }

    dst[out + slot*dst_s1 + token*dst_s2] = acc;
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_kernel(
        const int32_t * __restrict__ weight_words,  // [K/4, n_out, n_expert]
        const half    * __restrict__ weight_scales, // [K/32, n_out, n_expert]
        const int32_t * __restrict__ act_words,     // compact lanes [K/4, 1, n_lanes]
        const half    * __restrict__ act_scales,    // compact lane scales [K/32, 1, n_lanes]
        const int32_t * __restrict__ bounds,        // [2, n_expert]: start,count into compact lanes
        float         * __restrict__ dst,           // [n_out, 1, n_lanes] or [n_out,n_lanes] with dst_s2 lane stride
        int k_blocks,
        int64_t n_out,
        int64_t n_lanes,
        int64_t n_expert,
        int64_t max_tiles_m,
        int64_t weight_word_s1,
        int64_t weight_word_s2,
        int64_t act_word_s2,
        int64_t weight_scale_s1,
        int64_t weight_scale_s2,
        int64_t act_scale_s2,
        int64_t bounds_s0,
        int64_t bounds_s1,
        int64_t dst_s2) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int64_t expert = (int64_t) blockIdx.z;
    if (expert >= n_expert) {
        return;
    }

    const int32_t start = bounds[0*bounds_s0 + expert*bounds_s1];
    const int32_t count = bounds[1*bounds_s0 + expert*bounds_s1];
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes || (int64_t) blockIdx.y >= max_tiles_m) {
        return;
    }
    const int64_t local_m0 = (int64_t) blockIdx.y * 16;
    if (local_m0 >= count) {
        return;
    }

    const int64_t tile_m = (int64_t) start + local_m0;
    const int64_t tile_n = (int64_t) blockIdx.x * 16;
    const int64_t row_a  = tile_m + lane_lo;
    const int64_t col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const bool row_valid = row_a < (int64_t) start + count;
        const int32_t * a_row = row_valid ? act_words + row_a*act_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        const int32_t * b_row = col_b < n_out ? weight_words + col_b*weight_word_s1 + expert*weight_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                b_frag[g] = b_row != nullptr ? b_row[word] : 0;
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t out_m = tile_m + 2 * r + lane_hi;
            const int64_t out_n = tile_n + lane_lo;
            if (out_m < (int64_t) start + count && out_n < n_out) {
                const float as = __half2float(act_scales[out_m*act_scale_s2 + kb]);
                const float bs = __half2float(weight_scales[out_n*weight_scale_s1 + expert*weight_scale_s2 + kb]);
                acc_f[r] += (float) acc_i[r] * as * bs;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int64_t out_m = tile_m + 2 * r + lane_hi;
        const int64_t out_n = tile_n + lane_lo;
        if (out_m < (int64_t) start + count && out_n < n_out) {
            dst[out_n + out_m*dst_s2] = acc_f[r];
        }
    }
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_iq4xs_native_kernel(
        const int32_t * __restrict__ weight_words,     // IQ4_XS table values as packed int8, [K/4, n_out, n_expert]
        const half    * __restrict__ weight_d_scales,  // IQ4_XS d, repeated per K32, [K/32, n_out, n_expert]
        const int8_t  * __restrict__ weight_subscales, // IQ4_XS (ls - 32), [K/32, n_out, n_expert]
        const int32_t * __restrict__ act_words,        // compact lanes [K/4, 1, n_lanes]
        const half    * __restrict__ act_scales,       // compact lane scales [K/32, 1, n_lanes]
        const int32_t * __restrict__ bounds,           // [2, n_expert]: start,count into compact lanes
        float         * __restrict__ dst,              // [n_out, 1, n_lanes] or [n_out,n_lanes] with dst_s2 lane stride
        int k_blocks,
        int64_t n_out,
        int64_t n_lanes,
        int64_t n_expert,
        int64_t max_tiles_m,
        int64_t weight_word_s1,
        int64_t weight_word_s2,
        int64_t act_word_s2,
        int64_t weight_scale_s1,
        int64_t weight_scale_s2,
        int64_t act_scale_s2,
        int64_t bounds_s0,
        int64_t bounds_s1,
        int64_t dst_s2) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int64_t expert = (int64_t) blockIdx.z;
    if (expert >= n_expert) {
        return;
    }

    const int32_t start = bounds[0*bounds_s0 + expert*bounds_s1];
    const int32_t count = bounds[1*bounds_s0 + expert*bounds_s1];
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes || (int64_t) blockIdx.y >= max_tiles_m) {
        return;
    }
    const int64_t local_m0 = (int64_t) blockIdx.y * 16;
    if (local_m0 >= count) {
        return;
    }

    const int64_t tile_m = (int64_t) start + local_m0;
    const int64_t tile_n = (int64_t) blockIdx.x * 16;
    const int64_t row_a  = tile_m + lane_lo;
    const int64_t col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const bool row_valid = row_a < (int64_t) start + count;
        const int32_t * a_row = row_valid ? act_words + row_a*act_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        const int32_t * b_row = col_b < n_out ? weight_words + col_b*weight_word_s1 + expert*weight_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                b_frag[g] = b_row != nullptr ? b_row[word] : 0;
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t out_m = tile_m + 2 * r + lane_hi;
            const int64_t out_n = tile_n + lane_lo;
            if (out_m < (int64_t) start + count && out_n < n_out) {
                const int64_t scale_idx = out_n*weight_scale_s1 + expert*weight_scale_s2 + kb;
                const int ss = (int) weight_subscales[scale_idx];
                const float d = __half2float(weight_d_scales[scale_idx]) * __half2float(act_scales[out_m*act_scale_s2 + kb]);
                acc_f[r] += d * (float) ((int) acc_i[r] * ss);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int64_t out_m = tile_m + 2 * r + lane_hi;
        const int64_t out_n = tile_n + lane_lo;
        if (out_m < (int64_t) start + count && out_n < n_out) {
            dst[out_n + out_m*dst_s2] = acc_f[r];
        }
    }
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_pack_f32_rows_kernel(
        const float * __restrict__ src,
        int32_t     * __restrict__ payload,
        half        * __restrict__ scales,
        int64_t src_s0,
        int64_t src_s1,
        int64_t payload_s1,
        int64_t scale_s1,
        int rows,
        int k_words) {
    const int lane = (int) threadIdx.x & 31;
    const int kb = (int) blockIdx.x;
    const int row = (int) blockIdx.y;
    const int k_blocks = k_words / GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;
    if (row >= rows || kb >= k_blocks) {
        return;
    }

    const float x = src[(int64_t)(kb*32 + lane)*src_s0 + (int64_t)row*src_s1];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;

    if (lane == 0) {
        scales[(int64_t)row*scale_s1 + kb] = __float2half(scale);
    }
    if ((lane & 3) == 0) {
        const int base = kb*32 + lane;
        const int q0 = max(-127, min(127, (int) roundf(src[(int64_t)(base + 0)*src_s0 + (int64_t)row*src_s1] / scale)));
        const int q1 = max(-127, min(127, (int) roundf(src[(int64_t)(base + 1)*src_s0 + (int64_t)row*src_s1] / scale)));
        const int q2 = max(-127, min(127, (int) roundf(src[(int64_t)(base + 2)*src_s0 + (int64_t)row*src_s1] / scale)));
        const int q3 = max(-127, min(127, (int) roundf(src[(int64_t)(base + 3)*src_s0 + (int64_t)row*src_s1] / scale)));
        payload[(int64_t)row*payload_s1 + kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS + (lane >> 2)] =
            (int32_t) (((uint32_t)(uint8_t)(int8_t)q0) | ((uint32_t)(uint8_t)(int8_t)q1 << 8) |
                       ((uint32_t)(uint8_t)(int8_t)q2 << 16) | ((uint32_t)(uint8_t)(int8_t)q3 << 24));
    }
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_pack_q8_0_experts_kernel(
        const block_q8_0 * __restrict__ src,
        int32_t          * __restrict__ payload,
        half             * __restrict__ scales,
        int64_t src_s1,
        int64_t src_s2,
        int n_out,
        int n_expert,
        int k_blocks) {
    const int word = (int) threadIdx.x;
    const int kb = (int) blockIdx.x;
    const int out = (int) blockIdx.y;
    const int expert = (int) blockIdx.z;
    if (word >= GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS || kb >= k_blocks || out >= n_out || expert >= n_expert) {
        return;
    }

    const block_q8_0 & block = src[(int64_t)expert*src_s2 + (int64_t)out*src_s1 + kb];
    const int k_words = k_blocks * GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;
    if (word == 0) {
        scales[((int64_t)expert*n_out + out)*k_blocks + kb] = block.d;
    }
    const int i = word * 4;
    payload[((int64_t)expert*n_out + out)*k_words + kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS + word] =
        (int32_t) (((uint32_t)(uint8_t)block.qs[i + 0]) | ((uint32_t)(uint8_t)block.qs[i + 1] << 8) |
                   ((uint32_t)(uint8_t)block.qs[i + 2] << 16) | ((uint32_t)(uint8_t)block.qs[i + 3] << 24));
}

static __device__ __forceinline__ int8_t ggml_cuda_rdna_i8_iq4nl_value(const int q) {
    constexpr int8_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return table[q & 15];
}

static __device__ __forceinline__ int ggml_cuda_rdna_i8_iq4xs_ls(const block_iq4_xs & block, const int ib) {
    return ((block.scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((block.scales_h >> (2*ib)) & 0x03) << 4);
}

static __device__ __forceinline__ int8_t ggml_cuda_rdna_i8_iq4xs_table_value(const block_iq4_xs & block, const int ib, const int lane) {
    const uint8_t qbyte = block.qs[ib*16 + (lane & 15)];
    const int q = lane < 16 ? (qbyte & 0x0f) : (qbyte >> 4);
    return ggml_cuda_rdna_i8_iq4nl_value(q);
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_pack_iq4_xs_native_experts_kernel(
        const block_iq4_xs * __restrict__ src,
        int32_t            * __restrict__ payload,
        half               * __restrict__ d_scales,
        int8_t             * __restrict__ subscales,
        int64_t src_s1,
        int64_t src_s2,
        int n_out,
        int n_expert,
        int k_blocks) {
    const int word = (int) threadIdx.x;
    const int kb = (int) blockIdx.x;
    const int out = (int) blockIdx.y;
    const int expert = (int) blockIdx.z;
    if (word >= GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS || kb >= k_blocks || out >= n_out || expert >= n_expert) {
        return;
    }

    const int iq_block = kb / (QK_K / 32);
    const int ib = kb - iq_block * (QK_K / 32);
    const block_iq4_xs & block = src[(int64_t)expert*src_s2 + (int64_t)out*src_s1 + iq_block];
    const int k_words = k_blocks * GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;
    const int64_t scale_idx = ((int64_t)expert*n_out + out)*k_blocks + kb;
    if (word == 0) {
        d_scales[scale_idx] = block.d;
        subscales[scale_idx] = (int8_t) (ggml_cuda_rdna_i8_iq4xs_ls(block, ib) - 32);
    }

    const int lane0 = word * 4;
    const int8_t q0 = ggml_cuda_rdna_i8_iq4xs_table_value(block, ib, lane0 + 0);
    const int8_t q1 = ggml_cuda_rdna_i8_iq4xs_table_value(block, ib, lane0 + 1);
    const int8_t q2 = ggml_cuda_rdna_i8_iq4xs_table_value(block, ib, lane0 + 2);
    const int8_t q3 = ggml_cuda_rdna_i8_iq4xs_table_value(block, ib, lane0 + 3);
    payload[((int64_t)expert*n_out + out)*k_words + kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS + word] =
        (int32_t) (((uint32_t)(uint8_t)q0) | ((uint32_t)(uint8_t)q1 << 8) |
                   ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24));
}

static __device__ __forceinline__ int ggml_cuda_rdna_i8_get_int_b2(const void * x, const int i32) {
    return ((const int *) x)[i32];
}

static __device__ __forceinline__ int ggml_cuda_rdna_i8_iq3s_scale(const block_iq3_s & block, const int ib) {
    return 1 + 2*((block.scales[ib/2] >> (4*(ib%2))) & 0x0f);
}

static __device__ __forceinline__ int32_t ggml_cuda_rdna_i8_iq3s_grid_word(const block_iq3_s & block, const int ib, const int word) {
    const int iqs = 2*ib;
    const int2 qs_packed = make_int2(ggml_cuda_rdna_i8_get_int_b2(block.qs, iqs + 0), ggml_cuda_rdna_i8_get_int_b2(block.qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = block.qh[ib];
    const int signs_packed_32 = ggml_cuda_rdna_i8_get_int_b2(block.signs, ib);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    const int l0 = word & ~1;
    const int grid = iq3s_grid[qs[word] | ((qh << (8 - word)) & 0x100)];
    const int sb = signs_packed_8[l0/2];
    const int signs = (word & 1) == 0 ?
        __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0c) << 21), 0x00000000) :
        __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xc0) << 17), 0x00000000);
    return __vsub4(grid ^ signs, signs);
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_q8_0_direct_kernel(
        const block_q8_0 * __restrict__ weights,    // original Q8_0 weights [K/32, n_out, n_expert]
        const int32_t    * __restrict__ act_words,  // compact lanes [K/4, 1, n_lanes]
        const half       * __restrict__ act_scales, // compact lane scales [K/32, 1, n_lanes]
        const int32_t    * __restrict__ bounds,     // [2, n_expert]: start,count into compact lanes
        float            * __restrict__ dst,        // [n_out, 1, n_lanes] or [n_out,n_lanes] with dst_s2 lane stride
        int k_blocks,
        int64_t n_out,
        int64_t n_lanes,
        int64_t n_expert,
        int64_t max_tiles_m,
        int64_t weight_block_s1,
        int64_t weight_block_s2,
        int64_t act_word_s2,
        int64_t act_scale_s2,
        int64_t bounds_s0,
        int64_t bounds_s1,
        int64_t dst_s2) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int64_t expert = (int64_t) blockIdx.z;
    if (expert >= n_expert) {
        return;
    }

    const int32_t start = bounds[0*bounds_s0 + expert*bounds_s1];
    const int32_t count = bounds[1*bounds_s0 + expert*bounds_s1];
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes || (int64_t) blockIdx.y >= max_tiles_m) {
        return;
    }
    const int64_t local_m0 = (int64_t) blockIdx.y * 16;
    if (local_m0 >= count) {
        return;
    }

    const int64_t tile_m = (int64_t) start + local_m0;
    const int64_t tile_n = (int64_t) blockIdx.x * 16;
    const int64_t row_a  = tile_m + lane_lo;
    const int64_t col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const bool row_valid = row_a < (int64_t) start + count;
        const block_q8_0 * b_block = col_b < n_out ? weights + expert*weight_block_s2 + col_b*weight_block_s1 + kb : nullptr;
        const int32_t * a_row = row_valid ? act_words + row_a*act_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                if (b_block != nullptr) {
                    const int i = word * 4;
                    b_frag[g] = (int32_t) (((uint32_t)(uint8_t)b_block->qs[i + 0]) | ((uint32_t)(uint8_t)b_block->qs[i + 1] << 8) |
                                           ((uint32_t)(uint8_t)b_block->qs[i + 2] << 16) | ((uint32_t)(uint8_t)b_block->qs[i + 3] << 24));
                } else {
                    b_frag[g] = 0;
                }
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t out_m = tile_m + 2 * r + lane_hi;
            const int64_t out_n = tile_n + lane_lo;
            if (out_m < (int64_t) start + count && out_n < n_out && b_block != nullptr) {
                const float d = __half2float(b_block->d) * __half2float(act_scales[out_m*act_scale_s2 + kb]);
                acc_f[r] += d * (float) acc_i[r];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int64_t out_m = tile_m + 2 * r + lane_hi;
        const int64_t out_n = tile_n + lane_lo;
        if (out_m < (int64_t) start + count && out_n < n_out) {
            dst[out_n + out_m*dst_s2] = acc_f[r];
        }
    }
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_iq4xs_direct_kernel(
        const block_iq4_xs * __restrict__ weights,    // original IQ4_XS weights [K/256, n_out, n_expert]
        const int32_t      * __restrict__ act_words,  // compact lanes [K/4, 1, n_lanes]
        const half         * __restrict__ act_scales, // compact lane scales [K/32, 1, n_lanes]
        const int32_t      * __restrict__ bounds,     // [2, n_expert]: start,count into compact lanes
        float              * __restrict__ dst,        // [n_out, 1, n_lanes] or [n_out,n_lanes] with dst_s2 lane stride
        int k_blocks,
        int64_t n_out,
        int64_t n_lanes,
        int64_t n_expert,
        int64_t max_tiles_m,
        int64_t weight_block_s1,
        int64_t weight_block_s2,
        int64_t act_word_s2,
        int64_t act_scale_s2,
        int64_t bounds_s0,
        int64_t bounds_s1,
        int64_t dst_s2) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int64_t expert = (int64_t) blockIdx.z;
    if (expert >= n_expert) {
        return;
    }

    const int32_t start = bounds[0*bounds_s0 + expert*bounds_s1];
    const int32_t count = bounds[1*bounds_s0 + expert*bounds_s1];
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes || (int64_t) blockIdx.y >= max_tiles_m) {
        return;
    }
    const int64_t local_m0 = (int64_t) blockIdx.y * 16;
    if (local_m0 >= count) {
        return;
    }

    const int64_t tile_m = (int64_t) start + local_m0;
    const int64_t tile_n = (int64_t) blockIdx.x * 16;
    const int64_t row_a  = tile_m + lane_lo;
    const int64_t col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const bool row_valid = row_a < (int64_t) start + count;
        const int iq_block = kb / (QK_K / 32);
        const int ib = kb - iq_block * (QK_K / 32);
        const block_iq4_xs * b_block = col_b < n_out ? weights + expert*weight_block_s2 + col_b*weight_block_s1 + iq_block : nullptr;
        const int32_t * a_row = row_valid ? act_words + row_a*act_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                if (b_block != nullptr) {
                    const int lane0 = word * 4;
                    const int8_t q0 = ggml_cuda_rdna_i8_iq4xs_table_value(*b_block, ib, lane0 + 0);
                    const int8_t q1 = ggml_cuda_rdna_i8_iq4xs_table_value(*b_block, ib, lane0 + 1);
                    const int8_t q2 = ggml_cuda_rdna_i8_iq4xs_table_value(*b_block, ib, lane0 + 2);
                    const int8_t q3 = ggml_cuda_rdna_i8_iq4xs_table_value(*b_block, ib, lane0 + 3);
                    b_frag[g] = (int32_t) (((uint32_t)(uint8_t)q0) | ((uint32_t)(uint8_t)q1 << 8) |
                                           ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24));
                } else {
                    b_frag[g] = 0;
                }
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t out_m = tile_m + 2 * r + lane_hi;
            const int64_t out_n = tile_n + lane_lo;
            if (out_m < (int64_t) start + count && out_n < n_out && b_block != nullptr) {
                const int ss = ggml_cuda_rdna_i8_iq4xs_ls(*b_block, ib) - 32;
                const float d = __half2float(b_block->d) * __half2float(act_scales[out_m*act_scale_s2 + kb]);
                acc_f[r] += d * (float) ((int) acc_i[r] * ss);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int64_t out_m = tile_m + 2 * r + lane_hi;
        const int64_t out_n = tile_n + lane_lo;
        if (out_m < (int64_t) start + count && out_n < n_out) {
            dst[out_n + out_m*dst_s2] = acc_f[r];
        }
    }
}

static __global__ __launch_bounds__(32, 8) void ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_iq3s_direct_kernel(
        const block_iq3_s * __restrict__ weights,    // original IQ3_S weights [K/256, n_out, n_expert]
        const int32_t     * __restrict__ act_words,  // compact lanes [K/4, 1, n_lanes]
        const half        * __restrict__ act_scales, // compact lane scales [K/32, 1, n_lanes]
        const int32_t     * __restrict__ bounds,     // [2, n_expert]: start,count into compact lanes
        float             * __restrict__ dst,        // [n_out, 1, n_lanes] or [n_out,n_lanes] with dst_s2 lane stride
        int k_blocks,
        int64_t n_out,
        int64_t n_lanes,
        int64_t n_expert,
        int64_t max_tiles_m,
        int64_t weight_block_s1,
        int64_t weight_block_s2,
        int64_t act_word_s2,
        int64_t act_scale_s2,
        int64_t bounds_s0,
        int64_t bounds_s1,
        int64_t dst_s2) {
    const int lane = (int) threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    const int64_t expert = (int64_t) blockIdx.z;
    if (expert >= n_expert) {
        return;
    }

    const int32_t start = bounds[0*bounds_s0 + expert*bounds_s1];
    const int32_t count = bounds[1*bounds_s0 + expert*bounds_s1];
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes || (int64_t) blockIdx.y >= max_tiles_m) {
        return;
    }
    const int64_t local_m0 = (int64_t) blockIdx.y * 16;
    if (local_m0 >= count) {
        return;
    }

    const int64_t tile_m = (int64_t) start + local_m0;
    const int64_t tile_n = (int64_t) blockIdx.x * 16;
    const int64_t row_a  = tile_m + lane_lo;
    const int64_t col_b  = tile_n + lane_lo;

    float acc_f[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < k_blocks; ++kb) {
        const bool row_valid = row_a < (int64_t) start + count;
        const int iq_block = kb / (QK_K / 32);
        const int ib = kb - iq_block * (QK_K / 32);
        const block_iq3_s * b_block = col_b < n_out ? weights + expert*weight_block_s2 + col_b*weight_block_s1 + iq_block : nullptr;
        const int32_t * a_row = row_valid ? act_words + row_a*act_word_s2 + (int64_t) kb*GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS : nullptr;
        ggml_cuda_rdna_i8_p16_v8i32 acc_i = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int half_k32 = 0; half_k32 < 2; ++half_k32) {
            ggml_cuda_rdna_i8_p16_v4i32 a_frag;
            ggml_cuda_rdna_i8_p16_v4i32 b_frag;
#pragma unroll
            for (int g = 0; g < (int) DP16_PACKED_I8X16_WORDS; ++g) {
                const int word = half_k32 * (int) DP16_PACKED_I8X16_WORDS + g;
                a_frag[g] = a_row != nullptr ? a_row[word] : 0;
                b_frag[g] = b_block != nullptr ? ggml_cuda_rdna_i8_iq3s_grid_word(*b_block, ib, word) : 0;
            }
            acc_i = ggml_cuda_rdna_i8_packed16_wmma_i32_16x16x16(a_frag, b_frag, acc_i);
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int64_t out_m = tile_m + 2 * r + lane_hi;
            const int64_t out_n = tile_n + lane_lo;
            if (out_m < (int64_t) start + count && out_n < n_out && b_block != nullptr) {
                const int ss = ggml_cuda_rdna_i8_iq3s_scale(*b_block, ib);
                const float d = __half2float(b_block->d) * __half2float(act_scales[out_m*act_scale_s2 + kb]);
                acc_f[r] += d * (float) ((int) acc_i[r] * ss);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int64_t out_m = tile_m + 2 * r + lane_hi;
        const int64_t out_n = tile_n + lane_lo;
        if (out_m < (int64_t) start + count && out_n < n_out) {
            dst[out_n + out_m*dst_s2] = acc_f[r];
        }
    }
}

static size_t ggml_cuda_rdna_i8_packed16_moe_max_temp_bytes() {
    const char * env = std::getenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_MAX_TEMP_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return (size_t) 256u << 20;
    }
    return (size_t) std::strtoull(env, nullptr, 10);
}

} // namespace

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_should_use_rdna_i8_packed16_gemm(
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
    static const bool enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_GEMM");
    if (!enabled) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return false;
    }
    if (src0->type != GGML_TYPE_I32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] <= 0 || src0->ne[1] <= 0 || src1->ne[1] <= 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1]) {
        return false;
    }
    if (dst->nb[1] % (int64_t) sizeof(float) != 0) {
        return false;
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src1->ne[1] > std::numeric_limits<int>::max() || dst->nb[1] / (int64_t) sizeof(float) > std::numeric_limits<int>::max()) {
        return false;
    }
    const ggml_tensor * b_scales = ggml_cuda_rdna_i8_packed16_scales(src0);
    const ggml_tensor * a_scales = ggml_cuda_rdna_i8_packed16_scales(src1);
    if (!ggml_cuda_rdna_i8_packed16_valid_operand(src0, b_scales, src0->ne[1], src0->ne[0]) ||
            !ggml_cuda_rdna_i8_packed16_valid_operand(src1, a_scales, src1->ne[1], src1->ne[0])) {
        return false;
    }
    if (const char * only = std::getenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return false;
        }
    }
    return true;
#endif
}

bool ggml_cuda_mul_mat_rdna_i8_packed16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_should_use_rdna_i8_packed16_gemm(src0, src1, dst, cc)) {
        return false;
    }

    const ggml_tensor * b_scales = ggml_cuda_rdna_i8_packed16_scales(src0);
    const ggml_tensor * a_scales = ggml_cuda_rdna_i8_packed16_scales(src1);
    GGML_ASSERT(b_scales != nullptr && a_scales != nullptr);

    const int k_words  = (int) src0->ne[0];
    const int n        = (int) src0->ne[1];
    const int m        = (int) src1->ne[1];
    const int k_blocks = k_words / GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS;

    const int b_word_stride  = (int) (src0->nb[1] / sizeof(int32_t));
    const int a_word_stride  = (int) (src1->nb[1] / sizeof(int32_t));
    const int b_scale_stride = (int) (b_scales->nb[1] / sizeof(half));
    const int a_scale_stride = (int) (a_scales->nb[1] / sizeof(half));
    const int dst_stride     = (int) (dst->nb[1] / sizeof(float));

    const dim3 block(32, 1, 1);
    const dim3 grid((n + 15) / 16, (m + 15) / 16, 1);
    if (ggml_cuda_rdna_i8_packed16_log_enabled()) {
        GGML_LOG_INFO("%s: route=rdna_i8_packed16_gemm tensor=%s dst=%s m=%d n=%d k_words=%d k32=%d direct_payload=1 quantize=0 restage=0 grid=(%u,%u,%u)\n",
                __func__, src0->name, dst->name, m, n, k_words, k_blocks, grid.x, grid.y, grid.z);
    }
    ggml_cuda_rdna_i8_packed16_gemm_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const int32_t *) src1->data,
            (const half *) a_scales->data,
            (const int32_t *) src0->data,
            (const half *) b_scales->data,
            (float *) dst->data,
            m, n, k_blocks,
            a_word_stride, b_word_stride,
            a_scale_stride, b_scale_stride,
            dst_stride);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}

bool ggml_cuda_should_use_rdna_i8_packed16_gemm_id(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
#else
    static const bool enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID");
    if (!enabled) {
        return false;
    }
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || dst == nullptr) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) {
        return false;
    }
    if (src0->type != GGML_TYPE_I32 || src1->type != GGML_TYPE_I32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[2] <= 0 || src1->ne[1] <= 0 || src1->ne[2] <= 0) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (ids->ne[1] != src1->ne[2] || ids->ne[0] % src1->ne[1] != 0) {
        return false;
    }
    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != src1->ne[2]) {
        return false;
    }
    if (ids->nb[0] % (int64_t) sizeof(int32_t) != 0 || ids->nb[1] % (int64_t) sizeof(int32_t) != 0 ||
            dst->nb[1] % (int64_t) sizeof(float) != 0 || dst->nb[2] % (int64_t) sizeof(float) != 0) {
        return false;
    }
    if (src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] > std::numeric_limits<int>::max() ||
            src0->ne[2] > std::numeric_limits<int>::max() || src1->ne[1] > std::numeric_limits<int>::max() ||
            src1->ne[2] > std::numeric_limits<int>::max() || ids->ne[0] > std::numeric_limits<int>::max()) {
        return false;
    }
    if (dst->nb[1] / (int64_t) sizeof(float) > std::numeric_limits<int>::max() ||
            dst->nb[2] / (int64_t) sizeof(float) > std::numeric_limits<int>::max()) {
        return false;
    }
    const int64_t total = dst->ne[0]*dst->ne[1]*dst->ne[2];
    if (total <= 0 || total > (int64_t) std::numeric_limits<int>::max() * 256) {
        return false;
    }

    const ggml_tensor * weight_scales = ggml_cuda_rdna_i8_packed16_id_scales(src0, dst, 3);
    const ggml_tensor * act_scales    = ggml_cuda_rdna_i8_packed16_id_scales(src1, dst, 4);
    if (!ggml_cuda_rdna_i8_packed16_valid_operand_3d(src0, weight_scales, src0->ne[1], src0->ne[2], src0->ne[0]) ||
            !ggml_cuda_rdna_i8_packed16_valid_operand_3d(src1, act_scales, src1->ne[1], src1->ne[2], src1->ne[0])) {
        return false;
    }
    if (const char * only = std::getenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_TENSOR")) {
        if (only[0] != '\0' && std::strstr(src0->name, only) == nullptr && std::strstr(dst->name, only) == nullptr) {
            return false;
        }
    }
    return true;
#endif
}

bool ggml_cuda_mul_mat_id_rdna_i8_packed16(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_should_use_rdna_i8_packed16_gemm_id(src0, src1, ids, dst, cc)) {
        return false;
    }

    const ggml_tensor * weight_scales = ggml_cuda_rdna_i8_packed16_id_scales(src0, dst, 3);
    const ggml_tensor * act_scales    = ggml_cuda_rdna_i8_packed16_id_scales(src1, dst, 4);
    GGML_ASSERT(weight_scales != nullptr && act_scales != nullptr);

    const int64_t total = dst->ne[0]*dst->ne[1]*dst->ne[2];
    const int k_blocks = (int) (src0->ne[0] / GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS);

    const ggml_tensor * bounds = dst->src[5];
    if (ggml_cuda_rdna_i8_packed16_valid_id_bounds(src0, src1, ids, dst, bounds)) {
        const dim3 block(32, 1, 1);
        const int64_t max_tiles_m = (src1->ne[2] + 15) / 16;
        const dim3 grid((unsigned int) ((dst->ne[0] + 15) / 16), (unsigned int) max_tiles_m, (unsigned int) src0->ne[2]);
        if (ggml_cuda_rdna_i8_packed16_id_log_enabled()) {
            GGML_LOG_INFO("%s: route=rdna_i8_packed16_gemm_id_bounds_wmma tensor=%s dst=%s n_out=%lld n_lanes=%lld n_expert=%lld k_words=%lld k32=%d direct_payload=1 quantize=0 dequantize=0 restage=0 scalar_direct=0 grid=(%u,%u,%u) sidecars=dst_src3_4 bounds=dst_src5\n",
                    __func__, src0->name, dst->name, (long long) dst->ne[0], (long long) dst->ne[2],
                    (long long) src0->ne[2], (long long) src0->ne[0], k_blocks, grid.x, grid.y, grid.z);
        }
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));
        ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_kernel<<<grid, block, 0, ctx.stream()>>>(
                (const int32_t *) src0->data,
                (const half *) weight_scales->data,
                (const int32_t *) src1->data,
                (const half *) act_scales->data,
                (const int32_t *) bounds->data,
                (float *) dst->data,
                k_blocks,
                dst->ne[0], src1->ne[2], src0->ne[2], max_tiles_m,
                src0->nb[1] / (int64_t) sizeof(int32_t), src0->nb[2] / (int64_t) sizeof(int32_t),
                src1->nb[2] / (int64_t) sizeof(int32_t),
                weight_scales->nb[1] / (int64_t) sizeof(half), weight_scales->nb[2] / (int64_t) sizeof(half),
                act_scales->nb[2] / (int64_t) sizeof(half),
                bounds->nb[0] / (int64_t) sizeof(int32_t), bounds->nb[1] / (int64_t) sizeof(int32_t),
                dst->nb[2] / (int64_t) sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    const int threads = 256;
    const dim3 block(threads, 1, 1);
    const dim3 grid((unsigned int) ((total + threads - 1) / threads), 1, 1);

    if (ggml_cuda_rdna_i8_packed16_id_log_enabled()) {
        GGML_LOG_INFO("%s: route=rdna_i8_packed16_gemm_id tensor=%s dst=%s n_out=%lld n_ids=%lld n_tokens=%lld n_expert=%lld k_words=%lld k32=%d direct_payload=1 quantize=0 dequantize=0 restage=0 scalar_direct=1 grid=(%u,%u,%u) sidecars=dst_src3_4\n",
                __func__, src0->name, dst->name, (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
                (long long) src0->ne[2], (long long) src0->ne[0], k_blocks, grid.x, grid.y, grid.z);
    }

    ggml_cuda_rdna_i8_packed16_gemm_id_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const int32_t *) src0->data,
            (const half *) weight_scales->data,
            (const int32_t *) src1->data,
            (const half *) act_scales->data,
            (const int32_t *) ids->data,
            (float *) dst->data,
            total, k_blocks,
            dst->ne[0], dst->ne[1], dst->ne[2], src1->ne[1], src0->ne[2],
            src0->nb[1] / (int64_t) sizeof(int32_t), src0->nb[2] / (int64_t) sizeof(int32_t),
            src1->nb[1] / (int64_t) sizeof(int32_t), src1->nb[2] / (int64_t) sizeof(int32_t),
            weight_scales->nb[1] / (int64_t) sizeof(half), weight_scales->nb[2] / (int64_t) sizeof(half),
            act_scales->nb[1] / (int64_t) sizeof(half), act_scales->nb[2] / (int64_t) sizeof(half),
            ids->nb[0] / (int64_t) sizeof(int32_t), ids->nb[1] / (int64_t) sizeof(int32_t),
            dst->nb[1] / (int64_t) sizeof(float), dst->nb[2] / (int64_t) sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}

bool ggml_cuda_should_use_rdna_i8_packed16_moe_projection(
        const ggml_tensor * weights,
        const ggml_tensor * compact,
        const ggml_tensor * lanes,
        const ggml_tensor * bounds,
        const ggml_tensor * dst,
        int cc) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(weights); GGML_UNUSED(compact); GGML_UNUSED(lanes); GGML_UNUSED(bounds); GGML_UNUSED(dst); GGML_UNUSED(cc);
    return false;
#else
    static const bool q8_enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_Q8_0");
    static const bool iq4xs_native_enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_IQ4_XS_NATIVE");
    static const bool iq3s_native_enabled = ggml_cuda_rdna_i8_packed16_env_enabled("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_IQ3_S_NATIVE");
    if (weights == nullptr || compact == nullptr || lanes == nullptr || bounds == nullptr || dst == nullptr) return false;
    if (!GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA3) return false;
    const bool weights_q8_0 = weights->type == GGML_TYPE_Q8_0;
    const bool weights_iq4_xs = weights->type == GGML_TYPE_IQ4_XS;
    const bool weights_iq3_s = weights->type == GGML_TYPE_IQ3_S;
    if ((!weights_q8_0 || !q8_enabled) && (!weights_iq4_xs || !iq4xs_native_enabled) && (!weights_iq3_s || !iq3s_native_enabled)) return false;
    if (compact->type != GGML_TYPE_F32 || lanes->type != GGML_TYPE_I32 || bounds->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) return false;
    if (bounds->op != GGML_OP_MOE_ROUTED_LANES_EXPERT_BOUNDS || bounds->src[0] != lanes) return false;

    const int64_t n_in = compact->ne[0], n_lanes = compact->ne[1], n_out = weights->ne[1], n_expert = weights->ne[2];
    if (n_in <= 0 || n_out <= 0 || n_lanes <= 0 || n_expert <= 0 || n_expert > 4096) return false;
    if (weights->ne[0] != n_in || weights->ne[3] != 1 || compact->ne[2] != 1 || compact->ne[3] != 1 ||
            dst->ne[0] != n_out || dst->ne[1] != n_lanes || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            lanes->ne[0] != 4 || lanes->ne[1] != n_lanes + n_expert || bounds->ne[0] != 2 || bounds->ne[1] != n_expert || bounds->ne[2] != 1 || bounds->ne[3] != 1) return false;
    if (n_in % ((weights_iq4_xs || weights_iq3_s) ? QK_K : QK8_0) != 0) return false;
    const int64_t k_words = n_in / 4, k_blocks = n_in / 32;
    if (k_words <= 0 || (k_words % GGML_CUDA_RDNA_I8_PACKED16_K32_WORDS) != 0 || k_blocks <= 0) return false;
    if (compact->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) return false;
    if (compact->nb[1] % (int64_t) sizeof(float) != 0 || weights->nb[1] % (int64_t) ggml_type_size(weights->type) != 0 ||
            weights->nb[2] % (int64_t) ggml_type_size(weights->type) != 0 || bounds->nb[0] % (int64_t) sizeof(int32_t) != 0 ||
            bounds->nb[1] % (int64_t) sizeof(int32_t) != 0 || dst->nb[1] % (int64_t) sizeof(float) != 0) return false;
    if (n_in > std::numeric_limits<int>::max() || n_out > std::numeric_limits<int>::max() || n_lanes > std::numeric_limits<int>::max() || n_expert > std::numeric_limits<int>::max()) return false;

    const size_t aw = (size_t)n_lanes*(size_t)k_words;
    const size_t as = (size_t)n_lanes*(size_t)k_blocks;
    if (aw > ((size_t)-1)/sizeof(int32_t) || as > ((size_t)-1)/sizeof(half)) return false;
    const size_t temp_bytes = aw*sizeof(int32_t) + as*sizeof(half);
    return temp_bytes <= ggml_cuda_rdna_i8_packed16_moe_max_temp_bytes();
#endif
}

bool ggml_cuda_moe_routed_lanes_projection_rdna_i8_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx); GGML_UNUSED(dst); return false;
#else
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * compact = dst->src[1];
    const ggml_tensor * lanes = dst->src[2];
    const ggml_tensor * bounds = dst->src[3];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!ggml_cuda_should_use_rdna_i8_packed16_moe_projection(weights, compact, lanes, bounds, dst, cc)) return false;

    const int64_t n_in = compact->ne[0], n_lanes = compact->ne[1], n_out = weights->ne[1], n_expert = weights->ne[2];
    const int64_t k_words = n_in / 4, k_blocks = n_in / 32;
    const bool weights_iq4_xs = weights->type == GGML_TYPE_IQ4_XS;
    const bool weights_iq3_s = weights->type == GGML_TYPE_IQ3_S;
    const size_t aw = (size_t)n_lanes*(size_t)k_words;
    const size_t as = (size_t)n_lanes*(size_t)k_blocks;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int32_t> act_words_alloc(pool, aw);
    ggml_cuda_pool_alloc<half>    act_scales_alloc(pool, as);
    int32_t * act_words = act_words_alloc.get();
    half * act_scales = act_scales_alloc.get();
    if (act_words == nullptr || act_scales == nullptr) return false;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_rdna_i8_pack_f32_rows_kernel<<<dim3((unsigned)k_blocks, (unsigned)n_lanes, 1), dim3(32,1,1), 0, stream>>>(
            (const float *) compact->data, act_words, act_scales,
            compact->nb[0] / (int64_t) sizeof(float), compact->nb[1] / (int64_t) sizeof(float), k_words, k_blocks,
            (int)n_lanes, (int)k_words);
    CUDA_CHECK(cudaGetLastError());

    const int64_t max_tiles_m = (n_lanes + 15) / 16;
    const dim3 block(32,1,1);
    const dim3 grid((unsigned)((n_out + 15) / 16), (unsigned)max_tiles_m, (unsigned)n_expert);
    if (ggml_cuda_rdna_i8_packed16_id_log_enabled() || ggml_cuda_rdna_i8_packed16_log_enabled()) {
        const size_t temp_bytes = aw*sizeof(int32_t) + as*sizeof(half);
        GGML_LOG_INFO("%s: route=%s tensor=%s dst=%s n_in=%lld n_out=%lld n_lanes=%lld n_expert=%lld k_words=%lld k32=%lld temp_bytes=%zu direct_payload=1 quantize=producer_f32_to_i8 weight_payload=%s weight_scale=%s dequantize=0 requantize=0 restage=%s grid=(%u,%u,%u) bounds=src3\n",
                __func__, weights_iq4_xs ? "rdna_i8_packed16_moe_projection_iq4_xs_native" : (weights_iq3_s ? "rdna_i8_packed16_moe_projection_iq3_s_native" : "rdna_i8_packed16_moe_projection_q8_0"),
                weights->name, dst->name, (long long)n_in, (long long)n_out, (long long)n_lanes, (long long)n_expert,
                (long long)k_words, (long long)k_blocks, temp_bytes,
                weights_iq4_xs ? "iq4xs_kvalues_i8_direct" : (weights_iq3_s ? "iq3s_grid_i8_direct" : "q8_0_qs_i8_direct"),
                weights_iq4_xs ? "d_f16_plus_ls_minus_32_i8_direct" : (weights_iq3_s ? "d_f16_plus_iq3s_odd_scale_direct" : "d_f16_direct"),
                "activation_pool_only", grid.x, grid.y, grid.z);
    }
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    if (weights_iq4_xs) {
        ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_iq4xs_direct_kernel<<<grid, block, 0, stream>>>(
                (const block_iq4_xs *) weights->data, act_words, act_scales, (const int32_t *)bounds->data, (float *)dst->data,
                (int)k_blocks, n_out, n_lanes, n_expert, max_tiles_m,
                weights->nb[1] / (int64_t) ggml_type_size(weights->type), weights->nb[2] / (int64_t) ggml_type_size(weights->type), k_words,
                k_blocks,
                bounds->nb[0] / (int64_t) sizeof(int32_t), bounds->nb[1] / (int64_t) sizeof(int32_t),
                dst->nb[1] / (int64_t) sizeof(float));
    } else if (weights_iq3_s) {
        ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_iq3s_direct_kernel<<<grid, block, 0, stream>>>(
                (const block_iq3_s *) weights->data, act_words, act_scales, (const int32_t *)bounds->data, (float *)dst->data,
                (int)k_blocks, n_out, n_lanes, n_expert, max_tiles_m,
                weights->nb[1] / (int64_t) ggml_type_size(weights->type), weights->nb[2] / (int64_t) ggml_type_size(weights->type), k_words,
                k_blocks,
                bounds->nb[0] / (int64_t) sizeof(int32_t), bounds->nb[1] / (int64_t) sizeof(int32_t),
                dst->nb[1] / (int64_t) sizeof(float));
    } else {
        ggml_cuda_rdna_i8_packed16_gemm_id_bounds_wmma_q8_0_direct_kernel<<<grid, block, 0, stream>>>(
                (const block_q8_0 *) weights->data, act_words, act_scales, (const int32_t *)bounds->data, (float *)dst->data,
                (int)k_blocks, n_out, n_lanes, n_expert, max_tiles_m,
                weights->nb[1] / (int64_t) ggml_type_size(weights->type), weights->nb[2] / (int64_t) ggml_type_size(weights->type), k_words,
                k_blocks,
                bounds->nb[0] / (int64_t) sizeof(int32_t), bounds->nb[1] / (int64_t) sizeof(int32_t),
                dst->nb[1] / (int64_t) sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}
