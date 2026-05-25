#include "fattn-dot4-q8k-kq.cuh"

#include <cstdlib>

#ifdef GGML_USE_HIP

static constexpr int GGML_CUDA_Q8K_DOT4_KQ_D = 256;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_BLOCKS = GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_PACKED16_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 16;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_M = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_N = 16;

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

static inline int ggml_cuda_q8k_dot4_kq_env_int(const char * name, int fallback) {
    const char * env = getenv(name);
    return env ? atoi(env) : fallback;
}

static inline void ggml_cuda_q8k_dot4_kq_event_create(hipEvent_t * event) {
    CUDA_CHECK(hipEventCreate(event));
}

static inline void ggml_cuda_q8k_dot4_kq_event_destroy(hipEvent_t event) {
    if (event != nullptr) {
        CUDA_CHECK(hipEventDestroy(event));
    }
}

static inline float ggml_cuda_q8k_dot4_kq_event_elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return ggml_cuda_dp4a(a, b, c);
#endif
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_q8k_dot4_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_4chain(
        const int * __restrict__ q_payload,
        const int * __restrict__ k_payload,
        int idx) {
    int acc = 0;
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 0], k_payload[idx + 0], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 1], k_payload[idx + 1], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 2], k_payload[idx + 2], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 3], k_payload[idx + 3], acc);
    return acc;
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_q_packed16_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ q_payload,
        float       * __restrict__ q_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nq,
        int n_heads_q,
        int batch) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (q >= nq || hq >= n_heads_q || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q) * nb01);
    const int d = q_block * QK8_0 + lane;
    const float x = q_ptr[d];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q);
    ((int8_t *)(q_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        q_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = scale;
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_pack_k_packed16_kernel(
        const char * __restrict__ K,
        int        * __restrict__ k_payload,
        half       * __restrict__ k_scales,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nk,
        int n_heads_k,
        int batch) {
    const int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = batch * n_heads_k * nk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    if (linear >= total) {
        return;
    }

    const int qblk = linear % GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int t = linear / GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int k = t % nk;
    const int hk_b = t / nk;
    const int hk = hk_b % n_heads_k;
    const int b = hk_b / n_heads_k;
    const char * src = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k) * nb11 + int64_t(qblk) * nb10;
    int * dst = k_payload + (size_t(t) * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4));

    k_scales[size_t(t) * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_q8k_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        dst[i] = ggml_cuda_q8k_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ logits,
        float scale,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int half_tile = (tid >> 4) & 1;
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const int idx = qb * (QK8_0 / 4) + half_tile * 4;
        const int acc = ggml_cuda_q8k_dot4_4chain(q_row_payload, k_row_payload, idx);
        sum += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
    }
    sum += __shfl_xor(sum, 16, 32);
    if (half_tile == 0) {
        logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    }
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_q8block_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const char  * __restrict__ K,
        float       * __restrict__ logits,
        float scale,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int half_tile = (tid >> 4) & 1;
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const char * k_blk = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k_row) * nb11 + int64_t(qb) * nb10;
        const half ks_h = ggml_cuda_q8k_dot4_load_half_unaligned(k_blk);
        const int * q_blk = q_row_payload + qb * (QK8_0 / 4);
        const int idx = half_tile * 4;
        int k_i32[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            k_i32[i] = ggml_cuda_q8k_dot4_load_i32_unaligned(k_blk + sizeof(half) + 4 * (idx + i));
        }
        int acc = 0;
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 0], k_i32[0], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 1], k_i32[1], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 2], k_i32[2], acc);
        acc = ggml_cuda_q8k_dot4_i8_i8(q_blk[idx + 3], k_i32[3], acc);
        sum += float(acc) * q_row_scales[qb] * __half2float(ks_h);
    }
    sum += __shfl_xor(sum, 16, 32);
    if (half_tile == 0) {
        logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    }
    GGML_UNUSED(n_heads_k);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_q4_0(
        const char * __restrict__ V,
        int64_t nb20,
        int i) {
    const int ib = i / QK4_0;
    const int iq = i & 15;
    const int shift = (i & 31) >= 16 ? 4 : 0;
    const block_q4_0 * v = (const block_q4_0 *) (V + int64_t(ib) * nb20);
    const int q = (v->qs[iq] >> shift) & 0x0f;
    return (float(q) - 8.0f) * __half2float(v->d);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_mask_value(
        const char * __restrict__ mask,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int q,
        int k,
        int b) {
    if (!mask) {
        return 0.0f;
    }
    const char * p = mask + int64_t(b % ne33) * nb33 + int64_t(q) * nb31 + int64_t(k) * nb30;
    return __half2float(*(const half *) p);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel(
        const float * __restrict__ logits,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const float s = logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k] +
            ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out / denom;
    GGML_UNUSED(n_heads_k);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_reference_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ K,
        float       * __restrict__ logits,
        float scale,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q_row) * nb01);

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const char * k_blk = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k_row) * nb11 + int64_t(qb) * nb10;
        const half ks_h = ggml_cuda_q8k_dot4_load_half_unaligned(k_blk);
        const float ks = __half2float(ks_h);
#pragma unroll
        for (int lane = 0; lane < QK8_0; ++lane) {
            const int8_t kval = *(const int8_t *) (k_blk + sizeof(half) + lane);
            sum += q_ptr[qb * QK8_0 + lane] * float(kval) * ks;
        }
    }
    logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    GGML_UNUSED(n_heads_k);
}

static __global__ void ggml_cuda_q8k_dot4_kq_error_kernel(
        const float * __restrict__ candidate,
        const float * __restrict__ reference,
        float       * __restrict__ metrics,
        size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float diff = fabsf(candidate[i] - reference[i]);
    atomicAdd(metrics + 0, diff * diff);
    atomicAdd(metrics + 1, diff);
    atomicAdd(metrics + 2, diff > 0.5f ? 1.0f : 0.0f);
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_payload(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);
    ggml_cuda_pool_alloc<int>   k_payload(pool);
    ggml_cuda_pool_alloc<half>  k_scales(pool);
    ggml_cuda_pool_alloc<float> logits(pool);
    ggml_cuda_pool_alloc<float> ref_logits(pool);
    ggml_cuda_pool_alloc<float> metrics(pool);

    const size_t q_rows = (size_t) batch * n_heads_q * nq;
    const size_t k_rows = (size_t) batch * n_heads_k * nk;
    const size_t logits_ne = (size_t) batch * n_heads_q * nq * nk;
    const char * check_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_CHECK");
    const bool check = check_env && atoi(check_env) != 0;
    const char * variant_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT");
    const bool q8block_variant = variant_env && strcmp(variant_env, "q8block") == 0;
    const bool full_fa = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA");

    if (full_fa) {
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
            GGML_ABORT("q8k_dot4_kq full FA probe does not support sinks, max_bias, or logit_softcap");
        }
        if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
                mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
            GGML_ABORT("q8k_dot4_kq full FA probe mask shape/type unsupported");
        }
    }

    const bool timing_requested = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING");
    const int timing_every = max(1, ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING_EVERY", 1));
    static unsigned long long timing_counter = 0;
    const unsigned long long timing_call = timing_requested ? ++timing_counter : 0;
    const bool timing = timing_requested && (timing_call % (unsigned long long) timing_every) == 0;

    q_payload.alloc(q_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
    q_scales.alloc(q_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
    k_payload.alloc(k_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
    k_scales.alloc(k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
    logits.alloc(logits_ne);

    cudaStream_t stream = ctx.stream();
    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_q_quant = nullptr;
    hipEvent_t ev_k_pack = nullptr;
    hipEvent_t ev_kq = nullptr;
    hipEvent_t ev_ref = nullptr;
    hipEvent_t ev_err = nullptr;
    hipEvent_t ev_zero = nullptr;
    if (timing) {
        ggml_cuda_q8k_dot4_kq_event_create(&ev_start);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_kq);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_ref);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_err);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_zero);
        CUDA_CHECK(hipEventRecord(ev_start, stream));
    }

    dim3 q_grid(nq, n_heads_q, batch);
    dim3 block(256);
    ggml_cuda_q8k_dot4_quant_q_packed16_kernel<<<q_grid, block, 0, stream>>>(
        (const float *) Q->data, q_payload.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, batch);
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_q_quant, stream));
    }

    if (!q8block_variant) {
        dim3 pack_grid((k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + 255) / 256);
        ggml_cuda_q8k_dot4_pack_k_packed16_kernel<<<pack_grid, block, 0, stream>>>(
            (const char *) K->data, k_payload.ptr, k_scales.ptr,
            K->nb[0], K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch);
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_k_pack, stream));
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    dim3 kq_grid((nk + GGML_CUDA_Q8K_DOT4_KQ_TILE_N - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_N,
                 (nq + GGML_CUDA_Q8K_DOT4_KQ_TILE_M - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_M,
                 n_heads_q * batch);
    if (q8block_variant) {
        ggml_cuda_q8k_dot4_kq_q8block_kernel<<<kq_grid, block, 0, stream>>>(
            q_payload.ptr, q_scales.ptr, (const char *) K->data, logits.ptr, scale,
            K->nb[0], K->nb[1], K->nb[2], K->nb[3], nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
    } else {
        ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, logits.ptr, scale,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
    }
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_kq, stream));
    }

    if (check) {
        ref_logits.alloc(logits_ne);
        metrics.alloc(3);
        CUDA_CHECK(cudaMemsetAsync(metrics.ptr, 0, 3 * sizeof(float), stream));
        ggml_cuda_q8k_dot4_kq_reference_kernel<<<kq_grid, block, 0, stream>>>(
            (const float *) Q->data, (const char *) K->data, ref_logits.ptr, scale,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[0], K->nb[1], K->nb[2], K->nb[3],
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_ref, stream));
        }
        ggml_cuda_q8k_dot4_kq_error_kernel<<<(logits_ne + 255) / 256, 256, 0, stream>>>(
            logits.ptr, ref_logits.ptr, metrics.ptr, logits_ne);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_err, stream));
        }
        float h_metrics[3] = {0.0f, 0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(h_metrics, metrics.ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double rms = sqrt(double(h_metrics[0]) / double(logits_ne));
        const double mean_abs = double(h_metrics[1]) / double(logits_ne);
        GGML_LOG_INFO("%s: q8k_dot4_kq_check logits=%zu rms=%.6g mean_abs=%.6g gt0.5=%.0f\n",
            __func__, logits_ne, rms, mean_abs, double(h_metrics[2]));
    }

    if (full_fa) {
        dim3 fa_grid(nq, n_heads_q, batch);
        ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
            logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // KQ-only probe: deliberately leave FA output unchanged/zeroed so this route
        // cannot masquerade as production attention. It exists to validate runtime
        // packed-DOT4 KQ layout, route contracts, and ISA only.
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_zero, stream));
        CUDA_CHECK(hipEventSynchronize(ev_zero));
        const float q_quant_ms = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_q_quant);
        const float k_pack_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_q_quant, ev_k_pack);
        const float kq_ms      = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_k_pack, ev_kq);
        const float ref_ms     = check ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_kq, ev_ref) : 0.0f;
        const float err_ms     = check ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_ref, ev_err) : 0.0f;
        const float output_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(check ? ev_err : ev_kq, ev_zero);
        const float zero_ms    = full_fa ? 0.0f : output_ms;
        const float fa_ms      = full_fa ? output_ms : 0.0f;
        const float total_ms   = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_zero);
        GGML_LOG_INFO("%s: q8k_dot4_kq_timing call=%llu variant=%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d check=%d q_quant_ms=%.6f k_pack_ms=%.6f kq_ms=%.6f ref_ms=%.6f err_ms=%.6f zero_ms=%.6f total_ms=%.6f fa_ms=%.6f full_fa=%d\n",
            __func__, timing_call, q8block_variant ? "q8block" : "packed16", nq, nk, n_heads_q, n_heads_k, batch, check ? 1 : 0,
            q_quant_ms, k_pack_ms, kq_ms, ref_ms, err_ms, zero_ms, total_ms, fa_ms, full_fa ? 1 : 0);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_start);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_kq);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_ref);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_err);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_zero);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double(q_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double k_mib = double(k_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double logits_mib = double(logits_ne * sizeof(float)) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=rocm_q8k_dot4_kq variant=%s full_fa=%d nq=%d nk=%d heads_q=%d heads_k=%d batch=%d q_payload=%.3fMiB k_payload=%.3fMiB logits=%.3fMiB note=%s\n",
            __func__, q8block_variant ? "q8block" : "packed16", full_fa ? 1 : 0, nq, nk, n_heads_q, n_heads_k, batch, q_mib, k_mib, logits_mib,
            full_fa ? "full_fa_from_logits_probe" : "kq_only_zero_output");
    }

    GGML_UNUSED(sinks);
}

#endif // GGML_USE_HIP
