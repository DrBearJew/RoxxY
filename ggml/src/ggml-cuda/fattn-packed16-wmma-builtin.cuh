// fattn-packed16-wmma-builtin.cuh — RDNA3 raw WMMA builtins QK probe.
//
// Validates that __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 produces
// correct QK dot products. Scalar reference vs builtin must match.
//
// Fragment layout (from vLLM):
//   A: lane_lo = row, slot i = K dim
//   B: lane_lo = column, slot i = K dim
//   C: rows 2*i+lane_hi, columns lane_lo
//   lanes 0..31: even rows go to lanes 0..15, odd rows to lanes 16..31

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PBWMMA_D  = 256;
static constexpr int PBWMMA_BM = 16;
static constexpr int PBWMMA_BM32 = 32;
static constexpr int PBWMMA_BN = 16;

// ── RDNA3 WMMA intrinsic helpers ──────────────────────────────────

#if defined(__HIPCC__)
#define PBWMMA_RDNA3_BUILTIN 1

using pbwmma_v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using pbwmma_v8fp32  = float    __attribute__((ext_vector_type(8)));
using pbwmma_v4i32   = int      __attribute__((ext_vector_type(4)));
using pbwmma_v8i32   = int      __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ pbwmma_v8fp32 pbwmma_mma(
        pbwmma_v16fp16 a, pbwmma_v16fp16 b, pbwmma_v8fp32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

static __device__ __forceinline__ pbwmma_v8i32 pbwmma_mma_i8(
        pbwmma_v4i32 a, pbwmma_v4i32 b, pbwmma_v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ int pbwmma_extract_s8(
        const int word, const int lane) {
    const uint32_t u = static_cast<uint32_t>(word);
    const uint32_t b = (u >> (lane * 8)) & 0xffu;
    return static_cast<int>(static_cast<int8_t>(b));
}

static __device__ __forceinline__ int pbwmma_dot4_i8_i8(
        const int a, const int b, int acc) {
#if defined(__HIP_PLATFORM_AMD__) && ( \
        defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
        defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc += pbwmma_extract_s8(a, i) * pbwmma_extract_s8(b, i);
    }
    return acc;
#endif
}

// ── RDNA3 i8 WMMA layout contract ───────────────────────────────
// Keep these formulas in sync with scripts/hip/pbwmma-i8-calculator.py.
// DOT4 shadow probes validate this contract against the actual WMMA builtin.
static constexpr int PBWMMA_I8_WORDS_PER_K16 = 4;
static constexpr int PBWMMA_I8_BYTES_PER_WORD = 4;

static __device__ __forceinline__ int pbwmma_i8_k_word(const int k) {
    return k >> 2;
}

static __device__ __forceinline__ int pbwmma_i8_k_byte(const int k) {
    return k & 3;
}

static __device__ __forceinline__ int pbwmma_i8_a_row_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

static __device__ __forceinline__ int pbwmma_i8_b_col_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

static __device__ __forceinline__ int pbwmma_i8_d_row_from_acc(const int acc_reg, const int lane_hi) {
    return 2 * acc_reg + lane_hi;
}

static __device__ __forceinline__ int pbwmma_i8_d_col_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

static __device__ __forceinline__ int pbwmma_i8_d_acc_from_row(const int row) {
    return row >> 1;
}

static __device__ __forceinline__ int pbwmma_i8_d_lane_from_mn(const int m, const int n) {
    return n + 16 * (m & 1);
}

static __device__ __forceinline__ int pbwmma_i8_dot4_k16_ref_words(
        const int * __restrict__ q_words,
        const int * __restrict__ k_words) {
    int ref = 0;
#pragma unroll
    for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
        ref = pbwmma_dot4_i8_i8(q_words[g], k_words[g], ref);
    }
    return ref;
}

static __device__ __forceinline__ int pbwmma_i8_dot4_k16_ref_frag(
        const int * __restrict__ q_words,
        const pbwmma_v4i32 k_frag) {
    int ref = 0;
#pragma unroll
    for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
        ref = pbwmma_dot4_i8_i8(q_words[g], k_frag[g], ref);
    }
    return ref;
}

// ── RDNA3 f16 WMMA layout contract ──────────────────────────────
// Used by QK f16 probes and by PV-WMMA: A rows and B columns are lane_lo,
// accumulator register i maps to D row 2*i+lane_hi, D column lane_lo.
static __device__ __forceinline__ int pbwmma_f16_a_row_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

static __device__ __forceinline__ int pbwmma_f16_b_col_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

static __device__ __forceinline__ int pbwmma_f16_d_row_from_acc(const int acc_reg, const int lane_hi) {
    return 2 * acc_reg + lane_hi;
}

static __device__ __forceinline__ int pbwmma_f16_d_col_from_lane_lo(const int lane_lo) {
    return lane_lo;
}

// ── QK probe: one wave (32 threads), one 16×16 tile ────────────
//
// Input:
//   q_probe[16][256]  — query tile (host-random filled, F16)
//   k_probe[16][256]  — key tile   (host-random filled, F16)
//   out[16][16]       — output QK = Q @ K^T
//
// Launch: <<<1, 32, 0, stream>>>
static __global__ void pbwmma_qk_probe_kernel(
        const half  * __restrict__ q_probe,
        const half  * __restrict__ k_probe,
        float       * __restrict__ out) {

    const int lane    = threadIdx.x;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};

    for (int d0 = 0; d0 < PBWMMA_D; d0 += 16) {
        pbwmma_v16fp16 a;
        pbwmma_v16fp16 b;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = q_probe[lane_lo * PBWMMA_D + d0 + i];
            b[i] = k_probe[lane_lo * PBWMMA_D + d0 + i];
        }

        acc = pbwmma_mma(a, b, acc);
    }

    // Store: C row = 2*i + lane_hi, col = lane_lo
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + lane_hi;
        const int c = lane_lo;
        out[r * PBWMMA_BN + c] = acc[i];
    }
}

// ── Host validation ──────────────────────────────────────────────
static bool pbwmma_qk_probe_pass(hipStream_t stream) {
    constexpr int NQ   = PBWMMA_BM;
    constexpr int NK   = PBWMMA_BN;
    constexpr size_t qk_bytes  = NQ  * PBWMMA_D * sizeof(half);
    constexpr size_t kv_bytes  = NK  * PBWMMA_D * sizeof(half);
    constexpr size_t out_bytes = NQ  * NK      * sizeof(float);

    half  * h_q = (half  *) malloc(qk_bytes);
    half  * h_k = (half  *) malloc(kv_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    // Fill with random values.
    srand(42);
    for (size_t i = 0; i < NQ * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_q[i] = __float2half(v);
    }
    for (size_t i = 0; i < NK * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_k[i] = __float2half(v);
    }

    // Scalar reference: out[r][c] = Σ_d q[r][d] * k[c][d]
    for (int r = 0; r < NQ; r++) {
        for (int c = 0; c < NK; c++) {
            float sum = 0.0f;
            for (int d = 0; d < PBWMMA_D; d++)
                sum += __half2float(h_q[r * PBWMMA_D + d]) *
                       __half2float(h_k[c * PBWMMA_D + d]);
            h_ref[r * NK + c] = sum;
        }
    }

    half   * d_q = nullptr;
    half   * d_k = nullptr;
    float  * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_k,   kv_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k,   h_k,  kv_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pbwmma_qk_probe_kernel<<<1, 32, 0, stream>>>(d_q, d_k, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_k); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PBWMMA QK probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    fprintf(stderr, "PBWMMA QK probe PASSED: max_err=%f\n", max_err);
    return true;
}

// ── PV probe: one wave computes P[16x16] * V[16x16] ─────────────
static __global__ void pbwmma_pv_probe_kernel(
        const half  * __restrict__ p_probe,
        const half  * __restrict__ v_probe,
        float       * __restrict__ out) {

    const int lane    = threadIdx.x;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    pbwmma_v16fp16 a;
    pbwmma_v16fp16 b;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = p_probe[pbwmma_f16_a_row_from_lane_lo(lane_lo) * PBWMMA_BN + i];
        b[i] = v_probe[i * PBWMMA_BN + pbwmma_f16_b_col_from_lane_lo(lane_lo)];
    }

    pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
    acc = pbwmma_mma(a, b, acc);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = pbwmma_f16_d_row_from_acc(i, lane_hi);
        const int c = pbwmma_f16_d_col_from_lane_lo(lane_lo);
        out[r * PBWMMA_BN + c] = acc[i];
    }
}

static bool pbwmma_pv_probe_pass(hipStream_t stream) {
    constexpr int NR = PBWMMA_BM;
    constexpr int NK = PBWMMA_BN;
    constexpr int ND = PBWMMA_BN;
    constexpr size_t p_bytes   = NR * NK * sizeof(half);
    constexpr size_t v_bytes   = NK * ND * sizeof(half);
    constexpr size_t out_bytes = NR * ND * sizeof(float);

    half  * h_p   = (half  *) malloc(p_bytes);
    half  * h_v   = (half  *) malloc(v_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    srand(314159);
    for (int i = 0; i < NR * NK; ++i) {
        const float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        h_p[i] = __float2half(v);
    }
    for (int i = 0; i < NK * ND; ++i) {
        const float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_v[i] = __float2half(v);
    }

    for (int r = 0; r < NR; ++r) {
        for (int d = 0; d < ND; ++d) {
            float sum = 0.0f;
            for (int k = 0; k < NK; ++k) {
                sum += __half2float(h_p[r * NK + k]) * __half2float(h_v[k * ND + d]);
            }
            h_ref[r * ND + d] = sum;
        }
    }

    half  * d_p = nullptr;
    half  * d_v = nullptr;
    float * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_p, p_bytes));
    CUDA_CHECK(hipMalloc(&d_v, v_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_p, h_p, p_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_v, h_v, v_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pbwmma_pv_probe_kernel<<<1, 32, 0, stream>>>(d_p, d_v, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NR * ND; ++i) max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    CUDA_CHECK(hipFree(d_p));
    CUDA_CHECK(hipFree(d_v));
    CUDA_CHECK(hipFree(d_out));
    free(h_p); free(h_v); free(h_ref); free(h_gpu);

    if (max_err > 1e-2f) {
        fprintf(stderr, "PBWMMA PV WMMA probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    fprintf(stderr, "PBWMMA PV WMMA probe PASSED: max_err=%f\n", max_err);
    return true;
}

static __global__ void pbwmma_i8_qk_probe_kernel(
        const int8_t * __restrict__ q_probe,
        const int8_t * __restrict__ k_probe,
        int          * __restrict__ out) {
    const int lane    = threadIdx.x;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    pbwmma_v4i32 a;
    pbwmma_v4i32 b;
    const int a_row = pbwmma_i8_a_row_from_lane_lo(lane_lo);
    const int b_col = pbwmma_i8_b_col_from_lane_lo(lane_lo);
    #pragma unroll
    for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
        uint32_t av = 0, bv = 0;
        #pragma unroll
        for (int j = 0; j < PBWMMA_I8_BYTES_PER_WORD; ++j) {
            av |= uint32_t(uint8_t(q_probe[a_row * 16 + PBWMMA_I8_BYTES_PER_WORD*g + j])) << (8*j);
            bv |= uint32_t(uint8_t(k_probe[b_col * 16 + PBWMMA_I8_BYTES_PER_WORD*g + j])) << (8*j);
        }
        a[g] = int(av);
        b[g] = int(bv);
    }

    pbwmma_v8i32 acc = {0,0,0,0,0,0,0,0};
    acc = pbwmma_mma_i8(a, b, acc);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = pbwmma_i8_d_row_from_acc(i, lane_hi);
        const int c = pbwmma_i8_d_col_from_lane_lo(lane_lo);
        out[r * 16 + c] = acc[i];
    }
}

static bool pbwmma_i8_qk_probe_pass(hipStream_t stream) {
    constexpr int N = 16;
    constexpr size_t in_bytes = N * N * sizeof(int8_t);
    constexpr size_t out_bytes = N * N * sizeof(int);
    int8_t * h_q = (int8_t *) malloc(in_bytes);
    int8_t * h_k = (int8_t *) malloc(in_bytes);
    int * h_ref = (int *) malloc(out_bytes);
    int * h_gpu = (int *) malloc(out_bytes);

    srand(1337);
    for (int i = 0; i < N*N; ++i) {
        h_q[i] = int8_t((rand() % 255) - 127);
        h_k[i] = int8_t((rand() % 255) - 127);
    }
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int sum = 0;
            for (int d = 0; d < N; ++d) sum += int(h_q[r*N + d]) * int(h_k[c*N + d]);
            h_ref[r*N + c] = sum;
        }
    }

    int8_t * d_q = nullptr;
    int8_t * d_k = nullptr;
    int * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q, in_bytes));
    CUDA_CHECK(hipMalloc(&d_k, in_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q, h_q, in_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k, h_k, in_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));
    pbwmma_i8_qk_probe_kernel<<<1, 32, 0, stream>>>(d_q, d_k, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    int max_err = 0;
    for (int i = 0; i < N*N; ++i) max_err = max(max_err, abs(h_gpu[i] - h_ref[i]));
    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_k); free(h_ref); free(h_gpu);
    if (max_err != 0) {
        fprintf(stderr, "PBWMMA I8 QK probe FAILED: max_err=%d\n", max_err);
        return false;
    }
    fprintf(stderr, "PBWMMA I8 QK probe PASSED: max_err=%d\n", max_err);
    return true;
}

// ── I8 DOT4 shadow probe: use packed scalar DOT4 as a WMMA oracle ─────
// This is a layout/signedness instrument, not a performance path.  It feeds
// the same packed i8 fragments to RDNA3 WMMA and to a DOT4 chain, then checks
// that every WMMA output lane maps to the DOT4 scalar reference.
static __global__ void pbwmma_i8_dot4_shadow_probe_kernel(
        const int * __restrict__ q_words,
        const int * __restrict__ k_words,
        int       * __restrict__ wmma_out,
        int       * __restrict__ dot4_out) {
    const int lane    = threadIdx.x;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    pbwmma_v4i32 a;
    pbwmma_v4i32 b;
    const int a_row = pbwmma_i8_a_row_from_lane_lo(lane_lo);
    const int b_col = pbwmma_i8_b_col_from_lane_lo(lane_lo);
#pragma unroll
    for (int g = 0; g < PBWMMA_I8_WORDS_PER_K16; ++g) {
        a[g] = q_words[a_row * PBWMMA_I8_WORDS_PER_K16 + g];
        b[g] = k_words[b_col * PBWMMA_I8_WORDS_PER_K16 + g];
    }

    pbwmma_v8i32 acc = {0,0,0,0,0,0,0,0};
    acc = pbwmma_mma_i8(a, b, acc);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = pbwmma_i8_d_row_from_acc(i, lane_hi);
        const int c = pbwmma_i8_d_col_from_lane_lo(lane_lo);
        const int ref = pbwmma_i8_dot4_k16_ref_words(
            q_words + r * PBWMMA_I8_WORDS_PER_K16,
            k_words + c * PBWMMA_I8_WORDS_PER_K16);
        wmma_out[r * 16 + c] = acc[i];
        dot4_out[r * 16 + c] = ref;
    }
}

static bool pbwmma_i8_dot4_shadow_probe_pass(hipStream_t stream) {
    constexpr int N = 16;
    constexpr int WORDS_PER_ROW = 4;
    constexpr size_t words_bytes = N * WORDS_PER_ROW * sizeof(int);
    constexpr size_t out_bytes = N * N * sizeof(int);

    int * h_q = (int *) malloc(words_bytes);
    int * h_k = (int *) malloc(words_bytes);
    int * h_wmma = (int *) malloc(out_bytes);
    int * h_dot4 = (int *) malloc(out_bytes);

    srand(20260602);
    for (int i = 0; i < N * WORDS_PER_ROW; ++i) {
        uint32_t qw = 0;
        uint32_t kw = 0;
        for (int j = 0; j < 4; ++j) {
            qw |= uint32_t(uint8_t(int8_t((rand() % 255) - 127))) << (8*j);
            kw |= uint32_t(uint8_t(int8_t((rand() % 255) - 127))) << (8*j);
        }
        h_q[i] = int(qw);
        h_k[i] = int(kw);
    }

    int * d_q = nullptr;
    int * d_k = nullptr;
    int * d_wmma = nullptr;
    int * d_dot4 = nullptr;
    CUDA_CHECK(hipMalloc(&d_q, words_bytes));
    CUDA_CHECK(hipMalloc(&d_k, words_bytes));
    CUDA_CHECK(hipMalloc(&d_wmma, out_bytes));
    CUDA_CHECK(hipMalloc(&d_dot4, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q, h_q, words_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k, h_k, words_bytes, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_wmma, 0, out_bytes, stream));
    CUDA_CHECK(hipMemsetAsync(d_dot4, 0, out_bytes, stream));

    pbwmma_i8_dot4_shadow_probe_kernel<<<1, 32, 0, stream>>>(d_q, d_k, d_wmma, d_dot4);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_wmma, d_wmma, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipMemcpyAsync(h_dot4, d_dot4, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    int max_err = 0;
    int first = -1;
    for (int i = 0; i < N * N; ++i) {
        const int err = abs(h_wmma[i] - h_dot4[i]);
        if (err > max_err) max_err = err;
        if (err != 0 && first < 0) first = i;
    }

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_wmma));
    CUDA_CHECK(hipFree(d_dot4));
    free(h_q); free(h_k); free(h_wmma); free(h_dot4);

    if (max_err != 0) {
        fprintf(stderr, "PBWMMA I8 DOT4 shadow FAILED: max_err=%d first=%d\n", max_err, first);
        return false;
    }
    fprintf(stderr, "PBWMMA I8 DOT4 shadow PASSED: max_err=%d\n", max_err);
    return true;
}

// ── BM32 QK probe: one CTA (64 threads), two 16×16 tiles ──────────
//
// Input:
//   q_probe[32][256]  — query tile (host-random filled, F16)
//   k_probe[16][256]  — key tile   (host-random filled, F16)
//   out[32][16]       — output QK = Q @ K^T
//
// Launch: <<<1, 64, 0, stream>>>
// wave 0 (thread 0..31): computes rows 0..15
// wave 1 (thread 32..63): computes rows 16..31
static __global__ void pbwmma_qk_probe_bm32_kernel(
        const half  * __restrict__ q_probe,
        const half  * __restrict__ k_probe,
        float       * __restrict__ out) {

    if (threadIdx.x >= 64) return;

    const int wave_id = threadIdx.x >> 5;  // 0 or 1
    const int lane    = threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int r_base  = wave_id * 16;        // row 0 or 16

    pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};

    for (int d0 = 0; d0 < PBWMMA_D; d0 += 16) {
        pbwmma_v16fp16 a;
        pbwmma_v16fp16 b;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = q_probe[(r_base + lane_lo) * PBWMMA_D + d0 + i];
            b[i] = k_probe[lane_lo * PBWMMA_D + d0 + i];
        }

        acc = pbwmma_mma(a, b, acc);
    }

    // Store: C row = r_base + 2*i + lane_hi, col = lane_lo
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r_base + 2 * i + lane_hi;
        const int c = lane_lo;
        out[r * PBWMMA_BN + c] = acc[i];
    }
}

static bool pbwmma_qk_probe_bm32_pass(hipStream_t stream) {
    constexpr int NQ   = PBWMMA_BM32;
    constexpr int NK   = PBWMMA_BN;
    constexpr size_t qk_bytes  = NQ  * PBWMMA_D * sizeof(half);
    constexpr size_t kv_bytes  = NK  * PBWMMA_D * sizeof(half);
    constexpr size_t out_bytes = NQ  * NK      * sizeof(float);

    half  * h_q = (half  *) malloc(qk_bytes);
    half  * h_k = (half  *) malloc(kv_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    srand(12345);
    for (size_t i = 0; i < NQ * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_q[i] = __float2half(v);
    }
    for (size_t i = 0; i < NK * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_k[i] = __float2half(v);
    }

    // Scalar reference: out[r][c] = Σ_d q[r][d] * k[c][d]
    for (int r = 0; r < NQ; r++) {
        for (int c = 0; c < NK; c++) {
            float sum = 0.0f;
            for (int d = 0; d < PBWMMA_D; d++)
                sum += __half2float(h_q[r * PBWMMA_D + d]) *
                       __half2float(h_k[c * PBWMMA_D + d]);
            h_ref[r * NK + c] = sum;
        }
    }

    half   * d_q = nullptr;
    half   * d_k = nullptr;
    float  * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_k,   kv_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k,   h_k,  kv_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pbwmma_qk_probe_bm32_kernel<<<1, 64, 0, stream>>>(d_q, d_k, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_k); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PBWMMA BM32_2W QK probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    fprintf(stderr, "PBWMMA BM32_2W QK probe PASSED: max_err=%f\n", max_err);
    return true;
}


// ── BM64_X4 QK probe: one CTA (128 threads), four 16×16 tiles ──
//
// Validates four-wave WMMA row mapping without full FA.
//
// Input:
//   q_probe[64][256]  — query tile (host-random)
//   k_probe[16][256]  — one shared K tile
//   out[64][16]       — QK[64][16]
//
// Launch: <<<1, 128, 0, stream>>>
// wave 0 (t0..31):   rows 0..15
// wave 1 (t32..63):  rows 16..31
// wave 2 (t64..95):  rows 32..47
// wave 3 (t96..127): rows 48..63

static __global__ void pbwmma_qk_probe_bm64_x4_kernel(
        const half  * __restrict__ q_probe,
        const half  * __restrict__ k_probe,
        float       * __restrict__ out) {

    if (threadIdx.x >= 128) return;

    const int wave_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int r_base  = wave_id * 16;

    pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};

    for (int d0 = 0; d0 < PBWMMA_D; d0 += 16) {
        pbwmma_v16fp16 a;
        pbwmma_v16fp16 b;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = q_probe[(r_base + lane_lo) * PBWMMA_D + d0 + i];
            b[i] = k_probe[lane_lo * PBWMMA_D + d0 + i];
        }

        acc = pbwmma_mma(a, b, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r_base + 2 * i + lane_hi;
        const int c = lane_lo;
        out[r * PBWMMA_BN + c] = acc[i];
    }
}

static bool pbwmma_qk_probe_bm64_x4_pass(hipStream_t stream) {
    constexpr int NQ   = 64;
    constexpr int NK   = PBWMMA_BN;
    constexpr size_t qk_bytes  = NQ  * PBWMMA_D * sizeof(half);
    constexpr size_t kv_bytes  = NK  * PBWMMA_D * sizeof(half);
    constexpr size_t out_bytes = NQ  * NK      * sizeof(float);

    half  * h_q = (half  *) malloc(qk_bytes);
    half  * h_k = (half  *) malloc(kv_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    srand(987654);
    for (size_t i = 0; i < NQ * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_q[i] = __float2half(v);
    }
    for (size_t i = 0; i < NK * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_k[i] = __float2half(v);
    }

    // Scalar reference
    for (int r = 0; r < NQ; r++) {
        for (int c = 0; c < NK; c++) {
            float sum = 0.0f;
            for (int d = 0; d < PBWMMA_D; d++)
                sum += __half2float(h_q[r * PBWMMA_D + d]) *
                       __half2float(h_k[c * PBWMMA_D + d]);
            h_ref[r * NK + c] = sum;
        }
    }

    half   * d_q = nullptr;
    half   * d_k = nullptr;
    float  * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_k,   kv_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k,   h_k,  kv_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pbwmma_qk_probe_bm64_x4_kernel<<<1, 128, 0, stream>>>(d_q, d_k, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    // Check for duplicated/aliased rows: verify each row differs from others
    bool dup_found = false;
    for (int r1 = 0; r1 < NQ && !dup_found; r1++) {
        for (int r2 = r1 + 1; r2 < NQ; r2++) {
            bool same = true;
            for (int c = 0; c < NK && same; c++)
                if (fabsf(h_gpu[r1 * NK + c] - h_gpu[r2 * NK + c]) > 1e-6f) same = false;
            if (same) {
                fprintf(stderr, "PBWMMA BM64_X4 row ALIAS: r1=%d r2=%d\n", r1, r2);
                dup_found = true;
                break;
            }
        }
    }

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_k); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PBWMMA BM64_X4 QK probe FAILED: max_err=%f\n", max_err);
        return false;
    }
    if (dup_found) {
        fprintf(stderr, "PBWMMA BM64_X4 QK probe FAILED: row aliasing detected\n");
        return false;
    }
    fprintf(stderr, "PBWMMA BM64_X4 QK probe PASSED: max_err=%f\n", max_err);
    return true;
}

// ── GQA2 QK probe: one CTA (64 threads), two Q slots × one K tile ─
//
// Validates that two WMMA waves on different Q heads (slot 0, slot 1)
// sharing the same K tile produce correct independent QK outputs.
//
// Input:
//   q_probe[2][16][256]  — two Q head slots (host-random, different fills)
//   k_probe[16][256]     — one shared K tile
//   out[2][16][16]       — output QK[slot] = Q[slot] @ K^T
//
// Launch: <<<1, 64, 0, stream>>>
// wave 0 (threads 0..31):  slot 0, rows 0..15
// wave 1 (threads 32..63): slot 1, rows 0..15
static __global__ void pbwmma_qk_probe_gqa2_kernel(
        const half  * __restrict__ q_probe,
        const half  * __restrict__ k_probe,
        float       * __restrict__ out) {

    if (threadIdx.x >= 64) return;

    const int wave_id = threadIdx.x >> 5;  // 0 -> slot 0, 1 -> slot 1
    const int slot    = wave_id;
    const int lane    = threadIdx.x & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;

    pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};

    for (int d0 = 0; d0 < PBWMMA_D; d0 += 16) {
        pbwmma_v16fp16 a;
        pbwmma_v16fp16 b;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a[i] = q_probe[(slot * PBWMMA_BM + lane_lo) * PBWMMA_D + d0 + i];
            b[i] = k_probe[lane_lo * PBWMMA_D + d0 + i];
        }

        acc = pbwmma_mma(a, b, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 2 * i + lane_hi;
        const int c = lane_lo;
        out[(slot * PBWMMA_BM + r) * PBWMMA_BN + c] = acc[i];
    }
}

static bool pbwmma_qk_probe_gqa2_pass(hipStream_t stream) {
    constexpr int GQA_GROUP = 2;
    constexpr int NQ   = PBWMMA_BM * GQA_GROUP;
    constexpr int NK   = PBWMMA_BN;
    constexpr size_t qk_bytes  = NQ  * PBWMMA_D * sizeof(half);
    constexpr size_t kv_bytes  = NK  * PBWMMA_D * sizeof(half);
    constexpr size_t out_bytes = NQ  * NK      * sizeof(float);

    half  * h_q = (half  *) malloc(qk_bytes);
    half  * h_k = (half  *) malloc(kv_bytes);
    float * h_ref = (float *) malloc(out_bytes);
    float * h_gpu = (float *) malloc(out_bytes);

    // Fill slot 0 and slot 1 with DIFFERENT random values to catch aliasing.
    srand(424242);
    for (int slot = 0; slot < GQA_GROUP; ++slot) {
        srand(424242 + slot * 7919);
        for (size_t i = 0; i < PBWMMA_BM * PBWMMA_D; i++) {
            float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * (slot == 0 ? 2.0f : 3.0f);
            h_q[slot * PBWMMA_BM * PBWMMA_D + i] = __float2half(v);
        }
    }
    srand(99173);
    for (size_t i = 0; i < NK * PBWMMA_D; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 2.0f;
        h_k[i] = __float2half(v);
    }

    // Scalar reference
    for (int slot = 0; slot < GQA_GROUP; ++slot) {
        for (int r = 0; r < PBWMMA_BM; r++) {
            for (int c = 0; c < NK; c++) {
                float sum = 0.0f;
                for (int d = 0; d < PBWMMA_D; d++)
                    sum += __half2float(h_q[(slot * PBWMMA_BM + r) * PBWMMA_D + d]) *
                           __half2float(h_k[c * PBWMMA_D + d]);
                h_ref[(slot * PBWMMA_BM + r) * NK + c] = sum;
            }
        }
    }

    half   * d_q = nullptr;
    half   * d_k = nullptr;
    float  * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_q,   qk_bytes));
    CUDA_CHECK(hipMalloc(&d_k,   kv_bytes));
    CUDA_CHECK(hipMalloc(&d_out, out_bytes));
    CUDA_CHECK(hipMemcpyAsync(d_q,   h_q,  qk_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_k,   h_k,  kv_bytes,  hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemsetAsync(d_out, 0, out_bytes, stream));

    pbwmma_qk_probe_gqa2_kernel<<<1, 64, 0, stream>>>(d_q, d_k, d_out);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipMemcpyAsync(h_gpu, d_out, out_bytes, hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    float max_err = 0.0f;
    for (int i = 0; i < NQ * NK; i++)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));

    CUDA_CHECK(hipFree(d_q));
    CUDA_CHECK(hipFree(d_k));
    CUDA_CHECK(hipFree(d_out));
    free(h_q); free(h_k); free(h_ref); free(h_gpu);

    if (max_err > 1e-3f) {
        fprintf(stderr, "PBWMMA GQA2 QK probe FAILED: max_err=%f\n", max_err);
        for (int i = 0; i < NQ * NK && i < 32; i++) {
            float diff = fabsf(h_gpu[i] - h_ref[i]);
            if (diff > 1e-3f)
                fprintf(stderr, "  idx=%d slot=%d r=%d c=%d ref=%f gpu=%f diff=%f\n",
                    i, i / (PBWMMA_BM*NK), (i/NK) % PBWMMA_BM, i % NK, h_ref[i], h_gpu[i], diff);
        }
        return false;
    }
    fprintf(stderr, "PBWMMA GQA2 QK probe PASSED: max_err=%f\n", max_err);
    return true;
}

#else
static bool pbwmma_qk_probe_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA QK probe SKIPPED (not gfx1100)\n");
    return false;
}
static bool pbwmma_qk_probe_bm64_x4_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA BM64_X4 QK probe SKIPPED (not gfx1100)\n");
    return false;
}
static bool pbwmma_pv_probe_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA PV WMMA probe SKIPPED (not gfx1100)\n");
    return false;
}
static bool pbwmma_qk_probe_bm32_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA BM32_2W QK probe SKIPPED (not gfx1100)\n");
    return false;
}
static bool pbwmma_qk_probe_gqa2_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA GQA2 QK probe SKIPPED (not gfx1100)\n");
    return false;
}
#endif // __HIPCC__
