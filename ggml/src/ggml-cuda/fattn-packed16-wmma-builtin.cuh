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
static constexpr int PBWMMA_BN = 16;

// ── RDNA3 WMMA intrinsic helpers ──────────────────────────────────

// RDNA3/gfx11 builtins. Compiles on any HIPCC targeting gfx11.
// If targeting non-WMMA GPU, __builtin_amdgcn_wmma will fail at compile time.
#if defined(__HIPCC__)
#define PBWMMA_RDNA3_BUILTIN 1

using pbwmma_v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using pbwmma_v8fp32  = float    __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ pbwmma_v8fp32 pbwmma_mma(
        pbwmma_v16fp16 a, pbwmma_v16fp16 b, pbwmma_v8fp32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
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

#else
static bool pbwmma_qk_probe_pass(hipStream_t stream) {
    GGML_UNUSED(stream);
    fprintf(stderr, "PBWMMA QK probe SKIPPED (not RDNA3/gfx11)\n");
    return false;
}
#endif // __HIPCC__
