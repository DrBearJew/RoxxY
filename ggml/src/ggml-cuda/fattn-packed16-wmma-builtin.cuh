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
using pbwmma_v2i32   = int      __attribute__((ext_vector_type(2)));
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

static __device__ __forceinline__ pbwmma_v8i32 pbwmma_mma_i4(
        pbwmma_v2i32 a, pbwmma_v2i32 b, pbwmma_v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ pbwmma_v8i32 pbwmma_mma_u4s4(
        pbwmma_v2i32 a, pbwmma_v2i32 b, pbwmma_v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c, false);
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
static constexpr int PBWMMA_I4_WORDS_PER_K16 = 2;
static constexpr int PBWMMA_I4_NIBBLES_PER_WORD = 8;

static __device__ __forceinline__ int pbwmma_i4_k_word(const int k) {
    return k >> 3;
}

static __device__ __forceinline__ int pbwmma_i4_k_nibble(const int k) {
    return k & 7;
}

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

#endif // __HIPCC__
