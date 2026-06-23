// ═══════════════════════════════════════════════════════════════════════════
//  Packed16 Decode Kernel System — canonical route table
// ═══════════════════════════════════════════════════════════════════════════
//
//  Four kernel roles, dispatched by (nq, nk, K type):
//
//  ┌──────────────┬───────────────┬────────────────────┬────────────────────┐
//  │ Workload     │ Condition     │ Kernel             │ Notes              │
//  ├──────────────┼───────────────┼────────────────────┼────────────────────┤
//  │ Prefill      │ nq > 1        │ v4 (recthist)      │ Existing path      │
//  │ MTP verify   │ nq > 1        │ v4 (recthist)      │ DOT4 routes: nq>2  │
//  │ Small decode │ nq == 1,      │ BN64 decode         │ No split overhead  │
//  │              │ nk < 2048     │                    │                    │
//  │ Long decode  │ nq == 1,      │ split-K stage1+2    │ Parallel over K    │
//  │              │ nk >= 2048    │                    │                    │
//  └──────────────┴───────────────┴────────────────────┴────────────────────┘
//
//  Hard rules:
//    - nq == 1  → decode kernels (BN64 or split-K). Never v4.
//    - nq > 1   → v4 kernel. Never BN64 or split-K.
//    - nq > 2   → DOT4/I8 prefill routes become eligible (via MTP hint or
//                 standard prefill policy).
//    - split-K  → stage1 writes UNNORMALIZED partial_o. Stage2 merges.
//    - Q4PAIR   → experimental, disabled by default. No speed gain.
//    - MTP draft context must NOT use packed16 K (is_mtp_draft KV-cache gate).
//      The FA selector also rejects I32 K for MTP draft via fa_hint check.
//
//  MTP instruction layer:
//    MTP_VERIFY_QK is a semantic FA instruction. nq is only a kernel legality gate.
//    DOT4 is eligible for MTP_VERIFY_QK only when nq > 2.
//    MTP_DRAFT never enables DOT4 preference — it falls through to existing policy.
//    MTP chunk size (LLAMA_MTP_PREFILL_CHUNK) does not control route selection;
//    the instruction does. Chunk size only controls whether the chosen kernel is legal.
//
//  Instruction → implementation:
//    MTP_VERIFY_QK + q8_0/q4_0  → DOT4 (if env enabled)
//    MTP_VERIFY_QK + f16/f16    → WMMA/F16/VEC (DOT4 kv_format_incompatible)
//    MTP_DRAFT                  → existing policy (no DOT4 preference)
//
//  Env flags:
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN=64
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK=1          (force-enable)
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD=2048
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE=512
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR=1          (experimental)
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_INLINE_Q4=1       (experimental)
//    GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=dsplit       (experimental, duplicate QK)
//    GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=logits_debug (two-stage debug route)
//    GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_SPLITK=1         (debug override)
//    GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_BN64=1           (debug override)
//    GGML_CUDA_ROCM_Q8K_DOT4_FORCE_V4=1               (debug override)
//
//  Contract (all packed16 decode paths):
//    Q = F32, D=256        K = I32 packed16, D/4*4==D
//    dst = F32, D=256      V = q4_0/q8_0/f16
//    rotation = OFF        GQA ratio integer
//
//  Expected performance (7900 XTX, 27B Q4_K_M, q4_0 V, packed16 K):
//    Prefill (v4):  pp512=401  pp2k=689  pp8k=746  pp15k=709 t/s
//    Decode BN64:   ctx=512: 32 t/s,  ctx=8k: 28 t/s,  ctx=16k: 24 t/s
//    Decode splitK: ctx=512: 32 t/s,  ctx=8k: 32 t/s,  ctx=16k: 32 t/s
//
//  See: .harness/research/packed16-only-k-implementation-plan.md
//       .harness/research/packed16-decode-variant-roadmap-20260603.md
//       docs/rocm-tbq4-paths/packed16-benchmark-report-20260528.md
// ═══════════════════════════════════════════════════════════════════════════

// ── Packed16 decode kernel — BN outer QK + BN_VSUB V accumulation ────────
// Env: GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN (default 64), DECODE_VSUB (default 8)
//      GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR=1, DECODE_Q4PAIR=1
//      GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK=1, DECODE_SPLITK_SIZE=512

constexpr int DECODE_D = 256;
constexpr int DECODE_I32_PER_ROW = 64;
constexpr int DECODE_N_BLOCKS = 8;

struct small_verify_fa4_profile {
    unsigned long long qk_cycles;
    unsigned long long softmax_cycles;
    unsigned long long pv_cycles;
    unsigned long long write_cycles;
    unsigned long long reduce_cycles;
};

#define SMALL_VERIFY_PROFILE_PHASE_BEGIN() do { \
    if (profile) { \
        __syncthreads(); \
        if (tid == 0) { profile_t0 = clock64(); } \
        __syncthreads(); \
    } \
} while (0)

#define SMALL_VERIFY_PROFILE_PHASE_END(FIELD) do { \
    if (profile) { \
        __syncthreads(); \
        if (tid == 0) { \
            const unsigned long long profile_t1 = clock64(); \
            atomicAdd(&profile->FIELD, profile_t1 >= profile_t0 ? profile_t1 - profile_t0 : 0ull); \
        } \
        __syncthreads(); \
    } \
} while (0)

template <int BN, int BN_VSUB, bool INLINE_Q4, bool Q4PAIR, bool V4_144 = false>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows,
        int v_debug,
        int v_debug_hq,
        int v_debug_k,
        int v_debug_d0,
        int v_debug_count) {

    if (nq != 1) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    constexpr int I32_PER_ROW = 64, N_BLOCKS = 8;
    const int hq = blockIdx.x, b = blockIdx.y, hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;
    const int tid = threadIdx.x;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);          // BN
    float * probs  = logits + BN;                               // BN
    float * sm     = probs + BN;                                // 4: row_m, row_l, old_scale, pad
    int   * q_i32  = reinterpret_cast<int  *>(sm + 4);         // 64 ints
    float * q_scl  = reinterpret_cast<float *>(q_i32 + I32_PER_ROW); // 8 floats

    if (tid == 0) { sm[0] = -1e38f; sm[1] = 0.0f; }
    __syncthreads();

    float out0 = 0.0f, out1 = 0.0f;

    // Precompute q4 dequant indices (for 256-thread modes)
    const int q4_blk = tid >> 5;      // 0..7
    const int iq     = tid & 15;      // 0..15
    const int shift  = (tid & 16) ? 4 : 0;

    // Load Q once
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    for (int i = tid; i < I32_PER_ROW; i += blockDim.x)
        q_i32[i] = q_payload[q_base * I32_PER_ROW + i];
    for (int i = tid; i < N_BLOCKS; i += blockDim.x)
        q_scl[i] = q_scales[q_base * N_BLOCKS + i];
    __syncthreads();

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        // 1. QK dot
        if (tid < BN) {
            int kk = tid, k = k0 + kk;
            bool valid = (kk < tile_n && k < nk) && (k - q_offset <= 0);
            float s = -1e38f;
            if (valid) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(q_i32, q_scl,
                    k_payload + kb * I32_PER_ROW, k_scales + kb * N_BLOCKS) * scale;
            }
            logits[kk] = s;
        }
        __syncthreads();

        // 2. BN-tile softmax
        if (tid == 0) {
            float tile_m = -1e38f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[kk]);
            float m_prev = sm[0], l_prev = sm[1];
            float m_new = fmaxf(m_prev, tile_m);
            float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) {
                float p = expf(logits[kk] - m_new); probs[kk] = p; tile_l += p;
            }
            sm[0] = m_new; sm[1] = l_prev * old_scale + tile_l; sm[2] = old_scale;
        }
        __syncthreads();

        // 3. V accumulation — BN_VSUB subchunks
        static_assert(!V4_144 || (!INLINE_Q4 && !Q4PAIR), "V4_144 diagnostic path supports scalar q4 dequant only");
        if constexpr (Q4PAIR) {
            if (tid < 128) {
                const int pt = tid;      // pair_tid: thread index 0..127 maps to byte index
                const int pb = pt >> 4;   // q4 block 0..7
                const int pi = pt & 15;   // byte index 0..15
                out0 *= sm[2]; out1 *= sm[2];
                for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                    int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                    for (int kk = sub; kk < end; ++kk) {
                        int k = k0 + kk; float p = probs[kk];
                        const auto * vb = (const block_q4_0 *)(v_head + int64_t(k) * nb21 + int64_t(pb) * nb20);
                        uint8_t pk = vb->qs[pi]; float vd = __half2float(vb->d);
                        out0 += p * ((float(int(pk & 0x0f)) - 8.0f) * vd);
                        out1 += p * ((float(int(pk >> 4)) - 8.0f) * vd);
                    }
                }
            }
        } else if (tid < 256) {
            out0 *= sm[2];
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    int k = k0 + kk; float p = probs[kk];
                    float v;
                    if constexpr (INLINE_Q4) {
                        const auto * vb = (const block_q4_0 *)(v_head + int64_t(k) * nb21 + int64_t(q4_blk) * nb20);
                        int q = (vb->qs[iq] >> shift) & 0x0f;
                        v = (float(q) - 8.0f) * __half2float(vb->d);
                    } else if constexpr (V4_144) {
                        v = ggml_cuda_q8k_dot4_dequant_v4_k16d16_144(v_head, nb21, k, tid);
                    } else {
                        v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                    }
                    if (v_debug && hq == v_debug_hq && k == v_debug_k && tid >= v_debug_d0 && tid < v_debug_d0 + v_debug_count) {
                        printf("Q8K_V_DEBUG v4=%d hq=%d hk=%d b=%d k=%d d=%d v=%.9g nb20=%lld nb21=%lld nb22=%lld nb23=%lld\\n",
                            V4_144 ? 1 : 0, hq, hk, b, k, tid, (double) v,
                            (long long) nb20, (long long) nb21, (long long) nb22, (long long) nb23);
                    }
                    out0 += p * v;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if constexpr (Q4PAIR) {
        if (tid < 128) {
            float norm = 1.0f / sm[1];
            const int pt = tid;
            const int pb = pt >> 4, pi = pt & 15;
            size_t base = ((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * 256;
            dst[base + pb * 32 + pi]      = out0 * norm;
            dst[base + pb * 32 + pi + 16] = out1 * norm;
        }
    } else if (tid < 256) {
        const float final_out = out0 / sm[1];
        if (v_debug && hq == v_debug_hq && tid >= v_debug_d0 && tid < v_debug_d0 + v_debug_count) {
            printf("Q8K_O_DEBUG v4=%d hq=%d hk=%d b=%d d=%d out=%.9g m=%.9g l=%.9g\\n",
                V4_144 ? 1 : 0, hq, hk, b, tid, (double) final_out, (double) sm[0], (double) sm[1]);
        }
        dst[((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * 256 + tid] = final_out;
    }
}


// ── Wave-QK packed16 decode — one wave cooperatively computes one QK dot ─
// Safer isolated Variant C implementation: each wave computes one logit at a
// time using two packed16 words per lane and a wave reduction. PV remains the
// scalar q4_0 path so the only topology change is QK parallelism.
template <int BN, int BN_VSUB, bool Q4PAIR>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_waveqk_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {

    if (nq != 1) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    const int hq = blockIdx.x;
    const int b  = blockIdx.y;
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;
    const int tid  = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * probs  = logits + BN;
    float * sm     = probs + BN;
    int   * q_i32  = reinterpret_cast<int *>(sm + 4);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + DECODE_I32_PER_ROW);

    if (tid == 0) { sm[0] = -1e38f; sm[1] = 0.0f; }
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    for (int i = tid; i < DECODE_I32_PER_ROW; i += blockDim.x) q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + i];
    for (int i = tid; i < DECODE_N_BLOCKS; i += blockDim.x) q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + i];
    __syncthreads();

    float out0 = 0.0f, out1 = 0.0f;
    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    const int q4_blk = tid >> 5;
    const int iq     = tid & 15;
    const int shift  = (tid & 16) ? 4 : 0;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        for (int kk0 = 0; kk0 < BN; kk0 += 8) {
            const int kk = kk0 + wave;
            const int k  = k0 + kk;
            float sum = 0.0f;
            if (kk < tile_n && k < nk && (k - q_offset <= 0)) {
                const size_t kb = k_head_base + k;
                for (int idx = lane; idx < DECODE_I32_PER_ROW; idx += 32) {
                    int acc = 0;
                    acc = ggml_cuda_q8k_dot4_i8_i8(q_i32[idx], k_payload[kb * DECODE_I32_PER_ROW + idx], acc);
                    const int sb = idx / (QK8_0 / 4);
                    sum += float(acc) * q_scl[sb] * __half2float(k_scales[kb * DECODE_N_BLOCKS + sb]);
                }
            } else {
                sum = -1e38f;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off, 32);
            if (lane == 0) logits[kk] = (kk < tile_n && k < nk && (k - q_offset <= 0)) ? sum * scale : -1e38f;
        }
        __syncthreads();

        if (tid == 0) {
            float tile_m = -1e38f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[kk]);
            const float m_prev = sm[0], l_prev = sm[1];
            const float m_new = fmaxf(m_prev, tile_m);
            const float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) {
                const float p = expf(logits[kk] - m_new);
                probs[kk] = p;
                tile_l += p;
            }
            sm[0] = m_new; sm[1] = l_prev * old_scale + tile_l; sm[2] = old_scale;
        }
        __syncthreads();

        if constexpr (Q4PAIR) {
            if (tid < 128) {
                const int pt = tid, pb = pt >> 4, pi = pt & 15;
                out0 *= sm[2]; out1 *= sm[2];
                for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                    const int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const auto * vb = (const block_q4_0 *)(v_head + int64_t(k) * nb21 + int64_t(pb) * nb20);
                        const uint8_t pk = vb->qs[pi];
                        const float vd = __half2float(vb->d);
                        const float p = probs[kk];
                        out0 += p * ((float(int(pk & 0x0f)) - 8.0f) * vd);
                        out1 += p * ((float(int(pk >> 4)) - 8.0f) * vd);
                    }
                }
            }
        } else if (tid < DECODE_D) {
            out0 *= sm[2];
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                const int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    const int k = k0 + kk;
                    out0 += probs[kk] * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                }
            }
        }
        __syncthreads();
    }

    const size_t base = ((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * DECODE_D;
    if constexpr (Q4PAIR) {
        if (tid < 128) {
            const int pt = tid, pb = pt >> 4, pi = pt & 15;
            const float norm = 1.0f / sm[1];
            dst[base + pb * 32 + pi]      = out0 * norm;
            dst[base + pb * 32 + pi + 16] = out1 * norm;
        }
    } else if (tid < DECODE_D) {
        dst[base + tid] = out0 / sm[1];
    }
}


// ── D-split packed16 decode — one CTA per output-D slice ────────────────
// Debug/experimental lane for exposing more CTAs per decoded token. Each D
// slice recomputes QK+softmax and writes a disjoint dst[D] range, so there is
// no inter-CTA reduction and no graph-unsafe scratch. This is intentionally
// opt-in only: it trades duplicate QK work for extra PV/output parallelism.
template <int BN, int BN_VSUB, int D_CHUNK>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_dsplit_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {

    if (nq != 1) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");
    static_assert(DECODE_D % D_CHUNK == 0, "D_CHUNK must divide D");

    const int hq = blockIdx.x;
    const int ds = blockIdx.y;
    const int b  = blockIdx.z;
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;
    const int tid = threadIdx.x;
    const int dim = ds * D_CHUNK + tid;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);          // BN
    float * probs  = logits + BN;                              // BN
    float * sm     = probs + BN;                               // 4: row_m, row_l, old_scale, pad
    int   * q_i32  = reinterpret_cast<int *>(sm + 4);          // 64 ints
    float * q_scl  = reinterpret_cast<float *>(q_i32 + DECODE_I32_PER_ROW); // 8 floats

    if (tid == 0) { sm[0] = -1e38f; sm[1] = 0.0f; }
    __syncthreads();

    for (int i = tid; i < DECODE_I32_PER_ROW; i += blockDim.x) {
        const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
        q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + i];
    }
    for (int i = tid; i < DECODE_N_BLOCKS; i += blockDim.x) {
        const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
        q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + i];
    }
    __syncthreads();

    float out = 0.0f;
    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        if (tid < BN) {
            const int kk = tid;
            const int k  = k0 + kk;
            float s = -1e38f;
            if (kk < tile_n && k < nk && (k - q_offset <= 0)) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(q_i32, q_scl,
                    k_payload + kb * DECODE_I32_PER_ROW, k_scales + kb * DECODE_N_BLOCKS) * scale;
            }
            logits[kk] = s;
        }
        __syncthreads();

        if (tid == 0) {
            float tile_m = -1e38f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[kk]);
            const float m_prev = sm[0], l_prev = sm[1];
            const float m_new = fmaxf(m_prev, tile_m);
            const float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) {
                const float p = expf(logits[kk] - m_new);
                probs[kk] = p;
                tile_l += p;
            }
            sm[0] = m_new;
            sm[1] = l_prev * old_scale + tile_l;
            sm[2] = old_scale;
        }
        __syncthreads();

        if (tid < D_CHUNK && dim < DECODE_D) {
            out *= sm[2];
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                const int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    const int k = k0 + kk;
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim);
                    out += probs[kk] * v;
                }
            }
        }
        __syncthreads();
    }

    if (tid < D_CHUNK && dim < DECODE_D) {
        const size_t dst_base = ((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * DECODE_D;
        dst[dst_base + dim] = out / sm[1];
    }
}


// ── Grouped-GQA packed16 decode — one CTA per KV head ───────────────────
// Reuses each K/V tile across the query heads sharing one KV head. This keeps
// the packed16 K layout native and reduces V dequant/read duplication for GQA
// models (e.g. gqa=6/8) without changing cache format or model config.
template <int BN, int BN_VSUB, int GH_MAX>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_gqa_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {

    if (nq != 1 || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    const int hk = blockIdx.x;
    const int b  = blockIdx.y;
    if (b >= batch || hk >= n_heads_k) return;
    const int tid = threadIdx.x;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);                    // GH_MAX * BN
    float * probs  = logits + GH_MAX * BN;                                // GH_MAX * BN
    float * sm     = probs  + GH_MAX * BN;                                // GH_MAX * 4
    int   * q_i32  = reinterpret_cast<int *>(sm + GH_MAX * 4);            // GH_MAX * 64
    float * q_scl  = reinterpret_cast<float *>(q_i32 + GH_MAX * DECODE_I32_PER_ROW); // GH_MAX * 8

    for (int i = tid; i < GH_MAX * 4; i += blockDim.x) {
        const int slot = i & 3;
        sm[i] = slot == 0 ? -1e38f : 0.0f;
    }
    __syncthreads();

    float out[GH_MAX];
#pragma unroll
    for (int g = 0; g < GH_MAX; ++g) out[g] = 0.0f;

    for (int i = tid; i < GH_MAX * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int g = i / DECODE_I32_PER_ROW;
        const int j = i - g * DECODE_I32_PER_ROW;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
            q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + j];
        }
    }
    for (int i = tid; i < GH_MAX * DECODE_N_BLOCKS; i += blockDim.x) {
        const int g = i / DECODE_N_BLOCKS;
        const int j = i - g * DECODE_N_BLOCKS;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
            q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + j];
        }
    }
    __syncthreads();

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
            const int g  = i / BN;
            const int kk = i - g * BN;
            const int k  = k0 + kk;
            float s = -1e38f;
            if (g < gh && kk < tile_n && k < nk && (k - q_offset <= 0)) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_i32 + g * DECODE_I32_PER_ROW,
                    q_scl + g * DECODE_N_BLOCKS,
                    k_payload + kb * DECODE_I32_PER_ROW,
                    k_scales + kb * DECODE_N_BLOCKS) * scale;
            }
            logits[g * BN + kk] = s;
        }
        __syncthreads();

        if (tid < GH_MAX) {
            const int g = tid;
            if (g < gh) {
                float tile_m = -1e38f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[g * BN + kk]);
                float m_prev = sm[g * 4 + 0], l_prev = sm[g * 4 + 1];
                float m_new = fmaxf(m_prev, tile_m);
                float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) {
                    float p = expf(logits[g * BN + kk] - m_new);
                    probs[g * BN + kk] = p;
                    tile_l += p;
                }
                sm[g * 4 + 0] = m_new;
                sm[g * 4 + 1] = l_prev * old_scale + tile_l;
                sm[g * 4 + 2] = old_scale;
            }
        }
        __syncthreads();

        if (tid < DECODE_D) {
#pragma unroll
            for (int g = 0; g < GH_MAX; ++g) {
                if (g < gh) out[g] *= sm[g * 4 + 2];
            }
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    const int k = k0 + kk;
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                    for (int g = 0; g < GH_MAX; ++g) {
                        if (g < gh) out[g] += probs[g * BN + kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < DECODE_D) {
#pragma unroll
        for (int g = 0; g < GH_MAX; ++g) {
            if (g < gh) {
                const int hq = hq0 + g;
                const size_t dst_base = ((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * DECODE_D;
                dst[dst_base + tid] = out[g] / sm[g * 4 + 1];
            }
        }
    }
}


// ── Small-verify grouped-GQA scalar kernel ───────────────────────────────
// One CTA per (query row, KV head). This is intentionally separate from the
// decode split-K lane: nq=2..4 verification gets per-q causal bounds without
// inter-CTA partial/reduce state.
template <int BN, int BN_VSUB, int GH_MAX>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_small_verify_gqa_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        float scale,
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
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {

    if (nq < 2 || nq > 4 || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    const int q  = blockIdx.x;
    const int hk = blockIdx.y;
    const int b  = blockIdx.z;
    if (b >= batch || q >= nq || hk >= n_heads_k) return;
    const int tid = threadIdx.x;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * probs  = logits + GH_MAX * BN;
    float * sm     = probs  + GH_MAX * BN;
    int   * q_i32  = reinterpret_cast<int *>(sm + GH_MAX * 4);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + GH_MAX * DECODE_I32_PER_ROW);

    for (int i = tid; i < GH_MAX * 4; i += blockDim.x) {
        const int slot = i & 3;
        sm[i] = slot == 0 ? -1e38f : 0.0f;
    }
    __syncthreads();

    float out[GH_MAX];
#pragma unroll
    for (int g = 0; g < GH_MAX; ++g) out[g] = 0.0f;

    for (int i = tid; i < GH_MAX * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int g = i / DECODE_I32_PER_ROW;
        const int j = i - g * DECODE_I32_PER_ROW;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq) + size_t(q));
            q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + j];
        }
    }
    for (int i = tid; i < GH_MAX * DECODE_N_BLOCKS; i += blockDim.x) {
        const int g = i / DECODE_N_BLOCKS;
        const int j = i - g * DECODE_N_BLOCKS;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq) + size_t(q));
            q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + j];
        }
    }
    __syncthreads();

    const int q_pos = q_offset + q;
    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
            const int g  = i / BN;
            const int kk = i - g * BN;
            const int k  = k0 + kk;
            float s = -1e38f;
            if (g < gh && kk < tile_n && k < nk && (mask || k <= q_pos)) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_i32 + g * DECODE_I32_PER_ROW,
                    q_scl + g * DECODE_N_BLOCKS,
                    k_payload + kb * DECODE_I32_PER_ROW,
                    k_scales + kb * DECODE_N_BLOCKS) * scale;
                if (mask) {
                    s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                }
            }
            logits[g * BN + kk] = s;
        }
        __syncthreads();

        if (tid < GH_MAX) {
            const int g = tid;
            if (g < gh) {
                float tile_m = -1e38f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[g * BN + kk]);
                float m_prev = sm[g * 4 + 0], l_prev = sm[g * 4 + 1];
                float m_new = fmaxf(m_prev, tile_m);
                float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) {
                    float p = expf(logits[g * BN + kk] - m_new);
                    probs[g * BN + kk] = p;
                    tile_l += p;
                }
                sm[g * 4 + 0] = m_new;
                sm[g * 4 + 1] = l_prev * old_scale + tile_l;
                sm[g * 4 + 2] = old_scale;
            }
        }
        __syncthreads();

        if (tid < DECODE_D) {
#pragma unroll
            for (int g = 0; g < GH_MAX; ++g) {
                if (g < gh) out[g] *= sm[g * 4 + 2];
            }
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    const int k = k0 + kk;
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                    for (int g = 0; g < GH_MAX; ++g) {
                        if (g < gh) out[g] += probs[g * BN + kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < DECODE_D) {
#pragma unroll
        for (int g = 0; g < GH_MAX; ++g) {
            if (g < gh) {
                const int hq = hq0 + g;
                const float l = sm[g * 4 + 1];
                const size_t dst_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * DECODE_D;
                dst[dst_base + tid] = l > 0.0f ? out[g] / l : 0.0f;
            }
        }
    }
}


// ── Grouped-GQA PV-WMMA decode — scalar QK, WMMA P×V ───────────────────
// One CTA per KV head. QK/probs are computed exactly like grouped-GQA scalar,
// then q4_0 V projection is performed by RDNA3 f16 WMMA tiles:
//     P[16 x 16] * V[16 x 16] -> O[16 x 16]
// Rows above GH are padded with zero, so GH={1,2,4,6,8} share one kernel.
// When WMMA_QK=true, QK also uses RDNA3 i8 WMMA over packed16 Q/K fragments.
template <int BN, int GH_MAX, bool WMMA_QK = false>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_gqa_pvwmma_p16_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows) {

    if (nq != 1 || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;
    static_assert(BN % 16 == 0, "PV-WMMA decode requires BN multiple of 16");

    const int hk = blockIdx.x;
    const int b  = blockIdx.y;
    if (b >= batch || hk >= n_heads_k) return;

    const int tid     = threadIdx.x;
    const int wave    = tid >> 5;     // 0..7
    const int lane    = tid & 31;
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);                    // GH_MAX * BN
    float * probs  = logits + GH_MAX * BN;                                // GH_MAX * BN
    float * sm     = probs  + GH_MAX * BN;                                // GH_MAX * 4
    int   * q_i32  = reinterpret_cast<int *>(sm + GH_MAX * 4);            // GH_MAX * 64
    float * q_scl  = reinterpret_cast<float *>(q_i32 + GH_MAX * DECODE_I32_PER_ROW); // GH_MAX * 8

    for (int i = tid; i < GH_MAX * 4; i += blockDim.x) {
        const int slot = i & 3;
        sm[i] = slot == 0 ? -1e38f : 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < GH_MAX * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int g = i / DECODE_I32_PER_ROW;
        const int j = i - g * DECODE_I32_PER_ROW;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
            q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + j];
        }
    }
    for (int i = tid; i < GH_MAX * DECODE_N_BLOCKS; i += blockDim.x) {
        const int g = i / DECODE_N_BLOCKS;
        const int j = i - g * DECODE_N_BLOCKS;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
            q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + j];
        }
    }
    __syncthreads();

    float out_pv[2][8];
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out_pv[pass][i] = 0.0f;
    }

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = (k0 + BN <= nk) ? BN : (nk - k0);

        if constexpr (WMMA_QK) {
            for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
                const int g  = i / BN;
                const int kk = i - g * BN;
                const int k  = k0 + kk;
                logits[i] = (g < gh && kk < tile_n && k < nk && (k - q_offset <= 0)) ? 0.0f : -1e38f;
            }
            __syncthreads();

            if (wave < BN / 16) {
                const int kk_base = wave * 16;
                const int b_col = pbwmma_i8_b_col_from_lane_lo(lane_lo);
                const int a_row = pbwmma_i8_a_row_from_lane_lo(lane_lo);
                const int kk = kk_base + b_col;
                const int k  = k0 + kk;
                const bool valid_col = kk < tile_n && k < nk && (k - q_offset <= 0);
#pragma unroll
                for (int tc = 0; tc < DECODE_D / 16; ++tc) {
                    pbwmma_v4i32 a_frag;
                    pbwmma_v4i32 b_frag;
#pragma unroll
                    for (int wi = 0; wi < 4; ++wi) {
                        const int word = tc * 4 + wi;
                        a_frag[wi] = (a_row < gh) ? q_i32[a_row * DECODE_I32_PER_ROW + word] : 0;
                        b_frag[wi] = valid_col ? k_payload[(k_head_base + size_t(k)) * DECODE_I32_PER_ROW + word] : 0;
                    }
                    pbwmma_v8i32 acc = {0,0,0,0,0,0,0,0};
                    acc = pbwmma_mma_i8(a_frag, b_frag, acc);
                    const int sb = tc >> 1;
                    const float ks = valid_col ? __half2float(k_scales[(k_head_base + size_t(k)) * DECODE_N_BLOCKS + sb]) : 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int row = pbwmma_i8_d_row_from_acc(i, lane_hi);
                        if (row < gh && valid_col) {
                            logits[row * BN + kk] += float(acc[i]) * q_scl[row * DECODE_N_BLOCKS + sb] * ks * scale;
                        }
                    }
                }
            }
        } else {
            for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
                const int g  = i / BN;
                const int kk = i - g * BN;
                const int k  = k0 + kk;
                float s = -1e38f;
                if (g < gh && kk < tile_n && k < nk && (k - q_offset <= 0)) {
                    const size_t kb = k_head_base + k;
                    s = ggml_cuda_q8k_dot4_kq_dot_direct(
                        q_i32 + g * DECODE_I32_PER_ROW,
                        q_scl + g * DECODE_N_BLOCKS,
                        k_payload + kb * DECODE_I32_PER_ROW,
                        k_scales + kb * DECODE_N_BLOCKS) * scale;
                }
                logits[g * BN + kk] = s;
            }
        }
        __syncthreads();

        if (tid < GH_MAX) {
            const int g = tid;
            if (g < gh) {
                float tile_m = -1e38f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[g * BN + kk]);
                const float m_prev = sm[g * 4 + 0], l_prev = sm[g * 4 + 1];
                const float m_new = fmaxf(m_prev, tile_m);
                const float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) {
                    const float p = expf(logits[g * BN + kk] - m_new);
                    probs[g * BN + kk] = p;
                    tile_l += p;
                }
                sm[g * 4 + 0] = m_new;
                sm[g * 4 + 1] = l_prev * old_scale + tile_l;
                sm[g * 4 + 2] = old_scale;
            }
        }
        __syncthreads();

#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
                if (row < gh) out_pv[pass][i] *= sm[row * 4 + 2];
            }
        }

        for (int sub = 0; sub < BN; sub += 16) {
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int d_tile = (pass * 8 + wave) * 16;
                pbwmma_v16fp16 a_frag;
                pbwmma_v16fp16 b_frag;
                const int a_row = pbwmma_f16_a_row_from_lane_lo(lane_lo);
                const int b_col = pbwmma_f16_b_col_from_lane_lo(lane_lo);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kk = sub + i;
                    const int k  = k0 + kk;
                    a_frag[i] = (_Float16)((a_row < gh && kk < tile_n) ? probs[a_row * BN + kk] : 0.0f);
                    b_frag[i] = (_Float16)((kk < tile_n && k < nk && d_tile + b_col < DECODE_D) ?
                        ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d_tile + b_col) : 0.0f);
                }
                pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
                acc = pbwmma_mma(a_frag, b_frag, acc);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
                    if (row < gh) out_pv[pass][i] += acc[i];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int d_tile = (pass * 8 + wave) * 16;
        const int d_col  = d_tile + lane_lo;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
            if (row < gh && d_col < DECODE_D) {
                const int hq = hq0 + row;
                const float l = sm[row * 4 + 1];
                const size_t dst_base = ((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * DECODE_D;
                dst[dst_base + d_col] = out_pv[pass][i] / l;
            }
        }
    }
}


// ── Grouped-GQA Small-verify split-K Stage 1: one CTA per (K split, KV head, query) ─
// All GH_MAX query heads in a GQA group share the K load for their split range.
// Grid: (n_splits, n_heads_k, nq, batch)
// Partial buffers: partial_o[batch][nq][n_heads_k][n_splits][GH_MAX][D]
//                  partial_m[batch][nq][n_heads_k][n_splits][GH_MAX]
//                  partial_l[batch][nq][n_heads_k][n_splits][GH_MAX]
//
// When USE_WMMA_PV is true, P×V uses RDNA3 f16 WMMA tiles for 2-3× speedup.
template <int BN, int BN_VSUB, int GH_MAX>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_small_verify_gqa_splitk_stage1_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ partial_o,   // [batch, nq, n_heads_k, n_splits, GH_MAX, D]
        float       * __restrict__ partial_m,   // [batch, nq, n_heads_k, n_splits, GH_MAX]
        float       * __restrict__ partial_l,   // [batch, nq, n_heads_k, n_splits, GH_MAX]
        float scale,
        int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
        int64_t nb30, int64_t nb31, int64_t nb33, int64_t ne33,
        int nq, int nk, int n_heads_q, int n_heads_k,
        int gqa_ratio, int batch, int q_offset,
        int k_head_stride_rows, int k_batch_stride_rows,
        int split_size, int n_splits) {

    if (nq < 2 || nq > 4 || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    const int split = blockIdx.x;
    const int hk    = blockIdx.y;
    const int q     = blockIdx.z % nq;
    const int b     = blockIdx.z / nq;
    if (b >= batch || q >= nq || hk >= n_heads_k || split >= n_splits) return;

    const int tid = threadIdx.x;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    const int k_begin = split * split_size;
    const int k_end   = min(nk, k_begin + split_size);

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);                        // GH_MAX * BN
    float * probs  = logits + GH_MAX * BN;                                   // GH_MAX * BN
    float * sm     = probs  + GH_MAX * BN;                                   // GH_MAX * 4
    int   * q_i32  = reinterpret_cast<int *>(sm + GH_MAX * 4);              // GH_MAX * DECODE_I32_PER_ROW
    float * q_scl  = reinterpret_cast<float *>(q_i32 + GH_MAX * DECODE_I32_PER_ROW); // GH_MAX * DECODE_N_BLOCKS

    // Initialise running softmax state per GH head
    for (int i = tid; i < GH_MAX * 4; i += blockDim.x) {
        const int slot = i & 3;
        sm[i] = slot == 0 ? -1e38f : 0.0f;
    }
    __syncthreads();

    float out[GH_MAX];
#pragma unroll
    for (int g = 0; g < GH_MAX; ++g) out[g] = 0.0f;

    // Load Q payload for all gh heads
    for (int i = tid; i < GH_MAX * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int g = i / DECODE_I32_PER_ROW;
        const int j = i - g * DECODE_I32_PER_ROW;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + hq) * size_t(nq) + size_t(q));
            q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + j];
        }
    }
    for (int i = tid; i < GH_MAX * DECODE_N_BLOCKS; i += blockDim.x) {
        const int g = i / DECODE_N_BLOCKS;
        const int j = i - g * DECODE_N_BLOCKS;
        if (g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + hq) * size_t(nq) + size_t(q));
            q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + j];
        }
    }
    __syncthreads();

    const int q_pos = q_offset + q;
    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = (k0 + BN <= k_end) ? BN : (k_end - k0);

        // QK dot for all gh heads — shares the same K tile across GQA group
        for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
            const int g  = i / BN;
            const int kk = i - g * BN;
            const int k  = k0 + kk;
            float s = -1e38f;
            if (g < gh && kk < tile_n && k < nk && (mask || k <= q_pos)) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_i32 + g * DECODE_I32_PER_ROW,
                    q_scl + g * DECODE_N_BLOCKS,
                    k_payload + kb * DECODE_I32_PER_ROW,
                    k_scales + kb * DECODE_N_BLOCKS) * scale;
                if (mask) {
                    s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                }
            }
            logits[g * BN + kk] = s;
        }
        __syncthreads();

        // Softmax reduction per gh head
        if (tid < GH_MAX) {
            const int g = tid;
            if (g < gh) {
                float tile_m = -1e38f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[g * BN + kk]);
                float m_prev = sm[g * 4 + 0], l_prev = sm[g * 4 + 1];
                float m_new = fmaxf(m_prev, tile_m);
                float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) {
                    float p = expf(logits[g * BN + kk] - m_new);
                    probs[g * BN + kk] = p;
                    tile_l += p;
                }
                sm[g * 4 + 0] = m_new;
                sm[g * 4 + 1] = l_prev * old_scale + tile_l;
                sm[g * 4 + 2] = old_scale;
            }
        }
        __syncthreads();

        // PV accumulation — same V row read once, accumulated into gh heads
        if (tid < DECODE_D) {
#pragma unroll
            for (int g = 0; g < GH_MAX; ++g) {
                if (g < gh) out[g] *= sm[g * 4 + 2];
            }
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    const int k = k0 + kk;
                    const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                    for (int g = 0; g < GH_MAX; ++g) {
                        if (g < gh) out[g] += probs[g * BN + kk] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write unnormalised partials per GH head
    if (tid < DECODE_D) {
#pragma unroll
        for (int g = 0; g < GH_MAX; ++g) {
            if (g < gh) {
                const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(split);
                partial_o[(partial_base * GH_MAX + g) * DECODE_D + tid] = out[g];
            }
        }
    }
    if (tid < GH_MAX) {
        const int g = tid;
        if (g < gh) {
            const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(split);
            partial_m[partial_base * GH_MAX + g] = sm[g * 4 + 0];
            partial_l[partial_base * GH_MAX + g] = sm[g * 4 + 1];
        }
    }
}

// ── Grouped-GQA Batched-Q split-K Stage 1: K/V shared across all nq queries ──
//
// Grid: (n_splits, n_heads_k, batch) — one CTA per (split, kv_head, batch).
// All nq queries are processed inside a single CTA, sharing K and V reads.
// This eliminates the nq× redundant K/V bandwidth of the per-query stage1.
//
// K/V are streamed through global memory ONCE per tile. For each BN-wide tile
// of K positions, all nq×gh dot products are computed, then softmax + P×V
// are accumulated per-query into per-query partial registers.
//
// Partial output layout: [batch][nq][n_heads_k][n_splits][GH_MAX][D]
//                          partial_m[batch][nq][n_heads_k][n_splits][GH_MAX]
//                          partial_l[batch][nq][n_heads_k][n_splits][GH_MAX]
// Same layout as the per-query stage1 for compatibility with the existing reduce kernel.
//
template <int BN, int BN_VSUB, int GH_MAX, int NQ_MAX, bool FA4_QKPV = false, bool FA4_PVWMMA = false>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_small_verify_batched_gqa_splitk_stage1_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ partial_o,   // [batch, nq, n_heads_k, n_splits, GH_MAX, D]
        float       * __restrict__ partial_m,   // [batch, nq, n_heads_k, n_splits, GH_MAX]
        float       * __restrict__ partial_l,   // [batch, nq, n_heads_k, n_splits, GH_MAX]
        float scale,
        int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
        int64_t nb30, int64_t nb31, int64_t nb33, int64_t ne33,
        int nq, int nk, int n_heads_q, int n_heads_k,
        int gqa_ratio, int batch, int q_offset,
        int k_head_stride_rows, int k_batch_stride_rows,
        int split_size, int n_splits,
        small_verify_fa4_profile * __restrict__ profile) {

    if (nq < 2 || nq > NQ_MAX || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");

    const int split = blockIdx.x;
    const int hk   = blockIdx.y;
    const int b    = blockIdx.z;
    if (b >= batch || hk >= n_heads_k || split >= n_splits) return;
    const int tid = threadIdx.x;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    const int k_begin = split * split_size;
    const int k_end   = min(nk, k_begin + split_size);

    extern __shared__ __align__(16) unsigned char smem[];
    // Shared memory layout:
    //   q_i32[NQ_MAX * GH_MAX * DECODE_I32_PER_ROW]  — all Q payloads
    //   q_scl[NQ_MAX * GH_MAX * DECODE_N_BLOCKS]     — all Q scales
    //   k_i32[BN * DECODE_I32_PER_ROW]                — K tile payload, loaded once from global
    //   k_scl[BN] or [BN * DECODE_N_BLOCKS]           — K tile scales; FA4 keeps all 8 blocks
    //   logits[NQ_MAX * GH_MAX * BN]                  — ALL Q×K scores for this K tile
    //   probs[NQ_MAX * GH_MAX * BN]                   — post-softmax probs
    //   sm[NQ_MAX * GH_MAX * 4]                        — softmax running state per (q,g)
    constexpr int K_SCALE_SMEM = FA4_QKPV ? BN * DECODE_N_BLOCKS : BN;
    int   * q_i32  = reinterpret_cast<int   *>(smem);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + NQ_MAX * GH_MAX * DECODE_I32_PER_ROW);
    int   * k_i32  = reinterpret_cast<int   *>(q_scl + NQ_MAX * GH_MAX * DECODE_N_BLOCKS);
    float * k_scl  = reinterpret_cast<float *>(k_i32 + BN * DECODE_I32_PER_ROW);
    float * logits  = k_scl + K_SCALE_SMEM;
    float * probs   = logits + NQ_MAX * GH_MAX * BN;
    float * sm      = probs  + NQ_MAX * GH_MAX * BN;

    // Load all Q payloads and scales for all nq queries across gh heads
    for (int i = tid; i < NQ_MAX * GH_MAX * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int slot = i / DECODE_I32_PER_ROW;
        const int q_idx = slot / GH_MAX;
        const int g     = slot - q_idx * GH_MAX;
        const int j     = i - slot * DECODE_I32_PER_ROW;
        int val = 0;
        if (q_idx < nq && g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q_idx));
            val = q_payload[q_base * DECODE_I32_PER_ROW + j];
        }
        q_i32[i] = val;
    }
    for (int i = tid; i < NQ_MAX * GH_MAX * DECODE_N_BLOCKS; i += blockDim.x) {
        const int slot = i / DECODE_N_BLOCKS;
        const int q_idx = slot / GH_MAX;
        const int g     = slot - q_idx * GH_MAX;
        const int j     = i - slot * DECODE_N_BLOCKS;
        float val = 0.0f;
        if (q_idx < nq && g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q_idx));
            val = q_scales[q_base * DECODE_N_BLOCKS + j];
        }
        q_scl[i] = val;
    }

    // Initialize softmax state per (q, g)
    for (int i = tid; i < NQ_MAX * GH_MAX * 4; i += blockDim.x) {
        const int slot = i / 4;
        const int off  = i - slot * 4;
        sm[i] = (off == 0) ? -1e38f : 0.0f;
    }
    __syncthreads();

    // Per-thread scalar output accumulator: out[q_idx][g]
    float out[NQ_MAX][GH_MAX];
    if constexpr (!FA4_PVWMMA) {
#pragma unroll
        for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx)
#pragma unroll
            for (int g = 0; g < GH_MAX; ++g)
                out[q_idx][g] = 0.0f;
    }

    // PV-WMMA output accumulator. Two row blocks cover up to NQ_MAX*GH_MAX=32 logical rows;
    // each pass/wave covers one 16-column D tile, matching the decode PV-WMMA layout.
    constexpr int FA4_PV_ROW_BLOCKS = (NQ_MAX * GH_MAX + 15) / 16;
    float out_pv[FA4_PV_ROW_BLOCKS][2][8];
    if constexpr (FA4_PVWMMA) {
#pragma unroll
        for (int rb = 0; rb < FA4_PV_ROW_BLOCKS; ++rb)
#pragma unroll
            for (int pass = 0; pass < 2; ++pass)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    out_pv[rb][pass][i] = 0.0f;
    }

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    unsigned long long profile_t0 = 0;

    // ── Main K/V tile loop: K/V read ONCE per tile, Q×K computed for ALL nq in one sweep ──
    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = (k0 + BN <= k_end) ? BN : (k_end - k0);

        // ── Phase 1: ALL Q×K scores in one parallel sweep using DOT4 batched across 4 queries ──
        SMALL_VERIFY_PROFILE_PHASE_BEGIN();
        // Each thread processes up to 4 consecutive logits positions (q_idx, g, kk) where
        // (g, kk) are shared and K is loaded ONCE for all 4 queries. DOT4 computes 4 I8
        // element-products per instruction; with 4 query chains interleaved, K elements are
        // reused 4×, reducing K memory traffic by ~4× vs per-query dot calls.
        // Load the K tile once into shared memory. This preserves the FA2 reuse
        // invariant while allowing the QK compute below to run with full
        // (NQ*GH*BN) parallelism instead of serialising all q per thread.
        for (int i = tid; i < BN * DECODE_I32_PER_ROW; i += blockDim.x) {
            const int kk  = i / DECODE_I32_PER_ROW;
            const int pos = i - kk * DECODE_I32_PER_ROW;
            const int k   = k0 + kk;
            int val = 0;
            if (kk < tile_n && k < nk) {
                const size_t kb = k_head_base + k;
                val = k_payload[kb * DECODE_I32_PER_ROW + pos];
            }
            k_i32[i] = val;
        }
        if constexpr (FA4_QKPV) {
            for (int i = tid; i < BN * DECODE_N_BLOCKS; i += blockDim.x) {
                const int kk = i / DECODE_N_BLOCKS;
                const int qb = i - kk * DECODE_N_BLOCKS;
                const int k  = k0 + kk;
                float val = 0.0f;
                if (kk < tile_n && k < nk) {
                    const size_t kb = k_head_base + k;
                    val = __half2float(k_scales[kb * DECODE_N_BLOCKS + qb]);
                }
                k_scl[i] = val;
            }
        } else {
            for (int kk = tid; kk < BN; kk += blockDim.x) {
                const int k = k0 + kk;
                float val = 0.0f;
                if (kk < tile_n && k < nk) {
                    const size_t kb = k_head_base + k;
                    val = __half2float(k_scales[kb * DECODE_N_BLOCKS]);
                }
                k_scl[kk] = val;
            }
        }
        __syncthreads();

        if constexpr (FA4_QKPV) {
            // FA4 lane model: one thread owns (g, kk), loads K words once,
            // and computes up to four q lanes as independent accumulator chains.
            // This makes q0/q1/q2/q3 structural lanes instead of separate logits work-items.
            for (int idx = tid; idx < GH_MAX * BN; idx += blockDim.x) {
                const int g  = idx / BN;
                const int kk = idx - g * BN;
                const int k  = k0 + kk;
                float sums[NQ_MAX];
#pragma unroll
                for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) sums[q_idx] = 0.0f;

                const bool have_base = g < gh && kk < tile_n && k < nk;
#pragma unroll
                for (int qb = 0; qb < DECODE_N_BLOCKS; ++qb) {
                    int acc[NQ_MAX];
#pragma unroll
                    for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) acc[q_idx] = 0;
#pragma unroll
                    for (int word = 0; word < QK8_0 / 4; ++word) {
                        const int pos = qb * (QK8_0 / 4) + word;
                        const int k_val = k_i32[kk * DECODE_I32_PER_ROW + pos];
#pragma unroll
                        for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                            if (q_idx < nq && have_base) {
                                const int * qp = q_i32 + (q_idx * GH_MAX + g) * DECODE_I32_PER_ROW;
                                acc[q_idx] = ggml_cuda_q8k_dot4_i8_i8(qp[pos], k_val, acc[q_idx]);
                            }
                        }
                    }
#pragma unroll
                    for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                        if (q_idx < nq && have_base) {
                            const float * qs = q_scl + (q_idx * GH_MAX + g) * DECODE_N_BLOCKS;
                            sums[q_idx] += float(acc[q_idx]) * qs[qb] * k_scl[kk * DECODE_N_BLOCKS + qb];
                        }
                    }
                }
#pragma unroll
                for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                    float s = -1e38f;
                    if (q_idx < nq && have_base) {
                        const int q_pos = q_offset + q_idx;
                        const bool valid = mask ? true : (k <= q_pos);
                        if (valid) {
                            const float raw = sums[q_idx] * scale;
                            s = mask ? raw + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_idx, k, b) : raw;
                        }
                    }
                    logits[q_idx * GH_MAX * BN + g * BN + kk] = s;
                }
            }
        } else {
            const int NQK_STRIDE = NQ_MAX * GH_MAX * BN;
            for (int idx = tid; idx < NQK_STRIDE; idx += blockDim.x) {
                const int q_idx = idx / (GH_MAX * BN);
                const int guard = idx - q_idx * (GH_MAX * BN);
                const int g     = guard / BN;
                const int kk    = guard - g * BN;
                const int k     = k0 + kk;

                int acc = 0;
                const bool have = q_idx < nq && g < gh && kk < tile_n && k < nk;
                const int * qp = q_i32 + (q_idx * GH_MAX + g) * DECODE_I32_PER_ROW;
#pragma unroll
                for (int pos = 0; pos < DECODE_I32_PER_ROW; ++pos) {
                    const int k_val = k_i32[kk * DECODE_I32_PER_ROW + pos];
                    acc = have ? ggml_cuda_q8k_dot4_i8_i8(qp[pos], k_val, acc) : acc;
                }

                float s = -1e38f;
                if (have) {
                    const int q_pos = q_offset + q_idx;
                    bool valid = mask ? true : (k <= q_pos);
                    if (valid) {
                        const float * qs = q_scl + (q_idx * GH_MAX + g) * DECODE_N_BLOCKS;
                        const float raw = float(acc) * qs[0] * k_scl[kk] * scale;
                        s = mask ? raw + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_idx, k, b) : raw;
                    }
                }
                logits[idx] = s;
            }
        }
        __syncthreads();
        SMALL_VERIFY_PROFILE_PHASE_END(qk_cycles);

        // ── Phase 2: All-query softmax in one phase ──
        SMALL_VERIFY_PROFILE_PHASE_BEGIN();
        // Each thread handles one (q_idx, g) pair.
        if (tid < NQ_MAX * GH_MAX) {
            const int q_idx = tid / GH_MAX;
            const int g     = tid - q_idx * GH_MAX;
            if (q_idx < nq && g < gh) {
                float tile_m = -1e38f;
                const int logit_base = q_idx * GH_MAX * BN + g * BN;
                for (int kk = 0; kk < tile_n; ++kk) tile_m = fmaxf(tile_m, logits[logit_base + kk]);
                float m_prev = sm[(q_idx * GH_MAX + g) * 4 + 0];
                float l_prev = sm[(q_idx * GH_MAX + g) * 4 + 1];
                float m_new  = fmaxf(m_prev, tile_m);
                float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                float tile_l = 0.0f;
                for (int kk = 0; kk < tile_n; ++kk) {
                    float p = expf(logits[logit_base + kk] - m_new);
                    probs[logit_base + kk] = p;
                    tile_l += p;
                }
                sm[(q_idx * GH_MAX + g) * 4 + 0] = m_new;
                sm[(q_idx * GH_MAX + g) * 4 + 1] = l_prev * old_scale + tile_l;
                sm[(q_idx * GH_MAX + g) * 4 + 2] = old_scale;
            }
        }
        __syncthreads();
        SMALL_VERIFY_PROFILE_PHASE_END(softmax_cycles);

        // ── Phase 3: All-query P×V accumulation in one phase ──
        SMALL_VERIFY_PROFILE_PHASE_BEGIN();
        if constexpr (FA4_PVWMMA) {
            // FA4 PV-WMMA row layout is packed by active GH:
            //   row = q_idx * gh + g
            // Two 16-row WMMA blocks cover nq<=4, gh<=8. B/V fragments are shared
            // across both row blocks for each D tile, so V traffic is not multiplied
            // by query lane count as in per-query decode-style PV.
            const int wave    = tid >> 5;
            const int lane    = tid & 31;
            const int lane_lo = lane & 15;
            const int lane_hi = lane >> 4;
#pragma unroll
            for (int rb = 0; rb < FA4_PV_ROW_BLOCKS; ++rb) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int row = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                        const int q_idx = row / gh;
                        const int g = row - q_idx * gh;
                        if (q_idx < nq && g < gh) {
                            out_pv[rb][pass][i] *= sm[(q_idx * GH_MAX + g) * 4 + 2];
                        }
                    }
                }
            }
            for (int sub = 0; sub < BN; sub += 16) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int d_tile = (pass * 8 + wave) * 16;
                    const int b_col = pbwmma_f16_b_col_from_lane_lo(lane_lo);
                    pbwmma_v16fp16 b_frag;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int kk = sub + i;
                        const int k  = k0 + kk;
                        b_frag[i] = (_Float16)((kk < tile_n && k < nk && d_tile + b_col < DECODE_D) ?
                            ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d_tile + b_col) : 0.0f);
                    }
#pragma unroll
                    for (int rb = 0; rb < FA4_PV_ROW_BLOCKS; ++rb) {
                        pbwmma_v16fp16 a_frag;
                        const int a_row = pbwmma_f16_a_row_from_lane_lo(lane_lo);
                        const int row = rb * 16 + a_row;
                        const int q_idx = row / gh;
                        const int g = row - q_idx * gh;
                        const bool valid_row = q_idx < nq && g < gh;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            const int kk = sub + i;
                            a_frag[i] = (_Float16)((valid_row && kk < tile_n) ? probs[(q_idx * GH_MAX + g) * BN + kk] : 0.0f);
                        }
                        pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
                        acc = pbwmma_mma(a_frag, b_frag, acc);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int out_row = rb * 16 + pbwmma_f16_d_row_from_acc(i, lane_hi);
                            const int out_q = out_row / gh;
                            const int out_g = out_row - out_q * gh;
                            if (out_q < nq && out_g < gh) out_pv[rb][pass][i] += acc[i];
                        }
                    }
                }
            }
        } else if (tid < DECODE_D) {
            // Each of D=256 threads iterates across all nq queries for its dimension.
            if constexpr (FA4_QKPV) {
                // FA4 lane model for PV: load/dequant V once for this K row and
                // feed all q lanes. This is the scalar-PV scaffolding retained as
                // the correctness baseline for the PV-WMMA route.
#pragma unroll
                for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                    if (q_idx < nq) {
#pragma unroll
                        for (int g = 0; g < GH_MAX; ++g) {
                            if (g < gh) out[q_idx][g] *= sm[(q_idx * GH_MAX + g) * 4 + 2];
                        }
                    }
                }
                for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                    int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                        for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                            if (q_idx < nq) {
                                const int logit_base = q_idx * GH_MAX * BN;
#pragma unroll
                                for (int g = 0; g < GH_MAX; ++g) {
                                    if (g < gh) out[q_idx][g] += probs[logit_base + g * BN + kk] * v;
                                }
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
                    if (q_idx >= nq) break;
                    const int logit_base = q_idx * GH_MAX * BN;
#pragma unroll
                    for (int g = 0; g < GH_MAX; ++g) {
                        if (g < gh) out[q_idx][g] *= sm[(q_idx * GH_MAX + g) * 4 + 2];
                    }
                    for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                        int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                        for (int kk = sub; kk < end; ++kk) {
                            const int k = k0 + kk;
                            const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                            for (int g = 0; g < GH_MAX; ++g) {
                                if (g < gh) out[q_idx][g] += probs[logit_base + g * BN + kk] * v;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        SMALL_VERIFY_PROFILE_PHASE_END(pv_cycles);
    }

    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    // Write unnormalised partials per (q, g)
    if constexpr (FA4_PVWMMA) {
        const int wave    = tid >> 5;
        const int lane    = tid & 31;
        const int lane_lo = lane & 15;
        (void) wave;
#pragma unroll
        for (int rb = 0; rb < FA4_PV_ROW_BLOCKS; ++rb) {
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int d_col = (pass * 8 + wave) * 16 + lane_lo;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = rb * 16 + pbwmma_f16_d_row_from_acc(i, (lane >> 4));
                    const int q_idx = row / gh;
                    const int g = row - q_idx * gh;
                    if (q_idx < nq && g < gh && d_col < DECODE_D) {
                        const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(split);
                        partial_o[(partial_base * GH_MAX + g) * DECODE_D + d_col] = out_pv[rb][pass][i];
                    }
                }
            }
        }
    } else if (tid < DECODE_D) {
#pragma unroll
        for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
            if (q_idx < nq) {
#pragma unroll
                for (int g = 0; g < GH_MAX; ++g) {
                    if (g < gh) {
                        const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(split);
                        partial_o[(partial_base * GH_MAX + g) * DECODE_D + tid] = out[q_idx][g];
                    }
                }
            }
        }
    }
    if (tid < NQ_MAX * GH_MAX) {
        const int q_idx = tid / GH_MAX;
        const int g     = tid - q_idx * GH_MAX;
        if (q_idx < nq && g < gh) {
            const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(split);
            partial_m[partial_base * GH_MAX + g] = sm[(q_idx * GH_MAX + g) * 4 + 0];
            partial_l[partial_base * GH_MAX + g] = sm[(q_idx * GH_MAX + g) * 4 + 1];
        }
    }
    SMALL_VERIFY_PROFILE_PHASE_END(write_cycles);
}

static constexpr int BM_DOT4_PAGES_PV_IMPL_SCALAR               = 0;
static constexpr int BM_DOT4_PAGES_PV_IMPL_PVWMMA               = 1;
static constexpr int BM_DOT4_PAGES_PV_IMPL_PINT8PV              = 2;
static constexpr int BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4         = 3;
static constexpr int BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4  = 4;
static constexpr int BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA  = 5;

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_quantize_i8_symmetric(
        float x,
        float inv_scale) {
    int qi = inv_scale > 0.0f ? __float2int_rn(x * inv_scale) : 0;
    qi = qi < -127 ? -127 : qi;
    qi = qi > 127 ? 127 : qi;
    return qi;
}

// Positive signed-int8 DOT4 lane: clamp to [0, 127] so the packed bytes stay
// valid for the existing signed int8 DOT4 helper while encoding E ~= exp(logit-m).
static __device__ __forceinline__ int ggml_cuda_q8k_dot4_quantize_i8_positive_unit(
        float x) {
    int qi = __float2int_rn(x * 127.0f);
    qi = qi < 0 ? 0 : qi;
    qi = qi > 127 ? 127 : qi;
    return qi;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_pack_i8x4(
        const int x0,
        const int x1,
        const int x2,
        const int x3) {
    return int(uint32_t(uint8_t(int8_t(x0))) |
               (uint32_t(uint8_t(int8_t(x1))) << 8) |
               (uint32_t(uint8_t(int8_t(x2))) << 16) |
               (uint32_t(uint8_t(int8_t(x3))) << 24));
}

// ── BM/page DOT4 Stage 1: one CTA per (KV page, KV-head row-block, batch) ─
//
// This is the first vLLM-style scheduler scaffold for packed16 MTP verify.
// Unlike the FA4 scaffold above, grid.y contains 16-row blocks inside each KV
// head, so nq*GQA rows are distributed across many CTAs instead of being fused
// into one fat CTA per KV head.  The microkernel keeps scalar DOT4 QK, then
// optionally swaps the PV core between scalar q4_0, f16 PVWMMA, the existing
// signed-int8 weighted-P routes, and an experimental INT-FlashAttention-style
// transient-V-fragment DOT4/WMMA routes.  Its row/page organization matches the
// calculator-backed 16x16 WMMA layout in
// archived-code/20260623-nonruntime-experimental-scripts-docs-trim/files/scripts/hip/rdna3-wmma-bm-pages-calculator.py.
template <int BN, int BN_VSUB, int GH_MAX, int ROW_BLOCK, int NQ_MAX, int PV_IMPL = BM_DOT4_PAGES_PV_IMPL_SCALAR>
static __global__ __launch_bounds__(128, 1)
void ggml_cuda_q8k_dot4_bm_dot4_pages_stage1_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ partial_o,   // [batch, nq, n_heads_k, n_pages, GH_MAX, D]
        float       * __restrict__ partial_m,   // [batch, nq, n_heads_k, n_pages, GH_MAX]
        float       * __restrict__ partial_l,   // [batch, nq, n_heads_k, n_pages, GH_MAX]
        float scale,
        int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
        int64_t nb30, int64_t nb31, int64_t nb33, int64_t ne33,
        int nq, int nk, int n_heads_q, int n_heads_k,
        int gqa_ratio, int batch, int q_offset,
        int k_head_stride_rows, int k_batch_stride_rows,
        int n_pages,
        small_verify_fa4_profile * __restrict__ profile) {

    static_assert(BN % 16 == 0, "BM/page BN must be a multiple of the WMMA key tile");
    static_assert(ROW_BLOCK == 16, "BM/page row block follows RDNA3 16x16 WMMA rows");
    static_assert(BN % BN_VSUB == 0, "BN must be multiple of BN_VSUB");
    static_assert(PV_IMPL == BM_DOT4_PAGES_PV_IMPL_SCALAR ||
            PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PVWMMA ||
            PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PINT8PV ||
            PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4 ||
            PV_IMPL == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4 ||
            PV_IMPL == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA,
            "unsupported BM/page PV implementation");

    constexpr bool PVWMMA = PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PVWMMA;
    constexpr bool PINT8PV = PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PINT8PV;
    constexpr bool PINT8PV_DOT4 = PV_IMPL == BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4;
    constexpr bool INTFLASH_VFRAG_DOT4 = PV_IMPL == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4;
    constexpr bool INTFLASH_VFRAG_WMMA = PV_IMPL == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA;
    constexpr bool INTFLASH_VFRAG = INTFLASH_VFRAG_DOT4 || INTFLASH_VFRAG_WMMA;
    constexpr bool WMMA_PV = PVWMMA || INTFLASH_VFRAG_WMMA;

    if (nq < 2 || nq > NQ_MAX || gqa_ratio <= 0 || gqa_ratio > GH_MAX) return;

    const int page = blockIdx.x;
    const int y    = blockIdx.y;
    const int b    = blockIdx.z;
    const int row_blocks_per_hk = (nq * gqa_ratio + ROW_BLOCK - 1) / ROW_BLOCK;
    if (row_blocks_per_hk <= 0) return;
    const int hk = y / row_blocks_per_hk;
    const int row_block_id = y - hk * row_blocks_per_hk;
    if (b >= batch || hk >= n_heads_k || page >= n_pages || row_block_id >= row_blocks_per_hk) return;

    const int tid = threadIdx.x;
    const int hq0 = hk * gqa_ratio;
    const int gh  = min(gqa_ratio, n_heads_q - hq0);
    if (gh <= 0) return;

    const int k0 = page * BN;
    const int tile_n = min(BN, nk - k0);
    if (tile_n <= 0) return;

    extern __shared__ __align__(16) unsigned char smem[];
    int   * q_i32  = reinterpret_cast<int   *>(smem);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + ROW_BLOCK * DECODE_I32_PER_ROW);
    int   * k_i32  = reinterpret_cast<int   *>(q_scl + ROW_BLOCK * DECODE_N_BLOCKS);
    float * k_scl  = reinterpret_cast<float *>(k_i32 + BN * DECODE_I32_PER_ROW);
    float * logits = k_scl + BN * DECODE_N_BLOCKS;
    float * probs  = logits + ROW_BLOCK * BN;
    float * sm     = probs  + ROW_BLOCK * BN;

    // Load the row block of Q payload/scales.  Rows outside nq*gh are padded
    // with zeros and never written.
    for (int i = tid; i < ROW_BLOCK * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int row = i / DECODE_I32_PER_ROW;
        const int pos = i - row * DECODE_I32_PER_ROW;
        const int logical_row = row_block_id * ROW_BLOCK + row;
        const int q_idx = logical_row / gh;
        const int g     = logical_row - q_idx * gh;
        int val = 0;
        if (q_idx < nq && g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q_idx));
            val = q_payload[q_base * DECODE_I32_PER_ROW + pos];
        }
        q_i32[i] = val;
    }
    for (int i = tid; i < ROW_BLOCK * DECODE_N_BLOCKS; i += blockDim.x) {
        const int row = i / DECODE_N_BLOCKS;
        const int qb  = i - row * DECODE_N_BLOCKS;
        const int logical_row = row_block_id * ROW_BLOCK + row;
        const int q_idx = logical_row / gh;
        const int g     = logical_row - q_idx * gh;
        float val = 0.0f;
        if (q_idx < nq && g < gh) {
            const int hq = hq0 + g;
            const size_t q_base = ((size_t(b) * size_t(n_heads_q) + size_t(hq)) * size_t(nq) + size_t(q_idx));
            val = q_scales[q_base * DECODE_N_BLOCKS + qb];
        }
        q_scl[i] = val;
    }

    for (int i = tid; i < ROW_BLOCK * 4; i += blockDim.x) {
        const int off = i & 3;
        sm[i] = (off == 0) ? -1e38f : 0.0f;
    }
    __syncthreads();

    float out[ROW_BLOCK][2];
    if constexpr (!WMMA_PV) {
#pragma unroll
        for (int row = 0; row < ROW_BLOCK; ++row) {
            out[row][0] = 0.0f;
            out[row][1] = 0.0f;
        }
    }

    float out_pv[4][8];
    if constexpr (WMMA_PV) {
#pragma unroll
        for (int pass = 0; pass < 4; ++pass)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                out_pv[pass][i] = 0.0f;
    }

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;
    unsigned long long profile_t0 = 0;

    // QK: one 16-row block by one BM-wide KV page.
    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    for (int i = tid; i < BN * DECODE_I32_PER_ROW; i += blockDim.x) {
        const int kk  = i / DECODE_I32_PER_ROW;
        const int pos = i - kk * DECODE_I32_PER_ROW;
        const int k   = k0 + kk;
        int val = 0;
        if (kk < tile_n && k < nk) {
            const size_t kb = k_head_base + k;
            val = k_payload[kb * DECODE_I32_PER_ROW + pos];
        }
        k_i32[i] = val;
    }
    for (int i = tid; i < BN * DECODE_N_BLOCKS; i += blockDim.x) {
        const int kk = i / DECODE_N_BLOCKS;
        const int qb = i - kk * DECODE_N_BLOCKS;
        const int k  = k0 + kk;
        float val = 0.0f;
        if (kk < tile_n && k < nk) {
            const size_t kb = k_head_base + k;
            val = __half2float(k_scales[kb * DECODE_N_BLOCKS + qb]);
        }
        k_scl[i] = val;
    }
    __syncthreads();

    for (int idx = tid; idx < ROW_BLOCK * BN; idx += blockDim.x) {
        const int row = idx / BN;
        const int kk  = idx - row * BN;
        const int k   = k0 + kk;
        const int logical_row = row_block_id * ROW_BLOCK + row;
        const int q_idx = logical_row / gh;
        const int g     = logical_row - q_idx * gh;
        const bool have = q_idx < nq && g < gh && kk < tile_n && k < nk;

        float sum = 0.0f;
        if (have) {
            const int * qp = q_i32 + row * DECODE_I32_PER_ROW;
            const float * qs = q_scl + row * DECODE_N_BLOCKS;
#pragma unroll
            for (int qb = 0; qb < DECODE_N_BLOCKS; ++qb) {
                int acc = 0;
#pragma unroll
                for (int word = 0; word < QK8_0 / 4; ++word) {
                    const int pos = qb * (QK8_0 / 4) + word;
                    acc = ggml_cuda_q8k_dot4_i8_i8(qp[pos], k_i32[kk * DECODE_I32_PER_ROW + pos], acc);
                }
                sum += float(acc) * qs[qb] * k_scl[kk * DECODE_N_BLOCKS + qb];
            }
        }

        float s = -1e38f;
        if (have) {
            const int q_pos = q_offset + q_idx;
            const bool valid = mask ? true : (k <= q_pos);
            if (valid) {
                const float raw = sum * scale;
                s = mask ? raw + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_idx, k, b) : raw;
            }
        }
        logits[row * BN + kk] = s;
    }
    __syncthreads();
    SMALL_VERIFY_PROFILE_PHASE_END(qk_cycles);

    // Page-local online softmax state.  This stores unnormalised probabilities
    // so the existing max/sum-exp reducer can merge pages exactly like split-K.
    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    if (tid < ROW_BLOCK) {
        const int row = tid;
        const int logical_row = row_block_id * ROW_BLOCK + row;
        const int q_idx = logical_row / gh;
        const int g     = logical_row - q_idx * gh;
        if (q_idx < nq && g < gh) {
            float tile_m = -1e38f;
            const int base = row * BN;
            for (int kk = 0; kk < tile_n; ++kk) tile_m = fmaxf(tile_m, logits[base + kk]);

            const float m_prev = sm[row * 4 + 0];
            const float l_prev = sm[row * 4 + 1];
            const float m_new = fmaxf(m_prev, tile_m);
            const float old_scale = (l_prev > 0.0f && m_new > -1e30f) ? expf(m_prev - m_new) : 0.0f;
            constexpr float INTFLASH_SOFTMAX_INV_E_SCALE = 1.0f / 127.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < tile_n; ++kk) {
                const float logit = logits[base + kk];
                const float p = (m_new > -1e30f && logit > -1e30f) ? expf(logit - m_new) : 0.0f;
                if constexpr (INTFLASH_VFRAG) {
                    const int e_i8 = ggml_cuda_q8k_dot4_quantize_i8_positive_unit(p);
                    const float e_q = float(e_i8) * INTFLASH_SOFTMAX_INV_E_SCALE;
                    probs[base + kk] = e_q;
                    tile_l += e_q;
                } else {
                    probs[base + kk] = p;
                    tile_l += p;
                }
            }
            sm[row * 4 + 0] = m_new;
            sm[row * 4 + 1] = l_prev * old_scale + tile_l;
            sm[row * 4 + 2] = old_scale;
        }
    }
    __syncthreads();
    SMALL_VERIFY_PROFILE_PHASE_END(softmax_cycles);

    // PV phase.  The default path is the scalar q4_0 correctness baseline.
    // The PVWMMA path keeps the same 16-row row-block contract and uses four
    // wave32 waves to cover D=256 as 4 passes × 4 waves × 16 columns.
    // The explicit pint8pv routes quantize signed weighted-P = P * q4_0.delta
    // per (row, 16-key tile, q4 block) to int8. The scalar variant keeps a
    // per-element integer MAC baseline; the dot4 variant packs four weighted-P
    // and four centered q4_0 values into int8x4 words and uses the ROCm DOT4
    // helper for the PV core before one float rescale at tile exit.
    // The intflash_vfrag_dot4 route instead quantizes the unnormalised
    // exponent tile E directly to positive int8 and forms a transient per-
    // subtile, per-output-column int8 V fragment before DOT4, rescaling once
    // by v_step / 127 without folding V scale into E.
    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    if constexpr (PVWMMA) {
        const int wave    = tid >> 5;
        const int lane    = tid & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh) out_pv[pass][i] *= sm[row * 4 + 2];
            }
        }
        for (int sub = 0; sub < BN; sub += 16) {
#pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int d_tile = (pass * 4 + wave) * 16;
                const int b_col = pbwmma_f16_b_col_from_lane_lo(lane_lo);
                pbwmma_v16fp16 b_frag;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kk = sub + i;
                    const int k  = k0 + kk;
                    b_frag[i] = (_Float16)((kk < tile_n && k < nk && d_tile + b_col < DECODE_D) ?
                        ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d_tile + b_col) : 0.0f);
                }

                pbwmma_v16fp16 a_frag;
                const int a_row = pbwmma_f16_a_row_from_lane_lo(lane_lo);
                const int a_logical_row = row_block_id * ROW_BLOCK + a_row;
                const int a_q_idx = a_logical_row / gh;
                const int a_g     = a_logical_row - a_q_idx * gh;
                const bool a_valid = a_q_idx < nq && a_g < gh;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kk = sub + i;
                    a_frag[i] = (_Float16)((a_valid && kk < tile_n) ? probs[a_row * BN + kk] : 0.0f);
                }
                pbwmma_v8fp32 acc = {0,0,0,0,0,0,0,0};
                acc = pbwmma_mma(a_frag, b_frag, acc);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
                    const int logical_row = row_block_id * ROW_BLOCK + row;
                    const int q_idx = logical_row / gh;
                    const int g     = logical_row - q_idx * gh;
                    if (q_idx < nq && g < gh) out_pv[pass][i] += acc[i];
                }
            }
        }
    } else if constexpr (INTFLASH_VFRAG_WMMA) {
        const int wave    = tid >> 5;
        const int lane    = tid & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
        constexpr float INTFLASH_INV_E_SCALE = 1.0f / 127.0f;
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = pbwmma_i8_d_row_from_acc(i, lane_hi);
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh) out_pv[pass][i] *= sm[row * 4 + 2];
            }
        }
        for (int sub = 0; sub < BN; sub += 16) {
#pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int d_tile = (pass * 4 + wave) * 16;
                const int b_col = pbwmma_i8_b_col_from_lane_lo(lane_lo);
                const int d_col = d_tile + b_col;

                float max_abs = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int kk = sub + i;
                    const int k  = k0 + kk;
                    if (kk < tile_n && k < nk && d_col < DECODE_D) {
                        const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d_col);
                        max_abs = fmaxf(max_abs, fabsf(v));
                    }
                }
                const float v_step  = max_abs > 0.0f ? (max_abs / 127.0f) : 0.0f;
                const float inv_v   = max_abs > 0.0f ? (127.0f / max_abs) : 0.0f;
                const float pv_scale = v_step * INTFLASH_INV_E_SCALE;

                pbwmma_v4i32 a_frag;
                pbwmma_v4i32 b_frag;
                const int a_row = pbwmma_i8_a_row_from_lane_lo(lane_lo);
                const int a_logical_row = row_block_id * ROW_BLOCK + a_row;
                const int a_q_idx = a_logical_row / gh;
                const int a_g     = a_logical_row - a_q_idx * gh;
                const bool a_valid = a_q_idx < nq && a_g < gh;
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    uint32_t aw = 0;
                    uint32_t bw = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int kk = sub + wi * 4 + j;
                        const int k  = k0 + kk;
                        int e_i8 = 0;
                        int v_i8 = 0;
                        if (a_valid && kk < tile_n) {
                            e_i8 = ggml_cuda_q8k_dot4_quantize_i8_positive_unit(probs[a_row * BN + kk]);
                        }
                        if (kk < tile_n && k < nk && d_col < DECODE_D) {
                            const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, d_col);
                            v_i8 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(v, inv_v);
                        }
                        aw |= uint32_t(uint8_t(int8_t(e_i8))) << (8 * j);
                        bw |= uint32_t(uint8_t(int8_t(v_i8))) << (8 * j);
                    }
                    a_frag[wi] = int(aw);
                    b_frag[wi] = int(bw);
                }
                pbwmma_v8i32 acc = {0,0,0,0,0,0,0,0};
                acc = pbwmma_mma_i8(a_frag, b_frag, acc);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = pbwmma_i8_d_row_from_acc(i, lane_hi);
                    const int logical_row = row_block_id * ROW_BLOCK + row;
                    const int q_idx = logical_row / gh;
                    const int g     = logical_row - q_idx * gh;
                    if (q_idx < nq && g < gh) out_pv[pass][i] += pv_scale * float(acc[i]);
                }
            }
        }
    } else if constexpr (PINT8PV) {
        if (tid < 128) {
            const int q4_iq = tid & 15;
            const int q4_shift = (tid & 31) >= 16 ? 4 : 0;
            const int q4_blk0 = tid / QK4_0;
            const int q4_blk1 = (tid + 128) / QK4_0;
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh) {
                    out[row][0] *= sm[row * 4 + 2];
                    out[row][1] *= sm[row * 4 + 2];
                }
            }
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (!(q_idx < nq && g < gh)) continue;
                const int row_base = row * BN;
                for (int sub = 0; sub < tile_n; sub += 16) {
                    const int end = (sub + 16 < tile_n) ? sub + 16 : tile_n;
                    float max_abs0 = 0.0f;
                    float max_abs1 = 0.0f;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const char * v_row = v_head + int64_t(k) * nb21;
                        const block_q4_0 * v_blk0 = (const block_q4_0 *) (v_row + int64_t(q4_blk0) * nb20);
                        const block_q4_0 * v_blk1 = (const block_q4_0 *) (v_row + int64_t(q4_blk1) * nb20);
                        const float p = probs[row_base + kk];
                        max_abs0 = fmaxf(max_abs0, fabsf(p * __half2float(v_blk0->d)));
                        max_abs1 = fmaxf(max_abs1, fabsf(p * __half2float(v_blk1->d)));
                    }
                    const float step0 = max_abs0 > 0.0f ? (max_abs0 / 127.0f) : 0.0f;
                    const float step1 = max_abs1 > 0.0f ? (max_abs1 / 127.0f) : 0.0f;
                    const float inv_step0 = max_abs0 > 0.0f ? (127.0f / max_abs0) : 0.0f;
                    const float inv_step1 = max_abs1 > 0.0f ? (127.0f / max_abs1) : 0.0f;
                    int acc0 = 0;
                    int acc1 = 0;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const char * v_row = v_head + int64_t(k) * nb21;
                        const block_q4_0 * v_blk0 = (const block_q4_0 *) (v_row + int64_t(q4_blk0) * nb20);
                        const block_q4_0 * v_blk1 = (const block_q4_0 *) (v_row + int64_t(q4_blk1) * nb20);
                        const float p = probs[row_base + kk];
                        const int p_i8_0 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(p * __half2float(v_blk0->d), inv_step0);
                        const int p_i8_1 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(p * __half2float(v_blk1->d), inv_step1);
                        const int v_i4_0 = int((v_blk0->qs[q4_iq] >> q4_shift) & 0x0f) - 8;
                        const int v_i4_1 = int((v_blk1->qs[q4_iq] >> q4_shift) & 0x0f) - 8;
                        acc0 += p_i8_0 * v_i4_0;
                        acc1 += p_i8_1 * v_i4_1;
                    }
                    out[row][0] += step0 * float(acc0);
                    out[row][1] += step1 * float(acc1);
                }
            }
        }
    } else if constexpr (PINT8PV_DOT4) {
        if (tid < 128) {
            const int q4_iq = tid & 15;
            const int q4_shift = (tid & 31) >= 16 ? 4 : 0;
            const int q4_blk0 = tid / QK4_0;
            const int q4_blk1 = (tid + 128) / QK4_0;
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh) {
                    out[row][0] *= sm[row * 4 + 2];
                    out[row][1] *= sm[row * 4 + 2];
                }
            }
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (!(q_idx < nq && g < gh)) continue;
                const int row_base = row * BN;
                for (int sub = 0; sub < tile_n; sub += 16) {
                    const int end = (sub + 16 < tile_n) ? sub + 16 : tile_n;
                    float max_abs0 = 0.0f;
                    float max_abs1 = 0.0f;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const char * v_row = v_head + int64_t(k) * nb21;
                        const block_q4_0 * v_blk0 = (const block_q4_0 *) (v_row + int64_t(q4_blk0) * nb20);
                        const block_q4_0 * v_blk1 = (const block_q4_0 *) (v_row + int64_t(q4_blk1) * nb20);
                        const float p = probs[row_base + kk];
                        max_abs0 = fmaxf(max_abs0, fabsf(p * __half2float(v_blk0->d)));
                        max_abs1 = fmaxf(max_abs1, fabsf(p * __half2float(v_blk1->d)));
                    }
                    const float step0 = max_abs0 > 0.0f ? (max_abs0 / 127.0f) : 0.0f;
                    const float step1 = max_abs1 > 0.0f ? (max_abs1 / 127.0f) : 0.0f;
                    const float inv_step0 = max_abs0 > 0.0f ? (127.0f / max_abs0) : 0.0f;
                    const float inv_step1 = max_abs1 > 0.0f ? (127.0f / max_abs1) : 0.0f;
                    int acc0 = 0;
                    int acc1 = 0;
                    for (int kk = sub; kk < end; kk += 4) {
                        int p_i8_00 = 0, p_i8_01 = 0, p_i8_02 = 0, p_i8_03 = 0;
                        int p_i8_10 = 0, p_i8_11 = 0, p_i8_12 = 0, p_i8_13 = 0;
                        int v_i8_00 = 0, v_i8_01 = 0, v_i8_02 = 0, v_i8_03 = 0;
                        int v_i8_10 = 0, v_i8_11 = 0, v_i8_12 = 0, v_i8_13 = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int kk_i = kk + i;
                            if (kk_i >= end) {
                                continue;
                            }
                            const int k = k0 + kk_i;
                            const char * v_row = v_head + int64_t(k) * nb21;
                            const block_q4_0 * v_blk0 = (const block_q4_0 *) (v_row + int64_t(q4_blk0) * nb20);
                            const block_q4_0 * v_blk1 = (const block_q4_0 *) (v_row + int64_t(q4_blk1) * nb20);
                            const float p = probs[row_base + kk_i];
                            const int p_i8_0 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(p * __half2float(v_blk0->d), inv_step0);
                            const int p_i8_1 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(p * __half2float(v_blk1->d), inv_step1);
                            const int v_i8_0 = int((v_blk0->qs[q4_iq] >> q4_shift) & 0x0f) - 8;
                            const int v_i8_1 = int((v_blk1->qs[q4_iq] >> q4_shift) & 0x0f) - 8;
                            if (i == 0) {
                                p_i8_00 = p_i8_0; p_i8_10 = p_i8_1;
                                v_i8_00 = v_i8_0; v_i8_10 = v_i8_1;
                            } else if (i == 1) {
                                p_i8_01 = p_i8_0; p_i8_11 = p_i8_1;
                                v_i8_01 = v_i8_0; v_i8_11 = v_i8_1;
                            } else if (i == 2) {
                                p_i8_02 = p_i8_0; p_i8_12 = p_i8_1;
                                v_i8_02 = v_i8_0; v_i8_12 = v_i8_1;
                            } else {
                                p_i8_03 = p_i8_0; p_i8_13 = p_i8_1;
                                v_i8_03 = v_i8_0; v_i8_13 = v_i8_1;
                            }
                        }
                        const int p_pack0 = ggml_cuda_q8k_dot4_pack_i8x4(p_i8_00, p_i8_01, p_i8_02, p_i8_03);
                        const int p_pack1 = ggml_cuda_q8k_dot4_pack_i8x4(p_i8_10, p_i8_11, p_i8_12, p_i8_13);
                        const int v_pack0 = ggml_cuda_q8k_dot4_pack_i8x4(v_i8_00, v_i8_01, v_i8_02, v_i8_03);
                        const int v_pack1 = ggml_cuda_q8k_dot4_pack_i8x4(v_i8_10, v_i8_11, v_i8_12, v_i8_13);
                        acc0 = ggml_cuda_q8k_dot4_i8_i8(p_pack0, v_pack0, acc0);
                        acc1 = ggml_cuda_q8k_dot4_i8_i8(p_pack1, v_pack1, acc1);
                    }
                    out[row][0] += step0 * float(acc0);
                    out[row][1] += step1 * float(acc1);
                }
            }
        }
    } else if constexpr (INTFLASH_VFRAG_DOT4) {
        if (tid < 128) {
            const int d0 = tid;
            const int d1 = tid + 128;
            constexpr float INTFLASH_INV_E_SCALE = 1.0f / 127.0f;
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh) {
                    out[row][0] *= sm[row * 4 + 2];
                    out[row][1] *= sm[row * 4 + 2];
                }
            }
#pragma unroll
            for (int row = 0; row < ROW_BLOCK; ++row) {
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (!(q_idx < nq && g < gh)) continue;
                const int row_base = row * BN;
                for (int sub = 0; sub < tile_n; sub += 16) {
                    const int end = (sub + 16 < tile_n) ? sub + 16 : tile_n;
                    int e_packs[4] = {0, 0, 0, 0};
                    for (int kk = sub; kk < end; kk += 4) {
                        int e_i8_0 = 0, e_i8_1 = 0, e_i8_2 = 0, e_i8_3 = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int kk_i = kk + i;
                            if (kk_i >= end) {
                                continue;
                            }
                            const int e_i8 = ggml_cuda_q8k_dot4_quantize_i8_positive_unit(probs[row_base + kk_i]);
                            if (i == 0) {
                                e_i8_0 = e_i8;
                            } else if (i == 1) {
                                e_i8_1 = e_i8;
                            } else if (i == 2) {
                                e_i8_2 = e_i8;
                            } else {
                                e_i8_3 = e_i8;
                            }
                        }
                        e_packs[(kk - sub) >> 2] = ggml_cuda_q8k_dot4_pack_i8x4(e_i8_0, e_i8_1, e_i8_2, e_i8_3);
                    }

                    float max_abs0 = 0.0f;
                    float max_abs1 = 0.0f;
                    for (int kk = sub; kk < end; ++kk) {
                        const int k = k0 + kk;
                        const char * v_row = v_head + int64_t(k) * nb21;
                        const float v0 = ggml_cuda_q8k_dot4_dequant_q4_0(v_row, nb20, d0);
                        const float v1 = ggml_cuda_q8k_dot4_dequant_q4_0(v_row, nb20, d1);
                        max_abs0 = fmaxf(max_abs0, fabsf(v0));
                        max_abs1 = fmaxf(max_abs1, fabsf(v1));
                    }
                    const float v_step0 = max_abs0 > 0.0f ? (max_abs0 * INTFLASH_INV_E_SCALE) : 0.0f;
                    const float v_step1 = max_abs1 > 0.0f ? (max_abs1 * INTFLASH_INV_E_SCALE) : 0.0f;
                    const float pv_scale0 = v_step0 * INTFLASH_INV_E_SCALE;
                    const float pv_scale1 = v_step1 * INTFLASH_INV_E_SCALE;
                    const float inv_step0 = max_abs0 > 0.0f ? (127.0f / max_abs0) : 0.0f;
                    const float inv_step1 = max_abs1 > 0.0f ? (127.0f / max_abs1) : 0.0f;
                    int acc0 = 0;
                    int acc1 = 0;
                    for (int kk = sub; kk < end; kk += 4) {
                        int v_i8_00 = 0, v_i8_01 = 0, v_i8_02 = 0, v_i8_03 = 0;
                        int v_i8_10 = 0, v_i8_11 = 0, v_i8_12 = 0, v_i8_13 = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int kk_i = kk + i;
                            if (kk_i >= end) {
                                continue;
                            }
                            const int k = k0 + kk_i;
                            const char * v_row = v_head + int64_t(k) * nb21;
                            const float v0 = ggml_cuda_q8k_dot4_dequant_q4_0(v_row, nb20, d0);
                            const float v1 = ggml_cuda_q8k_dot4_dequant_q4_0(v_row, nb20, d1);
                            const int v_i8_0 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(v0, inv_step0);
                            const int v_i8_1 = ggml_cuda_q8k_dot4_quantize_i8_symmetric(v1, inv_step1);
                            if (i == 0) {
                                v_i8_00 = v_i8_0; v_i8_10 = v_i8_1;
                            } else if (i == 1) {
                                v_i8_01 = v_i8_0; v_i8_11 = v_i8_1;
                            } else if (i == 2) {
                                v_i8_02 = v_i8_0; v_i8_12 = v_i8_1;
                            } else {
                                v_i8_03 = v_i8_0; v_i8_13 = v_i8_1;
                            }
                        }
                        const int e_pack = e_packs[(kk - sub) >> 2];
                        const int v_pack0 = ggml_cuda_q8k_dot4_pack_i8x4(v_i8_00, v_i8_01, v_i8_02, v_i8_03);
                        const int v_pack1 = ggml_cuda_q8k_dot4_pack_i8x4(v_i8_10, v_i8_11, v_i8_12, v_i8_13);
                        acc0 = ggml_cuda_q8k_dot4_i8_i8(e_pack, v_pack0, acc0);
                        acc1 = ggml_cuda_q8k_dot4_i8_i8(e_pack, v_pack1, acc1);
                    }
                    out[row][0] += pv_scale0 * float(acc0);
                    out[row][1] += pv_scale1 * float(acc1);
                }
            }
        }
    } else if (tid < 128) {
#pragma unroll
        for (int row = 0; row < ROW_BLOCK; ++row) {
            const int logical_row = row_block_id * ROW_BLOCK + row;
            const int q_idx = logical_row / gh;
            const int g     = logical_row - q_idx * gh;
            if (q_idx < nq && g < gh) {
                out[row][0] *= sm[row * 4 + 2];
                out[row][1] *= sm[row * 4 + 2];
            }
        }
        for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
            const int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
            for (int kk = sub; kk < end; ++kk) {
                const int k = k0 + kk;
                const float v0 = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                const float v1 = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid + 128);
#pragma unroll
                for (int row = 0; row < ROW_BLOCK; ++row) {
                    const int logical_row = row_block_id * ROW_BLOCK + row;
                    const int q_idx = logical_row / gh;
                    const int g     = logical_row - q_idx * gh;
                    if (q_idx < nq && g < gh) {
                        const float p = probs[row * BN + kk];
                        out[row][0] += p * v0;
                        out[row][1] += p * v1;
                    }
                }
            }
        }
    }
    __syncthreads();
    SMALL_VERIFY_PROFILE_PHASE_END(pv_cycles);

    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    if constexpr (WMMA_PV) {
        const int wave    = tid >> 5;
        const int lane    = tid & 31;
        const int lane_lo = lane & 15;
        const int lane_hi = lane >> 4;
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            const int d_col = (pass * 4 + wave) * 16 + lane_lo;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = pbwmma_f16_d_row_from_acc(i, lane_hi);
                const int logical_row = row_block_id * ROW_BLOCK + row;
                const int q_idx = logical_row / gh;
                const int g     = logical_row - q_idx * gh;
                if (q_idx < nq && g < gh && d_col < DECODE_D) {
                    const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_pages) + size_t(page);
                    partial_o[(partial_base * GH_MAX + g) * DECODE_D + d_col] = out_pv[pass][i];
                }
            }
        }
    } else if (tid < 128) {
#pragma unroll
        for (int row = 0; row < ROW_BLOCK; ++row) {
            const int logical_row = row_block_id * ROW_BLOCK + row;
            const int q_idx = logical_row / gh;
            const int g     = logical_row - q_idx * gh;
            if (q_idx < nq && g < gh) {
                const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_pages) + size_t(page);
                partial_o[(partial_base * GH_MAX + g) * DECODE_D + tid]       = out[row][0];
                partial_o[(partial_base * GH_MAX + g) * DECODE_D + tid + 128] = out[row][1];
            }
        }
    }
    if (tid < ROW_BLOCK) {
        const int row = tid;
        const int logical_row = row_block_id * ROW_BLOCK + row;
        const int q_idx = logical_row / gh;
        const int g     = logical_row - q_idx * gh;
        if (q_idx < nq && g < gh) {
            const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q_idx)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_pages) + size_t(page);
            partial_m[partial_base * GH_MAX + g] = sm[row * 4 + 0];
            partial_l[partial_base * GH_MAX + g] = sm[row * 4 + 1];
        }
    }
    SMALL_VERIFY_PROFILE_PHASE_END(write_cycles);
}

// ── Grouped-GQA Small-verify split-K Stage 2: reduce partials per (q, hq, hk) ─
// Grid: (nq * n_heads_q, batch)  — one CTA per (q, query head)
// Each CTA merges n_splits partials for its (q, hq), mapping hq→hk via gqa_ratio.
template <int GH_MAX>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel(
        const float * __restrict__ partial_o,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float       * __restrict__ dst,
        int n_splits, int nq, int n_heads_q, int n_heads_k,
        int gqa_ratio, int batch,
        small_verify_fa4_profile * __restrict__ profile) {

    const int qh = blockIdx.x;
    const int b  = blockIdx.y;
    const int q  = qh / n_heads_q;
    const int hq = qh - q * n_heads_q;
    if (b >= batch || q >= nq || hq >= n_heads_q) return;
    const int hk = hq / gqa_ratio;
    const int g   = hq - hk * gqa_ratio;  // index within GQA group
    const int tid = threadIdx.x;

    float m = -1e38f;
    float l = 0.0f;
    float o = 0.0f;
    unsigned long long profile_t0 = 0;

    SMALL_VERIFY_PROFILE_PHASE_BEGIN();
    for (int s = 0; s < n_splits; ++s) {
        const size_t partial_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_k) + size_t(hk)) * size_t(n_splits) + size_t(s);
        const float mi = partial_m[partial_base * GH_MAX + g];
        const float li = partial_l[partial_base * GH_MAX + g];
        if (li == 0.0f) continue;
        const float m_new = fmaxf(m, mi);
        const float a = (l > 0.0f) ? expf(m - m_new) : 0.0f;
        const float bscale = expf(mi - m_new);

        if (tid < DECODE_D) {
            const float oi = partial_o[(partial_base * GH_MAX + g) * DECODE_D + tid];
            o = o * a + oi * bscale;
        }
        l = l * a + li * bscale;
        m = m_new;
    }

    if (tid < DECODE_D) {
        const size_t dst_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * DECODE_D;
        dst[dst_base + tid] = l > 0.0f ? o / l : 0.0f;
    }
    SMALL_VERIFY_PROFILE_PHASE_END(reduce_cycles);
}

// ── Split-K Stage 1: one CTA per K split, writes partial o/m/l ───────────
template <int BN, int BN_VSUB, bool V4_144 = false>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ partial_o,  // [batch, heads_q, n_splits, D]
        float       * __restrict__ partial_m,  // [batch, heads_q, n_splits]
        float       * __restrict__ partial_l,  // [batch, heads_q, n_splits]
        float scale,
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
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows,
        int split_size,
        int n_splits) {

    const int split = blockIdx.x;
    const int qh    = blockIdx.y;
    const int b     = blockIdx.z;
    const int q     = qh / n_heads_q;
    const int hq    = qh - q * n_heads_q;
    if (b >= batch || q >= nq || hq >= n_heads_q) return;
    const int hk = hq / gqa_ratio;
    const int tid = threadIdx.x;

    const int k_begin = split * split_size;
    const int k_end   = min(nk, k_begin + split_size);

    const size_t partial_base = (((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * size_t(n_splits) + split);

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);
    float * probs  = logits + BN;
    float * sm     = probs + BN;        // 3: row_m, row_l, old_scale
    int   * q_i32  = reinterpret_cast<int  *>(sm + 4);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + DECODE_I32_PER_ROW);

    if (tid == 0) { sm[0] = -1e38f; sm[1] = 0.0f; }
    __syncthreads();

    float out = 0.0f;

    // Load Q once
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq) + size_t(q));
    for (int i = tid; i < DECODE_I32_PER_ROW; i += blockDim.x)
        q_i32[i] = q_payload[q_base * DECODE_I32_PER_ROW + i];
    for (int i = tid; i < DECODE_N_BLOCKS; i += blockDim.x)
        q_scl[i] = q_scales[q_base * DECODE_N_BLOCKS + i];
    __syncthreads();

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = (k0 + BN <= k_end) ? BN : (k_end - k0);

        // QK. For nq>1 speculative verify, each query row has its own causal
        // boundary. q_offset is the first query row's absolute position.
        if (tid < BN) {
            int kk = tid, k = k0 + kk;
            const int q_pos = q_offset + q;
            bool valid = (kk < tile_n && k < nk) && (mask || k <= q_pos);
            float s = -1e38f;
            if (valid) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(q_i32, q_scl,
                    k_payload + kb * DECODE_I32_PER_ROW, k_scales + kb * DECODE_N_BLOCKS) * scale;
                if (mask) {
                    s += ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q, k, b);
                }
            }
            logits[kk] = s;
        }
        __syncthreads();

        // Softmax
        if (tid == 0) {
            float tile_m = -1e38f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[kk]);
            float m_prev = sm[0], l_prev = sm[1];
            float m_new = fmaxf(m_prev, tile_m);
            float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) {
                float p = expf(logits[kk] - m_new); probs[kk] = p; tile_l += p;
            }
            sm[0] = m_new; sm[1] = l_prev * old_scale + tile_l; sm[2] = old_scale;
        }
        __syncthreads();

        // PV accumulation
        if (tid < DECODE_D) {
            out *= sm[2];
            for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                for (int kk = sub; kk < end; ++kk) {
                    int k = k0 + kk; float p = probs[kk];
                    float v;
                    if constexpr (V4_144) {
                        v = ggml_cuda_q8k_dot4_dequant_v4_k16d16_144(v_head, nb21, k, tid);
                    } else {
                        v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
                    }
                    out += p * v;
                }
            }
        }
        __syncthreads();
    }

    // Write partials — unnormalized! No divide-by-l.
    if (tid < DECODE_D) {
        partial_o[partial_base * DECODE_D + tid] = out;
    }
    if (tid == 0) {
        partial_m[partial_base] = sm[0];
        partial_l[partial_base] = sm[1];
    }
}

// ── Split-K Stage 2: reduce partial o/m/l via online-softmax merge ────────
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_splitk_reduce_kernel(
        const float * __restrict__ partial_o,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float       * __restrict__ dst,
        int n_splits,
        int nq,
        int n_heads_q,
        int batch) {

    const int qh = blockIdx.x;
    const int b  = blockIdx.y;
    const int q  = qh / n_heads_q;
    const int hq = qh - q * n_heads_q;
    const int tid = threadIdx.x;
    if (b >= batch || q >= nq || hq >= n_heads_q) return;

    float m = -1e38f;
    float l = 0.0f;
    float o = 0.0f;

    for (int s = 0; s < n_splits; ++s) {
        const size_t partial_base = (((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * size_t(n_splits) + s);

        const float mi = partial_m[partial_base];
        const float li = partial_l[partial_base];
        if (li == 0.0f) continue;

        const float m_new = fmaxf(m, mi);
        const float a = (l > 0.0f) ? expf(m - m_new) : 0.0f;
        const float bscale = expf(mi - m_new);

        if (tid < DECODE_D) {
            const float oi = partial_o[partial_base * DECODE_D + tid];
            o = o * a + oi * bscale;
        }
        l = l * a + li * bscale;
        m = m_new;
    }

    if (tid < DECODE_D) {
        const size_t dst_base = ((size_t(b) * size_t(nq) + size_t(q)) * size_t(n_heads_q) + size_t(hq)) * DECODE_D;
        dst[dst_base + tid] = l > 0.0f ? o / l : 0.0f;
    }
}

#define LAUNCH_DECODE_GQA(BN, BN_VSUB, GH_MAX) { \
    int sm = (GH_MAX * BN * 2 + GH_MAX * 4 + GH_MAX * DECODE_I32_PER_ROW + GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_k, batch); \
    ggml_cuda_q8k_dot4_decode_gqa_p16_kernel<BN, BN_VSUB, GH_MAX><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_SMALL_VERIFY(BN, BN_VSUB, GH_MAX) { \
    int sm = (GH_MAX * BN * 2 + GH_MAX * 4 + GH_MAX * DECODE_I32_PER_ROW + GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(nq, n_heads_k, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_p16_kernel<BN, BN_VSUB, GH_MAX><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, (float *) dst->data, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_SMALL_VERIFY_SPLITK(BN, BN_VSUB, GH_MAX) { \
    int sm = (GH_MAX * BN * 2 + GH_MAX * 4 + GH_MAX * DECODE_I32_PER_ROW + GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float); \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE", ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512)); \
    int n_splits = (nk + split_size - 1) / split_size; \
    dim3 g1(n_splits, n_heads_k, nq * batch); \
    blockfa_partial_o.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX * 256); \
    blockfa_partial_m.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    blockfa_partial_l.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_stage1_kernel<BN, BN_VSUB, GH_MAX><<<g1, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(nq * n_heads_q, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel<GH_MAX><<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, n_heads_k, gqa_ratio, batch, nullptr); \
    CUDA_CHECK(cudaGetLastError()); \
}

#define LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(BN, BN_VSUB, GH_MAX, NQ_MAX, SPLIT_DEFAULT) { \
    int sm = (NQ_MAX * GH_MAX * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (NQ_MAX * GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (BN * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + BN * (int)sizeof(float) \
           + (NQ_MAX * GH_MAX * BN * 2) * (int)sizeof(float) \
           + (NQ_MAX * GH_MAX * 4) * (int)sizeof(float); \
    /* Batched small-verify has different occupancy/partial-reduce tradeoffs than nq=1 draft decode. */ \
    /* Do not let the generic DECODE_SPLITK_SIZE knob silently override this route's tuned default. */ \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE", ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_SPLIT_SIZE", SPLIT_DEFAULT)); \
    int n_splits = (nk + split_size - 1) / split_size; \
    dim3 g1(n_splits, n_heads_k, batch); \
    blockfa_partial_o.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX * 256); \
    blockfa_partial_m.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    blockfa_partial_l.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    ggml_cuda_q8k_dot4_small_verify_batched_gqa_splitk_stage1_kernel<BN, BN_VSUB, GH_MAX, NQ_MAX><<<g1, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits, nullptr); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(nq * n_heads_q, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel<GH_MAX><<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, n_heads_k, gqa_ratio, batch, nullptr); \
    CUDA_CHECK(cudaGetLastError()); \
}

#define LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(BN, BN_VSUB, GH_MAX, NQ_MAX, SPLIT_DEFAULT, PVWMMA) { \
    int sm = (NQ_MAX * GH_MAX * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (NQ_MAX * GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (BN * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (BN * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (NQ_MAX * GH_MAX * BN * 2) * (int)sizeof(float) \
           + (NQ_MAX * GH_MAX * 4) * (int)sizeof(float); \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_FA4_SPLIT_SIZE", ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_FA2_SPLIT_SIZE", ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", SPLIT_DEFAULT))); \
    int n_splits = (nk + split_size - 1) / split_size; \
    int fa4_profile_every = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_FA4_PROFILE_EVERY", 1); \
    if (fa4_profile_every < 1) fa4_profile_every = 1; \
    static unsigned long long fa4_profile_call_counter = 0; \
    const unsigned long long fa4_profile_call = ++fa4_profile_call_counter; \
    bool fa4_profile_enabled = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_SMALL_VERIFY_FA4_PROFILE") && ((fa4_profile_call % (unsigned long long) fa4_profile_every) == 0ull); \
    if (fa4_profile_enabled) { \
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone; \
        const hipError_t capture_err = hipStreamIsCapturing(stream, &capture_status); \
        if (capture_err == hipSuccess && capture_status != hipStreamCaptureStatusNone) { \
            static bool fa4_profile_capture_warned = false; \
            if (!fa4_profile_capture_warned) { fa4_profile_capture_warned = true; fprintf(stderr, "SMALL_VERIFY_FA4 PROFILE skipped during graph capture\n"); } \
            fa4_profile_enabled = false; \
        } \
    } \
    small_verify_fa4_profile * fa4_profile_dev = nullptr; \
    hipEvent_t fa4_profile_start = nullptr; \
    hipEvent_t fa4_profile_stop  = nullptr; \
    if (fa4_profile_enabled) { \
        CUDA_CHECK(hipMalloc((void **) &fa4_profile_dev, sizeof(small_verify_fa4_profile))); \
        CUDA_CHECK(hipMemsetAsync(fa4_profile_dev, 0, sizeof(small_verify_fa4_profile), stream)); \
        CUDA_CHECK(hipEventCreate(&fa4_profile_start)); \
        CUDA_CHECK(hipEventCreate(&fa4_profile_stop)); \
        CUDA_CHECK(hipEventRecord(fa4_profile_start, stream)); \
    } \
    dim3 g1(n_splits, n_heads_k, batch); \
    blockfa_partial_o.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX * 256); \
    blockfa_partial_m.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    blockfa_partial_l.alloc((size_t) batch * nq * n_heads_k * n_splits * GH_MAX); \
    ggml_cuda_q8k_dot4_small_verify_batched_gqa_splitk_stage1_kernel<BN, BN_VSUB, GH_MAX, NQ_MAX, true, PVWMMA><<<g1, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits, fa4_profile_dev); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(nq * n_heads_q, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel<GH_MAX><<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, n_heads_k, gqa_ratio, batch, fa4_profile_dev); \
    CUDA_CHECK(cudaGetLastError()); \
    if (fa4_profile_enabled) { \
        CUDA_CHECK(hipEventRecord(fa4_profile_stop, stream)); \
        small_verify_fa4_profile fa4_profile_host = {}; \
        CUDA_CHECK(hipMemcpyAsync(&fa4_profile_host, fa4_profile_dev, sizeof(fa4_profile_host), hipMemcpyDeviceToHost, stream)); \
        CUDA_CHECK(hipStreamSynchronize(stream)); \
        float fa4_kernel_ms = 0.0f; \
        CUDA_CHECK(hipEventElapsedTime(&fa4_kernel_ms, fa4_profile_start, fa4_profile_stop)); \
        fprintf(stderr, "SMALL_VERIFY_FA4 PROFILE: impl=%s call=%llu every=%d nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d split_size=%d n_splits=%d kernel_ms=%.3f qk_cycles=%llu softmax_cycles=%llu pv_cycles=%llu write_cycles=%llu reduce_cycles=%llu\n", \
            PVWMMA ? "small_verify_fa4_pvwmma" : "small_verify_fa4", \
            (unsigned long long) fa4_profile_call, fa4_profile_every, nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio, split_size, n_splits, (double) fa4_kernel_ms, \
            (unsigned long long) fa4_profile_host.qk_cycles, \
            (unsigned long long) fa4_profile_host.softmax_cycles, \
            (unsigned long long) fa4_profile_host.pv_cycles, \
            (unsigned long long) fa4_profile_host.write_cycles, \
            (unsigned long long) fa4_profile_host.reduce_cycles); \
        CUDA_CHECK(hipEventDestroy(fa4_profile_start)); \
        CUDA_CHECK(hipEventDestroy(fa4_profile_stop)); \
        CUDA_CHECK(hipFree(fa4_profile_dev)); \
    } \
}

#define LAUNCH_DECODE_BM_DOT4_PAGES(BN, BN_VSUB, GH_MAX, ROW_BLOCK, NQ_MAX, PV_IMPL) { \
    int n_pages = (nk + BN - 1) / BN; \
    int row_blocks_per_hk = (nq * gqa_ratio + ROW_BLOCK - 1) / ROW_BLOCK; \
    int sm = (ROW_BLOCK * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (ROW_BLOCK * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (BN * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (BN * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (ROW_BLOCK * BN * 2) * (int)sizeof(float) \
           + (ROW_BLOCK * 4) * (int)sizeof(float); \
    int profile_every = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_BM_DOT4_PAGES_PROFILE_EVERY", 1); \
    if (profile_every < 1) profile_every = 1; \
    static unsigned long long profile_call_counter = 0; \
    const unsigned long long profile_call = ++profile_call_counter; \
    bool profile_enabled = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_BM_DOT4_PAGES_PROFILE") && ((profile_call % (unsigned long long) profile_every) == 0ull); \
    if (profile_enabled) { \
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone; \
        const hipError_t capture_err = hipStreamIsCapturing(stream, &capture_status); \
        if (capture_err == hipSuccess && capture_status != hipStreamCaptureStatusNone) { \
            static bool capture_warned = false; \
            if (!capture_warned) { capture_warned = true; fprintf(stderr, "BM_DOT4_PAGES PROFILE skipped during graph capture\n"); } \
            profile_enabled = false; \
        } \
    } \
    small_verify_fa4_profile * profile_dev = nullptr; \
    hipEvent_t profile_start = nullptr; \
    hipEvent_t profile_stop  = nullptr; \
    if (profile_enabled) { \
        CUDA_CHECK(hipMalloc((void **) &profile_dev, sizeof(small_verify_fa4_profile))); \
        CUDA_CHECK(hipMemsetAsync(profile_dev, 0, sizeof(small_verify_fa4_profile), stream)); \
        CUDA_CHECK(hipEventCreate(&profile_start)); \
        CUDA_CHECK(hipEventCreate(&profile_stop)); \
        CUDA_CHECK(hipEventRecord(profile_start, stream)); \
    } \
    dim3 g1(n_pages, n_heads_k * row_blocks_per_hk, batch); \
    blockfa_partial_o.alloc((size_t) batch * nq * n_heads_k * n_pages * GH_MAX * 256); \
    blockfa_partial_m.alloc((size_t) batch * nq * n_heads_k * n_pages * GH_MAX); \
    blockfa_partial_l.alloc((size_t) batch * nq * n_heads_k * n_pages * GH_MAX); \
    ggml_cuda_q8k_dot4_bm_dot4_pages_stage1_kernel<BN, BN_VSUB, GH_MAX, ROW_BLOCK, NQ_MAX, PV_IMPL><<<g1, 128, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, n_pages, profile_dev); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(nq * n_heads_q, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel<GH_MAX><<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_pages, nq, n_heads_q, n_heads_k, gqa_ratio, batch, profile_dev); \
    CUDA_CHECK(cudaGetLastError()); \
    if (profile_enabled) { \
        CUDA_CHECK(hipEventRecord(profile_stop, stream)); \
        small_verify_fa4_profile profile_host = {}; \
        CUDA_CHECK(hipMemcpyAsync(&profile_host, profile_dev, sizeof(profile_host), hipMemcpyDeviceToHost, stream)); \
        CUDA_CHECK(hipStreamSynchronize(stream)); \
        float kernel_ms = 0.0f; \
        CUDA_CHECK(hipEventElapsedTime(&kernel_ms, profile_start, profile_stop)); \
        fprintf(stderr, "BM_DOT4_PAGES PROFILE: impl=%s call=%llu every=%d nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d BM=%d row_block=%d row_blocks_per_hk=%d pages=%d ctas=%d kernel_ms=%.3f qk_cycles=%llu softmax_cycles=%llu pv_cycles=%llu write_cycles=%llu reduce_cycles=%llu\n", \
            (PV_IMPL) == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA ? "bm_dot4_pages_intflash_vfrag_wmma" : ((PV_IMPL) == BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4 ? "bm_dot4_pages_intflash_vfrag_dot4" : ((PV_IMPL) == BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4 ? "bm_dot4_pages_pint8pv_dot4" : ((PV_IMPL) == BM_DOT4_PAGES_PV_IMPL_PINT8PV ? "bm_dot4_pages_pint8pv" : ((PV_IMPL) == BM_DOT4_PAGES_PV_IMPL_PVWMMA ? "bm_dot4_pages_pvwmma" : "bm_dot4_pages")))), \
            (unsigned long long) profile_call, profile_every, nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio, BN, ROW_BLOCK, row_blocks_per_hk, n_pages, n_pages * n_heads_k * row_blocks_per_hk * batch, (double) kernel_ms, \
            (unsigned long long) profile_host.qk_cycles, \
            (unsigned long long) profile_host.softmax_cycles, \
            (unsigned long long) profile_host.pv_cycles, \
            (unsigned long long) profile_host.write_cycles, \
            (unsigned long long) profile_host.reduce_cycles); \
        CUDA_CHECK(hipEventDestroy(profile_start)); \
        CUDA_CHECK(hipEventDestroy(profile_stop)); \
        CUDA_CHECK(hipFree(profile_dev)); \
    } \
}

#define LAUNCH_DECODE_GQA_PVWMMA(BN, GH_MAX) { \
    int sm = (GH_MAX * BN * 2 + GH_MAX * 4 + GH_MAX * DECODE_I32_PER_ROW + GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_k, batch); \
    ggml_cuda_q8k_dot4_decode_gqa_pvwmma_p16_kernel<BN, GH_MAX, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_GQA_WMMA_FULL(BN, GH_MAX) { \
    int sm = (GH_MAX * BN * 2 + GH_MAX * 4 + GH_MAX * DECODE_I32_PER_ROW + GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_k, batch); \
    ggml_cuda_q8k_dot4_decode_gqa_pvwmma_p16_kernel<BN, GH_MAX, true><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_Q4PAIR(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, true><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, \
        decode_v_debug, decode_v_debug_hq, decode_v_debug_k, decode_v_debug_d0, decode_v_debug_count); \
}

#define LAUNCH_DECODE_WAVEQK(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + DECODE_I32_PER_ROW + DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_waveqk_p16_kernel<BN, BN_VSUB, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_WAVEQK_Q4PAIR(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + DECODE_I32_PER_ROW + DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_waveqk_p16_kernel<BN, BN_VSUB, true><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_DSPLIT(BN, BN_VSUB, D_CHUNK) { \
    int sm = (BN * 2 + 4 + DECODE_I32_PER_ROW + DECODE_N_BLOCKS) * (int)sizeof(float); \
    dim3 g(n_heads_q, DECODE_D / D_CHUNK, batch); \
    ggml_cuda_q8k_dot4_decode_dsplit_p16_kernel<BN, BN_VSUB, D_CHUNK><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_INLINE_Q4(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, true, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, \
        decode_v_debug, decode_v_debug_hq, decode_v_debug_k, decode_v_debug_d0, decode_v_debug_count); \
}

#define LAUNCH_DECODE_SPLITK(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + DECODE_I32_PER_ROW + DECODE_N_BLOCKS) * (int)sizeof(float); \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512); \
    int n_splits = (nk + split_size - 1) / split_size; \
    dim3 g1(n_splits, n_heads_q * nq, batch); \
    if (v4_144_decode_diag) { \
        ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel<BN, BN_VSUB, true><<<g1, 256, sm, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            mask ? (const char *) mask->data : nullptr, blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
            k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    } else { \
        ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel<BN, BN_VSUB><<<g1, 256, sm, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            mask ? (const char *) mask->data : nullptr, blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
            V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
            k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    } \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(n_heads_q * nq, batch); \
    ggml_cuda_q8k_dot4_decode_splitk_reduce_kernel<<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, batch); \
}

#define LAUNCH_DECODE_VSUB(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    if (v4_144_decode_diag) { \
        ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, false, true><<<g, 256, sm, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, \
            decode_v_debug, decode_v_debug_hq, decode_v_debug_k, decode_v_debug_d0, decode_v_debug_count); \
    } else { \
        ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, false><<<g, 256, sm, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, \
            decode_v_debug, decode_v_debug_hq, decode_v_debug_k, decode_v_debug_d0, decode_v_debug_count); \
    } \
}
