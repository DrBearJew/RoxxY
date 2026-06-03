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

template <int BN, int BN_VSUB, bool INLINE_Q4, bool Q4PAIR>
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
        int k_batch_stride_rows) {

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
                    } else {
                        v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
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
        dst[((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * 256 + tid] = out0 / sm[1];
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
template <int BN, int BN_VSUB, int GH_MAX, int NQ_MAX>
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
        int split_size, int n_splits) {

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
    //   logits[GH_MAX * BN]                            — one GH×BN tile (reused per query)
    //   probs[GH_MAX * BN]                             — one GH×BN tile (reused per query)
    //   sm[NQ_MAX * GH_MAX * 4]                        — softmax running state per (q,g)
    int   * q_i32  = reinterpret_cast<int   *>(smem);
    float * q_scl  = reinterpret_cast<float *>(q_i32 + NQ_MAX * GH_MAX * DECODE_I32_PER_ROW);
    float * logits  = q_scl + NQ_MAX * GH_MAX * DECODE_N_BLOCKS;
    float * probs   = logits + GH_MAX * BN;
    float * sm      = probs  + GH_MAX * BN;

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

    // Per-thread output accumulator: out[q_idx][g]
    float out[NQ_MAX][GH_MAX];
#pragma unroll
    for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx)
#pragma unroll
        for (int g = 0; g < GH_MAX; ++g)
            out[q_idx][g] = 0.0f;

    const size_t k_head_base = size_t(b)  * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    // ── Main K/V tile loop: K and V are read ONCE per tile, shared across all nq ──
    for (int k0 = k_begin; k0 < k_end; k0 += BN) {
        const int tile_n = (k0 + BN <= k_end) ? BN : (k_end - k0);

        // ── QK dot for all nq × gh heads — K tile loaded once from global memory ──
        // For each query, compute dot product of its Q against this K tile
#pragma unroll
        for (int q_idx = 0; q_idx < NQ_MAX; ++q_idx) {
            if (q_idx < nq) {
                const int q_pos = q_offset + q_idx;

                for (int i = tid; i < GH_MAX * BN; i += blockDim.x) {
                    const int g  = i / BN;
                    const int kk = i - g * BN;
                    const int k  = k0 + kk;
                    float s = -1e38f;
                    if (g < gh && kk < tile_n && k < nk && (mask || k <= q_pos)) {
                        // K tile access: k_payload[k_head_base + k] — shared across all nq!
                        const size_t kb = k_head_base + k;
                        const float raw_score = ggml_cuda_q8k_dot4_kq_dot_direct(
                            q_i32 + (q_idx * GH_MAX + g) * DECODE_I32_PER_ROW,
                            q_scl + (q_idx * GH_MAX + g) * DECODE_N_BLOCKS,
                            k_payload + kb * DECODE_I32_PER_ROW,
                            k_scales + kb * DECODE_N_BLOCKS) * scale;
                        // Apply per-query causal boundary
                        if (mask) {
                            s = raw_score + ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_idx, k, b);
                        } else if (k <= q_pos) {
                            s = raw_score;
                        }
                    }
                    logits[g * BN + kk] = s;
                }
                __syncthreads();

                // Softmax per gh head
                if (tid < GH_MAX) {
                    const int g = tid;
                    if (g < gh) {
                        float tile_m = -1e38f;
                        for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[g * BN + kk]);
                        float m_prev = sm[(q_idx * GH_MAX + g) * 4 + 0];
                        float l_prev = sm[(q_idx * GH_MAX + g) * 4 + 1];
                        float m_new  = fmaxf(m_prev, tile_m);
                        float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
                        float tile_l = 0.0f;
                        for (int kk = 0; kk < BN; ++kk) {
                            float p = expf(logits[g * BN + kk] - m_new);
                            probs[g * BN + kk] = p;
                            tile_l += p;
                        }
                        sm[(q_idx * GH_MAX + g) * 4 + 0] = m_new;
                        sm[(q_idx * GH_MAX + g) * 4 + 1] = l_prev * old_scale + tile_l;
                        sm[(q_idx * GH_MAX + g) * 4 + 2] = old_scale;
                    }
                }
                __syncthreads();

                // P×V accumulation — V loaded ONCE per tile, shared across all q
                if (tid < DECODE_D) {
#pragma unroll
                    for (int g = 0; g < GH_MAX; ++g) {
                        if (g < gh) out[q_idx][g] *= sm[(q_idx * GH_MAX + g) * 4 + 2];
                    }
                    for (int sub = 0; sub < tile_n; sub += BN_VSUB) {
                        int end = (sub + BN_VSUB < tile_n) ? sub + BN_VSUB : tile_n;
                        for (int kk = sub; kk < end; ++kk) {
                            // V row — shared across all nq queries!
                            const int k = k0 + kk;
                            const float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
#pragma unroll
                            for (int g = 0; g < GH_MAX; ++g) {
                                if (g < gh) out[q_idx][g] += probs[g * BN + kk] * v;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    // Write unnormalised partials per (q, g)
    if (tid < DECODE_D) {
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
        int gqa_ratio, int batch) {

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
}

// ── Split-K Stage 1: one CTA per K split, writes partial o/m/l ───────────
template <int BN, int BN_VSUB>
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
                    float v = ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
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
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512); \
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
        n_splits, nq, n_heads_q, n_heads_k, gqa_ratio, batch); \
    CUDA_CHECK(cudaGetLastError()); \
}

#define LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(BN, BN_VSUB, GH_MAX, NQ_MAX) { \
    int sm = (NQ_MAX * GH_MAX * DECODE_I32_PER_ROW) * (int)sizeof(int) \
           + (NQ_MAX * GH_MAX * DECODE_N_BLOCKS) * (int)sizeof(float) \
           + (GH_MAX * BN * 2) * (int)sizeof(float) \
           + (NQ_MAX * GH_MAX * 4) * (int)sizeof(float); \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512); \
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
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(nq * n_heads_q, batch); \
    ggml_cuda_q8k_dot4_small_verify_gqa_splitk_reduce_kernel<GH_MAX><<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, n_heads_k, gqa_ratio, batch); \
    CUDA_CHECK(cudaGetLastError()); \
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
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
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
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}

#define LAUNCH_DECODE_SPLITK(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + DECODE_I32_PER_ROW + DECODE_N_BLOCKS) * (int)sizeof(float); \
    int split_size = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE", 512); \
    int n_splits = (nk + split_size - 1) / split_size; \
    dim3 g1(n_splits, n_heads_q * nq, batch); \
    ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel<BN, BN_VSUB><<<g1, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        mask ? (const char *) mask->data : nullptr, blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1, \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(n_heads_q * nq, batch); \
    ggml_cuda_q8k_dot4_decode_splitk_reduce_kernel<<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, nq, n_heads_q, batch); \
}

#define LAUNCH_DECODE_VSUB(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}
