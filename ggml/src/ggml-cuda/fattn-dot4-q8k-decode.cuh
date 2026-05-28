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
//  MTP semantic routing:
//    MTP_VERIFY is a semantic FA/QK role. nq is only a kernel legality gate.
//    DOT4 is eligible for MTP_VERIFY only when nq > 2.
//    MTP_DRAFT never enables DOT4 preference — it falls through to existing policy.
//    MTP chunk size (LLAMA_MTP_PREFILL_CHUNK) does not control route selection;
//    the hint does. Chunk size only controls whether the chosen kernel is legal.
//
//  Env flags:
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN=64
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK=1          (force-enable)
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD=2048
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE=512
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR=1          (experimental)
//    GGML_CUDA_ROCM_Q8K_DOT4_DECODE_INLINE_Q4=1       (experimental)
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


// ── Split-K Stage 1: one CTA per K split, writes partial o/m/l ───────────
template <int BN, int BN_VSUB>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ partial_o,  // [batch, heads_q, n_splits, D]
        float       * __restrict__ partial_m,  // [batch, heads_q, n_splits]
        float       * __restrict__ partial_l,  // [batch, heads_q, n_splits]
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
        int split_size,
        int n_splits) {

    if (nq != 1) return;
    const int split = blockIdx.x;
    const int hq    = blockIdx.y;
    const int b     = blockIdx.z;
    if (b >= batch || hq >= n_heads_q) return;
    const int hk = hq / gqa_ratio;
    const int tid = threadIdx.x;

    const int k_begin = split * split_size;
    const int k_end   = min(nk, k_begin + split_size);

    const size_t partial_base = ((size_t(b) * n_heads_q + hq) * size_t(n_splits) + split);

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
    const size_t q_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
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

        // QK
        if (tid < BN) {
            int kk = tid, k = k0 + kk;
            bool valid = (kk < tile_n && k < nk) && (k - q_offset <= 0);
            float s = -1e38f;
            if (valid) {
                const size_t kb = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(q_i32, q_scl,
                    k_payload + kb * DECODE_I32_PER_ROW, k_scales + kb * DECODE_N_BLOCKS) * scale;
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
        int n_heads_q,
        int batch) {

    const int hq = blockIdx.x;
    const int b  = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= batch || hq >= n_heads_q) return;

    float m = -1e38f;
    float l = 0.0f;
    float o = 0.0f;

    for (int s = 0; s < n_splits; ++s) {
        const size_t partial_base = ((size_t(b) * n_heads_q + hq) * size_t(n_splits) + s);

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
        const size_t dst_base = ((size_t(b) * 1 + 0) * size_t(n_heads_q) + hq) * DECODE_D;
        dst[dst_base + tid] = o / l;
    }
}

#define LAUNCH_DECODE_Q4PAIR(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, true><<<g, 256, sm, stream>>>( \
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
    dim3 g1(n_splits, n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_splitk_stage1_kernel<BN, BN_VSUB><<<g1, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, scale, \
        V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, \
        k_head_stride_rows, k_batch_stride_rows, split_size, n_splits); \
    CUDA_CHECK(cudaGetLastError()); \
    dim3 g2(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_splitk_reduce_kernel<<<g2, 256, 0, stream>>>( \
        blockfa_partial_o.ptr, blockfa_partial_m.ptr, blockfa_partial_l.ptr, (float *) dst->data, \
        n_splits, n_heads_q, batch); \
}

#define LAUNCH_DECODE_VSUB(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}
