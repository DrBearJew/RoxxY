// ── Packed16 decode kernel — BN outer QK + BN_VSUB V accumulation ────────
// Env: GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN (default 64), DECODE_VSUB (default 8)
//      GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR=1, DECODE_Q4PAIR=1

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


// ── Dispatch macros ────────────────────────────────────────────────────────

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

#define LAUNCH_DECODE_VSUB(BN, BN_VSUB) { \
    int sm = (BN * 2 + 4 + 64 + 8) * (int)sizeof(float); \
    dim3 g(n_heads_q, batch); \
    ggml_cuda_q8k_dot4_decode_p16_kernel<BN, BN_VSUB, false, false><<<g, 256, sm, stream>>>( \
        q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
        (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
        nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
}
