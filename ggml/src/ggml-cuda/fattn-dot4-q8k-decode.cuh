// Packed16 decode kernel (nq=1).  No V-tile pre-fetch, direct dequant.
// BN is configurable: 8, 16, 32, 64.
// Shared memory: logits[BN] + probs[BN] + sm[3] + q_i32[64] + q_scl[8].

template <bool USE_F16_V, bool USE_Q8_V, int BN>
static __global__ __launch_bounds__(256, 1)
void ggml_cuda_q8k_dot4_decode_packed16_kernel(
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

    constexpr int I32_PER_ROW = 64;
    constexpr int N_BLOCKS    = 8;

    const int hq = blockIdx.x;
    const int b  = blockIdx.y;
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;
    const int tid = threadIdx.x;

    extern __shared__ __align__(16) unsigned char smem[];
    float * logits = reinterpret_cast<float *>(smem);          // BN
    float * probs  = logits + BN;                               // BN
    float * sm     = probs + BN;                                // 3: row_m, row_l, old_scale
    int   * q_i32  = reinterpret_cast<int  *>(sm + 3);         // 64 ints
    float * q_scl  = reinterpret_cast<float *>(q_i32 + I32_PER_ROW); // 8 floats

    if (tid == 0) { sm[0] = -1e38f; sm[1] = 0.0f; }
    float out = 0.0f;

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

        // 1. QK dot — threads 0..BN-1
        if (tid < BN) {
            const int kk = tid;
            const int k = k0 + kk;
            bool valid = (kk < tile_n && k < nk) && (k - q_offset <= 0);
            float s = -1e38f;
            if (valid) {
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                        q_i32, q_scl,
                        k_payload + k_base * I32_PER_ROW,
                        k_scales  + k_base * N_BLOCKS) * scale;
            }
            logits[kk] = s;
        }
        __syncthreads();

        // 2. Tile softmax — tid==0
        if (tid == 0) {
            float m_prev = sm[0], l_prev = sm[1];
            float tile_m = -1e38f;
            for (int kk = 0; kk < tile_n; ++kk)
                tile_m = fmaxf(tile_m, logits[kk]);
            float m_new     = fmaxf(m_prev, tile_m);
            float old_scale = (l_prev > 0.0f) ? expf(m_prev - m_new) : 0.0f;
            float tile_l = 0.0f;
            for (int kk = 0; kk < tile_n; ++kk) {
                float p = expf(logits[kk] - m_new);
                probs[kk] = p;
                tile_l += p;
            }
            sm[0] = m_new;
            sm[1] = l_prev * old_scale + tile_l;
            sm[2] = old_scale;
        }
        __syncthreads();

        // 3. V accumulation — threads 0..255
        for (int dim0 = tid; dim0 < 256; dim0 += blockDim.x) {
            out *= sm[2];
            for (int kk = 0; kk < tile_n; ++kk) {
                const int k = k0 + kk;
                const float v = USE_F16_V
                    ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[dim0])
                    : (USE_Q8_V
                        ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, dim0)
                        : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, dim0));
                out += probs[kk] * v;
            }
        }
        __syncthreads();
    }

    // Write output
    for (int dim0 = tid; dim0 < 256; dim0 += blockDim.x) {
        dst[((size_t(b) * nq + 0) * size_t(n_heads_q) + hq) * 256 + dim0] = out / sm[1];
    }
}

// ── Decode kernel instantiation macro ─────────────────────────────────────

#define LAUNCH_DECODE_BN(BN) {\
    const int smem_decode = (BN * 2 + 3 + 64 + 8) * (int)sizeof(float); \
    const dim3 decode_grid(n_heads_q, batch); \
    if (use_f16_v) { \
        ggml_cuda_q8k_dot4_decode_packed16_kernel<true, false, BN><<<decode_grid, 256, smem_decode, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
    } else if (use_q8_v) { \
        ggml_cuda_q8k_dot4_decode_packed16_kernel<false, true, BN><<<decode_grid, 256, smem_decode, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
    } else { \
        ggml_cuda_q8k_dot4_decode_packed16_kernel<false, false, BN><<<decode_grid, 256, smem_decode, stream>>>( \
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data, \
            (float *) dst->data, scale, V->nb[0], V->nb[1], V->nb[2], V->nb[3], \
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows); \
    } \
}
