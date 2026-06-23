// tests/test-pdmq-row-equivalence.cpp
// Minimal ROCm packed16 DOT4-MMQ/PDMQ row-equivalence reproducer.
//
// This is intentionally outside llama-server and MTP state handling. It builds a
// single ggml_flash_attn_ext op over synthetic packed16 K=i32 + V=q4_0 data,
// forces the rocm_packed16_dot4_mmq route, disables HIP graph capture, and
// compares one batched nq=4 explicit-mask run against four row-by-row PDMQ runs
// using the same per-row mask. The target bug class is future-masked rows in
// compact raw_lds_q4 PDMQ.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void llama_kv_cache_register_packed16_with_layout_info(
        const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales,
        int layout_kind, uint32_t kv_capacity, uint32_t d);

static void fill_f32(std::vector<float> & data, float scale, int seed) {
    srand(seed);
    for (float & x : data) {
        x = ((float) rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

static void fill_f16(std::vector<uint16_t> & data, float scale, int seed) {
    std::vector<float> tmp(data.size());
    fill_f32(tmp, scale, seed);
    ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) data.data(), (int64_t) data.size());
}

static std::vector<uint8_t> quantize_q4_0_rows(const std::vector<float> & data, int rows, int cols) {
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, (int64_t) rows * cols));
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, data.data(), q.data(), 0, rows, cols, nullptr);
    if (written != q.size()) {
        std::fprintf(stderr, "q4_0 quantize wrote %zu bytes, expected %zu\n", written, q.size());
        std::abort();
    }
    return q;
}

static void make_packed16_from_f16(
        const std::vector<uint16_t> & K_f16,
        int nk,
        int n_heads_k,
        std::vector<int> & payload,
        std::vector<uint16_t> & scales) {
    constexpr int D = 256;
    constexpr int QK = 32;
    constexpr int I32_PER_ROW = D / 4;
    constexpr int NB = D / QK;
    std::vector<float> K_f32(K_f16.size());
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) K_f16.data(), K_f32.data(), (int64_t) K_f16.size());
    payload.assign((size_t) n_heads_k * nk * I32_PER_ROW, 0);
    scales.assign((size_t) n_heads_k * nk * NB, 0);
    for (int hk = 0; hk < n_heads_k; ++hk) {
        for (int k = 0; k < nk; ++k) {
            const size_t src_base = ((size_t) hk * nk + k) * D;
            const size_t row = (size_t) hk * nk + k;
            for (int b = 0; b < NB; ++b) {
                float amax = 0.0f;
                for (int i = 0; i < QK; ++i) {
                    amax = fmaxf(amax, fabsf(K_f32[src_base + b * QK + i]));
                }
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                ggml_fp32_to_fp16_row(&s, (ggml_fp16_t *) &scales[row * NB + b], 1);
                for (int g = 0; g < QK / 4; ++g) {
                    int word = 0;
                    for (int j = 0; j < 4; ++j) {
                        const float x = K_f32[src_base + b * QK + g * 4 + j] / s;
                        const int qi = (int) lrintf(fmaxf(-127.0f, fminf(127.0f, x)));
                        word |= (int(uint8_t(int8_t(qi))) << (8 * j));
                    }
                    payload[row * I32_PER_ROW + b * (QK / 4) + g] = word;
                }
            }
        }
    }
}

static std::vector<float> make_future_tail_mask(int nk, int nq) {
    // ggml tensor M is [nk,nq], so k is fastest: index = q*nk + k.
    // Row q sees K positions <= nk-nq+q and masks later in-block future rows.
    std::vector<float> mask((size_t) nk * nq, 0.0f);
    for (int q = 0; q < nq; ++q) {
        const int last_visible_k = nk - nq + q;
        for (int k = last_visible_k + 1; k < nk; ++k) {
            mask[(size_t) q * nk + k] = -INFINITY;
        }
    }
    return mask;
}

static std::vector<float> slice_q_row(const std::vector<float> & q_all, int q_row, int nq, int n_heads_q) {
    constexpr int D = 256;
    std::vector<float> q_one((size_t) D * n_heads_q);
    for (int h = 0; h < n_heads_q; ++h) {
        const size_t src = ((size_t) h * nq + q_row) * D;
        const size_t dst = (size_t) h * D;
        std::copy(q_all.begin() + src, q_all.begin() + src + D, q_one.begin() + dst);
    }
    return q_one;
}

static std::vector<float> slice_mask_row(const std::vector<float> & mask_all, int q_row, int nk) {
    std::vector<float> mask_one((size_t) nk);
    std::copy(mask_all.begin() + (size_t) q_row * nk,
              mask_all.begin() + (size_t) (q_row + 1) * nk,
              mask_one.begin());
    return mask_one;
}

static void zero_future_kv_rows(
        int q_row,
        int nq,
        int nk,
        int n_heads_k,
        std::vector<int> & k_payload,
        std::vector<uint16_t> & k_scales,
        std::vector<uint8_t> & v_q4) {
    constexpr int D = 256;
    constexpr int K_I32_PER_ROW = D / 4;
    constexpr int K_SCALE_PER_ROW = D / 32;
    const size_t v_row_bytes = ggml_row_size(GGML_TYPE_Q4_0, D);
    const int first_future_k = nk - nq + q_row + 1;
    for (int hk = 0; hk < n_heads_k; ++hk) {
        for (int k = first_future_k; k < nk; ++k) {
            const size_t row = (size_t) hk * nk + k;
            std::fill(k_payload.begin() + row * K_I32_PER_ROW,
                      k_payload.begin() + (row + 1) * K_I32_PER_ROW, 0);
            std::fill(k_scales.begin() + row * K_SCALE_PER_ROW,
                      k_scales.begin() + (row + 1) * K_SCALE_PER_ROW, 0);
            std::fill(v_q4.begin() + row * v_row_bytes,
                      v_q4.begin() + (row + 1) * v_row_bytes, 0);
        }
    }
}

struct run_result {
    bool ok = false;
    std::string err;
    std::vector<float> out;
};

static void setenv_default(const char * name, const char * value) {
    const char * cur = getenv(name);
    if (!cur || !*cur) {
        setenv(name, value, 1);
    }
}


static void force_pdmq_env() {
    setenv_default("GGML_CUDA_DISABLE_GRAPHS", "1");
    setenv_default("GGML_CUDA_ROCM_Q8K_DOT4_KQ", "1");
    setenv_default("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA", "1");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ", "1");
    setenv_default("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_dot4_mmq");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE", "1x32");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP", "1");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_REQUIRE", "1");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK", "1");
    setenv_default("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH", "raw_lds_q4");
    setenv_default("GGML_CUDA_PDMQ_DEBUG_SKIP_PROBES", "1");
    unsetenv("GGML_CUDA_DP16_FA_PV_WMMA");
    unsetenv("GGML_CUDA_DP16_FA_PV_I8_WMMA");
    unsetenv("GGML_CUDA_DP16_FA_PV_I4_WMMA");
}

static run_result run_pdmq(
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        const std::vector<float> & Q_data,
        const std::vector<int> & K_payload,
        const std::vector<uint16_t> & K_scales,
        const std::vector<uint8_t> & V_q4,
        const std::vector<float> & mask_f32,
        ggml_fattn_instruction inst) {
    constexpr int D = 256;
    constexpr int batch = 1;

    force_pdmq_env();

    run_result rr;
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        rr.err = "no GPU backend";
        return rr;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        rr.err = "ggml_init failed";
        return rr;
    }

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  D, nq, n_heads_q, batch);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, GGML_TYPE_I32,  D / 4, nk, n_heads_k, batch);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nk, n_heads_k, batch);
    ggml_tensor * M = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, nq);
    ggml_tensor * P = ggml_new_tensor_4d(ctx, GGML_TYPE_I32,  D / 4, nk * n_heads_k, 1, 1);
    ggml_tensor * S = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  D / 32, nk * n_heads_k, 1, 1);
    ggml_set_name(Q, "pdmq_roweq_Q");
    ggml_set_name(K, "pdmq_roweq_K_i32_view");
    ggml_set_name(V, "pdmq_roweq_V_q4_0");
    ggml_set_name(M, "pdmq_roweq_mask");
    ggml_set_name(P, "pdmq_roweq_K_payload");
    ggml_set_name(S, "pdmq_roweq_K_scales");

    std::vector<uint16_t> mask_f16(mask_f32.size());
    ggml_fp32_to_fp16_row(mask_f32.data(), (ggml_fp16_t *) mask_f16.data(), (int64_t) mask_f16.size());

    const float sm_scale = 1.0f / std::sqrt((float) D);
    ggml_tensor * O = ggml_flash_attn_ext(ctx, Q, K, V, M, sm_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(O, GGML_PREC_F32);
    ggml_flash_attn_ext_set_instruction(O, inst);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, O);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        rr.err = "alloc_ctx_tensors failed";
        ggml_free(ctx);
        return rr;
    }

    std::vector<int> K_view((size_t) (D / 4) * nk * n_heads_k, 0);
    llama_kv_cache_register_packed16_with_layout_info(K->data, P, S, 0, (uint32_t) nk, (uint32_t) D);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(K, K_view.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(P, K_payload.data(), 0, ggml_nbytes(P));
    ggml_backend_tensor_set(S, K_scales.data(), 0, ggml_nbytes(S));
    ggml_backend_tensor_set(V, V_q4.data(), 0, ggml_nbytes(V));
    ggml_backend_tensor_set(M, mask_f16.data(), 0, ggml_nbytes(M));

    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        rr.err = "compute failed status=" + std::to_string((int) st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return rr;
    }

    rr.out.resize(ggml_nelements(O));
    ggml_backend_tensor_get(O, rr.out.data(), 0, ggml_nbytes(O));
    rr.ok = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    // Match the existing packed16 harness: leave the short-lived ROCm backend to
    // process teardown to avoid legacy pool teardown aborts after custom routes.
    return rr;
}

static bool finite_all(const std::vector<float> & a) {
    for (float x : a) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

int main() {

    constexpr int D = 256;
    const int nq = getenv("PDMQ_ROWEQ_NQ") ? atoi(getenv("PDMQ_ROWEQ_NQ")) : 4;
    const int nk = getenv("PDMQ_ROWEQ_NK") ? atoi(getenv("PDMQ_ROWEQ_NK")) : 1280;
    const int n_heads_k = getenv("PDMQ_ROWEQ_HK") ? atoi(getenv("PDMQ_ROWEQ_HK")) : 1;
    const int gqa = getenv("PDMQ_ROWEQ_GQA") ? atoi(getenv("PDMQ_ROWEQ_GQA")) : 1;
    const int n_heads_q = n_heads_k * gqa;
    const float q_scale = getenv("PDMQ_ROWEQ_Q_SCALE") ? atof(getenv("PDMQ_ROWEQ_Q_SCALE")) : 0.75f;
    const float k_scale = getenv("PDMQ_ROWEQ_K_SCALE") ? atof(getenv("PDMQ_ROWEQ_K_SCALE")) : 0.75f;
    const float v_scale = getenv("PDMQ_ROWEQ_V_SCALE") ? atof(getenv("PDMQ_ROWEQ_V_SCALE")) : 0.75f;

    if (nq < 2 || nk < nq || n_heads_k < 1 || gqa < 1) {
        std::fprintf(stderr, "invalid shape nq=%d nk=%d hk=%d gqa=%d\n", nq, nk, n_heads_k, gqa);
        return 2;
    }

    force_pdmq_env();
    std::printf("pdmq row-equivalence repro: nq=%d nk=%d hq=%d hk=%d gqa=%d shape=%s gqa_group=%s split_k=%s vpath=%s qpack=%s\n",
            nq, nk, n_heads_q, n_heads_k, gqa,
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHAPE"),
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQA_GROUP"),
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_GQAX_SPLITK"),
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH"),
            getenv("GGML_CUDA_DP16_FA_QPACK_I8") ? getenv("GGML_CUDA_DP16_FA_QPACK_I8") : "0");

    std::vector<float> Q((size_t) D * nq * n_heads_q);
    std::vector<uint16_t> K((size_t) D * nk * n_heads_k);
    std::vector<float> V_f32((size_t) D * nk * n_heads_k);
    fill_f32(Q, q_scale, 123);
    fill_f16(K, k_scale, 124);
    fill_f32(V_f32, v_scale, 125);

    std::vector<int> K_payload;
    std::vector<uint16_t> K_scales;
    make_packed16_from_f16(K, nk, n_heads_k, K_payload, K_scales);
    std::vector<uint8_t> V_q4 = quantize_q4_0_rows(V_f32, nk * n_heads_k, D);
    std::vector<float> mask = make_future_tail_mask(nk, nq);

    run_result batched = run_pdmq(nq, nk, n_heads_q, n_heads_k, Q, K_payload, K_scales, V_q4, mask,
                                  GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK);
    if (!batched.ok || !finite_all(batched.out)) {
        std::fprintf(stderr, "batched PDMQ failed: %s finite=%d\n", batched.err.c_str(), batched.ok ? (int) finite_all(batched.out) : 0);
        return 3;
    }

    bool all_ok = true;
    for (int q = 0; q < nq; ++q) {
        std::vector<float> q_one = slice_q_row(Q, q, nq, n_heads_q);
        std::vector<float> m_one = slice_mask_row(mask, q, nk);
        std::vector<int> k_payload_row = K_payload;
        std::vector<uint16_t> k_scales_row = K_scales;
        std::vector<uint8_t> v_q4_row = V_q4;
        // Row-equivalent qwen35 prefix attention has not materialized future
        // in-block K/V rows yet. If a batched explicit-mask kernel really masks
        // them, replacing those rows with zeros in the serial reference must not
        // change the visible result.
        zero_future_kv_rows(q, nq, nk, n_heads_k, k_payload_row, k_scales_row, v_q4_row);
        run_result row = run_pdmq(1, nk, n_heads_q, n_heads_k, q_one, k_payload_row, k_scales_row, v_q4_row, m_one,
                                  GGML_FATTN_INST_MTP_DRAFT_DECODE_QK);
        if (!row.ok || !finite_all(row.out)) {
            std::printf("row=%d RESULT=FAIL err=%s finite=%d\n", q, row.err.c_str(), row.ok ? (int) finite_all(row.out) : 0);
            all_ok = false;
            continue;
        }

        double sum_sq = 0.0;
        float max_abs = 0.0f;
        int max_h = 0;
        int max_d = 0;
        for (int h = 0; h < n_heads_q; ++h) {
            for (int d = 0; d < D; ++d) {
                const float b = batched.out[((size_t) q * n_heads_q + h) * D + d];
                const float r = row.out[(size_t) h * D + d];
                const float diff = fabsf(b - r);
                if (diff > max_abs) {
                    max_abs = diff;
                    max_h = h;
                    max_d = d;
                }
                sum_sq += double(diff) * double(diff);
            }
        }
        const float rms = (float) std::sqrt(sum_sq / double((size_t) n_heads_q * D));
        const bool future_masked = q + 1 < nq;
        const bool pass = max_abs < 1.0e-4f && rms < 1.0e-5f;
        std::printf("row=%d future_masked=%d max_abs=%.9g rms=%.9g max_h=%d max_d=%d RESULT=%s\n",
                q, (int) future_masked, max_abs, rms, max_h, max_d, pass ? "PASS" : "FAIL");
        if (!pass) {
            all_ok = false;
        }
    }

    return all_ok ? 0 : 1;
}
