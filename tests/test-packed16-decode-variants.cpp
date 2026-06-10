// tests/test-packed16-decode-variants.cpp
// Targeted ROCm packed16 decode variant harness.
// Compares opt-in nq==1 packed16 decode implementations against scalar output.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

extern "C" void llama_kv_cache_register_packed16(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales);

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
        int nk, int n_heads_k,
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
                for (int i = 0; i < QK; ++i) amax = fmaxf(amax, fabsf(K_f32[src_base + b * QK + i]));
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                float hs = s;
                ggml_fp32_to_fp16_row(&hs, (ggml_fp16_t *) &scales[row * NB + b], 1);
                for (int g = 0; g < QK / 4; ++g) {
                    int word = 0;
                    for (int j = 0; j < 4; ++j) {
                        const float x = K_f32[src_base + b * QK + g * 4 + j] / s;
                        int qi = (int) lrintf(fmaxf(-127.0f, fminf(127.0f, x)));
                        word |= (int(uint8_t(int8_t(qi))) << (8 * j));
                    }
                    payload[row * I32_PER_ROW + b * (QK / 4) + g] = word;
                }
            }
        }
    }
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) md = std::max(md, fabsf(a[i] - b[i]));
    return md;
}

static float rms_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        s += d*d;
    }
    return (float) std::sqrt(s / std::max<size_t>(a.size(), 1));
}

static bool finite_all(const std::vector<float> & a) {
    for (float x : a) if (!std::isfinite(x)) return false;
    return true;
}

static std::vector<float> make_mask_f32(int nk, int nq) {
    std::vector<float> mask((size_t) nk * nq, 0.0f);
    const char * mode_env = getenv("PACKED16_DECODE_TEST_MASK");
    const std::string mode = mode_env && mode_env[0] ? std::string(mode_env) : std::string("zero");
    if (mode == "zero") {
        return mask;
    }
    if (mode == "causal_tail") {
        for (int q = 0; q < nq; ++q) {
            const int last_visible_k = nk - nq + q;
            for (int k = last_visible_k + 1; k < nk; ++k) {
                mask[(size_t) k * nq + q] = -INFINITY;
            }
        }
        return mask;
    }
    std::fprintf(stderr, "unknown PACKED16_DECODE_TEST_MASK=%s\n", mode.c_str());
    std::abort();
}

struct run_result {
    bool ok = false;
    std::string err;
    std::vector<float> out;
    double compute_ms = 0.0;
    int repeats = 1;
};

static run_result run_variant(
        const char * variant,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        const std::vector<float> & Q_data,
        const std::vector<int> & K_payload,
        const std::vector<uint16_t> & K_scales,
        const std::vector<uint8_t> & V_q4) {

    constexpr int D = 256;
    constexpr int batch = 1;

    setenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ", "1", 1);
    setenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA", "1", 1);
    unsetenv("GGML_CUDA_ROCM_MTP_SOURCE_F16_DOT4_UNSAFE");
    setenv("GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE", "1", 1);
    setenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN", "64", 1);
    setenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD", "100000000", 1);
    if (nq > 1 && strcmp(variant, "scalar") == 0) {
        unsetenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        unsetenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
    } else if (strcmp(variant, "splitk") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_decode_splitk", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "splitk", 1);
    } else if (strcmp(variant, "small_verify") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "small_verify", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_splitk") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_splitk", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "small_verify_splitk", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_batched_splitk") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_batched_splitk", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "small_verify_batched_splitk", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2") == 0 || strcmp(variant, "small_verify_fa2_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "packed16_fa2_vec") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_fa2_vec", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_FA2_VEC", "1", 1);
        unsetenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "packed16_fa2") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_fa2", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "packed16_fa2", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_hybrid_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_hybrid", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_sparsev_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_sparsev", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_pv_dot4_onthefly_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_pv_dot4_onthefly", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_pv_dot4_lds_a_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_pv_dot4_lds_a", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_fusedpv_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_fusedpv", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa2_pvwmma_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2_pvwmma", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa3_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa3", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa4") == 0 || strcmp(variant, "small_verify_fa4_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa4", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "small_verify_fa4_pvwmma") == 0 || strcmp(variant, "small_verify_fa4_pvwmma_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa4_pvwmma", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "bm_dot4_pages") == 0 || strcmp(variant, "bm_dot4_pages_tune") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "bm_dot4_pages_pvwmma") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages_pvwmma", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "bm_dot4_pages_pvwmma", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "bm_dot4_pages_pint8pv") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages_pint8pv", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "bm_dot4_pages_pint8pv", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "bm_dot4_pages_pint8pv_dot4") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages_pint8pv_dot4", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "bm_dot4_pages_pint8pv_dot4", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else if (strcmp(variant, "bm_dot4_pages_intflash_vfrag_dot4") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "bm_dot4_pages_intflash_vfrag_dot4", 1);
    } else if (strcmp(variant, "bm_dot4_pages_intflash_vfrag_wmma") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "bm_dot4_pages_intflash_vfrag_wmma", 1);
        char max_nq_buf[16];
        snprintf(max_nq_buf, sizeof(max_nq_buf), "%d", nq);
        setenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", max_nq_buf, 1);
    } else {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_decode", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", variant, 1);
    }
    setenv("COMPRESSED_KV_FATTN_LOG", "1", 1);

    run_result rr;
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        rr.err = "no GPU backend";
        return rr;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        rr.err = "ggml_init failed";
        return rr;
    }

    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  D, nq, n_heads_q, batch);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, GGML_TYPE_I32,  D/4, nk, n_heads_k, batch);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, D, nk, n_heads_k, batch);
    ggml_tensor * M = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, nq);
    ggml_tensor * P = ggml_new_tensor_4d(ctx, GGML_TYPE_I32,  D/4, nk * n_heads_k, 1, 1);
    ggml_tensor * S = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  D/32, nk * n_heads_k, 1, 1);
    ggml_set_name(Q, "decode_Q");
    ggml_set_name(K, "decode_K_i32_view");
    ggml_set_name(V, "decode_V_q4_0");
    ggml_set_name(M, "decode_mask");
    ggml_set_name(P, "decode_K_payload");
    ggml_set_name(S, "decode_K_scales");

    std::vector<float> mask_f32 = make_mask_f32(nk, nq);
    std::vector<uint16_t> mask_f16(mask_f32.size());
    ggml_fp32_to_fp16_row(mask_f32.data(), (ggml_fp16_t *) mask_f16.data(), (int64_t) mask_f16.size());

    const float sm_scale = 1.0f / std::sqrt((float) D);
    ggml_tensor * O = ggml_flash_attn_ext(ctx, Q, K, V, M, sm_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(O, GGML_PREC_F32);
    // MTP target verification with nq>1 is stamped as PREFILL_QK in the live graph;
    // the draft-decode instruction is only valid for nq==1.
    ((int32_t *) O->op_params)[4] = nq > 1 ? GGML_FATTN_INST_PREFILL_QK : GGML_FATTN_INST_MTP_DRAFT_DECODE_QK;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, O);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        rr.err = "alloc_ctx_tensors failed";
        ggml_free(ctx);
        return rr;
    }
    std::vector<int> K_view((size_t) (D/4) * nk * n_heads_k, 0);
    llama_kv_cache_register_packed16(K->data, P, S);
    ggml_backend_tensor_set(Q, Q_data.data(), 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(K, K_view.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(P, K_payload.data(), 0, ggml_nbytes(P));
    ggml_backend_tensor_set(S, K_scales.data(), 0, ggml_nbytes(S));
    ggml_backend_tensor_set(V, V_q4.data(), 0, ggml_nbytes(V));
    ggml_backend_tensor_set(M, mask_f16.data(), 0, ggml_nbytes(M));

    const int warmup = getenv("PACKED16_DECODE_TEST_WARMUP") ? std::max(0, atoi(getenv("PACKED16_DECODE_TEST_WARMUP"))) : 0;
    const int repeats = getenv("PACKED16_DECODE_TEST_REPEAT") ? std::max(1, atoi(getenv("PACKED16_DECODE_TEST_REPEAT"))) : 1;
    for (int i = 0; i < warmup; ++i) {
        const ggml_status st = ggml_backend_graph_compute(backend, graph);
        if (st != GGML_STATUS_SUCCESS) {
            rr.err = "warmup compute failed status=" + std::to_string((int) st);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return rr;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        const ggml_status st = ggml_backend_graph_compute(backend, graph);
        if (st != GGML_STATUS_SUCCESS) {
            rr.err = "compute failed status=" + std::to_string((int) st);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return rr;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    rr.out.resize(ggml_nelements(O));
    ggml_backend_tensor_get(O, rr.out.data(), 0, ggml_nbytes(O));
    rr.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    rr.repeats = repeats;
    rr.ok = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    // Do not explicitly free this short-lived HIP backend in the harness:
    // each variant needs a fresh backend to avoid graph replay reusing a prior
    // env-selected kernel, and this ROCm stack can abort in legacy pool teardown
    // after these custom packed16 launches despite successful synchronization.
    return rr;
}

static bool mkdir_p(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            const std::string part = path.substr(0, i);
            if (part.empty()) {
                continue;
            }
            if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) {
                std::fprintf(stderr, "failed to create directory %s: errno=%d\n", part.c_str(), errno);
                return false;
            }
        }
    }
    return true;
}

static bool write_binary_file(const std::string & path, const void * data, size_t bytes) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "failed to open %s for write: errno=%d\n", path.c_str(), errno);
        return false;
    }
    const size_t written = bytes == 0 ? 0 : std::fwrite(data, 1, bytes, f);
    const bool ok = written == bytes && std::fclose(f) == 0;
    if (!ok) {
        std::fprintf(stderr, "failed to write %s: wrote=%zu expected=%zu errno=%d\n", path.c_str(), written, bytes, errno);
    }
    return ok;
}

static bool push_unique(std::vector<std::string> & values, const std::string & value) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return false;
    }
    values.push_back(value);
    return true;
}

static bool write_fixture_meta(
        const std::string & dir,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa,
        const std::vector<std::string> & outputs) {
    constexpr int D = 256;
    const float sm_scale = 1.0f / std::sqrt((float) D);
    const std::string path = dir + "/meta.json";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "failed to open %s for write: errno=%d\n", path.c_str(), errno);
        return false;
    }
    std::fprintf(f,
        "{\n"
        "  \"format\": \"packed16_decode_fixture_v1\",\n"
        "  \"batch\": 1,\n"
        "  \"nq\": %d,\n"
        "  \"nk\": %d,\n"
        "  \"hq\": %d,\n"
        "  \"hk\": %d,\n"
        "  \"gqa\": %d,\n"
        "  \"d\": %d,\n"
        "  \"scale\": %.9g,\n"
        "  \"layout\": {\n"
        "    \"q_f32\": \"[batch,hq,nq,d]\",\n"
        "    \"k_payload_i32\": \"[batch,hk,nk,d/4] int32 packed i8x4\",\n"
        "    \"k_scales_f16\": \"[batch,hk,nk,d/32]\",\n"
        "    \"v_q4_0\": \"[batch,hk,nk,d/32] block_q4_0 raw: f16 delta + 16 qs bytes\",\n"
        "    \"mask_f16\": \"[nk,nq] additive mask\",\n"
        "    \"outputs\": \"[batch,nq,hq,d] float32\"\n"
        "  },\n"
        "  \"files\": {\n"
        "    \"q_f32\": \"q_f32.bin\",\n"
        "    \"k_payload_i32\": \"k_payload_i32.bin\",\n"
        "    \"k_scales_f16\": \"k_scales_f16.bin\",\n"
        "    \"v_q4_0\": \"v_q4_0.bin\",\n"
        "    \"mask_f16\": \"mask_f16.bin\"\n"
        "  },\n"
        "  \"outputs\": {\n",
        nq, nk, n_heads_q, n_heads_k, gqa, D, (double) sm_scale);
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::fprintf(f, "    \"%s\": \"out_%s.f32\"%s\n",
                outputs[i].c_str(), outputs[i].c_str(), i + 1 < outputs.size() ? "," : "");
    }
    std::fprintf(f, "  }\n}\n");
    const bool ok = std::fclose(f) == 0;
    if (ok) {
        std::fprintf(stderr, "packed16 fixture dumped: %s\n", dir.c_str());
    }
    return ok;
}

int main() {
    constexpr int D = 256;
    const int nq = getenv("PACKED16_DECODE_TEST_NQ") ? atoi(getenv("PACKED16_DECODE_TEST_NQ")) : 1;
    const int nk = getenv("PACKED16_DECODE_TEST_NK") ? atoi(getenv("PACKED16_DECODE_TEST_NK")) : 1024;
    const int n_heads_k = getenv("PACKED16_DECODE_TEST_HK") ? atoi(getenv("PACKED16_DECODE_TEST_HK")) : 2;
    const int gqa = getenv("PACKED16_DECODE_TEST_GQA") ? atoi(getenv("PACKED16_DECODE_TEST_GQA")) : 4;
    const int n_heads_q = n_heads_k * gqa;

    const int warmup = getenv("PACKED16_DECODE_TEST_WARMUP") ? std::max(0, atoi(getenv("PACKED16_DECODE_TEST_WARMUP"))) : 0;
    const int repeats = getenv("PACKED16_DECODE_TEST_REPEAT") ? std::max(1, atoi(getenv("PACKED16_DECODE_TEST_REPEAT"))) : 1;
    std::printf("packed16 decode variant harness: nq=%d nk=%d hq=%d hk=%d gqa=%d warmup=%d repeat=%d\n", nq, nk, n_heads_q, n_heads_k, gqa, warmup, repeats);

    const float q_scale = getenv("PACKED16_DECODE_TEST_Q_SCALE") ? atof(getenv("PACKED16_DECODE_TEST_Q_SCALE")) : 0.75f;
    const float k_scale = getenv("PACKED16_DECODE_TEST_K_SCALE") ? atof(getenv("PACKED16_DECODE_TEST_K_SCALE")) : 0.75f;
    const float v_scale = getenv("PACKED16_DECODE_TEST_V_SCALE") ? atof(getenv("PACKED16_DECODE_TEST_V_SCALE")) : 0.75f;
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

    std::vector<std::string> variants;
    const char * variants_env = getenv("PACKED16_DECODE_TEST_VARIANTS");
    if (variants_env && variants_env[0]) {
        std::stringstream ss(variants_env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) variants.push_back(item);
        }
    } else {
        variants = {"scalar", "gqa_scalar", "waveqk", "waveqk_q4pair", "pvwmma", "wmma_full", "dsplit", "splitk", "small_verify", "small_verify_splitk", "small_verify_batched_splitk", "small_verify_fa2", "packed16_fa2", "packed16_fa2_vec", "small_verify_fa2_tune", "small_verify_fa2_hybrid_tune", "small_verify_fa2_sparsev_tune", "small_verify_fa2_pv_dot4_onthefly_tune", "small_verify_fa2_pv_dot4_lds_a_tune", "small_verify_fa2_fusedpv_tune", "small_verify_fa2_pvwmma_tune", "small_verify_fa3_tune", "small_verify_fa4", "small_verify_fa4_pvwmma", "bm_dot4_pages", "bm_dot4_pages_pvwmma", "bm_dot4_pages_pint8pv", "bm_dot4_pages_pint8pv_dot4", "bm_dot4_pages_intflash_vfrag_dot4", "bm_dot4_pages_intflash_vfrag_wmma", "logits_debug"};
    }

    run_result scalar = run_variant("scalar", nq, nk, n_heads_q, n_heads_k, Q, K_payload, K_scales, V_q4);
    if (!scalar.ok || !finite_all(scalar.out)) {
        std::fprintf(stderr, "scalar baseline failed: %s finite=%d\n", scalar.err.c_str(), scalar.ok ? (int) finite_all(scalar.out) : 0);
        return 2;
    }

    const char * dump_dir_env = getenv("PACKED16_DECODE_TEST_DUMP_DIR");
    const bool dump_enabled = dump_dir_env && dump_dir_env[0];
    std::string dump_dir = dump_enabled ? std::string(dump_dir_env) : std::string();
    std::vector<std::string> dumped_outputs;
    if (dump_enabled) {
        if (!mkdir_p(dump_dir)) {
            return 3;
        }
        std::vector<float> mask_f32 = make_mask_f32(nk, nq);
        std::vector<uint16_t> mask_f16(mask_f32.size());
        ggml_fp32_to_fp16_row(mask_f32.data(), (ggml_fp16_t *) mask_f16.data(), (int64_t) mask_f16.size());
        bool dump_ok = true;
        dump_ok &= write_binary_file(dump_dir + "/q_f32.bin", Q.data(), Q.size() * sizeof(Q[0]));
        dump_ok &= write_binary_file(dump_dir + "/k_payload_i32.bin", K_payload.data(), K_payload.size() * sizeof(K_payload[0]));
        dump_ok &= write_binary_file(dump_dir + "/k_scales_f16.bin", K_scales.data(), K_scales.size() * sizeof(K_scales[0]));
        dump_ok &= write_binary_file(dump_dir + "/v_q4_0.bin", V_q4.data(), V_q4.size() * sizeof(V_q4[0]));
        dump_ok &= write_binary_file(dump_dir + "/mask_f16.bin", mask_f16.data(), mask_f16.size() * sizeof(mask_f16[0]));
        dump_ok &= write_binary_file(dump_dir + "/out_scalar.f32", scalar.out.data(), scalar.out.size() * sizeof(scalar.out[0]));
        push_unique(dumped_outputs, "scalar");
        if (!dump_ok) {
            return 3;
        }
    }

    bool all_ok = true;
    for (const std::string & variant_name : variants) {
        const char * v = variant_name.c_str();
        run_result r = strcmp(v, "scalar") == 0 ? scalar : run_variant(v, nq, nk, n_heads_q, n_heads_k, Q, K_payload, K_scales, V_q4);
        if (!r.ok) {
            std::printf("variant=%s RESULT=FAIL err=%s\n", v, r.err.c_str());
            all_ok = false;
            continue;
        }
        const float max_abs = max_abs_diff(r.out, scalar.out);
        const float rms = rms_diff(r.out, scalar.out);
        const bool finite = finite_all(r.out);
        const bool pass = finite && max_abs < 2.5e-2f && rms < 5.0e-3f;
        std::printf("variant=%-14s max_abs=%.8g rms=%.8g finite=%d compute_ms=%.6f repeat=%d RESULT=%s\n", v, max_abs, rms, (int) finite, r.compute_ms, r.repeats, pass ? "PASS" : "FAIL");
        if (dump_enabled) {
            const std::string name(v);
            const std::string path = dump_dir + "/out_" + name + ".f32";
            if (!write_binary_file(path, r.out.data(), r.out.size() * sizeof(r.out[0]))) {
                all_ok = false;
            } else {
                push_unique(dumped_outputs, name);
            }
        }
        if (!pass) all_ok = false;
    }

    if (dump_enabled && !write_fixture_meta(dump_dir, nq, nk, n_heads_q, n_heads_k, gqa, dumped_outputs)) {
        all_ok = false;
    }

    return all_ok ? 0 : 1;
}
