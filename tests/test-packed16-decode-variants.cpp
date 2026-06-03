// tests/test-packed16-decode-variants.cpp
// Targeted ROCm packed16 decode variant harness.
// Compares opt-in nq==1 packed16 decode implementations against scalar output.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
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
    } else if (strcmp(variant, "small_verify_fa2") == 0) {
        setenv("GGML_CUDA_FA_ROUTE_REQUIRE", "rocm_packed16_small_verify_fa2", 1);
        setenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL", "small_verify_fa2", 1);
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

    std::vector<float> mask_f32((size_t) nk * nq, 0.0f);
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

    std::vector<float> Q((size_t) D * nq * n_heads_q);
    std::vector<uint16_t> K((size_t) D * nk * n_heads_k);
    std::vector<float> V_f32((size_t) D * nk * n_heads_k);
    fill_f32(Q, 0.75f, 123);
    fill_f16(K, 0.75f, 124);
    fill_f32(V_f32, 0.75f, 125);
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
        variants = {"scalar", "gqa_scalar", "waveqk", "waveqk_q4pair", "pvwmma", "wmma_full", "dsplit", "splitk", "small_verify", "small_verify_splitk", "small_verify_batched_splitk", "small_verify_fa2", "logits_debug"};
    }

    run_result scalar = run_variant("scalar", nq, nk, n_heads_q, n_heads_k, Q, K_payload, K_scales, V_q4);
    if (!scalar.ok || !finite_all(scalar.out)) {
        std::fprintf(stderr, "scalar baseline failed: %s finite=%d\n", scalar.err.c_str(), scalar.ok ? (int) finite_all(scalar.out) : 0);
        return 2;
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
        if (!pass) all_ok = false;
    }

    return all_ok ? 0 : 1;
}
