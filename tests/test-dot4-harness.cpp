// tests/test-dot4-harness.cpp
// DOT4 correctness harness: compares DOT4 flash_attn output against reference (VEC/TILE fallback).
// Usage:
//   FULL_FA=1:  ./build-rocm-rdna3-fa/bin/test-dot4-harness
//   FULL_FA=0:  GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=0 ./build-rocm-rdna3-fa/bin/test-dot4-harness
//   Compare:     diff /tmp/dot4_out.txt /tmp/ref_out.txt (human review)

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

// ─── helpers ───────────────────────────────────────────────────

static void fill_f32(float * data, size_t n, float scale, int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

static void fill_f16(uint16_t * data, size_t n, float scale, int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        uint32_t bits; memcpy(&bits, &v, 4);
        data[i] = (uint16_t)(bits >> 16);
    }
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; i++) md = fmaxf(md, fabsf(a[i] - b[i]));
    return md;
}

static float mean_abs_diff(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += fabsf(a[i] - b[i]);
    return (float)(sum / n);
}

static bool has_nan(const float * data, size_t n) {
    for (size_t i = 0; i < n; i++) if (std::isnan(data[i])) return true;
    return false;
}

static bool has_inf(const float * data, size_t n) {
    for (size_t i = 0; i < n; i++) if (std::isinf(data[i])) return true;
    return false;
}

static float l2_norm(const float * data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double)data[i] * data[i];
    return (float)sqrt(sum);
}

static float cosine_similarity(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    return denom > 0.0 ? (float)(dot / denom) : 0.0f;
}

static float relative_l2_error(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = (double)a[i] - b[i];
        num += diff * diff;
        den += (double)b[i] * b[i];
    }
    return den > 0.0 ? (float)(sqrt(num) / sqrt(den)) : 0.0f;
}

// ── Proper f16 → f32 conversion: use ggml's built-in ──────
// ggml_fp16_to_fp32_row handles IEEE 754 half→float correctly.
// ── CPU reference: QK logits, softmax, attention O ──────────
// Uses ggml's known-correct f16→f32 conversion.

static void cpu_qk_logits(
    const float * Q, const uint16_t * K_f16,
    int D, int nq, int nk, int head_q, int head_k,
    float * logits)
{
    std::vector<float> K_f32(D * nk);
    ggml_fp16_to_fp32_row(K_f16, K_f32.data(), D * nk);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);
    for (int q = 0; q < nq; q++) {
        const float * Q_row = Q + (q * D + head_q * D * nq);
        for (int k = 0; k < nk; k++) {
            const float * K_col = K_f32.data() + (k * D + head_k * D * nk);
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += Q_row[d] * K_col[d];
            logits[q * nk + k] = dot * inv_sqrt_d;
        }
    }
}

static float cpu_softmax_max(const float * row, int n) {
    float m = row[0];
    for (int i = 1; i < n; i++) m = fmaxf(m, row[i]);
    return m;
}

static float cpu_softmax_sum(const float * row, int n, float row_max) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(row[i] - row_max);
    return sum;
}

static void cpu_attention_O(
    const float * Q, const uint16_t * K_f16, const uint16_t * V_f16,
    int D, int nq, int nk, int head_q, int head_k,
    float * O)
{
    std::vector<float> K_f32(D * nk);
    ggml_fp16_to_fp32_row(K_f16, K_f32.data(), D * nk);
    std::vector<float> V_f32(D * nk);
    ggml_fp16_to_fp32_row(V_f16, V_f32.data(), D * nk);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    std::vector<float> logits(nq * nk);
    cpu_qk_logits(Q, K_f16, D, nq, nk, head_q, head_k, logits.data());

    // Apply causal mask
    for (int q = 0; q < nq; q++) {
        for (int k = 0; k < nk; k++) {
            if (k > nk - nq + q) logits[q * nk + k] = -INFINITY;
        }
    }

    // Softmax + weighted sum
    memset(O, 0, D * nq * sizeof(float));
    for (int q = 0; q < nq; q++) {
        float * logit_row = logits.data() + q * nk;
        float m = cpu_softmax_max(logit_row, nk);
        float sum = 0.0f;
        for (int k = 0; k < nk; k++) {
            float w = expf(logit_row[k] - m);
            sum += w;
            const float * V_col = V_f32.data() + (k * D + head_k * D * nk);
            for (int d = 0; d < D; d++) O[q * D + d] += w * V_col[d];
        }
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < D; d++) O[q * D + d] *= inv_sum;
    }
}

static void dump8(const char * label, const float * data, size_t n) {
    printf("%s [%zu]:", label, n);
    size_t m = std::min(n, (size_t)8);
    for (size_t i = 0; i < m; i++) printf(" %.4f", data[i]);
    if (n > 8) printf(" ...");
    printf("\n");
}

// ─── compute one flash_attn_ext graph ──────────────────────────

struct fa_result {
    std::vector<float> output;  // [D * nq * n_heads]
    std::vector<float> softmax_lse; // [nq * n_heads]
    size_t nq, nk, D, n_heads;
    bool ok;
    std::string error;
};

static fa_result run_flash_attn(
    const char * env_full_fa,  // "1" or "0"
    int nq, int nk, int D, int n_heads,
    const float * Q_f32,
    const uint16_t * K_f16,
    const uint16_t * V_f16,
    float sm_scale,
    float max_bias,
    int32_t fa_inst)  // GGML_FATTN_INST_* value to stamp in op_params[4]
{
    fa_result r{};
    r.nq = (size_t)nq;
    r.nk = (size_t)nk;
    r.D = (size_t)D;
    r.n_heads = (size_t)n_heads;

    // Set env for this run
    if (env_full_fa) {
        setenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA", env_full_fa, 1);
    }

    // Init backends — use GPU backend, not name-based which won't match ROCm
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        r.error = "no backend";
        return r;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        r.error = "ggml_init failed";
        ggml_backend_free(backend);
        return r;
    }

    // Create tensors
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, n_heads, 1);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nk, n_heads, 1);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nk, n_heads, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, nq);

    ggml_set_name(Q, "Q"); ggml_set_name(K, "K"); ggml_set_name(V, "V"); ggml_set_name(mask, "mask");

    // Causal mask: lower triangular, -inf in masked positions
    std::vector<uint16_t> mask_f16(nk * nq);
    for (int q = 0; q < nq; q++) {
        for (int k = 0; k < nk; k++) {
            bool visible = (k <= nk - nq + q); // causal
            float v = visible ? 0.0f : -1e9f;
            uint32_t bits; memcpy(&bits, &v, 4);
            mask_f16[q * nk + k] = (uint16_t)(bits >> 16);
        }
    }

    // Build op
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, mask, sm_scale, max_bias, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    // Stamp the flash attention instruction so the dispatch path knows the workload.
    // op_params[4] is the instruction field (int32_t).
    ((int32_t *)out->op_params)[4] = fa_inst;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Allocate backend buffer for all context tensors
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { r.error = "alloc_ctx_tensors failed"; ggml_free(ctx); ggml_backend_free(backend); return r; }

    // Copy input data to backend tensors
    ggml_backend_tensor_set(Q, Q_f32, 0, ggml_nbytes(Q));
    ggml_backend_tensor_set(K, K_f16, 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, V_f16, 0, ggml_nbytes(V));
    ggml_backend_tensor_set(mask, mask_f16.data(), 0, ggml_nbytes(mask));

    // Allocate graph
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);

    ggml_status status = ggml_backend_graph_compute(backend, graph);

    // Verify CUDA handled the op
    printf("  backend=%s compute_status=%d\n", ggml_backend_name(backend), (int)status);
    if (status != GGML_STATUS_SUCCESS) {
        r.error = "compute failed";
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return r;
    }

    // Copy output
    r.output.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.output.data(), 0, ggml_nbytes(out));

    r.ok = true;

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return r;
}

// ─── main ──────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    const char * env_full_fa = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA");
    bool force_fa = (env_full_fa && strcmp(env_full_fa, "1") == 0);

    printf("=== DOT4 Correctness Harness ===\n");
    printf("FULL_FA env: %s\n\n", env_full_fa ? env_full_fa : "(unset)");

    // Test configurations
    struct {
        int nq, nk, D, nh;
        const char * name;
    } tests[] = {
        { 2,  256, 256, 2, "recthist (nq=2, small)" },
        { 4,  256, 256, 2, "recthist (nq=4, small)" },
        { 2, 1024, 256, 2, "recthist (nq=2, 1K)" },
        { 1, 1024, 256, 2, "decode   (nq=1, 1K)" },
        { 1,   64, 256, 2, "decode   (nq=1, 64)" },
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    // Map test config → flash attention instruction
    // recthist tests use PREFILL_QK, decode tests use DECODE_QK
    int32_t inst_map[] = {
        4,  // Test 1: PREFILL_QK (nq=2, recthist)
        4,  // Test 2: PREFILL_QK (nq=4, recthist)
        4,  // Test 3: PREFILL_QK (nq=2, 1K)
        5,  // Test 4: DECODE_QK  (nq=1, 1K)
        5,  // Test 5: DECODE_QK  (nq=1, 64)
    };

    bool all_ok = true;

    for (int ti = 0; ti < n_tests; ti++) {
        auto & tc = tests[ti];
        printf("═══ Test %d: %s ═══\n", ti+1, tc.name);
        printf("nq=%d nk=%d D=%d heads=%d\n", tc.nq, tc.nk, tc.D, tc.nh);

        size_t n_elems_q = tc.D * tc.nq * tc.nh;
        size_t n_elems_kv = tc.D * tc.nk * tc.nh;

        std::vector<float> Q_data(n_elems_q);
        std::vector<uint16_t> K_data(n_elems_kv);
        std::vector<uint16_t> V_data(n_elems_kv);

        fill_f32(Q_data.data(), n_elems_q, 1.0f, 42 + ti);
        fill_f16(K_data.data(), n_elems_kv, 1.0f, 43 + ti);
        fill_f16(V_data.data(), n_elems_kv, 1.0f, 44 + ti);

        float sm_scale = 1.0f / sqrtf((float)tc.D);

        // Run DOT4 path
        printf("  DOT4 path: ");
        fflush(stdout);
        auto r_dot4 = run_flash_attn("1", tc.nq, tc.nk, tc.D, tc.nh,
                                      Q_data.data(), K_data.data(), V_data.data(),
                                      sm_scale, 0.0f, inst_map[ti]);
        if (!r_dot4.ok) { printf("FAIL: %s\n", r_dot4.error.c_str()); all_ok = false; continue; }
        printf("OK (%zu elems)\n", r_dot4.output.size());

        // Run reference path
        printf("  REF  path: ");
        fflush(stdout);
        auto r_ref = run_flash_attn("0", tc.nq, tc.nk, tc.D, tc.nh,
                                     Q_data.data(), K_data.data(), V_data.data(),
                                     sm_scale, 0.0f, inst_map[ti]);
        if (!r_ref.ok) { printf("FAIL: %s\n", r_ref.error.c_str()); all_ok = false; continue; }
        printf("OK (%zu elems)\n", r_ref.output.size());

        // Compare
        size_t n_out = r_dot4.output.size();
        if (n_out != r_ref.output.size()) {
            printf("  *** SIZE MISMATCH: DOT4=%zu REF=%zu ***\n", n_out, r_ref.output.size());
            all_ok = false;
            continue;
        }

        float mad = max_abs_diff(r_dot4.output.data(), r_ref.output.data(), n_out);
        float mnd = mean_abs_diff(r_dot4.output.data(), r_ref.output.data(), n_out);
        float l2_dot4 = l2_norm(r_dot4.output.data(), n_out);
        float l2_ref = l2_norm(r_ref.output.data(), n_out);

        bool nan_dot4 = has_nan(r_dot4.output.data(), n_out);
        bool nan_ref  = has_nan(r_ref.output.data(), n_out);
        bool inf_dot4 = has_inf(r_dot4.output.data(), n_out);
        bool inf_ref  = has_inf(r_ref.output.data(), n_out);

        printf("  max_abs_diff = %e\n", mad);
        printf("  mean_abs_diff= %e\n", mnd);
        printf("  L2 DOT4      = %.6f\n", l2_dot4);
        printf("  L2 REF       = %.6f\n", l2_ref);

        float rel_l2 = relative_l2_error(r_dot4.output.data(), r_ref.output.data(), n_out);
        float cos_sim = cosine_similarity(r_dot4.output.data(), r_ref.output.data(), n_out);
        printf("  relative_l2  = %e\n", rel_l2);
        printf("  cosine_sim   = %.6f\n", cos_sim);
        printf("  NaN:  DOT4=%d REF=%d\n", (int)nan_dot4, (int)nan_ref);
        printf("  Inf:  DOT4=%d REF=%d\n", (int)inf_dot4, (int)inf_ref);

        dump8("  DOT4 out", r_dot4.output.data(), n_out);
        dump8("  REF  out", r_ref.output.data(), n_out);

        // ── CPU reference (skip for large nk to avoid OOM/timeout) ──
        if (tc.nk <= 1024) {
            int H = tc.nh;
            for (int h = 0; h < std::min(H, 2); h++) {
                int head_k = h;

                std::vector<float> O_cpu(tc.D * tc.nq);
                cpu_attention_O(Q_data.data(), K_data.data(), V_data.data(),
                                tc.D, tc.nq, tc.nk, h, head_k, O_cpu.data());

                std::vector<float> O_dot4_head(tc.D * tc.nq);
                for (int q = 0; q < tc.nq; q++) {
                    for (int d = 0; d < tc.D; d++) {
                        O_dot4_head[q * tc.D + d] = r_dot4.output[(h * tc.nq + q) * tc.D + d];
                    }
                }
                std::vector<float> O_ref_head(tc.D * tc.nq);
                for (int q = 0; q < tc.nq; q++) {
                    for (int d = 0; d < tc.D; d++) {
                        O_ref_head[q * tc.D + d] = r_ref.output[(h * tc.nq + q) * tc.D + d];
                    }
                }

                bool nan_cpu = has_nan(O_cpu.data(), tc.D * tc.nq);
                bool inf_cpu = has_inf(O_cpu.data(), tc.D * tc.nq);
                float mad_dot4_cpu = nan_cpu || inf_cpu ? INFINITY :
                    max_abs_diff(O_dot4_head.data(), O_cpu.data(), tc.D * tc.nq);
                float mad_ref_cpu  = nan_cpu || inf_cpu ? INFINITY :
                    max_abs_diff(O_ref_head.data(),  O_cpu.data(), tc.D * tc.nq);
                printf("  Head %d: DOT4vsCPU=%.6e  REFvsCPU=%.6e  CPU_nan=%d CPU_inf=%d\n",
                    h, mad_dot4_cpu, mad_ref_cpu, (int)nan_cpu, (int)inf_cpu);
                dump8("  CPU O", O_cpu.data(), tc.D * std::min(tc.nq, 2));
            }
        }

        // Pass criteria: no NaN/Inf, reasonable diff
        bool pass = !nan_dot4 && !nan_ref && !inf_dot4 && !inf_ref &&
                    mad < 1.0f && mnd < 0.1f;
        printf("  RESULT: %s\n\n", pass ? "PASS" : "FAIL");
        if (!pass) all_ok = false;
    }

    printf("═══ OVERALL: %s ═══\n", all_ok ? "ALL PASS" : "SOME FAILURES");
    return all_ok ? 0 : 1;
}
