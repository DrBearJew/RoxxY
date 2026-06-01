// tests/test-dp16-packed16-mmvq.cpp
// Focused DP16/MMVQ harness for the RDNA3 q8 route vs packed16_i32_scaled sidecar route.
//
// It constructs deterministic Q8_0 weights and F32 activations, runs GGML_OP_MUL_MAT
// through the ROCm backend twice, and compares:
//   - rocm_q8_dot4_mmvq
//   - rocm_packed16_dot4_mmvq
//
// Usage:
//   GGML_CUDA_DP16_TRACE=1 ./build-rocm-fixed/bin/test-dp16-packed16-mmvq --m 16 --n 1,2,3,4 --k 256,512

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct run_result {
    bool ok = false;
    std::string error;
    std::vector<float> output;
};

static std::vector<int> parse_list(const char * s) {
    std::vector<int> out;
    if (!s || !*s) {
        return out;
    }

    const char * p = s;
    while (*p) {
        char * end = nullptr;
        errno = 0;
        long v = std::strtol(p, &end, 10);
        if (errno != 0 || end == p || v <= 0 || v > INT32_MAX) {
            std::fprintf(stderr, "bad integer list value near '%s'\n", p);
            std::exit(2);
        }
        out.push_back((int) v);
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            std::fprintf(stderr, "bad integer list separator near '%s'\n", p);
            std::exit(2);
        }
    }
    return out;
}

static float deterministic_value(size_t i, int salt, float scale) {
    const float a = std::sin((float) (i + 1 + 17*salt) * 0.013f);
    const float b = std::cos((float) (i + 3 + 31*salt) * 0.021f);
    return scale * (0.75f*a + 0.25f*b);
}

static void fill_weights(std::vector<float> & w, int m, int k, int salt = 1) {
    w.resize((size_t) m * (size_t) k);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = deterministic_value(i, salt, 0.35f);
    }
}

static void fill_activations(std::vector<float> & x, int k, int n) {
    x.resize((size_t) k * (size_t) n);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = deterministic_value(i, 7, 0.45f);
    }
}

static std::vector<uint8_t> quantize_rows(const std::vector<float> & w, int m, int k, ggml_type type) {
    const size_t row_size = ggml_row_size(type, k);
    std::vector<uint8_t> q(row_size * (size_t) m);
    const size_t written = ggml_quantize_chunk(type, w.data(), q.data(), 0, m, k, nullptr);
    if (written != q.size()) {
        std::fprintf(stderr, "ggml_quantize_chunk wrote %zu bytes, expected %zu\n", written, q.size());
        std::exit(3);
    }
    return q;
}

static std::vector<uint8_t> quantize_q8_0_rows(const std::vector<float> & w, int m, int k) {
    return quantize_rows(w, m, k, GGML_TYPE_Q8_0);
}

static std::vector<float> cpu_dequant_q_matmul_ref(const std::vector<uint8_t> & q, const std::vector<float> & x, int m, int n, int k, ggml_type type) {
    const int64_t qk = ggml_blck_size(type);
    const size_t block_size = ggml_type_size(type);
    const size_t row_size = ggml_row_size(type, k);
    std::vector<float> out((size_t) m * (size_t) n, 0.0f);

    for (int row = 0; row < m; ++row) {
        const uint8_t * row_q = q.data() + (size_t) row * row_size;
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int blk = 0; blk < k / qk; ++blk) {
                const uint8_t * block = row_q + (size_t) blk * block_size;
                ggml_fp16_t dh;
                std::memcpy(&dh, block, sizeof(dh));
                const float d = ggml_fp16_to_fp32(dh);
                if (type == GGML_TYPE_Q4_0) {
                    const uint8_t * qs = block + sizeof(dh);
                    for (int i = 0; i < qk; ++i) {
                        const uint8_t byte = qs[i & 15];
                        const int qv = i < 16 ? (byte & 0x0f) : (byte >> 4);
                        acc += d * (float) (qv - 8) * x[(size_t) col * (size_t) k + (size_t) blk * (size_t) qk + (size_t) i];
                    }
                } else {
                    const int8_t * qs = reinterpret_cast<const int8_t *>(block + sizeof(dh));
                    for (int i = 0; i < qk; ++i) {
                        acc += d * (float) qs[i] * x[(size_t) col * (size_t) k + (size_t) blk * (size_t) qk + (size_t) i];
                    }
                }
            }
            out[(size_t) col * (size_t) m + (size_t) row] = acc;
        }
    }
    return out;
}

static std::vector<float> cpu_dequant_q8_0_matmul_ref(const std::vector<uint8_t> & q, const std::vector<float> & x, int m, int n, int k) {
    return cpu_dequant_q_matmul_ref(q, x, m, n, k, GGML_TYPE_Q8_0);
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float v = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        v = std::max(v, std::fabs(a[i] - b[i]));
    }
    return v;
}

static float mean_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return (float) (sum / std::max<size_t>(a.size(), 1));
}

static bool has_nonfinite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return true;
        }
    }
    return false;
}

static void set_route_env(const char * route) {
    setenv("GGML_CUDA_DP16_ROUTE_REQUIRE", route, 1);
    if (std::strcmp(route, "rocm_packed16_dot4_mmvq") == 0) {
        setenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ", "1", 1);
        unsetenv("GGML_CUDA_ROCM_Q8_DOT4_MMVQ");
        unsetenv("GGML_CUDA_ROCM_Q4_0_PACKED16_DOT4_MMVQ");
    } else if (std::strcmp(route, "rocm_q4_0_packed16_dot4_mmvq") == 0) {
        setenv("GGML_CUDA_ROCM_Q4_0_PACKED16_DOT4_MMVQ", "1", 1);
        unsetenv("GGML_CUDA_ROCM_Q8_DOT4_MMVQ");
        unsetenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ");
    } else {
        setenv("GGML_CUDA_ROCM_Q8_DOT4_MMVQ", "1", 1);
        unsetenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ");
        unsetenv("GGML_CUDA_ROCM_Q4_0_PACKED16_DOT4_MMVQ");
    }
}

static run_result run_gpu_mul_mat(
        const char * route,
        const std::vector<uint8_t> & q_weight,
        const std::vector<float> & x,
        int m,
        int n,
        int k,
        ggml_type weight_type = GGML_TYPE_Q8_0) {
    run_result rr;

    set_route_env(route);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        rr.error = "no GPU backend";
        return rr;
    }

    const size_t meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params wparams = { meta, nullptr, true };
    ggml_context * wctx = ggml_init(wparams);
    ggml_init_params cparams = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(cparams);
    if (!wctx || !ctx) {
        rr.error = "ggml_init failed";
        if (ctx)  ggml_free(ctx);
        if (wctx) ggml_free(wctx);
        ggml_backend_free(backend);
        return rr;
    }

    ggml_tensor * A = ggml_new_tensor_2d(wctx, weight_type, k, m);
    ggml_set_name(A, weight_type == GGML_TYPE_Q4_0 ? "dp16_mmvq_A_q4_0_weight" : "dp16_mmvq_A_q8_0_weight");
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_name(B, "dp16_mmvq_B_f32_activation");
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_set_name(C, "dp16_mmvq_C_f32_output");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!wbuf || !cbuf) {
        rr.error = "backend tensor allocation failed";
        if (cbuf) ggml_backend_buffer_free(cbuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return rr;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(cbuf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_backend_tensor_set(A, q_weight.data(), 0, q_weight.size());
    ggml_backend_tensor_set(B, x.data(), 0, x.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        rr.error = std::string("ggml_backend_graph_compute failed: ") + ggml_status_to_string(status);
        ggml_backend_buffer_free(cbuf);
        ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return rr;
    }

    rr.output.resize((size_t) ggml_nelements(C));
    ggml_backend_tensor_get(C, rr.output.data(), 0, rr.output.size() * sizeof(float));
    rr.ok = true;

    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return rr;
}

static run_result run_gpu_mul_mat_repeated(
        const char * route,
        const std::vector<uint8_t> & q_weight,
        const std::vector<float> & x,
        int m,
        int n,
        int k,
        ggml_type weight_type,
        int warmup,
        int repeat,
        const std::vector<uint8_t> * mutate_to) {
    run_result rr;
    set_route_env(route);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        rr.error = "no GPU backend";
        return rr;
    }

    const size_t meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params wparams = { meta, nullptr, true };
    ggml_context * wctx = ggml_init(wparams);
    ggml_init_params cparams = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(cparams);
    if (!wctx || !ctx) {
        rr.error = "ggml_init failed";
        if (ctx)  ggml_free(ctx);
        if (wctx) ggml_free(wctx);
        ggml_backend_free(backend);
        return rr;
    }

    ggml_tensor * A = ggml_new_tensor_2d(wctx, weight_type, k, m);
    ggml_set_name(A, weight_type == GGML_TYPE_Q4_0 ? "dp16_mmvq_A_q4_0_weight_repeat" : "dp16_mmvq_A_q8_0_weight_repeat");
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_name(B, "dp16_mmvq_B_f32_activation_repeat");
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_set_name(C, "dp16_mmvq_C_f32_output_repeat");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!wbuf || !cbuf) {
        rr.error = "backend tensor allocation failed";
        if (cbuf) ggml_backend_buffer_free(cbuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        ggml_free(ctx);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return rr;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(cbuf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_backend_tensor_set(A, q_weight.data(), 0, q_weight.size());
    ggml_backend_tensor_set(B, x.data(), 0, x.size() * sizeof(float));

    const int total = std::max(1, warmup + repeat);
    for (int i = 0; i < total; ++i) {
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            rr.error = std::string("ggml_backend_graph_compute failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(cbuf);
            ggml_backend_buffer_free(wbuf);
            ggml_free(ctx);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return rr;
        }
    }

    if (mutate_to) {
        ggml_backend_tensor_set(A, mutate_to->data(), 0, mutate_to->size());
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            rr.error = std::string("ggml_backend_graph_compute after mutation failed: ") + ggml_status_to_string(status);
            ggml_backend_buffer_free(cbuf);
            ggml_backend_buffer_free(wbuf);
            ggml_free(ctx);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return rr;
        }
    }

    rr.output.resize((size_t) ggml_nelements(C));
    ggml_backend_tensor_get(C, rr.output.data(), 0, rr.output.size() * sizeof(float));
    rr.ok = true;

    ggml_backend_buffer_free(cbuf);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return rr;
}

int main(int argc, char ** argv) {
    int m = 16;
    std::vector<int> ns = {1, 2, 3, 4};
    std::vector<int> ks = {256, 512};
    float tolerance = 1.0e-3f;
    std::string source = "q8_0";
    int warmup = 0;
    int repeat = 1;
    bool mutation_check = false;
    const bool allow_invalid = getenv("DP16_MMVQ_HARNESS_ALLOW_INVALID") && atoi(getenv("DP16_MMVQ_HARNESS_ALLOW_INVALID")) != 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--m" && i + 1 < argc) {
            m = parse_list(argv[++i]).at(0);
        } else if (arg == "--n" && i + 1 < argc) {
            ns = parse_list(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            ks = parse_list(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            tolerance = std::strtof(argv[++i], nullptr);
        } else if (arg == "--source" && i + 1 < argc) {
            source = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = parse_list(argv[++i]).at(0);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = parse_list(argv[++i]).at(0);
        } else if (arg == "--mutation-check") {
            mutation_check = true;
        } else if (arg == "--help") {
            std::printf("usage: %s [--m 16] [--n 1,2,3,4] [--k 256,512] [--tol 1e-3] [--source q8_0] [--warmup 0] [--repeat 1] [--mutation-check]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg.c_str());
            return 2;
        }
    }

    setenv("GGML_CUDA_DP16_TRACE", getenv("GGML_CUDA_DP16_TRACE") ? getenv("GGML_CUDA_DP16_TRACE") : "1", 1);
    ggml_backend_load_all();

    if (source != "q8_0" && source != "q4_0") {
        std::fprintf(stderr, "unsupported --source '%s' (supported: q8_0,q4_0)\n", source.c_str());
        return 2;
    }
    const ggml_type weight_type = source == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    const char * packed_route = weight_type == GGML_TYPE_Q4_0 ? "rocm_q4_0_packed16_dot4_mmvq" : "rocm_packed16_dot4_mmvq";

    std::printf("DP16 packed16 MMVQ harness: m=%d tol=%g source=%s warmup=%d repeat=%d mutation=%d\n",
            m, tolerance, source.c_str(), warmup, repeat, mutation_check ? 1 : 0);

    int failures = 0;
    for (int k : ks) {
        if (!allow_invalid && k % 256 != 0) {
            std::fprintf(stderr, "k=%d is not 256-aligned; this harness expects valid packed16 MMVQ shapes\n", k);
            ++failures;
            continue;
        }
        std::vector<float> weights;
        fill_weights(weights, m, k);
        const std::vector<uint8_t> q_weight = quantize_rows(weights, m, k, weight_type);

        for (int n : ns) {
            if (!allow_invalid && (n <= 0 || n > 4)) {
                std::fprintf(stderr, "n=%d is outside packed16 MMVQ decode range 1..4\n", n);
                ++failures;
                continue;
            }

            std::vector<float> x;
            fill_activations(x, k, n);
            const std::vector<float> cpu_ref = cpu_dequant_q_matmul_ref(q_weight, x, m, n, k, weight_type);

            std::printf("case m=%d n=%d k=%d\n", m, n, k);
            run_result q8;
            if (weight_type == GGML_TYPE_Q8_0) {
                q8 = run_gpu_mul_mat("rocm_q8_dot4_mmvq", q_weight, x, m, n, k, weight_type);
                if (!q8.ok) {
                    std::fprintf(stderr, "  q8 route failed: %s\n", q8.error.c_str());
                    ++failures;
                    continue;
                }
            }
            run_result packed16 = run_gpu_mul_mat(packed_route, q_weight, x, m, n, k, weight_type);
            if (!packed16.ok) {
                std::fprintf(stderr, "  packed16 route failed: %s\n", packed16.error.c_str());
                ++failures;
                continue;
            }
            if ((weight_type == GGML_TYPE_Q8_0 && has_nonfinite(q8.output)) || has_nonfinite(packed16.output)) {
                std::fprintf(stderr, "  non-finite output detected\n");
                ++failures;
                continue;
            }

            const float max_p16_cpu = max_abs_diff(packed16.output, cpu_ref);
            if (weight_type == GGML_TYPE_Q8_0) {
                const float max_q8_p16 = max_abs_diff(q8.output, packed16.output);
                const float mean_q8_p16 = mean_abs_diff(q8.output, packed16.output);
                const float max_q8_cpu = max_abs_diff(q8.output, cpu_ref);
                std::printf("  q8_vs_packed16 max=%g mean=%g | q8_vs_dequant_f32=%g packed16_vs_dequant_f32=%g\n",
                        max_q8_p16, mean_q8_p16, max_q8_cpu, max_p16_cpu);
                if (max_q8_p16 > tolerance) {
                    std::fprintf(stderr, "  FAIL: q8 vs packed16 diff %g > tol %g\n", max_q8_p16, tolerance);
                    ++failures;
                }
            } else {
                std::printf("  q4_0_packed16_vs_dequant_f32 max=%g\n", max_p16_cpu);
                if (max_p16_cpu > tolerance) {
                    std::fprintf(stderr, "  FAIL: q4_0 packed16 diff %g > tol %g\n", max_p16_cpu, tolerance);
                    ++failures;
                }
            }

            if (warmup > 0 || repeat > 1) {
                run_result steady = run_gpu_mul_mat_repeated(packed_route, q_weight, x, m, n, k, weight_type, warmup, repeat, nullptr);
                if (!steady.ok) {
                    std::fprintf(stderr, "  steady packed16 route failed: %s\n", steady.error.c_str());
                    ++failures;
                } else {
                    const float max_steady = weight_type == GGML_TYPE_Q8_0 ? max_abs_diff(steady.output, q8.output) : max_abs_diff(steady.output, cpu_ref);
                    std::printf("  steady_packed16_vs_%s max=%g warmup=%d repeat=%d\n", weight_type == GGML_TYPE_Q8_0 ? "q8" : "dequant_f32", max_steady, warmup, repeat);
                    if (max_steady > tolerance) {
                        std::fprintf(stderr, "  FAIL: steady packed16 diff %g > tol %g\n", max_steady, tolerance);
                        ++failures;
                    }
                }
            }

            if (mutation_check && weight_type == GGML_TYPE_Q8_0) {
                std::vector<float> weights2;
                fill_weights(weights2, m, k, 101);
                const std::vector<uint8_t> q_weight2 = quantize_q8_0_rows(weights2, m, k);
                run_result q8_mutated = run_gpu_mul_mat("rocm_q8_dot4_mmvq", q_weight2, x, m, n, k);
                run_result mutated = run_gpu_mul_mat_repeated("rocm_packed16_dot4_mmvq", q_weight, x, m, n, k, weight_type, 0, 1, &q_weight2);
                if (!q8_mutated.ok) {
                    std::fprintf(stderr, "  mutation q8 route failed: %s\n", q8_mutated.error.c_str());
                    ++failures;
                } else if (!mutated.ok) {
                    std::fprintf(stderr, "  mutation packed16 route failed: %s\n", mutated.error.c_str());
                    ++failures;
                } else {
                    const float max_mut_q8 = max_abs_diff(mutated.output, q8_mutated.output);
                    std::printf("  mutated_packed16_vs_q8 max=%g\n", max_mut_q8);
                    if (max_mut_q8 > tolerance) {
                        std::fprintf(stderr, "  FAIL: mutated packed16 diff %g > tol %g\n", max_mut_q8, tolerance);
                        ++failures;
                    }
                }
            }
        }
    }

    ggml_quantize_free();
    if (failures != 0) {
        std::fprintf(stderr, "DP16 packed16 MMVQ harness failed: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("DP16 packed16 MMVQ harness passed\n");
    return 0;
}
