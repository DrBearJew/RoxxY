// rdna-i8-packed16-sidecar-smoke.cpp
// Repo-controlled ggml backend smoke for private packed16 MUL_MAT_ID sidecars.
// Builds as a standalone host test against an existing ROCm/ggml build.
// Routes are default-off and enabled only inside this smoke via environment vars.

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int8_t lane_i8(int32_t w, int lane) {
    return (int8_t) (((uint32_t) w >> (8 * lane)) & 0xffu);
}

static int32_t pack4(int a, int b, int c, int d) {
    uint32_t w = 0;
    const int vals[4] = {a, b, c, d};
    for (int i = 0; i < 4; ++i) {
        w |= ((uint32_t) (uint8_t) (int8_t) vals[i]) << (8 * i);
    }
    return (int32_t) w;
}

static void pack_f32_rows_to_packed16(
        const std::vector<float> & src,
        int rows,
        int k_words,
        std::vector<int32_t> & payload,
        std::vector<ggml_fp16_t> & scales) {
    const int k = k_words * 4;
    const int k_blocks = k_words / 8;
    payload.assign((size_t) rows * k_words, 0);
    scales.assign((size_t) rows * k_blocks, ggml_fp32_to_fp16(1.0f));

    for (int row = 0; row < rows; ++row) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(src[(size_t) row * k + kb * 32 + i]));
            }
            const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            scales[(size_t) row * k_blocks + kb] = ggml_fp32_to_fp16(scale);
            for (int w = 0; w < 8; ++w) {
                int q[4];
                for (int lane = 0; lane < 4; ++lane) {
                    const float x = src[(size_t) row * k + kb * 32 + w * 4 + lane];
                    int qi = (int) lrintf(x / scale);
                    if (qi < -127) qi = -127;
                    if (qi >  127) qi =  127;
                    q[lane] = qi;
                }
                payload[(size_t) row * k_words + kb * 8 + w] = pack4(q[0], q[1], q[2], q[3]);
            }
        }
    }
}

static int run_dense_f32pack_case(ggml_backend_t backend) {
    constexpr int K_WORDS = 16; // 64 int8 lanes, 2 K32 blocks
    constexpr int K_BLOCKS = K_WORDS / 8;
    constexpr int K = K_WORDS * 4;
    constexpr int M_ACT = 17;
    constexpr int N_OUT = 19;

    const size_t meta = ggml_tensor_overhead() * 10 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for dense f32pack case\n");
        return 2;
    }

    ggml_tensor * W  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K_WORDS, N_OUT);
    ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K_WORDS, M_ACT);
    ggml_tensor * WS = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K_BLOCKS, N_OUT);
    ggml_tensor * AS = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K_BLOCKS, M_ACT);
    ggml_set_name(W, "packed16_f32pack_weight_payload");
    ggml_set_name(A, "packed16_f32pack_act_payload");
    ggml_set_name(WS, "packed16_f32pack_weight_scales");
    ggml_set_name(AS, "packed16_f32pack_act_scales");
    W->src[1] = WS;
    A->src[1] = AS;

    ggml_tensor * C = ggml_mul_mat(ctx, W, A);
    ggml_set_name(C, "packed16_f32pack_output");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "alloc failed for dense f32pack case\n");
        ggml_free(ctx);
        return 3;
    }

    std::vector<int32_t> hW((size_t) N_OUT * K_WORDS);
    std::vector<ggml_fp16_t> hWS((size_t) N_OUT * K_BLOCKS);
    for (int n = 0; n < N_OUT; ++n) {
        for (int w = 0; w < K_WORDS; ++w) {
            hW[(size_t)n*K_WORDS+w] = pack4((n+w)%7-3, (2*n+w)%9-4, (n+3*w)%11-5, (3*n+2*w)%13-6);
        }
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            hWS[(size_t)n*K_BLOCKS+kb] = ggml_fp32_to_fp16(0.01f * (1 + n + kb));
        }
    }

    std::vector<float> hA_f32((size_t) M_ACT * K);
    for (int m = 0; m < M_ACT; ++m) {
        for (int k = 0; k < K; ++k) {
            float x = 0.125f * (float) (((m * 37 + k * 13) % 31) - 15);
            if (((m + k) % 23) == 0) x = 0.0f;
            if (((m * 7 + k) % 29) == 0) x = 12.7f;
            if (((m * 11 + k) % 31) == 0) x = -12.8f;
            hA_f32[(size_t)m*K+k] = x;
        }
    }
    std::vector<int32_t> hA;
    std::vector<ggml_fp16_t> hAS;
    pack_f32_rows_to_packed16(hA_f32, M_ACT, K_WORDS, hA, hAS);

    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(int32_t));
    ggml_backend_tensor_set(A, hA.data(), 0, hA.size()*sizeof(int32_t));
    ggml_backend_tensor_set(WS, hWS.data(), 0, hWS.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(AS, hAS.data(), 0, hAS.size()*sizeof(ggml_fp16_t));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dense f32pack compute failed: %s\n", ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 4;
    }

    std::vector<float> out((size_t) M_ACT * N_OUT);
    ggml_backend_tensor_get(C, out.data(), 0, out.size()*sizeof(float));

    float max_abs = 0.0f;
    for (int m = 0; m < M_ACT; ++m) {
        for (int n = 0; n < N_OUT; ++n) {
            float ref = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                int dot = 0;
                for (int w = 0; w < 8; ++w) {
                    const int32_t aw = hA[(size_t)m*K_WORDS + kb*8 + w];
                    const int32_t ww = hW[(size_t)n*K_WORDS + kb*8 + w];
                    for (int lane = 0; lane < 4; ++lane) {
                        dot += (int) lane_i8(aw,lane) * (int) lane_i8(ww,lane);
                    }
                }
                ref += (float) dot * ggml_fp16_to_fp32(hAS[(size_t)m*K_BLOCKS+kb]) * ggml_fp16_to_fp32(hWS[(size_t)n*K_BLOCKS+kb]);
            }
            const float got = out[(size_t)m*N_OUT+n];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                std::fprintf(stderr, "dense f32pack mismatch m=%d n=%d got=%g ref=%g diff=%g\n", m, n, got, ref, got-ref);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                return 5;
            }
        }
    }

    std::printf("PASS dense_f32pack max_abs=%g\n", max_abs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

static int run_id_scalar_case(ggml_backend_t backend) {
    constexpr int K_WORDS = 16; // 64 int8 lanes, 2 K32 blocks
    constexpr int K_BLOCKS = K_WORDS / 8;
    constexpr int N_OUT = 16;
    constexpr int N_EXPERT = 3;
    constexpr int N_SRC1_SLOTS = 2;
    constexpr int N_IDS = 2;
    constexpr int N_TOKENS = 4;

    const size_t meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for scalar sidecar case\n");
        return 2;
    }

    ggml_tensor * W  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, N_OUT, N_EXPERT);
    ggml_tensor * A  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, N_SRC1_SLOTS, N_TOKENS);
    ggml_tensor * WS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, N_OUT, N_EXPERT);
    ggml_tensor * AS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, N_SRC1_SLOTS, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_IDS, N_TOKENS);
    ggml_set_name(W, "packed16_id_weight_payload");
    ggml_set_name(A, "packed16_id_act_payload");
    ggml_set_name(WS, "packed16_id_weight_scales");
    ggml_set_name(AS, "packed16_id_act_scales");
    ggml_set_name(ids, "packed16_id_ids");

    ggml_tensor * C = ggml_mul_mat_id(ctx, W, A, ids);
    ggml_set_name(C, "packed16_id_output");
    C->src[3] = WS;
    C->src[4] = AS;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "alloc failed for scalar sidecar case\n");
        ggml_free(ctx);
        return 3;
    }

    std::vector<int32_t> hW((size_t) N_EXPERT * N_OUT * K_WORDS);
    std::vector<int32_t> hA((size_t) N_TOKENS * N_SRC1_SLOTS * K_WORDS);
    std::vector<ggml_fp16_t> hWS((size_t) N_EXPERT * N_OUT * K_BLOCKS);
    std::vector<ggml_fp16_t> hAS((size_t) N_TOKENS * N_SRC1_SLOTS * K_BLOCKS);
    std::vector<int32_t> hIds((size_t) N_TOKENS * N_IDS);

    auto w_index = [](int k, int out, int expert) { return (size_t) expert*N_OUT*K_WORDS + (size_t) out*K_WORDS + k; };
    auto a_index = [](int k, int slot, int token) { return (size_t) token*N_SRC1_SLOTS*K_WORDS + (size_t) slot*K_WORDS + k; };
    auto ws_index = [](int kb, int out, int expert) { return (size_t) expert*N_OUT*K_BLOCKS + (size_t) out*K_BLOCKS + kb; };
    auto as_index = [](int kb, int slot, int token) { return (size_t) token*N_SRC1_SLOTS*K_BLOCKS + (size_t) slot*K_BLOCKS + kb; };
    auto id_index = [](int slot, int token) { return (size_t) token*N_IDS + slot; };

    for (int e = 0; e < N_EXPERT; ++e) {
        for (int out = 0; out < N_OUT; ++out) {
            for (int w = 0; w < K_WORDS; ++w) {
                hW[w_index(w,out,e)] = pack4((e+out+w)%7-3, (2*e+out+w)%9-4, (e+out+3*w)%11-5, (3*e+2*out+w)%13-6);
            }
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                hWS[ws_index(kb,out,e)] = ggml_fp32_to_fp16(0.01f * (1 + e + out + kb));
            }
        }
    }
    for (int t = 0; t < N_TOKENS; ++t) {
        for (int slot = 0; slot < N_SRC1_SLOTS; ++slot) {
            for (int w = 0; w < K_WORDS; ++w) {
                hA[a_index(w,slot,t)] = pack4((t+slot+2*w)%7-3, (2*t+slot+3*w)%9-4, (t+slot+w)%11-5, (5*t+slot+w)%13-6);
            }
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                hAS[as_index(kb,slot,t)] = ggml_fp32_to_fp16(0.02f * (1 + t + slot + kb));
            }
        }
        for (int slot = 0; slot < N_IDS; ++slot) {
            hIds[id_index(slot,t)] = (slot + 2*t) % N_EXPERT;
        }
    }

    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(int32_t));
    ggml_backend_tensor_set(A, hA.data(), 0, hA.size()*sizeof(int32_t));
    ggml_backend_tensor_set(WS, hWS.data(), 0, hWS.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(AS, hAS.data(), 0, hAS.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(ids, hIds.data(), 0, hIds.size()*sizeof(int32_t));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "scalar sidecar compute failed: %s\n", ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 4;
    }

    std::vector<float> out((size_t) N_TOKENS * N_IDS * N_OUT);
    ggml_backend_tensor_get(C, out.data(), 0, out.size()*sizeof(float));

    float max_abs = 0.0f;
    for (int t = 0; t < N_TOKENS; ++t) {
        for (int slot = 0; slot < N_IDS; ++slot) {
            const int e = hIds[id_index(slot,t)];
            const int act_slot = slot % N_SRC1_SLOTS;
            for (int row = 0; row < N_OUT; ++row) {
                float ref = 0.0f;
                for (int kb = 0; kb < K_BLOCKS; ++kb) {
                    int dot = 0;
                    for (int w = 0; w < 8; ++w) {
                        const int32_t aw = hA[a_index(kb*8+w,act_slot,t)];
                        const int32_t ww = hW[w_index(kb*8+w,row,e)];
                        for (int lane = 0; lane < 4; ++lane) {
                            dot += (int) lane_i8(aw,lane) * (int) lane_i8(ww,lane);
                        }
                    }
                    ref += (float) dot * ggml_fp16_to_fp32(hAS[as_index(kb,act_slot,t)]) * ggml_fp16_to_fp32(hWS[ws_index(kb,row,e)]);
                }
                const float got = out[(size_t)t*N_IDS*N_OUT + (size_t)slot*N_OUT + row];
                max_abs = fmaxf(max_abs, fabsf(got - ref));
                if (fabsf(got - ref) > 1e-4f) {
                    std::fprintf(stderr, "scalar mismatch t=%d slot=%d row=%d e=%d got=%g ref=%g diff=%g\n", t, slot, row, e, got, ref, got-ref);
                    ggml_backend_buffer_free(buf);
                    ggml_free(ctx);
                    return 5;
                }
            }
        }
    }

    std::printf("PASS id_scalar max_abs=%g\n", max_abs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

static int run_id_bounds_case(ggml_backend_t backend) {
    constexpr int K_WORDS = 16;
    constexpr int K_BLOCKS = K_WORDS / 8;
    constexpr int N_OUT = 16;
    constexpr int N_EXPERT = 3;
    constexpr int N_LANES = 7;

    const size_t meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for bounds sidecar case\n");
        return 2;
    }

    ggml_tensor * W  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, N_OUT, N_EXPERT);
    ggml_tensor * A  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, K_WORDS, 1, N_LANES);
    ggml_tensor * WS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, N_OUT, N_EXPERT);
    ggml_tensor * AS = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K_BLOCKS, 1, N_LANES);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, N_LANES);
    ggml_tensor * bounds = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, N_EXPERT);
    ggml_set_name(W, "packed16_bounds_weight_payload");
    ggml_set_name(A, "packed16_bounds_act_payload");
    ggml_set_name(WS, "packed16_bounds_weight_scales");
    ggml_set_name(AS, "packed16_bounds_act_scales");
    ggml_set_name(ids, "packed16_bounds_ids");
    ggml_set_name(bounds, "packed16_bounds_expert_bounds");

    ggml_tensor * C = ggml_mul_mat_id(ctx, W, A, ids);
    ggml_set_name(C, "packed16_bounds_output");
    C->src[3] = WS;
    C->src[4] = AS;
    C->src[5] = bounds;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "alloc failed for bounds sidecar case\n");
        ggml_free(ctx);
        return 3;
    }

    std::vector<int32_t> hW((size_t) N_EXPERT * N_OUT * K_WORDS);
    std::vector<int32_t> hA((size_t) N_LANES * K_WORDS);
    std::vector<ggml_fp16_t> hWS((size_t) N_EXPERT * N_OUT * K_BLOCKS);
    std::vector<ggml_fp16_t> hAS((size_t) N_LANES * K_BLOCKS);
    std::vector<int32_t> hIds((size_t) N_LANES);
    std::vector<int32_t> hBounds((size_t) N_EXPERT * 2);

    auto w_index = [](int k, int out, int expert) { return (size_t) expert*N_OUT*K_WORDS + (size_t) out*K_WORDS + k; };
    auto a_index = [](int k, int lane) { return (size_t) lane*K_WORDS + k; };
    auto ws_index = [](int kb, int out, int expert) { return (size_t) expert*N_OUT*K_BLOCKS + (size_t) out*K_BLOCKS + kb; };
    auto as_index = [](int kb, int lane) { return (size_t) lane*K_BLOCKS + kb; };
    auto bounds_index = [](int row, int expert) { return (size_t) expert*2 + row; };

    const int starts[N_EXPERT] = {0, 2, 5};
    const int counts[N_EXPERT] = {2, 3, 2};
    for (int e = 0; e < N_EXPERT; ++e) {
        hBounds[bounds_index(0,e)] = starts[e];
        hBounds[bounds_index(1,e)] = counts[e];
        for (int lane = starts[e]; lane < starts[e] + counts[e]; ++lane) {
            hIds[lane] = e;
        }
    }

    for (int e = 0; e < N_EXPERT; ++e) {
        for (int out = 0; out < N_OUT; ++out) {
            for (int w = 0; w < K_WORDS; ++w) {
                hW[w_index(w,out,e)] = pack4((e+out+w)%7-3, (2*e+out+w)%9-4, (e+out+3*w)%11-5, (3*e+2*out+w)%13-6);
            }
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                hWS[ws_index(kb,out,e)] = ggml_fp32_to_fp16(0.01f * (1 + e + out + kb));
            }
        }
    }
    for (int lane = 0; lane < N_LANES; ++lane) {
        for (int w = 0; w < K_WORDS; ++w) {
            hA[a_index(w,lane)] = pack4((lane+2*w)%7-3, (2*lane+3*w)%9-4, (lane+w)%11-5, (5*lane+w)%13-6);
        }
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            hAS[as_index(kb,lane)] = ggml_fp32_to_fp16(0.02f * (1 + lane + kb));
        }
    }

    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(int32_t));
    ggml_backend_tensor_set(A, hA.data(), 0, hA.size()*sizeof(int32_t));
    ggml_backend_tensor_set(WS, hWS.data(), 0, hWS.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(AS, hAS.data(), 0, hAS.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(ids, hIds.data(), 0, hIds.size()*sizeof(int32_t));
    ggml_backend_tensor_set(bounds, hBounds.data(), 0, hBounds.size()*sizeof(int32_t));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "bounds sidecar compute failed: %s\n", ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 4;
    }

    std::vector<float> out((size_t) N_LANES * N_OUT);
    ggml_backend_tensor_get(C, out.data(), 0, out.size()*sizeof(float));

    float max_abs = 0.0f;
    for (int lane = 0; lane < N_LANES; ++lane) {
        const int e = hIds[lane];
        for (int row = 0; row < N_OUT; ++row) {
            float ref = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                int dot = 0;
                for (int w = 0; w < 8; ++w) {
                    const int32_t aw = hA[a_index(kb*8+w,lane)];
                    const int32_t ww = hW[w_index(kb*8+w,row,e)];
                    for (int l = 0; l < 4; ++l) {
                        dot += (int) lane_i8(aw,l) * (int) lane_i8(ww,l);
                    }
                }
                ref += (float) dot * ggml_fp16_to_fp32(hAS[as_index(kb,lane)]) * ggml_fp16_to_fp32(hWS[ws_index(kb,row,e)]);
            }
            const float got = out[(size_t)lane*N_OUT + row];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                std::fprintf(stderr, "bounds mismatch lane=%d row=%d e=%d got=%g ref=%g diff=%g\n", lane, row, e, got, ref, got-ref);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                return 5;
            }
        }
    }

    std::printf("PASS id_bounds max_abs=%g\n", max_abs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

int main() {
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_LOG", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_BOUNDS", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG", "1", 1);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "no GPU backend\n");
        return 77;
    }

    int rc = run_dense_f32pack_case(backend);
    if (rc == 0) {
        rc = run_id_scalar_case(backend);
    }
    if (rc == 0) {
        rc = run_id_bounds_case(backend);
    }

    ggml_backend_free(backend);
    return rc;
}
