// rdna-i8-packed16-moe-projection-q8-route-smoke.cpp
// Test-only smoke for the default-off internal packed16 WMMA path inside
// GGML_OP_MOE_ROUTED_LANES_PROJECTION with Q8_0 expert weights.

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct q8_0_host {
    ggml_fp16_t d;
    int8_t qs[32];
};
static_assert(sizeof(q8_0_host) == 34, "q8_0_host must match block_q8_0 byte layout");

static int8_t lane_i8(int32_t w, int lane) {
    return (int8_t) (((uint32_t) w >> (8 * lane)) & 0xffu);
}

static int32_t pack4_host(int a, int b, int c, int d) {
    uint32_t w = 0;
    const int vals[4] = {a, b, c, d};
    for (int i = 0; i < 4; ++i) {
        w |= ((uint32_t) (uint8_t) (int8_t) vals[i]) << (8 * i);
    }
    return (int32_t) w;
}

static void pack_f32_rows_to_packed16_ref(
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
                    int qi = (int) roundf(src[(size_t) row * k + kb * 32 + w * 4 + lane] / scale);
                    if (qi < -127) qi = -127;
                    if (qi >  127) qi =  127;
                    q[lane] = qi;
                }
                payload[(size_t) row * k_words + kb * 8 + w] = pack4_host(q[0], q[1], q[2], q[3]);
            }
        }
    }
}

static void pack_q8_weights_ref(
        const std::vector<q8_0_host> & src,
        int k_blocks,
        int n_out,
        int n_expert,
        std::vector<int32_t> & payload,
        std::vector<ggml_fp16_t> & scales) {
    const int k_words = k_blocks * 8;
    payload.assign((size_t) n_expert * n_out * k_words, 0);
    scales.assign((size_t) n_expert * n_out * k_blocks, ggml_fp32_to_fp16(1.0f));
    for (int e = 0; e < n_expert; ++e) {
        for (int out = 0; out < n_out; ++out) {
            for (int kb = 0; kb < k_blocks; ++kb) {
                const q8_0_host & block = src[((size_t) e * n_out + out) * k_blocks + kb];
                scales[((size_t) e * n_out + out) * k_blocks + kb] = block.d;
                for (int w = 0; w < 8; ++w) {
                    payload[((size_t) e * n_out + out) * k_words + kb * 8 + w] = pack4_host(
                            block.qs[w * 4 + 0], block.qs[w * 4 + 1], block.qs[w * 4 + 2], block.qs[w * 4 + 3]);
                }
            }
        }
    }
}

int main() {
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_Q8_0", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_MAX_TEMP_BYTES", "104857600", 1);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        fprintf(stderr, "no GPU backend\n");
        return 77;
    }

    constexpr int K_BLOCKS = 2;
    constexpr int K_WORDS = K_BLOCKS * 8;
    constexpr int K = K_WORDS * 4;
    constexpr int N_OUT = 16;
    constexpr int N_EXPERT = 3;
    constexpr int N_LANES = 7;
    constexpr int N_SLOTS = 1;
    constexpr int N_ROWS = N_LANES;

    const size_t meta = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 1024 * 1024;
    ggml_init_params params = { meta, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 2;

    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_SLOTS, N_ROWS);
    ggml_tensor * route_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_SLOTS, N_ROWS);
    ggml_tensor * W        = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, K, N_OUT, N_EXPERT);
    ggml_tensor * compact  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N_LANES);
    ggml_set_name(selected, "packed16_moe_proj_q8_selected");
    ggml_set_name(route_w, "packed16_moe_proj_q8_route_weights");
    ggml_set_name(W, "packed16_moe_proj_q8_weight");
    ggml_set_name(compact, "packed16_moe_proj_q8_compact");

    ggml_tensor * lanes = ggml_moe_routed_lanes(ctx, selected, route_w, N_EXPERT);
    ggml_set_name(lanes, "packed16_moe_proj_q8_lanes");
    ggml_tensor * bounds = ggml_moe_routed_lanes_expert_bounds(ctx, lanes);
    ggml_set_name(bounds, "packed16_moe_proj_q8_bounds");
    ggml_tensor * out = ggml_moe_routed_lanes_projection(ctx, W, compact, lanes, bounds);
    ggml_set_name(out, "packed16_moe_proj_q8_output");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed\n");
        return 3;
    }

    const int hSelected[N_ROWS] = {0, 0, 1, 1, 1, 2, 2};
    std::vector<float> hRoute((size_t) N_ROWS, 1.0f);
    ggml_backend_tensor_set(selected, hSelected, 0, sizeof(hSelected));
    ggml_backend_tensor_set(route_w, hRoute.data(), 0, hRoute.size() * sizeof(float));

    std::vector<q8_0_host> hQ8((size_t) N_EXPERT * N_OUT * K_BLOCKS);
    for (int e = 0; e < N_EXPERT; ++e) {
        for (int row = 0; row < N_OUT; ++row) {
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                q8_0_host & block = hQ8[((size_t)e*N_OUT + row)*K_BLOCKS + kb];
                block.d = ggml_fp32_to_fp16(0.01f * (1 + e + row + kb));
                for (int i = 0; i < 32; ++i) {
                    block.qs[i] = (int8_t) (((e * 29 + row * 17 + kb * 11 + i * 7) % 23) - 11);
                }
            }
        }
    }
    ggml_backend_tensor_set(W, hQ8.data(), 0, hQ8.size() * sizeof(q8_0_host));

    std::vector<float> hCompact((size_t) N_LANES * K);
    for (int lane = 0; lane < N_LANES; ++lane) {
        for (int k = 0; k < K; ++k) {
            float x = 0.125f * (float) (((lane * 37 + k * 13) % 31) - 15);
            if (((lane + k) % 23) == 0) x = 0.0f;
            if (((lane * 7 + k) % 29) == 0) x = 12.7f;
            if (((lane * 11 + k) % 31) == 0) x = -12.8f;
            hCompact[(size_t) lane*K + k] = x;
        }
    }
    ggml_backend_tensor_set(compact, hCompact.data(), 0, hCompact.size() * sizeof(float));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(st));
        return 4;
    }

    std::vector<float> hOut((size_t) N_LANES * N_OUT);
    ggml_backend_tensor_get(out, hOut.data(), 0, hOut.size() * sizeof(float));

    std::vector<int32_t> hW_payload;
    std::vector<ggml_fp16_t> hW_scales;
    std::vector<int32_t> hA_payload;
    std::vector<ggml_fp16_t> hA_scales;
    pack_q8_weights_ref(hQ8, K_BLOCKS, N_OUT, N_EXPERT, hW_payload, hW_scales);
    pack_f32_rows_to_packed16_ref(hCompact, N_LANES, K_WORDS, hA_payload, hA_scales);

    float max_abs = 0.0f;
    for (int lane = 0; lane < N_LANES; ++lane) {
        const int e = hSelected[lane];
        for (int row = 0; row < N_OUT; ++row) {
            float ref = 0.0f;
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                int dot = 0;
                for (int w = 0; w < 8; ++w) {
                    const int32_t aw = hA_payload[(size_t)lane*K_WORDS + kb*8 + w];
                    const int32_t ww = hW_payload[((size_t)e*N_OUT + row)*K_WORDS + kb*8 + w];
                    for (int l = 0; l < 4; ++l) dot += (int)lane_i8(aw,l) * (int)lane_i8(ww,l);
                }
                ref += (float) dot * ggml_fp16_to_fp32(hA_scales[(size_t)lane*K_BLOCKS+kb]) *
                                  ggml_fp16_to_fp32(hW_scales[((size_t)e*N_OUT + row)*K_BLOCKS+kb]);
            }
            const float got = hOut[(size_t)lane*N_OUT + row];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                fprintf(stderr, "mismatch lane=%d row=%d expert=%d got=%g ref=%g diff=%g\n", lane, row, e, got, ref, got-ref);
                return 5;
            }
        }
    }

    printf("PASS moe_projection_q8_packed16_route max_abs=%g\n", max_abs);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
