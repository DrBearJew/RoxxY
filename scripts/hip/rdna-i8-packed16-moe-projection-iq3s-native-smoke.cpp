// rdna-i8-packed16-moe-projection-iq3s-native-smoke.cpp
// Test-only smoke for IQ3_S-native packed16 WMMA route inside
// GGML_OP_MOE_ROUTED_LANES_PROJECTION. Reference matches vec_dot_iq3_s_q8_1:
// signed iq3s_grid int8 payload, positive odd subscale, F16 d * q8_1.d.

#include <ggml.h>
#include <ggml-backend.h>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int QK_K_HOST = 256;
static_assert(QK_K == QK_K_HOST, "unexpected QK_K");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s layout mismatch");

static int iq3s_scale_host(const block_iq3_s & block, int ib) {
    return 1 + 2*((block.scales[ib/2] >> (4*(ib & 1))) & 0x0f);
}

static void set_iq3s_scale_nibble(block_iq3_s & block, int ib, int nibble) {
    block.scales[ib/2] |= (uint8_t) ((nibble & 0x0f) << (4*(ib & 1)));
}

static int iq3s_signed_q_host(const block_iq3_s & block, int ib, int lane) {
    const int l     = lane >> 3;
    const int lane8 = lane & 7;
    const bool hi   = lane8 >= 4;
    const int lane4 = lane8 & 3;
    const int qsi   = 8*ib + 2*l + (hi ? 1 : 0);
    const int shift = (hi ? 7 : 8) - 2*l;
    const int grid_idx = block.qs[qsi] | ((block.qh[ib] << shift) & 0x100);
    const int mag = (int) ((iq3s_grid[grid_idx] >> (8*lane4)) & 0xffu);
    const uint8_t signs = block.signs[4*ib + l];
    return (signs & kmask_iq2xs[lane8]) ? -mag : mag;
}

static void quantize_q8_1_ref(
        const std::vector<float> & compact,
        int n_lanes,
        int k,
        std::vector<int8_t> & qs,
        std::vector<ggml_fp16_t> & ds) {
    const int k_blocks = k / 32;
    qs.assign((size_t)n_lanes*k, 0);
    ds.assign((size_t)n_lanes*k_blocks, ggml_fp32_to_fp16(1.0f));
    for (int lane = 0; lane < n_lanes; ++lane) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(compact[(size_t)lane*k + kb*32 + i]));
            }
            const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
            ds[(size_t)lane*k_blocks + kb] = ggml_fp32_to_fp16(d);
            for (int i = 0; i < 32; ++i) {
                const float x = compact[(size_t)lane*k + kb*32 + i];
                int qi = (int) roundf(x / d);
                if (qi < -127) qi = -127;
                if (qi >  127) qi =  127;
                qs[(size_t)lane*k + kb*32 + i] = (int8_t) qi;
            }
        }
    }
}

static bool run_case(int k) {
    constexpr int N_OUT = 17;
    constexpr int N_EXPERT = 4;
    constexpr int N_LANES = 5;
    constexpr int N_SLOTS = 1;
    constexpr int N_ROWS = N_LANES;
    const int k_blocks = k / 32;
    const int iq_blocks = k / QK_K_HOST;

    ggml_init_params params = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_SLOTS, N_ROWS);
    ggml_tensor * route_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_SLOTS, N_ROWS);
    ggml_tensor * W        = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_S, k, N_OUT, N_EXPERT);
    ggml_tensor * compact  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, N_LANES);
    ggml_set_name(selected, "packed16_moe_iq3s_native_selected");
    ggml_set_name(route_w, "packed16_moe_iq3s_native_route_weights");
    ggml_set_name(W, "packed16_moe_iq3s_native_weight");
    ggml_set_name(compact, "packed16_moe_iq3s_native_compact");

    ggml_tensor * lanes = ggml_moe_routed_lanes(ctx, selected, route_w, N_EXPERT);
    ggml_set_name(lanes, "packed16_moe_iq3s_native_lanes");
    ggml_tensor * bounds = ggml_moe_routed_lanes_expert_bounds(ctx, lanes);
    ggml_set_name(bounds, "packed16_moe_iq3s_native_bounds");
    ggml_tensor * out = ggml_moe_routed_lanes_projection(ctx, W, compact, lanes, bounds);
    ggml_set_name(out, "packed16_moe_iq3s_native_output");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        fprintf(stderr, "no GPU backend\n");
        return false;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) return false;

    // Sorted lane order, expert 1 intentionally empty: counts [2,0,2,1].
    const int hSelected[N_ROWS] = {0, 0, 2, 2, 3};
    std::vector<float> hRoute((size_t)N_ROWS, 1.0f);
    ggml_backend_tensor_set(selected, hSelected, 0, sizeof(hSelected));
    ggml_backend_tensor_set(route_w, hRoute.data(), 0, hRoute.size()*sizeof(float));

    std::vector<block_iq3_s> hW((size_t)N_EXPERT*N_OUT*iq_blocks);
    const int scale_nibbles[4] = {0, 15, 7, 1};
    const uint8_t sign_patterns[4] = {0x00, 0xff, 0xaa, 0x55};
    for (int e = 0; e < N_EXPERT; ++e) {
        for (int row = 0; row < N_OUT; ++row) {
            for (int qb = 0; qb < iq_blocks; ++qb) {
                block_iq3_s & block = hW[((size_t)e*N_OUT + row)*iq_blocks + qb];
                std::memset(&block, 0, sizeof(block));
                block.d = ggml_fp32_to_fp16(0.00012f * (float)(1 + e + row + qb));
                for (int ib = 0; ib < 8; ++ib) {
                    set_iq3s_scale_nibble(block, ib, scale_nibbles[(e + row + qb + ib) & 3]);
                    for (int word = 0; word < 8; ++word) {
                        const int grid_idx = (e*73 + row*29 + qb*17 + ib*11 + word*37) & 0x1ff;
                        block.qs[8*ib + word] = (uint8_t) (grid_idx & 0xff);
                        if (grid_idx & 0x100) {
                            block.qh[ib] |= (uint8_t) (1u << word);
                        }
                    }
                    for (int l = 0; l < 4; ++l) {
                        block.signs[4*ib + l] = sign_patterns[(e + row + qb + ib + l) & 3];
                    }
                }
            }
        }
    }
    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(block_iq3_s));

    std::vector<float> hCompact((size_t)N_LANES*k);
    for (int lane = 0; lane < N_LANES; ++lane) {
        for (int kk = 0; kk < k; ++kk) {
            float x = 0.03125f * (float)(((lane*37 + kk*13) % 63) - 31);
            if (((lane + kk) % 41) == 0) x = 0.0f;
            if (((lane*7 + kk) % 53) == 0) x = 8.0f;
            if (((lane*11 + kk) % 59) == 0) x = -8.0f;
            hCompact[(size_t)lane*k + kk] = x;
        }
    }
    // Full all-zero K32 activation block.
    for (int i = 0; i < 32; ++i) hCompact[(size_t)4*k + i] = 0.0f;
    ggml_backend_tensor_set(compact, hCompact.data(), 0, hCompact.size()*sizeof(float));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(st));
        return false;
    }

    std::vector<float> hOut((size_t)N_LANES*N_OUT);
    ggml_backend_tensor_get(out, hOut.data(), 0, hOut.size()*sizeof(float));

    std::vector<int8_t> act_qs;
    std::vector<ggml_fp16_t> act_ds;
    quantize_q8_1_ref(hCompact, N_LANES, k, act_qs, act_ds);

    float max_abs = 0.0f;
    for (int lane = 0; lane < N_LANES; ++lane) {
        const int e = hSelected[lane];
        for (int row = 0; row < N_OUT; ++row) {
            float ref = 0.0f;
            for (int kb = 0; kb < k_blocks; ++kb) {
                const int qb = kb / 8;
                const int ib = kb - qb*8;
                const block_iq3_s & block = hW[((size_t)e*N_OUT + row)*iq_blocks + qb];
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    dot += iq3s_signed_q_host(block, ib, i) * (int) act_qs[(size_t)lane*k + kb*32 + i];
                }
                const int ss = iq3s_scale_host(block, ib);
                const float d = ggml_fp16_to_fp32(block.d) * ggml_fp16_to_fp32(act_ds[(size_t)lane*k_blocks + kb]);
                ref = fmaf(d, (float)(dot * ss), ref);
            }
            const float got = hOut[(size_t)lane*N_OUT + row];
            max_abs = fmaxf(max_abs, fabsf(got - ref));
            if (fabsf(got - ref) > 1e-4f) {
                fprintf(stderr, "mismatch K=%d lane=%d row=%d expert=%d got=%g ref=%g diff=%g\n", k, lane, row, e, got, ref, got-ref);
                return false;
            }
        }
    }

    printf("PASS iq3s_native_case K=%d max_abs=%g\n", k, max_abs);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main() {
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_IQ3_S_NATIVE", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_MAX_TEMP_BYTES", "104857600", 1);
    if (!run_case(256)) return 1;
    if (!run_case(512)) return 2;
    return 0;
}
