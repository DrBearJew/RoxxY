// rdna-i8-packed16-moe-projection-iq4xs-native-smoke.cpp
// Test-only smoke for IQ4_XS-native packed16 WMMA route inside
// GGML_OP_MOE_ROUTED_LANES_PROJECTION. Reference matches vec_dot_iq4_xs_q8_1:
// table-value int8 payload, integer (ls - 32) subscale, F16 d * q8_1.d.

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int QK_K_HOST = 256;

struct iq4_xs_host {
    ggml_fp16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K_HOST/64];
    uint8_t qs[QK_K_HOST/2];
};
static_assert(sizeof(iq4_xs_host) == 136, "iq4_xs_host layout mismatch");

static int iq4nl_value_host(int q) {
    static const int8_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return (int) table[q & 15];
}

static void set_iq4xs_ls(iq4_xs_host & block, int ib, int ls) {
    block.scales_l[ib/2] |= (uint8_t) ((ls & 0x0f) << (4*(ib%2)));
    block.scales_h |= (uint16_t) (((ls >> 4) & 0x03) << (2*ib));
}

static int get_iq4xs_ls(const iq4_xs_host & block, int ib) {
    return ((block.scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((block.scales_h >> (2*ib)) & 0x03) << 4);
}

static int iq4xs_table_value(const iq4_xs_host & block, int ib, int lane) {
    const uint8_t qbyte = block.qs[ib*16 + (lane & 15)];
    const int q = lane < 16 ? (qbyte & 0x0f) : (qbyte >> 4);
    return iq4nl_value_host(q);
}

static void quantize_q8_1_ref(
        const std::vector<float> & compact,
        int n_lanes,
        int k,
        std::vector<int8_t> & qs,
        std::vector<ggml_fp16_t> & ds) {
    const int k_blocks = k / 32;
    qs.assign((size_t)n_lanes*k, 0);
    ds.assign((size_t)n_lanes*k_blocks, ggml_fp32_to_fp16(0.0f));
    for (int lane = 0; lane < n_lanes; ++lane) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(compact[(size_t)lane*k + kb*32 + i]));
            }
            const float d = amax / 127.0f;
            ds[(size_t)lane*k_blocks + kb] = ggml_fp32_to_fp16(d);
            for (int i = 0; i < 32; ++i) {
                const float x = compact[(size_t)lane*k + kb*32 + i];
                int qi = amax == 0.0f ? 0 : (int) roundf(x / d);
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
    ggml_tensor * W        = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_XS, k, N_OUT, N_EXPERT);
    ggml_tensor * compact  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, N_LANES);
    ggml_set_name(selected, "packed16_moe_iq4xs_native_selected");
    ggml_set_name(route_w, "packed16_moe_iq4xs_native_route_weights");
    ggml_set_name(W, "packed16_moe_iq4xs_native_weight");
    ggml_set_name(compact, "packed16_moe_iq4xs_native_compact");

    ggml_tensor * lanes = ggml_moe_routed_lanes(ctx, selected, route_w, N_EXPERT);
    ggml_set_name(lanes, "packed16_moe_iq4xs_native_lanes");
    ggml_tensor * bounds = ggml_moe_routed_lanes_expert_bounds(ctx, lanes);
    ggml_set_name(bounds, "packed16_moe_iq4xs_native_bounds");
    ggml_tensor * out = ggml_moe_routed_lanes_projection(ctx, W, compact, lanes, bounds);
    ggml_set_name(out, "packed16_moe_iq4xs_native_output");

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

    std::vector<iq4_xs_host> hW((size_t)N_EXPERT*N_OUT*iq_blocks);
    const int ls_extremes[4] = {0, 31, 32, 63};
    for (int e = 0; e < N_EXPERT; ++e) {
        for (int row = 0; row < N_OUT; ++row) {
            for (int qb = 0; qb < iq_blocks; ++qb) {
                iq4_xs_host & block = hW[((size_t)e*N_OUT + row)*iq_blocks + qb];
                std::memset(&block, 0, sizeof(block));
                block.d = ggml_fp32_to_fp16(0.0002f * (float)(1 + e + row + qb));
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ls_extremes[(e + row + qb + ib) & 3];
                    set_iq4xs_ls(block, ib, ls);
                }
                for (int ib = 0; ib < 8; ++ib) {
                    for (int i = 0; i < 16; ++i) {
                        const int q0 = (e*7 + row*5 + qb*11 + ib*3 + i) & 15;
                        const int q1 = (e*13 + row*3 + qb*5 + ib*7 + i*9) & 15;
                        block.qs[ib*16 + i] = (uint8_t)(q0 | (q1 << 4));
                    }
                }
            }
        }
    }
    ggml_backend_tensor_set(W, hW.data(), 0, hW.size()*sizeof(iq4_xs_host));

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
                const iq4_xs_host & block = hW[((size_t)e*N_OUT + row)*iq_blocks + qb];
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    dot += iq4xs_table_value(block, ib, i) * (int) act_qs[(size_t)lane*k + kb*32 + i];
                }
                const int ss = get_iq4xs_ls(block, ib) - 32;
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

    printf("PASS iq4xs_native_case K=%d max_abs=%g\n", k, max_abs);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main() {
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_IQ4_XS_NATIVE", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_GEMM_ID_LOG", "1", 1);
    setenv("GGML_CUDA_RDNA_I8_PACKED16_MOE_PROJECTION_MAX_TEMP_BYTES", "104857600", 1);
    if (!run_case(256)) return 1;
    if (!run_case(512)) return 2;
    return 0;
}
