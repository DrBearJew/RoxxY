// tests/test-dot4-packed16-correctness.cpp
// DOT4 FlashAttention harness: compares packed16 K (I32) against q8_0 K reference.
// Usage: GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 ./test-dot4-packed16-correctness

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

static bool is_finite(float x) { return std::isfinite(x); }
static bool has_nan(const float * data, size_t n) {
    for (size_t i = 0; i < n; i++) if (std::isnan(data[i])) return true;
    return false;
}
static bool has_inf(const float * data, size_t n) {
    for (size_t i = 0; i < n; i++) if (std::isinf(data[i])) return true;
    return false;
}
static float max_abs_diff(const float * a, const float * b, size_t n) {
    float maxd = 0.0f;
    for (size_t i = 0; i < n; i++) maxd = std::max(maxd, std::fabs(a[i] - b[i]));
    return maxd;
}
static float mean_abs_diff(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += std::fabs(a[i] - b[i]);
    return (float)(sum / n);
}
static float l2_norm(const float * data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double)data[i] * data[i];
    return std::sqrt(sum);
}

static void fill_random(float * data, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}
static void fill_random_half(uint16_t * data, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        // Rough f16 conversion
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        uint16_t h = (uint16_t)((bits >> 16) & 0x8000) | (uint16_t)(((bits >> 13) & 0x3FF));
        data[i] = h;
    }
}

int main() {
    srand(42);

    const int nq = 2;
    const int nk = 256;
    const int D = 256;
    const int n_heads = 2;  // GQA ratio handled by head grouping
    const int batch = 1;

    printf("=== DOT4 Packed16 K Correctness Harness ===\n");
    printf("nq=%d nk=%d D=%d heads=%d batch=%d\n", nq, nk, D, n_heads, batch);

    // ---- Phase A: Create Q, K, V, mask tensors ----

    ggml_init_params params = { 1024*1024*64, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    // Q: f32, shape [D, nq, n_heads, batch]
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, n_heads, batch);
    ggml_set_name(Q, "Q");

    // K_ref: q8_0 for reference path
    ggml_tensor * K_ref = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, nk, n_heads, batch);
    ggml_set_name(K_ref, "K_ref");

    // V: f16 for both paths
    ggml_tensor * V = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nk, n_heads, batch);
    ggml_set_name(V, "V");

    // Causal mask: f16, [nk, nq, 1, batch]
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nq, 1, batch);
    ggml_set_name(mask, "mask");

    // ---- Fill with known data ----

    // Q: f32, random in [-1, 1]
    fill_random((float*)Q->data, ggml_nelements(Q), 1.0f);

    // K_ref: q8_0 needs block format. We'll fill f16 first, then convert.
    // For reference, we write f16 values and let ggml_quantize handle it.
    // Actually v: q8_0 tensor stores data as blocks. We can't easily fill it.
    // Strategy: create f16 K, quantize to q8_0 using ggml, then run.
    ggml_tensor * K_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, nk, n_heads, batch);
    fill_random_half((uint16_t*)K_f16->data, ggml_nelements(K_f16), 1.0f);

    // V: f16, same as K_f16
    std::memcpy(V->data, K_f16->data, ggml_nbytes(K_f16));

    // Causal mask: lower triangular (-inf in masked positions, 0 in visible)
    ggml_tensor * mask_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nk, nq, 1, batch);
    float * mask_data = (float*)mask_f32->data;
    for (int q = 0; q < nq; q++) {
        for (int k = 0; k < nk; k++) {
            mask_data[q * nk + k] = (k > nk - nq + q) ? -1e9f : 0.0f;
        }
    }
    // Convert to f16
    for (size_t i = 0; i < (size_t)nk * nq; i++) {
        float v = mask_data[i];
        uint32_t bits; std::memcpy(&bits, &v, 4);
        ((uint16_t*)mask->data)[i] = (uint16_t)(bits >> 16);
    }

    // ---- Quantize K_ref from K_f16 ----
    {
        // Simple q8_0 quantization: block size 32, each block has d (f16 scale) and 32 int8 values
        // We need to copy K_f16 data into q8_0 blocks.
        // ggml_type_traits has block size info.
        const int blk_size = 32; // q8_0 block size
        float * kf16_data = (float*)malloc(D * nk * n_heads * batch * sizeof(float));
        // Convert f16 to f32 for quantization
        for (size_t i = 0; i < (size_t)D * nk * n_heads * batch; i++) {
            uint16_t h = ((uint16_t*)K_f16->data)[i];
            uint32_t bits = ((uint32_t)h) << 16;
            std::memcpy(&kf16_data[i], &bits, 4);
        }

        // q8_0 block format: [d (f16)][qs (int8 x 32)]
        // d = max_abs / 127
        char * q_data = (char*)K_ref->data;
        size_t n_blocks = ggml_nelements(K_ref) / blk_size;
        for (size_t b = 0; b < n_blocks; b++) {
            float max_abs = 0.0f;
            for (int j = 0; j < blk_size; j++) {
                max_abs = std::max(max_abs, std::fabs(kf16_data[b * blk_size + j]));
            }
            float d = max_abs / 127.0f;
            uint16_t d_half;
            {
                uint32_t bits; std::memcpy(&bits, &d, 4);
                d_half = (uint16_t)(bits >> 16);
            }
            std::memcpy(q_data, &d_half, 2);
            for (int j = 0; j < blk_size; j++) {
                float v = kf16_data[b * blk_size + j];
                int8_t q = (int8_t)std::round(v / d);
                q_data[2 + j] = (char)q;
            }
            q_data += 2 + blk_size; // d (2 bytes) + qs (32 bytes) = 34 bytes
        }
        free(kf16_data);
    }

    ggml_free(ctx);

    // ---- Phase B: Run DOT4 path vs reference path ----
    // For now, just print the tensor shapes and data samples

    printf("\n--- Tensor shapes ---\n");
    printf("Q: [%lld %lld %lld %lld] type=%s\n",
        (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2], (long long)Q->ne[3],
        ggml_type_name(Q->type));
    printf("K_ref: [%lld %lld %lld %lld] type=%s\n",
        (long long)K_ref->ne[0], (long long)K_ref->ne[1], (long long)K_ref->ne[2], (long long)K_ref->ne[3],
        ggml_type_name(K_ref->type));
    printf("V: [%lld %lld %lld %lld] type=%s\n",
        (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2], (long long)V->ne[3],
        ggml_type_name(V->type));
    printf("mask: [%lld %lld %lld %lld] type=%s\n",
        (long long)mask->ne[0], (long long)mask->ne[1], (long long)mask->ne[2], (long long)mask->ne[3],
        ggml_type_name(mask->type));

    // Sample Q data
    printf("\n--- Q first 8 values (head 0, row 0) ---\n");
    float * q_data = (float*)Q->data;
    for (int i = 0; i < 8; i++) printf("  %.4f", q_data[i]);
    printf("\n");

    // Sample mask
    printf("\n--- Causal mask (nq=%d x nk=%d, first 3 rows) ---\n", nq, nk);
    for (int q = 0; q < std::min(nq, 3); q++) {
        printf("  q=%d: ", q);
        for (int k = 0; k < std::min(nk, 16); k++) {
            uint16_t h = ((uint16_t*)mask->data)[q * nk + k];
            uint32_t bits = ((uint32_t)h) << 16;
            float v; std::memcpy(&v, &bits, 4);
            printf("%.0f ", v);
        }
        printf("\n");
    }

    printf("\n=== HARNESS READY - need backend compute for comparison ===\n");
    return 0;
}
