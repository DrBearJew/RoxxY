// Standalone HIP microbench/check for transactional QBlock V4_144 tail paging.
// Build example:
//   hipcc --offload-arch=gfx1100 -O2 scripts/hip/transactional-v4-144-tail-page-check.hip -o /tmp/transactional-v4-144-tail-page-check
//   /tmp/transactional-v4-144-tail-page-check
//
// No llama.cpp headers are required. This mirrors only the source formulas from:
//   ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu
//   ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh
//   ggml/src/ggml-cuda/dot4-packed16/dp16-packed-i8-desc.cuh

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t err__ = (call); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

static constexpr int D = 256;
static constexpr int PAGE_TOKENS = 16;
static constexpr int D32 = 32;
static constexpr int D32_BLOCKS = D / D32;
static constexpr int WORDS_PER_D = 2;
static constexpr int PAYLOAD_WORDS = D * WORDS_PER_D;
static constexpr int PAYLOAD_BYTES = PAYLOAD_WORDS * 4;
static constexpr int SCALE_COUNT = D32_BLOCKS * PAGE_TOKENS;
static constexpr int SCALE_BYTES = SCALE_COUNT * 2;
static constexpr int PAGE_BYTES = PAYLOAD_BYTES + SCALE_BYTES;
static constexpr int ROW_BYTES = PAGE_BYTES / PAGE_TOKENS;

static constexpr int K_WORDS = D / 4;
static constexpr int K_QBLOCKS = D / 32;
static constexpr int K_D16 = 16;
static constexpr int K_WORDS_PER_D16 = K_D16 / 4;
static constexpr int WORD_BYTES = 4;
static constexpr int SCALE_BYTES_U16 = 2;
static constexpr int I8X16_BYTES = 16;
static constexpr int K_ROW_BYTES = K_WORDS * WORD_BYTES + K_QBLOCKS * SCALE_BYTES_U16;
static constexpr int K_PAGE_BYTES = K_ROW_BYTES * PAGE_TOKENS;
static constexpr int KV_PAGE_BYTES = K_PAGE_BYTES + PAGE_BYTES;

static_assert(PAYLOAD_BYTES == 2048, "bad V4_144 payload bytes");
static_assert(SCALE_BYTES == 256, "bad V4_144 scale bytes");
static_assert(PAGE_BYTES == 2304, "bad V4_144 page bytes");
static_assert(ROW_BYTES == 144, "bad V4_144 row bytes");
static_assert(K_ROW_BYTES == 272, "bad packed16 K row bytes");
static_assert(KV_PAGE_BYTES == 6656, "bad packed16 K + V4_144 page bytes");

__host__ __device__ static inline float half_bits_to_float(const uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                --e;
            }
            frac &= 0x03ffu;
            bits = sign | (uint32_t(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } cvt;
    cvt.u = bits;
    return cvt.f;
}

__host__ __device__ static inline int deterministic_code(const int pp, const int slot, const int d) {
    return (pp * 3 + slot * 5 + d * 7) & 0x0f;
}

__host__ __device__ static inline uint16_t deterministic_scale_bits(const int pp, const int slot, const int d32) {
    static constexpr uint16_t lut[8] = {
        0x3c00u, //  1.0
        0x3800u, //  0.5
        0xbc00u, // -1.0
        0x3400u, //  0.25
        0xb800u, // -0.5
        0x4000u, //  2.0
        0x3000u, //  0.125
        0xc000u, // -2.0
    };
    return lut[(pp * 5 + slot * 3 + d32) & 7];
}

__host__ __device__ static inline int v4_payload_word_index(const int slot, const int d) {
    return d * WORDS_PER_D + (slot >> 3);
}

__host__ __device__ static inline int v4_scale_index(const int slot, const int d) {
    return (d / D32) * PAGE_TOKENS + slot;
}

__host__ __device__ static inline float expected_v4_value(const int pp, const int slot, const int d) {
    const int q = deterministic_code(pp, slot, d) - 8;
    return float(q) * half_bits_to_float(deterministic_scale_bits(pp, slot, d / D32));
}

__host__ __device__ static inline uint64_t packed16_payload_offset(const int layout, const int kv_capacity, const int head, const int token, const int d_word) {
    if (layout == 0) {
        return uint64_t((head * kv_capacity + token) * K_WORDS + d_word) * WORD_BYTES;
    }
    const int d16 = d_word / K_WORDS_PER_D16;
    const int w4 = d_word - d16 * K_WORDS_PER_D16;
    return uint64_t(head) * uint64_t(kv_capacity) * uint64_t(K_WORDS) * WORD_BYTES +
           uint64_t(d16) * uint64_t(kv_capacity) * uint64_t(K_WORDS_PER_D16) * WORD_BYTES +
           uint64_t(token) * uint64_t(K_WORDS_PER_D16) * WORD_BYTES +
           uint64_t(w4) * WORD_BYTES;
}

__host__ __device__ static inline uint64_t packed16_scale_offset(const int layout, const int kv_capacity, const int head, const int token, const int qblock) {
    if (layout == 0) {
        return uint64_t((head * kv_capacity + token) * K_QBLOCKS + qblock) * SCALE_BYTES_U16;
    }
    return uint64_t(head) * uint64_t(kv_capacity) * uint64_t(K_QBLOCKS) * SCALE_BYTES_U16 +
           uint64_t(qblock) * uint64_t(kv_capacity) * SCALE_BYTES_U16 +
           uint64_t(token) * SCALE_BYTES_U16;
}

struct VQuery {
    int logical_token;
    int d;
};

struct KQuery {
    int layout;
    int kv_capacity;
    int head;
    int token;
    int d_word;
    int qblock;
};

__global__ void v4_tail_decode_kernel(
        const uint32_t * __restrict__ payload,
        const uint16_t * __restrict__ scales,
        const int * __restrict__ block_table,
        const VQuery * __restrict__ queries,
        float * __restrict__ out,
        int * __restrict__ visible,
        const int n_queries,
        const int logical_base_token,
        const int valid_tail_tokens) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_queries) {
        return;
    }
    const int rel = queries[i].logical_token - logical_base_token;
    if (rel < 0 || rel >= valid_tail_tokens) {
        visible[i] = 0;
        out[i] = -7777.0f;
        return;
    }
    const int logical_page = rel / PAGE_TOKENS;
    const int slot = rel & (PAGE_TOKENS - 1);
    const int pp = block_table[logical_page];
    const int d = queries[i].d;
    const uint32_t word = payload[pp * PAYLOAD_WORDS + v4_payload_word_index(slot, d)];
    const int q = int((word >> (4 * (slot & 7))) & 0x0fu) - 8;
    const uint16_t scale_bits = scales[pp * SCALE_COUNT + v4_scale_index(slot, d)];
    visible[i] = 1;
    out[i] = float(q) * half_bits_to_float(scale_bits);
}

__global__ void packed16_addr_kernel(
        const KQuery * __restrict__ queries,
        uint64_t * __restrict__ payload_offsets,
        uint64_t * __restrict__ scale_offsets,
        const int n_queries) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_queries) {
        return;
    }
    const KQuery q = queries[i];
    payload_offsets[i] = packed16_payload_offset(q.layout, q.kv_capacity, q.head, q.token, q.d_word);
    scale_offsets[i] = packed16_scale_offset(q.layout, q.kv_capacity, q.head, q.token, q.qblock);
}

static void fill_v4_pages(std::vector<uint32_t> & payload, std::vector<uint16_t> & scales, const int physical_pages) {
    payload.assign(size_t(physical_pages) * PAYLOAD_WORDS, 0u);
    scales.assign(size_t(physical_pages) * SCALE_COUNT, 0u);
    for (int pp = 0; pp < physical_pages; ++pp) {
        for (int slot = 0; slot < PAGE_TOKENS; ++slot) {
            for (int d = 0; d < D; ++d) {
                const int wi = pp * PAYLOAD_WORDS + v4_payload_word_index(slot, d);
                const int shift = 4 * (slot & 7);
                payload[wi] &= ~(uint32_t(0x0f) << shift);
                payload[wi] |= uint32_t(deterministic_code(pp, slot, d)) << shift;
            }
            for (int d32 = 0; d32 < D32_BLOCKS; ++d32) {
                scales[pp * SCALE_COUNT + d32 * PAGE_TOKENS + slot] = deterministic_scale_bits(pp, slot, d32);
            }
        }
    }
}

int main() {
    int device = 0;
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("transactional-v4-144-tail-page-check: device=%s arch=%s\n", prop.name, prop.gcnArchName);

    constexpr int physical_pages = 4;
    const int block_table_host[3] = {2, 0, 3};
    constexpr int logical_base_token = 64;
    constexpr int valid_tail_tokens = 34;

    std::vector<uint32_t> h_payload;
    std::vector<uint16_t> h_scales;
    fill_v4_pages(h_payload, h_scales, physical_pages);

    const std::vector<VQuery> h_vqueries = {
        {64, 0}, {65, 31}, {79, 32}, {80, 63}, {81, 127},
        {95, 128}, {96, 191}, {97, 255}, {98, 7}, {111, 99}, {63, 11},
    };
    const int nv = int(h_vqueries.size());

    uint32_t * d_payload = nullptr;
    uint16_t * d_scales = nullptr;
    int * d_block_table = nullptr;
    VQuery * d_vqueries = nullptr;
    float * d_vout = nullptr;
    int * d_visible = nullptr;
    HIP_CHECK(hipMalloc(&d_payload, h_payload.size() * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_scales, h_scales.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_block_table, sizeof(block_table_host)));
    HIP_CHECK(hipMalloc(&d_vqueries, h_vqueries.size() * sizeof(VQuery)));
    HIP_CHECK(hipMalloc(&d_vout, h_vqueries.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_visible, h_vqueries.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_payload, h_payload.data(), h_payload.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_block_table, block_table_host, sizeof(block_table_host), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vqueries, h_vqueries.data(), h_vqueries.size() * sizeof(VQuery), hipMemcpyHostToDevice));
    v4_tail_decode_kernel<<<1, 64>>>(d_payload, d_scales, d_block_table, d_vqueries, d_vout, d_visible, nv, logical_base_token, valid_tail_tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_vout(nv);
    std::vector<int> h_visible(nv);
    HIP_CHECK(hipMemcpy(h_vout.data(), d_vout, h_vout.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_visible.data(), d_visible, h_visible.size() * sizeof(int), hipMemcpyDeviceToHost));

    for (int i = 0; i < nv; ++i) {
        const int rel = h_vqueries[i].logical_token - logical_base_token;
        const bool exp_visible = rel >= 0 && rel < valid_tail_tokens;
        if (h_visible[i] != (exp_visible ? 1 : 0)) {
            std::fprintf(stderr, "visible mismatch i=%d got=%d expected=%d\n", i, h_visible[i], exp_visible ? 1 : 0);
            return 2;
        }
        if (!exp_visible) {
            if (h_vout[i] != -7777.0f) {
                std::fprintf(stderr, "stale token was not masked i=%d got=%g\n", i, h_vout[i]);
                return 3;
            }
            continue;
        }
        const int lp = rel / PAGE_TOKENS;
        const int slot = rel & (PAGE_TOKENS - 1);
        const int pp = block_table_host[lp];
        const float expected = expected_v4_value(pp, slot, h_vqueries[i].d);
        if (std::fabs(h_vout[i] - expected) > 1.0e-6f) {
            std::fprintf(stderr, "V4 decode mismatch i=%d token=%d d=%d pp=%d slot=%d got=%g expected=%g\n",
                    i, h_vqueries[i].logical_token, h_vqueries[i].d, pp, slot, h_vout[i], expected);
            return 4;
        }
    }

    const std::vector<KQuery> h_kqueries = {
        {0, 16, 0, 0, 0, 0},
        {0, 16, 1, 15, 63, 7},
        {1, 16, 0, 15, 63, 7},
        {1, 64, 2, 17, 12, 3},
        {1, 257, 1, 256, 55, 6},
        {0, 257, 2, 128, 31, 4},
    };
    const int nk = int(h_kqueries.size());
    KQuery * d_kqueries = nullptr;
    uint64_t * d_payload_offsets = nullptr;
    uint64_t * d_scale_offsets = nullptr;
    HIP_CHECK(hipMalloc(&d_kqueries, h_kqueries.size() * sizeof(KQuery)));
    HIP_CHECK(hipMalloc(&d_payload_offsets, h_kqueries.size() * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&d_scale_offsets, h_kqueries.size() * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_kqueries, h_kqueries.data(), h_kqueries.size() * sizeof(KQuery), hipMemcpyHostToDevice));
    packed16_addr_kernel<<<1, 64>>>(d_kqueries, d_payload_offsets, d_scale_offsets, nk);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> h_payload_offsets(nk);
    std::vector<uint64_t> h_scale_offsets(nk);
    HIP_CHECK(hipMemcpy(h_payload_offsets.data(), d_payload_offsets, h_payload_offsets.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_scale_offsets.data(), d_scale_offsets, h_scale_offsets.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < nk; ++i) {
        const KQuery q = h_kqueries[i];
        const uint64_t exp_payload = packed16_payload_offset(q.layout, q.kv_capacity, q.head, q.token, q.d_word);
        const uint64_t exp_scale = packed16_scale_offset(q.layout, q.kv_capacity, q.head, q.token, q.qblock);
        if (h_payload_offsets[i] != exp_payload || h_scale_offsets[i] != exp_scale) {
            std::fprintf(stderr, "packed16 offset mismatch i=%d got=(%llu,%llu) expected=(%llu,%llu)\n", i,
                    (unsigned long long) h_payload_offsets[i], (unsigned long long) h_scale_offsets[i],
                    (unsigned long long) exp_payload, (unsigned long long) exp_scale);
            return 5;
        }
    }

    HIP_CHECK(hipFree(d_payload));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_block_table));
    HIP_CHECK(hipFree(d_vqueries));
    HIP_CHECK(hipFree(d_vout));
    HIP_CHECK(hipFree(d_visible));
    HIP_CHECK(hipFree(d_kqueries));
    HIP_CHECK(hipFree(d_payload_offsets));
    HIP_CHECK(hipFree(d_scale_offsets));

    std::printf("transactional-v4-144-tail-page-check: PASS pages=%d valid_tail_tokens=%d K_row=%d V_page=%d KV_page=%d\n",
            physical_pages, valid_tail_tokens, K_ROW_BYTES, PAGE_BYTES, KV_PAGE_BYTES);
    return 0;
}
