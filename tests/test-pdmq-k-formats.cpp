#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define DP16_PACKED_I8_DESC_HOST_ONLY
#ifndef __host__
#define TEST_PDMQ_DEFINED_HOST
#endif
#ifndef __device__
#define TEST_PDMQ_DEFINED_DEVICE
#endif
#ifndef __forceinline__
#define TEST_PDMQ_DEFINED_FORCEINLINE
#endif
#include "../ggml/src/ggml-cuda/dot4-packed16/mtp-v4-144-tail-page-desc.cuh"
#undef DP16_PACKED_I8_DESC_HOST_ONLY
#ifdef TEST_PDMQ_DEFINED_HOST
#undef __host__
#undef TEST_PDMQ_DEFINED_HOST
#endif
#ifdef TEST_PDMQ_DEFINED_DEVICE
#undef __device__
#undef TEST_PDMQ_DEFINED_DEVICE
#endif
#ifdef TEST_PDMQ_DEFINED_FORCEINLINE
#undef __forceinline__
#undef TEST_PDMQ_DEFINED_FORCEINLINE
#endif

#define CHECK(COND) do { \
    if (!(COND)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
        std::abort(); \
    } \
} while (0)

static constexpr int D = 256;
static constexpr int QK = 32;
static constexpr int QBLOCKS = D / QK;
static constexpr int K_Q4_WORDS = D / 8;
static constexpr int I8_WORDS = D / 4;

static uint32_t pack_q4x8(const uint8_t * c) {
    uint32_t w = 0;
    for (int i = 0; i < 8; ++i) w |= uint32_t(c[i] & 0x0f) << (4 * i);
    return w;
}

static int code_from_packed8(const uint32_t * words4, int i) {
    return int((words4[i / 8] >> (4 * (i & 7))) & 0x0f);
}

static uint32_t pack_i8x4(const int * v) {
    uint32_t w = 0;
    for (int i = 0; i < 4; ++i) w |= uint32_t(uint8_t(int8_t(v[i]))) << (8 * i);
    return w;
}

static int unpack_i8(uint32_t w, int lane) {
    const uint32_t b = (w >> (8 * lane)) & 0xffu;
    return int(int8_t(uint8_t(b)));
}

static void expand_packed8_to_i8_words(const uint32_t * k_q4, uint32_t * k_i8) {
    for (int g = 0; g < I8_WORDS; ++g) {
        int vals[4];
        for (int i = 0; i < 4; ++i) {
            const int d = 4 * g + i;
            vals[i] = code_from_packed8(k_q4, d) - 8;
        }
        k_i8[g] = pack_i8x4(vals);
    }
}

static int dot_packed8_direct_i32(const uint32_t * q_i8, const uint32_t * k_q4) {
    int acc = 0;
    for (int d = 0; d < D; ++d) {
        acc += unpack_i8(q_i8[d / 4], d & 3) * (code_from_packed8(k_q4, d) - 8);
    }
    return acc;
}

static int dot_expanded_i8_i32(const uint32_t * q_i8, const uint32_t * k_q4) {
    uint32_t k_i8[I8_WORDS];
    expand_packed8_to_i8_words(k_q4, k_i8);
    int acc = 0;
    for (int g = 0; g < I8_WORDS; ++g) {
        for (int i = 0; i < 4; ++i) acc += unpack_i8(q_i8[g], i) * unpack_i8(k_i8[g], i);
    }
    return acc;
}

static float qk_packed8_direct(const uint32_t * q_i8, const float * q_scales, const uint32_t * k_q4, const float * k_scales) {
    float total = 0.0f;
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        int acc = 0;
        for (int d = qb * QK; d < (qb + 1) * QK; ++d) {
            acc += unpack_i8(q_i8[d / 4], d & 3) * (code_from_packed8(k_q4, d) - 8);
        }
        total += float(acc) * q_scales[qb] * k_scales[qb];
    }
    return total;
}

static float qk_expanded_i8(const uint32_t * q_i8, const float * q_scales, const uint32_t * k_q4, const float * k_scales) {
    uint32_t k_i8[I8_WORDS];
    expand_packed8_to_i8_words(k_q4, k_i8);
    float total = 0.0f;
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        int acc = 0;
        for (int g = qb * (QK / 4); g < (qb + 1) * (QK / 4); ++g) {
            for (int i = 0; i < 4; ++i) acc += unpack_i8(q_i8[g], i) * unpack_i8(k_i8[g], i);
        }
        total += float(acc) * q_scales[qb] * k_scales[qb];
    }
    return total;
}

static void pack_f32_row_to_packed8(const float * x, uint32_t * k_q4, float * k_scales) {
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        float amax = 0.0f, maxv = 0.0f;
        for (int i = 0; i < QK; ++i) {
            const float v = x[qb * QK + i];
            const float av = std::fabs(v);
            if (av > amax) { amax = av; maxv = v; }
        }
        const float d = maxv / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        k_scales[qb] = d;
        uint8_t codes[QK];
        for (int i = 0; i < QK; ++i) {
            const int c = id == 0.0f ? 8 : std::min(15, std::max(0, int(x[qb * QK + i] * id + 8.5f)));
            codes[i] = uint8_t(c);
        }
        for (int w = 0; w < 4; ++w) k_q4[qb * 4 + w] = pack_q4x8(codes + w * 8);
    }
}

static void pack_f32_row_to_q_i8(const float * x, uint32_t * q_i8, float * q_scales) {
    for (int qb = 0; qb < QBLOCKS; ++qb) {
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[qb * QK + i]));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        q_scales[qb] = s;
        for (int g = 0; g < QK / 4; ++g) {
            int vals[4];
            for (int i = 0; i < 4; ++i) {
                const float v = x[qb * QK + g * 4 + i] / s;
                vals[i] = std::min(127, std::max(-127, int(std::nearbyint(v))));
            }
            q_i8[qb * (QK / 4) + g] = pack_i8x4(vals);
        }
    }
}

template <typename DotFn>
static std::vector<float> attention(const std::vector<uint32_t> & q_words, const std::vector<float> & q_scales,
                                    const std::vector<uint32_t> & k_words, const std::vector<float> & k_scales,
                                    const std::vector<float> & v, int nq, int nk, int vd, bool causal, DotFn dot_fn) {
    std::vector<float> out(size_t(nq * vd), 0.0f);
    std::vector<float> logits(static_cast<size_t>(nk));
    for (int q = 0; q < nq; ++q) {
        float m = -INFINITY;
        for (int k = 0; k < nk; ++k) {
            float l = dot_fn(&q_words[size_t(q) * I8_WORDS], &q_scales[size_t(q) * QBLOCKS],
                            &k_words[size_t(k) * K_Q4_WORDS], &k_scales[size_t(k) * QBLOCKS]);
            if (causal && k > q) l = -INFINITY;
            logits[k] = l;
            m = std::max(m, l);
        }
        float denom = 0.0f;
        for (int k = 0; k < nk; ++k) {
            logits[k] = std::isfinite(logits[k]) ? std::exp(logits[k] - m) : 0.0f;
            denom += logits[k];
        }
        for (int k = 0; k < nk; ++k) {
            const float p = denom > 0.0f ? logits[k] / denom : 0.0f;
            for (int d = 0; d < vd; ++d) out[size_t(q) * vd + d] += p * v[size_t(k) * vd + d];
        }
    }
    return out;
}

static void test_nibble_layout() {
    uint8_t c[32];
    for (int i = 0; i < 32; ++i) c[i] = uint8_t(i & 15);
    uint32_t words4[4] = { pack_q4x8(c + 0), pack_q4x8(c + 8), pack_q4x8(c + 16), pack_q4x8(c + 24) };
    for (int i = 0; i < 32; ++i) CHECK(code_from_packed8(words4, i) == (i & 15));
}

static void test_deterministic_dots() {
    uint32_t q[I8_WORDS];
    uint32_t k[K_Q4_WORDS];
    float qs[QBLOCKS], ks[QBLOCKS];
    for (int i = 0; i < QBLOCKS; ++i) qs[i] = ks[i] = 1.0f;
    int ones[4] = {1, 1, 1, 1};
    for (int i = 0; i < I8_WORDS; ++i) q[i] = pack_i8x4(ones);
    uint8_t c9[8] = {9,9,9,9,9,9,9,9};
    for (int i = 0; i < K_Q4_WORDS; ++i) k[i] = pack_q4x8(c9);
    CHECK(dot_packed8_direct_i32(q, k) == 256);
    CHECK(dot_expanded_i8_i32(q, k) == 256);
    CHECK(qk_packed8_direct(q, qs, k, ks) == 256.0f);
    CHECK(qk_expanded_i8(q, qs, k, ks) == 256.0f);

    int neg2[4] = {-2, -2, -2, -2};
    for (int i = 0; i < I8_WORDS; ++i) q[i] = pack_i8x4(neg2);
    uint8_t c7[8] = {7,7,7,7,7,7,7,7};
    for (int i = 0; i < K_Q4_WORDS; ++i) k[i] = pack_q4x8(c7);
    CHECK(dot_packed8_direct_i32(q, k) == 512);
    CHECK(dot_expanded_i8_i32(q, k) == 512);
    CHECK(qk_packed8_direct(q, qs, k, ks) == 512.0f);
    CHECK(qk_expanded_i8(q, qs, k, ks) == 512.0f);
}

static void test_txn_tail_page_stage_contract() {
    constexpr uint32_t idx0 = 7734;
    constexpr uint32_t n_tokens = 5;
    constexpr uint32_t kv_size = 40960;
    constexpr uint32_t page_base = idx0 & ~(MTP_V4_144_PAGE_TOKENS - 1u);
    constexpr uint32_t page_end = page_base + MTP_V4_144_PAGE_TOKENS;
    constexpr uint32_t slot_begin = idx0 - page_base;
    constexpr uint32_t slot_end = slot_begin + n_tokens;
    CHECK(page_base == 7728);
    CHECK(page_end == 7744);
    CHECK(slot_begin == 6);
    CHECK(slot_end == 11);
    CHECK(MTP_PACKED16_K_ROW_BYTES == 272);
    CHECK(MTP_PACKED16_K_PAGE_BYTES == 4352);
    CHECK(MTP_V4_144_ROW_BYTES == 144);
    CHECK(MTP_V4_144_PAGE_BYTES == 2304);

    mtp_v4_144_tail_stage_desc_v1 k_stage = mtp_v4_144_tail_stage_make(
        MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K, kv_size, idx0, n_tokens, MTP_PACKED16_K_ROW_BYTES);
    CHECK(k_stage.slots_before == 6);
    CHECK(k_stage.slots_after == 5);
    CHECK(k_stage.merge_copy_slots == 11);
    CHECK(k_stage.payload_bytes == 1360);
    CHECK(k_stage.merge_copy_bytes == 2992);
    CHECK(mtp_v4_144_tail_stage_validate_static(k_stage) == MTP_V4_144_TAIL_STAGE_OK);
    k_stage.row_bytes = 144;
    CHECK(mtp_v4_144_tail_stage_validate_static(k_stage) == MTP_V4_144_TAIL_STAGE_BAD_ROW_BYTES);

    mtp_v4_144_tail_stage_desc_v1 v_stage = mtp_v4_144_tail_stage_make(
        MTP_V4_144_TAIL_STAGE_KIND_V4_144, kv_size, idx0, n_tokens, MTP_V4_144_ROW_BYTES);
    CHECK(v_stage.payload_bytes == 720);
    CHECK(v_stage.merge_copy_bytes == 1584);
    CHECK(mtp_v4_144_tail_stage_validate_static(v_stage) == MTP_V4_144_TAIL_STAGE_OK);

    dp16_packed_i8_desc_v1 k_desc = {};
    k_desc.version = DP16_PACKED_I8_DESC_VERSION;
    k_desc.lanes_per_vector = DP16_PACKED_I8X16_LANES;
    k_desc.words_per_vector = DP16_PACKED_I8X16_WORDS;
    k_desc.bytes_per_vector = DP16_PACKED_I8X16_BYTES;
    k_desc.bytes_per_word = DP16_PACKED_I8_WORD_BYTES;
    k_desc.layout_kind = DP16_PACKED_I8_LAYOUT_ROW;
    k_desc.axis_x = DP16_PACKED_I8_AXIS_D16;
    k_desc.axis_y = DP16_PACKED_I8_AXIS_TOKEN;
    k_desc.axis_z = DP16_PACKED_I8_AXIS_HEAD;
    k_desc.scale_layout = DP16_PACKED_I8_SCALE_LAYOUT_ROW;
    k_desc.scale_axis_x = DP16_PACKED_I8_AXIS_QBLOCK;
    k_desc.scale_axis_y = DP16_PACKED_I8_AXIS_TOKEN;
    k_desc.scale_axis_z = DP16_PACKED_I8_AXIS_HEAD;
    k_desc.logical_x = MTP_PACKED16_K_D / DP16_PACKED_I8X16_LANES;
    k_desc.logical_y = kv_size;
    k_desc.logical_z = 4;
    k_desc.physical_x = k_desc.logical_x;
    k_desc.physical_y = k_desc.logical_y;
    k_desc.physical_z = k_desc.logical_z;
    k_desc.x_stride_bytes = DP16_PACKED_I8X16_BYTES;
    k_desc.y_stride_bytes = MTP_PACKED16_K_WORDS * sizeof(uint32_t);
    k_desc.z_stride_bytes = uint64_t(kv_size) * k_desc.y_stride_bytes;
    k_desc.scale_x_stride_bytes = sizeof(uint16_t);
    k_desc.scale_y_stride_bytes = MTP_PACKED16_K_QBLOCKS * sizeof(uint16_t);
    k_desc.scale_z_stride_bytes = uint64_t(kv_size) * k_desc.scale_y_stride_bytes;
    CHECK(dp16_i8x16_desc_has_vector_abi(k_desc));
    const uint64_t k_payload_first = dp16_packed_i8_payload_byte_offset(k_desc, 0, idx0, 0, 0);
    const uint64_t k_payload_end = dp16_packed_i8_payload_byte_offset(k_desc, 0, idx0 + n_tokens - 1u, k_desc.logical_x - 1u, k_desc.words_per_vector - 1u) + k_desc.bytes_per_word;
    const uint64_t k_scale_first = dp16_packed_i8_scale_byte_offset(k_desc, 0, idx0, 0);
    const uint64_t k_scale_end = dp16_packed_i8_scale_byte_offset(k_desc, 0, idx0 + n_tokens - 1u, MTP_PACKED16_K_QBLOCKS - 1u) + sizeof(uint16_t);
    CHECK(k_payload_first == 1979904);
    CHECK(k_payload_end == 1981184);
    CHECK(k_scale_first == 123744);
    CHECK(k_scale_end == 123824);

    mtp_v4_144_tail_page_desc_v1 v_desc = {};
    v_desc.v4_page_stride_bytes = MTP_V4_144_PAGE_BYTES;
    v_desc.v4_head_stride_bytes = uint64_t(kv_size) * MTP_V4_144_ROW_BYTES;
    v_desc.v4_batch_stride_bytes = v_desc.v4_head_stride_bytes * 4u;
    const uint32_t physical_page = page_base / MTP_V4_144_PAGE_TOKENS;
    CHECK(physical_page == 483);
    const uint64_t v_payload_first = mtp_v4_144_tail_page_v4_payload_byte_offset(v_desc, physical_page, slot_begin, 0, 0, 0);
    const uint64_t v_payload_end = mtp_v4_144_tail_page_v4_payload_byte_offset(v_desc, physical_page, slot_end - 1u, MTP_V4_144_D - 1u, 0, 0) + sizeof(uint32_t);
    const uint64_t v_scale_first = mtp_v4_144_tail_page_v4_scale_byte_offset(v_desc, physical_page, slot_begin, 0, 0, 0);
    const uint64_t v_scale_end = mtp_v4_144_tail_page_v4_scale_byte_offset(v_desc, physical_page, slot_end - 1u, MTP_V4_144_D - MTP_V4_144_D32, 0, 0) + sizeof(uint16_t);
    CHECK(v_payload_first == 1112832);
    CHECK(v_payload_end == 1114880);
    CHECK(v_scale_first == 1114892);
    CHECK(v_scale_end == 1115126);
}

static void test_randomized_qk_and_attention() {
    std::mt19937 rng(0x5eed1234u);
    std::uniform_int_distribution<int> qi(-127, 127), code(0, 15);
    std::uniform_real_distribution<float> scale(-0.2f, 0.2f), val(-2.0f, 2.0f);

    for (int iter = 0; iter < 200; ++iter) {
        uint32_t q[I8_WORDS], k[K_Q4_WORDS];
        float qs[QBLOCKS], ks[QBLOCKS];
        for (int g = 0; g < I8_WORDS; ++g) {
            int tmp[4] = { qi(rng), qi(rng), qi(rng), qi(rng) };
            q[g] = pack_i8x4(tmp);
        }
        for (int w = 0; w < K_Q4_WORDS; ++w) {
            uint8_t tmp[8];
            for (int i = 0; i < 8; ++i) tmp[i] = uint8_t(code(rng));
            k[w] = pack_q4x8(tmp);
        }
        for (int s = 0; s < QBLOCKS; ++s) {
            qs[s] = std::max(0.001f, std::fabs(scale(rng)));
            ks[s] = scale(rng);
            if (std::fabs(ks[s]) < 0.001f) ks[s] = -0.03125f;
        }
        CHECK(dot_packed8_direct_i32(q, k) == dot_expanded_i8_i32(q, k));
        const float a = qk_packed8_direct(q, qs, k, ks);
        const float b = qk_expanded_i8(q, qs, k, ks);
        CHECK(std::fabs(a - b) <= 1e-5f * std::max(1.0f, std::fabs(a)));
    }

    const int nq = 4, nk = 5, vd = 17;
    std::vector<uint32_t> q_words(size_t(nq) * I8_WORDS), k_words(size_t(nk) * K_Q4_WORDS);
    std::vector<float> q_scales(size_t(nq) * QBLOCKS), k_scales(size_t(nk) * QBLOCKS), v(size_t(nk) * vd);
    std::vector<float> tmp(D);
    for (int q = 0; q < nq; ++q) {
        for (float & x : tmp) x = val(rng);
        pack_f32_row_to_q_i8(tmp.data(), &q_words[size_t(q) * I8_WORDS], &q_scales[size_t(q) * QBLOCKS]);
    }
    for (int k = 0; k < nk; ++k) {
        for (float & x : tmp) x = val(rng);
        pack_f32_row_to_packed8(tmp.data(), &k_words[size_t(k) * K_Q4_WORDS], &k_scales[size_t(k) * QBLOCKS]);
    }
    for (float & x : v) x = val(rng);
    auto direct = attention(q_words, q_scales, k_words, k_scales, v, nq, nk, vd, true, qk_packed8_direct);
    auto expanded = attention(q_words, q_scales, k_words, k_scales, v, nq, nk, vd, true, qk_expanded_i8);
    CHECK(direct.size() == expanded.size());
    for (size_t i = 0; i < direct.size(); ++i) {
        CHECK(std::fabs(direct[i] - expanded[i]) <= 1e-6f * std::max(1.0f, std::fabs(direct[i])));
    }
}

int main() {
    test_nibble_layout();
    test_deterministic_dots();
    test_txn_tail_page_stage_contract();
    test_randomized_qk_and_attention();
    std::puts("test-pdmq-k-formats: PASS");
    return 0;
}
