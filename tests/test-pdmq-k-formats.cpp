#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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
    test_randomized_qk_and_attention();
    std::puts("test-pdmq-k-formats: PASS");
    return 0;
}
