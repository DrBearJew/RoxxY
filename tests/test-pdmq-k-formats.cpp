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
#include "../ggml/src/ggml-cuda/dot4-packed16/mtp-qblock-txn-lineage.cuh"
#include "../ggml/src/ggml-cuda/dot4-packed16/mtp-v4-144-tail-page-state.cuh"
#include "../src/llama-mtp-qblock-paged-state.h"
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

static dp16_packed_i8_desc_v1 make_test_packed16_row_desc(uint32_t kv_size, uint32_t heads) {
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
    k_desc.logical_z = heads;
    k_desc.physical_x = k_desc.logical_x;
    k_desc.physical_y = k_desc.logical_y;
    k_desc.physical_z = k_desc.logical_z;
    k_desc.x_stride_bytes = DP16_PACKED_I8X16_BYTES;
    k_desc.y_stride_bytes = MTP_PACKED16_K_WORDS * sizeof(uint32_t);
    k_desc.z_stride_bytes = uint64_t(kv_size) * k_desc.y_stride_bytes;
    k_desc.scale_x_stride_bytes = sizeof(uint16_t);
    k_desc.scale_y_stride_bytes = MTP_PACKED16_K_QBLOCKS * sizeof(uint16_t);
    k_desc.scale_z_stride_bytes = uint64_t(kv_size) * k_desc.scale_y_stride_bytes;
    return k_desc;
}

static mtp_v4_144_tail_page_desc_v1 make_test_tail_page_desc(
        int32_t * block_table,
        uint32_t block_table_pages,
        uint32_t physical_pages,
        uint32_t logical_base_token,
        uint32_t logical_tokens,
        uint32_t valid_tail_tokens) {
    mtp_v4_144_tail_page_desc_v1 desc = {};
    desc.version = MTP_V4_144_TAIL_PAGE_ABI_VERSION;
    desc.abi_bytes = sizeof(mtp_v4_144_tail_page_desc_v1);
    desc.flags = MTP_V4_144_TAIL_FLAG_TAIL_ONLY | MTP_V4_144_TAIL_FLAG_ALIGNED_ONLY | MTP_V4_144_TAIL_FLAG_DEBUG_ABORTS;
    desc.page_tokens = MTP_V4_144_PAGE_TOKENS;
    desc.d = MTP_V4_144_D;
    desc.logical_base_token = logical_base_token;
    desc.logical_tokens = logical_tokens;
    desc.block_table = block_table;
    desc.block_table_pages = block_table_pages;
    desc.physical_pages = physical_pages;
    desc.valid_tail_tokens = valid_tail_tokens;
    desc.prefix_tokens = logical_base_token;
    desc.boundary_slot = logical_base_token & (MTP_V4_144_PAGE_TOKENS - 1u);
    desc.k_payload_base = reinterpret_cast<const void *>(uintptr_t(0x1000));
    desc.k_scale_base = reinterpret_cast<const void *>(uintptr_t(0x2000));
    desc.k_head_stride_bytes = uint64_t(40960) * MTP_PACKED16_K_ROW_BYTES;
    desc.k_page_stride_bytes = MTP_PACKED16_K_PAGE_BYTES;
    desc.v4_base = reinterpret_cast<const void *>(uintptr_t(0x3000));
    desc.v4_head_stride_bytes = uint64_t(40960) * MTP_V4_144_ROW_BYTES;
    desc.v4_page_stride_bytes = MTP_V4_144_PAGE_BYTES;
    desc.v4_batch_stride_bytes = desc.v4_head_stride_bytes * 4u;
    desc.kv_heads = 4;
    desc.batch = 1;
    desc.gqa_ratio = 6;
    desc.k_desc = make_test_packed16_row_desc(40960, 4);
    return desc;
}

static mtp_qblock_txn_lineage_v1 make_test_chain_lineage(uint32_t accepted_len) {
    mtp_qblock_txn_lineage_v1 lin = {};
    lin.version = MTP_QBLOCK_TXN_LINEAGE_ABI_VERSION;
    lin.abi_bytes = sizeof(mtp_qblock_txn_lineage_v1);
    lin.n_rows = accepted_len + 1u;
    lin.root_row = MTP_QBLOCK_TXN_ROOT_ROW;
    lin.flags = MTP_QBLOCK_TXN_LINEAGE_FLAG_IMPLICIT_CHAIN |
        MTP_QBLOCK_TXN_LINEAGE_FLAG_CONTIGUOUS_COMMIT |
        MTP_QBLOCK_TXN_LINEAGE_FLAG_CANONICAL_SLOT_ORDER;
    lin.accepted_leaf = uint8_t(accepted_len);
    lin.accepted_len = uint8_t(accepted_len);
    lin.accepted_mask = mtp_qblock_txn_lineage_chain_mask(lin.accepted_len);
    for (uint32_t row = 0; row < lin.n_rows; ++row) {
        lin.parent[row] = row == 0 ? 0 : uint8_t(row - 1u);
        lin.depth[row] = uint8_t(row);
        lin.kv_slot[row] = row == 0 ? 0 : uint8_t(row - 1u);
        lin.state_slot[row] = uint8_t(row);
    }
    return lin;
}

static void test_txn_tail_page_descriptor_and_lineage_contract() {
    int32_t block_table[3] = { 2, 0, 3 };
    mtp_v4_144_tail_page_desc_v1 desc = {};
    desc.version = MTP_V4_144_TAIL_PAGE_ABI_VERSION;
    desc.abi_bytes = sizeof(mtp_v4_144_tail_page_desc_v1);
    desc.flags = MTP_V4_144_TAIL_FLAG_TAIL_ONLY | MTP_V4_144_TAIL_FLAG_ALIGNED_ONLY | MTP_V4_144_TAIL_FLAG_DEBUG_ABORTS;
    desc.page_tokens = MTP_V4_144_PAGE_TOKENS;
    desc.d = MTP_V4_144_D;
    desc.logical_base_token = 64;
    desc.logical_tokens = 48;
    desc.block_table = block_table;
    desc.block_table_pages = 3;
    desc.physical_pages = 4;
    desc.valid_tail_tokens = 34;
    desc.prefix_tokens = 64;
    desc.boundary_slot = 0;
    desc.k_payload_base = reinterpret_cast<const void *>(uintptr_t(0x1000));
    desc.k_scale_base = reinterpret_cast<const void *>(uintptr_t(0x2000));
    desc.k_head_stride_bytes = uint64_t(40960) * MTP_PACKED16_K_ROW_BYTES;
    desc.k_page_stride_bytes = MTP_PACKED16_K_PAGE_BYTES;
    desc.v4_base = reinterpret_cast<const void *>(uintptr_t(0x3000));
    desc.v4_head_stride_bytes = uint64_t(40960) * MTP_V4_144_ROW_BYTES;
    desc.v4_page_stride_bytes = MTP_V4_144_PAGE_BYTES;
    desc.v4_batch_stride_bytes = desc.v4_head_stride_bytes * 4u;
    desc.kv_heads = 4;
    desc.batch = 1;
    desc.gqa_ratio = 6;
    desc.k_desc = make_test_packed16_row_desc(40960, 4);

    CHECK(mtp_v4_144_tail_page_validate_static(desc) == MTP_V4_144_TAIL_PAGE_OK);

    uint32_t physical_page = 0;
    uint32_t slot = 0;
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 64, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_OK);
    CHECK(physical_page == 2 && slot == 0);
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 79, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_OK);
    CHECK(physical_page == 2 && slot == 15);
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 80, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_OK);
    CHECK(physical_page == 0 && slot == 0);
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 97, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_OK);
    CHECK(physical_page == 3 && slot == 1);
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 63, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_TOKEN_NOT_VISIBLE);
    CHECK(mtp_v4_144_tail_page_logical_to_physical(desc, 98, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_TOKEN_NOT_VISIBLE);

    mtp_v4_144_tail_page_desc_v1 bad = desc;
    bad.valid_tail_tokens = 49;
    CHECK(mtp_v4_144_tail_page_validate_static(bad) == MTP_V4_144_TAIL_PAGE_BAD_VALID_TOKENS);
    bad = desc;
    bad.logical_base_token = 65;
    bad.boundary_slot = 1;
    CHECK(mtp_v4_144_tail_page_validate_static(bad) == MTP_V4_144_TAIL_PAGE_BAD_BOUNDARY);
    bad = desc;
    block_table[2] = 4;
    CHECK(mtp_v4_144_tail_page_logical_to_physical(bad, 96, &physical_page, &slot) == MTP_V4_144_TAIL_PAGE_BAD_PHYSICAL_PAGE);
    block_table[2] = 3;

    mtp_qblock_txn_lineage_v1 lin = {};
    lin.version = MTP_QBLOCK_TXN_LINEAGE_ABI_VERSION;
    lin.abi_bytes = sizeof(mtp_qblock_txn_lineage_v1);
    lin.n_rows = 5;
    lin.root_row = MTP_QBLOCK_TXN_ROOT_ROW;
    lin.flags = MTP_QBLOCK_TXN_LINEAGE_FLAG_IMPLICIT_CHAIN |
        MTP_QBLOCK_TXN_LINEAGE_FLAG_CONTIGUOUS_COMMIT |
        MTP_QBLOCK_TXN_LINEAGE_FLAG_CANONICAL_SLOT_ORDER;
    lin.accepted_leaf = 4;
    lin.accepted_len = 4;
    lin.accepted_mask = mtp_qblock_txn_lineage_chain_mask(lin.accepted_len);
    for (uint32_t row = 0; row < lin.n_rows; ++row) {
        lin.parent[row] = row == 0 ? 0 : uint8_t(row - 1u);
        lin.depth[row] = uint8_t(row);
        lin.kv_slot[row] = row == 0 ? 0 : uint8_t(row - 1u);
        lin.state_slot[row] = uint8_t(row);
    }
    CHECK(mtp_qblock_txn_lineage_validate_commit(lin, desc) == MTP_QBLOCK_TXN_LINEAGE_OK);
    CHECK(mtp_qblock_txn_lineage_final_state_slot(lin) == 4);
    CHECK(mtp_qblock_txn_lineage_new_valid_tail_tokens(desc, lin) == 38);

    mtp_qblock_txn_lineage_v1 lin_bad = lin;
    lin_bad.accepted_leaf = 3;
    CHECK(mtp_qblock_txn_lineage_validate_commit(lin_bad, desc) == MTP_QBLOCK_TXN_LINEAGE_BAD_ACCEPTED_LEN);
    lin_bad = lin;
    lin_bad.kv_slot[2] = 9;
    CHECK(mtp_qblock_txn_lineage_validate_commit(lin_bad, desc) == MTP_QBLOCK_TXN_LINEAGE_NOT_CANONICAL_SLOT_ORDER);
}

static void test_txn_tail_page_state_contract() {
    mtp_v4_144_tail_page_state_v1 state = {};
    CHECK(mtp_v4_144_tail_state_init(state, 64, 5) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_validate_static(state) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(state.free_count == 5);

    uint32_t p0 = 99, p1 = 99, p2 = 99;
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p1) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p2) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(p0 == 0 && p1 == 1 && p2 == 2);
    CHECK(state.free_count == 2);

    int32_t block_table[3] = { int32_t(p2), int32_t(p0), int32_t(p1) };
    mtp_v4_144_tail_page_desc_v1 desc = make_test_tail_page_desc(block_table, 3, 5, 64, 48, 0);
    mtp_qblock_txn_lineage_v1 lin = make_test_chain_lineage(4);
    CHECK(mtp_v4_144_tail_state_commit_lineage(state, desc, lin) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(state.valid_tail_tokens == 4);
    CHECK(state.logical_pages == 1);
    CHECK(state.block_table[0] == int32_t(p2));
    CHECK(state.owner[p2] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL));
    CHECK(state.page_valid_tokens[p2] == 4);
    CHECK(state.final_state_slot == 4);
    CHECK(state.owner[p0] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(state.owner[p1] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(state.free_count == 4);

    uint32_t physical_page = 0;
    uint32_t slot = 0;
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 67, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(physical_page == p2 && slot == 3);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 68, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE);

    const uint32_t accepted_len_cases[] = { 1, 3, 4 };
    for (uint32_t accepted_len : accepted_len_cases) {
        mtp_v4_144_tail_page_state_v1 len_state = {};
        CHECK(mtp_v4_144_tail_state_init(len_state, 320 + accepted_len * 16, 3) == MTP_V4_144_TAIL_STATE_OK);
        uint32_t page = 99;
        CHECK(mtp_v4_144_tail_state_alloc_page(len_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &page) == MTP_V4_144_TAIL_STATE_OK);
        int32_t len_block_table[1] = { int32_t(page) };
        mtp_v4_144_tail_page_desc_v1 len_desc = make_test_tail_page_desc(
            len_block_table, 1, 3, 320 + accepted_len * 16, 16, 0);
        CHECK(mtp_v4_144_tail_state_commit_lineage(len_state, len_desc, make_test_chain_lineage(accepted_len)) == MTP_V4_144_TAIL_STATE_OK);
        CHECK(len_state.valid_tail_tokens == accepted_len);
        CHECK(len_state.page_valid_tokens[page] == accepted_len);
        CHECK(len_state.final_state_slot == accepted_len);
        CHECK(len_state.free_count == 2);
    }

    mtp_v4_144_tail_page_state_v1 rollback_state = {};
    CHECK(mtp_v4_144_tail_state_init(rollback_state, 128, 4) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(rollback_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(rollback_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p1) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(rollback_state.free_count == 2);
    CHECK(mtp_v4_144_tail_state_rollback_txn_pages(rollback_state) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(rollback_state.free_count == 4);
    CHECK(rollback_state.owner[p0] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(rollback_state.owner[p1] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));

    mtp_v4_144_tail_page_state_v1 overflow_state = {};
    CHECK(mtp_v4_144_tail_state_init(overflow_state, 192, 2) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(overflow_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    int32_t overflow_table[1] = { int32_t(p0) };
    mtp_v4_144_tail_page_desc_v1 overflow_desc = make_test_tail_page_desc(overflow_table, 1, 2, 192, 4, 0);
    CHECK(mtp_v4_144_tail_state_commit_lineage(overflow_state, overflow_desc, make_test_chain_lineage(5)) == MTP_V4_144_TAIL_STATE_ACCEPTED_OVERFLOW);

    mtp_v4_144_tail_page_state_v1 free_page_state = {};
    CHECK(mtp_v4_144_tail_state_init(free_page_state, 256, 2) == MTP_V4_144_TAIL_STATE_OK);
    int32_t free_page_table[1] = { 1 };
    mtp_v4_144_tail_page_desc_v1 free_page_desc = make_test_tail_page_desc(free_page_table, 1, 2, 256, 16, 0);
    CHECK(mtp_v4_144_tail_state_commit_lineage(free_page_state, free_page_desc, make_test_chain_lineage(1)) == MTP_V4_144_TAIL_STATE_BAD_ACCEPTED_PAGE);
}

static void test_txn_tail_page_state_multipage_contract() {
    mtp_v4_144_tail_page_state_v1 state = {};
    CHECK(mtp_v4_144_tail_state_init(state, 512, 6) == MTP_V4_144_TAIL_STATE_OK);

    uint32_t p0 = 99, p1 = 99, p2 = 99, p3 = 99, p4 = 99;
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p1) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p2) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p3) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p4) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(p0 == 0 && p1 == 1 && p2 == 2 && p3 == 3 && p4 == 4);
    CHECK(state.free_count == 1);

    int32_t block_table[4] = { int32_t(p2), int32_t(p0), int32_t(p3), int32_t(p1) };
    mtp_v4_144_tail_page_desc_v1 desc = make_test_tail_page_desc(block_table, 4, 6, 512, 64, 14);
    CHECK(mtp_v4_144_tail_state_commit_lineage(state, desc, make_test_chain_lineage(4)) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(state.valid_tail_tokens == 18);
    CHECK(state.logical_pages == 2);
    CHECK(state.block_table[0] == int32_t(p2));
    CHECK(state.block_table[1] == int32_t(p0));
    CHECK(state.block_table[2] == MTP_V4_144_TAIL_PAGE_INVALID);
    CHECK(state.owner[p2] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL));
    CHECK(state.owner[p0] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_CANONICAL));
    CHECK(state.page_valid_tokens[p2] == MTP_V4_144_PAGE_TOKENS);
    CHECK(state.page_valid_tokens[p0] == 2);
    CHECK(state.owner[p1] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(state.owner[p3] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(state.owner[p4] == uint8_t(MTP_V4_144_TAIL_PAGE_OWNER_FREE));
    CHECK(state.free_count == 4);
    CHECK(mtp_v4_144_tail_state_validate_static(state) == MTP_V4_144_TAIL_STATE_OK);

    uint32_t physical_page = 0;
    uint32_t slot = 0;
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 512, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(physical_page == p2 && slot == 0);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 527, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(physical_page == p2 && slot == 15);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 528, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(physical_page == p0 && slot == 0);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 529, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(physical_page == p0 && slot == 1);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 530, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(state, 511, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE);

    mtp_v4_144_tail_page_state_v1 duplicate_state = {};
    CHECK(mtp_v4_144_tail_state_init(duplicate_state, 640, 3) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(duplicate_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    int32_t duplicate_table[2] = { int32_t(p0), int32_t(p0) };
    mtp_v4_144_tail_page_desc_v1 duplicate_desc = make_test_tail_page_desc(duplicate_table, 2, 3, 640, 32, 14);
    CHECK(mtp_v4_144_tail_state_commit_lineage(duplicate_state, duplicate_desc, make_test_chain_lineage(4)) != MTP_V4_144_TAIL_STATE_OK);

    mtp_v4_144_tail_page_state_v1 invalid_state = {};
    CHECK(mtp_v4_144_tail_state_init(invalid_state, 704, 3) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_alloc_page(invalid_state, MTP_V4_144_TAIL_PAGE_OWNER_TXN, &p0) == MTP_V4_144_TAIL_STATE_OK);
    int32_t invalid_table[1] = { 3 };
    mtp_v4_144_tail_page_desc_v1 invalid_desc = make_test_tail_page_desc(invalid_table, 1, 3, 704, 16, 0);
    CHECK(mtp_v4_144_tail_state_commit_lineage(invalid_state, invalid_desc, make_test_chain_lineage(1)) == MTP_V4_144_TAIL_STATE_BAD_BLOCK_TABLE);

    mtp_v4_144_tail_page_state_v1 empty_state = {};
    CHECK(mtp_v4_144_tail_state_init(empty_state, 768, 2) == MTP_V4_144_TAIL_STATE_OK);
    CHECK(mtp_v4_144_tail_state_logical_to_physical(empty_state, 768, &physical_page, &slot) == MTP_V4_144_TAIL_STATE_TOKEN_NOT_VISIBLE);
}

static void test_llama_mtp_qblock_paged_state_contract() {
    llama_mtp_qblock_paged_state_v1 state = {};
    CHECK(llama_mtp_qblock_paged_state_init(state, 1024, 6, 7) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_validate_static(state) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(state.active == 1);
    CHECK(state.generation == 7);
    CHECK(state.free_count == 6);

    uint32_t p0 = 99, p1 = 99, p2 = 99, p3 = 99, p4 = 99;
    CHECK(llama_mtp_qblock_paged_state_alloc_page(state, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p0) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(state, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p1) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(state, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p2) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(state, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p3) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(state, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p4) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(p0 == 0 && p1 == 1 && p2 == 2 && p3 == 3 && p4 == 4);

    int32_t accepted_table[4] = { int32_t(p2), int32_t(p0), int32_t(p3), int32_t(p1) };
    CHECK(llama_mtp_qblock_paged_state_commit_pages(state, 18, accepted_table, 4, 5) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(state.valid_tail_tokens == 18);
    CHECK(state.logical_pages == 2);
    CHECK(state.generation == 8);
    CHECK(state.block_table[0] == int32_t(p2));
    CHECK(state.block_table[1] == int32_t(p0));
    CHECK(state.block_table[2] == LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_PAGE);
    CHECK(state.owner[p2] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL));
    CHECK(state.owner[p0] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL));
    CHECK(state.page_valid_tokens[p2] == LLAMA_MTP_QBLOCK_PAGED_STATE_PAGE_TOKENS);
    CHECK(state.page_valid_tokens[p0] == 2);
    CHECK(state.owner[p1] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE));
    CHECK(state.owner[p3] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE));
    CHECK(state.owner[p4] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_FREE));
    CHECK(state.free_count == 4);

    uint32_t physical_page = 0;
    uint32_t slot = 0;
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(state, 1024, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == p2 && slot == 0);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(state, 1039, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == p2 && slot == 15);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(state, 1040, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == p0 && slot == 0);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(state, 1041, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == p0 && slot == 1);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(state, 1042, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE);

    llama_mtp_qblock_paged_consumer_map_v1 map = {};
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(state, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(map.logical_base_token == 1024);
    CHECK(map.valid_tail_tokens == 18);
    CHECK(map.block_table_pages == 2);
    CHECK(map.block_table[0] == int32_t(p2));
    CHECK(map.block_table[1] == int32_t(p0));
    CHECK(map.block_table[0] != int32_t(p1) && map.block_table[1] != int32_t(p1));
    CHECK(map.block_table[0] != int32_t(p3) && map.block_table[1] != int32_t(p3));
    CHECK(map.generation == state.generation);

    llama_mtp_qblock_paged_state_v1 imported = {};
    int32_t absolute_pages[2] = { 486, 488 };
    CHECK(llama_mtp_qblock_paged_state_import_committed_pages(imported, 7776, 2560, 18, absolute_pages, 2, 5, 123) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(imported.physical_pages == 2560);
    CHECK(imported.managed_pages == 2);
    CHECK(imported.owner[0] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL));
    CHECK(imported.owner[1] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL));
    CHECK(imported.block_table[0] == 486);
    CHECK(imported.block_table[1] == 488);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(imported, 7776, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == 486 && slot == 0);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(imported, 7793, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == 488 && slot == 1);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(imported, 7794, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE);
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(imported, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(map.physical_pages == 2560);
    CHECK(map.block_table_pages == 2);
    CHECK(map.block_table[0] == 486 && map.block_table[1] == 488);

    uint32_t claimed_slot = 99;
    CHECK(llama_mtp_qblock_paged_state_claim_txn_page(imported, 489, &claimed_slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(claimed_slot == 2);
    CHECK(imported.managed_pages == 3);
    CHECK(imported.managed_physical_page[claimed_slot] == 489);
    CHECK(imported.owner[claimed_slot] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN));
    CHECK(llama_mtp_qblock_paged_state_claim_txn_page(imported, 486, &claimed_slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_DUP_MANAGED_PAGE);
    int32_t absolute_commit_pages[3] = { 486, 488, 489 };
    CHECK(llama_mtp_qblock_paged_state_commit_pages(imported, 33, absolute_commit_pages, 3, 7) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(imported.valid_tail_tokens == 33);
    CHECK(imported.logical_pages == 3);
    CHECK(imported.generation == 124);
    CHECK(imported.block_table[2] == 489);
    CHECK(imported.owner[2] == uint8_t(LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_CANONICAL));
    CHECK(imported.page_valid_tokens[2] == 1);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(imported, 7808, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == 489 && slot == 0);
    CHECK(llama_mtp_qblock_paged_state_claim_txn_page(imported, 490, &claimed_slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(imported.managed_pages == 4);
    CHECK(llama_mtp_qblock_paged_state_rollback_txn_pages(imported) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(imported.managed_pages == 3);
    CHECK(llama_mtp_qblock_paged_state_find_managed_slot(imported, 490) == LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT);

    llama_mtp_qblock_paged_state_v1 imported_with_reject = {};
    int32_t one_absolute_page[1] = { 700 };
    CHECK(llama_mtp_qblock_paged_state_import_committed_pages(imported_with_reject, 8192, 2560, 16, one_absolute_page, 1, 1, 200) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_claim_txn_page(imported_with_reject, 701, nullptr) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_claim_txn_page(imported_with_reject, 702, nullptr) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    int32_t accept_one_reject_one[2] = { 700, 701 };
    CHECK(llama_mtp_qblock_paged_state_commit_pages(imported_with_reject, 18, accept_one_reject_one, 2, 2) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(imported_with_reject.managed_pages == 2);
    CHECK(llama_mtp_qblock_paged_state_find_managed_slot(imported_with_reject, 701) < imported_with_reject.managed_pages);
    CHECK(llama_mtp_qblock_paged_state_find_managed_slot(imported_with_reject, 702) == LLAMA_MTP_QBLOCK_PAGED_STATE_INVALID_SLOT);

    llama_mtp_qblock_paged_state_v1 unaligned_import = {};
    int32_t scratch_overlay_page[1] = { 2499 };
    CHECK(llama_mtp_qblock_paged_state_import_committed_pages(unaligned_import, 7783, 2560, 5, scratch_overlay_page, 1, 5, 300) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(unaligned_import.logical_base_token == 7783);
    CHECK(unaligned_import.valid_tail_tokens == 5);
    CHECK(unaligned_import.logical_pages == 1);
    CHECK(unaligned_import.managed_pages == 1);
    CHECK(unaligned_import.managed_physical_page[0] == 2499);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(unaligned_import, 7783, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == 2499 && slot == 0);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(unaligned_import, 7787, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(physical_page == 2499 && slot == 4);
    CHECK(llama_mtp_qblock_paged_state_logical_to_physical(unaligned_import, 7788, &physical_page, &slot) == LLAMA_MTP_QBLOCK_PAGED_STATE_TOKEN_NOT_VISIBLE);
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(unaligned_import, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(map.logical_base_token == 7783);
    CHECK(map.valid_tail_tokens == 5);
    CHECK(map.block_table[0] == 2499);

    int32_t duplicate_absolute_pages[2] = { 486, 486 };
    CHECK(llama_mtp_qblock_paged_state_import_committed_pages(imported, 7776, 2560, 18, duplicate_absolute_pages, 2, 5, 124) == LLAMA_MTP_QBLOCK_PAGED_STATE_BAD_BLOCK_TABLE);

    llama_mtp_qblock_paged_state_v1 empty = {};
    CHECK(llama_mtp_qblock_paged_state_init(empty, 2048, 2, 1) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(empty, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES);

    llama_mtp_qblock_paged_state_v1 too_large = {};
    CHECK(llama_mtp_qblock_paged_state_init(too_large, 3072, 6, 1) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    int32_t too_large_table[5] = {};
    for (uint32_t i = 0; i < 5; ++i) {
        uint32_t page = 99;
        CHECK(llama_mtp_qblock_paged_state_alloc_page(too_large, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &page) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
        too_large_table[i] = int32_t(page);
    }
    CHECK(llama_mtp_qblock_paged_state_commit_pages(too_large, 65, too_large_table, 5, 0) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(too_large.logical_pages == 5);
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(too_large, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_MAP_TOO_LARGE);

    llama_mtp_qblock_paged_state_v1 rollback = {};
    CHECK(llama_mtp_qblock_paged_state_init(rollback, 4096, 3, 1) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(rollback, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p0) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(llama_mtp_qblock_paged_state_alloc_page(rollback, LLAMA_MTP_QBLOCK_PAGED_PAGE_OWNER_TXN, &p1) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(rollback.free_count == 1);
    CHECK(llama_mtp_qblock_paged_state_rollback_txn_pages(rollback) == LLAMA_MTP_QBLOCK_PAGED_STATE_OK);
    CHECK(rollback.free_count == 3);
    CHECK(llama_mtp_qblock_paged_state_export_consumer_map(rollback, &map) == LLAMA_MTP_QBLOCK_PAGED_STATE_NO_VISIBLE_PAGES);
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

    dp16_packed_i8_desc_v1 k_desc = make_test_packed16_row_desc(kv_size, 4);
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
    test_txn_tail_page_descriptor_and_lineage_contract();
    test_txn_tail_page_state_contract();
    test_txn_tail_page_state_multipage_contract();
    test_llama_mtp_qblock_paged_state_contract();
    test_txn_tail_page_stage_contract();
    test_randomized_qk_and_attention();
    std::puts("test-pdmq-k-formats: PASS");
    return 0;
}
