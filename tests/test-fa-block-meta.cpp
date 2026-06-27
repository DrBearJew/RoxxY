#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../ggml/src/ggml-cuda/dot4-packed16/fa-block-meta.cuh"

#define CHECK(COND) do { \
    if (!(COND)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
        std::abort(); \
    } \
} while (0)

static void test_validate_pure_causal() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(
        /*valid_n_kv=*/512,
        /*q_tokens=*/512,
        /*q_offset=*/0,
        /*q_block_tokens=*/64,
        /*k_block_tokens=*/16);
    CHECK(ggml_cuda_fa_block_meta_validate_static(meta) == GGML_CUDA_FA_BLOCK_META_OK);

    auto bad = meta;
    bad.version = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_BAD_VERSION);

    bad = meta;
    bad.abi_bytes = sizeof(ggml_cuda_fa_block_meta_v1) - 4;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_BAD_ABI_BYTES);

    bad = meta;
    bad.flags = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_MISSING_CAUSAL);

    bad = meta;
    bad.k_block_tokens = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS);
}

static void test_pure_causal_block_plan() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(512, 512, 0, 64, 16);
    ggml_cuda_fa_block_plan_v1 plan = {};

    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(meta, 0, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(plan.q_begin == 0);
    CHECK(plan.q_end == 64);
    CHECK(plan.k_block_end == 32);
    CHECK(plan.full_begin == 0);
    CHECK(plan.full_end == 0);
    CHECK(plan.edge_back_begin == 0);
    CHECK(plan.edge_back_end == 4);
    CHECK(plan.skipped_begin == 4);
    CHECK(plan.skipped_end == 32);

    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(meta, 448, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(plan.q_begin == 448);
    CHECK(plan.q_end == 512);
    CHECK(plan.k_block_end == 32);
    CHECK(plan.full_begin == 0);
    CHECK(plan.full_end == 27);
    CHECK(plan.edge_back_begin == 27);
    CHECK(plan.edge_back_end == 32);
    CHECK(plan.skipped_begin == 32);
    CHECK(plan.skipped_end == 32);

    auto prefix = ggml_cuda_fa_block_meta_make_pure_causal_contig(1536, 512, 1024, 64, 16);
    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(prefix, 0, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(plan.k_block_end == 96);
    CHECK(plan.full_end == 63);
    CHECK(plan.edge_back_begin == 63);
    CHECK(plan.edge_back_end == 68);
    CHECK(plan.skipped_begin == 68);
    CHECK(plan.skipped_end == 96);
}

static void test_clip_k_range_to_visible() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(512, 512, 0, 64, 16);
    ggml_cuda_fa_block_plan_v1 plan = {};
    uint32_t clipped_end = 0;

    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(meta, 0, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(ggml_cuda_fa_block_meta_clip_k_range_to_visible(plan, 0, 32, &clipped_end) == 28);
    CHECK(clipped_end == 4);
    CHECK(ggml_cuda_fa_block_meta_clip_k_range_to_visible(plan, 2, 8, &clipped_end) == 4);
    CHECK(clipped_end == 4);
    CHECK(ggml_cuda_fa_block_meta_clip_k_range_to_visible(plan, 8, 16, &clipped_end) == 8);
    CHECK(clipped_end == 8);

    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(meta, 448, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(ggml_cuda_fa_block_meta_clip_k_range_to_visible(plan, 0, 32, &clipped_end) == 0);
    CHECK(clipped_end == 32);

    auto prefix = ggml_cuda_fa_block_meta_make_pure_causal_contig(1536, 512, 1024, 64, 16);
    CHECK(ggml_cuda_fa_block_meta_plan_pure_causal(prefix, 0, 64, &plan) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(ggml_cuda_fa_block_meta_clip_k_range_to_visible(plan, 0, 96, &clipped_end) == 28);
    CHECK(clipped_end == 68);
}

static ggml_cuda_fa_block_meta_v1 make_paged_meta(const int32_t * table, uint32_t table_pages) {
    ggml_cuda_fa_block_meta_v1 meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(
        /*valid_n_kv=*/40,
        /*q_tokens=*/16,
        /*q_offset=*/24,
        /*q_block_tokens=*/16,
        /*k_block_tokens=*/16);
    meta.mode = GGML_CUDA_FA_BLOCK_META_MODE_PAGED_CAUSAL;
    meta.flags |= GGML_CUDA_FA_BLOCK_META_FLAG_HAS_BLOCK_TABLE |
        GGML_CUDA_FA_BLOCK_META_FLAG_HAS_LAST_PAGE_TOKENS;
    meta.page_kind = GGML_CUDA_FA_BLOCK_PAGE_KIND_BLOCK_TABLE;
    meta.page_tokens = 16;
    meta.logical_base_token = 100;
    meta.logical_tokens = 40;
    meta.block_table = table;
    meta.block_table_pages = table_pages;
    meta.physical_pages = 16;
    meta.last_page_tokens = 8;
    return meta;
}

static void test_paged_block_table_validation() {
    const int32_t table[3] = { 4, 7, 9 };
    auto meta = make_paged_meta(table, 3);
    CHECK(ggml_cuda_fa_block_meta_validate_static(meta) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(ggml_cuda_fa_block_meta_validate_block_table(meta) == GGML_CUDA_FA_BLOCK_META_OK);

    auto bad = meta;
    bad.block_table = nullptr;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_MISSING_BLOCK_TABLE);

    bad = meta;
    bad.block_table_pages = 2;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TABLE_PAGES);

    bad = meta;
    bad.last_page_tokens = 16;
    CHECK(ggml_cuda_fa_block_meta_validate_static(bad) == GGML_CUDA_FA_BLOCK_META_BAD_LAST_PAGE_TOKENS);

    const int32_t dup[3] = { 4, 7, 7 };
    bad = make_paged_meta(dup, 3);
    CHECK(ggml_cuda_fa_block_meta_validate_block_table(bad) == GGML_CUDA_FA_BLOCK_META_DUPLICATE_PHYSICAL_PAGE);

    const int32_t oob[3] = { 4, 17, 9 };
    bad = make_paged_meta(oob, 3);
    CHECK(ggml_cuda_fa_block_meta_validate_block_table(bad) == GGML_CUDA_FA_BLOCK_META_BAD_PHYSICAL_PAGE);
}

static void test_plan_counts() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(512, 512, 0, 64, 16);
    ggml_cuda_fa_block_plan_counts_v1 counts = {};
    CHECK(ggml_cuda_fa_block_meta_plan_all_counts(meta, &counts) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(counts.version == GGML_CUDA_FA_BLOCK_PLAN_COUNTS_VERSION);
    CHECK(counts.q_blocks == 8);
    CHECK(counts.k_blocks == 32);
    CHECK(counts.full_tiles == 105);
    CHECK(counts.edge_front_tiles == 0);
    CHECK(counts.edge_back_tiles == 39);
    CHECK(counts.skipped_tiles == 112);
    CHECK(counts.padded_tile_refs == 0);

    auto prefix = ggml_cuda_fa_block_meta_make_pure_causal_contig(1536, 512, 1024, 64, 16);
    CHECK(ggml_cuda_fa_block_meta_plan_all_counts(prefix, &counts) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(counts.q_blocks == 8);
    CHECK(counts.k_blocks == 96);
    CHECK(counts.full_tiles == 616);
    CHECK(counts.edge_back_tiles == 40);
    CHECK(counts.skipped_tiles == 112);
    CHECK(counts.padded_tile_refs == 0);

    auto padded = ggml_cuda_fa_block_meta_make_pure_causal_contig(40, 16, 24, 16, 16);
    CHECK(ggml_cuda_fa_block_meta_plan_all_counts(padded, &counts) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(counts.q_blocks == 1);
    CHECK(counts.k_blocks == 3);
    CHECK(counts.full_tiles == 1);
    CHECK(counts.edge_back_tiles == 2);
    CHECK(counts.skipped_tiles == 0);
    CHECK(counts.padded_tile_refs == 1);
}

static void test_tile_full_visible() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(512, 512, 0, 64, 16);
    CHECK(ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 0, 16));
    CHECK(ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 416, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 432, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 512, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_skipped(meta, 448, 64, 416, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_skipped(meta, 448, 64, 432, 16));
    CHECK(ggml_cuda_fa_block_meta_tile_is_skipped(meta, 0, 64, 64, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 8, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_full_visible(meta, 448, 64, 0, 8));

    auto prefix = ggml_cuda_fa_block_meta_make_pure_causal_contig(1536, 512, 1024, 64, 16);
    CHECK(ggml_cuda_fa_block_meta_tile_is_full_visible(prefix, 0, 64, 0, 16));
    CHECK(ggml_cuda_fa_block_meta_tile_is_full_visible(prefix, 0, 64, 992, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_full_visible(prefix, 0, 64, 1008, 16));
    CHECK(!ggml_cuda_fa_block_meta_tile_is_skipped(prefix, 0, 64, 1008, 16));
    CHECK(ggml_cuda_fa_block_meta_tile_is_skipped(prefix, 0, 64, 1088, 16));
}

static void test_caps_validation() {
    auto meta = ggml_cuda_fa_block_meta_make_pure_causal_contig(1024, 512, 512, 64, 16);
    ggml_cuda_fa_block_meta_caps_v1 caps = {};
    caps.version = GGML_CUDA_FA_BLOCK_META_CAPS_VERSION;
    caps.abi_bytes = sizeof(ggml_cuda_fa_block_meta_caps_v1);
    caps.supported_modes = ggml_cuda_fa_block_meta_mode_bit(GGML_CUDA_FA_BLOCK_META_MODE_PURE_CAUSAL_CONTIG);
    caps.supported_flags = GGML_CUDA_FA_BLOCK_META_FLAG_CAUSAL;
    caps.page_kinds = GGML_CUDA_FA_BLOCK_PAGE_KIND_BIT_NONE;
    caps.min_n_kv_for_null_mask = 1024;

    CHECK(ggml_cuda_fa_block_meta_validate_caps(meta, caps, true) == GGML_CUDA_FA_BLOCK_META_OK);

    auto too_short = meta;
    too_short.valid_n_kv = 512;
    CHECK(ggml_cuda_fa_block_meta_validate_caps(too_short, caps, true) == GGML_CUDA_FA_BLOCK_META_BELOW_MIN_N_KV);
    CHECK(ggml_cuda_fa_block_meta_validate_caps(too_short, caps, false) == GGML_CUDA_FA_BLOCK_META_OK);

    auto bad_caps = caps;
    bad_caps.supported_modes = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_caps(meta, bad_caps, true) == GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_MODE);

    bad_caps = caps;
    bad_caps.supported_flags = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_caps(meta, bad_caps, true) == GGML_CUDA_FA_BLOCK_META_UNSUPPORTED_FLAGS);

    bad_caps = caps;
    bad_caps.version = 0;
    CHECK(ggml_cuda_fa_block_meta_validate_caps(meta, bad_caps, true) == GGML_CUDA_FA_BLOCK_META_BAD_CAPS_VERSION);
}

static void test_logical_to_physical() {
    const int32_t table[3] = { 4, 7, 9 };
    auto meta = make_paged_meta(table, 3);
    uint32_t physical = 0;
    uint32_t offset = 0;
    uint32_t valid = 0;

    CHECK(ggml_cuda_fa_block_meta_logical_to_physical(meta, 100, &physical, &offset, &valid) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical == 4);
    CHECK(offset == 0);
    CHECK(valid == 16);

    CHECK(ggml_cuda_fa_block_meta_logical_to_physical(meta, 132, &physical, &offset, &valid) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical == 9);
    CHECK(offset == 0);
    CHECK(valid == 8);

    CHECK(ggml_cuda_fa_block_meta_logical_to_physical(meta, 139, &physical, &offset, &valid) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical == 9);
    CHECK(offset == 7);
    CHECK(valid == 8);

    CHECK(ggml_cuda_fa_block_meta_logical_to_physical(meta, 140, &physical, &offset, &valid) == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE);
    CHECK(ggml_cuda_fa_block_meta_logical_to_physical(meta, 99, &physical, &offset, &valid) == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE);
}

static void test_logical_tile_to_physical_contig() {
    const int32_t table[3] = { 4, 7, 9 };
    auto meta = make_paged_meta(table, 3);
    uint32_t physical_begin = 0;
    uint32_t physical_page = 0;
    uint32_t page_offset = 0;

    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 100, 16, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical_page == 4);
    CHECK(page_offset == 0);
    CHECK(physical_begin == 64);

    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 120, 8, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical_page == 7);
    CHECK(page_offset == 4);
    CHECK(physical_begin == 116);

    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 132, 8, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(physical_page == 9);
    CHECK(page_offset == 0);
    CHECK(physical_begin == 144);

    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 112, 17, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE);
    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 132, 9, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE);
    CHECK(ggml_cuda_fa_block_meta_logical_tile_to_physical_contig(meta, 99, 1, &physical_begin, &physical_page, &page_offset) == GGML_CUDA_FA_BLOCK_META_TOKEN_NOT_VISIBLE);
}

static void test_describe_logical_tile() {
    const int32_t table[3] = { 4, 7, 9 };
    auto meta = make_paged_meta(table, 3);
    meta.logical_base_token = 96;
    meta.logical_tokens = 44;
    meta.last_page_tokens = 12;
    meta.q_tokens = 40;
    meta.valid_n_kv = 140;
    meta.q_offset = 100;
    meta.q_block_tokens = 16;
    meta.k_block_tokens = 16;

    ggml_cuda_fa_block_tile_desc_v1 desc = {};
    CHECK(ggml_cuda_fa_block_meta_describe_logical_tile(meta, 32, 8, 96, 16, &desc) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(desc.version == GGML_CUDA_FA_BLOCK_TILE_DESC_VERSION);
    CHECK(desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_FULL);
    CHECK(desc.physical_page == 4);
    CHECK(desc.page_offset == 0);
    CHECK(desc.physical_k_begin == 64);

    CHECK(ggml_cuda_fa_block_meta_describe_logical_tile(meta, 0, 16, 96, 16, &desc) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_EDGE);
    CHECK(desc.physical_page == 4);
    CHECK(desc.page_offset == 0);

    CHECK(ggml_cuda_fa_block_meta_describe_logical_tile(meta, 0, 16, 128, 16, &desc) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(desc.tile_class == GGML_CUDA_FA_BLOCK_TILE_SKIPPED);
    CHECK(desc.physical_k_begin == GGML_CUDA_FA_BLOCK_INVALID);

    CHECK(ggml_cuda_fa_block_meta_describe_logical_tile(meta, 32, 8, 112, 17, &desc) == GGML_CUDA_FA_BLOCK_META_BAD_BLOCK_TOKENS);

    ggml_cuda_fa_block_tile_desc_counts_v1 counts = {};
    CHECK(ggml_cuda_fa_block_meta_describe_all_counts(meta, &counts) == GGML_CUDA_FA_BLOCK_META_OK);
    CHECK(counts.version == GGML_CUDA_FA_BLOCK_TILE_DESC_COUNTS_VERSION);
    CHECK(counts.described_tiles == 7);
    CHECK(counts.full_tiles == 2);
    CHECK(counts.edge_tiles == 4);
    CHECK(counts.skipped_tiles == 1);
    CHECK(counts.physical_tiles == 6);
    CHECK(counts.physical_rejects == 2);
}

int main() {
    test_validate_pure_causal();
    test_pure_causal_block_plan();
    test_clip_k_range_to_visible();
    test_plan_counts();
    test_tile_full_visible();
    test_caps_validation();
    test_paged_block_table_validation();
    test_logical_to_physical();
    test_logical_tile_to_physical_contig();
    test_describe_logical_tile();
    return 0;
}
