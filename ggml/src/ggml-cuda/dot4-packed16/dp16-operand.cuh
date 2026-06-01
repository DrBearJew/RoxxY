// dp16-operand.cuh — operand roles/storage/layouts for the packed16/DOT4 substrate
#pragma once

#include "dp16-common.cuh"

// DP16 operand descriptors deliberately separate physical layout from how that
// data is produced or consumed. A packed16_i32_scaled payload may be a K cache,
// a persistent projection weight, or another future operand source.

enum dp16_operand_role {
    DP16_OPERAND_UNKNOWN,
    DP16_OPERAND_ACTIVATION,
    DP16_OPERAND_WEIGHT,
    DP16_OPERAND_K_CACHE,
    DP16_OPERAND_V_CACHE,
    DP16_OPERAND_OUTPUT,
};

enum dp16_storage {
    DP16_STORAGE_UNKNOWN,
    DP16_STORAGE_TRANSIENT_TILE,
    DP16_STORAGE_PERSISTENT_CACHE,
    DP16_STORAGE_PERSISTENT_WEIGHT,
    DP16_STORAGE_LDS_TILE,
    DP16_STORAGE_OUTPUT,
};

enum dp16_layout {
    DP16_LAYOUT_UNKNOWN,
    DP16_LAYOUT_F32,
    DP16_LAYOUT_F16,
    DP16_LAYOUT_Q8_BLOCK32,
    DP16_LAYOUT_PACKED_I8X4_I32,
    DP16_LAYOUT_PACKED16_I32_SCALED,
    DP16_LAYOUT_Q4_0_BLOCK32,
    DP16_LAYOUT_Q4_K,
};

struct dp16_operand_desc {
    dp16_operand_role role;
    dp16_storage storage;
    dp16_layout layout;

    ggml_type ggml_type;

    int64_t rows;
    int64_t cols;
    int64_t stride_row;
    int64_t stride_col;

    int scale_block;
    int pack_lanes;

    bool signed_i8;
    bool has_zero_point;
    bool has_min_bias;
    bool has_sideband_sums;
};

static inline const char * dp16_operand_role_name(const dp16_operand_role role) {
    switch (role) {
        case DP16_OPERAND_UNKNOWN:    return "unknown";
        case DP16_OPERAND_ACTIVATION: return "activation";
        case DP16_OPERAND_WEIGHT:     return "weight";
        case DP16_OPERAND_K_CACHE:    return "k_cache";
        case DP16_OPERAND_V_CACHE:    return "v_cache";
        case DP16_OPERAND_OUTPUT:     return "output";
        default:                      return "unknown";
    }
}

static inline const char * dp16_storage_name(const dp16_storage storage) {
    switch (storage) {
        case DP16_STORAGE_UNKNOWN:           return "unknown";
        case DP16_STORAGE_TRANSIENT_TILE:    return "transient_tile";
        case DP16_STORAGE_PERSISTENT_CACHE:  return "persistent_cache";
        case DP16_STORAGE_PERSISTENT_WEIGHT: return "persistent_weight";
        case DP16_STORAGE_LDS_TILE:          return "lds_tile";
        case DP16_STORAGE_OUTPUT:            return "output";
        default:                             return "unknown";
    }
}

static inline const char * dp16_layout_name(const dp16_layout layout) {
    switch (layout) {
        case DP16_LAYOUT_UNKNOWN:             return "unknown";
        case DP16_LAYOUT_F32:                 return "f32";
        case DP16_LAYOUT_F16:                 return "f16";
        case DP16_LAYOUT_Q8_BLOCK32:          return "q8_block32";
        case DP16_LAYOUT_PACKED_I8X4_I32:     return "packed_i8x4_i32";
        case DP16_LAYOUT_PACKED16_I32_SCALED: return "packed16_i32_scaled";
        case DP16_LAYOUT_Q4_0_BLOCK32:        return "q4_0_block32";
        case DP16_LAYOUT_Q4_K:                return "q4_K";
        default:                              return "unknown";
    }
}

static inline dp16_layout dp16_layout_from_ggml_type(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return DP16_LAYOUT_F32;
        case GGML_TYPE_F16:  return DP16_LAYOUT_F16;
        case GGML_TYPE_Q8_0: return DP16_LAYOUT_Q8_BLOCK32;
        case GGML_TYPE_Q8_1: return DP16_LAYOUT_Q8_BLOCK32;
        case GGML_TYPE_Q4_0: return DP16_LAYOUT_Q4_0_BLOCK32;
        case GGML_TYPE_Q4_K: return DP16_LAYOUT_Q4_K;
        default:             return DP16_LAYOUT_UNKNOWN;
    }
}

static inline int dp16_layout_default_scale_block(const dp16_layout layout) {
    switch (layout) {
        case DP16_LAYOUT_Q8_BLOCK32:
        case DP16_LAYOUT_PACKED16_I32_SCALED:
        case DP16_LAYOUT_Q4_0_BLOCK32:
            return 32;
        case DP16_LAYOUT_Q4_K:
            return QK_K;
        default:
            return 0;
    }
}

static inline int dp16_layout_default_pack_lanes(const dp16_layout layout) {
    switch (layout) {
        case DP16_LAYOUT_Q8_BLOCK32:
        case DP16_LAYOUT_PACKED_I8X4_I32:
        case DP16_LAYOUT_PACKED16_I32_SCALED:
            return 4;
        case DP16_LAYOUT_F32:
        case DP16_LAYOUT_F16:
            return 1;
        default:
            return 0;
    }
}

static inline bool dp16_layout_is_signed_i8(const dp16_layout layout) {
    return layout == DP16_LAYOUT_Q8_BLOCK32 ||
           layout == DP16_LAYOUT_PACKED_I8X4_I32 ||
           layout == DP16_LAYOUT_PACKED16_I32_SCALED;
}

static inline bool dp16_layout_is_q4(const dp16_layout layout) {
    return layout == DP16_LAYOUT_Q4_0_BLOCK32 || layout == DP16_LAYOUT_Q4_K;
}

static inline bool dp16_layout_is_packed16(const dp16_layout layout) {
    return layout == DP16_LAYOUT_PACKED16_I32_SCALED;
}

static inline dp16_operand_desc dp16_operand_desc_make(
        const dp16_operand_role role,
        const dp16_storage storage,
        const dp16_layout layout,
        const ggml_type ggml_type,
        const int64_t rows,
        const int64_t cols,
        const int64_t stride_row = 0,
        const int64_t stride_col = 0) {
    dp16_operand_desc desc = {};
    desc.role = role;
    desc.storage = storage;
    desc.layout = layout;
    desc.ggml_type = ggml_type;
    desc.rows = rows;
    desc.cols = cols;
    desc.stride_row = stride_row;
    desc.stride_col = stride_col;
    desc.scale_block = dp16_layout_default_scale_block(layout);
    desc.pack_lanes = dp16_layout_default_pack_lanes(layout);
    desc.signed_i8 = dp16_layout_is_signed_i8(layout);
    desc.has_zero_point = false;
    desc.has_min_bias = layout == DP16_LAYOUT_Q4_K;
    desc.has_sideband_sums = false;
    return desc;
}

static inline dp16_operand_desc dp16_operand_desc_from_type(
        const dp16_operand_role role,
        const dp16_storage storage,
        const ggml_type ggml_type,
        const int64_t rows,
        const int64_t cols,
        const int64_t stride_row = 0,
        const int64_t stride_col = 0) {
    return dp16_operand_desc_make(role, storage, dp16_layout_from_ggml_type(ggml_type),
            ggml_type, rows, cols, stride_row, stride_col);
}

static inline dp16_operand_desc dp16_operand_unknown() {
    return dp16_operand_desc_make(DP16_OPERAND_UNKNOWN, DP16_STORAGE_UNKNOWN,
            DP16_LAYOUT_UNKNOWN, GGML_TYPE_COUNT, 0, 0);
}
