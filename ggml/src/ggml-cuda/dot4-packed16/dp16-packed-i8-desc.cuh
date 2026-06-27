// dp16-packed-i8-desc.cuh — explicit packed-i8x16 vector/address ABI
#pragma once

#ifndef DP16_PACKED_I8_DESC_HOST_ONLY
#include "../common.cuh"
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

#include <cstddef>
#include <cstdint>

// One packed-i8 vector is the fundamental DP16/DOT4 movement grain used by the
// packed16 K-cache and future packed-weight projectors: four address-ordered
// i32 words, with byte lanes in little-endian/least-significant-byte-first
// order inside each word.
static constexpr uint32_t DP16_PACKED_I8_DESC_VERSION = 1;
static constexpr uint32_t DP16_PACKED_I8X16_LANES = 16;
static constexpr uint32_t DP16_PACKED_I8X16_WORDS = 4;
static constexpr uint32_t DP16_PACKED_I8X16_BYTES = 16;
static constexpr uint32_t DP16_PACKED_I8_WORD_BYTES = 4;
static constexpr uint32_t DP16_PACKED_I8_PAGE16_TOKENS = 16;

struct alignas(DP16_PACKED_I8X16_BYTES) dp16_i8x16_words {
    int32_t word[DP16_PACKED_I8X16_WORDS];
};

enum dp16_packed_i8_layout_kind {
    DP16_PACKED_I8_LAYOUT_UNKNOWN      = -1,
    DP16_PACKED_I8_LAYOUT_ROW          = 0,
    DP16_PACKED_I8_LAYOUT_D16_PLANAR   = 1,
    DP16_PACKED_I8_LAYOUT_PADDED_2D    = 2,
    DP16_PACKED_I8_LAYOUT_PAGE16_D16   = 3,
};

enum dp16_packed_i8_axis {
    DP16_PACKED_I8_AXIS_UNKNOWN = 0,
    DP16_PACKED_I8_AXIS_D16     = 1,
    DP16_PACKED_I8_AXIS_TOKEN   = 2,
    DP16_PACKED_I8_AXIS_HEAD    = 3,
    DP16_PACKED_I8_AXIS_QBLOCK  = 4,
    DP16_PACKED_I8_AXIS_STAGE   = 5,
};

enum dp16_packed_i8_scale_layout {
    DP16_PACKED_I8_SCALE_LAYOUT_UNKNOWN         = -1,
    DP16_PACKED_I8_SCALE_LAYOUT_ROW             = 0,
    DP16_PACKED_I8_SCALE_LAYOUT_QBLOCK_PLANAR   = 1,
    DP16_PACKED_I8_SCALE_LAYOUT_PAGE16_QBLOCK   = 2,
};

struct dp16_packed_i8_desc_v1 {
    uint32_t version = 0;

    uint32_t lanes_per_vector = 0;
    uint32_t words_per_vector = 0;
    uint32_t bytes_per_vector = 0;
    uint32_t bytes_per_word = 0;

    int32_t layout_kind = DP16_PACKED_I8_LAYOUT_UNKNOWN;
    int32_t axis_x = DP16_PACKED_I8_AXIS_UNKNOWN;
    int32_t axis_y = DP16_PACKED_I8_AXIS_UNKNOWN;
    int32_t axis_z = DP16_PACKED_I8_AXIS_UNKNOWN;

    uint32_t logical_x = 0;
    uint32_t logical_y = 0;
    uint32_t logical_z = 0;
    uint32_t physical_x = 0;
    uint32_t physical_y = 0;
    uint32_t physical_z = 0;

    uint32_t halo_x_before = 0;
    uint32_t halo_y_before = 0;
    uint32_t halo_x_after = 0;
    uint32_t halo_y_after = 0;

    uint64_t base_offset_bytes = 0;
    uint64_t x_stride_bytes = 0;
    uint64_t y_stride_bytes = 0;
    uint64_t z_stride_bytes = 0;
    uint64_t plane_stride_bytes = 0;

    int32_t scale_layout = DP16_PACKED_I8_SCALE_LAYOUT_UNKNOWN;
    int32_t scale_axis_x = DP16_PACKED_I8_AXIS_UNKNOWN;
    int32_t scale_axis_y = DP16_PACKED_I8_AXIS_UNKNOWN;
    int32_t scale_axis_z = DP16_PACKED_I8_AXIS_UNKNOWN;

    uint64_t scale_base_offset_bytes = 0;
    uint64_t scale_x_stride_bytes = 0;
    uint64_t scale_y_stride_bytes = 0;
    uint64_t scale_z_stride_bytes = 0;
    uint64_t scale_plane_stride_bytes = 0;
};

static __host__ __device__ __forceinline__ bool dp16_i8x16_desc_has_vector_abi(
        const dp16_packed_i8_desc_v1 & desc) {
    return desc.version == DP16_PACKED_I8_DESC_VERSION &&
        desc.lanes_per_vector == DP16_PACKED_I8X16_LANES &&
        desc.words_per_vector == DP16_PACKED_I8X16_WORDS &&
        desc.bytes_per_vector == DP16_PACKED_I8X16_BYTES &&
        desc.bytes_per_word == DP16_PACKED_I8_WORD_BYTES;
}

static __host__ __device__ __forceinline__ dp16_i8x16_words dp16_i8x16_words_zero() {
    dp16_i8x16_words v = {{0, 0, 0, 0}};
    return v;
}

static __host__ __device__ __forceinline__ int8_t dp16_i8x16_lane(
        const dp16_i8x16_words & v,
        const int lane) {
    const uint32_t w = (uint32_t) v.word[lane >> 2];
    return (int8_t) ((w >> ((lane & 3) * 8)) & 0xffu);
}

static __host__ __device__ __forceinline__ void dp16_i8x16_set_lane(
        dp16_i8x16_words & v,
        const int lane,
        const int8_t value) {
    const int word = lane >> 2;
    const int shift = (lane & 3) * 8;
    const uint32_t mask = 0xffu << shift;
    uint32_t w = (uint32_t) v.word[word];
    w = (w & ~mask) | (((uint32_t) (uint8_t) value) << shift);
    v.word[word] = (int32_t) w;
}

static __host__ __device__ __forceinline__ int32_t dp16_i8x16_dot_s32(
        const dp16_i8x16_words & a,
        const dp16_i8x16_words & b) {
    int32_t acc = 0;
#ifndef DP16_PACKED_I8_DESC_HOST_ONLY
#pragma unroll
#endif
    for (int lane = 0; lane < (int) DP16_PACKED_I8X16_LANES; ++lane) {
        acc += (int32_t) dp16_i8x16_lane(a, lane) * (int32_t) dp16_i8x16_lane(b, lane);
    }
    return acc;
}

static __host__ __device__ __forceinline__ uint64_t dp16_packed_i8_payload_byte_offset(
        const dp16_packed_i8_desc_v1 & desc,
        const uint32_t z,
        const uint32_t y,
        const uint32_t x,
        const uint32_t word) {
    if (desc.layout_kind == DP16_PACKED_I8_LAYOUT_PAGE16_D16) {
        const uint32_t y_halo = y + desc.halo_y_before;
        const uint32_t page = y_halo / DP16_PACKED_I8_PAGE16_TOKENS;
        const uint32_t slot = y_halo - page * DP16_PACKED_I8_PAGE16_TOKENS;
        return desc.base_offset_bytes +
            (uint64_t) z * desc.z_stride_bytes +
            (uint64_t) page * desc.plane_stride_bytes +
            (uint64_t) (x + desc.halo_x_before) * desc.x_stride_bytes +
            (uint64_t) slot * desc.y_stride_bytes +
            (uint64_t) word * desc.bytes_per_word;
    }
    return desc.base_offset_bytes +
        (uint64_t) z * desc.z_stride_bytes +
        (uint64_t) (y + desc.halo_y_before) * desc.y_stride_bytes +
        (uint64_t) (x + desc.halo_x_before) * desc.x_stride_bytes +
        (uint64_t) word * desc.bytes_per_word;
}

static __host__ __device__ __forceinline__ size_t dp16_packed_i8_payload_word_index(
        const dp16_packed_i8_desc_v1 & desc,
        const uint32_t z,
        const uint32_t y,
        const uint32_t x,
        const uint32_t word) {
    return (size_t) (dp16_packed_i8_payload_byte_offset(desc, z, y, x, word) / desc.bytes_per_word);
}

static __host__ __device__ __forceinline__ uint64_t dp16_packed_i8_scale_byte_offset(
        const dp16_packed_i8_desc_v1 & desc,
        const uint32_t z,
        const uint32_t y,
        const uint32_t x) {
    if (desc.scale_layout == DP16_PACKED_I8_SCALE_LAYOUT_PAGE16_QBLOCK) {
        const uint32_t page = y / DP16_PACKED_I8_PAGE16_TOKENS;
        const uint32_t slot = y - page * DP16_PACKED_I8_PAGE16_TOKENS;
        return desc.scale_base_offset_bytes +
            (uint64_t) z * desc.scale_z_stride_bytes +
            (uint64_t) page * desc.scale_plane_stride_bytes +
            (uint64_t) x * desc.scale_x_stride_bytes +
            (uint64_t) slot * desc.scale_y_stride_bytes;
    }
    return desc.scale_base_offset_bytes +
        (uint64_t) z * desc.scale_z_stride_bytes +
        (uint64_t) y * desc.scale_y_stride_bytes +
        (uint64_t) x * desc.scale_x_stride_bytes;
}
