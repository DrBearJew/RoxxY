#pragma once

// CK/Triton-style compressed-KV tile loader primitives for FlashAttention.
//
// The FA pipeline should not know how a KV row is packed.  It should ask a
// loader to materialize a logical K/V tile as f16 values, then feed that tile to
// the existing tensor-core/MFMA/WMMA math.  This header is the first shared layer
// for TBQ4, PlanarQuant, and IsoQuant loaders.

#include "common.cuh"
#include "tbq4-cuda.cuh"
#include "planar-iso-constants.cuh"

enum ggml_cuda_fattn_compressed_kv_domain {
    GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_ORIGINAL,
    GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_FWHT,
};

template <ggml_type type>
struct ggml_cuda_fattn_compressed_kv_traits;

template <>
struct ggml_cuda_fattn_compressed_kv_traits<GGML_TYPE_TBQ4_0> {
    static constexpr int qk = QK_TBQ4;
    static constexpr int bits = 4;
    static constexpr ggml_cuda_fattn_compressed_kv_domain domain = GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_FWHT;

    static __device__ __forceinline__ float load_elem(const char * __restrict__ row, const int i) {
        const block_tbq4_0 * blocks = (const block_tbq4_0 *) row;
        const int ib = i / QK_TBQ4;
        const int j  = i % QK_TBQ4;
        const block_tbq4_0 * blk = blocks + ib;
        const uint8_t byte = __ldg(&blk->qs[j / 2]);
        const uint8_t idx  = (j & 1) ? (byte >> 4) : (byte & 0xF);
        return d_tbq4_centroids[idx] * __half2float(__ldg(&blk->d));
    }
};

template <>
struct ggml_cuda_fattn_compressed_kv_traits<GGML_TYPE_PLANAR3_0> {
    static constexpr int qk = QK_PLANAR3;
    static constexpr int bits = 3;
    static constexpr ggml_cuda_fattn_compressed_kv_domain domain = GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_ORIGINAL;

    static __device__ __forceinline__ uint8_t unpack(const block_planar3_0 & blk, const int j) {
        const uint8_t low = (__ldg(&blk.qs[j / 4]) >> ((j % 4) * 2)) & 0x3;
        const uint8_t hi  = (__ldg(&blk.signs[j / 8]) >> (j % 8)) & 0x1;
        return low | (hi << 2);
    }

    static __device__ __forceinline__ float load_elem(const char * __restrict__ row, const int i) {
        const block_planar3_0 * blocks = (const block_planar3_0 *) row;
        const int ib = i / QK_PLANAR3;
        const int j  = i % QK_PLANAR3;
        const int j_pair = j & ~1;
        const block_planar3_0 & blk = blocks[ib];

        const float q0 = PI_CENTROIDS_3BIT[unpack(blk, j_pair + 0)];
        const float q1 = PI_CENTROIDS_3BIT[unpack(blk, j_pair + 1)];
        const int p = ((ib * QK_PLANAR3 + j_pair) / 2) & 63;
        const float c = PI_COS[p];
        const float s = PI_SIN[p];
        const float norm = __half2float(__ldg(&blk.d));

        return ((j & 1) ? (-s * q0 + c * q1) : (c * q0 + s * q1)) * norm;
    }
};

template <>
struct ggml_cuda_fattn_compressed_kv_traits<GGML_TYPE_ISO3_0> {
    static constexpr int qk = QK_ISO3;
    static constexpr int bits = 3;
    static constexpr ggml_cuda_fattn_compressed_kv_domain domain = GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_ORIGINAL;

    static __device__ __forceinline__ uint8_t unpack(const block_iso3_0 & blk, const int j) {
        const uint8_t low = (__ldg(&blk.qs[j / 4]) >> ((j % 4) * 2)) & 0x3;
        const uint8_t hi  = (__ldg(&blk.signs[j / 8]) >> (j % 8)) & 0x1;
        return low | (hi << 2);
    }

    static __device__ __forceinline__ float load_elem(const char * __restrict__ row, const int i) {
        const block_iso3_0 * blocks = (const block_iso3_0 *) row;
        const int ib = i / QK_ISO3;
        const int j  = i % QK_ISO3;
        const int j_group = j & ~3;
        const block_iso3_0 & blk = blocks[ib];

        float qvals[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            qvals[c] = PI_CENTROIDS_3BIT[unpack(blk, j_group + c)];
        }

        const int g = ((ib * QK_ISO3 + j_group) / 4) & 31;
        const float qw =  PI_QW[g];
        const float qx = -PI_QX[g];
        const float qy = -PI_QY[g];
        const float qz = -PI_QZ[g];
        const float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
        const float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
        const float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
        const float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];
        const float norm = __half2float(__ldg(&blk.d));

        const int offset = j & 3;
        const float vals[4] = {rw, rx, ry, rz};
        return vals[offset] * norm;
    }
};

template <>
struct ggml_cuda_fattn_compressed_kv_traits<GGML_TYPE_PLANAR4_0> {
    static constexpr int qk = QK_PLANAR4;
    static constexpr int bits = 4;
    static constexpr ggml_cuda_fattn_compressed_kv_domain domain = GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_ORIGINAL;

    static __device__ __forceinline__ uint8_t unpack(const block_planar4_0 & blk, const int j) {
        const uint8_t byte = __ldg(&blk.qs[j / 2]);
        return (j & 1) ? (byte >> 4) : (byte & 0xF);
    }

    static __device__ __forceinline__ float load_elem(const char * __restrict__ row, const int i) {
        const block_planar4_0 * blocks = (const block_planar4_0 *) row;
        const int ib = i / QK_PLANAR4;
        const int j  = i % QK_PLANAR4;
        const int j_pair = j & ~1;
        const block_planar4_0 & blk = blocks[ib];

        const float q0 = PI_CENTROIDS_4BIT[unpack(blk, j_pair + 0)];
        const float q1 = PI_CENTROIDS_4BIT[unpack(blk, j_pair + 1)];
        const int p = ((ib * QK_PLANAR4 + j_pair) / 2) & 63;
        const float c = PI_COS[p];
        const float s = PI_SIN[p];
        const float norm = __half2float(__ldg(&blk.d));

        return ((j & 1) ? (-s * q0 + c * q1) : (c * q0 + s * q1)) * norm;
    }
};

template <>
struct ggml_cuda_fattn_compressed_kv_traits<GGML_TYPE_ISO4_0> {
    static constexpr int qk = QK_ISO4;
    static constexpr int bits = 4;
    static constexpr ggml_cuda_fattn_compressed_kv_domain domain = GGML_CUDA_FATTN_COMPRESSED_KV_DOMAIN_ORIGINAL;

    static __device__ __forceinline__ uint8_t unpack(const block_iso4_0 & blk, const int j) {
        const uint8_t byte = __ldg(&blk.qs[j / 2]);
        return (j & 1) ? (byte >> 4) : (byte & 0xF);
    }

    static __device__ __forceinline__ float load_elem(const char * __restrict__ row, const int i) {
        const block_iso4_0 * blocks = (const block_iso4_0 *) row;
        const int ib = i / QK_ISO4;
        const int j  = i % QK_ISO4;
        const int j_group = j & ~3;
        const block_iso4_0 & blk = blocks[ib];

        float qvals[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            qvals[c] = PI_CENTROIDS_4BIT[unpack(blk, j_group + c)];
        }

        const int g = ((ib * QK_ISO4 + j_group) / 4) & 31;
        const float qw =  PI_QW[g];
        const float qx = -PI_QX[g];
        const float qy = -PI_QY[g];
        const float qz = -PI_QZ[g];
        const float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
        const float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
        const float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
        const float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];
        const float norm = __half2float(__ldg(&blk.d));

        const int offset = j & 3;
        const float vals[4] = {rw, rx, ry, rz};
        return vals[offset] * norm;
    }
};

// Backend-owned row mapping.  The first production adapter is intentionally
// contiguous-only; future paged/block-table adapters should expose the same
// `row_ptr(logical_row)` primitive without changing format traits.
struct ggml_cuda_fattn_contiguous_row_mapper {
    const char * base_ptr;
    int64_t stride_bytes;

    static constexpr bool paged = false;

    __device__ __forceinline__ const char * row_ptr(const int logical_row) const {
        return base_ptr + int64_t(logical_row) * stride_bytes;
    }
};

// Materialize `frag_m` logical compressed rows as a dense f16 tile with row
// stride `D_padded`.  This deliberately mirrors CK/Triton's separation between
// tile acquisition and math: callers can swap loader traits without changing the
// attention loop.  Synchronization is a backend responsibility and is deliberately
// kept outside this format/domain decode primitive.
template <ggml_type type, int D, int frag_m, int D_padded, typename row_mapper_t>
static __device__ __forceinline__ void ggml_cuda_fattn_materialize_compressed_rows_f16(
        const row_mapper_t & row_map,
        const int valid_rows,
        _Float16 * __restrict__ dst) {
    using traits = ggml_cuda_fattn_compressed_kv_traits<type>;
    static_assert(D % traits::qk == 0, "compressed KV row must contain whole quant blocks");
    static_assert(D_padded >= D, "D_padded must cover D");

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#pragma unroll
    for (int r = 0; r < frag_m; ++r) {
        _Float16 * row_out = dst + r * D_padded;
        const bool row_oob = r >= valid_rows;
        const char * row_in = row_map.row_ptr(r);

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i < D) {
                if (row_oob) {
                    row_out[i] = (_Float16) 0.0f;
                } else {
                    row_out[i] = (_Float16) traits::load_elem(row_in, i);
                }
            }
        }

#pragma unroll
        for (int i = D + threadIdx.x; i < D_padded; i += warp_size) {
            row_out[i] = (_Float16) 0.0f;
        }
    }
}

// Contiguous-row convenience overload used by the current rocWMMA backend.
template <ggml_type type, int D, int frag_m, int D_padded>
static __device__ __forceinline__ void ggml_cuda_fattn_materialize_compressed_rows_f16(
        const char * __restrict__ base_ptr,
        const int64_t stride_bytes,
        const int valid_rows,
        _Float16 * __restrict__ dst) {
    const ggml_cuda_fattn_contiguous_row_mapper row_map{base_ptr, stride_bytes};
    ggml_cuda_fattn_materialize_compressed_rows_f16<type, D, frag_m, D_padded>(
        row_map, valid_rows, dst);
}

static __device__ __forceinline__ void ggml_cuda_fattn_sync_compressed_tile() {
    __syncthreads();
}

// Backward-compatible synchronized tile-load helper.  New backend code should
// call `ggml_cuda_fattn_materialize_compressed_rows_f16` and synchronize at the
// rocWMMA/shared-memory boundary instead of hiding sync in format decode.
template <ggml_type type, int D, int frag_m, int D_padded>
static __device__ __forceinline__ void ggml_cuda_fattn_load_compressed_rows_f16(
        const char * __restrict__ base_ptr,
        const int64_t stride_bytes,
        const int valid_rows,
        _Float16 * __restrict__ dst) {
    ggml_cuda_fattn_materialize_compressed_rows_f16<type, D, frag_m, D_padded>(
        base_ptr, stride_bytes, valid_rows, dst);
    ggml_cuda_fattn_sync_compressed_tile();
}
