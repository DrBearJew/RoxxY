#include "argsort.cuh"
#include "top-k.cuh"
#include "quantize.cuh"
#include "rdna-i8-packed16.cuh"
#include "vecdotq.cuh"

#include <cstdlib>
#include <cfloat>
#include <climits>
#include <cstring>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

static __device__ __forceinline__ bool top_k_f32_i32_better(const float a_val, const int a_id, const float b_val, const int b_id) {
    return a_val > b_val || (a_val == b_val && a_id < b_id);
}

static __device__ __forceinline__ bool top_k_f32_i32_is_banned(
        const int token,
        const int n_ban,
        const int ban0,
        const int ban1,
        const int ban2,
        const int ban3,
        const int ban4,
        const int ban5,
        const int ban6,
        const int ban7) {
    return (n_ban > 0 && token == ban0) ||
           (n_ban > 1 && token == ban1) ||
           (n_ban > 2 && token == ban2) ||
           (n_ban > 3 && token == ban3) ||
           (n_ban > 4 && token == ban4) ||
           (n_ban > 5 && token == ban5) ||
           (n_ban > 6 && token == ban6) ||
           (n_ban > 7 && token == ban7);
}

template<int K_MAX>
static __device__ __forceinline__ void top_k_f32_i32_insert(
        const float val, const int id, const int k, float (& top_vals)[K_MAX], int (& top_ids)[K_MAX]) {
    if (!top_k_f32_i32_better(val, id, top_vals[k - 1], top_ids[k - 1])) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && top_k_f32_i32_better(val, id, top_vals[pos - 1], top_ids[pos - 1])) {
        top_vals[pos] = top_vals[pos - 1];
        top_ids[pos]  = top_ids[pos - 1];
        --pos;
    }
    top_vals[pos] = val;
    top_ids[pos]  = id;
}

static __global__ void k_top_k_f32_i32_small_k(
        const float * src,
        int *         dst,
        const int     ncols,
        const int     k,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t src_nb3,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t dst_nb3,
        const int     n_ban,
        const int     ban0,
        const int     ban1,
        const int     ban2,
        const int     ban3,
        const int     ban4,
        const int     ban5,
        const int     ban6,
        const int     ban7) {
    constexpr int k_max = 16;

    const int64_t row = blockIdx.x;
    const int64_t i1  = row % ne1;
    const int64_t i2  = (row / ne1) % ne2;
    const int64_t i3  = row / (ne1 * ne2);

    const float * src_row = src + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;
    int *         dst_row = dst + i1 * dst_nb1 + i2 * dst_nb2 + i3 * dst_nb3;

    float thread_vals[k_max];
    int   thread_ids[k_max];

#pragma unroll
    for (int i = 0; i < k_max; ++i) {
        thread_vals[i] = -INFINITY;
        thread_ids[i]  = 0x7fffffff;
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        if (!top_k_f32_i32_is_banned(col, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7)) {
            top_k_f32_i32_insert(src_row[col], col, k, thread_vals, thread_ids);
        }
    }

    extern __shared__ char smem[];
    float * shared_vals = reinterpret_cast<float *>(smem);
    int *   shared_ids  = reinterpret_cast<int *>(shared_vals + blockDim.x * k);

    const int shared_off = threadIdx.x * k;
    for (int i = 0; i < k; ++i) {
        shared_vals[shared_off + i] = thread_vals[i];
        shared_ids[shared_off + i]  = thread_ids[i];
    }

    __syncthreads();

    // Reduce per-thread top-k lists in parallel.  The previous implementation
    // let thread 0 merge all blockDim.x*k candidates serially; for vocab-sized
    // rows that made small-k TOP_K a visible part of the MTP LM-head path.
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float row_vals[k_max];
            int   row_ids[k_max];

#pragma unroll
            for (int i = 0; i < k_max; ++i) {
                row_vals[i] = -INFINITY;
                row_ids[i]  = 0x7fffffff;
            }

            const int dst_off  = threadIdx.x * k;
            const int peer_off = (threadIdx.x + stride) * k;

            for (int i = 0; i < k; ++i) {
                row_vals[i] = shared_vals[dst_off + i];
                row_ids[i]  = shared_ids [dst_off + i];
            }
            for (int i = 0; i < k; ++i) {
                top_k_f32_i32_insert(shared_vals[peer_off + i], shared_ids[peer_off + i], k, row_vals, row_ids);
            }
            for (int i = 0; i < k; ++i) {
                shared_vals[dst_off + i] = row_vals[i];
                shared_ids [dst_off + i] = row_ids[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) {
            dst_row[i] = shared_ids[i];
        }
    }
}

static __global__ void k_top_k_f32_i32_small_k_stage1(
        const float * src,
        float *       partial_vals,
        int *         partial_ids,
        const int     ncols,
        const int     k,
        const int     tile_cols,
        const int     ntiles,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t src_nb3,
        const int     n_ban,
        const int     ban0,
        const int     ban1,
        const int     ban2,
        const int     ban3,
        const int     ban4,
        const int     ban5,
        const int     ban6,
        const int     ban7) {
    constexpr int k_max = 16;

    const int64_t row  = blockIdx.x;
    const int     tile = blockIdx.y;
    const int64_t i1   = row % ne1;
    const int64_t i2   = (row / ne1) % ne2;
    const int64_t i3   = row / (ne1 * ne2);

    const float * src_row = src + i1 * src_nb1 + i2 * src_nb2 + i3 * src_nb3;

    float thread_vals[k_max];
    int   thread_ids[k_max];

#pragma unroll
    for (int i = 0; i < k_max; ++i) {
        thread_vals[i] = -INFINITY;
        thread_ids[i]  = 0x7fffffff;
    }

    const int col_begin = tile * tile_cols;
    const int col_end   = col_begin + tile_cols < ncols ? col_begin + tile_cols : ncols;
    for (int col = col_begin + threadIdx.x; col < col_end; col += blockDim.x) {
        if (!top_k_f32_i32_is_banned(col, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7)) {
            top_k_f32_i32_insert(src_row[col], col, k, thread_vals, thread_ids);
        }
    }

    extern __shared__ char smem[];
    float * shared_vals = reinterpret_cast<float *>(smem);
    int *   shared_ids  = reinterpret_cast<int *>(shared_vals + blockDim.x * k);

    const int shared_off = threadIdx.x * k;
    for (int i = 0; i < k; ++i) {
        shared_vals[shared_off + i] = thread_vals[i];
        shared_ids[shared_off + i]  = thread_ids[i];
    }

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float row_vals[k_max];
            int   row_ids[k_max];

#pragma unroll
            for (int i = 0; i < k_max; ++i) {
                row_vals[i] = -INFINITY;
                row_ids[i]  = 0x7fffffff;
            }

            const int dst_off  = threadIdx.x * k;
            const int peer_off = (threadIdx.x + stride) * k;

            for (int i = 0; i < k; ++i) {
                row_vals[i] = shared_vals[dst_off + i];
                row_ids[i]  = shared_ids [dst_off + i];
            }
            for (int i = 0; i < k; ++i) {
                top_k_f32_i32_insert(shared_vals[peer_off + i], shared_ids[peer_off + i], k, row_vals, row_ids);
            }
            for (int i = 0; i < k; ++i) {
                shared_vals[dst_off + i] = row_vals[i];
                shared_ids [dst_off + i] = row_ids[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int partial_off = (row * ntiles + tile) * k;
        for (int i = 0; i < k; ++i) {
            partial_vals[partial_off + i] = shared_vals[i];
            partial_ids [partial_off + i] = shared_ids[i];
        }
    }
}

static __global__ void k_top_k_f32_i32_small_k_stage2(
        const float * partial_vals,
        const int *   partial_ids,
        int *         dst,
        const int     k,
        const int     ntiles,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t dst_nb3) {
    constexpr int k_max = 16;

    const int64_t row = blockIdx.x;
    const int64_t i1  = row % ne1;
    const int64_t i2  = (row / ne1) % ne2;
    const int64_t i3  = row / (ne1 * ne2);

    int * dst_row = dst + i1 * dst_nb1 + i2 * dst_nb2 + i3 * dst_nb3;

    float thread_vals[k_max];
    int   thread_ids[k_max];

#pragma unroll
    for (int i = 0; i < k_max; ++i) {
        thread_vals[i] = -INFINITY;
        thread_ids[i]  = 0x7fffffff;
    }

    const int partial_base = row * ntiles * k;
    const int n_cand = ntiles * k;
    for (int cand = threadIdx.x; cand < n_cand; cand += blockDim.x) {
        top_k_f32_i32_insert(partial_vals[partial_base + cand], partial_ids[partial_base + cand], k, thread_vals, thread_ids);
    }

    extern __shared__ char smem[];
    float * shared_vals = reinterpret_cast<float *>(smem);
    int *   shared_ids  = reinterpret_cast<int *>(shared_vals + blockDim.x * k);

    const int shared_off = threadIdx.x * k;
    for (int i = 0; i < k; ++i) {
        shared_vals[shared_off + i] = thread_vals[i];
        shared_ids[shared_off + i]  = thread_ids[i];
    }

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float row_vals[k_max];
            int   row_ids[k_max];

#pragma unroll
            for (int i = 0; i < k_max; ++i) {
                row_vals[i] = -INFINITY;
                row_ids[i]  = 0x7fffffff;
            }

            const int dst_off  = threadIdx.x * k;
            const int peer_off = (threadIdx.x + stride) * k;

            for (int i = 0; i < k; ++i) {
                row_vals[i] = shared_vals[dst_off + i];
                row_ids[i]  = shared_ids [dst_off + i];
            }
            for (int i = 0; i < k; ++i) {
                top_k_f32_i32_insert(shared_vals[peer_off + i], shared_ids[peer_off + i], k, row_vals, row_ids);
            }
            for (int i = 0; i < k; ++i) {
                shared_vals[dst_off + i] = row_vals[i];
                shared_ids [dst_off + i] = row_ids[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) {
            dst_row[i] = shared_ids[i];
        }
    }
}

static void top_k_f32_i32_small_k_cuda(
        ggml_cuda_pool &   pool,
        const float *       src,
        int *               dst,
        const int           ncols,
        const int           nrows,
        const int           k,
        const ggml_tensor * src0,
        const ggml_tensor * dst_tensor,
        cudaStream_t        stream,
        const int           n_ban,
        const int           ban0,
        const int           ban1,
        const int           ban2,
        const int           ban3,
        const int           ban4,
        const int           ban5,
        const int           ban6,
        const int           ban7) {
    GGML_ASSERT(k > 0 && k <= 16);

    constexpr int block_size = 256;
    const dim3    block_dims(block_size, 1, 1);
    const size_t  shared_mem = block_size * k * (sizeof(float) + sizeof(int));

    if (ncols > 8192) {
        constexpr int tile_cols = 4096;
        const int ntiles = (ncols + tile_cols - 1) / tile_cols;
        ggml_cuda_pool_alloc<float> partial_vals_alloc(pool, (size_t) nrows * ntiles * k);
        ggml_cuda_pool_alloc<int>   partial_ids_alloc (pool, (size_t) nrows * ntiles * k);
        float * partial_vals = partial_vals_alloc.get();
        int *   partial_ids  = partial_ids_alloc.get();

        const dim3 stage1_grid(nrows, ntiles, 1);
        k_top_k_f32_i32_small_k_stage1<<<stage1_grid, block_dims, shared_mem, stream>>>(
                src, partial_vals, partial_ids, ncols, k, tile_cols, ntiles,
                src0->ne[1], src0->ne[2],
                src0->nb[1] / sizeof(float), src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

        const dim3 stage2_grid(nrows, 1, 1);
        k_top_k_f32_i32_small_k_stage2<<<stage2_grid, block_dims, shared_mem, stream>>>(
                partial_vals, partial_ids, dst, k, ntiles,
                src0->ne[1], src0->ne[2],
                dst_tensor->nb[1] / sizeof(int), dst_tensor->nb[2] / sizeof(int), dst_tensor->nb[3] / sizeof(int));
        return;
    }

    const dim3 block_nums(nrows, 1, 1);
    k_top_k_f32_i32_small_k<<<block_nums, block_dims, shared_mem, stream>>>(
            src, dst, ncols, k,
            src0->ne[1], src0->ne[2],
            src0->nb[1] / sizeof(float), src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
            dst_tensor->nb[1] / sizeof(int), dst_tensor->nb[2] / sizeof(int), dst_tensor->nb[3] / sizeof(int),
            n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);
}

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

static __device__ __forceinline__ void lm_head_top1_update(float & best_val, int & best_id, const float val, const int id) {
    if (val > best_val || (val == best_val && id < best_id)) {
        best_val = val;
        best_id  = id;
    }
}

static __device__ __forceinline__ bool lm_head_top1_is_banned(
        const int token,
        const int n_ban,
        const int ban0,
        const int ban1,
        const int ban2,
        const int ban3,
        const int ban4,
        const int ban5,
        const int ban6,
        const int ban7) {
    return (n_ban > 0 && token == ban0) ||
           (n_ban > 1 && token == ban1) ||
           (n_ban > 2 && token == ban2) ||
           (n_ban > 3 && token == ban3) ||
           (n_ban > 4 && token == ban4) ||
           (n_ban > 5 && token == ban5) ||
           (n_ban > 6 && token == ban6) ||
           (n_ban > 7 && token == ban7);
}

static __device__ __forceinline__ uint32_t lm_head_top1_float_ordered(const float val) {
    const uint32_t bits = __float_as_uint(val);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

static __device__ __forceinline__ unsigned long long lm_head_top1_make_key(const float val, const int id) {
    const uint32_t ord = lm_head_top1_float_ordered(val);
    const uint32_t inv = 0xffffffffu - (uint32_t) id;
    return ((unsigned long long) ord << 32) | (unsigned long long) inv;
}

static __global__ void k_lm_head_top1_init_keys(unsigned long long * keys, const int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        keys[i] = 0ull;
    }
}

static __global__ void k_lm_head_top1_keys_to_ids(const unsigned long long * keys, int * dst, const int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        const uint32_t inv = (uint32_t) (keys[i] & 0xffffffffull);
        dst[i] = (int) (0xffffffffu - inv);
    }
}

template<int ROWS_PER_BLOCK>
static __global__ void k_lm_head_top1_q6k_stage1(
        const void *       vx,
        const block_q8_1 * y,
        float *            partial_vals,
        int *              partial_ids,
        const int          n_vocab,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_partials,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
    constexpr int vdr       = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane       = threadIdx.x;
    const int row_lane   = threadIdx.y;
    const int hrow       = blockIdx.y;
    const int token      = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool active    = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sum = 0.0f;
    if (active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;
        const block_q8_1 * y_row = y + hrow * q8_blocks_per_row;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q6_K_q8_1(vx, y_row + kby, kbx_offset + kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);

    __shared__ float row_vals[ROWS_PER_BLOCK];
    __shared__ int   row_ids [ROWS_PER_BLOCK];

    if (lane == 0) {
        row_vals[row_lane] = active ? sum   : -INFINITY;
        row_ids [row_lane] = active ? token : 0x7fffffff;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float best_val = -INFINITY;
        int   best_id  = 0x7fffffff;
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            lm_head_top1_update(best_val, best_id, row_vals[i], row_ids[i]);
        }
        const int out = hrow * n_partials + blockIdx.x;
        partial_vals[out] = best_val;
        partial_ids [out] = best_id;
    }
}

template<int ROWS_PER_BLOCK>
static __global__ void k_lm_head_top1_q6k_atomic(
        const void *       vx,
        const block_q8_1 * y,
        unsigned long long * best_keys,
        const int          n_vocab,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
    constexpr int vdr       = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane     = threadIdx.x;
    const int row_lane = threadIdx.y;
    const int hrow     = blockIdx.y;
    const int token    = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool active  = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sum = 0.0f;
    if (active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;
        const block_q8_1 * y_row = y + hrow * q8_blocks_per_row;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q6_K_q8_1(vx, y_row + kby, kbx_offset + kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);

    __shared__ float row_vals[ROWS_PER_BLOCK];
    __shared__ int   row_ids [ROWS_PER_BLOCK];

    if (lane == 0) {
        row_vals[row_lane] = active ? sum   : -INFINITY;
        row_ids [row_lane] = active ? token : 0x7fffffff;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float best_val = -INFINITY;
        int   best_id  = 0x7fffffff;
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            lm_head_top1_update(best_val, best_id, row_vals[i], row_ids[i]);
        }
        atomicMax(&best_keys[hrow], lm_head_top1_make_key(best_val, best_id));
    }
}

template<int ROWS_PER_BLOCK, int HIDDEN_ROWS_PER_GROUP>
static __global__ void k_lm_head_top1_q6k_stage1_multirow(
        const void *       vx,
        const block_q8_1 * y,
        float *            partial_vals,
        int *              partial_ids,
        const int          n_vocab,
        const int          n_rows,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_partials,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
    constexpr int vdr       = VDR_Q6_K_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane       = threadIdx.x;
    const int row_lane   = threadIdx.y;
    const int hrow_base  = blockIdx.y * HIDDEN_ROWS_PER_GROUP;
    const int token      = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool tok_active = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sums[HIDDEN_ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
        sums[r] = 0.0f;
    }

    if (tok_active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));

            const block_q6_K * bq6_K = (const block_q6_K *) vx + kbx_offset + kbx;

            const int bq8_offset  = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);
            const int scale_offset = (QI6_K/4) * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/8);
            const int vh_shift     = 2 * ((kqs % (QI6_K/2)) / (QI6_K/4));

            const int vl = get_int_b2(bq6_K->ql, kqs);
            const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (kqs / (QI6_K/2)) + kqs % (QI6_K/4)) >> vh_shift;
            const int8_t * scales = bq6_K->scales + scale_offset;

#pragma unroll
            for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
                const int hrow = hrow_base + r;
                if (hrow < n_rows) {
                    const block_q8_1 * y_row = y + hrow * q8_blocks_per_row + kby;
                    int  u[QR6_K];
                    half d8[QR6_K];

#pragma unroll
                    for (int i = 0; i < QR6_K; ++i) {
                        u[i]  = get_int_b4(y_row[bq8_offset + 2*i].qs, kqs % QI8_1);
                        d8[i] = ((const half *) &y_row[bq8_offset + 2*i].ds)[0];
                    }

                    sums[r] += vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
        sums[r] = warp_reduce_sum<warp_size>(sums[r]);
    }

    __shared__ float row_vals[HIDDEN_ROWS_PER_GROUP][ROWS_PER_BLOCK];
    __shared__ int   row_ids [HIDDEN_ROWS_PER_GROUP][ROWS_PER_BLOCK];

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
            const int hrow = hrow_base + r;
            const bool active = tok_active && hrow < n_rows;
            row_vals[r][row_lane] = active ? sums[r] : -INFINITY;
            row_ids [r][row_lane] = active ? token   : 0x7fffffff;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
        for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
            const int hrow = hrow_base + r;
            if (hrow < n_rows) {
                float best_val = -INFINITY;
                int   best_id  = 0x7fffffff;
                for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                    lm_head_top1_update(best_val, best_id, row_vals[r][i], row_ids[r][i]);
                }
                const int out = hrow * n_partials + blockIdx.x;
                partial_vals[out] = best_val;
                partial_ids [out] = best_id;
            }
        }
    }
}

template<int ROWS_PER_BLOCK>
static __global__ void k_lm_head_top1_q8_0_stage1(
        const void *       vx,
        const block_q8_1 * y,
        float *            partial_vals,
        int *              partial_ids,
        const int          n_vocab,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_partials,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane       = threadIdx.x;
    const int row_lane   = threadIdx.y;
    const int hrow       = blockIdx.y;
    const int token      = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool active    = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sum = 0.0f;
    if (active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;
        const block_q8_1 * y_row = y + hrow * q8_blocks_per_row;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q8_0_q8_1(vx, y_row + kby, kbx_offset + kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);

    __shared__ float row_vals[ROWS_PER_BLOCK];
    __shared__ int   row_ids [ROWS_PER_BLOCK];

    if (lane == 0) {
        row_vals[row_lane] = active ? sum   : -INFINITY;
        row_ids [row_lane] = active ? token : 0x7fffffff;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float best_val = -INFINITY;
        int   best_id  = 0x7fffffff;
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            lm_head_top1_update(best_val, best_id, row_vals[i], row_ids[i]);
        }
        const int out = hrow * n_partials + blockIdx.x;
        partial_vals[out] = best_val;
        partial_ids [out] = best_id;
    }
}

template<int ROWS_PER_BLOCK>
static __global__ void k_lm_head_top1_q8_0_atomic(
        const void *       vx,
        const block_q8_1 * y,
        unsigned long long * best_keys,
        const int          n_vocab,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane     = threadIdx.x;
    const int row_lane = threadIdx.y;
    const int hrow     = blockIdx.y;
    const int token    = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool active  = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sum = 0.0f;
    if (active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;
        const block_q8_1 * y_row = y + hrow * q8_blocks_per_row;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q8_0_q8_1(vx, y_row + kby, kbx_offset + kbx, kqs);
        }
    }

    sum = warp_reduce_sum<warp_size>(sum);

    __shared__ float row_vals[ROWS_PER_BLOCK];
    __shared__ int   row_ids [ROWS_PER_BLOCK];

    if (lane == 0) {
        row_vals[row_lane] = active ? sum   : -INFINITY;
        row_ids [row_lane] = active ? token : 0x7fffffff;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float best_val = -INFINITY;
        int   best_id  = 0x7fffffff;
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            lm_head_top1_update(best_val, best_id, row_vals[i], row_ids[i]);
        }
        atomicMax(&best_keys[hrow], lm_head_top1_make_key(best_val, best_id));
    }
}

template<int ROWS_PER_BLOCK, int HIDDEN_ROWS_PER_GROUP>
static __global__ void k_lm_head_top1_q8_0_stage1_multirow(
        const void *       vx,
        const block_q8_1 * y,
        float *            partial_vals,
        int *              partial_ids,
        const int          n_vocab,
        const int          n_rows,
        const int          blocks_per_row_x,
        const int          q8_blocks_per_row,
        const int          stride_row_x,
        const int          n_partials,
        const int          n_ban,
        const int          ban0,
        const int          ban1,
        const int          ban2,
        const int          ban3,
        const int          ban4,
        const int          ban5,
        const int          ban6,
        const int          ban7) {
    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int lane        = threadIdx.x;
    const int row_lane    = threadIdx.y;
    const int hrow_base   = blockIdx.y * HIDDEN_ROWS_PER_GROUP;
    const int token       = blockIdx.x * ROWS_PER_BLOCK + row_lane;
    const bool tok_active = token < n_vocab && !lm_head_top1_is_banned(token, n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);

    float sums[HIDDEN_ROWS_PER_GROUP];
#pragma unroll
    for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
        sums[r] = 0.0f;
    }

    if (tok_active) {
        constexpr int blocks_per_iter = vdr * warp_size / qi;
        const int kbx_offset = token * stride_row_x;

        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (lane % (qi/vdr));

#pragma unroll
            for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
                const int hrow = hrow_base + r;
                if (hrow < n_rows) {
                    const block_q8_1 * y_row = y + hrow * q8_blocks_per_row + kby;
                    sums[r] += vec_dot_q8_0_q8_1(vx, y_row, kbx_offset + kbx, kqs);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
        sums[r] = warp_reduce_sum<warp_size>(sums[r]);
    }

    __shared__ float row_vals[HIDDEN_ROWS_PER_GROUP][ROWS_PER_BLOCK];
    __shared__ int   row_ids [HIDDEN_ROWS_PER_GROUP][ROWS_PER_BLOCK];

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
            const int hrow = hrow_base + r;
            const bool active = tok_active && hrow < n_rows;
            row_vals[r][row_lane] = active ? sums[r] : -INFINITY;
            row_ids [r][row_lane] = active ? token   : 0x7fffffff;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
        for (int r = 0; r < HIDDEN_ROWS_PER_GROUP; ++r) {
            const int hrow = hrow_base + r;
            if (hrow < n_rows) {
                float best_val = -INFINITY;
                int   best_id  = 0x7fffffff;
                for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                    lm_head_top1_update(best_val, best_id, row_vals[r][i], row_ids[r][i]);
                }
                const int out = hrow * n_partials + blockIdx.x;
                partial_vals[out] = best_val;
                partial_ids [out] = best_id;
            }
        }
    }
}

static __global__ void k_lm_head_top1_stage2(
        const float * partial_vals,
        const int *   partial_ids,
        int *         dst,
        const int     n_partials) {
    const int hrow = blockIdx.x;
    const int tid  = threadIdx.x;

    float best_val = -INFINITY;
    int   best_id  = 0x7fffffff;

    const int base = hrow * n_partials;
    for (int i = tid; i < n_partials; i += blockDim.x) {
        lm_head_top1_update(best_val, best_id, partial_vals[base + i], partial_ids[base + i]);
    }

    extern __shared__ char smem[];
    float * s_vals = reinterpret_cast<float *>(smem);
    int *   s_ids  = reinterpret_cast<int *>(s_vals + blockDim.x);

    s_vals[tid] = best_val;
    s_ids [tid] = best_id;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            lm_head_top1_update(s_vals[tid], s_ids[tid], s_vals[tid + stride], s_ids[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        dst[hrow] = s_ids[0];
    }
}

static void ggml_cuda_op_lm_head_top_k_q8_0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // LM head weight [n_embd, n_vocab]
    const ggml_tensor * src1 = dst->src[1]; // hidden rows [n_embd, n_rows]

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == 1 && "prototype LM_HEAD_TOP_K currently supports exact top1 only");
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_is_contiguous_rows(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd  = src0->ne[0];
    const int64_t n_vocab = src0->ne[1];
    const int64_t n_rows  = ggml_nrows(src1);

    const int n_ban_raw = ggml_get_op_params_i32(dst, 0);
    const int n_ban     = n_ban_raw < 0 ? 0 : (n_ban_raw > 8 ? 8 : n_ban_raw);
    const int ban0      = n_ban > 0 ? ggml_get_op_params_i32(dst, 1) : -1;
    const int ban1      = n_ban > 1 ? ggml_get_op_params_i32(dst, 2) : -1;
    const int ban2      = n_ban > 2 ? ggml_get_op_params_i32(dst, 3) : -1;
    const int ban3      = n_ban > 3 ? ggml_get_op_params_i32(dst, 4) : -1;
    const int ban4      = n_ban > 4 ? ggml_get_op_params_i32(dst, 5) : -1;
    const int ban5      = n_ban > 5 ? ggml_get_op_params_i32(dst, 6) : -1;
    const int ban6      = n_ban > 6 ? ggml_get_op_params_i32(dst, 7) : -1;
    const int ban7      = n_ban > 7 ? ggml_get_op_params_i32(dst, 8) : -1;

    if (const char * log_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
        if (atoi(log_env) != 0) {
            GGML_LOG_INFO("%s: route=cuda_lm_head_top1_q8_0 tensor=%s n_embd=%lld n_vocab=%lld n_rows=%lld ban_count=%d src1_cont_rows=%d src1_cont=%d src1_nb=(%lld,%lld,%lld,%lld) src0_buf=%s src1_buf=%s dst_buf=%s dst=%s\n",
                    __func__, src0->name, (long long) n_embd, (long long) n_vocab, (long long) n_rows, n_ban,
                    ggml_is_contiguous_rows(src1) ? 1 : 0, ggml_is_contiguous(src1) ? 1 : 0,
                    (long long) src1->nb[0], (long long) src1->nb[1], (long long) src1->nb[2], (long long) src1->nb[3],
                    src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "null",
                    src1->buffer ? ggml_backend_buffer_name(src1->buffer) : "null",
                    dst->buffer  ? ggml_backend_buffer_name(dst->buffer)  : "null",
                    dst->name);
        }
    }

    GGML_ASSERT(n_embd % QK8_0 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t q8_blocks_per_row = n_embd_padded / QK8_1;
    const size_t  q8_bytes = (size_t) n_rows * q8_blocks_per_row * sizeof(block_q8_1);

    ggml_cuda_pool_alloc<char> h_q8_alloc(ctx.pool(), q8_bytes);
    char * h_q8 = h_q8_alloc.get();

    const float * src1_d = (const float *) src1->data;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    quantize_row_q8_1_cuda(src1_d, nullptr, h_q8, src0->type, n_embd, s11, s12, s13,
            n_embd_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    int rows_per_block = 16;
    if (const char * rows_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_ROWS_PER_BLOCK")) {
        const int requested = atoi(rows_env);
        if ((requested == 8 || requested == 16 || requested == 32) && requested * warp_size <= 1024) {
            rows_per_block = requested;
        }
    }
    const int stride_row_x = src0->nb[1] / ggml_type_size(src0->type);
    const int blocks_per_row_x = n_embd / QK8_0;

    if (n_rows > 1) {
        if (const char * multirow_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_MULTIROW")) {
            if (atoi(multirow_env) != 0) {
#define LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_MULTIROW_RPB(RPB) do { \
                    constexpr int rows_per_block_t = (RPB); \
                    constexpr int hidden_rows_per_group_t = 8; \
                    const dim3 block_dims(warp_size, rows_per_block_t, 1); \
                    const int n_partials = (n_vocab + rows_per_block_t - 1) / rows_per_block_t; \
                    const dim3 stage1_grid(n_partials, (n_rows + hidden_rows_per_group_t - 1) / hidden_rows_per_group_t, 1); \
                    ggml_cuda_pool_alloc<float> partial_vals_alloc(ctx.pool(), (size_t) n_rows * n_partials); \
                    ggml_cuda_pool_alloc<int>   partial_ids_alloc (ctx.pool(), (size_t) n_rows * n_partials); \
                    float * partial_vals = partial_vals_alloc.get(); \
                    int *   partial_ids  = partial_ids_alloc.get(); \
                    k_lm_head_top1_q8_0_stage1_multirow<rows_per_block_t, hidden_rows_per_group_t><<<stage1_grid, block_dims, 0, stream>>>( \
                            src0->data, (const block_q8_1 *) h_q8, partial_vals, partial_ids, \
                            n_vocab, n_rows, blocks_per_row_x, q8_blocks_per_row, stride_row_x, n_partials, \
                            n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
                    const dim3 stage2_grid(n_rows, 1, 1); \
                    constexpr int stage2_block = 256; \
                    const size_t stage2_smem = stage2_block * (sizeof(float) + sizeof(int)); \
                    k_lm_head_top1_stage2<<<stage2_grid, stage2_block, stage2_smem, stream>>>( \
                            partial_vals, partial_ids, (int *) dst->data, n_partials); \
                } while (0)
                if (const char * log_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
                    if (atoi(log_env) != 0) {
                        GGML_LOG_INFO("%s: route=cuda_lm_head_top1_q8_0 multirow=1 hidden_rows_per_group=8\n", __func__);
                    }
                }
                switch (rows_per_block) {
                    case 8:  LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_MULTIROW_RPB(8);  break;
                    case 32: LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_MULTIROW_RPB(32); break;
                    case 16:
                    default: LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_MULTIROW_RPB(16); break;
                }
#undef LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_MULTIROW_RPB
                return;
            }
        }
    }

#define LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_RPB(RPB) do { \
        constexpr int rows_per_block_t = (RPB); \
        const dim3 block_dims(warp_size, rows_per_block_t, 1); \
        const dim3 stage1_grid((n_vocab + rows_per_block_t - 1) / rows_per_block_t, n_rows, 1); \
        const char * atomic_top1_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_ATOMIC"); \
        if (atomic_top1_env && atoi(atomic_top1_env) != 0) { \
            ggml_cuda_pool_alloc<unsigned long long> best_keys_alloc(ctx.pool(), (size_t) n_rows); \
            unsigned long long * best_keys = best_keys_alloc.get(); \
            constexpr int init_block = 256; \
            const int init_grid = (n_rows + init_block - 1) / init_block; \
            k_lm_head_top1_init_keys<<<init_grid, init_block, 0, stream>>>(best_keys, n_rows); \
            k_lm_head_top1_q8_0_atomic<rows_per_block_t><<<stage1_grid, block_dims, 0, stream>>>( \
                    src0->data, (const block_q8_1 *) h_q8, best_keys, \
                    n_vocab, blocks_per_row_x, q8_blocks_per_row, stride_row_x, \
                    n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
            k_lm_head_top1_keys_to_ids<<<init_grid, init_block, 0, stream>>>(best_keys, (int *) dst->data, n_rows); \
            return; \
        } \
        ggml_cuda_pool_alloc<float> partial_vals_alloc(ctx.pool(), (size_t) n_rows * ((n_vocab + rows_per_block_t - 1) / rows_per_block_t)); \
        ggml_cuda_pool_alloc<int>   partial_ids_alloc (ctx.pool(), (size_t) n_rows * ((n_vocab + rows_per_block_t - 1) / rows_per_block_t)); \
        float * partial_vals = partial_vals_alloc.get(); \
        int *   partial_ids  = partial_ids_alloc.get(); \
        k_lm_head_top1_q8_0_stage1<rows_per_block_t><<<stage1_grid, block_dims, 0, stream>>>( \
                src0->data, (const block_q8_1 *) h_q8, partial_vals, partial_ids, \
                n_vocab, blocks_per_row_x, q8_blocks_per_row, stride_row_x, (n_vocab + rows_per_block_t - 1) / rows_per_block_t, \
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
        const dim3 stage2_grid(n_rows, 1, 1); \
        constexpr int stage2_block = 256; \
        const size_t stage2_smem = stage2_block * (sizeof(float) + sizeof(int)); \
        k_lm_head_top1_stage2<<<stage2_grid, stage2_block, stage2_smem, stream>>>( \
                partial_vals, partial_ids, (int *) dst->data, (n_vocab + rows_per_block_t - 1) / rows_per_block_t); \
    } while (0)

    switch (rows_per_block) {
        case 8:  LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_RPB(8);  break;
        case 32: LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_RPB(32); break;
        case 16:
        default: LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_RPB(16); break;
    }

#undef LLAMA_LM_HEAD_TOP1_Q8_0_DISPATCH_RPB
}


static bool ggml_cuda_router_topk_weights_log_enabled() {
    const char * env = getenv("LLAMA_MTP_ROWEQ_ROUTER_TOPK_FUSED_LOG");
    return env != nullptr && env[0] != '\0' && atoi(env) != 0;
}

static __device__ __forceinline__ void router_topk_insert_pair(
        const float val, const int id, const int k, float * top_vals, int * top_ids) {
    if (k <= 0) {
        return;
    }
    if (!(val > top_vals[k - 1] || (val == top_vals[k - 1] && id < top_ids[k - 1]))) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && (val > top_vals[pos - 1] || (val == top_vals[pos - 1] && id < top_ids[pos - 1]))) {
        top_vals[pos] = top_vals[pos - 1];
        top_ids[pos]  = top_ids[pos - 1];
        --pos;
    }
    top_vals[pos] = val;
    top_ids[pos]  = id;
}

static __device__ __forceinline__ bool router_topk_group_allowed(
        const int group, const int * selected_groups, const int n_group_used) {
    for (int i = 0; i < n_group_used; ++i) {
        if (selected_groups[i] == group) {
            return true;
        }
    }
    return false;
}

template<int K_MAX, int GROUP_MAX>
static __global__ void k_router_topk_weights_smalln(
        const float * logits,
        float *       dst,
        const int     n_expert,
        const int     n_rows,
        const int     k,
        const int     n_groups,
        const int     n_group_used,
        const float   expert_weights_scale,
        const int64_t logits_nb0,
        const int64_t logits_nb1,
        const int64_t dst_nb0,
        const int64_t dst_nb1) {
    const int row = blockIdx.x;
    if (row >= n_rows || threadIdx.x != 0) {
        return;
    }

    const float * logits_row = logits + row * logits_nb1;
    float * dst_row = dst + row * dst_nb1;

    float row_max = -FLT_MAX;
    for (int e = 0; e < n_expert; ++e) {
        row_max = fmaxf(row_max, logits_row[e * logits_nb0]);
    }

    float denom = 0.0f;
    for (int e = 0; e < n_expert; ++e) {
        denom += expf(logits_row[e * logits_nb0] - row_max);
    }

    const bool grouped = n_groups > 1 && n_group_used > 0 && n_group_used < n_groups && n_groups <= GROUP_MAX;
    int selected_groups[GROUP_MAX];
    float selected_group_scores[GROUP_MAX];
#pragma unroll
    for (int i = 0; i < GROUP_MAX; ++i) {
        selected_groups[i] = 0x7fffffff;
        selected_group_scores[i] = -FLT_MAX;
    }

    if (grouped) {
        const int group_size = n_expert / n_groups > 0 ? n_expert / n_groups : 1;
        for (int g = 0; g < n_groups; ++g) {
            float top1 = -FLT_MAX;
            float top2 = -FLT_MAX;
            const int begin = g * group_size;
            const int end = g == n_groups - 1 ? n_expert : (begin + group_size < n_expert ? begin + group_size : n_expert);
            for (int e = begin; e < end; ++e) {
                const float v = logits_row[e * logits_nb0];
                if (v > top1) {
                    top2 = top1;
                    top1 = v;
                } else if (v > top2) {
                    top2 = v;
                }
            }
            const float score = expf(top1 - row_max) + (top2 == -FLT_MAX ? 0.0f : expf(top2 - row_max));
            router_topk_insert_pair(score, g, n_group_used, selected_group_scores, selected_groups);
        }
    }

    float top_vals[K_MAX];
    int top_ids[K_MAX];
#pragma unroll
    for (int i = 0; i < K_MAX; ++i) {
        top_vals[i] = -FLT_MAX;
        top_ids[i]  = 0x7fffffff;
    }

    const int group_size = grouped ? (n_expert / n_groups > 0 ? n_expert / n_groups : 1) : n_expert;
    for (int e = 0; e < n_expert; ++e) {
        if (grouped) {
            const int group = e / group_size < n_groups - 1 ? e / group_size : n_groups - 1;
            if (!router_topk_group_allowed(group, selected_groups, n_group_used)) {
                continue;
            }
        }
        router_topk_insert_pair(logits_row[e * logits_nb0], e, k, top_vals, top_ids);
    }

    float selected_probs[K_MAX];
    float selected_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < K_MAX; ++j) {
        selected_probs[j] = 0.0f;
    }

    if (denom > 0.0f) {
        for (int j = 0; j < k; ++j) {
            const int id = top_ids[j] == 0x7fffffff ? 0 : top_ids[j];
            const float p = expf(logits_row[id * logits_nb0] - row_max) / denom;
            selected_probs[j] = p;
            selected_sum += p;
            dst_row[j * dst_nb0] = (float) id;
        }
    } else {
        for (int j = 0; j < k; ++j) {
            const int id = top_ids[j] == 0x7fffffff ? 0 : top_ids[j];
            dst_row[j * dst_nb0] = (float) id;
        }
    }

    selected_sum = fmaxf(selected_sum, 6.103515625e-5f);
    const float scale = (expert_weights_scale != 0.0f && expert_weights_scale != 1.0f) ? expert_weights_scale : 1.0f;
    for (int j = 0; j < k; ++j) {
        dst_row[(k + j) * dst_nb0] = (selected_probs[j] / selected_sum) * scale;
    }
}

void ggml_cuda_op_router_topk_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->ne[2] == 1 && logits->ne[3] == 1);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous_rows(logits));

    const int n_expert = (int) logits->ne[0];
    const int n_rows   = (int) logits->ne[1];
    const int k        = ggml_get_op_params_i32(dst, 0);
    const int n_groups = ggml_get_op_params_i32(dst, 1);
    const int n_group_used = ggml_get_op_params_i32(dst, 2);
    const float expert_weights_scale = ggml_get_op_params_f32(dst, 3);

    GGML_ASSERT(k > 0 && k <= 16);
    GGML_ASSERT(dst->ne[0] == 2*k && dst->ne[1] == n_rows);
    GGML_ASSERT(n_rows > 0 && n_rows <= 4);
    GGML_ASSERT(n_expert > 0 && n_expert <= 512);
    GGML_ASSERT(n_groups >= 1 && n_groups <= 64);

    if (ggml_cuda_router_topk_weights_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=router_topk_weights_roweq_fused_candidate tensor=%s status=selected ncols_dst=%d n_expert=%d n_rows=%d k=%d n_groups=%d n_group_used=%d scale=%g\n",
                __func__, logits->name, n_rows, n_expert, n_rows, k, n_groups, n_group_used, (double) expert_weights_scale);
    }

    k_router_topk_weights_smalln<16, 64><<<n_rows, 1, 0, ctx.stream()>>>(
            (const float *) logits->data,
            (float *) dst->data,
            n_expert, n_rows, k, n_groups, n_group_used, expert_weights_scale,
            logits->nb[0] / (int64_t) sizeof(float), logits->nb[1] / (int64_t) sizeof(float),
            dst->nb[0] / (int64_t) sizeof(float), dst->nb[1] / (int64_t) sizeof(float));
}

static bool ggml_cuda_moe_routed_lanes_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MOE_ROUTED_LANES_LOG");
    return env != nullptr && env[0] != '\0' && atoi(env) != 0;
}

static __global__ void k_moe_routed_lanes_smalln(
        const int32_t * selected,
        int32_t *       dst,
        const int       n_expert,
        const int       n_slots,
        const int       n_rows,
        const int64_t   selected_nb0,
        const int64_t   selected_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const int n_lanes = n_slots * n_rows;
    const int total_rows = n_lanes + n_expert;
    for (int r = 0; r < total_rows; ++r) {
        for (int c = 0; c < 4; ++c) {
            *(int32_t *) ((char *) dst + c*dst_nb0 + r*dst_nb1) = -1;
        }
    }

    int lane = 0;
    for (int expert = 0; expert < n_expert; ++expert) {
        const int expert_row = n_lanes + expert;
        const int start = lane;
        for (int row = 0; row < n_rows; ++row) {
            for (int slot = 0; slot < n_slots; ++slot) {
                const int32_t id = *(const int32_t *) ((const char *) selected + slot*selected_nb0 + row*selected_nb1);
                if (id != expert) {
                    continue;
                }
                *(int32_t *) ((char *) dst + 0*dst_nb0 + lane*dst_nb1) = id;
                *(int32_t *) ((char *) dst + 1*dst_nb0 + lane*dst_nb1) = row;
                *(int32_t *) ((char *) dst + 2*dst_nb0 + lane*dst_nb1) = slot;
                *(int32_t *) ((char *) dst + 3*dst_nb0 + lane*dst_nb1) = row*n_slots + slot;
                ++lane;
            }
        }
        *(int32_t *) ((char *) dst + 0*dst_nb0 + expert_row*dst_nb1) = start;
        *(int32_t *) ((char *) dst + 1*dst_nb0 + expert_row*dst_nb1) = lane - start;
        *(int32_t *) ((char *) dst + 2*dst_nb0 + expert_row*dst_nb1) = expert;
        *(int32_t *) ((char *) dst + 3*dst_nb0 + expert_row*dst_nb1) = 0;
    }
}

void ggml_cuda_op_moe_routed_lanes(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * selected = dst->src[0];
    const ggml_tensor * weights  = dst->src[1];

    GGML_ASSERT(selected && selected->type == GGML_TYPE_I32);
    GGML_ASSERT(weights  && weights->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(selected->ne[2] == 1 && selected->ne[3] == 1);

    const int n_expert = ggml_get_op_params_i32(dst, 0);
    const int n_slots  = (int) selected->ne[0];
    const int n_rows   = (int) selected->ne[1];
    const int n_lanes  = n_slots * n_rows;

    GGML_ASSERT(n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(n_slots > 0 && n_slots <= 16);
    GGML_ASSERT(n_rows > 0 && n_rows <= 64);
    GGML_ASSERT(dst->ne[0] == 4 && dst->ne[1] == n_lanes + n_expert);
    GGML_ASSERT(weights->ne[0] == 1 && weights->ne[1] == n_slots && weights->ne[2] == n_rows);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_metadata tensor=%s status=selected n_expert=%d n_slots=%d n_rows=%d n_lanes=%d\n",
                __func__, selected->name, n_expert, n_slots, n_rows, n_lanes);
    }

    k_moe_routed_lanes_smalln<<<1, 1, 0, ctx.stream()>>>(
            (const int32_t *) selected->data,
            (int32_t *) dst->data,
            n_expert, n_slots, n_rows,
            selected->nb[0], selected->nb[1], dst->nb[0], dst->nb[1]);
}

static __global__ void k_moe_routed_lanes_row_slot_map_smalln(
        const int32_t * lanes,
        int32_t *       dst,
        const int       n_slots,
        const int       n_rows,
        const int       n_lanes,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int row = 0; row < n_rows; ++row) {
        for (int slot = 0; slot < n_slots; ++slot) {
            *(int32_t *) ((char *) dst + slot*dst_nb0 + row*dst_nb1) = -1;
        }
    }

    for (int lane = 0; lane < n_lanes; ++lane) {
        const int32_t row  = *(const int32_t *) ((const char *) lanes + 1*lanes_nb0 + lane*lanes_nb1);
        const int32_t slot = *(const int32_t *) ((const char *) lanes + 2*lanes_nb0 + lane*lanes_nb1);
        if (row >= 0 && row < n_rows && slot >= 0 && slot < n_slots) {
            *(int32_t *) ((char *) dst + slot*dst_nb0 + (int64_t) row*dst_nb1) = lane;
        }
    }
}

void ggml_cuda_op_moe_routed_lanes_row_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * lanes   = dst->src[1];

    GGML_ASSERT(weights && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(lanes && lanes->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(lanes->ne[0] == 4);

    const int n_expert = ggml_get_op_params_i32(lanes, 0);
    const int n_slots  = (int) weights->ne[1];
    const int n_rows   = (int) weights->ne[2];
    const int n_lanes  = n_slots*n_rows;

    GGML_ASSERT(n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(n_slots > 0 && n_slots <= 16);
    GGML_ASSERT(n_rows > 0 && n_rows <= 64);
    GGML_ASSERT(lanes->ne[1] == n_lanes + n_expert);
    GGML_ASSERT(dst->ne[0] == n_slots && dst->ne[1] == n_rows);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_row_slot_map_diag tensor=%s status=selected n_slots=%d n_rows=%d n_lanes=%d\n",
                __func__, lanes->name, n_slots, n_rows, n_lanes);
    }

    k_moe_routed_lanes_row_slot_map_smalln<<<1, 1, 0, ctx.stream()>>>(
            (const int32_t *) lanes->data,
            (int32_t *) dst->data,
            n_slots, n_rows, n_lanes,
            lanes->nb[0], lanes->nb[1], dst->nb[0], dst->nb[1]);
}

static __global__ void k_moe_routed_lanes_expert_bounds_smalln(
        const int32_t * lanes,
        int32_t *       dst,
        const int       n_expert,
        const int       n_lanes,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    const int expert = (int) blockIdx.x*blockDim.x + threadIdx.x;
    if (expert >= n_expert) {
        return;
    }

    const int expert_row = n_lanes + expert;
    const int32_t start = *(const int32_t *) ((const char *) lanes + 0*lanes_nb0 + (int64_t) expert_row*lanes_nb1);
    const int32_t count = *(const int32_t *) ((const char *) lanes + 1*lanes_nb0 + (int64_t) expert_row*lanes_nb1);
    *(int32_t *) ((char *) dst + 0*dst_nb0 + (int64_t) expert*dst_nb1) = start;
    *(int32_t *) ((char *) dst + 1*dst_nb0 + (int64_t) expert*dst_nb1) = count;
}

void ggml_cuda_op_moe_routed_lanes_expert_bounds(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * lanes = dst->src[0];

    GGML_ASSERT(lanes && lanes->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(lanes->ne[0] == 4);

    const int n_expert = ggml_get_op_params_i32(lanes, 0);
    const int n_lanes = (int) (lanes->ne[1] - n_expert);
    GGML_ASSERT(n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(n_lanes > 0);
    GGML_ASSERT(dst->ne[0] == 2 && dst->ne[1] == n_expert);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_expert_bounds_diag tensor=%s status=selected n_expert=%d n_lanes=%d\n",
                __func__, lanes->name, n_expert, n_lanes);
    }

    const int threads = 256;
    const int blocks = (n_expert + threads - 1) / threads;
    k_moe_routed_lanes_expert_bounds_smalln<<<blocks, threads, 0, ctx.stream()>>>(
            (const int32_t *) lanes->data,
            (int32_t *) dst->data,
            n_expert, n_lanes,
            lanes->nb[0], lanes->nb[1], dst->nb[0], dst->nb[1]);
}

template <ggml_type type>
struct moe_routed_lanes_projection_quant_traits;

template <>
struct moe_routed_lanes_projection_quant_traits<GGML_TYPE_Q8_0> {
    static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;

    static __device__ __forceinline__ float vec_dot(
            const void * __restrict__ weights,
            const block_q8_1 * __restrict__ compact_lane,
            const int kbx,
            const int kqs) {
        return vec_dot_q8_0_q8_1(weights, compact_lane, kbx, kqs);
    }
};

template <>
struct moe_routed_lanes_projection_quant_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int vdr = VDR_IQ4_XS_Q8_1_MMVQ;

    static __device__ __forceinline__ float vec_dot(
            const void * __restrict__ weights,
            const block_q8_1 * __restrict__ compact_lane,
            const int kbx,
            const int kqs) {
        return vec_dot_iq4_xs_q8_1(weights, compact_lane, kbx, kqs);
    }
};

template <>
struct moe_routed_lanes_projection_quant_traits<GGML_TYPE_IQ3_S> {
    static constexpr int vdr = VDR_IQ3_S_Q8_1_MMVQ;

    static __device__ __forceinline__ float vec_dot(
            const void * __restrict__ weights,
            const block_q8_1 * __restrict__ compact_lane,
            const int kbx,
            const int kqs) {
        return vec_dot_iq3_s_q8_1(weights, compact_lane, kbx, kqs);
    }
};

template <ggml_type type>
static __device__ __forceinline__ void moe_routed_lanes_projection_quant_impl(
        const void *    weights,
        const block_q8_1 * compact_q8,
        const int32_t * lanes,
        const int32_t * bounds,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_out,
        const int64_t   n_expert,
        const int64_t   n_lanes,
        const int64_t   weights_nb1_qblocks,
        const int64_t   weights_nb2_qblocks,
        const int64_t   q8_blocks_per_lane,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   bounds_nb0,
        const int64_t   bounds_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = moe_routed_lanes_projection_quant_traits<type>::vdr;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t expert = blockIdx.y;
    if (expert >= n_expert) {
        return;
    }

    const int64_t idx = (int64_t) blockIdx.x;
    const int32_t start = *(const int32_t *) ((const char *) bounds + 0*bounds_nb0 + expert*bounds_nb1);
    const int32_t count = *(const int32_t *) ((const char *) bounds + 1*bounds_nb0 + expert*bounds_nb1);
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes) {
        assert(false && "invalid routed-lane expert bounds");
        return;
    }
    if (count == 0) {
        return;
    }

    const int64_t total = (int64_t) count*n_out;
    if (idx >= total) {
        return;
    }

    const int64_t out = idx % n_out;
    const int64_t lane = (int64_t) start + idx / n_out;
    const int32_t lane_expert = *(const int32_t *) ((const char *) lanes + 0*lanes_nb0 + lane*lanes_nb1);
    if (lane_expert != expert) {
        assert(false && "routed-lane expert mismatch");
        return;
    }

    const int64_t kbx_offset_64 = out*weights_nb1_qblocks + expert*weights_nb2_qblocks;
    if (kbx_offset_64 > INT_MAX) {
        assert(false && "routed-lane quant projection qblock offset overflow");
        return;
    }
    const int kbx_offset = (int) kbx_offset_64;
    const block_q8_1 * compact_lane = compact_q8 + lane*q8_blocks_per_lane;

    float acc = 0.0f;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    for (int kbx = threadIdx.x / (qi/vdr); kbx < n_in/qk; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
        acc += moe_routed_lanes_projection_quant_traits<type>::vec_dot(weights, compact_lane + kby, kbx_offset + kbx, kqs);
    }

    acc = warp_reduce_sum<warp_size>(acc);
    if (threadIdx.x == 0) {
        *(float *) ((char *) dst + out*dst_nb0 + lane*dst_nb1) = acc;
    }
}

static __global__ void k_moe_routed_lanes_projection_q8_0(
        const void *    weights,
        const block_q8_1 * compact_q8,
        const int32_t * lanes,
        const int32_t * bounds,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_out,
        const int64_t   n_expert,
        const int64_t   n_lanes,
        const int64_t   weights_nb1_qblocks,
        const int64_t   weights_nb2_qblocks,
        const int64_t   q8_blocks_per_lane,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   bounds_nb0,
        const int64_t   bounds_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    moe_routed_lanes_projection_quant_impl<GGML_TYPE_Q8_0>(
            weights, compact_q8, lanes, bounds, dst,
            n_in, n_out, n_expert, n_lanes,
            weights_nb1_qblocks, weights_nb2_qblocks, q8_blocks_per_lane,
            lanes_nb0, lanes_nb1, bounds_nb0, bounds_nb1, dst_nb0, dst_nb1);
}

static __global__ void k_moe_routed_lanes_projection_iq4_xs(
        const void *    weights,
        const block_q8_1 * compact_q8,
        const int32_t * lanes,
        const int32_t * bounds,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_out,
        const int64_t   n_expert,
        const int64_t   n_lanes,
        const int64_t   weights_nb1_qblocks,
        const int64_t   weights_nb2_qblocks,
        const int64_t   q8_blocks_per_lane,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   bounds_nb0,
        const int64_t   bounds_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    moe_routed_lanes_projection_quant_impl<GGML_TYPE_IQ4_XS>(
            weights, compact_q8, lanes, bounds, dst,
            n_in, n_out, n_expert, n_lanes,
            weights_nb1_qblocks, weights_nb2_qblocks, q8_blocks_per_lane,
            lanes_nb0, lanes_nb1, bounds_nb0, bounds_nb1, dst_nb0, dst_nb1);
}

static __global__ void k_moe_routed_lanes_projection_iq3_s(
        const void *    weights,
        const block_q8_1 * compact_q8,
        const int32_t * lanes,
        const int32_t * bounds,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_out,
        const int64_t   n_expert,
        const int64_t   n_lanes,
        const int64_t   weights_nb1_qblocks,
        const int64_t   weights_nb2_qblocks,
        const int64_t   q8_blocks_per_lane,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   bounds_nb0,
        const int64_t   bounds_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    moe_routed_lanes_projection_quant_impl<GGML_TYPE_IQ3_S>(
            weights, compact_q8, lanes, bounds, dst,
            n_in, n_out, n_expert, n_lanes,
            weights_nb1_qblocks, weights_nb2_qblocks, q8_blocks_per_lane,
            lanes_nb0, lanes_nb1, bounds_nb0, bounds_nb1, dst_nb0, dst_nb1);
}

static __global__ void k_moe_routed_lanes_projection_f32(
        const float *   weights,
        const float *   compact,
        const int32_t * lanes,
        const int32_t * bounds,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_out,
        const int64_t   n_expert,
        const int64_t   n_lanes,
        const int64_t   weights_nb0,
        const int64_t   weights_nb1,
        const int64_t   weights_nb2,
        const int64_t   compact_nb0,
        const int64_t   compact_nb1,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   bounds_nb0,
        const int64_t   bounds_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    const int64_t expert = blockIdx.y;
    if (expert >= n_expert) {
        return;
    }

    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int32_t start = *(const int32_t *) ((const char *) bounds + 0*bounds_nb0 + expert*bounds_nb1);
    const int32_t count = *(const int32_t *) ((const char *) bounds + 1*bounds_nb0 + expert*bounds_nb1);
    if (start < 0 || count < 0 || (int64_t) start + count > n_lanes) {
        assert(false && "invalid routed-lane expert bounds");
        return;
    }
    if (count == 0) {
        return;
    }

    const int64_t total = (int64_t) count*n_out;
    if (idx >= total) {
        return;
    }

    const int64_t out = idx % n_out;
    const int64_t lane = (int64_t) start + idx / n_out;
    const int32_t lane_expert = *(const int32_t *) ((const char *) lanes + 0*lanes_nb0 + lane*lanes_nb1);
    if (lane_expert != expert) {
        assert(false && "routed-lane expert mismatch");
        return;
    }

    float acc = 0.0f;
    for (int64_t k = 0; k < n_in; ++k) {
        const float w = *(const float *) ((const char *) weights + k*weights_nb0 + out*weights_nb1 + expert*weights_nb2);
        const float x = *(const float *) ((const char *) compact + k*compact_nb0 + lane*compact_nb1);
        acc += w*x;
    }
    *(float *) ((char *) dst + out*dst_nb0 + lane*dst_nb1) = acc;
}

void ggml_cuda_op_moe_routed_lanes_projection(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * compact = dst->src[1];
    const ggml_tensor * lanes   = dst->src[2];
    const ggml_tensor * bounds  = dst->src[3];

    GGML_ASSERT(weights);
    const bool weights_f32     = weights->type == GGML_TYPE_F32;
    const bool weights_q8_0    = weights->type == GGML_TYPE_Q8_0;
    const bool weights_iq4_xs  = weights->type == GGML_TYPE_IQ4_XS;
    const bool weights_iq3_s   = weights->type == GGML_TYPE_IQ3_S;
    GGML_ASSERT(weights_f32 || weights_q8_0 || weights_iq4_xs || weights_iq3_s);
    GGML_ASSERT(compact && compact->type == GGML_TYPE_F32);
    GGML_ASSERT(lanes && lanes->type == GGML_TYPE_I32);
    GGML_ASSERT(bounds && bounds->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_in     = compact->ne[0];
    const int64_t n_lanes  = compact->ne[1];
    const int64_t n_out    = weights->ne[1];
    const int64_t n_expert = weights->ne[2];

    GGML_ASSERT(n_in > 0 && n_out > 0 && n_lanes > 0 && n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(weights->ne[0] == n_in);
    GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_lanes);
    GGML_ASSERT(lanes->ne[0] == 4 && lanes->ne[1] == n_lanes + n_expert);
    GGML_ASSERT(bounds->ne[0] == 2 && bounds->ne[1] == n_expert);

    if (ggml_cuda_moe_routed_lanes_projection_rdna_i8_packed16(ctx, dst)) {
        return;
    }

    if (weights_q8_0 || weights_iq4_xs || weights_iq3_s) {
        GGML_ASSERT(n_in % ggml_blck_size(weights->type) == 0);
        GGML_ASSERT(compact->nb[0] == (int64_t) sizeof(float));
        GGML_ASSERT(compact->nb[1] % (4*(int64_t) sizeof(float)) == 0);
        GGML_ASSERT(n_in % QK8_1 == 0);
        GGML_ASSERT(n_out <= INT_MAX && n_lanes <= INT_MAX && n_lanes <= INT_MAX / n_out);
        GGML_ASSERT(weights->nb[1] % ggml_type_size(weights->type) == 0);
        GGML_ASSERT(weights->nb[2] % ggml_type_size(weights->type) == 0);

        if (ggml_cuda_moe_routed_lanes_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=selected n_in=%d n_out=%d n_expert=%d n_lanes=%d\n",
                    __func__, weights_q8_0 ? "moe_routed_lanes_projection_q8_0_candidate" :
                            (weights_iq4_xs ? "moe_routed_lanes_projection_iq4_xs_candidate" : "moe_routed_lanes_projection_iq3_s_candidate"),
                    weights->name, (int) n_in, (int) n_out, (int) n_expert, (int) n_lanes);
        }

        cudaStream_t stream = ctx.stream();
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

        const int64_t n_in_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
        const int64_t q8_blocks_per_lane = n_in_padded / QK8_1;
        const size_t size_max = (size_t) -1;
        GGML_ASSERT(q8_blocks_per_lane > 0);
        GGML_ASSERT((size_t) q8_blocks_per_lane <= size_max / sizeof(block_q8_1));
        const size_t q8_bytes_per_lane = (size_t) q8_blocks_per_lane * sizeof(block_q8_1);
        GGML_ASSERT((size_t) n_lanes <= size_max / q8_bytes_per_lane);
        const size_t q8_bytes = (size_t) n_lanes * q8_bytes_per_lane;
        ggml_cuda_pool_alloc<char> compact_q8_alloc(ctx.pool(), q8_bytes);
        char * compact_q8 = compact_q8_alloc.get();

        const int64_t compact_s1 = compact->nb[1] / sizeof(float);
        const int64_t compact_s2 = compact->nb[2] / sizeof(float);
        const int64_t compact_s3 = compact->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) compact->data, nullptr, compact_q8, weights->type,
                n_in, compact_s1, compact_s2, compact_s3,
                n_in_padded, n_lanes, 1, 1, stream);

        const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
        const int blocks_x = (int) (n_out*n_lanes);
        const dim3 blocks(blocks_x, (unsigned int) n_expert, 1);
        const dim3 threads(warp_size, 1, 1);
        if (weights_q8_0) {
            k_moe_routed_lanes_projection_q8_0<<<blocks, threads, 0, stream>>>(
                    weights->data,
                    (const block_q8_1 *) compact_q8,
                    (const int32_t *) lanes->data,
                    (const int32_t *) bounds->data,
                    (float *) dst->data,
                    n_in, n_out, n_expert, n_lanes,
                    weights->nb[1] / ggml_type_size(weights->type),
                    weights->nb[2] / ggml_type_size(weights->type),
                    q8_blocks_per_lane,
                    lanes->nb[0], lanes->nb[1], bounds->nb[0], bounds->nb[1], dst->nb[0], dst->nb[1]);
        } else if (weights_iq4_xs) {
            k_moe_routed_lanes_projection_iq4_xs<<<blocks, threads, 0, stream>>>(
                    weights->data,
                    (const block_q8_1 *) compact_q8,
                    (const int32_t *) lanes->data,
                    (const int32_t *) bounds->data,
                    (float *) dst->data,
                    n_in, n_out, n_expert, n_lanes,
                    weights->nb[1] / ggml_type_size(weights->type),
                    weights->nb[2] / ggml_type_size(weights->type),
                    q8_blocks_per_lane,
                    lanes->nb[0], lanes->nb[1], bounds->nb[0], bounds->nb[1], dst->nb[0], dst->nb[1]);
        } else {
            GGML_ASSERT(weights_iq3_s);
            k_moe_routed_lanes_projection_iq3_s<<<blocks, threads, 0, stream>>>(
                    weights->data,
                    (const block_q8_1 *) compact_q8,
                    (const int32_t *) lanes->data,
                    (const int32_t *) bounds->data,
                    (float *) dst->data,
                    n_in, n_out, n_expert, n_lanes,
                    weights->nb[1] / ggml_type_size(weights->type),
                    weights->nb[2] / ggml_type_size(weights->type),
                    q8_blocks_per_lane,
                    lanes->nb[0], lanes->nb[1], bounds->nb[0], bounds->nb[1], dst->nb[0], dst->nb[1]);
        }
        return;
    }

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_projection_f32_diag tensor=%s status=selected n_in=%d n_out=%d n_expert=%d n_lanes=%d\n",
                __func__, weights->name, (int) n_in, (int) n_out, (int) n_expert, (int) n_lanes);
    }

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    const int threads = 256;
    const int blocks_x = (int) ((n_out*n_lanes + threads - 1) / threads);
    const dim3 blocks(blocks_x, (unsigned int) n_expert, 1);
    k_moe_routed_lanes_projection_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) weights->data,
            (const float *) compact->data,
            (const int32_t *) lanes->data,
            (const int32_t *) bounds->data,
            (float *) dst->data,
            n_in, n_out, n_expert, n_lanes,
            weights->nb[0], weights->nb[1], weights->nb[2], compact->nb[0], compact->nb[1],
            lanes->nb[0], lanes->nb[1], bounds->nb[0], bounds->nb[1], dst->nb[0], dst->nb[1]);
}

static __global__ void k_moe_routed_lanes_pack_slots_f32(
        const float *   slot_x,
        const int32_t * row_slot_to_lane,
        float *         dst,
        const int64_t   n_in,
        const int64_t   n_slots,
        const int64_t   n_rows,
        const int64_t   n_lanes,
        const int64_t   slot_x_nb0,
        const int64_t   slot_x_nb1,
        const int64_t   slot_x_nb2,
        const int64_t   map_nb0,
        const int64_t   map_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = n_in*n_slots*n_rows;
    if (idx >= total) {
        return;
    }

    const int64_t i = idx % n_in;
    const int64_t slot = (idx / n_in) % n_slots;
    const int64_t row = idx / (n_in*n_slots);
    const int32_t lane = *(const int32_t *) ((const char *) row_slot_to_lane + slot*map_nb0 + row*map_nb1);
    if (lane >= 0 && lane < n_lanes) {
        const float v = *(const float *) ((const char *) slot_x + i*slot_x_nb0 + slot*slot_x_nb1 + row*slot_x_nb2);
        *(float *) ((char *) dst + i*dst_nb0 + (int64_t) lane*dst_nb1) = v;
    }
}

void ggml_cuda_op_moe_routed_lanes_pack_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * slot_x = dst->src[0];
    const ggml_tensor * row_slot_to_lane = dst->src[1];

    GGML_ASSERT(slot_x && slot_x->type == GGML_TYPE_F32);
    GGML_ASSERT(row_slot_to_lane && row_slot_to_lane->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_in    = slot_x->ne[0];
    const int64_t n_slots = row_slot_to_lane->ne[0];
    const int64_t n_rows  = row_slot_to_lane->ne[1];
    const int64_t n_lanes = dst->ne[1];

    GGML_ASSERT(n_in > 0);
    GGML_ASSERT(n_slots > 0 && n_slots <= 16);
    GGML_ASSERT(n_rows > 0 && n_rows <= 64);
    GGML_ASSERT(n_lanes == n_slots*n_rows);
    GGML_ASSERT(slot_x->ne[1] == n_slots && slot_x->ne[2] == n_rows && slot_x->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == n_in && dst->ne[1] == n_lanes);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_pack_slots_diag tensor=%s status=selected n_in=%d n_slots=%d n_rows=%d n_lanes=%d\n",
                __func__, slot_x->name, (int) n_in, (int) n_slots, (int) n_rows, (int) n_lanes);
    }

    const int threads = 256;
    const int64_t total = n_in*n_slots*n_rows;
    const int blocks = (int) ((total + threads - 1) / threads);
    k_moe_routed_lanes_pack_slots_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) slot_x->data,
            (const int32_t *) row_slot_to_lane->data,
            (float *) dst->data,
            n_in, n_slots, n_rows, n_lanes,
            slot_x->nb[0], slot_x->nb[1], slot_x->nb[2], row_slot_to_lane->nb[0], row_slot_to_lane->nb[1],
            dst->nb[0], dst->nb[1]);
}

static __global__ void k_moe_routed_lanes_unpack_slots_f32(
        const float *   compact,
        const int32_t * row_slot_to_lane,
        float *         dst,
        const int64_t   n_out,
        const int64_t   n_slots,
        const int64_t   n_rows,
        const int64_t   n_lanes,
        const int64_t   compact_nb0,
        const int64_t   compact_nb1,
        const int64_t   map_nb0,
        const int64_t   map_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1,
        const int64_t   dst_nb2) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = n_out*n_slots*n_rows;
    if (idx >= total) {
        return;
    }

    const int64_t i = idx % n_out;
    const int64_t slot = (idx / n_out) % n_slots;
    const int64_t row = idx / (n_out*n_slots);
    const int32_t lane = *(const int32_t *) ((const char *) row_slot_to_lane + slot*map_nb0 + row*map_nb1);
    if (lane >= 0 && lane < n_lanes) {
        const float v = *(const float *) ((const char *) compact + i*compact_nb0 + (int64_t) lane*compact_nb1);
        *(float *) ((char *) dst + i*dst_nb0 + slot*dst_nb1 + row*dst_nb2) = v;
    }
}

void ggml_cuda_op_moe_routed_lanes_unpack_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * compact = dst->src[0];
    const ggml_tensor * row_slot_to_lane = dst->src[1];

    GGML_ASSERT(compact && compact->type == GGML_TYPE_F32);
    GGML_ASSERT(row_slot_to_lane && row_slot_to_lane->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_out   = compact->ne[0];
    const int64_t n_lanes = compact->ne[1];
    const int64_t n_slots = row_slot_to_lane->ne[0];
    const int64_t n_rows  = row_slot_to_lane->ne[1];

    GGML_ASSERT(n_out > 0);
    GGML_ASSERT(n_slots > 0 && n_slots <= 16);
    GGML_ASSERT(n_rows > 0 && n_rows <= 64);
    GGML_ASSERT(n_lanes == n_slots*n_rows);
    GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_slots && dst->ne[2] == n_rows);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_unpack_slots_diag tensor=%s status=selected n_out=%d n_slots=%d n_rows=%d n_lanes=%d\n",
                __func__, compact->name, (int) n_out, (int) n_slots, (int) n_rows, (int) n_lanes);
    }

    const int threads = 256;
    const int64_t total = n_out*n_slots*n_rows;
    const int blocks = (int) ((total + threads - 1) / threads);
    k_moe_routed_lanes_unpack_slots_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) compact->data,
            (const int32_t *) row_slot_to_lane->data,
            (float *) dst->data,
            n_out, n_slots, n_rows, n_lanes,
            compact->nb[0], compact->nb[1], row_slot_to_lane->nb[0], row_slot_to_lane->nb[1],
            dst->nb[0], dst->nb[1], dst->nb[2]);
}

static __global__ void k_moe_routed_lanes_gather_f32(
        const float *   x,
        const int32_t * lanes,
        float *         dst,
        const int64_t   n_embd,
        const int64_t   n_lanes,
        const int64_t   x_nb0,
        const int64_t   x_nb1,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = n_embd*n_lanes;
    if (idx >= total) {
        return;
    }

    const int64_t i = idx % n_embd;
    const int64_t lane = idx / n_embd;
    const int32_t row = *(const int32_t *) ((const char *) lanes + 1*lanes_nb0 + lane*lanes_nb1);
    const float v = *(const float *) ((const char *) x + i*x_nb0 + (int64_t) row*x_nb1);
    *(float *) ((char *) dst + i*dst_nb0 + lane*dst_nb1) = v;
}

void ggml_cuda_op_moe_routed_lanes_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * lanes = dst->src[1];

    GGML_ASSERT(x && x->type == GGML_TYPE_F32);
    GGML_ASSERT(lanes && lanes->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(lanes->ne[0] == 4);

    const int n_expert = ggml_get_op_params_i32(lanes, 0);
    const int64_t n_lanes = lanes->ne[1] - n_expert;
    GGML_ASSERT(n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(n_lanes > 0 && dst->ne[1] == n_lanes);
    GGML_ASSERT(dst->ne[0] == x->ne[0]);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_gather_diag tensor=%s status=selected n_embd=%d n_lanes=%d\n",
                __func__, x->name, (int) x->ne[0], (int) n_lanes);
    }

    const int threads = 256;
    const int64_t total = x->ne[0]*n_lanes;
    const int blocks = (int) ((total + threads - 1) / threads);
    k_moe_routed_lanes_gather_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) x->data,
            (const int32_t *) lanes->data,
            (float *) dst->data,
            x->ne[0], n_lanes,
            x->nb[0], x->nb[1], lanes->nb[0], lanes->nb[1], dst->nb[0], dst->nb[1]);
}

static __global__ void k_moe_routed_lanes_scatter_reduce_f32(
        const float *   compact,
        const float *   weights,
        const int32_t * lanes,
        float *         dst,
        const int64_t   n_embd,
        const int64_t   n_lanes,
        const int64_t   n_slots,
        const int64_t   n_rows,
        const int64_t   compact_nb0,
        const int64_t   compact_nb1,
        const int64_t   weights_nb0,
        const int64_t   weights_nb1,
        const int64_t   weights_nb2,
        const int64_t   lanes_nb0,
        const int64_t   lanes_nb1,
        const int64_t   dst_nb0,
        const int64_t   dst_nb1) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = n_embd*n_rows;
    if (idx >= total) {
        return;
    }

    const int64_t i = idx % n_embd;
    const int64_t row = idx / n_embd;
    float acc = 0.0f;
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int32_t key = (int32_t) (row*n_slots + slot);
        int64_t lane_found = -1;
        for (int64_t lane = 0; lane < n_lanes; ++lane) {
            const int32_t lane_key = *(const int32_t *) ((const char *) lanes + 3*lanes_nb0 + lane*lanes_nb1);
            if (lane_key == key) {
                lane_found = lane;
                break;
            }
        }
        if (lane_found >= 0) {
            const float v = *(const float *) ((const char *) compact + i*compact_nb0 + lane_found*compact_nb1);
            const float w = *(const float *) ((const char *) weights + 0*weights_nb0 + slot*weights_nb1 + row*weights_nb2);
            acc += v*w;
        }
    }
    *(float *) ((char *) dst + i*dst_nb0 + row*dst_nb1) = acc;
}

void ggml_cuda_op_moe_routed_lanes_scatter_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * compact = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * lanes   = dst->src[2];

    GGML_ASSERT(compact && compact->type == GGML_TYPE_F32);
    GGML_ASSERT(weights && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(lanes && lanes->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_expert = ggml_get_op_params_i32(lanes, 0);
    const int64_t n_slots = weights->ne[1];
    const int64_t n_rows  = weights->ne[2];
    const int64_t n_lanes = n_slots*n_rows;
    GGML_ASSERT(n_expert > 0 && n_expert <= 4096);
    GGML_ASSERT(n_slots > 0 && n_slots <= 16);
    GGML_ASSERT(n_rows > 0 && n_rows <= 64);
    GGML_ASSERT(compact->ne[1] == n_lanes);
    GGML_ASSERT(lanes->ne[0] == 4 && lanes->ne[1] == n_lanes + n_expert);
    GGML_ASSERT(dst->ne[0] == compact->ne[0] && dst->ne[1] == n_rows);

    if (ggml_cuda_moe_routed_lanes_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=moe_routed_lanes_scatter_reduce_diag tensor=%s status=selected n_embd=%d n_slots=%d n_rows=%d n_lanes=%d\n",
                __func__, compact->name, (int) compact->ne[0], (int) n_slots, (int) n_rows, (int) n_lanes);
    }

    const int threads = 256;
    const int64_t total = compact->ne[0]*n_rows;
    const int blocks = (int) ((total + threads - 1) / threads);
    k_moe_routed_lanes_scatter_reduce_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) compact->data,
            (const float *) weights->data,
            (const int32_t *) lanes->data,
            (float *) dst->data,
            compact->ne[0], n_lanes, n_slots, n_rows,
            compact->nb[0], compact->nb[1], weights->nb[0], weights->nb[1], weights->nb[2],
            lanes->nb[0], lanes->nb[1], dst->nb[0], dst->nb[1]);
}

void ggml_cuda_op_lm_head_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // LM head weight [n_embd, n_vocab]
    const ggml_tensor * src1 = dst->src[1]; // hidden rows [n_embd, n_rows]

    if (src0->type == GGML_TYPE_Q8_0) {
        ggml_cuda_op_lm_head_top_k_q8_0(ctx, dst);
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == 1 && "prototype LM_HEAD_TOP_K currently supports exact top1 only");
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_is_contiguous_rows(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd  = src0->ne[0];
    const int64_t n_vocab = src0->ne[1];
    const int64_t n_rows  = ggml_nrows(src1);

    const int n_ban_raw = ggml_get_op_params_i32(dst, 0);
    const int n_ban     = n_ban_raw < 0 ? 0 : (n_ban_raw > 8 ? 8 : n_ban_raw);
    const int ban0      = n_ban > 0 ? ggml_get_op_params_i32(dst, 1) : -1;
    const int ban1      = n_ban > 1 ? ggml_get_op_params_i32(dst, 2) : -1;
    const int ban2      = n_ban > 2 ? ggml_get_op_params_i32(dst, 3) : -1;
    const int ban3      = n_ban > 3 ? ggml_get_op_params_i32(dst, 4) : -1;
    const int ban4      = n_ban > 4 ? ggml_get_op_params_i32(dst, 5) : -1;
    const int ban5      = n_ban > 5 ? ggml_get_op_params_i32(dst, 6) : -1;
    const int ban6      = n_ban > 6 ? ggml_get_op_params_i32(dst, 7) : -1;
    const int ban7      = n_ban > 7 ? ggml_get_op_params_i32(dst, 8) : -1;

    if (const char * log_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
        if (atoi(log_env) != 0) {
            GGML_LOG_INFO("%s: route=cuda_lm_head_top1_q6k tensor=%s n_embd=%lld n_vocab=%lld n_rows=%lld ban_count=%d src1_cont_rows=%d src1_cont=%d src1_nb=(%lld,%lld,%lld,%lld) src0_buf=%s src1_buf=%s dst_buf=%s dst=%s\n",
                    __func__, src0->name, (long long) n_embd, (long long) n_vocab, (long long) n_rows, n_ban,
                    ggml_is_contiguous_rows(src1) ? 1 : 0, ggml_is_contiguous(src1) ? 1 : 0,
                    (long long) src1->nb[0], (long long) src1->nb[1], (long long) src1->nb[2], (long long) src1->nb[3],
                    src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "null",
                    src1->buffer ? ggml_backend_buffer_name(src1->buffer) : "null",
                    dst->buffer  ? ggml_backend_buffer_name(dst->buffer)  : "null",
                    dst->name);
        }
    }

    GGML_ASSERT(n_embd % QK_K == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t q8_blocks_per_row = n_embd_padded / QK8_1;
    const size_t  q8_bytes = (size_t) n_rows * q8_blocks_per_row * sizeof(block_q8_1);

    ggml_cuda_pool_alloc<char> h_q8_alloc(ctx.pool(), q8_bytes);
    char * h_q8 = h_q8_alloc.get();

    const float * src1_d = (const float *) src1->data;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    quantize_row_q8_1_cuda(src1_d, nullptr, h_q8, src0->type, n_embd, s11, s12, s13,
            n_embd_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    int rows_per_block = 16;
    if (const char * rows_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_ROWS_PER_BLOCK")) {
        const int requested = atoi(rows_env);
        if ((requested == 8 || requested == 16 || requested == 32) && requested * warp_size <= 1024) {
            rows_per_block = requested;
        }
    }
    const int stride_row_x = src0->nb[1] / ggml_type_size(src0->type);
    const int blocks_per_row_x = n_embd / QK_K;

    if (n_rows > 1) {
        if (const char * multirow_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_MULTIROW")) {
            if (atoi(multirow_env) != 0) {
#define LLAMA_LM_HEAD_TOP1_DISPATCH_MULTIROW_RPB(RPB) do { \
                    constexpr int rows_per_block_t = (RPB); \
                    constexpr int hidden_rows_per_group_t = 8; \
                    const dim3 block_dims(warp_size, rows_per_block_t, 1); \
                    const int n_partials = (n_vocab + rows_per_block_t - 1) / rows_per_block_t; \
                    const dim3 stage1_grid(n_partials, (n_rows + hidden_rows_per_group_t - 1) / hidden_rows_per_group_t, 1); \
                    ggml_cuda_pool_alloc<float> partial_vals_alloc(ctx.pool(), (size_t) n_rows * n_partials); \
                    ggml_cuda_pool_alloc<int>   partial_ids_alloc (ctx.pool(), (size_t) n_rows * n_partials); \
                    float * partial_vals = partial_vals_alloc.get(); \
                    int *   partial_ids  = partial_ids_alloc.get(); \
                    k_lm_head_top1_q6k_stage1_multirow<rows_per_block_t, hidden_rows_per_group_t><<<stage1_grid, block_dims, 0, stream>>>( \
                            src0->data, (const block_q8_1 *) h_q8, partial_vals, partial_ids, \
                            n_vocab, n_rows, blocks_per_row_x, q8_blocks_per_row, stride_row_x, n_partials, \
                            n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
                    const dim3 stage2_grid(n_rows, 1, 1); \
                    constexpr int stage2_block = 256; \
                    const size_t stage2_smem = stage2_block * (sizeof(float) + sizeof(int)); \
                    k_lm_head_top1_stage2<<<stage2_grid, stage2_block, stage2_smem, stream>>>( \
                            partial_vals, partial_ids, (int *) dst->data, n_partials); \
                } while (0)
                if (const char * log_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_LOG")) {
                    if (atoi(log_env) != 0) {
                        GGML_LOG_INFO("%s: multirow=1 hidden_rows_per_group=8\n", __func__);
                    }
                }
                switch (rows_per_block) {
                    case 8:  LLAMA_LM_HEAD_TOP1_DISPATCH_MULTIROW_RPB(8);  break;
                    case 32: LLAMA_LM_HEAD_TOP1_DISPATCH_MULTIROW_RPB(32); break;
                    case 16:
                    default: LLAMA_LM_HEAD_TOP1_DISPATCH_MULTIROW_RPB(16); break;
                }
#undef LLAMA_LM_HEAD_TOP1_DISPATCH_MULTIROW_RPB
                return;
            }
        }
    }

#define LLAMA_LM_HEAD_TOP1_DISPATCH_RPB(RPB) do { \
        constexpr int rows_per_block_t = (RPB); \
        const dim3 block_dims(warp_size, rows_per_block_t, 1); \
        const dim3 stage1_grid((n_vocab + rows_per_block_t - 1) / rows_per_block_t, n_rows, 1); \
        const char * atomic_top1_env = getenv("LLAMA_MTP_FUSED_LM_HEAD_TOPK_ATOMIC"); \
        if (atomic_top1_env && atoi(atomic_top1_env) != 0) { \
            ggml_cuda_pool_alloc<unsigned long long> best_keys_alloc(ctx.pool(), (size_t) n_rows); \
            unsigned long long * best_keys = best_keys_alloc.get(); \
            constexpr int init_block = 256; \
            const int init_grid = (n_rows + init_block - 1) / init_block; \
            k_lm_head_top1_init_keys<<<init_grid, init_block, 0, stream>>>(best_keys, n_rows); \
            k_lm_head_top1_q6k_atomic<rows_per_block_t><<<stage1_grid, block_dims, 0, stream>>>( \
                    src0->data, (const block_q8_1 *) h_q8, best_keys, \
                    n_vocab, blocks_per_row_x, q8_blocks_per_row, stride_row_x, \
                    n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
            k_lm_head_top1_keys_to_ids<<<init_grid, init_block, 0, stream>>>(best_keys, (int *) dst->data, n_rows); \
            return; \
        } \
        ggml_cuda_pool_alloc<float> partial_vals_alloc(ctx.pool(), (size_t) n_rows * ((n_vocab + rows_per_block_t - 1) / rows_per_block_t)); \
        ggml_cuda_pool_alloc<int>   partial_ids_alloc (ctx.pool(), (size_t) n_rows * ((n_vocab + rows_per_block_t - 1) / rows_per_block_t)); \
        float * partial_vals = partial_vals_alloc.get(); \
        int *   partial_ids  = partial_ids_alloc.get(); \
        k_lm_head_top1_q6k_stage1<rows_per_block_t><<<stage1_grid, block_dims, 0, stream>>>( \
                src0->data, (const block_q8_1 *) h_q8, partial_vals, partial_ids, \
                n_vocab, blocks_per_row_x, q8_blocks_per_row, stride_row_x, (n_vocab + rows_per_block_t - 1) / rows_per_block_t, \
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7); \
        const dim3 stage2_grid(n_rows, 1, 1); \
        constexpr int stage2_block = 256; \
        const size_t stage2_smem = stage2_block * (sizeof(float) + sizeof(int)); \
        k_lm_head_top1_stage2<<<stage2_grid, stage2_block, stage2_smem, stream>>>( \
                partial_vals, partial_ids, (int *) dst->data, (n_vocab + rows_per_block_t - 1) / rows_per_block_t); \
    } while (0)

    switch (rows_per_block) {
        case 8:  LLAMA_LM_HEAD_TOP1_DISPATCH_RPB(8);  break;
        case 32: LLAMA_LM_HEAD_TOP1_DISPATCH_RPB(32); break;
        case 16:
        default: LLAMA_LM_HEAD_TOP1_DISPATCH_RPB(16); break;
    }

#undef LLAMA_LM_HEAD_TOP1_DISPATCH_RPB
}


void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    const int n_ban_raw = ggml_get_op_params_i32(dst, 0);
    const int n_ban     = n_ban_raw < 0 ? 0 : (n_ban_raw > 8 ? 8 : n_ban_raw);
    const int ban0      = n_ban > 0 ? ggml_get_op_params_i32(dst, 1) : -1;
    const int ban1      = n_ban > 1 ? ggml_get_op_params_i32(dst, 2) : -1;
    const int ban2      = n_ban > 2 ? ggml_get_op_params_i32(dst, 3) : -1;
    const int ban3      = n_ban > 3 ? ggml_get_op_params_i32(dst, 4) : -1;
    const int ban4      = n_ban > 4 ? ggml_get_op_params_i32(dst, 5) : -1;
    const int ban5      = n_ban > 5 ? ggml_get_op_params_i32(dst, 6) : -1;
    const int ban6      = n_ban > 6 ? ggml_get_op_params_i32(dst, 7) : -1;
    const int ban7      = n_ban > 7 ? ggml_get_op_params_i32(dst, 8) : -1;
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    if ((n_ban > 0 || ncols > 1024) && k <= 16) {
        top_k_f32_i32_small_k_cuda(pool, src0_d, dst_d, ncols, nrows, k, src0, dst, stream,
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);
        return;
    }
    GGML_ASSERT(n_ban == 0 && "CUB TOP_K ban-list path is not implemented");
    GGML_ASSERT(ggml_is_contiguous(src0));
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if ((n_ban > 0 || ncols > 1024) && k <= 16) {
        top_k_f32_i32_small_k_cuda(pool, src0_d, dst_d, ncols, nrows, k, src0, dst, stream,
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(src0));
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    if (n_ban > 0 || ncols > 1024) {
        GGML_ASSERT(k <= 16);
        top_k_f32_i32_small_k_cuda(pool, src0_d, dst_d, ncols, nrows, k, src0, dst, stream,
                n_ban, ban0, ban1, ban2, ban3, ban4, ban5, ban6, ban7);
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(src0));
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
