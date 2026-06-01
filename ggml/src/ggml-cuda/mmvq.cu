#include "mmvq.cuh"
#include "mmvq-rdna3-dot4.cuh"
#include "dot4-packed16/mmvq/dp16-mmvq-q8-gemv.cuh"
#include "dot4-packed16/mmvq/dp16-mmvq-packed16-gemv.cuh"
#include "dot4-packed16/dp16-trace.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <mutex>
#include <unordered_map>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

struct ggml_cuda_dp16_packed16_weight_cache_key {
    const ggml_tensor * tensor;
    const void * data;
    ggml_backend_buffer_t buffer;
    int device;
    int64_t cols;
    int64_t rows;
    int64_t channels;
    int64_t samples;
    size_t nb0;
    size_t nb1;
    size_t nb2;
    size_t nb3;
    ggml_type source_type;
    dp16_layout layout;
    dp16_correction_policy correction;
    uint32_t flags;

    bool operator==(const ggml_cuda_dp16_packed16_weight_cache_key & other) const {
        return tensor == other.tensor && data == other.data && buffer == other.buffer && device == other.device &&
            cols == other.cols && rows == other.rows && channels == other.channels && samples == other.samples &&
            nb0 == other.nb0 && nb1 == other.nb1 && nb2 == other.nb2 && nb3 == other.nb3 &&
            source_type == other.source_type && layout == other.layout && correction == other.correction && flags == other.flags;
    }
};

struct ggml_cuda_dp16_packed16_weight_cache_key_hash {
    size_t operator()(const ggml_cuda_dp16_packed16_weight_cache_key & key) const {
        size_t h = std::hash<const void *>{}(key.tensor);
        const auto mix = [&](const size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<const void *>{}(key.data));
        mix(std::hash<const void *>{}(key.buffer));
        mix(std::hash<int>{}(key.device));
        mix(std::hash<int64_t>{}(key.cols));
        mix(std::hash<int64_t>{}(key.rows));
        mix(std::hash<int64_t>{}(key.channels));
        mix(std::hash<int64_t>{}(key.samples));
        mix(std::hash<size_t>{}(key.nb0));
        mix(std::hash<size_t>{}(key.nb1));
        mix(std::hash<size_t>{}(key.nb2));
        mix(std::hash<size_t>{}(key.nb3));
        mix(std::hash<int>{}((int) key.source_type));
        mix(std::hash<int>{}((int) key.layout));
        mix(std::hash<int>{}((int) key.correction));
        mix(std::hash<uint32_t>{}(key.flags));
        return h;
    }
};

struct ggml_cuda_dp16_packed16_weight_cache_entry {
    int32_t * payload = nullptr;
    half    * scales  = nullptr;
    dp16_packed16_weight_view view = {};
    cudaEvent_t ready = nullptr;
    bool ready_recorded = false;
    size_t packed_bytes_payload = 0;
    size_t packed_bytes_scales = 0;
};

static std::mutex & ggml_cuda_dp16_packed16_weight_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static inline void ggml_cuda_dp16_packed16_weight_cache_entry_free(
        const ggml_cuda_dp16_packed16_weight_cache_key & key,
        ggml_cuda_dp16_packed16_weight_cache_entry & entry) {
    if (entry.payload || entry.scales || entry.ready) {
        ggml_cuda_set_device(key.device);
    }
    if (entry.payload) {
        (void) cudaFree(entry.payload);
        entry.payload = nullptr;
    }
    if (entry.scales) {
        (void) cudaFree(entry.scales);
        entry.scales = nullptr;
    }
    if (entry.ready) {
        (void) cudaEventDestroy(entry.ready);
        entry.ready = nullptr;
    }
    entry.ready_recorded = false;
    entry.packed_bytes_payload = 0;
    entry.packed_bytes_scales = 0;
    entry.view = {};
}

struct ggml_cuda_dp16_packed16_weight_cache_state {
    std::unordered_map<
        ggml_cuda_dp16_packed16_weight_cache_key,
        ggml_cuda_dp16_packed16_weight_cache_entry,
        ggml_cuda_dp16_packed16_weight_cache_key_hash> entries;

    ~ggml_cuda_dp16_packed16_weight_cache_state() {
        for (auto & kv : entries) {
            ggml_cuda_dp16_packed16_weight_cache_entry_free(kv.first, kv.second);
        }
    }
};

static ggml_cuda_dp16_packed16_weight_cache_state & ggml_cuda_dp16_packed16_weight_cache_state_get() {
    static ggml_cuda_dp16_packed16_weight_cache_state state;
    return state;
}

static std::unordered_map<
        ggml_cuda_dp16_packed16_weight_cache_key,
        ggml_cuda_dp16_packed16_weight_cache_entry,
        ggml_cuda_dp16_packed16_weight_cache_key_hash> & ggml_cuda_dp16_packed16_weight_cache() {
    return ggml_cuda_dp16_packed16_weight_cache_state_get().entries;
}

void ggml_cuda_dp16_packed16_weight_cache_invalidate_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer) {
        return;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_dp16_packed16_weight_cache_mutex());
    auto & cache = ggml_cuda_dp16_packed16_weight_cache();
    for (auto it = cache.begin(); it != cache.end(); ) {
        if (it->first.buffer == buffer) {
            ggml_cuda_dp16_packed16_weight_cache_entry_free(it->first, it->second);
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_cuda_dp16_packed16_weight_cache_invalidate_tensor(const ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }

    const ggml_tensor * storage_tensor = tensor->view_src ? tensor->view_src : tensor;
    const void * data = storage_tensor->data;
    ggml_backend_buffer_t buffer = storage_tensor->buffer;

    std::lock_guard<std::mutex> lock(ggml_cuda_dp16_packed16_weight_cache_mutex());
    auto & cache = ggml_cuda_dp16_packed16_weight_cache();
    for (auto it = cache.begin(); it != cache.end(); ) {
        const auto & k = it->first;
        const bool same_tensor = k.tensor == tensor || k.tensor == storage_tensor;
        const bool same_data = data != nullptr && k.data == data;
        const bool same_buffer = buffer != nullptr && k.buffer == buffer;
        if (same_tensor || same_data || same_buffer) {
            ggml_cuda_dp16_packed16_weight_cache_entry_free(it->first, it->second);
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_cuda_dp16_packed16_weight_cache_remove_buffer(ggml_backend_buffer_t buffer) {
    ggml_cuda_dp16_packed16_weight_cache_invalidate_buffer(buffer);
}

static inline bool ggml_cuda_dp16_make_packed16_weight_cache_key(
        const ggml_tensor * src0,
        ggml_cuda_dp16_packed16_weight_cache_key * out) {
    if (!src0 || !out || src0->data == nullptr || src0->view_src != nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[2] <= 0 || src0->ne[3] <= 0) {
        return false;
    }
    if (src0->ne[0] % QK8_0 != 0) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type)) {
        return false;
    }

    *out = {};
    out->tensor = src0;
    out->data = src0->data;
    out->buffer = src0->buffer;
    out->device = ggml_cuda_get_device();
    out->cols = src0->ne[0];
    out->rows = src0->ne[1];
    out->channels = src0->ne[2];
    out->samples = src0->ne[3];
    out->nb0 = src0->nb[0];
    out->nb1 = src0->nb[1];
    out->nb2 = src0->nb[2];
    out->nb3 = src0->nb[3];
    out->source_type = src0->type;
    out->layout = DP16_LAYOUT_PACKED16_I32_SCALED;
    out->correction = src0->type == GGML_TYPE_Q4_0 ? DP16_CORR_PREPACK_SIGNED_I8 : DP16_CORR_NONE;
    out->flags = 0;
    return true;
}

static __device__ __forceinline__ int8_t ggml_cuda_dp16_q4_0_signed_value(const block_q4_0 & b, const int e) {
    const uint8_t byte = b.qs[e & 15];
    const int q = e < 16 ? (byte & 0x0f) : (byte >> 4);
    return (int8_t) (q - 8);
}

static __device__ __forceinline__ int32_t ggml_cuda_dp16_pack4_i8(
        const int8_t v0, const int8_t v1, const int8_t v2, const int8_t v3) {
    uint32_t p = 0;
    p |= (uint32_t) (uint8_t) v0;
    p |= (uint32_t) (uint8_t) v1 << 8;
    p |= (uint32_t) (uint8_t) v2 << 16;
    p |= (uint32_t) (uint8_t) v3 << 24;
    return (int32_t) p;
}

static __global__ void ggml_cuda_dp16_pack_q8_0_to_packed16_i32_scaled_kernel(
        const block_q8_0 * __restrict__ src,
        int32_t * __restrict__ payload,
        half * __restrict__ scales,
        const uint32_t blocks_per_row,
        const uint32_t rows,
        const uint32_t channels,
        const uint32_t samples,
        const uint32_t src_stride_row_blocks,
        const uint32_t src_stride_channel_blocks,
        const uint32_t src_stride_sample_blocks,
        const uint32_t payload_stride_row_i32,
        const uint32_t payload_stride_channel_i32,
        const uint32_t payload_stride_sample_i32,
        const uint32_t scale_stride_row_half,
        const uint32_t scale_stride_channel_half,
        const uint32_t scale_stride_sample_half) {
    const uint64_t total_blocks = (uint64_t) samples*channels*rows*blocks_per_row;
    for (uint64_t idx = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < total_blocks; idx += (uint64_t) gridDim.x*blockDim.x) {
        uint64_t t = idx;
        const uint32_t qb = t % blocks_per_row; t /= blocks_per_row;
        const uint32_t row = t % rows;           t /= rows;
        const uint32_t channel = t % channels;   t /= channels;
        const uint32_t sample = t;

        const block_q8_0 & bx = src[
            sample*src_stride_sample_blocks +
            channel*src_stride_channel_blocks +
            row*src_stride_row_blocks + qb];

        const int32_t * qx = (const int32_t *) bx.qs;
        int32_t * dst_payload = payload +
            sample*payload_stride_sample_i32 +
            channel*payload_stride_channel_i32 +
            row*payload_stride_row_i32 + qb*(QK8_0/4);
#pragma unroll
        for (int i = 0; i < QK8_0/4; ++i) {
            dst_payload[i] = qx[i];
        }

        scales[
            sample*scale_stride_sample_half +
            channel*scale_stride_channel_half +
            row*scale_stride_row_half + qb] = bx.d;
    }
}

static __global__ void ggml_cuda_dp16_pack_q4_0_to_packed16_i32_scaled_kernel(
        const block_q4_0 * __restrict__ src,
        int32_t * __restrict__ payload,
        half * __restrict__ scales,
        const uint32_t blocks_per_row,
        const uint32_t rows,
        const uint32_t channels,
        const uint32_t samples,
        const uint32_t src_stride_row_blocks,
        const uint32_t src_stride_channel_blocks,
        const uint32_t src_stride_sample_blocks,
        const uint32_t payload_stride_row_i32,
        const uint32_t payload_stride_channel_i32,
        const uint32_t payload_stride_sample_i32,
        const uint32_t scale_stride_row_half,
        const uint32_t scale_stride_channel_half,
        const uint32_t scale_stride_sample_half) {
    const uint64_t total_blocks = (uint64_t) samples*channels*rows*blocks_per_row;
    for (uint64_t idx = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < total_blocks; idx += (uint64_t) gridDim.x*blockDim.x) {
        uint64_t t = idx;
        const uint32_t qb = t % blocks_per_row; t /= blocks_per_row;
        const uint32_t row = t % rows;           t /= rows;
        const uint32_t channel = t % channels;   t /= channels;
        const uint32_t sample = t;

        const block_q4_0 & bx = src[
            sample*src_stride_sample_blocks +
            channel*src_stride_channel_blocks +
            row*src_stride_row_blocks + qb];

        int32_t * dst_payload = payload +
            sample*payload_stride_sample_i32 +
            channel*payload_stride_channel_i32 +
            row*payload_stride_row_i32 + qb*(QK8_0/4);
#pragma unroll
        for (int qi4 = 0; qi4 < QK8_0/4; ++qi4) {
            const int e = qi4*4;
            dst_payload[qi4] = ggml_cuda_dp16_pack4_i8(
                    ggml_cuda_dp16_q4_0_signed_value(bx, e + 0),
                    ggml_cuda_dp16_q4_0_signed_value(bx, e + 1),
                    ggml_cuda_dp16_q4_0_signed_value(bx, e + 2),
                    ggml_cuda_dp16_q4_0_signed_value(bx, e + 3));
        }

        scales[
            sample*scale_stride_sample_half +
            channel*scale_stride_channel_half +
            row*scale_stride_row_half + qb] = bx.d;
    }
}

static inline bool ggml_cuda_dp16_get_packed16_weight(
        const ggml_tensor * src0,
        dp16_packed16_weight_view * out) {
    if (out) {
        *out = {};
    }
    if (!src0 || !out) {
        return false;
    }

    ggml_cuda_dp16_packed16_weight_cache_key key = {};
    if (!ggml_cuda_dp16_make_packed16_weight_cache_key(src0, &key)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_dp16_packed16_weight_cache_mutex());
    auto & cache = ggml_cuda_dp16_packed16_weight_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        return false;
    }
    *out = it->second.view;
    return true;
}

static inline bool ggml_cuda_dp16_ensure_packed16_weight(
        const ggml_tensor * src0,
        cudaStream_t stream) {
    ggml_cuda_dp16_packed16_weight_cache_key key = {};
    if (!ggml_cuda_dp16_make_packed16_weight_cache_key(src0, &key)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_dp16_packed16_weight_cache_mutex());
    auto & cache = ggml_cuda_dp16_packed16_weight_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second.ready_recorded && it->second.ready) {
#if defined(GGML_USE_HIP)
            hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
            CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
            if (capture_status == hipStreamCaptureStatusNone) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, it->second.ready, 0));
            }
#else
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
            if (capture_status == cudaStreamCaptureStatusNone) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, it->second.ready, 0));
            }
#endif
        }
        dp16_trace_emit_sidecar(src0->type == GGML_TYPE_Q4_0 ? DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4 : DP16_ROUTE_MMVQ_PACKED16_DOT4,
                src0->type, DP16_LAYOUT_PACKED16_I32_SCALED,
                src0->type == GGML_TYPE_Q4_0 ? DP16_CORR_PREPACK_SIGNED_I8 : DP16_CORR_NONE, "hit", false,
                it->second.packed_bytes_payload, it->second.packed_bytes_scales);
        return true;
    }

    const int64_t blocks_per_row = src0->ne[0] / QK8_0;
    const int64_t rows = src0->ne[1];
    const int64_t channels = src0->ne[2];
    const int64_t samples = src0->ne[3];

    const int64_t payload_stride_row_i32 = src0->ne[0] / 4;
    const int64_t scale_stride_row_half = blocks_per_row;
    const int64_t payload_stride_channel_i32 = rows*payload_stride_row_i32;
    const int64_t scale_stride_channel_half = rows*scale_stride_row_half;
    const int64_t payload_stride_sample_i32 = channels*payload_stride_channel_i32;
    const int64_t scale_stride_sample_half = channels*scale_stride_channel_half;

    const size_t payload_count = (size_t) samples*payload_stride_sample_i32;
    const size_t scale_count = (size_t) samples*scale_stride_sample_half;

    ggml_cuda_dp16_packed16_weight_cache_entry * entry = nullptr;
    if (it == cache.end()) {
        ggml_cuda_dp16_packed16_weight_cache_entry new_entry = {};
        CUDA_CHECK(cudaMalloc((void **) &new_entry.payload, payload_count*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &new_entry.scales, scale_count*sizeof(half)));
        CUDA_CHECK(cudaEventCreateWithFlags(&new_entry.ready, cudaEventDisableTiming));
        new_entry.packed_bytes_payload = payload_count*sizeof(int32_t);
        new_entry.packed_bytes_scales = scale_count*sizeof(half);

        new_entry.view.payload = new_entry.payload;
        new_entry.view.scales = new_entry.scales;
        new_entry.view.rows = rows;
        new_entry.view.cols = src0->ne[0];
        new_entry.view.payload_stride_row_i32 = payload_stride_row_i32;
        new_entry.view.scale_stride_row_half = scale_stride_row_half;
        new_entry.view.payload_stride_channel_i32 = payload_stride_channel_i32;
        new_entry.view.scale_stride_channel_half = scale_stride_channel_half;
        new_entry.view.payload_stride_sample_i32 = payload_stride_sample_i32;
        new_entry.view.scale_stride_sample_half = scale_stride_sample_half;
        new_entry.view.source_type = src0->type;
        new_entry.view.flags = 0;

        auto inserted = cache.emplace(key, new_entry);
        entry = &inserted.first->second;
    } else {
        entry = &it->second;
    }

    const uint32_t src_stride_row_blocks = (uint32_t) (src0->nb[1] / ggml_type_size(src0->type));
    const uint32_t src_stride_channel_blocks = (uint32_t) (src0->nb[2] / ggml_type_size(src0->type));
    const uint32_t src_stride_sample_blocks = (uint32_t) (src0->nb[3] / ggml_type_size(src0->type));

    const uint64_t total_blocks = (uint64_t) samples*channels*rows*blocks_per_row;
    const int threads = 256;
    const int blocks = (int) std::min<uint64_t>((total_blocks + threads - 1) / threads, 65535);
    if (src0->type == GGML_TYPE_Q4_0) {
        ggml_cuda_dp16_pack_q4_0_to_packed16_i32_scaled_kernel<<<blocks, threads, 0, stream>>>(
                (const block_q4_0 *) src0->data,
                entry->payload,
                entry->scales,
                (uint32_t) blocks_per_row,
                (uint32_t) rows,
                (uint32_t) channels,
                (uint32_t) samples,
                src_stride_row_blocks,
                src_stride_channel_blocks,
                src_stride_sample_blocks,
                (uint32_t) payload_stride_row_i32,
                (uint32_t) payload_stride_channel_i32,
                (uint32_t) payload_stride_sample_i32,
                (uint32_t) scale_stride_row_half,
                (uint32_t) scale_stride_channel_half,
                (uint32_t) scale_stride_sample_half);
    } else {
        ggml_cuda_dp16_pack_q8_0_to_packed16_i32_scaled_kernel<<<blocks, threads, 0, stream>>>(
                (const block_q8_0 *) src0->data,
                entry->payload,
                entry->scales,
                (uint32_t) blocks_per_row,
                (uint32_t) rows,
                (uint32_t) channels,
                (uint32_t) samples,
                src_stride_row_blocks,
                src_stride_channel_blocks,
                src_stride_sample_blocks,
                (uint32_t) payload_stride_row_i32,
                (uint32_t) payload_stride_channel_i32,
                (uint32_t) payload_stride_sample_i32,
                (uint32_t) scale_stride_row_half,
                (uint32_t) scale_stride_channel_half,
                (uint32_t) scale_stride_sample_half);
    }
    CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    entry->ready_recorded = true;
    dp16_trace_emit_sidecar(src0->type == GGML_TYPE_Q4_0 ? DP16_ROUTE_MMVQ_Q4_0_PACKED16_DOT4 : DP16_ROUTE_MMVQ_PACKED16_DOT4,
            src0->type, DP16_LAYOUT_PACKED16_I32_SCALED,
            src0->type == GGML_TYPE_Q4_0 ? DP16_CORR_PREPACK_SIGNED_I8 : DP16_CORR_NONE, "miss", true,
            entry->packed_bytes_payload, entry->packed_bytes_scales);

    return true;
}

static inline dp16_problem ggml_cuda_dp16_mmvq_problem_init(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        const int64_t ncols_x,
        const int64_t nrows_x,
        const int64_t ncols_dst,
        const int cc,
        const bool has_fusion,
        const bool has_ids) {
    dp16_problem problem = dp16_problem_init(DP16_OP_DECODE_PROJ_GEMV);
    problem.m = nrows_x;
    problem.n = ncols_dst;
    problem.k = ncols_x;
    problem.batch = dst ? (int) dst->ne[3] : 1;
    problem.heads_q = dst ? (int) dst->ne[2] : 1;
    problem.heads_kv = src0 ? (int) src0->ne[2] : 1;
    problem.head_dim = (int) ncols_x;
    problem.src0_type = src0 ? src0->type : GGML_TYPE_COUNT;
    // MMVQ quantizes the activation side to q8_1 before the matvec kernel.
    problem.src1_type = GGML_TYPE_Q8_1;
    problem.src2_type = GGML_TYPE_COUNT;
    problem.dst_type = dst ? dst->type : GGML_TYPE_COUNT;
    problem.is_decode = ncols_dst <= 4;
    problem.has_fusion = has_fusion;
    problem.has_ids = has_ids;
    problem.cc = cc;

    problem.a = dp16_operand_desc_make(DP16_OPERAND_ACTIVATION, DP16_STORAGE_TRANSIENT_TILE,
            DP16_LAYOUT_Q8_BLOCK32, GGML_TYPE_Q8_1, ncols_dst, ncols_x,
            src1 ? src1->nb[1] : 0, src1 ? src1->nb[0] : 0);

    dp16_packed16_weight_view w16 = {};
    if (ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
        problem.has_packed16_b = true;
        problem.b_packed16 = w16;
        problem.b = dp16_operand_desc_make(DP16_OPERAND_WEIGHT, DP16_STORAGE_PERSISTENT_WEIGHT,
                DP16_LAYOUT_PACKED16_I32_SCALED, problem.src0_type, nrows_x, ncols_x,
                w16.payload_stride_row_i32, 1);
    } else {
        problem.has_packed16_b = false;
        problem.b = dp16_operand_desc_from_type(DP16_OPERAND_WEIGHT, DP16_STORAGE_PERSISTENT_WEIGHT,
                problem.src0_type, nrows_x, ncols_x,
                src0 ? src0->nb[1] : 0, src0 ? src0->nb[0] : 0);
    }

    problem.dst = dp16_operand_desc_from_type(DP16_OPERAND_OUTPUT, DP16_STORAGE_OUTPUT,
            problem.dst_type, nrows_x, ncols_dst,
            dst ? dst->nb[0] : 0, dst ? dst->nb[1] : 0);
    return problem;
}

static inline bool ggml_cuda_dp16_route_require_q8_mmvq() {
    return dp16_route_name_is_mmvq_q8_dot4(dp16_route_require_env());
}

static inline bool ggml_cuda_dp16_route_require_packed16_mmvq() {
    return dp16_route_name_is_mmvq_packed16_dot4(dp16_route_require_env());
}

static inline bool ggml_cuda_dp16_route_require_q4_0_packed16_mmvq() {
    return dp16_route_name_is_mmvq_q4_0_packed16_dot4(dp16_route_require_env());
}

static inline bool ggml_cuda_dp16_route_require_any_mmvq() {
    return ggml_cuda_dp16_route_require_q8_mmvq() ||
        ggml_cuda_dp16_route_require_packed16_mmvq() ||
        ggml_cuda_dp16_route_require_q4_0_packed16_mmvq();
}

static inline bool ggml_cuda_dp16_mmvq_q8_dot4_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_Q8_DOT4_MMVQ");
    return ggml_cuda_dp16_route_require_q8_mmvq() || (enabled && atoi(enabled) != 0);
}

static inline bool ggml_cuda_dp16_mmvq_packed16_dot4_env_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMVQ");
    return enabled && atoi(enabled) != 0;
}

static inline bool ggml_cuda_dp16_mmvq_q4_0_packed16_dot4_env_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_Q4_0_PACKED16_DOT4_MMVQ");
    return enabled && atoi(enabled) != 0;
}

static inline bool ggml_cuda_dp16_mmvq_packed16_dot4_enabled() {
    return ggml_cuda_dp16_route_require_packed16_mmvq() ||
        ggml_cuda_dp16_route_require_q4_0_packed16_mmvq() ||
        ggml_cuda_dp16_mmvq_packed16_dot4_env_enabled() ||
        ggml_cuda_dp16_mmvq_q4_0_packed16_dot4_env_enabled();
}

static inline bool ggml_cuda_dp16_mmvq_q8_dot4_supported(
        const ggml_type type,
        const int cc,
        const int warp_size,
        const int ncols_dst,
        const bool has_fusion,
        const bool has_ids,
        const int ncols_x,
        const int nrows_x) {
#if defined(GGML_USE_HIP)
    return ggml_cuda_dp16_mmvq_q8_dot4_enabled() &&
        GGML_CUDA_CC_IS_RDNA3(cc) &&
        warp_size == DP16_MMVQ_Q8_DOT4_WARP_SIZE &&
        type == GGML_TYPE_Q8_0 &&
        ncols_dst >= 1 && ncols_dst <= 4 &&
        (!has_fusion || ncols_dst == 1) && !has_ids &&
        ncols_x % 256 == 0 &&
        nrows_x > 0;
#else
    GGML_UNUSED_VARS(type, cc, warp_size, ncols_dst, has_fusion, has_ids, ncols_x, nrows_x);
    return false;
#endif
}

static inline bool ggml_cuda_dp16_mmvq_packed16_dot4_supported(
        const int cc,
        const int warp_size,
        const int ncols_dst,
        const bool has_fusion,
        const bool has_ids,
        const int ncols_x,
        const int nrows_x) {
#if defined(GGML_USE_HIP)
    return ggml_cuda_dp16_mmvq_packed16_dot4_enabled() &&
        GGML_CUDA_CC_IS_RDNA3(cc) &&
        warp_size == DP16_MMVQ_PACKED16_DOT4_WARP_SIZE &&
        ncols_dst >= 1 && ncols_dst <= 4 &&
        !has_fusion && !has_ids &&
        ncols_x % 256 == 0 &&
        nrows_x > 0;
#else
    GGML_UNUSED_VARS(cc, warp_size, ncols_dst, has_fusion, has_ids, ncols_x, nrows_x);
    return false;
#endif
}

static inline void ggml_cuda_dp16_mmvq_route_require_fail(
        const dp16_problem & problem,
        const dp16_plan & plan,
        const dp16_reject_reason reject) {
    GGML_ABORT(
        "DP16 route require failed: required=%s op=%s route=%s kernel=%s reject=%s "
        "fallback_disallowed=1 m=%lld n=%lld k=%lld A.role=%s A.storage=%s A.layout=%s "
        "B.role=%s B.storage=%s B.layout=%s dst.layout=%s",
        dp16_route_require_env(),
        dp16_op_name(problem.op),
        dp16_cstr_or_none(plan.route_name),
        dp16_cstr_or_none(plan.kernel_name),
        dp16_reject_name(reject),
        (long long) problem.m,
        (long long) problem.n,
        (long long) problem.k,
        dp16_operand_role_name(plan.a.role != DP16_OPERAND_UNKNOWN ? plan.a.role : problem.a.role),
        dp16_storage_name(plan.a.role != DP16_OPERAND_UNKNOWN ? plan.a.storage : problem.a.storage),
        dp16_layout_name(plan.a.role != DP16_OPERAND_UNKNOWN ? plan.a.layout : problem.a.layout),
        dp16_operand_role_name(plan.b.role != DP16_OPERAND_UNKNOWN ? plan.b.role : problem.b.role),
        dp16_storage_name(plan.b.role != DP16_OPERAND_UNKNOWN ? plan.b.storage : problem.b.storage),
        dp16_layout_name(plan.b.role != DP16_OPERAND_UNKNOWN ? plan.b.layout : problem.b.layout),
        dp16_layout_name(plan.dst.role != DP16_OPERAND_UNKNOWN ? plan.dst.layout : problem.dst.layout));
}

static inline void ggml_cuda_dp16_trace_mmvq_decode_plan(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        const int64_t ncols_x,
        const int64_t nrows_x,
        const int64_t ncols_dst,
        const int cc,
        const int warp_size,
        const bool has_fusion,
        const bool has_ids) {
    const bool trace = dp16_trace_enabled();
    const bool require_mmvq = ggml_cuda_dp16_route_require_any_mmvq();
    if (!trace && !require_mmvq) {
        return;
    }

    const dp16_problem problem = ggml_cuda_dp16_mmvq_problem_init(
            src0, src1, dst, ncols_x, nrows_x, ncols_dst, cc, has_fusion, has_ids);
    dp16_plan plan = dp16_plan_decode_proj_mmvq(problem, dp16_route_require_env());
    if (plan.backend == DP16_BACKEND_MMVQ_PACKED16_I32_DOT4) {
        plan.kernel_name = dp16_mmvq_packed16_kernel_name();
    }
    dp16_trace_emit_plan(problem, plan);

    if (!dp16_plan_accepted(plan)) {
        if (require_mmvq) {
            ggml_cuda_dp16_mmvq_route_require_fail(problem, plan, plan.reject);
        }
        return;
    }

    bool runtime_supported = false;
    if (plan.backend == DP16_BACKEND_MMVQ_Q8_DOT4) {
        runtime_supported = ggml_cuda_dp16_mmvq_q8_dot4_supported(
                problem.src0_type, cc, warp_size, (int) ncols_dst, has_fusion, has_ids, (int) ncols_x, (int) nrows_x);
    } else if (plan.backend == DP16_BACKEND_MMVQ_PACKED16_I32_DOT4 ||
            plan.backend == DP16_BACKEND_MMVQ_Q4_0_PACKED16_I32_DOT4) {
        runtime_supported = ggml_cuda_dp16_mmvq_packed16_dot4_supported(
                cc, warp_size, (int) ncols_dst, has_fusion, has_ids, (int) ncols_x, (int) nrows_x);
    }
    if (runtime_supported) {
        return;
    }

    // Keep existing MMVQ dispatch when the DP16 route is optional, but reject
    // silent fallback when explicitly required.
    dp16_plan fallback = plan;
    fallback.reject = DP16_REJECT_RUNTIME_DISABLED;
    fallback.kernel_name = "existing_mmvq_fallback";
    dp16_trace_emit_plan(problem, fallback);

    if (require_mmvq) {
        ggml_cuda_dp16_mmvq_route_require_fail(problem, fallback, DP16_REJECT_RUNTIME_DISABLED);
    }
}

static inline bool ggml_cuda_dp16_try_launch_packed16_mmvq(
        const ggml_tensor * src0,
        const void * src1_q8_1,
        float * dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_dst,
        const uint3 channel_ratio,
        const uint3 sample_ratio,
        const int stride_col_y,
        const int stride_col_dst,
        const int nchannels_dst,
        const int stride_channel_y,
        const int stride_channel_dst,
        const int nsamples_dst,
        const int stride_sample_y,
        const int stride_sample_dst,
        const int cc,
        const int warp_size,
        const bool has_fusion,
        const bool has_ids,
        cudaStream_t stream) {
    dp16_packed16_weight_view w16 = {};
    if (!ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
        return false;
    }
    if (!ggml_cuda_dp16_mmvq_packed16_dot4_supported(
            cc, warp_size, ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
        return false;
    }
    if (w16.cols != ncols_x || w16.rows < nrows_x) {
        return false;
    }

    const bool use_i32_lane = dp16_mmvq_packed16_use_i32_lane_kernel();
    switch (ncols_dst) {
        case 1:
            if (use_i32_lane) {
                dp16_mmvq_packed16_i32_n1_4_k256_launch<1>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                dp16_mmvq_packed16_i32_b32_n1_4_k256_launch<1>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            }
            return true;
        case 2:
            if (use_i32_lane) {
                dp16_mmvq_packed16_i32_n1_4_k256_launch<2>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                dp16_mmvq_packed16_i32_b32_n1_4_k256_launch<2>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            }
            return true;
        case 3:
            if (use_i32_lane) {
                dp16_mmvq_packed16_i32_n1_4_k256_launch<3>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                dp16_mmvq_packed16_i32_b32_n1_4_k256_launch<3>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            }
            return true;
        case 4:
            if (use_i32_lane) {
                dp16_mmvq_packed16_i32_n1_4_k256_launch<4>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                dp16_mmvq_packed16_i32_b32_n1_4_k256_launch<4>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            }
            return true;
        default:
            return false;
    }
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, c_ncols_dst, true, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
                (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, c_ncols_dst, false, small_k><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);

    mul_mat_vec_q_moe<type, rows_per_block><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int32_t * y_q8sum4, const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            if (ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
                if (has_fusion) {
                    dp16_mmvq_q8_dot4_fusion_n1_k256_launch<c_ncols_dst>(
                        vx, vy, fusion, dst, ncols_x, nrows_x, channel_ratio_fd, sample_ratio_fd,
                        stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                } else {
                    dp16_mmvq_q8_dot4_n1_4_k256_launch<c_ncols_dst>(
                        vx, vy, dst, ncols_x, nrows_x, channel_ratio_fd, sample_ratio_fd,
                        stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                }
                return;
            }

            if (ggml_cuda_rdna3_mmvq_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x, y_q8sum4)) {
                ggml_cuda_rdna3_mmvq_dot4_launch<type>(
                    vx, vy, y_q8sum4, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                    channel_ratio_fd, sample_ratio_fd, stride_row_x, stride_col_y, stride_col_dst,
                    nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
                return;
            }

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            if (ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
                dp16_mmvq_q8_dot4_n1_4_k256_launch<c_ncols_dst>(
                    vx, vy, dst, ncols_x, nrows_x, channel_ratio_fd, sample_ratio_fd,
                    stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                return;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            if (ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
                dp16_mmvq_q8_dot4_n1_4_k256_launch<c_ncols_dst>(
                    vx, vy, dst, ncols_x, nrows_x, channel_ratio_fd, sample_ratio_fd,
                    stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                return;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            if (ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
                dp16_mmvq_q8_dot4_n1_4_k256_launch<c_ncols_dst>(
                    vx, vy, dst, ncols_x, nrows_x, channel_ratio_fd, sample_ratio_fd,
                    stride_row_x, stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                return;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int32_t * y_q8sum4, const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, y_q8sum4, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    {
        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        const bool has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
        const bool has_ids = ids_d != nullptr;
        const bool should_prepare_packed16 = ggml_cuda_dp16_mmvq_packed16_dot4_env_enabled() ||
            ggml_cuda_dp16_mmvq_q4_0_packed16_dot4_env_enabled() ||
            ggml_cuda_dp16_route_require_q4_0_packed16_mmvq();
        if (should_prepare_packed16 &&
                !has_fusion && !has_ids &&
                ncols_dst >= 1 && ncols_dst <= 4 &&
                ne00 % 256 == 0) {
            ggml_cuda_dp16_ensure_packed16_weight(src0, stream);
        }
        ggml_cuda_dp16_trace_mmvq_decode_plan(src0, src1, dst, ne00, ne01, ncols_dst, cc, warp_size, has_fusion, has_ids);

        if (!has_ids) {
            const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / ne02);
            const uint3 sample_ratio_fd  = init_fastdiv_values(ne3 / ne03);
            if (ggml_cuda_dp16_try_launch_packed16_mmvq(
                    src0, src1_q8_1.get(), dst_d, ne00, ne01, ncols_dst,
                    channel_ratio_fd, sample_ratio_fd,
                    stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_y, stride_channel_dst,
                    ne3, s13, s3,
                    cc, warp_size, has_fusion, has_ids, stream)) {
                return;
            }
        }
    }

    ggml_cuda_pool_alloc<int32_t> src1_q8sum4(ctx.pool());
    const int32_t * src1_q8sum4_d = nullptr;
    {
        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        if (ggml_cuda_rdna3_mmvq_dot4_wants_q8sum4(src0->type, cc, warp_size, ncols_dst, ids_d != nullptr, ne00, ne01)) {
            const int64_t n_q8_blocks_total = ne13*ne12 * ne11*(ne10_padded / QK8_1);
            src1_q8sum4_d = src1_q8sum4.alloc(4*n_q8_blocks_total);
            ggml_cuda_rdna3_q8_1_sum4_precompute(src1_q8_1.get(), src1_q8sum4.get(), n_q8_blocks_total, stream);
        }
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               src1_q8sum4_d, ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    const int cc = ggml_cuda_info().devices[id].cc;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    ggml_cuda_dp16_trace_mmvq_decode_plan(src0, src1, dst, ne00, row_diff, src1_ncols, cc, warp_size, false, false);

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nullptr, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
