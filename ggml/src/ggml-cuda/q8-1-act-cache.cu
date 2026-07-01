#include "q8-1-act-cache.cuh"

#include <cstdlib>
#include <cstring>

namespace {

struct ggml_cuda_q8_1_act_cache_key {
    const ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int device = -1;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[4] = { 0, 0, 0, 0 };
    size_t nb[4] = { 0, 0, 0, 0 };
    int64_t ne0_padded = 0;

    bool operator==(const ggml_cuda_q8_1_act_cache_key & other) const {
        return tensor == other.tensor && data == other.data && buffer == other.buffer && device == other.device &&
            type == other.type && ne0_padded == other.ne0_padded &&
            std::memcmp(ne, other.ne, sizeof(ne)) == 0 && std::memcmp(nb, other.nb, sizeof(nb)) == 0;
    }
};

static void ggml_cuda_q8_1_act_cache_set_device_for_free(int device, int * old_device, bool * switched) {
    *old_device = -1;
    *switched = false;
    if (device < 0) {
        return;
    }

    CUDA_CHECK(cudaGetDevice(old_device));
    if (*old_device != device) {
        ggml_cuda_set_device(device);
        *switched = true;
    }
}

static void ggml_cuda_q8_1_act_cache_restore_device(int old_device, bool switched) {
    if (switched && old_device >= 0) {
        ggml_cuda_set_device(old_device);
    }
}

static void ggml_cuda_q8_1_act_cache_free_noexcept(block_q8_1 * data, int device) {
    if (data == nullptr) {
        return;
    }

    int old_device = -1;
    const cudaError_t get_err = cudaGetDevice(&old_device);
    const bool can_restore = get_err == cudaSuccess;
    if (device >= 0 && (!can_restore || old_device != device)) {
        (void) cudaSetDevice(device);
    }
    (void) cudaFree(data);
    if (device >= 0 && can_restore && old_device != device) {
        (void) cudaSetDevice(old_device);
    }
}

struct ggml_cuda_q8_1_act_cache_state {
    block_q8_1 * data = nullptr;
    size_t bytes = 0;
    int alloc_device = -1;
    bool has_key = false;
    bool valid = false;
    bool fill_pending = false;
    cudaStream_t writer_stream = nullptr;
    ggml_cuda_q8_1_act_cache_key key = {};

    void clear_metadata() {
        has_key = false;
        valid = false;
        fill_pending = false;
        writer_stream = nullptr;
        key = {};
    }

    void free_data() {
        if (data != nullptr) {
            int old_device = -1;
            bool switched = false;
            ggml_cuda_q8_1_act_cache_set_device_for_free(alloc_device, &old_device, &switched);
            CUDA_CHECK(cudaFree(data));
            ggml_cuda_q8_1_act_cache_restore_device(old_device, switched);
            data = nullptr;
            bytes = 0;
            alloc_device = -1;
        }
        clear_metadata();
    }

    ~ggml_cuda_q8_1_act_cache_state() {
        ggml_cuda_q8_1_act_cache_free_noexcept(data, alloc_device);
        data = nullptr;
    }
};

static thread_local ggml_cuda_q8_1_act_cache_state g_ggml_cuda_q8_1_act_cache;

static bool ggml_cuda_q8_1_act_cache_env_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0;
}

static bool ggml_cuda_q8_1_act_cache_log_enabled() {
    static const bool enabled = ggml_cuda_q8_1_act_cache_env_enabled("GGML_CUDA_Q8_1_ACT_CACHE_LOG");
    return enabled;
}

static bool ggml_cuda_q8_1_act_cache_enabled(const char * tensor_name) {
    static const bool enabled = ggml_cuda_q8_1_act_cache_env_enabled("GGML_CUDA_Q8_1_ACT_CACHE");
    if (!enabled) {
        return false;
    }
    if (const char * filter = std::getenv("GGML_CUDA_Q8_1_ACT_CACHE_FILTER")) {
        return filter[0] == '\0' || (tensor_name != nullptr && std::strstr(tensor_name, filter) != nullptr);
    }
    return true;
}

static bool ggml_cuda_q8_1_act_cache_stream_is_capturing(cudaStream_t stream, bool * query_ok) {
    if (query_ok != nullptr) {
        *query_ok = false;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t err = cudaStreamIsCapturing(stream, &capture_status);
    if (err != cudaSuccess) {
        return true;
    }
    if (query_ok != nullptr) {
        *query_ok = true;
    }
    return capture_status != cudaStreamCaptureStatusNone;
}

static bool ggml_cuda_q8_1_act_cache_make_key(
        const ggml_tensor * src1,
        int64_t ne0_padded,
        ggml_cuda_q8_1_act_cache_key * out) {
    if (src1 == nullptr || out == nullptr || src1->data == nullptr || ne0_padded <= 0) {
        return false;
    }

    *out = {};
    out->tensor = src1;
    out->data = src1->data;
    out->buffer = src1->buffer;
    out->device = ggml_cuda_get_device();
    out->type = src1->type;
    out->ne0_padded = ne0_padded;
    for (int i = 0; i < 4; ++i) {
        out->ne[i] = src1->ne[i];
        out->nb[i] = src1->nb[i];
    }
    return true;
}

} // namespace

void ggml_cuda_q8_1_act_cache_reset() {
    ggml_cuda_q8_1_act_cache_state & cache = g_ggml_cuda_q8_1_act_cache;
    cache.valid = false;
    cache.fill_pending = false;
    if (cache.data == nullptr) {
        cache.writer_stream = nullptr;
    }
}

bool ggml_cuda_q8_1_act_cache_try_get(
        const ggml_tensor * src1,
        int64_t ne0_padded,
        size_t bytes,
        cudaStream_t stream,
        const char * route,
        const char * tensor_name,
        block_q8_1 ** out,
        bool * needs_quantize) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (needs_quantize != nullptr) {
        *needs_quantize = false;
    }
    if (out == nullptr || needs_quantize == nullptr || bytes == 0 || !ggml_cuda_q8_1_act_cache_enabled(tensor_name)) {
        return false;
    }

    ggml_cuda_q8_1_act_cache_key key = {};
    if (!ggml_cuda_q8_1_act_cache_make_key(src1, ne0_padded, &key)) {
        if (ggml_cuda_q8_1_act_cache_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=reject reject=invalid_key bytes=%zu ne0_padded=%lld\n",
                    __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                    bytes, (long long) ne0_padded);
        }
        return false;
    }

    bool capture_query_ok = false;
    const bool stream_is_capturing = ggml_cuda_q8_1_act_cache_stream_is_capturing(stream, &capture_query_ok);
    if (stream_is_capturing) {
        if (ggml_cuda_q8_1_act_cache_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=reject reject=%s bytes=%zu\n",
                    __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                    capture_query_ok ? "capture_cache_pointer_unstable" : "capture_query_failed", bytes);
        }
        return false;
    }

    ggml_cuda_q8_1_act_cache_state & cache = g_ggml_cuda_q8_1_act_cache;
    if (cache.data != nullptr && cache.writer_stream != nullptr && cache.writer_stream != stream) {
        if (ggml_cuda_q8_1_act_cache_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=reject reject=cross_stream_unordered bytes=%zu owner_device=%d new_device=%d\n",
                    __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                    bytes, cache.alloc_device, key.device);
        }
        return false;
    }

    const bool device_match = cache.alloc_device < 0 || cache.alloc_device == key.device;
    if (!device_match) {
        if (ggml_cuda_q8_1_act_cache_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=drop reason=device_change old_device=%d new_device=%d bytes=%zu\n",
                    __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                    cache.alloc_device, key.device, cache.bytes);
        }
        cache.free_data();
    }

    const bool key_match = cache.has_key && cache.key == key;
    if (!key_match) {
        cache.valid = false;
        cache.fill_pending = false;
        cache.has_key = true;
        cache.key = key;
    }

    if (cache.bytes < bytes) {
        cache.free_data();
        int old_device = -1;
        bool switched = false;
        ggml_cuda_q8_1_act_cache_set_device_for_free(key.device, &old_device, &switched);
        CUDA_CHECK(cudaMalloc((void **) &cache.data, bytes));
        ggml_cuda_q8_1_act_cache_restore_device(old_device, switched);
        cache.bytes = bytes;
        cache.alloc_device = key.device;
        cache.valid = false;
        cache.fill_pending = false;
        cache.has_key = true;
        cache.key = key;
    }
    if (cache.data == nullptr) {
        return false;
    }

    *out = cache.data;
    *needs_quantize = !cache.valid;
    if (*needs_quantize) {
        cache.valid = false;
        cache.fill_pending = true;
        cache.writer_stream = stream;
    }
    if (ggml_cuda_q8_1_act_cache_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=%s bytes=%zu ne0_padded=%lld device=%d\n",
                __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                cache.valid ? "reuse" : "quantize", bytes, (long long) ne0_padded, cache.alloc_device);
    }
    return true;
}

void ggml_cuda_q8_1_act_cache_mark_valid(
        const char * route,
        const char * tensor_name) {
    ggml_cuda_q8_1_act_cache_state & cache = g_ggml_cuda_q8_1_act_cache;
    if (cache.data == nullptr || !cache.has_key || !cache.fill_pending || cache.writer_stream == nullptr) {
        cache.valid = false;
        cache.fill_pending = false;
        if (ggml_cuda_q8_1_act_cache_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=ignore reason=no_pending_fill\n",
                    __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-");
        }
        return;
    }

    cache.valid = true;
    cache.fill_pending = false;
    if (ggml_cuda_q8_1_act_cache_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=ready bytes=%zu device=%d\n",
                __func__, route ? route : "q8_1_act_cache", tensor_name ? tensor_name : "-",
                cache.bytes, cache.alloc_device);
    }
}
