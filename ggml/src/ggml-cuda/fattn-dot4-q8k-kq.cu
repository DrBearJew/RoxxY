#include "fattn-dot4-q8k-kq.cuh"
#include "dot4-packed16/mtp-v4-144-tail-page-desc.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Packed16 K cache tensor registry (shared with llama-kv-cache).
// The metadata is intentionally compact: it binds sidecar bytes to the physical
// layout selected by the producer without adding any hot-kernel arguments beyond
// the existing layout-specialized bool.
struct packed16_registry_entry {
    ggml_tensor * payload  = nullptr;
    ggml_tensor * scales   = nullptr;
    ggml_cuda_packed16_sidecar_meta meta = {};
    ggml_cuda_mtp_qblock_tail_page_map_v1 tail_page_map = {};
    ggml_cuda_mtp_qblock_full_page_map_v1 full_page_map = {};
    ggml_cuda_mtp_qblock_full_page_map_v1 full_page_map_base = {};
    std::vector<int32_t> full_page_table_host;
    int32_t * full_page_table_device = nullptr;
    size_t full_page_table_capacity = 0;
};

struct packed16_pending_full_page_map_entry {
    ggml_cuda_mtp_qblock_full_page_map_v1 map = {};
    std::vector<int32_t> host_block_table;
};

struct v4_k16d16_registry_entry {
    ggml_tensor * v_cache = nullptr;
    ggml_tensor * v_tail  = nullptr;
};

static std::mutex s_packed16_mutex;
static std::unordered_map<const void *, packed16_registry_entry> s_packed16_registry;
static std::unordered_map<const void *, packed16_pending_full_page_map_entry> s_packed16_pending_full_page_maps;
static ggml_cuda_mtp_qblock_tail_page_map_v1 s_packed16_tail_page_published_map = {};
static ggml_cuda_mtp_qblock_full_page_map_v1 s_packed16_full_page_published_map = {};
static int32_t * s_packed16_full_page_published_table_device = nullptr;
static size_t s_packed16_full_page_published_table_capacity = 0;
static ggml_tensor * s_packed16_full_page_published_payload = nullptr;
static ggml_tensor * s_packed16_full_page_published_scales = nullptr;
static ggml_cuda_packed16_sidecar_meta s_packed16_full_page_published_meta = {};
static ggml_cuda_mtp_qblock_tail_page_map_v1 s_packed16_tail_page_producer_snapshot_map = {};
static ggml_cuda_mtp_qblock_tail_page_map_v1 s_packed16_tail_page_dispatch_bind_map = {};
static ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 s_packed16_tail_page_last_dispatch_bind = {};
static uint64_t s_packed16_tail_page_dispatch_bind_count = 0;
struct packed16_tail_page_pending_dispatch_bind {
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint64_t req_begin = 0;
    uint64_t req_end = 0;
    uint64_t pending_id = 0;
    int32_t slot = -1;
};
static std::vector<packed16_tail_page_pending_dispatch_bind> s_packed16_tail_page_pending_dispatch_binds;
static uint64_t s_packed16_tail_page_pending_dispatch_bind_count = 0;
static uint32_t s_packed16_tail_page_consumer_no_map_nk_max = 0;
struct packed16_tail_page_retired_highwater_state {
    bool valid = false;
    uint32_t nk = 0;
    unsigned long long generation = 0;
    uint64_t reset_count = 0;
};
static packed16_tail_page_retired_highwater_state s_packed16_tail_page_retired_highwater = {};

struct packed16_tail_page_owned_write_ready_state {
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint32_t physical_page = 0;
    bool k_ready = false;
    bool v_ready = false;
    uint64_t k_ready_count = 0;
    uint64_t v_ready_count = 0;
    uint64_t pair_ready_count = 0;
    bool k_exclusive = false;
    bool v_exclusive = false;
    uint64_t k_exclusive_count = 0;
    uint64_t v_exclusive_count = 0;
};
static packed16_tail_page_owned_write_ready_state s_packed16_tail_page_owned_write_ready = {};

struct packed16_tail_page_owned_deferred_proof_attempt {
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint32_t logical_base = 0;
    uint32_t n_tokens = 0;
    uint32_t physical_page = 0;
    bool is_k = false;
    bool is_v = false;
    uint64_t attempt_id = 0;
    uint64_t record_event_seq = 0;
    uint64_t record_highwater_update_seq = 0;
    uint64_t record_resolve_count = 0;
    uint32_t record_live_highwater = 0;
    bool record_route_complete_same_map = false;
    bool record_k_ready = false;
    bool record_v_ready = false;
    uint64_t record_plan_id = 0;
    uint64_t record_plan_event_seq = 0;
    uint32_t record_plan_expected_layers = 0;
    uint32_t record_plan_expected_pages = 0;
    uint64_t record_plan_bound_layer_pages = 0;
    uint32_t record_plan_canonical_page_base = 0;
    uint32_t record_plan_page_end = 0;
    uint32_t record_plan_owned_page_base = 0;
    uint32_t record_plan_overlay_page_base = 0;
    bool record_plan_ready = false;
    bool record_plan_before_record = false;
    bool record_plan_route_complete = false;
    bool record_plan_write_through = false;
    bool record_plan_owned_write = false;
    bool record_plan_exclusive_skip = false;
    uint64_t record_live_window_highwater_event_seq = 0;
    uint64_t record_live_window_highwater_update_seq = 0;
    uint64_t record_live_window_route_complete_event_seq = 0;
    uint32_t record_live_window_highwater = 0;
    bool record_live_window_after_highwater = false;
    bool record_live_window_after_route_complete = false;
    bool record_live_window_route_complete_same_map = false;
    bool record_live_window_post_highwater = false;
    bool record_live_window_cross_highwater = false;
};

struct packed16_tail_page_owned_deferred_proof_state {
    std::vector<packed16_tail_page_owned_deferred_proof_attempt> attempts;
    uint64_t next_attempt_id = 0;
    uint64_t dropped_attempts = 0;
    uint64_t resolve_count = 0;
};
static packed16_tail_page_owned_deferred_proof_state s_packed16_tail_page_owned_deferred_proof = {};
static uint64_t s_packed16_tail_page_owned_deferred_proof_event_seq = 0;
static uint64_t s_packed16_tail_page_owned_deferred_proof_highwater_update_seq = 0;

struct packed16_tail_page_owned_write_plan_state {
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint64_t plan_id = 0;
    uint64_t plan_event_seq = 0;
    uint32_t expected_layer_count = 0;
    uint32_t expected_page_count = 0;
    uint64_t expected_layer_page_count = 0;
    uint64_t bind_count_at_plan = 0;
    uint64_t bound_layer_pages_at_plan = 0;
};
static packed16_tail_page_owned_write_plan_state s_packed16_tail_page_owned_write_plan = {};
static uint64_t s_packed16_tail_page_owned_write_next_plan_id = 0;

struct packed16_tail_page_owned_live_window_state {
    uint64_t highwater_event_seq = 0;
    uint64_t highwater_update_seq = 0;
    uint32_t highwater_nk = 0;
    unsigned long long highwater_generation = 0;
    uint64_t route_complete_event_seq = 0;
    uint64_t route_complete_bind_count = 0;
    uint32_t route_complete_highwater = 0;
    unsigned long long route_complete_generation = 0;
    ggml_cuda_mtp_qblock_tail_page_map_v1 route_complete_map = {};
    uint64_t reset_count = 0;
};
static packed16_tail_page_owned_live_window_state s_packed16_tail_page_owned_live_window = {};

struct packed16_tail_page_route_complete_state {
    ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
    uint32_t expected_layer_count = 0;
    uint32_t expected_page_count = 0;
    uint64_t expected_layer_page_count = 0;
    std::unordered_set<int32_t> bound_layers;
    std::unordered_set<uint64_t> bound_layer_pages;
    uint32_t bound_page_mask = 0;
    uint32_t min_bound_nk = 0;
    uint32_t max_bound_nk = 0;
    uint32_t min_physical_page = 0;
    uint32_t max_physical_page = 0;
    uint64_t slot_translate_count = 0;
    uint64_t slot_translate_reject_count = 0;
    uint64_t bind_count = 0;
    uint64_t complete_count = 0;
};
static packed16_tail_page_route_complete_state s_packed16_tail_page_route_complete = {};
static unsigned long long s_packed16_generation = 0;
static std::mutex s_v4_k16d16_mutex;
static std::unordered_map<const void *, v4_k16d16_registry_entry> s_v4_k16d16_registry;

static bool ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled() {
    static const bool enabled = []() {
        const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY_TRACE");
        return v && atoi(v) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_ready_trace_enabled() {
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        return true;
    }
    static const bool enabled = []() {
        const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_READY_TRACE");
        return v && atoi(v) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_exclusive_requested() {
    static const bool requested = []() {
        const char * paged = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * excl  = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_EXCLUSIVE");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && excl && atoi(excl) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_env_requested() {
    static const bool requested = []() {
        const char * all_bound = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_EXCLUSIVE_ALL_BOUND");
        return all_bound && atoi(all_bound) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_unsafe_requested() {
    static const bool requested = []() {
        const char * unsafe = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_EXCLUSIVE_ALL_BOUND_UNSAFE");
        return unsafe && atoi(unsafe) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_requested() {
    static const bool requested = []() {
        return ggml_cuda_mtp_qblock_owned_tail_write_exclusive_requested() &&
            ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_env_requested() &&
            ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_unsafe_requested();
    }();
    return requested;
}


static bool ggml_cuda_mtp_qblock_owned_tail_write_exclusive_trace_enabled() {
    if (ggml_cuda_mtp_qblock_owned_tail_write_ready_trace_enabled()) {
        return true;
    }
    static const bool enabled = []() {
        const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_EXCLUSIVE_TRACE");
        return v && atoi(v) != 0;
    }();
    return enabled;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested() {
    static const bool requested = []() {
        const char * paged = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * proof = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_DEFERRED_PROOF");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && proof && atoi(proof) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested() {
    static const bool requested = []() {
        const char * paged = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * proof = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_PLAN_PROOF");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && proof && atoi(proof) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested() {
    static const bool requested = []() {
        const char * paged = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * proof = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_LIVE_WINDOW_PROOF");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && proof && atoi(proof) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_require_prepublished_map_requested() {
    static const bool requested = []() {
        const char * paged   = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned   = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * require = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_REQUIRE_PREPUBLISHED_MAP");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && require && atoi(require) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_dual_dest_candidate_requested() {
    static const bool requested = []() {
        const char * paged = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
        const char * owned = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
        const char * dual  = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_DUAL_DEST_CANDIDATE");
        return paged && atoi(paged) != 0 && owned && atoi(owned) != 0 && dual && atoi(dual) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_tail_page_producer_snapshot_owned_proof_requested() {
    static const bool requested = []() {
        const char * import = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PRODUCER_STATE_IMPORT");
        const char * proof  = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PRODUCER_SNAPSHOT_OWNED_PROOF");
        return import && atoi(import) != 0 && proof && atoi(proof) != 0;
    }();
    return requested;
}

static bool ggml_cuda_mtp_qblock_owned_tail_write_any_proof_requested() {
    return ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested() ||
        ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested() ||
        ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested();
}

static bool ggml_cuda_mtp_qblock_tail_page_route_complete_trace_enabled() {
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled() ||
            ggml_cuda_mtp_qblock_owned_tail_write_exclusive_trace_enabled() ||
            ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested() ||
            ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested()) {
        return true;
    }
    static const bool enabled = []() {
        const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_ROUTE_COMPLETE_TRACE");
        return v && atoi(v) != 0;
    }();
    return enabled;
}

static dp16_packed_i8_desc_v1 ggml_cuda_make_packed16_i8_desc(
        const ggml_tensor * payload,
        const ggml_cuda_packed16_sidecar_meta & meta) {
    dp16_packed_i8_desc_v1 desc = {};
    desc.version = DP16_PACKED_I8_DESC_VERSION;
    desc.lanes_per_vector = DP16_PACKED_I8X16_LANES;
    desc.words_per_vector = DP16_PACKED_I8X16_WORDS;
    desc.bytes_per_vector = DP16_PACKED_I8X16_BYTES;
    desc.bytes_per_word = DP16_PACKED_I8_WORD_BYTES;
    desc.layout_kind = meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_PAGE16_D16 ?
        DP16_PACKED_I8_LAYOUT_PAGE16_D16 :
        (meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR ?
            DP16_PACKED_I8_LAYOUT_D16_PLANAR :
            (meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_ROW ? DP16_PACKED_I8_LAYOUT_ROW : DP16_PACKED_I8_LAYOUT_UNKNOWN));
    desc.axis_x = DP16_PACKED_I8_AXIS_D16;
    desc.axis_y = DP16_PACKED_I8_AXIS_TOKEN;
    desc.axis_z = DP16_PACKED_I8_AXIS_HEAD;
    desc.logical_x = meta.d ? uint32_t(meta.d / GGML_CUDA_PACKED16_K_D16) : 0;
    desc.logical_y = meta.kv_capacity;
    desc.logical_z = payload && meta.kv_capacity ? uint32_t(payload->ne[1] / meta.kv_capacity) : 0;
    desc.physical_x = desc.logical_x;
    desc.physical_y = desc.logical_y;
    desc.physical_z = desc.logical_z;
    desc.scale_layout = meta.scale_layout == GGML_CUDA_PACKED16_K_SCALE_LAYOUT_PAGE16_QBLOCK ?
        DP16_PACKED_I8_SCALE_LAYOUT_PAGE16_QBLOCK :
        (meta.scale_layout == GGML_CUDA_PACKED16_K_SCALE_LAYOUT_QBLOCK_PLANAR ?
            DP16_PACKED_I8_SCALE_LAYOUT_QBLOCK_PLANAR :
            (meta.scale_layout == GGML_CUDA_PACKED16_K_SCALE_LAYOUT_ROW ? DP16_PACKED_I8_SCALE_LAYOUT_ROW : DP16_PACKED_I8_SCALE_LAYOUT_UNKNOWN));
    desc.scale_axis_x = DP16_PACKED_I8_AXIS_QBLOCK;
    desc.scale_axis_y = DP16_PACKED_I8_AXIS_TOKEN;
    desc.scale_axis_z = DP16_PACKED_I8_AXIS_HEAD;

    constexpr uint64_t word_bytes = DP16_PACKED_I8_WORD_BYTES;
    constexpr uint64_t scale_bytes = sizeof(uint16_t);
    desc.z_stride_bytes = uint64_t(meta.payload_head_stride) * word_bytes;
    desc.scale_z_stride_bytes = uint64_t(meta.scale_head_stride) * scale_bytes;
    if (meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_PAGE16_D16) {
        desc.x_stride_bytes = uint64_t(meta.payload_d16_plane_stride) * word_bytes;
        desc.y_stride_bytes = uint64_t(meta.payload_token_stride) * word_bytes;
        desc.plane_stride_bytes = uint64_t(GGML_CUDA_PACKED16_K_WORDS_PER_PAGE16) * word_bytes;
        desc.scale_x_stride_bytes = uint64_t(meta.scale_qblock_plane_stride) * scale_bytes;
        desc.scale_y_stride_bytes = uint64_t(meta.scale_token_stride) * scale_bytes;
        desc.scale_plane_stride_bytes = uint64_t(GGML_CUDA_PACKED16_K_SCALE_PER_PAGE16) * scale_bytes;
    } else if (meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR) {
        desc.x_stride_bytes = uint64_t(meta.payload_d16_plane_stride) * word_bytes;
        desc.y_stride_bytes = uint64_t(meta.payload_token_stride) * word_bytes;
        desc.plane_stride_bytes = desc.x_stride_bytes;
        desc.scale_x_stride_bytes = uint64_t(meta.scale_qblock_plane_stride) * scale_bytes;
        desc.scale_y_stride_bytes = uint64_t(meta.scale_token_stride) * scale_bytes;
        desc.scale_plane_stride_bytes = desc.scale_x_stride_bytes;
    } else {
        desc.x_stride_bytes = DP16_PACKED_I8X16_BYTES;
        desc.y_stride_bytes = uint64_t(meta.payload_token_stride) * word_bytes;
        desc.plane_stride_bytes = 0;
        desc.scale_x_stride_bytes = scale_bytes;
        desc.scale_y_stride_bytes = uint64_t(meta.scale_token_stride) * scale_bytes;
        desc.scale_plane_stride_bytes = uint64_t(meta.scale_qblock_plane_stride) * scale_bytes;
    }
    return desc;
}

static ggml_cuda_packed16_sidecar_meta ggml_cuda_make_pdmq_k_sidecar_meta(
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const int k_format,
        const int layout_kind,
        const uint32_t kv_capacity,
        const uint32_t d,
        const unsigned long long generation) {
    ggml_cuda_packed16_sidecar_meta meta = {};
    meta.format_version = GGML_CUDA_PDMQ_K_FORMAT_VERSION;
    meta.generation = generation;
    meta.kv_capacity = kv_capacity;
    meta.d = (uint16_t) d;
    meta.block_size = QK8_0;
    meta.layout_kind = layout_kind;
    meta.scale_layout = layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_PAGE16_D16 ?
        GGML_CUDA_PACKED16_K_SCALE_LAYOUT_PAGE16_QBLOCK :
        (layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR ?
            GGML_CUDA_PACKED16_K_SCALE_LAYOUT_QBLOCK_PLANAR : GGML_CUDA_PACKED16_K_SCALE_LAYOUT_ROW);
    meta.k_format = (uint8_t) k_format;
    meta.code_bits = (uint8_t) ggml_cuda_pdmq_k_code_bits(k_format);
    meta.zero_point = (uint8_t) ggml_cuda_pdmq_k_zero_point(k_format);
    meta.code_order = 0;

    const size_t words = d ? size_t(ggml_cuda_pdmq_k_payload_words_per_token(k_format, d)) : (payload ? size_t(payload->ne[0]) : 0);
    const size_t qblocks = d ? size_t(d / QK8_0) : (scales ? size_t(scales->ne[0]) : 0);
    meta.payload_words_per_token = (uint32_t) words;
    meta.payload_head_stride = size_t(kv_capacity) * words;
    meta.scale_head_stride = size_t(kv_capacity) * qblocks;
    if (k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 && layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_PAGE16_D16) {
        meta.payload_token_stride = size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16);
        meta.scale_token_stride = 1;
        meta.payload_d16_plane_stride = size_t(GGML_CUDA_PACKED16_K_PAGE16_TOKENS) * size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16);
        meta.scale_qblock_plane_stride = size_t(GGML_CUDA_PACKED16_K_PAGE16_TOKENS);
    } else if (k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 && layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_D16_PLANAR) {
        meta.payload_token_stride = size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16);
        meta.scale_token_stride = 1;
        meta.payload_d16_plane_stride = size_t(kv_capacity) * size_t(GGML_CUDA_PACKED16_K_WORDS_PER_D16);
        meta.scale_qblock_plane_stride = size_t(kv_capacity);
    } else {
        meta.payload_token_stride = words;
        meta.scale_token_stride = qblocks;
        meta.payload_d16_plane_stride = 0;
        meta.scale_qblock_plane_stride = 1;
    }
    if (k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272) {
        meta.packed_i8_desc = ggml_cuda_make_packed16_i8_desc(payload, meta);
    } else {
        meta.packed_i8_desc = {};
        meta.packed_i8_desc.layout_kind = DP16_PACKED_I8_LAYOUT_UNKNOWN;
        meta.packed_i8_desc.scale_layout = DP16_PACKED_I8_SCALE_LAYOUT_UNKNOWN;
    }
    return meta;
}

static ggml_cuda_packed16_sidecar_meta ggml_cuda_make_packed16_sidecar_meta(
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const int layout_kind,
        const uint32_t kv_capacity,
        const uint32_t d,
        const unsigned long long generation) {
    return ggml_cuda_make_pdmq_k_sidecar_meta(payload, scales,
        GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272, layout_kind, kv_capacity, d, generation);
}

static bool ggml_cuda_mtp_qblock_tail_page_map_valid(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map);
static bool ggml_cuda_mtp_qblock_tail_page_entry_accepts(
        const packed16_registry_entry & entry,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map);
static bool ggml_cuda_mtp_qblock_full_page_map_valid(
        const ggml_cuda_mtp_qblock_full_page_map_v1 & map);
static bool ggml_cuda_mtp_qblock_full_page_map_install_locked(
        packed16_registry_entry & entry,
        const ggml_cuda_mtp_qblock_full_page_map_v1 & map,
        const int32_t * host_block_table,
        const char * reason);

extern "C" {
void llama_kv_cache_register_pdmq_k_with_layout_info(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales, int k_format, int layout_kind, uint32_t kv_capacity, uint32_t d) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    packed16_registry_entry & entry = s_packed16_registry[k_view_data];
    const ggml_tensor * old_payload = entry.payload;
    const ggml_tensor * old_scales  = entry.scales;
    const ggml_cuda_mtp_qblock_tail_page_map_v1 old_map = entry.tail_page_map;
    const ggml_cuda_mtp_qblock_full_page_map_v1 old_full_map_base = entry.full_page_map_base;
    const std::vector<int32_t> old_full_page_table_host = entry.full_page_table_host;
    entry.payload = payload;
    entry.scales  = scales;
    if (k_format != GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272) {
        layout_kind = GGML_CUDA_PACKED16_K_LAYOUT_ROW;
    } else if (layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_UNKNOWN) {
        layout_kind = ggml_cuda_packed16_k_layout_kind_from_env();
    }
    entry.meta = ggml_cuda_make_pdmq_k_sidecar_meta(payload, scales, k_format, layout_kind, kv_capacity, d, ++s_packed16_generation);
    if (old_payload != payload || old_scales != scales) {
        bool rebound_full_map = false;
        if (ggml_cuda_mtp_qblock_full_page_map_valid(old_full_map_base) &&
                old_full_page_table_host.size() == old_full_map_base.block_table_pages) {
            rebound_full_map = ggml_cuda_mtp_qblock_full_page_map_install_locked(
                entry, old_full_map_base, old_full_page_table_host.data(), "sidecar_update_rebind");
        }
        if (!rebound_full_map) {
            entry.full_page_map = {};
            entry.full_page_map_base = {};
            entry.full_page_table_host.clear();
        }
    }

    ggml_cuda_mtp_qblock_tail_page_map_v1 inherited_map = {};
    if (old_payload == payload && old_scales == scales && ggml_cuda_mtp_qblock_tail_page_map_valid(old_map)) {
        inherited_map = old_map;
    }
    for (const auto & kv : s_packed16_registry) {
        if (kv.first == k_view_data) {
            continue;
        }
        const packed16_registry_entry & candidate = kv.second;
        if (candidate.payload == payload && candidate.scales == scales &&
                ggml_cuda_mtp_qblock_tail_page_map_valid(candidate.tail_page_map)) {
            inherited_map = candidate.tail_page_map;
            break;
        }
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(inherited_map) &&
            ggml_cuda_mtp_qblock_tail_page_entry_accepts(entry, s_packed16_tail_page_published_map)) {
        inherited_map = s_packed16_tail_page_published_map;
    }
    entry.tail_page_map = ggml_cuda_mtp_qblock_tail_page_entry_accepts(entry, inherited_map) ? inherited_map : ggml_cuda_mtp_qblock_tail_page_map_v1{};

    auto pending_it = s_packed16_pending_full_page_maps.find(k_view_data);
    if (pending_it != s_packed16_pending_full_page_maps.end()) {
        const packed16_pending_full_page_map_entry & pending = pending_it->second;
        if (!pending.host_block_table.empty() && ggml_cuda_mtp_qblock_full_page_map_install_locked(
                    entry, pending.map, pending.host_block_table.data(), "sidecar_register_pending")) {
            s_packed16_pending_full_page_maps.erase(pending_it);
        }
    }
}

void llama_kv_cache_register_packed16_with_layout_info(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales, int layout_kind, uint32_t kv_capacity, uint32_t d) {
    llama_kv_cache_register_pdmq_k_with_layout_info(k_view_data, payload, scales,
        GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272, layout_kind, kv_capacity, d);
}

void llama_kv_cache_register_packed16_with_layout(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales, int layout_kind) {
    const uint32_t d = payload && payload->ne[0] > 0 ? uint32_t(payload->ne[0] * 4) : uint32_t(GGML_CUDA_PACKED16_K_TILE_D);
    llama_kv_cache_register_packed16_with_layout_info(k_view_data, payload, scales, layout_kind, 0, d);
}

void llama_kv_cache_register_packed16(const void * k_view_data, ggml_tensor * payload, ggml_tensor * scales) {
    llama_kv_cache_register_packed16_with_layout_info(k_view_data, payload, scales, ggml_cuda_packed16_k_layout_kind_from_env(), 0, 0);
}

void llama_kv_cache_get_packed16_tensors(const void * k_view_data, ggml_tensor ** payload, ggml_tensor ** scales) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        *payload = it->second.payload;
        *scales  = it->second.scales;
    } else {
        *payload = nullptr;
        *scales  = nullptr;
    }
}

void llama_kv_cache_get_packed16_metadata(const void * k_view_data, int * layout_kind, unsigned long long * generation) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        if (layout_kind) {
            *layout_kind = it->second.meta.layout_kind;
        }
        if (generation) {
            *generation = it->second.meta.generation;
        }
    } else {
        if (layout_kind) {
            *layout_kind = GGML_CUDA_PACKED16_K_LAYOUT_UNKNOWN;
        }
        if (generation) {
            *generation = 0;
        }
    }
}

void llama_kv_cache_get_packed16_sidecar_meta(const void * k_view_data, ggml_cuda_packed16_sidecar_meta * meta) {
    if (!meta) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        *meta = it->second.meta;
    } else {
        *meta = {};
        meta->layout_kind = GGML_CUDA_PACKED16_K_LAYOUT_UNKNOWN;
        meta->scale_layout = GGML_CUDA_PACKED16_K_SCALE_LAYOUT_UNKNOWN;
        meta->packed_i8_desc.layout_kind = DP16_PACKED_I8_LAYOUT_UNKNOWN;
        meta->packed_i8_desc.scale_layout = DP16_PACKED_I8_SCALE_LAYOUT_UNKNOWN;
    }
}

void llama_kv_cache_get_pdmq_k_sidecar_meta(const void * k_view_data, ggml_cuda_packed16_sidecar_meta * meta) {
    llama_kv_cache_get_packed16_sidecar_meta(k_view_data, meta);
}

void llama_kv_cache_get_packed16_packed_i8_desc(const void * k_view_data, dp16_packed_i8_desc_v1 * desc) {
    if (!desc) {
        return;
    }
    ggml_cuda_packed16_sidecar_meta meta = {};
    llama_kv_cache_get_packed16_sidecar_meta(k_view_data, &meta);
    *desc = meta.packed_i8_desc;
}

static bool ggml_cuda_mtp_qblock_tail_page_map_valid(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) {
    if (!map.active) {
        return false;
    }
    if (map.version != GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_VERSION ||
            map.abi_bytes != sizeof(ggml_cuda_mtp_qblock_tail_page_map_v1)) {
        return false;
    }
    if (map.page_tokens == 0 || map.valid_tail_tokens == 0 ||
            map.block_table_pages == 0 ||
            map.block_table_pages > GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES ||
            map.physical_pages == 0) {
        return false;
    }
    const uint32_t required_pages = (map.valid_tail_tokens + map.page_tokens - 1u) / map.page_tokens;
    if (required_pages == 0 || required_pages > map.block_table_pages) {
        return false;
    }
    for (uint32_t i = 0; i < required_pages; ++i) {
        if (map.block_table[i] < 0 || (uint32_t) map.block_table[i] >= map.physical_pages) {
            return false;
        }
    }
    return true;
}

static bool ggml_cuda_mtp_qblock_full_page_map_valid(
        const ggml_cuda_mtp_qblock_full_page_map_v1 & map) {
    if (!map.active) {
        return false;
    }
    if (map.version != GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_VERSION ||
            map.abi_bytes != sizeof(ggml_cuda_mtp_qblock_full_page_map_v1)) {
        return false;
    }
    if (map.page_tokens == 0 || map.valid_tokens == 0 || map.physical_pages == 0 ||
            map.block_table_pages == 0 || map.block_table == nullptr) {
        return false;
    }
    const uint32_t required_pages = (map.valid_tokens + map.page_tokens - 1u) / map.page_tokens;
    return required_pages != 0 && required_pages <= map.block_table_pages && map.block_table_pages <= map.physical_pages &&
        map.non_identity_page_begin <= map.non_identity_page_end && map.non_identity_page_end <= map.block_table_pages;
}

static uint64_t ggml_cuda_mtp_qblock_tail_page_map_generation(
        const uint32_t logical_base_token,
        const uint32_t valid_tail_tokens,
        const uint32_t flags) {
    const uint64_t raw = (uint64_t(logical_base_token) << 32) ^ uint64_t(valid_tail_tokens) ^ uint64_t(flags);
    return raw == 0 ? 1u : raw;
}

static bool ggml_cuda_mtp_qblock_tail_page_map_equal(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & a,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & b) {
    if (a.version != b.version || a.abi_bytes != b.abi_bytes || a.active != b.active || a.flags != b.flags ||
            a.logical_base_token != b.logical_base_token || a.valid_tail_tokens != b.valid_tail_tokens ||
            a.page_tokens != b.page_tokens || a.physical_pages != b.physical_pages ||
            a.block_table_pages != b.block_table_pages || a.generation != b.generation) {
        return false;
    }
    for (uint32_t i = 0; i < GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES; ++i) {
        if (a.block_table[i] != b.block_table[i]) {
            return false;
        }
    }
    return true;
}

static uint32_t ggml_cuda_mtp_qblock_tail_page_map_required_pages(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) {
    return map.page_tokens == 0 ? 0 : (map.valid_tail_tokens + map.page_tokens - 1u) / map.page_tokens;
}

static void ggml_cuda_mtp_qblock_full_page_map_clear_published_locked() {
    s_packed16_full_page_published_map = {};
    s_packed16_full_page_published_payload = nullptr;
    s_packed16_full_page_published_scales = nullptr;
    s_packed16_full_page_published_meta = {};
}

static bool ggml_cuda_mtp_qblock_full_page_map_published_sidecar_matches_locked(
        const packed16_registry_entry & entry) {
    return s_packed16_full_page_published_payload == entry.payload &&
        s_packed16_full_page_published_scales == entry.scales &&
        s_packed16_full_page_published_meta.k_format == entry.meta.k_format &&
        s_packed16_full_page_published_meta.kv_capacity == entry.meta.kv_capacity &&
        s_packed16_full_page_published_meta.d == entry.meta.d;
}

static bool ggml_cuda_mtp_qblock_full_page_map_publish_snapshot_locked(
        const packed16_registry_entry & entry) {
    const ggml_cuda_mtp_qblock_full_page_map_v1 & src = entry.full_page_map;
    if (!ggml_cuda_mtp_qblock_full_page_map_valid(src) || entry.payload == nullptr || entry.scales == nullptr) {
        return false;
    }
    if (s_packed16_full_page_published_table_capacity < src.block_table_pages) {
        if (s_packed16_full_page_published_table_device != nullptr) {
            CUDA_CHECK(hipFree(s_packed16_full_page_published_table_device));
            s_packed16_full_page_published_table_device = nullptr;
            s_packed16_full_page_published_table_capacity = 0;
        }
        CUDA_CHECK(hipMalloc((void **) &s_packed16_full_page_published_table_device,
            size_t(src.block_table_pages) * sizeof(int32_t)));
        s_packed16_full_page_published_table_capacity = src.block_table_pages;
    }
    CUDA_CHECK(hipMemcpy(s_packed16_full_page_published_table_device, src.block_table,
        size_t(src.block_table_pages) * sizeof(int32_t), hipMemcpyDeviceToDevice));
    s_packed16_full_page_published_map = src;
    s_packed16_full_page_published_map.block_table = s_packed16_full_page_published_table_device;
    s_packed16_full_page_published_payload = entry.payload;
    s_packed16_full_page_published_scales = entry.scales;
    s_packed16_full_page_published_meta = entry.meta;
    return true;
}

static void ggml_cuda_mtp_qblock_full_page_map_refresh_published_locked() {
    for (const auto & kv : s_packed16_registry) {
        if (ggml_cuda_mtp_qblock_full_page_map_publish_snapshot_locked(kv.second)) {
            return;
        }
    }
    // Keep the last owned device-table snapshot until explicit data invalidation.
    // Full current-K maps are batch/global authority and must survive transient
    // per-view registry churn during layer-by-layer verify.
}

static void ggml_cuda_mtp_qblock_full_page_map_restore_identity_locked(const char * reason) {
    size_t restored = 0;
    for (auto & kv : s_packed16_registry) {
        packed16_registry_entry & entry = kv.second;
        if (!ggml_cuda_mtp_qblock_full_page_map_valid(entry.full_page_map_base) || entry.full_page_table_device == nullptr ||
                entry.full_page_table_host.size() != entry.full_page_map_base.block_table_pages) {
            continue;
        }
        CUDA_CHECK(hipMemcpy(entry.full_page_table_device, entry.full_page_table_host.data(),
            size_t(entry.full_page_map_base.block_table_pages) * sizeof(int32_t), hipMemcpyHostToDevice));
        entry.full_page_map = entry.full_page_map_base;
        entry.full_page_map.block_table = entry.full_page_table_device;
        for (uint32_t i = 0; i < 4; ++i) {
            entry.full_page_map.debug_first_pages[i] = i < entry.full_page_map.block_table_pages ? entry.full_page_table_host[i] : -1;
        }
        ++restored;
    }
    ggml_cuda_mtp_qblock_full_page_map_refresh_published_locked();
    if (restored != 0 && ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=restore_base reason=%s restored=%zu\n",
                reason ? reason : "unknown",
                restored);
    }
}

static bool ggml_cuda_mtp_qblock_full_page_map_apply_owned_tail_overlay_locked(
        packed16_registry_entry & entry,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & tail_map,
        const char * reason) {
    if (!ggml_cuda_mtp_qblock_full_page_map_valid(entry.full_page_map_base) || entry.full_page_table_device == nullptr ||
            entry.full_page_table_host.size() != entry.full_page_map_base.block_table_pages ||
            !ggml_cuda_mtp_qblock_tail_page_map_valid(tail_map) ||
            (tail_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) == 0 ||
            (tail_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0 ||
            tail_map.page_tokens != entry.full_page_map_base.page_tokens || tail_map.logical_base_token % tail_map.page_tokens != 0) {
        return false;
    }
    const uint32_t required_pages = ggml_cuda_mtp_qblock_tail_page_map_required_pages(tail_map);
    const uint32_t logical_base_page = tail_map.logical_base_token / tail_map.page_tokens;
    const uint32_t tail_end = tail_map.logical_base_token + tail_map.valid_tail_tokens;
    if (required_pages == 0 || required_pages > tail_map.block_table_pages ||
            logical_base_page + required_pages > entry.full_page_map_base.block_table_pages) {
        return false;
    }
    std::vector<int32_t> table = entry.full_page_table_host;
    for (uint32_t lp = 0; lp < required_pages; ++lp) {
        const int32_t pp = tail_map.block_table[lp];
        if (pp < 0 || uint32_t(pp) >= entry.full_page_map_base.physical_pages) {
            return false;
        }
        table[logical_base_page + lp] = pp;
    }
    CUDA_CHECK(hipMemcpy(entry.full_page_table_device, table.data(),
        size_t(entry.full_page_map_base.block_table_pages) * sizeof(int32_t), hipMemcpyHostToDevice));
    entry.full_page_map = entry.full_page_map_base;
    entry.full_page_map.block_table = entry.full_page_table_device;
    entry.full_page_map.flags = entry.full_page_map_base.flags |
        GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_FLAG_OWNED_TAIL_OVERLAY;
    entry.full_page_map.valid_tokens = std::max(entry.full_page_map_base.valid_tokens, tail_end);
    const bool base_has_non_identity = entry.full_page_map_base.non_identity_page_begin < entry.full_page_map_base.non_identity_page_end;
    entry.full_page_map.non_identity_page_begin = base_has_non_identity ?
        std::min(entry.full_page_map_base.non_identity_page_begin, logical_base_page) : logical_base_page;
    entry.full_page_map.non_identity_page_end = base_has_non_identity ?
        std::max(entry.full_page_map_base.non_identity_page_end, logical_base_page + required_pages) : logical_base_page + required_pages;
    entry.full_page_map.generation = entry.full_page_map_base.generation ^ (tail_map.generation << 1) ^ uint64_t(entry.full_page_map.flags);
    if (entry.full_page_map.generation == 0) {
        entry.full_page_map.generation = 1;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        entry.full_page_map.debug_first_pages[i] = i < entry.full_page_map.block_table_pages ? table[i] : -1;
    }
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=overlay_owned_tail reason=%s logical_base=%u valid_tail=%u logical_base_page=%u pages=%u table0=%d flags=0x%x generation=%llu\n",
                reason ? reason : "unknown",
                tail_map.logical_base_token,
                tail_map.valid_tail_tokens,
                logical_base_page,
                required_pages,
                tail_map.block_table[0],
                tail_map.flags,
                (unsigned long long) tail_map.generation);
    }
    return true;
}

static size_t ggml_cuda_mtp_qblock_full_page_map_apply_owned_tail_overlay_all_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & tail_map,
        const char * reason) {
    size_t applied = 0;
    for (auto & kv : s_packed16_registry) {
        if (ggml_cuda_mtp_qblock_full_page_map_apply_owned_tail_overlay_locked(kv.second, tail_map, reason)) {
            ++applied;
        }
    }
    ggml_cuda_mtp_qblock_full_page_map_refresh_published_locked();
    return applied;
}

static bool ggml_cuda_mtp_qblock_full_page_map_install_locked(
        packed16_registry_entry & entry,
        const ggml_cuda_mtp_qblock_full_page_map_v1 & map,
        const int32_t * host_block_table,
        const char * reason) {
    const bool map_basic_ok = map.active &&
        map.version == GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_VERSION &&
        map.abi_bytes == sizeof(ggml_cuda_mtp_qblock_full_page_map_v1) &&
        map.logical_base_token == 0 && map.valid_tokens != 0 &&
        map.page_tokens == GGML_CUDA_PACKED16_K_PAGE16_TOKENS &&
        map.physical_pages != 0 && map.block_table_pages != 0 &&
        host_block_table != nullptr &&
        map.non_identity_page_begin <= map.non_identity_page_end && map.non_identity_page_end <= map.block_table_pages;
    const bool sidecar_ok = entry.payload && entry.scales &&
        entry.meta.k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 &&
        entry.meta.d == GGML_CUDA_PACKED16_K_TILE_D &&
        map_basic_ok && entry.meta.kv_capacity >= map.physical_pages * map.page_tokens;
    if (!map_basic_ok || !sidecar_ok) {
        return false;
    }
    const uint32_t required_pages = (map.valid_tokens + map.page_tokens - 1u) / map.page_tokens;
    if (required_pages == 0 || required_pages > map.block_table_pages || map.block_table_pages > map.physical_pages) {
        return false;
    }
    for (uint32_t i = 0; i < required_pages; ++i) {
        if (host_block_table[i] < 0 || uint32_t(host_block_table[i]) >= map.physical_pages) {
            return false;
        }
    }
    if (entry.full_page_table_capacity < map.block_table_pages) {
        if (entry.full_page_table_device != nullptr) {
            CUDA_CHECK(hipFree(entry.full_page_table_device));
            entry.full_page_table_device = nullptr;
            entry.full_page_table_capacity = 0;
        }
        CUDA_CHECK(hipMalloc((void **) &entry.full_page_table_device, size_t(map.block_table_pages) * sizeof(int32_t)));
        entry.full_page_table_capacity = map.block_table_pages;
    }
    CUDA_CHECK(hipMemcpy(entry.full_page_table_device, host_block_table,
        size_t(map.block_table_pages) * sizeof(int32_t), hipMemcpyHostToDevice));
    entry.full_page_table_host.assign(host_block_table, host_block_table + map.block_table_pages);
    entry.full_page_map_base = map;
    entry.full_page_map_base.block_table = entry.full_page_table_device;
    entry.full_page_map = entry.full_page_map_base;
    for (uint32_t i = 0; i < 4; ++i) {
        entry.full_page_map_base.debug_first_pages[i] = i < map.block_table_pages ? host_block_table[i] : -1;
        entry.full_page_map.debug_first_pages[i] = entry.full_page_map_base.debug_first_pages[i];
    }
    if (s_packed16_tail_page_owned_write_ready.k_ready && s_packed16_tail_page_owned_write_ready.v_ready &&
            ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map)) {
        (void) ggml_cuda_mtp_qblock_full_page_map_apply_owned_tail_overlay_locked(
            entry, s_packed16_tail_page_owned_write_ready.map, reason ? reason : "install");
    }
    ggml_cuda_mtp_qblock_full_page_map_refresh_published_locked();
    return true;
}

static bool ggml_cuda_mtp_qblock_tail_page_map_translate(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        const uint32_t logical_token,
        uint32_t * logical_page,
        uint32_t * slot,
        uint32_t * physical_page,
        uint64_t * physical_slot) {
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) || logical_token < map.logical_base_token) {
        return false;
    }
    const uint32_t rel = logical_token - map.logical_base_token;
    if (rel >= map.valid_tail_tokens) {
        return false;
    }
    const uint32_t lp = rel / map.page_tokens;
    const uint32_t required_pages = ggml_cuda_mtp_qblock_tail_page_map_required_pages(map);
    if (lp >= required_pages || lp >= map.block_table_pages || lp >= GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES) {
        return false;
    }
    const int32_t pp_i32 = map.block_table[lp];
    if (pp_i32 < 0 || uint32_t(pp_i32) >= map.physical_pages) {
        return false;
    }
    const uint32_t slot_u32 = rel % map.page_tokens;
    const uint32_t pp = uint32_t(pp_i32);
    if (logical_page != nullptr) {
        *logical_page = lp;
    }
    if (slot != nullptr) {
        *slot = slot_u32;
    }
    if (physical_page != nullptr) {
        *physical_page = pp;
    }
    if (physical_slot != nullptr) {
        *physical_slot = uint64_t(pp) * uint64_t(map.page_tokens) + uint64_t(slot_u32);
    }
    return true;
}

static bool ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked() {
    return s_packed16_tail_page_route_complete.expected_layer_count != 0 &&
        s_packed16_tail_page_route_complete.expected_page_count != 0 &&
        s_packed16_tail_page_route_complete.expected_layer_page_count != 0 &&
        s_packed16_tail_page_route_complete.bound_layer_pages.size() >=
            s_packed16_tail_page_route_complete.expected_layer_page_count;
}

static bool ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) {
    return ggml_cuda_mtp_qblock_tail_page_map_valid(map) &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_route_complete.map, map) &&
        ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked();
}


static bool ggml_cuda_mtp_qblock_tail_page_owned_plan_same_map_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) {
    return s_packed16_tail_page_owned_write_plan.plan_id != 0 &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_plan.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_plan.map, map);
}


static void ggml_cuda_mtp_qblock_tail_page_owned_plan_reset_locked(const char * reason) {
    if (s_packed16_tail_page_owned_write_plan.plan_id == 0 &&
            !ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_plan.map)) {
        return;
    }
    if (ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=reset reason=%s plan_id=%llu plan_seq=%llu expected_layers=%u expected_pages=%u bound_layer_pages_at_plan=%llu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu\n",
                reason ? reason : "unknown",
                (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_id,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_event_seq,
                s_packed16_tail_page_owned_write_plan.expected_layer_count,
                s_packed16_tail_page_owned_write_plan.expected_page_count,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.bound_layer_pages_at_plan,
                s_packed16_tail_page_owned_write_plan.map.logical_base_token,
                s_packed16_tail_page_owned_write_plan.map.valid_tail_tokens,
                s_packed16_tail_page_owned_write_plan.map.block_table[0],
                s_packed16_tail_page_owned_write_plan.map.flags,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.map.generation);
    }
    s_packed16_tail_page_owned_write_plan = {};
}

static void ggml_cuda_mtp_qblock_tail_page_owned_plan_note_expected_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        uint32_t expected_layer_count,
        uint32_t expected_page_count) {
    if (!ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested()) {
        return;
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) || expected_layer_count == 0 || expected_page_count == 0 ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) == 0 ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0) {
        ggml_cuda_mtp_qblock_tail_page_owned_plan_reset_locked("plan_expected_invalid");
        return;
    }

    const bool same_plan = ggml_cuda_mtp_qblock_tail_page_owned_plan_same_map_locked(map) &&
        s_packed16_tail_page_owned_write_plan.expected_layer_count == expected_layer_count &&
        s_packed16_tail_page_owned_write_plan.expected_page_count == expected_page_count;
    if (!same_plan) {
        s_packed16_tail_page_owned_write_plan.plan_id = ++s_packed16_tail_page_owned_write_next_plan_id;
        s_packed16_tail_page_owned_write_plan.map = map;
    }
    s_packed16_tail_page_owned_write_plan.plan_event_seq = ++s_packed16_tail_page_owned_deferred_proof_event_seq;
    s_packed16_tail_page_owned_write_plan.expected_layer_count = expected_layer_count;
    s_packed16_tail_page_owned_write_plan.expected_page_count = expected_page_count;
    s_packed16_tail_page_owned_write_plan.expected_layer_page_count = uint64_t(expected_layer_count) * uint64_t(expected_page_count);
    s_packed16_tail_page_owned_write_plan.bind_count_at_plan = s_packed16_tail_page_route_complete.bind_count;
    s_packed16_tail_page_owned_write_plan.bound_layer_pages_at_plan = s_packed16_tail_page_route_complete.bound_layer_pages.size();

    fprintf(stderr,
            "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=expected status=ok plan_id=%llu plan_seq=%llu new_plan=%d expected_layers=%u expected_pages=%u expected_layer_pages=%llu bound_layer_pages_at_plan=%llu bind_count_at_plan=%llu route_complete=%d logical_base=%u logical_end=%u valid_tail=%u page_tokens=%u table_pages=%u table0=%d flags=0x%x generation=%llu\n",
            (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_id,
            (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_event_seq,
            same_plan ? 0 : 1,
            expected_layer_count,
            expected_page_count,
            (unsigned long long) s_packed16_tail_page_owned_write_plan.expected_layer_page_count,
            (unsigned long long) s_packed16_tail_page_owned_write_plan.bound_layer_pages_at_plan,
            (unsigned long long) s_packed16_tail_page_owned_write_plan.bind_count_at_plan,
            ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(map) ? 1 : 0,
            map.logical_base_token,
            map.logical_base_token + map.valid_tail_tokens,
            map.valid_tail_tokens,
            map.page_tokens,
            map.block_table_pages,
            map.block_table[0],
            map.flags,
            (unsigned long long) map.generation);
}

static void ggml_cuda_mtp_qblock_tail_page_owned_live_window_route_reset_locked(const char * reason) {
    if (s_packed16_tail_page_owned_live_window.route_complete_event_seq != 0 &&
            ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=route_reset reason=%s route_complete_seq=%llu route_complete_bind_count=%llu route_complete_highwater=%u highwater_seq=%llu highwater_update_seq=%llu highwater=%u logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu reset_count=%llu\n",
                reason ? reason : "unknown",
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_bind_count,
                s_packed16_tail_page_owned_live_window.route_complete_highwater,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_update_seq,
                s_packed16_tail_page_owned_live_window.highwater_nk,
                s_packed16_tail_page_owned_live_window.route_complete_map.logical_base_token,
                s_packed16_tail_page_owned_live_window.route_complete_map.valid_tail_tokens,
                s_packed16_tail_page_owned_live_window.route_complete_map.block_table[0],
                s_packed16_tail_page_owned_live_window.route_complete_map.flags,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_map.generation,
                (unsigned long long) s_packed16_tail_page_owned_live_window.reset_count);
    }
    s_packed16_tail_page_owned_live_window.route_complete_event_seq = 0;
    s_packed16_tail_page_owned_live_window.route_complete_bind_count = 0;
    s_packed16_tail_page_owned_live_window.route_complete_highwater = 0;
    s_packed16_tail_page_owned_live_window.route_complete_generation = 0;
    s_packed16_tail_page_owned_live_window.route_complete_map = {};
}

static void ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked(const char * reason) {
    const bool had_state = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) ||
        s_packed16_tail_page_route_complete.expected_layer_count != 0 ||
        s_packed16_tail_page_route_complete.expected_page_count != 0 ||
        !s_packed16_tail_page_route_complete.bound_layers.empty() ||
        !s_packed16_tail_page_route_complete.bound_layer_pages.empty();
    if (had_state && ggml_cuda_mtp_qblock_tail_page_route_complete_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_ROUTE_COMPLETE: op=reset reason=%s expected_layers=%u expected_pages=%u expected_layer_pages=%llu bound_layers=%zu bound_layer_pages=%zu page_mask=0x%x complete=%d logical_base=%u valid_tail=%u table0=%d generation=%llu\n",
                reason ? reason : "unknown",
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                (unsigned long long) s_packed16_tail_page_route_complete.expected_layer_page_count,
                s_packed16_tail_page_route_complete.bound_layers.size(),
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                s_packed16_tail_page_route_complete.bound_page_mask,
                ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked() ? 1 : 0,
                s_packed16_tail_page_route_complete.map.logical_base_token,
                s_packed16_tail_page_route_complete.map.valid_tail_tokens,
                s_packed16_tail_page_route_complete.map.block_table[0],
                (unsigned long long) s_packed16_tail_page_route_complete.map.generation);
    }
    ggml_cuda_mtp_qblock_tail_page_owned_plan_reset_locked(reason);
    ggml_cuda_mtp_qblock_tail_page_owned_live_window_route_reset_locked(reason);
    s_packed16_tail_page_route_complete = {};
}

void llama_kv_cache_note_mtp_qblock_tail_page_route_expected(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 * map,
        uint32_t expected_layer_count) {
    if (map == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    const uint32_t expected_page_count = ggml_cuda_mtp_qblock_tail_page_map_required_pages(*map);
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(*map) || expected_layer_count == 0 || expected_page_count == 0) {
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("route_expected_invalid");
        return;
    }
    if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
            !ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_route_complete.map, *map)) {
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("route_expected_new_map");
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map)) {
        s_packed16_tail_page_route_complete.map = *map;
    }
    s_packed16_tail_page_route_complete.expected_layer_count = expected_layer_count;
    s_packed16_tail_page_route_complete.expected_page_count = expected_page_count;
    s_packed16_tail_page_route_complete.expected_layer_page_count = uint64_t(expected_layer_count) * uint64_t(expected_page_count);
    if (ggml_cuda_mtp_qblock_tail_page_route_complete_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_ROUTE_COMPLETE: op=expected status=ok expected_layers=%u expected_pages=%u expected_layer_pages=%llu bound_layers=%zu bound_layer_pages=%zu page_mask=0x%x complete=%d logical_base=%u logical_end=%u valid_tail=%u page_tokens=%u table_pages=%u table0=%d flags=0x%x generation=%llu\n",
                expected_layer_count,
                expected_page_count,
                (unsigned long long) s_packed16_tail_page_route_complete.expected_layer_page_count,
                s_packed16_tail_page_route_complete.bound_layers.size(),
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                s_packed16_tail_page_route_complete.bound_page_mask,
                ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked() ? 1 : 0,
                map->logical_base_token,
                map->logical_base_token + map->valid_tail_tokens,
                map->valid_tail_tokens,
                map->page_tokens,
                map->block_table_pages,
                map->block_table[0],
                map->flags,
                (unsigned long long) map->generation);
    }
    ggml_cuda_mtp_qblock_tail_page_owned_plan_note_expected_locked(*map, expected_layer_count, expected_page_count);
}

static void ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked(const char * trigger);

static void ggml_cuda_mtp_qblock_tail_page_route_complete_record_bind_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        int layer,
        int nk,
        const char * node_name) {
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) || layer < 0 || nk <= 0) {
        return;
    }
    const bool was_complete = ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked();
    if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
            !ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_route_complete.map, map)) {
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("route_bind_new_map");
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map)) {
        s_packed16_tail_page_route_complete.map = map;
    }
    if (ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked() &&
            !ggml_cuda_mtp_qblock_tail_page_route_complete_trace_enabled() &&
            !ggml_cuda_mtp_qblock_owned_tail_write_any_proof_requested()) {
        return;
    }
    s_packed16_tail_page_route_complete.bound_layers.insert(layer);
    ++s_packed16_tail_page_route_complete.bind_count;
    const uint32_t nk_u32 = (uint32_t) nk;
    if (s_packed16_tail_page_route_complete.min_bound_nk == 0 || nk_u32 < s_packed16_tail_page_route_complete.min_bound_nk) {
        s_packed16_tail_page_route_complete.min_bound_nk = nk_u32;
    }
    if (nk_u32 > s_packed16_tail_page_route_complete.max_bound_nk) {
        s_packed16_tail_page_route_complete.max_bound_nk = nk_u32;
    }

    const uint32_t required_pages = ggml_cuda_mtp_qblock_tail_page_map_required_pages(map);
    uint32_t translated_pages = 0;
    uint32_t last_logical_page = 0;
    uint32_t last_slot = 0;
    uint32_t last_physical_page = 0;
    uint64_t last_physical_slot = 0;
    for (uint32_t lp = 0; lp < required_pages && lp < GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES; ++lp) {
        const uint32_t logical_token = map.logical_base_token + lp * map.page_tokens;
        uint32_t logical_page = 0;
        uint32_t slot = 0;
        uint32_t physical_page = 0;
        uint64_t physical_slot = 0;
        if (!ggml_cuda_mtp_qblock_tail_page_map_translate(map, logical_token, &logical_page, &slot, &physical_page, &physical_slot)) {
            ++s_packed16_tail_page_route_complete.slot_translate_reject_count;
            continue;
        }
        const uint64_t layer_page_key = (uint64_t(uint32_t(layer)) << 32) | uint64_t(logical_page);
        s_packed16_tail_page_route_complete.bound_layer_pages.insert(layer_page_key);
        if (logical_page < 32) {
            s_packed16_tail_page_route_complete.bound_page_mask |= (uint32_t(1) << logical_page);
        }
        if (s_packed16_tail_page_route_complete.min_physical_page == 0 || physical_page < s_packed16_tail_page_route_complete.min_physical_page) {
            s_packed16_tail_page_route_complete.min_physical_page = physical_page;
        }
        if (physical_page > s_packed16_tail_page_route_complete.max_physical_page) {
            s_packed16_tail_page_route_complete.max_physical_page = physical_page;
        }
        ++s_packed16_tail_page_route_complete.slot_translate_count;
        ++translated_pages;
        last_logical_page = logical_page;
        last_slot = slot;
        last_physical_page = physical_page;
        last_physical_slot = physical_slot;
    }

    const bool complete = ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked();
    if (complete) {
        ++s_packed16_tail_page_route_complete.complete_count;
    }
    if (ggml_cuda_mtp_qblock_tail_page_route_complete_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_ROUTE_COMPLETE: op=bind status=ok layer=%d node=%s nk=%d expected_layers=%u expected_pages=%u expected_layer_pages=%llu bound_layers=%zu bound_layer_pages=%zu page_mask=0x%x complete=%d translated_pages=%u translate_rejects=%llu min_nk=%u max_nk=%u min_phys=%u max_phys=%u last_lp=%u last_slot=%u last_phys=%u last_phys_slot=%llu bind_count=%llu complete_count=%llu logical_base=%u logical_end=%u valid_tail=%u page_tokens=%u table_pages=%u table0=%d generation=%llu\n",
                layer,
                node_name ? node_name : "(null)",
                nk,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                (unsigned long long) s_packed16_tail_page_route_complete.expected_layer_page_count,
                s_packed16_tail_page_route_complete.bound_layers.size(),
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                s_packed16_tail_page_route_complete.bound_page_mask,
                complete ? 1 : 0,
                translated_pages,
                (unsigned long long) s_packed16_tail_page_route_complete.slot_translate_reject_count,
                s_packed16_tail_page_route_complete.min_bound_nk,
                s_packed16_tail_page_route_complete.max_bound_nk,
                s_packed16_tail_page_route_complete.min_physical_page,
                s_packed16_tail_page_route_complete.max_physical_page,
                last_logical_page,
                last_slot,
                last_physical_page,
                (unsigned long long) last_physical_slot,
                (unsigned long long) s_packed16_tail_page_route_complete.bind_count,
                (unsigned long long) s_packed16_tail_page_route_complete.complete_count,
                map.logical_base_token,
                map.logical_base_token + map.valid_tail_tokens,
                map.valid_tail_tokens,
                map.page_tokens,
                map.block_table_pages,
                map.block_table[0],
                (unsigned long long) map.generation);
    }
    if (complete && ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested() &&
            (!was_complete || s_packed16_tail_page_owned_live_window.route_complete_event_seq == 0)) {
        const uint64_t event_seq = ++s_packed16_tail_page_owned_deferred_proof_event_seq;
        s_packed16_tail_page_owned_live_window.route_complete_event_seq = event_seq;
        s_packed16_tail_page_owned_live_window.route_complete_bind_count = s_packed16_tail_page_route_complete.bind_count;
        s_packed16_tail_page_owned_live_window.route_complete_highwater = s_packed16_tail_page_consumer_no_map_nk_max;
        s_packed16_tail_page_owned_live_window.route_complete_generation = s_packed16_generation;
        s_packed16_tail_page_owned_live_window.route_complete_map = map;
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=route_complete event_seq=%llu after_highwater=%d highwater_seq=%llu highwater_update_seq=%llu highwater=%u bind_count=%llu expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu\n",
                (unsigned long long) event_seq,
                s_packed16_tail_page_owned_live_window.highwater_event_seq != 0 &&
                    event_seq > s_packed16_tail_page_owned_live_window.highwater_event_seq ? 1 : 0,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_update_seq,
                s_packed16_tail_page_owned_live_window.highwater_nk,
                (unsigned long long) s_packed16_tail_page_route_complete.bind_count,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                map.logical_base_token,
                map.valid_tail_tokens,
                map.block_table[0],
                map.flags,
                (unsigned long long) map.generation);
        ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked("route_complete_live_window");
    }
}

enum packed16_tail_page_owned_deferred_proof_reason : uint8_t {
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK = 0,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_KIND,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_MAP,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NOT_OWNED_MAP,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_PHYSICAL_PAGE_MISMATCH,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_RANGE_NOT_COVERED,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_READY_MAP_MISMATCH,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_KV_PAIR_NOT_READY,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_ROUTE_INCOMPLETE,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BEFORE_ROUTE_HIGHWATER,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_K_DEFERRED_NOT_SEEN,
    PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_REASON_COUNT,
};

static const char * ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_reason_name(
        const packed16_tail_page_owned_deferred_proof_reason reason) {
    switch (reason) {
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK:                    return "ok";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_KIND:              return "bad_kind";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_MAP:               return "bad_map";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NOT_OWNED_MAP:         return "not_owned_map";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_PHYSICAL_PAGE_MISMATCH:return "physical_page_mismatch";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_RANGE_NOT_COVERED:     return "range_not_covered";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_READY_MAP_MISMATCH:    return "ready_map_mismatch";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_KV_PAIR_NOT_READY:     return "kv_pair_not_ready";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_ROUTE_INCOMPLETE:      return "route_incomplete";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER:    return "no_route_highwater";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BEFORE_ROUTE_HIGHWATER:return "before_route_highwater";
        case PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_K_DEFERRED_NOT_SEEN:   return "k_deferred_not_seen";
        default:                                                            return "unknown";
    }
}

static bool ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_attempt_same_range(
        const packed16_tail_page_owned_deferred_proof_attempt & a,
        const packed16_tail_page_owned_deferred_proof_attempt & b) {
    return a.logical_base == b.logical_base &&
        a.n_tokens == b.n_tokens &&
        a.physical_page == b.physical_page &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(a.map, b.map);
}

static packed16_tail_page_owned_deferred_proof_reason ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_base_reason_locked(
        const packed16_tail_page_owned_deferred_proof_attempt & attempt) {
    const uint64_t req_begin = attempt.logical_base;
    const uint64_t req_end = req_begin + attempt.n_tokens;
    const uint64_t map_begin = attempt.map.logical_base_token;
    const uint64_t map_end = map_begin + attempt.map.valid_tail_tokens;
    if (!attempt.is_k && !attempt.is_v) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_KIND;
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(attempt.map)) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_MAP;
    }
    if ((attempt.map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) == 0 ||
            (attempt.map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NOT_OWNED_MAP;
    }
    uint32_t mapped_logical_page = 0;
    uint32_t mapped_slot = 0;
    uint32_t mapped_physical_page = 0;
    uint64_t mapped_physical_slot = 0;
    if (!ggml_cuda_mtp_qblock_tail_page_map_translate(
            attempt.map, attempt.logical_base, &mapped_logical_page, &mapped_slot, &mapped_physical_page, &mapped_physical_slot) ||
            mapped_physical_page != attempt.physical_page) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_PHYSICAL_PAGE_MISMATCH;
    }
    if (attempt.n_tokens == 0 || req_end <= req_begin || req_begin < map_begin || req_end > map_end) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_RANGE_NOT_COVERED;
    }
    const bool same_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, attempt.map) &&
        s_packed16_tail_page_owned_write_ready.physical_page == attempt.physical_page;
    if (!same_map) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_READY_MAP_MISMATCH;
    }
    if (!s_packed16_tail_page_owned_write_ready.k_ready || !s_packed16_tail_page_owned_write_ready.v_ready) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_KV_PAIR_NOT_READY;
    }
    if (!ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(attempt.map)) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_ROUTE_INCOMPLETE;
    }
    if (s_packed16_tail_page_consumer_no_map_nk_max == 0) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER;
    }
    if (attempt.logical_base < s_packed16_tail_page_consumer_no_map_nk_max) {
        return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BEFORE_ROUTE_HIGHWATER;
    }
    return PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK;
}

static void ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_reset_locked(const char * reason) {
    if (s_packed16_tail_page_owned_deferred_proof.attempts.empty() &&
            s_packed16_tail_page_owned_deferred_proof.next_attempt_id == 0 &&
            s_packed16_tail_page_owned_deferred_proof.dropped_attempts == 0) {
        return;
    }
    if (ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_DEFERRED_PROOF: op=reset reason=%s attempts=%zu next_attempt_id=%llu dropped=%llu resolves=%llu\n",
                reason ? reason : "unknown",
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.next_attempt_id,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.dropped_attempts,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
    }
    if (ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=attempt_reset reason=%s attempts=%zu next_attempt_id=%llu dropped=%llu resolves=%llu\n",
                reason ? reason : "unknown",
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.next_attempt_id,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.dropped_attempts,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
    }
    if (ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=attempt_reset reason=%s attempts=%zu next_attempt_id=%llu dropped=%llu resolves=%llu highwater_seq=%llu highwater_update_seq=%llu highwater=%u route_complete_seq=%llu route_complete_highwater=%u reset_count=%llu\n",
                reason ? reason : "unknown",
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.next_attempt_id,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.dropped_attempts,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_update_seq,
                s_packed16_tail_page_owned_live_window.highwater_nk,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_event_seq,
                s_packed16_tail_page_owned_live_window.route_complete_highwater,
                (unsigned long long) s_packed16_tail_page_owned_live_window.reset_count);
    }
    s_packed16_tail_page_owned_deferred_proof = {};
}

static void ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked(const char * trigger) {
    const bool deferred_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested();
    const bool plan_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested();
    const bool live_window_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested();
    if (!ggml_cuda_mtp_qblock_owned_tail_write_any_proof_requested() ||
            s_packed16_tail_page_owned_deferred_proof.attempts.empty()) {
        return;
    }
    uint64_t reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_REASON_COUNT] = {};
    std::vector<bool> k_would_skip_by_attempt(s_packed16_tail_page_owned_deferred_proof.attempts.size(), false);
    std::vector<bool> k_all_bound_would_skip_by_attempt(s_packed16_tail_page_owned_deferred_proof.attempts.size(), false);
    std::vector<bool> k_plan_bound_no_miss_by_attempt(s_packed16_tail_page_owned_deferred_proof.attempts.size(), false);
    uint64_t k_attempts = 0;
    uint64_t v_attempts = 0;
    uint64_t k_would_skip = 0;
    uint64_t v_would_skip = 0;
    uint64_t k_route_ready = 0;
    uint64_t v_route_ready = 0;
    uint64_t k_no_highwater = 0;
    uint64_t v_no_highwater = 0;
    uint64_t k_pre_highwater = 0;
    uint64_t v_pre_highwater = 0;
    uint64_t k_post_highwater = 0;
    uint64_t v_post_highwater = 0;
    uint64_t k_cross_highwater = 0;
    uint64_t v_cross_highwater = 0;
    uint64_t k_post_highwater_tokens = 0;
    uint64_t v_post_highwater_tokens = 0;
    uint64_t k_retired_no_highwater = 0;
    uint64_t v_retired_no_highwater = 0;
    uint64_t k_retired_pre_highwater = 0;
    uint64_t v_retired_pre_highwater = 0;
    uint64_t k_retired_post_highwater = 0;
    uint64_t v_retired_post_highwater = 0;
    uint64_t k_record_no_highwater = 0;
    uint64_t v_record_no_highwater = 0;
    uint64_t k_record_pre_highwater = 0;
    uint64_t v_record_pre_highwater = 0;
    uint64_t k_record_post_highwater = 0;
    uint64_t v_record_post_highwater = 0;
    uint64_t k_record_cross_highwater = 0;
    uint64_t v_record_cross_highwater = 0;
    uint64_t k_record_no_to_live_highwater = 0;
    uint64_t v_record_no_to_live_highwater = 0;
    uint64_t first_record_event_seq = 0;
    uint64_t last_record_event_seq = 0;
    uint64_t k_plan_ready = 0;
    uint64_t v_plan_ready = 0;
    uint64_t k_plan_before_record = 0;
    uint64_t v_plan_before_record = 0;
    uint64_t k_plan_write_through = 0;
    uint64_t v_plan_write_through = 0;
    uint64_t k_plan_owned_write = 0;
    uint64_t v_plan_owned_write = 0;
    uint64_t k_plan_exclusive_skip = 0;
    uint64_t v_plan_exclusive_skip = 0;
    uint64_t k_plan_route_complete = 0;
    uint64_t v_plan_route_complete = 0;
    uint64_t k_plan_route_incomplete = 0;
    uint64_t v_plan_route_incomplete = 0;
    uint64_t k_plan_no_highwater_owned = 0;
    uint64_t v_plan_no_highwater_owned = 0;
    uint64_t k_plan_pre_highwater_owned = 0;
    uint64_t v_plan_pre_highwater_owned = 0;
    uint64_t k_plan_post_highwater_owned = 0;
    uint64_t v_plan_post_highwater_owned = 0;
    uint64_t k_plan_cross_highwater_owned = 0;
    uint64_t v_plan_cross_highwater_owned = 0;
    uint64_t k_plan_all_expected_known = 0;
    uint64_t v_plan_all_expected_known = 0;
    uint64_t k_plan_bound_before_record = 0;
    uint64_t v_plan_bound_before_record = 0;
    uint64_t k_plan_bound_at_resolve = 0;
    uint64_t v_plan_bound_at_resolve = 0;
    uint64_t k_plan_missing_layer_pages_at_record = 0;
    uint64_t v_plan_missing_layer_pages_at_record = 0;
    uint64_t k_plan_known_no_miss = 0;
    uint64_t v_plan_known_no_miss = 0;
    uint64_t k_plan_bound_no_miss_would_skip = 0;
    uint64_t v_plan_bound_no_miss_would_skip = 0;
    uint64_t v_plan_bound_no_miss_k_missing = 0;
    uint64_t k_live_after_highwater = 0;
    uint64_t v_live_after_highwater = 0;
    uint64_t k_live_after_route_complete = 0;
    uint64_t v_live_after_route_complete = 0;
    uint64_t k_live_after_both = 0;
    uint64_t v_live_after_both = 0;
    uint64_t k_live_after_both_pre_highwater = 0;
    uint64_t v_live_after_both_pre_highwater = 0;
    uint64_t k_live_after_both_post_highwater = 0;
    uint64_t v_live_after_both_post_highwater = 0;
    uint64_t k_live_after_both_cross_highwater = 0;
    uint64_t v_live_after_both_cross_highwater = 0;
    uint64_t k_live_after_both_owned = 0;
    uint64_t v_live_after_both_owned = 0;
    uint64_t k_live_base_ok = 0;
    uint64_t v_live_base_ok = 0;
    uint64_t k_all_bound_would_skip = 0;
    uint64_t v_all_bound_would_skip = 0;
    uint64_t v_all_bound_k_missing = 0;
    for (size_t i = 0; i < s_packed16_tail_page_owned_deferred_proof.attempts.size(); ++i) {
        const packed16_tail_page_owned_deferred_proof_attempt & attempt = s_packed16_tail_page_owned_deferred_proof.attempts[i];
        if (attempt.is_k) {
            ++k_attempts;
        } else if (attempt.is_v) {
            ++v_attempts;
        }
        if (attempt.record_event_seq != 0) {
            if (first_record_event_seq == 0 || attempt.record_event_seq < first_record_event_seq) {
                first_record_event_seq = attempt.record_event_seq;
            }
            if (attempt.record_event_seq > last_record_event_seq) {
                last_record_event_seq = attempt.record_event_seq;
            }
        }
        if (attempt.record_plan_ready) {
            if (attempt.is_k) {
                ++k_plan_ready;
            } else if (attempt.is_v) {
                ++v_plan_ready;
            }
        }
        if (attempt.record_plan_before_record) {
            if (attempt.is_k) {
                ++k_plan_before_record;
            } else if (attempt.is_v) {
                ++v_plan_before_record;
            }
        }
        if (attempt.record_plan_write_through) {
            if (attempt.is_k) {
                ++k_plan_write_through;
            } else if (attempt.is_v) {
                ++v_plan_write_through;
            }
        }
        if (attempt.record_plan_owned_write) {
            if (attempt.is_k) {
                ++k_plan_owned_write;
            } else if (attempt.is_v) {
                ++v_plan_owned_write;
            }
            const uint64_t req_begin = attempt.logical_base;
            const uint64_t req_end = req_begin + attempt.n_tokens;
            const uint32_t record_highwater = attempt.record_live_highwater;
            if (record_highwater == 0) {
                if (attempt.is_k) {
                    ++k_plan_no_highwater_owned;
                } else if (attempt.is_v) {
                    ++v_plan_no_highwater_owned;
                }
            } else if (req_begin < record_highwater) {
                if (attempt.is_k) {
                    ++k_plan_pre_highwater_owned;
                } else if (attempt.is_v) {
                    ++v_plan_pre_highwater_owned;
                }
                if (req_end > record_highwater) {
                    if (attempt.is_k) {
                        ++k_plan_cross_highwater_owned;
                    } else if (attempt.is_v) {
                        ++v_plan_cross_highwater_owned;
                    }
                }
            } else {
                if (attempt.is_k) {
                    ++k_plan_post_highwater_owned;
                } else if (attempt.is_v) {
                    ++v_plan_post_highwater_owned;
                }
            }
        }
        const uint64_t plan_expected_layer_pages =
            uint64_t(attempt.record_plan_expected_layers) * uint64_t(attempt.record_plan_expected_pages);
        const bool plan_all_expected_known = attempt.record_plan_ready && plan_expected_layer_pages != 0;
        if (plan_all_expected_known) {
            if (attempt.is_k) {
                ++k_plan_all_expected_known;
            } else if (attempt.is_v) {
                ++v_plan_all_expected_known;
            }
            const uint64_t missing_at_record = attempt.record_plan_bound_layer_pages >= plan_expected_layer_pages ?
                0 : plan_expected_layer_pages - attempt.record_plan_bound_layer_pages;
            if (missing_at_record == 0 && attempt.record_plan_before_record) {
                if (attempt.is_k) {
                    ++k_plan_bound_before_record;
                } else if (attempt.is_v) {
                    ++v_plan_bound_before_record;
                }
            } else {
                if (attempt.is_k) {
                    k_plan_missing_layer_pages_at_record += missing_at_record;
                } else if (attempt.is_v) {
                    v_plan_missing_layer_pages_at_record += missing_at_record;
                }
            }
        }
        const bool plan_same_map_bound_at_resolve = plan_all_expected_known &&
            ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(attempt.map) &&
            s_packed16_tail_page_route_complete.expected_layer_count == attempt.record_plan_expected_layers &&
            s_packed16_tail_page_route_complete.expected_page_count == attempt.record_plan_expected_pages &&
            s_packed16_tail_page_route_complete.expected_layer_page_count == plan_expected_layer_pages;
        if (plan_same_map_bound_at_resolve) {
            if (attempt.is_k) {
                ++k_plan_bound_at_resolve;
            } else if (attempt.is_v) {
                ++v_plan_bound_at_resolve;
            }
        }
        const bool plan_known_no_miss = plan_all_expected_known && attempt.record_plan_before_record &&
            attempt.record_plan_owned_write && s_packed16_tail_page_consumer_no_map_nk_max == 0;
        if (plan_known_no_miss) {
            if (attempt.is_k) {
                ++k_plan_known_no_miss;
            } else if (attempt.is_v) {
                ++v_plan_known_no_miss;
            }
        }
        if (attempt.is_k && plan_known_no_miss && plan_same_map_bound_at_resolve) {
            k_plan_bound_no_miss_by_attempt[i] = true;
            ++k_plan_bound_no_miss_would_skip;
        }
        if (attempt.record_plan_exclusive_skip) {
            if (attempt.is_k) {
                ++k_plan_exclusive_skip;
            } else if (attempt.is_v) {
                ++v_plan_exclusive_skip;
            }
        }
        if (attempt.record_plan_route_complete) {
            if (attempt.is_k) {
                ++k_plan_route_complete;
            } else if (attempt.is_v) {
                ++v_plan_route_complete;
            }
        } else if (attempt.record_plan_ready) {
            if (attempt.is_k) {
                ++k_plan_route_incomplete;
            } else if (attempt.is_v) {
                ++v_plan_route_incomplete;
            }
        }
        if (attempt.record_live_window_after_highwater) {
            if (attempt.is_k) {
                ++k_live_after_highwater;
            } else if (attempt.is_v) {
                ++v_live_after_highwater;
            }
        }
        if (attempt.record_live_window_after_route_complete) {
            if (attempt.is_k) {
                ++k_live_after_route_complete;
            } else if (attempt.is_v) {
                ++v_live_after_route_complete;
            }
        }
        if (attempt.record_live_window_after_highwater && attempt.record_live_window_after_route_complete) {
            if (attempt.is_k) {
                ++k_live_after_both;
            } else if (attempt.is_v) {
                ++v_live_after_both;
            }
            if (attempt.record_plan_owned_write) {
                if (attempt.is_k) {
                    ++k_live_after_both_owned;
                } else if (attempt.is_v) {
                    ++v_live_after_both_owned;
                }
            }
            if (attempt.record_live_window_post_highwater) {
                if (attempt.is_k) {
                    ++k_live_after_both_post_highwater;
                } else if (attempt.is_v) {
                    ++v_live_after_both_post_highwater;
                }
            } else if (attempt.record_live_window_cross_highwater) {
                if (attempt.is_k) {
                    ++k_live_after_both_cross_highwater;
                } else if (attempt.is_v) {
                    ++v_live_after_both_cross_highwater;
                }
            } else {
                if (attempt.is_k) {
                    ++k_live_after_both_pre_highwater;
                } else if (attempt.is_v) {
                    ++v_live_after_both_pre_highwater;
                }
            }
        }
        const packed16_tail_page_owned_deferred_proof_reason reason =
            ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_base_reason_locked(attempt);
        const bool route_ready = reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK ||
            reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER ||
            reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BEFORE_ROUTE_HIGHWATER;
        if (route_ready) {
            if (attempt.is_k) {
                ++k_route_ready;
            } else if (attempt.is_v) {
                ++v_route_ready;
            }
            const uint64_t req_begin = attempt.logical_base;
            const uint64_t req_end = req_begin + attempt.n_tokens;
            const uint32_t live_highwater = s_packed16_tail_page_consumer_no_map_nk_max;
            if (live_highwater == 0) {
                if (attempt.is_k) {
                    ++k_no_highwater;
                } else if (attempt.is_v) {
                    ++v_no_highwater;
                }
            } else if (attempt.logical_base < live_highwater) {
                if (attempt.is_k) {
                    ++k_pre_highwater;
                } else if (attempt.is_v) {
                    ++v_pre_highwater;
                }
                if (req_end > live_highwater) {
                    const uint64_t post_tokens = req_end - live_highwater;
                    if (attempt.is_k) {
                        ++k_cross_highwater;
                        k_post_highwater_tokens += post_tokens;
                    } else if (attempt.is_v) {
                        ++v_cross_highwater;
                        v_post_highwater_tokens += post_tokens;
                    }
                }
            } else {
                if (attempt.is_k) {
                    ++k_post_highwater;
                    k_post_highwater_tokens += attempt.n_tokens;
                } else if (attempt.is_v) {
                    ++v_post_highwater;
                    v_post_highwater_tokens += attempt.n_tokens;
                }
            }
            const uint32_t record_highwater = attempt.record_live_highwater;
            if (record_highwater == 0) {
                if (attempt.is_k) {
                    ++k_record_no_highwater;
                } else if (attempt.is_v) {
                    ++v_record_no_highwater;
                }
                if (live_highwater != 0) {
                    if (attempt.is_k) {
                        ++k_record_no_to_live_highwater;
                    } else if (attempt.is_v) {
                        ++v_record_no_to_live_highwater;
                    }
                }
            } else if (req_begin < record_highwater) {
                if (attempt.is_k) {
                    ++k_record_pre_highwater;
                } else if (attempt.is_v) {
                    ++v_record_pre_highwater;
                }
                if (req_end > record_highwater) {
                    if (attempt.is_k) {
                        ++k_record_cross_highwater;
                    } else if (attempt.is_v) {
                        ++v_record_cross_highwater;
                    }
                }
            } else {
                if (attempt.is_k) {
                    ++k_record_post_highwater;
                } else if (attempt.is_v) {
                    ++v_record_post_highwater;
                }
            }
            if (!s_packed16_tail_page_retired_highwater.valid || s_packed16_tail_page_retired_highwater.nk == 0) {
                if (attempt.is_k) {
                    ++k_retired_no_highwater;
                } else if (attempt.is_v) {
                    ++v_retired_no_highwater;
                }
            } else if (attempt.logical_base < s_packed16_tail_page_retired_highwater.nk) {
                if (attempt.is_k) {
                    ++k_retired_pre_highwater;
                } else if (attempt.is_v) {
                    ++v_retired_pre_highwater;
                }
            } else {
                if (attempt.is_k) {
                    ++k_retired_post_highwater;
                } else if (attempt.is_v) {
                    ++v_retired_post_highwater;
                }
            }
        }
        if (attempt.record_live_window_after_highwater && attempt.record_live_window_after_route_complete &&
                attempt.record_live_window_post_highwater && reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK) {
            if (attempt.is_k) {
                ++k_live_base_ok;
            } else if (attempt.is_v) {
                ++v_live_base_ok;
            }
        }
        if (attempt.is_k && reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER &&
                s_packed16_tail_page_consumer_no_map_nk_max == 0) {
            k_all_bound_would_skip_by_attempt[i] = true;
            ++k_all_bound_would_skip;
        }
        if (attempt.is_k && reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK) {
            k_would_skip_by_attempt[i] = true;
            ++k_would_skip;
        } else if (!attempt.is_v) {
            ++reason_counts[reason];
        } else if (reason != PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK) {
            ++reason_counts[reason];
        }
    }
    for (size_t i = 0; i < s_packed16_tail_page_owned_deferred_proof.attempts.size(); ++i) {
        const packed16_tail_page_owned_deferred_proof_attempt & attempt = s_packed16_tail_page_owned_deferred_proof.attempts[i];
        if (!attempt.is_v) {
            continue;
        }
        const packed16_tail_page_owned_deferred_proof_reason reason =
            ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_base_reason_locked(attempt);
        bool k_seen = false;
        if (reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_OK) {
            for (size_t j = 0; j < s_packed16_tail_page_owned_deferred_proof.attempts.size(); ++j) {
                if (k_would_skip_by_attempt[j] && ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_attempt_same_range(
                        s_packed16_tail_page_owned_deferred_proof.attempts[j], attempt)) {
                    k_seen = true;
                    break;
                }
            }
            if (k_seen) {
                ++v_would_skip;
            } else {
                ++reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_K_DEFERRED_NOT_SEEN];
            }
        } else if (reason == PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER &&
                s_packed16_tail_page_consumer_no_map_nk_max == 0) {
            for (size_t j = 0; j < s_packed16_tail_page_owned_deferred_proof.attempts.size(); ++j) {
                if (k_all_bound_would_skip_by_attempt[j] && ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_attempt_same_range(
                        s_packed16_tail_page_owned_deferred_proof.attempts[j], attempt)) {
                    k_seen = true;
                    break;
                }
            }
            if (k_seen) {
                ++v_all_bound_would_skip;
            } else {
                ++v_all_bound_k_missing;
            }
        }
        const uint64_t plan_expected_layer_pages =
            uint64_t(attempt.record_plan_expected_layers) * uint64_t(attempt.record_plan_expected_pages);
        const bool plan_same_map_bound_at_resolve = attempt.record_plan_ready && plan_expected_layer_pages != 0 &&
            attempt.record_plan_before_record && attempt.record_plan_owned_write &&
            s_packed16_tail_page_consumer_no_map_nk_max == 0 &&
            ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(attempt.map) &&
            s_packed16_tail_page_route_complete.expected_layer_count == attempt.record_plan_expected_layers &&
            s_packed16_tail_page_route_complete.expected_page_count == attempt.record_plan_expected_pages &&
            s_packed16_tail_page_route_complete.expected_layer_page_count == plan_expected_layer_pages;
        if (plan_same_map_bound_at_resolve) {
            bool k_plan_seen = false;
            for (size_t j = 0; j < s_packed16_tail_page_owned_deferred_proof.attempts.size(); ++j) {
                if (k_plan_bound_no_miss_by_attempt[j] && ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_attempt_same_range(
                        s_packed16_tail_page_owned_deferred_proof.attempts[j], attempt)) {
                    k_plan_seen = true;
                    break;
                }
            }
            if (k_plan_seen) {
                ++v_plan_bound_no_miss_would_skip;
            } else {
                ++v_plan_bound_no_miss_k_missing;
            }
        }
    }
    ++s_packed16_tail_page_owned_deferred_proof.resolve_count;
    const bool same_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, s_packed16_tail_page_route_complete.map);
    const bool pair_ready = same_map && s_packed16_tail_page_owned_write_ready.k_ready && s_packed16_tail_page_owned_write_ready.v_ready;
    const bool route_complete = ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked();
    const ggml_cuda_mtp_qblock_tail_page_map_v1 & log_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) ?
        s_packed16_tail_page_route_complete.map : s_packed16_tail_page_owned_deferred_proof.attempts.front().map;
    if (deferred_proof_requested) {
        fprintf(stderr,
            "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_DEFERRED_PROOF: op=resolve trigger=%s status=%s attempts=%zu k_attempts=%llu v_attempts=%llu would_skip_k=%llu would_skip_v=%llu route_ready_k=%llu route_ready_v=%llu no_highwater_k=%llu no_highwater_v=%llu pre_highwater_k=%llu pre_highwater_v=%llu post_highwater_k=%llu post_highwater_v=%llu cross_highwater_k=%llu cross_highwater_v=%llu post_highwater_tokens_k=%llu post_highwater_tokens_v=%llu record_no_highwater_k=%llu record_no_highwater_v=%llu record_pre_highwater_k=%llu record_pre_highwater_v=%llu record_post_highwater_k=%llu record_post_highwater_v=%llu record_cross_highwater_k=%llu record_cross_highwater_v=%llu record_no_to_live_highwater_k=%llu record_no_to_live_highwater_v=%llu first_record_seq=%llu last_record_seq=%llu highwater_update_seq=%llu retired_valid=%d retired_highwater=%u retired_generation=%llu retired_reset_count=%llu retired_no_highwater_k=%llu retired_no_highwater_v=%llu retired_pre_highwater_k=%llu retired_pre_highwater_v=%llu retired_post_highwater_k=%llu retired_post_highwater_v=%llu pair_ready=%d route_complete=%d no_map_highwater=%u expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu dropped=%llu bad_kind=%llu bad_map=%llu not_owned_map=%llu physical_page_mismatch=%llu range_not_covered=%llu ready_map_mismatch=%llu kv_pair_not_ready=%llu route_incomplete=%llu no_route_highwater=%llu before_route_highwater=%llu k_deferred_not_seen=%llu resolve_count=%llu\n",
            trigger ? trigger : "unknown",
            (k_would_skip != 0 && v_would_skip != 0) ? "would_skip" : "blocked",
            s_packed16_tail_page_owned_deferred_proof.attempts.size(),
            (unsigned long long) k_attempts,
            (unsigned long long) v_attempts,
            (unsigned long long) k_would_skip,
            (unsigned long long) v_would_skip,
            (unsigned long long) k_route_ready,
            (unsigned long long) v_route_ready,
            (unsigned long long) k_no_highwater,
            (unsigned long long) v_no_highwater,
            (unsigned long long) k_pre_highwater,
            (unsigned long long) v_pre_highwater,
            (unsigned long long) k_post_highwater,
            (unsigned long long) v_post_highwater,
            (unsigned long long) k_cross_highwater,
            (unsigned long long) v_cross_highwater,
            (unsigned long long) k_post_highwater_tokens,
            (unsigned long long) v_post_highwater_tokens,
            (unsigned long long) k_record_no_highwater,
            (unsigned long long) v_record_no_highwater,
            (unsigned long long) k_record_pre_highwater,
            (unsigned long long) v_record_pre_highwater,
            (unsigned long long) k_record_post_highwater,
            (unsigned long long) v_record_post_highwater,
            (unsigned long long) k_record_cross_highwater,
            (unsigned long long) v_record_cross_highwater,
            (unsigned long long) k_record_no_to_live_highwater,
            (unsigned long long) v_record_no_to_live_highwater,
            (unsigned long long) first_record_event_seq,
            (unsigned long long) last_record_event_seq,
            (unsigned long long) s_packed16_tail_page_owned_deferred_proof_highwater_update_seq,
            s_packed16_tail_page_retired_highwater.valid ? 1 : 0,
            s_packed16_tail_page_retired_highwater.nk,
            (unsigned long long) s_packed16_tail_page_retired_highwater.generation,
            (unsigned long long) s_packed16_tail_page_retired_highwater.reset_count,
            (unsigned long long) k_retired_no_highwater,
            (unsigned long long) v_retired_no_highwater,
            (unsigned long long) k_retired_pre_highwater,
            (unsigned long long) v_retired_pre_highwater,
            (unsigned long long) k_retired_post_highwater,
            (unsigned long long) v_retired_post_highwater,
            pair_ready ? 1 : 0,
            route_complete ? 1 : 0,
            s_packed16_tail_page_consumer_no_map_nk_max,
            s_packed16_tail_page_route_complete.expected_layer_count,
            s_packed16_tail_page_route_complete.expected_page_count,
            s_packed16_tail_page_route_complete.bound_layer_pages.size(),
            log_map.logical_base_token,
            log_map.valid_tail_tokens,
            log_map.block_table[0],
            log_map.flags,
            (unsigned long long) log_map.generation,
            (unsigned long long) s_packed16_tail_page_owned_deferred_proof.dropped_attempts,
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_KIND],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BAD_MAP],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NOT_OWNED_MAP],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_PHYSICAL_PAGE_MISMATCH],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_RANGE_NOT_COVERED],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_READY_MAP_MISMATCH],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_KV_PAIR_NOT_READY],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_ROUTE_INCOMPLETE],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_NO_ROUTE_HIGHWATER],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_BEFORE_ROUTE_HIGHWATER],
            (unsigned long long) reason_counts[PACKED16_TAIL_PAGE_OWNED_DEFERRED_PROOF_K_DEFERRED_NOT_SEEN],
            (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
    }
    if (plan_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=resolve trigger=%s status=%s attempts=%zu k_attempts=%llu v_attempts=%llu plan_ready_k=%llu plan_ready_v=%llu plan_before_record_k=%llu plan_before_record_v=%llu plan_write_through_k=%llu plan_write_through_v=%llu plan_owned_write_k=%llu plan_owned_write_v=%llu plan_exclusive_skip_k=%llu plan_exclusive_skip_v=%llu plan_route_complete_k=%llu plan_route_complete_v=%llu plan_route_incomplete_k=%llu plan_route_incomplete_v=%llu plan_no_highwater_owned_k=%llu plan_no_highwater_owned_v=%llu plan_pre_highwater_owned_k=%llu plan_pre_highwater_owned_v=%llu plan_post_highwater_owned_k=%llu plan_post_highwater_owned_v=%llu plan_cross_highwater_owned_k=%llu plan_cross_highwater_owned_v=%llu plan_all_expected_known_k=%llu plan_all_expected_known_v=%llu plan_bound_before_record_k=%llu plan_bound_before_record_v=%llu plan_bound_at_resolve_k=%llu plan_bound_at_resolve_v=%llu plan_missing_layer_pages_at_record_k=%llu plan_missing_layer_pages_at_record_v=%llu plan_known_no_miss_k=%llu plan_known_no_miss_v=%llu plan_bound_no_miss_would_skip_k=%llu plan_bound_no_miss_would_skip_v=%llu plan_bound_no_miss_k_missing_v=%llu first_record_seq=%llu last_record_seq=%llu current_plan_id=%llu current_plan_seq=%llu current_plan_expected_layers=%u current_plan_expected_pages=%u current_plan_bound_layer_pages_at_plan=%llu pair_ready=%d route_complete=%d no_map_highwater=%u expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu resolve_count=%llu\n",
                trigger ? trigger : "unknown",
                (k_plan_ready != 0 && v_plan_ready != 0) ? "planned" : "blocked",
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) k_attempts,
                (unsigned long long) v_attempts,
                (unsigned long long) k_plan_ready,
                (unsigned long long) v_plan_ready,
                (unsigned long long) k_plan_before_record,
                (unsigned long long) v_plan_before_record,
                (unsigned long long) k_plan_write_through,
                (unsigned long long) v_plan_write_through,
                (unsigned long long) k_plan_owned_write,
                (unsigned long long) v_plan_owned_write,
                (unsigned long long) k_plan_exclusive_skip,
                (unsigned long long) v_plan_exclusive_skip,
                (unsigned long long) k_plan_route_complete,
                (unsigned long long) v_plan_route_complete,
                (unsigned long long) k_plan_route_incomplete,
                (unsigned long long) v_plan_route_incomplete,
                (unsigned long long) k_plan_no_highwater_owned,
                (unsigned long long) v_plan_no_highwater_owned,
                (unsigned long long) k_plan_pre_highwater_owned,
                (unsigned long long) v_plan_pre_highwater_owned,
                (unsigned long long) k_plan_post_highwater_owned,
                (unsigned long long) v_plan_post_highwater_owned,
                (unsigned long long) k_plan_cross_highwater_owned,
                (unsigned long long) v_plan_cross_highwater_owned,
                (unsigned long long) k_plan_all_expected_known,
                (unsigned long long) v_plan_all_expected_known,
                (unsigned long long) k_plan_bound_before_record,
                (unsigned long long) v_plan_bound_before_record,
                (unsigned long long) k_plan_bound_at_resolve,
                (unsigned long long) v_plan_bound_at_resolve,
                (unsigned long long) k_plan_missing_layer_pages_at_record,
                (unsigned long long) v_plan_missing_layer_pages_at_record,
                (unsigned long long) k_plan_known_no_miss,
                (unsigned long long) v_plan_known_no_miss,
                (unsigned long long) k_plan_bound_no_miss_would_skip,
                (unsigned long long) v_plan_bound_no_miss_would_skip,
                (unsigned long long) v_plan_bound_no_miss_k_missing,
                (unsigned long long) first_record_event_seq,
                (unsigned long long) last_record_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_id,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.plan_event_seq,
                s_packed16_tail_page_owned_write_plan.expected_layer_count,
                s_packed16_tail_page_owned_write_plan.expected_page_count,
                (unsigned long long) s_packed16_tail_page_owned_write_plan.bound_layer_pages_at_plan,
                pair_ready ? 1 : 0,
                route_complete ? 1 : 0,
                s_packed16_tail_page_consumer_no_map_nk_max,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                log_map.logical_base_token,
                log_map.valid_tail_tokens,
                log_map.block_table[0],
                log_map.flags,
                (unsigned long long) log_map.generation,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
    }
    if (live_window_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=resolve trigger=%s status=%s attempts=%zu k_attempts=%llu v_attempts=%llu would_skip_k=%llu would_skip_v=%llu after_highwater_k=%llu after_highwater_v=%llu after_route_complete_k=%llu after_route_complete_v=%llu after_both_k=%llu after_both_v=%llu after_both_owned_k=%llu after_both_owned_v=%llu after_both_pre_highwater_k=%llu after_both_pre_highwater_v=%llu after_both_post_highwater_k=%llu after_both_post_highwater_v=%llu after_both_cross_highwater_k=%llu after_both_cross_highwater_v=%llu live_base_ok_k=%llu live_base_ok_v=%llu first_record_seq=%llu last_record_seq=%llu highwater_seq=%llu highwater_update_seq=%llu highwater=%u highwater_generation=%llu route_complete_seq=%llu route_complete_bind_count=%llu route_complete_highwater=%u route_complete_generation=%llu current_highwater=%u pair_ready=%d route_complete=%d expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu resolve_count=%llu\n",
                trigger ? trigger : "unknown",
                (k_live_base_ok != 0 && v_live_base_ok != 0 && k_would_skip != 0 && v_would_skip != 0) ? "live_window_would_skip" :
                    ((k_live_after_both_post_highwater != 0 || v_live_after_both_post_highwater != 0) ? "live_window_candidate" : "blocked"),
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) k_attempts,
                (unsigned long long) v_attempts,
                (unsigned long long) k_would_skip,
                (unsigned long long) v_would_skip,
                (unsigned long long) k_live_after_highwater,
                (unsigned long long) v_live_after_highwater,
                (unsigned long long) k_live_after_route_complete,
                (unsigned long long) v_live_after_route_complete,
                (unsigned long long) k_live_after_both,
                (unsigned long long) v_live_after_both,
                (unsigned long long) k_live_after_both_owned,
                (unsigned long long) v_live_after_both_owned,
                (unsigned long long) k_live_after_both_pre_highwater,
                (unsigned long long) v_live_after_both_pre_highwater,
                (unsigned long long) k_live_after_both_post_highwater,
                (unsigned long long) v_live_after_both_post_highwater,
                (unsigned long long) k_live_after_both_cross_highwater,
                (unsigned long long) v_live_after_both_cross_highwater,
                (unsigned long long) k_live_base_ok,
                (unsigned long long) v_live_base_ok,
                (unsigned long long) first_record_event_seq,
                (unsigned long long) last_record_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_update_seq,
                s_packed16_tail_page_owned_live_window.highwater_nk,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_generation,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_bind_count,
                s_packed16_tail_page_owned_live_window.route_complete_highwater,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_generation,
                s_packed16_tail_page_consumer_no_map_nk_max,
                pair_ready ? 1 : 0,
                route_complete ? 1 : 0,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                log_map.logical_base_token,
                log_map.valid_tail_tokens,
                log_map.block_table[0],
                log_map.flags,
                (unsigned long long) log_map.generation,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=all_bound_resolve trigger=%s status=%s attempts=%zu k_attempts=%llu v_attempts=%llu all_bound_would_skip_k=%llu all_bound_would_skip_v=%llu all_bound_k_missing_v=%llu current_highwater=%u pair_ready=%d route_complete=%d expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu resolve_count=%llu\n",
                trigger ? trigger : "unknown",
                (k_all_bound_would_skip != 0 && v_all_bound_would_skip != 0) ? "all_bound_would_skip" : "blocked",
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) k_attempts,
                (unsigned long long) v_attempts,
                (unsigned long long) k_all_bound_would_skip,
                (unsigned long long) v_all_bound_would_skip,
                (unsigned long long) v_all_bound_k_missing,
                s_packed16_tail_page_consumer_no_map_nk_max,
                pair_ready ? 1 : 0,
                route_complete ? 1 : 0,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                log_map.logical_base_token,
                log_map.valid_tail_tokens,
                log_map.block_table[0],
                log_map.flags,
                (unsigned long long) log_map.generation,
                (unsigned long long) s_packed16_tail_page_owned_deferred_proof.resolve_count);
    }
}

static void ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_record_locked(
        const char * kind,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        uint32_t logical_base,
        uint32_t n_tokens,
        uint32_t physical_page) {
    const bool deferred_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested();
    const bool plan_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested();
    const bool live_window_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested();
    if (!ggml_cuda_mtp_qblock_owned_tail_write_any_proof_requested()) {
        return;
    }
    packed16_tail_page_owned_deferred_proof_attempt attempt = {};
    attempt.map = map;
    attempt.logical_base = logical_base;
    attempt.n_tokens = n_tokens;
    attempt.physical_page = physical_page;
    attempt.is_k = kind && strcmp(kind, "K") == 0;
    attempt.is_v = kind && strcmp(kind, "V") == 0;
    attempt.attempt_id = ++s_packed16_tail_page_owned_deferred_proof.next_attempt_id;
    attempt.record_event_seq = ++s_packed16_tail_page_owned_deferred_proof_event_seq;
    attempt.record_highwater_update_seq = s_packed16_tail_page_owned_deferred_proof_highwater_update_seq;
    attempt.record_resolve_count = s_packed16_tail_page_owned_deferred_proof.resolve_count;
    attempt.record_live_highwater = s_packed16_tail_page_consumer_no_map_nk_max;
    attempt.record_route_complete_same_map = ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(map);
    attempt.record_k_ready = s_packed16_tail_page_owned_write_ready.k_ready;
    attempt.record_v_ready = s_packed16_tail_page_owned_write_ready.v_ready;

    const bool plan_same_map = ggml_cuda_mtp_qblock_tail_page_owned_plan_same_map_locked(map);
    uint32_t plan_mapped_physical_page = 0;
    const bool plan_mapped = ggml_cuda_mtp_qblock_tail_page_map_translate(
            map, logical_base, nullptr, nullptr, &plan_mapped_physical_page, nullptr);
    const uint32_t page_tokens = map.page_tokens;
    attempt.record_plan_id = plan_same_map ? s_packed16_tail_page_owned_write_plan.plan_id : 0;
    attempt.record_plan_event_seq = plan_same_map ? s_packed16_tail_page_owned_write_plan.plan_event_seq : 0;
    attempt.record_plan_expected_layers = plan_same_map ? s_packed16_tail_page_owned_write_plan.expected_layer_count : 0;
    attempt.record_plan_expected_pages = plan_same_map ? s_packed16_tail_page_owned_write_plan.expected_page_count : 0;
    attempt.record_plan_bound_layer_pages = plan_same_map ? s_packed16_tail_page_route_complete.bound_layer_pages.size() : 0;
    attempt.record_plan_canonical_page_base = page_tokens != 0 ? (logical_base / page_tokens) * page_tokens : 0;
    attempt.record_plan_page_end = page_tokens != 0 ? attempt.record_plan_canonical_page_base + page_tokens : 0;
    attempt.record_plan_owned_page_base = page_tokens != 0 ? physical_page * page_tokens : 0;
    attempt.record_plan_overlay_page_base = attempt.record_plan_owned_page_base;
    attempt.record_plan_ready = plan_same_map;
    attempt.record_plan_before_record = attempt.record_plan_event_seq != 0 && attempt.record_plan_event_seq < attempt.record_event_seq;
    attempt.record_plan_route_complete = plan_same_map && attempt.record_route_complete_same_map;
    attempt.record_plan_owned_write = plan_same_map && plan_mapped && plan_mapped_physical_page == physical_page &&
        (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) != 0 &&
        (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) == 0;
    attempt.record_plan_write_through = attempt.record_plan_owned_write;
    attempt.record_plan_exclusive_skip = false;
    if (live_window_proof_requested) {
        attempt.record_live_window_highwater_event_seq = s_packed16_tail_page_owned_live_window.highwater_event_seq;
        attempt.record_live_window_highwater_update_seq = s_packed16_tail_page_owned_live_window.highwater_update_seq;
        attempt.record_live_window_route_complete_event_seq = s_packed16_tail_page_owned_live_window.route_complete_event_seq;
        attempt.record_live_window_highwater = s_packed16_tail_page_owned_live_window.highwater_nk;
        attempt.record_live_window_route_complete_same_map =
            s_packed16_tail_page_owned_live_window.route_complete_event_seq != 0 &&
            ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_live_window.route_complete_map) &&
            ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_live_window.route_complete_map, map);
        attempt.record_live_window_after_highwater =
            attempt.record_live_window_highwater_event_seq != 0 &&
            attempt.record_event_seq > attempt.record_live_window_highwater_event_seq;
        attempt.record_live_window_after_route_complete =
            attempt.record_live_window_route_complete_same_map &&
            attempt.record_live_window_route_complete_event_seq != 0 &&
            attempt.record_event_seq > attempt.record_live_window_route_complete_event_seq;
        const uint64_t req_begin = logical_base;
        const uint64_t req_end = req_begin + n_tokens;
        attempt.record_live_window_post_highwater =
            attempt.record_live_window_highwater != 0 && req_begin >= attempt.record_live_window_highwater;
        attempt.record_live_window_cross_highwater =
            attempt.record_live_window_highwater != 0 && req_begin < attempt.record_live_window_highwater &&
            req_end > attempt.record_live_window_highwater;
    }

    const uint64_t record_plan_expected_layer_pages =
        uint64_t(attempt.record_plan_expected_layers) * uint64_t(attempt.record_plan_expected_pages);
    const uint64_t record_plan_missing_layer_pages =
        attempt.record_plan_bound_layer_pages >= record_plan_expected_layer_pages ?
            0 : record_plan_expected_layer_pages - attempt.record_plan_bound_layer_pages;
    const bool record_plan_bound_before_record = attempt.record_plan_before_record &&
        record_plan_expected_layer_pages != 0 && record_plan_missing_layer_pages == 0;

    s_packed16_tail_page_owned_deferred_proof.attempts.push_back(attempt);
    static constexpr size_t MAX_OWNED_DEFERRED_PROOF_ATTEMPTS = 1024;
    if (s_packed16_tail_page_owned_deferred_proof.attempts.size() > MAX_OWNED_DEFERRED_PROOF_ATTEMPTS) {
        s_packed16_tail_page_owned_deferred_proof.attempts.erase(s_packed16_tail_page_owned_deferred_proof.attempts.begin());
        ++s_packed16_tail_page_owned_deferred_proof.dropped_attempts;
    }
    if (deferred_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_DEFERRED_PROOF: op=record kind=%s attempt_id=%llu record_seq=%llu attempts=%zu logical_base=%u valid_tail=%u req_end=%llu physical_page=%u record_highwater=%u record_highwater_update_seq=%llu record_resolve_count=%llu record_route_complete=%d record_k_ready=%d record_v_ready=%d table0=%d flags=0x%x generation=%llu\n",
                attempt.is_k ? "K" : (attempt.is_v ? "V" : "?"),
                (unsigned long long) attempt.attempt_id,
                (unsigned long long) attempt.record_event_seq,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                logical_base,
                n_tokens,
                (unsigned long long) ((uint64_t) logical_base + n_tokens),
                physical_page,
                attempt.record_live_highwater,
                (unsigned long long) attempt.record_highwater_update_seq,
                (unsigned long long) attempt.record_resolve_count,
                attempt.record_route_complete_same_map ? 1 : 0,
                attempt.record_k_ready ? 1 : 0,
                attempt.record_v_ready ? 1 : 0,
                map.block_table[0],
                map.flags,
                (unsigned long long) map.generation);
    }
    if (plan_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=record kind=%s attempt_id=%llu record_seq=%llu attempts=%zu logical_base=%u valid_tail=%u req_end=%llu physical_page=%u record_highwater=%u plan_id=%llu plan_seq=%llu plan_ready=%d plan_before_record=%d plan_expected_layers=%u plan_expected_pages=%u plan_expected_layer_pages=%llu plan_bound_layer_pages=%llu plan_missing_layer_pages=%llu plan_bound_before_record=%d plan_route_complete=%d plan_write_through=%d plan_owned_write=%d plan_exclusive_skip=%d canonical_page_base=%u page_end=%u owned_page_base=%u overlay_page_base=%u table0=%d flags=0x%x generation=%llu\n",
                attempt.is_k ? "K" : (attempt.is_v ? "V" : "?"),
                (unsigned long long) attempt.attempt_id,
                (unsigned long long) attempt.record_event_seq,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                logical_base,
                n_tokens,
                (unsigned long long) ((uint64_t) logical_base + n_tokens),
                physical_page,
                attempt.record_live_highwater,
                (unsigned long long) attempt.record_plan_id,
                (unsigned long long) attempt.record_plan_event_seq,
                attempt.record_plan_ready ? 1 : 0,
                attempt.record_plan_before_record ? 1 : 0,
                attempt.record_plan_expected_layers,
                attempt.record_plan_expected_pages,
                (unsigned long long) record_plan_expected_layer_pages,
                (unsigned long long) attempt.record_plan_bound_layer_pages,
                (unsigned long long) record_plan_missing_layer_pages,
                record_plan_bound_before_record ? 1 : 0,
                attempt.record_plan_route_complete ? 1 : 0,
                attempt.record_plan_write_through ? 1 : 0,
                attempt.record_plan_owned_write ? 1 : 0,
                attempt.record_plan_exclusive_skip ? 1 : 0,
                attempt.record_plan_canonical_page_base,
                attempt.record_plan_page_end,
                attempt.record_plan_owned_page_base,
                attempt.record_plan_overlay_page_base,
                map.block_table[0],
                map.flags,
                (unsigned long long) map.generation);
    }
    if (live_window_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=record kind=%s attempt_id=%llu record_seq=%llu attempts=%zu logical_base=%u valid_tail=%u req_end=%llu physical_page=%u highwater=%u highwater_seq=%llu highwater_update_seq=%llu route_complete_seq=%llu after_highwater=%d after_route_complete=%d route_complete_same_map=%d post_highwater=%d cross_highwater=%d plan_owned_write=%d plan_exclusive_skip=%d table0=%d flags=0x%x generation=%llu\n",
                attempt.is_k ? "K" : (attempt.is_v ? "V" : "?"),
                (unsigned long long) attempt.attempt_id,
                (unsigned long long) attempt.record_event_seq,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                logical_base,
                n_tokens,
                (unsigned long long) ((uint64_t) logical_base + n_tokens),
                physical_page,
                attempt.record_live_window_highwater,
                (unsigned long long) attempt.record_live_window_highwater_event_seq,
                (unsigned long long) attempt.record_live_window_highwater_update_seq,
                (unsigned long long) attempt.record_live_window_route_complete_event_seq,
                attempt.record_live_window_after_highwater ? 1 : 0,
                attempt.record_live_window_after_route_complete ? 1 : 0,
                attempt.record_live_window_route_complete_same_map ? 1 : 0,
                attempt.record_live_window_post_highwater ? 1 : 0,
                attempt.record_live_window_cross_highwater ? 1 : 0,
                attempt.record_plan_owned_write ? 1 : 0,
                attempt.record_plan_exclusive_skip ? 1 : 0,
                map.block_table[0],
                map.flags,
                (unsigned long long) map.generation);
    }
}

static void ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked(const char * reason) {
    const bool had_state = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) ||
        s_packed16_tail_page_owned_write_ready.k_ready || s_packed16_tail_page_owned_write_ready.v_ready;
    if (had_state) {
        ggml_cuda_mtp_qblock_full_page_map_restore_identity_locked(reason ? reason : "owned_ready_reset");
    }
    if (had_state && ggml_cuda_mtp_qblock_owned_tail_write_ready_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_READY: op=reset reason=%s k_ready=%d v_ready=%d pair_ready=%d logical_base=%u valid_tail=%u table0=%d generation=%llu\n",
                reason ? reason : "unknown",
                s_packed16_tail_page_owned_write_ready.k_ready ? 1 : 0,
                s_packed16_tail_page_owned_write_ready.v_ready ? 1 : 0,
                (s_packed16_tail_page_owned_write_ready.k_ready && s_packed16_tail_page_owned_write_ready.v_ready) ? 1 : 0,
                s_packed16_tail_page_owned_write_ready.map.logical_base_token,
                s_packed16_tail_page_owned_write_ready.map.valid_tail_tokens,
                s_packed16_tail_page_owned_write_ready.map.block_table[0],
                (unsigned long long) s_packed16_tail_page_owned_write_ready.map.generation);
    }
    ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_reset_locked(reason);
    s_packed16_tail_page_owned_write_ready = {};
}

static void ggml_cuda_mtp_qblock_tail_page_record_owned_write_ready(
        const char * kind,
        bool ready,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        uint32_t logical_base,
        uint32_t n_tokens,
        uint32_t physical_page,
        const char * reason) {
    const bool is_k = kind && strcmp(kind, "K") == 0;
    const bool is_v = kind && strcmp(kind, "V") == 0;
    uint32_t mapped_logical_page = 0;
    uint32_t mapped_slot = 0;
    uint32_t mapped_physical_page = 0;
    uint64_t mapped_physical_slot = 0;
    const bool mapped_page_ok = ggml_cuda_mtp_qblock_tail_page_map_translate(
        map, logical_base, &mapped_logical_page, &mapped_slot, &mapped_physical_page, &mapped_physical_slot);
    const bool map_ok = ready &&
        (is_k || is_v) &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(map) &&
        (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) != 0 &&
        (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) == 0 &&
        mapped_page_ok && mapped_physical_page == physical_page &&
        n_tokens > 0 && logical_base >= map.logical_base_token &&
        (uint64_t) logical_base + n_tokens <= (uint64_t) map.logical_base_token + map.valid_tail_tokens;

    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    if (!map_ok) {
        if (ggml_cuda_mtp_qblock_owned_tail_write_ready_trace_enabled()) {
            fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_READY: op=record kind=%s status=skip reason=%s logical_base=%u valid_tail=%u physical_page=%u map_valid=%d flags=0x%x table0=%d generation=%llu\n",
                    kind ? kind : "?",
                    reason ? reason : "not_ready",
                    logical_base,
                    n_tokens,
                    physical_page,
                    ggml_cuda_mtp_qblock_tail_page_map_valid(map) ? 1 : 0,
                    map.flags,
                    map.block_table[0],
                    (unsigned long long) map.generation);
        }
        return;
    }

    if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
            !ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, map)) {
        ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("owned_ready_new_map");
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map)) {
        s_packed16_tail_page_owned_write_ready.map = map;
        s_packed16_tail_page_owned_write_ready.physical_page = physical_page;
    }

    if (is_k) {
        s_packed16_tail_page_owned_write_ready.k_ready = true;
        ++s_packed16_tail_page_owned_write_ready.k_ready_count;
    } else {
        s_packed16_tail_page_owned_write_ready.v_ready = true;
        ++s_packed16_tail_page_owned_write_ready.v_ready_count;
    }
    ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_record_locked(
            is_k ? "K" : "V", map, logical_base, n_tokens, physical_page);
    const bool pair_ready = s_packed16_tail_page_owned_write_ready.k_ready &&
        s_packed16_tail_page_owned_write_ready.v_ready &&
        s_packed16_tail_page_owned_write_ready.physical_page == physical_page &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, map);
    size_t full_overlay_maps = 0;
    if (pair_ready) {
        ++s_packed16_tail_page_owned_write_ready.pair_ready_count;
        full_overlay_maps = ggml_cuda_mtp_qblock_full_page_map_apply_owned_tail_overlay_all_locked(map, "pair_ready");
    }
    if (ggml_cuda_mtp_qblock_owned_tail_write_ready_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_READY: op=record kind=%s status=ok reason=%s k_ready=%d v_ready=%d pair_ready=%d full_overlay_maps=%zu logical_base=%u valid_tail=%u physical_page=%u flags=0x%x generation=%llu k_ready_count=%llu v_ready_count=%llu pair_ready_count=%llu\n",
                is_k ? "K" : "V",
                reason ? reason : "ok",
                s_packed16_tail_page_owned_write_ready.k_ready ? 1 : 0,
                s_packed16_tail_page_owned_write_ready.v_ready ? 1 : 0,
                pair_ready ? 1 : 0,
                full_overlay_maps,
                map.logical_base_token,
                map.valid_tail_tokens,
                physical_page,
                map.flags,
                (unsigned long long) map.generation,
                (unsigned long long) s_packed16_tail_page_owned_write_ready.k_ready_count,
                (unsigned long long) s_packed16_tail_page_owned_write_ready.v_ready_count,
                (unsigned long long) s_packed16_tail_page_owned_write_ready.pair_ready_count);
    }
    if (pair_ready && (s_packed16_tail_page_consumer_no_map_nk_max != 0 ||
            ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(map))) {
        ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked("ready_record");
    }
}

static bool ggml_cuda_mtp_qblock_tail_page_owned_write_exclusive_allowed(
        const char * kind,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        uint32_t logical_base,
        uint32_t n_tokens,
        uint32_t physical_page) {
    const bool is_k = kind && strcmp(kind, "K") == 0;
    const bool is_v = kind && strcmp(kind, "V") == 0;
    const bool exclusive_requested = ggml_cuda_mtp_qblock_owned_tail_write_exclusive_requested();
    const bool trace = ggml_cuda_mtp_qblock_owned_tail_write_exclusive_trace_enabled();
    if (!exclusive_requested && !trace) {
        return false;
    }
    const char * reason = "ok";
    bool allowed = false;
    bool pair_ready = false;
    bool same_map = false;
    uint32_t no_map_highwater = 0;
    bool k_ready = false;
    bool v_ready = false;
    bool k_exclusive = false;
    bool v_exclusive = false;
    bool route_complete_same_map = false;
    const bool all_bound_env_requested = ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_env_requested();
    const bool all_bound_unsafe_requested = ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_unsafe_requested();
    const bool all_bound_exclusive_requested = ggml_cuda_mtp_qblock_owned_tail_write_exclusive_all_bound_requested();
    bool all_bound_no_miss = false;

    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    no_map_highwater = s_packed16_tail_page_consumer_no_map_nk_max;
    k_ready = s_packed16_tail_page_owned_write_ready.k_ready;
    v_ready = s_packed16_tail_page_owned_write_ready.v_ready;
    k_exclusive = s_packed16_tail_page_owned_write_ready.k_exclusive;
    v_exclusive = s_packed16_tail_page_owned_write_ready.v_exclusive;
    same_map = ggml_cuda_mtp_qblock_tail_page_map_valid(map) &&
        ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, map) &&
        s_packed16_tail_page_owned_write_ready.physical_page == physical_page;
    pair_ready = same_map && k_ready && v_ready;
    route_complete_same_map = ggml_cuda_mtp_qblock_tail_page_route_complete_same_map_locked(map);

    const uint64_t req_begin = logical_base;
    const uint64_t req_end = req_begin + n_tokens;
    const uint64_t map_begin = map.logical_base_token;
    const uint64_t map_end = map_begin + map.valid_tail_tokens;
    if (!exclusive_requested) {
        reason = "exclusive_env_disabled";
    } else if (!is_k && !is_v) {
        reason = "bad_kind";
    } else if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map)) {
        reason = "bad_map";
    } else if ((map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) == 0 ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0) {
        reason = "not_owned_map";
    } else {
        uint32_t mapped_logical_page = 0;
        uint32_t mapped_slot = 0;
        uint32_t mapped_physical_page = 0;
        uint64_t mapped_physical_slot = 0;
        if (!ggml_cuda_mtp_qblock_tail_page_map_translate(
                map, logical_base, &mapped_logical_page, &mapped_slot, &mapped_physical_page, &mapped_physical_slot) ||
                mapped_physical_page != physical_page) {
            reason = "physical_page_mismatch";
        } else if (n_tokens == 0 || req_end <= req_begin || req_begin < map_begin || req_end > map_end) {
            reason = "range_not_covered";
        } else if (!same_map) {
            reason = "ready_map_mismatch";
        } else if (!pair_ready) {
            reason = "kv_pair_not_ready";
        } else if (!route_complete_same_map) {
            reason = "route_incomplete";
        } else {
            all_bound_no_miss = all_bound_exclusive_requested && no_map_highwater == 0;
            if (all_bound_env_requested && !all_bound_unsafe_requested && no_map_highwater == 0) {
                reason = "all_bound_unsafe_gate_disabled";
            } else if (!all_bound_no_miss && no_map_highwater == 0) {
                reason = "no_route_highwater";
            } else if (!all_bound_no_miss && logical_base < no_map_highwater) {
                reason = "before_route_highwater";
            } else if (is_v && !s_packed16_tail_page_owned_write_ready.k_exclusive) {
                reason = "k_exclusive_not_seen";
            } else {
                allowed = true;
                reason = all_bound_no_miss ? "all_bound_no_miss" : "route_highwater";
                if (is_k) {
                    s_packed16_tail_page_owned_write_ready.k_exclusive = true;
                    ++s_packed16_tail_page_owned_write_ready.k_exclusive_count;
                    k_exclusive = true;
                } else {
                    s_packed16_tail_page_owned_write_ready.v_exclusive = true;
                    ++s_packed16_tail_page_owned_write_ready.v_exclusive_count;
                    v_exclusive = true;
                }
            }
        }
    }

    if (trace) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_EXCLUSIVE: kind=%s status=%s reason=%s logical_base=%u valid_tail=%u physical_page=%u no_map_highwater=%u all_bound_mode=%d all_bound_env=%d all_bound_unsafe=%d all_bound_no_miss=%d k_ready=%d v_ready=%d pair_ready=%d route_complete=%d k_exclusive=%d v_exclusive=%d flags=0x%x generation=%llu k_exclusive_count=%llu v_exclusive_count=%llu\n",
                is_k ? "K" : (is_v ? "V" : "?"),
                allowed ? "ok" : "skip",
                reason,
                logical_base,
                n_tokens,
                physical_page,
                no_map_highwater,
                all_bound_exclusive_requested ? 1 : 0,
                all_bound_env_requested ? 1 : 0,
                all_bound_unsafe_requested ? 1 : 0,
                all_bound_no_miss ? 1 : 0,
                k_ready ? 1 : 0,
                v_ready ? 1 : 0,
                pair_ready ? 1 : 0,
                route_complete_same_map ? 1 : 0,
                (s_packed16_tail_page_owned_write_ready.k_exclusive || k_exclusive) ? 1 : 0,
                (s_packed16_tail_page_owned_write_ready.v_exclusive || v_exclusive) ? 1 : 0,
                map.flags,
                (unsigned long long) map.generation,
                (unsigned long long) s_packed16_tail_page_owned_write_ready.k_exclusive_count,
                (unsigned long long) s_packed16_tail_page_owned_write_ready.v_exclusive_count);
    }
    return allowed;
}

static bool ggml_cuda_mtp_qblock_tail_page_entry_accepts(
        const packed16_registry_entry & entry,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map) {
    const uint64_t required_tokens = uint64_t(map.physical_pages) * uint64_t(map.page_tokens);
    return ggml_cuda_mtp_qblock_tail_page_map_valid(map) &&
        entry.payload && entry.scales &&
        entry.meta.k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 &&
        entry.meta.d == GGML_CUDA_PACKED16_K_TILE_D &&
        uint64_t(entry.meta.kv_capacity) >= required_tokens;
}

static bool ggml_cuda_mtp_qblock_tail_page_dispatch_trace_enabled() {
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        return true;
    }
    static const bool enabled = []() {
        const char * dispatch = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_DISPATCH_TRACE");
        const char * consumer = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_CONSUMER_TRACE");
        const char * proof = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PROOF");
        return (dispatch && atoi(dispatch) != 0) ||
            (consumer && atoi(consumer) != 0) ||
            (proof && atoi(proof) != 0);
    }();
    return enabled;
}

static bool ggml_cuda_mtp_qblock_tail_page_map_physical_page_for_token(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & map,
        const uint64_t token,
        int32_t * physical_page) {
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) || map.page_tokens == 0) {
        return false;
    }
    const uint64_t map_begin = map.logical_base_token;
    const uint64_t map_end = map_begin + map.valid_tail_tokens;
    if (token < map_begin || token >= map_end) {
        return false;
    }
    const uint64_t rel_page = (token - map_begin) / map.page_tokens;
    if (rel_page >= map.block_table_pages || rel_page >= GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES) {
        return false;
    }
    const int32_t page = map.block_table[rel_page];
    if (page < 0 || (uint32_t) page >= map.physical_pages) {
        return false;
    }
    if (physical_page) {
        *physical_page = page;
    }
    return true;
}

static bool ggml_cuda_mtp_qblock_tail_page_map_covers_same_physical_pages(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & bind_map,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & pending_map,
        const uint64_t req_begin,
        const uint64_t req_end) {
    if (req_end <= req_begin ||
            bind_map.page_tokens == 0 ||
            bind_map.page_tokens != pending_map.page_tokens ||
            (bind_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0 ||
            (pending_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0) {
        return false;
    }
    for (uint64_t token = req_begin; token < req_end;) {
        int32_t bind_page = -1;
        int32_t pending_page = -1;
        if (!ggml_cuda_mtp_qblock_tail_page_map_physical_page_for_token(bind_map, token, &bind_page) ||
                !ggml_cuda_mtp_qblock_tail_page_map_physical_page_for_token(pending_map, token, &pending_page) ||
                bind_page != pending_page) {
            return false;
        }
        const uint64_t next_page = ((token / bind_map.page_tokens) + 1u) * (uint64_t) bind_map.page_tokens;
        token = next_page > token ? next_page : token + 1u;
    }
    return true;
}

static void ggml_cuda_mtp_qblock_tail_page_resolve_pending_dispatch_binds_locked(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 & bind_map,
        const char * node_name,
        int layer,
        int graph_inst,
        int nk,
        uint64_t bind_count,
        bool trace) {
    if (s_packed16_tail_page_pending_dispatch_binds.empty()) {
        return;
    }
    for (auto it = s_packed16_tail_page_pending_dispatch_binds.begin(); it != s_packed16_tail_page_pending_dispatch_binds.end();) {
        const bool exact_match = ggml_cuda_mtp_qblock_tail_page_map_equal(bind_map, it->map) &&
            bind_map.logical_base_token <= it->req_begin &&
            (uint64_t) bind_map.logical_base_token + bind_map.valid_tail_tokens >= it->req_end;
        const bool cover_match = exact_match || ggml_cuda_mtp_qblock_tail_page_map_covers_same_physical_pages(
            bind_map, it->map, it->req_begin, it->req_end);
        if (!cover_match) {
            ++it;
            continue;
        }
        if (trace) {
            fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_DEFERRED_BIND: status=%s pending_id=%llu slot=%d node=%s layer=%d graph_inst=%d nk=%d req_begin=%llu req_end=%llu pending_logical_base=%u pending_valid_tail=%u pending_table0=%d pending_flags=0x%x pending_generation=%llu bind_logical_base=%u bind_valid_tail=%u bind_table0=%d bind_flags=0x%x bind_generation=%llu bind_count=%llu\n",
                    exact_match ? "resolved_exact" : "resolved_cover",
                    (unsigned long long) it->pending_id,
                    it->slot,
                    node_name ? node_name : "(null)",
                    layer,
                    graph_inst,
                    nk,
                    (unsigned long long) it->req_begin,
                    (unsigned long long) it->req_end,
                    it->map.logical_base_token,
                    it->map.valid_tail_tokens,
                    it->map.block_table[0],
                    it->map.flags,
                    (unsigned long long) it->map.generation,
                    bind_map.logical_base_token,
                    bind_map.valid_tail_tokens,
                    bind_map.block_table[0],
                    bind_map.flags,
                    (unsigned long long) bind_map.generation,
                    (unsigned long long) bind_count);
        }
        it = s_packed16_tail_page_pending_dispatch_binds.erase(it);
    }
}

void llama_kv_cache_note_mtp_qblock_tail_page_pending_dispatch_bind(
        const ggml_cuda_mtp_qblock_tail_page_map_v1 * map,
        uint64_t req_begin,
        uint64_t req_end,
        int slot) {
    if (!map || req_end <= req_begin ||
            !ggml_cuda_mtp_qblock_tail_page_map_valid(*map) ||
            (map->flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0) {
        return;
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_covers_same_physical_pages(*map, *map, req_begin, req_end)) {
        return;
    }
    const bool trace = ggml_cuda_mtp_qblock_tail_page_dispatch_trace_enabled();
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    for (const auto & pending : s_packed16_tail_page_pending_dispatch_binds) {
        if (pending.req_begin == req_begin && pending.req_end == req_end && pending.slot == slot &&
                ggml_cuda_mtp_qblock_tail_page_map_equal(pending.map, *map)) {
            return;
        }
    }
    packed16_tail_page_pending_dispatch_bind pending = {};
    pending.map = *map;
    pending.req_begin = req_begin;
    pending.req_end = req_end;
    pending.pending_id = ++s_packed16_tail_page_pending_dispatch_bind_count;
    pending.slot = slot;
    s_packed16_tail_page_pending_dispatch_binds.push_back(pending);
    static constexpr size_t MAX_PENDING_DISPATCH_BINDS = 128;
    if (s_packed16_tail_page_pending_dispatch_binds.size() > MAX_PENDING_DISPATCH_BINDS) {
        s_packed16_tail_page_pending_dispatch_binds.erase(s_packed16_tail_page_pending_dispatch_binds.begin());
    }
    if (trace) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_DEFERRED_BIND: status=pending pending_id=%llu slot=%d req_begin=%llu req_end=%llu pending_logical_base=%u pending_valid_tail=%u pending_table0=%d pending_flags=0x%x pending_generation=%llu pending_count=%zu\n",
                (unsigned long long) pending.pending_id,
                slot,
                (unsigned long long) req_begin,
                (unsigned long long) req_end,
                pending.map.logical_base_token,
                pending.map.valid_tail_tokens,
                pending.map.block_table[0],
                pending.map.flags,
                (unsigned long long) pending.map.generation,
                s_packed16_tail_page_pending_dispatch_binds.size());
    }
}

void llama_kv_cache_register_mtp_qblock_tail_page_map(const void * k_view_data, const ggml_cuda_mtp_qblock_tail_page_map_v1 * map) {
    if (!k_view_data) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it == s_packed16_registry.end()) {
        return;
    }
    packed16_registry_entry & entry = it->second;
    const bool sidecar_ok = entry.payload && entry.scales &&
        entry.meta.k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 &&
        entry.meta.d == GGML_CUDA_PACKED16_K_TILE_D &&
        entry.meta.kv_capacity >= (map ? map->physical_pages * map->page_tokens : 0u);
    const char * producer_map_env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PRODUCER_MAP");
    const char * paged_attention_env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
    const bool publish_global = (producer_map_env && *producer_map_env && atoi(producer_map_env) != 0) ||
        (paged_attention_env && *paged_attention_env && atoi(paged_attention_env) != 0);
    const bool map_valid = map && ggml_cuda_mtp_qblock_tail_page_map_valid(*map);
    if (map && sidecar_ok && map_valid) {
        if (publish_global) {
            if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
                    !ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, *map)) {
                ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("register_new_map");
            }
            if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
                    !ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_route_complete.map, *map)) {
                ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("register_new_map");
            }
            s_packed16_tail_page_published_map = *map;
            for (auto & kv : s_packed16_registry) {
                packed16_registry_entry & candidate = kv.second;
                if (ggml_cuda_mtp_qblock_tail_page_entry_accepts(candidate, *map)) {
                    candidate.tail_page_map = *map;
                }
            }
        } else {
            entry.tail_page_map = *map;
        }
    } else if (publish_global) {
        ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("register_invalid_global");
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("register_invalid_global");
        s_packed16_tail_page_published_map = {};
        for (auto & kv : s_packed16_registry) {
            kv.second.tail_page_map = {};
        }
    } else {
        const ggml_tensor * payload = entry.payload;
        const ggml_tensor * scales = entry.scales;
        ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("register_local_clear");
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("register_local_clear");
        entry.tail_page_map = {};
        if (payload && scales) {
            for (auto & kv : s_packed16_registry) {
                packed16_registry_entry & candidate = kv.second;
                if (candidate.payload == payload && candidate.scales == scales) {
                    candidate.tail_page_map = {};
                }
            }
        }
    }
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=register key=%p publish=%d sidecar_ok=%d map_valid=%d entry_active=%d global_active=%d registry_size=%zu logical_base=%u valid_tail=%u flags=0x%x table0=%d\n",
                k_view_data,
                publish_global ? 1 : 0,
                sidecar_ok ? 1 : 0,
                map_valid ? 1 : 0,
                ggml_cuda_mtp_qblock_tail_page_map_valid(entry.tail_page_map) ? 1 : 0,
                ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_published_map) ? 1 : 0,
                s_packed16_registry.size(),
                map ? map->logical_base_token : 0u,
                map ? map->valid_tail_tokens : 0u,
                map ? map->flags : 0u,
                map ? map->block_table[0] : 0);
    }
}

void llama_kv_cache_clear_mtp_qblock_tail_page_map(const void * k_view_data) {
    if (!k_view_data) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end()) {
        if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
            fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=clear key=%p global_active=%d dispatch_bind_count=%llu registry_size=%zu\n",
                    k_view_data,
                    ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_published_map) ? 1 : 0,
                    (unsigned long long) s_packed16_tail_page_dispatch_bind_count,
                    s_packed16_registry.size());
        }
        ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("clear_tail_page_map");
        ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("clear_tail_page_map");
        s_packed16_tail_page_published_map = {};
        s_packed16_tail_page_dispatch_bind_map = {};
        s_packed16_tail_page_dispatch_bind_count = 0;
        s_packed16_tail_page_pending_dispatch_binds.clear();
        for (auto & kv : s_packed16_registry) {
            kv.second.tail_page_map = {};
        }
    }
}

void llama_kv_cache_get_mtp_qblock_tail_page_map(const void * k_view_data, ggml_cuda_mtp_qblock_tail_page_map_v1 * map) {
    if (!map) {
        return;
    }
    *map = {};
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    const bool found = it != s_packed16_registry.end();
    if (found && ggml_cuda_mtp_qblock_tail_page_map_valid(it->second.tail_page_map)) {
        *map = it->second.tail_page_map;
        return;
    }
    if (found && ggml_cuda_mtp_qblock_tail_page_producer_snapshot_owned_proof_requested()) {
        if (ggml_cuda_mtp_qblock_tail_page_entry_accepts(it->second, s_packed16_tail_page_published_map)) {
            *map = s_packed16_tail_page_published_map;
            if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
                fprintf(stderr,
                        "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=get_global_fallback key=%p logical_base=%u valid_tail=%u flags=0x%x table0=%d\n",
                        k_view_data,
                        map->logical_base_token,
                        map->valid_tail_tokens,
                        map->flags,
                        map->block_table[0]);
            }
            return;
        }
        if (ggml_cuda_mtp_qblock_tail_page_entry_accepts(it->second, s_packed16_tail_page_producer_snapshot_map)) {
            *map = s_packed16_tail_page_producer_snapshot_map;
            if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
                fprintf(stderr,
                        "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=get_producer_snapshot_fallback key=%p logical_base=%u valid_tail=%u flags=0x%x table0=%d\n",
                        k_view_data,
                        map->logical_base_token,
                        map->valid_tail_tokens,
                        map->flags,
                        map->block_table[0]);
            }
            return;
        }
    }
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        const bool global_active = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_published_map);
        const bool global_accepted = found && ggml_cuda_mtp_qblock_tail_page_entry_accepts(it->second, s_packed16_tail_page_published_map);
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=get_miss key=%p found=%d global_active=%d global_accepted=%d registry_size=%zu k_format=%u d=%u capacity=%u no_map_nk_max=%u\n",
                k_view_data,
                found ? 1 : 0,
                global_active ? 1 : 0,
                global_accepted ? 1 : 0,
                s_packed16_registry.size(),
                found ? (unsigned) it->second.meta.k_format : 0u,
                found ? (unsigned) it->second.meta.d : 0u,
                found ? (unsigned) it->second.meta.kv_capacity : 0u,
                (unsigned) s_packed16_tail_page_consumer_no_map_nk_max);
    }
}

void llama_kv_cache_register_mtp_qblock_full_page_map_host(
        const void * k_view_data,
        const ggml_cuda_mtp_qblock_full_page_map_v1 * map,
        const int32_t * host_block_table) {
    if (!k_view_data) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    const bool map_basic_ok = map && map->active &&
        map->version == GGML_CUDA_MTP_QBLOCK_FULL_PAGE_MAP_VERSION &&
        map->abi_bytes == sizeof(ggml_cuda_mtp_qblock_full_page_map_v1) &&
        map->logical_base_token == 0 && map->valid_tokens != 0 &&
        map->page_tokens == GGML_CUDA_PACKED16_K_PAGE16_TOKENS &&
        map->physical_pages != 0 && map->block_table_pages != 0 &&
        host_block_table != nullptr &&
        map->non_identity_page_begin <= map->non_identity_page_end && map->non_identity_page_end <= map->block_table_pages;
    bool table_ok = map_basic_ok;
    if (table_ok) {
        const uint32_t required_pages = (map->valid_tokens + map->page_tokens - 1u) / map->page_tokens;
        table_ok = required_pages != 0 && required_pages <= map->block_table_pages && map->block_table_pages <= map->physical_pages;
        for (uint32_t i = 0; table_ok && i < required_pages; ++i) {
            table_ok = host_block_table[i] >= 0 && uint32_t(host_block_table[i]) < map->physical_pages;
        }
    }
    auto it = s_packed16_registry.find(k_view_data);
    auto clear_entry = [&]() {
        if (it != s_packed16_registry.end()) {
            it->second.full_page_map = {};
            it->second.full_page_map_base = {};
            it->second.full_page_table_host.clear();
        }
    };
    if (!map_basic_ok || !table_ok) {
        clear_entry();
        s_packed16_pending_full_page_maps.erase(k_view_data);
        return;
    }
    auto store_pending = [&]() {
        packed16_pending_full_page_map_entry pending;
        pending.map = *map;
        pending.map.block_table = nullptr;
        pending.host_block_table.assign(host_block_table, host_block_table + map->block_table_pages);
        s_packed16_pending_full_page_maps[k_view_data] = std::move(pending);
        if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
            fprintf(stderr,
                    "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=pending_full key=%p pages=%u valid_tokens=%u physical_pages=%u registry_found=%d\n",
                    k_view_data,
                    map->block_table_pages,
                    map->valid_tokens,
                    map->physical_pages,
                    it != s_packed16_registry.end() ? 1 : 0);
        }
    };
    if (it == s_packed16_registry.end()) {
        store_pending();
        return;
    }
    packed16_registry_entry & entry = it->second;
    const bool sidecar_ready = entry.payload && entry.scales &&
        entry.meta.k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272 &&
        entry.meta.d == GGML_CUDA_PACKED16_K_TILE_D &&
        entry.meta.kv_capacity >= map->physical_pages * map->page_tokens;
    if (!sidecar_ready) {
        clear_entry();
        store_pending();
        return;
    }
    if (!ggml_cuda_mtp_qblock_full_page_map_install_locked(entry, *map, host_block_table, "register")) {
        clear_entry();
        s_packed16_pending_full_page_maps.erase(k_view_data);
        return;
    }
    s_packed16_pending_full_page_maps.erase(k_view_data);
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=register key=%p pages=%u valid_tokens=%u physical_pages=%u table_ptr=%p first_pages=[%d,%d,%d,%d] generation=%llu\n",
                k_view_data,
                entry.full_page_map.block_table_pages,
                entry.full_page_map.valid_tokens,
                entry.full_page_map.physical_pages,
                (const void *) entry.full_page_map.block_table,
                entry.full_page_map.debug_first_pages[0],
                entry.full_page_map.debug_first_pages[1],
                entry.full_page_map.debug_first_pages[2],
                entry.full_page_map.debug_first_pages[3],
                (unsigned long long) entry.full_page_map.generation);
    }
}

void llama_kv_cache_clear_mtp_qblock_full_page_map(const void * k_view_data) {
    if (!k_view_data) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    s_packed16_pending_full_page_maps.erase(k_view_data);
    if (it != s_packed16_registry.end()) {
        if (ggml_cuda_mtp_qblock_full_page_map_valid(s_packed16_full_page_published_map) &&
                ggml_cuda_mtp_qblock_full_page_map_published_sidecar_matches_locked(it->second)) {
            ggml_cuda_mtp_qblock_full_page_map_clear_published_locked();
        }
        it->second.full_page_map = {};
        it->second.full_page_map_base = {};
        it->second.full_page_table_host.clear();
        ggml_cuda_mtp_qblock_full_page_map_refresh_published_locked();
    }
}

void llama_kv_cache_get_mtp_qblock_full_page_map(const void * k_view_data, ggml_cuda_mtp_qblock_full_page_map_v1 * map) {
    if (!map) {
        return;
    }
    *map = {};
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    if (it != s_packed16_registry.end() && ggml_cuda_mtp_qblock_full_page_map_valid(it->second.full_page_map)) {
        *map = it->second.full_page_map;
        return;
    }
    if (it != s_packed16_registry.end() && ggml_cuda_mtp_qblock_full_page_map_valid(s_packed16_full_page_published_map) &&
            ggml_cuda_mtp_qblock_full_page_map_published_sidecar_matches_locked(it->second)) {
        *map = s_packed16_full_page_published_map;
        if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
            fprintf(stderr,
                    "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=get_published_fallback key=%p found=%d valid_tokens=%u pages=%u table_ptr=%p flags=0x%x sidecar_match=1\n",
                    k_view_data,
                    it != s_packed16_registry.end() ? 1 : 0,
                    map->valid_tokens,
                    map->block_table_pages,
                    (const void *) map->block_table,
                    map->flags);
        }
    } else if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_FULL_PAGE_MAP_REGISTRY: op=get_miss key=%p found=%d published=%d registry_size=%zu\n",
                k_view_data,
                it != s_packed16_registry.end() ? 1 : 0,
                ggml_cuda_mtp_qblock_full_page_map_valid(s_packed16_full_page_published_map) ? 1 : 0,
                s_packed16_registry.size());
    }
}

bool llama_kv_cache_get_mtp_qblock_tail_page_published_map(ggml_cuda_mtp_qblock_tail_page_map_v1 * map) {
    if (!map) {
        return false;
    }
    *map = {};
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    if (ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_published_map)) {
        *map = s_packed16_tail_page_published_map;
        return true;
    }
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_producer_snapshot_map)) {
        return false;
    }
    *map = s_packed16_tail_page_producer_snapshot_map;
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=producer_snapshot_get logical_base=%u valid_tail=%u flags=0x%x table0=%d\n",
                map->logical_base_token,
                map->valid_tail_tokens,
                map->flags,
                map->block_table[0]);
    }
    return true;
}

void llama_kv_cache_record_mtp_qblock_tail_page_dispatch_bind(
        const void * k_view_data,
        const ggml_cuda_mtp_qblock_tail_page_map_v1 * map,
        const char * node_name,
        int layer,
        int graph_inst,
        int nk) {
    if (!k_view_data || !map) {
        return;
    }
    const bool trace = ggml_cuda_mtp_qblock_tail_page_dispatch_trace_enabled();

    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    auto it = s_packed16_registry.find(k_view_data);
    const bool found = it != s_packed16_registry.end();
    const bool map_valid = ggml_cuda_mtp_qblock_tail_page_map_valid(*map);
    const bool entry_accepts = found && ggml_cuda_mtp_qblock_tail_page_entry_accepts(it->second, *map);
    const bool entry_match = found && ggml_cuda_mtp_qblock_tail_page_map_valid(it->second.tail_page_map) &&
        ggml_cuda_mtp_qblock_tail_page_map_equal(it->second.tail_page_map, *map);
    const bool record_ok = map_valid && entry_accepts && entry_match;
    if (record_ok) {
        s_packed16_tail_page_dispatch_bind_map = *map;
        ++s_packed16_tail_page_dispatch_bind_count;
        s_packed16_tail_page_last_dispatch_bind = {};
        s_packed16_tail_page_last_dispatch_bind.version = GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_VERSION;
        s_packed16_tail_page_last_dispatch_bind.abi_bytes = sizeof(ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1);
        s_packed16_tail_page_last_dispatch_bind.active = 1;
        s_packed16_tail_page_last_dispatch_bind.map = *map;
        s_packed16_tail_page_last_dispatch_bind.bind_count = s_packed16_tail_page_dispatch_bind_count;
        s_packed16_tail_page_last_dispatch_bind.layer = layer;
        s_packed16_tail_page_last_dispatch_bind.graph_inst = graph_inst;
        s_packed16_tail_page_last_dispatch_bind.nk = nk;
        snprintf(s_packed16_tail_page_last_dispatch_bind.node_name,
                sizeof(s_packed16_tail_page_last_dispatch_bind.node_name),
                "%s", node_name ? node_name : "(null)");
        ggml_cuda_mtp_qblock_tail_page_resolve_pending_dispatch_binds_locked(
                *map,
                node_name,
                layer,
                graph_inst,
                nk,
                s_packed16_tail_page_dispatch_bind_count,
                trace);
        ggml_cuda_mtp_qblock_tail_page_route_complete_record_bind_locked(*map, layer, nk, node_name);
        ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked("dispatch_bind");
    }
    if (trace) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_DISPATCH_BIND: tail_dispatch_bound=%d key=%p node=%s layer=%d graph_inst=%d nk=%d logical_base=%u valid_tail=%u page_tokens=%u physical_pages=%u table_pages=%u table0=%d flags=0x%x generation=%llu bind_count=%llu found=%d map_valid=%d entry_accepts=%d entry_match=%d registry_size=%zu\n",
                record_ok ? 1 : 0,
                k_view_data,
                node_name ? node_name : "(null)",
                layer,
                graph_inst,
                nk,
                map->logical_base_token,
                map->valid_tail_tokens,
                map->page_tokens,
                map->physical_pages,
                map->block_table_pages,
                map->block_table[0],
                map->flags,
                (unsigned long long) map->generation,
                (unsigned long long) s_packed16_tail_page_dispatch_bind_count,
                found ? 1 : 0,
                map_valid ? 1 : 0,
                entry_accepts ? 1 : 0,
                entry_match ? 1 : 0,
                s_packed16_registry.size());
    }
}

bool llama_kv_cache_get_mtp_qblock_tail_page_last_dispatch_bind(
        ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1 * out) {
    if (!out) {
        return false;
    }
    *out = {};
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    if (!s_packed16_tail_page_last_dispatch_bind.active ||
            s_packed16_tail_page_last_dispatch_bind.version != GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_DISPATCH_BIND_VERSION ||
            s_packed16_tail_page_last_dispatch_bind.abi_bytes != sizeof(ggml_cuda_mtp_qblock_tail_page_dispatch_bind_v1) ||
            s_packed16_tail_page_last_dispatch_bind.bind_count == 0 ||
            !ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_last_dispatch_bind.map)) {
        return false;
    }
    *out = s_packed16_tail_page_last_dispatch_bind;
    return true;
}

void llama_kv_cache_record_mtp_qblock_tail_page_consumer_miss(uint32_t nk, const char * reason) {
    if (nk == 0 || reason == nullptr || strcmp(reason, "no_registered_map") != 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    if (nk <= s_packed16_tail_page_consumer_no_map_nk_max) {
        return;
    }
    const uint32_t old_nk_max = s_packed16_tail_page_consumer_no_map_nk_max;
    s_packed16_tail_page_consumer_no_map_nk_max = nk;
    const bool any_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_any_proof_requested();
    const bool deferred_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested();
    const bool live_window_proof_requested = ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested();
    uint64_t event_seq = 0;
    uint64_t highwater_update_seq = s_packed16_tail_page_owned_deferred_proof_highwater_update_seq;
    if (any_proof_requested) {
        event_seq = ++s_packed16_tail_page_owned_deferred_proof_event_seq;
        highwater_update_seq = ++s_packed16_tail_page_owned_deferred_proof_highwater_update_seq;
    }
    if (live_window_proof_requested) {
        s_packed16_tail_page_owned_live_window.highwater_event_seq = event_seq;
        s_packed16_tail_page_owned_live_window.highwater_update_seq = highwater_update_seq;
        s_packed16_tail_page_owned_live_window.highwater_nk = nk;
        s_packed16_tail_page_owned_live_window.highwater_generation = s_packed16_generation;
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=highwater event_seq=%llu highwater_update_seq=%llu old_nk_max=%u new_nk_max=%u reason=%s pending_attempts=%zu route_complete_seq=%llu route_complete_highwater=%u generation=%llu\n",
                (unsigned long long) event_seq,
                (unsigned long long) highwater_update_seq,
                old_nk_max,
                nk,
                reason,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_event_seq,
                s_packed16_tail_page_owned_live_window.route_complete_highwater,
                (unsigned long long) s_packed16_generation);
    }
    if (deferred_proof_requested) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_DEFERRED_PROOF: op=highwater_update event_seq=%llu highwater_update_seq=%llu old_nk_max=%u new_nk_max=%u reason=%s pending_attempts=%zu generation=%llu\n",
                (unsigned long long) event_seq,
                (unsigned long long) highwater_update_seq,
                old_nk_max,
                nk,
                reason,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                (unsigned long long) s_packed16_generation);
    }
    ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked("consumer_highwater");
    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=consumer_no_map_highwater old_nk_max=%u new_nk_max=%u reason=%s\n",
                old_nk_max,
                nk,
                reason);
    }
}

uint32_t llama_kv_cache_get_mtp_qblock_tail_page_consumer_no_map_nk_max(void) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    return s_packed16_tail_page_consumer_no_map_nk_max;
}

void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle(bool data_invalidates) {
    llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle_preserve_snapshot(data_invalidates, false);
}

void llama_kv_cache_reset_mtp_qblock_tail_page_lifecycle_preserve_snapshot(bool data_invalidates, bool keep_producer_snapshot) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    const bool had_global_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_published_map);
    const bool had_snapshot_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_producer_snapshot_map);
    const uint64_t old_dispatch_bind_count = s_packed16_tail_page_dispatch_bind_count;
    const bool global_scratch = had_global_map &&
        (s_packed16_tail_page_published_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0;
    const bool global_owned = had_global_map &&
        (s_packed16_tail_page_published_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) != 0 &&
        (s_packed16_tail_page_published_map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) == 0;
    const bool preserve_owned_snapshot = global_owned &&
        ggml_cuda_mtp_qblock_tail_page_producer_snapshot_owned_proof_requested();
    const uint32_t old_no_map_nk_max = s_packed16_tail_page_consumer_no_map_nk_max;
    size_t active_entries = 0;
    for (const auto & kv : s_packed16_registry) {
        if (ggml_cuda_mtp_qblock_tail_page_map_valid(kv.second.tail_page_map)) {
            ++active_entries;
        }
    }

    bool snapshot_saved = false;
    if (data_invalidates && keep_producer_snapshot && (global_scratch || preserve_owned_snapshot)) {
        s_packed16_tail_page_producer_snapshot_map = s_packed16_tail_page_published_map;
        snapshot_saved = true;
    } else if (data_invalidates && !keep_producer_snapshot) {
        s_packed16_tail_page_producer_snapshot_map = {};
    }

    if (ggml_cuda_mtp_qblock_owned_tail_write_live_window_proof_requested()) {
        ++s_packed16_tail_page_owned_live_window.reset_count;
        const bool same_map = ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_owned_write_ready.map) &&
            ggml_cuda_mtp_qblock_tail_page_map_valid(s_packed16_tail_page_route_complete.map) &&
            ggml_cuda_mtp_qblock_tail_page_map_equal(s_packed16_tail_page_owned_write_ready.map, s_packed16_tail_page_route_complete.map);
        const bool pair_ready = same_map && s_packed16_tail_page_owned_write_ready.k_ready && s_packed16_tail_page_owned_write_ready.v_ready;
        const bool route_complete = ggml_cuda_mtp_qblock_tail_page_route_complete_now_locked();
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_LIVE_WINDOW_PROOF: op=lifecycle_reset data_invalidates=%d keep_snapshot=%d preserve_owned_snapshot=%d snapshot_saved=%d attempts=%zu pair_ready=%d route_complete=%d old_highwater=%u highwater_seq=%llu highwater_update_seq=%llu route_complete_seq=%llu route_complete_highwater=%u expected_layers=%u expected_pages=%u bound_layer_pages=%zu logical_base=%u valid_tail=%u table0=%d flags=0x%x generation=%llu reset_count=%llu\n",
                data_invalidates ? 1 : 0,
                keep_producer_snapshot ? 1 : 0,
                preserve_owned_snapshot ? 1 : 0,
                snapshot_saved ? 1 : 0,
                s_packed16_tail_page_owned_deferred_proof.attempts.size(),
                pair_ready ? 1 : 0,
                route_complete ? 1 : 0,
                old_no_map_nk_max,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_event_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.highwater_update_seq,
                (unsigned long long) s_packed16_tail_page_owned_live_window.route_complete_event_seq,
                s_packed16_tail_page_owned_live_window.route_complete_highwater,
                s_packed16_tail_page_route_complete.expected_layer_count,
                s_packed16_tail_page_route_complete.expected_page_count,
                s_packed16_tail_page_route_complete.bound_layer_pages.size(),
                s_packed16_tail_page_route_complete.map.logical_base_token,
                s_packed16_tail_page_route_complete.map.valid_tail_tokens,
                s_packed16_tail_page_route_complete.map.block_table[0],
                s_packed16_tail_page_route_complete.map.flags,
                (unsigned long long) s_packed16_generation,
                (unsigned long long) s_packed16_tail_page_owned_live_window.reset_count);
    }

    // Proof-only: resolve once while live ready/route/highwater state is still intact.
    // This distinguishes current-boundary ordering from historical retired-highwater evidence.
    ggml_cuda_mtp_qblock_tail_page_owned_deferred_proof_resolve_locked("pre_lifecycle_reset_current_highwater");

    ggml_cuda_mtp_qblock_tail_page_owned_write_ready_reset_locked("lifecycle_reset");
    ggml_cuda_mtp_qblock_tail_page_route_complete_reset_locked("lifecycle_reset");
    s_packed16_tail_page_published_map = {};
    s_packed16_tail_page_dispatch_bind_map = {};
    s_packed16_tail_page_dispatch_bind_count = 0;
    s_packed16_tail_page_pending_dispatch_binds.clear();
    for (auto & kv : s_packed16_registry) {
        kv.second.tail_page_map = {};
    }
    if (data_invalidates) {
        ggml_cuda_mtp_qblock_full_page_map_clear_published_locked();
        s_packed16_pending_full_page_maps.clear();
        for (auto & kv : s_packed16_registry) {
            kv.second.full_page_map = {};
            kv.second.full_page_map_base = {};
            kv.second.full_page_table_host.clear();
        }
        s_packed16_tail_page_consumer_no_map_nk_max = 0;
        if (old_no_map_nk_max != 0) {
            s_packed16_tail_page_retired_highwater.valid = true;
            s_packed16_tail_page_retired_highwater.nk = old_no_map_nk_max;
            s_packed16_tail_page_retired_highwater.generation = s_packed16_generation;
            ++s_packed16_tail_page_retired_highwater.reset_count;
        }
        if (old_no_map_nk_max != 0 && ggml_cuda_mtp_qblock_owned_tail_write_deferred_proof_requested()) {
            fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_DEFERRED_PROOF: op=highwater_reset reason=lifecycle_reset data_invalidates=1 old_nk_max=%u new_nk_max=0 retired_valid=%d retired_highwater=%u retired_generation=%llu retired_reset_count=%llu generation=%llu\n",
                    old_no_map_nk_max,
                    s_packed16_tail_page_retired_highwater.valid ? 1 : 0,
                    s_packed16_tail_page_retired_highwater.nk,
                    (unsigned long long) s_packed16_tail_page_retired_highwater.generation,
                    (unsigned long long) s_packed16_tail_page_retired_highwater.reset_count,
                    (unsigned long long) s_packed16_generation);
        }
        s_packed16_tail_page_owned_live_window.highwater_event_seq = 0;
        s_packed16_tail_page_owned_live_window.highwater_update_seq = 0;
        s_packed16_tail_page_owned_live_window.highwater_nk = 0;
        s_packed16_tail_page_owned_live_window.highwater_generation = 0;
    }
    ++s_packed16_generation;

    if (ggml_cuda_mtp_qblock_tail_page_registry_trace_enabled()) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_REGISTRY: op=lifecycle_reset data_invalidates=%d keep_snapshot=%d snapshot_saved=%d old_global_active=%d old_snapshot_active=%d old_active_entries=%zu old_dispatch_bind_count=%llu old_no_map_nk_max=%u new_no_map_nk_max=%u generation=%llu registry_size=%zu\n",
                data_invalidates ? 1 : 0,
                keep_producer_snapshot ? 1 : 0,
                snapshot_saved ? 1 : 0,
                had_global_map ? 1 : 0,
                had_snapshot_map ? 1 : 0,
                active_entries,
                (unsigned long long) old_dispatch_bind_count,
                old_no_map_nk_max,
                s_packed16_tail_page_consumer_no_map_nk_max,
                (unsigned long long) s_packed16_generation,
                s_packed16_registry.size());
    }
}

bool llama_kv_cache_mtp_qblock_tail_page_published_scratch_map_covers(uint32_t logical_base, uint32_t n_tokens) {
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    const ggml_cuda_mtp_qblock_tail_page_map_v1 & map = s_packed16_tail_page_published_map;
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) == 0 ||
            n_tokens == 0) {
        return false;
    }
    const uint64_t req_begin = logical_base;
    const uint64_t req_end = req_begin + n_tokens;
    const uint64_t map_begin = map.logical_base_token;
    const uint64_t map_end = map_begin + map.valid_tail_tokens;
    return req_begin >= map_begin && req_end <= map_end;
}

bool llama_kv_cache_mtp_qblock_tail_page_published_owned_map_covers(uint32_t logical_base, uint32_t n_tokens, uint32_t * physical_page, ggml_cuda_mtp_qblock_tail_page_map_v1 * out_map) {
    if (physical_page != nullptr) {
        *physical_page = 0;
    }
    if (out_map != nullptr) {
        *out_map = {};
    }
    std::lock_guard<std::mutex> lock(s_packed16_mutex);
    const ggml_cuda_mtp_qblock_tail_page_map_v1 & map = s_packed16_tail_page_published_map;
    if (!ggml_cuda_mtp_qblock_tail_page_map_valid(map) ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE) == 0 ||
            (map.flags & GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY) != 0 ||
            n_tokens == 0 || map.page_tokens == 0 || map.block_table_pages == 0) {
        return false;
    }
    const uint64_t req_begin = logical_base;
    const uint64_t req_end = req_begin + n_tokens;
    const uint64_t map_begin = map.logical_base_token;
    const uint64_t map_end = map_begin + map.valid_tail_tokens;
    if (req_end <= req_begin || req_begin < map_begin || req_end > map_end) {
        return false;
    }
    const uint64_t rel_page = (req_begin - map_begin) / map.page_tokens;
    if (rel_page >= map.block_table_pages || rel_page >= GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES || map.block_table[rel_page] < 0) {
        return false;
    }
    if (physical_page != nullptr) {
        *physical_page = (uint32_t) map.block_table[rel_page];
    }
    if (out_map != nullptr) {
        *out_map = map;
    }
    return true;
}

void llama_kv_cache_register_v4_k16d16(const void * v_view_data, ggml_tensor * v_cache, ggml_tensor * v_tail) {
    std::lock_guard<std::mutex> lock(s_v4_k16d16_mutex);
    s_v4_k16d16_registry[v_view_data] = {v_cache, v_tail};
}

void llama_kv_cache_get_v4_k16d16_tensors(const void * v_view_data, ggml_tensor ** v_cache, ggml_tensor ** v_tail) {
    std::lock_guard<std::mutex> lock(s_v4_k16d16_mutex);
    auto it = s_v4_k16d16_registry.find(v_view_data);
    if (it != s_v4_k16d16_registry.end()) {
        *v_cache = it->second.v_cache;
        *v_tail  = it->second.v_tail;
    } else {
        *v_cache = nullptr;
        *v_tail  = nullptr;
    }
}
}

#ifdef GGML_USE_HIP

static constexpr int GGML_CUDA_Q8K_DOT4_KQ_D = 256;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_BLOCKS = GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0;

static constexpr int GGML_CUDA_Q8K_DOT4_KQ_PACKED16_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 16;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_M = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_TILE_N = 16;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_PAR_TILE_K = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_MAX_GQA = 8;
static constexpr int GGML_CUDA_Q8K_DOT4_KQ_GQA6 = 6;

static inline bool ggml_cuda_q8k_dot4_v4_144_pv4_standard_enabled() {
    const char * disable_v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE");
    if (disable_v && atoi(disable_v) != 0) {
        return false;
    }
    const char * legacy_pv4 = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4");
    if (legacy_pv4 && *legacy_pv4) {
        return atoi(legacy_pv4) != 0;
    }
    return true;
}

static inline bool ggml_cuda_q8k_dot4_v4_144_dp16_dot_profile_enabled() {
    const char * disable_v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE");
    if (disable_v && atoi(disable_v) != 0) {
        return false;
    }
    const char * profile = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PROFILE");
    return profile && *profile && atoi(profile) != 0;
}

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled(const char * name) {
    const char * env = getenv(name);
    if (env && *env) {
        return atoi(env) != 0;
    }
    if (strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PROFILE") == 0) {
        return ggml_cuda_q8k_dot4_v4_144_dp16_dot_profile_enabled();
    }
    if (strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PV4") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_DECODE_SPLITK") == 0) {
        return ggml_cuda_q8k_dot4_v4_144_pv4_standard_enabled();
    }
    return false;
}

static inline int ggml_cuda_q8k_dot4_kq_env_int(const char * name, int fallback) {
    const char * env = getenv(name);
    return env ? atoi(env) : fallback;
}

static inline bool ggml_cuda_q8k_dot4_decode_stage_auto_enabled() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO");
}

static inline int ggml_cuda_q8k_dot4_decode_stage_threshold(const int fallback) {
    if (!ggml_cuda_q8k_dot4_decode_stage_auto_enabled()) {
        return fallback;
    }
    const int configured = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO_MIN_NK", 4096);
    return configured > 0 ? configured : fallback;
}

static inline int ggml_cuda_q8k_dot4_decode_stage_split_size(const int nk, const int fallback) {
    const char * explicit_size = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE");
    if (explicit_size && *explicit_size) {
        const int requested = atoi(explicit_size);
        return requested > 0 ? requested : fallback;
    }
    if (!ggml_cuda_q8k_dot4_decode_stage_auto_enabled()) {
        return fallback;
    }

    // Default-off long-context decode policy.  Keep existing kernels and merge
    // ABI; only coarsen K shards as nk grows so generated-token decode stops
    // launching dozens of 512-token shards at 32K+ contexts by default.
    if (nk >= 32768) {
        const int requested = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO_SPLIT_SIZE_NK32768", 2048);
        return requested > 0 ? requested : fallback;
    }
    if (nk >= 16384) {
        const int requested = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO_SPLIT_SIZE_NK16384", 2048);
        return requested > 0 ? requested : fallback;
    }
    if (nk >= 8192) {
        const int requested = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO_SPLIT_SIZE_NK8192", 1024);
        return requested > 0 ? requested : fallback;
    }
    if (nk >= 4096) {
        const int requested = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_AUTO_SPLIT_SIZE_NK4096", 1024);
        return requested > 0 ? requested : fallback;
    }
    return fallback;
}

struct ggml_cuda_packed16_timing_trace_event {
    bool active = false;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

static inline bool ggml_cuda_packed16_timing_trace_enabled() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_TIMING_TRACE");
    return env && *env && atoi(env) != 0;
}

static inline void ggml_cuda_packed16_timing_trace_begin(
        ggml_cuda_packed16_timing_trace_event & ev,
        cudaStream_t stream,
        const char * phase) {
    ev = {};
    if (!ggml_cuda_packed16_timing_trace_enabled()) {
        return;
    }
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture_status) == hipSuccess && capture_status != hipStreamCaptureStatusNone) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "PACKED16_TIMING_TRACE skipped during graph capture phase=%s\n", phase ? phase : "-");
        }
        return;
    }
    CUDA_CHECK(hipEventCreate(&ev.start));
    CUDA_CHECK(hipEventCreate(&ev.stop));
    CUDA_CHECK(hipEventRecord(ev.start, stream));
    ev.active = true;
}

static inline float ggml_cuda_packed16_timing_trace_end(
        ggml_cuda_packed16_timing_trace_event & ev,
        cudaStream_t stream) {
    if (!ev.active) {
        return -1.0f;
    }
    CUDA_CHECK(hipEventRecord(ev.stop, stream));
    CUDA_CHECK(hipEventSynchronize(ev.stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, ev.start, ev.stop));
    CUDA_CHECK(hipEventDestroy(ev.start));
    CUDA_CHECK(hipEventDestroy(ev.stop));
    ev = {};
    return elapsed_ms;
}

static inline bool ggml_cuda_mtp_qblock_paged_attention_requested() {
    static const bool requested = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION");
    return requested;
}

static inline bool ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() {
    static const bool requested = ggml_cuda_mtp_qblock_paged_attention_requested() &&
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE");
    return requested;
}

static inline bool ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_k_requested() {
    static const bool requested = ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON_K") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_POISON_CANONICAL") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_POISON_CANONICAL_K"));
    return requested;
}

static inline bool ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_v_requested() {
    static const bool requested = ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON_V") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_POISON_CANONICAL") ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_POISON_CANONICAL_V"));
    return requested;
}

static inline int ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_min_idx() {
    return ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON_MIN_IDX", 0);
}

static inline int ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_max_idx() {
    return ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_MTP_QBLOCK_PAGED_ATTENTION_OWNED_TAIL_WRITE_CANONICAL_POISON_MAX_IDX", -1);
}

static inline bool ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_range_hits(
        const uint32_t idx0,
        const uint32_t n_tokens,
        const int poison_min_idx,
        const int poison_max_idx) {
    if (n_tokens == 0) {
        return false;
    }
    const uint64_t begin = idx0;
    const uint64_t end = begin + uint64_t(n_tokens);
    if (poison_min_idx > 0 && end <= uint64_t(poison_min_idx)) {
        return false;
    }
    if (poison_max_idx >= 0 && begin >= uint64_t(poison_max_idx)) {
        return false;
    }
    return true;
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_backend_proof_enabled() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PROOF");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_requested() {
    return ggml_cuda_mtp_qblock_paged_attention_requested() ||
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_consumer_requested() {
    return ggml_cuda_mtp_qblock_paged_attention_requested() ||
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_CONSUMER");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_producer_map_requested() {
    return ggml_cuda_mtp_qblock_paged_attention_requested() ||
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_PRODUCER_MAP");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_scratch_map_requested() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_SCRATCH_MAP");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_requested() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_SCRATCH_EXCLUSIVE");
}

static inline int ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_min_idx() {
    return ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_SCRATCH_EXCLUSIVE_MIN_IDX", 0);
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_after_nomap_max_requested() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_SCRATCH_EXCLUSIVE_AFTER_NOMAP_MAX");
}

static inline bool ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_single_map_unsafe_requested() {
    return ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_SCRATCH_EXCLUSIVE_SINGLE_MAP_UNSAFE");
}

static mtp_v4_144_tail_stage_status ggml_cuda_mtp_qblock_txn_tail_backend_geom_status(
        const uint32_t kind,
        const uint32_t cache_tokens,
        const uint32_t row_bytes) {
    const mtp_v4_144_tail_stage_desc_v1 desc = mtp_v4_144_tail_stage_make(
        kind,
        cache_tokens,
        0,
        MTP_V4_144_PAGE_TOKENS,
        row_bytes);
    return mtp_v4_144_tail_stage_validate_static(desc);
}

struct ggml_cuda_mtp_qblock_txn_tail_backend_idx_probe {
    bool copied = false;
    bool capture_skipped = false;
    bool contiguous = false;
    bool capacity_ok = false;
    bool spans_pages = false;
    uint32_t n = 0;
    uint32_t idx0 = 0;
    uint32_t idx_last = 0;
    uint32_t page_base = 0;
    uint32_t page_end = 0;
    uint32_t slot_begin = 0;
    uint32_t slot_end_excl = 0;
    uint32_t slots_before = 0;
    uint32_t slots_after = 0;
    uint32_t merge_copy_slots = 0;
    mtp_v4_144_tail_stage_status status = MTP_V4_144_TAIL_STAGE_BAD_WRITE_TOKENS;
};

static ggml_cuda_mtp_qblock_txn_tail_backend_idx_probe ggml_cuda_mtp_qblock_txn_tail_backend_probe_indices(
        const ggml_tensor * idxs,
        const int nk_cur,
        const int kv_size,
        const uint32_t kind,
        const uint32_t row_bytes,
        cudaStream_t stream) {
    ggml_cuda_mtp_qblock_txn_tail_backend_idx_probe probe = {};
    if (idxs == nullptr || idxs->data == nullptr || nk_cur <= 0 || nk_cur > 8 || kv_size <= 0) {
        return probe;
    }
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
    if (capture_status != hipStreamCaptureStatusNone) {
        probe.capture_skipped = true;
        probe.status = MTP_V4_144_TAIL_STAGE_OK;
        return probe;
    }

    int64_t h_idx[8] = {};
    if (idxs->type == GGML_TYPE_I64) {
        CUDA_CHECK(cudaMemcpyAsync(h_idx, idxs->data, (size_t) nk_cur * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    } else if (idxs->type == GGML_TYPE_I32) {
        int32_t h_idx32[8] = {};
        CUDA_CHECK(cudaMemcpyAsync(h_idx32, idxs->data, (size_t) nk_cur * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < nk_cur; ++i) {
            h_idx[i] = h_idx32[i];
        }
    } else {
        return probe;
    }
    if (idxs->type == GGML_TYPE_I64) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    probe.copied = true;
    probe.n = (uint32_t) nk_cur;
    if (h_idx[0] < 0 || h_idx[0] >= kv_size) {
        probe.status = MTP_V4_144_TAIL_STAGE_BAD_SLOT_RANGE;
        return probe;
    }
    probe.idx0 = (uint32_t) h_idx[0];
    probe.contiguous = true;
    for (int i = 0; i < nk_cur; ++i) {
        if (h_idx[i] != (int64_t) probe.idx0 + i || h_idx[i] < 0 || h_idx[i] >= kv_size) {
            probe.contiguous = false;
            break;
        }
    }
    probe.idx_last = probe.idx0 + (uint32_t) nk_cur - 1u;
    probe.page_base = probe.idx0 & ~(MTP_V4_144_PAGE_TOKENS - 1u);
    probe.page_end = probe.page_base + MTP_V4_144_PAGE_TOKENS;
    probe.slot_begin = probe.idx0 - probe.page_base;
    probe.slot_end_excl = probe.slot_begin + (uint32_t) nk_cur;
    probe.spans_pages = probe.slot_end_excl > MTP_V4_144_PAGE_TOKENS;
    probe.slots_before = probe.slot_begin;
    probe.slots_after = probe.spans_pages ? 0u : MTP_V4_144_PAGE_TOKENS - probe.slot_end_excl;
    probe.merge_copy_slots = probe.slots_before + probe.slots_after;
    probe.capacity_ok = (uint32_t) kv_size >= probe.page_end;

    const mtp_v4_144_tail_stage_desc_v1 desc = mtp_v4_144_tail_stage_make(
        kind,
        (uint32_t) kv_size,
        probe.idx0,
        (uint32_t) nk_cur,
        row_bytes);
    probe.status = mtp_v4_144_tail_stage_validate_static(desc);
    return probe;
}

static inline const char * ggml_cuda_q8k_dot4_k_physical_name(const ggml_tensor * K) {
    if (!K) {
        return "unknown";
    }
    if (K->type == GGML_TYPE_Q8_0) {
        return "q8_0_block32";
    }
    if (K->type == GGML_TYPE_I32) {
        return "packed16_q8_sidechannel_i32_f16scales";
    }
    if (K->type == GGML_TYPE_F16) {
        return "f16_source_op_local_packed16";
    }
    return "unknown";
}

static bool ggml_cuda_q8k_dot4_packed16_sidecar_valid(
        const ggml_tensor * K,
        const ggml_tensor * payload,
        const ggml_tensor * scales,
        const bool verbose,
        const char * context) {
    const char * reject = nullptr;
    int64_t head_stride_payload = 0;
    int64_t head_stride_scales = 0;

    if (!K || K->type != GGML_TYPE_I32) {
        reject = "not_i32_packed16_k";
    } else if (!payload || !scales) {
        reject = "missing_packed16_k_sidecar";
    } else if (payload->type != GGML_TYPE_I32 || scales->type != GGML_TYPE_F16) {
        reject = "bad_packed16_k_sidecar_type";
    } else if (K->ne[2] <= 0 || payload->ne[1] % K->ne[2] != 0 || scales->ne[1] % K->ne[2] != 0) {
        reject = "bad_packed16_k_head_stride";
    } else {
        head_stride_payload = payload->ne[1] / K->ne[2];
        head_stride_scales = scales->ne[1] / K->ne[2];
        if (payload->ne[0] != GGML_CUDA_Q8K_DOT4_KQ_D / 4 ||
                scales->ne[0] != GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0 ||
                head_stride_payload < K->ne[1] || head_stride_scales < K->ne[1] ||
                payload->ne[1] < K->ne[1] * K->ne[2] ||
                scales->ne[1] < K->ne[1] * K->ne[2] ||
                payload->nb[0] != (int64_t) sizeof(int) ||
                scales->nb[0] != (int64_t) sizeof(half) ||
                payload->nb[1] != (GGML_CUDA_Q8K_DOT4_KQ_D / 4) * (int64_t) sizeof(int) ||
                scales->nb[1] != (GGML_CUDA_Q8K_DOT4_KQ_D / QK8_0) * (int64_t) sizeof(half) ||
                payload->nb[2] != payload->ne[1] * payload->nb[1] ||
                scales->nb[2] != scales->ne[1] * scales->nb[1] ||
                payload->nb[3] != payload->ne[2] * payload->nb[2] ||
                scales->nb[3] != scales->ne[2] * scales->nb[2]) {
            reject = "bad_packed16_k_sidecar_shape";
        }
    }

    if (!reject) {
        return true;
    }

    if (verbose) {
        fprintf(stderr,
            "q8k_dot4_kq reject route=rocm_fa2_packed16_dot4_decode "
            "reject=%s context=%s K_data=%p K=[%lld,%lld,%lld,%lld] "
            "payload=%p scales=%p head_stride_payload=%lld head_stride_scales=%lld\n",
            reject, context ? context : "unknown", K ? K->data : nullptr,
            K ? (long long) K->ne[0] : 0, K ? (long long) K->ne[1] : 0,
            K ? (long long) K->ne[2] : 0, K ? (long long) K->ne[3] : 0,
            (const void *) payload, (const void *) scales,
            (long long) head_stride_payload, (long long) head_stride_scales);
    }
    return false;
}

static inline void ggml_cuda_q8k_dot4_kq_event_create(hipEvent_t * event) {
    CUDA_CHECK(hipEventCreate(event));
}

static inline void ggml_cuda_q8k_dot4_kq_event_destroy(hipEvent_t event) {
    if (event != nullptr) {
        CUDA_CHECK(hipEventDestroy(event));
    }
}

static inline float ggml_cuda_q8k_dot4_kq_event_elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_i8_i8(const int a, const int b, const int c) {
#if defined(RDNA3) || defined(RDNA4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return ggml_cuda_dp4a(a, b, c);
#endif
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_q8k_dot4_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ int ggml_cuda_q8k_dot4_4chain(
        const int * __restrict__ q_payload,
        const int * __restrict__ k_payload,
        int idx) {
    int acc = 0;
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 0], k_payload[idx + 0], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 1], k_payload[idx + 1], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 2], k_payload[idx + 2], acc);
    acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx + 3], k_payload[idx + 3], acc);
    return acc;
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_block(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ kq_sums,
        int buf) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    float partial = 0.0f;

    if (tid < 32) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = lane + i * 32;
            const int qb = idx / (QK8_0 / 4);
            const int acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx], k_payload[idx], 0);
            partial += float(acc) * q_scales[qb] * __half2float(k_scales[qb]);
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset, 32);
        }
        if (lane == 0) {
            kq_sums[buf] = partial;
        }
    }
    __syncthreads();
    return kq_sums[buf];
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_pack_k_packed16_from_q8_indexed_kernel(
        const char   * __restrict__ K,       // q8_0 K cache (block_q8_0 rows)
        int          * __restrict__ k_payload,
        half         * __restrict__ k_scales,
        const int64_t  * __restrict__ k_idxs,
        int64_t nb11,     // K row stride in bytes
        int kv_size,
        int n_heads_k,
        int batch,
        int nk_cur) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int qblk = tid;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || qblk >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const int64_t cell = k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) return;

    // Read q8_0 block: [d (half), qs[32] (int8)]
    const size_t q8_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    const char * src = K + q8_row * nb11 + qblk * sizeof(block_q8_0);
    // block_q8_0 layout: d (half, 2 bytes) + qs[32] (int8, 32 bytes) = 34 bytes per block
    // Write to packed16: k_scales = d, k_payload = qs as i32
    const size_t packed_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    k_scales[packed_row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_q8k_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        k_payload[packed_row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4) + i] =
            ggml_cuda_q8k_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_k_packed16_kernel(
        const half  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nk,
        int n_heads_k,
        int batch) {
    const int tid = threadIdx.x;
    const int k = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (k >= nk || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const half * k_ptr = (const half *) ((const char *) K + int64_t(b) * nb03 + int64_t(hk) * nb02 + int64_t(k) * nb01);
    const int d = q_block * QK8_0 + lane;
    float x = __half2float(k_ptr[d]);
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k);
    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }

}

template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_quant_k_packed8_q4_indexed_kernel(
        const char  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size,
        bool src_f16,
        int scale_mode,
        float scale_mul) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) return;
    const int64_t cell = (int64_t) k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) return;
    const char * row = K + int64_t(b) * nb03 + int64_t(k_local) * nb01 + int64_t(hk) * src_head_stride_bytes;
    float vals[QK8_0];
    float amax = 0.0f, maxv = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        const int d = q_block * QK8_0 + i;
        const float x = src_f16 ? __half2float(((const half *) row)[d]) : ((const float *) row)[d];
        vals[i] = x;
        const float ax = fabsf(x);
        if (ax > amax) { amax = ax; maxv = x; }
    }
    float d_f = maxv / -8.0f;
    if (scale_mode == 1 || scale_mode == 4) {
        // Symmetric positive scale, preserving the code zero point: value=(code-8)*d.
        d_f = amax > 0.0f ? amax / 7.0f : 1.0f;
    } else if (scale_mode == 2) {
        // Symmetric positive scale with extra negative headroom.
        d_f = amax > 0.0f ? amax / 8.0f : 1.0f;
    }
    d_f *= scale_mul;
    if (d_f == 0.0f || !isfinite(d_f)) {
        d_f = 1.0f;
    }
    float id = 1.0f / d_f;
    uint8_t codes[QK8_0];
#pragma unroll
    for (int i = 0; i < QK8_0; ++i) codes[i] = (uint8_t) min(15, max(0, int(vals[i] * id + 8.5f)));
    if (scale_mode == 3 || scale_mode == 4) {
        float num = 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            const float q = float(int(codes[i]) - 8);
            num += vals[i] * q;
            den += q * q;
        }
        if (den > 0.0f) {
            const float d_mse = num / den;
            if (d_mse != 0.0f && isfinite(d_mse)) {
                d_f = d_mse * scale_mul;
                id = 1.0f / d_f;
#pragma unroll
                for (int i = 0; i < QK8_0; ++i) codes[i] = (uint8_t) min(15, max(0, int(vals[i] * id + 8.5f)));
            }
        }
    }
    const half d_h = __float2half(d_f);
    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
    const size_t payload_row = (head_base + size_t(cell)) * size_t(GGML_CUDA_Q8K_DOT4_KQ_D / 8);
    const size_t scale_row   = (head_base + size_t(cell)) * size_t(GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
    k_scales[scale_row + size_t(q_block)] = d_h;
#pragma unroll
    for (int d16 = 0; d16 < 2; ++d16) {
        const int d16_global = q_block * 2 + d16;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w0 |= uint32_t(codes[d16 * 16 + i] & 0x0f) << (4 * i);
            w1 |= uint32_t(codes[d16 * 16 + 8 + i] & 0x0f) << (4 * i);
        }
        k_payload[payload_row + size_t(d16_global * 2 + 0)] = (int) w0;
        k_payload[payload_row + size_t(d16_global * 2 + 1)] = (int) w1;
    }
}

// Indexed variant: writes to absolute KV cache slots using k_idxs.
// Uses fixed kv_size head stride so persistent cache survives chunk growth.
template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_k_packed16_indexed_kernel(
        const half  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const int64_t cell = (int64_t) k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) {
        return;
    }

    // Read K source as float — k_cur may be f32 (model output) or f16.
    // Always use 4-byte float stride for head offset.
    const float * k_ptr = (const float *) ((const char *) K
            + int64_t(b)       * nb03
            + int64_t(k_local) * nb01
            + int64_t(hk)      * src_head_stride_bytes);
    const int d = q_block * QK8_0 + lane;
    const float x = k_ptr[d];

    // Pass 1: block-wise amax for initial scale.
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale0 = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi0 = max(-128, min(127, int(lrintf(x / scale0))));

    // Pass 2: MSE-optimal scale = sum(x * qi) / sum(qi^2)
    const float xi = (float) qi0;
    float num = x * xi;
    float den = xi * xi;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        num += __shfl_xor(num, mask, 32);
        den += __shfl_xor(den, mask, 32);
    }
    const float scale = (den > 0.0f) ? (num / den) : scale0;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    // Fixed kv_size head stride — row = head * kv_size + cell
    const size_t row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    ((int8_t *)(k_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        k_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = __float2half(scale);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_quant_q_packed16_kernel(
        const float * __restrict__ Q,
        int         * __restrict__ q_payload,
        float       * __restrict__ q_scales,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int nq,
        int n_heads_q,
        int batch) {
    const int tid = threadIdx.x;
    const int q = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (q >= nq || hq >= n_heads_q || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q) * nb01);
    const int d = q_block * QK8_0 + lane;
    const float x = q_ptr[d];
    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t row = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q);
    ((int8_t *)(q_payload + row * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)))[d] = (int8_t) qi;
    if (lane == 0) {
        q_scales[row * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + q_block] = scale;
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_pack_k_packed16_kernel(
        const char * __restrict__ K,
        int        * __restrict__ k_payload,
        half       * __restrict__ k_scales,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nk,
        int n_heads_k,
        int batch) {
    const int linear = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int total = batch * n_heads_k * nk * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    if (linear >= total) {
        return;
    }

    const int qblk = linear % GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int t = linear / GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const int k = t % nk;
    const int hk_b = t / nk;
    const int hk = hk_b % n_heads_k;
    const int b = hk_b / n_heads_k;
    const char * src = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k) * nb11 + int64_t(qblk) * nb10;
    int * dst = k_payload + (size_t(t) * (GGML_CUDA_Q8K_DOT4_KQ_D / 4) + qblk * (QK8_0 / 4));

    k_scales[size_t(t) * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + qblk] = ggml_cuda_q8k_dot4_load_half_unaligned(src);
#pragma unroll
    for (int i = 0; i < QK8_0 / 4; ++i) {
        dst[i] = ggml_cuda_q8k_dot4_load_i32_unaligned(src + sizeof(half) + 4 * i);
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        float       * __restrict__ logits,
        float scale,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int half_tile = (tid >> 4) & 1;
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;

    const size_t q_base = ((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row);
    const size_t k_base = ((size_t(b) * n_heads_k + hk) * (size_t)nk + k_row);
    const int * q_row_payload = q_payload + q_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const int * k_row_payload = k_payload + k_base * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
    const float * q_row_scales = q_scales + q_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    const half * k_row_scales = k_scales + k_base * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const int idx = qb * (QK8_0 / 4) + half_tile * 4;
        const int acc = ggml_cuda_q8k_dot4_4chain(q_row_payload, k_row_payload, idx);
        sum += float(acc) * q_row_scales[qb] * __half2float(k_row_scales[qb]);
    }
    sum += __shfl_xor(sum, 16, 32);
    if (half_tile == 0) {
        logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    }
    GGML_UNUSED(n_heads_k);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_q4_0(
        const char * __restrict__ V,
        int64_t nb20,
        int i) {
    const int ib = i / QK4_0;
    const int iq = i & 15;
    const int shift = (i & 31) >= 16 ? 4 : 0;
    const block_q4_0 * v = (const block_q4_0 *) (V + int64_t(ib) * nb20);
    const int q = (v->qs[iq] >> shift) & 0x0f;
    return (float(q) - 8.0f) * __half2float(v->d);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_q8_0(
        const char * __restrict__ V,
        int64_t nb20,
        int i) {
    const int ib = i / QK8_0;
    const int iq = i % QK8_0;
    const block_q8_0 * v = (const block_q8_0 *) (V + int64_t(ib) * nb20);
    return float(v->qs[iq]) * __half2float(v->d);
}

static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_D32 = 32;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_PAYLOAD_BYTES = 2048;
static constexpr int GGML_CUDA_V4_K16D16_144_DECODE_WORDS_PER_D = 2;

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_dequant_v4_k16d16_144(
        const char * __restrict__ V_head,
        const int64_t nb21,
        const int k,
        const int d) {
    const int k16_base = k & ~(GGML_CUDA_V4_K16D16_144_DECODE_K - 1);
    const int slot = k & (GGML_CUDA_V4_K16D16_144_DECODE_K - 1);
    const char * block = V_head + int64_t(k16_base) * nb21;
    const uint32_t word = ((const uint32_t *) block)[d * GGML_CUDA_V4_K16D16_144_DECODE_WORDS_PER_D + (slot >> 3)];
    const int q = int((word >> (4 * (slot & 7))) & 0x0fu) - 8;
    const int d32 = d / GGML_CUDA_V4_K16D16_144_DECODE_D32;
    const half * scales = (const half *) (block + GGML_CUDA_V4_K16D16_144_DECODE_PAYLOAD_BYTES);
    return float(q) * __half2float(scales[d32 * GGML_CUDA_V4_K16D16_144_DECODE_K + slot]);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_mask_value(
        const char * __restrict__ mask,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int q,
        int k,
        int b) {
    if (!mask) {
        return 0.0f;
    }
    const char * p = mask + int64_t(b % ne33) * nb33 + int64_t(q) * nb31 + int64_t(k) * nb30;
    return __half2float(*(const half *) p);
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel(
        const float * __restrict__ logits,
        const char  * __restrict__ V,
        const char  * __restrict__ mask,
        float       * __restrict__ dst,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int64_t nb30,
        int64_t nb31,
        int64_t nb33,
        int64_t ne33,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.x;
    const int hq = blockIdx.y;
    const int b = blockIdx.z;
    if (q_row >= nq || hq >= n_heads_q || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float out = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const float s = logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k] +
            ggml_cuda_q8k_dot4_mask_value(mask, nb30, nb31, nb33, ne33, q_row, k, b);
        const float next_max = fmaxf(row_max, s);
        const float old_scale = denom > 0.0f ? expf(row_max - next_max) : 0.0f;
        const float p = expf(s - next_max);
        out = out * old_scale + p * ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, tid);
        denom = denom * old_scale + p;
        row_max = next_max;
    }
    dst[((size_t(b) * nq + q_row) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = out / denom;
    GGML_UNUSED(n_heads_k);
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_direct_d64(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales) {
    // 4 independent D64 accumulators — shorter dependency chains = better ILP on RDNA3
    static constexpr int CHUNK_BLOCKS = 2;  // 2 blocks × 32 = D64
    float sum = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS / CHUNK_BLOCKS; ++chunk) {
        int acc = 0;
#pragma unroll
        for (int qb = 0; qb < CHUNK_BLOCKS; ++qb) {
            const int block = chunk * CHUNK_BLOCKS + qb;
            const int base = block * (QK8_0 / 4);
#pragma unroll
            for (int i = 0; i < QK8_0 / 4; ++i) {
                const int idx = base + i;
                acc = ggml_cuda_q8k_dot4_i8_i8(q_payload[idx], k_payload[idx], acc);
            }
            sum += float(acc) * q_scales[block] * __half2float(k_scales[block]);
            acc = 0;
        }
    }
    return sum;
}

static __device__ __forceinline__ float ggml_cuda_q8k_dot4_kq_dot_direct(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales) {
    return ggml_cuda_q8k_dot4_kq_dot_direct_d64(q_payload, q_scales, k_payload, k_scales);
}

template <bool USE_F16_V, bool USE_Q8_V, int BN, int BM>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel(
        const int   * __restrict__ q_payload,
        const float * __restrict__ q_scales,
        const int   * __restrict__ k_payload,
        const half  * __restrict__ k_scales,
        const char  * __restrict__ V,
        float       * __restrict__ dst,
        float scale,
        int64_t nb20,
        int64_t nb21,
        int64_t nb22,
        int64_t nb23,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch,
        int q_offset,
        int k_head_stride_rows,
        int k_batch_stride_rows,
        int debug) {
    static constexpr int BM_VAL = BM;
    static constexpr bool F16_V = USE_F16_V;
    static constexpr bool Q8_V  = USE_Q8_V;
    static constexpr int I32_PER_ROW = GGML_CUDA_Q8K_DOT4_KQ_D / 4;
    static constexpr int N_BLOCKS   = GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
    static constexpr int V_TILE_SIZE = BN * GGML_CUDA_Q8K_DOT4_KQ_D;
    extern __shared__ __align__(16) unsigned char smem[];
    float * logits          = reinterpret_cast<float *>(smem);
    float * row_m           = logits + BM_VAL * BN;
    float * row_l           = row_m + BM_VAL;
    float * old_s           = row_l + BM_VAL;
    float * v_tile          = old_s + BM_VAL;
    float * v_tile_next     = v_tile + V_TILE_SIZE;
    int   * q_payload_tile  = reinterpret_cast<int  *>(v_tile_next + V_TILE_SIZE);
    float * q_scales_tile   = reinterpret_cast<float *>(q_payload_tile + BM_VAL * I32_PER_ROW);

    const int tid = int(threadIdx.x);
    const int q0 = int(blockIdx.x) * BM_VAL;
    const int hq = int(blockIdx.y);
    const int b = int(blockIdx.z);
    const int hk = hq / gqa_ratio;
    if (b >= batch || hq >= n_heads_q || hk >= n_heads_k) return;

    const size_t q_head_base = ((size_t(b) * n_heads_q + hq) * size_t(nq));
    const size_t k_head_base = size_t(b) * size_t(k_batch_stride_rows)
                             + size_t(hk) * size_t(k_head_stride_rows);
    const char * v_head = V + int64_t(b) * nb23 + int64_t(hk) * nb22;

    for (int off = tid; off < BM_VAL * I32_PER_ROW; off += blockDim.x) {
        const int qr = off / I32_PER_ROW;
        const int q = q0 + qr;
        q_payload_tile[off] = (q < nq) ? q_payload[(q_head_base + q) * I32_PER_ROW + (off - qr * I32_PER_ROW)] : 0;
    }
    for (int off = tid; off < BM_VAL * N_BLOCKS; off += blockDim.x) {
        const int qr = off / N_BLOCKS;
        const int q = q0 + qr;
        q_scales_tile[off] = (q < nq) ? q_scales[(q_head_base + q) * N_BLOCKS + (off - qr * N_BLOCKS)] : 1.0f;
    }

    if (tid < BM_VAL) { row_m[tid] = -FLT_MAX / 2.0f; row_l[tid] = 0.0f; old_s[tid] = 0.0f; }
    float out[BM_VAL]; for (int qr = 0; qr < BM_VAL; ++qr) out[qr] = 0.0f;

    // Pre-fetch first V tile
    {
        const int k0 = 0;
        const int tile_n = min(BN, nk - k0);
        for (int off = tid; off < V_TILE_SIZE; off += blockDim.x) {
            const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
            const int k = k0 + kk;
            v_tile[off] = (kk < tile_n && k < nk)
                ? (F16_V
                    ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                    : (Q8_V
                        ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                        : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nk; k0 += BN) {
        const int tile_n = min(BN, nk - k0);
        const bool have_next = k0 + BN < nk;
        const int next_k0 = k0 + BN;
        const int next_tile_n = min(BN, nk - next_k0);

        if (k0 > q_offset + min(q0 + BM_VAL - 1, nq - 1)) continue;
        const bool exact_tile = (q0 + BM_VAL <= nq) && (tile_n == BN) && (k0 + BN - 1 <= q_offset + q0);

        if (exact_tile) {
            // QK dot on threads 0-63, V pre-fetch on 64-255 (if have_next)
            if (tid < BM_VAL * BN) {
                const int qr = tid / BN, kk = tid - qr * BN;
                const size_t k_base = k_head_base + k0 + kk;
                logits[tid] = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW, q_scales_tile + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW, k_scales + k_base * N_BLOCKS) * scale;
            } else if (have_next && tid >= BM_VAL * BN) {
                // Pre-fetch next V tile while QK dot runs
                const int pf_start = BM_VAL * BN;
                const int num_pf = blockDim.x - pf_start;
                for (int off = tid - pf_start; off < V_TILE_SIZE; off += num_pf) {
                    const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
                    const int k = next_k0 + kk;
                    v_tile_next[off] = (kk < next_tile_n && k < nk)
                        ? (F16_V
                            ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                            : (Q8_V
                                ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                                : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
                }
            }
            __syncthreads();
            if (debug && b == 0 && hq == 0 && q0 == 0 && k0 == 0 && tid < min(8, BN)) {
                printf("dot4_debug_qk: q=0 h=0 k=%d logit=%.8e scale=%.8e q_offset=%d\n", tid, logits[tid], scale, q_offset);
            }
            // Softmax
            if (tid < BM_VAL) {
                float tile_m = -FLT_MAX / 2.0f;
                for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
                const float m_new = fmaxf(row_m[tid], tile_m);
                const float old_scale = expf(row_m[tid] - m_new);
                float tile_l = 0.0f;
                for (int kk = 0; kk < BN; ++kk) tile_l += expf(logits[tid * BN + kk] - m_new);
                row_l[tid] = row_l[tid] * old_scale + tile_l;
                row_m[tid] = m_new; old_s[tid] = old_scale;
            }
            __syncthreads();
            if (debug && b == 0 && hq == 0 && q0 == 0 && (k0 == 0 || !have_next) && tid == 0) {
                printf("dot4_debug_ml: q=0 h=0 k0=%d m=%.8e l=%.8e\n", k0, row_m[0], row_l[0]);
            }
            // V accumulation from current v_tile
            if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
                for (int qr = 0; qr < BM_VAL; ++qr) {
                    float acc = out[qr] * old_s[qr];
                    for (int kk = 0; kk < BN; ++kk)
                        acc += expf(logits[qr * BN + kk] - row_m[qr]) * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                    out[qr] = acc;
                }
            }
            __syncthreads();
            // Swap V tiles for next iteration
            if (have_next) {
                float * tmp = v_tile;
                v_tile = v_tile_next;
                v_tile_next = tmp;
            }
            __syncthreads();
            continue;
        }

        // Edge tile: V pre-fetch on threads 64-255 in parallel with QK dot
        if (tid < BM_VAL * BN) {
            const int qr = tid / BN, kk = tid - qr * BN, q = q0 + qr, k = k0 + kk;
            float s = -FLT_MAX / 2.0f;
            if (q < nq && kk < tile_n && k < nk && (k - q_offset <= q)) {
                const size_t k_base = k_head_base + k;
                s = ggml_cuda_q8k_dot4_kq_dot_direct(
                    q_payload_tile + qr * I32_PER_ROW, q_scales_tile + qr * N_BLOCKS,
                    k_payload + k_base * I32_PER_ROW, k_scales + k_base * N_BLOCKS) * scale;
            }
            logits[tid] = s;
        } else if (have_next && tid >= BM_VAL * BN) {
            const int pf_start = BM_VAL * BN;
            const int num_pf = blockDim.x - pf_start;
            for (int off = tid - pf_start; off < V_TILE_SIZE; off += num_pf) {
                const int kk = off / GGML_CUDA_Q8K_DOT4_KQ_D;
                const int k = next_k0 + kk;
                v_tile_next[off] = (kk < next_tile_n && k < nk)
                    ? (F16_V
                        ? __half2float(((const half *)(v_head + int64_t(k) * nb21))[off - kk * GGML_CUDA_Q8K_DOT4_KQ_D])
                        : (Q8_V
                            ? ggml_cuda_q8k_dot4_dequant_q8_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D)
                            : ggml_cuda_q8k_dot4_dequant_q4_0(v_head + int64_t(k) * nb21, nb20, off - kk * GGML_CUDA_Q8K_DOT4_KQ_D))) : 0.0f;
            }
        }
        __syncthreads();
        if (debug && b == 0 && hq == 0 && q0 == 0 && k0 == 0 && tid < min(8, BN)) {
            printf("dot4_debug_qk: q=0 h=0 k=%d logit=%.8e scale=%.8e q_offset=%d\n", tid, logits[tid], scale, q_offset);
        }
        if (tid < BM_VAL) {
            float tile_m = -FLT_MAX / 2.0f;
            for (int kk = 0; kk < BN; ++kk) tile_m = fmaxf(tile_m, logits[tid * BN + kk]);
            const float m_new = fmaxf(row_m[tid], tile_m);
            const float old_scale = expf(row_m[tid] - m_new);
            float tile_l = 0.0f;
            for (int kk = 0; kk < BN; ++kk) tile_l += expf(logits[tid * BN + kk] - m_new);
            row_l[tid] = row_l[tid] * old_scale + tile_l;
            row_m[tid] = m_new; old_s[tid] = old_scale;
        }
        __syncthreads();
        if (debug && b == 0 && hq == 0 && q0 == 0 && (k0 == 0 || !have_next) && tid == 0) {
            printf("dot4_debug_ml: q=0 h=0 k0=%d m=%.8e l=%.8e\n", k0, row_m[0], row_l[0]);
        }
        if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
            for (int qr = 0; qr < BM_VAL; ++qr) {
                const int q = q0 + qr;
                if (q >= nq) continue;
                float acc = out[qr] * old_s[qr];
                for (int kk = 0; kk < BN; ++kk) {
                    const int k = k0 + kk;
                    if (kk >= tile_n || k >= nk || (k - q_offset > q)) continue;
                    acc += expf(logits[qr * BN + kk] - row_m[qr]) * v_tile[kk * GGML_CUDA_Q8K_DOT4_KQ_D + tid];
                }
                out[qr] = acc;
            }
        }
        __syncthreads();
        if (have_next) {
            float * tmp = v_tile;
            v_tile = v_tile_next;
            v_tile_next = tmp;
        }
        __syncthreads();
    }

    if (tid < GGML_CUDA_Q8K_DOT4_KQ_D) {
        for (int qr = 0; qr < BM_VAL; ++qr) {
            const int q = q0 + qr;
            if (q < nq) {
                const float denom = fmaxf(row_l[qr], 1.0e-20f);
                const float val = out[qr] / denom;
                if (debug && b == 0 && hq == 0 && q == 0 && tid < 8) {
                    const bool bad = !isfinite(row_m[qr]) || !isfinite(row_l[qr]) || row_l[qr] <= 0.0f || !isfinite(val) || fabsf(val) > 1.0e6f;
                    printf("dot4_debug_o: q=0 h=0 d=%d m=%.8e l=%.8e dst=%.8e bad=%d\n", tid, row_m[qr], row_l[qr], val, (int) bad);
                }
                dst[((size_t(b) * nq + q) * (size_t)n_heads_q + hq) * GGML_CUDA_Q8K_DOT4_KQ_D + tid] = val;
            }
        }
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_q8k_dot4_kq_reference_kernel(
        const float * __restrict__ Q,
        const char  * __restrict__ K,
        float       * __restrict__ logits,
        float scale,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb10,
        int64_t nb11,
        int64_t nb12,
        int64_t nb13,
        int nq,
        int nk,
        int n_heads_q,
        int n_heads_k,
        int gqa_ratio,
        int batch) {
    const int tid = threadIdx.x;
    const int q_row = blockIdx.y * GGML_CUDA_Q8K_DOT4_KQ_TILE_M + (tid >> 5);
    const int k_row = blockIdx.x * GGML_CUDA_Q8K_DOT4_KQ_TILE_N + (tid & 15);
    const int hq = blockIdx.z % n_heads_q;
    const int b = blockIdx.z / n_heads_q;
    if (q_row >= nq || k_row >= nk || b >= batch) {
        return;
    }
    const int hk = hq / gqa_ratio;
    const float * q_ptr = (const float *) ((const char *) Q + int64_t(b) * nb03 + int64_t(hq) * nb02 + int64_t(q_row) * nb01);

    float sum = 0.0f;
#pragma unroll
    for (int qb = 0; qb < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++qb) {
        const char * k_blk = K + int64_t(b) * nb13 + int64_t(hk) * nb12 + int64_t(k_row) * nb11 + int64_t(qb) * nb10;
        const half ks_h = ggml_cuda_q8k_dot4_load_half_unaligned(k_blk);
        const float ks = __half2float(ks_h);
#pragma unroll
        for (int lane = 0; lane < QK8_0; ++lane) {
            const int8_t kval = *(const int8_t *) (k_blk + sizeof(half) + lane);
            sum += q_ptr[qb * QK8_0 + lane] * float(kval) * ks;
        }
    }
    logits[((size_t(b) * n_heads_q + hq) * (size_t)nq + q_row) * (size_t)nk + k_row] = sum * scale;
    GGML_UNUSED(n_heads_k);
}

static __global__ void ggml_cuda_q8k_dot4_kq_error_kernel(
        const float * __restrict__ candidate,
        const float * __restrict__ reference,
        float       * __restrict__ metrics,
        size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float diff = fabsf(candidate[i] - reference[i]);
    atomicAdd(metrics + 0, diff * diff);
    atomicAdd(metrics + 1, diff);
    atomicAdd(metrics + 2, diff > 0.5f ? 1.0f : 0.0f);
}

#include "fattn-packed16-wmma-builtin.cuh"
#include "fattn-dot4-q8k-decode.cuh"

static inline bool ggml_cuda_pack_k_packed16_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && atoi(env) != 0;
}

static inline bool ggml_cuda_q8k_dot4_consumer_read_mode_trace_enabled() {
    const char * v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_CONSUMER_READ_MODE_TRACE");
    if (v && atoi(v) != 0) {
        return true;
    }
    v = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_TXN_TAIL_PAGE_CONSUMER_READ_MODE_TRACE");
    return v && atoi(v) != 0;
}

static inline const char * ggml_cuda_q8k_dot4_fattn_node_name(const ggml_tensor * dst) {
    return dst && dst->name[0] ? dst->name : "-";
}

static inline int ggml_cuda_q8k_dot4_fattn_layer_from_node_name(const ggml_tensor * dst) {
    const char * name = ggml_cuda_q8k_dot4_fattn_node_name(dst);
    constexpr const char * prefix = "__fattn__-";
    constexpr size_t prefix_len = sizeof("__fattn__-") - 1;
    if (strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long layer = strtol(name + prefix_len, &end, 10);
    return end && end != name + prefix_len ? (int) layer : -1;
}

// Packed16 K quality knobs for acceptance/drift experiments. Defaults preserve
// existing behavior: persistent indexed K-cache packing keeps the current
// one-step MSE scale at block32 granularity.
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE=maxabs|mse|0|1
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL=<positive float>
//   GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS=1|2|4|8
// Coarser scale groups are diagnostics for the "F16 scale mask" hypothesis:
// they requantize K with one shared scale over multiple 32-dim blocks and write
// that scale back into each existing scale slot. This preserves the tensor ABI
// and tests quality sensitivity; it does not reduce scale-memory traffic yet.
// LLAMA_MTP_* aliases are accepted for MTP-specific sweeps.
static inline int ggml_cuda_pack_k_packed16_scale_mode(int fallback) {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_MODE");
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_MODE");
    }
    if (!env || env[0] == '\0') {
        return fallback;
    }
    const std::string mode(env);
    if (mode == "maxabs" || mode == "amax" || mode == "0") {
        return 0;
    }
    if (mode == "mse" || mode == "mse1" || mode == "1") {
        return 1;
    }
    return fallback;
}

static inline float ggml_cuda_pack_k_packed16_scale_mul() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_MUL");
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_MUL");
    }
    if (!env || env[0] == '\0') {
        return 1.0f;
    }
    const float v = strtof(env, nullptr);
    return (v > 0.0f && v < 100.0f) ? v : 1.0f;
}

static inline int ggml_cuda_pack_k_packed8_q4_scale_mode() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED8_Q4_K_SCALE_MODE");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_PACKED8_Q4_SCALE_MODE");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED8_Q4_K_SCALE_MODE");
    }
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_PDMQ_PACKED8_Q4_SCALE_MODE");
    }
    if (!env || env[0] == '\0') {
        return 0;
    }
    const std::string mode(env);
    if (mode == "q4_0" || mode == "legacy" || mode == "current" || mode == "0") return 0;
    if (mode == "sym7" || mode == "symmetric" || mode == "amax7" || mode == "1") return 1;
    if (mode == "sym8" || mode == "amax8" || mode == "2") return 2;
    if (mode == "mse" || mode == "mse_q4_0" || mode == "3") return 3;
    if (mode == "mse_sym" || mode == "mse_sym7" || mode == "4") return 4;
    return 0;
}

static inline float ggml_cuda_pack_k_packed8_q4_scale_mul() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED8_Q4_K_SCALE_MUL");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_PACKED8_Q4_SCALE_MUL");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED8_Q4_K_SCALE_MUL");
    }
    if (!env || env[0] == '\0') {
        return 1.0f;
    }
    const float v = strtof(env, nullptr);
    return (v > 0.0f && v < 100.0f) ? v : 1.0f;
}

static inline int ggml_cuda_pack_k_packed16_scale_group_qblocks() {
    const char * env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP_QBLOCKS");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_PACKED16_K_SCALE_GROUP");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP_QBLOCKS");
    }
    if (!env) {
        env = getenv("LLAMA_MTP_PACKED16_K_SCALE_GROUP");
    }
    if (!env || env[0] == '\0') {
        return 1;
    }
    const int v = atoi(env);
    if (v <= 1) return 1;
    if (v <= 2) return 2;
    if (v <= 4) return 4;
    return GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
}

static __device__ __forceinline__ int ggml_cuda_pack_k_packed16_load_i32_unaligned(const char * p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ half ggml_cuda_pack_k_packed16_load_half_unaligned(const char * p) {
    half v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static __device__ __forceinline__ int ggml_cuda_pack_k_packed16_i8x4_word_from_lanes(const int qi, const int lane) {
    const int lane0 = lane & ~3;
    const uint32_t q0 = uint32_t(__shfl_sync(0xffffffff, qi, lane0 + 0, 32)) & 0xffu;
    const uint32_t q1 = uint32_t(__shfl_sync(0xffffffff, qi, lane0 + 1, 32)) & 0xffu;
    const uint32_t q2 = uint32_t(__shfl_sync(0xffffffff, qi, lane0 + 2, 32)) & 0xffu;
    const uint32_t q3 = uint32_t(__shfl_sync(0xffffffff, qi, lane0 + 3, 32)) & 0xffu;
    return int(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_k_packed16_from_q8_indexed_kernel(
        const char     * __restrict__ K,       // q8_0 K cache (block_q8_0 rows)
        int            * __restrict__ k_payload,
        half           * __restrict__ k_scales,
        const int64_t  * __restrict__ k_idxs,
        int64_t nb11,     // K row stride in bytes
        int kv_size,
        int n_heads_k,
        int batch,
        int nk_cur,
        int layout_kind) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int d_word = tid;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || d_word >= GGML_CUDA_PACKED16_K_WORDS) {
        return;
    }

    const int64_t cell = k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) {
        return;
    }

    constexpr int WORDS_PER_QBLOCK = QK8_0 / 4;
    const int qblk = d_word / WORDS_PER_QBLOCK;
    const int word_in_qblk = d_word - qblk * WORDS_PER_QBLOCK;
    const size_t q8_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size) + size_t(cell);
    const char * src = K + q8_row * nb11 + qblk * sizeof(block_q8_0);

    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
    if (word_in_qblk == 0) {
        k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, int(cell), qblk, layout_kind)] =
            ggml_cuda_pack_k_packed16_load_half_unaligned(src);
    }
    k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, int(cell), d_word, layout_kind)] =
        ggml_cuda_pack_k_packed16_load_i32_unaligned(src + sizeof(half) + 4 * word_in_qblk);
}

// Indexed variant: writes to absolute KV cache slots using k_idxs.
// Uses fixed kv_size head stride so persistent cache survives chunk growth.
template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_quant_k_packed16_indexed_kernel(
        const half  * __restrict__ K,
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size,
        int scale_mode,
        float scale_mul,
        int scale_group_qblocks,
        int layout_kind,
        int overlay_page_base) {
    GGML_UNUSED(nb02);
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int q_block = tid >> 5;
    const int lane = tid & 31;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch || q_block >= GGML_CUDA_Q8K_DOT4_KQ_BLOCKS) {
        return;
    }

    const int64_t cell = (int64_t) k_idxs[k_local];
    if (cell < 0 || cell >= kv_size) {
        return;
    }
    const int overlay_cell = overlay_page_base >= 0 ? overlay_page_base + k_local : -1;
    const bool overlay_valid = overlay_cell >= 0 && overlay_cell < kv_size;

    // Preserve the existing packer contract: graph code may pass f32 or f16
    // storage, and existing callers provide byte strides describing the row.
    const float * k_ptr = (const float *) ((const char *) K
            + int64_t(b)       * nb03
            + int64_t(k_local) * nb01
            + int64_t(hk)      * src_head_stride_bytes);

    const int d = q_block * QK8_0 + lane;
    const float x = k_ptr[d];

    float amax = fabsf(x);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
    }

    // Preserve the original block32 packer exactly unless the diagnostic asks
    // for coarser F16 scale-mask groups.
    if (scale_group_qblocks <= 1) {
        const float scale0 = amax > 0.0f ? amax / 127.0f : 1.0f;
        const int qi0 = max(-128, min(127, int(lrintf(x / scale0))));

        float scale = scale0;
        if (scale_mode != 0) {
            const float xi = (float) qi0;
            float num = x * xi;
            float den = xi * xi;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                num += __shfl_xor(num, mask, 32);
                den += __shfl_xor(den, mask, 32);
            }
            scale = (den > 0.0f) ? (num / den) : scale0;
        }
        scale *= scale_mul;
        if (!(scale > 0.0f)) {
            scale = scale0;
        }
        const int qi = max(-128, min(127, int(lrintf(x / scale))));

        const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
        const int packed_word = ggml_cuda_pack_k_packed16_i8x4_word_from_lanes(qi, lane);
        if ((lane & 3) == 0) {
            k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, int(cell), d / 4, layout_kind)] = packed_word;
            if (overlay_valid) {
                k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, overlay_cell, d / 4, layout_kind)] = packed_word;
            }
        }
        if (lane == 0) {
            const half hscale = __float2half(scale);
            k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, int(cell), q_block, layout_kind)] = hscale;
            if (overlay_valid) {
                k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, overlay_cell, q_block, layout_kind)] = hscale;
            }
        }
        return;
    }

    __shared__ float block_amax[GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    __shared__ float block_num [GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    __shared__ float block_den [GGML_CUDA_Q8K_DOT4_KQ_BLOCKS];
    if (lane == 0) {
        block_amax[q_block] = amax;
    }
    __syncthreads();

    const int group_qblocks = scale_group_qblocks <= 1 ? 1 : (scale_group_qblocks <= 2 ? 2 : (scale_group_qblocks <= 4 ? 4 : GGML_CUDA_Q8K_DOT4_KQ_BLOCKS));
    const int group_start = (q_block / group_qblocks) * group_qblocks;
    const int group_end = min(group_start + group_qblocks, GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    float group_amax = 0.0f;
#pragma unroll
    for (int s = 0; s < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++s) {
        if (s >= group_start && s < group_end) {
            group_amax = fmaxf(group_amax, block_amax[s]);
        }
    }
    const float scale0 = group_amax > 0.0f ? group_amax / 127.0f : 1.0f;
    const int qi0 = max(-128, min(127, int(lrintf(x / scale0))));

    float scale = scale0;
    if (scale_mode != 0) {
        const float xi = (float) qi0;
        float num = x * xi;
        float den = xi * xi;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            num += __shfl_xor(num, mask, 32);
            den += __shfl_xor(den, mask, 32);
        }
        if (lane == 0) {
            block_num[q_block] = num;
            block_den[q_block] = den;
        }
        __syncthreads();

        float group_num = 0.0f;
        float group_den = 0.0f;
#pragma unroll
        for (int s = 0; s < GGML_CUDA_Q8K_DOT4_KQ_BLOCKS; ++s) {
            if (s >= group_start && s < group_end) {
                group_num += block_num[s];
                group_den += block_den[s];
            }
        }
        scale = (group_den > 0.0f) ? (group_num / group_den) : scale0;
    }
    scale *= scale_mul;
    if (!(scale > 0.0f)) {
        scale = scale0;
    }
    const int qi = max(-128, min(127, int(lrintf(x / scale))));

    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
    const int packed_word = ggml_cuda_pack_k_packed16_i8x4_word_from_lanes(qi, lane);
    if ((lane & 3) == 0) {
        k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, int(cell), d / 4, layout_kind)] = packed_word;
        if (overlay_valid) {
            k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, overlay_cell, d / 4, layout_kind)] = packed_word;
        }
    }
    if (lane == 0) {
        const half hscale = __float2half(scale);
        k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, int(cell), q_block, layout_kind)] = hscale;
        if (overlay_valid) {
            k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, overlay_cell, q_block, layout_kind)] = hscale;
        }
    }
}

static __global__ __launch_bounds__(128, 1) void ggml_cuda_mtp_qblock_txn_tail_k_copy_page_to_scratch_kernel(
        const int  * __restrict__ k_payload,
        const half * __restrict__ k_scales,
        int        * __restrict__ scratch_payload,
        half       * __restrict__ scratch_scales,
        int page_base,
        int kv_size,
        int n_heads_k,
        int layout_kind) {
    const int tid = threadIdx.x;
    const int slot = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    if (slot >= MTP_V4_144_PAGE_TOKENS || hk >= n_heads_k) {
        return;
    }

    const int token = page_base + slot;
    if (token < 0 || token >= kv_size) {
        return;
    }
    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
    const size_t scratch_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(MTP_V4_144_PAGE_TOKENS) + size_t(slot);

    for (int d_word = tid; d_word < GGML_CUDA_PACKED16_K_WORDS; d_word += blockDim.x) {
        scratch_payload[scratch_row * size_t(GGML_CUDA_PACKED16_K_WORDS) + size_t(d_word)] =
            k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, token, d_word, layout_kind)];
    }
    for (int qblk = tid; qblk < GGML_CUDA_PACKED16_K_QBLOCKS; qblk += blockDim.x) {
        scratch_scales[scratch_row * size_t(GGML_CUDA_PACKED16_K_QBLOCKS) + size_t(qblk)] =
            k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, token, qblk, layout_kind)];
    }
}

template <typename idx_t>
static __global__ __launch_bounds__(32, 1) void ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel(
        idx_t * __restrict__ local_idxs,
        int nk_cur,
        int slot_begin) {
    const int i = threadIdx.x;
    if (i < nk_cur) {
        local_idxs[i] = (idx_t) (slot_begin + i);
    }
}

static __global__ __launch_bounds__(128, 1) void ggml_cuda_mtp_qblock_txn_tail_k_commit_scratch_page_kernel(
        const int  * __restrict__ scratch_payload,
        const half * __restrict__ scratch_scales,
        int        * __restrict__ k_payload,
        half       * __restrict__ k_scales,
        int page_base,
        int kv_size,
        int n_heads_k,
        int layout_kind) {
    const int tid = threadIdx.x;
    const int slot = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    if (slot >= MTP_V4_144_PAGE_TOKENS || hk >= n_heads_k) {
        return;
    }

    const int token = page_base + slot;
    if (token < 0 || token >= kv_size) {
        return;
    }
    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);
    const size_t scratch_row = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(MTP_V4_144_PAGE_TOKENS) + size_t(slot);

    for (int d_word = tid; d_word < GGML_CUDA_PACKED16_K_WORDS; d_word += blockDim.x) {
        k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, token, d_word, layout_kind)] =
            scratch_payload[scratch_row * size_t(GGML_CUDA_PACKED16_K_WORDS) + size_t(d_word)];
    }
    for (int qblk = tid; qblk < GGML_CUDA_PACKED16_K_QBLOCKS; qblk += blockDim.x) {
        k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, token, qblk, layout_kind)] =
            scratch_scales[scratch_row * size_t(GGML_CUDA_PACKED16_K_QBLOCKS) + size_t(qblk)];
    }
}

template <typename idx_t>
static __global__ __launch_bounds__(128, 1) void ggml_cuda_mtp_qblock_poison_canonical_k_indexed_kernel(
        int         * __restrict__ k_payload,
        half        * __restrict__ k_scales,
        const idx_t * __restrict__ k_idxs,
        int nk_cur,
        int n_heads_k,
        int batch,
        int kv_size,
        int layout_kind,
        int poison_min_idx,
        int poison_max_idx) {
    const int tid = threadIdx.x;
    const int k_local = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    if (k_local >= nk_cur || hk >= n_heads_k || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) k_idxs[k_local];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    if ((poison_min_idx > 0 && cell < poison_min_idx) || (poison_max_idx >= 0 && cell >= poison_max_idx)) {
        return;
    }
    const size_t head_base = (size_t(b) * size_t(n_heads_k) + size_t(hk)) * size_t(kv_size);

    for (int d_word = tid; d_word < GGML_CUDA_PACKED16_K_WORDS; d_word += blockDim.x) {
        k_payload[ggml_cuda_packed16_k_payload_index(head_base, kv_size, cell, d_word, layout_kind)] = 0x7f7f7f7f;
    }
    for (int qblk = tid; qblk < GGML_CUDA_PACKED16_K_QBLOCKS; qblk += blockDim.x) {
        k_scales[ggml_cuda_packed16_k_scale_index(head_base, kv_size, cell, qblk, layout_kind)] = __float2half(16.0f);
    }
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];

    // ── Launch proof: DOT4 actually dispatched ───────────────────
    if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
        if (log_env && atoi(log_env) != 0) {
            GGML_LOG_INFO(
                "fa_dot4_launch: fa_inst=%d nq=%lld nk=%lld d=%lld K=%s k_phys=%s V=%s\n",
                fa_inst_i32,
                (long long) Q->ne[1],
                (long long) K->ne[1],
                (long long) Q->ne[0],
                ggml_type_name(K->type),
                ggml_cuda_q8k_dot4_k_physical_name(K),
                ggml_type_name(V->type));
        }
    }

    const bool v4_144_nomtp_decode =
        V->type == GGML_TYPE_V4_K16D16_144 &&
        (fa_inst_i32 == GGML_FATTN_INST_NONE || fa_inst_i32 == GGML_FATTN_INST_DECODE_QK);
    const bool v4_144_pv4_standard = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PROFILE") ||
        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PV4");
    const bool v4_144_decode_diag_env =
        V->type == GGML_TYPE_V4_K16D16_144 &&
        (v4_144_nomtp_decode || v4_144_pv4_standard ||
         ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT"));

    // I32 packed16 K contract: DOT4-dispatch path only.
    const bool k_is_i32_packed16 = K->type == GGML_TYPE_I32;

    if (k_is_i32_packed16) {
        GGML_ASSERT(Q->type == GGML_TYPE_F32);
        GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 || v4_144_decode_diag_env);
        if (V->type != GGML_TYPE_F16) {
            GGML_ASSERT(k_is_i32_packed16 && "quantized V with DOT4 FA requires packed16 K (I32 type)");
        }
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(K->ne[0] * 4 == Q->ne[0]);  // D/4 * 4 == D
        GGML_ASSERT(V->ne[0] == Q->ne[0]);
        GGML_ASSERT(K->ne[1] > 0);
        GGML_ASSERT(K->ne[2] > 0);
        GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
        GGML_ASSERT(K->data != nullptr);
        GGML_ASSERT(V->data != nullptr);
        GGML_ASSERT(dst->data != nullptr);
    }

    const int nq = (int) Q->ne[1];
    const int nk = (int) K->ne[1];
    const int n_heads_q = (int) Q->ne[2];
    const int n_heads_k = (int) K->ne[2];
    const int batch = (int) Q->ne[3];
    const int gqa_ratio = n_heads_q / n_heads_k;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int>   q_payload(pool);
    ggml_cuda_pool_alloc<float> q_scales(pool);
    ggml_cuda_pool_alloc<int>   k_payload(pool);
    ggml_cuda_pool_alloc<half>  k_scales(pool);
    ggml_cuda_pool_alloc<float> logits(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_o(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_m(pool);
    ggml_cuda_pool_alloc<float> blockfa_partial_l(pool);
    ggml_cuda_pool_alloc<float> ref_logits(pool);
    ggml_cuda_pool_alloc<float> metrics(pool);

    const size_t q_rows = (size_t) batch * n_heads_q * nq;
    const size_t k_rows = (size_t) batch * n_heads_k * nk;
    const size_t logits_ne = (size_t) batch * n_heads_q * nq * nk;
    const char * check_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_CHECK");
    const bool check = check_env && atoi(check_env) != 0;
    const char * variant_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT");
    const bool blockfa_recthist_v4_single = variant_env && strcmp(variant_env, "blockfa_recthist_v4_single") == 0;
    if (variant_env && variant_env[0] != '\0' && strcmp(variant_env, "packed16") != 0 && !blockfa_recthist_v4_single) {
        GGML_ABORT("unsupported q8k_dot4_kq variant '%s'; supported variants: packed16, blockfa_recthist_v4_single", variant_env);
    }
    const bool blockfa_runtime_any_explicit = blockfa_recthist_v4_single;
    const bool full_fa_env = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA");
    // Packed16 scalar decode is a real attention lane, not the legacy KQ-only
    // probe. Auto-enable FULL_FA for the exact persistent packed16/I32 + q4_0
    // scalar decode contracts so restored no-MTP decode cannot zero dst.  The
    // V4_144 diagnostic adapter must pass through the same gate or the selector
    // can say rocm_packed16_decode while compute falls back to KQ-only zero output.
    const bool auto_full_fa_packed16_decode =
        !full_fa_env &&
        (fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK ||
         fa_inst_i32 == GGML_FATTN_INST_DECODE_QK ||
         fa_inst_i32 == GGML_FATTN_INST_NONE) &&
        Q->ne[1] == 1 &&
        K->type == GGML_TYPE_I32 &&
        (V->type == GGML_TYPE_Q4_0 || v4_144_decode_diag_env);
    const bool full_fa = full_fa_env || auto_full_fa_packed16_decode;

    // Auto-select blockfa_recthist_v4_single for f16/q8_0/q4_0 V when FULL_FA=1
    // and no explicit variant is set. The default packed16 path for f16/q8_0 would
    // read V data as q4_0 blocks (incorrect); and the default q4_0 path uses the
    // legacy fattn_from_logits kernel which has poor occupancy at scale.
    // V4_144 is only included for nq==1 diagnostic decode so it reaches the
    // packed16 decode fast-path below and returns before any q4_0 blockFA fallback.
    const bool blockfa_recthist_v4_single_auto =
        full_fa && !blockfa_runtime_any_explicit &&
        (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0 ||
         (v4_144_decode_diag_env && Q->ne[1] == 1));

    const bool blockfa_recthist_v4_single_effective = blockfa_recthist_v4_single || blockfa_recthist_v4_single_auto;
    const bool blockfa_recthist_any_effective = blockfa_recthist_v4_single_effective;
    const bool blockfa_runtime_any_effective = blockfa_recthist_any_effective;
    const bool fused_variant_effective = blockfa_runtime_any_effective;

    const char * variant_name = blockfa_recthist_v4_single_auto && !blockfa_recthist_v4_single ? "blockfa_recthist_v4_single_auto" :
            (blockfa_recthist_v4_single ? "blockfa_recthist_v4_single" : "packed16");
    const char * variant_note = blockfa_recthist_v4_single_effective ? "blockfa_recthist_v4_single" :
            (full_fa ? "full_fa_from_logits" : "kq_only_diagnostic_zero_output");
    if (ggml_cuda_q8k_dot4_consumer_read_mode_trace_enabled() && k_is_i32_packed16) {
        fprintf(stderr,
                "MTP_QBLOCK_CONSUMER_READ_MODE: backend=rocm_q8k_dot4_kq read_mode=canonical node=%s layer=%d graph_inst=%d variant=%s note=%s nq=%d nk=%d hq=%d hk=%d gqa_ratio=%d K=%s k_phys=%s V=%s full_fa=%d blockfa=%d v4_144_decode=%d\n",
                ggml_cuda_q8k_dot4_fattn_node_name(dst),
                ggml_cuda_q8k_dot4_fattn_layer_from_node_name(dst),
                fa_inst_i32,
                variant_name,
                variant_note,
                nq,
                nk,
                n_heads_q,
                n_heads_k,
                gqa_ratio,
                ggml_type_name(K->type),
                ggml_cuda_q8k_dot4_k_physical_name(K),
                ggml_type_name(V->type),
                full_fa ? 1 : 0,
                blockfa_recthist_any_effective ? 1 : 0,
                v4_144_decode_diag_env ? 1 : 0);
    }
    if (fused_variant_effective && !full_fa) {
        GGML_ABORT("q8k_dot4_kq blockFA variant requires GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1");
    }

    if (full_fa) {
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
        if (sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
            GGML_ABORT("q8k_dot4_kq full FA path does not support sinks, max_bias, or logit_softcap");
        }
        if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
                mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
            GGML_ABORT("q8k_dot4_kq full FA path mask shape/type unsupported");
        }
        if (blockfa_runtime_any_effective && mask && getenv("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL") == nullptr) {
            // Auto-selected blockfa_recthist_v4_single: assume causal mask is
            // correctly supplied (standard attention with causal mask).
            if (blockfa_recthist_v4_single_auto) {
                // OK — auto path implies causal mask is intentional.
            } else {
                GGML_ABORT("q8k_dot4_kq blockFA variants require no mask or GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1");
            }
        }
    }

    const bool timing_requested = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING");
    const int dot4_debug = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_DOT4_DEBUG") ? 1 : 0;
    bool stream_is_capturing = false;
#ifdef USE_CUDA_GRAPH
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &capture_status));
    stream_is_capturing = capture_status != hipStreamCaptureStatusNone;
#endif // USE_CUDA_GRAPH
    const int timing_every = max(1, ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_KQ_TIMING_EVERY", 1));
    static unsigned long long timing_counter = 0;
    const unsigned long long timing_call = timing_requested ? ++timing_counter : 0;
    const bool timing = timing_requested && !stream_is_capturing && (timing_call % (unsigned long long) timing_every) == 0;

    q_payload.alloc(q_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
    q_scales.alloc(q_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    // Persistent packed16 K cache: hipMalloc once per K tensor, reuse across passes.
    // Skips re-pack when k_rows hasn't grown since last repack.
    static std::mutex s_cache_mutex;
    static std::unordered_map<const void *, std::pair<int *, half *>> s_cache;
    static std::unordered_map<const void *, size_t> s_cache_rows;  // k_rows at last repack
    const bool use_packed16 = ggml_cuda_q8k_dot4_packed16_k_cache_enabled();
    if (k_is_i32_packed16 && !use_packed16) {
        GGML_ABORT("q8k_dot4_kq packed16-q8 side-channel K requires packed16 K cache sidecars; K=%s physical=%s",
                ggml_type_name(K->type), ggml_cuda_q8k_dot4_k_physical_name(K));
    }
    bool skip_k_repack = false;
    if (use_packed16) {
        // Prefer GGML tensors from registry (allocated by KV cache). Fall back to hipMalloc.
        ggml_tensor * payload_tensor = nullptr;
        ggml_tensor * scales_tensor  = nullptr;
        llama_kv_cache_get_packed16_tensors(K->data, &payload_tensor, &scales_tensor);
        // Packed16 I32 K MUST have registry entries (pre-quantized in cache)
        // with flat physical rows: row = hk * head_stride + k.
        if (k_is_i32_packed16 && !ggml_cuda_q8k_dot4_packed16_sidecar_valid(
                    K, payload_tensor, scales_tensor, true, "launch")) {
            GGML_ABORT("q8k_dot4_kq packed16 sidecar contract failed for physical I32+F16-scale K");
        }
        if (payload_tensor && scales_tensor) {
            // Use GGML tensors directly — no hipMalloc needed.
            k_payload.ptr = (int *) payload_tensor->data;
            k_scales.ptr  = (half *) scales_tensor->data;
            // One-time DOT4 packed16 registry dump
            static bool dot4_registry_printed = false;
            if (!dot4_registry_printed) {
                dot4_registry_printed = true;
                const int64_t head_stride = (payload_tensor->ne[1] > 0 && K->ne[2] > 0) ? payload_tensor->ne[1] / K->ne[2] : 0;
                fprintf(stderr, "DOT4 packed16 registry: payload=%p scales=%p "
                        "payload_ne=(%lld,%lld,%lld,%lld) scales_ne=(%lld,%lld,%lld,%lld) "
                        "physical=packed16_q8_sidechannel_i32_f16scales head_stride=%lld row=hk*head_stride+k\n",
                        (void*)payload_tensor->data, (void*)scales_tensor->data,
                        (long long)payload_tensor->ne[0], (long long)payload_tensor->ne[1],
                        (long long)payload_tensor->ne[2], (long long)payload_tensor->ne[3],
                        (long long)scales_tensor->ne[0], (long long)scales_tensor->ne[1],
                        (long long)scales_tensor->ne[2], (long long)scales_tensor->ne[3],
                        (long long)head_stride);
            }
            // Check if cache unchanged since last repack.
            std::lock_guard<std::mutex> lock(s_cache_mutex);
            size_t & prev_rows = s_cache_rows[K->data];
            if (prev_rows >= (size_t)k_rows) {
                skip_k_repack = true;
            }
        } else {
            // Fallback: hipMalloc persistent buffers.
            const size_t need_payload = (size_t)k_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4);
            const size_t need_scales  = (size_t)k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS;
            std::lock_guard<std::mutex> lock(s_cache_mutex);
            auto & e = s_cache[K->data];
            size_t & prev_rows = s_cache_rows[K->data];
            if (e.first == nullptr || prev_rows < (size_t)k_rows) {
                if (e.first && prev_rows < (size_t)k_rows) {
                    CUDA_CHECK(hipFree(e.first));
                    CUDA_CHECK(hipFree(e.second));
                }
                CUDA_CHECK(hipMalloc(&e.first,  need_payload * sizeof(int)));
                CUDA_CHECK(hipMalloc(&e.second, need_scales  * sizeof(half)));
            } else {
                skip_k_repack = true;
            }
            k_payload.ptr = e.first;
            k_scales.ptr  = e.second;
        }
    } else {
        k_payload.alloc(k_rows * (GGML_CUDA_Q8K_DOT4_KQ_D / 4));
        k_scales.alloc(k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);
    }
    if (!fused_variant_effective) {
        logits.alloc(logits_ne);
    }
    const char * blockfa_split_k_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_SPLIT_K");
    const int blockfa_split_k_requested = blockfa_split_k_env ? atoi(blockfa_split_k_env) : 1;
    const int blockfa_split_k = blockfa_runtime_any_effective ? max(1, blockfa_split_k_requested) : 1;
    const int blockfa_bn = blockfa_runtime_any_effective ? ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BN", 8) : 8;
    const int blockfa_q_offset = (full_fa && blockfa_runtime_any_effective && mask) ? max(0, nk - nq) : 0;
    const int blockfa_recthist_prefix_k = blockfa_recthist_any_effective ? blockfa_q_offset : 0;
    const int blockfa_recthist_tail_k = blockfa_recthist_any_effective ? nk - blockfa_q_offset : 0;
    if (blockfa_recthist_any_effective) {
        if (blockfa_split_k_requested != 1) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single supports splitK=1 only, got %d", blockfa_split_k_requested);
        }
        if (blockfa_bn != 8 && blockfa_bn != 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires BN8 or BN16, got %d", blockfa_bn);
        }
        const int v4_bm = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BM", 8);
        if (v4_bm != 8 && v4_bm != 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires BM8 or BM16, got %d", v4_bm);
        }
        if (v4_bm == 16 && blockfa_bn == 16) {
            GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single BM16+BN16 not supported (no idle threads for V prefetch)");
        }
    }

    cudaStream_t stream = ctx.stream();
    dim3 q_grid(nq, n_heads_q, batch);
    dim3 block(256);

    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_q_quant = nullptr;
    hipEvent_t ev_k_pack = nullptr;
    hipEvent_t ev_kq = nullptr;
    hipEvent_t ev_ref = nullptr;
    hipEvent_t ev_err = nullptr;
    hipEvent_t ev_blockfa = nullptr;
    hipEvent_t ev_zero = nullptr;
    if (timing) {
        ggml_cuda_q8k_dot4_kq_event_create(&ev_start);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_kq);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_ref);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_err);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_create(&ev_zero);
        CUDA_CHECK(hipEventRecord(ev_start, stream));
    }

    ggml_cuda_q8k_dot4_quant_q_packed16_kernel<<<q_grid, block, 0, stream>>>(
        (const float *) Q->data, q_payload.ptr, q_scales.ptr,
        Q->nb[1], Q->nb[2], Q->nb[3], nq, n_heads_q, batch);
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_q_quant, stream));
    }

    {
        if (!skip_k_repack) {
            // Packed16 I32 K is pre-quantized; skip repack and use registry data.
            const bool k_is_already_packed16 = (K->type == GGML_TYPE_I32);
            if (k_is_already_packed16) {
                skip_k_repack = true; // already in k_payload.ptr from registry above
            } else if (K->type == GGML_TYPE_F16) {
                // Direct f16→packed16 quantization (one kernel, no intermediate)
                dim3 k_quant_grid(nk, n_heads_k, batch);
                ggml_cuda_q8k_dot4_quant_k_packed16_kernel<<<k_quant_grid, block, 0, stream>>>(
                    (const half *) K->data, k_payload.ptr, k_scales.ptr,
                    K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // q8_0→packed16 repack (existing path)
                dim3 pack_grid((k_rows * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS + 255) / 256);
                ggml_cuda_q8k_dot4_pack_k_packed16_kernel<<<pack_grid, block, 0, stream>>>(
                    (const char *) K->data, k_payload.ptr, k_scales.ptr,
                    K->nb[0], K->nb[1], K->nb[2], K->nb[3], nk, n_heads_k, batch);
                CUDA_CHECK(cudaGetLastError());
            }
            if (use_packed16) {
                std::lock_guard<std::mutex> lock(s_cache_mutex);
                s_cache_rows[K->data] = (size_t)k_rows;
            }
        }
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_k_pack, stream));
        }
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    dim3 kq_grid((nk + GGML_CUDA_Q8K_DOT4_KQ_TILE_N - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_N,
                 (nq + GGML_CUDA_Q8K_DOT4_KQ_TILE_M - 1) / GGML_CUDA_Q8K_DOT4_KQ_TILE_M,
                 n_heads_q * batch);
    if (!fused_variant_effective) {
        ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, logits.ptr, scale,
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_kq, stream));
    }

    if (check && !fused_variant_effective) {
        ref_logits.alloc(logits_ne);
        metrics.alloc(3);
        CUDA_CHECK(cudaMemsetAsync(metrics.ptr, 0, 3 * sizeof(float), stream));
        ggml_cuda_q8k_dot4_kq_reference_kernel<<<kq_grid, block, 0, stream>>>(
            (const float *) Q->data, (const char *) K->data, ref_logits.ptr, scale,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[0], K->nb[1], K->nb[2], K->nb[3],
            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_ref, stream));
        }
        ggml_cuda_q8k_dot4_kq_error_kernel<<<(logits_ne + 255) / 256, 256, 0, stream>>>(
            logits.ptr, ref_logits.ptr, metrics.ptr, logits_ne);
        CUDA_CHECK(cudaGetLastError());
        if (timing) {
            CUDA_CHECK(hipEventRecord(ev_err, stream));
        }
        float h_metrics[3] = {0.0f, 0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(h_metrics, metrics.ptr, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double rms = sqrt(double(h_metrics[0]) / double(logits_ne));
        const double mean_abs = double(h_metrics[1]) / double(logits_ne);
        GGML_LOG_INFO("%s: q8k_dot4_kq_check logits=%zu rms=%.6g mean_abs=%.6g gt0.5=%.0f\n",
            __func__, logits_ne, rms, mean_abs, double(h_metrics[2]));
    }

    if (full_fa) {
        dim3 fa_grid(nq, n_heads_q, batch);
        if (blockfa_runtime_any_effective) {
            const int q_offset = blockfa_q_offset;
            const bool use_causal = mask && (blockfa_recthist_v4_single_auto ||
                ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL"));
            if (blockfa_recthist_v4_single_effective) {
                if (!use_causal) {
                    GGML_ABORT("q8k_dot4_kq blockfa_recthist_v4_single requires causal-tail mask and GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1");
                }
                // Debug instrumentation for I32 packed16 contracts.
                static bool debug_i32_contract_printed = false;
                if (k_is_i32_packed16 && !debug_i32_contract_printed &&
                        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
                    debug_i32_contract_printed = true;
                    fprintf(stderr,
                        "q8k_dot4_i32: Q type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: K type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: V type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p\n"
                        "q8k_dot4_i32: payload=%p scales=%p nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d\n",
                        Q->type, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                        Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3], Q->data,
                        K->type, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3], K->data,
                        V->type, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        V->nb[0], V->nb[1], V->nb[2], V->nb[3], V->data,
                        k_payload.ptr, k_scales.ptr, nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio);
                }
                // Ensure all prior GPU work (pack_k on any stream) finished before FA reads registry data.
                const bool force_sync = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_FORCE_SYNC");
                if (force_sync) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                const int q_offset = blockfa_q_offset;
                const int v4_bn = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BN", 8);
                const int v4_bm = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_BM", 8);
                const size_t smem_v4 = (size_t(v4_bm * v4_bn + 3 * v4_bm + 2 * v4_bn * GGML_CUDA_Q8K_DOT4_KQ_D) + size_t(v4_bm * (GGML_CUDA_Q8K_DOT4_KQ_D / 4)) + size_t(v4_bm * GGML_CUDA_Q8K_DOT4_KQ_BLOCKS)) * sizeof(float);
                const dim3 recthist_grid((nq + v4_bm - 1) / v4_bm, n_heads_q, batch);
                const bool use_f16_v = (V->type == GGML_TYPE_F16);
                const bool use_q8_v  = (V->type == GGML_TYPE_Q8_0);
                // ---- Packed16 decode / small verify fast-path ----
                const int32_t fa_inst = ((const int32_t *)dst->op_params)[4];
                const bool is_mtp_draft_decode = (fa_inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK);
                const bool is_standard_decode =
                    fa_inst == GGML_FATTN_INST_DECODE_QK || fa_inst == GGML_FATTN_INST_NONE;
                const char * packed16_decode_impl_env = getenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
                const char * packed16_decode_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
                const bool packed16_small_verify_requested =
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_batched_splitk") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa2") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pvwmma") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv_dot4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_dot4") == 0) ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_wmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_batched_splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa2") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
                const bool packed16_small_verify_splitk_requested =
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_splitk") == 0);
                const bool packed16_decode_splitk_requested =
                    packed16_small_verify_splitk_requested ||
                    (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "splitk") == 0) ||
                    (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_splitk") == 0);
                const bool v4_144_nomtp_decode =
                    V->type == GGML_TYPE_V4_K16D16_144 && is_standard_decode;
                const bool v4_144_decode_diag =
                    V->type == GGML_TYPE_V4_K16D16_144 &&
                    (v4_144_nomtp_decode || v4_144_pv4_standard ||
                     ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT"));
                const bool use_default_packed16_decode =
                    nq == 1 && K->type == GGML_TYPE_I32 && (V->type == GGML_TYPE_Q4_0 || v4_144_decode_diag) &&
                    (is_mtp_draft_decode || is_standard_decode);
                const int decode_bn = ggml_cuda_q8k_dot4_kq_env_int(
                    "GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN", (use_default_packed16_decode || packed16_small_verify_requested || packed16_small_verify_splitk_requested) ? 64 : 0);
                const int decode_max_nq = (packed16_small_verify_requested || packed16_small_verify_splitk_requested) ?
                    ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ", 4) :
                    (packed16_decode_splitk_requested ? ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_MAX_NQ", 1) : 1);

                if (decode_bn > 0 && nq <= decode_max_nq && (K->type == GGML_TYPE_I32 || is_mtp_draft_decode)) {
                    // For f16-source K, packed16 is already materialized in k_payload/k_scales.
                    // Strides are computed from packed16 layout, not from K tensor strides.
                    const int k_head_stride_rows  = nk;
                    const int k_batch_stride_rows = n_heads_k * k_head_stride_rows;
                    const int decode_vsub = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_VSUB", 8);
                    const char * packed16_decode_impl =
                        (packed16_decode_impl_env && packed16_decode_impl_env[0]) ? packed16_decode_impl_env :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_q4pair") == 0) ? "q4pair" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_scalar") == 0) ? "gqa_scalar" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_waveqk") == 0) ? "waveqk" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_waveqk_q4pair") == 0) ? "waveqk_q4pair" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_pvwmma") == 0) ? "pvwmma" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_pvwmma") == 0) ? "gqa_pvwmma" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_wmma_full") == 0) ? "wmma_full" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_gqa_wmma_full") == 0) ? "gqa_wmma_full" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_dsplit") == 0) ? "dsplit" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_logits_debug") == 0) ? "logits_debug" :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify") == 0) ? (nq == 1 ? "splitk" : "small_verify") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_splitk") == 0) ? (nq == 1 ? "splitk" : "small_verify_splitk") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_batched_splitk") == 0) ? (nq == 1 ? "splitk" : "small_verify_batched_splitk") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa2") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa2") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0) ? (nq == 1 ? "splitk" : "small_verify_fa4_pvwmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pvwmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pint8pv") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_pint8pv_dot4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_intflash_vfrag_wmma") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0) ? (nq == 1 ? "splitk" : "bm_dot4_pages_intflash_vfrag_dot4") :
                        (packed16_decode_route && strcmp(packed16_decode_route, "rocm_packed16_decode_splitk") == 0) ? "splitk" :
                        "scalar";
                    const bool impl_gqa_scalar = strcmp(packed16_decode_impl, "gqa_scalar") == 0;
                    const bool impl_scalar = strcmp(packed16_decode_impl, "scalar") == 0;
                    const bool impl_inline_q4 = strcmp(packed16_decode_impl, "inline_q4") == 0;
                    const bool impl_q4pair = strcmp(packed16_decode_impl, "q4pair") == 0;
                    const bool impl_waveqk = strcmp(packed16_decode_impl, "waveqk") == 0 || strcmp(packed16_decode_impl, "gqa_waveqk") == 0;
                    const bool impl_waveqk_q4pair = strcmp(packed16_decode_impl, "waveqk_q4pair") == 0;
                    const bool impl_pvwmma = strcmp(packed16_decode_impl, "pvwmma") == 0 || strcmp(packed16_decode_impl, "gqa_pvwmma") == 0;
                    const bool impl_wmma_full = strcmp(packed16_decode_impl, "wmma_full") == 0 || strcmp(packed16_decode_impl, "gqa_wmma_full") == 0;
                    const bool impl_dsplit = strcmp(packed16_decode_impl, "dsplit") == 0;
                    const bool impl_logits_debug = strcmp(packed16_decode_impl, "logits_debug") == 0;
                    const bool impl_splitk = strcmp(packed16_decode_impl, "splitk") == 0;
                    bool impl_small_verify = strcmp(packed16_decode_impl, "small_verify") == 0;
                    bool impl_small_verify_splitk = strcmp(packed16_decode_impl, "small_verify_splitk") == 0;
                    bool impl_small_verify_batched_splitk = strcmp(packed16_decode_impl, "small_verify_batched_splitk") == 0;
                    bool impl_small_verify_fa2 = strcmp(packed16_decode_impl, "small_verify_fa2") == 0;
                    bool impl_small_verify_fa4 = strcmp(packed16_decode_impl, "small_verify_fa4") == 0;
                    bool impl_small_verify_fa4_pvwmma = strcmp(packed16_decode_impl, "small_verify_fa4_pvwmma") == 0;
                    bool impl_bm_dot4_pages = strcmp(packed16_decode_impl, "bm_dot4_pages") == 0;
                    bool impl_bm_dot4_pages_pvwmma = strcmp(packed16_decode_impl, "bm_dot4_pages_pvwmma") == 0;
                    bool impl_bm_dot4_pages_pint8pv = strcmp(packed16_decode_impl, "bm_dot4_pages_pint8pv") == 0;
                    bool impl_bm_dot4_pages_pint8pv_dot4 = strcmp(packed16_decode_impl, "bm_dot4_pages_pint8pv_dot4") == 0;
                    bool impl_bm_dot4_pages_intflash_vfrag_dot4 = strcmp(packed16_decode_impl, "bm_dot4_pages_intflash_vfrag_dot4") == 0;
                    bool impl_bm_dot4_pages_intflash_vfrag_wmma = strcmp(packed16_decode_impl, "bm_dot4_pages_intflash_vfrag_wmma") == 0;
                    const bool impl_supported = impl_scalar || impl_gqa_scalar || impl_inline_q4 || impl_q4pair || impl_waveqk || impl_waveqk_q4pair || impl_pvwmma || impl_wmma_full || impl_dsplit || impl_logits_debug || impl_splitk || impl_small_verify || impl_small_verify_splitk || impl_small_verify_batched_splitk || impl_small_verify_fa2 || impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma || impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma;
                    if (!impl_supported) {
                        GGML_ABORT("packed16 decode impl '%s' is not implemented yet; supported in this build: scalar, gqa_scalar, inline_q4, q4pair, waveqk, waveqk_q4pair, splitk, small_verify, small_verify_splitk, small_verify_batched_splitk, small_verify_fa2, small_verify_fa4, small_verify_fa4_pvwmma, bm_dot4_pages, bm_dot4_pages_pvwmma, bm_dot4_pages_pint8pv, bm_dot4_pages_pint8pv_dot4, bm_dot4_pages_intflash_vfrag_dot4, bm_dot4_pages_intflash_vfrag_wmma, dsplit, pvwmma, gqa_pvwmma, wmma_full, gqa_wmma_full, logits_debug", packed16_decode_impl);
                    }
                    const bool packed16_decode_impl_explicit = packed16_decode_impl_env && packed16_decode_impl_env[0];
                    const bool packed16_decode_log =
                        ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_PACKED16_DECODE_LOG") ||
                        ggml_cuda_q8k_dot4_kq_env_enabled("COMPRESSED_KV_FATTN_LOG");
                    if (packed16_decode_log) {
                        static std::unordered_set<std::string> packed16_decode_impl_seen;
                        std::string shape_key = std::string(packed16_decode_impl) + "_nq" + std::to_string(nq) + "_nk" + std::to_string(nk) + "_hq" + std::to_string(n_heads_q) + "_hk" + std::to_string(n_heads_k) + "_gqa" + std::to_string(gqa_ratio);
                        if (packed16_decode_impl_explicit || packed16_decode_impl_seen.find(shape_key) == packed16_decode_impl_seen.end()) {
                            packed16_decode_impl_seen.insert(shape_key);
                            fprintf(stderr,
                                "packed16_decode_impl selected=%s route=%s nq=%d nk=%d hq=%d hk=%d batch=%d gqa=%d V=%s\n",
                                packed16_decode_impl, packed16_decode_route ? packed16_decode_route : "",
                                nq, nk, n_heads_q, n_heads_k, batch, gqa_ratio, ggml_type_name(V->type));
                        }
                    }
                    const int decode_v_debug = ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG") ? 1 : 0;
                    const int decode_v_debug_hq = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_HQ", 0);
                    const int decode_v_debug_k  = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_K", 0);
                    const int decode_v_debug_d0 = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_D0", 0);
                    const int decode_v_debug_count = ggml_cuda_q8k_dot4_kq_env_int("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_V_DEBUG_COUNT", 8);
                    if (v4_144_decode_diag && !(impl_scalar || impl_splitk)) {
                        GGML_ABORT("V4_K16D16_144 q8k decode diagnostic currently supports packed16 decode impl=scalar or splitk, got %s", packed16_decode_impl);
                    }
                    const bool inline_q4 = !v4_144_decode_diag && (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_INLINE_Q4") || impl_inline_q4);
                    const bool q4pair = !v4_144_decode_diag && V->type == GGML_TYPE_Q4_0 &&
                        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_Q4PAIR") || impl_q4pair);
                    const int splitk_threshold = ggml_cuda_q8k_dot4_kq_env_int(
                        "GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD",
                        ggml_cuda_q8k_dot4_decode_stage_threshold(2048));
                    const int decode_splitk_size = ggml_cuda_q8k_dot4_decode_stage_split_size(nk, 512);
                    const bool v4_144_splitk = v4_144_decode_diag &&
                        (v4_144_nomtp_decode || v4_144_pv4_standard || impl_splitk || ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_V4_K16D16_144_DECODE_SPLITK"));
                    const bool splitk = !impl_logits_debug && !impl_dsplit &&
                        (ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK") || impl_splitk || impl_small_verify_splitk || nk >= splitk_threshold) &&
                        (V->type == GGML_TYPE_Q4_0 || v4_144_splitk);
                    if (splitk && ggml_cuda_q8k_dot4_kq_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_STAGE_LOG")) {
                        fprintf(stderr,
                            "Q8K_DECODE_STAGE_POLICY: auto=%d impl=%s nq=%d nk=%d V=%s threshold=%d split_size=%d explicit_impl=%d v4_splitk=%d\n",
                            ggml_cuda_q8k_dot4_decode_stage_auto_enabled() ? 1 : 0,
                            packed16_decode_impl, nq, nk, ggml_type_name(V->type), splitk_threshold, decode_splitk_size,
                            packed16_decode_impl_explicit ? 1 : 0, v4_144_splitk ? 1 : 0);
                    }
                    // small_verify/batched_splitk only apply for nq >= 2; nq == 1 falls through
                    // to regular decode (splitk/BN64/etc)
                    if (impl_small_verify && nq < 2) impl_small_verify = false;
                    if (impl_small_verify_splitk && nq < 2) impl_small_verify_splitk = false;
                    if (impl_small_verify_batched_splitk && nq < 2) impl_small_verify_batched_splitk = false;
                    if (impl_small_verify_fa2 && nq < 2) impl_small_verify_fa2 = false;
                    if (impl_small_verify_fa4 && nq < 2) impl_small_verify_fa4 = false;
                    if (impl_small_verify_fa4_pvwmma && nq < 2) impl_small_verify_fa4_pvwmma = false;
                    if (impl_bm_dot4_pages && nq < 2) impl_bm_dot4_pages = false;
                    if (impl_bm_dot4_pages_pvwmma && nq < 2) impl_bm_dot4_pages_pvwmma = false;
                    if (impl_bm_dot4_pages_pint8pv && nq < 2) impl_bm_dot4_pages_pint8pv = false;
                    if (impl_bm_dot4_pages_pint8pv_dot4 && nq < 2) impl_bm_dot4_pages_pint8pv_dot4 = false;
                    if (impl_bm_dot4_pages_intflash_vfrag_dot4 && nq < 2) impl_bm_dot4_pages_intflash_vfrag_dot4 = false;
                    if (impl_bm_dot4_pages_intflash_vfrag_wmma && nq < 2) impl_bm_dot4_pages_intflash_vfrag_wmma = false;
                    if ((impl_small_verify || impl_small_verify_splitk || impl_small_verify_batched_splitk || impl_small_verify_fa2 || impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma || impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma) && !(nq >= 2 && nq <= decode_max_nq && K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0)) {
                        GGML_ABORT("packed16 %s requires I32 K, q4_0 V, and 2 <= nq <= %d; got nq=%d K=%s V=%s",
                            packed16_decode_impl, decode_max_nq, nq, ggml_type_name(K->type), ggml_type_name(V->type));
                    }
                    if (impl_small_verify) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 small_verify supports gqa_ratio <= 8, got %d", gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SMALL_VERIFY(64, 8, 8) }
                        else { GGML_ABORT("packed16 small_verify: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_splitk) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 small_verify_splitk supports gqa_ratio <= 8, got %d", gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SMALL_VERIFY_SPLITK(64, 8, 8) }
                        else { GGML_ABORT("packed16 small_verify_splitk: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_batched_splitk) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 2, 512) }
                            else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 3, 512) }
                            else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 4, 512) }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_fa2) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            // Qwen 27B MTP verify is GQA6.  Specialize the qtile state to
                            // GH_MAX=6 instead of the generic GH_MAX=8 path: less LDS,
                            // fewer unrolled PV/softmax lanes, and lower register pressure
                            // without changing route semantics or adding another PV variant.
                            if (gqa_ratio == 6) {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 2, 224) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 3, 224) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 6, 4, 224) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 2, 112) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 3, 112) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_BATCHED_SPLITK(64, 8, 8, 4, 112) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_bm_dot4_pages || impl_bm_dot4_pages_pvwmma || impl_bm_dot4_pages_pint8pv || impl_bm_dot4_pages_pint8pv_dot4 || impl_bm_dot4_pages_intflash_vfrag_dot4 || impl_bm_dot4_pages_intflash_vfrag_wmma) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_vsub == 8) {
                            if (impl_bm_dot4_pages_pvwmma) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PVWMMA) }
                            } else if (impl_bm_dot4_pages_pint8pv) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PINT8PV) }
                            } else if (impl_bm_dot4_pages_pint8pv_dot4) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_PINT8PV_DOT4) }
                            } else if (impl_bm_dot4_pages_intflash_vfrag_dot4) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_DOT4) }
                            } else if (impl_bm_dot4_pages_intflash_vfrag_wmma) {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_INTFLASH_VFRAG_WMMA) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 2, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                                else if (nq <= 3) { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 3, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                                else { LAUNCH_DECODE_BM_DOT4_PAGES(128, 8, 8, 16, 4, BM_DOT4_PAGES_PV_IMPL_SCALAR) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected VSUB=8, got VSUB=%d", packed16_decode_impl, decode_vsub); }
                        return;
                    }
                    if (impl_small_verify_fa4 || impl_small_verify_fa4_pvwmma) {
                        if (gqa_ratio > 8) GGML_ABORT("packed16 %s supports gqa_ratio <= 8, got %d", packed16_decode_impl, gqa_ratio);
                        if (decode_bn == 64 && decode_vsub == 8) {
                            if (impl_small_verify_fa4_pvwmma) {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 2, 128, true) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 3, 128, true) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 4, 128, true) }
                            } else {
                                if (nq <= 2) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 2, 128, false) }
                                else if (nq <= 3) { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 3, 128, false) }
                                else { LAUNCH_DECODE_SMALL_VERIFY_FA4_BATCHED_SPLITK(64, 8, 8, 4, 128, false) }
                            }
                        }
                        else { GGML_ABORT("packed16 %s: expected BN=64 VSUB=8, got BN=%d VSUB=%d", packed16_decode_impl, decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_logits_debug) {
                        ggml_cuda_pool_alloc<float> decode_logits(pool);
                        decode_logits.alloc(logits_ne);
                        ggml_cuda_q8k_dot4_kq_kernel<<<kq_grid, block, 0, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, decode_logits.ptr, scale,
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
                        CUDA_CHECK(cudaGetLastError());
                        dim3 fa_grid(nq, n_heads_q, batch);
                        ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                            decode_logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
                        CUDA_CHECK(cudaGetLastError());
                        return;
                    }
                    if (splitk) {
                        const int split_size = decode_splitk_size;
                        const int n_splits = (nk + split_size - 1) / split_size;
                        blockfa_partial_o.alloc((size_t) batch * nq * n_heads_q * n_splits * 256);
                        blockfa_partial_m.alloc((size_t) batch * nq * n_heads_q * n_splits);
                        blockfa_partial_l.alloc((size_t) batch * nq * n_heads_q * n_splits);
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_SPLITK(64, 8) }
                        else { GGML_ABORT("q8k_dot4_kq split-K decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (impl_dsplit) {
                        if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 dsplit decode currently requires V=q4_0");
                        if (decode_bn == 64 && decode_vsub == 8) { LAUNCH_DECODE_DSPLIT(64, 8, 64) }
                        else { GGML_ABORT("q8k_dot4_kq D-split decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub); }
                        return;
                    }
                    if (decode_bn == 64 && decode_vsub == 8) {
                        if (impl_wmma_full) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 gqa_wmma_full decode currently requires V=q4_0");
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_wmma_full decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA_WMMA_FULL(64, 8)
                        } else if (impl_waveqk_q4pair) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 waveqk_q4pair decode currently requires V=q4_0");
                            LAUNCH_DECODE_WAVEQK_Q4PAIR(64, 8)
                        } else if (impl_waveqk) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 waveqk decode currently requires V=q4_0");
                            LAUNCH_DECODE_WAVEQK(64, 8)
                        } else if (impl_pvwmma) {
                            if (V->type != GGML_TYPE_Q4_0) GGML_ABORT("packed16 gqa_pvwmma decode currently requires V=q4_0");
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_pvwmma decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA_PVWMMA(64, 8)
                        } else if (impl_gqa_scalar) {
                            if (gqa_ratio > 8) GGML_ABORT("packed16 gqa_scalar decode supports gqa_ratio <= 8, got %d", gqa_ratio);
                            LAUNCH_DECODE_GQA(64, 8, 8)
                        } else if (q4pair)      LAUNCH_DECODE_Q4PAIR(64, 8)
                        else if (inline_q4) LAUNCH_DECODE_INLINE_Q4(64, 8)
                        else             LAUNCH_DECODE_VSUB(64, 8)
                    } else {
                        GGML_ABORT("q8k_dot4_kq decode: expected BN=64 VSUB=8, got BN=%d VSUB=%d", decode_bn, decode_vsub);
                    }
                    return;
                }
                // Fixed KV cache head strides for persistent packed16 K.
                GGML_ASSERT(K->nb[1] % sizeof(int) == 0);
                const int k_head_stride_rows  = (int)(K->nb[2] / K->nb[1]);
                const int k_batch_stride_rows = (int)(K->nb[3] / K->nb[1]);
                if (v4_bm == 16 && v4_bn == 8) {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 16><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                } else if (v4_bm == 8 && v4_bn == 16) {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 16, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                } else {
                    if (use_f16_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<true, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else if (use_q8_v) {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, true, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    } else {
                        ggml_cuda_q8k_dot4_blockfa_recthist_bm8_q4_0_single_kernel<false, false, 8, 8><<<recthist_grid, block, smem_v4, stream>>>(
                            q_payload.ptr, q_scales.ptr, k_payload.ptr, k_scales.ptr, (const char *) V->data,
                            (float *) dst->data, scale,
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch, q_offset, k_head_stride_rows, k_batch_stride_rows, dot4_debug);
                    }
                }
                if (timing) {
                    CUDA_CHECK(hipEventRecord(ev_blockfa, stream));
                }
            }
        } else {
            ggml_cuda_q8k_dot4_fattn_from_logits_q4_0_kernel<<<fa_grid, block, 0, stream>>>(
                logits.ptr, (const char *) V->data, mask ? (const char *) mask->data : nullptr, (float *) dst->data,
                V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0, mask ? mask->ne[3] : 1,
                nq, nk, n_heads_q, n_heads_k, gqa_ratio, batch);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Diagnostic KQ-only mode: deliberately leave FA output zeroed so this route
        // cannot masquerade as production attention. It exists only to validate
        // runtime packed-DOT4 KQ layout, route contracts, and ISA.
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }
    if (timing) {
        CUDA_CHECK(hipEventRecord(ev_zero, stream));
        CUDA_CHECK(hipEventSynchronize(ev_zero));
        const float q_quant_ms = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_q_quant);
        const float k_pack_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_q_quant, ev_k_pack);
        const float kq_ms      = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_k_pack, ev_kq);
        const float ref_ms     = check && !fused_variant_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_kq, ev_ref) : 0.0f;
        const float err_ms     = check && !fused_variant_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_ref, ev_err) : 0.0f;
        const float output_ms  = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(check && !fused_variant_effective ? ev_err : ev_kq, ev_zero);
        const float zero_ms    = full_fa ? 0.0f : output_ms;
        const float fa_ms      = full_fa ? output_ms : 0.0f;
        const float blockfa_ms = full_fa && blockfa_runtime_any_effective ? ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_kq, ev_blockfa) : 0.0f;
        const float total_ms   = ggml_cuda_q8k_dot4_kq_event_elapsed_ms(ev_start, ev_zero);
        GGML_LOG_INFO("%s: q8k_dot4_kq_timing call=%llu variant=%s nq=%d nk=%d heads_q=%d heads_k=%d batch=%d check=%d q_quant_ms=%.6f k_pack_ms=%.6f kq_ms=%.6f ref_ms=%.6f err_ms=%.6f zero_ms=%.6f total_ms=%.6f fa_ms=%.6f blockfa_ms=%.6f q_offset=%d prefix_k=%d tail_k=%d full_fa=%d blockfa_split_k=%d blockfa_bn=%d\n",
            __func__, timing_call, variant_name, nq, nk, n_heads_q, n_heads_k, batch, check && !fused_variant_effective ? 1 : 0,
            q_quant_ms, k_pack_ms, kq_ms, ref_ms, err_ms, zero_ms, total_ms, fa_ms, blockfa_ms,
            blockfa_recthist_v4_single_effective ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k,
            full_fa ? 1 : 0, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_start);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_q_quant);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_k_pack);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_kq);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_ref);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_err);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_blockfa);
        ggml_cuda_q8k_dot4_kq_event_destroy(ev_zero);
    }

    const char * log_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_LOG");
    if (log_env && atoi(log_env) != 0) {
        const double q_mib = double(q_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double k_mib = double(k_rows * GGML_CUDA_Q8K_DOT4_KQ_D) / (1024.0 * 1024.0);
        const double logits_mib = fused_variant_effective ? 0.0 : double(logits_ne * sizeof(float)) / (1024.0 * 1024.0);
        GGML_LOG_INFO("%s: route=rocm_q8k_dot4_kq variant=%s full_fa=%d nq=%d nk=%d heads_q=%d heads_k=%d batch=%d q_payload=%.3fMiB k_payload=%.3fMiB logits=%.3fMiB note=%s blockfa_split_k=%d blockfa_bn=%d q_offset=%d prefix_k=%d tail_k=%d\n",
            __func__, variant_name, full_fa ? 1 : 0, nq, nk, n_heads_q, n_heads_k, batch, q_mib, k_mib, logits_mib, variant_note, blockfa_runtime_any_effective ? blockfa_split_k : 0, blockfa_runtime_any_effective ? blockfa_bn : 0,
            blockfa_recthist_v4_single_effective ? blockfa_q_offset : 0, blockfa_recthist_prefix_k, blockfa_recthist_tail_k);
    }

    // Detach hipMalloc'd persistent buffers from pool allocator destructors.
    if (ggml_cuda_q8k_dot4_packed16_k_cache_enabled()) {
        k_payload.ptr = nullptr;
        k_scales.ptr  = nullptr;
    }

    GGML_UNUSED(sinks);
}


// ggml OP_PACK_K_PACKED16 backend: quantize Kcur -> packed16 (I32 payload + F16 scales).
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k_cur   = dst->src[0];  // source K rows
    ggml_tensor * scales  = dst->src[1];  // F16 scales output
    ggml_tensor * k_idxs  = dst->src[2];  // row indices (absolute cache slots)
    ggml_tensor * payload = dst;          // I32 payload (dst is view of payload)

    GGML_ASSERT(k_idxs != nullptr);
    GGML_ASSERT(payload->type == GGML_TYPE_I32);
    GGML_ASSERT(scales->type == GGML_TYPE_F16);
    const bool payload_is_packed16 = payload->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 4;
    const bool payload_is_packed8  = payload->ne[0] == GGML_CUDA_Q8K_DOT4_KQ_D / 8;
    GGML_ASSERT(payload_is_packed16 || payload_is_packed8);
    GGML_ASSERT(scales->ne[0]  == GGML_CUDA_Q8K_DOT4_KQ_BLOCKS);

    const int D = GGML_CUDA_Q8K_DOT4_KQ_D;
    const int batch = (int) k_cur->ne[3];

    int n_heads = 0;
    int nk_cur  = 0;
    int64_t src_head_stride_bytes = 0;

    if (k_cur->ne[0] == D) {
        // Split-head source: [D, n_heads, nk_cur, batch]
        n_heads = (int) k_cur->ne[1];
        nk_cur  = (int) k_cur->ne[2];
        src_head_stride_bytes = k_cur->nb[1];
    } else {
        // Combined GQA source: [D * n_heads, nk_cur, 1, batch]
        GGML_ASSERT(k_cur->ne[0] % D == 0);
        n_heads = (int) (k_cur->ne[0] / D);
        nk_cur  = (int) k_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(k_cur->type);
    }

    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(payload->ne[1] % n_heads == 0);

    const int kv_size = (int) (payload->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);

    const bool is_q8_k = k_cur->type == GGML_TYPE_Q8_0;
    if (is_q8_k) {
        // Active token count for q8_0 source comes from row indices.
        nk_cur = (int) k_idxs->ne[0];
        GGML_ASSERT(nk_cur <= kv_size);
    }

    const int64_t src_token_stride_bytes = (k_cur->ne[0] == D)
        ? k_cur->nb[2]
        : k_cur->nb[1];

    const int k_scale_mode = ggml_cuda_pack_k_packed16_scale_mode(1);
    const float k_scale_mul = ggml_cuda_pack_k_packed16_scale_mul();
    const int k_scale_group_qblocks = ggml_cuda_pack_k_packed16_scale_group_qblocks();
    const int packed8_q4_scale_mode = ggml_cuda_pack_k_packed8_q4_scale_mode();
    const float packed8_q4_scale_mul = ggml_cuda_pack_k_packed8_q4_scale_mul();
    const int requested_layout = ggml_cuda_packed16_k_layout_kind_from_env();
    const int k_format = payload_is_packed8 ? GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144 : GGML_CUDA_PDMQ_K_FORMAT_PACKED16_Q8_272;
    const int effective_layout = payload_is_packed8 ? GGML_CUDA_PACKED16_K_LAYOUT_ROW : requested_layout;
    if (payload_is_packed16 && effective_layout == GGML_CUDA_PACKED16_K_LAYOUT_PAGE16_D16 && (kv_size % GGML_CUDA_PACKED16_K_PAGE16_TOKENS) != 0) {
        GGML_ABORT("packed16 page16_d16 layout requires kv_capacity multiple of %d, got %d", GGML_CUDA_PACKED16_K_PAGE16_TOKENS, kv_size);
    }
    cudaStream_t stream = ctx.stream();

    const bool txn_tail_backend_trace = ggml_cuda_mtp_qblock_txn_tail_backend_proof_enabled();
    const bool txn_tail_page_request = ggml_cuda_mtp_qblock_txn_tail_page_requested();
    const bool txn_tail_backend_probe = txn_tail_backend_trace || txn_tail_page_request;
    ggml_cuda_packed16_sidecar_meta txn_meta = {};
    uint32_t txn_row_bytes = payload_is_packed16 ? MTP_PACKED16_K_ROW_BYTES : 0u;
    mtp_v4_144_tail_stage_status txn_geom_status = MTP_V4_144_TAIL_STAGE_BAD_KIND;
    ggml_cuda_mtp_qblock_txn_tail_backend_idx_probe txn_idx_probe = {};
    bool txn_k_page_merge_active = false;

    if (txn_tail_backend_probe && nk_cur > 1 && nk_cur <= 8) {
        txn_meta = ggml_cuda_make_pdmq_k_sidecar_meta(payload, scales, k_format, effective_layout, (uint32_t) kv_size, (uint32_t) D, 0);
        const dp16_packed_i8_desc_v1 & desc = txn_meta.packed_i8_desc;
        txn_geom_status = ggml_cuda_mtp_qblock_txn_tail_backend_geom_status(
            payload_is_packed16 ? MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K : MTP_V4_144_TAIL_STAGE_KIND_NONE,
            (uint32_t) kv_size,
            txn_row_bytes);
        txn_idx_probe = ggml_cuda_mtp_qblock_txn_tail_backend_probe_indices(
            k_idxs,
            nk_cur,
            kv_size,
            payload_is_packed16 ? MTP_V4_144_TAIL_STAGE_KIND_PACKED16_K : MTP_V4_144_TAIL_STAGE_KIND_NONE,
            txn_row_bytes,
            stream);
        txn_k_page_merge_active = txn_tail_page_request && payload_is_packed16 && !is_q8_k &&
            txn_idx_probe.copied && txn_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK && txn_idx_probe.contiguous &&
            txn_idx_probe.capacity_ok && !txn_idx_probe.spans_pages && txn_idx_probe.merge_copy_slots != 0 &&
            k_cur->type == GGML_TYPE_F32 &&
            (k_idxs->type == GGML_TYPE_I64 || k_idxs->type == GGML_TYPE_I32);
        if (txn_tail_backend_trace) {
            fprintf(stderr,
            "MTP_QBLOCK_TXN_TAIL_BACKEND: kind=K active=%d eligible=%d geom_status=%u idx_status=%u idx_probe=%d idx_capture_skipped=%d idx_contiguous=%d idx_capacity_ok=%d idx0=%u idx_last=%u page=[%u,%u) slots=[%u,%u) merge_copy_slots=%u spans_pages=%d nk_cur=%d n_heads=%d batch=%d kv_size=%d idx_type=%s src_type=%s format=%s layout=%s payload_ne=(%lld,%lld,%lld,%lld) scales_ne=(%lld,%lld,%lld,%lld) row_bytes=%u page_bytes=%u payload_words_per_token=%u qblocks=%u payload_token_stride_words=%zu payload_head_stride_words=%zu scale_token_stride_halfs=%zu scale_head_stride_halfs=%zu desc_layout=%u desc_scale_layout=%u desc_y_stride=%llu desc_scale_y_stride=%llu src_token_stride=%lld src_head_stride=%lld source_note=%s\n",
            txn_k_page_merge_active ? 1 : 0,
            (txn_geom_status == MTP_V4_144_TAIL_STAGE_OK && payload_is_packed16 && (txn_idx_probe.capture_skipped || (txn_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK && txn_idx_probe.contiguous))) ? 1 : 0,
            (unsigned) txn_geom_status,
            (unsigned) txn_idx_probe.status,
            txn_idx_probe.copied ? 1 : 0,
            txn_idx_probe.capture_skipped ? 1 : 0,
            txn_idx_probe.contiguous ? 1 : 0,
            txn_idx_probe.capacity_ok ? 1 : 0,
            txn_idx_probe.idx0,
            txn_idx_probe.idx_last,
            txn_idx_probe.page_base,
            txn_idx_probe.page_end,
            txn_idx_probe.slot_begin,
            txn_idx_probe.slot_end_excl,
            txn_idx_probe.merge_copy_slots,
            txn_idx_probe.spans_pages ? 1 : 0,
            nk_cur,
            n_heads,
            batch,
            kv_size,
            k_idxs->type == GGML_TYPE_I64 ? "i64" : "i32",
            ggml_type_name(k_cur->type),
            ggml_cuda_pdmq_k_format_name(k_format),
            ggml_cuda_packed16_k_layout_kind_name(effective_layout),
            (long long) payload->ne[0], (long long) payload->ne[1], (long long) payload->ne[2], (long long) payload->ne[3],
            (long long) scales->ne[0], (long long) scales->ne[1], (long long) scales->ne[2], (long long) scales->ne[3],
            txn_row_bytes,
            txn_row_bytes * MTP_V4_144_PAGE_TOKENS,
            txn_meta.payload_words_per_token,
            (uint32_t) (D / QK8_0),
            txn_meta.payload_token_stride,
            txn_meta.payload_head_stride,
            txn_meta.scale_token_stride,
            txn_meta.scale_head_stride,
            (unsigned) desc.layout_kind,
            (unsigned) desc.scale_layout,
            (unsigned long long) desc.y_stride_bytes,
            (unsigned long long) desc.scale_y_stride_bytes,
            (long long) src_token_stride_bytes,
            (long long) src_head_stride_bytes,
            is_q8_k ? "q8_shadow_source" : "f16_or_f32_source");
        if (txn_idx_probe.copied && payload_is_packed16 && txn_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK) {
            const uint32_t first_slot = txn_idx_probe.slot_begin;
            const uint32_t last_slot = txn_idx_probe.slot_end_excl - 1u;
            const uint32_t first_token = txn_idx_probe.page_base + first_slot;
            const uint32_t last_token = txn_idx_probe.page_base + last_slot;
            const uint32_t last_x = desc.logical_x ? desc.logical_x - 1u : 0u;
            const uint32_t last_payload_word = desc.words_per_vector ? desc.words_per_vector - 1u : 0u;
            const uint32_t last_qblock = (uint32_t) (D / QK8_0 - 1);
            const uint64_t payload_first = dp16_packed_i8_payload_byte_offset(desc, 0, first_token, 0, 0);
            const uint64_t payload_last_end = dp16_packed_i8_payload_byte_offset(desc, 0, last_token, last_x, last_payload_word) + desc.bytes_per_word;
            const uint64_t scale_first = dp16_packed_i8_scale_byte_offset(desc, 0, first_token, 0);
            const uint64_t scale_last_end = dp16_packed_i8_scale_byte_offset(desc, 0, last_token, last_qblock) + sizeof(uint16_t);
            const uint32_t payload_row_bytes = (uint32_t) (desc.logical_x * desc.words_per_vector * desc.bytes_per_word);
            const uint32_t scale_row_bytes = (uint32_t) ((D / QK8_0) * sizeof(uint16_t));
            const bool row_payload_contiguous = desc.y_stride_bytes == payload_row_bytes;
            const bool row_scale_contiguous = desc.scale_y_stride_bytes == scale_row_bytes;
            fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_ADDR: kind=K active=%d eligible=%d page=[%u,%u) slots=[%u,%u) token=[%u,%u] payload_byte_span=[%llu,%llu) scale_byte_span=[%llu,%llu) payload_row_bytes=%u scale_row_bytes=%u row_payload_contiguous=%d row_scale_contiguous=%d layout=%s desc_x_stride=%llu desc_y_stride=%llu desc_z_stride=%llu desc_scale_x_stride=%llu desc_scale_y_stride=%llu desc_scale_z_stride=%llu\n",
                txn_k_page_merge_active ? 1 : 0,
                (!txn_idx_probe.spans_pages && txn_idx_probe.capacity_ok) ? 1 : 0,
                txn_idx_probe.page_base,
                txn_idx_probe.page_end,
                txn_idx_probe.slot_begin,
                txn_idx_probe.slot_end_excl,
                first_token,
                last_token,
                (unsigned long long) payload_first,
                (unsigned long long) payload_last_end,
                (unsigned long long) scale_first,
                (unsigned long long) scale_last_end,
                payload_row_bytes,
                scale_row_bytes,
                row_payload_contiguous ? 1 : 0,
                row_scale_contiguous ? 1 : 0,
                ggml_cuda_packed16_k_layout_kind_name(effective_layout),
                (unsigned long long) desc.x_stride_bytes,
                (unsigned long long) desc.y_stride_bytes,
                (unsigned long long) desc.z_stride_bytes,
                (unsigned long long) desc.scale_x_stride_bytes,
                (unsigned long long) desc.scale_y_stride_bytes,
                (unsigned long long) desc.scale_z_stride_bytes);
            }
        }
    }

    static bool pack_debug_printed = false;
    if (!pack_debug_printed && ggml_cuda_pack_k_packed16_env_enabled("GGML_CUDA_ROCM_Q8K_DOT4_KQ_DEBUG_I32")) {
        pack_debug_printed = true;
        fprintf(stderr,
            "pack_k_i32: k_cur type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
            "payload ne=[%lld,%lld,%lld,%lld] scales ne=[%lld,%lld,%lld,%lld] "
            "n_heads=%d nk_cur=%d kv_size=%d src_head_stride=%lld scale_mode=%d scale_mul=%g scale_group_qblocks=%d packed8_scale_mode=%d packed8_scale_mul=%g layout=%s\n",
            k_cur->type, (long long) k_cur->ne[0], (long long) k_cur->ne[1], (long long) k_cur->ne[2], (long long) k_cur->ne[3],
            k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3],
            (long long) payload->ne[0], (long long) payload->ne[1], (long long) payload->ne[2], (long long) payload->ne[3],
            (long long) scales->ne[0], (long long) scales->ne[1], (long long) scales->ne[2], (long long) scales->ne[3],
            n_heads, nk_cur, kv_size, (long long) src_head_stride_bytes, k_scale_mode, (double) k_scale_mul, k_scale_group_qblocks,
            packed8_q4_scale_mode, (double) packed8_q4_scale_mul, ggml_cuda_packed16_k_layout_kind_name(effective_layout));
    }

    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(256);

    const int scratch_page_base = (kv_size >= 2 * (int) MTP_V4_144_PAGE_TOKENS)
        ? ((kv_size / (int) MTP_V4_144_PAGE_TOKENS) - 1) * (int) MTP_V4_144_PAGE_TOKENS
        : -1;
    uint32_t txn_tail_owned_reserved_physical_page = 0;
    ggml_cuda_mtp_qblock_tail_page_map_v1 txn_tail_owned_reserved_map = {};
    const bool txn_tail_owned_reserved_map_ok = txn_k_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        llama_kv_cache_mtp_qblock_tail_page_published_owned_map_covers(
            txn_idx_probe.idx0, (uint32_t) nk_cur, &txn_tail_owned_reserved_physical_page, &txn_tail_owned_reserved_map);
    const bool txn_tail_owned_require_prepublished_map = txn_k_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        ggml_cuda_mtp_qblock_owned_tail_write_require_prepublished_map_requested();
    const bool txn_tail_owned_missing_required_map = txn_tail_owned_require_prepublished_map &&
        !txn_tail_owned_reserved_map_ok;
    const int txn_tail_owned_page_base = txn_tail_owned_reserved_map_ok ?
        (int) (txn_tail_owned_reserved_physical_page * (uint32_t) MTP_V4_144_PAGE_TOKENS) :
        (txn_tail_owned_missing_required_map ? -1 : scratch_page_base);
    const bool txn_tail_owned_write_active = txn_k_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        !txn_tail_owned_missing_required_map &&
        txn_tail_owned_page_base >= 0 &&
        (uint32_t) txn_tail_owned_page_base >= txn_idx_probe.page_end;
    if (txn_tail_owned_missing_required_map) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_FAIL_CLOSED: kind=K reason=missing_prepublished_owned_map idx0=%u n_tokens=%u scratch_page_base=%d page_end=%u\n",
                txn_idx_probe.idx0,
                (uint32_t) nk_cur,
                scratch_page_base,
                txn_idx_probe.page_end);
    }
    if (txn_k_page_merge_active && ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
            ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested()) {
        const bool writer_from_map = txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active;
        const bool legacy_fallback_owned = !txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active;
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=writer_authority kind=K writer_from_map=%d legacy_fallback_owned=%d owned_write_active=%d map_generation=%llu physical_page_from_map=%u overlay_page_base=%d idx0=%u n_tokens=%u map_logical_base=%u map_valid_tail=%u table0=%d\n",
                writer_from_map ? 1 : 0,
                legacy_fallback_owned ? 1 : 0,
                txn_tail_owned_write_active ? 1 : 0,
                (unsigned long long) (txn_tail_owned_reserved_map_ok ? txn_tail_owned_reserved_map.generation : 0ull),
                txn_tail_owned_reserved_map_ok ? txn_tail_owned_reserved_physical_page : 0xffffffffu,
                txn_tail_owned_write_active ? txn_tail_owned_page_base : -1,
                txn_idx_probe.idx0,
                (uint32_t) nk_cur,
                txn_tail_owned_reserved_map.logical_base_token,
                txn_tail_owned_reserved_map.valid_tail_tokens,
                txn_tail_owned_reserved_map.block_table[0]);
    }
    if (txn_k_page_merge_active && ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested()) {
        const char * owned_ready_reason = txn_tail_owned_reserved_map_ok ?
            (txn_tail_owned_write_active ? "ok" : "owned_page_visible") :
            (txn_tail_owned_missing_required_map ? "missing_prepublished_owned_map" : "no_owned_map");
        ggml_cuda_mtp_qblock_tail_page_record_owned_write_ready(
            "K",
            txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active,
            txn_tail_owned_reserved_map,
            txn_idx_probe.idx0,
            (uint32_t) nk_cur,
            txn_tail_owned_reserved_physical_page,
            owned_ready_reason);
    }
    const bool txn_tail_owned_exclusive_active = txn_tail_owned_write_active &&
        ggml_cuda_mtp_qblock_tail_page_owned_write_exclusive_allowed(
            "K",
            txn_tail_owned_reserved_map,
            txn_idx_probe.idx0,
            (uint32_t) nk_cur,
            txn_tail_owned_reserved_physical_page);
    const bool txn_tail_scratch_map_active = txn_k_page_merge_active &&
        !txn_tail_owned_write_active &&
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_map_requested() &&
        scratch_page_base >= 0 &&
        (uint32_t) scratch_page_base >= txn_idx_probe.page_end;
    const int txn_tail_scratch_exclusive_min_idx = ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_min_idx();
    const bool txn_tail_scratch_exclusive_after_nomap_max = ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_after_nomap_max_requested();
    const uint32_t txn_tail_consumer_no_map_nk_max = txn_tail_scratch_exclusive_after_nomap_max ?
        llama_kv_cache_get_mtp_qblock_tail_page_consumer_no_map_nk_max() : 0u;
    const bool txn_tail_scratch_exclusive_min_ok =
        txn_tail_scratch_exclusive_min_idx <= 0 || txn_idx_probe.idx0 >= (uint32_t) txn_tail_scratch_exclusive_min_idx;
    const bool txn_tail_scratch_exclusive_after_nomap_ok =
        !txn_tail_scratch_exclusive_after_nomap_max ||
        (txn_tail_consumer_no_map_nk_max > 0 && txn_idx_probe.idx0 >= txn_tail_consumer_no_map_nk_max);
    const bool txn_tail_scratch_exclusive_route_complete_ok =
        txn_tail_scratch_exclusive_after_nomap_max ||
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_single_map_unsafe_requested();
    const bool txn_tail_scratch_exclusive_active = txn_tail_scratch_map_active &&
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_requested() &&
        txn_tail_scratch_exclusive_route_complete_ok &&
        txn_tail_scratch_exclusive_min_ok && txn_tail_scratch_exclusive_after_nomap_ok;

    ggml_cuda_packed16_timing_trace_event pack_k_timing;
    ggml_cuda_packed16_timing_trace_begin(pack_k_timing, stream, "pack_k");

    if (txn_k_page_merge_active) {
        ggml_cuda_pool & pool = ctx.pool();
        if (txn_tail_owned_write_active || txn_tail_scratch_map_active) {
            const int overlay_page_base = txn_tail_owned_write_active ? txn_tail_owned_page_base : scratch_page_base;
            const bool owned_overlay_dual_candidate = txn_tail_owned_write_active &&
                !txn_tail_owned_exclusive_active && !txn_tail_scratch_exclusive_active &&
                ggml_cuda_mtp_qblock_owned_tail_write_dual_dest_candidate_requested();
            // Scratch/owned overlays are write-through by default: aged-out or
            // uncovered consumers still need canonical rows. Owned exclusive is
            // a diagnostic subgate; all-bound no-write-through also requires an
            // explicit UNSAFE archaeology key after the age-out proof.
            if (!txn_tail_owned_exclusive_active && !txn_tail_scratch_exclusive_active) {
                const int dual_overlay_page_base = owned_overlay_dual_candidate ? overlay_page_base : -1;
                if (k_idxs->type == GGML_TYPE_I64) {
                    ggml_cuda_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                        (const half *) k_cur->data,
                        (int *) payload->data,
                        (half *) scales->data,
                        (const int64_t *) k_idxs->data,
                        src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                        src_head_stride_bytes,
                        nk_cur, n_heads, batch, kv_size,
                        k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, dual_overlay_page_base);
                } else {
                    GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
                    ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                        (const half *) k_cur->data,
                        (int *) payload->data,
                        (half *) scales->data,
                        (const int32_t *) k_idxs->data,
                        src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                        src_head_stride_bytes,
                        nk_cur, n_heads, batch, kv_size,
                        k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, dual_overlay_page_base);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            if (owned_overlay_dual_candidate) {
                // The canonical pack above also populated the owned overlay.
            } else if (k_idxs->type == GGML_TYPE_I64) {
                ggml_cuda_pool_alloc<int64_t> local_idxs_alloc(pool);
                int64_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int64_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, overlay_page_base);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                    (const half *) k_cur->data,
                    (int *) payload->data,
                    (half *) scales->data,
                    local_idxs,
                    src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                    src_head_stride_bytes,
                    nk_cur, n_heads, batch, kv_size,
                    k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, -1);
            } else {
                GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
                ggml_cuda_pool_alloc<int32_t> local_idxs_alloc(pool);
                int32_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int32_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, overlay_page_base);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                    (const half *) k_cur->data,
                    (int *) payload->data,
                    (half *) scales->data,
                    local_idxs,
                    src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                    src_head_stride_bytes,
                    nk_cur, n_heads, batch, kv_size,
                    k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, -1);
            }
            const int poison_min_idx = ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_min_idx();
            const int poison_max_idx = ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_max_idx();
            if (txn_tail_owned_write_active && ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_k_requested() &&
                    ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_range_hits(
                        txn_idx_probe.idx0, (uint32_t) nk_cur, poison_min_idx, poison_max_idx)) {
                if (k_idxs->type == GGML_TYPE_I64) {
                    ggml_cuda_mtp_qblock_poison_canonical_k_indexed_kernel<int64_t><<<grid, dim3(128), 0, stream>>>(
                        (int *) payload->data,
                        (half *) scales->data,
                        (const int64_t *) k_idxs->data,
                        nk_cur, n_heads, batch, kv_size, effective_layout, poison_min_idx, poison_max_idx);
                } else {
                    GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
                    ggml_cuda_mtp_qblock_poison_canonical_k_indexed_kernel<int32_t><<<grid, dim3(128), 0, stream>>>(
                        (int *) payload->data,
                        (half *) scales->data,
                        (const int32_t *) k_idxs->data,
                        nk_cur, n_heads, batch, kv_size, effective_layout, poison_min_idx, poison_max_idx);
                }
                CUDA_CHECK(cudaGetLastError());
                fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_CANONICAL_POISON: kind=K status=ok reason=owned_write_active idx0=%u n_tokens=%u poison_min_idx=%d poison_max_idx=%d canonical_page_base=%u owned_page_base=%d physical_page=%u map_logical_base=%u map_valid_tail=%u table0=%d flags=0x%x generation=%llu\n",
                    txn_idx_probe.idx0,
                    (uint32_t) nk_cur,
                    poison_min_idx,
                    poison_max_idx,
                    txn_idx_probe.page_base,
                    txn_tail_owned_page_base,
                    txn_tail_owned_reserved_physical_page,
                    txn_tail_owned_reserved_map.logical_base_token,
                    txn_tail_owned_reserved_map.valid_tail_tokens,
                    txn_tail_owned_reserved_map.block_table[0],
                    txn_tail_owned_reserved_map.flags,
                    (unsigned long long) txn_tail_owned_reserved_map.generation);
            }
            if (txn_tail_owned_write_active && overlay_page_base >= 0) {
                const size_t scratch_rows = size_t(batch) * size_t(n_heads) * size_t(MTP_V4_144_PAGE_TOKENS);
                ggml_cuda_pool_alloc<int>  owned_payload_alloc(pool);
                ggml_cuda_pool_alloc<half> owned_scales_alloc(pool);
                int  * owned_payload = owned_payload_alloc.alloc(scratch_rows * size_t(GGML_CUDA_PACKED16_K_WORDS));
                half * owned_scales  = owned_scales_alloc.alloc (scratch_rows * size_t(GGML_CUDA_PACKED16_K_QBLOCKS));

                dim3 page_grid(MTP_V4_144_PAGE_TOKENS, n_heads, batch);
                dim3 page_block(128);
                ggml_cuda_mtp_qblock_txn_tail_k_copy_page_to_scratch_kernel<<<page_grid, page_block, 0, stream>>>(
                    (const int *) payload->data,
                    (const half *) scales->data,
                    owned_payload,
                    owned_scales,
                    (int) txn_idx_probe.page_base,
                    kv_size,
                    n_heads,
                    effective_layout);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_mtp_qblock_txn_tail_k_commit_scratch_page_kernel<<<page_grid, page_block, 0, stream>>>(
                    owned_payload,
                    owned_scales,
                    (int *) payload->data,
                    (half *) scales->data,
                    overlay_page_base,
                    kv_size,
                    n_heads,
                    effective_layout);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            const size_t scratch_rows = size_t(batch) * size_t(n_heads) * size_t(MTP_V4_144_PAGE_TOKENS);
            ggml_cuda_pool_alloc<int>  scratch_payload_alloc(pool);
            ggml_cuda_pool_alloc<half> scratch_scales_alloc(pool);
            int  * scratch_payload = scratch_payload_alloc.alloc(scratch_rows * size_t(GGML_CUDA_PACKED16_K_WORDS));
            half * scratch_scales  = scratch_scales_alloc.alloc (scratch_rows * size_t(GGML_CUDA_PACKED16_K_QBLOCKS));

            dim3 page_grid(MTP_V4_144_PAGE_TOKENS, n_heads, batch);
            dim3 page_block(128);
            ggml_cuda_mtp_qblock_txn_tail_k_copy_page_to_scratch_kernel<<<page_grid, page_block, 0, stream>>>(
                (const int *) payload->data,
                (const half *) scales->data,
                scratch_payload,
                scratch_scales,
                (int) txn_idx_probe.page_base,
                kv_size,
                n_heads,
                effective_layout);
            CUDA_CHECK(cudaGetLastError());

            if (k_idxs->type == GGML_TYPE_I64) {
                ggml_cuda_pool_alloc<int64_t> local_idxs_alloc(pool);
                int64_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int64_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, (int) txn_idx_probe.slot_begin);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                    (const half *) k_cur->data,
                    scratch_payload,
                    scratch_scales,
                    local_idxs,
                    src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                    src_head_stride_bytes,
                    nk_cur, n_heads, batch, MTP_V4_144_PAGE_TOKENS,
                    k_scale_mode, k_scale_mul, k_scale_group_qblocks, GGML_CUDA_PACKED16_K_LAYOUT_ROW, -1);
            } else {
                GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
                ggml_cuda_pool_alloc<int32_t> local_idxs_alloc(pool);
                int32_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int32_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, (int) txn_idx_probe.slot_begin);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                    (const half *) k_cur->data,
                    scratch_payload,
                    scratch_scales,
                    local_idxs,
                    src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
                    src_head_stride_bytes,
                    nk_cur, n_heads, batch, MTP_V4_144_PAGE_TOKENS,
                    k_scale_mode, k_scale_mul, k_scale_group_qblocks, GGML_CUDA_PACKED16_K_LAYOUT_ROW, -1);
            }
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_mtp_qblock_txn_tail_k_commit_scratch_page_kernel<<<page_grid, page_block, 0, stream>>>(
                scratch_payload,
                scratch_scales,
                (int *) payload->data,
                (half *) scales->data,
                (int) txn_idx_probe.page_base,
                kv_size,
                n_heads,
                effective_layout);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (payload_is_packed8 && is_q8_k) {
        GGML_ABORT("packed8 q4 K writer does not yet accept q8_0 source rows");
    } else if (is_q8_k) {
        // Indexed pack from q8_0 cache blocks preserves calibrated quantization.
        ggml_cuda_pack_k_packed16_from_q8_indexed_kernel<<<grid, block, 0, stream>>>(
            (const char *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int64_t *) k_idxs->data,
            k_cur->nb[1], kv_size, n_heads, batch, nk_cur, effective_layout);
    } else if (payload_is_packed8 && k_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_quant_k_packed8_q4_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const char *) k_cur->data, (int *) payload->data, (half *) scales->data, (const int64_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3], src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_cur->type == GGML_TYPE_F16,
            packed8_q4_scale_mode, packed8_q4_scale_mul);
    } else if (payload_is_packed8) {
        GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
        ggml_cuda_quant_k_packed8_q4_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const char *) k_cur->data, (int *) payload->data, (half *) scales->data, (const int32_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3], src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_cur->type == GGML_TYPE_F16,
            packed8_q4_scale_mode, packed8_q4_scale_mul);
    } else if (k_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_quant_k_packed16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int64_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, -1);
    } else {
        GGML_ASSERT(k_idxs->type == GGML_TYPE_I32);
        ggml_cuda_quant_k_packed16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const half *) k_cur->data,
            (int *) payload->data,
            (half *) scales->data,
            (const int32_t *) k_idxs->data,
            src_token_stride_bytes, k_cur->nb[2], k_cur->nb[3],
            src_head_stride_bytes,
            nk_cur, n_heads, batch, kv_size, k_scale_mode, k_scale_mul, k_scale_group_qblocks, effective_layout, -1);
    }
    CUDA_CHECK(cudaGetLastError());
    const float pack_k_timing_ms = ggml_cuda_packed16_timing_trace_end(pack_k_timing, stream);
    if (pack_k_timing_ms >= 0.0f) {
        fprintf(stderr,
            "PACKED16_TIMING_TRACE phase=pack_k ms=%.3f nk_cur=%d n_heads=%d batch=%d kv_size=%d layout=%s format=%s src_type=%s idx_type=%s payload_words=%u qblocks=%d txn_merge=%d owned=%d scratch=%d\n",
            (double) pack_k_timing_ms,
            nk_cur,
            n_heads,
            batch,
            kv_size,
            ggml_cuda_packed16_k_layout_kind_name(effective_layout),
            ggml_cuda_pdmq_k_format_name(k_format),
            ggml_type_name(k_cur->type),
            k_idxs->type == GGML_TYPE_I64 ? "i64" : "i32",
            (unsigned) ggml_cuda_pdmq_k_payload_words_per_token(k_format, (uint32_t) D),
            D / QK8_0,
            txn_k_page_merge_active ? 1 : 0,
            txn_tail_owned_write_active ? 1 : 0,
            txn_tail_scratch_map_active ? 1 : 0);
    }

    // Bind the just-produced sidecar bytes to their physical layout. Consumers
    // must use this metadata instead of re-reading the environment later.
    llama_kv_cache_register_pdmq_k_with_layout_info(dst->data, payload, scales, k_format, effective_layout, (uint32_t) kv_size, (uint32_t) D);
    if (k_cur && k_cur->data) {
        llama_kv_cache_register_pdmq_k_with_layout_info(k_cur->data, payload, scales, k_format, effective_layout, (uint32_t) kv_size, (uint32_t) D);
    }

    if (txn_k_page_merge_active && ggml_cuda_mtp_qblock_txn_tail_page_consumer_requested() &&
            ggml_cuda_mtp_qblock_txn_tail_page_producer_map_requested()) {
        ggml_cuda_mtp_qblock_tail_page_map_v1 map = {};
        map.version = GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_VERSION;
        map.abi_bytes = sizeof(map);
        map.active = 1;
        map.flags = txn_tail_owned_write_active ? GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_OWNED_TAIL_WRITE :
            (txn_tail_scratch_map_active ? GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_FLAG_SCRATCH_OVERLAY : 0u);
        map.page_tokens = MTP_V4_144_PAGE_TOKENS;
        map.physical_pages = (uint32_t) (kv_size / (int) MTP_V4_144_PAGE_TOKENS);
        const uint32_t tail_logical_page = txn_idx_probe.page_base / (uint32_t) MTP_V4_144_PAGE_TOKENS;
        const uint32_t window_pages = tail_logical_page + 1u < GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES ?
            tail_logical_page + 1u : GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES;
        const uint32_t window_base_page = tail_logical_page + 1u - window_pages;
        map.logical_base_token = window_base_page * (uint32_t) MTP_V4_144_PAGE_TOKENS;
        map.valid_tail_tokens = txn_idx_probe.idx0 + (uint32_t) nk_cur - map.logical_base_token;
        map.block_table_pages = (map.valid_tail_tokens + map.page_tokens - 1u) / map.page_tokens;
        if (map.block_table_pages > GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES) {
            map.block_table_pages = GGML_CUDA_MTP_QBLOCK_TAIL_PAGE_MAP_MAX_PAGES;
        }
        for (uint32_t lp = 0; lp < map.block_table_pages; ++lp) {
            map.block_table[lp] = (int32_t) (window_base_page + lp);
        }
        if (txn_tail_owned_write_active || txn_tail_scratch_map_active) {
            const uint32_t overlay_physical_page = (uint32_t) ((txn_tail_owned_write_active ? txn_tail_owned_page_base : scratch_page_base) / (int) MTP_V4_144_PAGE_TOKENS);
            const uint32_t tail_rel_page = tail_logical_page - window_base_page;
            if (tail_rel_page < map.block_table_pages) {
                map.block_table[tail_rel_page] = (int32_t) overlay_physical_page;
            }
        }
        map.generation = ggml_cuda_mtp_qblock_tail_page_map_generation(map.logical_base_token, map.valid_tail_tokens, map.flags);
        llama_kv_cache_register_mtp_qblock_tail_page_map(dst->data, &map);
        if (k_cur && k_cur->data) {
            llama_kv_cache_register_mtp_qblock_tail_page_map(k_cur->data, &map);
        }
    }
}

static constexpr int GGML_CUDA_V4_K16D16_D = 256;
static constexpr int GGML_CUDA_V4_K16D16_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_D_TILE = 16;
static constexpr int GGML_CUDA_V4_K16D16_WORDS_PER_D = 2;
static constexpr int GGML_CUDA_V4_K16D16_ROW_BYTES = sizeof(block_v4_k16d16);
static constexpr int GGML_CUDA_V4_K16D16_BLOCK_BYTES = GGML_CUDA_V4_K16D16_ROW_BYTES * GGML_CUDA_V4_K16D16_K;
static constexpr int GGML_CUDA_V4_K16D16_PAYLOAD_BYTES = GGML_CUDA_V4_K16D16_D * GGML_CUDA_V4_K16D16_WORDS_PER_D * (int) sizeof(uint32_t);
static_assert(GGML_CUDA_V4_K16D16_PAYLOAD_BYTES + (GGML_CUDA_V4_K16D16_D / GGML_CUDA_V4_K16D16_D_TILE) * (int) sizeof(half) == GGML_CUDA_V4_K16D16_BLOCK_BYTES, "bad V4_K16D16 byte math");

static __device__ __forceinline__ float ggml_cuda_v4_load_src(
        const char * __restrict__ src,
        const bool src_f16,
        const int64_t nb01,
        const int64_t nb02,
        const int64_t nb03,
        const int64_t src_head_stride_bytes,
        const int token,
        const int head,
        const int batch,
        const int d) {
    const char * p = src + int64_t(batch) * nb03 + int64_t(token) * nb01 + int64_t(head) * src_head_stride_bytes;
    return src_f16 ? __half2float(((const half *) p)[d]) : ((const float *) p)[d];
}

template<typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_v4_k16d16_indexed_kernel(
        const char  * __restrict__ Vcur,
        char        * __restrict__ v4_cache,
        half        * __restrict__ v_tail,
        const idx_t * __restrict__ v_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int64_t v4_nb1,
        int64_t v4_nb2,
        int64_t tail_nb1,
        int64_t tail_nb2,
        int nk_cur,
        int n_heads_v,
        int batch,
        int kv_size,
        bool src_f16) {
    const int d = int(threadIdx.x);
    const int token = int(blockIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    if (d >= GGML_CUDA_V4_K16D16_D || token >= nk_cur || hv >= n_heads_v || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) v_idxs[token];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    const int tail_slot = cell & (GGML_CUDA_V4_K16D16_K - 1);

    const float x = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d);
    half * tail_row = (half *) ((char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + tail_slot) * tail_nb1);
    tail_row[d] = __float2half(x);

    if (tail_slot != GGML_CUDA_V4_K16D16_K - 1) {
        return;
    }

    const int block_start = cell - tail_slot;
    char * block = v4_cache + int64_t(b) * v4_nb2 + int64_t(hv * kv_size + block_start) * v4_nb1;
    const int d_tile = d / GGML_CUDA_V4_K16D16_D_TILE;
    const int d0 = d_tile * GGML_CUDA_V4_K16D16_D_TILE;

    float amax = 0.0f;
    float maxv = 0.0f;
#pragma unroll
    for (int kk = 0; kk < GGML_CUDA_V4_K16D16_K; ++kk) {
        const half * row = (const half *) ((const char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + kk) * tail_nb1);
#pragma unroll
        for (int dd = 0; dd < GGML_CUDA_V4_K16D16_D_TILE; ++dd) {
            const float v = __half2float(row[d0 + dd]);
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
                maxv = v;
            }
        }
    }
    float scale = maxv / -8.0f;
    if (!(scale != 0.0f) || !isfinite(scale)) {
        scale = 1.0f;
    }
    const float inv_scale = 1.0f / scale;

#pragma unroll
    for (int g = 0; g < GGML_CUDA_V4_K16D16_WORDS_PER_D; ++g) {
        uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kk = g * 8 + j;
            const half * row = (const half *) ((const char *) v_tail + int64_t(b) * tail_nb2 + int64_t(hv * GGML_CUDA_V4_K16D16_K + kk) * tail_nb1);
            const float v = __half2float(row[d]);
            int q = (int) (v * inv_scale + 8.5f);
            q = q < 0 ? 0 : (q > 15 ? 15 : q);
            word |= (uint32_t(q) & 0x0fu) << (4 * j);
        }
        ((uint32_t *) (block + d * GGML_CUDA_V4_K16D16_WORDS_PER_D * (int) sizeof(uint32_t)))[g] = word;
    }
    if ((d & (GGML_CUDA_V4_K16D16_D_TILE - 1)) == 0) {
        ((half *) (block + GGML_CUDA_V4_K16D16_PAYLOAD_BYTES))[d_tile] = __float2half(scale);
    }
}

void ggml_cuda_op_pack_v4_k16d16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * v_cur  = dst->src[0];
    ggml_tensor * v_tail = dst->src[1];
    ggml_tensor * v_idxs = dst->src[2];
    ggml_tensor * v4     = dst;

    GGML_ASSERT(v4->type == GGML_TYPE_V4_K16D16);
    GGML_ASSERT(v_tail->type == GGML_TYPE_F16);
    GGML_ASSERT(v_idxs->type == GGML_TYPE_I64 || v_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(v4->ne[0] == GGML_CUDA_V4_K16D16_D);
    GGML_ASSERT(v_tail->ne[0] == GGML_CUDA_V4_K16D16_D);
    GGML_ASSERT(v_tail->ne[1] % GGML_CUDA_V4_K16D16_K == 0);

    const int D = GGML_CUDA_V4_K16D16_D;
    const bool src_f16 = v_cur->type == GGML_TYPE_F16;
    const int batch = (int) v_cur->ne[3];
    int n_heads = 0;
    int nk_cur = 0;
    int64_t src_head_stride_bytes = 0;
    if (v_cur->ne[0] == D) {
        n_heads = (int) v_cur->ne[1];
        nk_cur = (int) v_cur->ne[2];
        src_head_stride_bytes = v_cur->nb[1];
    } else {
        GGML_ASSERT(v_cur->ne[0] % D == 0);
        n_heads = (int) (v_cur->ne[0] / D);
        nk_cur = (int) v_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(v_cur->type);
    }
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(v4->ne[1] % n_heads == 0);
    GGML_ASSERT(v_tail->ne[1] == GGML_CUDA_V4_K16D16_K * n_heads);
    const int kv_size = (int) (v4->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);

    const int64_t src_token_stride_bytes = (v_cur->ne[0] == D) ? v_cur->nb[2] : v_cur->nb[1];
    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(D);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_packed16_timing_trace_event pack_v4_timing;
    ggml_cuda_packed16_timing_trace_begin(pack_v4_timing, stream, "pack_v4");
    if (v_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_pack_v4_k16d16_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v4->data, (half *) v_tail->data, (const int64_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v4->nb[1], v4->nb[2], v_tail->nb[1], v_tail->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    } else {
        ggml_cuda_pack_v4_k16d16_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v4->data, (half *) v_tail->data, (const int32_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v4->nb[1], v4->nb[2], v_tail->nb[1], v_tail->nb[2], nk_cur, n_heads, batch, kv_size, src_f16);
    }
    CUDA_CHECK(cudaGetLastError());
    const float pack_v4_timing_ms = ggml_cuda_packed16_timing_trace_end(pack_v4_timing, stream);
    if (pack_v4_timing_ms >= 0.0f) {
        fprintf(stderr,
            "PACKED16_TIMING_TRACE phase=pack_v4 ms=%.3f nk_cur=%d n_heads=%d batch=%d kv_size=%d src_type=%s idx_type=%s row_bytes=%d\n",
            (double) pack_v4_timing_ms,
            nk_cur,
            n_heads,
            batch,
            kv_size,
            ggml_type_name(v_cur->type),
            v_idxs->type == GGML_TYPE_I64 ? "i64" : "i32",
            GGML_CUDA_V4_K16D16_ROW_BYTES);
    }
}

static constexpr int GGML_CUDA_V4_K16D16_144_D = 256;
static constexpr int GGML_CUDA_V4_K16D16_144_K = 16;
static constexpr int GGML_CUDA_V4_K16D16_144_D32 = 32;
static constexpr int GGML_CUDA_V4_K16D16_144_D32_BLOCKS = GGML_CUDA_V4_K16D16_144_D / GGML_CUDA_V4_K16D16_144_D32;
static constexpr int GGML_CUDA_V4_K16D16_144_ROW_BYTES = sizeof(block_v4_k16d16_144);
static constexpr int GGML_CUDA_V4_K16D16_144_BLOCK_BYTES = GGML_CUDA_V4_K16D16_144_ROW_BYTES * GGML_CUDA_V4_K16D16_144_K;
static constexpr int GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES = GGML_CUDA_V4_K16D16_144_D * 2 * (int) sizeof(uint32_t);
static constexpr int GGML_CUDA_V4_K16D16_144_SCALE_BYTES = GGML_CUDA_V4_K16D16_144_D32_BLOCKS * GGML_CUDA_V4_K16D16_144_K * (int) sizeof(half);
static_assert(GGML_CUDA_V4_K16D16_144_BLOCK_BYTES == 2304, "bad V4_K16D16_144 block bytes");
static_assert(GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES == 2048, "bad V4_K16D16_144 payload bytes");
static_assert(GGML_CUDA_V4_K16D16_144_SCALE_BYTES == 256, "bad V4_K16D16_144 scale bytes");
static_assert(GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES + GGML_CUDA_V4_K16D16_144_SCALE_BYTES == GGML_CUDA_V4_K16D16_144_BLOCK_BYTES, "bad V4_K16D16_144 byte math");

static __device__ __forceinline__ void ggml_cuda_v4_k16d16_144_store_nibble(
        uint32_t * __restrict__ wordp,
        const int slot,
        const uint32_t q_unsigned) {
    const uint32_t shift = uint32_t(4 * (slot & 7));
    const uint32_t mask  = uint32_t(0x0fu) << shift;
    uint32_t old = *wordp;
    uint32_t assumed;
    do {
        assumed = old;
        const uint32_t desired = (assumed & ~mask) | ((q_unsigned & 0x0fu) << shift);
        old = atomicCAS((unsigned int *) wordp, (unsigned int) assumed, (unsigned int) desired);
    } while (old != assumed);
}

static __device__ __forceinline__ uint16_t ggml_cuda_v4_k16d16_144_half_bits(const half h) {
    union {
        half h;
        uint16_t u;
    } cvt;
    cvt.h = h;
    return cvt.u;
}

static __device__ __forceinline__ void ggml_cuda_v4_k16d16_144_store_scale(
        half * __restrict__ scales,
        const int scale_idx,
        const half scale) {
    uint32_t * wordp = (uint32_t *) (scales + (scale_idx & ~1));
    const uint32_t shift = uint32_t(16 * (scale_idx & 1));
    const uint32_t mask  = uint32_t(0xffffu) << shift;
    const uint32_t bits  = uint32_t(ggml_cuda_v4_k16d16_144_half_bits(scale)) << shift;
    uint32_t old = *wordp;
    uint32_t assumed;
    do {
        assumed = old;
        const uint32_t desired = (assumed & ~mask) | bits;
        old = atomicCAS((unsigned int *) wordp, (unsigned int) assumed, (unsigned int) desired);
    } while (old != assumed);
}

static __device__ __forceinline__ uint32_t ggml_cuda_v4_k16d16_144_load_nibble(
        const uint32_t * __restrict__ wordp,
        const int slot) {
    const uint32_t shift = uint32_t(4 * (slot & 7));
    return (*wordp >> shift) & 0x0fu;
}

template<typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_pack_v4_k16d16_144_indexed_kernel(
        const char  * __restrict__ Vcur,
        char        * __restrict__ v144_cache,
        const idx_t * __restrict__ v_idxs,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t src_head_stride_bytes,
        int64_t v144_nb1,
        int64_t v144_nb2,
        int nk_cur,
        int n_heads_v,
        int batch,
        int kv_size,
        bool src_f16,
        int overlay_page_base) {
    const int d = int(threadIdx.x);
    const int token = int(blockIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    if (d >= GGML_CUDA_V4_K16D16_144_D || token >= nk_cur || hv >= n_heads_v || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) v_idxs[token];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    const int slot = cell & (GGML_CUDA_V4_K16D16_144_K - 1);
    const int k16_base = cell & ~(GGML_CUDA_V4_K16D16_144_K - 1);
    const int overlay_cell = overlay_page_base >= 0 ? overlay_page_base + token : -1;
    const bool overlay_valid = overlay_cell >= 0 && overlay_cell < kv_size;
    const int overlay_slot = overlay_valid ? (overlay_cell & (GGML_CUDA_V4_K16D16_144_K - 1)) : 0;
    const int overlay_k16_base = overlay_valid ? (overlay_cell & ~(GGML_CUDA_V4_K16D16_144_K - 1)) : 0;

    char * block = v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + k16_base) * v144_nb1;
    char * overlay_block = overlay_valid ?
        v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + overlay_k16_base) * v144_nb1 : nullptr;
    const int d32 = d >> 5;
    const int d0 = d32 * GGML_CUDA_V4_K16D16_144_D32;

    float amax = 0.0f;
    float maxv = 0.0f;
#pragma unroll
    for (int dd = 0; dd < GGML_CUDA_V4_K16D16_144_D32; ++dd) {
        const float v = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d0 + dd);
        const float av = fabsf(v);
        if (av > amax) {
            amax = av;
            maxv = v;
        }
    }
    const float scale = maxv / -8.0f;
    const float inv_scale = scale ? 1.0f / scale : 0.0f;

    const float x = ggml_cuda_v4_load_src(Vcur, src_f16, nb01, nb02, nb03, src_head_stride_bytes, token, hv, b, d);
    int q = (int8_t) (x * inv_scale + 8.5f);
    q = q > 15 ? 15 : q;

    uint32_t * payload = (uint32_t *) block;
    ggml_cuda_v4_k16d16_144_store_nibble(&payload[d * 2 + (slot >> 3)], slot, (uint32_t) q);
    if (overlay_valid) {
        uint32_t * overlay_payload = (uint32_t *) overlay_block;
        ggml_cuda_v4_k16d16_144_store_nibble(&overlay_payload[d * 2 + (overlay_slot >> 3)], overlay_slot, (uint32_t) q);
    }

    if ((d & (GGML_CUDA_V4_K16D16_144_D32 - 1)) == 0) {
        const half hscale = __float2half(scale);
        half * scales = (half *) (block + GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES);
        const int scale_idx = d32 * GGML_CUDA_V4_K16D16_144_K + slot;
        ggml_cuda_v4_k16d16_144_store_scale(scales, scale_idx, hscale);
        if (overlay_valid) {
            half * overlay_scales = (half *) (overlay_block + GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES);
            const int overlay_scale_idx = d32 * GGML_CUDA_V4_K16D16_144_K + overlay_slot;
            ggml_cuda_v4_k16d16_144_store_scale(overlay_scales, overlay_scale_idx, hscale);
        }
    }
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_mtp_qblock_txn_tail_v4_144_copy_page_to_scratch_kernel(
        const char * __restrict__ v144_cache,
        char       * __restrict__ scratch_page,
        int page_base,
        int64_t v144_nb1,
        int64_t v144_nb2,
        int kv_size,
        int n_heads_v) {
    const int word = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    constexpr int page_words = GGML_CUDA_V4_K16D16_144_BLOCK_BYTES / (int) sizeof(uint32_t);
    if (word >= page_words || hv >= n_heads_v) {
        return;
    }
    const char * src_page = v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + page_base) * v144_nb1;
    char * dst_page = scratch_page + (int64_t(b) * n_heads_v + hv) * GGML_CUDA_V4_K16D16_144_BLOCK_BYTES;
    ((uint32_t *) dst_page)[word] = ((const uint32_t *) src_page)[word];
}

static __global__ __launch_bounds__(256, 1) void ggml_cuda_mtp_qblock_txn_tail_v4_144_commit_scratch_page_kernel(
        const char * __restrict__ scratch_page,
        char       * __restrict__ v144_cache,
        int page_base,
        int64_t v144_nb1,
        int64_t v144_nb2,
        int kv_size,
        int n_heads_v) {
    const int word = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    constexpr int page_words = GGML_CUDA_V4_K16D16_144_BLOCK_BYTES / (int) sizeof(uint32_t);
    if (word >= page_words || hv >= n_heads_v) {
        return;
    }
    const char * src_page = scratch_page + (int64_t(b) * n_heads_v + hv) * GGML_CUDA_V4_K16D16_144_BLOCK_BYTES;
    char * dst_page = v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + page_base) * v144_nb1;
    ((uint32_t *) dst_page)[word] = ((const uint32_t *) src_page)[word];
}

template <typename idx_t>
static __global__ __launch_bounds__(256, 1) void ggml_cuda_mtp_qblock_poison_canonical_v4_144_indexed_kernel(
        char        * __restrict__ v144_cache,
        const idx_t * __restrict__ v_idxs,
        int64_t v144_nb1,
        int64_t v144_nb2,
        int nk_cur,
        int n_heads_v,
        int batch,
        int kv_size,
        int poison_min_idx,
        int poison_max_idx) {
    const int d = int(threadIdx.x);
    const int token = int(blockIdx.x);
    const int hv = int(blockIdx.y);
    const int b = int(blockIdx.z);
    if (d >= GGML_CUDA_V4_K16D16_144_D || token >= nk_cur || hv >= n_heads_v || b >= batch) {
        return;
    }

    const int64_t cell64 = (int64_t) v_idxs[token];
    if (cell64 < 0 || cell64 >= kv_size) {
        return;
    }
    const int cell = (int) cell64;
    if ((poison_min_idx > 0 && cell < poison_min_idx) || (poison_max_idx >= 0 && cell >= poison_max_idx)) {
        return;
    }
    const int slot = cell & (GGML_CUDA_V4_K16D16_144_K - 1);
    const int k16_base = cell & ~(GGML_CUDA_V4_K16D16_144_K - 1);

    char * block = v144_cache + int64_t(b) * v144_nb2 + int64_t(hv * kv_size + k16_base) * v144_nb1;
    uint32_t * payload = (uint32_t *) block;
    ggml_cuda_v4_k16d16_144_store_nibble(&payload[d * 2 + (slot >> 3)], slot, 0x0fu);

    if ((d & (GGML_CUDA_V4_K16D16_144_D32 - 1)) == 0) {
        half * scales = (half *) (block + GGML_CUDA_V4_K16D16_144_PAYLOAD_BYTES);
        const int d32 = d >> 5;
        const int scale_idx = d32 * GGML_CUDA_V4_K16D16_144_K + slot;
        ggml_cuda_v4_k16d16_144_store_scale(scales, scale_idx, __float2half(16.0f));
    }
}

void ggml_cuda_op_pack_v4_k16d16_144(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * v_cur  = dst->src[0];
    ggml_tensor * v_idxs = dst->src[2];
    ggml_tensor * v144   = dst;

    GGML_ASSERT(v144->type == GGML_TYPE_V4_K16D16_144);
    GGML_ASSERT(v_idxs->type == GGML_TYPE_I64 || v_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(v144->ne[0] == GGML_CUDA_V4_K16D16_144_D);

    const int D = GGML_CUDA_V4_K16D16_144_D;
    const bool src_f16 = v_cur->type == GGML_TYPE_F16;
    const int batch = (int) v_cur->ne[3];
    int n_heads = 0;
    int nk_cur = 0;
    int64_t src_head_stride_bytes = 0;
    if (v_cur->ne[0] == D) {
        n_heads = (int) v_cur->ne[1];
        nk_cur = (int) v_cur->ne[2];
        src_head_stride_bytes = v_cur->nb[1];
    } else {
        GGML_ASSERT(v_cur->ne[0] % D == 0);
        n_heads = (int) (v_cur->ne[0] / D);
        nk_cur = (int) v_cur->ne[1];
        src_head_stride_bytes = (int64_t) D * (int64_t) ggml_type_size(v_cur->type);
    }
    GGML_ASSERT(n_heads > 0);
    GGML_ASSERT(v144->ne[1] % n_heads == 0);
    const int kv_size = (int) (v144->ne[1] / n_heads);
    GGML_ASSERT(kv_size >= nk_cur);
    GGML_ASSERT((kv_size % GGML_CUDA_V4_K16D16_144_K) == 0);
    GGML_ASSERT(v144->nb[1] == GGML_CUDA_V4_K16D16_144_ROW_BYTES);

    const int64_t src_token_stride_bytes = (v_cur->ne[0] == D) ? v_cur->nb[2] : v_cur->nb[1];
    cudaStream_t stream = ctx.stream();
    const bool txn_tail_backend_trace = ggml_cuda_mtp_qblock_txn_tail_backend_proof_enabled();
    const bool txn_tail_page_request = ggml_cuda_mtp_qblock_txn_tail_page_requested();
    const bool txn_tail_backend_probe = txn_tail_backend_trace || txn_tail_page_request;
    const bool txn_v_stride_ok = v144->nb[1] == MTP_V4_144_ROW_BYTES && v144->nb[1] * MTP_V4_144_PAGE_TOKENS == MTP_V4_144_PAGE_BYTES;
    mtp_v4_144_tail_stage_status txn_v_geom_status = MTP_V4_144_TAIL_STAGE_BAD_KIND;
    ggml_cuda_mtp_qblock_txn_tail_backend_idx_probe txn_v_idx_probe = {};
    bool txn_v_page_merge_active = false;

    if (txn_tail_backend_probe && nk_cur > 1 && nk_cur <= 8) {
        txn_v_geom_status = ggml_cuda_mtp_qblock_txn_tail_backend_geom_status(
            MTP_V4_144_TAIL_STAGE_KIND_V4_144,
            (uint32_t) kv_size,
            MTP_V4_144_ROW_BYTES);
        txn_v_idx_probe = ggml_cuda_mtp_qblock_txn_tail_backend_probe_indices(
            v_idxs,
            nk_cur,
            kv_size,
            MTP_V4_144_TAIL_STAGE_KIND_V4_144,
            MTP_V4_144_ROW_BYTES,
            stream);
        txn_v_page_merge_active = txn_tail_page_request && txn_v_stride_ok &&
            txn_v_idx_probe.copied && txn_v_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK && txn_v_idx_probe.contiguous &&
            txn_v_idx_probe.capacity_ok && !txn_v_idx_probe.spans_pages && txn_v_idx_probe.merge_copy_slots != 0 &&
            (v_cur->type == GGML_TYPE_F32 || v_cur->type == GGML_TYPE_F16) &&
            (v_idxs->type == GGML_TYPE_I64 || v_idxs->type == GGML_TYPE_I32);
        if (txn_tail_backend_trace) {
            fprintf(stderr,
            "MTP_QBLOCK_TXN_TAIL_BACKEND: kind=V active=%d eligible=%d geom_status=%u idx_status=%u idx_probe=%d idx_capture_skipped=%d idx_contiguous=%d idx_capacity_ok=%d idx0=%u idx_last=%u page=[%u,%u) slots=[%u,%u) merge_copy_slots=%u spans_pages=%d nk_cur=%d n_heads=%d batch=%d kv_size=%d idx_type=%s src_type=%s row_bytes=%u page_bytes=%u payload_bytes_per_page=%u scale_bytes_per_page=%u v144_ne=(%lld,%lld,%lld,%lld) v144_nb=(%zu,%zu,%zu,%zu) src_token_stride=%lld src_head_stride=%lld src_nb=(%zu,%zu,%zu,%zu) nibble_slot_atomic=1 scale_slot_merge=1 layout_note=v4_144_payload_words_pack_8_slots\n",
            txn_v_page_merge_active ? 1 : 0,
            (txn_v_geom_status == MTP_V4_144_TAIL_STAGE_OK && txn_v_stride_ok && (txn_v_idx_probe.capture_skipped || (txn_v_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK && txn_v_idx_probe.contiguous))) ? 1 : 0,
            (unsigned) txn_v_geom_status,
            (unsigned) txn_v_idx_probe.status,
            txn_v_idx_probe.copied ? 1 : 0,
            txn_v_idx_probe.capture_skipped ? 1 : 0,
            txn_v_idx_probe.contiguous ? 1 : 0,
            txn_v_idx_probe.capacity_ok ? 1 : 0,
            txn_v_idx_probe.idx0,
            txn_v_idx_probe.idx_last,
            txn_v_idx_probe.page_base,
            txn_v_idx_probe.page_end,
            txn_v_idx_probe.slot_begin,
            txn_v_idx_probe.slot_end_excl,
            txn_v_idx_probe.merge_copy_slots,
            txn_v_idx_probe.spans_pages ? 1 : 0,
            nk_cur,
            n_heads,
            batch,
            kv_size,
            v_idxs->type == GGML_TYPE_I64 ? "i64" : "i32",
            ggml_type_name(v_cur->type),
            MTP_V4_144_ROW_BYTES,
            MTP_V4_144_PAGE_BYTES,
            MTP_V4_144_PAYLOAD_BYTES_PER_PAGE,
            MTP_V4_144_SCALE_BYTES_PER_PAGE,
            (long long) v144->ne[0], (long long) v144->ne[1], (long long) v144->ne[2], (long long) v144->ne[3],
            v144->nb[0], v144->nb[1], v144->nb[2], v144->nb[3],
            (long long) src_token_stride_bytes,
            (long long) src_head_stride_bytes,
            v_cur->nb[0], v_cur->nb[1], v_cur->nb[2], v_cur->nb[3]);
        if (txn_v_idx_probe.copied && txn_v_idx_probe.status == MTP_V4_144_TAIL_STAGE_OK) {
            mtp_v4_144_tail_page_desc_v1 addr_desc = {};
            addr_desc.v4_head_stride_bytes = (uint64_t) v144->nb[2];
            addr_desc.v4_page_stride_bytes = (uint64_t) v144->nb[1] * MTP_V4_144_PAGE_TOKENS;
            addr_desc.v4_batch_stride_bytes = (uint64_t) v144->nb[3];
            const uint32_t physical_page = txn_v_idx_probe.page_base / MTP_V4_144_PAGE_TOKENS;
            const uint32_t first_slot = txn_v_idx_probe.slot_begin;
            const uint32_t last_slot = txn_v_idx_probe.slot_end_excl - 1u;
            const uint64_t payload_first = mtp_v4_144_tail_page_v4_payload_byte_offset(addr_desc, physical_page, first_slot, 0, 0, 0);
            const uint64_t payload_last_end = mtp_v4_144_tail_page_v4_payload_byte_offset(addr_desc, physical_page, last_slot, MTP_V4_144_D - 1u, 0, 0) + sizeof(uint32_t);
            const uint64_t scale_first = mtp_v4_144_tail_page_v4_scale_byte_offset(addr_desc, physical_page, first_slot, 0, 0, 0);
            const uint64_t scale_last_end = mtp_v4_144_tail_page_v4_scale_byte_offset(addr_desc, physical_page, last_slot, MTP_V4_144_D - MTP_V4_144_D32, 0, 0) + sizeof(uint16_t);
            fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_ADDR: kind=V active=%d eligible=%d page=[%u,%u) slots=[%u,%u) physical_page=%u payload_byte_span=[%llu,%llu) scale_byte_span=[%llu,%llu) v4_page_stride=%llu v4_head_stride=%llu payload_bytes_per_page=%u scale_bytes_per_page=%u payload_word_groups=2 slot_group_first=%u slot_group_last=%u scale_slot_merge=1\n",
                txn_v_page_merge_active ? 1 : 0,
                (!txn_v_idx_probe.spans_pages && txn_v_idx_probe.capacity_ok && txn_v_stride_ok) ? 1 : 0,
                txn_v_idx_probe.page_base,
                txn_v_idx_probe.page_end,
                txn_v_idx_probe.slot_begin,
                txn_v_idx_probe.slot_end_excl,
                physical_page,
                (unsigned long long) payload_first,
                (unsigned long long) payload_last_end,
                (unsigned long long) scale_first,
                (unsigned long long) scale_last_end,
                (unsigned long long) addr_desc.v4_page_stride_bytes,
                (unsigned long long) addr_desc.v4_head_stride_bytes,
                MTP_V4_144_PAYLOAD_BYTES_PER_PAGE,
                MTP_V4_144_SCALE_BYTES_PER_PAGE,
                first_slot >> 3,
                last_slot >> 3);
            }
        }
    }
    dim3 grid(nk_cur, n_heads, batch);
    dim3 block(D);
    const int scratch_page_base = (kv_size >= 2 * (int) MTP_V4_144_PAGE_TOKENS)
        ? ((kv_size / (int) MTP_V4_144_PAGE_TOKENS) - 1) * (int) MTP_V4_144_PAGE_TOKENS
        : -1;
    uint32_t txn_tail_owned_reserved_physical_page = 0;
    ggml_cuda_mtp_qblock_tail_page_map_v1 txn_tail_owned_reserved_map = {};
    const bool txn_tail_owned_reserved_map_ok = txn_v_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        llama_kv_cache_mtp_qblock_tail_page_published_owned_map_covers(
            txn_v_idx_probe.idx0, (uint32_t) nk_cur, &txn_tail_owned_reserved_physical_page, &txn_tail_owned_reserved_map);
    const bool txn_tail_owned_require_prepublished_map = txn_v_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        ggml_cuda_mtp_qblock_owned_tail_write_require_prepublished_map_requested();
    const bool txn_tail_owned_missing_required_map = txn_tail_owned_require_prepublished_map &&
        !txn_tail_owned_reserved_map_ok;
    const int txn_tail_owned_page_base = txn_tail_owned_reserved_map_ok ?
        (int) (txn_tail_owned_reserved_physical_page * (uint32_t) MTP_V4_144_PAGE_TOKENS) :
        (txn_tail_owned_missing_required_map ? -1 : scratch_page_base);
    const bool txn_tail_owned_write_active = txn_v_page_merge_active &&
        ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
        !txn_tail_owned_missing_required_map &&
        txn_tail_owned_page_base >= 0 &&
        (uint32_t) txn_tail_owned_page_base >= txn_v_idx_probe.page_end;
    if (txn_tail_owned_missing_required_map) {
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_FAIL_CLOSED: kind=V reason=missing_prepublished_owned_map idx0=%u n_tokens=%u scratch_page_base=%d page_end=%u\n",
                txn_v_idx_probe.idx0,
                (uint32_t) nk_cur,
                scratch_page_base,
                txn_v_idx_probe.page_end);
    }
    if (txn_v_page_merge_active && ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested() &&
            ggml_cuda_mtp_qblock_owned_tail_write_plan_proof_requested()) {
        const bool writer_from_map = txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active;
        const bool legacy_fallback_owned = !txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active;
        fprintf(stderr,
                "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_PLAN_PROOF: op=writer_authority kind=V writer_from_map=%d legacy_fallback_owned=%d owned_write_active=%d map_generation=%llu physical_page_from_map=%u overlay_page_base=%d idx0=%u n_tokens=%u map_logical_base=%u map_valid_tail=%u table0=%d\n",
                writer_from_map ? 1 : 0,
                legacy_fallback_owned ? 1 : 0,
                txn_tail_owned_write_active ? 1 : 0,
                (unsigned long long) (txn_tail_owned_reserved_map_ok ? txn_tail_owned_reserved_map.generation : 0ull),
                txn_tail_owned_reserved_map_ok ? txn_tail_owned_reserved_physical_page : 0xffffffffu,
                txn_tail_owned_write_active ? txn_tail_owned_page_base : -1,
                txn_v_idx_probe.idx0,
                (uint32_t) nk_cur,
                txn_tail_owned_reserved_map.logical_base_token,
                txn_tail_owned_reserved_map.valid_tail_tokens,
                txn_tail_owned_reserved_map.block_table[0]);
    }
    if (txn_v_page_merge_active && ggml_cuda_mtp_qblock_paged_attention_owned_tail_write_requested()) {
        const char * owned_ready_reason = txn_tail_owned_reserved_map_ok ?
            (txn_tail_owned_write_active ? "ok" : "owned_page_visible") :
            (txn_tail_owned_missing_required_map ? "missing_prepublished_owned_map" : "no_owned_map");
        ggml_cuda_mtp_qblock_tail_page_record_owned_write_ready(
            "V",
            txn_tail_owned_reserved_map_ok && txn_tail_owned_write_active,
            txn_tail_owned_reserved_map,
            txn_v_idx_probe.idx0,
            (uint32_t) nk_cur,
            txn_tail_owned_reserved_physical_page,
            owned_ready_reason);
    }
    const bool txn_tail_owned_exclusive_active = txn_tail_owned_write_active &&
        ggml_cuda_mtp_qblock_tail_page_owned_write_exclusive_allowed(
            "V",
            txn_tail_owned_reserved_map,
            txn_v_idx_probe.idx0,
            (uint32_t) nk_cur,
            txn_tail_owned_reserved_physical_page);
    const bool txn_tail_scratch_map_active = txn_v_page_merge_active &&
        !txn_tail_owned_write_active &&
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_map_requested() &&
        scratch_page_base >= 0 &&
        (uint32_t) scratch_page_base >= txn_v_idx_probe.page_end;
    const int txn_tail_scratch_exclusive_min_idx = ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_min_idx();
    const bool txn_tail_scratch_exclusive_after_nomap_max = ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_after_nomap_max_requested();
    const uint32_t txn_tail_consumer_no_map_nk_max = txn_tail_scratch_exclusive_after_nomap_max ?
        llama_kv_cache_get_mtp_qblock_tail_page_consumer_no_map_nk_max() : 0u;
    const bool txn_tail_scratch_exclusive_min_ok =
        txn_tail_scratch_exclusive_min_idx <= 0 || txn_v_idx_probe.idx0 >= (uint32_t) txn_tail_scratch_exclusive_min_idx;
    const bool txn_tail_scratch_exclusive_after_nomap_ok =
        !txn_tail_scratch_exclusive_after_nomap_max ||
        (txn_tail_consumer_no_map_nk_max > 0 && txn_v_idx_probe.idx0 >= txn_tail_consumer_no_map_nk_max);
    const bool txn_tail_scratch_exclusive_route_complete_ok =
        txn_tail_scratch_exclusive_after_nomap_max ||
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_single_map_unsafe_requested();
    const bool txn_tail_scratch_exclusive_published_map_ok = txn_tail_scratch_map_active &&
        llama_kv_cache_mtp_qblock_tail_page_published_scratch_map_covers(txn_v_idx_probe.idx0, (uint32_t) nk_cur);
    const bool txn_tail_scratch_exclusive_active = txn_tail_scratch_map_active &&
        ggml_cuda_mtp_qblock_txn_tail_page_scratch_exclusive_requested() &&
        txn_tail_scratch_exclusive_route_complete_ok &&
        txn_tail_scratch_exclusive_min_ok && txn_tail_scratch_exclusive_after_nomap_ok &&
        txn_tail_scratch_exclusive_published_map_ok;

    ggml_cuda_packed16_timing_trace_event pack_v4_144_timing;
    ggml_cuda_packed16_timing_trace_begin(pack_v4_144_timing, stream, "pack_v4_144");

    if (txn_v_page_merge_active) {
        ggml_cuda_pool & pool = ctx.pool();
        if (txn_tail_owned_write_active || txn_tail_scratch_map_active) {
            const int overlay_page_base = txn_tail_owned_write_active ? txn_tail_owned_page_base : scratch_page_base;
            const bool owned_overlay_dual_candidate = txn_tail_owned_write_active &&
                !txn_tail_owned_exclusive_active && !txn_tail_scratch_exclusive_active &&
                ggml_cuda_mtp_qblock_owned_tail_write_dual_dest_candidate_requested();
            if (!txn_tail_owned_exclusive_active && !txn_tail_scratch_exclusive_active) {
                const int dual_overlay_page_base = owned_overlay_dual_candidate ? overlay_page_base : -1;
                if (v_idxs->type == GGML_TYPE_I64) {
                    ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                        (const char *) v_cur->data, (char *) v144->data, (const int64_t *) v_idxs->data,
                        src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                        v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, dual_overlay_page_base);
                } else {
                    GGML_ASSERT(v_idxs->type == GGML_TYPE_I32);
                    ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                        (const char *) v_cur->data, (char *) v144->data, (const int32_t *) v_idxs->data,
                        src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                        v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, dual_overlay_page_base);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            if (owned_overlay_dual_candidate) {
                // The canonical pack above also populated the owned overlay.
            } else if (v_idxs->type == GGML_TYPE_I64) {
                ggml_cuda_pool_alloc<int64_t> local_idxs_alloc(pool);
                int64_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int64_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, overlay_page_base);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                    (const char *) v_cur->data, (char *) v144->data, local_idxs,
                    src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                    v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, -1);
            } else {
                GGML_ASSERT(v_idxs->type == GGML_TYPE_I32);
                ggml_cuda_pool_alloc<int32_t> local_idxs_alloc(pool);
                int32_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int32_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, overlay_page_base);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                    (const char *) v_cur->data, (char *) v144->data, local_idxs,
                    src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                    v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, -1);
            }
            const int poison_min_idx = ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_min_idx();
            const int poison_max_idx = ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_max_idx();
            if (txn_tail_owned_write_active && ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_v_requested() &&
                    ggml_cuda_mtp_qblock_owned_tail_write_poison_canonical_range_hits(
                        txn_v_idx_probe.idx0, (uint32_t) nk_cur, poison_min_idx, poison_max_idx)) {
                if (v_idxs->type == GGML_TYPE_I64) {
                    ggml_cuda_mtp_qblock_poison_canonical_v4_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                        (char *) v144->data,
                        (const int64_t *) v_idxs->data,
                        v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, poison_min_idx, poison_max_idx);
                } else {
                    GGML_ASSERT(v_idxs->type == GGML_TYPE_I32);
                    ggml_cuda_mtp_qblock_poison_canonical_v4_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                        (char *) v144->data,
                        (const int32_t *) v_idxs->data,
                        v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, poison_min_idx, poison_max_idx);
                }
                CUDA_CHECK(cudaGetLastError());
                fprintf(stderr,
                    "MTP_QBLOCK_TXN_TAIL_PAGE_OWNED_CANONICAL_POISON: kind=V status=ok reason=owned_write_active idx0=%u n_tokens=%u poison_min_idx=%d poison_max_idx=%d canonical_page_base=%u owned_page_base=%d physical_page=%u map_logical_base=%u map_valid_tail=%u table0=%d flags=0x%x generation=%llu\n",
                    txn_v_idx_probe.idx0,
                    (uint32_t) nk_cur,
                    poison_min_idx,
                    poison_max_idx,
                    txn_v_idx_probe.page_base,
                    txn_tail_owned_page_base,
                    txn_tail_owned_reserved_physical_page,
                    txn_tail_owned_reserved_map.logical_base_token,
                    txn_tail_owned_reserved_map.valid_tail_tokens,
                    txn_tail_owned_reserved_map.block_table[0],
                    txn_tail_owned_reserved_map.flags,
                    (unsigned long long) txn_tail_owned_reserved_map.generation);
            }
            if (txn_tail_owned_write_active && overlay_page_base >= 0) {
                const size_t scratch_bytes = size_t(batch) * size_t(n_heads) * size_t(GGML_CUDA_V4_K16D16_144_BLOCK_BYTES);
                ggml_cuda_pool_alloc<char> owned_page_alloc(pool);
                char * owned_page = owned_page_alloc.alloc(scratch_bytes);

                constexpr int v4_page_words = GGML_CUDA_V4_K16D16_144_BLOCK_BYTES / (int) sizeof(uint32_t);
                dim3 page_grid((v4_page_words + 255) / 256, n_heads, batch);
                dim3 page_block(256);
                ggml_cuda_mtp_qblock_txn_tail_v4_144_copy_page_to_scratch_kernel<<<page_grid, page_block, 0, stream>>>(
                    (const char *) v144->data,
                    owned_page,
                    (int) txn_v_idx_probe.page_base,
                    v144->nb[1],
                    v144->nb[2],
                    kv_size,
                    n_heads);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_mtp_qblock_txn_tail_v4_144_commit_scratch_page_kernel<<<page_grid, page_block, 0, stream>>>(
                    owned_page,
                    (char *) v144->data,
                    overlay_page_base,
                    v144->nb[1],
                    v144->nb[2],
                    kv_size,
                    n_heads);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            const size_t scratch_bytes = size_t(batch) * size_t(n_heads) * size_t(GGML_CUDA_V4_K16D16_144_BLOCK_BYTES);
            ggml_cuda_pool_alloc<char> scratch_page_alloc(pool);
            char * scratch_page = scratch_page_alloc.alloc(scratch_bytes);

            constexpr int v4_page_words = GGML_CUDA_V4_K16D16_144_BLOCK_BYTES / (int) sizeof(uint32_t);
            dim3 page_grid((v4_page_words + 255) / 256, n_heads, batch);
            dim3 page_block(256);
            ggml_cuda_mtp_qblock_txn_tail_v4_144_copy_page_to_scratch_kernel<<<page_grid, page_block, 0, stream>>>(
                (const char *) v144->data,
                scratch_page,
                (int) txn_v_idx_probe.page_base,
                v144->nb[1],
                v144->nb[2],
                kv_size,
                n_heads);
            CUDA_CHECK(cudaGetLastError());

            if (v_idxs->type == GGML_TYPE_I64) {
                ggml_cuda_pool_alloc<int64_t> local_idxs_alloc(pool);
                int64_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int64_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, (int) txn_v_idx_probe.slot_begin);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
                    (const char *) v_cur->data, scratch_page, local_idxs,
                    src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                    GGML_CUDA_V4_K16D16_144_ROW_BYTES,
                    int64_t(n_heads) * GGML_CUDA_V4_K16D16_144_BLOCK_BYTES,
                    nk_cur, n_heads, batch, MTP_V4_144_PAGE_TOKENS, src_f16, -1);
            } else {
                GGML_ASSERT(v_idxs->type == GGML_TYPE_I32);
                ggml_cuda_pool_alloc<int32_t> local_idxs_alloc(pool);
                int32_t * local_idxs = local_idxs_alloc.alloc((size_t) nk_cur);
                ggml_cuda_mtp_qblock_txn_tail_k_fill_local_idxs_kernel<int32_t><<<1, 32, 0, stream>>>(
                    local_idxs, nk_cur, (int) txn_v_idx_probe.slot_begin);
                CUDA_CHECK(cudaGetLastError());
                ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
                    (const char *) v_cur->data, scratch_page, local_idxs,
                    src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
                    GGML_CUDA_V4_K16D16_144_ROW_BYTES,
                    int64_t(n_heads) * GGML_CUDA_V4_K16D16_144_BLOCK_BYTES,
                    nk_cur, n_heads, batch, MTP_V4_144_PAGE_TOKENS, src_f16, -1);
            }
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_mtp_qblock_txn_tail_v4_144_commit_scratch_page_kernel<<<page_grid, page_block, 0, stream>>>(
                scratch_page,
                (char *) v144->data,
                (int) txn_v_idx_probe.page_base,
                v144->nb[1],
                v144->nb[2],
                kv_size,
                n_heads);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (v_idxs->type == GGML_TYPE_I64) {
        ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int64_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v144->data, (const int64_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, -1);
    } else {
        ggml_cuda_pack_v4_k16d16_144_indexed_kernel<int32_t><<<grid, block, 0, stream>>>(
            (const char *) v_cur->data, (char *) v144->data, (const int32_t *) v_idxs->data,
            src_token_stride_bytes, v_cur->nb[2], v_cur->nb[3], src_head_stride_bytes,
            v144->nb[1], v144->nb[2], nk_cur, n_heads, batch, kv_size, src_f16, -1);
    }
    CUDA_CHECK(cudaGetLastError());
    const float pack_v4_144_timing_ms = ggml_cuda_packed16_timing_trace_end(pack_v4_144_timing, stream);
    if (pack_v4_144_timing_ms >= 0.0f) {
        fprintf(stderr,
            "PACKED16_TIMING_TRACE phase=pack_v4_144 ms=%.3f nk_cur=%d n_heads=%d batch=%d kv_size=%d src_type=%s idx_type=%s row_bytes=%d txn_merge=%d owned=%d scratch=%d\n",
            (double) pack_v4_144_timing_ms,
            nk_cur,
            n_heads,
            batch,
            kv_size,
            ggml_type_name(v_cur->type),
            v_idxs->type == GGML_TYPE_I64 ? "i64" : "i32",
            GGML_CUDA_V4_K16D16_144_ROW_BYTES,
            txn_v_page_merge_active ? 1 : 0,
            txn_tail_owned_write_active ? 1 : 0,
            txn_tail_scratch_map_active ? 1 : 0);
    }
}

#endif // GGML_USE_HIP
