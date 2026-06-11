#include "mmvq.cuh"
#include "mmvq-rdna3-dot4.cuh"
#include "mmvq-moe-q8-dot4.cuh"
#include "dot4-packed16/mmvq/dp16-mmvq-q8-gemv.cuh"
#include "dot4-packed16/mmvq/dp16-mmvq-packed16-gemv.cuh"
#include "dot4-packed16/dp16-trace.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static bool ggml_cuda_mtp_mmvq_serial_columns_disabled(const char * env) {
    return env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0;
}

struct ggml_cuda_mtp_mmvq_route_census_entry {
    std::string route;
    std::string tensor;
    std::string type;
    int64_t ncols_x = 0;
    int64_t nrows_x = 0;
    int64_t ncols_dst = 0;
    bool has_ids = false;
    bool has_fusion = false;
    uint64_t calls = 0;
    uint64_t approx_outputs = 0;
};

static bool ggml_cuda_mtp_mmvq_route_census_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_MTP_MMVQ_ROUTE_CENSUS");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
    }();
    return enabled;
}

static std::mutex & ggml_cuda_mtp_mmvq_route_census_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::string, ggml_cuda_mtp_mmvq_route_census_entry> & ggml_cuda_mtp_mmvq_route_census_map() {
    // Keep census storage alive until process exit; the server shutdown path can
    // terminate after C++ static destructors would otherwise clear a local map
    // before the atexit summary hook runs.
    static auto * map = new std::unordered_map<std::string, ggml_cuda_mtp_mmvq_route_census_entry>();
    return *map;
}

static void ggml_cuda_mtp_mmvq_route_census_dump() {
    if (!ggml_cuda_mtp_mmvq_route_census_enabled()) {
        return;
    }

    std::vector<ggml_cuda_mtp_mmvq_route_census_entry> rows;
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_mtp_mmvq_route_census_mutex());
        rows.reserve(ggml_cuda_mtp_mmvq_route_census_map().size());
        for (const auto & kv : ggml_cuda_mtp_mmvq_route_census_map()) {
            rows.push_back(kv.second);
        }
    }

    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.approx_outputs != b.approx_outputs) {
            return a.approx_outputs > b.approx_outputs;
        }
        return a.calls > b.calls;
    });

    GGML_LOG_INFO("mtp_mmvq_route_census summary count=%zu\n", rows.size());
    const size_t limit = std::min<size_t>(rows.size(), 96);
    for (size_t i = 0; i < limit; ++i) {
        const auto & r = rows[i];
        GGML_LOG_INFO("mtp_mmvq_route_census rank=%zu route=%s tensor=%s type=%s ncols_dst=%lld ncols_x=%lld nrows_x=%lld ids=%d fusion=%d calls=%llu approx_outputs=%llu\n",
                i + 1, r.route.c_str(), r.tensor.c_str(), r.type.c_str(),
                (long long) r.ncols_dst, (long long) r.ncols_x, (long long) r.nrows_x,
                r.has_ids ? 1 : 0, r.has_fusion ? 1 : 0,
                (unsigned long long) r.calls, (unsigned long long) r.approx_outputs);
    }
}

static void ggml_cuda_mtp_mmvq_route_census_record(
        const char * route,
        const ggml_tensor * src0,
        const int64_t ncols_x,
        const int64_t nrows_x,
        const int64_t ncols_dst,
        const bool has_ids,
        const bool has_fusion) {
    if (!ggml_cuda_mtp_mmvq_route_census_enabled()) {
        return;
    }

    static const bool registered = []() {
        std::atexit(ggml_cuda_mtp_mmvq_route_census_dump);
        return true;
    }();
    GGML_UNUSED(registered);

    const char * tensor = src0 && src0->name[0] != '\0' ? src0->name : "-";
    const char * type = src0 ? ggml_type_name(src0->type) : "-";
    std::string key = std::string(route ? route : "-") + "|" + tensor + "|" + type + "|" +
        std::to_string((long long) ncols_dst) + "|" + std::to_string((long long) ncols_x) + "|" +
        std::to_string((long long) nrows_x) + "|" + (has_ids ? "ids" : "noids") + "|" +
        (has_fusion ? "fusion" : "nofusion");

    std::lock_guard<std::mutex> lock(ggml_cuda_mtp_mmvq_route_census_mutex());
    auto & entry = ggml_cuda_mtp_mmvq_route_census_map()[key];
    if (entry.calls == 0) {
        entry.route = route ? route : "-";
        entry.tensor = tensor;
        entry.type = type;
        entry.ncols_x = ncols_x;
        entry.nrows_x = nrows_x;
        entry.ncols_dst = ncols_dst;
        entry.has_ids = has_ids;
        entry.has_fusion = has_fusion;
    }
    entry.calls++;
    entry.approx_outputs += (uint64_t) std::max<int64_t>(nrows_x, 0) * (uint64_t) std::max<int64_t>(ncols_dst, 0);
}

static int ggml_cuda_mtp_mmvq_serial_columns_max() {
    const char * env = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_MAX");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && v > 0 ? (int) v : 4;
}

static bool ggml_cuda_mtp_mmvq_serial_columns_token_all(const char * begin, size_t len) {
    return (len == 1 && begin[0] == '1') ||
           (len == 2 && strncmp(begin, "on",   len) == 0) ||
           (len == 3 && strncmp(begin, "all",  len) == 0) ||
           (len == 4 && strncmp(begin, "true", len) == 0);
}

static bool ggml_cuda_mtp_mmvq_serial_columns_token_match(const char * name, const char * begin, const char * end) {
    while (begin < end && (*begin == ' ' || *begin == '\t' || *begin == '\n' || *begin == '\r')) {
        ++begin;
    }
    while (end > begin && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) {
        --end;
    }

    const size_t len = (size_t) (end - begin);
    if (len == 0) {
        return false;
    }
    if (ggml_cuda_mtp_mmvq_serial_columns_token_all(begin, len)) {
        return true;
    }
    return name != nullptr && std::string(name).find(std::string(begin, len)) != std::string::npos;
}

static bool ggml_cuda_mtp_mmvq_serial_columns_match_name(const char * env, const char * name) {
    if (env == nullptr || env[0] == '\0') {
        return true;
    }

    bool has_include   = false;
    bool include_match = false;

    const char * p = env;
    while (*p != '\0') {
        const char * end = strchr(p, ',');
        if (end == nullptr) {
            end = p + strlen(p);
        }

        const char * begin = p;
        while (begin < end && (*begin == ' ' || *begin == '\t' || *begin == '\n' || *begin == '\r')) {
            ++begin;
        }

        bool exclude = false;
        if (begin < end && (*begin == '!' || *begin == '-')) {
            exclude = true;
            ++begin;
        }

        const bool token_match = ggml_cuda_mtp_mmvq_serial_columns_token_match(name, begin, end);
        if (exclude) {
            if (token_match) {
                return false;
            }
        } else if (begin < end) {
            has_include = true;
            include_match = include_match || token_match;
        }

        p = *end == '\0' ? end : end + 1;
    }

    return has_include ? include_match : true;
}

static bool ggml_cuda_mtp_mmvq_serial_columns_active() {
    const char * global = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_GLOBAL");
    if (!ggml_cuda_mtp_mmvq_serial_columns_disabled(global)) {
        return true;
    }

    const char * active = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE");
    return !ggml_cuda_mtp_mmvq_serial_columns_disabled(active);
}

static const char * ggml_cuda_mtp_mmvq_serial_columns_filter() {
    const char * active_filter = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_ACTIVE_FILTER");
    if (!ggml_cuda_mtp_mmvq_serial_columns_disabled(active_filter)) {
        return active_filter;
    }

    return getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS");
}

static bool ggml_cuda_mtp_mmvq_serial_columns_ids_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_IDS");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_serial_columns_enabled(
        const ggml_tensor * src0,
        int64_t             ncols_dst,
        bool                has_ids,
        bool                has_fusion) {
    GGML_UNUSED(has_fusion);
    const char * env = ggml_cuda_mtp_mmvq_serial_columns_filter();
    if (ggml_cuda_mtp_mmvq_serial_columns_disabled(env)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_serial_columns_active()) {
        return false;
    }
    if (ncols_dst <= 1 || ncols_dst > ggml_cuda_mtp_mmvq_serial_columns_max()) {
        return false;
    }
    if (has_ids && !ggml_cuda_mtp_mmvq_serial_columns_ids_enabled()) {
        return false;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(env, src0 ? src0->name : nullptr);
}

static bool ggml_cuda_mtp_mmvq_serial_columns_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_serial_columns_single_launch_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_SERIAL_COLUMNS_SINGLE_LAUNCH");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q6k_reuse_weight_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q6k_reuse_weight_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q6k_reuse_weight_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_q6k_reuse_weight_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ncols_dst > 1;
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static int ggml_cuda_mtp_mmvq_q6k_reuse_weight_nwarps() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_NWARPS");
    if (env == nullptr || env[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && (v == 2 || v == 4) ? (int) v : 1;
}

static int ggml_cuda_mtp_mmvq_q6k_reuse_weight_rows() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_ROWS");
    if (env == nullptr || env[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && v == 2 ? 2 : 1;
}

static bool ggml_cuda_mtp_mmvq_q6k_interleaved_act_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT");
    return env == nullptr || env[0] == '\0' || (strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0);
}

static bool ggml_cuda_mtp_mmvq_q6k_interleaved_act_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static const char * ggml_cuda_mtp_mmvq_q6k_interleaved_act_filter() {
    const char * filter = getenv("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_FILTER");
    if (filter != nullptr && filter[0] != '\0') {
        return filter;
    }
    return getenv("LLAMA_MTP_MMVQ_Q6K_REUSE_WEIGHT_FILTER");
}

static bool ggml_cuda_mtp_mmvq_q6k_interleaved_act_name_allowed(const char * name) {
    const char * filter = ggml_cuda_mtp_mmvq_q6k_interleaved_act_filter();
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_q6k_interleaved_act_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ggml_cuda_mtp_mmvq_q6k_reuse_weight_ncols_allowed(ncols_dst);
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static int ggml_cuda_mtp_mmvq_interleaved_act_rows(const char * env_name) {
    const char * env = getenv(env_name);
    if (env == nullptr || env[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && v == 2 ? 2 : 1;
}

static int ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw(const char * env_name) {
    const char * env = getenv(env_name);
    if (env == nullptr || env[0] == '\0') {
        return strcmp(env_name, "LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_NWARPS") == 0 ? 2 : 1;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && (v == 2 || v == 4) ? (int) v : 1;
}

static bool ggml_cuda_mtp_mmvq_interleaved_act_multi_type_nwarps_unsafe() {
    const char * env = getenv("LLAMA_MTP_MMVQ_INTERLEAVED_ACT_MULTI_TYPE_NWARPS_UNSAFE");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static int ggml_cuda_mtp_mmvq_interleaved_act_nwarps(const char * env_name) {
    const int requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw(env_name);
    if (requested == 1 || ggml_cuda_mtp_mmvq_interleaved_act_multi_type_nwarps_unsafe()) {
        return requested;
    }

    const int n_multiwarps =
        (ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_NWARPS") > 1 ? 1 : 0) +
        (ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_NWARPS") > 1 ? 1 : 0) +
        (ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_NWARPS") > 1 ? 1 : 0) +
        (ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_NWARPS") > 1 ? 1 : 0) +
        (ggml_cuda_mtp_mmvq_interleaved_act_nwarps_raw("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_NWARPS") > 1 ? 1 : 0);
    return n_multiwarps > 1 ? 1 : requested;
}

static bool ggml_cuda_mtp_mmvq_legacy_interleaved_act_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_legacy_interleaved_act_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_legacy_interleaved_act_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_legacy_interleaved_act_type_allowed(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }

    const char * filter = getenv("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_TYPES");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, ggml_type_name(type));
}

static bool ggml_cuda_mtp_mmvq_legacy_interleaved_act_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ncols_dst > 1 && ncols_dst <= 5;
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static bool ggml_cuda_mtp_mmvq_lowk_interleaved_act_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_lowk_interleaved_act_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_lowk_interleaved_act_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_lowk_interleaved_act_type_allowed(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            break;
        default:
            return false;
    }

    const char * filter = getenv("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_TYPES");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, ggml_type_name(type));
}

static bool ggml_cuda_mtp_mmvq_lowk_interleaved_act_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ncols_dst > 1 && ncols_dst <= 5;
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static bool ggml_cuda_mtp_mmvq_q4k_interleaved_act_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT");
    return env == nullptr || env[0] == '\0' || (strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0);
}

static bool ggml_cuda_mtp_mmvq_q4k_interleaved_act_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q4k_interleaved_act_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_q4k_interleaved_act_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ncols_dst > 1 && ncols_dst <= 5;
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static bool ggml_cuda_mtp_mmvq_q5k_interleaved_act_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q5k_interleaved_act_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_q5k_interleaved_act_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_q5k_interleaved_act_ncols_allowed(const int ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return ncols_dst > 1 && ncols_dst <= 5;
    }
    const char * p = env;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t' || *p == ',') {
            ++p;
        }
        const char * begin = p;
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) {
            --end;
        }
        if (end - begin == 3 && strncmp(begin, "all", 3) == 0) {
            return true;
        }
        char buf[16] = {};
        const size_t len = std::min<size_t>((size_t) (end - begin), sizeof(buf) - 1);
        memcpy(buf, begin, len);
        char * parse_end = nullptr;
        const long v = strtol(buf, &parse_end, 10);
        if (parse_end != buf && v == ncols_dst) {
            return true;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return false;
}

static int ggml_cuda_mtp_mmvq_moe_exp_tile() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_EXP_TILE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0) {
        return 0;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    if (end == env) {
        return 4;
    }
    return v == 2 || v == 4 ? (int) v : 0;
}

static bool ggml_cuda_mtp_mmvq_moe_exp_tile_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_EXP_TILE_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static int ggml_cuda_mtp_mmvq_moe_exp_tile_rows() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_EXP_TILE_ROWS");
    if (env == nullptr || env[0] == '\0') {
        return 2;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && (v == 1 || v == 2 || v == 4) ? (int) v : 2;
}


static bool ggml_cuda_mtp_mmvq_moe_exp_tile_shape_allowed(const int ncols_x, const int nrows_x) {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_EXP_TILE_KIND");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "all") == 0 || strcmp(env, "1") == 0 || strcmp(env, "true") == 0) {
        return true;
    }
    if (strcmp(env, "gateup") == 0 || strcmp(env, "gate_up") == 0) {
        return ncols_x >= nrows_x;
    }
    if (strcmp(env, "down") == 0) {
        return ncols_x < nrows_x;
    }
    return false;
}

static bool ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static bool ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_log_enabled() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_LOG");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

static int ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_min_ncols() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_MIN_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && v > 0 ? (int) v : 1;
}

static int ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_max_ncols() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_MAX_NCOLS");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && v > 0 ? (int) v : 4;
}

static int ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_rows() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_ROWS");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && (v == 1 || v == 2 || v == 4) ? (int) v : 4;
}

static int ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_exp_tile() {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_EXP_TILE");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    char * end = nullptr;
    const long v = strtol(env, &end, 10);
    return end != env && (v == 2 || v == 4) ? (int) v : 4;
}

static bool ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_name_allowed(const char * name) {
    const char * filter = getenv("LLAMA_MTP_MMVQ_MOE_IQ3S_SIDECAR_FILTER");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, name);
}

static bool ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_shape_allowed(
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int nchannels_y, const int nchannels_dst) {
    return ncols_x == 2048 && nrows_x == 1024 &&
        ncols_dst >= ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_min_ncols() &&
        ncols_dst <= ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_max_ncols() &&
        nchannels_y == 1 && nchannels_dst == 8;
}

struct ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace {
    int32_t * packs        = nullptr;
    uint8_t * scales       = nullptr;
    half    * ds           = nullptr;
    int32_t * unique_ids   = nullptr;
    int32_t * ids_local    = nullptr;
    int32_t * unique_count = nullptr;

    size_t packs_bytes        = 0;
    size_t scales_bytes       = 0;
    size_t ds_bytes           = 0;
    size_t unique_ids_bytes   = 0;
    size_t ids_local_bytes    = 0;
    size_t unique_count_bytes = 0;

    ~ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace() {
        if (packs)        { (void) cudaFree(packs); }
        if (scales)       { (void) cudaFree(scales); }
        if (ds)           { (void) cudaFree(ds); }
        if (unique_ids)   { (void) cudaFree(unique_ids); }
        if (ids_local)    { (void) cudaFree(ids_local); }
        if (unique_count) { (void) cudaFree(unique_count); }
    }
};

static thread_local ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace g_ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace;

static void ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc(void ** ptr, size_t * capacity, const size_t needed) {
    if (*capacity >= needed) {
        return;
    }
    if (*ptr != nullptr) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
        *capacity = 0;
    }
    if (needed > 0) {
        CUDA_CHECK(cudaMalloc(ptr, needed));
        *capacity = needed;
    }
}

static ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace & ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace_get(
        const size_t max_unique, const size_t nrows_x, const size_t blocks_per_row_x,
        size_t * sidecar_total_bytes) {
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace & ws = g_ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace;

    constexpr size_t iq3s_sidecar_iqs_groups = 8;
    constexpr size_t iq3s_sidecar_packs_per_iqs = 8;
    const size_t sidecar_blocks = max_unique * nrows_x * blocks_per_row_x;
    const size_t packs_bytes = sidecar_blocks * iq3s_sidecar_iqs_groups * iq3s_sidecar_packs_per_iqs * sizeof(int32_t);
    const size_t scales_bytes = sidecar_blocks * iq3s_sidecar_iqs_groups * sizeof(uint8_t);
    const size_t ds_bytes = sidecar_blocks * sizeof(half);
    const size_t ids_bytes = max_unique * sizeof(int32_t);
    const size_t count_bytes = sizeof(int32_t);

    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.packs,        &ws.packs_bytes,        packs_bytes);
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.scales,       &ws.scales_bytes,       scales_bytes);
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.ds,           &ws.ds_bytes,           ds_bytes);
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.unique_ids,   &ws.unique_ids_bytes,   ids_bytes);
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.ids_local,    &ws.ids_local_bytes,    ids_bytes);
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_ensure_alloc((void **) &ws.unique_count, &ws.unique_count_bytes, count_bytes);

    if (sidecar_total_bytes) {
        *sidecar_total_bytes = packs_bytes + scales_bytes + ds_bytes;
    }
    return ws;
}

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

static inline bool ggml_cuda_dp16_get_packed16_weight_if_ready(
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
    if (it == cache.end() || !it->second.ready_recorded || !it->second.ready) {
        return false;
    }

#if defined(GGML_USE_HIP)
    const hipError_t err = hipEventQuery(it->second.ready);
    if (err == hipErrorNotReady) {
        return false;
    }
    CUDA_CHECK(err);
#else
    const cudaError_t err = cudaEventQuery(it->second.ready);
    if (err == cudaErrorNotReady) {
        return false;
    }
    CUDA_CHECK(err);
#endif

    *out = it->second.view;
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
        const bool fusion_x_bias_only,
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
    problem.is_decode = ncols_dst <= dp16_mmvq_packed16_runtime_max_n();
    problem.has_fusion = has_fusion;
    problem.fusion_x_bias_only = fusion_x_bias_only;
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

static thread_local bool g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope = false;
static thread_local const char * g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name = nullptr;

struct ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope_guard {
    bool old;
    const char * old_tensor_name;
    ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope_guard(const bool enabled, const char * tensor_name)
        : old(g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope),
          old_tensor_name(g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name) {
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope = enabled;
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name = enabled ? tensor_name : nullptr;
    }
    ~ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope_guard() {
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope = old;
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name = old_tensor_name;
    }
};

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ");
    return !enabled || atoi(enabled) != 0;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_log_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_LOG");
    return (enabled && atoi(enabled) != 0) || dp16_trace_enabled();
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(
        const char * func,
        const char * route,
        const ggml_tensor * tensor,
        const char * reject,
        const int ncols_x,
        const int nrows_x,
        const int ncols_dst,
        const bool has_fusion,
        const bool has_ids) {
    if (!ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        return;
    }
    GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=reject reject=%s ncols_x=%d nrows_x=%d ncols_dst=%d fusion=%d ids=%d\n",
            func, route, tensor ? tensor->name : "-", reject,
            ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0, has_ids ? 1 : 0);
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(
        const char * func,
        const char * route,
        const ggml_tensor * tensor,
        const char * reject,
        const int ncols_x,
        const int nrows_x,
        const int ncols_dst,
        const bool has_fusion,
        const bool has_ids) {
    if (!ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        return;
    }
    GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=reject reject=%s capture=1 ncols_x=%d nrows_x=%d ncols_dst=%d fusion=%d ids=%d\n",
            func, route, tensor ? tensor->name : "-", reject,
            ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0, has_ids ? 1 : 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_glu_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_GLU");
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() && (!enabled || atoi(enabled) != 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_eh_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_EH");
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() && (!enabled || atoi(enabled) != 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_down_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_DOWN");
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() && (!enabled || atoi(enabled) != 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_QKV");
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() && (!enabled || atoi(enabled) != 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_ATTN_OUT");
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() && (!enabled || atoi(enabled) != 0);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream, &capture_status));
    return capture_status != hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    return capture_status != cudaStreamCaptureStatusNone;
#endif
}

enum ggml_cuda_mtp_q8_dot4_mmvq_profile_route_id {
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_GLU = 0,
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_EH,
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_QKV,
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_ATTN_OUT,
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_DOWN,
    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_COUNT,
};

struct ggml_cuda_mtp_q8_dot4_mmvq_profile_stat {
    uint64_t count = 0;
    double total_ms = 0.0;
    float max_ms = 0.0f;
};

static std::mutex & ggml_cuda_mtp_q8_dot4_mmvq_profile_mutex() {
    static std::mutex mutex;
    return mutex;
}

static ggml_cuda_mtp_q8_dot4_mmvq_profile_stat * ggml_cuda_mtp_q8_dot4_mmvq_profile_stats() {
    static ggml_cuda_mtp_q8_dot4_mmvq_profile_stat stats[GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_COUNT];
    return stats;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_profile_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PROFILE");
    return enabled && atoi(enabled) != 0;
}

static inline int ggml_cuda_mtp_q8_dot4_mmvq_profile_every() {
    const char * every = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PROFILE_EVERY");
    if (!every || every[0] == '\0') {
        return 1;
    }
    const int value = atoi(every);
    return value > 0 ? value : 1;
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_profile_record(
        const ggml_cuda_mtp_q8_dot4_mmvq_profile_route_id route_id,
        const char * route,
        const char * tensor,
        const int64_t ncols_x,
        const int64_t nrows_x,
        const int64_t ncols_dst,
        const bool has_fusion,
        const float elapsed_ms) {
    uint64_t count = 0;
    double total_ms = 0.0;
    float max_ms = 0.0f;
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_mtp_q8_dot4_mmvq_profile_mutex());
        ggml_cuda_mtp_q8_dot4_mmvq_profile_stat & stat = ggml_cuda_mtp_q8_dot4_mmvq_profile_stats()[route_id];
        stat.count++;
        stat.total_ms += (double) elapsed_ms;
        if (elapsed_ms > stat.max_ms) {
            stat.max_ms = elapsed_ms;
        }
        count = stat.count;
        total_ms = stat.total_ms;
        max_ms = stat.max_ms;
    }

    const int every = ggml_cuda_mtp_q8_dot4_mmvq_profile_every();
    if (count <= 4 || (every > 0 && (count % (uint64_t) every) == 0)) {
        const double avg_ms = count > 0 ? total_ms / (double) count : 0.0;
        GGML_LOG_INFO("%s: mtp_weight_timing route=%s tensor=%s status=measured count=%llu last_ms=%.6f avg_ms=%.6f total_ms=%.6f max_ms=%.6f ncols_x=%lld nrows_x=%lld ncols_dst=%lld fusion=%d\n",
                __func__, route ? route : "-", tensor ? tensor : "-",
                (unsigned long long) count, (double) elapsed_ms, avg_ms, total_ms, (double) max_ms,
                (long long) ncols_x, (long long) nrows_x, (long long) ncols_dst, has_fusion ? 1 : 0);
    }
}

template <typename TryLaunch>
static inline bool ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
        const bool profile_candidate,
        const ggml_cuda_mtp_q8_dot4_mmvq_profile_route_id route_id,
        const char * route,
        const char * tensor,
        const int64_t ncols_x,
        const int64_t nrows_x,
        const int64_t ncols_dst,
        const bool has_fusion,
        cudaStream_t stream,
        const TryLaunch & try_launch) {
    if (!profile_candidate || !ggml_cuda_mtp_q8_dot4_mmvq_profile_enabled() ||
            ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream)) {
        return try_launch();
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
#if defined(GGML_USE_HIP)
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
#else
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
#endif
    CUDA_CHECK(cudaEventRecord(start, stream));
    const bool launched = try_launch();
    if (!launched) {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return false;
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
#if defined(GGML_USE_HIP)
    CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
#else
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
#endif
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    ggml_cuda_mtp_q8_dot4_mmvq_profile_record(route_id, route, tensor, ncols_x, nrows_x, ncols_dst, has_fusion, elapsed_ms);
    return true;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(const char * name) {
    if (!name || name[0] == '\0') {
        return false;
    }

    if (const char * filter = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_FILTER")) {
        if (filter[0] != '\0') {
            return strstr(name, filter) != nullptr;
        }
    }

    // Qwen3.5/3.6 MTP convention: the one NextN block is stored as the final
    // block index.  Keep this opt-in path MTP-scoped by default instead of
    // enabling q8 DOT4 MMVQ for every q8_0 trunk/expert weight.
    const char * layer = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_LAYER");
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%s.", (layer && layer[0] != '\0') ? layer : "64");
    return strstr(name, prefix) == name && strstr(name, ".weight") != nullptr;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_tensor_allowed(const ggml_tensor * src0) {
    return ggml_cuda_mtp_q8_dot4_mmvq_env_enabled() &&
        src0 && src0->type == GGML_TYPE_Q8_0 &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(src0->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_eh_tensor_allowed(const ggml_tensor * src0) {
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_eh_enabled() &&
        src0 && src0->type == GGML_TYPE_Q8_0 && src0->name[0] != '\0' &&
        strstr(src0->name, ".nextn.eh_proj.weight") != nullptr &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(src0->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_down_tensor_allowed(const ggml_tensor * src0) {
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_down_enabled() &&
        src0 && src0->type == GGML_TYPE_Q8_0 && src0->name[0] != '\0' &&
        strstr(src0->name, ".ffn_down.weight") != nullptr &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(src0->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(const ggml_tensor * src0) {
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_enabled() &&
        src0 && src0->type == GGML_TYPE_Q8_0 && src0->name[0] != '\0' &&
        (strstr(src0->name, ".attn_q.weight") != nullptr ||
         strstr(src0->name, ".attn_k.weight") != nullptr ||
         strstr(src0->name, ".attn_v.weight") != nullptr) &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(src0->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_tensor_allowed(const ggml_tensor * src0) {
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_enabled() &&
        src0 && src0->type == GGML_TYPE_Q8_0 && src0->name[0] != '\0' &&
        strstr(src0->name, ".attn_output.weight") != nullptr &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(src0->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_packed_glu_tensors_allowed(
        const ggml_tensor * up,
        const ggml_tensor * gate) {
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_glu_enabled() &&
        up && gate &&
        up->type == GGML_TYPE_Q8_0 && gate->type == GGML_TYPE_Q8_0 &&
        ggml_are_same_shape(up, gate) && ggml_are_same_stride(up, gate) &&
        up->name[0] != '\0' && gate->name[0] != '\0' &&
        strstr(up->name, ".ffn_up.weight") != nullptr &&
        strstr(gate->name, ".ffn_gate.weight") != nullptr &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(up->name) &&
        ggml_cuda_mtp_q8_dot4_mmvq_name_allowed(gate->name);
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_qkv_reuse_act_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_QKV_REUSE_ACT");
    return ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_enabled() && enabled && atoi(enabled) != 0;
}

static inline bool ggml_cuda_mtp_mmvq_reuse_act_env_enabled() {
    const char * enabled = getenv("LLAMA_MTP_MMVQ_REUSE_ACT");
    return enabled != nullptr && enabled[0] != '\0' && atoi(enabled) != 0;
}

static inline bool ggml_cuda_mtp_mmvq_reuse_act_log_enabled() {
    const char * enabled = getenv("LLAMA_MTP_MMVQ_REUSE_ACT_LOG");
    return enabled != nullptr && enabled[0] != '\0' && atoi(enabled) != 0;
}

static inline bool ggml_cuda_mtp_mmvq_reuse_act_enabled(const ggml_tensor * src0, bool has_ids) {
    if (!ggml_cuda_mtp_mmvq_reuse_act_env_enabled() || has_ids || src0 == nullptr) {
        return false;
    }
    const char * filter = getenv("LLAMA_MTP_MMVQ_REUSE_ACT_FILTER");
    return ggml_cuda_mtp_mmvq_serial_columns_match_name(filter, src0->name);
}

struct ggml_cuda_mtp_q8_dot4_mmvq_act_cache_key {
    const void * data = nullptr;
    int device = -1;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[4] = { 0, 0, 0, 0 };
    size_t nb[4] = { 0, 0, 0, 0 };
    int64_t ne0_padded = 0;

    bool operator==(const ggml_cuda_mtp_q8_dot4_mmvq_act_cache_key & other) const {
        return data == other.data && device == other.device && type == other.type && ne0_padded == other.ne0_padded &&
            memcmp(ne, other.ne, sizeof(ne)) == 0 && memcmp(nb, other.nb, sizeof(nb)) == 0;
    }
};

struct ggml_cuda_mtp_q8_dot4_mmvq_act_cache_state {
    char * data = nullptr;
    size_t bytes = 0;
    bool has_key = false;
    bool valid = false;
    cudaStream_t writer_stream = nullptr;
    ggml_cuda_mtp_q8_dot4_mmvq_act_cache_key key = {};

    ~ggml_cuda_mtp_q8_dot4_mmvq_act_cache_state() {
        if (data) {
            if (key.device >= 0) {
                ggml_cuda_set_device(key.device);
            }
            (void) cudaFree(data);
            data = nullptr;
        }
    }
};

static thread_local ggml_cuda_mtp_q8_dot4_mmvq_act_cache_state g_ggml_cuda_mtp_q8_dot4_mmvq_act_cache;

void ggml_cuda_mtp_q8_dot4_mmvq_act_cache_reset() {
    g_ggml_cuda_mtp_q8_dot4_mmvq_act_cache.valid = false;
    g_ggml_cuda_mtp_q8_dot4_mmvq_act_cache.writer_stream = nullptr;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_act_cache_make_key(
        const ggml_tensor * src1,
        const int64_t ne0_padded,
        ggml_cuda_mtp_q8_dot4_mmvq_act_cache_key * out) {
    if (!src1 || !out || src1->data == nullptr || ne0_padded <= 0) {
        return false;
    }

    *out = {};
    out->data = src1->data;
    out->device = ggml_cuda_get_device();
    out->type = src1->type;
    out->ne0_padded = ne0_padded;
    for (int i = 0; i < 4; ++i) {
        out->ne[i] = src1->ne[i];
        out->nb[i] = src1->nb[i];
    }
    return true;
}

static inline bool ggml_cuda_mtp_q8_dot4_mmvq_act_cache_try_get(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const bool has_ids,
        const int64_t ne0_padded,
        const size_t bytes,
        cudaStream_t stream,
        char ** out,
        bool * needs_quantize) {
#if defined(GGML_USE_HIP)
    if (out) {
        *out = nullptr;
    }
    if (needs_quantize) {
        *needs_quantize = false;
    }
    const bool q8_dot4_qkv_reuse =
        ggml_cuda_mtp_q8_dot4_mmvq_qkv_reuse_act_enabled() &&
        ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(src0);
    const bool generic_reuse = ggml_cuda_mtp_mmvq_reuse_act_enabled(src0, has_ids);
    const bool log_reuse = ggml_cuda_mtp_q8_dot4_mmvq_log_enabled() || ggml_cuda_mtp_mmvq_reuse_act_log_enabled();
    if (!out || !needs_quantize || has_ids || bytes == 0 || (!q8_dot4_qkv_reuse && !generic_reuse)) {
        return false;
    }

    ggml_cuda_mtp_q8_dot4_mmvq_act_cache_key key = {};
    if (!ggml_cuda_mtp_q8_dot4_mmvq_act_cache_make_key(src1, ne0_padded, &key)) {
        return false;
    }

    ggml_cuda_mtp_q8_dot4_mmvq_act_cache_state & cache = g_ggml_cuda_mtp_q8_dot4_mmvq_act_cache;
    const bool key_match = cache.has_key && cache.key == key;
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    // Do not capture the TLS cache pointer into hipGraph: graph update tracking does
    // not know about this internal allocation, so a later grow/free could leave an
    // old graph exec with a stale pointer. Fall back to the existing per-node pool
    // quantization path while capturing.
    if (stream_is_capturing) {
        if (log_reuse) {
            GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_act_reuse tensor=%s status=reject reject=capture_cache_pointer_unstable bytes=%zu\n",
                    __func__, src0 ? src0->name : "-", bytes);
        }
        return false;
    }
    if (!key_match) {
        cache.valid = false;
        cache.writer_stream = nullptr;
        cache.has_key = true;
        cache.key = key;
    }

    if (cache.valid && cache.writer_stream != stream) {
        if (log_reuse) {
            GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_act_reuse tensor=%s status=reject reject=cross_stream_unordered bytes=%zu\n",
                    __func__, src0 ? src0->name : "-", bytes);
        }
        return false;
    }

    if (cache.bytes < bytes) {
        if (cache.data) {
            CUDA_CHECK(cudaFree(cache.data));
            cache.data = nullptr;
            cache.bytes = 0;
        }
        CUDA_CHECK(cudaMalloc((void **) &cache.data, bytes));
        cache.bytes = bytes;
        cache.valid = false;
    }

    if (!cache.data) {
        return false;
    }

    *out = cache.data;
    *needs_quantize = !cache.valid;
    if (*needs_quantize) {
        cache.writer_stream = stream;
    }
    if (log_reuse) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_act_reuse tensor=%s status=%s mode=%s capture=%d bytes=%zu\n",
                __func__, src0 ? src0->name : "-", cache.valid ? "reuse" : "quantize",
                generic_reuse ? "generic" : "q8_dot4_qkv", stream_is_capturing ? 1 : 0, bytes);
    }
    return true;
#else
    GGML_UNUSED_VARS(src0, src1, has_ids, ne0_padded, bytes, stream, out, needs_quantize);
    return false;
#endif
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_act_cache_mark_valid(const char * tensor_name) {
#if defined(GGML_USE_HIP)
    g_ggml_cuda_mtp_q8_dot4_mmvq_act_cache.valid = true;
    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled() || ggml_cuda_mtp_mmvq_reuse_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_act_reuse tensor=%s status=ready\n",
                __func__, tensor_name ? tensor_name : "-");
    }
#else
    GGML_UNUSED(tensor_name);
#endif
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_log_selected(
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const bool has_fusion, const bool has_ids) {
    if (!g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope || !ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        return;
    }
    GGML_LOG_INFO("%s: mtp_weight_route route=rocm_q8_dot4_mmvq tensor=%s status=selected "
            "ncols_x=%lld nrows_x=%lld ncols_dst=%lld fusion=%d ids=%d\n",
            __func__, g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ? g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-",
            (long long) ncols_x, (long long) nrows_x, (long long) ncols_dst,
            has_fusion ? 1 : 0, has_ids ? 1 : 0);
}

static inline void ggml_cuda_mtp_q8_dot4_mmvq_log_tensor(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const bool has_fusion, const bool has_ids) {
    if (!ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        return;
    }
    GGML_LOG_INFO("%s: mtp_weight_route route=rocm_q8_dot4_mmvq tensor=%s status=%s "
            "ncols_x=%lld nrows_x=%lld ncols_dst=%lld src0=%s src1=%s dst=%s fusion=%d ids=%d\n",
            __func__, src0 ? src0->name : "-",
            (ncols_dst >= 1 && ncols_dst <= dp16_mmvq_packed16_runtime_max_n() && (ncols_x % 256) == 0 && !has_ids) ? "candidate" : "not_applicable",
            (long long) ncols_x, (long long) nrows_x, (long long) ncols_dst,
            src0 ? ggml_type_name(src0->type) : "-",
            src1 ? ggml_type_name(src1->type) : "-",
            dst ? ggml_type_name(dst->type) : "-",
            has_fusion ? 1 : 0, has_ids ? 1 : 0);
}

static inline bool ggml_cuda_dp16_mmvq_q8_dot4_enabled() {
    const char * enabled = getenv("GGML_CUDA_ROCM_Q8_DOT4_MMVQ");
    return ggml_cuda_dp16_route_require_q8_mmvq() ||
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope ||
        (enabled && atoi(enabled) != 0);
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
        !has_ids &&
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
        ncols_dst >= 1 && ncols_dst <= dp16_mmvq_packed16_runtime_max_n() &&
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

#define GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(LAUNCH_EXPR) \
    switch (ncols_dst) { \
        case 1:  { constexpr int c_ncols_dst = 1;  LAUNCH_EXPR; return true; } \
        case 2:  { constexpr int c_ncols_dst = 2;  LAUNCH_EXPR; return true; } \
        case 3:  { constexpr int c_ncols_dst = 3;  LAUNCH_EXPR; return true; } \
        case 4:  { constexpr int c_ncols_dst = 4;  LAUNCH_EXPR; return true; } \
        case 5:  { constexpr int c_ncols_dst = 5;  LAUNCH_EXPR; return true; } \
        case 6:  { constexpr int c_ncols_dst = 6;  LAUNCH_EXPR; return true; } \
        case 7:  { constexpr int c_ncols_dst = 7;  LAUNCH_EXPR; return true; } \
        case 8:  { constexpr int c_ncols_dst = 8;  LAUNCH_EXPR; return true; } \
        case 9:  { constexpr int c_ncols_dst = 9;  LAUNCH_EXPR; return true; } \
        case 10: { constexpr int c_ncols_dst = 10; LAUNCH_EXPR; return true; } \
        case 11: { constexpr int c_ncols_dst = 11; LAUNCH_EXPR; return true; } \
        case 12: { constexpr int c_ncols_dst = 12; LAUNCH_EXPR; return true; } \
        case 13: { constexpr int c_ncols_dst = 13; LAUNCH_EXPR; return true; } \
        case 14: { constexpr int c_ncols_dst = 14; LAUNCH_EXPR; return true; } \
        case 15: { constexpr int c_ncols_dst = 15; LAUNCH_EXPR; return true; } \
        case 16: { constexpr int c_ncols_dst = 16; LAUNCH_EXPR; return true; } \
        default: return false; \
    }

#define GGML_CUDA_DP16_DISPATCH_NCOLS_1_8(...) \
    switch (ncols_dst) { \
        case 1:  { constexpr int c_ncols_dst = 1;  __VA_ARGS__; return true; } \
        case 2:  { constexpr int c_ncols_dst = 2;  __VA_ARGS__; return true; } \
        case 3:  { constexpr int c_ncols_dst = 3;  __VA_ARGS__; return true; } \
        case 4:  { constexpr int c_ncols_dst = 4;  __VA_ARGS__; return true; } \
        case 5:  { constexpr int c_ncols_dst = 5;  __VA_ARGS__; return true; } \
        case 6:  { constexpr int c_ncols_dst = 6;  __VA_ARGS__; return true; } \
        case 7:  { constexpr int c_ncols_dst = 7;  __VA_ARGS__; return true; } \
        case 8:  { constexpr int c_ncols_dst = 8;  __VA_ARGS__; return true; } \
        default: return false; \
    }

#define GGML_CUDA_DP16_DISPATCH_NCOLS_2_5(...) \
    switch (ncols_dst) { \
        case 2:  { constexpr int c_ncols_dst = 2;  __VA_ARGS__; return true; } \
        case 3:  { constexpr int c_ncols_dst = 3;  __VA_ARGS__; return true; } \
        case 4:  { constexpr int c_ncols_dst = 4;  __VA_ARGS__; return true; } \
        case 5:  { constexpr int c_ncols_dst = 5;  __VA_ARGS__; return true; } \
        default: return false; \
    }

template<int NCOLS_DST>
static inline void ggml_cuda_dp16_launch_packed16_dot4(
        const dp16_packed16_weight_view & w16,
        const void * src1_q8_1,
        float * dst,
        const int ncols_x,
        const int nrows_x,
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
        cudaStream_t stream,
        const bool use_i32_lane) {
    if (use_i32_lane) {
        dp16_mmvq_packed16_i32_n1_16_k256_launch<NCOLS_DST>(w16, src1_q8_1, dst,
                ncols_x, nrows_x, channel_ratio, sample_ratio,
                stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_y, stride_sample_dst, stream);
    } else {
        dp16_mmvq_packed16_i32_b32_n1_16_k256_launch<NCOLS_DST>(w16, src1_q8_1, dst,
                ncols_x, nrows_x, channel_ratio, sample_ratio,
                stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_y, stride_sample_dst, stream);
    }
}

template<int NCOLS_DST>
static inline void ggml_cuda_dp16_launch_packed16_fusion_dot4(
        const dp16_packed16_weight_view & w16,
        const void * src1_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
        float * dst,
        const int ncols_x,
        const int nrows_x,
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
        cudaStream_t stream) {
    dp16_mmvq_packed16_i32_b32_fusion_n1_16_k256_launch<NCOLS_DST>(w16, src1_q8_1, fusion_dev, dst,
            ncols_x, nrows_x, channel_ratio, sample_ratio,
            stride_col_y, stride_col_dst, nchannels_dst,
            stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_y, stride_sample_dst, stream);
}

template<int NCOLS_DST, bool HAS_XBIAS>
static inline void ggml_cuda_dp16_launch_packed16_lds_m2n_dot4(
        const dp16_packed16_weight_view & w16,
        const void * src1_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
        float * dst,
        const int ncols_x,
        const int nrows_x,
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
        cudaStream_t stream) {
    dp16_mmvq_packed16_i32_b32_lds_m2n_k256_launch<NCOLS_DST, HAS_XBIAS>(w16, src1_q8_1, fusion_dev, dst,
            ncols_x, nrows_x, channel_ratio, sample_ratio,
            stride_col_y, stride_col_dst, nchannels_dst,
            stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_y, stride_sample_dst, stream);
}

template<int NCOLS_DST, bool HAS_XBIAS>
static inline void ggml_cuda_dp16_launch_packed16_reuse_n_dot4(
        const dp16_packed16_weight_view & w16,
        const void * src1_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
        float * dst,
        const int ncols_x,
        const int nrows_x,
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
        cudaStream_t stream) {
    dp16_mmvq_packed16_i32_b32_reuse_n_k256_launch<NCOLS_DST, HAS_XBIAS>(w16, src1_q8_1, fusion_dev, dst,
            ncols_x, nrows_x, channel_ratio, sample_ratio,
            stride_col_y, stride_col_dst, nchannels_dst,
            stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_y, stride_sample_dst, stream);
}

template<int NCOLS_DST>
static inline void ggml_cuda_dp16_launch_packed16_fused_glu_dot4(
        const dp16_packed16_weight_view & up16,
        const dp16_packed16_weight_view & gate16,
        const void * src1_q8_1,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
        float * dst,
        const int ncols_x,
        const int nrows_x,
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
        cudaStream_t stream) {
    dp16_mmvq_packed16_i32_b32_fused_glu_n1_16_k256_launch<NCOLS_DST>(up16, gate16, src1_q8_1, fusion_dev, dst,
            ncols_x, nrows_x, channel_ratio, sample_ratio,
            stride_col_y, stride_col_dst, nchannels_dst,
            stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_y, stride_sample_dst, stream);
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
        const bool fusion_x_bias_only,
        const bool has_ids) {
    const bool trace = dp16_trace_enabled();
    const bool require_mmvq = ggml_cuda_dp16_route_require_any_mmvq();
    if (!trace && !require_mmvq) {
        return;
    }

    const dp16_problem problem = ggml_cuda_dp16_mmvq_problem_init(
            src0, src1, dst, ncols_x, nrows_x, ncols_dst, cc, has_fusion, fusion_x_bias_only, has_ids);
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
        if (!runtime_supported && fusion_x_bias_only && !has_ids &&
                (ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(src0) ||
                 ggml_cuda_mtp_q8_dot4_mmvq_packed_down_tensor_allowed(src0) ||
                 ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_tensor_allowed(src0))) {
            // Generic packed16 dispatch has no fusion arguments, but the
            // MTP-specialized packed routes below do support x_bias-only fusion.
            runtime_supported = true;
        }
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
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, DP16_ROUTE_MMVQ_PACKED16_DOT4, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    if (src0->type == GGML_TYPE_Q8_0 && dp16_mmvq_packed16_generic_reuse_n_enabled() &&
            dp16_mmvq_packed16_reuse_n_supported_n(ncols_dst)) {
        const ggml_cuda_mm_fusion_args_device no_fusion = {};
        GGML_CUDA_DP16_DISPATCH_NCOLS_1_8(
                ggml_cuda_dp16_launch_packed16_reuse_n_dot4<c_ncols_dst, false>(w16, src1_q8_1, no_fusion, dst,
                    ncols_x, nrows_x, channel_ratio, sample_ratio,
                    stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_y, stride_sample_dst, stream)
        )
    }

    const bool use_i32_lane = dp16_mmvq_packed16_use_i32_lane_kernel();
    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            ggml_cuda_dp16_launch_packed16_dot4<c_ncols_dst>(w16, src1_q8_1, dst,
                ncols_x, nrows_x, channel_ratio, sample_ratio,
                stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_y, stride_sample_dst, stream, use_i32_lane)
    )
}

static inline bool ggml_cuda_dp16_try_launch_mtp_packed_q8_eh_proj(
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
#if defined(GGML_USE_HIP)
    constexpr const char * route = "rocm_mtp_i8_eh_proj_dot4";
    if (!ggml_cuda_mtp_q8_dot4_mmvq_packed_eh_tensor_allowed(src0)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "tensor_not_allowed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_ids) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "ids_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || warp_size != DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "arch_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (ncols_dst < 1 || ncols_dst > dp16_mmvq_packed16_runtime_max_n() || ncols_x % 256 != 0 || nrows_x <= 0) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "shape_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    dp16_packed16_weight_view w16 = {};
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (stream_is_capturing) {
        if (!ggml_cuda_dp16_get_packed16_weight_if_ready(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(__func__, route, src0, "capture_packed_weight_unverified", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    } else {
        if (!ggml_cuda_dp16_ensure_packed16_weight(src0, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    }
    if (w16.cols != ncols_x || w16.rows < nrows_x) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=rocm_mtp_i8_eh_proj_dot4 tensor=%s status=selected capture=%d ncols_x=%d nrows_x=%d ncols_dst=%d packed_payload_bytes=%zu packed_scale_bytes=%zu\n",
                __func__, src0->name, stream_is_capturing ? 1 : 0, ncols_x, nrows_x, ncols_dst,
                (size_t) (w16.payload_stride_sample_i32 ? w16.payload_stride_sample_i32 : w16.rows*w16.payload_stride_row_i32) * sizeof(int32_t),
                (size_t) (w16.scale_stride_sample_half ? w16.scale_stride_sample_half : w16.rows*w16.scale_stride_row_half) * sizeof(half));
    }

    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            ggml_cuda_dp16_launch_packed16_dot4<c_ncols_dst>(w16, src1_q8_1, dst,
                ncols_x, nrows_x, channel_ratio, sample_ratio,
                stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_y, stride_sample_dst, stream, false)
    )
#else
    GGML_UNUSED_VARS(src0, src1_q8_1, dst, ncols_x, nrows_x, ncols_dst,
            channel_ratio, sample_ratio, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_y,
            stride_channel_dst, nsamples_dst, stride_sample_y, stride_sample_dst, cc, warp_size,
            has_fusion, has_ids, stream);
    return false;
#endif
}

static inline bool ggml_cuda_dp16_try_launch_mtp_packed_q8_ffn_down(
        const ggml_tensor * src0,
        const ggml_cuda_mm_fusion_args_host * fusion_host,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
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
#if defined(GGML_USE_HIP)
    constexpr const char * route = "rocm_mtp_i8_ffn_down_dot4";
    if (has_ids) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "ids_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!ggml_cuda_mtp_q8_dot4_mmvq_packed_down_tensor_allowed(src0)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "tensor_not_allowed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && !fusion_host) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_host_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && (fusion_host->gate || fusion_host->gate_bias)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_gate_or_gate_bias", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || warp_size != DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "arch_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (ncols_dst < 1 || ncols_dst > dp16_mmvq_packed16_runtime_max_n() || ncols_x % 256 != 0 || nrows_x <= 0) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "shape_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    dp16_packed16_weight_view w16 = {};
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (stream_is_capturing) {
        if (!ggml_cuda_dp16_get_packed16_weight_if_ready(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(__func__, route, src0, "capture_packed_weight_unverified", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    } else {
        if (!ggml_cuda_dp16_ensure_packed16_weight(src0, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    }
    if (w16.cols != ncols_x || w16.rows < nrows_x) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    // Across-N MMVQ prototypes (no-LDS acc[N] reuse-N and LDS M2xN) are
    // retained as historical force experiments only; real-MTP profiling made
    // both production no-go. Do not route them for MTP ffn_down/attn_out.
    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=selected capture=%d ncols_x=%d nrows_x=%d ncols_dst=%d fusion=%d reuse_n=0 lds_m2n=0 rows_per_block=0 qblocks_tile=0 lds_bytes=0 packed_payload_bytes=%zu packed_scale_bytes=%zu\n",
                __func__, route, src0->name, stream_is_capturing ? 1 : 0, ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0,
                (size_t) (w16.payload_stride_sample_i32 ? w16.payload_stride_sample_i32 : w16.rows*w16.payload_stride_row_i32) * sizeof(int32_t),
                (size_t) (w16.scale_stride_sample_half ? w16.scale_stride_sample_half : w16.rows*w16.scale_stride_row_half) * sizeof(half));
    }

    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            if (has_fusion) {
                ggml_cuda_dp16_launch_packed16_fusion_dot4<c_ncols_dst>(w16, src1_q8_1, fusion_dev, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                ggml_cuda_dp16_launch_packed16_dot4<c_ncols_dst>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream, false);
            }
    )
#else
    GGML_UNUSED_VARS(src0, fusion_host, fusion_dev, src1_q8_1, dst, ncols_x, nrows_x, ncols_dst,
            channel_ratio, sample_ratio, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_y,
            stride_channel_dst, nsamples_dst, stride_sample_y, stride_sample_dst, cc, warp_size,
            has_fusion, has_ids, stream);
    return false;
#endif
}

static inline bool ggml_cuda_dp16_try_launch_mtp_packed_q8_qkv_proj(
        const ggml_tensor * src0,
        const ggml_cuda_mm_fusion_args_host * fusion_host,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
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
#if defined(GGML_USE_HIP)
    constexpr const char * route = "rocm_mtp_i8_qkv_proj_dot4";
    if (has_ids) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "ids_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(src0)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "tensor_not_allowed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && !fusion_host) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_host_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && (fusion_host->gate || fusion_host->gate_bias)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_gate_or_gate_bias", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || warp_size != DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "arch_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (ncols_dst < 1 || ncols_dst > dp16_mmvq_packed16_runtime_max_n() || ncols_x % 256 != 0 || nrows_x <= 0) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "shape_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    dp16_packed16_weight_view w16 = {};
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (stream_is_capturing) {
        if (!ggml_cuda_dp16_get_packed16_weight_if_ready(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(__func__, route, src0, "capture_packed_weight_unverified", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    } else {
        if (!ggml_cuda_dp16_ensure_packed16_weight(src0, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    }
    if (w16.cols != ncols_x || w16.rows < nrows_x) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=rocm_mtp_i8_qkv_proj_dot4 tensor=%s status=selected capture=%d ncols_x=%d nrows_x=%d ncols_dst=%d fusion=%d packed_payload_bytes=%zu packed_scale_bytes=%zu\n",
                __func__, src0->name, stream_is_capturing ? 1 : 0, ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0,
                (size_t) (w16.payload_stride_sample_i32 ? w16.payload_stride_sample_i32 : w16.rows*w16.payload_stride_row_i32) * sizeof(int32_t),
                (size_t) (w16.scale_stride_sample_half ? w16.scale_stride_sample_half : w16.rows*w16.scale_stride_row_half) * sizeof(half));
    }

    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            if (has_fusion) {
                ggml_cuda_dp16_launch_packed16_fusion_dot4<c_ncols_dst>(w16, src1_q8_1, fusion_dev, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                ggml_cuda_dp16_launch_packed16_dot4<c_ncols_dst>(w16, src1_q8_1, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream, false);
            }
    )
#else
    GGML_UNUSED_VARS(src0, fusion_host, fusion_dev, src1_q8_1, dst, ncols_x, nrows_x, ncols_dst,
            channel_ratio, sample_ratio, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_y,
            stride_channel_dst, nsamples_dst, stride_sample_y, stride_sample_dst, cc, warp_size,
            has_fusion, has_ids, stream);
    return false;
#endif
}

static inline bool ggml_cuda_dp16_try_launch_mtp_packed_q8_attn_out(
        const ggml_tensor * src0,
        const ggml_cuda_mm_fusion_args_host * fusion_host,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
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
#if defined(GGML_USE_HIP)
    constexpr const char * route = "rocm_mtp_i8_attn_out_dot4";
    if (has_ids) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "ids_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_tensor_allowed(src0)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "tensor_not_allowed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && !fusion_host) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_host_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_fusion && (fusion_host->gate || fusion_host->gate_bias)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_gate_or_gate_bias", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || warp_size != DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "arch_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (ncols_dst < 1 || ncols_dst > dp16_mmvq_packed16_runtime_max_n() || ncols_x % 256 != 0 || nrows_x <= 0) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "shape_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    dp16_packed16_weight_view w16 = {};
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (stream_is_capturing) {
        if (!ggml_cuda_dp16_get_packed16_weight_if_ready(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(__func__, route, src0, "capture_packed_weight_unverified", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    } else {
        if (!ggml_cuda_dp16_ensure_packed16_weight(src0, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_get_packed16_weight(src0, &w16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    }
    if (w16.cols != ncols_x || w16.rows < nrows_x) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    // Across-N MMVQ prototypes (no-LDS acc[N] reuse-N and LDS M2xN) are
    // retained as historical force experiments only; real-MTP profiling made
    // both production no-go. Do not route them for MTP ffn_down/attn_out.
    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=selected capture=%d ncols_x=%d nrows_x=%d ncols_dst=%d fusion=%d reuse_n=0 lds_m2n=0 rows_per_block=0 qblocks_tile=0 lds_bytes=0 packed_payload_bytes=%zu packed_scale_bytes=%zu\n",
                __func__, route, src0->name, stream_is_capturing ? 1 : 0, ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0,
                (size_t) (w16.payload_stride_sample_i32 ? w16.payload_stride_sample_i32 : w16.rows*w16.payload_stride_row_i32) * sizeof(int32_t),
                (size_t) (w16.scale_stride_sample_half ? w16.scale_stride_sample_half : w16.rows*w16.scale_stride_row_half) * sizeof(half));
    }

    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            if (has_fusion) {
                ggml_cuda_dp16_launch_packed16_fusion_dot4<c_ncols_dst>(w16, src1_q8_1, fusion_dev, dst,
                        ncols_x, nrows_x, channel_ratio, sample_ratio,
                        stride_col_y, stride_col_dst, nchannels_dst,
                        stride_channel_y, stride_channel_dst,
                        nsamples_dst, stride_sample_y, stride_sample_dst, stream);
            } else {
                ggml_cuda_dp16_launch_packed16_dot4<c_ncols_dst>(w16, src1_q8_1, dst,
                ncols_x, nrows_x, channel_ratio, sample_ratio,
                stride_col_y, stride_col_dst, nchannels_dst,
                stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_y, stride_sample_dst, stream, false);
            }
    )
#else
    GGML_UNUSED_VARS(src0, fusion_host, fusion_dev, src1_q8_1, dst, ncols_x, nrows_x, ncols_dst,
            channel_ratio, sample_ratio, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_y,
            stride_channel_dst, nsamples_dst, stride_sample_y, stride_sample_dst, cc, warp_size,
            has_fusion, has_ids, stream);
    return false;
#endif
}

static inline bool ggml_cuda_dp16_try_launch_mtp_packed_q8_fused_glu(
        const ggml_tensor * src0,
        const ggml_cuda_mm_fusion_args_host * fusion_host,
        const ggml_cuda_mm_fusion_args_device fusion_dev,
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
#if defined(GGML_USE_HIP)
    constexpr const char * route = "rocm_mtp_i8_ffn_gate_up_dot4";
    if (!has_fusion) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "no_fusion", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (has_ids) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "ids_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!fusion_host) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_host_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!fusion_host->gate) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "fusion_gate_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!ggml_cuda_mtp_q8_dot4_mmvq_packed_glu_tensors_allowed(src0, fusion_host->gate)) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "tensor_not_allowed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || warp_size != DP16_MMVQ_PACKED16_DOT4_WARP_SIZE) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "arch_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }
    if (ncols_dst < 1 || ncols_dst > dp16_mmvq_packed16_runtime_max_n() || ncols_x % 256 != 0 || nrows_x <= 0) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "shape_unsupported", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    dp16_packed16_weight_view up16 = {};
    dp16_packed16_weight_view gate16 = {};
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (stream_is_capturing) {
        if (!ggml_cuda_dp16_get_packed16_weight_if_ready(src0, &up16) ||
                !ggml_cuda_dp16_get_packed16_weight_if_ready(fusion_host->gate, &gate16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject_capture(__func__, route, src0, "capture_packed_weight_unverified", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    } else {
        if (!ggml_cuda_dp16_ensure_packed16_weight(src0, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_ensure_packed16_weight(fusion_host->gate, stream)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "gate_packed_weight_prepare_failed", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
        if (!ggml_cuda_dp16_get_packed16_weight(src0, &up16) ||
                !ggml_cuda_dp16_get_packed16_weight(fusion_host->gate, &gate16)) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_missing", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
            return false;
        }
    }
    if (up16.cols != ncols_x || gate16.cols != ncols_x || up16.rows < nrows_x || gate16.rows < nrows_x) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_route_reject(__func__, route, src0, "packed_weight_shape_mismatch", ncols_x, nrows_x, ncols_dst, has_fusion, has_ids);
        return false;
    }

    if (ggml_cuda_mtp_q8_dot4_mmvq_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=rocm_mtp_i8_ffn_gate_up_dot4 tensor=%s gate=%s status=selected capture=%d ncols_x=%d nrows_x=%d ncols_dst=%d packed_payload_bytes=%zu packed_scale_bytes=%zu\n",
                __func__, src0->name, fusion_host->gate->name, stream_is_capturing ? 1 : 0, ncols_x, nrows_x, ncols_dst,
                (size_t) (up16.payload_stride_sample_i32 ? up16.payload_stride_sample_i32 : up16.rows*up16.payload_stride_row_i32) * sizeof(int32_t),
                (size_t) (up16.scale_stride_sample_half ? up16.scale_stride_sample_half : up16.rows*up16.scale_stride_row_half) * sizeof(half));
    }

    GGML_CUDA_DP16_DISPATCH_NCOLS_1_16(
            ggml_cuda_dp16_launch_packed16_fused_glu_dot4<c_ncols_dst>(up16, gate16, src1_q8_1, fusion_dev, dst,
                    ncols_x, nrows_x, channel_ratio, sample_ratio,
                    stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_y, stride_channel_dst,
                    nsamples_dst, stride_sample_y, stride_sample_dst, stream)
    )
#else
    GGML_UNUSED_VARS(src0, fusion_host, fusion_dev, src1_q8_1, dst, ncols_x, nrows_x, ncols_dst,
            channel_ratio, sample_ratio, stride_col_y, stride_col_dst, nchannels_dst, stride_channel_y,
            stride_channel_dst, nsamples_dst, stride_sample_y, stride_sample_dst, cc, warp_size,
            has_fusion, has_ids, stream);
    return false;
#endif
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

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1_interleaved_mmvq(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t i00 = i0;
    const int64_t i01 = i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int64_t q8_block = i0 / QK8_1;
    const int64_t iqs      = i0 % QK8_1;
    const int64_t nblocks0 = ne0 / QK8_1;
    const int64_t ib       = ((i3*ne2.z + i2) * nblocks0 + q8_block) * ne1 + i1;

    block_q8_1 * y = (block_q8_1 *) vy;
    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}

static void quantize_row_q8_1_interleaved_mmvq_cuda(
        const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne0 % QK8_1 == 0);
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);
    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_interleaved_mmvq<<<num_blocks, block_size, 0, stream>>>(
            x, vy, ne00, s01, s02, s03, ne0, (uint32_t) ne1, ne2_fastdiv);
}

// Lab-only Q6_K multi-column MMVQ variant. The stock generic kernel calls
// vec_dot_q6_K_q8_1 once per destination column, which reloads the same Q6_K
// weight block for every verifier row. This version keeps the same per-column
// k-order/reduction shape for the common no-fusion/no-ids path, but decodes the
// Q6_K weight lane once and reuses it across ncols_dst Q8_1 activation rows.
template <int ncols_dst, int nwarps, int rows_per_block, bool interleaved_act = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_q6_K_reuse_weight(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t nchannels_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int tid = warp_size*threadIdx.y + threadIdx.x;

    const uint32_t row0        = rows_per_block*blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset_base = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp[ncols_dst][rows_per_block] = {{ 0.0f }};

    for (int kbx = tid / QI6_K; kbx < blocks_per_row_x; kbx += nwarps) {
        const int iqs = tid % QI6_K;
        const int kby = kbx * (qk/QK8_1);

        const int bq8_offset  = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
        const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
        const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

        int vl[rows_per_block];
        int vh[rows_per_block];
        const int8_t * scales[rows_per_block];
        float d6[rows_per_block];
        bool valid_row[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            valid_row[r] = rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst;
            const block_q6_K * bq6_K = (const block_q6_K *) vx + kbx_offset_base + r*stride_row_x + kbx;
            if (valid_row[r]) {
                vl[r]     = get_int_b2(bq6_K->ql, iqs);
                vh[r]     = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
                scales[r] = bq6_K->scales + scale_offset;
                d6[r]     = __half2float(bq6_K->d);
            } else {
                vl[r]     = 0;
                vh[r]     = 0;
                scales[r] = nullptr;
                d6[r]     = 0.0f;
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            int  u[QR6_K];
            half d8[QR6_K];
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const int q8i = bq8_offset + 2*i;
                const block_q8_1 * bq8 = interleaved_act ?
                    (y + (kby + q8i)*stride_col_y + j) :
                    (y + j*stride_col_y + kby + q8i);
                u[i]  = get_int_b4(bq8->qs, iqs % QI8_1);
                d8[i] = ((const half *)&bq8->ds)[0];
            }
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (valid_row[r]) {
                    tmp[j][r] += vec_dot_q6_K_q8_1_impl_mmvq(vl[r], vh[r], u, scales[r], d6[r], d8);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmp_shared[threadIdx.y-1][j][r][threadIdx.x] = tmp[j][r];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
    } else {
        (void) tmp_shared;
    }

    float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            if constexpr (nwarps > 1) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
                }
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
        if (threadIdx.x < rows_per_block && (rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            dst_row[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }

    GGML_UNUSED_VARS(nchannels_dst, nsamples_dst);
}

// Lab-only Q4_K multi-column MMVQ variant using the same interleaved Q8_1
// activation feed as the Q6_K route. This copies the outer schedule/reduction
// shape, but keeps Q4_K's own packed-scale/min decode and dot helper.
template <int ncols_dst, int nwarps, int rows_per_block, bool interleaved_act = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_q4_K_reuse_weight(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t nchannels_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    constexpr int threads_per_kblock = QI4_K / vdr;
    constexpr int blocks_per_iter = vdr * nwarps;

    const int tid = warp_size*threadIdx.y + threadIdx.x;

    const uint32_t row0        = rows_per_block*blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset_base = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp[ncols_dst][rows_per_block] = {{ 0.0f }};

    for (int kbx = tid / threads_per_kblock; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int iqs = vdr * (tid % threads_per_kblock);
        const int kby = kbx * (qk/QK8_1);

        const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
        const int q4_lane    = (iqs/2) % 4;
        const int j_scale    = bq8_offset / 2;

        int v4[rows_per_block][2];
        uint16_t aux[rows_per_block][2];
        half2 dm4[rows_per_block];
        bool valid_row[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            valid_row[r] = rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst;
            const block_q4_K * bq4_K = (const block_q4_K *) vx + kbx_offset_base + r*stride_row_x + kbx;
            if (valid_row[r]) {
                const int * q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*q4_lane);
                v4[r][0] = q4[0];
                v4[r][1] = q4[4];

                const uint16_t * scales = (const uint16_t *) bq4_K->scales;
                if (j_scale < 2) {
                    aux[r][0] = scales[j_scale+0] & 0x3f3f;
                    aux[r][1] = scales[j_scale+2] & 0x3f3f;
                } else {
                    aux[r][0] = ((scales[j_scale+2] >> 0) & 0x0f0f) | ((scales[j_scale-2] & 0xc0c0) >> 2);
                    aux[r][1] = ((scales[j_scale+2] >> 4) & 0x0f0f) | ((scales[j_scale-0] & 0xc0c0) >> 2);
                }
                dm4[r] = bq4_K->dm;
            } else {
                v4[r][0] = 0;
                v4[r][1] = 0;
                aux[r][0] = 0;
                aux[r][1] = 0;
                dm4[r] = make_half2(0.0f, 0.0f);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            int  u[2*QR4_K];
            half d8[QR4_K];
#pragma unroll
            for (int i = 0; i < QR4_K; ++i) {
                const int q8i = bq8_offset + i;
                const block_q8_1 * bq8 = interleaved_act ?
                    (y + (kby + q8i)*stride_col_y + j) :
                    (y + j*stride_col_y + kby + q8i);
                d8[i] = ((const half *)&bq8->ds)[0];

                const int * q8 = (const int *) bq8->qs + q4_lane;
                u[2*i+0] = q8[0];
                u[2*i+1] = q8[4];
            }
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (valid_row[r]) {
                    const uint8_t * sc = (const uint8_t *) aux[r];
                    const uint8_t * m  = sc + 2;
                    tmp[j][r] += vec_dot_q4_K_q8_1_impl_vmmq(v4[r], u, sc, m, dm4[r], d8);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmp_shared[threadIdx.y-1][j][r][threadIdx.x] = tmp[j][r];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
    } else {
        (void) tmp_shared;
    }

    float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            if constexpr (nwarps > 1) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
                }
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
        if (threadIdx.x < rows_per_block && (rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            dst_row[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }

    GGML_UNUSED_VARS(nchannels_dst, nsamples_dst);
}

// Lab-only Q5_K multi-column MMVQ variant using the same interleaved Q8_1
// activation feed as the Q4_K/Q6_K routes. This shares the outer
// schedule/reduction shape but keeps Q5_K's ql/qh + packed-scale/min decode.
template <int ncols_dst, int nwarps, int rows_per_block, bool interleaved_act = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_q5_K_reuse_weight(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t nchannels_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    constexpr int threads_per_kblock = QI5_K / vdr;
    constexpr int blocks_per_iter = vdr * nwarps;

    const int tid = warp_size*threadIdx.y + threadIdx.x;

    const uint32_t row0        = rows_per_block*blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset_base = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp[ncols_dst][rows_per_block] = {{ 0.0f }};

    for (int kbx = tid / threads_per_kblock; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int iqs = vdr * (tid % threads_per_kblock);
        const int kby = kbx * (qk/QK8_1);

        const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
        const int q5_lane    = (iqs/2) % 4;
        const int j_scale    = bq8_offset / 2;

        int vl5[rows_per_block][2];
        int vh5[rows_per_block][2];
        uint16_t aux[rows_per_block][2];
        half2 dm5[rows_per_block];
        bool valid_row[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            valid_row[r] = rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst;
            const block_q5_K * bq5_K = (const block_q5_K *) vx + kbx_offset_base + r*stride_row_x + kbx;
            if (valid_row[r]) {
                const int * ql = (const int *)(bq5_K->qs + 16*bq8_offset + 4*q5_lane);
                const int * qh = (const int *)(bq5_K->qh + 4*q5_lane);
                vl5[r][0] = ql[0];
                vl5[r][1] = ql[4];
                vh5[r][0] = qh[0] >> bq8_offset;
                vh5[r][1] = qh[4] >> bq8_offset;

                const uint16_t * scales = (const uint16_t *) bq5_K->scales;
                if (j_scale < 2) {
                    aux[r][0] = scales[j_scale+0] & 0x3f3f;
                    aux[r][1] = scales[j_scale+2] & 0x3f3f;
                } else {
                    aux[r][0] = ((scales[j_scale+2] >> 0) & 0x0f0f) | ((scales[j_scale-2] & 0xc0c0) >> 2);
                    aux[r][1] = ((scales[j_scale+2] >> 4) & 0x0f0f) | ((scales[j_scale-0] & 0xc0c0) >> 2);
                }
                dm5[r] = bq5_K->dm;
            } else {
                vl5[r][0] = 0;
                vl5[r][1] = 0;
                vh5[r][0] = 0;
                vh5[r][1] = 0;
                aux[r][0] = 0;
                aux[r][1] = 0;
                dm5[r] = make_half2(0.0f, 0.0f);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            int  u[2*QR5_K];
            half d8[QR5_K];
#pragma unroll
            for (int i = 0; i < QR5_K; ++i) {
                const int q8i = bq8_offset + i;
                const block_q8_1 * bq8 = interleaved_act ?
                    (y + (kby + q8i)*stride_col_y + j) :
                    (y + j*stride_col_y + kby + q8i);
                d8[i] = ((const half *)&bq8->ds)[0];

                const int * q8 = (const int *) bq8->qs + q5_lane;
                u[2*i+0] = q8[0];
                u[2*i+1] = q8[4];
            }
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (valid_row[r]) {
                    const uint8_t * sc = (const uint8_t *) aux[r];
                    const uint8_t * m  = sc + 2;
                    tmp[j][r] += vec_dot_q5_K_q8_1_impl_vmmq(vl5[r], vh5[r], u, sc, m, dm5[r], d8);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmp_shared[threadIdx.y-1][j][r][threadIdx.x] = tmp[j][r];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
    } else {
        (void) tmp_shared;
    }

    float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            if constexpr (nwarps > 1) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
                }
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
        if (threadIdx.x < rows_per_block && (rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            dst_row[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }

    GGML_UNUSED_VARS(nchannels_dst, nsamples_dst);
}

// Generic legacy-quant interleaved-activation route for non-K GGUF block types
// where one weight block maps to one Q8_1 activation block. This is default-off
// framework coverage for models with hot Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 tensors.
template <ggml_type type, int ncols_dst, int nwarps, int rows_per_block, bool interleaved_act = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_legacy_interleaved_act(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t nchannels_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    static_assert(qk == QK8_1, "legacy interleaved route expects one Q8_1 block per weight block");

    const int tid = warp_size*threadIdx.y + threadIdx.x;

    const uint32_t row0        = rows_per_block*blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset_base = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    float tmp[ncols_dst][rows_per_block] = {{ 0.0f }};

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi/vdr));
        const int kby = kbx;

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * bq8 = interleaved_act ?
                (y + kby*stride_col_y + j) :
                (y + j*stride_col_y + kby);
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst) {
                    tmp[j][r] += vec_dot_q_cuda(vx, bq8, kbx_offset_base + r*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmp_shared[threadIdx.y-1][j][r][threadIdx.x] = tmp[j][r];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
    } else {
        (void) tmp_shared;
    }

    float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            if constexpr (nwarps > 1) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
                }
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
        if (threadIdx.x < rows_per_block && (rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            dst_row[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }

    GGML_UNUSED_VARS(nchannels_dst, nsamples_dst);
}

// Generic low-bit K-quant interleaved-activation route for Q2_K/Q3_K. These
// block types span multiple Q8_1 activation blocks, so they need explicit
// interleaved gathers rather than the legacy one-block vecdot wrapper.
template <ggml_type type, int ncols_dst, int nwarps, int rows_per_block, bool interleaved_act = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_lowk_interleaved_act(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 channel_ratio, const uint3 sample_ratio,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t nchannels_dst, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint32_t nsamples_dst,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    static_assert(type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K, "lowk interleaved route supports Q2_K/Q3_K only");

    constexpr int qk = QK_K;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int qr = type == GGML_TYPE_Q2_K ? QR2_K : QR3_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    const int tid = warp_size*threadIdx.y + threadIdx.x;

    const uint32_t row0        = rows_per_block*blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;

    const uint32_t channel_x = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset_base = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    const int blocks_per_row_x = ncols_x / qk;

    float tmp[ncols_dst][rows_per_block] = {{ 0.0f }};

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int iqs = vdr * (tid % (qi/vdr));
        const int kby = kbx * (qk/QK8_1);

        bool valid_row[rows_per_block];
        int v2[rows_per_block];
        const uint8_t * scales2[rows_per_block];
        half2 dm2[rows_per_block];
        int vl3[rows_per_block];
        int vh3[rows_per_block];
        const uint8_t * scales3[rows_per_block];
        int scale_offset3[rows_per_block];
        float d3[rows_per_block];

        int bq8_offset = 0;
        if constexpr (type == GGML_TYPE_Q2_K) {
            bq8_offset = QR2_K * (iqs / QI8_1);
            const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                valid_row[r] = rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst;
                const block_q2_K * bq2_K = (const block_q2_K *) vx + kbx_offset_base + r*stride_row_x + kbx;
                if (valid_row[r]) {
                    v2[r]      = get_int_b4(bq2_K->qs, iqs);
                    scales2[r] = bq2_K->scales + scale_offset;
                    dm2[r]     = bq2_K->dm;
                } else {
                    v2[r]      = 0;
                    scales2[r] = nullptr;
                    dm2[r]     = make_half2(0.0f, 0.0f);
                }
            }
        } else {
            bq8_offset = QR3_K * (iqs / (QI3_K/2));
            const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                valid_row[r] = rows_per_block == 1 || uint32_t(row0 + r) < stride_col_dst;
                const block_q3_K * bq3_K = (const block_q3_K *) vx + kbx_offset_base + r*stride_row_x + kbx;
                if (valid_row[r]) {
                    vl3[r]           = get_int_b2(bq3_K->qs, iqs);
                    vh3[r]           = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;
                    scales3[r]       = bq3_K->scales;
                    scale_offset3[r] = scale_offset;
                    d3[r]            = bq3_K->d;
                } else {
                    vl3[r]           = 0;
                    vh3[r]           = 0;
                    scales3[r]       = nullptr;
                    scale_offset3[r] = 0;
                    d3[r]            = 0.0f;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            int  u[qr];
            half d8[qr];
#pragma unroll
            for (int i = 0; i < qr; ++i) {
                const block_q8_1 * bq8 = interleaved_act ?
                    (y + (kby + bq8_offset + i)*stride_col_y + j) :
                    (y + j*stride_col_y + kby + bq8_offset + i);
                u[i]  = get_int_b4(bq8->qs, iqs % QI8_1);
                d8[i] = ((const half *)&bq8->ds)[0];
            }
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (valid_row[r]) {
                    if constexpr (type == GGML_TYPE_Q2_K) {
                        tmp[j][r] += vec_dot_q2_K_q8_1_impl_mmvq(v2[r], u, scales2[r], dm2[r], d8);
                    } else {
                        tmp[j][r] += vec_dot_q3_K_q8_1_impl_mmvq(vl3[r], vh3[r], u, scales3[r], scale_offset3[r], d3[r], d8);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if constexpr (nwarps > 1) {
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int r = 0; r < rows_per_block; ++r) {
                    tmp_shared[threadIdx.y-1][j][r][threadIdx.x] = tmp[j][r];
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
    } else {
        (void) tmp_shared;
    }

    float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            if constexpr (nwarps > 1) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][r] += tmp_shared[l][j][r][threadIdx.x];
                }
            }
            tmp[j][r] = warp_reduce_sum<warp_size>(tmp[j][r]);
        }
        if (threadIdx.x < rows_per_block && (rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            dst_row[j*stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }

    GGML_UNUSED_VARS(nchannels_dst, nsamples_dst);
}

// One-launch version of the exact serial-column repair. Each CUDA/HIP block
// computes exactly one output column using the same ncols_dst=1 reduction shape
// as the existing per-column loop; blockIdx.z is flattened as
// sample_dst*ncols_dst + column. This reduces launch overhead without changing
// per-column arithmetic order.
template <ggml_type type, bool small_k = false>
__launch_bounds__(calc_nwarps(type, 1, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_serial_columns(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t ncols_dst,
        const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, 1, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(1, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t column_dst  = blockIdx.z % ncols_dst;
    const uint32_t sample_dst  = blockIdx.z / ncols_dst;

    const uint32_t channel_x  = ids ? ids[channel_dst + column_dst*ids_stride] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y  = ids ? fastmodulo(channel_dst, nchannels_y)     : channel_dst;
    const uint32_t sample_x   = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y   = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y + column_dst*stride_col_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[rows_per_cuda_block] = { 0.0f };

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][rows_per_cuda_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    float * dst_col = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst +
        column_dst*stride_col_dst + row0;

#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i][threadIdx.x];
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < rows_per_cuda_block &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
        dst_col[threadIdx.x] = tmp[threadIdx.x];
    }
}

// One-launch fused serial-column kernel. This is the Stage4 row-equivalent
// small-N path for verifier FFN/MoE projections with GLU/bias fusion.  It keeps
// the ncols_dst=1 warp/reduction geometry and flattens destination columns into
// blockIdx.z, so every verifier row is computed like an independent serial
// single-token MMVQ call while avoiding N host launches.
template <ggml_type type, bool small_k = false>
__launch_bounds__(calc_nwarps(type, 1, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_serial_columns_fused(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t ncols_dst,
        const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, 1, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(1, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t column_dst  = blockIdx.z % ncols_dst;
    const uint32_t sample_dst  = blockIdx.z / ncols_dst;

    const uint32_t channel_x = ids ? ids[channel_dst + column_dst*ids_stride] : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y = ids ? fastmodulo(channel_dst, nchannels_y)      : channel_dst;
    const uint32_t sample_x  = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y  = sample_dst;

    const bool use_gate      = fusion.gate      != nullptr;
    const bool use_bias      = fusion.x_bias    != nullptr;
    const bool use_gate_bias = fusion.gate_bias != nullptr && use_gate;
    const void  * vgate      = fusion.gate;
    const float * x_bias     = (const float *) fusion.x_bias;
    const float * gate_bias  = (const float *) fusion.gate_bias;
    const ggml_glu_op active_glu = fusion.glu_op;

    float x_biases[rows_per_cuda_block]    = { 0.0f };
    float gate_biases[rows_per_cuda_block] = { 0.0f };

    const uint32_t channel_bias = ids ? channel_x : channel_dst;
    if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
        if (use_bias) {
            const float * xb = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // Serial equivalence: a standalone ncols_dst=1 launch sees the bias
            // column as zero.  Do not add column_dst here.
            x_biases[threadIdx.x] = xb[threadIdx.x];
        }
        if (use_gate_bias) {
            const float * gb = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            gate_biases[threadIdx.x] = gb[threadIdx.x];
        }
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) +
        sample_y*stride_sample_y + channel_y*stride_channel_y + column_dst*stride_col_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[rows_per_cuda_block]      = { 0.0f };
    float tmp_gate[rows_per_cuda_block] = { 0.0f };

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if (use_gate) {
                tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[nwarps-1 > 0 ? nwarps-1 : 1][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
            if (use_gate) {
                tmp_shared_gate[threadIdx.y-1][i][threadIdx.x] = tmp_gate[i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    float * dst_col = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst +
        column_dst*stride_col_dst + row0;

#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i][threadIdx.x];
            if (use_gate) {
                tmp_gate[i] += tmp_shared_gate[l][i][threadIdx.x];
            }
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if (use_gate) {
            tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        }
    }

    if (threadIdx.x < rows_per_cuda_block &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
        float result = tmp[threadIdx.x];
        if (use_bias) {
            result += x_biases[threadIdx.x];
        }
        if (use_gate) {
            float gate_value = tmp_gate[threadIdx.x];
            if (use_gate_bias) {
                gate_value += gate_biases[threadIdx.x];
            }
            switch (active_glu) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                    break;
                default:
                    result *= gate_value;
                    break;
            }
        }
        dst_col[threadIdx.x] = result;
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
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

    float x_biases[c_rows_per_block]    = { 0.0f };
    float gate_biases[c_rows_per_block] = { 0.0f };
    if constexpr (has_fusion) {
        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            if (use_bias) {
                x_bias = x_bias + channel_x*stride_channel_dst + row0;
                x_biases[threadIdx.x] = x_bias[threadIdx.x];
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + channel_x*stride_channel_dst + row0;
                gate_biases[threadIdx.x] = gate_bias[threadIdx.x];
            }
        }
    }

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            if (use_bias) {
                result += x_biases[threadIdx.x];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if (use_gate_bias) {
                    gate_value += gate_biases[threadIdx.x];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    default:
                        result *= gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(fusion, use_gate, use_bias, use_gate_bias, vgate, x_bias, gate_bias, active_glu, x_biases, gate_biases, tmp_gate);
    }
}


// Explicit Q8_0 MoE small-route kernel for RDNA3 dot4 route selection.
// The generic route-direct MoE kernel also specializes Q8_0, but this named kernel gives
// the Q8_0 path an independent opt-in route marker, row-tile policy, and canary surface.
template <int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(MMVQ_MAX_BATCH_SIZE*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q8_0_moe_dot4(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t token_idx = threadIdx.y;
    const int      row0      = c_rows_per_block * blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y * stride_channel_y + token_idx * stride_col_y;
    const int kbx_offset = channel_x * stride_channel_x + row0 * stride_row_x;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu = GGML_GLU_OP_SWIGLU;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    float x_biases[c_rows_per_block]    = { 0.0f };
    float gate_biases[c_rows_per_block] = { 0.0f };
    if constexpr (has_fusion) {
        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            if (use_bias) {
                x_bias = x_bias + channel_x * stride_channel_dst + row0;
                x_biases[threadIdx.x] = x_bias[threadIdx.x];
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + channel_x * stride_channel_dst + row0;
                gate_biases[threadIdx.x] = gate_bias[threadIdx.x];
            }
        }
    }

    float tmp[c_rows_per_block]      = { 0.0f };
    float tmp_gate[c_rows_per_block] = { 0.0f };

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q8_0_q8_1(vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q8_0_q8_1(vgate, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            if (use_bias) {
                result += x_biases[threadIdx.x];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if (use_gate_bias) {
                    gate_value += gate_biases[threadIdx.x];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    default:
                        result *= gate_value;
                        break;
                }
            }
        }
        dst[channel_dst * stride_channel_dst + token_idx * stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(fusion, use_gate, use_bias, use_gate_bias, vgate, x_bias, gate_bias, active_glu, x_biases, gate_biases, tmp_gate);
    }
}

struct ggml_cuda_mmvq_iq3_s_q8_lane_cache {
    int   u0[4];
    int   u1[4];
    float d8;
};

static __device__ __forceinline__ ggml_cuda_mmvq_iq3_s_q8_lane_cache ggml_cuda_mmvq_iq3_s_q8_lane_load(
        const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    ggml_cuda_mmvq_iq3_s_q8_lane_cache c;
    const block_q8_1 * bq8 = &bq8_1[iqs/2];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        c.u0[l0/2] = get_int_b4(bq8->qs, l0 + 0);
        c.u1[l0/2] = get_int_b4(bq8->qs, l0 + 1);
    }
    c.d8 = __low2float(bq8->ds);
    return c;
}

static __device__ __forceinline__ float ggml_cuda_mmvq_dot_iq3_s_q8_cached(
        const void * __restrict__ vx, const int kbx, const int iqs,
        const ggml_cuda_mmvq_iq3_s_q8_lane_cache & q8) {
    const block_iq3_s * bq3 = (const block_iq3_s *) vx + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        sumi = ggml_cuda_dp4a(grid_l, q8.u0[l0/2], sumi);
        sumi = ggml_cuda_dp4a(grid_h, q8.u1[l0/2], sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * q8.d8;
    return d * sumi;
}

static constexpr int GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS = 8;
static constexpr int GGML_CUDA_MMVQ_IQ3S_SIDECAR_PACKS_PER_IQS = 8;

static __global__ void ggml_cuda_mmvq_iq3_s_moe_sidecar_collect_unique(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ unique_ids,
        int32_t * __restrict__ unique_count,
        int32_t * __restrict__ ids_local,
        const uint32_t nchannels_dst, const uint32_t ncols_dst, const uint32_t ids_stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int count = 0;
    for (uint32_t token = 0; token < ncols_dst; ++token) {
        for (uint32_t channel = 0; channel < nchannels_dst; ++channel) {
            const int32_t id = ids[channel + token*ids_stride];
            int local = -1;
            for (int i = 0; i < count; ++i) {
                if (unique_ids[i] == id) {
                    local = i;
                    break;
                }
            }
            if (local < 0) {
                local = count;
                unique_ids[count++] = id;
            }
            ids_local[token*nchannels_dst + channel] = local;
        }
    }
    *unique_count = count;
}

static __global__ void ggml_cuda_mmvq_iq3_s_moe_sidecar_gen(
        const void * __restrict__ vx,
        const int32_t * __restrict__ unique_ids,
        const int32_t * __restrict__ unique_count,
        int32_t * __restrict__ packs,
        uint8_t * __restrict__ scales,
        half * __restrict__ ds,
        const uint32_t max_unique, const uint32_t nrows_x, const uint32_t blocks_per_row_x,
        const uint32_t stride_row_x, const uint32_t stride_channel_x) {
    const uint64_t idx = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t) max_unique*nrows_x*blocks_per_row_x*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    if (idx >= total) {
        return;
    }

    const int group = idx % GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    uint64_t t = idx / GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    const uint32_t kbx = t % blocks_per_row_x;
    t /= blocks_per_row_x;
    const uint32_t row = t % nrows_x;
    const uint32_t unique_idx = t / nrows_x;

    if (unique_idx >= (uint32_t) *unique_count) {
        return;
    }

    const uint32_t expert = (uint32_t) unique_ids[unique_idx];
    const block_iq3_s * bq3 = (const block_iq3_s *) vx + expert*stride_channel_x + row*stride_row_x + kbx;
    const int iqs = 2*group;

    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

    const uint64_t sidecar_block = ((uint64_t) unique_idx*nrows_x + row)*blocks_per_row_x + kbx;
    const uint64_t out_base = (sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group)*GGML_CUDA_MMVQ_IQ3S_SIDECAR_PACKS_PER_IQS;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        packs[out_base + l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
        packs[out_base + l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
    }

    scales[sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group] =
        (uint8_t) (1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F));
    if (group == 0) {
        ds[sidecar_block] = bq3->d;
    }
}


static __global__ void ggml_cuda_mmvq_iq3_s_moe_sidecar_gen_slots(
        const void * __restrict__ vx,
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ packs,
        uint8_t * __restrict__ scales,
        half * __restrict__ ds,
        const uint32_t ncols_dst, const uint32_t nchannels_dst,
        const uint32_t nrows_x, const uint32_t blocks_per_row_x,
        const uint32_t stride_row_x, const uint32_t stride_channel_x,
        const uint32_t ids_stride) {
    const uint64_t idx = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const uint64_t nslots = (uint64_t) ncols_dst*nchannels_dst;
    const uint64_t total = nslots*nrows_x*blocks_per_row_x*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    if (idx >= total) {
        return;
    }

    const int group = idx % GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    uint64_t t = idx / GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    const uint32_t kbx = t % blocks_per_row_x;
    t /= blocks_per_row_x;
    const uint32_t row = t % nrows_x;
    const uint32_t slot = t / nrows_x;
    const uint32_t token = slot / nchannels_dst;
    const uint32_t channel = slot - token*nchannels_dst;

    const uint32_t expert = (uint32_t) ids[channel + token*ids_stride];
    const block_iq3_s * bq3 = (const block_iq3_s *) vx + expert*stride_channel_x + row*stride_row_x + kbx;
    const int iqs = 2*group;

    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

    const uint64_t sidecar_block = ((uint64_t) slot*nrows_x + row)*blocks_per_row_x + kbx;
    const uint64_t out_base = (sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group)*GGML_CUDA_MMVQ_IQ3S_SIDECAR_PACKS_PER_IQS;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        packs[out_base + l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
        packs[out_base + l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
    }

    scales[sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group] =
        (uint8_t) (1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F));
    if (group == 0) {
        ds[sidecar_block] = bq3->d;
    }
}

static __device__ __forceinline__ float ggml_cuda_mmvq_dot_iq3_s_sidecar_q8_cached(
        const int32_t * __restrict__ packs,
        const uint8_t * __restrict__ scales,
        const half * __restrict__ ds,
        const uint64_t sidecar_block, const int iqs,
        const ggml_cuda_mmvq_iq3_s_q8_lane_cache & q8) {
    const int group = iqs / 2;
    const uint64_t base = (sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group)*GGML_CUDA_MMVQ_IQ3S_SIDECAR_PACKS_PER_IQS;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = ggml_cuda_dp4a(packs[base + 2*j + 0], q8.u0[j], sumi);
        sumi = ggml_cuda_dp4a(packs[base + 2*j + 1], q8.u1[j], sumi);
    }
    sumi *= (int) scales[sidecar_block*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS + group];
    const float d = __half2float(ds[sidecar_block]) * q8.d8;
    return d * sumi;
}

// Lab-only IQ3_S MoE gate/up sidecar. The route-local sidecar is dense in
// unique expert order, while ids_local maps every original token/top-k route
// slot back to a local sidecar expert. Output remains in original route-slot
// order, preserving MUL_MAT_ID layout.
template <int c_rows_per_block, int EXP_TILE>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<GGML_TYPE_IQ3_S>()*ggml_cuda_get_physical_warp_size()*EXP_TILE, 1)
static __global__ void mul_mat_vec_q_iq3_s_moe_gateup_sidecar_tile(
        const int32_t * __restrict__ packs,
        const uint8_t * __restrict__ scales,
        const half * __restrict__ ds,
        const void * __restrict__ vy,
        const int32_t * __restrict__ ids_local,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t nchannels_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = 4; // IQ3_S MMVQ: vdr(2) * warp(32) / qi(16)

    const uint32_t token_idx   = threadIdx.y;
    const uint32_t expert_lane = threadIdx.z;
    const uint32_t channel_dst = blockIdx.y * EXP_TILE + expert_lane;
    const bool     active      = token_idx < ncols_dst && channel_dst < nchannels_dst;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    const uint32_t local_expert = active ? (uint32_t) ids_local[token_idx*nchannels_dst + channel_dst] : 0;

    float tmp[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = 0.0f;
    }

    __shared__ int   q8_u0[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ int   q8_u1[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ float q8_d8[MMVQ_MAX_BATCH_SIZE][warp_size];

    const int lane_group = threadIdx.x & 7;
    const int kqs = 2 * lane_group;
    const int kbx_start = threadIdx.x >> 3;
    const int iter_count = (blocks_per_row_x + blocks_per_iter - 1) / blocks_per_iter;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + token_idx*stride_col_y;

    for (int iter = 0; iter < iter_count; ++iter) {
        const int kbx = kbx_start + iter*blocks_per_iter;
        const bool valid_k = kbx < blocks_per_row_x;

        if (expert_lane == 0) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
            if (valid_k && token_idx < ncols_dst) {
                const int kby = kbx * (qk/QK8_1);
                q8 = ggml_cuda_mmvq_iq3_s_q8_lane_load(&y[kby], kqs);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    q8.u0[j] = 0;
                    q8.u1[j] = 0;
                }
                q8.d8 = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8_u0[token_idx][threadIdx.x][j] = q8.u0[j];
                q8_u1[token_idx][threadIdx.x][j] = q8.u1[j];
            }
            q8_d8[token_idx][threadIdx.x] = q8.d8;
        }
        __syncthreads();

        if (active && valid_k) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8.u0[j] = q8_u0[token_idx][threadIdx.x][j];
                q8.u1[j] = q8_u1[token_idx][threadIdx.x][j];
            }
            q8.d8 = q8_d8[token_idx][threadIdx.x];
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                const uint64_t sidecar_block = ((uint64_t) local_expert*nrows_x + (row0 + i))*blocks_per_row_x + kbx;
                tmp[i] += ggml_cuda_mmvq_dot_iq3_s_sidecar_q8_cached(packs, scales, ds, sidecar_block, kqs, q8);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (active && threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}


// Lab-only route-slot sidecar variant. This avoids a separate collect-unique
// launch by generating the sidecar directly in original token/top-k route-slot
// order. Repeated experts are duplicated in the sidecar, but output ordering is
// inherently restored.
template <int c_rows_per_block, int EXP_TILE>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<GGML_TYPE_IQ3_S>()*ggml_cuda_get_physical_warp_size()*EXP_TILE, 1)
static __global__ void mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile(
        const int32_t * __restrict__ packs,
        const uint8_t * __restrict__ scales,
        const half * __restrict__ ds,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t nchannels_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = 4;

    const uint32_t token_idx   = threadIdx.y;
    const uint32_t expert_lane = threadIdx.z;
    const uint32_t channel_dst = blockIdx.y * EXP_TILE + expert_lane;
    const bool     active      = token_idx < ncols_dst && channel_dst < nchannels_dst;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    const uint32_t slot        = token_idx*nchannels_dst + channel_dst;

    float tmp[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = 0.0f;
    }

    __shared__ int   q8_u0[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ int   q8_u1[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ float q8_d8[MMVQ_MAX_BATCH_SIZE][warp_size];

    const int lane_group = threadIdx.x & 7;
    const int kqs = 2 * lane_group;
    const int kbx_start = threadIdx.x >> 3;
    const int iter_count = (blocks_per_row_x + blocks_per_iter - 1) / blocks_per_iter;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + token_idx*stride_col_y;

    for (int iter = 0; iter < iter_count; ++iter) {
        const int kbx = kbx_start + iter*blocks_per_iter;
        const bool valid_k = kbx < blocks_per_row_x;

        if (expert_lane == 0) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
            if (valid_k && token_idx < ncols_dst) {
                const int kby = kbx * (qk/QK8_1);
                q8 = ggml_cuda_mmvq_iq3_s_q8_lane_load(&y[kby], kqs);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    q8.u0[j] = 0;
                    q8.u1[j] = 0;
                }
                q8.d8 = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8_u0[token_idx][threadIdx.x][j] = q8.u0[j];
                q8_u1[token_idx][threadIdx.x][j] = q8.u1[j];
            }
            q8_d8[token_idx][threadIdx.x] = q8.d8;
        }
        __syncthreads();

        if (active && valid_k) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8.u0[j] = q8_u0[token_idx][threadIdx.x][j];
                q8.u1[j] = q8_u1[token_idx][threadIdx.x][j];
            }
            q8.d8 = q8_d8[token_idx][threadIdx.x];
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                const uint64_t sidecar_block = ((uint64_t) slot*nrows_x + (row0 + i))*blocks_per_row_x + kbx;
                tmp[i] += ggml_cuda_mmvq_dot_iq3_s_sidecar_q8_cached(packs, scales, ds, sidecar_block, kqs, q8);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (active && threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Dedicated IQ3_S MoE gate/up expert tile. Gate/up has one q8 activation
// channel shared by all selected experts. This version parallelizes the expert
// tile across threadIdx.z so each expert slot has its own warp while the CTA
// shares one Q8_1 lane cache per token/lane/k iteration.
template <int c_rows_per_block, int EXP_TILE>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<GGML_TYPE_IQ3_S>()*ggml_cuda_get_physical_warp_size()*EXP_TILE, 1)
static __global__ void mul_mat_vec_q_iq3_s_moe_gateup_tile(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_dst) {

    constexpr int qk = QK_K;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = 4; // IQ3_S MMVQ: vdr(2) * warp(32) / qi(16)

    const uint32_t token_idx   = threadIdx.y;
    const uint32_t expert_lane = threadIdx.z;
    const uint32_t channel_dst = blockIdx.y * EXP_TILE + expert_lane;
    const bool     active      = channel_dst < nchannels_dst;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;

    const uint32_t channel_x = active ? (uint32_t) ids[channel_dst + token_idx * ids_stride] : 0;
    const int      kbx_offset = channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = 0.0f;
    }

    __shared__ int   q8_u0[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ int   q8_u1[MMVQ_MAX_BATCH_SIZE][warp_size][4];
    __shared__ float q8_d8[MMVQ_MAX_BATCH_SIZE][warp_size];

    const int lane_group = threadIdx.x & 7;
    const int kqs = 2 * lane_group;
    const int kbx_start = threadIdx.x >> 3;
    const int iter_count = (blocks_per_row_x + blocks_per_iter - 1) / blocks_per_iter;
    const block_q8_1 * y = ((const block_q8_1 *) vy) + token_idx*stride_col_y;

    for (int iter = 0; iter < iter_count; ++iter) {
        const int kbx = kbx_start + iter*blocks_per_iter;
        const bool valid_k = kbx < blocks_per_row_x;

        if (expert_lane == 0) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
            if (valid_k) {
                const int kby = kbx * (qk/QK8_1);
                q8 = ggml_cuda_mmvq_iq3_s_q8_lane_load(&y[kby], kqs);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    q8.u0[j] = 0;
                    q8.u1[j] = 0;
                }
                q8.d8 = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8_u0[token_idx][threadIdx.x][j] = q8.u0[j];
                q8_u1[token_idx][threadIdx.x][j] = q8.u1[j];
            }
            q8_d8[token_idx][threadIdx.x] = q8.d8;
        }
        __syncthreads();

        if (active && valid_k) {
            ggml_cuda_mmvq_iq3_s_q8_lane_cache q8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                q8.u0[j] = q8_u0[token_idx][threadIdx.x][j];
                q8.u1[j] = q8_u1[token_idx][threadIdx.x][j];
            }
            q8.d8 = q8_d8[token_idx][threadIdx.x];
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                tmp[i] += ggml_cuda_mmvq_dot_iq3_s_q8_cached(
                    vx, kbx_offset + i*stride_row_x + kbx, kqs, q8);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (active && threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Lab-only MoE expert-slot tiled variant. This keeps the same per-(token, expert,
// row) accumulation order as mul_mat_vec_q_moe, but each warp computes multiple
// expert slots for the same token and row tile.
template <ggml_type type, int c_rows_per_block, int EXP_TILE>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe_exp_tile(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    if (token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_base = blockIdx.y * EXP_TILE;

    uint32_t channel_dst_e[EXP_TILE];
    uint32_t channel_x_e[EXP_TILE];
    uint32_t channel_y_e[EXP_TILE];
    int      kbx_offset_e[EXP_TILE];
#pragma unroll
    for (int e = 0; e < EXP_TILE; ++e) {
        const uint32_t channel_dst = channel_base + e;
        channel_dst_e[e] = channel_dst;
        if (channel_dst < nchannels_dst) {
            channel_x_e[e]  = ids[channel_dst + token_idx * ids_stride];
            channel_y_e[e]  = fastmodulo(channel_dst, nchannels_y);
            kbx_offset_e[e] = channel_x_e[e]*stride_channel_x + row0*stride_row_x;
        } else {
            channel_x_e[e]  = 0;
            channel_y_e[e]  = 0;
            kbx_offset_e[e] = 0;
        }
    }

    float tmp[EXP_TILE][c_rows_per_block];
#pragma unroll
    for (int e = 0; e < EXP_TILE; ++e) {
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[e][i] = 0.0f;
        }
    }

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int e = 0; e < EXP_TILE; ++e) {
            if (channel_dst_e[e] >= nchannels_dst) {
                continue;
            }
            const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y_e[e]*stride_channel_y + token_idx*stride_col_y;
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                tmp[e][i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset_e[e] + i*stride_row_x + kbx, kqs);
            }
        }
    }

#pragma unroll
    for (int e = 0; e < EXP_TILE; ++e) {
        const uint32_t channel_dst = channel_dst_e[e];
        if (channel_dst >= nchannels_dst) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[e][i] = warp_reduce_sum<warp_size>(tmp[e][i]);
        }

        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[e][threadIdx.x];
        }
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

template <int c_ncols_dst, int c_nwarps, int c_rows_per_block, bool c_interleaved_act = false>
static void mul_mat_vec_q_q6_K_reuse_weight_launch(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int warp_size, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const dim3 block_nums((nrows_x + c_rows_per_block - 1) / c_rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    mul_mat_vec_q_q6_K_reuse_weight<c_ncols_dst, c_nwarps, c_rows_per_block, c_interleaved_act><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols_x, channel_ratio, sample_ratio,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static bool mul_mat_vec_q_q6_K_reuse_weight_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_q6k_reuse_weight_enabled()) {
        return false;
    }
    if (src0 == nullptr || src0->type != GGML_TYPE_Q6_K || !ggml_cuda_mtp_mmvq_q6k_reuse_weight_name_allowed(src0->name)) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK_K != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    if (!ggml_cuda_mtp_mmvq_q6k_reuse_weight_ncols_allowed(ncols_dst)) {
        return false;
    }

    const int nwarps_requested = ggml_cuda_mtp_mmvq_q6k_reuse_weight_nwarps();
    const int rows_requested = ggml_cuda_mtp_mmvq_q6k_reuse_weight_rows();
    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    if (ggml_cuda_mtp_mmvq_q6k_reuse_weight_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_q6k_reuse_weight tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_q6_K_reuse_weight_launch<NCOLS, NWARPS, ROWS>( \
            vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(NCOLS) \
    do { \
        if (rows_requested == 2) { \
            if (nwarps_requested == 4) { \
                GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 4, 2); \
            } else if (nwarps_requested == 2) { \
                GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 2, 2); \
            } else { \
                GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 1, 2); \
            } \
        } else if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 4, 1); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 2, 1); \
        } else { \
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS(NCOLS, 1, 1); \
        } \
    } while (0)

    switch (ncols_dst) {
        case 1:
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(1);
            return true;
        case 2:
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(2);
            return true;
        case 3:
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(3);
            return true;
        case 4:
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(4);
            return true;
        case 5:
            GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH(5);
            return true;
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH
#undef GGML_CUDA_MMVQ_Q6K_REUSE_LAUNCH_NWARPS_ROWS
}

static bool mul_mat_vec_q_q6_K_interleaved_act_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy_interleaved, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y_interleaved, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y_interleaved, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y_interleaved,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_q6k_interleaved_act_enabled()) {
        return false;
    }
    if (src0 == nullptr || src0->type != GGML_TYPE_Q6_K || !ggml_cuda_mtp_mmvq_q6k_interleaved_act_name_allowed(src0->name)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_q6k_interleaved_act_ncols_allowed(ncols_dst) || ncols_dst < 2 || ncols_dst > 5) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK_K != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
    const int rows_requested   = ggml_cuda_mtp_mmvq_interleaved_act_rows("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_ROWS");
    const int nwarps_requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps("LLAMA_MTP_MMVQ_Q6K_INTERLEAVED_ACT_NWARPS");

    if (ggml_cuda_mtp_mmvq_q6k_interleaved_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_q6k_interleaved_act tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_q6_K_reuse_weight_launch<NCOLS, NWARPS, ROWS, true>( \
            vx, vy_interleaved, dst, ncols_x, nrows_x, stride_row_x, stride_col_y_interleaved, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y_interleaved, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y_interleaved, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, ROWS) \
    do { \
        if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 4, ROWS); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 2, ROWS); \
        } else { \
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 1, ROWS); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH(NCOLS) \
    do { \
        if (rows_requested == 2) { \
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 2); \
        } else { \
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 1); \
        } \
    } while (0)

    switch (ncols_dst) {
        case 2:
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH(2);
            return true;
        case 3:
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH(3);
            return true;
        case 4:
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH(4);
            return true;
        case 5:
            GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH(5);
            return true;
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH
#undef GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_ROWS
#undef GGML_CUDA_MMVQ_Q6K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS
}

template <int c_ncols_dst, int c_nwarps, int c_rows_per_block, bool c_interleaved_act = false>
static void mul_mat_vec_q_q4_K_reuse_weight_launch(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int warp_size, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const dim3 block_nums((nrows_x + c_rows_per_block - 1) / c_rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    mul_mat_vec_q_q4_K_reuse_weight<c_ncols_dst, c_nwarps, c_rows_per_block, c_interleaved_act><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols_x, channel_ratio, sample_ratio,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static bool mul_mat_vec_q_q4_K_interleaved_act_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy_interleaved, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y_interleaved, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y_interleaved, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y_interleaved,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_q4k_interleaved_act_enabled()) {
        return false;
    }
    if (src0 == nullptr || src0->type != GGML_TYPE_Q4_K || !ggml_cuda_mtp_mmvq_q4k_interleaved_act_name_allowed(src0->name)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_q4k_interleaved_act_ncols_allowed(ncols_dst) || ncols_dst < 2 || ncols_dst > 5) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK_K != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
    const int rows_requested   = ggml_cuda_mtp_mmvq_interleaved_act_rows("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_ROWS");
    const int nwarps_requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps("LLAMA_MTP_MMVQ_Q4K_INTERLEAVED_ACT_NWARPS");

    if (ggml_cuda_mtp_mmvq_q4k_interleaved_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_q4k_interleaved_act tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_q4_K_reuse_weight_launch<NCOLS, NWARPS, ROWS, true>( \
            vx, vy_interleaved, dst, ncols_x, nrows_x, stride_row_x, stride_col_y_interleaved, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y_interleaved, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y_interleaved, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, ROWS) \
    do { \
        if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 4, ROWS); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 2, ROWS); \
        } else { \
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 1, ROWS); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH(NCOLS) \
    do { \
        if (rows_requested == 2) { \
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 2); \
        } else { \
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 1); \
        } \
    } while (0)

    switch (ncols_dst) {
        case 2:
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH(2);
            return true;
        case 3:
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH(3);
            return true;
        case 4:
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH(4);
            return true;
        case 5:
            GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH(5);
            return true;
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH
#undef GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_ROWS
#undef GGML_CUDA_MMVQ_Q4K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS
}

template <int c_ncols_dst, int c_nwarps, int c_rows_per_block, bool c_interleaved_act = false>
static void mul_mat_vec_q_q5_K_reuse_weight_launch(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int warp_size, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const dim3 block_nums((nrows_x + c_rows_per_block - 1) / c_rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    mul_mat_vec_q_q5_K_reuse_weight<c_ncols_dst, c_nwarps, c_rows_per_block, c_interleaved_act><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols_x, channel_ratio, sample_ratio,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static bool mul_mat_vec_q_q5_K_interleaved_act_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy_interleaved, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y_interleaved, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y_interleaved, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y_interleaved,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_q5k_interleaved_act_enabled()) {
        return false;
    }
    if (src0 == nullptr || src0->type != GGML_TYPE_Q5_K || !ggml_cuda_mtp_mmvq_q5k_interleaved_act_name_allowed(src0->name)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_q5k_interleaved_act_ncols_allowed(ncols_dst) || ncols_dst < 2 || ncols_dst > 5) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK_K != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
    const int rows_requested   = ggml_cuda_mtp_mmvq_interleaved_act_rows("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_ROWS");
    const int nwarps_requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps("LLAMA_MTP_MMVQ_Q5K_INTERLEAVED_ACT_NWARPS");

    if (ggml_cuda_mtp_mmvq_q5k_interleaved_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_q5k_interleaved_act tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_q5_K_reuse_weight_launch<NCOLS, NWARPS, ROWS, true>( \
            vx, vy_interleaved, dst, ncols_x, nrows_x, stride_row_x, stride_col_y_interleaved, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y_interleaved, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y_interleaved, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, ROWS) \
    do { \
        if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 4, ROWS); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 2, ROWS); \
        } else { \
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS(NCOLS, 1, ROWS); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH(NCOLS) \
    do { \
        if (rows_requested == 2) { \
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 2); \
        } else { \
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_ROWS(NCOLS, 1); \
        } \
    } while (0)

    switch (ncols_dst) {
        case 2:
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH(2);
            return true;
        case 3:
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH(3);
            return true;
        case 4:
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH(4);
            return true;
        case 5:
            GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH(5);
            return true;
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH
#undef GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_ROWS
#undef GGML_CUDA_MMVQ_Q5K_INTERLEAVED_ACT_LAUNCH_NWARPS_ROWS
}

template <ggml_type type, int c_ncols_dst, int c_nwarps, int c_rows_per_block, bool c_interleaved_act = false>
static void mul_mat_vec_q_legacy_interleaved_act_launch_typed(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int warp_size, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const dim3 block_nums((nrows_x + c_rows_per_block - 1) / c_rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    mul_mat_vec_q_legacy_interleaved_act<type, c_ncols_dst, c_nwarps, c_rows_per_block, c_interleaved_act><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols_x, channel_ratio, sample_ratio,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static bool mul_mat_vec_q_legacy_interleaved_act_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy_interleaved, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y_interleaved, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y_interleaved, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y_interleaved,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_legacy_interleaved_act_enabled()) {
        return false;
    }
    if (src0 == nullptr || !ggml_cuda_mtp_mmvq_legacy_interleaved_act_type_allowed(src0->type) ||
            !ggml_cuda_mtp_mmvq_legacy_interleaved_act_name_allowed(src0->name)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_legacy_interleaved_act_ncols_allowed(ncols_dst) || ncols_dst < 2 || ncols_dst > 5) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK8_1 != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
    const int rows_requested   = ggml_cuda_mtp_mmvq_interleaved_act_rows("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_ROWS");
    const int nwarps_requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps("LLAMA_MTP_MMVQ_LEGACY_INTERLEAVED_ACT_NWARPS");

    if (ggml_cuda_mtp_mmvq_legacy_interleaved_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_legacy_interleaved_act tensor=%s type=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ggml_type_name(src0->type), ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_legacy_interleaved_act_launch_typed<TYPE, NCOLS, NWARPS, ROWS, true>( \
            vx, vy_interleaved, dst, ncols_x, nrows_x, stride_row_x, stride_col_y_interleaved, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y_interleaved, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y_interleaved, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, ROWS) \
    do { \
        if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 4, ROWS); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 2, ROWS); \
        } else { \
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 1, ROWS); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, NCOLS) \
    do { \
        if (rows_requested == 2) { \
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, 2); \
        } else { \
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, 1); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(TYPE) \
    do { \
        switch (ncols_dst) { \
            case 2: GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 2); return true; \
            case 3: GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 3); return true; \
            case 4: GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 4); return true; \
            case 5: GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 5); return true; \
            default: return false; \
        } \
    } while (0)

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q4_0);
        case GGML_TYPE_Q4_1:
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q4_1);
        case GGML_TYPE_Q5_0:
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q5_0);
        case GGML_TYPE_Q5_1:
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q5_1);
        case GGML_TYPE_Q8_0:
            GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q8_0);
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_SWITCH_NCOLS
#undef GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_NCOLS
#undef GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_ROWS
#undef GGML_CUDA_MMVQ_LEGACY_INTERLEAVED_ACT_LAUNCH_TYPED
}

template <ggml_type type, int c_ncols_dst, int c_nwarps, int c_rows_per_block, bool c_interleaved_act = false>
static void mul_mat_vec_q_lowk_interleaved_act_launch_typed(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const int warp_size, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const dim3 block_nums((nrows_x + c_rows_per_block - 1) / c_rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, c_nwarps, 1);
    mul_mat_vec_q_lowk_interleaved_act<type, c_ncols_dst, c_nwarps, c_rows_per_block, c_interleaved_act><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, dst, ncols_x, channel_ratio, sample_ratio,
            stride_row_x, stride_col_y, stride_col_dst,
            nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
            nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

static bool mul_mat_vec_q_lowk_interleaved_act_try_launch(
        const ggml_tensor * src0, const void * vx, const void * vy_interleaved, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y_interleaved, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y_interleaved, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y_interleaved,
        const int stride_sample_dst, const int cc, const int warp_size, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_lowk_interleaved_act_enabled()) {
        return false;
    }
    if (src0 == nullptr || !ggml_cuda_mtp_mmvq_lowk_interleaved_act_type_allowed(src0->type) ||
            !ggml_cuda_mtp_mmvq_lowk_interleaved_act_name_allowed(src0->name)) {
        return false;
    }
    if (!ggml_cuda_mtp_mmvq_lowk_interleaved_act_ncols_allowed(ncols_dst) || ncols_dst < 2 || ncols_dst > 5) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA(cc) || warp_size != 32 || ncols_x % QK_K != 0 || nrows_x <= 0 || nchannels_dst % nchannels_x != 0 || nsamples_dst % nsamples_x != 0) {
        return false;
    }

    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);
    const int rows_requested   = ggml_cuda_mtp_mmvq_interleaved_act_rows("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_ROWS");
    const int nwarps_requested = ggml_cuda_mtp_mmvq_interleaved_act_nwarps("LLAMA_MTP_MMVQ_LOWK_INTERLEAVED_ACT_NWARPS");

    if (ggml_cuda_mtp_mmvq_lowk_interleaved_act_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_lowk_interleaved_act tensor=%s type=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d nwarps=%d rows=%d\n",
                __func__, src0->name, ggml_type_name(src0->type), ncols_x, nrows_x, ncols_dst, nwarps_requested, rows_requested);
    }

#define GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, NWARPS, ROWS) \
    mul_mat_vec_q_lowk_interleaved_act_launch_typed<TYPE, NCOLS, NWARPS, ROWS, true>( \
            vx, vy_interleaved, dst, ncols_x, nrows_x, stride_row_x, stride_col_y_interleaved, stride_col_dst, \
            channel_ratio_fd, stride_channel_x, stride_channel_y_interleaved, stride_channel_dst, \
            sample_ratio_fd, stride_sample_x, stride_sample_y_interleaved, stride_sample_dst, \
            warp_size, nchannels_dst, nsamples_dst, stream)

#define GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, ROWS) \
    do { \
        if (nwarps_requested == 4) { \
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 4, ROWS); \
        } else if (nwarps_requested == 2) { \
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 2, ROWS); \
        } else { \
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_TYPED(TYPE, NCOLS, 1, ROWS); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, NCOLS) \
    do { \
        if (rows_requested == 2) { \
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, 2); \
        } else { \
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_ROWS(TYPE, NCOLS, 1); \
        } \
    } while (0)

#define GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_SWITCH_NCOLS(TYPE) \
    do { \
        switch (ncols_dst) { \
            case 2: GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 2); return true; \
            case 3: GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 3); return true; \
            case 4: GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 4); return true; \
            case 5: GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS(TYPE, 5); return true; \
            default: return false; \
        } \
    } while (0)

    switch (src0->type) {
        case GGML_TYPE_Q2_K:
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q2_K);
        case GGML_TYPE_Q3_K:
            GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_SWITCH_NCOLS(GGML_TYPE_Q3_K);
        default:
            return false;
    }

#undef GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_SWITCH_NCOLS
#undef GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_NCOLS
#undef GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_ROWS
#undef GGML_CUDA_MMVQ_LOWK_INTERLEAVED_ACT_LAUNCH_TYPED
}

template <ggml_type type, bool small_k = false>
static void mul_mat_vec_q_serial_columns_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t ncols_dst,
        const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const int warp_size, const mmvq_parameter_table_id table_id,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const int nwarps = calc_nwarps(type, 1, table_id);
    const int rows_per_block = calc_rows_per_block(1, table_id, small_k, nwarps);
    const dim3 block_nums((nrows_x + rows_per_block - 1) / rows_per_block,
            nchannels_dst, nsamples_dst*ncols_dst);
    const dim3 block_dims(warp_size, nwarps, 1);

    mul_mat_vec_q_serial_columns<type, small_k><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, ids, dst, ncols_x, nchannels_y, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
            stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type, bool small_k = false>
static void mul_mat_vec_q_serial_columns_fused_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t ncols_dst,
        const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const int warp_size, const mmvq_parameter_table_id table_id,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {

    const int nwarps = calc_nwarps(type, 1, table_id);
    const int rows_per_block = calc_rows_per_block(1, table_id, small_k, nwarps);
    const dim3 block_nums((nrows_x + rows_per_block - 1) / rows_per_block,
            nchannels_dst, nsamples_dst*ncols_dst);
    const dim3 block_dims(warp_size, nwarps, 1);

    mul_mat_vec_q_serial_columns_fused<type, small_k><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, ncols_dst,
            stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
            stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
            stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

static void mul_mat_vec_q_serial_columns_fused_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_dst > 1 && ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

#define GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(TYPE) \
        case TYPE: \
            mul_mat_vec_q_serial_columns_fused_launch<TYPE>( \
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, nchannels_y_fd, \
                    stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                    stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, \
                    warp_size, table_id, nchannels_dst, nsamples_dst, stream); \
            break

    switch (type_x) {
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q1_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q4_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q4_1);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q5_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q5_1);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q8_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_MXFP4);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_NVFP4);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q2_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q3_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q4_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q5_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_Q6_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ2_XXS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ2_XS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ2_S);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ3_XXS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ1_S);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ1_M);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ4_NL);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ4_XS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH(GGML_TYPE_IQ3_S);
        default:
            GGML_ABORT("fatal error");
            break;
    }

#undef GGML_CUDA_MMVQ_SERIAL_COLUMNS_FUSED_DISPATCH
}

static void mul_mat_vec_q_serial_columns_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_dst > 1 && ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

#define GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(TYPE) \
        case TYPE: \
            mul_mat_vec_q_serial_columns_launch<TYPE>( \
                    vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, nchannels_y_fd, \
                    stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                    stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, \
                    warp_size, table_id, nchannels_dst, nsamples_dst, stream); \
            break

    switch (type_x) {
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q1_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q4_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q4_1);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q5_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q5_1);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q8_0);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_MXFP4);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_NVFP4);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q2_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q3_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q4_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q5_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_Q6_K);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ2_XXS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ2_XS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ2_S);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ3_XXS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ1_S);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ1_M);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ4_NL);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ4_XS);
        GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH(GGML_TYPE_IQ3_S);
        default:
            GGML_ABORT("fatal error");
            break;
    }

#undef GGML_CUDA_MMVQ_SERIAL_COLUMNS_DISPATCH
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

static int ggml_cuda_mtp_mmvq_moe_rows_per_block(
        const ggml_type type, const bool has_fusion,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t ncols_dst) {
    const char * env = getenv("LLAMA_MTP_MMVQ_MOE_ROWS_PER_BLOCK");
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        const long v = strtol(env, &end, 10);
        return end != env && (v == 1 || v == 2 || v == 4 || v == 8) ? (int) v : 2;
    }

    // Real-ish MoE down projection: K=512 -> rows=2048, tokens<=4, top-k routes.
    // Larger row tiles reduce CTA count and reuse each Q8 activation load across more output rows.
    if (!has_fusion && ncols_dst <= 4 && ncols_x <= 1024 && nrows_x >= 1024) {
        if (type == GGML_TYPE_IQ4_XS) {
            return 8;
        }
        if (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_Q8_0) {
            return 4;
        }
    }

    return 2;
}

template <ggml_type type, int rows_per_block>
static void mul_mat_vec_q_moe_launch_rows(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const bool has_fusion, cudaStream_t stream) {
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);

    if (has_fusion) {
        mul_mat_vec_q_moe<type, rows_per_block, true><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        mul_mat_vec_q_moe<type, rows_per_block, false><<<block_nums, block_dims, 0, stream>>>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const bool has_fusion, cudaStream_t stream) {

    switch (ggml_cuda_mtp_mmvq_moe_rows_per_block(type, has_fusion, ncols_x, nrows_x, ncols_dst)) {
        case 1:
            mul_mat_vec_q_moe_launch_rows<type, 1>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, warp_size, nchannels_dst, has_fusion, stream);
            break;
        case 4:
            mul_mat_vec_q_moe_launch_rows<type, 4>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, warp_size, nchannels_dst, has_fusion, stream);
            break;
        case 8:
            mul_mat_vec_q_moe_launch_rows<type, 8>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, warp_size, nchannels_dst, has_fusion, stream);
            break;
        default:
            mul_mat_vec_q_moe_launch_rows<type, 2>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    ncols_dst, ids_stride, warp_size, nchannels_dst, has_fusion, stream);
            break;
    }
}


static bool ggml_cuda_mtp_mmvq_moe_q8_0_dot4_try_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int cc, const int warp_size, const int nchannels_dst, const bool has_fusion,
        const char * tensor_name, cudaStream_t stream) {
    const bool log_route = ggml_cuda_mtp_mmvq_moe_q8_0_dot4_log_enabled() ||
            ggml_cuda_mtp_q8_dot4_mmvq_log_enabled() ||
            ggml_cuda_mtp_mmvq_moe_exp_tile_log_enabled();

    const char * reject = ggml_cuda_mtp_mmvq_moe_q8_0_dot4_reject_reason(
            cc, warp_size, ids != nullptr, ncols_x, nrows_x, ncols_dst, nchannels_dst);
    if (reject != nullptr) {
        if (log_route) {
            GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_small_route_dot4_q8_0 tensor=%s status=reject reject=%s ncols_x=%u nrows_x=%u ncols_dst=%u fusion=%d ids=%d routes=%lld broadcast_y=%d\n",
                    __func__, tensor_name ? tensor_name : "-", reject,
                    ncols_x, nrows_x, ncols_dst, has_fusion ? 1 : 0, ids ? 1 : 0,
                    (long long) ncols_dst * (long long) nchannels_dst,
                    nchannels_y.z == 1 ? 1 : 0);
        }
        return false;
    }

    const int rows = ggml_cuda_mtp_mmvq_moe_q8_0_dot4_rows_per_block(has_fusion, ncols_x, nrows_x);
    if (log_route) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_small_route_dot4_q8_0 tensor=%s status=selected ncols_x=%u nrows_x=%u ncols_dst=%u tokens=%u topk=%d routes=%lld rows=%d fusion=%d broadcast_y=%d\n",
                __func__, tensor_name ? tensor_name : "-",
                ncols_x, nrows_x, ncols_dst, ncols_dst, nchannels_dst,
                (long long) ncols_dst * (long long) nchannels_dst,
                rows, has_fusion ? 1 : 0, nchannels_y.z == 1 ? 1 : 0);
    }

#define GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS(ROWS) do { \
        constexpr int rows_per_block_t = (ROWS); \
        const int64_t nblocks_rows = (nrows_x + rows_per_block_t - 1) / rows_per_block_t; \
        const dim3 block_nums(nblocks_rows, nchannels_dst); \
        const dim3 block_dims(warp_size, ncols_dst); \
        if (has_fusion) { \
            mul_mat_vec_q8_0_moe_dot4<rows_per_block_t, true><<<block_nums, block_dims, 0, stream>>>( \
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride); \
        } else { \
            mul_mat_vec_q8_0_moe_dot4<rows_per_block_t, false><<<block_nums, block_dims, 0, stream>>>( \
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride); \
        } \
    } while (0)

    switch (rows) {
        case 1:  GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS(1); break;
        case 4:  GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS(4); break;
        case 8:  GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS(8); break;
        case 2:
        default: GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS(2); break;
    }

#undef GGML_CUDA_MTP_MMVQ_MOE_Q8_0_DOT4_DISPATCH_ROWS

    return true;
}

template <int EXP_TILE, int ROWS_PER_BLOCK>
static void mul_mat_vec_q_iq3_s_moe_gateup_tile_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = ROWS_PER_BLOCK;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, (nchannels_dst + EXP_TILE - 1) / EXP_TILE);
    const dim3 block_dims(warp_size, ncols_dst, EXP_TILE);

    mul_mat_vec_q_iq3_s_moe_gateup_tile<rows_per_block, EXP_TILE><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_dst,
        ncols_dst, ids_stride, nchannels_dst);
}

static void mul_mat_vec_q_iq3_s_moe_gateup_tile_switch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const int exp_tile, const int rows_per_block, cudaStream_t stream) {
    if (exp_tile == 2 && rows_per_block == 1) {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<2, 1>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    } else if (exp_tile == 2 && rows_per_block == 4) {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<2, 4>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    } else if (exp_tile == 2) {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<2, 2>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    } else if (rows_per_block == 1) {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<4, 1>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    } else if (rows_per_block == 4) {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<4, 4>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    } else {
        mul_mat_vec_q_iq3_s_moe_gateup_tile_launch<4, 2>(
            vx, vy, ids, dst, ncols_x, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
    }
}

template <int EXP_TILE, int ROWS_PER_BLOCK>
static void mul_mat_vec_q_iq3_s_moe_gateup_sidecar_tile_launch(
        const int32_t * sidecar_packs, const uint8_t * sidecar_scales, const half * sidecar_ds,
        const void * vy, const int32_t * ids_local, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = ROWS_PER_BLOCK;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, (nchannels_dst + EXP_TILE - 1) / EXP_TILE);
    const dim3 block_dims(warp_size, ncols_dst, EXP_TILE);

    mul_mat_vec_q_iq3_s_moe_gateup_sidecar_tile<rows_per_block, EXP_TILE><<<block_nums, block_dims, 0, stream>>>(
        sidecar_packs, sidecar_scales, sidecar_ds, vy, ids_local, dst, ncols_x, nrows_x,
        stride_col_y, stride_col_dst, stride_channel_dst,
        ncols_dst, nchannels_dst);
}

template <int EXP_TILE, int ROWS_PER_BLOCK>
static void mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch(
        const int32_t * sidecar_packs, const uint8_t * sidecar_scales, const half * sidecar_ds,
        const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = ROWS_PER_BLOCK;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, (nchannels_dst + EXP_TILE - 1) / EXP_TILE);
    const dim3 block_dims(warp_size, ncols_dst, EXP_TILE);

    mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile<rows_per_block, EXP_TILE><<<block_nums, block_dims, 0, stream>>>(
        sidecar_packs, sidecar_scales, sidecar_ds, vy, dst, ncols_x, nrows_x,
        stride_col_y, stride_col_dst, stride_channel_dst,
        ncols_dst, nchannels_dst);
}

static bool mul_mat_vec_q_iq3_s_moe_gateup_sidecar_try_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_y, const int nchannels_dst, cudaStream_t stream) {

    if (!ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_enabled()) {
        return false;
    }

    const char * tensor_name = g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ?
        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-";

    const auto reject = [&](const char * reason) {
        if (ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_iq3s_sidecar tensor=%s status=reject reject=%s ncols_x=%u nrows_x=%u ncols_dst=%u nchannels_y=%d nchannels_dst=%d\n",
                    __func__, tensor_name, reason, ncols_x, nrows_x, ncols_dst, nchannels_y, nchannels_dst);
        }
        return false;
    };

    if (!ids) {
        return reject("no_ids");
    }
    if (warp_size != 32) {
        return reject("warp_size");
    }
    const bool stream_is_capturing = ggml_cuda_mtp_q8_dot4_mmvq_stream_is_capturing(stream);
    if (!ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_name_allowed(tensor_name)) {
        return reject("name_filter");
    }
    if (!ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_shape_allowed(ncols_x, nrows_x, ncols_dst, nchannels_y, nchannels_dst)) {
        return reject("shape");
    }
    if (ncols_x % QK_K != 0) {
        return reject("k_not_qk_k_multiple");
    }

    const uint32_t blocks_per_row_x = ncols_x / QK_K;
    const uint32_t max_unique = ncols_dst * (uint32_t) nchannels_dst;

    const size_t sidecar_blocks = (size_t) max_unique*nrows_x*blocks_per_row_x;
    const size_t packs_needed = sidecar_blocks*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS*GGML_CUDA_MMVQ_IQ3S_SIDECAR_PACKS_PER_IQS*sizeof(int32_t);
    const size_t scales_needed = sidecar_blocks*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS*sizeof(uint8_t);
    const size_t ds_needed = sidecar_blocks*sizeof(half);
    const size_t ids_needed = (size_t) max_unique*sizeof(int32_t);
    const size_t count_needed = sizeof(int32_t);
    size_t sidecar_total_bytes = packs_needed + scales_needed + ds_needed;

    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace * ws_ptr = &g_ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace;
    if (stream_is_capturing) {
        if (ws_ptr->packs_bytes < packs_needed || ws_ptr->scales_bytes < scales_needed || ws_ptr->ds_bytes < ds_needed ||
                ws_ptr->unique_ids_bytes < ids_needed || ws_ptr->ids_local_bytes < ids_needed || ws_ptr->unique_count_bytes < count_needed) {
            return reject("stream_capture_workspace_cold");
        }
    } else {
        ws_ptr = &ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace_get(max_unique, nrows_x, blocks_per_row_x, &sidecar_total_bytes);
    }
    ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_workspace & ws = *ws_ptr;

    const uint64_t gen_items = (uint64_t) max_unique*nrows_x*blocks_per_row_x*GGML_CUDA_MMVQ_IQ3S_SIDECAR_IQS_GROUPS;
    const dim3 gen_grid((uint32_t) ((gen_items + 255) / 256));
    ggml_cuda_mmvq_iq3_s_moe_sidecar_gen_slots<<<gen_grid, dim3(256), 0, stream>>>(
            vx, ids, ws.packs, ws.scales, ws.ds,
            ncols_dst, nchannels_dst, nrows_x, blocks_per_row_x,
            stride_row_x, stride_channel_x, ids_stride);

    const int sidecar_rows = ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_rows();
    const int sidecar_exp_tile = ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_exp_tile();
    if (sidecar_exp_tile == 2 && sidecar_rows == 1) {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<2, 1>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    } else if (sidecar_exp_tile == 2 && sidecar_rows == 2) {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<2, 2>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    } else if (sidecar_exp_tile == 2) {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<2, 4>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    } else if (sidecar_rows == 1) {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<4, 1>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    } else if (sidecar_rows == 2) {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<4, 2>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    } else {
        mul_mat_vec_q_iq3_s_moe_gateup_sidecar_slots_tile_launch<4, 4>(
                ws.packs, ws.scales, ws.ds, vy, dst, ncols_x, nrows_x,
                stride_col_y, stride_col_dst, stride_channel_dst,
                ncols_dst, warp_size, nchannels_dst, stream);
    }

    if (ggml_cuda_mtp_mmvq_moe_iq3s_sidecar_log_enabled()) {
        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_iq3s_sidecar_slots tensor=%s status=selected ncols_x=%u nrows_x=%u ncols_dst=%u nchannels_dst=%d route_slots=%u sidecar_bytes=%zu rows=%d exp_tile=%d\n",
                __func__, tensor_name, ncols_x, nrows_x, ncols_dst, nchannels_dst, max_unique, sidecar_total_bytes, sidecar_rows, sidecar_exp_tile);
    }
    return true;
}

template <ggml_type type, int EXP_TILE, int ROWS_PER_BLOCK>
static void mul_mat_vec_q_moe_exp_tile_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = ROWS_PER_BLOCK;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, (nchannels_dst + EXP_TILE - 1) / EXP_TILE);
    const dim3 block_dims(warp_size, ncols_dst);

    mul_mat_vec_q_moe_exp_tile<type, rows_per_block, EXP_TILE><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride, nchannels_dst);
}

static void mul_mat_vec_q_moe_exp_tile_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const int exp_tile, const int rows_per_block, cudaStream_t stream) {

#define GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(TYPE) \
        case TYPE: \
            if (exp_tile == 2 && rows_per_block == 1) { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 2, 1>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } else if (exp_tile == 2 && rows_per_block == 4) { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 2, 4>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } else if (exp_tile == 2) { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 2, 2>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } else if (rows_per_block == 1) { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 4, 1>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } else if (rows_per_block == 4) { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 4, 4>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } else { \
                mul_mat_vec_q_moe_exp_tile_launch<TYPE, 4, 2>( \
                    vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    stride_channel_x, stride_channel_y, stride_channel_dst, \
                    ncols_dst, ids_stride, warp_size, nchannels_dst, stream); \
            } \
            break

    switch (type_x) {
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q1_0);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q4_0);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q4_1);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q5_0);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q5_1);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q8_0);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_MXFP4);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_NVFP4);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q2_K);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q3_K);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q4_K);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q5_K);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_Q6_K);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ2_XXS);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ2_XS);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ2_S);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ3_XXS);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ1_S);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ1_M);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ4_NL);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ4_XS);
        GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH(GGML_TYPE_IQ3_S);
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GGML_CUDA_MMVQ_MOE_EXP_TILE_DISPATCH
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

    if constexpr (type == GGML_TYPE_IQ3_S) {
        if (!has_fusion && has_ids && mul_mat_vec_q_iq3_s_moe_gateup_sidecar_try_launch(
                    vx, vy, ids, dst, ncols_x, nrows_x,
                    stride_row_x, stride_col_y, stride_col_dst,
                    stride_channel_x, stride_channel_dst,
                    ncols_dst, ids_stride, warp_size, nchannels_y, nchannels_dst, stream)) {
            return;
        }
    }

    if (has_ids && ncols_dst > 1) {
        const int moe_exp_tile = has_fusion ? 0 : ggml_cuda_mtp_mmvq_moe_exp_tile();
        if (moe_exp_tile > 1 && ggml_cuda_mtp_mmvq_moe_exp_tile_shape_allowed(ncols_x, nrows_x)) {
            const int moe_exp_tile_rows = ggml_cuda_mtp_mmvq_moe_exp_tile_rows();
            if constexpr (type == GGML_TYPE_IQ3_S) {
                const bool gateup_shape = nchannels_y_fd.z == 1 && ncols_x >= nrows_x;
                if (gateup_shape) {
                    if (ggml_cuda_mtp_mmvq_moe_exp_tile_log_enabled()) {
                        GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_iq3_gateup_tile tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d exp_tile=%d rows=%d\n",
                                __func__, g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ? g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-",
                                ncols_x, nrows_x, ncols_dst, moe_exp_tile, moe_exp_tile_rows);
                    }
                    mul_mat_vec_q_iq3_s_moe_gateup_tile_switch(
                        vx, vy, ids, dst, ncols_x, nrows_x,
                        stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_dst,
                        ncols_dst, ids_stride, warp_size, nchannels_dst, moe_exp_tile, moe_exp_tile_rows, stream);
                    return;
                }
            }
            if (ggml_cuda_mtp_mmvq_moe_exp_tile_log_enabled()) {
                GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_exp_tile tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d exp_tile=%d rows=%d\n",
                        __func__, g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ? g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-",
                        ncols_x, nrows_x, ncols_dst, moe_exp_tile, moe_exp_tile_rows);
            }
            mul_mat_vec_q_moe_exp_tile_switch_type(
                vx, type, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_dst, moe_exp_tile, moe_exp_tile_rows, stream);
            return;
        }

        if constexpr (type == GGML_TYPE_Q8_0) {
            if (ggml_cuda_mtp_mmvq_moe_q8_0_dot4_try_launch(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                        stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        ncols_dst, ids_stride, cc, warp_size, nchannels_dst, has_fusion,
                        g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ? g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-", stream)) {
                return;
            }
        }

        // Multi-token MUL_MAT_ID path - dedicated route-direct MoE GEMV kernel.
        // If gate/up fusion is present, keep the tiny-route path fused here instead of
        // falling back to grouped MMQ/WMMA-shaped work.
        if (has_fusion && ggml_cuda_mtp_mmvq_moe_exp_tile_log_enabled()) {
            GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_moe_small_route_fused_gateup tensor=%s status=selected ncols_x=%d nrows_x=%d ncols_dst=%d tokens=%d topk=%d routes=%lld broadcast_y=%d\n",
                    __func__, g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name ? g_ggml_cuda_dp16_mmvq_mtp_q8_dot4_tensor_name : "-",
                    ncols_x, nrows_x, ncols_dst, ncols_dst, nchannels_dst, (long long) ncols_dst*nchannels_dst,
                    nchannels_y == 1 ? 1 : 0);
        }
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, has_fusion, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            if (ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    type, cc, warp_size, c_ncols_dst, has_fusion, has_ids, ncols_x, nrows_x)) {
                ggml_cuda_mtp_q8_dot4_mmvq_log_selected(ncols_x, nrows_x, c_ncols_dst, has_fusion, has_ids);
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
                ggml_cuda_mtp_q8_dot4_mmvq_log_selected(ncols_x, nrows_x, c_ncols_dst, has_fusion, has_ids);
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
                ggml_cuda_mtp_q8_dot4_mmvq_log_selected(ncols_x, nrows_x, c_ncols_dst, has_fusion, has_ids);
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
                ggml_cuda_mtp_q8_dot4_mmvq_log_selected(ncols_x, nrows_x, c_ncols_dst, has_fusion, has_ids);
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
        const char * mtp_fuse_glu = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_FUSE_GLU");
        const char * mtp_packed_glu = getenv("GGML_CUDA_ROCM_MTP_Q8_DOT4_MMVQ_PACKED_GLU");
        const bool allow_mtp_fuse_glu = !mtp_fuse_glu || atoi(mtp_fuse_glu) != 0;
        const bool allow_mtp_packed_glu = !mtp_packed_glu || atoi(mtp_packed_glu) != 0;
        const bool allow_mtp_fused_n1_4 = !ids && dst->ne[1] >= 1 && dst->ne[1] <= 4 &&
            (allow_mtp_fuse_glu || allow_mtp_packed_glu) &&
            ggml_cuda_mtp_q8_dot4_mmvq_tensor_allowed(src0);
        const bool allow_mtp_reuse_n_xbias = false;
        GGML_ASSERT( !ids || (dst->ne[2] > 0 && dst->ne[2] <= MMVQ_MAX_BATCH_SIZE));
        GGML_ASSERT(  ids || dst->ne[1] == 1 || allow_mtp_fused_n1_4 || allow_mtp_reuse_n_xbias);

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
    const size_t src1_q8_1_bytes = (size_t) (ne13*ne12 * ne11*ne10_padded) * sizeof(block_q8_1)/QK8_1;

    {
        const bool early_has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
        const int64_t ncols_dst_i = ne1;
        const bool q6k_interleaved_wanted = !ids_d && !early_has_fusion &&
                ggml_cuda_mtp_mmvq_q6k_interleaved_act_enabled() &&
                src0->type == GGML_TYPE_Q6_K &&
                ggml_cuda_mtp_mmvq_q6k_interleaved_act_name_allowed(src0->name) &&
                ggml_cuda_mtp_mmvq_q6k_interleaved_act_ncols_allowed((int) ncols_dst_i) &&
                ncols_dst_i >= 2 && ncols_dst_i <= 5;
        const bool q4k_interleaved_wanted = !ids_d && !early_has_fusion &&
                ggml_cuda_mtp_mmvq_q4k_interleaved_act_enabled() &&
                src0->type == GGML_TYPE_Q4_K &&
                ggml_cuda_mtp_mmvq_q4k_interleaved_act_name_allowed(src0->name) &&
                ggml_cuda_mtp_mmvq_q4k_interleaved_act_ncols_allowed((int) ncols_dst_i) &&
                ncols_dst_i >= 2 && ncols_dst_i <= 5;
        const bool q5k_interleaved_wanted = !ids_d && !early_has_fusion &&
                ggml_cuda_mtp_mmvq_q5k_interleaved_act_enabled() &&
                src0->type == GGML_TYPE_Q5_K &&
                ggml_cuda_mtp_mmvq_q5k_interleaved_act_name_allowed(src0->name) &&
                ggml_cuda_mtp_mmvq_q5k_interleaved_act_ncols_allowed((int) ncols_dst_i) &&
                ncols_dst_i >= 2 && ncols_dst_i <= 5;
        const bool legacy_interleaved_wanted = !ids_d && !early_has_fusion &&
                ggml_cuda_mtp_mmvq_legacy_interleaved_act_enabled() &&
                ggml_cuda_mtp_mmvq_legacy_interleaved_act_type_allowed(src0->type) &&
                ggml_cuda_mtp_mmvq_legacy_interleaved_act_name_allowed(src0->name) &&
                ggml_cuda_mtp_mmvq_legacy_interleaved_act_ncols_allowed((int) ncols_dst_i) &&
                ncols_dst_i >= 2 && ncols_dst_i <= 5;
        const bool lowk_interleaved_wanted = !ids_d && !early_has_fusion &&
                ggml_cuda_mtp_mmvq_lowk_interleaved_act_enabled() &&
                ggml_cuda_mtp_mmvq_lowk_interleaved_act_type_allowed(src0->type) &&
                ggml_cuda_mtp_mmvq_lowk_interleaved_act_name_allowed(src0->name) &&
                ggml_cuda_mtp_mmvq_lowk_interleaved_act_ncols_allowed((int) ncols_dst_i) &&
                ncols_dst_i >= 2 && ncols_dst_i <= 5;
        if (q6k_interleaved_wanted || q4k_interleaved_wanted || q5k_interleaved_wanted || legacy_interleaved_wanted || lowk_interleaved_wanted) {
            const int64_t nchannels_dst_i      = ne2;
            const int64_t blocks_per_col_i     = ne10_padded / QK8_1;
            const int64_t stride_col_y_i       = ne11;
            const int64_t stride_channel_y_i   = blocks_per_col_i * ne11;
            const int64_t stride_sample_y_i    = ne12 * stride_channel_y_i;
            const int64_t stride_col_dst_i     = dst->nb[1] / ts_dst;
            const int64_t stride_channel_dst_i = dst->nb[2] / ts_dst;
            const int64_t stride_sample_dst_i  = dst->nb[3] / ts_dst;
            const int64_t stride_row_x_i       = src0->nb[1] / ts_src0;
            const int64_t stride_channel_x_i   = src0->nb[2] / ts_src0;
            const int64_t stride_sample_x_i    = src0->nb[3] / ts_src0;
            const int64_t src1_s11_i           = src1->nb[1] / ts_src1;
            const int64_t src1_s12_i           = src1->nb[2] / ts_src1;
            const int64_t src1_s13_i           = src1->nb[3] / ts_src1;

            ggml_cuda_pool_alloc<char> src1_q8_1_interleaved(ctx.pool());
            char * src1_q8_1_interleaved_d = src1_q8_1_interleaved.alloc(src1_q8_1_bytes);
            quantize_row_q8_1_interleaved_mmvq_cuda(
                    src1_d, src1_q8_1_interleaved_d,
                    ne10, src1_s11_i, src1_s12_i, src1_s13_i,
                    ne10_padded, ne11, ne12, ne13, stream);

            const int device = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[device].cc;
            const int warp_size = ggml_cuda_info().devices[device].warp_size;
            if (q6k_interleaved_wanted && mul_mat_vec_q_q6_K_interleaved_act_try_launch(
                        src0, src0->data, src1_q8_1_interleaved_d, dst_d, ne00, ne01, ncols_dst_i,
                        stride_row_x_i, stride_col_y_i, stride_col_dst_i,
                        ne02, nchannels_dst_i, stride_channel_x_i, stride_channel_y_i, stride_channel_dst_i,
                        ne03, ne3, stride_sample_x_i, stride_sample_y_i, stride_sample_dst_i,
                        cc, warp_size, stream)) {
                ggml_cuda_mtp_mmvq_route_census_record("mmvq_q6k_interleaved_act", src0, ne00, ne01, ncols_dst_i, false, false);
                return;
            }
            if (q4k_interleaved_wanted && mul_mat_vec_q_q4_K_interleaved_act_try_launch(
                        src0, src0->data, src1_q8_1_interleaved_d, dst_d, ne00, ne01, ncols_dst_i,
                        stride_row_x_i, stride_col_y_i, stride_col_dst_i,
                        ne02, nchannels_dst_i, stride_channel_x_i, stride_channel_y_i, stride_channel_dst_i,
                        ne03, ne3, stride_sample_x_i, stride_sample_y_i, stride_sample_dst_i,
                        cc, warp_size, stream)) {
                ggml_cuda_mtp_mmvq_route_census_record("mmvq_q4k_interleaved_act", src0, ne00, ne01, ncols_dst_i, false, false);
                return;
            }
            if (q5k_interleaved_wanted && mul_mat_vec_q_q5_K_interleaved_act_try_launch(
                        src0, src0->data, src1_q8_1_interleaved_d, dst_d, ne00, ne01, ncols_dst_i,
                        stride_row_x_i, stride_col_y_i, stride_col_dst_i,
                        ne02, nchannels_dst_i, stride_channel_x_i, stride_channel_y_i, stride_channel_dst_i,
                        ne03, ne3, stride_sample_x_i, stride_sample_y_i, stride_sample_dst_i,
                        cc, warp_size, stream)) {
                ggml_cuda_mtp_mmvq_route_census_record("mmvq_q5k_interleaved_act", src0, ne00, ne01, ncols_dst_i, false, false);
                return;
            }
            if (legacy_interleaved_wanted && mul_mat_vec_q_legacy_interleaved_act_try_launch(
                        src0, src0->data, src1_q8_1_interleaved_d, dst_d, ne00, ne01, ncols_dst_i,
                        stride_row_x_i, stride_col_y_i, stride_col_dst_i,
                        ne02, nchannels_dst_i, stride_channel_x_i, stride_channel_y_i, stride_channel_dst_i,
                        ne03, ne3, stride_sample_x_i, stride_sample_y_i, stride_sample_dst_i,
                        cc, warp_size, stream)) {
                ggml_cuda_mtp_mmvq_route_census_record("mmvq_legacy_interleaved_act", src0, ne00, ne01, ncols_dst_i, false, false);
                return;
            }
            if (lowk_interleaved_wanted && mul_mat_vec_q_lowk_interleaved_act_try_launch(
                        src0, src0->data, src1_q8_1_interleaved_d, dst_d, ne00, ne01, ncols_dst_i,
                        stride_row_x_i, stride_col_y_i, stride_col_dst_i,
                        ne02, nchannels_dst_i, stride_channel_x_i, stride_channel_y_i, stride_channel_dst_i,
                        ne03, ne3, stride_sample_x_i, stride_sample_y_i, stride_sample_dst_i,
                        cc, warp_size, stream)) {
                ggml_cuda_mtp_mmvq_route_census_record("mmvq_lowk_interleaved_act", src0, ne00, ne01, ncols_dst_i, false, false);
                return;
            }
        }
    }

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * src1_q8_1_d = nullptr;
    bool src1_q8_1_needs_quantize = true;
    const bool src1_q8_1_act_cache_used = ggml_cuda_mtp_q8_dot4_mmvq_act_cache_try_get(
            src0, src1, ids_d != nullptr, ne10_padded, src1_q8_1_bytes, stream,
            &src1_q8_1_d, &src1_q8_1_needs_quantize);
    if (!src1_q8_1_act_cache_used) {
        src1_q8_1_d = src1_q8_1.alloc(src1_q8_1_bytes);
        src1_q8_1_needs_quantize = true;
    }
    if (src1_q8_1_needs_quantize) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1_d, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (src1_q8_1_act_cache_used) {
            ggml_cuda_mtp_q8_dot4_mmvq_act_cache_mark_valid(src0->name);
        }
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

    ggml_cuda_pool_alloc<int32_t> src1_q8sum4(ctx.pool());
    const int32_t * src1_q8sum4_d = nullptr;
    auto prepare_q8sum4 = [&](int64_t ncols_check) {
        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        if (src1_q8sum4_d == nullptr && ggml_cuda_rdna3_mmvq_dot4_wants_q8sum4(src0->type, cc, warp_size, ncols_check, ids_d != nullptr, ne00, ne01)) {
            const int64_t n_q8_blocks_total = ne13*ne12 * ne11*(ne10_padded / QK8_1);
            src1_q8sum4_d = src1_q8sum4.alloc(4*n_q8_blocks_total);
            ggml_cuda_rdna3_q8_1_sum4_precompute(src1_q8_1_d, src1_q8sum4.get(), n_q8_blocks_total, stream);
        }
    };

    const bool mtp_q8_dot4_mmvq_scope_enabled = ggml_cuda_mtp_q8_dot4_mmvq_tensor_allowed(src0);
    ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope_guard mtp_q8_dot4_mmvq_scope(mtp_q8_dot4_mmvq_scope_enabled, src0->name);

    {
        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        const bool has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
        const bool fusion_x_bias_only = fusion_local.x_bias != nullptr && fusion_local.gate == nullptr && fusion_local.gate_bias == nullptr;
        const bool has_ids = ids_d != nullptr;
        if (mtp_q8_dot4_mmvq_scope_enabled) {
            ggml_cuda_mtp_q8_dot4_mmvq_log_tensor(src0, src1, dst, ne00, ne01, ncols_dst, has_fusion, has_ids);
        }
        if (ggml_cuda_mtp_mmvq_serial_columns_enabled(src0, ncols_dst, has_ids, has_fusion)) {
            if (ggml_cuda_mtp_mmvq_serial_columns_log_enabled()) {
                GGML_LOG_INFO("%s: mtp_weight_route route=mmvq_serial_columns tensor=%s status=selected ncols_x=%lld nrows_x=%lld ncols_dst=%lld ids=%d fusion=%d\n",
                        __func__, src0->name, (long long) ne00, (long long) ne01, (long long) ncols_dst,
                        has_ids ? 1 : 0, has_fusion ? 1 : 0);
            }
            const bool n1_dp16_q8_dot4 = ggml_cuda_dp16_mmvq_q8_dot4_supported(
                    src0->type, cc, warp_size, 1, false, has_ids, ne00, ne01);
            const bool n1_rdna3_q4k_dot4 = ggml_cuda_rdna3_mmvq_dot4_wants_q8sum4(
                    src0->type, cc, warp_size, 1, has_ids, ne00, ne01);
            if (ggml_cuda_mtp_mmvq_serial_columns_single_launch_enabled() && n1_rdna3_q4k_dot4) {
                prepare_q8sum4(1);
                if (ggml_cuda_rdna3_mmvq_dot4_serial_columns_supported(
                            src0->type, cc, warp_size, ncols_dst, has_fusion, has_ids, ne00, ne01, src1_q8sum4_d)) {
                    if (ggml_cuda_mtp_mmvq_serial_columns_log_enabled()) {
                        GGML_LOG_INFO("%s: mtp_weight_route route=rdna3_mmvq_dot4_serial_columns tensor=%s status=selected ncols_x=%lld nrows_x=%lld ncols_dst=%lld ids=%d fusion=%d\n",
                                __func__, src0->name, (long long) ne00, (long long) ne01, (long long) ncols_dst, has_ids ? 1 : 0, has_fusion ? 1 : 0);
                    }
                    const uint3 nchannels_y_fd   = has_ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
                    const uint3 channel_ratio_fd = has_ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / ne02);
                    const uint3 sample_ratio_fd  = init_fastdiv_values(ne3 / ne03);
                    ggml_cuda_rdna3_mmvq_dot4_serial_columns_launch<GGML_TYPE_Q4_K>(
                            src0->data, src1_q8_1_d, src1_q8sum4_d, ids_d, fusion_local, dst_d,
                            ne00, ne01, ncols_dst, nchannels_y_fd, channel_ratio_fd, sample_ratio_fd,
                            s01, stride_col_y, stride_col_dst, nchannels_dst,
                            s02, stride_channel_y, stride_channel_dst,
                            ne3, s03, s13, s3, ids_stride, stream);
                    ggml_cuda_mtp_mmvq_route_census_record("rdna3_mmvq_dot4_serial_columns", src0, ne00, ne01, ncols_dst, has_ids, has_fusion);
                    return;
                }
            }
            const bool use_single_launch_serial_columns = ggml_cuda_mtp_mmvq_serial_columns_single_launch_enabled() &&
                    (has_fusion || (!n1_dp16_q8_dot4 && !n1_rdna3_q4k_dot4));
            if (use_single_launch_serial_columns) {
                if (ggml_cuda_mtp_mmvq_serial_columns_log_enabled()) {
                    GGML_LOG_INFO("%s: mtp_weight_route route=%s tensor=%s status=selected ncols_x=%lld nrows_x=%lld ncols_dst=%lld ids=%d fusion=%d\n",
                            __func__, has_fusion ? "mmvq_serial_columns_fused_single_launch" : "mmvq_serial_columns_single_launch",
                            src0->name, (long long) ne00, (long long) ne01, (long long) ncols_dst,
                            has_ids ? 1 : 0, has_fusion ? 1 : 0);
                }
                if (has_fusion) {
                    mul_mat_vec_q_serial_columns_fused_switch_type(
                        src0->data, src0->type, src1_q8_1_d, ids_d, fusion_local, dst_d, ne00,
                        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
                        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                        ne03,              ne3,           s03, s13,              s3, ids_stride, stream);
                } else {
                    mul_mat_vec_q_serial_columns_switch_type(
                        src0->data, src0->type, src1_q8_1_d, ids_d, dst_d, ne00,
                        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
                        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                        ne03,              ne3,           s03, s13,              s3, ids_stride, stream);
                }
                ggml_cuda_mtp_mmvq_route_census_record(
                        has_fusion ? "mmvq_serial_columns_fused_single_launch" : "mmvq_serial_columns_single_launch",
                        src0, ne00, ne01, ncols_dst, has_ids, has_fusion);
                return;
            }
            prepare_q8sum4(1);
            for (int64_t c = 0; c < ncols_dst; ++c) {
                const char * src1_q8_1_col = src1_q8_1_d + (size_t) c * (size_t) stride_col_y * sizeof(block_q8_1);
                const int32_t * src1_q8sum4_col = src1_q8sum4_d ? src1_q8sum4_d + 4 * c * stride_col_y : nullptr;
                const int32_t * ids_col = ids_d ? ids_d + c * ids_stride : nullptr;
                float * dst_col = dst_d + c * stride_col_dst;
                mul_mat_vec_q_switch_type(
                    src0->data, src0->type, src1_q8_1_col, ids_col, fusion_local, dst_col, ne00,
                    ne01,              1,                 s01, stride_col_y,     stride_col_dst,
                    ne02, nchannels_y, nchannels_dst,     s02, stride_channel_y, stride_channel_dst,
                    ne03,              ne3,               s03, s13,              s3,               src1_q8sum4_col, ids_stride, stream);
            }
            return;
        }
        const bool should_prepare_mtp_x_bias_packed16 = fusion_x_bias_only &&
            (ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(src0) ||
             ggml_cuda_mtp_q8_dot4_mmvq_packed_down_tensor_allowed(src0) ||
             ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_tensor_allowed(src0));
        const bool should_prepare_packed16 = ggml_cuda_dp16_mmvq_packed16_dot4_env_enabled() ||
            ggml_cuda_dp16_mmvq_q4_0_packed16_dot4_env_enabled() ||
            ggml_cuda_dp16_route_require_q4_0_packed16_mmvq() ||
            should_prepare_mtp_x_bias_packed16;
        if (should_prepare_packed16 &&
                (!has_fusion || should_prepare_mtp_x_bias_packed16) && !has_ids &&
                ncols_dst >= 1 && ncols_dst <= dp16_mmvq_packed16_runtime_max_n() &&
                ne00 % 256 == 0) {
            ggml_cuda_dp16_ensure_packed16_weight(src0, stream);
        }
        ggml_cuda_dp16_trace_mmvq_decode_plan(src0, src1, dst, ne00, ne01, ncols_dst, cc, warp_size, has_fusion, fusion_x_bias_only, has_ids);

        if (!has_ids) {
            const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / ne02);
            const uint3 sample_ratio_fd  = init_fastdiv_values(ne3 / ne03);

            const bool profile_glu_candidate = has_fusion && fusion && fusion->gate &&
                ggml_cuda_mtp_q8_dot4_mmvq_packed_glu_tensors_allowed(src0, fusion->gate);
            if (ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
                    profile_glu_candidate,
                    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_GLU,
                    "rocm_mtp_i8_ffn_gate_up_dot4", src0->name,
                    ne00, ne01, ncols_dst, has_fusion, stream,
                    [&]() {
                        return ggml_cuda_dp16_try_launch_mtp_packed_q8_fused_glu(
                                src0, fusion, fusion_local, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                                channel_ratio_fd, sample_ratio_fd,
                                stride_col_y, stride_col_dst, nchannels_dst,
                                stride_channel_y, stride_channel_dst,
                                ne3, s13, s3,
                                cc, warp_size, has_fusion, has_ids, stream);
                    })) {
                return;
            }

            const bool profile_eh_candidate = !has_fusion && ggml_cuda_mtp_q8_dot4_mmvq_packed_eh_tensor_allowed(src0);
            if (ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
                    profile_eh_candidate,
                    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_EH,
                    "rocm_mtp_i8_eh_proj_dot4", src0->name,
                    ne00, ne01, ncols_dst, has_fusion, stream,
                    [&]() {
                        return ggml_cuda_dp16_try_launch_mtp_packed_q8_eh_proj(
                                src0, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                                channel_ratio_fd, sample_ratio_fd,
                                stride_col_y, stride_col_dst, nchannels_dst,
                                stride_channel_y, stride_channel_dst,
                                ne3, s13, s3,
                                cc, warp_size, has_fusion, has_ids, stream);
                    })) {
                return;
            }

            const bool profile_qkv_candidate = ggml_cuda_mtp_q8_dot4_mmvq_packed_qkv_tensor_allowed(src0);
            if (ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
                    profile_qkv_candidate,
                    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_QKV,
                    "rocm_mtp_i8_qkv_proj_dot4", src0->name,
                    ne00, ne01, ncols_dst, has_fusion, stream,
                    [&]() {
                        return ggml_cuda_dp16_try_launch_mtp_packed_q8_qkv_proj(
                                src0, fusion, fusion_local, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                                channel_ratio_fd, sample_ratio_fd,
                                stride_col_y, stride_col_dst, nchannels_dst,
                                stride_channel_y, stride_channel_dst,
                                ne3, s13, s3,
                                cc, warp_size, has_fusion, has_ids, stream);
                    })) {
                return;
            }

            const bool profile_attn_out_candidate = ggml_cuda_mtp_q8_dot4_mmvq_packed_attn_out_tensor_allowed(src0);
            if (ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
                    profile_attn_out_candidate,
                    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_ATTN_OUT,
                    "rocm_mtp_i8_attn_out_dot4", src0->name,
                    ne00, ne01, ncols_dst, has_fusion, stream,
                    [&]() {
                        return ggml_cuda_dp16_try_launch_mtp_packed_q8_attn_out(
                                src0, fusion, fusion_local, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                                channel_ratio_fd, sample_ratio_fd,
                                stride_col_y, stride_col_dst, nchannels_dst,
                                stride_channel_y, stride_channel_dst,
                                ne3, s13, s3,
                                cc, warp_size, has_fusion, has_ids, stream);
                    })) {
                return;
            }

            const bool profile_down_candidate = ggml_cuda_mtp_q8_dot4_mmvq_packed_down_tensor_allowed(src0);
            if (ggml_cuda_mtp_q8_dot4_mmvq_profiled_try_launch(
                    profile_down_candidate,
                    GGML_CUDA_MTP_Q8_DOT4_MMVQ_PROFILE_DOWN,
                    "rocm_mtp_i8_ffn_down_dot4", src0->name,
                    ne00, ne01, ncols_dst, has_fusion, stream,
                    [&]() {
                        return ggml_cuda_dp16_try_launch_mtp_packed_q8_ffn_down(
                                src0, fusion, fusion_local, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                                channel_ratio_fd, sample_ratio_fd,
                                stride_col_y, stride_col_dst, nchannels_dst,
                                stride_channel_y, stride_channel_dst,
                                ne3, s13, s3,
                                cc, warp_size, has_fusion, has_ids, stream);
                    })) {
                return;
            }
            if (ggml_cuda_dp16_try_launch_packed16_mmvq(
                    src0, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                    channel_ratio_fd, sample_ratio_fd,
                    stride_col_y, stride_col_dst, nchannels_dst,
                    stride_channel_y, stride_channel_dst,
                    ne3, s13, s3,
                    cc, warp_size, has_fusion, has_ids, stream)) {
                return;
            }
        }
    }

    const bool final_has_fusion = fusion_local.gate != nullptr || fusion_local.x_bias != nullptr || fusion_local.gate_bias != nullptr;
    const bool final_has_ids = ids_d != nullptr;
    if (!final_has_ids && !final_has_fusion) {
        const int device = ggml_cuda_get_device();
        const int cc = ggml_cuda_info().devices[device].cc;
        const int warp_size = ggml_cuda_info().devices[device].warp_size;
        if (mul_mat_vec_q_q6_K_reuse_weight_try_launch(
                    src0, src0->data, src1_q8_1_d, dst_d, ne00, ne01, ncols_dst,
                    s01, stride_col_y, stride_col_dst,
                    ne02, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                    ne03, ne3, s03, s13, s3, cc, warp_size, stream)) {
            ggml_cuda_mtp_mmvq_route_census_record("mmvq_q6k_reuse_weight", src0, ne00, ne01, ncols_dst, false, false);
            return;
        }
    }

    ggml_cuda_mtp_mmvq_route_census_record("generic_mmvq", src0, ne00, ne01, ncols_dst, final_has_ids, final_has_fusion);

    prepare_q8sum4(ncols_dst);

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1_d, ids_d, fusion_local, dst_d, ne00,
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
    const bool mtp_q8_dot4_mmvq_scope_enabled = ggml_cuda_mtp_q8_dot4_mmvq_tensor_allowed(src0);
    ggml_cuda_dp16_mmvq_mtp_q8_dot4_scope_guard mtp_q8_dot4_mmvq_scope(mtp_q8_dot4_mmvq_scope_enabled, src0->name);
    if (mtp_q8_dot4_mmvq_scope_enabled) {
        ggml_cuda_mtp_q8_dot4_mmvq_log_tensor(src0, src1, dst, ne00, row_diff, src1_ncols, false, false);
    }
    ggml_cuda_dp16_trace_mmvq_decode_plan(src0, src1, dst, ne00, row_diff, src1_ncols, cc, warp_size, false, false, false);

    ggml_cuda_mm_fusion_args_device fusion_local{};
    ggml_cuda_mtp_mmvq_route_census_record("generic_mmvq_split", src0, ne00, row_diff, src1_ncols, false, false);
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nullptr, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
