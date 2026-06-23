#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cuda_mtp_mmq_census_entry {
    std::string route;
    std::string tensor;
    std::string dst;
    std::string type;
    int64_t ncols_x = 0;
    int64_t nrows_x = 0;
    int64_t ncols_dst = 0;
    int64_t ncols_max = 0;
    int mmq_x = 0;
    int mmq_y = 0;
    bool has_ids = false;
    bool stream_k = false;
    bool need_check = false;
    bool experimental = false;
    size_t shared_bytes = 0;
    uint64_t calls = 0;
    uint64_t approx_outputs = 0;
    uint64_t timing_calls = 0;
    double total_ms = 0.0;
    float max_ms = 0.0f;
};

static bool ggml_cuda_mtp_mmq_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
}

bool ggml_cuda_mtp_mmq_timing_enabled() {
    static const bool enabled = ggml_cuda_mtp_mmq_env_enabled("LLAMA_MTP_MMQ_TIMING");
    return enabled;
}

bool ggml_cuda_mtp_mmq_route_census_enabled() {
    static const bool enabled = ggml_cuda_mtp_mmq_env_enabled("LLAMA_MTP_MMQ_ROUTE_CENSUS") || ggml_cuda_mtp_mmq_timing_enabled();
    return enabled;
}

bool ggml_cuda_mtp_mmq_stream_is_capturing(cudaStream_t stream) {
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

static std::mutex & ggml_cuda_mtp_mmq_census_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::string, ggml_cuda_mtp_mmq_census_entry> & ggml_cuda_mtp_mmq_census_map() {
    static auto * map = new std::unordered_map<std::string, ggml_cuda_mtp_mmq_census_entry>();
    return *map;
}

static std::string ggml_cuda_mtp_mmq_census_key(const mmq_args & args, ggml_type type, const int mmq_x) {
    const char * route = args.route_name && args.route_name[0] ? args.route_name : (args.ids_dst ? "id_mmq" : "direct_mmq");
    const char * tensor = args.tensor_name && args.tensor_name[0] ? args.tensor_name : "-";
    return std::string(route) + "|" + tensor + "|" + ggml_type_name(type) + "|" +
        std::to_string((long long) args.ncols_dst) + "|" + std::to_string((long long) args.ncols_x) + "|" +
        std::to_string((long long) args.nrows_x) + "|" + std::to_string((long long) args.ncols_max) + "|" +
        std::to_string(mmq_x) + "|" + (args.ids_dst ? "ids" : "noids") + "|" + (args.use_stream_k ? "streamk" : "nostreamk");
}

static ggml_cuda_mtp_mmq_census_entry & ggml_cuda_mtp_mmq_census_entry_for(const mmq_args & args, ggml_type type, const int mmq_x) {
    auto & entry = ggml_cuda_mtp_mmq_census_map()[ggml_cuda_mtp_mmq_census_key(args, type, mmq_x)];
    if (entry.calls == 0 && entry.timing_calls == 0) {
        entry.route = args.route_name && args.route_name[0] ? args.route_name : (args.ids_dst ? "id_mmq" : "direct_mmq");
        entry.tensor = args.tensor_name && args.tensor_name[0] ? args.tensor_name : "-";
        entry.dst = args.dst_name && args.dst_name[0] ? args.dst_name : "-";
        entry.type = ggml_type_name(type);
        entry.ncols_x = args.ncols_x;
        entry.nrows_x = args.nrows_x;
        entry.ncols_dst = args.ncols_dst;
        entry.ncols_max = args.ncols_max;
        entry.mmq_x = mmq_x;
        entry.has_ids = args.ids_dst != nullptr;
        entry.stream_k = args.use_stream_k;
    }
    return entry;
}

static void ggml_cuda_mtp_mmq_census_dump() {
    if (!ggml_cuda_mtp_mmq_route_census_enabled()) {
        return;
    }

    std::vector<ggml_cuda_mtp_mmq_census_entry> rows;
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_mtp_mmq_census_mutex());
        rows.reserve(ggml_cuda_mtp_mmq_census_map().size());
        for (const auto & kv : ggml_cuda_mtp_mmq_census_map()) {
            rows.push_back(kv.second);
        }
    }

    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        if (a.total_ms != b.total_ms) {
            return a.total_ms > b.total_ms;
        }
        if (a.approx_outputs != b.approx_outputs) {
            return a.approx_outputs > b.approx_outputs;
        }
        return a.calls > b.calls;
    });

    GGML_LOG_INFO("mtp_mmq_census summary count=%zu timing=%d\n", rows.size(), ggml_cuda_mtp_mmq_timing_enabled() ? 1 : 0);
    size_t dump_limit = 512;
    if (const char * env_limit = getenv("LLAMA_MTP_MMQ_CENSUS_LIMIT")) {
        const long parsed = std::strtol(env_limit, nullptr, 10);
        if (parsed > 0) {
            dump_limit = (size_t) parsed;
        }
    }
    const size_t limit = std::min<size_t>(rows.size(), dump_limit);
    for (size_t i = 0; i < limit; ++i) {
        const auto & r = rows[i];
        const double avg_ms = r.timing_calls > 0 ? r.total_ms / (double) r.timing_calls : 0.0;
        GGML_LOG_INFO("mtp_mmq_census rank=%zu route=%s tensor=%s dst=%s type=%s mmq_x=%d mmq_y=%d ncols_dst=%lld ncols_max=%lld ncols_x=%lld nrows_x=%lld ids=%d stream_k=%d need_check=%d experimental=%d shared=%zu calls=%llu approx_outputs=%llu timing_calls=%llu total_ms=%.6f avg_ms=%.6f max_ms=%.6f\n",
                i + 1, r.route.c_str(), r.tensor.c_str(), r.dst.c_str(), r.type.c_str(), r.mmq_x, r.mmq_y,
                (long long) r.ncols_dst, (long long) r.ncols_max, (long long) r.ncols_x, (long long) r.nrows_x,
                r.has_ids ? 1 : 0, r.stream_k ? 1 : 0, r.need_check ? 1 : 0, r.experimental ? 1 : 0, r.shared_bytes,
                (unsigned long long) r.calls, (unsigned long long) r.approx_outputs,
                (unsigned long long) r.timing_calls, r.total_ms, avg_ms, (double) r.max_ms);
    }
}

static void ggml_cuda_mtp_mmq_register_atexit_once() {
    static const bool registered = []() {
        std::atexit(ggml_cuda_mtp_mmq_census_dump);
        return true;
    }();
    GGML_UNUSED(registered);
}

void ggml_cuda_mtp_mmq_census_record(const mmq_args & args, ggml_type type, const int mmq_x, const int mmq_y,
        const bool need_check, const size_t nbytes_shared, const bool use_experimental) {
    if (!ggml_cuda_mtp_mmq_route_census_enabled()) {
        return;
    }
    ggml_cuda_mtp_mmq_register_atexit_once();
    std::lock_guard<std::mutex> lock(ggml_cuda_mtp_mmq_census_mutex());
    auto & entry = ggml_cuda_mtp_mmq_census_entry_for(args, type, mmq_x);
    entry.mmq_y = mmq_y;
    entry.need_check = need_check;
    entry.experimental = use_experimental;
    entry.shared_bytes = nbytes_shared;
    entry.calls++;
    entry.approx_outputs += (uint64_t) std::max<int64_t>(args.nrows_x, 0) * (uint64_t) std::max<int64_t>(args.ncols_dst, 0);
}

void ggml_cuda_mtp_mmq_timing_record(const mmq_args & args, ggml_type type, const int mmq_x, const float elapsed_ms) {
    if (!ggml_cuda_mtp_mmq_timing_enabled()) {
        return;
    }
    ggml_cuda_mtp_mmq_register_atexit_once();
    std::lock_guard<std::mutex> lock(ggml_cuda_mtp_mmq_census_mutex());
    auto & entry = ggml_cuda_mtp_mmq_census_entry_for(args, type, mmq_x);
    entry.timing_calls++;
    entry.total_ms += (double) elapsed_ms;
    if (elapsed_ms > entry.max_ms) {
        entry.max_ms = elapsed_ms;
    }
}

static bool ggml_cuda_mtp_prefill_force_mmq_runtime() {
    static const bool force = []() {
        const char * env = getenv("LLAMA_MTP_PREFILL_FORCE_MMQ");
        if (env == nullptr) {
            return false;
        }
        char * end = nullptr;
        const long val = std::strtol(env, &end, 10);
        if (end == env) {
            GGML_LOG_WARN("LLAMA_MTP_PREFILL_FORCE_MMQ ignored: expected integer, got '%s'\n", env);
            return false;
        }
        const bool enabled = val != 0;
        if (enabled) {
            GGML_LOG_INFO("LLAMA_MTP_PREFILL_FORCE_MMQ: forcing supported quantized matmuls through MMQ\n");
        }
        return enabled;
    }();
    return force;
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
#if !defined(GGML_HIP_MMQ_QWEN35_INSTANCES_ONLY)
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
#endif
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
#if !defined(GGML_HIP_MMQ_QWEN35_INSTANCES_ONLY)
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
#endif
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
#if !defined(GGML_HIP_MMQ_QWEN35_INSTANCES_ONLY)
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
#endif
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
#if !defined(GGML_HIP_MMQ_QWEN35_INSTANCES_ONLY)
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
#endif
        default:
#if defined(GGML_HIP_MMQ_QWEN35_INSTANCES_ONLY)
            GGML_ABORT("MMQ Qwen35 instance whitelist does not include V type %s", ggml_type_name(args.type_x));
#else
            GGML_ABORT("fatal error");
#endif
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

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

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :  // block_fp4_mmq holds 256 values
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1,
            src0->name, dst->name, "direct_mmq"};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12,
        src0->name, dst->name, "id_mmq"};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols,
        src0->name, dst->name, "op_mmq"};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (ggml_cuda_mtp_prefill_force_mmq_runtime()) {
        return true;
    }

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 256;
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
