#pragma once

#include "common.cuh"

struct ggml_backend_cuda_context;
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_op_pack_v4_k16d16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_op_pack_v4_k16d16_144(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, struct ggml_tensor ** payload, struct ggml_tensor ** scales);
void llama_kv_cache_get_packed16_shadow_k(const void * k_view_data, struct ggml_tensor ** shadow_k);
}

#ifdef GGML_USE_HIP

// The old q8k DOT4 attention route was a lab/probe path and is intentionally
// demoted. Keep these helpers as disabled stubs so legacy selection code and
// route-contract checks compile, while packed16 K packing/registry remains live.
static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(ggml_fattn_instruction inst) {
    GGML_UNUSED(inst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_allow_source_q4_0() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_legal_kv(const ggml_tensor * K, const ggml_tensor * V) {
    GGML_UNUSED(K);
    GGML_UNUSED(V);
    return false;
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(ggml_fattn_instruction inst) {
    GGML_UNUSED(inst);
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_allow_source_q4_0() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_legal_kv(const ggml_tensor * K, const ggml_tensor * V) {
    GGML_UNUSED(K);
    GGML_UNUSED(V);
    return false;
}

inline void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("q8k DOT4 attention route has been removed");
}

#endif // GGML_USE_HIP
