#pragma once

#include "common.cuh"
#include "fattn-packed16-common.cuh"

#include <cstdlib>
#include <cstring>

struct ggml_backend_cuda_context;
void ggml_cuda_op_pack_k_packed16(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

extern "C" {
void llama_kv_cache_get_packed16_tensors(const void * k_view_data, struct ggml_tensor ** payload, struct ggml_tensor ** scales);
void llama_kv_cache_get_packed16_shadow_k(const void * k_view_data, struct ggml_tensor ** shadow_k);
}

#ifdef GGML_USE_HIP

// Lab-only ROCm q8K DOT4 KQ/FA route. Keep this behind both the generic unsafe
// gate and an explicit route contract; standalone fused/tile8 probes currently
// underperform stable tiled/GQA FA. See
// docs/rocm-tbq4-paths/23-q8k-dot4-fa-root-cause-and-next-plan.md before
// changing these gates or promoting variants.
// Route-contract family names:
//   rocm_q8k_dot4_kq                          — broad/backcompat
//   rocm_q8k_dot4_recthist_mtp_verify         — MTP verify recthist-v4
//   rocm_q8k_dot4_decode_mtp_draft            — MTP draft decode (any)
//   rocm_q8k_dot4_decode_bn64_mtp_draft       — BN64 decode
//   rocm_q8k_dot4_decode_splitk_mtp_draft     — split-K decode
//   rocm_q8k_dot4_qtile4_gqa6_kvshared        — qtile4/GQA6 K+V shared full FA
//   rocm_packed16_qtile4_gqa6_kvshared        — packed16 sidecar alias for qtile4/GQA6 K+V shared full FA
// Env-gated auto policy (nq>=4, GQA6, I32 K, q4_0 V only):
//   GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_AUTO=1
//   GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_MIN_NQ=4 (default; use 2 only for isolation)
static inline bool ggml_cuda_q8k_dot4_kq_route_required() {
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (!required || required[0] == '\0' || strcmp(required, "any") == 0) {
        const char * disable_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ_AUTO_DISABLE");
        return !(disable_env && atoi(disable_env) != 0);
    }
    // Broad contract plus all precise family names.
    return strcmp(required, "rocm_q8k_dot4_kq") == 0
        || strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0
        || strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0
        || strcmp(required, "rocm_q8k_dot4_qtile4_gqa6_kvshared") == 0
        || strcmp(required, "rocm_packed16_qtile4_gqa6_kvshared") == 0
        || strcmp(required, "qtile4_gqa6_kvshared") == 0;
}

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    // Default-enabled. Explicitly disabled by EXPERIMENTAL_UNSAFE=0 or Q8K_DOT4_KQ=0.
    {
        const char * v = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
        if (!v) v = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
        if (v && atoi(v) == 0) return false;
    }
    {
        const char * v = getenv("GGML_CUDA_ROCM_Q8K_DOT4_KQ");
        if (v && atoi(v) == 0) return false;
    }
    return ggml_cuda_q8k_dot4_kq_route_required();
}

static inline bool ggml_cuda_q8k_dot4_kq_route_is_decode_contract(const char * required) {
    return required && (
        strcmp(required, "rocm_mtp_draft_dot4_decode") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_bn64") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_splitk") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0 ||
        strcmp(required, "dot4_decode") == 0);
}

static inline bool ggml_cuda_q8k_dot4_kq_route_is_verify_contract(const char * required) {
    return required && (
        strcmp(required, "rocm_mtp_verify_dot4_recthist") == 0 ||
        strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0 ||
        strcmp(required, "dot4_recthist") == 0);
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!ggml_cuda_q8k_dot4_kq_enabled()) {
        return false;
    }
    const int32_t fa_inst = ((const int32_t *)dst->op_params)[4];
    const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (ggml_cuda_q8k_dot4_kq_route_is_decode_contract(required_route) &&
            !(fa_inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK && Q->ne[1] == 1)) {
        return false;
    }
    if (ggml_cuda_q8k_dot4_kq_route_is_verify_contract(required_route) &&
            !(fa_inst == GGML_FATTN_INST_MTP_VERIFY_QK && Q->ne[1] > 1)) {
        return false;
    }

    if (!GGML_CUDA_CC_IS_RDNA3(cc) || Q->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }
    // rocm_q8k_dot4_kq is the true GGML_TYPE_Q8_0/block32 K reader.
    // Packed16-q8 side-channel K is physical I32 payload + F16 scales and
    // must enter through packed16-specific route contracts.
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (K->ne[0] != Q->ne[0] || K->ne[0] != 256 || V->ne[0] != 256 || dst->ne[0] != 256) {
        return false;
    }
    if (Q->ne[1] <= 2) {
        return false;
    }
    if (K->ne[1] < Q->ne[1] || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if (V->ne[1] < K->ne[1] || V->ne[2] != K->ne[2] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    return true;
}

// ── Instruction-aware DOT4 gate helpers ──────────────────────────

static inline bool ggml_cuda_q8k_dot4_kq_env_enabled() {
    return ggml_cuda_q8k_dot4_kq_enabled();
}

static inline bool ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(
        ggml_fattn_instruction inst) {
    // Route-contract env: if set, DOT4 must be explicitly required.
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (!required) {
        return true;
    }

    // Broad routes accepted by all DOT4 instruction paths.
    if (strcmp(required, "rocm_q8k_dot4_kq") == 0 ||
        strcmp(required, "rocm_q8k_dot4_qtile4_gqa6_kvshared") == 0 ||
        strcmp(required, "rocm_packed16_qtile4_gqa6_kvshared") == 0 ||
        strcmp(required, "qtile4_gqa6_kvshared") == 0) {
        return true;
    }
    if (ggml_cuda_q8k_dot4_kq_route_is_verify_contract(required) &&
        inst == GGML_FATTN_INST_MTP_VERIFY_QK) {
        return true;
    }
    if (ggml_cuda_q8k_dot4_kq_route_is_decode_contract(required) &&
        inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        return true;
    }

    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_allow_source_q4_0() {
    // Q4_0 source K materialization to packed16 is experimental.
    const char * env = getenv("GGML_CUDA_ROCM_DOT4_SOURCE_Q4_0_PACK16");
    return env && atoi(env) != 0;
}

static inline bool ggml_cuda_q8k_dot4_kq_legal_kv(
        const ggml_tensor * K, const ggml_tensor * V) {
    // Legal K: persistent packed16 I32 (handled upstream), q8_0,
    // or source f16 (materialized op-locally). Optionally q4_0.
    const bool k_ok = K->type == GGML_TYPE_F16 ||
                      K->type == GGML_TYPE_Q8_0 ||
                      (K->type == GGML_TYPE_Q4_0 && ggml_cuda_q8k_dot4_kq_allow_source_q4_0());

    const bool v_ok = V->type == GGML_TYPE_F16 ||
                      V->type == GGML_TYPE_Q8_0 ||
                      V->type == GGML_TYPE_Q4_0;

    return k_ok && v_ok;
}

void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else

static inline bool ggml_cuda_q8k_dot4_kq_enabled() {
    return false;
}

static inline bool ggml_cuda_q8k_dot4_kq_supported(const int cc, const ggml_tensor * dst) {
    GGML_UNUSED(cc); GGML_UNUSED(dst);
    return false;
}

inline void ggml_cuda_flash_attn_ext_q8k_dot4_kq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("q8k_dot4_kq requires HIP");
}

#endif // GGML_USE_HIP
