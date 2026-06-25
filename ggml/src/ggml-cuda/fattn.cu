#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-dot4-q8k-kq.cuh"
#include "fattn-packed16-wmma-tile.cuh"
#include "fattn-packed16-dot4-mmq.cuh"
#include "dot4-packed16/dp16-plan.cuh"
#include "tbq4-cuda.cuh"
#include "fattn.cuh"

#include <cstdlib>
#include <cstring>

void ggml_cuda_tbq4_innerq_fattn_set_scale(const float * scale, cudaStream_t stream) {
    tbq4_innerq_upload_scale(scale, stream);
    tbq4_innerq_activate(stream);
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

#ifdef GGML_HIP_QWEN35_ATTENTION_INSTANCES_ONLY
    GGML_UNUSED(ctx);
    GGML_UNUSED(cc);
    GGML_UNUSED(KQV);
    GGML_UNUSED(Q);
    GGML_UNUSED(K);
    GGML_UNUSED(V);
    GGML_UNUSED(mask);
    GGML_ABORT("MMA_F16 is disabled in GGML_HIP_QWEN35_ATTENTION_INSTANCES_ONLY; RDNA3 uses VEC/tile/PWMMA/PDMQ paths");
#endif

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0,      GGML_TYPE_Q4_0)
#if !defined(GGML_USE_HIP) || defined(GGML_HIP_COMPILE_STANDARD_Q8_FATTN)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_Q4_0)
#endif
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16,      GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

#ifdef GGML_USE_HIP
    // Persistent packed16 K sidecar: K is an I32 payload view and the matching
    // f16 scales sidecar is passed through the VEC sinks pointer override.
    FATTN_VEC_CASE(256, GGML_TYPE_I32, GGML_TYPE_Q4_0)
#endif

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
    BEST_FATTN_KERNEL_Q8K_DOT4_KQ = 583, // experimental ROCm KQ-only packed16 DOT4 probe
    BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC = 584, // experimental ROCm packed16 q8_0-K shadow inside stable VEC FA
    BEST_FATTN_KERNEL_PACKED16_WMMA_TILE = 585, // packed16 K cache → f16 tile materializer → WMMA/MFMA FA
    BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ = 586, // packed16 K cache → q8 Q tile → sudot4/MMQ FA
    BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC = 587, // MTP decode: exact/source f16 K + q4_0 V through VEC FA
    BEST_FATTN_KERNEL_PACKED16_DECODE = 588, // packed16 I32 K + q4_0 V nq==1 MTP decode lane
    BEST_FATTN_KERNEL_PACKED16_FA2_VEC = 589, // packed16 I32 K sidecar + q4_0 V through VEC/FA2-style online FA
};

static const char * ggml_cuda_fattn_instruction_name(const ggml_fattn_instruction inst);

static const char * ggml_cuda_fattn_kernel_name(const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_NONE:               return "none";
        case BEST_FATTN_KERNEL_TILE:               return "tile";
        case BEST_FATTN_KERNEL_VEC:                return "vec";
        case BEST_FATTN_KERNEL_MMA_F16:            return "mma_f16";
        case BEST_FATTN_KERNEL_Q8K_DOT4_KQ:        return "rocm_q8k_dot4_kq";
        case BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC:return "rocm_q8k_dot4_packed16_vec";
        case BEST_FATTN_KERNEL_PACKED16_WMMA_TILE:    return "rocm_packed16_wmma_tile";
        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:       return "rocm_packed16_dot4_mmq";
        case BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC:        return "rocm_mtp_f16k_q4v_decode";
        case BEST_FATTN_KERNEL_PACKED16_DECODE:         return "rocm_packed16_decode";
        case BEST_FATTN_KERNEL_PACKED16_FA2_VEC:        return "rocm_packed16_fa2_vec";
    }
    return "unknown";
}

static int32_t ggml_cuda_fattn_get_instruction(const ggml_tensor * dst) {
    return ((const int32_t *)dst->op_params)[4];
}

static const char * ggml_cuda_fattn_node_name(const ggml_tensor * dst) {
    return dst && dst->name[0] ? dst->name : "-";
}

static int ggml_cuda_fattn_layer_from_node_name(const ggml_tensor * dst) {
    const char * name = ggml_cuda_fattn_node_name(dst);
    constexpr const char * prefix = "__fattn__-";
    constexpr size_t prefix_len = sizeof("__fattn__-") - 1;
    if (strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long layer = strtol(name + prefix_len, &end, 10);
    return end && end != name + prefix_len ? (int) layer : -1;
}

enum ggml_cuda_fattn_k_physical_repr {
    GGML_CUDA_FATTN_K_PHYS_UNKNOWN = 0,
    GGML_CUDA_FATTN_K_PHYS_F16_EXACT,
    GGML_CUDA_FATTN_K_PHYS_Q8_0_BLOCK32,
    GGML_CUDA_FATTN_K_PHYS_PDMQ_I32_SIDECAR,
};

static ggml_cuda_fattn_k_physical_repr ggml_cuda_fattn_k_physical_repr_from_tensor(const ggml_tensor * K) {
    if (!K) {
        return GGML_CUDA_FATTN_K_PHYS_UNKNOWN;
    }
    if (K->type == GGML_TYPE_F16) {
        return GGML_CUDA_FATTN_K_PHYS_F16_EXACT;
    }
    if (K->type == GGML_TYPE_Q8_0) {
        return GGML_CUDA_FATTN_K_PHYS_Q8_0_BLOCK32;
    }
    if (K->type == GGML_TYPE_I32) {
        return GGML_CUDA_FATTN_K_PHYS_PDMQ_I32_SIDECAR;
    }
    return GGML_CUDA_FATTN_K_PHYS_UNKNOWN;
}

static const char * ggml_cuda_fattn_k_physical_repr_name(const ggml_tensor * K) {
    switch (ggml_cuda_fattn_k_physical_repr_from_tensor(K)) {
        case GGML_CUDA_FATTN_K_PHYS_F16_EXACT:              return "f16_exact";
        case GGML_CUDA_FATTN_K_PHYS_Q8_0_BLOCK32:           return "q8_0_block32";
        case GGML_CUDA_FATTN_K_PHYS_PDMQ_I32_SIDECAR:      return "pdmq_i32_payload_f16scales";
        case GGML_CUDA_FATTN_K_PHYS_UNKNOWN:
        default:                                            return "unknown";
    }
}

static bool ggml_cuda_v4_144_pv4_standard_enabled() {
#ifdef GGML_USE_HIP
    const char * disable_v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE");
    if (disable_v && atoi(disable_v) != 0) {
        return false;
    }
    const char * legacy_pv4 = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4");
    if (legacy_pv4 && *legacy_pv4) {
        return atoi(legacy_pv4) != 0;
    }
    return true;
#else
    return false;
#endif
}

static bool ggml_cuda_v4_144_dp16_dot_profile_enabled() {
#ifdef GGML_USE_HIP
    const char * disable_v = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4_DISABLE");
    if (disable_v && atoi(disable_v) != 0) {
        return false;
    }
    const char * profile = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PROFILE");
    return profile && *profile && atoi(profile) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_env_enabled_name(const char * name) {
#ifdef GGML_USE_HIP
    const char * env = getenv(name);
    if (env && *env) {
        return atoi(env) != 0;
    }
    if (strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PROFILE") == 0) {
        return ggml_cuda_v4_144_dp16_dot_profile_enabled();
    }
    if (strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PV4") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_DECODE_SPLITK") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_GQA6_WAVEGROUP") == 0 ||
            strcmp(name, "GGML_CUDA_ROCM_V4_K16D16_144_PWMMA_PREFILL") == 0) {
        return ggml_cuda_v4_144_pv4_standard_enabled();
    }
    return false;
#else
    GGML_UNUSED(name);
    return false;
#endif
}

static constexpr bool ggml_cuda_standard_q8_fattn_vec_compiled() {
#if !defined(GGML_USE_HIP) || defined(GGML_CUDA_FA_ALL_QUANTS) || defined(GGML_HIP_COMPILE_STANDARD_Q8_FATTN)
    return true;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_attention_env_active() {
#ifdef GGML_USE_HIP
    return !ggml_cuda_env_enabled_name("LLAMA_MTP_DISABLE_FA") &&
           !ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_MTP_DISABLE_FA");
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_hidden_producer_dot4_enabled() {
#ifdef GGML_USE_HIP
    return !ggml_cuda_env_enabled_name("LLAMA_MTP_DISABLE_PRODUCER_DOT4_FA2") &&
           !ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_MTP_DISABLE_PRODUCER_DOT4");
#else
    return false;
#endif
}

static bool ggml_cuda_fattn_inst_is_mtp_verify_qk(const int32_t inst) {
    return inst == GGML_FATTN_INST_MTP_VERIFY_QK ||
           inst == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK;
}

static bool ggml_cuda_fattn_is_mtp_hidden_state_producer(const ggml_tensor * dst) {
    if (!dst || !ggml_cuda_mtp_attention_env_active()) {
        return false;
    }

    const ggml_fattn_instruction inst = (ggml_fattn_instruction) ggml_cuda_fattn_get_instruction(dst);
    switch (inst) {
        case GGML_FATTN_INST_NONE:
        case GGML_FATTN_INST_PREFILL_QK:
        case GGML_FATTN_INST_MTP_DRAFT:
            return true;
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK:
        case GGML_FATTN_INST_MTP_VERIFY_QK:
        case GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK:
        case GGML_FATTN_INST_DECODE_QK:
        case GGML_FATTN_INST_SPEC_VERIFY_QK:
        case GGML_FATTN_INST_BATCH_VERIFY_QK:
        default:
            return false;
    }
}

static bool ggml_cuda_fattn_mtp_hidden_producer_use_conservative_route(const ggml_tensor * dst) {
    return ggml_cuda_fattn_is_mtp_hidden_state_producer(dst) &&
           !ggml_cuda_mtp_hidden_producer_dot4_enabled();
}

// PR3: f16-source K op-local materialization.
// This route quantizes source K=f16 into a temporary packed16 representation
// inside the DOT4 launch path. It preserves the persistent MTP KV cache, but
// MTP draft-logit A/B showed it changes top-1 badly enough to cause zero
// draft acceptance. Keep it behind an explicit unsafe lab gate until a quality-safe
// MTP attention fastpath exists.
static bool ggml_cuda_mtp_source_f16_dot4_unsafe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_SOURCE_F16_DOT4_UNSAFE");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

// When enabled with the unsafe gate, MTP_VERIFY_QK + source K=f16 will quantize
// K into a temporary packed16 representation inside the DOT4 launch path.
static bool ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER");
    return env && atoi(env) != 0 && ggml_cuda_mtp_source_f16_dot4_unsafe_enabled();
#else
    return false;
#endif
}

// ── MTP_VERIFY_QK WMMA/MMA fallback env gates ──────────────────────
// Default off. Part 3: instruction-aware fallback when DOT4 is disabled
// or illegal. WMMA/MMA are f16/f32 only — no q8/q4 dequant fallback.

static bool ggml_cuda_mtp_verify_mma_f16_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_MMA_F16");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_verify_mma_f16_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_mtp_verify_mma_f16_enabled()) {
        return false;
    }

#ifdef GGML_HIP_QWEN35_ATTENTION_INSTANCES_ONLY
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }

    if (!amd_mfma_available(cc)) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const bool k_ok =
        K->type == GGML_TYPE_F16 ||
        K->type == GGML_TYPE_F32;

    const bool v_ok =
        V->type == GGML_TYPE_F16 ||
        V->type == GGML_TYPE_F32;

    if (!k_ok || !v_ok) {
        return false;
    }

    if (Q->ne[0] == 40 || Q->ne[0] == 72 ||
        Q->ne[0] == 512 || Q->ne[0] == 576) {
        return false;
    }

    if (mask && mask->ne[2] != 1) {
        return false;
    }

    return true;
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif
}

// Separate support check for f16-source materialization.
// Does NOT call q8k_dot4_kq_supported() — that rejects source K=f16.
// The DOT4 launch path already has f16→packed16 quantization.

static bool ggml_cuda_mtp_verify_dot4_nq2_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_verify_f16k_dot4_adapter_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled()) {
        return false;
    }
    if (!ggml_cuda_q8k_dot4_kq_enabled()) {
        return false;
    }
    // nq==1 is decode (not recthist).
    // nq==2 behind explicit env for MTP_VERIFY_QK only.
    // General PREFILL_QK / SPEC_VERIFY_QK / BATCH_VERIFY_QK bypass.
    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    const int nq_min = (ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32) &&
                        !ggml_cuda_mtp_verify_dot4_nq2_enabled()) ? 3 : 2;

    // V must be f16, q8_0, or q4_0 for DOT4 with source-f16 K materialization.
    const bool v_ok = V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0;

    return GGML_CUDA_CC_IS_RDNA3(cc) &&
           Q->type == GGML_TYPE_F32 &&
           K->type == GGML_TYPE_F16 &&
           v_ok &&
           dst->type == GGML_TYPE_F32 &&
           Q->ne[0] == 256 &&
           K->ne[0] == Q->ne[0] &&
           V->ne[0] == Q->ne[0] &&
           Q->ne[1] >= nq_min &&
           K->ne[1] >= Q->ne[1] &&
           Q->ne[2] % K->ne[2] == 0 &&
           V->ne[2] == K->ne[2] &&
           Q->ne[3] == K->ne[3] &&
           V->ne[3] == K->ne[3];
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif
}


static bool ggml_cuda_fattn_route_contract_matches(const char * required, const best_fattn_kernel kernel) {
    if (!required || required[0] == '\0' || strcmp(required, "any") == 0) {
        return true;
    }
    const char * got = ggml_cuda_fattn_kernel_name(kernel);
    if (strcmp(required, got) == 0) {
        return true;
    }
    if ((strcmp(required, "rocm_quant_prefill_f16") == 0 ||
         strcmp(required, "f16_temp") == 0) &&
            kernel == BEST_FATTN_KERNEL_MMA_F16) {
        return true;
    }
    if ((strcmp(required, "rocm_q8k_dot4_kq") == 0 ||
         strcmp(required, DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE) == 0) &&
            kernel == BEST_FATTN_KERNEL_Q8K_DOT4_KQ) {
        return true;
    }
    if ((strcmp(required, DP16_ROUTE_FA2_PACKED16_DOT4_DECODE) == 0 ||
         strcmp(required, "rocm_packed16_decode") == 0 ||
         strcmp(required, "rocm_packed16_decode_scalar") == 0 ||
         strcmp(required, "rocm_packed16_decode_q4pair") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_scalar") == 0 ||
         strcmp(required, "rocm_packed16_decode_waveqk") == 0 ||
         strcmp(required, "rocm_packed16_decode_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_dsplit") == 0 ||
         strcmp(required, "rocm_packed16_decode_logits_debug") == 0 ||
         strcmp(required, "rocm_packed16_decode_waveqk_q4pair") == 0 ||
         strcmp(required, "rocm_packed16_decode_splitk") == 0 ||
         strcmp(required, "rocm_packed16_small_verify") == 0 ||
         strcmp(required, "rocm_packed16_small_verify_splitk") == 0 ||
         strcmp(required, "rocm_packed16_small_verify_batched_splitk") == 0 ||
         strcmp(required, "rocm_packed16_small_verify_fa2") == 0 ||
         strcmp(required, "rocm_packed16_fa2") == 0 ||
         strcmp(required, "rocm_packed16_small_verify_fa4") == 0 ||
         strcmp(required, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
         strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0) &&
            kernel == BEST_FATTN_KERNEL_PACKED16_DECODE) {
        return true;
    }
    if ((strcmp(required, "rocm_mtp_verify_dot4_recthist") == 0 ||
         strcmp(required, "rocm_mtp_draft_dot4_decode") == 0 ||
         strcmp(required, "rocm_mtp_draft_dot4_decode_bn64") == 0 ||
         strcmp(required, "rocm_mtp_draft_dot4_decode_splitk") == 0 ||
         strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0 ||
         strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0 ||
         strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0 ||
         strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0) &&
            kernel == BEST_FATTN_KERNEL_Q8K_DOT4_KQ) {
        return true;
    }
    if ((strcmp(required, "rocm_mtp_f16k_q4v_decode") == 0 ||
         strcmp(required, "rocm_mtp_f16k_q4v_vec") == 0 ||
         strcmp(required, DP16_ROUTE_FA1_VEC_FALLBACK) == 0) &&
            (kernel == BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC || kernel == BEST_FATTN_KERNEL_VEC)) {
        return true;
    }
    if (strcmp(required, "rocm_mtp_verify_mma_f16") == 0 &&
            kernel == BEST_FATTN_KERNEL_MMA_F16) {
        return true;
    }
    if (strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0 && kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) {
        return true;
    }
    if (strcmp(required, "rocm_packed16_fa2_vec") == 0 && kernel == BEST_FATTN_KERNEL_PACKED16_FA2_VEC) {
        return true;
    }
    if (strcmp(required, "rocm_packed16_wmma_tile") == 0 && kernel == BEST_FATTN_KERNEL_PACKED16_WMMA_TILE) {
        return true;
    }
    if ((strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
         strcmp(required, "packed16_dot4_mmq") == 0) &&
            kernel == BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ) {
        return true;
    }
    // DOT4 is NOT accepted for packed16_wmma_tile in the route contract.
    // nq==1 decode will naturally route to DOT4 (I32 decode path always returns DOT4),
    // and the contract_final check allows nq==1 DOT4. See ggml_cuda_fattn_apply_route_contract.
    return false;
}

static bool ggml_cuda_fattn_route_contract_is_f16_temp(const char * required) {
    return required && (strcmp(required, "rocm_quant_prefill_f16") == 0 ||
        strcmp(required, "f16_temp") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_mtp_f16k_q4v_vec(const char * required) {
    return required && (strcmp(required, "rocm_mtp_f16k_q4v_decode") == 0 ||
        strcmp(required, "rocm_mtp_f16k_q4v_vec") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_q8_mtp_decode(const char * required) {
    return required && (
        strcmp(required, "rocm_mtp_draft_dot4_decode") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_bn64") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_splitk") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0 ||
        strcmp(required, "dot4_decode") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_q8_mtp_verify(const char * required) {
    return required && (
        strcmp(required, "rocm_mtp_verify_dot4_recthist") == 0 ||
        strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0 ||
        strcmp(required, "dot4_recthist") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_packed16_small_verify(const char * required) {
    return required && (
        strcmp(required, "rocm_packed16_small_verify") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_batched_splitk") == 0 ||
        strcmp(required, "rocm_packed16_fa2") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa2") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4_pvwmma") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_i8(const char * required) {
    return required && (strcmp(required, "rocm_q8k_dot4_kq") == 0 ||
        strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0 ||
        strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
        strcmp(required, "packed16_dot4_mmq") == 0 ||
        strcmp(required, "rocm_packed16_decode") == 0 ||
        strcmp(required, "rocm_packed16_decode_scalar") == 0 ||
        strcmp(required, "rocm_packed16_decode_q4pair") == 0 ||
        strcmp(required, "rocm_packed16_decode_gqa_scalar") == 0 ||
        strcmp(required, "rocm_packed16_decode_waveqk") == 0 ||
         strcmp(required, "rocm_packed16_decode_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_dsplit") == 0 ||
         strcmp(required, "rocm_packed16_decode_logits_debug") == 0 ||
        strcmp(required, "rocm_packed16_decode_waveqk_q4pair") == 0 ||
        strcmp(required, "rocm_packed16_decode_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_batched_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa2") == 0 ||
        strcmp(required, "rocm_packed16_fa2") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
}

static bool ggml_cuda_mtp_hotcold_experiment_enabled() {
    const char * env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT");
    return env && env[0] != '\0' && atoi(env) != 0;
}

static bool ggml_cuda_mtp_hotcold_legacy_q4v_enabled() {
    const char * env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_LEGACY_Q4V");
    return env && env[0] != '\0' && atoi(env) != 0;
}

static bool ggml_cuda_mtp_hotcold_v_type_supported(const ggml_tensor * V) {
    if (!V) {
        return false;
    }
    if (V->type == GGML_TYPE_V4_K16D16_144) {
        return true;
    }
    return V->type == GGML_TYPE_Q4_0 && ggml_cuda_mtp_hotcold_legacy_q4v_enabled();
}

static bool ggml_cuda_mtp_hotcold_packed16_decode_contract_applicable(
        const ggml_tensor * dst,
        const bool require_cold_allowed) {
#ifdef GGML_USE_HIP
    if (!dst) {
        return false;
    }
    const char * hotcold_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K");
    if (!hotcold_env || atoi(hotcold_env) == 0) {
        return false;
    }
    if (!ggml_cuda_mtp_hotcold_experiment_enabled()) {
        GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K is deprecated/default-off; set LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT=1 for the old hot/cold experiment");
    }

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    if (fa_inst_i32 != GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V || !K->data) {
        return false;
    }

    const char * window_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW");
    const int64_t hot_window = window_env ? atoll(window_env) : 256;
    if (hot_window <= 0) {
        return false;
    }
    if (const char * cold_unsafe_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE")) {
        if (cold_unsafe_env[0] != '\0' && atoi(cold_unsafe_env) != 0) {
            GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE was deprecated; increase LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW or disable hot/cold K");
        }
    }
    const bool cold_allowed = hot_window >= K->ne[1];
    if (require_cold_allowed && !cold_allowed) {
        return false;
    }

    ggml_tensor * payload = nullptr;
    ggml_tensor * scales  = nullptr;
    llama_kv_cache_get_packed16_tensors(K->data, &payload, &scales);
    return Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        K->type == GGML_TYPE_F16 && ggml_cuda_mtp_hotcold_v_type_supported(V) &&
        Q->ne[0] == 256 && Q->ne[1] == 1 && K->ne[0] == Q->ne[0] && V->ne[0] == Q->ne[0] &&
        K->ne[1] > 0 && Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] &&
        payload != nullptr && scales != nullptr;
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(require_cold_allowed);
    return false;
#endif
}

static bool ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(
        const ggml_tensor * dst,
        const bool require_cold_allowed) {
#ifdef GGML_USE_HIP
    if (!dst) {
        return false;
    }
    const char * hotcold_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K");
    if (!hotcold_env || atoi(hotcold_env) == 0) {
        return false;
    }
    if (!ggml_cuda_mtp_hotcold_experiment_enabled()) {
        GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K is deprecated/default-off; set LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT=1 for the old hot/cold experiment");
    }

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    const bool smallq_verify_inst =
        ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32) ||
        fa_inst_i32 == GGML_FATTN_INST_NONE;
    if (!smallq_verify_inst) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V || !K->data) {
        return false;
    }

    const char * window_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW");
    const int64_t hot_window = window_env ? atoll(window_env) : 256;
    if (hot_window <= 0) {
        return false;
    }
    if (const char * cold_unsafe_env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE")) {
        if (cold_unsafe_env[0] != '\0' && atoi(cold_unsafe_env) != 0) {
            GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE was deprecated; increase LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW or disable hot/cold K");
        }
    }
    const bool cold_allowed = hot_window >= K->ne[1];
    if (require_cold_allowed && !cold_allowed) {
        return false;
    }

    const int max_nq = getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ") ?
        std::max(2, atoi(getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ"))) : 4;

    ggml_tensor * payload = nullptr;
    ggml_tensor * scales  = nullptr;
    llama_kv_cache_get_packed16_tensors(K->data, &payload, &scales);
    return Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        K->type == GGML_TYPE_F16 && ggml_cuda_mtp_hotcold_v_type_supported(V) &&
        Q->ne[0] == 256 && K->ne[0] == Q->ne[0] && V->ne[0] == Q->ne[0] &&
        Q->ne[1] >= 2 && Q->ne[1] <= max_nq && K->ne[1] > 0 &&
        Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] &&
        payload != nullptr && scales != nullptr;
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(require_cold_allowed);
    return false;
#endif
}

static bool ggml_cuda_fattn_route_contract_applicable(
        const char * required,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (ggml_cuda_fattn_route_contract_is_f16_temp(required)) {
        if (!f16_policy) {
            return false;
        }
        if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || Q->ne[1] <= 2) {
            return false;
        }
        return V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0;
    }

    if (ggml_cuda_fattn_route_contract_is_q8_mtp_decode(required)) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        return fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            K->type == GGML_TYPE_Q8_0 &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0) &&
            Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 &&
            Q->ne[1] == 1;
    }

    if (ggml_cuda_fattn_route_contract_is_q8_mtp_verify(required)) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const bool q8_block_kv = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
        return ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32) &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] > 1 &&
            q8_block_kv;
    }

    if (ggml_cuda_fattn_route_contract_is_packed16_small_verify(required)) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const int max_nq = getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ") ?
            std::max(2, atoi(getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ"))) : 4;
        const bool smallq_verify_inst =
            ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32) ||
            fa_inst_i32 == GGML_FATTN_INST_NONE;
        const bool persistent_packed16 =
            K->type == GGML_TYPE_I32 && (K->ne[0] * 4 == Q->ne[0] || K->ne[0] * 8 == Q->ne[0]);
        const bool hotcold_f16_sidecar =
            ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, false);
        return smallq_verify_inst &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_cuda_mtp_hotcold_v_type_supported(V) && Q->ne[0] == 256 && V->ne[0] == 256 &&
            Q->ne[1] >= 2 && Q->ne[1] <= max_nq && K->ne[1] > 0 && K->ne[2] > 0 &&
            Q->ne[2] % K->ne[2] == 0 && Q->ne[3] == K->ne[3] &&
            V->ne[2] == K->ne[2] && V->ne[3] == K->ne[3] &&
            (persistent_packed16 || hotcold_f16_sidecar);
    }

    if (ggml_cuda_fattn_route_contract_is_mtp_f16k_q4v_vec(required) || strcmp(required, DP16_ROUTE_FA1_VEC_FALLBACK) == 0) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        return fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_Q4_0 &&
            Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 &&
            Q->ne[1] == 1;
    }

    if (strcmp(required, DP16_ROUTE_FA2_PACKED16_DOT4_DECODE) == 0 ||
            strcmp(required, "rocm_packed16_decode") == 0 ||
            strcmp(required, "rocm_packed16_decode_scalar") == 0 ||
            strcmp(required, "rocm_packed16_decode_q4pair") == 0 ||
            strcmp(required, "rocm_packed16_decode_gqa_scalar") == 0 ||
            strcmp(required, "rocm_packed16_decode_waveqk") == 0 ||
         strcmp(required, "rocm_packed16_decode_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_dsplit") == 0 ||
         strcmp(required, "rocm_packed16_decode_logits_debug") == 0 ||
            strcmp(required, "rocm_packed16_decode_waveqk_q4pair") == 0 ||
            strcmp(required, "rocm_packed16_decode_splitk") == 0 ||
            strcmp(required, "rocm_packed16_small_verify") == 0 ||
            strcmp(required, "rocm_packed16_small_verify_splitk") == 0 ||
            strcmp(required, "rocm_packed16_small_verify_batched_splitk") == 0 ||
            strcmp(required, "rocm_packed16_small_verify_fa2") == 0 ||
            strcmp(required, "rocm_packed16_fa2") == 0 ||
            strcmp(required, "rocm_packed16_small_verify_fa4") == 0 ||
            strcmp(required, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
            strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const bool persistent_packed16 =
            K->type == GGML_TYPE_I32 && K->ne[0] * 4 == Q->ne[0] &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0 ||
             V->type == GGML_TYPE_V4_K16D16_144);
        const bool hotcold_f16_sidecar =
            ggml_cuda_mtp_hotcold_packed16_decode_contract_applicable(dst, false);
        return fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            Q->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] == 1 &&
            (persistent_packed16 || hotcold_f16_sidecar);
    }

    if (strcmp(required, DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE) == 0) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        return fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK &&
            Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_Q4_0 &&
            Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[1] == 1;
    }

    if (strcmp(required, "rocm_packed16_dot4_mmq") == 0 || strcmp(required, "packed16_dot4_mmq") == 0) {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const bool smallq_verify_inst =
            ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32) ||
            fa_inst_i32 == GGML_FATTN_INST_NONE;
        if (smallq_verify_inst && Q->ne[1] >= 2) {
            const bool persistent_packed16 =
                K->type == GGML_TYPE_I32 && (K->ne[0] * 4 == Q->ne[0] || K->ne[0] * 8 == Q->ne[0]);
            const bool hotcold_f16_sidecar =
                ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, false);
            return Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
                ggml_cuda_mtp_hotcold_v_type_supported(V) && Q->ne[0] == 256 && V->ne[0] == 256 &&
                K->ne[1] > 0 && K->ne[2] > 0 &&
                Q->ne[2] % K->ne[2] == 0 && Q->ne[3] == K->ne[3] &&
                V->ne[2] == K->ne[2] && V->ne[3] == K->ne[3] &&
                (persistent_packed16 || hotcold_f16_sidecar) &&
                ggml_cuda_packed16_dot4_mmq_enabled();
        }
    }

    if (!ggml_cuda_fattn_route_contract_is_i8(required)) {
        return true;
    }

    // I8/dot4 route contracts are prefill-only. During graph reservation and
    // decode the same context also probes one-token attention graphs; those
    // should log as not-applicable instead of aborting a prefill contract.
    // nq==2 is allowed when GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2=1.
    const int nq_min = ggml_cuda_mtp_verify_dot4_nq2_enabled() ? 2 : 3;
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || Q->ne[1] < nq_min) {
        return false;
    }

    // Shape check: Q must be 256, V must be 256. K can be 256 (q8_0) or D/4 (I32 packed16)
    const bool k_shape_ok = (K->type == GGML_TYPE_I32)
        ? (K->ne[0] * 4 == Q->ne[0])
        : (K->ne[0] == 256);
    if (Q->ne[0] != 256 || !k_shape_ok || V->ne[0] != 256 || dst->ne[0] != 256) {
        return false;
    }

    if (strcmp(required, "rocm_q8k_dot4_kq") == 0) {
        // rocm_q8k_dot4_kq is the true GGML_TYPE_Q8_0/block32 route.
        // Persistent packed16-q8 side-channel K is physical I32 + F16 scales
        // and must be required through packed16-specific route names.
        const bool q8_block_kv = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
        const bool f16_adapter =
            K->type == GGML_TYPE_F16 &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0) &&
            ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled() &&
            ggml_cuda_fattn_inst_is_mtp_verify_qk(ggml_cuda_fattn_get_instruction(dst));
        return (q8_block_kv || f16_adapter) && ggml_cuda_q8k_dot4_kq_enabled();
    }
    if (strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0) {
        return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 && ggml_cuda_q8k_dot4_packed16_vec_enabled();
    }
    if (strcmp(required, "rocm_packed16_dot4_mmq") == 0 || strcmp(required, "packed16_dot4_mmq") == 0) {
        return K->type == GGML_TYPE_I32 &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0) &&
            ggml_cuda_packed16_dot4_mmq_enabled();
    }
    return false;
}

static best_fattn_kernel ggml_cuda_fattn_select_rocm_quant_prefill_f16_backend(
        const int cc,
        const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    // Match the existing full f16 FlashAttention selector. This helper is only
    // a selector for the quantized-KV f16-temp route; do not add a stricter
    // shape gate here or GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 silently falls back
    // to the slower q8/tbq4 VEC path for RDNA3 D=256/GQA=6 prefill.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    return BEST_FATTN_KERNEL_NONE;
}

static const char * ggml_cuda_rocm_quant_prefill_f16_policy_reason_name(
        const ggml_cuda_rocm_quant_prefill_f16_policy & policy) {
    if (policy.allowed) {
        return "ok";
    }
    if (!policy.forced && !policy.automatic) {
        return "disabled";
    }
    if (!policy.contiguous_ok) {
        return "noncontig_tbq4";
    }
    if (!policy.under_budget) {
        return "exact_tmp_over_budget";
    }
    return "rejected";
}

static void ggml_cuda_fattn_log_route_contract(
        const char * required,
        const char * status,
        const best_fattn_kernel selected,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy) {
    const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG");
    if (!log_env || atoi(log_env) == 0) {
        return;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const char * reason = f16_policy ? ggml_cuda_rocm_quant_prefill_f16_policy_reason_name(*f16_policy) : "none";
    const double exact_mib  = f16_policy ? (double) f16_policy->tmp_bytes  / (1024.0 * 1024.0) : 0.0;
    const double stable_mib = exact_mib;
    GGML_LOG_INFO("%s: node=%s layer=%d fa_route_contract require=%s status=%s selected=%s f16_allowed=%d reason=%s Q=[%lld,%lld,%lld,%lld] K=%s k_phys=%s V=%s exact=%.2fMiB stable=%.2fMiB\n",
            __func__, ggml_cuda_fattn_node_name(dst), ggml_cuda_fattn_layer_from_node_name(dst),
            required, status, ggml_cuda_fattn_kernel_name(selected),
            f16_policy && f16_policy->allowed ? 1 : 0, reason,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            ggml_type_name(K->type), ggml_cuda_fattn_k_physical_repr_name(K), ggml_type_name(V->type), exact_mib, stable_mib);
}

static best_fattn_kernel ggml_cuda_fattn_apply_route_contract(
        const ggml_tensor * dst,
        const best_fattn_kernel selected,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy) {
#ifdef GGML_USE_HIP
    const char * required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (!required || required[0] == '\0') {
        return selected;
    }
    if (!ggml_cuda_fattn_route_contract_applicable(required, dst, f16_policy)) {
        ggml_cuda_fattn_log_route_contract(required, "not_applicable", selected, dst, f16_policy);
        return selected;
    }
    if (ggml_cuda_fattn_route_contract_matches(required, selected)) {
        // Final guard: packed16_wmma_tile must never accept DOT4 for nq>1.
        if ((strcmp(required, "rocm_packed16_wmma_tile") == 0 ||
             strcmp(required, "packed16_wmma_tile") == 0) &&
            selected != BEST_FATTN_KERNEL_PACKED16_WMMA_TILE) {
            const ggml_tensor * Q = dst->src[0];
            if (Q->ne[1] > 1) {
                GGML_ABORT("route contract violation: rocm_packed16_wmma_tile required but selected %s (nq=%d)",
                    ggml_cuda_fattn_kernel_name(selected), (int)Q->ne[1]);
            }
        }
        ggml_cuda_fattn_log_route_contract(required, "selected", selected, dst, f16_policy);
        return selected;
    }

    ggml_cuda_fattn_log_route_contract(required, "rejected", selected, dst, f16_policy);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const char * reason = f16_policy ? ggml_cuda_rocm_quant_prefill_f16_policy_reason_name(*f16_policy) : "none";
    const double exact_mib  = f16_policy ? (double) f16_policy->tmp_bytes  / (1024.0 * 1024.0) : 0.0;
    const double stable_mib = exact_mib;
    GGML_ABORT("required FA route %s not selected; got %s; applicable=1 f16_reason=%s; Q=[%lld,%lld,%lld,%lld] K=%s V=%s exact_tmp=%.2fMiB stable_tmp=%.2fMiB",
            required, ggml_cuda_fattn_kernel_name(selected), reason,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            ggml_type_name(K->type), ggml_type_name(V->type), exact_mib, stable_mib);
#else
    GGML_UNUSED(dst); GGML_UNUSED(f16_policy);
    return selected;
#endif // GGML_USE_HIP
}

// ── Selector call-site context ────────────────────────────────────
// get_best_fattn_kernel() is called from TWO entry points:
//   1. support probe  → ggml_cuda_flash_attn_ext_supported()
//   2. actual dispatch → ggml_cuda_flash_attn_ext()
// Logs must only fire in dispatch context to avoid false-positives.
enum ggml_cuda_fattn_select_context {
    GGML_CUDA_FATTN_SELECT_SUPPORT_PROBE = 0,
    GGML_CUDA_FATTN_SELECT_DISPATCH      = 1,
};

// File-scope context tag. Set by the caller before invoking
// get_best_fattn_kernel, read by internal log sites.
static ggml_cuda_fattn_select_context g_fattn_select_ctx = GGML_CUDA_FATTN_SELECT_SUPPORT_PROBE;

struct ggml_cuda_fattn_capture_context {
    bool known;
    bool active;
};

static ggml_cuda_fattn_capture_context g_fattn_capture_ctx = { false, false };

static bool ggml_cuda_fattn_capture_active() {
    return g_fattn_capture_ctx.known && g_fattn_capture_ctx.active;
}

static void ggml_cuda_fattn_log_selection(const best_fattn_kernel kernel, const ggml_tensor * dst) {
    if (g_fattn_select_ctx != GGML_CUDA_FATTN_SELECT_DISPATCH) {
        return;
    }

    const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG");
    if (!log_env || atoi(log_env) == 0) {
        return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    const char * fa_inst_name =
        ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32);

    const bool q8k_q4v_vec   = (kernel == BEST_FATTN_KERNEL_VEC || kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    const bool packed16_q4v_fa2_vec = kernel == BEST_FATTN_KERNEL_PACKED16_FA2_VEC && K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0;
    const char * route = ggml_cuda_fattn_kernel_name(kernel);
    if (kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) {
        route = "q8k_dot4_packed16_vec";
    } else if (packed16_q4v_fa2_vec) {
        route = "packed16k_q4v_fa2_vec";
    } else if (kernel == BEST_FATTN_KERNEL_Q8K_DOT4_KQ && K->type == GGML_TYPE_I32) {
        route = DP16_ROUTE_FA2_PACKED16_DOT4_DECODE;
    } else if (kernel == BEST_FATTN_KERNEL_PACKED16_DECODE) {
        route = "rocm_packed16_decode";
    } else if (q8k_q4v_vec) {
        route = "q8k_q4v_vec";
    }

    GGML_LOG_INFO("%s: node=%s layer=%d kernel=%s route=%s fa_inst=%s Q=%s K=%s k_phys=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld\n",
        __func__, ggml_cuda_fattn_node_name(dst), ggml_cuda_fattn_layer_from_node_name(dst),
        ggml_cuda_fattn_kernel_name(kernel), route, fa_inst_name,
        ggml_type_name(Q->type), ggml_type_name(K->type), ggml_cuda_fattn_k_physical_repr_name(K), ggml_type_name(V->type),
        (long long) Q->ne[1], (long long) K->ne[1],
        (long long) Q->ne[0], (long long) V->ne[0]);
}

static void ggml_cuda_fattn_log_rocm_quant_prefill_f16_once(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
        const ggml_cuda_rocm_quant_prefill_f16_policy & policy) {
#ifdef GGML_USE_HIP
    const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG");
    if (!log_env || atoi(log_env) == 0) {
        return;
    }

    static bool logged_forced_allowed = false;
    static bool logged_auto_allowed   = false;
    static bool logged_reject         = false;
    bool & logged = policy.allowed ? (policy.forced ? logged_forced_allowed : logged_auto_allowed) : logged_reject;
    if (logged) {
        return;
    }
    logged = true;

    const char * mode = policy.forced ? "forced" : (policy.automatic ? "auto" : "candidate");
    GGML_LOG_INFO("%s: rocm_quant_prefill_f16=%s allowed=%d reason=%s K=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld "
            "exact=%.2f MiB route_max=%lld MiB contiguous_ok=%d route_under_budget=%d\n",
            __func__, mode, policy.allowed ? 1 : 0,
            ggml_cuda_rocm_quant_prefill_f16_policy_reason_name(policy),
            ggml_type_name(K->type), ggml_type_name(V->type),
            (long long) Q->ne[1], (long long) K->ne[1],
            (long long) Q->ne[0], (long long) V->ne[0],
            (double) policy.tmp_bytes / (1024.0 * 1024.0),
            (long long) policy.max_mib,
            policy.contiguous_ok ? 1 : 0,
            policy.under_budget ? 1 : 0);
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(policy);
#endif // GGML_USE_HIP
}

static bool ggml_cuda_rocm_experimental_unsafe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_UNSAFE_EXPERIMENTS");
    }
    return env && atoi(env) != 0;
#else
    return false;
#endif // GGML_USE_HIP
}

static bool ggml_cuda_fattn_route_log_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("COMPRESSED_KV_FATTN_LOG");
    return env && atoi(env) != 0;
#else
    return false;
#endif // GGML_USE_HIP
}

static bool ggml_cuda_fattn_mixed_kv_supported(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    if (K->type == V->type) {
        return true;
    }
#ifdef GGML_CUDA_FA_ALL_QUANTS
    GGML_UNUSED(Q);
    return true;
#else
    GGML_UNUSED(Q);
    const bool q8_q4_mix = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    return q8_q4_mix;
#endif // GGML_CUDA_FA_ALL_QUANTS
}

static void ggml_cuda_fattn_log_mixed_kv_reject(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    if (!ggml_cuda_fattn_route_log_enabled()) {
        return;
    }

    // This probe can run once per layer/op.  When COMPRESSED_KV_FATTN_LOG is
    // enabled for route proof, printing every identical mixed-KV reject creates
    // huge logs and can dominate short live requests.  Keep the diagnostic value
    // but emit only once per observed shape/type tuple.
    struct reject_key {
        ggml_type q_type;
        ggml_type k_type;
        ggml_type v_type;
        int64_t nq;
        int64_t nkv;
        int64_t d_q;
        int64_t d_v;
    };
    static reject_key printed[64];
    static int n_printed = 0;
    static bool overflow_printed = false;

    const reject_key key = { Q->type, K->type, V->type, Q->ne[1], K->ne[1], Q->ne[0], V->ne[0] };
    for (int i = 0; i < n_printed; ++i) {
        const reject_key & p = printed[i];
        if (p.q_type == key.q_type && p.k_type == key.k_type && p.v_type == key.v_type &&
                p.nq == key.nq && p.nkv == key.nkv && p.d_q == key.d_q && p.d_v == key.d_v) {
            return;
        }
    }
    if (n_printed < (int)(sizeof(printed) / sizeof(printed[0]))) {
        printed[n_printed++] = key;
    } else if (overflow_printed) {
        return;
    } else {
        overflow_printed = true;
    }

    GGML_LOG_INFO("%s: route=none reject=unsupported_mixed_kv Q=%s K=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld%s\n",
        __func__, ggml_type_name(Q->type), ggml_type_name(K->type), ggml_type_name(V->type),
        (long long) Q->ne[1], (long long) K->ne[1],
        (long long) Q->ne[0], (long long) V->ne[0],
        overflow_printed ? " (further unique mixed-KV reject shapes suppressed)" : "");
}

enum ggml_cuda_rocm_quant_prefill_f16_mode {
    GGML_CUDA_ROCM_QUANT_PREFILL_F16_OFF,
    GGML_CUDA_ROCM_QUANT_PREFILL_F16_ALLOW,
    GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER,
    GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE,
};

static ggml_cuda_rocm_quant_prefill_f16_mode ggml_cuda_rocm_quant_prefill_f16_mode_from_env() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_QUANT_PREFILL_MODE");
    if (!env) {
        env = getenv("GGML_CUDA_ROCM_QUANT_PREFILL_F16_MODE");
    }
    if (env) {
        if (strcmp(env, "off") == 0 || strcmp(env, "OFF") == 0) {
            return GGML_CUDA_ROCM_QUANT_PREFILL_F16_OFF;
        }
        if (strcmp(env, "allow") == 0 || strcmp(env, "ALLOW") == 0 ||
                strcmp(env, "allow_f16_temp") == 0 || strcmp(env, "allow-f16-temp") == 0) {
            return GGML_CUDA_ROCM_QUANT_PREFILL_F16_ALLOW;
        }
        if (strcmp(env, "prefer") == 0 || strcmp(env, "PREFER") == 0 ||
                strcmp(env, "prefer_f16_temp") == 0 || strcmp(env, "prefer-f16-temp") == 0) {
            return GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER;
        }
        if (strcmp(env, "require") == 0 || strcmp(env, "REQUIRE") == 0 ||
                strcmp(env, "require_f16_temp") == 0 || strcmp(env, "require-f16-temp") == 0) {
            return GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE;
        }
    }
#endif // GGML_USE_HIP
    return GGML_CUDA_ROCM_QUANT_PREFILL_F16_ALLOW;
}

// ── DOT4 guardrail: K-representation classification ───────────────
// Source f16 "materialized op-locally into packed16" must be tracked
// explicitly — not as loose string logic. Both verify and decode
// selectors use the same K-repr concept, but with different roles.

enum ggml_cuda_dot4_k_repr {
    GGML_CUDA_DOT4_K_REPR_NONE = 0,

    // Persistent K tensor is already packed16/I32.
    GGML_CUDA_DOT4_K_REPR_PERSISTENT_PACKED16_I32,

    // Original q8_0/q4_0 path.
    GGML_CUDA_DOT4_K_REPR_Q8_0,

    // Source f16 K is materialized op-locally into packed16.
    GGML_CUDA_DOT4_K_REPR_SOURCE_F16_OP_LOCAL_PACKED16,
};

static const char * ggml_cuda_dot4_k_repr_name(
        const ggml_cuda_dot4_k_repr repr) {
    switch (repr) {
        case GGML_CUDA_DOT4_K_REPR_NONE:
            return "-";
        case GGML_CUDA_DOT4_K_REPR_PERSISTENT_PACKED16_I32:
            return "persistent_packed16_i32";
        case GGML_CUDA_DOT4_K_REPR_Q8_0:
            return "q8_0";
        case GGML_CUDA_DOT4_K_REPR_SOURCE_F16_OP_LOCAL_PACKED16:
            return "source_f16_op_local_packed16";
    }

    return "unknown";
}

static const char * ggml_cuda_dot4_v_repr_name(const ggml_tensor * V) {
    if (!V) {
        return "-";
    }

    switch (V->type) {
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        default:             return ggml_type_name(V->type);
    }
}

static ggml_cuda_dot4_k_repr ggml_cuda_dot4_resolve_k_repr(
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V) {
    if (!Q || !K || !V) {
        return GGML_CUDA_DOT4_K_REPR_NONE;
    }

    const bool persistent_packed16 =
        K->type == GGML_TYPE_I32 &&
        K->ne[0] * 4 == Q->ne[0];

    if (persistent_packed16) {
        return GGML_CUDA_DOT4_K_REPR_PERSISTENT_PACKED16_I32;
    }

    const bool q8_k =
        K->type == GGML_TYPE_Q8_0 &&
        K->ne[0] == Q->ne[0];

    if (q8_k) {
        return GGML_CUDA_DOT4_K_REPR_Q8_0;
    }

    const bool source_f16_k =
        K->type == GGML_TYPE_F16 &&
        K->ne[0] == Q->ne[0];

    if (source_f16_k) {
        return GGML_CUDA_DOT4_K_REPR_SOURCE_F16_OP_LOCAL_PACKED16;
    }

    return GGML_CUDA_DOT4_K_REPR_NONE;
}

// ── Backend family taxonomy ────────────────────────────────────────
// Names the execution family before routing changes. 'selected=rocm_q8k_dot4_kq'
// is too broad — it can mean recthist, BN64 decode, or split-K decode.
// backend_family makes logs and route contracts readable.

enum ggml_cuda_fattn_backend_family {
    GGML_CUDA_FATTN_BACKEND_EXISTING = 0,

    // DOT4 families
    GGML_CUDA_FATTN_BACKEND_DOT4_RECTHIST_V4,
    GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_BN64,
    GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_SPLITK,
    GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ,
    GGML_CUDA_FATTN_BACKEND_PACKED16_DECODE,

    // Tensor-core / fallback families
    GGML_CUDA_FATTN_BACKEND_MMA_F16,
    GGML_CUDA_FATTN_BACKEND_VEC,
    GGML_CUDA_FATTN_BACKEND_TILE,
};

static const char * ggml_cuda_fattn_backend_family_name(
        const ggml_cuda_fattn_backend_family family) {
    switch (family) {
        case GGML_CUDA_FATTN_BACKEND_EXISTING:           return "existing";
        case GGML_CUDA_FATTN_BACKEND_DOT4_RECTHIST_V4:   return "dot4_recthist_v4";
        case GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_BN64:   return "dot4_decode_bn64";
        case GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_SPLITK: return "dot4_decode_splitk";
        case GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ:   return "packed16_dot4_mmq";
        case GGML_CUDA_FATTN_BACKEND_PACKED16_DECODE:      return "packed16_decode";
        case GGML_CUDA_FATTN_BACKEND_MMA_F16:            return "mma_f16";
        case GGML_CUDA_FATTN_BACKEND_VEC:                return "vec";
        case GGML_CUDA_FATTN_BACKEND_TILE:               return "tile";
    }

    return "unknown";
}

// Maps selected kernel → backend family. Plumbing only, no behavior change.
// Used for logging when existing policy picked WMMA/MMA/VEC/TILE as a
// fallback for MTP instructions.

static ggml_cuda_fattn_backend_family ggml_cuda_fattn_backend_family_from_kernel(
        const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_Q8K_DOT4_KQ:
            return GGML_CUDA_FATTN_BACKEND_DOT4_RECTHIST_V4;

        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:
            return GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ;

        case BEST_FATTN_KERNEL_PACKED16_DECODE:
            return GGML_CUDA_FATTN_BACKEND_PACKED16_DECODE;

        case BEST_FATTN_KERNEL_MMA_F16:
            return GGML_CUDA_FATTN_BACKEND_MMA_F16;

        case BEST_FATTN_KERNEL_VEC:
        case BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC:
            return GGML_CUDA_FATTN_BACKEND_VEC;

        case BEST_FATTN_KERNEL_TILE:
            return GGML_CUDA_FATTN_BACKEND_TILE;

        default:
            return GGML_CUDA_FATTN_BACKEND_EXISTING;
    }
}

// ── Instruction-driven FA scheduler ────────────────────────────────

// Fast-route classification: recthist vs decode.
enum ggml_cuda_fattn_fast_route {
    GGML_CUDA_FAST_ROUTE_NONE = 0,
    GGML_CUDA_FAST_ROUTE_DOT4_RECTHIST,
    GGML_CUDA_FAST_ROUTE_DOT4_DECODE,
};

static const char * ggml_cuda_fattn_fast_route_name(enum ggml_cuda_fattn_fast_route r) {
    switch (r) {
        case GGML_CUDA_FAST_ROUTE_DOT4_RECTHIST: return "dot4_recthist";
        case GGML_CUDA_FAST_ROUTE_DOT4_DECODE:   return "dot4_decode";
        default:                                   return "none";
    }
}

static enum ggml_cuda_fattn_fast_route route_for_instruction(
        ggml_fattn_instruction inst,
        const ggml_tensor * Q) {
    const int64_t nq = Q->ne[1];

    switch (inst) {
        case GGML_FATTN_INST_MTP_VERIFY_QK:
        case GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK:
        case GGML_FATTN_INST_SPEC_VERIFY_QK:
        case GGML_FATTN_INST_BATCH_VERIFY_QK:
        case GGML_FATTN_INST_PREFILL_QK:
            return nq > 1
                ? GGML_CUDA_FAST_ROUTE_DOT4_RECTHIST
                : GGML_CUDA_FAST_ROUTE_NONE;

        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK:
        case GGML_FATTN_INST_DECODE_QK:
            return nq == 1
                ? GGML_CUDA_FAST_ROUTE_DOT4_DECODE
                : GGML_CUDA_FAST_ROUTE_NONE;

        default:
            return GGML_CUDA_FAST_ROUTE_NONE;
    }
}

// ── VEC/TILE fallback hunter ──────────────────────────────────────
// Every FA instruction tries DOT4 first.
// If DOT4 rejects, the reason is logged.
// VEC/TILE fallbacks are recorded as debt with precise rejection causes.

enum ggml_cuda_fa_reject_reason {
    FA_REJECT_NONE = 0,
    FA_REJECT_ENV_DISABLED,
    FA_REJECT_NOT_RDNA3,
    FA_REJECT_NQ_WRONG_FOR_ROUTE,
    FA_REJECT_D_NOT_256,
    FA_REJECT_MISSING_K_REPR,
    FA_REJECT_UNSUPPORTED_V,
    FA_REJECT_MASK_LAYOUT,
    FA_REJECT_ALIBI,
    FA_REJECT_LOGIT_SOFTCAP,
    FA_REJECT_SINKS,
    FA_REJECT_WORKSPACE,
    FA_REJECT_ROUTE_CONTRACT,
};

static const char * ggml_cuda_fa_reject_reason_name(
        const ggml_cuda_fa_reject_reason r) {
    switch (r) {
        case FA_REJECT_NONE:                return "none";
        case FA_REJECT_ENV_DISABLED:        return "env_disabled";
        case FA_REJECT_NOT_RDNA3:           return "not_rdna3";
        case FA_REJECT_NQ_WRONG_FOR_ROUTE:  return "nq_wrong_for_route";
        case FA_REJECT_D_NOT_256:           return "d_not_256";
        case FA_REJECT_MISSING_K_REPR:      return "missing_k_repr";
        case FA_REJECT_UNSUPPORTED_V:       return "unsupported_v";
        case FA_REJECT_MASK_LAYOUT:         return "mask_layout";
        case FA_REJECT_ALIBI:               return "alibi";
        case FA_REJECT_LOGIT_SOFTCAP:       return "logit_softcap";
        case FA_REJECT_SINKS:               return "sinks";
        case FA_REJECT_WORKSPACE:           return "workspace";
        case FA_REJECT_ROUTE_CONTRACT:      return "route_contract";
    }

    return "unknown";
}

// Fast-path result: selection + rejection diagnosis.
struct ggml_cuda_fa_fastpath_result {
    best_fattn_kernel selected;
    const char * route;
    ggml_cuda_fa_reject_reason reject;
};

// ── Env gates ─────────────────────────────────────────────────────

static bool ggml_cuda_fa_hunt_vec_tile_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_FA_HUNT_VEC_TILE");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_fa_no_vec_tile_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_FA_NO_VEC_TILE_FOR_INSTRUCTIONS");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

// ── Accepted fallback classifications ─────────────────────────────

static bool ggml_cuda_fa_reject_is_accepted(ggml_cuda_fa_reject_reason r) {
    // Structural / platform limitations are accepted. Missing
    // adapters or unsupported types are actionable debt.
    switch (r) {
        case FA_REJECT_NOT_RDNA3:
        case FA_REJECT_ALIBI:
        case FA_REJECT_LOGIT_SOFTCAP:
        case FA_REJECT_SINKS:
        case FA_REJECT_MASK_LAYOUT:
            return true;
        default:
            return false;
    }
}

static const char * ggml_cuda_fa_fallback_severity(ggml_cuda_fa_reject_reason r) {
    return ggml_cuda_fa_reject_is_accepted(r) ? "accepted" : "bad";
}

// ── Fallback scoreboard ───────────────────────────────────────────

struct ggml_cuda_fa_fallback_stats {
    uint64_t total_vet_tile;
    uint64_t by_reason[FA_REJECT_ROUTE_CONTRACT + 1];
};

static ggml_cuda_fa_fallback_stats g_fa_fallback_stats;

static void ggml_cuda_fa_stats_record(ggml_cuda_fa_reject_reason r) {
    g_fa_fallback_stats.total_vet_tile++;
    if (r >= 0 && r <= FA_REJECT_ROUTE_CONTRACT) {
        g_fa_fallback_stats.by_reason[r]++;
    }
}

// ── Debt logger ───────────────────────────────────────────────────

static void ggml_cuda_fa_log_vec_tile_debt(
        const ggml_tensor * dst,
        const ggml_fattn_instruction inst,
        const ggml_cuda_fa_fastpath_result fast,
        const best_fattn_kernel fallback) {
    if (!ggml_cuda_fa_hunt_vec_tile_enabled()) {
        return;
    }

    if (fallback != BEST_FATTN_KERNEL_VEC &&
        fallback != BEST_FATTN_KERNEL_TILE) {
        return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_LOG_INFO(
        "fa_vec_tile_debt: inst=%s attempted_route=%s selected=%s "
        "reject=%s severity=%s nq=%lld nk=%lld d=%lld K=%s V=%s\n",
        ggml_cuda_fattn_instruction_name(inst),
        fast.route ? fast.route : "-",
        ggml_cuda_fattn_kernel_name(fallback),
        ggml_cuda_fa_reject_reason_name(fast.reject),
        ggml_cuda_fa_fallback_severity(fast.reject),
        Q ? (long long) Q->ne[1] : -1LL,
        K ? (long long) K->ne[1] : -1LL,
        Q ? (long long) Q->ne[0] : -1LL,
        K ? ggml_type_name(K->type) : "-",
        V ? ggml_type_name(V->type) : "-");

    ggml_cuda_fa_stats_record(fast.reject);
}

// ── Strict abort ──────────────────────────────────────────────────

static void ggml_cuda_fa_abort_on_vec_tile(
        const ggml_tensor * dst,
        const ggml_fattn_instruction inst,
        const ggml_cuda_fa_fastpath_result fast,
        const best_fattn_kernel fallback) {
    if (!ggml_cuda_fa_no_vec_tile_enabled()) {
        return;
    }

    if (fallback != BEST_FATTN_KERNEL_VEC &&
        fallback != BEST_FATTN_KERNEL_TILE) {
        return;
    }

    // Accepted fallbacks do not abort even in strict mode.
    if (ggml_cuda_fa_reject_is_accepted(fast.reject)) {
        return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ABORT(
        "FA instruction fell to %s: inst=%s attempted_route=%s reject=%s "
        "nq=%lld nk=%lld d=%lld K=%s V=%s",
        ggml_cuda_fattn_kernel_name(fallback),
        ggml_cuda_fattn_instruction_name(inst),
        fast.route ? fast.route : "-",
        ggml_cuda_fa_reject_reason_name(fast.reject),
        Q ? (long long) Q->ne[1] : -1LL,
        K ? (long long) K->ne[1] : -1LL,
        Q ? (long long) Q->ne[0] : -1LL,
        K ? ggml_type_name(K->type) : "-",
        V ? ggml_type_name(V->type) : "-");
}

// ── Action mapper: convert reject reason to next patch action ─────

static const char * ggml_cuda_fa_next_action_for_fallback(
        ggml_cuda_fa_reject_reason reason,
        ggml_type k_type,
        ggml_type v_type,
        int64_t d) {
    switch (reason) {
        case FA_REJECT_MISSING_K_REPR:
            if (k_type == GGML_TYPE_F16) {
                return "already_solved_source_f16_op_local_packed16";
            }
            if (k_type == GGML_TYPE_Q4_0) {
                return "add_source_q4_0_to_packed16_materialization";
            }
            if (k_type == GGML_TYPE_Q8_0) {
                return "add_source_q8_0_to_packed16_materialization";
            }
            if (k_type == GGML_TYPE_BF16) {
                return "add_source_bf16_to_packed16_materialization";
            }
            return "unknown_k_repr_adapter";

        case FA_REJECT_UNSUPPORTED_V:
            if (v_type == GGML_TYPE_Q4_0 || v_type == GGML_TYPE_Q8_0) {
                return "add_or_fix_v_loader";
            }
            return "unsupported_v_document_or_adapter";

        case FA_REJECT_MASK_LAYOUT:
            return "add_mask_layout_support_or_keep_fallback";

        case FA_REJECT_ALIBI:
            return "alibi_not_supported_keep_fallback";

        case FA_REJECT_LOGIT_SOFTCAP:
            return "softcap_support_needed_or_keep_fallback";

        case FA_REJECT_D_NOT_256:
            if (d == 128 || d == 64) {
                return "add_d_variant_support";
            }
            return "unsupported_d_keep_fallback";

        case FA_REJECT_WORKSPACE:
            return "fix_workspace_planner";

        default:
            return "no_action";
    }
}

// ── Top-level instruction fast-path dispatcher ────────────────────

// File-scope tracker: the dispatch inside get_best_fattn_kernel populates
// this when an instruction path falls through; the caller in
// ggml_cuda_flash_attn_ext reads it to run hunter hooks.
struct ggml_cuda_fa_inst_tracker {
    bool active;
    ggml_fattn_instruction inst;
    ggml_cuda_fa_fastpath_result fast;
};

static ggml_cuda_fa_inst_tracker g_fa_inst_tracker;

// Forward declarations: called from ggml_cuda_try_instruction_fastpath
// (defined later in this file).
static best_fattn_kernel ggml_cuda_select_mtp_draft_decode_fattn(
        const int cc,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy);

static best_fattn_kernel ggml_cuda_select_mtp_verify_fattn(
        const int cc,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy,
        const ggml_fattn_instruction inst);

static ggml_cuda_fa_fastpath_result ggml_cuda_try_instruction_fastpath(
        const int cc,
        const ggml_tensor * dst,
        const ggml_fattn_instruction inst) {
    const ggml_tensor * Q = dst->src[0];
    const enum ggml_cuda_fattn_fast_route wanted = route_for_instruction(inst, Q);

    switch (wanted) {
        case GGML_CUDA_FAST_ROUTE_DOT4_RECTHIST: {
            // Delegate to the old selector for full gate coverage, then
            // wrap in fastpath_result with a proper reject reason.
            ggml_cuda_rocm_quant_prefill_f16_policy mtp_f16_policy = {};
            const best_fattn_kernel selected =
                ggml_cuda_select_mtp_verify_fattn(cc, dst, &mtp_f16_policy, inst);
            if (selected != BEST_FATTN_KERNEL_NONE) {
                return { selected, "dot4_recthist", FA_REJECT_NONE };
            }
            // Rejected: determine the reason from the old selector's logic.
            // The old selector logs the reason internally; we synthesize one from
            // the same gates it checks.
            if (!ggml_cuda_q8k_dot4_kq_enabled()) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_ENV_DISABLED };
            }
            if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_NOT_RDNA3 };
            }
            if (Q->ne[1] <= 1) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_NQ_WRONG_FOR_ROUTE };
            }
            if (Q->ne[0] != 256) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_D_NOT_256 };
            }
            // For nq==2 on MTP_VERIFY_QK, the old selector requires
            // GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2=1; PREFILL_QK bypasses.
            if (Q->ne[1] == 2 && ggml_cuda_fattn_inst_is_mtp_verify_qk(inst)) {
                const char * nq2_env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_DOT4_NQ2");
                if (!nq2_env || atoi(nq2_env) == 0) {
                    return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_NQ_WRONG_FOR_ROUTE };
                }
            }
            // K/V legality: the old selector has full K/V checks.
            // Synthesize a reason from type checks.
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * V = dst->src[2];
            const bool k_ok = K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 ||
                              K->type == GGML_TYPE_I32;
            const bool v_ok = V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 ||
                              V->type == GGML_TYPE_Q4_0;
            if (!k_ok) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_MISSING_K_REPR };
            }
            if (!v_ok) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_UNSUPPORTED_V };
            }
            return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_ROUTE_CONTRACT };
        }

        case GGML_CUDA_FAST_ROUTE_DOT4_DECODE: {
            ggml_cuda_rocm_quant_prefill_f16_policy mtp_f16_policy = {};
            const best_fattn_kernel selected =
                ggml_cuda_select_mtp_draft_decode_fattn(cc, dst, &mtp_f16_policy);
            if (selected != BEST_FATTN_KERNEL_NONE) {
                return { selected, "dot4_decode", FA_REJECT_NONE };
            }
            if (!ggml_cuda_q8k_dot4_kq_enabled()) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_ENV_DISABLED };
            }
            if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_NOT_RDNA3 };
            }
            if (Q->ne[1] != 1) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_NQ_WRONG_FOR_ROUTE };
            }
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * V = dst->src[2];
            if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_MISSING_K_REPR };
            }
            if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q4_0) {
                return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_UNSUPPORTED_V };
            }
            return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_ROUTE_CONTRACT };
        }

        default:
            return { BEST_FATTN_KERNEL_NONE, "none", FA_REJECT_NONE };
    }
}

// ── Canonical log helper ──────────────────────────────────────────
// Centralized instruction-route logging for all MTP selectors.
// Single format: fa_instruction, dot4_role, backend_family, k_repr, v_repr.

static void ggml_cuda_fattn_log_instruction_route(
        const ggml_tensor * dst,
        const char * fa_instruction,
        const char * dot4_role,
        const ggml_cuda_fattn_backend_family backend_family,
        const ggml_cuda_dot4_k_repr k_repr,
        const char * impl_status,
        const best_fattn_kernel selected) {
#ifdef GGML_USE_HIP
    // Only log in dispatch context; suppress during support probe.
    if (g_fattn_select_ctx != GGML_CUDA_FATTN_SELECT_DISPATCH) {
        return;
    }

    const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG");
    if (!log_env || atoi(log_env) == 0) {
        return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_LOG_INFO(
        "%s: node=%s layer=%d fa_instruction=%s dot4_role=%s backend_family=%s "
        "nq=%lld nk=%lld K=%s V=%s k_repr=%s v_repr=%s "
        "impl_status=%s selected=%s\n",
        __func__,
        ggml_cuda_fattn_node_name(dst),
        ggml_cuda_fattn_layer_from_node_name(dst),
        fa_instruction ? fa_instruction : "-",
        dot4_role ? dot4_role : "-",
        ggml_cuda_fattn_backend_family_name(backend_family),
        Q ? (long long) Q->ne[1] : -1LL,
        K ? (long long) K->ne[1] : -1LL,
        K ? ggml_type_name(K->type) : "-",
        V ? ggml_type_name(V->type) : "-",
        ggml_cuda_dot4_k_repr_name(k_repr),
        ggml_cuda_dot4_v_repr_name(V),
        impl_status ? impl_status : "-",
        ggml_cuda_fattn_kernel_name(selected));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(fa_instruction);
    GGML_UNUSED(dot4_role);
    GGML_UNUSED(backend_family);
    GGML_UNUSED(k_repr);
    GGML_UNUSED(impl_status);
    GGML_UNUSED(selected);
#endif
}

// ── DOT4 guardrail: decode policy object ───────────────────────────
// Centralized BN vs split-K thresholds for MTP draft decode.
// nq==1 → BN64 (nk < threshold) or split-K (nk >= threshold).

struct ggml_cuda_dot4_decode_policy {
    int bn;
    int vsub;
    int64_t splitk_threshold;
    int splitk_size;
    bool splitk_enabled;
    bool bn64_enabled;
};

static ggml_cuda_dot4_decode_policy ggml_cuda_dot4_decode_policy_from_env() {
    ggml_cuda_dot4_decode_policy p = {};
    p.bn = 64;
    p.vsub = 8;
    p.splitk_threshold = 2048;
    p.splitk_size = 512;
    p.splitk_enabled = true;
    p.bn64_enabled = true;

    const char * bn_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_BN");
    const char * vsub_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_VSUB");
    const char * thresh_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_THRESHOLD");
    const char * size_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_SPLITK_SIZE");

    if (bn_env)    p.bn    = atoi(bn_env);
    if (vsub_env)  p.vsub  = atoi(vsub_env);
    if (thresh_env) p.splitk_threshold = atoll(thresh_env);
    if (size_env)  p.splitk_size  = atoi(size_env);
    if (getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_SPLITK")) p.splitk_enabled = false;
    if (getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_BN64"))   p.bn64_enabled   = false;

    return p;
}

// ── DOT4 guardrail: workspace plan ─────────────────────────────────
// Recthist and split-K both need scratch/partials, but in different
// ways. Without a planner, per-branch allocations make memory behavior
// impossible to reason about.

struct ggml_cuda_dot4_workspace_plan {
    bool required;
    size_t k_payload_bytes;
    size_t k_scales_bytes;
    size_t partial_o_bytes;
    size_t partial_m_bytes;
    size_t partial_l_bytes;
    const char * reason;
};

// ── DOT4 guardrail: kill switches ──────────────────────────────────
// Enable flags open gates; kill switches close them fast during
// regression isolation. Keep separate from env gate chain.

static bool ggml_cuda_dot4_mtp_verify_disabled() {
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_DOT4_DISABLE");
    return env && atoi(env) != 0;
}

static bool ggml_cuda_dot4_mtp_draft_decode_disabled() {
    const char * env = getenv("GGML_CUDA_ROCM_MTP_DRAFT_DECODE_DOT4_DISABLE");
    return env && atoi(env) != 0;
}

static bool ggml_cuda_dot4_source_f16_pack_disabled() {
    const char * env = getenv("GGML_CUDA_ROCM_DOT4_SOURCE_F16_PACK16_DISABLE");
    return env && atoi(env) != 0;
}

static bool ggml_cuda_mtp_verify_dot4_disabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_DOT4_DISABLE");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_draft_decode_dot4_disabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_DRAFT_DECODE_DOT4_DISABLE");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

// ── DOT4 guardrail: fallback contract ──────────────────────────────
// Single rule: if DOT4 instruction path is not legal, fall back to
// existing policy UNLESS route contract explicitly requires DOT4.
// No silent fallback without reason logged.
//
// Log format:
//   impl_status=<reason> selected=<existing_policy|route_contract_rejected>
//   rejected features: mask/ALiBi/sinks/logit_softcap/mask_layout

// ── DOT4 guardrail: instruction legality table ─────────────────────
//
// Instruction legality:
//
// MTP_VERIFY_QK:
//   nq == 1  -> decode territory, not recthist
//   nq > 1   -> DOT4 recthist-v4 eligible
//
// MTP_DRAFT_DECODE_QK:
//   nq == 1  -> DOT4 decode BN64 or split-K eligible
//   nq > 1   -> not decode
//
// MTP_DRAFT:
//   existing policy only; no DOT4 preference
//
// NONE:
//   unchanged

// ── DOT4 archetype roles ─────────────────────────────────────────
// The DOT4 archetype defines workload roles:
//
//   Prefill      nq > 1  → v4 recthist
//   MTP verify   nq > 1  → v4 recthist
//   Small decode nq == 1 → BN64 decode
//   Long decode  nq == 1 → split-K
//
// MTP verify is a named DOT4 workload. The role is derived from the
// FA instruction, not from K/V type. K/V type is a legality gate.

enum ggml_cuda_dot4_role {
    GGML_CUDA_DOT4_ROLE_NONE = 0,
    GGML_CUDA_DOT4_ROLE_PREFILL_RECTHIST_V4,
    GGML_CUDA_DOT4_ROLE_RECTHIST_V4_MTP_VERIFY,
    GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT,
    GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT,
};

static const char * ggml_cuda_dot4_role_name(enum ggml_cuda_dot4_role role) {
    switch (role) {
        case GGML_CUDA_DOT4_ROLE_RECTHIST_V4_MTP_VERIFY:  return "recthist_v4_mtp_verify";
        case GGML_CUDA_DOT4_ROLE_PREFILL_RECTHIST_V4:     return "recthist_v4_prefill";
        case GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT:   return "decode_bn64_mtp_draft";
        case GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT: return "decode_splitk_mtp_draft";
        default:                                            return "-";
    }
}

// Forward decl — defined after ggml_cuda_dot4_decode_policy_from_env().
static int64_t ggml_cuda_q8k_dot4_decode_splitk_threshold();

static enum ggml_cuda_dot4_role ggml_cuda_dot4_role_from_instruction(
        int32_t inst,
        const ggml_tensor * Q,
        const ggml_tensor * K) {
    // MTP verify: nq > 1 → recthist-v4.
    if (ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) && Q->ne[1] > 1) {
        return GGML_CUDA_DOT4_ROLE_RECTHIST_V4_MTP_VERIFY;
    }
    // MTP draft decode: nq == 1 → BN64 or split-K based on nk threshold.
    if (inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        if (Q->ne[1] != 1) {
            return GGML_CUDA_DOT4_ROLE_NONE;
        }
        const int64_t threshold = ggml_cuda_q8k_dot4_decode_splitk_threshold();
        return K->ne[1] >= threshold
            ? GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT
            : GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT;
    }
    return GGML_CUDA_DOT4_ROLE_NONE;
}

// ── DOT4 role resolver for MTP_VERIFY_QK ────────────────────────────

static ggml_cuda_dot4_role ggml_cuda_dot4_role_for_mtp_verify(
        const ggml_tensor * Q) {
    if (!Q) {
        return GGML_CUDA_DOT4_ROLE_NONE;
    }

    // nq == 1 is decode territory, not recthist.
    if (Q->ne[1] == 1) {
        return GGML_CUDA_DOT4_ROLE_NONE;
    }

    // nq >= 2 is recthist archetype for MTP_VERIFY_QK.
    return GGML_CUDA_DOT4_ROLE_RECTHIST_V4_MTP_VERIFY;
}

static const char * ggml_cuda_fattn_instruction_name(
        const ggml_fattn_instruction inst) {
    switch (inst) {
        case GGML_FATTN_INST_NONE:                  return "none";
        case GGML_FATTN_INST_MTP_DRAFT:             return "mtp_draft";
        case GGML_FATTN_INST_MTP_VERIFY_QK:         return "mtp_verify_qk";
        case GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK:  return "mtp_qblock_verify_qk";
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK:   return "mtp_draft_decode_qk";
        case GGML_FATTN_INST_PREFILL_QK:            return "prefill_qk";
        case GGML_FATTN_INST_DECODE_QK:             return "decode_qk";
        case GGML_FATTN_INST_SPEC_VERIFY_QK:      return "spec_verify_qk";
        case GGML_FATTN_INST_BATCH_VERIFY_QK:     return "batch_verify_qk";
    }

    return "unknown";
}

static dp16_fa_inst ggml_cuda_dp16_fa_inst_from_instruction(
        const ggml_fattn_instruction inst) {
    switch (inst) {
        case GGML_FATTN_INST_MTP_DRAFT:
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK:
            return DP16_FA_INST_MTP_DRAFT_DECODE;
        case GGML_FATTN_INST_MTP_VERIFY_QK:
        case GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK:
            return DP16_FA_INST_MTP_VERIFY;
        case GGML_FATTN_INST_PREFILL_QK:
            return DP16_FA_INST_PREFILL;
        case GGML_FATTN_INST_SPEC_VERIFY_QK:
        case GGML_FATTN_INST_BATCH_VERIFY_QK:
            return DP16_FA_INST_SHORT_EXTEND;
        case GGML_FATTN_INST_DECODE_QK:
            return DP16_FA_INST_DECODE;
        default:
            return DP16_FA_INST_UNKNOWN;
    }
}

static dp16_layout ggml_cuda_dp16_fa_layout_from_tensor(
        const ggml_tensor * tensor,
        const bool allow_packed16_i32 = false) {
    if (!tensor) {
        return DP16_LAYOUT_UNKNOWN;
    }
    if (allow_packed16_i32 && tensor->type == GGML_TYPE_I32) {
        return DP16_LAYOUT_PACKED16_I32_SCALED;
    }
    return dp16_layout_from_ggml_type(tensor->type);
}

static const char * ggml_cuda_dp16_fa_route_require_name() {
    const char * fa_required = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
    if (fa_required && fa_required[0] != '\0' && strcmp(fa_required, "any") != 0) {
        return fa_required;
    }
    const char * dp16_required = dp16_route_require_env();
    if (dp16_required && dp16_required[0] != '\0') {
        return dp16_required;
    }
    return nullptr;
}

static bool ggml_cuda_dp16_fa_route_required() {
    return ggml_cuda_dp16_fa_route_require_name() != nullptr ||
        dp16_env_enabled("GGML_CUDA_FA_ROUTE_REQUIRE_DOT4");
}

static bool ggml_cuda_mtp_packed16_hotcold_k_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K");
    if (env && atoi(env) != 0 && !ggml_cuda_mtp_hotcold_experiment_enabled()) {
        GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K is deprecated/default-off; set LLAMA_MTP_PACKED16_HOTCOLD_K_EXPERIMENT=1 for the old hot/cold experiment");
    }
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static int64_t ggml_cuda_mtp_packed16_hotcold_k_window() {
#ifdef GGML_USE_HIP
    const char * env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW");
    return env ? atoll(env) : 256;
#else
    return 0;
#endif
}

static bool ggml_cuda_mtp_packed16_hotcold_cold_unsafe_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE");
    if (env && env[0] != '\0' && atoi(env) != 0) {
        GGML_ABORT("LLAMA_MTP_PACKED16_HOTCOLD_K_COLD_UNSAFE was deprecated; increase LLAMA_MTP_PACKED16_HOTCOLD_K_WINDOW or disable hot/cold K");
    }
#endif
    return false;
}

static bool ggml_cuda_fattn_has_packed16_sidecar(
        const ggml_tensor * K,
        ggml_tensor ** payload_out = nullptr,
        ggml_tensor ** scales_out = nullptr) {
#ifdef GGML_USE_HIP
    ggml_tensor * payload = nullptr;
    ggml_tensor * scales  = nullptr;
    if (K && K->data) {
        llama_kv_cache_get_packed16_tensors(K->data, &payload, &scales);
    }
    if (payload_out) {
        *payload_out = payload;
    }
    if (scales_out) {
        *scales_out = scales;
    }
    return payload != nullptr && scales != nullptr;
#else
    GGML_UNUSED(K);
    if (payload_out) {
        *payload_out = nullptr;
    }
    if (scales_out) {
        *scales_out = nullptr;
    }
    return false;
#endif
}

static bool ggml_cuda_mtp_hotcold_packed16_decode_supported(
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_mtp_packed16_hotcold_k_enabled() || !dst) {
        return false;
    }
    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    if (fa_inst_i32 != GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return false;
    }
    const int64_t hot_window = ggml_cuda_mtp_packed16_hotcold_k_window();
    // The post-prefix MTP draft decode path is intentionally mixed: the cold
    // prefix is read from the persistent packed16 sidecar while the newest hot
    // tail is read from the exact F16 shadow K.  Requiring the whole KV bucket
    // to fit inside the hot window made long-prefix decode fall back to the old
    // F16 VEC path even though both representations are available.
    return Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           Q->ne[0] == 256 && Q->ne[1] == 1 &&
           K->type == GGML_TYPE_F16 && K->ne[0] == Q->ne[0] &&
           ggml_cuda_mtp_hotcold_v_type_supported(V) && V->ne[0] == Q->ne[0] &&
           K->ne[1] > 0 && hot_window > 0 &&
           Q->ne[2] % K->ne[2] == 0 &&
           V->ne[2] == K->ne[2] && Q->ne[3] == K->ne[3] &&
           V->ne[3] == K->ne[3] &&
           ggml_cuda_fattn_has_packed16_sidecar(K);
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

static dp16_fa_problem ggml_cuda_make_dp16_fa_problem(
        const ggml_tensor * dst,
        const int cc,
        const bool stream_is_capturing) {
    GGML_UNUSED(cc);

    dp16_fa_problem problem = {};
    if (!dst) {
        return problem;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!Q || !K || !V) {
        return problem;
    }

    const ggml_fattn_instruction inst =
        (ggml_fattn_instruction) ggml_cuda_fattn_get_instruction(dst);
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    problem.inst = ggml_cuda_dp16_fa_inst_from_instruction(inst);
    problem.nq = (int) Q->ne[1];
    problem.nk_bucket = (int) K->ne[1];
    problem.d_head = (int) Q->ne[0];
    problem.n_heads_q = (int) Q->ne[2];
    problem.n_heads_kv = (int) K->ne[2];
    problem.gqa_ratio = (problem.n_heads_kv > 0 && problem.n_heads_q % problem.n_heads_kv == 0)
        ? problem.n_heads_q / problem.n_heads_kv
        : 0;
    problem.batch = (int) Q->ne[3];

    problem.q_type = Q->type;
    problem.k_type = K->type;
    problem.v_type = V->type;

    problem.q_layout = ggml_cuda_dp16_fa_layout_from_tensor(Q);
    problem.k_layout = ggml_cuda_dp16_fa_layout_from_tensor(K, true);
    problem.v_layout = ggml_cuda_dp16_fa_layout_from_tensor(V);

    const int implicit_flags = ggml_get_op_params_i32(dst, 7);
    const bool implicit_causal_mask = (implicit_flags & 1) != 0;
    problem.causal = (mask != nullptr || implicit_causal_mask) && max_bias == 0.0f;
    problem.has_mask = mask != nullptr;
    problem.effective_mask = mask != nullptr || implicit_causal_mask;
    problem.has_sliding_window = false;
    problem.has_sink = sinks != nullptr;

    problem.is_mtp = inst == GGML_FATTN_INST_MTP_DRAFT ||
                     inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK ||
                     ggml_cuda_fattn_inst_is_mtp_verify_qk(inst);
    problem.is_draft_decode = inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK ||
                              inst == GGML_FATTN_INST_DECODE_QK;
    problem.is_verify = ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) ||
                        inst == GGML_FATTN_INST_SPEC_VERIFY_QK ||
                        inst == GGML_FATTN_INST_BATCH_VERIFY_QK;
    problem.is_prefill = inst == GGML_FATTN_INST_PREFILL_QK;
    problem.packed16_k_ready = problem.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED;
    problem.capture = stream_is_capturing;
    problem.route_required = ggml_cuda_dp16_fa_route_required();
    return problem;
}

static best_fattn_kernel ggml_cuda_dp16_fa_plan_to_best_kernel(
        const dp16_fa_problem & problem,
        const dp16_fa_plan & plan,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy) {
    GGML_UNUSED(dst);
    GGML_UNUSED(f16_policy);

    if (!plan.selected) {
        return BEST_FATTN_KERNEL_NONE;
    }

    switch (plan.backend) {
        case DP16_BACKEND_FA2_Q8K_DOT4_DECODE:
        case DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE:
            return BEST_FATTN_KERNEL_NONE;

        case DP16_BACKEND_FA2_PACKED16_DOT4_DECODE:
        case DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY:
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;

        case DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL:
            return BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;

        case DP16_BACKEND_FA1_VEC_FALLBACK:
            if (problem.is_draft_decode &&
                    problem.nq == 1 &&
                    problem.d_head == 256 &&
                    problem.k_type == GGML_TYPE_F16 &&
                    problem.v_type == GGML_TYPE_Q4_0) {
                return BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC;
            }
            return BEST_FATTN_KERNEL_VEC;

        default:
            return BEST_FATTN_KERNEL_NONE;
    }
}

static bool ggml_cuda_dp16_fa_route_is_q8_decode_alias(const char * required) {
    return required && (
        strcmp(required, DP16_ROUTE_FA_Q8K_DOT4_KQ) == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_bn64") == 0 ||
        strcmp(required, "rocm_mtp_draft_dot4_decode_splitk") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0 ||
        strcmp(required, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0 ||
        strcmp(required, "rocm_mtp_verify_dot4_recthist") == 0 ||
        strcmp(required, "rocm_q8k_dot4_recthist_mtp_verify") == 0);
}

static bool ggml_cuda_dp16_fa_route_is_vec_alias(const char * required) {
    return required && (
        strcmp(required, DP16_ROUTE_FA1_VEC_FALLBACK) == 0 ||
        strcmp(required, "rocm_mtp_f16k_q4v_decode") == 0 ||
        strcmp(required, "rocm_mtp_f16k_q4v_vec") == 0);
}

static bool ggml_cuda_dp16_fa_route_is_packed16_decode_alias(const char * required) {
    return required && (
        strcmp(required, DP16_ROUTE_FA2_PACKED16_DOT4_DECODE) == 0 ||
        strcmp(required, "rocm_packed16_decode") == 0 ||
        strcmp(required, "rocm_packed16_decode_scalar") == 0 ||
        strcmp(required, "rocm_packed16_decode_q4pair") == 0 ||
        strcmp(required, "rocm_packed16_decode_gqa_scalar") == 0 ||
        strcmp(required, "rocm_packed16_decode_waveqk") == 0 ||
         strcmp(required, "rocm_packed16_decode_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_pvwmma") == 0 ||
         strcmp(required, "rocm_packed16_decode_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_gqa_wmma_full") == 0 ||
         strcmp(required, "rocm_packed16_decode_dsplit") == 0 ||
         strcmp(required, "rocm_packed16_decode_logits_debug") == 0 ||
        strcmp(required, "rocm_packed16_decode_waveqk_q4pair") == 0 ||
        strcmp(required, "rocm_packed16_decode_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_batched_splitk") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa2") == 0 ||
        strcmp(required, "rocm_packed16_fa2") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4") == 0 ||
        strcmp(required, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
        strcmp(required, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
}

static bool ggml_cuda_dp16_fa_route_is_f16_adapt_alias(const char * required) {
    return required && strcmp(required, DP16_ROUTE_FA2_F16K_ADAPT_DOT4_DECODE) == 0;
}

static bool ggml_cuda_dp16_fa_route_is_pdmq_alias(const char * required) {
    return required && (
        strcmp(required, DP16_ROUTE_FA_PACKED16_MMQ) == 0 ||
        strcmp(required, "packed16_dot4_mmq") == 0);
}

static bool ggml_cuda_dp16_fa_route_is_pwmma_alias(const char * required) {
    return required && (
        strcmp(required, DP16_ROUTE_FA_PACKED16_WMMA_TILE) == 0 ||
        strcmp(required, "packed16_wmma_tile") == 0);
}

static bool ggml_cuda_dp16_fa_route_contract_applicable(
        const char * required,
        const dp16_fa_problem & problem) {
    if (!required || required[0] == '\0' || strcmp(required, "any") == 0) {
        return false;
    }
    if (ggml_cuda_dp16_fa_route_is_q8_decode_alias(required)) {
        return problem.is_mtp && problem.k_layout == DP16_LAYOUT_Q8_BLOCK32 &&
            ((problem.is_draft_decode && problem.nq == 1) ||
             (problem.is_verify && problem.nq > 1));
    }
    if (ggml_cuda_dp16_fa_route_is_packed16_decode_alias(required)) {
        return problem.is_mtp && problem.is_draft_decode && problem.nq == 1 &&
            problem.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED && problem.packed16_k_ready;
    }
    if (ggml_cuda_dp16_fa_route_is_f16_adapt_alias(required)) {
        return problem.is_mtp && problem.is_draft_decode && problem.nq == 1 &&
            problem.k_layout == DP16_LAYOUT_F16 && problem.v_layout == DP16_LAYOUT_Q4_0_BLOCK32;
    }
    if (ggml_cuda_dp16_fa_route_is_vec_alias(required)) {
        return problem.is_mtp && problem.is_draft_decode && problem.nq == 1;
    }
    if (ggml_cuda_dp16_fa_route_is_pdmq_alias(required)) {
        const bool persistent_packed16 = problem.k_layout == DP16_LAYOUT_PACKED16_I32_SCALED && problem.packed16_k_ready;
        const bool f16_sidecar_smallq_shape =
            problem.k_layout == DP16_LAYOUT_F16 && problem.v_layout == DP16_LAYOUT_Q4_0_BLOCK32 &&
            problem.d_head == 256 && problem.nq >= 2 && problem.nq <= 4;
        return problem.is_mtp && problem.is_verify && problem.nq > 1 &&
            (persistent_packed16 || f16_sidecar_smallq_shape);
    }
    if (ggml_cuda_dp16_fa_route_is_pwmma_alias(required)) {
        return problem.is_prefill && problem.nq > 1;
    }
    return false;
}

static bool ggml_cuda_dp16_fa_route_contract_matches_plan(
        const char * required,
        const dp16_fa_problem & problem,
        const dp16_fa_plan & plan) {
    GGML_UNUSED(problem);
    if (!required || required[0] == '\0' || strcmp(required, "any") == 0) {
        return true;
    }
    if (plan.route && strcmp(required, plan.route) == 0) {
        return true;
    }
    if (ggml_cuda_dp16_fa_route_is_q8_decode_alias(required)) {
        return plan.backend == DP16_BACKEND_FA2_Q8K_DOT4_DECODE;
    }
    if (ggml_cuda_dp16_fa_route_is_packed16_decode_alias(required)) {
        return plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE;
    }
    if (ggml_cuda_dp16_fa_route_is_f16_adapt_alias(required)) {
        return plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE;
    }
    if (ggml_cuda_dp16_fa_route_is_pdmq_alias(required)) {
        return plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY;
    }
    if (ggml_cuda_dp16_fa_route_is_pwmma_alias(required)) {
        return plan.backend == DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL;
    }
    if (ggml_cuda_dp16_fa_route_is_vec_alias(required)) {
        return plan.backend == DP16_BACKEND_FA1_VEC_FALLBACK;
    }
    return true;
}

static void ggml_cuda_dp16_fa_emit_plan_trace_if_needed(
        const dp16_fa_problem & problem,
        const dp16_fa_plan & plan) {
    if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
        dp16_trace_emit_fa_plan(problem, plan);
    }
}

static void ggml_cuda_dp16_fa_emit_packed16_mtp_draft_trace_if_needed(
        const ggml_tensor * dst,
        const int cc) {
    if (!dst || g_fattn_select_ctx != GGML_CUDA_FATTN_SELECT_DISPATCH) {
        return;
    }

    const ggml_fattn_instruction inst =
        (ggml_fattn_instruction) ggml_cuda_fattn_get_instruction(dst);
    if (inst != GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        return;
    }

    const dp16_fa_problem problem =
        ggml_cuda_make_dp16_fa_problem(dst, cc, ggml_cuda_fattn_capture_active());
    if (!(problem.is_mtp && problem.is_draft_decode && problem.packed16_k_ready)) {
        return;
    }

    const dp16_fa_plan plan = dp16_plan_mtp_fa(problem);
    ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);
}

static void ggml_cuda_dp16_fa_abort_route_mismatch(
        const char * required,
        const dp16_fa_problem & problem,
        const dp16_fa_plan & plan) {
    GGML_ABORT(
        "required FA route %s not selected by DP16 plan; inst=%s plane=%s backend=%s k_repr=%s capture=%d capture_safe=%d fallback=%d experimental=%d reason=%s nq=%d nk=%d d=%d K=%s V=%s",
        required,
        dp16_fa_inst_name(problem.inst),
        dp16_fa_plane_name(plan.plane),
        dp16_backend_name(plan.backend),
        dp16_k_repr_name(plan.k_repr),
        problem.capture ? 1 : 0,
        plan.capture_safe ? 1 : 0,
        plan.fallback ? 1 : 0,
        plan.experimental ? 1 : 0,
        plan.reason ? plan.reason : "none",
        problem.nq,
        problem.nk_bucket,
        problem.d_head,
        ggml_type_name(problem.k_type),
        ggml_type_name(problem.v_type));
}

static bool ggml_cuda_dp16_fa_plan_owns_packed16_compute(
        const dp16_fa_plan & plan) {
    return plan.selected && !plan.fallback &&
        plan.k_repr == DP16_K_REPR_PACKED16_I32_PERSISTENT &&
        (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE ||
         plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY ||
         plan.backend == DP16_BACKEND_FA2_PACKED16_WMMA_PREFILL);
}

static bool ggml_cuda_fattn_kernel_is_packed16_compute_owner(
        const best_fattn_kernel kernel) {
    return kernel == BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ ||
           kernel == BEST_FATTN_KERNEL_PACKED16_WMMA_TILE ||
           kernel == BEST_FATTN_KERNEL_PACKED16_FA2_VEC ||
           kernel == BEST_FATTN_KERNEL_PACKED16_DECODE;
}

static void ggml_cuda_dp16_fa_enforce_final_packed16_contract(
        const ggml_tensor * dst,
        const best_fattn_kernel selected) {
#ifdef GGML_USE_HIP
    if (g_fattn_select_ctx != GGML_CUDA_FATTN_SELECT_DISPATCH || !dst) {
        return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return;
    }
    if (K->type != GGML_TYPE_I32) {
        return;
    }

    const dp16_fa_problem problem =
        ggml_cuda_make_dp16_fa_problem(dst, 0, ggml_cuda_fattn_capture_active());
    const dp16_fa_plan plan = dp16_plan_mtp_fa(problem);
    if (!ggml_cuda_dp16_fa_plan_owns_packed16_compute(plan)) {
        return;
    }
    ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);
    if (ggml_cuda_fattn_kernel_is_packed16_compute_owner(selected)) {
        return;
    }

    dp16_trace_emit_fa_plan(problem, plan);
    GGML_ABORT(
        "DP16 packed16 compute contract violation: plan route=%s backend=%s k_repr=%s reason=%s but final selected=%s; refusing silent f16/generic fallback for compressed KV (inst=%s nq=%lld nk=%lld d=%lld K=%s k_phys=%s V=%s)",
        plan.route ? plan.route : "none",
        dp16_backend_name(plan.backend),
        dp16_k_repr_name(plan.k_repr),
        plan.reason ? plan.reason : "none",
        ggml_cuda_fattn_kernel_name(selected),
        ggml_cuda_fattn_instruction_name((ggml_fattn_instruction) ggml_cuda_fattn_get_instruction(dst)),
        (long long) Q->ne[1],
        (long long) K->ne[1],
        (long long) Q->ne[0],
        ggml_type_name(K->type),
        ggml_cuda_fattn_k_physical_repr_name(K),
        ggml_type_name(V->type));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(selected);
#endif
}

// ── MTP instruction selector ────────────────────────────────────────
//
// MTP_VERIFY_QK is a semantic FlashAttention instruction.
// It maps to the DOT4 recthist-v4 archetype when legal.
// Legal K representations:
//   1. persistent packed16 I32,
//   2. q8_0/q4_0,
//   3. source f16 materialized op-locally into packed16.
// nq==1 is decode territory, not recthist.
// nq==2 is recthist archetype but env-gated pending validation/default decision.
// MTP_DRAFT intentionally does not prefer DOT4 and must not use packed16 K.
//
// Frozen: MTP_FA_INSTRUCTION_PR3_FROZEN

static const char * ggml_cuda_mtp_inst_name(int32_t inst) {
    switch (inst) {
        case GGML_FATTN_INST_MTP_DRAFT:             return "mtp_draft";
        case GGML_FATTN_INST_MTP_VERIFY_QK:         return "mtp_verify_qk";
        case GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK:  return "mtp_qblock_verify_qk";
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK:   return "mtp_draft_decode_qk";
        default:                                      return "none";
    }
}

// ── MTP draft decode helpers ────────────────────────────────────────

// BK threshold — delegates to centralized decode policy.
static int64_t ggml_cuda_q8k_dot4_decode_splitk_threshold() {
    return ggml_cuda_dot4_decode_policy_from_env().splitk_threshold;
}

static bool ggml_cuda_mtp_draft_dot4_decode_enabled() {
#ifdef GGML_USE_HIP
    return !ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE_DISABLE") &&
           !ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_DISABLE_DOT4_FA2");
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_f16k_q4v_vec_enabled() {
#ifdef GGML_USE_HIP
    // Exact f16K/q4V MTP draft-decode lane. This is the safe/control fast path
    // for current source-K MTP draft attention; it does not use the unsafe
    // f16->packed16 DOT4 adapter. Keep explicit opt-outs for A/B and emergency
    // fallback, and honor the old env when it is set to 0.
    const char * legacy = getenv("GGML_CUDA_ROCM_MTP_F16K_Q4V_VEC");
    if (legacy && atoi(legacy) == 0) {
        return false;
    }
    const char * disable = getenv("GGML_CUDA_ROCM_MTP_F16K_Q4V_VEC_DISABLE");
    if (disable && atoi(disable) != 0) {
        return false;
    }
    disable = getenv("LLAMA_MTP_DISABLE_F16K_Q4V_VEC");
    if (disable && atoi(disable) != 0) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

// ── Draft decode kill switches ────────────────────────────────────

static int64_t ggml_cuda_mtp_draft_dot4_decode_splitk_threshold() {
    const char * env = getenv(
        "GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE_SPLITK_THRESHOLD");
    return env ? atoll(env) : 2048;
}

static bool ggml_cuda_mtp_draft_dot4_decode_bn64_disabled() {
    const char * env = getenv(
        "GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE_DISABLE_BN64");
    return env && atoi(env) != 0;
}

static bool ggml_cuda_mtp_draft_dot4_decode_splitk_disabled() {
    const char * env = getenv(
        "GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE_DISABLE_SPLITK");
    return env && atoi(env) != 0;
}

static ggml_cuda_dot4_role ggml_cuda_dot4_role_for_mtp_draft_decode(
        const ggml_tensor * Q,
        const ggml_tensor * K) {
    if (!Q || !K) {
        return GGML_CUDA_DOT4_ROLE_NONE;
    }

    // Decode instruction must be scalar.
    if (Q->ne[1] != 1) {
        return GGML_CUDA_DOT4_ROLE_NONE;
    }

    const int64_t threshold =
        ggml_cuda_mtp_draft_dot4_decode_splitk_threshold();

    if (K->ne[1] >= threshold) {
        return GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT;
    }

    return GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT;
}

static bool ggml_cuda_mtp_f16k_q4v_vec_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    GGML_UNUSED(cc);
    if (!ggml_cuda_mtp_f16k_q4v_vec_enabled()) {
        return false;
    }

    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    if (fa_inst_i32 != GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }

    return Q->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 &&
           Q->ne[0] == 256 &&
           Q->ne[1] == 1 &&
           K->type == GGML_TYPE_F16 &&
           V->type == GGML_TYPE_Q4_0 &&
           K->ne[0] == Q->ne[0] &&
           V->ne[0] == Q->ne[0] &&
           K->ne[1] > 0 &&
           K->ne[1] % FATTN_KQ_STRIDE == 0 &&
           Q->ne[2] % K->ne[2] == 0 &&
           V->ne[2] == K->ne[2] &&
           Q->ne[3] == K->ne[3] &&
           V->ne[3] == K->ne[3] &&
           (!mask || (mask->type == GGML_TYPE_F16 &&
                      mask->ne[0] == K->ne[1] &&
                      mask->ne[1] == Q->ne[1] &&
                      mask->ne[2] == 1 &&
                      mask->ne[3] == Q->ne[3]));
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif
}

static bool ggml_cuda_mtp_draft_dot4_decode_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!ggml_cuda_mtp_draft_dot4_decode_enabled()) {
        return false;
    }
    if (!ggml_cuda_q8k_dot4_kq_enabled()) {
        return false;
    }

    if (ggml_cuda_mtp_draft_decode_dot4_disabled()) {
        return false;
    }

    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[1] != 1) {
        return false;
    }

    // V: f16 / q8_0 / q4_0 for source-f16 decode.
    const bool v_ok =
        V->type == GGML_TYPE_F16  ||
        V->type == GGML_TYPE_Q8_0 ||
        V->type == GGML_TYPE_Q4_0;

    // K: source f16 (op-local packed16) or original q8_0 K + q4_0 V.
    // No persistent I32 packed16 for MTP draft (enforced by KV-cache gate).
    const bool source_f16_k =
        K->type == GGML_TYPE_F16 &&
        K->ne[0] == Q->ne[0];

    const bool q8_k_q4_v_original =
        K->type == GGML_TYPE_Q8_0 &&
        V->type == GGML_TYPE_Q4_0 &&
        K->ne[0] == Q->ne[0];

    // The source-f16 -> op-local packed16 DOT4 adapter is intentionally unsafe
    // for MTP draft decode: it was measured to change top-1 logits and produce
    // zero acceptance. Keep it lab-only instead of silently selecting it when the
    // generic DOT4 env knobs are set.
    if (source_f16_k && !ggml_cuda_mtp_source_f16_dot4_unsafe_enabled()) {
        return false;
    }

    return Q->ne[0] == 256 &&
           V->ne[0] == Q->ne[0] &&
           v_ok &&
           (source_f16_k || q8_k_q4_v_original) &&
           K->ne[1] > 0 &&
           Q->ne[2] % K->ne[2] == 0 &&
           V->ne[2] == K->ne[2] &&
           Q->ne[3] == K->ne[3] &&
           V->ne[3] == K->ne[3] &&
           // Kill switch checks: BN64 / splitK disabled.
           (ggml_cuda_dot4_role_for_mtp_draft_decode(Q, K) ==
                GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT
                ? !ggml_cuda_mtp_draft_dot4_decode_bn64_disabled()
                : !ggml_cuda_mtp_draft_dot4_decode_splitk_disabled());
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif
}

// ── MTP draft decode selector (nq == 1 → DOT4 BN64/split-K) ─────────

static best_fattn_kernel ggml_cuda_select_mtp_draft_decode_fattn(
        const int cc,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const auto dot4_role = ggml_cuda_dot4_role_from_instruction(
            GGML_FATTN_INST_MTP_DRAFT_DECODE_QK, Q, K);

    const auto log_decode = [&](const char * impl_status, best_fattn_kernel selected,
                                 const char * decode_impl) {
        if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
            if (log_env && atoi(log_env) != 0) {
                const char * k_repr = selected == BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC ?
                    "source_f16_exact" :
                    ggml_cuda_dot4_k_repr_name(ggml_cuda_dot4_resolve_k_repr(Q, K, V));
                const char * v_repr = ggml_cuda_dot4_v_repr_name(V);
                const bool dot4_active = (selected == BEST_FATTN_KERNEL_Q8K_DOT4_KQ);
                const bool packed16_decode_active = (selected == BEST_FATTN_KERNEL_PACKED16_DECODE);
                const auto family = dot4_active
                    ? (dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT
                        ? GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_SPLITK
                        : GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_BN64)
                    : (packed16_decode_active
                        ? GGML_CUDA_FATTN_BACKEND_PACKED16_DECODE
                        : ggml_cuda_fattn_backend_family_from_kernel(selected));
                const char * family_name = ggml_cuda_fattn_backend_family_name(family);
                GGML_LOG_INFO("%s: fa_instruction=mtp_draft_decode_qk "
                        "dot4_role=%s backend_family=%s nq=%lld nk=%lld K=%s V=%s "
                        "%s=%s v_repr=%s decode_impl=%s "
                        "impl_status=%s selected=%s\n",
                        __func__,
                        ggml_cuda_dot4_role_name(dot4_role),
                        family_name,
                        (long long) Q->ne[1], (long long) K->ne[1],
                        ggml_type_name(K->type), ggml_type_name(V->type),
                        dot4_active ? "k_repr" : "k_repr_candidate", k_repr, v_repr,
                        decode_impl, impl_status,
                        selected == BEST_FATTN_KERNEL_Q8K_DOT4_KQ ? "rocm_q8k_dot4_kq" :
                        selected == BEST_FATTN_KERNEL_PACKED16_DECODE ? "rocm_packed16_decode" :
                        selected == BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC ? "rocm_mtp_f16k_q4v_decode" :
                        selected == BEST_FATTN_KERNEL_VEC ? "rocm_fattn_vec" :
                        "<existing_policy>");
            }
        }
    };

    // nq must be 1 for decode.
    if (Q->ne[1] != 1) {
        log_decode("nq_not_1_decode_ineligible", BEST_FATTN_KERNEL_NONE, "none");
        return BEST_FATTN_KERNEL_NONE;
    }

    const char * required = ggml_cuda_dp16_fa_route_require_name();
    dp16_fa_problem problem = ggml_cuda_make_dp16_fa_problem(dst, cc, ggml_cuda_fattn_capture_active());
    dp16_fa_plan plan = dp16_plan_mtp_fa(problem);
    const bool hotcold_sidecar_decode = ggml_cuda_mtp_hotcold_packed16_decode_supported(dst);

    const bool disable_bn64 = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_BN64") ||
        ggml_cuda_mtp_draft_dot4_decode_bn64_disabled();
    const bool disable_splitk = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_SPLITK") ||
        ggml_cuda_mtp_draft_dot4_decode_splitk_disabled();
    const bool role_disabled =
        (dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_BN64_MTP_DRAFT && disable_bn64) ||
        (dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT && disable_splitk);

    if ((plan.backend == DP16_BACKEND_FA2_Q8K_DOT4_DECODE ||
            plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE ||
            plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE) && role_disabled) {
        plan = dp16_make_fa1_vec_fallback(
            dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT ?
                "splitk_disabled" : "bn64_disabled");
    }

    const bool base_decode_legal =
        GGML_CUDA_CC_IS_RDNA3(cc) &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        Q->ne[0] == 256 && V->ne[0] == Q->ne[0] &&
        K->ne[1] > 0 &&
        Q->ne[2] % K->ne[2] == 0 &&
        V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] &&
        V->ne[3] == K->ne[3];

    if ((plan.backend == DP16_BACKEND_FA2_Q8K_DOT4_DECODE ||
            plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE ||
            plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE) && !base_decode_legal) {
        plan = dp16_make_fa1_vec_fallback("shape_or_device_reject");
    }

    if (plan.backend == DP16_BACKEND_FA2_Q8K_DOT4_DECODE &&
            !(K->type == GGML_TYPE_Q8_0 && K->ne[0] == Q->ne[0] &&
              (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0))) {
        plan = dp16_make_fa1_vec_fallback("missing_legal_k_representation");
    }

    if (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE &&
            !((K->type == GGML_TYPE_I32 && K->ne[0] * 4 == Q->ne[0] &&
               (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0 ||
                V->type == GGML_TYPE_V4_K16D16_144)) ||
              hotcold_sidecar_decode)) {
        plan = dp16_make_fa1_vec_fallback("packed16_decode_not_ready");
    }

    if (plan.backend == DP16_BACKEND_FA2_F16K_ADAPT_DOT4_DECODE) {
        if (!(K->type == GGML_TYPE_F16 && K->ne[0] == Q->ne[0] && V->type == GGML_TYPE_Q4_0)) {
            plan = dp16_make_fa1_vec_fallback("missing_f16_adapt_inputs");
        } else if (problem.capture) {
            plan = dp16_make_fa1_vec_fallback("capture_sidecar_not_ready");
        }
    }

    if (hotcold_sidecar_decode && !role_disabled && base_decode_legal) {
        plan = {};
        plan.valid = true;
        plan.selected = true;
        plan.default_allowed = false;
        plan.capture_safe = true;
        plan.experimental = true;
        plan.fallback = false;
        plan.plane = DP16_FA_PLANE_FA2_DOT4;
        plan.backend = DP16_BACKEND_FA2_PACKED16_DOT4_DECODE;
        plan.k_repr = DP16_K_REPR_PACKED16_I32_PERSISTENT;
        plan.shape = DP16_FA_SHAPE_1X64;
        plan.vpath = DP16_FA_VPATH_DIRECT_PV;
        plan.route = DP16_ROUTE_FA2_PACKED16_DOT4_DECODE;
        plan.reason = "mtp_draft_decode_f16_graph_packed16_sidecar_hotcold";
        plan.k_tile = 64;
    }

    const bool dp16_route_contract_applicable =
        ggml_cuda_dp16_fa_route_contract_applicable(required, problem) ||
        (ggml_cuda_dp16_fa_route_is_packed16_decode_alias(required) &&
         !ggml_cuda_fattn_route_contract_is_packed16_small_verify(required) &&
         ggml_cuda_mtp_hotcold_packed16_decode_contract_applicable(dst, false));
    if (dp16_route_contract_applicable &&
            !ggml_cuda_dp16_fa_route_contract_matches_plan(required, problem, plan)) {
        ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);
        ggml_cuda_dp16_fa_abort_route_mismatch(required, problem, plan);
    }
    ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);

    best_fattn_kernel selected = ggml_cuda_dp16_fa_plan_to_best_kernel(problem, plan, dst, f16_policy);
    if (plan.backend == DP16_BACKEND_FA1_VEC_FALLBACK) {
        if (selected == BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC &&
                ggml_cuda_mtp_f16k_q4v_vec_supported(cc, dst)) {
            selected = ggml_cuda_fattn_apply_route_contract(dst, selected, f16_policy);
            log_decode(plan.reason ? plan.reason : "fa1_vec_fallback_selected", selected, "fa1_vec_control");
            return selected;
        }
        if (selected == BEST_FATTN_KERNEL_VEC &&
                (dp16_mtp_force_vec_fallback() || ggml_cuda_dp16_fa_route_is_vec_alias(required))) {
            selected = ggml_cuda_fattn_apply_route_contract(dst, selected, f16_policy);
            log_decode(plan.reason ? plan.reason : "fa1_vec_fallback_selected", selected, "fa1_vec_generic");
            return selected;
        }
    } else if (selected != BEST_FATTN_KERNEL_NONE) {
        selected = ggml_cuda_fattn_apply_route_contract(dst, selected, f16_policy);
        const char * decode_impl = dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT ?
            "splitk_stage1_stage2" : "bn64";
        log_decode(plan.reason ? plan.reason : "dp16_decode_selected", selected, decode_impl);
        return selected;
    }

    if (ggml_cuda_mtp_f16k_q4v_vec_supported(cc, dst)) {
        const auto legacy = ggml_cuda_fattn_apply_route_contract(
                dst, BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC, f16_policy);
        log_decode("legacy_f16k_q4v_vec_selected", legacy, "f16k_q4v_vec");
        return legacy;
    }

    {
        const bool dot4_env = ggml_cuda_q8k_dot4_kq_enabled() || dp16_mtp_enable_dot4_fa2();
        const bool decode_env = ggml_cuda_mtp_draft_dot4_decode_enabled() || dp16_mtp_enable_dot4_fa2();
        const bool source_f16_k_candidate = K->type == GGML_TYPE_F16 && K->ne[0] == Q->ne[0];
        const bool source_f16_unsafe = ggml_cuda_mtp_source_f16_dot4_unsafe_enabled() || dp16_mtp_enable_f16_adapt_dot4();
        const bool shapes_ok = (Q->ne[0] == 256 && V->ne[0] == Q->ne[0] && K->ne[0] == Q->ne[0]);

        const char * status = !dot4_env                                      ? "env_disabled"
                            : !decode_env                                   ? "decode_env_disabled"
                            : source_f16_k_candidate && !source_f16_unsafe  ? "source_f16_dot4_unsafe_disabled"
                            : !shapes_ok                                    ? "shape_or_device_reject"
                            :                                                  "missing_legal_k_representation";
        log_decode(status, BEST_FATTN_KERNEL_NONE, "none");
    }

    return BEST_FATTN_KERNEL_NONE;
}

static int ggml_cuda_mtp_qblock_max_nq() {
    const char * env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_MAX_NQ");
    if (!env || env[0] == '\0') {
        env = getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ");
    }
    return env ? std::max(2, atoi(env)) : 8;
}

static bool ggml_cuda_mtp_qblock_pdmq_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_QBLOCK_PDMQ");
    return !env || atoi(env) != 0;
#else
    return false;
#endif
}

static best_fattn_kernel ggml_cuda_select_mtp_verify_fattn(
        const int cc,
        const ggml_tensor * dst,
        const ggml_cuda_rocm_quant_prefill_f16_policy * f16_policy,
        const ggml_fattn_instruction inst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_cuda_dot4_role dot4_role =
        ggml_cuda_dot4_role_for_mtp_verify(Q);

    const ggml_cuda_dot4_k_repr k_repr =
        ggml_cuda_dot4_resolve_k_repr(Q, K, V);

    const char * inst_name = ggml_cuda_fattn_instruction_name(inst);
    const bool qblock_inst = inst == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK;

    // Archetype: nq > 1 -> v4 recthist.
    // Current implementation gate: nq > 2 (DOT4 helpers reject nq <= 2).
    // nq == 2 behind explicit env for MTP_VERIFY_QK only.
    if (Q->ne[1] <= 1) {
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(GGML_CUDA_DOT4_ROLE_NONE),
            GGML_CUDA_FATTN_BACKEND_EXISTING,
            GGML_CUDA_DOT4_K_REPR_NONE,
            "nq_eq_1_decode_not_recthist",
            BEST_FATTN_KERNEL_NONE);

        return BEST_FATTN_KERNEL_NONE;
    }
    const char * required_for_nq2 = ggml_cuda_dp16_fa_route_require_name();
    const bool require_pdmq_for_nq2 = required_for_nq2 &&
        (strcmp(required_for_nq2, "rocm_packed16_dot4_mmq") == 0 ||
         strcmp(required_for_nq2, "packed16_dot4_mmq") == 0);
    const bool smallq_pdmq_env_for_nq2 = getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ") &&
        atoi(getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ")) != 0;
    if (Q->ne[1] == 2 && ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) &&
            !ggml_cuda_mtp_verify_dot4_nq2_enabled() &&
            !require_pdmq_for_nq2 && !smallq_pdmq_env_for_nq2) {
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_EXISTING,
            k_repr,
            "nq_eq_2_disabled",
            BEST_FATTN_KERNEL_NONE);

        return BEST_FATTN_KERNEL_NONE;
    }

    const char * required = ggml_cuda_dp16_fa_route_require_name();
    const char * packed16_decode_impl_env = getenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
    const bool packed16_decode_impl_smallq_fa2 = packed16_decode_impl_env && (
        strcmp(packed16_decode_impl_env, "packed16_fa2") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa2") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa2_tune") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa4") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa4_tune") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma") == 0 ||
        strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma_tune") == 0);
    const bool packed16_fa2_env = getenv("GGML_CUDA_ROCM_PACKED16_FA2") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_FA2")) != 0;
    const bool mtp_verify_smallq_fa2_env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_FA2") &&
        atoi(getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_FA2")) != 0;
    const bool mtp_verify_smallq_pdmq_env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ") &&
        atoi(getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ")) != 0;
    const bool mtp_qblock_pdmq_policy = qblock_inst && ggml_cuda_mtp_qblock_pdmq_enabled();
    const bool require_packed16_dot4_mmq = required &&
        (strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
         strcmp(required, "packed16_dot4_mmq") == 0) &&
        ggml_cuda_fattn_route_contract_applicable(required, dst, nullptr);
    const bool require_packed16_small_verify =
        ggml_cuda_fattn_route_contract_is_packed16_small_verify(required) &&
        ggml_cuda_fattn_route_contract_applicable(required, dst, nullptr);
    const int small_verify_max_nq = qblock_inst ? ggml_cuda_mtp_qblock_max_nq() :
        (getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ") ?
        std::max(2, atoi(getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ"))) : 4);
    const bool smallq_packed16_persistent =
        K->type == GGML_TYPE_I32 && (K->ne[0] * 4 == Q->ne[0] || K->ne[0] * 8 == Q->ne[0]);
    const bool smallq_hotcold_sidecar_relaxed =
        ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, false);
    const bool smallq_hotcold_sidecar =
        ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, true);
    const bool smallq_fa2_shape =
        ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_cuda_mtp_hotcold_v_type_supported(V) && Q->ne[0] == 256 && V->ne[0] == Q->ne[0] &&
        Q->ne[1] >= 2 && Q->ne[1] <= small_verify_max_nq && K->ne[1] > 0 &&
        K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] &&
        (smallq_packed16_persistent || smallq_hotcold_sidecar);
    const bool smallq_pdmq_shape =
        ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_cuda_mtp_hotcold_v_type_supported(V) && Q->ne[0] == 256 && V->ne[0] == Q->ne[0] &&
        Q->ne[1] >= 2 && Q->ne[1] <= small_verify_max_nq && K->ne[1] > 0 &&
        K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 && V->ne[2] == K->ne[2] &&
        Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] &&
        (smallq_packed16_persistent || smallq_hotcold_sidecar_relaxed);
    if ((require_packed16_dot4_mmq || mtp_verify_smallq_pdmq_env || mtp_qblock_pdmq_policy) &&
            smallq_pdmq_shape && !smallq_packed16_persistent &&
            smallq_hotcold_sidecar_relaxed && !smallq_hotcold_sidecar) {
        GGML_ABORT("required packed16 DOT4-MMQ F16-sidecar route is cold-blocked; route=%s nq=%lld nk=%lld K=%s V=%s window=%lld cold_unsafe=%d",
            required ? required : "rocm_packed16_dot4_mmq",
            (long long) Q->ne[1],
            (long long) K->ne[1],
            ggml_type_name(K->type),
            ggml_type_name(V->type),
            (long long) ggml_cuda_mtp_packed16_hotcold_k_window(),
            ggml_cuda_mtp_packed16_hotcold_cold_unsafe_enabled() ? 1 : 0);
    }
    if ((require_packed16_dot4_mmq || mtp_verify_smallq_pdmq_env || mtp_qblock_pdmq_policy) &&
            smallq_pdmq_shape && smallq_hotcold_sidecar &&
            ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
        const auto selected = ggml_cuda_fattn_apply_route_contract(
                dst, BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ, f16_policy);
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ,
            k_repr,
            require_packed16_dot4_mmq ? "required_mtp_verify_smallq_pdmq" : (qblock_inst ? "mtp_qblock_smallq_pdmq" : "mtp_verify_smallq_pdmq_opt_in"),
            selected);
        return selected;
    }
    if ((require_packed16_small_verify || mtp_verify_smallq_fa2_env || packed16_fa2_env || packed16_decode_impl_smallq_fa2) &&
            smallq_fa2_shape && ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
        const auto selected = ggml_cuda_fattn_apply_route_contract(
                dst, BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ, f16_policy);
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ,
            k_repr,
            require_packed16_small_verify ? "required_mtp_verify_smallq_pdmq" : (qblock_inst ? "mtp_qblock_smallq_fa2_alias_pdmq" : "mtp_verify_smallq_pdmq_opt_in"),
            selected);
        return selected;
    }

    dp16_fa_problem problem = ggml_cuda_make_dp16_fa_problem(dst, cc, ggml_cuda_fattn_capture_active());
    dp16_fa_plan plan = dp16_plan_mtp_fa(problem);

    if (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY &&
            !ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
        plan = dp16_make_fa1_vec_fallback("pdmq_not_ready");
    }

    if (ggml_cuda_dp16_fa_route_contract_applicable(required, problem) &&
            !ggml_cuda_dp16_fa_route_contract_matches_plan(required, problem, plan)) {
        ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);
        ggml_cuda_dp16_fa_abort_route_mismatch(required, problem, plan);
    }
    ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);

    best_fattn_kernel selected = ggml_cuda_dp16_fa_plan_to_best_kernel(problem, plan, dst, f16_policy);
    if (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY &&
            selected != BEST_FATTN_KERNEL_NONE) {
        selected = ggml_cuda_fattn_apply_route_contract(dst, selected, f16_policy);
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ,
            k_repr,
            plan.reason ? plan.reason : "dp16_pdmq_candidate",
            selected);
        return selected;
    }

    if (plan.backend == DP16_BACKEND_FA1_VEC_FALLBACK &&
            selected == BEST_FATTN_KERNEL_VEC &&
            (dp16_mtp_force_vec_fallback() || ggml_cuda_dp16_fa_route_is_vec_alias(required))) {
        selected = ggml_cuda_fattn_apply_route_contract(dst, selected, f16_policy);
        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_VEC,
            GGML_CUDA_DOT4_K_REPR_NONE,
            plan.reason ? plan.reason : "dp16_fa1_vec_fallback",
            selected);
        return selected;
    }

    // DOT4 not legal — determine why.
    {
        const bool dot4_env = ggml_cuda_q8k_dot4_kq_enabled() || dp16_mtp_enable_dot4_fa2();
        const bool f16_k = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
        const bool adapter_on = ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled() || dp16_mtp_enable_f16_adapt_dot4();

        const char * status = !dot4_env                ? "env_disabled"
                            : f16_k && !adapter_on     ? "source_f16_materialization_disabled"
                            : f16_k && adapter_on      ? "shape_or_device_reject"
                            :                            "missing_legal_k_representation";

        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_EXISTING,
            k_repr,
            status,
            BEST_FATTN_KERNEL_NONE);
    }

    // ── Instruction-aware fallback: MMA_F16 ─────────────────────────
    // Only after DOT4 is rejected. f16/f32 K/V only — no q8/q4 dequant.
    // Default off; env-gated.

    if (ggml_cuda_mtp_verify_mma_f16_supported(cc, dst)) {
        const best_fattn_kernel selected_mma = BEST_FATTN_KERNEL_MMA_F16;

        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_MMA_F16,
            GGML_CUDA_DOT4_K_REPR_NONE,
            "dot4_rejected_mma_f16_selected",
            selected_mma);

        return selected_mma;
    }

    return BEST_FATTN_KERNEL_NONE; // signal: use existing policy
}

// ── Instruction dispatch block (below) uses g_fattn_select_ctx
// which is defined earlier in this file.

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];


    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // ── MTP early diagnostic ────────────────────────────────────────
    // Fires for every MTP hint, regardless of KV type. Logs whether DOT4
    // is eligible for MTP_VERIFY based on KV format compatibility.
    {
        const int32_t fa_inst_i32 = ((const int32_t *)KQV->op_params)[4];
        const bool is_mtp = (fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT ||
                             ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32));
        if (is_mtp) {
            const char * inst_name = fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT
                ? "mtp_draft" : "mtp_verify";
            const bool quantized_kv = ggml_is_quantized(K->type) || ggml_is_quantized(V->type);
            const bool dot4_kv_ok = (K->type == GGML_TYPE_I32) ||
                (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0);
            if (!dot4_kv_ok && ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32)) {
                if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
                    if (log_env && atoi(log_env) != 0) {
                        GGML_LOG_INFO("%s: fa_route fa_inst=mtp_verify nq=%lld dot4_candidate=0 dot4_reject_reason=kv_type_incompatible K=%s V=%s\n",
                                __func__, (long long) Q->ne[1],
                                ggml_type_name(K->type), ggml_type_name(V->type));
                    }
                }
            }
            GGML_UNUSED(quantized_kv);
        }
    }

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    // I32 packed16 K: DOT4-only format.  Bypass generic shape switch that
    // would reject K->ne[0]=64 vs V->ne[0]=128/256 mismatch.
    if (K->type == GGML_TYPE_I32) {
        // Route requirement detection (must be before shape checks)
        const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        const bool require_packed16_wmma =
            required_route &&
            (strcmp(required_route, "rocm_packed16_wmma_tile") == 0 ||
             strcmp(required_route, "packed16_wmma_tile") == 0);
        const bool require_packed16_fa2_vec =
            required_route && strcmp(required_route, "rocm_packed16_fa2_vec") == 0;
        const bool require_qtile4_gqa6_kvshared =
            required_route &&
            (strcmp(required_route, "rocm_q8k_dot4_qtile4_gqa6_kvshared") == 0 ||
             strcmp(required_route, "rocm_packed16_qtile4_gqa6_kvshared") == 0 ||
             strcmp(required_route, "qtile4_gqa6_kvshared") == 0);
        const bool require_packed16_dot4_mmq =
            required_route &&
            (strcmp(required_route, "rocm_packed16_dot4_mmq") == 0 ||
             strcmp(required_route, "packed16_dot4_mmq") == 0);

        const bool k_shape_ok = K->ne[0] * 4 == Q->ne[0] || K->ne[0] * 8 == Q->ne[0];  // packed D/4 or D/8 I32 K
        // V layout: FA expects [D,n_kv,heads,batch], native v_trans is [n_kv,heads,D,batch]
        const bool v_shape_fa    = V->ne[0] == Q->ne[0];
        const bool v_shape_trans = V->type == GGML_TYPE_F16 && V->ne[0] == K->ne[1] && V->ne[1] == K->ne[2] && V->ne[2] == Q->ne[0];
        const bool v_shape_ok    = v_shape_fa || (require_packed16_wmma && v_shape_trans);
        const bool head_ok    = Q->ne[2] > 0 && K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0;

        if (!(Q->type == GGML_TYPE_F32 && ggml_cuda_packed16_dot4_mmq_v_supported(V->type) && dst->type == GGML_TYPE_F32 &&
              k_shape_ok && v_shape_ok && K->ne[1] > 0 && head_ok)) {
            if (dp16_trace_enabled()) {
                dp16_problem dp_problem = dp16_problem_init(DP16_OP_FA_QKPV);
                dp_problem.m = Q->ne[1];
                dp_problem.n = K->ne[1];
                dp_problem.k = Q->ne[0];
                dp_problem.batch = Q->ne[3];
                dp_problem.heads_q = Q->ne[2];
                dp_problem.heads_kv = K->ne[2];
                dp_problem.head_dim = Q->ne[0];
                dp_problem.src0_type = Q->type;
                dp_problem.src1_type = K->type;
                dp_problem.src2_type = V->type;
                dp_problem.dst_type = dst->type;
                dp_problem.is_decode = Q->ne[1] == 1;
                dp_problem.cc = cc;
                dp_problem.a = dp16_operand_desc_from_type(DP16_OPERAND_ACTIVATION, DP16_STORAGE_TRANSIENT_TILE,
                        Q->type, Q->ne[1], Q->ne[0], Q->nb[1], Q->nb[0]);
                dp_problem.b = dp16_operand_desc_from_type(DP16_OPERAND_K_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
                        K->type, K->ne[1], Q->ne[0], K->nb[1], K->nb[0]);
                dp_problem.v = dp16_operand_desc_from_type(DP16_OPERAND_V_CACHE, DP16_STORAGE_PERSISTENT_CACHE,
                        V->type, V->ne[1], V->ne[0], V->nb[1], V->nb[0]);
                dp_problem.dst = dp16_operand_desc_from_type(DP16_OPERAND_OUTPUT, DP16_STORAGE_OUTPUT,
                        dst->type, dst->ne[1], dst->ne[0], dst->nb[1], dst->nb[0]);
                const dp16_reject_reason dp16_reason = (!k_shape_ok || !v_shape_ok || Q->ne[0] != 256) ?
                    DP16_REJECT_K_NOT_ALIGNED : DP16_REJECT_TYPE_UNSUPPORTED;
                dp16_trace_emit_reject(dp_problem, dp16_reason, "rocm_packed16_dot4_mmq");
            }
            // Log why I32 K was rejected so we can debug shape mismatches.
            static bool i32_reject_printed = false;
            if (!i32_reject_printed) {
                i32_reject_printed = true;
                fprintf(stderr,
                    "fattn.cu: I32 K rejected — Q type=%d ne=[%lld,%lld,%lld,%lld] "
                    "K ne=[%lld,%lld,%lld,%lld] V ne=[%lld,%lld,%lld,%lld]\n",
                    Q->type, (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                    (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                    (long long) V->ne[0], (long long) V->ne[1], (long long) V->ne[2], (long long) V->ne[3]);
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        // Production policy: persistent packed16 I32-K MTP routes are enabled
        // by default after route/replay validation. Keep an explicit disable
        // knob for A/B and emergency fallback.
        {
            const int32_t fi = ((const int32_t *)dst->op_params)[4];
            if (ggml_cuda_fattn_inst_is_mtp_verify_qk(fi) ||
                fi == GGML_FATTN_INST_MTP_DRAFT ||
                fi == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
                const char * mtp_disable_p16 = getenv("LLAMA_MTP_DISABLE_PACKED16_FA");
                const bool disable_p16_mtp = mtp_disable_p16 && atoi(mtp_disable_p16) != 0;
                if (disable_p16_mtp) {
                    if (getenv("LLAMA_MTP_FA_ROUTE")) {
                        fprintf(stderr, "MTP_FA_ROUTE: inst=%d k_type=%d selected=none reason=packed16_mtp_disabled\n",
                            fi, K->type);
                    }
                    return BEST_FATTN_KERNEL_NONE;
                }
            }
        }

        // ── Route dispatch ───────────────────────────────────
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const ggml_fattn_instruction inst = (ggml_fattn_instruction)fa_inst_i32;
        ggml_cuda_packed16_sidecar_meta packed16_meta = {};
        llama_kv_cache_get_packed16_sidecar_meta(K->data, &packed16_meta);
        const bool packed16_layout_row = packed16_meta.layout_kind == GGML_CUDA_PACKED16_K_LAYOUT_ROW;
        const bool packed8_q4_144_k_format =
            K->type == GGML_TYPE_I32 &&
            K->ne[0] * 8 == Q->ne[0] &&
            packed16_meta.k_format == GGML_CUDA_PDMQ_K_FORMAT_PACKED8_Q4_144 &&
            packed16_meta.code_bits == 4 && packed16_meta.zero_point == 8;

        const char * packed16_fa2_vec_min_nk_env = getenv("GGML_CUDA_ROCM_PACKED16_LDS_A8_MIN_NK");
        const int packed16_fa2_vec_min_nk = packed16_fa2_vec_min_nk_env ? atoi(packed16_fa2_vec_min_nk_env) : 12288;
        const bool packed16_fa2_vec_long_context = K->ne[1] >= packed16_fa2_vec_min_nk;
        const bool packed16_fa2_vec_force = require_packed16_fa2_vec;
        const bool packed16_fa2_vec_qblock =
            inst == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK &&
            getenv("GGML_CUDA_ROCM_MTP_QBLOCK_VEC") &&
            atoi(getenv("GGML_CUDA_ROCM_MTP_QBLOCK_VEC")) != 0;
        const char * packed16_fa2_vec_env = getenv("GGML_CUDA_ROCM_PACKED16_FA2_VEC");
        const bool packed16_fa2_vec_global_enabled = packed16_fa2_vec_env && atoi(packed16_fa2_vec_env) != 0;
        const int packed16_fa2_vec_max_nq = packed16_fa2_vec_qblock ? ggml_cuda_mtp_qblock_max_nq() : 4;
        if ((packed16_fa2_vec_force || packed16_fa2_vec_qblock || (packed16_fa2_vec_global_enabled && !packed16_fa2_vec_long_context)) &&
                Q->ne[1] >= 1 && Q->ne[1] <= packed16_fa2_vec_max_nq &&
                K->type == GGML_TYPE_I32 &&
                V->type == GGML_TYPE_Q4_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256) {
            if (!packed16_layout_row) {
                if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH && packed16_fa2_vec_force) {
                    GGML_ABORT("required rocm_packed16_fa2_vec route is row-layout only but packed16 descriptor layout=%s; use rocm_packed16_dot4_mmq/PDMQ for D16-planar",
                            ggml_cuda_packed16_k_layout_kind_name(packed16_meta.layout_kind));
                }
            } else {
                if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH &&
                        (getenv("LLAMA_MTP_FA_ROUTE") || getenv("GGML_CUDA_ROCM_PACKED16_DECODE_LOG") ||
                         getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") || getenv("COMPRESSED_KV_FATTN_LOG"))) {
                    static int packed16_fa2_vec_log_count = 0;
                    const bool packed16_fa2_vec_verbose_all =
                        (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE")) != 0) ||
                        (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")) != 0);
                    if (packed16_fa2_vec_verbose_all || packed16_fa2_vec_log_count++ < 16) {
                        fprintf(stderr,
                            "packed16_fa2_vec selected=rocm_packed16_fa2_vec route=packed16k_q4v_fa2_vec "
                            "inst=%d nq=%lld nk=%lld hq=%lld hk=%lld batch=%lld force=%d qblock=%d long_context=%d min_nk=%d layout=%s\n",
                            fa_inst_i32,
                            (long long) Q->ne[1], (long long) K->ne[1],
                            (long long) Q->ne[2], (long long) K->ne[2], (long long) Q->ne[3],
                            packed16_fa2_vec_force ? 1 : 0, packed16_fa2_vec_qblock ? 1 : 0,
                            packed16_fa2_vec_long_context ? 1 : 0,
                            packed16_fa2_vec_min_nk,
                            ggml_cuda_packed16_k_layout_kind_name(packed16_meta.layout_kind));
                    }
                }
                return BEST_FATTN_KERNEL_PACKED16_FA2_VEC;
            }
        }

        const bool require_packed16_dot4_decode =
            required_route && strcmp(required_route, DP16_ROUTE_FA2_PACKED16_DOT4_DECODE) == 0;
        if (require_packed16_dot4_decode &&
                fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK && Q->ne[1] == 1) {
            const dp16_fa_problem problem =
                ggml_cuda_make_dp16_fa_problem(dst, cc, ggml_cuda_fattn_capture_active());
            const dp16_fa_plan plan = dp16_plan_mtp_fa(problem);
            ggml_cuda_dp16_fa_emit_plan_trace_if_needed(problem, plan);
            if (!ggml_cuda_dp16_fa_route_contract_matches_plan(required_route, problem, plan)) {
                ggml_cuda_dp16_fa_abort_route_mismatch(required_route, problem, plan);
            }
            if (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_DECODE) {
                return BEST_FATTN_KERNEL_PACKED16_DECODE;
            }
            if (plan.backend == DP16_BACKEND_FA2_PACKED16_DOT4_MMQ_VERIFY) {
                return ggml_cuda_packed16_dot4_mmq_supported(cc, dst) ?
                    BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ : BEST_FATTN_KERNEL_NONE;
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        const bool require_packed16_decode_alias =
            required_route &&
            (strcmp(required_route, "rocm_packed16_decode") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_scalar") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_q4pair") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_gqa_scalar") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_waveqk") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_pvwmma") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_gqa_pvwmma") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_wmma_full") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_gqa_wmma_full") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_dsplit") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_logits_debug") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_waveqk_q4pair") == 0 ||
             strcmp(required_route, "rocm_packed16_decode_splitk") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify_splitk") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify_batched_splitk") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify_fa2") == 0 ||
             strcmp(required_route, "rocm_packed16_fa2") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify_fa4") == 0 ||
             strcmp(required_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
             strcmp(required_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
        const bool require_packed16_decode_splitk_alias =
            required_route && strcmp(required_route, "rocm_packed16_decode_splitk") == 0;
        const bool require_packed16_small_verify =
            required_route && strcmp(required_route, "rocm_packed16_small_verify") == 0;
        const bool require_packed16_small_verify_splitk =
            required_route && strcmp(required_route, "rocm_packed16_small_verify_splitk") == 0;
        const bool require_packed16_small_verify_batched_splitk =
            required_route && (strcmp(required_route, "rocm_packed16_small_verify_batched_splitk") == 0 ||
                strcmp(required_route, "rocm_packed16_small_verify_fa2") == 0 ||
                strcmp(required_route, "rocm_packed16_fa2") == 0 ||
                strcmp(required_route, "rocm_packed16_small_verify_fa4") == 0 ||
                strcmp(required_route, "rocm_packed16_small_verify_fa4_pvwmma") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages_pvwmma") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages_pint8pv") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages_pint8pv_dot4") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
                strcmp(required_route, "rocm_packed16_bm_dot4_pages_intflash_vfrag_wmma") == 0);
        const char * decode_max_nq_env = getenv("GGML_CUDA_ROCM_Q8K_DOT4_DECODE_MAX_NQ");
        const int decode_max_nq = decode_max_nq_env ? atoi(decode_max_nq_env) : 1;
        const bool packed16_i32_qblock_verify_inst = inst == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK;
        const char * small_verify_max_nq_env = getenv("GGML_CUDA_ROCM_SMALL_VERIFY_MAX_NQ");
        const int small_verify_max_nq = packed16_i32_qblock_verify_inst
            ? ggml_cuda_mtp_qblock_max_nq()
            : (small_verify_max_nq_env ? atoi(small_verify_max_nq_env) : 4);
        if (require_packed16_decode_alias &&
                ((fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK && Q->ne[1] == 1) ||
                 (require_packed16_decode_splitk_alias && Q->ne[1] <= decode_max_nq && V->type == GGML_TYPE_Q4_0) ||
                 ((require_packed16_small_verify || require_packed16_small_verify_splitk || require_packed16_small_verify_batched_splitk) && Q->ne[1] >= 2 && Q->ne[1] <= small_verify_max_nq && V->type == GGML_TYPE_Q4_0))) {
            return ggml_cuda_packed16_dot4_mmq_supported(cc, dst) ?
                BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ : BEST_FATTN_KERNEL_NONE;
        }

        const bool packed16_decode_enabled =
            require_packed16_decode_alias ||
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PDMQ_K_CACHE") ||
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE") ||
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_Q8K_DOT4_KQ") ||
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PACKED16_DECODE_EXPERIMENT");

        if (require_qtile4_gqa6_kvshared) {
            const int gqa_ratio_i32 = K->ne[2] > 0 ? (int) (Q->ne[2] / K->ne[2]) : 0;
            const bool qtile4_supported =
                Q->ne[0] == 256 && Q->ne[1] >= 4 &&
                gqa_ratio_i32 == 6 && K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 &&
                K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0 &&
                k_shape_ok && v_shape_fa;
            if (!qtile4_supported) {
                if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
                    GGML_ABORT("required qtile4/GQA6 KV-shared route was not selected; Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=%s gqa=%d",
                        (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                        (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                        ggml_type_name(V->type), gqa_ratio_i32);
                }
                return BEST_FATTN_KERNEL_NONE;
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        const bool require_q8k_dot4_kq =
            required_route &&
            (strcmp(required_route, "rocm_q8k_dot4_kq") == 0 ||
             strcmp(required_route, "rocm_mtp_verify_dot4_recthist") == 0 ||
             strcmp(required_route, "rocm_mtp_draft_dot4_decode") == 0 ||
             strcmp(required_route, "rocm_mtp_draft_dot4_decode_bn64") == 0 ||
             strcmp(required_route, "rocm_mtp_draft_dot4_decode_splitk") == 0 ||
             strcmp(required_route, "rocm_q8k_dot4_recthist_mtp_verify") == 0 ||
             strcmp(required_route, "rocm_q8k_dot4_decode_mtp_draft") == 0 ||
             strcmp(required_route, "rocm_q8k_dot4_decode_bn64_mtp_draft") == 0 ||
             strcmp(required_route, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0 ||
             strcmp(required_route, "dot4_decode") == 0 ||
             strcmp(required_route, "dot4_recthist") == 0);
        if (require_q8k_dot4_kq) {
            if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
                GGML_ABORT("required FA route %s expects true GGML_TYPE_Q8_0/block32 K but saw packed16-q8 side-channel K (physical=%s); use %s or %s",
                    required_route,
                    ggml_cuda_fattn_k_physical_repr_name(K),
                    DP16_ROUTE_FA2_PACKED16_DOT4_DECODE,
                    DP16_ROUTE_FA_PACKED16_MMQ);
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        const bool require_mtp_f16k_q4v_vec =
            required_route && ggml_cuda_fattn_route_contract_is_mtp_f16k_q4v_vec(required_route) &&
            ggml_cuda_fattn_route_contract_applicable(required_route, dst, nullptr);
        if (require_mtp_f16k_q4v_vec) {
            return BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC;
        }

        const bool standard_packed16_q4_dot4_mmq_smallq =
            ggml_cuda_fattn_inst_is_mtp_verify_qk(inst) &&
            V->type == GGML_TYPE_Q4_0 && Q->ne[1] <= small_verify_max_nq;
        if (require_packed16_dot4_mmq || standard_packed16_q4_dot4_mmq_smallq) {
            // Packed16 I32 K + q4_0 V defaults to DOT4/MMQ only for the compact
            // small-Q lane.  Larger prefill has enough Q rows to amortize tiled
            // PWMMA setup, so it must fall through to the packed16 WMMA auto
            // selector below unless PDMQ is explicitly route-required.
            if (!ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
                if (require_packed16_dot4_mmq) {
                    GGML_ABORT("required rocm_packed16_dot4_mmq route was not selected; Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=%s",
                        (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                        (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                        ggml_type_name(V->type));
                }
            } else {
                return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
            }
        }
        if (require_packed16_wmma && ggml_cuda_packed16_wmma_tile_enabled()) {
            if (Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_NONE;
            }
            return BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
        }

        // Packed16 MTP verify is now a validated FA2/PDMQ lane for persistent
        // packed16 I32 K + q4_0 V. Keep an explicit disable knob for fallback.
        const char * mtp_disable_p16_env = getenv("LLAMA_MTP_DISABLE_PACKED16_FA");
        const bool mtp_packed16_enabled = !(mtp_disable_p16_env && atoi(mtp_disable_p16_env) != 0);
        const bool prefill_or_verify =
            Q->ne[1] > 1 &&
            (inst == GGML_FATTN_INST_NONE ||
             inst == GGML_FATTN_INST_PREFILL_QK ||
             inst == GGML_FATTN_INST_SPEC_VERIFY_QK ||
             inst == GGML_FATTN_INST_BATCH_VERIFY_QK ||
             (mtp_packed16_enabled && ggml_cuda_fattn_inst_is_mtp_verify_qk(inst)));
        // nq == 1 decode: restore the old no-MTP packed16 q8k DOT4 scalar
        // path for the exact K=i32,V=q4_0 scalar contract. Keep MTP/QBlock on
        // the validated PDMQ path unless a route explicitly asks otherwise.
        if (Q->ne[1] == 1) {
            ggml_cuda_dp16_fa_emit_packed16_mtp_draft_trace_if_needed(dst, cc);
            const bool standard_nomtp_q4_packed16_decode =
                (inst == GGML_FATTN_INST_NONE || inst == GGML_FATTN_INST_DECODE_QK) &&
                K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && K->ne[0] * 4 == Q->ne[0] &&
                Q->ne[2] % K->ne[2] == 0 && packed16_decode_enabled;
            const bool v4_144_nomtp_packed16_decode =
                (inst == GGML_FATTN_INST_NONE || inst == GGML_FATTN_INST_DECODE_QK ||
                 inst == GGML_FATTN_INST_PREFILL_QK || inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) &&
                K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_V4_K16D16_144 &&
                Q->ne[0] == 256 && V->ne[0] == 256 && K->ne[0] * 4 == Q->ne[0] &&
                Q->ne[2] % K->ne[2] == 0 && packed16_decode_enabled;
            if (standard_nomtp_q4_packed16_decode || v4_144_nomtp_packed16_decode) {
                return BEST_FATTN_KERNEL_PACKED16_DECODE;
            }
            const bool standard_nomtp_q4_pdmq_decode =
                (inst == GGML_FATTN_INST_NONE || inst == GGML_FATTN_INST_DECODE_QK) && V->type == GGML_TYPE_Q4_0;
            const bool standard_mtp_q4_pdmq_decode =
                inst == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK && V->type == GGML_TYPE_Q4_0;
            const bool typed_v_pdmq_decode = V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16;
            const char * pdmq_vpath = getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_VPATH");
            const bool explicit_v4_144_pdmq_decode = V->type == GGML_TYPE_V4_K16D16_144 &&
                (require_packed16_dot4_mmq || ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PROFILE") ||
                 ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PV4") ||
                 (pdmq_vpath && (strcmp(pdmq_vpath, "v4_k16d16_144") == 0 || strcmp(pdmq_vpath, "persistent_v4_k16d16_144") == 0)));
            const bool v4_k16d16_pdmq_decode = V->type == GGML_TYPE_V4_K16D16 || explicit_v4_144_pdmq_decode;
            if ((standard_nomtp_q4_pdmq_decode || standard_mtp_q4_pdmq_decode || typed_v_pdmq_decode || v4_k16d16_pdmq_decode) &&
                    ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
                return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        const bool qtile4_gqa6_kvshared_auto =
            getenv("GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_AUTO") &&
            atoi(getenv("GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_AUTO")) != 0;
        const int qtile4_gqa6_kvshared_min_nq = getenv("GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_MIN_NQ") ?
            std::max(1, atoi(getenv("GGML_CUDA_ROCM_Q8K_DOT4_QTILE4_GQA6_KVSHARED_MIN_NQ"))) : 4;
        const int packed16_gqa_ratio = K->ne[2] > 0 ? (int) (Q->ne[2] / K->ne[2]) : 0;
        if (qtile4_gqa6_kvshared_auto &&
                prefill_or_verify && Q->ne[1] >= qtile4_gqa6_kvshared_min_nq && Q->ne[0] == 256 &&
                packed16_gqa_ratio == 6 && K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0 &&
                K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0 &&
                ggml_cuda_q8k_dot4_kq_enabled()) {
            return BEST_FATTN_KERNEL_NONE;
        }

        // nq > 1 prefill/verify:
        // - Small-verify decode route: for nq=2..4 verify, use the decode-style
        //   small_verify kernel which is optimized for small-batch nq>1 (shared K/V,
        //   DOT4-batched QK, fused softmax+P×V). Much cheaper than full prefill.
        //   Gated behind the existing packed16 decode env: GGML_CUDA_ROCM_Q8K_DOT4_KQ.
        // - PDMQ/DOT4-MMQ remains the small-Q and forced-route path for q4/q8/f16 V.
        // - PWMMA BM64 i8-QK PV-WMMA DBV is the production prefill path once the
        //   prompt is large enough to amortize its setup work.
        // - DOT4_KQ is the last resort.
        const char * packed16_decode_impl_env = getenv("GGML_CUDA_ROCM_PACKED16_DECODE_IMPL");
        const bool packed16_fa2_env = getenv("GGML_CUDA_ROCM_PACKED16_FA2") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_FA2")) != 0;
        const bool packed16_decode_small_verify_env =
            (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify") == 0) ||
            (packed16_decode_impl_env && strcmp(packed16_decode_impl_env, "small_verify_splitk") == 0) ||
            (packed16_decode_impl_env && (strcmp(packed16_decode_impl_env, "small_verify_batched_splitk") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa2") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa2_tune") == 0 ||
                strcmp(packed16_decode_impl_env, "packed16_fa2") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa4") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa4_tune") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma") == 0 ||
                strcmp(packed16_decode_impl_env, "small_verify_fa4_pvwmma_tune") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_tune") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_pvwmma") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_pint8pv_dot4") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_dot4") == 0 ||
                strcmp(packed16_decode_impl_env, "bm_dot4_pages_intflash_vfrag_wmma") == 0));
        const bool dot4_sup = ggml_cuda_packed16_dot4_mmq_supported(cc, dst);
        const bool packed16_dot4_mmq_smallq =
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SMALLQ") &&
            atoi(getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SMALLQ")) != 0;
        const bool packed16_dot4_mmq_shortctx =
            getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHORTCTX") &&
            atoi(getenv("GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_SHORTCTX")) != 0;
        const int packed16_dot4_mmq_max_nk = getenv("GGML_CUDA_ROCM_PACKED16_LDS_A8_MIN_NK") ?
            atoi(getenv("GGML_CUDA_ROCM_PACKED16_LDS_A8_MIN_NK")) : 12288;
        if ((packed16_dot4_mmq_shortctx || (packed16_dot4_mmq_smallq && Q->ne[1] <= small_verify_max_nq)) &&
                prefill_or_verify && dot4_sup &&
                Q->ne[1] >= 2 &&
                K->ne[1] < packed16_dot4_mmq_max_nk &&
                K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0) {
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
        }
        const bool typed_v_pdmq_smallq = V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16;
        if (typed_v_pdmq_smallq && prefill_or_verify && dot4_sup &&
                Q->ne[1] >= 2 && Q->ne[1] <= 8 && K->type == GGML_TYPE_I32) {
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
        }
        // Keep packed16 WMMA/PVMMA available for prefill. BM32 direct-V remains
        // the conservative short-context route, while long-context prefill can use
        // DBV PV-WMMA after the causal inactive-row PV accumulation bug was fixed
        // in the tile kernel. Set GGML_CUDA_ROCM_PACKED16_WMMA_DBV_AUTO=0 to
        // force the pre-promotion BM32 behavior for comparison/diagnostics.
        const bool small_verify_decode_opt_in =
            getenv("LLAMA_MTP_FA_ROUTE") &&
            (packed16_fa2_env ||
             packed16_decode_small_verify_env ||
             require_packed16_small_verify ||
             require_packed16_small_verify_splitk ||
             require_packed16_small_verify_batched_splitk);
        if (small_verify_decode_opt_in &&
            Q->ne[1] >= 2 && Q->ne[1] <= small_verify_max_nq &&
            K->type == GGML_TYPE_I32 && V->type == GGML_TYPE_Q4_0 && dot4_sup) {
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
        }
        const bool auto_verbose =
            (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE"))) ||
            (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")));
        // Persistent V4 K16D16 formats default to dedicated scalar DOT4-MMQ decoders.
        // V4_144 is promoted to PWMMA only for no-spec prefill; MTP/verify paths
        // still require the legacy lab knob until speculative acceptance parity is restored.
        const bool v4_k16d16_v = V->type == GGML_TYPE_V4_K16D16 || V->type == GGML_TYPE_V4_K16D16_144;
        const bool v4_144_nomtp_prefill =
            V->type == GGML_TYPE_V4_K16D16_144 &&
            (inst == GGML_FATTN_INST_NONE || inst == GGML_FATTN_INST_PREFILL_QK);
        const bool v4_144_mtp_batched =
            V->type == GGML_TYPE_V4_K16D16_144 &&
            (inst == GGML_FATTN_INST_MTP_VERIFY_QK || inst == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK ||
             inst == GGML_FATTN_INST_MTP_DRAFT);
        const bool v4_144_mtp_pwmma =
            v4_144_mtp_batched && ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PWMMA_MTP");
        const bool v4_144_pv4_pwmma_disable =
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_DISABLE");
        const char * v4_144_pv4_pwmma_candidate_env = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PV4_PWMMA_PREFILL_CANDIDATE");
        const bool v4_144_pv4_pwmma_candidate_enabled =
            !v4_144_pv4_pwmma_candidate_env || atoi(v4_144_pv4_pwmma_candidate_env) != 0;
        const bool v4_144_pv4_pwmma_prefill_candidate =
            V->type == GGML_TYPE_V4_K16D16_144 &&
            !v4_144_pv4_pwmma_disable &&
            v4_144_pv4_pwmma_candidate_enabled &&
            !v4_144_mtp_batched;
        const bool v4_144_pwmma_prefill =
            V->type == GGML_TYPE_V4_K16D16_144 &&
            !v4_144_pv4_pwmma_disable &&
            (v4_144_nomtp_prefill || v4_144_mtp_pwmma ||
             v4_144_pv4_pwmma_prefill_candidate ||
             (!v4_144_mtp_batched && ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_V4_K16D16_144_PWMMA_PREFILL")));
        const char * v4_144_pwmma_min_nq_env = getenv("GGML_CUDA_ROCM_V4_K16D16_144_PWMMA_PREFILL_MIN_NQ");
        const int v4_144_pwmma_min_nq = v4_144_pwmma_min_nq_env ? std::max(1, atoi(v4_144_pwmma_min_nq_env)) : 16;
        const bool v4_144_pwmma_nq_ok = !v4_144_pwmma_prefill || Q->ne[1] >= v4_144_pwmma_min_nq;
        const bool pure_prefill_inst = inst == GGML_FATTN_INST_NONE || inst == GGML_FATTN_INST_PREFILL_QK;
        const bool packed8_pwmma_candidate_requested =
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED8_Q4_144_PWMMA_PREFILL_CANDIDATE") ||
            ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL");
        const int packed8_pwmma_min_nq = getenv("GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NQ") ?
            std::max(1, atoi(getenv("GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NQ"))) : 16;
        const int packed8_pwmma_min_nk = getenv("GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NK") ?
            std::max(1, atoi(getenv("GGML_CUDA_ROCM_PACKED8_Q4_PWMMA_PREFILL_MIN_NK"))) : 512;
        const bool packed8_pwmma_prefill =
            packed8_pwmma_candidate_requested && packed8_q4_144_k_format && pure_prefill_inst &&
            Q->ne[1] >= packed8_pwmma_min_nq && K->ne[1] >= packed8_pwmma_min_nk;
        const bool k_wmma_shape_ok = K->type != GGML_TYPE_I32 || K->ne[0] * 4 == Q->ne[0] || packed8_pwmma_prefill;
        const bool wmma_sup = (!v4_k16d16_v || (v4_144_pwmma_prefill && v4_144_pwmma_nq_ok)) &&
            k_wmma_shape_ok &&
            ggml_cuda_packed16_wmma_tile_enabled();
        best_fattn_kernel packed16_kernel = BEST_FATTN_KERNEL_NONE;
        const char * packed16_route_name = "none";

        if (prefill_or_verify && dot4_sup) {
            // ── Packed16 PWMMA auto-selection ──────────
            // Use DBV PV-WMMA for long-context prefill; keep BM32 direct-V for shorter chunks.
            const char * explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
            const bool impl_autoset =
                getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET") &&
                atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET")) != 0;
            const bool force_smem_wmma = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_FORCE_SMEM");
            const bool impl_is_wmma =
                explicit_impl && ((force_smem_wmma && strcmp(explicit_impl, "smem") == 0) ||
                    strcmp(explicit_impl, "bm32_regout_directv") == 0 ||
                    strcmp(explicit_impl, "bm32_regout_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_regout_directv") == 0 ||
                    strcmp(explicit_impl, "bm64_regout_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_regout_directv_512t") == 0 ||
                    strcmp(explicit_impl, "bm64_512t_wavegate_directv") == 0 ||
                    strcmp(explicit_impl, "bm64_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_512t_wavegate_stagev_kshared") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_p16_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_k32acc_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_kshared_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_k32acc_kshared_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_pvwmma_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev") == 0);
            const bool impl_auto = !force_smem_wmma && (impl_autoset || !explicit_impl || !*explicit_impl || strcmp(explicit_impl, "smem") == 0);
            // Qwen 27B-like shape: gqa_ratio=6, heads_q=24, heads_k=4.
            // Qwen 35B-like shape: gqa_ratio=8, heads_q=16, heads_k=2.
            // A/B showed BM32 reg-out direct-V already wins at pp512 for these
            // target shapes; do not leave pp512 on the underfilled PDMQ prefill path.
            const bool is_27b_like = (gqa_ratio == 6 && K->ne[1] >= 512);
            const bool is_35b_like = (gqa_ratio == 8 && K->ne[1] >= 512);
            // General/unknown shapes keep the more conservative threshold at
            // nk >= 1024 (context length, not query chunk size).
            const bool is_big_q = (Q->ne[1] >= 16);
            const bool is_long_context = (K->ne[1] >= 1024);
            const bool is_pvwmma_context = (K->ne[1] >= 1024);
            const bool wmma_available = wmma_sup;
            const char * dbv_auto_env = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_DBV_AUTO");
            const bool dbv_auto_enabled = !dbv_auto_env || atoi(dbv_auto_env) != 0;
            const bool kshared_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_DBV_KSHARED_PREFILL");
            const bool streamk_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_STREAMK_PREFILL");
            const bool bn32_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_BN32_PREFILL");
            const bool use_kshared_pvwmma = kshared_prefill_requested && impl_auto && wmma_available && pure_prefill_inst &&
                !packed8_pwmma_prefill && is_pvwmma_context;
            const bool use_bn32_pvwmma = bn32_prefill_requested && !use_kshared_pvwmma && impl_auto && wmma_available && pure_prefill_inst &&
                !packed8_pwmma_prefill && is_pvwmma_context;
            const bool use_streamk_pvwmma = streamk_prefill_requested && !use_kshared_pvwmma && !use_bn32_pvwmma && impl_auto && wmma_available && pure_prefill_inst &&
                !packed8_pwmma_prefill && is_pvwmma_context;
            const bool use_pvwmma = use_kshared_pvwmma || use_bn32_pvwmma || (dbv_auto_enabled && impl_auto && wmma_available && (is_pvwmma_context || packed8_pwmma_prefill)) || use_streamk_pvwmma;
            const bool use_bm32_regout = !packed8_pwmma_prefill && !use_pvwmma && impl_auto && wmma_available &&
                (v4_144_nomtp_prefill || v4_144_mtp_pwmma || is_big_q || is_27b_like || is_35b_like || is_long_context);
            const bool use_auto_wmma = use_pvwmma || use_bm32_regout;

            // IMPL is process-global because the launcher reads it, but auto-set
            // values remain mutable: short chunks can use BM32 and later long-context
            // chunks can promote to DBV. A user-supplied IMPL has no AUTOSET marker
            // and stays fixed.
            if (use_auto_wmma || (impl_is_wmma && wmma_available)) {
                if (use_pvwmma) {
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL",
                        use_kshared_pvwmma ? "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev" :
                            (use_bn32_pvwmma ? "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev" :
                                (use_streamk_pvwmma ? "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev" :
                                    (packed8_pwmma_prefill ? "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev" : "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev"))), 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET", "1", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "64", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                } else if (use_bm32_regout) {
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL", "bm32_regout_directv", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET", "1", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "32", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                }
                packed16_kernel = BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
                if (use_pvwmma) {
                    packed16_route_name = use_kshared_pvwmma ? "pwmma_bm64_i8qk_pvwmma_dbv_kshared" :
                        (use_bn32_pvwmma ? "pwmma_bm64_i8qk_pvwmma_bn32" :
                            (use_streamk_pvwmma ? "pwmma_bm64_i8qk_pvwmma_dbv_streamk" :
                                (packed8_pwmma_prefill ? "pwmma_bm64_i8qk_packed8_expand_pvwmma_dbv" : "pwmma_bm64_i8qk_pvwmma_dbv")));
                } else if (use_bm32_regout) {
                    packed16_route_name = "pwmma_bm32_regout_directv";
                } else {
                    explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
                    if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_packed8_expand_pvwmma_dbv";
                    } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv_streamk";
                    } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_pvwmma_bn32";
                    } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv_kshared";
                    } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv";
                    } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_512t_wavegate_stagev") == 0) {
                        packed16_route_name = "pwmma_bm64_i8qk_pvwmma";
                    } else {
                        const int bm = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM") ? atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM")) : 32;
                        packed16_route_name = (bm == 16) ? "pwmma_bm16" : (bm == 32 ? "pwmma_bm32" : "pwmma_bm64");
                    }
                }
            } else {
                packed16_kernel = BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
                packed16_route_name = "dot4_mmq_gqa1";
            }
        } else if (prefill_or_verify && wmma_sup) {
            packed16_kernel = BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
            // Auto-select DBV PV-WMMA for long context and BM32 regout for short context.
            // GGML_CUDA_ROCM_PACKED16_WMMA_DBV_AUTO=0 keeps the conservative BM32 route.
            const char * explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
            const bool impl_autoset =
                getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET") &&
                atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET")) != 0;
            const bool force_smem_wmma = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_FORCE_SMEM");
            const bool impl_auto = !force_smem_wmma && (impl_autoset || !explicit_impl || !*explicit_impl || strcmp(explicit_impl, "smem") == 0);
            if (impl_auto) {
                const char * dbv_auto_env = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_DBV_AUTO");
                const bool dbv_auto_enabled = !dbv_auto_env || atoi(dbv_auto_env) != 0;
                const bool kshared_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_DBV_KSHARED_PREFILL");
                const bool streamk_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_STREAMK_PREFILL");
                const bool bn32_prefill_requested = ggml_cuda_env_enabled_name("GGML_CUDA_ROCM_PACKED16_WMMA_BN32_PREFILL");
                const bool use_kshared_pvwmma = kshared_prefill_requested && pure_prefill_inst && !packed8_pwmma_prefill && K->ne[1] >= 1024;
                const bool use_bn32_pvwmma = bn32_prefill_requested && !use_kshared_pvwmma && pure_prefill_inst && !packed8_pwmma_prefill && K->ne[1] >= 1024;
                const bool use_streamk_pvwmma = streamk_prefill_requested && !use_kshared_pvwmma && !use_bn32_pvwmma && pure_prefill_inst && !packed8_pwmma_prefill && K->ne[1] >= 1024;
                if (packed8_pwmma_prefill || use_kshared_pvwmma || use_bn32_pvwmma || use_streamk_pvwmma || (dbv_auto_enabled && K->ne[1] >= 1024)) {
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL",
                        use_kshared_pvwmma ? "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev" :
                            (use_bn32_pvwmma ? "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev" :
                                (use_streamk_pvwmma ? "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev" :
                                    (packed8_pwmma_prefill ? "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev" : "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev"))), 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET", "1", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "64", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                    packed16_route_name = use_kshared_pvwmma ? "pwmma_bm64_i8qk_pvwmma_dbv_kshared" :
                        (use_bn32_pvwmma ? "pwmma_bm64_i8qk_pvwmma_bn32" :
                            (use_streamk_pvwmma ? "pwmma_bm64_i8qk_pvwmma_dbv_streamk" :
                                (packed8_pwmma_prefill ? "pwmma_bm64_i8qk_packed8_expand_pvwmma_dbv" : "pwmma_bm64_i8qk_pvwmma_dbv")));
                } else {
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL", "bm32_regout_directv", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL_AUTOSET", "1", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "32", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                    packed16_route_name = "pwmma_bm32_regout_directv";
                }
            } else {
                explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
                if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_packed8_expand_pvwmma_dbv_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_packed8_expand_pvwmma_dbv";
                } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_streamk_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv_streamk";
                } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_bn32_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_pvwmma_bn32";
                } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_kshared_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv_kshared";
                } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_pvwmma_dbv";
                } else if (explicit_impl && strcmp(explicit_impl, "bm64_i8qk_pvwmma_512t_wavegate_stagev") == 0) {
                    packed16_route_name = "pwmma_bm64_i8qk_pvwmma";
                } else {
                    const int bm = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM") ? atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM")) : 32;
                    packed16_route_name = (bm == 16) ? "pwmma_bm16" : (bm == 32 ? "pwmma_bm32" : "pwmma_bm64");
                }
            }
        } else if (ggml_cuda_q8k_dot4_kq_enabled() || getenv("GGML_CUDA_ROCM_PDMQ_K_CACHE") || getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE")) {
            packed16_kernel = BEST_FATTN_KERNEL_NONE;
            packed16_route_name = "dot4_kq_removed";
        } else {
            packed16_kernel = BEST_FATTN_KERNEL_NONE;
            packed16_route_name = "none";
        }

        if (auto_verbose && g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
            fprintf(stderr, "%s: PACKED16 FA ROUTE nq=%lld nk=%lld D=%lld hq=%lld hk=%lld gqa=%d"
                " K=I32 k_format=%s V=%s auto=1 selected=%s"
                " dot4_mmq_sup=%d wmma_sup=%d forced_route=%s\n",
                __func__,
                (long long)Q->ne[1], (long long)K->ne[1], (long long)Q->ne[0],
                (long long)Q->ne[2], (long long)K->ne[2], gqa_ratio,
                packed8_q4_144_k_format ? "packed8_q4_144" : "packed16_q8_272", ggml_type_name(V->type),
                packed16_route_name,
                dot4_sup ? 1 : 0, wmma_sup ? 1 : 0,
                required_route ? required_route : "none");
        }

        return packed16_kernel;
    }

    // ── Instruction dispatch: try BEFORE shape/type switch ──────────
    // Fires for ALL K+V types including same-type f16/f16, not just mixed KV.
    // If instruction fastpath rejects (e.g., FULL_FA=0), falls through to
    // existing VEC/TILE/CPU paths — no NONE gap.
    {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        if (fa_inst_i32 != GGML_FATTN_INST_NONE) {
#ifdef GGML_USE_HIP
            const ggml_fattn_instruction inst = (ggml_fattn_instruction)fa_inst_i32;
            ggml_cuda_fa_fastpath_result fast =
                ggml_cuda_try_instruction_fastpath(cc, dst, inst);

            if (fast.selected != BEST_FATTN_KERNEL_NONE) {
                return fast.selected;
            }

            // Fast path rejected. Save context for hunter hooks.
            g_fa_inst_tracker.active = true;
            g_fa_inst_tracker.inst = inst;
            g_fa_inst_tracker.fast = fast;
#endif // GGML_USE_HIP
        }
    }

#ifdef GGML_USE_HIP
    // Live MTP verifier extension rows can arrive as inst=NONE even when the
    // reserve/proof graph stamped MTP_VERIFY_QK.  If the user explicitly asks
    // for the packed16 small-Q route and the graph-facing F16 K has a registered
    // hot/cold packed16 sidecar, select the packed16 decode family before the
    // generic mixed-KV rejection sends F16-K with standard PV4/V144 to fallback.
    // Legacy q4_0 V hot/cold remains opt-in via LLAMA_MTP_PACKED16_HOTCOLD_K_LEGACY_Q4V.
    {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        const bool require_pdmq = required_route &&
            (strcmp(required_route, "rocm_packed16_dot4_mmq") == 0 ||
             strcmp(required_route, "packed16_dot4_mmq") == 0);
        const bool opt_in_pdmq = getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ") &&
            atoi(getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_PDMQ")) != 0;
        const bool require_smallq = ggml_cuda_fattn_route_contract_is_packed16_small_verify(required_route);
        const char * smallq_env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_SMALLQ_FA2");
        const bool opt_in_smallq = smallq_env && atoi(smallq_env) != 0;
        const bool smallq_inst =
            fa_inst_i32 == GGML_FATTN_INST_NONE ||
            ggml_cuda_fattn_inst_is_mtp_verify_qk(fa_inst_i32);
        const bool sidecar_applicable_relaxed =
            smallq_inst && ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, false);
        const bool sidecar_supported =
            smallq_inst && ggml_cuda_mtp_hotcold_packed16_small_verify_contract_applicable(dst, true);
        if ((require_pdmq || opt_in_pdmq) && sidecar_applicable_relaxed && !sidecar_supported) {
            const ggml_tensor * Q_req = dst->src[0];
            const ggml_tensor * K_req = dst->src[1];
            const ggml_tensor * V_req = dst->src[2];
            GGML_ABORT("required packed16 DOT4-MMQ F16-sidecar route is cold-blocked; route=%s nq=%lld nk=%lld K=%s V=%s window=%lld cold_unsafe=%d",
                required_route ? required_route : "rocm_packed16_dot4_mmq",
                Q_req ? (long long) Q_req->ne[1] : -1LL,
                K_req ? (long long) K_req->ne[1] : -1LL,
                K_req ? ggml_type_name(K_req->type) : "none",
                V_req ? ggml_type_name(V_req->type) : "none",
                (long long) ggml_cuda_mtp_packed16_hotcold_k_window(),
                ggml_cuda_mtp_packed16_hotcold_cold_unsafe_enabled() ? 1 : 0);
        }
        if ((require_pdmq || opt_in_pdmq) && sidecar_supported && ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
            return ggml_cuda_fattn_apply_route_contract(dst, BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ, nullptr);
        }
        if (require_smallq && sidecar_applicable_relaxed && !sidecar_supported) {
            const ggml_tensor * Q_req = dst->src[0];
            const ggml_tensor * K_req = dst->src[1];
            const ggml_tensor * V_req = dst->src[2];
            GGML_ABORT("required packed16 small-Q FA2 F16-sidecar route is cold-blocked; route=%s nq=%lld nk=%lld K=%s V=%s window=%lld cold_unsafe=%d",
                required_route,
                Q_req ? (long long) Q_req->ne[1] : -1LL,
                K_req ? (long long) K_req->ne[1] : -1LL,
                K_req ? ggml_type_name(K_req->type) : "none",
                V_req ? ggml_type_name(V_req->type) : "none",
                (long long) ggml_cuda_mtp_packed16_hotcold_k_window(),
                ggml_cuda_mtp_packed16_hotcold_cold_unsafe_enabled() ? 1 : 0);
        }
        if ((require_smallq || opt_in_smallq) && sidecar_supported &&
                ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
        }
    }
#endif // GGML_USE_HIP

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_mixed_kv_supported(Q, K, V)) {
        // Instruction paths with legal DOT4 K-representation adapters
        // can handle mixed KV (e.g., K=f16,V=q4_0 → source f16 → packed16).
        // Check instruction dispatch BEFORE rejecting mixed KV.
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        if (fa_inst_i32 != GGML_FATTN_INST_NONE) {
            // Let instruction dispatch try. If it selects non-NONE,
            // return it and skip the mixed KV rejection.
#ifdef GGML_USE_HIP
            const ggml_fattn_instruction inst = (ggml_fattn_instruction)fa_inst_i32;
            ggml_cuda_fa_fastpath_result fast =
                ggml_cuda_try_instruction_fastpath(cc, dst, inst);

            if (fast.selected != BEST_FATTN_KERNEL_NONE) {
                return fast.selected;
            }

            // Fast path rejected. Save context for hunter hooks.
            g_fa_inst_tracker.active = true;
            g_fa_inst_tracker.inst = inst;
            g_fa_inst_tracker.fast = fast;
#endif // GGML_USE_HIP
        }

        // Instruction path also couldn't handle it. Reject.
        ggml_cuda_fattn_log_mixed_kv_reject(Q, K, V);
        return BEST_FATTN_KERNEL_NONE;
    }

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_TBQ4_0:
            return BEST_FATTN_KERNEL_NONE;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

#ifdef GGML_USE_HIP
    // ── Instruction-driven FA dispatch (HIP-only legacy block) ──
    // For instruction paths: handled upstream at the mixed KV gate.
    // For MTP_DRAFT: log and fall through to existing policy.
    {
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        
        // MTP_DRAFT only: log and continue (no DOT4 preference).
        // Other instruction paths were handled before mixed KV rejection.
        if (fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT) {
            if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
                if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
                    if (log_env && atoi(log_env) != 0) {
                        GGML_LOG_INFO("%s: fa_instruction=mtp_draft nq=%lld impl_status=legacy_fallback\n",
                                __func__, (long long) Q->ne[1]);
                    }
                }
            }
        }
    }

    // HIP/ROCm: keep quantized KV one/two-token decode on VEC. The f16-temp
    // TILE/MMA selector is opt-in only: GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
    // for promoted paths such as 27B MTP, or GGML_CUDA_ROCM_QUANT_PREFILL_F16_AUTO=1
    // for explicit A/B probes. Absence of both envs must preserve q8K/tbq4V VEC
    // so 35B prefill cannot silently sink into the f16-temp path.
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel) {
        ggml_cuda_rocm_quant_prefill_f16_policy f16_policy =
            ggml_cuda_fattn_rocm_quant_prefill_f16_policy(Q, K, V);
        const ggml_cuda_rocm_quant_prefill_f16_mode f16_mode = ggml_cuda_rocm_quant_prefill_f16_mode_from_env();
        const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        const bool require_i8_route  = ggml_cuda_fattn_route_contract_is_i8(required_route);
        const bool require_f16_route = ggml_cuda_fattn_route_contract_is_f16_temp(required_route);
        const bool f16_prefill_applicable = ggml_cuda_fattn_route_contract_applicable("f16_temp", dst, &f16_policy);
        const bool stable_quant_f16_prefill =
            f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_ALLOW &&
            !require_i8_route && f16_prefill_applicable &&
            Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 &&
            Q->ne[1] > 2;
        if ((stable_quant_f16_prefill || require_f16_route ||
             f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE ||
             f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER) && f16_prefill_applicable) {
            f16_policy.forced = require_f16_route || f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE;
            f16_policy.automatic = !f16_policy.forced &&
                (f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER || stable_quant_f16_prefill);
            f16_policy.allowed = f16_policy.contiguous_ok && f16_policy.under_budget;
        }
        const bool allow_quant_prefill_f16 = f16_mode != GGML_CUDA_ROCM_QUANT_PREFILL_F16_OFF && f16_policy.allowed;
        if (allow_quant_prefill_f16) {
            ggml_cuda_fattn_log_rocm_quant_prefill_f16_once(Q, K, V, f16_policy);
        } else {
            ggml_cuda_fattn_log_rocm_quant_prefill_f16_once(Q, K, V, f16_policy);
        }
        const auto return_quantized_route = [&](best_fattn_kernel selected) {
            return ggml_cuda_fattn_apply_route_contract(dst, selected, &f16_policy);
        };
        const bool raw_q8_standard_vec = K->type == GGML_TYPE_Q8_0 &&
            (V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0);

        if (!require_f16_route && ggml_cuda_fattn_mtp_hidden_producer_use_conservative_route(dst)) {
            best_fattn_kernel producer_kernel = (raw_q8_standard_vec && !ggml_cuda_standard_q8_fattn_vec_compiled()) ?
                BEST_FATTN_KERNEL_NONE : BEST_FATTN_KERNEL_VEC;
            const char * producer_reason = producer_kernel == BEST_FATTN_KERNEL_NONE ?
                "producer_dot4_disabled_q8_standard_vec_not_compiled" : "producer_dot4_disabled_vec";
            if (allow_quant_prefill_f16 && f16_prefill_applicable) {
                const best_fattn_kernel f16_backend = ggml_cuda_fattn_select_rocm_quant_prefill_f16_backend(cc, dst);
                if (f16_backend != BEST_FATTN_KERNEL_NONE) {
                    producer_kernel = f16_backend;
                    producer_reason = "producer_dot4_disabled_exact_f16_temp";
                }
            }
            if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
                if (log_env && atoi(log_env) != 0) {
                    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
                    GGML_LOG_INFO("%s: mtp_hidden_producer_policy phase=producer inst=%s selected=%s reason=%s K=%s k_phys=%s V=%s nq=%lld nkv=%lld f16_allowed=%d enable_env=LLAMA_MTP_ENABLE_PRODUCER_DOT4_FA2\n",
                            __func__,
                            ggml_cuda_fattn_instruction_name((ggml_fattn_instruction) fa_inst_i32),
                            ggml_cuda_fattn_kernel_name(producer_kernel), producer_reason,
                            ggml_type_name(K->type), ggml_cuda_fattn_k_physical_repr_name(K), ggml_type_name(V->type),
                            (long long) Q->ne[1], (long long) K->ne[1],
                            allow_quant_prefill_f16 ? 1 : 0);
                }
            }
            return return_quantized_route(producer_kernel);
        }

        if (!require_f16_route && ggml_cuda_standard_q8_fattn_vec_compiled() && ggml_cuda_q8k_dot4_packed16_vec_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC);
        }

        if (!require_f16_route && ggml_cuda_q8k_dot4_kq_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_NONE);
        }

        if (require_f16_route && f16_prefill_applicable) {
            if (!allow_quant_prefill_f16) {
                GGML_ABORT("requested ROCm f16 prefill route was rejected: %s",
                        ggml_cuda_rocm_quant_prefill_f16_policy_reason_name(f16_policy));
            }
        }

        if (!require_f16_route && !require_i8_route && f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER &&
                f16_prefill_applicable && allow_quant_prefill_f16) {
            // prefer/require modes are explicit selector overrides; choose a
            // launchable f16 backend immediately instead of just falling through.
            best_fattn_kernel f16_backend = ggml_cuda_fattn_select_rocm_quant_prefill_f16_backend(cc, dst);
            if (f16_backend == BEST_FATTN_KERNEL_NONE) {
                GGML_ABORT("requested ROCm f16 prefill route was rejected: no_launchable_f16_backend");
            }
            return return_quantized_route(f16_backend);
        }

        if (!allow_quant_prefill_f16) {
            if (raw_q8_standard_vec && !ggml_cuda_standard_q8_fattn_vec_compiled()) {
                return return_quantized_route(BEST_FATTN_KERNEL_NONE);
            }
            return return_quantized_route(BEST_FATTN_KERNEL_VEC);
        }

        // allow mode with GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 restores the old
        // behavior: skip quantized VEC for long prefill and fall through to the
        // existing full f16 FlashAttention selector below.
    }
#endif // GGML_USE_HIP


    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // Use MFMA flash attention for CDNA (MI100+) for other supported head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const bool fattn_dispatch_trace =
        (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")) != 0) ||
        (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE")) != 0);
    if (fattn_dispatch_trace) {
        setbuf(stderr, NULL);
    }
    // ── Entry proof: every FA op must pass through here ──────────
    {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
            if (log_env && atoi(log_env) != 0) {
                const char * qblock_active = getenv("LLAMA_MTP_QBLOCK_ACTIVE");
                const char * qblock_nq     = getenv("LLAMA_MTP_QBLOCK_NQ");
                GGML_LOG_INFO(
                    "fa_entry: node=%s layer=%d inst=%s nq=%lld nk=%lld qblock=%d qblock_nq_env=%s\n",
                    ggml_cuda_fattn_node_name(dst),
                    ggml_cuda_fattn_layer_from_node_name(dst),
                    ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                    (long long) Q->ne[1],
                    (long long) K->ne[1],
                    qblock_active && atoi(qblock_active) != 0 ? 1 : 0,
                    qblock_nq ? qblock_nq : "-");
            }
        }
    }

    g_fattn_select_ctx = GGML_CUDA_FATTN_SELECT_DISPATCH;
#ifdef GGML_USE_HIP
    {
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(ctx.stream(), &capture_status) == hipSuccess) {
            g_fattn_capture_ctx = { true, capture_status != hipStreamCaptureStatusNone };
        } else {
            g_fattn_capture_ctx = { false, false };
        }
    }
#else
    g_fattn_capture_ctx = { false, false };
#endif
    if (fattn_dispatch_trace) {
        fprintf(stderr, "FATTN COMPUTE ENTER dst=%p\n", (void*)dst); fflush(stderr);
    }
    const best_fattn_kernel best_kernel = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    const int implicit_flags = ggml_get_op_params_i32(dst, 7);
    if ((implicit_flags & 1) &&
            best_kernel != BEST_FATTN_KERNEL_PACKED16_WMMA_TILE &&
            best_kernel != BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ) {
        GGML_ABORT("implicit causal FA mask metadata requires a metadata-aware packed16 dispatch; selected=%s node=%s",
            ggml_cuda_fattn_kernel_name(best_kernel), ggml_cuda_fattn_node_name(dst));
    }
    if (fattn_dispatch_trace) {
        fprintf(stderr, "FATTN COMPUTE SELECT selected=%d name=%s dst=%p\n", (int)best_kernel, ggml_cuda_fattn_kernel_name(best_kernel), (void*)dst); fflush(stderr);
    }
    ggml_cuda_fattn_log_selection(best_kernel, dst);
    ggml_cuda_dp16_fa_enforce_final_packed16_contract(dst, best_kernel);

#ifdef GGML_USE_HIP
    // Final dispatch-level proof for packed16 small-Q FA2 routes. Some of the
    // generic packed16 selector returns happen after the instruction fast-path
    // and do not have an f16-policy object to thread through; enforce the
    // product-visible small-Q contract here instead of relying on route logs.
    if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
        const char * required_route = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
        if (ggml_cuda_fattn_route_contract_is_packed16_small_verify(required_route) &&
                ggml_cuda_fattn_route_contract_applicable(required_route, dst, nullptr)) {
            if (!ggml_cuda_fattn_route_contract_matches(required_route, best_kernel)) {
                ggml_cuda_fattn_log_route_contract(required_route, "rejected", best_kernel, dst, nullptr);
                GGML_ABORT("required packed16 small-Q FA2 route %s not selected; got %s",
                    required_route, ggml_cuda_fattn_kernel_name(best_kernel));
            }
            ggml_cuda_fattn_log_route_contract(required_route, "selected", best_kernel, dst, nullptr);
        }
    }
#endif // GGML_USE_HIP

    // ── VEC/TILE hunter hook ───────────────────────────────────
    // If an instruction path fell through to VEC/TILE, the dispatch
    // saved context in g_fa_inst_tracker. Now log debt + optionally abort.
    if (g_fa_inst_tracker.active) {
        ggml_cuda_fa_log_vec_tile_debt(dst, g_fa_inst_tracker.inst,
            g_fa_inst_tracker.fast, best_kernel);
        ggml_cuda_fa_abort_on_vec_tile(dst, g_fa_inst_tracker.inst,
            g_fa_inst_tracker.fast, best_kernel);
        g_fa_inst_tracker.active = false;
    }

    // ── Final select proof ─────────────────────────────────────
    // Only log in dispatch context (not during support probe).
    if (g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const char * qblock_active_env = getenv("LLAMA_MTP_QBLOCK_ACTIVE");
        const bool qblock_active = qblock_active_env && atoi(qblock_active_env) != 0;
        if (qblock_active) {
            const char * qblock_nq_env = getenv("LLAMA_MTP_QBLOCK_NQ");
            const long long qblock_expected_nq = qblock_nq_env && qblock_nq_env[0] ? atoll(qblock_nq_env) : -1LL;
            const long long qblock_actual_nq = Q ? (long long) Q->ne[1] : -1LL;
            const bool qblock_inst = fa_inst_i32 == GGML_FATTN_INST_MTP_QBLOCK_VERIFY_QK;
            const bool qblock_nq_ok = qblock_actual_nq > 1 && (qblock_expected_nq <= 0 || qblock_actual_nq == qblock_expected_nq);
            const char * qblock_exclude_layer_env = getenv("LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_BATCH_EXCLUDE_LAYER");
            const bool qblock_mixed_prefix = qblock_exclude_layer_env && qblock_exclude_layer_env[0] != '\0' && atoi(qblock_exclude_layer_env) >= 0;
            const char * qblock_boundary_env = getenv("LLAMA_MTP_QBLOCK_PREFIX_FULL_ATTN_BOUNDARY_COMPARE");
            const bool qblock_boundary_compare = qblock_boundary_env && qblock_boundary_env[0] != '\0' && atoi(qblock_boundary_env) != 0;
            const bool qblock_row_serial_ok = (qblock_mixed_prefix || qblock_boundary_compare) && !qblock_inst && qblock_actual_nq == 1;
            const bool qblock_invariant_ok = qblock_row_serial_ok || (qblock_inst && qblock_nq_ok);
            const bool qblock_strict = getenv("LLAMA_MTP_QBLOCK_STRICT") && atoi(getenv("LLAMA_MTP_QBLOCK_STRICT")) != 0;
            const bool qblock_trace = getenv("LLAMA_MTP_QBLOCK_TRACE") && atoi(getenv("LLAMA_MTP_QBLOCK_TRACE")) != 0;
            if (qblock_trace || !qblock_invariant_ok) {
                fprintf(stderr,
                        "MTP_QBLOCK_FA_INVARIANT: inst=%s node=%s layer=%d selected=%s expected_nq=%lld actual_nq=%lld nk=%lld qblock_inst=%d nq_ok=%d mixed_ok=%d strict=%d\n",
                        ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                        ggml_cuda_fattn_node_name(dst),
                        ggml_cuda_fattn_layer_from_node_name(dst),
                        ggml_cuda_fattn_kernel_name(best_kernel),
                        qblock_expected_nq,
                        qblock_actual_nq,
                        K ? (long long) K->ne[1] : -1LL,
                        qblock_inst ? 1 : 0,
                        qblock_nq_ok ? 1 : 0,
                        qblock_row_serial_ok ? 1 : 0,
                        qblock_strict ? 1 : 0);
            }
            if (qblock_strict && !qblock_invariant_ok) {
                GGML_ABORT("MTP QBlock FA invariant failed: inst=%s expected_nq=%lld actual_nq=%lld selected=%s",
                        ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                        qblock_expected_nq,
                        qblock_actual_nq,
                        ggml_cuda_fattn_kernel_name(best_kernel));
            }
        }
        if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
            if (log_env && atoi(log_env) != 0) {
                const char * qblock_active = getenv("LLAMA_MTP_QBLOCK_ACTIVE");
                const char * qblock_nq     = getenv("LLAMA_MTP_QBLOCK_NQ");
                GGML_LOG_INFO(
                    "fa_final_select: node=%s layer=%d inst=%s selected=%s nq=%lld nk=%lld d=%lld K=%s V=%s final=1 qblock=%d qblock_nq_env=%s qblock_role=%s\n",
                    ggml_cuda_fattn_node_name(dst),
                    ggml_cuda_fattn_layer_from_node_name(dst),
                    ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                    ggml_cuda_fattn_kernel_name(best_kernel),
                    Q ? (long long) Q->ne[1] : -1LL,
                    K ? (long long) K->ne[1] : -1LL,
                    Q ? (long long) Q->ne[0] : -1LL,
                    K ? ggml_type_name(K->type) : "-",
                    V ? ggml_type_name(V->type) : "-",
                    qblock_active && atoi(qblock_active) != 0 ? 1 : 0,
                    qblock_nq ? qblock_nq : "-",
                    qblock_active && atoi(qblock_active) != 0 ? "mtp_qblock_verify" : "-");
            }
        }
    }

    if (fattn_dispatch_trace) {
        fprintf(stderr, "PWMMA DISPATCH selected=%d name=%s\n", (int) best_kernel, ggml_cuda_fattn_kernel_name(best_kernel));
    }

    switch (best_kernel) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8K_DOT4_KQ:
        case BEST_FATTN_KERNEL_PACKED16_DECODE:
            ggml_cuda_flash_attn_ext_q8k_dot4_kq(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC:
        case BEST_FATTN_KERNEL_MTP_F16K_Q4V_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_PACKED16_FA2_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_PACKED16_WMMA_TILE:
            ggml_cuda_flash_attn_ext_packed16_wmma_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:
            ggml_cuda_flash_attn_ext_packed16_dot4_mmq(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    g_fattn_select_ctx = GGML_CUDA_FATTN_SELECT_SUPPORT_PROBE;
    g_fattn_capture_ctx = { false, false };
    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    const ggml_tensor * Q = dst->src[0];
    const best_fattn_kernel k = ggml_cuda_get_best_fattn_kernel(device, dst);
    const bool result = k != BEST_FATTN_KERNEL_NONE;
    if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
        if (log_env && atoi(log_env) != 0) {
            GGML_LOG_INFO("fa_supported: inst=%d nq=%lld kernel=%d result=%d\n",
                fa_inst_i32, (long long) Q->ne[1], (int)k, (int)result);
        }
    }
    return result;
}
