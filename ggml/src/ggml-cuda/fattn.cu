#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-tbq4-launch.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn-wmma-q8q4-i8.cuh"
#include "fattn-dot4-q8q4.cuh"
#include "fattn-dot4-q8k-kq.cuh"
#include "fattn-packed16-wmma-tile.cuh"
#include "fattn-packed16-dot4-mmq.cuh"
void ggml_cuda_flash_attn_ext_wmma_tbq4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_wmma_compressed_kv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#include "cpy-planar-iso.cuh"
#include "fattn.cuh"

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
    // Planar/IsoQuant matched pairs + mixed with F16/Q4_0/Q8_0
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR3_0, GGML_TYPE_PLANAR3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0,    GGML_TYPE_ISO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR4_0, GGML_TYPE_PLANAR4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO4_0,    GGML_TYPE_ISO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,    GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,    GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_PLANAR3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_ISO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_PLANAR4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_ISO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0,    GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO4_0,    GGML_TYPE_F16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,       GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0,      GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_Q8_0)
    // Stable no-TBQ public path: q8_0 K + q4_0 V should run without
    // GGML_CUDA_FA_ALL_QUANTS or lab/unsafe ROCm route knobs.
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16,      GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR3_0, GGML_TYPE_PLANAR3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO3_0,    GGML_TYPE_ISO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_PLANAR4_0, GGML_TYPE_PLANAR4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_ISO4_0,    GGML_TYPE_ISO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,    GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TBQ4_0,    GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0,      GGML_TYPE_TBQ4_0)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
    BEST_FATTN_KERNEL_MMA_TBQ4 = 500,
    BEST_FATTN_KERNEL_WMMA_TBQ4 = 550, // AMD rocWMMA TBQ4 path (experimental, disabled in favor of VEC)
    BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV = 560, // experimental Planar/Iso compressed-KV WMMA path
    BEST_FATTN_KERNEL_Q8Q4_WMMA_I8 = 570, // experimental ROCm q8_0 K + q4_0 V direct i8-WMMA path
    BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL = 580, // experimental ROCm packed-dot4 i8 QK inside FA
    BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL = 581, // experimental ROCm packed-dot4 i8 QK + TBQ4 V
    BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL = 582, // experimental ROCm TBQ4 packed-dot4 QK inside FA
    BEST_FATTN_KERNEL_Q8K_DOT4_KQ = 583, // experimental ROCm KQ-only packed16 DOT4 probe
    BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC = 584, // experimental ROCm packed16 q8_0-K shadow inside stable VEC FA
    BEST_FATTN_KERNEL_PACKED16_WMMA_TILE = 585, // packed16 K cache → f16 tile materializer → WMMA/MFMA FA
    BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ = 586, // packed16 K cache → q8 Q tile → sudot4/MMQ FA
};

static const char * ggml_cuda_fattn_instruction_name(const ggml_fattn_instruction inst);

static const char * ggml_cuda_fattn_kernel_name(const best_fattn_kernel kernel) {
    switch (kernel) {
        case BEST_FATTN_KERNEL_NONE:               return "none";
        case BEST_FATTN_KERNEL_TILE:               return "tile";
        case BEST_FATTN_KERNEL_VEC:                return "vec";
        case BEST_FATTN_KERNEL_WMMA_F16:           return "wmma_f16";
        case BEST_FATTN_KERNEL_MMA_F16:            return "mma_f16";
        case BEST_FATTN_KERNEL_MMA_TBQ4:           return "mma_tbq4";
        case BEST_FATTN_KERNEL_WMMA_TBQ4:          return "wmma_tbq4";
        case BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV: return "wmma_compressed_kv";
        case BEST_FATTN_KERNEL_Q8Q4_WMMA_I8:       return "q8q4_wmma_i8";
        case BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL:  return "q8q4_dot4_prefill";
        case BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL:return "q8tbq4_dot4_prefill";
        case BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL:  return "tbq4_dot4_prefill";
        case BEST_FATTN_KERNEL_Q8K_DOT4_KQ:        return "rocm_q8k_dot4_kq";
        case BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC:return "rocm_q8k_dot4_packed16_vec";
        case BEST_FATTN_KERNEL_PACKED16_WMMA_TILE:    return "rocm_packed16_wmma_tile";
        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:       return "rocm_packed16_dot4_mmq";
    }
    return "unknown";
}

static int32_t ggml_cuda_fattn_get_instruction(const ggml_tensor * dst) {
    return ((const int32_t *)dst->op_params)[4];
}

// PR3: f16-source K op-local materialization.
// When enabled, MTP_VERIFY_QK + source K=f16 will quantize K into a
// temporary packed16 representation inside the DOT4 launch path, without
// altering the persistent MTP KV cache.
//
// Legal K representations for DOT4 recthist:
//   - persistent packed16 I32
//   - q8_0/q4_0
//   - source f16 materialized op-locally into packed16
static bool ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_F16K_DOT4_ADAPTER");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

// ── MTP_VERIFY_QK WMMA/MMA fallback env gates ──────────────────────
// Default off. Part 3: instruction-aware fallback when DOT4 is disabled
// or illegal. WMMA/MMA are f16/f32 only — no q8/q4 dequant fallback.

static bool ggml_cuda_mtp_verify_wmma_f16_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_WMMA_F16");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_verify_mma_f16_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_VERIFY_MMA_F16");
    return env && atoi(env) != 0;
#else
    return false;
#endif
}

static bool ggml_cuda_mtp_verify_wmma_f16_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_mtp_verify_wmma_f16_enabled()) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V) {
        return false;
    }

    if (!ggml_cuda_should_use_wmma_fattn(cc)) {
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

    if (K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }

    // Existing WMMA f16 path excludes these dimensions.
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

static bool ggml_cuda_mtp_verify_mma_f16_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_mtp_verify_mma_f16_enabled()) {
        return false;
    }

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
    const int nq_min = (fa_inst_i32 == GGML_FATTN_INST_MTP_VERIFY_QK &&
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
    if ((strcmp(required, "rocm_q8_tbq4_f16_temp") == 0 ||
         strcmp(required, "rocm_q8q4_f16_temp") == 0 ||
         strcmp(required, "rocm_quant_prefill_f16") == 0 ||
         strcmp(required, "f16_temp") == 0) &&
            (kernel == BEST_FATTN_KERNEL_MMA_F16 || kernel == BEST_FATTN_KERNEL_WMMA_F16)) {
        return true;
    }
    if ((strcmp(required, "rocm_q8q4_wmma_i8") == 0 || strcmp(required, "q8q4_wmma_i8") == 0) &&
            kernel == BEST_FATTN_KERNEL_Q8Q4_WMMA_I8) {
        return true;
    }
    if (strcmp(required, "rocm_q8k_dot4_kq") == 0 && kernel == BEST_FATTN_KERNEL_Q8K_DOT4_KQ) {
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
    if (strcmp(required, "rocm_mtp_verify_wmma_f16") == 0 &&
            kernel == BEST_FATTN_KERNEL_WMMA_F16) {
        return true;
    }
    if (strcmp(required, "rocm_mtp_verify_mma_f16") == 0 &&
            kernel == BEST_FATTN_KERNEL_MMA_F16) {
        return true;
    }
    if (strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0 && kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) {
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
    if ((strcmp(required, "q8q4_dot4_prefill") == 0 || strcmp(required, "rocm_q8q4_dot4") == 0 ||
         strcmp(required, "q8q4_dot4") == 0) && kernel == BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL) {
        return true;
    }
    if ((strcmp(required, "q8tbq4_dot4_prefill") == 0 || strcmp(required, "rocm_q8_tbq4_dot4") == 0 ||
         strcmp(required, "q8tbq4_dot4") == 0) && kernel == BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL) {
        return true;
    }
    if ((strcmp(required, "tbq4_dot4_prefill") == 0 || strcmp(required, "rocm_tbq4_dot4") == 0 ||
         strcmp(required, "tbq4_dot4") == 0) && kernel == BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL) {
        return true;
    }
    return false;
}

static bool ggml_cuda_fattn_route_contract_is_f16_temp(const char * required) {
    return required && (strcmp(required, "rocm_q8_tbq4_f16_temp") == 0 ||
        strcmp(required, "rocm_q8q4_f16_temp") == 0 ||
        strcmp(required, "rocm_quant_prefill_f16") == 0 ||
        strcmp(required, "f16_temp") == 0);
}

static bool ggml_cuda_fattn_route_contract_is_i8(const char * required) {
    return required && (strcmp(required, "rocm_q8k_dot4_kq") == 0 ||
        strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0 ||
        strcmp(required, "rocm_packed16_dot4_mmq") == 0 ||
        strcmp(required, "packed16_dot4_mmq") == 0 ||
        strcmp(required, "q8q4_dot4_prefill") == 0 ||
        strcmp(required, "q8tbq4_dot4_prefill") == 0 ||
        strcmp(required, "tbq4_dot4_prefill") == 0 ||
        strcmp(required, "q8q4_wmma_i8") == 0 ||
        strcmp(required, "rocm_q8q4_dot4") == 0 ||
        strcmp(required, "q8q4_dot4") == 0 ||
        strcmp(required, "rocm_q8_tbq4_dot4") == 0 ||
        strcmp(required, "q8tbq4_dot4") == 0 ||
        strcmp(required, "rocm_tbq4_dot4") == 0 ||
        strcmp(required, "tbq4_dot4") == 0 ||
        strcmp(required, "rocm_q8q4_wmma_i8") == 0);
}

static bool ggml_cuda_q8q4_wmma_i8_require_selected_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_REQUIRE_SELECTED");
    return env && atoi(env) != 0;
#else
    return false;
#endif // GGML_USE_HIP
}

static bool ggml_cuda_q8q4_wmma_i8_require_selected_applies(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_q8q4_wmma_i8_require_selected_enabled() ||
            !ggml_cuda_q8q4_wmma_i8_enabled() ||
            !ggml_cuda_q8q4_wmma_i8_unsafe_enabled() ||
            !ggml_cuda_q8q4_wmma_i8_layer_filter_allows(dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return false;
    }

    // Layer-scoped fail-fast is only for the q8_0 K / q4_0 V WMMA-I8 lab lane.
    // Other quantized routes (notably q8_0/tbq4_0) must remain free to use their
    // own I8/DOT4/fallback policy without tripping this contract.
    return Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 &&
        dst->type == GGML_TYPE_F32 && Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 &&
        Q->ne[1] > 2;
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
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
        if (strcmp(required, "rocm_q8_tbq4_f16_temp") == 0) {
            return V->type == GGML_TYPE_TBQ4_0;
        }
        if (strcmp(required, "rocm_q8q4_f16_temp") == 0) {
            return V->type == GGML_TYPE_Q4_0;
        }
        return V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_TBQ4_0 || V->type == GGML_TYPE_Q8_0;
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
        // Legal K representations:
        //   - q8_0 K (original)
        //   - I32 packed16 K (packed16-only mode)
        //   - source f16 K materialized op-locally (MTP_VERIFY adapter)
        const bool original_kv =
            (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0) ||
            (K->type == GGML_TYPE_I32 && (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0));
        const bool f16_adapter =
            K->type == GGML_TYPE_F16 &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0) &&
            ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled() &&
            ggml_cuda_fattn_get_instruction(dst) == GGML_FATTN_INST_MTP_VERIFY_QK;
        return (original_kv || f16_adapter) && ggml_cuda_q8k_dot4_kq_enabled();
    }
    if (strcmp(required, "rocm_q8k_dot4_packed16_vec") == 0) {
        return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 && ggml_cuda_q8k_dot4_packed16_vec_enabled();
    }
    if (strcmp(required, "rocm_packed16_dot4_mmq") == 0 || strcmp(required, "packed16_dot4_mmq") == 0) {
        return K->type == GGML_TYPE_I32 &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0) &&
            ggml_cuda_packed16_dot4_mmq_enabled();
    }
    if (strcmp(required, "q8q4_dot4_prefill") == 0 ||
            strcmp(required, "rocm_q8q4_dot4") == 0 ||
            strcmp(required, "q8q4_dot4") == 0 ||
            strcmp(required, "q8q4_wmma_i8") == 0 ||
            strcmp(required, "rocm_q8q4_wmma_i8") == 0) {
        return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    }
    if (strcmp(required, "q8tbq4_dot4_prefill") == 0 ||
            strcmp(required, "rocm_q8_tbq4_dot4") == 0 ||
            strcmp(required, "q8tbq4_dot4") == 0) {
        return K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TBQ4_0;
    }
    if (strcmp(required, "tbq4_dot4_prefill") == 0 ||
            strcmp(required, "rocm_tbq4_dot4") == 0 ||
            strcmp(required, "tbq4_dot4") == 0) {
        return K->type == GGML_TYPE_TBQ4_0 && V->type == GGML_TYPE_TBQ4_0;
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
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 &&
            Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

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
    GGML_LOG_INFO("%s: fa_route_contract require=%s status=%s selected=%s f16_allowed=%d reason=%s Q=[%lld,%lld,%lld,%lld] K=%s V=%s exact=%.2fMiB stable=%.2fMiB\n",
            __func__, required, status, ggml_cuda_fattn_kernel_name(selected),
            f16_policy && f16_policy->allowed ? 1 : 0, reason,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
            ggml_type_name(K->type), ggml_type_name(V->type), exact_mib, stable_mib);
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

    const bool tbq4_vec_norm_hoist = kernel == BEST_FATTN_KERNEL_VEC && K->type == GGML_TYPE_TBQ4_0 && ggml_cuda_tbq4_vec_norm_hoist_enabled();
    const bool sparse_v_dequant = kernel == BEST_FATTN_KERNEL_VEC && V->type == GGML_TYPE_TBQ4_0 && Q->ne[1] == 1 && ggml_cuda_sparse_v_dequant_enabled();
    const bool q8k_tbq4v_vec = kernel == BEST_FATTN_KERNEL_VEC && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_TBQ4_0;
    const bool q8k_q4v_vec   = (kernel == BEST_FATTN_KERNEL_VEC || kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    const int sparse_v_tau_level = sparse_v_dequant ? ggml_cuda_sparse_v_tau_level() : 0;
    const bool tbq4_lds_d_k = kernel == BEST_FATTN_KERNEL_VEC && K->type == GGML_TYPE_TBQ4_0 && (Q->ne[0] == 128 || Q->ne[0] == 256) && Q->ne[1] == 1 &&
        ggml_cuda_tbq4_lds_route_d_k_enabled() && (!sparse_v_dequant || sparse_v_tau_level == 0);
    const char * route = ggml_cuda_fattn_kernel_name(kernel);
    if (kernel == BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC) {
        route = "q8k_dot4_packed16_vec";
    } else if (tbq4_lds_d_k) {
        if (Q->ne[0] == 256) {
            route = sparse_v_dequant ? "tbq4_lds_route_d_k_d256_sparsev" : "tbq4_lds_route_d_k_d256";
        } else {
            route = sparse_v_dequant ? "tbq4_lds_route_d_k_d128_sparsev" : "tbq4_lds_route_d_k_d128";
        }
    } else if (q8k_tbq4v_vec) {
        route = sparse_v_dequant ? "q8k_tbq4v_sparsev" : "q8k_tbq4v_vec";
    } else if (q8k_q4v_vec) {
        route = "q8k_q4v_vec";
    } else if (tbq4_vec_norm_hoist) {
        route = sparse_v_dequant ? "tbq4_vec_norm_hoist_sparsev" : "tbq4_vec_norm_hoist";
    } else if (kernel == BEST_FATTN_KERNEL_VEC && K->type == GGML_TYPE_TBQ4_0) {
        route = sparse_v_dequant ? "tbq4_vec_sparsev" : "tbq4_vec";
    }

    if (tbq4_lds_d_k) {
        const int lds_tile_rows = Q->ne[0] == 256 ? ggml_cuda_tbq4_lds_d_k_tile_rows<256>() : ggml_cuda_tbq4_lds_d_k_tile_rows<128>();
        const int lds_f16_stride = Q->ne[0] == 256 ? ggml_cuda_tbq4_lds_d_k_f16_stride_half2<256>() : ggml_cuda_tbq4_lds_d_k_f16_stride_half2<128>();
        const int lds_packed_stride = Q->ne[0] == 256 ? ggml_cuda_tbq4_lds_d_k_packed_stride<256>() : ggml_cuda_tbq4_lds_d_k_packed_stride<128>();
        GGML_LOG_INFO("%s: kernel=%s route=%s fa_inst=%s d=%lld tile_rows=%d f16_stride=%d packed_stride=%d sparse_v_tau=%d fallback=tbq4_vec Q=%s K=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld\n",
            __func__, ggml_cuda_fattn_kernel_name(kernel), route, fa_inst_name,
            (long long) Q->ne[0], lds_tile_rows, lds_f16_stride, lds_packed_stride,
            sparse_v_tau_level,
            ggml_type_name(Q->type), ggml_type_name(K->type), ggml_type_name(V->type),
            (long long) Q->ne[1], (long long) K->ne[1],
            (long long) Q->ne[0], (long long) V->ne[0]);
        return;
    }

    GGML_LOG_INFO("%s: kernel=%s route=%s fa_inst=%s Q=%s K=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld sparse_v_tau_level=%d\n",
        __func__, ggml_cuda_fattn_kernel_name(kernel), route, fa_inst_name,
        ggml_type_name(Q->type), ggml_type_name(K->type), ggml_type_name(V->type),
        (long long) Q->ne[1], (long long) K->ne[1],
        (long long) Q->ne[0], (long long) V->ne[0], sparse_v_tau_level);
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

static int64_t ggml_cuda_q8q4_wmma_i8_prefill_max_nq() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_Q8Q4_WMMA_I8_PREFILL_MAX_NQ");
    // If the existing ROCm quant-prefill F16 policy is enabled, keep long
    // prompt processing on that path and reserve Q8Q4 WMMA-I8 for smaller
    // decode/speculative batches. Set to 0 to force all prefill to F16, or set
    // higher to let larger prompt chunks use the experimental I8 path.
    return env ? atoll(env) : 256;
#else
    return 0;
#endif // GGML_USE_HIP
}

static bool ggml_cuda_tbq4_wmma_fattn_enabled() {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_rocm_experimental_unsafe_enabled()) {
        return false;
    }
    const char * env = getenv("GGML_CUDA_ROCM_TBQ4_WMMA_FATTN");
    if (!env) {
        env = getenv("TBQ4_WMMA_FATTN"); // backward-compatible prototype knob
    }
    return env && atoi(env) != 0;
#else
    return false;
#endif // GGML_USE_HIP
}

static void ggml_cuda_tbq4_wmma_fattn_log_reject(const char * reason, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_route_log_enabled()) {
        return;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_LOG_INFO("%s: route=wmma_tbq4 reject=%s Q=%s K=%s V=%s mask=%s sinks=%d nq=%lld nkv=%lld d_q=%lld d_v=%lld\n",
        __func__, reason,
        ggml_type_name(Q->type), ggml_type_name(K->type), ggml_type_name(V->type),
        mask ? ggml_type_name(mask->type) : "none", sinks ? 1 : 0,
        (long long) Q->ne[1], (long long) K->ne[1],
        (long long) Q->ne[0], (long long) V->ne[0]);
}

static bool ggml_cuda_tbq4_wmma_fattn_supported(
        const int cc, const ggml_tensor * dst,
        const float max_bias, const float logit_softcap) {
#ifdef GGML_USE_HIP
    if (!ggml_cuda_tbq4_wmma_fattn_enabled()) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const auto reject = [&](const char * reason) {
        ggml_cuda_tbq4_wmma_fattn_log_reject(reason, dst);
        return false;
    };

    if (!ggml_cuda_should_use_wmma_fattn(cc)) {
        return reject("wmma_unavailable_or_not_compiled");
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_TBQ4_0 || V->type != GGML_TYPE_TBQ4_0 || dst->type != GGML_TYPE_F32) {
        return reject("type");
    }
    if ((Q->ne[0] != 128 && Q->ne[0] != 256) || K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0] || dst->ne[0] != Q->ne[0]) {
        return reject("shape_d");
    }
    if (Q->ne[1] <= 2 || K->ne[1] < Q->ne[1] || V->ne[1] != K->ne[1]) {
        return reject("shape_tokens");
    }
    if (Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2] || Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return reject("shape_heads_sequences");
    }
    if (dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] || dst->ne[3] != Q->ne[3]) {
        return reject("shape_dst");
    }
    if (K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return reject("nkv_not_full_tile");
    }
    if (Q->nb[0] != ggml_element_size(Q) || K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) {
        return reject("stride_nb0");
    }
    if (sinks != nullptr) {
        return reject("sinks");
    }
    if (max_bias != 0.0f) {
        return reject("alibi");
    }
    if (logit_softcap != 0.0f) {
        return reject("logit_softcap");
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != Q->ne[3])) {
        return reject("mask_layout");
    }

    return true;
#else
    GGML_UNUSED(cc); GGML_UNUSED(dst); GGML_UNUSED(max_bias); GGML_UNUSED(logit_softcap);
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
    const bool tbq4_q8_mix = (K->type == GGML_TYPE_TBQ4_0 && V->type == GGML_TYPE_Q8_0) ||
                             (K->type == GGML_TYPE_Q8_0   && V->type == GGML_TYPE_TBQ4_0);
    const bool q8_q4_mix = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0;
    return tbq4_q8_mix || q8_q4_mix;
#endif // GGML_CUDA_FA_ALL_QUANTS
}

static void ggml_cuda_fattn_log_mixed_kv_reject(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    if (!ggml_cuda_fattn_route_log_enabled()) {
        return;
    }
    GGML_LOG_INFO("%s: route=none reject=unsupported_mixed_kv Q=%s K=%s V=%s nq=%lld nkv=%lld d_q=%lld d_v=%lld\n",
        __func__, ggml_type_name(Q->type), ggml_type_name(K->type), ggml_type_name(V->type),
        (long long) Q->ne[1], (long long) K->ne[1],
        (long long) Q->ne[0], (long long) V->ne[0]);
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

    // Tensor-core / fallback families
    GGML_CUDA_FATTN_BACKEND_WMMA_F16,
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
        case GGML_CUDA_FATTN_BACKEND_WMMA_F16:           return "wmma_f16";
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
        case BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL:
        case BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL:
        case BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL:
            return GGML_CUDA_FATTN_BACKEND_DOT4_RECTHIST_V4;

        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:
            return GGML_CUDA_FATTN_BACKEND_PACKED16_DOT4_MMQ;

        case BEST_FATTN_KERNEL_WMMA_F16:
            return GGML_CUDA_FATTN_BACKEND_WMMA_F16;

        case BEST_FATTN_KERNEL_MMA_F16:
            return GGML_CUDA_FATTN_BACKEND_MMA_F16;

        case BEST_FATTN_KERNEL_VEC:
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

// ── Unified fast-path attempt: DOT4 recthist ─────────────────────

static ggml_cuda_fa_fastpath_result ggml_cuda_try_dot4_recthist(
        const int cc,
        const ggml_tensor * dst,
        const ggml_fattn_instruction inst) {
    if (!ggml_cuda_q8k_dot4_kq_env_enabled()) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_ENV_DISABLED };
    }

    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_NOT_RDNA3 };
    }

    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 1) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_NQ_WRONG_FOR_ROUTE };
    }

    if (Q->ne[0] != 256) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_D_NOT_256 };
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!ggml_cuda_q8k_dot4_kq_legal_kv(K, V)) {
        // Legal K: persistent packed16 I32, q8_0, or source f16.
        // Legal V: f16, q8_0, q4_0.
        const bool k_ok = K->type == GGML_TYPE_F16 ||
                          K->type == GGML_TYPE_Q8_0 ||
                          (K->type == GGML_TYPE_Q4_0 && ggml_cuda_q8k_dot4_kq_allow_source_q4_0());
        const bool v_ok = V->type == GGML_TYPE_F16 ||
                          V->type == GGML_TYPE_Q8_0 ||
                          V->type == GGML_TYPE_Q4_0;

        if (!k_ok) {
            return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_MISSING_K_REPR };
        }
        if (!v_ok) {
            return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_UNSUPPORTED_V };
        }

        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_MISSING_K_REPR };
    }

    if (!ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(inst)) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_recthist", FA_REJECT_ROUTE_CONTRACT };
    }

    return { BEST_FATTN_KERNEL_Q8K_DOT4_KQ, "dot4_recthist", FA_REJECT_NONE };
}

// ── Unified fast-path attempt: DOT4 decode ───────────────────────

static ggml_cuda_fa_fastpath_result ggml_cuda_try_dot4_decode(
        const int cc,
        const ggml_tensor * dst,
        const ggml_fattn_instruction inst) {
    if (!ggml_cuda_q8k_dot4_kq_env_enabled()) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_ENV_DISABLED };
    }

    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_NOT_RDNA3 };
    }

    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] != 1) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_NQ_WRONG_FOR_ROUTE };
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Decode requires source f16 or q8_0 K for DOT4.
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_MISSING_K_REPR };
    }

    if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q4_0) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_UNSUPPORTED_V };
    }

    if (!ggml_cuda_q8k_dot4_kq_route_for_instruction_ok(inst)) {
        return { BEST_FATTN_KERNEL_NONE, "dot4_decode", FA_REJECT_ROUTE_CONTRACT };
    }

    return { BEST_FATTN_KERNEL_Q8K_DOT4_KQ, "dot4_decode", FA_REJECT_NONE };
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
            if (Q->ne[1] == 2 && inst == GGML_FATTN_INST_MTP_VERIFY_QK) {
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
        "%s: fa_instruction=%s dot4_role=%s backend_family=%s "
        "nq=%lld nk=%lld K=%s V=%s k_repr=%s v_repr=%s "
        "impl_status=%s selected=%s\n",
        __func__,
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
    if (inst == GGML_FATTN_INST_MTP_VERIFY_QK && Q->ne[1] > 1) {
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
        case GGML_FATTN_INST_NONE:                return "none";
        case GGML_FATTN_INST_MTP_DRAFT:           return "mtp_draft";
        case GGML_FATTN_INST_MTP_VERIFY_QK:       return "mtp_verify_qk";
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK: return "mtp_draft_decode_qk";
        case GGML_FATTN_INST_PREFILL_QK:          return "prefill_qk";
        case GGML_FATTN_INST_DECODE_QK:           return "decode_qk";
        case GGML_FATTN_INST_SPEC_VERIFY_QK:      return "spec_verify_qk";
        case GGML_FATTN_INST_BATCH_VERIFY_QK:     return "batch_verify_qk";
    }

    return "unknown";
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

static bool ggml_cuda_mtp_verify_dot4_recthist_supported(
        const int cc,
        const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (!ggml_cuda_q8k_dot4_kq_enabled()) {
        return false;
    }

    if (ggml_cuda_mtp_verify_dot4_disabled()) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Archetype: MTP verify nq > 1 → v4 recthist.
    // Current implementation gate: nq > 2 (DOT4 helpers reject nq <= 2).
    // nq == 2 behind explicit env for MTP_VERIFY_QK only.
    // General PREFILL_QK / SPEC_VERIFY_QK / BATCH_VERIFY_QK bypass.
    if (Q->ne[1] <= 1) {
        return false;
    }
    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    if (Q->ne[1] == 2 &&
        fa_inst_i32 == GGML_FATTN_INST_MTP_VERIFY_QK &&
        !ggml_cuda_mtp_verify_dot4_nq2_enabled()) {
        return false;
    }

    // Legal K representations for DOT4 recthist:
    //   A. Persistent packed16 I32, V = f16/q8_0/q4_0
    //   B. q8_0/q4_0
    //   C. Source f16 materialized op-locally into packed16 (env-gated)
    const bool packed16_k =
        K->type == GGML_TYPE_I32 &&
        K->ne[0] * 4 == Q->ne[0] &&
        (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0);

    const bool q8q4_kv =
        K->type == GGML_TYPE_Q8_0 &&
        V->type == GGML_TYPE_Q4_0 &&
        K->ne[0] == Q->ne[0] &&
        V->ne[0] == Q->ne[0];

    // PR3: Source f16 materialization.
    // K=f16, V=f16/q8_0/q4_0 with adapter enabled: the DOT4 launch
    // path already has f16→packed16 quantization. Use separate support
    // helper.
    if (packed16_k || q8q4_kv) {
        return ggml_cuda_q8k_dot4_kq_supported(cc, dst);
    }

    if (K->type == GGML_TYPE_F16 && (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_Q4_0)) {
        return ggml_cuda_mtp_verify_f16k_dot4_adapter_supported(cc, dst);
    }

    return false;
#else
    GGML_UNUSED(cc);
    GGML_UNUSED(dst);
    return false;
#endif
}

static const char * ggml_cuda_mtp_inst_name(int32_t inst) {
    switch (inst) {
        case GGML_FATTN_INST_MTP_DRAFT:           return "mtp_draft";
        case GGML_FATTN_INST_MTP_VERIFY_QK:       return "mtp_verify_qk";
        case GGML_FATTN_INST_MTP_DRAFT_DECODE_QK: return "mtp_draft_decode_qk";
        default:                                    return "none";
    }
}

// ── MTP draft decode helpers ────────────────────────────────────────

// BK threshold — delegates to centralized decode policy.
static int64_t ggml_cuda_q8k_dot4_decode_splitk_threshold() {
    return ggml_cuda_dot4_decode_policy_from_env().splitk_threshold;
}

static bool ggml_cuda_mtp_draft_dot4_decode_enabled() {
#ifdef GGML_USE_HIP
    const char * env = getenv("GGML_CUDA_ROCM_MTP_DRAFT_DOT4_DECODE");
    return env && atoi(env) != 0;
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

    // K: source f16 (op-local packed16) or q8_0/q4_0 original.
    // No persistent I32 packed16 for MTP draft (enforced by KV-cache gate).
    const bool source_f16_k =
        K->type == GGML_TYPE_F16 &&
        K->ne[0] == Q->ne[0];

    const bool q8q4_original =
        K->type == GGML_TYPE_Q8_0 &&
        V->type == GGML_TYPE_Q4_0 &&
        K->ne[0] == Q->ne[0];

    return Q->ne[0] == 256 &&
           V->ne[0] == Q->ne[0] &&
           v_ok &&
           (source_f16_k || q8q4_original) &&
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
                const char * k_repr = ggml_cuda_dot4_k_repr_name(
                        ggml_cuda_dot4_resolve_k_repr(Q, K, V));
                const char * v_repr = ggml_cuda_dot4_v_repr_name(V);
                const bool dot4_active = (selected == BEST_FATTN_KERNEL_Q8K_DOT4_KQ);
                const auto family = dot4_active
                    ? (dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT
                        ? GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_SPLITK
                        : GGML_CUDA_FATTN_BACKEND_DOT4_DECODE_BN64)
                    : GGML_CUDA_FATTN_BACKEND_EXISTING;
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
                        "<existing_policy>");
            }
        }
    };

    // nq must be 1 for decode.
    if (Q->ne[1] != 1) {
        log_decode("nq_not_1_decode_ineligible", BEST_FATTN_KERNEL_NONE, "none");
        return BEST_FATTN_KERNEL_NONE;
    }

    // Disable switches.
    if (getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_BN64")) {
        log_decode("bn64_disabled", BEST_FATTN_KERNEL_NONE, "bn64_disabled");
        return BEST_FATTN_KERNEL_NONE;
    }
    if (getenv("GGML_CUDA_ROCM_Q8K_DOT4_DISABLE_SPLITK")) {
        log_decode("splitk_disabled", BEST_FATTN_KERNEL_NONE, "splitk_disabled");
        return BEST_FATTN_KERNEL_NONE;
    }

    if (ggml_cuda_mtp_draft_dot4_decode_supported(cc, dst)) {
        const auto selected = ggml_cuda_fattn_apply_route_contract(
                dst, BEST_FATTN_KERNEL_Q8K_DOT4_KQ, f16_policy);
        const bool is_splitk = (dot4_role == GGML_CUDA_DOT4_ROLE_DECODE_SPLITK_MTP_DRAFT);
        const char * decode_impl = is_splitk ? "splitk_stage1_stage2" : "bn64";
        if (is_splitk) {
            // split-K stage1 writes unnormalized partial_o; stage2 merges.
            // Normalization must not happen in stage1.
        }
        log_decode("dot4_decode_selected", selected, decode_impl);
        return selected;
    }

    // Not legal — determine why.
    {
        const bool dot4_env = ggml_cuda_q8k_dot4_kq_enabled();
        const bool decode_env = ggml_cuda_mtp_draft_dot4_decode_enabled();
        const bool shapes_ok = (Q->ne[0] == 256 && V->ne[0] == Q->ne[0] &&
                                 K->ne[0] == Q->ne[0]);

        const char * status = !dot4_env            ? "env_disabled"
                            : !decode_env         ? "decode_env_disabled"
                            : !shapes_ok          ? "shape_or_device_reject"
                            :                        "missing_legal_k_representation";
        log_decode(status, BEST_FATTN_KERNEL_NONE, "none");
    }

    return BEST_FATTN_KERNEL_NONE;
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
    if (Q->ne[1] == 2 && inst == GGML_FATTN_INST_MTP_VERIFY_QK &&
            !ggml_cuda_mtp_verify_dot4_nq2_enabled()) {
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

    // Try DOT4 recthist as implementation.
    if (ggml_cuda_mtp_verify_dot4_recthist_supported(cc, dst)) {
        const auto selected = ggml_cuda_fattn_apply_route_contract(
                dst, BEST_FATTN_KERNEL_Q8K_DOT4_KQ, f16_policy);

        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_DOT4_RECTHIST_V4,
            k_repr,
            "dot4_candidate",
            selected);

        return selected;
    }

    // DOT4 not legal — determine why.
    {
        const bool dot4_env = ggml_cuda_q8k_dot4_kq_enabled();
        const bool f16_k = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
        const bool adapter_on = ggml_cuda_mtp_verify_f16k_dot4_adapter_enabled();

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

    // ── Instruction-aware fallback: WMMA_F16 / MMA_F16 ───────────────
    // Only after DOT4 is rejected. f16/f32 K/V only — no q8/q4 dequant.
    // Default off; env-gated per backend.

    if (ggml_cuda_mtp_verify_wmma_f16_supported(cc, dst)) {
        const best_fattn_kernel selected = BEST_FATTN_KERNEL_WMMA_F16;

        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_WMMA_F16,
            GGML_CUDA_DOT4_K_REPR_NONE,
            "dot4_rejected_wmma_f16_selected",
            selected);

        return selected;
    }

    if (ggml_cuda_mtp_verify_mma_f16_supported(cc, dst)) {
        const best_fattn_kernel selected = BEST_FATTN_KERNEL_MMA_F16;

        ggml_cuda_fattn_log_instruction_route(
            dst,
            inst_name,
            ggml_cuda_dot4_role_name(dot4_role),
            GGML_CUDA_FATTN_BACKEND_MMA_F16,
            GGML_CUDA_DOT4_K_REPR_NONE,
            "dot4_rejected_mma_f16_selected",
            selected);

        return selected;
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
                             fa_inst_i32 == GGML_FATTN_INST_MTP_VERIFY_QK);
        if (is_mtp) {
            const char * inst_name = fa_inst_i32 == GGML_FATTN_INST_MTP_DRAFT
                ? "mtp_draft" : "mtp_verify";
            const bool quantized_kv = ggml_is_quantized(K->type) || ggml_is_quantized(V->type);
            const bool dot4_kv_ok = (K->type == GGML_TYPE_I32) ||
                (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0);
            if (!dot4_kv_ok && fa_inst_i32 == GGML_FATTN_INST_MTP_VERIFY_QK) {
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

        const bool k_shape_ok = K->ne[0] * 4 == Q->ne[0];  // D/4 * 4 == D
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
                    Q->type, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                    K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                    V->ne[0], V->ne[1], V->ne[2], V->ne[3]);
            }
            return BEST_FATTN_KERNEL_NONE;
        }

        // Debug policy: keep MTP off persistent packed16 I32-K routes by
        // default. MTP DOT4 route experiments use instruction-specific
        // f16-source materialization/decoding; set LLAMA_MTP_DISABLE_PACKED16_FA=0
        // only for focused packed16 MTP experiments.
        {
            const int32_t fi = ((const int32_t *)dst->op_params)[4];
            if (fi == GGML_FATTN_INST_MTP_VERIFY_QK ||
                fi == GGML_FATTN_INST_MTP_DRAFT ||
                fi == GGML_FATTN_INST_MTP_DRAFT_DECODE_QK) {
                const char * mtp_disable_p16 = getenv("LLAMA_MTP_DISABLE_PACKED16_FA");
                const bool disable_p16_mtp = mtp_disable_p16 == nullptr || atoi(mtp_disable_p16) != 0;
                if (disable_p16_mtp) {
                    if (getenv("LLAMA_MTP_FA_ROUTE")) {
                        fprintf(stderr, "MTP_FA_ROUTE: inst=%d k_type=%d selected=none reason=packed16_mtp_disabled_by_default\n",
                            fi, K->type);
                    }
                    return BEST_FATTN_KERNEL_NONE;
                }
            }
        }

        // ── Route dispatch ───────────────────────────────────
        const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
        const ggml_fattn_instruction inst = (ggml_fattn_instruction)fa_inst_i32;

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
             strcmp(required_route, "rocm_q8k_dot4_decode_splitk_mtp_draft") == 0);
        if (require_q8k_dot4_kq) {
            return BEST_FATTN_KERNEL_Q8K_DOT4_KQ;
        }

        const bool require_packed16_dot4_mmq =
            required_route &&
            (strcmp(required_route, "rocm_packed16_dot4_mmq") == 0 ||
             strcmp(required_route, "packed16_dot4_mmq") == 0);
        if (require_packed16_dot4_mmq) {
            // Route-require is a hard contract, including nq==1 decode.  This is
            // the opt-in path for validating packed16 I32 K + q4_0 V entirely in
            // the DOT4/MMQ attention family instead of silently falling back to
            // the older q8k_dot4_kq decode kernel.
            if (!ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
                GGML_ABORT("required rocm_packed16_dot4_mmq route was not selected; Q=[%lld,%lld,%lld,%lld] K=[%lld,%lld,%lld,%lld] V=%s",
                    (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                    (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3],
                    ggml_type_name(V->type));
            }
            return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
        }
        if (require_packed16_wmma && ggml_cuda_packed16_wmma_tile_enabled()) {
        fprintf(stderr, "ROUTE: require_packed16_wmma=1 enabled=1 nq=%lld\n", (long long)Q->ne[1]);
            if (Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_Q8K_DOT4_KQ;  // nq==1 decode has no WMMA kernel yet
            }
            return BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
        }

        // MTP_VERIFY_QK must NOT use packed16 I32 K — the MTP draft head is
        // trained on exact f16 K, and even small differences from lossy 4-bit
        // packed16 K cause 100%% draft rejection.  Route MTP verify through the
        // instruction fastpath (q8k_dot4_kq / VEC / WMMA_F16) instead.
        const char * mtp_disable_p16_env = getenv("LLAMA_MTP_DISABLE_PACKED16_FA");
        const bool mtp_packed16_experiment = mtp_disable_p16_env && atoi(mtp_disable_p16_env) == 0;
        const bool prefill_or_verify =
            Q->ne[1] > 1 &&
            (inst == GGML_FATTN_INST_PREFILL_QK ||
             inst == GGML_FATTN_INST_SPEC_VERIFY_QK ||
             inst == GGML_FATTN_INST_BATCH_VERIFY_QK ||
             (mtp_packed16_experiment && inst == GGML_FATTN_INST_MTP_VERIFY_QK));
        const bool v_requires_dot4_mmq = V->type == GGML_TYPE_TBQ4_0 || V->type == GGML_TYPE_PLANAR3_0 || V->type == GGML_TYPE_ISO3_0;

        // nq == 1 decode: always route to DOT4 (BN64/split-K).
        // packed16_wmma_tile only supports nq > 1.
        // When WMMA is explicitly required via env, nq==1 is still
        // allowed to use DOT4 (WMMA decode kernel doesn't exist).
        if (Q->ne[1] == 1) {
            if (v_requires_dot4_mmq && ggml_cuda_packed16_dot4_mmq_supported(cc, dst)) {
                return BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
            }
            return BEST_FATTN_KERNEL_Q8K_DOT4_KQ;
        }

        // nq > 1 prefill/verify: DOT4-MMQ is first priority (fastest packed16 path).
        // PWMMA BM32 is preferred fallback. DOT4_KQ is last resort.
        const bool auto_verbose =
            (getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE") && atoi(getenv("GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE"))) ||
            (getenv("COMPRESSED_KV_FATTN_LOG") && atoi(getenv("COMPRESSED_KV_FATTN_LOG")));
        const bool dot4_sup = ggml_cuda_packed16_dot4_mmq_supported(cc, dst);
        const bool wmma_sup = ggml_cuda_packed16_wmma_tile_enabled();
        best_fattn_kernel packed16_kernel = BEST_FATTN_KERNEL_NONE;
        const char * packed16_route_name = "none";

        if ((prefill_or_verify || v_requires_dot4_mmq) && dot4_sup) {
            // ── BM32_regout_directv auto-selection ──────────
            // Wins on 27B at all pp ≥512 and on 35B at pp ≥1024.
            const char * explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
            const bool impl_is_wmma =
                explicit_impl && (
                    strcmp(explicit_impl, "bm32_regout_directv") == 0 ||
                    strcmp(explicit_impl, "bm32_regout_stagev") == 0 ||
                    strcmp(explicit_impl, "bm64_regout_directv") == 0 ||
                    strcmp(explicit_impl, "bm64_regout_stagev") == 0);
            const bool impl_auto = (!explicit_impl || !*explicit_impl || strcmp(explicit_impl, "smem") == 0);
            // 27B-like shape: gqa_ratio=6, heads_q=24, heads_k=4
            const bool is_27b_like = (gqa_ratio == 6 && K->ne[1] >= 1024);
            // 35B-like or general: nk >= 1024 (context length, not query chunk size)
            const bool is_long_context = (K->ne[1] >= 1024);
            const bool wmma_available = wmma_sup;
            const bool use_bm32_regout = !v_requires_dot4_mmq && impl_auto && wmma_available && (is_27b_like || is_long_context);

            // Once we auto-select WMMA for long context, keep using it for
            // subsequent calls (IMPL is already set to bm32_regout_directv).
            if (!v_requires_dot4_mmq && (use_bm32_regout || (impl_is_wmma && wmma_available))) {
                if (use_bm32_regout) {
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL", "bm32_regout_directv", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "32", 1);
                    setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                }
                packed16_kernel = BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
                packed16_route_name = "pwmma_bm32_regout_directv";
            } else {
                packed16_kernel = BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ;
                packed16_route_name =
                    V->type == GGML_TYPE_TBQ4_0    ? "dot4_mmq_tbq4" :
                    V->type == GGML_TYPE_PLANAR3_0 ? "dot4_mmq_planar3" :
                    V->type == GGML_TYPE_ISO3_0    ? "dot4_mmq_iso3" : "dot4_mmq_gqa1";
            }
        } else if (prefill_or_verify && wmma_sup) {
            packed16_kernel = BEST_FATTN_KERNEL_PACKED16_WMMA_TILE;
            // Auto-select BM32_regout_directv as best PWMMA impl
            const char * explicit_impl = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL");
            const bool impl_auto = (!explicit_impl || !*explicit_impl || strcmp(explicit_impl, "smem") == 0);
            if (impl_auto) {
                setenv("GGML_CUDA_ROCM_PACKED16_WMMA_IMPL", "bm32_regout_directv", 1);
                setenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM", "32", 1);
                setenv("GGML_CUDA_ROCM_PACKED16_WMMA_CAUSAL_SKIP", "1", 1);
                packed16_route_name = "pwmma_bm32_regout_directv";
            } else {
                const int bm = getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM") ? atoi(getenv("GGML_CUDA_ROCM_PACKED16_WMMA_BM")) : 32;
                packed16_route_name = (bm == 16) ? "pwmma_bm16" : (bm == 32 ? "pwmma_bm32" : "pwmma_bm64");
            }
        } else if (ggml_cuda_q8k_dot4_kq_enabled() || getenv("GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE")) {
            packed16_kernel = BEST_FATTN_KERNEL_Q8K_DOT4_KQ;
            packed16_route_name = "dot4_kq_oracle";
        } else {
            packed16_kernel = BEST_FATTN_KERNEL_NONE;
            packed16_route_name = "none";
        }

        if (auto_verbose && g_fattn_select_ctx == GGML_CUDA_FATTN_SELECT_DISPATCH) {
            fprintf(stderr, "%s: PACKED16 FA ROUTE nq=%lld nk=%lld D=%lld hq=%lld hk=%lld gqa=%d"
                " K=I32 V=%s auto=1 selected=%s"
                " dot4_mmq_sup=%d wmma_sup=%d forced_route=%s\n",
                __func__,
                (long long)Q->ne[1], (long long)K->ne[1], (long long)Q->ne[0],
                (long long)Q->ne[2], (long long)K->ne[2], gqa_ratio,
                ggml_type_name(V->type),
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
        case GGML_TYPE_TBQ4_0: {
            const bool v_is_tbq4 = V->type == GGML_TYPE_TBQ4_0;
            const bool v_is_q8_0 = V->type == GGML_TYPE_Q8_0;
            if (!v_is_tbq4 && !v_is_q8_0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if ((Q->ne[0] != 128 && Q->ne[0] != 256) || V->ne[0] != Q->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
#ifdef GGML_USE_HIP
            // Experimental packed-dot4/rocWMMA prefill paths. Default decode/quantized-KV
            // route remains VEC unless the unsafe lab gate is explicitly set.
            if (v_is_tbq4 && ggml_cuda_tbq4_dot4_prefill_supported(cc, dst)) {
                return BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL;
            }
            if (v_is_tbq4 && ggml_cuda_tbq4_wmma_fattn_supported(cc, dst, max_bias, logit_softcap)) {
                return BEST_FATTN_KERNEL_WMMA_TBQ4;
            }
#endif // GGML_USE_HIP
            if (v_is_q8_0) {
                return amd_wmma_available(cc) ? BEST_FATTN_KERNEL_VEC : BEST_FATTN_KERNEL_NONE;
            }
            if (turing_mma_available(cc)) {
                return BEST_FATTN_KERNEL_MMA_TBQ4;
            }
            if (amd_wmma_available(cc) && (Q->ne[0] == 128 || Q->ne[0] == 256)) {
                return BEST_FATTN_KERNEL_VEC;
            }
            return BEST_FATTN_KERNEL_NONE;
        }
        case GGML_TYPE_PLANAR3_0:
        case GGML_TYPE_ISO3_0:
        case GGML_TYPE_PLANAR4_0:
        case GGML_TYPE_ISO4_0:
            // Planar/IsoQuant: VEC remains the default compressed-KV path.
            // Experimental rocWMMA path uses the same composable loader interface
            // as TBQ4, but is opt-in to avoid perturbing stable long-context runs.
            if (Q->ne[0] == 128 || Q->ne[0] == 256) {
#ifdef GGML_USE_HIP
                const char * compressed_wmma_fattn_env = getenv("COMPRESSED_KV_WMMA_FATTN");
                if (ggml_cuda_rocm_experimental_unsafe_enabled() && K->type == V->type && Q->ne[1] > 2 &&
                        compressed_wmma_fattn_env && atoi(compressed_wmma_fattn_env) != 0 && amd_wmma_available(cc)) {
                    return BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV;
                }
#endif // GGML_USE_HIP
                break; // falls through to VEC/TILE selection
            }
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
        const bool stable_q8q4_f16_prefill =
            f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_ALLOW &&
            !require_i8_route && f16_prefill_applicable &&
            Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 &&
            Q->ne[1] > 2;
        if ((stable_q8q4_f16_prefill || require_f16_route ||
             f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE ||
             f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER) && f16_prefill_applicable) {
            f16_policy.forced = require_f16_route || f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_REQUIRE;
            f16_policy.automatic = !f16_policy.forced &&
                (f16_mode == GGML_CUDA_ROCM_QUANT_PREFILL_F16_PREFER || stable_q8q4_f16_prefill);
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

        if (!require_f16_route && ggml_cuda_q8q4_wmma_i8_require_selected_applies(dst)) {
            if (ggml_cuda_q8q4_wmma_i8_supported(cc, dst, max_bias, logit_softcap)) {
                return return_quantized_route(BEST_FATTN_KERNEL_Q8Q4_WMMA_I8);
            }
            GGML_ABORT("required selected ROCm q8q4_wmma_i8 route was not selected; layer=%d selected_policy=applies K=%s V=%s Q=[%lld,%lld,%lld,%lld]",
                    ggml_cuda_q8q4_wmma_i8_layer_id(dst), ggml_type_name(K->type), ggml_type_name(V->type),
                    (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3]);
        }

        // WMMA-I8 route contract: when explicitly required, check before
        // packed16_vec / DOT4 so they can't claim the op first.
        {
            const char * route_req = getenv("GGML_CUDA_FA_ROUTE_REQUIRE");
            const bool q8q4_wmma_i8_required = route_req &&
                (strcmp(route_req, "q8q4_wmma_i8") == 0 || strcmp(route_req, "rocm_q8q4_wmma_i8") == 0);
            if (q8q4_wmma_i8_required) {
                if (ggml_cuda_q8q4_wmma_i8_supported(cc, dst, max_bias, logit_softcap)) {
                    return return_quantized_route(BEST_FATTN_KERNEL_Q8Q4_WMMA_I8);
                }
                GGML_ABORT("required ROCm q8q4_wmma_i8 route was not selected; K=%s V=%s Q=[%lld,%lld,%lld,%lld]",
                    ggml_type_name(K->type), ggml_type_name(V->type),
                    (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3]);
            }
        }

        if (!require_f16_route && ggml_cuda_q8k_dot4_packed16_vec_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC);
        }

        if (!require_f16_route && ggml_cuda_q8k_dot4_kq_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8K_DOT4_KQ);
        }

        if (!require_f16_route && ggml_cuda_q8q4_dot4_prefill_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL);
        }
        if (!require_f16_route && ggml_cuda_q8tbq4_dot4_prefill_supported(cc, dst)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL);
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

        const int64_t q8q4_wmma_prefill_max_nq = ggml_cuda_q8q4_wmma_i8_prefill_max_nq();
        const bool prefer_f16_prefill = allow_quant_prefill_f16 && Q->ne[1] > q8q4_wmma_prefill_max_nq;
        if (!prefer_f16_prefill && ggml_cuda_q8q4_wmma_i8_supported(cc, dst, max_bias, logit_softcap)) {
            return return_quantized_route(BEST_FATTN_KERNEL_Q8Q4_WMMA_I8);
        }

        if (!allow_quant_prefill_f16) {
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

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
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

// TBQ4 ncols switch — generic on DKQ/DV, supports D=128 and D=256.
template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        // 8/ncols2 branch: only for Turing where ncols=8 is valid.
        // Volta+ requires ncols >= 32, so skip this branch there.
        if (turing_mma_available(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_tbq4_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_tbq4_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_tbq4_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_tbq4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * KQV  = dst;

    const int DKQ = (int)Q->ne[0];
    const int DV  = (int)V->ne[0];
    GGML_ASSERT(DKQ == 128 || DKQ == 256);
    GGML_ASSERT(DV == DKQ);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const ggml_tensor * mask = dst->src[3];
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Pre-rotate Q into rotated domain via SEPARATE kernel (avoids nvcc register spill bug)
    const int64_t nrows = Q->ne[1] * Q->ne[2] * Q->ne[3];
    tbq4_rotate_input_cuda((float *) Q->data, nrows, DKQ, ctx.stream());

#define TBQ4_DISPATCH_NCOLS1(DKQ_VAL, DV_VAL)                                        \
    if (use_gqa_opt && gqa_ratio > 4) {                                              \
        ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1<DKQ_VAL, DV_VAL, 8>(ctx, dst); \
    } else if (use_gqa_opt && gqa_ratio > 2) {                                       \
        ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1<DKQ_VAL, DV_VAL, 4>(ctx, dst); \
    } else if (use_gqa_opt && gqa_ratio > 1) {                                       \
        ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1<DKQ_VAL, DV_VAL, 2>(ctx, dst); \
    } else {                                                                         \
        ggml_cuda_flash_attn_ext_mma_tbq4_switch_ncols1<DKQ_VAL, DV_VAL, 1>(ctx, dst); \
    }

    if (DKQ == 128) {
        TBQ4_DISPATCH_NCOLS1(128, 128)
    } else {
        TBQ4_DISPATCH_NCOLS1(256, 256)
    }

#undef TBQ4_DISPATCH_NCOLS1

    // Apply rotate_inverse to the output (rotated-domain → original domain).
    tbq4_rotate_output_cuda((float *) KQV->data, nrows, DV, ctx.stream());
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
                GGML_LOG_INFO(
                    "fa_entry: inst=%s nq=%lld nk=%lld\n",
                    ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                    (long long) Q->ne[1],
                    (long long) K->ne[1]);
            }
        }
    }

    // Initialize planar/iso rotation constants on first use.
    // Must happen before any planar/iso kernel (VEC, cpy, set-rows).
    {
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];
        if (K->type == GGML_TYPE_PLANAR3_0 || K->type == GGML_TYPE_ISO3_0 ||
            K->type == GGML_TYPE_PLANAR4_0 || K->type == GGML_TYPE_ISO4_0 ||
            V->type == GGML_TYPE_PLANAR3_0 || V->type == GGML_TYPE_ISO3_0 ||
            V->type == GGML_TYPE_PLANAR4_0 || V->type == GGML_TYPE_ISO4_0) {
            ggml_cuda_init_planar_iso_constants();
        }
    }

    g_fattn_select_ctx = GGML_CUDA_FATTN_SELECT_DISPATCH;
    if (fattn_dispatch_trace) {
        fprintf(stderr, "FATTN COMPUTE ENTER dst=%p\n", (void*)dst); fflush(stderr);
    }
    const best_fattn_kernel best_kernel = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    if (fattn_dispatch_trace) {
        fprintf(stderr, "FATTN COMPUTE SELECT selected=%d name=%s dst=%p\n", (int)best_kernel, ggml_cuda_fattn_kernel_name(best_kernel), (void*)dst); fflush(stderr);
    }
    ggml_cuda_fattn_log_selection(best_kernel, dst);

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
        if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
            if (log_env && atoi(log_env) != 0) {
                GGML_LOG_INFO(
                    "fa_final_select: inst=%s selected=%s nq=%lld nk=%lld d=%lld K=%s V=%s final=1\n",
                    ggml_cuda_fattn_instruction_name((ggml_fattn_instruction)fa_inst_i32),
                    ggml_cuda_fattn_kernel_name(best_kernel),
                    Q ? (long long) Q->ne[1] : -1LL,
                    K ? (long long) K->ne[1] : -1LL,
                    Q ? (long long) Q->ne[0] : -1LL,
                    K ? ggml_type_name(K->type) : "-",
                    V ? ggml_type_name(V->type) : "-");
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
        case BEST_FATTN_KERNEL_VEC: {
            const ggml_tensor * Q = dst->src[0];
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * V = dst->src[2];

            // TBQ4 stores K/V in its signed-FWHT domain. Match the TBQ4 MMA path:
            // rotate every Q row (tokens × heads × sequences) before attention and
            // inverse-rotate every output row when V is TBQ4.
            const int64_t nrows = Q->ne[1] * Q->ne[2] * Q->ne[3];
            if (K->type == GGML_TYPE_TBQ4_0) {
                tbq4_rotate_input_cuda((float *) Q->data, nrows, Q->ne[0], ctx.stream());
            }
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            if (V->type == GGML_TYPE_TBQ4_0) {
                tbq4_rotate_output_cuda((float *) dst->data, nrows, V->ne[0], ctx.stream());
            }
            break;
        }
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_TBQ4:
            ggml_cuda_flash_attn_ext_mma_tbq4(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_TBQ4:
            ggml_cuda_flash_attn_ext_wmma_tbq4(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_COMPRESSED_KV:
            ggml_cuda_flash_attn_ext_wmma_compressed_kv(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8Q4_WMMA_I8:
            ggml_cuda_flash_attn_ext_q8q4_wmma_i8(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8Q4_DOT4_PREFILL:
            ggml_cuda_flash_attn_ext_q8q4_dot4_prefill(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8TBQ4_DOT4_PREFILL:
            ggml_cuda_flash_attn_ext_q8tbq4_dot4_prefill(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_TBQ4_DOT4_PREFILL:
            ggml_cuda_flash_attn_ext_tbq4_dot4_prefill(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8K_DOT4_KQ:
            ggml_cuda_flash_attn_ext_q8k_dot4_kq(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_Q8K_DOT4_PACKED16_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_PACKED16_WMMA_TILE:
            fprintf(stderr, "PWMMA SWITCH CASE HIT\n");
            ggml_cuda_flash_attn_ext_packed16_wmma_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_PACKED16_DOT4_MMQ:
            ggml_cuda_flash_attn_ext_packed16_dot4_mmq(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    g_fattn_select_ctx = GGML_CUDA_FATTN_SELECT_SUPPORT_PROBE;
    const int32_t fa_inst_i32 = ((const int32_t *)dst->op_params)[4];
    const ggml_tensor * Q = dst->src[0];
    fprintf(stderr, "FATTN SUPPORT ENTER dst=%p\n", (void*)dst); fflush(stderr);
    const best_fattn_kernel k = ggml_cuda_get_best_fattn_kernel(device, dst);
    fprintf(stderr, "FATTN SUPPORT SELECT selected=%d name=%s dst=%p\n", (int)k, ggml_cuda_fattn_kernel_name(k), (void*)dst); fflush(stderr);
    const bool result = k != BEST_FATTN_KERNEL_NONE;
    if (const char * log_env = getenv("COMPRESSED_KV_FATTN_LOG")) {
        if (log_env && atoi(log_env) != 0) {
            GGML_LOG_INFO("fa_supported: inst=%d nq=%lld kernel=%d result=%d\n",
                fa_inst_i32, (long long) Q->ne[1], (int)k, (int)result);
        }
    }
    return result;
}
