# Path 1 — rocWMMA FP16-dequant TBQ4 FA

## Goal
Make `tbq4_0` fast on gfx1100 by adapting existing rocWMMA flash-attention instead of porting CUDA MMA.

## Implement
- Work in `ggml/src/ggml-cuda/fattn-wmma-f16.cu`.
- Add TBQ4 K/V input mode for `GGML_TYPE_TBQ4_0`.
- Reuse dequant logic from `ggml/src/ggml-cuda/fattn-mma-tbq4.cuh`.
- Dequant raw TBQ4 K/V rows into shared FP16 tiles shaped for existing `rocwmma::load_matrix_sync` calls.
- First support only `D=128`, `V_is_K_view=false`, ROCm/gfx1100.

## Do not
- Do not modify `mma.cuh`.
- Do not enable `BEST_FATTN_KERNEL_MMA_TBQ4` on AMD.
- Do not implement native int4 yet.

## Verify
```bash
cd /tmp/llama.cpp-mtp
cmake --build build-rocm-tq --target llama-server -j8
```
Then compare `tbq4_0` output against current non-fused TBQ4 and benchmark tok/s.
