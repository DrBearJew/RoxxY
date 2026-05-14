# Path 1 Detail — rocWMMA TBQ4 integration steps

## Implement first
- Add a separate ROCm-only launcher: `ggml_cuda_flash_attn_ext_wmma_tbq4(...)`.
- Keep existing `flash_attn_ext_f16` unchanged until TBQ4 path passes.
- Include/extract TBQ4 helpers from `fattn-mma-tbq4.cuh`:
  - `block_tbq4_0`
  - centroid lookup
  - rotate input/output kernels
  - dequant row/tile helper

## Kernel shape
- Start only with:
  - `D=128`
  - `ncols=16` or `32`
  - `K.type == V.type == GGML_TYPE_TBQ4_0`
  - `V_is_K_view=false`
- Dequant K tile into shared FP16 before KQ `rocwmma::load_matrix_sync`.
- Dequant V tile into shared FP16 before VKQ `rocwmma::load_matrix_sync`.

## Dispatch
- In `fattn.cu`, route AMD TBQ4 to this new rocWMMA TBQ4 launcher only when `GGML_HIP_ROCWMMA_FATTN && RDNA3`.
- Keep NVIDIA/Turing on existing `BEST_FATTN_KERNEL_MMA_TBQ4`.
