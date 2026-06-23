# Path 2 — Native RDNA3 WMMA microkernel

## Goal
Create a gfx1100-specific replacement for the CUDA/Turing MMA path.

## Implement
- Work in a new isolated header, not broad `mma.cuh` changes first.
- Add RDNA3 tile loaders using wave32 layout:
  - A/B fragments require lane replication across lanes `0..15` and `16..31`.
  - Use `DATA_LAYOUT_I_MAJOR_MIRRORED` for A/B.
- Use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for FP16-dequant prototype.
- Start with standalone 16x16x16 test kernel before plugging into FA.
- Only after correctness, wire into TBQ4 `D=128` KQ matmul.

## Do not
- Do not globally replace RDNA4 guards with `AMD_WMMA_AVAILABLE`.
- Do not reuse CUDA `ldmatrix` assumptions.
- Do not touch production dispatch until standalone kernel passes.

## Verify
- Build standalone kernel on `/opt/rocm-7.2.3`.
- Confirm generated ISA contains `v_wmma_f32_16x16x16_f16`.
- Compare 16x16 GEMM against CPU reference.
