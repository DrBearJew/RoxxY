# RDNA3 WMMA layout notes

## Facts
- gfx1100 uses RDNA3 wave32 WMMA.
- A/B fragments require lane replication between lanes `0..15` and `16..31`.
- Existing CUDA `ldmatrix` assumptions do not map directly.
- The failed build showed:
  - wanted `DATA_LAYOUT_I_MAJOR_MIRRORED`
  - got `DATA_LAYOUT_I_MAJOR`
  - expected larger A/B fragments

## Use this when implementing native path
- A fragment: lane `i % 16` owns one column/row group depending on operand convention.
- B fragment: lane `i % 16` owns matching row/column group.
- Use AMD examples as source of truth:
  - GPUOpen RDNA3 WMMA article
  - `adelj88/rocm_wmma_gemm`
  - rocWMMA `load_matrix_sync`

## Rule
Treat “mirrored” as backend lane-packing, not logical matrix layout. Keep it out of shared CUDA/Turing templates until a standalone RDNA3 kernel is proven.
