# Compressed-KV Triton experiment lane

This directory is Track B from `docs/rocm-tbq4-paths/21-compressed-kv-fa-two-track-roadmap.md`.

## Scope

- Prototype compressed-KV tile materializers and FlashAttention-shaped loops in Triton/ROCm.
- Feed conclusions back into the C++/HIP materializer/backend contract.
- Stay isolated from production llama.cpp CMake, wrappers, and server runtime.

## Environment

Use the local conda LLM environment:

```bash
/home/mrtrent/miniconda3/envs/LLM/bin/python --version
/home/mrtrent/miniconda3/envs/LLM/bin/python scripts/hip/check-triton-feasibility.py
```

Known-good checkpoint on this machine:

- Triton 3.7.0
- torch `2.12.0a0+rocm7.13.0a20260412`
- HIP `7.13.60980`
- AMD Radeon RX 7900 XTX / `gfx1100`

## Isolation rules

- No files here are referenced by production CMake.
- No wrapper or llama-swap config should set Triton-related runtime flags.
- Dense materializer outputs in this directory are test artifacts only; production must keep compressed KV in global memory and materialize only per-tile values on chip.

## Checks

```bash
experiments/compressed_kv_triton/run_all.sh
```

The aggregate check runs:

1. `scripts/hip/check-triton-feasibility.py`
2. `compat_gate.py`
3. `materializers.py`
4. `paged_materializers.py`
5. `qk_only.py`
6. `qk_2d_tiled.py`
7. `online_softmax.py`
8. `full_qkv.py`
9. `qkv_2d_tiled.py`
10. `varlen_qkv.py`
11. `mask_semantics.py`
12. `segmented_qkv.py`
13. `compare_2d_segmented.py`
14. `autotune_metadata.py`

Current prototype status is recorded in `docs/rocm-tbq4-paths/24-triton-prototype-results.md`.
