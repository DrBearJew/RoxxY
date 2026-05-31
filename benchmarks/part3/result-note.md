# Part 3 Result — BM32/x2-wave packed16 PWMMA

## Build
- **Hash**: 9f12df452 + Part 3 patches
- **Build dir**: build-rocm-fixed (amdclang++, gfx1100)
- **Model**: Qwen3.6-35B-A3B-IQ4_XS (35.51B, D=256, heads_q=16, heads_kv=2)
- **GPU**: Radeon RX 7900 XTX (RDNA3, gfx1100, 24GB)

## Patch Summary
- Added `ggml_cuda_rocm_packed16_wmma_bm()` env selector (default BM=16)
- Renamed kernel to `packed16_wmma_tile_bm16_1w_kernel`, kept intact
- Added `packed16_wmma_tile_bm32_2w_kernel` — 2 WMMA waves sharing V tile
- Added `pbwmma_qk_probe_bm32_kernel` + `pbwmma_qk_probe_bm32_pass()`
- Host launcher: BM dispatch, grid computed from dynamic BM
- All constants localized: `PWMMA_BM16=16`, `PWMMA_BM32=32`
- No graph/backend/scheduler files touched

## Performance Matrix

| Prefill | BM16_1W tok/s | BM32_2W tok/s | BM32 vs BM16 |
|---------|--------------|--------------|-------------|
| pp512   | 2590 ± 40    | 2606 ± 45    | +0.6% (noise) |
| pp1024  | 2394 ± 15    | 2435 ± 16    | **+1.7%** |
| tg1     | 82.5 ± 20    | 82.1 ± 21    | identical |

## Correctness Gates

| Gate | Status |
|------|--------|
| BM16 QK probe (unchanged) | PASSED max_err=0.000072 |
| BM32 QK probe | **PASSED max_err=0.000063** |
| PWMMA_TILE route forced | CONFIRMED (selected=585) |
| DOT4 fallback for nq>1 | NONE |
| CPU fallback | NONE |
| Causal skip working | YES (BM16: 7936, BM32: 3840 CTA tiles) |
| No NaNs | CONFIRMED |
| No crashes | CONFIRMED (all 44 layers EXIT OK) |
| BM16 unchanged | CONFIRMED (2590 tok/s matches Part 2 anchor) |
| V layout FA-safe | CONFIRMED (nb=(2,1024,512,524288)) |

## Skip Stats

| Variant | CTA tiles skipped | Row-tiles skipped |
|---------|------------------|-------------------|
| BM16 | 7936 | 126,976 |
| BM32 | 3840 | 122,880 |

BM32 has half the CTA skips (half as many Q tiles), with skip verification
using BM32 geometry (`q_first=32*t, q_last=32*t+31`, `k0 > q_last`).

## Generation Correctness

llama-cli crashes during model loading (server_context_impl::load_model)
on BOTH BM16 and BM32 — pre-existing build compatibility issue in this
feature branch. llama-bench works correctly. BM32 runs the same number
of FA kernel invocations as BM16 with identical EXIT OK status.

## Decision

**CLEARED FOR PART 4 (GQA K/V reuse).**

BM32 is:
- Correct and selectable via `GGML_CUDA_ROCM_PACKED16_WMMA_BM=32`
- Slightly faster at pp1024 (+1.7%) but speed-neutral at pp512
- Not yet fast enough to justify auto-routing as default
- Default remains BM16
- Causal skip preserved
- All hard stops green

Recommendation: keep BM32 as off-by-default until GQA reuse (Part 4)
and/or larger prefill benchmarks show a clearer advantage. Do not jump
to BM64/x4 before Part 4 GQA reuse per the original plan.
