# Packed16 DOT4 Flash Attention — Benchmark Report 2026-05-28

**Hardware**: RX 7900 XTX (24 GiB), ROCm 7.2.3, gfx1100
**Models**: Qwen3.6 27B Q4_K_M, Qwen3.6 35B A3B IQ4_XS
**Environment**: `GGML_CUDA_ROCM_Q8K_DOT4_KQ=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=blockfa_recthist_v4_single GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1 GGML_CUDA_ROCM_Q8K_DOT4_BLOCKFA_ASSUME_CAUSAL=1 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1`

---

## 1. 27B Qwen3.6 Q4_K_M — PPL + VRAM

| Config | K format | V format | PPL c512 | c512 VRAM | 32k VRAM | 64k VRAM |
|--------|----------|----------|----------|-----------|----------|----------|
| baseline (std FA) | f16 | f16 | 1.0128 | 40.5 MiB | 2048 MiB | 4096 MiB |
| p16 + f16 | I32 int8 | f16 | 1.0342 | 24.5 MiB | 1568 MiB | 3136 MiB |
| **p16 + q4_0** | **I32 int8** | **q4_0** | **1.0237** | **13.0 MiB** | **832 MiB** | **1664 MiB** |
| q8_0 + q4_0 (shadow) | q8_0 | q4_0 | 3.0373 | 13.0 MiB | 832 MiB | 1664 MiB |

**Winner**: `p16 + q4_0` — baseline-class PPL at 1/3 the VRAM. Saves **1216 MiB** at 32k vs baseline.

---

## 2. 35B Qwen3.6 A3B IQ4_XS — PPL + VRAM

Experimental flags: `LLAMA_MTP_PREFILL_CHUNK=1024 LLAMA_MTP_PREFILL_FORCE_MMQ=1 GGML_CUDA_ROCM_QUANT_PREFILL_F16=1`

| Config | K format | V format | PPL c512 | c512 VRAM | 32k VRAM |
|--------|----------|----------|----------|-----------|----------|
| baseline (std FA) | f16 | f16 | 1.0061 | 10.0 MiB | 640 MiB |
| p16 + f16 | I32 int8 | f16 | 1.0058 | 7.7 MiB | 490 MiB |
| **p16 + q4_0** | **I32 int8** | **q4_0** | **1.0061** | **4.1 MiB** | **260 MiB** |

**Winner**: All configs identical PPL (1.006). MoE model is large enough that K/V quantization is transparent. `p16 + q4_0` saves **380 MiB** at 32k vs baseline.

---

## 3. 27B Prefill Speed — DOT4 Flash Attention

`llama-bench`, batch=512. Standard FA route (no `QUANT_PREFILL_F16`).

| Prefill | t/s | % of pp512 |
|---------|-----|-----------|
| pp512 | 902 | 100% |
| pp1024 | 902 | 100% |
| pp2048 | 887 | 98% |
| pp4096 | 857 | 95% |
| pp8192 | 814 | 90% |
| pp16384 | 735 | 82% |
| pp32768 | 619 | 69% |
| pp65536 | — | OOM @ b512 |

No quadratic cliff — DOT4 FA O(n²) prefill holds 69% of peak at 32k.

**Decode**: 28.2 t/s (pure decode), 27.3 t/s (after pp512). Matmul-dominated.

---

## 4. Root Cause of PPL=45

**Attn rotation domain mismatch**: `attn_rot_v` was independently active from `--cache-type-v q4_0` while `attn_rot_k=false` (I32 K not quantized). V was rotated into a different domain than K.

**Fix** (one line, `src/llama-kv-cache.cpp`):
```cpp
attn_rot_v = ... && attn_rot_k;  // V rotation only when K also rotated
```

Without rotation: `p16 + q4_0` PPL improved from 45 → 1.0237.

---

## 5. Packed16 K Cache Architecture

```
V=f16:
  k_cur f32 → amax/MSE indexed pack → I32 packed16 K (256 bytes/row)
  PPL 1.0342, 24.5 MiB @ c512

V=q4_0/q8_0:
  k_cur f32 → amax/MSE indexed pack → I32 packed16 K (256 bytes/row)
  + q4_0 V (144 bytes/row)
  PPL 1.0237, 13.0 MiB @ c512
```

No persistent shadow K needed. No rotation. Single `attn_rot_v &= attn_rot_k` gate.

---

## 6. MTP + Packed16 at 128k — Server Load Test

**Loaded successfully** with full config:
- Qwen3.6 27B Q4_K_M MTP model
- DOT4 v4 flash attention, packed16 K + q4_0 V (target + draft)
- 128k context (`--ctx-size 131072`)
- MTP draft-mtp, n_max=2, batch 2048, ubatch 1024

**VRAM**: 20.3 GiB total (15.5 model + 3.8 KV + 1.0 compute), 3.6 GiB free.

**Generation**: Produced text successfully ("example # -*- coding: utf-8 -*-").

Initial crash (`hipStreamSynchronize` during graph capture) was from debug `cudaStreamSynchronize()` in rowdump code — removed.

---

## 7. q8_0 V Status

**Broken**: `USE_Q8_V` template lost in git revert. q8_0 V falls through to `dequant_q4_0`, producing garbage (PPL 2.51). Needs re-add.

---

## 8. Vulkan

Build broken — needs `shaderc` (glslc) from Vulkan SDK. Local deps wrapper script (`/tmp/glslc-localdeps-wrapper.sh`) missing. System has `libvulkan-dev` + `glslang-tools` but no `glslc` binary.

---

## Summary

| Metric | Winner | Value |
|--------|--------|-------|
| Best PPL | 35B p16+q4_0 | 1.0061 (same as baseline) |
| Best VRAM | 35B p16+q4_0 | 260 MiB @ 32k |
| Best 27B PPL | p16+q4_0 | 1.0237 (+1.1% vs baseline) |
| Best 27B VRAM | p16+q4_0 | 832 MiB @ 32k |
| Prefill at 32k | DOT4 FA | 619 t/s (69% of peak) |
| 128k MTP | Working | 20.3 GiB, 3.6 GiB free |
| Key fix | attn_rot_v gate | 1 line, PPL 45 → 1.02 |
