# llama.cpp — Packed16 FlashAttention for RDNA3

This is an RDNA3-focused llama.cpp fork (RX 7900 XTX / gfx1100) with two
production FlashAttention backends for quantized KV caches:

1. **DOT4-MMQ** — 2628 tok/s pp512 (35B), production default
2. **PWMMA BM32 reg-out** — 2707 tok/s pp512 (35B), champion fallback

Both operate on a **packed16 K cache** — an I32-packed persistent K
representation that eliminates per-tile dequantization and enables
high-throughput matrix-multiply attention on RDNA3.

Target models: Qwen3.6 27B MTP and Qwen3.6 35B-A3B MoE.

---

## Why packed16 FlashAttention matters

### The problem

Standard q8_0 KV caches require per-tile dequantization inside the attention
kernel: for every K tile, the kernel unpacks 8-bit integers to f16, multiplying
by per-group scales. This dequantization competes with the actual Q·K^T matmul
for memory bandwidth and ALU throughput. At 256 dimensions and batch sizes of
512-4096, the dequant overhead is significant — often 30-50% of attention time.

A second problem is **V tensor transposition**. Standard FA expects V in
transposed layout (sequence-major), but KV caches store V contiguously
(sequence-first). The transposition cost shows up as either a GPU-side copy
or indirect indexing inside the attention kernel.

### The solution: packed16 K cache + direct-V

**Packed16 K cache** stores K as dense 32-bit integer rows. Each 32-bit word
holds four 8-bit packed values (one per byte). Scales are stored separately as
f16. This gives three key advantages:

1. **Zero-copy KQ**: The DOT4-MMQ and PWMMA kernels read packed integers
   directly — extract one byte with bitmask, multiply by scale, feed into
   matrix multiply. No intermediate dequant buffer.

2. **Coalesced global reads**: 32-bit words are naturally aligned and coalesced
   on RDNA3's L1/L2 cache hierarchy. Standard q8_0 reads 8-bit bytes with
   non-unit stride.

3. **I32 selector**: The packed16 K cache uses a 32-bit integer per element
   instead of 8-bit. This is detectable by the FA route selector
   (`get_best_fattn_kernel()`), enabling automatic backend dispatch.

**Direct-V** eliminates the V transposition copy by having each kernel declare
its preferred V layout. FA kernels request D-contiguous V (layout=FA), VEC
kernels request transposed V. The KV cache stores V contiguously and the
scheduler assigns the correct view per consumer — zero copies.

### What we gain

| Metric | Before (q8_0 VEC FA) | After (packed16 DOT4-MMQ) |
|--------|---------------------|---------------------------|
| K dequant per tile | Full 8→f16 + scale multiply | Byte extract + f16 scale |
| K memory per row (D=256) | 256 bytes + 8 scales | 64 ints (256 bytes) + 8 scales |
| V transposition | GPU-side copy or indirect | Zero — consumer-driven layout |
| K coalescing | 8-bit, non-unit stride | 32-bit, unit stride |

---

## Performance

All numbers on RX 7900 XTX (gfx1100, 24 GB), ROCm 6.4, amdclang++.
Benchmarked with `llama-bench -fa 1 -ngl 99`.

### Prefill (nq > 1)

| Variant | 35B pp512 | 35B pp1024 | 35B pp2048 | 35B pp4096 |
|---------|----------|-----------|-----------|-----------|
| **DOT4-MMQ GQA1** (default) | 2628 | 2541 | 2320 | 2050 |
| DOT4-MMQ KSHARED (opt-in) | 2649 | 2533 | — | — |
| **PWMMA BM32 reg-out** (champion) | **2707** | **2633** | — | 2569* |
| PWMMA BM16 | 2590 | 2394 | — | — |
| PWMMA BM64 512t | 2612 | 2578 | — | — |

| Variant | 27B pp512 | 27B pp1024 |
|---------|----------|-----------|
| DOT4-MMQ GQA1 | 894 | — |
| **PWMMA BM32 reg-out** | **929** | — |
| PWMMA BM64 512t | 922 | — |
| DOT4-MMQ KSHARED | 905 | — |

\* pp1024+

### Decode (nq = 1)

| Model | tg128 (packed16 + DOT4 decode) |
|-------|-------------------------------|
| 35B | 92.8 tok/s |
| 27B | 28.7 tok/s |

Decode uses DOT4 decode kernels (BN64/split-K), not the prefill WMMA kernels.

### Speedup summary

On 35B pp512, BM32_regout_directv is **+3% faster than DOT4-MMQ** (2707 vs
2628) and **substantially faster than standard q8_0 VEC FA** (the packed16
format alone eliminates per-tile dequant overhead).

On 27B, BM32_regout_directv is **+3.9% faster than DOT4-MMQ** (929 vs 894).

---

## Architecture

### Packed16 K cache

The K cache persists as I32 rows. Each q8_0 group (32 dimensions, 1 scale)
is stored as eight 32-bit integers, each holding 4 packed bytes. Scales are
f16. Layout:

```
Row j, dimensions [0..255]:
  int payload[j * 64 + 0]  → bytes for dims  0,1,2,3
  int payload[j * 64 + 1]  → bytes for dims  4,5,6,7
  ...
  int payload[j * 64 + 63] → bytes for dims 252,253,254,255
  half scales[j * 8 + 0]   → scale for dims 0..31
  ...
  half scales[j * 8 + 7]   → scale for dims 224..255
```

Total per row: 64 ints (256B) + 8 halfs (16B) = 272 bytes. Same as q8_0
(256B data + 16B scales), but 32-bit aligned.

### Consumer-driven V layout

V is stored contiguously in KV cache (sequence-first). Each consumer kernel
declares its preferred layout:

```cpp
enum { PWMMA_V_LAYOUT_FA = 0, ... };
cgraph_local_set_v_layout(tensor, PWMMA_V_LAYOUT_FA);
```

FA kernels request D-contiguous layout (stride=2 for f16 V). VEC kernels
request transposed layout. Zero copies — the scheduler assigns the correct
view per consumer.

---

## DOT4-MMQ (production default)

DOT4-MMQ is the default packed16 FlashAttention kernel. It uses:

- **M16N64 tile** for QK: 16 Q rows × 64 K columns per CTA
- **DOT4 I4 acceleration**: 4-way dot-product per thread on packed I4 values
- **Online softmax** with shared-memory probs buffer
- **Staged V tile** in LDS (32 KiB for M16N64, 0 for M4N64)
- **GQA1 only** in auto-route (GQA2 is slower beyond pp512)

```
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1    # enable packed16 K cache
# DOT4-MMQ auto-loaded when packed16 K is active
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0             # disable DOT4-MMQ (escape hatch)
```

Experimental DOT4-MMQ variants:

```bash
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared    # K in LDS, +1.5% 35B, +15% 27B
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared_stagev
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared_directv   # no V staging, slower
```

KSHARED caches K payload+scales in LDS once per tile for cooperative QK matmul.
On 27B (gqa=6, K reused 6× per Q head), this gives +15%. On 35B (gqa=8), +1.5%.

---

## PWMMA kernel family

PWMMA kernels use raw RDNA3 WMMA builtins (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`)
for QK matmul. All variants read from the same packed16 K cache.

### Available variants

| Variant | Impl | BM | Waves | CTA threads | out[] | LDS | Notes |
|---------|------|----|-------|-------------|-------|-----|-------|
| BM16 smem | 0 | 16 | 1 | 256 | smem | ~60K | Original, stable |
| BM32 regout stagev | 1 | 32 | 2 | 256 | 32 | ~60K | Staged V |
| **BM32 regout directv** | **2** | **32** | **2** | **256** | **32** | **~20K** | **Champion** |
| BM64 regout directv 512t | 5 | 64 | 4 | 512 | 32 | ~9K | Stable, slower |
| BM16 GQA2 | — | 16 | 1 | 256 | smem | ~60K | V-tile reuse |

### Selection

```bash
GGML_CUDA_ROCM_PACKED16_WMMA_TILE=1              # enable PWMMA
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile  # force PWMMA
GGML_CUDA_ROCM_PACKED16_WMMA_BM=32               # 16, 32 (default), or 64
GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv  # 0-5
```

BM32 regout directv is the recommended PWMMA variant:
- Float32 accumulator in registers (no half-precision compromise)
- Direct V load (no V tile staging in LDS)
- Beats DOT4-MMQ at pp1024+ on 35B and all lengths on 27B
- 20 KiB LDS vs 60 KiB for smem variants

---

## Route policy

The route selector (`get_best_fattn_kernel()`) auto-selects based on K type:

```
K type → I32 (packed16) ?
  ├─ nq > 1 → DOT4-MMQ GQA1 (default)
  │   └─ DOT4-MMQ unavailable? → PWMMA BM32 (fallback)
  │       └─ PWMMA unavailable? → DOT4-KQ (safety net)
  └─ nq = 1 → DOT4 decode BN64 / split-K
```

**Never auto-selected**: DOT4-MMQ GQA2, PWMMA BM64, PWMMA GQA2, CPU FA,
old BM16 smem variant, or KSHARED.

The route contract is strict: if `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile`
is set and PWMMA cannot run, the kernel **aborts** — no silent DOT4 fallback.

```bash
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1    # log route decisions
```

---

## Build

WMMA-only build for gfx1100:

```bash
bash scripts/configure-rocm-gfx1100-wmma.sh
cd build-rocm-fixed
cmake --build . --target llama-bench llama-server -j$(nproc)
```

Requires: ROCm 6.2+, `amdclang++`, gfx1100 (RX 7900 XTX/XT).

Key CMake flags:
- `CMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++`
- `GPU_TARGETS=gfx1100`
- `-Wno-gpu-maybe-exceed-local-memory` (suppress LDS-size warnings for WMMA kernels)

Multi-backend build (ROCm + Vulkan):

```bash
cmake -S . -B build-rocm-vulkan \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_HIP_FLAGS="-DRDNA2_MATMUL_OPT_V1=1" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-vulkan --target llama-server llama-bench -j
```

---

## Run recipes

### 35B MoE with packed16 FA (auto-route)

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
./build-rocm-fixed/bin/llama-server \
  --device ROCm0 \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --ctx-size 40960 --parallel 1
```

### Force PWMMA champion

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile \
GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv \
./build-rocm-fixed/bin/llama-bench -m model.gguf -fa 1 -ngl 99 -p 512 -n 1
```

### A/B benchmark sweep

```bash
# Baseline: DOT4-MMQ (auto)
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
./build-rocm-fixed/bin/llama-bench -m model.gguf -fa 1 -ngl 99 -p 512,1024,2048 -n 1

# PWMMA BM32 regout
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile \
GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv \
./build-rocm-fixed/bin/llama-bench -m model.gguf -fa 1 -ngl 99 -p 512,1024,2048 -n 1

# KSHARED
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 \
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared \
./build-rocm-fixed/bin/llama-bench -m model.gguf -fa 1 -ngl 99 -p 512,1024 -n 1
```

---

## MTP (Multi-Token Prediction)

MTP is supported for both 27B and 35B. Key settings:

```bash
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
--cache-type-k-draft q8_0 --cache-type-v-draft q4_0
LLAMA_MTP_PREFILL_CHUNK=1024  # match --ubatch-size
```

MTP impact on FA routing:
- **MTP_DRAFT** (nq=1): routes to DOT4 decode kernel — unchanged
- **MTP_VERIFY** (nq≥2): auto-routes to DOT4-MMQ GQA1 when packed16 is active

Disable DOT4-MMQ for MTP verify if needed:

```bash
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0  # reverts to DOT4-KQ oracle
```

---

## Key source files

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | Route selection, dispatch, scheduler |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Route policies, auto-selection logic |
| `ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cuh` | DOT4-MMQ kernel (GQA1/GQA2/KSHARED) |
| `ggml/src/ggml-cuda/fattn-packed16-wmma-tile.cuh` | PWMMA kernels (BM16/32/64, regout variants) |
| `ggml/src/ggml-cuda/fattn-packed16-wmma-builtin.cuh` | Raw WMMA builtins, I8 extraction |
| `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh` | DOT4-KQ oracle (safety net) |
| `ggml/src/ggml-cuda/fattn-dot4-q8k-decode.cuh` | DOT4 decode kernel (nq=1) |

---

## Related work

This branch builds on:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base runtime,
  ggml backends, FlashAttention, MTP/TBQ upstream.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)
  — MTP/TurboQuant fork foundation, tensor sharing, CUDA TBQ4 FA.
- [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) —
  RDNA3 rocWMMA GEMM reference (autotuner, config lookup, LDS buffering).
- [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) —
  official AMD matrix instruction calculator (WMMA shapes, throughput).
- [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire) —
  dispatch-screening and WMMA references.
- [Stormrage34/llama.cpp-turboquant-hip](https://github.com/Stormrage34/llama.cpp-turboquant-hip) —
  first AMD VEC TurboQuant-style path.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) —
  original TurboQuant block-format reference.
- Model sources: [llmfan46](https://huggingface.co/llmfan46),
  [HauhauCS](https://huggingface.co/HauhauCS),
  [havenoammo](https://huggingface.co/havenoammo),
  [Radamanthys11](https://huggingface.co/Radamanthys11) — Qwen3.6/MTP GGUF releases.

---

## License

Follows upstream llama.cpp licensing terms.
