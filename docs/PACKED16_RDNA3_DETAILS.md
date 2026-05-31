# Packed16 RDNA3 technical notes

Advanced details for the `tbq4-rdna3-experiment` README. Most users only need the root README quick start.

## How it works

Standard q8_0 KV-cache attention spends work inside the attention kernel
unpacking K tiles from 8-bit values to f16 and applying per-group scales. On
large prefill batches, that dequantization competes with the actual Q·K^T
matrix multiply for bandwidth and ALU time.

This branch changes the K-cache representation and route policy:

1. **Packed16 K cache** stores K rows as 32-bit words. Each word carries four
   packed 8-bit values; f16 scales remain separate.
2. **DOT4-MMQ and PWMMA kernels** read the packed words directly. They extract
   bytes, apply scales, and feed RDNA3 matrix-multiply paths without creating
   an intermediate dequant buffer.
3. **Direct-V layout** lets FlashAttention consumers request D-contiguous V
   while VEC consumers can request transposed V. The scheduler assigns the
   right view instead of forcing a V transposition copy.
4. **Route selection** detects packed16 K by I32 K-cache type and dispatches to
   packed16-specific prefill/decode kernels.

| Metric | Standard q8_0 VEC FA | Packed16 route |
|---|---|---|
| K dequant per tile | Full 8→f16 + scale multiply | Byte extract + f16 scale |
| K memory per D=256 row | 256 bytes + 8 scales | 64 ints / 256 bytes + 8 scales |
| K alignment | 8-bit payload | 32-bit aligned payload |
| V transposition | copy or indirect indexing | consumer-selected layout |

---


## Validation checklist

Before trusting a benchmark, verify route attribution and basic correctness.

| Check | Why |
|---|---|
| Enable `GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1` | Confirms the intended packed16 route is active |
| Force `GGML_CUDA_FA_ROUTE_REQUIRE=...` | Prevents silent fallback when testing a route |
| Compare against DOT4-KQ with `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0` | Provides oracle/safety-net comparison |
| Repeat pp512/pp1024 runs | Catches unstable route or clock variance |
| Watch for NaNs/divergence | Required for custom attention kernels |
| Run a `llama-server` smoke test | Confirms the non-benchmark path works |

A clean upstream q8_0 VEC FA baseline table is still TODO and should be added
before using this README as a broad speedup claim against upstream.

---


## Architecture notes

### Packed16 K-cache layout

K persists as I32 rows. For a D=256 row, each q8_0 group of 32 dimensions is
stored as eight 32-bit integers, each holding four packed bytes. Scales are f16.

```text
Row j, dimensions [0..255]:
  int payload[j * 64 + 0]  -> bytes for dims   0,  1,  2,  3
  int payload[j * 64 + 1]  -> bytes for dims   4,  5,  6,  7
  ...
  int payload[j * 64 + 63] -> bytes for dims 252,253,254,255
  half scales[j * 8 + 0]   -> scale for dims   0..31
  ...
  half scales[j * 8 + 7]   -> scale for dims 224..255
```

Total per D=256 row: 64 ints / 256 bytes + 8 halfs / 16 bytes = 272 bytes.
That matches q8_0 payload+scale size while giving 32-bit-aligned K payloads.

### Direct-V layout

V is stored contiguously in KV cache. Consumers declare their preferred layout:

```cpp
enum { PWMMA_V_LAYOUT_FA = 0, ... };
cgraph_local_set_v_layout(tensor, PWMMA_V_LAYOUT_FA);
```

FlashAttention kernels request D-contiguous V. VEC kernels request transposed V.
The scheduler assigns the correct view per consumer instead of requiring a
separate V transpose copy.

### DOT4-MMQ

DOT4-MMQ is the default packed16 prefill kernel:

- M16N64 QK tile: 16 Q rows × 64 K columns per CTA
- DOT4 I4 acceleration on packed values
- online softmax
- shared-memory probability buffer
- staged V tile in LDS for the default path
- GQA1 only in auto-route

KSHARED caches K payload+scales in LDS once per tile for cooperative QK matmul.
It is opt-in because it is workload-dependent.

### PWMMA

PWMMA kernels use raw RDNA3 WMMA builtins:

```cpp
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32
```

Available variants:

| Variant | Impl | BM | Waves | CTA threads | Output | LDS | Notes |
|---|---:|---:|---:|---:|---|---:|---|
| BM16 smem | 0 | 16 | 1 | 256 | smem | ~60K | original, stable |
| BM32 regout stagev | 1 | 32 | 2 | 256 | registers | ~60K | staged V |
| BM32 regout direct-V | 2 | 32 | 2 | 256 | registers | ~20K | champion route |
| BM64 regout direct-V 512t | 5 | 64 | 4 | 512 | registers | ~9K | stable, slower |
| BM16 GQA2 | — | 16 | 1 | 256 | smem | ~60K | V-tile reuse experiment |

---


## Key source files

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | Route selection, dispatch, scheduler |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Route policies and auto-selection logic |
| `ggml/src/ggml-cuda/fattn-packed16-dot4-mmq.cuh` | DOT4-MMQ kernel family |
| `ggml/src/ggml-cuda/fattn-packed16-wmma-tile.cuh` | PWMMA BM16/BM32/BM64 kernels |
| `ggml/src/ggml-cuda/fattn-packed16-wmma-builtin.cuh` | Raw WMMA builtins and I8 extraction |
| `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cuh` | DOT4-KQ oracle/safety-net route |
| `ggml/src/ggml-cuda/fattn-dot4-q8k-decode.cuh` | DOT4 decode kernel for `nq = 1` |

---


## Kernel routes

| Route | Auto-selected? | Use case | Notes |
|---|---:|---|---|
| DOT4-MMQ GQA1 | Yes | packed16 prefill | production default |
| PWMMA BM32 reg-out direct-V | Fallback / forced | fastest measured prefill route | route can be required explicitly |
| DOT4 decode BN64 / split-K | Yes for `nq = 1` | decode | used by MTP draft/decode |
| DOT4-KQ | Fallback/oracle | validation and safety net | used when DOT4-MMQ is disabled |
| DOT4-MMQ KSHARED | No | experimental K-in-LDS variant | opt-in only |
| PWMMA BM64 / GQA2 / old BM16 smem | No | experiments | not selected by default |

Route policy:

```text
K type -> I32 packed16?
  ├─ nq > 1 -> DOT4-MMQ GQA1
  │   └─ if unavailable -> PWMMA BM32
  │       └─ if unavailable -> DOT4-KQ safety net
  └─ nq = 1 -> DOT4 decode BN64 / split-K
```

If `GGML_CUDA_FA_ROUTE_REQUIRE=...` is set and the required route cannot run,
the kernel aborts instead of silently falling back.

---


## Advanced: route forcing and A/B tests

Most users can skip this section. These flags are for kernel comparisons, route
contract tests, and debugging.

### Force PWMMA BM32 reg-out direct-V

```bash
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile \
GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv \
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 \
./build-rocm/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -fa 1 -ngl 99 -p 512,1024,2048 -n 1
```

### Force DOT4-MMQ

```bash
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 \
./build-rocm/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -fa 1 -ngl 99 -p 512,1024,2048 -n 1
```

### Test KSHARED

```bash
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared \
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 \
./build-rocm/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -fa 1 -ngl 99 -p 512,1024 -n 1
```


## Runtime flags

| Flag | Purpose |
|---|---|
| `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` | Explicitly enable packed16 K cache; currently default-on for HIP |
| `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=0` | Disable packed16 K cache for A/B testing |
| `GGML_CUDA_ROCM_PACKED16_DISABLE=1` | Disable all packed16 K-cache allocation |
| `GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1` | Log route decisions; recommended for benchmarks |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0` | Disable DOT4-MMQ and fall back to DOT4-KQ oracle path |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ_IMPL=kshared` | Opt into KSHARED DOT4-MMQ variant |
| `GGML_CUDA_ROCM_PACKED16_WMMA_TILE=0` | Disable PWMMA route family |
| `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_wmma_tile` | Require PWMMA route; abort if unavailable |
| `GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq` | Require DOT4-MMQ route; abort if unavailable |
| `GGML_CUDA_ROCM_PACKED16_WMMA_BM=32` | Select PWMMA block-M size, e.g. 16/32/64 |
| `GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm32_regout_directv` | Select champion PWMMA implementation |

---


## Known limitations

- Primary target is RX 7900 XTX / gfx1100.
- Packed16 K cache is default-on for HIP in this branch. Set
  `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` only to make the benchmark
  contract explicit.
- Disable packed16 with `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=0` or `GGML_CUDA_ROCM_PACKED16_DISABLE=1`.
- PWMMA requires ROCm compiler support for RDNA3 WMMA builtins.
- DOT4-MMQ GQA2, PWMMA BM64, BM16 smem, and KSHARED are not auto-selected.
- Decode uses DOT4 decode kernels, not the prefill WMMA kernels.
- `GGML_CUDA_FA_ROUTE_REQUIRE=...` intentionally aborts if the route cannot run.
- Non-Qwen3.6 models and non-RDNA3 hardware need separate validation.

---



## Run notes

Normal use should not require route forcing. The branch allocates packed16 K
cache by default on HIP and auto-selects the packed16 FlashAttention route for
the supported shapes.

For ordinary testing, use the simple `llama-server` and `llama-bench` commands
from the quick start. Route-forcing and kernel A/B commands are in
[technical notes](docs/PACKED16_RDNA3_DETAILS.md).

MTP settings used during development:

```bash
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
--cache-type-v-draft q4_0
LLAMA_MTP_PREFILL_CHUNK=1024  # match --ubatch-size
```

---


## Related work

This branch builds on:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base runtime,
  ggml backends, FlashAttention, and upstream infrastructure.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)
  — MTP/TurboQuant fork foundation, tensor sharing, CUDA TBQ4 FA.
- [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) — RDNA3
  rocWMMA GEMM reference, autotuner, config lookup, LDS buffering.
- [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
  — official AMD matrix-instruction calculator for WMMA shapes and throughput.
- [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire) —
  dispatch-screening and WMMA references.
- [Stormrage34/llama.cpp-turboquant-hip](https://github.com/Stormrage34/llama.cpp-turboquant-hip)
  — AMD VEC TurboQuant-style path.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
  — original TurboQuant block-format reference.

---

