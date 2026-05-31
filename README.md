# llama.cpp — Packed16 FlashAttention for RDNA3

RDNA3-focused `llama.cpp` branch for packed16 K-cache FlashAttention. The
normal path is simple: build the branch, run `llama-server` or `llama-bench`,
and let the route selector pick the packed16 kernels automatically.

Packed16 is a **runtime K-cache layout**, not a new GGUF model format. K is
stored as I32 payload rows, with each 32-bit word carrying four packed 8-bit K
values, while f16 scales remain separate.

---

## Quick start

Build the branch, run a model, and let the branch pick the right packed16 route
automatically. **No route-forcing env vars are needed for normal use.**

You need a compatible GGUF model. The tested model families and example
Hugging Face sources are listed below under [Tested models](#tested-models).

```bash
git clone https://github.com/DrBearJew/llama.cpp
cd llama.cpp
git checkout tbq4-rdna3-experiment

cmake -S . -B build-rocm \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-server llama-bench -j$(nproc)
```

Run a server:

```bash
./build-rocm/bin/llama-server \
  --device ROCm0 \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --flash-attn on \
  --cache-type-v q4_0 \
  --ctx-size 40960 --parallel 1
```

Benchmark prefill:

```bash
./build-rocm/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -fa 1 -ngl 99 -p 512 -n 1
```

On the benchmark system used for the results below, this was approximately
**2628 tok/s** for 35B pp512 on the default packed16 route.

Optional: add route logging when benchmarking or debugging:

```bash
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 \
./build-rocm/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -fa 1 -ngl 99 -p 512 -n 1
```

Expected route log shape:

```text
PACKED16 FA ROUTE ... selected=dot4_mmq_gqa1 ...
```

If the verbose route log does not show a packed16 route, the benchmark is not
measuring the new kernels in this branch.

| Task | Use |
|---|---|
| Run inference | plain `llama-server` command above |
| Benchmark default path | plain `llama-bench` command above |
| Confirm route | add `GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1` |
| Compare kernels | [Advanced route-forcing section](#advanced-route-forcing-and-ab-tests) |
| Disable packed16 | [Runtime flags](#runtime-flags) |

---

## Headline results

RX 7900 XTX / gfx1100, `llama-bench -fa 1 -ngl 99`. Record your ROCm
and compiler versions when rerunning these numbers.

### Prefill (`nq > 1`)

| Model | Route | pp512 | pp1024 | pp2048 | pp4096 |
|---|---|---:|---:|---:|---:|
| 35B | DOT4-MMQ GQA1, default | 2628 | 2541 | 2320 | 2050 |
| 35B | DOT4-MMQ KSHARED, opt-in | 2649 | 2533 | — | — |
| 35B | PWMMA BM32 reg-out direct-V | **2707** | **2633** | — | 2569* |
| 35B | PWMMA BM16 | 2590 | 2394 | — | — |
| 35B | PWMMA BM64 512t | 2612 | 2578 | — | — |
| 27B | DOT4-MMQ GQA1, default | 894 | — | — | — |
| 27B | DOT4-MMQ KSHARED, opt-in | 905 | — | — | — |
| 27B | PWMMA BM32 reg-out direct-V | **929** | — | — | — |
| 27B | PWMMA BM64 512t | 922 | — | — | — |

\* pp1024+ configuration.

### Decode (`nq = 1`)

Decode uses DOT4 decode kernels, not the prefill WMMA kernels.

| Model | tg128, packed16 + DOT4 decode |
|---|---:|
| 35B | 92.8 tok/s |
| 27B | 28.7 tok/s |

### What these numbers show

- DOT4-MMQ is the production default packed16 prefill route.
- PWMMA BM32 reg-out direct-V is the fastest measured prefill route on the
  listed workloads: +3.0% over DOT4-MMQ on 35B pp512 and +3.9% on 27B pp512.
- A clean upstream q8_0 VEC FA baseline table is still TODO; current tables
  compare the packed16 route family and measured variants.

---

## Supported hardware

| Hardware | Status | Notes |
|---|---|---|
| RX 7900 XTX / gfx1100 | Tested | Primary development and benchmark target |
| RX 7900 XT / gfx1100 | Expected | Same architecture class; not separately reported here |
| Other RDNA3 | Unknown | May need target-specific validation |
| RDNA2 / gfx1030 | Not targeted | This branch focuses on RDNA3 packed16/WMMA work |
| RDNA4 / gfx12xx | Untested | Needs separate route/compiler validation |
| CDNA / MI300X | Not targeted | Different architecture assumptions |
| NVIDIA/CUDA | Not targeted | Branch is ROCm/RDNA3-specific |

## Tested models

| Model / family | Status | Notes |
|---|---|---|
| Qwen3.6 35B-A3B MoE GGUF | Tested | Main 35B benchmark target |
| Qwen3.6 27B MTP GGUF | Tested | MTP + packed16 route target |
| Other Qwen GGUFs | Unknown | May work if shapes and cache assumptions match |
| Llama-family GGUFs | Untested | Not the target of this branch |
| Other MoE families | Untested | GQA/MoE assumptions may differ |

Model sources used during development include GGUF releases from
[llmfan46](https://huggingface.co/llmfan46),
[HauhauCS](https://huggingface.co/HauhauCS),
[havenoammo](https://huggingface.co/havenoammo), and
[Radamanthys11](https://huggingface.co/Radamanthys11).

---

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

## Build

Standard ROCm build for gfx1100:

```bash
cmake -S . -B build-rocm \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-bench llama-server -j$(nproc)
```

Requirements:

- ROCm HIP toolchain
- rocWMMA headers/libraries available to CMake for the PWMMA path
- gfx1100-class RDNA3 GPU; RX 7900 XTX is the primary target

Optional helper script:

```bash
bash scripts/configure-rocm-gfx1100-wmma.sh
```

The helper script is only a convenience wrapper around CMake. It creates
`build-rocm-fixed`, sets `GPU_TARGETS=gfx1100`, uses `/opt/rocm/bin/amdclang++`,
and enables the ROCm/FlashAttention options used during development. You do not
need it if your normal CMake ROCm build works.

Optional ROCm + Vulkan build:

```bash
cmake -S . -B build-rocm-vulkan \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_HIP_FLAGS="-DRDNA2_MATMUL_OPT_V1=1" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-vulkan --target llama-server llama-bench -j
```

---

## Run notes

Normal use should not require route forcing. The branch allocates packed16 K
cache by default on HIP and auto-selects the packed16 FlashAttention route for
the supported shapes.

For ordinary testing, use the simple `llama-server` and `llama-bench` commands
from the quick start. Use the advanced route-forcing commands near the end only
when comparing kernels or debugging dispatch.

### MTP settings

MTP is supported for the tested 27B and 35B targets.

```bash
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
--cache-type-v-draft q4_0
LLAMA_MTP_PREFILL_CHUNK=1024  # match --ubatch-size
```

Routing impact:

- `MTP_DRAFT` (`nq = 1`) routes to DOT4 decode.
- `MTP_VERIFY` (`nq >= 2`) routes to DOT4-MMQ GQA1 when packed16 is active.

Disable DOT4-MMQ for MTP verify/oracle comparison:

```bash
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0
```

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

## License

Follows upstream `llama.cpp` licensing terms.
