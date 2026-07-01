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

## Runtime flags

| Flag | Purpose |
|---|---|
| `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` | Explicitly enable packed16 K cache; currently default-on for HIP |
| `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=0` | Disable packed16 K cache for A/B testing |
| `GGML_CUDA_ROCM_PACKED16_DISABLE=1` | Disable all packed16 K-cache allocation |
| `GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1` | Log route decisions; recommended for benchmarks |
| `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=0` | Disable DOT4-MMQ and fall back to DOT4-KQ oracle path |
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


## Benchmark details

RX 7900 XTX / gfx1100, `llama-bench -fa 1 -ngl 99` unless noted. Record your
ROCm and compiler versions when rerunning these numbers.

### Long-context server smoke

Qwen3.6 27B Q4_K_M MTP, `llama-server`, `ctx=49152`, `--cache-type-v q4_0`,
`--cache-type-v-draft q4_0`, packed16/I32 K, production auto route on commit
`0ea6db58e`.

| Prompt / predict | Route | Prompt tok/s | Decode tok/s | SHA |
|---|---|---:|---:|---|
| 32k / tg128 | PWMMA BM64 i8-QK PV-WMMA DBV | **589.43** | 33.17 | `33fc0c55` |

Route evidence: `selected=pwmma_bm64_i8qk_pvwmma_dbv`, `desc_layout=0`,
packed16/I32 K, q4 V, no BM32 prefill route selected. Draft acceptance and very
short tg32 runs are intentionally omitted from this table; these rows are
route/hash/speed smoke tests, not acceptance-quality benchmarks.

Route smoke snapshot from the same model/settings, pp32k-style prompt smoke
(`tokens_evaluated=32768`):

| K selection | V selection | Prompt tok/s | Decode tok/s | SHA | Notes |
|---|---|---:|---:|---|---|
| omit `--cache-type-k` → packed16 / 272B K | `q4_0` | ~583–589 | ~33 | `33fc0c55` | headline prefill path |
| `--cache-type-k q8_0` or `q4_0` → packed8 / 144B K | `q4_0` | ~560–566 | ~31 | `33fc0c55` | smaller K, not faster yet |
| q4 K | `q8_0` | ~550 | ~39 | `33fc0c55` | faster decode, slower prefill |
| q4 K | `f16` | ~551 | ~38 | `33fc0c55` | faster decode, slower prefill |

Older 8k clean auto-table smoke: prompt ~744 tok/s, decode ~51 tok/s, SHA `4219d799`.

### Historical llama-bench prefill (`nq > 1`)

| Model | Route | pp512 | pp1024 | pp2048 | pp4096 |
|---|---|---:|---:|---:|---:|
| 35B | DOT4-MMQ GQA1, historical/pinned | 2628 | 2541 | 2320 | 2050 |
| 35B | DOT4-MMQ KSHARED, opt-in | 2649 | 2533 | — | — |
| 35B | PWMMA BM32 reg-out direct-V, production auto | **2707** | **2633** | — | 2569* |
| 35B | PWMMA BM16 | 2590 | 2394 | — | — |
| 35B | PWMMA BM64 512t | 2612 | 2578 | — | — |
| 27B | DOT4-MMQ GQA1, historical/pinned | 894 | — | — | — |
| 27B | DOT4-MMQ KSHARED, opt-in | 905 | — | — | — |
| 27B | PWMMA BM32 reg-out direct-V, production auto | **929** | — | — | — |
| 27B | PWMMA BM64 512t | 922 | — | — | — |

\* pp1024+ configuration.

### Decode (`nq = 1`)

Decode uses DOT4 decode kernels, not the prefill WMMA kernels.

| Model | tg128, packed16 + DOT4 decode |
|---|---:|
| 35B | 92.8 tok/s |
| 27B | 28.7 tok/s |

Takeaways:

- PWMMA BM64 i8-QK PV-WMMA DBV is the production auto packed16 prefill route
  for target Qwen row/default shapes from `nk >= 512`; the first long prefill
  chunk now goes BM64 DBV instead of BM32 direct-V.
- Optional packed8/packed4 q4 K storage is operational and route-validated,
  but current prefill expands q4 K to i8 before WMMA and is not faster than
  the packed16 headline baseline.
- DOT4-MMQ/PDMQ remains available for route-pinned validation, small-Q/MTP
  roles, decode, and experimental V formats.
- A clean upstream q8_0 VEC FA baseline table is still TODO; current tables
  compare the packed16 route family and measured variants.

### V-cache quality smoke

Short WikiText-2 raw smoke on the 27B MTP model, `ctx=512`, `chunks=4`
(~1020 evaluated tokens/candidate), all with `K=i32` and
`selected=rocm_packed16_dot4_mmq`. This is a fast sanity check, not a full
quality benchmark.

![WikiText-2 V-cache quality smoke](assets/wikitext-v-cache-quality-20260531.png)

| V cache | PPL / ratio vs f16 V | Mean KLD vs f16 V | Median KLD | Same top token |
|---|---:|---:|---:|---:|
| f16 | `5.6891 ± 0.4459` | baseline | baseline | baseline |
| q4_0 | `1.00198 ± 0.00422` ratio | `0.004550 ± 0.000338` | `0.001818` | `97.06%` |
| q8_0 | `1.00101 ± 0.00336` ratio | `0.002825 ± 0.000334` | `0.000863` | `97.94%` |

Takeaway: q4_0 is the default compression choice; q8_0 is the higher-precision
choice and is measurably closer to f16 V on this smoke.

### 128k active MTP VRAM smoke

A 128k-context server-ready VRAM smoke on the 27B MTP GGUF with active
`draft-mtp`, `q4_0` V, and `--spec-draft-type-v q4_0`. The ROCm packed16 run
omits main and draft K CLI overrides; the Vulkan comparison uses normal f16
main K plus q4 V.

![128k active MTP VRAM smoke](assets/active-mtp-vram-128k-20260531-v3.png)

| Run | Total VRAM used | Delta over idle |
|---|---:|---:|
| ROCm packed16/I32 route + q4 V active MTP | `21.760 GiB` | `21.079 GiB` |
| Vulkan f16 K + q4 V active MTP | `23.180 GiB` | `22.500 GiB` |

Measured saving: Vulkan uses `+1.420 GiB` more total VRAM (`+1.422 GiB` delta
over idle). ROCm route evidence included the packed16 DOT4/MMQ route and
`PDMQ QK probe PASSED`.

---


## Experiment journey

This branch is the second-stage result of the DOT4 FlashAttention work. The
earlier prototype asked a narrower question: can RDNA3 `sudot4` make quantized
KV-cache attention fast enough to recover f16-V-like decode speed while keeping
q4_0-style VRAM savings? That work produced the first packed16 K-cache, DOT4
QK kernels, BN64 decode, split-K decode, and route-contract experiments.

The main lessons from that prototype were:

- **Attention was not just math-bound.** K layout, V layout, routing, and cache
  representation mattered as much as the DOT4 instruction.
- **Packed16 was the useful abstraction.** Storing K as I32 payload rows plus
  f16 scales gave the kernels aligned, DOT4-ready data without a per-tile
  dequant buffer.
- **Route contracts mattered.** Forced routes and verbose dispatch logs were
  necessary to avoid benchmarking the wrong fallback path.
- **Decode and prefill wanted different kernels.** BN64/split-K decode solved
  one side of the problem; this branch focuses on the packed16 prefill route
  family with DOT4-MMQ and PWMMA variants.

The older prototype notes live at
[DrBearJew/dot4-flash-attention](https://github.com/DrBearJew/dot4-flash-attention).
They are useful background, but this branch is the cleaner llama.cpp integration
for the current packed16 FA path.

---

## Credits and related work

This branch builds on:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base runtime,
  ggml backends, FlashAttention, and upstream infrastructure.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)
  — MTP/TurboQuant fork foundation, tensor sharing, CUDA TBQ4 FA.
- [DrBearJew/dot4-flash-attention](https://github.com/DrBearJew/dot4-flash-attention)
  — earlier DOT4 FlashAttention prototype, packed16 K-cache experiment notes,
  and the path that led to this branch.
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

