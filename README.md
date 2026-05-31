# llama.cpp — Packed16 FlashAttention for RDNA3

RDNA3-focused `llama.cpp` branch for packed16 K-cache FlashAttention. The
normal path is simple: build the branch, run `llama-server` or `llama-bench`,
and let the route selector pick the packed16 kernels automatically.

Packed16 is a **runtime K-cache layout**, not a new GGUF model format. K is
stored as I32 payload rows, with each 32-bit word carrying four packed 8-bit K
values, while f16 scales remain separate.

---

## PROPER STARTING OPTIONS — I32/DOT4 FlashAttention

For this branch's RDNA3 FlashAttention work, **K is the packed16/I32 route**.
Do **not** pass a K cache-type flag for this path. In particular, do not use
`--cache-type-k q8_0` when testing the I32/DOT4 route.

Use the V cache type to choose the value format, and let packed16 allocate the
physical I32 K payload/scales:

```bash
LLAMA_MTP_ENABLE_FA=1 \
LLAMA_MTP_PREFILL_CHUNK=1024 \
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
./build-rocm/bin/llama-server \
  --device ROCm0 \
  --model /path/to/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --flash-attn on \
  --cache-type-v q4_0 \
  --ctx-size 40960 --batch-size 2048 --ubatch-size 2048 \
  --parallel 1 --no-warmup \
  --spec-type draft-mtp --spec-default \
  --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --spec-draft-type-v q4_0 \
  --spec-draft-prio 2 --spec-draft-prio-batch 2
```

The generic quantized-matmul/f16-temp knobs `LLAMA_MTP_PREFILL_FORCE_MMQ` and
`GGML_CUDA_ROCM_QUANT_PREFILL_F16` are intentionally omitted here; the packed16
FlashAttention route is governed by `GGML_CUDA_ROCM_PACKED16_DOT4_MMQ` plus the
route contract above.

Use one of these V choices:

```text
--cache-type-v q4_0       # default: 4.5 bits/V-value; 2.25-bit contribution to total K+V average
--cache-type-v q8_0       # higher precision: 8.5 bits/V-value; 4.25-bit contribution to total K+V average
```

The intended q4 architecture is **not i16 V**. It is packed q4 V payload plus
f16 scales feeding only the `P @ V` side; QK remains packed16 I32 DOT4 K.
For users who want more V precision, `q8_0` V keeps the same packed16 I32 K path
and raises V from 4.5 to 8.5 bits/V-value. In whole-KV VRAM accounting, q4 V
contributes 2.25 bits to the total K+V average, while q8 V contributes 4.25 bits;
packed16-K + q8-V is about 17 bits per K+V pair.

`tbq4_0` is no longer a proper starting option. Treat it, plus `planar3_0` and
`iso3_0`, as legacy/experimental V-format research only.

Expected route evidence:

```text
selected=rocm_packed16_dot4_mmq K=i32 V=<value-type>
```

If the log says K is q8_0 for FlashAttention, you are not validating this path.

### Fast WikiText quality smoke

Short WikiText-2 raw smoke on the 27B MTP model, `ctx=512`, `chunks=4`
(~1020 evaluated tokens/candidate), all with `K=i32` and
`selected=rocm_packed16_dot4_mmq`. This is a fast sanity check, not a full
quality benchmark.

![WikiText-2 V-cache quality smoke](docs/assets/wikitext-v-cache-quality-20260531.png)

| V cache | PPL / ratio vs f16 V | Mean KLD vs f16 V | Median KLD | Same top token |
|---|---:|---:|---:|---:|
| f16 | `5.6891 ± 0.4459` | baseline | baseline | baseline |
| q4_0 | `1.00198 ± 0.00422` ratio | `0.004550 ± 0.000338` | `0.001818` | `97.06%` |
| q8_0 | `1.00101 ± 0.00336` ratio | `0.002825 ± 0.000334` | `0.000863` | `97.94%` |

Takeaway: q4_0 is the default compression choice; q8_0 is the higher-precision
choice and is measurably closer to f16 V on this smoke. Evidence:
[`.harness/research/i32-vformat-wikitext-kld-ppl-20260531.md`](.harness/research/i32-vformat-wikitext-kld-ppl-20260531.md).

### 128k active MTP VRAM smoke

A 128k-context server-ready VRAM smoke on the 27B MTP GGUF with active
`draft-mtp`, `q4_0` V, and `--spec-draft-type-v q4_0`. The ROCm packed16 run
omits main and draft K CLI overrides; the Vulkan comparison uses normal f16 main
K plus q4 V.

![128k active MTP VRAM smoke](docs/assets/active-mtp-vram-128k-20260531-v3.png)

| Run | Total VRAM used | Delta over idle |
|---|---:|---:|
| ROCm packed16/I32 route + q4 V active MTP | `21.760 GiB` | `21.079 GiB` |
| Vulkan f16 K + q4 V active MTP | `23.180 GiB` | `22.500 GiB` |

Measured saving: Vulkan uses `+1.420 GiB` more total VRAM (`+1.422 GiB` delta
over idle). ROCm route evidence included the packed16 DOT4/MMQ route and `PDMQ QK
probe PASSED`. Evidence:
[`.harness/research/active-mtp-vram-128k-20260531.md`](.harness/research/active-mtp-vram-128k-20260531.md).

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

This builds both route families documented below: `GGML_HIP=ON` builds the ROCm
backend and DOT4-MMQ path, while `GGML_HIP_ROCWMMA_FATTN=ON` adds PWMMA support.

Requirements:

- ROCm HIP toolchain
- rocWMMA headers/libraries available to CMake for the PWMMA path
- gfx1100-class RDNA3 GPU; RX 7900 XTX is the primary target

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

## Technical details

Most users can stop here. For internals and debugging, see
[Packed16 RDNA3 technical notes](docs/PACKED16_RDNA3_DETAILS.md).

---

## Experiment notes

This branch came out of a longer AMD/RDNA attention experiment, not a one-shot
kernel drop.

- The first prototype work focused on DOT4 FlashAttention for llama.cpp HIP on
  RDNA3: pack K into DOT4-friendly rows, use `sudot4` for QK, and make
  quantized KV cache attention fast enough that attention stops dominating
  decode.
- That prototype proved the important lesson: the K layout matters as much as
  the math instruction. Packed16 made K reads aligned and DOT4-ready without a
  separate dequant buffer.
- The current branch carries that lesson into a cleaner packed16 K-cache
  FlashAttention path for Qwen3.6-style workloads, with DOT4-MMQ as the default
  prefill path and PWMMA as the WMMA-family comparison/fallback route.
- If you want the longer prototype history, including the earlier DOT4 decode,
  split-K, and packed16 cache experiments, read
  [DrBearJew/dot4-flash-attention](https://github.com/DrBearJew/dot4-flash-attention).

---

## Credits and further reading

This branch builds on a lot of prior work. Special thanks to:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base runtime,
  ggml backends, FlashAttention infrastructure, and the upstream project this
  branch extends.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)
  — MTP/TurboQuant fork foundation, tensor sharing, and CUDA TBQ4 FA work that
  provided the branch foundation.
- [DrBearJew/dot4-flash-attention](https://github.com/DrBearJew/dot4-flash-attention)
  — earlier DOT4 FlashAttention prototype notes, packed16 K-cache experiments,
  split-K decode work, and the experiment trail that led here.
- [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) — RDNA3
  rocWMMA GEMM reference, autotuner, config lookup, and LDS buffering ideas.
- [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
  — AMD matrix-instruction details for WMMA shapes, lane layout, and throughput
  sanity checks.
- [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire) — AMD/RDNA
  dispatch-screening and WMMA references.
- [Stormrage34/llama.cpp-turboquant-hip](https://github.com/Stormrage34/llama.cpp-turboquant-hip)
  — AMD VEC TurboQuant-style path and practical ROCm fork lessons.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
  — original TurboQuant block-format reference.

For deeper internals, see
[Packed16 RDNA3 technical notes](docs/PACKED16_RDNA3_DETAILS.md).

---

## License

Follows upstream `llama.cpp` licensing terms.
