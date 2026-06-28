# RoxxY — RDNA3 Packed-K FlashAttention

![RoxxY](assets/github-social-preview.jpg)

AMD-first `llama.cpp` fork for RDNA3 / gfx1100.

Focus: fast local Qwen/MTP inference with packed K-cache FlashAttention,
PV4/V144 value-cache routing, and strict route canaries.

This is a runtime/cache-layout fork, not a new GGUF format. Use your normal GGUF
models. RoxxY chooses the GPU K-cache layout at runtime.

## 🎯 Project goals

- Make RDNA3 consumer GPUs useful for long-context local inference.
- Keep the default path simple: build, run, let the selector choose the fast route.
- Use packed I32 K sidecars for FlashAttention instead of raw K cache tensors.
- Keep experimental q4 K storage available without making it look faster than it is.

## ✅ Recommended path

For normal MTP runs, use:

| Component | Recommendation | Why |
|---|---|---|
| K cache | default packed16/I32 | fastest validated prefill path |
| V cache | `q4_0` | best measured speed/VRAM tradeoff |
| MTP | `--spec-type draft-mtp` | active-RS pure MTP path |
| PV4/V144 | automatic default | current fast V route, no env needed |

K has three user-facing modes:

| How to select K | Runtime layout | Use |
|---|---|---|
| omit `--cache-type-k` | packed16/I32 q8-style K | fastest validated default |
| `--cache-type-k q8_0` | packed8/I32 q4 K | compact q4 K request, compatibility spelling |
| `--cache-type-k q4_0` | packed8/I32 q4 K | compact q4 K request |

Pick V precision separately with `--cache-type-v`.

## 🚀 Quick Start — Qwen3.6 27B MTP

```bash
MODEL=/path/to/Qwen3.6-27B-Q4_K_M-mtp.gguf

./build-rocm/bin/llama-server \
  --device ROCm0 \
  --model "$MODEL" \
  --flash-attn on \
  --cache-type-v q4_0 \
  --cache-type-v-draft q4_0 \
  --ctx-size 40960 --batch-size 2048 --ubatch-size 512 \
  --parallel 1 --no-warmup \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --spec-draft-prio 2 --spec-draft-prio-batch 2
```

That command is the baseline for dense models on this build. Do not pass
`--spec-default`: it enables the ngram speculative path, and ngram does not work
correctly with this ROCm/MTP build. Keep `--spec-draft-n-max 4` for dense models
so QBlock verification stays on the validated fast path.

## 🧱 K-cache formats

| Name | Setting | K row size at D=256 | Use case | Notes |
|---|---|---:|---|---|
| packed16 | omit `--cache-type-k` | 272B | recommended default | same K row size class as f16; fastest validated prefill route |
| packed8 | `--cache-type-k q8_0` | 144B | compact K cache | smaller K cache, compatibility spelling |
| packed8 | `--cache-type-k q4_0` | 144B | compact K cache | same compact route, explicit q4 spelling |

For most users, omit `--cache-type-k`. It uses the 272B default K cache, which
has the same row-size class as f16 K. Use one of the compact 144B K-cache options
only when you want the smaller K cache and have validated the route for your
model/context.

## 🎛️ V-cache choices

| V cache | Use case | Notes |
|---|---|---|
| `q4_0` | recommended fast MTP default | fastest measured path |
| `q8_0` | higher V precision | more VRAM, slower than q4_0 in measured profile |
| `f16` | highest V precision | most VRAM, slower than q4_0 in measured profile |

`q4_0` V is only for the `P @ V` side. QK uses the selected packed I32 K path.

## 📊 Route smoke snapshot

Qwen3.6 27B Q4_K_M MTP, `llama-server`, pp32k-style prompt smoke
(`tokens_evaluated=32768`), `ctx=49152`, `tg128`. These are route/hash sanity
checks, not a full benchmark suite.

| K selection | V selection | Prompt tok/s | Decode tok/s | SHA | Notes |
|---|---|---:|---:|---|---|
| omit `--cache-type-k` → packed16 / 272B K | `q4_0` | ~583–589 | ~33 | `33fc0c55` | headline prefill path |
| `--cache-type-k q8_0` or `--cache-type-k q4_0` → packed8 / 144B K | `q4_0` | ~560–566 | ~31 | `33fc0c55` | smaller K, not faster yet |
| q4 K | `q8_0` | ~550 | ~39 | `33fc0c55` | faster decode, slower prefill |
| q4 K | `f16` | ~551 | ~38 | `33fc0c55` | faster decode, slower prefill |

Older 8k clean auto-table smoke: prompt ~744 tok/s, decode ~51 tok/s, SHA `4219d799`.

## 🛣️ Roadmap

1. **Paged Attention for long context throughput.** This is the top priority: make
   the packed-K path work efficiently with paged attention so long-context runs
   keep throughput instead of falling off as context grows.
2. **Improve packed8.** The compact q4 K path is operational, but it still expands
   into the existing i8 WMMA route. The next step is making packed8 faster, not
   just smaller.
3. **Gemma support.** Add and validate the model-specific plumbing needed for
   Gemma-family runs.

## Validation and benchmark notes

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
choice and is measurably closer to f16 V on this smoke.

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
probe PASSED`.

---

## Build from source

You need a compatible GGUF model. The tested model families and example
Hugging Face sources are listed below under [Tested models](#tested-models).

```bash
git clone https://github.com/DrBearJew/RoxxY
cd RoxxY
# The default branch is tbq4-rdna3-experiment.

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

On the benchmark system used for the results below, current packed16 auto starts row/default prefill on the PWMMA BM64 i8-QK PV-WMMA DBV route from `nk >= 512` for target Qwen shapes. Short pp512 historical measurements were around **2700 tok/s** on the prior BM32 reg-out route, which is now an opt-out/diagnostic comparison path.

---

## Headline results

RX 7900 XTX / gfx1100, `llama-bench -fa 1 -ngl 99` unless noted.
Record your ROCm and compiler versions when rerunning these numbers.

### Current long-context server smoke

Qwen3.6 27B Q4_K_M MTP, `llama-server`, `ctx=49152`, `--cache-type-v q4_0`, `--cache-type-v-draft q4_0`, packed16/I32 K, production auto route on commit `0ea6db58e`.

| Prompt / predict | Route | Prompt tok/s | Decode tok/s | SHA |
|---|---|---:|---:|---|
| 32k / tg128 | PWMMA BM64 i8-QK PV-WMMA DBV | **589.43** | 33.17 | `33fc0c55` |

Route evidence: `selected=pwmma_bm64_i8qk_pvwmma_dbv`, `desc_layout=0`, packed16/I32 K, q4 V, no BM32 prefill route selected. Draft acceptance and very short tg32 runs are intentionally omitted from this headline table; these rows are route/hash/speed smoke tests, not acceptance-quality benchmarks.

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

### What these numbers show

- PWMMA BM64 i8-QK PV-WMMA DBV is the production auto packed16 prefill route for target Qwen row/default shapes from `nk >= 512`; the first long prefill chunk now goes BM64 DBV instead of BM32 direct-V.
- Optional packed8/packed4 q4 K storage is operational and route-validated, but current prefill expands q4 K to i8 before WMMA and is not faster than the packed16 headline baseline.
- DOT4-MMQ/PDMQ remains available for route-pinned validation, small-Q/MTP roles, decode, and experimental V formats.
- PWMMA BM32 reg-out direct-V remains a historical/diagnostic short pp512 comparison route in the listed table (+3.0% over DOT4-MMQ on 35B pp512 and +3.9% on 27B pp512), while DBV PV-WMMA is the current auto prefill route.
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
| Qwen3.6 9B MTP GGUF | Tested | 9B MTP route target; V4/q8_0/f16 V-cache paths validated |
| Other Qwen GGUFs | Unknown | May work if shapes and cache assumptions match |
| Llama-family GGUFs | Untested | Not the target of this branch |
| Other MoE families | Untested | GQA/MoE assumptions may differ |

Model sources used during development include GGUF releases from
[llmfan46](https://huggingface.co/llmfan46),
[HauhauCS](https://huggingface.co/HauhauCS),
[havenoammo](https://huggingface.co/havenoammo),
[Radamanthys11](https://huggingface.co/Radamanthys11),
[unsloth](https://huggingface.co/unsloth), and
[mradermacher](https://huggingface.co/mradermacher).

---

## Build details

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
