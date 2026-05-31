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

## License

Follows upstream `llama.cpp` licensing terms.
