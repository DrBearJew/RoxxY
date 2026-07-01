# RoxxY: RDNA3 Packed-K FlashAttention

![RoxxY](assets/github-social-preview.jpg)

AMD-first `llama.cpp` fork for RDNA3 / gfx1100 GPUs (RX 7900 XTX and similar).

Standard quantized-KV attention spends real prefill time unpacking 8-bit K
values back to f16 inside the kernel, competing with the actual matmul for
bandwidth. RoxxY stores K in a packed layout that RDNA3's matrix units can
read directly with no unpack step, and auto-selects the fastest matching
kernel at runtime.

Goals:

- Make RDNA3 consumer GPUs viable for long-context local inference.
- Keep the default path simple: build, run, the runtime picks the fast route.
- Don't oversell experimental options: the compact K-cache variant is
  smaller, not (yet) faster, and the README says so.

Use your normal GGUF models: this isn't a new model format, just a faster
runtime.

## Quick start

```bash
git clone https://github.com/DrBearJew/RoxxY
cd RoxxY
# default branch is tbq4-rdna3-experiment

cmake -S . -B build-rocm \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-server llama-bench -j$(nproc)
```

Run a model:

```bash
./build-rocm/bin/llama-server \
  --device ROCm0 \
  --model /path/to/model.gguf \
  --flash-attn on \
  --cache-type-v q4_0 \
  --ctx-size 40960 --parallel 1
```

For Qwen3.6 MTP models, add:

```bash
--spec-type draft-mtp \
--spec-draft-n-max 4 --spec-draft-p-min 0 \
--spec-draft-prio 2 --spec-draft-prio-batch 2
```

Don't pass `--spec-default`: ngram speculative decoding doesn't work
correctly on this build.

## Recommended settings

| Setting | Value | Why |
|---|---|---|
| K cache | leave `--cache-type-k` unset | fastest validated path |
| V cache | `--cache-type-v q4_0` | best speed/VRAM tradeoff |
| MTP | `--spec-type draft-mtp` | working MTP path on this build |

K cache comes in two sizes per row (D=256):

| K cache | Setting | Row size | Notes |
|---|---|---:|---|
| packed16 (default) | leave `--cache-type-k` unset | 272 B | same size class as f16, fastest validated route |
| packed8 | `--cache-type-k q4_0` or `q8_0` | 144 B | half the size, not faster yet; use it for VRAM, not speed |

## Performance

RX 7900 XTX, Qwen3.6 27B Q4_K_M MTP, `--spec-type draft-mtp`:

| Context | Prefill | Decode |
|---|---:|---:|
| 32k prompt | **~589 tok/s** | **~33 tok/s** |

VRAM at 128k context with active MTP (27B): **21.8 GiB** on the packed16
route vs **23.2 GiB** on Vulkan with f16 K, about **1.4 GiB less**.

![128k active MTP VRAM smoke](docs/assets/active-mtp-vram-128k-20260531-v3.png)

V-cache quality on a WikiText-2 smoke, relative to full f16 V:

| V cache | PPL ratio vs f16 | Same top token | Notes |
|---|---:|---:|---|
| f16 | 1.000 (reference) | 100% | most VRAM, slowest |
| **q4_0 (default)** | 1.002 | 97.1% | **fastest, smallest VRAM** |
| q8_0 | **1.001** | **97.9%** | closest to f16, more VRAM than q4_0 |

`q4_0` is the recommended default: it gives up a fraction of a percent of
quality for the best speed and VRAM. Use `q8_0` if you have VRAM to spare and
want to close that gap.

![WikiText-2 V-cache quality smoke](docs/assets/wikitext-v-cache-quality-20260531.png)

Full benchmark tables, per-route breakdowns, and methodology are in the
[technical notes](docs/PACKED16_RDNA3_DETAILS.md).

## Roadmap

1. Paged attention for long-context throughput. Top priority: speed shouldn't
   fall off as context grows.
2. Faster packed8 (the compact K-cache option): currently smaller, not faster.
3. Gemma support.

## Supported hardware

Tested on RX 7900 XTX (gfx1100). Other RDNA3 cards are expected to work but
aren't separately validated. RDNA2, RDNA4, CDNA, and CUDA GPUs are out of
scope for this branch.

## Tested models

Qwen3.6 27B, 35B-A3B, and 9B GGUFs, including MTP variants. Other model
families are untested: GQA/MoE assumptions may not match. Model sources used
during development include GGUF releases from
[llmfan46](https://huggingface.co/llmfan46),
[HauhauCS](https://huggingface.co/HauhauCS),
[havenoammo](https://huggingface.co/havenoammo),
[Radamanthys11](https://huggingface.co/Radamanthys11),
[unsloth](https://huggingface.co/unsloth), and
[mradermacher](https://huggingface.co/mradermacher).

## Learn more

- [Technical notes](docs/PACKED16_RDNA3_DETAILS.md): how packed16 works,
  kernel routes, full benchmark tables, and debugging flags.

## Credits

This branch builds on:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): base runtime,
  ggml backends, FlashAttention, and upstream infrastructure.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp):
  MTP/TurboQuant fork foundation, tensor sharing, CUDA TBQ4 FA.
- [DrBearJew/dot4-flash-attention](https://github.com/DrBearJew/dot4-flash-attention):
  earlier DOT4 FlashAttention prototype, packed16 K-cache experiment notes,
  and the path that led to this branch.
- [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm): RDNA3
  rocWMMA GEMM reference, autotuner, config lookup, LDS buffering.
- [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator):
  official AMD matrix-instruction calculator for WMMA shapes and throughput.
- [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire):
  dispatch-screening and WMMA references.
- [Stormrage34/llama.cpp-turboquant-hip](https://github.com/Stormrage34/llama.cpp-turboquant-hip):
  AMD VEC TurboQuant-style path.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant):
  original TurboQuant block-format reference.

## License

Follows upstream `llama.cpp` licensing terms.
