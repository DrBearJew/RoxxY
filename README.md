# RoxxY — RDNA3 Packed-K FlashAttention

![RoxxY](assets/github-social-preview.jpg)

AMD-first `llama.cpp` fork for RDNA3 / gfx1100 GPUs (RX 7900 XTX and similar).

Standard quantized-KV attention spends real prefill time unpacking 8-bit K
values back to f16 inside the kernel, competing with the actual matmul for
bandwidth. RoxxY stores K in a packed layout that RDNA3's matrix units can
read directly — no unpack step — and auto-selects the fastest matching kernel
at runtime.

Goals:

- Make RDNA3 consumer GPUs viable for long-context local inference.
- Keep the default path simple: build, run, the runtime picks the fast route.
- Don't oversell experimental options — the compact K-cache variant is
  smaller, not (yet) faster, and the README says so.

Use your normal GGUF models — this isn't a new model format, just a faster
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

Don't pass `--spec-default` — ngram speculative decoding doesn't work
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
| packed8 | `--cache-type-k q4_0` | 144 B | half the size, not faster yet — use it for VRAM, not speed |

## Performance

RX 7900 XTX, `llama-bench -fa 1 -ngl 99`:

| Model | Prefill (pp512) | Decode (tg128) |
|---|---:|---:|
| Qwen3.6 27B Q4_K_M | ~929 tok/s | ~28.7 tok/s |
| Qwen3.6 35B-A3B MoE | ~2707 tok/s | ~92.8 tok/s |

Long-context server smoke, 27B MTP, 32k-token prompt: **~589 tok/s prefill**,
**~33 tok/s decode**.

VRAM at 128k context with active MTP (27B): **21.8 GiB** on the packed16
route vs **23.2 GiB** on Vulkan with f16 K — about **1.4 GiB less**.

Quality cost of `q4_0` V-cache vs full f16 V: perplexity ratio **1.002**
(effectively unchanged), **97%** same top-token match on a WikiText-2 smoke.

Full benchmark tables, per-route breakdowns, and methodology are in the
[technical notes](docs/PACKED16_RDNA3_DETAILS.md).

## Roadmap

1. Paged attention for long-context throughput — the top priority, so speed
   doesn't fall off as context grows.
2. Faster packed8 (the compact K-cache option) — currently smaller, not faster.
3. Gemma support.

## Supported hardware

Tested on RX 7900 XTX (gfx1100). Other RDNA3 cards are expected to work but
aren't separately validated. RDNA2, RDNA4, CDNA, and CUDA GPUs are out of
scope for this branch.

## Tested models

Qwen3.6 27B, 35B-A3B, and 9B GGUFs, including MTP variants. Other model
families are untested — GQA/MoE assumptions may not match. Model sources used
during development include GGUF releases from
[llmfan46](https://huggingface.co/llmfan46),
[HauhauCS](https://huggingface.co/HauhauCS),
[havenoammo](https://huggingface.co/havenoammo),
[Radamanthys11](https://huggingface.co/Radamanthys11),
[unsloth](https://huggingface.co/unsloth), and
[mradermacher](https://huggingface.co/mradermacher).

## Learn more

- [Technical notes](docs/PACKED16_RDNA3_DETAILS.md) — how packed16 works,
  kernel routes, full benchmark tables, debugging flags, and credits.

This branch builds on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
and earlier RDNA3 attention prototyping; see the technical notes for the full
list of prior work this owes credit to.

## License

Follows upstream `llama.cpp` licensing terms.
