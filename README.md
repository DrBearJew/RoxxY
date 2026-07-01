# RoxxY — RDNA3 Packed-K FlashAttention

![RoxxY](assets/github-social-preview.jpg)

AMD-first `llama.cpp` fork for RDNA3 / gfx1100 GPUs (RX 7900 XTX and similar).

It speeds up long-context local inference on RDNA3 by storing the attention
K-cache in a packed layout instead of raw f16, and auto-selects the fastest
matching kernel at runtime. Use your normal GGUF models — this isn't a new
model format, just a faster runtime.

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

A smaller (but not currently faster) K cache is available via
`--cache-type-k q4_0`, if VRAM is tighter than speed. See the
[technical notes](docs/PACKED16_RDNA3_DETAILS.md) for the full breakdown.

## Performance

RX 7900 XTX, Qwen3.6 27B Q4_K_M MTP, 32k-token prompt: **~589 tok/s prefill**,
**~33 tok/s decode**. Full benchmark tables and route names are in the
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
