# llama.cpp — TurboQuant + MTP (AMD ROCm)

Merged branch of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) combining:

- **MTP** (Multi-Token Prediction / speculative decoding) — 2× generation speed on dense models
- **TurboQuant KV cache compression** — 50% KV memory reduction, enabling 2× context sizes

**Built for and tested on AMD RX 7900 XTX (gfx1100) with ROCm.**

## What This Adds

| Feature | Source | What It Does |
|---------|--------|--------------|
| `--spec-type draft-mtp` | llama.cpp MTP PRs | Draft-n speculative decoding, 2× tok/s on dense models |
| `--cache-type-k turbo4` | domvox TurboQuant fork | 4-bit KV cache compression (K) |
| `--cache-type-v turbo4` | domvox TurboQuant fork | 4-bit KV cache compression (V) — halves KV memory vs q8_0 |

## Benchmarks (RX 7900 XTX, 24 GB VRAM, ROCm 7.2.3)

### Qwen 3.6 27B Dense (Q4_K_M)

| Config | Context | Tok/s (gen) | VRAM |
|--------|---------|-------------|------|
| MTP only, q8_0 KV | 32k | 46–50 | 19.6 GB |
| MTP + turbo4 KV | 32k | 41–43 | 18.4 GB |
| MTP + turbo4 KV | **64k** | 37–48 | 18.9→24.3 GB |

- **turbo4 at 64k uses ~same VRAM as q8_0 at 32k** — 2× context capacity
- Prefill speed: ~550 tok/s (21.5k token document in ~39s)
- MTP draft acceptance rate: ~80% (draft-n-max 3)

### Qwen 3.6 35B MoE (Q4_K_M, 3B active)

| Config | Tok/s | VRAM |
|--------|-------|------|
| Standard (no turbo, no MTP) | 70–82 | 21.1 GB |

*MTP provides no gain on MoE models — only benefits dense architectures.*

## Build (ROCm)

```bash
mkdir build-rocm-tq && cd build-rocm-tq
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1100
cmake --build . -j$(nproc)
```

## Usage

```bash
llama-server \
  --model Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --ctx-size 65536 \
  --cache-type-k turbo4 \
  --cache-type-v turbo4 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --parallel 1 \
  --cache-ram 256 \
  --flash-attn on
```

| Flag | Purpose |
|------|---------|
| `--cache-type-k turbo4` | TurboQuant 4-bit KV cache (K) |
| `--cache-type-v turbo4` | TurboQuant 4-bit KV cache (V) |
| `--spec-type draft-mtp` | Enable MTP speculative decoding |
| `--spec-draft-n-max 3` | 3 draft tokens per step |
| `--parallel 1` | Required for MTP |

*Turbo KV types: `turbo2`, `turbo3`, `turbo4` (lower = more compressed).*

## Key Trade-offs

- **turbo4**: +100% context capacity, −10–15% generation speed vs q8_0 KV
- **MTP**: +100% speed on dense models, zero gain on MoE
- **32k q8_0 baseline** (no turbo) is fastest but caps at 32k on 24 GB VRAM

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base project
- MTP: upstream PRs by jartu0, the-new-sky, et al
- [domvox/llama.cpp](https://github.com/domvox/llama.cpp) — TurboQuant
- Merge, ROCm HIP port, benchmarks: **DrBearJew**

## License

MIT (same as upstream llama.cpp)
