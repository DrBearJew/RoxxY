# 128k active MTP VRAM smoke — q4 V

Date: 2026-05-31

## Setup

- Model: `/mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf`
- Context: `--ctx-size 131072`
- Batch: `--batch-size 1024 --ubatch-size 1024`
- Slots: `--parallel 1`
- Warmup: `--no-warmup`
- Measurement: `rocm-smi --showmeminfo vram`, steady state after server ready
- Artifact: `.harness/tmp/vram-128k-active-mtp-q4v-20260531-112745`

## ROCm packed16 active MTP

Main K CLI override omitted. Spec draft K CLI override omitted.

Relevant flags:

```bash
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 \
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
./build-rocm-fixed/bin/llama-server \
  --device ROCm0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-default \
  --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --spec-draft-type-v q4_0
```

Evidence:

```text
common_speculative_init: adding speculative implementation 'draft-mtp'
srv    load_model: speculative decoding context initialized
FATTN COMPUTE SELECT selected=586 name=rocm_packed16_dot4_mmq
PDMQ v20260531 route=rocm_packed16_dot4_mmq ... V=q4_0
PDMQ QK probe PASSED: max_err=0.000488
```

Measured VRAM:

```text
used_last = 21.760 GiB
used_max  = 21.762 GiB
delta_last_over_idle = 21.079 GiB
delta_max_over_idle  = 21.081 GiB
```

## Vulkan comparison

Vulkan binary: `/home/mrtrent/llama.cpp/build/bin/llama-server`

Vulkan comparison uses normal f16 main K and q4 V:

```bash
--cache-type-k f16 \
--cache-type-v q4_0 \
--spec-type draft-mtp --spec-default \
--spec-draft-n-max 3 --spec-draft-p-min 0 \
--spec-draft-type-v q4_0
```

Measured VRAM:

```text
used_last = 23.180 GiB
used_max  = 23.182 GiB
delta_last_over_idle = 22.500 GiB
delta_max_over_idle  = 22.502 GiB
```

## Result

| Run | Total VRAM used | Delta over idle |
|---|---:|---:|
| ROCm packed16 active MTP + q4 V | 21.760 GiB | 21.079 GiB |
| Vulkan f16 K + q4 V active MTP | 23.180 GiB | 22.500 GiB |

Measured saving:

```text
Vulkan - ROCm = 1.420 GiB total used
Vulkan - ROCm = 1.422 GiB delta over idle
```
