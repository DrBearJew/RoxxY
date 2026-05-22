# llama.cpp ROCm/Vulkan TurboQuant KV — RX 7900 XTX

This branch is a local RDNA3-focused llama.cpp fork for Qwen3.6 27B MTP and
Qwen3.6 35B-A3B on a 24 GB RX 7900 XTX.

The default goal is simple:

- keep compressed KV usable at long context;
- keep ROCm and Vulkan selectable at runtime;
- keep unstable kernels env-gated;
- prefer small, reversible mitigations over broad kernel rewrites.

## Current recommendation

| Target | Status | Use |
|---|---|---|
| 27B ROCm MTP | promoted local path | `q8_0` K + `tbq4_0` V, MTP n3, f16-temp prefill on |
| 35B ROCm no-MTP | stable prompt path | `q8_0` K + `tbq4_0` V, MMQ selector on, no speculative MTP |
| 35B ROCm MTP | experimental but usable | MTP n2 plus explicit f16-temp prefill |
| Vulkan | baseline / comparison | q8/q4 is restored; TBQ4 Vulkan parity is not claimed |
| q8/q4 INT8 WMMA | lab only | explicit unsafe gate only; not production/default |

For 35B on 24 GB, use an IQ4_XS / Q4_K_S-class model for long context. Q4_K_M is
too tight for the current 35B 40k-context + f16-temp experiments.

## Important env gates

### MTP prefill stability

```bash
LLAMA_MTP_PREFILL_CHUNK=1024
LLAMA_MTP_PREFILL_FORCE_MMQ=1
```

`LLAMA_MTP_PREFILL_CHUNK` should match `--ubatch-size`.

MTP has separate draft-context KV flags. Set them explicitly; otherwise the draft
context defaults to f16 KV:

```bash
--cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0
```

### ROCm quantized-KV f16 prefill

```bash
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
```

This explicitly routes quantized-KV prefill through bounded f16-temp TILE/MMA.
It is required for the promoted 27B MTP path and for 35B MTP experiments.

Default behavior when the f16 env is absent:

```text
q8K/tbq4V VEC stays the fallback/default.
```

A/B-only auto probe:

```bash
GGML_CUDA_ROCM_QUANT_PREFILL_F16_AUTO=1
```

`_AUTO` is intentionally **not** default-on. This prevents 35B prefill from
silently falling into the f16-temp route.

Useful f16-temp controls:

```bash
GGML_CUDA_ROCM_QUANT_PREFILL_F16_MAX_MIB=1024
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_ALLOC=1
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=<ctx>
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_BUCKET_NKV=4096
```

`STABLE_ALLOC` avoids HIP pool growth by reusing one rounded temp allocation.
`STABLE_NKV` is the manual override and wins over bucketed sizing. `STABLE_BUCKET_NKV`
rounds f16 temps to nkv buckets when full-context stable scratch is too large.
`COMPRESSED_KV_FATTN_LOG=1` emits a one-shot f16-prefill decision line.

### TBQ/RDNA3 policy helpers

```bash
TBQ_AUTO_ASYMMETRIC=0        # opt out of high-GQA tbq4/tbq4 -> q8/tbq4 K promotion
GGML_CUDA_MMQ_MAX_X=48       # preferred manual RDNA3/gfx1100 cap
GGML_CUDA_MMQ_MAX_X_AUTO=1   # opt-in helper; manual MAX_X still wins
```

If symmetric `tbq4_0` K+V is requested on high-GQA models, K is promoted to
`q8_0` by default while V remains `tbq4_0`. This mirrors the local quality
policy without importing TheTom's Turbo/TQ enum architecture.

### q8/q4 WMMA-I8 lab route

```bash
GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1
```

Keep this lab-only. Backend-op tests pass, but greedy generation parity is not
proven.

## Run recipes

### 27B ROCm MTP

```bash
LLAMA_MTP_PREFILL_CHUNK=1024 \
LLAMA_MTP_PREFILL_FORCE_MMQ=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 \
./build-rocm/bin/llama-server \
  --model /path/to/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v tbq4_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --spec-type draft-mtp --spec-default \
  --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --spec-draft-prio 2 --spec-draft-prio-batch 2 \
  --parallel 1
```

### 35B ROCm, no MTP

```bash
RDNA2_MATMUL_OPT_V1=1 \
GGML_CUDA_MMQ_MAX_X=48 \
./build-rocm/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v tbq4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --parallel 1
```

### 35B ROCm MTP experiment

```bash
RDNA2_MATMUL_OPT_V1=1 \
GGML_CUDA_MMQ_MAX_X=48 \
LLAMA_MTP_PREFILL_CHUNK=1024 \
LLAMA_MTP_PREFILL_FORCE_MMQ=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960 \
./build-rocm/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --ctx-size 40960 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v tbq4_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --spec-type draft-mtp --spec-default \
  --spec-draft-n-max 2 --spec-draft-p-min 0 \
  --spec-draft-prio 2 --spec-draft-prio-batch 2 \
  --parallel 1
```

## Build

ROCm:

```bash
cmake -S . -B build-rocm -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-server llama-bench test-backend-ops -j
```

35B MMQ selector build:

```bash
cmake -S . -B build-rocm-rdna2-fa \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_FLAGS="-DRDNA2_MATMUL_OPT_V1=1" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-rdna2-fa --target llama-server llama-bench test-backend-ops -j
```

Vulkan comparison build:

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target llama-server llama-bench -j
```

## Validation evidence

Recent local evidence on RX 7900 XTX:

| Artifact | Result |
|---|---|
| `benches/rocm-rdna3/35b-iq4xs-coherence-temp06-32k-8k-f16fix-20260521-061142` | IQ4_XS, temp 0.6 long-prose canary passed; `prompt_n=36103` at `2032 tok/s`; `predicted_n=1505` at `58.5 tok/s`; peak delta from loaded `0.339 GiB`; no repeated 120-char chunk |
| `benches/rocm-rdna3/35b-server-depth0-prompt32k-decode8k-f16fix-20260521-054546` | Q4_K_M stress run was too VRAM-tight for default use, but f16-stable prefill avoided the sink; use smaller 35B quant for 24 GB |
| `benches/rocm-rdna3/q8q4-wmma-i8-finalization-20260521-050103` | q8/q4 WMMA-I8 backend sweep passes eligible rows; still lab-only due generation drift |

Pass gate for 35B f16-temp prefill:

- no OOM;
- no request-time sleep/stall;
- route logs show TILE/MMA prefill when explicitly enabled;
- VEC remains fallback when f16 envs are absent;
- peak VRAM delta remains bounded;
- long-prose coherence does not collapse into obvious repetition.

## Files to know

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | FlashAttention route selection |
| `ggml/src/ggml-cuda/fattn-common.cuh` | f16-temp env gates and stable allocation sizing |
| `ggml/src/ggml-cuda/fattn-wmma-q8q4-i8.cuh` | lab q8/q4 WMMA-I8 route |
| `tests/test-backend-ops.cpp` | FA backend-op coverage, mask/sink variants |
| `scripts/hip/run-mtp-f16-mmq-vram-sweep.py` | server VRAM/speed sweep harness |
| `docs/rocm-tbq4-paths/harness.py` | ROCm/TBQ4 smoke harness |

## Deprecated / not default

Do not enable these in user-facing wrappers:

```bash
TBQ4_WMMA_FATTN
COMPRESSED_KV_WMMA_FATTN
GGML_CUDA_ROCM_Q8Q4_WMMA_I8 without _UNSAFE
```

rocWMMA compressed-KV and INT8 WMMA work remains research/lab material until
runtime generation parity is proven.

## Credits

This branch is integration work on top of several upstream projects and public
references:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — base runtime,
  ggml backends, FlashAttention, MTP/TBQ upstream work.
- [Indras-Mirror/llama.cpp-mtp](https://github.com/Indras-Mirror/llama.cpp-mtp)
  — MTP/TurboQuant fork foundation, RotorQuant, tensor sharing, CUDA TBQ4 FA.
- [Stormrage34/llama.cpp-turboquant-hip](https://github.com/Stormrage34/llama.cpp-turboquant-hip)
  — first working AMD VEC TurboQuant-style path; this branch follows the same
  inline-dequant-inside-FA pattern for RDNA3 `q8_0/tbq4_0`.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
  — original TurboQuant block-format/FWHT/centroid reference.
- [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) and
  [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire) — ROCm/WMMA
  and dispatch-screening references used during lab-route work.
- [llmfan46](https://huggingface.co/llmfan46), [HauhauCS](https://huggingface.co/HauhauCS),
  [havenoammo](https://huggingface.co/havenoammo), and
  [Radamanthys11](https://huggingface.co/Radamanthys11) — Qwen3.6/MTP GGUFs,
  model releases, and extraction/grafting references used in validation.
- [allanchan339/vLLM Qwen chat-template fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix)
  and [froggeric/Qwen Fixed Chat Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — Qwen chat-template fixes used by local serving wrappers.

## License

This repository follows the upstream llama.cpp licensing terms.
