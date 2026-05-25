# llama.cpp ROCm/Vulkan TurboQuant KV — RX 7900 XTX

This branch is a local RDNA3-focused llama.cpp fork for Qwen3.6 27B MTP and
Qwen3.6 35B-A3B on a 24 GB RX 7900 XTX.

The default goal is simple:

- keep compressed KV usable at long context;
- keep ROCm and Vulkan selectable at runtime;
- keep unstable kernels env-gated;
- prefer small, reversible mitigations over broad kernel rewrites.

## Gemma 4 support

This branch includes Gemma 4 MTP assistant support on top of the TBQ4 RDNA3
stack.

### Included

- GGUF arch + tensor mapping for `gemma4-assistant` (`nextn.pre_projection`,
  `nextn.post_projection`).
- Shared-KV MTP draft wiring via `llama_set_mtp_source(ctx_dft, ctx_tgt)`.
- Gemma 4 assistant draft graph path in `src/models/gemma4.cpp`.
- Server-side source wiring for Gemma 4 assistant MTP contexts.
- Placement checks/guardrails for draft/target shared-KV device compatibility.

## Current recommendation

| Target | Status | Use |
|---|---|---|
| ROCm no-TBQ KV | stable public path | `q8_0` K + `q4_0` V, no experimental env knobs |
| 27B ROCm MTP | promoted local path | `q8_0` K + `tbq4_0` V, MTP n3, f16-temp prefill on |
| 35B ROCm no-MTP | stable prompt path | `q8_0` K + `tbq4_0` V, MMQ selector on, no speculative MTP |
| 35B ROCm MTP | experimental but usable | MTP n3 plus explicit f16-temp prefill, no force-MMQ by default |
| Vulkan | baseline / comparison | q8/q4 is restored; TBQ4 Vulkan parity is not claimed |

Q4_K_M works for both 27B and 35B. Context length, MTP depth, KV format, and
batch/ubatch sizing determine the practical VRAM budget.

## Runtime knobs

### MTP prefill stability

```bash
LLAMA_MTP_PREFILL_CHUNK=1024
# LLAMA_MTP_PREFILL_FORCE_MMQ=1   # diagnostic only; not recommended by default
```

`LLAMA_MTP_PREFILL_CHUNK` should match `--ubatch-size`. `LLAMA_MTP_PREFILL_FORCE_MMQ=1`
forces supported quantized matmuls through MMQ and is useful for A/B diagnostics,
but it reduced 35B MoE temp-0.6 generation throughput in local ROCm/gfx1100 tests.

MTP has separate draft-context KV flags. Use these when draft KV should match
the target q8/tbq4 KV format:

```bash
--cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0
```

### ROCm quantized-KV f16 prefill

```bash
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1
```

Use this for the promoted ROCm q8/tbq4 prefill path. It routes quantized-KV
prefill through the bounded f16-temp TILE/MMA selector used by the 27B MTP path
and 35B MTP experiments.

Without this opt-in, q8K/tbq4V uses the quantized-KV vector fallback. For A/B
sweeps, `GGML_CUDA_ROCM_QUANT_PREFILL_F16_AUTO=1` lets supported long-prefill
q8/tbq4 shapes choose f16-temp when they fit the configured budget.

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
GGML_CUDA_MMQ_MAX_X_AUTO=1   # opt-in helper; manual MAX_X still wins
GGML_CUDA_MMQ_MAX_X=48       # manual fallback/override for RDNA3/gfx1100 A/Bs
```

If symmetric `tbq4_0` K+V is requested on high-GQA models, K is promoted to
`q8_0` by default while V remains `tbq4_0`. This mirrors the local quality
policy without importing TheTom's Turbo/TQ enum architecture.

## Run recipes

### Stable ROCm q8/q4 without TBQ

Use this when sharing a simple no-TBQ test recipe. Do not set any ROCm lab-route
environment variables for this path. Long prefill uses the bounded f16-temp path
when applicable; decode and unsupported shapes fall back to the normal VEC path.

```bash
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
  --model /path/to/model.gguf \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q4_0
```

### 27B ROCm MTP

```bash
LLAMA_MTP_PREFILL_CHUNK=1024 \
LLAMA_MTP_PREFILL_FORCE_MMQ=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 \
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
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
GGML_CUDA_MMQ_MAX_X_AUTO=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960 \
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --ctx-size 40960 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v tbq4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --parallel 1
```

### 35B ROCm MTP recommended

Local gfx1100 temp-0.6 validation favored draft depth 3 without force-MMQ:
`mtp_n3` reached 2677 prompt tok/s and 98.8 generation tok/s with 3192/4044
draft acceptance and ~21.23 GiB peak VRAM. `LLAMA_MTP_PREFILL_FORCE_MMQ=1` was
worse as a force-control (75.9 generation tok/s), so leave it unset by default.

```bash
RDNA2_MATMUL_OPT_V1=1 \
GGML_CUDA_MMQ_MAX_X_AUTO=1 \
LLAMA_MTP_PREFILL_CHUNK=1024 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16=1 \
GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960 \
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
  --model /path/to/Qwen3.6-35B-A3B-IQ4_XS-00001-of-00002.gguf \
  --ctx-size 40960 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v tbq4_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --temp 0.6 --top-p 0.95 \
  --spec-type draft-mtp --spec-default \
  --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --spec-draft-prio 2 --spec-draft-prio-batch 2 \
  --parallel 1
```

## Build

Single multi-backend build (ROCm + Vulkan):

```bash
cmake -S . -B build-rocm-vulkan \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_HIP_FLAGS="-DRDNA2_MATMUL_OPT_V1=1" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-vulkan --target llama-server llama-bench test-backend-ops -j
```

Backend selection is runtime-only:

- ROCm path: add `--device ROCm0`
- Vulkan path: add `--device Vulkan0`

## Files to know

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | FlashAttention route selection |
| `ggml/src/ggml-cuda/fattn-common.cuh` | f16-temp env gates and stable allocation sizing |
| `tests/test-backend-ops.cpp` | FA backend-op coverage, mask/sink variants |
| `scripts/hip/run-mtp-f16-mmq-vram-sweep.py` | server VRAM/speed sweep harness |
| `docs/rocm-tbq4-paths/harness.py` | ROCm/TBQ4 smoke harness |

## Deprecated / not default

Do not enable these in user-facing wrappers. Experimental compressed-KV WMMA
routes require both the route flag and the generic unsafe gate:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
TBQ4_WMMA_FATTN=1                  # or GGML_CUDA_ROCM_TBQ4_WMMA_FATTN=1
COMPRESSED_KV_WMMA_FATTN=1
```

The q8K DOT4 KQ/FA probe route is also lab-only. It is not a default
optimization path; leave these unset in user-facing wrappers unless deliberately
reproducing the DOT4 experiments:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_DOT4_KQ=1
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_q8k_dot4_kq
GGML_CUDA_ROCM_Q8K_DOT4_KQ_FULL_FA=1      # only for full-FA/fused probes
GGML_CUDA_ROCM_Q8K_DOT4_KQ_VARIANT=fused_tile8_parallel
```

Current DOT4 outcome and next-plan guardrails are in
`docs/rocm-tbq4-paths/23-q8k-dot4-fa-root-cause-and-next-plan.md`: do not
promote the standalone tile8/split-KV probes; any next prototype should reuse
the stable `launch_fattn` tiled/GQA/combine scaffolding before adding DOT4.

A separate q8_0-K/q4_0-V VEC scaffold A/B knob keeps the stable VEC kernel and
only changes Q columns per block. It is also unsafe/opt-in:

```bash
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_Q4V_VEC_COLS=4
```

The stable no-TBQ q8/q4 path does not use these flags.

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
