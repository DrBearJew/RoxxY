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
GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1               # lab-only; never set in normal serving env
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1          # second lab-only acknowledgement gate
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_QSCALE16=1       # optional quality probe: per-WMMA-K Q scales
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6=1      # optional Qwen3.6-27B GQA=6 lab reopen gate
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27      # required for bounded experiments; do not route all layers
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MAX=38      # current weird-prefix isolation candidate: skip final full-attn layer
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYER=7      # optional diagnostic layer exclusion
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_SKIP_LAYERS=7,11  # optional comma/range list, e.g. 7,11-13
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_REQUIRE_SELECTED=1 # fail if an included layer cannot select this route
```

Keep this lab-only and keep it out of normal launch environments. On 27B, the
normal fast baseline is f16/f16 or the promoted q8/tbq4 path; q8_0/q4_0 WMMA-I8
is a correctness/selector experiment and can be drastically slower than that
baseline. Only use it in bounded A/B runs with explicit `LAYER_MIN/MAX`,
`REQUIRE_SELECTED=1`, and artifact-backed checks. Backend-op tests pass, but
greedy generation parity is not proven for unrestricted routing. `QSCALE16` is an opt-in stabilization probe that
quantizes Q per 16-wide WMMA K tile instead of per q8_0 block. `LAYER_MIN/MAX`
and `SKIP_LAYER(S)` are diagnostic safety knobs for layer-filtered logit/top1
checks only; do not use them as a default policy without an artifact-backed
prompt, long-shape, and generation/coherence sweep. `REQUIRE_SELECTED=1` is
layer-scoped and q8/q4-only: it only applies after the include/skip policy allows
a `q8_0` K / `q4_0` V D=256 prefill layer, so `LAYER_MIN=27` can fail fast for
intended routed layers without requiring earlier layers or unrelated q8/tbq4
lanes. The base support gate covers the already validated GQA=4/8 shapes;
Qwen3.6-27B's GQA=6 shape additionally requires
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6=1`, and that remains a separate lab
reopen gate. For current Qwen3.6-35B lab runs, `LAYER_MIN=27` is the
conservative starting point, and `LAYER_MAX=38` is the current weird-prefix
isolation candidate. The 2026-05-24 generation sweep used the Qwen3.6 merged chat template
and `<|think_off|>`; the recurring Arabic `فاق` prefix was not a fixed text bug
but one bad sampled first-token mode. It appeared when the routed full-attention
set included layer 23, while a related `无影` prefix appeared intermittently when
layer 39 was included. Route-off was 12/12 stable; single routed layers were
stable; meta policy hunts repeated `min27_max38` 12/12 stable, while `min27`,
`min23`, and `min23_skip24` failed with weird prefixes/hash splits. Follow-up
one-layer-removal hunts showed every passing candidate removed layer 39. The
candidate-only target broad logits matrix passed for `off` vs `min27_max38`, and
the broader generation/coherence smoke passed 5/5 for `min27_max38` with no
unexpected non-ASCII or marker leakage. Do not promote `min23` or
`min23_skip24` despite prompt-throughput wins. Treat `LAYER_MIN=27 LAYER_MAX=38`
as the current opt-in lab candidate, not a default. To keep 35B validation
bounded, the broad-matrix and generation/coherence scripts now default to the
candidate pair only (`off min27_max38`); set `MODE_PROFILE=full` or explicit
`MODE_LIST=...` for diagnostic sweeps. Use `CASE_LIST=...` or `CASE_LIMIT=1`
for 2-4 run 35B smoke checks. Use `scripts/hip/run-q8q4-wmma-i8-policy-hunt.sh`
for replicated weird-prefix/hash-split delta debugging instead of one-off manual
needle hunts; it defaults to a 4-run candidate smoke and requires
`POLICY_PROFILE=full` for the old multi-policy repeat hunt. Re-run
`scripts/hip/run-q8q4-wmma-i8-long384-repro.sh` and
`scripts/hip/run-q8q4-wmma-i8-generation-coherence.sh` after route changes. See
`docs/rocm-tbq4-paths/08-q8q4-wmma-i8-min27-validation.md` for the current
validation summary, including rel RMS, KLD/JS/TVD, perf, generation/coherence,
and thinking-leak caveats.

## Run recipes

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
GGML_CUDA_MMQ_MAX_X=48 \
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
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
./build-rocm-vulkan/bin/llama-server \
  --device ROCm0 \
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
