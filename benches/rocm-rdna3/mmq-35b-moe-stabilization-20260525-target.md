# 35B MoE ROCm stabilization target

Scope: `/home/mrtrent/llama.cpp-mtp-tbq4-rdna3-daily` only. DOT4/q8q4 FlashAttention experiments remain isolated in `/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github`.

## Target recipe

Stabilize a production-oriented RX 7900 XTX / gfx1100 recipe for Qwen3.6 35B-A3B MoE with:

- target KV: `--cache-type-k q8_0 --cache-type-v tbq4_0`
- draft KV when MTP is enabled: `--cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0`
- long prompt f16 bridge: `GGML_CUDA_ROCM_QUANT_PREFILL_F16=1`
- stable f16 temp allocation sized to context: `GGML_CUDA_ROCM_QUANT_PREFILL_F16_STABLE_NKV=40960`
- MMQ policy helper: `GGML_CUDA_MMQ_MAX_X_AUTO=1`
- RDNA2/RDNA3 MoE LDS path opt-in: `RDNA2_MATMUL_OPT_V1=1`
- MTP chunk aligned with ubatch: `LLAMA_MTP_PREFILL_CHUNK=1024`
- primary batch sizing: `--batch-size 1024 --ubatch-size 1024`
- primary context target: `--ctx-size 40960`
- realistic sampling validation: `temperature=0.6`, fixed seed, real top-p if the client exposes it

## Non-goals / guardrails

- Do not make DOT4/q8q4 FA the default path.
- Do not silently repurpose non-MoE dense MMQ auto policy; non-MoE auto should stay native unless new evidence says otherwise.
- Do not recommend `LLAMA_MTP_PREFILL_FORCE_MMQ=1` as default unless a fresh realistic-sampling A/B beats no-force on both prompt and decode.
- Preserve manual override precedence: `GGML_CUDA_MMQ_MAX_X` wins over `GGML_CUDA_MMQ_MAX_X_AUTO`.
- Keep f16-prefill opt-in; `AUTO` remains an A/B convenience.

## Acceptance criteria

A recipe is stable enough to document when:

1. Focused MMQ/MUL_MAT_ID correctness checks pass with the patched policy.
2. Route logs prove the intended MMQ auto cap and f16-prefill path were used.
3. Realistic sampling (`temp=0.6`) runs complete for:
   - no-MTP baseline
   - MTP draft `n_max=2`
   - MTP draft `n_max=3`
   - one force-MMQ control on the best MTP case
4. Metrics are captured per run: prompt tok/s, generation tok/s, wall time, draft accepted/drafted, peak VRAM, warnings.
5. Recommended README/env block matches the best measured no-force recipe and documents VRAM/diagnostic caveats.

## Current best evidence before this pass

Artifact: `benches/rocm-rdna3/mmq-35b-moe-prod-cmd-ab-vram-20260525-072116`

Best observed case was `no_force_ub1024`:

- prompt: 2744.90 tok/s
- generation: 124.17 tok/s
- draft acceptance: 3234/3668
- peak VRAM: 21.24 GiB

The force-MMQ control was worse:

- `full_ub1024_force`: 2652.03 prompt tok/s, 99.32 gen tok/s, 2933/4225 accepted

Caveat: those runs used greedy/temp-0 style generation, so acceptance and decode throughput must be rechecked under realistic sampling.
