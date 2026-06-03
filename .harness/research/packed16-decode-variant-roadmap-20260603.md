# Packed16 MTP Draft Decode Variant Roadmap

Context: `nq == 1`, `K = I32 packed16 + half scales`, `V = q4_0`, `dst = f32`, `D = 256`, MTP draft decode (`GGML_FATTN_INST_MTP_DRAFT_DECODE_QK`). Route identity is `BEST_FATTN_KERNEL_PACKED16_DECODE`; implementation selection lives in `ggml/src/ggml-cuda/fattn-dot4-q8k-kq.cu` and kernels/macros in `ggml/src/ggml-cuda/fattn-dot4-q8k-decode.cuh`.

## Implemented / selectable today

- `rocm_packed16_decode`, `rocm_packed16_decode_scalar`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=scalar`
  - One CTA per `(batch, query_head)`, BN64 online softmax, q4_0 V.
- `rocm_packed16_decode_q4pair`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=q4pair`
  - Same QK as scalar; pairs q4 nibbles per thread for V.
- `rocm_packed16_decode_gqa_scalar`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=gqa_scalar`
  - One CTA per `(batch, kv_head)`, up to `GH_MAX=8` query heads share K/V tile reads.
- `rocm_packed16_decode_splitk`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=splitk`
  - Stage1 partial unnormalized `(o,m,l)` per K split; stage2 online-softmax merge.
- `rocm_packed16_decode_logits_debug`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=logits_debug`
  - Two-stage isolation: global logits via packed16 KQ, then q4_0 V attention from logits.
- `rocm_packed16_decode_dsplit`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL=dsplit`
  - One CTA per output-D slice (`D_CHUNK=64`). Recomputes QK+softmax per slice and writes disjoint `dst[D]`; graph-safe and reduction-free, but opt-in only because QK is duplicated.
- `rocm_packed16_decode_pvwmma`, `rocm_packed16_decode_gqa_pvwmma`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL={pvwmma,gqa_pvwmma}`
  - One CTA per KV head. QK/probs remain scalar/direct packed16 dot4; PV uses RDNA3 f16 WMMA tiles (`P[16x16] * V[16x16]`) with GH rows padded to 16.
- `rocm_packed16_decode_waveqk`, `rocm_packed16_decode_waveqk_q4pair`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL={waveqk,waveqk_q4pair,gqa_waveqk}`
  - One wave cooperatively computes one packed16 QK logit using two i32 words per lane and a wave reduction; PV remains scalar/q4pair.
- `rocm_packed16_decode_wmma_full`, `rocm_packed16_decode_gqa_wmma_full`
  - `GGML_CUDA_ROCM_PACKED16_DECODE_IMPL={wmma_full,gqa_wmma_full}`
  - One CTA per KV head. QK uses RDNA3 i8 WMMA over packed16 fragments, then PV uses the same f16 WMMA route as PV-WMMA.

## Explicit aliases reserved for complex variants

All listed complex aliases are now buildable opt-in routes. None are automatic defaults until correctness/perf matrices pass.

## Variant C — wave-parallel QK dot4 decode

Implemented as a reduced-risk opt-in kernel after the first attempt hit a ROCm compiler ICE:

- 8 waves per 256-thread CTA.
- Each wave computes one `(head, k)` logit at a time.
- Each lane consumes two packed16 i32 words (`64` words total) and reduces within the wave.
- Store BN logits to shared memory; reuse existing scalar/q4pair online softmax/PV loop.

Prior blocker retained as caution:

- First wave-QK implementation crashed ROCm 7.2.3 AMDGPU backend in Register Coalescer (`clang++: error: unable to execute command: Segmentation fault (core dumped)`).
- Current implementation avoids large per-thread arrays/template fanout and builds on ROCm 7.2.3, but still needs route-specific correctness/perf validation before default use.

## Variant F / 18 — grouped-GQA PV-WMMA decode

Implemented as an opt-in first WMMA decode variant:

- CTA owns one `(batch, kv_head)` and `GH <= 8` query heads.
- QK remains packed16 scalar/direct dot4; logits/probs are `GH x BN` in shared memory.
- PV uses RDNA3 f16 WMMA builtins: `P[16 x 16] * V[16 x 16] -> O[16 x 16]`.
- GH rows above the active GQA group are zero-padded.
- q4_0 V is dequantized directly into WMMA B fragments.
- Output covers full `D=256` with 8 waves and two 128-D passes per CTA.

Caveats before default promotion:

- This is not a full-QK WMMA path; only PV is WMMA.
- q4_0 direct dequant into fragments may limit speedup versus staged-V variants.
- Needs scalar-vs-PVWMMA correctness and perf matrix over `nk` and `GH` before auto-selection.

## Variant G / 19 — grouped-GQA full WMMA QK + PV decode

Implemented as an opt-in first full-WMMA decode route:

- CTA owns one `(batch, kv_head)` and `GH <= 8` query heads.
- QK uses RDNA3 i8 WMMA (`v_wmma_i32_16x16x16_iu8`) over packed16 Q/K fragments.
- QK accumulates each K16 WMMA result into f32 logits with per-32-dim Q/K scales.
- Online-softmax remains the shared f32 path.
- PV uses RDNA3 f16 WMMA (`P[16x16] * V[16x16]`) as in the PV-WMMA route.

Caveats before default promotion:

- This is a fixed first specialization: `BN=64`, `GH_MAX=8`, `D=256`.
- GH rows above the active group are padded/masked.
- Must pass scalar-vs-full-WMMA route correctness over `nk` and `GH`, then live decode parity/perf, before any auto-selection.

## Variant I — D-split decode

Implemented as an opt-in debug/perf exploration route. It is deliberately not automatic because it duplicates QK for each D slice.

Use when investigating whether decode is PV/output limited enough that extra CTAs offset duplicated QK. If it wins in a narrow context, the next production form should pair D-split with shared/global logits (similar to `logits_debug`) to avoid duplicate QK.

## Acceptance matrix before production default

- Correctness: compare scalar vs variant for `nk={128,512,1024,2048,4096,8192,16384}`, `GH={1,2,4,6,8}` where available.
- Metrics: max_abs, RMS, top-k/logit stability, output token parity on short live prompts.
- Runtime: prompt contexts 1k/6k/16k with 16-32 decode tokens, route logs enabled.
- Stop condition: no silent fallback; route alias must either run the selected kernel or abort with a specific reason.
