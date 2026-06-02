# Packed16 FA Producer/Consumer Overhaul Design — 2026-06-02

## Evidence

Profile artifacts:

- PWMMA baseline: `/home/mrtrent/.harness/artifacts/pwmma-profile-context-sweep-20260602-211758/summary.json`
- Forced DOT4/PDMQ: `/home/mrtrent/.harness/artifacts/dot4-vs-pwmma-profile-20260602-214038/comparison.json`

Forced DOT4/PDMQ command:

```bash
GGML_CUDA_PDMQ_PROFILE=1 \
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 \
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq \
./build-rocm-ninja/bin/llama-bench \
  -m /mnt/CC6AA71F6AA70574/models/MTP/Qwen3.6-27B-Q4_K_M-mtp.gguf \
  -p 1024,2048,4096 -n 0 -d 0 -b 4096 -ub 1024 \
  -ctk f16 -ctv f16 -ngl 99 -fa 1 -r 1 -o json
```

## DOT4 vs PWMMA result

Prompt throughput:

| route | pp1024 | pp2048 | pp4096 |
|---|---:|---:|---:|
| PWMMA standard PV-WMMA | 908.25 tok/s | 915.98 tok/s | 865.95 tok/s |
| forced DOT4/PDMQ | 903.81 tok/s | 882.15 tok/s | 759.41 tok/s |

Phase comparison, median per effective `nk`:

| nk | DOT4 ms | PWMMA ms | DOT4/PWMMA ms | DOT4 QK / PWMMA QK | DOT4 PV / PWMMA PV | DOT4 QK/SM/PV |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 6.219 | 2.214 | 2.81x | 3.15x | 12.76x | 15.1 / 5.3 / 79.6% |
| 2048 | 13.557 | 5.671 | 2.39x | 2.36x | 9.73x | 14.9 / 4.8 / 80.3% |
| 3072 | 21.081 | 9.103 | 2.32x | 2.33x | 9.09x | 15.7 / 4.7 / 79.7% |
| 4096 | 29.835 | 12.287 | 2.43x | 2.30x | 8.77x | 15.9 / 4.6 / 79.5% |

Conclusion: forced DOT4/PDMQ is not a good long-context prefill replacement. Its PV path dominates (~80%) and is 8.8–12.8x slower than PV-WMMA. DOT4 QK is also 2.3–3.1x slower than i8 WMMA QK in this path. Do not pursue DOT4-QK + scalar/PDMQ-PV for prefill.

## Big-overhaul direction

The target is not a DOT4 replacement. The target is a producer/consumer PWMMA pipeline that keeps the current i8-QK + PV-WMMA math but reduces sequential per-K-tile stalls and V/K traffic exposure.

Current BM64 PV-WMMA structure is effectively serialized per K tile:

```text
load/decode V tile
optional K staging
QK WMMA for tile k
mask + online softmax for tile k
PV-WMMA for tile k
repeat k+1
```

The FA3/FA4-inspired RDNA3 analogue should be:

```text
producers: prefetch/stage K/V tile k+1
consumers: QK/softmax/PV on tile k
```

RDNA3 has no Hopper TMA/WGMMA/TMEM, so this is software double buffering in LDS/registers, not true async tensor-memory overlap.

## Proposed route

New opt-in implementation name first:

```text
GGML_CUDA_ROCM_PACKED16_WMMA_IMPL=bm64_i8qk_pvwmma_db_512t_wavegate_stagev
```

Promote only after proof.

### CTA/thread partition

Keep 512-thread CTA and BM64/BN16 first.

- Wave 0-3: QK i8 WMMA consumers, one 16-row group each.
- Wave 4-7: PV-WMMA consumers, one 16-row group each, or reused as current PV groups.
- Loader role: not dedicated permanently at first; use all 512 threads for cooperative K/V load into inactive buffer, then compute. If register pressure allows, reserve one or two waves for loader in a later variant.

Rationale: persistent dedicated loader waves may starve QK/PV on RDNA3; start with double-buffered LDS and current wavegate ownership.

### Double buffers

Allocate two V buffers:

```cpp
__shared__ half v_tile_f16[2][BN][D];
```

For K-shared variants only, allocate two K buffers:

```cpp
__shared__ int  k_i32_smem[2][BN * D/4];
__shared__ half k_s_smem[2][BN * D/QK8_0];
```

For default PV-WMMA (currently no K_SHARED), double-buffer V first. K payload remains direct global read by QK, because prior K_SHARED did not win. Re-test K double-buffer only after V double-buffer is measured.

### Pipeline sketch

1. Preload V tile 0 into buffer 0.
2. For each `kt`:
   - launch cooperative preload for tile `kt+1` into `next` buffer as early as possible;
   - compute QK for current tile using current K payload and current V buffer;
   - mask + softmax current logits;
   - PV-WMMA current probabilities against current V buffer;
   - swap buffers.

Because HIP shared-memory loads are not async like TMA, initial implementation will still use barriers. The goal is to separate load placement and reduce repeated V load exposure, not promise full Hopper-style overlap.

### Practical first implementation

Implement a conservative `db_v` kernel:

```text
bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev
```

- Double-buffer V only.
- No K_SHARED.
- Same math as impl 14.
- Same probes: QK probe, i8 DOT4 shadow optional, PV WMMA probe/live shadow.
- Same final output.

Acceptance:

```text
PBWMMA I8 QK probe PASSED
PBWMMA PV WMMA probe PASSED
optional live DOT4/PV shadow PASSED
llama-bench pp1024/2048/4096 no regression >1% vs impl 14
GGML_CUDA_PWMMA_PROFILE shows reduced PV ms or total ms
```

### Second implementation if DBV wins

Implement `dbvk`:

```text
bm64_i8qk_pvwmma_dbvk_512t_wavegate_stagev
```

- Double-buffer V + K payload/scales.
- Reuse K_SHARED ideas, but only if profile shows QK global/K traffic exposure remains significant.

### Third implementation if scheduling dominates

Add LPT/reverse causal q-tile scheduling outside kernel selection:

```text
GGML_CUDA_ROCM_PACKED16_WMMA_QTILE_ORDER=reverse_causal
```

This is lower math risk but may require output-index remapping or grid-to-q tile swizzle.

## What not to do next

- Do not replace PV-WMMA with DOT4/PDMQ PV for prefill: profile shows huge PV loss.
- Do not prioritize softmax polynomial work first: PWMMA profile has softmax at ~13%.
- Do not promote BN32 yet: earlier pp2048 result was slower than BN16 PV-WMMA.

## Implementation status

Initial opt-in impl ID 16 was added:

```text
16 = bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev
```

The first attempted 3D shared V-buffer inside the generic template (`v_tile_f16_db[2][BN][D]`) built through semantic checking but crashed ROCm clang 22 during backend codegen, similar to earlier BN32 template-pressure failures. The next patch split impl 16 into a dedicated DBV kernel with fewer template parameters and a real alternating V buffer.

Smoke artifact: `dbv-dedicated-smoke-20260602-220035` selects `IMPL=bm64_i8qk_pvwmma_dbv_512t_wavegate_stagev`, passes QK/i8/PV probes, emits `PWMMA PROFILE` lines, and runs pp1024 at `915.04 tok/s` in a one-rep smoke.

A/B artifact: `dbv-vs-pvwmma-ab-20260602-220737` compares standard impl 14 vs dedicated DBV at pp1024/2048/4096. Throughput is neutral (`-0.13%`, `-0.02%`, `+0.01%`), while profiled kernel medians improve by roughly `1.6–3.1%`. Do not promote plain DBV yet; keep as opt-in scaffold. The next real overlap attempt should reduce whole-graph time, not just profiled kernel medians.

Follow-up artifact: `pwmma-bfrag-hoist-9b-ab-20260602-225954` applies a DOT4/PDMQ lesson inside the PWMMA route rather than promoting standalone DOT4: hoist the PV-WMMA V/B fragment out of the row-block loop in impl 16 so the same staged V fragment is reused across the four BM64 row groups. Route remains `rocm_packed16_wmma_tile`; probes pass. Profiled kernel medians improve ~18% (`nk=1024..4096`) with PV share dropping from ~51% to ~36%. No-profile 9B confirmation `pwmma-bfrag-hoist-9b-noprofile-20260602-230034` shows pp2048 `+0.95%` and pp4096 `+1.49%` throughput over impl 14. Longer-context 9B artifacts `pwmma-bfrag-hoist-9b-longctx-20260602-230537` and `pwmma-bfrag-hoist-9b-32k-20260602-230644` show the end-to-end gain scales with context: pp8192 `+2.98%`, pp16384 `+6.22%`, pp32768 `+9.93%`, all forced to `rocm_packed16_wmma_tile` with no DOT4/vec fallback. Multi-rep artifact `pwmma-bfrag-hoist-9b-longctx-r3-20260602-231025` confirms the signal at pp16384 `+4.57%` (`2132.94±3.77` → `2230.45±2.83 tok/s`) and pp32768 `+9.68%` (`1586.24±4.63` → `1739.76±3.26 tok/s`) with zero route-contract violations. 27B single-rep confirmations `pwmma-bfrag-hoist-27b-longctx-20260602-232148` and `pwmma-bfrag-hoist-27b-32k-20260602-232504` also stay on `rocm_packed16_wmma_tile` and show pp8192 `+12.88%`, pp16384 `+9.11%`, pp32768 `+7.97%`. 27B MTP cohesion artifact `pwmma-hoist-cohesion-27b-mtp-word-16k-20260602-235032` passes a 16k word-retrieval gate: standard and DBV hoist both output `apple`, outputs are identical, route counts are clean (`rocm_packed16_wmma_tile` only for prefill; no DOT4-MMQ/vec fallback or route-contract violations). Impl 16 is now promoted to the production long-context auto PWMMA prefill route.
