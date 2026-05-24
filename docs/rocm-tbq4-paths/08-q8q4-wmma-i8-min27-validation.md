# Q8/Q4 WMMA-I8 min27 validation note

Date: 2026-05-22
Branch commit: `e945b3342`
Route: ROCm `q8_0` K / `q4_0` V flash-attention lab path with `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27`

## Recommendation

Use WMMA-I8 for scoped q8_0/q4_0 route experiments with explicit layer filters.
The current validated policy is `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27` and
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MAX=38`, with
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6=1` for Qwen3.6-27B.

The route accelerates the QK stage with i8 WMMA. It is separate from the promoted
q8/tbq4 f16-temp path, which remains the serving baseline for 27B/35B MTP runs.

Evidence:

```text
benches/rocm-rdna3/q8q4-wmma-i8-generation-coherence-20260524-072905
benches/rocm-rdna3/q8q4-wmma-i8-targeted-risk-rerun-20260524-074150
benches/rocm-rdna3/q8q4-wmma-i8-seed42-repeat-risk-20260524-074609
benches/rocm-rdna3/q8q4-wmma-i8-repeat-controls-risk-20260524-074911
benches/rocm-rdna3/q8q4-wmma-i8-off-repeat-risk-20260524-082222
benches/rocm-rdna3/q8q4-wmma-i8-risk-layer-fix-20260524-081242
benches/rocm-rdna3/q8q4-wmma-i8-layermax-risk-20260524-082804
benches/rocm-rdna3/q8q4-wmma-i8-layer-single-risk-20260524-083543
benches/rocm-rdna3/q8q4-wmma-i8-policy-hunt-20260524-085621
benches/rocm-rdna3/q8q4-wmma-i8-policy-hunt-20260524-091219
benches/rocm-rdna3/q8q4-wmma-i8-broad-policy-matrix-20260524-093149
benches/rocm-rdna3/q8q4-wmma-i8-generation-coherence-20260524-093719
benches/rocm-rdna3/q8q4-wmma-i8-broad-policy-matrix-20260524-094459
```

Summary:

- The generation harness uses `llama-completion` with
  `docs/rocm-tbq4-paths/qwen36-merged-template.jinja`, `--jinja`,
  `--conversation --single-turn`, and a `<|think_off|>` system prompt.
- `min27_max38` passed the candidate broad logits matrix and broader
  generation/coherence smoke.
- Layer-set isolation identified layer 39 as the recurring prefix/hash-split
  contributor; policies ending at layer 38 were stable in the repeated hunts.
- The broad-matrix and generation/coherence scripts default to the candidate
  pair (`off min27_max38`). Use `MODE_PROFILE=full`, `MODE_LIST=...`,
  `CASE_LIST=...`, or `CASE_LIMIT=1` for diagnostic sweeps.

2026-05-24 GQA=6 reopen addendum: Qwen3.6-27B has `n_head=24`, `n_head_kv=4`
(GQA=6), so it remains behind an extra explicit
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6=1` gate rather than joining the base
GQA=4/8 gate. Backend-op parity now includes a GQA=6 q8_0/q4_0 causal-tail row
with and without attention sinks, and the runtime logs confirmed
`route=q8q4_wmma_i8 ... heads=24 ... gqa=6`. 27B candidate validation with the
extra gate passed:

```text
benches/rocm-rdna3/q8q4-wmma-i8-gqa6-backend-parity-20260524-114025
benches/rocm-rdna3/q8q4-wmma-i8-27b-gqa6-broad-candidate-20260524-114239
benches/rocm-rdna3/q8q4-wmma-i8-27b-gqa6-generation-candidate-20260524-114339
benches/rocm-rdna3/q8q4-wmma-i8-27b-gqa6-policy-repeat-20260524-114101
```

27B broad candidate results: `off` vs `min27_max38`, four prompts/eight runs,
`top1=true` for all comparisons, min top10 9, max rel RMS 0.0899371635, max KL
0.00756904956, max JS 0.00183906384, max TVD 0.0366849298, and max active
scratch 103.5 MiB under the adjusted 27B smoke bound. 27B generation candidate
passed 5/5 for both modes with route counts 3/6 and no repetition, marker, or
unexpected non-ASCII failures. 27B replicated policy hunt passed 4/4 off and
4/4 `min27_max38`, with stable hashes per policy.

Performance scope: q8_0/q4_0 WMMA-I8 is useful for selector/correctness work and
for comparing q8/q4 route variants. It is not the q8/tbq4 f16-temp serving path.
The current prototype accelerates QK with i8 WMMA while softmax, V dequant/PV,
and split-K reduction remain outside the i8 WMMA fast path. Use route logs plus
throughput artifacts when comparing it with f16-temp runs.

## Why min27

- It keeps fewer full-attention layers on the experimental route than `LAYER_MIN=23`.
- It matched top1 in the clean broad matrix.
- It had lower max/average relative RMS than `LAYER_MIN=23` in the broad matrix.
- It still delivered most of the measured performance gain over route-off.
- `LAYER_MIN=23` adds only about 2-6% over `LAYER_MIN=27` in the current perf sample.

## Correctness / logits matrix

Artifact:

```text
benches/rocm-rdna3/q8q4-wmma-i8-broad-clean-matrix-20260522-063822
```

Coverage:

```text
long_384_notes
long_640_actions
short_policy_sentence
short_regex_semver
```

Summary:

```text
all 12 comparisons top1=true
min top10 intersection = 9
max rel RMS = 0.057233864
```

By mode:

| mode | max rel RMS | avg rel RMS | route counts | min top10 |
|---|---:|---:|---|---:|
| `LAYER_MIN=23` | 0.057233864 | 0.039171467 | `[5, 10]` | 9 |
| `LAYER_MIN=27` | 0.039916571 | 0.032553890 | `[4, 8]` | 9 |
| `LAYER_MIN=23 + SKIP_LAYER=23` | 0.039916571 | 0.032553890 | `[4, 8]` | 9 |

In this model, `LAYER_MIN=27` and `LAYER_MIN=23 + SKIP_LAYER=23` produced
identical logits in the broad matrix.

## KLD / distribution drift

Artifact:

```text
benches/rocm-rdna3/q8q4-wmma-i8-broad-clean-matrix-20260522-063822/kld-summary.tsv
```

For `LAYER_MIN=27` across the 4-prompt matrix:

| metric | value |
|---|---:|
| max KL(off || min27) | 0.003748 nats |
| mean KL(off || min27) | 0.001499 nats |
| max JS divergence | 0.000958 nats |
| mean JS divergence | 0.000379 nats |
| max TVD | 0.03597 |
| mean TVD | 0.01614 |

Worst min27 case was `long_384_notes`:

```text
KL(off||min27)=0.003748
JS=0.000958
TVD=0.03597
top1=true
top10=9
off top1 prob=0.7006
min27 prob at off top1=0.6666
```

This is measurable distribution drift, but small in this smoke set and did not
move top1.

## Performance

Artifact:

```text
benches/rocm-rdna3/q8q4-wmma-i8-bench-min23-min27-20260522-073420
```

`llama-bench`, Qwen3.6-35B A3B IQ4_XS, `q8_0/q4_0` KV, FA on:

| n_prompt | off | min27 | min23 | min27 vs off | min23 vs off | min23 vs min27 |
|---:|---:|---:|---:|---:|---:|---:|
| 907 | 1037 tok/s | 1311 tok/s | 1391 tok/s | 1.265x | 1.341x | 1.060x |
| 1258 | 829 tok/s | 1076 tok/s | 1097 tok/s | 1.298x | 1.324x | 1.020x |

Interpretation: `LAYER_MIN=27` captures most of the performance gain while
avoiding the extra routed layer used by `LAYER_MIN=23`.

## Repro gate

Script added in `e945b3342`:

```text
scripts/hip/run-q8q4-wmma-i8-long384-repro.sh
```

It runs this matrix twice on `long_384_notes`:

```text
off
LAYER_MIN=23
LAYER_MIN=27
LAYER_MIN=23 + SKIP_LAYER=23
```

Clean gate artifact:

```text
benches/rocm-rdna3/q8q4-wmma-i8-long384-repro-matrix-20260522-054351
```

Result:

```text
gate_status=pass
```

## Thinking leakage / coherence note

Artifacts:

```text
benches/rocm-rdna3/q8q4-wmma-i8-coherence-min27-20260522-074015
benches/rocm-rdna3/q8q4-wmma-i8-coherence-min27-chat-20260522-074127
benches/rocm-rdna3/q8q4-wmma-i8-coherence-min27-budget0-20260522-074236
```

No-thinking leak smoke failed for both route-off and min27. The model emitted
`<think>` / thinking-process text even when reasoning was disabled or budgeted to
zero. Since route-off leaked too, this is a model/template/serving behavior, not
evidence against the WMMA min27 route.

Do not claim no-thinking safety for this invocation. Fix or sanitize thinking
output separately before using these prompts as user-visible coherence tests.

## Current policy

Validated scoped env for q8_0/q4_0 WMMA-I8 experiments:

```bash
GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MAX=38
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_REQUIRE_SELECTED=1
```

Add `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_ALLOW_GQA6=1` for Qwen3.6-27B.
