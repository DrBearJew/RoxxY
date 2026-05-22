# Q8/Q4 WMMA-I8 min27 validation note

Date: 2026-05-22
Branch commit: `e945b3342`
Route: ROCm `q8_0` K / `q4_0` V flash-attention lab path with `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27`

## Recommendation

Keep `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27` as the conservative lab policy.

Do not promote unrestricted Q8/Q4 WMMA-I8 routing to a default. Keep both
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1` and
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1` opt-in.

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

Use this only as an explicit lab opt-in:

```bash
GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1
GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27
```

Keep defaults route-off.
