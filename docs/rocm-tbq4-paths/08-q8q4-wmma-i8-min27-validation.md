# Q8/Q4 WMMA-I8 min27 validation note

Date: 2026-05-22
Branch commit: `e945b3342`
Route: ROCm `q8_0` K / `q4_0` V flash-attention lab path with `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27`

## Recommendation

Keep `GGML_CUDA_ROCM_Q8Q4_WMMA_I8_LAYER_MIN=27` as the conservative lab policy.

Do not promote unrestricted Q8/Q4 WMMA-I8 routing to a default. Keep both
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8=1` and
`GGML_CUDA_ROCM_Q8Q4_WMMA_I8_UNSAFE=1` opt-in.

2026-05-24 addendum: keep `LAYER_MIN=27` as the conservative starting point, but
prefer `LAYER_MAX=38` for the next lab candidate. `min23` and `min23_skip24`
remain useful diagnostics/perf probes, but do not promote them as the lab policy
because the risk-table generation prompt showed repeatable stochastic flakiness
in the earlier routed layer class.

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

- The generation harness now uses `llama-completion` with
  `docs/rocm-tbq4-paths/qwen36-merged-template.jinja`, `--jinja`,
  `--conversation --single-turn`, and a `<|think_off|>` system prompt.
- Broad smoke: route-off, `min27`, and `min23` passed 5/5; `min23_skip24`
  passed 4/5 and failed the risk-table prompt with unexpected non-ASCII prefix
  `فاق`.
- Targeted seed sweep for `min23_skip24` passed 10/10 across seeds 40-49, but
  exact seed-42 repeats reproduced the same two hashes with 3/12 failures.
- Control repeats for the same prompt/seed showed `min23` 1/8 failures and
  `min27` 8/8 passes in the first control run, but a later stricter prefix check
  saw `min27` produce a different weird prefix (`无影`) once in 6 repeats.
- Route-off repeated 12/12 stable for the same prompt/seed.
- Single routed layers 23, 27, 31, 35, and 39 each repeated 8/8 stable, so this
  is not an obvious single-layer corruption.
- Layer-set isolation showed `min27_max38`, `min27_max37`, `min24_max38`, and
  `min23_skip23_max38` repeated 8/8 stable and converged to the same risk-table
  hash with three routed full-attention layers.
- Meta policy hunt `20260524-085621` repeated 12 times: `off`, `min27_max38`,
  and `min23_skip23_max38` passed with one hash; `min27`, `min23`, and
  `min23_skip24` failed with weird prefixes/hash splits.
- Follow-up one-layer-removal hunt `20260524-091219` showed every passing
  candidate removed layer 39 (`min27_minus39`, `min23_max38`, `min23_minus39`),
  while failing candidates that kept layer 39 produced either `فاق` or `无影`.
- Comparison broad logits matrix `20260524-093149` showed `min27_max38` top1
  parity on all four prompts, min top10 overlap 9, route counts 3/6, active
  scratch max 69 MiB, max rel RMS 0.039252117, max KL 0.00352963842, max JS
  0.000903031675, and max TVD 0.0341042235. The overall comparison gate failed
  only because diagnostic `min23_max38` had top10 min 8.
- Broader generation/coherence smoke `20260524-093719` passed 5/5 for every
  tested mode, including `min27_max38`, under the special Qwen template and
  `<|think_off|>` prompt. No unexpected non-ASCII, marker leak, NaN/Inf, or
  repetition failure appeared.
- Candidate-only broad logits matrix `20260524-094459` passed its gate for
  `off` vs `min27_max38`, with the same top1/top10/drift envelope as the
  comparison matrix.

Decision: do not promote `min23` or `min23_skip24`. Treat `LAYER_MIN=27
LAYER_MAX=38` as the current opt-in lab candidate because it excludes the final
full-attn layer from this route, fixed the observed weird-prefix prompt in
replicated policy hunts, passed the target broad logits matrix, and passed the
broader generation/coherence smoke. Keep WMMA-I8 disabled by default; rerun the
broad matrix, policy hunt, and generation/coherence gates after route changes or
before any broader default/policy promotion. To avoid accidental slow 35B sweeps,
the broad-matrix and generation/coherence scripts default to the candidate pair
only (`off min27_max38`), with `MODE_PROFILE=full` or explicit `MODE_LIST=...`
reserved for diagnostic sweeps. Use `CASE_LIST=...` or `CASE_LIMIT=1` for a
2-run 35B smoke. For future hunts, use
`scripts/hip/run-q8q4-wmma-i8-policy-hunt.sh` instead of ad-hoc repeats; it
defaults to `off` vs `min27_max38` with two repeats each, while
`POLICY_PROFILE=full` restores the old multi-policy repeat hunt. It classifies
weird prefixes/non-ASCII/hash splits and emits one-layer-removal candidates for
delta debugging.

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
4/4 `min27_max38`, with stable hashes per policy. This reopens 27B as a lab
candidate only; keep the extra env gate until longer-context and route-policy
coverage match the GQA=4/8 path.

Do not use this as a normal 27B serving path. A follow-up speed sanity exposed a
baseline mistake: normal f16/f16 27B prefill was about 905 tok/s for pp4096,
and promoted q8/tbq4 with f16-temp was about 766 tok/s, while the q8_0/q4_0
WMMA-I8 lab lane was only about 118 tok/s in the same 4k shape. The WMMA-I8
lane can be faster than the broken q8_0/q4_0 fallback it replaces, but it is
not a replacement for the normal fast baseline. Keep `GGML_CUDA_ROCM_Q8Q4_WMMA_I8*`
out of default env files and recipes.

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
