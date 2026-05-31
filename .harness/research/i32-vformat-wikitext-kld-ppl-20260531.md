# WikiText KLD/PPL smoke for I32 packed16 V formats — 2026-05-31

## Goal

Run a more meaningful fast quality smoke than the synthetic prompt test by using WikiText-2 raw text.

Compare V formats while keeping K on the I32 packed16 DOT4/MMQ route:

```text
baseline:  V=f16
candidate: V=q4_0
candidate: V=q8_0
```

## Command profile

All runs used:

```text
model: Qwen3.6-27B-Q4_K_M-mtp.gguf
source: /home/mrtrent/llama.cpp-mtp-tbq4-rdna3/wikitext-2-raw/wiki.test.raw
ctx: 512
chunks: 4
batch: 512
ubatch: 512
flash-attn: on
route require: GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq
```

No K cache-type CLI flag was used.

## Artifact

```text
.harness/tmp/i32-vformat-wikitext-kld-ppl-20260531-105304
```

Important files:

```text
wiki.test.raw
base-f16.logits
f16-baseline.log
q4-kld.log
q8-kld.log
summary.json
```

Artifact size:

```text
485M
```

## Route contract

All runs stayed on the intended route:

```text
f16 baseline: 32/32 rocm_packed16_dot4_mmq K=i32 V=f16
q4_0 KLD:    32/32 rocm_packed16_dot4_mmq K=i32 V=q4_0
q8_0 KLD:    32/32 rocm_packed16_dot4_mmq K=i32 V=q8_0
```

Errors scanned and absent:

```text
I32 K rejected
GGML_ASSERT
fatal error
ABORT
failed to decode
required rocm_packed16_dot4_mmq route was not selected
```

No lingering model processes after the run.

## Runtime

```text
f16 baseline: 9.11 s
q4_0 KLD:     8.93 s
q8_0 KLD:     8.45 s
```

## Results

### f16 V baseline

```text
PPL = 5.6891 ± 0.44594
```

### q4_0 V vs f16 V

```text
Mean PPL(Q):               5.686762 ± 0.444911
Mean PPL(base):            5.675535 ± 0.442247
Mean ln(PPL(Q)/base):      0.001976 ± 0.004212
Mean PPL(Q)/base:          1.001978 ± 0.004220
Mean PPL(Q)-base:          0.011227 ± 0.024010
Mean KLD:                  0.004550 ± 0.000338
Median KLD:                0.001818
90.0% KLD:                 0.009897
95.0% KLD:                 0.016511
99.0% KLD:                 0.050043
99.9% KLD:                 0.141503
Maximum KLD:               0.147076
RMS Δp:                    1.984 ± 0.166 %
Same top p:               97.059 ± 0.529 %
```

### q8_0 V vs f16 V

```text
Mean PPL(Q):               5.681263 ± 0.445554
Mean PPL(base):            5.675535 ± 0.442247
Mean ln(PPL(Q)/base):      0.001009 ± 0.003353
Mean PPL(Q)/base:          1.001009 ± 0.003356
Mean PPL(Q)-base:          0.005728 ± 0.019112
Mean KLD:                  0.002825 ± 0.000334
Median KLD:                0.000863
90.0% KLD:                 0.004887
95.0% KLD:                 0.008788
99.0% KLD:                 0.031250
99.9% KLD:                 0.134189
Maximum KLD:               0.215891
RMS Δp:                    1.683 ± 0.171 %
Same top p:               97.941 ± 0.445 %
```

## Interpretation

This WikiText smoke is much more meaningful than the earlier synthetic prompt check, though still small (`4` chunks, about `1020` evaluated tokens per candidate).

Findings:

- f16 baseline PPL is sane on WikiText for this short slice: `5.6891 ± 0.44594`.
- q4_0 V and q8_0 V both have tiny PPL deltas vs f16 V relative to uncertainty.
- q8_0 V is closer to f16 V than q4_0 V by mean/median KLD and same-top probability.
- q4_0 V still looks viable as the default compression target on this smoke.

Suggested next stronger run:

```text
ctx=512, chunks=16
```

or:

```text
ctx=1024, chunks=8
```
