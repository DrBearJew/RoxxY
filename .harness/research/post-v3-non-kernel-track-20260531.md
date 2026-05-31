# Post-v3 non-kernel track — 2026-05-31

## Premise

User clarified that the v3 MMVQ/B2 kernel is already in production. Do not spend this lane on another B2 kernel attempt from this checkout.

Current local checkout tip inspected for this note:

```text
b10d538b3
```

Local branch does not include the production v3 kernel in tracked source at this point, so any post-v3 performance claims need the production v3 binary/branch/artifact before profiling.

## Pivot

Treat B2 as externally handled. Next work should target production-safe non-kernel items:

1. **Post-v3 bottleneck refresh**
   - Profile the actual production v3 binary/branch, not the old b10d538b3 checkout.
   - Output: top kernels/API/copy/sync after v3, tokens/sec, next target.

2. **Acceptance/canary hardening**
   - Keep route canaries and no-spec deterministic canary around v3.
   - Add a compact deterministic output check for the MMVQ path because backend-op tests missed v2 server drift.

3. **Host/runtime cleanup only if post-v3 profile supports it**
   - Candidate areas: graph launch count, quantize_q8_1, k_get_rows_float, rms_norm/bin-bcast chains, scheduler syncs.
   - Do not enable A3 sync gates by default without longer v3 evidence.

## Immediate next acceptance criteria

```text
- Use production v3 binary/branch/artifact.
- Run no-spec deterministic 64/256-token output match vs known-good if possible.
- Run route canaries.
- Capture copy/sync trace and ROCprof summary.
- Choose one non-kernel target based on post-v3 evidence.
```

## Acceptance harness

Added:

```text
benchmarks/tbq4-post-v3-acceptance.sh
```

Purpose:

```text
Run llama-bench pp2048/tg32 at depths 0,4096,8192,16384,32768 and compare against the original TBQ4 baseline with MIN_RATIO=0.95 by default.
```

Use with the actual production v3 binary/model, for example:

```bash
LLAMA_BENCH=/path/to/v3/llama-bench \
MODEL=/path/to/tbq4-model.gguf \
OUT_DIR=.harness/tmp/tbq4-v3-acceptance-$(date +%Y%m%d-%H%M%S) \
benchmarks/tbq4-post-v3-acceptance.sh
```
