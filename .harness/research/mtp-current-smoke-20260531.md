# MTP current smoke — 2026-05-31

User clarified the original TBQ4 numbers were MTP-related, so a quick server-side MTP `tg32` smoke was run instead of relying only on the llama-bench acceptance script.

## Current local build

Binary:

```text
/home/mrtrent/llama.cpp-tree-tbq4-rdna3-github/build-rocm-fixed/bin/llama-server
```

Command shape:

```text
--spec-type draft-mtp --spec-default --spec-draft-n-max 3 --spec-draft-p-min 0
--spec-draft-prio 2 --spec-draft-prio-batch 2
--cache-type-k-draft q8_0 --cache-type-v-draft tbq4_0
--cache-type-k f16 --cache-type-v q4_0
N_PREDICT=32, prompt='The capital of France is'
```

Artifact:

```text
.harness/tmp/mtp-current-specdefault-tg32-20260531-083034/
```

Result:

```text
predicted_n=32
predicted_per_second=29.8291
draft_n=29
draft_n_accepted=21
```

A speed-probe run without `--spec-default` but with draft cache envs was similar:

```text
.harness/tmp/mtp-current-tg32-smoke-20260531-082121/
predicted_per_second=30.6621
draft_accept_rate=0.37209
```

## Older May17 build smoke

Binary:

```text
/home/mrtrent/llama.cpp-mtp-tbq4-rdna3/build-rocm/bin/llama-server
```

Artifact:

```text
.harness/tmp/mtp-oldbuild-tg32-smoke-20260531-082239/
```

Result:

```text
predicted_n=32
predicted_per_second=42.2373
draft_n=29
draft_n_accepted=21
draft_accept_rate=0.72414
```

## Interpretation

These server-side MTP smoke numbers do not reproduce the user-provided original `tg32 ~60 tok/s` table. Current local build is around 30 tok/s for this smoke; older May17 server binary is around 42 tok/s.

This means at least one of the following differs from the original table:

```text
- benchmark harness/tooling;
- production v3 binary/branch;
- runtime flags/env;
- prompt/depth/request shape;
- cache/layout route;
- whether the table measured llama-bench base tg32 vs server MTP accepted-token throughput.
```

The llama-bench acceptance script was also fixed locally to remove invalid `-pg pp/tg` usage; llama-bench generates pp/tg from `-p 2048 -n 32` directly.
