# MTP Persistent Packed16/MMQ Experiment

## Objective

Evaluate whether MTP draft verify can safely use persistent packed16 K-cache routing through `rocm_packed16_dot4_mmq` before considering any default-policy change.

## Production policy

Defaults stay conservative:

- target FA/DOT4 remains the production path;
- MTP draft FA remains disabled unless `LLAMA_MTP_ENABLE_FA=1`;
- MTP persistent packed16 remains disabled unless `LLAMA_MTP_DISABLE_PACKED16_FA=0`;
- no user-facing flag is added for this experiment.

## Experiment enablement

Required env for the hostile MTP packed16/MMQ lane:

```bash
LLAMA_MTP_ENABLE_FA=1
LLAMA_MTP_FA_INST=verify
LLAMA_MTP_DISABLE_PACKED16_FA=0
GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1
GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1
GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1
GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1
GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq
```

## Artifacts added

- `scripts/check-mtp-packed16-mmq-canary.sh`
  - Runs strict isolation with `RUN_MTP_PACKED16_MMQ=1`.
  - Requires MTP-specific route selection, not target-side packed16:
    - `fa_final_select: inst=mtp_verify_qk selected=rocm_packed16_dot4_mmq ... K=i32 V=f16`
  - Requires `PDMQ QK probe PASSED`.
  - Requires `bad_h=0`, `bad_logits=0`, nonzero generated/accepted drafts, and nonzero `d1`.
- `benchmarks/mtp-packed16-route-matrix.sh`
  - Compares route families using identical prompt/seed/server settings.
  - Default lanes: `source-dot4 packed16-mmq`.
  - Optional lanes:
    - `pwmma-bm32-directv`
    - `pwmma-bm32-stagev`
    - `pwmma-bm64-512t`
    - `pwmma-bm64-wavegate-directv`
  - Emits:
    - `summary.csv`
    - `summary.json`
    - `routes.csv` with route, shape, correctness, acceptance, timing, and failure class.

## Commands run

### Canary

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
OUT_DIR=/tmp/mtp-packed16-mmq-canary-proceed2 \
BASE_PORT=19320 \
N_PREDICT=4 \
scripts/check-mtp-packed16-mmq-canary.sh
```

Result:

```text
PASS: MTP packed16/MMQ canary bad_h=0 checked_h=46080 bad_logits=0 accepted=1 generated=1 d1=1/1 nq=2/4 route=rocm_packed16_dot4_mmq
```

### Route-family matrix

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
OUT_DIR=/tmp/mtp-packed16-route-matrix-proceed2 \
BASE_PORT=19520 \
N_PREDICT=4 \
ROUTES='source-dot4 packed16-mmq' \
benchmarks/mtp-packed16-route-matrix.sh
```

Result summary:

```csv
case,failure_class,selected,nq_seen,nk_seen,kv,pdmq_probe_passed,pwmma_seen,pwmma_impl,bad_h,checked_h,bad_logits,accepted,generated,rate,prompt_eval_ms,eval_ms,total_ms
packed16-mmq,ok,rocm_packed16_dot4_mmq,2/4,256,K=i32 V=f16,1,0,,0,46080,0,1,1,1.0,242.11,89.62,331.73
source-dot4,ok,vec/rocm_q8k_dot4_kq,2/4,256,K=f16 V=f16,0,0,,0,46080,0,1,1,1.0,246.20,90.44,336.64
```

### Acceptance smoke

```bash
cd /home/mrtrent/llama.cpp-tree-tbq4-rdna3-github
OUT_DIR=/tmp/mtp-packed16-acceptance-proceed \
BASE_PORT=19620 \
N_PREDICT=6 \
FA_VALUES='on' \
N_MAX_VALUES='1 2' \
P_MIN_VALUES='0 0.4' \
PROMPTS_FILE=benchmarks/prompts/mtp-acceptance-smoke.txt \
EXTRA_ENV='LLAMA_MTP_ENABLE_FA=1 LLAMA_MTP_FA_INST=verify LLAMA_MTP_DISABLE_PACKED16_FA=0 GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1 GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1 GGML_CUDA_ROCM_PACKED16_DOT4_MMQ=1 GGML_CUDA_ROCM_PACKED16_AUTO_VERBOSE=1 GGML_CUDA_FA_ROUTE_REQUIRE=rocm_packed16_dot4_mmq' \
benchmarks/mtp-acceptance-matrix.sh
```

Aggregate result:

```text
rows=28
bad_h_sum=0
bad_logits_sum=0
with_acceptance_stats=26
generated_sum=94
accepted_sum=46
zero_generated_or_no_stats=p00_fa-on_n1_p0.4,p00_fa-on_n2_p0.4
```

### PWMMA quick route lanes

All quick PWMMA route lanes below used `N_PREDICT=4`, `nq=2/4`, `nk=256`, `K=i32 V=f16`.

```csv
case,failure_class,selected,pwmma_impl,bad_h,checked_h,bad_logits,accepted,generated,rate,prompt_eval_ms,eval_ms,total_ms
pwmma-bm32-directv,ok,rocm_packed16_wmma_tile,bm32_regout_directv,0,46080,0,1,1,1.0,262.76,94.04,356.80
pwmma-bm32-stagev,ok,rocm_packed16_wmma_tile,bm32_regout_stagev,0,46080,0,1,1,1.0,250.39,96.54,346.93
pwmma-bm64-512t,ok,rocm_packed16_wmma_tile,bm64_regout_directv_512t,0,46080,0,1,1,1.0,269.19,98.80,367.99
pwmma-bm64-wavegate-directv,ok,rocm_packed16_wmma_tile,bm64_512t_wavegate_directv,0,46080,0,1,1,1.0,251.96,100.27,352.23
```

### Long-context shape sweep

Long prompt run with `CTX_SIZE=4096`, `BATCH_SIZE=256`, `UBATCH_SIZE=256`, `N_PREDICT=8` reached `nk=2048` and broad prefill/verify shapes.

```csv
case,failure_class,selected,nq_seen,nk_seen,kv,bad_h,checked_h,bad_logits,accepted,generated,rate,prompt_eval_ms,eval_ms,total_ms
source-dot4,ok,vec/rocm_q8k_dot4_kq,2/4/8/252/256,256/512/768/1024/1280/1536/1792/2048,K=f16 V=f16,0,9267200,0,3,3,1.0,4097.66,338.00,4435.66
packed16-mmq,ok,rocm_packed16_dot4_mmq,2/4/8/252/256,256/512/768/1024/1280/1536/1792/2048,K=i32 V=f16,0,9297920,0,1,5,0.2,4107.18,390.60,4497.78
pwmma-bm32-directv,ok,rocm_packed16_wmma_tile,2/4/8/252/256,256/512/768/1024/1280/1536/1792/2048,K=i32 V=f16,0,9282560,0,2,4,0.5,4162.07,414.91,4576.98
pwmma-bm64-512t,ok,rocm_packed16_wmma_tile,2/4/8/252/256,256/512/768/1024/1280/1536/1792/2048,K=i32 V=f16,0,9282560,0,2,4,0.5,4154.97,506.34,4661.32
```

### Long-context `p_min` calibration start

Same long prompt with `CTX_SIZE=4096`, `N_PREDICT=32`, `SPEC_DRAFT_N_MAX=1`.

```csv
case,p_min,failure_class,bad_h,bad_logits,accepted,generated,rate,eval_ms,total_ms
source-dot4,0,ok,0,0,15,15,1.0,1407.51,5425.64
source-dot4,0.4,ok,0,0,15,15,1.0,1371.92,5434.71
packed16-mmq,0,ok,0,0,5,25,0.2,1528.30,5555.97
packed16-mmq,0.2,ok,0,0,11,18,0.61111,1156.93,5157.22
packed16-mmq,0.4,ok,0,0,11,14,0.78571,1129.76,5144.38
```

Interpretation: packed16/MMQ has no correctness issue, but raw `p_min=0` over-generates low-confidence drafts. The fixed top-k confidence gate materially improves packed16/MMQ acceptance and wall time on this prompt. Source-DOT4 remains the acceptance baseline.

### Default-regression sweep

After experiment scripts were added, the production/default regression checks still passed:

```text
scripts/check-mtp-fa-mask-canary.sh: PASS bad_h=0 bad_logits=0 accepted=1 generated=1 d1=1/1
STRICT_ROUTES=1 benchmarks/mtp-dot4-isolation.sh: PASS all default/source-DOT4 lanes bad_h=0 bad_logits=0 d1=1/1
```

## Current conclusion

- Persistent packed16/MMQ for MTP verify is reachable and smoke-correct.
- Strict canary proves MTP-specific `rocm_packed16_dot4_mmq` selection with `K=i32 V=f16` at `nq=2/4`.
- Long-context shape coverage now reaches `nk=2048` with `bad_h=0` and `bad_logits=0` for source-DOT4, packed16/MMQ, BM32 direct-V, and BM64 512t.
- Multi-prompt acceptance smoke produced no hidden-state or teacher-logit corruption.
- Source-F16 DOT4 still has better long-context acceptance at `p_min=0` (`15/15`) than persistent packed16/MMQ (`5/25`).
- Packed16/MMQ improves substantially with confidence gating (`p_min=0.4`: `11/14`, lower eval/total time than `p_min=0`), so next work is calibration rather than corruption debugging.
- This remains an experiment; do not promote to default from this evidence alone.

## Failure classification

Use this order:

1. `assert_or_abort`: `GGML_ASSERT`, route-require abort, V-layout abort, or HIP launch abort.
2. `route_selection_failure`: strict route absent or target route selected but MTP route missing.
3. `bad_logits`: teacher probe finite/rank failure.
4. `bad_h`: MTP hidden-state canary failure.
5. `no_drafts_generated`: p-min or sampling generated no drafts.
6. `acceptance_regression`: generated drafts but acceptance materially lower than source-F16 DOT4 baseline.
7. `perf_regression`: correctness/acceptance OK but accepted-token throughput lower.
8. `ok`: all correctness and route checks pass.

## Promotion criteria

Packed16/MMQ can move from experiment to candidate only when all hold:

- strict MTP route checks pass across larger shape coverage;
- multi-prompt matrix has `bad_h=0` and `bad_logits=0`;
- acceptance is comparable to source-F16 DOT4 at practical `p_min` values;
- no silent fallback is accepted;
- accepted-token throughput improves in longer runs;
- target production route remains unchanged.

## Next runs

1. Continue `p_min` calibration with `0.6` and `0.8` on long-context prompts; current best packed16/MMQ sample is `p_min=0.4`.
2. Repeat the `p_min` sweep on more diverse prompts before drawing a policy conclusion.
3. Run `N_PREDICT=64` for source-DOT4 vs packed16/MMQ at the best packed16 `p_min` candidate.
4. Run source-F16 DOT4 acceptance matrix with the same prompt/p-min grid used for packed16/MMQ, then compare acceptance and teacher-rank summaries directly.
5. Add explicit negative tests for unsupported route requirements so failures classify as `request_error`, `route_selection_failure`, or `assert_or_abort` rather than silent fallback.
